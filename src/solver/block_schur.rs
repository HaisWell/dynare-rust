// SPDX-License-Identifier: GPL-3.0-or-later
//! Stable right deflating subspace of the first-order pencil by structured
//! reduction: a faster route to what `mjdgges` computes in
//! `dyn_first_order_solver.m`, used by [`super::FirstOrderSolver::solve`].
//!
//! Dynare's pencil `(E, D)` (dimension `nspred + nsfwrd`) is very
//! sparse and most of its roots are structural: exact zeros and infinities
//! from the aux lead/lag chains, the `both` identity rows and the exogenous
//! processes, many of them in long Jordan chains (Dynare's own output shows
//! them as rounding clusters of moduli ~0.004 / ~448). A single ordered QZ of
//! the whole pencil spends most of its time in `dtgsen`, whose swaps are slow
//! around those defective clusters. Here:
//!
//! 1. the pencil is permuted to block upper triangular form (Dulmage–Mendelsohn,
//!    as MATLAB's `dmperm`, which Dynare's `block` option is built on);
//! 2. inside each non-scalar diagonal block, zero and infinite roots are
//!    deflated exactly: a column of `D` (resp. `E`) that vanishes on the block
//!    is an infinite (resp. zero) root and one Householder reflection supported
//!    on the nonzeros of the other matrix's column splits it off at the
//!    top-left; a vanishing row splits it off at the bottom-right. Then 1.
//!    again on the remaining core, until nothing changes;
//! 3. each remaining core gets an *unordered* real QZ (`dgges` with
//!    `sort = 'N'`), applied to the whole pencil. On a semi-structural model
//!    with long aux chains most of the pencil ends up in scalar blocks and one
//!    core remains (its nonzero finite roots plus the zero/infinite roots in
//!    Jordan chains that are not exposed by exact zeros).
//!
//! The pencil is now in (unordered) generalized real Schur form, obtained with
//! permutations, Householder reflections, exact zero tests and QZ steps, i.e. it
//! is orthogonally equivalent to the one `mjdgges` factors. Instead of `dtgsen`
//! the stable subspace is then assembled bottom-up over the runs of
//! consecutive stable / unstable diagonal blocks. If the trailing part (runs
//! below `k`) has the stable basis `V` with `A_tr V = Y S_W`, `B_tr V = Y T_W`,
//! `(S_W, T_W)` upper (quasi-)triangular, then
//! * a stable run `k` adds the vectors `e_k`, and `(S_W, T_W)` grows to
//!   `[A_kk, A_k,tr V; 0, S_W]`, `[B_kk, B_k,tr V; 0, T_W]`;
//! * an unstable run `k` extends the old vectors to `[X; V]`, where `(X, L)`
//!   solves the generalized Sylvester equation (LAPACK `dtgsyl`)
//!
//! ```text
//!   A_kk X - L S_W = -A_k,tr V,    B_kk X - L T_W = -B_k,tr V,
//! ```
//!
//! which is exactly the computation `dtgsen` performs to swap the unstable
//! roots of run `k` below the stable roots of the trailing part, without the
//! sweeps of rotations over the full pencil ([`gen_sylvester`] is the
//! `dtgsy2` recursion with direct small solves).
//!
//! Whenever an assumption fails the caller falls back to the full `mjdgges`
//! path (see [`Decline`]).

use super::dmperm::dmperm;
use crate::linalg::qz::{eigenvalue, qz_unordered};
use crate::linalg::{Mat, gemm_raw};
use nalgebra::Complex;

/// Stable right deflating subspace of `(E, D)`: `E V = Y S`, `D V = Y T`.
#[derive(Debug, Clone)]
pub(crate) struct StableSubspace {
    /// nd x sdim basis (not orthonormal), rows in the pencil's column order.
    pub v: Mat,
    /// sdim x sdim upper quasi-triangular `S` and upper triangular `T`.
    pub s: Mat,
    pub t: Mat,
    /// All nd generalized eigenvalues, the `sdim` stable ones first (as `dr.eigval`).
    pub eigval: Vec<Complex<f64>>,
    pub sdim: usize,
}

/// Why the structured path declined (the caller then runs the full QZ).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decline {
    /// Empty pencil.
    NoStructure,
    /// Structurally singular pencil (no perfect matching, or a column/row
    /// vanishing in both matrices): let mjdgges report it.
    Singular,
    /// A 0/0 eigenvalue, a QZ failure, a root within rounding of `qz_criterium`,
    /// or an ill-conditioned Sylvester equation.
    Numerical,
}

/// Roots whose modulus is within this relative distance of `qz_criterium` are
/// left to the full QZ, so the stable/unstable split never depends on the
/// rounding differences between the two computations.
const CRIT_GUARD: f64 = 1e-9;

/// The pencil after the structural reduction: `(a, b) = Q' (E, D) Z` block
/// upper triangular with diagonal blocks `blocks` (start, size), in order.
struct Reduced {
    a: Mat,
    b: Mat,
    z: Mat,
    blocks: Vec<(usize, usize)>,
}

fn swap_rows(a: &mut Mat, b: &mut Mat, i: usize, j: usize) {
    if i != j {
        a.swap_rows(i, j);
        b.swap_rows(i, j);
    }
}

fn swap_cols(a: &mut Mat, b: &mut Mat, z: &mut Mat, i: usize, j: usize) {
    if i != j {
        a.swap_columns(i, j);
        b.swap_columns(i, j);
        z.swap_columns(i, j);
    }
}

/// Householder vector `v` (on `idx`, `idx[0]` the target) with `(I - beta v v') x = alpha e_target`.
fn householder(x: &[f64]) -> (Vec<f64>, f64, f64) {
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let alpha = if x[0] >= 0.0 { -norm } else { norm };
    let mut v = x.to_vec();
    v[0] -= alpha;
    let vtv: f64 = v.iter().map(|t| t * t).sum();
    (v, 2.0 / vtv, alpha)
}

/// Row reflection on rows `idx` of `m`, columns `c0..`.
fn reflect_rows(m: &mut Mat, idx: &[usize], v: &[f64], beta: f64, c0: usize) {
    let nr = m.nrows();
    for c in c0..m.ncols() {
        let col = &mut m.as_mut_slice()[c * nr..(c + 1) * nr];
        let s: f64 = idx.iter().zip(v).map(|(&i, &vi)| vi * col[i]).sum();
        if s != 0.0 {
            let s = beta * s;
            for (&i, &vi) in idx.iter().zip(v) {
                col[i] -= s * vi;
            }
        }
    }
}

/// Column reflection on columns `idx` of `m`, rows `0..r1`.
fn reflect_cols(m: &mut Mat, idx: &[usize], v: &[f64], beta: f64, r1: usize) {
    let nr = m.nrows();
    let data = m.as_mut_slice();
    let mut acc = vec![0.0; r1];
    for (&c, &vc) in idx.iter().zip(v) {
        for (r, a) in acc.iter_mut().enumerate() {
            *a += vc * data[r + c * nr];
        }
    }
    for (&c, &vc) in idx.iter().zip(v) {
        let f = beta * vc;
        for (r, a) in acc.iter().enumerate() {
            data[r + c * nr] -= f * a;
        }
    }
}

/// Exact deflation of zero / infinite roots of the diagonal block `[lo, hi)`.
/// Returns the remaining core `[lo', hi')`; the split-off 1 x 1 blocks are
/// pushed to `out`. `Err(Singular)` if a column or row vanishes in both matrices.
fn deflate(
    r: &mut Reduced,
    mut lo: usize,
    mut hi: usize,
    out: &mut Vec<(usize, usize)>,
) -> Result<(usize, usize), Decline> {
    let nd = r.a.nrows();
    'outer: while hi - lo >= 2 {
        // a column vanishing in D (infinite root) or E (zero root) on the block rows
        for j in lo..hi {
            let az = (lo..hi).all(|i| r.a[(i, j)] == 0.0);
            let bz = (lo..hi).all(|i| r.b[(i, j)] == 0.0);
            if az && bz {
                return Err(Decline::Singular);
            }
            if !(az || bz) {
                continue;
            }
            swap_cols(&mut r.a, &mut r.b, &mut r.z, j, lo);
            let x = if bz { &r.a } else { &r.b };
            let mut idx: Vec<usize> = (lo..hi).filter(|&i| x[(i, lo)] != 0.0).collect();
            if idx.len() == 1 {
                swap_rows(&mut r.a, &mut r.b, idx[0], lo);
            } else {
                if let Some(pos) = idx.iter().position(|&i| i == lo) {
                    idx.swap(0, pos);
                } else {
                    idx.insert(0, lo);
                }
                let xv: Vec<f64> = idx.iter().map(|&i| x[(i, lo)]).collect();
                let (v, beta, alpha) = householder(&xv);
                reflect_rows(&mut r.a, &idx, &v, beta, lo);
                reflect_rows(&mut r.b, &idx, &v, beta, lo);
                let x = if bz { &mut r.a } else { &mut r.b };
                for &i in &idx[1..] {
                    x[(i, lo)] = 0.0;
                }
                x[(lo, lo)] = alpha;
            }
            out.push((lo, 1));
            lo += 1;
            continue 'outer;
        }
        // a row vanishing in D or E on the block columns
        for i in lo..hi {
            let az = (lo..hi).all(|j| r.a[(i, j)] == 0.0);
            let bz = (lo..hi).all(|j| r.b[(i, j)] == 0.0);
            if az && bz {
                return Err(Decline::Singular);
            }
            if !(az || bz) {
                continue;
            }
            let last = hi - 1;
            swap_rows(&mut r.a, &mut r.b, i, last);
            let x = if bz { &r.a } else { &r.b };
            let mut idx: Vec<usize> = (lo..hi).filter(|&j| x[(last, j)] != 0.0).collect();
            if idx.len() == 1 {
                swap_cols(&mut r.a, &mut r.b, &mut r.z, idx[0], last);
            } else {
                if let Some(pos) = idx.iter().position(|&j| j == last) {
                    idx.swap(0, pos);
                } else {
                    idx.insert(0, last);
                }
                let xv: Vec<f64> = idx.iter().map(|&j| x[(last, j)]).collect();
                let (v, beta, alpha) = householder(&xv);
                reflect_cols(&mut r.a, &idx, &v, beta, hi);
                reflect_cols(&mut r.b, &idx, &v, beta, hi);
                reflect_cols(&mut r.z, &idx, &v, beta, nd);
                let x = if bz { &mut r.a } else { &mut r.b };
                for &j in &idx[1..] {
                    x[(last, j)] = 0.0;
                }
                x[(last, last)] = alpha;
            }
            out.push((last, 1));
            hi -= 1;
            continue 'outer;
        }
        break;
    }
    Ok((lo, hi))
}

/// Block triangularisation of the window `[lo, hi)` (dmperm on its pattern,
/// applied as row/column permutations); returns its diagonal blocks.
fn split(r: &mut Reduced, lo: usize, hi: usize) -> Result<Vec<(usize, usize)>, Decline> {
    let m = hi - lo;
    let pattern =
        Mat::from_fn(m, m, |i, j| if r.a[(lo + i, lo + j)] != 0.0 || r.b[(lo + i, lo + j)] != 0.0 { 1.0 } else { 0.0 });
    let dm = dmperm(&pattern);
    if dm.r != dm.s || *dm.r.last().unwrap() != m {
        return Err(Decline::Singular);
    }
    let nb = dm.r.len() - 1;
    if nb > 1 {
        let rows: Vec<usize> = dm.p.iter().map(|&i| lo + i).collect();
        let cols: Vec<usize> = dm.q.iter().map(|&j| lo + j).collect();
        let nd = r.a.nrows();
        let permute_rows = |x: &mut Mat| {
            for c in lo..nd {
                let tmp: Vec<f64> = rows.iter().map(|&i| x[(i, c)]).collect();
                for (k, t) in tmp.into_iter().enumerate() {
                    x[(lo + k, c)] = t;
                }
            }
        };
        permute_rows(&mut r.a);
        permute_rows(&mut r.b);
        let permute_cols = |x: &mut Mat, nrows: usize| {
            let tmp: Vec<Vec<f64>> =
                cols.iter().map(|&j| x.column(j).rows(0, nrows).iter().copied().collect()).collect();
            for (k, t) in tmp.into_iter().enumerate() {
                x.column_mut(lo + k).rows_mut(0, nrows).copy_from_slice(&t);
            }
        };
        permute_cols(&mut r.a, hi);
        permute_cols(&mut r.b, hi);
        permute_cols(&mut r.z, nd);
    }
    Ok((0..nb).map(|k| (lo + dm.r[k], dm.r[k + 1] - dm.r[k])).collect())
}

fn reduce(e: &Mat, d: &Mat) -> Result<Reduced, Decline> {
    let nd = e.nrows();
    let mut r = Reduced { a: e.clone(), b: d.clone(), z: Mat::identity(nd, nd), blocks: Vec::new() };
    let mut todo = split(&mut r, 0, nd)?;
    let mut done = Vec::with_capacity(nd);
    while let Some((lo, m)) = todo.pop() {
        if m == 1 {
            done.push((lo, 1));
            continue;
        }
        let (clo, chi) = deflate(&mut r, lo, lo + m, &mut done)?;
        match chi - clo {
            k if k == m => done.push((lo, m)), // irreducible core
            1 => done.push((clo, 1)),
            _ => todo.extend(split(&mut r, clo, chi)?),
        }
    }
    done.sort_unstable();
    r.blocks = done;
    Ok(r)
}

/// Relative size below which a pivot of the Sylvester sweep counts as a
/// (near) common root of the two pencils (then the full QZ takes over).
const SYLV_PIVOT_TOL: f64 = 1e-10;

/// Diagonal block partition (1 x 1 / 2 x 2) of an upper quasi-triangular matrix.
fn diag_blocks(n: usize, sub: impl Fn(usize) -> f64) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let q = if i + 1 < n && sub(i) != 0.0 { 2 } else { 1 };
        out.push((i, q));
        i += q;
    }
    out
}

/// Gaussian elimination with partial pivoting on a small dense system
/// (`n <= 8`, row-major `k`), in place on `x`. False on a relatively tiny pivot.
fn small_solve(n: usize, k: &mut [f64; 64], x: &mut [f64; 8]) -> bool {
    let scale = k[..n * n].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| k[i * n + col].abs().total_cmp(&k[j * n + col].abs())).unwrap();
        if !(k[piv * n + col].abs() > SYLV_PIVOT_TOL * scale) {
            return false;
        }
        if piv != col {
            for c in 0..n {
                k.swap(piv * n + c, col * n + c);
            }
            x.swap(piv, col);
        }
        let d = k[col * n + col];
        for r in col + 1..n {
            let f = k[r * n + col] / d;
            if f != 0.0 {
                for c in col..n {
                    k[r * n + c] -= f * k[col * n + c];
                }
                x[r] -= f * x[col];
            }
        }
    }
    for r in (0..n).rev() {
        let mut v = x[r];
        for c in r + 1..n {
            v -= k[r * n + c] * x[c];
        }
        x[r] = v / k[r * n + r];
    }
    true
}

/// Generalized Sylvester equation `A X - L S = C`, `B X - L T = F` (LAPACK
/// `dtgsyl` with `ijob = 0`, `trans = 'N'`, i.e. the `dtgsy2` sweep, without its
/// overflow scaling and condition estimation): `(A, B)` m x m and `(S, T)`
/// p x p in generalized real Schur form (column major, leading dimensions
/// `lda`, `lds`). On exit `c` holds `X` and `f` holds `L` (m x p, leading
/// dimension m). Blocks `(I, J)` of the 1 x 1 / 2 x 2 partitions are solved
/// with `J` ascending and `I` descending (systems of size `2 |I| |J| <= 8`),
/// then `C, F` of the rows above `I` get `- [A, B](above, I) X_IJ` and those of
/// the columns after `J` get `+ L_IJ [S, T](J, after)`.
/// Returns false on a pivot below `SYLV_PIVOT_TOL` (near common roots).
#[allow(clippy::too_many_arguments)]
fn gen_sylvester(
    a: &[f64],
    b: &[f64],
    lda: usize,
    m: usize,
    s: &[f64],
    t: &[f64],
    lds: usize,
    p: usize,
    c: &mut [f64],
    f: &mut [f64],
) -> bool {
    let rows = diag_blocks(m, |i| a[i + 1 + i * lda]);
    let cols = diag_blocks(p, |j| s[j + 1 + j * lds]);
    let mut k = [0.0f64; 64];
    let mut x = [0.0f64; 8];
    for &(j0, qj) in &cols {
        // pull the contributions of the finished columns of L:
        // C(:, J) += L(:, <J) S(<J, J), F(:, J) += L(:, <J) T(<J, J)
        {
            let (done, rest) = f.split_at_mut(j0 * m);
            for cl in 0..qj {
                let j = j0 + cl;
                let ccol = &mut c[j * m..(j + 1) * m];
                let fcol = &mut rest[cl * m..(cl + 1) * m];
                for jpp in 0..j0 {
                    let (sv, tv) = (s[jpp + j * lds], t[jpp + j * lds]);
                    if sv == 0.0 && tv == 0.0 {
                        continue;
                    }
                    let l = &done[jpp * m..(jpp + 1) * m];
                    for ((cv, fv), &lv) in ccol.iter_mut().zip(fcol.iter_mut()).zip(l) {
                        *cv += lv * sv;
                        *fv += lv * tv;
                    }
                }
            }
        }
        for &(i0, qi) in rows.iter().rev() {
            if qi == 1 && qj == 1 {
                let (a0, b0) = (a[i0 + i0 * lda], b[i0 + i0 * lda]);
                let (s0, t0) = (s[j0 + j0 * lds], t[j0 + j0 * lds]);
                let den = t0 * a0 - s0 * b0;
                if !(den.abs() > SYLV_PIVOT_TOL * (t0.abs() * a0.abs() + s0.abs() * b0.abs())) {
                    return false;
                }
                let (cc, ff) = (c[i0 + j0 * m], f[i0 + j0 * m]);
                let xv = (t0 * cc - s0 * ff) / den;
                let lv = if t0.abs() >= s0.abs() { (b0 * xv - ff) / t0 } else { (a0 * xv - cc) / s0 };
                c[i0 + j0 * m] = xv;
                f[i0 + j0 * m] = lv;
            } else {
                // unknowns X(r, c) at r + c qi, L(r, c) at qi qj + r + c qi
                let nq = qi * qj;
                let n = 2 * nq;
                k[..n * n].fill(0.0);
                for cl in 0..qj {
                    for r in 0..qi {
                        let e1 = r + cl * qi;
                        let e2 = nq + e1;
                        for kk in 0..qi {
                            k[e1 * n + kk + cl * qi] = a[i0 + r + (i0 + kk) * lda];
                            k[e2 * n + kk + cl * qi] = b[i0 + r + (i0 + kk) * lda];
                        }
                        for kk in 0..qj {
                            k[e1 * n + nq + r + kk * qi] = -s[j0 + kk + (j0 + cl) * lds];
                            k[e2 * n + nq + r + kk * qi] = -t[j0 + kk + (j0 + cl) * lds];
                        }
                        x[e1] = c[i0 + r + (j0 + cl) * m];
                        x[e2] = f[i0 + r + (j0 + cl) * m];
                    }
                }
                if !small_solve(n, &mut k, &mut x) {
                    return false;
                }
                for cl in 0..qj {
                    for r in 0..qi {
                        c[i0 + r + (j0 + cl) * m] = x[r + cl * qi];
                        f[i0 + r + (j0 + cl) * m] = x[nq + r + cl * qi];
                    }
                }
            }
            // rows above I: C(i', J) -= A(i', I) X_IJ, F(i', J) -= B(i', I) X_IJ
            for cl in j0..j0 + qj {
                for kk in i0..i0 + qi {
                    let xv = c[kk + cl * m];
                    if xv == 0.0 {
                        continue;
                    }
                    let (acol, bcol) = (&a[kk * lda..kk * lda + i0], &b[kk * lda..kk * lda + i0]);
                    let ccol = &mut c[cl * m..cl * m + i0];
                    let fcol = &mut f[cl * m..cl * m + i0];
                    for (((cv, fv), &av), &bv) in ccol.iter_mut().zip(fcol.iter_mut()).zip(acol).zip(bcol) {
                        *cv -= av * xv;
                        *fv -= bv * xv;
                    }
                }
            }
        }
    }
    true
}

/// A 1 x 1 or 2 x 2 diagonal block of the triangularised pencil with its
/// generalized eigenvalue(s) as dgges reports them.
#[derive(Debug, Clone, Copy)]
struct Diag {
    start: usize,
    size: usize,
    alphar: f64,
    alphai: f64,
    beta: f64,
}

/// QZ (unordered) of every core of `red` (applied to the whole pencil: rows of
/// the core from the left, its columns and `z` from the right), then the list
/// of 1 x 1 / 2 x 2 diagonal blocks of the resulting quasi-triangular pencil.
fn triangularise(red: &mut Reduced, zhreshold: f64) -> Result<Vec<Diag>, Decline> {
    let nd = red.a.nrows();
    let mut diag = Vec::with_capacity(nd);
    let blocks = std::mem::take(&mut red.blocks);
    for &(i0, m) in &blocks {
        if m == 1 {
            let (x, y) = (red.a[(i0, i0)], red.b[(i0, i0)]);
            if x.abs() <= zhreshold && y.abs() <= zhreshold {
                return Err(Decline::Numerical); // 0/0: mjdgges info -30
            }
            diag.push(Diag { start: i0, size: 1, alphar: x, alphai: 0.0, beta: y });
            continue;
        }
        let i1 = i0 + m;
        let ak = red.a.view((i0, i0), (m, m)).clone_owned();
        let bk = red.b.view((i0, i0), (m, m)).clone_owned();
        let qz = qz_unordered(&ak, &bk, zhreshold);
        if qz.info != 0 {
            return Err(Decline::Numerical);
        }
        // rows of the core: [S | Q' A12] ; columns: A01 Z ; z <- z Z
        if i1 < nd {
            for x in [&mut red.a, &mut red.b] {
                let right = x.view((i0, i1), (m, nd - i1)).clone_owned();
                gemm_raw(
                    true,
                    false,
                    m,
                    nd - i1,
                    m,
                    1.0,
                    qz.q.as_slice(),
                    m,
                    right.as_slice(),
                    m,
                    0.0,
                    &mut x.as_mut_slice()[i0 + i1 * nd..],
                    nd,
                );
            }
        }
        if i0 > 0 {
            for x in [&mut red.a, &mut red.b] {
                let above = x.view((0, i0), (i0, m)).clone_owned();
                gemm_raw(
                    false,
                    false,
                    i0,
                    m,
                    m,
                    1.0,
                    above.as_slice(),
                    i0,
                    qz.z.as_slice(),
                    m,
                    0.0,
                    &mut x.as_mut_slice()[i0 * nd..],
                    nd,
                );
            }
        }
        let zc = red.z.columns(i0, m).clone_owned();
        gemm_raw(
            false,
            false,
            nd,
            m,
            m,
            1.0,
            zc.as_slice(),
            nd,
            qz.z.as_slice(),
            m,
            0.0,
            &mut red.z.as_mut_slice()[i0 * nd..],
            nd,
        );
        red.a.view_mut((i0, i0), (m, m)).copy_from(&qz.s);
        red.b.view_mut((i0, i0), (m, m)).copy_from(&qz.t);
        let mut k = 0;
        while k < m {
            let size = if k + 1 < m && qz.s[(k + 1, k)] != 0.0 { 2 } else { 1 };
            diag.push(Diag { start: i0 + k, size, alphar: qz.alphar[k], alphai: qz.alphai[k], beta: qz.beta[k] });
            k += size;
        }
    }
    Ok(diag)
}

/// Stable deflating subspace of the pencil `E - lambda D` (`mjdgges(E, D, ...)`
/// semantics: roots with `|lambda| < qz_criterium` are stable).
pub(crate) fn stable_subspace(e: &Mat, d: &Mat, qz_criterium: f64, zhreshold: f64) -> Result<StableSubspace, Decline> {
    let nd = e.nrows();
    if nd == 0 {
        return Err(Decline::NoStructure);
    }
    let mut red = reduce(e, d)?;
    let diag = triangularise(&mut red, zhreshold)?;
    let (a, b) = (&red.a, &red.b);

    let crit2 = qz_criterium * qz_criterium;
    // V (rows: reduced coordinates), S_W and T_W live in the bottom-right corner
    // of nd x nd buffers: the stable vectors of block k are prepended on the left.
    let mut v = vec![0.0; nd * nd];
    let mut sw = vec![0.0; nd * nd];
    let mut tw = vec![0.0; nd * nd];
    let mut p = 0usize; // stable vectors found so far
    // Classify the diagonal blocks with mjdgges' selection criterion (the same
    // for both roots of a complex pair), then group consecutive blocks of the
    // same class: a run is itself a pencil in generalized Schur form.
    let mut stable_eig = Vec::with_capacity(nd);
    let mut unstable_eig = Vec::with_capacity(nd);
    let mut runs: Vec<(usize, usize, bool)> = Vec::new();
    for blk in &diag {
        let (ar, ai, be) = (blk.alphar, blk.alphai, blk.beta);
        let stable = ar * ar + ai * ai < crit2 * (be * be);
        let z = eigenvalue(ar, ai, be);
        if (z.norm() - qz_criterium).abs() <= CRIT_GUARD * qz_criterium {
            return Err(Decline::Numerical);
        }
        let out = if stable { &mut stable_eig } else { &mut unstable_eig };
        out.push(z);
        if blk.size == 2 {
            out.push(z.conj());
        }
        match runs.last_mut() {
            Some(r) if r.2 == stable => r.1 += blk.size,
            _ => runs.push((blk.start, blk.size, stable)),
        }
    }
    let mut ra = vec![0.0; nd * nd];
    let mut rb = vec![0.0; nd * nd];

    for &(i0, m, stable) in runs.iter().rev() {
        let i1 = i0 + m;
        let c0 = nd - p;
        if stable {
            // new stable vectors e_{i0..i1}; K = [A, B][run rows, below] V
            let c1 = c0 - m;
            if p > 0 && i1 < nd {
                gemm_raw(
                    false,
                    false,
                    m,
                    p,
                    nd - i1,
                    1.0,
                    &a.as_slice()[i0 + i1 * nd..],
                    nd,
                    &v[i1 + c0 * nd..],
                    nd,
                    0.0,
                    &mut sw[c1 + c0 * nd..],
                    nd,
                );
                gemm_raw(
                    false,
                    false,
                    m,
                    p,
                    nd - i1,
                    1.0,
                    &b.as_slice()[i0 + i1 * nd..],
                    nd,
                    &v[i1 + c0 * nd..],
                    nd,
                    0.0,
                    &mut tw[c1 + c0 * nd..],
                    nd,
                );
            }
            for j in 0..m {
                v[i0 + j + (c1 + j) * nd] = 1.0;
                for i in 0..m {
                    sw[c1 + i + (c1 + j) * nd] = a[(i0 + i, i0 + j)];
                    tw[c1 + i + (c1 + j) * nd] = b[(i0 + i, i0 + j)];
                }
            }
            p += m;
        } else if p > 0 && i1 < nd {
            // dtgsyl: A_kk X - L S_W = -A[k, below] V ; B_kk X - L T_W = -B[k, below] V
            gemm_raw(
                false,
                false,
                m,
                p,
                nd - i1,
                -1.0,
                &a.as_slice()[i0 + i1 * nd..],
                nd,
                &v[i1 + c0 * nd..],
                nd,
                0.0,
                &mut ra,
                m,
            );
            gemm_raw(
                false,
                false,
                m,
                p,
                nd - i1,
                -1.0,
                &b.as_slice()[i0 + i1 * nd..],
                nd,
                &v[i1 + c0 * nd..],
                nd,
                0.0,
                &mut rb,
                m,
            );
            let ok = gen_sylvester(
                &a.as_slice()[i0 + i0 * nd..],
                &b.as_slice()[i0 + i0 * nd..],
                nd,
                m,
                &sw[c0 + c0 * nd..],
                &tw[c0 + c0 * nd..],
                nd,
                p,
                &mut ra,
                &mut rb,
            );
            if !ok || !ra[..m * p].iter().all(|x| x.is_finite()) {
                return Err(Decline::Numerical);
            }
            // V[run rows, stable columns] = X
            for j in 0..p {
                v[i0 + (c0 + j) * nd..i1 + (c0 + j) * nd].copy_from_slice(&ra[j * m..(j + 1) * m]);
            }
        }
    }

    let c0 = nd - p;
    // back to the pencil's coordinates: V <- Z V
    let mut vo = Mat::zeros(nd, p);
    gemm_raw(false, false, nd, p, nd, 1.0, red.z.as_slice(), nd, &v[c0 * nd..], nd, 0.0, vo.as_mut_slice(), nd);
    let s = Mat::from_fn(p, p, |i, j| sw[c0 + i + (c0 + j) * nd]);
    let t = Mat::from_fn(p, p, |i, j| tw[c0 + i + (c0 + j) * nd]);
    let mut eigval = stable_eig;
    eigval.extend(unstable_eig);
    Ok(StableSubspace { v: vo, s, t, eigval, sdim: p })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{mul, qz::mjdgges};

    /// `D V M = E V` with `M = T^{-1} S`.
    fn check(e: &Mat, d: &Mat, ss: &StableSubspace) {
        let mut m = ss.s.clone();
        crate::linalg::trsm_left(true, false, 1.0, &ss.t, &mut m);
        let lhs = mul(&mul(d, false, &ss.v, false), false, &m, false);
        let rhs = mul(e, false, &ss.v, false);
        let res = (&lhs - &rhs).amax();
        assert!(res < 1e-10 * (1.0 + rhs.amax()), "residual {res}");
    }

    /// Same stable subspace and roots as the dense QZ, to `tol`. Dense QZ
    /// resolves defective zero / infinite roots only to ~eps^(1/k) (Jordan
    /// chains of length k) and that perturbs the other roots, so pencils with
    /// such chains need a looser `tol` than the residual check of `check`.
    fn compare_with_dense(e: &Mat, d: &Mat, tol: f64) {
        let ss = stable_subspace(e, d, 1.000001, 1e-6).unwrap();
        let qz = mjdgges(e, d, 1.000001, 1e-6);
        assert_eq!(ss.sdim, qz.sdim);
        assert_eq!(ss.eigval.len(), qz.eigval.len());
        // moduli of the well-separated finite nonzero roots
        let mid = |v: &[Complex<f64>]| {
            let mut a: Vec<f64> = v.iter().map(|z| z.norm()).filter(|&x| (1e-2..1e2).contains(&x)).collect();
            a.sort_by(f64::total_cmp);
            a
        };
        let (a, b) = (mid(&ss.eigval), mid(&qz.eigval));
        assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < tol * x.max(1.0), "{a:?} vs {b:?}");
        }
        check(e, d, &ss);
        let w1 = qz.z.columns(0, qz.sdim).clone_owned();
        let proj = &w1 * w1.transpose();
        let resid = &ss.v - &proj * &ss.v;
        assert!(resid.amax() < tol * (1.0 + ss.v.amax()), "{}", resid.amax());
    }

    #[test]
    fn matches_dense_qz_on_block_triangular_pencil() {
        // a 3x3 mixed block coupled to stable and unstable scalars, hidden by a permutation
        #[rustfmt::skip]
        let e0 = Mat::from_row_slice(6, 6, &[
            0.5, 0.0, 0.3, 1.0, 0.2, 0.7,
            0.0, 2.0, 0.0, 0.0, 0.4, 0.1,
            0.0, 0.0, 0.9, 0.3, 1.0, 0.0,
            0.0, 0.0, 0.2, 1.7, 0.0, 0.5,
            0.0, 0.0, 0.4, 0.1, 0.3, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.8,
        ]);
        #[rustfmt::skip]
        let d0 = Mat::from_row_slice(6, 6, &[
            1.0, 0.1, 0.0, 0.0, 0.0, 0.3,
            0.0, 0.0, 0.0, 0.2, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.2, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.1, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let perm = [3, 0, 5, 1, 4, 2];
        let e = Mat::from_fn(6, 6, |i, j| e0[(perm[i], perm[(j + 2) % 6])]);
        let d = Mat::from_fn(6, 6, |i, j| d0[(perm[i], perm[(j + 2) % 6])]);
        compare_with_dense(&e, &d, 1e-12);
    }

    /// A random block upper triangular pencil hidden by row/column permutations:
    /// dense diagonal blocks (complex pairs on both sides of the unit circle),
    /// exact zero / infinite scalar roots, blocks whose D (E) has a vanishing
    /// column or row (deflated inside the block), sparse coupling above.
    fn random_pencil(seed: u64) -> (Mat, Mat) {
        use rand::{Rng, SeedableRng, seq::SliceRandom};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut sizes = Vec::new();
        let mut n = 0;
        while n < 40 {
            let k = *[1, 1, 1, 2, 3, 4, 6].choose(&mut rng).unwrap();
            sizes.push(k);
            n += k;
        }
        let (mut e, mut d) = (Mat::zeros(n, n), Mat::zeros(n, n));
        let mut i0 = 0;
        for &k in &sizes {
            let scale: f64 = if rng.gen_bool(0.5) { 0.4 } else { 2.5 };
            for i in 0..k {
                for j in 0..k {
                    e[(i0 + i, i0 + j)] = scale * rng.gen_range(-1.0..1.0);
                    d[(i0 + i, i0 + j)] = rng.gen_range(-1.0..1.0) + if i == j { 1.5 } else { 0.0 };
                }
            }
            match (k, rng.gen_range(0..5)) {
                (1, 0) => d[(i0, i0)] = 0.0,                                           // infinite root
                (1, 1) => e[(i0, i0)] = 0.0,                                           // zero root
                (k, 2) if k > 1 => (0..k).for_each(|i| d[(i0 + i, i0)] = 0.0),         // D column vanishes
                (k, 3) if k > 1 => (0..k).for_each(|j| e[(i0 + k - 1, i0 + j)] = 0.0), // E row vanishes
                _ => {}
            }
            i0 += k;
        }
        for i in 0..n {
            for j in 0..n {
                let above = sizes
                    .iter()
                    .scan(0, |s, &k| {
                        *s += k;
                        Some(*s)
                    })
                    .any(|end| i < end && j >= end);
                if above && rng.gen_bool(0.15) {
                    e[(i, j)] = rng.gen_range(-1.0..1.0);
                    if rng.gen_bool(0.5) {
                        d[(i, j)] = rng.gen_range(-1.0..1.0);
                    }
                }
            }
        }
        let mut rows: Vec<usize> = (0..n).collect();
        let mut cols: Vec<usize> = (0..n).collect();
        rows.shuffle(&mut rng);
        cols.shuffle(&mut rng);
        (Mat::from_fn(n, n, |i, j| e[(rows[i], cols[j])]), Mat::from_fn(n, n, |i, j| d[(rows[i], cols[j])]))
    }

    #[test]
    fn matches_dense_qz_on_random_structured_pencils() {
        let mut complex_pairs = 0;
        for seed in 0..40 {
            let (e, d) = random_pencil(seed);
            compare_with_dense(&e, &d, 1e-6);
            complex_pairs +=
                stable_subspace(&e, &d, 1.000001, 1e-6).unwrap().eigval.iter().filter(|z| z.im > 0.0).count();
        }
        assert!(complex_pairs > 20, "the generator should produce complex pairs");
    }

    #[test]
    fn declines_zero_over_zero_and_singular_pencils() {
        // 0/0 root in a scalar block -> mjdgges info -30 territory
        let e = Mat::from_diagonal(&nalgebra::DVector::from_vec(vec![0.0, 0.5, 2.0]));
        let d = Mat::from_diagonal(&nalgebra::DVector::from_vec(vec![0.0, 1.0, 1.0]));
        assert_eq!(stable_subspace(&e, &d, 1.000001, 1e-6).unwrap_err(), Decline::Singular);
        let e = Mat::from_diagonal(&nalgebra::DVector::from_vec(vec![1e-8, 0.5, 2.0]));
        let d = Mat::from_diagonal(&nalgebra::DVector::from_vec(vec![1e-9, 1.0, 1.0]));
        assert_eq!(stable_subspace(&e, &d, 1.000001, 1e-6).unwrap_err(), Decline::Numerical);
        // a root on the stability boundary (within CRIT_GUARD)
        let e = Mat::from_diagonal(&nalgebra::DVector::from_vec(vec![1.000001, 0.5]));
        let d = Mat::identity(2, 2);
        assert_eq!(stable_subspace(&e, &d, 1.000001, 1e-6).unwrap_err(), Decline::Numerical);
    }
}
