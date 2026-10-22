// SPDX-License-Identifier: GPL-3.0-or-later
//! Initial state covariances: `compute_Pinf_Pstar.m` (diffuse initialisation, lik_init = 3)
//! and `lyapunov_symm.m` (stationary initialisation, lik_init = 1, Dynare's default
//! `lyapunov_solver` branch).

use crate::linalg::schur::{RealSchur, ordschur, schur};
use crate::linalg::{Mat, mul};

/// Solves the Stein (discrete Lyapunov) equation X = S X S' + B for a quasi-upper-triangular
/// S (n x n), column by column from the bottom right, exactly as the `while i >= 2` loops of
/// `lyapunov_symm.m` and `compute_Pinf_Pstar.m`.
///
/// `is_real_block(s)` decides from the sub-diagonal entry S(i,i-1) whether column i is a 1x1
/// step (`compute_Pinf_Pstar`: `S(i,i-1) == 0`; `lyapunov_symm`: `|S(i,i-1)| < threshold`).
/// Dynare solves each step's dense system `q \ rhs` with `q = I - S(1:i,1:i) S(i,i)` (or the
/// 2i x 2i block version); `q` inherits S's quasi-triangular structure, so here those systems
/// are solved by block back-substitution in O(i^2) instead of O(i^3) (same solution).
///
/// Runs [`stein_downdate`] (same solution, a third of the flops, contiguous column loops)
/// whenever the 1x1/2x2 decisions of `is_real_block` agree with S's exact sub-diagonal
/// zeros, which is always the case for LAPACK's standardised Schur form with
/// `compute_Pinf_Pstar`'s test; otherwise (`lyapunov_symm`'s threshold ignoring a tiny
/// non-zero S(i,i-1)) the literal port [`stein_dynare`], which keeps Dynare's treatment of
/// that entry.
pub(crate) fn stein_quasi_triangular(s: &Mat, b: &Mat, is_real_block: impl Fn(f64) -> bool) -> Mat {
    let n = s.nrows();
    let mut consistent = true;
    let mut i = n;
    while i >= 2 {
        let sub = s[(i - 1, i - 2)];
        if is_real_block(sub) {
            consistent &= sub == 0.0;
            i -= 1;
        } else {
            consistent &= sub != 0.0 && (i < 3 || s[(i - 2, i - 3)] == 0.0);
            i -= 2;
        }
    }
    if consistent { stein_downdate(s, b) } else { stein_dynare(s, b, is_real_block) }
}

/// X = S X S' + B for a quasi-upper-triangular S (exact zeros below its 1x1/2x2 diagonal
/// blocks) and symmetric B (its upper triangle is used), by peeling off the trailing
/// diagonal block: with S = [S11 S12; 0 S22], X = [X11 X12; X12' X22],
///
/// * X22 - S22 X22 S22' = B22 (1x1, or a 4x4 system for a 2x2 block),
/// * X12 - S11 X12 S22' = B12 + S12 X22 S22' (block back-substitution over S11),
/// * then X11 = S11 X11 S11' + B11 + G S12' + S12 G' + S12 X22 S12' with G = S11 X12,
///
/// mathematically the same X as Dynare's column recursion ([`stein_dynare`]), in about
/// 2n^3/3 instead of 2n^3 flops. The result is exactly symmetric.
pub(crate) fn stein_downdate(s: &Mat, b: &Mat) -> Mat {
    let n = s.nrows();
    let mut x = Mat::zeros(n, n);
    if n == 0 {
        return x;
    }
    let sd = s.as_slice();
    let at = |r: usize, c: usize| sd[c * n + r];
    let mut bw = b.as_slice().to_vec(); // upper triangle = current B11
    let xd = x.as_mut_slice();
    // scratch, n x 2 column-major
    let mut rr = vec![0.0; 2 * n]; // R, then X12
    let mut g = vec![0.0; 2 * n]; // G, then H
    let mut hi_end = n;
    while hi_end > 0 {
        let hi = hi_end - 1;
        let lo = if hi >= 1 && at(hi, hi - 1) != 0.0 { hi - 1 } else { hi };
        let p = hi - lo + 1;
        // ---- X22 ------------------------------------------------------------------------
        let mut x22 = [0.0f64; 4]; // column-major p x p
        let mut s22 = [0.0f64; 4];
        for c in 0..p {
            for r in 0..p {
                s22[c * p + r] = at(lo + r, lo + c);
            }
        }
        if p == 1 {
            x22[0] = bw[lo * n + lo] / (1.0 - s22[0] * s22[0]);
        } else {
            // vec(X) - (S22 kron S22) vec(X) = vec(B22), vec index c*2 + r
            let mut a = [0.0f64; 16];
            for c in 0..2 {
                for r in 0..2 {
                    let row = c * 2 + r;
                    for c2 in 0..2 {
                        for r2 in 0..2 {
                            let col = c2 * 2 + r2;
                            // (S X S')(r, c) = sum S(r, r2) X(r2, c2) S(c, c2)
                            let v = s22[r2 * 2 + r] * s22[c2 * 2 + c];
                            a[col * 4 + row] = if row == col { 1.0 } else { 0.0 } - v;
                        }
                    }
                }
            }
            let b01 = bw[hi * n + lo]; // B(lo, hi), upper triangle
            let mut rhs = [bw[lo * n + lo], b01, b01, bw[hi * n + hi]];
            solve_small(&mut a, 4, &mut rhs);
            // as Dynare: x(i, i-1) = x(i-1, i)
            x22 = [rhs[0], rhs[2], rhs[2], rhs[3]];
        }
        for c in 0..p {
            for r in 0..p {
                xd[(lo + c) * n + lo + r] = x22[c * p + r];
            }
        }
        if lo == 0 {
            break;
        }
        // ---- X12: rows 0..lo ----------------------------------------------------------------
        // Y = X22 S22' (p x p), R = B12 + S12 Y
        let mut y = [0.0f64; 4];
        for c in 0..p {
            for r in 0..p {
                let mut acc = 0.0;
                for k in 0..p {
                    acc += x22[k * p + r] * s22[k * p + c];
                }
                y[c * p + r] = acc;
            }
        }
        for c in 0..p {
            let rc = &mut rr[c * n..c * n + lo];
            rc.copy_from_slice(&bw[(lo + c) * n..(lo + c) * n + lo]);
            for k in 0..p {
                let f = y[c * p + k];
                let sk = &sd[(lo + k) * n..(lo + k) * n + lo];
                for (v, sv) in rc.iter_mut().zip(sk) {
                    *v += sv * f;
                }
            }
        }
        // Block back-substitution for X12 - S11 X12 M = R with M = S22' (M(c2, c) = S22(c, c2)),
        // accumulating G = S11 X12 on the way: when block K is solved, g(K, :) holds
        // sum_{j > K} S(K, j) X12(j, :), so R_K + g_K M is its right-hand side and
        // G_K = g_K + S_KK X_K.
        for c in 0..p {
            g[c * n..c * n + lo].fill(0.0);
        }
        let mut kh_end = lo;
        while kh_end > 0 {
            let kh = kh_end - 1;
            let kl = if kh >= 1 && at(kh, kh - 1) != 0.0 { kh - 1 } else { kh };
            let kb = kh - kl + 1;
            let nn = kb * p;
            // (I - M' kron S_KK) vec(X_K) = vec(R_K + g_K M), vec index c*kb + q
            let mut rk = [0.0f64; 4];
            for c in 0..p {
                for q in 0..kb {
                    let mut v = rr[c * n + kl + q];
                    for c2 in 0..p {
                        v += g[c2 * n + kl + q] * s22[c2 * p + c];
                    }
                    rk[c * kb + q] = v;
                }
            }
            if nn == 1 {
                rk[0] /= 1.0 - at(kl, kl) * s22[0];
            } else {
                let mut a = [0.0f64; 16];
                for c in 0..p {
                    for q in 0..kb {
                        let row = c * kb + q;
                        for c2 in 0..p {
                            for q2 in 0..kb {
                                let col = c2 * kb + q2;
                                let v = at(kl + q, kl + q2) * s22[c2 * p + c];
                                a[col * nn + row] = if row == col { 1.0 } else { 0.0 } - v;
                            }
                        }
                    }
                }
                solve_small(&mut a, nn, &mut rk);
            }
            for c in 0..p {
                let mut gk = [0.0f64; 2];
                for q in 0..kb {
                    let mut v = g[c * n + kl + q];
                    for q2 in 0..kb {
                        v += at(kl + q, kl + q2) * rk[c * kb + q2];
                    }
                    gk[q] = v;
                }
                for q in 0..kb {
                    rr[c * n + kl + q] = rk[c * kb + q];
                    g[c * n + kl + q] = gk[q];
                }
            }
            // g(0..kl, c) += S(0..kl, K) X_K(:, c)
            if kl > 0 {
                for c in 0..p {
                    for q in 0..kb {
                        let f = rk[c * kb + q];
                        if f != 0.0 {
                            let sk = &sd[(kl + q) * n..(kl + q) * n + kl];
                            for (v, sv) in g[c * n..c * n + kl].iter_mut().zip(sk) {
                                *v += sv * f;
                            }
                        }
                    }
                }
            }
            kh_end = kl;
        }
        // store X12 and X12'
        for c in 0..p {
            xd[(lo + c) * n..(lo + c) * n + lo].copy_from_slice(&rr[c * n..c * n + lo]);
            for r in 0..lo {
                xd[r * n + lo + c] = rr[c * n + r];
            }
        }
        // ---- B11 += H S12' + S12 H', H = G + S12 X22 / 2 ------------------------------------
        for c in 0..p {
            let gc = &mut g[c * n..c * n + lo];
            for k in 0..p {
                let f = 0.5 * x22[c * p + k];
                for (v, sv) in gc.iter_mut().zip(&sd[(lo + k) * n..(lo + k) * n + lo]) {
                    *v += sv * f;
                }
            }
        }
        for col in 0..lo {
            let bc = &mut bw[col * n..col * n + col + 1];
            for k in 0..p {
                let s12 = &sd[(lo + k) * n..(lo + k) * n + col + 1];
                let hk = &g[k * n..k * n + col + 1];
                let (sc, hc) = (s12[col], hk[col]);
                for ((v, h), sv) in bc.iter_mut().zip(hk).zip(s12) {
                    *v += h * sc + sv * hc;
                }
            }
        }
        hi_end = lo;
    }
    x
}

/// Literal port of the `while i >= 2` column recursion of `lyapunov_symm.m` /
/// `compute_Pinf_Pstar.m` (see [`stein_quasi_triangular`]).
pub(crate) fn stein_dynare(s: &Mat, b: &Mat, is_real_block: impl Fn(f64) -> bool) -> Mat {
    let n = s.nrows();
    let mut x = Mat::zeros(n, n);
    if n == 0 {
        return x;
    }
    let mut y = vec![0.0; n]; // x(:, i+1:end) * S(i, i+1:end)'
    let mut y1 = vec![0.0; n];
    let mut rhs = vec![0.0; 2 * n];
    let mut sol = vec![0.0; 2 * n];
    let mut i = n; // 1-based column index as in the MATLAB code
    while i >= 2 {
        let ic = i - 1; // 0-based column i
        if is_real_block(s[(ic, ic - 1)]) {
            // c = S(1:i,:)*(x(:,i+1:end)*S(i,i+1:end)') + S(i,i)*S(1:i,i+1:end)*x(i+1:end,i)
            tail_terms(s, &x, ic, i, &mut y, &mut rhs[..i]);
            for r in 0..i {
                rhs[r] += b[(r, ic)];
            }
            let m = [s[(ic, ic)], 0.0, 0.0, 0.0];
            solve_shifted(s, i, 1, &m, &rhs[..i], &mut sol[..i]);
            for r in 0..i {
                x[(r, ic)] = sol[r];
            }
            for r in 0..ic {
                x[(ic, r)] = x[(r, ic)];
            }
            i -= 1;
        } else {
            let jc = ic - 1; // 0-based column i-1
            // c (column i) and c1 (column i-1)
            let (a, bb, c, d) = (s[(ic, ic)], s[(ic, jc)], s[(jc, ic)], s[(jc, jc)]);
            {
                let (r1, r2) = rhs.split_at_mut(i);
                tail_terms2(s, &x, ic, jc, i, &mut y, &mut y1, r1, &mut r2[..i]);
            }
            for r in 0..i {
                rhs[r] += b[(r, ic)];
                rhs[i + r] += b[(r, jc)];
            }
            // [u w] - S [u w] M = [r1 r2], M = [a c; b d]
            let m = [a, bb, c, d]; // column-major 2x2
            solve_shifted(s, i, 2, &m, &rhs[..2 * i], &mut sol[..2 * i]);
            for r in 0..i {
                x[(r, ic)] = sol[r];
                x[(r, jc)] = sol[i + r];
            }
            for r in 0..ic {
                x[(ic, r)] = x[(r, ic)];
            }
            for r in 0..jc {
                x[(jc, r)] = x[(r, jc)];
            }
            i -= 2;
        }
    }
    if i == 1 {
        // c = S(1,:)*(x(:,2:end)*S(1,2:end)') + S(1,1)*S(1,2:end)*x(2:end,1)
        let mut c = 0.0;
        for col in 0..n {
            let mut acc = 0.0;
            for j in 1..n {
                acc += x[(col, j)] * s[(0, j)];
            }
            c += s[(0, col)] * acc;
        }
        let mut acc = 0.0;
        for j in 1..n {
            acc += s[(0, j)] * x[(j, 0)];
        }
        c += s[(0, 0)] * acc;
        x[(0, 0)] = (b[(0, 0)] + c) / (1.0 - s[(0, 0)] * s[(0, 0)]);
    }
    x
}

/// out(0..i) = S(1:i,:)*(x(:,i+1:end)*S(row,i+1:end)') + S(row,row)*S(1:i,i+1:end)*x(i+1:end,row)
/// with `row` = ic (0-based) and i = ic+1 = number of rows.
fn tail_terms(s: &Mat, x: &Mat, ic: usize, i: usize, y: &mut [f64], out: &mut [f64]) {
    let n = s.nrows();
    out.iter_mut().for_each(|v| *v = 0.0);
    if i == n {
        return;
    }
    for (col, yc) in y.iter_mut().enumerate() {
        let mut acc = 0.0;
        for j in i..n {
            acc += x[(col, j)] * s[(ic, j)];
        }
        *yc = acc;
    }
    let sii = s[(ic, ic)];
    for (r, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for col in 0..n {
            acc += s[(r, col)] * y[col];
        }
        let mut acc2 = 0.0;
        for j in i..n {
            acc2 += s[(r, j)] * x[(j, ic)];
        }
        *o = acc + sii * acc2;
    }
}

/// The c / c1 terms of the 2x2-block step (rows ic = i-1 and jc = i-2, 0-based).
#[allow(clippy::too_many_arguments)]
fn tail_terms2(
    s: &Mat,
    x: &Mat,
    ic: usize,
    jc: usize,
    i: usize,
    y: &mut [f64],
    y1: &mut [f64],
    c: &mut [f64],
    c1: &mut [f64],
) {
    let n = s.nrows();
    c.iter_mut().for_each(|v| *v = 0.0);
    c1.iter_mut().for_each(|v| *v = 0.0);
    if i == n {
        return;
    }
    for col in 0..n {
        let (mut a0, mut a1) = (0.0, 0.0);
        for j in i..n {
            a0 += x[(col, j)] * s[(ic, j)];
            a1 += x[(col, j)] * s[(jc, j)];
        }
        y[col] = a0;
        y1[col] = a1;
    }
    let (sii, sij, sji, sjj) = (s[(ic, ic)], s[(ic, jc)], s[(jc, ic)], s[(jc, jc)]);
    for r in 0..i {
        let (mut p0, mut p1) = (0.0, 0.0);
        for col in 0..n {
            p0 += s[(r, col)] * y[col];
            p1 += s[(r, col)] * y1[col];
        }
        let (mut xi, mut xj) = (0.0, 0.0);
        for j in i..n {
            xi += s[(r, j)] * x[(j, ic)];
            xj += s[(r, j)] * x[(j, jc)];
        }
        // c  = S*(..S(i,:)') + S(i,i)*S(:,i+1:)*x(i+1:,i) + S(i,i-1)*S(:,i+1:)*x(i+1:,i-1)
        // c1 = S*(..S(i-1,:)') + S(i-1,i-1)*S(:,i+1:)*x(i+1:,i-1) + S(i-1,i)*S(:,i+1:)*x(i+1:,i)
        c[r] = p0 + sii * xi + sij * xj;
        c1[r] = p1 + sjj * xj + sji * xi;
    }
}

/// Solves X - S(1:i,1:i) X M = R for X (i x m, m = 1 or 2, column-major in `rhs`/`out`),
/// M m x m column-major in `m4`, by block back-substitution over S's 1x1/2x2 diagonal blocks.
fn solve_shifted(s: &Mat, i: usize, m: usize, m4: &[f64; 4], rhs: &[f64], out: &mut [f64]) {
    // ym = X*M rows already solved
    let mut ym = vec![0.0; i * m];
    let mut r = i;
    while r > 0 {
        let hi = r - 1;
        let lo = if hi >= 1 && s[(hi, hi - 1)] != 0.0 { hi - 1 } else { hi };
        let kb = hi - lo + 1;
        // rhs_K = R_K + sum_{j > hi} S(K, j) * ym(j, :)
        let mut rk = [0.0; 4]; // column-major kb x m
        for c in 0..m {
            for (q, row) in (lo..=hi).enumerate() {
                let mut acc = rhs[c * i + row];
                for j in (hi + 1)..i {
                    acc += s[(row, j)] * ym[c * i + j];
                }
                rk[c * kb + q] = acc;
            }
        }
        // (I - M' (x) S_KK) vec(X_K) = vec(rhs_K); vec index c*kb + q
        let nn = kb * m;
        let mut a = [0.0; 16];
        for c in 0..m {
            for q in 0..kb {
                let row = c * kb + q;
                for c2 in 0..m {
                    for q2 in 0..kb {
                        let col = c2 * kb + q2;
                        // (S X M)(q, c) = sum_{q2,c2} S(q,q2) X(q2,c2) M(c2,c)
                        let v = s[(lo + q, lo + q2)] * m4[c * m + c2];
                        a[col * nn + row] = if row == col { 1.0 } else { 0.0 } - v;
                    }
                }
            }
        }
        solve_small(&mut a, nn, &mut rk);
        for c in 0..m {
            for q in 0..kb {
                out[c * i + lo + q] = rk[c * kb + q];
            }
        }
        for q in 0..kb {
            for c in 0..m {
                let mut acc = 0.0;
                for c2 in 0..m {
                    acc += rk[c2 * kb + q] * m4[c * m + c2];
                }
                ym[c * i + lo + q] = acc;
            }
        }
        r = lo;
    }
}

/// Gaussian elimination with partial pivoting for n <= 4 (column-major `a`), in place on b.
fn solve_small(a: &mut [f64; 16], n: usize, b: &mut [f64; 4]) {
    for k in 0..n {
        let mut p = k;
        for r in (k + 1)..n {
            if a[k * n + r].abs() > a[k * n + p].abs() {
                p = r;
            }
        }
        if p != k {
            for c in 0..n {
                a.swap(c * n + k, c * n + p);
            }
            b.swap(k, p);
        }
        let piv = a[k * n + k];
        for r in (k + 1)..n {
            let f = a[k * n + r] / piv;
            if f != 0.0 {
                for c in (k + 1)..n {
                    a[c * n + r] -= f * a[c * n + k];
                }
                b[r] -= f * b[k];
            }
        }
    }
    for k in (0..n).rev() {
        let mut acc = b[k];
        for c in (k + 1)..n {
            acc -= a[c * n + k] * b[c];
        }
        b[k] = acc / a[k * n + k];
    }
}

/// Port of matrix_solver/lyapunov_symm.m with `method = 0` (the branch used by
/// `lyapunov_solver.m` under Dynare's default options): solves X = A X A' + B on the stable
/// subspace (unit roots, |eig| > 2 - qz_criterium, are ordered first by `ordschur` and
/// dropped). `lyapunov_complex_threshold` (default 1e-15) separates 1x1 from 2x2 blocks.
pub fn lyapunov_symm(a: &Mat, b: &Mat, qz_criterium: f64, lyapunov_complex_threshold: f64) -> anyhow::Result<Mat> {
    let n = a.nrows();
    if n == 1 {
        return Ok(b / (1.0 - a[(0, 0)] * a[(0, 0)]));
    }
    let mut sch = schur(a)?;
    let e1: Vec<bool> = sch.eig_abs().iter().map(|&e| e > 2.0 - qz_criterium).collect();
    let k = e1.iter().filter(|&&s| s).count();
    if k > 0 {
        ordschur(&mut sch, &e1)?;
    }
    let ns = n - k;
    let t = sch.t.view((k, k), (ns, ns)).into_owned();
    let u2 = sch.u.columns(k, ns).into_owned();
    let bb = mul(&mul(&u2, true, b, false), false, &u2, false);
    let x = stein_quasi_triangular(&t, &bb, |s| s.abs() < lyapunov_complex_threshold);
    Ok(mul(&mul(&u2, false, &x, false), false, &u2, true))
}

/// `lyapunov_solver.m` with default options: P = lyapunov_symm(T, R Q R', ...).
pub fn lyapunov_solver(
    t: &Mat,
    r: &Mat,
    q: &Mat,
    qz_criterium: f64,
    lyapunov_complex_threshold: f64,
) -> anyhow::Result<Mat> {
    let rqr = mul(&mul(r, false, q, false), false, r, true);
    lyapunov_symm(t, &rqr, qz_criterium, lyapunov_complex_threshold)
}

/// Port of estimation/compute_Pinf_Pstar.m: Kitagawa transformation of the transition matrix
/// (real Schur form with the unit roots, |eig| > 2 - qz_criterium, ordered first), Pstar for
/// the stationary block (same Stein recursion as `lyapunov_symm`, 1x1 test `ST(i,i-1) == 0`),
/// projection onto the static (zero-column) variables, and Pinf with unit roots that do not
/// influence the observables (`norm(...) < 1e-8`) dropped.
///
/// `mf`: 0-based observed rows of T. `restrict_columns`: 0-based state columns
/// (`dr.restrict_columns`, the 6-argument call of `dsge_likelihood`); pass an empty slice for
/// the 5-argument call (`DsgeSmoother`), where the dynamic columns are those with
/// `max(abs(T)) >= 1e-10`. Panics only on LAPACK failure of the Schur steps (never observed).
///
/// Known, unavoidable difference to MATLAB: when T has a (numerically) multiple unit root
/// whose eigenspace contains directions that do not move the observables, the Schur vectors
/// within that eigenspace are determined by rounding, and Dynare's `norm(...) < 1e-8` test on
/// them can go either way depending on the LAPACK build (MATLAB ships NAG's LAPACK 3.9.1, we
/// use Accelerate's). See tests/dsge_sw07.rs `sw07_point15_unit_root_basis` and
/// [`compute_pinf_pstar_with_schur`], which takes the Schur form from outside.
pub fn compute_pinf_pstar(
    mf: &[usize],
    t: &Mat,
    r: &Mat,
    q: &Mat,
    qz_criterium: f64,
    restrict_columns: &[usize],
) -> (Mat, Mat) {
    try_compute_pinf_pstar(mf, t, r, q, qz_criterium, restrict_columns)
        .expect("compute_pinf_pstar: Schur decomposition failed")
}

/// Fallible version of [`compute_pinf_pstar`].
pub fn try_compute_pinf_pstar(
    mf: &[usize],
    t: &Mat,
    r: &Mat,
    q: &Mat,
    qz_criterium: f64,
    restrict_columns: &[usize],
) -> anyhow::Result<(Mat, Mat)> {
    compute_pinf_pstar_with_schur(mf, t, r, q, qz_criterium, restrict_columns, None)
}

/// [`try_compute_pinf_pstar`] with the (unordered) real Schur form of the dynamic block
/// `T(indx, indx)` supplied by the caller instead of computed by [`schur`] (e.g. MATLAB's
/// `schur` output, to separate LAPACK differences from the rest of the algorithm).
pub fn compute_pinf_pstar_with_schur(
    mf: &[usize],
    t: &Mat,
    r: &Mat,
    q: &Mat,
    qz_criterium: f64,
    restrict_columns: &[usize],
    schur_of_dynamic_block: Option<RealSchur>,
) -> anyhow::Result<(Mat, Mat)> {
    let np_all = t.nrows();
    if q.iter().all(|&v| v == 0.0) {
        return Ok((Mat::zeros(np_all, np_all), Mat::zeros(np_all, np_all)));
    }
    let (indx, indx0): (Vec<usize>, Vec<usize>) = if !restrict_columns.is_empty() {
        let indx = restrict_columns.to_vec();
        let indx0 = (0..np_all).filter(|j| !indx.contains(j)).collect();
        (indx, indx0)
    } else {
        let colmax = |j: usize| t.column(j).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        ((0..np_all).filter(|&j| colmax(j) >= 1e-10).collect(), (0..np_all).filter(|&j| colmax(j) < 1e-10).collect())
    };
    let np0 = indx0.len();
    let np = indx.len();
    let t0 = t.select_rows(&indx0).select_columns(&indx);
    let r0 = r.select_rows(&indx0);
    let tr = t.select_rows(&indx).select_columns(&indx);
    let rr = r.select_rows(&indx);

    let mut sch = match schur_of_dynamic_block {
        Some(s) => s,
        None => schur(&tr)?,
    };
    let e1: Vec<bool> = sch.eig_abs().iter().map(|&e| e > 2.0 - qz_criterium).collect();
    ordschur(&mut sch, &e1)?;
    let nk = sch.eig_abs().iter().filter(|&&e| e > 2.0 - qz_criterium).count();
    let (qt, st) = (sch.u, sch.t);

    let ns = np - nk;
    // R1 = QT'*R; only the stationary block B(nk+1:end, nk+1:end) of B = R1*Q*R1' is used
    let r1 = mul(&qt, true, &rr, false);
    let r1s = r1.rows(nk, ns).into_owned();
    let bs = mul(&mul(&r1s, false, q, false), false, &r1s, true);
    // Pstar = blkdiag(zeros(nk), X) in the Schur basis
    let s_st = st.view((nk, nk), (ns, ns)).into_owned();
    let x = stein_quasi_triangular(&s_st, &bs, |s| s == 0.0);

    // Pinf = blkdiag(I(nk), 0) with the unit roots that do not move the observables dropped
    let mut keep = vec![true; nk];
    let qts = qt.columns(nk, ns).into_owned(); // QT(:, nk+1:end)
    let n = np0 + np;
    // The final QT*Pstar*QT' and QT*Pinf*QT' only involve the non-zero blocks: with A the
    // n x ns matrix QT*ST0(:, J) (J = the stationary columns), C = QT*R10 and w_k the
    // columns QT*STinf1(:, np0+k),
    //   Pstar = A X A' + C Q C',   Pinf = sum_{k kept} w_k w_k'
    // (np0 = 0: A = QT(:, J), C = 0, w_k = QT(:, k)). Same sums as Dynare's dense products
    // up to the order of the non-zero terms.
    let (a, c, w) = if np0 > 0 {
        // ST = [zeros(n, np0) [T0*QT; ST]], R1 = [R0; R1]
        let t0qt = mul(&t0, false, &qt, false); // ST(1:np0, np0+1:end)
        let stinf = st.view((0, 0), (nk, nk)).into_owned();
        let istinf = if nk > 0 {
            stinf.clone().try_inverse().ok_or_else(|| anyhow::anyhow!("compute_pinf_pstar: singular STinf"))?
        } else {
            Mat::zeros(0, 0)
        };
        let st00 = t0qt.columns(0, nk).into_owned();
        // ST0(1:np0, J) = ST(1:np0, J) - ST00*(iSTinf*ST(np0+1:np0+nk, J))
        let st_tail = st.view((0, nk), (nk, ns)).into_owned();
        let blk = t0qt.columns(nk, ns).into_owned() - mul(&st00, false, &mul(&istinf, false, &st_tail, false), false);
        // R10(1:np0, :) = R0 - ST00*(iSTinf*R1(np0+1:np0+nk, :)); R10(np0+1:np0+nk, :) = 0
        let r1k = r1.rows(0, nk).into_owned();
        let r10top = &r0 - mul(&st00, false, &mul(&istinf, false, &r1k, false), false);
        // STinf0 = ST00*(eye(nk) - iSTinf*(STinf - eye(nk)))
        let sttriu = &stinf - Mat::identity(nk, nk);
        let stinf0 = mul(&st00, false, &(Mat::identity(nk, nk) - mul(&istinf, false, &sttriu, false)), false);

        // norm(QTinf(mf,:)*ST([indx0; indx], k+np0)) < 1e-8, with QTinf = blkdiag(eye(np0), QT),
        // QTinf(1:np0, np0+1:np0+nk) = STinf0, QTinf([indx0; indx], :) = QTinf, and
        // mf = ismember([indx0; indx], mf) a mask over row positions, as Dynare does.
        let perm: Vec<usize> = indx0.iter().chain(indx.iter()).copied().collect();
        let mut iperm = vec![0usize; n];
        for (pos, &dst) in perm.iter().enumerate() {
            iperm[dst] = pos;
        }
        let stf = |row: usize, j: usize| if row < np0 { t0qt[(row, j)] } else { st[(row - np0, j)] }; // ST(row, np0+j)
        for k in 0..nk {
            let col: Vec<f64> = perm.iter().map(|&row| stf(row, k)).collect();
            let mut nrm = 0.0;
            for row in (0..n).filter(|&pos| mf.contains(&perm[pos])) {
                let src = iperm[row]; // row of the unpermuted QTinf
                let v: f64 = if src < np0 {
                    col[src] + (0..nk).map(|j| stinf0[(src, j)] * col[np0 + j]).sum::<f64>()
                } else {
                    (0..np).map(|j| qt[(src - np0, j)] * col[np0 + j]).sum()
                };
                nrm += v * v;
            }
            if nrm.sqrt() < 1e-8 {
                keep[k] = false;
            }
        }
        // QT = blkdiag(eye(np0), QT) with rows QT([indx0; indx], :) = QT
        let qs_s = mul(&qts, false, &s_st, false); // QT(:, J) * ST(J, J)
        let qr = mul(&qts, false, &r1s, false); // QT(:, J) * R1(J, :)
        let mut a = Mat::zeros(n, ns);
        let mut c = Mat::zeros(n, q.ncols());
        let mut w = Mat::zeros(n, nk);
        for (p, &dst) in indx0.iter().enumerate() {
            a.row_mut(dst).copy_from(&blk.row(p));
            c.row_mut(dst).copy_from(&r10top.row(p));
            w.row_mut(dst).copy_from(&stinf0.row(p));
        }
        for (i, &dst) in indx.iter().enumerate() {
            a.row_mut(dst).copy_from(&qs_s.row(i));
            c.row_mut(dst).copy_from(&qr.row(i));
            w.row_mut(dst).copy_from(&qt.view((i, 0), (1, nk)));
        }
        (a, Some(c), w)
    } else {
        for k in 0..nk {
            let mut nrm = 0.0;
            for &row in mf {
                let v: f64 = (0..np).map(|c| qt[(row, c)] * st[(c, k)]).sum();
                nrm += v * v;
            }
            if nrm.sqrt() < 1e-8 {
                keep[k] = false;
            }
        }
        // As Dynare: no row permutation back when there are no static columns.
        (qts, None, qt.columns(0, nk).into_owned())
    };
    let mut pstar = mul(&mul(&a, false, &x, false), false, &a, true);
    if let Some(c) = c {
        crate::linalg::gemm(1.0, &mul(&c, false, q, false), false, &c, true, 1.0, &mut pstar);
    }
    let mut pinf = Mat::zeros(n, n);
    for k in (0..nk).filter(|&k| keep[k]) {
        let wk = w.column(k);
        let wk = wk.as_slice();
        for (j, pc) in pinf.as_mut_slice().chunks_exact_mut(n).enumerate() {
            let f = wk[j];
            if f != 0.0 {
                for (v, wi) in pc.iter_mut().zip(wk) {
                    *v += wi * f;
                }
            }
        }
    }
    Ok((pstar, pinf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_stein(t: &Mat, b: &Mat) -> Mat {
        let mut x = b.clone();
        for _ in 0..5000 {
            x = t * &x * t.transpose() + b;
        }
        x
    }

    /// Quasi-upper-triangular S with 2x2 blocks at the given leading indices, spectral radius < 1.
    fn quasi_triangular(n: usize, blocks: &[usize], seed: u64) -> Mat {
        let mut st = seed;
        let mut rnd = move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let mut s = Mat::zeros(n, n);
        for c in 0..n {
            for r in 0..c {
                s[(r, c)] = rnd();
            }
            s[(c, c)] = 1.8 * rnd();
        }
        for &k in blocks {
            // standardised 2x2 block [a b; c a] with b c < 0
            let (a, b, c) = (0.6 * rnd(), 0.3 + 0.2 * rnd(), -(0.3 + 0.2 * rnd()));
            s[(k, k)] = a;
            s[(k + 1, k + 1)] = a;
            s[(k, k + 1)] = b;
            s[(k + 1, k)] = c;
        }
        s
    }

    #[test]
    fn stein_downdate_matches_dynare_recursion() {
        for (n, blocks, seed) in [
            (1usize, vec![], 1u64),
            (2, vec![0usize], 2),
            (5, vec![1], 3),
            (9, vec![0, 3, 7], 4),
            (30, vec![2, 10, 11 + 2, 28], 5),
            (64, vec![0, 5, 20, 40, 62], 6),
        ] {
            let s = quasi_triangular(n, &blocks, seed);
            let r = Mat::from_fn(n, 3, |i, j| ((i * 3 + j) as f64 * 0.7).sin());
            let b = &r * r.transpose();
            let x1 = stein_dynare(&s, &b, |v| v == 0.0);
            let x2 = stein_downdate(&s, &b);
            let res = (&x2 - &s * &x2 * s.transpose() - &b).amax();
            let d = (&x1 - &x2).amax() / x1.amax();
            assert!(d < 1e-12 && res < 1e-11 * x1.amax(), "n {n}: rel diff {d:e}, residual {res:e}");
            assert_eq!(x2, x2.transpose());
        }
    }

    #[test]
    #[ignore]
    fn bench_stein() {
        let n = 64;
        let s = quasi_triangular(n, &[0, 5, 20, 40, 62], 6);
        let r = Mat::from_fn(n, 3, |i, j| ((i * 3 + j) as f64 * 0.7).sin());
        let b = &r * r.transpose();
        for (name, f) in [("dynare", 0), ("downdate", 1)] {
            let t = std::time::Instant::now();
            for _ in 0..2000 {
                std::hint::black_box(if f == 0 { stein_dynare(&s, &b, |v| v == 0.0) } else { stein_downdate(&s, &b) });
            }
            println!("{name}: {:.1} us", t.elapsed().as_secs_f64() * 1e6 / 2000.0);
        }
    }

    #[test]
    fn lyapunov_matches_fixed_point() {
        let a = Mat::from_row_slice(
            4,
            4,
            &[0.5, 0.2, 0.0, 0.1, -0.3, 0.4, 0.2, 0.0, 0.0, 0.5, 0.3, -0.2, 0.1, 0.0, 0.2, 0.6],
        );
        let r = Mat::from_row_slice(4, 2, &[1.0, 0.0, 0.5, 1.0, 0.0, 0.3, 0.2, 0.0]);
        let q = Mat::from_row_slice(2, 2, &[1.0, 0.2, 0.2, 0.5]);
        let p = lyapunov_solver(&a, &r, &q, 1.000001, 1e-15).unwrap();
        let p2 = naive_stein(&a, &(&r * &q * r.transpose()));
        assert!((p - p2).amax() < 1e-12);
    }

    #[test]
    fn diffuse_init_random_walk_plus_ar() {
        // x1 random walk, x2 AR(1), x3 static = x1 + x2 (zero column)
        let t = Mat::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 0.6, 0.0, 1.0, 0.6, 0.0]);
        let r = Mat::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let q = Mat::identity(2, 2);
        let (pstar, pinf) = compute_pinf_pstar(&[2], &t, &r, &q, 1.000001, &[]);
        // Pinf: unit variance on the random walk and on x3 (which loads on it)
        assert!((pinf[(0, 0)] - 1.0).abs() < 1e-12 && (pinf[(2, 2)] - 1.0).abs() < 1e-12 && pinf[(1, 1)].abs() < 1e-12);
        let v = 1.0 / (1.0 - 0.36);
        assert!((pstar[(1, 1)] - v).abs() < 1e-12);
        assert!(pstar[(0, 0)].abs() < 1e-12);
    }
}
