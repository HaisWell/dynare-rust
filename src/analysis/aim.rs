// SPDX-License-Identifier: GPL-3.0-or-later
//! `aim_solver` / `dr=aim`: port of Dynare's `stochastic_solver/AIM_first_order_solver.m`
//! and of the subset of Gary Anderson's AIM package shipped in `matlab/AIM/` (`SPAmalg.m`,
//! `SPExact_shift.m`, `SPNumeric_shift.m`, `SPBuild_a.m`, `SPEigensystem.m`, `SPCopy_w.m`,
//! `SPReduced_form.m`, `SPObstruct.m`, `SPShiftright.m`, `SPAimerr.m`).
//!
//! The system is Dynare's dynamic Jacobian `[f_{-1} f_0 f_{+1} f_u]` (one lag, one lead after
//! the aux-variable substitution); AIM returns `x_t = B x_{t-1}`, from which `ghx`/`ghu` are
//! read in DR order. As in Dynare, the eigenvalues of the state transition are displayed
//! (`disp(dr.eigval)`) on every call.

use anyhow::{Result, bail};
use nalgebra::Complex;

use crate::linalg::{Mat, eig_general_vectors, lu, norm1, rcond_from_lu, solve};
use crate::solver::{DecisionRule, FirstOrderSolver, SolveError};

/// `SPAimerr.m`.
pub fn aim_err(c: i32) -> &'static str {
    match c {
        1 => "Aim: unique solution.",
        2 => "Aim: roots not correctly computed by real_schur.",
        3 => "Aim: too many big roots.",
        35 => "Aim: too many big roots, and q(:,right) is singular.",
        4 => "Aim: too few big roots.",
        45 => "Aim: too few big roots, and q(:,right) is singular.",
        5 => "Aim: q(:,right) is singular.",
        61 => "Aim: too many exact shiftrights.",
        62 => "Aim: too many numeric shiftrights.",
        _ => "Aimerr: return code not properly specified",
    }
}

/// `SPShiftright(x, n)`: columns shifted right by `n`, zeros on the left.
fn shift_right(x: &Mat, n: usize) -> Mat {
    let (r, c) = x.shape();
    let mut y = Mat::zeros(r, c);
    for j in n..c {
        y.set_column(j, &x.column(j - n));
    }
    y
}

fn rcond(a: &Mat) -> f64 {
    if a.nrows() == 0 {
        return f64::INFINITY;
    }
    let (f, _, info) = lu(a);
    if info > 0 { 0.0 } else { rcond_from_lu(&f, norm1(a)) }
}

/// `-a \ b` (square `a`), NaN-filled when `a` is exactly singular (MATLAB warns and returns Inf/NaN).
fn neg_solve(a: &Mat, b: &Mat) -> Mat {
    match solve(a, b) {
        Some(x) => -x,
        None => Mat::from_element(b.nrows(), b.ncols(), f64::NAN),
    }
}

/// Output of [`spamalg`]: `b` (neq x neq*nlag) and `aimcode`.
pub struct Amalg {
    pub b: Mat,
    pub aimcode: i32,
}

/// `SPAmalg(h, neq, nlag = 1, nlead = 1, condn, uprbnd)`.
pub fn spamalg(h: &Mat, neq: usize, condn: f64, uprbnd: f64) -> Amalg {
    let (nlag, nlead) = (1usize, 1usize);
    let qrows = neq * nlead;
    let qcols = neq * (nlag + nlead);
    let bcols = neq * nlag;
    let mut q = Mat::zeros(qrows, qcols);
    let mut h = h.clone();
    let mut iq = 0usize;
    let fail = |c| Amalg { b: Mat::zeros(0, 0), aimcode: c };
    // SPExact_shift
    let right = qcols..qcols + neq;
    let zero_rows = |h: &Mat| -> Vec<usize> { (0..neq).filter(|&i| right.clone().all(|j| h[(i, j)] == 0.0)).collect() };
    let mut nexact = 0;
    let mut zr = zero_rows(&h);
    while !zr.is_empty() && iq <= qrows {
        for (k, &i) in zr.iter().enumerate() {
            if iq + k < qrows {
                for j in 0..qcols {
                    q[(iq + k, j)] = h[(i, j)];
                }
            }
            let row = h.rows(i, 1).into_owned();
            h.set_row(i, &shift_right(&row, neq).row(0));
        }
        iq += zr.len();
        nexact += zr.len();
        zr = zero_rows(&h);
    }
    if iq > qrows {
        return fail(61);
    }
    // SPNumeric_shift
    let mut nnumeric = 0;
    let qr_zero_rows = |h: &Mat| -> (Mat, Vec<usize>) {
        let hr = h.columns(qcols, neq).into_owned();
        let qr = hr.col_piv_qr();
        let r = qr.r();
        let qm = qr.q();
        (qm, (0..neq).filter(|&i| r[(i, i)].abs() <= condn).collect())
    };
    let (mut qm, mut zr) = qr_zero_rows(&h);
    while !zr.is_empty() && iq <= qrows {
        h = qm.transpose() * &h;
        for (k, &i) in zr.iter().enumerate() {
            if iq + k < qrows {
                for j in 0..qcols {
                    q[(iq + k, j)] = h[(i, j)];
                }
            }
            let row = h.rows(i, 1).into_owned();
            h.set_row(i, &shift_right(&row, neq).row(0));
        }
        iq += zr.len();
        nnumeric += zr.len();
        (qm, zr) = qr_zero_rows(&h);
    }
    if iq > qrows {
        return fail(62);
    }
    // SPBuild_a
    let hleft = neg_solve(&h.columns(qcols, neq).into_owned(), &h.columns(0, qcols).into_owned());
    let mut a = Mat::zeros(qcols, qcols);
    if qcols > neq {
        for k in 0..qcols - neq {
            a[(k, neq + k)] = 1.0;
        }
    }
    for i in 0..neq {
        for j in 0..qcols {
            a[(qcols - neq + i, j)] = hleft[(i, j)];
        }
    }
    let mut js: Vec<usize> = (0..qcols).collect();
    loop {
        let zc: Vec<bool> = (0..a.ncols()).map(|j| a.column(j).iter().all(|&x| x == 0.0)).collect();
        if !zc.iter().any(|&z| z) {
            break;
        }
        let keep: Vec<usize> = (0..a.ncols()).filter(|&j| !zc[j]).collect();
        a = a.select_columns(&keep).select_rows(&keep);
        js = keep.iter().map(|&k| js[k]).collect();
    }
    let ia = js.len();
    let mut lgroots = 0;
    if ia != 0 {
        if a.iter().any(|x| !x.is_finite()) {
            println!("A is NAN or INF");
            return fail(63);
        }
        // SPEigensystem: [w, d] = eig(a'), sorted by decreasing modulus, real(w) + imag(w)
        let (rts, vr) = eig_general_vectors(&a.transpose());
        let n = rts.len();
        let mut w = Mat::zeros(n, n);
        let mut k = 0;
        while k < n {
            if rts[k].im != 0.0 && k + 1 < n {
                let (re, im) = (vr.column(k).into_owned(), vr.column(k + 1).into_owned());
                w.set_column(k, &(&re + &im));
                w.set_column(k + 1, &(&re - &im));
                k += 2;
            } else {
                w.set_column(k, &vr.column(k));
                k += 1;
            }
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| (-rts[i].norm()).partial_cmp(&-rts[j].norm()).unwrap_or(std::cmp::Ordering::Equal));
        let w = w.select_columns(&idx);
        lgroots = rts.iter().filter(|z| z.norm() > uprbnd).count();
        // SPCopy_w
        if iq < qrows {
            for (r, row) in (iq..qrows).enumerate() {
                for (c, &jc) in js.iter().enumerate() {
                    q[(row, jc)] = w[(c, r)];
                }
            }
        }
    }
    let test = nexact + nnumeric + lgroots;
    let mut aimcode = 0;
    if test > qrows {
        aimcode = 3;
    } else if test < qrows {
        aimcode = 4;
    }
    let mut b = Mat::zeros(0, 0);
    if aimcode == 0 {
        // SPReduced_form
        let qleft = q.columns(0, qcols - qrows).into_owned();
        let qright = q.columns(qcols - qrows, qrows).into_owned();
        let mut nonsing = rcond(&qright) > condn;
        if nonsing {
            b = neg_solve(&qright, &qleft).view((0, 0), (neq, bcols)).into_owned();
        } else {
            let themax: Vec<f64> =
                (0..qrows).map(|i| qright.row(i).iter().fold(0.0f64, |m, x| m.max(x.abs()))).collect();
            let scale = |m: &Mat| Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] / themax[i]);
            let (sr, sl) = (scale(&qright), scale(&qleft));
            nonsing = rcond(&sr) > condn;
            if nonsing {
                b = neg_solve(&sr, &sl).view((0, 0), (neq, bcols)).into_owned();
            }
        }
        aimcode = if nonsing { 1 } else { 5 };
    }
    Amalg { b, aimcode }
}

/// `SPObstruct(cof, b, neq, 1, 1)`: `[f_{-1}, f_0 + f_{+1} b]`.
fn spobstruct(cof: &Mat, b: &Mat, neq: usize) -> Mat {
    let mut scof = Mat::zeros(neq, 2 * neq);
    let fb = cof.columns(2 * neq, neq) * b;
    for i in 0..neq {
        for j in 0..neq {
            scof[(i, j)] = cof[(i, j)];
            scof[(i, neq + j)] = cof[(i, neq + j)] + fb[(i, j)];
        }
    }
    scof
}

/// MATLAB `disp` of a column vector in `format short` (the `disp(dr.eigval)` of
/// `AIM_first_order_solver.m`).
pub fn disp_column(v: &[Complex<f64>]) -> String {
    let mut s = String::new();
    if v.is_empty() {
        return s;
    }
    if v.iter().any(|z| z.im != 0.0) {
        for z in v {
            let sign = if z.im < 0.0 || (z.im == 0.0 && z.im.is_sign_negative()) { '-' } else { '+' };
            s += &format!("  {:>7.4} {sign} {:.4}i\n", z.re, z.im.abs());
        }
        return s;
    }
    let x: Vec<f64> = v.iter().map(|z| z.re).collect();
    if x.iter().all(|&a| a.fract() == 0.0 && a.abs() < 1e9) {
        for a in &x {
            s += &format!("{:>6}\n", *a as i64);
        }
        return s;
    }
    let m = x.iter().fold(0.0f64, |m, a| m.max(a.abs()));
    let (scale, header) = if m >= 1000.0 || m < 1e-3 {
        let e = m.log10().floor() as i32;
        (10f64.powi(e), Some(format!("   1.0e{}{:02} *\n\n", if e < 0 { '-' } else { '+' }, e.abs())))
    } else {
        (1.0, None)
    };
    if let Some(h) = header {
        s += &h;
    }
    for a in &x {
        if *a == 0.0 {
            s += &format!("{:>10}\n", 0);
        } else {
            s += &format!("{:>10.4}\n", a / scale);
        }
    }
    s
}

/// Port of `AIM_first_order_solver(g1, M_, dr, qz_criterium)`: the first-order decision rule
/// by AIM. Prints the eigenvalues of the state transition (as Dynare). `Err(SolveError)`
/// carries Dynare's info (3/4: Blanchard-Kahn); AIM failures that make Dynare stop are `Err`
/// of the outer `Result`.
pub fn aim_first_order_solver(
    solver: &FirstOrderSolver,
    g1: &Mat,
    ys: &[f64],
    qz_criterium: f64,
) -> Result<Result<DecisionRule, SolveError>> {
    let s = &solver.structure;
    let n = s.n_endo;
    let neq = g1.nrows();
    let condn = 1e-10;
    let uprbnd = 1.0 + 1e-6;
    let h = g1.columns(0, 3 * n).into_owned();
    let am = spamalg(&h, neq, condn, uprbnd);
    if am.aimcode != 1 {
        let e = aim_err(am.aimcode);
        println!("Error in AIM: aimcode={} : {e}", am.aimcode);
        if am.aimcode < 1 || am.aimcode > 5 {
            bail!("Error in AIM: aimcode={} ; {e}", am.aimcode);
        }
        // AIM_first_order_solver.m then calls convertAimCodeToInfo(aimCode) -- a misspelt
        // variable: MATLAB stops there.
        bail!("Unrecognized function or variable 'aimCode'.");
    }
    let ov = &s.order_var;
    let (nstatic, nspred, nsfwrd) = (s.nstatic, s.nspred, s.nsfwrd);
    let b = &am.b;
    let ghx = Mat::from_fn(n, nspred, |i, j| b[(ov[i], ov[nstatic + j])]);
    let ghu = if s.n_exo > 0 {
        let scof = spobstruct(&h, b, neq);
        let scof1 = Mat::from_fn(neq, n, |i, j| scof[(i, n + ov[j])]);
        let fu = g1.columns(3 * n, s.n_exo).into_owned();
        neg_solve(&scof1, &fu)
    } else {
        Mat::zeros(n, 0)
    };
    // kalman_transition_matrix(dr, nstatic+(1:nspred), 1:nspred): the state rows of ghx
    let a = ghx.rows(nstatic, nspred).into_owned();
    let eigval = crate::linalg::eig_general(&a);
    // disp of a numeric array ends with a blank line
    println!("{}", disp_column(&eigval));
    let dr = DecisionRule {
        ys: ys.to_vec(),
        ghx,
        ghu,
        order_var: s.order_var.clone(),
        inv_order_var: s.inv_order_var.clone(),
        state_var: s.state_var.clone(),
        nstatic,
        nfwrd: s.nfwrd,
        npred: s.npred,
        nboth: s.nboth,
        nspred,
        nsfwrd,
        eigval_abs: eigval.iter().map(|z| z.norm()).collect(),
        sdim: eigval.iter().filter(|z| z.norm() < qz_criterium).count(),
        eigval,
    };
    let nd = nspred + nsfwrd;
    let nba = nd - dr.sdim;
    if nba != nsfwrd {
        let mut temp = dr.eigval_abs.clone();
        temp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // temp has nspred entries (MATLAB indexes it with nd, the indices below stay in range
        // only for nba > nsfwrd, the explosive case)
        let (info, lo, hi) = if nba > nsfwrd { (3, nd - nba, nd - nsfwrd) } else { (4, nd - nsfwrd, nd - nba) };
        let pen: f64 =
            temp.get(lo..hi.min(temp.len())).unwrap_or(&[]).iter().map(|t| (t - 1.0 - qz_criterium).powi(2)).sum();
        return Ok(Err(SolveError::new(info, pen)));
    }
    Ok(Ok(dr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disp_formats() {
        let r = |v: &[f64]| disp_column(&v.iter().map(|&x| Complex::new(x, 0.0)).collect::<Vec<_>>());
        assert_eq!(r(&[0.4005, 0.8, 0.5]), "    0.4005\n    0.8000\n    0.5000\n");
        assert_eq!(r(&[0.95, 0.0]), "    0.9500\n         0\n");
        assert_eq!(r(&[1.0, 0.0]), "     1\n     0\n");
    }
}
