// SPDX-License-Identifier: GPL-3.0-or-later
// ---- block_trust_region ----
//! Port of Dynare 7.2's `block_trust_region` MEX (`mex/sources/block_trust_region/`:
//! `trust_region.f08`, `mexFunction.f08`, `matlab_fcn_closure.f08`), used by `dynare_solve`
//! for `solve_algo = 13` (Dulmage-Mendelsohn block decomposition) and `14` (whole system).
//!
//! Exit codes (`info`): -1 initial guess is a solution, 0 ill-behaved at the initial guess,
//! 1 success, 2 maxiter, 3 spurious convergence, 4 slow progress, 5 tolx too small.

use super::dmperm::dmperm;
use super::steady_state::{SolveOutcome, StaticFn, SteadyOptions};
use crate::linalg::{Mat, lu, lu_solve};
use std::os::raw::c_int;

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn wvec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

/// Minimum-norm least-squares solution (LAPACK `dgelsd`, rcond = -1), `None` on failure.
fn dgelsd_solve(a: &Mat, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.nrows() as c_int;
    let mut am = a.clone();
    let mut bm = b.to_vec();
    let mut sv = vec![0.0; n as usize];
    let (mut rank, mut info): (c_int, c_int) = (0, 0);
    let rcond = -1.0f64;
    let one: c_int = 1;
    let mut wq = [0.0f64; 1];
    let mut iq = [0 as c_int; 1];
    let lw: c_int = -1;
    unsafe {
        crate::linalg::lapack::dgelsd_(
            &n,
            &n,
            &one,
            am.as_mut_ptr(),
            &n,
            bm.as_mut_ptr(),
            &n,
            sv.as_mut_ptr(),
            &rcond,
            &mut rank,
            wq.as_mut_ptr(),
            &lw,
            iq.as_mut_ptr(),
            &mut info,
        );
    }
    let lwork = wq[0] as c_int;
    let mut work = vec![0.0; lwork.max(1) as usize];
    let mut iwork = vec![0 as c_int; (iq[0] as usize).max(1)];
    unsafe {
        crate::linalg::lapack::dgelsd_(
            &n,
            &n,
            &one,
            am.as_mut_ptr(),
            &n,
            bm.as_mut_ptr(),
            &n,
            sv.as_mut_ptr(),
            &rcond,
            &mut rank,
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    (info == 0).then_some(bm)
}

fn matvec(a: &Mat, x: &[f64], trans: bool) -> Vec<f64> {
    let n = a.nrows();
    if trans {
        (0..n).map(|j| (0..n).map(|i| a[(i, j)] * x[i]).sum()).collect()
    } else {
        (0..n).map(|i| (0..n).map(|j| a[(i, j)] * x[j]).sum()).collect()
    }
}

/// `dogleg` of `trust_region.f08`. `gn` is the cached Gauss-Newton direction.
fn dogleg(r: &Mat, b: &[f64], d: &[f64], delta: f64, gn: &mut Vec<f64>, recompute: bool) -> Vec<f64> {
    let n = b.len();
    if recompute {
        let (f, ipiv, info) = lu(r);
        if info == 0 {
            let mut m = Mat::from_column_slice(n, 1, b);
            lu_solve(&f, &ipiv, false, &mut m);
            *gn = m.iter().copied().collect();
        } else {
            *gn = dgelsd_solve(r, b).expect("Failed to compute the Gauss-Newton direction");
        }
    }
    let xn = norm2(&wvec(d, gn));
    if xn <= delta {
        return gn.clone();
    }
    let mut s = matvec(r, b, true);
    for i in 0..n {
        s[i] /= d[i];
    }
    let sn = norm2(&s);
    let (alpha, snm);
    if sn > 0.0 {
        for i in 0..n {
            s[i] = s[i] / sn / d[i];
        }
        let t = matvec(r, &s, false);
        let tn = norm2(&t);
        snm = sn / (tn * tn);
        if snm < delta {
            let bn = norm2(b);
            let dxn = delta / xn;
            let snmd = snm / delta;
            let tmp = (bn / sn) * (bn / xn) * snmd;
            alpha = dxn * (1.0 - snmd * snmd)
                / (tmp - dxn * snmd * snmd + ((tmp - dxn).powi(2) + (1.0 - dxn * dxn) * (1.0 - snmd * snmd)).sqrt());
        } else {
            alpha = 0.0;
        }
    } else {
        alpha = delta / xn;
        snm = 0.0;
    }
    let c = (1.0 - alpha) * snm.min(delta);
    (0..n).map(|i| alpha * gn[i] + c * s[i]).collect()
}

pub(crate) type Eval<'a> = &'a dyn Fn(&[f64]) -> (Vec<f64>, Mat);

/// `trust_region_solve`; `ev(x)` returns `(fvec, fjac)` restricted to the block.
pub(crate) fn trust_region_solve(ev: Eval, x: &mut Vec<f64>, tolx: f64, tolf: f64, maxiter: usize, factor: f64) -> i32 {
    let n = x.len();
    let maxslowiter = 15;
    let (mut niter, mut ncsucc, mut ncslow, mut info) = (1usize, 0usize, 0usize, 0i32);
    let mut delta = 0.0;
    let mut gn = vec![0.0; n];
    let mut dg = vec![0.0; n];
    let upd = |x: &[f64]| {
        let (fv, fj) = ev(x);
        let fnn = norm2(&fv);
        let jcn: Vec<f64> = (0..n).map(|i| norm2(&fj.column(i).iter().copied().collect::<Vec<_>>())).collect();
        (fv, fj, fnn, jcn)
    };
    let (mut fvec, mut fjac, mut fnn, mut jcn) = upd(x);
    let mut recompute = true;
    if fvec.iter().any(|v| v.is_nan()) || fjac.iter().any(|v| v.is_nan()) {
        return info;
    }
    if norm2(&fvec) < tolf {
        return -1;
    }
    while info == 0 {
        if niter == 1 {
            for i in 0..n {
                dg[i] = if jcn[i] != 0.0 { jcn[i] } else { 1.0 };
            }
        } else {
            for i in 0..n {
                dg[i] = (0.1 * dg[i]).max(jcn[i]);
            }
        }
        let mut xnorm = norm2(&wvec(x, &dg));
        if niter == 1 {
            delta = if xnorm > 0.0 { xnorm } else { 1.0 };
            delta *= factor;
        }
        let p = dogleg(&fjac, &fvec, &dg, delta, &mut gn, recompute);
        recompute = false;
        let p: Vec<f64> = p.iter().map(|v| -v).collect();
        let pnorm = norm2(&wvec(&dg, &p));
        let x2: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + b).collect();
        let fn2 = norm2(&ev(&x2).0);
        if fn2 < tolf {
            *x = x2;
            info = 1;
            continue;
        }
        let actred = if fn2 < fnn { 1.0 - (fn2 / fnn).powi(2) } else { -1.0 };
        let jp = matvec(&fjac, &p, false);
        let w: Vec<f64> = fvec.iter().zip(&jp).map(|(a, b)| a + b).collect();
        let t = norm2(&w);
        let prered = if t < fnn { 1.0 - (t / fnn).powi(2) } else { 0.0 };
        let ratio = if prered > 0.0 { actred / prered } else { 0.0 };
        if niter == 1 {
            delta = delta.min(pnorm);
        }
        if ratio < 0.1 {
            delta /= 2.0;
            ncsucc = 0;
        } else {
            ncsucc += 1;
            if (1.0 - ratio).abs() <= 0.1 {
                delta = 2.0 * pnorm;
            } else if ratio >= 0.5 || ncsucc > 1 {
                delta = delta.max(2.0 * pnorm);
            }
        }
        if ratio >= 1e-4 {
            let x0 = x.clone();
            *x = x2;
            (fvec, fjac, fnn, jcn) = upd(x);
            recompute = true;
            if fvec.iter().any(|v| v.is_nan()) || fjac.iter().any(|v| v.is_nan()) {
                *x = x0;
                (fvec, fjac, fnn, jcn) = upd(x);
                delta /= 2.0;
                ncslow += 1;
                niter += 1;
                continue;
            }
            xnorm = norm2(&wvec(x, &dg));
        }
        ncslow += 1;
        if actred >= 0.001 {
            ncslow = 0;
        }
        niter += 1;
        if niter == maxiter {
            info = 2;
            continue;
        }
        if delta <= tolx * xnorm {
            info = 3;
            continue;
        }
        if tolx > 0.0 && (0.1 * delta).max(pnorm) <= 10.0 * f64::EPSILON * xnorm {
            info = 5;
            continue;
        }
        if ncslow == maxslowiter {
            info = 4;
        }
    }
    info
}

/// `block_trust_region(f, x, tolf, tolx, maxit, factor, block_decompose, debug)`.
pub fn block_trust_region(f: StaticFn, x0: &[f64], opts: &SteadyOptions, block_decompose: bool) -> SolveOutcome {
    let (tolf, tolx, maxit, factor) = (opts.tolf, opts.tolx, opts.maxit, opts.trust_region_initial_step_bound_factor);
    let mut x = x0.to_vec();
    let mut info = 0;
    if block_decompose {
        let (fvec0, fjac) = f(&x);
        let dm = dmperm(&fjac);
        let nb = dm.r.len() - 1;
        for i in (0..nb).rev() {
            let rows: Vec<usize> = (dm.r[i]..dm.r[i + 1]).map(|k| dm.p[k]).collect();
            let cols: Vec<usize> = (dm.s[i]..dm.s[i + 1]).map(|k| dm.q[k]).collect();
            if rows.len() != cols.len() {
                let fb: Vec<f64> = rows.iter().map(|&k| fvec0[k]).collect();
                if norm2(&fb) < tolf {
                    continue;
                }
                println!(
                    "DYNARE_SOLVE (solve_algo=13): the Dulmage-Mendelsohn decomposition returned a non-square block. This means that the Jacobian is singular. You may want to try another value for solve_algo."
                );
                return SolveOutcome { x, errorflag: true, errorcode: -20 };
            }
            let xall = x.clone();
            let ev = |xb: &[f64]| {
                let mut xa = xall.clone();
                for (k, &c) in cols.iter().enumerate() {
                    xa[c] = xb[k];
                }
                let (fv, fj) = f(&xa);
                (
                    rows.iter().map(|&r| fv[r]).collect::<Vec<_>>(),
                    Mat::from_fn(rows.len(), cols.len(), |a, b| fj[(rows[a], cols[b])]),
                )
            };
            let mut xb: Vec<f64> = cols.iter().map(|&c| x[c]).collect();
            info = trust_region_solve(&ev, &mut xb, tolx, tolf, maxit, factor);
            for (k, &c) in cols.iter().enumerate() {
                x[c] = xb[k];
            }
        }
        let fvec = f(&x).0;
        // Fortran maxval(abs(.)): NaN-ignoring in practice only if all finite; plain max here
        let fmax = fvec.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if fmax > tolf {
            info = trust_region_solve(&|y: &[f64]| f(y), &mut x, tolx, tolf, maxit, factor);
        } else if nb > 1 {
            info = 1;
        }
    } else {
        info = trust_region_solve(&|y: &[f64]| f(y), &mut x, tolx, tolf, maxit, factor);
    }
    SolveOutcome { x, errorflag: !(info == 1 || info == -1), errorcode: info }
}
