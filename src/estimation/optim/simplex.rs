// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of MATLAB's `fminsearch` (Lagarias et al. Nelder–Mead), used by
//! Dynare for `mode_compute = 7` with the options set in
//! `dynare_minimize_objective.m`.

use super::OptimResult;
use crate::estimation::Objective;

/// `optimset('MaxFunEvals',1000000,'MaxIter',6000,'TolFun',1e-8,'TolX',1e-6)`.
#[derive(Clone, Debug)]
pub struct FminsearchOptions {
    pub max_fun_evals: usize,
    pub max_iter: usize,
    pub tol_fun: f64,
    pub tol_x: f64,
}

impl Default for FminsearchOptions {
    fn default() -> Self {
        FminsearchOptions { max_fun_evals: 1_000_000, max_iter: 6000, tol_fun: 1e-8, tol_x: 1e-6 }
    }
}

/// MATLAB `eps(x)`: distance from |x| to the next larger double.
fn eps_of(x: f64) -> f64 {
    let a = x.abs();
    if !a.is_finite() {
        return f64::NAN;
    }
    if a < f64::MIN_POSITIVE {
        return f64::from_bits(1);
    }
    f64::from_bits(a.to_bits() + 1) - a
}

/// Port of `fminsearch.m` (raw objective, as Dynare calls it).
pub fn fminsearch(obj: &dyn Objective, x0: &[f64], opts: &FminsearchOptions) -> OptimResult {
    let n = x0.len();
    let (rho, chi, psi, sigma) = (1.0, 2.0, 0.5, 0.5);
    let np1 = n + 1;
    let f = |z: &[f64]| obj.eval(z).0;
    let mut v: Vec<Vec<f64>> = vec![x0.to_vec()];
    let mut fv = vec![f(x0)];
    for j in 0..n {
        let mut y = x0.to_vec();
        y[j] = if y[j] != 0.0 { 1.05 * y[j] } else { 0.00025 };
        fv.push(f(&y));
        v.push(y);
    }
    let sort = |v: &mut Vec<Vec<f64>>, fv: &mut Vec<f64>| {
        let mut idx: Vec<usize> = (0..fv.len()).collect();
        idx.sort_by(|&a, &b| fv[a].partial_cmp(&fv[b]).unwrap_or(std::cmp::Ordering::Equal));
        *v = idx.iter().map(|&i| v[i].clone()).collect();
        *fv = idx.iter().map(|&i| fv[i]).collect();
    };
    sort(&mut v, &mut fv);
    let mut itercount = 1;
    let mut func_evals = np1;
    let mut converged = false;
    while func_evals < opts.max_fun_evals && itercount < opts.max_iter {
        let tf = opts.tol_fun.max(10.0 * eps_of(fv[0]));
        let vmax = v[0].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let tx = opts.tol_x.max(10.0 * eps_of(vmax));
        if fv[1..].iter().all(|fj| (fv[0] - fj).abs() <= tf)
            && v[1..].iter().all(|vj| vj.iter().zip(&v[0]).all(|(a, b)| (a - b).abs() <= tx))
        {
            converged = true;
            break;
        }
        let xbar: Vec<f64> = (0..n).map(|k| v[..n].iter().map(|vj| vj[k]).sum::<f64>() / n as f64).collect();
        let comb = |a: f64, b: f64| -> Vec<f64> { (0..n).map(|k| a * xbar[k] + b * v[n][k]).collect() };
        let xr = comb(1.0 + rho, -rho);
        let fxr = f(&xr);
        func_evals += 1;
        if fxr < fv[0] {
            let xe = comb(1.0 + rho * chi, -rho * chi);
            let fxe = f(&xe);
            func_evals += 1;
            if fxe < fxr {
                v[n] = xe;
                fv[n] = fxe;
            } else {
                v[n] = xr;
                fv[n] = fxr;
            }
        } else if fxr < fv[n - 1] {
            v[n] = xr;
            fv[n] = fxr;
        } else {
            let mut shrink = false;
            if fxr < fv[n] {
                let xc = comb(1.0 + psi * rho, -psi * rho);
                let fxc = f(&xc);
                func_evals += 1;
                if fxc <= fxr {
                    v[n] = xc;
                    fv[n] = fxc;
                } else {
                    shrink = true;
                }
            } else {
                let xcc = comb(1.0 - psi, psi);
                let fxcc = f(&xcc);
                func_evals += 1;
                if fxcc < fv[n] {
                    v[n] = xcc;
                    fv[n] = fxcc;
                } else {
                    shrink = true;
                }
            }
            if shrink {
                for j in 1..np1 {
                    let vj: Vec<f64> = (0..n).map(|k| v[0][k] + sigma * (v[j][k] - v[0][k])).collect();
                    fv[j] = f(&vj);
                    v[j] = vj;
                }
                func_evals += n;
            }
        }
        sort(&mut v, &mut fv);
        itercount += 1;
    }
    let message = if converged {
        "Optimization terminated: the current x satisfies the termination criteria using OPTIONS.TolX and F(X) satisfies the convergence criteria using OPTIONS.TolFun".to_string()
    } else if func_evals >= opts.max_fun_evals {
        "Exiting: Maximum number of function evaluations has been exceeded".to_string()
    } else {
        "Exiting: Maximum number of iterations has been exceeded".to_string()
    };
    OptimResult {
        x: v[0].clone(),
        fval: fv[0],
        hessian: None,
        inv_hessian: None,
        iterations: itercount,
        func_count: func_evals,
        exitflag: if converged { 1 } else { 0 },
        message,
    }
}
