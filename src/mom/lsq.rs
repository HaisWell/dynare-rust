// SPDX-License-Identifier: GPL-3.0-or-later
//! Substitute for `mode_compute = 13` (MATLAB's `lsqnonlin`, Optimization Toolbox, Dynare's
//! default optimizer for GMM/SMM when the toolbox is licensed): a bounded Levenberg–Marquardt
//! on the residual vector of the moment distance (`options_mom_.mom.vector_output = 1`),
//! forward-difference Jacobian, steps projected on the bounds. It is not MATLAB's
//! trust-region-reflective algorithm; it converges to the same (local) minimum within the
//! tolerances, not along the same path.

use rayon::prelude::*;

/// Options (`optimoptions('lsqnonlin', 'MaxFunEvals', 5000, 'MaxIter', 5000, 'TolFun', 1e-6,
/// 'TolX', 1e-6)` in `dynare_minimize_objective.m`).
#[derive(Clone, Debug)]
pub struct LsqOptions {
    pub max_iter: usize,
    pub max_fun_evals: usize,
    pub tol_fun: f64,
    pub tol_x: f64,
    pub verbose: bool,
}

impl Default for LsqOptions {
    fn default() -> Self {
        LsqOptions { max_iter: 5000, max_fun_evals: 5000, tol_fun: 1e-6, tol_x: 1e-6, verbose: true }
    }
}

pub struct LsqResult {
    pub x: Vec<f64>,
    /// Sum of squared residuals.
    pub fval: f64,
    pub iterations: usize,
    pub func_count: usize,
    pub message: String,
}

fn ssq(r: &[f64]) -> f64 {
    r.iter().map(|v| v * v).sum()
}

/// Minimizes `||f(x)||^2` subject to `lb <= x <= ub`.
/// `jac` (analytic Jacobian, one `Vec` per column) replaces the forward differences when given.
pub fn lsq_bounded(
    f: &(dyn Fn(&[f64]) -> Vec<f64> + Sync),
    jac: Option<&(dyn Fn(&[f64]) -> Option<Vec<Vec<f64>>> + Sync)>,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    o: &LsqOptions,
) -> LsqResult {
    let n = x0.len();
    let clamp = |x: &mut [f64]| {
        for i in 0..n {
            x[i] = x[i].max(lb[i]).min(ub[i]);
        }
    };
    let mut x = x0.to_vec();
    clamp(&mut x);
    let mut r = f(&x);
    let mut fx = ssq(&r);
    let mut evals = 1;
    let mut lambda = 1e-3;
    let mut iter = 0;
    let mut message = "maximum number of iterations reached".to_string();
    if o.verbose {
        println!("\n Iteration  Func-count     Resnorm      lambda");
        println!("{:>10} {:>11} {:>13.6e}", 0, evals, fx);
    }
    while iter < o.max_iter && evals < o.max_fun_evals {
        iter += 1;
        // forward-difference Jacobian (columns in parallel)
        let analytic_cols = jac.and_then(|j| j(&x));
        let have_analytic = analytic_cols.is_some();
        let cols: Vec<Vec<f64>> = analytic_cols.unwrap_or_else(|| {
            (0..n)
                .into_par_iter()
                .map(|j| {
                    let mut xp = x.clone();
                    let mut h = f64::EPSILON.sqrt() * x[j].abs().max(1.0);
                    if xp[j] + h > ub[j] {
                        h = -h;
                    }
                    xp[j] += h;
                    let h = xp[j] - x[j];
                    let rp = f(&xp);
                    rp.iter().zip(&r).map(|(a, b)| (a - b) / h).collect()
                })
                .collect()
        });
        if !have_analytic {
            evals += n;
        }
        let m = r.len();
        let mut jtj = vec![vec![0.0; n]; n];
        let mut jtr = vec![0.0; n];
        for a in 0..n {
            for b in 0..n {
                jtj[a][b] = (0..m).map(|k| cols[a][k] * cols[b][k]).sum();
            }
            jtr[a] = (0..m).map(|k| cols[a][k] * r[k]).sum();
        }
        let mut improved = false;
        for _ in 0..30 {
            let a = crate::linalg::Mat::from_fn(n, n, |i, j| {
                jtj[i][j] + if i == j { lambda * jtj[i][i].max(1e-12) } else { 0.0 }
            });
            let b = crate::linalg::Mat::from_fn(n, 1, |i, _| -jtr[i]);
            let Some(dx) = crate::linalg::solve(&a, &b) else {
                lambda *= 10.0;
                continue;
            };
            let mut xt: Vec<f64> = (0..n).map(|i| x[i] + dx[(i, 0)]).collect();
            clamp(&mut xt);
            let rt = f(&xt);
            evals += 1;
            let ft = ssq(&rt);
            if ft.is_finite() && ft < fx {
                let step: f64 = (0..n).map(|i| (xt[i] - x[i]).powi(2)).sum::<f64>().sqrt();
                let xn: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                let rel = (fx - ft) / fx.max(f64::MIN_POSITIVE);
                x = xt;
                r = rt;
                fx = ft;
                lambda = (lambda / 10.0).max(1e-12);
                improved = true;
                if o.verbose {
                    println!("{iter:>10} {evals:>11} {fx:>13.6e} {lambda:>11.3e}");
                }
                if rel < o.tol_fun * 1e-4 || step < o.tol_x * 1e-4 * (1.0 + xn) {
                    message =
                        "Local minimum possible: the change in the sum of squares or in x is below the tolerance."
                            .into();
                    return LsqResult { x, fval: fx, iterations: iter, func_count: evals, message };
                }
                break;
            }
            lambda *= 10.0;
            if lambda > 1e16 {
                break;
            }
        }
        if !improved {
            message = "Local minimum found: no step decreases the sum of squares.".into();
            break;
        }
    }
    LsqResult { x, fval: fx, iterations: iter, func_count: evals, message }
}
