// SPDX-License-Identifier: GPL-3.0-or-later
//! OccBin's inversion filter (IVF): ports of `+occbin/IVF_core.m`, `+occbin/match_function.m`,
//! the likelihood of `+occbin/IVF_posterior.m`, and of `optimization/csolve.m` with the
//! `dynare_solve.m` preamble it runs under (`options_.occbin.solver.solve_algo = 3`).
//!
//! Each period the shocks mapped to the observables are solved so that the piecewise-linear
//! model reproduces the observations exactly; the likelihood is the shocks' Gaussian density
//! times the Jacobian of the shocks -> observables mapping.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use super::pkf::PointModel;
use super::regimes::Regimes;
use super::solver::SimulOptions;
use crate::linalg::{self, Mat};

/// Residuals and Jacobian of a system of equations (`[f, grad] = feval(f, x)`).
pub type ResidualFn<'a> = dyn FnMut(&[f64]) -> (Vec<f64>, Mat) + 'a;

/// `options_.occbin.solver` (defaults of `occbin.set_default_options`).
#[derive(Clone, Debug)]
pub struct MatchSolverOptions {
    pub maxit: usize,
    pub tolf: f64,
}

impl Default for MatchSolverOptions {
    fn default() -> Self {
        MatchSolverOptions { maxit: 10, tolf: 1e-5 }
    }
}

/// Port of `csolve.m` (Sims): Newton with line search on `sum(abs(f))`. `fun(x)` returns the
/// residuals and the Jacobian. Returns the solution and `rc` (0 = converged).
pub fn csolve(fun: &mut ResidualFn, x0: &[f64], crit: f64, itmax: usize) -> (Vec<f64>, i32) {
    let delta = 1e-6;
    let alpha = 1e-3;
    let nv = x0.len();
    let mut x = x0.to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let sabs = |f: &[f64]| f.iter().map(|v| v.abs()).sum::<f64>();
    let (mut f0, _) = fun(&x);
    let mut af0 = sabs(&f0);
    let mut af00 = af0;
    let mut itct = 0usize;
    let rc;
    loop {
        let randomize = itct > 3 && af00 - af0 < crit * af0.max(1.0) && itct % 2 == 1;
        let dx0: Vec<f64> = if randomize {
            let nx = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            (0..nv)
                .map(|_| {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    nx / z
                })
                .collect::<Vec<f64>>()
        } else {
            let (f, mut grad) = fun(&x);
            f0 = f;
            if rcond(&grad) < 1e-12 {
                for i in 0..nv {
                    grad[(i, i)] += delta;
                }
            }
            let (lu, ipiv, _) = linalg::lu(&grad);
            let mut b = Mat::from_column_slice(nv, 1, &f0);
            linalg::lu_solve(&lu, &ipiv, false, &mut b);
            b.iter().map(|v| -v).collect()
        };
        let mut lambda: f64 = 1.0;
        let mut fmin = f0.clone();
        let mut xmin = x.clone();
        let mut afmin = af0;
        let dx_size = dx0.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut factor: f64 = 0.6;
        let mut shrink = true;
        loop {
            let xt: Vec<f64> = x.iter().zip(&dx0).map(|(a, d)| a + lambda * d).collect();
            let (f, _) = fun(&xt);
            let af = sabs(&f);
            if af < afmin {
                afmin = af;
                fmin = f;
                xmin = xt;
            }
            if (lambda > 0.0 && af0 - af < alpha * lambda * af0) || (lambda < 0.0 && af0 - af < 0.0) {
                if !shrink {
                    factor = factor.powf(0.6);
                    shrink = true;
                }
                if (lambda * (1.0 - factor)).abs() * dx_size > 0.1 * delta {
                    lambda *= factor;
                } else if lambda > 0.0 && factor == 0.6 {
                    lambda = -0.3;
                } else {
                    // rc = 1, 2 or 3 (line search failed); only the final rc matters
                    break;
                }
            } else if lambda > 0.0 && af - af0 > (1.0 - alpha) * lambda * af0 {
                if shrink {
                    factor = factor.powf(0.6);
                    shrink = false;
                }
                lambda /= factor;
            } else {
                break;
            }
        }
        itct += 1;
        x = xmin;
        f0 = fmin;
        af00 = af0;
        af0 = afmin;
        if itct >= itmax {
            rc = 4;
            break;
        } else if af0 < crit {
            rc = 0;
            break;
        }
    }
    (x, rc)
}

/// MATLAB `rcond` (LAPACK dgecon on the LU factors); NaN for a matrix with NaN.
fn rcond(a: &Mat) -> f64 {
    if a.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let (lu, _, info) = linalg::lu(a);
    if info > 0 {
        return 0.0;
    }
    linalg::rcond_from_lu(&lu, linalg::norm1(a))
}

/// `dynare_solve(f, x, maxit, tolf, ...)` with `solve_algo = 3` and an analytic Jacobian:
/// returns the solution and `errorflag`.
pub fn dynare_solve_csolve(fun: &mut ResidualFn, x0: &[f64], o: &MatchSolverOptions) -> (Vec<f64>, bool) {
    let (f, _) = fun(x0);
    if f.iter().all(|v| v.is_finite()) && f.iter().fold(0.0f64, |m, v| m.max(v.abs())) < o.tolf {
        return (x0.to_vec(), false);
    }
    let (x, rc) = csolve(fun, x0, o.tolf, o.maxit);
    (x, rc != 0)
}

/// Output of [`ivf_core`] (the filtered shocks, states and regimes of the IVF).
#[derive(Clone, Debug)]
pub struct IvfCore {
    /// nobs x nobs-shocks (columns in observable order, NaN when missing).
    pub filtered_errs: Mat,
    /// nobs x pp residuals of the observation equations.
    pub resids: Mat,
    /// nobs x n_endo: states (deviations, declaration order).
    pub stateval: Mat,
    /// Per period, `ss.R(:, exo_pos)` (n_endo x pp; columns of missing observables are zero).
    pub emat: Vec<Mat>,
    pub regime_history: Vec<Regimes>,
    /// `error_code(1)`, `error_code(4)` (0 = success).
    pub info: i32,
    pub info4: f64,
}

/// One residual evaluation of `match_function.m` at shocks `err` (deviations from the steady
/// state in `out.0`).
struct Matched {
    resids: Vec<f64>,
    grad: Mat,
    state: Vec<f64>,
    e: Mat,
    regime: Option<Regimes>,
}

fn match_function(pm: &mut PointModel, base: &SimulOptions, err: &[f64], varobs: &[usize], obs: &[f64]) -> Matched {
    let mut o = base.clone();
    o.shocks = Mat::from_row_slice(1, err.len(), err);
    let n = varobs.len();
    match pm.solve(&o) {
        Ok(out) => {
            let nv = pm.dm.n_vars;
            let state: Vec<f64> = (0..nv).map(|j| out.piecewise[(0, j)]).collect();
            let r = &out.ss[0].r;
            let e = Mat::from_fn(nv, o.exo_pos.len(), |i, k| r[(i, o.exo_pos[k])]);
            let grad = Mat::from_fn(n, o.exo_pos.len(), |i, k| r[(varobs[i], o.exo_pos[k])]);
            let resids = (0..n).map(|i| (state[varobs[i]] + pm.ys[varobs[i]]) - obs[i]).collect();
            Matched { resids, grad, state, e, regime: Some(out.regime_history[0].clone()) }
        }
        Err(_) => Matched {
            resids: vec![100.0; n],
            grad: Mat::from_element(n, o.exo_pos.len(), f64::NAN),
            state: Vec::new(),
            e: Mat::zeros(0, 0),
            regime: None,
        },
    }
}

/// Port of `IVF_core.m`. `obs`: nobs x pp observations in levels (NaN = missing), `varobs` the
/// declaration indices of the observables, `err_index` the shock of each observable
/// (`IVF_shock_observable_mapping`). `simul`: the likelihood's solver options.
pub fn ivf_core(
    pm: &mut PointModel,
    simul: &SimulOptions,
    solver: &MatchSolverOptions,
    obs: &Mat,
    varobs: &[usize],
    err_index: &[usize],
) -> IvfCore {
    let (t_len, pp) = (obs.nrows(), obs.ncols());
    let nv = pm.dm.n_vars;
    let mut res = IvfCore {
        filtered_errs: Mat::zeros(t_len, pp),
        resids: Mat::zeros(t_len, pp),
        stateval: Mat::zeros(t_len, nv),
        emat: Vec::with_capacity(t_len),
        regime_history: Vec::with_capacity(t_len),
        info: 0,
        info4: 0.0,
    };
    let mut init_val = vec![0.0; nv];
    for t in 0..t_len {
        let inan: Vec<usize> = (0..pp).filter(|&j| !obs[(t, j)].is_nan()).collect();
        let cur: Vec<f64> = inan.iter().map(|&j| obs[(t, j)]).collect();
        let vo: Vec<usize> = inan.iter().map(|&j| varobs[j]).collect();
        let mut o = simul.clone();
        o.exo_pos = inan.iter().map(|&j| err_index[j]).collect();
        o.endo_init = init_val.clone();
        let x0 = vec![0.0; inan.len()];
        let mut f = |x: &[f64]| {
            let m = match_function(pm, &o, x, &vo, &cur);
            (m.resids, m.grad)
        };
        let (x, failed) = dynare_solve_csolve(&mut f, &x0, solver);
        if failed {
            res.info = 304;
            res.info4 = 1000.0;
            return res;
        }
        for (k, &j) in inan.iter().enumerate() {
            res.filtered_errs[(t, j)] = x[k];
        }
        let m = match_function(pm, &o, &x, &vo, &cur);
        for (k, &j) in inan.iter().enumerate() {
            res.resids[(t, j)] = m.resids[k];
        }
        if m.state.is_empty() {
            // MATLAB errors out here (state_out unassigned); treat as a failed inversion
            res.info = 304;
            res.info4 = 1000.0;
            return res;
        }
        for j in 0..nv {
            res.stateval[(t, j)] = m.state[j];
        }
        let mut e = Mat::zeros(nv, pp);
        for (k, &j) in inan.iter().enumerate() {
            e.set_column(j, &m.e.column(k));
        }
        res.emat.push(e);
        init_val = m.state.clone();
        res.regime_history.push(m.regime.unwrap());
        let maxe = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if maxe > 1e8 {
            res.info = 306;
            res.info4 = maxe / 1000.0;
            return res;
        }
        let maxr = (0..pp).fold(0.0f64, |a, j| a.max(res.resids[(t, j)].abs()));
        if maxr > 0.001 {
            res.info = 303;
            res.info4 = maxr * 100.0;
            return res;
        }
    }
    res
}

/// The likelihood part of `IVF_posterior.m`: `0.5 * sum(likei(presample+1:end))`, or the
/// failure code (301 inf, 302 NaN, 303 residuals). `cov`: `Sigma_e(err_index, err_index)`.
pub fn ivf_likelihood(
    core: &IvfCore,
    obs: &Mat,
    varobs: &[usize],
    cov: &Mat,
    presample: usize,
) -> Result<(f64, Vec<f64>), (i32, f64)> {
    let (nobs, pp) = (obs.nrows(), obs.ncols());
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let any_nan = obs.iter().any(|v| v.is_nan());
    let mut likei = vec![f64::NAN; nobs];
    let quad = |e: &[f64], c: &Mat| {
        // e / C * e'
        let ct = c.transpose();
        let sol = linalg::solve(&ct, &Mat::from_column_slice(e.len(), 1, e))
            .unwrap_or_else(|| Mat::from_element(e.len(), 1, f64::INFINITY));
        e.iter().zip(sol.iter()).map(|(a, b)| a * b).sum::<f64>()
    };
    let ldet = |m: &Mat| det(m);
    if !any_nan {
        let log_det_cov = ldet(cov).ln();
        for t in 0..nobs {
            let g = Mat::from_fn(pp, pp, |i, j| core.emat[t][(varobs[i], j)]);
            let ldj = log_det_cov + 2.0 * ldet(&g).abs().ln();
            let e: Vec<f64> = (0..pp).map(|j| core.filtered_errs[(t, j)]).collect();
            likei[t] = ldj + quad(&e, cov);
        }
        for l in likei.iter_mut() {
            *l += pp as f64 * ln2pi;
        }
    } else {
        for t in 0..nobs {
            let inan: Vec<usize> = (0..pp).filter(|&j| !obs[(t, j)].is_nan()).collect();
            let k = inan.len();
            let c = Mat::from_fn(k, k, |i, j| cov[(inan[i], inan[j])]);
            let g = Mat::from_fn(k, k, |i, j| core.emat[t][(varobs[inan[i]], inan[j])]);
            let ldj = ldet(&c).ln() + 2.0 * ldet(&g).abs().ln();
            let e: Vec<f64> = inan.iter().map(|&j| core.filtered_errs[(t, j)]).collect();
            likei[t] = ldj + quad(&e, &c) + k as f64 * ln2pi;
        }
    }
    let like = 0.5 * likei[presample.min(nobs)..].iter().sum::<f64>();
    if like.is_infinite() {
        return Err((301, 1000.0));
    }
    if like.is_nan() {
        return Err((302, 1000.0));
    }
    let maxresid = core.resids.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    if maxresid > 1e-3 {
        return Err((303, core.resids.iter().map(|v| v * v).sum()));
    }
    Ok((like, likei.iter().map(|l| 0.5 * l).collect()))
}

/// MATLAB `det` (LU).
fn det(a: &Mat) -> f64 {
    let n = a.nrows();
    if n == 0 {
        return 1.0;
    }
    let (f, ipiv, _) = linalg::lu(a);
    let mut d = 1.0;
    for i in 0..n {
        d *= f[(i, i)];
        if ipiv[i] as usize != i + 1 {
            d = -d;
        }
    }
    d
}
