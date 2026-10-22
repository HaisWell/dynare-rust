// SPDX-License-Identifier: GPL-3.0-or-later
//! The Hessian at the mode, as `dynare_estimation_1.m` computes it after each `mode_compute`
//! other than 6 (estimation completion):
//!
//! * `cova_compute = 0`: none (the optimizer's own `hessian_mat` is kept for the mode file of
//!   `mode_compute = 5`, and nothing else uses it);
//! * `analytic_derivation` with `dsge_likelihood`: the exact analytic Hessian
//!   (`analytic_Hessian = 'full'`, [`crate::estimation::analytic`]);
//! * `mode_compute = 5` with `optim = ('Hessian', 0 | 2)` and `dsge_likelihood`: `mr_hessian.m`
//!   on the univariate filter (`kalman_algo = 2`, or 4 with `lik_init = 3`), i.e. the outer
//!   product of the per-observable scores (flag 0), with numerical second derivatives on the
//!   diagonal (flag 2, `mr_hessian(..., newratflag > 0)`); with the univariate filter already
//!   selected and flag 0, newrat's own outer-product Hessian;
//! * otherwise `hessian.m` on the objective, or on the penalized objective
//!   (`penalty_objective_function.m` with the mode's value as base penalty) with
//!   `use_penalized_objective_for_hessian`.

use anyhow::Result;

use crate::dsge::DsgePosterior;
use crate::estimation::analytic::{AnalyticHessian, AnalyticPosterior};
use crate::estimation::optim::{HessInfo, mr_hessian};
use crate::estimation::{Bounds, Evaluation, Objective, hessian, penalty_objective};
use crate::linalg::Mat;

/// `penalty_objective_function(x, fcn, base_penalty)` as an objective (`hessian.m` on the
/// penalized objective).
pub struct Penalized<'a> {
    pub obj: &'a dyn Objective,
    pub base_penalty: f64,
}

impl Objective for Penalized<'_> {
    fn dim(&self) -> usize {
        self.obj.dim()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        (penalty_objective(self.obj, x, self.base_penalty, false).0, 0)
    }
    fn eval_full(&self, x: &[f64], _contributions: bool) -> Evaluation {
        Evaluation { fval: self.eval(x).0, info: 0, info4: 0.0, contributions: None }
    }
}

/// What the Hessian at the mode depends on.
pub struct ModeHessian<'a> {
    /// `objective_function`.
    pub obj: &'a dyn Objective,
    /// The linear `dsge_likelihood` posterior when it is the objective (None for
    /// dsge_var_likelihood / OccBin objectives).
    pub dsge_likelihood: Option<&'a DsgePosterior<'a>>,
    /// `analytic_derivation`.
    pub analytic: Option<&'a AnalyticPosterior<'a>>,
    pub cova_compute: bool,
    /// `use_penalized_objective_for_hessian`.
    pub penalized: bool,
    /// `options_.newrat.hess` / `optim = ('Hessian', k)` of mode_compute 5 (`newratflag`).
    pub newratflag: i32,
    pub bounds: &'a Bounds,
    /// `bayestopt_.p2`.
    pub prior_sd: &'a [f64],
}

/// `optim = ('Hessian', k)` of `mode_compute = 5` (default `options_.newrat.hess = 1`; 0 with
/// `analytic_derivation`, whose flag must be 0, see `dynare_minimize_objective.m`).
pub fn newratflag(optim: Option<&crate::modfile::OptValue>, analytic: bool) -> Result<i32> {
    let mut flag = if analytic { 0 } else { 1 };
    for (k, v) in crate::estimation::optim::minimize::key_values(optim, "newrat")? {
        if k == "Hessian" {
            flag = crate::estimation::optim::minimize::kv_num(&k, &v)? as i32;
        }
    }
    Ok(flag)
}

impl ModeHessian<'_> {
    /// The Hessian after `optimizer` (`optimizer_hh`: its `hessian_mat`), at the mode `x` with
    /// value `fval`; None when `cova_compute = 0` and the optimizer gave none.
    pub fn compute(&self, optimizer: i32, x: &[f64], fval: f64, optimizer_hh: Option<Mat>) -> Result<Option<Mat>> {
        if !self.cova_compute {
            return Ok(optimizer_hh);
        }
        let is_dsge = self.dsge_likelihood.is_some();
        if let (Some(ap), true) = (self.analytic, is_dsge) {
            let g = ap.eval_grad(x, AnalyticHessian::Full).expect("analytic objective");
            return match g.hess {
                Some(h) if g.info == 0 => Ok(Some(h)),
                _ => anyhow::bail!(
                    "analytic_derivation: the Hessian at the mode could not be computed (info {})",
                    g.info
                ),
            };
        }
        if !(optimizer == 5 && self.newratflag != 1 && is_dsge) {
            return Ok(Some(if self.penalized {
                hessian(&Penalized { obj: self.obj, base_penalty: fval }, x, [1e-2, 1.0])
            } else {
                hessian(self.obj, x, [1e-2, 1.0])
            }));
        }
        // mode_compute = 5 with newratflag 0 or 2: mr_hessian on the univariate filter
        let lin = self.dsge_likelihood.unwrap();
        let ka0 = lin.data.kalman.kalman_algo;
        let mut data = lin.data.clone();
        if !(ka0 == 2 || ka0 == 4) {
            data.kalman.kalman_algo = if lin.data.kalman.lik_init == 3 { 4 } else { 2 };
        } else if self.newratflag == 0 {
            // hh already contains the outer product gradient with univariate filter
            return Ok(optimizer_hh);
        }
        data.kalman.obs_contributions = true;
        let post = DsgePosterior { dsge: lin.dsge, data: &data, bounds: lin.bounds };
        let n = x.len();
        let crit = 1e-5; // options_.newrat.tolerance.f
        let hess_info = crate::estimation::optim::take_last_hess_info().unwrap_or(HessInfo {
            gstep: [1e-2, 1.0],
            htol: crit,
            htol_rel: f64::NAN,
            h1: vec![1e-6; n],
            robust: false,
        });
        let mr =
            mr_hessian(&post, x, fval, (self.newratflag > 0) as u8, crit, &hess_info, self.bounds, self.prior_sd, true);
        Ok(mr.hessian_mat.map(|h| Mat::from_column_slice(n, n, h.as_slice())))
    }
}
