// SPDX-License-Identifier: GPL-3.0-or-later
//! Optimal simple rules: port of `+osr/run.m`, `+osr/objective.m` and
//! `+osr/get_variance_of_endogenous_variables.m` (order-1 `optim_weights` loss, or minus the
//! unconditional `planner_objective` welfare at order 1, order 2 or in the linear-quadratic
//! case), with `osr_params_bounds`.

use anyhow::{Result, bail};

use super::{Resolver, welfare};
use crate::estimation::{Bounds, Evaluation, Objective, OptimResult};
use crate::linalg::Mat;
use crate::model::Model;
use crate::model::higher::DynamicDerivatives;

/// The branch of `evaluate_planner_objective.m` that `osr.objective` reaches: `resol` solves at
/// `options_.order`, and no `stoch_simul` resets the order when the Hessian is zero.
pub enum WelfareKind {
    /// `options_.order = 1`.
    Order1,
    /// `options_.order = 2`, zero model Hessian.
    LinearQuadratic,
    /// `options_.order = 2` (second-order decision rule from these derivatives).
    Order2(DynamicDerivatives),
}

/// The OSR loss as a function of the `osr_params`.
pub struct OsrObjective<'a> {
    pub resolver: &'a Resolver<'a>,
    pub base_params: Vec<f64>,
    pub i_params: Vec<usize>,
    /// `optim_weights` loss: DR positions `inv_order_var(i_var)` and `W(i_var, i_var)`;
    /// `None` = minus the unconditional planner welfare.
    pub weights: Option<(Vec<usize>, Mat)>,
    pub sigma_e: Mat,
    /// Which `evaluate_planner_objective` branch the welfare loss takes.
    pub welfare: WelfareKind,
}

impl OsrObjective<'_> {
    /// `osr.objective`: `(loss, info, info(4))`.
    pub fn loss(&self, x: &[f64]) -> (f64, i32, f64) {
        let mut p = self.base_params.clone();
        for (&i, &v) in self.i_params.iter().zip(x) {
            p[i] = v;
        }
        let res = match &self.welfare {
            WelfareKind::Order2(d) if self.weights.is_none() => {
                self.resolver.resolve_k(&p, 2, d, &self.sigma_e).map(|(k, p)| (k.first.clone(), Some(k), p))
            }
            _ => self.resolver.resolve_params(&p).map(|(d, p)| (d, None, p)),
        };
        let (dr, kdr, p) = match res {
            Ok(x) => x,
            Err(e) => {
                let i4 = if [3, 4, 5, 6, 19, 20, 21, 23, 26, 81, 84, 85].contains(&e.info) && e.info2.is_finite() {
                    e.info2
                } else {
                    0.1
                };
                return (1e8, e.info, i4);
            }
        };
        let loss = match &self.weights {
            Some((rows, w)) => {
                let vx = welfare::variance(&dr, rows, &self.sigma_e, self.resolver.qz_criterium);
                w.iter().zip(vx.iter()).map(|(a, b)| a * b).sum::<f64>()
            }
            None => {
                let m = self.resolver.model;
                let qz = self.resolver.qz_criterium;
                let v = match (&self.welfare, &kdr) {
                    (WelfareKind::Order2(_), Some(k)) => {
                        welfare::evaluate_planner_objective_order2(m, &p, k, &self.sigma_e, qz)
                    }
                    (WelfareKind::LinearQuadratic, _) => {
                        welfare::evaluate_planner_objective(m, &p, &dr, &self.sigma_e, true, qz)
                    }
                    _ => welfare::evaluate_planner_objective(m, &p, &dr, &self.sigma_e, false, qz),
                };
                v.map(|v| -v.unconditional).unwrap_or(f64::NAN)
            }
        };
        if loss.is_infinite() {
            return (1e8, 174, 0.1);
        }
        if loss.is_nan() {
            return (1e8, 175, 0.1);
        }
        (loss, 0, 0.0)
    }
}

impl Objective for OsrObjective<'_> {
    fn dim(&self) -> usize {
        self.i_params.len()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let (l, i, _) = self.loss(x);
        (l, i)
    }
    fn eval_full(&self, x: &[f64], _contributions: bool) -> Evaluation {
        let (fval, info, info4) = self.loss(x);
        Evaluation { fval, info, info4, contributions: None }
    }
}

/// Builds `W(i_var, i_var)` from the `optim_weights` entries (declaration-order indices):
/// returns the sorted unique `i_var` and the weight block.
pub fn weight_block(model: &Model, entries: &[(usize, usize, f64)]) -> (Vec<usize>, Mat) {
    let n = model.n_endo();
    let mut w = Mat::zeros(n, n);
    let mut iv = Vec::new();
    for &(i, j, v) in entries {
        w[(i, j)] = v;
        iv.push(i);
        iv.push(j);
    }
    iv.sort_unstable();
    iv.dedup();
    let blk = Mat::from_fn(iv.len(), iv.len(), |a, b| w[(iv[a], iv[b])]);
    (iv, blk)
}

/// Runs the optimizer of `opt_algo` through `dynare_minimize_objective.m`
/// ([`crate::estimation::optim::minimize`], stream O), as `+osr/run.m` does: `bounds` =
/// `M_.osr.param_bounds` (per osr parameter; unbounded without `osr_params_bounds`), no prior
/// standard deviations, `optim` = `options_.optim_opt`.
#[allow(clippy::too_many_arguments)]
pub fn minimize(
    obj: &OsrObjective,
    x0: &[f64],
    opt_algo: i32,
    bounds: Option<&[(f64, f64)]>,
    names: &[String],
    optim: Option<&crate::modfile::OptValue>,
    verbose: bool,
    seed: u64,
) -> Result<OptimResult> {
    if [5, 6, 10, 11].contains(&opt_algo) {
        bail!("OSR: OSR does not support opt_algo={opt_algo}.");
    }
    if bounds.is_some() && ![1, 2, 5, 9].contains(&opt_algo) {
        bail!("OSR: OSR with bounds on parameters requires a constrained optimizer, i.e. opt_algo= 1,2 or 9.");
    }
    let n = x0.len();
    let b = match bounds {
        Some(b) => Bounds { lb: b.iter().map(|v| v.0).collect(), ub: b.iter().map(|v| v.1).collect() },
        None => Bounds::unbounded(n),
    };
    let prior_sd = vec![f64::NAN; n];
    let ctx = crate::estimation::optim::MinimizeContext {
        bounds: &b,
        names,
        prior_sd: &prior_sd,
        optim,
        verbose,
        seed,
        stream: u64::MAX - 200,
    };
    crate::estimation::optim::minimize(opt_algo, obj, x0, &ctx)
}
