// SPDX-License-Identifier: GPL-3.0-or-later
//! Substitute for `mode_compute = 102` (MATLAB's `simulannealbnd`, Global
//! Optimization Toolbox, which this port cannot call). It follows the documented default
//! algorithm of `simulannealbnd` with Dynare's options (`optimoptions('simulannealbnd',
//! 'display','iter','TolFun',1e-8)`):
//!
//! * `annealingfast`: trial point = x + T .* u, u a uniformly random unit direction; a
//!   coordinate outside the bounds is redrawn uniformly between the bound and x;
//! * `temperatureexp`: T_i = T0 * 0.95^k_i, k_i incremented every iteration (T0 = 100);
//! * `acceptancesa`: a worse point is accepted with probability 1/(1 + exp(delta/max(T)));
//! * reannealing every `ReannealInterval` (100) accepted points: k_i =
//!   log((T0 / T_i) * max_j(s_j) / s_i) from finite-difference gradients s (raises T);
//! * stops when the average change of the objective over `StallIterLimit` (500 n) iterations
//!   is below `TolFun`, or after `MaxFunEvals` (3000 n) evaluations or `MaxIter`.
//!
//! The random numbers come from ChaCha8 (`--seed`), so iterates differ from MATLAB's.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::OptimResult;
use super::minimize::kv_num;
use crate::estimation::{Bounds, Objective};
use crate::modfile::OptValue;

#[derive(Clone, Debug)]
pub struct SimulannealbndOptions {
    pub initial_temperature: f64,
    pub reanneal_interval: usize,
    pub tol_fun: f64,
    /// `None` = 500 * nvars.
    pub stall_iter_limit: Option<usize>,
    /// `None` = 3000 * nvars.
    pub max_fun_evals: Option<usize>,
    pub max_iter: usize,
    pub verbose: bool,
    pub seed: u64,
    pub stream: u64,
}

impl Default for SimulannealbndOptions {
    fn default() -> Self {
        SimulannealbndOptions {
            initial_temperature: 100.0,
            reanneal_interval: 100,
            tol_fun: 1e-8,
            stall_iter_limit: None,
            max_fun_evals: None,
            max_iter: usize::MAX,
            verbose: true,
            seed: 0,
            stream: 0,
        }
    }
}

impl SimulannealbndOptions {
    pub fn apply_optim(&mut self, kv: &[(String, OptValue)]) -> Result<()> {
        for (k, v) in kv {
            match k.as_str() {
                "TolFun" | "FunctionTolerance" => self.tol_fun = kv_num(k, v)?,
                "MaxFunEvals" | "MaxFunctionEvaluations" => self.max_fun_evals = Some(kv_num(k, v)? as usize),
                "MaxIter" | "MaxIterations" => self.max_iter = kv_num(k, v)? as usize,
                "InitialTemperature" => self.initial_temperature = kv_num(k, v)?,
                "ReannealInterval" => self.reanneal_interval = kv_num(k, v)? as usize,
                "StallIterLimit" | "MaxStallIterations" => self.stall_iter_limit = Some(kv_num(k, v)? as usize),
                "Display" | "display" => {
                    if let OptValue::Str(s) = v {
                        self.verbose = !s.trim_matches(|c| c == '\'' || c == '"').eq_ignore_ascii_case("off");
                    }
                }
                other => println!("WARNING: simulannealbnd: option {other} is ignored."),
            }
        }
        Ok(())
    }
}

/// Minimizes `obj` within `bounds`.
pub fn simulannealbnd(
    obj: &dyn Objective,
    x0: &[f64],
    bounds: &Bounds,
    o: &SimulannealbndOptions,
) -> Result<OptimResult> {
    let n = x0.len();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(o.seed);
    rng.set_stream(o.stream);
    let (lb, ub) = (&bounds.lb, &bounds.ub);
    // dynare_minimize_objective.m passes the raw objective (Inf on failure: never accepted)
    let mut base_penalty = 0.0;
    let f = |x: &[f64], _bp: &mut f64| -> f64 { obj.eval(x).0 };
    let mut x = x0.to_vec();
    let mut fx = f(&x, &mut base_penalty);
    let (mut xbest, mut fbest) = (x.clone(), fx);
    let t0 = vec![o.initial_temperature; n];
    let mut k = vec![0.0f64; n];
    let mut temp = t0.clone();
    let stall = o.stall_iter_limit.unwrap_or(500 * n).max(1);
    let max_evals = o.max_fun_evals.unwrap_or(3000 * n);
    let mut evals = 1usize;
    let mut iter = 0usize;
    let mut accepted = 0usize;
    let mut history: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
    let mut message = String::from("Maximum number of function evaluations exceeded");
    if o.verbose {
        println!("\n                           Best        Current           Mean");
        println!("Iteration   f-count         f(x)         f(x)    temperature");
        println!("{:5}    {:7}    {:12.6}   {:12.6}   {:12.4}", 0, evals, fbest, fx, 100.0);
    }
    while evals < max_evals && iter < o.max_iter {
        iter += 1;
        // annealingfast + sahonorbounds
        let dir: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
        let norm = dir.iter().map(|v: &f64| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        let mut y: Vec<f64> = (0..n).map(|i| x[i] + temp[i] * dir[i] / norm).collect();
        for i in 0..n {
            if y[i] < lb[i] {
                y[i] = lb[i] + rng.r#gen::<f64>() * (x[i] - lb[i]);
            } else if y[i] > ub[i] {
                y[i] = ub[i] - rng.r#gen::<f64>() * (ub[i] - x[i]);
            }
        }
        let fy = f(&y, &mut base_penalty);
        evals += 1;
        // acceptancesa
        let delta = fy - fx;
        let tmax = temp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let accept = delta < 0.0 || rng.r#gen::<f64>() < 1.0 / (1.0 + (delta / tmax).exp());
        let fprev = fx;
        if accept {
            x = y;
            fx = fy;
            accepted += 1;
            if fx < fbest {
                xbest.clone_from(&x);
                fbest = fx;
            }
        }
        history.push_back((fx - fprev).abs());
        if history.len() > stall {
            history.pop_front();
        }
        // temperatureexp
        for i in 0..n {
            k[i] += 1.0;
            temp[i] = t0[i] * 0.95f64.powf(k[i]);
        }
        // reannealing
        if accept && o.reanneal_interval > 0 && accepted.is_multiple_of(o.reanneal_interval) {
            let s: Vec<f64> = (0..n)
                .map(|i| {
                    let h = 1e-6 * x[i].abs().max(1.0);
                    let mut xp = x.clone();
                    xp[i] = (x[i] + h).min(ub[i]);
                    let fp = f(&xp, &mut base_penalty);
                    evals += 1;
                    ((fp - fx) / (xp[i] - x[i]).max(f64::MIN_POSITIVE)).abs()
                })
                .collect();
            let smax = s.iter().copied().fold(0.0, f64::max);
            for i in 0..n {
                if s[i] > 0.0 && smax.is_finite() {
                    k[i] = ((t0[i] / temp[i]) * smax / s[i]).ln().max(0.0);
                    temp[i] = t0[i] * 0.95f64.powf(k[i]);
                }
            }
        }
        if o.verbose && iter.is_multiple_of(10 * n.max(1)) {
            let tm = temp.iter().sum::<f64>() / n as f64;
            println!("{iter:5}    {evals:7}    {fbest:12.6}   {fx:12.6}   {tm:12.4}");
        }
        if history.len() == stall && history.iter().sum::<f64>() / (stall as f64) < o.tol_fun {
            message =
                "Optimization terminated: change in best function value less than options.FunctionTolerance".into();
            break;
        }
    }
    if iter >= o.max_iter {
        message = "Maximum number of iterations exceeded".into();
    }
    if o.verbose {
        println!("{message}.");
    }
    Ok(OptimResult {
        x: xbest,
        fval: fbest,
        hessian: None,
        inv_hessian: None,
        iterations: iter,
        func_count: evals,
        exitflag: 1,
        message,
    })
}
