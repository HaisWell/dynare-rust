// SPDX-License-Identifier: GPL-3.0-or-later
//! Substitute for `mode_compute = 12` (MATLAB's `particleswarm`, Global
//! Optimization Toolbox). **Not MATLAB's code**: an implementation of the
//! algorithm as MathWorks documents it ("Particle Swarm Optimization
//! Algorithm": random neighbourhoods of adaptive size, adaptive inertia,
//! self/social adjustment weights 1.49, velocities clamped at the bounds),
//! with Dynare's `options_.particleswarm` defaults and the swarm
//! initialisation of `dynare_minimize_objective.m` (particle 1 at the start,
//! the others uniform within 10% of the range, or 20% of |x0| for unbounded
//! parameters, redrawn until the objective is finite) and its penalised
//! objective (base penalty = worst initial value).
//!
//! The swarm's evaluations of one iteration run in parallel (rayon) after all
//! random numbers of the iteration are drawn: the run does not depend on the
//! number of threads.

use anyhow::{Result, bail};
use rand::Rng;
use rayon::prelude::*;

use super::OptimResult;
use super::minimize::{kv_num, kv_str};
use crate::estimation::mcmc::rng_stream;
use crate::estimation::{Bounds, Objective, penalty_objective};
use crate::modfile::OptValue;

/// `options_.particleswarm` (`default_option_values.m`) and MATLAB's other defaults.
#[derive(Clone, Debug)]
pub struct ParticleswarmOptions {
    /// `SwarmSize` (MATLAB: `min(100, 10*nvars)`).
    pub swarm_size: Option<usize>,
    /// `FunctionTolerance` (1e-6).
    pub function_tolerance: f64,
    /// `InertiaRange` ([0.1, 1.1]).
    pub inertia_range: [f64; 2],
    /// `MaxIterations` (Dynare: 100000).
    pub max_iterations: usize,
    /// `MaxStallIterations` (20).
    pub max_stall_iterations: usize,
    /// `MaxTime` (inf), seconds.
    pub max_time: f64,
    /// `MinNeighborsFraction` (0.25).
    pub min_neighbors_fraction: f64,
    /// `ObjectiveLimit` (-inf).
    pub objective_limit: f64,
    /// `SelfAdjustmentWeight` (1.49).
    pub self_adjustment: f64,
    /// `SocialAdjustmentWeight` (1.49).
    pub social_adjustment: f64,
    /// `InitialSwarmSpan` (2000): range of the initial velocities of unbounded parameters.
    pub initial_swarm_span: f64,
    /// `Display` ('iter').
    pub verbose: bool,
    /// `DisplayInterval` (1).
    pub display_interval: usize,
    pub seed: u64,
    pub stream: u64,
    pub parallel: bool,
}

impl Default for ParticleswarmOptions {
    fn default() -> Self {
        ParticleswarmOptions {
            swarm_size: None,
            function_tolerance: 1e-6,
            inertia_range: [0.1, 1.1],
            max_iterations: 100_000,
            max_stall_iterations: 20,
            max_time: f64::INFINITY,
            min_neighbors_fraction: 0.25,
            objective_limit: f64::NEG_INFINITY,
            self_adjustment: 1.49,
            social_adjustment: 1.49,
            initial_swarm_span: 2000.0,
            verbose: true,
            display_interval: 1,
            seed: 0,
            stream: 12,
            parallel: true,
        }
    }
}

impl ParticleswarmOptions {
    /// The `SupportedListOfOptions` of `dynare_minimize_objective.m`, case 12.
    pub fn apply_optim(&mut self, kv: &[(String, OptValue)]) -> Result<()> {
        for (k, v) in kv {
            match k.as_str() {
                "SwarmSize" => self.swarm_size = Some(kv_num(k, v)? as usize),
                "FunctionTolerance" => self.function_tolerance = kv_num(k, v)?,
                "InertiaRange" => match v {
                    OptValue::Vec(r) if r.len() == 2 => self.inertia_range = [r[0], r[1]],
                    _ => bail!("particleswarm: InertiaRange must be a two-element vector"),
                },
                "MaxIterations" => self.max_iterations = kv_num(k, v)? as usize,
                "MaxStallIterations" => self.max_stall_iterations = kv_num(k, v)? as usize,
                "MaxTime" => self.max_time = kv_num(k, v)?,
                "MinNeighborsFraction" => self.min_neighbors_fraction = kv_num(k, v)?,
                "ObjectiveLimit" => self.objective_limit = kv_num(k, v)?,
                "SelfAdjustmentWeight" => self.self_adjustment = kv_num(k, v)?,
                "SocialAdjustmentWeight" => self.social_adjustment = kv_num(k, v)?,
                "InitialSwarmSpan" => self.initial_swarm_span = kv_num(k, v)?,
                "Display" => self.verbose = kv_str(k, v)? != "off",
                "DisplayInterval" => self.display_interval = (kv_num(k, v)? as usize).max(1),
                "UseParallel" | "UseVectorized" | "FunValCheck" | "HybridFcn" | "OutputFcn" | "PlotFcn"
                | "CreationFcn" | "MaxStallTime" | "InitialSwarmMatrix" => {
                    println!("WARNING: particleswarm: option {k} is ignored by dynare-rust's substitute.")
                }
                other => println!("WARNING: particleswarm: Unknown option ({other})!"),
            }
        }
        Ok(())
    }
}

/// `mode_compute = 12`. `bounds` are Dynare's `[bounds.lb bounds.ub]` (may be infinite).
pub fn particleswarm(
    obj: &dyn Objective,
    x0: &[f64],
    bounds: &Bounds,
    opts: &ParticleswarmOptions,
) -> Result<OptimResult> {
    let n = x0.len();
    let (lb, ub) = (&bounds.lb, &bounds.ub);
    let s = opts.swarm_size.unwrap_or((10 * n).min(100)).max(2);
    let mut rng = rng_stream(opts.seed, opts.stream);
    let t0 = std::time::Instant::now();
    // Swarm initialisation of dynare_minimize_objective.m (raw objective, exit_flag).
    if opts.verbose {
        print!("particleswarm: inititalizing swarm matrix...");
    }
    let mut pos: Vec<Vec<f64>> = Vec::with_capacity(s);
    let mut fvals: Vec<f64> = Vec::with_capacity(s);
    let mut tries = 0usize;
    while pos.len() < s {
        let cand: Vec<f64> = if pos.is_empty() {
            x0.to_vec()
        } else {
            (0..n)
                .map(|i| {
                    let range = ub[i] - lb[i];
                    let delta = if range.is_finite() {
                        0.1 * range
                    } else if x0[i] != 0.0 {
                        0.2 * x0[i].abs()
                    } else {
                        0.1
                    };
                    (x0[i] + (2.0 * rng.r#gen::<f64>() - 1.0) * delta).max(lb[i]).min(ub[i])
                })
                .collect()
        };
        let (f, info) = obj.eval(&cand);
        if info == 0 && f.is_finite() {
            pos.push(cand);
            fvals.push(f);
        } else if pos.is_empty() {
            bail!("particleswarm: the objective is not finite at the initial point");
        }
        tries += 1;
        if tries > 1000 * s {
            bail!("particleswarm: cannot find {s} initial particles with a finite objective");
        }
    }
    if opts.verbose {
        println!("done!");
        println!(
            "NOTE: mode_compute=12 runs dynare-rust's particle swarm, not MATLAB's particleswarm (Global Optimization Toolbox)."
        );
    }
    let penalty = fvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pen = |x: &[f64]| penalty_objective(obj, x, penalty, false).0;
    let mut funccount = 0usize;
    // Initial velocities: uniform in [-r, r], r = ub - lb (InitialSwarmSpan when unbounded).
    let span: Vec<f64> = (0..n)
        .map(|i| {
            let r = ub[i] - lb[i];
            if r.is_finite() { r } else { opts.initial_swarm_span }
        })
        .collect();
    let mut vel: Vec<Vec<f64>> =
        (0..s).map(|_| (0..n).map(|i| span[i] * (2.0 * rng.r#gen::<f64>() - 1.0)).collect()).collect();
    // The initial swarm is scored with the penalised objective (= fvals where finite).
    let mut fx: Vec<f64> = fvals.clone();
    let mut pbest = pos.clone();
    let mut pbest_f = fx.clone();
    let best_of = |f: &[f64]| (0..f.len()).min_by(|&a, &b| f[a].total_cmp(&f[b])).unwrap();
    let mut ib = best_of(&pbest_f);
    let mut best_f = pbest_f[ib];
    let mut best_x = pbest[ib].clone();
    let min_nbr = ((s as f64 * opts.min_neighbors_fraction).floor() as usize).max(2).min(s - 1).max(1);
    let mut nbr = min_nbr;
    let mut inertia = opts.inertia_range[1];
    let mut stall = 0usize;
    // Best value at each iteration, for the stall test.
    let mut history: Vec<f64> = vec![best_f];
    let mut iter = 0usize;
    let (exitflag, message);
    if opts.verbose {
        println!("\n                                 Best            Mean     Stall");
        println!("Iteration     f-count            f(x)            f(x)    Iterations");
        let mean = fx.iter().sum::<f64>() / s as f64;
        println!("{:>5}{:>16}{:>16.6e}{:>16.6e}{:>10}", 0, s, best_f, mean, 0);
    }
    loop {
        if best_f <= opts.objective_limit {
            exitflag = -3;
            message = "Optimization ended: objective value below ObjectiveLimit.".to_string();
            break;
        }
        if iter >= opts.max_iterations {
            exitflag = 0;
            message = "Optimization ended: number of iterations exceeded OPTIONS.MaxIterations.".to_string();
            break;
        }
        if t0.elapsed().as_secs_f64() > opts.max_time {
            exitflag = -5;
            message = "Optimization ended: time exceeded OPTIONS.MaxTime.".to_string();
            break;
        }
        let k = opts.max_stall_iterations;
        if history.len() > k {
            let old = history[history.len() - 1 - k];
            let change = (old - best_f).abs() / old.abs().max(1.0);
            if change < opts.function_tolerance {
                exitflag = 1;
                message = "Optimization ended: relative change in the objective value over the last OPTIONS.MaxStallIterations iterations is less than OPTIONS.FunctionTolerance.".to_string();
                break;
            }
        }
        iter += 1;
        // Draw everything first (sequential stream), then evaluate in parallel.
        for i in 0..s {
            let mut others: Vec<usize> = (0..s).filter(|&j| j != i).collect();
            // Random subset of `nbr` neighbours (partial Fisher–Yates).
            for t in 0..nbr.min(others.len()) {
                let r = t + rng.gen_range(0..others.len() - t);
                others.swap(t, r);
            }
            let nb = &others[..nbr.min(others.len())];
            let g = *nb.iter().min_by(|&&a, &&b| pbest_f[a].total_cmp(&pbest_f[b])).unwrap();
            let g = if pbest_f[g] < pbest_f[i] { g } else { i };
            for d in 0..n {
                let u1: f64 = rng.r#gen();
                let u2: f64 = rng.r#gen();
                let v = inertia * vel[i][d]
                    + opts.self_adjustment * u1 * (pbest[i][d] - pos[i][d])
                    + opts.social_adjustment * u2 * (pbest[g][d] - pos[i][d]);
                let mut xn = pos[i][d] + v;
                let mut vn = v;
                if xn < lb[d] {
                    xn = lb[d];
                    vn = 0.0;
                } else if xn > ub[d] {
                    xn = ub[d];
                    vn = 0.0;
                }
                vel[i][d] = vn;
                pos[i][d] = xn;
            }
        }
        fx =
            if opts.parallel { pos.par_iter().map(|p| pen(p)).collect() } else { pos.iter().map(|p| pen(p)).collect() };
        funccount += s;
        for i in 0..s {
            if fx[i] < pbest_f[i] {
                pbest_f[i] = fx[i];
                pbest[i].copy_from_slice(&pos[i]);
            }
        }
        ib = best_of(&pbest_f);
        if pbest_f[ib] < best_f {
            best_f = pbest_f[ib];
            best_x = pbest[ib].clone();
            stall = stall.saturating_sub(1);
            nbr = min_nbr;
        } else {
            stall += 1;
            nbr = (nbr + min_nbr).min(s - 1);
        }
        // Adaptive inertia, updated every iteration from the stall counter.
        if stall < 2 {
            inertia *= 2.0;
        } else if stall > 5 {
            inertia /= 2.0;
        }
        inertia = inertia.clamp(opts.inertia_range[0], opts.inertia_range[1]);
        history.push(best_f);
        if opts.verbose && iter.is_multiple_of(opts.display_interval) {
            let fin: Vec<f64> = fx.iter().cloned().filter(|v| v.is_finite()).collect();
            let mean = fin.iter().sum::<f64>() / fin.len().max(1) as f64;
            println!("{:>5}{:>16}{:>16.6e}{:>16.6e}{:>10}", iter, s + funccount, best_f, mean, stall);
        }
    }
    if opts.verbose {
        println!("{message}");
    }
    Ok(OptimResult {
        x: best_x,
        fval: best_f,
        hessian: None,
        inv_hessian: None,
        iterations: iter,
        func_count: funccount + tries,
        exitflag,
        message,
    })
}
