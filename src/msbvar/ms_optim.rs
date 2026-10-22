// SPDX-License-Identifier: GPL-3.0-or-later
//! The posterior-mode search of Dynare's `ms_estimation` (the `-estimate` branch of the
//! `ms_sbvar_command_line` mex, whose C sources are not shipped; the structure below was read off the
//! `est_final_*.out` / `est_intermediate_*.out` files and the console output of Dynare 7.2).
//!
//! * `find_mode` = one Dynare "run" sequence: block iterations alternate pass 0 (the VAR parameters
//!   `b, g / lambda, psi, zeta` given `Q`: quasi-Newton with a numerical gradient, the role of csminwel, at most
//!   the current iteration cap) and pass 1 (the transition-matrix parameters given the VAR parameters: EM steps,
//!   closed form for the Dirichlet design; the order and the values after each pass were checked against
//!   `est_intermediate_*.out`). Block iterations stop when the posterior changes by less than the criterion
//!   (`convergence_starting_value`, at most `max_block_iterations` iterations); the criterion is then
//!   multiplied by `convergence_increment_value` and the iteration cap of pass 1 by
//!   `max_iterations_increment_value` until the criterion is below `convergence_ending_value`. The whole
//!   sequence is repeated (`max_repeated_optimization_runs`) until the parameters move by less than
//!   `parameter_convergence_criterion` (Euclidean norm) and the posterior by less than
//!   `function_convergence_criterion`.
//! * `maximise` = initial run, then stages of `number_of_large_perturbations` restarts from posterior
//!   draws of an adaptive random-walk Metropolis chain around the current best mode (Dynare: 30000
//!   calibration draws, 10000 burn-in, 500 draws), then stages of `number_of_small_perturbations` restarts
//!   from `number_of_posterior_draws_after_perturbation` Metropolis steps away from the best mode; each kind
//!   of stage is repeated (at most `max_number_of_stages`) until the best mode moves by less than
//!   `random_parameter_convergence_criterion` / `random_function_convergence_criterion`.
//!
//! A final joint quasi-Newton step polishes the winner when the transition probabilities are interior.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::hamilton;
use super::mdd;
use super::ms_estimation::{MsSetup, bfgs_crit, chain_alpha, chain_columns};

/// `options_.ms` optimisation settings (`initialize_ms_sbvar_options.m` defaults).
#[derive(Clone, Debug)]
pub struct OptimOptions {
    pub cb: f64,
    pub ce: f64,
    pub ci: f64,
    pub ib: usize,
    pub ii: f64,
    pub mb: usize,
    pub repeat_max: usize,
    pub repeat_tol_obj: f64,
    pub repeat_tol_parms: f64,
    pub random: usize,
    pub random_small: usize,
    pub random_small_ndraws: usize,
    pub random_max: usize,
    pub random_tol_obj: f64,
    pub random_tol_parms: f64,
}

impl Default for OptimOptions {
    fn default() -> Self {
        OptimOptions {
            cb: 1e-3,
            ce: 1e-6,
            ci: 0.1,
            ib: 50,
            ii: 2.0,
            mb: 100,
            repeat_max: 10,
            repeat_tol_obj: 0.1,
            repeat_tol_parms: 0.1,
            random: 5,
            random_small: 5,
            random_small_ndraws: 1,
            random_max: 20,
            random_tol_obj: 0.1,
            random_tol_parms: 0.1,
        }
    }
}

/// Log of the search (the `Objective function (...)` / `End ... run` lines of `est_final_*.out`).
#[derive(Default, Clone, Debug)]
pub struct Trace {
    pub lines: Vec<String>,
    pub find_mode_calls: usize,
}

fn norm_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Pass 0: EM steps for the Dirichlet parameters of every chain given the VAR parameters, until the
/// posterior improves by less than `tol` (at most 200 steps). Returns the log posterior.
pub fn em_q(s: &MsSetup, th: &mut [f64], tol: f64) -> f64 {
    let mut obj = s.log_posterior(th);
    if s.layout.q.iter().all(|v| v.is_empty()) {
        return obj;
    }
    let sizes: Vec<usize> = s.chains.iter().map(|c| c.n).collect();
    let digit = |mut i: usize, c: usize| {
        for k in (c + 1..sizes.len()).rev() {
            i /= sizes[k];
        }
        i % sizes[c]
    };
    for _ in 0..200 {
        let Some((r, f)) = s.filter(th) else { return obj };
        let sm = hamilton::smooth(&f, &r.q);
        let ns = s.nstates;
        let t_n = f.filtered.nrows() - 1;
        let mut nmat = vec![vec![0.0; ns]; ns];
        for t in 0..t_n {
            for i in 0..ns {
                let p = f.predicted[(t + 1, i)];
                if p <= 0.0 {
                    continue;
                }
                let w = sm[(t + 1, i)] / p;
                for j in 0..ns {
                    nmat[i][j] += w * r.q[(i, j)] * f.filtered[(t, j)];
                }
            }
        }
        let mut cand = th.to_vec();
        for (c, ch) in s.chains.iter().enumerate() {
            let n = ch.n;
            let mut nc = vec![vec![0.0; n]; n];
            for i in 0..ns {
                for j in 0..ns {
                    nc[digit(i, c)][digit(j, c)] += nmat[i][j];
                }
            }
            for (i, &o) in s.layout.q[c].iter().enumerate() {
                let (d1rows, d2row, nr) = chain_columns(n, i);
                let n1: f64 = d1rows.iter().take(nr).map(|&(row, _)| nc[row][i]).sum();
                let n2 = nc[d2row][i];
                let (a1, a2) = chain_alpha(ch, i);
                let (u, v) = (a1 - 1.0 + n1, a2 - 1.0 + n2);
                let d = if u <= 0.0 {
                    0.0
                } else if v <= 0.0 {
                    1.0
                } else {
                    u / (u + v)
                };
                cand[o] = d;
            }
        }
        let new = s.log_posterior(&cand);
        if !(new >= obj) {
            break;
        }
        th.copy_from_slice(&cand);
        let gain = new - obj;
        obj = new;
        if gain < tol {
            break;
        }
    }
    obj
}

/// Indices of the VAR parameters (everything but the transition-matrix block).
fn var_indices(s: &MsSetup) -> Vec<usize> {
    let mut isq = vec![false; s.layout.len];
    for v in &s.layout.q {
        for &o in v {
            isq[o] = true;
        }
    }
    (0..s.layout.len).filter(|&i| !isq[i]).collect()
}

/// Pass 1: quasi-Newton on the VAR parameters (`ln zeta` coordinates) given `Q`.
pub fn var_block(s: &MsSetup, th: &mut [f64], max_iter: usize, crit: f64) -> f64 {
    let idx = var_indices(s);
    let phi_all = s.to_phi(th);
    let base = th.to_vec();
    let embed = |x: &[f64]| {
        let mut p = phi_all.clone();
        for (k, &i) in idx.iter().enumerate() {
            p[i] = x[k];
        }
        let mut t = s.from_phi(&p);
        // the transition block stays in theta coordinates (it may sit on the boundary)
        for v in &s.layout.q {
            for &o in v {
                t[o] = base[o];
            }
        }
        t
    };
    let obj = |x: &[f64]| {
        let v = s.log_posterior(&embed(x));
        if v.is_finite() { -v } else { f64::INFINITY }
    };
    let x0: Vec<f64> = idx.iter().map(|&i| phi_all[i]).collect();
    let (x, f, _) = bfgs_crit(&obj, x0, max_iter, crit);
    if f.is_finite() && -f >= s.log_posterior(th) {
        th.copy_from_slice(&embed(&x));
    }
    s.log_posterior(th)
}

/// One sequence of block iterations over the criterion schedule.
fn block_pass(s: &MsSetup, th: &mut [f64], o: &OptimOptions) -> f64 {
    let mut obj = s.log_posterior(th);
    let mut tol = o.cb.max(1e-14);
    let mut max_it = o.ib.max(1) as f64;
    loop {
        for _ in 0..o.mb.max(1) {
            let prev = obj;
            var_block(s, th, max_it.round() as usize, tol);
            obj = em_q(s, th, tol * 0.01);
            if (obj - prev).abs() < tol {
                break;
            }
        }
        tol *= o.ci;
        max_it *= o.ii;
        if !(o.ci > 0.0 && o.ci < 1.0) || tol < o.ce * (1.0 - 1e-9) {
            break;
        }
    }
    obj
}

/// Dynare's FindMode with repeated runs.
pub fn find_mode(s: &MsSetup, start: &[f64], o: &OptimOptions, trace: &mut Trace) -> (Vec<f64>, f64) {
    trace.find_mode_calls += 1;
    let mut th = start.to_vec();
    let mut obj = s.log_posterior(&th);
    trace.lines.push(format!("Objective function (initial): {}", fmt_g(obj)));
    for run in 1..=o.repeat_max.max(1) {
        let prev_th = th.clone();
        let prev = obj;
        obj = block_pass(s, &mut th, o);
        let (dp, df) = (norm_diff(&th, &prev_th), obj - prev);
        trace.lines.push(format!(
            "Objective function (run {run}): {}  (Norm parameter change: {}  Objective function change: {})",
            fmt_g(obj),
            fe(dp),
            fe(df)
        ));
        if dp < o.repeat_tol_parms && df.abs() < o.repeat_tol_obj {
            break;
        }
    }
    (th, obj)
}

fn fe(x: f64) -> String {
    super::ms_files::fmt_e(x, 6)
}

/// `%g`-like formatting with 6 significant digits.
pub fn fmt_g(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    let a = x.abs();
    if a != 0.0 && !(1e-4..1e6).contains(&a) {
        return super::ms_files::fmt_e(x, 5);
    }
    let digits = if a == 0.0 { 0 } else { a.log10().floor() as i32 };
    let dec = (5 - digits).max(0) as usize;
    let s = format!("{x:.dec$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

/// Moves the transition parameters off the boundary (the Metropolis sampler works in logit coordinates).
fn interior(s: &MsSetup, th: &[f64]) -> Vec<f64> {
    let mut t = th.to_vec();
    for v in &s.layout.q {
        for &o in v {
            t[o] = t[o].clamp(1e-8, 1.0 - 1e-8);
        }
    }
    t
}

/// Initial run, large and small perturbation stages, final polish.
pub fn maximise(s: &MsSetup, start: &[f64], o: &OptimOptions, seed: u64, trace: &mut Trace) -> (Vec<f64>, f64) {
    let (mut best, mut best_obj) = find_mode(s, start, o, trace);
    s.normalise_signs(&mut best);
    trace.lines.push(format!("//== End initial run: maximum objective = {}", fe(best_obj)));
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6d73_5f65_7374);
    if o.random > 0 {
        for stage in 1..=o.random_max {
            let (draws, _) = mdd::rwmh(s, &interior(s, &best), 500, 10000, 1, 30000, rng.r#gen());
            let (mut sb, mut sb_obj) = (best.clone(), f64::NEG_INFINITY);
            for _ in 0..o.random {
                let k = rng.gen_range(0..draws.nrows());
                let st: Vec<f64> = draws.row(k).iter().copied().collect();
                let (mut t, v) = find_mode(s, &st, o, trace);
                s.normalise_signs(&mut t);
                if v > sb_obj {
                    sb = t;
                    sb_obj = v;
                }
            }
            let (prev, prev_obj) = (best.clone(), best_obj);
            if sb_obj > best_obj {
                best = sb;
                best_obj = sb_obj;
            }
            let (dp, df) = (norm_diff(&best, &prev), best_obj - prev_obj);
            trace.lines.push(format!(
                "//== End perturbation run {stage}: maximum objective = {}  (norm parameter change: {}  objective function change: {})", fe(best_obj), fe(dp), fe(df)
            ));
            if dp < o.random_tol_parms && df.abs() < o.random_tol_obj {
                break;
            }
        }
    }
    if o.random_small > 0 && o.random_max > 0 {
        for _stage in 1..=o.random_max {
            let (mut sb, mut sb_obj) = (best.clone(), f64::NEG_INFINITY);
            for _ in 0..o.random_small {
                let (draws, _) = mdd::rwmh(s, &interior(s, &best), o.random_small_ndraws.max(1), 0, 1, 0, rng.r#gen());
                let st: Vec<f64> = draws.row(draws.nrows() - 1).iter().copied().collect();
                let (mut t, v) = find_mode(s, &st, o, trace);
                s.normalise_signs(&mut t);
                if v > sb_obj {
                    sb = t;
                    sb_obj = v;
                }
            }
            let (prev, prev_obj) = (best.clone(), best_obj);
            if sb_obj > best_obj {
                best = sb;
                best_obj = sb_obj;
            }
            let (dp, df) = (norm_diff(&best, &prev), best_obj - prev_obj);
            trace.lines.push(format!(
                "//== End local perturbation run: maximum objective = {}  (norm parameter change: {}  objective function change: {})", fe(best_obj), fe(dp), fe(df)
            ));
            if dp < o.random_tol_parms && df.abs() < o.random_tol_obj {
                break;
            }
        }
    }
    // polish: joint quasi-Newton when Q is interior
    let q_interior = s.layout.q.iter().flatten().all(|&i| best[i] > 1e-9 && best[i] < 1.0 - 1e-9);
    if q_interior {
        let obj = |phi: &[f64]| {
            let v = s.log_posterior(&s.from_phi(phi));
            if v.is_finite() { -v } else { f64::INFINITY }
        };
        let (p, f, _) = bfgs_crit(&obj, s.to_phi(&best), 3000, 0.0);
        if f.is_finite() && -f > best_obj {
            best = s.from_phi(&p);
            best_obj = s.log_posterior(&best);
        }
    }
    (best, best_obj)
}
