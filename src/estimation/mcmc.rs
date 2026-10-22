// SPDX-License-Identifier: GPL-3.0-or-later
//! Random-walk Metropolis–Hastings: ports of `posterior_sampler.m`,
//! `posterior_sampler_initialization.m`, `posterior_sampler_core.m`,
//! `posterior_sampler_iteration.m` (default `random_walk_metropolis_hastings`
//! sampler with `rand_multivariate_normal` proposal) and
//! `calibrate_mh_scale_parameter.m` (`mh_tune_jscale`).
//!
//! Chains (`mh_nblck` blocks) run in parallel, each with its own
//! reproducible ChaCha stream (stream `j+1` of `seed` for chain `j`, stream 0
//! for the initial values), like Dynare's `seed + j` per block.
//!
//! Cores left over by the chains (and all cores during the single-chain
//! `mh_tune_jscale`) are used by *pre-fetching* (Brockwell 2006, "Parallel
//! Markov chain Monte Carlo simulation by pre-fetching"): the proposals of the
//! next few iterations under the most likely accept/reject outcomes are
//! generated ahead from a copy of the random stream and their posteriors are
//! evaluated in parallel; the sequential accept/reject walk then consumes them.
//! The chain is exactly (bit for bit) the sequential one: every proposal, random
//! number and acceptance decision is the same, only some extra evaluations of
//! branches the chain did not take are wasted. See [`MhOptions::prefetch`].

use anyhow::{Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use super::{Bounds, Objective};
use crate::linalg::{Mat, chol_upper};

/// MCMC options (`options_.mh_*`).
#[derive(Clone, Debug)]
pub struct MhOptions {
    /// `mh_replic`: draws per chain.
    pub replic: usize,
    /// `mh_nblck`: number of chains (Dynare default 2).
    pub nblocks: usize,
    /// `bayestopt_.jscale`: per-parameter jump scale (all equal to
    /// `options_.mh_jscale` unless set per parameter in `estimated_params`).
    pub jscale: Vec<f64>,
    /// `options_.mh_jscale`: scales the initial-value draws.
    pub mh_jscale: f64,
    /// `options_.mh_init_scale_factor` (default 2).
    pub init_scale_factor: f64,
    /// Seed (`options_.DynareRandomStreams.seed`).
    pub seed: u64,
    /// Call the progress callback every this many draws per chain (0 = never).
    pub progress_every: usize,
    /// Index of the first draw kept in memory (0 = all; set it to
    /// `floor(mh_drop * replic)` to skip storing the burn-in, which is all
    /// [`crate::estimation::posterior_stats`] needs; Brooks–Gelman needs all draws).
    pub store_from: usize,
    /// Posterior evaluations per chain run in parallel by pre-fetching (see the
    /// module documentation): 1 = plain sequential chains, 0 = automatic
    /// (`(2 threads / 3) / nblocks`, at most [`AUTO_PREFETCH_MAX`], and only for objectives
    /// costing over [`AUTO_PREFETCH_MIN_EVAL_SECS`]). Results do not depend on it.
    pub prefetch: usize,
    /// Chain starts, mh-file writers and end states of a persisted session
    /// (`<fname>/metropolis/`, `load_mh_file`, `mh_recover`); None = fresh, in memory only.
    pub session: Option<std::sync::Arc<super::mh_files::SessionIo>>,
}

/// Upper bound of the automatic pre-fetching width: on Apple silicon the
/// likelihood's matrix products run on the matrix unit shared by a core cluster,
/// so more concurrent evaluations mostly queue (see `tests/mcmc_scaling.rs`).
pub const AUTO_PREFETCH_MAX: usize = 6;

impl MhOptions {
    /// Dynare defaults for `npar` parameters: `mh_replic = 20000`,
    /// `mh_nblck = 2`, `mh_jscale = 2.38/sqrt(npar)` (`dynare_estimation_init.m`),
    /// `mh_init_scale_factor = 2`.
    pub fn new(npar: usize) -> Self {
        let js = 2.38 / (npar as f64).sqrt();
        MhOptions {
            replic: 20000,
            nblocks: 2,
            jscale: vec![js; npar],
            mh_jscale: js,
            init_scale_factor: 2.0,
            seed: 0,
            progress_every: 1000,
            store_from: 0,
            prefetch: 0,
            session: None,
        }
    }
}

/// One chain (block) of draws, stored row-major in f64.
#[derive(Clone, Debug)]
pub struct Chain {
    pub npar: usize,
    /// Draws `store_from..total_draws`, row-major (`npar` values per draw).
    pub draws: Vec<f64>,
    /// Log posterior kernel of each stored draw.
    pub logpost: Vec<f64>,
    /// Index (0-based, in the chain) of the first stored draw.
    pub first_stored: usize,
    /// Total number of draws generated.
    pub total_draws: usize,
    /// Number of accepted proposals.
    pub accepted: usize,
    /// Initial value (`record.InitialParameters`) and its log posterior.
    pub init: Vec<f64>,
    pub init_logpost: f64,
}

impl Chain {
    /// Draw number `k` (0-based, `k >= first_stored`).
    pub fn draw(&self, k: usize) -> &[f64] {
        let r = k - self.first_stored;
        &self.draws[r * self.npar..(r + 1) * self.npar]
    }
    /// Number of stored draws.
    pub fn len(&self) -> usize {
        self.logpost.len()
    }
    pub fn is_empty(&self) -> bool {
        self.logpost.is_empty()
    }
    /// `record.AcceptanceRatio`.
    pub fn acceptance_ratio(&self) -> f64 {
        self.accepted as f64 / self.total_draws as f64
    }
    /// Last draw (`record.LastParameters`).
    pub fn last(&self) -> &[f64] {
        self.draw(self.total_draws - 1)
    }
}

/// Progress report (per chain, every `progress_every` draws).
#[derive(Clone, Debug)]
pub struct MhProgress {
    pub chain: usize,
    pub draws_done: usize,
    pub total: usize,
    pub acceptance: f64,
}

#[derive(Clone, Debug)]
pub struct MhResult {
    pub chains: Vec<Chain>,
    /// `mh_init_scale_factor` after the automatic reductions of the initialization.
    pub init_scale_factor: f64,
    /// Pre-fetching width used (posterior evaluations per chain round; 1 = sequential).
    pub prefetch: usize,
}

pub(crate) fn rng_stream(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed);
    r.set_stream(stream);
    r
}

/// `rand_multivariate_normal(Mean, Sigma_upper_chol, n)`: `mean + randn(1,n) * U`.
/// `u` is upper triangular, column-major; writes into `out`.
#[inline]
pub(crate) fn rand_mvn<R: Rng>(rng: &mut R, mean: &[f64], u: &Mat, z: &mut [f64], out: &mut [f64]) {
    let n = mean.len();
    for zi in z.iter_mut() {
        *zi = rng.sample(StandardNormal);
    }
    let us = u.as_slice();
    for j in 0..n {
        let col = &us[j * n..j * n + j + 1];
        let mut s = 0.0;
        for (zi, c) in z[..=j].iter().zip(col) {
            s += zi * c;
        }
        out[j] = mean[j] + s;
    }
}

pub(crate) fn logpost_of(obj: &dyn Objective, x: &[f64]) -> f64 {
    -obj.eval(x).0
}

/// A random-walk MH chain in the state of `posterior_sampler_iteration.m`
/// (current point, its log posterior, the random stream), stepped either one
/// iteration at a time or several at once by pre-fetching.
pub(crate) struct Walker<'a> {
    obj: &'a dyn Objective,
    bounds: &'a Bounds,
    pub(crate) rng: ChaCha8Rng,
    pub(crate) last: Vec<f64>,
    pub(crate) last_lp: f64,
    /// Evaluations per pre-fetching round (1 = sequential).
    width: usize,
    steps: usize,
    accepted: usize,
    /// Posterior evaluations of the iterations done (proposals inside the bounds;
    /// wasted pre-fetched branches not counted).
    pub(crate) evaluated: usize,
    z: Vec<f64>,
    par: Vec<f64>,
}

/// One node of the pre-fetching tree: an iteration reached through a given
/// sequence of accept/reject outcomes.
struct Node {
    proposal: Vec<f64>,
    inside: bool,
    /// The random stream after the proposal's normal draws, and after the uniform
    /// of the acceptance test (drawn only when the posterior is finite).
    rng_after_normals: ChaCha8Rng,
    rng_after_uniform: Option<ChaCha8Rng>,
    uniform: f64,
    logpost: f64,
    /// Next iteration after a rejection / an acceptance, if pre-fetched.
    child: [Option<usize>; 2],
}

impl<'a> Walker<'a> {
    pub(crate) fn new(
        obj: &'a dyn Objective,
        bounds: &'a Bounds,
        rng: ChaCha8Rng,
        x: &[f64],
        lp: f64,
        width: usize,
    ) -> Self {
        let n = x.len();
        Walker {
            obj,
            bounds,
            rng,
            last: x.to_vec(),
            last_lp: lp,
            width: width.max(1),
            steps: 0,
            accepted: 0,
            evaluated: 0,
            z: vec![0.0; n],
            par: vec![0.0; n],
        }
    }

    /// Strictly inside the bounds (`posterior_sampler_iteration.m`).
    #[inline]
    pub(crate) fn inside(&self, p: &[f64]) -> bool {
        let b = self.bounds;
        p.iter().zip(&b.lb).zip(&b.ub).all(|((v, l), u)| v > l && v < u)
    }

    /// One iteration: proposal `last + randn * u` (u upper triangular), rejected
    /// outright outside the open bounds, accepted when `log(rand) < lp - last_lp`
    /// (`rand` drawn only for a finite posterior, as MATLAB's short-circuit `&&`).
    fn step(&mut self, u: &Mat) -> bool {
        rand_mvn(&mut self.rng, &self.last, u, &mut self.z, &mut self.par);
        let inside = self.inside(&self.par);
        self.evaluated += inside as usize;
        let lp = if inside { logpost_of(self.obj, &self.par) } else { f64::NEG_INFINITY };
        let acc = lp > f64::NEG_INFINITY && self.rng.r#gen::<f64>().ln() < lp - self.last_lp;
        if acc {
            self.last.copy_from_slice(&self.par);
            self.last_lp = lp;
        }
        self.steps += 1;
        self.accepted += acc as usize;
        acc
    }

    /// Runs between 1 and `max_steps` iterations (`max_steps >= 1`), calling
    /// `on_step(current point, its log posterior, accepted)` after each; returns
    /// the number of iterations done. Identical, iteration by iteration, to
    /// calling [`Self::step`] repeatedly.
    pub(crate) fn advance(&mut self, u: &Mat, max_steps: usize, on_step: &mut dyn FnMut(&[f64], f64, bool)) -> usize {
        if self.width <= 1 || max_steps <= 1 {
            let acc = self.step(u);
            on_step(&self.last, self.last_lp, acc);
            return 1;
        }
        let nodes = self.build_tree(u, max_steps);
        let mut done = 0;
        let mut cur = Some(0);
        while let Some(i) = cur {
            let nd = &nodes[i];
            let lp = if nd.inside { nd.logpost } else { f64::NEG_INFINITY };
            let finite = lp > f64::NEG_INFINITY;
            let acc = finite && nd.uniform.ln() < lp - self.last_lp;
            if acc {
                self.last.copy_from_slice(&nd.proposal);
                self.last_lp = lp;
            }
            self.rng = match (&nd.rng_after_uniform, finite) {
                (Some(r), true) => r.clone(),
                _ => nd.rng_after_normals.clone(),
            };
            self.steps += 1;
            self.accepted += acc as usize;
            self.evaluated += nd.inside as usize;
            done += 1;
            on_step(&self.last, self.last_lp, acc);
            // The pre-fetched children of an evaluated node assumed a finite posterior
            // (a uniform drawn); an infinite one leaves the stream elsewhere.
            if nd.inside && !finite {
                break;
            }
            cur = nd.child[acc as usize];
        }
        done
    }

    /// Greedy pre-fetching tree: expands the most probable outcome paths (with the
    /// chain's acceptance rate so far as the probability of an acceptance) until
    /// `width` evaluations are scheduled, then evaluates them in parallel. Node 0
    /// is the next iteration; proposals outside the bounds need no evaluation.
    fn build_tree(&self, u: &Mat, max_steps: usize) -> Vec<Node> {
        struct Cand {
            prob: f64,
            depth: usize,
            /// Current point of the path: the start (None) or an accepted node's proposal.
            base: Option<usize>,
            rng: ChaCha8Rng,
            parent: Option<(usize, usize)>,
        }
        let pa = ((self.accepted as f64 + 1.0) / (self.steps as f64 + 3.0)).clamp(0.02, 0.98);
        let npar = self.last.len();
        let mut nodes: Vec<Node> = Vec::with_capacity(2 * self.width);
        let mut frontier = vec![Cand { prob: 1.0, depth: 0, base: None, rng: self.rng.clone(), parent: None }];
        let mut evals = 0;
        let max_nodes = 4 * self.width + 16;
        let mut z = vec![0.0; npar];
        while !frontier.is_empty() && nodes.len() < max_nodes {
            let k = (0..frontier.len()).max_by(|&a, &b| frontier[a].prob.total_cmp(&frontier[b].prob)).unwrap();
            let Cand { prob, depth, base, mut rng, parent } = frontier.swap_remove(k);
            if depth >= max_steps {
                continue;
            }
            let mut proposal = vec![0.0; npar];
            let mean = match base {
                None => &self.last,
                Some(b) => &nodes[b].proposal,
            };
            rand_mvn(&mut rng, mean, u, &mut z, &mut proposal);
            let inside = self.inside(&proposal);
            if inside && evals == self.width {
                continue; // no evaluation budget left for this path
            }
            let rng_after_normals = rng.clone();
            let idx = nodes.len();
            if let Some((p, branch)) = parent {
                nodes[p].child[branch] = Some(idx);
            }
            let (uniform, rng_after_uniform) = if inside {
                evals += 1;
                let uni = rng.r#gen::<f64>();
                frontier.push(Cand {
                    prob: prob * (1.0 - pa),
                    depth: depth + 1,
                    base,
                    rng: rng.clone(),
                    parent: Some((idx, 0)),
                });
                frontier.push(Cand {
                    prob: prob * pa,
                    depth: depth + 1,
                    base: Some(idx),
                    rng: rng.clone(),
                    parent: Some((idx, 1)),
                });
                (uni, Some(rng))
            } else {
                frontier.push(Cand { prob, depth: depth + 1, base, rng: rng.clone(), parent: Some((idx, 0)) });
                (0.0, None)
            };
            nodes.push(Node {
                proposal,
                inside,
                rng_after_normals,
                rng_after_uniform,
                uniform,
                logpost: f64::NEG_INFINITY,
                child: [None, None],
            });
        }
        let todo: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].inside).collect();
        let obj = self.obj;
        let lps: Vec<f64> = todo.par_iter().map(|&i| logpost_of(obj, &nodes[i].proposal)).collect();
        for (&i, lp) in todo.iter().zip(lps) {
            nodes[i].logpost = lp;
        }
        nodes
    }
}

/// Pre-fetching width for `chains` chains running at the same time, given the
/// requested [`MhOptions::prefetch`] (0 = automatic) and the measured cost of one
/// evaluation: automatic pre-fetching needs spare threads and an objective
/// expensive enough (over [`AUTO_PREFETCH_MIN_EVAL_SECS`]) to amortise a parallel
/// round. The draws never depend on the width, so a timing-based choice is safe.
pub fn prefetch_width(requested: usize, chains: usize, eval_secs: f64) -> usize {
    if requested > 0 {
        return requested;
    }
    if eval_secs < AUTO_PREFETCH_MIN_EVAL_SECS {
        return 1;
    }
    // Two thirds of the threads: past ~12 concurrent posterior evaluations on 18 cores the
    // shared matrix units are saturated and wasted branches only slow the others.
    (2 * rayon::current_num_threads() / 3 / chains.max(1)).clamp(1, AUTO_PREFETCH_MAX)
}

/// Objectives cheaper than this are not pre-fetched automatically.
pub const AUTO_PREFETCH_MIN_EVAL_SECS: f64 = 1e-4;

/// Port of `posterior_sampler_initialization.m` (fresh run, no `load_mh_file`):
/// with several chains, draws each start from
/// `N(xparam1, (mh_init_scale_factor*mh_jscale)^2 * invhess)` until it lies in
/// the bounds with a finite posterior; after 100 failures the scale factor is
/// reduced by 10% per further failure (`nointeractive` behaviour), giving up
/// after 10 reductions. With one chain, starts at the mode.
pub fn initial_values(
    obj: &dyn Objective,
    xparam1: &[f64],
    d: &Mat,
    bounds: &Bounds,
    opts: &MhOptions,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, f64)> {
    let npar = xparam1.len();
    let mut factor = opts.init_scale_factor;
    let mut ix2 = Vec::with_capacity(opts.nblocks);
    let mut ilogpo2 = Vec::with_capacity(opts.nblocks);
    let inside = |c: &[f64]| (0..npar).all(|i| c[i] >= bounds.lb[i] && c[i] <= bounds.ub[i]);
    if opts.nblocks > 1 {
        let mut rng = rng_stream(opts.seed, 0);
        let mut z = vec![0.0; npar];
        let mut cand = vec![0.0; npar];
        for j in 0..opts.nblocks {
            let mut validate = false;
            let mut init_iter = 0;
            let mut trial = 1;
            while !validate && trial <= 10 {
                let u = d * (factor * opts.mh_jscale);
                rand_mvn(&mut rng, xparam1, &u, &mut z, &mut cand);
                if inside(&cand) {
                    let lp = logpost_of(obj, &cand);
                    if lp.is_finite() {
                        ix2.push(cand.clone());
                        ilogpo2.push(lp);
                        validate = true;
                    }
                }
                init_iter += 1;
                if !validate && init_iter > 100 {
                    factor *= 0.9;
                    trial += 1;
                }
            }
            if !validate {
                bail!("posterior_sampler_initialization: I'm unable to find a starting value for block {}", j + 1);
            }
        }
    } else {
        if !inside(xparam1) {
            bail!("posterior_sampler_initialization: The posterior mode lies outside the prior bounds.");
        }
        ix2.push(xparam1.to_vec());
        ilogpo2.push(logpost_of(obj, xparam1));
    }
    Ok((ix2, ilogpo2, factor))
}

/// Port of `posterior_sampler.m` / `posterior_sampler_core.m` /
/// `posterior_sampler_iteration.m` for the random-walk Metropolis–Hastings:
/// proposal `x + randn * chol(invhess) * diag(jscale)`, rejected outright when
/// not strictly inside `bounds` (normally [`crate::estimation::mcmc_bounds`]),
/// accepted when `log(rand) < logpost - last_logpost`. Every draw (accepted or
/// repeated) is stored, as in Dynare's `x2`/`logpo2`.
pub fn rw_metropolis(
    obj: &dyn Objective,
    xparam1: &[f64],
    invhess: &Mat,
    bounds: &Bounds,
    opts: &MhOptions,
    progress: Option<&(dyn Fn(&MhProgress) + Sync)>,
) -> Result<MhResult> {
    let npar = xparam1.len();
    if opts.jscale.len() != npar {
        bail!("rw_metropolis: jscale has {} entries for {npar} parameters", opts.jscale.len());
    }
    let ses = opts.session.as_deref();
    let Some(d) = ses.and_then(|s| s.proposal_chol.clone()).or_else(|| chol_upper(invhess)) else {
        bail!("rw_metropolis: the proposal covariance (inverse Hessian) is not positive definite");
    };
    let t0 = std::time::Instant::now();
    let (ix2, ilogpo2, factor, eval_secs) = match ses.and_then(|s| s.start_points()) {
        Some((x, lp)) => (x, lp, opts.init_scale_factor, ses.unwrap().eval_secs),
        None => {
            let (x, lp, f) = initial_values(obj, xparam1, &d, bounds, opts)?;
            (x, lp, f, t0.elapsed().as_secs_f64() / opts.nblocks.max(1) as f64)
        }
    };
    // proposal_covariance_Cholesky_decomposition = d*diag(jscale)
    let mut u = d.clone();
    for j in 0..npar {
        for i in 0..npar {
            u[(i, j)] *= opts.jscale[j];
        }
    }
    let nkeep = opts.replic.saturating_sub(opts.store_from);
    let width = prefetch_width(opts.prefetch, opts.nblocks, eval_secs);
    let run_chain = |b: usize| -> Chain {
        // continued chains start from a saved generator state; draws go to the mh files
        let (rng, nruns) = match ses {
            Some(s) => s.chain_rng(b, opts.seed, opts.replic),
            None => (rng_stream(opts.seed, b as u64 + 1), opts.replic),
        };
        let mut writer = ses.and_then(|s| s.take_writer(b));
        let mut w = Walker::new(obj, bounds, rng, &ix2[b], ilogpo2[b], width);
        let mut draws = Vec::with_capacity(nkeep * npar);
        let mut logpost = Vec::with_capacity(nkeep);
        let mut it = 0usize;
        let mut accepted = 0usize;
        let mut done = 0;
        while done < nruns {
            let lim = writer.as_ref().map_or(usize::MAX, |wr| wr.until_flush());
            let mut on_step = |last: &[f64], last_lp: f64, acc: bool| {
                accepted += acc as usize;
                if let Some(wr) = writer.as_mut() {
                    wr.push(last, last_lp, acc as u8 as f64, 1.0);
                }
                if it >= opts.store_from {
                    draws.extend_from_slice(last);
                    logpost.push(last_lp);
                }
                it += 1;
                if let Some(cb) = progress
                    && opts.progress_every > 0
                    && it.is_multiple_of(opts.progress_every)
                {
                    cb(&MhProgress { chain: b, draws_done: it, total: nruns, acceptance: accepted as f64 / it as f64 });
                }
            };
            done += w.advance(&u, (nruns - done).min(lim), &mut on_step);
            if let Some(wr) = writer.as_mut() {
                wr.after(&w.rng);
            }
        }
        if let Some(s) = ses {
            s.finish_chain(b, writer, &w.rng, &w.last, w.last_lp, w.accepted as f64, nruns as f64, nruns);
        }
        Chain {
            npar,
            draws,
            logpost,
            first_stored: opts.store_from.min(nruns),
            total_draws: nruns,
            accepted: w.accepted,
            init: ix2[b].clone(),
            init_logpost: ilogpo2[b],
        }
    };
    // One OS thread per chain when the pool has room for all of them (the chains'
    // pre-fetched evaluations then run on the rayon pool); otherwise rayon
    // schedules the chains on its workers.
    let chains: Vec<Chain> = if opts.nblocks > 1 && opts.nblocks <= rayon::current_num_threads() {
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..opts.nblocks)
                .map(|b| {
                    sc.spawn(move || {
                        crate::parallel::set_current_thread_high_qos();
                        run_chain(b)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("MCMC chain panicked")).collect()
        })
    } else {
        (0..opts.nblocks).into_par_iter().map(run_chain).collect()
    };
    Ok(MhResult { chains, init_scale_factor: factor, prefetch: width })
}

/// `options_.mh_tune_jscale` (defaults from `default_option_values.m`).
#[derive(Clone, Debug)]
pub struct TuneOptions {
    pub target: f64,
    /// Initial scale; None = `2.38/sqrt(npar)`.
    pub guess: Option<f64>,
    pub maxiter: usize,
    pub rho: f64,
    pub stepsize: usize,
    pub c1: f64,
    pub c2: f64,
    pub c3: usize,
    pub seed: u64,
    /// Progress callback `(iteration, overall acceptance, scale)` every `stepsize` draws.
    pub verbose: bool,
    /// Posterior evaluations run in parallel by pre-fetching (as [`MhOptions::prefetch`]:
    /// 1 = sequential, 0 = automatic). The result does not depend on it.
    pub prefetch: usize,
}

impl Default for TuneOptions {
    fn default() -> Self {
        TuneOptions {
            target: 0.33,
            guess: None,
            maxiter: 200_000,
            rho: 0.7,
            stepsize: 1000,
            c1: 0.02,
            c2: 0.05,
            c3: 4,
            seed: 0,
            verbose: false,
            prefetch: 0,
        }
    }
}

/// Port of `estimation/calibrate_mh_scale_parameter.m`: one RW-MH chain from
/// `x` whose scale is adapted every `stepsize` draws towards the target
/// acceptance ratio; stops once both the local and the overall acceptance
/// ratios are within `c1`/`c2` of the target for more than `c3` consecutive
/// windows (or after `maxiter` draws).
pub fn calibrate_mh_scale_parameter(
    obj: &dyn Objective,
    covariance: &Mat,
    x: &[f64],
    bounds: &Bounds,
    opts: &TuneOptions,
) -> Result<f64> {
    let n = x.len();
    let Some(dd) = chol_upper(covariance) else {
        bail!("The covariance matrix has to be a symmetric positive definite matrix!");
    };
    let rng = rng_stream(opts.seed, u64::MAX);
    let t0 = std::time::Instant::now();
    let logpo1 = logpost_of(obj, x);
    let eval_secs = t0.elapsed().as_secs_f64();
    let mut correction = 1.0_f64;
    let mut scale = opts.guess.unwrap_or(2.38 / (n as f64).sqrt());
    let (mut j, mut jj, mut isux, mut jsux, mut i) = (1usize, 1usize, 0usize, 0usize, 0usize);
    let mut w = Walker::new(obj, bounds, rng, x, logpo1, prefetch_width(opts.prefetch, 1, eval_secs));
    let mut u = &dd * scale;
    while j <= opts.maxiter {
        // Iterations j..=j+s-1 share the scale: stop each batch at the next adaptation.
        let to_boundary = if opts.stepsize == 0 { usize::MAX } else { opts.stepsize - (j - 1) % opts.stepsize };
        let max_steps = to_boundary.min(opts.maxiter - j + 1);
        let mut acc_batch = 0usize;
        let s = w.advance(&u, max_steps, &mut |_, _, acc| acc_batch += acc as usize);
        isux += acc_batch;
        jsux += acc_batch;
        j += s - 1;
        jj += s - 1;
        if j.is_multiple_of(opts.stepsize) {
            let r1 = jsux as f64 / jj as f64;
            let r2 = isux as f64 / j as f64;
            let c1 = r1 / opts.target;
            if (c1 - 1.0).abs() > 0.05 {
                correction = correction.powf(opts.rho) * c1.powf(1.0 - opts.rho);
            } else {
                correction = c1;
            }
            if c1 > 0.0 {
                scale *= correction;
            } else {
                scale /= 10.0;
            }
            u = &dd * scale;
            if opts.verbose {
                println!("mh_tune_jscale: draw {j}, acceptance {r2:.4} [{r1:.4}], scale {scale}");
            }
            jsux = 0;
            jj = 0;
            if (r2 - opts.target).abs() < opts.c2 && (r1 - opts.target).abs() < opts.c1 {
                i += 1;
            } else {
                i = 0;
            }
            if i > opts.c3 {
                break;
            }
        }
        j += 1;
        jj += 1;
    }
    Ok(scale)
}

/// Port of `utilities/estimation/tune_mcmc_mh_jscale_wrapper.m`: tunes
/// `mh_jscale` with the proposal covariance `invhess` and the MCMC bounds.
pub fn mh_tune_jscale(
    obj: &dyn Objective,
    invhess: &Mat,
    xparam1: &[f64],
    bounds: &Bounds,
    opts: &TuneOptions,
) -> Result<f64> {
    calibrate_mh_scale_parameter(obj, invhess, xparam1, bounds, opts)
}
