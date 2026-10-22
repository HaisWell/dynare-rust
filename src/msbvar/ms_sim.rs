// SPDX-License-Identifier: GPL-3.0-or-later
//! `ms_irf`, `ms_variance_decomposition`, `ms_forecast` of the Markov-switching SBVAR (part 2b).
//! Dynare's C code is not shipped; the definitions below were pinned down by running Dynare 7.1 (7.2 gives the same output) on a
//! 2-regime model (README "MS-SBVAR, part 2b"):
//!
//! * a regime-`s` impulse response keeps the regime fixed over the horizon: shock `j` (one standard
//!   deviation of equation `j` in regime `s`, i.e. `1/sqrt(zeta_j(s))`) moves `y` by row `j` of
//!   `diag(zeta^-1/2) A0^-1`, then `R_h = sum_l R_{h-l} B_l` with `B = A+ A0^-1`; file row `h+1` is horizon `h`,
//!   columns `shock*n + variable`;
//! * "ergodic" / `filtered_probabilities` mix the regimes with the ergodic distribution / the filtered
//!   probabilities of the last observation; percentiles are those of that (exact) mixture, per element;
//! * variance decomposition: the share of shock `j` in the forecast-error variance of variable `i`
//!   (columns `i*n + j`), the mean version being the ratio of the regime-weighted variance contributions.

use anyhow::{Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use super::hamilton::Filtered;
use super::ms_estimation::{MsSetup, Regimes};
use crate::linalg::Mat;

/// Which regime weights the result is computed for.
#[derive(Clone, Debug, PartialEq)]
pub enum Mode {
    /// Regime `k` (0-based), regime fixed.
    Regime(usize),
    /// Filtered probabilities of the last observation.
    Filtered,
    /// Ergodic distribution of the chain.
    Ergodic,
}

fn inv_or_id(m: &Mat) -> Mat {
    m.clone().try_inverse().unwrap_or_else(|| Mat::identity(m.nrows(), m.ncols()))
}

/// Impulse responses `R_h` (`shock x variable`), `h = 0..horizon-1`, in regime `s`.
pub fn irf_regime(set: &MsSetup, r: &Regimes, s: usize, horizon: usize) -> Vec<Mat> {
    let n = set.nvar;
    let a0inv = inv_or_id(&r.a0[s]);
    let bhat = &r.aplus[s] * &a0inv;
    let c0 = Mat::from_fn(n, n, |j, i| a0inv[(j, i)] / r.zeta[s][j].sqrt());
    let mut out: Vec<Mat> = vec![c0];
    for h in 1..horizon {
        let mut m = Mat::zeros(n, n);
        for l in 1..=set.nlags.min(h) {
            let bl = bhat.view(((l - 1) * n, 0), (n, n));
            m += &out[h - l] * bl;
        }
        out.push(m);
    }
    out
}

/// `(irf, cumulative variance contributions)` as `horizon x n^2` matrices: IRF columns `shock*n+var`,
/// contribution columns `var*n+shock`.
pub fn regime_response(set: &MsSetup, r: &Regimes, s: usize, horizon: usize) -> (Mat, Mat) {
    let n = set.nvar;
    let rs = irf_regime(set, r, s, horizon);
    let irf = Mat::from_fn(horizon, n * n, |h, c| rs[h][(c / n, c % n)]);
    let mut vc = Mat::zeros(horizon, n * n);
    for h in 0..horizon {
        for i in 0..n {
            for j in 0..n {
                let prev = if h > 0 { vc[(h - 1, i * n + j)] } else { 0.0 };
                vc[(h, i * n + j)] = prev + rs[h][(j, i)].powi(2);
            }
        }
    }
    (irf, vc)
}

/// Shares of the shocks in the variance of each variable (rows of `vc` normalised per variable).
pub fn shares(vc: &Mat, n: usize) -> Mat {
    Mat::from_fn(vc.nrows(), vc.ncols(), |h, c| {
        let i = c / n;
        let tot: f64 = (0..n).map(|j| vc[(h, i * n + j)]).sum();
        if tot > 0.0 { vc[(h, c)] / tot } else { 0.0 }
    })
}

/// Ergodic distribution of the column-stochastic `q`.
pub fn ergodic(q: &Mat) -> Vec<f64> {
    let s = q.nrows();
    let mut m = q.clone();
    for _ in 0..14 {
        m = &m * &m;
    }
    let v = &m * Mat::from_element(s, 1, 1.0 / s as f64);
    let t: f64 = v.iter().sum();
    v.iter().map(|x| x / t).collect()
}

/// Regime weights of `mode`.
pub fn weights(mode: &Mode, f: &Filtered, q: &Mat) -> Vec<f64> {
    match mode {
        Mode::Filtered => {
            // Dynare weights by the one-step-ahead probabilities `Q f_T` (verified: 0.7039 vs f_T 0.7516 of
            // regime 1 on the test model reproduces its percentiles and variance decomposition)
            let t = f.filtered.nrows() - 1;
            let ft = Mat::from_fn(f.filtered.ncols(), 1, |s, _| f.filtered[(t, s)]);
            (q * ft).iter().cloned().collect()
        }
        Mode::Ergodic => ergodic(q),
        Mode::Regime(k) => (0..q.nrows()).map(|s| if s == *k { 1.0 } else { 0.0 }).collect(),
    }
}

/// Quantile of a weighted sample: the smallest value whose cumulative weight reaches `p`.
pub fn wquantile(items: &mut [(f64, f64)], p: f64) -> f64 {
    items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let tot: f64 = items.iter().map(|x| x.1).sum();
    let mut c = 0.0;
    for (v, w) in items.iter() {
        c += w / tot;
        if c >= p - 1e-12 {
            return *v;
        }
    }
    items.last().map(|x| x.0).unwrap_or(f64::NAN)
}

/// Percentile blocks (`P*horizon x cols`, block `k` = percentile `k`) of a weighted sample of `horizon x cols` matrices.
pub fn percentile_blocks(samples: &[(f64, Mat)], pcts: &[f64]) -> Mat {
    let (h, c) = (samples[0].1.nrows(), samples[0].1.ncols());
    let mut out = Mat::zeros(pcts.len() * h, c);
    let mut items: Vec<(f64, f64)> = Vec::with_capacity(samples.len());
    for t in 0..h {
        for j in 0..c {
            items.clear();
            items.extend(samples.iter().filter(|(w, _)| *w > 0.0).map(|(w, m)| (m[(t, j)], *w)));
            for (k, &p) in pcts.iter().enumerate() {
                out[(k * h + t, j)] = wquantile(&mut items, p);
            }
        }
    }
    out
}

/// What is computed.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Irf,
    /// Variance decomposition percentiles.
    Vd,
    /// Variance decomposition mean.
    VdMean,
}

/// Impulse responses / variance decomposition for one weighting. `thetas`: the parameter vectors (the
/// mode, or the posterior draws under parameter uncertainty; the regime mixture is integrated exactly).
pub fn compute(
    set: &MsSetup,
    thetas: &[Vec<f64>],
    mode: &Mode,
    kind: Kind,
    horizon: usize,
    pcts: &[f64],
) -> Result<Mat> {
    let n = set.nvar;
    let nt = thetas.len() as f64;
    let mut samples: Vec<(f64, Mat)> = vec![];
    let mut mean_shares: Option<Mat> = None;
    for th in thetas {
        let Some((r, f)) = set.filter(th) else { bail!("parameter vector outside the parameter space") };
        let w = weights(mode, &f, &r.q);
        let mut vc_w = Mat::zeros(horizon, n * n);
        for (s, &ws) in w.iter().enumerate() {
            if ws <= 0.0 {
                continue;
            }
            let (irf, vc) = regime_response(set, &r, s, horizon);
            match kind {
                Kind::Irf => samples.push((ws / nt, irf)),
                Kind::Vd => samples.push((ws / nt, shares(&vc, n))),
                Kind::VdMean => vc_w += vc * ws,
            }
        }
        if kind == Kind::VdMean {
            let sh = shares(&vc_w, n) / nt;
            mean_shares = Some(match mean_shares {
                Some(m) => m + sh,
                None => sh,
            });
        }
    }
    Ok(match kind {
        Kind::VdMean => mean_shares.unwrap(),
        _ => percentile_blocks(&samples, pcts),
    })
}

/// Dynare 7.2 starts the forecast one period early: its first forecast row is the one-step forecast of the last
/// observation `y_T` from `y_{T-1}, y_{T-2}, ...` (found by comparing with its output; 0 would give `T+1`).
pub const FORECAST_LAG: usize = 1;

/// Forecast simulation. Returns `P*horizon x n` percentile blocks. `start`: regime at the last
/// observation, or `None` to draw it from the filtered probabilities; with `evolve` the regime then moves with
/// `q` from period `T+1` on (otherwise it stays at `start` over the horizon).
pub fn forecast(
    set: &MsSetup,
    thetas: &[Vec<f64>],
    start: Option<usize>,
    horizon: usize,
    paths_per_theta: usize,
    pcts: &[f64],
    evolve: bool,
    seed: u64,
) -> Result<Mat> {
    let n = set.nvar;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let t_n = set.y.nrows();
    let mut draws: Vec<(f64, Mat)> = Vec::with_capacity(thetas.len() * paths_per_theta);
    let pick = |w: &[f64], u: f64| -> usize {
        let tot: f64 = w.iter().sum();
        let mut c = 0.0;
        for (i, x) in w.iter().enumerate() {
            c += x / tot;
            if u <= c {
                return i;
            }
        }
        w.len() - 1
    };
    for th in thetas {
        let Some((r, f)) = set.filter(th) else { bail!("parameter vector outside the parameter space") };
        let nst = r.q.nrows();
        let tprob: Vec<f64> = (0..nst).map(|s| f.filtered[(f.filtered.nrows() - 1, s)]).collect();
        let pre: Vec<(Mat, Mat)> = (0..nst)
            .map(|s| {
                let a0inv = inv_or_id(&r.a0[s]);
                let bhat = &r.aplus[s] * &a0inv;
                let c0 = Mat::from_fn(n, n, |j, i| a0inv[(j, i)] / r.zeta[s][j].sqrt());
                (bhat, c0)
            })
            .collect();
        for _ in 0..paths_per_theta {
            let mut s = match start {
                Some(k) => k,
                None => pick(&tprob, rng.r#gen::<f64>()),
            };
            let mut hist: Vec<Vec<f64>> =
                (0..set.nlags).map(|l| (0..n).map(|v| set.y[(t_n - 1 - l - FORECAST_LAG, v)]).collect()).collect();
            let mut path = Mat::zeros(horizon, n);
            for h in 0..horizon {
                if evolve {
                    let col: Vec<f64> = (0..nst).map(|i| r.q[(i, s)]).collect();
                    s = pick(&col, rng.r#gen::<f64>());
                }
                let (bhat, c0) = &pre[s];
                let mut x = Mat::zeros(1, set.ncoef);
                for l in 0..set.nlags {
                    for v in 0..n {
                        x[(0, l * n + v)] = hist[l][v];
                    }
                }
                x[(0, set.ncoef - 1)] = 1.0;
                let mean = &x * bhat;
                let eps: Vec<f64> = (0..n).map(|_| rng.sample::<f64, _>(StandardNormal)).collect();
                let e = Mat::from_column_slice(1, n, &eps) * c0;
                let y: Vec<f64> = (0..n).map(|v| mean[(0, v)] + e[(0, v)]).collect();
                for v in 0..n {
                    path[(h, v)] = y[v];
                }
                hist.insert(0, y);
                hist.truncate(set.nlags);
            }
            draws.push((1.0, path));
        }
    }
    Ok(percentile_blocks(&draws, pcts))
}
