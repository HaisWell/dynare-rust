// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `gaussian_filter.m` + `gaussian_filter_bank.m` (Gaussian particle filter of
//! Kotecha & Djuric with a sigma-point / Monte Carlo proposal).

use super::law_of_motion::{iterate, measurement};
use super::linalg::{
    chol_lower, mrdivide, neff, probability2, qr2_r, reduced_rank_cholesky, select_rows, weighted_mean,
};
use super::resampling::resample;
use super::rng::Draws;
use super::sigma_points::{SigmaPoints, for_approximation};
use super::{Approximation, FilterOutput, ParticleOptions, ReducedForm, ResamplingStatus};
use crate::linalg::Mat;
use anyhow::{Result, anyhow, bail};

/// `reduced_rank_cholesky(X)'`.
pub(crate) fn rrc_lower(x: &Mat) -> Result<Mat> {
    reduced_rank_cholesky(x)
        .map(|t| t.transpose())
        .ok_or_else(|| anyhow!("reduced_rank_cholesky: not positive semidefinite"))
}

/// Output of the update step of a sigma-point filter.
pub(crate) struct BankOutput {
    pub pred_state_mean: Vec<f64>,
    /// `PredictedStateVariance` (unscented) or `dState' * dState` (cubature / Monte Carlo).
    pub pred_state_variance: Mat,
    pub pred_obs_sqrt: Mat,
    pub pred_obs_variance: Option<Mat>,
    pub prediction_error: Vec<f64>,
    pub state_mean: Vec<f64>,
    pub state_sqrt: Mat,
}

/// Prediction + update with sigma points over (state, shock): the common core of
/// `gaussian_filter_bank.m` and `nonlinear_kalman_filter.m`.
/// `nodes`: K x (ns+nu) (rows), weights `wm`/`wc`; `square_root` = the cubature / Monte
/// Carlo branch (QR of the weighted deviations), else the unscented covariance branch.
/// The shocks are `shock_mean + qc * nodes(:, ns+1:end)'` (zero mean when `None`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sigma_update(
    rf: &ReducedForm,
    obs: &[f64],
    mean: &[f64],
    sqrt_p: &Mat,
    qc: &Mat,
    shock_mean: Option<&[f64]>,
    hc: &Mat,
    sp: &SigmaPoints,
    square_root: bool,
) -> Result<(BankOutput, Mat)> {
    let ns = rf.ns();
    let nu = rf.nu();
    let pp = rf.pp();
    if sqrt_p.ncols() != ns {
        bail!("vertcat: dimensions of arrays being concatenated are not consistent (rank-deficient state variance)");
    }
    let k = sp.len();
    // sigma_points = xbar + sqr_Px * nodes'
    let mut states = Mat::zeros(ns, k);
    let mut eps = Mat::zeros(nu, k);
    for c in 0..k {
        for i in 0..ns {
            let mut v = mean[i];
            for j in 0..ns {
                v += sqrt_p[(i, j)] * sp.nodes[(c, j)];
            }
            states[(i, c)] = v;
        }
        for i in 0..nu {
            let mut v = shock_mean.map_or(0.0, |m| m[i]);
            for j in 0..nu {
                v += qc[(i, j)] * sp.nodes[(c, ns + j)];
            }
            eps[(i, c)] = v;
        }
    }
    let tmp = iterate(rf, &states, &eps, None).y;
    let ts = select_rows(&tmp, &rf.mf0);
    let to = select_rows(&tmp, &rf.mf1);
    let psm = weighted_mean(&ts, &sp.wm);
    let pom = weighted_mean(&to, &sp.wm);
    let pe: Vec<f64> = (0..pp).map(|i| obs[i] - pom[i]).collect();
    if square_root {
        let w: Vec<f64> = sp.wm.iter().map(|v| v.sqrt()).collect();
        let ds = Mat::from_fn(k, ns, |c, i| (ts[(i, c)] - psm[i]) * w[c]);
        let dob = Mat::from_fn(k, pp, |c, i| (to[(i, c)] - pom[i]) * w[c]);
        let psv = ds.transpose() * &ds;
        // big_mat = [dObserved dState; H_chol zeros]
        let big = Mat::from_fn(k + pp, pp + ns, |r, c| {
            if r < k {
                if c < pp { dob[(r, c)] } else { ds[(r, c - pp)] }
            } else if c < pp {
                hc[(r - k, c)]
            } else {
                0.0
            }
        });
        let mat = qr2_r(&big).transpose();
        let povs = mat.view((0, 0), (pp, pp)).into_owned();
        let coss = mat.view((pp, 0), (ns, pp)).into_owned();
        let s = mat.view((pp, pp), (ns, ns)).into_owned();
        let gain = mrdivide(&coss, &povs);
        let state_mean: Vec<f64> =
            (0..ns).map(|i| psm[i] + (0..pp).map(|j| gain[(i, j)] * pe[j]).sum::<f64>()).collect();
        Ok((
            BankOutput {
                pred_state_mean: psm,
                pred_state_variance: psv,
                pred_obs_sqrt: povs,
                pred_obs_variance: None,
                prediction_error: pe,
                state_mean,
                state_sqrt: s,
            },
            tmp,
        ))
    } else {
        let wc = &sp.wc;
        let ds = Mat::from_fn(ns, k, |i, c| ts[(i, c)] - psm[i]);
        let dob = Mat::from_fn(pp, k, |i, c| to[(i, c)] - pom[i]);
        let dsw = Mat::from_fn(ns, k, |i, c| ds[(i, c)] * wc[c]);
        let dow = Mat::from_fn(pp, k, |i, c| dob[(i, c)] * wc[c]);
        let psv = &dsw * ds.transpose();
        let pov = &dow * dob.transpose() + &rf.h;
        let psoc = &dsw * dob.transpose();
        let gain = mrdivide(&psoc, &pov);
        let state_mean: Vec<f64> =
            (0..ns).map(|i| psm[i] + (0..pp).map(|j| gain[(i, j)] * pe[j]).sum::<f64>()).collect();
        let mut v = &psv - &gain * &pov * gain.transpose();
        v = (&v + v.transpose()) * 0.5;
        Ok((
            BankOutput {
                pred_state_mean: psm,
                pred_state_variance: psv,
                pred_obs_sqrt: Mat::zeros(0, 0),
                pred_obs_variance: Some(pov),
                prediction_error: pe,
                state_mean,
                state_sqrt: v, // the (symmetrized) variance; callers factor it
            },
            tmp,
        ))
    }
}

/// `gaussian_filter_bank.m`: (PredictedStateMean, PredictedStateVarianceSquareRoot,
/// StateVectorMean, StateVectorVarianceSquareRoot).
#[allow(clippy::too_many_arguments)]
fn bank(
    rf: &ReducedForm,
    obs: &[f64],
    mean: &[f64],
    sqrt_p: &Mat,
    qc: &Mat,
    hc: &Mat,
    o: &ParticleOptions,
    draws: &mut dyn Draws,
) -> Result<(Vec<f64>, Mat, Vec<f64>, Mat)> {
    let nx = rf.ns() + rf.nu();
    let sp = match for_approximation(o.proposal, nx, o) {
        Some(sp) => sp,
        None => {
            let n = o.number_of_particles;
            let nodes = draws.randn_mat(n, nx);
            SigmaPoints { nodes, wm: vec![1.0 / n as f64; n], wc: vec![1.0 / n as f64; n] }
        }
    };
    let square_root = o.proposal != Approximation::Unscented;
    let (b, _) = sigma_update(rf, obs, mean, sqrt_p, qc, None, hc, &sp, square_root)?;
    let pred_sqrt = chol_lower(&b.pred_state_variance)
        .ok_or_else(|| anyhow!("chol: the predicted state variance is not positive definite"))?;
    let s = if square_root { b.state_sqrt } else { rrc_lower(&b.state_sqrt)? };
    Ok((b.pred_state_mean, pred_sqrt, b.state_mean, s))
}

/// `sqrt(H)` (elementwise, as `gaussian_densities` / `gaussian_mixture_densities` use it).
pub(crate) fn sqrt_elementwise(h: &Mat) -> Mat {
    h.map(|v| v.sqrt())
}

pub fn filter(
    rf: &ReducedForm,
    y: &Mat,
    start: usize,
    o: &ParticleOptions,
    draws: &mut dyn Draws,
) -> Result<FilterOutput> {
    if o.pruning {
        bail!("gaussian_filter: pruning is not yet implemented.");
    }
    let t_len = y.ncols();
    let ns = rf.ns();
    let pp = rf.pp();
    let n = o.number_of_particles;
    let dist = for_approximation(o.distribution, ns, o);
    let hc = rrc_lower(&rf.h)?;
    let hsqrt = sqrt_elementwise(&rf.h);
    let mut mean = rf.state_mean.clone();
    let mut s = rrc_lower(&rf.state_variance)?;
    let qc = rrc_lower(&rf.q)?;
    let mut lik = vec![f64::NAN; t_len];
    for t in 0..t_len {
        let obs: Vec<f64> = (0..pp).map(|i| y[(i, t)]).collect();
        let (pmean, psqrt, m1, s1) = bank(rf, &obs, &mean, &s, &qc, &hc, o, draws)?;
        mean = m1;
        s = s1;
        let (particles, mut sw) = match &dist {
            Some(sp) => {
                let k = sp.len();
                let p = Mat::from_fn(ns, k, |i, c| {
                    mean[i] + (0..s.ncols()).map(|j| s[(i, j)] * sp.nodes[(c, j)]).sum::<f64>()
                });
                let iw = densities(rf, &obs, &mean, &s, &pmean, &psqrt, &p, &hsqrt);
                let sw: Vec<f64> = iw.iter().zip(&sp.wm).map(|(a, w)| w * a).collect();
                (p, sw)
            }
            None => {
                let z = draws.randn_mat(s.ncols(), n);
                let mut p = &s * z;
                for c in 0..n {
                    for i in 0..ns {
                        p[(i, c)] += mean[i];
                    }
                }
                let iw = densities(rf, &obs, &mean, &s, &pmean, &psqrt, &p, &hsqrt);
                (p, iw.iter().map(|v| v / n as f64).collect())
            }
        };
        for v in sw.iter_mut() {
            *v += 1e-6;
        }
        let ssum: f64 = sw.iter().sum();
        lik[t] = ssum.ln();
        for v in sw.iter_mut() {
            *v /= ssum;
        }
        let mut particles = particles;
        if dist.is_none()
            && ((o.resampling == ResamplingStatus::Generic && neff(&sw) < o.resampling_threshold * t_len as f64)
                || o.resampling == ResamplingStatus::Systematic)
        {
            particles = resample(&particles, &sw, o, draws)?;
            sw = vec![1.0 / n as f64; n];
        }
        mean = weighted_mean(&particles, &sw);
        let k = particles.ncols();
        let mut cov = Mat::zeros(ns, ns);
        for c in 0..k {
            for i in 0..ns {
                let di = particles[(i, c)] - mean[i];
                for j in 0..ns {
                    cov[(i, j)] += sw[c] * di * (particles[(j, c)] - mean[j]);
                }
            }
        }
        s = rrc_lower(&cov)?;
    }
    Ok(FilterOutput::from_lik(lik, start))
}

/// `gaussian_densities` (subfunction of `gaussian_filter.m`).
#[allow(clippy::too_many_arguments)]
fn densities(
    rf: &ReducedForm,
    obs: &[f64],
    mean: &[f64],
    s: &Mat,
    pmean: &[f64],
    psqrt: &Mat,
    particles: &Mat,
    hsqrt: &Mat,
) -> Vec<f64> {
    let proposal = probability2(mean, s, particles);
    let prior = probability2(pmean, psqrt, particles);
    let yt = measurement(rf, particles);
    let likelihood = probability2(obs, hsqrt, &yt);
    likelihood.iter().zip(&prior).zip(&proposal).map(|((l, p), q)| l * p / q).collect()
}
