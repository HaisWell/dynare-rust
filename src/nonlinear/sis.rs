// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `sequential_importance_particle_filter.m` (bootstrap particle filter).

use super::law_of_motion::iterate;
use super::linalg::{chol_lower, det, mldivide, neff, rcond, select_rows};
use super::resampling::resample;
use super::rng::Draws;
use super::{FilterOutput, ParticleOptions, ReducedForm, ResamplingStatus};
use crate::linalg::Mat;
use anyhow::{Result, anyhow};

/// Initial particles `chol(V)' * randn(rank, N) + mean` and the pruned state space of
/// Dynare's filters (`StateVectors_`, `mf0_`).
pub(crate) fn initial_particles(rf: &ReducedForm, n: usize, draws: &mut dyn Draws) -> Result<Mat> {
    let s = chol_lower(&rf.state_variance)
        .ok_or_else(|| anyhow!("chol: the initial state variance is not positive definite"))?;
    let z = draws.randn_mat(s.ncols(), n);
    let mut x = &s * z;
    for c in 0..n {
        for i in 0..rf.ns() {
            x[(i, c)] += rf.state_mean[i];
        }
    }
    Ok(x)
}

/// `(StateVectors_, mf0_)` for pruning (order 2: same states; order 3: three stacked copies).
pub(crate) fn pruned_init(rf: &ReducedForm, states: &Mat) -> (Mat, Vec<usize>) {
    if rf.order == 3 {
        let ns = rf.ns();
        let n = states.ncols();
        let st = Mat::from_fn(3 * ns, n, |i, c| states[(i % ns, c)]);
        let m = rf.m();
        let mf0: Vec<usize> = (0..3).flat_map(|k| rf.mf0.iter().map(move |&j| j + k * m)).collect();
        (st, mf0)
    } else {
        (states.clone(), rf.mf0.clone())
    }
}

/// Sums of `PredictionError .* (H \ PredictionError)` over the observables, per particle;
/// `tmp` holds the particles' restricted variables, `mf1` the observed rows.
pub(crate) fn mahalanobis_rows(y: &[f64], tmp: &Mat, mf1: &[usize], h: &Mat) -> Vec<f64> {
    let pp = mf1.len();
    let n = tmp.ncols();
    let diagonal = (0..pp).all(|i| (0..pp).all(|j| i == j || h[(i, j)] == 0.0));
    if diagonal {
        let hd: Vec<f64> = (0..pp).map(|i| h[(i, i)]).collect();
        return (0..n)
            .map(|c| {
                let mut q = 0.0;
                for i in 0..pp {
                    let pe = y[i] - tmp[(mf1[i], c)];
                    q += pe * (pe / hd[i]);
                }
                q
            })
            .collect();
    }
    let pe = Mat::from_fn(pp, n, |i, c| y[i] - tmp[(mf1[i], c)]);
    let hpe = mldivide(h, &pe);
    (0..n).map(|c| (0..pp).map(|i| pe[(i, c)] * hpe[(i, c)]).sum()).collect()
}

/// [`mahalanobis_rows`] with the observables already selected (pp x N).
pub(crate) fn mahalanobis(y: &[f64], obs: &Mat, h: &Mat) -> Vec<f64> {
    let rows: Vec<usize> = (0..obs.nrows()).collect();
    mahalanobis_rows(y, obs, &rows, h)
}

pub fn filter(
    rf: &ReducedForm,
    y: &Mat,
    start: usize,
    o: &ParticleOptions,
    draws: &mut dyn Draws,
) -> Result<FilterOutput> {
    let t_len = y.ncols();
    let pp = rf.pp();
    let n = o.number_of_particles;
    let h = &rf.h;
    let const_lik = (2.0 * std::f64::consts::PI).ln() * pp as f64 + det(h).ln();
    if rcond(h) < 1e-12 {
        return Ok(FilterOutput::nan(t_len));
    }
    let qc = chol_lower(&rf.q).ok_or_else(|| anyhow!("chol: Q is not positive definite"))?;
    let mut weights = vec![1.0 / n as f64; n];
    let mut states = initial_particles(rf, n, draws)?;
    let pruning = o.pruning && rf.order > 1;
    let (mut states_, mf0_) = pruned_init(rf, &states);
    let mut lik = vec![f64::NAN; t_len];
    for t in 0..t_len {
        let eps = &qc * draws.randn_mat(rf.nu(), n);
        let it = iterate(rf, &states, &eps, pruning.then_some(&states_));
        let tmp = it.y;
        let yt: Vec<f64> = (0..pp).map(|i| y[(i, t)]).collect();
        let q = mahalanobis_rows(&yt, &tmp, &rf.mf1, h);
        let lnw: Vec<f64> = q.iter().map(|v| -0.5 * (const_lik + v)).collect();
        let dfac = lnw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let wtilde: Vec<f64> = weights.iter().zip(&lnw).map(|(w, l)| w * (l - dfac).exp()).collect();
        let sw: f64 = wtilde.iter().sum();
        lik[t] = sw.ln() + dfac;
        weights = wtilde.iter().map(|w| w / sw).collect();
        let do_resample = (o.resampling == ResamplingStatus::Generic
            && neff(&weights) < o.resampling_threshold * t_len as f64)
            || o.resampling == ResamplingStatus::Systematic;
        if do_resample {
            if pruning {
                let pr = it.pruned.as_ref().unwrap();
                let mut both = select_rows(&tmp, &rf.mf0);
                let p2 = select_rows(pr, &mf0_);
                let ns = rf.ns();
                both = Mat::from_fn(ns + p2.nrows(), n, |i, c| if i < ns { both[(i, c)] } else { p2[(i - ns, c)] });
                let r = resample(&both, &weights, o, draws)?;
                states = r.rows(0, ns).into_owned();
                states_ = r.rows(ns, r.nrows() - ns).into_owned();
            } else {
                states = resample(&select_rows(&tmp, &rf.mf0), &weights, o, draws)?;
            }
            weights = vec![1.0 / n as f64; n];
        } else if o.resampling == ResamplingStatus::None {
            states = select_rows(&tmp, &rf.mf0);
            if pruning {
                states_ = select_rows(it.pruned.as_ref().unwrap(), &mf0_);
            }
        }
        // (Dynare 7.2: with resampling = generic and no resampling needed, the particles
        // are not propagated -- neither branch updates StateVectors. Ported as is.)
    }
    Ok(FilterOutput::from_lik(lik, start))
}
