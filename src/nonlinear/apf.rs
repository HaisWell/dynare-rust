// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `auxiliary_particle_filter.m` (Pitt & Shephard; first-stage weights from a
//! Student-t(3) kernel of the zero-shock prediction error, as Dynare 7.2 writes it).

use super::law_of_motion::iterate;
use super::linalg::{chol_lower, det, neff, rcond, select_rows};
use super::resampling::{gather_columns, resample, resample_indices};
use super::rng::Draws;
use super::sis::{initial_particles, mahalanobis, pruned_init};
use super::{FilterOutput, ParticleOptions, ReducedForm, ResamplingMethod, ResamplingStatus};
use crate::linalg::Mat;
use anyhow::{Result, anyhow, bail};

pub fn filter(
    rf: &ReducedForm,
    y: &Mat,
    start: usize,
    o: &ParticleOptions,
    draws: &mut dyn Draws,
) -> Result<FilterOutput> {
    if o.resampling_method == ResamplingMethod::Smooth {
        bail!("auxiliary_particle_filter: resampling_method=smooth is not supported.");
    }
    let t_len = y.ncols();
    let pp = rf.pp();
    let n = o.number_of_particles;
    let ns = rf.ns();
    let h = &rf.h;
    let qc = chol_lower(&rf.q).ok_or_else(|| anyhow!("chol: Q is not positive definite"))?;
    if rcond(h) < 1e-12 {
        return Ok(FilterOutput::nan(t_len));
    }
    let const_lik = (2.0 * std::f64::consts::PI).ln() * pp as f64 + det(h).ln();
    let mut weights = vec![1.0 / n as f64; n];
    let mut states = initial_particles(rf, n, draws)?;
    let pruning = o.pruning && rf.order > 1;
    if pruning && rf.order == 3 {
        bail!("auxiliary_particle_filter: pruning at order 3 fails in Dynare 7.2 (undefined ghx)");
    }
    let (mut states_, mf0_) = pruned_init(rf, &states);
    // Student-t(3) constant: exp(gammaln(2) - gammaln(1.5)) / sqrt(3*pi)
    let ddl = 3.0f64;
    let tconst =
        (statrs::function::gamma::ln_gamma((ddl + 1.0) / 2.0) - statrs::function::gamma::ln_gamma(ddl / 2.0)).exp();
    let mut lik = vec![f64::NAN; t_len];
    for t in 0..t_len {
        let yt: Vec<f64> = (0..pp).map(|i| y[(i, t)]).collect();
        let zero = Mat::zeros(rf.nu(), n);
        let tmp0 = iterate(rf, &states, &zero, pruning.then_some(&states_)).y;
        let z = mahalanobis(&yt, &select_rows(&tmp0, &rf.mf1), h);
        let mut tau: Vec<f64> = weights
            .iter()
            .zip(&z)
            .map(|(w, zz)| {
                w * (tconst / ((ddl * std::f64::consts::PI).sqrt() * (1.0 + zz * zz / ddl).powf((ddl + 1.0) / 2.0))
                    + 1e-99)
            })
            .collect();
        let st: f64 = tau.iter().sum();
        for v in tau.iter_mut() {
            *v /= st;
        }
        let indx = resample_indices(&tau, o, draws)?;
        let w1: Vec<f64> = indx.iter().map(|&i| weights[i] / tau[i]).collect();
        let eps = &qc * draws.randn_mat(rf.nu(), n);
        let sel = gather_columns(&states, &indx);
        let sel_ = pruning.then(|| gather_columns(&states_, &indx));
        let it = iterate(rf, &sel, &eps, sel_.as_ref());
        if pruning {
            states_ = select_rows(it.pruned.as_ref().unwrap(), &mf0_);
        }
        states = select_rows(&it.y, &rf.mf0);
        let q = mahalanobis(&yt, &select_rows(&it.y, &rf.mf1), h);
        let w2: Vec<f64> = w1.iter().zip(&q).map(|(w, qq)| w * ((-0.5 * (const_lik + qq)).exp() + 1e-99)).collect();
        let s2: f64 = w2.iter().sum();
        lik[t] = (s2 / n as f64).ln();
        weights = w2.iter().map(|w| w / s2).collect();
        if (o.resampling == ResamplingStatus::Generic && neff(&weights) < o.resampling_threshold * t_len as f64)
            || o.resampling == ResamplingStatus::Systematic
        {
            if pruning {
                let both = Mat::from_fn(2 * ns, n, |i, c| if i < ns { states[(i, c)] } else { states_[(i - ns, c)] });
                let r = resample(&both, &weights, o, draws)?;
                states = r.rows(0, ns).into_owned();
                states_ = r.rows(ns, ns).into_owned();
            } else {
                states = resample(&states, &weights, o, draws)?;
            }
            weights = vec![1.0 / n as f64; n];
        }
    }
    Ok(FilterOutput::from_lik(lik, start))
}
