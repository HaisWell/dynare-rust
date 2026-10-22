// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `conditional_particle_filter.m` (each particle moved with its own sigma-point
//! Kalman update, `conditional_filter_proposal`). The particles' updates run in parallel; their
//! draws are taken beforehand, in MATLAB's order (particle by particle).

use super::gf::sigma_update;
use super::law_of_motion::measurement;
use super::linalg::{chol_lower, neff, probability2, select_rows, weighted_mean};
use super::resampling::resample;
use super::rng::Draws;
use super::sigma_points::{SigmaPoints, for_approximation};
use super::{Approximation, CpfWeights, FilterOutput, ParticleOptions, ReducedForm, ResamplingStatus};
use crate::linalg::Mat;
use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;

/// One `conditional_filter_proposal`: (new particle, weight, flag) or a MATLAB error.
#[allow(clippy::too_many_arguments)]
fn proposal(
    rf: &ReducedForm,
    y: &[f64],
    state: &[f64],
    sw: f64,
    qc: &Mat,
    hc: &Mat,
    sp: &SigmaPoints,
    z: &[f64],
    o: &ParticleOptions,
) -> Result<(Vec<f64>, f64, bool)> {
    let ns = rf.ns();
    let pp = rf.pp();
    let square_root = o.proposal != Approximation::Unscented;
    let zero = Mat::zeros(ns, ns);
    let (b, tmp) = sigma_update(rf, y, state, &zero, qc, None, hc, sp, square_root)?;
    let fail = || Ok((vec![0.0; ns], 0.0, true));
    let s = if square_root {
        b.state_sqrt.clone()
    } else {
        match chol_lower(&b.state_sqrt) {
            Some(l) => l,
            None => return fail(),
        }
    };
    let zeros = vec![0.0; pp];
    let mut weight = f64::NAN;
    if o.cpf_weights == CpfWeights::AmisanoTristani {
        let e = Mat::from_column_slice(pp, 1, &b.prediction_error);
        let povs = if square_root {
            b.pred_obs_sqrt.clone()
        } else {
            chol_lower(b.pred_obs_variance.as_ref().unwrap())
                .ok_or_else(|| anyhow!("chol: predicted observed variance"))?
        };
        weight = sw * probability2(&zeros, &povs, &e)[0];
    }
    let prop: Vec<f64> =
        (0..ns).map(|i| b.state_mean[i] + (0..s.ncols()).map(|j| s[(i, j)] * z[j]).sum::<f64>()).collect();
    if o.cpf_weights == CpfWeights::MurrayJonesParslow {
        // cubature / Monte Carlo: Dynare's PredictedStateVariance is dState*dState' (K x K)
        let psv = if square_root {
            let ts = select_rows(&tmp, &rf.mf0);
            let psm = weighted_mean(&ts, &sp.wm);
            let k = sp.len();
            let ds = Mat::from_fn(k, ns, |c, i| (ts[(i, c)] - psm[i]) * sp.wm[c].sqrt());
            &ds * ds.transpose()
        } else {
            b.pred_state_variance.clone()
        };
        let psv = (&psv + psv.transpose()) * 0.5;
        let Some(psqrt) = chol_lower(&psv) else { return fail() };
        if psqrt.nrows() != ns {
            bail!(
                "conditional_particle_filter: dimension mismatch in probability2 (cubature proposal with murrayjonesparslow)"
            );
        }
        let x = Mat::from_column_slice(ns, 1, &prop);
        let prior = probability2(&b.pred_state_mean, &psqrt, &x)[0];
        let posterior = probability2(&b.state_mean, &s, &x)[0];
        let likelihood = probability2(y, hc, &measurement(rf, &x))[0];
        weight = sw * likelihood * (prior / posterior);
    }
    Ok((prop, weight, false))
}

pub fn filter(
    rf: &ReducedForm,
    y: &Mat,
    start: usize,
    o: &ParticleOptions,
    draws: &mut dyn Draws,
) -> Result<FilterOutput> {
    if o.pruning {
        bail!("conditional_particle_filter: pruning is not yet implemented.");
    }
    let t_len = y.ncols();
    let n = o.number_of_particles;
    let ns = rf.ns();
    let nu = rf.nu();
    let pp = rf.pp();
    let hc = chol_lower(&rf.h).ok_or_else(|| anyhow!("chol: H is not positive definite"))?;
    let qc = chol_lower(&rf.q).ok_or_else(|| anyhow!("chol: Q is not positive definite"))?;
    let s0 = chol_lower(&rf.state_variance).ok_or_else(|| anyhow!("chol: initial state variance"))?;
    // sigma points over the shocks only (the state part of the nodes is zero)
    let pad = |sp: SigmaPoints| -> SigmaPoints {
        let k = sp.len();
        let nodes = Mat::from_fn(k, ns + nu, |r, c| if c < ns { 0.0 } else { sp.nodes[(r, c - ns)] });
        SigmaPoints { nodes, wm: sp.wm, wc: sp.wc }
    };
    let fixed = for_approximation(o.proposal, nu, o).map(pad);
    let mc_n = n / 10;
    let z = draws.randn_mat(s0.ncols(), n);
    let mut particles = &s0 * z;
    for c in 0..n {
        for i in 0..ns {
            particles[(i, c)] += rf.state_mean[i];
        }
    }
    let mut sw = vec![1.0 / n as f64; n];
    let mut lik = vec![f64::NAN; t_len];
    for t in 0..t_len {
        let yt: Vec<f64> = (0..pp).map(|i| y[(i, t)]).collect();
        // draws in MATLAB's order: per particle, (Monte Carlo nodes), then randn(ns, 1)
        let mut per: Vec<(Option<SigmaPoints>, Vec<f64>)> = Vec::with_capacity(n);
        for _ in 0..n {
            let mc = fixed.is_none().then(|| {
                let w = 1.0 / mc_n as f64;
                pad(SigmaPoints { nodes: draws.randn_mat(mc_n, nu), wm: vec![w; mc_n], wc: vec![w; mc_n] })
            });
            let mut zz = vec![0.0; ns];
            draws.randn(&mut zz);
            per.push((mc, zz));
        }
        let res: Vec<Result<(Vec<f64>, f64, bool)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let sp = per[i].0.as_ref().or(fixed.as_ref()).unwrap();
                let st: Vec<f64> = (0..ns).map(|r| particles[(r, i)]).collect();
                proposal(rf, &yt, &st, sw[i], &qc, &hc, sp, &per[i].1, o)
            })
            .collect();
        let mut any_flag = false;
        for (i, r) in res.into_iter().enumerate() {
            let (p, w, f) = r?;
            any_flag |= f;
            for k in 0..ns {
                particles[(k, i)] = p[k];
            }
            sw[i] = w;
        }
        if any_flag {
            lik[t] = f64::NEG_INFINITY;
            return Ok(FilterOutput { lik_total: f64::NEG_INFINITY, lik });
        }
        let s: f64 = sw.iter().sum();
        lik[t] = s.ln();
        for v in sw.iter_mut() {
            *v /= s;
        }
        if (o.resampling == ResamplingStatus::Generic && neff(&sw) < o.resampling_threshold * t_len as f64)
            || o.resampling == ResamplingStatus::Systematic
        {
            particles = resample(&particles, &sw, o, draws)?;
            sw = vec![1.0 / n as f64; n];
        }
    }
    Ok(FilterOutput::from_lik(lik, start))
}
