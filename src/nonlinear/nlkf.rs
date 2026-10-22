// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `nonlinear_kalman_filter.m` (unscented / cubature / Monte Carlo Kalman filter).
//!
//! As in Dynare 7.2, `lik(t)` is `log(sum(probability2(y_t, chol(H)', y_i) .* weights))`, the
//! sigma-point average of the measurement density at the predicted observables (not the
//! Gaussian predictive density of the Kalman update).

use super::gf::sigma_update;
use super::linalg::{chol_lower, probability2, select_rows};
use super::rng::Draws;
use super::sigma_points::{SigmaPoints, for_approximation};
use super::{Approximation, FilterOutput, ParticleOptions, ReducedForm};
use crate::linalg::Mat;
use anyhow::{Result, anyhow, bail};

pub fn filter(
    rf: &ReducedForm,
    y: &Mat,
    start: usize,
    o: &ParticleOptions,
    draws: &mut dyn Draws,
) -> Result<FilterOutput> {
    if o.pruning {
        bail!("nonlinear_kalman_filter: pruning is not yet implemented.");
    }
    let t_len = y.ncols();
    let ns = rf.ns();
    let pp = rf.pp();
    let nx = ns + rf.nu();
    let sp = match for_approximation(o.proposal, nx, o) {
        Some(sp) => sp,
        None => {
            let n = o.number_of_particles;
            SigmaPoints { nodes: draws.randn_mat(n, nx), wm: vec![1.0 / n as f64; n], wc: vec![1.0 / n as f64; n] }
        }
    };
    let square_root = o.proposal != Approximation::Unscented;
    let hc = chol_lower(&rf.h).ok_or_else(|| anyhow!("chol: H is not positive definite"))?;
    let qc = chol_lower(&rf.q).ok_or_else(|| anyhow!("chol: Q is not positive definite"))?;
    let mut mean = rf.state_mean.clone();
    let mut s = chol_lower(&rf.state_variance)
        .ok_or_else(|| anyhow!("chol: the initial state variance is not positive definite"))?;
    let mut lik = vec![f64::NAN; t_len];
    for t in 0..t_len {
        let obs: Vec<f64> = (0..pp).map(|i| y[(i, t)]).collect();
        let (b, tmp) = sigma_update(rf, &obs, &mean, &s, &qc, None, &hc, &sp, square_root)?;
        mean = b.state_mean;
        if square_root {
            s = b.state_sqrt;
        } else {
            match chol_lower(&b.state_sqrt) {
                Some(l) => s = l,
                None => {
                    lik[t] = f64::NEG_INFINITY;
                    return Ok(FilterOutput { lik_total: f64::NEG_INFINITY, lik });
                }
            }
            if chol_lower(b.pred_obs_variance.as_ref().unwrap()).is_none() {
                lik[t] = f64::NEG_INFINITY;
                return Ok(FilterOutput { lik_total: f64::NEG_INFINITY, lik });
            }
        }
        let dens = probability2(&obs, &hc, &select_rows(&tmp, &rf.mf1));
        let v: f64 = dens.iter().zip(&sp.wm).map(|(d, w)| d * w).sum();
        lik[t] = v.ln();
    }
    Ok(FilterOutput::from_lik(lik, start))
}
