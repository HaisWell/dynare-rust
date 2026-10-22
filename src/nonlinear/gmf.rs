// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `gaussian_mixture_filter.m` + `gaussian_mixture_filter_bank.m` +
//! `gaussian_mixture_densities.m` + `probability.m` (Gaussian-mixture filter, sigma-point
//! distribution approximation; Dynare 7.2 rejects the Monte Carlo approximation, so the
//! `importance_sampling` / `fit_gaussian_mixture` branch is unreachable and not ported).

use super::gf::{rrc_lower, sigma_update, sqrt_elementwise};
use super::law_of_motion::measurement;
use super::linalg::{det, mldivide, probability2};
use super::resampling::resample_indices;
use super::rng::Draws;
use super::sigma_points::{SigmaPoints, for_approximation, spherical_radial, unscented};
use super::{Approximation, FilterOutput, ParticleOptions, ReducedForm};
use crate::linalg::Mat;
use anyhow::{Result, bail};

/// One Gaussian component.
#[derive(Clone)]
struct Comp {
    mu: Vec<f64>,
    sqrt: Mat,
    w: f64,
}

/// `C` output of `probability.m`: the mixture density at the columns of `x`.
fn mixture_density(comps: &[Comp], x: &Mat) -> Vec<f64> {
    let (dim, nov) = (x.nrows(), x.ncols());
    let normfact = (2.0 * std::f64::consts::PI).powf(dim as f64 / 2.0);
    let mut c = vec![0.0; nov];
    for comp in comps {
        let xx = Mat::from_fn(dim, nov, |i, j| x[(i, j)] - comp.mu[i]);
        let foo = mldivide(&comp.sqrt, &xx);
        let pd: f64 = (0..dim).map(|i| comp.sqrt[(i, i)]).product();
        for (j, cj) in c.iter_mut().enumerate() {
            let q: f64 = (0..dim).map(|i| foo[(i, j)] * foo[(i, j)]).sum();
            let l = (-0.5 * q).exp() / (normfact * pd).abs() + 1e-99;
            *cj += comp.w * l;
        }
    }
    c.iter().map(|v| v + 1e-99).collect()
}

pub fn filter(
    rf: &ReducedForm,
    y: &Mat,
    start: usize,
    o: &ParticleOptions,
    draws: &mut dyn Draws,
) -> Result<FilterOutput> {
    if o.pruning {
        bail!("gaussian_mixture_filter: pruning is not yet implemented.");
    }
    let t_len = y.ncols();
    let ns = rf.ns();
    let nu = rf.nu();
    let pp = rf.pp();
    let g_n = o.mixture_state_variables;
    let dist = match o.distribution {
        Approximation::MonteCarlo => bail!("The montecarlo approximation for the distribution is not implemented."),
        a => for_approximation(a, ns, o).unwrap(),
    };
    let hc = rrc_lower(&rf.h)?;
    let hsqrt = sqrt_elementwise(&rf.h);
    let qc = rrc_lower(&rf.q)?;
    let temp = rrc_lower(&rf.state_variance)?;
    if temp.ncols() != ns {
        bail!("gaussian_mixture_filter: rank-deficient initial state variance");
    }
    let mut states: Vec<Comp> = (0..g_n)
        .map(|g| {
            let off = (g as f64 - (g_n as f64 - 1.0) / 2.0) / 10.0;
            Comp {
                mu: (0..ns).map(|i| rf.state_mean[i] + temp[(i, i)] * off).collect(),
                sqrt: &temp / (g_n as f64).sqrt(),
                w: 1.0 / g_n as f64,
            }
        })
        .collect();
    // structural shocks mixture
    let (ss_mu, ss_sqrt, ss_w): (Vec<Vec<f64>>, Vec<Mat>, Vec<f64>) = if o.mixture_structural_shocks == 0 {
        (vec![vec![0.0; nu]], vec![qc.clone()], vec![1.0])
    } else {
        let sp = match o.proposal {
            Approximation::Cubature => spherical_radial(nu),
            Approximation::Unscented => unscented(nu, &o.unscented),
            Approximation::MonteCarlo => bail!("The montecarlo approximation for the proposal is not implemented."),
        };
        let mus: Vec<Vec<f64>> = (0..sp.len())
            .map(|r| (0..nu).map(|i| (0..nu).map(|j| qc[(i, j)] * sp.nodes[(r, j)]).sum()).collect())
            .collect();
        let sq: Vec<Mat> = (0..sp.len())
            .map(|r| if o.mixture_structural_shocks == 1 { qc.clone() } else { &qc / sp.wm[r].sqrt() })
            .collect();
        (mus, sq, sp.wm.clone())
    };
    let i_n = ss_w.len();
    let nodes3 = match o.proposal {
        Approximation::Cubature => spherical_radial(ns + nu),
        Approximation::Unscented => unscented(ns + nu, &o.unscented),
        Approximation::MonteCarlo => bail!("The montecarlo approximation for the proposal is not implemented."),
    };
    let square_root = o.proposal == Approximation::Cubature;
    let gp = g_n * i_n;
    let normfact_o = (2.0 * std::f64::consts::PI).powf(0.5 * pp as f64);
    let mut lik = vec![f64::NAN; t_len];
    for t in 0..t_len {
        let obs: Vec<f64> = (0..pp).map(|i| y[(i, t)]).collect();
        let mut prior: Vec<Option<Comp>> = vec![None; gp];
        let mut post: Vec<Option<Comp>> = vec![None; gp];
        for i in 0..i_n {
            for (g, st) in states.iter().enumerate() {
                let gprime = g + i * g_n;
                // gaussian_mixture_filter_bank
                let sp = SigmaPoints { nodes: nodes3.nodes.clone(), wm: nodes3.wm.clone(), wc: nodes3.wc.clone() };
                let (b, _) =
                    sigma_update(rf, &obs, &st.mu, &st.sqrt, &ss_sqrt[i], Some(&ss_mu[i]), &hc, &sp, square_root)?;
                let pe = &b.prediction_error;
                let (ipov, sqrdet, s_post) = if square_root {
                    let ipovs =
                        b.pred_obs_sqrt.clone().try_inverse().unwrap_or_else(|| Mat::from_element(pp, pp, f64::NAN));
                    let ipov = ipovs.transpose() * &ipovs;
                    let sqrdet = 1.0 / det(&ipov).sqrt();
                    (ipov, sqrdet, b.state_sqrt.clone())
                } else {
                    let pov = b.pred_obs_variance.clone().unwrap();
                    let sqrdet = det(&pov).sqrt();
                    let ipov = pov.try_inverse().unwrap_or_else(|| Mat::from_element(pp, pp, f64::NAN));
                    (ipov, sqrdet, rrc_lower(&b.state_sqrt)?)
                };
                let mut q = 0.0;
                for a in 0..pp {
                    for c in 0..pp {
                        q += pe[a] * ipov[(a, c)] * pe[c];
                    }
                }
                let data_lik = (-0.5 * q).exp() / (normfact_o * sqrdet).abs() + 1e-99;
                let wprior = st.w * ss_w[i];
                let psqrt = rrc_lower(&b.pred_state_variance)?;
                if psqrt.ncols() != ns || s_post.ncols() != ns {
                    bail!("gaussian_mixture_filter: rank-deficient component covariance");
                }
                prior[gprime] = Some(Comp { mu: b.pred_state_mean.clone(), sqrt: psqrt, w: wprior });
                post[gprime] = Some(Comp { mu: b.state_mean.clone(), sqrt: s_post, w: wprior * data_lik });
            }
        }
        let mut prior: Vec<Comp> = prior.into_iter().map(Option::unwrap).collect();
        let mut post: Vec<Comp> = post.into_iter().map(Option::unwrap).collect();
        let sp: f64 = prior.iter().map(|c| c.w).sum();
        prior.iter_mut().for_each(|c| c.w /= sp);
        let sq: f64 = post.iter().map(|c| c.w).sum();
        post.iter_mut().for_each(|c| c.w /= sq);
        let mut sw = vec![0.0; gp];
        for (k, swk) in sw.iter_mut().enumerate() {
            let c = &post[k];
            let np = dist.len();
            let particles = Mat::from_fn(ns, np, |i, r| {
                c.mu[i] + (0..ns).map(|j| c.sqrt[(i, j)] * dist.nodes[(r, j)]).sum::<f64>()
            });
            let pr = mixture_density(&prior, &particles);
            let pq = mixture_density(&post, &particles);
            let ll = probability2(&obs, &hsqrt, &measurement(rf, &particles));
            *swk = (0..np).map(|r| c.w * dist.wm[r] * (ll[r] * pr[r] / pq[r])).sum();
        }
        let s: f64 = sw.iter().sum();
        lik[t] = s.ln();
        sw.iter_mut().for_each(|v| *v /= s);
        let mut u = vec![0.0; gp];
        draws.rand(&mut u);
        let mut order: Vec<usize> = (0..gp).collect();
        order.sort_by(|&a, &b| u[a].partial_cmp(&u[b]).unwrap_or(std::cmp::Ordering::Equal));
        let indx = resample_indices(&sw, o, draws)?;
        states = order[..g_n]
            .iter()
            .map(|&k| {
                let c = &post[indx[k]];
                Comp { mu: c.mu.clone(), sqrt: c.sqrt.clone(), w: 1.0 / g_n as f64 }
            })
            .collect();
    }
    Ok(FilterOutput::from_lik(lik, start))
}
