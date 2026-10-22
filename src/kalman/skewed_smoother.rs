// SPDX-License-Identifier: GPL-3.0-or-later
//! [estimation completion] Port of `kalman/kalman_smoother_pruned_skewed.m` (Dynare 7.2): the
//! pruned skewed Kalman smoother that `DsgeSmoother.m` runs with `kalman_algo = 5`.
//!
//! Forward pass: the skewed filter without pruning (all CSN skewness dimensions kept); backward
//! pass: the CSN distributions of the smoothed states and of the smoothed shocks (from the
//! joint distribution of `[x_t; x_{t+1}]` mapped by `A = [-(R'R)^-1 R' G, (R'R)^-1 R']`),
//! pruned with `prune_tol` and summarized by their means ([`csn_mean`]). As in Dynare, only
//! the smoothed states (`alphahat`), shocks (`etahat`), measurement errors (`Y - F alphahat`)
//! and the smoothed initial state are computed: the updated states stay NaN (`x_t_t` is
//! never filled in the MATLAB code) and there are no k-step-ahead or uncertainty objects.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)] // index loops mirror the MATLAB code

use anyhow::{Result, bail};

use super::csn::{blkdiag, csn_mean, pinv, vcat, vstack};
use super::skewed::{Csn, Mvnlogcdf, csn_prune_distribution, sym};
use crate::linalg::{Mat, Vector, mul};

/// What [`kalman_smoother_pruned_skewed`] returns.
pub struct SkewedSmoother {
    /// Smoothed states `x_t_T`, mm x nobs (means of the smoothed CSN distributions).
    pub alphahat: Mat,
    /// Smoothed shocks `eta_t_T`, n_exo x nobs.
    pub etahat: Mat,
    /// `Y - F*x_t_T`, pp x nobs.
    pub epsilonhat: Mat,
    /// Smoothed initial state `x_0_T`.
    pub alphahat0: Vector,
}

fn inv_or_nan(a: &Mat) -> Mat {
    crate::linalg::solve(a, &Mat::identity(a.nrows(), a.nrows()))
        .unwrap_or_else(|| Mat::from_element(a.nrows(), a.ncols(), f64::NAN))
}

fn rcond(a: &Mat) -> f64 {
    let (lu, _, info) = crate::linalg::lu(a);
    if info > 0 {
        return 0.0;
    }
    crate::linalg::rcond_from_lu(&lu, crate::linalg::norm1(a))
}

/// `csn_prune_distribution` then `csn_mean`.
fn pruned_mean(
    mu: &Vector,
    sigma: &Mat,
    gamma: &Mat,
    nu: &Vector,
    delta: &Mat,
    prune_tol: f64,
    cdf: Mvnlogcdf,
) -> Result<Vector> {
    let (g, n, d) = csn_prune_distribution(sigma, gamma, nu, delta, prune_tol);
    csn_mean(mu, sigma, &g, &n, &d, cdf)
}

/// Port of kalman_smoother_pruned_skewed.m. `y`: demeaned data (pp x nobs; NaN propagates as
/// in MATLAB); `mu0`, `sigma0`: the initial state `a_initial` and `Pstar` (`Gamma_0 = 0`,
/// `nu_0 = 0`, `Delta_0 = I`); `z`: observed rows of the state (`F`); `csn`: the shocks'
/// CSN parameters (`M_.csn`); `h`: `M_.H`.
pub fn kalman_smoother_pruned_skewed(
    y: &Mat,
    mu0: &Vector,
    sigma0: &Mat,
    t: &Mat,
    r: &Mat,
    z: &[usize],
    csn: &Csn,
    h: &Mat,
    kalman_tol: f64,
    prune_tol: f64,
    cdf: Mvnlogcdf,
) -> Result<SkewedSmoother> {
    let (x_nbr, pp, n) = (t.nrows(), y.nrows(), y.ncols());
    let eta_nbr = csn.mu.len();
    if n < 2 {
        bail!("kalman_smoother_pruned_skewed: at least two observations are needed");
    }
    let g_singular = rcond(t) < kalman_tol;
    let mu_eta = r * &csn.mu;
    let sigma_eta = mul(&mul(r, false, &csn.sigma, false), false, r, true);
    let rtr_inv = inv_or_nan(&mul(r, true, r, false));
    let gamma_eta = &csn.gamma * &rtr_inv * r.transpose(); // Gamma_eta/(R'*R)*R'
    let ge_se = &gamma_eta * &sigma_eta;
    let delta22_common = &csn.delta + &ge_se * gamma_eta.transpose();
    let delta_eta = csn.delta.clone();
    // A = [-inv(R'*R)*R'*G, inv(R'*R)*R']
    let irt = &rtr_inv * r.transpose();
    let mut a_mat = Mat::zeros(eta_nbr, 2 * x_nbr);
    a_mat.view_mut((0, 0), (eta_nbr, x_nbr)).copy_from(&(-(&irt * t)));
    a_mat.view_mut((0, x_nbr), (eta_nbr, x_nbr)).copy_from(&irt);
    let fsel = |a: &Mat| Mat::from_fn(pp, a.ncols(), |i, j| a[(z[i], j)]); // F*A
    let (gamma0, nu0, delta0) = (Mat::zeros(x_nbr, x_nbr), Vector::zeros(x_nbr), Mat::identity(x_nbr, x_nbr));

    // ---- forward pass (t = 1..obs_nbr; the extra prediction for obs_nbr+1 is unused) ----
    let mut mu_tm1 = Vec::with_capacity(n); // mu_t_tm1
    let mut sig_tm1 = Vec::with_capacity(n); // Sigma_t_tm1
    let mut mu_tt: Vec<Vector> = Vec::with_capacity(n);
    let mut sig_tt: Vec<Mat> = Vec::with_capacity(n);
    let mut gam_tt: Vec<Mat> = Vec::with_capacity(n);
    let mut nu_tt: Vec<Vector> = Vec::with_capacity(n);
    let mut del_tt: Vec<Mat> = Vec::with_capacity(n);
    for tt in 0..n {
        let (mu_p, sig_p, gam_p, nu_p, del_p) = if tt == 0 {
            (mu0, sigma0, &gamma0, &nu0, &delta0)
        } else {
            (&mu_tt[tt - 1], &sig_tt[tt - 1], &gam_tt[tt - 1], &nu_tt[tt - 1], &del_tt[tt - 1])
        };
        let gs = gam_p * sig_p;
        let gsgt = &gs * t.transpose();
        let mu_pred = t * mu_p + &mu_eta;
        let sig_pred = sym(&(mul(&mul(t, false, sig_p, false), false, t, true) + &sigma_eta));
        let isig = if g_singular { pinv(&sig_pred) } else { inv_or_nan(&sig_pred) };
        let d11 = del_p + &gs * gam_p.transpose() - &gsgt * &isig * gsgt.transpose();
        let d22 = &delta22_common - &ge_se * &isig * ge_se.transpose();
        let d12 = -(&gsgt * &isig * ge_se.transpose());
        let gam_pred = vstack(&gsgt, &ge_se) * &isig;
        let nu_pred = vcat(nu_p, &csn.nu);
        let q = d11.nrows();
        let mut del_pred = Mat::zeros(q + eta_nbr, q + eta_nbr);
        del_pred.view_mut((0, 0), (q, q)).copy_from(&d11);
        del_pred.view_mut((0, q), (q, eta_nbr)).copy_from(&d12);
        del_pred.view_mut((q, 0), (eta_nbr, q)).copy_from(&d12.transpose());
        del_pred.view_mut((q, q), (eta_nbr, eta_nbr)).copy_from(&d22);
        let del_pred = sym(&del_pred);
        let fs = fsel(&sig_pred);
        let omega = Mat::from_fn(pp, pp, |i, j| fs[(i, z[j])] + h[(i, j)]);
        let iomega = inv_or_nan(&omega);
        let k_gauss = Mat::from_fn(x_nbr, pp, |i, j| sig_pred[(i, z[j])]) * &iomega;
        let k_skewed = &gam_pred * &k_gauss;
        let pe = Vector::from_fn(pp, |i, _| y[(i, tt)] - mu_pred[z[i]]);
        mu_tt.push(&mu_pred + &k_gauss * &pe);
        sig_tt.push(sym(&(&sig_pred - &k_gauss * fs)));
        nu_tt.push(&nu_pred - &k_skewed * &pe);
        gam_tt.push(gam_pred);
        del_tt.push(del_pred);
        mu_tm1.push(mu_pred);
        sig_tm1.push(sig_pred);
    }

    // ---- backward pass ----
    let mut x_t_t_ = Mat::from_element(x_nbr, n, f64::NAN);
    let mut eta_t_t = Mat::from_element(eta_nbr, n, f64::NAN);
    let mut mu_t_tt: Vec<Vector> = vec![Vector::zeros(0); n]; // mu_t_T
    let mut sig_t_tt: Vec<Mat> = vec![Mat::zeros(0, 0); n]; // Sigma_t_T
    let last = n - 1;
    mu_t_tt[last] = mu_tt[last].clone();
    sig_t_tt[last] = sig_tt[last].clone();
    let nu_t_big = nu_tt[last].clone(); // nu_t_T{t} = nu_t_t{obs_nbr} for every t
    let x = pruned_mean(&mu_t_tt[last], &sig_t_tt[last], &gam_tt[last], &nu_t_big, &del_tt[last], prune_tol, cdf)?;
    x_t_t_.set_column(last, &x);
    let mut o_next = Mat::zeros(0, x_nbr); // O_t{t+1}
    let mut dtilde_next = Mat::zeros(0, 0); // Delta_tilde_t{t+1}
    for tt in (0..last).rev() {
        let jt = mul(&sig_tt[tt], false, t, true) * pinv(&sig_tm1[tt + 1]);
        let dmu = &mu_t_tt[tt + 1] - &mu_tm1[tt + 1];
        mu_t_tt[tt] = &mu_tt[tt] + &jt * &dmu;
        let mu_jt = vcat(&(&mu_tt[tt] + &jt * &dmu), &mu_t_tt[tt + 1]);
        let dsig = &sig_t_tt[tt + 1] - &sig_tm1[tt + 1];
        sig_t_tt[tt] = &sig_tt[tt] + &jt * &dsig * jt.transpose();
        let mut sig_jt = Mat::zeros(2 * x_nbr, 2 * x_nbr);
        sig_jt.view_mut((0, 0), (x_nbr, x_nbr)).copy_from(&(&sig_tt[tt] + &jt * &dsig * jt.transpose()));
        sig_jt.view_mut((0, x_nbr), (x_nbr, x_nbr)).copy_from(&(&jt * &sig_t_tt[tt + 1]));
        sig_jt.view_mut((x_nbr, 0), (x_nbr, x_nbr)).copy_from(&(&sig_t_tt[tt + 1] * jt.transpose()));
        sig_jt.view_mut((x_nbr, x_nbr), (x_nbr, x_nbr)).copy_from(&sig_t_tt[tt + 1]);
        let m_t = &sig_t_tt[tt + 1] * jt.transpose() * pinv(&sig_t_tt[tt]);
        let n_t = -(&gamma_eta * t) + &gamma_eta * &m_t;
        let first = tt == last - 1;
        let o_t = if first { n_t.clone() } else { vstack(&n_t, &(&o_next * &m_t)) };
        let inner = &sig_t_tt[tt + 1] - &m_t * &sig_t_tt[tt] * m_t.transpose();
        let gq = gam_tt[tt].nrows();
        let (dtilde_t, gam_jt) = if first {
            let dt = &delta_eta + &gamma_eta * &inner * gamma_eta.transpose();
            let mut gj = Mat::zeros(gq + eta_nbr, 2 * x_nbr);
            gj.view_mut((0, 0), (gq, x_nbr)).copy_from(&gam_tt[tt]);
            gj.view_mut((gq, 0), (eta_nbr, x_nbr)).copy_from(&(-(&gamma_eta * t)));
            gj.view_mut((gq, x_nbr), (eta_nbr, x_nbr)).copy_from(&gamma_eta);
            (dt, gj)
        } else {
            let tm = vstack(&gamma_eta, &o_next);
            let dt = blkdiag(&delta_eta, &dtilde_next) + &tm * &inner * tm.transpose();
            let on = o_next.nrows();
            let mut gj = Mat::zeros(gq + eta_nbr + on, 2 * x_nbr);
            gj.view_mut((0, 0), (gq, x_nbr)).copy_from(&gam_tt[tt]);
            gj.view_mut((gq, 0), (eta_nbr, x_nbr)).copy_from(&(-(&gamma_eta * t)));
            gj.view_mut((gq, x_nbr), (eta_nbr, x_nbr)).copy_from(&gamma_eta);
            gj.view_mut((gq + eta_nbr, x_nbr), (on, x_nbr)).copy_from(&o_next);
            (dt, gj)
        };
        let gam_t_big = vstack(&gam_tt[tt], &o_t);
        let del_t_big = blkdiag(&del_tt[tt], &dtilde_t);
        let del_jt = blkdiag(&blkdiag(&del_tt[tt], &delta_eta), &dtilde_next);
        let mu_yt = &a_mat * &mu_jt;
        let sig_yt = &a_mat * &sig_jt * a_mat.transpose();
        let gxsa = &gam_jt * &sig_jt * a_mat.transpose();
        let psig = pinv(&sig_yt);
        let gam_yt = &gxsa * &psig;
        let del_yt = &del_jt + &gam_jt * &sig_jt * gam_jt.transpose() - &gxsa * &psig * gxsa.transpose();
        let x = pruned_mean(&mu_t_tt[tt], &sig_t_tt[tt], &gam_t_big, &nu_t_big, &del_t_big, prune_tol, cdf)?;
        x_t_t_.set_column(tt, &x);
        let e = pruned_mean(&mu_yt, &sig_yt, &gam_yt, &nu_t_big, &del_yt, prune_tol, cdf)?;
        eta_t_t.set_column(tt + 1, &e);
        o_next = o_t;
        dtilde_next = dtilde_t;
    }
    // ---- period 0 ----
    let j0 = mul(sigma0, false, t, true) * pinv(&sig_tm1[0]);
    let dmu = &mu_t_tt[0] - &mu_tm1[0];
    let mu_0_t = mu0 + &j0 * &dmu;
    let mu_j0 = vcat(&(mu0 + &j0 * &dmu), &mu_t_tt[0]);
    let dsig = &sig_t_tt[0] - &sig_tm1[0];
    let sig_0_t = sigma0 + &j0 * &dsig * j0.transpose();
    let mut sig_j0 = Mat::zeros(2 * x_nbr, 2 * x_nbr);
    sig_j0.view_mut((0, 0), (x_nbr, x_nbr)).copy_from(&(sigma0 + &j0 * &dsig * j0.transpose()));
    sig_j0.view_mut((0, x_nbr), (x_nbr, x_nbr)).copy_from(&(&j0 * &sig_t_tt[0]));
    sig_j0.view_mut((x_nbr, 0), (x_nbr, x_nbr)).copy_from(&(&sig_t_tt[0] * j0.transpose()));
    sig_j0.view_mut((x_nbr, x_nbr), (x_nbr, x_nbr)).copy_from(&sig_t_tt[0]);
    let m_0 = &sig_t_tt[0] * j0.transpose() * pinv(&sig_0_t);
    let n_0 = -(&gamma_eta * t) + &gamma_eta * &m_0;
    let o_0 = vstack(&n_0, &(&o_next * &m_0));
    let tm = vstack(&gamma_eta, &o_next);
    let inner = &sig_t_tt[0] - &m_0 * &sig_0_t * m_0.transpose();
    let dtilde_0 = blkdiag(&delta_eta, &dtilde_next) + &tm * &inner * tm.transpose();
    let on = o_next.nrows();
    let mut gam_j0 = Mat::zeros(x_nbr + eta_nbr + on, 2 * x_nbr);
    gam_j0.view_mut((0, 0), (x_nbr, x_nbr)).copy_from(&gamma0);
    gam_j0.view_mut((x_nbr, 0), (eta_nbr, x_nbr)).copy_from(&(-(&gamma_eta * t)));
    gam_j0.view_mut((x_nbr, x_nbr), (eta_nbr, x_nbr)).copy_from(&gamma_eta);
    gam_j0.view_mut((x_nbr + eta_nbr, x_nbr), (on, x_nbr)).copy_from(&o_next);
    let gam_0_t = vstack(&gamma0, &o_0);
    let del_0_t = blkdiag(&delta0, &dtilde_0);
    let del_j0 = blkdiag(&blkdiag(&delta0, &delta_eta), &dtilde_next);
    let mu_y0 = &a_mat * &mu_j0;
    let sig_y0 = &a_mat * &sig_j0 * a_mat.transpose();
    let gxsa = &gam_j0 * &sig_j0 * a_mat.transpose();
    let psig = pinv(&sig_y0);
    let gam_y0 = &gxsa * &psig;
    let del_y0 = &del_j0 + &gam_j0 * &sig_j0 * gam_j0.transpose() - &gxsa * &psig * gxsa.transpose();
    let x0 = pruned_mean(&mu_0_t, &sig_0_t, &gam_0_t, &nu_t_big, &del_0_t, prune_tol, cdf)?;
    let e0 = pruned_mean(&mu_y0, &sig_y0, &gam_y0, &nu_t_big, &del_y0, prune_tol, cdf)?;
    eta_t_t.set_column(0, &e0);
    let epsilonhat = Mat::from_fn(pp, n, |i, j| y[(i, j)] - x_t_t_[(z[i], j)]);
    Ok(SkewedSmoother { alphahat: x_t_t_, etahat: eta_t_t, epsilonhat, alphahat0: x0 })
}
