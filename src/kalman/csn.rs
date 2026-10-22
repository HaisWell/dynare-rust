// SPDX-License-Identifier: GPL-3.0-or-later
//! [estimation completion] Closed skew normal (CSN) helpers of Dynare 7.2's pruned skewed
//! Kalman filter and smoother (`distributions/closed_skew_normal/`):
//!
//! * [`csn_mean`], `csn_mean.m` (Dominguez-Molina, Gonzalez-Farias and Gupta 2003, eq. (1)),
//!   with either CDF routine (`gaussian_log_mvncdf_mendell_elston` or Dynare's `mvncdf`);
//! * [`csn_statespace_linear_transform`], `csn_statespace_linear_transform.m` with
//!   `csn_singular_to_csn_proper.m` (`skewed_kalman_rank_deficiency_transform`).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)] // index loops mirror the MATLAB code

use anyhow::{Result, bail};

use super::skewed::{Mvnlogcdf, gaussian_log_mvncdf_mendell_elston, pinv_matlab, sym};
use crate::linalg::{Mat, Vector, mul};

fn normpdf(x: f64, s: f64) -> f64 {
    (-0.5 * (x / s) * (x / s)).exp() / (2.0 * std::f64::consts::PI).sqrt() / s
}

fn rcond(a: &Mat) -> f64 {
    if a.nrows() == 0 {
        return f64::INFINITY;
    }
    let (lu, _, info) = crate::linalg::lu(a);
    if info > 0 {
        return 0.0;
    }
    crate::linalg::rcond_from_lu(&lu, crate::linalg::norm1(a))
}

/// MATLAB `A / B` (`A * inv(B)` through LU), NaN-filled when `B` is singular.
fn mrdivide(a: &Mat, b: &Mat) -> Mat {
    match crate::linalg::solve(&b.transpose(), &a.transpose()) {
        Some(x) => x.transpose(),
        None => Mat::from_element(a.nrows(), b.nrows(), f64::NAN),
    }
}

/// Port of csn_mean.m: the mean of `CSN(mu, Sigma, Gamma, nu, Delta)`. Errors only come from
/// Dynare's `mvncdf` (non positive definite covariance, more than 25 dimensions), as in MATLAB.
pub fn csn_mean(mu: &Vector, sigma: &Mat, gamma: &Mat, nu: &Vector, delta: &Mat, cdf: Mvnlogcdf) -> Result<Vector> {
    let q = delta.nrows();
    if q == 0 {
        return Ok(mu.clone()); // Gamma, nu and Delta all empty: term3 = 1, no correction
    }
    if gamma.nrows() != q || nu.len() != q {
        bail!("Problem with Gamma, nu, Delta being empty / not empty");
    }
    let delta1 = delta + gamma * sigma * gamma.transpose();
    let mut der = Vector::zeros(q);
    for jj in 0..q {
        // reorder: the other dimensions first, jj last
        let ord: Vec<usize> = (0..q).filter(|&i| i != jj).chain(std::iter::once(jj)).collect();
        let d2 = Mat::from_fn(q, q, |i, j| delta1[(ord[i], ord[j])]);
        let nu2: Vec<f64> = ord.iter().map(|&i| -nu[i]).collect();
        let dqq = d2[(q - 1, q - 1)];
        let cond_mean: Vec<f64> = (0..q - 1).map(|i| (d2[(i, q - 1)] * nu2[q - 1]) / dqq).collect();
        let cond_var = Mat::from_fn(q - 1, q - 1, |i, j| d2[(i, j)] - (d2[(i, q - 1)] / dqq) * d2[(q - 1, j)]);
        let term1 = normpdf(nu2[q - 1], dqq.sqrt());
        let term2 = if q == 1 {
            1.0
        } else {
            match cdf {
                Mvnlogcdf::MendellElston => {
                    let nd: Vec<f64> = (0..q - 1).map(|i| 1.0 / cond_var[(i, i)].sqrt()).collect();
                    let ev = Vector::from_fn(q - 1, |i, _| nd[i] * (nu2[i] - cond_mean[i]));
                    let c = sym(&Mat::from_fn(q - 1, q - 1, |i, j| nd[i] * cond_var[(i, j)] * nd[j]));
                    gaussian_log_mvncdf_mendell_elston(&ev, &c).exp()
                }
                Mvnlogcdf::Mvncdf => super::mvncdf::mvncdf(&nu2[..q - 1], &cond_mean, &cond_var)?,
            }
        };
        der[jj] = term1 * term2;
    }
    let delta1 = sym(&delta1);
    let term3 = match cdf {
        Mvnlogcdf::MendellElston => {
            let nd: Vec<f64> = (0..q).map(|i| 1.0 / delta1[(i, i)].sqrt()).collect();
            let ev = Vector::from_fn(q, |i, _| nd[i] * (0.0 - nu[i]));
            let c = sym(&Mat::from_fn(q, q, |i, j| nd[i] * delta1[(i, j)] * nd[j]));
            gaussian_log_mvncdf_mendell_elston(&ev, &c).exp()
        }
        Mvnlogcdf::Mvncdf => super::mvncdf::mvncdf(&vec![0.0; q], nu.as_slice(), &delta1)?,
    };
    Ok(mu + sigma * gamma.transpose() * (der / term3))
}

/// A CSN distribution `(mu, Sigma, Gamma, nu, Delta)`.
#[derive(Clone, Debug)]
pub struct CsnDist {
    pub mu: Vector,
    pub sigma: Mat,
    pub gamma: Mat,
    pub nu: Vector,
    pub delta: Mat,
}

/// Port of csn_statespace_linear_transform.m: the distribution of `G_bar * x` for
/// `x ~ CSN(mu_bar, Sigma_bar, Gamma_bar, nu_bar, Delta_bar)`, with the singular-covariance
/// reduction of csn_singular_to_csn_proper.m (`rcond(Sigma_bar) < 1e-10`) and the
/// rank-deficient (SVD) or full-rank formulas.
pub fn csn_statespace_linear_transform(g_bar: &Mat, x: &CsnDist) -> CsnDist {
    let tol = 1e-8;
    let mu_tr = g_bar * &x.mu;
    let sigma_tr = mul(&mul(g_bar, false, &x.sigma, false), false, g_bar, true);
    let (mut g, mut sigma_bar, mut gamma_bar) = (g_bar.clone(), x.sigma.clone(), x.gamma.clone());
    if rcond(&sigma_bar) < 1e-10 {
        // csn_singular_to_csn_proper (hmdel = 0): [S, Lambda] = schur(Sigma_bar) of the symmetric
        // matrix = its eigen-decomposition; keep the eigenvalues above tol
        let (w, v) = crate::linalg::sym_eig(&sigma_bar);
        let hold: Vec<usize> = (0..w.len()).filter(|&i| w[i] > tol).collect();
        let s_hat = Mat::from_fn(v.nrows(), hold.len(), |i, j| v[(i, hold[j])]);
        sigma_bar = Mat::from_fn(hold.len(), hold.len(), |i, j| if i == j { w[hold[i]] } else { 0.0 });
        gamma_bar = &gamma_bar * &s_hat;
        g = &g * &s_hat;
    }
    let (nrow, ncol) = (g.nrows(), g.ncols());
    let is_singular = if nrow > ncol { rcond(&mul(&g, true, &g, false)) < 1e-8 } else { rcond(&sigma_tr) < 1e-8 };
    let (gamma_tr, nu_tr, delta_tr) = if is_singular {
        // csn_statespace_linear_transform_rank_deficient
        let (u, s, v) = crate::identification::la::svd_econ(&g);
        let hold: Vec<usize> = (0..s.len()).filter(|&i| s[i].abs() > tol).collect();
        let s_mat = Mat::from_fn(u.nrows(), hold.len(), |i, j| u[(i, hold[j])]);
        let t_mat = Mat::from_fn(v.nrows(), hold.len(), |i, j| v[(i, hold[j])]);
        let lam = Mat::from_fn(hold.len(), hold.len(), |i, j| if i == j { s[hold[i]] } else { 0.0 });
        let tmp = mul(&mul(&t_mat, true, &sigma_bar, false), false, &t_mat, false);
        let tmp2 = &gamma_bar * &sigma_bar;
        let tmp3 = mrdivide(&(&tmp2 * &t_mat), &tmp);
        let gamma_tr = mrdivide(&mrdivide(&tmp3, &lam), &mul(&s_mat, true, &s_mat, false)) * s_mat.transpose();
        let delta_tr =
            &x.delta + &tmp2 * gamma_bar.transpose() - &tmp3 * t_mat.transpose() * &sigma_bar * gamma_bar.transpose();
        (gamma_tr, x.nu.clone(), delta_tr)
    } else if nrow >= ncol {
        let gamma_tr = if nrow == ncol {
            mrdivide(&gamma_bar, &g)
        } else {
            mrdivide(&gamma_bar, &mul(&g, true, &g, false)) * g.transpose()
        };
        (gamma_tr, x.nu.clone(), x.delta.clone())
    } else {
        let tmp = mrdivide(&(&sigma_bar * g.transpose()), &sigma_tr);
        let gamma_tr = &gamma_bar * &tmp;
        let gs = &sigma_bar * gamma_bar.transpose();
        let delta_tr = &x.delta + &gamma_bar * &gs - &gamma_bar * &tmp * &g * &gs;
        (gamma_tr, x.nu.clone(), delta_tr)
    };
    CsnDist { mu: mu_tr, sigma: sigma_tr, gamma: gamma_tr, nu: nu_tr, delta: delta_tr }
}

/// `blkdiag(a, b)`.
pub fn blkdiag(a: &Mat, b: &Mat) -> Mat {
    let mut m = Mat::zeros(a.nrows() + b.nrows(), a.ncols() + b.ncols());
    m.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
    m.view_mut((a.nrows(), a.ncols()), (b.nrows(), b.ncols())).copy_from(b);
    m
}

/// `[a; b]`.
pub fn vstack(a: &Mat, b: &Mat) -> Mat {
    let mut m = Mat::zeros(a.nrows() + b.nrows(), a.ncols().max(b.ncols()));
    m.view_mut((0, 0), (a.nrows(), a.ncols())).copy_from(a);
    m.view_mut((a.nrows(), 0), (b.nrows(), b.ncols())).copy_from(b);
    m
}

/// `[a b]` (vectors).
pub fn vcat(a: &Vector, b: &Vector) -> Vector {
    Vector::from_iterator(a.len() + b.len(), a.iter().chain(b.iter()).copied())
}

/// `pinv`, re-exported for the smoother.
pub fn pinv(a: &Mat) -> Mat {
    pinv_matlab(a)
}
