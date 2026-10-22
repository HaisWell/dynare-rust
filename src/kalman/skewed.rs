// SPDX-License-Identifier: GPL-3.0-or-later
//! Skewed shocks: the pruned skewed Kalman filter (`kalman_algo = 5`) of
//! Guljanov, Mutschler and Trede (2024), as Dynare 7.2 implements it:
//!
//! * [`Csn::from_skew`], `csn_update_specification.m` (independent skew normal shocks with
//!   variance `Sigma_e` and skewness coefficients `Skew_e(k,k,k)`);
//! * [`gaussian_log_mvncdf_mendell_elston`], `distributions/gaussian/gaussian_log_mvncdf_mendell_elston.m`;
//! * [`csn_prune_distribution`], `distributions/closed_skew_normal/csn_prune_distribution.m`;
//! * [`kalman_filter_pruned_skewed`], `kalman/likelihood/kalman_filter_pruned_skewed.m`
//!   ([estimation completion] with `rank_deficiency_transform` and both `mvnlogcdf` routines;
//!   the smoother is [`super::skewed_smoother`]).

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use anyhow::{Result, bail};

use crate::linalg::{Mat, Vector, mul};

/// `M_.csn`: closed skew normal parameters of the shocks.
#[derive(Clone, Debug)]
pub struct Csn {
    pub mu: Vector,
    pub sigma: Mat,
    pub gamma: Mat,
    pub nu: Vector,
    pub delta: Mat,
}

/// `sn_var_skew_to_scale_shape_univariate` of csn_update_specification.m: (omega, alpha).
fn sn_var_skew_to_scale_shape(var: f64, skew: f64) -> (f64, f64) {
    let pi = std::f64::consts::PI;
    let a23 = skew.abs().powf(2.0 / 3.0);
    let delta = (pi / 2.0 * a23 / (a23 + ((4.0 - pi) / 2.0).powf(2.0 / 3.0))).sqrt();
    let alpha = skew.signum() * delta / (1.0 - delta * delta).sqrt();
    let omega = (var / (1.0 - 2.0 / pi * delta * delta)).sqrt();
    (omega, alpha)
}

/// Theoretical bound of the skewness coefficient of a skew normal distribution.
pub fn sn_skewness_bound() -> f64 {
    let pi = std::f64::consts::PI;
    ((2f64.sqrt() * (pi - 4.0)) / (pi - 2.0).powf(1.5)).abs()
}

impl Csn {
    /// Port of csn_update_specification.m; `skew[k]` = `Skew_e(k,k,k)` (0 = Gaussian).
    pub fn from_skew(var: &Mat, skew: &[f64]) -> Result<Csn> {
        let n = var.nrows();
        let mut c = Csn {
            mu: Vector::zeros(n),
            sigma: var.clone(),
            gamma: Mat::zeros(n, n),
            nu: Vector::zeros(n),
            delta: Mat::identity(n, n),
        };
        if skew.iter().any(|&s| s != 0.0) {
            let isdiag = (0..n).all(|i| (0..n).all(|j| i == j || var[(i, j)] == 0.0));
            if !isdiag {
                bail!(
                    "csn_update_specification: Skewness is currently only supported for independent skew normally distributed shocks, i.e. they cannot be correlated. Remove the corr parameters."
                );
            }
            let s2pi = (2.0 / std::f64::consts::PI).sqrt();
            for (j, &s) in skew.iter().enumerate() {
                if s.abs() > f64::EPSILON {
                    let (omega, alpha) = sn_var_skew_to_scale_shape(var[(j, j)], s);
                    c.sigma[(j, j)] = omega * omega;
                    c.gamma[(j, j)] = alpha / omega;
                    c.mu[j] = -omega * alpha / (1.0 + alpha * alpha).sqrt() * s2pi;
                }
            }
        }
        Ok(c)
    }
}

fn phid(z: f64) -> f64 {
    super::mvncdf::erfc(-z / std::f64::consts::SQRT_2) / 2.0 // [estimation completion] libm erfc, as MATLAB
}

fn phip(z: f64) -> f64 {
    (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Port of gaussian_log_mvncdf_mendell_elston.m (cutoffs 6 and 38): log P(X <= z) for
/// X ~ N(0, R), R a correlation matrix.
pub fn gaussian_log_mvncdf_mendell_elston(z: &Vector, r: &Mat) -> f64 {
    let (c1, c2) = (6.0, 38.0);
    let mut zj: Vector = z.map(|v| v.clamp(-c1, c1));
    let mut r = r.clone();
    let n = zj.len();
    if n == 0 {
        return 0.0;
    }
    let mut cdf = phid(zj[0]);
    let mut pdf = phip(zj[0]);
    let mut log_cdf = cdf.ln();
    for _ in 1..n {
        let a = pdf / cdf;
        let r0 = r.column(0).into_owned();
        let tz = &zj + &r0 * a;
        let rjj = &r0 * r.row(0);
        let tr = &r - rjj * ((a + zj[0]) * a);
        let c: Vector = tr.diagonal().map(f64::sqrt);
        let m = zj.len();
        let zn = Vector::from_fn(m, |i, _| (tz[i] / c[i]).clamp(-c2, c2));
        let rn = Mat::from_fn(m, m, |i, j| tr[(i, j)] / (c[i] * c[j]));
        cdf = phid(zn[1]);
        pdf = phip(zn[1]);
        zj = zn.rows(1, m - 1).into_owned();
        r = rn.view((1, 1), (m - 1, m - 1)).into_owned();
        log_cdf += cdf.ln();
    }
    log_cdf
}

/// Port of csn_prune_distribution.m: drops the skewness dimensions whose largest absolute
/// correlation with the normal part is below `tol`.
pub fn csn_prune_distribution(sigma: &Mat, gamma: &Mat, nu: &Vector, delta: &Mat, tol: f64) -> (Mat, Vector, Mat) {
    let (p, q) = (sigma.nrows(), gamma.nrows());
    let gs = gamma * sigma;
    let mut pm = Mat::zeros(p + q, p + q);
    pm.view_mut((0, 0), (p, p)).copy_from(sigma);
    pm.view_mut((0, p), (p, q)).copy_from(&gs.transpose());
    pm.view_mut((p, 0), (q, p)).copy_from(&gs);
    pm.view_mut((p, p), (q, q)).copy_from(&(delta + &gs * gamma.transpose()));
    let pm = (&pm + pm.transpose()) * 0.5;
    // [estimation completion] MATLAB: abs(normalization*P*normalization) with
    // normalization = diag(1./sqrt(diag(P))); a (rounding-noise) negative variance makes it
    // complex and the absolute value is |P_ij| / sqrt(|P_ii| |P_jj|), not NaN.
    let d: Vec<f64> = (0..p + q).map(|i| 1.0 / pm[(i, i)].abs().sqrt()).collect();
    let keep: Vec<usize> = (0..q)
        .filter(|&i| {
            let mx = (0..p).map(|j| (d[p + i] * pm[(p + i, j)] * d[j]).abs()).fold(f64::NEG_INFINITY, f64::max);
            !(mx < tol)
        })
        .collect();
    let g = Mat::from_fn(keep.len(), p, |i, j| gamma[(keep[i], j)]);
    let n = Vector::from_fn(keep.len(), |i, _| nu[keep[i]]);
    let dl = Mat::from_fn(keep.len(), keep.len(), |i, j| delta[(keep[i], keep[j])]);
    (g, n, dl)
}

/// MATLAB `pinv` (SVD, tolerance `max(size)*norm*eps`).
pub(crate) fn pinv_matlab(a: &Mat) -> Mat {
    let svd = a.clone().svd(true, true);
    let (u, vt, s) = (svd.u.unwrap(), svd.v_t.unwrap(), svd.singular_values);
    let smax = s.iter().copied().fold(0.0, f64::max);
    let tol = a.nrows().max(a.ncols()) as f64 * smax * f64::EPSILON;
    let mut out = Mat::zeros(a.ncols(), a.nrows());
    for k in 0..s.len() {
        if s[k] > tol {
            out += vt.row(k).transpose() * u.column(k).transpose() / s[k];
        }
    }
    out
}

pub(crate) fn sym(m: &Mat) -> Mat {
    (m + m.transpose()) * 0.5
}

fn inv(a: &Mat) -> Option<Mat> {
    crate::linalg::solve(a, &Mat::identity(a.nrows(), a.nrows()))
}

fn rcond(a: &Mat) -> f64 {
    let (lu, _, info) = crate::linalg::lu(a);
    if info > 0 {
        return 0.0;
    }
    crate::linalg::rcond_from_lu(&lu, crate::linalg::norm1(a))
}

fn det(a: &Mat) -> f64 {
    let (lu, ipiv, _) = crate::linalg::lu(a);
    let mut d = 1.0;
    for i in 0..a.nrows() {
        d *= lu[(i, i)];
        if ipiv[i] as usize != i + 1 {
            d = -d;
        }
    }
    d
}

/// `options_.skewed_kalman.mvnlogcdf`: how the filter and `csn_mean` evaluate Gaussian CDFs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mvnlogcdf {
    /// `'gaussian_log_mvncdf_mendell_elston'` (default).
    #[default]
    MendellElston,
    /// `'mvncdf'`: Dynare's `mvncdf` replacement ([`super::mvncdf`]).
    Mvncdf,
}

impl Mvnlogcdf {
    /// The option value (quotes stripped); Dynare compares with `strcmp`, an unknown name
    /// leaves the CDFs undefined (MATLAB error), reported here.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().trim_matches(|c| c == '\'' || c == '"') {
            "gaussian_log_mvncdf_mendell_elston" => Ok(Mvnlogcdf::MendellElston),
            "mvncdf" => Ok(Mvnlogcdf::Mvncdf),
            other => bail!(
                "skewed_kalman_mvnlogcdf = '{other}': possible values are 'gaussian_log_mvncdf_mendell_elston' and 'mvncdf' \
                 (Dynare fails with an undefined log CDF)"
            ),
        }
    }
}

/// Options of the pruned skewed filter (`options_.skewed_kalman`).
#[derive(Clone, Debug)]
pub struct SkewedOptions {
    pub kalman_tol: f64,
    pub rescale_prediction_error_covariance: bool,
    pub prune_tol: f64,
    pub presample: usize,
    /// [estimation completion] `skewed_kalman_mvnlogcdf`.
    pub mvnlogcdf: Mvnlogcdf,
    /// [estimation completion] `skewed_kalman_rank_deficiency_transform`.
    pub rank_deficiency_transform: bool,
}

/// Port of kalman_filter_pruned_skewed.m from the Gaussian initial state `N(mu0, sigma0)`
/// (Gamma_0 = 0, nu_0 = 0, Delta_0 = I). `y`: demeaned data (pp x nobs, no missing values),
/// `z`: observed rows of the state. Returns (LIK, per-period minus log densities); LIK is
/// +Inf when Omega is badly conditioned or a CDF fails (the `try/catch` of the MATLAB code).
/// With `rank_deficiency_transform` the prediction step transforms the joint distribution of
/// `[x_{t-1}; eta_t]` by `[T R]` (csn_statespace_linear_transform.m).
pub fn kalman_filter_pruned_skewed(
    y: &Mat,
    mu0: &Vector,
    sigma0: &Mat,
    t: &Mat,
    r: &Mat,
    z: &[usize],
    csn: &Csn,
    h: &Mat,
    o: &SkewedOptions,
) -> (f64, Vec<f64>) {
    let (m, pp, n) = (t.nrows(), y.nrows(), y.ncols());
    let rr = r.ncols();
    let fail = (f64::INFINITY, Vec::new());
    let zsel = |a: &Mat| Mat::from_fn(z.len(), a.ncols(), |i, j| a[(z[i], j)]); // Z*A
    let const2pi = -0.5 * pp as f64 * (2.0 * std::f64::consts::PI).ln();
    // rank_deficiency_transform = false: the shocks' CSN parameters mapped by R
    let (mu_eta, sigma_eta, gamma_eta, ge_se, delta22_common, t_bar) = if o.rank_deficiency_transform {
        let mut tb = Mat::zeros(m, m + rr);
        tb.view_mut((0, 0), (m, m)).copy_from(t);
        tb.view_mut((0, m), (m, rr)).copy_from(r);
        (Vector::zeros(0), Mat::zeros(0, 0), Mat::zeros(0, 0), Mat::zeros(0, 0), Mat::zeros(0, 0), Some(tb))
    } else {
        let mu_eta = r * &csn.mu;
        let sigma_eta = mul(&mul(r, false, &csn.sigma, false), false, r, true);
        let rtr = mul(r, true, r, false);
        let Some(irtr) = inv(&rtr) else { return fail };
        let gamma_eta = &csn.gamma * irtr * r.transpose();
        let ge_se = &gamma_eta * &sigma_eta;
        let delta22_common = &csn.delta + &ge_se * gamma_eta.transpose();
        (mu_eta, sigma_eta, gamma_eta, ge_se, delta22_common, None)
    };
    let nu_eta = csn.nu.clone();
    let (mut mu, mut sigma) = (mu0.clone(), sigma0.clone());
    let (mut gamma, mut nu, mut delta) = (Mat::zeros(m, m), Vector::zeros(m), Mat::identity(m, m));
    let mut likk = vec![0.0; n];
    let mut rescale = o.rescale_prediction_error_covariance;
    let mut omega_singular = true;
    for tt in 0..n {
        let (mu_t, sigma_t, mut gamma_t, mut nu_t, mut delta_t) = if let Some(tb) = &t_bar {
            let joint = super::csn::CsnDist {
                mu: super::csn::vcat(&mu, &csn.mu),
                sigma: super::csn::blkdiag(&sigma, &csn.sigma),
                gamma: super::csn::blkdiag(&gamma, &csn.gamma),
                nu: super::csn::vcat(&nu, &csn.nu),
                delta: super::csn::blkdiag(&delta, &csn.delta),
            };
            let d = super::csn::csn_statespace_linear_transform(tb, &joint);
            (d.mu, d.sigma, d.gamma, d.nu, d.delta)
        } else {
            let gs = &gamma * &sigma;
            let gst = &gs * t.transpose();
            let mu_t = t * &mu + &mu_eta;
            let sigma_t = sym(&(mul(&mul(t, false, &sigma, false), false, t, true) + &sigma_eta));
            let isig = pinv_matlab(&sigma_t);
            let q = gamma.nrows();
            let mut top = Mat::zeros(q + rr, m);
            top.view_mut((0, 0), (q, m)).copy_from(&gst);
            top.view_mut((q, 0), (rr, m)).copy_from(&ge_se);
            let gamma_t = top * &isig;
            let mut nu_t = Vector::zeros(q + rr);
            nu_t.rows_mut(0, q).copy_from(&nu);
            nu_t.rows_mut(q, rr).copy_from(&nu_eta);
            let d11 = &delta + &gs * gamma.transpose() - &gst * &isig * gst.transpose();
            let d22 = &delta22_common - &ge_se * &isig * ge_se.transpose();
            let d12 = -(&gst * &isig * ge_se.transpose());
            let mut delta_t = Mat::zeros(q + rr, q + rr);
            delta_t.view_mut((0, 0), (q, q)).copy_from(&d11);
            delta_t.view_mut((0, q), (q, rr)).copy_from(&d12);
            delta_t.view_mut((q, 0), (rr, q)).copy_from(&d12.transpose());
            delta_t.view_mut((q, q), (rr, rr)).copy_from(&d22);
            let _ = &gamma_eta;
            (mu_t, sigma_t, gamma_t, nu_t, sym(&delta_t))
        };
        if o.prune_tol > 0.0 {
            let (g, nn, d) = csn_prune_distribution(&sigma_t, &gamma_t, &nu_t, &delta_t, o.prune_tol);
            gamma_t = g;
            nu_t = nn;
            delta_t = d;
        }
        let pe = Vector::from_fn(pp, |i, _| y[(i, tt)] - mu_t[z[i]]);
        let zs = zsel(&sigma_t);
        let omega = Mat::from_fn(pp, pp, |i, j| zs[(i, z[j])] + h[(i, j)]);
        let sig: Vec<f64> = (0..pp).map(|i| omega[(i, i)].sqrt()).collect();
        let scaled = Mat::from_fn(pp, pp, |i, j| omega[(i, j)] / (sig[i] * sig[j]));
        let mut bad = false;
        if rescale {
            if (0..pp).any(|i| omega[(i, i)] < o.kalman_tol) || rcond(&scaled) < o.kalman_tol {
                bad = true;
            }
        } else if rcond(&omega) < o.kalman_tol {
            if (0..pp).any(|i| omega[(i, i)] < o.kalman_tol) || rcond(&scaled) < o.kalman_tol {
                bad = true;
            } else {
                rescale = true;
            }
        }
        if bad {
            return fail;
        }
        omega_singular = false;
        let (log_det, iomega) = if rescale {
            let Some(is) = inv(&scaled) else { return fail };
            let ld = det(&scaled).ln() + 2.0 * sig.iter().map(|s| s.ln()).sum::<f64>();
            rescale = o.rescale_prediction_error_covariance;
            (ld, Mat::from_fn(pp, pp, |i, j| is[(i, j)] / (sig[i] * sig[j])))
        } else {
            let Some(io) = inv(&omega) else { return fail };
            (det(&omega).ln(), io)
        };
        // K_Gauss = Sigma*Z'*invOmega
        let sz = Mat::from_fn(m, pp, |i, j| sigma_t[(i, z[j])]);
        let k_gauss = &sz * &iomega;
        let k_skewed = &gamma_t * &k_gauss;
        let tmp = &gamma_t * &sigma_t;
        let ztmp_t = Mat::from_fn(pp, tmp.nrows(), |i, j| tmp[(j, z[i])]); // Z*tmp'
        let delta_y = sym(&(&delta_t + &tmp * gamma_t.transpose() - &k_skewed * ztmp_t));
        let cbc = sym(&(&delta_y + &k_skewed * &omega * k_skewed.transpose()));
        let qn = cbc.nrows();
        let (log_bottom, log_top) = match o.mvnlogcdf {
            Mvnlogcdf::MendellElston => {
                let log_bottom = if qn > 0 {
                    let nd: Vec<f64> = (0..qn).map(|i| 1.0 / cbc[(i, i)].sqrt()).collect();
                    let c = sym(&Mat::from_fn(qn, qn, |i, j| nd[i] * cbc[(i, j)] * nd[j]));
                    gaussian_log_mvncdf_mendell_elston(&Vector::from_fn(qn, |i, _| -nd[i] * nu_t[i]), &c)
                } else {
                    0.0
                };
                let log_top = if qn > 0 {
                    let nd: Vec<f64> = (0..qn).map(|i| 1.0 / delta_y[(i, i)].sqrt()).collect();
                    let c = sym(&Mat::from_fn(qn, qn, |i, j| nd[i] * delta_y[(i, j)] * nd[j]));
                    let kpe = &k_skewed * &pe - &nu_t;
                    gaussian_log_mvncdf_mendell_elston(&Vector::from_fn(qn, |i, _| nd[i] * kpe[i]), &c)
                } else {
                    0.0
                };
                (log_bottom, log_top)
            }
            Mvnlogcdf::Mvncdf => {
                // log(mvncdf(zeros(q,1), nu, cdf_bottom_cov)), log(mvncdf(K_Skewed*pe, nu, Delta_y));
                // an mvncdf error discards the draw (try/catch, LIK = Inf)
                let Ok(b) = super::mvncdf::mvncdf(&vec![0.0; qn], nu_t.as_slice(), &cbc) else { return fail };
                let kpe = &k_skewed * &pe;
                let Ok(tp) = super::mvncdf::mvncdf(kpe.as_slice(), nu_t.as_slice(), &delta_y) else { return fail };
                (b.ln(), tp.ln())
            }
        };
        let log_pdf = const2pi - 0.5 * log_det - 0.5 * pe.dot(&(&iomega * &pe));
        likk[tt] = log_top - log_bottom + log_pdf;
        // update
        mu = &mu_t + &k_gauss * &pe;
        let zsig = zsel(&sigma_t);
        sigma = sym(&(&sigma_t - &k_gauss * zsig));
        gamma = gamma_t;
        nu = &nu_t - &k_skewed * &pe;
        delta = sym(&delta_t);
    }
    if omega_singular {
        return fail;
    }
    let lik: f64 = -likk[o.presample.min(n)..].iter().sum::<f64>();
    (lik, likk.iter().map(|v| -v).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mendell_elston_univariate_and_independent() {
        let z = Vector::from_vec(vec![0.5]);
        let v = gaussian_log_mvncdf_mendell_elston(&z, &Mat::identity(1, 1));
        assert!((v - phid(0.5).ln()).abs() < 1e-14);
        let z = Vector::from_vec(vec![0.5, -0.3]);
        let v = gaussian_log_mvncdf_mendell_elston(&z, &Mat::identity(2, 2));
        assert!((v - (phid(0.5).ln() + phid(-0.3).ln())).abs() < 1e-10);
    }

    #[test]
    fn csn_specification_has_zero_mean_and_given_variance() {
        let var = Mat::from_diagonal(&Vector::from_vec(vec![0.25, 1.0]));
        let c = Csn::from_skew(&var, &[0.5, 0.0]).unwrap();
        // skew normal: mean mu + omega*delta*sqrt(2/pi) = 0, variance omega^2 (1 - 2 delta^2/pi) = var
        let (omega, alpha) = (c.sigma[(0, 0)].sqrt(), c.gamma[(0, 0)] * c.sigma[(0, 0)].sqrt());
        let delta = alpha / (1.0 + alpha * alpha).sqrt();
        assert!((c.mu[0] + omega * delta * (2.0 / std::f64::consts::PI).sqrt()).abs() < 1e-15);
        assert!((omega * omega * (1.0 - 2.0 * delta * delta / std::f64::consts::PI) - 0.25).abs() < 1e-14);
        assert_eq!(c.sigma[(1, 1)], 1.0);
    }
}
