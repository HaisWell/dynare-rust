// SPDX-License-Identifier: GPL-3.0-or-later
//! Skewed shocks (`skew` entries of `shocks` blocks, `M_.Skew_e`) outside the skewed Kalman
//! filter, and the other `shocks` declarations `stoch_simul` reads (stream shocks/shock_paths).
//!
//! In Dynare 7.2 `M_.Skew_e` is used by `stoch_simul` only in three places:
//! * `check_model.m`: the skewness coefficients must be within the bound of the skew normal
//!   distribution, `|sqrt(2)(pi-4)/(pi-2)^(3/2)|` ([`check_model`], which also ports the
//!   variance/covariance checks);
//! * the `COSKEWNESS OF EXOGENOUS SHOCKS` table of the model summary ([`skew_table`],
//!   with the `MATRIX OF COVARIANCE OF MEASUREMENT ERRORS` that follows it);
//! * `simult.m` (`stoch_simul(periods > 0)`): with a non-empty `Skew_e` the shocks are drawn
//!   from the closed skew normal `M_.csn = csn_update_specification(Sigma_e, Skew_e)` by
//!   `rand_multivariate_csn.m` ([`rand_multivariate_csn`], [`draw_skewed_shocks`]) instead of
//!   `randn * chol(Sigma_e)`.
//!
//! Theoretical moments, IRFs (including the order 2-3 GIRFs of `irf.m`), decision rules,
//! variance decompositions and identification ignore `Skew_e`; elsewhere Dynare uses it only
//! in estimation (`kalman_algo = 5`).

use anyhow::{Result, bail};

use crate::kalman::skewed::{Csn, sn_skewness_bound};
use crate::linalg::{Mat, Vector, chol_upper};

/// `chol(A, 'lower')`.
fn chol_lower(a: &Mat) -> Option<Mat> {
    chol_upper(a).map(|u| u.transpose())
}
use crate::model::Model;

/// `M_.Skew_e` rows: (i, j, k, value), sorted exogenous indices.
pub type SkewEntries = [(usize, usize, usize, f64)];

/// What `stoch_simul` reads from the `shocks` blocks besides `Sigma_e`: `M_.Skew_e`, `M_.H`
/// and the observed variables (`options_.varobs`, the labels of the `M_.H` table).
#[derive(Clone, Debug, Default)]
pub struct ShocksInfo {
    pub skew_e: Vec<(usize, usize, usize, f64)>,
    pub h: Mat,
    pub varobs: Vec<String>,
}

impl ShocksInfo {
    pub fn from_modfile(mf: &crate::modfile::ModFile) -> ShocksInfo {
        ShocksInfo { skew_e: mf.q3.skew_e.clone(), h: mf.h.clone(), varobs: mf.varobs.clone() }
    }
}

/// `check_consistency_covariances.m`.
fn consistent_covariances(c: &Mat) -> bool {
    let n = c.nrows();
    (0..n).all(|i| (0..i).all(|j| c[(i, j)].abs() <= c[(i, i)].sqrt() * c[(j, j)].sqrt()))
}

/// The checks of `check_model.m` on the shocks (`stoch_simul` calls it first).
pub fn check_model(sigma_e: &Mat, h: &Mat, skew: &SkewEntries) -> Result<()> {
    if (0..sigma_e.nrows()).any(|i| sigma_e[(i, i)] < 0.0) {
        bail!("You have specified negative shock variances. That is not allowed.");
    }
    if !consistent_covariances(sigma_e) {
        bail!(
            "The specified covariances for the structural errors are not consistent with the variances as they imply a correlation larger than +-1"
        );
    }
    if (0..h.nrows()).any(|i| h[(i, i)] < 0.0) {
        bail!("You have specified negative measurement error variances. That is not allowed.");
    }
    if h.iter().any(|&v| v != 0.0) && !consistent_covariances(h) {
        bail!(
            "The specified covariances for the measurement errors are not consistent with the variances as they imply a correlation larger than +-1"
        );
    }
    let bound = sn_skewness_bound();
    if skew.iter().any(|e| e.3.abs() > bound) {
        bail!("Skewness parameters are larger than theoretical bound of skew normal distribution of ±{bound:.6}.");
    }
    Ok(())
}

/// The `COSKEWNESS OF EXOGENOUS SHOCKS` table of stoch_simul.m's model summary (non-empty
/// `Skew_e`), printed by `analysis::model_summary_full` between the shock covariance and the
/// measurement-error covariance.
/// The model-summary tables after the shock covariance: [`skew_table`] then the
/// measurement-error covariance ([`crate::analysis::me_table`]).
pub fn summary_tables(model: &Model, skew: &SkewEntries, h: &Mat, varobs: &[String]) -> String {
    skew_table(model, skew) + &crate::analysis::me_table(h, varobs)
}

pub fn skew_table(model: &Model, skew: &SkewEntries) -> String {
    use crate::analysis::table::dyntable;
    let maxlen = |v: &[String]| v.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    let mut s = String::new();
    if !skew.is_empty() {
        let n = &model.exo_names;
        let labels: Vec<String> = skew
            .iter()
            .map(|&(i, j, k, _)| {
                if i == j && j == k { format!("Skew({})", n[i]) } else { format!("Skew({}, {}, {})", n[i], n[j], n[k]) }
            })
            .collect();
        let values = Mat::from_iterator(skew.len(), 1, skew.iter().map(|e| e.3));
        s += &dyntable(
            "COSKEWNESS OF EXOGENOUS SHOCKS",
            &["Entry".to_string(), "Value".to_string()],
            &labels,
            &values,
            maxlen(&labels) + 2,
            10,
            4,
        );
    }
    s
}

/// `csn_update_specification(Sigma_e, Skew_e)` (independent skew normal shocks only).
pub fn csn_specification(sigma_e: &Mat, skew: &SkewEntries) -> Result<Csn> {
    if skew.iter().any(|&(i, j, k, _)| i != j || j != k) {
        bail!(
            "csn_update_specification: Skewness is currently only supported for independent skew normally distributed shocks, i.e. there cannot be co-skewness between shocks."
        );
    }
    let mut v = vec![0.0; sigma_e.nrows()];
    for &(i, _, _, x) in skew {
        v[i] = x;
    }
    Csn::from_skew(sigma_e, &v)
}

/// Port of `rand_multivariate_csn.m`: `n` draws (columns) of CSN(mu, Sigma, Gamma, nu,
/// Delta), the standard normals taken from `randn` in MATLAB's order (the rejection draws
/// `randn(q, 1)` one draw at a time, then `randn(length(mu), n)` column by column).
pub fn rand_multivariate_csn(
    n: usize,
    mu: &Vector,
    sigma: &Mat,
    gamma: &Mat,
    nu: &Vector,
    delta: &Mat,
    randn: &mut dyn FnMut() -> f64,
) -> Result<Mat> {
    let q = nu.len();
    let p = mu.len();
    let sigma2 = delta + gamma * sigma * gamma.transpose();
    let Some(l) = chol_lower(&sigma2) else {
        bail!("rand_multivariate_csn: Delta + Gamma*Sigma*Gamma' is not positive definite")
    };
    let mut v = Mat::zeros(q, n);
    let mut z = Vector::zeros(q);
    for i in 0..n {
        loop {
            for r in 0..q {
                z[r] = randn();
            }
            let draws = -nu + &l * &z;
            if draws.iter().all(|&d| d >= 0.0) {
                v.set_column(i, &draws);
                break;
            }
        }
    }
    // tmp = Sigma * Gamma' / Sigma2
    let Some(s2inv) = sigma2.clone().try_inverse() else { bail!("rand_multivariate_csn: singular Sigma2") };
    let tmp = sigma * gamma.transpose() * &s2inv;
    let mut vn = v;
    for mut c in vn.column_iter_mut() {
        c += nu;
    }
    let mut exp = &tmp * vn;
    for mut c in exp.column_iter_mut() {
        c += mu;
    }
    let var = sigma - &tmp * gamma * sigma;
    let var = (&var + var.transpose()) * 0.5;
    let Some(l2) = chol_lower(&var) else {
        bail!("rand_multivariate_csn: the conditional variance is not positive definite")
    };
    let mut e = Mat::zeros(p, n);
    for j in 0..n {
        for r in 0..p {
            e[(r, j)] = randn();
        }
    }
    Ok(exp + l2 * e)
}

/// `simult.m`'s skewed shocks: `exo_simul(:, i_exo_var) = rand_multivariate_csn(periods,
/// csn(i_exo_var))'` over the shocks with non-zero variance (periods x n_exo).
pub fn draw_skewed_shocks(
    sigma_e: &Mat,
    skew: &SkewEntries,
    periods: usize,
    randn: &mut dyn FnMut() -> f64,
) -> Result<Mat> {
    let csn = csn_specification(sigma_e, skew)?;
    let nexo = sigma_e.nrows();
    let iv: Vec<usize> = (0..nexo).filter(|&i| sigma_e[(i, i)] != 0.0).collect();
    let sub = |m: &Mat| Mat::from_fn(iv.len(), iv.len(), |a, b| m[(iv[a], iv[b])]);
    let subv = |v: &Vector| Vector::from_fn(iv.len(), |a, _| v[iv[a]]);
    let x = rand_multivariate_csn(
        periods,
        &subv(&csn.mu),
        &sub(&csn.sigma),
        &sub(&csn.gamma),
        &subv(&csn.nu),
        &sub(&csn.delta),
        randn,
    )?;
    let mut out = Mat::zeros(periods, nexo);
    for (a, &i) in iv.iter().enumerate() {
        for t in 0..periods {
            out[(t, i)] = x[(a, t)];
        }
    }
    Ok(out)
}
