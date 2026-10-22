// SPDX-License-Identifier: GPL-3.0-or-later
//! Variance decompositions at order 1: ports of `moments/compute_variance_decomposition.m`
//! (unconditional, stationary variables only), `moments/conditional_variance_decomposition.m`
//! and `get_lower_cholesky_covariance.m`.

use anyhow::{Result, bail};

use super::moments::{LyapunovSchur, MomentOptions};
use crate::linalg::{Mat, chol_upper, mul};
use crate::solver::DecisionRule;

/// Port of `get_lower_cholesky_covariance.m`: lower Cholesky factor of the non-zero-variance
/// block of `sigma`, with `add_tiny` on the diagonal if that block is not positive definite.
pub fn lower_cholesky(sigma: &Mat, add_tiny: f64) -> Mat {
    let n = sigma.nrows();
    let nz: Vec<usize> = (0..n).filter(|&i| sigma[(i, i)].sqrt() != 0.0).collect();
    let sub = Mat::from_fn(nz.len(), nz.len(), |i, j| sigma[(nz[i], nz[j])]);
    let u = chol_upper(&sub).unwrap_or_else(|| {
        println!(
            "Non-positive definite covariance matrix encountered. Using add_tiny_number_to_cholesky one the diagonal."
        );
        let s2 = &sub + Mat::identity(nz.len(), nz.len()) * add_tiny;
        chol_upper(&s2).unwrap_or_else(|| Mat::zeros(nz.len(), nz.len()))
    });
    let mut out = Mat::zeros(n, n);
    for (a, &i) in nz.iter().enumerate() {
        for (b, &j) in nz.iter().enumerate() {
            out[(i, j)] = u[(b, a)];
        }
    }
    out
}

/// Port of `compute_variance_decomposition.m`: nvar x exo_nbr shares of the stationary
/// variables (NaN rows for the others). `lyap` is the Schur form of the state transition
/// (lyapunov_symm methods 1/2 reuse it), `aa` = ghx(stationary rows), `srows` their DR rows.
#[allow(clippy::too_many_arguments)]
pub fn compute_variance_decomposition(
    lyap: &LyapunovSchur,
    aa: &Mat,
    ghu: &Mat,
    ghu_states: &Mat,
    srows: &[usize],
    stationary: &[usize],
    nvar: usize,
    sigma_e: &Mat,
    var_stationary: &[f64],
    o: &MomentOptions,
) -> Mat {
    let nexo = sigma_e.nrows();
    if nexo == 1 {
        return Mat::from_element(nvar, 1, 1.0);
    }
    let mut vd = Mat::from_element(nvar, nexo, f64::NAN);
    let cs = lower_cholesky(sigma_e, o.add_tiny_number_to_cholesky);
    let b1 = mul(ghu_states, false, &cs, false);
    let b2 = mul(&ghu.select_rows(srows), false, &cs, false);
    let ns = stationary.len();
    let mut sum = vec![0.0; ns];
    let mut parts = Mat::zeros(ns, nexo);
    for i in 0..nexo {
        let c = b1.column(i).into_owned();
        let bb = &c * c.transpose();
        let p = lyap.solve(&bb, o.lyapunov_complex_threshold);
        let ap = if aa.ncols() > 0 { mul(aa, false, &p, false) } else { Mat::zeros(ns, 0) };
        for r in 0..ns {
            let mut x = 0.0;
            for k in 0..aa.ncols() {
                x += ap[(r, k)] * aa[(r, k)];
            }
            x += b2[(r, i)] * b2[(r, i)];
            parts[(r, i)] = x;
            sum[r] += x;
        }
    }
    let rel = (0..ns).map(|r| ((sum[r] - var_stationary[r]) / var_stationary[r]).abs()).fold(0.0, f64::max);
    let abs = (0..ns).map(|r| (sum[r] - var_stationary[r]).abs()).fold(0.0, f64::max);
    if rel > 1e-4 && abs > 1e-7 {
        println!("Warning: Aggregate variance and sum of variances by shocks differ by more than 0.01 %");
    }
    for r in 0..ns {
        for i in 0..nexo {
            vd[(stationary[r], i)] = parts[(r, i)] / sum[r];
        }
    }
    vd
}

/// Port of `conditional_variance_decomposition.m`: for each horizon in `steps`, an
/// nvar x exo_nbr matrix of shares of the h-step forecast-error variance (full state space,
/// so nonstationary variables are included). Shares are fractions (Dynare multiplies by 100
/// when printing).
///
/// Dynare iterates V = T V T' + B_i B_i' on the full endo_nbr x endo_nbr state space; only
/// diag(V) is used, so this accumulates (T^s B_i)^2 instead (same numbers up to rounding,
/// O(n^2) per step instead of O(n^3)).
pub fn conditional_variance_decomposition(
    dr: &DecisionRule,
    sigma_e: &Mat,
    steps: &[usize],
    ivar: &[usize],
) -> Result<Vec<Mat>> {
    let cv = conditional_variances(dr, sigma_e, steps, ivar)?;
    let nexo = sigma_e.nrows();
    Ok(cv
        .into_iter()
        .map(|c| {
            Mat::from_fn(c.nrows(), nexo, |k, i| {
                let s: f64 = (0..nexo).map(|j| c[(k, j)]).sum();
                c[(k, i)] / s
            })
        })
        .collect())
}

/// `ConditionalVariance` of conditional_variance_decomposition.m: per step, nvar x exo_nbr
/// h-step forecast-error variances by shock (before normalisation).
pub fn conditional_variances(dr: &DecisionRule, sigma_e: &Mat, steps: &[usize], ivar: &[usize]) -> Result<Vec<Mat>> {
    if steps.contains(&0) {
        bail!("Conditional variance decomposition: All periods must be strictly positive");
    }
    let n = dr.ghx.nrows();
    let nexo = sigma_e.nrows();
    let (nstatic, nspred) = (dr.nstatic, dr.nspred);
    let diagonal = (0..nexo).all(|i| (0..nexo).all(|j| i == j || sigma_e[(i, j)] == 0.0));
    // B = impulse_matrix .* sqrt(diag(Sigma_e))'  or  impulse_matrix * chol(Sigma_e)'
    let b = if diagonal {
        Mat::from_fn(n, nexo, |r, j| dr.ghu[(r, j)] * sigma_e[(j, j)].sqrt())
    } else {
        let u = chol_upper(sigma_e).unwrap_or_else(|| lower_cholesky(sigma_e, 1e-14).transpose());
        mul(&dr.ghu, false, &u, true)
    };
    let hmax = *steps.iter().max().unwrap_or(&0);
    let rows: Vec<usize> = ivar.iter().map(|&v| dr.inv_order_var[v]).collect();
    // cv[m][(k, i)]: conditional variance of ivar[k] at steps[m] due to shock i
    let mut cv = vec![Mat::zeros(ivar.len(), nexo); steps.len()];
    for i in 0..nexo {
        let mut x: Vec<f64> = b.column(i).iter().copied().collect();
        let mut v = vec![0.0; n];
        let mut m = 0;
        for h in 1..=hmax {
            if h > 1 {
                let st: Vec<f64> = x[nstatic..nstatic + nspred].to_vec();
                for (r, xr) in x.iter_mut().enumerate() {
                    *xr = (0..nspred).map(|k| dr.ghx[(r, k)] * st[k]).sum();
                }
            }
            for r in 0..n {
                v[r] += x[r] * x[r];
            }
            while m < steps.len() && steps[m] == h {
                for (k, &r) in rows.iter().enumerate() {
                    cv[m][(k, i)] = v[r];
                }
                m += 1;
            }
        }
    }
    // Steps are visited in increasing order above, as Dynare's loop requires.
    Ok(cv)
}
