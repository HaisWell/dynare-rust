// SPDX-License-Identifier: GPL-3.0-or-later
// ---- heterogeneity ----
//! Port of the `rouwenhorst` MEX (`mex/sources/heterogeneity/rouwenhorst.f08`,
//! `utils/markov.f08`): `[y, p_vec, P_mat] = rouwenhorst(rho, sigma, N, tol, max_iter)`.

use super::time_iteration::compute_stationary;
use crate::linalg::Mat;
use anyhow::{Result, bail};

/// `N`-point Rouwenhorst discretisation of an AR(1) with persistence `rho` and unconditional
/// standard deviation `sigma`. Returns the grid in levels (normalised so that
/// `sum(p_vec .* y) = 1`), the stationary distribution and the transition matrix.
pub fn rouwenhorst(rho: f64, sigma: f64, n: usize, tol: f64, maxiter: usize) -> Result<(Vec<f64>, Vec<f64>, Mat)> {
    if n < 2 {
        bail!("rouwenhorst: N must be at least 2");
    }
    let p = (1.0 + rho) / 2.0;
    let mut pm = Mat::zeros(n, n);
    pm[(0, 0)] = p;
    pm[(0, 1)] = 1.0 - p;
    pm[(1, 0)] = 1.0 - p;
    pm[(1, 1)] = p;
    for k in 3..=n {
        let old = pm.clone();
        let mut new = Mat::zeros(n, n);
        for i in 0..k - 1 {
            for j in 0..k - 1 {
                let v = old[(i, j)];
                new[(i, j)] += p * v;
                new[(i, j + 1)] += (1.0 - p) * v;
                new[(i + 1, j)] += (1.0 - p) * v;
                new[(i + 1, j + 1)] += p * v;
            }
        }
        for i in 0..k {
            for j in 0..k {
                pm[(i, j)] = new[(i, j)];
            }
        }
        for i in 1..k - 1 {
            for j in 0..n {
                pm[(i, j)] /= 2.0;
            }
        }
    }
    let (pvec, _) = compute_stationary(&pm, tol, maxiter);
    let mut s: Vec<f64> = (0..n).map(|i| -1.0 + i as f64 * (2.0 / (n as f64 - 1.0))).collect();
    let mean: f64 = s.iter().zip(&pvec).map(|(a, b)| a * b).sum();
    let var: f64 = pvec.iter().zip(&s).map(|(p, x)| p * (x - mean).powi(2)).sum();
    let scale = sigma / var.sqrt();
    for v in s.iter_mut() {
        *v *= scale;
    }
    let mut y: Vec<f64> = s.iter().map(|v| v.exp()).collect();
    let norm: f64 = pvec.iter().zip(&y).map(|(a, b)| a * b).sum();
    for v in y.iter_mut() {
        *v /= norm;
    }
    Ok((y, pvec, pm))
}
