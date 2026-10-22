// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration nodes of the stochastic extended path: port of `ep/setup_integration_nodes.m`,
//! `gauss_hermite_weights_and_nodes.m`, `cartesian_product_of_sets.m` and
//! `cubature_with_gaussian_weight.m`.

use anyhow::{Result, bail};

use crate::linalg::{Mat, sym_eig};

/// `options_.ep.stochastic.IntegrationAlgorithm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integration {
    /// `Tensor-Gaussian-Quadrature` (default).
    Tensor,
    /// `Stroud-Cubature-3`.
    Stroud3,
    /// `Stroud-Cubature-5` (the `integration=stroud` option).
    Stroud5,
    /// `Unscented`.
    Unscented,
}

impl Integration {
    pub fn dynare_name(self) -> &'static str {
        match self {
            Integration::Tensor => "Tensor-Gaussian-Quadrature",
            Integration::Stroud3 => "Stroud-Cubature-3",
            Integration::Stroud5 => "Stroud-Cubature-5",
            Integration::Unscented => "Unscented",
        }
    }
}

/// `gauss_hermite_weights_and_nodes(n)`: nodes and weights of the Gauss-Hermite rule for a
/// standard normal (weights sum to one), nodes ascending.
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut j = Mat::zeros(n, n);
    for i in 0..n.saturating_sub(1) {
        let b = (((i + 1) as f64) / 2.0).sqrt();
        j[(i, i + 1)] = b;
        j[(i + 1, i)] = b;
    }
    let (vals, vecs) = sym_eig(&j);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    let nodes = idx.iter().map(|&k| 2f64.sqrt() * vals[k]).collect();
    let weights = idx.iter().map(|&k| vecs[(0, k)] * vecs[(0, k)]).collect();
    (nodes, weights)
}

/// `cartesian_product_of_sets` (`ndgrid` order: the first set varies fastest).
pub fn cartesian_product(sets: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let total: usize = sets.iter().map(Vec::len).product();
    let mut out = vec![vec![0.0; sets.len()]; total];
    for (r, row) in out.iter_mut().enumerate() {
        let mut q = r;
        for (i, s) in sets.iter().enumerate() {
            row[i] = s[q % s.len()];
            q /= s.len();
        }
    }
    out
}

/// `cubature_with_gaussian_weight(d, n, 'Stroud')` for `n` = 3 or 5: nodes (`d x m`, one
/// per column) and weights.
pub fn stroud(d: usize, n: usize) -> Result<(Mat, Vec<f64>)> {
    match n {
        3 => {
            let r = (d as f64).sqrt();
            let mut nodes = Mat::zeros(d, 2 * d);
            for i in 0..d {
                nodes[(i, i)] = r;
                nodes[(i, d + i)] = -r;
            }
            Ok((nodes, vec![1.0 / (2 * d) as f64; 2 * d]))
        }
        5 => {
            let df = d as f64;
            let r = (df + 2.0).sqrt();
            let s = ((df + 2.0) / 2.0).sqrt();
            let m = 2 * d * d + 1;
            // (Dynare uses the degree n where the dimension is expected in A and B; the
            // weights are normalized by setup_integration_nodes.)
            let nf = n as f64;
            let a = 2.0 / (nf + 2.0);
            let b = (4.0 - df) / (2.0 * (nf + 2.0) * (nf + 2.0));
            let c = 1.0 / ((nf + 2.0) * (nf + 2.0));
            let mut nodes = Mat::zeros(d, m);
            let mut w = vec![0.0; m];
            w[0] = a;
            for i in 0..d {
                nodes[(i, 1 + i)] = r;
                w[1 + i] = b;
                nodes[(i, 1 + d + i)] = -r;
                w[1 + d + i] = b;
            }
            let mut skip = 1 + 2 * d;
            for i in 0..d.saturating_sub(1) {
                for j in i + 1..d {
                    // ee(d,i,j): e_i+e_j, e_i-e_j, -(e_i-e_j), -(e_i+e_j)
                    let cols = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
                    for (q, (si, sj)) in cols.iter().enumerate() {
                        nodes[(i, skip + q)] = s * si;
                        nodes[(j, skip + q)] = s * sj;
                        w[skip + q] = c;
                    }
                    skip += 4;
                }
            }
            Ok((nodes, w))
        }
        _ => bail!("cubature_with_gaussian_weight:: Cubature (Stroud tables) is not yet implemented with n = {n}!"),
    }
}

/// Port of `setup_integration_nodes.m`: nodes (`nnodes x n`, one node per row, already
/// correlated by `omega`, the upper Cholesky factor of the covariance of the shocks with
/// positive variance) and weights.
pub fn setup_integration_nodes(alg: Integration, quadrature_nodes: usize, omega: &Mat) -> Result<(Mat, Vec<f64>)> {
    let n = omega.nrows();
    match alg {
        Integration::Tensor => {
            let (nodes0, weights0) = gauss_hermite(quadrature_nodes);
            // nodes0 = repmat(nodes0, 1, n) * Omega
            let rep = Mat::from_fn(quadrature_nodes, n, |i, _| nodes0[i]);
            let corr = &rep * omega;
            let sets: Vec<Vec<f64>> = (0..n).map(|i| corr.column(i).iter().copied().collect()).collect();
            let wsets: Vec<Vec<f64>> = (0..n).map(|_| weights0.clone()).collect();
            let rows = cartesian_product(&sets);
            let wrows = cartesian_product(&wsets);
            let nodes = Mat::from_fn(rows.len(), n, |r, c| rows[r][c]);
            let weights = wrows.iter().map(|r| r.iter().product()).collect();
            Ok((nodes, weights))
        }
        Integration::Stroud3 | Integration::Stroud5 => {
            if alg == Integration::Stroud5 && n == 1 {
                println!(
                    "Warning: Stroud-Cubature-5 is not defined for a single shock, falling back to Gaussian quadrature with 3 nodes."
                );
                let (nodes, weights) = gauss_hermite(3);
                let nodes = Mat::from_fn(3, 1, |i, _| nodes[i] * omega[(0, 0)]);
                return Ok((nodes, weights));
            }
            let (nodes, w) = stroud(n, if alg == Integration::Stroud3 { 3 } else { 5 })?;
            // transpose(Omega' * nodes)
            let nodes = (omega.transpose() * nodes).transpose();
            let sw: f64 = w.iter().sum();
            Ok((nodes, w.iter().map(|v| v / sw).collect()))
        }
        Integration::Unscented => {
            let k = 3.0;
            let nf = n as f64;
            let c = omega.transpose() * (nf + k).sqrt();
            let mut nodes = Mat::zeros(2 * n + 1, n);
            for i in 0..n {
                for j in 0..n {
                    nodes[(1 + i, j)] = -c[(i, j)];
                    nodes[(1 + n + i, j)] = c[(i, j)];
                }
            }
            let mut w = vec![1.0 / (2.0 * (nf + k)); 2 * n + 1];
            w[0] = k / (nf + k);
            Ok((nodes, w))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_hermite_moments() {
        let (x, w) = gauss_hermite(5);
        let e = [-2.020182870, -0.9585724646, 0.0, 0.9585724646, 2.020182870];
        for (a, b) in x.iter().zip(e) {
            assert!((a / 2f64.sqrt() - b).abs() < 1e-8);
        }
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let (x, w) = gauss_hermite(9);
        let m2: f64 = x.iter().zip(&w).map(|(x, w)| w * x * x).sum();
        let m4: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(4)).sum();
        assert!((m2 - 1.0).abs() < 1e-12 && (m4 - 3.0).abs() < 1e-11);
    }

    #[test]
    fn stroud5_moments() {
        let d = 3;
        let (nodes, w) = stroud(d, 5).unwrap();
        let sw: f64 = w.iter().sum();
        for i in 0..d {
            let m2: f64 = (0..nodes.ncols()).map(|c| w[c] / sw * nodes[(i, c)].powi(2)).sum();
            assert!(m2 > 0.0);
        }
    }

    #[test]
    fn tensor_product_order() {
        let p = cartesian_product(&[vec![1.0, 2.0], vec![10.0, 20.0, 30.0]]);
        assert_eq!(p[0], vec![1.0, 10.0]);
        assert_eq!(p[1], vec![2.0, 10.0]);
        assert_eq!(p[2], vec![1.0, 20.0]);
        assert_eq!(p.len(), 6);
    }
}
