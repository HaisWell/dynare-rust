// SPDX-License-Identifier: GPL-3.0-or-later
//! Sigma points: `spherical_radial_sigma_points.m` (cubature) and `unscented_sigma_points.m`.

use super::{Approximation, ParticleOptions, Unscented};
use crate::linalg::Mat;

/// Nodes (one per row, as MATLAB's `nodes`), mean weights and covariance weights.
#[derive(Debug, Clone)]
pub struct SigmaPoints {
    pub nodes: Mat,
    pub wm: Vec<f64>,
    pub wc: Vec<f64>,
}

impl SigmaPoints {
    pub fn len(&self) -> usize {
        self.nodes.nrows()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.nrows() == 0
    }
}

/// `spherical_radial_sigma_points(n)`: `sqrt(n) * [I -I]'`, weights `1/(2n)`.
pub fn spherical_radial(n: usize) -> SigmaPoints {
    let s = (n as f64).sqrt();
    let mut nodes = Mat::zeros(2 * n, n);
    for i in 0..n {
        nodes[(i, i)] = s;
        nodes[(n + i, i)] = -s;
    }
    let w = 1.0 / (2.0 * n as f64);
    SigmaPoints { nodes, wm: vec![w; 2 * n], wc: vec![w; 2 * n] }
}

/// `unscented_sigma_points(n, unscented_options)` (Julier & Uhlmann, scaled).
pub fn unscented(n: usize, o: &Unscented) -> SigmaPoints {
    let nf = n as f64;
    let lambda = o.alpha * o.alpha * (nf + o.kappa) - nf;
    let s = (nf + lambda).sqrt();
    let mut nodes = Mat::zeros(2 * n + 1, n);
    for i in 0..n {
        nodes[(1 + i, i)] = s;
        nodes[(1 + n + i, i)] = -s;
    }
    let w0 = lambda / (nf + lambda);
    let wc0 = w0 + (1.0 - o.alpha * o.alpha + o.beta);
    let t = 1.0 / (2.0 * (nf + lambda));
    let mut wm = vec![t; 2 * n + 1];
    let mut wc = vec![t; 2 * n + 1];
    wm[0] = w0;
    wc[0] = wc0;
    SigmaPoints { nodes, wm, wc }
}

/// Sigma points of the deterministic approximations (`None` for Monte Carlo).
pub fn for_approximation(a: Approximation, n: usize, o: &ParticleOptions) -> Option<SigmaPoints> {
    match a {
        Approximation::Cubature => Some(spherical_radial(n)),
        Approximation::Unscented => Some(unscented(n, &o.unscented)),
        Approximation::MonteCarlo => None,
    }
}
