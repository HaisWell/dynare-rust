// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `hessian.m`: the finite-difference Hessian Dynare computes at the
//! mode after `mode_compute = 4` and `mode_compute = 5` (with the default
//! `newrat.hess = 1`), see `dynare_estimation_1.m`.

use rayon::prelude::*;

use super::Objective;
use crate::linalg::Mat;

/// Port of `hessian.m` (Abramowitz–Stegun 25.3.23 / 25.3.27), on the raw
/// objective (`options_.hessian.use_penalized_objective = false`).
/// `gstep` is `options_.gstep` (default `[1e-2, 1.0]`).
/// The `n(n-1)/2 + 2n + 1` evaluations run in parallel (rayon); the result is
/// identical to the sequential order.
pub fn hessian(obj: &dyn Objective, x: &[f64], gstep: [f64; 2]) -> Mat {
    let n = x.len();
    let e6 = f64::EPSILON.powf(1.0 / 6.0);
    let mut h1 = vec![0.0; n];
    let mut h_1 = vec![0.0; n];
    for i in 0..n {
        let h = x[i].abs().max(gstep[0].sqrt()) * e6 * gstep[1];
        h1[i] = (x[i] + h) - x[i];
        h_1[i] = x[i] - (x[i] - h);
    }
    let f = |z: &[f64]| obj.eval(z).0;
    let f0 = f(x);
    let (f1, f_1): (Vec<f64>, Vec<f64>) = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut z = x.to_vec();
            z[i] = x[i] + h1[i];
            let a = f(&z);
            z[i] = x[i] - h_1[i];
            (a, f(&z))
        })
        .unzip();
    let temp: Vec<f64> = (0..n).map(|i| f1[i] + f_1[i] - f0).collect();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect();
    let off: Vec<f64> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut zp = x.to_vec();
            let mut zm = x.to_vec();
            zp[i] = x[i] + h1[i];
            zp[j] = x[j] + h_1[j];
            zm[i] = x[i] - h1[i];
            zm[j] = x[j] - h_1[j];
            -(-f(&zp) - f(&zm) + temp[i] + temp[j]) / (2.0 * h1[i] * h_1[j])
        })
        .collect();
    let mut hh = Mat::zeros(n, n);
    for i in 0..n {
        hh[(i, i)] = (f1[i] + f_1[i] - 2.0 * f0) / (h1[i] * h_1[i]);
    }
    for (&(i, j), v) in pairs.iter().zip(off) {
        hh[(i, j)] = v;
        hh[(j, i)] = v;
    }
    hh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::FnObjective;

    #[test]
    fn dynare_unit_test() {
        // hessian.m unit test: f = x^2 log(y) at (1, e).
        let obj = FnObjective { n: 2, f: |v: &[f64]| v[0] * v[0] * v[1].ln() };
        let e = std::f64::consts::E;
        let h = hessian(&obj, &[1.0, e], [1e-2, 1.0]);
        let truth = [2.0, 2.0 / e, 2.0 / e, -1.0 / (e * e)];
        for (k, t) in truth.iter().enumerate() {
            assert!((h[(k % 2, k / 2)] - t).abs() < 1e-6);
        }
    }
}
