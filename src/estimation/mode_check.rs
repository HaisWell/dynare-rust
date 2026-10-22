// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `estimation/mode_check.m`: the posterior kernel (and the likelihood kernel)
//! along each parameter axis around the mode, all other parameters fixed. The evaluations
//! (parameters x points) run in parallel on the rayon pool; plotting is in `plots.rs`.

use rayon::prelude::*;

use super::{Bounds, Objective};

/// `options_.mode_check`.
#[derive(Clone, Debug)]
pub struct ModeCheckOptions {
    /// `mode_check_neighbourhood_size` (0.5).
    pub neighbourhood_size: f64,
    /// `mode_check_symmetric_plots` (true).
    pub symmetric_plots: bool,
    /// `mode_check_number_of_points` (20).
    pub number_of_points: f64,
    /// `mode_check.nolik` (false): skip the likelihood kernel.
    pub nolik: bool,
    /// `options_.huge_number` (1e7).
    pub huge_number: f64,
}

impl Default for ModeCheckOptions {
    fn default() -> Self {
        ModeCheckOptions {
            neighbourhood_size: 0.5,
            symmetric_plots: true,
            number_of_points: 20.0,
            nolik: false,
            huge_number: 1e7,
        }
    }
}

/// One parameter's slice (`mcheck.cross.(name)` = `[z, -y]`).
#[derive(Clone, Debug)]
pub struct ModeCheckSlice {
    /// Value of the parameter at the mode (`mcheck.emode.(name)`).
    pub mode: f64,
    /// Grid (sorted, `union(z1, z2)`).
    pub z: Vec<f64>,
    /// Log posterior kernel `-fval` at each point (NaN where the model fails).
    pub log_post: Vec<f64>,
    /// Log likelihood kernel `-(fval + lnprior(x) - lnprior(mode))` (NaN with `nolik`).
    pub log_lik: Vec<f64>,
    /// A bound within 1e-8 of the mode: (value, is_lower).
    pub binding_bound: Option<(f64, bool)>,
}

/// MATLAB's colon operator `a:d:b` for floating-point steps (`colonop.m`): the count is
/// rounded with a tolerance, the end point is hit exactly when it lies on the grid, the
/// first half is built from `a` and the second half from the end, the middle element of an
/// even count is `(a+b)/2`.
pub fn matlab_colon(a: f64, d: f64, b: f64) -> Vec<f64> {
    if !(a.is_finite() && d.is_finite() && b.is_finite()) || d == 0.0 || (b - a) / d < 0.0 {
        return if a == b && d != 0.0 { vec![a] } else { vec![] };
    }
    let tol = 2.0 * f64::EPSILON * a.abs().max(b.abs());
    let sig = d.signum();
    let mut n = ((b - a) / d).round();
    if sig * (a + n * d - b) > tol {
        n -= 1.0;
    }
    let n = n.max(0.0) as usize;
    let end = if (a + n as f64 * d - b).abs() <= tol { b } else { a + n as f64 * d };
    let mut v = vec![0.0; n + 1];
    for k in 0..=n / 2 {
        v[k] = a + k as f64 * d;
        v[n - k] = end - k as f64 * d;
    }
    if n.is_multiple_of(2) {
        v[n / 2] = (a + end) / 2.0;
    }
    v
}

/// The grid of parameter `kk` (`l1`, `l2` and `z` of mode_check.m).
pub fn mode_check_grid(
    x: f64,
    lb: f64,
    ub: f64,
    prior_sd: f64,
    opts: &ModeCheckOptions,
) -> (Vec<f64>, Option<(f64, bool)>) {
    let tol_bounds = 1e-8;
    let ll = opts.neighbourhood_size;
    let symmetric = opts.symmetric_plots && !ll.is_infinite();
    let (mut l1, mut l2);
    let mut m1 = false;
    if x != 0.0 && !lb.is_infinite() && !ub.is_infinite() {
        l1 = lb.max((1.0 - x.signum() * ll) * x);
        l2 = ub.min((1.0 + x.signum() * ll) * x);
    } else {
        // (sic) mode_check.m swaps the bounds here
        let mut upper_bound = lb;
        if upper_bound.is_infinite() {
            upper_bound = -1e-6 * opts.huge_number;
        }
        let mut lower_bound = ub;
        if lower_bound.is_infinite() {
            lower_bound = -1e-6 * opts.huge_number;
        }
        l1 = lower_bound.max(-prior_sd);
        l2 = upper_bound.min(prior_sd);
    }
    let mut binding = None;
    if (x - lb).abs() < tol_bounds {
        binding = Some((lb, true));
    } else if (x - ub).abs() < tol_bounds {
        binding = Some((ub, false));
    }
    if symmetric && binding.is_none() {
        if l2 < (1.0 + ll) * x {
            l1 = x - (l2 - x);
            m1 = true;
        }
        if !m1 && l1 > (1.0 - ll) * x && x + (x - l1) < ub {
            l2 = x + (x - l1);
        }
    }
    let half = opts.number_of_points / 2.0;
    let z1 = matlab_colon(l1, (x - l1) / half, x);
    let z2 = matlab_colon(x, (l2 - x) / half, l2);
    let mut z: Vec<f64> = z1.into_iter().chain(z2).filter(|v| !v.is_nan()).collect();
    z.sort_by(|a, b| a.partial_cmp(b).unwrap());
    z.dedup();
    (z, binding)
}

/// Runs mode_check.m's evaluations: for every parameter, the objective (minus log
/// posterior kernel) on its grid. `log_prior` is `priordens`. A point counts as failed
/// (NaN) when the objective is not finite or returns a nonzero info.
pub fn mode_check(
    obj: &dyn Objective,
    log_prior: &(dyn Fn(&[f64]) -> f64 + Sync),
    xparam: &[f64],
    bounds: &Bounds,
    prior_sd: &[f64],
    opts: &ModeCheckOptions,
) -> Vec<ModeCheckSlice> {
    let n = xparam.len();
    let grids: Vec<(Vec<f64>, Option<(f64, bool)>)> =
        (0..n).map(|k| mode_check_grid(xparam[k], bounds.lb[k], bounds.ub[k], prior_sd[k], opts)).collect();
    let dy = log_prior(xparam);
    let tasks: Vec<(usize, usize)> =
        grids.iter().enumerate().flat_map(|(k, (z, _))| (0..z.len()).map(move |i| (k, i))).collect();
    let values: Vec<(f64, f64)> = tasks
        .par_iter()
        .map(|&(k, i)| {
            let mut xx = xparam.to_vec();
            xx[k] = grids[k].0[i];
            let (f, info) = obj.eval(&xx);
            let y1 = if info == 0 && f.is_finite() { f } else { f64::NAN };
            let y2 = if opts.nolik { f64::NAN } else { y1 + log_prior(&xx) - dy };
            (y1, y2)
        })
        .collect();
    let mut out: Vec<ModeCheckSlice> = grids
        .into_iter()
        .enumerate()
        .map(|(k, (z, b))| ModeCheckSlice {
            mode: xparam[k],
            log_post: Vec::with_capacity(z.len()),
            log_lik: Vec::with_capacity(z.len()),
            z,
            binding_bound: b,
        })
        .collect();
    for (&(k, _), (y1, y2)) in tasks.iter().zip(values) {
        out[k].log_post.push(-y1);
        out[k].log_lik.push(-y2);
    }
    out
}

/// The console output of mode_check.m (`MODE CHECK`, fval at the mode, most negative
/// variance of the Hessian diagonal).
pub fn print_mode_check_header(fval: f64, hh: Option<&crate::linalg::Mat>, names: &[String], xparam: &[f64]) {
    let Some(hh) = hh else { return };
    println!("\nMODE CHECK\n\nFval obtained by the optimization routine: {fval:.6}");
    let (k, s_min) =
        (0..hh.nrows()).map(|i| (i, hh[(i, i)])).fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
    if s_min < f64::EPSILON {
        println!("Most negative variance {s_min:.6} for parameter {} ({} = {:.6})", k + 1, names[k], xparam[k]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colon_like_matlab() {
        assert_eq!(matlab_colon(0.0, 0.1, 1.0).len(), 11);
        let v = matlab_colon(0.1, (0.7 - 0.1) / 10.0, 0.7);
        assert_eq!(v.len(), 11);
        assert_eq!(v[10], 0.7);
        assert_eq!(v[5], (0.1 + 0.7) / 2.0);
        // descending
        let v = matlab_colon(0.5, (-10.0 - 0.5) / 10.0, -10.0);
        assert_eq!((v.len(), v[0], v[10]), (11, 0.5, -10.0));
        assert!(matlab_colon(1.0, 0.1, 0.0).is_empty());
    }

    #[test]
    fn grid_bounded_and_unbounded() {
        let o = ModeCheckOptions::default();
        // bounded beta-like parameter in (0, 1) at 0.8: l1 = 0.4, l2 = min(1, 1.2) = 1 < 1.2
        // -> symmetric: l1 = 0.8 - 0.2 = 0.6
        let (z, b) = mode_check_grid(0.8, 0.0, 1.0, 0.1, &o);
        assert!(b.is_none());
        assert_eq!(z.len(), 21);
        assert!((z[0] - 0.6).abs() < 1e-15 && (z[20] - 1.0).abs() < 1e-15);
        // unbounded (normal prior): Dynare's swapped bounds give l2 = -10, l1 = 2x+10
        let (z, _) = mode_check_grid(0.5, f64::NEG_INFINITY, f64::INFINITY, 0.2, &o);
        assert!((z[0] + 10.0).abs() < 1e-12 && (z[z.len() - 1] - 11.0).abs() < 1e-12);
        // binding lower bound: no symmetrization
        let (z, b) = mode_check_grid(1e-10, 0.0, 1.0, 0.1, &o);
        assert_eq!(b, Some((0.0, true)));
        assert!(z[0] >= 0.0);
    }

    #[test]
    fn quadratic_slices() {
        struct Q;
        impl Objective for Q {
            fn dim(&self) -> usize {
                2
            }
            fn eval(&self, x: &[f64]) -> (f64, i32) {
                if x[1] > 1.15 {
                    return (f64::INFINITY, 3);
                }
                ((x[0] - 1.0).powi(2) + 2.0 * (x[1] - 1.0).powi(2), 0)
            }
        }
        let lp = |x: &[f64]| -0.5 * x[0] * x[0];
        let b = Bounds { lb: vec![0.0, 0.0], ub: vec![5.0, 5.0] };
        let s = mode_check(&Q, &lp, &[1.0, 1.0], &b, &[1.0, 1.0], &ModeCheckOptions::default());
        let i = s[0].z.iter().position(|&v| v == 1.0).unwrap();
        assert_eq!(s[0].log_post[i], 0.0);
        assert_eq!(s[0].log_lik[i], 0.0);
        let j = 0;
        let z = s[0].z[j];
        assert!((s[0].log_post[j] + (z - 1.0).powi(2)).abs() < 1e-14);
        assert!((s[0].log_lik[j] + (z - 1.0).powi(2) + lp(&[z, 1.0]) - lp(&[1.0, 1.0])).abs() < 1e-14);
        assert!(s[1].log_post.iter().zip(&s[1].z).all(|(v, z)| (*z > 1.15) == v.is_nan()));
    }
}
