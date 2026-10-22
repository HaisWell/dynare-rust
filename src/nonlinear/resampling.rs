// SPDX-License-Identifier: GPL-3.0-or-later
//! Resampling: `resample.m`, `traditional_resampling.m` (Kitagawa / stratified),
//! `multivariate_smooth_resampling.m` + `univariate_smooth_resampling.m`, `kitagawa.m`.
//!
//! Particles are stored one per column here (MATLAB passes them transposed, one per row).

use super::rng::Draws;
use super::{ParticleOptions, ResamplingMethod};
use crate::linalg::{Mat, sym_eig};
use anyhow::{Result, bail};

/// Port of `traditional_resampling.m` returning the (0-based) indices. One noise value =
/// Kitagawa's systematic resampling; `N` values = the "stratified" branch, which in Dynare 7.2
/// computes `fliplr(cumprod(noise.^(1./(N:-1:1))))` with `noise` a column, i.e. (implicit
/// expansion) the N x N matrix whose first column after `fliplr` is `cumprod(noise)`:
/// `randvec(i) = prod(noise(1:i))`, ported as such.
pub fn traditional_indices(weights: &[f64], noise: &[f64]) -> Vec<usize> {
    let n = weights.len();
    let mut c = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &w in weights {
        acc += w;
        c.push(acc);
    }
    let mut indx = vec![0usize; n];
    if noise.len() == 1 {
        let mut j = 0;
        for (i, ix) in indx.iter_mut().enumerate() {
            let r = (i as f64 + noise[0]) / n as f64;
            while j + 1 < n && r > c[j] {
                j += 1;
            }
            *ix = j;
        }
    } else {
        assert_eq!(noise.len(), n, "traditional_resampling: noise must have 1 or N entries");
        let mut p = 1.0;
        for (i, ix) in indx.iter_mut().enumerate() {
            p *= noise[i];
            let k = c.iter().filter(|&&cj| p > cj).count();
            *ix = k.min(n - 1);
        }
    }
    indx
}

/// `kitagawa.m` (`estimation/smc`): `m` systematic draws from `weights` with offset `noise`.
pub fn kitagawa(weights: &[f64], noise: f64, m: usize) -> Vec<usize> {
    let mut c = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for &w in weights {
        acc += w;
        c.push(acc);
    }
    let nw = weights.len();
    let mut j = 0;
    (0..m)
        .map(|i| {
            let r = (i as f64 + noise) * (1.0 / m as f64);
            while j + 1 < nw && r > c[j] {
                j += 1;
            }
            j
        })
        .collect()
}

/// `resample(0, weights, ParticleOptions)`: the resampled indices.
pub fn resample_indices(weights: &[f64], o: &ParticleOptions, draws: &mut dyn Draws) -> Result<Vec<usize>> {
    match o.resampling_method {
        ResamplingMethod::Kitagawa => {
            let u = draws.rand1();
            Ok(traditional_indices(weights, &[u]))
        }
        ResamplingMethod::Stratified => {
            let mut u = vec![0.0; weights.len()];
            draws.rand(&mut u);
            Ok(traditional_indices(weights, &u))
        }
        ResamplingMethod::Smooth => bail!("Particle = 0 is incompatible with the resampling_method=smooth method!"),
    }
}

/// `resample(particles', weights', ParticleOptions)'`: `particles` is k x N (one particle per
/// column), returns the resampled k x N particles.
pub fn resample(particles: &Mat, weights: &[f64], o: &ParticleOptions, draws: &mut dyn Draws) -> Result<Mat> {
    match o.resampling_method {
        ResamplingMethod::Kitagawa | ResamplingMethod::Stratified => {
            let idx = resample_indices(weights, o, draws)?;
            Ok(gather_columns(particles, &idx))
        }
        ResamplingMethod::Smooth => Ok(multivariate_smooth_resampling(particles, weights, draws)),
    }
}

/// `particles(:, idx)`.
pub fn gather_columns(p: &Mat, idx: &[usize]) -> Mat {
    let k = p.nrows();
    let mut out = Mat::zeros(k, idx.len());
    for (c, &j) in idx.iter().enumerate() {
        out.column_mut(c).copy_from(&p.column(j));
    }
    out
}

/// Port of `multivariate_smooth_resampling.m` (Pitt): decorrelate the particles with the
/// eigenvectors of their second moment, resample each component with the continuous
/// (piecewise linear) `univariate_smooth_resampling.m`, transform back. `particles`: k x N.
pub fn multivariate_smooth_resampling(particles: &Mat, weights: &[f64], draws: &mut dyn Draws) -> Mat {
    let (k, n) = (particles.nrows(), particles.ncols());
    // particles'*(particles/N) with particles N x k (MATLAB layout) = P P' / N here
    let mut m2 = Mat::zeros(k, k);
    for c in 0..n {
        for i in 0..k {
            let pi = particles[(i, c)];
            for j in 0..k {
                m2[(i, j)] += pi * (particles[(j, c)] / n as f64);
            }
        }
    }
    // (The signs of the eigenvectors are LAPACK's choice and differ between MATLAB's and
    // Accelerate's builds, so this resampling agrees with Dynare only statistically.)
    let (d, p) = sym_eig(&m2);
    // vectors = P .* sqrt(D)'
    let vectors = Mat::from_fn(k, k, |i, j| p[(i, j)] * d[j].sqrt());
    // orthogonalized (N x k) = (particles' * vectors) ./ D'
    let orth = Mat::from_fn(n, k, |r, j| {
        let mut s = 0.0;
        for i in 0..k {
            s += particles[(i, r)] * vectors[(i, j)];
        }
        s / d[j]
    });
    let mut newp = Mat::zeros(n, k);
    for j in 0..k {
        let mut ord: Vec<usize> = (0..n).collect();
        ord.sort_by(|&a, &b| orth[(a, j)].partial_cmp(&orth[(b, j)]).unwrap_or(std::cmp::Ordering::Equal));
        let w: Vec<f64> = ord.iter().map(|&r| weights[r]).collect();
        let x: Vec<f64> = ord.iter().map(|&r| orth[(r, j)]).collect();
        let col = univariate_smooth_resampling(&w, &x, n, draws);
        for r in 0..n {
            newp[(r, j)] = col[r];
        }
    }
    // new_particles * vectors' -> back to k x N
    Mat::from_fn(k, n, |i, r| {
        let mut s = 0.0;
        for j in 0..k {
            s += newp[(r, j)] * vectors[(i, j)];
        }
        s
    })
}

/// Port of `univariate_smooth_resampling.m` (Pitt 2002): `particles` sorted ascending,
/// `weights` in the same order; one uniform draw.
pub fn univariate_smooth_resampling(
    weights: &[f64],
    particles: &[f64],
    n_new: usize,
    draws: &mut dyn Draws,
) -> Vec<f64> {
    let m = particles.len();
    let mut lambda_tilde = Vec::with_capacity(m + 1);
    lambda_tilde.push(0.5 * weights[0]);
    for i in 0..m - 1 {
        lambda_tilde.push(0.5 * (weights[i] + weights[i + 1]));
    }
    lambda_tilde.push(0.5 * weights[m - 1]);
    let mut lambda_bar = Vec::with_capacity(m + 1);
    let mut acc = 0.0;
    for &l in &lambda_tilde {
        acc += l;
        lambda_bar.push(acc);
    }
    let u = draws.rand1();
    let mut out = vec![0.0; n_new];
    // 1-based rj and j as in MATLAB
    let mut rj = 0usize;
    let mut j = 1usize;
    for (i, o) in out.iter_mut().enumerate() {
        let u_j = (i as f64 + u) / n_new as f64;
        while j <= m + 1 && u_j > lambda_bar[j - 1] {
            rj = j;
            j += 1;
        }
        *o = if rj == 0 {
            particles[0]
        } else if rj >= m {
            particles[m - 1]
        } else {
            let u_star = (u_j - lambda_bar[rj - 1]) / lambda_tilde[rj];
            (particles[rj] - particles[rj - 1]) * u_star + particles[rj - 1]
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kitagawa_systematic() {
        let w = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(traditional_indices(&w, &[0.5]), vec![1, 2, 3, 3]);
        assert_eq!(kitagawa(&w, 0.5, 4), vec![1, 2, 3, 3]);
    }

    #[test]
    fn stratified_is_dynare_cumprod() {
        let w = [0.25; 4];
        // randvec = [0.9, 0.45, 0.405, 0.081]: counts of c = [.25 .5 .75 1] below
        assert_eq!(traditional_indices(&w, &[0.9, 0.5, 0.9, 0.2]), vec![3, 1, 1, 0]);
    }
}
