// SPDX-License-Identifier: GPL-3.0-or-later
//! [estimation completion] Dynare 7.2's own multivariate normal CDF, used by
//! `skewed_kalman_mvnlogcdf = 'mvncdf'` (pruned skewed Kalman filter and smoother, `csn_mean.m`).
//!
//! MATLAB's `mvncdf` belongs to the Statistics Toolbox; without it Dynare puts its own
//! replacement on the path (`dynare_config.m`: `missing/stats/mvncdf.m`, with the helpers of
//! `missing/stats_matlab/`, which Dynare also adds under MATLAB). The ports here follow those
//! files (the algorithms MATLAB documents for `mvncdf`):
//!
//! * [`mvncdf`]: `missing/stats/mvncdf.m` (upper limits only, `mvncdf(XU, mu, Sigma)`): one
//!   dimension and diagonal covariances in closed form, `d = 2` [`bvncdf`], `d = 3` [`tvncdf`],
//!   `4 <= d <= 25` [`mvtcdfqmc`] with tolerance 1e-4, larger `d` an error;
//! * [`bvncdf`]: `missing/stats_matlab/bvncdf.m`, Drezner and Wesolowsky (1990) / Genz (2004)
//!   Gauss-Legendre quadrature (3, 6 or 10 points);
//! * [`tvncdf`]: `missing/stats_matlab/tvncdf.m`, Genz (2004)'s trivariate formula with MATLAB's
//!   `quadgk` (absolute tolerance `tol/3`; [`quadgk`] ports its Gauss-Kronrod (7,15) scheme on 10
//!   initial subintervals of the transformed interval, with bisection of the subintervals whose
//!   error is too large);
//! * [`mvtcdfqmc`]: `missing/stats_matlab/mvtcdfqmc.m`, Genz's randomized quasi-Monte Carlo
//!   (Richtmyer lattice, 25 random shifts, antithetic points). Its shifts come from MATLAB's
//!   global `rand` stream, which dynare-rust cannot reproduce; here they come from a ChaCha8
//!   generator with a fixed seed, so an evaluation stays a pure function of its arguments (the
//!   MCMC requires it). Both are estimates within the 1e-4 absolute tolerance (3.5 standard
//!   errors) of the exact probability, not bit-identical to MATLAB's.

#![allow(clippy::needless_range_loop, clippy::excessive_precision)] // index loops and constants mirror the MATLAB code

use anyhow::{Result, bail};

use crate::linalg::{Mat, chol_upper};

// erfc from the system libm, as MATLAB's (see crate::model::expr)
unsafe extern "C" {
    #[link_name = "erfc"]
    fn libm_erfc(x: f64) -> f64;
}

/// `erfc` (system libm).
pub fn erfc(x: f64) -> f64 {
    unsafe { libm_erfc(x) }
}

/// `0.5*erfc(-z/sqrt(2))`.
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// `-sqrt(2)*erfcinv(2p)`.
fn norm_inv(p: f64) -> f64 {
    -std::f64::consts::SQRT_2 * statrs::function::erf::erfc_inv(2.0 * p)
}

/// Port of `missing/stats/mvncdf.m` for `mvncdf(XU, mu, Sigma)` (lower limits `-Inf`): the
/// probability `P(X <= xu)`, `X ~ N(mu, Sigma)`. Errors as Dynare's (non positive definite
/// covariance, more than 25 dimensions). `d = 0` gives 1.
pub fn mvncdf(xu: &[f64], mu: &[f64], sigma: &Mat) -> Result<f64> {
    let d = xu.len();
    if d == 0 {
        return Ok(1.0);
    }
    if d > 25 {
        bail!("mvncdf: number of dimensions D must not exceed 25");
    }
    let is_diagonal = (0..d).all(|i| (0..d).all(|j| i == j || sigma[(i, j)] == 0.0));
    if is_diagonal {
        if (0..d).any(|i| sigma[(i, i)] <= 0.0) {
            bail!("mvncdf: diagonal elements of covariance matrix SIGMA must be positive");
        }
    } else if chol_upper(sigma).is_none() {
        bail!("mvncdf: the covariance matrix SIGMA must be positive semi-definite");
    }
    let s: Vec<f64> = (0..d).map(|i| sigma[(i, i)].sqrt()).collect();
    let xu0: Vec<f64> = (0..d).map(|i| (xu[i] - mu[i]) / s[i]).collect();
    let p = if d == 1 {
        norm_cdf(xu0[0]) // normcdf(XU0) - normcdf(-Inf)
    } else if is_diagonal {
        xu0.iter().map(|&z| norm_cdf(z)).product()
    } else if d == 2 {
        bvncdf(&[xu[0], xu[1]], &[mu[0], mu[1]], sigma)?
    } else {
        let rho = Mat::from_fn(d, d, |i, j| sigma[(i, j)] / (s[i] * s[j]));
        if d == 3 {
            tvncdf(&[xu0[0], xu0[1], xu0[2]], &[rho[(1, 0)], rho[(2, 0)], rho[(2, 1)]], 1e-8)?
        } else {
            mvtcdfqmc(&vec![f64::NEG_INFINITY; d], &xu0, &rho, 1e-4, 1e7)?.0
        }
    };
    Ok(p.clamp(0.0, 1.0))
}

/// Port of `missing/stats_matlab/bvncdf.m` for one point: `P(X1 <= x1, X2 <= x2)`,
/// `X ~ N(mu, Sigma)` (2x2 covariance).
pub fn bvncdf(x: &[f64; 2], mu: &[f64; 2], sigma: &Mat) -> Result<f64> {
    if chol_upper(sigma).is_none() {
        bail!("bvncdf: the covariance matrix Sigma is not positive definite and/or symmetric");
    }
    let dh = (x[0] - mu[0]) / sigma[(0, 0)].sqrt();
    let dk = (x[1] - mu[1]) / sigma[(1, 1)].sqrt();
    let r = sigma[(0, 1)] / (sigma[(0, 0)] * sigma[(1, 1)]).sqrt();
    let inf = f64::INFINITY;
    Ok(if dh == inf && dk == inf {
        1.0
    } else if dh == f64::NEG_INFINITY || dk == f64::NEG_INFINITY {
        0.0
    } else if dk == inf {
        0.5 * erfc(-dh / std::f64::consts::SQRT_2)
    } else if dh == inf {
        0.5 * erfc(-dk / std::f64::consts::SQRT_2)
    } else if dh.is_nan() || dk.is_nan() {
        f64::NAN
    } else {
        calculate_bvncdf(-dh, -dk, r)
    })
}

/// `calculate_bvncdf` of bvncdf.m: `P(X1 > dh, X2 > dk)` for a standard bivariate normal with
/// correlation `r`.
fn calculate_bvncdf(dh: f64, dk: f64, r: f64) -> f64 {
    const W3: [f64; 3] = [0.1713244923791705, 0.3607615730481384, 0.4679139345726904];
    const X3: [f64; 3] = [0.9324695142031522, 0.6612093864662647, 0.2386191860831970];
    const W6: [f64; 6] = [
        0.04717533638651177,
        0.1069393259953183,
        0.1600783285433464,
        0.2031674267230659,
        0.2334925365383547,
        0.2491470458134029,
    ];
    const X6: [f64; 6] = [
        0.9815606342467191,
        0.9041172563704750,
        0.7699026741943050,
        0.5873179542866171,
        0.3678314989981802,
        0.1252334085114692,
    ];
    const W10: [f64; 10] = [
        0.01761400713915212,
        0.04060142980038694,
        0.06267204833410906,
        0.08327674157670475,
        0.1019301198172404,
        0.1181945319615184,
        0.1316886384491766,
        0.1420961093183821,
        0.1491729864726037,
        0.1527533871307259,
    ];
    const X10: [f64; 10] = [
        0.9931285991850949,
        0.9639719272779138,
        0.9122344282513259,
        0.8391169718222188,
        0.7463319064601508,
        0.6360536807265150,
        0.5108670019508271,
        0.3737060887154196,
        0.2277858511416451,
        0.07652652113349733,
    ];
    let (w, x): (&[f64], &[f64]) = if r.abs() < 0.3 {
        (&W3, &X3)
    } else if r.abs() < 0.75 {
        (&W6, &X6)
    } else {
        (&W10, &X10)
    };
    let s2 = std::f64::consts::SQRT_2;
    let pi = std::f64::consts::PI;
    let mut dk = dk;
    let mut hk = dh * dk;
    let mut bvn;
    if r.abs() < 0.925 {
        let phi_dh = 0.5 * erfc(dh / s2);
        let phi_dk = 0.5 * erfc(dk / s2);
        let hs = (dh * dh + dk * dk) / 2.0;
        let asr = r.asin();
        let mut sum = 0.0;
        for i in 0..w.len() {
            let sn1 = (asr * (1.0 - x[i]) / 2.0).sin();
            let sn2 = (asr * (1.0 + x[i]) / 2.0).sin();
            sum +=
                w[i] * ((sn1 * hk - hs) / (1.0 - sn1 * sn1)).exp() + w[i] * ((sn2 * hk - hs) / (1.0 - sn2 * sn2)).exp();
        }
        bvn = sum * asr / (4.0 * pi) + phi_dh * phi_dk;
    } else {
        let twopi = 2.0 * pi;
        if r < 0.0 {
            dk = -dk;
            hk = -hk;
        }
        bvn = 0.0;
        if r.abs() < 1.0 {
            let as_ = (1.0 - r) * (1.0 + r);
            let mut a = as_.sqrt();
            let bs = (dh - dk).powi(2);
            let c = (4.0 - hk) / 8.0;
            let d = (12.0 - hk) / 16.0;
            let asr = -(bs / as_ + hk) / 2.0;
            if asr > -100.0 {
                bvn = a * asr.exp() * (1.0 - (c * (bs - as_)) * (1.0 - d * bs / 5.0) / 3.0 + (c * d) * as_ * as_ / 5.0);
            }
            if hk > -100.0 {
                let b = bs.sqrt();
                let phi_ba = 0.5 * erfc((b / a) / s2);
                let sp = twopi.sqrt() * phi_ba;
                bvn -= ((-hk / 2.0).exp() * sp) * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
            }
            a /= 2.0;
            for is in [-1.0, 1.0] {
                let mut sum = 0.0;
                for i in 0..w.len() {
                    let xs = (a + a * is * x[i]).powi(2);
                    let rs = (1.0 - xs).sqrt();
                    let asr1 = -(bs / xs + hk) / 2.0;
                    let ind1 = if asr1 > -100.0 { 1.0 } else { 0.0 };
                    let sp1 = 1.0 + c * xs * (1.0 + d * xs);
                    let ep1 = (-hk * (1.0 - rs) / (2.0 * (1.0 + rs))).exp() / rs;
                    sum += a * w[i] * (asr1 * ind1).exp() * (ep1 * ind1 - sp1 * ind1);
                }
                bvn += sum;
            }
            bvn = -bvn / twopi;
        }
        if r > 0.0 {
            bvn += 0.5 * erfc(dh.max(dk) / s2);
        } else if r < 0.0 {
            let phi_dh = 0.5 * erfc(dh / s2);
            let phi_dk = 0.5 * erfc(dk / s2);
            bvn = -bvn + (phi_dh - phi_dk).max(0.0);
        }
    }
    bvn.clamp(0.0, 1.0)
}

/// Port of `missing/stats_matlab/tvncdf.m` for one point: standard trivariate normal CDF at
/// `x` with correlations `rho = [rho_21, rho_31, rho_32]`.
pub fn tvncdf(x: &[f64; 3], rho: &[f64; 3], tol: f64) -> Result<f64> {
    let mut imax = 0;
    for k in 1..3 {
        if rho[k].abs() > rho[imax].abs() {
            imax = k;
        }
    }
    let (r21, r31, r32, x) = match imax {
        0 => (rho[2], rho[1], rho[0], [x[2], x[1], x[0]]),
        1 => (rho[0], rho[2], rho[1], [x[1], x[0], x[2]]),
        _ => (rho[0], rho[1], rho[2], *x),
    };
    let phi_x1 = 0.5 * erfc(-x[0] / std::f64::consts::SQRT_2);
    let s = Mat::from_row_slice(2, 2, &[1.0, r32, r32, 1.0]);
    let p1 = phi_x1 * bvncdf(&[x[1], x[2]], &[0.0, 0.0], &s)?;
    let integral = |rj1: f64, rk1: f64, b1: f64, bj: f64, bk: f64| -> f64 {
        if rj1.abs() > 0.0 && b1.is_finite() && bj.is_finite() && !bk.is_nan() {
            quadgk(&|th: f64| tvn_integrand(th, b1, bj, bk, rj1, rk1, r32), 0.0, rj1.asin(), tol / 3.0)
        } else {
            0.0
        }
    };
    let p2 = integral(r21, r31, x[0], x[1], x[2]);
    let p3 = integral(r31, r21, x[0], x[2], x[1]);
    Ok((p1 + (p2 + p3) / (2.0 * std::f64::consts::PI)).clamp(0.0, 1.0))
}

fn tvn_integrand(theta: f64, b1: f64, bj: f64, bk: f64, rho_j1: f64, rho_k1: f64, rho_32: f64) -> f64 {
    let sintheta = theta.sin();
    let cossqtheta = theta.cos().powi(2);
    let expon = ((b1 * sintheta - bj).powi(2) / cossqtheta + b1 * b1) / 2.0;
    let sinphi = sintheta * rho_k1 / rho_j1;
    let numeru = bk * cossqtheta - b1 * (sinphi - rho_32 * sintheta) - bj * (rho_32 - sintheta * sinphi);
    let denomu = (cossqtheta * (cossqtheta - sinphi * sinphi - rho_32 * (rho_32 - 2.0 * sintheta * sinphi))).sqrt();
    let phi = 0.5 * erfc(-(numeru / denomu) / std::f64::consts::SQRT_2);
    (-expon).exp() * phi
}

/// MATLAB's `quadgk(f, a, b, 'AbsTol', abstol, 'RelTol', 0)` on a finite interval: the
/// substitution `x = (b-a)/4 t (3 - t^2) + (b+a)/2` on `t` in [-1, 1], 10 initial
/// subintervals, Gauss-Kronrod (7,15) on each (error `|Q_K - Q_G|`), subintervals whose error
/// is below their share of the tolerance accepted, the others bisected (at most 650 of them).
pub fn quadgk(f: &dyn Fn(f64) -> f64, a: f64, b: f64, abstol: f64) -> f64 {
    const PN: [f64; 7] = [
        0.2077849550078985,
        0.4058451513773972,
        0.5860872354676911,
        0.7415311855993944,
        0.8648644233597691,
        0.9491079123427585,
        0.9914553711208126,
    ];
    const PW: [f64; 7] = [
        0.2044329400752989,
        0.1903505780647854,
        0.1690047266392679,
        0.1406532597155259,
        0.1047900103222502,
        0.06309209262997855,
        0.02293532201052922,
    ];
    const PW7: [f64; 7] = [0.0, 0.3818300505051189, 0.0, 0.2797053914892767, 0.0, 0.1294849661688697, 0.0];
    if a == b {
        return 0.0;
    }
    let mut nodes = [0.0; 15];
    let mut wt = [0.0; 15];
    let mut ewt = [0.0; 15];
    for k in 0..7 {
        nodes[k] = -PN[6 - k];
        nodes[8 + k] = PN[k];
        wt[k] = PW[6 - k];
        wt[8 + k] = PW[k];
        ewt[k] = PW[6 - k] - PW7[6 - k];
        ewt[8 + k] = PW[k] - PW7[k];
    }
    wt[7] = 0.2094821410847278;
    ewt[7] = 0.2094821410847278 - 0.4179591836734694;
    let (ba4, mid) = ((b - a) / 4.0, (b + a) / 2.0);
    let g = |t: f64| f(ba4 * t * (3.0 - t * t) + mid) * 3.0 * ba4 * (1.0 - t * t);
    let mut subs: Vec<(f64, f64)> = (0..10).map(|i| (-1.0 + 0.2 * i as f64, -1.0 + 0.2 * (i + 1) as f64)).collect();
    subs[9].1 = 1.0;
    let pathlen = 2.0;
    let (mut q_ok, mut err_ok) = (0.0, 0.0);
    loop {
        let mut qs = Vec::with_capacity(subs.len());
        let mut es = Vec::with_capacity(subs.len());
        for &(l, r) in &subs {
            let (m, h) = ((l + r) / 2.0, (r - l) / 2.0);
            let (mut q, mut e) = (0.0, 0.0);
            for k in 0..15 {
                let fx = g(nodes[k] * h + m);
                q += wt[k] * fx;
                e += ewt[k] * fx;
            }
            qs.push(q * h);
            es.push(e * h);
        }
        let q = q_ok + qs.iter().sum::<f64>();
        let tol = abstol.max(0.0 * q.abs());
        let mut keep = Vec::new();
        let mut err_rest = 0.0;
        for (i, &(l, r)) in subs.iter().enumerate() {
            let halfh = (r - l) / 2.0;
            if es[i].abs() <= (2.0 * tol / pathlen) * halfh.abs() {
                q_ok += qs[i];
                err_ok += es[i];
            } else {
                keep.push(i);
                err_rest += es[i].abs();
            }
        }
        if keep.is_empty() || err_ok.abs() + err_rest <= tol || !q.is_finite() {
            return q;
        }
        if 2 * keep.len() > 650 {
            return q;
        }
        let mut next = Vec::with_capacity(2 * keep.len());
        for &i in &keep {
            let (l, r) = subs[i];
            let m = (l + r) / 2.0;
            next.push((l, m));
            next.push((m, r));
        }
        subs = next;
    }
}

/// Port of `missing/stats_matlab/mvtcdfqmc.m` for the normal case (`nu = Inf`): returns
/// `(p, err, funevals)` for `P(a <= X <= b)`, `X ~ N(0, Rho)`, `Rho` a correlation matrix.
/// The random shifts use a fixed-seed ChaCha8 stream instead of MATLAB's `rand` (see the
/// module documentation).
pub fn mvtcdfqmc(a: &[f64], b: &[f64], rho: &Mat, tol: f64, maxfunevals: f64) -> Result<(f64, f64, f64)> {
    use rand::{Rng, SeedableRng};
    if a.iter().zip(b).any(|(x, y)| x > y) {
        bail!("mvtcdfqmc: lower limits a must be less than upper limits b");
    }
    if a.iter().chain(b).any(|v| v.is_nan()) {
        return Ok((f64::NAN, f64::NAN, 0.0));
    }
    if !a.iter().zip(b).all(|(x, y)| x < y) {
        return Ok((0.0, 0.0, 0.0));
    }
    let keep: Vec<usize> = (0..a.len()).filter(|&i| !(a[i] == f64::NEG_INFINITY && b[i] == f64::INFINITY)).collect();
    if keep.is_empty() {
        return Ok((1.0, 0.0, 0.0));
    }
    let m = keep.len();
    // [~, ord] = sort(b - a) (stable)
    let mut ord: Vec<usize> = (0..m).collect();
    ord.sort_by(|&i, &j| {
        (b[keep[i]] - a[keep[i]]).partial_cmp(&(b[keep[j]] - a[keep[j]])).unwrap_or(std::cmp::Ordering::Equal)
    });
    let idx: Vec<usize> = ord.iter().map(|&i| keep[i]).collect();
    let r = Mat::from_fn(m, m, |i, j| rho[(idx[i], idx[j])]);
    let c =
        chol_upper(&r).ok_or_else(|| anyhow::anyhow!("mvtcdfqmc: the correlation matrix is not positive definite"))?;
    let cd: Vec<f64> = (0..m).map(|i| c[(i, i)]).collect();
    let av: Vec<f64> = (0..m).map(|i| a[idx[i]] / cd[i]).collect();
    let bv: Vec<f64> = (0..m).map(|i| b[idx[i]] / cd[i]).collect();
    let cn = Mat::from_fn(m, m, |i, j| c[(i, j)] / cd[j]);
    const MCREPS: usize = 25;
    let mcdims = m - 1;
    const P: [usize; 28] = [
        31, 47, 73, 113, 173, 263, 397, 593, 907, 1361, 2053, 3079, 4621, 6947, 10427, 15641, 23473, 35221, 52837,
        79259, 118891, 178349, 267523, 401287, 601942, 902933, 1354471, 2031713,
    ];
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x6d76_6e63_6466); // fixed: pure function
    let (mut p, mut sigsq, mut funevals, mut err) = (0.0, f64::INFINITY, 0.0, f64::NAN);
    let q: Vec<f64> = (1..=mcdims).map(|k| 2f64.powf(k as f64 / (mcdims + 1) as f64)).collect();
    let eps = f64::EPSILON;
    let f_qrsvn = |w: &dyn Fn(usize, usize) -> f64, n: usize| -> f64 {
        let mut total = 0.0;
        let mut y = vec![0.0; m];
        for row in 0..n {
            let mut d = norm_cdf(av[0]);
            let mut emd = norm_cdf(bv[0]) - d;
            let mut t = emd;
            for i in 1..m {
                let z = (d + emd * w(row, i - 1)).clamp(eps / 2.0, 1.0 - eps / 2.0);
                y[i - 1] = norm_inv(z);
                let ysum: f64 = (0..i).map(|k| y[k] * cn[(k, i)]).sum();
                d = norm_cdf(av[i] - ysum);
                emd = norm_cdf(bv[i] - ysum) - d;
                t *= emd;
            }
            total += t;
        }
        total / n as f64
    };
    for &np in &P[4..] {
        if funevals + 2.0 * MCREPS as f64 * np as f64 > maxfunevals {
            break;
        }
        let mut that = [0.0; MCREPS];
        for rep in 0..MCREPS {
            let shift: Vec<f64> = (0..mcdims).map(|_| rng.r#gen::<f64>()).collect();
            let w = |row: usize, k: usize| (2.0 * ((((row + 1) as f64) * q[k] + shift[k]).rem_euclid(1.0)) - 1.0).abs();
            let w1 = |row: usize, k: usize| 1.0 - w(row, k);
            that[rep] = (f_qrsvn(&w, np) + f_qrsvn(&w1, np)) / 2.0;
        }
        let mean = that.iter().sum::<f64>() / MCREPS as f64;
        let var = that.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (MCREPS - 1) as f64;
        let sigsq_that = var / MCREPS as f64;
        funevals += 2.0 * MCREPS as f64 * np as f64;
        p += (mean - p) / (1.0 + sigsq_that / sigsq);
        sigsq = sigsq_that / (1.0 + sigsq_that / sigsq);
        err = 3.5 * sigsq.sqrt();
        if err < tol {
            return Ok((p, err, funevals));
        }
    }
    Ok((p, err, funevals))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_forms() {
        // P(X1<=0, X2<=0) = 1/4 + asin(rho)/(2 pi)
        for rho in [0.5, -0.7, 0.95, -0.98, 0.1] {
            let s = Mat::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
            let p = mvncdf(&[0.0, 0.0], &[0.0, 0.0], &s).unwrap();
            assert!((p - (0.25 + rho.asin() / (2.0 * std::f64::consts::PI))).abs() < 1e-15, "{rho}: {p}");
        }
        // independent trivariate through the correlated path (tiny correlation)
        let p = tvncdf(&[0.5, -0.3, 0.8], &[0.0, 0.0, 0.0], 1e-8).unwrap();
        assert!((p - norm_cdf(0.5) * norm_cdf(-0.3) * norm_cdf(0.8)).abs() < 1e-15);
        let s = Mat::identity(4, 4) * 0.9 + Mat::from_element(4, 4, 0.1);
        let p = mvncdf(&[0.0; 4], &[0.0; 4], &s).unwrap();
        assert!(p > 0.0 && p < 0.5);
    }
}
