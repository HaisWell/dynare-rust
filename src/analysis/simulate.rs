// SPDX-License-Identifier: GPL-3.0-or-later
//! Stochastic simulation at order 1 and simulated moments: ports of
//! `stochastic_solver/simult.m` / `simult_.m`, `sample_hp_filter.m` and the numbers of
//! `moments/disp_moments.m`.
//!
//! Random numbers: Dynare draws `randn(nxs, periods)` from MATLAB's generator; dynare-rust
//! draws standard normals from ChaCha8 seeded with `--seed` (stream [`SIMUL_STREAM`]), so
//! simulated paths are reproducible but are not MATLAB's; given the same shocks
//! ([`simulated_moments`] takes them as input) every number matches Dynare's.

use anyhow::{Result, bail};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::linalg::{Mat, chol_upper, mul};
use crate::solver::DecisionRule;

/// ChaCha8 stream used for `stoch_simul(periods=...)` shocks.
pub const SIMUL_STREAM: u64 = 0x5354_4f43_4853_494d;

/// `simult.m`'s shocks: `exo_simul(:, i_exo_var) = randn(nxs, periods)' * chol(Sigma_e(i,i))`
/// for the shocks with non-zero variance; periods x exo_nbr.
pub fn draw_shocks(sigma_e: &Mat, periods: usize, seed: u64) -> Result<Mat> {
    let nexo = sigma_e.nrows();
    let iv: Vec<usize> = (0..nexo).filter(|&i| sigma_e[(i, i)] != 0.0).collect();
    let sub = Mat::from_fn(iv.len(), iv.len(), |a, b| sigma_e[(iv[a], iv[b])]);
    let Some(chol) = chol_upper(&sub) else {
        bail!("simult: the covariance matrix of the shocks is not positive definite");
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(SIMUL_STREAM);
    // randn(nxs, periods) is filled column by column: period after period.
    let mut z = Mat::zeros(periods, iv.len());
    for t in 0..periods {
        for k in 0..iv.len() {
            z[(t, k)] = StandardNormal.sample(&mut rng);
        }
    }
    let e = mul(&z, false, &chol, false);
    let mut out = Mat::zeros(periods, nexo);
    for (k, &i) in iv.iter().enumerate() {
        out.column_mut(i).copy_from(&e.column(k));
    }
    Ok(out)
}

/// ---- shocks/shock_paths: [`draw_shocks`] for skewed shocks (`M_.Skew_e` non-empty):
/// closed skew normal draws (`crate::analysis::skew::draw_skewed_shocks`) from the same
/// ChaCha8 normal stream.
pub fn draw_skewed(sigma_e: &Mat, skew: &crate::analysis::skew::SkewEntries, periods: usize, seed: u64) -> Result<Mat> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(SIMUL_STREAM);
    crate::analysis::skew::draw_skewed_shocks(sigma_e, skew, periods, &mut || StandardNormal.sample(&mut rng))
}

/// `simult_(M_, options_, y0, dr, ex_, 1)`: n_endo x (periods + 1) levels, column 0 = `y0`.
pub fn simult(dr: &DecisionRule, y0: &[f64], ex: &Mat) -> Mat {
    let n = dr.ghx.nrows();
    let periods = ex.nrows();
    let (nstatic, nspred) = (dr.nstatic, dr.nspred);
    let mut out = Mat::zeros(n, periods + 1);
    // deviations in DR order
    let mut y: Vec<f64> = (0..n).map(|r| y0[dr.order_var[r]] - dr.ys[dr.order_var[r]]).collect();
    for r in 0..n {
        out[(dr.order_var[r], 0)] = y0[dr.order_var[r]];
    }
    let eps = mul(&dr.ghu, false, ex, true); // n x periods
    for t in 0..periods {
        let st: Vec<f64> = y[nstatic..nstatic + nspred].to_vec();
        for (r, yr) in y.iter_mut().enumerate() {
            *yr = (0..nspred).map(|k| dr.ghx[(r, k)] * st[k]).sum::<f64>() + eps[(r, t)];
        }
        for (r, &yr) in y.iter().enumerate() {
            let v = dr.order_var[r];
            out[(v, t + 1)] = yr + dr.ys[v];
        }
    }
    out
}

/// Port of `sample_hp_filter.m`: HP cycle of each column of `y` (T x k), solving the
/// symmetric positive definite pentadiagonal system `D trend = y` by banded Cholesky.
pub fn sample_hp_filter(y: &Mat, lambda: f64) -> Mat {
    let t = y.nrows();
    let s = lambda;
    if t < 3 {
        return Mat::zeros(t, y.ncols());
    }
    // diagonals d0 (main), d1 (first off), d2 (second off)
    let mut d0 = vec![1.0 + 6.0 * s; t];
    let mut d1 = vec![-4.0 * s; t - 1];
    let d2 = vec![s; t - 2];
    d0[0] = 1.0 + s;
    d0[t - 1] = 1.0 + s;
    d1[0] = -2.0 * s;
    d1[t - 2] = -2.0 * s;
    d0[1] = 1.0 + 5.0 * s;
    d0[t - 2] = 1.0 + 5.0 * s;
    // L with diagonal l0 and sub-diagonals l1, l2 (D = L L')
    let mut l0 = vec![0.0; t];
    let mut l1 = vec![0.0; t];
    let mut l2 = vec![0.0; t];
    for i in 0..t {
        if i >= 2 {
            l2[i] = d2[i - 2] / l0[i - 2];
        }
        if i >= 1 {
            let a = d1[i - 1] - if i >= 2 { l2[i] * l1[i - 1] } else { 0.0 };
            l1[i] = a / l0[i - 1];
        }
        let diag = d0[i] - l1[i] * l1[i] - l2[i] * l2[i];
        l0[i] = diag.sqrt();
    }
    let mut out = y.clone();
    for c in 0..y.ncols() {
        let mut z = vec![0.0; t];
        for i in 0..t {
            let mut v = y[(i, c)];
            if i >= 1 {
                v -= l1[i] * z[i - 1];
            }
            if i >= 2 {
                v -= l2[i] * z[i - 2];
            }
            z[i] = v / l0[i];
        }
        let mut x = vec![0.0; t];
        for i in (0..t).rev() {
            let mut v = z[i];
            if i + 1 < t {
                v -= l1[i + 1] * x[i + 1];
            }
            if i + 2 < t {
                v -= l2[i + 2] * x[i + 2];
            }
            x[i] = v / l0[i];
        }
        for i in 0..t {
            out[(i, c)] = y[(i, c)] - x[i];
        }
    }
    out
}

/// `get_filtered_time_series.m` without the one-sided HP / bandpass filters.
pub fn filter_series(y: &Mat, hp_filter: f64) -> Mat {
    if hp_filter != 0.0 {
        sample_hp_filter(y, hp_filter)
    } else {
        let mut out = y.clone();
        for c in 0..y.ncols() {
            let m = y.column(c).mean();
            out.column_mut(c).add_scalar_mut(-m);
        }
        out
    }
}

/// The numbers of `disp_moments.m`.
#[derive(Clone, Debug)]
pub struct SimulatedMoments {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub variance: Vec<f64>,
    pub skewness: Vec<f64>,
    pub kurtosis: Vec<f64>,
    /// `oo_.var`.
    pub var: Mat,
    pub corr: Mat,
    /// `oo_.autocorr{1..ar}`.
    pub autocorr: Vec<Mat>,
    /// `oo_.variance_decomposition` (percent), simulating one shock at a time.
    pub vardec: Option<Mat>,
}

/// Port of `disp_moments.m` (no measurement errors): `endo_simul` is n_endo x periods
/// (`oo_.endo_simul`, without the initial point), `exo_simul` periods x exo_nbr.
#[allow(clippy::too_many_arguments)]
pub fn simulated_moments(
    dr: &DecisionRule,
    sigma_e: &Mat,
    endo_simul: &Mat,
    exo_simul: &Mat,
    ivar: &[usize],
    drop: usize,
    ar: usize,
    hp_filter: f64,
    nodecomposition: bool,
) -> Result<SimulatedMoments> {
    let filter = if hp_filter != 0.0 { super::q2::Filter::Hp(hp_filter) } else { super::q2::Filter::None };
    simulated_moments_q2(sigma_e, endo_simul, exo_simul, ivar, drop, ar, &filter, nodecomposition, &|ex| {
        simult(dr, &dr.ys, ex)
    })
}

/// [`simulated_moments`] with any filter of `get_filtered_time_series.m` and any
/// simulator (`simult_` at the solution's order): `simulate(ex)` returns n_endo x (periods+1)
/// from the steady state (column 0 = initial point), as `simult_(M_, options_, y0, dr, ex, order)`.
#[allow(clippy::too_many_arguments)]
pub fn simulated_moments_q2(
    sigma_e: &Mat,
    endo_simul: &Mat,
    exo_simul: &Mat,
    ivar: &[usize],
    drop: usize,
    ar: usize,
    filter: &super::q2::Filter,
    nodecomposition: bool,
    simulate: &dyn Fn(&Mat) -> Mat,
) -> Result<SimulatedMoments> {
    let periods = endo_simul.ncols();
    let nvar = ivar.len();
    let y = Mat::from_fn(periods - drop, nvar, |t, k| endo_simul[(ivar[k], drop + t)]);
    let mean: Vec<f64> = (0..nvar).map(|k| y.column(k).mean()).collect();
    let y = super::q2::filtered_series(&y, &mean, filter)?;
    let tt = y.nrows();
    let tf = tt as f64;
    let colmean = |f: &dyn Fn(f64) -> f64, k: usize| y.column(k).iter().map(|&v| f(v)).sum::<f64>() / tf;
    let mut s2: Vec<f64> = (0..nvar).map(|k| colmean(&|v| v * v, k)).collect();
    let mut s: Vec<f64> = s2.iter().map(|v| v.sqrt()).collect();
    let mut var = mul(&y, true, &y, false) / tf;
    let mut skewness: Vec<f64> = (0..nvar).map(|k| colmean(&|v| v * v * v, k) / s2[k].powf(1.5)).collect();
    let mut kurtosis: Vec<f64> = (0..nvar).map(|k| colmean(&|v| v.powi(4), k) / (s2[k] * s2[k]) - 3.0).collect();
    let zero: Vec<usize> = (0..nvar).filter(|&k| s[k].abs() < 1e-10).collect();
    for &k in &zero {
        skewness[k] = f64::NAN;
        kurtosis[k] = f64::NAN;
        s[k] = 0.0;
        s2[k] = 0.0;
        for j in 0..nvar {
            var[(k, j)] = 0.0;
            var[(j, k)] = 0.0;
        }
    }
    let mut corr = Mat::from_fn(nvar, nvar, |i, j| {
        let c: f64 = (0..tt).map(|t| y[(t, i)] * y[(t, j)]).sum::<f64>() / tf;
        c / (s[i] * s[j])
    });
    let nanify = |m: &mut Mat| {
        for &k in &zero {
            for j in 0..nvar {
                m[(k, j)] = f64::NAN;
                m[(j, k)] = f64::NAN;
            }
        }
    };
    nanify(&mut corr);
    let std_n1 = |k: usize, from: usize, to: usize| {
        let n = (to - from) as f64;
        let m = (from..to).map(|t| y[(t, k)]).sum::<f64>() / n;
        ((from..to).map(|t| (y[(t, k)] - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    };
    let mut autocorr = Vec::new();
    for i in 1..=ar {
        if ar >= tt {
            break;
        }
        let n = tt - ar;
        let sa: Vec<f64> = (0..nvar).map(|k| std_n1(k, ar, tt)).collect();
        let sb: Vec<f64> = (0..nvar).map(|k| std_n1(k, ar - i, tt - i)).collect();
        let mut m = Mat::from_fn(nvar, nvar, |a, b| {
            let c: f64 = (0..n).map(|t| y[(ar + t, a)] * y[(ar - i + t, b)]).sum();
            c / (n as f64 * sa[a] * sb[b])
        });
        nanify(&mut m);
        autocorr.push(m);
    }
    let nexo = sigma_e.nrows();
    let vardec = if nodecomposition {
        None
    } else if nexo == 1 {
        Some(Mat::from_element(nvar, 1, 100.0))
    } else {
        let iv: Vec<usize> = (0..nexo).filter(|&i| sigma_e[(i, i)] != 0.0).collect();
        let sub = Mat::from_fn(iv.len(), iv.len(), |a, b| sigma_e[(iv[a], iv[b])]);
        let chol = chol_upper(&sub).ok_or_else(|| anyhow::anyhow!("disp_moments: Sigma_e is not positive definite"))?;
        // shock_mat(:, i_exo_var) = exo_simul(:, i_exo_var) / chol_S
        let ex_sub = Mat::from_fn(exo_simul.nrows(), iv.len(), |t, k| exo_simul[(t, iv[k])]);
        let mut z = ex_sub.transpose(); // solve chol' * z' = ex'
        crate::linalg::trsm_left(true, true, 1.0, &chol, &mut z);
        let shock_mat = z.transpose();
        let mut vd = Mat::zeros(nvar, nexo);
        for (k, &i) in iv.iter().enumerate() {
            let mut single = Mat::zeros(exo_simul.nrows(), iv.len());
            single.column_mut(k).copy_from(&shock_mat.column(k));
            let ex_k = mul(&single, false, &chol, false);
            let mut ex = Mat::zeros(exo_simul.nrows(), nexo);
            for (kk, &ii) in iv.iter().enumerate() {
                ex.column_mut(ii).copy_from(&ex_k.column(kk));
            }
            let sim = simulate(&ex);
            let ys = Mat::from_fn(periods - drop, nvar, |t, a| sim[(ivar[a], 1 + drop + t)]);
            let ms: Vec<f64> = (0..nvar).map(|a| ys.column(a).mean()).collect();
            let ys = super::q2::filtered_series(&ys, &ms, filter)?;
            for a in 0..nvar {
                let n = ys.nrows() as f64;
                let m = ys.column(a).mean();
                let v = ys.column(a).iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
                vd[(a, i)] = v / s2[a] * 100.0;
            }
        }
        for &k in &zero {
            for j in 0..nexo {
                vd[(k, j)] = f64::NAN;
            }
        }
        Some(vd)
    };
    Ok(SimulatedMoments { mean, std: s, variance: s2, skewness, kurtosis, var, corr, autocorr, vardec })
}
