// SPDX-License-Identifier: GPL-3.0-or-later
//! Theoretical moments at order 1: ports of `matrix_solver/lyapunov_symm.m` (with the
//! unit-root Schur vectors it returns), `moments/th_autocovariances.m` (stationary-only
//! moments for models with unit roots, and the HP-filtered spectral version) and the numbers
//! of `moments/disp_th_moments.m`.

use anyhow::{Result, bail};
use nalgebra::{Complex, DMatrix};

use super::vardec::lower_cholesky;
use crate::kalman::stein_quasi_triangular;
use crate::linalg::schur::{ordschur, schur};
use crate::linalg::{Mat, mul};
use crate::solver::DecisionRule;

/// Settings of the theoretical moments (`options_` fields, Dynare defaults).
#[derive(Clone, Debug)]
pub struct MomentOptions {
    /// `options_.ar` (5).
    pub ar: usize,
    /// `options_.hp_filter` (0 = no filter).
    pub hp_filter: f64,
    pub nodecomposition: bool,
    /// `options_.qz_criterium` (1 + 1e-6).
    pub qz_criterium: f64,
    /// `options_.schur_vec_tol` (1e-11).
    pub schur_vec_tol: f64,
    /// `options_.lyapunov_complex_threshold` (1e-15).
    pub lyapunov_complex_threshold: f64,
    /// `options_.filtered_theoretical_moments_grid` (512).
    pub grid: usize,
    /// `options_.add_tiny_number_to_cholesky` (1e-14).
    pub add_tiny_number_to_cholesky: f64,
    /// `options_.bandpass.passband` when `bandpass_filter` is set.
    pub bandpass: Option<[f64; 2]>,
}

impl Default for MomentOptions {
    fn default() -> Self {
        MomentOptions {
            ar: 5,
            hp_filter: 0.0,
            nodecomposition: false,
            qz_criterium: 1.0 + 1e-6,
            schur_vec_tol: 1e-11,
            lyapunov_complex_threshold: 1e-15,
            grid: 512,
            add_tiny_number_to_cholesky: 1e-14,
            bandpass: None,
        }
    }
}

/// The real Schur form of `lyapunov_symm.m` (methods 1/2 keep it between calls): unit roots
/// (|eig| > 2 - qz_criterium) ordered first and split off.
pub struct LyapunovSchur {
    /// `T(k+1:end, k+1:end)`.
    t: Mat,
    /// `U(:, k+1:end)`.
    u_stable: Mat,
    /// `U(:, 1:k)`: the unit-root Schur vectors (`u` output of lyapunov_symm).
    pub u_unit: Mat,
    /// 1x1 transition (`size(a,1) == 1` branch).
    scalar: Option<f64>,
}

impl LyapunovSchur {
    pub fn new(a: &Mat, qz_criterium: f64) -> Result<Self> {
        let n = a.nrows();
        if n == 1 {
            return Ok(LyapunovSchur {
                t: Mat::zeros(0, 0),
                u_stable: Mat::zeros(1, 0),
                u_unit: Mat::zeros(1, 0),
                scalar: Some(a[(0, 0)]),
            });
        }
        if n == 0 {
            return Ok(LyapunovSchur {
                t: Mat::zeros(0, 0),
                u_stable: Mat::zeros(0, 0),
                u_unit: Mat::zeros(0, 0),
                scalar: None,
            });
        }
        let mut sch = schur(a)?;
        let e1: Vec<bool> = sch.eig_abs().iter().map(|&e| e > 2.0 - qz_criterium).collect();
        let k = e1.iter().filter(|&&s| s).count();
        if k > 0 {
            ordschur(&mut sch, &e1)?;
        }
        let ns = n - k;
        Ok(LyapunovSchur {
            t: sch.t.view((k, k), (ns, ns)).into_owned(),
            u_stable: sch.u.columns(k, ns).into_owned(),
            u_unit: sch.u.columns(0, k).into_owned(),
            scalar: None,
        })
    }

    /// X = A X A' + B on the stable subspace (`lyapunov_symm(a, b, ...)`).
    pub fn solve(&self, b: &Mat, lyapunov_complex_threshold: f64) -> Mat {
        if let Some(a) = self.scalar {
            return b / (1.0 - a * a);
        }
        if b.nrows() == 0 {
            return Mat::zeros(0, 0);
        }
        let u2 = &self.u_stable;
        let bb = mul(&mul(u2, true, b, false), false, u2, false);
        let x = if bb.nrows() == 0 {
            bb
        } else {
            stein_quasi_triangular(&self.t, &bb, |s| s.abs() < lyapunov_complex_threshold)
        };
        mul(&mul(u2, false, &x, false), false, u2, true)
    }
}

/// `oo_.gamma_y` and friends for the requested variables `ivar` (declaration indices).
#[derive(Clone, Debug)]
pub struct TheoreticalMoments {
    pub ivar: Vec<usize>,
    /// Positions (in `ivar`) of the stationary variables (`stationary_vars`).
    pub stationary: Vec<usize>,
    /// `gamma_y{1}`: nvar x nvar covariance, NaN rows/columns for nonstationary variables.
    pub gamma0: Mat,
    /// `gamma_y{2..ar+1}`: autocorrelation matrices (corr(y_t, y_{t-k}')).
    pub autocorr: Vec<Mat>,
    /// `gamma_y{ar+2}`: nvar x exo_nbr shares (fractions), NaN for nonstationary variables.
    pub vardec: Option<Mat>,
}

impl TheoreticalMoments {
    /// `oo_.mean` (disp_th_moments): steady state, NaN for nonstationary variables.
    pub fn mean(&self, dr: &DecisionRule) -> Vec<f64> {
        (0..self.ivar.len())
            .map(|k| if self.stationary.contains(&k) { dr.ys[self.ivar[k]] } else { f64::NAN })
            .collect()
    }
    /// Positions with `abs(diag(gamma_y{1})) > 1e-12` (`i1` of disp_th_moments).
    pub fn nonzero_variance(&self) -> Vec<usize> {
        (0..self.ivar.len()).filter(|&k| self.gamma0[(k, k)].abs() > 1e-12).collect()
    }
    /// `corr(i1, i1)`.
    pub fn correlation(&self) -> Mat {
        let n = self.ivar.len();
        let i1 = self.nonzero_variance();
        let mut c = Mat::from_element(n, n, f64::NAN);
        for &a in &i1 {
            for &b in &i1 {
                c[(a, b)] = self.gamma0[(a, b)] / (self.gamma0[(a, a)].sqrt() * self.gamma0[(b, b)].sqrt());
            }
        }
        c
    }
}

/// Port of `th_autocovariances.m` at order 1. `sigma_e` = `M_.Sigma_e`.
pub fn th_autocovariances(
    dr: &DecisionRule,
    sigma_e: &Mat,
    ivar: &[usize],
    o: &MomentOptions,
) -> Result<TheoreticalMoments> {
    let nvar = ivar.len();
    let nexo = sigma_e.nrows();
    let (nstatic, nspred) = (dr.nstatic, dr.nspred);
    let a = dr.ghx.rows(nstatic, nspred).into_owned();
    let ghu_s = dr.ghu.rows(nstatic, nspred).into_owned();
    let rows: Vec<usize> = ivar.iter().map(|&v| dr.inv_order_var[v]).collect();

    if o.hp_filter != 0.0 || o.bandpass.is_some() {
        return th_autocovariances_filtered(dr, sigma_e, ivar, &a, &ghu_s, &rows, o);
    }
    let b_sig_b = mul(&mul(&ghu_s, false, sigma_e, false), false, &ghu_s, true);
    let lyap = LyapunovSchur::new(&a, o.qz_criterium)?;
    let p = lyap.solve(&b_sig_b, o.lyapunov_complex_threshold);
    let mut stationary: Vec<usize> = (0..nvar).collect();
    if lyap.u_unit.ncols() > 0 {
        let x = mul(&dr.ghx, false, &lyap.u_unit, false);
        stationary = (0..nvar).filter(|&k| (0..x.ncols()).all(|j| x[(rows[k], j)].abs() < o.schur_vec_tol)).collect();
    }
    let srows: Vec<usize> = stationary.iter().map(|&k| rows[k]).collect();
    let ns = stationary.len();
    let aa = dr.ghx.select_rows(&srows);
    let bb = dr.ghu.select_rows(&srows);
    let mut v = mul(&mul(&aa, false, &p, false), false, &aa, true);
    v += mul(&mul(&bb, false, sigma_e, false), false, &bb, true);
    let mut gamma0 = Mat::from_element(nvar, nvar, f64::NAN);
    for (i, &si) in stationary.iter().enumerate() {
        for (j, &sj) in stationary.iter().enumerate() {
            let x = v[(i, j)];
            gamma0[(si, sj)] = if x.abs() < 1e-12 { 0.0 } else { x };
        }
    }
    let mut autocorr = Vec::new();
    if o.ar > 0 {
        let mut vxy = mul(&mul(&a, false, &p, false), false, &aa, true);
        vxy += mul(&mul(&ghu_s, false, sigma_e, false), false, &bb, true);
        let sy: Vec<f64> = stationary.iter().map(|&k| gamma0[(k, k)].sqrt()).collect();
        for lag in 1..=o.ar {
            if lag > 1 {
                vxy = mul(&a, false, &vxy, false);
            }
            let c = mul(&aa, false, &vxy, false);
            let mut g = Mat::from_element(nvar, nvar, f64::NAN);
            for i in 0..ns {
                for j in 0..ns {
                    // A variance below 1e-12 is set to 0 above; Dynare then divides its
                    // (exactly zero) numerator by 0 and gets NaN. Rounding can leave a
                    // ~1e-17 numerator here, so a zero denominator gives NaN directly.
                    let d = sy[i] * sy[j];
                    g[(stationary[i], stationary[j])] = if d == 0.0 { f64::NAN } else { c[(i, j)] / d };
                }
            }
            autocorr.push(g);
        }
    }
    let vardec = if !o.nodecomposition && nexo > 0 && ns > 0 {
        let var_stat: Vec<f64> = (0..ns).map(|i| gamma0[(stationary[i], stationary[i])]).collect();
        Some(super::vardec::compute_variance_decomposition(
            &lyap,
            &aa,
            &dr.ghu,
            &ghu_s,
            &srows,
            &stationary,
            nvar,
            sigma_e,
            &var_stat,
            o,
        ))
    } else {
        None
    };
    Ok(TheoreticalMoments { ivar: ivar.to_vec(), stationary, gamma0, autocorr, vardec })
}

/// The `hp_filter` branch of th_autocovariances.m (Uhlig 2001): spectral density of the
/// filtered series on `filtered_theoretical_moments_grid` frequencies, inverted by the
/// inverse DFT (only lags 0..ar are needed). All variables count as stationary.
fn th_autocovariances_filtered(
    dr: &DecisionRule,
    sigma_e: &Mat,
    ivar: &[usize],
    a: &Mat,
    ghu_s: &Mat,
    rows: &[usize],
    o: &MomentOptions,
) -> Result<TheoreticalMoments> {
    let nvar = ivar.len();
    let nexo = sigma_e.nrows();
    let ns = a.nrows();
    let ngrid = o.grid;
    if ngrid == 0 {
        bail!("filtered_theoretical_moments_grid must be positive");
    }
    let aa = dr.ghx.select_rows(rows);
    let bb = dr.ghu.select_rows(rows);
    let lambda = o.hp_filter;
    let decomp = !o.nodecomposition && nexo > 1;
    let cs = if decomp { lower_cholesky(sigma_e, o.add_tiny_number_to_cholesky) } else { Mat::zeros(0, 0) };
    let ss = if decomp { mul(&cs, false, &cs, true) } else { Mat::zeros(0, 0) };
    type C = Complex<f64>;
    let cz = |m: &Mat| DMatrix::<C>::from_fn(m.nrows(), m.ncols(), |i, j| C::new(m[(i, j)], 0.0));
    let (ac, ghuc, aac, bbc) = (cz(a), cz(ghu_s), cz(&aa), cz(&bb));
    let sigc = cz(sigma_e);
    let nlag = o.ar + 1;
    let mut acc = vec![Mat::zeros(nvar, nvar); nlag];
    let mut acc_vv = vec![0.0; nvar];
    let mut acc_i = Mat::zeros(nvar, nexo);
    let two_pi = 2.0 * std::f64::consts::PI;
    // MATLAB's grid `0 : 2 pi/ngrid : 2 pi (1 - .5/ngrid)` and the bandpass gain
    let freqs = super::q2::matlab_colon(0.0, two_pi / ngrid as f64, two_pi * (1.0 - 0.5 / ngrid as f64));
    let gains = match o.bandpass {
        Some([h, l]) => super::q2::bandpass_gain(&freqs, h, l),
        None => super::q2::hp_gain(&freqs, lambda),
    };
    for ig in 0..ngrid {
        let w = freqs[ig];
        let gain = gains[ig];
        if gain == 0.0 {
            continue;
        }
        let tneg = C::new(w.cos(), -w.sin());
        // G = aa * (I - A e^{-iw})^{-1} ghu_s * e^{-iw} + bb;  f_hp = gain^2/(2 pi) G Sigma G^H
        let g = if ns > 0 {
            let m = DMatrix::<C>::identity(ns, ns) - &ac * tneg;
            let sol =
                m.lu().solve(&ghuc).ok_or_else(|| anyhow::anyhow!("th_autocovariances: singular I - A e^(-iw)"))?;
            &aac * sol * tneg + &bbc
        } else {
            bbc.clone()
        };
        let scale = gain * gain / two_pi;
        let gs = &g * &sigc;
        let f = &gs * g.adjoint();
        for (k, acck) in acc.iter_mut().enumerate() {
            let ph = two_pi * (ig * k % ngrid) as f64 / ngrid as f64;
            let e = C::new(ph.cos(), ph.sin());
            for i in 0..nvar {
                for j in 0..nvar {
                    acck[(i, j)] += (f[(i, j)] * e).re * scale;
                }
            }
        }
        if decomp {
            let gss = &g * cz(&ss);
            for i in 0..nvar {
                let mut d = C::new(0.0, 0.0);
                for k in 0..nexo {
                    d += gss[(i, k)] * g[(i, k)].conj();
                }
                acc_vv[i] += d.re * scale;
                for s in 0..nexo {
                    let mut z = C::new(0.0, 0.0);
                    for k in 0..nexo {
                        z += g[(i, k)] * cs[(k, s)];
                    }
                    acc_i[(i, s)] += z.norm_sqr() * scale;
                }
            }
        }
    }
    let norm = two_pi / ngrid as f64; // real(ifft(.)) * 2 pi
    let gamma0 = &acc[0] * norm;
    let sy: Vec<f64> = (0..nvar).map(|k| gamma0[(k, k)].sqrt()).collect();
    let autocorr =
        (1..nlag).map(|k| Mat::from_fn(nvar, nvar, |i, j| acc[k][(i, j)] * norm / (sy[i] * sy[j]))).collect();
    let vardec = if o.nodecomposition || nexo == 0 {
        None
    } else if nexo == 1 {
        Some(Mat::from_element(nvar, 1, 1.0))
    } else {
        Some(Mat::from_fn(nvar, nexo, |i, s| (acc_i[(i, s)] * norm).abs() / (acc_vv[i] * norm)))
    };
    Ok(TheoreticalMoments { ivar: ivar.to_vec(), stationary: (0..nvar).collect(), gamma0, autocorr, vardec })
}
