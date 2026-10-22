// SPDX-License-Identifier: GPL-3.0-or-later
//! OccBin generalized IRFs and forecasts: ports of `+occbin/irf.m` and `+occbin/forecast.m`
//! (Dynare 7.2). In Dynare these are MATLAB functions without a `.mod` command; here they are
//! [`Session::irf`] and [`Session::forecast`], with the options of
//! `options_.occbin.irf` / `options_.occbin.forecast` ([`IrfOptions`], [`ForecastOptions`]).

use anyhow::{Result, anyhow, bail};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use super::regimes::Regimes;
use super::{Session, SimulResult};
use crate::linalg::Mat;
use crate::modfile::ModFile;

/// `options_.occbin.irf` (defaults of `occbin.set_default_options`) and `options_.irf`.
#[derive(Clone, Debug)]
pub struct IrfOptions {
    /// Shocks (`exo_names`; default: all).
    pub exo_names: Vec<String>,
    /// "pos" and/or "neg".
    pub shocksigns: Vec<String>,
    /// One size per shock (or one for all); `None` = the shocks' standard deviations (0.01 when
    /// below 1e-9).
    pub shocksize: Option<Vec<f64>>,
    /// 0: from the steady state; t > 0: conditional on the smoothed state of period t and the
    /// smoothed shocks of t+1 (needs the OccBin smoother).
    pub t0: usize,
    pub init_regime: Option<Vec<Regimes>>,
    pub check_ahead_periods: usize,
    pub maxit: usize,
    /// `options_.irf`.
    pub periods: usize,
}

impl IrfOptions {
    pub fn new(mf: &ModFile) -> Self {
        IrfOptions {
            exo_names: mf.model.exo_names.clone(),
            shocksigns: vec!["pos".into(), "neg".into()],
            shocksize: None,
            t0: 0,
            init_regime: None,
            check_ahead_periods: 30,
            maxit: 30,
            periods: 40,
        }
    }
}

/// One GIRF (`irfs.piecewise.<var>_<shock>_<sign>` for all variables): (periods+1) x n_endo
/// differences between the shocked and the baseline simulations.
#[derive(Clone, Debug)]
pub struct Irf {
    pub shock: String,
    pub sign: String,
    pub piecewise: Mat,
    pub linear: Mat,
}

/// The smoothed objects a conditional GIRF (`t0 > 0`) starts from: states (deviations, DR
/// order, n x nobs), shocks (n_exo x nobs) and the DR ordering.
pub struct SmoothedStart<'a> {
    pub alphahat: &'a Mat,
    pub etahat: &'a Mat,
    pub order_var: &'a [usize],
}

/// `options_.occbin.forecast`.
#[derive(Clone, Debug)]
pub struct ForecastOptions {
    pub check_ahead_periods: usize,
    pub maxit: usize,
    /// Expected future shocks (`SHOCKS0`): name and path (one value per forecast period).
    pub shocks0: Vec<(String, Vec<f64>)>,
    pub frcst_regimes: Option<Vec<Regimes>>,
    /// Monte Carlo replications with random future shocks (0 = point forecast).
    pub replic: usize,
    pub seed: u64,
}

impl Default for ForecastOptions {
    fn default() -> Self {
        ForecastOptions {
            check_ahead_periods: 30,
            maxit: 30,
            shocks0: Vec::new(),
            frcst_regimes: None,
            replic: 0,
            seed: 0,
        }
    }
}

/// `occbin.forecast` output: periods x n_endo levels (`forecast.piecewise.Mean`,
/// `forecast.linear.Mean`; with replications, the means over the successful ones and their
/// min/max), and the regimes of the point forecast.
#[derive(Clone, Debug)]
pub struct OccbinForecast {
    pub piecewise: Mat,
    pub linear: Mat,
    pub piecewise_min: Option<Mat>,
    pub piecewise_max: Option<Mat>,
    pub regimes: Vec<Regimes>,
    /// Successful replications (replic > 0).
    pub successful: usize,
}

fn diff(a: &Mat, b: &Mat) -> Mat {
    a - b
}

impl Session {
    /// Port of `+occbin/irf.m` at the current parameters.
    pub fn irf(&mut self, mf: &ModFile, o: &IrfOptions, smoothed: Option<&SmoothedStart>) -> Result<Vec<Irf>> {
        let m = &mf.model;
        let ne = m.n_exo();
        let mut base = self.simul.clone();
        base.init_regime = o.init_regime.clone();
        base.check_ahead_periods = o.check_ahead_periods;
        base.maxit = o.maxit;
        base.periods = o.periods;
        base.exo_pos = (0..ne).collect();
        base.piecewise_only = false;
        let mut shocks_base = Mat::zeros(o.periods + 1, ne);
        let endo_init = if o.t0 == 0 {
            Vec::new()
        } else {
            let s = smoothed.ok_or_else(|| {
                anyhow!("occbin.irfs: smoother must be run before requesting GIRFs based on smoothed results")
            })?;
            for k in 0..ne {
                shocks_base[(0, k)] = s.etahat[(k, o.t0)];
            }
            let mut e = vec![0.0; m.n_endo()];
            for (r, &v) in s.order_var.iter().enumerate() {
                e[v] = s.alphahat[(r, o.t0 - 1)];
            }
            e
        };
        base.shocks = shocks_base.clone();
        base.endo_init = endo_init.clone();
        let out_base = self
            .solve_with(mf, base.clone())
            .map_err(|e| anyhow!("occbin.irfs: could not compute the solution ({e})"))?;
        let exo_index: Vec<usize> = o
            .exo_names
            .iter()
            .map(|n| m.exo_index(n).ok_or_else(|| anyhow!("occbin.irf: unknown shock {n}")))
            .collect::<Result<_>>()?;
        let shocksize: Vec<f64> = match &o.shocksize {
            None => {
                exo_index.iter().map(|&k| mf.sigma_e[(k, k)].sqrt()).map(|s| if s < 1e-9 { 0.01 } else { s }).collect()
            }
            Some(v) if v.len() == 1 => vec![v[0]; exo_index.len()],
            Some(v) => v.clone(),
        };
        let mut irfs = Vec::new();
        for sign in &o.shocksigns {
            for (c, &jexo) in exo_index.iter().enumerate() {
                let mut sh = shocks_base.clone();
                match sign.as_str() {
                    "pos" => sh[(0, jexo)] += shocksize[c],
                    "neg" => sh[(0, jexo)] -= shocksize[c],
                    s => bail!("occbin.irf: unknown shock sign {s}"),
                }
                let mut so = base.clone();
                so.shocks = sh;
                let out: SimulResult = match self.solve_with(mf, so) {
                    Ok(x) => x,
                    Err(_) => {
                        println!("WARNING: occbin.irfs: simulation failed");
                        continue;
                    }
                };
                irfs.push(Irf {
                    shock: m.exo_names[jexo].clone(),
                    sign: sign.clone(),
                    piecewise: diff(&out.piecewise, &out_base.piecewise),
                    linear: diff(out.linear.as_ref().unwrap(), out_base.linear.as_ref().unwrap()),
                });
            }
        }
        Ok(irfs)
    }

    /// Port of `+occbin/forecast.m`: from `endo_histval` (levels, `M_.endo_histval(:,1)`),
    /// `horizon` periods with the expected shocks `shocks0` (and, with `replic`, random shocks
    /// drawn from `Sigma_e` with dynare-rust's ChaCha8 stream instead of MATLAB's `randn`).
    pub fn forecast(
        &mut self,
        mf: &ModFile,
        o: &ForecastOptions,
        endo_histval: &[f64],
        horizon: usize,
    ) -> Result<OccbinForecast> {
        let m = &mf.model;
        let ne = m.n_exo();
        let mut base = self.simul.clone();
        base.maxit = o.maxit;
        base.check_ahead_periods = o.check_ahead_periods;
        base.periods = horizon;
        base.piecewise_only = false;
        // initial condition in deviations from the steady state (solve once to get ys)
        let ys = {
            let mut probe = base.clone();
            probe.shocks = Mat::zeros(1, ne);
            probe.exo_pos = (0..ne).collect();
            probe.endo_init = Vec::new();
            self.solve_with(mf, probe)?.ys
        };
        base.endo_init = endo_histval.iter().zip(&ys).map(|(h, y)| h - y).collect();
        base.init_regime = o.frcst_regimes.clone();
        base.init_binding_indicator = Vec::new();
        let mut shocks_base = Mat::zeros(horizon, ne);
        for (name, path) in &o.shocks0 {
            let k = m.exo_index(name).ok_or_else(|| anyhow!("occbin.forecast: unknown exogenous shock {name}"))?;
            if path.len() != horizon {
                bail!("occbin.forecast: the path of {name} must have {horizon} values");
            }
            for (t, v) in path.iter().enumerate() {
                shocks_base[(t, k)] = *v;
            }
        }
        if o.replic == 0 {
            base.shocks = shocks_base;
            base.exo_pos = (0..ne).collect();
            let out =
                self.solve_with(mf, base).map_err(|e| anyhow!("occbin.forecast: forecast simulation failed ({e})"))?;
            return Ok(OccbinForecast {
                piecewise: out.piecewise.clone(),
                linear: out.linear.clone().unwrap(),
                piecewise_min: None,
                piecewise_max: None,
                regimes: out.regime_history,
                successful: 1,
            });
        }
        // Monte Carlo: SHOCKS = shocks_base(:,ishock) + (U sqrt(S) draws)'
        let ishock: Vec<usize> = (0..ne).filter(|&k| mf.sigma_e[(k, k)].sqrt() != 0.0).collect();
        let k = ishock.len();
        let cov = Mat::from_fn(k, k, |i, j| mf.sigma_e[(ishock[i], ishock[j])]);
        let svd = cov.clone().svd(true, false);
        let u = svd.u.unwrap();
        let mut us = u.clone();
        for j in 0..k {
            let s = svd.singular_values[j].sqrt();
            for i in 0..k {
                us[(i, j)] *= s;
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(o.seed);
        base.exo_pos = ishock.clone();
        let n = m.n_endo();
        let (mut sum_p, mut sum_l) = (Mat::zeros(horizon, n), Mat::zeros(horizon, n));
        let mut mn = Mat::from_element(horizon, n, f64::INFINITY);
        let mut mx = Mat::from_element(horizon, n, f64::NEG_INFINITY);
        let mut ok = 0usize;
        let mut regimes = Vec::new();
        for _ in 0..o.replic {
            let draws = Mat::from_fn(k, horizon, |_, _| StandardNormal.sample(&mut rng));
            let add = &us * draws;
            let mut so = base.clone();
            so.shocks = Mat::from_fn(horizon, k, |t, j| shocks_base[(t, ishock[j])] + add[(j, t)]);
            if let Ok(out) = self.solve_with(mf, so) {
                ok += 1;
                sum_p += &out.piecewise;
                sum_l += out.linear.as_ref().unwrap();
                for t in 0..horizon {
                    for j in 0..n {
                        mn[(t, j)] = mn[(t, j)].min(out.piecewise[(t, j)]);
                        mx[(t, j)] = mx[(t, j)].max(out.piecewise[(t, j)]);
                    }
                }
                regimes = out.regime_history;
            }
        }
        if ok < o.replic {
            println!("\noccbin.forecast: forecast simulation was only successful in {ok} of {} simulations.", o.replic);
        }
        let d = ok.max(1) as f64;
        Ok(OccbinForecast {
            piecewise: sum_p / d,
            linear: sum_l / d,
            piecewise_min: Some(mn),
            piecewise_max: Some(mx),
            regimes,
            successful: ok,
        })
    }
}
