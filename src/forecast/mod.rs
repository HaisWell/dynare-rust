// SPDX-License-Identifier: GPL-3.0-or-later
//! Forecasts (stream F): Dynare's `forecasts.run` / `forecasts.stochastic_classical` (point
//! forecast with confidence bands at one parameter vector, order 1) and
//! `simulate_posterior_forecasts` (the per-draw forecasts of `prior_posterior_statistics_core.m`),
//! plus the full-state-space smoother they start from, CSV output and figures.
//!
//! Conventions: "DR order" vectors are indexed like `dr.ghx` rows (`order_var`), "declaration
//! order" like `M_.endo_names`. Deviations are from the steady state `ys`.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rand::Rng;
use rand_distr::StandardNormal;

pub mod commands; // driver entry points (forecast, shock_decomposition, calib_smoother)
pub mod conditional; // stream C: conditional_forecast

use crate::dsge::{DsgeModel, LikelihoodData, Resolved};
use crate::estimation::special::norminv;
use crate::graphics::{self, Color, Figure, GraphOptions, LineStyle, Panel, Series};
use crate::kalman::{SmootherOutput, compute_pinf_pstar};
use crate::linalg::{Mat, chol_upper, mul};
use crate::solver::{DecisionRule, StateSpaceIndex, kalman_transition_matrix};

// ---------------------------------------------------------------------------
// smoother on the full state space (DsgeSmoother.m with smoother_var_list = all endo)
// ---------------------------------------------------------------------------

/// DsgeSmoother at one parameter vector, kept in DR order (all endogenous variables,
/// auxiliary ones included), plus the model solution it was computed with.
pub struct FullSmooth {
    pub resolved: Resolved,
    /// alphahat, etahat, ahat (updated), filtered (one step ahead), all deviations.
    pub out: SmootherOutput,
    /// `oo_.Smoother.Constant` (store_smoother_results.m), declaration order: the steady state,
    /// `log(ys)` under `loglinear` (aux variables of lagged/leaded exogenous variables
    /// excepted) and, under `prefilter`, the sample mean for the observables.
    pub constant: Vec<f64>,
    /// The constant `simult_` adds back (`dr.ys`, logged under `loglinear`): [`Self::constant`]
    /// without the prefilter mean.
    pub ys_constant: Vec<f64>,
}

impl FullSmooth {
    pub fn dr(&self) -> &DecisionRule {
        &self.resolved.dr
    }
    pub fn ys(&self) -> &[f64] {
        &self.resolved.steady_state
    }
    /// States (mm x nobs, DR order, deviations) -> levels, nobs x nvars, declaration order
    /// restricted to `vars`: deviations + [`Self::constant`] (store_smoother_results.m; the
    /// same constants as prior_posterior_statistics_core.m's smoothed, updated and filtered
    /// variables, except that the latter also logs the steady state of exogenous aux variables).
    pub fn levels(&self, states: &Mat, vars: &[usize]) -> Mat {
        let dr = self.dr();
        Mat::from_fn(states.ncols(), vars.len(), |t, j| states[(dr.inv_order_var[vars[j]], t)] + self.constant[vars[j]])
    }
    /// Smoothed deviations from the steady state, n_endo x nobs, declaration order
    /// (`Smoothed_Variables_deviation_from_mean` of evaluate_smoother.m).
    pub fn smoothed_deviation(&self) -> Mat {
        let dr = self.dr();
        let a = &self.out.alphahat;
        Mat::from_fn(a.nrows(), a.ncols(), |v, t| a[(dr.inv_order_var[v], t)])
    }
    /// Last smoothed state, DR order, deviation (forecast start `y0`).
    pub fn last_state(&self) -> Vec<f64> {
        let a = &self.out.alphahat;
        a.column(a.ncols() - 1).iter().copied().collect()
    }
}

/// Port of DsgeSmoother.m as the driver uses it (measurement errors `M_.H`, full state space,
/// lik_init 1 or 3) at parameter vector `x`.
pub fn dsge_smoother(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    x: &[f64],
    state_uncertainty: bool,
) -> Result<FullSmooth> {
    let resolved =
        dsge.resolve(x).map_err(|e| anyhow!("smoother: no solution at the requested point (info {})", e.info))?;
    let dr = &resolved.dr;
    let full = StateSpaceIndex::full(dr);
    let (t, r) = kalman_transition_matrix(dr, &full);
    let mf: Vec<usize> = dsge.varobs.iter().map(|&v| dr.inv_order_var[v]).collect();
    let mut y = data.y.clone();
    for (i, &v) in dsge.varobs.iter().enumerate() {
        let c = dsge.level_constant(v, &resolved.steady_state); // stream M: log(ys) under loglinear
        for j in 0..y.ncols() {
            y[(i, j)] -= c;
        }
    }
    let opts = &data.kalman;
    let (pstar, pinf) = compute_pinf_pstar(&mf, &t, &r, &resolved.q, opts.qz_criterium, &[]);
    let pinf = (opts.lik_init == 3).then_some(&pinf);
    let qvec = data.hetero.as_ref().map(|hs| hs.qvec(&resolved.q, y.ncols())); // heteroskedastic_filter
    let out = match &data.skewed {
        // ---- estimation completion: kalman_algo = 5 (kalman_smoother_pruned_skewed.m) ----
        Some(spec) => crate::estimation::skewed::smoother(dsge, data, spec, x, &resolved, &y, &t, &r, &mf)?,
        None => crate::kalman::smoother_heteroskedastic(
            &y,
            &data.missing,
            &mf,
            &t,
            &r,
            &resolved.q,
            Some(&resolved.h),
            &pstar,
            pinf,
            opts,
            state_uncertainty,
            qvec.as_deref(),
        )?, // M_.H (stream M)
    };
    let ys_constant = simult_constant(dsge, &resolved.steady_state);
    let mut constant = ys_constant.clone();
    if let Some(m) = &data.prefilter_mean {
        for (i, &v) in dsge.varobs.iter().enumerate() {
            constant[v] = m[i];
        }
    }
    Ok(FullSmooth { resolved, out, constant, ys_constant })
}

/// The levels `simult_` adds to the deviations (declaration order): `dr.ys`, with
/// `loglinear` `log(ys)` except for aux variables of lagged/leaded exogenous variables
/// (`get_all_variables_but_lagged_leaded_exogenous`).
pub fn simult_constant(dsge: &DsgeModel, ys: &[f64]) -> Vec<f64> {
    (0..ys.len()).map(|v| dsge.level_constant(v, ys)).collect()
}

// ---------------------------------------------------------------------------
// forecasts
// ---------------------------------------------------------------------------

/// Point forecast with confidence bands (`oo_.forecast`): levels, horizon x nvar.
#[derive(Clone, Debug)]
pub struct Forecast {
    /// Declaration-order indices of the variables (columns).
    pub vars: Vec<usize>,
    pub names: Vec<String>,
    pub mean: Mat,
    pub lower: Mat,
    pub upper: Mat,
    pub conf_sig: f64,
    /// `HPDinf_ME` / `HPDsup_ME` (forecasts.run with `M_.H` non-zero): bands including the
    /// measurement-error variance; NaN for variables that are not observed.
    pub lower_me: Option<Mat>,
    pub upper_me: Option<Mat>,
}

/// Measurement-error bands of `forecasts.stochastic_classical` at order 1: for each observed
/// variable of the forecast, `var_yf_ME = var_yf + diag(M_.H)` and the same `-norminv` width.
/// `varobs` are the declaration indices of the observed variables (columns of `h`). A no-op
/// when `h` has no non-zero entry (`forecasts.run` then writes no `_ME` fields).
pub fn add_measurement_error_bands(f: &mut Forecast, h: &Mat, varobs: &[usize]) {
    if !h.iter().any(|&x| x != 0.0) {
        return;
    }
    let fact = norminv((1.0 - f.conf_sig) / 2.0, 0.0, 1.0);
    let (mut lo, mut up) = (
        Mat::from_element(f.mean.nrows(), f.mean.ncols(), f64::NAN),
        Mat::from_element(f.mean.nrows(), f.mean.ncols(), f64::NAN),
    );
    for (c, v) in f.vars.iter().enumerate() {
        if let Some(o) = varobs.iter().position(|x| x == v) {
            for r in 0..f.mean.nrows() {
                let w = (f.upper[(r, c)] - f.lower[(r, c)]) / 2.0;
                let var = (w / -fact).powi(2) + h[(o, o)];
                let w_me = -fact * var.sqrt();
                let m = (f.upper[(r, c)] + f.lower[(r, c)]) / 2.0;
                lo[(r, c)] = m - w_me;
                up[(r, c)] = m + w_me;
            }
        }
    }
    f.lower_me = Some(lo);
    f.upper_me = Some(up);
}

/// DR-order positions of the state variables (`k2 = dr.inv_order_var(dr.state_var)`).
fn state_rows(dr: &DecisionRule) -> Vec<usize> {
    dr.state_var.iter().map(|&s| dr.inv_order_var[s]).collect()
}

/// Port of `forecasts.stochastic_classical` (order 1, no measurement error, no
/// `varexo_det`): `yf` from `simult_` without shocks starting at `y0` (DR order, deviation
/// from the steady state), variances from `sigma_y1 = ghx*sigma_y1(states,states)*ghx' + ghu*Sigma_e*ghu'`
/// and bands `yf -/+ norminv((1-conf_sig)/2)*sqrt(var)`. `vars`: declaration order. `ys` is
/// the level `simult_` adds back, declaration order: the steady state, or under `loglinear`
/// its log ([`simult_constant`]).
#[allow(clippy::too_many_arguments)]
pub fn stochastic_classical(
    dr: &DecisionRule,
    ys: &[f64],
    y0: &[f64],
    horizon: usize,
    sigma_e: &Mat,
    vars: &[usize],
    names: &[String],
    conf_sig: f64,
) -> Forecast {
    let n = dr.ghx.nrows();
    let k2 = state_rows(dr);
    let fact = norminv((1.0 - conf_sig) / 2.0, 0.0, 1.0);
    let ghu_s = mul(&mul(&dr.ghu, false, sigma_e, false), false, &dr.ghu, true);
    let mut y = y0.to_vec();
    let mut sig = Mat::zeros(n, n);
    let nv = vars.len();
    let (mut mean, mut lower, mut upper) = (Mat::zeros(horizon, nv), Mat::zeros(horizon, nv), Mat::zeros(horizon, nv));
    for h in 0..horizon {
        let s: Vec<f64> = k2.iter().map(|&k| y[k]).collect();
        y = (0..n).map(|i| (0..k2.len()).map(|j| dr.ghx[(i, j)] * s[j]).sum()).collect();
        let ss = Mat::from_fn(k2.len(), k2.len(), |a, b| sig[(k2[a], k2[b])]);
        sig = mul(&mul(&dr.ghx, false, &ss, false), false, &dr.ghx, true) + &ghu_s;
        for (j, &v) in vars.iter().enumerate() {
            let r = dr.inv_order_var[v];
            let m = y[r] + ys[v];
            let w = -fact * sig[(r, r)].max(0.0).sqrt();
            mean[(h, j)] = m;
            lower[(h, j)] = m - w;
            upper[(h, j)] = m + w;
        }
    }
    Forecast {
        vars: vars.to_vec(),
        names: names.to_vec(),
        mean,
        lower,
        upper,
        conf_sig,
        lower_me: None,
        upper_me: None,
    }
}

/// The simulated forecast of `forecasts.stochastic_classical` / `simult_varexo_det.m` at order
/// > 1: `sim(ex)` simulates `sim_periods` periods with the shocks `ex` (`sim_periods x n_exo`)
/// > and returns `n x (sim_periods + 1)` levels (declaration order, column 0 the initial
/// > value). Mean: no shocks; bands: the `(1-conf_sig)/2` and `1-(1-conf_sig)/2` quantiles
/// > (MATLAB's `quantile`) of `replic` simulations, shocks `randn(nxs, sim_periods)'*chol(S)`
/// > on the `nxs` shocks of non-zero variance. The first `horizon` periods are kept.
#[allow(clippy::too_many_arguments)]
pub fn simulated_forecast(
    sim: &dyn Fn(&Mat) -> Mat,
    n_exo: usize,
    sim_periods: usize,
    horizon: usize,
    sigma_e: &Mat,
    vars: &[usize],
    names: &[String],
    conf_sig: f64,
    replic: usize,
    draws: &mut dyn crate::nonlinear::rng::Draws,
) -> Result<Forecast> {
    let nv = vars.len();
    let y = sim(&Mat::zeros(sim_periods, n_exo));
    let mean = Mat::from_fn(horizon, nv, |h, j| y[(vars[j], h + 1)]);
    let i_exo: Vec<usize> = (0..n_exo).filter(|&i| sigma_e[(i, i)] != 0.0).collect();
    let nxs = i_exo.len();
    let s = Mat::from_fn(nxs, nxs, |a, b| sigma_e[(i_exo[a], i_exo[b])]);
    let c = chol_upper(&s).ok_or_else(|| anyhow!("forecast: Sigma_e is not positive definite"))?;
    let mut sims: Vec<Vec<f64>> = vec![Vec::with_capacity(replic); horizon * nv];
    let mut z = vec![0.0; nxs * sim_periods];
    let mut ex = Mat::zeros(sim_periods, n_exo);
    for _ in 0..replic {
        draws.randn(&mut z); // randn(nxs, sim_periods), column-major
        for t in 0..sim_periods {
            for (a, &ia) in i_exo.iter().enumerate() {
                ex[(t, ia)] = (0..nxs).map(|b| z[t * nxs + b] * c[(b, a)]).sum();
            }
        }
        let y = sim(&ex);
        for h in 0..horizon {
            for (j, &v) in vars.iter().enumerate() {
                sims[h * nv + j].push(y[(v, h + 1)]);
            }
        }
    }
    let (p1, p2) = ((1.0 - conf_sig) / 2.0, 1.0 - (1.0 - conf_sig) / 2.0);
    let (mut lower, mut upper) = (Mat::zeros(horizon, nv), Mat::zeros(horizon, nv));
    for h in 0..horizon {
        for j in 0..nv {
            let x = &mut sims[h * nv + j];
            x.sort_by(|a, b| a.total_cmp(b));
            lower[(h, j)] = matlab_quantile(x, p1);
            upper[(h, j)] = matlab_quantile(x, p2);
        }
    }
    Ok(Forecast {
        vars: vars.to_vec(),
        names: names.to_vec(),
        mean,
        lower,
        upper,
        conf_sig,
        lower_me: None,
        upper_me: None,
    })
}

/// `quantile` as Dynare 7.2 runs it (its own `missing/stats/quantile.m`, which shadows
/// MATLAB's; default method R-8, Hyndman & Fan 1996) of sorted data:
/// `h = (n + 1/3) p + 1/3`, linear between `x_floor(h)` and the next.
pub fn matlab_quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let nf = n as f64;
    if n == 1 || p < (2.0 / 3.0) / (nf + 1.0 / 3.0) {
        return sorted[0];
    }
    if p >= (nf - 1.0 / 3.0) / (nf + 1.0 / 3.0) {
        return sorted[n - 1];
    }
    let pos = (nf + 1.0 / 3.0) * p + 1.0 / 3.0; // 1-based
    let lo = pos.floor() as usize;
    let fr = pos - lo as f64;
    sorted[lo - 1] + fr * (sorted[lo] - sorted[lo - 1])
}

/// Shock draws for [`simulate_posterior_forecasts`]: `B1 = ghu(:, i_exo_var)*chol(Sigma_e(i_exo_var, i_exo_var))'`.
pub struct ShockLoading {
    pub b1: Mat,
}

impl ShockLoading {
    pub fn new(dr: &DecisionRule, sigma_e: &Mat) -> Result<ShockLoading> {
        let i_exo: Vec<usize> = (0..sigma_e.nrows()).filter(|&i| sigma_e[(i, i)] != 0.0).collect();
        let s = Mat::from_fn(i_exo.len(), i_exo.len(), |a, b| sigma_e[(i_exo[a], i_exo[b])]);
        let c =
            chol_upper(&s).ok_or_else(|| anyhow!("simulate_posterior_forecasts: Sigma_e is not positive definite"))?;
        let ghu = Mat::from_fn(dr.ghu.nrows(), i_exo.len(), |i, j| dr.ghu[(i, i_exo[j])]);
        Ok(ShockLoading { b1: mul(&ghu, false, &c, true) })
    }
}

/// Port of `simulate_posterior_forecasts` (one repetition): forecasts from `y0` (DR order,
/// deviation), with shocks `B1*randn` when `shocks` is given. Returns horizon x n_endo,
/// declaration order, deviations (the `maxlag` initial row is dropped, as the caller does).
pub fn simulate_posterior_forecasts<R: Rng>(
    dr: &DecisionRule,
    y0: &[f64],
    horizon: usize,
    mut shocks: Option<(&ShockLoading, &mut R)>,
) -> Mat {
    let n = dr.ghx.nrows();
    let k2 = state_rows(dr);
    let mut y = y0.to_vec();
    let mut out = Mat::zeros(horizon, n);
    for h in 0..horizon {
        let s: Vec<f64> = k2.iter().map(|&k| y[k]).collect();
        let mut next: Vec<f64> = (0..n).map(|i| (0..k2.len()).map(|j| dr.ghx[(i, j)] * s[j]).sum()).collect();
        if let Some((l, rng)) = shocks.as_mut() {
            let e: Vec<f64> = (0..l.b1.ncols()).map(|_| rng.sample(StandardNormal)).collect();
            for (i, v) in next.iter_mut().enumerate() {
                *v += (0..e.len()).map(|j| l.b1[(i, j)] * e[j]).sum::<f64>();
            }
        }
        for v in 0..n {
            out[(h, v)] = next[dr.inv_order_var[v]];
        }
        y = next;
    }
    out
}

// ---------------------------------------------------------------------------
// dates, CSV, figures
// ---------------------------------------------------------------------------

/// `horizon` dates after `dates` (the estimation sample): ISO `YYYY-MM-DD` dates step by
/// the sample's month gap (keeping end-of-month days at month ends), `YYYYQq` / `YYYYMm` /
/// `YYYY` labels step by one period; anything else is numbered `nobs+1..`.
pub fn extend_dates(dates: Option<&[String]>, nobs: usize, horizon: usize) -> Vec<String> {
    let fallback = || (1..=horizon).map(|h| (nobs + h).to_string()).collect();
    let Some(d) = dates.filter(|d| !d.is_empty()) else { return fallback() };
    let last = d[d.len() - 1].trim();
    let iso = |s: &str| -> Option<(i32, u32, u32)> {
        let p: Vec<&str> = s.split('-').collect();
        if p.len() != 3 {
            return None;
        }
        Some((p[0].parse().ok()?, p[1].parse().ok()?, p[2].parse().ok()?))
    };
    let mdays = |y: i32, m: u32| -> u32 {
        match m {
            2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
            2 => 28,
            4 | 6 | 9 | 11 => 30,
            _ => 31,
        }
    };
    if let Some((y, m, day)) = iso(last) {
        let step = d
            .len()
            .checked_sub(2)
            .and_then(|i| iso(d[i].trim()))
            .map_or(3, |(y0, m0, _)| ((y - y0) * 12 + m as i32 - m0 as i32).max(1));
        let eom = day == mdays(y, m);
        return (1..=horizon as i32)
            .map(|h| {
                let k = (y * 12 + m as i32 - 1) + h * step;
                let (yy, mm) = (k.div_euclid(12), (k.rem_euclid(12) + 1) as u32);
                let dd = if eom { mdays(yy, mm) } else { day.min(mdays(yy, mm)) };
                format!("{yy:04}-{mm:02}-{dd:02}")
            })
            .collect();
    }
    for (sep, per) in [('Q', 4), ('q', 4), ('M', 12), ('m', 12)] {
        if let Some((a, b)) = last.split_once(sep)
            && let (Ok(y), Ok(p)) = (a.parse::<i32>(), b.parse::<i32>())
        {
            return (1..=horizon as i32)
                .map(|h| {
                    let k = y * per + p - 1 + h;
                    format!("{}{sep}{}", k.div_euclid(per), k.rem_euclid(per) + 1)
                })
                .collect();
        }
    }
    if let Ok(y) = last.trim_end_matches(['Y', 'y']).parse::<i32>() {
        return (1..=horizon as i32).map(|h| (y + h).to_string()).collect();
    }
    fallback()
}

/// `<fname>_forecast.csv`: `date,horizon,variable,mean,lower,upper` (long format).
pub fn write_forecast_csv(path: &Path, f: &Forecast, dates: &[String]) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut w = String::from("date,horizon,variable,mean,lower,upper");
    if f.lower_me.is_some() {
        w += ",lower_me,upper_me";
    }
    w += "\n";
    for (j, name) in f.names.iter().enumerate() {
        for h in 0..f.mean.nrows() {
            w += &format!("{},{},{},{},{},{}", dates[h], h + 1, name, f.mean[(h, j)], f.lower[(h, j)], f.upper[(h, j)]);
            if let (Some(l), Some(u)) = (&f.lower_me, &f.upper_me) {
                let e = |x: f64| if x.is_nan() { String::new() } else { x.to_string() };
                w += &format!(",{},{}", e(l[(h, j)]), e(u[(h, j)]));
            }
            w += "\n";
        }
    }
    std::fs::write(path, w).with_context(|| format!("writing {}", path.display()))
}

/// Port of `forecasts.graph`: 3x4 panels per figure (`forcst<k>`, "Forecasts (k)"), the mean
/// and both band limits as blue lines.
pub fn forecast_figures(f: &Forecast) -> Vec<Figure> {
    let (nr, nc) = (3, 4);
    let horizon = f.mean.nrows();
    let x: Vec<f64> = (1..=horizon).map(|h| h as f64).collect();
    let mut figs: Vec<Figure> = Vec::new();
    for (j, name) in f.names.iter().enumerate() {
        if j % (nr * nc) == 0 {
            let k = figs.len() + 1;
            figs.push(Figure {
                name: format!("forcst{k}"),
                title: format!("Forecasts ({k})"),
                rows: nr,
                cols: nc,
                panels: vec![],
            });
        }
        let line = |y: Vec<f64>| Series::Line {
            x: x.clone(),
            y,
            color: Color::Blue,
            style: LineStyle::Solid,
            width: 1.0,
            label: None,
        };
        let col = |m: &Mat| m.column(j).iter().copied().collect::<Vec<f64>>();
        figs.last_mut().unwrap().panels.push(Panel {
            title: name.clone(),
            series: vec![line(col(&f.mean)), line(col(&f.lower)), line(col(&f.upper))],
            ..Default::default()
        });
    }
    figs
}

/// Saves figures through [`graphics::save`], warning (not failing) on renderer errors.
pub fn save_figures(figs: &[Figure], fname: &str, g: &GraphOptions) {
    for fig in figs {
        if let Err(e) = graphics::save(fig, fname, g) {
            println!("WARNING: figure {} not saved: {e}", fig.name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates() {
        let d: Vec<String> = ["2025-12-31", "2026-03-31", "2026-06-30"].iter().map(|s| s.to_string()).collect();
        assert_eq!(extend_dates(Some(&d), 3, 3), ["2026-09-30", "2026-12-31", "2027-03-31"]);
        let q: Vec<String> = ["2026Q3", "2026Q4"].iter().map(|s| s.to_string()).collect();
        assert_eq!(extend_dates(Some(&q), 2, 2), ["2027Q1", "2027Q2"]);
        assert_eq!(extend_dates(None, 10, 2), ["11", "12"]);
    }
}
