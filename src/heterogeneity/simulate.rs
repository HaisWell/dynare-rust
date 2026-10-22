// SPDX-License-Identifier: GPL-3.0-or-later
//! `heterogeneity_simulate`: port of `+heterogeneity/simulate.m`, `simulate_irfs.m`,
//! `simulate_stochastic_shocks.m` and `simulate_news_shocks.m`: impulse responses, stochastic
//! simulations and news-shock sequences of the linearised model from the general-equilibrium
//! Jacobians `dr.G` of [`super::solve::solve_hetero`].
//!
//! Graphs (`plot_irfs.m`, `plot_simulation.m`) are produced by [`plot_irfs`] / [`plot_simulation`]
//! (TeX eps-loader files are not written). Deviation: the shocks
//! of a stochastic simulation are drawn from dynare-rust's ChaCha8 stream, not from MATLAB's
//! generator (the convolution given the draws, [`stochastic_paths`], is identical).

use anyhow::{Result, bail};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use super::solve::HetDr;
use crate::analysis::vardec::lower_cholesky;
use crate::linalg::{Mat, chol_upper};
use crate::model::Model;
use crate::modfile::DetShockKind;

/// Options of `heterogeneity_simulate`; `Option` fields are `None` when the user did not set them.
#[derive(Clone, Debug, Default)]
pub struct SimOptions {
    /// `irf` (default 40).
    pub irf: Option<usize>,
    /// `periods` (default 0).
    pub periods: Option<usize>,
    pub irf_shocks: Option<Vec<String>>,
    pub relative_irf: Option<bool>,
    pub irf_plot_threshold: Option<f64>,
    /// `add_tiny_number_to_cholesky` (default 1e-14).
    pub add_tiny: Option<f64>,
    pub var_list: Vec<String>,
}

/// Results (`oo_.irfs`, `oo_.endo_simul`, `oo_.exo_simul`).
#[derive(Clone, Debug, Default)]
pub struct SimResult {
    /// `oo_.irfs.(var)_(shock)` in the order of the loops (shocks, then variables).
    pub irfs: Vec<(String, Vec<f64>)>,
    /// `oo_.endo_simul` (`endo_nbr x periods`, levels).
    pub endo_simul: Option<Mat>,
    /// `oo_.exo_simul` of the news-shock mode (`rows x exo_nbr`).
    pub exo_simul: Option<Mat>,
}

fn shock_indices(model: &Model, opts: &SimOptions) -> Result<Vec<usize>> {
    match &opts.irf_shocks {
        Some(l) if !l.is_empty() => l
            .iter()
            .map(|s| {
                model
                    .exo_names
                    .iter()
                    .position(|e| e == s)
                    .ok_or_else(|| anyhow::anyhow!("Shock \"{s}\" not found in M_.exo_names"))
            })
            .collect(),
        _ => Ok((0..model.exo_names.len()).collect()),
    }
}

/// `simulate_irfs.m`.
pub fn simulate_irfs(
    model: &Model,
    dr: &HetDr,
    sigma_e: &Mat,
    opts: &SimOptions,
    shocks: &[usize],
) -> Result<Vec<(String, Vec<f64>)>> {
    let t = dr.truncation_horizon;
    let horizon = opts.irf.unwrap_or(40);
    if horizon > t {
        bail!(
            "Requested irf={horizon} exceeds Jacobian truncation horizon {t}. Increase truncation_horizon in heterogeneity_solve or reduce irf."
        );
    }
    let cs = if sigma_e.nrows() > 0 { Some(lower_cholesky(sigma_e, opts.add_tiny.unwrap_or(1e-14))) } else { None };
    let relative = opts.relative_irf.unwrap_or(false);
    let vars: Vec<&String> =
        if opts.var_list.is_empty() { model.endo_names.iter().collect() } else { opts.var_list.iter().collect() };
    let mut out = Vec::new();
    for &si in shocks {
        let sname = &model.exo_names[si];
        let std = match &cs {
            Some(c) if c.nrows() > si => c[(si, si)],
            _ => 0.0,
        };
        let first = if relative { 100.0 } else { std };
        for v in &vars {
            if !model.endo_names.contains(v) {
                println!("Warning: Variable \"{v}\" not found, skipping");
                continue;
            }
            let Some(g) = dr.g_block(v, sname) else {
                println!("Warning: No Jacobian found for variable \"{v}\" and shock \"{sname}\", skipping");
                continue;
            };
            out.push((format!("{v}_{sname}"), (0..horizon).map(|r| g[(r, 0)] * first).collect()));
        }
    }
    Ok(out)
}

/// The convolution of `simulate_stochastic_shocks.m` given the draws `eps` (`exo_nbr x
/// (periods + T - 1)`): `endo_nbr x periods`, in levels (`y_ss + sum_s G(s, 1) eps(t + T - s)`).
pub fn stochastic_paths(
    model: &Model,
    dr: &HetDr,
    y_ss: &[f64],
    eps: &Mat,
    periods: usize,
    shocks: &[usize],
) -> Result<Mat> {
    let t = dr.truncation_horizon;
    let mut out = Mat::zeros(model.endo_names.len(), periods);
    for (iv, v) in model.endo_names.iter().enumerate() {
        for tt in 0..periods {
            out[(iv, tt)] = y_ss[iv];
        }
        for &si in shocks {
            let sname = &model.exo_names[si];
            let Some(g) = dr.g_block(v, sname) else {
                bail!("No Jacobian found for variable \"{v}\" and shock \"{sname}\". Run heterogeneity_solve first.");
            };
            for tt in 0..periods {
                // 1-based: path(t) += G(s,1) * eps(t + T - s)
                let acc: f64 = (0..t).map(|s| g[(s, 0)] * eps[(si, tt + t - 1 - s)]).sum();
                out[(iv, tt)] += acc;
            }
        }
    }
    Ok(out)
}

/// Draws of `simulate_stochastic_shocks.m` (shocks with zero variance stay zero).
pub fn draw_shocks(sigma_e: &Mat, total: usize, seed: u64) -> Result<Mat> {
    let nx = sigma_e.nrows();
    if nx == 0 {
        bail!("M_.Sigma_e is empty. Declare shock variances in shocks block.");
    }
    let active: Vec<usize> = (0..nx).filter(|&i| sigma_e[(i, i)] != 0.0).collect();
    let mut eps = Mat::zeros(nx, total);
    if active.is_empty() {
        return Ok(eps);
    }
    let sub = Mat::from_fn(active.len(), active.len(), |i, j| sigma_e[(active[i], active[j])]);
    let Some(u) = chol_upper(&sub) else {
        bail!("simulate_stochastic_shocks: the covariance matrix of the shocks is not positive definite");
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for tt in 0..total {
        let z: Vec<f64> = (0..active.len()).map(|_| StandardNormal.sample(&mut rng)).collect();
        for (a, &i) in active.iter().enumerate() {
            // (z' * chol_S)' = chol_S' z
            eps[(i, tt)] = (0..active.len()).map(|k| u[(k, a)] * z[k]).sum();
        }
    }
    Ok(eps)
}

/// `oo_.exo_simul` built by `make_ex_.m` from the deterministic shocks of the `shocks` block, with
/// `periods = 0` (the `periods` option is incompatible with news shocks).
pub fn news_exo_simul(model: &Model, det: &[crate::modfile::DetShock], max_lag: usize, max_lead: usize) -> Result<Mat> {
    let nx = model.exo_names.len();
    let mut rows = max_lag + max_lead;
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    for d in det {
        let v = match d.kind {
            DetShockKind::Level => d.value,
            DetShockKind::MultiplySteadyState => 0.0,
            DetShockKind::MultiplyInitialSteadyState => {
                bail!("Option relative_to_initval of mshocks block cannot be used without an endval block")
            }
        };
        let periods = d.sim_periods(None)?;
        for p in periods {
            let k = p + max_lag as i64;
            if k == 0 {
                bail!("make_ex_: The model does not have lags, so you cannot set values for period 0");
            }
            if k < 0 {
                bail!("make_ex_: invalid shock period");
            }
            let k = k as usize;
            rows = rows.max(k);
            entries.push((k - 1, d.exo, v));
        }
    }
    let mut m = Mat::zeros(rows, nx);
    for (r, c, v) in entries {
        m[(r, c)] = v;
    }
    Ok(m)
}

/// `simulate_news_shocks.m` given `exo_simul`.
pub fn simulate_news_shocks(
    model: &Model,
    dr: &HetDr,
    y_ss: &[f64],
    exo_simul: &Mat,
    max_lag: usize,
    max_lead: usize,
    shocks: &[usize],
) -> Result<Mat> {
    let t = dr.truncation_horizon;
    let periods = exo_simul.nrows() as i64 - max_lag as i64 - max_lead as i64;
    if periods <= 0 {
        bail!("Invalid simulation horizon: periods must be positive.");
    }
    let periods = periods as usize;
    if periods > t {
        bail!(
            "Simulation horizon periods={periods} exceeds Jacobian truncation horizon {t}. Reduce periods or increase truncation_horizon in heterogeneity_solve."
        );
    }
    let mut out = Mat::zeros(model.endo_names.len(), periods);
    for (iv, v) in model.endo_names.iter().enumerate() {
        let mut dev = vec![0.0; periods];
        for &si in shocks {
            let sname = &model.exo_names[si];
            let Some(g) = dr.g_block(v, sname) else {
                bail!("No Jacobian found for variable \"{v}\" and shock \"{sname}\". Run heterogeneity_solve first.");
            };
            for r in 0..periods {
                for s in 0..periods {
                    dev[r] += g[(r, s)] * exo_simul[(max_lag + s, si)];
                }
            }
        }
        for r in 0..periods {
            out[(iv, r)] = y_ss[iv] + dev[r];
        }
    }
    Ok(out)
}

/// `heterogeneity.simulate`. `det_shocks` is `M_.det_shocks` (news-shock mode when non-empty).
pub fn simulate(
    model: &Model,
    dr: &HetDr,
    y_ss: &[f64],
    sigma_e: &Mat,
    det_shocks: &[crate::modfile::DetShock],
    opts: &SimOptions,
    seed: u64,
) -> Result<SimResult> {
    let shocks = shock_indices(model, opts)?;
    let mut res = SimResult::default();
    let max_lag = usize::from(model.lead_lag_incidence.iter().any(|l| l[0] != 0));
    let max_lead = usize::from(model.lead_lag_incidence.iter().any(|l| l[2] != 0));
    if !det_shocks.is_empty() {
        let mut bad = Vec::new();
        if opts.irf.is_some_and(|v| v != 40) {
            bad.push(format!("irf={} (default: 40)", opts.irf.unwrap()));
        }
        if opts.periods.is_some_and(|v| v != 0) {
            bad.push(format!("periods={} (default: 0)", opts.periods.unwrap()));
        }
        if opts.irf_shocks.as_ref().is_some_and(|l| !l.is_empty()) {
            bad.push("irf_shocks".to_string());
        }
        if opts.relative_irf == Some(true) {
            bad.push("relative_irf=true (default: false)".to_string());
        }
        if opts.irf_plot_threshold.is_some_and(|v| v != 1e-10) {
            bad.push(format!("irf_plot_threshold={} (default: 1e-10)", opts.irf_plot_threshold.unwrap()));
        }
        if !bad.is_empty() {
            bail!(
                "News shock simulation (shocks block with periods/values) is incompatible with stochastic IRF options.\nThe following non-default options were detected:\n  - {}\nRemove these options or use a shocks block without periods/values for stochastic simulation.",
                bad.join("\n  - ")
            );
        }
        let ex = news_exo_simul(model, det_shocks, max_lag, max_lead)?;
        res.endo_simul = Some(simulate_news_shocks(model, dr, y_ss, &ex, max_lag, max_lead, &shocks)?);
        res.exo_simul = Some(ex);
    } else {
        res.irfs = simulate_irfs(model, dr, sigma_e, opts, &shocks)?;
        let periods = opts.periods.unwrap_or(0);
        if periods > 0 {
            let eps = draw_shocks(sigma_e, periods + dr.truncation_horizon - 1, seed)?;
            res.endo_simul = Some(stochastic_paths(model, dr, y_ss, &eps, periods, &shocks)?);
        }
    }
    Ok(res)
}

/// `heterogeneity.plot_irfs`: one set of figures `<fname>_IRF_<shock>[k]` per shock, reusing the
/// `stoch_simul` IRF figures (variables below `irf_plot_threshold` are dropped).
pub fn plot_irfs(
    model: &Model,
    irfs: &[(String, Vec<f64>)],
    opts: &SimOptions,
    gopts: &crate::graphics::GraphOptions,
    fname: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let mut written = Vec::new();
    if !gopts.enabled {
        return Ok(written);
    }
    let thr = opts.irf_plot_threshold.unwrap_or(1e-10);
    let relative = opts.relative_irf.unwrap_or(false);
    for sname in model.exo_names.iter() {
        let suffix = format!("_{sname}");
        let sel: Vec<(&str, &Vec<f64>)> =
            irfs.iter().filter(|(n, _)| n.ends_with(&suffix)).map(|(n, v)| (&n[..n.len() - suffix.len()], v)).collect();
        if sel.is_empty() {
            continue;
        }
        let t = sel[0].1.len();
        let y = Mat::from_fn(sel.len(), t, |i, j| sel[i].1[j]);
        let names: Vec<String> = sel.iter().map(|(n, _)| n.to_string()).collect();
        for fig in crate::analysis::irf::irf_figures(sname, &names, &y, relative, thr) {
            written.extend(crate::graphics::save(&fig, fname, gopts)?);
        }
    }
    Ok(written)
}

/// `heterogeneity.plot_simulation`: figures `<fname><suffix>[k]` (`_simulation` or
/// `_det_simulation`) of the variables of `var_list` (all endogenous variables when empty).
pub fn plot_simulation(
    model: &Model,
    data: &Mat,
    var_list: &[String],
    news: bool,
    gopts: &crate::graphics::GraphOptions,
    fname: &str,
) -> Result<Vec<std::path::PathBuf>> {
    use crate::graphics::{Color, Figure, LineStyle, Panel, Series};
    let mut written = Vec::new();
    if !gopts.enabled {
        return Ok(written);
    }
    let vars: Vec<String> = if var_list.is_empty() { model.endo_names.clone() } else { var_list.to_vec() };
    let idx: Vec<usize> = vars
        .iter()
        .map(|v| {
            model
                .endo_names
                .iter()
                .position(|e| e == v)
                .ok_or_else(|| anyhow::anyhow!("Variable \"{v}\" not found in M_.endo_names."))
        })
        .collect::<Result<_>>()?;
    let (suffix, title) = if news {
        ("_det_simulation", "Heterogeneity: Deterministic Shock Sequence")
    } else {
        ("_simulation", "Heterogeneity: Stochastic Simulation")
    };
    let (nbplt, nr, nc, lr, lc, nstar) = crate::analysis::irf::pltorg(idx.len());
    let x: Vec<f64> = (1..=data.ncols()).map(|t| t as f64).collect();
    for f in 0..nbplt {
        let sel = &idx[f * nstar..(f * nstar + nstar).min(idx.len())];
        let last = f + 1 == nbplt && nbplt > 1;
        let (rows, cols) = if last && lr > 0 { (lr, lc) } else { (nr, nc) };
        let panels = sel
            .iter()
            .map(|&i| Panel {
                title: model.endo_names[i].clone(),
                series: vec![Series::Line {
                    x: x.clone(),
                    y: data.row(i).iter().copied().collect(),
                    color: Color::Black,
                    style: LineStyle::Solid,
                    width: 1.0,
                    label: None,
                }],
                ..Default::default()
            })
            .collect();
        let (name, ttl) = if nbplt == 1 {
            (suffix.trim_start_matches('_').to_string(), title.to_string())
        } else {
            (format!("{}{}", suffix.trim_start_matches('_'), f + 1), format!("{title} figure {}", f + 1))
        };
        written.extend(crate::graphics::save(&Figure { name, title: ttl, rows, cols, panels }, fname, gopts)?);
    }
    Ok(written)
}
