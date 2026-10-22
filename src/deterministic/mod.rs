// SPDX-License-Identifier: GPL-3.0-or-later
//! Deterministic (perfect-foresight) simulations: Dynare's `oo_` state for `initval` /
//! `endval` / `histval` / `steady`, `perfect_foresight_setup` (`make_ex_.m`, `make_y_.m`)
//! and `perfect_foresight_solver` (homotopy loop of `perfect_foresight_solver.m`,
//! `perfect_foresight_solver_core.m`, Newton on the stacked system of `sim1.m`, and
//! `sim1_linear.m` for `model(linear)` / `linear_approximation`), plus `rplot`.
//!
//! The driver keeps one [`DetState`] per run and forwards `steady`, `endval` and the
//! [`Command::Deterministic`] commands to it.

pub mod sparse_lu;
pub mod stacked;
pub mod steady;
// ---- iterative / LBJ / stacked-MCP solvers, lmmcp, complementarity conditions
pub mod block_solve;
pub mod diagnostics;
pub mod exo_det;
pub mod iterative;
pub mod lmmcp;
pub mod mcp;
pub mod periods; // ---- shocks/shock_paths ----
pub mod pf_q1;
pub mod shock_paths;
pub mod spmat;
pub mod workspace; // ---- shocks/shock_paths ----

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};

use crate::graphics::{self, Color, Figure, GraphFormat, GraphOptions, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::model::Model;
use crate::modfile::{Command, DetShockKind, ModFile, OptValue, Options, opt};
use crate::solver::SteadyOptions;
pub use stacked::{StackedJacobian, StackedProblem};
pub use steady::{SteadyStateModel, evaluate_steady_state};

/// `options_` fields used by the deterministic commands (Dynare defaults). Options persist
/// across commands, as in Dynare.
#[derive(Clone, Debug)]
pub struct SimulOptions {
    /// `options_.periods`.
    pub periods: usize,
    /// `options_.simul.maxit` (50).
    pub maxit: usize,
    /// `options_.dynatol.f` / `.x` (1e-5).
    pub tolf: f64,
    pub tolx: f64,
    pub stack_solve_algo: u8,
    pub noprint: bool,
    pub verbosity: u8,
    pub no_homotopy: bool,
    pub homotopy_initial_step_size: f64,
    pub homotopy_min_step_size: f64,
    pub homotopy_step_size_increase_success_count: usize,
    pub homotopy_max_completion_share: f64,
    pub homotopy_linearization_fallback: bool,
    pub homotopy_exclude_varexo: Vec<String>,
    pub endval_steady: bool,
    pub endval_steady_nocheck: bool,
    pub linear_approximation: bool,
    pub endogenous_terminal_period: bool,
    /// `options_.simul.steady_*` (terminal steady state recomputation; defaults are the
    /// `solve_algo`/`steady.maxit`/`solve_tolf`/`solve_tolx` defaults, not later `steady` options).
    pub steady: SteadyOptions,
    /// `options_.steadystate.nocheck`.
    pub steadystate_nocheck: bool,
    /// iterative and robust linear solvers, lmmcp, check_jacobian_singularity, ...
    pub q1: pf_q1::PfQ1Options,
    // ---- shocks/shock_paths ----
    /// `options_.simul.first_simulation_period` / `last_simulation_period` (dates).
    pub first_simulation_period: Option<crate::dates::Date>,
    pub last_simulation_period: Option<crate::dates::Date>,
}

impl Default for SimulOptions {
    fn default() -> Self {
        SimulOptions {
            periods: 0,
            maxit: 50,
            tolf: 1e-5,
            tolx: 1e-5,
            stack_solve_algo: 0,
            noprint: false,
            verbosity: 1,
            no_homotopy: false,
            homotopy_initial_step_size: 1.0,
            homotopy_min_step_size: 1e-3,
            homotopy_step_size_increase_success_count: 3,
            homotopy_max_completion_share: 1.0,
            homotopy_linearization_fallback: false,
            homotopy_exclude_varexo: Vec::new(),
            endval_steady: false,
            endval_steady_nocheck: false,
            linear_approximation: false,
            endogenous_terminal_period: false,
            steady: SteadyOptions::default(),
            steadystate_nocheck: false,
            q1: Default::default(),
            first_simulation_period: None,
            last_simulation_period: None,
        }
    }
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

fn flag(o: &Options, k: &str) -> Option<bool> {
    match opt(o, k) {
        Some(OptValue::Flag) => Some(true),
        Some(OptValue::Num(v)) => Some(*v != 0.0),
        Some(OptValue::Str(s)) => Some(s != "false" && s != "0"),
        _ => None,
    }
}

const PF_OPTIONS: &[&str] = &[
    "periods",
    "maxit",
    "tolf",
    "tolx",
    "stack_solve_algo",
    "noprint",
    "print",
    "verbosity",
    "no_homotopy",
    "homotopy_initial_step_size",
    "homotopy_min_step_size",
    "homotopy_step_size_increase_success_count",
    "homotopy_max_completion_share",
    "homotopy_linearization_fallback",
    "homotopy_exclude_varexo",
    "endval_steady",
    "endval_steady_nocheck",
    "linear_approximation",
    "endogenous_terminal_period",
    "steady_solve_algo",
    "steady_maxit",
    "steady_tolf",
    "steady_tolx",
    "nograph",
    "graph_format",
    "nodisplay",
    "solve_algo",
    "markowitz",
    "steady_markowitz",
    "first_simulation_period",
    "last_simulation_period",
];

impl SimulOptions {
    /// Applies the options of one command.
    pub fn update(&mut self, o: &Options) {
        for (k, _) in o {
            if !PF_OPTIONS.contains(&k.as_str()) && !pf_q1::PF_Q1_OPTIONS.contains(&k.as_str()) {
                println!(
                    "WARNING: option `{k}` is not supported by dynare-rust's perfect foresight solver and is ignored."
                );
            }
        }
        if let Some(v) = num(o, "periods") {
            self.periods = v as usize;
        }
        // ---- shocks/shock_paths: dates of the simulation (get_simulation_periods.m)
        for (k, slot) in [
            ("first_simulation_period", &mut self.first_simulation_period),
            ("last_simulation_period", &mut self.last_simulation_period),
        ] {
            match crate::dates::option(o, k) {
                Ok(Some(crate::dates::DateOrIndex::Date(d))) => *slot = Some(d),
                Ok(None) => {}
                _ => println!("WARNING: option `{k}` must be a date; ignored."),
            }
        }
        if let Some(v) = num(o, "maxit") {
            self.maxit = v as usize;
        }
        if let Some(v) = num(o, "tolf") {
            self.tolf = v;
        }
        if let Some(v) = num(o, "tolx") {
            self.tolx = v;
        }
        if let Some(v) = num(o, "stack_solve_algo") {
            self.stack_solve_algo = v as u8; // every value 0..=10 (pf_q1::core)
            if v < 0.0 || v > 10.0 || matches!(self.stack_solve_algo, 4 | 5) {
                println!(
                    "ERROR: PERFECT_FORESIGHT_SOLVER: you must use stack_solve_algo={{0,1,2,3,6,7,8,9,10}} when not using block nor bytecode option"
                );
            }
        }
        self.q1.update(o);
        if let Some(b) = flag(o, "noprint") {
            self.noprint = b;
        }
        if flag(o, "print") == Some(true) {
            self.noprint = false;
        }
        if let Some(v) = num(o, "verbosity") {
            self.verbosity = v as u8;
        }
        if let Some(b) = flag(o, "no_homotopy") {
            self.no_homotopy = b;
        }
        if let Some(v) = num(o, "homotopy_initial_step_size") {
            self.homotopy_initial_step_size = v;
        }
        if let Some(v) = num(o, "homotopy_min_step_size") {
            self.homotopy_min_step_size = v;
        }
        if let Some(v) = num(o, "homotopy_step_size_increase_success_count") {
            self.homotopy_step_size_increase_success_count = v as usize;
        }
        if let Some(v) = num(o, "homotopy_max_completion_share") {
            self.homotopy_max_completion_share = v;
        }
        if let Some(b) = flag(o, "homotopy_linearization_fallback") {
            self.homotopy_linearization_fallback = b;
        }
        match opt(o, "homotopy_exclude_varexo") {
            Some(OptValue::List(l)) => {
                self.homotopy_exclude_varexo =
                    l.iter().filter_map(|v| if let OptValue::Str(s) = v { Some(s.clone()) } else { None }).collect()
            }
            Some(OptValue::Str(s)) => self.homotopy_exclude_varexo = vec![s.clone()],
            _ => {}
        }
        if let Some(b) = flag(o, "endval_steady") {
            self.endval_steady = b;
        }
        if let Some(b) = flag(o, "endval_steady_nocheck") {
            self.endval_steady_nocheck = b;
        }
        if let Some(b) = flag(o, "linear_approximation") {
            self.linear_approximation = b;
        }
        if let Some(b) = flag(o, "endogenous_terminal_period") {
            self.endogenous_terminal_period = b;
        }
        if let Some(v) = num(o, "steady_solve_algo") {
            self.steady.solve_algo = v as u8;
        }
        if let Some(v) = num(o, "steady_maxit") {
            self.steady.maxit = v as usize;
        }
        if let Some(v) = num(o, "steady_tolf") {
            self.steady.tolf = v;
        }
        if let Some(v) = num(o, "steady_tolx") {
            self.steady.tolx = v;
        }
    }
}

/// Outcome of `perfect_foresight_solver` (`oo_.deterministic_simulation`).
#[derive(Clone, Debug, Default)]
pub struct SimulationStatus {
    pub status: bool,
    pub error: f64,
    pub homotopy_completion_share: f64,
    pub homotopy_iterations: usize,
    pub iterations: usize,
}

/// The parts of Dynare's `oo_` (and `M_.endo_histval`) that deterministic commands use.
pub struct DetState {
    pub params: Vec<f64>,
    pub steady_state: Vec<f64>,
    pub exo_steady_state: Vec<f64>,
    pub initial_steady_state: Option<Vec<f64>>,
    pub initial_exo_steady_state: Option<Vec<f64>>,
    /// `oo_.endo_simul`: n_endo x (maximum_lag + periods + maximum_lead).
    pub endo_simul: Option<Mat>,
    /// `oo_.exo_simul`: (maximum_lag + periods + maximum_lead) x n_exo.
    pub exo_simul: Option<Mat>,
    /// Global `options_.solve_algo` / `steady.maxit` / `solve_tolf` / `solve_tolx` (set by `steady`).
    pub steady_opts: SteadyOptions,
    /// `options_.fsolve_options` (lower-cased name, value) pairs of the meaningful MATLAB `fsolve` options.
    pub fsolve_overrides: Vec<(String, f64)>,
    /// `options_.steady.non_zero`.
    pub steady_non_zero: bool,
    /// Equation names for the failure printout of `steady` (set by the driver).
    pub eq_names: Vec<Option<String>>,
    pub simul: SimulOptions,
    pub ssm: Option<SteadyStateModel>,
    pub last: SimulationStatus,
    pub maximum_lag: usize,
    pub maximum_lead: usize,
    /// Stream EP: `oo_.deterministic_simulation.controlled_paths_by_period`
    /// (`perfect_foresight_controlled_paths`), set by `setup`.
    pub controlled: Option<Vec<crate::ep::controlled::ControlledPeriod>>,
    /// Stream EP: `oo_.pfwee` (perfect foresight with expectation errors).
    pub pfwee: Option<crate::ep::pfwee::PfweeInfo>,
    /// Stream EP: `oo_.initval_series` from `initval_file` (n_endo x T endogenous incl.
    /// auxiliary variables, T x n_exo exogenous).
    pub initval_series: Option<(Mat, Mat)>,
    /// Stream EP: `M_.endo_histval` (n_endo x maximum_lag) / `M_.exo_histval` (n_exo x
    /// maximum_lag) from `histval_file`.
    pub histval_file: Option<(Mat, Mat)>,
    /// controlled paths of `shock_paths` blocks (exogenize / endogenize stanzas).
    pub q1_controlled: Vec<crate::ep::controlled::ControlledPath>,
    /// the file has `shock_paths` blocks.
    pub q1_shock_paths: bool,
}

/// Does the file need the deterministic machinery for `steady` (endval, exogenous initval,
/// steady_state_model or deterministic commands)?
pub fn uses_det_state(mf: &ModFile) -> bool {
    !mf.steady_state_model.is_empty()
        || mf.initval_exo.iter().any(|&(_, v)| v != 0.0)
        || mf.commands.iter().any(|c| matches!(c, Command::Endval(_) | Command::Deterministic(..)))
        // ---- stream EP: extended_path reads oo_.steady_state / M_.endo_histval from DetState
        || mf.commands.iter().any(|c| matches!(c, Command::Other(n, _) if crate::ep::handles(n)))
        // ---- complementarity conditions (steady with solve_algo = 10)
        || !mf.model.mcp.is_empty()
        || !mf.q1.shock_paths.is_empty()
}

/// MATLAB `%g` (6 significant digits).
pub fn fmt_g(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Inf".into() } else { "-Inf".into() };
    }
    if v == 0.0 {
        return "0".into();
    }
    let s = format!("{:.5e}", v);
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    if !(-4..6).contains(&e) {
        let mant = trim_zeros(mant);
        format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        let dec = (5 - e).max(0) as usize;
        trim_zeros(&format!("{:.*}", dec, v)).to_string()
    }
}

/// C `%e` (6 decimals, two-digit exponent).
pub fn fmt_e(v: f64) -> String {
    if !v.is_finite() {
        return fmt_g(v);
    }
    let s = format!("{:.6e}", v);
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

fn trim_zeros(s: &str) -> &str {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { s }
}

fn inf_norm(v: &[f64]) -> f64 {
    // MATLAB norm(v, Inf): NaN if any NaN
    let mut m = 0.0f64;
    for &x in v {
        if x.is_nan() {
            return f64::NAN;
        }
        m = m.max(x.abs());
    }
    m
}

impl DetState {
    /// State after the global `initval` block (`oo_.steady_state` / `oo_.exo_steady_state`
    /// with the aux variables set, as the generated driver does).
    pub fn new(mf: &ModFile) -> Result<DetState> {
        let m = &mf.model;
        let n = m.n_endo();
        let mut ys = vec![0.0; n];
        for &(i, v) in &mf.initval {
            ys[i] = v;
        }
        let mut xs = vec![0.0; m.n_exo()];
        for &(k, v) in &mf.initval_exo {
            xs[k] = v;
        }
        m.set_auxiliary_variables(&mut ys, &xs, &m.params);
        let lli = &m.lead_lag_incidence;
        let maximum_lag = lli.iter().any(|r| r[0] > 0) as usize;
        let maximum_lead = lli.iter().any(|r| r[2] > 0) as usize;
        Ok(DetState {
            params: m.params.clone(),
            steady_state: ys,
            exo_steady_state: xs,
            initial_steady_state: None,
            initial_exo_steady_state: None,
            endo_simul: None,
            exo_simul: None,
            steady_opts: SteadyOptions { linear: m.linear, ..Default::default() },
            fsolve_overrides: Vec::new(),
            steady_non_zero: false,
            eq_names: Vec::new(),
            simul: SimulOptions::default(),
            ssm: SteadyStateModel::from_modfile(mf)?,
            last: SimulationStatus::default(),
            maximum_lag,
            maximum_lead,
            pfwee: None,
            controlled: None,
            initval_series: None,
            histval_file: None,
            q1_controlled: Vec::new(),
            q1_shock_paths: !mf.q1.shock_paths.is_empty(),
        })
    }

    /// The `steady` command (`steady.m`): solves from `oo_.steady_state` and prints it.
    pub fn steady(&mut self, model: &Model, o: &Options) -> Result<()> {
        if let Some(v) = num(o, "solve_algo") {
            self.steady_opts.solve_algo = v as u8;
        }
        if let Some(v) = num(o, "maxit") {
            self.steady_opts.maxit = v as usize;
        }
        if let Some(v) = num(o, "tolf") {
            self.steady_opts.tolf = v;
        }
        if let Some(v) = num(o, "tolx") {
            self.steady_opts.tolx = v;
        }
        // fsolve_options (options_.fsolve_options, persistent): applied last in dynare_solve.m, so
        // they override tolf/tolx/maxit of the MATLAB `fsolve` that solve_algo=0 uses. Here
        // solve_algo=0 is the trust-region solver of solve_algo=9: TolFun/FunctionTolerance,
        // TolX/StepTolerance and MaxIter/MaxIterations map onto it; Display (only 'off'/'final'/'iter'
        // output of fsolve) and MaxFunEvals/MaxFunctionEvaluations (50000, never binding here) have
        // no counterpart and are accepted.
        if let Some(OptValue::List(l)) = opt(o, "fsolve_options") {
            self.fsolve_overrides.clear();
            for kv in l.chunks(2) {
                if let [OptValue::Str(k), OptValue::Num(v)] = kv {
                    match k.to_ascii_lowercase().as_str() {
                        "tolfun" | "functiontolerance" | "tolx" | "steptolerance" | "maxiter" | "maxiterations" => {
                            self.fsolve_overrides.push((k.to_ascii_lowercase(), *v))
                        }
                        _ => {}
                    }
                }
            }
        }
        if flag(o, "non_zero").unwrap_or(false) {
            self.steady_non_zero = true; // options_.steady.non_zero persists
        }
        let mut sopts = self.steady_opts;
        if sopts.solve_algo == 0 {
            for (k, v) in &self.fsolve_overrides {
                match k.as_str() {
                    "tolfun" | "functiontolerance" => sopts.tolf = *v,
                    "tolx" | "steptolerance" => sopts.tolx = *v,
                    _ => sopts.maxit = *v as usize,
                }
            }
        }
        let nocheck = flag(o, "nocheck").unwrap_or(false);
        if nocheck {
            self.simul.steadystate_nocheck = true;
        }
        let noprint = flag(o, "noprint").unwrap_or(false);
        match evaluate_steady_state(
            model,
            self.ssm.as_ref(),
            &self.steady_state,
            &self.exo_steady_state,
            &self.params,
            &sopts,
            !self.simul.steadystate_nocheck,
        ) {
            Ok((ys, p)) => {
                self.steady_state = ys;
                self.params = p;
                if !noprint {
                    print_steady_state(model, &self.steady_state);
                }
                Ok(())
            }
            Err(e) => {
                if !noprint {
                    // steady.m: display_static_residuals (with options_.steady.non_zero)
                    let r = steady::static_residuals(model, &self.steady_state, &self.exo_steady_state, &self.params);
                    print!(
                        "{}",
                        crate::misc::residuals_text(model, &r, &self.eq_names, sopts.tolf, self.steady_non_zero)
                    );
                }
                bail!("steady: {e}")
            }
        }
    }

    /// The `endval` block at its position: `oo_.initial_steady_state` <- `oo_.steady_state`,
    /// then the new values (expressions read the current values, as the generated driver).
    pub fn endval(&mut self, model: &Model, entries: &[(String, String)]) -> Result<()> {
        self.initial_steady_state = Some(self.steady_state.clone());
        self.initial_exo_steady_state = Some(self.exo_steady_state.clone());
        for (name, src) in entries {
            let e = model.compile_expr(src).map_err(|e| anyhow!("endval: {name} = {src}: {e}"))?;
            let v = e.eval(&self.steady_state, &self.exo_steady_state, &self.params);
            if let Some(j) = model.endo_index(name) {
                self.steady_state[j] = v;
            } else if let Some(k) = model.exo_index(name) {
                self.exo_steady_state[k] = v;
            } else {
                bail!("endval: '{name}' is not a variable");
            }
        }
        model.set_auxiliary_variables(&mut self.steady_state, &self.exo_steady_state, &self.params);
        Ok(())
    }

    /// Runs one deterministic command.
    pub fn command(
        &mut self,
        mf: &ModFile,
        name: &str,
        o: &Options,
        vars: &[String],
        dir: &Path,
        fname: &str,
    ) -> Result<()> {
        let model = &mf.model;
        match name {
            "perfect_foresight_setup" => {
                self.simul.update(o);
                self.setup(mf)
            }
            "perfect_foresight_solver" => {
                self.simul.update(o);
                self.solve(model)?;
                self.write_results(model, dir, fname)
            }
            "simul" => {
                self.simul.update(o);
                self.setup(mf)?;
                self.solve(model)?;
                self.write_results(model, dir, fname)
            }
            "rplot" => self.rplot(model, o, vars, dir, fname),
            // ---- the position of a shock_paths block: its exogenize stanzas become
            // M_.perfect_foresight_controlled_paths entries, evaluated here
            "shock_paths" => {
                let k = match o.iter().find(|(k, _)| k == "__index") {
                    Some((_, OptValue::Num(v))) => *v as usize,
                    _ => 0,
                };
                let b = &mf.q1.shock_paths[k];
                if b.overwrite {
                    // ---- shocks/shock_paths: a date never matches an integer learnt_in
                    self.q1_controlled.retain(|c| match (c.learnt_in_date, b.learnt_in_date) {
                        (Some(a), Some(d)) => a != d,
                        (None, None) => c.learnt_in != b.learnt_in,
                        _ => true,
                    });
                }
                let cps =
                    shock_paths::controlled_paths(mf, b, &self.params, &self.steady_state, &self.exo_steady_state)?;
                self.q1_controlled.extend(cps);
                Ok(())
            }
            "model_diagnostics" => {
                // full port of model_diagnostics.m (options_.order = 2 by default)
                let order = match o.iter().find(|(k, _)| k == "order") {
                    Some((_, OptValue::Num(v))) => *v as usize,
                    _ => 2,
                };
                diagnostics::model_diagnostics(
                    model,
                    self.ssm.as_ref(),
                    &self.steady_state,
                    &self.exo_steady_state,
                    &self.params,
                    &self.steady_opts,
                    self.simul.steadystate_nocheck,
                    order,
                );
                Ok(())
            }
            _ => bail!("unknown deterministic command {name}"),
        }
    }

    // ---- shocks/shock_paths ----
    /// `get_simulation_periods(options_)`: (periods, first_simulation_period,
    /// last_simulation_period); `options_.periods` is set from the dates when it is 0.
    pub fn simulation_periods(&mut self) -> Result<(usize, Option<crate::dates::Date>, Option<crate::dates::Date>)> {
        let s = &self.simul;
        if s.periods == 0 && (s.first_simulation_period.is_none() || s.last_simulation_period.is_none()) {
            bail!(
                "perfect_foresight_setup: the `periods` option must be set (or both first_simulation_period and last_simulation_period)"
            );
        }
        let r = periods::get_simulation_periods(s.periods, s.first_simulation_period, s.last_simulation_period)?;
        self.simul.periods = r.0;
        Ok(r)
    }

    fn ncols(&self) -> usize {
        self.maximum_lag + self.simul.periods + self.maximum_lead
    }

    /// Port of `perfect_foresight_setup.m` (`make_ex_.m`, `make_y_.m`).
    pub fn setup(&mut self, mf: &ModFile) -> Result<()> {
        let m = &mf.model;
        let (n, nx) = (m.n_endo(), m.n_exo());
        if let Some(k) = self.params.iter().position(|p| p.is_nan()) {
            println!(
                "WARNING: Some of the parameters have no value ({}) when using perfect_foresight_setup.",
                m.param_names[k]
            );
        }
        let missing: Vec<&str> =
            (0..n).filter(|&j| m.lead_lag_incidence[j][1] == 0).map(|j| m.endo_names[j].as_str()).collect();
        if !missing.is_empty() {
            bail!(
                "PERFECT_FORESIGHT_SETUP: error in model specification : the variable(s) {} don't appear as current period variables.",
                missing.join(", ")
            );
        }
        // ---- shocks/shock_paths: get_simulation_periods.m (dates of the simulation)
        let (periods, fsp, lsp) = self.simulation_periods()?;
        let mut bad = String::new();
        for s in &mf.det_shocks {
            if !s.dates.is_empty() && lsp.is_none() {
                bail!(
                    "PERFECT_FORESIGHT_SETUP: temporary shocks are specified using dates but neither first_simulation_period nor last_simulation_period option was passed"
                );
            }
            if s.sim_periods(fsp)?.iter().any(|&p| p > periods as i64) {
                bad.push_str(&format!(
                    "\n   At least one expected value for {} has been declared after the terminal period.",
                    m.exo_names[s.exo]
                ));
            }
        }
        if !bad.is_empty() {
            println!("\nPERFECT_FORESIGHT_SETUP: Problem with the declaration of the expected shocks:\n{bad}\n");
            bail!(
                "PERFECT_FORESIGHT_SETUP: Please check the declaration of the shocks or increase the number of periods in the simulation."
            );
        }
        if self.simul.endval_steady && self.maximum_lead == 0 {
            bail!("PERFECT_FORESIGHT_SETUP: Option endval_steady cannot be used on a purely backward or static model.");
        }
        shock_paths::check_setup(mf, lsp)?; // ---- shocks/shock_paths
        let (ml, ncols) = (self.maximum_lag, self.ncols());
        let (endo_hist, exo_hist) = if let Some((e, x)) = &self.histval_file {
            (Some(e.clone()), Some(x.clone())) // ---- stream EP: histval_file
        } else if mf.histval.is_empty() {
            (None, None)
        } else {
            Some(self.histval(mf)).unzip()
        };
        // ---- a permanent shock in shock_paths recreates endval's variables
        if mf.q1.shock_paths.iter().any(|b| b.contains_endval()) {
            self.initial_steady_state = Some(self.steady_state.clone());
            self.initial_exo_steady_state = Some(self.exo_steady_state.clone());
        }
        // make_ex_
        let mut ex = Mat::zeros(ncols, nx);
        for r in 0..ncols {
            for k in 0..nx {
                ex[(r, k)] = match (&exo_hist, &self.initial_exo_steady_state) {
                    (Some(h), _) if r < ml => h[(k, r)],
                    (None, Some(ix)) if r < ml => ix[k],
                    _ => self.exo_steady_state[k],
                };
            }
        }
        if exo_hist.is_some() && self.initial_exo_steady_state.is_some() && self.initval_series.is_none() {
            bail!("histval and endval cannot be used simultaneously");
        }
        // ---- stream EP: initval_file (make_ex_.m with oo_.initval_series)
        let series_off = m.orig_maximum_lag.saturating_sub(ml);
        if let Some((_, sx)) = &self.initval_series {
            if sx.nrows() < series_off + ncols {
                bail!(
                    "perfect_foresight_setup: the initval_file data has {} observations, {} are needed",
                    sx.nrows(),
                    series_off + ncols
                );
            }
            ex = sx.rows(series_off, ncols).clone_owned();
            if let Some(h) = &exo_hist {
                for r in 0..ml {
                    for k in 0..nx {
                        ex[(r, k)] = h[(k, r)];
                    }
                }
            }
        }
        for s in &mf.det_shocks {
            for p in s.sim_periods(fsp)? {
                let k = p + ml as i64; // 1-based row
                if k == 0 {
                    bail!("make_ex_: The model does not have lags, so you cannot set values for period 0");
                }
                if !s.dates.is_empty() && k <= 1 {
                    bail!("make_ex_: some shocks are set at a date before the first simulation period");
                }
                if k < 0 {
                    bail!("make_ex_: a shock is set before the initial period");
                }
                let r = (k - 1) as usize;
                ex[(r, s.exo)] = match s.kind {
                    DetShockKind::Level => s.value,
                    DetShockKind::MultiplySteadyState => self.exo_steady_state[s.exo] * s.value,
                    DetShockKind::MultiplyInitialSteadyState => {
                        let ix = self.initial_exo_steady_state.as_ref().ok_or_else(|| {
                            anyhow!(
                                "Option relative_to_initval of mshocks block cannot be used without an endval block"
                            )
                        })?;
                        ix[s.exo] * s.value
                    }
                };
            }
        }
        // ---- shock_paths (make_ex_.m)
        if !mf.q1.shock_paths.is_empty() {
            let endval = mf.q1.shock_paths.iter().any(|b| b.contains_endval());
            let np = periods + endval as usize;
            let mut paths = Mat::from_fn(nx, np, |k, c| ex[(ml + c, k)]);
            shock_paths::evaluate_paths(
                mf,
                &self.params,
                &self.steady_state,
                &self.exo_steady_state,
                periods,
                fsp,
                &mut paths,
            )?;
            for c in 0..periods {
                for k in 0..nx {
                    ex[(ml + c, k)] = paths[(k, c)];
                }
            }
            if endval {
                self.exo_steady_state = paths.column(periods).iter().copied().collect();
                for r in ml + periods..ncols {
                    for k in 0..nx {
                        ex[(r, k)] = self.exo_steady_state[k];
                    }
                }
            }
        }
        // make_y_
        let mut ey = Mat::zeros(n, ncols);
        for c in 0..ncols {
            for j in 0..n {
                ey[(j, c)] = match (&endo_hist, &self.initial_steady_state) {
                    (Some(h), _) if c < ml => h[(j, c)],
                    (None, Some(is)) if c < ml => is[j],
                    _ => self.steady_state[j],
                };
            }
        }
        // ---- stream EP: initval_file (make_y_.m with oo_.initval_series)
        if let Some((se, _)) = &self.initval_series {
            ey = se.columns(series_off, ncols).clone_owned();
            if let Some(h) = &endo_hist {
                if self.initial_steady_state.is_some() {
                    bail!("histval and endval cannot be used simultaneously");
                }
                for c in 0..ml {
                    for j in 0..n {
                        ey[(j, c)] = h[(j, c)];
                    }
                }
            }
        }
        self.exo_simul = Some(ex);
        self.endo_simul = Some(ey);
        // ---- stream EP: perfect_foresight_controlled_paths
        self.controlled = crate::ep::controlled::by_period_with(mf, &self.q1_controlled, periods, None, fsp)?;
        Ok(())
    }

    /// `M_.endo_histval` / `M_.exo_histval` as the generated histval code builds them: a
    /// zero-initialized series over `orig_maximum_lag` periods, the histval entries, aux lag
    /// variables from their definitions (`dynamic_set_auxiliary_series`), NaN -> 0, last
    /// `maximum_lag` periods.
    pub fn histval(&self, mf: &ModFile) -> (Mat, Mat) {
        let m = &mf.model;
        let (n, nx) = (m.n_endo(), m.n_exo());
        let ll = m.orig_maximum_lag.max(1) as i32;
        // series index: d in -(ll-1)..=0 -> d + ll - 1
        let mut endo = vec![vec![0.0; ll as usize]; m.orig_endo_nbr];
        let mut exo = vec![vec![0.0; ll as usize]; nx];
        let mut direct = Vec::new(); // auxiliary rows set directly (smoother2histval)
        for &(is_endo, idx, lag, v) in &mf.histval {
            let d = lag + ll - 1;
            if is_endo && idx >= m.orig_endo_nbr {
                direct.push((idx, lag, v));
            } else if d >= 0 {
                if is_endo {
                    endo[idx][d as usize] = v;
                } else {
                    exo[idx][d as usize] = v;
                }
            }
        }
        let ml = self.maximum_lag;
        let mut eh = Mat::zeros(n, ml);
        let mut xh = Mat::zeros(nx, ml);
        for c in 0..ml {
            let d = c as i32 - ml as i32 + 1; // date 1-ml .. 0
            let at = |s: &Vec<f64>, d: i32| -> f64 {
                let i = d + ll - 1;
                if i >= 0 && i < ll { s[i as usize] } else { 0.0 }
            };
            for j in 0..m.orig_endo_nbr {
                eh[(j, c)] = at(&endo[j], d);
            }
            for k in 0..nx {
                xh[(k, c)] = at(&exo[k], d);
            }
            for a in &m.aux_vars {
                let v = match (a.ty, a.orig_index, a.orig_lead_lag) {
                    (1, Some(o), Some(l)) => at(&endo[o], d + l),
                    (3, Some(o), Some(l)) => at(&exo[o], d + l),
                    _ => 0.0,
                };
                eh[(a.endo_index, c)] = v;
            }
        }
        for (idx, lag, v) in direct {
            let c = ml as i32 - 1 + lag;
            if idx < n && c >= 0 && (c as usize) < ml {
                eh[(idx, c as usize)] = v;
            }
        }
        (eh, xh)
    }

    /// Port of `perfect_foresight_solver.m` (homotopy loop, linearization fallback).
    pub fn solve(&mut self, model: &Model) -> Result<()> {
        let o = self.simul.clone();
        let (endo0, exo0) = match (&self.endo_simul, &self.exo_simul) {
            (Some(e), Some(x)) if e.ncols() == self.ncols() => (e.clone(), x.clone()),
            _ => bail!(
                "PERFECT_FORESIGHT_SOLVER: 'oo_.endo_simul' has wrong size. Did you run 'perfect_foresight_setup' ?"
            ),
        };
        if o.homotopy_initial_step_size > 1.0 || o.homotopy_initial_step_size < 0.0 {
            bail!(
                "perfect_foresight_solver: The value given to homotopy_initial_step_size option must be in the [0,1] interval"
            );
        }
        if model.block_solve.as_ref().is_some_and(|b| b.dynamic.is_some()) {
            // perfect_foresight_solver_core.m, options_.block
            if o.linear_approximation {
                bail!("perfect_foresight_solver: option linear_approximation is not available with the block option");
            }
            if o.endogenous_terminal_period {
                bail!(
                    "perfect_foresight_solver: option endogenous_terminal_period is not available with the block option"
                );
            }
            if self.controlled.is_some() {
                bail!(
                    "perfect_foresight_solver: perfect_foresight_controlled_paths or shock_paths+exogenize is not available with the block option"
                );
            }
        }
        // ---- homotopy_marginal_linearization_fallback
        let mlf = o.q1.homotopy_marginal_linearization_fallback;
        if o.no_homotopy
            && (o.homotopy_initial_step_size != 1.0
                || o.homotopy_max_completion_share != 1.0
                || o.homotopy_linearization_fallback
                || mlf != 0.0)
        {
            bail!(
                "perfect_foresight_solver: the no_homotopy option is incompatible with homotopy_initial_step_size, homotopy_max_completion_share, homotopy_linearization_fallback and homotopy_marginal_linearization_fallback options"
            );
        }
        if !(0.0..=1.0).contains(&mlf) {
            bail!(
                "perfect_foresight_solver: The value given to homotopy_marginal_linearization_fallback option must be in the [0,1] interval"
            );
        }
        if o.homotopy_linearization_fallback && mlf > 0.0 {
            bail!(
                "perfect_foresight_solver: Options homotopy_linearization_fallback and homotopy_marginal_linearization_fallback cannot be used together"
            );
        }
        if o.homotopy_max_completion_share < 1.0 && !o.homotopy_linearization_fallback && mlf == 0.0 {
            bail!(
                "perfect_foresight_solver: Option homotopy_max_completion_share has a value less than 1, so you must also specify either homotopy_linearization_fallback or homotopy_marginal_linearization_fallback"
            );
        }
        let (ml, ncols) = (self.maximum_lag, self.ncols());
        let periods = o.periods;
        let n = model.n_endo();
        let base_ss = self.initial_steady_state.clone().unwrap_or_else(|| self.steady_state.clone());
        let base_xs = self.initial_exo_steady_state.clone().unwrap_or_else(|| self.exo_steady_state.clone());
        let endobase = Mat::from_fn(n, ncols, |j, _| base_ss[j]);
        let exobase = Mat::from_fn(ncols, model.n_exo(), |_, k| base_xs[k]);
        let lastperiods: Vec<usize> = (ml + periods..ncols).collect();
        let recompute = if o.endval_steady || self.q1_shock_paths {
            true // `|| ~isempty(M_.shock_paths)`
        } else if self.initial_steady_state.is_some() {
            lastperiods.iter().all(|&j| {
                let y: Vec<f64> = endo0.column(j).iter().copied().collect();
                let x: Vec<f64> = exo0.row(j).iter().copied().collect();
                inf_norm(&steady::static_residuals(model, &y, &x, &self.params)) <= o.steady.tolf
            })
        } else {
            false
        };
        let h = self.homotopy_loop(model, &o, &endo0, &exo0, &endobase, &exobase, recompute)?;
        self.last = SimulationStatus {
            status: false,
            error: h.maxerror,
            homotopy_completion_share: h.completed_share,
            homotopy_iterations: h.iteration,
            iterations: h.solver_iter,
        };
        if h.completed_share == 1.0 {
            if self.controlled.is_some() {
                self.exo_simul = Some(h.exo.clone()); // ---- stream EP: endogenized exogenous values
            }
            self.endo_simul = Some(h.endo);
            if recompute {
                self.steady_state = h.steady_state;
            }
            if !o.noprint {
                print!("Perfect foresight solution found.\n\n");
            }
            self.last.status = true;
        } else if o.homotopy_linearization_fallback && h.completed_share > 0.0 {
            let s = h.completed_share;
            let e = Mat::from_fn(n, ncols, |j, c| endobase[(j, c)] + (h.endo[(j, c)] - endobase[(j, c)]) / s);
            if recompute {
                self.steady_state = e.column(ncols - 1).iter().copied().collect();
            }
            // ---- stream EP: with controlled paths the exogenous path is extrapolated too
            let x = if self.controlled.is_some() {
                let x = Mat::from_fn(exo0.nrows(), exo0.ncols(), |r, k| {
                    exobase[(r, k)] + (h.exo[(r, k)] - exobase[(r, k)]) / s
                });
                self.exo_simul = Some(x.clone());
                x
            } else {
                exo0.clone()
            };
            let pe = model.params_with_steady_state(&self.params, &self.steady_state); // stream PP
            self.last.error = compute_maxerror(model, &e, &x, &pe, ml, periods);
            self.endo_simul = Some(e);
            if !o.noprint {
                print!(
                    "Perfect foresight solution found for {:.1}% of the shock, then extrapolation was performed using linearization\n\n",
                    s * 100.0
                );
            }
            self.last.status = true;
        } else if mlf > 0.0 && h.completed_share > mlf {
            // ---- homotopy_marginal_linearization_fallback (perfect_foresight_solver.m)
            let completed = h.completed_share;
            let extra_share = completed - mlf;
            if !o.noprint {
                print!(
                    "Only {:.1}% of the shock could be simulated. Since marginal linearization was requested as a fallback,\nnow running an extra simulation for {:.1}% of the shock.\n\n",
                    completed * 100.0,
                    extra_share * 100.0
                );
                print!("{}\n\n", "*".repeat(80));
            }
            let t0 = Instant::now();
            let excluded: Vec<usize> = o.homotopy_exclude_varexo.iter().filter_map(|s| model.exo_index(s)).collect();
            let sc = self.create_scenario(
                model,
                &o,
                extra_share,
                &endo0,
                &exo0,
                &endobase,
                &exobase,
                recompute,
                &h.endo,
                &h.exo,
                &excluded,
            );
            let (mut ok, mut extra) = (false, None);
            if sc.steady_success {
                let (ss, xs) =
                    sc.steady.clone().unwrap_or_else(|| (h.steady_state.clone(), h.exo_steady_state.clone()));
                let (mut e, mut x) = (sc.endo, sc.exo);
                let r = self.core(model, &o, &mut e, &mut x, &ss, &xs, sc.cp_scaled.as_deref());
                ok = r.0;
                extra = Some((e, x));
            }
            if !ok {
                if !o.noprint {
                    print!(
                        "The extra simulation for {:.1}% of the shock did not run when using the first simulation as a guess value.\nNow trying a full homotopy loop to get that extra simulation working.\n\n",
                        extra_share * 100.0
                    );
                    print!("{}\n\n", "*".repeat(80));
                }
                let mut o2 = o.clone();
                o2.homotopy_max_completion_share = extra_share;
                let h2 = self.homotopy_loop(model, &o2, &endo0, &exo0, &endobase, &exobase, recompute)?;
                ok = h2.completed_share == extra_share;
                extra = Some((h2.endo, h2.exo));
            }
            let el = t0.elapsed().as_secs_f64();
            if ok {
                let (ee, ex) = extra.unwrap();
                let f = (1.0 - completed) / mlf;
                let e = Mat::from_fn(n, ncols, |j, c| h.endo[(j, c)] + (h.endo[(j, c)] - ee[(j, c)]) * f);
                if recompute {
                    self.steady_state = e.column(ncols - 1).iter().copied().collect();
                }
                let x = if self.controlled.is_some() {
                    let x = Mat::from_fn(exo0.nrows(), exo0.ncols(), |r, k| {
                        h.exo[(r, k)] + (h.exo[(r, k)] - ex[(r, k)]) * f
                    });
                    self.exo_simul = Some(x.clone());
                    x
                } else {
                    exo0.clone()
                };
                let pe = model.params_with_steady_state(&self.params, &self.steady_state);
                self.last.error = compute_maxerror(model, &e, &x, &pe, ml, periods);
                self.endo_simul = Some(e);
                if !o.noprint {
                    print!(
                        "Perfect foresight solution found for {:.1}% of the shock, then extrapolation\nwas performed using marginal linearization (extra simulation took {:.1} seconds).\n\n",
                        completed * 100.0,
                        el
                    );
                }
            } else {
                self.endo_simul = Some(h.endo);
                self.exo_simul = Some(h.exo);
                self.steady_state = h.steady_state;
                self.exo_steady_state = h.exo_steady_state;
                print!(
                    "perfect_foresight_solver: marginal linearization failed, unable to\nfind solution for {:.1}% of the shock (extra simulation took {:.1} seconds).\nTry to modify the value of homotopy_marginal_linearization_fallback option.\n\n",
                    extra_share * 100.0,
                    el
                );
            }
            self.last.status = ok;
        } else {
            self.endo_simul = Some(h.endo);
            self.exo_simul = Some(h.exo);
            self.steady_state = h.steady_state;
            self.exo_steady_state = h.exo_steady_state;
            print!("Failed to solve perfect foresight model\n\n");
        }
        Ok(())
    }

    /// `create_scenario` of `perfect_foresight_solver.m` (`shareorig = 1`): exogenous paths and
    /// initial conditions at `share` of the shock, terminal steady state recomputed (or
    /// interpolated), controlled paths scaled; `endo`/`exo` are the guesses.
    #[allow(clippy::too_many_arguments)]
    fn create_scenario(
        &self,
        model: &Model,
        o: &SimulOptions,
        new_share: f64,
        endoorig: &Mat,
        exoorig: &Mat,
        endobase: &Mat,
        exobase: &Mat,
        recompute: bool,
        endo: &Mat,
        exo: &Mat,
        excluded: &[usize],
    ) -> Scenario {
        let ml = self.maximum_lag;
        let periods = o.periods;
        let ncols = self.ncols();
        let mut e2 = endo.clone();
        let mut x2 = Mat::from_fn(ncols, model.n_exo(), |r, k| {
            exoorig[(r, k)] * new_share + exobase[(r, k)] * (1.0 - new_share)
        });
        for &k in excluded {
            for r in 0..ncols {
                x2[(r, k)] = exoorig[(r, k)];
            }
        }
        for c in 0..ml {
            for j in 0..model.n_endo() {
                e2[(j, c)] = new_share * endoorig[(j, c)] + (1.0 - new_share) * endobase[(j, c)];
            }
        }
        let mut steady_success = true;
        let mut steady = None;
        if recompute {
            let mut so = o.steady;
            so.linear = model.linear;
            let saved_ss: Vec<Vec<f64>> =
                (ml + periods..ncols).map(|j| e2.column(j).iter().copied().collect()).collect();
            for j in ml + periods..ncols {
                let guess: Vec<f64> = e2.column(j).iter().copied().collect();
                let xj: Vec<f64> = x2.row(j).iter().copied().collect();
                let check = !o.endval_steady_nocheck;
                let r = evaluate_steady_state(model, self.ssm.as_ref(), &guess, &xj, &self.params, &so, check).or_else(
                    |_| {
                        let g = self.initial_steady_state.as_ref().unwrap_or(&self.steady_state);
                        evaluate_steady_state(model, self.ssm.as_ref(), g, &xj, &self.params, &so, check)
                    },
                );
                match r {
                    Ok((y, _)) => e2.set_column(j, &crate::linalg::Vector::from_vec(y)),
                    Err(_) => {
                        for (k, j2) in (ml + periods..ncols).enumerate() {
                            e2.set_column(j2, &crate::linalg::Vector::from_vec(saved_ss[k].clone()));
                        }
                        steady_success = false;
                        break;
                    }
                }
            }
            if steady_success {
                steady =
                    Some((e2.column(ncols - 1).iter().copied().collect(), x2.row(ncols - 1).iter().copied().collect()));
            }
        } else {
            for c in ml + periods..ncols {
                for j in 0..model.n_endo() {
                    e2[(j, c)] = new_share * endoorig[(j, c)] + (1.0 - new_share) * endobase[(j, c)];
                }
            }
        }
        // ---- stream EP: controlled paths in create_scenario (values scaled by the share,
        // endogenized exogenous guesses kept from the current iterate)
        let cp_scaled =
            self.controlled.as_ref().map(|cp| crate::ep::controlled::scale(cp, new_share, endobase, exo, &mut x2, ml));
        Scenario { steady_success, endo: e2, exo: x2, steady, cp_scaled }
    }

    #[allow(clippy::too_many_arguments)]
    fn homotopy_loop(
        &self,
        model: &Model,
        o: &SimulOptions,
        endoorig: &Mat,
        exoorig: &Mat,
        endobase: &Mat,
        exobase: &Mat,
        recompute: bool,
    ) -> Result<Homotopy> {
        let ml = self.maximum_lag;
        let periods = o.periods;
        let max_share = o.homotopy_max_completion_share;
        let excluded: Vec<usize> = o
            .homotopy_exclude_varexo
            .iter()
            .map(|s| {
                model.exo_index(s).ok_or_else(|| {
                    anyhow!("perfect_foresight_solver: the variable list homotopy_exclude_varexo must only contain exogenous variable names.")
                })
            })
            .collect::<Result<_>>()?;
        let mut completed = 0.0;
        let mut step = o.homotopy_initial_step_size.min(max_share);
        let mut success_counter = 0;
        let mut iteration = 0;
        let mut endo = endoorig.clone();
        let mut exo = exoorig.clone();
        let mut ss = self.steady_state.clone();
        let mut xs = self.exo_steady_state.clone();
        let mut maxerror = f64::NAN;
        let mut solver_iter = 0;
        let mut table: Vec<String> = Vec::new();
        while step > o.homotopy_min_step_size {
            iteration += 1;
            let saved = (endo.clone(), exo.clone());
            let mut new_share = completed + step;
            if new_share > max_share {
                new_share = max_share;
                step = new_share - completed;
            }
            let t0 = Instant::now();
            let sc = self.create_scenario(
                model, o, new_share, endoorig, exoorig, endobase, exobase, recompute, &endo, &exo, &excluded,
            );
            let steady_success = sc.steady_success;
            if let Some((s2, x2s)) = sc.steady {
                ss = s2;
                xs = x2s;
            }
            let cp_scaled = sc.cp_scaled;
            endo = sc.endo;
            exo = sc.exo;
            let mut success = false;
            if steady_success {
                if completed == 0.0 {
                    if iteration == 1 && new_share == 1.0 {
                        // initial guess from perfect_foresight_setup
                    } else if self.maximum_lead > 0 {
                        let last = endo.column(ml + periods).clone_owned();
                        for c in ml..ml + periods {
                            endo.set_column(c, &last);
                        }
                    } else {
                        for c in ml..ml + periods {
                            endo.set_column(c, &endobase.column(c).clone_owned());
                        }
                    }
                    // ---- stream EP: with controlled paths, the exogenous guesses over the
                    // simulation periods restart from the base path (all of them, as Dynare)
                    if self.controlled.is_some() && !(iteration == 1 && new_share == 1.0) {
                        for r in ml..ml + periods {
                            for k in 0..model.n_exo() {
                                exo[(r, k)] = exobase[(r, k)];
                            }
                        }
                    }
                }
                let r = self.core(model, o, &mut endo, &mut exo, &ss, &xs, cp_scaled.as_deref());
                success = r.0;
                maxerror = r.1;
                solver_iter = r.2;
            } else {
                maxerror = f64::NAN;
            }
            let elapsed = t0.elapsed().as_secs_f64();
            if o.no_homotopy || (iteration == 1 && success && new_share == 1.0) {
                if success {
                    completed = new_share;
                }
                break;
            }
            if iteration == 1 && !o.noprint {
                println!("\nEntering the homotopy method iterations...");
                table.push("\nIter. \t | Share \t | Status \t | Max. residual\t | Duration (sec)\n".into());
                table
                    .push("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n".into());
            }
            if success {
                if !o.noprint {
                    table.push(format!(
                        "{} \t | {:.5} \t | succeeded \t | {} \t | {:.1}\n",
                        iteration,
                        new_share,
                        fmt_e(maxerror),
                        elapsed
                    ));
                }
                completed = new_share;
                if completed >= max_share {
                    print!("{}", table.concat());
                    break;
                }
                success_counter += 1;
                if o.homotopy_step_size_increase_success_count > 0
                    && success_counter >= o.homotopy_step_size_increase_success_count
                {
                    success_counter = 0;
                    step *= 2.0;
                }
            } else {
                endo = saved.0;
                exo = saved.1;
                success_counter = 0;
                step /= 2.0;
                if !o.noprint {
                    if !steady_success {
                        table.push(format!(
                            "{} \t | {:.5} \t | failed (in endval steady) \t\t | {:.1}\n",
                            iteration, new_share, elapsed
                        ));
                    } else {
                        table.push(format!(
                            "{} \t | {:.5} \t | failed \t | {} \t | {:.1}\n",
                            iteration,
                            new_share,
                            fmt_e(maxerror),
                            elapsed
                        ));
                    }
                }
            }
            print!("{}", table.concat());
        }
        println!();
        Ok(Homotopy {
            completed_share: completed,
            endo,
            exo,
            steady_state: ss,
            exo_steady_state: xs,
            iteration,
            maxerror,
            solver_iter,
        })
    }

    /// `perfect_foresight_solver_core.m` (non-block, non-bytecode): `sim1_linear` for linear
    /// models / `linear_approximation`, `sim1` otherwise. Returns (success, maxerror, iterations).
    #[allow(clippy::too_many_arguments)]
    fn core(
        &self,
        model: &Model,
        o: &SimulOptions,
        endo: &mut Mat,
        exo: &mut Mat,
        ss: &[f64],
        xs: &[f64],
        cp: Option<&[crate::ep::controlled::ControlledPeriod]>,
    ) -> (bool, f64, usize) {
        let has_lead_lag = self.maximum_lag > 0 && self.maximum_lead > 0;
        let linear = o.linear_approximation
            || (model.linear && has_lead_lag && o.stack_solve_algo != 1 && o.stack_solve_algo != 6);
        // stream PP: STEADY_STATE() in the model reads oo_.steady_state (perfect_foresight_problem.m)
        let params = model.params_with_steady_state(&self.params, ss);
        // ---- stream EP: perfect_foresight_controlled_paths (sim1 / sim1_linear with a controlled_paths_by_period)
        if let Some(cp) = cp {
            return crate::ep::controlled::sim1(model, o, endo, exo, ss, xs, &params, self.maximum_lag, cp, linear);
        }
        // ---- model(block): block-wise solving (solve_block_decomposed_problem.m)
        if let Some(bs) = model.block_solve.as_ref().and_then(|b| b.dynamic.as_ref()) {
            return block_solve::solve_block_decomposed(model, bs, o, endo, exo, &params, self.maximum_lag);
        }
        // ---- stack_solve_algo 1-3 and 6-10, lmmcp, robust_lin_solve, purely
        // backward/forward/static models
        if let Some(r) = pf_q1::core(
            model,
            o,
            self.steady_opts.solve_algo,
            endo,
            exo,
            &params,
            &self.params,
            self.maximum_lag,
            self.maximum_lead,
            linear,
        ) {
            return r;
        }
        if linear {
            sim1_linear(model, o, endo, exo, ss, xs, &params, self.maximum_lag)
        } else {
            sim1(model, o, endo, exo, &params, self.maximum_lag)
        }
    }

    /// Writes `oo_.endo_simul` / `oo_.exo_simul` to `<dir>/<fname>/Output/<fname>_endo_simul.csv`
    /// and `_exo_simul.csv` (one row per period, period 1 = first simulated period).
    pub fn write_results(&self, model: &Model, dir: &Path, fname: &str) -> Result<()> {
        let out = dir.join(fname).join("Output");
        std::fs::create_dir_all(&out)?;
        let ml = self.maximum_lag as i64;
        if let Some(e) = &self.endo_simul {
            let mut s = String::from("period");
            for nme in &model.endo_names {
                s.push(',');
                s.push_str(nme);
            }
            s.push('\n');
            for c in 0..e.ncols() {
                s.push_str(&(c as i64 + 1 - ml).to_string());
                for j in 0..e.nrows() {
                    s.push_str(&format!(",{:.17e}", e[(j, c)]));
                }
                s.push('\n');
            }
            std::fs::write(out.join(format!("{fname}_endo_simul.csv")), s)?;
        }
        if let Some(x) = &self.exo_simul {
            let mut s = String::from("period");
            for nme in &model.exo_names {
                s.push(',');
                s.push_str(nme);
            }
            s.push('\n');
            for r in 0..x.nrows() {
                s.push_str(&(r as i64 + 1 - ml).to_string());
                for k in 0..x.ncols() {
                    s.push_str(&format!(",{:.17e}", x[(r, k)]));
                }
                s.push('\n');
            }
            std::fs::write(out.join(format!("{fname}_exo_simul.csv")), s)?;
        }
        Ok(())
    }

    /// `rplot.m` (rplottype 0: all variables in one panel), as a `graphics::Figure`.
    fn rplot(&self, model: &Model, o: &Options, vars: &[String], dir: &Path, fname: &str) -> Result<()> {
        let e = self.endo_simul.as_ref().ok_or_else(|| anyhow!("rplot: oo_.endo_simul is empty."))?;
        let ml = self.maximum_lag as i64;
        let first = (ml - 1).max(0) as usize; // max(1, maximum_lag), 0-based
        let xs: Vec<f64> = (first..e.ncols()).map(|c| (c as i64 + 1 - ml) as f64).collect();
        let mut panel = Panel {
            title: if vars.len() == 1 { format!("Plot of {}", vars[0]) } else { format!("Plot of {}", vars.join(" ")) },
            x_label: Some("Periods".into()),
            legend: vars.len() > 1,
            ..Default::default()
        };
        for (k, v) in vars.iter().enumerate() {
            let y: Vec<f64> = if let Some(j) = model.endo_index(v) {
                (first..e.ncols()).map(|c| e[(j, c)]).collect()
            } else if let (Some(kx), Some(x)) = (model.exo_index(v), &self.exo_simul) {
                (first..x.nrows()).map(|r| x[(r, kx)]).collect()
            } else {
                bail!("rplot: One of the variables specified does not exist");
            };
            panel.series.push(Series::Line {
                x: xs.clone(),
                y,
                color: Color::Palette(k),
                style: LineStyle::Solid,
                width: 1.0,
                label: Some(v.clone()),
            });
        }
        let fig = Figure {
            name: format!("SimulatedTrajectory_{}", vars.first().map(String::as_str).unwrap_or("")),
            title: "Simulated Trajectory".into(),
            rows: 1,
            cols: 1,
            panels: vec![panel],
        };
        let formats = match opt(o, "graph_format") {
            Some(OptValue::Str(s)) => GraphFormat::parse(s).into_iter().collect(),
            Some(OptValue::List(l)) => {
                l.iter().filter_map(|v| if let OptValue::Str(s) = v { GraphFormat::parse(s) } else { None }).collect()
            }
            _ => vec![GraphFormat::Pdf],
        };
        let gopts = GraphOptions {
            enabled: !flag(o, "nograph").unwrap_or(false),
            formats,
            dir: graphics::graph_dir(dir, fname),
        };
        graphics::save(&fig, fname, &gopts)?;
        Ok(())
    }
}

/// Result of `create_scenario`.
struct Scenario {
    steady_success: bool,
    endo: Mat,
    exo: Mat,
    /// Recomputed terminal steady state (endogenous, exogenous).
    steady: Option<(Vec<f64>, Vec<f64>)>,
    cp_scaled: Option<Vec<crate::ep::controlled::ControlledPeriod>>,
}

struct Homotopy {
    completed_share: f64,
    endo: Mat,
    exo: Mat,
    steady_state: Vec<f64>,
    exo_steady_state: Vec<f64>,
    iteration: usize,
    maxerror: f64,
    solver_iter: usize,
}

/// `disp_steady_state.m`.
pub fn print_steady_state(model: &Model, ys: &[f64]) {
    print!("{}", steady_state_text(model, ys));
}

/// The text `disp_steady_state.m` prints: a blank line, the title, a blank line, then
/// `fprintf('%s \t\t %g\n', ...)` per declared variable, with the names padded as
/// `char(M_.endo_names)` pads them (to the longest name, auxiliary variables included).
pub fn steady_state_text(model: &Model, ys: &[f64]) -> String {
    let w = model.endo_names.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut s = String::from("\nSTEADY-STATE RESULTS:\n\n");
    for i in 0..model.orig_endo_nbr {
        s += &format!("{:<w$} \t\t {}\n", model.endo_names[i], fmt_g(ys[i]));
    }
    s
}

fn exo_row(exo: &Mat, r: usize) -> Vec<f64> {
    exo.row(r).iter().copied().collect()
}

fn conditions(endo: &Mat, ml: usize, periods: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = endo.nrows();
    let y0: Vec<f64> = if ml > 0 { endo.column(ml - 1).iter().copied().collect() } else { vec![f64::NAN; n] };
    let yt: Vec<f64> = if endo.ncols() > ml + periods {
        endo.column(ml + periods).iter().copied().collect()
    } else {
        vec![f64::NAN; n]
    };
    let mut y = vec![0.0; n * periods];
    for t in 0..periods {
        for j in 0..n {
            y[t * n + j] = endo[(j, ml + t)];
        }
    }
    (y0, yt, y)
}

/// `compute_maxerror.m`.
pub fn compute_maxerror(model: &Model, endo: &Mat, exo: &Mat, params: &[f64], ml: usize, periods: usize) -> f64 {
    let (y0, yt, y) = conditions(endo, ml, periods);
    let mut p = StackedProblem::new(model);
    let mut res = vec![0.0; y.len()];
    p.eval(&y, &y0, &yt, &|t| exo_row(exo, t + ml), params, &mut res, None);
    inf_norm(&res)
}

/// Port of `sim1.m` (no controlled paths): Newton on the stacked system.
pub fn sim1(
    model: &Model,
    o: &SimulOptions,
    endo: &mut Mat,
    exo: &Mat,
    params: &[f64],
    ml: usize,
) -> (bool, f64, usize) {
    let verbose = o.verbosity > 0 && !o.noprint;
    let n = model.n_endo();
    let mut periods = o.periods;
    let (y0, mut yt, mut y) = conditions(endo, ml, periods);
    if verbose {
        println!();
        println!("{}", "-".repeat(56));
        println!("MODEL SIMULATION:");
        println!();
    }
    let h1 = Instant::now();
    let mut iter = 1;
    let mut converged = false;
    let mut prob = StackedProblem::new(model);
    let mut a = StackedJacobian::new(&prob.s, periods);
    let mut res = vec![0.0; n * periods];
    let mut err = f64::NAN;
    while !(converged || iter > o.maxit) {
        let h2 = Instant::now();
        prob.eval(&y, &y0, &yt, &|t| exo_row(exo, t + ml), params, &mut res, Some(&mut a));
        if o.endogenous_terminal_period && iter > 1 {
            for it in 0..periods {
                if inf_norm(&res[it * n..(it + 1) * n]) < o.tolf / 1e7 {
                    let itm = it + 1;
                    if itm < periods {
                        yt = y[itm * n..(itm + 1) * n].to_vec();
                        for t in itm..periods {
                            for j in 0..n {
                                endo[(j, ml + t)] = y[t * n + j];
                            }
                        }
                        y.truncate(itm * n);
                        periods = itm;
                        res.truncate(itm * n);
                        a = StackedJacobian::new(&prob.s, periods);
                        prob.eval(&y, &y0, &yt, &|t| exo_row(exo, t + ml), params, &mut res, Some(&mut a));
                    }
                    break;
                }
            }
        }
        err = inf_norm(&res);
        if err < o.tolf {
            converged = true;
        } else {
            let dy = lin_solve(&mut a, &prob.s, &res);
            for (v, d) in y.iter_mut().zip(&dy) {
                *v -= d;
            }
        }
        if verbose {
            println!("Iter: {},\t err. = {},\t time = {}", iter, fmt_g(err), fmt_g(h2.elapsed().as_secs_f64()));
        }
        iter += 1;
    }
    for t in 0..periods {
        for j in 0..n {
            endo[(j, ml + t)] = y[t * n + j];
        }
    }
    if o.endogenous_terminal_period {
        err = compute_maxerror(model, endo, exo, params, ml, o.periods);
    }
    let success;
    if converged {
        if y.iter().any(|v| !v.is_finite()) {
            success = false;
            if verbose {
                println!();
                println!("Total time of simulation: {}.", fmt_g(h1.elapsed().as_secs_f64()));
                println!("Simulation terminated with NaN or Inf in the residuals or endogenous variables.");
                println!(
                    "There is most likely something wrong with your model. Try model_diagnostics or another simulation method."
                );
                println!("{}", "-".repeat(105));
            }
        } else {
            if verbose {
                println!();
                println!("Total time of simulation: {}.", fmt_g(h1.elapsed().as_secs_f64()));
                println!("{}", "-".repeat(56));
            }
            success = true;
        }
    } else {
        if verbose {
            println!();
            println!("Total time of simulation: {}.", fmt_g(h1.elapsed().as_secs_f64()));
            println!("Maximum number of iterations is reached (modify option maxit).");
            println!("{}", "-".repeat(62));
        }
        success = false;
    }
    if verbose {
        println!();
    }
    (success, err, iter)
}

/// `lin_solve.m` with `stack_solve_algo = 0`: `A\b`, 0 if `norm(b) < sqrt(eps)`,
/// non-finite entries set to 0.
fn lin_solve(a: &mut StackedJacobian, s: &stacked::StackedStructure, b: &[f64]) -> Vec<f64> {
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nb < f64::EPSILON.sqrt() {
        return vec![0.0; b.len()];
    }
    let mut x = b.to_vec();
    if !a.solve(s, &mut x) {
        println!("Warning: Matrix is singular to working precision.");
        x.iter_mut().for_each(|v| *v = f64::NAN);
    }
    for v in &mut x {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    x
}

/// Port of `sim1_linear.m` (no controlled paths): one solve of the system linearized at the
/// steady state `ss`, `xs`.
#[allow(clippy::too_many_arguments)]
pub fn sim1_linear(
    model: &Model,
    o: &SimulOptions,
    endo: &mut Mat,
    exo: &Mat,
    ss: &[f64],
    xs: &[f64],
    params: &[f64],
    ml: usize,
) -> (bool, f64, usize) {
    let verbose = o.verbosity > 0;
    let n = model.n_endo();
    let periods = o.periods;
    if verbose {
        println!();
        println!("{}", "-".repeat(80));
        println!("MODEL SIMULATION:");
        println!();
    }
    let mut y3n = Vec::with_capacity(3 * n);
    for _ in 0..3 {
        y3n.extend_from_slice(ss);
    }
    let mut d1 = vec![0.0; n];
    model.dynamic_residuals(&y3n, xs, params, &mut d1);
    let prob = StackedProblem::new(model);
    let pattern = model.dynamic_jacobian_pattern();
    let mut vals = vec![0.0; pattern.len()];
    let mut ws = model.workspace();
    model.dynamic_jacobian_values_with(&y3n, xs, params, &mut ws, &mut vals);
    if !o.steadystate_nocheck && d1.iter().any(|v| v.abs() > f64::EPSILON.cbrt()) {
        println!("ERROR: Jacobian is not evaluated at the steady state!");
        return (false, f64::NAN, 1);
    }
    // deviations from the steady state
    let mut dev = endo.clone();
    for c in 0..dev.ncols() {
        for j in 0..n {
            dev[(j, c)] -= ss[j];
        }
    }
    let xdev = Mat::from_fn(exo.nrows(), exo.ncols(), |r, k| exo[(r, k)] - xs[k]);
    let (y0, yt, mut yv) = conditions(&dev, ml, periods);
    let h2 = Instant::now();
    let mut a = StackedJacobian::new(&prob.s, periods);
    for t in 0..periods {
        a.set_period(t, &vals);
    }
    let linres = |yv: &[f64]| -> Vec<f64> {
        let mut res = vec![0.0; n * periods];
        let mut z = vec![0.0; 3 * n + model.n_exo()];
        for t in 0..periods {
            let prev = if t == 0 { &y0[..] } else { &yv[(t - 1) * n..t * n] };
            let next = if t + 1 == periods { &yt[..] } else { &yv[(t + 1) * n..(t + 2) * n] };
            z[..n].copy_from_slice(prev);
            z[n..2 * n].copy_from_slice(&yv[t * n..(t + 1) * n]);
            z[2 * n..3 * n].copy_from_slice(next);
            for k in 0..model.n_exo() {
                z[3 * n + k] = xdev[(t + ml, k)];
            }
            let r = &mut res[t * n..(t + 1) * n];
            for (q, &(i, c)) in pattern.iter().enumerate() {
                // 0 * NaN in the unused initial/terminal blocks is skipped as Dynare's sparse product
                if vals[q] != 0.0 {
                    r[i] += vals[q] * z[c];
                }
            }
        }
        res
    };
    let res0 = linres(&yv);
    let err0 = inf_norm(&res0);
    let mut bx = res0.clone();
    let ok = {
        let nb = bx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nb < f64::EPSILON.sqrt() {
            bx.fill(0.0);
            true
        } else {
            a.solve(&prob.s, &mut bx)
        }
    };
    if !ok {
        if verbose {
            println!();
            println!("Singularity problem! The jacobian matrix of the stacked model cannot be inverted.");
        }
        return (false, f64::NAN, 1);
    }
    for v in &mut bx {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    for (v, d) in yv.iter_mut().zip(&bx) {
        *v -= d;
    }
    let res1 = linres(&yv);
    let err = inf_norm(&res1);
    if verbose {
        println!(
            "Iter: 1,\t Initial err. = {},\t err. = {},\t time = {}",
            fmt_g(err0),
            fmt_g(err),
            fmt_g(h2.elapsed().as_secs_f64())
        );
        println!("{}", "-".repeat(80));
    }
    let success = !(res1.iter().any(|v| !v.is_finite()) || yv.iter().any(|v| !v.is_finite()));
    if !success && verbose {
        println!();
        println!("Simulation terminated with NaN or Inf in the residuals or endogenous variables.");
        println!(
            "There is most likely something wrong with your model. Try model_diagnostics or another simulation method."
        );
    }
    for t in 0..periods {
        for j in 0..n {
            endo[(j, ml + t)] = yv[t * n + j] + ss[j];
        }
    }
    if verbose {
        println!();
    }
    (success, err, 1)
}

/// A light `model_diagnostics.m`: static Jacobian rank and collinear equations at the
/// current steady state.
pub fn model_diagnostics(model: &Model, ys: &[f64], xs: &[f64], p: &[f64]) {
    let j = model.static_jacobian(ys, xs, p);
    let sv = crate::linalg::singular_values(&j);
    let tol = sv.iter().cloned().fold(0.0, f64::max) * (j.nrows().max(j.ncols()) as f64) * f64::EPSILON;
    let rank = sv.iter().filter(|&&s| s > tol).count();
    if rank < j.nrows() {
        println!(
            "MODEL_DIAGNOSTICS: The Jacobian of the static model is singular\nMODEL_DIAGNOSTICS: there is {} colinear relationships between the variables and the equations",
            j.nrows() - rank
        );
    } else {
        println!("MODEL_DIAGNOSTICS: No obvious problems with this mod-file were detected.");
    }
}
