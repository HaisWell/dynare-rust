// SPDX-License-Identifier: GPL-3.0-or-later
//! Optimal policy (streams P, P2): `ramsey_model` / `ramsey_policy`, `evaluate_planner_objective`,
//! `discretionary_policy` and `osr`.
//!
//! The symbolic side (Lagrange multipliers, Ramsey FOCs, planner objective derivatives, the
//! `optimal_policy_discount_factor` parameter) is done by the preprocessor port in
//! [`crate::model::planner`]. This module holds the numerical side:
//!
//! * [`ramsey`]: the Ramsey steady state (`dyn_ramsey_static.m`, with or without
//!   `steady_state_model`, `csolve.m`) and `ramsey_constraints`;
//! * [`discretion`]: `discretionary_policy_1.m` / `discretionary_policy_engine.m`;
//! * [`osr`]: `+osr/run.m`, `+osr/objective.m` (orders 1 and 2, bounds);
//! * [`welfare`]: `evaluate_planner_objective.m` (order 1, order 2, linear-quadratic) and the
//!   theoretical variances it needs.
//!
//! [`Session`] runs the commands in file order (the driver hands it every
//! [`Command::Policy`] and, for Ramsey models, `stoch_simul`), keeping `M_.params`, the
//! `stoch_simul` options and the last decision rule between commands as Dynare's globals do.
//! The `stoch_simul` part of the output goes through stream S's `analysis::stoch_simul` (order
//! 1: model summary, decision rules, moments, decompositions, IRF files and figures) or stream
//! K's `korder::stoch_simul::run_at` (order >= 2), when the driver gave an output context.

pub mod discretion;
pub mod osr;
pub mod ramsey;
pub mod welfare;

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;

use crate::analysis::{LinearSolution, OutputContext, StochSimulOptions};
use crate::deterministic::SteadyStateModel;
use crate::linalg::Mat;
use crate::model::Model;
use crate::model::higher::DynamicDerivatives;
use crate::modfile::{Command, ModFile, OptValue, Options, opt};
use crate::solver::korder::KOrderDr;
use crate::solver::{DecisionRule, FirstOrderSolver, SolveError, SolverOptions, SteadyOptions};
pub use discretion::DiscretionOptions;
pub use ramsey::RamseyConstraint;
pub use welfare::PlannerObjectiveValue;

/// Steady state + decision rule at a parameter vector (`resol.m` for the policy problems).
pub struct Resolver<'a> {
    pub model: &'a Model,
    pub solver: FirstOrderSolver,
    pub steady_opts: SteadyOptions,
    /// `oo_.steady_state` the steady-state search starts from (initval).
    pub ys_guess: Vec<f64>,
    pub qz_criterium: f64,
    /// `Some` for `discretionary_policy` (decision rule from the Dennis algorithm).
    pub discretion: Option<DiscretionOptions>,
    /// `steady_state_model` (`options_.steadystate_flag = 2`).
    pub ssm: Option<SteadyStateModel>,
    /// `ramsey_constraints` (enforced in the Ramsey static problem, see [`ramsey`]).
    pub constraints: Vec<RamseyConstraint>,
}

impl<'a> Resolver<'a> {
    pub fn new(mf: &'a ModFile, discretion: bool) -> Self {
        let model = &mf.model;
        let mut ys_guess = vec![0.0; model.n_endo()];
        for &(i, v) in &mf.initval {
            ys_guess[i] = v;
        }
        Resolver {
            model,
            solver: FirstOrderSolver::new(&model.lead_lag_incidence, model.n_exo(), SolverOptions::default()),
            steady_opts: steady_options(mf),
            ys_guess,
            qz_criterium: 1.0 + 1e-6,
            discretion: discretion.then(DiscretionOptions::default),
            ssm: SteadyStateModel::from_modfile(mf).ok().flatten(),
            constraints: Vec::new(),
        }
    }

    /// [`Resolver::new`] plus the `ramsey_constraints` of the file (Ramsey models only).
    pub fn with_constraints(mf: &'a ModFile) -> Result<Self> {
        let mut r = Self::new(mf, false);
        if mf.model.planner().is_some_and(|pl| pl.ramsey.is_some()) {
            r.constraints = ramsey::parse_constraints(&mf.model, &mf.policy.ramsey_constraints)?;
        }
        Ok(r)
    }

    /// The steady state and the parameters (a steady-state model may change them) at
    /// parameters `p` (Ramsey: `dyn_ramsey_static`; otherwise `evaluate_steady_state`).
    pub fn steady_state_params(&self, p: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SolveError> {
        let m = self.model;
        if m.planner().is_some_and(|pl| pl.ramsey.is_some()) {
            let rs = ramsey::RamseyStatic { model: m, ssm: self.ssm.as_ref(), constraints: &self.constraints };
            return rs.steady_state(p, &self.ys_guess, &self.steady_opts);
        }
        let x = vec![0.0; m.n_exo()];
        crate::deterministic::evaluate_steady_state(
            m,
            self.ssm.as_ref(),
            &self.ys_guess,
            &x,
            p,
            &self.steady_opts,
            true,
        )
    }

    /// The steady state at parameters `p`.
    pub fn steady_state(&self, p: &[f64]) -> Result<Vec<f64>, SolveError> {
        self.steady_state_params(p).map(|(y, _)| y)
    }

    /// The first-order decision rule at parameters `p`.
    pub fn resolve(&self, p: &[f64]) -> Result<DecisionRule, SolveError> {
        self.resolve_params(p).map(|(dr, _)| dr)
    }

    /// The first-order decision rule and the parameters it was computed with.
    pub fn resolve_params(&self, p: &[f64]) -> Result<(DecisionRule, Vec<f64>), SolveError> {
        if let Some(o) = &self.discretion {
            return discretion::discretionary_policy_1(self.model, p, o, self.steady_opts.tolf)
                .map(|dr| (dr, p.to_vec()))
                .map_err(|(info, info2)| SolveError::new(info, info2));
        }
        let (ys, p) = self.steady_state_params(p)?;
        let g1 = self.model.dynamic_jacobian(&ys, &vec![0.0; self.model.n_exo()], &p);
        self.solver.solve(&g1, &ys).map(|dr| (dr, p))
    }

    /// The decision rule at `order >= 2` (`deriv` = `Model::dynamic_derivatives(order)`).
    pub fn resolve_k(
        &self,
        p: &[f64],
        order: usize,
        deriv: &DynamicDerivatives,
        sigma_e: &Mat,
    ) -> Result<(KOrderDr, Vec<f64>), SolveError> {
        let (ys, p) = self.steady_state_params(p)?;
        let x = vec![0.0; self.model.n_exo()];
        let g1 = self.model.dynamic_jacobian(&ys, &x, &p);
        let dr1 = self.solver.solve(&g1, &ys)?;
        let vals = deriv.eval(&ys, &x, &p);
        crate::solver::korder::solve(&dr1, &self.model.lead_lag_incidence, &g1, deriv, &vals, sigma_e, order)
            .map(|k| (k, p))
            .map_err(|_| SolveError::new(5, f64::NAN))
    }
}

/// Options of the last `steady(...)` command (solve_algo, maxit, tolf, tolx).
fn steady_options(mf: &ModFile) -> SteadyOptions {
    let mut o = SteadyOptions { linear: mf.model.linear, ..Default::default() };
    for c in &mf.commands {
        if let Command::Steady(opts) = c {
            if let Some(v) = num(opts, "solve_algo") {
                o.solve_algo = v as u8;
            }
            if let Some(v) = num(opts, "maxit") {
                o.maxit = v as usize;
            }
            if let Some(v) = num(opts, "tolf") {
                o.tolf = v;
            }
            if let Some(v) = num(opts, "tolx") {
                o.tolx = v;
            }
        }
    }
    o
}

fn num(o: &Options, k: &str) -> Option<f64> {
    match opt(o, k) {
        Some(OptValue::Num(v)) => Some(*v),
        _ => None,
    }
}

/// Result of an OSR run (`oo_.osr`).
#[derive(Clone, Debug)]
pub struct OsrResult {
    pub objective_function: f64,
    /// (parameter name, optimal value)
    pub optim_params: Vec<(String, f64)>,
    pub iterations: usize,
    pub func_count: usize,
}

/// Policy options that are not `stoch_simul` options (kept out of the stoch_simul warnings).
const POLICY_OPTIONS: &[&str] = &[
    "instruments",
    "planner_discount",
    "planner_discount_latex_name",
    "opt_algo",
    "discretionary_tol",
    "maxit",
    "optim",
];

/// State carried between commands (Dynare's `M_.params`, `options_`, `oo_.dr`,
/// `oo_.planner_objective_value`, `oo_.osr`).
#[derive(Default)]
pub struct Session {
    pub params: Option<Vec<f64>>,
    pub dr: Option<DecisionRule>,
    /// Second- or third-order decision rule of the last stoch_simul-like command.
    pub kdr: Option<KOrderDr>,
    /// `options_.order` of the last stoch_simul-like command.
    pub order: usize,
    pub planner_objective_value: Option<PlannerObjectiveValue>,
    pub osr: Option<OsrResult>,
    /// Theoretical variance of the last var_list (declaration order of the list).
    pub var: Option<Mat>,
    pub var_list: Vec<usize>,
    pub quiet: bool,
    /// `options_` of the stoch_simul part (persistent across commands, as in Dynare).
    pub ss_opts: StochSimulOptions,
    /// Where the stoch_simul output (files, figures) goes; `None`: no stoch_simul output.
    pub ctx: Option<OutputContext>,
    hessian_eq_zero: Option<bool>,
    /// `oo_.exo_simul(1,:)` of the last order-1 stoch_simul simulation (`periods > 0`): the
    /// initial shock `u` of `evaluate_planner_objective`'s conditional welfare.
    pub exo_simul_first: Option<Vec<f64>>,
    /// `options_.ramsey.periods` / `options_.ramsey.drop` (`evaluate_planner_objective(periods=,
    /// drop=)`; read by the order >= 3 simulated welfare). `None`: Dynare's defaults 10000/1000.
    pub ramsey_periods: Option<usize>,
    pub ramsey_drop: Option<usize>,
    /// Shocks of the order >= 3 simulated welfare (`n_exo x periods`) instead of drawing them
    /// (tests feeding MATLAB's draws).
    pub planner_shocks: Option<Mat>,
}

/// Whether [`Session::run`] handles this command (the driver calls it instead of its own arm).
pub fn handles(mf: &ModFile, c: &Command) -> bool {
    match c {
        Command::Policy(..) => true,
        Command::StochSimul(..) => mf.model.planner().is_some_and(|p| p.ramsey.is_some()),
        _ => false,
    }
}

impl Session {
    pub fn new(quiet: bool) -> Self {
        Session { quiet, ss_opts: StochSimulOptions::default(), ..Default::default() }
    }

    /// The driver's output context (`stoch_simul` printout through `analysis`, IRF files and
    /// figures, results JSON files).
    pub fn set_output(&mut self, ctx: OutputContext) {
        self.ctx = Some(ctx);
    }

    fn params(&self, mf: &ModFile) -> Vec<f64> {
        self.params.clone().unwrap_or_else(|| mf.model.params.clone())
    }

    /// `M_.hessian_eq_zero`.
    fn hessian_eq_zero(&mut self, m: &Model) -> bool {
        *self.hessian_eq_zero.get_or_insert_with(|| m.dynamic_derivatives(2).hessian_eq_zero())
    }

    /// Runs one command; `out` = `(output dir, fname)` for the results files.
    pub fn run(&mut self, mf: &ModFile, c: &Command, out: Option<(&Path, &str)>) -> Result<()> {
        match c {
            Command::StochSimul(o, vars) => self.ramsey_stoch_simul(mf, o, vars, out),
            Command::Policy(name, o, vars) => match name.as_str() {
                "ramsey_model" => self.apply_options(o, false),
                "osr_params" => Ok(()),
                "ramsey_policy" => {
                    self.ramsey_stoch_simul(mf, o, vars, out)?;
                    self.evaluate_planner_objective(mf, out)
                }
                "evaluate_planner_objective" => {
                    if let Some(v) = num(o, "periods") {
                        self.ramsey_periods = Some(v as usize);
                    }
                    if let Some(v) = num(o, "drop") {
                        self.ramsey_drop = Some(v as usize);
                    }
                    self.evaluate_planner_objective(mf, out)
                }
                "discretionary_policy" => self.discretionary_policy(mf, o, vars, out),
                "osr" => self.osr(mf, o, vars, out),
                _ => bail!("unknown policy command {name}"),
            },
            _ => Ok(()),
        }
    }

    /// Updates the persistent `stoch_simul` options with a command's options.
    fn apply_options(&mut self, o: &Options, warn: bool) -> Result<()> {
        let o2: Options = o.iter().filter(|(k, _)| !POLICY_OPTIONS.contains(&k.as_str())).cloned().collect();
        let unknown = self.ss_opts.apply_checked(&o2)?;
        if warn {
            for k in unknown {
                println!(
                    "WARNING: option `{k}` is not supported by dynare-rust's optimal policy commands and is ignored."
                );
            }
        }
        Ok(())
    }

    fn var_list(mf: &ModFile, vars: &[String]) -> Result<Vec<usize>> {
        let m = &mf.model;
        if vars.is_empty() {
            return Ok((0..m.orig_endo_nbr).collect());
        }
        let mut out = Vec::new();
        for v in vars {
            let j = m.endo_index(v).ok_or_else(|| anyhow!("unknown variable {v}"))?;
            if !out.contains(&j) {
                out.push(j);
            }
        }
        Ok(out)
    }

    /// The order `stoch_simul` uses (`stoch_simul.m`: `model(linear)` and a zero Hessian force
    /// order 1).
    fn effective_order(&mut self, m: &Model) -> usize {
        let mut order = self.ss_opts.order.max(1);
        if order > 1 && m.linear {
            order = 1;
        }
        if order > 1 && self.hessian_eq_zero(m) {
            if !self.quiet {
                println!("WARNING: stoch_simul: using order = 1 because Hessian is equal to zero");
            }
            order = 1;
        }
        order
    }

    /// `stoch_simul` on a Ramsey model (`ramsey_model` + `stoch_simul`, or `ramsey_policy`).
    fn ramsey_stoch_simul(
        &mut self,
        mf: &ModFile,
        o: &Options,
        vars: &[String],
        out: Option<(&Path, &str)>,
    ) -> Result<()> {
        self.apply_options(o, true)?;
        self.exo_simul_first = None; // set again by an order-1 simulation (report)
        let m = &mf.model;
        ramsey::check_instruments(m, !mf.steady_state_model.is_empty())?;
        let order = self.effective_order(m);
        self.order = order;
        let p0 = self.params(mf);
        let res = Resolver::with_constraints(mf)?;
        if !res.constraints.is_empty() && !self.quiet {
            println!(
                "NOTE: ramsey_constraints are enforced in the Ramsey steady-state problem (bounded solve); Dynare 7.2 ignores them."
            );
        }
        let (ys, p) = res.steady_state_params(&p0).map_err(|e| anyhow!("Ramsey policy: {e}"))?;
        let iv = Self::var_list(mf, vars)?;
        if order == 1 {
            let x = vec![0.0; m.n_exo()];
            let g1 = m.dynamic_jacobian(&ys, &x, &p);
            let dr = res.solver.solve(&g1, &ys).map_err(|e| anyhow!("Ramsey policy: {e}"))?;
            self.params = Some(p);
            self.report(mf, &dr, &iv, vars, &res, out, "ramsey")?;
            self.dr = Some(dr);
            self.kdr = None;
        } else {
            let kdr = self.korder_stoch_simul(mf, order, vars, &ys, &p)?;
            self.params = Some(p);
            self.report_k(mf, &kdr, &iv, &res, out, "ramsey")?;
            self.dr = Some(kdr.first.clone());
            self.kdr = Some(kdr);
        }
        Ok(())
    }

    /// The order >= 2 `stoch_simul` (stream K) at `(ys, p)`: prints and writes its files when
    /// there is an output context, else only solves.
    fn korder_stoch_simul(
        &self,
        mf: &ModFile,
        order: usize,
        vars: &[String],
        ys: &[f64],
        p: &[f64],
    ) -> Result<KOrderDr> {
        let so = self.korder_options(order);
        match &self.ctx {
            Some(ctx) => {
                let r = crate::solver::korder::stoch_simul::run_at(
                    mf,
                    &so,
                    vars,
                    &ctx.dir,
                    &ctx.fname,
                    ctx.seed,
                    Some((ys, p)),
                )?;
                Ok(r.dr)
            }
            None => crate::solver::korder::solve_model(
                &mf.model,
                ys,
                p,
                &mf.sigma_e,
                order,
                SolverOptions { qz_criterium: self.ss_opts.qz_criterium, ..Default::default() },
            ),
        }
    }

    /// Stream K's options from the persistent `options_`.
    fn korder_options(&self, order: usize) -> crate::solver::korder::stoch_simul::StochSimulOptions {
        let s = &self.ss_opts;
        let mut k = crate::solver::korder::stoch_simul::StochSimulOptions {
            order,
            irf: s.irf,
            drop: s.drop,
            periods: s.periods,
            ar: s.ar,
            nomoments: s.nomoments,
            noprint: s.noprint || self.quiet,
            nofunctions: s.nofunctions,
            nograph: s.nograph,
            graph_formats: s.graph_format.clone(),
            qz_criterium: s.qz_criterium,
            ..Default::default()
        };
        if let Some(sh) = &s.irf_shocks {
            k.irf_shocks = sh.clone();
        }
        k
    }

    /// Order-1 output: stream S's `stoch_simul` printout/files when there is an output
    /// context, the policy results JSON, and the theoretical variance of the var_list.
    #[allow(clippy::too_many_arguments)]
    fn report(
        &mut self,
        mf: &ModFile,
        dr: &DecisionRule,
        iv: &[usize],
        vars: &[String],
        res: &Resolver,
        out: Option<(&Path, &str)>,
        what: &str,
    ) -> Result<()> {
        let rows: Vec<usize> = iv.iter().map(|&j| dr.inv_order_var[j]).collect();
        let var = welfare::variance(dr, &rows, &mf.sigma_e, res.qz_criterium);
        if let Some(ctx) = &self.ctx {
            let sol = LinearSolution { dr: dr.clone(), sigma_e: mf.sigma_e.clone(), params: self.params(mf) };
            let mut o = self.ss_opts.clone();
            o.order = 1;
            o.noprint |= self.quiet;
            let names: Vec<String> = if vars.is_empty() {
                iv.iter().map(|&j| mf.model.endo_names[j].clone()).collect()
            } else {
                vars.to_vec()
            };
            let r = crate::analysis::stoch_simul(&mf.model, &sol, &o, &names, ctx, None)?;
            if let Some(ex) = r.exo_simul.as_ref().filter(|e| e.nrows() > 0) {
                self.exo_simul_first = Some(ex.row(0).iter().copied().collect());
            }
        } else if !self.quiet {
            let sol = LinearSolution { dr: dr.clone(), sigma_e: mf.sigma_e.clone(), params: self.params(mf) };
            print!("{}", crate::analysis::model_summary(&mf.model, &sol));
            print!("\n{}", crate::analysis::disp_dr(&mf.model, dr, iv, 1e-6));
            print_moments(&mf.model, dr, iv, &var);
        }
        self.var = Some(var.clone());
        self.var_list = iv.to_vec();
        if let Some((dir, fname)) = out {
            let m = &mf.model;
            let j = json!({
                "command": what,
                "order": 1,
                "endo_names": m.endo_names,
                "steady_state": dr.ys,
                "params": self.params(mf),
                "ghx": mat_rows(&dr.ghx),
                "ghu": mat_rows(&dr.ghu),
                "order_var": dr.order_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
                "state_var": dr.state_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
                "var_list": iv.iter().map(|&j| m.endo_names[j].clone()).collect::<Vec<_>>(),
                "var": mat_rows(&var),
            });
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            std::fs::write(dir.join(format!("{fname}_{what}.json")), serde_json::to_string_pretty(&j)?)?;
        }
        Ok(())
    }

    /// Order >= 2: the policy results JSON (the printout and IRFs come from stream K).
    fn report_k(
        &mut self,
        mf: &ModFile,
        kdr: &KOrderDr,
        iv: &[usize],
        res: &Resolver,
        out: Option<(&Path, &str)>,
        what: &str,
    ) -> Result<()> {
        let dr = &kdr.first;
        let rows: Vec<usize> = iv.iter().map(|&j| dr.inv_order_var[j]).collect();
        let var = welfare::variance(dr, &rows, &mf.sigma_e, res.qz_criterium);
        self.var = Some(var.clone());
        self.var_list = iv.to_vec();
        if let Some((dir, fname)) = out {
            let m = &mf.model;
            let j = json!({
                "command": what,
                "order": kdr.order,
                "endo_names": m.endo_names,
                "steady_state": dr.ys,
                "params": self.params(mf),
                "ghx": mat_rows(&dr.ghx),
                "ghu": mat_rows(&dr.ghu),
                "ghxx": mat_rows(&kdr.ghxx),
                "ghxu": mat_rows(&kdr.ghxu),
                "ghuu": mat_rows(&kdr.ghuu),
                "ghs2": kdr.ghs2.as_slice(),
                "order_var": dr.order_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
                "state_var": dr.state_var.iter().map(|v| v + 1).collect::<Vec<_>>(),
                "var_list": iv.iter().map(|&j| m.endo_names[j].clone()).collect::<Vec<_>>(),
                "var": mat_rows(&var),
            });
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            std::fs::write(dir.join(format!("{fname}_{what}.json")), serde_json::to_string_pretty(&j)?)?;
        }
        Ok(())
    }

    /// Port of `evaluate_planner_objective.m` on the last decision rule.
    fn evaluate_planner_objective(&mut self, mf: &ModFile, out: Option<(&Path, &str)>) -> Result<()> {
        let m = &mf.model;
        let p = self.params(mf);
        let beta = m.planner_discount(&p).unwrap_or(1.0);
        let discretionary = m.planner().is_some_and(|pl| pl.discretionary);
        let qz = self.ss_opts.qz_criterium;
        let v = self.welfare(mf, &p, discretionary, qz)?;
        if beta >= 1.0 {
            println!(
                "evaluate_planner_objective: the planner discount factor is not strictly smaller than 1. Unconditional welfare will not be finite."
            );
        }
        if !self.quiet {
            let ramsey = m.planner().is_some_and(|pl| pl.ramsey.is_some());
            if ramsey {
                println!("\nApproximated value of unconditional welfare:  {:10.8}", v.unconditional);
                println!("\nApproximated value of conditional welfare:");
                println!(
                    "    - with initial Lagrange multipliers set to 0: {:10.8}",
                    v.conditional_zero_initial_multiplier
                );
                println!(
                    "    - with initial Lagrange multipliers set to steady state: {:10.8}\n",
                    v.conditional_steady_initial_multiplier
                );
            } else if discretionary {
                println!(
                    "\nApproximated value of unconditional welfare with discretionary policy:  {:10.8}",
                    v.unconditional
                );
                println!(
                    "\nApproximated value of conditional welfare with discretionary policy: {:10.8}",
                    v.conditional_steady_initial_multiplier
                );
            } else {
                println!("\nApproximated value of unconditional welfare:  {:10.8}", v.unconditional);
                println!(
                    "\nApproximated value of conditional welfare: {:10.8}",
                    v.conditional_steady_initial_multiplier
                );
            }
        }
        if let Some((dir, fname)) = out {
            let j = json!({
                "order": self.order,
                "unconditional": v.unconditional,
                "conditional_zero_initial_multiplier": v.conditional_zero_initial_multiplier,
                "conditional_steady_initial_multiplier": v.conditional_steady_initial_multiplier,
            });
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join(format!("{fname}_planner_objective.json")), serde_json::to_string_pretty(&j)?)?;
        }
        self.planner_objective_value = Some(v);
        Ok(())
    }

    /// The branch of `evaluate_planner_objective.m` for the current order and model.
    fn welfare(&mut self, mf: &ModFile, p: &[f64], discretionary: bool, qz: f64) -> Result<PlannerObjectiveValue> {
        let m = &mf.model;
        let hz = self.order == 2 && self.hessian_eq_zero(m);
        let dr = self.dr.as_ref().ok_or_else(|| {
            anyhow!("evaluate_planner_objective requires decision rules to have previously been computed (e.g. by stoch_simul or discretionary_policy)")
        })?;
        let none = || anyhow!("evaluate_planner_objective: no planner_objective");
        let u0 = self.exo_simul_first.as_deref().unwrap_or(&[]);
        if discretionary || self.order <= 1 {
            return welfare::evaluate_planner_objective_u(m, p, dr, &mf.sigma_e, discretionary, qz, u0)
                .ok_or_else(none);
        }
        if self.order == 2 {
            if hz {
                return welfare::evaluate_planner_objective_u(m, p, dr, &mf.sigma_e, true, qz, u0).ok_or_else(none);
            }
            let kdr = self
                .kdr
                .as_ref()
                .ok_or_else(|| anyhow!("evaluate_planner_objective: no second-order decision rule"))?;
            return welfare::evaluate_planner_objective_order2_u(m, p, kdr, &mf.sigma_e, qz, u0).ok_or_else(none);
        }
        // order > 2: k_order_welfare + simulated unconditional welfare
        let kdr = self.kdr.as_ref().ok_or_else(|| anyhow!("evaluate_planner_objective: no k-order decision rule"))?;
        let periods = self.ramsey_periods.unwrap_or(10000);
        let drop = self.ramsey_drop.unwrap_or(1000);
        let ex = match &self.planner_shocks {
            Some(e) => e.clone(),
            None => {
                // exo_simul(i_exo_var,:) = chol(Sigma_e(i_exo_var,i_exo_var))*randn(nxs, periods)
                use rand::SeedableRng;
                let se = &mf.sigma_e;
                let ne = m.n_exo();
                let iv: Vec<usize> = (0..ne).filter(|&i| se[(i, i)] != 0.0).collect();
                let sub = Mat::from_fn(iv.len(), iv.len(), |i, j| se[(iv[i], iv[j])]);
                let l = sub
                    .cholesky()
                    .ok_or_else(|| anyhow!("evaluate_planner_objective: Sigma_e is not positive definite"))?
                    .l();
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(self.ctx.as_ref().map_or(0, |c| c.seed));
                let z = Mat::from_fn(iv.len(), periods, |_, _| {
                    rand::Rng::sample::<f64, _>(&mut rng, rand_distr::StandardNormal)
                });
                let x = l.transpose() * z; // chol(S) is upper: R'R = S
                let mut ex = Mat::zeros(ne, periods);
                for (a, &i) in iv.iter().enumerate() {
                    ex.set_row(i, &x.row(a));
                }
                ex
            }
        };
        welfare::evaluate_planner_objective_korder(m, p, kdr, &mf.sigma_e, &ex, drop, u0)
    }

    /// Port of `discretionary_policy.m`.
    fn discretionary_policy(
        &mut self,
        mf: &ModFile,
        o: &Options,
        vars: &[String],
        out: Option<(&Path, &str)>,
    ) -> Result<()> {
        self.apply_options(o, true)?;
        let m = &mf.model;
        let pl = m.planner().ok_or_else(|| anyhow!("discretionary_policy: no planner_objective"))?;
        let instr_nbr = m.n_endo() - m.n_equations();
        if instr_nbr == 0 {
            bail!(
                "discretionary_policy:: There are no available instruments, because the model has as many equations as variables."
            );
        }
        if pl.instruments.len() < instr_nbr {
            bail!("discretionary_policy:: There are fewer declared instruments than omitted equations.");
        } else if pl.instruments.len() > instr_nbr {
            bail!("discretionary_policy:: There are more declared instruments than omitted equations.");
        }
        let p = self.params(mf);
        let mut res = Resolver::new(mf, true);
        if let (Some(d), Some(v)) = (res.discretion.as_mut(), num(o, "discretionary_tol")) {
            d.tol = v;
        }
        if let (Some(d), Some(v)) = (res.discretion.as_mut(), num(o, "maxit")) {
            d.maxit = v as usize;
        }
        if let Some(d) = res.discretion.as_mut() {
            d.qz_criterium = self.ss_opts.qz_criterium;
        }
        let dr = res.resolve(&p).map_err(|e| anyhow!("discretionary_policy: {e}"))?;
        self.order = 1;
        self.params = Some(p);
        let iv = Self::var_list(mf, vars)?;
        self.report(mf, &dr, &iv, vars, &res, out, "discretionary_policy")?;
        if !self.quiet && !self.ss_opts.noprint {
            // discretionary_policy.m: disp_steady_state, then the multipliers
            crate::deterministic::print_steady_state(m, &dr.ys);
            for j in m.orig_endo_nbr.saturating_sub(1)..m.endo_names.len() {
                if m.endo_names[j].starts_with("mult_") {
                    println!("{} \t\t {}", m.endo_names[j], crate::deterministic::fmt_g(dr.ys[j]));
                }
            }
        }
        self.dr = Some(dr);
        self.kdr = None;
        self.evaluate_planner_objective(mf, out)
    }

    /// Port of `+osr/run.m`.
    fn osr(&mut self, mf: &ModFile, o: &Options, vars: &[String], out: Option<(&Path, &str)>) -> Result<()> {
        self.apply_options(o, true)?;
        let m = &mf.model;
        let order = self.ss_opts.order.max(1);
        let pd = &mf.policy;
        let use_weights = !pd.optim_weights.is_empty();
        if use_weights && order != 1 {
            bail!("OSR: OSR without planner_objective only supports order=1.");
        }
        if !use_weights && order > 2 {
            bail!(
                "OSR: osr with planner_objective at order={order} (simulated welfare) is not ported; use order = 1 or 2."
            );
        }
        if pd.osr_params.is_empty() {
            bail!("OSR: no osr_params declared");
        }
        let p0 = self.params(mf);
        if pd.osr_params.iter().any(|&i| p0[i].is_nan()) {
            bail!("OSR: At least one of the initial parameter values for osr_params is NaN");
        }
        if m.lead_lag_incidence.iter().all(|r| r[2] == 0) {
            bail!("OSR: Backward or static model: no point in using OSR");
        }
        let res = Resolver::new(mf, false);
        let weights = if use_weights {
            // i_var = unique(i_var); positions in the decision rule
            let (iv, w) = osr::weight_block(m, &pd.optim_weights);
            let dr0 = res.resolve(&p0).ok();
            let rows = match &dr0 {
                Some(dr) => iv.iter().map(|&j| dr.inv_order_var[j]).collect(),
                None => {
                    let s = crate::solver::DrStructure::new(&m.lead_lag_incidence, m.n_exo());
                    iv.iter().map(|&j| s.inv_order_var[j]).collect()
                }
            };
            Some((rows, w))
        } else {
            None
        };
        // the welfare branch osr.objective takes (resol at options_.order, no stoch_simul reset)
        let welfare_kind = if use_weights || order == 1 {
            osr::WelfareKind::Order1
        } else if self.hessian_eq_zero(m) {
            osr::WelfareKind::LinearQuadratic
        } else {
            osr::WelfareKind::Order2(m.dynamic_derivatives(2))
        };
        let obj = osr::OsrObjective {
            resolver: &res,
            base_params: p0.clone(),
            i_params: pd.osr_params.clone(),
            weights,
            sigma_e: mf.sigma_e.clone(),
            welfare: welfare_kind,
        };
        let x0: Vec<f64> = pd.osr_params.iter().map(|&i| p0[i]).collect();
        if !self.quiet {
            println!("\nOPTIMAL SIMPLE RULE");
        }
        let (l0, info0, _) = obj.loss(&x0);
        if info0 != 0 {
            println!("OSR: the initial parameters give info {info0} ({})", crate::solver::describe(info0));
        } else if !self.quiet {
            println!("\nOSR: Initial value of the objective function: {} \n", crate::analysis::table::fmt_g(l0, 0, 6));
        }
        if l0.is_infinite() {
            bail!("OSR: Initial likelihood is infinite");
        }
        let opt_algo = num(o, "opt_algo").unwrap_or(4.0) as i32;
        let bounds: Vec<(f64, f64)> = pd
            .osr_params
            .iter()
            .map(|&i| {
                pd.osr_bounds
                    .iter()
                    .rev()
                    .find(|b| b.0 == i)
                    .map(|b| (b.1, b.2))
                    .unwrap_or((f64::NEG_INFINITY, f64::INFINITY))
            })
            .collect();
        let names: Vec<String> = pd.osr_params.iter().map(|&i| m.param_names[i].clone()).collect();
        let r = osr::minimize(
            &obj,
            &x0,
            opt_algo,
            (!pd.osr_bounds.is_empty()).then_some(&bounds[..]),
            &names,
            opt(o, "optim"),
            !self.quiet,
            self.ctx.as_ref().map_or(0, |c| c.seed), // --seed (stochastic optimizers 2, 9)
        )?;
        let mut p = p0.clone();
        for (&i, &v) in pd.osr_params.iter().zip(&r.x) {
            p[i] = v;
        }
        let optim_params: Vec<(String, f64)> =
            pd.osr_params.iter().zip(&r.x).map(|(&i, &v)| (m.param_names[i].clone(), v)).collect();
        if !self.quiet {
            println!("\nOPTIMAL VALUE OF THE PARAMETERS\n");
            for (n, v) in &optim_params {
                println!("{n:<12} {v:>12.6}");
            }
            println!("\nObjective function : {}\n", crate::analysis::table::fmt_g(r.fval, 16, 6));
        }
        self.osr = Some(OsrResult {
            objective_function: r.fval,
            optim_params: optim_params.clone(),
            iterations: r.iterations,
            func_count: r.func_count,
        });
        // stoch_simul at the optimum
        let st_order = self.effective_order(m);
        self.order = st_order;
        let iv = Self::var_list(mf, vars)?;
        let (ys, p) =
            res.steady_state_params(&p).map_err(|e| anyhow!("osr: no solution at the optimal parameters: {e}"))?;
        self.params = Some(p.clone());
        if st_order == 1 {
            let dr = res.resolve(&p).map_err(|e| anyhow!("osr: no solution at the optimal parameters: {e}"))?;
            self.report(mf, &dr, &iv, vars, &res, out, "osr")?;
            self.dr = Some(dr);
            self.kdr = None;
        } else {
            let kdr = self.korder_stoch_simul(mf, st_order, vars, &ys, &p)?;
            self.report_k(mf, &kdr, &iv, &res, out, "osr")?;
            self.dr = Some(kdr.first.clone());
            self.kdr = Some(kdr);
        }
        if let Some((dir, fname)) = out {
            let j = json!({
                "objective_function": r.fval,
                "optim_params": optim_params.iter().map(|(n, v)| json!({"name": n, "value": v})).collect::<Vec<_>>(),
                "iterations": r.iterations,
                "func_count": r.func_count,
            });
            std::fs::write(dir.join(format!("{fname}_osr_results.json")), serde_json::to_string_pretty(&j)?)?;
        }
        Ok(())
    }
}

fn mat_rows(m: &Mat) -> Vec<Vec<f64>> {
    (0..m.nrows()).map(|i| m.row(i).iter().copied().collect()).collect()
}

/// The mean / std / variance table of `disp_th_moments.m` (order 1; used without output context).
fn print_moments(m: &Model, dr: &DecisionRule, iv: &[usize], var: &Mat) {
    println!("\nTHEORETICAL MOMENTS");
    println!("{:<12}{:>10}{:>11}{:>11}", "VARIABLE", "MEAN", "STD. DEV.", "VARIANCE");
    for (k, &j) in iv.iter().enumerate() {
        let v = var[(k, k)];
        println!("{:<12}{:>10.4}{:>11.4}{:>11.4}", m.endo_names[j], dr.ys[j], v.max(0.0).sqrt(), v);
    }
}
