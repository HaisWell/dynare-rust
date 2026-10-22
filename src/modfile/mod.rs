// SPDX-License-Identifier: GPL-3.0-or-later
//! .mod file front end: macro processor, lexer and parser producing a [`ModFile`]
//! (see DESIGN.md, section `modfile`). The equations go into a
//! [`crate::model::ModelBuilder`], which turns them into a [`crate::model::Model`]
//! (Dynare's `M_`) once the whole file is read.

pub mod lexer;
pub mod macro_proc;
pub mod option_check;
mod parser;
pub mod steadystate_m;
pub mod verbatim;
pub use parser::{HeteroShock, Q3Decl};

use crate::linalg::Mat;
use crate::model::Model;
use anyhow::Result;
use std::path::Path;

pub use parser::{Ast, SpecialOp, ast_to_node, parse_expression, special_to_node};
pub use parser::{MATLAB_STATEMENT, OPTIONS_ASSIGNMENT, SET_PARAM};
pub use parser::{SET_SHOCK_STDERR, eval_ast}; // (set_shock_stderr_value, shock_paths values) // [stream X]

/// Parses `path` after running the macro processor with the `-D` style `defines`
/// (`("STAGE", "9")` == `-DSTAGE=9`; an empty value defines `true`).
pub fn parse_mod_file(path: &Path, defines: &[(String, String)]) -> Result<ModFile> {
    let lines = macro_proc::preprocess(path, defines)?;
    let toks = lexer::tokenize(&lines)?;
    with_external_steadystate(path, parser::parse(toks)?, &lines, |l| parser::parse(lexer::tokenize(l)?))
}

/// An external `<fname>_steadystate.m` (Dynare's `steadystate_flag = 1`) is MATLAB code: a file
/// on Dynare's template (see [`steadystate_m`]) is run as the equivalent `steady_state_model`
/// block, appended to the file and parsed again with `reparse`; any other file is refused
/// explicitly instead of being silently ignored. A `steady_state_model` block takes priority
/// over the file (Dynare's `default_option_values.m`), so the file is then irrelevant.
fn with_external_steadystate(
    path: &Path,
    mf: ModFile,
    lines: &[macro_proc::SrcLine],
    reparse: impl Fn(&[macro_proc::SrcLine]) -> Result<ModFile>,
) -> Result<ModFile> {
    if !mf.steady_state_model.is_empty() {
        return Ok(mf);
    }
    if let (Some(dir), Some(stem)) = (path.parent(), path.file_stem().and_then(|s| s.to_str())) {
        let f = dir.join(format!("{stem}_steadystate.m"));
        if f.exists() {
            let src = std::fs::read_to_string(&f).unwrap_or_default();
            match steadystate_m::translate(&src, &mf.model.param_names) {
                Ok(stmts) => {
                    let file: std::sync::Arc<str> = std::sync::Arc::from(f.display().to_string());
                    let mut all = lines.to_vec();
                    let block = std::iter::once("steady_state_model;".to_string())
                        .chain(stmts)
                        .chain(std::iter::once("end;".to_string()));
                    all.extend(block.enumerate().map(|(k, text)| macro_proc::SrcLine {
                        file: file.clone(),
                        line: k + 1,
                        text,
                    }));
                    return reparse(&all);
                }
                Err(e) => anyhow::bail!(
                    "{}: an external steady-state file is MATLAB code that dynare-rust only runs when it follows \
                     Dynare's template (parameters copied with eval loops, plain assignments, ys copied back): {e:#}; \
                     port it to a `steady_state_model;` block in the .mod file (same assignments, one per line), \
                     or delete the file to solve the steady state numerically (initval/guess values)",
                    f.display()
                ),
            }
        }
    }
    Ok(mf)
}

/// [stream X] Command-line settings of the preprocessing (`-I`, `nograph`...).
#[derive(Clone, Debug, Default)]
pub struct ParseOptions {
    pub defines: Vec<(String, String)>,
    /// `-I<path>`: extra `@#include` search paths.
    pub include_paths: Vec<std::path::PathBuf>,
    /// Statements parsed before the file (e.g. `options_.nograph = 1;` for the `nograph` flag).
    pub prelude: Vec<String>,
    /// `stochastic`, `nostrict`, `exclude_eqs=`, `include_eqs=`.
    pub stochastic: bool,
    pub nostrict: bool,
    pub exclude_eqs: Option<String>,
    pub include_eqs: Option<String>,
    /// Directory where `exclude_eqs=FILE` / `include_eqs=FILE` are looked for.
    pub cwd: Option<std::path::PathBuf>,
}

/// [stream X] [`parse_mod_file`] with [`ParseOptions`].
pub fn parse_mod_file_opts(path: &Path, o: &ParseOptions) -> Result<ModFile> {
    let mut lines = macro_proc::preprocess_with(path, &o.defines, &o.include_paths)?;
    let pre: std::sync::Arc<str> = std::sync::Arc::from("<command line>");
    let prelude = o.prelude.iter().enumerate().map(|(k, t)| macro_proc::SrcLine {
        file: pre.clone(),
        line: k + 1,
        text: t.clone(),
    });
    lines.splice(0..0, prelude);
    let toks = lexer::tokenize(&lines)?;
    let flags = parser::q4_flags(o)?;
    let mf = parser::parse_with(toks, flags.clone())?;
    with_external_steadystate(path, mf, &lines, |l| parser::parse_with(lexer::tokenize(l)?, flags.clone()))
}

/// Parses .mod source held in memory (no `@#include` relative to a file).
pub fn parse_mod_str(src: &str, defines: &[(String, String)]) -> Result<ModFile> {
    let lines = macro_proc::preprocess_str(src, "<string>", defines)?;
    let toks = lexer::tokenize(&lines)?;
    parser::parse(toks)
}

/// A parsed .mod file.
#[derive(Clone, Debug)]
pub struct ModFile {
    pub model: Model,
    /// `estim_params_` after each `estimated_params` block, with the number of commands before
    /// it: a command sees the last block written before it (Dynare runs the blocks in file
    /// order), e.g. `estimated_params(overwrite)` between two `method_of_moments` commands.
    pub estimated_params_history: Vec<(usize, Vec<EstimatedParam>)>,
    /// `initval` block: endogenous index, value (evaluated with the calibration at that point).
    pub initval: Vec<(usize, f64)>,
    pub endval: Vec<(usize, f64)>,
    pub varobs: Vec<String>,
    /// `M_.Sigma_e` from the `shocks` block (stderr^2, var, covariances, corr).
    pub sigma_e: Mat,
    /// `M_.H`: measurement-error covariance (nvobs x nvobs, `varobs` order) from the
    /// `shocks` block entries on observed endogenous variables.
    pub h: Mat,
    /// `M_.Correlation_matrix` / `M_.Correlation_matrix_ME`: unit diagonal plus the `corr`
    /// entries of the `shocks` block (covariances set with `var a, b = x` are not in it).
    pub corr_e: Mat,
    pub corr_h: Mat,
    /// `estimated_params` entries in file order (see [`ModFile::estimated_params_dynare_order`]).
    pub estimated_params: Vec<EstimatedParam>,
    pub estimated_params_init: Vec<EstimatedParamInit>,
    /// `estimated_params_init(use_calibration)`.
    pub estimated_params_init_use_calibration: bool,
    /// Commands in file order.
    pub commands: Vec<Command>,
    /// OccBin: the `occbin_constraints` block (see `crate::occbin`).
    pub occbin_constraints: Vec<crate::occbin::OccbinConstraint>,
    // ---- deterministic simulations (stream D) ----
    /// Exogenous values of the `initval` block (`oo_.exo_steady_state`), exo index, value.
    pub initval_exo: Vec<(usize, f64)>,
    /// Deterministic shocks (`M_.det_shocks`) from `shocks`/`mshocks` blocks, in file order.
    pub det_shocks: Vec<DetShock>,
    /// `histval` entries: endogenous (`true`) or exogenous, index, lag (<= 0), value.
    pub histval: Vec<(bool, usize, i32, f64)>,
    /// `steady_state_model` block: (lhs name, rhs in .mod syntax), in order.
    pub steady_state_model: Vec<(String, String)>,
    /// `predetermined_variables` (endogenous indices).
    pub predetermined_variables: Vec<usize>,
    /// Variables declared with `varexo_det` that stay ordinary exogenous variables (exogenous
    /// indices): only in optimal-policy models; otherwise they become `Model::exo_det_names`.
    pub exo_det: Vec<usize>,
    /// TeX and long names of the declared symbols (stream B, LaTeX output).
    pub labels: SymbolLabels,
    /// The model block as written (stream B, `write_latex_original_model`).
    pub original_model: OriginalModel,
    /// [stream P] `optim_weights`, `osr_params`, `osr_params_bounds`, `ramsey_constraints`.
    pub policy: PolicyDecl,
    /// [stream X] `observation_trends` block: (observed variable, trend coefficient as an
    /// expression of the parameters in .mod syntax), in block order.
    pub observation_trends: Vec<(String, String)>,
    /// [stream X] `homotopy_setup` block: (parameter or exogenous name, start value
    /// (`None` = the current value), end value).
    pub homotopy_setup: Vec<(String, Option<f64>, f64)>,
    /// Stream EP: `shocks(learnt_in=...)`, `endval(learnt_in=...)` (`M_.learnt_shocks`, `M_.learnt_endval`).
    pub ep: crate::ep::EpDecl,
    /// `epilogue`, `init2shocks`, `model_local_variable` labels.
    pub q4: crate::q4::Q4Decl,
    /// `shock_paths` blocks (`M_.shock_paths`).
    pub q1: crate::deterministic::shock_paths::Q1Decl,
    /// `heteroskedastic_shocks`, `filter_initial_state`, `M_.Skew_e`.
    pub q3: Q3Decl,
    /// model_info: `model(block, mfs=, static_mfs=, cutoff=, no_static)` options.
    pub block_opts: crate::model::blocks::ModelOptions,
    // ---- var models ----
    /// `var_model`, `trend_component_model`, `var_expectation_model` (see `crate::varmodels`).
    pub varmodels: crate::varmodels::VarModels,
    // ---- heterogeneity ----
    /// Heterogeneous-agent models: the `model(heterogeneity=...)` block and `SUM()` aggregates.
    pub het: crate::heterogeneity::HetDecl,
    /// The MATLAB variables of the `verbatim` blocks (e.g. the `initial_guess` structure of
    /// `heterogeneity_compute_steady_state(variable = initial_guess)`).
    pub verbatim_workspace: std::collections::HashMap<String, crate::matfile::MatValue>,
}

/// Type of a deterministic shock (`M_.det_shocks(i).type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetShockKind {
    Level,
    /// `mshocks`: value times the exogenous steady state.
    MultiplySteadyState,
    /// `mshocks(relative_to_initval)`.
    MultiplyInitialSteadyState,
}

/// One entry of `M_.det_shocks`: `var e; periods 1:3; values 0.1;`.
#[derive(Clone, Debug, PartialEq)]
pub struct DetShock {
    pub exo: usize,
    /// 1-based simulation periods (0 = initial condition, allowed when the model has lags).
    pub periods: Vec<i64>,
    pub value: f64,
    pub kind: DetShockKind,
    // ---- shocks/shock_paths ----
    /// `periods 2023Q1:2023Q4`: the periods are dates (`periods` is then empty); they map to
    /// simulation periods through `first_simulation_period` ([`DetShock::sim_periods`]).
    pub dates: Vec<crate::dates::Date>,
}

impl DetShock {
    /// The 1-based simulation periods (`M_.det_shocks(i).periods - first_simulation_period + 1`
    /// for dates, as make_ex_.m).
    pub fn sim_periods(&self, first_simulation_period: Option<crate::dates::Date>) -> anyhow::Result<Vec<i64>> {
        crate::deterministic::periods::to_sim_periods(&self.periods, &self.dates, first_simulation_period)
    }
}

/// The model block before any transformation: model-local variables (`# r = ...;`) and the
/// equations `(lhs, rhs, tags)` as parsed (`rhs` is `0` for equations without `=`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OriginalModel {
    pub locals: Vec<(String, Ast)>,
    pub equations: Vec<(Ast, Ast, Vec<(String, String)>)>,
}

/// Option key under which commands such as `bvar_density 4;` or
/// `model_comparison m1(0.7) m2(0.3);` keep their raw trailing tokens (an `OptValue::List` of
/// `OptValue::Str`).
pub const RAW_ARGS_KEY: &str = "__args";
/// one block entry (`irf_calibration` / `moment_calibration`), raw tokens joined
/// by blanks, under this option key in file order.
pub const BLOCK_BODY_KEY: &str = "__body";

/// `M_.endo_names_tex` / `M_.endo_names_long` & co. for the declared symbols, in declaration
/// order per kind. Defaults as Dynare: TeX name = name with `_` escaped, long name = name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SymbolLabels {
    pub endo_tex: Vec<String>,
    pub endo_long: Vec<String>,
    pub exo_tex: Vec<String>,
    pub exo_long: Vec<String>,
    pub param_tex: Vec<String>,
    pub param_long: Vec<String>,
    /// `M_.exo_det_names_tex` / `M_.exo_det_names_long` (`varexo_det`).
    pub exo_det_tex: Vec<String>,
    pub exo_det_long: Vec<String>,
}

impl SymbolLabels {
    pub fn push(&mut self, kind: crate::model::SymbolKind, tex: String, long: String) {
        use crate::model::SymbolKind as K;
        let (t, l) = match kind {
            K::Endo => (&mut self.endo_tex, &mut self.endo_long),
            K::Exo => (&mut self.exo_tex, &mut self.exo_long),
            K::Param => (&mut self.param_tex, &mut self.param_long),
            K::Local | K::Trend => return,
        };
        t.push(tex);
        l.push(long);
    }
}

/// [stream P] Optimal-policy declarations outside the commands (indices 0-based).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PolicyDecl {
    /// `optim_weights`: `(row, col, weight)` entries of `M_.osr.variable_weights`
    /// (`var w;` sets (i,i), `var1, var2 w;` sets (i,j) only), in file order.
    pub optim_weights: Vec<(usize, usize, f64)>,
    /// `osr_params` (parameter indices, file order).
    pub osr_params: Vec<usize>,
    /// `osr_params_bounds`: (parameter, lower, upper).
    pub osr_bounds: Vec<(usize, f64, f64)>,
    /// `ramsey_constraints` lines as written (enforced by `policy::ramsey`).
    pub ramsey_constraints: Vec<String>,
}

impl ModFile {
    /// The estimated parameters in Dynare's `xparam1` order (`set_prior.m`): `stderr` of
    /// exogenous shocks, `stderr` of measurement errors, `corr` of shocks, `corr` of
    /// measurement errors, then the structural parameters, each group in file order.
    pub fn estimated_params_dynare_order(&self) -> Vec<EstimatedParam> {
        let group = |e: &EstimatedParam| match e.kind {
            EstimatedKind::StdErr(_) => 0,
            EstimatedKind::MeStdErr(_) => 1,
            EstimatedKind::Corr(..) => 2,
            EstimatedKind::MeCorr(..) => 3,
            EstimatedKind::Skew(_) => 4,      // estim_params_.nsx, before np
            EstimatedKind::InitState(_) => 5, // ---- estimation completion ---- nendoinit
            EstimatedKind::Param(_) => 6,
        };
        let mut v = self.estimated_params.clone();
        v.sort_by_key(group); // stable: file order within a group
        v
    }

    /// Dynare's name of an estimated parameter (`bayestopt_.name`, `parameter_names` of a
    /// mode file): `stderr e`, `corr e1, e2` or the parameter name.
    pub fn estimated_param_name(&self, e: &EstimatedParam) -> String {
        let m = &self.model;
        match e.kind {
            EstimatedKind::Param(i) => m.param_names[i].clone(),
            EstimatedKind::StdErr(k) => format!("stderr {}", m.exo_names[k]),
            EstimatedKind::Corr(a, b) => format!("corr {}, {}", m.exo_names[a], m.exo_names[b]),
            EstimatedKind::MeStdErr(k) => format!("stderr {}", m.endo_names[k]),
            EstimatedKind::MeCorr(a, b) => format!("corr {}, {}", m.endo_names[a], m.endo_names[b]),
            EstimatedKind::Skew(k) => format!("skew {}", m.exo_names[k]),
            EstimatedKind::InitState(k) => format!("init {}", m.endo_names[k]), // ---- estimation completion
        }
    }

    /// Position in `varobs` of an observed endogenous variable (the index into `M_.H`).
    pub fn varobs_position(&self, endo: usize) -> Option<usize> {
        let name = &self.model.endo_names[endo];
        self.varobs.iter().position(|v| v == name)
    }
}

/// A command (statement with options) of the .mod file.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    Steady(Options),
    Check(Options),
    StochSimul(Options, Vec<String>),
    /// Options + variable list.
    Estimation(Options, Vec<String>),
    // ---- stream F ----
    /// `forecast(options) varlist;`
    Forecast(Options, Vec<String>),
    /// `shock_decomposition(options) varlist;`
    ShockDecomposition(Options, Vec<String>),
    /// `calib_smoother(options) varlist;`
    CalibSmoother(Options, Vec<String>),
    /// `shock_groups(name=NAME); 'label' = shk1, shk2; ... end;`: name, (label, shocks).
    ShockGroups(String, Vec<(String, Vec<String>)>),
    // ---- end stream F ----
    /// [stream P] `ramsey_model`, `ramsey_policy`, `discretionary_policy`, `osr`,
    /// `evaluate_planner_objective`: name, options, variable list.
    Policy(String, Options, Vec<String>),
    // ---- stream C ----
    /// `conditional_forecast_paths; var v; periods ...; values ...; end;`: per constrained
    /// variable (in block order), its (first period, last period, value) ranges, 1-based.
    ConditionalForecastPaths(Vec<(String, Vec<(usize, usize, f64)>)>),
    /// `conditional_forecast(options);`
    ConditionalForecast(Options),
    /// `plot_conditional_forecast(periods=) varlist;`
    PlotConditionalForecast(Options, Vec<String>),
    /// `realtime_shock_decomposition(options) varlist;`
    RealtimeShockDecomposition(Options, Vec<String>),
    /// `plot_shock_decomposition(options) varlist;`
    PlotShockDecomposition(Options, Vec<String>),
    /// `initial_condition_decomposition(options) varlist;`
    InitialConditionDecomposition(Options, Vec<String>),
    /// `squeeze_shock_decomposition varlist;`
    SqueezeShockDecomposition(Vec<String>),
    // ---- end stream C ----
    /// Parsed but unsupported: name kept for a warning.
    Other(String, Options),
    /// OccBin commands and `shocks(surprise)` blocks (see `crate::occbin`).
    Occbin(crate::occbin::OccbinCommand),
    /// `endval` block at its position (stream D): (variable name, value expression in .mod
    /// syntax, evaluated when the block is reached, as Dynare's driver does).
    Endval(Vec<(String, String)>),
    /// Deterministic-simulation commands (stream D): `perfect_foresight_setup`,
    /// `perfect_foresight_solver`, `simul`, `rplot`, `model_diagnostics`: name, options, var list.
    Deterministic(String, Options, Vec<String>),
    /// MoM: `matched_moments`, `matched_irfs[_weights]` blocks and `method_of_moments`
    /// (see `crate::mom`).
    Mom(crate::mom::MomCommand),
}

/// Command options: keys lower-case, in file order.
pub type Options = Vec<(String, OptValue)>;

#[derive(Clone, Debug, PartialEq)]
pub enum OptValue {
    Flag,
    Num(f64),
    Str(String),
    List(Vec<OptValue>),
    Vec(Vec<f64>),
}

/// Looks up an option by (lower-case) key.
pub fn opt<'a>(opts: &'a Options, key: &str) -> Option<&'a OptValue> {
    opts.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstimatedKind {
    Param(usize),
    /// Standard error of an exogenous shock.
    StdErr(usize),
    /// Correlation of two exogenous shocks.
    Corr(usize, usize),
    /// Standard error of the measurement error on an observed endogenous variable
    /// (endogenous index; see [`ModFile::varobs_position`] for its row of `M_.H`).
    MeStdErr(usize),
    /// Correlation of two measurement errors (endogenous indices).
    MeCorr(usize, usize),
    /// Skewness coefficient of an exogenous shock (`skew e`, `M_.Skew_e(k,k,k)`).
    Skew(usize),
    /// ---- estimation completion ---- `estimate_initial_states_endogenous_prior`: the initial
    /// value of a state variable (endogenous index), `M_.endo_initial_state.values(k)`,
    /// `estim_params_.endo_init_vals` (ordered sixth in `xparam1`, before the parameters).
    InitState(usize),
}

/// Prior shapes; Dynare's `pshape` codes are 1 beta, 2 gamma, 3 normal, 4 inv_gamma(1),
/// 5 uniform, 6 inv_gamma2, 8 weibull (see [`PriorShape::dynare_code`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriorShape {
    Beta,
    Gamma,
    Normal,
    InvGamma1,
    Uniform,
    InvGamma2,
    Weibull,
}

impl PriorShape {
    pub fn from_keyword(s: &str) -> Option<Self> {
        Some(match s {
            "beta_pdf" => Self::Beta,
            "gamma_pdf" => Self::Gamma,
            "normal_pdf" => Self::Normal,
            "inv_gamma_pdf" | "inv_gamma1_pdf" => Self::InvGamma1,
            "uniform_pdf" => Self::Uniform,
            "inv_gamma2_pdf" => Self::InvGamma2,
            "weibull_pdf" => Self::Weibull,
            _ => return None,
        })
    }

    pub fn dynare_code(self) -> u8 {
        match self {
            Self::Beta => 1,
            Self::Gamma => 2,
            Self::Normal => 3,
            Self::InvGamma1 => 4,
            Self::Uniform => 5,
            Self::InvGamma2 => 6,
            Self::Weibull => 8,
        }
    }
}

/// One `estimated_params` entry: `NAME, INIT [, LB, UB], SHAPE, MEAN, SD [, P3, P4, JSCALE];`.
#[derive(Clone, Debug, PartialEq)]
pub struct EstimatedParam {
    pub kind: EstimatedKind,
    /// 2nd field (initial value), `None` if empty.
    pub init: Option<f64>,
    /// Explicit bounds.
    pub lb: Option<f64>,
    pub ub: Option<f64>,
    /// `None` = maximum likelihood (no prior).
    pub prior: Option<PriorShape>,
    /// Prior mean / standard deviation (NaN if empty, e.g. `uniform_pdf, , , 0, 1`).
    pub mean: f64,
    pub sd: f64,
    /// 3rd/4th prior parameters (e.g. generalized beta support, uniform bounds).
    pub p3: Option<f64>,
    pub p4: Option<f64>,
    pub jscale: Option<f64>,
}

/// One `estimated_params_init` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct EstimatedParamInit {
    pub kind: EstimatedKind,
    pub value: f64,
}
