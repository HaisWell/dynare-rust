# Supported features in detail

What dynare-rust implements, area by area: the commands and options, the Dynare functions each part
ports, the tests and references that validate it, the accuracy reached against Dynare 7.2 and the
known differences. [COVERAGE.md](COVERAGE.md) has the status of every command and option;
[REFERENCES.md](REFERENCES.md) describes the public reference models (`reference/sw07`, a
Smets-Wouters 2007 variant with simulated data, `reference/qpm`, `reference/gm05`) and the known
differences in detail. Unless stated otherwise, "Dynare" below means the Dynare 7.2 output stored
under `reference/`.

## Preprocessor

- Macro language (checked against `dynare-preprocessor ... savemacro`): `@#define` (variables,
  `@#define NAME`, macro functions `f(x) = ...`), `@#if/@#ifdef/@#ifndef/@#elseif/@#else/@#endif`,
  `@#include`, `@#includepath`, `@#for` (tuples, `when`), `@#echo`, `@#error`, `@#echomacrovars`
  (also `(save)`: `Output/<fname>_macrovars.json`), `@{}`; expressions with tuples,
  comprehensions, set operations (`| & - * ^`), slices, `in`, casts and the manual's function list.
  Command line: `-D`, `-I`, `savemacro[=FILE]`, `linemacro`, `onlymacro`, `nograph`, `onlymodel`.
- Statements run in file order: parameter assignments and `set_param_value`,
  `set_shock_stderr_value`, `set_shock_skew_value` after a command change the parameters there;
  `options_.X = V` sets a default for later commands; command options persist in a global
  `options_` as in Dynare. `load_params_and_steady_state` (read at preprocessing, as Dynare does),
  `estimated_params(overwrite)`, `estimated_params_bounds`, `estimated_params_remove`,
  `observation_trends`, `homotopy_setup`.
- Declarations `var`, `varexo`, `varexo_det`, `parameters`, `predetermined_variables`, with
  `$tex$` names and `(long_name='...')`; mod-file local variables (`phi = 0.1;` without a
  declaration); `var(log)` (`LOG_x` auxiliary variables); nonstationary models with
  `trend_var(growth_factor=...)`, `log_trend_var(log_growth_factor=...)`, `var(deflator=...)`,
  `var(log_deflator=...)` (detrending and the balanced-growth test with
  `model(balanced_growth_test_tol=...)`, Dynare's error message); complementarity conditions
  `eq ⟂ x > lb;`, `_|_` and the obsolete `[mcp='...']` tag.
- `model`: several blocks are concatenated; `#` locals (also with leads/lags), equation tags,
  `[static]`/`[dynamic]` equation pairs, `model_local_variable` (with TeX names),
  `model_options`, `model_remove`, `model_replace`, `var_remove`,
  `model(linear, differentiate_forward_vars[=(...)])`. `block`, `bytecode`, `use_dll`, `mfs`,
  `no_static`, ... are accepted; they only select Dynare's solving/compilation strategies, and the
  model is always solved with dynare-rust's compiled tapes and sparse solvers.
- The transform pass (`src/model/dtree.rs`) emulates the preprocessor's expression tree (node
  creation order, simplification rules, constant folding, operand ordering), so auxiliary variables
  of every type (`AUX_ENDO_LEAD_*`, `AUX_EXO_LEAD_*`, `AUX_ENDO_LAG_*`, `AUX_EXO_LAG_*`,
  `AUX_EXPECT_*`, `AUX_DIFF_*`, ...), their equations, `M_.aux_vars(k).orig_expr`, default
  equation names and the constant-equation substitution are those of Dynare 7.2.
- Operators: `+ - * / ^`, comparisons, `exp log ln log10 sqrt cbrt abs sign sin cos tan asin acos
  atan sinh cosh tanh asinh acosh atanh erf erfc max min normcdf normpdf` (case-insensitive), with
  Dynare's derivative conventions (`abs' = sign`, `sign' = 0`, `max`/`min` take the second
  argument's derivative at a tie); `EXPECTATION(k)(...)`, `diff(...)`; `STEADY_STATE(expr)`
  (supported in `steady`, `stoch_simul` orders 1-3, estimation, perfect foresight, extended path
  and OccBin; not in `identification`).
- `initval`, `endval`, `histval`, `steady`, `steady_state_model` (closed-form steady state, with
  temporary variables; used by the estimation and `stoch_simul`).
- `shocks` (stderr/var/corr, deterministic `periods`/`values`, `surprise` for OccBin), `mshocks`
  (incl. `relative_to_initval`), `varobs`, `estimated_params` (all prior shapes, explicit bounds,
  stderr/corr of shocks and measurement errors; no priors = maximum likelihood),
  `estimated_params_init` (incl. `use_calibration`), `shock_groups`,
  `conditional_forecast_paths`, `occbin_constraints`, `planner_objective`, `optim_weights`,
  `osr_params`, `osr_params_bounds`, `ramsey_constraints`.
- `varexo_det` (`reference/exo_det/`, `tests/exo_det.rs`): handled as Dynare in the steady state,
  `check`, `stoch_simul`, estimation, smoother, decompositions, JSON and LaTeX; their paths enter
  the forecasts (`simult_varexo_det.m`, orders 1-2, 1e-9). Dynare's errors are reproduced (leads or
  lags on them, perfect foresight, order 3, `pruning` forecasts, `extended_path`, ...).
- Limitations: multi-output assignments `[a, b] = f(...)` in `steady_state_model` and MATLAB
  steady-state files `<fname>_steadystate.m` cannot run; both stop with an explicit error naming
  the alternative.

## Data, dates, JSON and command-line options

`src/dates.rs`, `src/dataset.rs`, `tests/q4.rs`, `reference/q4/`.

- `dates` (port of dseries' `@dates`): annual, bi-annual, quarterly, monthly, weekly (ISO) and
  daily dates with Dynare's labels (`2004Q1`, `2004S2`, `2020W53`, `2003-03-31`), arithmetic and
  ranges; .mod literals with shifts (`2023Q1+6`).
- Data (port of `makedataset.m`): .csv, .xls/.xlsx, .mat or .m datafiles, the `data(...)` command,
  `set_time`, `dsample`. As in dseries only the first date of a datafile counts (a csv with ISO
  dates is a daily dataset, reproduced wherever Dynare shows dates; CSV outputs keep the labels).
- Command line (also on the first line of the .mod: `// --+ options: json=compute +--`):
  `json=parse|check|transform|compute` (model, aux variables, syntax tree, declarations and
  Jacobian sparsity as the preprocessor writes them; derivative strings without temporary terms,
  and only part of the statements), `jsonstdout`, `onlyjson`, `jsonderivsimple`, `exclude_eqs=` /
  `include_eqs=`, `stochastic`, `nostrict`, `params_derivs_order`; MATLAB-specific flags accepted.
- `external_function(name=, nargs=, first_deriv_provided[=], second_deriv_provided[=])`: a Rust
  closure (`dynare::model::external::register_external_function[_with_derivatives]`) or a simple
  MATLAB function file `NAME.m` (scalar arithmetic, `if`, matrix literals; anything else is refused
  with file and line). Order 2 matches Dynare to ~1e-14 in all three derivative settings.

## Steady state, `check`, small commands

`src/misc/`, `tests/misc_x.rs`, `reference/misc/`.

- `steady` with `solve_algo` 0/3/6/7/8/13/14 (3 = `csolve.m`, 6 = `newton_solve.m`, 13/14 = exact
  port of the `block_trust_region` MEX; 0/7/8 by the closest ported solver), 10 (`lmmcp`);
  `homotopy_mode=1|2|3`, `homotopy_steps` with `homotopy_setup` (1e-10).
- `check` (eigenvalue table and Blanchard–Kahn message, `qz_zero_threshold`), `resid`,
  `model_info`, `model_diagnostics` (full port, identical output), `prior table|plot|simulate|
  moments|optimize` (`moments(distribution)` / `irfs(distribution)`: the implied prior moments of
  `print_moments_implied_prior.m` within Monte Carlo error; `optimize`: `optimize_prior.m`),
  `save_params_and_steady_state`, `smoother2histval`, `trace_plot`, `set_dynare_seed`,
  `dynare_version`. MATLAB-only commands (bytecode printing, workspace transfers,
  `compilation_setup`) are accepted with a note.
- Checked against Dynare: `lik_init` 2/5, `observation_trends` (+ `prefilter`), `.m` datafiles
  (fval 1e-12, smoother 2e-15), `estimated_params_bounds/remove`, the `resid` / `model_info` /
  `prior table` text, the `save_params_and_steady_state` file, macro expansion.

## Block decomposition

`src/model/blocks.rs`, `src/model/blocks_graph.rs`, `tests/model_info_blocks.rs`, `reference/blocks/`.

Port of the preprocessor's `computingPassBlock` (normalization by Boost's matching, prologue and
epilogue, dependency graph with `mfs` 0-3, block types, `is_linear`) for the static and dynamic
models. `dynare-blocks file.mod` prints `M_.block_structure[_stat]` as JSON;
`model_info(block_static|block_dynamic[, incidence])` prints Dynare's tables. Block structures are
identical to the preprocessor's on 416 models (reference models, Dynare's test suite and examples,
`model(block, mfs=, static_mfs=, cutoff=)` variants); the `model_info` text is identical on 18.

## Estimation

- Likelihood: Kalman filters (multivariate, univariate, diffuse, missing data), `kalman_algo`,
  `lik_init` 1-5 (4: Riccati steady state; Dynare 7.2's arm64 MEX for it fails to load and falls
  back to 1), `diffuse_filter`, `observation_trends`, `prefilter`, `loglinear`, `logdata`,
  `presample`, `first_obs`, `nobs`, `prior_trunc`, `endogenous_prior`, tolerances. Measurement
  errors (`M_.H`, stderr/var/corr in `shocks` and `estimated_params`; non-diagonal H in the
  univariate filters and smoother by state augmentation, as Dynare). When a stderr or corr is
  estimated, `Sigma_e` (and `H`) is rebuilt as `set_all_parameters.m` does, so covariances set with
  `var a, b = x` are zeroed and only `corr` entries survive.
- `mode_compute` (`src/estimation/optim/`, dispatched as `dynare_minimize_objective.m`): 0;
  2 simulated annealing; 4 csminwel; 5 newrat; 6 gmhmaxlik; 7 `fminsearch`; 8 Dynare's simplex;
  9 CMA-ES (population evaluated in parallel); 10 `simpsa.m`; 101 `solvopt.m`;
  `additional_optimizer_steps`, `optim` (every optimizer's keys), `silent_optimizer`, `mode_file`
  (handled as `check_mode_file.m`). The deterministic ones (4, 5, 7, 8, 101) reproduce MATLAB's
  iterates and evaluation counts on the reference problems (`reference/estimation/`).
  MATLAB-toolbox optimizers are replaced by substitutes that print a note and are **not** MATLAB's
  algorithms: 1 (`fmincon`) box-projected BFGS, 3 (`fminunc`) BFGS, 12 (`particleswarm`) particle
  swarm as MathWorks documents it, 102 (`simulannealbnd`) its documented default algorithm. 13
  (`lsqnonlin`) is for `method_of_moments` only, as in Dynare 7.2.
- On SW07 from the initial values (minus log posterior 374.505), `mode_compute` 1, 2, 3, 4, 5, 6,
  8, 101 and 9 (initial step scaled by 0.05) reach Dynare's newrat mode 349.8561 within 4e-4
  (2 and 6 end 7e-6 below it along a flat direction); 12 stops 2e-3 above; 10 stays at 366.5 with
  its default budget; 9 with Dynare's default step stops at once because its first generation is
  infeasible, as Dynare's `cmaes.m` does.
- Without priors the estimation is maximum likelihood (ML table, no MCMC); `mode_compute=0`
  without a mode file only runs the smoother.
- The Hessian at the mode as `dynare_estimation_1.m` computes it: `cova_compute = 0`,
  `use_penalized_objective_for_hessian`, newrat's `optim = ('Hessian', 0 | 2)` (`mr_hessian.m`;
  1e-11 / 3e-7 at Dynare's mode), `check_hessian_at_the_mode.m`.
- `analytic_derivation` (analytic score of `dsge_likelihood`, `analytic_Hessian = 'opg' | 'full'
  | 'asymptotic'`, `analytic_derivation_mode` 0, 1, -1, -2, Dynare's errors), used by csminwel,
  solvopt, newrat and the BFGS substitutes. Gradients and Hessians agree to 1e-12, csminwel/solvopt
  modes to 1e-9; newrat drifts through rounding noise (fval within 5e-7) (`tests/est_complete_ad.rs`).
- Further options (`reference/est_complete/`, `tests/est_complete_options.rs`,
  `tests/estimation_q3.rs`; objective values 1e-12 relative, smoothed objects 1e-14):
  `conditional_likelihood`, `rescale_prediction_error_covariance`, `lyapunov = fixed_point |
  doubling | square_root_solver` (the latter is the Schur solution), `filter_initial_state`,
  `heteroskedastic_shocks` + `heteroskedastic_filter`, `no_init_estimation_check_first_obs`,
  `frequentist_smoother`, `use_univariate_smoother_if_singularity_is_detected`, `dr=` and
  `qz_zero_threshold` in the likelihood, `huge_number`, `nonlinear_filter_initialization`,
  `consider_all_endogenous[_and_auxiliary]`, `consider_only_observed`, `dirname`.
- DSGE-VAR (`dsge_var_likelihood.m`): `dsge_var = w` (also `Inf`), `dsge_prior_weight` in
  `estimated_params`, `dsge_varlag`; `oo_.dsge_var.posterior_mode` in the results; with
  `bayesian_irf` the DSGE-VAR posterior IRFs (`<fname>_Bayesian_IRF_BVAR-DSGE_<shock>.csv`).
- Skewed shocks (`skew` in `shocks` and `estimated_params`, `kalman_algo = 5`: pruned skewed
  Kalman filter and smoother with all `skewed_kalman_*` options): likelihoods and smoothed objects
  to 1e-15; with measurement errors the smoother's pruning depends on rounding noise in Dynare's
  own algorithm and agrees to about 3e-3 (`tests/est_complete_skewed.rs`).
- Generalized priors (third/fourth prior parameters of beta, gamma, Weibull, inverse gammas,
  bounds on a normal) as Dynare.
- Initial states (`estimate_initial_states_endogenous_prior`, slice sampler only, as in Dynare):
  likelihood, projection and smoothed initial state to 1e-10, posterior means statistically
  (`tests/est_complete_init_states.rs`); `draw_init_state_with_rotated_slice` is not ported.
- Results: `<fname>_mode.mat`, `<fname>_results.json` (mode, Laplace log data density, MHM log data
  density after an MCMC, posterior moments), `<fname>_posterior.csv`, smoother CSVs; with `TeX` the
  `<fname>_Posterior_Mode_k.tex` / `_Posterior_Mean_k.tex` tables.
- `model_comparison[(marginal_density = laplace|modifiedharmonicmean)] M1[(PRIOR)] M2 ...;` reads
  `Mi/Output/Mi_results.json` and writes `<fname>_model_comparison.json` (+ TeX); it also uses the
  marginal densities of DSMH and hssmc runs.

### Posterior samplers and MCMC

- `posterior_sampling_method`: `random_walk_metropolis_hastings` (Gaussian or Student proposal;
  the default runs a pre-fetching chain whose draws are bit-identical to a sequential chain),
  `independent_metropolis_hastings`, `slice` (`rotated`, `mode_files`, `slice_initialize_with_mode`,
  `initial_step_size`, `use_prior_draws`), `tailored_random_block_metropolis_hastings`,
  `dsmh` (dynamic striated Metropolis–Hastings), and the SMC samplers `hssmc` (`particles`,
  `steps`, `lambda`, `scale`, `target`), `dime_mcmc` (`nchain`, `niter`, `tune`, `aimh_prob`,
  `sigma`, `gamma`, `df_proposal_dist`, `rho`) and `online` (Liu–West filter; order 2/3 only).
  `posterior_sampler_options` keys as `check_posterior_sampler_options.m`. Chains and particles run
  in parallel and are reproducible with `--seed`. Every sampler also runs in the nonlinear
  (particle filter) estimation.
- Options: `mh_replic`, `mh_nblocks`, `mh_jscale`, `mh_tune_jscale`, `mh_tune_guess`, `mh_drop`,
  `mh_conf_sig`, `mh_init_scale_factor`, `mcmc_jumping_covariance`, `sub_draws`,
  `posterior_max_subsample_draws`, `nodiagnostic`, `taper_steps`, `geweke_interval`,
  `raftery_lewis_diagnostics` + `raftery_lewis_qrs`, `brooks_gelman_plotrows`, posterior kernel
  densities (`no_posterior_kernel_density`), `contemporaneous_correlation`. `mh_thin` does not
  exist in Dynare 7.2.
- MCMC files in Dynare's format (`<fname>/metropolis/*_mh<k>_blck<b>.mat`, history files,
  `metropolis.log`): `load_mh_file`, `mh_recover`, `load_results_after_load_mh`,
  `use_mh_covariance_matrix`, `mh_initialize_from_previous_mcmc*`, `mh_posterior_mode_estimation`.
  3000 + 2000 draws or a recovered crash equal one 5000-draw run bit for bit
  (`tests/mcmc_files.rs`). Dynare reads and continues dynare-rust's chains and vice versa, with
  posterior statistics equal to 3e-15 and the MHM density to 1e-12 (`reference/mcmc/`).
- On the NK test model hssmc and DIME agree with a 2 x 20000-draw RW MCMC within 3 posterior sd
  and hssmc's log marginal density is within 0.3 of the MHM (hssmc needs many particles in high
  dimensions); on `rbcnl` the online filter's final bands agree with Dynare's own run.
- Posterior analysis: `bayesian_irf`, `moments_varendo` (with `irf`, `irf_shocks`,
  `relative_irf`, `ar`, `conditional_variance_decomposition`, `nodecomposition`): posterior
  mean/median/variance/HPD/deciles of IRFs, theoretical moments and variance decompositions from
  `sub_draws` re-solved in parallel. `smoother`, `filtered_vars`, `forecast` after an MCMC: the
  posterior distributions of `prior_posterior_statistics.m` / `pm3.m` (smoothed variables and
  shocks, updated and filtered variables, `MeanForecast`, `PointForecast`).

## Forecasts, smoothers and decompositions

- Smoother outputs: `smoother`, `filtered_vars`, `smoothed_state_uncertainty`,
  `filter_step_ahead`, `filter_covariance`, `filter_decomposition` (multivariate and univariate);
  `smoother_redux` and `selected_variables_only` are accepted (same results).
- `forecast=H` after an estimation (point forecast from the last smoothed state with `conf_sig`
  bands, `<fname>_forecast.csv`); the `forecast(periods=, conf_sig=) vars;` command from the steady
  state (or `histval`) at the current parameters and `options_.order` (orders 2-3: mean without
  shocks, bands from 1000 simulations; identical on MATLAB's normals). Bands with measurement
  errors (`lower_me`/`upper_me`).
- `calib_smoother(datafile=, first_obs=, ...) vars;`. As in Dynare, `calibration` means the current
  `M_.params`, i.e. the estimation's point after an estimation.
- `shock_decomposition(parameter_set=, init_state=, use_shock_groups=,
  forecast_type=unconditional|conditional, with_epilogue) vars;`,
  `realtime_shock_decomposition(forecast=, save_realtime=, fast_realtime=, presample=,
  parameter_set=, with_epilogue)` (vintages smoothed in parallel), `initial_condition_decomposition`,
  `squeeze_shock_decomposition`, `plot_shock_decomposition(realtime=0..3, vintage=,
  type=qoq|yoy|aoa, fig_name=, use_shock_groups=, flip, diff, detail_plot, max_nrows=,
  plot_init_date=, plot_end_date=, init2shocks[=NAME], steadystate, write_xls, colormap=,
  screen_shocks)`; each plot also writes its decomposition as CSV; `write_xls` writes a real .xlsx
  (`WriteShockDecomp2Excel.m` layout); `interactive` is accepted with a note.
  Differences: Dynare 7.2 stops on `forecast_type=conditional` unless every shock is controlled,
  dynare-rust uses the uncontrolled shocks' zero mean; for posterior parameter sets Dynare uses
  `oo_.MeanForecast`, dynare-rust `oo_.forecast` (with a warning); a standalone plot command has no
  initial date, as in Dynare.
- Conditional forecasts: `conditional_forecast_paths`,
  `conditional_forecast(parameter_set=calibration|prior_mode|prior_mean|posterior_mode|
  posterior_mean|posterior_median, controlled_varexo=, periods=, replic=, conf_sig=)` and
  `plot_conditional_forecast` (`+conditional_forecasts/run.m`): conditional and unconditional
  means, bands and zero-shock paths, controlled shocks.
- `epilogue` block (with dseries' rounding to 6 decimals), epilogue variables in the decompositions
  (`epilogue_shock_decomposition.m`); after `estimation(smoother)` and perfect foresight the
  epilogue is also evaluated over the smoothed variables / `oo_.endo_simul`.

## Stochastic simulation and analysis

- `stoch_simul` at order 1 (`src/analysis`), printed like Dynare (model summary, policy and
  transition functions, theoretical or simulated moments): `irf`, `irf_shocks`, `relative_irf`,
  `irf_plot_threshold`, `nograph`, `graph_format`, `noprint`/`print`, `nofunctions`, `nomoments`,
  `nocorr`, `nodecomposition`, `nomodelsummary`, `ar`, `periods`, `drop`, `simul_replic`,
  `conditional_variance_decomposition`, `qz_criterium`, `qz_zero_threshold`, `dr_display_tol`,
  `contemporaneous_correlation`, `histval` as the start of simulations, measurement errors (the
  covariance table and decompositions WITH MEASUREMENT ERROR), the `tex` tables. With unit roots
  the theoretical moments keep the stationary variables only, as `th_autocovariances.m` decides.
  After an `estimation` the command runs at its mode or posterior mean. Results in
  `<fname>_stoch_simul.json`, IRF CSVs and figures.
- Filters: `hp_filter` (theoretical: Uhlig's spectral method), `one_sided_hp_filter`,
  `bandpass_filter`, `spectral_density`; `loglinear` (order 1), `dr=cycle_reduction|
  logarithmic_reduction` with their tolerances, `aim_solver` / `dr=aim` (Anderson-Moore),
  `partial_information` (`dr1_PI.m`, `PI_gensys.m`, `PCL_Part_info_*.m`; order forced to 1).
  Checked on `reference/q2/rbc_q2.mod` and `reference/stoch_simul2/`: decision rules, IRFs,
  theoretical and simulated moments (MATLAB's shocks), spectral densities to 1e-9 or better, cycle
  and logarithmic reduction to 1e-7, printed tables and TeX files identical character for character.
- Orders 2 and 3 (`src/solver/korder/`): `ghxx ghxu ghuu ghs2` and `ghxxx ... ghuss`, the k-order
  tensors `g_0..g_3` (Dynare's folded layout), `k_order_solver`, `pruning`, second-order mean
  correction, pruned-state-space theoretical moments and variance decompositions, `periods`
  simulations (`simult_`, `k_order_simul`), generalized IRFs (`irf`, `replic`, `drop`,
  `relative_irf`), simulated moments with every filter. As the preprocessor does, `stoch_simul`
  without `order` runs at order 2; models with a zero Hessian fall back to order 1.
- Dynare's refusals are reproduced: `loglinear`, `aim_solver` at order > 1;
  `dr=cycle_reduction|logarithmic_reduction` with `k_order_solver`; `solve_algo` 10/11; the
  known failures of `partial_information` with `noprint`, `nocorr` + `ar > 0`, `spectral_density`.
- Skewed shocks in `stoch_simul`: the `check_model.m` bound, the COSKEWNESS tables and
  closed-skew-normal draws for `periods > 0`; Dynare uses `Skew_e` nowhere else.
- Simulations draw from dynare-rust's ChaCha8 stream (`--seed`), not MATLAB's `randn`; given
  MATLAB's draws the results are Dynare's.
- Reporting (`src/reporting.rs`): dseries from `send_irfs_to_workspace` and `report()` objects
  (`addPage`, `addSection`, `addGraph`, `addTable`, `addSeries`, `addVspace`, `write`,
  `compile`) produce `report.tex` with the layout of Dynare's reporting classes; most formatting
  options are accepted and ignored.

## Identification

`src/identification/`, port of `+identification`; `reference/identification/`.

- Order 1: rank and collinearity checks of the Jacobians of the reduced form (Tau), the first two
  moments (J, Iskrev 2010), the spectrum (G, Qu and Tkachenko 2012), the minimal system (Komunjer
  and Ng 2011) and the LRE model; analytic parameter derivatives (`analytic_derivation_mode` 0;
  -1/-2 finite differences); identification strength from the asymptotic information matrix (or
  simulated moments). Options `parameter_set` (incl. `posterior_mode|mean|median` after an
  estimation), `ar`, `useautocorr`, `prior_mc`, `prior_range`, `advanced`, `max_dim_cova_group`,
  `periods`, `replic`, `grid_nbr`, `tol_rank`, `tol_deriv`, `tol_sv`, `normalize_jacobians`,
  `no_identification_*`, `diffuse_filter`, `lik_init`, `kalman_tol`, `prior_trunc`,
  `checks_via_subsets` + `max_dim_subsets_groups`, `gsa_sample_file`, `load_ident_files`.
- Orders 2 and 3 on the pruned state space (Mutschler 2015), derivatives by Dynare's two-sided
  finite differences (within 1e-6 of Dynare's).
- Same console output as Dynare; results in `<fname>/identification/<fname>_identif.json`, figures.
  On `nk_ident.mod` all Jacobians agree to 2e-15 and the asymptotic Hessian to 3e-14; on SW07
  (observables loading on its unit roots dropped, `diffuse_filter`) J agrees to 1e-13 and the
  printed diagnosis is identical, in 0.2 s against Dynare's 2 s. The simulation-based numbers
  (strength sample, `prior_mc`) agree statistically only.

## Global sensitivity analysis

`sensitivity` / `dynare_sensitivity` (`src/gsa/`, `reference/gsa/`, `reference/gsa2/`,
`tests/gsa_nk.rs`, `tests/gsa_sw07.rs`, `tests/gsa2.rs`).

- Stability mapping (`stab`, `Nsam`, `ilptau`, `pprior`, `prior_range`, `neighborhood_width`,
  `load_stab`, `alpha2_stab`, `pvalue_ks`, `pvalue_corr`), reduced-form mapping (`redform`,
  `namendo`, `namlagendo`, `namexo`, `threshold_redform`, `ksstat_redform`, `alpha2_redform`,
  `logtrans_redform`, `load_redform`), Morris screening (`morris=1|2`, `morris_nliv`,
  `morris_ntra`), RMSE mapping (`rmse`, `lik_only`, `var_rmse`, `pfilt_rmse`, `istart_rmse`,
  `alpha_rmse`, `alpha2_rmse`, `load_rmse`, `mode_file`), `identification=1`
  (`map_identification.m`), `ppost` (posterior subsample from the MCMC files), `glue`,
  `irf_calibration` / `moment_calibration` mapping (`map_calibration.m`). No option is unported.
- Outputs in `<fname>/gsa/` (`<fname>_prior.mat` with Dynare's variable names, figures).

| piece | agreement with Dynare |
|---|---|
| LPTAU/Sobol, Latin hypercube, Morris trajectories | bit for bit / same draws (MATLAB's `mt19937ar`, seed 0) |
| sample in the prior ranges; classification of every point | 1e-10 or exact; identical |
| Smirnov statistics, p-values, printed tables | 1e-14 / 1e-12, identical text |
| reduced-form coefficients / Morris elementary effects | 1e-9 / 1e-12 |
| log posterior / likelihood / RMSE / R2 per draw | 1e-8 / 1e-8 / 1e-6 (worst draw) |
| calibration restrictions, `map_identification` | restriction values 1e-7, `SAvdec`/`SAcc`/`SAnorm` 1e-6 |
| SW07, 2048 draws: classification; RMSE / R2 of the 245 stable draws | identical except 58 draws with a non-finite initial steady-state residual (info 22 here, 20 in Dynare) / 6e-13 median |
| SW07 log posterior per draw | 243 of 245 within 1e-6; at the other 2 Dynare's own value moves by ~1 log point under 1e-14 relative perturbations (badly conditioned F) |

Differences: without `threshold_redform` Dynare's reduced-form mapping needs the external
SS-ANOVA-R toolbox; dynare-rust prints first-order sensitivity indices from a cubic regression on
each parameter instead. Dynare 7.2 fails on `lik_only=1` and on `redform`/`morris` without TeX;
dynare-rust runs both. Dynare 7.2's preprocessor has no `glue` option; dynare-rust reads `glue=1`.
If earlier commands of the same .mod draw MATLAB random numbers, Dynare's GSA sample differs.

## Calibration restrictions in the estimation

`irf_calibration` and `moment_calibration` (`src/gsa/calibration.rs`, `tests/calib_est.rs`,
`reference/calib_est/`) are applied wherever Dynare 7.2 calls `endogenous_prior_restrictions.m`:
`dsge_likelihood.m` (incl. conditional-likelihood, OccBin and skewed branches: info 49 and the
penalty `info(4)`, seen by the mode finders, `mode_check` and the MCMC), `prior simulate`,
`prior moments`, `prior moments/irfs(distribution)`, the GSA stability mapping, and
`method_of_moments` (dropped with Dynare's warning). As in Dynare they are not used by the
nonlinear likelihood, DSGE-VAR or identification.

| check | agreement |
|---|---|
| `dsge_likelihood` at 46 points: fval / info / `info(4)` | 1e-12 / identical / 1e-10 |
| restriction values and per-restriction penalties | 1e-11 / 1e-10 |
| mode with a restriction binding (`mode_compute=8`), printout incl. "OPTIMIZATION PROBLEM!" | identical |
| mode + MCMC (2 x 2000) with binding restrictions | mode 1e-4, no violating draw, means within one sd |
| `prior simulate` violation share | statistically (0.5393 vs 0.54335) |
| GSA stability mapping with restrictions | identical |

With `mode_compute=4`, csminwel follows Dynare until its first random "cliff" perturbation.

## Method of moments

`method_of_moments` (`src/mom/`, port of `+mom/run.m`; `tests/mom.rs`, `tests/mom_analytic.rs`,
`reference/mom/`).

- `matched_moments`, `matched_irfs[(overwrite)]`, `matched_irfs_weights`;
  `mom_method = GMM | SMM | IRF_MATCHING` with `datafile`, `first_obs`, `nobs`, `order` (GMM 1-3
  on the pruned state space), `pruning`, `prefilter`, `hp_filter`, `one_sided_hp_filter`,
  `bandpass_filter` (SMM only, as Dynare), `weighting_matrix` (`IDENTITY_MATRIX`, `DIAGONAL`,
  `OPTIMAL`, .mat file), `weighting_matrix_scaling_factor`, `bartlett_kernel_lag`,
  `penalized_estimator`, `se_tolx`, `burnin`, `seed`, `simulation_multiple`,
  `bounded_shock_support`, `mode_compute` (every value, plus 13), `additional_optimizer_steps`,
  `verbose`, `relative_irf`, `add_tiny_number_to_cholesky`, `drop`, `replic`, `cova_compute`,
  `prior_trunc`, `qz_criterium`, `schur_vec_tol`, `irf_matching_file`, `mode_check`,
  `analytic_standard_errors`, `analytic_jacobian`; Bayesian IRF matching with `mh_replic > 0` (the
  MCMC of `+mom/run.m`, diagnostics, MHM). `use_pct` (Parallel Computing Toolbox) does not apply.
- Iterated weighting matrices, sandwich standard errors, J-test, Dynare's printed tables, results
  in `<fname>_mom_results.json`.
- Accuracy (RBC with GMM at orders 1-3, SMM at orders 1-2 fed MATLAB's draws, penalized GMM, IRF
  matching on a small NK model): data moments 2e-15, objective and model moments 1e-12, optimal
  weighting matrix 1e-12, numerical standard errors 2e-9, J-test 1e-13, IRF-matching Hessian
  2e-12; estimates agree to 1e-6..1e-5 where the optimizer is deterministic. Analytic derivatives
  at order 1 agree to 1e-13; at orders 2-3 they are Richardson-extrapolated differences (~1e-5 /
  1e-3 relative to the largest entry).

## Nonlinear estimation (particle filters)

`src/nonlinear/`, `reference/nonlinear/`, `tests/nonlinear.rs`, `tests/nonlinear_estimation.rs`.

- `estimation(order = 2|3, ...)` runs `non_linear_dsge_likelihood.m` with `filter_algorithm`
  `sis`, `apf`, `gf`, `gmf`, `cpf` or `nlkf`; options `number_of_particles`,
  `resampling = systematic|none|generic`, `resampling_threshold`,
  `resampling_method = kitagawa|stratified|smooth`, `proposal_approximation`,
  `distribution_approximation = cubature|unscented|montecarlo`, `cpf_weights`,
  `particle_filter_options` (`pruning`, `mixture_state_variables`, `unscented_*`,
  `initial_state_prior_std`, ...), every `mode_compute`, `mode_file`, the MCMC samplers and DSMH.
  Measurement errors (diagonal or correlated); order 3 with or without pruning. Particles move in
  parallel and every evaluation restarts its random stream, so the likelihood is a deterministic
  function of the parameters.
- On `rbcnl.mod` (orders 2 and 3, 33 filter configurations), fed MATLAB's own uniform and normal
  draws, every filter gives Dynare's likelihood to 1e-13 relative, except
  `resampling_method = smooth` (statistically: eigenvector signs are LAPACK's choice). The
  estimation with `gf` + `mode_compute = 4` finds Dynare's mode (fval -389.019612, Laplace
  366.69602 vs 366.69594).
- Per-evaluation times (T = 60, 2 observables, 3 states, 18 cores):

| filter | MATLAB order 2 / 3 | dynare-rust order 2 / 3 |
|---|---|---|
| `sis`, 5000 particles | 91 / 98 ms | 27 / 34 ms |
| `apf`, 5000 particles | 142 / 265 ms | 62 / 61 ms |
| `gf`, cubature | 8.8 / 12.3 ms | 0.22 / 0.56 ms |
| `nlkf`, unscented | 4.9 / 8.8 ms | 0.16 / 0.47 ms |
| `cpf`, unscented, 500 particles | 1853 / 1811 ms | 23 / 24 ms |
| `gmf`, cubature | 329 / 223 ms | 6.7 / 9.8 ms |

## Deterministic simulations

`src/deterministic/`, `src/ep/`; `tests/deterministic.rs`, `tests/q1_deterministic.rs`,
`tests/shocks.rs`, `tests/ep.rs`; most cases agree with Dynare to 1e-14 or 1e-15.

- `perfect_foresight_setup`, `perfect_foresight_solver` (Newton on the stacked system of `sim1.m`
  with a sparse LU; `maxit`, `tolf`, `tolx`, `noprint`, `homotopy_*` incl.
  `homotopy_linearization_fallback` and `homotopy_marginal_linearization_fallback`,
  `endval_steady`, `steady_*`, `linear_approximation`, `endogenous_terminal_period`,
  `robust_lin_solve`, `allow_nonfinite_values`, `check_jacobian_singularity`), `model(linear)`
  (`sim1_linear.m`), the legacy `simul`, `rplot`. Every `stack_solve_algo`: 1/6 `sim1_lbj.m`,
  2/3 `lin_solve.m` (GMRES/BiCGStab with `block_diagonal_lu*`, `incomplete_lu`, `iter_tol`,
  `iter_maxit`, `gmres_restart`), 7 (`dynare_solve` with `solve_algo`), 8/9 (sparse LU in place of
  ParU/Pardiso), 10; purely backward, forward and static models period by period. `lmmcp` (port of
  `lmmcp.m`) for complementarity problems.
- Deterministic shocks: `shocks`/`mshocks` with dates in `periods` and `learnt_in`; `values (xx)`
  with a MATLAB workspace vector; `shocks(overwrite)`; `shock_paths` (temporary/permanent,
  `exogenize`/`endogenize`, all scopes, `database` from a dseries or table). The MATLAB assignments
  they refer to are evaluated when they use `dseries`, `table`, `array2table`, matrix literals,
  `linspace`, `ones`, `zeros`, `repmat`, ranges, arithmetic and indexing; other statements are
  skipped with a warning.
- `perfect_foresight_controlled_paths` (`exogenize`/`endogenize`, also with `learnt_in`).
- `initval_file` / `histval_file` (.csv, .xlsx, .mat; `first_obs`, `last_obs`, `nobs`,
  `first_simulation_period`, `last_simulation_period` as integers or dates; auxiliary series as
  `dynamic_set_auxiliary_series`). The `series` option (a MATLAB dseries) is not supported.
- Perfect foresight with expectation errors: `perfect_foresight_with_expectation_errors_setup` /
  `_solver` (`constant_simulation_length`), `shocks(learnt_in=p[, overwrite])` (`values`, `add`,
  `multiply`), `mshocks(learnt_in=p)`, `endval(learnt_in=p)` (`=`, `+=`, `*=`).
- Extended path: `extended_path(periods, solver_periods, order, hybrid,
  use_first_order_solution, integration=tensor_gaussian_quadrature|stroud|unscented,
  tree=perfect|sparse, number_of_quadrature_nodes, lmmcp)` (`ep_problem_0.m` / `ep_problem_1.m`,
  trust-region solver on the sparse Jacobian, homotopy on failures, `solve_algo=10` on the
  stochastic tree to 3e-13). Given the same shocks the paths are Dynare's. Not ported: `hybrid` > 2
  (needs `dr.g_0`, which Dynare 7.2 never sets), `innovation_distribution` (only settable in a
  `verbatim` block), `extended_path_mc` (no command calls it).
- Speed (setup + solver): RBC with 150-200 periods 0.2-0.7 ms against MATLAB's 4-27 ms; SW07 with a
  temporary shock (incl. steady state) 1.7 ms against 4.6 ms.

## OccBin

`src/occbin/`, `reference/occbin/`, `reference/occbin_est/`, `tests/occbin.rs`, `tests/occbin_est.rs`.

- Up to two occasionally binding constraints: `occbin_constraints` (`bind`, `relax`,
  `error_bind`, `error_relax`, `STEADY_STATE()`), `[name=, bind=, relax=]` equation tags,
  `shocks(surprise[,overwrite])`, `occbin_setup`, `occbin_solver` (all `simul_*` options),
  `occbin_write_regimes` (written as `.csv` instead of `.xls`), `occbin_graph`. Paths, regime
  histories and the per-period state space agree with Dynare to ~1e-14 on a ZLB NK model (5 option
  variants and the `maxit` error), a borrowing-constraint model and Dynare's two-constraint RBC
  example.
- Estimation: a model with `occbin_constraints` is estimated with the piecewise Kalman filter
  (regime guesses, `filter_use_relaxation`, brute-force guesses, mixture of solutions) or, with
  `likelihood_inversion_filter`, the inversion filter; every `mode_compute`, `mode_check`,
  `mh_tune_jscale` and MCMC runs on them. `smoother` / `calib_smoother` run
  `occbin.DSGE_smoother` or the inversion-filter smoother; `occbin_write_regimes(smoother)`. After
  an OccBin MCMC, `smoother`/`filtered_vars` give the posterior distributions of the smoothed
  variables and shocks, and `forecast=H` the posterior `MeanForecast`/`PointForecast`.
  `occbin_setup` options `likelihood_*`, `smoother_*` and `filter_use_relaxation` as
  `+occbin/setup.m` forwards them (the others are ignored, as in Dynare 7.2). `occbin.irf` /
  `occbin.forecast` are available as library functions.
- Csminwel on the piecewise Kalman filter follows Dynare's iterates for 4 iterations, then drifts on
  the non-smooth likelihood (same mode to 2e-3, fval within 1e-3); as in Dynare the numerical
  Hessian there is often not positive definite (use `mode_compute=6` before an MCMC).
- Not ported: `lik_init` other than 1 with OccBin, `filter_step_ahead` > 1, the particle-based
  OccBin filters, `smoother_redux`, `smoother_first_period_occbin_update`.

## Optimal policy

`src/policy/`, `src/model/planner.rs`, `reference/policy/`, `tests/policy.rs`.

- `planner_objective`, `ramsey_model` (+ `stoch_simul` at order 1 or 2), `ramsey_policy`,
  `evaluate_planner_objective` (orders 1 and 2, linear-quadratic branch), `discretionary_policy`
  (`discretionary_tol`, `maxit`), `osr` with `osr_params` / `optim_weights` or the planner
  objective at order 1 or 2 (`opt_algo`: every `mode_compute` that `+osr/run.m` allows, `optim`,
  `osr_params_bounds`), `planner_discount`. The Ramsey FOCs are built symbolically as the
  preprocessor does (one `MULT_i` per equation, `optimal_policy_discount_factor` appended). The
  Ramsey steady state is `dyn_ramsey_static.m`, with or without a `steady_state_model` conditional
  on the `instruments`.
- Accuracy: Ramsey decision rules 1e-13 (one instrument) / 1e-11 (two instruments); with a
  `steady_state_model`: steady state 5e-12, `ghxx` 3e-10, second-order welfare 1e-11;
  `ramsey_policy(order=2, planner_discount=0.98)`: first order 1e-13, `ghxx` 1.5e-9 relative;
  welfare at order 1 to all digits; discretion 1e-16; OSR parameters 2e-9 (csminwel) / exact
  (simplex); OSR on second-order welfare with binding bounds: optimum exact, objective 1e-11.
- Differences: `ramsey_constraints`, which Dynare 7.2 parses and ignores, are enforced in the
  Ramsey steady state (projected Newton inside the box; info 81 if there is no root inside);
  `opt_algo` 1 runs the box-projected BFGS substitute. Not supported: welfare at order 3,
  `ramsey_policy` after perfect foresight, MATLAB steady-state files, `histval`/deterministic shocks
  as initial conditions of the conditional welfare.

## BVAR and LaTeX

- BVAR (Sims' Minnesota prior, `+bvar` package): `bvar_density(OPTIONS) MAXNLAGS;`,
  `bvar_forecast(OPTIONS) NLAGS;`, `bvar_irf(NLAGS[, 'Cholesky'|'SquareRoot']);` with `datafile`,
  `first_obs`, `nobs`, `presample`, `prefilter`, `noconstant`, `loglinear`,
  `bvar_prior_{tau,decay,lambda,mu,omega,flat,train}`, `bvar_replic`, `forecast`, `irf`,
  `conf_sig`; results in `<fname>_bvar_{density,forecast_<p>,irf_<p>}.json` (the fields of
  `oo_.bvar`). Draws use dynare-rust's random stream. Files without a `model` block are accepted.
- LaTeX: `write_latex_original_model`, `write_latex_dynamic_model`, `write_latex_static_model`
  (`write_equation_tags`), `write_latex_definitions`, `write_latex_parameter_table`,
  `write_latex_prior_table`, `collect_latex_files`, `estimation(TeX)` tables, `stoch_simul` TeX
  tables.

## Figures

Pure-Rust renderer (`src/graphics`: PDF, EPS, SVG, PNG); figures are written, never displayed.
Options `graph_format` (incl. `none`), `nograph`, `plot_priors`, `nodiagnostic`, `mode_check`
(+ `_neighbourhood_size`, `_symmetric_plots`, `_number_of_points`), `generate_trace_plots`.
Figures go to `<fname>/graphs/` (GSA and identification in their own folders), named as Dynare
names them, for every command that plots in Dynare.

## VAR family and PAC models

`src/varmodels/`, `tests/varmodels.rs`, `tests/pac.rs`, `reference/varmodels/`, `reference/pac/`.

- `var_model(model_name, eqtags, structural)` (equations may have `diff(x)` on the left),
  `trend_component_model(model_name, eqtags, targets)`, `var_expectation_model(model_name,
  expression, auxiliary_model_name, horizon, discount, time_shift)` and the `var_expectation(m)`
  operator; `var_expectation.initialize`, `var_expectation.update`, `get_companion_matrix` as
  statements. Parameters to 1e-12, companion matrices and first-order solution against Dynare
  (VAR with constant and lags, structural VAR, trend component with ECM equations, finite, infinite
  and shifted horizons). Dynare 7.2's `update_parameters.m` puts the *index* of a parameter used in
  `expression` into alpha; this is reproduced.
- `pac_model(model_name, auxiliary_model_name, discount, growth, kind, auxname)`,
  `pac_expectation(m)`, `pac_target_info(m)` blocks and `pac_target_nonstationary(m)`: the PAC
  equation is decomposed and the expectation becomes an auxiliary variable defined by the `h_<m>_*`
  parameters (VAR or trend-component auxiliary model, growth neutrality correction) or by the MCE
  recursion. These auxiliary variables are ordinary endogenous variables, so they appear in the
  `stoch_simul` tables (Dynare hides them). `pac.initialize`, `pac.update.expectation`,
  `pac.mce.parameters` as statements.
- `pac.estimate.nls` / `pac.estimate.iterative_ols` with `eparams`, `edata = dseries(...)` or
  `simul_backward_model(...)` data and date ranges; MATLAB optimizer names run a projected
  Gauss-Newton (not the MATLAB algorithms). Results in `<fname>_pac_<model>.json`.
- Checked on four models (VAR + composite target + growth, trend component + growth, simple target,
  MCE): h / growth neutrality / `mce_alpha` parameters to 1e-10, steady state, decision rules and
  IRFs to 1e-9, NLS and iterative OLS estimates 1e-7 / 1e-9. Dynare's
  `examples/semistructural/pac_model.mod` runs end to end: with MATLAB's draws replayed the paths
  match to 3e-14 and the NLS estimates to fmincon's tolerance (3e-8). Not ported: rule-of-thumb
  agents, `pac.bgp.*`, `pac.print`, `pac.check`.

## Heterogeneous-agent models

`src/heterogeneity/`, `tests/heterogeneity.rs`, `reference/heterogeneity/`.

- `heterogeneity_dimension`, `var/varexo/parameters(heterogeneity=NAME)`,
  `model(heterogeneity=NAME)` (complementarity conditions), `shocks(heterogeneity=NAME)`, `SUM(x)`,
  built as the 7.2 preprocessor builds them, with its errors.
- Steady state: `heterogeneity_load_steady_state`, `heterogeneity_compute_steady_state`,
  `rouwenhorst` (ports of the `+heterogeneity` functions and of the steady-state MEX: time
  iteration with Fischer-Burmeister and the `block_trust_region` solver, forward iteration, Broyden
  calibration of free parameters). The steady-state input structure is read from a MAT file: the
  `verbatim` MATLAB blocks that examples use to build it cannot run.
- `heterogeneity_solve(truncation_horizon = T)` (sequence-space Jacobian method of
  `+heterogeneity/solve.m`; the MEX it calls, whose sources are not distributed, are
  re-implemented and validated against Dynare's output) and `heterogeneity_simulate(irf, periods,
  irf_shocks, relative_irf, ...)` (IRFs, stochastic simulation, news-shock sequences; figures as
  `plot_irfs.m` / `plot_simulation.m`, no TeX eps-loader).
- Checked on Krusell-Smith and one-asset HANK: preprocessor structures and generated residual code
  (1e-13), `oo_.heterogeneity`, computed policies, distribution and calibrated parameters
  (1e-11 / 1e-12); `dr.G`, `F`, `J_ha`, `curlyYs`, `curlyDs`, `J`, IRFs, the stochastic convolution
  given MATLAB's draws and news-shock paths to 1e-12. Stochastic draws come from dynare-rust's
  stream, not MATLAB's `randn`.

## MS-SBVAR

`src/msbvar/`, `reference/msbvar/`, `tests/msbvar*.rs`, `tests/sbvar_rest.rs`. Dynare's C sources
(the `ms_sbvar_command_line` MEX) are not shipped; the model is reconstructed from the files Dynare
writes and validated numerically against Dynare.

### Declarations and `sbvar`

- `svar_identification` (exclusions, `upper_cholesky`, `lower_cholesky`, `restriction equation`
  linear restrictions), `markov_switching` (`chain`, `number_of_regimes`, scalar or vector
  `duration` (Dynare 7.2's preprocessor aborts on a vector), `restrictions`, `parameters`,
  `number_of_lags`) and `svar`, validated as the preprocessor does.
- `sbvar(datafile=, freq=, initial_year=, final_year=, nlags=,
  coefficients_prior_hyperparameters=, forecast=, no_bayesian_prior, contemp_reduced_form,
  vlistlog)`: Sims-Zha prior, posterior mode of `A0` (BFGS in place of `csminwel`), `A+`, IRFs,
  forecast, log marginal data density with its Gibbs terms, Dynare's `.prn` files and figures.
  `A0` (up to column signs), `A+`, IRFs, forecasts and density terms agree to 1e-5 or better on
  five configurations; the Gibbs log mdd within Monte Carlo error (seed sd ~0.15).
  `contemp_reduced_form` is not checked against Dynare; with `no_bayesian_prior` Dynare 7.2 fails
  in the density part and dynare-rust omits it.
- `cross_restrictions` fails in Dynare 7.2 (`msstart2.m`). In dynare-rust a restriction line mixing
  lag 0 and lags > 0 is the cross restriction `R a+_j + Q a_j = 0`, and `sbvar` returns the exact
  restricted posterior mode (no marginal data density).

### Markov-switching commands

- `ms_estimation` (`specification = none | sims_zha`, `contemp_reduced_form`, `no_create_init`,
  the `convergence_*`, `max_*` and perturbation options): Dynare's priors (initial likelihood and
  prior equal Dynare's to 1e-6 in six reference cases) and search structure (block iterations,
  then perturbation stages). Dynare's file formats (`init_<tag>.dat`, `est_final_<tag>.out`, ...),
  so later commands also run on Dynare's estimates or in a separate run. The search is stochastic
  in Dynare too; dynare-rust reaches Dynare's mode or better within 0.05 in every reference case.
  `sims_zha` with restrictions on `A+` crashes Dynare's MEX and is refused.
- `ms_compute_probabilities` (filtered, smoothed, `real_time_smoothed`; 2e-6, the files' precision).
- `ms_compute_mdd` (`dw_mdd_switch.c` proposal, `proposal_type` 1-6; Muller, bridge, Waggoner-Zha):
  on Dynare's own draws the estimates equal its printed values (1e-4); with fresh draws, means over
  20 seeds agree with 20 Dynare runs to 0.035. Dynare mixes two kernels differing by `nvars ln 2`;
  this is reproduced by default (`MddOptions::consistent` normalises both).
- `ms_simulation` (adaptive random-walk Metropolis, `mh_replic`, `drop`, `thinning_factor`,
  `adaptive_mh_draws`, `save_draws`), `ms_irf`, `ms_forecast`, `ms_variance_decomposition` (per
  regime, ergodic mixture, `filtered_probabilities`, `regime=`, `regimes`, `parameter_uncertainty`,
  `free_parameters`, `error_bands`), with Dynare's file layouts and figures. At Dynare's mode:
  per-regime IRFs and variance decompositions 2e-5 (printed precision), ergodic/filtered IRF
  percentiles exact, mixture variance decompositions 3e-3 (Dynare samples the regime weights),
  forecast percentiles 0.1 at 100000 draws (Dynare's own noise ~0.03). Dynare 7.2's first forecast
  row is the one-step forecast of the last observation, which is reproduced.
- Known differences: Dynare's reported "Final prior Value" includes an extra constant (ln 4 with
  one chain); our files report the kernel the search maximises. Dynare's block-EM/csminwel with its
  random restarts and its C sampler's random numbers are not reproduced. Dynare's
  variance-decomposition percentiles (`error_bands`) differ from the exact-mixture percentiles by
  ~4e-4 and do not sum to one. `parameter_uncertainty` is not checked against Dynare (its draws are
  in its own `null()` basis; use draws produced by dynare-rust).
