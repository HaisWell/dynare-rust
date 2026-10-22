# dynare-rust design

dynare-rust is a Rust port of Dynare 7.2 (MATLAB sources under `<DYNARE_ROOT>/matlab`, C++
preprocessor under `<DYNARE_ROOT>/preprocessor`). It runs Dynare `.mod` files end to end and
reproduces Dynare's numbers: decision rules to ~1e-10 and likelihoods to ~1e-8 relative or
better (most areas reach 1e-12).

## Principles

- **Port the algorithm, not just the result.** Same branches, tolerances, defaults, error
  codes and fallbacks as Dynare (e.g. a singular diffuse filter falls back to the univariate
  filter). Code that mirrors a Dynare function says so in a doc comment
  (`/// Port of kalman/likelihood/kalman_filter_d.m`).
- **Every area is checked against Dynare.** Each area has a `reference/<area>/` folder with
  test models, the MATLAB scripts that dump Dynare's results and their JSON output, and an
  integration test in `tests/` (see [docs/REFERENCES.md](docs/REFERENCES.md)).
- **Determinism.** A posterior evaluation is a pure function of the parameter vector, bit for
  bit, whatever thread runs it and whatever it evaluated before. MCMC pre-fetching relies on
  this to reproduce the sequential chain exactly. Random draws come from seeded ChaCha8
  streams (one per chain or per task), so runs are reproducible with `--seed`.
- **Fast hot paths.** Anything evaluated per likelihood preallocates its workspace, takes no
  locks, and is `Send + Sync`. Large matrix products go through `linalg::gemm` (Accelerate or
  OpenBLAS, or the pure-Rust `matrixmultiply` with `--blas rust`).

## Data types and conventions

- Matrices are `nalgebra::DMatrix<f64>` (`linalg::Mat`), column-major like MATLAB and LAPACK.
- Errors are `anyhow::Result` at API boundaries; Dynare's `info` codes (`i32`) are kept where
  Dynare returns them (3 explosive, 4 indeterminate, 42 above a bound, 46 negative det(F), ...).
- Indices are 0-based in Rust; JSON references keep MATLAB's 1-based indices.
- Dynare's global state (`M_`, `options_`, `oo_`) is explicit: `model::Model` is `M_`, options
  are parsed per command, and the driver keeps the state that persists between commands.
- Figures go through `graphics::{Figure, save}` into `<fname>/graphs/` (PDF, PNG or EPS).

## Crate layout

| path | contents |
|---|---|
| `src/modfile/` | macro processor, lexer, parser (`ModFile`, `Command`), option checks, `verbatim` and MATLAB-statement handling, translation of template `_steadystate.m` files |
| `src/model/` | symbolic model (`Model` = `M_`), the preprocessor's DataTree and transform pass (`dtree*.rs`: aux variables, `diff`, `EXPECTATION`, var(log), detrending, Ramsey FOCs), derivatives up to order 3, compiled evaluation tapes, block decomposition, external functions |
| `src/linalg/` | BLAS/LAPACK bindings, ordered QZ (`mjdgges`), ordered real Schur |
| `src/solver/` | steady state (`dynare_solve` algorithms), first-order solver (structured QZ with a full-QZ fallback, cycle and logarithmic reduction), restricted state space, orders 2-3 (`korder/`: k-order rules, simulation, GIRFs, pruned state space) |
| `src/kalman/` | diffuse, standard and univariate filters with missing data, `compute_Pinf_Pstar`, Lyapunov initializations, smoothers, measurement-error augmentation, heteroskedastic and skewed filters |
| `src/dsge.rs` | `DsgeModel` and `DsgePosterior`: parameters -> steady state -> solution -> likelihood + prior (`dsge_likelihood.m`) |
| `src/data.rs`, `src/dataset.rs`, `src/dates.rs`, `src/data_m.rs` | data files (csv, xlsx, mat, m), `makedataset`, Dynare's `dates` |
| `src/estimation/` | priors and bounds, mode finders (`optim/`), Hessian, `mode_check`, MCMC (RW Metropolis with pre-fetching, `samplers/` for the other samplers), MH files and continuation, posterior statistics, marginal densities, convergence diagnostics, model comparison, DSGE-VAR, `analytic_derivation`, plots |
| `src/forecast/` | forecasts, the full smoother, post-estimation commands, conditional forecasts |
| `src/analysis/` | `stoch_simul` at order 1 (IRFs, moments, variance decompositions, simulations), `check`, Bayesian IRFs and moments, shock and realtime decompositions, partial information, AIM |
| `src/deterministic/` | perfect foresight (stacked Newton, sparse LU, Krylov solvers, LMMCP, block solve), `shocks`/`shock_paths`, `varexo_det` |
| `src/ep/` | extended path and stochastic extended path, expectation errors, controlled paths, `initval_file`/`histval_file` |
| `src/occbin/` | OccBin solver, piecewise Kalman and inversion filters, smoother, IRFs, forecasts |
| `src/policy/` | Ramsey, discretionary policy, OSR, welfare |
| `src/identification/`, `src/gsa/`, `src/mom/` | identification, global sensitivity analysis, method of moments |
| `src/nonlinear/` | particle filters, nonlinear estimation, DSMH, SMC samplers, online filter |
| `src/bvar/`, `src/msbvar/`, `src/varmodels/` | BVAR, MS-SBVAR, VAR / trend-component / PAC models |
| `src/heterogeneity/` | heterogeneous-agent models (steady state, sequence-space solution, simulation) |
| `src/latex/`, `src/q4/`, `src/graphics/`, `src/reporting.rs` | LaTeX output, epilogue, decomposition output, JSON output, figure renderer, reports |
| `src/misc/` | small commands, homotopy, `model_info`, command-line options |
| `src/driver.rs`, `src/main.rs`, `src/parallel.rs`, `src/progress.rs` | command execution, CLI, thread pool, progress reporting |
| `src/bin/` | helper binaries (`dynare-blocks`, `dynare-progress`, `repair-hessian`) |

## Main interfaces

```rust
// Preprocessing
pub fn modfile::parse_mod_file(path: &Path, defines: &[(String, String)]) -> Result<ModFile>;
// ModFile { model: Model, varobs, sigma_e, h, estimated_params, commands: Vec<Command>, ... }
// Command: one variant per command family; Options = Vec<(String, OptValue)> in file order.

// Model (M_)
impl Model {
    fn static_residuals(&self, y, x, p, out);
    fn static_jacobian(&self, y, x, p) -> Mat;
    fn dynamic_jacobian(&self, ys, x, p) -> Mat;      // [y_{t-1}, y_t, y_{t+1}, x] columns
    fn dynamic_derivatives(&self, order) -> DynamicDerivatives;
}

// First-order solution
FirstOrderSolver::new(lli, n_exo, opts).solve(&g1, &ys) -> Result<DecisionRule, SolveError>;
// DecisionRule { ys, ghx, ghu, order_var, state_var, nstatic, nfwrd, npred, nboth, ... }

// Likelihood
kalman::likelihood(y, missing, mf, t, r, q, h, pstar, pinf, opts) -> LikOutput;
pub trait Objective: Sync { fn eval(&self, x: &[f64]) -> (f64, i32); fn dim(&self) -> usize; }
// DsgePosterior, OccbinPosterior and NonlinearPosterior implement Objective:
// fval = -(loglik + logprior), +inf on failure, Dynare's info code.
```

Derivatives are computed symbolically once, simplified, and compiled to flat instruction tapes,
so a Jacobian evaluation per MCMC draw costs microseconds. Auxiliary variables are created as
Dynare's preprocessor creates them (same names, order and `lead_lag_incidence`).

## Driver

`driver::run` loops over the commands in file order and keeps the state Dynare keeps in its
globals: the last estimation (model, data, mode, posterior moments), the persistent
`stoch_simul` options, the perfect-foresight state, and one session object per area that has
state across commands (OccBin, optimal policy, method of moments, BVAR, decompositions).
Commands that do not apply (MATLAB-only or code-generation features) are accepted with a note.

## Estimation and MCMC

The mode finders are reached through `estimation::optim::minimize` (every `mode_compute`), the
samplers through `samplers::posterior_sampler`. The random-walk Metropolis sampler runs the
chains in parallel; spare threads pre-fetch each chain's likely next proposals (the proposals
under the most likely accept/reject outcomes, drawn from a copy of the chain's stream), and the
chain consumes those it reaches, so the draws are identical to a sequential run. Chain files
follow Dynare's `<fname>/metropolis/` layout; the generator state is stored where Dynare stores
MATLAB's, so `load_mh_file` and `mh_recover` continue each chain bit for bit.

The results file `<fname>/Output/<fname>_results.json` holds the marginal densities read by
`model_comparison` (`log_data_density_laplace`, `mcmc.log_data_density_mhm`, or
`MarginalDensity.{LaplaceApproximation, ModifiedHarmonicMean}`).

## Testing

`cargo test --release` runs the unit tests and the integration tests in `tests/`, which compare
with the JSON references under `reference/`. Tests that need MATLAB or take long are
`#[ignore]`d. Typical tolerances: decision rules and state-space matrices 1e-10 absolute,
likelihoods 1e-8 relative or better, priors 1e-12.

To add a feature: port the Dynare function (cite the `.m` file), write a dump script
`reference/<area>/dump_<area>.m` that records Dynare's output in JSON, add an integration test,
keep the driver change to one call into the area's module, and keep every `match` on `Command`
exhaustive.
