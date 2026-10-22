# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - 2026-10-22

Initial public release.

- Port of Dynare 7.2: 153 of the 173 commands, blocks and macro directives of the reference
  manual; the other 20 are not applicable (MATLAB-only or code-generation features).
- Preprocessor and macro language, steady state, perturbation at orders 1-3, perfect
  foresight, extended path, OccBin, optimal policy, Kalman filters and smoothers, ML and
  Bayesian estimation (mode finders, MCMC samplers, SMC), forecasts, shock decompositions,
  identification, global sensitivity analysis, method of moments, particle filters, BVAR,
  MS-SBVAR, VAR/PAC models, heterogeneous-agent models, LaTeX output and figures.
- Validated against Dynare 7.2 reference output committed under `reference/`.
- Builds on macOS (Accelerate) and Linux (OpenBLAS).
