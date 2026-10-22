# Contributing

Bug reports, missing features and pull requests are welcome.

## Reporting a difference from Dynare

The most useful report is a small `.mod` file, the Dynare version and output you expected, and
what `dynare-rs` printed. Please say which platform you ran on.

## Pull requests

- Port Dynare's algorithm rather than an equivalent one: same branches, defaults, tolerances
  and fallbacks, so the numbers match. Name the Dynare function a piece of code mirrors in a
  doc comment (`/// Port of kalman/likelihood/kalman_filter_d.m`). [DESIGN.md](DESIGN.md) has
  the conventions and module contracts.
- Add a test against Dynare's output: put the test model and the script that dumps Dynare's
  results under `reference/<area>/` (see [docs/REFERENCES.md](docs/REFERENCES.md)) and
  commit the reference JSON with the test. Tests must not need MATLAB; anything that does is
  `#[ignore = "reason"]`.
- Before pushing:

  ```sh
  cargo fmt --all
  cargo clippy --release --all-targets   # no warnings
  cargo test --release
  ```

- Update [CHANGELOG.md](CHANGELOG.md) and, for new commands or options,
  `docs/gen_coverage.py` / [docs/COVERAGE.md](docs/COVERAGE.md).

By contributing you agree that your contribution is licensed under GPL-3.0-or-later, the
license of the project.
