// SPDX-License-Identifier: GPL-3.0-or-later
//! `repair-hessian <mode.mat>`: make the Hessian in a mode file positive definite by
//! replacing its eigenvalues with their absolute values (floored at a tiny positive value),
//! so it can seed the MCMC proposal.
//!
//! A manual step between mode finding and the MCMC: the finite-difference Hessian at a
//! newrat mode can carry small negative eigenvalues from numerical noise, which Dynare
//! rejects ("covariance matrix has to be positive definite").
//! Dynare does not do this itself, so it is a separate, explicit tool. The original file is
//! kept as `<name>_newrat.mat`.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use dynare::linalg::{Mat, sym_eig};
use dynare::matfile::{self, MatOut};

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).ok_or_else(|| anyhow!("usage: repair-hessian <mode.mat>"))?);
    let vars = matfile::read(&path)?;
    let hh = vars.get("hh").and_then(|v| v.as_matrix()).ok_or_else(|| anyhow!("{}: no hh", path.display()))?;
    let x = vars.get("xparam1").and_then(|v| v.as_vec()).ok_or_else(|| anyhow!("no xparam1"))?;
    let fval = vars.get("fval").and_then(|v| v.as_scalar()).unwrap_or(f64::NAN);
    let names = vars.get("parameter_names").and_then(|v| v.as_strings()).unwrap_or_default();
    if !hh.is_square() {
        bail!("hh is not square");
    }
    let sym = (&hh + hh.transpose()) * 0.5;
    let (w, v) = sym_eig(&sym);
    let neg: Vec<f64> = w.iter().copied().filter(|&e| e <= 0.0).collect();
    if neg.is_empty() {
        println!("{}: Hessian already positive definite (min eigenvalue {:.3e}); nothing to do.", path.display(), w[0]);
        return Ok(());
    }
    let floor = w.iter().map(|e| e.abs()).fold(0.0, f64::max) * f64::EPSILON * w.len() as f64;
    let w2: Vec<f64> = w.iter().map(|e| e.abs().max(floor)).collect();
    let fixed = Mat::from_fn(hh.nrows(), hh.ncols(), |i, j| (0..w2.len()).map(|k| v[(i, k)] * w2[k] * v[(j, k)]).sum());
    let stem = path.file_stem().and_then(|s| s.to_str()).context("file name")?;
    let backup = path.with_file_name(format!("{stem}_newrat.mat"));
    std::fs::copy(&path, &backup).with_context(|| format!("backing up to {}", backup.display()))?;
    matfile::write(
        &path,
        &[
            ("xparam1", MatOut::Vector(&x)),
            ("hh", MatOut::Matrix(&fixed)),
            ("fval", MatOut::Scalar(fval)),
            ("parameter_names", MatOut::Strings(&names)),
        ],
    )?;
    println!(
        "{}: {} non-positive eigenvalue(s) flipped to |.| (most negative {:.3e}, largest {:.3e}); original kept as {}.",
        path.display(),
        neg.len(),
        neg.iter().copied().fold(f64::INFINITY, f64::min),
        w[w.len() - 1],
        backup.display()
    );
    Ok(())
}
