// SPDX-License-Identifier: GPL-3.0-or-later
//! `sensitivity(ppost=1)` — the posterior sample of a previous MCMC.
//!
//! Dynare reads the parameter draws `PosteriorIRF.m` stored in
//! `<fname>/metropolis/<fname>_param_irf<k>.mat` (a random subsample of the MCMC after
//! `mh_drop`, `options_.sub_draws` draws, at most 1200). dynare-rust draws the same kind of
//! subsample (`get_posterior_subsample.m`) directly from the Dynare-compatible MCMC files in
//! `<fname>/metropolis` (from its own random stream), so the posterior GSA agrees with Dynare's
//! statistically.

use std::path::Path;

use anyhow::{Context, Result};

use super::Ctx;
use super::stability::{StabMap, classify_sample};
use crate::dsge::DsgeModel;
use crate::estimation::mh_files::{MhPaths, load_last_record, read_chains};

/// The posterior subsample as a stability map (every draw counts as stable, as
/// `+gsa/run.m` sets `istable = 1:size(lpmat,1)`), with the reduced form of each draw.
pub fn posterior_sample(
    ctx: &Ctx,
    dsge: &DsgeModel,
    dir: &Path,
    mh_drop: f64,
    sub_draws: Option<usize>,
    seed: u64,
) -> Result<StabMap> {
    let paths = MhPaths::new(&dir.join(&ctx.fname), &ctx.fname);
    let (rec, _) = load_last_record(&paths)
        .context("sensitivity(ppost): no MCMC draws (run an estimation with mh_replic > 0 first)")?;
    // cut_sample.m: first kept draw max(1, floor(mh_drop * TotalNumberOfMhDraws) + 1)
    let first = (mh_drop * rec.total_draws() as f64).floor() as usize;
    let chains = read_chains(&paths, &rec, first, &vec![0.0; rec.nblck])?;
    let b = crate::analysis::posterior::number_of_subdraws(&chains, sub_draws)?;
    let draws = crate::analysis::posterior::posterior_subsample(&chains, b, seed)?;
    let nshock = ctx.nshock;
    let lpmat0: Vec<Vec<f64>> = draws.iter().map(|d| d[..nshock].to_vec()).collect();
    let lpmat: Vec<Vec<f64>> = draws.iter().map(|d| d[nshock..].to_vec()).collect();
    let mut m = classify_sample(dsge, lpmat0, lpmat, true);
    // run.m: istable = all draws, no unstable ones
    let n = m.lpmat.len();
    if m.istable.len() != n {
        println!(
            "WARNING: sensitivity(ppost): {} posterior draws do not solve the model; they are left out.",
            n - m.istable.len()
        );
    }
    m.x0 = m.istable.first().map(|&k| m.point(k));
    println!(
        "sensitivity(ppost): {b} posterior draws from {} ({} chain(s), mh_drop = {mh_drop}).",
        paths.folder.display(),
        rec.nblck
    );
    Ok(m)
}
