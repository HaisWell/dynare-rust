// SPDX-License-Identifier: GPL-3.0-or-later
//! `dynare-rs model.mod [-DNAME=VALUE ...]`: run a .mod file like `dynare model` in MATLAB.

use std::path::PathBuf;

use clap::Parser;
use dynare::driver::{RunOptions, run};

#[derive(Parser)]
#[command(name = "dynare-rs", version, about = "Rust port of Dynare 7.2 for estimating linear(ised) models")]
struct Cli {
    /// The .mod file.
    modfile: PathBuf,
    /// Dynare's options after the file name: savemacro[=FILE], onlymacro, linemacro, nograph,
    /// onlymodel, nointeractive, nolog, ... (see `dynare::misc::cli`).
    #[arg(value_name = "DYNARE_OPTIONS")]
    dynare_options: Vec<String>,
    /// @#include search path, as Dynare's -I<path>.
    #[arg(short = 'I', value_name = "PATH")]
    include: Vec<PathBuf>,
    /// Macro definitions, as Dynare's -DNAME=VALUE (e.g. -DSTAGE=2). A bare -DNAME means true.
    #[arg(short = 'D', value_name = "NAME[=VALUE]")]
    defines: Vec<String>,
    /// Seed of the MCMC random streams.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Override mh_replic (e.g. for a quick run).
    #[arg(long)]
    mh_replic: Option<usize>,
    /// Number of threads (default: all cores).
    #[arg(long)]
    threads: Option<usize>,
    /// Posterior evaluations run in parallel per MCMC chain (and in mh_tune_jscale)
    /// by pre-fetching future iterations: 0 = automatic (2/3 of the threads / chains, at most 6),
    /// 1 = plain sequential chains. The draws do not depend on it.
    #[arg(long, default_value_t = 0)]
    prefetch: usize,
    /// gemm/gemv backend: accelerate (default) or rust (matrixmultiply). Same as DYNARE_BLAS.
    #[arg(long, value_name = "accelerate|rust")]
    blas: Option<String>,
    /// Less console output from the optimizers.
    #[arg(long)]
    quiet: bool,
    /// Open a small Terminal window with the live progress (phase, optimizer value,
    /// one bar per MCMC chain, ETA), shown by the `dynare-progress` viewer.
    #[arg(long)]
    window: bool,
}

fn main() {
    let cli = Cli::parse();
    // Compute threads at the highest QoS class (macOS keeps them on the fastest cores).
    dynare::parallel::init_thread_pool(cli.threads);
    match cli.blas.as_deref().map(str::to_ascii_lowercase).as_deref() {
        None => {}
        Some("accelerate") => dynare::linalg::set_blas_backend(dynare::linalg::BlasBackend::Accelerate),
        Some("rust") => dynare::linalg::set_blas_backend(dynare::linalg::BlasBackend::Rust),
        Some(b) => {
            eprintln!("ERROR: --blas {b}: expected accelerate or rust");
            std::process::exit(2);
        }
    }
    let defines = cli
        .defines
        .iter()
        .map(|d| match d.split_once('=') {
            Some((k, v)) => (k.to_string(), v.to_string()),
            None => (d.clone(), String::new()),
        })
        .collect();
    if cli.window {
        open_progress_window(&cli.modfile);
    }
    let mut dyn_opts = match dynare::misc::cli::CliOptions::parse(&cli.dynare_options) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ERROR: {e:#}");
            std::process::exit(2);
        }
    };
    dyn_opts.include_paths.extend(cli.include.iter().cloned());
    let mut defines = defines;
    if let Err(e) = dyn_opts.merge_first_line(&cli.modfile, &mut defines) {
        // `// --+ options: ... +--` on the first line of the .mod file
        eprintln!("ERROR: {e:#}");
        std::process::exit(2);
    }
    let ro = RunOptions {
        defines,
        seed: cli.seed,
        mh_replic: cli.mh_replic,
        quiet: cli.quiet,
        prefetch: cli.prefetch,
        cli: dyn_opts,
    };
    if let Err(e) = dynare::model::external::catch_external_errors(|| run(&cli.modfile, &ro)) {
        dynare::progress::finish(false, &format!("{e:#}"));
        eprintln!("ERROR: {e:#}");
        std::process::exit(1);
    }
}

/// Runs `dynare-progress --window` (installed next to this binary) on the progress
/// file of `modfile`; the viewer waits for the file if the run has not written it yet.
fn open_progress_window(modfile: &std::path::Path) {
    let Some(fname) = modfile.file_stem().and_then(|s| s.to_str()) else { return };
    let dir = modfile.parent().map(PathBuf::from).unwrap_or_default();
    let file = dynare::progress::file_path(&dir, fname);
    let viewer = std::env::current_exe().ok().map(|e| e.with_file_name("dynare-progress"));
    let ok = viewer
        .is_some_and(|v| std::process::Command::new(v).arg("--window").arg(&file).status().is_ok_and(|s| s.success()));
    if !ok {
        eprintln!("WARNING: could not open the progress window (dynare-progress); the run continues.");
    }
}
