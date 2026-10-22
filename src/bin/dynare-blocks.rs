// SPDX-License-Identifier: GPL-3.0-or-later
//! `dynare-blocks <file.mod> [-Dname=value ...]`: prints the preprocessor's block decomposition
//! of the model (`M_.block_structure_stat` and `M_.block_structure`, 1-based, as JSON
//! `{"static": ..., "dynamic": ...}`, `null` when the model cannot be normalized). Used to compare
//! with the Dynare preprocessor (`reference/blocks/extract_blocks.py`, 7.2 by default).

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use dynare::misc::model_info::{dynamic_block_structure, static_block_structure};

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().ok_or_else(|| anyhow!("usage: dynare-blocks <file.mod> [-Dname=value ...]"))?);
    let defines: Vec<(String, String)> = args
        .filter_map(|a| {
            a.strip_prefix("-D").map(|d| {
                d.split_once('=')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .unwrap_or((d.to_string(), "true".to_string()))
            })
        })
        .collect();
    let mf = dynare::modfile::parse_mod_file(&path, &defines)?;
    let out = serde_json::json!({
        "static": static_block_structure(&mf).map(|b| b.to_json(false)),
        "dynamic": dynamic_block_structure(&mf).map(|b| b.to_json(true)),
    });
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
