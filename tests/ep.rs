//! Extended path and stochastic extended path against Dynare 7.2 (reference/ep/ref_ep_*.json,
//! written by reference/ep/dump_ep.m) on an RBC with irreversible investment (a `min()`
//! complementarity condition that binds in many periods) and two shocks: given MATLAB's shock
//! draws (`oo_.exo_simul`), the simulated paths must match. Cases: order 0; order 1 and 2 on
//! the full tree with Gauss-Hermite tensor nodes; order 2 on the sparse tree; Stroud
//! cubature; unscented transform (correlated shocks); the hybrid (second-order) correction.
mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use dynare::deterministic::DetState;
use dynare::ep::{EpModel, EpOptions, extended_path};
use dynare::linalg::Mat;
use dynare::modfile::{Command, OptValue, parse_mod_file};
use serde_json::Value;

fn ref_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/ep")
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| match r {
            Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect(),
            x => vec![x.as_f64().unwrap()],
        })
        .collect()
}

/// Runs one case and returns (max abs difference of the paths, Rust time, MATLAB time).
fn check(name: &str, case: u32) -> Option<(f64, f64, f64)> {
    let p = ref_dir().join(format!("ref_ep_{name}.json"));
    let r: Value = common::json_ref(&p);
    let mf = parse_mod_file(&ref_dir().join("rbcii.mod"), &[("CASE".into(), case.to_string())]).unwrap();
    let mut det = DetState::new(&mf).unwrap();
    let mut o = EpOptions::default();
    for c in &mf.commands {
        match c {
            Command::Steady(opts) => {
                let mut opts = opts.clone();
                opts.push(("noprint".into(), OptValue::Flag));
                det.steady(&mf.model, &opts).unwrap();
            }
            Command::Other(n, opts) if n == "extended_path" => {
                o.update(opts).unwrap();
            }
            _ => {}
        }
    }
    let ss_ref: Vec<f64> = r["steady_state"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    for (a, b) in det.steady_state.iter().zip(&ss_ref) {
        assert!((a - b).abs() < 1e-9, "{name}: steady state {a} vs {b}");
    }
    let ex = rows(&r["exo_simul"]);
    let shocks = Mat::from_fn(ex.len() - 1, ex[0].len(), |t, k| ex[t + 1][k]);
    let m = EpModel {
        model: &mf.model,
        params: &det.params,
        steady_state: &det.steady_state,
        exo_steady_state: &det.exo_steady_state,
        sigma_e: &mf.sigma_e,
    };
    let t0 = Instant::now();
    let out = extended_path(&m, &o, &det.steady_state.clone(), &shocks).unwrap();
    let el = t0.elapsed().as_secs_f64();
    assert!(out.status, "{name}: extended path failed");
    let want = rows(&r["endo_simul"]);
    let mut err = 0.0f64;
    for (v, row) in want.iter().enumerate() {
        for (t, w) in row.iter().enumerate() {
            err = err.max((out.endo_simul[(v, t)] - w).abs());
        }
    }
    let t_ml = r["t_ep"].as_f64().unwrap_or(f64::NAN);
    println!("{name:16} max |diff| = {err:.2e}   dynare-rust {:8.2} ms   Dynare {:8.1} ms", el * 1e3, t_ml * 1e3);
    Some((err, el, t_ml))
}

#[test]
fn ep_order0_matches_dynare() {
    if let Some((err, _, _)) = check("ep0", 1) {
        assert!(err < 1e-9, "max diff {err}");
    }
}

#[test]
fn sep_tensor_full_tree_matches_dynare() {
    for (name, case) in [("sep_tensor1", 2), ("sep_tensor2", 3)] {
        if let Some((err, _, _)) = check(name, case) {
            assert!(err < 1e-9, "{name}: max diff {err}");
        }
    }
}

#[test]
fn sep_sparse_tree_matches_dynare() {
    if let Some((err, _, _)) = check("sep_sparse2", 4) {
        assert!(err < 1e-9, "max diff {err}");
    }
}

#[test]
fn sep_cubature_and_unscented_match_dynare() {
    for (name, case) in [("sep_stroud1", 5), ("sep_unscented2", 6)] {
        if let Some((err, _, _)) = check(name, case) {
            assert!(err < 1e-9, "{name}: max diff {err}");
        }
    }
}

#[test]
fn ep_first_order_initial_guess_matches_dynare() {
    if let Some((err, _, _)) = check("ep0_first_order_guess", 8) {
        assert!(err < 1e-9, "max diff {err}");
    }
}

#[test]
fn sep_hybrid_matches_dynare() {
    if let Some((err, _, _)) = check("sep_hybrid1", 7) {
        assert!(err < 1e-9, "max diff {err}");
    }
}
