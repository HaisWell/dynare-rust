//! `perfect_foresight_controlled_paths` against Dynare 7.2 (reference/ep/ref_controlled_*.json,
//! written by reference/ep/dump_controlled.m, the manual's example): Newton (`sim1`),
//! `linear_approximation` (`sim1_linear`), homotopy (controlled values scaled by the share)
//! and expectation errors (`learnt_in`). The exogenous paths include the endogenized shocks.

mod common;
use std::path::{Path, PathBuf};

use dynare::deterministic::DetState;
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

fn check(name: &str, case: u32) {
    let p = ref_dir().join(format!("ref_controlled_{name}.json"));
    let r: Value = common::json_ref(&p);
    let mf = parse_mod_file(&ref_dir().join("controlled.mod"), &[("CASE".into(), case.to_string())]).unwrap();
    let out = std::env::temp_dir().join(format!("dynare_rs_controlled_{name}"));
    std::fs::create_dir_all(&out).unwrap();
    let mut det = DetState::new(&mf).unwrap();
    for (ci, c) in mf.commands.iter().enumerate() {
        match c {
            Command::Steady(o) => {
                let mut o = o.clone();
                o.push(("noprint".into(), OptValue::Flag));
                det.steady(&mf.model, &o).unwrap()
            }
            Command::Deterministic(n, o, _) => {
                det.simul.update(o);
                det.simul.noprint = true;
                det.simul.verbosity = 0;
                match n.as_str() {
                    "perfect_foresight_setup" => det.setup(&mf).unwrap(),
                    "perfect_foresight_solver" => det.solve(&mf.model).unwrap(),
                    _ => {}
                }
            }
            Command::Other(n, o) if n == "perfect_foresight_with_expectation_errors_setup" => {
                dynare::ep::pfwee::setup(&mf, &mut det, o, &ref_dir()).unwrap();
                det.simul.noprint = true;
                det.simul.verbosity = 0;
            }
            Command::Other(n, _) if dynare::ep::handles(n) => {
                dynare::ep::command(&mf, &mut det, ci, &out, "controlled", 0).unwrap();
            }
            _ => {}
        }
    }
    assert!(det.last.status, "{name}: simulation failed");
    let want = rows(&r["endo_simul"]);
    let got = det.endo_simul.as_ref().unwrap();
    let mut err = 0.0f64;
    for (j, row) in want.iter().enumerate() {
        for (t, w) in row.iter().enumerate() {
            err = err.max((got[(j, t)] - w).abs());
        }
    }
    let wx = rows(&r["exo_simul"]);
    let gx = det.exo_simul.as_ref().unwrap();
    let mut errx = 0.0f64;
    for (t, row) in wx.iter().enumerate() {
        for (k, w) in row.iter().enumerate() {
            errx = errx.max((gx[(t, k)] - w).abs());
        }
    }
    println!("{name:10} endo {err:.2e}  exo (incl. endogenized shocks) {errx:.2e}");
    assert!(err < 1e-9 && errx < 1e-9, "{name}: endo {err} exo {errx}");
}

#[test]
fn controlled_paths_newton() {
    check("newton", 1);
}

#[test]
fn controlled_paths_linear_approximation() {
    check("linear", 2);
}

#[test]
fn controlled_paths_homotopy() {
    check("homotopy", 3);
}

#[test]
fn controlled_paths_expectation_errors() {
    check("learnt", 4);
}
