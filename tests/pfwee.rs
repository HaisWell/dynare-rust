//! Perfect foresight with expectation errors against Dynare 7.2 (reference/ep/ref_pfwee_*.json,
//! written by reference/ep/dump_pfwee.m): `shocks`/`mshocks`/`endval` with `learnt_in`
//! (values, add, multiply, `+=`, `*=`, overwrite), `constant_simulation_length`, a CSV
//! `datafile`, and a regular `endval` block revised later.

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
    let p = ref_dir().join(format!("ref_pfwee_{name}.json"));
    let r: Value = common::json_ref(&p);
    let mf = parse_mod_file(&ref_dir().join("pfwee.mod"), &[("CASE".into(), case.to_string())]).unwrap();
    let out = std::env::temp_dir().join(format!("dynare_rs_pfwee_{name}"));
    std::fs::create_dir_all(&out).unwrap();
    let mut det = DetState::new(&mf).unwrap();
    for (ci, c) in mf.commands.iter().enumerate() {
        match c {
            Command::Steady(o) => {
                let mut o = o.clone();
                o.push(("noprint".into(), OptValue::Flag));
                det.steady(&mf.model, &o).unwrap()
            }
            Command::Endval(v) => det.endval(&mf.model, v).unwrap(),
            Command::Other(n, o) if n == "perfect_foresight_with_expectation_errors_setup" => {
                dynare::ep::pfwee::setup(&mf, &mut det, o, &ref_dir()).unwrap();
                det.simul.noprint = true;
                det.simul.verbosity = 0;
            }
            Command::Other(n, _) if dynare::ep::handles(n) => {
                dynare::ep::command(&mf, &mut det, ci, &out, "pfwee", 0).unwrap();
            }
            _ => {}
        }
    }
    let want = rows(&r["endo_simul"]);
    let got = det.endo_simul.as_ref().unwrap();
    assert_eq!(got.ncols(), want[0].len(), "{name}: endo_simul columns");
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
    let ss: Vec<f64> = r["steady_state"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let errss = ss.iter().zip(&det.steady_state).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("{name:10} endo {err:.2e}  exo {errx:.2e}  terminal steady state {errss:.2e}");
    assert!(err < 1e-9 && errx < 1e-12 && errss < 1e-9, "{name}: endo {err} exo {errx} ss {errss}");
}

#[test]
fn pfwee_learnt_in_blocks() {
    check("learnt", 1);
}

#[test]
fn pfwee_constant_simulation_length() {
    check("constant", 2);
}

#[test]
fn pfwee_datafile() {
    check("datafile", 3);
}

#[test]
fn pfwee_with_endval() {
    check("endval", 4);
}
