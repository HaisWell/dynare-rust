//! `initval_file` / `histval_file` against Dynare 7.2 (reference/ep/ref_datafiles_*.json,
//! written by reference/ep/dump_datafiles.m) on a model with two lags (an auxiliary lag
//! variable) and an exogenous lead: CSV and xlsx initval files (periods from the file,
//! `first_obs`/`last_obs`), a CSV histval file with dates for a perfect-foresight simulation and
//! for an extended path (on MATLAB's shocks).

mod common;
use std::path::{Path, PathBuf};

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

fn max_diff(got: &Mat, want: &[Vec<f64>]) -> f64 {
    assert_eq!((got.nrows(), got.ncols()), (want.len(), want[0].len()), "shape");
    let mut e = 0.0f64;
    for (i, r) in want.iter().enumerate() {
        for (j, w) in r.iter().enumerate() {
            e = e.max((got[(i, j)] - w).abs());
        }
    }
    e
}

fn check(name: &str, case: u32) {
    let p = ref_dir().join(format!("ref_datafiles_{name}.json"));
    let r: Value = common::json_ref(&p);
    let mf = parse_mod_file(&ref_dir().join("datafiles.mod"), &[("CASE".into(), case.to_string())]).unwrap();
    let names: Vec<String> =
        r["endo_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(names.len(), mf.model.n_endo(), "{name}: endo_names {names:?} vs {:?}", mf.model.endo_names);
    let mut det = DetState::new(&mf).unwrap();
    let mut ep = None;
    for c in &mf.commands {
        match c {
            Command::Steady(o) => {
                let mut o = o.clone();
                o.push(("noprint".into(), OptValue::Flag));
                det.steady(&mf.model, &o).unwrap()
            }
            Command::Other(n, o) if n == "initval_file" => {
                dynare::ep::datafiles::initval_file(&mf, &mut det, o, &ref_dir()).unwrap()
            }
            Command::Other(n, o) if n == "histval_file" => {
                dynare::ep::datafiles::histval_file(&mf, &mut det, o, &ref_dir()).unwrap()
            }
            Command::Other(n, o) if n == "extended_path" => {
                let mut eo = EpOptions::default();
                eo.update(o).unwrap();
                ep = Some(eo);
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
            _ => {}
        }
    }
    if let Some(eo) = ep {
        let (eh, _) = det.histval_file.clone().expect("histval_file");
        let want_h = r["endo_histval"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
        for (j, w) in want_h.iter().enumerate() {
            assert!((eh[(j, 0)] - w).abs() < 1e-14, "{name}: endo_histval");
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
        let init: Vec<f64> = eh.column(0).iter().copied().collect();
        let out = extended_path(&m, &eo, &init, &shocks).unwrap();
        let e = max_diff(&out.endo_simul, &rows(&r["endo_simul"]));
        println!("{name:14} extended path paths {e:.2e}");
        assert!(e < 1e-9, "{name}: {e}");
        return;
    }
    assert_eq!(det.simul.periods as f64, r["periods"].as_f64().unwrap(), "{name}: periods");
    let e = max_diff(det.endo_simul.as_ref().unwrap(), &rows(&r["endo_simul"]));
    let ex = max_diff(det.exo_simul.as_ref().unwrap(), &rows(&r["exo_simul"]));
    println!("{name:14} endo_simul {e:.2e} exo_simul {ex:.2e}");
    assert!(e < 1e-9 && ex < 1e-14, "{name}: endo {e} exo {ex}");
}

#[test]
fn initval_file_csv() {
    check("initval_csv", 1);
}

#[test]
fn initval_file_xlsx_first_obs_last_obs() {
    check("initval_xlsx", 2);
}

#[test]
fn histval_file_perfect_foresight() {
    check("histval", 3);
}

#[test]
fn histval_file_extended_path() {
    check("histval_ep", 4);
}

// ---- Dynare 7.2: histval_file reads exactly M_.orig_maximum_lag observations ----

#[test]
fn histval_file_long_file_first_rows() {
    check("histval_long", 5);
}

#[test]
fn histval_file_first_obs_only() {
    check("histval_first_obs", 6);
}

#[test]
fn histval_file_last_obs_only() {
    check("histval_last_obs", 7);
}

/// The error of CASE `case`, from the parser or from the histval_file / initval_file command.
fn error_of(case: u32) -> String {
    let mf = match parse_mod_file(&ref_dir().join("datafiles.mod"), &[("CASE".into(), case.to_string())]) {
        Ok(mf) => mf,
        Err(e) => return format!("{e:#}"),
    };
    let mut det = DetState::new(&mf).unwrap();
    for c in &mf.commands {
        let r = match c {
            Command::Other(n, o) if n == "initval_file" => {
                dynare::ep::datafiles::initval_file(&mf, &mut det, o, &ref_dir())
            }
            Command::Other(n, o) if n == "histval_file" => {
                dynare::ep::datafiles::histval_file(&mf, &mut det, o, &ref_dir())
            }
            _ => Ok(()),
        };
        if let Err(e) = r {
            return format!("{e:#}");
        }
    }
    panic!("case {case}: no error");
}

/// Error messages of Dynare 7.2 (reference/ep/ref_datafiles_errors.json): the range errors
/// of histval_file verbatim (MATLAB's recycled format included), the removed options as
/// preprocessor syntax errors, and the failed assertion of initval_file with nobs.
#[test]
fn histval_file_errors_match_dynare_7_2() {
    let r: Value = common::json_ref(ref_dir().join("ref_datafiles_errors.json"));
    for e in r.as_array().unwrap() {
        let case = e["case"].as_u64().unwrap() as u32;
        let want = e["message"].as_str().unwrap();
        let got = error_of(case);
        match case {
            // MATLAB: "Dynare: preprocessing failed"; the preprocessor printed
            // "ERROR: datafiles.mod: line N, cols A-B: syntax error, unexpected NOBS"
            11 => {
                assert!(want.contains("preprocessing failed") && got.contains("syntax error, unexpected NOBS"), "{got}")
            }
            12 => assert!(
                want.contains("preprocessing failed")
                    && got.contains("syntax error, unexpected LAST_SIMULATION_PERIOD"),
                "{got}"
            ),
            _ => assert_eq!(got, want, "case {case}"),
        }
    }
}
