//! `smoother2histval` with the default invars against Dynare 7.2 (reference/s2h/ref_s2h.json
//! from reference/s2h/dump_s2h.m): the AUX_ENDO_LAG and AUX_EXO_LAG rows of `M_.endo_histval`
//! are set from the smoothed values, so the forecast from the histval (which reads the lagged
//! shock through AUX_EXO_LAG) matches MATLAB. Also after `calib_smoother` (s2h_calib.mod,
//! ref_s2h_calib.json), whose oo_.SmoothedVariables / SmoothedShocks smoother2histval reads.
mod common;

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

fn refdir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/s2h")
}

/// Runs `<name>.mod`, checks the histval message and the forecast against `ref_<name>.json`;
/// returns the work directory.
fn run_and_check_forecast(name: &str) -> (Value, PathBuf) {
    let r: Value = common::json_ref(refdir().join(format!("ref_{name}.json")));
    let work = common::temp_dir(name);
    for f in [format!("{name}.mod"), "s2h_data.csv".to_string()] {
        std::fs::copy(refdir().join(&f), work.join(&f)).unwrap();
    }
    let out =
        Command::new(env!("CARGO_BIN_EXE_dynare-rs")).arg(format!("{name}.mod")).current_dir(&work).output().unwrap();
    assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    // default invars: y z AUX_ENDO_LAG_0_1 AUX_EXO_LAG_3_0 (fieldnames of oo_.SmoothedVariables) + e u
    let fields = r["smoothed_fields"].as_array().unwrap().len();
    assert!(stdout.contains(&format!("({} variables)", fields + 2)), "{stdout}");
    let csv = std::fs::read_to_string(work.join(format!("{name}/Output/{name}_forecast.csv"))).unwrap();
    for var in ["y", "z"] {
        let want: Vec<f64> =
            r[format!("forecast_{var}")].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let got: Vec<f64> = csv
            .lines()
            .skip(1)
            .map(|l| l.split(',').collect::<Vec<_>>())
            .filter(|c| c[2] == var)
            .map(|c| c[3].parse().unwrap())
            .collect();
        assert_eq!(got.len(), want.len(), "{var}");
        for (h, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-10, "forecast {var}[{h}]: {a} vs {b}");
        }
    }
    (r, work)
}

#[test]
fn smoother2histval_default_invars_forecast() {
    run_and_check_forecast("s2h");
}

#[test]
fn smoother2histval_after_calib_smoother() {
    let (r, work) = run_and_check_forecast("s2h_calib");
    // oo_.SmoothedShocks.u of calib_smoother
    let csv = std::fs::read_to_string(work.join("s2h_calib/Output/s2h_calib_calib_smoothed_shocks.csv")).unwrap();
    let mut lines = csv.lines();
    let col = lines.next().unwrap().split(',').position(|c| c == "u").unwrap();
    let got: Vec<f64> = lines.map(|l| l.split(',').nth(col).unwrap().parse().unwrap()).collect();
    let want: Vec<f64> = r["smoothed_shock_u"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(got.len(), want.len());
    for (t, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-10, "SmoothedShocks.u[{t}]: {a} vs {b}");
    }
}
