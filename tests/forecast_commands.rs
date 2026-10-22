//! The `forecast`, `shock_decomposition` (with `shock_groups`) and `calib_smoother` commands
//! and the `forecast=H` estimation option, run through the driver on SW07 after the STAGE 9
//! estimation (tests/data/sw07_commands.mod), against Dynare 7.2 (ref_forecast.json,
//! ref_shock_decomp.json).
mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dynare::driver::{RunOptions, run};
use serde_json::Value;

fn load(name: &str) -> Option<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07").join(name);
    Some(common::json_ref(p))
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

/// Long-format CSV -> rows of cells (header dropped).
fn csv(p: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(p).unwrap().lines().skip(1).map(|l| l.split(',').map(str::to_string).collect()).collect()
}

fn setup() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let refdir = root.join("reference/sw07");
    let mode = refdir.join("sw07/Output/sw07_mode.mat");
    assert!(mode.exists(), "missing reference {}", mode.display());
    let dir = std::env::temp_dir().join(format!("dynare_rs_fc_{}", std::process::id()));
    std::fs::create_dir_all(dir.join("sw07/Output")).unwrap();
    for f in ["sw07.mod", "sw07_obs.csv"] {
        std::fs::copy(refdir.join(f), dir.join(f)).unwrap();
    }
    std::fs::copy(refdir.join("sw07/Output/sw07_mode.mat"), dir.join("sw07/Output/sw07_mode.mat")).unwrap();
    std::fs::copy(root.join("tests/data/sw07_commands.mod"), dir.join("sw07_commands.mod")).unwrap();
    Some(dir)
}

#[test]
fn commands_after_estimation() {
    let (dir, f, sd) = (
        setup().expect("reference/setup unavailable"),
        load("ref_forecast.json").expect("reference/setup unavailable"),
        load("ref_shock_decomp.json").expect("reference/setup unavailable"),
    );
    run(&dir.join("sw07_commands.mod"), &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let out = dir.join("sw07_commands/Output");

    // forecast(periods=8, conf_sig=0.8) overwrote the estimation's forecast=12 file
    let rows = csv(&out.join("sw07_commands_forecast.csv"));
    assert_eq!(rows.len(), 16);
    let mut worst: f64 = 0.0;
    for r in &rows {
        let h: usize = r[1].parse().unwrap();
        for (k, field) in [(3, "Mean"), (4, "HPDinf"), (5, "HPDsup")] {
            let refv = vecf(&f["forecast_simul"][field][&r[2]])[h - 1];
            worst = worst.max((r[k].parse::<f64>().unwrap() - refv).abs());
        }
    }
    println!("forecast command: {worst:e}");
    assert!(worst < 1e-9);

    // shock decomposition, per shock, against oo_.shock_decomposition
    let names =
        |k: &str| -> Vec<String> { sd[k].as_array().unwrap().iter().map(|s| s.as_str().unwrap().into()).collect() };
    let (endo, exo) = (names("endo_names"), names("exo_names"));
    let size: Vec<usize> = vecf(&sd["size"]).iter().map(|&x| x as usize).collect();
    let (nv, nc) = (size[0], size[1]);
    let z = vecf(&sd["z"]);
    let rows = csv(&out.join("sw07_commands_shock_decomposition.csv"));
    let mut t_of: HashMap<String, usize> = HashMap::new();
    let mut worst: f64 = 0.0;
    for r in &rows {
        let n = t_of.len();
        let t = *t_of.entry(r[0].clone()).or_insert(n);
        let v = endo.iter().position(|e| *e == r[1]).unwrap();
        let c = match r[2].as_str() {
            "Initial values" => exo.len(),
            "Smoothed" => exo.len() + 1,
            s => exo.iter().position(|e| e == s).unwrap(),
        };
        worst = worst.max((r[3].parse::<f64>().unwrap() - z[v + nv * (c + nc * t)]).abs());
    }
    println!("shock decomposition: {worst:e} ({} rows)", rows.len());
    assert_eq!(rows.len(), 2 * 91 * (exo.len() + 2));
    assert!(worst < 1e-9);

    // groups: Prices + demand + Others/initial values add up to the smoothed series
    let rows = csv(&out.join("sw07_commands_shock_decomposition_main.csv"));
    let mut sums: HashMap<(String, String), (f64, f64)> = HashMap::new();
    for r in &rows {
        let e = sums.entry((r[0].clone(), r[1].clone())).or_default();
        let v: f64 = r[3].parse().unwrap();
        if r[2] == "Smoothed" { e.1 = v } else { e.0 += v }
    }
    assert!(rows.iter().any(|r| r[2] == "Prices") && rows.iter().any(|r| r[2] == "Others + Initial Values"));
    let worst = sums.values().map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(worst < 1e-9, "groups do not add up: {worst}");

    // calib_smoother and the estimation's smoother files exist
    for f in ["calib_smoothed_variables", "calib_smoothed_shocks", "calib_filtered_variables", "smoothed_variables"] {
        assert!(out.join(format!("sw07_commands_{f}.csv")).exists(), "{f}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
