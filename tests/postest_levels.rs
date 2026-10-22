//! Post-estimation objects under `loglinear` and `prefilter`, through the driver, against
//! Dynare 7.1 (reference/opts/dump_postest.m -> reference/opts/ref_postest.json):
//!   - reference/opts/model_rbc.mod -DSTAGE=4 (loglinear, mode file): `forecast=6` of the
//!     estimation (forecasts.run 'smoother': forecasts in logs), shock_decomposition, the
//!     decomposition over that forecast (`forecast_type=unconditional`), conditional_forecast at
//!     the posterior mode (paths without free shocks);
//!   - STAGE=5: the `forecast` command (forecasts.run 'simul': log(ys)) and
//!     conditional_forecast(parameter_set=calibration);
//!   - reference/ml/model_ml.mod -DPOSTEST=1 (prefilter=1, ML, mode file): `forecast=6` (mean
//!     moved to the sample mean, bands not: Dynare's quirk), shock decompositions,
//!     realtime_shock_decomposition (vintages demeaned with their own sample mean),
//!     conditional_forecast(parameter_set=calibration);
//!   - POSTEST=2/3/4: the errors Dynare raises (forecast command, conditional forecast at an
//!     estimated parameter set, forecast var_list without every observable).
mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dynare::driver::{RunOptions, run};
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn reference() -> Option<Value> {
    Some(common::json_ref(root().join("reference/opts/ref_postest.json")))
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn names(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

/// CSV -> rows of cells (header dropped).
fn csv(p: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect()
}

fn f(s: &str) -> f64 {
    s.parse().unwrap()
}

/// Runs `<fname>.mod` with `define` in a scratch copy of reference/<dir>; returns the scratch
/// dir and the driver's result.
fn run_mod(dir: &str, files: &[&str], fname: &str, define: (&str, &str)) -> (PathBuf, anyhow::Result<()>) {
    let src = root().join("reference").join(dir);
    let tmp =
        std::env::temp_dir().join(format!("dynare_rs_postest_{}_{fname}_{}{}", std::process::id(), define.0, define.1));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for f in files {
        std::fs::copy(src.join(f), tmp.join(f)).unwrap();
    }
    let ro = RunOptions { defines: vec![(define.0.into(), define.1.into())], quiet: true, ..Default::default() };
    let r = run(&tmp.join(format!("{fname}.mod")), &ro);
    (tmp, r)
}

/// Removes a scratch dir (kept with `KEEP=1`, to look at the outputs).
fn cleanup(tmp: &Path) {
    if std::env::var("KEEP").is_err() {
        let _ = std::fs::remove_dir_all(tmp);
    }
}

/// `<fname>_forecast.csv` against `oo_.forecast` (Mean, HPDinf, HPDsup).
fn check_forecast(path: &Path, r: &Value, nvars: usize, horizon: usize) -> f64 {
    let rows = csv(path);
    assert_eq!(rows.len(), nvars * horizon);
    let mut worst: f64 = 0.0;
    for row in &rows {
        let h: usize = row[1].parse().unwrap();
        for (k, field) in [(3, "Mean"), (4, "HPDinf"), (5, "HPDsup")] {
            let want = vecf(&r[field][&row[2]])[h - 1];
            worst = worst.max((f(&row[k]) - want).abs());
        }
    }
    worst
}

/// A decomposition CSV (`date,variable,component,value`) against MATLAB's z (flattened
/// column-major, size endo x (nshocks+2) x T); periods numbered by first appearance.
fn check_z(path: &Path, z: &Value, size: &Value, endo: &[String], exo: &[String]) -> f64 {
    let rows = csv(path);
    let size: Vec<usize> = vecf(size).iter().map(|&x| x as usize).collect();
    let z = vecf(z);
    let (nv, nc, nt) = (size[0], size[1], size[2]);
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
        worst = worst.max((f(&r[3]) - z[v + nv * (c + nc * t)]).abs());
    }
    assert_eq!(t_of.len(), nt, "{}: periods", path.display());
    worst
}

/// `<fname>_conditional_forecast.csv` (paths without free shocks) and the controlled shocks
/// against conditional_forecasts.run with randn shadowed by zeros.
fn check_conditional(out: &Path, fname: &str, cf0: &Value) -> f64 {
    let mut worst: f64 = 0.0;
    for r in csv(&out.join(format!("{fname}_conditional_forecast.csv"))) {
        let t: usize = r[1].parse().unwrap();
        worst = worst.max((f(&r[9]) - vecf(&cf0["cond"]["Mean"][&r[2]])[t]).abs());
        worst = worst.max((f(&r[10]) - vecf(&cf0["uncond"]["Mean"][&r[2]])[t]).abs());
    }
    for r in csv(&out.join(format!("{fname}_conditional_forecast_shocks.csv"))) {
        let t: usize = r[1].parse().unwrap();
        worst = worst.max((f(&r[6]) - vecf(&cf0["controlled_exo_variables"]["Mean"][&r[2]])[t - 1]).abs());
    }
    worst
}

/// Smoothed variables of the estimation against oo_.SmoothedVariables (log levels / sample means).
fn check_smoothed(out: &Path, fname: &str, r: &Value) -> f64 {
    let t = dynare::data::Table::read_csv(&out.join(format!("{fname}_smoothed_variables.csv"))).unwrap();
    let mut worst: f64 = 0.0;
    for (j, name) in t.names.iter().enumerate() {
        for (i, w) in vecf(&r["SmoothedVariables"][name]).iter().enumerate() {
            worst = worst.max((t.values[(i, j)] - w).abs());
        }
    }
    worst
}

const TOL: f64 = 1e-10;

#[test]
fn loglinear_rbc() {
    let r = reference().expect("reference/setup unavailable");
    let files = ["model_rbc.mod", "rbc_levels.csv", "rbc_mode.mat"];

    // STAGE 4: forecast=6, shock decompositions, conditional forecast at the mode
    let (tmp, res) = run_mod("opts", &files, "model_rbc", ("STAGE", "4"));
    res.unwrap();
    let out = tmp.join("model_rbc/Output");
    let s = &r["rbc4"];
    let (endo, exo) = (names(&s["endo_names"]), names(&s["exo_names"]));
    let w = check_smoothed(&out, "model_rbc", s);
    println!("loglinear: smoothed variables {w:.1e}");
    assert!(w < TOL);
    let w = check_forecast(&out.join("model_rbc_forecast.csv"), &s["forecast"], 4, 6);
    println!("loglinear: forecast=6 (in logs) {w:.1e}");
    assert!(w < TOL);
    let w = check_z(&out.join("model_rbc_shock_decomposition.csv"), &s["sd"], &s["sd_size"], &endo, &exo);
    println!("loglinear: shock_decomposition {w:.1e}");
    assert!(w < TOL);
    let w = check_z(
        &out.join("model_rbc_shock_decomposition_unconditional_forecast.csv"),
        &s["sd_fc"],
        &s["sd_fc_size"],
        &endo,
        &exo,
    );
    println!("loglinear: shock_decomposition(forecast_type=unconditional) {w:.1e}");
    assert!(w < TOL);
    let w = check_conditional(&out, "model_rbc", &s["cf0"]);
    println!("loglinear: conditional_forecast at the mode {w:.1e}");
    assert!(w < TOL);
    cleanup(&tmp);

    // STAGE 5: forecast command, conditional forecast at the calibration (= the mode)
    let (tmp, res) = run_mod("opts", &files, "model_rbc", ("STAGE", "5"));
    res.unwrap();
    let out = tmp.join("model_rbc/Output");
    let s = &r["rbc5"];
    let w = check_forecast(&out.join("model_rbc_forecast.csv"), &s["forecast"], 3, 5);
    println!("loglinear: forecast command {w:.1e}");
    assert!(w < TOL);
    let w = check_conditional(&out, "model_rbc", &s["cf0"]);
    println!("loglinear: conditional_forecast(parameter_set=calibration) {w:.1e}");
    assert!(w < TOL);
    cleanup(&tmp);
}

#[test]
fn prefilter_ml() {
    let r = reference().expect("reference/setup unavailable");
    let files = ["model_ml.mod", "model_ml_obs.csv", "ml_mode.mat"];
    let (tmp, res) = run_mod("ml", &files, "model_ml", ("POSTEST", "1"));
    res.unwrap();
    let out = tmp.join("model_ml/Output");
    let s = &r["ml1"];
    let (endo, exo) = (names(&s["endo_names"]), names(&s["exo_names"]));
    let w = check_smoothed(&out, "model_ml", s);
    println!("prefilter: smoothed variables {w:.1e}");
    assert!(w < TOL);
    let w = check_forecast(&out.join("model_ml_forecast.csv"), &s["forecast"], 2, 6);
    println!("prefilter: forecast=6 (mean + sample mean, bands around the steady state) {w:.1e}");
    assert!(w < TOL);
    let w = check_z(&out.join("model_ml_shock_decomposition.csv"), &s["sd"], &s["sd_size"], &endo, &exo);
    println!("prefilter: shock_decomposition {w:.1e}");
    assert!(w < TOL);
    let w = check_z(
        &out.join("model_ml_shock_decomposition_unconditional_forecast.csv"),
        &s["sd_fc"],
        &s["sd_fc_size"],
        &endo,
        &exo,
    );
    println!("prefilter: shock_decomposition(forecast_type=unconditional) {w:.1e}");
    assert!(w < TOL);
    let mut w = check_z(
        &out.join("model_ml_realtime_shock_decomposition_pool.csv"),
        &s["rt_pool"],
        &s["rt_pool_size"],
        &endo,
        &exo,
    );
    for t in [150, 200] {
        let k = format!("rt_time_{t}");
        let p = out.join(format!("model_ml_realtime_shock_decomposition_time_{t}.csv"));
        w = w.max(check_z(&p, &s[&k], &s[format!("{k}_size")], &endo, &exo));
        let k = format!("rtf_time_{t}");
        let p = out.join(format!("model_ml_realtime_forecast_shock_decomposition_time_{t}.csv"));
        w = w.max(check_z(&p, &s[&k], &s[format!("{k}_size")], &endo, &exo));
    }
    println!("prefilter: realtime_shock_decomposition (pool, vintages 150/200, forecasts) {w:.1e}");
    assert!(w < TOL);
    let w = check_conditional(&out, "model_ml", &s["cf0"]);
    println!("prefilter: conditional_forecast(parameter_set=calibration) {w:.1e}");
    assert!(w < TOL);
    cleanup(&tmp);

    // what Dynare 7.1 rejects under prefilter
    for k in 2..=4 {
        let (tmp, res) = run_mod("ml", &files, "model_ml", ("POSTEST", &k.to_string()));
        let err = format!("{:#}", res.expect_err("Dynare stops here"));
        let want = r[format!("ml{k}_error")].as_str().unwrap();
        println!("prefilter: POSTEST={k}: {err}");
        assert!(err.contains(want), "POSTEST={k}: {err:?} does not contain Dynare's {want:?}");
        cleanup(&tmp);
    }
}
