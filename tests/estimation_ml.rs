//! Maximum-likelihood estimation (estimated_params without priors) end to end against
//! Dynare 7.1: reference/ml/model_ml.mod (stationary, lik_init = 1, measurement error, missing
//! values), mode_compute = 4; reference/ml/ref.json from reference/ml/dump_reference.m.
mod common;

use std::path::Path;

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::Objective;
use dynare::kalman::KalmanOptions;
use dynare::modfile::parse_mod_file;
use serde_json::Value;

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

#[test]
fn ml_estimation_against_dynare() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/ml");
    let text = common::read_ref(src.join("ref.json"));
    let r: Value = serde_json::from_str(&text).unwrap();

    // likelihood at Dynare's points
    let mf = parse_mod_file(&src.join("model_ml.mod"), &[]).unwrap();
    let dsge = DsgeModel::new(&mf, None).unwrap();
    assert!(dsge.ml);
    let x_init = vecf(&r["xparam_init"]);
    assert_eq!(dsge.init_vector().unwrap(), x_init);
    let lb = vecf(&r["bounds"]["lb"]);
    assert_eq!(dsge.bounds.lb, lb);
    let table = Table::read_csv(&src.join("model_ml_obs.csv")).unwrap();
    let data = LikelihoodData::new(&table.sample(&mf.varobs, 1, None).unwrap(), KalmanOptions::default());
    let post = DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
    for p in r["points"].as_array().unwrap() {
        let (f, info) = post.eval(&vecf(&p["x"]));
        match p["minus_loglik"].as_f64() {
            Some(v) => assert!(((f - v) / v).abs() < 1e-12, "{f} vs {v}"),
            None => assert_eq!(info as i64, p["info"].as_i64().unwrap()),
        }
    }

    // the whole estimation command, in a scratch copy (outputs go next to the .mod)
    let dir = std::env::temp_dir().join(format!("dynare_rs_ml_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["model_ml.mod", "model_ml_obs.csv"] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    let ro = dynare::driver::RunOptions { quiet: true, ..Default::default() };
    dynare::driver::run(&dir.join("model_ml.mod"), &ro).unwrap();
    let res: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("model_ml/Output/model_ml_results.json")).unwrap())
            .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let names: Vec<&str> = res["parameter_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let want_names: Vec<&str> = r["parameter_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(names, want_names);
    let (xm, xd) = (vecf(&res["mode"]), vecf(&r["xparam_mode"]));
    let fm = res["minus_log_posterior_at_mode"].as_f64().unwrap();
    let fd = r["fval_mode"].as_f64().unwrap();
    let dx = xm.iter().zip(&xd).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("ML mode: fval {fm:.10} vs Dynare {fd:.10}, max |dx| {dx:.1e}");
    assert!((fm - fd).abs() < 1e-8 && dx < 1e-6);
    // standard errors: oo_.mle_std_at_mode (Dynare's display order: parameters, shocks, MEs)
    let sd = vecf(&res["mode_sd"]);
    let s = &r["std_at_mode"];
    let want = [
        s["shocks_std"]["ex"].as_f64().unwrap(),
        s["shocks_std"]["ez"].as_f64().unwrap(),
        s["measurement_errors_std"]["x"].as_f64().unwrap(),
        s["parameters"]["a"].as_f64().unwrap(),
        s["parameters"]["b"].as_f64().unwrap(),
        s["parameters"]["c"].as_f64().unwrap(),
    ];
    for (a, b) in sd.iter().zip(want) {
        assert!(((a - b) / b).abs() < 1e-5, "s.d. {a} vs {b}");
    }
}
