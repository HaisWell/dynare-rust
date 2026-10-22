//! Measurement errors (`stderr`/`corr` on observed variables in `shocks` and
//! `estimated_params`, `M_.H` in dsge_likelihood and DsgeSmoother) against Dynare 7.2:
//! reference/me/sw07_me.mod (reference/sw07/sw07.mod + measurement errors), ref_10.json (diagonal
//! H) and ref_11.json (correlated measurement errors: the univariate filters and smoother
//! augment the state vector), written by reference/me/dump_reference.m.
mod common;

use std::path::{Path, PathBuf};

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::Objective;
use dynare::kalman::KalmanOptions;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/me")
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn mat(v: &Value) -> Mat {
    let rows = v.as_array().unwrap();
    let c = rows[0].as_array().map_or(1, |x| x.len());
    Mat::from_fn(rows.len(), c, |i, j| rows[i].as_array().map_or_else(|| vecf(&rows[i])[0], |x| vecf(&x[j])[0]))
}

fn check(stage: u32) {
    let text = common::read_ref(dir().join(format!("ref_{stage}.json")));
    let r: Value = serde_json::from_str(&text).unwrap();
    let mf = parse_mod_file(&dir().join("sw07_me.mod"), &[("STAGE".into(), stage.to_string())]).unwrap();
    let names: Vec<String> =
        r["estim_param_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().into()).collect();
    let x_init = vecf(&r["xparam_init"]);
    let dsge = DsgeModel::new(&mf, Some(&x_init)).unwrap();
    assert_eq!(dsge.estimated_names, names);
    let x0 = dynare::dsge::initial_vector(&mf).unwrap();
    for (k, (a, b)) in x0.iter().zip(&x_init).enumerate() {
        assert!((a - b).abs() < 1e-14, "init {}: {a} vs {b}", names[k]);
    }
    let dh = (&dsge.set_all_parameters_h(&x_init).2 - mat(&r["H_calib"])).amax(); // M_.H after dynare_estimation_init
    assert!(dh < 1e-15, "calibrated H differs by {dh}");

    let table = Table::read_csv(&dir().join("sw07_obs.csv")).unwrap();
    let sample = table.sample(&mf.varobs, 5, None).unwrap();
    let data = LikelihoodData::new(&sample, KalmanOptions { presample: 8, lik_init: 3, ..Default::default() });
    let post = DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let x = vecf(&p["x"]);
        let (_, _, h) = dsge.set_all_parameters_h(&x);
        assert!((&h - mat(&p["H"])).amax() < 1e-14, "H at point {k}");
        let (f, info) = post.eval(&x);
        let Some(v) = p["minus_loglik"].as_f64() else {
            // Dynare rejects the point (fval = Inf; stage 11 point 4 is above a bound: info 42)
            let want = p["info"].as_i64().unwrap();
            println!("stage {stage} point {k}: {f} (info {info}) vs Dynare Inf (info {want})");
            assert!(f.is_infinite() && info as i64 == want, "stage {stage} point {k}");
            continue;
        };
        let rel = ((f - v) / v).abs();
        println!("stage {stage} point {k}: {f:.10} vs Dynare {v:.10} (rel {rel:.1e}, info {info})");
        assert!(rel < 1e-9, "stage {stage} point {k}");
    }

    let xm = vecf(&r["xparam_mode"]);
    let t = std::time::Instant::now();
    for _ in 0..50 {
        std::hint::black_box(post.eval(&xm));
    }
    println!(
        "stage {stage}: {:.3} ms per posterior evaluation (MATLAB: {:.1} ms)",
        t.elapsed().as_secs_f64() * 1e3 / 50.0,
        r["timing_seconds_per_lik"].as_f64().unwrap() * 1e3
    );

    // smoother at the initial point (with H)
    let sm = dynare::driver::run_smoother(&dsge, &data, &x_init, false).unwrap();
    let mut worst: f64 = 0.0;
    for (v, name) in mf.model.endo_names[..mf.model.orig_endo_nbr].iter().enumerate() {
        let Some(want) = r["SmoothedVariables_init"].get(name) else { continue };
        for (t, w) in vecf(want).iter().enumerate() {
            worst = worst.max((sm.smoothed[(t, v)] - w).abs() / w.abs().max(1.0));
        }
    }
    let mut worst_shocks: f64 = 0.0;
    for (k, name) in mf.model.exo_names.iter().enumerate() {
        for (t, w) in vecf(&r["SmoothedShocks_init"][name]).iter().enumerate() {
            worst_shocks = worst_shocks.max((sm.shocks[(t, k)] - w).abs() / w.abs().max(1.0));
        }
    }
    let mut worst_me: f64 = 0.0;
    let me = sm.measurement_errors.as_ref().expect("smoothed measurement errors");
    for (i, name) in mf.varobs.iter().enumerate() {
        let Some(want) = r["SmoothedMeasurementErrors_init"].get(name) else { continue };
        for (t, w) in vecf(want).iter().enumerate() {
            if w.is_finite() {
                worst_me = worst_me.max((me[(t, i)] - w).abs() / w.abs().max(1.0));
            }
        }
    }
    println!(
        "stage {stage} smoother: variables {worst:.1e} shocks {worst_shocks:.1e} measurement errors {worst_me:.1e}"
    );
    assert!(worst < 1e-8 && worst_shocks < 1e-8 && worst_me < 1e-8);
}

#[test]
fn sw07_diagonal_measurement_errors() {
    check(10);
}

#[test]
fn sw07_correlated_measurement_errors() {
    check(11);
}
