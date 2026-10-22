//! Estimation completion: `estimate_initial_states_endogenous_prior` against Dynare 7.2
//! (reference/est_complete/init_states, dump_is.m): parameter layout, priors and bounds of the
//! initial states, dsge_likelihood's fval, set_init_state.m and the smoothed initial state of
//! draw_init_state_from_smoother.m.

use std::path::PathBuf;

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::Objective;
use dynare::estimation::init_states::{augment, set_init_state, smoothed_initial_state};
use dynare::kalman::KalmanOptions;
use dynare::modfile::parse_mod_file;
use serde_json::Value;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/est_complete/init_states")
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn maxrel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1.0)).fold(0.0, f64::max)
}

#[test]
fn init_states_against_dynare() {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("ref_is.json")).unwrap()).unwrap();
    let mf = parse_mod_file(&dir().join("est_is.mod"), &[]).unwrap();
    let mf = augment(&mf).unwrap();
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let names: Vec<String> = r["names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(dsge.estimated_names, names);
    assert!(maxrel(&dsge.init_vector().unwrap(), &vecf(&r["x_init"])) < 1e-12);
    // (the prior-quantile bounds of the other parameters agree to ~1e-9: special functions)
    assert!(maxrel(&dsge.bounds.lb, &vecf(&r["lb"])) < 1e-8 && maxrel(&dsge.bounds.ub, &vecf(&r["ub"])) < 1e-8);
    let p1: Vec<f64> = dsge.priors.iter().map(|p| p.mean).collect();
    assert!(maxrel(&p1, &vecf(&r["p1"])) < 1e-12);
    let table = Table::read_csv(&dir().join("is_data.csv")).unwrap();
    let kopts = KalmanOptions { lik_init: 2, ..Default::default() };
    let data = LikelihoodData::new(&table.sample(&mf.varobs, 1, Some(120)).unwrap(), kopts);
    let post = DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let x = vecf(&p["x"]);
        let f = post.eval(&x).0;
        let fr = p["fval"].as_f64().unwrap();
        eprintln!("point {k}: fval {f:.12} vs {fr:.12}");
        assert!((f - fr).abs() < 1e-9 * fr.abs(), "point {k}: fval {f} vs {fr}");
        let ys0 = dynare::estimation::init_states::InitStateOps::steady_state_at(&post, &x).unwrap();
        let (xc, icheck) = set_init_state(&post, &x, &ys0);
        assert_eq!(icheck, p["icheck"].as_bool().unwrap(), "point {k}: icheck");
        let e = maxrel(&xc, &vecf(&p["set_init_state"]));
        eprintln!("point {k}: set_init_state rel err {e:.2e}");
        assert!(e < 1e-9, "point {k}: {xc:?} vs {:?}", vecf(&p["set_init_state"]));
        let ys1 =
            dynare::estimation::init_states::InitStateOps::steady_state_at(&post, &vecf(&r["points"][0]["x"])).unwrap();
        let (xc1, icheck1) = set_init_state(&post, &x, &ys1);
        assert_eq!(icheck1, p["icheck_ys1"].as_bool().unwrap());
        assert!(maxrel(&xc1, &vecf(&p["set_init_state_ys1"])) < 1e-9);
        let (a0, v0) = smoothed_initial_state(&post, &x).unwrap();
        let ea = maxrel(&a0, &vecf(&p["alphahat0"]));
        let vr = vecf(&p["V0"]);
        let ev = v0.iter().zip(&vr).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
            / vr.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        eprintln!("point {k}: alphahat0 rel err {ea:.2e}, V0 rel err {ev:.2e}");
        assert!(ea < 1e-8 && ev < 1e-8, "point {k}: alphahat0 {a0:?} vs {:?}", vecf(&p["alphahat0"]));
    }
}

/// `estimated_params_init(use_calibration)` with the initial states (Dynare 7.2's
/// do_parameter_initialization.m: block six of xparam1; 7.1 crashed), ref_is_usecal.json.
#[test]
fn init_states_use_calibration() {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("ref_is_usecal.json")).unwrap()).unwrap();
    let mf = parse_mod_file(&dir().join("est_is.mod"), &[("USECAL".into(), "true".into())]).unwrap();
    assert!(mf.estimated_params_init_use_calibration);
    let mf = augment(&mf).unwrap();
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let names: Vec<String> = r["names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(dsge.estimated_names, names);
    let x = dsge.init_vector().unwrap();
    assert!(maxrel(&x, &vecf(&r["x_init"])) < 1e-12, "{x:?} vs {:?}", vecf(&r["x_init"]));
}

/// The slice sampler with the initial states (draw_init_state_from_smoother.m, set_init_state.m
/// in the slice steps) on est_is.mod, 2000 draws: posterior means within half a posterior
/// standard deviation of Dynare's (4000 draws, ref_mcmc.json; different random streams).
#[test]
fn init_states_slice_mcmc() {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("ref_mcmc.json")).unwrap()).unwrap();
    let tmp = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp/est_is_mcmc");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["est_is.mod", "is_model.inc", "is_data.csv"] {
        std::fs::copy(dir().join(f), tmp.join(f)).unwrap();
    }
    let ro = dynare::driver::RunOptions {
        quiet: true,
        seed: 3,
        defines: vec![("MH".into(), "2000".into())],
        ..Default::default()
    };
    dynare::driver::run(&tmp.join("est_is.mod"), &ro).unwrap();
    let res: Value =
        serde_json::from_str(&std::fs::read_to_string(tmp.join("est_is/Output/est_is_results.json")).unwrap()).unwrap();
    let names: Vec<&str> = res["parameter_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let want: Vec<&str> = r["names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(names, want);
    let (m, mr, sr) = (vecf(&res["mcmc"]["posterior_mean"]), vecf(&r["mean"]), vecf(&r["std"]));
    for i in 0..m.len() {
        eprintln!("{}: {:.5} vs Dynare {:.5} (sd {:.5})", names[i], m[i], mr[i], sr[i]);
        assert!(
            (m[i] - mr[i]).abs() < 0.5 * sr[i],
            "{}: posterior mean {} vs {} (sd {})",
            names[i],
            m[i],
            mr[i],
            sr[i]
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
