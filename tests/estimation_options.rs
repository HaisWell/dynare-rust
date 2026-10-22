//! Estimation options against Dynare 7.2 (reference/opts/dump_reference.m):
//! `loglinear` (data in levels or `logdata`), `endogenous_prior`, `prefilter`.
//! model_rbc.mod is a small nonlinear RBC (numerical steady state at every evaluation);
//! prefilter + endogenous_prior use reference/ml/model_ml.mod -DPREFILTER.
//! Checked: dsge_likelihood fval at three points, the decision rule (loglinear), and the
//! smoothed variables the estimation command writes (smoother at the initial values).
mod common;

use std::path::{Path, PathBuf};

use dynare::linalg::Mat;
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference")
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

/// Runs the .mod's estimation command in a scratch copy, then checks fval at Dynare's points
/// with the same data handling, and the smoothed variables CSV.
fn check(dir: &str, files: &[&str], fname: &str, define: (&str, &str), refname: &str) {
    let src = root().join(dir);
    let text = common::read_ref(root().join("opts").join(refname));
    let r: Value = serde_json::from_str(&text).unwrap();
    let defines = vec![(define.0.to_string(), define.1.to_string())];
    let mf = dynare::modfile::parse_mod_file(&src.join(format!("{fname}.mod")), &defines).unwrap();
    let est = mf
        .commands
        .iter()
        .find_map(|c| if let dynare::modfile::Command::Estimation(o, _) = c { Some(o.clone()) } else { None })
        .unwrap();
    let has = |k: &str| dynare::modfile::opt(&est, k).is_some();
    let x0 = vecf(&r["xparam_init"]);
    let dsge = dynare::dsge::DsgeModel::new(&mf, Some(&x0)).unwrap();
    assert_eq!(dsge.loglinear, has("loglinear"));
    let dys = dsge.ys_guess.iter().zip(vecf(&r["steady_state"])).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(dys < 1e-10, "steady state differs by {dys}");
    let res = dsge.resolve(&x0).unwrap();
    let (dghx, dghu) = ((&res.dr.ghx - mat(&r["ghx"])).amax(), (&res.dr.ghu - mat(&r["ghu"])).amax());
    assert!(dghx < 1e-10 && dghu < 1e-10, "ghx {dghx:e} ghu {dghu:e}");

    // data as makedataset.m prepares it
    let datafile = match dynare::modfile::opt(&est, "datafile") {
        Some(dynare::modfile::OptValue::Str(s)) => s.clone(),
        _ => panic!(),
    };
    let table = dynare::data::Table::read(&src.join(&datafile), &Default::default()).unwrap();
    let mut sample = table.sample(&mf.varobs, 1, None).unwrap();
    if has("loglinear") && !has("logdata") {
        sample.iter_mut().for_each(|v| *v = v.ln());
    }
    let mut data = dynare::dsge::LikelihoodData::new(&sample, Default::default());
    if has("prefilter") {
        data.prefilter();
    }
    data.endogenous_prior = has("endogenous_prior");
    let ddata =
        (&data.y.transpose() - mat(&r["data"])).iter().filter(|v| !v.is_nan()).fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(ddata < 1e-14, "data differ by {ddata}");
    let post = dynare::dsge::DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let x = vecf(&p["x"]);
        let f = dynare::estimation::Objective::eval(&post, &x).0;
        let v = p["minus_loglik"].as_f64().unwrap();
        println!("{refname} point {k}: {f:.10} vs Dynare {v:.10} (rel {:.1e})", ((f - v) / v).abs());
        assert!(((f - v) / v).abs() < 1e-10, "{refname} point {k}");
    }

    // the estimation command (mode_compute=0, no mode file: smoother at the initial values)
    let tmp = std::env::temp_dir().join(format!("dynare_rs_opts_{}_{refname}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    for f in files {
        std::fs::copy(src.join(f), tmp.join(f)).unwrap();
    }
    let ro = dynare::driver::RunOptions { defines, quiet: true, ..Default::default() };
    dynare::driver::run(&tmp.join(format!("{fname}.mod")), &ro).unwrap();
    let csv =
        dynare::data::Table::read_csv(&tmp.join(format!("{fname}/Output/{fname}_smoothed_variables.csv"))).unwrap();
    let _ = std::fs::remove_dir_all(&tmp);
    let mut worst: f64 = 0.0;
    for (j, name) in csv.names.iter().enumerate() {
        let want = vecf(&r["SmoothedVariables"][name]);
        for (t, w) in want.iter().enumerate() {
            worst = worst.max((csv.values[(t, j)] - w).abs() / w.abs().max(1.0));
        }
    }
    println!("{refname}: smoothed variables {worst:.1e}");
    assert!(worst < 1e-9);
}

#[test]
fn loglinear_levels() {
    check("opts", &["model_rbc.mod", "rbc_levels.csv"], "model_rbc", ("STAGE", "1"), "ref_rbc_1.json");
}

#[test]
fn loglinear_logdata() {
    check("opts", &["model_rbc.mod", "rbc_logs.csv"], "model_rbc", ("STAGE", "2"), "ref_rbc_2.json");
}

#[test]
fn loglinear_endogenous_prior() {
    check("opts", &["model_rbc.mod", "rbc_levels.csv"], "model_rbc", ("STAGE", "3"), "ref_rbc_3.json");
}

#[test]
fn prefilter_endogenous_prior() {
    check("ml", &["model_ml.mod", "model_ml_obs.csv"], "model_ml", ("PREFILTER", ""), "ref_ml_prefilter.json");
}
