//! QPM (Berg-Karam-Laxton 2006) and GM05 (Gali-Monacelli 2005), the public benchmark models,
//! against Dynare 7.2: reference/{qpm,gm05}/ref.json, written by reference/<m>/dump_reference.m (STAGE 1 setup,
//! estimation patched to mode_compute=0). Steady state, priors/bounds, decision rule and
//! state space at the mode file's point, and dsge_likelihood fval at six points.
mod common;

use std::path::{Path, PathBuf};

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::Objective;
use dynare::kalman::KalmanOptions;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use serde_json::Value;

fn dir(m: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference").join(m)
}

fn load(m: &str) -> Option<Value> {
    Some(common::json_ref(dir(m).join("ref.json")))
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
    let r = rows.len();
    let c = rows[0].as_array().map_or(1, |x| x.len());
    Mat::from_fn(r, c, |i, j| rows[i].as_array().map_or_else(|| vecf(&rows[i])[0], |x| vecf(&x[j])[0]))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn check_model(m: &str) {
    let r = load(m).expect("reference/setup unavailable");
    let d = dir(m);
    let mf = parse_mod_file(&d.join(format!("model_{m}.mod")), &[("STAGE".into(), "1".into())]).unwrap();
    let names: Vec<String> =
        r["estim_param_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let x_init = vecf(&r["xparam_init"]);
    let xm = vecf(&r["xparam_mode"]);
    let dsge = DsgeModel::new(&mf, Some(&x_init)).unwrap();
    assert_eq!(dsge.estimated_names, names, "{m}: estimated parameter names / order");
    assert_eq!(dynare::dsge::initial_vector(&mf).unwrap().len(), x_init.len());
    let dx = max_abs_diff(&dynare::dsge::initial_vector(&mf).unwrap(), &x_init);
    assert!(dx < 1e-14, "{m}: initial vector differs by {dx}");

    let dys = max_abs_diff(&dsge.ys_guess, &vecf(&r["steady_state"]));
    println!("{m}: steady state at the initial point: max diff {dys:.1e}");
    assert!(dys < 1e-9, "{m}: steady state differs by {dys}");

    let lb: Vec<f64> =
        vecf(&r["bounds"]["lb"]).into_iter().map(|v| if v.is_nan() { f64::NEG_INFINITY } else { v }).collect();
    let ub: Vec<f64> =
        vecf(&r["bounds"]["ub"]).into_iter().map(|v| if v.is_nan() { f64::INFINITY } else { v }).collect();
    for k in 0..names.len() {
        assert!(dsge.bounds.lb[k] == lb[k] && dsge.bounds.ub[k] == ub[k], "{m} bounds {}", names[k]);
    }

    let res = dsge.resolve(&xm).unwrap();
    let dss = max_abs_diff(&res.steady_state, &vecf(&r["mode_dr"]["ys"]));
    let dt = (&res.t - mat(&r["mode_ss"]["T"])).amax();
    let dr = (&res.r - mat(&r["mode_ss"]["R"])).amax();
    let ghx = mat(&r["mode_dr"]["ghx"]);
    let dghx = (&res.dr.ghx - &ghx).amax();
    let dghu = (&res.dr.ghu - mat(&r["mode_dr"]["ghu"])).amax();
    println!("{m}: at the mode: ys {dss:.1e} ghx {dghx:.1e} ghu {dghu:.1e} T {dt:.1e} R {dr:.1e}");
    assert!(dss < 1e-9 && dghx < 1e-9 && dghu < 1e-9 && dt < 1e-9 && dr < 1e-9);

    let table = Table::read_csv(&d.join(format!("model_{m}_obs.csv"))).unwrap();
    let sample = table.sample(&mf.varobs, 1, None).unwrap();
    let ref_data = mat(&r["data"]);
    assert_eq!((sample.nrows(), sample.ncols()), (ref_data.nrows(), ref_data.ncols()));
    let data = LikelihoodData::new(&sample, KalmanOptions { lik_init: 3, ..Default::default() });
    let post = DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let x = vecf(&p["x"]);
        let (f, info) = post.eval(&x);
        match p["minus_loglik"].as_f64() {
            Some(v) => {
                let rel = ((f - v) / v).abs();
                println!("{m} point {k}: {f:.10} vs Dynare {v:.10}  (rel {rel:.1e})");
                assert!(rel < 1e-9, "{m} point {k}");
            }
            None => {
                println!("{m} point {k}: info {info} (Dynare info {})", p["info"]);
                assert!(!f.is_finite());
                assert_eq!(info as i64, p["info"].as_i64().unwrap());
            }
        }
    }
    // smoother at the initial point (what dynare_estimation_1 runs with mode_compute=0 and
    // no mode file): oo_.SmoothedVariables / SmoothedShocks
    let sm = dynare::driver::run_smoother(&dsge, &data, &x_init, false).unwrap();
    let mut worst: f64 = 0.0;
    for (v, name) in mf.model.endo_names[..mf.model.orig_endo_nbr].iter().enumerate() {
        let Some(want) = r["SmoothedVariables_init"].get(name) else { continue };
        let want = vecf(want);
        for (t, w) in want.iter().enumerate() {
            worst = worst.max((sm.smoothed[(t, v)] - w).abs() / w.abs().max(1.0));
        }
    }
    let mut worst_shocks: f64 = 0.0;
    for (k, name) in mf.model.exo_names.iter().enumerate() {
        let want = vecf(&r["SmoothedShocks_init"][name]);
        for (t, w) in want.iter().enumerate() {
            worst_shocks = worst_shocks.max((sm.shocks[(t, k)] - w).abs() / w.abs().max(1.0));
        }
    }
    println!("{m}: smoother at the initial point: variables {worst:.1e}, shocks {worst_shocks:.1e}");
    assert!(worst < 1e-8 && worst_shocks < 1e-8);

    let t = std::time::Instant::now();
    let n = 50;
    for _ in 0..n {
        std::hint::black_box(post.eval(&xm));
    }
    println!(
        "{m}: {:.3} ms per posterior evaluation (MATLAB: {:.1} ms)",
        t.elapsed().as_secs_f64() * 1e3 / n as f64,
        r["timing_seconds_per_lik"].as_f64().unwrap() * 1e3
    );
}

#[test]
fn qpm_against_dynare() {
    check_model("qpm");
}

#[test]
fn gm05_against_dynare() {
    check_model("gm05");
}
