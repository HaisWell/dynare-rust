//! : `prior moments(distribution) irfs(distribution)` and `prior optimize` on
//! reference/q2/rbc_q2.mod (CASE 9) against Dynare 7.2 (reference/q2/ref_q2_prior.json and
//! q2_c9_matlab.log, written by dump_prior.m). The prior draws are not MATLAB's: the implied
//! prior moments are compared within their Monte Carlo error; the optimized prior mode must
//! reach the prior mode.
mod common;

use std::path::Path;

use dynare::matfile::{MatValue, read};
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn vec_of(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vec_of).collect(),
        x => vec![x.as_f64().unwrap_or(f64::NAN)],
    }
}

fn numeric(v: &MatValue) -> (Vec<usize>, Vec<f64>) {
    match v {
        MatValue::Numeric { dims, data } => (dims.clone(), data.clone()),
        _ => panic!("not numeric"),
    }
}

#[test]
fn prior_distribution_and_optimize() {
    let r: Value = common::json_ref(root().join("reference/q2/ref_q2_prior.json"));
    let dir = std::env::temp_dir().join(format!("dynare_q2_prior_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(root().join("reference/q2/rbc_q2.mod"), dir.join("rbc_q2.mod")).unwrap();
    let ro = dynare::driver::RunOptions {
        seed: 0,
        quiet: true,
        defines: vec![("CASE".into(), "9".into())],
        ..Default::default()
    };
    dynare::driver::run(&dir.join("rbc_q2.mod"), &ro).unwrap();

    // implied prior moments: FirstOrderMoments (7 x 20000), SecondOrderMoments (7 x 7 x 20000)
    let m = read(&dir.join("rbc_q2/prior/rbc_q2_endogenous_variables_prior_draws.mat")).unwrap();
    let (d1, f) = numeric(&m["FirstOrderMoments"]);
    let (n, nmc) = (d1[0], d1[1]);
    assert_eq!((n, nmc), (7, 20000));
    let keep: Vec<usize> = (0..nmc).filter(|&j| (0..n).all(|i| !f[j * n + i].is_nan())).collect();
    let want_mean = vec_of(&r["mean_of_mean"]);
    let want_var = vec_of(&r["var_of_mean"]);
    for i in 0..n {
        let x: Vec<f64> = keep.iter().map(|&j| f[j * n + i]).collect();
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
        let se = (var.max(want_var[i]) / x.len() as f64).sqrt();
        assert!((mean - want_mean[i]).abs() <= 4.0 * se + 1e-12, "E[mean] of {i}: {mean} vs {}", want_mean[i]);
        if want_var[i] > 1e-16 {
            assert!(
                (var.sqrt() / want_var[i].sqrt() - 1.0).abs() < 0.1,
                "sd of mean {i}: {} vs {}",
                var.sqrt(),
                want_var[i].sqrt()
            );
        }
    }
    let (_, v) = numeric(&m["SecondOrderMoments"]);
    let want = vec_of(&r["mean_of_var"]);
    let want_vv = vec_of(&r["var_of_var"]);
    for a in 0..n {
        for b in 0..n {
            // MATLAB arrays come row-major from jsonencode
            let (w, wv) = (want[a * n + b], want_vv[a * n + b]);
            let x: Vec<f64> = keep.iter().map(|&j| v[j * n * n + b * n + a]).collect();
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            let var = x.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / x.len() as f64;
            // heavy tails (inverse gamma priors with infinite variance): 5 standard errors
            let se = (var.max(wv) / x.len() as f64).sqrt();
            assert!((mean - w).abs() <= 5.0 * se + 1e-12, "E[var]({a},{b}): {mean} vs {w} (se {se})");
        }
    }
    // implied prior IRFs to ea (7 x 8 x 20000)
    let MatValue::Struct { fields, elems, .. } = &m["ImpulseResponseFunctions"] else { panic!() };
    assert_eq!(fields, &vec!["ea".to_string()]);
    let (dims, irf) = numeric(&elems[0][0]);
    assert_eq!(dims, vec![7, 8, 20000]);
    let want_m = vec_of(&r["irf_ea_mean"]);
    let want_s = vec_of(&r["irf_ea_std"]);
    for i in 0..n {
        for t in 0..8 {
            let x: Vec<f64> = (0..nmc).map(|j| irf[j * 56 + t * 7 + i]).filter(|v| !v.is_nan()).collect();
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            let (w, ws) = (want_m[i * 8 + t], want_s[i * 8 + t]);
            assert!((mean - w).abs() <= 5.0 * ws / (x.len() as f64).sqrt() + 1e-12, "irf {i},{t}: {mean} vs {w}");
        }
    }

    // prior optimize: the optimized mode is the prior mode (as in Dynare's run)
    let o: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("rbc_q2/Output/rbc_q2_prior_optimize.json")).unwrap())
            .unwrap();
    let got = vec_of(&o["optimized"]);
    let mode = vec_of(&o["prior_mode"]);
    let log = std::fs::read_to_string(root().join("reference/q2/q2_c9_matlab.log")).unwrap();
    for (k, (g, p)) in got.iter().zip(&mode).enumerate() {
        // csminwel stops on the objective (crit 1e-7) from a random start: a few percent at most
        assert!((g - p).abs() <= 3e-2 * p.abs(), "parameter {k}: {g} vs prior mode {p}");
    }
    // Dynare's printed prior modes are the same numbers
    for (name, p) in o["names"].as_array().unwrap().iter().zip(&mode) {
        let line = format!("parameter {}", name.as_str().unwrap());
        assert!(log.lines().any(|l| l.contains(&line[10..])), "{line}");
        let txt = format!("  Prior mode ............... {}.", dynare_num2str(*p));
        assert!(log.contains(&txt), "{txt}");
    }
}

/// `num2str` with 5 significant digits (the prior modes here are below 10).
fn dynare_num2str(x: f64) -> String {
    let dgt = x.abs().log10().floor() as i32;
    let p = (dgt + 5).max(5) as usize;
    let s = format!("{:.*e}", p - 1, x);
    let v: f64 = s.parse().unwrap();
    let mut t = format!("{v}");
    if t.contains('e') {
        t = s;
    }
    t
}
