//! Small known gaps closed against Dynare 7.2 (reference/gaps, dump.m):
//! - `simul_replic > 1` with skewed shocks draws every replication from the skew normal;
//! - the `covar_ME` TeX table of stoch_simul (identical to Dynare's file except the date line);
//! - order-2 `<fname>_stoch_simul.json` carries the measurement-error variance decompositions;
//! - `forecast` bands with measurement errors (`HPDinf_ME`, `HPDsup_ME`) at order 1;
//! - `varexo_det` with `ramsey_model` is a preprocessor error.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/gaps")
}

/// Runs `dynare-rs <name>.mod` on a private copy (optionally with `edit` applied to the text).
fn run(name: &str, edit: &dyn Fn(&str) -> String, tag: &str) -> (PathBuf, std::process::Output) {
    let work = std::env::temp_dir().join(format!("dynare_rust_gaps_{tag}"));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let src = std::fs::read_to_string(refdir().join(format!("{name}.mod"))).unwrap();
    std::fs::write(work.join(format!("{name}.mod")), edit(&src)).unwrap();
    let out =
        Command::new(env!("CARGO_BIN_EXE_dynare-rs")).arg(format!("{name}.mod")).current_dir(&work).output().unwrap();
    (work, out)
}

fn skewness(z: &[f64]) -> f64 {
    let n = z.len() as f64;
    let m = z.iter().sum::<f64>() / n;
    let sd = (z.iter().map(|a| (a - m).powi(2)).sum::<f64>() / n).sqrt();
    z.iter().map(|a| (a - m).powi(3)).sum::<f64>() / n / sd.powi(3)
}

#[test]
fn simul_replic_uses_skewed_shocks() {
    let (work, out) = run("gap_skew", &|s| s.to_string(), "skew");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let bytes = std::fs::read(work.join("gap_skew/Output/gap_skew_simul")).unwrap();
    let v: Vec<f64> = bytes.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let t = 40000;
    let n = v.len() / (3 * t); // endogenous variables incl. auxiliary ones
    assert_eq!(v.len(), 3 * n * t);
    let mut sk = Vec::new();
    for r in 0..3 {
        let y: Vec<f64> = (0..t).map(|s| v[r * n * t + s * n + 1]).collect();
        sk.push(skewness(&y));
    }
    // the first replication is the reference: all replications share its (clearly non-zero) skewness
    assert!(sk[0].abs() > 0.3, "{sk:?}");
    for r in 1..3 {
        assert!((sk[r] - sk[0]).abs() < 0.15, "replication {r}: {sk:?}");
    }
}

#[test]
fn measurement_error_covariance_tex_matches_dynare() {
    let (work, out) = run("gap_me", &|s| s.to_string(), "tex");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let ours = std::fs::read_to_string(work.join("gap_me/latex/gap_me_covar_ME.tex")).unwrap();
    let want = std::fs::read_to_string(refdir().join("ref_covar_ME.tex")).unwrap();
    // identical except the date line
    assert_eq!(ours.lines().skip(1).collect::<Vec<_>>(), want.lines().skip(1).collect::<Vec<_>>());
}

#[test]
fn forecast_bands_with_measurement_errors() {
    let (work, out) = run("gap_me", &|s| s.to_string(), "fc");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let csv = std::fs::read_to_string(work.join("gap_me/Output/gap_me_forecast.csv")).unwrap();
    let r: Value = serde_json::from_str(&std::fs::read_to_string(refdir().join("ref_gap_me.json")).unwrap()).unwrap();
    let mut seen = 0;
    for l in csv.lines().skip(1) {
        let f: Vec<&str> = l.split(',').collect();
        let (h, var) = (f[1].parse::<usize>().unwrap() - 1, f[2]);
        let (lo, up) = (f[6], f[7]);
        if var == "k" {
            assert!(lo.is_empty() && up.is_empty(), "k is not observed");
            continue;
        }
        let wl = r["HPDinf_ME"][var][h].as_f64().unwrap();
        let wu = r["HPDsup_ME"][var][h].as_f64().unwrap();
        assert!((lo.parse::<f64>().unwrap() - wl).abs() < 1e-10, "{var} {h}: {lo} vs {wl}");
        assert!((up.parse::<f64>().unwrap() - wu).abs() < 1e-10, "{var} {h}: {up} vs {wu}");
        seen += 1;
    }
    assert_eq!(seen, 12);
}

#[test]
fn order2_json_has_measurement_error_decompositions() {
    let (work, out) =
        run("gap_me", &|s| s.replace("order = 1", "order = 2").replace("forecast(periods = 6);", ""), "o2");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let j: Value =
        serde_json::from_str(&std::fs::read_to_string(work.join("gap_me/Output/gap_me_stoch_simul.json")).unwrap())
            .unwrap();
    let v = j["variance_decomposition_ME"].as_array().expect("variance_decomposition_ME");
    assert_eq!(v.len(), 2); // observed variables c and y
    for row in v {
        let r = row.as_array().unwrap();
        assert_eq!(r.len(), 3); // x, z, ME
        let s: f64 = r.iter().map(|x| x.as_f64().unwrap()).sum();
        assert!((s - 100.0).abs() < 1e-8);
    }
}

#[test]
fn ramsey_with_varexo_det_is_an_error() {
    let mod_src = "var pi y i;\nvarexo e;\nvarexo_det d;\nparameters b;\nb = 0.99;\nmodel(linear);\npi = b*pi(+1) + y + e + d;\ny = y(+1) - (i - pi(+1));\nend;\nshocks;\nvar e; stderr 0.01;\nend;\nplanner_objective -(pi^2 + y^2);\nramsey_model(instruments=(i), planner_discount=b);\n";
    let work = std::env::temp_dir().join("dynare_rust_gaps_ramsey");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("rd.mod"), mod_src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_dynare-rs")).arg("rd.mod").current_dir(&work).output().unwrap();
    assert!(!out.status.success());
    let all = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(
        all.contains("ramsey_model and ramsey_policy are incompatible with deterministic exogenous variables"),
        "{all}"
    );
}
