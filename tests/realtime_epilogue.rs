//! realtime_shock_decomposition(with_epilogue) against Dynare 7.2
//! (reference/realtime_epilogue/ref_rt_epi.json, written by reference/realtime_epilogue/dump.m):
//! the pools and the saved vintages of oo_.realtime_shock_decomposition,
//! oo_.conditional_shock_decomposition, oo_.realtime_forecast_shock_decomposition and
//! oo_.realtime_conditional_shock_decomposition for the endogenous variable y and the epilogue
//! variables ya, gya (lags up to 4 quarters), plus plot_shock_decomposition(realtime=1..3).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

fn ref_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/realtime_epilogue")
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs())
}

/// A long-format csv (date,variable,component,value) as (variable, component) -> values.
fn read_long(p: &Path) -> HashMap<(String, String), Vec<f64>> {
    let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let mut m: HashMap<(String, String), Vec<f64>> = HashMap::new();
    for l in text.lines().skip(1) {
        let f: Vec<&str> = l.split(',').collect();
        m.entry((f[1].into(), f[2].into())).or_default().push(f[3].parse().unwrap_or(f64::NAN));
    }
    m
}

#[test]
fn realtime_with_epilogue() {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(ref_dir().join("ref_rt_epi.json")).unwrap()).unwrap();
    assert!(r["ok"].as_bool().unwrap(), "{:?}", r["error"]);
    let dir = std::env::temp_dir().join(format!("dynare_rs_rtepi_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(ref_dir().join("q4_data.csv"), dir.join("q4_data.csv")).unwrap();
    // the epilogue variables are asked for in the var list (dynare-rust writes the listed rows)
    let modtext = std::fs::read_to_string(ref_dir().join("rt_epi.mod"))
        .unwrap()
        .replace("with_epilogue) y;", "with_epilogue) y ya gya;");
    std::fs::write(dir.join("rt_epi.mod"), modtext).unwrap();
    dynare::driver::run(&dir.join("rt_epi.mod"), &dynare::driver::RunOptions::default()).unwrap();
    let out = dir.join("rt_epi/Output");
    // endogenous variables y c k, then the epilogue ya gya
    let vars = [("y", 0usize), ("ya", 3), ("gya", 4)];
    let comps = ["e", "Initial values", "Smoothed"];
    let mut checked = 0;
    for group in [
        "realtime_shock_decomposition",
        "conditional_shock_decomposition",
        "realtime_forecast_shock_decomposition",
        "realtime_conditional_shock_decomposition",
    ] {
        for (key, a) in r[group].as_object().unwrap() {
            let got = read_long(&out.join(format!("rt_epi_{group}_{key}.csv")));
            let a = a.as_array().unwrap(); // [var][comp][t]
            assert_eq!(a.len(), 5, "{group} {key}: variables");
            for (name, v) in vars {
                for (ci, comp) in comps.iter().enumerate() {
                    let want: Vec<f64> =
                        a[v][ci].as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
                    let have = &got[&(name.to_string(), comp.to_string())];
                    assert_eq!(have.len(), want.len(), "{group} {key} {name} {comp}: periods");
                    for (t, (h, w)) in have.iter().zip(&want).enumerate() {
                        assert!(close(*h, *w, 1e-9), "{group} {key} {name} {comp} t={t}: {h} vs {w}");
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 1000);
    // plot_shock_decomposition(realtime=1..3) with epilogue variables ran and wrote their rows
    let mut plots = 0;
    for e in std::fs::read_dir(&out).unwrap() {
        let n = e.unwrap().file_name().to_string_lossy().to_string();
        if n.starts_with("rt_epi_shock_decomposition_rt") {
            let text = std::fs::read_to_string(out.join(&n)).unwrap();
            let names: Vec<&str> = text.lines().skip(1).map(|l| l.split(',').nth(2).unwrap()).collect();
            assert!(names.contains(&"ya") && names.contains(&"gya"), "{n}");
            plots += 1;
        }
    }
    assert_eq!(plots, 3);
}
