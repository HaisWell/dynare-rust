//! `estimation(order = 2, filter_algorithm = gf, ...)` end to end through the driver vs
//! Dynare 7.1 (reference/nonlinear/ref_estimation.json, dump_estimation.m): the Gaussian
//! filter with cubature rules is deterministic, so mode_compute = 4 must land on Dynare's mode.

mod common;
use dynare::driver::{RunOptions, run};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/nonlinear")
}

#[test]
fn gf_mode_matches_dynare() {
    let r: Value = common::json_ref(refdir().join("ref_estimation.json"));
    let work = std::env::temp_dir().join(format!("dynare_rs_nl_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    for f in ["rbcnl.mod", "rbcnl_data.csv"] {
        std::fs::copy(refdir().join(f), work.join(f)).unwrap();
    }
    let ro = RunOptions { defines: vec![("ESTGF".into(), "1".into())], quiet: true, ..Default::default() };
    run(&work.join("rbcnl.mod"), &ro).unwrap();
    let res: Value =
        serde_json::from_str(&std::fs::read_to_string(work.join("rbcnl/Output/rbcnl_results.json")).unwrap()).unwrap();
    let mode: Vec<f64> = res["mode"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let mref: Vec<f64> = r["gf"]["mode"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for (a, b) in mode.iter().zip(&mref) {
        // csminwel stops once the improvement is below 1e-7: the flat directions (the ME
        // stderr) end within ~1e-4 of each other
        assert!((a - b).abs() / b.abs() < 2e-4, "mode {a} vs Dynare {b}");
    }
    let lap = res["log_data_density_laplace"].as_f64().unwrap();
    let lref = r["gf"]["log_density_laplace"].as_f64().unwrap();
    println!("mode {mode:?}\nDynare {mref:?}\nLaplace {lap} vs {lref}");
    assert!((lap - lref).abs() < 1e-3);
    let _ = std::fs::remove_dir_all(&work);
}
