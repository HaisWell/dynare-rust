//! Stochastic extended path with `options_.ep.solve_algo = 10` (lmmcp on the tree), against
//! MATLAB (reference/ep_lmmcp/dump.m) on MATLAB's own shocks.
use std::path::{Path, PathBuf};

use dynare::deterministic::DetState;
use dynare::ep::{EpModel, EpOptions, extended_path};
use dynare::linalg::Mat;
use dynare::modfile::{Command, OptValue, parse_mod_file};
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/ep_lmmcp")
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| match r {
            Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect(),
            x => vec![x.as_f64().unwrap_or(f64::NAN)],
        })
        .collect()
}

#[test]
fn stochastic_extended_path_lmmcp() {
    for (name, case) in [("sep1", "1"), ("sep2sparse_hybrid", "2"), ("ep0", "3")] {
        let r: Value =
            serde_json::from_str(&std::fs::read_to_string(dir().join(format!("ref_{name}.json"))).unwrap()).unwrap();
        let mf = parse_mod_file(&dir().join("ep_lmmcp.mod"), &[("CASE".into(), case.into())]).unwrap();
        let mut det = DetState::new(&mf).unwrap();
        let mut o = EpOptions::default();
        for c in &mf.commands {
            match c {
                Command::Steady(opts) => {
                    let mut opts = opts.clone();
                    opts.push(("noprint".into(), OptValue::Flag));
                    det.steady(&mf.model, &opts).unwrap();
                }
                Command::Other(n, opts) if n == "extended_path" => {
                    let mut opts = opts.clone();
                    // dump.m: options_.ep.solve_algo = 10
                    opts.push(("solve_algo".into(), OptValue::Num(10.0)));
                    o.update(&opts).unwrap();
                }
                _ => {}
            }
        }
        assert_eq!(o.solve_algo, 10);
        let ex = rows(&r["exo_simul"]);
        let shocks = Mat::from_fn(ex.len() - 1, ex[0].len(), |t, k| ex[t + 1][k]);
        let m = EpModel {
            model: &mf.model,
            params: &det.params,
            steady_state: &det.steady_state,
            exo_steady_state: &det.exo_steady_state,
            sigma_e: &mf.sigma_e,
        };
        let out = extended_path(&m, &o, &det.steady_state.clone(), &shocks).unwrap();
        assert!(out.status, "{name}: extended path failed");
        let want = rows(&r["endo_simul"]);
        let mut err = 0.0f64;
        for (v, row) in want.iter().enumerate() {
            for (t, w) in row.iter().enumerate() {
                err = err.max((out.endo_simul[(v, t)] - w).abs());
            }
        }
        println!("{name}: max |diff| {err:.2e}");
        assert!(err < 1e-9, "{name}: max diff {err}");
    }
}
