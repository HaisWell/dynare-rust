//! `solve_algo = 13 / 14` (Dynare 7.1's `block_trust_region` MEX) against MATLAB
//! (reference/btr/ref_btr.json, written by reference/btr/dump.m): solutions and exit codes of the
//! MEX called directly on small nonlinear systems, and `steady` on reference/btr/btr_ss.mod.

use std::path::{Path, PathBuf};

use dynare::deterministic::DetState;
use dynare::linalg::Mat;
use dynare::modfile::{Command, OptValue, parse_mod_file};
use dynare::solver::block_trust_region::block_trust_region;
use dynare::solver::steady_state::SteadyOptions;
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/btr")
}

fn reference() -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join("ref_btr.json")).unwrap()).unwrap()
}

fn f_rosen(x: &[f64]) -> (Vec<f64>, Mat) {
    (vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]], Mat::from_row_slice(2, 2, &[-20.0 * x[0], 10.0, -1.0, 0.0]))
}

fn f_chain(x: &[f64]) -> (Vec<f64>, Mat) {
    let f = vec![
        x[0].powi(3) - 8.0,
        x[1] + x[0] * x[1] * x[1] - 3.0,
        x[2] * x[2] + x[3] * x[3] - 5.0,
        x[2] * x[3] - 2.0 + 0.1 * x[1],
    ];
    #[rustfmt::skip]
    let j = Mat::from_row_slice(4, 4, &[
        3.0 * x[0] * x[0], 0.0, 0.0, 0.0,
        x[1] * x[1], 1.0 + 2.0 * x[0] * x[1], 0.0, 0.0,
        0.0, 0.0, 2.0 * x[2], 2.0 * x[3],
        0.0, 0.1, x[3], x[2]]);
    (f, j)
}

fn f_exp(x: &[f64]) -> (Vec<f64>, Mat) {
    (
        vec![x[0].exp() - 2.0 * x[1] - 1.0, x[0] + x[1].exp() - 2.0],
        Mat::from_row_slice(2, 2, &[x[0].exp(), -2.0, 1.0, x[1].exp()]),
    )
}

fn f_sing(x: &[f64]) -> (Vec<f64>, Mat) {
    (vec![x[0] * x[0], x[1] - 1.0], Mat::from_row_slice(2, 2, &[2.0 * x[0], 0.0, 0.0, 1.0]))
}

type F = fn(&[f64]) -> (Vec<f64>, Mat);

#[test]
fn mex_direct_cases() {
    let r = reference();
    #[rustfmt::skip]
    let cases: [(&str, F, Vec<f64>, usize, f64, bool); 10] = [
        ("rosen", f_rosen, vec![-1.2, 1.0], 50, 1.0, false),
        ("rosen_b", f_rosen, vec![-1.2, 1.0], 50, 1.0, true),
        ("chain", f_chain, vec![3.0, 1.0, 2.0, 0.5], 50, 1.0, true),
        ("chain_w", f_chain, vec![3.0, 1.0, 2.0, 0.5], 50, 1.0, false),
        ("chain_f", f_chain, vec![30.0, 10.0, 5.0, 5.0], 50, 100.0, true),
        ("maxit", f_exp, vec![8.0, -6.0], 4, 1.0, false),
        ("singular", f_sing, vec![0.0, 0.5], 50, 1.0, false),
        ("sing_dm", f_sing, vec![0.0, 0.5], 50, 1.0, true),
        ("sol0", f_rosen, vec![1.0, 1.0], 50, 1.0, true),
        ("exp50", f_exp, vec![8.0, -6.0], 50, 1.0, true),
    ];
    for (name, f, x0, maxit, factor, dm) in cases {
        let opts = SteadyOptions { maxit, trust_region_initial_step_bound_factor: factor, ..Default::default() };
        let o = block_trust_region(&f, &x0, &opts, dm);
        let e = &r[name];
        assert_eq!(o.errorcode as i64, e["errorcode"].as_i64().unwrap(), "{name}: errorcode");
        assert_eq!(o.errorflag, e["errorflag"].as_i64().unwrap() != 0, "{name}: errorflag");
        let mut want: Vec<f64> = e["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        // chain_f starts on x[2] = x[3], and the system is symmetric in (x[2], x[3]): the
        // iterates stay on that line up to rounding, so which of the two mirror-image roots
        // they leave it for is decided by last-bit BLAS/libm differences. MATLAB on macOS
        // (and Accelerate) picks x[2] < x[3]; OpenBLAS on Linux and Windows picks the other.
        // The mirror root is reached by a different path, so it agrees only to the solver's
        // stopping tolerance (2.2e-7).
        let mut tol = 1e-9;
        if name == "chain_f" && !cfg!(target_os = "macos") && (o.x[2] - want[3]).abs() < (o.x[2] - want[2]).abs() {
            want.swap(2, 3);
            tol = 1e-6;
        }
        for (i, &v) in want.iter().enumerate() {
            assert!((o.x[i] - v).abs() <= tol * v.abs().max(1.0), "{name}: x[{i}] {} vs {v}", o.x[i]);
        }
    }
}

fn steady(algo: f64) -> Option<Vec<f64>> {
    let mf = parse_mod_file(&dir().join("btr_ss.mod"), &[]).unwrap();
    let mut det = DetState::new(&mf).unwrap();
    for c in &mf.commands {
        if let Command::Steady(o) = c {
            let mut o: Vec<_> = o.iter().filter(|(k, _)| k != "solve_algo").cloned().collect();
            o.push(("solve_algo".into(), OptValue::Num(algo)));
            o.push(("noprint".into(), OptValue::Flag));
            return det.steady(&mf.model, &o).ok().map(|_| det.steady_state.to_vec());
        }
    }
    None
}

#[test]
fn steady_solve_algo_13_14() {
    let r = reference();
    for (algo, key) in [(13.0, "ss13"), (14.0, "ss14")] {
        let e = &r[key];
        let got = steady(algo);
        if e["ok"].as_bool().unwrap() {
            let got = got.unwrap_or_else(|| panic!("{key}: steady failed but MATLAB solved it"));
            for (i, v) in e["x"].as_array().unwrap().iter().enumerate() {
                let v = v.as_f64().unwrap();
                assert!((got[i] - v).abs() <= 1e-8 * v.abs().max(1.0), "{key}: x[{i}] {} vs {v}", got[i]);
            }
        } else {
            assert!(got.is_none(), "{key}: MATLAB fails, Rust must fail as well");
        }
    }
}
