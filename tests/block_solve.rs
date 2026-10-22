//! `model(block)`: block-wise perfect foresight (`solve_block_decomposed_problem.m`) and block-wise
//! steady state (`evaluate_steady_state.m`) against Dynare 7.2
//! (reference/block_solve/ref_block_solve*.json, written by reference/block_solve/dump.m), and
//! against the non-block solution of the same model.

use std::path::{Path, PathBuf};

use dynare::deterministic::DetState;
use dynare::modfile::{Command, ModFile, OptValue, parse_mod_file};
use serde_json::Value;

fn ref_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/block_solve")
}

fn reference(file: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(ref_dir().join(file)).unwrap()).unwrap()
}

/// Runs the commands like the driver, without writing output files.
fn run(mf: &ModFile, tolf: Option<f64>) -> DetState {
    let mut det = DetState::new(mf).unwrap();
    for c in &mf.commands {
        match c {
            Command::Steady(o) => {
                let mut o = o.clone();
                o.push(("noprint".into(), OptValue::Flag));
                det.steady(&mf.model, &o).unwrap()
            }
            Command::Endval(v) => det.endval(&mf.model, v).unwrap(),
            Command::Deterministic(name, o, _) => {
                det.simul.update(o);
                det.simul.noprint = true;
                det.simul.verbosity = 0;
                if let Some(t) = tolf {
                    det.simul.tolf = t;
                }
                match name.as_str() {
                    "perfect_foresight_setup" => det.setup(mf).unwrap(),
                    "perfect_foresight_solver" => det.solve(&mf.model).unwrap(),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    det
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    match v {
        Value::Array(a) if a.first().is_some_and(|x| x.is_array()) => {
            a.iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()).collect()
        }
        Value::Array(a) => a.iter().map(|x| vec![x.as_f64().unwrap_or(f64::NAN)]).collect(),
        _ => panic!("not a matrix"),
    }
}

fn load(file: &str, defines: &[(&str, &str)]) -> ModFile {
    let d: Vec<(String, String)> = defines.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    parse_mod_file(&ref_dir().join(file), &d).unwrap()
}

/// Block solution against MATLAB's. Dynare stops its Newton iterations at residual 1e-5 (dynatol.f) and the
/// stacked systems are ill-conditioned, so the unconverged iterates agree to ~1e-5 only (the printed
/// residual traces agree to 5-6 digits).
fn check_matlab(case: &str, file: &str, defines: &[(&str, &str)], tol: f64) {
    check_matlab_in("ref_block_solve.json", case, file, defines, tol)
}
fn check_matlab_in(json: &str, case: &str, file: &str, defines: &[(&str, &str)], tol: f64) {
    let r = &reference(json)[case];
    let mf = load(file, defines);
    assert!(mf.model.block_solve.as_ref().is_some_and(|b| b.dynamic.is_some() && b.statik.is_some()));
    let names: Vec<String> =
        r["endo_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let n = mf.model.orig_endo_nbr;
    let det = run(&mf, None);
    assert!(det.last.status, "{case}: solver failed");
    let e = det.endo_simul.as_ref().unwrap();
    let re = rows(&r["endo_simul"]);
    assert_eq!((e.nrows(), e.ncols()), (re.len(), re[0].len()), "{case}: endo_simul shape");
    let mut worst: f64 = 0.0;
    for (j, name) in names.iter().enumerate().take(n) {
        let jr = mf.model.endo_index(name).unwrap();
        for c in 0..e.ncols() {
            let d = (e[(jr, c)] - re[j][c]).abs() / re[j][c].abs().max(1.0);
            worst = worst.max(d);
            if d >= tol {
                eprintln!("{case}: {name} column {c}: {} vs {} (rel {d:e})", e[(jr, c)], re[j][c]);
            }
        }
    }
    let ss: Vec<f64> = r["steady_state"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for j in 0..n {
        let jr = mf.model.endo_index(&names[j]).unwrap();
        assert!((det.steady_state[jr] - ss[j]).abs() / ss[j].abs().max(1.0) < tol, "{case}: steady state {}", names[j]);
    }
    assert!(worst < tol, "{case}: max rel. diff to MATLAB {worst:e}");
    println!("{case}: max rel. diff to MATLAB {worst:.2e}");
}

/// Block solution against the non-block solution at a tight `tolf`.
fn check_nonblock(file: &str, defines: &[(&str, &str)], tol: f64) {
    let src = std::fs::read_to_string(ref_dir().join(file)).unwrap();
    let nb = src.replace("model(block, ", "model(").replace("model(block);", "model;");
    assert_ne!(nb, src, "{file}: no model(block)");
    let d: Vec<(String, String)> = defines.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    let tmp = std::env::temp_dir().join(format!("dynare_block_solve_{}_{}", std::process::id(), file));
    std::fs::write(&tmp, &nb).unwrap();
    let mfn = parse_mod_file(&tmp, &d).unwrap();
    let _ = std::fs::remove_file(&tmp);
    let mfb = load(file, defines);
    assert!(mfn.model.block_solve.is_none() && mfb.model.block_solve.is_some());
    let (dn, db) = (run(&mfn, Some(1e-12)), run(&mfb, Some(1e-12)));
    let (en, eb) = (dn.endo_simul.as_ref(), db.endo_simul.as_ref());
    if let (Some(en), Some(eb)) = (en, eb) {
        assert!(dn.last.status && db.last.status, "{file}: solver failed");
        let worst = en.iter().zip(eb.iter()).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0.0, f64::max);
        assert!(worst < tol, "{file}: block vs non-block {worst:e}");
        println!("{file}: block vs non-block {worst:.2e}");
    }
    for (a, b) in dn.steady_state.iter().zip(&db.steady_state) {
        assert!((a - b).abs() / b.abs().max(1.0) < tol, "{file}: steady state {a} vs {b}");
    }
}

#[test]
fn rbc_permanent_shock() {
    check_matlab("rbc_perm", "rbc_b.mod", &[("CASE", "1")], 5e-5);
}

#[test]
fn rbc_temporary_shock() {
    check_matlab("rbc_temp", "rbc_b.mod", &[("CASE", "2")], 5e-5);
}

#[test]
fn rbc_steady_state_model() {
    check_matlab("rbc_ssm", "rbc_b.mod", &[("CASE", "3")], 5e-5);
}

#[test]
fn rbc_predetermined() {
    check_matlab("rbc_predet", "rbc_predet_b.mod", &[], 5e-5);
}

#[test]
fn q1_pf_lbj() {
    check_matlab("q1_pf_lbj", "q1_pf_b.mod", &[("CASE", "1")], 5e-5);
}

#[test]
fn q1_pf_gmres() {
    check_matlab("q1_pf_gmres", "q1_pf_b.mod", &[("CASE", "2")], 5e-5);
}

#[test]
fn purely_backward() {
    check_matlab("blk_back_pf_1", "blk_back_pf_1.mod", &[], 5e-5);
    check_matlab("blk_back_pf_8", "blk_back_pf_8.mod", &[], 5e-5);
}

#[test]
fn new_keynesian_mfs() {
    check_matlab("blk_nk_pf_3", "blk_nk_pf_3.mod", &[], 5e-5);
}

#[test]
fn sw07_perfect_foresight() {
    // linear in practice: converged to rounding (3e-14), unlike the nonlinear cases above
    check_matlab("sw07", "sw07_pf_b.mod", &[], 1e-9);
}

#[test]
fn steady_block_by_block() {
    let all = reference("ref_block_solve_ss.json");
    for (case, file) in [("blk_nk_ss_0", "blk_nk_ss_0.mod"), ("blk_nk_ss_6", "blk_nk_ss_6.mod")] {
        let r = &all[case];
        let mf = load(file, &[]);
        let det = run(&mf, None);
        let names: Vec<String> =
            r["endo_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        let ss: Vec<f64> = r["steady_state"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        for (j, name) in names.iter().enumerate() {
            let jr = mf.model.endo_index(name).unwrap();
            assert!((det.steady_state[jr] - ss[j]).abs() < 1e-9, "{case}: steady state {name}");
        }
    }
}

#[test]
fn matches_non_block_solution() {
    check_nonblock("rbc_b.mod", &[("CASE", "1")], 1e-10);
    check_nonblock("rbc_b.mod", &[("CASE", "2")], 1e-10);
    check_nonblock("rbc_predet_b.mod", &[], 1e-10);
    check_nonblock("q1_pf_b.mod", &[("CASE", "1")], 1e-10);
    check_nonblock("blk_fwd_pf_0.mod", &[], 1e-10);
    check_nonblock("blk_back_pf_1.mod", &[], 1e-10);
    check_nonblock("blk_back_pf_8.mod", &[], 1e-10);
    check_nonblock("blk_nk_pf_3.mod", &[], 1e-10);
    check_nonblock("blk_nk_ss_0.mod", &[], 1e-10);
    check_nonblock("blk_nk_ss_6.mod", &[], 1e-10);
    check_nonblock("blk_back_ss_0.mod", &[], 1e-10);
    check_nonblock("blk_fwd_ss_10.mod", &[], 1e-10);
}

/// `stack_solve_algo=4`: Newton direction with the `lnsrch1` path-length search on the stacked block
/// (solve_two_boundaries_stacked.m); residual trace 1.27, 1.23, 0.94, ... 1.8e-8 in Dynare 7.1.
#[test]
fn stack_solve_algo_4_path_length_search() {
    check_matlab_in("ref_block_solve_a4.json", "blk_nk_pf_4", "blk_nk_pf_4.mod", &[], 5e-5);
}
