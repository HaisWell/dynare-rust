//! Perfect-foresight simulations against Dynare 7.2 (reference/deterministic/ref_deterministic.json,
//! written by reference/deterministic/dump_deterministic.m): RBC with permanent / temporary
//! shocks, steady_state_model, linear_approximation, histval + endogenous_terminal_period,
//! a model(linear) with the `simul` command, and the SW07 model (reference/sw07) under temporary shocks.
mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use dynare::deterministic::DetState;
use dynare::modfile::{Command, ModFile, parse_mod_file};
use serde_json::Value;

fn ref_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/deterministic")
}

fn reference() -> Option<Value> {
    let p = ref_dir().join("ref_deterministic.json");
    Some(common::json_ref(p))
}

/// Runs the commands like the driver, without writing output files.
fn run(mf: &ModFile, quiet: bool) -> DetState {
    let mut det = DetState::new(mf).unwrap();
    for c in &mf.commands {
        match c {
            Command::Steady(o) => {
                let mut o = o.clone();
                if quiet {
                    o.push(("noprint".into(), dynare::modfile::OptValue::Flag));
                }
                det.steady(&mf.model, &o).unwrap()
            }
            Command::Endval(v) => det.endval(&mf.model, v).unwrap(),
            Command::Deterministic(name, o, _) => {
                det.simul.update(o);
                if quiet {
                    det.simul.noprint = true;
                    det.simul.verbosity = 0;
                }
                match name.as_str() {
                    "perfect_foresight_setup" => det.setup(mf).unwrap(),
                    "perfect_foresight_solver" => det.solve(&mf.model).unwrap(),
                    "simul" => {
                        det.setup(mf).unwrap();
                        det.solve(&mf.model).unwrap()
                    }
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

fn check(case: &str, file: &str, defines: &[(&str, &str)], tol: f64) {
    let r = reference().expect("reference/setup unavailable");
    let r = &r[case];
    let defines: Vec<(String, String)> = defines.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    let mf = parse_mod_file(&ref_dir().join(file), &defines).unwrap();
    let names: Vec<String> =
        r["endo_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let n = mf.model.orig_endo_nbr;
    assert_eq!(mf.model.endo_names[..n], names[..n], "{case}: endo names");
    assert_eq!(mf.model.n_endo(), names.len(), "{case}: number of endogenous variables");
    let t0 = Instant::now();
    let det = run(&mf, true);
    let el = t0.elapsed().as_secs_f64();
    assert!(det.last.status, "{case}: solver failed");
    assert_eq!(det.maximum_lag as f64, r["maximum_lag"].as_f64().unwrap());
    assert_eq!(det.maximum_lead as f64, r["maximum_lead"].as_f64().unwrap());
    let e = det.endo_simul.as_ref().unwrap();
    let re = rows(&r["endo_simul"]);
    assert_eq!((e.nrows(), e.ncols()), (re.len(), re[0].len()), "{case}: endo_simul shape");
    // declared variables (aux lead variables are numbered after preprocessor expression ids,
    // which differ between Dynare and dynare-rust)
    let mut worst: f64 = 0.0;
    for (j, name) in names.iter().enumerate().take(n) {
        let jr = mf.model.endo_index(name).unwrap();
        for c in 0..e.ncols() {
            let (a, b) = (e[(jr, c)], re[j][c]);
            let d = (a - b).abs() / b.abs().max(1.0);
            worst = worst.max(d);
            assert!(d < tol, "{case}: {name} column {c}: {a} vs {b} (rel {d:e})");
        }
    }
    let x = det.exo_simul.as_ref().unwrap();
    let rx = rows(&r["exo_simul"]);
    for rr in 0..x.nrows() {
        for k in 0..x.ncols() {
            assert!((x[(rr, k)] - rx[rr][k]).abs() < 1e-14, "{case}: exo_simul({rr},{k})");
        }
    }
    let ss: Vec<f64> = r["steady_state"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    for j in 0..n {
        assert!((det.steady_state[j] - ss[j]).abs() / ss[j].abs().max(1.0) < tol, "{case}: steady state {j}");
    }
    println!(
        "{case}: max rel. diff {worst:.2e}; dynare-rust {:.2} ms (incl. steady), MATLAB solver {:.2} ms",
        el * 1e3,
        r["t_solver"].as_f64().unwrap() * 1e3
    );
}

#[test]
fn rbc_permanent_shock() {
    check("rbc_perm", "rbc.mod", &[("CASE", "1")], 1e-9);
}

#[test]
fn rbc_temporary_shock() {
    check("rbc_temp", "rbc.mod", &[("CASE", "2")], 1e-9);
}

#[test]
fn rbc_steady_state_model() {
    check("rbc_ssm", "rbc.mod", &[("CASE", "3")], 1e-9);
}

#[test]
fn rbc_linear_approximation() {
    check("rbc_linapprox", "rbc.mod", &[("CASE", "4")], 1e-9);
}

#[test]
fn rbc_histval_endogenous_terminal_period() {
    check("rbc_histval", "rbc.mod", &[("CASE", "5")], 1e-9);
}

#[test]
fn linear_model_simul() {
    check("linear", "linear.mod", &[], 1e-9);
}

#[test]
fn sw07_temporary_shock() {
    check("sw07", "sw07_pf.mod", &[], 1e-9);
}

#[test]
fn rbc_homotopy() {
    check("rbc_homotopy", "rbc.mod", &[("CASE", "6")], 1e-9);
}

#[test]
fn rbc_predetermined_variables() {
    check("rbc_predet", "rbc_predet.mod", &[], 1e-9);
}

#[test]
fn mshocks_and_exogenous_leads_lags() {
    check("mshocks", "mshocks.mod", &[], 1e-9);
}

#[test]
fn stepped_periods_range_is_rejected() {
    // `periods 1:2:9` is a syntax error for the 7.1 preprocessor ("unexpected ':'"): only
    // `first:last` ranges exist; dynare-rust rejects it with a message naming the line.
    let src = "var y; varexo e; parameters r; r=0.5; model; y = r*y(-1)+e; end; shocks; var e; stderr 1; end;\n\
shocks(overwrite); var e; periods 1:2:9; values 1; end; perfect_foresight_setup(periods=10); perfect_foresight_solver;\n";
    let e = dynare::modfile::parse_mod_str(src, &[]).err().unwrap().to_string();
    assert!(e.contains(":2") && e.contains("`first:last`") && e.contains("1:2:9"), "{e}");
}

#[test]
fn parses_deterministic_blocks() {
    use dynare::modfile::{DetShockKind, parse_mod_str};
    let src = "var y k;\nvarexo e;\nvarexo_det d;\npredetermined_variables k;\nparameters a;\na = 0.5;\n\
model;\ny = a*k + e + d;\nk(+1) = 0.9*k + y;\nend;\n\
steady_state_model;\nk = 1/(1-a);\ny = k*(1-0.9);\nend;\n\
initval;\ne = 0.1;\ny = 2*e;\nend;\nhistval;\nk(0) = 3;\nend;\n\
shocks;\nvar e; periods 1:3 5, 7; values 0.1 (a*2) -0.3;\nend;\n\
mshocks(relative_to_initval);\nvar e; periods 2; values 1.5;\nend;\n\
endval;\ne = 0.2;\nend;\nsimul(periods=20, maxit=10);\nrplot y k;\n";
    // the 7.1 preprocessor rejects varexo_det in a file with a perfect-foresight solver
    let e = parse_mod_str(src, &[]).err().unwrap().to_string();
    assert!(e.contains("and varexo_det declaration"), "{e}");
    let src = src.replace("varexo_det d;\n", "").replace(" + d;", ";");
    let mf = parse_mod_str(&src, &[]).unwrap();
    assert!(mf.exo_det.is_empty() && mf.model.exo_det_names.is_empty());
    assert_eq!(mf.predetermined_variables, vec![1]);
    assert_eq!(mf.steady_state_model.len(), 2);
    assert_eq!(mf.initval_exo, vec![(0, 0.1)]);
    assert!((mf.initval[0].1 - 0.2).abs() < 1e-15, "initval expressions see earlier values");
    assert_eq!(mf.histval, vec![(true, 1, 0, 3.0)]);
    let ds = &mf.det_shocks;
    assert_eq!(ds.len(), 4);
    assert_eq!(ds[0].periods, vec![1, 2, 3]);
    assert_eq!((ds[1].periods.clone(), ds[1].value), (vec![5], 1.0));
    assert_eq!((ds[2].periods.clone(), ds[2].value), (vec![7], -0.3));
    assert_eq!(ds[3].kind, DetShockKind::MultiplyInitialSteadyState);
    // predetermined k: k(+1) -> k, k -> k(-1)
    assert_eq!(mf.model.lead_lag_incidence[1][2], 0);
    assert!(mf.model.lead_lag_incidence[1][0] > 0);
    assert!(matches!(&mf.commands[0], Command::Endval(v) if v[0].0 == "e"));
    assert!(matches!(&mf.commands[1], Command::Deterministic(n, _, _) if n == "simul"));
    assert!(matches!(&mf.commands[2], Command::Deterministic(n, _, v) if n == "rplot" && v == &["y", "k"]));
}

/// `steady` printout (disp_steady_state.m) character for character against Dynare's log
/// (reference/deterministic/steady_print.log): blank line, title, blank line, names padded to
/// the longest endogenous name (auxiliary ones included), " \t\t ", `%g`.
#[test]
fn steady_printout_matches_dynare_log() {
    let log = common::read_ref(ref_dir().join("steady_print.log"));
    let mf = parse_mod_file(&ref_dir().join("steady_print.mod"), &[]).unwrap();
    let m = dynare::dsge::DsgeModel::new(&mf, None).unwrap();
    let ours = dynare::deterministic::steady_state_text(&mf.model, &m.ys_calib);
    // Dynare's block: from the blank line before the first title to the line before the second
    let start = log.find("\nSTEADY-STATE RESULTS:").unwrap();
    let end = start + 1 + log[start + 1..].find("\nSTEADY-STATE RESULTS:").unwrap();
    let theirs = &log[start..end];
    assert_eq!(ours, theirs);
}
