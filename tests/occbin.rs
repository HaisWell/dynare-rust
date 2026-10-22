//! OccBin vs Dynare 7.2 (reference/occbin/ref_occbin.json, written by
//! reference/occbin/dump_occbin.m): a ZLB New Keynesian model (one constraint, four option
//! variants), a borrowing-constraint model whose baseline regime is the binding one, and
//! Dynare's two-constraint RBC example.
mod common;

use dynare::linalg::Mat;
use dynare::modfile::{Command, parse_mod_file};
use dynare::occbin::{OccbinCommand, Session, SimulResult};
use serde_json::Value;
use std::path::Path;

fn reference() -> Option<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/occbin/ref_occbin.json");
    Some(common::json_ref(p))
}

/// Flattens a jsonencode'd array (any nesting) in the order the values appear.
fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Bool(b) => vec![*b as u8 as f64],
        x => vec![x.as_f64().unwrap_or(f64::NAN)],
    }
}

/// Row-major MATLAB matrix (r x c).
fn mat(v: &Value, r: usize, c: usize) -> Mat {
    let f = flat(v);
    assert_eq!(f.len(), r * c);
    Mat::from_row_slice(r, c, &f)
}

fn max_diff(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Runs the OccBin commands of the .mod file (the last occbin_solver's result).
fn run(file: &str, defines: &[(String, String)]) -> (dynare::modfile::ModFile, SimulResult) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/occbin").join(file);
    let mf = parse_mod_file(&path, defines).unwrap();
    let mut s = Session::new(&mf, "test");
    let mut last = None;
    for c in &mf.commands {
        if let Command::Occbin(oc) = c {
            match oc {
                OccbinCommand::Solver(o) => last = Some(s.run_solver(&mf, o).unwrap()),
                OccbinCommand::Setup(o) => s.setup(&mf, o).unwrap(),
                OccbinCommand::SurpriseShocks { .. } => {
                    let tmp = std::env::temp_dir();
                    s.command(&mf, oc, &tmp, "test", &tmp).unwrap()
                }
                _ => {}
            }
        }
    }
    (mf, last.expect("no occbin_solver"))
}

fn check(key: &str, file: &str, defines: &[(String, String)]) {
    let all = reference().expect("reference/setup unavailable");
    let r = &all[key];
    let (mf, out) = run(file, defines);
    let m = &mf.model;
    let n = m.n_endo();

    // preprocessor: parameter names (occbin_<c>_bind appended on first use), pswitch
    let names: Vec<String> =
        r["param_names"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
    assert_eq!(m.param_names, names, "{key}: param_names");
    let psw: Vec<usize> = flat(&r["pswitch"]).iter().map(|&x| x as usize - 1).collect();
    assert_eq!(dynare::occbin::pswitch(m, &mf.occbin_constraints).unwrap(), psw, "{key}: pswitch");
    let params = flat(&r["params"]);
    for (a, b) in m.params.iter().zip(&params) {
        assert!((a - b).abs() <= 1e-14 * b.abs().max(1.0), "{key}: params");
    }

    // regimes' Jacobians and residuals at the steady state
    let ys = flat(&r["ys"]);
    for (a, b) in out.ys.iter().zip(&ys) {
        assert!((a - b).abs() < 1e-10, "{key}: ys {a} vs {b}");
    }
    let ne = m.n_exo();
    let g1 = mat(&r["g1"], n, 3 * n + ne);
    let d0 = dynare::occbin::solver::get_deriv(m, &out.ys, &m.params);
    let mut p1 = m.params.clone();
    for &k in &psw {
        p1[k] = 1.0;
    }
    let d1 = dynare::occbin::solver::get_deriv(m, &out.ys, &p1);
    let g1s = mat(&r["g1_star"], n, 3 * n + ne);
    let cat = |d: &dynare::occbin::solver::RegimeMats| {
        let mut g = Mat::zeros(n, 3 * n + ne);
        g.columns_mut(0, n).copy_from(&d.c);
        g.columns_mut(n, n).copy_from(&d.b);
        g.columns_mut(2 * n, n).copy_from(&d.a);
        g.columns_mut(3 * n, ne).copy_from(&d.j);
        g
    };
    assert!(max_diff(&cat(&d0), &g1) < 1e-10, "{key}: g1 {}", max_diff(&cat(&d0), &g1));
    assert!(max_diff(&cat(&d1), &g1s) < 1e-10, "{key}: g1 (all binding)");
    let rs = flat(&r["resid_star"]);
    for (a, b) in d1.d.iter().zip(&rs) {
        assert!((a - b).abs() < 1e-10, "{key}: resid_star");
    }

    // simulated paths
    let np = out.piecewise.nrows();
    let pw = mat(&r["piecewise"], np, n);
    let lin = mat(&r["linear"], np, n);
    let dpw = max_diff(&out.piecewise, &pw);
    let dlin = max_diff(out.linear.as_ref().unwrap(), &lin);
    eprintln!("{key}: {np} periods, max |piecewise diff| {dpw:.2e}, linear {dlin:.2e}");
    assert!(dpw < 1e-9, "{key}: piecewise differs by {dpw}");
    assert!(dlin < 1e-9, "{key}: linear differs by {dlin}");

    // shocks
    let nsp = out.shocks_sequence.nrows();
    let npos = out.exo_pos.len();
    assert!(max_diff(&out.shocks_sequence, &mat(&r["shocks_sequence"], nsp, npos)) == 0.0);
    let pos: Vec<usize> = flat(&r["exo_pos"]).iter().map(|&x| x as usize - 1).collect();
    assert_eq!(out.exo_pos, pos);

    // regime history
    let fields: &[(&str, &str)] = if mf.occbin_constraints.len() == 1 {
        &[("regime", "regimestart")]
    } else {
        &[("regime1", "regimestart1"), ("regime2", "regimestart2")]
    };
    assert_eq!(out.regime_history.len(), nsp);
    for (k, (fr, fs)) in fields.iter().enumerate() {
        let rr = r[*fr].as_array().unwrap();
        let rs = r[*fs].as_array().unwrap();
        for t in 0..nsp {
            let g = &out.regime_history[t][k];
            let want_r: Vec<bool> = flat(&rr[t]).iter().map(|&x| x != 0.0).collect();
            let want_s: Vec<usize> = flat(&rs[t]).iter().map(|&x| x as usize).collect();
            assert_eq!(g.regime, want_r, "{key}: {fr} period {}", t + 1);
            assert_eq!(g.start, want_s, "{key}: {fs} period {}", t + 1);
        }
    }

    // state space of each shock period: T (n x n x nsp), R (n x ne x nsp), C (n x nsp)
    let tt = flat(&r["T"]);
    let rr = flat(&r["R"]);
    let cc = flat(&r["C"]);
    let mut dss: f64 = 0.0;
    for sp in 0..nsp {
        let s = &out.ss[sp];
        for i in 0..n {
            for j in 0..n {
                let v = tt[(i * n + j) * nsp + sp];
                dss = dss.max((s.t[(i, j)] - v).abs());
            }
            for e in 0..ne {
                let v = if nsp == 1 { rr[i * ne + e] } else { rr[(i * ne + e) * nsp + sp] };
                dss = dss.max((s.r[(i, e)] - v).abs());
            }
            dss = dss.max((s.c[i] - cc[i * nsp + sp]).abs());
        }
    }
    eprintln!("{key}: max |T,R,C diff| {dss:.2e}");
    assert!(dss < 1e-9, "{key}: state space differs by {dss}");
}

fn variant(k: u32) -> Vec<(String, String)> {
    vec![("VARIANT".into(), k.to_string())]
}

#[test]
fn nk_zlb() {
    check("nk_zlb", "nk_zlb.mod", &[]);
}

#[test]
fn nk_zlb_curb_retrench() {
    check("nk_zlb_v1", "nk_zlb.mod", &variant(1));
}

#[test]
fn nk_zlb_short_check_ahead() {
    check("nk_zlb_v2", "nk_zlb.mod", &variant(2));
}

#[test]
fn nk_zlb_max_check_ahead_reset_regime() {
    check("nk_zlb_v3", "nk_zlb.mod", &variant(3));
}

#[test]
fn borrowing_constraint() {
    check("borrcon", "borrcon.mod", &[]);
}

#[test]
fn rbc_two_constraints() {
    check("rbc_2c", "rbc_2c.mod", &[]);
}

/// stream PP: the same model with `STEADY_STATE(Invest)` inside the model equations (Dynare's
/// example as written) gives the same paths and regimes.
#[test]
fn rbc_two_constraints_steady_state_operator() {
    check("rbc_2c", "rbc_2c_ss.mod", &[]);
}

#[test]
fn surprise_shocks_and_setup() {
    let (_, out) = run("rbc_2c.mod", &[]);
    // periods 1:9, 10, 50, 90, 130, 131:169 -> 169 rows, one shock
    assert_eq!(out.shocks_sequence.shape(), (169, 1));
    assert_eq!(out.shocks_sequence[(0, 0)], -0.0001);
    assert_eq!(out.shocks_sequence[(9, 0)], -0.01);
    assert_eq!(out.shocks_sequence[(49, 0)], -0.02);
    assert_eq!(out.shocks_sequence[(130, 0)], 0.0);
}

#[test]
fn nk_zlb_myopic_max_check_ahead() {
    check("nk_zlb_v4", "nk_zlb.mod", &variant(4));
}

#[test]
fn rbc_two_constraints_curb_retrench() {
    check("rbc_2c_v1", "rbc_2c.mod", &variant(1));
}

/// Dynare stops with "Simulation did not converge" (print_info 311); so does dynare-rust.
#[test]
fn nk_zlb_maxit_error() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/occbin/nk_zlb.mod");
    let mf = parse_mod_file(&path, &variant(5)).unwrap();
    let mut s = Session::new(&mf, "test");
    let mut err = None;
    for c in &mf.commands {
        if let Command::Occbin(oc) = c {
            match oc {
                OccbinCommand::Solver(o) => err = s.run_solver(&mf, o).err(),
                OccbinCommand::Setup(o) => s.setup(&mf, o).unwrap(),
                OccbinCommand::SurpriseShocks { .. } => {
                    let tmp = std::env::temp_dir();
                    s.command(&mf, oc, &tmp, "test", &tmp).unwrap()
                }
                _ => {}
            }
        }
    }
    let e = format!("{:#}", err.expect("expected an error"));
    assert!(e.contains("311"), "{e}");
    if let Some(r) = reference() {
        assert!(r["nk_zlb_v5"]["error"].as_str().unwrap().contains("did not converge"));
    }
}
