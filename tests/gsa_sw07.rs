//! Global sensitivity analysis of the SW07 model against Dynare 7.2
//! (`reference/gsa/dump_sw07_gsa.m`, STAGE 20/21 of `reference/sw07/sw07.mod`).
//!
//! 29 structural parameters (LPTAU points) and 11 shock std's sampled by Latin hypercube.
//! STAGE 20 uses the stationary filter's `qz_criterium = 1-1e-6` (every draw has unit roots:
//! nothing is stable), STAGE 21 the diffuse filter and the RMSE mapping. Dynare scrambles the
//! LPTAU columns with `randperm` above 30 structural parameters or with `ilptau=2`; SW07 has
//! 29, so STAGE 20 is also run with `ilptau=2` to cover that branch.
mod common;

use std::path::{Path, PathBuf};

use dynare::dsge::DsgeModel;
use dynare::gsa::{self, GsaOutput};
use dynare::modfile::{Command, OptValue, parse_mod_file};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn load(name: &str) -> Option<Value> {
    Some(common::json_ref(root().join("reference/gsa").join(name)))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vec1(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(num).collect(),
        Value::Null => vec![],
        x => vec![num(x)],
    }
}

fn mat(v: &Value) -> Vec<Vec<f64>> {
    match v {
        Value::Array(a) if a.first().is_some_and(Value::is_array) => a.iter().map(vec1).collect(),
        Value::Array(a) => a.iter().map(|x| vec![num(x)]).collect(),
        _ => vec![],
    }
}

fn idx(v: &Value) -> Vec<usize> {
    vec1(v).into_iter().map(|x| x as usize - 1).collect()
}

/// Relative difference; a JSON `null` (MATLAB NaN or +-Inf, e.g. -Inf log posterior where the
/// diffuse filter fails) matches any non-finite value.
fn rel(a: f64, b: f64) -> f64 {
    if b.is_nan() { if a.is_finite() { f64::INFINITY } else { 0.0 } } else { (a - b).abs() / b.abs().max(1.0) }
}

/// see tests/gsa_nk.rs: SAMPLE_TOL (prior-range tails of gaminv/betainv; 9e-11 here)
const SAMPLE_TOL: f64 = 1e-9;

fn run_stage(stage: u32, f: impl FnOnce(&GsaOutput, &dynare::modfile::ModFile, &Path)) -> GsaOutput {
    let tmp = std::env::temp_dir().join(format!("dynare_rs_gsa_sw07_{}_{stage}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["sw07.mod", "sw07_obs.csv"] {
        std::fs::copy(root().join("reference/sw07").join(f), tmp.join(f)).unwrap();
    }
    let mf = parse_mod_file(&tmp.join("sw07.mod"), &[("STAGE".into(), stage.to_string())]).unwrap();
    let (ci, o) = mf
        .commands
        .iter()
        .enumerate()
        .find_map(|(i, c)| match c {
            Command::Other(n, o) if n == "sensitivity" => Some((i, o.clone())),
            _ => None,
        })
        .expect("no sensitivity command");
    let out = gsa::run(&mf, &tmp, "sw07", &o, ci).unwrap();
    let out = out.unwrap_or_else(|| panic!("stage {stage}: no output"));
    f(&out, &mf, &tmp);
    let _ = std::fs::remove_dir_all(&tmp);
    out
}

fn check(out: &GsaOutput, r: &Value) {
    let (l, l0) = (mat(&r["lpmat"]), mat(&r["lpmat0"]));
    assert_eq!(out.stab.lpmat.len(), l.len());
    let mut worst: f64 = 0.0;
    for (a, b) in out.stab.lpmat.iter().zip(&l).chain(out.stab.lpmat0.iter().zip(&l0)) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            worst = worst.max(rel(*x, *y));
        }
    }
    println!("sample: {worst:e}");
    assert!(worst < SAMPLE_TOL, "sample differs by {worst:e}");
    assert_eq!(out.stab.istable, idx(&r["istable"]), "istable");
    assert_eq!(out.stab.iindeterm, idx(&r["iindeterm"]), "iindeterm");
    assert_eq!(out.stab.ixun, idx(&r["ixun"]), "ixun");
    assert_eq!(out.stab.iwrong, idx(&r["iwrong"]), "iwrong");
    let n = check_infox(&out.dsge, &out.stab.infox, r);
    println!("infox: identical but for {n} draws with a non-finite residual at the initial guess");
}

/// `infox` must equal Dynare's, with one exception. Where the static residuals are not finite
/// at the initial guess (`oo_.steady_state` with the draw's parameters), Dynare's
/// `dynare_solve` retries from `rand`/`randn` guesses (`solve_randomize_initial_guess`), which
/// cannot be reproduced; when the retry fails too Dynare reports info 20 at a finite point,
/// dynare-rust info 22 at the guess (both "wrong", the classification is the same). Those
/// draws are allowed, after checking that the residual is indeed non-finite at the guess.
fn check_infox(dsge: &DsgeModel, ours: &[i32], r: &Value) -> usize {
    let infox: Vec<i32> = vec1(&r["infox"]).into_iter().map(|x| x as i32).collect();
    assert_eq!(ours.len(), infox.len(), "infox length");
    let (l, l0) = (mat(&r["lpmat"]), mat(&r["lpmat0"]));
    let m = &dsge.model;
    let x_ss = vec![0.0; m.n_exo()];
    let mut n = 0;
    for (j, (&a, &b)) in ours.iter().zip(&infox).enumerate() {
        if a == b {
            continue;
        }
        assert!(a == 22 && b == 20, "infox[{j}]: {a} vs Dynare {b}");
        let x: Vec<f64> = l0[j].iter().chain(&l[j]).copied().collect();
        let (params, _) = dsge.set_all_parameters(&x);
        let mut g = dsge.ys_guess.clone();
        m.set_auxiliary_variables(&mut g, &x_ss, &params);
        let mut res = vec![0.0; m.n_endo()];
        m.static_residuals(&g, &x_ss, &params, &mut res);
        assert!(res.iter().any(|v| !v.is_finite()), "infox[{j}]: 22 vs Dynare 20 with a finite initial residual");
        n += 1;
    }
    n
}

/// Stationary filter: unit roots make every solvable draw "explosive", as in Dynare.
fn stage20_stability_map(refname: &str, ilptau2: bool) {
    let r = load(refname).expect("reference/setup unavailable");
    // no stable draw: gsa::run stops after the stability map
    let tag = if ilptau2 { "20s" } else { "20" };
    let tmp = std::env::temp_dir().join(format!("dynare_rs_gsa_sw07_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["sw07.mod", "sw07_obs.csv"] {
        std::fs::copy(root().join("reference/sw07").join(f), tmp.join(f)).unwrap();
    }
    let mf = parse_mod_file(&tmp.join("sw07.mod"), &[("STAGE".into(), "20".into())]).unwrap();
    let (ci, mut o) = mf
        .commands
        .iter()
        .enumerate()
        .find_map(|(i, c)| match c {
            Command::Other(n, o) if n == "sensitivity" => Some((i, o.clone())),
            _ => None,
        })
        .unwrap();
    if ilptau2 {
        // as sensitivity(..., ilptau=2) in dump_sw07_gsa.m
        o.push(("ilptau".into(), OptValue::Num(2.0)));
    }
    let res = gsa::run(&mf, &tmp, "sw07", &o, ci).unwrap();
    assert!(res.is_none(), "Dynare finds no stable draw at qz_criterium = 1-1e-6");
    // the saved sample and classification
    let vars = dynare::matfile::read(&tmp.join("sw07/gsa/sw07_prior.mat")).unwrap();
    let _ = std::fs::remove_dir_all(&tmp);
    let get = |k: &str| vars.get(k).and_then(|v| v.as_matrix()).unwrap();
    let rows = |m: &dynare::linalg::Mat| -> Vec<Vec<f64>> {
        (0..m.nrows()).map(|i| m.row(i).iter().copied().collect()).collect()
    };
    let (l, l0) = (rows(&get("lpmat")), rows(&get("lpmat0")));
    let (rl, rl0) = (mat(&r["lpmat"]), mat(&r["lpmat0"]));
    assert_eq!((l.len(), l0.len()), (2048, 2048));
    assert_eq!((l[0].len(), l0[0].len()), (29, 11));
    assert_eq!(l.len(), rl.len());
    assert_eq!(l0.len(), rl0.len());
    let mut worst: f64 = 0.0;
    for (a, b) in l.iter().zip(&rl).chain(l0.iter().zip(&rl0)) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            worst = worst.max(rel(*x, *y));
        }
    }
    println!("{refname}: sample {worst:e}");
    assert!(worst < SAMPLE_TOL, "sample differs by {worst:e}");
    let v1 = |k: &str| -> Vec<usize> {
        vars.get(k).and_then(|v| v.as_vec()).unwrap_or_default().into_iter().map(|x| x as usize - 1).collect()
    };
    assert_eq!(v1("istable"), idx(&r["istable"]));
    assert_eq!(v1("iindeterm"), idx(&r["iindeterm"]));
    assert_eq!(v1("ixun"), idx(&r["ixun"]));
    assert_eq!(v1("iwrong"), idx(&r["iwrong"]));
    let infox: Vec<i32> = vars.get("infox").and_then(|v| v.as_vec()).unwrap().into_iter().map(|x| x as i32).collect();
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let n = check_infox(&dsge, &infox, &r);
    println!("{refname}: infox identical but for {n} draws with a non-finite residual at the initial guess");
}

#[test]
fn sw07_stage20_stability_map() {
    stage20_stability_map("ref_gsa_sw07_stage20.json", false);
}

/// The same with the LPTAU columns scrambled by `randperm` (Dynare's branch for more than 30
/// structural parameters, forced with `ilptau=2`).
#[test]
fn sw07_stage20_scrambled_lptau() {
    stage20_stability_map("ref_gsa_sw07_stage20_ilptau2.json", true);
}

/// Diffuse filter: stability map, then the RMSE mapping on the stable draws.
#[test]
fn sw07_stage21_rmse() {
    let r = load("ref_gsa_sw07_stage21.json").expect("reference/setup unavailable");
    let mut exact = None;
    let out = run_stage(21, |out, mf, dir| {
        let stab = gsa::stability::classify_sample(&out.dsge, mat(&r["lpmat0"]), mat(&r["lpmat"]), false);
        exact = Some(gsa::rmse::run(&out.ctx, mf, dir, &out.dsge, &out.options, &out.options.raw, &stab).unwrap());
    });
    check(&out, &r);
    // At a few draws the prediction-error covariance F is badly conditioned and the log
    // posterior depends on rounding, in Dynare as here: Dynare's own value moves by more than
    // 1e-6 under 1e-14 relative perturbations of the parameters at 2 of the 245 draws
    // (tests/kalman_robust.rs, which checks that dynare-rust agrees at the other 243). So the
    // share of draws that agree is checked (243/245), and the RMSEs (smoother), which agree.
    let stats = |a: &[f64], b: &[f64], what: &str| -> (f64, f64) {
        assert_eq!(a.len(), b.len(), "{what} length");
        let e: Vec<f64> = a.iter().zip(b).map(|(x, y)| rel(*x, *y)).collect();
        let close = e.iter().filter(|&&v| v < 1e-6).count() as f64 / e.len() as f64;
        let mut s = e.clone();
        s.sort_by(f64::total_cmp);
        eprintln!("{what}: {:.0}% of the draws within 1e-6, median {:e}", 100.0 * close, s[s.len() / 2]);
        (close, s[s.len() / 2])
    };
    let flat = |m: &[Vec<f64>]| m.iter().flatten().copied().collect::<Vec<f64>>();
    let ex = exact.unwrap();
    let (c, _) = stats(&ex.logpo2, &vec1(&r["logpo2"]), "logpo2 (Dynare's sample)");
    assert!(c > 0.98);
    let (c, m) = stats(&flat(&ex.rmse_mc), &flat(&mat(&r["rmse_MC"])), "rmse_MC (Dynare's sample)");
    assert!(c > 0.99 && m < 1e-10);
    let (c, m) = stats(&flat(&ex.r2_mc), &flat(&mat(&r["r2_MC"])), "r2_MC (Dynare's sample)");
    assert!(c > 0.99 && m < 1e-10);
    let res = out.rmse.as_ref().unwrap();
    // our own sample (9e-11 from Dynare's, see SAMPLE_TOL)
    let (_, m) = stats(&flat(&res.rmse_mc), &flat(&mat(&r["rmse_MC"])), "rmse_MC");
    assert!(m < 1e-9);
}
