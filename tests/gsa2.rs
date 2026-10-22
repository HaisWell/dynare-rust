//! `sensitivity` completion against Dynare 7.2 (`reference/gsa2/dump.m` on `reference/gsa2/nk_gsa.mod`):
//! `ppost=1` with `rmse=1` (after a short MCMC), `load_rmse=1`, `load_redform=1` and
//! `load_ident_files=1` inside `sensitivity` with `identification=1`.
mod common;

use std::path::{Path, PathBuf};

use dynare::gsa::{self, GsaOutput};
use dynare::modfile::{Command, ModFile, OptValue, Options, parse_mod_file};
use serde_json::Value;

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/gsa2")
}

fn load(name: &str) -> Option<Value> {
    Some(common::json_ref(refdir().join(name)))
}

fn vec1(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vec1).collect(),
        Value::Null => vec![f64::NAN],
        x => vec![x.as_f64().unwrap_or(f64::NAN)],
    }
}

fn mat(v: &Value) -> Vec<Vec<f64>> {
    match v {
        Value::Array(a) if a.first().is_some_and(Value::is_array) => a.iter().map(vec1).collect(),
        Value::Array(a) => a.iter().map(|x| vec![x.as_f64().unwrap_or(f64::NAN)]).collect(),
        _ => vec![],
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-12)
}

fn close(a: &[f64], b: &[f64], what: &str, tol: f64) {
    assert_eq!(a.len(), b.len(), "{what} length");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(rel(*x, *y) < tol, "{what}[{k}]: {x} vs {y}");
    }
}

fn flat(m: &[Vec<f64>]) -> Vec<f64> {
    m.iter().flatten().copied().collect()
}

struct Setup {
    dir: PathBuf,
    mf: ModFile,
    cmds: Vec<(usize, Options)>,
}

/// A private copy of the model (and, with `mcmc`, of Dynare's MCMC files) with the stage's commands.
fn setup(tag: &str, stage: u32, mcmc: bool) -> Setup {
    let dir = std::env::temp_dir().join(format!("dynare_rs_gsa2_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["nk_gsa.mod", "nk_gsa_data.csv"] {
        std::fs::copy(refdir().join(f), dir.join(f)).unwrap();
    }
    if mcmc {
        for sub in ["metropolis", "Output"] {
            std::fs::create_dir_all(dir.join("nk_gsa").join(sub)).unwrap();
            for e in std::fs::read_dir(refdir().join(sub)).unwrap() {
                let e = e.unwrap();
                std::fs::copy(e.path(), dir.join("nk_gsa").join(sub).join(e.file_name())).unwrap();
            }
        }
    }
    let mf = parse_mod_file(&dir.join("nk_gsa.mod"), &[("STAGE".into(), stage.to_string())]).unwrap();
    let cmds = mf
        .commands
        .iter()
        .enumerate()
        .filter_map(|(i, c)| match c {
            Command::Other(n, o) if n == "sensitivity" => Some((i, o.clone())),
            _ => None,
        })
        .collect();
    Setup { dir, mf, cmds }
}

fn with(o: &Options, extra: &[(&str, f64)]) -> Options {
    let mut o = o.clone();
    for (k, v) in extra {
        o.retain(|(kk, _)| kk != k);
        o.push((k.to_string(), OptValue::Num(*v)));
    }
    o
}

fn run(s: &Setup, k: usize, o: &Options) -> GsaOutput {
    gsa::run(&s.mf, &s.dir, "nk_gsa", o, s.cmds[k].0).unwrap().expect("no output")
}

/// ppost + rmse: Dynare's own posterior draws through our RMSE mapping, then our draws from Dynare's chain.
#[test]
fn ppost_rmse() {
    let r = load("ref_gsa2_stage10.json").expect("reference/setup unavailable");
    let s = setup("post", 10, true);
    let out = run(&s, 0, &s.cmds[0].1);
    let ours = out.rmse.as_ref().unwrap();
    // our draws: a subsample of Dynare's chain, sub_draws = 100
    assert_eq!(out.stab.istable.len(), 100);
    assert_eq!(ours.rmse_mc.len(), 100);
    assert!(ours.logpo2.iter().all(|v| v.is_finite()));
    let x_dyn = mat(&r["x"]);
    let np = x_dyn[0].len();
    for j in 0..np {
        let m = |rows: &[Vec<f64>]| rows.iter().map(|v| v[j]).sum::<f64>() / rows.len() as f64;
        let md = m(&x_dyn);
        let sd = (x_dyn.iter().map(|v| (v[j] - md).powi(2)).sum::<f64>() / x_dyn.len() as f64).sqrt();
        let xs: Vec<Vec<f64>> = out.stab.istable.iter().map(|&k| out.stab.point(k)).collect();
        assert!((m(&xs) - md).abs() < 0.6 * sd, "posterior mean of parameter {j}: {} vs {md}", m(&xs));
    }
    // Dynare's draws: logpo2 (stock_logpo), likelihood, rmse_MC, r2_MC, and the fit at the posterior mean
    let n0 = 3; // shock parameters
    let lp0: Vec<Vec<f64>> = x_dyn.iter().map(|v| v[..n0].to_vec()).collect();
    let lp: Vec<Vec<f64>> = x_dyn.iter().map(|v| v[n0..].to_vec()).collect();
    let stab = gsa::stability::classify_sample(&out.dsge, lp0, lp, false);
    let res = gsa::rmse::run(&out.ctx, &s.mf, &s.dir, &out.dsge, &out.options, &out.options.raw, &stab).unwrap();
    close(&res.logpo2, &vec1(&r["logpo2"]), "logpo2", 1e-7);
    close(&res.logpo2, &vec1(&r["param_logpo"]), "logpo2 vs stock_logpo", 1e-7);
    close(&res.likelihood, &vec1(&r["likelihood"]), "likelihood", 1e-7);
    close(&flat(&res.rmse_mc), &flat(&mat(&r["rmse_MC"])), "rmse_MC", 1e-6);
    close(&flat(&res.r2_mc), &flat(&mat(&r["r2_MC"])), "r2_MC", 1e-6);
    // the saved file carries Dynare's variables (the fit at the posterior mean: Dynare's xparam1 of `_mean.mat`)
    let saved = dynare::matfile::read(&out.ctx.out_dir.join("nk_gsa_post.mat")).unwrap();
    for k in ["x", "logpo2", "likelihood", "rmse_MC", "r2_MC", "rmse_pmean", "r2_pmean", "vvarvecm"] {
        assert!(saved.contains_key(k), "nk_gsa_post.mat lacks {k}");
    }
    let pm = saved["rmse_pmean"].as_vec().unwrap();
    assert_eq!(pm.len(), 3);
    assert!(pm.iter().all(|v| v.is_finite() && *v > 0.0));
    let _ = std::fs::remove_dir_all(&s.dir);
}

/// load_rmse: the second call reads back what the first saved (Dynare's results, rmse_MC / r2_MC / logpo2).
#[test]
fn load_rmse_reloads_saved_results() {
    let r = load("ref_gsa2_stage11.json").expect("reference/setup unavailable");
    let s = setup("rmse", 11, false);
    let first = run(&s, 0, &s.cmds[0].1);
    let a = first.rmse.as_ref().unwrap();
    close(&flat(&a.rmse_mc), &flat(&mat(&r["rmse_MC"])), "rmse_MC", 1e-5);
    // doctor the saved file: a load must return these values, not recompute
    let path = first.ctx.out_dir.join("nk_gsa_prior_rmse.mat");
    let mut m = dynare::matfile::read(&path).unwrap();
    let rm = m["rmse_MC"].as_matrix().unwrap() * 2.0;
    m.insert("rmse_MC".into(), dynare::matfile::MatValue::matrix(&rm));
    let vars: Vec<(&str, &dynare::matfile::MatValue)> = m.iter().map(|(k, v)| (k.as_str(), v)).collect();
    dynare::matfile::write_values(&path, &vars).unwrap();
    let second = run(&s, 1, &s.cmds[1].1);
    assert!(second.options.load_rmse);
    let b = second.rmse.as_ref().unwrap();
    close(&flat(&b.rmse_mc), &flat(&a.rmse_mc).iter().map(|v| 2.0 * v).collect::<Vec<_>>(), "loaded rmse_MC", 1e-12);
    close(&b.logpo2, &a.logpo2, "loaded logpo2", 1e-12);
    close(&flat(&b.r2_mc), &flat(&a.r2_mc), "loaded r2_MC", 1e-12);
    // Dynare's own file layout (variables in `<fname>_prior.mat`) is read as well
    let _ = std::fs::remove_file(&path);
    let dyn_vars = ["x", "logpo2", "likelihood", "rmse_MC", "r2_MC"];
    let xr = mat(&r["x"]);
    let xm = dynare::linalg::Mat::from_fn(xr.len(), xr[0].len(), |i, j| xr[i][j]);
    let (lp, lk) = (vec1(&r["logpo2"]), vec1(&r["likelihood"]));
    let (rm, r2) = (mat(&r["rmse_MC"]), mat(&r["r2_MC"]));
    let to_m = |m: &[Vec<f64>]| dynare::linalg::Mat::from_fn(m.len(), m[0].len(), |i, j| m[i][j]);
    let (rmm, r2m) = (to_m(&rm), to_m(&r2));
    let l0 = to_m(&xr.iter().map(|v| v[..3].to_vec()).collect::<Vec<_>>());
    let l1 = to_m(&xr.iter().map(|v| v[3..].to_vec()).collect::<Vec<_>>());
    use dynare::matfile::MatOut;
    let vars = [
        ("lpmat0", MatOut::Matrix(&l0)),
        ("lpmat", MatOut::Matrix(&l1)),
        (dyn_vars[0], MatOut::Matrix(&xm)),
        (dyn_vars[1], MatOut::Vector(&lp)),
        (dyn_vars[2], MatOut::Vector(&lk)),
        (dyn_vars[3], MatOut::Matrix(&rmm)),
        (dyn_vars[4], MatOut::Matrix(&r2m)),
    ];
    dynare::matfile::write(&first.ctx.out_dir.join("nk_gsa_prior.mat"), &vars).unwrap();
    let third = run(&s, 1, &with(&s.cmds[1].1, &[("load_stab", 1.0)]));
    close(&flat(&third.rmse.as_ref().unwrap().rmse_mc), &flat(&mat(&r["rmse_MC"])), "rmse_MC from Dynare's file", 1e-9);
    let _ = std::fs::remove_dir_all(&s.dir);
}

/// load_redform: the saved `gsa_` of each entry (`<fname>_<type>_<y>_vs_<x>_map.mat`) replaces the recomputation.
#[test]
fn load_redform_reloads_saved_mapping() {
    let _ = load("ref_gsa2_stage11.json").expect("reference/setup unavailable");
    let s = setup("redform", 7, false);
    let o = with(&s.cmds[0].1, &[("Nsam", 256.0)]);
    let first = run(&s, 0, &o);
    assert!(!first.redform_si.is_empty());
    let dir = first.ctx.out_dir.join("redform_prior").join("y_vs_e_i");
    let file = dir.join("nk_gsa_prior_y_vs_e_i_map.mat");
    assert!(file.exists(), "{} not written", file.display());
    // doctor the file, then load
    let np = first.redform_si[0].1.len();
    let fake: Vec<f64> = (0..np).map(|k| 0.01 * (k + 1) as f64).collect();
    let g = dynare::matfile::MatValue::structure(vec![("si", dynare::matfile::MatValue::column(&fake))]);
    dynare::matfile::write_values(&file, &[("gsa_", &g)]).unwrap();
    let second = run(&s, 0, &with(&o, &[("load_redform", 1.0)]));
    let (_, si) = second.redform_si.iter().find(|(l, _)| l == "y vs e_i").expect("y vs e_i");
    close(si, &fake, "loaded si", 1e-12);
    // the other entries come back as computed
    let (_, orig) = first.redform_si.iter().find(|(l, _)| l == "y vs e_i").unwrap();
    assert_ne!(orig, si);
    let _ = std::fs::remove_dir_all(&s.dir);
}

/// load_ident_files inside sensitivity(identification=1): the Morris screening is read back.
#[test]
fn load_ident_files_reloads_screening() {
    let _ = load("ref_gsa2_stage11.json").expect("reference/setup unavailable");
    let s = setup("ident", 1, false);
    let o = with(&s.cmds[0].1, &[("identification", 1.0), ("Nsam", 64.0), ("morris_ntra", 8.0)]);
    let first = run(&s, 0, &o);
    let a = first.ident_sa.as_ref().expect("screening");
    let second = run(&s, 0, &with(&o, &[("load_ident_files", 1.0)]));
    let b = second.ident_sa.as_ref().expect("loaded screening");
    close(&flat(&b.sa_vdec), &flat(&a.sa_vdec), "SAvdec", 1e-14);
    close(&flat(&b.sa_cc), &flat(&a.sa_cc), "SAcc", 1e-14);
    close(&flat(&b.sa_norm), &flat(&a.sa_norm), "SAnorm", 1e-14);
    let _ = std::fs::remove_dir_all(&s.dir);
}
