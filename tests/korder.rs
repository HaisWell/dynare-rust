//! Higher-order perturbation vs Dynare 7.2: reference/korder/ref_<model>.json,
//! written by reference/korder/dump_korder.m for rbc.mod and nk.mod.
mod common;

use dynare::dsge::DsgeModel;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use dynare::solver::SolverOptions;
use dynare::solver::korder::{self, KOrderDr, simulate};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/korder")
}

fn load(model: &str) -> Option<Value> {
    Some(common::json_ref(dir().join(format!("ref_{model}.json"))))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(num).collect(),
        x => vec![num(x)],
    }
}

/// Matrix from jsonencode (array of rows, or a flat vector read as a column).
fn mat(v: &Value) -> Mat {
    match v {
        Value::Array(rows) if rows.first().is_some_and(|x| x.is_array()) => {
            let r = rows.len();
            let c = rows[0].as_array().unwrap().len();
            Mat::from_fn(r, c, |i, j| num(&rows[i][j]))
        }
        other => {
            let f = vecf(other);
            Mat::from_column_slice(f.len(), 1, &f)
        }
    }
}

fn max_abs_diff(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    (a - b).abs().max()
}

struct Case {
    r: Value,
    dsge: DsgeModel,
    ys: Vec<f64>,
}

fn setup(model: &str) -> Option<Case> {
    let r = load(model)?;
    let mf = parse_mod_file(&dir().join(format!("{model}.mod")), &[("ORDER".into(), "3".into())]).unwrap();
    let dsge = DsgeModel::new(&mf, Some(&[])).unwrap();
    let ys = dsge.ys_calib.clone();
    Some(Case { r, dsge, ys })
}

fn solve(c: &Case, order: usize) -> KOrderDr {
    let m = &c.dsge.model;
    korder::solve_model(m, &c.ys, &m.params, &c.dsge.base_sigma_e, order, SolverOptions::default()).unwrap()
}

fn check_derivatives(model: &str) {
    let c = setup(model).expect("reference/setup unavailable");
    let m = &c.dsge.model;
    let ys_ref = vecf(&c.r["ys"]);
    for (a, b) in c.ys.iter().zip(&ys_ref) {
        assert!((a - b).abs() < 1e-10, "{model}: steady state {a} vs {b}");
    }
    let x = vec![0.0; m.n_exo()];
    let g1 = m.dynamic_jacobian(&c.ys, &x, &m.params);
    assert!(max_abs_diff(&g1, &mat(&c.r["g1"])) < 1e-12, "{model}: g1");
    let d = m.dynamic_derivatives(3);
    let v = d.eval(&c.ys, &x, &m.params);
    for (key, k) in [("g2", 2usize), ("g3", 3)] {
        let refm = mat(&c.r[key]);
        let mut want: std::collections::HashMap<Vec<usize>, f64> = std::collections::HashMap::new();
        for i in 0..refm.nrows() {
            let mut idx: Vec<usize> = (0..=k).map(|j| refm[(i, j)] as usize - 1).collect();
            idx[1..].sort_unstable(); // Dynare's indices are unique but not sorted
            assert!(want.insert(idx, refm[(i, k + 1)]).is_none());
        }
        let got: Vec<(Vec<usize>, f64)> = if k == 2 {
            d.g2_index.iter().zip(&v.g2).map(|(&(e, a, b), &x)| (vec![e, a, b], x)).collect()
        } else {
            d.g3_index.iter().zip(&v.g3).map(|(&(e, a, b, cc), &x)| (vec![e, a, b, cc], x)).collect()
        };
        for (idx, val) in &got {
            let w = want.remove(idx).unwrap_or(0.0);
            assert!((val - w).abs() < 1e-10 * (1.0 + w.abs()), "{model} {key}{idx:?}: {val} vs {w}");
        }
        for (idx, w) in want {
            assert!(w.abs() < 1e-14, "{model} {key}{idx:?}: missing entry {w}");
        }
    }
}

const O2: [&str; 6] = ["ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2"];
const O3: [&str; 6] = ["ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss"];

fn field<'a>(dr: &'a KOrderDr, f: &str) -> &'a Mat {
    match f {
        "ghx" => &dr.first.ghx,
        "ghu" => &dr.first.ghu,
        "ghxx" => &dr.ghxx,
        "ghxu" => &dr.ghxu,
        "ghuu" => &dr.ghuu,
        "ghs2" => &dr.ghs2,
        "ghxxx" => &dr.ghxxx,
        "ghxxu" => &dr.ghxxu,
        "ghxuu" => &dr.ghxuu,
        "ghuuu" => &dr.ghuuu,
        "ghxss" => &dr.ghxss,
        "ghuss" => &dr.ghuss,
        _ => unreachable!(),
    }
}

fn check_decision_rules(model: &str) {
    let c = setup(model).expect("reference/setup unavailable");
    let ov: Vec<usize> = vecf(&c.r["order_var"]).iter().map(|&v| v as usize - 1).collect();
    let dr2 = solve(&c, 2);
    assert_eq!(dr2.first.order_var, ov, "{model}: order_var");
    let dr3 = solve(&c, 3);
    let tol = 1e-10;
    for (dr, key, fields) in [(&dr2, "o2", &O2[..]), (&dr3, "o3k", &O2[..]), (&dr3, "o3k", &O3[..])] {
        for f in fields {
            let want = mat(&c.r[key][f]);
            let got = field(dr, f);
            let scale = 1.0 + want.abs().max();
            let e = max_abs_diff(got, &want);
            println!("{model} {key}.{f}: max abs diff {e:.2e} (scale {scale:.2e})");
            assert!(e < tol * scale, "{model} {key}.{f}: {e}");
        }
    }
    for (dr, key, maxk) in [(&dr2, "o2k", 2usize), (&dr3, "o3k", 3)] {
        for j in 0..=maxk {
            let want = mat(&c.r[key][format!("g_{j}")]);
            let got = dr.g_k(j);
            let e = max_abs_diff(&got, &want);
            println!("{model} {key}.g_{j}: max abs diff {e:.2e}");
            assert!(e < tol * (1.0 + want.abs().max()), "{model} {key}.g_{j}: {e}");
        }
    }
}

fn check_simulations(model: &str) {
    let c = setup(model).expect("reference/setup unavailable");
    let dr2 = solve(&c, 2);
    let dr3 = solve(&c, 3);
    let ex = mat(&c.r["sim_ex"]);
    let y0 = vecf(&c.r["sim_y0b"]);
    for (dr, pruning, key) in
        [(&dr2, false, "sim_o2"), (&dr2, true, "sim_o2_pruned"), (&dr3, true, "sim_o3_pruned"), (&dr3, false, "sim_o3")]
    {
        let want = mat(&c.r[key]);
        let got = simulate::simult(dr, &y0, &ex, pruning);
        let e = max_abs_diff(&got, &want);
        println!("{model} {key}: max abs diff {e:.2e}");
        assert!(e < 1e-9 * (1.0 + want.abs().max()), "{model} {key}: {e}");
    }
    // generalized IRFs with MATLAB's draws
    let long = num(&c.r["girf_long"]) as usize;
    let drop = num(&c.r["girf_drop"]) as usize;
    let draws: Vec<Mat> = c.r["girf_draws"].as_array().unwrap().iter().map(mat).collect();
    let cs = mat(&c.r["girf_cs"]);
    let e1: Vec<f64> = cs.column(0).iter().copied().collect();
    for (dr, ord) in [(&dr2, 2), (&dr3, 3)] {
        for p in [0, 1] {
            let key = format!("girf_o{ord}_p{p}");
            let want = mat(&c.r[&key]);
            let got = simulate::girf(dr, &e1, long, drop, &draws, p == 1);
            let e = max_abs_diff(&got, &want);
            println!("{model} {key}: max abs diff {e:.2e}");
            assert!(e < 1e-10 * (1.0 + want.abs().max()), "{model} {key}: {e}");
        }
    }
}

#[test]
fn rbc_derivatives() {
    check_derivatives("rbc");
}
#[test]
fn nk_derivatives() {
    check_derivatives("nk");
}
#[test]
fn rbc_decision_rules() {
    check_decision_rules("rbc");
}
#[test]
fn nk_decision_rules() {
    check_decision_rules("nk");
}
#[test]
fn rbc_simulations() {
    check_simulations("rbc");
}
#[test]
fn nk_simulations() {
    check_simulations("nk");
}

fn check_moments(model: &str) {
    use dynare::solver::korder::pruned;
    let c = setup(model).expect("reference/setup unavailable");
    let sig = &c.dsge.base_sigma_e;
    let dr2 = solve(&c, 2);
    let dr3 = solve(&c, 3);
    let n = dr2.first.ghx.nrows();
    let all: Vec<usize> = (0..n).collect();
    let rel = |got: &Mat, want: &Mat| max_abs_diff(got, want) / (1.0 + want.abs().max());
    for (dr, key) in [(&dr2, "pss2"), (&dr3, "pss3")] {
        let p = pruned::pruned_state_space_system(dr, sig, &all, 5).unwrap();
        let ey = Mat::from_column_slice(n, 1, &p.e_y);
        let e1 = rel(&ey, &mat(&c.r[key]["E_y"]));
        let e2 = rel(&p.var_y, &mat(&c.r[key]["Var_y"]));
        println!("{model} {key}: E_y {e1:.2e}, Var_y {e2:.2e}");
        assert!(e1 < 1e-10 && e2 < 1e-9, "{model} {key}");
    }
    // stoch_simul's displayed moments (var_list = declared variables, declaration order)
    let norig = c.dsge.model.orig_endo_nbr;
    for (dr, key) in [(&dr2, "mom_o2_pruned"), (&dr3, "mom_o3_pruned")] {
        let obs: Vec<usize> = (0..norig).map(|j| dr.first.inv_order_var[j]).collect();
        let p = pruned::pruned_state_space_system(dr, sig, &obs, 5).unwrap();
        let m = Mat::from_column_slice(norig, 1, &p.e_y);
        let e1 = rel(&m, &mat(&c.r[key]["mean"]));
        let e2 = rel(&p.var_y, &mat(&c.r[key]["var"]));
        let ac = c.r[key]["autocorr"].as_array().unwrap();
        let mut e3: f64 = 0.0;
        for (i, a) in ac.iter().enumerate() {
            // NaN (constant variables) on both sides
            let want = mat(a).map(|v| if v.is_nan() { 0.0 } else { v });
            let got = p.corr_yi[i].map(|v| if v.is_nan() { 0.0 } else { v });
            e3 = e3.max(rel(&got, &want));
        }
        println!("{model} {key}: mean {e1:.2e}, var {e2:.2e}, autocorr {e3:.2e}");
        assert!(e1 < 1e-10 && e2 < 1e-9 && e3 < 1e-9, "{model} {key}");
    }
    let m = pruned::second_order_mean(&dr2, sig).unwrap();
    let m = Mat::from_column_slice(norig, 1, &m[..norig]);
    let e = rel(&m, &mat(&c.r["mom_o2"]["mean"]));
    println!("{model} mom_o2 mean: {e:.2e}");
    assert!(e < 1e-10);
}

#[test]
fn rbc_moments() {
    check_moments("rbc");
}
#[test]
fn nk_moments() {
    check_moments("nk");
}

#[test]
fn stoch_simul_entry_point() {
    use dynare::solver::korder::stoch_simul::{StochSimulOptions, stoch_simul};
    let c = setup("rbc").expect("reference/setup unavailable");
    let mf = parse_mod_file(&dir().join("rbc.mod"), &[("ORDER".into(), "3".into())]).unwrap();
    let so = StochSimulOptions {
        order: 3,
        pruning: true,
        irf: 6,
        replic: 20,
        drop: 10,
        noprint: true,
        ..Default::default()
    };
    let out = stoch_simul(&mf, &so, &[], 1).unwrap();
    let want = mat(&c.r["mom_o3_pruned"]["mean"]);
    for (i, m) in out.mean.iter().enumerate() {
        assert!((m - want[(i, 0)]).abs() < 1e-10);
    }
    // 7 variables x 2 shocks, the TFP response of output is positive on impact
    assert_eq!(out.irfs.len(), 14);
    let (_, y_ez) = out.irfs.iter().find(|(k, _)| k == "y_ez").unwrap();
    assert!(y_ez[0] > 0.0 && y_ez.len() == 6);
}
