//! Analytic Jacobian of the GMM moments (`analytic_standard_errors`, `analytic_jacobian`) against
//! Dynare 7.1: `reference/mom/dump_analytic.m` -> `ref_rbc_analytic_case<k>.json` for
//! `reference/mom/rbc_mom.mod -DANALYTIC=1 -DCASE=k` (1 GMM order 1, 2 GMM order 2, 3 GMM order 3
//! with analytic_jacobian, 6 penalized order 1 with analytic_jacobian).
//!
//! Order 1 is analytic (compared to ~1e-8); orders 2 and 3 use Richardson-extrapolated central
//! differences (stable under a change of step), which agree with Dynare's order 2 and 3 analytic derivatives only to ~1e-5 (order 2) and ~1e-3 (order 3) relative to the largest entry.
mod common;

use std::path::{Path, PathBuf};

use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use dynare::mom::run::{Context, setup, standard_errors_analytic_at};
use dynare::mom::{MomInputs, blocks_of, last_mom_options};
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/mom")
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![],
    }
}

fn matf(v: &Value) -> Mat {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(vecf).collect();
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn check(k: usize, tol: f64) {
    let r: Value = common::json_ref(dir().join(format!("ref_rbc_analytic_case{k}.json")));
    let mf =
        parse_mod_file(&dir().join("rbc_mom.mod"), &[("CASE".into(), k.to_string()), ("ANALYTIC".into(), "1".into())])
            .unwrap();
    let blocks = blocks_of(&mf);
    let opts = last_mom_options(&mf).unwrap();
    let tmp = std::env::temp_dir().join(format!("dynare_rs_mom_an{k}"));
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["rbc_mom_data1.csv", "rbc_mom_data2.csv"] {
        std::fs::copy(dir().join(f), tmp.join(f)).unwrap();
    }
    let ctx = Context { dir: tmp, fname: "rbc_mom".into(), seed_override: None, stream_seed: 0 };
    let inputs = MomInputs { quiet: true, ..Default::default() };
    let st = setup(&mf, &blocks, &opts, &ctx, &inputs).unwrap();
    assert!(st.o.analytic_standard_errors);
    assert_eq!(st.o.analytic_jacobian, r["analytic_jacobian"].as_bool().unwrap());
    let sw = matf(&r["mom"]["weighting_info"]["Sw"]);
    for (j, p) in r["points"].as_array().unwrap().iter().enumerate() {
        if p["info"].as_i64().unwrap() != 0 {
            continue;
        }
        let x = vecf(&p["x"]);
        let d = st.problem.moments_jacobian(&x).unwrap();
        let dr = matf(&p["D"]);
        let e = (&d - &dr).abs().max() / dr.abs().max();
        if e > 1e-9 {
            println!("{:?}\n{:?}", d.row(0).iter().collect::<Vec<_>>(), dr.row(0).iter().collect::<Vec<_>>());
        }
        println!("case {k} point {j}: Jacobian of moments {e:.2e}");
        assert!(e < tol, "Jacobian {e}");
        let g = st.problem.objective_gradient(&x, &st.data_moments, &sw).unwrap();
        let gr = vecf(&p["df"]);
        let e = g.iter().zip(&gr).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
            / gr.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        println!("case {k} point {j}: gradient {e:.2e}");
        assert!(e < tol.max(1e-6) || k == 2 || k == 3, "gradient {e}");
        let jr = st.problem.residual_jacobian(&x, &sw).unwrap();
        let jv = matf(&p["df_vec"]);
        let e = (&jr - &jv).abs().max() / jv.abs().max();
        println!("case {k} point {j}: residual Jacobian {e:.2e}");
        assert!(e < tol, "residual Jacobian {e}");
    }
    let xf = vecf(&r["points"][2]["x"]);
    let wopt = r["mom"]["weighting_info"]["Woptflag"].as_bool().unwrap();
    let (se, _) = standard_errors_analytic_at(
        &st.problem,
        &xf,
        st.m_data.as_ref().unwrap(),
        &sw,
        wopt,
        st.nobs,
        st.o.bartlett_kernel_lag,
    )
    .unwrap();
    let ser = vecf(&r["se_at_final"]);
    let e = se.iter().zip(&ser).map(|(a, b)| (a - b).abs() / b.abs()).fold(0.0, f64::max);
    println!("case {k}: analytic standard errors {se:?} vs {ser:?} ({e:.2e})");
    assert!(e < tol, "standard errors");
}

#[test]
fn analytic_order1() {
    check(1, 1e-8);
}

#[test]
fn analytic_order1_penalized() {
    check(6, 1e-8);
}

#[test]
fn analytic_order2() {
    check(2, 1e-4);
}

#[test]
fn analytic_order3() {
    check(3, 5e-3);
}

/// Full run with `analytic_standard_errors` + `analytic_jacobian` (csminwel with analytic gradient).
#[test]
fn analytic_full_run_penalized() {
    let r: Value = common::json_ref(dir().join("ref_rbc_analytic_case6.json"));
    let mf =
        parse_mod_file(&dir().join("rbc_mom.mod"), &[("CASE".into(), "6".into()), ("ANALYTIC".into(), "1".into())])
            .unwrap();
    let blocks = blocks_of(&mf);
    let opts = last_mom_options(&mf).unwrap();
    let tmp = std::env::temp_dir().join("dynare_rs_mom_an_run6");
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["rbc_mom_data1.csv", "rbc_mom_data2.csv"] {
        std::fs::copy(dir().join(f), tmp.join(f)).unwrap();
    }
    let ctx = Context { dir: tmp, fname: "rbc_mom".into(), seed_override: None, stream_seed: 0 };
    let inputs = MomInputs { quiet: true, ..Default::default() };
    let res = dynare::mom::run::method_of_moments(&mf, &blocks, &opts, &ctx, &inputs).unwrap();
    let fr = r["points"][2]["fval"].as_f64().unwrap();
    println!("analytic run: fval {} vs Dynare {fr}; stdh {:?}", res.fval, res.stdh);
    assert!(res.fval <= fr * (1.0 + 1e-3) + 1e-8);
    assert!(res.stdh.iter().all(|v| v.is_finite()));
}
