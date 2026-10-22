//! `external_function` vs Dynare 7.2: reference/external_function/ref_ext_rbc_d<k>.json, written
//! by reference/external_function/dump_external_function.m for ext_rbc.mod, an RBC model whose
//! CES production function is the MATLAB file ces_out.m (loaded by `model::external`).
//!
//! * `DERIVS=1`: gradient and Hessian are extra outputs of ces_out (order 2 to 1e-8),
//! * `DERIVS=2`: they come from the separate files ces_grad.m / ces_hess.m (order 2 to 1e-8),
//! * `DERIVS=0`: Dynare's numerical derivatives (`jacob_element` / `hess_element`, h = 1e-6).
//!   The order-2 terms carry `hess_element`'s rounding noise (~1e-16 / h^2: ghxx is 10x the
//!   analytic one here, in Dynare too); the same formulas on the same libm reproduce Dynare's
//!   values to ~1e-14 all the same.

use dynare::dsge::DsgeModel;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use dynare::solver::korder::{self, KOrderDr};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/external_function")
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// Matrix from jsonencode (array of rows, or a flat vector read as a column).
fn mat(v: &Value) -> Mat {
    match v {
        Value::Array(rows) if rows.first().is_some_and(|x| x.is_array()) => {
            let r = rows.len();
            let c = rows[0].as_array().unwrap().len();
            Mat::from_fn(r, c, |i, j| num(&rows[i][j]))
        }
        Value::Array(a) => Mat::from_fn(a.len(), 1, |i, _| num(&a[i])),
        x => Mat::from_element(1, 1, num(x)),
    }
}

fn solve(derivs: &str, order: usize) -> (Value, Vec<f64>, KOrderDr) {
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(dir().join(format!("ref_ext_rbc_d{derivs}.json"))).unwrap())
            .unwrap();
    let mf = parse_mod_file(&dir().join("ext_rbc.mod"), &[("DERIVS".into(), derivs.into())]).unwrap();
    let dsge = DsgeModel::new(&mf, Some(&[])).unwrap();
    let m = &dsge.model;
    let ys = dsge.ys_calib.clone();
    let dr = korder::solve_model(m, &ys, &m.params, &dsge.base_sigma_e, order, Default::default()).unwrap();
    (r, ys, dr)
}

fn field<'a>(dr: &'a KOrderDr, f: &str) -> &'a Mat {
    match f {
        "ghx" => &dr.first.ghx,
        "ghu" => &dr.first.ghu,
        "ghxx" => &dr.ghxx,
        "ghxu" => &dr.ghxu,
        "ghuu" => &dr.ghuu,
        "ghs2" => &dr.ghs2,
        _ => unreachable!(),
    }
}

const ALL: [&str; 6] = ["ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2"];

fn compare(derivs: &str, fields: &[(&str, f64)]) -> KOrderDr {
    let (r, ys, dr) = solve(derivs, 2);
    let ys_ref = mat(&r["steady_state"]);
    for (i, y) in ys.iter().enumerate() {
        assert!(
            (y - ys_ref[(i, 0)]).abs() < 1e-8 * (1.0 + y.abs()),
            "DERIVS={derivs}: steady state {i}: {y} vs {}",
            ys_ref[(i, 0)]
        );
    }
    let ov: Vec<usize> = mat(&r["order_var"]).iter().map(|&v| v as usize - 1).collect();
    assert_eq!(dr.first.order_var, ov, "DERIVS={derivs}: order_var");
    for &(f, tol) in fields {
        let want = mat(&r[f]);
        let got = field(&dr, f);
        let want = if want.shape() != got.shape() && want.len() == got.len() {
            Mat::from_column_slice(got.nrows(), got.ncols(), want.as_slice())
        } else {
            want
        };
        assert_eq!(got.shape(), want.shape(), "DERIVS={derivs}: {f} shape");
        let e = (got - &want).abs().max();
        println!("DERIVS={derivs} {f}: max abs diff {e:.2e}");
        assert!(e < tol * (1.0 + want.abs().max()), "DERIVS={derivs}: {f} differs by {e:.2e}");
    }
    dr
}

#[test]
fn provided_derivatives_same_function() {
    compare("1", &ALL.map(|f| (f, 1e-8)));
}

#[test]
fn provided_derivatives_separate_files() {
    compare("2", &ALL.map(|f| (f, 1e-8)));
}

#[test]
fn numerical_derivatives() {
    // Finite differences of the external function, as Dynare takes them (the order-2 terms
    // are ~1e-1 away from the analytic DERIVS=1 result, so they are compared to Dynare's own
    // numerical result). First order is robust; second differences amplify last-bit libm
    // differences by 1/h^2, so the order-2 fields only need to agree to 1e-3. That holds with
    // Apple's libm (MATLAB's on macOS); with another libm the order-2 terms are different
    // rounding noise of the same ~1e-1 size (ghxx differs by 6e-2 on Linux and Windows).
    let o2 = if cfg!(target_os = "macos") { 1e-3 } else { 2e-1 };
    compare("0", &[("ghx", 1e-8), ("ghu", 1e-8), ("ghxx", o2), ("ghxu", o2), ("ghuu", o2), ("ghs2", o2)]);
}

/// A Rust closure registered under the function's name replaces the .m file.
#[test]
fn rust_registered_function() {
    use dynare::model::external::register_external_function_with_derivatives;
    use std::sync::Arc;
    let r = -0.25f64;
    let f = move |a: &[f64]| (a[1] * a[0].powf(r) + 1.0 - a[1]).powf(1.0 / r);
    let g = move |a: &[f64]| {
        let (k, al) = (a[0], a[1]);
        let kr = k.powf(r);
        let e1 = (al * kr + 1.0 - al).powf(1.0 / r - 1.0);
        vec![al * k.powf(r - 1.0) * e1, (kr - 1.0) * e1 / r]
    };
    let h = move |a: &[f64]| {
        let (k, al) = (a[0], a[1]);
        let kr = k.powf(r);
        let b = al * kr + 1.0 - al;
        let (e1, e2) = (b.powf(1.0 / r - 1.0), b.powf(1.0 / r - 2.0));
        let hkk = al * (r - 1.0) * k.powf(r - 2.0) * e1 + al * al * (1.0 - r) * k.powf(2.0 * r - 2.0) * e2;
        let hka = k.powf(r - 1.0) * e1 + al * k.powf(r - 1.0) * (1.0 / r - 1.0) * e2 * (kr - 1.0);
        let haa = (kr - 1.0).powi(2) * (1.0 / r - 1.0) * e2 / r;
        vec![hkk, hka, hka, haa]
    };
    // a name no .m file has: the .mod is written to a temporary directory without .m files
    register_external_function_with_derivatives("ces_rust_test", f, Some(Arc::new(g)), Some(Arc::new(h)));
    let src = std::fs::read_to_string(dir().join("ext_rbc.mod")).unwrap().replace("ces_out", "ces_rust_test");
    let tmp = std::env::temp_dir().join(format!("dynare_rs_ext_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let path = tmp.join("ext_rbc_rust.mod");
    std::fs::write(&path, src).unwrap();
    let mf = parse_mod_file(&path, &[("DERIVS".into(), "1".into())]).unwrap();
    let dsge = DsgeModel::new(&mf, Some(&[])).unwrap();
    let m = &dsge.model;
    let dr = korder::solve_model(m, &dsge.ys_calib, &m.params, &dsge.base_sigma_e, 2, Default::default()).unwrap();
    let (r, _, _) = solve("1", 2);
    for f in ["ghx", "ghxx", "ghs2"] {
        let want = mat(&r[f]);
        let got = field(&dr, f);
        let want = if want.shape() != got.shape() {
            Mat::from_column_slice(got.nrows(), got.ncols(), want.as_slice())
        } else {
            want
        };
        assert!((got - &want).abs().max() < 1e-8 * (1.0 + want.abs().max()), "Rust closure: {f}");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn unsupported_m_code_is_refused() {
    let tmp = std::env::temp_dir().join(format!("dynare_rs_ext_bad_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("badf.m"), "function y = badf(x)\ny = 0;\nfor i = 1:3\n  y = y + x;\nend\nend\n").unwrap();
    std::fs::write(
        tmp.join("bad.mod"),
        "var y; varexo e; parameters a; a = 1;\nexternal_function(name = badf, nargs = 1);\nmodel;\ny = badf(a) + e;\nend;\n",
    )
    .unwrap();
    let err = parse_mod_file(&tmp.join("bad.mod"), &[]).unwrap_err().to_string();
    assert!(err.contains("badf.m:3") && err.contains("'for' is not supported"), "{err}");
    let _ = std::fs::remove_dir_all(&tmp);
}
