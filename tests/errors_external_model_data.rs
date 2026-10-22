//! Clean errors instead of panics: runtime errors in a user's external MATLAB function, order 3
//! with an external function, a model without equations (only BVAR files may have none), an
//! `xls_range` with reversed corners; and `second_deriv_provided = NAME` registered from Rust
//! with only a Hessian (reference/external_function, ref_ext_rbc_d1.json).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dynare::data::{DataOptions, Table};
use dynare::dsge::DsgeModel;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use dynare::solver::korder;
use serde_json::Value;

fn ext_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/external_function")
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dynare_rs_errs_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run_cli(dir: &Path, file: &str) -> (bool, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_dynare-rs")).arg(file).current_dir(dir).output().unwrap();
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn runtime_error_in_external_m_function_is_a_clean_error() {
    let d = tmp("ext_rt");
    // the output is not assigned when x <= 100; an out-of-bounds index in the second one
    std::fs::write(d.join("badf.m"), "function y = badf(x)\nif x > 100\n  y = 1;\nend\nend\n").unwrap();
    std::fs::write(d.join("badg.m"), "function y = badg(x)\nv = [1, 2];\ny = v(3) * x;\nend\n").unwrap();
    for f in ["badf", "badg"] {
        std::fs::write(
            d.join(format!("m_{f}.mod")),
            format!("var y; varexo e; parameters a; a = 1;\nexternal_function(name = {f}, nargs = 1);\nmodel;\ny = {f}(a) + e;\nend;\nsteady;\nstoch_simul(order = 1, irf = 0);\n"),
        )
        .unwrap();
        let (ok, err) = run_cli(&d, &format!("m_{f}.mod"));
        assert!(!ok && err.contains("ERROR:") && err.contains(f) && !err.contains("panicked"), "{f}: {err}");
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn third_order_with_external_function_is_refused() {
    let mf = parse_mod_file(&ext_dir().join("ext_rbc.mod"), &[("DERIVS".into(), "1".into())]).unwrap();
    let dsge = DsgeModel::new(&mf, Some(&[])).unwrap();
    let m = &dsge.model;
    let err = korder::solve_model(m, &dsge.ys_calib, &m.params, &dsge.base_sigma_e, 3, Default::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("external functions") && err.contains("order"), "{err}");
    // and from the command line
    let d = tmp("ext_o3");
    for f in ["ext_rbc.mod", "ces_out.m"] {
        std::fs::copy(ext_dir().join(f), d.join(f)).unwrap();
    }
    let src = std::fs::read_to_string(d.join("ext_rbc.mod"))
        .unwrap()
        .replace("stoch_simul(order = 2", "stoch_simul(order = 3");
    std::fs::write(d.join("ext_rbc.mod"), src).unwrap();
    let (ok, err) = run_cli(&d, "ext_rbc.mod");
    assert!(!ok && err.contains("ERROR:") && !err.contains("panicked"), "{err}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn second_derivative_named_rust_function_with_only_a_hessian() {
    use dynare::model::external::register_external_function_with_derivatives;
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
    register_external_function_with_derivatives("ces_test_value", f, Some(Arc::new(g)), None);
    // registered with a (meaningless) scalar value and only a Hessian
    register_external_function_with_derivatives("ces_test_hess", |_: &[f64]| 123.0, None, Some(Arc::new(h)));
    let src = std::fs::read_to_string(ext_dir().join("ext_rbc.mod"))
        .unwrap()
        .replace(
            "first_deriv_provided, second_deriv_provided",
            "first_deriv_provided, second_deriv_provided = ces_test_hess",
        )
        .replace("ces_out", "ces_test_value");
    let d = tmp("ext_hess");
    std::fs::write(d.join("ext_rbc_h.mod"), src).unwrap();
    let mf = parse_mod_file(&d.join("ext_rbc_h.mod"), &[("DERIVS".into(), "1".into())]).unwrap();
    let dsge = DsgeModel::new(&mf, Some(&[])).unwrap();
    let m = &dsge.model;
    let dr = korder::solve_model(m, &dsge.ys_calib, &m.params, &dsge.base_sigma_e, 2, Default::default()).unwrap();
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(ext_dir().join("ref_ext_rbc_d1.json")).unwrap()).unwrap();
    for (f, got) in [("ghxx", &dr.ghxx), ("ghs2", &dr.ghs2)] {
        let rows = r[f].as_array().unwrap();
        let want: Vec<f64> = if rows.first().is_some_and(|x| x.is_array()) {
            // row-major in the JSON; compare column-major
            let (nr, nc) = (rows.len(), rows[0].as_array().unwrap().len());
            (0..nc).flat_map(|j| (0..nr).map(move |i| (i, j))).map(|(i, j)| rows[i][j].as_f64().unwrap()).collect()
        } else {
            rows.iter().map(|x| x.as_f64().unwrap()).collect()
        };
        let got: &Mat = got;
        assert_eq!(got.len(), want.len(), "{f}");
        let wmax = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        for (x, y) in got.iter().zip(&want) {
            assert!((x - y).abs() < 1e-8 * (1.0 + wmax), "{f}: {x} vs Dynare {y}");
        }
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn model_without_equations_only_for_bvar_files() {
    let d = tmp("empty_model");
    std::fs::write(d.join("e1.mod"), "var y c;\nvarexo e;\nmodel;\nend;\nstoch_simul;\n").unwrap();
    let err = parse_mod_file(&d.join("e1.mod"), &[]).unwrap_err().to_string();
    assert!(err.contains("At least one model equation must be declared"), "{err}");
    std::fs::write(d.join("e2.mod"), "var y c;\nvarexo e;\nstoch_simul;\n").unwrap();
    assert!(parse_mod_file(&d.join("e2.mod"), &[]).is_err());
    // BVAR files need no model
    std::fs::write(
        d.join("b.mod"),
        "var y c;\nvarobs y c;\nbvar_density(datafile = d, first_obs = 1, presample = 4) 2;\n",
    )
    .unwrap();
    parse_mod_file(&d.join("b.mod"), &[]).unwrap();
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn xls_range_with_reversed_corners_is_refused() {
    let f = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/data/sw07_obs.xlsx");
    for r in ["D100:B3", "B100:D3", "D3:B100"] {
        let opts = DataOptions { xls_sheet: Some("data".into()), xls_range: Some(r.into()) };
        let err = Table::read(&f, &opts).unwrap_err().to_string();
        assert!(err.contains("xls_range") && err.contains("upper-left"), "{r}: {err}");
    }
}
