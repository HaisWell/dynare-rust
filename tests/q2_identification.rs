//! : identification at order 2 and 3 (pruned state space) and `checks_via_subsets`
//! against Dynare 7.2 (reference/q2/ref_q2_ident_c<k>.json, q2_c<k>_matlab.log, written by
//! dump_ident_q2.m / dump_gsa_q2.m).
mod common;

use std::path::Path;

use dynare::identification::run::{IdentOptions, build_context};
use dynare::identification::{Analysis, Persistent};
use dynare::linalg::Mat;
use dynare::modfile::{Command, ModFile, parse_mod_file};
use rand::SeedableRng;
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn load(name: &str) -> Option<Value> {
    Some(common::json_ref(root().join("reference/q2").join(name)))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        x => vec![num(x)],
    }
}

fn mat(v: &Value) -> Mat {
    match v {
        Value::Array(rows) if rows.first().is_some_and(|r| r.is_array()) => {
            let nc = rows[0].as_array().unwrap().len();
            Mat::from_fn(rows.len(), nc, |i, j| num(&rows[i][j]))
        }
        Value::Array(a) => Mat::from_fn(a.len(), 1, |i, _| num(&a[i])),
        x => Mat::from_element(1, 1, num(x)),
    }
}

/// Relative to the largest entry of the reference column.
fn assert_cols_close(name: &str, a: &Mat, b: &Mat, rtol: f64) {
    assert_eq!(a.shape(), b.shape(), "{name}: shape");
    for j in 0..a.ncols() {
        let scale = (0..b.nrows()).map(|i| b[(i, j)].abs()).fold(0.0, f64::max).max(1e-300);
        for i in 0..a.nrows() {
            let d = (a[(i, j)] - b[(i, j)]).abs();
            assert!(d <= rtol * scale, "{name}[{i},{j}]: rust {} vs dynare {} (scale {scale})", a[(i, j)], b[(i, j)]);
        }
    }
}

fn modfile(file: &str, case: usize) -> ModFile {
    parse_mod_file(&root().join("reference/q2").join(file), &[("CASE".into(), case.to_string())]).unwrap()
}

fn ident_options(mf: &ModFile) -> IdentOptions {
    let o = mf
        .commands
        .iter()
        .find_map(|c| match c {
            Command::Other(n, o) if n == "identification" => Some(o.clone()),
            _ => None,
        })
        .expect("identification command");
    IdentOptions::from_options(&o).unwrap()
}

fn point(mf: &ModFile, theta_ref: &[f64]) -> Analysis {
    let opts = ident_options(mf);
    let ctx = build_context(mf, opts).unwrap();
    let (theta, _, _) = ctx.point_vector().unwrap();
    for (a, b) in theta.iter().zip(theta_ref) {
        assert!((a - b).abs() < 1e-12 * b.abs().max(1.0), "point {a} vs {b}");
    }
    let mut pers = Persistent::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    ctx.analysis(&theta, false, true, &mut pers, &mut rng).unwrap().unwrap()
}

fn check_point(an: &Analysis, r: &Value, rtol: f64) {
    let rr = &r["ide_reducedform_point"];
    let rf = an.reducedform.as_ref().unwrap();
    assert_cols_close("dREDUCEDFORM", &rf.jacobian, &mat(&rr["dREDUCEDFORM"]), rtol);
    let m = &r["ide_moments_point"];
    let mo = an.moments.as_ref().unwrap();
    assert_cols_close("dMOMENTS", &mo.jacobian, &mat(&m["dMOMENTS"]), rtol);
    let want: Vec<f64> = vecf(&m["MOMENTS"]);
    assert_eq!(mo.values.len(), want.len());
    for (a, b) in mo.values.iter().zip(&want) {
        assert!((a - b).abs() <= 1e-10 * b.abs().max(1e-12), "MOMENTS {a} vs {b}");
    }
    let sp = an.spectrum.as_ref().unwrap();
    assert_cols_close("dSPECTRUM", &sp.jacobian, &mat(&r["ide_spectrum_point"]["dSPECTRUM"]), rtol);
    let mi = an.minimal.as_ref().unwrap();
    assert_cols_close("dMINIMAL", &mi.jacobian, &mat(&r["ide_minimal_point"]["dMINIMAL"]), rtol);
    let rd = &r["ide_dynamic_point"];
    assert_cols_close("dDYNAMIC", &an.dynamic.jacobian, &mat(&rd["dDYNAMIC"]), rtol);
    for (label, blk, rb) in [
        ("moments", mo, m),
        ("reducedform", rf, rr),
        ("spectrum", sp, &r["ide_spectrum_point"]),
        ("minimal", mi, &r["ide_minimal_point"]),
    ] {
        assert_eq!(blk.check.rank as f64, num(&rb["rank"]), "{label}: rank");
        assert_eq!(blk.ind.len(), vecf(&rb[format!("ind_d{}", label.to_uppercase()).as_str()]).len(), "{label}: ind");
    }
}

/// Order 2, `analytic_derivation_mode = -1`: the same finite differences as Dynare.
#[test]
fn order2_numerical_jacobians() {
    let r = load("ref_q2_ident_c10.json").expect("reference/setup unavailable");
    let mf = modfile("rbc_q2.mod", 10);
    let an = point(&mf, &vecf(&r["params_point"]));
    // Dynare differentiates the mean analytically (dr.derivs from finite differences): O(h^2)
    check_point(&an, &r, 1e-6);
}

/// Order 3, `analytic_derivation_mode = -1`.
#[test]
fn order3_numerical_jacobians() {
    let r = load("ref_q2_ident_c11.json").expect("reference/setup unavailable");
    let mf = modfile("rbc_q2.mod", 11);
    let an = point(&mf, &vecf(&r["params_point"]));
    check_point(&an, &r, 1e-6);
}

/// Order 2 with Dynare's default analytic derivatives: the finite differences agree to the
/// truncation error.
#[test]
fn order2_against_analytic_jacobians() {
    let r = load("ref_q2_ident_c12.json").expect("reference/setup unavailable");
    let mf = modfile("rbc_q2.mod", 12);
    let an = point(&mf, &vecf(&r["params_point"]));
    check_point(&an, &r, 1e-5);
}

/// The printout of the order-2 run is Dynare's.
#[test]
fn order2_printout() {
    let log = common::read_ref(root().join("reference/q2/q2_c10_matlab.log"));
    let dir = std::env::temp_dir().join(format!("dynare_q2_ident_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(root().join("reference/q2/rbc_q2.mod"), dir.join("rbc_q2.mod")).unwrap();
    let mf = modfile("rbc_q2.mod", 10);
    let o = mf
        .commands
        .iter()
        .find_map(|c| match c {
            Command::Other(n, o) if n == "identification" => Some(o.clone()),
            _ => None,
        })
        .unwrap();
    dynare::identification::run(&mf, &dir, "rbc_q2", &o, 1).unwrap();
    for line in [
        "Based on Pruned State Space System (order=2)",
        "    All parameters are identified in the Jacobian of first-order minimal system and second-order accurate mean (rank(Deltabar) is full with tol = robust).",
        "SPECTRUM (MUTSCHLER, 2015):",
        "    All parameters are identified in the Jacobian of first two moments (rank(Mbar) is full with tol = robust).",
    ] {
        assert!(log.contains(line), "{line}");
    }
}
