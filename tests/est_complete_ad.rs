//! Estimation completion: `analytic_derivation` against Dynare 7.2 (reference/est_complete/ad).
//! `ref_points_<stage>.json`: dsge_likelihood's fval, DLIK and Hess ('', 'opg', 'full',
//! 'asymptotic') at three parameter vectors of est_ad.mod (stages: multivariate filter,
//! univariate filter, correlated measurement errors / expanded state, ML, lik_init = 2).

use std::path::PathBuf;

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::Objective;
use dynare::estimation::analytic::{AnalyticContext, AnalyticHessian, AnalyticOptions, AnalyticPosterior};
use dynare::kalman::KalmanOptions;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use serde_json::Value;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/est_complete/ad")
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

/// A MATLAB matrix from jsonencode (array of rows).
fn matf(v: &Value) -> Mat {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(vecf).collect();
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

struct Setup {
    dsge: DsgeModel,
    data: LikelihoodData,
}

fn setup(stage: u32) -> Setup {
    let mf =
        parse_mod_file(&dir().join("est_ad.mod"), &[("STAGE".into(), stage.to_string()), ("AD".into(), "1".into())])
            .unwrap();
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let table = Table::read_csv(&dir().join("ad_data.csv")).unwrap();
    let kopts = KalmanOptions {
        kalman_algo: if stage == 2 || stage == 3 { 2 } else { 0 },
        lik_init: if stage == 5 { 2 } else { 1 },
        ..Default::default()
    };
    let data = LikelihoodData::new(&table.sample(&mf.varobs, 1, Some(200)).unwrap(), kopts);
    Setup { dsge, data }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

fn check_stage(stage: u32) {
    let s = setup(stage);
    let post = DsgePosterior { dsge: &s.dsge, data: &s.data, bounds: &s.dsge.bounds };
    let ctx = AnalyticContext::new(&post, AnalyticOptions { mode: 0 }).unwrap();
    let ap = AnalyticPosterior { post: &post, ctx: &ctx };
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(dir().join(format!("ref_points_{stage}.json"))).unwrap())
            .unwrap();
    let pts = r["points"].as_array().unwrap();
    for (ip, p) in pts.iter().enumerate() {
        let x = vecf(&p["x"]);
        let f = post.eval(&x).0;
        let fr = p["fval_plain"].as_f64().unwrap();
        assert!(rel(f, fr) < 1e-9, "stage {stage} point {ip}: fval {f} vs {fr}");
        for (kind, key) in [
            (AnalyticHessian::None, "none"),
            (AnalyticHessian::Opg, "opg"),
            (AnalyticHessian::Full, "full"),
            (AnalyticHessian::Asymptotic, "asymptotic"),
        ] {
            let g = ap.eval_grad(&x, kind).unwrap();
            assert_eq!(g.info, 0, "stage {stage} point {ip} {key}: info");
            let gr = vecf(&p[key]["DLIK"]);
            let gmax = gr.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let err = g.grad.iter().zip(&gr).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / gmax.max(1.0);
            eprintln!("stage {stage} point {ip} {key}: DLIK rel err {err:.2e}");
            assert!(err < 1e-8, "stage {stage} point {ip} {key}: DLIK {:?} vs {:?}", g.grad, gr);
            if kind != AnalyticHessian::None {
                let hr = matf(&p[key]["Hess"]);
                let h = g.hess.expect("Hess");
                let hmax = hr.amax();
                let herr = (&h - &hr).amax() / hmax.max(1.0);
                eprintln!("stage {stage} point {ip} {key}: Hess rel err {herr:.2e}");
                assert!(herr < 1e-6, "stage {stage} point {ip} {key}: Hess\n{h}\nvs\n{hr}");
            }
        }
    }
}

#[test]
fn ad_multivariate() {
    check_stage(1);
}

#[test]
fn ad_univariate() {
    check_stage(2);
}

#[test]
fn ad_expanded_state() {
    check_stage(3);
}

#[test]
fn ad_ml() {
    check_stage(4);
}

#[test]
fn ad_lik_init_2() {
    check_stage(5);
}

/// Runs est_ad.mod (mh_replic = 0) in a scratch copy with the macro `defines` and returns the
/// mode file (xparam1, fval, hh if saved) and the results JSON.
fn run(tag: &str, defines: &[(&str, &str)]) -> (Vec<f64>, f64, Option<Mat>, Value) {
    let tmp = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("target/tmp/est_ad_{tag}"));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["est_ad.mod", "ad_model.inc", "ad_data.csv"] {
        std::fs::copy(dir().join(f), tmp.join(f)).unwrap();
    }
    let ro = dynare::driver::RunOptions {
        quiet: true,
        mh_replic: Some(0),
        defines: defines.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
        ..Default::default()
    };
    dynare::driver::run(&tmp.join("est_ad.mod"), &ro).unwrap();
    let m = dynare::matfile::read(&tmp.join("est_ad/Output/est_ad_mode.mat")).unwrap();
    let res: Value =
        serde_json::from_str(&std::fs::read_to_string(tmp.join("est_ad/Output/est_ad_results.json")).unwrap()).unwrap();
    (m["xparam1"].as_vec().unwrap(), m["fval"].as_vec().unwrap()[0], m.get("hh").and_then(|h| h.as_matrix()), res)
}

/// `tol_x`/`tol_h` <= 1e-6: the optimizer follows Dynare's path (csminwel, solveopt);
/// larger: the path drifts from Dynare's through rounding noise amplified by the numerical
/// univariate steps (newrat's mr_gstep) or by a flat likelihood, and the Hessian is checked
/// at Dynare's mode separately ([`hessian_at_dynare_mode`]).
fn check_run(key: &str, defines: &[(&str, &str)], tol_x: f64, tol_h: f64) {
    check_run_with(key, defines, ModeTol::Rel(tol_x), tol_h)
}

/// How a run's mode is compared with Dynare's.
enum ModeTol {
    /// Max elementwise relative difference (relative to max(|x_dynare|, 1e-2)); fval to 1e-7
    /// relative if the tolerance is <= 1e-6, else 1e-6.
    Rel(f64),
    /// For a mode that is not identified along some direction (flat likelihood): fval to 1e-8
    /// relative, and the mode in the norm of Dynare's Hessian at its mode (hh, of minus the log
    /// posterior), `sqrt(dx' hh dx) < tol`. This is the distance in posterior standard
    /// deviations: `|v'dx| <= tol * sqrt(v' hh^-1 v)` for every direction v, so it is strict
    /// along the identified directions and lets the mode slide along the flat ones, where
    /// rounding-level differences move the optimizer's stopping point a long way. To second
    /// order `dx' hh dx = 2 (f(x) - f(x_dynare))`, the scale of the fval check.
    HessianNorm(f64),
}

fn check_run_with(key: &str, defines: &[(&str, &str)], mode_tol: ModeTol, tol_h: f64) {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("ref_runs.json")).unwrap()).unwrap();
    let want = &r[key];
    let (x, f, hh, res) = run(key, defines);
    let xr = vecf(&want["xparam1"]);
    let fr = want["fval"].as_f64().unwrap();
    let dx = x.iter().zip(&xr).map(|(a, b)| (a - b).abs() / b.abs().max(1e-2)).fold(0.0, f64::max);
    eprintln!("{key}: fval {f:.12} vs {fr:.12}, max rel dx {dx:.2e}");
    match mode_tol {
        ModeTol::Rel(tol_x) => {
            let tol_f = if tol_x <= 1e-6 { 1e-7 } else { 1e-6 };
            assert!((f - fr).abs() < tol_f * fr.abs().max(1.0), "{key}: fval {f} vs {fr}");
            assert!(dx < tol_x, "{key}: mode {x:?} vs {xr:?}");
        }
        ModeTol::HessianNorm(tol) => {
            assert!((f - fr).abs() < 1e-8 * fr.abs().max(1.0), "{key}: fval {f} vs {fr}");
            let hr = matf(&want["hh"]);
            let d = Mat::from_fn(x.len(), 1, |i, _| x[i] - xr[i]);
            let dn = (d.transpose() * &hr * &d)[(0, 0)].max(0.0).sqrt();
            let eig = hr.clone().symmetric_eigenvalues();
            eprintln!("{key}: mode distance in hh norm {dn:.2e} (hh eigenvalues {:.2e}..{:.2e})", eig.min(), eig.max());
            assert!(dn < tol, "{key}: mode {x:?} vs {xr:?}, hh-norm distance {dn}");
        }
    }
    match (&hh, want["hh"].as_array().map(|a| !a.is_empty()).unwrap_or(false)) {
        (Some(h), true) => {
            let hr = matf(&want["hh"]);
            let dh = (h - &hr).amax() / hr.amax();
            eprintln!("{key}: hh rel err {dh:.2e}");
            assert!(dh < tol_h, "{key}: hh\n{h}\nvs\n{hr}");
        }
        (None, false) => {}
        (h, w) => panic!("{key}: hh saved {} vs Dynare {w}", h.is_some()),
    }
    if let Some(l) = want["laplace"].as_f64() {
        let lr = res["log_data_density_laplace"].as_f64().unwrap();
        eprintln!("{key}: Laplace {lr:.10} vs {l:.10}");
        assert!((lr - l).abs() < 1e-5_f64.max(tol_h * 10.0), "{key}: Laplace {lr} vs {l}");
    }
}

/// The analytic ('full') Hessian at Dynare's mode of run `key` (est_ad stage `stage`).
fn hessian_at_dynare_mode(key: &str, stage: u32) {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("ref_runs.json")).unwrap()).unwrap();
    let want = &r[key];
    let s = setup(stage);
    let post = DsgePosterior { dsge: &s.dsge, data: &s.data, bounds: &s.dsge.bounds };
    let ctx = AnalyticContext::new(&post, AnalyticOptions { mode: 0 }).unwrap();
    let ap = AnalyticPosterior { post: &post, ctx: &ctx };
    let x = vecf(&want["xparam1"]);
    let g = ap.eval_grad(&x, AnalyticHessian::Full).unwrap();
    let hr = matf(&want["hh"]);
    let dh = (g.hess.as_ref().unwrap() - &hr).amax() / hr.amax();
    eprintln!("{key}: full Hessian at Dynare's mode rel err {dh:.2e}");
    assert!(dh < 1e-6, "{key}: hh {} vs {hr}", g.hess.unwrap());
    assert!((g.fval - want["fval"].as_f64().unwrap()).abs() < 1e-9 * g.fval.abs());
}

#[test]
fn ad_hessian_at_dynare_modes() {
    hessian_at_dynare_mode("ad_mc5", 1);
    hessian_at_dynare_mode("ad_exp_mc5", 3);
    hessian_at_dynare_mode("ad_ml_mc4", 4);
}

/// mr_hessian at the mode (`optim = ('Hessian', 0 | 2)` of mode_compute 5) at Dynare's mode,
/// from a fresh hess_info (ref_hess.json).
#[test]
fn newrat_hessian_options_at_dynare_mode() {
    use dynare::estimation::mode_hessian::ModeHessian;
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("ref_hess.json")).unwrap()).unwrap();
    let s = setup(1);
    let post = DsgePosterior { dsge: &s.dsge, data: &s.data, bounds: &s.dsge.bounds };
    let prior_sd = s.dsge.prior_sd();
    for (key, flag) in [("mc5_hess0", 0), ("mc5_hess2", 2)] {
        let w = &r[key];
        let x = vecf(&w["x"]);
        let fval = w["fval"].as_f64().unwrap();
        let mh = ModeHessian {
            obj: &post,
            dsge_likelihood: Some(&post),
            analytic: None,
            cova_compute: true,
            penalized: false,
            newratflag: flag,
            bounds: &s.dsge.bounds,
            prior_sd: &prior_sd,
        };
        let h = mh.compute(5, &x, fval, None).unwrap().unwrap();
        let hr = matf(&w["hh"]);
        let dh = (&h - &hr).amax() / hr.amax();
        eprintln!("{key}: mr_hessian at Dynare's mode rel err {dh:.2e}");
        assert!(dh < 1e-5, "{key}: {h} vs {hr}");
    }
}

#[test]
fn ad_run_csminwel() {
    check_run("ad_mc4", &[("STAGE", "1"), ("MC", "4"), ("AD", "1")], 1e-6, 1e-6);
}

#[test]
fn ad_run_newrat() {
    check_run("ad_mc5", &[("STAGE", "1"), ("MC", "5"), ("AD", "1")], 5e-3, 5e-2);
}

#[test]
fn ad_run_solveopt() {
    check_run("ad_mc101", &[("STAGE", "1"), ("MC", "101"), ("AD", "1")], 1e-6, 1e-6);
}

#[test]
fn ad_run_univariate() {
    check_run("ad_uni_mc4", &[("STAGE", "2"), ("MC", "4"), ("AD", "1")], 1e-6, 1e-6);
}

/// Correlated measurement errors (stage 3): the posterior is nearly flat along one direction
/// (hh's eigenvalues 6..3e7; newrat's own Hessian reaches a condition number ~1e9), so where
/// newrat stops along it is set by rounding noise (ours: 21 iterations, 5.6e-3 relative from
/// 7.2's mode but 5e-4 posterior standard deviations in hh's norm). Dynare 7.1
/// took 23 iterations and 7.2 takes 19 only because 7.2's univariate filter symmetrizes P,
/// (P+P')/2 (ported; 7.2 run with 7.1's univariate_kalman_filter.m reproduces 7.1's run to the
/// last digit): the modes differ by 4.5e-3 relative, the fvals by 4e-10 relative. Hence fval and
/// the identified part of the mode are compared.
#[test]
fn ad_run_expanded_newrat() {
    check_run_with("ad_exp_mc5", &[("STAGE", "3"), ("MC", "5"), ("AD", "1")], ModeTol::HessianNorm(5e-3), 5e-2);
}

#[test]
fn ad_run_ml() {
    check_run("ad_ml_mc4", &[("STAGE", "4"), ("MC", "4"), ("AD", "1")], 5e-3, 5e-2);
}

#[test]
fn newrat_hessian_opg() {
    check_run("mc5_hess0", &[("STAGE", "1"), ("MC", "5"), ("HESS", "0")], 5e-3, 5e-2);
}

#[test]
fn newrat_hessian_mixed() {
    check_run("mc5_hess2", &[("STAGE", "1"), ("MC", "5"), ("HESS", "2")], 5e-3, 5e-2);
}

#[test]
fn penalized_hessian() {
    // Elsewhere csminwel stops 3e-6 away along the flat directions (fval within 1e-11, 1e-4
    // posterior s.d.), so the mode is measured in posterior standard deviations there (see
    // ModeTol::HessianNorm), and the Hessian at that point differs by 2.3e-6.
    let (mode_tol, tol_h) =
        if cfg!(target_os = "macos") { (ModeTol::Rel(1e-6), 1e-6) } else { (ModeTol::HessianNorm(1e-3), 1e-5) };
    check_run_with("mc4_pen", &[("STAGE", "1"), ("MC", "4"), ("PEN", "1")], mode_tol, tol_h);
}

/// analytic_derivation_mode 1 (Kronecker; here the Sylvester solution), -1 (finite differences
/// of the solution) and -2 (finite differences of the steady state and Jacobian), stage 6.
#[test]
fn ad_derivation_modes() {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("ref_modes.json")).unwrap()).unwrap();
    let s = setup(6);
    let post = DsgePosterior { dsge: &s.dsge, data: &s.data, bounds: &s.dsge.bounds };
    let x = vecf(&r["x"]);
    for (mode, key, tol_g, tol_h) in
        [(1, "mode_1", 1e-8, 1e-8), (-1, "mode_m1", 1e-8, 1e-8), (-2, "mode_m2", 1e-8, 1e-8)]
    {
        let ctx = AnalyticContext::new(&post, AnalyticOptions { mode }).unwrap();
        let ap = AnalyticPosterior { post: &post, ctx: &ctx };
        let g = ap.eval_grad(&x, AnalyticHessian::Full).unwrap();
        let gr = vecf(&r[key]["DLIK"]);
        let gmax = gr.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let err = g.grad.iter().zip(&gr).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / gmax;
        let hr = matf(&r[key]["Hess"]);
        let herr = (g.hess.as_ref().unwrap() - &hr).amax() / hr.amax();
        eprintln!("{key}: DLIK rel err {err:.2e}, Hess rel err {herr:.2e}");
        assert!(err < tol_g && herr < tol_h, "{key}: {:?} vs {gr:?}", g.grad);
    }
}

/// Dynare's errors for options incompatible with analytic_derivation.
#[test]
fn ad_option_errors() {
    use dynare::estimation::analytic::check_options;
    use dynare::modfile::OptValue;
    let base = || vec![("analytic_derivation".to_string(), OptValue::Flag)];
    let err = |o: Vec<(String, OptValue)>, ka: u8, li: u8, miss: bool, order: i32| {
        check_options(&o, ka, li, miss, order).unwrap_err().to_string()
    };
    assert!(check_options(&base(), 0, 1, false, 1).unwrap().is_some());
    assert!(check_options(&Vec::new(), 0, 3, true, 2).unwrap().is_none());
    assert_eq!(err(base(), 0, 3, false, 1), "analytic derivation is incompatible with diffuse filter");
    assert_eq!(err(base(), 5, 1, false, 1), "analytic derivation is incompatible with pruned skewed Kalman filter");
    for (k, msg) in [
        (
            "fast_kalman_filter",
            "estimation option conflict: analytic_derivation isn't available for fast_kalman_filter",
        ),
        (
            "heteroskedastic_filter",
            "estimation option conflict: analytic_derivation isn't available for heteroskedastic_filter",
        ),
        ("loglinear", "The analytic_derivation and loglinear options are not compatible"),
        ("endogenous_prior", "The analytic_derivation and endogenous_prior options are not compatible"),
        ("conditional_likelihood", "The analytic_derivation and conditional_likelihood are not compatible!"),
        (
            "estimate_initial_states_endogenous_prior",
            "estimation option conflict: estimate_initial_states_endogenous_prior isn't available for analytic_derivation",
        ),
    ] {
        let mut o = base();
        o.push((k.to_string(), OptValue::Flag));
        assert_eq!(err(o, 0, 1, false, 1), msg);
    }
    assert!(err(base(), 0, 1, true, 1).contains("missing observations"));
    assert!(check_options(&base(), 2, 1, true, 1).is_ok());
    assert!(err(base(), 0, 1, false, 2).contains("order=1"));
    // mode -1 with estimated measurement errors fails in Dynare 7.1 (reshape error)
    let s = setup(1);
    let post = DsgePosterior { dsge: &s.dsge, data: &s.data, bounds: &s.dsge.bounds };
    assert!(AnalyticContext::new(&post, AnalyticOptions { mode: -1 }).is_err());
}

/// cova_compute = 0: no Hessian (mode file without hh, no Laplace approximation) and Dynare's
/// error when the MCMC needs it (check_posterior_sampler_options.m).
#[test]
fn cova_compute_zero() {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("ref_runs.json")).unwrap()).unwrap();
    assert!(r["mc4_cova0"]["error"].as_str().unwrap().contains("MCMC cannot start"));
    let (x, f, hh, res) = run("cova0", &[("STAGE", "1"), ("MC", "4"), ("COVA", "0")]);
    assert!(hh.is_none());
    assert!(
        res["log_data_density_laplace"].is_null()
            || res["log_data_density_laplace"].as_f64().is_none_or(|v| v.is_nan())
    );
    let want = &r["mc4"];
    assert!((f - want["fval"].as_f64().unwrap()).abs() < 1e-7 * f.abs());
    assert_eq!(x.len(), vecf(&want["xparam1"]).len());
    // with the default mh_replic, the MCMC cannot start
    let tmp = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp/est_ad_cova0_mcmc");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["est_ad.mod", "ad_model.inc", "ad_data.csv"] {
        std::fs::copy(dir().join(f), tmp.join(f)).unwrap();
    }
    let ro = dynare::driver::RunOptions {
        quiet: true,
        defines: [("STAGE", "1"), ("MC", "4"), ("COVA", "0")]
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect(),
        ..Default::default()
    };
    let e = dynare::driver::run(&tmp.join("est_ad.mod"), &ro).unwrap_err();
    assert!(e.to_string().contains("check_posterior_sampler_options:: MCMC cannot start"), "{e}");
}
