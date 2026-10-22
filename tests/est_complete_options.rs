//! Estimation options of the estimation-completion stream vs Dynare 7.2
//! (reference/est_complete/options/dump_options.m -> ref_options.json, est_opts.mod):
//! `dr = cycle_reduction | logarithmic_reduction` and `qz_zero_threshold` in the likelihood,
//! initial_estimation_checks.m's stochastic-singularity errors and
//! `no_init_estimation_check_first_obs`, `frequentist_smoother`, posterior_moments.m's kernel
//! density (`no_posterior_kernel_density`), `nonlinear_filter_initialization = 3`; and
//! dynare-rust-only runs for `posterior_nograph`/`posterior_graph`, `contemporaneous_correlation`
//! in the posterior moments, `use_univariate_smoother_if_singularity_is_detected`,
//! `huge_number`, `estimate_initial_states_endogenous_prior`'s errors.
mod common;

use std::path::{Path, PathBuf};

use dynare::linalg::Mat;
use dynare::modfile::{Command, OptValue, Options, opt};
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/est_complete/options")
}

fn reference() -> Option<Value> {
    Some(common::json_ref(dir().join("ref_options.json")))
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn mf_of(stage: u32) -> dynare::modfile::ModFile {
    dynare::modfile::parse_mod_file(&dir().join("est_opts.mod"), &[("STAGE".into(), stage.to_string())]).unwrap()
}

fn est_options(mf: &dynare::modfile::ModFile) -> Options {
    mf.commands.iter().find_map(|c| if let Command::Estimation(o, _) = c { Some(o.clone()) } else { None }).unwrap()
}

/// Runs est_opts.mod with `-DSTAGE=stage` in a scratch copy; returns the result and the folder.
fn run(stage: u32) -> (anyhow::Result<()>, PathBuf) {
    let tmp = std::env::temp_dir().join(format!("dynare_rs_est_complete_{}_{stage}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for f in ["est_opts.mod", "rbc_levels.csv", "rbc_firstobs.csv", "rbc_mode.mat"] {
        std::fs::copy(dir().join(f), tmp.join(f)).unwrap();
    }
    let ro = dynare::driver::RunOptions {
        defines: vec![("STAGE".into(), stage.to_string())],
        quiet: true,
        ..Default::default()
    };
    (dynare::driver::run(&tmp.join("est_opts.mod"), &ro), tmp)
}

/// dsge_likelihood at Dynare's points, with the estimation's solver options.
fn check_points(stage: u32, datafile: &str, r: &Value) {
    let mf = mf_of(stage);
    let o = est_options(&mf);
    let mut ss = dynare::analysis::StochSimulOptions::default();
    ss.apply(&o);
    let x0 = vecf(&r["xparam_init"]);
    let dsge = dynare::estimation::est_completion::with_solver_options(
        dynare::dsge::DsgeModel::new(&mf, Some(&x0)).unwrap(),
        &ss,
    )
    .unwrap();
    let table = dynare::data::Table::read(&dir().join(datafile), &Default::default()).unwrap();
    let mut sample = table.sample(&mf.varobs, 1, None).unwrap();
    sample.iter_mut().for_each(|v| *v = v.ln()); // loglinear
    let data = dynare::dsge::LikelihoodData::new(&sample, Default::default());
    let post = dynare::dsge::DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let f = dynare::estimation::Objective::eval(&post, &vecf(&p["x"])).0;
        let v = p["fval"].as_f64().unwrap();
        println!("stage {stage} point {k}: {f:.10} vs Dynare {v:.10}");
        assert!(((f - v) / v).abs() < 1e-10, "stage {stage} point {k}: {f} vs {v}");
    }
}

#[test]
fn dr_cycle_and_logarithmic_reduction_qz_zero_threshold() {
    let r = reference().expect("reference/setup unavailable");
    for stage in 1..=3 {
        let s = &r[format!("stage{stage}")];
        let mut ss = dynare::analysis::StochSimulOptions::default();
        ss.apply(&est_options(&mf_of(stage)));
        assert_eq!(ss.dr_cycle_reduction, s["dr_cycle_reduction"].as_bool().unwrap());
        assert_eq!(ss.dr_logarithmic_reduction, s["dr_logarithmic_reduction"].as_bool().unwrap());
        assert_eq!(ss.qz_zero_threshold, s["qz_zero_threshold"].as_f64().unwrap());
        check_points(stage, "rbc_levels.csv", s);
        let (res, tmp) = run(stage);
        res.unwrap();
        let _ = std::fs::remove_dir_all(tmp);
    }
}

#[test]
fn initial_estimation_checks_errors() {
    let r = reference().expect("reference/setup unavailable");
    for stage in [4, 5, 7] {
        let want = r[format!("stage{stage}")]["error"].as_str().unwrap().to_string();
        assert!(!want.is_empty());
        let (res, tmp) = run(stage);
        let got = format!("{:#}", res.unwrap_err());
        println!("stage {stage}: {got}");
        assert!(got.contains(&want), "stage {stage}: `{got}` vs Dynare `{want}`");
        let _ = std::fs::remove_dir_all(tmp);
    }
    // no_init_estimation_check_first_obs: the estimation runs, same likelihood as Dynare
    let s = &r["stage6"];
    assert!(s["error"].as_str().unwrap().is_empty());
    check_points(6, "rbc_firstobs.csv", s);
    let (res, tmp) = run(6);
    res.unwrap();
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn frequentist_smoother() {
    let r = reference().expect("reference/setup unavailable");
    let (res, tmp) = run(8);
    res.unwrap();
    let csv = dynare::data::Table::read_csv(&tmp.join("est_opts/Output/est_opts_smoothed_variables.csv")).unwrap();
    let mut worst: f64 = 0.0;
    for (j, name) in csv.names.iter().enumerate() {
        for (t, w) in vecf(&r["stage8"]["SmoothedVariables"][name]).iter().enumerate() {
            worst = worst.max((csv.values[(t, j)] - w).abs() / w.abs().max(1.0));
        }
    }
    println!("frequentist_smoother: smoothed variables {worst:.1e}");
    assert!(worst < 1e-9);
    let _ = std::fs::remove_dir_all(tmp);
    // frequentist_smoother = false: no classical smoother although `smoother` is set
    let (res, tmp) = run(11);
    res.unwrap();
    assert!(!tmp.join("est_opts/Output/est_opts_smoothed_variables.csv").exists());
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn posterior_kernel_density() {
    let r = reference().expect("reference/setup unavailable");
    let k = &r["kde"];
    let d = dynare::estimation::est_completion::posterior_density(&vecf(&k["xx"]));
    let want = vecf(&k["density"]); // 512 rows [abscissa, density]
    let fmax = (0..512).map(|i| want[2 * i + 1]).fold(0.0, f64::max);
    let (mut da, mut df) = (0.0f64, 0.0f64);
    for i in 0..512 {
        da = da.max((d[(i, 0)] - want[2 * i]).abs());
        df = df.max((d[(i, 1)] - want[2 * i + 1]).abs() / fmax);
    }
    println!("kernel density: abscissa {da:.1e}, density {df:.1e} (relative to its max)");
    assert!(da < 1e-12 && df < 1e-10);
    let c = dynare::estimation::est_completion::posterior_density(&[1.0; 50]);
    assert!(c.iter().all(|v| v.is_nan()) && vecf(&r["kde_constant"]).iter().all(|v| v.is_nan()));
}

#[test]
fn nonlinear_filter_initialization_3() {
    let r = reference().expect("reference/setup unavailable");
    let n = &r["nonlinear_init3"];
    let nl = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/nonlinear");
    let mf = dynare::modfile::parse_mod_file(&nl.join("rbcnl.mod"), &[("ORDER".into(), "2".into())]).unwrap();
    let x0 = vecf(&n["x0"]);
    let x1 = vecf(&n["x1"]);
    let table = dynare::data::Table::read(&nl.join("rbcnl_data.csv"), &Default::default()).unwrap();
    let y = table.sample(&mf.varobs, 1, None).unwrap().transpose();
    for c in n["cases"].as_array().unwrap() {
        let approx = c["approximation"].as_str().unwrap();
        let o: Options = vec![
            ("filter_algorithm".into(), OptValue::Str(c["algorithm"].as_str().unwrap().into())),
            ("proposal_approximation".into(), OptValue::Str(approx.into())),
            ("distribution_approximation".into(), OptValue::Str(approx.into())),
            ("nonlinear_filter_initialization".into(), OptValue::Num(3.0)),
        ];
        let p = dynare::nonlinear::ParticleOptions::from_options(&o).unwrap();
        assert_eq!(p.initialization, 3);
        assert_eq!(p.initial_state_prior_std, c["initial_state_prior_std"].as_f64().unwrap());
        let dsge = dynare::dsge::DsgeModel::new(&mf, Some(&x0)).unwrap();
        let nm = dynare::nonlinear::NonlinearModel::new(dsge, 2, p, y.clone(), 0, 0).unwrap();
        let b = dynare::estimation::Bounds::unbounded(x0.len());
        for (x, fref) in [&x0, &x1].into_iter().zip(vecf(&c["fval"])) {
            let mut d = dynare::nonlinear::rng::ReplayDraws::new(vec![], vec![]);
            let f = nm.evaluate_with(x, &b, &mut d).unwrap().fval;
            println!("initialization 3 {}: {f:.10} vs Dynare {fref:.10}", c["name"]);
            assert!((f - fref).abs() / fref.abs().max(1.0) < 1e-10);
        }
    }
    let bad: Options = vec![("nonlinear_filter_initialization".into(), OptValue::Num(3.0))];
    assert!(opt(&bad, "nonlinear_filter_initialization").is_some());
}

#[test]
fn posterior_graphs_moments_and_densities() {
    // STAGE 9: posterior_nograph, contemporaneous_correlation, kernel densities
    let (res, tmp) = run(9);
    res.unwrap();
    let out = tmp.join("est_opts/Output");
    let moments = std::fs::read_to_string(out.join("est_opts_posterior_moments.csv")).unwrap();
    let cc: Vec<&str> = moments.lines().filter(|l| l.starts_with("contemporaneous_correlation,")).collect();
    assert_eq!(cc.len(), 3); // (c,c), (c,y), (y,y)
    let diag: Vec<f64> = cc[0].split(',').skip(4).take(1).map(|v| v.parse().unwrap()).collect();
    assert!((diag[0] - 1.0).abs() < 1e-12, "corr(c,c) = {}", diag[0]);
    let dens = std::fs::read_to_string(out.join("est_opts_posterior_moments_density.csv")).unwrap();
    assert!(dens.lines().filter(|l| l.starts_with("contemporaneous_correlation,c,y,")).count() == 512);
    assert!(dens.lines().filter(|l| l.starts_with("covariance,c,y,")).count() == 512);
    assert!(out.join("est_opts_posterior_smoothed_variables_density.mat").exists());
    let graphs = tmp.join("est_opts/graphs");
    let figs: Vec<String> = std::fs::read_dir(&graphs)
        .map_or(vec![], |d| d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect());
    assert!(!figs.iter().any(|f| f.contains("SmoothedVariables") || f.contains("Bayesian_IRF")), "{figs:?}");
    let _ = std::fs::remove_dir_all(&tmp);
    // STAGE 10: no_posterior_kernel_density; posterior_graph re-enables the pm3 figures
    let (res, tmp) = run(10);
    res.unwrap();
    let out = tmp.join("est_opts/Output");
    assert!(out.join("est_opts_posterior_moments.csv").exists());
    assert!(!out.join("est_opts_posterior_moments_density.csv").exists());
    assert!(!out.join("est_opts_posterior_smoothed_variables_density.mat").exists());
    let figs: Vec<String> = std::fs::read_dir(tmp.join("est_opts/graphs"))
        .map_or(vec![], |d| d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect());
    assert!(figs.iter().any(|f| f.contains("SmoothedVariables")), "{figs:?}");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn estimate_initial_states_errors() {
    let (res, tmp) = run(12);
    assert!(
        format!("{:#}", res.unwrap_err())
            .contains("Init state estimation with endogenous prior is only compatible with slice sampler")
    );
    let _ = std::fs::remove_dir_all(tmp);
    // with the slice sampler the joint estimation runs (tests/est_complete_init_states.rs)
    let (res, tmp) = run(13);
    res.unwrap();
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn univariate_smoother_if_singularity() {
    use dynare::kalman::{KalmanOptions, MissingData, smoother};
    // two observables of the same state, no measurement error: F is singular (error 422)
    let t = Mat::from_row_slice(1, 1, &[0.5]);
    let rr = Mat::from_row_slice(1, 1, &[1.0]);
    let q = Mat::from_row_slice(1, 1, &[1.0]);
    let pstar = Mat::from_row_slice(1, 1, &[1.0 / 0.75]);
    let y = Mat::from_fn(2, 10, |_, j| (j as f64 * 0.3).sin());
    let missing = MissingData::from_data(&y);
    let mf = [0usize, 0];
    let base = KalmanOptions { lik_init: 1, ..Default::default() };
    let ok = smoother(&y, &missing, &mf, &t, &rr, &q, None, &pstar, None, &base, false).unwrap();
    assert!(ok.univariate && ok.info == 422);
    let mut strict = base.clone();
    let est: Options = vec![];
    dynare::estimation::est_completion::kalman_options(&est, &mut strict);
    assert!(strict.smoother_singularity_error);
    let e = smoother(&y, &missing, &mf, &t, &rr, &q, None, &pstar, None, &strict, false).unwrap_err();
    assert!(format!("{e}").contains("F is singular in stationary period"));
    let mut relaxed = base.clone();
    dynare::estimation::est_completion::kalman_options(
        &vec![("use_univariate_smoother_if_singularity_is_detected".into(), OptValue::Flag)],
        &mut relaxed,
    );
    assert!(!relaxed.smoother_singularity_error);
    dynare::estimation::est_completion::kalman_options(&vec![("noprint".into(), OptValue::Flag)], &mut relaxed);
    assert!(!relaxed.smoother_singularity_error);
}

#[test]
fn huge_number_guard() {
    use dynare::estimation::optim::minimize::{HUGE_NUMBER, HugeNumberGuard, huge_number};
    assert_eq!(huge_number(), HUGE_NUMBER);
    {
        let _g = HugeNumberGuard::set(Some(1e3));
        assert_eq!(huge_number(), 1e3);
        let b = dynare::estimation::Bounds::unbounded(2);
        let (lb, ub) = dynare::estimation::optim::minimize::finite_bounds(&b, huge_number());
        assert_eq!((lb[0], ub[1]), (-1e3, 1e3));
    }
    assert_eq!(huge_number(), HUGE_NUMBER);
    let o: Options = vec![("posterior_nograph".into(), OptValue::Flag), ("posterior_graph".into(), OptValue::Flag)];
    assert!(!dynare::estimation::est_completion::no_graph_posterior(&o));
    assert!(dynare::estimation::est_completion::no_graph_posterior(&o[..1].to_vec()));
}
