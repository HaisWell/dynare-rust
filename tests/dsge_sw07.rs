//! End-to-end structural check: reference/sw07/sw07.mod parsed by our preprocessor ->
//! steady state -> decision rule -> restricted T, R, against Dynare (ref.json).
mod common;

use std::path::Path;

use dynare::dsge::DsgeModel;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use serde_json::Value;

fn load(name: &str) -> Option<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07").join(name);
    Some(common::json_ref(p))
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

/// MATLAB matrices arrive as arrays of rows.
fn mat(v: &Value) -> Mat {
    let rows = v.as_array().unwrap();
    let r = rows.len();
    let c = rows[0].as_array().map_or(1, |x| x.len());
    Mat::from_fn(r, c, |i, j| rows[i].as_array().map_or_else(|| vecf(&rows[i])[0], |x| vecf(&x[j])[0]))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn sw07_state_space_at_mode() {
    let r = load("ref.json").expect("reference/setup unavailable");
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07/sw07.mod");
    let mf = parse_mod_file(&path, &[("STAGE".into(), "9".into())]).unwrap();
    let xm = vecf(&r["xparam_mode"]);
    let m = DsgeModel::new(&mf, Some(&xm)).unwrap();
    assert_eq!(m.estimated_names.len(), 40);

    if let Some(s) = load("ref_solver.json") {
        let d = max_abs_diff(&m.ys_calib, &vecf(&s["ys_calib"]));
        assert!(d < 1e-10, "ys_calib differs by {d}");
    }
    let d = max_abs_diff(&m.ys_guess, &vecf(&r["steady_state"]));
    assert!(d < 1e-10, "steady state at the initial point differs by {d}");

    // priors and bounds (bayestopt_, BoundsInfo)
    // jsonencode writes +-Inf as null
    let lb: Vec<f64> =
        vecf(&r["bounds"]["lb"]).into_iter().map(|v| if v.is_nan() { f64::NEG_INFINITY } else { v }).collect();
    let ub: Vec<f64> =
        vecf(&r["bounds"]["ub"]).into_iter().map(|v| if v.is_nan() { f64::INFINITY } else { v }).collect();
    for k in 0..m.estimated_names.len() {
        let same = |a: f64, b: f64| a == b || (a - b).abs() <= 1e-12 * a.abs().max(1.0) || (a.is_nan() && b.is_nan());
        assert!(
            same(m.bounds.lb[k], lb[k]) && same(m.bounds.ub[k], ub[k]),
            "bounds {}: {} {} vs {lb_k} {ub_k}",
            m.estimated_names[k],
            m.bounds.lb[k],
            m.bounds.ub[k],
            lb_k = lb[k],
            ub_k = ub[k]
        );
    }
    for p in r["points"].as_array().unwrap() {
        let x = vecf(&p["x"]);
        let lp = m.log_prior(&x);
        match p["log_prior"].as_f64() {
            Some(v) => assert!((lp - v).abs() < 1e-10, "log prior {lp} vs {v}"),
            None => assert!(!lp.is_finite()),
        }
    }

    let res = m.resolve(&xm).unwrap();
    let t = mat(&r["mode_ss"]["T"]);
    let rr = mat(&r["mode_ss"]["R"]);
    let dt = (&res.t - &t).amax();
    let dr = (&res.r - &rr).amax();
    let dq = (&res.q - mat(&r["mode_ss"]["Q"])).amax();
    let dp = max_abs_diff(&res.params, &vecf(&r["mode_ss"]["params"]));
    println!("T {dt:e} R {dr:e} Q {dq:e} params {dp:e}");
    assert!(dt < 1e-10 && dr < 1e-10 && dq < 1e-12 && dp < 1e-14);
}

/// The full posterior kernel (dsge_likelihood fval) against Dynare at every reference point.
#[test]
fn sw07_posterior_kernel() {
    use dynare::data::Table;
    use dynare::dsge::{DsgePosterior, LikelihoodData};
    use dynare::estimation::Objective;
    use dynare::kalman::KalmanOptions;
    let r = load("ref.json").expect("reference/setup unavailable");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let xm = vecf(&r["xparam_mode"]);
    let m = DsgeModel::new(&mf, Some(&xm)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let sample = table.sample(&mf.varobs, 5, None).unwrap();
    let kopts = KalmanOptions { presample: 8, lik_init: 3, ..Default::default() };
    let data = LikelihoodData::new(&sample, kopts);
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &m.bounds };
    for (k, p) in r["points"].as_array().unwrap().iter().enumerate() {
        let x = vecf(&p["x"]);
        let (f, info) = post.eval(&x);
        match p["minus_loglik"].as_f64() {
            Some(v) => {
                println!("point {k}: {f:.10} vs Dynare {v:.10}  (rel {:.1e})", ((f - v) / v).abs());
                assert!(((f - v) / v).abs() < 1e-9, "point {k}");
            }
            None => {
                println!("point {k}: info {info} (Dynare info {})", p["info"]);
                assert!(!f.is_finite());
            }
        }
    }
    let t = std::time::Instant::now();
    let n = 200;
    for _ in 0..n {
        std::hint::black_box(post.eval(&xm));
    }
    println!("{:.3} ms per posterior evaluation (MATLAB: 1.6 ms)", t.elapsed().as_secs_f64() * 1e3 / n as f64);
}

/// Wide check of the posterior kernel, bounds switched off: Dynare's fval (ref_points_wide.json)
/// at the 40 points of points_wide.json (8 on a line along stderr ea, then draws from
/// N(mode, 4*inv(hh)) with a finite prior; both written by dump_points_wide.m).
#[test]
fn sw07_posterior_kernel_wide() {
    use dynare::data::Table;
    use dynare::dsge::{DsgePosterior, LikelihoodData};
    use dynare::estimation::Objective;
    use dynare::kalman::KalmanOptions;
    let r = load("ref.json").expect("reference/setup unavailable");
    let pts = load("points_wide.json").expect("reference/setup unavailable");
    let d = load("ref_points_wide.json").expect("reference/setup unavailable");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let xm = vecf(&r["xparam_mode"]);
    let m = DsgeModel::new(&mf, Some(&xm)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let data = LikelihoodData::new(
        &table.sample(&mf.varobs, 5, None).unwrap(),
        KalmanOptions { presample: 8, lik_init: 3, ..Default::default() },
    );
    let unb = dynare::estimation::Bounds::unbounded(xm.len());
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &unb };
    let pts = pts["x"].as_array().unwrap();
    assert_eq!(pts.len(), 40);
    // the first 8 points move stderr ea only
    let k_ea = m.estimated_names.iter().position(|n| n == "stderr ea").unwrap();
    for p in &pts[..8] {
        let x = vecf(p);
        assert!((0..xm.len()).all(|i| i == k_ea || x[i] == xm[i]));
    }
    let (mut worst, mut failed) = (0.0f64, 0);
    for (k, p) in pts.iter().enumerate() {
        let x = vecf(p);
        let (fv, info) = post.eval(&x);
        let dinfo = d["info"][k].as_i64().unwrap();
        match d["fval"][k].as_f64().filter(|_| dinfo == 0) {
            Some(dv) => {
                let rel = ((fv - dv) / dv).abs();
                println!("{k:2}: rust {fv:14.6}  dynare {dv:14.6}  rel {rel:.1e}");
                worst = worst.max(rel);
            }
            None => {
                println!("{k:2}: rust info {info}, Dynare info {dinfo}");
                assert!(!fv.is_finite(), "point {k}: {fv}, Dynare info {dinfo}");
                failed += 1;
            }
        }
    }
    println!("worst relative difference {worst:.2e}, {failed} point(s) rejected by both");
    assert!(worst < 1e-9, "{worst}");
}

/// Regression for points_wide.json point 15 (0-based), Dynare internals from dump_point15.m.
///
/// T has a double unit root there (the random walks `pistar` and `labtrend`).
/// compute_Pinf_Pstar.m drops a unit root from Pinf when `norm(QTinf(mf,:)*ST(...,k)) < 1e-8`,
/// a test that depends on the Schur basis of the unit-root block, i.e. on the LAPACK build
/// (MATLAB: NAG; here: Accelerate). Given MATLAB's Schur factors we reproduce Dynare's
/// Pinf/Pstar and fval, and with our own factors the diffuse directions are the same.
#[test]
fn sw07_point15_unit_root_basis() {
    use dynare::kalman::{self, KalmanOptions, MissingData};
    use dynare::linalg::schur::RealSchur;
    let r = load("ref.json").expect("reference/setup unavailable");
    let p = load("ref_point15.json").expect("reference/setup unavailable");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let m = DsgeModel::new(&mf, Some(&vecf(&r["xparam_mode"]))).unwrap();
    let x = vecf(&p["x"]);
    let res = m.resolve(&x).unwrap();
    assert!(max_abs_diff(&res.steady_state, &vecf(&p["steady_state"])) < 1e-9);
    assert!((&res.t - mat(&p["T"])).amax() < 1e-10 && (&res.r - mat(&p["R"])).amax() < 1e-10);
    let idx = &m.ss_index;
    let opts = KalmanOptions { presample: 8, lik_init: 3, ..Default::default() };
    let y = mat(&p["Y"]);
    let md = MissingData::from_data(&y);
    let want_lik = p["fval"].as_f64().unwrap() - vecf(&p["DLIK"])[0]; // fval + lnprior
    let lik = |pstar: &Mat, pinf: &Mat| {
        kalman::likelihood(&y, &md, &idx.mf, &res.t, &res.r, &res.q, None, pstar, pinf, &opts).minus_loglik
    };
    // with MATLAB's Schur factors: Dynare's Pinf/Pstar and likelihood
    let sch =
        RealSchur { t: mat(&p["schur_T"]), u: mat(&p["schur_U"]), wr: vecf(&p["schur_wr"]), wi: vecf(&p["schur_wi"]) };
    let (pstar, pinf) = kalman::compute_pinf_pstar_with_schur(
        &idx.mf,
        &res.t,
        &res.r,
        &res.q,
        1.000001,
        &idx.restrict_columns,
        Some(sch),
    )
    .unwrap();
    let (dps, dpi) = (mat(&p["Pstar"]), mat(&p["Pinf"]));
    assert!((&pstar - &dps).amax() < 1e-9 * dps.amax() && (&pinf - &dpi).amax() < 1e-9);
    let l = lik(&pstar, &pinf);
    assert!((l - want_lik).abs() < 1e-10 * want_lik, "{l} vs {want_lik}");
    // with our own Schur factors: the same diffuse directions, the same likelihood
    let (pstar2, pinf2) = kalman::compute_pinf_pstar(&idx.mf, &res.t, &res.r, &res.q, 1.000001, &idx.restrict_columns);
    assert!((&pstar2 - &dps).amax() < 1e-9 * dps.amax() && (&pinf2 - &dpi).amax() < 1e-9);
    let tr = |a: &Mat| a.trace();
    println!("trace Pinf: ours {} Dynare {}", tr(&pinf2), tr(&dpi));
    let l2 = lik(&pstar2, &pinf2);
    println!(
        "minus loglik with our Schur {l2:.10}, Dynare {want_lik:.10}, rel {:.1e}",
        ((l2 - want_lik) / want_lik).abs()
    );
    assert!(((l2 - want_lik) / want_lik).abs() < 1e-12);
}

/// Cost of each step of one posterior evaluation at the mode (printed with --nocapture).
/// "full QZ" is Dynare's dense path (`FirstOrderSolver::solve_full_qz`), i.e. the
/// solver as it was before the structured QZ.
#[test]
fn sw07_posterior_timing() {
    use dynare::data::Table;
    use dynare::dsge::{DsgePosterior, LikelihoodData};
    use dynare::estimation::Objective;
    use dynare::kalman::{KalmanOptions, KalmanWorkspace, try_compute_pinf_pstar};
    use dynare::solver::{FirstOrderSolver, SolverOptions, kalman_transition_matrix};
    let r = load("ref.json").expect("reference/setup unavailable");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let xm = vecf(&r["xparam_mode"]);
    let m = DsgeModel::new(&mf, Some(&xm)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let sample = table.sample(&mf.varobs, 5, None).unwrap();
    let kopts = KalmanOptions { presample: 8, lik_init: 3, ..Default::default() };
    let data = LikelihoodData::new(&sample, kopts.clone());
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &m.bounds };
    let solver = FirstOrderSolver::new(&m.model.lead_lag_incidence, m.model.n_exo(), SolverOptions::default());
    let idx = &m.ss_index;

    let reps = 300;
    let time = |f: &mut dyn FnMut()| {
        f();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        t.elapsed().as_secs_f64() * 1e6 / reps as f64
    };
    let (params, q) = m.set_all_parameters(&xm);
    let ys = m.steady_state(&params).unwrap();
    let x_ss = vec![0.0; m.model.n_exo()];
    let g1 = m.model.dynamic_jacobian(&ys, &x_ss, &params);
    let dr = solver.solve(&g1, &ys).unwrap();
    let (t, rr) = kalman_transition_matrix(&dr, idx);
    let (pstar, pinf) =
        try_compute_pinf_pstar(&idx.mf, &t, &rr, &q, kopts.qz_criterium, &idx.restrict_columns).unwrap();
    let mut y = data.y.clone();
    for (i, &v) in idx.mfys.iter().enumerate() {
        for j in 0..y.ncols() {
            y[(i, j)] -= ys[v];
        }
    }
    let mut ws = KalmanWorkspace::new(t.nrows(), y.nrows(), q.nrows());

    let rows = [
        ("set_all_parameters", time(&mut || drop(std::hint::black_box(m.set_all_parameters(&xm))))),
        ("steady state", time(&mut || drop(std::hint::black_box(m.steady_state(&params))))),
        ("dynamic Jacobian", time(&mut || drop(std::hint::black_box(m.model.dynamic_jacobian(&ys, &x_ss, &params))))),
        (
            "first-order solve (full QZ, before)",
            time(&mut || drop(std::hint::black_box(solver.solve_full_qz(&g1, &ys)))),
        ),
        ("first-order solve (structured)", time(&mut || drop(std::hint::black_box(solver.solve(&g1, &ys))))),
        ("kalman_transition_matrix", time(&mut || drop(std::hint::black_box(kalman_transition_matrix(&dr, idx))))),
        (
            "compute_Pinf_Pstar",
            time(&mut || {
                drop(std::hint::black_box(try_compute_pinf_pstar(
                    &idx.mf,
                    &t,
                    &rr,
                    &q,
                    kopts.qz_criterium,
                    &idx.restrict_columns,
                )))
            }),
        ),
        (
            "Kalman filters",
            time(&mut || {
                drop(std::hint::black_box(ws.likelihood(
                    &y,
                    &data.missing,
                    &idx.mf,
                    &t,
                    &rr,
                    &q,
                    None,
                    &pstar,
                    &pinf,
                    &kopts,
                )))
            }),
        ),
        (
            "posterior evaluation (total)",
            time(&mut || {
                let _ = std::hint::black_box(post.eval(&xm));
            }),
        ),
    ];
    println!("{:<38} {:>10}", "component", "us/call");
    for (name, us) in rows {
        println!("{name:<38} {us:>10.1}");
    }
}
