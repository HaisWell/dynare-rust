//! Kalman module vs Dynare 7.2 on SW07: reference/sw07/ref.json (dump_reference.m)
//! and reference/sw07/ref_kalman.json (dump_kalman.m).
//!
//! Smoother mapping: DsgeSmoother (smoother_redux = false) runs on the `smoother_var_list`
//! state space, here all 50 endogenous variables in DR order, so state row i is variable
//! `order_var[i]` (declaration order) and `SmoothedVariables.<name>(t) = alphahat(i, t) +
//! ys(order_var[i])`, `FilteredVariables.<name>(t) = aK(1, i, t+1) + ys(order_var[i])`.

#![allow(clippy::needless_range_loop)]
mod common;

use dynare::kalman::{self, KalmanOptions, KalmanWorkspace, MissingData};
use dynare::linalg::Mat;
use serde_json::Value;
use std::path::Path;
use std::sync::OnceLock;

fn load(name: &str) -> Option<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07").join(name);
    Some(common::json_ref(&p))
}

fn reference() -> Option<&'static Value> {
    static R: OnceLock<Option<Value>> = OnceLock::new();
    R.get_or_init(|| load("ref.json")).as_ref()
}

fn ref_kalman() -> Option<&'static Value> {
    static R: OnceLock<Option<Value>> = OnceLock::new();
    R.get_or_init(|| load("ref_kalman.json")).as_ref()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// MATLAB jsonencode matrix (list of rows; a vector is a flat list = column).
fn mat(v: &Value) -> Mat {
    let rows = v.as_array().expect("matrix");
    if rows.is_empty() {
        return Mat::zeros(0, 0);
    }
    if rows[0].is_array() {
        let nc = rows[0].as_array().unwrap().len();
        Mat::from_fn(rows.len(), nc, |i, j| num(&rows[i][j]))
    } else {
        Mat::from_fn(rows.len(), 1, |i, _| num(&rows[i]))
    }
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(num).collect(),
        other => vec![num(other)],
    }
}

fn idx0(v: &Value) -> Vec<usize> {
    vecf(v).iter().map(|&x| x as usize - 1).collect()
}

fn opts_from(r: &Value) -> KalmanOptions {
    let o = &r["options"];
    KalmanOptions {
        kalman_tol: num(&o["kalman_tol"]),
        diffuse_kalman_tol: num(&o["diffuse_kalman_tol"]),
        riccati_tol: num(&o["riccati_tol"]),
        presample: num(&o["presample"]) as usize,
        qz_criterium: num(&o["qz_criterium"]),
        lik_init: 3,
        kalman_algo: 0,
        use_univariate_filters_if_singularity_is_detected: num(&o["use_univariate_filters_if_singularity_is_detected"])
            != 0.0,
        rescale_prediction_error_covariance: false,
        lyapunov_complex_threshold: 1e-15,
        ..Default::default()
    }
}

fn maxabs(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn maxrel(a: &Mat, b: &Mat) -> f64 {
    let scale = b.amax().max(1e-300);
    maxabs(a, b) / scale
}

#[test]
fn pinf_pstar_at_mode() {
    let r = reference().expect("reference/setup unavailable");
    let ss = &r["mode_ss"];
    let (t, rm, q) = (mat(&ss["T"]), mat(&ss["R"]), mat(&ss["Q"]));
    let mf = idx0(&r["bayestopt"]["mf1"]);
    let rc = idx0(&r["mode_dr"]["restrict_columns"]);
    let (pstar, pinf) = kalman::compute_pinf_pstar(&mf, &t, &rm, &q, num(&r["options"]["qz_criterium"]), &rc);
    let (es, ei) = (maxabs(&pstar, &mat(&ss["Pstar"])), maxabs(&pinf, &mat(&ss["Pinf"])));
    println!("Pstar max abs err {es:.3e} (max |Pstar| {:.3e}), Pinf max abs err {ei:.3e}", mat(&ss["Pstar"]).amax());
    assert!(es < 1e-9 * mat(&ss["Pstar"]).amax().max(1.0), "Pstar err {es}");
    assert!(ei < 1e-9, "Pinf err {ei}");
}

fn mode_problem(r: &Value) -> (Mat, MissingData, Vec<usize>, Mat, Mat, Mat, Mat, Mat) {
    let ss = &r["mode_ss"];
    let y = mat(&r["mode_kalman"]["Y"]);
    let md = MissingData::from_data(&y);
    let mf = idx0(&r["bayestopt"]["mf1"]);
    (y, md, mf, mat(&ss["T"]), mat(&ss["R"]), mat(&ss["Q"]), mat(&ss["Pstar"]), mat(&ss["Pinf"]))
}

#[test]
fn likelihood_at_mode() {
    let r = reference().expect("reference/setup unavailable");
    let (y, md, mf, t, rm, q, pstar, pinf) = mode_problem(r);
    let opts = opts_from(r);
    let out = kalman::likelihood(&y, &md, &mf, &t, &rm, &q, None, &pstar, &pinf, &opts);
    let total = num(&r["mode_kalman"]["total"]);
    let rel = (out.minus_loglik - total).abs() / total.abs();
    println!(
        "minus_loglik {:.12} ref {total:.12} rel err {rel:.3e} univariate {} d {}",
        out.minus_loglik, out.univariate, out.diffuse_periods
    );
    assert_eq!(out.info, 0);
    // dsge_likelihood at the mode: kalman_filter_d is singular -> univariate_kalman_filter_d,
    // then the multivariate kalman_filter succeeds (ref_kalman.json: lik_cols = 1). The
    // univariate regular filter of dump_reference.m gives the same values.
    assert!(!out.univariate);
    assert_eq!(out.diffuse_periods, num(&r["mode_kalman"]["diffuse_periods"]) as usize);
    assert!(rel < 1e-10, "rel err {rel}");
    // also = fval + log prior of points[0]
    let p0 = &r["points"][0];
    let fv = num(&p0["minus_loglik"]) + num(&p0["log_prior"]);
    assert!((out.minus_loglik - fv).abs() / fv < 1e-10);
    // per period: dlik (1 x pp) and lik (nobs x pp)
    let dlik = vecf(&r["mode_kalman"]["dlik"]);
    let lik = mat(&r["mode_kalman"]["lik"]);
    let dsum: f64 = dlik.iter().sum();
    let mut worst: f64 = (out.lik[0] - dsum).abs();
    for s in 0..lik.nrows() {
        let rs: f64 = lik.row(s).iter().sum();
        worst = worst.max((out.lik[s + 1] - rs).abs());
    }
    println!("per-period lik max abs err {worst:.3e}");
    assert!(worst < 1e-8);
}

#[test]
fn univariate_filters_by_observable() {
    let r = reference().expect("reference/setup unavailable");
    let (y, _md, mf, t, rm, q, pstar, pinf) = mode_problem(r);
    let o = opts_from(r);
    let (mm, pp) = (t.nrows(), y.nrows());
    let mut ws = KalmanWorkspace::new(mm, pp, q.nrows());
    ws.set_system(&t, &rm, &q);
    ws.set_state(&vec![0.0; mm], &pstar, Some(&pinf));
    let h = vec![0.0; pp];
    let last = y.ncols() - 1;
    let (dl, d) = ws.diffuse_univariate(&y, None, &mf, &h, 0, last, o.kalman_tol, o.diffuse_kalman_tol, o.presample);
    assert_eq!(d, 1);
    assert_eq!(dl, num(&r["mode_kalman"]["dLIK"]));
    let dlik = vecf(&r["mode_kalman"]["dlik"]);
    let e = ws.dlik_by_obs.iter().zip(&dlik).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(e < 1e-10, "dlik err {e}");
    let a1 = vecf(&r["mode_kalman"]["a_after_diffuse"]);
    let ea = ws.state().iter().zip(&a1).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let p1 = mat(&r["mode_kalman"]["Pstar_after_diffuse"]);
    let ep = maxrel(&ws.cov(), &p1);
    println!("after diffuse: a err {ea:.3e}, Pstar rel err {ep:.3e}");
    assert!(ea < 1e-9 && ep < 1e-11);
    let lik_tot = ws.univariate(&y, None, &mf, &h, d, last, o.kalman_tol, o.riccati_tol, o.presample, d);
    assert!((lik_tot - num(&r["mode_kalman"]["LIK"])).abs() < 1e-8);
    let lik = mat(&r["mode_kalman"]["lik"]);
    let mut worst: f64 = 0.0;
    for s in 0..lik.nrows() {
        for i in 0..pp {
            worst = worst.max((ws.lik_by_obs[s * pp + i] - lik[(s, i)]).abs());
        }
    }
    println!("lik by observable max abs err {worst:.3e}");
    assert!(worst < 1e-9);
}

#[test]
fn likelihood_points_and_branches() {
    let r = ref_kalman().expect("reference/setup unavailable");
    let base = opts_from(r);
    let mf = idx0(&r["mf1"]);
    let rc = idx0(&r["restrict_columns"]);
    let hdiag = vecf(&r["Hdiag"]);
    let pp = mf.len();
    let hmat = Mat::from_fn(pp, pp, |i, j| if i == j { hdiag[i] } else { 0.0 });
    let data_miss = mat(&r["data_miss"]); // nobs x pp
    let mut worst_rel: f64 = 0.0;
    let mut nrun = 0;
    for p in r["points"].as_array().unwrap() {
        let (t, rm, q) = (mat(&p["T"]), mat(&p["R"]), mat(&p["Q"]));
        let constant = vecf(&p["constant"]);
        let (pstar, pinf) = kalman::compute_pinf_pstar(&mf, &t, &rm, &q, base.qz_criterium, &rc);
        let es = maxabs(&pstar, &mat(&p["Pstar"])) / mat(&p["Pstar"]).amax();
        let ei = maxabs(&pinf, &mat(&p["Pinf"]));
        let plyap = kalman::lyapunov_solver(&t, &rm, &q, base.qz_criterium, base.lyapunov_complex_threshold).unwrap();
        let el = maxabs(&plyap, &mat(&p["Plyap"])) / mat(&p["Plyap"]).amax();
        println!("point {}: Pstar rel {es:.2e} Pinf {ei:.2e} Plyap rel {el:.2e}", p["k"]);
        assert!(es < 1e-9 && ei < 1e-9 && el < 1e-9);
        // data: full = ref.json data; missing variant from ref_kalman
        let full = reference().map(|rr| mat(&rr["data"]));
        let mut ws = KalmanWorkspace::new(t.nrows(), pp, q.nrows());
        for run in p["runs"].as_array().unwrap() {
            let lik_init = num(&run["lik_init"]) as u8;
            let algo = num(&run["kalman_algo"]) as u8;
            let missing = num(&run["missing"]) != 0.0;
            let with_h = num(&run["H"]) != 0.0;
            let raw = if missing {
                data_miss.clone()
            } else {
                match &full {
                    Some(f) => f.clone(),
                    None => continue,
                }
            };
            let y = Mat::from_fn(pp, raw.nrows(), |i, tt| raw[(tt, i)] - constant[i]);
            let md = MissingData::from_data(&y);
            assert_eq!(md.no_more_missing_observations + 1, num(&run["no_more_missing_observations"]) as usize);
            let opts = KalmanOptions { lik_init, kalman_algo: algo, ..base.clone() };
            let ps = if lik_init == 1 { &plyap } else { &pstar };
            let out =
                ws.likelihood(&y, &md, &mf, &t, &rm, &q, if with_h { Some(&hmat) } else { None }, ps, &pinf, &opts);
            let want = num(&run["LIK"]);
            let rel = (out.minus_loglik - want).abs() / want.abs();
            let cols = num(&run["lik_cols"]) as usize;
            println!(
                "  lik_init {lik_init} algo {algo} missing {missing} H {with_h}: {:.10} vs {want:.10} rel {rel:.2e} (univariate {} / ref {})",
                out.minus_loglik,
                out.univariate,
                cols > 1
            );
            assert_eq!(out.info, 0);
            assert_eq!(out.univariate, cols > 1, "filter path");
            worst_rel = worst_rel.max(rel);
            // per-period lik (after presample)
            let lik = mat(&run["lik"]);
            let presample = opts.presample;
            let mut worst = 0.0f64;
            for s in 0..lik.nrows() {
                let rs: f64 = lik.row(s).iter().sum();
                worst = worst.max((out.lik[s + presample] - rs).abs());
            }
            assert!(worst < 1e-7, "per-period err {worst}");
            nrun += 1;
        }
    }
    println!("{nrun} runs, worst relative error {worst_rel:.3e}");
    assert!(worst_rel < 1e-8);
}

#[test]
fn smoother_vs_dynare() {
    let r = ref_kalman().expect("reference/setup unavailable");
    let rr = reference().expect("reference/setup unavailable");
    let base = opts_from(r);
    let sys = &r["smoother_system"];
    let (t, rm, q) = (mat(&sys["T"]), mat(&sys["R"]), mat(&sys["Q"]));
    let mf = idx0(&sys["mf"]);
    let pp = mf.len();
    let constant = vecf(&sys["constant"]);
    let order_var = idx0(&sys["order_var"]);
    let ys = vecf(&sys["ys"]);
    let (pstar, pinf) = kalman::compute_pinf_pstar(&mf, &t, &rm, &q, base.qz_criterium, &[]);
    let es = maxabs(&pstar, &mat(&sys["Pstar"])) / mat(&sys["Pstar"]).amax();
    let ei = maxabs(&pinf, &mat(&sys["Pinf"]));
    println!("smoother system: Pstar rel {es:.2e} Pinf {ei:.2e}");
    assert!(es < 1e-9 && ei < 1e-9);
    let plyap = kalman::lyapunov_solver(&t, &rm, &q, base.qz_criterium, base.lyapunov_complex_threshold).unwrap();
    let hdiag = vecf(&r["Hdiag"]);
    let hmat = Mat::from_fn(pp, pp, |i, j| if i == j { hdiag[i] } else { 0.0 });
    let full = mat(&rr["data"]);
    let data_miss = mat(&r["data_miss"]);
    for (c, run) in r["smoother"].as_array().unwrap().iter().enumerate() {
        let lik_init = num(&run["lik_init"]) as u8;
        let algo = num(&run["kalman_algo"]) as u8;
        let missing = num(&run["missing"]) != 0.0;
        let with_h = num(&run["H"]) != 0.0;
        let raw = if missing { &data_miss } else { &full };
        let y = Mat::from_fn(pp, raw.nrows(), |i, tt| raw[(tt, i)] - constant[i]);
        let md = MissingData::from_data(&y);
        let opts = KalmanOptions { lik_init, kalman_algo: algo, ..base.clone() };
        let (ps, pi) = if lik_init == 1 { (&plyap, None) } else { (&pstar, Some(&pinf)) };
        let out =
            kalman::smoother(&y, &md, &mf, &t, &rm, &q, if with_h { Some(&hmat) } else { None }, ps, pi, &opts, true)
                .unwrap();
        let ea = maxrel(&out.alphahat, &mat(&run["alphahat"]));
        let ee = maxrel(&out.etahat, &mat(&run["etahat"]));
        let eh = maxrel(&out.ahat, &mat(&run["ahat"]));
        let ak1 = mat(&run["aK1"]);
        let ef = maxrel(&out.filtered, &ak1.columns(1, y.ncols()).into_owned());
        let a0 = mat(&run["alphahat0"]);
        let e0 = if a0.nrows() > 0 {
            maxrel(&Mat::from_column_slice(a0.nrows(), 1, out.alphahat0.as_slice()), &a0)
        } else {
            0.0
        };
        let vd = mat(&run["Vdiag"]);
        let vs = out.state_uncertainty.as_ref().unwrap();
        let vmine = Mat::from_fn(vd.nrows(), vd.ncols(), |i, tt| vs[tt][(i, i)]);
        let ev = maxrel(&vmine, &vd);
        println!(
            "smoother {c} (lik_init {lik_init} algo {algo} missing {missing} H {with_h}): d {} (ref {}) univ {} info {} | alphahat {ea:.2e} etahat {ee:.2e} ahat {eh:.2e} filtered {ef:.2e} alphahat0 {e0:.2e} Vdiag {ev:.2e}",
            out.d, run["d"], out.univariate, out.info
        );
        assert_eq!(out.d, num(&run["d"]) as usize);
        assert!(ea < 1e-8 && ee < 1e-8 && eh < 1e-8 && ef < 1e-8 && e0 < 1e-8 && ev < 1e-8);
        if c == 0 {
            // Dynare's stored oo_ results at the mode
            assert!(out.univariate);
            let mut worst: f64 = 0.0;
            let mut worstf: f64 = 0.0;
            let names = rr["endo_names"].as_array().unwrap();
            for (i, &dv) in order_var.iter().enumerate() {
                let name = names[dv].as_str().unwrap();
                let Some(sv) = rr["SmoothedVariables"].get(name) else { continue };
                let sv = vecf(sv);
                let fv = vecf(&rr["FilteredVariables"][name]);
                for tt in 0..y.ncols() {
                    worst = worst.max((out.alphahat[(i, tt)] + ys[dv] - sv[tt]).abs() / sv[tt].abs().max(1.0));
                    worstf = worstf.max((out.filtered[(i, tt)] + ys[dv] - fv[tt]).abs() / fv[tt].abs().max(1.0));
                }
            }
            let mut worsts: f64 = 0.0;
            for (k, name) in rr["exo_names"].as_array().unwrap().iter().enumerate() {
                let sh = vecf(&rr["SmoothedShocks"][name.as_str().unwrap()]);
                for tt in 0..y.ncols() {
                    worsts = worsts.max((out.etahat[(k, tt)] - sh[tt]).abs() / sh[tt].abs().max(1.0));
                }
            }
            println!(
                "oo_.SmoothedVariables err {worst:.2e}, FilteredVariables {worstf:.2e}, SmoothedShocks {worsts:.2e}"
            );
            assert!(worst < 1e-8 && worstf < 1e-8 && worsts < 1e-8);
        }
    }
}

/// cargo test --release --test kalman_sw07 -- --ignored --nocapture timing
#[test]
#[ignore]
fn timing() {
    let r = reference().expect("reference/setup unavailable");
    let (y, md, mf, t, rm, q, pstar, pinf) = mode_problem(r);
    let opts = opts_from(r);
    let rc = idx0(&r["mode_dr"]["restrict_columns"]);
    let mut ws = KalmanWorkspace::new(t.nrows(), y.nrows(), q.nrows());
    let n = 2000;
    for _ in 0..50 {
        ws.likelihood(&y, &md, &mf, &t, &rm, &q, None, &pstar, &pinf, &opts);
    }
    let t0 = std::time::Instant::now();
    let mut acc = 0.0;
    for _ in 0..n {
        acc += ws.likelihood(&y, &md, &mf, &t, &rm, &q, None, &pstar, &pinf, &opts).minus_loglik;
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
    let t1 = std::time::Instant::now();
    for _ in 0..200 {
        let (ps, pi) = kalman::compute_pinf_pstar(&mf, &t, &rm, &q, opts.qz_criterium, &rc);
        acc += ps[(0, 0)] + pi[(0, 0)];
    }
    let us2 = t1.elapsed().as_secs_f64() * 1e6 / 200.0;
    println!("likelihood (filters) {us:.1} us, compute_pinf_pstar {us2:.1} us, total {:.1} us  [{acc:.3}]", us + us2);
    println!("MATLAB dsge_likelihood: {:.1} us", num(&r["timing_seconds_per_lik"]) * 1e6);
}
