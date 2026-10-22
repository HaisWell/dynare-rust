//! mode_check.m on SW07 against Dynare (reference/sw07/ref_mode_check.json, written by
//! dump_mode_check.m): grids, log posterior and log likelihood kernels of all 40 slices.
mod common;

use std::path::Path;

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::mode_check::{ModeCheckOptions, mode_check};
use dynare::estimation::plots::{dynare_param_name, mode_check_figures};
use dynare::kalman::KalmanOptions;
use dynare::modfile::parse_mod_file;
use serde_json::Value;

/// MATLAB matrices arrive as arrays of rows.
fn mat(v: &Value) -> dynare::linalg::Mat {
    let rows = v.as_array().unwrap();
    let c = rows[0].as_array().map_or(1, |x| x.len());
    dynare::linalg::Mat::from_fn(rows.len(), c, |i, j| {
        rows[i].as_array().map_or_else(|| vecf(&rows[i])[0], |x| vecf(&x[j])[0])
    })
}

/// Two grid points where T has a double unit root (`crhoa` z(15), `cprobw` z(6)), with
/// MATLAB's Schur form of the dynamic block of T (`reference/sw07/dump_mode_check_schur.m`):
/// dynare-rust's Pinf/Pstar and filters then give Dynare's fval, and so do our own Schur factors.
fn with_dynare_schur(m: &DsgeModel, data: &LikelihoodData) {
    use dynare::kalman::{KalmanWorkspace, compute_pinf_pstar, compute_pinf_pstar_with_schur};
    use dynare::linalg::schur::RealSchur;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let text = common::read_ref(dir.join("ref_mode_check_schur.json"));
    let pts: Value = serde_json::from_str(&text).unwrap();
    for p in pts.as_array().unwrap() {
        let x = vecf(&p["x"]);
        let res = m.resolve(&x).unwrap();
        let idx = &m.ss_index;
        let mut y = data.y.clone();
        for (i, &v) in idx.mfys.iter().enumerate() {
            for c in 0..y.ncols() {
                y[(i, c)] -= res.steady_state[v];
            }
        }
        let fval = |pstar: &dynare::linalg::Mat, pinf: &dynare::linalg::Mat| {
            let mut ws = KalmanWorkspace::new(res.t.nrows(), y.nrows(), res.q.nrows());
            let l = ws.likelihood(
                &y,
                &data.missing,
                &idx.mf,
                &res.t,
                &res.r,
                &res.q,
                Some(&res.h),
                pstar,
                pinf,
                &data.kalman,
            );
            l.minus_loglik - m.log_prior(&x)
        };
        let sch = RealSchur {
            t: mat(&p["schur_T"]),
            u: mat(&p["schur_U"]),
            wr: vecf(&p["schur_wr"]),
            wi: vecf(&p["schur_wi"]),
        };
        let qz = data.kalman.qz_criterium;
        let (pstar, pinf) =
            compute_pinf_pstar_with_schur(&idx.mf, &res.t, &res.r, &res.q, qz, &idx.restrict_columns, Some(sch))
                .unwrap();
        let (want, tr) = (p["fval"].as_f64().unwrap(), p["trace_Pinf"].as_f64().unwrap());
        assert!((pinf.trace() - tr).abs() < 1e-12, "{}: trace(Pinf) {} vs Dynare {tr}", p["name"], pinf.trace());
        let ours = fval(&pstar, &pinf);
        let (pstar2, pinf2) = compute_pinf_pstar(&idx.mf, &res.t, &res.r, &res.q, qz, &idx.restrict_columns);
        println!(
            "{}: fval with Dynare's Schur basis {ours:.10} (trace Pinf {}), with ours {:.10} (trace Pinf {}); Dynare {want:.10}",
            p["name"],
            pinf.trace(),
            fval(&pstar2, &pinf2),
            pinf2.trace()
        );
        assert!((ours - want).abs() < 1e-9 * want.abs(), "{}: {ours} vs {want}", p["name"]);
        assert!((pinf2.trace() - tr).abs() < 1e-9, "{}: trace(Pinf) with our Schur {}", p["name"], pinf2.trace());
        let theirs = fval(&pstar2, &pinf2);
        assert!((theirs - want).abs() < 1e-9 * want.abs(), "{}: {theirs} vs {want} with our Schur", p["name"]);
    }
}

/// Numbers; JSON null (MATLAB NaN) -> NaN.
fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

#[test]
fn sw07_mode_check_matches_dynare() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let text = common::read_ref(dir.join("ref_mode_check.json"));
    let r: Value = serde_json::from_str(&text).unwrap();
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let x = vecf(&r["xparam1"]);
    let m = DsgeModel::new(&mf, Some(&x)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let data = LikelihoodData::new(
        &table.sample(&mf.varobs, 5, None).unwrap(),
        KalmanOptions { presample: 8, lik_init: 3, ..Default::default() },
    );
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &m.bounds };
    let prior_sd: Vec<f64> = m.priors.iter().map(|p| p.sd).collect();
    let lp = |v: &[f64]| m.log_prior(v);
    let t = std::time::Instant::now();
    let slices = mode_check(&post, &lp, &x, &m.bounds, &prior_sd, &ModeCheckOptions::default());
    println!(
        "mode_check: {:.2} s (Dynare: {:.1} s)",
        t.elapsed().as_secs_f64(),
        r["seconds"].as_f64().unwrap_or(f64::NAN)
    );

    let params = r["params"].as_array().unwrap();
    assert_eq!(params.len(), slices.len());
    let (mut worst_z, mut worst_post, mut worst_lik) = (0.0f64, 0.0f64, 0.0f64);
    let mut nan_points = 0;
    for (k, (p, s)) in params.iter().zip(&slices).enumerate() {
        assert_eq!(p["name"].as_str().unwrap(), dynare_param_name(&m.estimated_names[k]));
        let z = vecf(&p["z"]);
        assert_eq!(z.len(), s.z.len(), "{}: grid size", m.estimated_names[k]);
        for (a, b) in z.iter().zip(&s.z) {
            worst_z = worst_z.max((a - b).abs() / b.abs().max(1.0));
        }
        for (name, refv, ours, worst) in [
            ("log_post", vecf(&p["log_post"]), &s.log_post, &mut worst_post),
            ("log_lik", vecf(&p["log_lik"]), &s.log_lik, &mut worst_lik),
        ] {
            for (i, (a, b)) in refv.iter().zip(ours).enumerate() {
                assert_eq!(
                    a.is_nan(),
                    b.is_nan(),
                    "{} {name}[{i}] at z = {}: Dynare {a}, ours {b}",
                    m.estimated_names[k],
                    s.z[i]
                );
                if a.is_nan() {
                    nan_points += 1;
                    continue;
                }
                let d = (a - b).abs() / a.abs().max(1.0);
                if d > 1e-9 {
                    println!(
                        "  {} {name}[{i}] z = {:.6} (mode {:.6}): Dynare {a:.10}, ours {b:.10}, rel {d:.1e}",
                        m.estimated_names[k], s.z[i], s.mode
                    );
                }
                *worst = worst.max(d);
            }
        }
    }
    println!(
        "max rel. diff: z {worst_z:.1e}, log-post {worst_post:.1e}, log-lik {worst_lik:.1e}; {nan_points} failed points in both"
    );
    // Every point agrees to < 1e-9, including those where T has a double unit root and which
    // Schur vector compute_Pinf_Pstar keeps in Pinf depends on the LAPACK build (MATLAB: NAG;
    // here: Accelerate); with_dynare_schur checks two of them against MATLAB's Schur form.
    assert!(worst_z < 1e-14, "grid {worst_z}");
    // OpenBLAS's Schur vectors (Linux, Windows) give up to 2.1e-9.
    let tol = if cfg!(target_os = "macos") { 1e-9 } else { 1e-8 };
    assert!(worst_post < tol && worst_lik < tol, "{worst_post} {worst_lik}");
    with_dynare_schur(&m, &data);

    let figs = mode_check_figures(&slices, &m.estimated_names, false);
    assert_eq!(figs.len(), 5);
    assert_eq!(figs[4].panels.len(), 4);
}
