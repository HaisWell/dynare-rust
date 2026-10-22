//! CMA-ES (`cmaes.m`, `mode_compute = 9`): known minima, bounds, reproducibility,
//! parallel == sequential, and (ignored) the SW07 posterior.
mod common;

use dynare::estimation::optim::{CmaesOptions, cmaes, cmaes_insigma};
use dynare::estimation::{Bounds, FnObjective};

fn rosen(x: &[f64]) -> f64 {
    (0..x.len() - 1).map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2)).sum()
}

fn box_bounds(n: usize, l: f64, u: f64) -> Bounds {
    Bounds { lb: vec![l; n], ub: vec![u; n] }
}

fn quiet(seed: u64) -> CmaesOptions {
    CmaesOptions { verbose: false, seed, ..Default::default() }
}

#[test]
fn insigma_follows_dynare() {
    let b = Bounds { lb: vec![0.0, -1.0, f64::NEG_INFINITY], ub: vec![1.0, 1.0, 2.0] };
    assert_eq!(cmaes_insigma(&b), vec![0.2, 0.4, 0.01]);
    // badly conditioned: the largest entry is shrunk until max/min <= 1e6
    let b = Bounds { lb: vec![0.0, 0.0], ub: vec![1e-3, 1e5] };
    let h = cmaes_insigma(&b);
    assert!(h[1] / h[0] <= 1e6 && h[1] / h[0] > 0.9e6, "{h:?}");
}

#[test]
fn rosenbrock_5d_bounded() {
    let n = 5;
    let obj = FnObjective { n, f: rosen };
    let x0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
    let b = box_bounds(n, -5.0, 5.0);
    let r = cmaes(&obj, &x0, &cmaes_insigma(&b), &b, &quiet(1)).unwrap();
    println!("rosen5: f {:.3e} x {:?} it {} fc {} {}", r.fval, r.x, r.iterations, r.func_count, r.message);
    assert!(r.fval < 1e-8, "{}", r.fval);
    assert!(r.x.iter().all(|v| (v - 1.0).abs() < 1e-3));
    assert_eq!(r.exitflag, 1);
}

#[test]
fn ill_conditioned_quadratic_with_active_bound() {
    // f = sum 10^(6 i/(n-1)) (x_i - c_i)^2, c_0 outside the box: minimum at the bound.
    let n = 8;
    let c: Vec<f64> = (0..n).map(|i| 0.5 + 0.1 * i as f64).collect();
    let mut c2 = c.clone();
    c2[0] = 3.0; // ub = 2
    let f = move |x: &[f64]| -> f64 {
        (0..x.len()).map(|i| 10f64.powf(6.0 * i as f64 / 7.0) * (x[i] - c2[i]).powi(2)).sum()
    };
    let obj = FnObjective { n, f };
    let b = box_bounds(n, -2.0, 2.0);
    let r = cmaes(&obj, &vec![0.0; n], &cmaes_insigma(&b), &b, &quiet(7)).unwrap();
    println!("quad: f {:.6} x {:?} it {} fc {} {}", r.fval, r.x, r.iterations, r.func_count, r.message);
    assert!(r.x.iter().zip(&b.lb).zip(&b.ub).all(|((v, l), u)| v >= l && v <= u));
    assert!((r.fval - 1.0).abs() < 1e-6, "{}", r.fval); // (2-3)^2
    assert!((r.x[0] - 2.0).abs() < 1e-6);
    for i in 1..n {
        assert!((r.x[i] - c[i]).abs() < 1e-3, "x[{i}] = {}", r.x[i]);
    }
}

#[test]
fn infinite_values_and_unbounded() {
    // +inf outside the unit ball (as a failed dsge_likelihood), no bounds.
    let n = 4;
    let f = |x: &[f64]| -> f64 {
        let r2: f64 = x.iter().map(|v| v * v).sum();
        if r2 > 1.0 { f64::INFINITY } else { x.iter().enumerate().map(|(i, v)| (v - 0.1 * i as f64).powi(2)).sum() }
    };
    let obj = FnObjective { n, f };
    let b = Bounds::unbounded(n);
    let r = cmaes(&obj, &vec![0.0; n], &cmaes_insigma(&b), &b, &quiet(3)).unwrap();
    println!("ball: f {:.3e} it {} fc {} {}", r.fval, r.iterations, r.func_count, r.message);
    // CMA-ES stops on tolfun, so the value it stops at depends on the path, which a different
    // libm perturbs (2.8e-10 on Linux and Windows).
    assert!(r.fval < if cfg!(target_os = "macos") { 1e-10 } else { 1e-9 });
}

#[test]
fn reproducible_and_parallel_equals_sequential() {
    let n = 6;
    let obj = FnObjective { n, f: rosen };
    let b = box_bounds(n, -3.0, 3.0);
    let s = cmaes_insigma(&b);
    let x0 = vec![0.5; n];
    let a = cmaes(&obj, &x0, &s, &b, &CmaesOptions { parallel: true, ..quiet(11) }).unwrap();
    let c = cmaes(&obj, &x0, &s, &b, &CmaesOptions { parallel: false, ..quiet(11) }).unwrap();
    let d = cmaes(&obj, &x0, &s, &b, &CmaesOptions { parallel: true, ..quiet(12) }).unwrap();
    assert_eq!(
        a.x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        c.x.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(a.fval.to_bits(), c.fval.to_bits());
    assert_eq!((a.iterations, a.func_count), (c.iterations, c.func_count));
    assert_ne!(a.func_count as f64 + a.fval, d.func_count as f64 + d.fval);
}

#[test]
fn optim_options() {
    use dynare::modfile::OptValue;
    let mut o = CmaesOptions::default();
    o.apply_optim(&[
        ("MaxIter".into(), OptValue::Num(5.0)),
        ("TolFun".into(), OptValue::Num(1e-3)),
        ("verbosity".into(), OptValue::Num(0.0)),
        ("SaveFiles".into(), OptValue::Num(0.0)),
    ])
    .unwrap();
    assert_eq!(o.max_iter, Some(5.0));
    assert_eq!(o.tol_fun, 1e-3);
    assert!(!o.verbose);
    assert!(o.apply_optim(&[("CMAESResume".into(), OptValue::Num(1.0))]).is_err());
    let n = 3;
    let obj = FnObjective { n, f: rosen };
    let b = box_bounds(n, -5.0, 5.0);
    let r = cmaes(&obj, &vec![0.0; n], &cmaes_insigma(&b), &b, &o).unwrap();
    assert_eq!(r.iterations, 5);
    assert_eq!(r.message, "maxiter");
    assert_eq!(r.exitflag, -1);
}

/// SW07 posterior from the `estimated_params` init values, Dynare's defaults:
/// `cargo test --release --test estimation_cmaes sw07 -- --ignored --nocapture`
/// (`CMAES_MAXFUNEVALS`, `CMAES_SEED`, `CMAES_SIGMA_SCALE` override).
#[test]
#[ignore]
fn sw07_mode_compute_9() {
    use dynare::data::Table;
    use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
    use dynare::estimation::Objective;
    use dynare::kalman::KalmanOptions;
    use dynare::modfile::parse_mod_file;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let x0 = DsgeModel::new(&mf, None).unwrap().init_vector().unwrap();
    let m = DsgeModel::new(&mf, Some(&x0)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let data = LikelihoodData::new(
        &table.sample(&mf.varobs, 5, None).unwrap(),
        KalmanOptions { presample: 8, lik_init: 3, ..Default::default() },
    );
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &m.bounds };
    let f0 = post.eval(&x0).0;
    let env = |k: &str| std::env::var(k).ok();
    let mut o = CmaesOptions {
        verbose: true,
        disp_modulo: 100,
        seed: env("CMAES_SEED").map_or(0, |s| s.parse().unwrap()),
        ..Default::default()
    };
    if let Some(v) = env("CMAES_MAXFUNEVALS") {
        o.max_fun_evals = v.parse().unwrap();
    }
    // CMAES_SIGMA_SCALE multiplies Dynare's H0 (with H0 itself, every candidate of the
    // first generation is +inf on SW07 and cmaes.m stops at once on 'equalfunvals', after 17
    // evaluations; CMAES_SIGMA_SCALE=0.05 lets it move).
    let scale: f64 = env("CMAES_SIGMA_SCALE").map_or(1.0, |s| s.parse().unwrap());
    let insigma: Vec<f64> = cmaes_insigma(&m.bounds).iter().map(|v| v * scale).collect();
    let t = std::time::Instant::now();
    let r = cmaes(&post, &x0, &insigma, &m.bounds, &o).unwrap();
    let secs = t.elapsed().as_secs_f64();
    let fmode = 349.85607197568856; // reference/sw07/ref.json points[0]
    println!(
        "SW07 mode_compute=9: fval {:.6} (start {f0:.4}, Dynare mode {fmode}, diff {:.3e}), {} iterations, {} evaluations, {:.1}s, stop: {}",
        r.fval,
        r.fval - fmode,
        r.iterations,
        r.func_count,
        secs,
        r.message
    );
    assert!(r.fval <= f0);
}

/// Distribution of iterations / evaluations / success over 20 seeds against Dynare's
/// cmaes.m (`reference/estimation/dump_cmaes.m` -> `ref_cmaes.json`): Rosenbrock 5d,
/// box [-5,5], H0 = 0.2*(ub-lb).
#[test]
fn rosenbrock_statistics_match_matlab() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/estimation/ref_cmaes.json");
    let txt = common::read_ref(path);
    let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
    let col = |k: &str| -> Vec<f64> { v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() };
    let med = |mut x: Vec<f64>| {
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (x[x.len() / 2 - 1] + x[x.len() / 2]) / 2.0
    };
    let (m_it, m_fc, m_f) = (col("it"), col("fc"), col("f"));
    let n = 5;
    let obj = FnObjective { n, f: rosen };
    let x0 = vec![-1.2, 1.0, -1.2, 1.0, -1.2];
    let b = box_bounds(n, -5.0, 5.0);
    let s = cmaes_insigma(&b);
    let runs: Vec<_> = (1..=20).map(|seed| cmaes(&obj, &x0, &s, &b, &quiet(seed)).unwrap()).collect();
    let r_it: Vec<f64> = runs.iter().map(|r| r.iterations as f64).collect();
    let r_fc: Vec<f64> = runs.iter().map(|r| r.func_count as f64).collect();
    let ok = |f: &[f64]| f.iter().filter(|&&v| v < 1e-8).count();
    let r_f: Vec<f64> = runs.iter().map(|r| r.fval).collect();
    let (mi, ri) = (med(m_it), med(r_it));
    let (mf, rf) = (med(m_fc), med(r_fc));
    println!("median iterations MATLAB {mi} Rust {ri}; evaluations {mf} / {rf}; successes {} / {}", ok(&m_f), ok(&r_f));
    assert!((ri / mi - 1.0).abs() < 0.15, "median iterations {ri} vs {mi}");
    assert!((rf / mf - 1.0).abs() < 0.15, "median evaluations {rf} vs {mf}");
    assert!(ok(&r_f) >= 16, "only {} of 20 runs reached the minimum", ok(&r_f));
    assert!(runs.iter().all(|r| r.message.contains("tolfun") || r.message.contains("tolx")));
}
