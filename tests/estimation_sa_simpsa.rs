//! Simulated annealing (`simulated_annealing.m`, mode_compute=2) and SIMPSA
//! (`simpsa.m`, mode_compute=10): convergence on test functions, bounds,
//! reproducibility, and (ignored) SW07 runs with Dynare's defaults:
//! `cargo test --release --test estimation_sa_simpsa sw07 -- --ignored --nocapture`.

use dynare::estimation::optim::minimize::{HUGE_NUMBER, finite_bounds, key_values};
use dynare::estimation::optim::{SaOptions, SimpsaOptions, simpsa, simulated_annealing};
use dynare::estimation::{FnObjective, Objective};
use dynare::modfile::OptValue;

fn rosen(x: &[f64]) -> f64 {
    (0..x.len() - 1).map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2)).sum()
}

/// Shifted, badly scaled quadratic with minimum 3 at c.
fn quad(x: &[f64]) -> f64 {
    let c = [1.5, -2.0, 0.5, 3.0];
    let w = [1.0, 10.0, 0.5, 4.0];
    3.0 + x.iter().zip(c).zip(w).map(|((a, b), w)| w * (a - b) * (a - b)).sum::<f64>()
}

fn sa_quiet() -> SaOptions {
    SaOptions { verbosity: 0, ..Default::default() }
}

fn simpsa_quiet() -> SimpsaOptions {
    SimpsaOptions { display: false, ..Default::default() }
}

#[test]
fn sa_defaults_on_quadratic_and_rosenbrock() {
    let obj = FnObjective { n: 4, f: quad };
    let (lb, ub) = (vec![-10.0; 4], vec![10.0; 4]);
    let r = simulated_annealing(&obj, &[0.0; 4], &lb, &ub, &sa_quiet()).unwrap();
    println!("SA quad: f {} x {:?} evals {} exit {}", r.fval, r.x, r.func_count, r.exitflag);
    assert_eq!(r.exitflag, 0, "{}", r.message);
    assert!((r.fval - 3.0).abs() < 1e-6);
    assert!(r.x.iter().zip([1.5, -2.0, 0.5, 3.0]).all(|(a, b)| (a - b).abs() < 1e-3));
    assert!((obj.eval(&r.x).0 - r.fval).abs() < 1e-14);

    for n in [2usize, 5] {
        let obj = FnObjective { n, f: rosen };
        let mut x0 = vec![-1.2; n];
        x0[n - 1] = 1.0;
        let (lb, ub) = (vec![-5.0; n], vec![5.0; n]);
        // Defaults: MaxIter = 1e5 evaluations ends in Rosenbrock's valley (exitflag 1).
        let d = simulated_annealing(&obj, &x0, &lb, &ub, &sa_quiet()).unwrap();
        // With more evaluations it meets its termination criteria.
        let mut o = sa_quiet();
        o.apply_optim(&[("MaxIter".into(), OptValue::Num(5e6))]).unwrap();
        let r = simulated_annealing(&obj, &x0, &lb, &ub, &o).unwrap();
        println!(
            "SA rosen{n}: defaults f {:e} ({} evals, exit {}); MaxIter=5e6: f {:e} x {:?} ({} evals, exit {})",
            d.fval, d.func_count, d.exitflag, r.fval, r.x, r.func_count, r.exitflag
        );
        // Coordinate-wise moves crawl along the curved valley: Dynare's
        // simulated_annealing.m stops at the same level (reference/estimation/
        // check_sa_simpsa.m: rosen2 5e-5..4e-4, rosen5 2e-3..4e-3, seeds 1-3).
        assert!(d.fval < 1e-2, "rosen{n}: {}", d.fval);
        assert_eq!(r.exitflag, 0, "{}", r.message);
        assert!(r.fval < 1e-2, "rosen{n}: {}", r.fval);
        assert!(r.x.iter().all(|v| (v - 1.0).abs() < 0.1));
    }
}

#[test]
fn sa_options_bounds_and_reproducibility() {
    let obj = FnObjective { n: 4, f: quad };
    // Tight box excluding the unconstrained minimum in coordinate 2 and 4.
    let (lb, ub) = (vec![-1.0, -1.0, -1.0, -1.0], vec![2.0, 2.0, 2.0, 2.0]);
    let mut seen_out = false;
    let o = sa_quiet();
    let r = simulated_annealing(&obj, &[0.0; 4], &lb, &ub, &o).unwrap();
    assert!(r.x.iter().zip(&lb).zip(&ub).all(|((v, l), u)| v >= l && v <= u));
    assert!((r.x[1] + 1.0).abs() < 1e-3 && (r.x[3] - 2.0).abs() < 1e-3, "{:?}", r.x);
    seen_out |= r.func_count > 0;
    assert!(seen_out);
    let r2 = simulated_annealing(&obj, &[0.0; 4], &lb, &ub, &o).unwrap();
    assert_eq!(r.x, r2.x);
    assert_eq!(r.func_count, r2.func_count);
    let r3 = simulated_annealing(&obj, &[0.0; 4], &lb, &ub, &SaOptions { seed: 7, ..o.clone() }).unwrap();
    assert_ne!(r.x, r3.x);
    // optim overrides: MaxIter stops early with exitflag 1.
    let mut o2 = sa_quiet();
    let kv = key_values(
        Some(&OptValue::List(vec![
            OptValue::Str("'MaxIter'".into()),
            OptValue::Num(500.0),
            OptValue::Str("rt".into()),
            OptValue::Num(0.5),
        ])),
        "sa",
    )
    .unwrap();
    o2.apply_optim(&kv).unwrap();
    assert_eq!((o2.max_iter, o2.rt), (500, 0.5));
    let r = simulated_annealing(&obj, &[0.0; 4], &lb, &ub, &o2).unwrap();
    assert_eq!((r.exitflag, r.func_count), (1, 500));
    // Errors: start out of bounds, non-positive temperature.
    assert!(simulated_annealing(&obj, &[5.0; 4], &lb, &ub, &o).is_err());
    assert!(simulated_annealing(&obj, &[0.0; 4], &lb, &ub, &SaOptions { initial_temperature: 0.0, ..o }).is_err());
}

#[test]
fn sa_survives_infinite_objective() {
    // +inf (failure) on part of the box: never accepted from a finite point.
    let obj = FnObjective {
        n: 2,
        f: |x: &[f64]| if x[0] > 1.0 { f64::INFINITY } else { (x[0] - 0.5).powi(2) + x[1] * x[1] },
    };
    let r = simulated_annealing(&obj, &[0.0, 0.3], &[-3.0, -3.0], &[3.0, 3.0], &sa_quiet()).unwrap();
    assert!(r.fval < 1e-6 && r.x[0] <= 1.0, "{} {:?}", r.fval, r.x);
}

#[test]
fn simpsa_defaults_on_quadratic_and_rosenbrock() {
    let obj = FnObjective { n: 4, f: quad };
    let (lb, ub) = (vec![-10.0; 4], vec![10.0; 4]);
    let r = simpsa(&obj, &[0.0; 4], &lb, &ub, &simpsa_quiet()).unwrap();
    println!(
        "SIMPSA quad: f {} x {:?} evals {} it {} exit {} {}",
        r.fval, r.x, r.func_count, r.iterations, r.exitflag, r.message
    );
    assert!(r.exitflag == 1 || r.exitflag == 2, "{}", r.message);
    assert!((r.fval - 3.0).abs() < 1e-3);
    assert!((obj.eval(&r.x).0 - r.fval).abs() < 1e-14);

    for n in [2usize, 5] {
        let obj = FnObjective { n, f: rosen };
        let mut x0 = vec![-1.2; n];
        x0[n - 1] = 1.0;
        let (lb, ub) = (vec![-5.0; n], vec![5.0; n]);
        // Default tolerances (1e-4) stop early on Rosenbrock's valley; tighten them.
        let mut o = simpsa_quiet();
        o.apply_optim(&[("TolFun".into(), OptValue::Num(1e-12)), ("TolX".into(), OptValue::Num(1e-10))]).unwrap();
        let d = simpsa(&obj, &x0, &lb, &ub, &simpsa_quiet()).unwrap();
        let r = simpsa(&obj, &x0, &lb, &ub, &o).unwrap();
        println!(
            "SIMPSA rosen{n}: defaults f {:e} ({} evals, {}); tight f {:e} x {:?} ({} evals, {})",
            d.fval, d.func_count, d.message, r.fval, r.x, r.func_count, r.message
        );
        assert!(d.fval < rosen(&x0));
        assert!(r.fval < 1e-6, "rosen{n}: {}", r.fval);
        assert!(r.x.iter().all(|v| (v - 1.0).abs() < 1e-2));
    }
}

#[test]
fn simpsa_options_bounds_and_reproducibility() {
    let obj = FnObjective { n: 4, f: quad };
    let (lb, ub) = (vec![-1.0; 4], vec![2.0; 4]);
    let o = simpsa_quiet();
    let r = simpsa(&obj, &[0.0; 4], &lb, &ub, &o).unwrap();
    assert!(r.x.iter().zip(&lb).zip(&ub).all(|((v, l), u)| v >= l && v <= u));
    assert!((r.x[1] + 1.0).abs() < 1e-2 && (r.x[3] - 2.0).abs() < 1e-2, "{:?}", r.x);
    let r2 = simpsa(&obj, &[0.0; 4], &lb, &ub, &SimpsaOptions { parallel: false, ..o.clone() }).unwrap();
    assert_eq!(r.x, r2.x, "parallel evaluation must not change the result");
    assert_eq!((r.func_count, r.iterations), (r2.func_count, r2.iterations));
    let r3 = simpsa(&obj, &[0.0; 4], &lb, &ub, &SimpsaOptions { seed: 3, ..o.clone() }).unwrap();
    assert_ne!(r.x, r3.x);
    let mut o2 = o.clone();
    o2.apply_optim(&[
        ("MaxIter".into(), OptValue::Num(20.0)),
        ("TolX".into(), OptValue::Num(-1.0)),
        ("EndTemperature".into(), OptValue::Num(0.5)),
        ("MaxFunEvals".into(), OptValue::Str("100".into())),
    ])
    .unwrap();
    assert_eq!((o2.max_iter_total, o2.tolx, o2.temp_end, o2.max_fun_evals), (20, None, 0.5, 100));
    let r = simpsa(&obj, &[0.0; 4], &lb, &ub, &o2).unwrap();
    assert_eq!(r.exitflag, 0, "{}", r.message);
    assert_eq!(r.iterations, 80); // MAX_ITER_TOTAL*NDIM
}

// ---------------------------------------------------------------------------
// SW07 (ignored)
// ---------------------------------------------------------------------------

/// The SW07 STAGE-9 mode: csminwel from the init vector and ref.json points[0] (Dynare's newrat
/// mode) both give 349.856072; the posterior is flat along `constelab`, where SA goes ~7e-6 lower.
const SW07_MODE_FVAL: f64 = 349.856072;

/// Starts to run (`SW07_START=init|refjson` = ref.json xparam_mode, default both).
fn sw07_starts<'a>(x0: &'a [f64], xmode: &'a [f64]) -> Vec<(&'static str, &'a [f64])> {
    let want = std::env::var("SW07_START").ok();
    [("init", x0), ("refjson", xmode)].into_iter().filter(|(n, _)| want.as_deref().is_none_or(|w| w == *n)).collect()
}

/// Parameters within 1e-6 (relative to the range) of a bound, and the largest moves from the mode.
fn report_point(x: &[f64], xmode: &[f64], lb: &[f64], ub: &[f64]) {
    for i in 0..x.len() {
        let r = (ub[i] - lb[i]).min(1e3);
        if x[i] - lb[i] < 1e-6 * r || ub[i] - x[i] < 1e-6 * r {
            println!("  parameter {} at a bound: {} in [{}, {}] (mode {})", i + 1, x[i], lb[i], ub[i], xmode[i]);
        }
    }
    let mut d: Vec<(usize, f64)> =
        (0..x.len()).map(|i| (i, (x[i] - xmode[i]).abs() / xmode[i].abs().max(1e-3))).collect();
    d.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (i, rel) in d.iter().take(5) {
        println!("  parameter {}: {} (mode {}, relative change {rel:.3})", i + 1, x[*i], xmode[*i]);
    }
}

fn with_sw07(f: impl FnOnce(&dyn Objective, &[f64], &[f64], &[f64], &[f64])) {
    use dynare::data::Table;
    use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
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
    let refj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("ref.json")).unwrap()).unwrap();
    let xmode: Vec<f64> = refj["xparam_mode"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let (lb, ub) = finite_bounds(&m.bounds, HUGE_NUMBER);
    println!("SW07: {} parameters, f(init) {:.6}, f(mode) {:.6}", x0.len(), post.eval(&x0).0, post.eval(&xmode).0);
    f(&post, &x0, &xmode, &lb, &ub);
}

#[test]
#[ignore]
fn sw07_mode_compute_2() {
    with_sw07(|post, x0, xmode, lb, ub| {
        let o = SaOptions { verbosity: 0, ..Default::default() };
        for (name, start) in sw07_starts(x0, xmode) {
            let t = std::time::Instant::now();
            let r = simulated_annealing(post, start, lb, ub, &o).unwrap();
            println!(
                "SW07 SA from {name}: fval {:.6} (mode {SW07_MODE_FVAL:.6}), {} evaluations, exit {} ({}), {:.1}s",
                r.fval,
                r.func_count,
                r.exitflag,
                r.message,
                t.elapsed().as_secs_f64()
            );
            report_point(&r.x, xmode, lb, ub);
            // Measured (seed 0): 349.856065 from both starts within the 1e5 evaluations; SIMPSA
            // (mode_compute=10) stays at 349.856072 from the mode and stops at 366.58 from init.
            assert!(r.fval < SW07_MODE_FVAL + 1e-3, "{}", r.fval);
            assert!(r.fval <= post.eval(start).0);
        }
    });
}

#[test]
#[ignore]
fn sw07_mode_compute_10() {
    with_sw07(|post, x0, xmode, lb, ub| {
        let o = SimpsaOptions { display: false, ..Default::default() };
        for (name, start) in sw07_starts(x0, xmode) {
            let t = std::time::Instant::now();
            let r = simpsa(post, start, lb, ub, &o).unwrap();
            println!(
                "SW07 SIMPSA from {name}: fval {:.6} (mode {SW07_MODE_FVAL:.6}), {} evaluations, {} iterations, exit {} ({}), {:.1}s",
                r.fval,
                r.func_count,
                r.iterations,
                r.exitflag,
                r.message,
                t.elapsed().as_secs_f64()
            );
            report_point(&r.x, xmode, lb, ub);
            assert!(r.fval <= post.eval(start).0);
        }
    });
}
