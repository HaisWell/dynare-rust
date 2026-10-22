//! Multi-thread scaling of the SW07 posterior evaluation (what parallel MCMC
//! chains do): N threads each evaluate `DsgePosterior::eval` at the mode in a
//! loop; prints the per-evaluation time and the total throughput.
//!
//! `cargo test --release --test mcmc_scaling -- --ignored --nocapture`
//!
//! Environment knobs:
//! - `BENCH_THREADS=1,2,4` thread counts (default 1,2,4,6,8,12,18)
//! - `BENCH_EVALS=300` evaluations per thread
//! - `BENCH_QOS=0|ui|initiated|default|utility|bg` QoS class of each thread
//!   (default `ui` = QOS_CLASS_USER_INTERACTIVE; 0 = leave unset)
//! - `DYNARE_BLAS=accelerate|rust` small-matrix BLAS backend (see `dynare::linalg`)
//! - `VECLIB_MAXIMUM_THREADS=1` Accelerate's own threading
mod common;

use std::path::Path;
use std::sync::Barrier;
use std::time::Instant;

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::Objective;
use dynare::kalman::KalmanOptions;
use dynare::modfile::parse_mod_file;
use serde_json::Value;

fn qos_from_env() -> Option<dynare::parallel::Qos> {
    match std::env::var("BENCH_QOS") {
        Ok(s) if s == "0" => None,
        Ok(s) if s == "1" => Some(dynare::parallel::Qos::UserInteractive),
        Ok(s) => Some(dynare::parallel::Qos::parse(&s).expect("BENCH_QOS")),
        Err(_) => Some(dynare::parallel::Qos::UserInteractive),
    }
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

#[test]
#[ignore]
fn sw07_eval_thread_scaling() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let r: Value = common::json_ref(dir.join("ref.json"));
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let xm = vecf(&r["xparam_mode"]);
    let m = DsgeModel::new(&mf, Some(&xm)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let data = LikelihoodData::new(
        &table.sample(&mf.varobs, 5, None).unwrap(),
        KalmanOptions { presample: 8, lik_init: 3, ..Default::default() },
    );
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &m.bounds };
    let f0 = post.eval(&xm).0;

    let env = |k: &str| std::env::var(k).ok();
    let threads: Vec<usize> = env("BENCH_THREADS")
        .map(|s| s.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 6, 8, 12, 18]);
    let evals: usize = env("BENCH_EVALS").map_or(300, |s| s.parse().unwrap());
    let qos = qos_from_env();
    println!(
        "blas backend {:?}, QoS {qos:?}, VECLIB_MAXIMUM_THREADS={:?}, {evals} evals/thread",
        dynare::linalg::blas_backend(),
        env("VECLIB_MAXIMUM_THREADS")
    );
    println!("{:>7} {:>12} {:>12} {:>14} {:>9}", "threads", "ms/eval avg", "ms/eval max", "evals/s total", "speedup");
    let mut base = 0.0;
    for &n in &threads {
        let barrier = Barrier::new(n);
        let per: Vec<f64> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..n)
                .map(|_| {
                    let (post, xm, barrier) = (&post, &xm, &barrier);
                    sc.spawn(move || {
                        if let Some(q) = qos {
                            dynare::parallel::set_current_thread_qos(q);
                        }
                        for _ in 0..20 {
                            std::hint::black_box(post.eval(xm));
                        }
                        barrier.wait();
                        let t = Instant::now();
                        for _ in 0..evals {
                            let f = post.eval(std::hint::black_box(xm)).0;
                            assert_eq!(f.to_bits(), f0.to_bits(), "result depends on the thread");
                        }
                        t.elapsed().as_secs_f64() * 1e3 / evals as f64
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let avg = per.iter().sum::<f64>() / n as f64;
        let max = per.iter().cloned().fold(0.0, f64::max);
        let thr = per.iter().map(|ms| 1e3 / ms).sum::<f64>();
        if n == threads[0] {
            base = thr / n as f64;
        }
        println!("{n:>7} {avg:>12.3} {max:>12.3} {thr:>14.0} {:>9.2}", thr / base);
    }
}

/// The same scaling for bare `linalg::gemm_raw` products (n x n times n x n),
/// isolating the BLAS backend: Accelerate's matrix unit is shared per core
/// cluster, the Rust kernel runs on each core's own NEON units.
/// `BENCH_GEMM=m,n,k` shape (default the Kalman filter's 74,74,60: P <- T W'),
/// `BENCH_THREADS`, `BENCH_QOS`, `DYNARE_BLAS` as above.
#[test]
#[ignore]
fn gemm_thread_scaling() {
    let env = |k: &str| std::env::var(k).ok();
    let threads: Vec<usize> = env("BENCH_THREADS")
        .map(|s| s.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 6, 8, 12, 18]);
    let dims: Vec<usize> =
        env("BENCH_GEMM").map_or(vec![74, 74, 60], |s| s.split(',').map(|t| t.trim().parse().unwrap()).collect());
    let (m, n, k) = (dims[0], dims[1], dims[2]);
    let qos = qos_from_env();
    let flops = 2.0 * (m * n * k) as f64;
    let reps = (2e9 / flops).ceil() as usize; // ~2 GFlop per thread
    println!(
        "gemm m={m} n={n} k={k}, blas backend {:?}, QoS {qos:?}, {reps} products/thread",
        dynare::linalg::blas_backend()
    );
    println!("{:>7} {:>12} {:>14} {:>9}", "threads", "us/gemm avg", "GFlop/s total", "speedup");
    let mut base = 0.0;
    for &nt in &threads {
        let barrier = Barrier::new(nt);
        let per: Vec<f64> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..nt)
                .map(|_| {
                    let barrier = &barrier;
                    sc.spawn(move || {
                        if let Some(q) = qos {
                            dynare::parallel::set_current_thread_qos(q);
                        }
                        let a: Vec<f64> = (0..m * k).map(|i| ((i * 7919) % 101) as f64 / 101.0).collect();
                        let b: Vec<f64> = (0..n * k).map(|i| ((i * 104729) % 97) as f64 / 97.0).collect();
                        let mut c = vec![0.0; m * n];
                        barrier.wait();
                        let t = Instant::now();
                        for _ in 0..reps {
                            dynare::linalg::gemm_raw(false, true, m, n, k, 1.0, &a, m, &b, n, 0.0, &mut c, m);
                            std::hint::black_box(&mut c);
                        }
                        t.elapsed().as_secs_f64() * 1e6 / reps as f64
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let avg = per.iter().sum::<f64>() / nt as f64;
        let gf = per.iter().map(|us| flops / us / 1e3).sum::<f64>();
        if nt == threads[0] {
            base = gf / nt as f64;
        }
        println!("{nt:>7} {avg:>12.2} {gf:>14.1} {:>9.2}", gf / base);
    }
}

/// Which part of the posterior evaluation slows down when other chains run:
/// times the components on this thread while `BENCH_LOAD` (default 5) other
/// threads evaluate the posterior in a loop (0 = idle machine).
#[test]
#[ignore]
fn sw07_components_under_load() {
    use dynare::kalman::{KalmanWorkspace, try_compute_pinf_pstar};
    use dynare::solver::kalman_transition_matrix;
    use std::sync::atomic::{AtomicBool, Ordering};
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let r: Value = common::json_ref(dir.join("ref.json"));
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let xm = vecf(&r["xparam_mode"]);
    let m = DsgeModel::new(&mf, Some(&xm)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let kopts = KalmanOptions { presample: 8, lik_init: 3, ..Default::default() };
    let data = LikelihoodData::new(&table.sample(&mf.varobs, 5, None).unwrap(), kopts.clone());
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &m.bounds };
    let idx = &m.ss_index;
    let (params, q) = m.set_all_parameters(&xm);
    let ys = m.steady_state(&params).unwrap();
    let x_ss = vec![0.0; m.model.n_exo()];
    let g1 = m.model.dynamic_jacobian(&ys, &x_ss, &params);
    let res = m.resolve(&xm).unwrap();
    let (t, rr) = kalman_transition_matrix(&res.dr, idx);
    let (pstar, pinf) =
        try_compute_pinf_pstar(&idx.mf, &t, &rr, &q, kopts.qz_criterium, &idx.restrict_columns).unwrap();
    let mut y = data.y.clone();
    for (i, &v) in idx.mfys.iter().enumerate() {
        for j in 0..y.ncols() {
            y[(i, j)] -= ys[v];
        }
    }
    let load: usize = std::env::var("BENCH_LOAD").map_or(5, |s| s.parse().unwrap());
    let stop = AtomicBool::new(false);
    dynare::parallel::set_current_thread_high_qos();
    std::thread::scope(|sc| {
        for _ in 0..load {
            sc.spawn(|| {
                dynare::parallel::set_current_thread_high_qos();
                while !stop.load(Ordering::Relaxed) {
                    std::hint::black_box(post.eval(&xm));
                }
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reps = 300;
        let time = |f: &mut dyn FnMut()| {
            f();
            let t0 = Instant::now();
            for _ in 0..reps {
                f();
            }
            t0.elapsed().as_secs_f64() * 1e6 / reps as f64
        };
        let mut ws = KalmanWorkspace::new(t.nrows(), y.nrows(), q.nrows());
        let solver = dynare::solver::FirstOrderSolver::new(
            &m.model.lead_lag_incidence,
            m.model.n_exo(),
            dynare::solver::SolverOptions::default(),
        );
        let rows = [
            ("first-order solve", time(&mut || drop(std::hint::black_box(solver.solve(&g1, &ys))))),
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
                    std::hint::black_box(post.eval(&xm));
                }),
            ),
        ];
        stop.store(true, Ordering::Relaxed);
        println!("{} other threads evaluating; blas backend {:?}", load, dynare::linalg::blas_backend());
        for (name, us) in rows {
            println!("{name:<32} {us:>9.1} us");
        }
    });
}
