//! Speed probe for the benchmark model (reference/sw07, docs/REFERENCES.md): per-evaluation
//! time of `DsgePosterior::eval` at the mode with 1 and 6 threads evaluating concurrently (MCMC
//! chains share the AMX units of a cluster), and the value against Dynare's (ref.json, point 0).
//! The cost of each step is `kalman_perf.rs kalman_breakdown` and `dsge_sw07.rs`.
//!
//! `cargo test --release --test public_models_perf -- --ignored --nocapture`
mod common;

use std::path::Path;

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::Objective;
use dynare::kalman::KalmanOptions;
use dynare::modfile::parse_mod_file;

#[test]
#[ignore]
fn sw07_posterior_threads() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let r: serde_json::Value = common::json_ref(dir.join("ref.json"));
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let xm: Vec<f64> = r["xparam_mode"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let m = DsgeModel::new(&mf, Some(&xm)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let kopts = KalmanOptions { presample: 8, lik_init: 3, ..Default::default() };
    let data = LikelihoodData::new(&table.sample(&mf.varobs, 5, None).unwrap(), kopts);
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &m.bounds };
    let f0 = post.eval(&xm).0;
    let want = r["points"][0]["minus_loglik"].as_f64().unwrap();
    println!("fval at the mode {f0:.10} (Dynare {want:.10}, rel {:.1e})", ((f0 - want) / want).abs());
    let n = 2000;
    for threads in [1usize, 6] {
        let per: Vec<f64> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        for _ in 0..50 {
                            assert_eq!(post.eval(&xm).0, f0);
                        }
                        let t = std::time::Instant::now();
                        for _ in 0..n {
                            std::hint::black_box(post.eval(&xm));
                        }
                        t.elapsed().as_secs_f64() * 1e3 / n as f64
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mean = per.iter().sum::<f64>() / per.len() as f64;
        println!(
            "{threads} thread(s): {mean:.3} ms per evaluation per thread ({:.0} evals/s total)",
            threads as f64 * 1e3 / mean
        );
    }
}
