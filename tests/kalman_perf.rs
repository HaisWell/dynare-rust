//! Performance probes for the Kalman part of the SW07 posterior (reference/sw07, all `#[ignore]`):
//!
//! * `kalman_breakdown`: cost of each filter stage and of compute_Pinf_Pstar at the mode;
//! * `gemm_sizes`, `gemm_backends_threads`: BLAS calls of the filter sizes;
//! * `t_structure`: sparsity of the restricted transition matrix.
//!
//! The per-evaluation time with 1 and 6 concurrent threads is `public_models_perf.rs`.
//!
//! `cargo test --release --test kalman_perf -- --ignored --nocapture`
mod common;

use std::path::Path;

use dynare::kalman::{self, HMatrix, KalmanOptions, KalmanWorkspace, MissingData};
use dynare::linalg::Mat;
use serde_json::Value;

fn load(name: &str) -> Option<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07").join(name);
    Some(common::json_ref(p))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn mat(v: &Value) -> Mat {
    let rows = v.as_array().unwrap();
    if rows[0].is_array() {
        let nc = rows[0].as_array().unwrap().len();
        Mat::from_fn(rows.len(), nc, |i, j| num(&rows[i][j]))
    } else {
        Mat::from_fn(rows.len(), 1, |i, _| num(&rows[i]))
    }
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        other => vec![num(other)],
    }
}

fn idx0(v: &Value) -> Vec<usize> {
    vecf(v).iter().map(|&x| x as usize - 1).collect()
}

/// Median over 7 batches of `reps` calls, in microseconds per call.
fn time(reps: usize, f: &mut dyn FnMut()) -> f64 {
    for _ in 0..reps.min(20) {
        f();
    }
    let mut v: Vec<f64> = (0..7)
        .map(|_| {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_secs_f64() * 1e6 / reps as f64
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[3]
}

#[test]
#[ignore]
fn kalman_breakdown() {
    use dynare::data::Table;
    use dynare::dsge::{DsgeModel, LikelihoodData};
    use dynare::modfile::parse_mod_file;
    let r = load("ref.json").expect("reference/setup unavailable");
    // the state space at the mode as DsgePosterior::evaluate builds it
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let modf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let xm = vecf(&r["xparam_mode"]);
    let m = DsgeModel::new(&modf, Some(&xm)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let opts = KalmanOptions { presample: 8, lik_init: 3, ..Default::default() };
    let data = LikelihoodData::new(&table.sample(&modf.varobs, 5, None).unwrap(), opts.clone());
    let res = m.resolve(&xm).unwrap();
    let (t, rm, q) = (res.t.clone(), res.r.clone(), res.q.clone());
    let mut y = data.y.clone();
    for (i, &v) in m.ss_index.mfys.iter().enumerate() {
        for j in 0..y.ncols() {
            y[(i, j)] -= res.steady_state[v];
        }
    }
    let md = MissingData::from_data(&y);
    let (mf, rc) = (m.ss_index.mf.clone(), m.ss_index.restrict_columns.clone());
    let (pstar, pinf) = kalman::compute_pinf_pstar(&mf, &t, &rm, &q, opts.qz_criterium, &rc);
    let (mm, pp, rr) = (t.nrows(), y.nrows(), q.nrows());
    let nnz = (0..mm).filter(|&j| t.column(j).iter().any(|&x| x != 0.0)).count();
    println!("mm {mm} pp {pp} rr {rr} nnz {nnz} restrict_columns {} nobs {}", rc.len(), y.ncols());
    let mut ws = KalmanWorkspace::new(mm, pp, rr);
    let out = ws.likelihood(&y, &md, &mf, &t, &rm, &q, None, &pstar, &pinf, &opts);
    println!(
        "minus_loglik {:.12} diffuse periods {} univariate {}",
        out.minus_loglik, out.diffuse_periods, out.univariate
    );
    let h = HMatrix::Zero;
    let zeros = vec![0.0; mm];
    let last = y.ncols() - 1;
    // state after the diffuse stage (start of the regular filter)
    ws.set_state(&zeros, &pstar, Some(&pinf));
    ws.diffuse_univariate(&y, None, &mf, &vec![0.0; pp], 0, last, 1e-10, 1e-6, 8);
    let (a1, p1) = (ws.state(), ws.cov());
    let reps = 300;
    let rows = [
        (
            "compute_Pinf_Pstar",
            time(100, &mut || {
                std::hint::black_box(kalman::compute_pinf_pstar(&mf, &t, &rm, &q, opts.qz_criterium, &rc));
            }),
        ),
        ("set_system", time(reps, &mut || ws.set_system(&t, &rm, &q))),
        ("set_state", time(reps, &mut || ws.set_state(&zeros, &pstar, Some(&pinf)))),
        (
            "kalman_filter_d (fails)",
            time(reps, &mut || {
                ws.set_state(&zeros, &pstar, Some(&pinf));
                std::hint::black_box(ws.diffuse_multivariate(&y, None, &mf, &h, 0, last, 1e-10, 1e-6, 8));
            }),
        ),
        (
            "univariate_kalman_filter_d",
            time(reps, &mut || {
                ws.set_state(&zeros, &pstar, Some(&pinf));
                std::hint::black_box(ws.diffuse_univariate(&y, None, &mf, &vec![0.0; pp], 0, last, 1e-10, 1e-6, 8));
            }),
        ),
        (
            "kalman_filter (regular periods)",
            time(reps, &mut || {
                ws.set_state(&a1, &p1, None);
                std::hint::black_box(ws.multivariate(&y, None, &mf, &h, 1, last, 1e-10, 1e-6, false, 8, 1));
            }),
        ),
        (
            "likelihood (total)",
            time(reps, &mut || {
                std::hint::black_box(ws.likelihood(&y, &md, &mf, &t, &rm, &q, None, &pstar, &pinf, &opts));
            }),
        ),
    ];
    println!("{:<34} {:>9}", "step", "us/call");
    for (name, us) in rows {
        println!("{name:<34} {us:>9.1}");
    }
}

#[test]
#[ignore]
fn gemm_sizes() {
    use dynare::linalg::gemm_raw;
    let a: Vec<f64> = (0..74 * 74).map(|i| (i as f64 * 0.37).sin()).collect();
    let b = a.clone();
    let mut c = vec![0.0; 74 * 74];
    for (ta, tb, m, n, k) in [
        (false, false, 74, 60, 60),
        (false, true, 74, 74, 60),
        (false, true, 60, 60, 22),
        (false, false, 74, 22, 22),
        (false, false, 22, 22, 22),
        (false, false, 4, 4, 4),
        (false, false, 64, 60, 60),
        (false, false, 63, 60, 60),
        (false, false, 64, 64, 64),
        (false, true, 74, 64, 60),
        (false, true, 74, 63, 60),
        (false, true, 74, 64, 64),
        (false, false, 74, 24, 24),
        (false, true, 60, 60, 22),
    ] {
        let us = time(2000, &mut || gemm_raw(ta, tb, m, n, k, 1.0, &a, 74, &b, 74, 1.0, &mut c, 74));
        let gf = (m * n * k) as f64 * 2.0 / us / 1e3;
        println!("gemm {m}x{n}x{k} ({ta},{tb}): {us:.2} us, {gf:.1} GFlop/s");
    }
    // the same gemm when the CPU has just written its operands (as in the filter loop)
    let mut b2 = b.clone();
    let us = time(2000, &mut || {
        for x in b2.iter_mut().take(74 * 22) {
            *x += 1e-300;
        }
        for x in c.iter_mut().take(74 * 60) {
            *x *= 0.5;
        }
        gemm_raw(false, true, 60, 60, 22, 1.0, &a, 74, &b2, 74, 1.0, &mut c, 74)
    });
    let us0 = time(2000, &mut || {
        for x in b2.iter_mut().take(74 * 22) {
            *x += 1e-300;
        }
        for x in c.iter_mut().take(74 * 60) {
            *x *= 0.5;
        }
    });
    println!("gemm 60x60x22 after CPU writes to B and C: {:.2} us (+ {us0:.2} us for the writes)", us - us0);
    for (wb, wc) in [(true, false), (false, true)] {
        let w0 = time(2000, &mut || {
            if wb {
                b2.iter_mut().take(74 * 22).for_each(|x| *x += 1e-300);
            }
            if wc {
                c.iter_mut().take(74 * 60).for_each(|x| *x *= 0.5);
            }
        });
        let w1 = time(2000, &mut || {
            if wb {
                b2.iter_mut().take(74 * 22).for_each(|x| *x += 1e-300);
            }
            if wc {
                c.iter_mut().take(74 * 60).for_each(|x| *x *= 0.5);
            }
            gemm_raw(false, true, 60, 60, 22, 1.0, &a, 74, &b2, 74, 1.0, &mut c, 74)
        });
        println!("  CPU writes B {wb} C {wc}: gemm {:.2} us", w1 - w0);
    }
    // C written by CPU, gemm with beta = 0 (C write-only)
    let w0 = time(2000, &mut || c.iter_mut().take(74 * 60).for_each(|x| *x *= 0.5));
    let w1 = time(2000, &mut || {
        c.iter_mut().take(74 * 60).for_each(|x| *x *= 0.5);
        gemm_raw(false, true, 60, 60, 22, 1.0, &a, 74, &b2, 74, 0.0, &mut c, 74)
    });
    println!("  CPU writes C, beta 0: gemm {:.2} us", w1 - w0);
}

/// Accelerate vs nalgebra (matrixmultiply) gemm with 1 and 6 concurrent threads.
#[test]
#[ignore]
fn gemm_backends_threads() {
    use dynare::linalg::gemm_raw;
    for threads in [1usize, 6] {
        for backend in ["accelerate", "nalgebra"] {
            let per: Vec<f64> = std::thread::scope(|s| {
                let hs: Vec<_> = (0..threads)
                    .map(|_| {
                        s.spawn(move || {
                            let a = Mat::from_fn(74, 60, |i, j| ((i * 60 + j) as f64 * 0.37).sin());
                            let b = Mat::from_fn(60, 60, |i, j| ((i * 7 + j) as f64 * 0.11).cos());
                            let mut c = Mat::zeros(74, 60);
                            time(20000, &mut || {
                                if backend == "accelerate" {
                                    gemm_raw(
                                        false,
                                        false,
                                        74,
                                        60,
                                        60,
                                        1.0,
                                        a.as_slice(),
                                        74,
                                        b.as_slice(),
                                        60,
                                        0.0,
                                        c.as_mut_slice(),
                                        74,
                                    );
                                } else {
                                    c.gemm(1.0, &a, &b, 0.0);
                                }
                                std::hint::black_box(&c);
                            })
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mean = per.iter().sum::<f64>() / per.len() as f64;
            println!(
                "{threads} threads {backend:<10} gemm 74x60x60: {mean:.2} us ({:.1} GFlop/s per thread)",
                74.0 * 3600.0 * 2.0 / mean / 1e3
            );
        }
    }
}

#[test]
#[ignore]
fn t_structure() {
    let r = load("ref.json").expect("reference/setup unavailable");
    let ss = &r["mode_ss"];
    let (t, rm) = (mat(&ss["T"]), mat(&ss["R"]));
    let mf = idx0(&r["bayestopt"]["mf1"]);
    let mm = t.nrows();
    let nzc: Vec<usize> = (0..mm).filter(|&j| t.column(j).iter().any(|&x| x != 0.0)).collect();
    let mut unit = 0;
    let mut zero = 0;
    let mut nnz_tot = 0;
    for i in 0..mm {
        let nz: Vec<usize> = (0..mm).filter(|&j| t[(i, j)] != 0.0).collect();
        nnz_tot += nz.len();
        let rz = (0..rm.ncols()).filter(|&j| rm[(i, j)] != 0.0).count();
        if nz.is_empty() {
            zero += 1;
        }
        if nz.len() == 1 && t[(i, nz[0])] == 1.0 {
            unit += 1;
        }
        let small: Vec<(usize, f64)> =
            if nz.len() <= 3 { nz.iter().map(|&j| (j, t[(i, j)])).collect() } else { vec![] };
        println!("row {i:2}: nnz {:2} R nnz {rz:2} obs {} {small:?}", nz.len(), mf.contains(&i));
    }
    println!(
        "nonzero cols {} unit rows {unit} zero rows {zero} density {:.2}",
        nzc.len(),
        nnz_tot as f64 / (mm * mm) as f64
    );
    let rnz = (0..mm).filter(|&i| (0..rm.ncols()).any(|j| rm[(i, j)] != 0.0)).count();
    println!("rows of R non-zero: {rnz}");
    let zc: Vec<usize> = (0..mm).filter(|j| !nzc.contains(j)).collect();
    let zobs = zc.iter().filter(|j| mf.contains(j)).count();
    println!("zero cols {zc:?}, observed among them {zobs}; mf {mf:?}");
    let sparse: Vec<usize> = (0..mm).filter(|&i| (0..mm).filter(|&j| t[(i, j)] != 0.0).count() <= 15).collect();
    let sp_nnz: usize = sparse.iter().map(|&i| (0..mm).filter(|&j| t[(i, j)] != 0.0).count()).sum();
    println!("rows with <= 15 nnz: {} (total nnz {sp_nnz})", sparse.len());
}
