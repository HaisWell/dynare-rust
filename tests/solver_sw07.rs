//! Solver module vs Dynare 7.2 on sw07.mod (reference/sw07/ref.json and
//! reference/sw07/ref_solver.json, the latter written by reference/sw07/dump_solver.m).

mod common;

use dynare::linalg::Mat;
use dynare::solver::{
    FirstOrderSolver, SolverOptions, SteadyOptions, kalman_transition_matrix, state_space_index, steady_state,
};
use serde_json::Value;
use std::path::Path;

/// SW07: 50 endogenous variables (45 declared + 5 auxiliary), 11 shocks, 24 state variables.
const N: usize = 50;
const NX: usize = 11;
const NSPRED: usize = 24;

fn load(name: &str) -> Option<Value> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07").join(name);
    Some(common::json_ref(&p))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::INFINITY) // jsonencode(..., 'ConvertInfAndNaN') writes Inf as null
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(num).collect(),
        x => vec![num(x)],
    }
}

fn veci(v: &Value) -> Vec<usize> {
    vecf(v).into_iter().map(|x| x as usize - 1).collect()
}

/// MATLAB matrix from jsonencode (array of rows); scalars / vectors handled for shape (r, c).
fn mat(v: &Value, r: usize, c: usize) -> Mat {
    let mut m = Mat::zeros(r, c);
    match v {
        Value::Array(rows) if rows.first().is_some_and(|x| x.is_array()) => {
            for (i, row) in rows.iter().enumerate() {
                for (j, x) in row.as_array().unwrap().iter().enumerate() {
                    m[(i, j)] = num(x);
                }
            }
        }
        other => {
            let f = vecf(other);
            assert_eq!(f.len(), r * c);
            if r == 1 || c == 1 {
                m.copy_from_slice(&f);
            } else {
                panic!("unexpected matrix encoding");
            }
        }
    }
    m
}

fn sparse(v: &Value, r: usize, c: usize) -> Mat {
    let (i, j, x) = (vecf(&v["i"]), vecf(&v["j"]), vecf(&v["v"]));
    let mut m = Mat::zeros(r, c);
    for k in 0..x.len() {
        m[(i[k] as usize - 1, j[k] as usize - 1)] = x[k];
    }
    m
}

fn lli(r: &Value) -> Vec<[usize; 3]> {
    let rows: Vec<Vec<f64>> = r["lead_lag_incidence"].as_array().unwrap().iter().map(vecf).collect();
    (0..rows[0].len()).map(|i| [rows[0][i] as usize, rows[1][i] as usize, rows[2][i] as usize]).collect()
}

fn max_diff(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.shape(), b.shape());
    (a - b).amax()
}

fn varobs(r: &Value) -> Vec<usize> {
    let names: Vec<&str> = r["endo_names"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    r["varobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| names.iter().position(|n| *n == v.as_str().unwrap()).unwrap())
        .collect()
}

fn solver_opts(r: &Value) -> SolverOptions {
    SolverOptions {
        qz_criterium: r["options"]["qz_criterium"].as_f64().unwrap(),
        qz_zero_threshold: r["options"]["qz_zero_threshold"].as_f64().unwrap(),
        ..Default::default()
    }
}

#[test]
fn dr_ordering_matches_dynare() {
    let r = load("ref.json").expect("reference/setup unavailable");
    let s = FirstOrderSolver::new(&lli(&r), NX, solver_opts(&r)).structure;
    let d = &r["mode_dr"];
    assert_eq!(s.order_var, veci(&d["order_var"]));
    assert_eq!(s.inv_order_var, veci(&d["inv_order_var"]));
    assert_eq!(s.state_var, veci(&d["state_var"]));
    assert_eq!(s.nstatic, d["nstatic"].as_u64().unwrap() as usize);
    assert_eq!(s.npred, d["npred"].as_u64().unwrap() as usize);
    assert_eq!(s.nboth, d["nboth"].as_u64().unwrap() as usize);
    assert_eq!(s.nfwrd, d["nfwrd"].as_u64().unwrap() as usize);
}

#[test]
fn decision_rule_and_state_space_at_mode() {
    let r = load("ref.json").expect("reference/setup unavailable");
    let (n, nx) = (N, NX);
    let g1 = mat(&r["mode_g1"], n, 3 * n + nx);
    let d = &r["mode_dr"];
    let ys = vecf(&d["ys"]);
    let solver = FirstOrderSolver::new(&lli(&r), nx, solver_opts(&r));
    let dr = solver.solve(&g1, &ys).expect("solve at mode");
    let ghx_ref = mat(&d["ghx"], n, NSPRED);
    let ghu_ref = mat(&d["ghu"], n, nx);
    let ex = max_diff(&dr.ghx, &ghx_ref);
    let eu = max_diff(&dr.ghu, &ghu_ref);
    println!("mode: max|ghx - ref| = {ex:.3e}, max|ghu - ref| = {eu:.3e}");
    assert!(ex < 1e-10 && eu < 1e-10);

    // eigenvalue moduli (sorted; Inf encoded as null)
    let mut a = dr.eigval_abs.clone();
    let mut b = vecf(&d["eigval_abs"]);
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    assert_eq!(a.len(), b.len());
    // Same number of stable roots. Defective zero / infinite eigenvalues come out
    // as clusters whose moduli are rounding noise (Dynare: ~5e-3, 28, 158, ...),
    // so only the well-conditioned moduli in [0.05, 1] are compared.
    let crit = r["options"]["qz_criterium"].as_f64().unwrap();
    assert_eq!(a.iter().filter(|&&x| x < crit).count(), b.iter().filter(|&&x| x < crit).count());
    let mid = |v: &[f64]| v.iter().copied().filter(|&x| (0.05..crit).contains(&x)).collect::<Vec<_>>();
    let (am, bm) = (mid(&a), mid(&b));
    assert_eq!(am.len(), bm.len());
    let ee = am.iter().zip(&bm).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    println!("mode: max|eig - ref| over the {} moduli in [0.05, 1] = {ee:.3e}", am.len());
    assert!(ee < 1e-10);

    // restricted state space
    let obs = varobs(&r);
    let idx = state_space_index(&dr, &obs);
    assert_eq!(idx.restrict_var_list, veci(&d["restrict_var_list"]));
    assert_eq!(idx.restrict_columns, veci(&d["restrict_columns"]));
    let bt = &r["bayestopt"];
    assert_eq!(idx.mf, veci(&bt["mf1"]));
    assert_eq!(idx.mf0, veci(&bt["mf0"]));
    assert_eq!(idx.mfys, veci(&bt["mfys"]));
    assert_eq!(idx.smoother_var_list, veci(&bt["smoother_var_list"]));
    assert_eq!(idx.smoother_mf, veci(&bt["smoother_mf"]));
    let (t, rr) = kalman_transition_matrix(&dr, &idx);
    let m = idx.restrict_var_list.len();
    let et = max_diff(&t, &mat(&r["mode_ss"]["T"], m, m));
    let er = max_diff(&rr, &mat(&r["mode_ss"]["R"], m, nx));
    println!("mode: max|T - ref| = {et:.3e}, max|R - ref| = {er:.3e}");
    assert!(et < 1e-10 && er < 1e-10);
}

/// Steady state (solve_algo = 1 from both initial guesses) and decision rules
/// at the parameter points of ref_solver.json.
#[test]
fn steady_state_and_solution_at_points() {
    let (r, rs) = (
        load("ref.json").expect("reference/setup unavailable"),
        load("ref_solver.json").expect("reference/setup unavailable"),
    );
    let (n, nx) = (N, NX);
    let o = &rs["options"];
    let sopts = SteadyOptions {
        solve_algo: o["solve_algo"].as_u64().unwrap() as u8,
        maxit: o["steady_maxit"].as_u64().unwrap() as usize,
        tolf: o["solve_tolf"].as_f64().unwrap(),
        tolx: o["solve_tolx"].as_f64().unwrap(),
        linear: o["linear"].as_bool().unwrap(),
        ..Default::default()
    };
    let solver = FirstOrderSolver::new(&lli(&r), nx, solver_opts(&r));
    let obs = varobs(&r);
    let (mut worst_ys, mut worst_dr) = (0.0f64, 0.0f64);
    for c in rs["cases"].as_array().unwrap() {
        let name = format!("{}/{}", c["name"].as_str().unwrap(), c["guess"].as_str().unwrap());
        let guess = vecf(&c["guess_aux"]);
        let r0 = vecf(&c["static_resid"]);
        let j0 = sparse(&c["static_g1"], n, n);
        // the static model is linear in y: f(y) = r0 + J (y - guess)
        let f = |y: &[f64]| -> (Vec<f64>, Mat) {
            let dy = Mat::from_iterator(n, 1, y.iter().zip(&guess).map(|(a, b)| a - b));
            let jd = &j0 * dy;
            ((0..n).map(|i| r0[i] + jd[i]).collect(), j0.clone())
        };
        let ys = steady_state(&f, &guess, &sopts).unwrap_or_else(|e| panic!("{name}: {e}"));
        let ys_ref = vecf(&c["ys"]);
        let e_ys = ys.iter().zip(&ys_ref).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("{name:28} max|ys - ref| = {e_ys:.3e}");
        assert!(e_ys < 1e-9, "{name}: steady state differs by {e_ys:e}");
        worst_ys = worst_ys.max(e_ys);

        let info = vecf(&c["info"]);
        let Some(g1v) = c.get("dynamic_g1") else { continue };
        let g1 = sparse(g1v, n, 3 * n + nx);
        match solver.solve(&g1, &ys_ref) {
            Ok(dr) => {
                assert_eq!(info[0], 0.0, "{name}: Dynare returned info {info:?}");
                let ex = max_diff(&dr.ghx, &mat(&c["ghx"], n, NSPRED));
                let eu = max_diff(&dr.ghu, &mat(&c["ghu"], n, nx));
                let idx = state_space_index(&dr, &obs);
                let (t, rr) = kalman_transition_matrix(&dr, &idx);
                let m = idx.restrict_var_list.len();
                let et = max_diff(&t, &mat(&c["T"], m, m));
                let er = max_diff(&rr, &mat(&c["R"], m, nx));
                println!("{name:28} ghx {ex:.2e} ghu {eu:.2e} T {et:.2e} R {er:.2e}");
                let e = ex.max(eu).max(et).max(er);
                assert!(e < 1e-10, "{name}: decision rule differs by {e:e}");
                worst_dr = worst_dr.max(e);
            }
            Err(e) => {
                println!("{name:28} info {} info2 {} (Dynare {info:?})", e.info, e.info2);
                assert_eq!(e.info as f64, info[0], "{name}");
                assert!((e.info2 - info[1]).abs() < 1e-10 * (1.0 + info[1].abs()), "{name}: penalty {e:?} vs {info:?}");
            }
        }
    }
    println!("worst steady state error {worst_ys:.3e}, worst decision rule error {worst_dr:.3e}");
}

/// The structured QZ path (`FirstOrderSolver::solve`) against Dynare's dense
/// path (`solve_full_qz`) at the mode and at every ref_solver.json case: it must
/// apply without falling back, and give the same decision rule, the same info
/// code and the same info(2) penalty.
#[test]
fn structured_path_matches_full_qz() {
    let r = load("ref.json").expect("reference/setup unavailable");
    let (n, nx) = (N, NX);
    let solver = FirstOrderSolver::new(&lli(&r), nx, solver_opts(&r));
    let mut cases = vec![("mode".to_string(), mat(&r["mode_g1"], n, 3 * n + nx), vecf(&r["mode_dr"]["ys"]))];
    if let Some(rs) = load("ref_solver.json") {
        for c in rs["cases"].as_array().unwrap() {
            if let Some(g1v) = c.get("dynamic_g1") {
                let name = format!("{}/{}", c["name"].as_str().unwrap(), c["guess"].as_str().unwrap());
                cases.push((name, sparse(g1v, n, 3 * n + nx), vecf(&c["ys"])));
            }
        }
    }
    let (mut worst, mut failures) = (0.0f64, 0);
    for (name, g1, ys) in &cases {
        let full = solver.solve_full_qz(g1, ys);
        let fast = solver.solve_structured_only(g1, ys);
        if let Err(e) = &fast {
            assert_ne!(e.info, dynare::solver::first_order::STRUCTURED_DECLINED, "{name}: structured path declined");
        }
        match (fast, full) {
            (Ok(a), Ok(b)) => {
                // relative to the largest coefficient (SW07: |ghx| up to ~12 at the mode)
                let scale = b.ghx.amax().max(b.ghu.amax()).max(1.0);
                let e = max_diff(&a.ghx, &b.ghx).max(max_diff(&a.ghu, &b.ghu)) / scale;
                assert!(e < 1e-12, "{name}: structured vs full QZ differ by {e:e} (relative)");
                assert_eq!(a.sdim, b.sdim, "{name}");
                worst = worst.max(e);
            }
            (Err(a), Err(b)) => {
                assert_eq!(a.info, b.info, "{name}");
                assert!((a.info2 - b.info2).abs() < 1e-12 * (1.0 + b.info2.abs()), "{name}: {a:?} vs {b:?}");
                failures += 1;
            }
            (a, b) => panic!("{name}: structured {:?} vs full {:?}", a.err(), b.err()),
        }
    }
    println!(
        "{} cases ({failures} BK failures): worst structured vs full QZ relative difference {worst:.2e}",
        cases.len()
    );
}

#[test]
#[ignore]
fn timing_first_order_sw07() {
    let r = load("ref.json").expect("reference/setup unavailable");
    let (n, nx) = (N, NX);
    let g1 = mat(&r["mode_g1"], n, 3 * n + nx);
    let ys = vecf(&r["mode_dr"]["ys"]);
    let solver = FirstOrderSolver::new(&lli(&r), nx, solver_opts(&r));
    let obs = varobs(&r);
    let dr = solver.solve(&g1, &ys).unwrap();
    let idx = state_space_index(&dr, &obs);
    let reps = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let dr = solver.solve(&g1, &ys).unwrap();
        std::hint::black_box(kalman_transition_matrix(&dr, &idx));
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    println!("first-order solve + T/R: {us:.1} us per call");
    // the QZ alone, on a dense pencil of the same size
    let nd = dr.eigval.len();
    let e = Mat::from_fn(nd, nd, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
    let d = Mat::from_fn(nd, nd, |i, j| f64::from(u8::from(i == j)) + ((i + 2 * j) % 5) as f64 * 0.01);
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        std::hint::black_box(dynare::linalg::qz::mjdgges(&e, &d, 1.000001, 1e-6));
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    println!("mjdgges {nd}x{nd} (dense test pencil): {us:.1} us per call");
}

/// No Dynare reference (the model uses solve_algo = 1): the default
/// solve_algo = 4 path (Dulmage-Mendelsohn blocks + trust region), 2 and 9
/// must cope with the structurally singular static Jacobian of the unit roots.
#[test]
fn other_solve_algos_on_singular_static_model() {
    let rs = load("ref_solver.json").expect("reference/setup unavailable");
    let n = N;
    let c = rs["cases"].as_array().unwrap().iter().find(|c| c["name"] == "ss_shift").unwrap();
    let guess = vecf(&c["guess_aux"]);
    let r0 = vecf(&c["static_resid"]);
    let j0 = sparse(&c["static_g1"], n, n);
    let f = |y: &[f64]| -> (Vec<f64>, Mat) {
        let dy = Mat::from_iterator(n, 1, y.iter().zip(&guess).map(|(a, b)| a - b));
        let jd = &j0 * dy;
        ((0..n).map(|i| r0[i] + jd[i]).collect(), j0.clone())
    };
    for algo in [2u8, 4, 9] {
        let opts = SteadyOptions { solve_algo: algo, maxit: 1000, ..Default::default() };
        match steady_state(&f, &guess, &opts) {
            Ok(ys) => {
                let res = f(&ys).0.iter().map(|x| x.abs()).fold(0.0, f64::max);
                println!("solve_algo {algo}: converged, max|resid| = {res:.2e}");
                assert!(res < opts.tolf);
            }
            Err(e) => println!("solve_algo {algo}: {e}"),
        }
    }
}
