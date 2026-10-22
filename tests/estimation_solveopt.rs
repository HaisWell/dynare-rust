//! `mode_compute = 101` (solvopt.m + apprgrdn.m) against Dynare 7.2
//! (`reference/estimation/ref_solveopt.json`, written by `dump_solveopt.m`).
//! The port reproduces MATLAB's evaluation sequence bit for bit on these problems
//! (checked by logging every objective call on both sides); the JSON holds 16
//! significant digits, hence the 1e-15 tolerances.
mod common;

use dynare::estimation::Objective;
use dynare::estimation::optim::{SolveoptOptions, solveopt};
use dynare::linalg::Mat;
use serde_json::Value;

struct Rosen {
    n: usize,
    a: f64,
}

impl Objective for Rosen {
    fn dim(&self) -> usize {
        self.n
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        // same summation order as rosen5.m: sum([r1^2; r2^2; ...])
        let mut s = 0.0;
        for i in 0..self.n - 1 {
            s += (self.a * (x[i + 1] - x[i] * x[i])).powi(2);
            s += (1.0 - x[i]).powi(2);
        }
        (s, 0)
    }
}

struct Quad {
    a: Mat,
    b: Vec<f64>,
}

impl Objective for Quad {
    fn dim(&self) -> usize {
        self.a.ncols()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let mut s = 0.0;
        for k in 0..self.a.nrows() {
            let r: f64 = (0..x.len()).map(|j| self.a[(k, j)] * x[j]).sum::<f64>() - self.b[k];
            s += 0.5 * r * r;
        }
        (s, 0)
    }
}

fn load() -> Option<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/estimation/ref_solveopt.json");
    Some(common::json_ref(path))
}

fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn quiet() -> SolveoptOptions {
    SolveoptOptions { verbosity: -1.0, ..Default::default() }
}

fn check(name: &str, obj: &dyn Objective, x0: &[f64], opts: &SolveoptOptions, r: &Value) {
    let res = solveopt(obj, x0, opts).unwrap();
    let rx = vecf(&r["x"]);
    let rf = r["f"].as_f64().unwrap();
    let rk = r["k"].as_u64().unwrap() as usize;
    let rnf = r["n_f_evals"].as_u64().unwrap() as usize;
    let ref_flag = r["exitflag"].as_i64().unwrap();
    println!(
        "{name}: rust f {:.6e} k {} nf {} flag {} | matlab f {rf:.6e} k {rk} nf {rnf} flag {ref_flag}",
        res.fval, res.iterations, res.func_count, res.exitflag
    );
    let dx = res.x.iter().zip(&rx).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0.0, f64::max);
    // Bit-for-bit on macOS arm64 (the platform the reference was produced on); elsewhere a
    // different libm can perturb the path, so only the optimum is required to agree closely:
    // in Rosenbrock's valley f ~ 1e-14 leaves x free to ~1e-7 (5e-9 on Linux and Windows),
    // and f itself, ~0 at the optimum, needs an absolute floor. A run stopped by the iteration
    // limit (exitflag < 0) ends mid-path, wherever the perturbed path has got to, so there only
    // the exit flag and message are compared.
    let (tol_x, floor_f) = if cfg!(target_os = "macos") { (1e-10, 0.0) } else { (1e-7, 1e-12) };
    if cfg!(target_os = "macos") || ref_flag > 0 {
        assert!(dx < tol_x, "{name}: x {:?} vs {rx:?}", res.x);
        assert!((res.fval - rf).abs() <= 1e-10 * rf.abs().max(1e-300) + floor_f, "{name}: f {} vs {rf}", res.fval);
    }
    if cfg!(target_os = "macos") {
        assert!(dx < 1e-15, "{name}: x {:?} vs {rx:?}", res.x);
        assert_eq!(res.iterations, rk, "{name}: iterations");
        assert_eq!(res.func_count, rnf, "{name}: function evaluations");
    }
    assert_eq!(res.exitflag as i64, if ref_flag > 0 { 1 } else { ref_flag }, "{name}: exitflag");
    assert_eq!(res.message, r["message"].as_str().unwrap(), "{name}: message");
}

#[test]
fn solveopt_matches_dynare() {
    let r = load().expect("reference/setup unavailable");
    let rosen2 = Rosen { n: 2, a: 10.0 };
    check("rosen2", &rosen2, &[-1.2, 1.0], &quiet(), &r["rosen2"]);
    check("rosen4", &Rosen { n: 4, a: 10.0 }, &[-1.2, 1.0, -1.2, 1.0], &quiet(), &r["rosen4"]);
    let q = Quad {
        a: Mat::from_row_slice(4, 3, &[2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 1.0, 1.0]),
        b: vec![1.0, 2.0, 3.0, 4.0],
    };
    check("quad3", &q, &[0.0, 0.0, 0.0], &quiet(), &r["quad3"]);
    check("rosen2_far", &Rosen { n: 2, a: 1.0 }, &[3.0, -2.0], &quiet(), &r["rosen2_far"]);
    check("rosen2_it25", &rosen2, &[-1.2, 1.0], &SolveoptOptions { max_iter: 25, ..quiet() }, &r["rosen2_it25"]);
}

#[test]
fn sequential_and_parallel_gradients_agree() {
    let o = Rosen { n: 4, a: 10.0 };
    let x0 = [-1.2, 1.0, -1.2, 1.0];
    let a = solveopt(&o, &x0, &SolveoptOptions { parallel: true, ..quiet() }).unwrap();
    let b = solveopt(&o, &x0, &SolveoptOptions { parallel: false, ..quiet() }).unwrap();
    assert_eq!(a.x, b.x);
    assert_eq!(a.func_count, b.func_count);
}

#[test]
fn failure_region_is_a_too_long_step() {
    // +inf outside x0 > -2 (like dsge_likelihood outside the bounds): the 1-D search shrinks.
    let o = dynare::estimation::FnObjective {
        n: 2,
        f: |x: &[f64]| if x[0] <= -1.5 { f64::INFINITY } else { (x[0] + 1.0).powi(2) + (x[1] - 2.0).powi(2) },
    };
    let r = solveopt(&o, &[1.0, 1.0], &quiet()).unwrap();
    assert!(r.fval.is_finite() && r.fval < 1e-8, "{r:?}");
}

#[test]
fn optim_overrides() {
    use dynare::modfile::OptValue;
    let mut o = SolveoptOptions::default();
    o.apply_optim(&[
        ("TolX".into(), OptValue::Num(1e-8)),
        ("MaxIter".into(), OptValue::Num(10.0)),
        ("verbosity".into(), OptValue::Num(0.0)),
        ("SpaceDilation".into(), OptValue::Num(3.0)),
        ("LBGradientStep".into(), OptValue::Num(1e-10)),
        ("TolFun".into(), OptValue::Num(1e-9)),
    ])
    .unwrap();
    assert_eq!(
        (o.tol_x, o.max_iter, o.verbosity, o.space_dilation, o.lb_gradient_step, o.tol_fun),
        (1e-8, 10, 0.0, 3.0, 1e-10, 1e-9)
    );
}

/// SW07 from the `estimated_params` init column (first_obs=5, presample=8, diffuse_filter):
/// `cargo test --release --test estimation_solveopt sw07 -- --ignored --nocapture`.
#[test]
#[ignore]
fn sw07_mode_compute_101() {
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
    let f0 = post.eval(&x0).0;
    println!("SW07: {} parameters, minus log posterior at the init values {f0:.4}", x0.len());
    let t = std::time::Instant::now();
    let r = solveopt(&post, &x0, &SolveoptOptions { verbosity: 50.0, ..Default::default() }).unwrap();
    println!(
        "SW07 mode_compute=101: fval {:.6} (STAGE-9 mode 349.856072; measured 349.856069), {} iterations, {} evaluations, exitflag {} ({}), {:.1}s",
        r.fval,
        r.iterations,
        r.func_count,
        r.exitflag,
        r.message,
        t.elapsed().as_secs_f64()
    );
    assert!((r.fval - 349.856072).abs() < 1e-3, "solveopt did not reach the SW07 mode");
}
