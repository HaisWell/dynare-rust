//! `mode_compute = 8` (Dynare's `simplex_optimization_routine.m`) and `mode_compute = 7`
//! (MATLAB's `fminsearch` with Dynare's options) against MATLAB
//! (`reference/estimation/ref_simplex.json`, written by `reference/estimation/dump_simplex.m`).
//! Both are deterministic: iteration and function counts must match exactly.
mod common;

use dynare::estimation::{FminsearchOptions, FnObjective, SimplexOptions, dynare_simplex, fminsearch};
use serde_json::Value;

fn load() -> Option<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/estimation/ref_simplex.json");
    Some(common::json_ref(path))
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap()).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![],
    }
}

fn rosen(x: &[f64]) -> f64 {
    (0..x.len() - 1).map(|i| (10.0 * (x[i + 1] - x[i] * x[i])).powi(2) + (1.0 - x[i]).powi(2)).sum()
}

fn quad(x: &[f64]) -> f64 {
    let a = [[2., 1., 0.], [0., 1., 1.], [1., 0., 3.], [1., 1., 1.]];
    let b = [1., 2., 3., 4.];
    (0..4).map(|k| 0.5 * ((0..3).map(|j| a[k][j] * x[j]).sum::<f64>() - b[k]).powi(2)).sum()
}

fn cases() -> Vec<(&'static str, fn(&[f64]) -> f64, Vec<f64>)> {
    vec![
        ("r2", rosen as fn(&[f64]) -> f64, vec![-1.2, 1.0]),
        ("r4", rosen, vec![-1.2, 1.0, -1.2, 1.0]),
        ("q3", quad, vec![0.3, -0.2, 0.5]),
        ("r5", rosen, vec![0.0; 5]),
    ]
}

fn check(name: &str, r: &Value, x: &[f64], f: f64, it: usize, fc: usize) {
    let rx = vecf(&r["x"]);
    let rf = r["f"].as_f64().unwrap();
    println!("{name}: f {f:e} (MATLAB {rf:e}), it {it} ({}), fc {fc} ({})", r["it"], r["fc"]);
    assert_eq!(it as u64, r["it"].as_u64().unwrap(), "{name}: iterations");
    assert_eq!(fc as u64, r["fc"].as_u64().unwrap(), "{name}: function count");
    for (a, b) in x.iter().zip(&rx) {
        assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{name}: x {a} vs {b}");
    }
    assert!((f - rf).abs() <= 1e-9 * (1e-6 + rf.abs()), "{name}: f {f} vs {rf}");
}

#[test]
fn dynare_simplex_matches_matlab() {
    let r = load().expect("reference/setup unavailable");
    let quiet = SimplexOptions { verbosity: 0, ..Default::default() };
    for (name, f, x0) in cases() {
        let obj = FnObjective { n: x0.len(), f };
        let res = dynare_simplex(&obj, &x0, &quiet).unwrap();
        let rr = &r[format!("s_{name}")];
        check(&format!("simplex {name}"), rr, &res.x, res.fval, res.iterations, res.func_count);
        assert_eq!(res.exitflag as i64, rr["ef"].as_i64().unwrap());
    }
    let obj = FnObjective { n: 4, f: rosen };
    let res = dynare_simplex(&obj, &[-1.2, 1.0, -1.2, 1.0], &SimplexOptions { maxfcall: Some(300), ..quiet.clone() })
        .unwrap();
    check("simplex cap", &r["s_cap"], &res.x, res.fval, res.iterations, res.func_count);
    assert_eq!(res.exitflag, 0);
    assert_eq!(res.message, r["s_cap"]["msg"].as_str().unwrap());
}

#[test]
fn fminsearch_matches_matlab() {
    let r = load().expect("reference/setup unavailable");
    for (name, f, x0) in cases() {
        let obj = FnObjective { n: x0.len(), f };
        let res = fminsearch(&obj, &x0, &FminsearchOptions::default());
        check(&format!("fminsearch {name}"), &r[format!("f_{name}")], &res.x, res.fval, res.iterations, res.func_count);
    }
}

#[test]
fn simplex_optim_keys() {
    use dynare::estimation::optim::minimize::key_values;
    use dynare::modfile::OptValue;
    let v = OptValue::List(vec![
        OptValue::Str("'MaxIter'".into()),
        OptValue::Num(7.0),
        OptValue::Str("'InitialSimplexSize'".into()),
        OptValue::Num(0.1),
    ]);
    let mut o = SimplexOptions::default();
    o.apply_optim(&key_values(Some(&v), "simplex").unwrap()).unwrap();
    assert_eq!(o.maxiter, 7);
    assert_eq!(o.delta_factor, 0.1);
}

/// Substitutes for the toolbox optimizers (mode_compute 1, 3, 12): not MATLAB's
/// algorithms, so only the minimum reached is checked.
#[test]
fn toolbox_substitutes_reach_the_minimum() {
    use dynare::estimation::Bounds;
    use dynare::estimation::optim::{MinimizeContext, minimize};
    let names: Vec<String> = (0..4).map(|i| format!("p{i}")).collect();
    let sd = vec![f64::NAN; 4];
    let bounds = Bounds { lb: vec![-5.0; 4], ub: vec![5.0; 4] };
    for mc in [1, 3] {
        let ctx = MinimizeContext {
            bounds: &bounds,
            names: &names,
            prior_sd: &sd,
            optim: None,
            verbose: false,
            seed: 3,
            stream: 9,
        };
        let obj = FnObjective { n: 4, f: rosen };
        let r = minimize(mc, &obj, &[-1.2, 1.0, -1.2, 1.0], &ctx).unwrap();
        println!("mode_compute={mc}: f {:e}, {} evals, {}", r.fval, r.func_count, r.message);
        assert!(r.fval < 1e-8, "mode_compute={mc}: f = {}", r.fval);
        assert!(r.x.iter().all(|v| (-5.0..=5.0).contains(v)));
    }
    // Particle swarm (stochastic; the 4-d Rosenbrock has a local minimum at f = 1.27, so
    // use the 2-d one and the quadratic).
    let ctx = MinimizeContext {
        bounds: &bounds,
        names: &names,
        prior_sd: &sd,
        optim: None,
        verbose: false,
        seed: 3,
        stream: 9,
    };
    let r = minimize(
        12,
        &FnObjective { n: 2, f: rosen },
        &[-1.2, 1.0],
        &MinimizeContext { bounds: &Bounds { lb: vec![-5.0; 2], ub: vec![5.0; 2] }, ..ctx },
    )
    .unwrap();
    println!("mode_compute=12 rosen2: f {:e}, {} evals", r.fval, r.func_count);
    assert!(r.fval < 1e-2);
    let r = minimize(
        12,
        &FnObjective { n: 3, f: quad },
        &[0.3, -0.2, 0.5],
        &MinimizeContext { bounds: &Bounds { lb: vec![-5.0; 3], ub: vec![5.0; 3] }, ..ctx },
    )
    .unwrap();
    println!("mode_compute=12 quad3: f {:.10}, {} evals", r.fval, r.func_count);
    assert!((r.fval - 1.3066666666666).abs() < 1e-4);
    // fmincon substitute with an active bound: minimum of rosen with x0 <= 0.5.
    let b = Bounds { lb: vec![-5.0, -5.0], ub: vec![0.5, 5.0] };
    let ctx = MinimizeContext {
        bounds: &b,
        names: &names[..2],
        prior_sd: &sd[..2],
        optim: None,
        verbose: false,
        seed: 0,
        stream: 0,
    };
    let r = minimize(1, &FnObjective { n: 2, f: rosen }, &[-1.2, 1.0], &ctx).unwrap();
    assert!((r.x[0] - 0.5).abs() < 1e-6 && (r.x[1] - 0.25).abs() < 1e-4, "{:?}", r.x);
}
