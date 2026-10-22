//! Nonlinear filters vs Dynare 7.2: reference/nonlinear/ref_rbcnl_o{2,3}.json,
//! written by reference/nonlinear/dump_nonlinear.m (rbcnl.mod: RBC at order 2 and 3 with
//! measurement errors on y and c, simulated data). The stochastic filters are fed MATLAB's
//! own uniform/normal draws (logged by shadow/rand.m, shadow/randn.m).
mod common;

use dynare::dsge::DsgeModel;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use dynare::nonlinear::rng::ReplayDraws;
use dynare::nonlinear::{
    Approximation, CpfWeights, FilterAlgorithm, NonlinearModel, ParticleOptions, ResamplingMethod, ResamplingStatus,
    law_of_motion, run_filter,
};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/nonlinear")
}

fn load(order: usize) -> Option<Value> {
    Some(common::json_ref(dir().join(format!("ref_rbcnl_o{order}.json"))))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(num).collect(),
        Value::Null => vec![],
        x => vec![num(x)],
    }
}

fn mat(v: &Value) -> Mat {
    match v {
        Value::Array(rows) if rows.first().is_some_and(|x| x.is_array()) => {
            let r = rows.len();
            let c = rows[0].as_array().unwrap().len();
            Mat::from_fn(r, c, |i, j| num(&rows[i][j]))
        }
        other => {
            let v = vecf(other);
            Mat::from_column_slice(v.len(), 1, &v)
        }
    }
}

fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn mdiff(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    (a - b).abs().max()
}

fn model(r: &Value) -> NonlinearModel {
    let order = r["order"].as_u64().unwrap() as usize;
    let mf = parse_mod_file(&dir().join("rbcnl.mod"), &[("ORDER".into(), order.to_string())]).unwrap();
    let x0 = vecf(&r["x0"]);
    let dsge = DsgeModel::new(&mf, Some(&x0)).unwrap();
    let y = mat(&r["data"]).transpose();
    let po = ParticleOptions { number_of_particles: 200, ..Default::default() };
    NonlinearModel::new(dsge, order, po, y, 0, 0).unwrap()
}

fn clone_model(_nm: &NonlinearModel, r: &Value) -> NonlinearModel {
    model(r)
}

fn case_options(kv: &[Value]) -> ParticleOptions {
    let mut p = ParticleOptions { number_of_particles: 200, ..Default::default() };
    for pair in kv.chunks(2) {
        let k = pair[0].as_str().unwrap();
        let v = &pair[1];
        let s = v.as_str().unwrap_or("");
        let ap = |s: &str| match s {
            "cubature" => Approximation::Cubature,
            "unscented" => Approximation::Unscented,
            _ => Approximation::MonteCarlo,
        };
        match k {
            "filter_algorithm" => p.algorithm = FilterAlgorithm::parse(s).unwrap(),
            "number_of_particles" => p.number_of_particles = num(v) as usize,
            "resampling.method" => {
                p.resampling_method = match s {
                    "kitagawa" => ResamplingMethod::Kitagawa,
                    "stratified" => ResamplingMethod::Stratified,
                    _ => ResamplingMethod::Smooth,
                }
            }
            "resampling.status" => {
                p.resampling = match s {
                    "systematic" => ResamplingStatus::Systematic,
                    "none" => ResamplingStatus::None,
                    _ => ResamplingStatus::Generic,
                }
            }
            "resampling.threshold" => p.resampling_threshold = num(v),
            "proposal" => p.proposal = ap(s),
            "distribution" => p.distribution = ap(s),
            "cpf_weights" => {
                p.cpf_weights =
                    if s == "murrayjonesparslow" { CpfWeights::MurrayJonesParslow } else { CpfWeights::AmisanoTristani }
            }
            "pruning" => p.pruning = v.as_bool().unwrap(),
            "mixture_state_variables" => p.mixture_state_variables = num(v) as usize,
            "mixture_structural_shocks" => p.mixture_structural_shocks = num(v) as usize,
            other => panic!("unknown case option {other}"),
        }
    }
    p
}

/// Deterministic filters (no draws at all).
fn deterministic(p: &ParticleOptions) -> bool {
    match p.algorithm {
        FilterAlgorithm::Gf => p.proposal != Approximation::MonteCarlo && p.distribution != Approximation::MonteCarlo,
        FilterAlgorithm::Nlkf => p.proposal != Approximation::MonteCarlo,
        _ => false,
    }
}

fn check_reduced_form(nm: &NonlinearModel, r: &Value) {
    let rf = nm.reduced_form(&vecf(&r["x0"])).unwrap();
    let w = &r["rf"];
    assert!(maxdiff(&rf.steadystate, &vecf(&w["steadystate"])) < 1e-10);
    assert!(maxdiff(&rf.constant, &vecf(&w["constant"])) < 1e-10);
    assert_eq!(
        rf.mf0.iter().map(|v| v + 1).collect::<Vec<_>>(),
        vecf(&w["mf0"]).iter().map(|v| *v as usize).collect::<Vec<_>>()
    );
    assert_eq!(
        rf.mf1.iter().map(|v| v + 1).collect::<Vec<_>>(),
        vecf(&w["mf1"]).iter().map(|v| *v as usize).collect::<Vec<_>>()
    );
    for (a, k) in [(&rf.ghx, "ghx"), (&rf.ghu, "ghu"), (&rf.ghxx, "ghxx"), (&rf.ghuu, "ghuu"), (&rf.ghxu, "ghxu")] {
        let d = mdiff(a, &mat(&w[k]));
        assert!(d < 1e-9, "{k}: {d:e}");
    }
    assert!(mdiff(&rf.h, &mat(&w["H"])) < 1e-18);
    assert!(maxdiff(&rf.state_mean, &vecf(&w["StateVectorMean"])) < 1e-10);
    let d = mdiff(&rf.state_variance, &mat(&w["StateVectorVariance"]));
    assert!(d < 1e-10, "StateVectorVariance {d:e}");
}

fn run_cases(order: usize) {
    let r = load(order).expect("reference/setup unavailable");
    let nm = model(&r);
    check_reduced_form(&nm, &r);
    let x0 = vecf(&r["x0"]);
    let x1 = vecf(&r["x1"]);
    let mut failures = Vec::new();
    for c in r["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let p = case_options(c["options"].as_array().unwrap());
        for (ip, x) in [(0, &x0), (1, &x1)] {
            let pr = &c[format!("p{ip}")];
            let matlab_error = pr["error"].as_str().is_some_and(|s| !s.is_empty());
            let lik_ref = num(&pr["LIK"]);
            let rf = nm.reduced_form(x).unwrap();
            let (lik, used) = if ip == 0 || deterministic(&p) {
                let mut d = ReplayDraws::new(vecf(&pr["normals"]), vecf(&pr["uniforms"]));
                if ip == 1 {
                    d = ReplayDraws::new(vec![], vec![]);
                }
                let out = run_filter(&rf, &nm.y, 0, &p, &mut d);
                (out, Some(d))
            } else {
                continue;
            };
            // the whole posterior kernel (non_linear_dsge_likelihood: -(loglik + logprior)).
            // (gaussian_mixture_filter.m does not reset the seed: its fval in the dump used
            // another stretch of MATLAB's stream than the logged direct call.)
            if !matlab_error && (ip == 0 || deterministic(&p)) && !name.contains("smooth") && !name.starts_with("gmf") {
                let mut nm2 = clone_model(&nm, &r);
                nm2.particle = p.clone();
                let mut d = if ip == 0 {
                    ReplayDraws::new(vecf(&pr["normals"]), vecf(&pr["uniforms"]))
                } else {
                    ReplayDraws::new(vec![], vec![])
                };
                let b = dynare::estimation::Bounds::unbounded(x.len());
                let fref = num(&pr["fval"]);
                let info_ref = num(&pr["info"]) as i32;
                match nm2.evaluate_with(x, &b, &mut d) {
                    Ok(e) => {
                        let rel = (e.fval - fref).abs() / fref.abs().max(1.0);
                        if !(rel < 1e-11) {
                            failures.push(format!("order {order} {name} p{ip}: fval {} vs {fref}", e.fval));
                        }
                    }
                    Err(f) => {
                        println!("order {order} {name:<24} p{ip}: info {} (MATLAB info {info_ref})", f.info);
                        if f.info != info_ref {
                            failures.push(format!("order {order} {name} p{ip}: info {} vs MATLAB {info_ref}", f.info));
                        }
                    }
                }
            }
            match (&lik, matlab_error) {
                (Err(e), true) => println!("order {order} {name:<24} p{ip}: both fail ({e})"),
                (Err(e), false) => {
                    failures.push(format!("order {order} {name} p{ip}: rust error {e}, MATLAB LIK {lik_ref}"))
                }
                (Ok(o), true) => {
                    failures.push(format!("order {order} {name} p{ip}: MATLAB error, rust LIK {}", o.lik_total))
                }
                (Ok(o), false) => {
                    let d = used.unwrap();
                    let (un, uu) = d.used();
                    let exact_draws = un == d.normals.len() && uu == d.uniforms.len();
                    let rl = vecf(&pr["lik"]);
                    let diff = if lik_ref.is_finite() {
                        (o.lik_total - lik_ref).abs()
                    } else if o.lik_total.is_finite() {
                        f64::INFINITY
                    } else {
                        0.0
                    };
                    let rel = diff / lik_ref.abs().max(1.0);
                    println!(
                        "order {order} {name:<24} p{ip}: LIK rust {:>20.12} matlab {:>20.12} rel {rel:.2e} draws used {un}/{} {uu}/{}",
                        o.lik_total,
                        lik_ref,
                        d.normals.len(),
                        d.uniforms.len()
                    );
                    // the first period precedes any resampling
                    if rl.first().is_some_and(|v| v.is_finite()) {
                        let d0 = (o.lik[0] - rl[0]).abs() / rl[0].abs().max(1.0);
                        assert!(d0 < 1e-12, "{name} p{ip}: lik(1) {d0:e}");
                    }
                    // smooth resampling: eigenvector signs are LAPACK's choice (statistical match).
                    // Otherwise the same draws give the same path, but resampling thresholds and
                    // T log-likelihood increments accumulate libm differences: 1e-6 on the total
                    // (the first period, checked above, stays at 1e-12).
                    let tol = if name.contains("smooth") { 1e-2 } else { 1e-6 };
                    if !(rel < tol) || !exact_draws {
                        failures.push(format!(
                            "order {order} {name} p{ip}: rel {rel:e}, draws {un}/{} {uu}/{}",
                            d.normals.len(),
                            d.uniforms.len()
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn filters_order2() {
    run_cases(2);
}

#[test]
fn filters_order3() {
    run_cases(3);
}

#[test]
fn local_state_space_iteration_3() {
    let r = load(3).expect("reference/setup unavailable");
    let nm = model(&r);
    let rf = nm.reduced_form(&vecf(&r["x0"])).unwrap();
    let l = &r["lssi3"];
    for k in ["ghxxx", "ghuuu", "ghxxu", "ghxuu", "ghxss", "ghuss"] {
        let a = match k {
            "ghxxx" => &rf.ghxxx,
            "ghuuu" => &rf.ghuuu,
            "ghxxu" => &rf.ghxxu,
            "ghxuu" => &rf.ghxuu,
            "ghxss" => &rf.ghxss,
            _ => &rf.ghuss,
        };
        assert!(mdiff(a, &mat(&l[k])) < 1e-8, "{k}");
    }
    let ss = Mat::from_column_slice(rf.ns(), 1, &rf.state_ss);
    let yhat = mat(&l["yhat"]);
    let states = Mat::from_fn(yhat.nrows(), yhat.ncols(), |i, j| yhat[(i, j)] + ss[(i, 0)]);
    let eps = mat(&l["epsilon"]);
    let y = law_of_motion::iterate(&rf, &states, &eps, None).y;
    let d = mdiff(&y, &mat(&l["y"]));
    println!("lssi3 unpruned: {d:e}");
    assert!(d < 1e-12);
    let y3 = mat(&l["yhat3"]);
    let ns = rf.ns();
    let st3 = Mat::from_fn(y3.nrows(), y3.ncols(), |i, j| y3[(i, j)] + ss[(i % ns, 0)]);
    let it = law_of_motion::iterate(&rf, &st3, &eps, Some(&st3));
    let d1 = mdiff(&it.y, &mat(&l["y_pruned"]));
    let d2 = mdiff(it.pruned.as_ref().unwrap(), &mat(&l["y_pruned_state"]));
    println!("lssi3 pruned: y {d1:e}, pruned state {d2:e}");
    assert!(d1 < 1e-12 && d2 < 1e-12);
}

/// Per-evaluation times (`cargo test --release --test nonlinear -- --ignored --nocapture`),
/// against MATLAB's (reference/nonlinear/ref_timing.json).
#[test]
#[ignore]
fn timings() {
    let matlab: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("ref_timing.json")).unwrap()).unwrap();
    for order in [2, 3] {
        let r = load(order).expect("reference/setup unavailable");
        let x0 = vecf(&r["x0"]);
        let cases: Vec<(&str, Vec<Value>)> = vec![
            ("sis_5000", serde_json::from_str(r#"["filter_algorithm","sis","number_of_particles",5000]"#).unwrap()),
            ("sis_20000", serde_json::from_str(r#"["filter_algorithm","sis","number_of_particles",20000]"#).unwrap()),
            ("apf_5000", serde_json::from_str(r#"["filter_algorithm","apf","number_of_particles",5000]"#).unwrap()),
            (
                "gf_cubature",
                serde_json::from_str(r#"["filter_algorithm","gf","proposal","cubature","distribution","cubature"]"#)
                    .unwrap(),
            ),
            ("nlkf_unscented", serde_json::from_str(r#"["filter_algorithm","nlkf","proposal","unscented"]"#).unwrap()),
            (
                "cpf_unscented_500",
                serde_json::from_str(r#"["filter_algorithm","cpf","proposal","unscented","number_of_particles",500]"#)
                    .unwrap(),
            ),
            (
                "gmf_cubature",
                serde_json::from_str(r#"["filter_algorithm","gmf","proposal","cubature","distribution","cubature"]"#)
                    .unwrap(),
            ),
        ];
        for (name, kv) in cases {
            let mut nm = model(&r);
            nm.particle = case_options(&kv);
            let b = dynare::estimation::Bounds::unbounded(x0.len());
            let _ = nm.evaluate(&x0, &b);
            let t = std::time::Instant::now();
            let mut n = 0;
            while t.elapsed().as_secs_f64() < 1.0 || n < 3 {
                let _ = nm.evaluate(&x0, &b);
                n += 1;
            }
            let rs = t.elapsed().as_secs_f64() / n as f64;
            let ms = num(&matlab[format!("o{order}_{name}")]);
            println!(
                "order {order} {name:<20} rust {:>9.3} ms   MATLAB {:>9.1} ms   x{:.0}",
                rs * 1e3,
                ms * 1e3,
                ms / rs
            );
        }
    }
}

#[test]
#[ignore]
fn timing_parts() {
    use dynare::nonlinear::rng::{Draws, SeededDraws};
    for order in [2, 3] {
        let r = load(order).unwrap();
        let nm = model(&r);
        let rf = nm.reduced_form(&vecf(&r["x0"])).unwrap();
        let n = 5000;
        let mut d = SeededDraws::new(1);
        let st = Mat::from_fn(rf.ns(), n, |i, _| rf.state_mean[i]);
        let t = std::time::Instant::now();
        for _ in 0..60 {
            let _ = d.randn_mat(rf.nu(), n);
        }
        println!("order {order} randn: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
        let eps = d.randn_mat(rf.nu(), n) * 0.01;
        let t = std::time::Instant::now();
        for _ in 0..60 {
            let _ = law_of_motion::iterate(&rf, &st, &eps, None);
        }
        println!(
            "order {order} iterate x60: {:.2} ms (m = {}, ns = {})",
            t.elapsed().as_secs_f64() * 1e3,
            rf.m(),
            rf.ns()
        );
    }
}
