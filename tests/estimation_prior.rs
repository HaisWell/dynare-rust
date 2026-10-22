//! Priors, bounds and prior density against Dynare 7.2 on SW07
//! (`reference/sw07/ref.json`, `reference/sw07/ref_prior_trunc.json`).
mod common;

use dynare::estimation::{Prior, PriorShape, estimation_bounds, log_prior_density, mcmc_bounds, prior_bounds};
use serde_json::Value;

fn load() -> Option<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/sw07/ref.json");
    Some(common::json_ref(path))
}

fn vecf(v: &Value) -> Vec<Option<f64>> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64()).collect(),
        Value::Number(n) => vec![n.as_f64()],
        _ => vec![],
    }
}

/// Explicit bounds of `estimated_params` in sw07.mod.
fn explicit_bounds(name: &str) -> (Option<f64>, Option<f64>) {
    match name {
        "stderr ea" | "stderr em" => (Some(0.01), Some(3.0)),
        "csigma" => (Some(0.25), Some(3.0)),
        "cfc" | "crpi" => (Some(1.0), Some(3.0)),
        "constelab" => (Some(-10.0), Some(10.0)),
        _ => (None, None),
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    a == b || (a - b).abs() <= tol * (1.0 + b.abs())
}

struct Ref {
    priors: Vec<Prior>,
    names: Vec<String>,
    r: Value,
}

fn build() -> Option<Ref> {
    let r = load()?;
    let b = r.get("bayestopt")?;
    let pshape = vecf(&b["pshape"]);
    let (p1, p2, p3, p4) = (vecf(&b["p1"]), vecf(&b["p2"]), vecf(&b["p3"]), vecf(&b["p4"]));
    let names: Vec<String> =
        r["estim_param_names"].as_array()?.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
    let priors = (0..pshape.len())
        .map(|i| {
            let shape = PriorShape::from_code(pshape[i].unwrap() as u8).unwrap();
            // null in the JSON = -inf/+inf (the defaults set_prior fills in)
            Prior::new(shape, p1[i].unwrap(), p2[i].unwrap(), p3[i], p4[i]).unwrap()
        })
        .collect();
    Some(Ref { priors, names, r })
}

#[test]
fn hyperparameters_match_bayestopt() {
    let rf = build().expect("reference/setup unavailable");
    let b = &rf.r["bayestopt"];
    let (p3, p4, p5, p6, p7) = (vecf(&b["p3"]), vecf(&b["p4"]), vecf(&b["p5"]), vecf(&b["p6"]), vecf(&b["p7"]));
    let mut worst: f64 = 0.0;
    for (i, p) in rf.priors.iter().enumerate() {
        let unb = |v: Option<f64>, def: f64| v.unwrap_or(def);
        assert_eq!(p.p3, unb(p3[i], f64::NEG_INFINITY), "p3 {}", rf.names[i]);
        assert_eq!(p.p4, unb(p4[i], f64::INFINITY), "p4 {}", rf.names[i]);
        for (mine, theirs, what) in [(p.p5, p5[i], "p5"), (p.p6, p6[i], "p6"), (p.p7, p7[i], "p7")] {
            let t = theirs.unwrap();
            let rel = (mine - t).abs() / (1.0 + t.abs());
            worst = worst.max(rel);
            assert!(close(mine, t, 1e-12), "{what} {}: {mine} vs {t}", rf.names[i]);
        }
    }
    eprintln!("max relative error on p5/p6/p7: {worst:e}");
}

#[test]
fn bounds_match_dynare_estimation_init() {
    let rf = build().expect("reference/setup unavailable");
    let lb = vecf(&rf.r["bounds"]["lb"]);
    let ub = vecf(&rf.r["bounds"]["ub"]);
    let explicit: Vec<_> = rf.names.iter().map(|n| explicit_bounds(n)).collect();
    let b = estimation_bounds(&rf.priors, &explicit, 0.0);
    for i in 0..rf.priors.len() {
        assert_eq!(b.lb[i], lb[i].unwrap_or(f64::NEG_INFINITY), "lb {}", rf.names[i]);
        assert_eq!(b.ub[i], ub[i].unwrap_or(f64::INFINITY), "ub {}", rf.names[i]);
    }
    // MCMC bounds: prior support only.
    let xm: Vec<f64> = vecf(&rf.r["xparam_mode"]).into_iter().map(|v| v.unwrap()).collect();
    let mb = mcmc_bounds(&xm, &rf.priors, 0.0).unwrap();
    assert_eq!(mb, prior_bounds(&rf.priors, 0.0));
    // prior_trunc > 0 gives quantile bounds inside the support
    let tb = prior_bounds(&rf.priors, 1e-10);
    for i in 0..rf.priors.len() {
        assert!(tb.lb[i] >= mb.lb[i] && tb.ub[i] <= mb.ub[i] && tb.lb[i] < tb.ub[i], "{}", rf.names[i]);
    }
}

#[test]
fn log_prior_at_reference_points() {
    let rf = build().expect("reference/setup unavailable");
    let points = rf.r["points"].as_array().expect("reference/setup unavailable");
    let mut worst: f64 = 0.0;
    for (k, pt) in points.iter().enumerate() {
        let x: Vec<f64> = vecf(&pt["x"]).into_iter().map(|v| v.unwrap()).collect();
        let lp = log_prior_density(&x, &rf.priors);
        match pt["log_prior"].as_f64() {
            Some(t) => {
                worst = worst.max((lp - t).abs());
                assert!((lp - t).abs() < 1e-12, "point {k}: {lp} vs {t}");
            }
            None => assert!(lp == f64::NEG_INFINITY || lp.is_nan(), "point {k}: {lp}, Dynare -inf"),
        }
    }
    eprintln!("max abs error on log prior: {worst:e}");
}

fn load_ml() -> Option<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/estimation/ml_ref.json");
    Some(common::json_ref(path))
}

#[test]
fn all_shapes_and_truncated_bounds_match_dynare() {
    let r = load_ml().expect("reference/setup unavailable");
    let shapes = [
        PriorShape::Beta,
        PriorShape::Gamma,
        PriorShape::Normal,
        PriorShape::InvGamma1,
        PriorShape::Uniform,
        PriorShape::InvGamma2,
        PriorShape::Weibull,
    ];
    let s = &r["shapes"];
    let (p5, p6, p7) = (vecf(&s["p5"]), vecf(&s["p6"]), vecf(&s["p7"]));
    let priors: Vec<Prior> = shapes.iter().map(|&sh| Prior::new(sh, 0.4, 0.2, None, None).unwrap()).collect();
    for (i, p) in priors.iter().enumerate() {
        assert!(close(p.p6, p6[i].unwrap(), 1e-12), "p6 {:?}: {} vs {:?}", p.shape, p.p6, p6[i]);
        assert!(close(p.p7, p7[i].unwrap(), 1e-12), "p7 {:?}: {} vs {:?}", p.shape, p.p7, p7[i]);
        if p.shape != PriorShape::Uniform {
            // (set_prior.m sets the uniform's p5 to NaN; compute_prior_mode returns p6)
            assert!(close(p.p5, p5[i].unwrap(), 1e-12), "p5 {:?}", p.shape);
        }
    }
    let lp = log_prior_density(&[0.3, 0.5, -0.1, 0.35, 0.2, 0.45, 0.6], &priors);
    let t = r["shapes_lp"].as_f64().unwrap();
    assert!((lp - t).abs() < 1e-12, "{lp} vs {t}");
    // prior_trunc = 1e-10 (missing/stats betainv/gaminv/norminv/wblinv ports)
    let tb = prior_bounds(&priors, 1e-10);
    let (lb, ub) = (vecf(&r["shapes_trunc"]["lb"]), vecf(&r["shapes_trunc"]["ub"]));
    let mut worst: f64 = 0.0;
    for i in 0..7 {
        for (m, t) in [(tb.lb[i], lb[i].unwrap()), (tb.ub[i], ub[i].unwrap())] {
            worst = worst.max((m - t).abs() / t.abs());
            assert!(close(m, t, 1e-6), "{:?}: {m} vs {t}", shapes[i]);
        }
    }
    eprintln!("max relative error on prior_trunc=1e-10 bounds: {worst:e}");
}

/// prior_trunc = 1e-10 bounds of the SW07 priors and chi2inv at their number (40), from
/// reference/sw07/dump_prior_trunc.m.
#[test]
fn sw07_truncated_bounds_and_chi2inv_match_dynare() {
    let rf = build().expect("reference/setup unavailable");
    let r: Value = common::json_ref(concat!(env!("CARGO_MANIFEST_DIR"), "/reference/sw07/ref_prior_trunc.json"));
    let tb = prior_bounds(&rf.priors, r["prior_trunc"].as_f64().unwrap());
    let (lb, ub) = (vecf(&r["lb"]), vecf(&r["ub"]));
    assert_eq!(lb.len(), rf.priors.len());
    let mut worst: f64 = 0.0;
    for i in 0..rf.priors.len() {
        for (m, t) in [(tb.lb[i], lb[i].unwrap()), (tb.ub[i], ub[i].unwrap())] {
            worst = worst.max((m - t).abs() / t.abs());
            assert!(close(m, t, 1e-9), "{}: {m} vs {t}", rf.names[i]);
        }
    }
    eprintln!("max relative error on the SW07 prior_trunc=1e-10 bounds: {worst:e}");
    let dof = r["chi2_dof"].as_f64().unwrap();
    assert_eq!(dof as usize, rf.priors.len());
    for (p, t) in vecf(&r["chi2_p"]).into_iter().zip(vecf(&r["chi2"])) {
        let c = dynare::estimation::special::chi2inv(p.unwrap(), dof);
        assert!(close(c, t.unwrap(), 1e-12), "chi2inv({p:?}, {dof}): {c} vs {t:?}");
    }
}
