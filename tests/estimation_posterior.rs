//! Posterior statistics against Dynare 7.2 on deterministic series
//! (`reference/estimation/ml_ref_posterior.json`, from `run_ref_posterior.m`).
mod common;

use dynare::estimation::{mcmc_ifac, posterior_moments};
use serde_json::Value;

fn load() -> Option<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/estimation/ml_ref_posterior.json");
    Some(common::json_ref(path))
}

fn series() -> Vec<f64> {
    (1..=5001)
        .map(|t| {
            let t = t as f64;
            (0.1 * t).sin() + 0.5 * (0.37 * t.powf(1.1)).cos() + 0.01 * ((t * t) % 17.0)
        })
        .collect()
}

#[test]
fn inefficiency_factor_matches_mcmc_ifac() {
    let r = load().expect("reference/setup unavailable");
    let x = series();
    for (key, data, nc) in
        [("ifac_1000", &x[..], 1000.0), ("ifac_2500_5", &x[..], 2500.5), ("ifac_short", &x[..333], 1000.0)]
    {
        let f = mcmc_ifac(data, nc);
        let t = r[key].as_f64().unwrap();
        eprintln!("{key}: {f} vs {t}");
        assert!((f - t).abs() < 1e-9 * t.abs(), "{key}: {f} vs {t}");
    }
}

#[test]
fn posterior_moments_match() {
    let r = load().expect("reference/setup unavailable");
    let x = series();
    for (key, n, sig) in [("pm", 5001, 0.9), ("pm2", 2000, 0.8)] {
        let mut xx = x[..n].to_vec();
        let m = posterior_moments(&mut xx, sig);
        let d = &r[key];
        let close = |a: f64, b: &Value| (a - b.as_f64().unwrap()).abs() < 1e-12;
        assert!(close(m.mean, &d["mean"]), "{key} mean");
        assert!(close(m.median, &d["median"]), "{key} median");
        assert!(close(m.var, &d["var"]), "{key} var");
        assert!(close(m.hpd.0, &d["hpd"][0]) && close(m.hpd.1, &d["hpd"][1]), "{key} hpd {:?}", m.hpd);
        for k in 0..9 {
            assert!(close(m.deciles[k], &d["deciles"][k]), "{key} decile {k}");
        }
    }
}
