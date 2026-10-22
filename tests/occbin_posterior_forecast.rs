//! Posterior forecasts after an OccBin MCMC against Dynare 7.2 (reference/occbin_fcst/, dump.m):
//! the same 200 parameter draws (Dynare's own, all kept draws of a short MCMC) are fed to
//! dynare-rust's OccBin posterior statistics with `forecast=8`; `MeanForecast` (no future
//! shocks) must reproduce Dynare's mean/median/HPD/variance, `PointForecast` (random future
//! shocks) its mean only statistically.
mod common;

use std::path::{Path, PathBuf};

use dynare::data::Table;
use dynare::dsge::{DsgeModel, LikelihoodData};
use dynare::kalman::KalmanOptions;
use dynare::modfile::{Command, OptValue, parse_mod_file};
use dynare::occbin::estimation::OccbinEstimation;
use dynare::occbin::{OccbinCommand, Session};
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/occbin_fcst")
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

/// Column `col` (0 = mean, 1 = median, 2 = sd, 3 = hpd_lower, 4 = hpd_upper) of `var` in a summary csv.
fn column(csv: &str, var: &str, col: usize) -> Vec<f64> {
    csv.lines()
        .skip(1)
        .filter(|l| l.split(',').nth(1) == Some(var))
        .map(|l| l.split(',').nth(2 + col).unwrap().parse().unwrap())
        .collect()
}

#[test]
fn occbin_posterior_forecasts() {
    let text = common::read_ref(dir().join("ref_occbin_fcst.json"));
    let r: Value = serde_json::from_str(&text).unwrap();
    let draws: Vec<Vec<f64>> = r["draws"].as_array().unwrap().iter().map(vecf).collect();
    let mf = parse_mod_file(&dir().join("occbin_fcst.mod"), &[("MCMC".to_string(), "1".to_string())]).unwrap();
    let mut s = Session::new(&mf, "occbin_fcst");
    for c in &mf.commands {
        if let Command::Occbin(OccbinCommand::Setup(o)) = c {
            s.setup(&mf, o).unwrap();
        }
    }
    let dsge = DsgeModel::new(&mf, Some(&draws[0])).unwrap();
    let table = Table::read_csv(&dir().join("nk_est_data.csv")).unwrap();
    let sample = table.sample(&mf.varobs, 1, None).unwrap();
    let data = LikelihoodData::new(&sample, KalmanOptions { lik_init: 1, ..Default::default() });
    let occ = OccbinEstimation::new(&mf, &s, &dsge, &data).unwrap().unwrap();
    let o = vec![("smoother".to_string(), OptValue::Flag), ("forecast".to_string(), OptValue::Num(8.0))];
    let out = std::env::temp_dir().join("occbin_fcst_test");
    dynare::estimation::q3_occbin::posterior_from_draws(&dsge, &data, &occ, &o, &draws, 1, None, &out, "occbin_fcst")
        .unwrap();
    let mean_csv = std::fs::read_to_string(out.join("occbin_fcst_posterior_mean_forecast.csv")).unwrap();
    let point_csv = std::fs::read_to_string(out.join("occbin_fcst_posterior_point_forecast.csv")).unwrap();
    for var in ["y", "pi", "i", "inot", "r_nat", "u_s"] {
        for (col, key, scale_sd) in [
            (0, "mean_forecast_mean", false),
            (1, "mean_forecast_median", false),
            (3, "mean_forecast_hpdinf", false),
            (4, "mean_forecast_hpdsup", false),
        ] {
            let ours = column(&mean_csv, var, col);
            let want = vecf(&r[key][var]);
            assert_eq!(ours.len(), want.len(), "{var} {key}");
            let _ = scale_sd;
            for (t, (a, b)) in ours.iter().zip(&want).enumerate() {
                assert!((a - b).abs() < 1e-8 + 1e-6 * b.abs(), "{var} {key} t={t}: {a} vs {b}");
            }
        }
        let sd = column(&mean_csv, var, 2);
        let var_ref = vecf(&r["mean_forecast_var"][var]);
        for (a, b) in sd.iter().zip(&var_ref) {
            assert!((a * a - b).abs() < 1e-10 + 1e-6 * b.abs(), "{var} var {a} {b}");
        }
        // PointForecast: mean over 200 draws of one random-shock path; compare with the
        // sampling noise of the shocks (within a few standard errors, using the cross-draw sd)
        let pm = column(&point_csv, var, 0);
        let psd = column(&point_csv, var, 2);
        let want = vecf(&r["point_forecast_mean"][var]);
        for t in 0..pm.len() {
            let se = psd[t] / (draws.len() as f64).sqrt();
            assert!(
                (pm[t] - want[t]).abs() < 5.0 * se * std::f64::consts::SQRT_2 + 1e-9,
                "{var} point t={t}: {} vs {} (se {se})",
                pm[t],
                want[t]
            );
        }
    }
}
