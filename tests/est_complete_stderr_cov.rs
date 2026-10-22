//! Estimated stderrs of two shocks while the `shocks` block calibrates their covariance (stage 1)
//! or correlation (stage 2): reference/est_complete/stderr_cov (dump_sc.m -> ref_sc.json, sc.mod).
//! Dynare 7.1 rebuilds Sigma_e as diag(sd)*M_.Correlation_matrix*diag(sd) (set_all_parameters.m);
//! a calibrated covariance (`var e, u = x`) is not in Correlation_matrix, so it is zeroed, unless a corr
//! among shocks is estimated (stage 3) or it is a measurement-error covariance (stage 4): then
//! check_for_calibrated_covariances.m records it and set_all_parameters.m restores it.

mod common;

use std::path::{Path, PathBuf};

use dynare::modfile::parse_mod_file;
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/est_complete/stderr_cov")
}

fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn stderr_estimated_with_calibrated_covariance() {
    let r: Value = common::json_ref(dir().join("ref_sc.json"));
    for stage in 1..=4 {
        let s = &r[format!("stage{stage}")];
        let mf = parse_mod_file(&dir().join("sc.mod"), &[("STAGE".into(), stage.to_string())]).unwrap();
        let x0 = vecf(&s["xparam_init"]);
        let dsge = dynare::dsge::DsgeModel::new(&mf, Some(&x0)).unwrap();
        let table = dynare::data::Table::read(&dir().join("rbc_levels.csv"), &Default::default()).unwrap();
        let mut sample = table.sample(&mf.varobs, 1, None).unwrap();
        sample.iter_mut().for_each(|v| *v = v.ln());
        let data = dynare::dsge::LikelihoodData::new(&sample, Default::default());
        let post = dynare::dsge::DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
        // set_all_parameters at point 2: M_.Sigma_e and M_.H
        let x2 = vecf(&s["points"][1]["x"]);
        let (_, q, h) = dsge.set_all_parameters_h(&x2);
        for (m, key) in [(&q, "sigma_e_at_point2"), (&h, "h_at_point2")] {
            let rows = s[key].as_array().unwrap();
            for (i, row) in rows.iter().enumerate() {
                for (j, v) in vecf(row).iter().enumerate() {
                    assert!((m[(i, j)] - v).abs() < 1e-15, "stage {stage} {key}[{i},{j}]: {} vs {v}", m[(i, j)]);
                }
            }
        }
        for (k, p) in s["points"].as_array().unwrap().iter().enumerate() {
            let f = dynare::estimation::Objective::eval(&post, &vecf(&p["x"])).0;
            let v = p["fval"].as_f64().unwrap();
            println!("stage {stage} point {k}: {f:.10} vs Dynare {v:.10}");
            assert!((f - v).abs() < 1e-8 * v.abs().max(1.0), "stage {stage} point {k}: {f} vs {v}");
        }
    }
}
