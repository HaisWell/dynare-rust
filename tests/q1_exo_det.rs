//! : `varexo_det` in a stochastic model against Dynare 7.2 (reference/q1/q1_xdet.mod,
//! ref_q1xdet.json and q1xdet.log from reference/q1/dump_q1xdet.m): the deterministic
//! exogenous variable is out of `M_.exo_names` / `Sigma_e` (stoch_simul output), the decision
//! rules are computed at `oo_.exo_det_steady_state`, `dr.ghud` and the forecast with the
//! deterministic path (`simult_varexo_det.m`).

use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/q1")
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

/// `set_shock_stderr_value('e', v)`: `M_.Sigma_e(i,i) = v^2` before the first command, a
/// command at its place after.
#[test]
fn set_shock_stderr_value() {
    let src = "var y; varexo e u; parameters rho; rho = 0.5;\nmodel; y = rho*y(-1) + e + u; end;\nshocks; var e; stderr 0.1; var u; stderr 0.2; end;\nset_shock_stderr_value('e', 0.3);\nstoch_simul(order=1, irf=0);\nset_shock_stderr_value('u', 0.05);\nstoch_simul(order=1, irf=0);\n";
    let mf = dynare::modfile::parse_mod_str(src, &[]).unwrap();
    assert!((mf.sigma_e[(0, 0)] - 0.09).abs() < 1e-15);
    assert!((mf.sigma_e[(1, 1)] - 0.04).abs() < 1e-15);
    let cmd = mf.commands.iter().find_map(|c| match c {
        dynare::modfile::Command::Other(n, o) if n == dynare::modfile::SET_SHOCK_STDERR => Some(o.clone()),
        _ => None,
    });
    assert_eq!(cmd.unwrap(), vec![("u".to_string(), dynare::modfile::OptValue::Num(0.05))]);
}

#[test]
fn varexo_det_decision_rules_and_forecast() {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(dir().join("ref_q1xdet.json")).unwrap()).unwrap();
    let mf = parse_mod_file(&dir().join("q1_xdet.mod"), &[]).unwrap();
    let m = &mf.model;
    assert_eq!(m.exo_names, vec!["e".to_string()]);
    assert_eq!(m.exo_det_names, vec!["g".to_string()]);
    assert_eq!(mf.sigma_e.nrows(), 1);
    let sol = dynare::analysis::solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
    let n = m.n_endo();
    let ys = vecf(&r["ys"]);
    for (a, b) in sol.dr.ys.iter().zip(&ys) {
        assert!((a - b).abs() < 1e-9 * b.abs().max(1.0), "ys {a} vs {b}");
    }
    let ghx = vecf(&r["ghx"]);
    let nsp = sol.dr.ghx.ncols();
    let ghx = Mat::from_row_slice(n, nsp, &ghx);
    for i in 0..n {
        for j in 0..nsp {
            assert!((sol.dr.ghx[(i, j)] - ghx[(i, j)]).abs() < 1e-9, "ghx({i},{j})");
        }
    }
    // dr.ghud (exo_det_length = 4)
    assert_eq!(r["exo_det_length"].as_u64().unwrap(), 4);
    let len = dynare::deterministic::exo_det::exo_det_length(&mf);
    assert_eq!(len, 4);
    let g = dynare::deterministic::exo_det::ghud(m, &sol.dr, &sol.dr.ys, &sol.params, len);
    for (k, gr) in r["ghud"].as_array().unwrap().iter().enumerate() {
        let gr = vecf(gr);
        for i in 0..n {
            assert!((g[k][(i, 0)] - gr[i]).abs() < 1e-10, "ghud{{{}}}({i}): {} vs {}", k + 1, g[k][(i, 0)], gr[i]);
        }
    }
    // forecast with the deterministic path
    let tmp = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp/q1_xdet");
    std::fs::create_dir_all(&tmp).unwrap();
    let opts = vec![
        ("periods".to_string(), dynare::modfile::OptValue::Num(8.0)),
        ("nograph".to_string(), dynare::modfile::OptValue::Flag),
    ];
    let f = dynare::forecast::commands::forecast_command(
        &mf,
        None,
        &opts,
        &[],
        &tmp,
        "q1_xdet",
        &tmp,
        &dynare::forecast::commands::ForecastOrder::first(),
    )
    .unwrap();
    for (key, mat) in [("Mean", &f.mean), ("HPDinf", &f.lower), ("HPDsup", &f.upper)] {
        for (c, name) in f.names.iter().enumerate() {
            let want = vecf(&r["forecast"][key][name.as_str()]);
            for h in 0..8 {
                assert!((mat[(h, c)] - want[h]).abs() < 1e-9, "{key}.{name}[{h}]: {} vs {}", mat[(h, c)], want[h]);
            }
        }
    }
}
