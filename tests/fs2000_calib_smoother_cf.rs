//! Dynare 7.2 test fs2000_calib_smoother_cf: `conditional_forecast`
//! (parameter_set = calibration) after `calib_smoother` on fs2000. Reference from
//! reference/fs2000_cf/dump.m (Dynare 7.2, randn shadowed by zeros so the forecast Mean is
//! the zero-shock path): the conditional / unconditional paths and the smoothed variables.
mod common;

use std::collections::HashMap;
use std::path::Path;

use dynare::driver::{RunOptions, run};
use serde_json::Value;

fn csv(p: &Path) -> Vec<Vec<String>> {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    s.lines().map(|l| l.split(',').map(str::to_string).collect()).collect()
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: rust {a} vs dynare {b}");
}

#[test]
fn fs2000_calib_smoother_cf() {
    let refdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/fs2000_cf");
    let r: Value = common::json_ref(refdir.join("ref.json"));
    let dir = common::temp_dir("fs2000_cf");
    for f in ["fs2000_model.inc", "fs2000_calib_smoother_cf.mod", "fs2000_cf_data.csv"] {
        std::fs::copy(refdir.join(f), dir.join(f)).unwrap();
    }
    run(&dir.join("fs2000_calib_smoother_cf.mod"), &RunOptions { quiet: true, ..Default::default() }).unwrap();
    let out = dir.join("fs2000_calib_smoother_cf/Output");

    // smoothed variables of calib_smoother
    let sm = csv(&out.join("fs2000_calib_smoother_cf_calib_smoothed_variables.csv"));
    let col: HashMap<&str, usize> = sm[0].iter().enumerate().map(|(j, s)| (s.as_str(), j)).collect();
    for v in ["m", "P", "c", "y"] {
        let want = r["smoothed"][v].as_array().unwrap();
        assert_eq!(want.len(), sm.len() - 1, "{v}: nobs");
        for (t, w) in want.iter().enumerate() {
            close(sm[t + 1][col[v]].parse().unwrap(), w.as_f64().unwrap(), 1e-8, &format!("smoothed {v}[{t}]"));
        }
    }

    // conditional / unconditional forecast (zero-shock paths)
    let cf = csv(&out.join("fs2000_calib_smoother_cf_conditional_forecast.csv"));
    let h = &cf[0];
    let ix = |n: &str| h.iter().position(|s| s == n).unwrap();
    let (iv, ih, ic, iu) = (ix("variable"), ix("horizon"), ix("cond_path"), ix("uncond_path"));
    let mut n = 0;
    for row in &cf[1..] {
        let (v, t): (&str, usize) = (&row[iv], row[ih].parse().unwrap());
        let Some(c) = r["cond_mean"].get(v) else { continue };
        close(row[ic].parse().unwrap(), c[t].as_f64().unwrap(), 1e-8, &format!("cond {v}[{t}]"));
        close(row[iu].parse().unwrap(), r["uncond_mean"][v][t].as_f64().unwrap(), 1e-8, &format!("uncond {v}[{t}]"));
        n += 1;
    }
    assert_eq!(n, 14 * 11);
    // the constrained path is hit
    let gp = &r["cond_mean"]["gp_obs"];
    assert!((gp[1].as_f64().unwrap() - 1.012).abs() < 1e-12 && (gp[5].as_f64().unwrap() - 1.008).abs() < 1e-12);
}
