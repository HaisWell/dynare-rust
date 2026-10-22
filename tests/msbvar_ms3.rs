//! MS-SBVAR part 3: real-time smoothed probabilities (rts_small.out written by Dynare 7.2), figures of the
//! ms_* commands, `save_draws`.

use std::path::Path;
use std::sync::Arc;

use dynare::linalg::Mat;
use dynare::modfile::{Command, parse_mod_file};
use dynare::msbvar::hamilton;
use dynare::msbvar::ms_estimation::{MsOptions, MsSetup};
use dynare::msbvar::prepare_sample;
use serde_json::Value;

fn dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/msbvar/ms")
}

fn load(f: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join(f)).unwrap()).unwrap()
}

fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn matf(v: &Value) -> Mat {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(vecf).collect();
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

/// Setup of ms_small and Dynare's mode in this port's basis.
fn setup_and_theta() -> (Arc<MsSetup>, Vec<f64>) {
    let mf = parse_mod_file(&dir().join("ms_small.mod"), &[]).unwrap();
    let Some(Command::Other(_, o)) =
        mf.commands.iter().find(|c| matches!(c, Command::Other(n, _) if n == "ms_estimation"))
    else {
        panic!("no ms_estimation")
    };
    let p = prepare_sample(&mf, o, &dir()).unwrap();
    let mo = MsOptions::from_options(o).unwrap();
    let s = MsSetup::build(&p.z, &p.decl, &p.restr, &p.settings, &mo, (p.opts.freq, 1960, 1)).unwrap();
    let m = &load("ref_ms.json")["small"];
    let mut th = vecf(&m["theta"]);
    for j in 0..s.nvar {
        let (u_ml, v_ml) = (matf(&m["U"][j]), matf(&m["V"][j]));
        for &o in &s.layout.b[j] {
            let b = Mat::from_column_slice(s.n0[j], 1, &th[o..o + s.n0[j]]);
            let nb = s.ui[j].transpose() * &u_ml * b;
            th[o..o + s.n0[j]].copy_from_slice(nb.as_slice());
        }
        for &o in &s.layout.g[j] {
            let g = Mat::from_column_slice(s.np[j], 1, &th[o..o + s.np[j]]);
            let ng = s.vi[j].transpose() * &v_ml * g;
            th[o..o + s.np[j]].copy_from_slice(ng.as_slice());
        }
    }
    (Arc::new(s), th)
}

#[test]
fn real_time_smoothed_matches_dynare() {
    let (s, th) = setup_and_theta();
    let (r, f) = {
        let (r, f) = s.filter(&th).unwrap();
        (r, f)
    };
    let m = hamilton::real_time_smoothed(&f, &r.q);
    let text = std::fs::read_to_string(dir().join("rts_small.out")).unwrap();
    let mut worst: f64 = 0.0;
    let mut n = 0;
    for (k, l) in text.lines().enumerate() {
        for (j, c) in l.split(',').enumerate() {
            let c = c.trim();
            if c.is_empty() {
                if j < m.ncols() {
                    assert!(m[(k, j)].is_nan(), "blank cell ({k},{j}) must be NaN");
                }
                continue;
            }
            let want: f64 = c.parse().unwrap();
            worst = worst.max((m[(k, j)] - want).abs());
            n += 1;
        }
    }
    assert_eq!(n, 115 * 116 / 2 * 2);
    assert!(worst < 2e-6, "real-time smoothed probabilities differ by {worst:e}");
}

fn run_dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("msbvar_ms3_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(dir().join("d2.csv"), d.join("d2.csv")).unwrap();
    d
}

fn run(d: &Path, extra: &str) -> Vec<std::path::PathBuf> {
    let mut text = std::fs::read_to_string(dir().join("ms_small.mod")).unwrap();
    text.push_str(extra);
    let f = d.join("m.mod");
    std::fs::write(&f, text).unwrap();
    let mf = parse_mod_file(&f, &[]).unwrap();
    for c in &mf.commands {
        if let Command::Other(n, _) = c {
            if dynare::msbvar::ms_commands::handles(n) {
                dynare::msbvar::ms_commands::run_ms_command(&mf, c, d, "m", 3).unwrap();
            }
        }
    }
    let mut files = vec![];
    fn walk(p: &Path, out: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(p).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() { walk(&p, out) } else { out.push(p) }
        }
    }
    walk(d, &mut files);
    files
}

#[test]
fn figures_and_flat_draws() {
    let d = run_dir("fig");
    let files = run(
        &d,
        "ms_simulation(mh_replic=300, drop=100, save_draws);\nms_irf(horizon=6, regimes, graph_format=svg);\nms_forecast(horizon=4, error_band_percentiles=[0.16, 0.5, 0.84], data_obs_nbr=8, graph_format=svg);\nms_variance_decomposition(horizon=6, graph_format=svg);\nms_compute_probabilities(graph_format=svg);\nms_compute_probabilities(real_time_smoothed);\n",
    );
    let has = |s: &str| files.iter().any(|p| p.file_name().unwrap().to_string_lossy().contains(s));
    assert!(has("MS-IRF-regime_0.svg") && has("MS-IRF-regime_1.svg"));
    assert!(has("MS-Forecast-ergodic.svg"));
    assert!(has("MS-VarianceDecomposition-ergodic.svg"));
    assert!(has("MS-Probabilities-Chain-1.svg"));
    assert!(has("real_time_smoothed_m.out"));
    let head = std::fs::read_to_string(d.join("draws_header_m.out")).unwrap();
    let draws = std::fs::read_to_string(d.join("draws_m.out")).unwrap();
    let rows: Vec<&str> = draws.lines().collect();
    assert_eq!(rows.len(), 300);
    assert_eq!(rows[0].split_whitespace().count(), head.split_whitespace().count());
    assert_eq!(head.split_whitespace().count(), 36);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn nograph_writes_no_figures() {
    let d = run_dir("nograph");
    let files = run(&d, "ms_irf(horizon=4, nograph);\nms_compute_probabilities(nograph);\n");
    assert!(!files.iter().any(|p| p.file_name().unwrap().to_string_lossy().starts_with("MS-IRF")));
    assert!(files.iter().any(|p| p.file_name().unwrap().to_string_lossy().starts_with("ir_percentiles_ergodic")));
    let _ = std::fs::remove_dir_all(&d);
}
