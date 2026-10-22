//! `shocks` / `mshocks` / `shock_paths` against Dynare 7.2 (reference/shocks/ref_shocks.json and
//! sh_stoch_*.log, written by reference/shocks/dump_shocks.m):
//! - dates in `periods` with `first_simulation_period`, `last_simulation_period` or both,
//!   vector values `(xx)` from the MATLAB workspace, `shocks(overwrite)`, `mshocks` (sh_det 1-3);
//! - expectation errors with date-valued `learnt_in` (shocks, mshocks, endval, controlled paths)
//!   (sh_det 4);
//! - `shock_paths` with dates, dseries / table / single-row table databases, controlled
//!   stanzas with dates (sh_sp 1), integer periods and tables (sh_sp 2), and with expectation
//!   errors: `learnt_in` integers and dates, `prev.` / `learnt_in(...)` scopes, `overwrite`
//!   (sh_sp 3);
//! - stochastic declarations: correlations evaluated at the end of their block, measurement
//!   errors, `shocks(overwrite)`, the coskewness and measurement-error tables (sh_stoch);
//! - closed skew normal shock draws (`rand_multivariate_csn.m`) fed with MATLAB's normals.

use std::path::{Path, PathBuf};

use dynare::deterministic::DetState;
use dynare::linalg::{Mat, Vector};
use dynare::modfile::{Command, OptValue, parse_mod_file};
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/shocks")
}

fn reference() -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join("ref_shocks.json")).unwrap()).unwrap()
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| match r {
            Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect(),
            x => vec![x.as_f64().unwrap_or(f64::NAN)],
        })
        .collect()
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect(),
        x => vec![x.as_f64().unwrap()],
    }
}

/// Runs the deterministic commands of `file` with `-DCASE=case`.
fn run(file: &str, case: u32) -> (dynare::modfile::ModFile, DetState) {
    let mf = parse_mod_file(&dir().join(file), &[("CASE".into(), case.to_string())]).unwrap();
    let out = std::env::temp_dir().join(format!("dynare_rs_shocks_{file}_{case}"));
    std::fs::create_dir_all(&out).unwrap();
    let mut det = DetState::new(&mf).unwrap();
    for (ci, c) in mf.commands.iter().enumerate() {
        match c {
            Command::Steady(o) => {
                let mut o = o.clone();
                o.push(("noprint".into(), OptValue::Flag));
                det.steady(&mf.model, &o).unwrap()
            }
            Command::Endval(v) => det.endval(&mf.model, v).unwrap(),
            Command::Deterministic(name, o, v) => {
                let mut o = o.clone();
                if name != "shock_paths" {
                    o.push(("noprint".into(), OptValue::Flag));
                }
                det.command(&mf, name, &o, v, &out, "sh").unwrap()
            }
            Command::Other(n, o) if n == "perfect_foresight_with_expectation_errors_setup" => {
                dynare::ep::pfwee::setup(&mf, &mut det, o, &dir()).unwrap();
                det.simul.noprint = true;
                det.simul.verbosity = 0;
            }
            Command::Other(n, _) if dynare::ep::handles(n) => {
                dynare::ep::command(&mf, &mut det, ci, &out, "sh", 0).unwrap()
            }
            _ => {}
        }
    }
    (mf, det)
}

fn check(name: &str, file: &str, case: u32) {
    let r = &reference()[name];
    assert!(r.get("failure").is_none(), "{name}: Dynare failed: {}", r["failure"]);
    let (_, det) = run(file, case);
    let want = rows(&r["endo_simul"]);
    let got = det.endo_simul.as_ref().unwrap();
    assert_eq!(got.ncols(), want[0].len(), "{name}: endo_simul columns");
    let mut err = 0.0f64;
    for (j, row) in want.iter().enumerate() {
        for (t, w) in row.iter().enumerate() {
            err = err.max((got[(j, t)] - w).abs() / w.abs().max(1.0));
        }
    }
    let wx = rows(&r["exo_simul"]);
    let gx = det.exo_simul.as_ref().unwrap();
    assert_eq!(gx.nrows(), wx.len(), "{name}: exo_simul rows");
    let mut errx = 0.0f64;
    for (t, row) in wx.iter().enumerate() {
        for (k, w) in row.iter().enumerate() {
            errx = errx.max((gx[(t, k)] - w).abs());
        }
    }
    let ss = vecf(&r["steady_state"]);
    let errss = ss.iter().zip(&det.steady_state).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let xs = vecf(&r["exo_steady_state"]);
    let errxs = xs.iter().zip(&det.exo_steady_state).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let mut errinfo = 0.0f64;
    if let Some(ti) = r.get("terminal_info") {
        let info = det.pfwee.as_ref().unwrap();
        for (k, row) in rows(ti).iter().enumerate() {
            for (p, w) in row.iter().enumerate() {
                errinfo = errinfo.max((info.terminal[(k, p)] - w).abs());
            }
        }
        // shocks_info: exo x periods x periods, jsonencode nests [exo][period][info]
        for (k, a) in r["shocks_info"].as_array().unwrap().iter().enumerate() {
            for (t, b) in a.as_array().unwrap().iter().enumerate() {
                for (p, w) in vecf(b).iter().enumerate() {
                    errinfo = errinfo.max((info.shocks[p][(k, t)] - w).abs());
                }
            }
        }
    }
    println!("{name:6} endo {err:.2e}  exo {errx:.2e}  ss {errss:.2e}  exo ss {errxs:.2e}  info {errinfo:.2e}");
    assert!(
        err < 1e-9 && errx < 1e-13 && errss < 1e-9 && errxs < 1e-14 && errinfo < 1e-13,
        "{name}: endo {err} exo {errx} ss {errss} exo ss {errxs} info {errinfo}"
    );
}

#[test]
fn det_shocks_with_dates_first_simulation_period() {
    check("det1", "sh_det.mod", 1);
}

#[test]
fn det_shocks_with_dates_last_simulation_period() {
    check("det2", "sh_det.mod", 2);
}

#[test]
fn det_shocks_with_dates_first_and_last() {
    check("det3", "sh_det.mod", 3);
}

#[test]
fn expectation_errors_with_date_learnt_in() {
    check("det4", "sh_det.mod", 4);
}

#[test]
fn shock_paths_dates_and_databases() {
    check("sp1", "sh_sp.mod", 1);
}

#[test]
fn shock_paths_integer_periods_table() {
    check("sp2", "sh_sp.mod", 2);
}

#[test]
fn shock_paths_expectation_errors() {
    check("sp3", "sh_sp.mod", 3);
}

/// Sigma_e / H after the stochastic blocks, and the model-summary tables.
#[test]
fn stochastic_declarations() {
    for case in [1u32, 2] {
        let r = &reference()[format!("stoch{case}").as_str()];
        let mf = parse_mod_file(&dir().join("sh_stoch.mod"), &[("CASE".into(), case.to_string())]).unwrap();
        let se = rows(&r["Sigma_e"]);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (mf.sigma_e[(i, j)] - se[i][j]).abs() < 1e-18,
                    "case {case}: Sigma_e({i},{j}) {} vs {}",
                    mf.sigma_e[(i, j)],
                    se[i][j]
                );
            }
        }
        let h = rows(&r["H"]);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (mf.h[(i, j)] - h[i][j]).abs() < 1e-18,
                    "case {case}: H({i},{j}) {} vs {}",
                    mf.h[(i, j)],
                    h[i][j]
                );
            }
        }
        let text = dynare::analysis::skew::summary_tables(&mf.model, &mf.q3.skew_e, &mf.h, &mf.varobs);
        let log = std::fs::read_to_string(dir().join(format!("sh_stoch_{case}.log"))).unwrap();
        let start = log.find("COSKEWNESS").unwrap();
        let theirs: Vec<&str> = log[start..].lines().map(str::trim_end).filter(|l| !l.is_empty()).collect();
        let ours: Vec<&str> = text.lines().map(str::trim_end).filter(|l| !l.is_empty()).collect();
        assert_eq!(ours, theirs, "case {case}");
    }
}

/// rand_multivariate_csn.m with MATLAB's standard normals (rng(7)).
#[test]
fn closed_skew_normal_draws() {
    let r = &reference()["csn"];
    let m = |v: &Value| {
        let rr = rows(v);
        Mat::from_fn(rr.len(), rr[0].len(), |i, j| rr[i][j])
    };
    let (sigma, gamma) = (m(&r["Sigma"]), m(&r["Gamma"]));
    let mu = Vector::from_vec(vecf(&r["mu"]));
    // csn_update_specification
    let se = Mat::from_diagonal(&Vector::from_vec(vec![0.01f64.powi(2), 0.02f64.powi(2), 0.005f64.powi(2)]));
    let csn = dynare::analysis::skew::csn_specification(&se, &[(0, 0, 0, 0.6), (2, 2, 2, -0.4)]).unwrap();
    let d = ((&csn.sigma - &sigma).abs().max(), (&csn.gamma - &gamma).abs().max(), (&csn.mu - &mu).abs().max());
    println!("csn_update_specification: {d:?}");
    assert!(d.0 < 1e-18 && d.1 < 1e-11 && d.2 < 1e-17, "{d:?}");
    let z = vecf(&r["normals"]);
    let mut k = 0;
    let mut randn = || {
        k += 1;
        z[k - 1]
    };
    let x = dynare::analysis::skew::rand_multivariate_csn(
        1000, &csn.mu, &csn.sigma, &csn.gamma, &csn.nu, &csn.delta, &mut randn,
    )
    .unwrap();
    let want = m(&r["draws"]);
    let err = (&x - &want).abs().max();
    println!("csn draws: max abs diff {err:.2e}");
    assert!(err < 1e-15, "{err}");
    // the ChaCha8 stream of stoch_simul: the sample moments of 1e5 draws (MATLAB: rng(8))
    let s = dynare::analysis::simulate::draw_skewed(&se, &[(0, 0, 0, 0.6), (2, 2, 2, -0.4)], 100000, 3).unwrap();
    let (wm, wv, ws) = (vecf(&r["mean"]), vecf(&r["var"]), vecf(&r["skewness"]));
    for i in 0..3 {
        let c: Vec<f64> = s.column(i).iter().copied().collect();
        let n = c.len() as f64;
        let mean = c.iter().sum::<f64>() / n;
        let var = c.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let m3 = c.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
        let skew = m3 / (var * (n - 1.0) / n).powf(1.5);
        println!(
            "shock {i}: mean {mean:.3e} ({:.3e}) var {var:.4e} ({:.4e}) skew {skew:.3} ({:.3})",
            wm[i], wv[i], ws[i]
        );
        let sd = se[(i, i)].sqrt();
        assert!((mean - wm[i]).abs() < 0.02 * sd, "mean {i}");
        assert!((var / wv[i] - 1.0).abs() < 0.02, "var {i}");
        assert!((skew - ws[i]).abs() < 0.05, "skew {i}");
    }
}

const HEAD: &str = "var c k; varexo x z; parameters a; a = 0.5;
model; c + k = k(-1)^a*x + z; 1/c = 0.95/c(+1)*a*k^(a-1)*x; end;
initval; x = 1; z = 0; k = 0.3; c = 0.7; end;
";

fn parse(body: &str) -> anyhow::Result<dynare::modfile::ModFile> {
    dynare::modfile::parse_mod_str(&format!("{HEAD}{body}"), &[])
}

/// `periods` with dates, ranges and lists; `values` with expressions and workspace vectors
/// (MATLAB expressions evaluated by dynare-rust).
#[test]
fn deterministic_shocks_syntax() {
    use dynare::dates::Date;
    let mf = parse(
        "xx = [0.1; 0.2];
         yy = transpose(linspace(1, 2, 3)) .* 2;
         zz = yy(2:3);
         shocks;
           var x; periods 1:2, 4:5 2001Q1:2001Q3 -1Q4; values (a*2) (xx) (yy) 3;
           var z; periods 7:8; values (zz);
         end;",
    )
    .unwrap();
    let d: Vec<(Vec<i64>, Vec<String>, f64)> =
        mf.det_shocks.iter().map(|s| (s.periods.clone(), s.dates.iter().map(Date::label).collect(), s.value)).collect();
    let q = |s: &str| vec![s.to_string()];
    let want: Vec<(Vec<i64>, Vec<String>, f64)> = vec![
        (vec![1, 2], vec![], 1.0),
        (vec![4], vec![], 0.1),
        (vec![5], vec![], 0.2),
        (vec![], q("2001Q1"), 2.0),
        (vec![], q("2001Q2"), 3.0),
        (vec![], q("2001Q3"), 4.0),
        (vec![], q("-1Q4"), 3.0),
        (vec![7], vec![], 3.0),
        (vec![8], vec![], 4.0),
    ];
    assert_eq!(d, want);
    // a vector value needs as many elements as periods in its range
    assert!(parse("xx = [1 2 3]; shocks; var x; periods 1:2; values (xx); end;").is_err());
    assert!(parse("shocks; var x; periods 1:2023Q1; values 1; end;").is_err());
}

/// `learnt_in` given as a date, space-separated options of mshocks / shock_paths, and the
/// heterogeneity option.
#[test]
fn learnt_in_and_options() {
    let mf = parse(
        "shocks(learnt_in = 2023Q2, overwrite); var x; periods 2023Q3; add 0.1; end;
         mshocks(learnt_in = 3 relative_to_initval); var x; periods 4; values 1.1; end;
         endval(learnt_in = 2024Q1); x += 0.1; end;
         shock_paths(overwrite learnt_in = 2023Q4); var z; periods 2023Q4:end; values prev.z + 0.1; end;",
    )
    .unwrap();
    let s = &mf.ep.learnt_shocks;
    assert_eq!(s[0].learnt_in_date.map(|d| d.label()), Some("2023Q2".into()));
    assert_eq!(s[0].dates[0].label(), "2023Q3");
    assert_eq!((s[1].learnt_in, s[1].kind), (3, dynare::ep::decl::LearntKind::MultiplyInitialSteadyState));
    assert_eq!(mf.ep.learnt_endval[0].learnt_in_date.map(|d| d.label()), Some("2024Q1".into()));
    let b = &mf.q1.shock_paths[0];
    assert!(b.overwrite && b.contains_date && b.contains_endval());
    assert_eq!(b.learnt_in_date.map(|d| d.label()), Some("2023Q4".into()));
    // heterogeneous-agent shocks are parsed (tests/heterogeneity.rs); an undeclared dimension is an error
    let e = parse("shocks(heterogeneity = households); var x; stderr 0.01; end;").unwrap_err().to_string();
    assert!(e.contains("Unknown heterogeneity dimension: households"), "{e}");
    let e = parse("shock_paths; var x; periods 1; values db.x; end;").unwrap_err().to_string();
    assert!(e.contains("unknown scope 'db'"), "{e}");
}

/// Correlations use the variances known at the end of their block; `shocks(overwrite)`
/// resets Sigma_e and H but not the skewness.
#[test]
fn stochastic_blocks_semantics() {
    let mf = parse(
        "varobs c k;
         shocks; var x = 4; var z; stderr 1; corr x, z = 0.5; var c = 1; var k = 4; corr c, k = 0.25; skew x = 0.2; end;
         shocks; var x = 9; end;",
    )
    .unwrap();
    assert_eq!((mf.sigma_e[(0, 0)], mf.sigma_e[(0, 1)], mf.sigma_e[(1, 0)]), (9.0, 1.0, 1.0));
    assert_eq!(mf.h[(0, 1)], 0.5);
    let mf = parse(
        "varobs c;
         shocks; var x = 4; var z = 1; corr x, z = 0.5; var c = 1; skew x = 0.2; end;
         shocks(overwrite); var z = 2; end;",
    )
    .unwrap();
    assert_eq!((mf.sigma_e[(0, 0)], mf.sigma_e[(0, 1)], mf.sigma_e[(1, 1)], mf.h[(0, 0)]), (0.0, 0.0, 2.0, 0.0));
    assert_eq!(mf.q3.skew_e.len(), 1);
}
