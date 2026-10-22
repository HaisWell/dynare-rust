//! `pac.estimate.*` in a .mod file: observed exogenous regressors are read from the data, a
//! variable missing from the data is an error (only the residual is not observed, as in
//! `pac.estimate.init`), the range forms (`dates('...')+n:dates('...')-m`, `2005Q1:2010Q4`) and
//! the optimizer names and bounds of `nls.m`. Data: the simulation of reference/pac/pc2.

use dynare::modfile::{Command, OptValue, parse_mod_file};
use dynare::varmodels::pac_estimate::nls_bounds;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/pac")
}

/// Writes `sim.csv` (the pc2 simulation; `edit(names, row)` may change it) and `m.mod` (pc2's
/// model with `eq_edit` applied to its text, then `calls`).
fn setup(
    tag: &str,
    edit: &dyn Fn(&[String], &mut Vec<f64>),
    drop_col: Option<&str>,
    eq_edit: (&str, &str),
    calls: &str,
) -> PathBuf {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(refdir().join("ref_pc2.json")).unwrap()).unwrap();
    let dir = std::env::temp_dir().join(format!("pac_est_{tag}"));
    std::fs::create_dir_all(&dir).unwrap();
    let names: Vec<String> =
        r["sim"]["names"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
    let keep: Vec<usize> = (0..names.len()).filter(|&j| Some(names[j].as_str()) != drop_col).collect();
    let mut csv = format!(",{}\n", keep.iter().map(|&j| names[j].as_str()).collect::<Vec<_>>().join(","));
    for (i, row) in r["sim"]["data"].as_array().unwrap().iter().enumerate() {
        let mut row: Vec<f64> = row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        edit(&names, &mut row);
        let vals: Vec<String> = keep.iter().map(|&j| format!("{:e}", row[j])).collect();
        csv.push_str(&format!("{}Q{},{}\n", 2000 + i / 4, i % 4 + 1, vals.join(",")));
    }
    std::fs::write(dir.join("sim.csv"), csv).unwrap();
    let src = std::fs::read_to_string(refdir().join("pc2.mod")).unwrap().replace(eq_edit.0, eq_edit.1);
    let model: Vec<&str> = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("pac.") && !l.trim_start().starts_with("stoch_simul"))
        .collect();
    let mod_text =
        format!("{}\npac.initialize('pacman');\npac.update.expectation('pacman');\n{SETUP}{calls}\n", model.join("\n"));
    std::fs::write(dir.join("m.mod"), mod_text).unwrap();
    dir
}

const SETUP: &str = "clear eparams;\neparams.e_c_m = .9;\neparams.c_z_1 = .5;\neparams.c_z_2 = .2;\nedata = dseries('sim.csv');\nedata.ez = dseries(NaN, dates('2000Q1'));\n";
const NO_EDIT: (&str, &str) = ("", "");

fn run_all(dir: &Path) -> anyhow::Result<dynare::modfile::ModFile> {
    let mut mf = parse_mod_file(&dir.join("m.mod"), &[])?;
    for ci in 0..mf.commands.len() {
        if let Command::Other(n, o) = mf.commands[ci].clone()
            && dynare::varmodels::handles(&n)
        {
            let mut o = o;
            o.push(("dir".into(), OptValue::Str(dir.to_string_lossy().into_owned())));
            dynare::varmodels::run(&mut mf, &n, &o)?;
        }
    }
    Ok(mf)
}

fn estimates(mf: &dynare::modfile::ModFile) -> Vec<f64> {
    ["e_c_m", "c_z_1", "c_z_2"]
        .iter()
        .map(|n| mf.model.params[mf.model.param_names.iter().position(|x| x == n).unwrap()])
        .collect()
}

fn no_edit(_: &[String], _: &mut Vec<f64>) {}

#[test]
fn exogenous_regressor_is_read_from_the_data() {
    // zpac gets `+ ex1` and the data's ex1 holds the true residual of zpac: the equation then fits
    // exactly, the estimates are the calibration (pc2: .1, .7, -.3); with ex1 dropped they would
    // be the NLS estimates of the reference instead.
    let edit = |names: &[String], row: &mut Vec<f64>| {
        let (i, j) = (names.iter().position(|n| n == "ex1").unwrap(), names.iter().position(|n| n == "ez").unwrap());
        row[i] = row[j];
    };
    let rg = "dates('2005Q1'):dates('2005Q1')+500";
    for (tag, call) in [
        ("exo_nls", format!("pac.estimate.nls('zpac', eparams, edata, {rg}, 'GaussNewton');")),
        ("exo_ols", format!("pac.estimate.iterative_ols('zpac', eparams, edata, {rg});")),
    ] {
        let dir =
            setup(tag, &edit, None, ("pac_expectation(pacman) + ez;", "pac_expectation(pacman) + ex1 + ez;"), &call);
        let e = estimates(&run_all(&dir).unwrap());
        for (x, y) in e.iter().zip([0.1, 0.7, -0.3]) {
            assert!((x - y).abs() < 1e-5, "{tag}: {e:?}");
        }
    }
}

#[test]
fn variable_missing_from_the_data_is_an_error() {
    let call = "pac.estimate.nls('zpac', eparams, edata, dates('2005Q1'):dates('2005Q1')+500, 'GaussNewton');";
    let dir = setup("missing", &no_edit, Some("x1"), NO_EDIT, call);
    let err = run_all(&dir).unwrap_err().to_string();
    assert!(err.contains("x1") && err.contains("not in the database"), "{err}");
    // an exogenous regressor missing too: two unobserved exogenous variables
    let dir = setup(
        "missing_exo",
        &no_edit,
        Some("ex1"),
        ("pac_expectation(pacman) + ez;", "pac_expectation(pacman) + ex1 + ez;"),
        call,
    );
    let err = run_all(&dir).unwrap_err().to_string();
    assert!(err.contains("more than one residual"), "{err}");
}

#[test]
fn range_offsets_and_date_literals() {
    let run = |tag: &str, rg: &str| {
        estimates(
            &run_all(&setup(
                tag,
                &no_edit,
                None,
                NO_EDIT,
                &format!("pac.estimate.nls('zpac', eparams, edata, {rg}, 'GaussNewton');"),
            ))
            .unwrap(),
        )
    };
    let base = run("rg_plain", "dates('2006Q1'):dates('2010Q2')");
    let full = run("rg_full", "dates('2005Q1'):dates('2005Q1')+500");
    assert!(base.iter().zip(&full).any(|(a, b)| (a - b).abs() > 1e-6), "the subsample must change the estimates");
    for (tag, rg) in [
        ("rg_off", "dates('2005Q1')+4:dates('2010Q4')-2"),
        ("rg_lit", "2006Q1:2010Q2"),
        ("rg_lit_off", "2005Q3+2:2010Q1+1"),
    ] {
        let e = run(tag, rg);
        assert_eq!(e, base, "{rg}");
    }
}

#[test]
fn range_stored_in_a_variable() {
    // `rg = <range>;` then `pac.estimate.*(..., rg, ...)`: the preprocessor passes the assignment
    // through as a MATLAB statement, so the variable holds the same dates range as the literal.
    let run = |tag: &str, assign: &str, est: &str| {
        let calls = format!(
            "{assign}\n{est}('zpac', eparams, edata, rg{});",
            if est.ends_with("nls") { ", 'GaussNewton'" } else { "" }
        );
        estimates(&run_all(&setup(tag, &no_edit, None, NO_EDIT, &calls)).unwrap())
    };
    let direct = |est: &str| {
        let opt = if est.ends_with("nls") { ", 'GaussNewton'" } else { "" };
        estimates(
            &run_all(&setup(
                "rgv_direct",
                &no_edit,
                None,
                NO_EDIT,
                &format!("{est}('zpac', eparams, edata, dates('2006Q1'):dates('2010Q2'){opt});"),
            ))
            .unwrap(),
        )
    };
    for est in ["pac.estimate.nls", "pac.estimate.iterative_ols"] {
        let base = direct(est);
        for (tag, assign) in [
            ("rgv_dates", "rg = dates('2006Q1'):dates('2010Q2');"),
            ("rgv_off", "rg = dates('2005Q1')+4:dates('2010Q4')-2;"),
            ("rgv_lit", "rg = 2006Q1:2010Q2;"),
            ("rgv_lit_off", "rg = 2005Q3+2:2010Q1+1;"),
        ] {
            assert_eq!(run(tag, assign, est), base, "{est}: {assign}");
        }
    }
}

#[test]
fn optimizers_of_nls() {
    let call = |o: &str| format!("pac.estimate.nls('zpac', eparams, edata, dates('2005Q1'):dates('2005Q1')+500, {o});");
    let err = run_all(&setup("opt_bad", &no_edit, None, NO_EDIT, &call("'newton'"))).unwrap_err().to_string();
    assert!(err.contains("newton is not an implemented optimization routine"), "{err}");
    // nls.m's particleswarm bounds: the lower bound is assigned twice, [1, Inf]
    let e = estimates(&run_all(&setup("opt_ps", &no_edit, None, NO_EDIT, &call("'particleswarm'"))).unwrap());
    assert!(e[0] >= 1.0, "{e:?}");
    assert_eq!(nls_bounds(Some("particleswarm"), &[]).unwrap(), Some((1.0, f64::INFINITY)));
    assert_eq!(nls_bounds(Some("lsqnonlin"), &[]).unwrap(), Some((0.0, 1.0)));
    // levenberg-marquardt does not handle bounds
    let lm = ["Algorithm".to_string(), "levenberg-marquardt".to_string()];
    assert_eq!(nls_bounds(Some("lsqnonlin"), &lm).unwrap(), None);
    assert_eq!(nls_bounds(None, &[]).unwrap(), None);
    // the optional arguments reach the command
    let dir = setup("opt_lm", &no_edit, None, NO_EDIT, &call("'lsqnonlin', 'Algorithm', 'levenberg-marquardt'"));
    let mf = parse_mod_file(&dir.join("m.mod"), &[]).unwrap();
    let args: Vec<String> = mf
        .commands
        .iter()
        .filter_map(|c| if let Command::Other(n, o) = c { (n == "pac.estimate.nls").then_some(o) } else { None })
        .flat_map(|o| {
            o.iter().filter_map(|(k, v)| {
                if let ("optimizer_arg", OptValue::Str(s)) = (k.as_str(), v) { Some(s.clone()) } else { None }
            })
        })
        .collect();
    assert_eq!(args, lm);
    let e = estimates(&run_all(&dir).unwrap());
    assert!((e[0] - 0.10115929138123508).abs() < 1e-5, "{e:?}");
}
