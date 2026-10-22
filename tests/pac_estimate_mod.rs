//! `pac.estimate.nls` / `pac.estimate.iterative_ols` written in a .mod file (structure assigned
//! inline, dseries loaded from a CSV), against the Dynare 7.1 results of reference/pac.

use dynare::modfile::{Command, OptValue, parse_mod_file};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/pac")
}

fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![],
    }
}

fn setup(name: &str, tag: &str, calls: &str) -> (PathBuf, Value) {
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(refdir().join(format!("ref_{name}.json"))).unwrap()).unwrap();
    let dir = std::env::temp_dir().join(format!("pac_est_mod_{tag}"));
    std::fs::create_dir_all(&dir).unwrap();
    let names: Vec<String> =
        r["sim"]["names"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
    let mut csv = format!(",{}\n", names.join(","));
    for (i, row) in r["sim"]["data"].as_array().unwrap().iter().enumerate() {
        let vals: Vec<String> = row.as_array().unwrap().iter().map(|v| format!("{:e}", v.as_f64().unwrap())).collect();
        csv.push_str(&format!("{}Q{},{}\n", 2000 + i / 4, i % 4 + 1, vals.join(",")));
    }
    std::fs::write(dir.join("sim.csv"), csv).unwrap();
    let src = std::fs::read_to_string(refdir().join(format!("{name}.mod"))).unwrap();
    // drop the commands of the reference file, keep the model
    let model: Vec<&str> = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("pac.") && !l.trim_start().starts_with("stoch_simul"))
        .collect();
    let mod_text =
        format!("{}\npac.initialize('pacman');\npac.update.expectation('pacman');\n{calls}\n", model.join("\n"));
    std::fs::write(dir.join("m.mod"), mod_text).unwrap();
    (dir, r)
}

fn run_all(dir: &Path) -> dynare::modfile::ModFile {
    let mut mf = parse_mod_file(&dir.join("m.mod"), &[]).unwrap();
    for ci in 0..mf.commands.len() {
        if let Command::Other(n, o) = mf.commands[ci].clone()
            && dynare::varmodels::handles(&n)
        {
            let mut o = o;
            o.push(("dir".into(), OptValue::Str(dir.to_string_lossy().into_owned())));
            dynare::varmodels::run(&mut mf, &n, &o).unwrap();
        }
    }
    mf
}

const SETUP: &str = "clear eparams;\neparams.e_c_m = .9;\neparams.c_z_1 = .5;\neparams.c_z_2 = .2;\nedata = dseries('sim.csv');\nedata.ez = dseries(NaN, dates('2000Q1'));\nrg = dates('2005Q1'):dates('2005Q1')+500;\n";

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{what}: {x} vs {y}");
    }
}

#[test]
fn nls_in_mod_file_matches_matlab() {
    let (dir, r) = setup("pc2", "nls", &format!("{SETUP}pac.estimate.nls('zpac', eparams, edata, rg, 'GaussNewton');"));
    let mf = run_all(&dir);
    close(&mf.model.params, &flat(&r["nls"]["params_after"]), 1e-6, "params after NLS");
}

#[test]
fn iterative_ols_in_mod_file_matches_matlab() {
    let (dir, r) = setup("pc2", "ols", &format!("{SETUP}pac.estimate.iterative_ols('zpac', eparams, edata, rg);"));
    let mf = run_all(&dir);
    close(&mf.model.params, &flat(&r["ols"]["params_after"]), 1e-6, "params after OLS");
}

#[test]
fn bounded_optimizer_keeps_error_correction_in_unit_interval() {
    let (dir, r) = setup(
        "pc2",
        "lsq",
        &format!("{SETUP}eparams.e_c_m = 1.5;\npac.estimate.nls('zpac', eparams, edata, rg, 'lsqnonlin');"),
    );
    let mf = run_all(&dir);
    let k = mf.model.param_names.iter().position(|n| n == "e_c_m").unwrap();
    let e = mf.model.params[k];
    assert!((0.0..=1.0).contains(&e), "e_c_m = {e}");
    // the unconstrained optimum is interior, so the bounded one is the same
    let want = flat(&r["nls"]["estimator"])[0];
    assert!((e - want).abs() < 1e-5, "{e} vs {want}");
}
