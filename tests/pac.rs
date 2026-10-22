//! PAC models against Dynare 7.2 (reference/pac, dump.m): `pac_model`, `pac_target_info`,
//! `pac_expectation`, `pac.initialize`, `pac.update.expectation`, `pac.mce.parameters`
//! (the h / growth neutrality / mce_alpha parameters and the first-order solution of the
//! models that use them), and `pac.estimate.nls` / `pac.estimate.iterative_ols` on data
//! simulated by `simul_backward_model` in MATLAB.

use dynare::analysis::{calibrated_steady_state, solve_calibration};
use dynare::modfile::{Command, ModFile, parse_mod_file};
use dynare::varmodels::pac_estimate::{PacData, estimate_iterative_ols, estimate_nls, pac_residuals};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/pac")
}

fn reference(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(refdir().join(format!("ref_{name}.json"))).unwrap()).unwrap()
}

fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::Null => vec![f64::NAN],
        _ => vec![],
    }
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
}

/// Parses `name.mod` and runs its `var_expectation.*` / `pac.*` statements in file order.
fn run_commands(name: &str) -> ModFile {
    let mut mf = parse_mod_file(&refdir().join(format!("{name}.mod")), &[]).unwrap();
    for ci in 0..mf.commands.len() {
        if let Command::Other(n, o) = mf.commands[ci].clone()
            && dynare::varmodels::handles(&n)
        {
            dynare::varmodels::run(&mut mf, &n, &o).unwrap();
        }
    }
    mf
}

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{what}[{i}]: {x} vs {y}");
    }
}

fn pac_param(n: &str) -> bool {
    n.starts_with("h_") || n.contains("pac_growth_neutrality") || n.starts_with("mce_alpha")
}

/// Declared parameters by name; the parameters created for the PAC models (whose names contain
/// Dynare's auxiliary variable numbering) by position.
fn check_params(name: &str, mf: &ModFile, r: &Value) {
    let rn = strings(&r["param_names"]);
    let rv = flat(&r["params"]);
    assert_eq!(rn.len(), mf.model.param_names.len(), "{name}: number of parameters");
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (i, n) in rn.iter().enumerate() {
        if pac_param(n) {
            a.push(mf.model.params[i]);
            b.push(rv[i]);
            assert!(
                pac_param(&mf.model.param_names[i]),
                "{name}: parameter {i} is {} not {n}",
                mf.model.param_names[i]
            );
        } else {
            assert_eq!(&mf.model.param_names[i], n, "{name}: parameter name");
            close(&[mf.model.params[i]], &[rv[i]], 1e-12, &format!("{name}: {n}"));
        }
    }
    assert!(!a.is_empty());
    close(&a, &b, 1e-10, &format!("{name}: PAC parameters"));
}

#[test]
fn var_composite_target_and_growth() {
    let mf = run_commands("pc1");
    check_params("pc1", &mf, &reference("pc1"));
}

#[test]
fn trend_component_growth() {
    let mf = run_commands("pc2");
    check_params("pc2", &mf, &reference("pc2"));
}

#[test]
fn var_simple_target() {
    let mf = run_commands("pc3");
    check_params("pc3", &mf, &reference("pc3"));
}

#[test]
fn model_consistent_expectations_parameters() {
    let mf = run_commands("pc4");
    check_params("pc4", &mf, &reference("pc4"));
}

/// Steady state, ghu/ghx of the original variables and impulse responses (recursion on the
/// Rust decision rule) against stoch_simul.
fn solution_matches(name: &str) {
    let r = reference(name);
    let mf = run_commands(name);
    let endo = &mf.model.endo_names;
    let rendo = strings(&r["endo_names"]);
    let ys = calibrated_steady_state(&mf, &mf.model.params).unwrap();
    let rys = flat(&r["ys"]);
    let sol = solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
    let dr = &sol.dr;
    let n = endo.len();
    let nexo = mf.model.exo_names.len();
    let orig = r["orig_endo_nbr"].as_u64().unwrap() as usize;
    for k in 0..orig {
        let j = endo.iter().position(|e| *e == rendo[k]).unwrap();
        close(&[ys[j]], &[rys[k]], 1e-9, &format!("{name}: steady state {}", rendo[k]));
    }
    let rorder = flat(&r["order_var"]);
    let rstate = flat(&r["state_var"]);
    let (rghx, rghu) = (flat(&r["ghx"]), flat(&r["ghu"]));
    let nsp = rstate.len();
    // jsonencode gives rows
    for (i, &oi) in rorder.iter().enumerate() {
        let vname = &rendo[oi as usize - 1];
        if vname.starts_with("AUX_") {
            continue;
        }
        let ri = dr.order_var.iter().position(|&e| endo[e] == *vname).unwrap();
        for s in 0..nexo {
            close(&[dr.ghu[(ri, s)]], &[rghu[i * nexo + s]], 1e-9, &format!("{name}: ghu {vname} {s}"));
        }
        for (c, &sv) in rstate.iter().enumerate() {
            let cname = &rendo[sv as usize - 1];
            if cname.starts_with("AUX_") {
                continue;
            }
            let rc = dr.state_var.iter().position(|&e| endo[e] == *cname).unwrap();
            close(&[dr.ghx[(ri, rc)]], &[rghx[i * nsp + c]], 1e-9, &format!("{name}: ghx {vname} {cname}"));
        }
    }
    let nsv = dr.state_var.len();
    let irfs = r["irfs"].as_object().unwrap();
    for s in 0..nexo {
        let mut yprev = vec![0.0; n];
        for t in 0..8 {
            let mut yt = vec![0.0; n];
            for i in 0..n {
                let mut v = 0.0;
                for c in 0..nsv {
                    v += dr.ghx[(i, c)] * yprev[dr.state_var[c]];
                }
                if t == 0 {
                    v += dr.ghu[(i, s)] * mf.sigma_e[(s, s)].sqrt(); // one standard deviation, as stoch_simul
                }
                yt[dr.order_var[i]] = v;
            }
            for k in 0..orig {
                let key = format!("{}_{}", rendo[k], mf.model.exo_names[s]);
                let j = endo.iter().position(|e| *e == rendo[k]).unwrap();
                let want = flat(&irfs[&key])[t];
                close(&[yt[j]], &[want], 1e-9, &format!("{name}: irf {key} t={t}"));
            }
            yprev = yt;
        }
    }
}

#[test]
fn first_order_solution_var_pac() {
    solution_matches("pc3");
}

#[test]
fn first_order_solution_mce_pac() {
    solution_matches("pc4");
}

fn sim_data(r: &Value) -> PacData {
    let names = strings(&r["sim"]["names"]);
    let rows = r["sim"]["data"].as_array().unwrap();
    let mut cols = vec![Vec::new(); names.len()];
    for row in rows {
        for (j, v) in row.as_array().unwrap().iter().enumerate() {
            cols[j].push(v.as_f64().unwrap());
        }
    }
    PacData::new(names, cols)
}

fn range(r: &Value) -> (usize, usize) {
    let v = flat(&r["range_rows"]);
    (v[0] as usize, v[1] as usize)
}

/// The simulated data satisfies the PAC equation (with its residual) under the h parameters
/// computed here: checks the whole reduced form of the expectation.
fn equation_holds(name: &str, tol: f64) {
    let r = reference(name);
    let mf = run_commands(name);
    let data = sim_data(&r);
    let res = pac_residuals(&mf.varmodels, "zpac", &mf.model.param_names, &mf.model.params, &data, range(&r)).unwrap();
    let worst = res.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let at = res.iter().position(|v| v.abs() == worst).unwrap();
    let bad = res.iter().filter(|v| v.abs() > 1e-9).count();
    assert!(
        worst < tol && bad <= 5,
        "{name}: PAC equation residual {bad} periods above 1e-9, {worst} at {at}: {:?}",
        &res[..6.min(res.len())]
    );
}

#[test]
fn simulated_data_satisfies_pac_equation() {
    // pc1 has contemporaneous links between the variables: simul_backward_model solves each
    // period with a nonlinear solver whose tolerance leaves an error of a few 1e-6 in some periods
    equation_holds("pc1", 1e-5);
    equation_holds("pc2", 1e-9);
    equation_holds("pc3", 1e-9);
}

fn guess() -> Vec<(String, f64)> {
    vec![("e_c_m".into(), 0.9), ("c_z_1".into(), 0.5), ("c_z_2".into(), 0.2)]
}

fn without_residual(mut d: PacData) -> PacData {
    let k = d.names.iter().position(|n| n == "ez").unwrap();
    d.names.remove(k);
    d.cols.remove(k);
    d
}

fn nls_matches(name: &str, tol: f64) {
    let r = reference(name);
    let mf = run_commands(name);
    let data = without_residual(sim_data(&r));
    let mut params = mf.model.params.clone();
    let est =
        estimate_nls(&mf.varmodels, "zpac", &mf.model.param_names, &mut params, &guess(), &data, range(&r)).unwrap();
    let m = &r["nls"];
    assert_eq!(est.parnames, strings(&m["parnames"]));
    close(&est.estimator, &flat(&m["estimator"]), tol, &format!("{name}: NLS estimates"));
    close(&[est.ssr], &[m["ssr"].as_f64().unwrap()], tol, &format!("{name}: ssr"));
    close(&[est.s2], &[m["s2"].as_f64().unwrap()], tol, &format!("{name}: s2"));
    close(&[est.r2], &[m["R2"].as_f64().unwrap()], tol, &format!("{name}: R2"));
    close(&est.residual, &flat(&m["residual"]), 1e-7, &format!("{name}: residuals"));
    let c = est.covariance.as_ref().unwrap();
    let rc = flat(&m["covariance"]);
    let cv: Vec<f64> =
        (0..c.nrows()).flat_map(|i| (0..c.ncols()).map(move |j| (i, j))).map(|(i, j)| c[(i, j)]).collect();
    close(&cv, &rc, 1e-4, &format!("{name}: covariance"));
    close(&est.student, &flat(&m["student"]), 1e-4, &format!("{name}: student"));
    // the h parameters are updated at the estimates
    close(&params, &flat(&m["params_after"]), 1e-6, &format!("{name}: M_.params after NLS"));
}

fn ols_matches(name: &str) {
    let r = reference(name);
    let mf = run_commands(name);
    let data = without_residual(sim_data(&r));
    let mut params = mf.model.params.clone();
    let est =
        estimate_iterative_ols(&mf.varmodels, "zpac", &mf.model.param_names, &mut params, &guess(), &data, range(&r))
            .unwrap();
    let m = &r["ols"];
    assert_eq!(est.parnames, strings(&m["parnames"]));
    close(&est.estimator, &flat(&m["estimator"]), 1e-9, &format!("{name}: OLS estimates"));
    close(&[est.ssr], &[m["ssr"].as_f64().unwrap()], 1e-9, &format!("{name}: OLS ssr"));
    close(&est.residual, &flat(&m["residual"]), 1e-8, &format!("{name}: OLS residuals"));
    close(&params, &flat(&m["params_after"]), 1e-8, &format!("{name}: M_.params after OLS"));
}

#[test]
fn nls_var_simple_target() {
    nls_matches("pc3", 1e-7);
}

#[test]
fn nls_trend_component() {
    nls_matches("pc2", 1e-7);
}

#[test]
fn nls_composite_target() {
    nls_matches("pc1", 1e-7);
}

#[test]
fn iterative_ols_var_and_trend_component() {
    ols_matches("pc3");
    ols_matches("pc2");
}

#[test]
fn errors_like_dynare() {
    let bad = |src: &str| dynare::modfile::parse_mod_str(src, &[]).err().map(|e| format!("{e:#}")).unwrap_or_default();
    let head = "var y x z; varexo e ez; parameters a b c d beta; a=.5; b=.1; c=.2; d=.3; beta=.9;\nvar_model(model_name=m, eqtags=['e1']);\npac_model(auxiliary_model_name=m, discount=beta, model_name=p);\n";
    // no equation uses pac_expectation(p)
    let e = bad(&format!("{head}model;\n[name='e1']\ny = a*y(-1) + e;\nx = y;\nz = x;\nend;"));
    assert!(e.contains("pac_expectation"), "{e}");
    // unknown PAC model in an equation
    let e = bad("var y z; varexo e; parameters a; a=.5;\nmodel;\ny = a*y(-1) + pac_expectation(zz) + e;\nz = y;\nend;");
    assert!(e.contains("unknown PAC model"), "{e}");
}
