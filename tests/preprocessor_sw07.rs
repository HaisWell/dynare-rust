//! Preprocessor checks against Dynare 7.2's output for sw07.mod (reference/sw07/ref.json, and
//! tests/data/preprocessor_{sw07,qpm,gm05}.json from reference/pp/extract_driver.py).
mod common;

use dynare::modfile::{Command, EstimatedKind, ModFile, OptValue, PriorShape, opt, parse_mod_file};
use serde_json::Value;
use std::path::PathBuf;
use std::time::Instant;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn reference() -> Option<Value> {
    let p = root().join("reference/sw07/ref.json");
    Some(common::json_ref(p))
}

fn parse(stage: &str) -> ModFile {
    parse_mod_file(&root().join("reference/sw07/sw07.mod"), &[("STAGE".into(), stage.into())]).unwrap()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(nums).collect()
}

#[test]
fn sw07_structure_matches_dynare() {
    let r = reference().expect("reference/setup unavailable");
    let m = parse("9");
    let model = &m.model;
    assert_eq!(model.n_endo(), 50);
    assert_eq!(model.orig_endo_nbr, 45);
    assert_eq!(model.n_exo(), 11);
    assert_eq!(model.n_params(), 34);
    assert_eq!(model.n_equations(), 50);
    assert_eq!(model.orig_maximum_lead, 4);
    assert_eq!(model.orig_maximum_lag, 3);

    let names = strings(&r["endo_names"]);
    // stream PP: lead aux names (Dynare's expression indices) are reproduced exactly
    for (k, (a, b)) in model.endo_names.iter().zip(&names).enumerate() {
        assert_eq!(a, b, "endo {k}");
    }
    assert_eq!(model.exo_names, strings(&r["exo_names"]));
    assert_eq!(model.param_names, strings(&r["param_names"]));

    // aux vars: Dynare's lead aux names carry internal ids; map them to ours by position
    let rename = |s: &str| -> String {
        let mut out = s.to_string();
        for (a, b) in model.endo_names.iter().zip(&names) {
            if b.starts_with("AUX_ENDO_LEAD_") && s.starts_with(&format!("{b}(")) {
                out = s.replacen(b.as_str(), a, 1);
            }
        }
        out
    };
    let aux = r["aux_vars"].as_array().unwrap();
    assert_eq!(model.aux_vars.len(), aux.len());
    for (a, ra) in model.aux_vars.iter().zip(aux) {
        assert_eq!(a.endo_index + 1, ra["endo_index"].as_u64().unwrap() as usize);
        assert_eq!(a.ty as u64, ra["type"].as_u64().unwrap());
        assert_eq!(a.orig_expr, rename(ra["orig_expr"].as_str().unwrap()), "aux {}", a.endo_index);
        match ra["orig_index"].as_u64() {
            Some(oi) => {
                assert_eq!(a.orig_index.map(|i| i as u64 + 1), Some(oi));
                assert_eq!(a.orig_lead_lag.map(i64::from), ra["orig_lead_lag"].as_i64());
            }
            None => assert_eq!(a.orig_index, None),
        }
    }

    // lead_lag_incidence is 3 x n in the JSON (M_.lead_lag_incidence')
    let lli = matrix(&r["lead_lag_incidence"]);
    for (j, row) in model.lead_lag_incidence.iter().enumerate() {
        for (t, &v) in row.iter().enumerate() {
            assert_eq!(v as f64, lli[t][j], "lli var {j} row {t}");
        }
    }

    // calibration (ref.json's params_calib holds M_.params after the mode was loaded, so
    // compare with the .mod values) and shocks
    let p = |n: &str| model.params[model.param_index(n).unwrap()];
    assert_eq!((p("ctou"), p("curvp"), p("calfa"), p("crpi")), (0.025, 10.0, 0.19, 2.03));
    assert_eq!(p("constelab"), -0.1);
    assert!(model.params.iter().all(|v| v.is_finite()));
    // ref.json's Sigma_e is also at the mode; check the shocks block itself
    let k = |n: &str| model.exo_index(n).unwrap();
    assert_eq!(m.sigma_e[(0, 0)], 0.45 * 0.45);
    assert_eq!(m.sigma_e[(k("epistar"), k("epistar"))], 0.05 * 0.05);
    assert_eq!(m.sigma_e[(k("e_lr"), k("e_lr"))], 0.10 * 0.10);
    assert_eq!(m.sigma_e.iter().filter(|v| **v != 0.0).count(), 11);
    assert_eq!(m.varobs, strings(&r["varobs"]));
}

#[test]
fn sw07_jacobians_match_dynare() {
    let r = reference().expect("reference/setup unavailable");
    let m = parse("9");
    let model = &m.model;
    let ys = nums(&r["mode_dr"]["ys"]);
    let pm = nums(&r["mode_ss"]["params"]);
    let x = vec![0.0; model.n_exo()];

    let g1 = model.dynamic_jacobian(&ys, &x, &pm);
    let rg = matrix(&r["mode_g1"]);
    assert_eq!(g1.shape(), (50, 161));
    let mut maxdiff: f64 = 0.0;
    for i in 0..50 {
        for j in 0..161 {
            let d = (g1[(i, j)] - rg[i][j]).abs();
            assert!(d < 1e-12, "g1({i},{j}) = {} vs {} [{}]", g1[(i, j)], rg[i][j], model.equation_string(i));
            maxdiff = maxdiff.max(d);
        }
    }
    println!("max |g1 - dynare| = {maxdiff:e}");

    let gs = model.static_jacobian(&ys, &x, &pm);
    let rs = matrix(&r["static_g1_mode"]);
    for i in 0..50 {
        for j in 0..50 {
            assert!((gs[(i, j)] - rs[i][j]).abs() < 1e-12, "static g1({i},{j}) = {} vs {}", gs[(i, j)], rs[i][j]);
        }
    }

    // static residuals at the calibrated steady state
    let ss = nums(&r["steady_state"]);
    let pc = nums(&r["params_calib"]);
    let mut res = vec![0.0; 50];
    model.static_residuals(&ss, &x, &pc, &mut res);
    let worst = res.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    // Dynare's steady state solves to its own tolerance (6.5e-7 here)
    assert!(worst < 1e-6, "max static residual {worst}");

    // set_auxiliary_variables reproduces the aux part of the steady state
    let mut y = ss.clone();
    y[45..].iter_mut().for_each(|v| *v = f64::NAN);
    model.set_auxiliary_variables(&mut y, &x, &pc);
    for j in 45..50 {
        assert!((y[j] - ss[j]).abs() < 1e-12, "aux {j}");
    }

    // initval: the nine observables, in file order, with parameter expressions
    assert_eq!(m.initval.len(), 9);
    let j = |n: &str| model.endo_index(n).unwrap();
    assert_eq!(m.initval[0], (j("dy"), 0.43));
    assert!(m.initval.contains(&(j("labobs"), -0.1)));
    let robs =
        ((1.0 + 0.81 / 100.0) / ((1.0 / (1.0 + 0.16 / 100.0)) * (1.0f64 + 0.43 / 100.0).powf(-1.39)) - 1.0) * 100.0;
    let (_, v) = m.initval.iter().find(|(k, _)| *k == j("robs")).unwrap();
    assert!((v - robs).abs() < 1e-14, "robs initval {v} vs {robs}");
}

#[test]
fn sw07_estimated_params() {
    let r = reference().expect("reference/setup unavailable");
    let m = parse("9");
    assert_eq!(m.estimated_params.len(), 40);
    let ordered = m.estimated_params_dynare_order();
    let names: Vec<String> = ordered.iter().map(|e| m.estimated_param_name(e)).collect();
    assert_eq!(names, strings(&r["estim_param_names"]));
    let bt = &r["bayestopt"];
    let (pshape, p1, p2) = (nums(&bt["pshape"]), nums(&bt["p1"]), nums(&bt["p2"]));
    let lb = nums(&r["bounds"]["lb"]);
    for (k, e) in ordered.iter().enumerate() {
        assert_eq!(e.prior.unwrap().dynare_code() as f64, pshape[k], "{}", names[k]);
        assert!((e.mean - p1[k]).abs() < 1e-15, "{}", names[k]);
        assert!((e.sd - p2[k]).abs() < 1e-15, "{}", names[k]);
        if let Some(l) = e.lb {
            assert!((l - lb[k]).abs() < 1e-15, "{} lb", names[k]);
        }
    }
    // every syntax variant
    let find = |n: &str| ordered.iter().find(|e| m.estimated_param_name(e) == n).unwrap().clone();
    let b = find("crhoa");
    assert_eq!((b.init, b.lb, b.prior, b.mean, b.sd), (Some(0.95), None, Some(PriorShape::Beta), 0.5, 0.2));
    let c = find("constelab");
    assert_eq!((c.init, c.lb, c.ub, c.prior), (Some(-0.1), Some(-10.0), Some(10.0), Some(PriorShape::Normal)));
    let g = find("constebeta");
    assert_eq!((g.init, g.prior, g.mean, g.sd), (Some(0.16), Some(PriorShape::Gamma), 0.25, 0.1));
    let s = find("stderr em");
    assert_eq!(
        (s.init, s.lb, s.ub, s.prior, s.mean),
        (Some(0.24), Some(0.01), Some(3.0), Some(PriorShape::InvGamma1), 0.1)
    );
    assert!(matches!(s.kind, EstimatedKind::StdErr(4)));
    assert_eq!(ordered[0].init, Some(0.45));
    // the stage-9 file has no estimated_params_init
    assert!(!m.estimated_params_init_use_calibration);
}

#[test]
fn sw07_commands_by_stage() {
    let m9 = parse("9");
    assert!(matches!(&m9.commands[0], Command::Steady(o) if opt(o, "solve_algo") == Some(&OptValue::Num(1.0))));
    let Command::Estimation(o, vars) = &m9.commands[1] else { panic!("{:?}", m9.commands) };
    assert_eq!(m9.commands.len(), 2);
    assert_eq!(vars.len(), 8);
    assert_eq!(opt(o, "mode_file"), Some(&OptValue::Str("sw07/Output/sw07_mode".into())));
    assert_eq!(opt(o, "first_obs"), Some(&OptValue::Num(5.0)));
    assert_eq!(opt(o, "nograph"), Some(&OptValue::Flag));
    assert_eq!(opt(o, "mh_replic"), Some(&OptValue::Num(0.0)));

    let m3 = parse("3");
    let Command::Estimation(o, vars) = &m3.commands[1] else { panic!() };
    assert_eq!(vars[0], "dy");
    assert_eq!(opt(o, "datafile"), Some(&OptValue::Str("sw07_obs.csv".into())));
    assert_eq!(opt(o, "mh_replic"), Some(&OptValue::Num(1_000_000.0)));
    assert_eq!(opt(o, "mh_tune_jscale"), Some(&OptValue::Flag));
    assert_eq!(
        opt(o, "graph_format"),
        Some(&OptValue::List(vec![OptValue::Str("pdf".into()), OptValue::Str("eps".into())]))
    );
    assert_eq!(m3.model.n_endo(), 50);

    let m1 = parse("1");
    assert!(m1.estimated_params_init_use_calibration);
    let Command::Estimation(o, _) = &m1.commands[1] else { panic!() };
    assert_eq!(opt(o, "mode_compute"), Some(&OptValue::Num(4.0)));

    // no -D: the file's own default (STAGE = 3)
    let m = parse_mod_file(&root().join("reference/sw07/sw07.mod"), &[]).unwrap();
    assert_eq!(m.commands, m3.commands);
}

/// Compares a parsed model with data extracted from the `driver.m` that Dynare's
/// preprocessor generates for it (tests/data/preprocessor_<name>.json): names, aux variables,
/// lead_lag_incidence, the sparsity patterns of the dynamic and static Jacobians (which pins
/// down the equation order) and the calibration.
fn compare_with_dynare(name: &str, m: &ModFile, d: &Value) -> bool {
    let model = &m.model;
    let names = strings(&d["endo_names"]);
    let orig = d["orig_endo_nbr"].as_u64().unwrap() as usize;
    if orig != model.orig_endo_nbr || names[..orig] != model.endo_names[..orig] {
        eprintln!("{name}: the .mod file changed since tests/data/preprocessor_{name}.json was generated, skipping");
        return false;
    }
    assert_eq!(model.endo_names.len(), names.len(), "{name}");
    for (a, b) in model.endo_names.iter().zip(&names) {
        assert_eq!(a, b, "{name}"); // stream PP: exact, lead aux names included
    }
    assert_eq!(model.exo_names, strings(&d["exo_names"]), "{name}");
    assert_eq!(model.param_names, strings(&d["param_names"]), "{name}");
    assert_eq!(model.orig_maximum_lag as u64, d["orig_maximum_lag"].as_u64().unwrap(), "{name}");
    assert_eq!(model.orig_maximum_lead as u64, d["orig_maximum_lead"].as_u64().unwrap(), "{name}");
    let rename = |s: &str| -> String {
        let mut out = s.to_string();
        for (a, b) in model.endo_names.iter().zip(&names) {
            if b.starts_with("AUX_ENDO_LEAD_") && out.contains(&format!("{b}(")) {
                out = out.replace(&format!("{b}("), &format!("{a}("));
            }
        }
        out
    };
    let aux = d["aux_vars"].as_array().unwrap();
    assert_eq!(model.aux_vars.len(), aux.len(), "{name}");
    for (a, ra) in model.aux_vars.iter().zip(aux) {
        assert_eq!(a.endo_index as u64 + 1, ra["endo_index"].as_u64().unwrap(), "{name}");
        assert_eq!(a.ty as u64, ra["type"].as_u64().unwrap(), "{name}");
        assert_eq!(a.orig_expr, rename(ra["orig_expr"].as_str().unwrap()), "{name}: aux {}", a.endo_index);
        if let Some(oi) = ra["orig_index"].as_u64() {
            assert_eq!(a.orig_index.map(|i| i as u64 + 1), Some(oi), "{name}");
            assert_eq!(a.orig_lead_lag.map(i64::from), ra["orig_lead_lag"].as_i64(), "{name}");
        }
    }
    let lli = d["lead_lag_incidence"].as_array().unwrap();
    for (j, row) in lli.iter().enumerate() {
        let row: Vec<usize> = row.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        // Dynare drops the lag (lead) column of a model without lags (leads)
        let keep: Vec<usize> = (0..3)
            .filter(|&t| row.len() == 3 || t == 1 || model.lead_lag_incidence.iter().any(|r| r[t] != 0))
            .collect();
        let ours: Vec<usize> = keep.iter().map(|&t| model.lead_lag_incidence[j][t]).collect();
        assert_eq!(row, ours, "{name}: lli of {}", model.endo_names[j]);
    }
    let pattern = |r: &str, c: &str| -> Vec<(usize, usize)> {
        let rows = nums(&d[r]);
        let cols = nums(&d[c]);
        let mut v: Vec<(usize, usize)> =
            rows.iter().zip(&cols).map(|(&r, &c)| (r as usize - 1, c as usize - 1)).collect();
        v.sort_by_key(|&(r, c)| (c, r));
        v
    };
    let dynp = pattern("dynamic_g1_sparse_rowval", "dynamic_g1_sparse_colval");
    assert_eq!(model.dynamic_jacobian_pattern(), dynp.as_slice(), "{name}: dynamic g1 pattern");
    // static pattern: compare with the non-zeros of our static Jacobian at a generic point
    let stp = pattern("static_g1_sparse_rowval", "static_g1_sparse_colval");
    let y: Vec<f64> = (0..model.n_endo()).map(|k| 0.37 + 0.01 * k as f64).collect();
    // generic parameter values: at the calibration some entries could cancel numerically
    let p: Vec<f64> = (0..model.n_params()).map(|k| 0.123 + 0.0071 * k as f64).collect();
    // generic exogenous values too (e*e(+1), sign(u) vanish at 0)
    let x: Vec<f64> = (0..model.n_exo()).map(|k| 0.013 + 0.01 * k as f64).collect();
    let gs = model.static_jacobian(&y, &x, &p);
    let mut ours = Vec::new();
    for c in 0..gs.ncols() {
        for r in 0..gs.nrows() {
            if gs[(r, c)] != 0.0 {
                ours.push((r, c));
            }
        }
    }
    // In Dynare's static model STEADY_STATE(x) stays an operator node (x/STEADY_STATE(x) is not
    // simplified away), so structural zeros of those equations differ numerically: skip them.
    let ss_rows: Vec<usize> = (0..model.n_equations())
        .filter(|&i| {
            let (dag, r) = model.residual_node(i);
            dag.has_steady_state(r)
        })
        .collect();
    // the symbolic pattern with the preprocessor's simplifications is Dynare's exactly
    let mut sym = dynare::model::blocks::static_jacobian_incidence(model);
    sym.sort_by_key(|&(r, c)| (c, r));
    assert_eq!(sym, stp, "{name}: symbolic static g1 pattern");
    // the numerical non-zeros are a subset: some symbolic non-zeros cancel at every point (the
    // static `cf = chab/(1+chab)*cf(-1) + 1/(1+chab)*cf(+1) + ...` of SW07 w.r.t. `cf`, `P/(P*c)`
    // w.r.t. P in q1_fsns's detrended model)
    let ours: Vec<_> = ours.into_iter().filter(|(r, _)| !ss_rows.contains(r)).collect();
    assert!(ours.iter().all(|e| stp.contains(e)), "{name}: numerical static g1 pattern");
    for (k, v) in d["params"].as_array().unwrap().iter().enumerate() {
        if let Some(v) = v.as_f64() {
            // driver.m prints 15 significant digits
            assert!(
                (model.params[k] - v).abs() <= 1e-14 * v.abs(),
                "{name}: {} = {} vs {v}",
                model.param_names[k],
                model.params[k]
            );
        }
    }
    true
}

fn data(name: &str) -> Value {
    let p = root().join(format!("tests/data/preprocessor_{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn sw07_matches_dynare_driver() {
    let m = parse("9");
    assert!(compare_with_dynare("sw07", &m, &data("sw07")));
}

/// The QPM and GM05 benchmark models (reference/qpm, reference/gm05): every estimation stage
/// parses, and the default stage (3) matches the preprocessor's driver.
#[test]
fn qpm_and_gm05_match_dynare_driver() {
    for name in ["qpm", "gm05"] {
        let path = root().join("reference").join(name).join(format!("model_{name}.mod"));
        for stage in ["1", "2", "3"] {
            let m =
                parse_mod_file(&path, &[("STAGE".into(), stage.into())]).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            let model = &m.model;
            assert!(model.n_endo() >= model.orig_endo_nbr);
            assert!(!m.estimated_params.is_empty());
            assert!(m.commands.iter().any(|c| matches!(c, Command::Estimation(..))));
            let ys = vec![0.0; model.n_endo()];
            let g1 = model.dynamic_jacobian(&ys, &vec![0.0; model.n_exo()], &model.params);
            assert!(g1.iter().all(|v| v.is_finite()), "{name}");
            if stage == "3" {
                assert!(compare_with_dynare(name, &m, &data(name)), "{name}");
            }
        }
    }
}

/// `cargo test --release -- --ignored --nocapture bench_dynamic_jacobian`
#[test]
#[ignore]
fn bench_dynamic_jacobian() {
    let r = reference().expect("reference/setup unavailable");
    let m = parse("9");
    let model = &m.model;
    let ys = nums(&r["mode_dr"]["ys"]);
    let pm = nums(&r["mode_ss"]["params"]);
    let x = vec![0.0; model.n_exo()];
    let mut ws = model.workspace();
    let mut out = dynare::linalg::Mat::zeros(model.n_equations(), 3 * model.n_endo() + model.n_exo());
    let n = 20_000;
    let t = Instant::now();
    for _ in 0..n {
        model.dynamic_jacobian_with(&ys, &x, &pm, &mut ws, &mut out);
    }
    let dense = t.elapsed().as_secs_f64() / n as f64;
    let mut vals = vec![0.0; model.dynamic_jacobian_pattern().len()];
    let y3: Vec<f64> = ys.iter().chain(&ys).chain(&ys).copied().collect();
    let t = Instant::now();
    for _ in 0..n {
        model.dynamic_jacobian_values_with(&y3, &x, &pm, &mut ws, &mut vals);
    }
    let sparse = t.elapsed().as_secs_f64() / n as f64;
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(model.dynamic_jacobian(&ys, &x, &pm));
    }
    let alloc = t.elapsed().as_secs_f64() / n as f64;
    println!(
        "dynamic_jacobian: dense in place {:.2} us, values only ({} nnz) {:.2} us, allocating {:.2} us",
        dense * 1e6,
        vals.len(),
        sparse * 1e6,
        alloc * 1e6
    );
}

#[test]
fn model_is_send_sync_and_thread_safe() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<dynare::model::Model>();
    assert_send_sync::<ModFile>();
    let r = reference().expect("reference/setup unavailable");
    let m = parse("9");
    let ys = nums(&r["mode_dr"]["ys"]);
    let pm = nums(&r["mode_ss"]["params"]);
    let x = vec![0.0; m.model.n_exo()];
    let g = m.model.dynamic_jacobian(&ys, &x, &pm);
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| assert_eq!(m.model.dynamic_jacobian(&ys, &x, &pm), g));
        }
    });
}

/// stream PP: small models exercising every aux-variable type, model-local variables with
/// leads, `STEADY_STATE()`, `EXPECTATION()`, `diff()`, constant equations, predetermined
/// variables, `[static]`/`[dynamic]` and OccBin equations, against Dynare's preprocessor
/// (reference/pp/<name>.json from reference/pp/extract_driver.py).
#[test]
fn pp_models_match_dynare() {
    let dir = root().join("reference/pp");
    let mut n = 0;
    for name in [
        "pp_ss",
        "pp_leads",
        "ppdet_leads",
        "pp_expect",
        "pp_const",
        "pp_static",
        "pp_edit",
        "pp_macro",
        "pp_ss2",
        "pp_mflocal",
    ] {
        let m = parse_mod_file(&dir.join(format!("{name}.mod")), &[]).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let d: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap()).unwrap();
        assert!(compare_with_dynare(name, &m, &d), "{name}");
        // equation tags (M_.equations_tags: equation, key, value)
        let mut ours = Vec::new();
        for (i, tags) in m.model.equation_tags.iter().enumerate() {
            let mut t: Vec<_> = tags.iter().filter(|(k, _)| k != "static" && k != "dynamic").cloned().collect();
            t.sort();
            for (k, v) in t {
                ours.push((i as u64 + 1, k, v));
            }
        }
        let theirs: Vec<(u64, String, String)> = d["equations_tags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| (r[0].as_u64().unwrap(), r[1].as_str().unwrap().to_string(), r[2].as_str().unwrap().to_string()))
            .filter(|(_, k, _)| k != "static" && k != "dynamic")
            .collect();
        assert_eq!(ours, theirs, "{name}: equation tags");
        n += 1;
    }
    assert_eq!(n, 10);
}

// ---- var(log), deflators, trend_var, steady_state_model temporaries (reference/q1)
#[test]
fn q1_models_match_dynare() {
    let dir = root().join("reference/q1");
    for name in ["q1_log", "q1_logexp", "q1_ns2", "q1_fsns"] {
        let m = parse_mod_file(&dir.join(format!("{name}.mod")), &[]).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let d: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap()).unwrap();
        assert!(compare_with_dynare(name, &m, &d), "{name}");
    }
    // the preprocessor's balanced-growth test (trend variables at 2, initval values)
    let e = parse_mod_file(&dir.join("q1_bg.mod"), &[]).unwrap_err().to_string();
    assert!(
        e.contains(
            "equation 1 (line 8) w.r.t. trend variable A and endogenous variable y is not null (abs. value = 0.127659)"
        ),
        "{e}"
    );
}

/// stream PP: the macro processor's output equals Dynare 7.1's `savemacro` expansion of
/// reference/pp/pp_macro.mod (arrays, tuples, comprehensions, macro functions, `@#for` over
/// tuples with `when`), up to comments and blank lines.
#[test]
fn pp_macro_expansion_matches_dynare() {
    let dir = root().join("reference/pp");
    let ours = dynare::modfile::macro_proc::preprocess(&dir.join("pp_macro.mod"), &[]).unwrap();
    let norm = |s: &str| -> Option<String> {
        let s = s.split("//").next().unwrap().trim();
        (!s.is_empty()).then(|| s.split_whitespace().collect::<Vec<_>>().join(" "))
    };
    let a: Vec<String> = ours.iter().filter_map(|l| norm(&l.text)).collect();
    let b: Vec<String> =
        std::fs::read_to_string(dir.join("pp_macro.macroexp")).unwrap().lines().filter_map(norm).collect();
    assert_eq!(a, b);
}
