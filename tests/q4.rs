//!  against Dynare 7.2 (estimation cases) and 7.1 (reference/q4/ref_q4_*.json, written by
//! reference/q4/dump_q4.m): the `dates` class, the `data` command / ISO (daily) csv dates,
//! `epilogue` + `shock_decomposition(with_epilogue)`, `plot_shock_decomposition(write_xls,
//! steadystate, init2shocks)`, `initval_file` / `histval_file` with dates.

use std::path::{Path, PathBuf};

use dynare::dates::{Date, Freq};
use serde_json::Value;

fn ref_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/q4")
}

fn load(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(ref_dir().join(name)).unwrap()).unwrap()
}

/// A private copy of reference/q4 (the driver writes `<fname>/Output` next to the .mod).
fn work_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dynare_rs_q4_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for e in std::fs::read_dir(ref_dir()).unwrap() {
        let p = e.unwrap().path();
        if matches!(p.extension().and_then(|x| x.to_str()), Some("mod" | "csv")) {
            std::fs::copy(&p, dir.join(p.file_name().unwrap())).unwrap();
        }
    }
    dir
}

fn run(dir: &Path, file: &str, case: u32) {
    let ro = dynare::driver::RunOptions { defines: vec![("CASE".into(), case.to_string())], ..Default::default() };
    dynare::driver::run(&dir.join(file), &ro).unwrap();
}

/// MATLAB's jsonencode of a numeric array: nested lists, null = NaN.
fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs())
}

/// Sheets of an .xlsx file: name -> rows of cells (None = empty, Some(Err) = text).
fn read_xlsx(p: &Path) -> Vec<(String, Vec<Vec<Option<Result<f64, String>>>>)> {
    use calamine::{Data, Reader, open_workbook_auto};
    let mut wb = open_workbook_auto(p).unwrap();
    let names = wb.sheet_names();
    names
        .iter()
        .map(|n| {
            let r = wb.worksheet_range(n).unwrap();
            let rows = r
                .rows()
                .map(|row| {
                    row.iter()
                        .map(|c| match c {
                            Data::Float(f) => Some(Ok(*f)),
                            Data::Int(i) => Some(Ok(*i as f64)),
                            Data::String(s) => Some(Err(s.clone())),
                            _ => None,
                        })
                        .collect()
                })
                .collect();
            (n.clone(), rows)
        })
        .collect()
}

/// MATLAB's readcell of the written sheet, flattened column-major by jsonencode.
fn matlab_sheet(cells: &Value, ncols: usize) -> Vec<Vec<Value>> {
    let c = cells.as_array().unwrap();
    let nrows = c.len() / ncols;
    (0..nrows).map(|i| (0..ncols).map(|j| c[j * nrows + i].clone()).collect()).collect()
}

fn as_vec<T: Clone>(v: &Value, f: impl Fn(&Value) -> T) -> Vec<T> {
    match v.as_array() {
        Some(a) => a.iter().map(f).collect(),
        None => vec![f(v)],
    }
}

/// shock_decomposition(with_epilogue) and plot_shock_decomposition(write_xls, steadystate[,
/// init2shocks]): the workbook against oo_.shock_decomposition (with the epilogue rows) and
/// MATLAB's own workbook. MATLAB's writetable on macOS drops the numbers of the first
/// component column (the "e" column reads back empty), so that column is checked against
/// oo_.shock_decomposition only.
fn check_decomposition_case(case: u32, xls: &str, init: &str) {
    let r = load(&format!("ref_q4_est_case{case}.json"));
    assert!(r["ok"].as_bool().unwrap(), "{:?}", r["error"]);
    let dir = work_dir(&format!("case{case}"));
    run(&dir, "q4_est.mod", case);
    // dataset_.init: ISO dates are daily in Dynare (first date + first_obs - 1 days)
    assert_eq!(dynare::dataset::last_sample_init().unwrap().label(), init);
    // the epilogue over the smoothed variables (MATLAB: q4_est.epilogue_dynamic on them)
    if let Some(sm) = r.get("epilogue_smoothed") {
        let text = std::fs::read_to_string(dir.join("q4_est/Output/q4_est_epilogue_smoothed_variables.csv")).unwrap();
        let header: Vec<&str> = text.lines().next().unwrap().split(',').skip(1).collect();
        let rows: Vec<Vec<f64>> =
            text.lines().skip(1).map(|l| l.split(',').skip(1).map(|x| x.parse().unwrap()).collect()).collect();
        let names = as_vec(&r["epilogue_names"], |v| v.as_str().unwrap().to_string());
        assert_eq!(header, names);
        for (k, n) in names.iter().enumerate() {
            let want = as_vec(&sm[n.as_str()], num);
            assert_eq!(rows.len(), want.len());
            for (t, row) in rows.iter().enumerate() {
                assert!(close(row[k], want[t], 1e-12), "epilogue smoothed {n} {t}: {} vs {}", row[k], want[t]);
            }
        }
    }
    assert_eq!(r["dataset_init"].as_str().unwrap(), init);
    let sd = r["shock_decomposition"].as_array().unwrap(); // [var][comp][t]
    let epi_ss: Vec<f64> = as_vec(&r["epilogue_steady_state"], num);
    let got = read_xlsx(&dir.join("q4_est/Output").join(xls));
    let want_x = as_vec(&r["xls"], |v| v.clone());
    let want_sheets = as_vec(&want_x[0]["sheets"], |v| v.clone());
    assert_eq!(got.len(), want_sheets.len());
    let iv = r.get("initval_decomposition").and_then(|v| v.as_array());
    for ((name, rows), ws) in got.iter().zip(&want_sheets) {
        assert_eq!(name, ws["name"].as_str().unwrap());
        let ncols = rows[0].len();
        let m = matlab_sheet(&ws["cells"], ncols);
        assert_eq!(rows.len(), m.len(), "{name}: rows");
        let v = ["y", "c", "k", "ya", "gya", "cy", "ky"].iter().position(|n| n == name).unwrap();
        for (i, (gr, wr)) in rows.iter().zip(&m).enumerate() {
            for (j, (g, w)) in gr.iter().zip(wr).enumerate() {
                if i == 0 {
                    assert_eq!(g.clone().unwrap().unwrap_err(), w.as_str().unwrap(), "{name} header {j}");
                    continue;
                }
                let gv = match g {
                    Some(Ok(x)) => *x,
                    _ => f64::NAN,
                };
                if j == 1 {
                    // first component: oo_.shock_decomposition (+ the initial condition moved to it by init2shocks)
                    let mut want = num(&sd[v][0][i - 1]);
                    if let Some(iv) = iv {
                        want += num(&iv[v][0][i - 1]);
                    }
                    assert!(close(gv, want, 1e-9), "{name} ({i},{j}): {gv} vs {want}");
                    continue;
                }
                let wv = if w.is_number() { num(w) } else { f64::NAN };
                assert!(close(gv, wv, 1e-9), "{name} ({i},{j}): {gv} vs {wv}");
            }
        }
        // the Steady State column (steadystate option): dr.ys or the epilogue steady state
        let ss = if case == 2 { 0.0 } else { epi_ss[v] };
        assert!(close(rows[1][ncols - 1].clone().unwrap().unwrap(), ss, 1e-12));
    }
}

#[test]
fn data_command_epilogue_write_xls() {
    check_decomposition_case(1, "q4_est_shock_decomposition_w_x.xlsx", "1991Q2");
}

#[test]
fn iso_dates_are_daily() {
    check_decomposition_case(2, "q4_est_shock_decomposition_iso.xlsx", "1990-04-04");
}

#[test]
fn init2shocks() {
    check_decomposition_case(4, "q4_est_shock_decomposition_i2s_init2shocks.xlsx", "1991Q2");
}

/// Two endogenous variables in the epilogue (Dynare 7.2 fixed the dseries orientation in
/// epilogue_shock_decomposition.m), a per-variable begin date (`ky`: max over variables of
/// first observation + own lag) and prefix names (`y`, `ya`).
#[test]
fn epilogue_two_variables() {
    check_decomposition_case(3, "q4_est_shock_decomposition_w_x.xlsx", "1991Q2");
}

#[test]
fn dates_class() {
    let r = load("ref_q4_dates.json");
    for d in r["dates"].as_array().unwrap() {
        let s = d["input"].as_str().unwrap();
        let x = Date::parse(s).unwrap_or_else(|| panic!("{s}"));
        assert_eq!(x.label(), d["label"].as_str().unwrap(), "{s}");
        assert_eq!(x.freq.number(), d["freq"].as_i64().unwrap(), "{s}");
        assert_eq!(x.time, d["time"].as_i64().unwrap(), "{s}");
        assert_eq!(x.add(5).label(), d["plus5"].as_str().unwrap(), "{s}");
        assert_eq!(x.add(-3).label(), d["minus3"].as_str().unwrap(), "{s}");
    }
    let a = Date::parse("2015W50").unwrap();
    let b = Date::parse("2016W3").unwrap();
    let w: Vec<String> = Date::range(a, b).unwrap().iter().map(|d| d.label()).collect();
    let want: Vec<String> =
        r["weekly_range"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(w, want);
    let strs = ["2004Q5", "2004M13", "2004-3-31", "2004-02-30", "2003W53", "2004", "2004QQ1", "2004-02-29"];
    for (s, want) in strs.iter().zip(r["isdate"].as_array().unwrap()) {
        assert_eq!(Date::parse(s).is_some(), want.as_bool().unwrap(), "isdate({s})");
    }
    assert_eq!(Date::parse_mod("2023Q1+6").unwrap(), Date::from_parts(Freq::Quarterly, 2024, 3));
}

/// initval_file / histval_file with DATE options (first_obs, last_obs,
/// first/last_simulation_period) on reference/q4/q4_datafiles.mod.
fn check_datafiles(case: u32) {
    use dynare::deterministic::DetState;
    use dynare::modfile::{Command, OptValue, parse_mod_file};
    let r = load(&format!("ref_q4_datafiles_case{case}.json"));
    assert!(r["ok"].as_bool().unwrap());
    let mf = parse_mod_file(&ref_dir().join("q4_datafiles.mod"), &[("CASE".into(), case.to_string())]).unwrap();
    let mut det = DetState::new(&mf).unwrap();
    for c in &mf.commands {
        match c {
            Command::Steady(o) => {
                let mut o = o.clone();
                o.push(("noprint".into(), OptValue::Flag));
                det.steady(&mf.model, &o).unwrap()
            }
            Command::Other(n, o) if n == "initval_file" => {
                dynare::ep::datafiles::initval_file(&mf, &mut det, o, &ref_dir()).unwrap()
            }
            Command::Other(n, o) if n == "histval_file" => {
                dynare::ep::datafiles::histval_file(&mf, &mut det, o, &ref_dir()).unwrap()
            }
            Command::Deterministic(n, o, _) => {
                det.simul.update(o);
                det.simul.noprint = true;
                det.simul.verbosity = 0;
                match n.as_str() {
                    "perfect_foresight_setup" => det.setup(&mf).unwrap(),
                    "perfect_foresight_solver" => det.solve(&mf.model).unwrap(),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    assert_eq!(det.simul.periods as f64, num(&r["periods"]), "periods");
    let want: Vec<Vec<f64>> = as_vec(&r["endo_simul"], |row| as_vec(row, num));
    let got = det.endo_simul.as_ref().unwrap();
    assert_eq!((got.nrows(), got.ncols()), (want.len(), want[0].len()));
    let mut e = 0.0f64;
    for (i, row) in want.iter().enumerate() {
        for (j, w) in row.iter().enumerate() {
            e = e.max((got[(i, j)] - w).abs());
        }
    }
    println!("case {case}: endo_simul {e:.2e}");
    assert!(e < 1e-9, "case {case}: {e}");
}

/// `include_eqs=` / `exclude_eqs=` on reference/q4/q4_exclude.mod: the 7.1 preprocessor gives,
/// with include_eqs=[res, euler, zeq], endo_names c k z AUX_EXO_LEAD_42 (y(+1) of the Euler
/// equation, y having become exogenous) and exo_names y e u (y takes its symbol-id place).
#[test]
fn include_exclude_eqs() {
    let mut o = dynare::modfile::ParseOptions { include_eqs: Some("[res, euler, zeq]".into()), ..Default::default() };
    let mf = dynare::modfile::parse_mod_file_opts(&ref_dir().join("q4_exclude.mod"), &o).unwrap();
    assert_eq!(mf.model.endo_names, ["c", "k", "z", "AUX_EXO_LEAD_42"]);
    assert_eq!(mf.model.exo_names, ["y", "e", "u"]);
    o.include_eqs = None;
    o.exclude_eqs = Some("zeq".into());
    // z only appears as 0*z elsewhere (simplified away): it is removed (the 7.1 preprocessor
    // stops with an internal error on this file)
    let mf = dynare::modfile::parse_mod_file_opts(&ref_dir().join("q4_exclude.mod"), &o).unwrap();
    assert_eq!(mf.model.endo_names, ["y", "c", "k"]);
    assert_eq!(mf.model.exo_names, ["e", "u"]);
    o.exclude_eqs = Some("[name=(nosuch)]".into());
    assert!(dynare::modfile::parse_mod_file_opts(&ref_dir().join("q4_exclude.mod"), &o).is_err());
}

/// `json=compute`: modfile.json / modfile-original.json against the 7.1 preprocessor's
/// (reference/q4/json, written by `dynare-preprocessor FILE json=compute`): declarations,
/// aux_vars, model, abstract_syntax_tree, variable_mapping, Jacobian sparsity, model_info,
/// varobs. Not compared: statements (a subset is written), the temporary-term counts and the
/// second/third-order sparse indices (not written).
fn check_json(modfile: &Path, defines: &[(String, String)], fname: &str, want_files: &[(&str, &str)], stage: &str) {
    let o = dynare::modfile::ParseOptions { defines: defines.to_vec(), ..Default::default() };
    dynare::model::dtree::json_capture();
    let mf = dynare::modfile::parse_mod_file_opts(modfile, &o).unwrap();
    let cap = dynare::model::dtree::json_take().unwrap();
    let dir = std::env::temp_dir().join(format!("dynare_rs_q4_json_{fname}_{}", std::process::id()));
    let jo = dynare::q4::json::JsonOptions { stage: stage.into(), stdout: false, derivsimple: false };
    dynare::q4::json::write(&mf, &cap, &dir, fname, &jo).unwrap();
    let skip =
        ["statements", "dynamic_tmp_nbr", "static_tmp_nbr", "dynamic_g2_sparse_indices", "dynamic_g3_sparse_indices"];
    for (ours, theirs) in want_files {
        let want = load(&format!("json/{theirs}"));
        let got: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join(fname).join("model/json").join(ours)).unwrap())
                .unwrap();
        // the preprocessor writes a TrinaryOpNode's third argument under a second "arg2" key
        // (the last one wins when read back); dynare-rust writes it as "arg3"
        fn norm(v: &Value) -> Value {
            match v {
                Value::Object(m) => {
                    let mut m: serde_json::Map<String, Value> = m.iter().map(|(k, x)| (k.clone(), norm(x))).collect();
                    if let Some(a3) = m.remove("arg3") {
                        m.insert("arg2".into(), a3);
                    }
                    Value::Object(m)
                }
                Value::Array(a) => Value::Array(a.iter().map(norm).collect()),
                x => x.clone(),
            }
        }
        for (k, w) in want.as_object().unwrap() {
            if skip.contains(&k.as_str()) {
                continue;
            }
            assert_eq!(got.get(k).map(norm).as_ref(), Some(w), "{theirs}: key {k}");
        }
    }
}

#[test]
fn json_matches_preprocessor() {
    check_json(
        &ref_dir().join("json/q4_json.mod"),
        &[],
        "q4_json",
        &[("modfile.json", "q4_json_modfile.json"), ("modfile-original.json", "q4_json_modfile-original.json")],
        "compute",
    );
    // sw07_stage9_modfile.json: `dynare-preprocessor sw07.mod json=compute -DSTAGE=9` (7.2)
    let sw07 = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07/sw07.mod");
    check_json(
        &sw07,
        &[("STAGE".into(), "9".into())],
        "sw07",
        &[("modfile.json", "sw07_stage9_modfile.json")],
        "compute",
    );
}

#[test]
fn initval_file_dates() {
    check_datafiles(1);
}

#[test]
fn initval_file_simulation_period_dates() {
    check_datafiles(2);
}

#[test]
fn histval_file_dates() {
    check_datafiles(3);
}
