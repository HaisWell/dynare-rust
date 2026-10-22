//! : stoch_simul options against Dynare 7.2 on reference/q2/rbc_q2.mod
//! (reference/q2/ref_q2_c<k>.json and q2_c<k>_matlab.log, written by dump_q2.m):
//! `loglinear`, `bandpass_filter` (theoretical and Baxter-King), `one_sided_hp_filter`,
//! `spectral_density`, `contemporaneous_correlation`, `dr=cycle_reduction|logarithmic_reduction`,
//! `tex` tables.
mod common;

use std::path::{Path, PathBuf};

use dynare::analysis::{self, OutputContext, StochSimulOptions, TexNames};
use dynare::linalg::Mat;
use dynare::modfile::{Command, ModFile, parse_mod_file};
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn load(case: usize) -> Option<Value> {
    Some(common::json_ref(root().join(format!("reference/q2/ref_q2_c{case}.json"))))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vec_of(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vec_of).collect(),
        x => vec![num(x)],
    }
}

fn mat(v: &Value, nrows: usize, ncols: usize) -> Mat {
    match v {
        Value::Array(rows) if rows.first().is_some_and(|r| r.is_array()) => {
            Mat::from_fn(nrows, ncols, |i, j| num(&rows[i][j]))
        }
        _ => {
            let x = vec_of(v);
            Mat::from_fn(nrows, ncols, |i, j| x[i * ncols + j])
        }
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs())
}

fn assert_mat(what: &str, a: &Mat, b: &Mat, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            assert!(close(a[(i, j)], b[(i, j)], tol), "{what}[{i},{j}]: rust {} vs dynare {}", a[(i, j)], b[(i, j)]);
        }
    }
}

fn assert_vec(what: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(close(*x, *y, tol), "{what}[{k}]: rust {x} vs dynare {y}");
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dynare_q2_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Case {
    mf: ModFile,
    /// Options as Dynare has them at each stoch_simul command (they persist).
    opts: Vec<(StochSimulOptions, Vec<String>)>,
}

fn setup(case: usize) -> Case {
    let mf = parse_mod_file(&root().join("reference/q2/rbc_q2.mod"), &[("CASE".into(), case.to_string())]).unwrap();
    let mut o = StochSimulOptions::default();
    let mut opts = Vec::new();
    for c in &mf.commands {
        if let Command::StochSimul(co, vars) = c {
            let unknown = o.apply(co);
            assert!(unknown.is_empty(), "unknown options {unknown:?}");
            opts.push((o.clone(), vars.clone()));
        }
    }
    Case { mf, opts }
}

fn ctx(dir: &Path) -> OutputContext {
    OutputContext { dir: dir.to_path_buf(), fname: "rbc_q2".into(), out_dir: dir.join("rbc_q2/Output"), seed: 1 }
}

/// Runs command `k` of a case with MATLAB's shocks (when it simulates); returns the results
/// and the solution.
fn run(c: &Case, k: usize, r: &Value, dir: &Path) -> (analysis::StochSimulResults, analysis::LinearSolution) {
    let (o, vars) = &c.opts[k];
    let sol = analysis::solve_q2(&c.mf, o, None).unwrap().unwrap();
    let cmd = &r["cmds"][k];
    let nexo = c.mf.model.n_exo();
    let ex = (o.periods > 0).then(|| mat(&cmd["exo_simul"], o.periods, nexo));
    let cx = ctx(dir);
    let tex = o.tex.then(|| TexNames::from_modfile(&c.mf, &cx));
    let res = analysis::stoch_simul_q2(&c.mf.model, &sol, o, vars, &cx, ex.as_ref(), tex.as_ref()).unwrap();
    (res, sol)
}

fn check_dr(c: &Case, sol: &analysis::LinearSolution, cmd: &Value, tol: f64) {
    let n = c.mf.model.n_endo();
    let ov: Vec<usize> = vec_of(&cmd["order_var"]).iter().map(|&x| x as usize - 1).collect();
    assert_eq!(ov, sol.dr.order_var);
    assert_mat("ghx", &sol.dr.ghx, &mat(&cmd["ghx"], n, sol.dr.nspred), tol);
    assert_mat("ghu", &sol.dr.ghu, &mat(&cmd["ghu"], n, c.mf.model.n_exo()), tol);
    assert_vec("ys", &sol.dr.ys, &vec_of(&cmd["ys"]), 1e-12);
}

fn check_irfs(c: &Case, res: &analysis::StochSimulResults, cmd: &Value, tol: f64) {
    for s in &res.irfs {
        for (k, &v) in res.ivar.iter().enumerate() {
            let key = format!("{}_{}", c.mf.model.endo_names[v], c.mf.model.exo_names[s.shock]);
            let got: Vec<f64> = s.y.row(k).iter().copied().collect();
            assert_vec(&key, &got, &vec_of(&cmd["irfs"][&key]), tol);
        }
    }
}

fn check_spectral(res: &analysis::StochSimulResults, cmd: &Value) {
    let sd = res.spectral_density.as_ref().expect("spectral density");
    assert_vec("freqs", &sd.freqs, &vec_of(&cmd["SpectralDensity"]["freqs"]), 1e-15); // serde_json parses within 1 ulp
    let want = mat(&cmd["SpectralDensity"]["density"], sd.density.nrows(), sd.density.ncols());
    assert_mat("density", &sd.density, &want, 1e-9);
}

fn check_simulated(res: &analysis::StochSimulResults, cmd: &Value, n: usize, nvar: usize, nexo: usize, periods: usize) {
    assert_mat("endo_simul", res.endo_simul.as_ref().unwrap(), &mat(&cmd["endo_simul"], n, periods), 1e-10);
    let sm = res.simulated.as_ref().unwrap();
    assert_vec("mean", &sm.mean, &vec_of(&cmd["mean"]), 1e-9);
    assert_vec("skewness", &sm.skewness, &vec_of(&cmd["skewness"]), 1e-7);
    assert_vec("kurtosis", &sm.kurtosis, &vec_of(&cmd["kurtosis"]), 1e-7);
    assert_mat("var", &sm.var, &mat(&cmd["var"], nvar, nvar), 1e-9);
    for (l, a) in sm.autocorr.iter().enumerate() {
        assert_mat("autocorr", a, &mat(&cmd["autocorr"][l], nvar, nvar), 1e-9);
    }
    assert_mat("sim vardec", sm.vardec.as_ref().unwrap(), &mat(&cmd["variance_decomposition"], nvar, nexo), 1e-8);
}

#[test]
fn loglinear_order1() {
    let r = load(1).expect("reference/setup unavailable");
    let c = setup(1);
    let dir = scratch("c1");
    let n = c.mf.model.n_endo();
    let nexo = c.mf.model.n_exo();
    // command 1: theoretical moments, IRFs, conditional vardec, spectral density, TeX
    let (res, sol) = run(&c, 0, &r, &dir);
    let cmd = &r["cmds"][0];
    check_dr(&c, &sol, cmd, 1e-10);
    check_irfs(&c, &res, cmd, 1e-10);
    let th = res.theoretical.as_ref().unwrap();
    assert_mat("var", &th.gamma0, &mat(&cmd["var"], n, n), 1e-10);
    assert_vec("mean", &th.mean(&sol.dr), &vec_of(&cmd["mean"]), 1e-12);
    assert_mat("vardec", &(th.vardec.as_ref().unwrap() * 100.0), &mat(&cmd["variance_decomposition"], n, nexo), 1e-8);
    let cvd = res.conditional_vardec.as_ref().unwrap();
    for (s, m) in cvd.iter().enumerate() {
        let want = Mat::from_fn(n, nexo, |i, e| num(&cmd["conditional_variance_decomposition"][i][s][e]));
        assert_mat("cvd", m, &want, 1e-9);
    }
    check_spectral(&res, cmd);
    // TeX tables of the theoretical moments, identical to Dynare's but the date line
    for t in [
        "covar_ex_shocks",
        "th_moments",
        "th_var_decomp_uncond",
        "th_var_decomp_cond_h1",
        "th_var_decomp_cond_h4",
        "th_corr_matrix",
        "th_autocorr_matrix",
    ] {
        let got = std::fs::read_to_string(dir.join(format!("rbc_q2/latex/rbc_q2_{t}.tex"))).unwrap();
        let want = std::fs::read_to_string(root().join(format!("reference/q2/latex_c1/rbc_q2_{t}.tex"))).unwrap();
        let strip = |s: &str| s.lines().skip(1).collect::<Vec<_>>().join("\n");
        assert_eq!(strip(&got), strip(&want), "{t}.tex");
    }
    // command 2: simulated moments in logs with MATLAB's shocks
    let (res, _) = run(&c, 1, &r, &dir);
    let cmd = &r["cmds"][1];
    check_simulated(&res, cmd, n, 3, nexo, 600);
    for t in ["sim_moments", "sim_corr_matrix", "sim_autocorr_matrix", "sim_var_decomp"] {
        let got = std::fs::read_to_string(dir.join(format!("rbc_q2/latex/rbc_q2_{t}.tex"))).unwrap();
        let want = std::fs::read_to_string(root().join(format!("reference/q2/latex_c1/rbc_q2_{t}.tex"))).unwrap();
        let strip = |s: &str| s.lines().skip(1).collect::<Vec<_>>().join("\n");
        assert_eq!(strip(&got), strip(&want), "{t}.tex");
    }
}

/// The printed decision rule and theoretical tables of the loglinear command are Dynare's.
#[test]
fn loglinear_printout_matches_dynare_log() {
    let log = common::read_ref(root().join("reference/q2/q2_c1_matlab.log"));
    let c = setup(1);
    let (o, vars) = &c.opts[0];
    let sol = analysis::solve_q2(&c.mf, o, None).unwrap().unwrap();
    let ivar = analysis::var_indices(&c.mf.model, vars).unwrap();
    let mut text = analysis::model_summary(&c.mf.model, &sol);
    text.push('\n');
    text += &analysis::disp_dr_q2(&c.mf.model, &sol.dr, &ivar, o.dr_display_tol, true);
    let th = analysis::theoretical_moments_q2(&c.mf.model, &sol, &ivar, o, 1, None, None).unwrap();
    text += &th.text;
    assert!(log.contains(&text), "printout differs from Dynare's log:\n{text}");
}

#[test]
fn bandpass_filter() {
    let r = load(2).expect("reference/setup unavailable");
    let c = setup(2);
    let dir = scratch("c2");
    let n = c.mf.model.n_endo();
    let nexo = c.mf.model.n_exo();
    let (res, _) = run(&c, 0, &r, &dir);
    let cmd = &r["cmds"][0];
    let th = res.theoretical.as_ref().unwrap();
    assert_mat("var", &th.gamma0, &mat(&cmd["var"], n, n), 1e-9);
    for (l, a) in th.autocorr.iter().enumerate() {
        assert_mat("autocorr", a, &mat(&cmd["autocorr"][l], n, n), 1e-9);
    }
    assert_mat("vardec", &(th.vardec.as_ref().unwrap() * 100.0), &mat(&cmd["variance_decomposition"], n, nexo), 1e-8);
    assert_mat(
        "contemporaneous_correlation",
        res.contemporaneous_correlation.as_ref().unwrap(),
        &mat(&cmd["contemporaneous_correlation"], n, n),
        1e-9,
    );
    check_spectral(&res, cmd);
    // Baxter-King filtered simulated moments with MATLAB's shocks
    let (res, _) = run(&c, 1, &r, &dir);
    check_simulated(&res, &r["cmds"][1], n, 4, nexo, 400);
    let log = std::fs::read_to_string(root().join("reference/q2/q2_c2_matlab.log")).unwrap();
    let (o, vars) = &c.opts[1];
    let names: Vec<String> = vars.clone();
    let ivar = analysis::var_indices(&c.mf.model, vars).unwrap();
    let text =
        analysis::simulated_moments_report_q2(&c.mf.model, res.simulated.as_ref().unwrap(), &names, o, 1, None, &ivar)
            .unwrap();
    assert!(log.contains(&text), "printout differs from Dynare's log:\n{text}");
}

#[test]
fn one_sided_hp_filter() {
    let r = load(3).expect("reference/setup unavailable");
    let c = setup(3);
    let dir = scratch("c3");
    let n = c.mf.model.n_endo();
    let (res, _) = run(&c, 0, &r, &dir);
    let cmd = &r["cmds"][0];
    check_simulated(&res, cmd, n, 4, c.mf.model.n_exo(), 400);
    assert_mat(
        "contemporaneous_correlation",
        res.contemporaneous_correlation.as_ref().unwrap(),
        &mat(&cmd["contemporaneous_correlation"], 4, 4),
        1e-9,
    );
    // theoretical moments are refused, as in Dynare
    let (o, _) = &c.opts[0];
    let mut o = o.clone();
    o.periods = 0;
    let sol = analysis::solve_q2(&c.mf, &o, None).unwrap().unwrap();
    assert!(analysis::theoretical_moments_q2(&c.mf.model, &sol, &[0], &o, 1, None, None).is_err());
}

#[test]
fn hp_filter_spectral_density() {
    let r = load(8).expect("reference/setup unavailable");
    let c = setup(8);
    let dir = scratch("c8");
    let (res, _) = run(&c, 0, &r, &dir);
    check_spectral(&res, &r["cmds"][0]);
}

#[test]
fn cycle_and_logarithmic_reduction() {
    let r = load(4).expect("reference/setup unavailable");
    let c = setup(4);
    let dir = scratch("c4");
    for k in 0..3 {
        let (res, sol) = run(&c, k, &r, &dir);
        let cmd = &r["cmds"][k];
        // iterative solvers: Dynare's MEX and this port stop at the same tolerance
        check_dr(&c, &sol, cmd, 1e-7);
        check_irfs(&c, &res, cmd, 1e-7);
        assert!(sol.dr.eigval.is_empty());
    }
    // and they agree with the QZ solution
    let (o, _) = &c.opts[1];
    let mut q = o.clone();
    q.dr_cycle_reduction = false;
    q.dr_logarithmic_reduction = false;
    let a = analysis::solve_q2(&c.mf, o, None).unwrap().unwrap();
    let b = analysis::solve_q2(&c.mf, &q, None).unwrap().unwrap();
    assert_mat("log reduction vs QZ", &a.dr.ghx, &b.dr.ghx, 1e-9);
}

// ---- order >= 2 ----

fn korder_opts(c: &Case, k: usize) -> dynare::solver::korder::stoch_simul::StochSimulOptions {
    let mut so = dynare::solver::korder::stoch_simul::StochSimulOptions::from_analysis(&c.opts[k].0);
    so.noprint = true;
    so
}

/// Order 2, simulated moments with the HP filter and the one-shock-at-a-time decomposition
/// (simult_ at order 2), fed with MATLAB's shocks.
#[test]
fn order2_simulated_moments_hp_filter() {
    use dynare::solver::korder::{simulate, solve_model};
    let r = load(5).expect("reference/setup unavailable");
    let c = setup(5);
    let (o, vars) = &c.opts[0];
    let cmd = &r["cmds"][0];
    let m = &c.mf.model;
    let ys = analysis::calibrated_steady_state(&c.mf, &m.params).unwrap();
    let dr = solve_model(m, &ys, &m.params, &c.mf.sigma_e, 2, Default::default()).unwrap();
    let nexo = m.n_exo();
    let ex = mat(&cmd["exo_simul"], o.periods, nexo);
    let y = simulate::simult(&dr, &ys, &ex, false);
    let endo = y.columns(1, o.periods).into_owned();
    let ivar = analysis::var_indices(m, vars).unwrap();
    let filter = analysis::q2::Filter::from_options(o).unwrap();
    let sm = analysis::simulate::simulated_moments_q2(
        &c.mf.sigma_e,
        &endo,
        &ex,
        &ivar,
        o.drop,
        o.ar,
        &filter,
        false,
        &|e| simulate::simult(&dr, &ys, e, false),
    )
    .unwrap();
    let res = analysis::StochSimulResults { endo_simul: Some(endo), simulated: Some(sm), ..Default::default() };
    check_simulated(&res, cmd, m.n_endo(), 4, nexo, o.periods);
    let sm = res.simulated.as_ref().unwrap();
    assert_mat("contemporaneous_correlation", &sm.corr, &mat(&cmd["contemporaneous_correlation"], 4, 4), 1e-9);
    // printout (the order > 1 note)
    let log = std::fs::read_to_string(root().join("reference/q2/q2_c5_matlab.log")).unwrap();
    let text = analysis::simulated_moments_report_q2(m, sm, vars, o, 2, None, &ivar).unwrap();
    assert!(log.contains(&text), "printout differs from Dynare's log:\n{text}");
}

/// Order 2 (with and without pruning) and order 3 (pruning) theoretical moments, variance
/// decompositions and contemporaneous correlations.
#[test]
fn order2_3_theoretical_moments() {
    use dynare::solver::korder::stoch_simul::stoch_simul;
    let r = load(6).expect("reference/setup unavailable");
    let c = setup(6);
    let nexo = c.mf.model.n_exo();
    for k in 0..3 {
        let so = korder_opts(&c, k);
        let out = stoch_simul(&c.mf, &so, &c.opts[k].1, 1).unwrap();
        let cmd = &r["cmds"][k];
        assert_vec("mean", &out.mean, &vec_of(&cmd["mean"]), 1e-9);
        assert_mat("var", &out.var, &mat(&cmd["var"], 4, 4), 1e-9);
        for (l, a) in out.autocorr.iter().enumerate() {
            assert_mat("autocorr", a, &mat(&cmd["autocorr"][l], 4, 4), 1e-8);
        }
        if k < 2 {
            assert_mat(
                "vardec",
                out.variance_decomposition.as_ref().unwrap(),
                &mat(&cmd["variance_decomposition"], 4, nexo),
                1e-8,
            );
            let cvd = out.conditional_variance_decomposition.as_ref().unwrap();
            for (s, m) in cvd.iter().enumerate() {
                let want = Mat::from_fn(4, nexo, |i, e| num(&cmd["conditional_variance_decomposition"][i][s][e]));
                assert_mat("cvd", m, &want, 1e-9);
            }
        }
        if k == 1 {
            assert_mat(
                "contemporaneous_correlation",
                out.contemporaneous_correlation.as_ref().unwrap(),
                &mat(&cmd["contemporaneous_correlation"], 4, 4),
                1e-8,
            );
        }
    }
}

/// `relative_irf` at order 2 (pruning): GIRFs to a shock of 0.01 (`cs(:,i)/cs(i,i)/100`).
/// The draws are not MATLAB's, so the comparison is statistical (2000 replications).
#[test]
fn order2_relative_irf() {
    use dynare::solver::korder::stoch_simul::stoch_simul;
    let r = load(7).expect("reference/setup unavailable");
    let c = setup(7);
    let so = korder_opts(&c, 0);
    let out = stoch_simul(&c.mf, &so, &c.opts[0].1, 7).unwrap();
    let cmd = &r["cmds"][0];
    for (key, y) in &out.irfs {
        let want = vec_of(&cmd["irfs"][key]);
        let scale = want.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        for (a, b) in y.iter().zip(&want) {
            assert!((a - b).abs() <= 0.02 * scale + 1e-12, "{key}: {a} vs {b}");
        }
    }
}

/// Options Dynare refuses at order > 1.
#[test]
fn order2_refused_options() {
    use dynare::solver::korder::stoch_simul::stoch_simul;
    let c = setup(6);
    // (dr=cycle_reduction is refused with k_order_solver only: at order 2 without it, the
    // first-order step uses cycle reduction, tests/stoch_simul2.rs)
    for f in [
        |a: &mut StochSimulOptions| a.loglinear = true,
        |a: &mut StochSimulOptions| (a.dr_cycle_reduction, a.k_order_solver) = (true, true),
        |a: &mut StochSimulOptions| a.aim_solver = true,
    ] {
        let mut a = c.opts[0].0.clone();
        f(&mut a);
        let mut so = dynare::solver::korder::stoch_simul::StochSimulOptions::from_analysis(&a);
        so.noprint = true;
        assert!(stoch_simul(&c.mf, &so, &[], 1).is_err());
    }
    let mut a = c.opts[0].0.clone();
    a.hp_filter = 1600.0;
    let mut so = dynare::solver::korder::stoch_simul::StochSimulOptions::from_analysis(&a);
    so.noprint = true;
    assert!(stoch_simul(&c.mf, &so, &[], 1).is_err(), "theoretical filtered moments at order 2");
}
