//! stoch_simul / check on reference/analysis/nk.mod against Dynare 7.2
//! (reference/analysis/ref_nk.json and nk_matlab.log, written by dump_stoch_simul.m).
mod common;

use std::path::Path;

use dynare::analysis::{self, StochSimulOptions, check, irf, moments, simulate, vardec};
use dynare::linalg::Mat;
use dynare::modfile::{Command, parse_mod_file};
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn load() -> Option<Value> {
    let p = root().join("reference/analysis/ref_nk.json");
    Some(common::json_ref(p))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vec_of(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(num).collect(),
        x => vec![num(x)],
    }
}

/// MATLAB matrix (row-major nested arrays; a vector may come flat).
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

struct Setup {
    mf: dynare::modfile::ModFile,
    sol: analysis::LinearSolution,
    /// Options as Dynare has them at each stoch_simul command (they persist).
    opts: Vec<(StochSimulOptions, Vec<String>)>,
}

fn setup() -> Setup {
    let mf = parse_mod_file(&root().join("reference/analysis/nk.mod"), &[]).unwrap();
    let sol = analysis::solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
    let mut o = StochSimulOptions::default();
    let mut opts = Vec::new();
    for c in &mf.commands {
        if let Command::StochSimul(co, vars) = c {
            assert!(o.apply(co).is_empty());
            opts.push((o.clone(), vars.clone()));
        }
    }
    Setup { mf, sol, opts }
}

fn ivar_of(s: &Setup, k: usize) -> Vec<usize> {
    analysis::var_indices(&s.mf.model, &s.opts[k].1).unwrap()
}

#[test]
fn nk_decision_rule() {
    let r = load().expect("reference/setup unavailable");
    let s = setup();
    let c = &r["cmds"][0];
    let ov: Vec<usize> = vec_of(&c["order_var"]).iter().map(|&x| x as usize - 1).collect();
    assert_eq!(ov, s.sol.dr.order_var);
    let n = s.mf.model.n_endo();
    assert_mat("ghx", &s.sol.dr.ghx, &mat(&c["ghx"], n, s.sol.dr.nspred), 1e-12);
    assert_mat("ghu", &s.sol.dr.ghu, &mat(&c["ghu"], n, s.mf.model.n_exo()), 1e-12);
}

#[test]
fn nk_theoretical_moments_and_irfs() {
    let r = load().expect("reference/setup unavailable");
    let s = setup();
    let c = &r["cmds"][0];
    let (o, _) = &s.opts[0];
    let ivar = ivar_of(&s, 0);
    let nvar = ivar.len();
    let th = moments::th_autocovariances(&s.sol.dr, &s.sol.sigma_e, &ivar, &o.moment_options()).unwrap();
    // p and q have unit roots
    assert_eq!(th.stationary, vec![0, 1, 2, 3, 6, 7]);
    let g = &c["gamma_y"];
    assert_mat("gamma_y{1}", &th.gamma0, &mat(&g[0], nvar, nvar), 1e-10);
    for k in 0..o.ar {
        assert_mat(&format!("gamma_y{{{}}}", k + 2), &th.autocorr[k], &mat(&g[k + 1], nvar, nvar), 1e-10);
    }
    let nexo = s.mf.model.n_exo();
    let vd = mat(&c["variance_decomposition"], nvar, nexo);
    assert_mat("variance_decomposition", &(th.vardec.as_ref().unwrap() * 100.0), &vd, 1e-10);
    let mean = vec_of(&c["mean"]);
    for (a, b) in th.mean(&s.sol.dr).iter().zip(&mean) {
        assert!(close(*a, *b, 1e-12));
    }
    // conditional variance decomposition: [var][step][shock]
    let steps = &o.conditional_variance_decomposition;
    let cvd = vardec::conditional_variance_decomposition(&s.sol.dr, &s.sol.sigma_e, steps, &ivar).unwrap();
    let rc = &c["conditional_variance_decomposition"];
    for (h, m) in cvd.iter().enumerate() {
        let want = Mat::from_fn(nvar, nexo, |i, j| num(&rc[i][h][j]));
        assert_mat(&format!("cvd step {}", steps[h]), m, &want, 1e-10);
    }
    // IRFs (oo_.irfs.<var>_<shock>)
    let cs = vardec::lower_cholesky(&s.sol.sigma_e, 1e-14);
    let shocks = irf::irf_shock_indices(&s.mf.model.exo_names, None).unwrap();
    let irfs = irf::stoch_simul_irfs(&s.sol.dr, &cs, &ivar, &shocks, o.irf, false);
    assert_eq!(irfs.len(), nexo);
    let mut n = 0;
    for si in &irfs {
        for (k, &v) in ivar.iter().enumerate() {
            let key = format!("{}_{}", s.mf.model.endo_names[v], s.mf.model.exo_names[si.shock]);
            let want = vec_of(&c["irfs"][&key]);
            for t in 0..o.irf {
                assert!(close(si.y[(k, t)], want[t], 1e-10), "{key}[{t}]");
            }
            n += 1;
        }
    }
    assert_eq!(n, nvar * nexo);
}

#[test]
fn nk_hp_filtered_moments_and_relative_irfs() {
    let r = load().expect("reference/setup unavailable");
    let s = setup();
    let c = &r["cmds"][1];
    let (o, _) = &s.opts[1];
    assert_eq!(o.hp_filter, 1600.0);
    let ivar = ivar_of(&s, 1);
    let nvar = ivar.len();
    let th = moments::th_autocovariances(&s.sol.dr, &s.sol.sigma_e, &ivar, &o.moment_options()).unwrap();
    let g = &c["gamma_y"];
    assert_mat("hp gamma_y{1}", &th.gamma0, &mat(&g[0], nvar, nvar), 1e-10);
    for k in 0..o.ar {
        assert_mat("hp autocorr", &th.autocorr[k], &mat(&g[k + 1], nvar, nvar), 1e-10);
    }
    let nexo = s.mf.model.n_exo();
    assert_mat(
        "hp vardec",
        &(th.vardec.as_ref().unwrap() * 100.0),
        &mat(&c["variance_decomposition"], nvar, nexo),
        1e-10,
    );
    let cs = vardec::lower_cholesky(&s.sol.sigma_e, 1e-14);
    let shocks = irf::irf_shock_indices(&s.mf.model.exo_names, o.irf_shocks.as_deref()).unwrap();
    assert_eq!(shocks, vec![0, 2]);
    let irfs = irf::stoch_simul_irfs(&s.sol.dr, &cs, &ivar, &shocks, o.irf, o.relative_irf);
    for si in &irfs {
        for (k, &v) in ivar.iter().enumerate() {
            let key = format!("{}_{}", s.mf.model.endo_names[v], s.mf.model.exo_names[si.shock]);
            let want = vec_of(&c["irfs"][&key]);
            for t in 0..o.irf {
                assert!(close(si.y[(k, t)], want[t], 1e-10), "{key}[{t}]");
            }
        }
    }
}

/// Simulated moments fed with MATLAB's shocks (`oo_.exo_simul`).
#[test]
fn nk_simulated_moments_with_matlab_shocks() {
    let r = load().expect("reference/setup unavailable");
    let s = setup();
    let nexo = s.mf.model.n_exo();
    let n = s.mf.model.n_endo();
    for (k, cmd) in [(2usize, 2usize), (3, 3)] {
        let c = &r["cmds"][cmd];
        let (o, _) = &s.opts[k];
        let ivar = ivar_of(&s, k);
        let nvar = ivar.len();
        let ex = mat(&c["exo_simul"], o.periods, nexo);
        let y = simulate::simult(&s.sol.dr, &s.sol.dr.ys, &ex);
        let endo = y.columns(1, o.periods).into_owned();
        assert_mat("endo_simul", &endo, &mat(&c["endo_simul"], n, o.periods), 1e-10);
        let sm = simulate::simulated_moments(
            &s.sol.dr,
            &s.sol.sigma_e,
            &endo,
            &ex,
            &ivar,
            o.drop,
            o.ar,
            o.hp_filter,
            o.nodecomposition,
        )
        .unwrap();
        let cmp = |what: &str, a: &[f64], b: &[f64]| {
            for (x, y) in a.iter().zip(b) {
                assert!(close(*x, *y, 1e-9), "{what} cmd{}: {x} vs {y}", cmd + 1);
            }
        };
        cmp("mean", &sm.mean, &vec_of(&c["mean"]));
        cmp("skewness", &sm.skewness, &vec_of(&c["skewness"]));
        cmp("kurtosis", &sm.kurtosis, &vec_of(&c["kurtosis"]));
        assert_mat("var", &sm.var, &mat(&c["var"], nvar, nvar), 1e-9);
        for (l, a) in sm.autocorr.iter().enumerate() {
            assert_mat("autocorr", a, &mat(&c["autocorr"][l], nvar, nvar), 1e-9);
        }
        assert_mat("sim vardec", sm.vardec.as_ref().unwrap(), &mat(&c["variance_decomposition"], nvar, nexo), 1e-9);
    }
}

#[test]
fn nk_check_eigenvalues() {
    let r = load().expect("reference/setup unavailable");
    let s = setup();
    let ys = analysis::calibrated_steady_state(&s.mf, &s.mf.model.params).unwrap();
    let c = check::check(&s.mf.model, &s.mf.model.params, &ys, Default::default()).unwrap();
    assert!(c.result);
    assert_eq!(c.edim, Some(3));
    let re = vec_of(&r["cmds"][0]["eigval_re"]);
    let im = vec_of(&r["cmds"][0]["eigval_im"]);
    let mut want: Vec<f64> = re.iter().zip(&im).map(|(a, b)| a.hypot(*b)).filter(|m| *m < 1e10).collect();
    let mut got: Vec<f64> = c.eigval.iter().map(|z| z.norm()).filter(|m| *m < 1e10).collect();
    want.sort_by(f64::total_cmp);
    got.sort_by(f64::total_cmp);
    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    }
}

/// The printed tables of the first two stoch_simul commands are Dynare's, character for
/// character (nk_matlab.log is Dynare's diary of the same run).
#[test]
fn nk_printout_matches_dynare_log() {
    let log = common::read_ref(root().join("reference/analysis/nk_matlab.log"));
    let s = setup();
    for k in 0..2 {
        let (o, _) = &s.opts[k];
        let ivar = ivar_of(&s, k);
        let mut text = analysis::model_summary(&s.mf.model, &s.sol);
        text.push('\n');
        text += &analysis::disp_dr(&s.mf.model, &s.sol.dr, &ivar, o.dr_display_tol);
        let (_, _, th) = analysis::theoretical_moments_report(&s.mf.model, &s.sol, &ivar, o).unwrap();
        text += &th;
        assert!(log.contains(&text), "command {}: printout differs from Dynare's log:\n{text}", k + 1);
    }
    // check: everything but the numerically infinite eigenvalue
    let ys = analysis::calibrated_steady_state(&s.mf, &s.mf.model.params).unwrap();
    let c = check::check(&s.mf.model, &s.mf.model.params, &ys, Default::default()).unwrap();
    let out = check::format_check(&c);
    for line in out.lines().filter(|l| !l.contains("e+1")) {
        assert!(log.contains(line), "check line not in the log: {line:?}");
    }
}
