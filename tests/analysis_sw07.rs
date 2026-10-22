//! Theoretical analysis on sw07.mod (two unit roots, 11 shocks) at the posterior mode
//! against Dynare 7.2 (reference/sw07/ref_stoch_simul.json, from reference/sw07/dump_stoch_simul.m).
mod common;

use std::path::Path;

use dynare::analysis::{StochSimulOptions, check, irf, moments, vardec};
use dynare::dsge::DsgeModel;
use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use dynare::solver::SolverOptions;
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
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

/// Absolute tolerance scaled by the magnitude of the reference entry.
fn assert_mat(what: &str, a: &Mat, b: &Mat, tol: f64) -> f64 {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    let mut worst: f64 = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let (x, y) = (a[(i, j)], b[(i, j)]);
            if x.is_nan() || y.is_nan() {
                assert!(x.is_nan() && y.is_nan(), "{what}[{i},{j}]: rust {x} vs dynare {y} (NaN pattern)");
                continue;
            }
            let e = (x - y).abs() / (1.0 + y.abs());
            worst = worst.max(e);
            assert!(e <= tol, "{what}[{i},{j}]: rust {x} vs dynare {y}");
        }
    }
    worst
}

struct Sw07 {
    dsge: DsgeModel,
    res: dynare::dsge::Resolved,
    r: Value,
}

fn setup() -> Option<Sw07> {
    let r: Value = common::json_ref(root().join("reference/sw07/ref_stoch_simul.json"));
    let mf = parse_mod_file(&root().join("reference/sw07/sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let xm = vec_of(&r["xparam_mode"]);
    let dsge = DsgeModel::new(&mf, Some(&xm)).unwrap();
    let res = dsge.resolve(&xm).unwrap();
    Some(Sw07 { dsge, res, r })
}

#[test]
fn sw07_theoretical_moments_with_unit_roots() {
    let p = setup().expect("reference/setup unavailable");
    let t = &p.r["theoretical"];
    let model = &p.dsge.model;
    let dr = &p.res.dr;
    let ov: Vec<usize> = vec_of(&t["order_var"]).iter().map(|&x| x as usize - 1).collect();
    assert_eq!(ov, dr.order_var);
    let ivar: Vec<usize> = (0..model.orig_endo_nbr).collect();
    let nvar = ivar.len();
    let nexo = model.n_exo();
    let mut o = StochSimulOptions { ar: 3, conditional_variance_decomposition: vec![1, 4, 12], ..Default::default() };
    o.irf = 12;
    let th = moments::th_autocovariances(dr, &p.res.q, &ivar, &o.moment_options()).unwrap();
    let mean = vec_of(&t["mean"]);
    let nonstat = mean.iter().filter(|m| m.is_nan()).count();
    assert_eq!(nvar - th.stationary.len(), nonstat, "number of nonstationary variables");
    for (a, b) in th.mean(dr).iter().zip(&mean) {
        assert!((a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-9 * (1.0 + b.abs()), "mean {a} vs {b}");
    }
    let g = &t["gamma_y"];
    let e0 = assert_mat("gamma_y{1}", &th.gamma0, &mat(&g[0], nvar, nvar), 1e-9);
    let mut ea: f64 = 0.0;
    for k in 0..3 {
        ea = ea.max(assert_mat("autocorr", &th.autocorr[k], &mat(&g[k + 1], nvar, nvar), 1e-9));
    }
    // Rows of zero-variance variables (variance < 1e-12, set to 0) are rounding
    // noise over rounding noise in Dynare too; they are not compared.
    let mut vd_r = th.vardec.as_ref().unwrap() * 100.0;
    let mut vd_d = mat(&t["variance_decomposition"], nvar, nexo);
    for i in 0..nvar {
        if th.gamma0[(i, i)] == 0.0 {
            vd_r.row_mut(i).fill(f64::NAN);
            vd_d.row_mut(i).fill(f64::NAN);
        }
    }
    let ev = assert_mat("variance_decomposition", &vd_r, &vd_d, 1e-9);
    let cvd =
        vardec::conditional_variance_decomposition(dr, &p.res.q, &o.conditional_variance_decomposition, &ivar).unwrap();
    let rc = &t["conditional_variance_decomposition"];
    let mut ec: f64 = 0.0;
    // Shares of variables whose h-step variance is at the rounding level  (a variable whose
    // impact response is zero at h = 1, where Dynare's ghu row is 1e-17..1e-37 noise) are noise
    // over noise in Dynare too and are not compared.
    let cv = vardec::conditional_variances(dr, &p.res.q, &o.conditional_variance_decomposition, &ivar).unwrap();
    for (h, m) in cvd.iter().enumerate() {
        let zero = |i: usize| th.gamma0[(i, i)] == 0.0 || cv[h].row(i).sum() < 1e-24;
        let want = Mat::from_fn(nvar, nexo, |i, j| if zero(i) { f64::NAN } else { num(&rc[i][h][j]) });
        let got = Mat::from_fn(nvar, nexo, |i, j| if zero(i) { f64::NAN } else { m[(i, j)] });
        ec = ec.max(assert_mat("cvd", &got, &want, 1e-9));
    }
    // IRFs of every variable to every shock
    let cs = vardec::lower_cholesky(&p.res.q, 1e-14);
    let shocks = irf::irf_shock_indices(&model.exo_names, None).unwrap();
    let irfs = irf::stoch_simul_irfs(dr, &cs, &ivar, &shocks, 12, false);
    let mut ei: f64 = 0.0;
    let mut n = 0;
    for si in &irfs {
        for (k, &v) in ivar.iter().enumerate() {
            let key = format!("{}_{}", model.endo_names[v], model.exo_names[si.shock]);
            if let Some(w) = t["irfs"].get(&key) {
                let w = vec_of(w);
                for tt in 0..12 {
                    let e = (si.y[(k, tt)] - w[tt]).abs() / (1.0 + w[tt].abs());
                    assert!(e < 1e-9, "{key}[{tt}]: {} vs {}", si.y[(k, tt)], w[tt]);
                    ei = ei.max(e);
                }
                n += 1;
            }
        }
    }
    // every pair Dynare stores (all 45 x 11 are above its 1e-10 threshold)
    assert_eq!(n, t["irfs"].as_object().unwrap().len());
    assert_eq!(n, 495);
    println!(
        "SW07 max rel. errors: gamma0 {e0:.1e}, autocorr {ea:.1e}, vardec {ev:.1e}, cvd {ec:.1e}, irfs {ei:.1e} ({n} IRFs)"
    );
}

#[test]
fn sw07_hp_filtered_moments() {
    let p = setup().expect("reference/setup unavailable");
    let h = &p.r["hp"];
    let model = &p.dsge.model;
    let names: Vec<String> =
        h["var_list"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let ivar: Vec<usize> = names.iter().map(|n| model.endo_index(n).unwrap()).collect();
    let nvar = ivar.len();
    let o = StochSimulOptions { ar: 2, hp_filter: 1600.0, ..Default::default() };
    let th = moments::th_autocovariances(&p.res.dr, &p.res.q, &ivar, &o.moment_options()).unwrap();
    let g = &h["gamma_y"];
    let e0 = assert_mat("hp gamma0", &th.gamma0, &mat(&g[0], nvar, nvar), 1e-9);
    for k in 0..2 {
        assert_mat("hp autocorr", &th.autocorr[k], &mat(&g[k + 1], nvar, nvar), 1e-9);
    }
    let ev = assert_mat(
        "hp vardec",
        &(th.vardec.as_ref().unwrap() * 100.0),
        &mat(&h["variance_decomposition"], nvar, model.n_exo()),
        1e-9,
    );
    println!("SW07 HP: gamma0 {e0:.1e}, vardec {ev:.1e}");
}

/// check at the mode: same order/rank verdict, same numbers of stable and explosive roots, and
/// the same well-conditioned eigenvalues. SW07's pencil also has exact zeros and infinities
/// (computed as 1e-17 / 1e16 noise) and a defective double zero root (MATLAB: 1.3e-8 x2, the
/// eps^(1/2) noise level, LAPACK-build dependent), so only roots with modulus in [1e-6, 1e6]
/// are compared: all 25 finite nonzero ones (0.17 .. 1.26, the double unit root included).
#[test]
fn sw07_check_eigenvalues() {
    let p = setup().expect("reference/setup unavailable");
    let c = check::check(&p.dsge.model, &p.res.params, &p.res.steady_state, SolverOptions::default()).unwrap();
    assert!(c.result);
    assert_eq!(c.edim.unwrap(), c.nsfwrd);
    let re = vec_of(&p.r["check"]["eigval_re"]);
    let im = vec_of(&p.r["check"]["eigval_im"]);
    let want_all: Vec<f64> =
        re.iter().zip(&im).map(|(a, b)| if a.is_nan() { f64::INFINITY } else { a.hypot(*b) }).collect();
    let got_all: Vec<f64> = c.eigval.iter().map(|z| z.norm()).collect();
    let crit = 1.0 + 1e-6;
    assert_eq!(got_all.iter().filter(|m| **m < crit).count(), want_all.iter().filter(|m| **m < crit).count(), "sdim");
    let band = |v: &[f64]| {
        let mut b: Vec<f64> = v.iter().copied().filter(|m| (1e-6..1e6).contains(m)).collect();
        b.sort_by(f64::total_cmp);
        b
    };
    let (got, want) = (band(&got_all), band(&want_all));
    assert_eq!(got.len(), want.len());
    assert_eq!(want.len(), 25);
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-8 * (1.0 + b), "{a} vs {b}");
    }
}
