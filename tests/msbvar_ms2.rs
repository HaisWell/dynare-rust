//! MS-SBVAR part 2b: `ms_irf`, `ms_variance_decomposition`, `ms_forecast` against Dynare 7.2
//! (reference/msbvar/ms/ref_ms2.json from gen_ref2.py; model ms_small, commands of ms_small2.mod). Deterministic
//! outputs are evaluated at Dynare's posterior mode; forecasts are simulation based and compared statistically.

use std::path::Path;
use std::sync::Arc;

use dynare::linalg::Mat;
use dynare::modfile::{Command, parse_mod_file};
use dynare::msbvar::ms_estimation::{MsOptions, MsSetup};
use dynare::msbvar::ms_sim::{self, Kind, Mode};
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

fn maxdiff(a: &Mat, b: &Mat) -> f64 {
    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()), "shape");
    (a - b).iter().fold(0.0, |m, x| m.max(x.abs()))
}

#[test]
fn irf_per_regime() {
    let (s, th) = setup_and_theta();
    let r = load("ref_ms2.json");
    for (k, key) in [(0, "irf_r0"), (1, "irf_r1")] {
        let got = ms_sim::compute(&s, std::slice::from_ref(&th), &Mode::Regime(k), Kind::Irf, 6, &[0.5]).unwrap();
        let d = maxdiff(&got, &matf(&r[key]));
        println!("{key}: {d:e}");
        assert!(d < 2e-5, "{key} {d}");
    }
}

#[test]
fn irf_mixtures() {
    let (s, th) = setup_and_theta();
    let r = load("ref_ms2.json");
    let pc: Vec<f64> = (1..20).map(|i| i as f64 * 0.05).collect();
    for (mode, key) in [(Mode::Filtered, "irf_filt19"), (Mode::Ergodic, "irf_erg19")] {
        let got = ms_sim::compute(&s, std::slice::from_ref(&th), &mode, Kind::Irf, 3, &pc).unwrap();
        let want = matf(&r[key]);
        for blk in 0..19 {
            let d =
                maxdiff(&got.view((blk * 3, 0), (3, 4)).into_owned(), &want.view((blk * 3, 0), (3, 4)).into_owned());
            assert!(d < 2e-5, "{key} block {blk}: {d}");
        }
    }
    let got =
        ms_sim::compute(&s, std::slice::from_ref(&th), &Mode::Filtered, Kind::Irf, 6, &[0.16, 0.5, 0.84]).unwrap();
    assert!(maxdiff(&got, &matf(&r["irf_filt3"])) < 2e-5);
    let got = ms_sim::compute(&s, std::slice::from_ref(&th), &Mode::Ergodic, Kind::Irf, 6, &[0.5]).unwrap();
    assert!(maxdiff(&got, &matf(&r["irf_erg_med"])) < 2e-5);
}

#[test]
fn variance_decomposition_means() {
    let (s, th) = setup_and_theta();
    let r = load("ref_ms2.json");
    // per regime: deterministic (2e-5 = precision of the printed parameters); ergodic / filtered: Dynare averages
    // over 10000 sampled regimes, so the weights carry Monte-Carlo noise (about 5e-3, share sensitivity 0.19)
    for (mode, key, tol) in [
        (Mode::Regime(0), "vd_r0", 2e-5),
        (Mode::Regime(1), "vd_r1", 2e-5),
        (Mode::Ergodic, "vd_erg", 3e-3),
        (Mode::Filtered, "vd_filt", 3e-3),
    ] {
        let got = ms_sim::compute(&s, std::slice::from_ref(&th), &mode, Kind::VdMean, 6, &[]).unwrap();
        let d = maxdiff(&got, &matf(&r[key]));
        assert!(d < tol, "{key} {d}");
    }
}

#[test]
fn forecasts() {
    let (s, th) = setup_and_theta();
    let r = load("ref_ms2.json");
    // regime-specific forecasts keep the regime; the unconditional one starts from the filtered probabilities and
    // lets the regime move with Q. Percentiles of 100000 draws against Dynare's 10000: Monte-Carlo noise ~0.03.
    for (start, key, evolve) in [(Some(0), "fc_reg1_3", false), (Some(1), "fc_reg2_3", false), (None, "fc_all_3", true)]
    {
        let got =
            ms_sim::forecast(&s, std::slice::from_ref(&th), start, 4, 100000, &[0.1, 0.5, 0.9], evolve, 7).unwrap();
        let d = maxdiff(&got, &matf(&r[key]));
        assert!(d < 0.1, "{key}: {d}");
    }
}
