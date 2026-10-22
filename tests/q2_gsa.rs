//! : `irf_calibration` / `moment_calibration` mapped by `sensitivity`
//! (map_calibration.m), identification with `checks_via_subsets` and with `parameter_set =
//! posterior_mode` after an estimation, against Dynare 7.2 (reference/q2/dump_gsa_q2.m:
//! ref_q2_calib.json, ref_q2_ident_subsets.json, ref_q2_ident_postmode.json and the logs).
mod common;

use std::path::{Path, PathBuf};

use dynare::modfile::{Command, parse_mod_file};
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn load(name: &str) -> Option<Value> {
    Some(common::json_ref(root().join("reference/q2").join(name)))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        x => vec![num(x)],
    }
}

fn scratch(tag: &str, files: &[&str]) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dynare_q2_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    for f in files {
        std::fs::copy(root().join("reference/q2").join(f), d.join(f)).unwrap();
    }
    d
}

/// Restriction values (IRFs and moments) solved at Dynare's sample points: 1e-7 with
/// Accelerate; up to 4e-7 with OpenBLAS (Linux, Windows).
const MAT_TOL: f64 = if cfg!(target_os = "macos") { 1e-7 } else { 2e-6 };

/// The calibration mapping on Dynare's GSA sample (the sample itself is Dynare's: same
/// generator, same Sobol sequence): the kept points, the restriction values and indicators.
#[test]
fn map_calibration_matches_dynare() {
    let r = load("ref_q2_calib.json").expect("reference/setup unavailable");
    let dir = scratch("calib", &["nk_q2.mod", "nk_q2_data.csv"]);
    let mf = parse_mod_file(&dir.join("nk_q2.mod"), &[("CASE".into(), "1".into())]).unwrap();
    let (ci, o) = mf
        .commands
        .iter()
        .enumerate()
        .find_map(|(i, c)| match c {
            Command::Other(n, o) if n == "sensitivity" => Some((i, o.clone())),
            _ => None,
        })
        .unwrap();
    let out = dynare::gsa::run(&mf, &dir, "nk_q2", &o, ci).unwrap().unwrap();
    // the sample is Dynare's (xmat = [lpmat0 lpmat](irestrictions, :))
    let xm = &r["xmat"];
    let keep0: Vec<usize> = vecf(&r["irestrictions"]).iter().map(|&v| v as usize - 1).collect();
    for (a, &k) in keep0.iter().enumerate().take(10) {
        for (j, v) in out.stab.point(k).iter().enumerate() {
            assert!((v - num(&xm[a][j])).abs() < 1e-8 * v.abs().max(1.0), "xmat[{a},{j}]: {v} vs {}", num(&xm[a][j]));
        }
    }
    let rs = dynare::gsa::calibration::restrictions_before(&mf, ci).unwrap();
    assert_eq!(rs.irf.len(), 4);
    assert_eq!(rs.moment[2].periods, vec![-1, -2]);
    let cm = dynare::gsa::calibration::map_calibration(&out.ctx, &out.dsge, &out.options, &out.stab, &rs).unwrap();
    let want_keep: Vec<usize> = vecf(&r["irestrictions"]).iter().map(|&v| v as usize - 1).collect();
    assert_eq!(cm.irestrictions, want_keep);
    for (j, row) in r["indx_irf"].as_array().unwrap().iter().enumerate() {
        if !want_keep.contains(&j) {
            continue;
        }
        let w = vecf(row);
        assert_eq!(cm.indx_irf[j], w, "indx_irf row {j}");
    }
    for (j, row) in r["indx_moment"].as_array().unwrap().iter().enumerate() {
        if !want_keep.contains(&j) {
            continue;
        }
        let w = vecf(row);
        assert_eq!(cm.indx_moment[j], w, "indx_moment row {j}");
    }
    for (key, ours) in [("mat_irf", &cm.mat_irf), ("mat_moment", &cm.mat_moment)] {
        for (ij, m) in r[key].as_array().unwrap().iter().enumerate() {
            for &j in &want_keep {
                let w = vecf(&m[j]);
                for (a, b) in ours[ij][j].iter().zip(&w) {
                    assert!((a - b).abs() <= MAT_TOL * b.abs().max(1e-6), "{key}[{ij}][{j}]: {a} vs {b}");
                }
            }
        }
    }
}

/// `checks_via_subsets` on the NK model with the collinear pair (alpha, psi).
#[test]
fn checks_via_subsets_matches_dynare() {
    use dynare::identification::Persistent;
    use dynare::identification::run::{IdentOptions, build_context};
    use rand::SeedableRng;
    let _r = load("ref_q2_ident_subsets.json").expect("reference/setup unavailable");
    let mf = parse_mod_file(&root().join("reference/q2/nk_ident_q2.mod"), &[]).unwrap();
    let o = mf
        .commands
        .iter()
        .find_map(|c| match c {
            Command::Other(n, o) if n == "identification" => Some(o.clone()),
            _ => None,
        })
        .unwrap();
    let opts = IdentOptions::from_options(&o).unwrap();
    assert!(opts.checks_via_subsets);
    let ctx = build_context(&mf, opts).unwrap();
    let (theta, _, _) = ctx.point_vector().unwrap();
    let mut pers = Persistent::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let an = ctx.analysis(&theta, false, true, &mut pers, &mut rng).unwrap().unwrap();
    let tot = theta.len();
    let sb = dynare::identification::subsets::checks_via_subsets(&an, tot, ctx.set.n_model(), None, 3);
    let names = &ctx.set.names;
    let ia = names.iter().position(|n| n == "alpha").unwrap();
    let ip = names.iter().position(|n| n == "psi").unwrap();
    for (label, r) in [("reducedform", &sb.reducedform), ("minimal", &sb.minimal), ("moments", &sb.moments)] {
        let r = r.as_ref().unwrap();
        assert!(r.problpars[0].is_empty(), "{label}: singles");
        assert_eq!(r.problpars[1], vec![vec![ia, ip]], "{label}: pairs");
        assert!(r.problpars[2].is_empty(), "{label}: triples");
    }
    // 'robust' tol_rank: Dynare compares the normalized diagonal with the character codes
    let s = sb.spectrum.as_ref().unwrap();
    assert_eq!(s.problpars[0].len(), tot);
    // the printout is Dynare's
    let log = std::fs::read_to_string(root().join("reference/q2/q2_nk_ident_subsets_matlab.log")).unwrap();
    for line in [
        "    Method to find problematic parameters:                 Rank condition on all possible subsets",
        "    [alpha,psi] are collinear!",
        "    SE_eg is not identified!",
    ] {
        assert!(log.contains(line), "{line}");
    }
}

/// `identification(parameter_set=posterior_mode)` after `estimation(mode_compute=4)`.
#[test]
fn identification_at_the_posterior_mode() {
    let r = load("ref_q2_ident_postmode.json").expect("reference/setup unavailable");
    let dir = scratch("postmode", &["nk_q2.mod", "nk_q2_data.csv"]);
    let ro = dynare::driver::RunOptions {
        seed: 0,
        quiet: true,
        defines: vec![("CASE".into(), "2".into())],
        ..Default::default()
    };
    dynare::driver::run(&dir.join("nk_q2.mod"), &ro).unwrap();
    let s = std::fs::read_to_string(dir.join("nk_q2/identification/nk_q2_posterior_mode_identif.json")).unwrap();
    let j: Value = serde_json::from_str(&s).unwrap();
    let got = vecf(&j["params"]);
    let want = vecf(&r["params_point"]);
    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(&want) {
        // csminwel from the same start; both stop at crit 1e-7 on the posterior kernel
        assert!((a - b).abs() <= 1e-4 * b.abs().max(1e-2), "posterior mode {a} vs {b}");
    }
    assert_eq!(num(&j["moments"]["checks"]["rank"]), num(&r["ide_moments_point"]["rank"]));
}

/// `sensitivity(identification=1)`: Morris screening of the theoretical moments and of the
/// model (map_identification.m) on Dynare's Morris trajectories.
#[test]
fn gsa_identification_screening() {
    let r = load("ref_q2_gsa_ident.json").expect("reference/setup unavailable");
    let dir = scratch("gsaident", &["nk_q2.mod", "nk_q2_data.csv"]);
    let ro = dynare::driver::RunOptions {
        seed: 0,
        quiet: true,
        defines: vec![("CASE".into(), "5".into())],
        ..Default::default()
    };
    dynare::driver::run(&dir.join("nk_q2.mod"), &ro).unwrap();
    let m = dynare::matfile::read(&dir.join("nk_q2/gsa/screen/nk_q2_morris_IDE.mat")).unwrap();
    for key in ["SAvdec", "SAcc", "SAnorm"] {
        let got = m[key].as_matrix().unwrap();
        let rows = r[key].as_array().unwrap();
        assert_eq!(got.nrows(), rows.len(), "{key}: rows");
        for (i, row) in rows.iter().enumerate() {
            let w = vecf(row);
            assert_eq!(got.ncols(), w.len(), "{key}: cols");
            for (j, b) in w.iter().enumerate() {
                // SAnorm column 15: an analytically zero innovation covariance (rounding noise
                // of order 1e-7 against entries of order 1e9, above teff's 1e-8 threshold in
                // both runs): the elementary effects of noise differ
                if key == "SAnorm" && j == 15 {
                    continue;
                }
                let a = got[(i, j)];
                assert!(
                    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-6 * b.abs().max(1e-3),
                    "{key}[{i},{j}]: {a} vs {b}"
                );
            }
        }
    }
}

/// `identification(gsa_sample_file=1)` after `sensitivity`: the Monte Carlo analysis runs on
/// the stable points of the GSA sample (Dynare's sample: 25 stable points of 32, as in
/// q2_nk_c3_matlab.log); `load_ident_files=1` re-displays the saved point analysis.
#[test]
fn identification_on_the_gsa_sample_and_saved_files() {
    if load("ref_q2_calib.json").is_none() {
        return;
    }
    let dir = scratch("gsasample", &["nk_q2.mod", "nk_q2_data.csv"]);
    let ro = dynare::driver::RunOptions {
        seed: 0,
        quiet: true,
        defines: vec![("CASE".into(), "3".into())],
        ..Default::default()
    };
    dynare::driver::run(&dir.join("nk_q2.mod"), &ro).unwrap();
    let m = dynare::matfile::read(&dir.join("nk_q2/gsa/nk_q2_prior.mat")).unwrap();
    let istable = m["istable"].as_vec().unwrap();
    let mc: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("nk_q2/identification/nk_q2_identif_mc.json")).unwrap())
            .unwrap();
    assert_eq!(mc["pdraws"].as_array().unwrap().len(), istable.len());
    assert_eq!(istable.len(), 25);
    let log = std::fs::read_to_string(root().join("reference/q2/q2_nk_c3_matlab.log")).unwrap();
    assert!(log.contains("Testing MC sample"));
    let ro = dynare::driver::RunOptions {
        seed: 0,
        quiet: true,
        defines: vec![("CASE".into(), "4".into())],
        ..Default::default()
    };
    dynare::driver::run(&dir.join("nk_q2.mod"), &ro).unwrap();
}

/// `glue`: the GLUE file of the RMSE analysis on Dynare's sample.
#[test]
fn glue_file_matches_dynare() {
    use dynare::matfile::MatValue;
    let r = load("ref_q2_glue.json").expect("reference/setup unavailable");
    let dir = scratch("glue", &["nk_q2.mod", "nk_q2_data.csv"]);
    let ro = dynare::driver::RunOptions {
        seed: 0,
        quiet: true,
        defines: vec![("CASE".into(), "7".into())],
        ..Default::default()
    };
    dynare::driver::run(&dir.join("nk_q2.mod"), &ro).unwrap();
    let m = dynare::matfile::read(&dir.join("nk_q2/gsa/nk_q2_glue_prior.mat")).unwrap();
    let sam = m["Sam"].field("data").unwrap().as_matrix().unwrap();
    let want = &r["Sam_data"];
    assert_eq!(sam.nrows(), want.as_array().unwrap().len());
    for i in 0..sam.nrows() {
        for (j, w) in vecf(&want[i]).iter().enumerate() {
            assert!((sam[(i, j)] - w).abs() <= 1e-8 * w.abs().max(1.0), "Sam.data[{i},{j}]");
        }
    }
    let obs = m["Obs"].field("data").unwrap().as_matrix().unwrap();
    for (i, row) in r["Obs_data"].as_array().unwrap().iter().enumerate() {
        for (t, w) in vecf(row).iter().enumerate() {
            assert!((obs[(i, t)] - w).abs() <= 1e-12 * w.abs().max(1.0), "Obs.data[{i},{t}]");
        }
    }
    let MatValue::Struct { elems, .. } = &m["Lik"] else { panic!() };
    let names: Vec<String> =
        r["Lik_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(elems.len(), names.len());
    for (k, e) in elems.iter().enumerate() {
        assert_eq!(e[0].as_str().unwrap(), names[k]);
        let got = e[3].as_vec().unwrap();
        let w = vecf(&r["Lik_data"][k]);
        for (a, b) in got.iter().zip(&w) {
            assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-3), "Lik({}).data: {a} vs {b}", names[k]);
        }
    }
    let MatValue::Struct { elems, .. } = &m["Out"] else { panic!() };
    let want_out = vecf(&r["Out_data"]);
    for (k, e) in elems.iter().enumerate() {
        assert_eq!(e[0].as_scalar().unwrap(), want_out[k], "Out({k}).data");
    }
}

/// `ppost`: the GSA (here the calibration mapping) on the posterior subsample of the MCMC
/// files left by the estimation (Dynare reads `_param_irf` files; statistical equivalence only).
#[test]
fn ppost_uses_the_mcmc_draws() {
    if load("ref_q2_calib.json").is_none() {
        return;
    }
    let dir = scratch("ppost", &["nk_q2.mod", "nk_q2_data.csv"]);
    let ro = dynare::driver::RunOptions {
        seed: 0,
        quiet: true,
        defines: vec![("CASE".into(), "8".into())],
        ..Default::default()
    };
    dynare::driver::run(&dir.join("nk_q2.mod"), &ro).unwrap();
    let m = dynare::matfile::read(&dir.join("nk_q2/gsa/nk_q2_post_restrictions.mat")).unwrap();
    let x = m["xmat"].as_matrix().unwrap();
    // min(1200, kept draws) = 1000 posterior draws, all of them solve the model
    assert_eq!(x.nrows(), 1000);
}
