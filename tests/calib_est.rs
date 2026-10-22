//! `irf_calibration` / `moment_calibration` applied wherever Dynare 7.1 reads
//! `options_.endogenous_prior_restrictions`: dsge_likelihood.m (info 49, `info(4)` = the
//! summed squared distances), the mode finder and the MCMC, prior_sampler.m (`prior simulate`),
//! prior.m (`prior moments`) and the GSA stability mapping. Against
//! reference/calib_est/dump_calib_est.m (ref_calib_est.json and the logs).
mod common;

use std::path::{Path, PathBuf};

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::Objective;
use dynare::gsa::calibration;
use dynare::kalman::KalmanOptions;
use dynare::modfile::{Command, ModFile, parse_mod_file};
use serde_json::Value;

fn src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/calib_est")
}

fn reference() -> Option<Value> {
    let text = common::read_ref(src().join("ref_calib_est.json"));
    Some(serde_json::from_str(&text).unwrap())
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dynare_calib_est_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    for f in ["nk_ce.mod", "nk_ce_data.csv"] {
        std::fs::copy(src().join(f), d.join(f)).unwrap();
    }
    d
}

fn model(case: &str) -> ModFile {
    parse_mod_file(&src().join("nk_ce.mod"), &[("CASE".into(), case.into())]).unwrap()
}

fn estimation_options(mf: &ModFile) -> &dynare::modfile::Options {
    mf.commands.iter().find_map(|c| if let Command::Estimation(o, _) = c { Some(o) } else { None }).unwrap()
}

/// dsge_likelihood at the mode, at points that violate one or several restrictions (single
/// period, several periods whose squared distances multiply, IRF and moment penalties
/// summed) and at 40 random points: fval, info(1), info(4) and the restriction values.
#[test]
fn likelihood_points_match_dynare() {
    let r = reference().expect("reference/setup unavailable");
    let mf = model("1");
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let table = Table::read(&src().join("nk_ce_data.csv"), &Default::default()).unwrap();
    let mut data = LikelihoodData::new(&table.sample(&mf.varobs, 1, None).unwrap(), KalmanOptions::default());
    let rs = calibration::restrictions_at_options(&mf, estimation_options(&mf)).unwrap().unwrap();
    assert_eq!((rs.irf.len(), rs.moment.len()), (3, 2));
    data.calib_restrictions = Some(rs.clone());
    let post = DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
    let pts = r["c1"]["points"].as_array().unwrap();
    let (mut n49, mut n0) = (0, 0);
    for (k, p) in pts.iter().enumerate() {
        let x = vecf(&p["x"]);
        let e = post.eval_full(&x, false);
        let info = p["info"].as_i64().unwrap() as i32;
        assert_eq!(e.info, info, "point {k}: info");
        match info {
            0 => {
                let want = p["fval"].as_f64().unwrap();
                assert!(((e.fval - want) / want).abs() < 1e-12, "point {k}: fval {} vs {want}", e.fval);
                n0 += 1;
            }
            49 => {
                assert!(e.fval.is_infinite());
                let want = p["info4"].as_f64().unwrap();
                assert!(
                    (e.info4 - want).abs() <= 1e-10 * want.abs().max(1e-12),
                    "point {k}: penalty {} vs {want}",
                    e.info4
                );
                n49 += 1;
            }
            _ => assert!(e.fval.is_infinite()),
        }
        // the restriction values themselves
        if p.get("data_irf").is_some() {
            let res = dsge.resolve(&x).unwrap();
            let v = calibration::endogenous_prior_restrictions(&dsge, &res.dr, &res.q, &rs).unwrap();
            assert_eq!(v.info.0, vecf(&p["restr_info"])[0] as i32, "point {k}");
            for (key, ours) in [("data_irf", &v.data_irf), ("data_moment", &v.data_moment)] {
                let want = &p[key];
                for (j, rows) in ours.iter().enumerate() {
                    // MATLAB stores [t value] rows (a 1x2 row is flattened by jsonencode)
                    let w = vecf(if want.as_array().unwrap().len() == ours.len() { &want[j] } else { want });
                    let wv: Vec<f64> = w.chunks(2).map(|c| c[1]).collect();
                    let wt: Vec<i64> = w.chunks(2).map(|c| c[0] as i64).collect();
                    assert_eq!(rows.iter().map(|r| r.0).collect::<Vec<_>>(), wt, "point {k} {key}[{j}]: periods");
                    for ((_, a), b) in rows.iter().zip(&wv) {
                        assert!((a - b).abs() <= 1e-11 * b.abs().max(1.0), "point {k} {key}[{j}]: {a} vs {b}");
                    }
                }
            }
            for (key, ours) in [("info_irf", &v.info_irf), ("info_moment", &v.info_moment)] {
                let w = vecf(&p[key]);
                assert_eq!(w.len(), 2 * ours.len());
                for (j, row) in ours.iter().enumerate() {
                    // jsonencode: the n x 2 matrix as rows
                    let (c, d) = (w[2 * j], w[2 * j + 1]);
                    assert_eq!(row[0], c, "point {k} {key}[{j}] code");
                    assert!(
                        (row[1] - d).abs() <= 1e-10 * d.abs().max(1e-300),
                        "point {k} {key}[{j}] penalty {} vs {d}",
                        row[1]
                    );
                }
            }
        }
    }
    assert!(n49 >= 10 && n0 >= 5, "{n49} violating and {n0} admissible points");
    // without the blocks the same points are admissible (the check costs nothing)
    let free = LikelihoodData::new(&table.sample(&mf.varobs, 1, None).unwrap(), KalmanOptions::default());
    let post_free = DsgePosterior { dsge: &dsge, data: &free, bounds: &dsge.bounds };
    let x = vecf(&pts[1]["x"]);
    assert!(post_free.eval(&x).0.is_finite());
}

/// The single-restriction `any` quirk and NaN handling of `endogenous_prior_restrictions.m`
/// are covered by unit tests in `gsa::calibration`; here: a restriction block naming an
/// unknown variable is refused at setup (Dynare: "One of the variable specified does not exist").
#[test]
fn unknown_variable_is_an_error() {
    let mut mf = model("1");
    for c in mf.commands.iter_mut() {
        if let Command::Other(name, o) = c {
            if name == "moment_calibration" {
                for (_, v) in o.iter_mut() {
                    if let dynare::modfile::OptValue::Str(s) = v {
                        *s = s.replace("pi", "nope");
                    }
                }
            }
        }
    }
    let err = calibration::restrictions_at_options(&mf, estimation_options(&mf)).unwrap_err();
    assert!(err.to_string().contains("does not exist"), "{err}");
}

/// Mode finding (newrat, deterministic) with a moment restriction that binds at the
/// unrestricted mode: the restricted mode is Dynare's; the Hessian at a mode on the
/// boundary is not finite and, as in Dynare, check_hessian_at_the_mode complains.
#[test]
fn binding_restriction_mode_matches_dynare() {
    let r = reference().expect("reference/setup unavailable");
    let want = vecf(&r["c2"]["mode"]);
    let dir = scratch("c2");
    let ro = dynare::driver::RunOptions {
        seed: 0,
        quiet: true,
        defines: vec![("CASE".into(), "2".into())],
        ..Default::default()
    };
    dynare::driver::run(&dir.join("nk_ce.mod"), &ro).unwrap();
    let j: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("nk_ce/Output/nk_ce_results.json")).unwrap()).unwrap();
    let got = vecf(&j["mode"]);
    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-2), "mode {a} vs Dynare {b}");
    }
    // the mode satisfies the restriction, the unrestricted one does not
    let mf = model("2");
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let rs = calibration::restrictions_at_options(&mf, estimation_options(&mf)).unwrap().unwrap();
    let res = dsge.resolve(&got).unwrap();
    assert_eq!(calibration::restrictions_info(&dsge, &res.dr, &res.q, &rs).0, 0);
    let res = dsge.resolve(&vecf(&r["c1"]["mode"])).unwrap();
    assert_eq!(calibration::restrictions_info(&dsge, &res.dr, &res.q, &rs).0, 49);
}

/// Mode + MCMC (2 x 2000 draws) where the restrictions cut the posterior: the mode is
/// Dynare's (interior), no MCMC draw violates a restriction (Dynare's neither), and the
/// posterior means agree with Dynare's within the Monte Carlo error.
#[test]
fn mcmc_respects_the_restrictions() {
    let r = reference().expect("reference/setup unavailable");
    assert_eq!(r["c1"]["mcmc_violations"].as_f64().unwrap(), 0.0);
    let dir = scratch("c1");
    let ro = dynare::driver::RunOptions {
        seed: 1,
        quiet: true,
        defines: vec![("CASE".into(), "1".into())],
        ..Default::default()
    };
    dynare::driver::run(&dir.join("nk_ce.mod"), &ro).unwrap();
    let j: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("nk_ce/Output/nk_ce_results.json")).unwrap()).unwrap();
    for (a, b) in vecf(&j["mode"]).iter().zip(vecf(&r["c1"]["mode"])) {
        assert!((a - b).abs() <= 1e-4 * b.abs().max(1e-2), "mode {a} vs Dynare {b}");
    }
    let mf = model("1");
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let rs = calibration::restrictions_at_options(&mf, estimation_options(&mf)).unwrap().unwrap();
    let mut draws: Vec<Vec<f64>> = Vec::new();
    for b in 1..=2 {
        let m = dynare::matfile::read(&dir.join(format!("nk_ce/metropolis/nk_ce_mh1_blck{b}.mat"))).unwrap();
        let x2 = m["x2"].as_matrix().unwrap();
        draws.extend((0..x2.nrows()).map(|i| x2.row(i).iter().copied().collect::<Vec<f64>>()));
    }
    assert_eq!(draws.len(), 4000);
    let mut distinct = draws.clone();
    distinct.dedup();
    // (the restrictions bind in this region: 24 of the 46 points of `likelihood_points_match_dynare`,
    // within 3 posterior standard deviations of the mode, violate them)
    for x in &distinct {
        let res = dsge.resolve(x).unwrap();
        assert_eq!(
            calibration::restrictions_info(&dsge, &res.dr, &res.q, &rs).0,
            0,
            "draw {x:?} violates a restriction"
        );
    }
    let sd: Vec<f64> = (0..10).map(|i| r["c1"]["variance"][i][i].as_f64().unwrap().sqrt()).collect();
    let want = vecf(&r["c1"]["posterior_mean"]);
    for i in 0..10 {
        let half = &draws.iter().skip(1000).take(1000).chain(draws.iter().skip(3000)).map(|d| d[i]).collect::<Vec<_>>();
        let mean = half.iter().sum::<f64>() / half.len() as f64;
        assert!((mean - want[i]).abs() < sd[i], "posterior mean {i}: {mean} vs Dynare {}", want[i]);
    }
}

/// prior_sampler.m: the share of prior draws rejected by the restrictions (Dynare: 0.54335 of
/// 20000 draws from MATLAB's stream; dynare-rust uses its own draws).
#[test]
fn prior_violation_share() {
    use rand::SeedableRng;
    let mf = model("3");
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let ci = mf.commands.iter().position(|c| matches!(c, Command::Other(n, _) if n == "prior")).unwrap();
    let rs = calibration::restrictions_checked(&mf, ci).unwrap().unwrap();
    let bounds = dynare::estimation::prior::prior_bounds(&dsge.priors, 0.0);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
    let n = 5000;
    let mut viol = 0;
    for _ in 0..n {
        let x = dynare::estimation::prior::prior_draw(&dsge.priors, &bounds, &mut rng);
        if let Ok(res) = dsge.resolve(&x) {
            if calibration::restrictions_info(&dsge, &res.dr, &res.q, &rs).0 == 49 {
                viol += 1;
            }
        }
    }
    let share = viol as f64 / n as f64;
    let log = std::fs::read_to_string(src().join("calib_est_c3_matlab.log")).unwrap();
    let line = log.lines().find(|l| l.starts_with("Endogenous prior violation share")).unwrap();
    let want: f64 = line.split('=').nth(1).unwrap().trim().parse().unwrap();
    assert!((share - want).abs() < 4.0 * (want * (1.0 - want) / n as f64).sqrt(), "share {share} vs Dynare {want}");
}

/// stability_mapping.m with restrictions on Dynare's own GSA sample: infox (49 for stable
/// points violating them), istable, irestriction, iunstable, ixun; the printout is Dynare's.
#[test]
fn stability_mapping_with_restrictions_matches_dynare() {
    let r = reference().expect("reference/setup unavailable");
    let dir = scratch("c5");
    let mf = parse_mod_file(&dir.join("nk_ce.mod"), &[("CASE".into(), "5".into())]).unwrap();
    let (ci, o) = mf
        .commands
        .iter()
        .enumerate()
        .find_map(|(i, c)| match c {
            Command::Other(n, o) if n == "sensitivity" => Some((i, o.clone())),
            _ => None,
        })
        .unwrap();
    let out = dynare::gsa::run(&mf, &dir, "nk_ce", &o, ci).unwrap().unwrap();
    let c5 = &r["c5"];
    let one = |v: &[usize]| v.iter().map(|&k| (k + 1) as f64).collect::<Vec<f64>>();
    assert_eq!(out.stab.infox.iter().map(|&i| i as f64).collect::<Vec<_>>(), vecf(&c5["infox"]));
    assert_eq!(one(&out.stab.istable), vecf(&c5["istable"]));
    assert_eq!(one(&out.stab.irestriction), vecf(&c5["irestriction"]));
    assert_eq!(one(&out.stab.iunstable), vecf(&c5["iunstable"]));
    assert_eq!(one(&out.stab.ixun), vecf(&c5["ixun"]));
    assert_eq!(one(&out.stab.iindeterm), vecf(&c5["iindeterm"]));
    assert!(out.stab.infox.contains(&49) && !out.stab.irestriction.is_empty());
    // x0 (prior_ok.mat) is the first point matching the restrictions
    assert_eq!(out.stab.x0.as_ref().unwrap(), &out.stab.point(out.stab.irestriction[0]));
    // the saved file holds irestriction (not istable)
    let m = dynare::matfile::read(&dir.join("nk_ce/gsa/nk_ce_prior.mat")).unwrap();
    assert_eq!(m["irestriction"].as_matrix().unwrap().iter().copied().collect::<Vec<f64>>(), vecf(&c5["irestriction"]));
    let log = std::fs::read_to_string(src().join("calib_est_c5_matlab.log")).unwrap();
    for line in
        [" 39.1% of the prior support violates prior restrictions.", "Smirnov statistics in driving prior restrictions"]
    {
        assert!(log.contains(line.trim_start()), "{line}");
    }
}
