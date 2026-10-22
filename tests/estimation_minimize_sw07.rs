//! Every `mode_compute` through `optim::minimize` (`dynare_minimize_objective.m`) on the SW07
//! posterior (STAGE 9 setup: first_obs=5, presample=8, diffuse_filter), compared with the
//! mode found by csminwel (mode_compute=4, a validated port of csminwel1.m) from the
//! `estimated_params` init values (the calibration): minus log posterior 349.856072, the same
//! point as `reference/sw07/ref.json` points[0] (the stage-2 newrat mode of MATLAB + Dynare
//! 7.2). Measured from the init values (374.5050): csminwel 349.856072 (103 iterations, 4392
//! evaluations), newrat (5) 349.856381, the quasi-Newton substitutes (1, 3) 349.856432,
//! simplex (8) 349.856088 (stops at MaxIter), solveopt (101) 349.856069.
//!
//! `MC=8,101 START=init|mode OPTIM="MaxIter,100" SEED=0 VERBOSE=1 \
//!  cargo test --release --test estimation_minimize_sw07 -- --ignored --nocapture`

use dynare::data::Table;
use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
use dynare::estimation::Objective;
use dynare::estimation::optim::{MinimizeContext, minimize};
use dynare::kalman::KalmanOptions;
use dynare::modfile::{OptValue, parse_mod_file};

/// csminwel's mode from the init values (see the module doc).
const SW07_MODE_FVAL: f64 = 349.856072;
/// `reference/sw07/ref.json` points[0].
const REF_JSON_FVAL: f64 = 349.85607197568856;

#[test]
#[ignore]
fn sw07_minimize() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let env = |k: &str| std::env::var(k).ok();
    let x0: Vec<f64> = match env("START").as_deref() {
        Some("mode") => {
            let r: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("ref.json")).unwrap()).unwrap();
            r["xparam_mode"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
        }
        _ => DsgeModel::new(&mf, None).unwrap().init_vector().unwrap(),
    };
    let m = DsgeModel::new(&mf, Some(&x0)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let data = LikelihoodData::new(
        &table.sample(&mf.varobs, 5, None).unwrap(),
        KalmanOptions { presample: 8, lik_init: 3, ..Default::default() },
    );
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &m.bounds };
    let f0 = post.eval(&x0).0;
    // OPTIM="Key,value,Key,value" (numbers parsed, else strings).
    let optim = env("OPTIM").map(|s| {
        OptValue::List(
            s.split(',')
                .map(|t| t.trim().parse::<f64>().map(OptValue::Num).unwrap_or(OptValue::Str(t.trim().to_string())))
                .collect(),
        )
    });
    let prior_sd: Vec<f64> = m.priors.iter().map(|p| p.sd).collect();
    for mc in env("MC").unwrap_or("8".into()).split(',').map(|s| s.parse::<i32>().unwrap()) {
        let ctx = MinimizeContext {
            bounds: &m.bounds,
            names: &m.estimated_names,
            prior_sd: &prior_sd,
            optim: optim.as_ref(),
            verbose: env("VERBOSE").is_some(),
            seed: env("SEED").map_or(0, |s| s.parse().unwrap()),
            stream: 1,
        };
        let t = std::time::Instant::now();
        let r = minimize(mc, &post, &x0, &ctx).unwrap();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "SW07 mode_compute={mc}: fval {:.6} (start {f0:.4}, csminwel mode {SW07_MODE_FVAL:.6}, diff {:+.2e}; ref.json {REF_JSON_FVAL:.3}), \
             {} iterations, {} evaluations, {:.1}s, exitflag {}: {}",
            r.fval,
            r.fval - SW07_MODE_FVAL,
            r.iterations,
            r.func_count,
            secs,
            r.exitflag,
            r.message
        );
        assert!(r.fval <= f0);
        if [1, 3, 4, 5, 101].contains(&mc) && env("START").is_none() && optim.is_none() {
            // the gradient-based ones reach the mode from the init values (see the module doc)
            assert!((r.fval - SW07_MODE_FVAL).abs() < 1e-3, "mode_compute={mc}: {}", r.fval);
        }
    }
}
