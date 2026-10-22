//! : every posterior sampler (independent MH, slice, TaRB, Student random walk) in
//! the nonlinear (particle filter) estimation, with Dynare's mh files and `load_mh_file`, on
//! reference/nonlinear/rbcnl.mod with the deterministic Gaussian filter (`gf`, cubature) and
//! on the bootstrap particle filter (`sis`).

use dynare::driver::{RunOptions, run};
use dynare::estimation::mh_files::{self, MhPaths};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/nonlinear")
}

const COMMON: &str = "datafile='rbcnl_data.csv', order=2, filter_algorithm=gf, proposal_approximation=cubature, \
                      distribution_approximation=cubature, nograph, nodisplay, nodiagnostic";

/// rbcnl.mod with its estimation commands replaced by `est`.
fn model(work: &Path, est: &str) {
    let src = std::fs::read_to_string(refdir().join("rbcnl.mod")).unwrap();
    let cut = src.find("@#if defined(ESTGF)").unwrap();
    std::fs::write(work.join("rbcnl.mod"), format!("{}{est}\n@#endif\n", &src[..cut])).unwrap();
}

fn results(work: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(work.join("rbcnl/Output/rbcnl_results.json")).unwrap()).unwrap()
}

fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn samplers_with_particle_filters() {
    let work = std::env::temp_dir().join(format!("dynare_rs_nl_samplers_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    std::fs::copy(refdir().join("rbcnl_data.csv"), work.join("rbcnl_data.csv")).unwrap();
    let ro = RunOptions { quiet: true, seed: 3, ..Default::default() };
    // the gf mode
    model(&work, &format!("estimation({COMMON}, mode_compute=4, mh_replic=0);"));
    run(&work.join("rbcnl.mod"), &ro).unwrap();
    let mode = vecf(&results(&work)["mode"]);
    let sd = vecf(&results(&work)["mode_sd"]);
    let mf = "mode_compute=0, mode_file='rbcnl/Output/rbcnl_mode.mat'";
    let cases = [
        (
            "imh",
            "posterior_sampling_method='independent_metropolis_hastings', mh_replic=1500, mh_nblocks=2, mh_jscale=0.8".to_string(),
        ),
        (
            "student",
            "posterior_sampler_options=('proposal_distribution','rand_multivariate_student','student_degrees_of_freedom',5), mh_replic=1500, mh_nblocks=2, mh_jscale=0.8".to_string(),
        ),
        (
            "slice",
            "posterior_sampling_method='slice', posterior_sampler_options=('slice_initialize_with_mode',1), prior_trunc=1e-10, mh_replic=150, mh_nblocks=2".to_string(),
        ),
        (
            "tarb",
            "posterior_sampling_method='tailored_random_block_metropolis_hastings', mh_replic=40, mh_nblocks=2".to_string(),
        ),
    ];
    for (name, opts) in &cases {
        model(&work, &format!("estimation({COMMON}, {mf}, {opts});"));
        run(&work.join("rbcnl.mod"), &ro).unwrap();
        let r = results(&work);
        let m = vecf(&r["mcmc"]["posterior_mean"]);
        // posterior means within a few standard deviations of the mode (short chains)
        for i in 0..m.len() {
            assert!(
                (m[i] - mode[i]).abs() < 4.0 * sd[i],
                "{name}: parameter {i}: mean {} mode {} sd {}",
                m[i],
                mode[i],
                sd[i]
            );
        }
        let p = MhPaths::new(&work.join("rbcnl"), "rbcnl");
        let (rec, _) = mh_files::load_last_record(&p).unwrap();
        assert_eq!(rec.mcmc_concluded_successfully, 1.0, "{name}");
        assert_eq!(rec.nblck, 2);
        println!("{name}: mean {m:?}, sampler {}", rec.sampler);
    }
    // load_mh_file with the bootstrap particle filter: 300 + 200 draws = one 500-draw run
    let sis = "datafile='rbcnl_data.csv', order=2, filter_algorithm=sis, number_of_particles=300, nograph, nodisplay, nodiagnostic";
    let draws = |dir: &Path| {
        let p = MhPaths::new(&dir.join("rbcnl"), "rbcnl");
        let (rec, _) = mh_files::load_last_record(&p).unwrap();
        mh_files::read_chains(&p, &rec, 0, &[]).unwrap()
    };
    model(&work, &format!("estimation({sis}, {mf}, mh_replic=500, mh_nblocks=2, mh_jscale=0.5);"));
    run(&work.join("rbcnl.mod"), &ro).unwrap();
    let want = draws(&work);
    model(&work, &format!("estimation({sis}, {mf}, mh_replic=300, mh_nblocks=2, mh_jscale=0.5);"));
    run(&work.join("rbcnl.mod"), &ro).unwrap();
    model(&work, &format!("estimation({sis}, {mf}, load_mh_file, mh_replic=200, mh_nblocks=2, mh_jscale=0.5);"));
    run(&work.join("rbcnl.mod"), &ro).unwrap();
    let got = draws(&work);
    for (a, b) in got.iter().zip(&want) {
        assert_eq!(a.total_draws, 500);
        assert_eq!(a.draws, b.draws);
        assert_eq!(a.logpost, b.logpost);
    }
    let _ = std::fs::remove_dir_all(&work);
}

/// hssmc, dime_mcmc and online with the Gaussian filter: posterior means near the gf mode,
/// the hssmc log marginal density near the Laplace approximation, Dynare's files written.
#[test]
fn smc_samplers_with_particle_filters() {
    let work = std::env::temp_dir().join(format!("dynare_rs_nl_smc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    std::fs::copy(refdir().join("rbcnl_data.csv"), work.join("rbcnl_data.csv")).unwrap();
    let ro = RunOptions { quiet: true, seed: 5, ..Default::default() };
    model(&work, &format!("estimation({COMMON}, mode_compute=4, mh_replic=0);"));
    run(&work.join("rbcnl.mod"), &ro).unwrap();
    let r0 = results(&work);
    let (mode, sd, lap) = (vecf(&r0["mode"]), vecf(&r0["mode_sd"]), r0["log_data_density_laplace"].as_f64().unwrap());
    let cases = [
        ("hssmc", "posterior_sampling_method='hssmc', posterior_sampler_options=('particles', 2000, 'steps', 25)"),
        (
            "dime",
            "posterior_sampling_method='dime_mcmc', posterior_sampler_options=('nchain', 30, 'niter', 600, 'tune', 300)",
        ),
        ("online", "posterior_sampling_method='online', posterior_sampler_options=('particles', 3000)"),
    ];
    for (name, opts) in cases {
        model(&work, &format!("estimation({COMMON}, {opts});"));
        run(&work.join("rbcnl.mod"), &ro).unwrap();
        let r = results(&work);
        let post = r[name]["posterior"].as_array().unwrap();
        let means: Vec<f64> = post.iter().map(|p| p["mean"].as_f64().unwrap()).collect();
        println!("{name}: means {means:?} mode {mode:?}");
        if name == "online" {
            // The Liu-West filter does not reproduce the batch posterior (it inflates the
            // measurement errors); Dynare 7.1's own run (2000 particles, final period) gives
            // these 95% bands, and the posterior means must fall inside them.
            let bands = [
                (0.003455, 0.027204),
                (0.012636, 0.023240),
                (0.001447, 0.007655),
                (0.879329, 0.956038),
                (1.925709, 2.624336),
            ];
            for (i, (lo, hi)) in bands.iter().enumerate() {
                assert!(means[i] > *lo && means[i] < *hi, "online {i}: {} outside Dynare's [{lo}, {hi}]", means[i]);
            }
            continue;
        }
        let (tol, rel) = (3.0, 0.05);
        for i in 0..means.len() {
            assert!(
                (means[i] - mode[i]).abs() < tol * sd[i] + rel * mode[i].abs(),
                "{name} {i}: {} vs mode {} sd {}",
                means[i],
                mode[i],
                sd[i]
            );
        }
        if name == "hssmc" {
            let m = r["MarginalDensity"]["hssmc"].as_f64().unwrap();
            println!("hssmc log density {m} vs Laplace {lap}");
            assert!((m - lap).abs() < 2.0, "hssmc log density {m} vs Laplace {lap}");
            assert!(work.join("rbcnl/hssmc/particles-25-25.mat").exists());
        }
    }
    assert!(work.join("rbcnl/dime/chains.mat").exists());
    assert!(work.join("rbcnl/online/parameters_particles_final.mat").exists());
    let _ = std::fs::remove_dir_all(&work);
}
