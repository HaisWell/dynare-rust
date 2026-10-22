//! Bayesian IRFs and posterior moments (`bayesian_irf`, `moments_varendo`).
//!
//! MATLAB draws its own MCMC chains and sub-sample, so these are checked by construction:
//! with every draw at one point the posterior statistics are that point's stoch_simul
//! numbers, with two points the mean is the sub-sample average; and an estimation on
//! simulated data runs end to end through the driver.

use dynare::analysis::{self, StochSimulOptions, irf, moments, posterior, simulate, vardec};
use dynare::dsge::DsgeModel;
use dynare::estimation::mcmc::Chain;
use dynare::modfile::parse_mod_str;

const MODEL: &str = r#"
var y pi i a;
varexo e_a e_d e_i;
parameters beta sigma kappa phi_pi phi_y rho_a rho_i;
beta = 0.99; sigma = 1.5; kappa = 0.12; phi_pi = 1.6; phi_y = 0.4; rho_a = 0.8; rho_i = 0.6;
model(linear);
y = y(+1) - (1/sigma)*(i - pi(+1)) + e_d;
pi = beta*pi(+1) + kappa*y + a;
i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + e_i;
a = rho_a*a(-1) + e_a;
end;
shocks;
var e_a; stderr 0.5;
var e_d; stderr 0.8;
var e_i; stderr 0.25;
end;
varobs y pi i;
estimated_params;
stderr e_a, inv_gamma_pdf, 0.5, 2;
stderr e_d, inv_gamma_pdf, 0.8, 2;
stderr e_i, inv_gamma_pdf, 0.25, 2;
rho_a, beta_pdf, 0.8, 0.1;
phi_pi, normal_pdf, 1.6, 0.2;
end;
"#;

fn chain(points: &[Vec<f64>]) -> Chain {
    let npar = points[0].len();
    Chain {
        npar,
        draws: points.iter().flatten().copied().collect(),
        logpost: vec![0.0; points.len()],
        first_stored: 0,
        total_draws: points.len(),
        accepted: points.len(),
        init: points[0].clone(),
        init_logpost: 0.0,
    }
}

fn x0() -> Vec<f64> {
    vec![0.5, 0.8, 0.25, 0.8, 1.6]
}

#[test]
fn degenerate_posterior_reproduces_stoch_simul() {
    let mf = parse_mod_str(MODEL, &[]).unwrap();
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let x = x0();
    let chains = vec![chain(&vec![x.clone(); 50]), chain(&vec![x.clone(); 50])];
    assert_eq!(posterior::number_of_subdraws(&chains, None).unwrap(), 100);
    let draws = posterior::posterior_subsample(&chains, 40, 7).unwrap();
    let ivar = vec![0, 1, 2];
    let o = StochSimulOptions { irf: 10, ar: 2, conditional_variance_decomposition: vec![1, 4], ..Default::default() };
    let pirf = posterior::posterior_irf(&dsge, &draws, &ivar, &o).unwrap();
    let r = dsge.resolve(&x).unwrap();
    let cs = vardec::lower_cholesky(&r.q, 1e-14);
    let pt = irf::stoch_simul_irfs(&r.dr, &cs, &ivar, &[0, 1, 2], 10, false);
    for (s, si) in pt.iter().enumerate() {
        for j in 0..3 {
            for t in 0..10 {
                let m = &pirf.stats[s][j][t];
                assert!((m.mean - si.y[(j, t)]).abs() < 1e-14 && (m.median - si.y[(j, t)]).abs() < 1e-14);
                assert!((m.hpd.0 - si.y[(j, t)]).abs() < 1e-14 && (m.hpd.1 - si.y[(j, t)]).abs() < 1e-14);
            }
        }
    }
    let pm = posterior::posterior_moments_varendo(&dsge, &draws, &ivar, &o).unwrap();
    let th = moments::th_autocovariances(&r.dr, &r.q, &ivar, &o.moment_options()).unwrap();
    for i in 0..3 {
        for j in i..3 {
            assert!((pm.covariance[i][j].as_ref().unwrap().mean - th.gamma0[(i, j)]).abs() < 1e-13);
        }
        for k in 0..2 {
            assert!((pm.correlation[k][i][i].mean - th.autocorr[k][(i, i)]).abs() < 1e-13);
        }
        for s in 0..3 {
            assert!(
                (pm.variance_decomposition.as_ref().unwrap()[i][s].mean - th.vardec.as_ref().unwrap()[(i, s)]).abs()
                    < 1e-13
            );
        }
    }
    let cvd = vardec::conditional_variance_decomposition(&r.dr, &r.q, &[1, 4], &ivar).unwrap();
    for h in 0..2 {
        for i in 0..3 {
            for s in 0..3 {
                assert!((pm.conditional.as_ref().unwrap()[h][i][s].mean - cvd[h][(i, s)]).abs() < 1e-13);
            }
        }
    }
}

#[test]
fn posterior_irf_mean_is_the_subsample_average() {
    let mf = parse_mod_str(MODEL, &[]).unwrap();
    let dsge = DsgeModel::new(&mf, None).unwrap();
    let xa = x0();
    let mut xb = x0();
    xb[3] = 0.5;
    xb[0] = 0.7;
    let chains = vec![chain(&[xa.clone(), xb.clone(), xa.clone()]), chain(&[xb.clone(), xb.clone(), xa.clone()])];
    let draws = posterior::posterior_subsample(&chains, 30, 3).unwrap();
    let na = draws.iter().filter(|d| **d == xa).count() as f64;
    assert!(na > 0.0 && na < 30.0);
    let o = StochSimulOptions { irf: 6, irf_shocks: Some(vec!["e_a".into()]), ..Default::default() };
    let pirf = posterior::posterior_irf(&dsge, &draws, &[1], &o).unwrap();
    let ir = |x: &[f64]| {
        let r = dsge.resolve(x).unwrap();
        let cs = vardec::lower_cholesky(&r.q, 1e-14);
        irf::stoch_simul_irfs(&r.dr, &cs, &[1], &[0], 6, false).remove(0).y
    };
    let (ya, yb) = (ir(&xa), ir(&xb));
    for t in 0..6 {
        let want = (na * ya[(0, t)] + (30.0 - na) * yb[(0, t)]) / 30.0;
        assert!((pirf.stats[0][0][t].mean - want).abs() < 1e-12);
    }
}

/// Simulated data -> estimation with an MCMC, `bayesian_irf` and `moments_varendo` -> the
/// output files, then a `stoch_simul` at the posterior mean.
#[test]
fn estimation_with_bayesian_irf_end_to_end() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("analysis_bayesian");
    std::fs::create_dir_all(&dir).unwrap();
    let mf = parse_mod_str(MODEL, &[]).unwrap();
    let sol = analysis::solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
    let ex = simulate::draw_shocks(&sol.sigma_e, 120, 11).unwrap();
    let y = simulate::simult(&sol.dr, &sol.dr.ys, &ex);
    let mut csv = String::from("y,pi,i\n");
    for t in 1..=120 {
        csv += &format!("{},{},{}\n", y[(0, t)], y[(1, t)], y[(2, t)]);
    }
    std::fs::write(dir.join("nk_obs.csv"), csv).unwrap();
    let src = format!(
        "{MODEL}\nestimation(datafile='nk_obs.csv', mode_compute=4, mh_replic=600, mh_nblocks=2, mh_drop=0.5, \
         bayesian_irf, irf=8, moments_varendo, conditional_variance_decomposition=[1 4], sub_draws=200, nograph) y pi;\n\
         stoch_simul(irf=8, ar=2, nograph, noprint) y pi i;\n"
    );
    let path = dir.join("nk_est.mod");
    std::fs::write(&path, src).unwrap();
    dynare::driver::run(&path, &dynare::driver::RunOptions { seed: 5, quiet: true, ..Default::default() }).unwrap();
    let out = dir.join("nk_est").join("Output");
    let birf = std::fs::read_to_string(out.join("nk_est_Bayesian_IRF_e_a.csv")).unwrap();
    assert_eq!(birf.lines().count(), 1 + 2 * 8); // header + 2 variables x 8 periods
    let pm = std::fs::read_to_string(out.join("nk_est_posterior_moments.csv")).unwrap();
    assert!(pm.lines().any(|l| l.starts_with("variance_decomposition,y,e_a,")));
    assert!(pm.lines().any(|l| l.starts_with("conditional_variance_decomposition_4,pi,e_d,")));
    // HPD intervals bracket the posterior mean
    for l in birf.lines().skip(1) {
        let f: Vec<f64> = l.split(',').skip(2).map(|x| x.parse().unwrap()).collect();
        assert!(f[3] <= f[0] + 1e-12 && f[0] <= f[4] + 1e-12, "{l}");
    }
    assert!(out.join("nk_est_IRF_e_a.csv").exists());
    assert!(out.join("nk_est_stoch_simul.json").exists());
}
