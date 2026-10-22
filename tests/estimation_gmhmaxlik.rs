//! `mode_compute = 6` (gmhmaxlik.m / gmhmaxlik_core.m): a literal sequential
//! transcription of the MATLAB code is reproduced bit for bit whatever the
//! pre-fetching width, and on a Gaussian target the returned covariance, Hessian,
//! mode and scale are what the algorithm aims at.

use dynare::estimation::{Bounds, GmhmaxlikOptions, Objective, VarInit, gmhmaxlik};
use dynare::linalg::{Mat, chol_upper, sym_eig};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// f(x) = 0.5 (x-m)' P (x-m) + c, the minus log kernel of N(m, inv(P)).
struct Gauss {
    m: Vec<f64>,
    p: Mat,
    c: f64,
}

impl Objective for Gauss {
    fn dim(&self) -> usize {
        self.m.len()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let n = self.m.len();
        let d: Vec<f64> = (0..n).map(|i| x[i] - self.m[i]).collect();
        let mut s = 0.0;
        for j in 0..n {
            for i in 0..n {
                s += d[i] * self.p[(i, j)] * d[j];
            }
        }
        (0.5 * s + self.c, 0)
    }
}

/// Gaussian kernel in a box with an infeasible region (+inf, like a failed solution).
struct Holed;
impl Objective for Holed {
    fn dim(&self) -> usize {
        4
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        if x[0] + x[1] > 1.2 {
            return (f64::INFINITY, 1);
        }
        (0.5 * x.iter().enumerate().map(|(i, v)| (v - 0.2) * (v - 0.2) * (1.0 + i as f64)).sum::<f64>(), 0)
    }
}

fn holed_bounds() -> Bounds {
    Bounds { lb: vec![-1.0, -2.0, -3.0, -1.5], ub: vec![2.0, 1.0, 3.0, 1.5] }
}

fn ref_stream(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed);
    r.set_stream(stream);
    r
}

/// `iScale*dd*randn(npar,1) + mean` with `dd = chol(C)'`, i.e. mean + randn(1,n)*(iScale*U).
fn ref_proposal(rng: &mut ChaCha8Rng, mean: &[f64], u: &Mat) -> Vec<f64> {
    let n = mean.len();
    let z: Vec<f64> = (0..n).map(|_| rng.sample(rand_distr::StandardNormal)).collect();
    (0..n).map(|j| mean[j] + (0..=j).map(|i| z[i] * u[(i, j)]).sum::<f64>()).collect()
}

struct RefOut {
    mode: Vec<f64>,
    var: Mat,
    scale: f64,
    mean: Vec<f64>,
}

/// Literal transcription of gmhmaxlik_core.m (sequential, one evaluation at a time).
#[allow(clippy::too_many_arguments)]
fn ref_core(
    f: &dyn Objective,
    rng: &mut ChaCha8Rng,
    xparam1: &[f64],
    b: &Bounds,
    o: &GmhmaxlikOptions,
    mut iscale: f64,
    last_call: bool,
    mean_par: &[f64],
    var_cov: &Mat,
) -> RefOut {
    let n = xparam1.len();
    let inside = |p: &[f64]| (0..n).all(|k| p[k] > b.lb[k] && p[k] < b.ub[k]);
    let lpf = |p: &[f64]| if inside(p) { -f.eval(p).0 } else { f64::NEG_INFINITY };
    let mut cov_jump = var_cov.clone();
    let mut mode_par = xparam1.to_vec();
    // [1]
    let (mut j, mut jj, mut jsux, mut test) = (1usize, 1usize, 0usize, 0usize);
    let mut ix2 = mode_par.clone();
    let mut ilogpo2 = -f.eval(&ix2).0;
    let mut mlogpo2 = ilogpo2;
    let dd = chol_upper(&cov_jump).unwrap_or_else(|| Mat::identity(n, n));
    while j <= o.nscale {
        let proposal = ref_proposal(rng, &ix2, &(&dd * iscale));
        let logpo2 = lpf(&proposal);
        if logpo2 > f64::NEG_INFINITY && rng.r#gen::<f64>().ln() < logpo2 - ilogpo2 {
            ix2 = proposal;
            if logpo2 > mlogpo2 {
                mode_par = ix2.clone();
                mlogpo2 = logpo2;
            }
            ilogpo2 = logpo2;
            jsux += 1;
        }
        if j % 500 == 0 {
            let cfactor = (jsux as f64 / jj as f64) / o.target;
            if cfactor > 0.0 {
                iscale *= cfactor;
            } else {
                iscale /= 10.0;
            }
            jsux = 0;
            jj = 0;
            if cfactor > 0.90 && cfactor < 1.10 {
                test += 1;
            }
            if test > 4 {
                break;
            }
        }
        j += 1;
        jj += 1;
    }
    // [2]
    let mut j = 1usize;
    ilogpo2 = -f.eval(&ix2).0;
    let mut mean = mean_par.to_vec();
    while j <= o.number {
        j += 1;
        let proposal = ref_proposal(rng, &ix2, &(&dd * iscale));
        let logpo2 = lpf(&proposal);
        if logpo2 > f64::NEG_INFINITY && rng.r#gen::<f64>().ln() < logpo2 - ilogpo2 {
            ix2 = proposal;
            if logpo2 > mlogpo2 {
                mode_par = ix2.clone();
                mlogpo2 = logpo2;
            }
            ilogpo2 = logpo2;
        }
        let old = mean.clone();
        let wt = 1.0 / j as f64;
        for k in 0..n {
            mean[k] = old[k] + wt * (ix2[k] - old[k]);
        }
        for c in 0..n {
            for r in 0..n {
                let om = old[r] * old[c];
                let cj = cov_jump[(r, c)];
                cov_jump[(r, c)] = cj + om - mean[r] * mean[c] + wt * (ix2[r] * ix2[c] - cj - om);
            }
        }
    }
    let var = cov_jump.clone();
    let scale;
    if last_call {
        // [3]
        let (mut j, mut jj, mut jsux, mut test) = (1usize, 1usize, 0usize, 0usize);
        ilogpo2 = -f.eval(&ix2).0;
        let dd = chol_upper(&cov_jump).unwrap();
        while j <= o.nscale {
            let proposal = ref_proposal(rng, &ix2, &(&dd * iscale));
            let logpo2 = lpf(&proposal);
            if logpo2 > f64::NEG_INFINITY && rng.r#gen::<f64>().ln() < logpo2 - ilogpo2 {
                ix2 = proposal;
                if logpo2 > mlogpo2 {
                    mode_par = ix2.clone();
                    mlogpo2 = logpo2;
                }
                ilogpo2 = logpo2;
                jsux += 1;
            }
            if j % 1000 == 0 {
                let cfactor = (jsux as f64 / jj as f64) / o.target;
                iscale *= cfactor;
                jsux = 0;
                jj = 0;
                if cfactor > 0.90 && cfactor < 1.10 {
                    test += 1;
                }
                if test > 4 {
                    break;
                }
            }
            j += 1;
            jj += 1;
        }
        scale = iscale;
        // [4]
        if o.nclimb > 0 {
            let (mut j, mut jsux, mut test) = (1usize, 0usize, 0usize);
            let maxdiag = (0..n).map(|i| cov_jump[(i, i)]).fold(f64::NEG_INFINITY, f64::max);
            while j <= o.nclimb {
                let proposal = ref_proposal(rng, &mode_par, &(&dd * iscale));
                let logpo2 = lpf(&proposal);
                if logpo2 > mlogpo2 {
                    mode_par = proposal;
                    mlogpo2 = logpo2;
                    jsux += 1;
                }
                if j % 200 == 0 {
                    if jsux <= 1 {
                        test += 1;
                    } else {
                        test = 0;
                    }
                    jsux = 0;
                    if test > 4 {
                        iscale /= 1.10;
                    }
                    if (iscale * maxdiag).sqrt() < 1e-4 {
                        break;
                    }
                }
                j += 1;
            }
        }
    } else {
        scale = iscale;
    }
    RefOut { mode: mode_par, var, scale, mean }
}

/// Literal transcription of gmhmaxlik.m (varinit 'prior').
fn ref_gmhmaxlik(
    f: &dyn Objective,
    xinit: &[f64],
    iscale: f64,
    b: &Bounds,
    priorstd: &[f64],
    o: &GmhmaxlikOptions,
) -> (Vec<f64>, Mat, f64, f64) {
    let n = xinit.len();
    let mut rng = ref_stream(o.seed, o.stream);
    let mut var = Mat::from_fn(n, n, |i, j| if i == j { priorstd[i] * priorstd[i] } else { 0.0 });
    let mut scale = iscale;
    let mut out = None;
    for i in 1..=o.iterations {
        let r = ref_core(f, &mut rng, xinit, b, o, scale, i == o.iterations, xinit, &var);
        scale = r.scale;
        var = r.var.clone();
        let _ = &r.mean;
        out = Some(r);
    }
    let r = out.unwrap();
    let fval = f.eval(&r.mode).0;
    (r.mode, r.var, r.scale, fval)
}

fn bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn prefetching_reproduces_the_literal_transcription() {
    let (obj, b) = (Holed, holed_bounds());
    let x0 = vec![0.5, -0.3, 0.1, 0.4];
    let priorstd = vec![0.5, 0.4, 0.3, 0.2];
    let mut o = GmhmaxlikOptions {
        iterations: 2,
        number: 3000,
        nscale: 20_000,
        nclimb: 6000,
        silent: true,
        seed: 11,
        ..Default::default()
    };
    let (rx, rvar, rscale, rfval) = ref_gmhmaxlik(&obj, &x0, 0.7, &b, &priorstd, &o);
    let mut counts = vec![];
    for pf in [1, 2, 3, 7] {
        o.prefetch = pf;
        let r = gmhmaxlik(&obj, &x0, None, 0.7, &b, &priorstd, &o).unwrap();
        assert!(bits_eq(&r.x, &rx), "prefetch {pf}: mode {:?} vs {:?}", r.x, rx);
        assert!(bits_eq(r.post_variance.as_slice(), rvar.as_slice()), "prefetch {pf}: covariance");
        assert_eq!(r.scale.to_bits(), rscale.to_bits(), "prefetch {pf}: scale");
        assert_eq!(r.fval.to_bits(), rfval.to_bits(), "prefetch {pf}: fval");
        counts.push(r.func_count);
    }
    assert!(counts.iter().all(|&c| c == counts[0]), "{counts:?}");
}

#[test]
fn initial_covariance_options() {
    let (obj, b) = (Holed, holed_bounds());
    let x0 = vec![0.0; 4];
    let o =
        GmhmaxlikOptions { iterations: 1, number: 500, nscale: 1000, nclimb: 0, silent: true, ..Default::default() };
    // 'previous' without a Hessian is an error, as in gmhmaxlik.m
    let bad = GmhmaxlikOptions { varinit: Some(VarInit::Previous), ..o.clone() };
    assert!(gmhmaxlik(&obj, &x0, None, 0.5, &b, &[1.0; 4], &bad).is_err());
    // Hinit given: CovJump = inv(Hinit); same as the prior with priorstd = sqrt(diag(inv(H)))
    let h = Mat::from_diagonal(&nalgebra::DVector::from_vec(vec![4.0, 1.0, 0.25, 16.0]));
    let a = gmhmaxlik(&obj, &x0, Some(&h), 0.5, &b, &[9.0; 4], &o).unwrap();
    let c = gmhmaxlik(&obj, &x0, None, 0.5, &b, &[0.5, 1.0, 2.0, 0.25], &o).unwrap();
    assert!(bits_eq(&a.x, &c.x) && a.scale == c.scale);
    // infinite prior sd -> variance 10
    let d = gmhmaxlik(&obj, &x0, None, 0.5, &b, &[f64::INFINITY; 4], &o).unwrap();
    let h10 = Mat::identity(4, 4) * 0.1; // inv = 10*eye exactly
    let e = gmhmaxlik(&obj, &x0, Some(&h10), 0.5, &b, &[1.0; 4], &o).unwrap();
    assert!(bits_eq(&d.x, &e.x));
}

/// A correlated 6-dimensional Gaussian with heterogeneous scales.
fn gauss6() -> (Gauss, Mat) {
    let n = 6;
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let a = Mat::from_fn(n, n, |_, _| rng.r#gen::<f64>() - 0.5);
    let mut sigma = &a * a.transpose() / n as f64 + Mat::identity(n, n) * 0.05;
    let scales = [0.05, 0.2, 1.0, 3.0, 0.5, 0.01];
    for j in 0..n {
        for i in 0..n {
            sigma[(i, j)] *= scales[i] * scales[j];
        }
    }
    let p = sigma.clone().try_inverse().unwrap();
    let m: Vec<f64> = (0..n).map(|i| scales[i] * (rng.r#gen::<f64>() - 0.5)).collect();
    (Gauss { m, p, c: 42.0 }, sigma)
}

#[test]
fn gaussian_target_covariance_hessian_and_scale() {
    let (obj, sigma) = gauss6();
    let n = 6;
    let b = Bounds::unbounded(n);
    // start away from the mode, prior sd 3x too large
    let x0: Vec<f64> = (0..n).map(|i| obj.m[i] + 0.5 * sigma[(i, i)].sqrt()).collect();
    let priorstd: Vec<f64> = (0..n).map(|i| 3.0 * sigma[(i, i)].sqrt()).collect();
    for seed in 1..=8 {
        let o = GmhmaxlikOptions { silent: true, seed, ..Default::default() };
        let t = std::time::Instant::now();
        let r = gmhmaxlik(&obj, &x0, None, 2.38 / (n as f64).sqrt(), &b, &priorstd, &o).unwrap();
        println!(
            "gmhmaxlik 6-dim Gaussian: {:.2}s, {} evals, scale {:.4} (2.38/sqrt(n) = {:.4}), fval - min = {:.3e}",
            t.elapsed().as_secs_f64(),
            r.func_count,
            r.scale,
            2.38 / (n as f64).sqrt(),
            r.fval - obj.c
        );
        // covariance: correlation-normalised error
        let mut worst: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let e = (r.post_variance[(i, j)] - sigma[(i, j)]) / (sigma[(i, i)] * sigma[(j, j)]).sqrt();
                worst = worst.max(e.abs());
            }
        }
        println!("max |PostVariance - Sigma| / (sd_i sd_j) = {worst:.3}");
        assert!(worst < 0.2, "{worst}");
        // hh = inv(PostVariance), positive definite, close to the precision
        let (ev, _) = sym_eig(&((&r.hessian + r.hessian.transpose()) * 0.5));
        assert!(ev[0] > 0.0, "{ev}");
        // eigenvalues of hh*Sigma = those of U*hh*U' (Sigma = U'U): 1 for a perfect estimate
        let u = chol_upper(&sigma).unwrap();
        let m = &u * &r.hessian * u.transpose();
        let (pe, _) = sym_eig(&((&m + m.transpose()) * 0.5));
        assert!(pe[0] > 0.75 && pe[n - 1] < 1.3, "eig(hh*Sigma) {pe}");
        // the climb ends near the mode: minus log kernel within ~0.01 of the minimum
        assert!(r.fval - obj.c < 0.05, "{}", r.fval - obj.c);
        // acceptance 1/3 with the posterior covariance: scale a bit below 2.38/sqrt(n)
        let s0 = 2.38 / (n as f64).sqrt();
        assert!(r.scale > 0.5 * s0 && r.scale < 1.3 * s0, "scale {}", r.scale);
        println!("  eig(hh*Sigma) in [{:.3}, {:.3}]", pe[0], pe[n - 1]);
    }
}

/// SW07, `mode_compute = 6` from the `estimated_params` init column (as the cleaned
/// single-call model: first_obs=5, presample=8, diffuse_filter):
/// `cargo test --release --test estimation_gmhmaxlik sw07 -- --ignored --nocapture`.
/// `GMH_AFTER_CSMINWEL=1` runs mode_compute=4 first and starts gmhmaxlik from its
/// mode with the hessian.m Hessian (`additional_optimizer_steps=[6]`).
/// `GMH_SEED`, `GMH_PREFETCH` set the stream and the pre-fetching width.
#[test]
#[ignore]
fn sw07_mode_compute_6() {
    use dynare::data::Table;
    use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
    use dynare::estimation::{CsminwelOptions, csminwel, hessian};
    use dynare::kalman::KalmanOptions;
    use dynare::modfile::parse_mod_file;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let x0 = DsgeModel::new(&mf, None).unwrap().init_vector().unwrap();
    let m = DsgeModel::new(&mf, Some(&x0)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let data = LikelihoodData::new(
        &table.sample(&mf.varobs, 5, None).unwrap(),
        KalmanOptions { presample: 8, lik_init: 3, ..Default::default() },
    );
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &m.bounds };
    let n = x0.len();
    let f0 = post.eval(&x0).0;
    println!("SW07: {n} parameters, minus log posterior at the init values {f0:.4}");
    let env = |k: &str| std::env::var(k).ok();
    let t = std::time::Instant::now();
    let (start, hinit, f_start) = if env("GMH_AFTER_CSMINWEL").is_some() {
        let r = csminwel(&post, &x0, &CsminwelOptions { verbose: false, ..Default::default() });
        let h = hessian(&post, &r.x, [1e-2, 1.0]);
        println!("csminwel: fval {:.6} ({:.1}s)", r.fval, t.elapsed().as_secs_f64());
        (r.x, Some(h), r.fval)
    } else {
        (x0.clone(), None, f0)
    };
    let prior_sd: Vec<f64> = m.priors.iter().map(|p| p.sd).collect();
    let o = GmhmaxlikOptions {
        silent: false,
        seed: env("GMH_SEED").map_or(0, |s| s.parse().unwrap()),
        prefetch: env("GMH_PREFETCH").map_or(0, |s| s.parse().unwrap()),
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let r = gmhmaxlik(&post, &start, hinit.as_ref(), 2.38 / (n as f64).sqrt(), &m.bounds, &prior_sd, &o).unwrap();
    let secs = t.elapsed().as_secs_f64();
    let (ev, _) = sym_eig(&((&r.hessian + r.hessian.transpose()) * 0.5));
    println!(
        "SW07 mode_compute=6: fval {:.6} (start {f_start:.6}), min eig(hh) {:.4e}, max eig(hh) {:.4e}, \
         Scale {:.6e}, {} evaluations ({} per round), {:.1}s",
        r.fval,
        ev[0],
        ev[n - 1],
        r.scale,
        r.func_count,
        r.prefetch,
        secs
    );
    assert!(r.fval <= f_start);
}
