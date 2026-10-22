//! Alternative posterior samplers (`src/estimation/samplers/`): independent MH,
//! slice (plain, rotated, prior-initialized), TaRB and the Student random walk,
//! checked against analytic moments and against the Gaussian RW sampler on a
//! correlated Gaussian and on a bounded skewed target; reproducibility; and
//! (ignored) the SW07 posterior.

use std::time::Instant;

use dynare::estimation::samplers::{SamplerOptions, SamplingMethod, posterior_sampler_full, prior_mode_start};
use dynare::estimation::{Bounds, Chain, MhOptions, Objective, Prior, PriorShape, log_prior_density, prior_bounds};
use dynare::linalg::Mat;
use dynare::modfile::OptValue;

struct Gauss {
    m: Vec<f64>,
    p: Mat,
}

impl Objective for Gauss {
    fn dim(&self) -> usize {
        self.m.len()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let n = self.m.len();
        let d: Vec<f64> = (0..n).map(|i| x[i] - self.m[i]).collect();
        let mut s = 0.0;
        for i in 0..n {
            for j in 0..n {
                s += d[i] * self.p[(i, j)] * d[j];
            }
        }
        (0.5 * s, 0)
    }
}

fn gauss() -> (Gauss, Mat) {
    let sigma =
        Mat::from_row_slice(4, 4, &[1.0, 0.6, 0.0, 0.2, 0.6, 2.0, -0.5, 0.0, 0.0, -0.5, 0.5, 0.1, 0.2, 0.0, 0.1, 0.3]);
    let p = sigma.clone().try_inverse().unwrap();
    (Gauss { m: vec![1.0, -2.0, 0.5, 3.0], p }, sigma)
}

/// The priors as the target: minus the log prior density, +inf outside the support.
struct PriorTarget(Vec<Prior>);

impl Objective for PriorTarget {
    fn dim(&self) -> usize {
        self.0.len()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let v = -log_prior_density(x, &self.0);
        (v, if v.is_finite() { 0 } else { 1 })
    }
}

fn skewed() -> (PriorTarget, Vec<f64>, Vec<f64>) {
    let pr = vec![
        Prior::new(PriorShape::Gamma, 2.0, 2f64.sqrt(), None, None).unwrap(),
        Prior::new(PriorShape::Beta, 2.0 / 7.0, (10.0f64 / 392.0).sqrt(), None, None).unwrap(),
        Prior::new(PriorShape::InvGamma2, 0.5, 0.2, None, None).unwrap(),
    ];
    let mean = pr.iter().map(|p| p.mean).collect();
    let sd = pr.iter().map(|p| p.sd).collect();
    (PriorTarget(pr), mean, sd)
}

fn moments(chains: &[Chain]) -> (Vec<f64>, Vec<f64>) {
    let n = chains[0].npar;
    let mut s = vec![0.0; n];
    let mut s2 = vec![0.0; n];
    let mut k = 0.0;
    for c in chains {
        for r in c.draws.chunks(n) {
            for i in 0..n {
                s[i] += r[i];
                s2[i] += r[i] * r[i];
            }
            k += 1.0;
        }
    }
    let m: Vec<f64> = s.iter().map(|v| v / k).collect();
    let sd = (0..n).map(|i| (s2[i] / k - m[i] * m[i]).sqrt()).collect();
    (m, sd)
}

fn opts(method: &str, kv: &[(&str, f64)]) -> SamplerOptions {
    let mut list = vec![];
    for (k, v) in kv {
        list.push(OptValue::Str(format!("'{k}'")));
        list.push(OptValue::Num(*v));
    }
    let list = OptValue::List(list);
    let so = if kv.is_empty() { None } else { Some(&list) };
    SamplerOptions::from_options(Some(&OptValue::Str(method.into())), so).unwrap()
}

fn mh(n: usize, replic: usize, nblocks: usize, jscale: f64) -> MhOptions {
    let mut o = MhOptions::new(n);
    o.replic = replic;
    o.nblocks = nblocks;
    o.seed = 11;
    o.progress_every = 0;
    o.store_from = replic / 5;
    o.jscale = vec![jscale; n];
    o.mh_jscale = jscale;
    o
}

/// Runs a sampler and returns the largest |mean error| and |sd ratio - 1| (in sd units).
#[allow(clippy::too_many_arguments)]
fn check(
    name: &str,
    obj: &dyn Objective,
    x0: &[f64],
    cov: &Mat,
    bounds: &Bounds,
    m: &MhOptions,
    so: &SamplerOptions,
    mean: &[f64],
    sd: &[f64],
) -> (f64, f64) {
    let t = Instant::now();
    let r = posterior_sampler_full(obj, x0, cov, bounds, m, so, None).unwrap();
    let (mh, sh) = moments(&r.mh.chains);
    eprintln!("{name}: mean {mh:?} vs {mean:?}; sd {sh:?} vs {sd:?}");
    let em = (0..mean.len()).map(|i| (mh[i] - mean[i]).abs() / sd[i]).fold(0.0, f64::max);
    let es = (0..mean.len()).map(|i| (sh[i] / sd[i] - 1.0).abs()).fold(0.0, f64::max);
    eprintln!(
        "{name:>14}: {:.2?}, acceptance {:.3}, evals/draw {:.1}, max |mean err|/sd {em:.3}, max |sd ratio-1| {es:.3}",
        t.elapsed(),
        r.acceptance.iter().sum::<f64>() / r.acceptance.len() as f64,
        r.evals_per_draw.iter().sum::<f64>() / r.evals_per_draw.len() as f64
    );
    (em, es)
}

#[test]
fn samplers_recover_correlated_gaussian() {
    let (g, sigma) = gauss();
    let n = 4;
    let sd: Vec<f64> = (0..n).map(|i| sigma[(i, i)].sqrt()).collect();
    // finite bounds (slice needs them), +-12 sd: irrelevant for the moments
    let bounds = Bounds {
        lb: (0..n).map(|i| g.m[i] - 12.0 * sd[i]).collect(),
        ub: (0..n).map(|i| g.m[i] + 12.0 * sd[i]).collect(),
    };
    let cases: Vec<(&str, SamplerOptions, MhOptions, f64)> = vec![
        ("rw normal", opts("random_walk_metropolis_hastings", &[]), mh(n, 40_000, 4, 1.2), 0.08),
        ("rw student", opts("random_walk_metropolis_hastings", &[]).with_student(5.0), mh(n, 40_000, 4, 1.0), 0.08),
        ("imh normal", opts("independent_metropolis_hastings", &[]), mh(n, 20_000, 4, 1.2), 0.05),
        ("imh student", opts("independent_metropolis_hastings", &[]).with_student(4.0), mh(n, 20_000, 4, 1.0), 0.05),
        ("slice", opts("slice", &[("slice_initialize_with_mode", 1.0)]), mh(n, 4_000, 4, 1.0), 0.06),
        (
            "rotated slice",
            opts("slice", &[("slice_initialize_with_mode", 1.0), ("rotated", 1.0)]),
            mh(n, 4_000, 4, 1.0),
            0.06,
        ),
        ("tarb", opts("tailored_random_block_metropolis_hastings", &[]), mh(n, 1_500, 4, 1.0), 0.1),
    ];
    for (name, so, m, tol) in cases {
        let (em, es) = check(name, &g, &g.m, &sigma, &bounds, &m, &so, &g.m, &sd);
        assert!(em < tol && es < 2.0 * tol, "{name}: mean err {em}, sd err {es}");
    }
}

trait WithStudent {
    fn with_student(self, df: f64) -> Self;
}
impl WithStudent for SamplerOptions {
    fn with_student(mut self, df: f64) -> Self {
        self.proposal = dynare::estimation::samplers::Proposal::Student;
        self.student_df = df;
        self
    }
}

#[test]
fn samplers_recover_bounded_skewed_target() {
    let (t, mean, sd) = skewed();
    let n = 3;
    let bounds = prior_bounds(&t.0, 1e-10);
    // proposal covariance: the target's covariance (independent components)
    let cov = Mat::from_fn(n, n, |i, j| if i == j { sd[i] * sd[i] } else { 0.0 });
    let x0 = prior_mode_start(&t.0, &bounds);
    let cases: Vec<(&str, SamplerOptions, MhOptions, f64)> = vec![
        ("rw normal", opts("random_walk_metropolis_hastings", &[]), mh(n, 60_000, 4, 1.2), 0.06),
        ("imh student", opts("independent_metropolis_hastings", &[]).with_student(3.0), mh(n, 40_000, 4, 1.0), 0.06),
        ("slice (prior)", opts("slice", &[]).with_priors(&t.0), mh(n, 5_000, 4, 1.0), 0.06),
        ("slice 1 chain", opts("slice", &[]).with_priors(&t.0), mh(n, 15_000, 1, 1.0), 0.06),
        // A Gaussian (Laplace) proposal is too light-tailed for the inverse gamma: TaRB with
        // it gets stuck in the right tail (as an IMH would); the Student proposal mixes.
        (
            "tarb student",
            opts("tailored_random_block_metropolis_hastings", &[]).with_student(3.0).with_priors(&t.0),
            mh(n, 2_000, 4, 1.0),
            0.1,
        ),
        (
            "tarb newrat",
            opts("tailored_random_block_metropolis_hastings", &[("mode_compute", 5.0)])
                .with_student(3.0)
                .with_priors(&t.0),
            mh(n, 1_000, 4, 1.0),
            0.12,
        ),
    ];
    for (name, so, m, tol) in cases {
        let (em, es) = check(name, &t, &x0, &cov, &bounds, &m, &so, &mean, &sd);
        assert!(em < tol && es < 2.0 * tol, "{name}: mean err {em}, sd err {es}");
    }
}

#[test]
fn draws_do_not_depend_on_prefetch_width_and_are_reproducible() {
    let (g, sigma) = gauss();
    let n = 4;
    let bounds = Bounds { lb: vec![-20.0; n], ub: vec![20.0; n] };
    for (name, so) in [
        ("imh", opts("independent_metropolis_hastings", &[])),
        ("slice", opts("slice", &[("slice_initialize_with_mode", 1.0)])),
        ("rotated", opts("slice", &[("slice_initialize_with_mode", 1.0), ("rotated", 1.0)])),
    ] {
        let mut runs = vec![];
        for width in [1, 1, 3, 7] {
            let mut m = mh(n, 400, 2, 1.5);
            m.store_from = 0;
            m.prefetch = width;
            runs.push(posterior_sampler_full(&g, &g.m, &sigma, &bounds, &m, &so, None).unwrap());
        }
        for r in &runs[1..] {
            for (a, b) in r.mh.chains.iter().zip(&runs[0].mh.chains) {
                assert!(a.draws.iter().zip(&b.draws).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}");
                assert_eq!(a.logpost, b.logpost);
            }
            assert_eq!(r.evals_per_draw, runs[0].evals_per_draw);
        }
        let mut m = mh(n, 400, 2, 1.5);
        m.seed = 12;
        let other = posterior_sampler_full(&g, &g.m, &sigma, &bounds, &m, &so, None).unwrap();
        assert_ne!(other.mh.chains[0].draws, runs[0].mh.chains[0].draws, "{name}: the seed must matter");
    }
}

#[test]
fn option_errors() {
    let s = |x: &str| OptValue::Str(x.into());
    // rotated unimodal slice needs the Hessian (slice_initialize_with_mode)
    let so = opts("slice", &[("rotated", 1.0)]);
    let (g, sigma) = gauss();
    let b = Bounds { lb: vec![-20.0; 4], ub: vec![20.0; 4] };
    assert!(posterior_sampler_full(&g, &g.m, &sigma, &b, &mh(4, 10, 1, 1.0), &so, None).is_err());
    // slice needs finite bounds
    let so = opts("slice", &[("slice_initialize_with_mode", 1.0)]);
    assert!(posterior_sampler_full(&g, &g.m, &sigma, &Bounds::unbounded(4), &mh(4, 10, 1, 1.0), &so, None).is_err());
    assert!(SamplerOptions::from_options(Some(&s("dime_mcmc")), None).is_err());
    let so = SamplerOptions::from_options(
        Some(&s("independent_metropolis_hastings")),
        Some(&OptValue::List(vec![
            s("proposal_distribution"),
            s("'rand_multivariate_student'"),
            s("student_degrees_of_freedom"),
            OptValue::Num(7.0),
        ])),
    )
    .unwrap();
    assert_eq!(so.method, SamplingMethod::Independent);
    assert_eq!(so.student_df, 7.0);
    assert!(
        SamplerOptions::from_options(
            Some(&s("independent_metropolis_hastings")),
            Some(&OptValue::List(vec![s("proposal_distribution"), s("foo")]))
        )
        .is_err()
    );
}

/// SW07 at the reference mode: RW vs IMH vs slice vs TaRB, modest draws.
/// `cargo test --release --test estimation_samplers sw07 -- --ignored --nocapture`
/// (`SW07_SAMPLERS=rw,imh,slice,tarb` selects, `SW07_DRAWS_<NAME>` sets draws per chain).
#[test]
#[ignore]
fn sw07_samplers() {
    use dynare::data::Table;
    use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
    use dynare::estimation::{PosteriorOptions, mcmc_bounds, posterior_stats};
    use dynare::kalman::KalmanOptions;
    use dynare::modfile::parse_mod_file;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07");
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("ref.json")).unwrap()).unwrap();
    let xmode: Vec<f64> = r["xparam_mode"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let n = xmode.len();
    let hh: Vec<Vec<f64>> = r["hh_mode"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let hh = Mat::from_fn(n, n, |i, j| hh[i][j]);
    let invhess = dynare::estimation::inverse_hessian(&hh).unwrap();
    let mf = parse_mod_file(&dir.join("sw07.mod"), &[("STAGE".into(), "9".into())]).unwrap();
    let m = DsgeModel::new(&mf, Some(&xmode)).unwrap();
    let table = Table::read_csv(&dir.join("sw07_obs.csv")).unwrap();
    let data = LikelihoodData::new(
        &table.sample(&mf.varobs, 5, None).unwrap(),
        KalmanOptions { presample: 8, lik_init: 3, ..Default::default() },
    );
    // prior_trunc = 1e-10 (Dynare's default) so that the slice sampler has finite bounds
    let bounds = mcmc_bounds(&xmode, &m.priors, 1e-10).unwrap();
    let post = DsgePosterior { dsge: &m, data: &data, bounds: &bounds };
    println!("SW07: {n} parameters, minus log posterior at the mode {:.6}", post.eval(&xmode).0);
    let which = std::env::var("SW07_SAMPLERS").unwrap_or("rw,imh,slice,tarb".into());
    let draws =
        |k: &str, d: usize| std::env::var(format!("SW07_DRAWS_{}", k.to_uppercase())).map_or(d, |v| v.parse().unwrap());
    let nb = 4;
    let mut results: Vec<(String, Vec<f64>, Vec<f64>)> = vec![];
    for name in which.split(',') {
        let (so, replic, js) = match name {
            "rw" => (opts("random_walk_metropolis_hastings", &[]), draws(name, 20_000), 0.3),
            "imh" => (opts("independent_metropolis_hastings", &[]).with_student(10.0), draws(name, 20_000), 1.0),
            "slice" => {
                (opts("slice", &[("slice_initialize_with_mode", 1.0)]).with_priors(&m.priors), draws(name, 400), 1.0)
            }
            "tarb" => {
                (opts("tailored_random_block_metropolis_hastings", &[]).with_priors(&m.priors), draws(name, 40), 1.0)
            }
            _ => panic!("unknown sampler {name}"),
        };
        let mut o = mh(n, replic, nb, js);
        o.store_from = replic / 4;
        let t = Instant::now();
        let res = posterior_sampler_full(&post, &xmode, &invhess, &bounds, &o, &so, None).unwrap();
        let secs = t.elapsed().as_secs_f64();
        let st = posterior_stats(&res.mh.chains, &PosteriorOptions { mh_drop: 0.25, ..Default::default() }).unwrap();
        let sd: Vec<f64> = st.params.iter().map(|p| p.var.sqrt()).collect();
        println!(
            "{name:>6}: {nb} x {replic} draws in {secs:.1}s ({:.1} draws/s, {:.0} evals/draw), acceptance {:.3}, max ifac {:.1}",
            (nb * replic) as f64 / secs,
            res.evals_per_draw.iter().sum::<f64>() / nb as f64,
            res.acceptance.iter().sum::<f64>() / nb as f64,
            st.ifac.iter().cloned().fold(0.0, f64::max)
        );
        results.push((name.to_string(), st.mean.clone(), sd));
    }
    let (_, m0, s0) = &results[0];
    for (name, mm, ss) in &results[1..] {
        let dm: Vec<f64> = (0..n).map(|i| (mm[i] - m0[i]).abs() / s0[i]).collect();
        let ds: Vec<f64> = (0..n).map(|i| (ss[i] / s0[i] - 1.0).abs()).collect();
        let med = |v: &[f64]| {
            let mut w = v.to_vec();
            w.sort_by(f64::total_cmp);
            w[w.len() / 2]
        };
        println!(
            "{name:>6} vs {}: |mean diff|/sd median {:.3} max {:.3}; |sd ratio-1| median {:.3} max {:.3}",
            results[0].0,
            med(&dm),
            dm.iter().cloned().fold(0.0, f64::max),
            med(&ds),
            ds.iter().cloned().fold(0.0, f64::max)
        );
    }
}
