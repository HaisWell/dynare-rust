//! `--low-memory`: run-length encoded chains ([`Chain::runs`]) give the same draws and the
//! same posterior statistics as dense ones, and [`dynare::estimation::mcmc::bounded`] does not
//! change results. A test binary of its own: the low-memory switch is process-wide.

use dynare::estimation::Chain;
use dynare::estimation::mcmc::{bounded, set_low_memory};
use dynare::estimation::posterior::{PosteriorOptions, posterior_stats};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// A random-walk-like sequence: each step repeats the previous draw with probability
/// `1 - accept`, as a rejected Metropolis-Hastings proposal does.
fn mh_like(npar: usize, n: usize, accept: f64, seed: u64) -> Vec<(Vec<f64>, f64)> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut x = vec![0.0; npar];
    let mut lp = 0.0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if rng.r#gen::<f64>() < accept {
            for v in x.iter_mut() {
                *v += 0.3 * rng.sample::<f64, _>(StandardNormal);
            }
            lp = -0.5 * x.iter().map(|v| v * v).sum::<f64>();
        }
        out.push((x.clone(), lp));
    }
    out
}

fn chain(seq: &[(Vec<f64>, f64)], npar: usize, rle: bool) -> Chain {
    let mut c = Chain::empty(npar);
    for (x, lp) in seq {
        c.push(x, *lp, rle);
    }
    c.total_draws = seq.len();
    c.accepted = seq.len() / 3;
    c.init = vec![0.0; npar];
    c.init_logpost = 0.0;
    c
}

#[test]
fn run_length_encoding_is_transparent() {
    let (npar, n) = (5, 20_000);
    let seqs: Vec<_> = (0..4).map(|b| mh_like(npar, n, 0.3, 11 + b)).collect();
    let dense: Vec<Chain> = seqs.iter().map(|s| chain(s, npar, false)).collect();
    let rle: Vec<Chain> = seqs.iter().map(|s| chain(s, npar, true)).collect();
    for (d, r) in dense.iter().zip(&rle) {
        assert_eq!(d.len(), n);
        assert_eq!(r.len(), n);
        assert!(r.runs.len() < n / 2, "{} rows kept of {n}", r.runs.len());
        assert_eq!(r.draws.len(), r.runs.len() * npar);
        for k in 0..n {
            assert_eq!(d.draw(k), r.draw(k), "draw {k}");
            assert_eq!(d.logpost_at(k).to_bits(), r.logpost_at(k).to_bits(), "logpost {k}");
        }
        assert_eq!(d.last(), r.last());
    }
    let o = PosteriorOptions::default();
    let a = posterior_stats(&dense, &o).unwrap();
    set_low_memory(true);
    let b = bounded(|| posterior_stats(&rle, &o)).unwrap();
    set_low_memory(false);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.mean), bits(&b.mean));
    assert_eq!(bits(a.covariance.as_slice()), bits(b.covariance.as_slice()));
    assert_eq!(a.mhm.log_marginal.to_bits(), b.mhm.log_marginal.to_bits());
    for (p, q) in a.params.iter().zip(&b.params) {
        assert_eq!((p.hpd.0.to_bits(), p.hpd.1.to_bits()), (q.hpd.0.to_bits(), q.hpd.1.to_bits()));
        assert_eq!(p.median.to_bits(), q.median.to_bits());
        assert_eq!(p.var.to_bits(), q.var.to_bits());
    }
}

#[test]
fn stored_from_an_offset() {
    // chains read back from the mh files start at the first kept draw (`first_stored > 0`)
    let (npar, n, from) = (3, 5_000, 2_000);
    let seq = mh_like(npar, n, 0.25, 5);
    let mut d = chain(&seq[from..], npar, false);
    let mut r = chain(&seq[from..], npar, true);
    d.first_stored = from;
    r.first_stored = from;
    for k in from..n {
        assert_eq!(d.draw(k), r.draw(k));
        assert_eq!(d.logpost_at(k).to_bits(), r.logpost_at(k).to_bits());
    }
    assert_eq!(d.draw(from), seq[from].0.as_slice());
}
