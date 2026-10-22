// SPDX-License-Identifier: GPL-3.0-or-later
//! Unit tests: small state-space models against a brute-force joint-Gaussian computation
//! (likelihood, smoothed states/shocks, one-step predictions, smoothed variances), the
//! multivariate vs univariate filters, and the diffuse filters vs a large-variance prior.

use super::*;
use crate::linalg::Mat;

/// Deterministic pseudo-random numbers in (-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

struct Model {
    t: Mat,
    r: Mat,
    q: Mat,
    mf: Vec<usize>,
}

/// 4 states (the last one static: zero column in T), 2 shocks, 2 observables.
fn stationary_model() -> Model {
    let t = Mat::from_row_slice(
        4,
        4,
        &[
            0.7, 0.2, 0.0, 0.0, //
            -0.1, 0.5, 0.3, 0.0, //
            0.0, 0.0, 0.6, 0.0, //
            0.4, 0.1, 0.2, 0.0,
        ],
    );
    let r = Mat::from_row_slice(4, 2, &[1.0, 0.0, 0.3, 0.5, 0.0, 1.0, 0.5, 0.2]);
    let q = Mat::from_row_slice(2, 2, &[0.8, 0.1, 0.1, 0.5]);
    Model { t, r, q, mf: vec![3, 0] }
}

/// random walk (state 0) + AR(1) (state 1); observables: level = rw + ar (static state 2),
/// and the AR state.
fn diffuse_model() -> Model {
    let t = Mat::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 0.6, 0.0, 1.0, 0.6, 0.0]);
    let r = Mat::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let q = Mat::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 1.0]);
    Model { t, r, q, mf: vec![2, 1] }
}

fn simulate(m: &Model, n: usize, h: &[f64], seed: u64) -> Mat {
    let mut g = Lcg(seed);
    let (mm, rr, pp) = (m.t.nrows(), m.q.nrows(), m.mf.len());
    let lq = m.q.clone().cholesky().unwrap().l();
    let mut a = Vector::zeros(mm);
    let mut y = Mat::zeros(pp, n);
    for t in 0..n {
        let e = Vector::from_fn(rr, |_, _| g.next() * 1.7);
        a = &m.t * &a + &m.r * (&lq * e);
        for i in 0..pp {
            y[(i, t)] = a[m.mf[i]] + h[i].sqrt() * g.next() * 1.7;
        }
    }
    y
}

use crate::linalg::Vector;

/// Joint Gaussian of everything, in terms of the primitives u = [alpha_0; eta_1..eta_n] with
/// alpha_t = T alpha_{t-1} + R eta_t, Var(alpha_0) = p0.
struct Joint {
    a: Vec<Mat>, // alpha_t = a[t] u, t = 0..=n
    su: Mat,
    rows: Vec<(usize, usize)>, // observed (t, i), t 1-based
    cyy: Mat,
    ycol: Vector,
}

fn joint(m: &Model, y: &Mat, h: &[f64], p0: &Mat) -> Joint {
    let (mm, rr, pp, n) = (m.t.nrows(), m.q.nrows(), m.mf.len(), y.ncols());
    let dim = mm + rr * n;
    let mut su = Mat::zeros(dim, dim);
    su.view_mut((0, 0), (mm, mm)).copy_from(p0);
    for t in 0..n {
        su.view_mut((mm + rr * t, mm + rr * t), (rr, rr)).copy_from(&m.q);
    }
    let mut a = vec![Mat::zeros(mm, dim)];
    a[0].view_mut((0, 0), (mm, mm)).copy_from(&Mat::identity(mm, mm));
    for t in 1..=n {
        let mut at = &m.t * &a[t - 1];
        let blk = at.view((0, mm + rr * (t - 1)), (mm, rr)) + &m.r;
        at.view_mut((0, mm + rr * (t - 1)), (mm, rr)).copy_from(&blk);
        a.push(at);
    }
    let mut rows = Vec::new();
    for t in 1..=n {
        for i in 0..pp {
            if !y[(i, t - 1)].is_nan() {
                rows.push((t, i));
            }
        }
    }
    let ny = rows.len();
    let b = Mat::from_fn(ny, dim, |k, j| a[rows[k].0][(m.mf[rows[k].1], j)]);
    let mut cyy = &b * &su * b.transpose();
    for k in 0..ny {
        cyy[(k, k)] += h[rows[k].1];
    }
    let ycol = Vector::from_fn(ny, |k, _| y[(rows[k].1, rows[k].0 - 1)]);
    Joint { a, su, rows, cyy, ycol }
}

impl Joint {
    fn minus_loglik(&self) -> f64 {
        let ny = self.ycol.len() as f64;
        let ch = self.cyy.clone().cholesky().unwrap();
        let logdet: f64 = 2.0 * ch.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
        let q = self.ycol.dot(&ch.solve(&self.ycol));
        0.5 * (ny * (2.0 * std::f64::consts::PI).ln() + logdet + q)
    }
    /// E[alpha_t | observations with period <= upto], Var(alpha_t | same)
    fn cond_state(&self, m: &Model, t: usize, upto: usize) -> (Vector, Mat) {
        let keep: Vec<usize> = (0..self.rows.len()).filter(|&k| self.rows[k].0 <= upto).collect();
        let at = &self.a[t];
        let dim = self.su.nrows();
        let b = Mat::from_fn(keep.len(), dim, |k, j| self.a[self.rows[keep[k]].0][(m.mf[self.rows[keep[k]].1], j)]);
        let cyy = self.cyy.select_rows(&keep).select_columns(&keep);
        let yk = Vector::from_fn(keep.len(), |k, _| self.ycol[keep[k]]);
        let cay = at * &self.su * b.transpose();
        if keep.is_empty() {
            return (Vector::zeros(at.nrows()), at * &self.su * at.transpose());
        }
        let inv = cyy.try_inverse().unwrap();
        let mean = &cay * &inv * yk;
        let var = at * &self.su * at.transpose() - &cay * inv * cay.transpose();
        (mean, var)
    }
    /// E[eta_t | all observations], t 1-based
    fn cond_shock(&self, m: &Model, t: usize) -> Vector {
        let rr = m.q.nrows();
        let mm = m.t.nrows();
        let dim = self.su.nrows();
        let e = Mat::from_fn(rr, dim, |i, j| if j == mm + rr * (t - 1) + i { 1.0 } else { 0.0 });
        let b = Mat::from_fn(self.rows.len(), dim, |k, j| self.a[self.rows[k].0][(m.mf[self.rows[k].1], j)]);
        let c = &e * &self.su * b.transpose();
        c * self.cyy.clone().try_inverse().unwrap() * &self.ycol
    }
}

fn opts(lik_init: u8, algo: u8) -> KalmanOptions {
    KalmanOptions { lik_init, kalman_algo: algo, riccati_tol: 0.0, ..Default::default() }
}

fn hmat(h: &[f64]) -> Mat {
    Mat::from_fn(h.len(), h.len(), |i, j| if i == j { h[i] } else { 0.0 })
}

fn with_holes(mut y: Mat) -> Mat {
    y[(0, 1)] = f64::NAN;
    y[(1, 4)] = f64::NAN;
    y[(0, 4)] = f64::NAN; // fully missing period
    y[(1, 6)] = f64::NAN;
    y
}

#[test]
fn missing_data_description() {
    let y = with_holes(Mat::from_element(2, 10, 1.0));
    let md = MissingData::from_data(&y);
    assert!(md.any_missing);
    assert_eq!(md.number_of_observations, 16);
    assert_eq!(md.aindex[4], Vec::<usize>::new());
    assert_eq!(md.aindex[1], vec![1]);
    assert_eq!(md.no_more_missing_observations, 7); // Dynare: 8 (1-based)
    let full = MissingData::from_data(&Mat::zeros(2, 3));
    assert!(!full.any_missing && full.no_more_missing_observations == 0);
}

#[test]
fn stationary_likelihood_vs_joint_gaussian() {
    let m = stationary_model();
    let h = [0.3, 0.2];
    let p0 = lyapunov_solver(&m.t, &m.r, &m.q, 1.000001, 1e-15).unwrap();
    for holes in [false, true] {
        let y0 = simulate(&m, 12, &h, 7);
        let y = if holes { with_holes(y0) } else { y0 };
        let md = MissingData::from_data(&y);
        let want = joint(&m, &y, &h, &p0).minus_loglik();
        let zero = Mat::zeros(4, 4);
        for algo in [1u8, 2] {
            let out = likelihood(&y, &md, &m.mf, &m.t, &m.r, &m.q, Some(&hmat(&h)), &p0, &zero, &opts(1, algo));
            assert_eq!(out.info, 0);
            assert_eq!(out.univariate, algo == 2);
            assert!(
                (out.minus_loglik - want).abs() < 1e-10 * want.abs(),
                "algo {algo} holes {holes}: {} vs {want}",
                out.minus_loglik
            );
            assert_eq!(out.lik.len(), 12);
            assert!((out.lik.iter().sum::<f64>() - want).abs() < 1e-9);
        }
        // presample drops the first periods
        let o = KalmanOptions { presample: 3, ..opts(1, 1) };
        let out = likelihood(&y, &md, &m.mf, &m.t, &m.r, &m.q, Some(&hmat(&h)), &p0, &zero, &o);
        let tail: f64 = out.lik[3..].iter().sum();
        assert!((out.minus_loglik - tail).abs() < 1e-12);
    }
}

#[test]
fn steady_state_shortcut_close_to_full_recursion() {
    let m = stationary_model();
    let h = [0.3, 0.2];
    let y = simulate(&m, 200, &h, 11);
    let md = MissingData::from_data(&y);
    let p0 = lyapunov_solver(&m.t, &m.r, &m.q, 1.000001, 1e-15).unwrap();
    let zero = Mat::zeros(4, 4);
    let mut ws = KalmanWorkspace::new(4, 2, 2);
    for algo in [1u8, 2] {
        let full = ws.likelihood(&y, &md, &m.mf, &m.t, &m.r, &m.q, Some(&hmat(&h)), &p0, &zero, &opts(1, algo));
        let o = KalmanOptions { riccati_tol: 1e-6, ..opts(1, algo) };
        let fast = ws.likelihood(&y, &md, &m.mf, &m.t, &m.r, &m.q, Some(&hmat(&h)), &p0, &zero, &o);
        assert!(ws.steady_state_start.is_some(), "steady state reached");
        assert!((full.minus_loglik - fast.minus_loglik).abs() < 1e-6, "{} {}", full.minus_loglik, fast.minus_loglik);
    }
}

#[test]
fn singular_multivariate_falls_back_to_univariate() {
    // two observables of the same state, no measurement error: F singular
    let m = Model { mf: vec![0, 0], ..stationary_model() };
    let y0 = simulate(&stationary_model(), 10, &[0.0, 0.0], 3);
    let y = Mat::from_fn(2, 10, |_, t| y0[(1, t)]);
    let md = MissingData::from_data(&y);
    let p0 = lyapunov_solver(&m.t, &m.r, &m.q, 1.000001, 1e-15).unwrap();
    let zero = Mat::zeros(4, 4);
    let out = likelihood(&y, &md, &m.mf, &m.t, &m.r, &m.q, None, &p0, &zero, &opts(1, 0));
    assert_eq!(out.info, 0);
    assert!(out.univariate);
    // univariate: the second (identical) observation has Fi = 0 and is skipped
    let m1 = Model { mf: vec![0], ..stationary_model() };
    let y1 = y.rows(0, 1).into_owned();
    let one = likelihood(&y1, &MissingData::from_data(&y1), &m1.mf, &m1.t, &m1.r, &m1.q, None, &p0, &zero, &opts(1, 1));
    assert!((out.minus_loglik - one.minus_loglik).abs() < 1e-8, "{} {}", out.minus_loglik, one.minus_loglik);
    let strict = KalmanOptions { use_univariate_filters_if_singularity_is_detected: false, ..opts(1, 0) };
    let out = likelihood(&y, &md, &m.mf, &m.t, &m.r, &m.q, None, &p0, &zero, &strict);
    assert_eq!(out.info, 50);
    assert!(out.minus_loglik.is_infinite());
}

#[test]
fn diffuse_filters_vs_large_prior_variance() {
    let m = diffuse_model();
    let h = [0.2, 0.1];
    let (pstar, pinf) = compute_pinf_pstar(&m.mf, &m.t, &m.r, &m.q, 1.000001, &[]);
    for holes in [false, true] {
        let y0 = simulate(&m, 15, &h, 5);
        let y = if holes { with_holes(y0) } else { y0 };
        let md = MissingData::from_data(&y);
        let mv = likelihood(&y, &md, &m.mf, &m.t, &m.r, &m.q, Some(&hmat(&h)), &pstar, &pinf, &opts(3, 0));
        let uv = likelihood(&y, &md, &m.mf, &m.t, &m.r, &m.q, Some(&hmat(&h)), &pstar, &pinf, &opts(3, 4));
        assert_eq!((mv.info, uv.info), (0, 0));
        assert!(!mv.univariate && uv.univariate);
        assert!(mv.diffuse_periods >= 1 && uv.diffuse_periods >= 1);
        assert!((mv.minus_loglik - uv.minus_loglik).abs() < 1e-9, "{} {}", mv.minus_loglik, uv.minus_loglik);
        // exact diffuse likelihood = lim (minus loglik with P0 = Pstar + k*Pinf) - 0.5*rank*log(k)
        let k = 1e7;
        let p0 = &pstar + &pinf * k;
        let big = joint(&m, &y, &h, &p0).minus_loglik();
        let rank = 1.0;
        let approx = big - 0.5 * rank * k.ln();
        assert!((mv.minus_loglik - approx).abs() < 1e-4, "{} vs {approx}", mv.minus_loglik);
    }
}

#[test]
fn smoother_vs_joint_gaussian() {
    let m = stationary_model();
    let h = [0.3, 0.2];
    let p0 = lyapunov_solver(&m.t, &m.r, &m.q, 1.000001, 1e-15).unwrap();
    let y = with_holes(simulate(&m, 10, &h, 9));
    let md = MissingData::from_data(&y);
    let j = joint(&m, &y, &h, &p0);
    for algo in [1u8, 2] {
        let out = smoother(&y, &md, &m.mf, &m.t, &m.r, &m.q, Some(&hmat(&h)), &p0, None, &opts(1, algo), true).unwrap();
        assert_eq!(out.univariate, algo == 2);
        assert_eq!(out.d, 0);
        let mut err: f64 = 0.0;
        for t in 1..=10 {
            let (mean, var) = j.cond_state(&m, t, 10);
            err = err.max((out.alphahat.column(t - 1) - &mean).amax());
            err = err.max((&out.state_uncertainty.as_ref().unwrap()[t - 1] - var).amax());
            err = err.max((out.etahat.column(t - 1) - j.cond_shock(&m, t)).amax());
            let (upd, _) = j.cond_state(&m, t, t);
            err = err.max((out.ahat.column(t - 1) - upd).amax());
            if t < 10 {
                let (pred, _) = j.cond_state(&m, t + 1, t);
                err = err.max((out.filtered.column(t - 1) - pred).amax());
            }
        }
        let (m0, _) = j.cond_state(&m, 0, 10);
        err = err.max((&out.alphahat0 - m0).amax());
        assert!(err < 1e-10, "algo {algo}: max error {err}");
    }
}

#[test]
fn diffuse_smoothers_agree_and_match_large_prior() {
    // pp = 2: Finf (2x2) has rank 1 -> H1 fails (420) and H3 is used;
    // pp = 1: H1's multivariate diffuse smoother works and must agree with H3.
    for (mf, h) in [(vec![2usize, 1], vec![0.2, 0.1]), (vec![2usize], vec![0.2])] {
        let m = Model { mf, ..diffuse_model() };
        let pp = m.mf.len();
        let (pstar, pinf) = compute_pinf_pstar(&m.mf, &m.t, &m.r, &m.q, 1.000001, &[]);
        let mut y = simulate(&m, 12, &h, 21);
        y[(0, 1)] = f64::NAN;
        y[(pp - 1, 6)] = f64::NAN;
        let md = MissingData::from_data(&y);
        let hm = hmat(&h);
        let uv = smoother(&y, &md, &m.mf, &m.t, &m.r, &m.q, Some(&hm), &pstar, Some(&pinf), &opts(3, 4), true).unwrap();
        assert!(uv.univariate && uv.d >= 1);
        // large-variance prior on the diffuse state: alpha_0 ~ N(0, Pstar + k*Pinf)
        // (DsgeSmoother's diffuse smoothers start from T*Pstar*T' + RQR' at t = 1).
        let k = 1e8;
        let j = joint(&m, &y, &h, &(&pstar + &pinf * k));
        let mut err: f64 = 0.0;
        for t in 1..=12 {
            let (mean, _) = j.cond_state(&m, t, 12);
            err = err.max((uv.alphahat.column(t - 1) - mean).amax());
            err = err.max((uv.etahat.column(t - 1) - j.cond_shock(&m, t)).amax());
        }
        assert!(err < 1e-5, "pp {pp}: univariate diffuse smoother vs large prior: {err}");
        let mv = smoother(&y, &md, &m.mf, &m.t, &m.r, &m.q, Some(&hm), &pstar, Some(&pinf), &opts(3, 0), true).unwrap();
        if pp == 2 {
            assert!(mv.univariate && mv.info == 420);
        } else {
            assert!(!mv.univariate && mv.d == uv.d, "H1 used");
            let e = (&mv.alphahat - &uv.alphahat).amax().max((&mv.etahat - &uv.etahat).amax());
            assert!(e < 1e-8, "multivariate vs univariate diffuse smoother: {e}");
            let ev = (0..12)
                .map(|t| {
                    (&mv.state_uncertainty.as_ref().unwrap()[t] - &uv.state_uncertainty.as_ref().unwrap()[t]).amax()
                })
                .fold(0.0, f64::max);
            assert!(ev < 1e-8, "state uncertainty H1 vs H3: {ev}");
        }
    }
}

#[test]
fn workspace_is_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<KalmanWorkspace>();
    check::<LikOutput>();
    check::<SmootherOutput>();
}
