// SPDX-License-Identifier: GPL-3.0-or-later
//! Hamilton filter and Kim smoother for the Markov-switching SBVAR.
//!
//! Conventions of the Waggoner-Zha code: `Q[(i, j)] = P(s_t = i | s_{t-1} = j)` (columns sum to
//! one); the initial regime `s_0` is uniform on the `S` regimes, the first observation is
//! predicted with `Q * p0`. Probabilities are stored for `t = 0..=T` (row 0 is the initial
//! distribution, resp. the smoothed `P(s_0 | Y_T)`), as in `filtered_*.out` / `smoothed_*.out`.

use crate::linalg::Mat;

pub struct Filtered {
    /// `sum_t ln p(y_t | Y_{t-1})`.
    pub loglik: f64,
    /// `ln p(y_t | Y_{t-1})`, length `T`.
    pub lnp: Vec<f64>,
    /// `(T+1) x S`: `P(s_t | Y_t)`, row 0 the initial (uniform) distribution.
    pub filtered: Mat,
    /// `(T+1) x S`: `P(s_t | Y_{t-1})` (row 0 unused, equals the initial distribution).
    pub predicted: Mat,
}

/// `lnd`: `T x S` conditional log densities `ln p(y_t | Y_{t-1}, s_t = s)`.
pub fn filter(lnd: &Mat, q: &Mat) -> Filtered {
    let (t_n, s_n) = (lnd.nrows(), lnd.ncols());
    let mut filtered = Mat::zeros(t_n + 1, s_n);
    let mut predicted = Mat::zeros(t_n + 1, s_n);
    for s in 0..s_n {
        filtered[(0, s)] = 1.0 / s_n as f64;
        predicted[(0, s)] = 1.0 / s_n as f64;
    }
    let mut lnp = Vec::with_capacity(t_n);
    let mut loglik = 0.0;
    for t in 0..t_n {
        let mut pred = vec![0.0; s_n];
        for i in 0..s_n {
            for j in 0..s_n {
                pred[i] += q[(i, j)] * filtered[(t, j)];
            }
        }
        let mx = (0..s_n).map(|s| lnd[(t, s)]).fold(f64::NEG_INFINITY, f64::max);
        if !mx.is_finite() {
            loglik = f64::NEG_INFINITY;
            lnp.push(f64::NEG_INFINITY);
            for s in 0..s_n {
                predicted[(t + 1, s)] = pred[s];
                filtered[(t + 1, s)] = pred[s];
            }
            continue;
        }
        let w: Vec<f64> = (0..s_n).map(|s| (lnd[(t, s)] - mx).exp() * pred[s]).collect();
        let sum: f64 = w.iter().sum();
        let l = mx + sum.ln();
        lnp.push(l);
        loglik += l;
        for s in 0..s_n {
            predicted[(t + 1, s)] = pred[s];
            filtered[(t + 1, s)] = w[s] / sum;
        }
    }
    Filtered { loglik, lnp, filtered, predicted }
}

/// Kim's backward recursion: `(T+1) x S` smoothed probabilities `P(s_t | Y_T)`.
pub fn smooth(f: &Filtered, q: &Mat) -> Mat {
    let t_n = f.filtered.nrows() - 1;
    let s_n = f.filtered.ncols();
    let mut sm = Mat::zeros(t_n + 1, s_n);
    for s in 0..s_n {
        sm[(t_n, s)] = f.filtered[(t_n, s)];
    }
    for t in (0..t_n).rev() {
        for j in 0..s_n {
            let mut acc = 0.0;
            for i in 0..s_n {
                let p = f.predicted[(t + 1, i)];
                if p > 0.0 {
                    acc += q[(i, j)] * sm[(t + 1, i)] / p;
                }
            }
            sm[(t, j)] = f.filtered[(t, j)] * acc;
        }
    }
    sm
}

/// Draws a regime path `s_0..s_T` from `p(S_T | Y_T)` (backward sampling); `u` yields uniforms.
pub fn sample_path(f: &Filtered, q: &Mat, u: &mut dyn FnMut() -> f64) -> Vec<usize> {
    let t_n = f.filtered.nrows() - 1;
    let s_n = f.filtered.ncols();
    let pick = |w: &[f64], r: f64| -> usize {
        let tot: f64 = w.iter().sum();
        let mut c = 0.0;
        for (i, x) in w.iter().enumerate() {
            c += x / tot;
            if r <= c {
                return i;
            }
        }
        w.len() - 1
    };
    let mut path = vec![0usize; t_n + 1];
    path[t_n] = pick(&(0..s_n).map(|s| f.filtered[(t_n, s)]).collect::<Vec<_>>(), u());
    for t in (0..t_n).rev() {
        let nxt = path[t + 1];
        let w: Vec<f64> = (0..s_n).map(|j| f.filtered[(t, j)] * q[(nxt, j)]).collect();
        path[t] = pick(&w, u());
    }
    path
}

/// Real-time smoothed probabilities (`ms_compute_probabilities(real_time_smoothed)`): row `k`, regime
/// block `c >= k` is `P(s_k | Y_c)` (smoothing at `k` with the data up to `c`; `c = k` is the filtered
/// probability, the last block `c = T` the full-sample smoothed one). Result `(T+1) x ((T+1) S)` with
/// NaN below the diagonal (blank in Dynare's file).
pub fn real_time_smoothed(f: &Filtered, q: &Mat) -> Mat {
    let t_n = f.filtered.nrows() - 1;
    let s_n = f.filtered.ncols();
    let mut out = Mat::from_element(t_n + 1, (t_n + 1) * s_n, f64::NAN);
    for c in 0..=t_n {
        let trunc = Filtered {
            loglik: 0.0,
            lnp: Vec::new(),
            filtered: f.filtered.rows(0, c + 1).into_owned(),
            predicted: f.predicted.rows(0, c + 1).into_owned(),
        };
        let sm = smooth(&trunc, q);
        for k in 0..=c {
            for s in 0..s_n {
                out[(k, c * s_n + s)] = sm[(k, s)];
            }
        }
    }
    out
}
