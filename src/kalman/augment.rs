// SPDX-License-Identifier: GPL-3.0-or-later
//! State augmentation for correlated measurement errors in the univariate filters and
//! smoother (Durbin and Koopman 2012, section 6.4.3), as dsge_likelihood.m and
//! DsgeSmoother.m do when H is not diagonal.
//!
//! Dynare appends the measurement errors e_t to the state, b_t = [a_t; e_t], with
//! T_b = blkdiag(T, 0), R_b = blkdiag(R, I), Q_b = blkdiag(Q, H), Z_b = [Z, I], H_b = 0.
//! Our filters observe single states (selection `mf`), so we use the equivalent state
//! c_t = M b_t = [a_t; Z a_t + e_t] (M = [I 0; Z I]): y_t = c_t(mm + i) exactly, and
//!
//!   c_{t+1} = [T 0; Z T 0] c_t + [R 0; Z R I] [u; e],   Q_c = blkdiag(Q, H),
//!   P_c = M blkdiag(P, H) M',  Pinf_c = M blkdiag(Pinf, 0) M'.
//!
//! The first mm components of c are a, and the shock vector [u; e] is the same as Dynare's,
//! so smoothed shocks and measurement errors (the last pp smoothed shocks) carry over as they
//! are. The likelihood is the same function of the data; only rounding differs.

use crate::linalg::Mat;

pub(crate) struct AugmentedSystem {
    pub t: Mat,
    pub r: Mat,
    pub q: Mat,
    /// observed rows of the augmented state: mm..mm+pp
    pub mf: Vec<usize>,
}

pub(crate) fn system(t: &Mat, r: &Mat, q: &Mat, h: &Mat, mf: &[usize]) -> AugmentedSystem {
    let (mm, rr, pp) = (t.nrows(), r.ncols(), mf.len());
    let mut tc = Mat::zeros(mm + pp, mm + pp);
    let mut rc = Mat::zeros(mm + pp, rr + pp);
    tc.view_mut((0, 0), (mm, mm)).copy_from(t);
    rc.view_mut((0, 0), (mm, rr)).copy_from(r);
    for (i, &k) in mf.iter().enumerate() {
        for j in 0..mm {
            tc[(mm + i, j)] = t[(k, j)];
        }
        for j in 0..rr {
            rc[(mm + i, j)] = r[(k, j)];
        }
        rc[(mm + i, rr + i)] = 1.0;
    }
    let mut qc = Mat::zeros(rr + pp, rr + pp);
    qc.view_mut((0, 0), (rr, rr)).copy_from(q);
    qc.view_mut((rr, rr), (pp, pp)).copy_from(h);
    AugmentedSystem { t: tc, r: rc, q: qc, mf: (mm..mm + pp).collect() }
}

/// M blkdiag(P, H) M' (H = None: zero block, for Pinf).
pub(crate) fn cov(p: &Mat, h: Option<&Mat>, mf: &[usize]) -> Mat {
    let (mm, pp) = (p.nrows(), mf.len());
    let mut c = Mat::zeros(mm + pp, mm + pp);
    c.view_mut((0, 0), (mm, mm)).copy_from(p);
    for (i, &k) in mf.iter().enumerate() {
        for j in 0..mm {
            c[(mm + i, j)] = p[(k, j)];
            c[(j, mm + i)] = p[(j, k)];
        }
    }
    for (i, &k) in mf.iter().enumerate() {
        for (j, &l) in mf.iter().enumerate() {
            c[(mm + i, mm + j)] = p[(k, l)] + h.map_or(0.0, |h| h[(i, j)]);
        }
    }
    c
}

/// [a; Z a] (the measurement errors' prediction is 0).
pub(crate) fn state(a: &[f64], mf: &[usize]) -> Vec<f64> {
    a.iter().copied().chain(mf.iter().map(|&k| a[k])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cov_matches_m_blkdiag_m() {
        let p = Mat::from_fn(3, 3, |i, j| 1.0 / (1.0 + i as f64 + j as f64));
        let h = Mat::from_row_slice(2, 2, &[0.5, 0.1, 0.1, 0.3]);
        let mf = [2, 0];
        let mut m = Mat::identity(5, 5);
        m[(3, 2)] = 1.0;
        m[(4, 0)] = 1.0;
        let mut b = Mat::zeros(5, 5);
        b.view_mut((0, 0), (3, 3)).copy_from(&p);
        b.view_mut((3, 3), (2, 2)).copy_from(&h);
        let want = &m * b * m.transpose();
        assert!((cov(&p, Some(&h), &mf) - want).amax() < 1e-15);
    }
}
