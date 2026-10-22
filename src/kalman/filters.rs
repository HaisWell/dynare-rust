// SPDX-License-Identifier: GPL-3.0-or-later
//! Likelihood filters of `kalman/likelihood/` on a preallocated workspace.
//!
//! All filters work on the restricted state space (mm states, pp observables, rr shocks) with
//! a selection measurement equation y_t = a[mf] (+ measurement error), which is Dynare's
//! `Zflag = 0` form (and equivalent to its `Zflag = 1` selection matrix). The state mean `a`
//! and covariance `P`/`Pstar`/`Pinf` live in the workspace and are updated in place, so a
//! sequence of calls mirrors the MATLAB call sequence of `dsge_likelihood.m`.
//!
//! Speed-ups that keep Dynare's arithmetic (up to rounding):
//! * Z is a selection: Z*a, Z*P*Z' and P*Z' are gathers.
//! * T*M*T' only involves the non-zero columns of T (Dynare's restricted state space keeps
//!   static variables, whose columns are zero), and only the corresponding block of M.
//!   Rows of T with at most two non-zeros (lags, AR(1) processes: about half of them in
//!   semi-structural models) are handled by copies, and the two gemms run over the dense rows only.
//! * inv(F) by inline Gauss-Jordan elimination; LAPACK's dgecon estimate is only computed
//!   when the exact rcond (from that inverse) does not already decide the `rcond < tol` test.
//! * RQR' is computed once per likelihood.
//! * the univariate filter's sequential rank-one updates P - PZ_i K_i' are accumulated as one
//!   rank-p update (each PZ_i is corrected for the previous updates), applied to the block
//!   needed by the transition.
//! * `rank(A, tol) > 0` (the only use of `rank` in the filters) is decided from max |a_ij| and
//!   the Frobenius norm, falling back to an SVD only when those bounds are inconclusive.

#![allow(clippy::needless_range_loop)] // index loops mirror the MATLAB code

use super::{HMatrix, MissingData};
use crate::linalg::{self, Mat};
use std::os::raw::c_int;

const L2PI: f64 = 1.837_877_066_409_345_5; // log(2*pi)

/// Preallocated buffers for the filters of one model size (mm states, pp observables, rr
/// shocks). Not shared between threads (each MCMC chain owns one); `Send + Sync`.
///
/// Internally the state vector is permuted so that the states with a non-zero column in T
/// come first (`perm`); T*M*T' then only needs the leading nnz x nnz block of M, which BLAS
/// reads in place. Inputs and outputs of the public methods are in the original order.
#[derive(Clone, Debug)]
pub struct KalmanWorkspace {
    pub(crate) mm: usize,
    pub(crate) pp: usize,
    pub(crate) rr: usize,
    // system (permuted)
    nnz: usize,
    /// dense rows of T (more than 2 non-zeros): permuted positions d0..d0+nd
    d0: usize,
    nd: usize,
    /// the other rows: (permuted row, number of non-zeros, [(permuted column, value); 2])
    sparse_rows: Vec<(usize, usize, [(usize, f64); 2])>,
    perm: Vec<usize>,   // permuted position -> original state
    iperm: Vec<usize>,  // original state -> permuted position
    tp: Vec<f64>,       // mm x mm: T(perm, perm); columns nnz.. are zero
    qq: Vec<f64>,       // mm x mm: R Q R' (permuted)
    qq_idx: Vec<usize>, // rows (= columns) of QQ that can be non-zero: the non-zero rows of R
    qqc: Vec<f64>,      // QQ(qq_idx, qq_idx), column-major
    rp: Vec<f64>,       // mm x rr: R(perm, :)
    rq: Vec<f64>,       // mm x rr
    // state (permuted)
    a: Vec<f64>,
    p: Vec<f64>, // P or Pstar, mm x mm
    pinf: Vec<f64>,
    a_save: Vec<f64>,
    p_save: Vec<f64>,
    // scratch
    w: Vec<f64>,  // mm x nnz
    av: Vec<f64>, // mm
    an: Vec<f64>, // mm
    kf: Vec<f64>, // mm
    pz: Vec<f64>, // mm x pp
    pz2: Vec<f64>,
    kk: Vec<f64>, // mm x pp
    k2: Vec<f64>,
    oldk: Vec<f64>,
    f: Vec<f64>, // pp x pp
    lu: Vec<f64>,
    udiag: Vec<f64>,      // pp: pivots of the LU factorisation of the last F
    gj_scratch: Vec<f64>, // 2 pp
    finv: Vec<f64>,
    f2: Vec<f64>,
    v: Vec<f64>, // pp
    sig: Vec<f64>,
    fi: Vec<f64>, // pp
    used: Vec<bool>,
    ipiv: Vec<c_int>,
    svd_a: Vec<f64>,
    svd_s: Vec<f64>,
    svd_w: Vec<f64>,
    gecon_w: Vec<f64>,
    gecon_iw: Vec<c_int>,
    mfp: Vec<usize>, // observed rows in the permuted state
    zi: Vec<usize>,
    di: Vec<usize>,
    /// set when a log-determinant of a negative determinant was taken (MATLAB: complex value)
    pub(crate) complex: bool,
    // outputs of the last calls
    /// per-period values of the last multivariate diffuse / univariate diffuse filter
    pub dlik: Vec<f64>,
    /// `llik` of the last univariate diffuse filter: diffuse_periods x pp, row-major by period
    /// (column = observable index d_index(i), as in `univariate_kalman_filter_d.m`).
    pub dlik_by_obs: Vec<f64>,
    /// per-period values of the last regular filter (periods start..last)
    pub lik: Vec<f64>,
    /// last univariate filter: (last-start+1) x pp, row-major by period (column = position i
    /// within the period's d_index, as in `univariate_kalman_filter.m`).
    pub lik_by_obs: Vec<f64>,
    /// first period (0-based) handled by the steady-state filter in the last regular filter call
    pub steady_state_start: Option<usize>,
    // ---- heteroskedastic_filter (time-varying Q, `Qvec`) ----
    /// `Qvec(:,:,k)` (rr x rr, k = 0..=smpl), set by [`set_heteroskedastic`](Self::set_heteroskedastic)
    /// before [`set_system`](Self::set_system); `None` = homoskedastic (the default).
    qvec_raw: Option<Vec<Mat>>,
    /// per k: R Qvec(:,:,k) R' (permuted, mm x mm), then its compact block (qq_idx)
    qvec_qq: Vec<f64>,
    qvec_qqc: Vec<f64>,
    /// `filter_initial_state`: the first prediction `a_0_given_tm1 = T*a`
    /// (original state order); `None` = 0.
    pub(crate) a_init: Option<Vec<f64>>,
}

/// Result of a diffuse filter: (dLIK, number of diffuse periods). dLIK is +Inf when the
/// multivariate filter hit a singular Finf/Fstar and NaN when it never left the diffuse stage
/// (or the univariate one met a negative variance).
pub type DiffuseResult = (f64, usize);

impl KalmanWorkspace {
    /// Allocates all buffers for mm states, pp observables and rr shocks.
    /// `(mm, pp, rr)` the workspace was sized for.
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.mm, self.pp, self.rr)
    }

    pub fn new(mm: usize, pp: usize, rr: usize) -> Self {
        let m2 = mm * mm;
        let mp = mm * pp.max(1);
        let p2 = pp * pp;
        Self {
            mm,
            pp,
            rr,
            nnz: 0,
            d0: 0,
            nd: mm,
            sparse_rows: Vec::with_capacity(mm),
            perm: (0..mm).collect(),
            iperm: (0..mm).collect(),
            tp: vec![0.0; m2],
            qq: vec![0.0; m2],
            qq_idx: Vec::with_capacity(mm),
            qqc: vec![0.0; m2],
            rp: vec![0.0; mm * rr],
            rq: vec![0.0; mm * rr],
            a: vec![0.0; mm],
            p: vec![0.0; m2],
            pinf: vec![0.0; m2],
            a_save: vec![0.0; mm],
            p_save: vec![0.0; m2],
            w: vec![0.0; m2],
            av: vec![0.0; mm],
            an: vec![0.0; mm],
            kf: vec![0.0; mm],
            pz: vec![0.0; mp],
            pz2: vec![0.0; mp],
            kk: vec![0.0; mp],
            k2: vec![0.0; mp],
            oldk: vec![0.0; mp],
            f: vec![0.0; p2],
            lu: vec![0.0; p2],
            udiag: vec![0.0; pp],
            gj_scratch: vec![0.0; 2 * pp],
            finv: vec![0.0; p2],
            f2: vec![0.0; p2],
            v: vec![0.0; pp],
            sig: vec![0.0; pp],
            fi: vec![0.0; pp],
            used: vec![false; pp],
            ipiv: vec![0; pp.max(mm)],
            svd_a: vec![0.0; m2],
            svd_s: vec![0.0; mm],
            svd_w: vec![0.0; 10 * mm.max(1) + 10],
            gecon_w: vec![0.0; 4 * pp.max(1)],
            gecon_iw: vec![0; pp.max(1)],
            mfp: Vec::with_capacity(pp),
            zi: Vec::with_capacity(pp),
            di: Vec::with_capacity(pp),
            complex: false,
            dlik: Vec::new(),
            dlik_by_obs: Vec::new(),
            lik: Vec::new(),
            lik_by_obs: Vec::new(),
            steady_state_start: None,
            qvec_raw: None,
            qvec_qq: Vec::new(),
            qvec_qqc: Vec::new(),
            a_init: None,
        }
    }

    /// `filter_initial_state` (set_Kalman_starting_values.m): the initial state
    /// prediction `T*a` that [`super::KalmanWorkspace::likelihood`] starts from (None = 0).
    pub fn set_initial_state(&mut self, a0: Option<Vec<f64>>) {
        self.a_init = a0;
    }

    /// `heteroskedastic_filter`: the per-period shock covariances `Qvec`
    /// (`get_Qvec_heteroskedastic_filter.m`, k = 0..=smpl; the filters use `R*Qvec(:,:,t+1)*R'`
    /// for the transition after period t, 0-based, and never switch to the steady-state
    /// filter). `None` restores the homoskedastic filters. Call before [`set_system`](Self::set_system).
    pub fn set_heteroskedastic(&mut self, qvec: Option<Vec<Mat>>) {
        self.qvec_raw = qvec;
    }

    /// A copy of the `Qvec` set (None when homoskedastic).
    pub(crate) fn hetero_qvec(&self) -> Option<Vec<Mat>> {
        self.qvec_raw.clone()
    }

    /// Is a `Qvec` set?
    #[inline]
    pub fn is_heteroskedastic(&self) -> bool {
        self.qvec_raw.is_some()
    }

    /// QQ <- R*Qvec(:,:,k)*R' (no-op when homoskedastic).
    #[inline]
    fn hetero_period(&mut self, k: usize) {
        if self.qvec_raw.is_none() {
            return;
        }
        let m2 = self.mm * self.mm;
        let kk = self.qq_idx.len() * self.qq_idx.len();
        let n = self.qvec_qq.len() / m2.max(1);
        let k = k.min(n.saturating_sub(1));
        self.qq.copy_from_slice(&self.qvec_qq[k * m2..(k + 1) * m2]);
        self.qqc[..kk].copy_from_slice(&self.qvec_qqc[k * kk..(k + 1) * kk]);
    }

    /// R*Qvec(:,:,k)*R' for every k (permuted), after `qq_idx` is known.
    fn build_hetero(&mut self) {
        let Some(qv) = self.qvec_raw.take() else { return };
        let (mm, rr) = (self.mm, self.rr);
        let k = self.qq_idx.len();
        self.qvec_qq.clear();
        self.qvec_qqc.clear();
        let mut rq = vec![0.0; mm * rr];
        let mut qq = vec![0.0; mm * mm];
        for q in &qv {
            linalg::gemm_raw(false, false, mm, rr, rr, 1.0, &self.rp, mm, q.as_slice(), rr, 0.0, &mut rq, mm);
            linalg::gemm_raw(false, true, mm, mm, rr, 1.0, &rq, mm, &self.rp, mm, 0.0, &mut qq, mm);
            self.qvec_qq.extend_from_slice(&qq);
            for &c in &self.qq_idx {
                for &r in &self.qq_idx {
                    self.qvec_qqc.push(qq[c * mm + r]);
                }
            }
        }
        let _ = k;
        self.qvec_raw = Some(qv);
    }

    /// Loads T and R (permuted, see below) and computes RQR' (`QQ = R*Q*transpose(R)` of
    /// every filter).
    ///
    /// The states are ordered [nz & sparse, nz & dense, zero & dense, zero & sparse], where nz
    /// means a non-zero column of T and sparse a row of T with at most two non-zeros (lags
    /// of other states, AR(1) exogenous processes): the non-zero columns come first, and the
    /// dense rows form one block, so that T*M*T' is two gemms over the dense rows plus copies
    /// for the sparse ones (see [`transition_cov`](Self::transition_cov)).
    pub fn set_system(&mut self, t: &Mat, r: &Mat, q: &Mat) {
        let mm = self.mm;
        assert_eq!((t.nrows(), t.ncols()), (mm, mm), "set_system: T shape");
        assert_eq!((r.nrows(), r.ncols()), (mm, self.rr), "set_system: R shape");
        assert_eq!((q.nrows(), q.ncols()), (self.rr, self.rr), "set_system: Q shape");
        // non-zero columns and row counts of T (column-major pass)
        let mut rowcnt = vec![0usize; mm];
        let mut colnz = vec![false; mm];
        for (j, col) in t.as_slice().chunks_exact(mm).enumerate() {
            for (i, &x) in col.iter().enumerate() {
                if x != 0.0 {
                    rowcnt[i] += 1;
                    colnz[j] = true;
                }
            }
        }
        let nzcol = |j: usize| colnz[j];
        let sparse = |i: usize| rowcnt[i] <= 2;
        self.perm.clear();
        self.perm.extend((0..mm).filter(|&j| nzcol(j) && sparse(j)));
        self.d0 = self.perm.len();
        self.perm.extend((0..mm).filter(|&j| nzcol(j) && !sparse(j)));
        self.nnz = self.perm.len();
        self.perm.extend((0..mm).filter(|&j| !nzcol(j) && !sparse(j)));
        self.nd = self.perm.len() - self.d0;
        self.perm.extend((0..mm).filter(|&j| !nzcol(j) && sparse(j)));
        for (p, &o) in self.perm.iter().enumerate() {
            self.iperm[o] = p;
        }
        for (c, &oc) in self.perm.iter().enumerate() {
            for (rw, &or) in self.perm.iter().enumerate() {
                self.tp[c * mm + rw] = t[(or, oc)];
            }
        }
        self.sparse_rows.clear();
        for i in (0..self.d0).chain(self.d0 + self.nd..mm) {
            let mut e = [(0usize, 0.0f64); 2];
            let mut k = 0;
            for j in 0..self.nnz {
                let v = self.tp[j * mm + i];
                if v != 0.0 {
                    e[k] = (j, v);
                    k += 1;
                }
            }
            self.sparse_rows.push((i, k, e));
        }
        let rr = self.rr;
        for c in 0..rr {
            for (rw, &or) in self.perm.iter().enumerate() {
                self.rp[c * mm + rw] = r[(or, c)];
            }
        }
        linalg::gemm_raw(false, false, mm, rr, rr, 1.0, &self.rp, mm, q.as_slice(), rr, 0.0, &mut self.rq, mm);
        linalg::gemm_raw(false, true, mm, mm, rr, 1.0, &self.rq, mm, &self.rp, mm, 0.0, &mut self.qq, mm);
        self.qq_idx.clear();
        self.qq_idx.extend((0..mm).filter(|&i| (0..rr).any(|c| self.rp[c * mm + i] != 0.0)));
        let k = self.qq_idx.len();
        for (ci, &c) in self.qq_idx.iter().enumerate() {
            for (ri, &r) in self.qq_idx.iter().enumerate() {
                self.qqc[ci * k + ri] = self.qq[c * mm + r];
            }
        }
        if self.qvec_raw.is_some() {
            self.build_hetero();
        }
    }

    /// Sets the filter's initial state (original state order): a = a0, P(star) = pstar,
    /// Pinf = pinf (if given). Call after [`set_system`](Self::set_system).
    pub fn set_state(&mut self, a0: &[f64], pstar: &Mat, pinf: Option<&Mat>) {
        let mm = self.mm;
        for (p, &o) in self.perm.iter().enumerate() {
            self.a[p] = a0[o];
        }
        for (c, &oc) in self.perm.iter().enumerate() {
            for (rw, &or) in self.perm.iter().enumerate() {
                self.p[c * mm + rw] = pstar[(or, oc)];
            }
        }
        if let Some(pi) = pinf {
            for (c, &oc) in self.perm.iter().enumerate() {
                for (rw, &or) in self.perm.iter().enumerate() {
                    self.pinf[c * mm + rw] = pi[(or, oc)];
                }
            }
        }
    }

    pub(crate) fn save_state(&mut self) {
        self.a_save.copy_from_slice(&self.a);
        self.p_save.copy_from_slice(&self.p);
    }

    pub(crate) fn restore_state(&mut self) {
        self.a.copy_from_slice(&self.a_save);
        self.p.copy_from_slice(&self.p_save);
    }

    /// Current state mean after the last filter call (original state order).
    pub fn state(&self) -> Vec<f64> {
        (0..self.mm).map(|o| self.a[self.iperm[o]]).collect()
    }

    /// Current state covariance P / Pstar (original state order).
    pub fn cov(&self) -> Mat {
        let mm = self.mm;
        Mat::from_fn(mm, mm, |i, j| self.p[self.iperm[j] * mm + self.iperm[i]])
    }

    // ---- building blocks -------------------------------------------------------

    /// mfp <- permuted positions of the observed states.
    fn set_observed(&mut self, mf: &[usize]) {
        assert_eq!(mf.len(), self.pp, "mf length");
        self.mfp.clear();
        for &m in mf {
            self.mfp.push(self.iperm[m]);
        }
    }

    /// a <- T * a (only the leading nnz columns of the permuted T are non-zero).
    fn transition_a(&mut self) {
        let nnz = self.nnz;
        linalg::gemv_raw(false, self.mm, nnz, 1.0, &self.tp, self.mm, &self.a[..nnz], 0.0, &mut self.av);
        std::mem::swap(&mut self.a, &mut self.av);
    }

    /// M <- T M T' (+ QQ) in place (M = P or Pinf); reads only M's leading nnz x nnz block.
    ///
    /// W = T M and then T W' by gemm over the dense rows of T only (one block d0..d0+nd of
    /// the permuted states); a sparse row i of T, T(i,:) = c1 e_l1' + c2 e_l2', gives
    /// W(i,:) = c1 M(l1,:) + c2 M(l2,:) and the column (T M T')(:,i) = c1 W(:,l1) + c2 W(:,l2),
    /// the same values as the dense products (whose other terms are exact zeros) up to the
    /// rounding of the two-term sum.
    fn transition_cov(&mut self, which_pinf: bool, add_qq: bool) {
        let (mm, nnz) = (self.mm, self.nnz);
        let m = if which_pinf { &mut self.pinf } else { &mut self.p };
        if nnz == 0 {
            if add_qq {
                m.copy_from_slice(&self.qq);
            } else {
                m.fill(0.0);
            }
            return;
        }
        let (d0, nd) = (self.d0, self.nd);
        // W = T(:,1:nnz) * M(1:nnz,1:nnz): dense rows, then sparse rows
        if nd > 0 {
            linalg::gemm_raw(false, false, nd, nnz, nnz, 1.0, &self.tp[d0..], mm, m, mm, 0.0, &mut self.w[d0..], mm);
        }
        for &(i, k, e) in &self.sparse_rows {
            let ([(l0, c0), (l1, c1)], w) = (e, &mut self.w);
            match k {
                0 => (0..nnz).for_each(|j| w[j * mm + i] = 0.0),
                1 => (0..nnz).for_each(|j| w[j * mm + i] = c0 * m[j * mm + l0]),
                _ => (0..nnz).for_each(|j| w[j * mm + i] = c0 * m[j * mm + l0] + c1 * m[j * mm + l1]),
            }
        }
        // T*M*T': columns of the dense rows by gemm, then those of the sparse rows
        if nd > 0 {
            linalg::gemm_raw(
                false,
                true,
                mm,
                nd,
                nnz,
                1.0,
                &self.w,
                mm,
                &self.tp[d0..],
                mm,
                0.0,
                &mut m[d0 * mm..],
                mm,
            );
        }
        for &(i, k, [(l0, c0), (l1, c1)]) in &self.sparse_rows {
            let (mc, w) = (&mut m[i * mm..(i + 1) * mm], &self.w);
            let (w0, w1) = (&w[l0 * mm..(l0 + 1) * mm], &w[l1 * mm..(l1 + 1) * mm]);
            match k {
                0 => mc.fill(0.0),
                1 => mc.iter_mut().zip(w0).for_each(|(v, a)| *v = c0 * a),
                _ => mc.iter_mut().zip(w0).zip(w1).for_each(|((v, a), b)| *v = c0 * a + c1 * b),
            }
        }
        if add_qq {
            // + QQ, which is zero outside the rows and columns where R is non-zero
            let k = self.qq_idx.len();
            for (ci, &c) in self.qq_idx.iter().enumerate() {
                let mc = &mut m[c * mm..(c + 1) * mm];
                for (&r, q) in self.qq_idx.iter().zip(&self.qqc[ci * k..(ci + 1) * k]) {
                    mc[r] += q;
                }
            }
        }
    }

    /// M(1:nnz,1:nnz) += alpha * X(1:nnz,1:n) * Y(1:nnz,1:n)' for mm x n buffers X, Y
    /// (M = P or Pinf; the rest of M is about to be overwritten by `transition_cov`).
    fn block_update(&mut self, which_pinf: bool, x: Buf, y: Buf, n: usize, alpha: f64) {
        let (mm, nnz) = (self.mm, self.nnz);
        if n == 0 || nnz == 0 {
            return;
        }
        let pick = |b: Buf| -> &[f64] {
            match b {
                Buf::Pz => &self.pz,
                Buf::Pz2 => &self.pz2,
                Buf::K => &self.kk,
                Buf::K2 => &self.k2,
            }
        };
        let (xs, ys) = (pick(x), pick(y));
        let m = if which_pinf { &mut self.pinf } else { &mut self.p };
        linalg::gemm_raw(false, true, nnz, nnz, n, alpha, xs, mm, ys, mm, 1.0, m, mm);
    }

    /// rank(A, tol) > 0 for A = Pinf or Z*Pinf*Z' (n x n over the current zi).
    fn rank_positive(&mut self, which: RankOf, tol: f64) -> bool {
        let mm = self.mm;
        let n = match which {
            RankOf::Pinf => mm,
            RankOf::ZPinfZ(n) => n,
        };
        match which {
            RankOf::Pinf => self.svd_a[..mm * mm].copy_from_slice(&self.pinf),
            RankOf::ZPinfZ(n) => {
                for c in 0..n {
                    for r in 0..n {
                        self.svd_a[c * n + r] = self.pinf[self.zi[c] * mm + self.zi[r]];
                    }
                }
            }
        }
        let a = &mut self.svd_a[..n * n];
        let mut maxabs = 0.0f64;
        let mut fro = 0.0;
        for &x in a.iter() {
            if x.is_nan() {
                return true; // MATLAB's svd errors on NaN; keep going and let NaN propagate
            }
            maxabs = maxabs.max(x.abs());
            fro += x * x;
        }
        if maxabs > tol {
            return true;
        }
        if fro.sqrt() <= tol {
            return false;
        }
        linalg::singular_values_raw(n, n, a, n, &mut self.svd_s, &mut self.svd_w);
        self.svd_s[0] > tol
    }

    /// inv(F) for the n x n matrix in `self.f` into `self.finv` (MATLAB `inv`), the pivots and
    /// interchanges of its LU factorisation (`self.udiag`, `self.ipiv`, for the log
    /// determinant), and the reciprocal 1-norm condition number as far as the caller's test
    /// `rcond < tol` needs it. Returns (info, rcond): info > 0 if F is exactly singular
    /// (rcond = 0, as MATLAB).
    ///
    /// The F matrices are small (at most pp x pp), where LAPACK's call overhead dominates,
    /// so the inverse is computed inline by Gauss-Jordan elimination with dgetf2's partial
    /// pivoting (whose eliminations below the pivot are the LU factorisation's).
    ///
    /// MATLAB's `rcond` is LAPACK's dgecon estimate on the LU factors; the exact
    /// 1/(norm(F,1)*norm(inv(F),1)) can be smaller by a modest factor, which would flip the
    /// singularity test at points near the threshold. The estimate is, however, never below
    /// the exact value (dgecon's norm(inv(F),1) estimate is norm(inv(F)*w,1) for some w with
    /// norm(w,1) = 1). So: if the exact value, from the inverse we need anyway, is >= 2*tol,
    /// the estimate is too and the test is decided: that value is returned and dgecon
    /// skipped. Otherwise F is LU-factorised (dgetf2) and dgecon's estimate returned, exactly
    /// as MATLAB. The factor 2 covers the rounding of both computations.
    fn lu_rcond(&mut self, n: usize, tol: f64) -> (i32, f64) {
        let anorm = (0..n).map(|c| self.f[c * n..(c + 1) * n].iter().map(|x| x.abs()).sum::<f64>()).fold(0.0, f64::max);
        let x = &mut self.finv[..n * n];
        x.copy_from_slice(&self.f[..n * n]);
        let info = gauss_jordan_inverse(x, n, &mut self.ipiv[..n], &mut self.udiag[..n], &mut self.gj_scratch);
        if info > 0 {
            return (info, 0.0);
        }
        let inorm = (0..n).map(|c| x[c * n..(c + 1) * n].iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
        let exact = 1.0 / (anorm * inorm);
        if exact >= 2.0 * tol {
            return (0, exact);
        }
        let a = &mut self.lu[..n * n];
        a.copy_from_slice(&self.f[..n * n]);
        let info = lu_in_place(a, n, &mut self.ipiv[..n]);
        if info > 0 {
            return (info, 0.0);
        }
        for i in 0..n {
            self.udiag[i] = a[i * n + i];
        }
        let rc = linalg::gecon_raw(n, a, n, anorm, &mut self.gecon_w, &mut self.gecon_iw);
        (0, rc)
    }

    /// log|det(F)| from the LU pivots and interchanges of the last [`lu_rcond`](Self::lu_rcond),
    /// and whether det(F) is negative (MATLAB's `log(det(F))` is then complex; whether that
    /// matters depends on the period, see the callers).
    ///
    /// As MATLAB's `log(det(F))`, the determinant is the product of the pivots, so that it
    /// overflows to Inf or underflows to 0 where MATLAB's does (log det = +-Inf: the filter's
    /// LIK is then infinite and dsge_likelihood takes the univariate fallback).
    fn logdet_lu(&self, n: usize) -> (f64, bool) {
        let mut det = 1.0f64;
        let mut neg = false;
        for i in 0..n {
            det *= self.udiag[i];
            if self.ipiv[i] as usize != i + 1 {
                neg = !neg;
            }
        }
        if det < 0.0 {
            neg = !neg;
        }
        (det.abs().ln(), neg)
    }

    /// v' * finv * v for the first n entries of self.v.
    fn quad_finv(&self, n: usize) -> f64 {
        let mut s = 0.0;
        for c in 0..n {
            let col = &self.finv[c * n..(c + 1) * n];
            let acc: f64 = col.iter().zip(&self.v[..n]).map(|(f, v)| f * v).sum();
            s += self.v[c] * acc;
        }
        s
    }

    /// Sets zi/di for period t (all observables when `missing` is None); zi are permuted.
    fn set_index(&mut self, missing: Option<&MissingData>, t: usize) {
        self.di.clear();
        self.zi.clear();
        match missing {
            Some(m) => self.di.extend_from_slice(&m.aindex[t]),
            None => self.di.extend(0..self.pp),
        }
        for &d in &self.di {
            self.zi.push(self.mfp[d]);
        }
    }

    /// pz(:, j) <- M(:, zi[j]) for M = P (or Pinf)
    fn gather_cols(&mut self, from_pinf: bool, into_pz2: bool) {
        let mm = self.mm;
        let src = if from_pinf { &self.pinf } else { &self.p };
        let dst = if into_pz2 { &mut self.pz2 } else { &mut self.pz };
        for (j, &z) in self.zi.iter().enumerate() {
            dst[j * mm..(j + 1) * mm].copy_from_slice(&src[z * mm..(z + 1) * mm]);
        }
    }

    // ---- multivariate diffuse filter ---------------------------------------------

    /// Port of kalman/likelihood/kalman_filter_d.m (`missing = None`; rank test on Z*Pinf*Z')
    /// and missing_observations_kalman_filter_d.m (`missing = Some`; rank test on Pinf),
    /// periods `start..last` (0-based, inclusive). Updates a, Pstar, Pinf in place; per-period
    /// densities in `self.dlik`.
    #[allow(clippy::too_many_arguments)]
    pub fn diffuse_multivariate(
        &mut self,
        y: &Mat,
        missing: Option<&MissingData>,
        mf: &[usize],
        h: &HMatrix,
        start: usize,
        last: usize,
        kalman_tol: f64,
        diffuse_kalman_tol: f64,
        presample: usize,
    ) -> DiffuseResult {
        let mm = self.mm;
        self.set_observed(mf);
        self.dlik.clear();
        self.complex = false;
        let mut t = start;
        loop {
            if t > last {
                break;
            }
            let rank_pos = match missing {
                Some(_) => self.rank_positive(RankOf::Pinf, diffuse_kalman_tol),
                None => {
                    self.set_index(None, t);
                    self.rank_positive(RankOf::ZPinfZ(self.pp), diffuse_kalman_tol)
                }
            };
            if !rank_pos {
                break;
            }
            self.set_index(missing, t);
            self.hetero_period(t + 1);
            let n = self.di.len();
            self.dlik.push(0.0);
            let s = self.dlik.len() - 1;
            if n == 0 {
                self.transition_a();
                self.transition_cov(false, true);
                self.transition_cov(true, false);
                t += 1;
                continue;
            }
            for j in 0..n {
                self.v[j] = y[(self.di[j], t)] - self.a[self.zi[j]];
            }
            // Finf = Pinf(z,z)
            for c in 0..n {
                for r in 0..n {
                    self.f[c * n + r] = self.pinf[self.zi[c] * mm + self.zi[r]];
                }
            }
            let (_, rc) = self.lu_rcond(n, diffuse_kalman_tol);
            if rc < diffuse_kalman_tol {
                if !self.f[..n * n].iter().all(|x| x.abs() < diffuse_kalman_tol) {
                    return (f64::INFINITY, self.dlik.len());
                }
                // Fstar = Pstar(z,z) + H(d,d)
                self.fill_f_from_p(n, h);
                let (_, rc2) = self.lu_rcond(n, kalman_tol);
                if rc2 < kalman_tol {
                    return (f64::INFINITY, self.dlik.len());
                }
                let (ld, neg) = self.logdet_lu(n);
                self.complex |= neg && s >= presample;
                // Kstar = Pstar(:,z) * iFstar
                self.gather_cols(false, false);
                linalg::gemm_raw(false, false, mm, n, n, 1.0, &self.pz, mm, &self.finv, n, 0.0, &mut self.kk, mm);
                self.dlik[s] = ld + self.quad_finv(n) + n as f64 * L2PI;
                self.transition_cov(true, false); // Pinf = T*Pinf*T'
                // Pstar = T*(Pstar - Pstar(:,z)*Kstar')*T' + QQ
                self.block_update(false, Buf::Pz, Buf::K, n, -1.0);
                self.transition_cov(false, true);
                // a = T*(a + Kstar*v)
                linalg::gemv_raw(false, mm, n, 1.0, &self.kk, mm, &self.v, 1.0, &mut self.a);
                self.transition_a();
            } else {
                let (ld, neg) = self.logdet_lu(n);
                self.complex |= neg && s >= presample;
                self.dlik[s] = ld + n as f64 * L2PI;
                // Kinf = Pinf(:,z)*iFinf
                self.gather_cols(true, true); // pz2 = Pinf(:,z)
                linalg::gemm_raw(false, false, mm, n, n, 1.0, &self.pz2, mm, &self.finv, n, 0.0, &mut self.k2, mm);
                // Fstar = Pstar(z,z) + H
                self.fill_f_from_p(n, h);
                // Kstar = (Pstar(:,z) - Kinf*Fstar)*iFinf
                self.gather_cols(false, false); // pz = Pstar(:,z)
                self.w[..mm * n].copy_from_slice(&self.pz[..mm * n]);
                linalg::gemm_raw(false, false, mm, n, n, -1.0, &self.k2, mm, &self.f, n, 1.0, &mut self.w, mm);
                linalg::gemm_raw(false, false, mm, n, n, 1.0, &self.w, mm, &self.finv, n, 0.0, &mut self.kk, mm);
                // Pstar = T*(Pstar - Pstar(:,z)*Kinf' - Pinf(:,z)*Kstar')*T' + QQ
                self.block_update(false, Buf::Pz, Buf::K2, n, -1.0);
                self.block_update(false, Buf::Pz2, Buf::K, n, -1.0);
                self.transition_cov(false, true);
                // Pinf = T*(Pinf - Pinf(:,z)*Kinf')*T'
                self.block_update(true, Buf::Pz2, Buf::K2, n, -1.0);
                self.transition_cov(true, false);
                // a = T*(a + Kinf*v)
                linalg::gemv_raw(false, mm, n, 1.0, &self.k2, mm, &self.v, 1.0, &mut self.a);
                self.transition_a();
            }
            t += 1;
        }
        if t == last + 1 {
            return (f64::NAN, self.dlik.len());
        }
        for x in &mut self.dlik {
            *x *= 0.5;
        }
        let dlik_sum: f64 = self.dlik.iter().skip(presample).sum();
        (dlik_sum, self.dlik.len())
    }

    /// f <- P(z,z) + H(d,d) for the current index.
    fn fill_f_from_p(&mut self, n: usize, h: &HMatrix) {
        let mm = self.mm;
        for (c, &zc) in self.zi.iter().enumerate() {
            let pc = &self.p[zc * mm..(zc + 1) * mm];
            for (fv, &zr) in self.f[c * n..(c + 1) * n].iter_mut().zip(&self.zi) {
                *fv = pc[zr];
            }
        }
        match h {
            HMatrix::Zero => {}
            HMatrix::Diag(d) => {
                for (i, &di) in self.di.iter().enumerate() {
                    self.f[i * n + i] += d[di];
                }
            }
            HMatrix::Full(hm) => {
                for c in 0..n {
                    for r in 0..n {
                        self.f[c * n + r] += hm[(self.di[r], self.di[c])];
                    }
                }
            }
        }
    }

    // ---- multivariate regular filter ------------------------------------------------

    /// Port of kalman/likelihood/kalman_filter.m (`missing = None`) and
    /// missing_observations_kalman_filter.m (`missing = Some`, which first symmetrises P and
    /// tests convergence of the gain only from `no_more_missing_observations` on), including
    /// the rcond / rescaled-F singularity test and the switch to kalman_filter_ss.m once the
    /// gain has converged (`max(abs(K(:)-oldK)) <= riccati_tol`).
    ///
    /// Periods `start..=last` (0-based); returns LIK (+Inf when F is badly conditioned, which
    /// dsge_likelihood treats as singular). Per-period values in `self.lik`.
    #[allow(clippy::too_many_arguments)]
    pub fn multivariate(
        &mut self,
        y: &Mat,
        missing: Option<&MissingData>,
        mf: &[usize],
        h: &HMatrix,
        start: usize,
        last: usize,
        kalman_tol: f64,
        riccati_tol: f64,
        rescale_prediction_error_covariance: bool,
        presample: usize,
        diffuse_periods: usize,
    ) -> f64 {
        let (mm, pp) = (self.mm, self.pp);
        self.set_observed(mf);
        self.steady_state_start = None;
        self.lik.clear();
        self.complex = false;
        // MATLAB sums lik(1+presample-diffuse_periods:end): a complex log det before that is dropped
        let first_summed = presample.saturating_sub(diffuse_periods);
        if missing.is_some() {
            // P = tril(P) + transpose(tril(P,-1))
            for c in 0..mm {
                for r in 0..c {
                    self.p[c * mm + r] = self.p[r * mm + c];
                }
            }
        }
        let nmmo = missing.map_or(0, |m| m.no_more_missing_observations);
        let mut first_k = true; // oldK = Inf
        let mut notsteady = true;
        let mut f_singular = true;
        let mut log_df = 0.0;
        let mut log_df_neg = false;
        let mut t = start;
        let hetero = self.qvec_raw.is_some(); // no steady-state switch with Qvec
        while notsteady && t <= last {
            self.set_index(missing, t);
            self.hetero_period(t + 1);
            let n = self.di.len();
            if n == 0 {
                self.lik.push(0.0);
                self.transition_a();
                self.transition_cov(false, true);
                t += 1;
                continue;
            }
            for j in 0..n {
                self.v[j] = y[(self.di[j], t)] - self.a[self.zi[j]];
            }
            self.fill_f_from_p(n, h);
            let mut rescale = rescale_prediction_error_covariance;
            let mut bad = false;
            let (_, rc) = if rescale { (0, 0.0) } else { self.lu_rcond(n, kalman_tol) };
            if rescale || rc < kalman_tol {
                let mut diag_small = false;
                for i in 0..n {
                    let d = self.f[i * n + i];
                    if d < kalman_tol {
                        diag_small = true;
                    }
                    self.sig[i] = d.sqrt();
                }
                if diag_small {
                    bad = true;
                } else {
                    for c in 0..n {
                        for r in 0..n {
                            self.f2[c * n + r] = self.f[c * n + r] / (self.sig[r] * self.sig[c]);
                        }
                    }
                    std::mem::swap(&mut self.f, &mut self.f2); // f = scaled F, f2 = F
                    let (_, rcs) = self.lu_rcond(n, kalman_tol);
                    if rcs < kalman_tol {
                        bad = true;
                    } else {
                        rescale = true;
                    }
                    std::mem::swap(&mut self.f, &mut self.f2);
                }
            }
            if bad {
                return f64::INFINITY;
            }
            f_singular = false;
            if rescale {
                // lu holds the factors of F./(sig*sig')
                let (ld, neg) = self.logdet_lu(n);
                log_df = ld + 2.0 * self.sig[..n].iter().map(|s| s.ln()).sum::<f64>();
                log_df_neg = neg;
                for c in 0..n {
                    for r in 0..n {
                        self.finv[c * n + r] /= self.sig[r] * self.sig[c];
                    }
                }
            } else {
                (log_df, log_df_neg) = self.logdet_lu(n);
            }
            self.complex |= log_df_neg && self.lik.len() >= first_summed;
            self.lik.push(log_df + self.quad_finv(n) + n as f64 * L2PI);
            // K = P(:,z)*iF
            self.gather_cols(false, false);
            linalg::gemm_raw(false, false, mm, n, n, 1.0, &self.pz, mm, &self.finv, n, 0.0, &mut self.kk, mm);
            // P = T*(P - K*P(z,:))*T' + QQ
            // pz2 = P(z,1:nnz) as an n x nnz matrix (the rows, as Dynare: P is not exactly
            // symmetric, and using the columns instead makes this near-singular recursion
            // drift); gathered column by column of P
            let nnz = self.nnz;
            for j in 0..nnz {
                let pj = &self.p[j * mm..(j + 1) * mm];
                for (x, &z) in self.pz2[j * n..(j + 1) * n].iter_mut().zip(&self.zi) {
                    *x = pj[z];
                }
            }
            if nnz > 0 {
                // P(1:nnz,1:nnz) -= K(1:nnz,:) * pz2
                linalg::gemm_raw(false, false, nnz, nnz, n, -1.0, &self.kk, mm, &self.pz2, n, 1.0, &mut self.p, mm);
            }
            self.transition_cov(false, true);
            // a = T*(a + K*v)
            linalg::gemv_raw(false, mm, n, 1.0, &self.kk, mm, &self.v, 1.0, &mut self.a);
            self.transition_a();
            if (missing.is_none() || t >= nmmo) && !hetero {
                let len = mm * n;
                if first_k {
                    notsteady = true;
                    first_k = false;
                } else {
                    notsteady = max_abs_diff(&self.kk[..len], &self.oldk[..len]) > riccati_tol;
                }
                self.oldk[..len].copy_from_slice(&self.kk[..len]);
            }
            t += 1;
        }
        if f_singular {
            // MATLAB: error('The variance of the forecast error remains singular until the end of the sample')
            return f64::INFINITY;
        }
        for x in &mut self.lik {
            *x *= 0.5;
        }
        if t <= last {
            self.steady_state_start = Some(t);
            // kalman_filter_ss.m with the last K, iF, log_dF (all observables from here on)
            self.set_index(None, t);
            let mut idx = self.lik.len();
            // kalman_filter_ss.m keeps the last (possibly complex) log det
            self.complex |= log_df_neg && last - start >= first_summed;
            self.lik.resize(last - start + 1, 0.0);
            while t <= last {
                for j in 0..pp {
                    self.v[j] = y[(j, t)] - self.a[self.mfp[j]];
                }
                linalg::gemv_raw(false, mm, pp, 1.0, &self.kk, mm, &self.v, 1.0, &mut self.a);
                self.transition_a();
                let q = self.quad_finv(pp);
                self.lik[idx] = 0.5 * (q + log_df + pp as f64 * L2PI);
                idx += 1;
                t += 1;
            }
        }
        sum_from(&self.lik, presample, diffuse_periods)
    }

    // ---- univariate diffuse filter ----------------------------------------------------

    /// Port of kalman/likelihood/univariate_kalman_filter_d.m (H diagonal: `h_diag[i]`).
    /// Periods from `start`; updates a, Pstar, Pinf in place. Per-observable densities in
    /// `self.dlik_by_obs`, per-period sums in `self.dlik`.
    #[allow(clippy::too_many_arguments)]
    pub fn diffuse_univariate(
        &mut self,
        y: &Mat,
        missing: Option<&MissingData>,
        mf: &[usize],
        h_diag: &[f64],
        start: usize,
        last: usize,
        kalman_tol: f64,
        diffuse_kalman_tol: f64,
        presample: usize,
    ) -> DiffuseResult {
        let (mm, pp) = (self.mm, self.pp);
        self.set_observed(mf);
        self.dlik.clear();
        self.dlik_by_obs.clear();
        let mut new_rank = self.rank_positive(RankOf::Pinf, diffuse_kalman_tol);
        let mut t = start;
        while new_rank && t <= last {
            self.dlik.push(0.0);
            self.dlik_by_obs.resize(self.dlik_by_obs.len() + pp, 0.0);
            let s = self.dlik.len() - 1;
            self.set_index(missing, t);
            for i in 0..self.di.len() {
                let (d, z) = (self.di[i], self.zi[i]);
                let pe = y[(d, t)] - self.a[z];
                let fstar = self.p[z * mm + z] + h_diag[d];
                let finf = self.pinf[z * mm + z];
                // Kstar = Pstar(:,z), Kinf = Pinf(:,z)
                self.av.copy_from_slice(&self.p[z * mm..(z + 1) * mm]);
                if finf > diffuse_kalman_tol && new_rank {
                    self.an.copy_from_slice(&self.pinf[z * mm..(z + 1) * mm]);
                    let (kstar, kinf, kf) = (&self.av, &self.an, &mut self.kf);
                    // Kinf_Finf = Kinf/Finf
                    for (q, k) in kf.iter_mut().zip(kinf) {
                        *q = k / finf;
                    }
                    for r in 0..mm {
                        self.a[r] += kf[r] * pe;
                    }
                    let ratio = fstar / finf;
                    for c in 0..mm {
                        let kfc = kf[c];
                        let kfc_r = kfc * ratio;
                        let ksc = kstar[c];
                        let pc = &mut self.p[c * mm..(c + 1) * mm];
                        for (((v, ki), ks), kq) in pc.iter_mut().zip(kinf).zip(kstar).zip(kf.iter()) {
                            *v += ki * kfc_r - ks * kfc - kq * ksc;
                        }
                        let pic = &mut self.pinf[c * mm..(c + 1) * mm];
                        for (v, ki) in pic.iter_mut().zip(kinf) {
                            *v -= ki * kfc;
                        }
                    }
                    let l = finf.ln() + L2PI;
                    self.dlik_by_obs[s * pp + d] = l;
                    self.dlik[s] += l;
                } else if fstar > kalman_tol {
                    let l = fstar.ln() + pe * pe / fstar + L2PI;
                    self.dlik_by_obs[s * pp + d] = l;
                    self.dlik[s] += l;
                    let kstar = &self.av;
                    for r in 0..mm {
                        self.a[r] += kstar[r] * (pe / fstar);
                    }
                    for c in 0..mm {
                        let kc = kstar[c] / fstar;
                        let pc = &mut self.p[c * mm..(c + 1) * mm];
                        for (v, ks) in pc.iter_mut().zip(kstar) {
                            *v -= ks * kc;
                        }
                    }
                } else if fstar < 0.0 || finf < 0.0 {
                    return (f64::NAN, self.dlik.len());
                }
            }
            self.transition_a();
            self.hetero_period(t + 1);
            self.transition_cov(false, true);
            self.transition_cov(true, false);
            if new_rank {
                self.set_index(None, t);
                new_rank = self.rank_positive(RankOf::ZPinfZ(pp), diffuse_kalman_tol);
            }
            t += 1;
        }
        if t > last {
            return (f64::NAN, self.dlik.len());
        }
        for x in &mut self.dlik {
            *x *= 0.5;
        }
        for x in &mut self.dlik_by_obs {
            *x *= 0.5;
        }
        let d: f64 = self.dlik.iter().skip(presample).sum();
        (d, self.dlik.len())
    }

    // ---- univariate regular filter -----------------------------------------------------

    /// Port of kalman/likelihood/univariate_kalman_filter.m (H diagonal: `h_diag[i]`) with the
    /// switch to univariate_kalman_filter_ss.m once the gains have converged. Periods
    /// `start..=last`; returns LIK (NaN on a negative variance). Per-observable densities in
    /// `self.lik_by_obs`, per-period sums in `self.lik`.
    #[allow(clippy::too_many_arguments)]
    pub fn univariate(
        &mut self,
        y: &Mat,
        missing: Option<&MissingData>,
        mf: &[usize],
        h_diag: &[f64],
        start: usize,
        last: usize,
        kalman_tol: f64,
        riccati_tol: f64,
        presample: usize,
        diffuse_periods: usize,
    ) -> f64 {
        let (mm, pp) = (self.mm, self.pp);
        self.set_observed(mf);
        let smpl = (last + 1).saturating_sub(start);
        self.steady_state_start = None;
        self.lik_by_obs.clear();
        self.lik_by_obs.resize(smpl * pp, 0.0);
        self.lik.clear();
        self.lik.resize(smpl, 0.0);
        let nmmo = missing.map_or(0, |m| m.no_more_missing_observations);
        self.kk[..mm * pp].fill(f64::NAN); // K = NaN(mm,pp)
        let mut first_k = true; // oldK = Inf
        let mut notsteady = true;
        let mut t = start;
        let hetero = self.qvec_raw.is_some();
        while notsteady && t <= last {
            let s = t - start;
            self.set_index(missing, t);
            self.hetero_period(t + 1);
            let nobs = self.di.len();
            // Dynare 7.2: P = (P+P')/2 at the start of each period
            symmetrize(&mut self.p, mm);
            // sequential updates, accumulated: pz(:,j) = PZ_j, k2(:,j) = K_j for used j.
            // Dynare 7.2 symmetrizes P after each update, P_i = P_{i-1} - (PZ_i K_i' + K_i PZ_i')/2,
            // which the accumulated corrections below reproduce.
            let mut nused = 0;
            for i in 0..nobs {
                let (d, z) = (self.di[i], self.zi[i]);
                // PZ = P(:,z) - sum_j (PZ_j * K_j(z) + K_j * PZ_j(z)) / 2
                {
                    let (head, tail) = self.pz.split_at_mut(nused * mm);
                    let col = &mut tail[..mm];
                    col.copy_from_slice(&self.p[z * mm..(z + 1) * mm]);
                    for j in 0..nused {
                        let kz = self.k2[j * mm + z];
                        let pj = &head[j * mm..(j + 1) * mm];
                        let kj = &self.k2[j * mm..(j + 1) * mm];
                        let pzz = pj[z];
                        if kz != 0.0 || pzz != 0.0 {
                            for r in 0..mm {
                                col[r] -= 0.5 * (pj[r] * kz + kj[r] * pzz);
                            }
                        }
                    }
                }
                let col = &self.pz[nused * mm..(nused + 1) * mm];
                let fi = col[z] + h_diag[d];
                if fi > kalman_tol {
                    let pe = y[(d, t)] - self.a[z];
                    let kcol = &mut self.k2[nused * mm..(nused + 1) * mm];
                    for r in 0..mm {
                        kcol[r] = col[r] / fi;
                    }
                    if t >= nmmo {
                        self.kk[i * mm..(i + 1) * mm].copy_from_slice(kcol);
                    }
                    let l = fi.ln() + pe * pe / fi + L2PI;
                    self.lik_by_obs[s * pp + i] = l;
                    for r in 0..mm {
                        self.a[r] += kcol[r] * pe;
                    }
                    nused += 1;
                } else if fi < 0.0 {
                    return f64::NAN;
                }
            }
            // P = T*(P - sum (PZ_j K_j' + K_j PZ_j')/2)*T' + QQ ; a = T*a
            self.block_update(false, Buf::Pz, Buf::K2, nused, -0.5);
            self.block_update(false, Buf::K2, Buf::Pz, nused, -0.5);
            self.transition_cov(false, true);
            self.transition_a();
            if t >= nmmo && !hetero {
                let len = mm * pp;
                if first_k {
                    notsteady = true;
                    first_k = false;
                } else {
                    notsteady = max_abs_diff(&self.kk[..len], &self.oldk[..len]) > riccati_tol;
                }
                self.oldk[..len].copy_from_slice(&self.kk[..len]);
            }
            t += 1;
        }
        let done = t - start;
        for x in &mut self.lik_by_obs[..done * pp] {
            *x *= 0.5;
        }
        if t <= last {
            self.steady_state_start = Some(t);
            // univariate_kalman_filter_ss.m: constant gains computed once from P
            let mut nused = 0;
            for i in 0..pp {
                let z = self.mfp[i];
                {
                    let (head, tail) = self.pz.split_at_mut(nused * mm);
                    let col = &mut tail[..mm];
                    col.copy_from_slice(&self.p[z * mm..(z + 1) * mm]);
                    for j in 0..nused {
                        let kz = self.k2[j * mm + z];
                        if kz != 0.0 {
                            let pj = &head[j * mm..(j + 1) * mm];
                            for r in 0..mm {
                                col[r] -= pj[r] * kz;
                            }
                        }
                    }
                }
                let col = &self.pz[nused * mm..(nused + 1) * mm];
                let fi = col[z] + h_diag[i];
                self.used[i] = fi > kalman_tol;
                if self.used[i] {
                    self.fi[i] = fi;
                    let kcol = &mut self.k2[nused * mm..(nused + 1) * mm];
                    for r in 0..mm {
                        kcol[r] = col[r] / fi;
                    }
                    nused += 1;
                }
            }
            while t <= last {
                let s = t - start;
                let mut j = 0;
                for i in 0..pp {
                    if !self.used[i] {
                        continue;
                    }
                    let z = self.mfp[i];
                    let fi = self.fi[i];
                    let pe = y[(i, t)] - self.a[z];
                    let kcol = &self.k2[j * mm..(j + 1) * mm];
                    for r in 0..mm {
                        self.a[r] += kcol[r] * pe;
                    }
                    self.lik_by_obs[s * pp + i] = 0.5 * (fi.ln() + pe * pe / fi + L2PI);
                    j += 1;
                }
                self.transition_a();
                t += 1;
            }
        }
        for s in 0..smpl {
            self.lik[s] = self.lik_by_obs[s * pp..(s + 1) * pp].iter().sum();
        }
        let first = presample.saturating_sub(diffuse_periods);
        let mut tot = 0.0;
        for s in first..smpl {
            for x in &self.lik_by_obs[s * pp..(s + 1) * pp] {
                tot += x;
            }
        }
        tot
    }
}

#[derive(Clone, Copy)]
enum Buf {
    Pz,
    Pz2,
    K,
    K2,
}

#[derive(Clone, Copy)]
enum RankOf {
    Pinf,
    ZPinfZ(usize),
}

/// LU with partial pivoting of the column-major n x n matrix `a` in place (LAPACK dgetf2:
/// first maximal |pivot|, row interchanges over all columns, 1-based `ipiv`). Returns info
/// (> 0: first exactly zero pivot; the factorisation is completed as dgetf2 does).
fn lu_in_place(a: &mut [f64], n: usize, ipiv: &mut [c_int]) -> i32 {
    let mut info = 0;
    for k in 0..n {
        let col = &a[k * n..(k + 1) * n];
        let mut p = k;
        let mut best = col[k].abs();
        for (i, v) in col.iter().enumerate().skip(k + 1) {
            if v.abs() > best {
                best = v.abs();
                p = i;
            }
        }
        ipiv[k] = (p + 1) as c_int;
        if a[k * n + p] == 0.0 {
            if info == 0 {
                info = (k + 1) as i32;
            }
            continue;
        }
        if p != k {
            for c in a.chunks_exact_mut(n) {
                c.swap(k, p);
            }
        }
        let (left, right) = a.split_at_mut((k + 1) * n);
        let lk = &mut left[k * n + k..k * n + n];
        let rp = 1.0 / lk[0];
        let lk = &mut lk[1..];
        for x in lk.iter_mut() {
            *x *= rp;
        }
        let lk = &*lk;
        for c in right.chunks_exact_mut(n) {
            let akc = c[k];
            if akc != 0.0 {
                for (x, l) in c[k + 1..].iter_mut().zip(lk) {
                    *x -= l * akc;
                }
            }
        }
    }
    info
}

/// In-place inverse of the column-major n x n matrix `a` by Gauss-Jordan elimination with
/// partial pivoting (first maximal |pivot| among the rows not yet pivoted, as dgetf2). The
/// eliminations below the pivot are those of the LU factorisation, so `ipiv` (1-based) and
/// the pivots `udiag` are dgetf2's U diagonal and interchanges up to rounding. `m` is
/// scratch of length n. Returns info > 0 at the first exactly zero pivot (`a` is then
/// garbage).
fn gauss_jordan_inverse(a: &mut [f64], n: usize, ipiv: &mut [c_int], udiag: &mut [f64], m: &mut [f64]) -> i32 {
    for k in 0..n {
        let col = &a[k * n..(k + 1) * n];
        let mut p = k;
        let mut best = col[k].abs();
        for (i, v) in col.iter().enumerate().skip(k + 1) {
            if v.abs() > best {
                best = v.abs();
                p = i;
            }
        }
        ipiv[k] = (p + 1) as c_int;
        if col[p] == 0.0 {
            return (k + 1) as i32;
        }
        if p != k {
            for c in a.chunks_exact_mut(n) {
                c.swap(k, p);
            }
        }
        let piv = a[k * n + k];
        udiag[k] = piv;
        let r = 1.0 / piv;
        // multipliers m = A(:,k) with m(k) = 0; row k scaled by 1/piv into fr (fr(k) = 1/piv);
        // column k restarts from 0, so that every column j gets A(:,j) -= m * fr(j)
        let (m, fr) = m.split_at_mut(n);
        m.copy_from_slice(&a[k * n..(k + 1) * n]);
        m[k] = 0.0;
        for (j, f) in fr[..n].iter_mut().enumerate() {
            *f = a[j * n + k] * r;
        }
        fr[k] = r;
        a[k * n..(k + 1) * n].fill(0.0);
        let m = &*m;
        let mut cols = a.chunks_exact_mut(4 * n);
        let mut j = 0;
        for c4 in &mut cols {
            let (c01, c23) = c4.split_at_mut(2 * n);
            let (c0, c1) = c01.split_at_mut(n);
            let (c2, c3) = c23.split_at_mut(n);
            let (f0, f1, f2, f3) = (fr[j], fr[j + 1], fr[j + 2], fr[j + 3]);
            for ((((x0, x1), x2), x3), mi) in
                c0.iter_mut().zip(c1.iter_mut()).zip(c2.iter_mut()).zip(c3.iter_mut()).zip(m)
            {
                *x0 -= mi * f0;
                *x1 -= mi * f1;
                *x2 -= mi * f2;
                *x3 -= mi * f3;
            }
            j += 4;
        }
        for c in cols.into_remainder().chunks_exact_mut(n) {
            let f0 = fr[j];
            for (x0, mi) in c.iter_mut().zip(m) {
                *x0 -= mi * f0;
            }
            j += 1;
        }
        for (j, &f) in fr[..n].iter().enumerate() {
            a[j * n + k] = f;
        }
    }
    // undo the row interchanges as column interchanges, in reverse order
    for k in (0..n).rev() {
        let p = ipiv[k] as usize - 1;
        if p != k {
            for i in 0..n {
                a.swap(k * n + i, p * n + i);
            }
        }
    }
    0
}

/// MATLAB `max(abs(K(:)-oldK))`, which ignores NaN (NaN only if all entries are NaN).
/// M = (M+M')/2 for an n x n column-major matrix.
fn symmetrize(m: &mut [f64], n: usize) {
    for c in 0..n {
        for r in (c + 1)..n {
            let v = 0.5 * (m[c * n + r] + m[r * n + c]);
            m[c * n + r] = v;
            m[r * n + c] = v;
        }
    }
}

fn max_abs_diff(k: &[f64], old: &[f64]) -> f64 {
    // f64::max returns the non-NaN operand, like MATLAB's max; four independent running
    // maxima (max is order-independent) so that the loop is not one dependency chain
    let mut acc = [f64::NAN; 4];
    let (k4, old4) = (k.chunks_exact(4), old.chunks_exact(4));
    let (kr, oldr) = (k4.remainder(), old4.remainder());
    for (a, b) in k4.zip(old4) {
        for j in 0..4 {
            acc[j] = acc[j].max((a[j] - b[j]).abs());
        }
    }
    for (a, b) in kr.iter().zip(oldr) {
        acc[0] = acc[0].max((a - b).abs());
    }
    acc[0].max(acc[1]).max(acc[2].max(acc[3]))
}

/// `if presample > diffuse_periods, sum(lik(1+presample-diffuse_periods:end)) else sum(lik)`.
fn sum_from(lik: &[f64], presample: usize, diffuse_periods: usize) -> f64 {
    let first = presample.saturating_sub(diffuse_periods);
    lik.iter().skip(first).sum()
}
