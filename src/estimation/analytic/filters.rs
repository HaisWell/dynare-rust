// SPDX-License-Identifier: GPL-3.0-or-later
//! Kalman filters with analytic parameter derivatives: the `analytic_derivation` branches of
//! `kalman/likelihood/kalman_filter.m` (+ `computeDLIK.m`, `kalman_filter_ss.m`) and
//! `univariate_kalman_filter.m` (+ `univariate_computeDLIK.m`, `univariate_computeDstate.m`,
//! `univariate_kalman_filter_ss.m`), including the second derivatives of the `'full'`
//! analytic Hessian. Index-for-index ports: `k` derivative slices are `Vec<Mat>`, `(:,:,i,j)`
//! slices are stored at `i + j*k`, pairs `(i, j)`, `i <= j`, are the `jcount` columns in
//! Dynare's order (`for j=1:k, for i=1:j`).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)] // index-for-index port of the MATLAB loops

use super::AnalyticHessian;
use crate::linalg::{Mat, mul};

/// `analytic_deriv_info` of `dsge_likelihood.m`: derivatives of the state-space matrices
/// with respect to the `k` estimated parameters (restricted state space, `xparam1` order).
#[derive(Clone, Debug)]
pub struct DerivInfo {
    pub k: usize,
    /// `DT(:,:,j)`, mm x mm.
    pub dt: Vec<Mat>,
    /// `DYss`, mm x k (zero columns for stderr/corr parameters).
    pub dyss: Mat,
    /// `DOm(:,:,j)` (`Om = R Q R'`), mm x mm.
    pub dom: Vec<Mat>,
    /// `DH(:,:,j)`, pp x pp.
    pub dh: Vec<Mat>,
    /// `DP(:,:,j)`: derivative of the initial state covariance (zero unless lik_init = 1).
    pub dp: Vec<Mat>,
    pub second: Option<SecondDerivs>,
}

/// The second-order members of `analytic_deriv_info` (`analytic_Hessian = 'full'`).
#[derive(Clone, Debug)]
pub struct SecondDerivs {
    /// `D2T`: vec(mm x mm) x k(k+1)/2.
    pub d2t: Mat,
    /// `D2Yss(r,:,:)` for each state row `r` (k x k).
    pub d2yss: Vec<Mat>,
    /// `D2Om`: vech(mm x mm) x k(k+1)/2.
    pub d2om: Mat,
    /// `D2H(:,:,i,j)` at `i + j*k`, pp x pp.
    pub d2h: Vec<Mat>,
    /// `D2P`: vech(mm x mm) x k(k+1)/2.
    pub d2p: Mat,
}

/// `dyn_vech`: the upper triangle, column by column ((1,1), (1,2), (2,2), (1,3), ...).
pub fn vech(m: &Mat) -> Vec<f64> {
    let n = m.nrows();
    let mut v = Vec::with_capacity(n * (n + 1) / 2);
    for j in 0..n {
        for i in 0..=j {
            v.push(m[(i, j)]);
        }
    }
    v
}

/// `dyn_unvech`: the symmetric matrix of a `dyn_vech` vector.
pub fn unvech(v: &[f64]) -> Mat {
    let n = (((8 * v.len() + 1) as f64).sqrt() as usize - 1) / 2;
    let mut m = Mat::zeros(n, n);
    let mut c = 0;
    for j in 0..n {
        for i in 0..=j {
            m[(i, j)] = v[c];
            m[(j, i)] = v[c];
            c += 1;
        }
    }
    m
}

/// Column `c` of a `vec`-stored matrix (`reshape(D2T(:,c), size(T))`).
pub fn unvec(m: &Mat, c: usize, n: usize) -> Mat {
    Mat::from_column_slice(n, n, m.column(c).as_slice())
}

fn col_vech(m: &Mat, c: usize) -> Mat {
    unvech(m.column(c).as_slice())
}

fn set_col(m: &mut Mat, c: usize, v: &[f64]) {
    m.column_mut(c).copy_from_slice(v);
}

/// Output of a filter with derivatives.
#[derive(Clone, Debug)]
pub struct FilterOut {
    /// `LIK` (Inf: singular forecast-error variance, NaN: negative variance).
    pub lik: f64,
    /// `DLIK` (k).
    pub dlik: Vec<f64>,
    /// `dlikk` / `dlik` (smpl x k): per-period scores.
    pub dlikk: Mat,
    /// `Hess` as the filter returns it (`-Hess/2`, symmetrized for the multivariate filter).
    pub hess: Option<Mat>,
}

fn inverse(a: &Mat) -> Mat {
    crate::linalg::solve(a, &Mat::identity(a.nrows(), a.nrows()))
        .unwrap_or_else(|| Mat::from_element(a.nrows(), a.nrows(), f64::NAN))
}

/// MATLAB's `rcond` (LAPACK dgecon on the LU factors; 0 when exactly singular).
fn rcond(a: &Mat) -> f64 {
    let (lu, _, info) = crate::linalg::lu(a);
    if info > 0 {
        return 0.0;
    }
    crate::linalg::rcond_from_lu(&lu, crate::linalg::norm1(a))
}

/// `log(det(A))` through the LU factorization (NaN for a negative determinant).
fn log_det(a: &Mat) -> f64 {
    let (lu, ipiv, info) = crate::linalg::lu(a);
    if info > 0 {
        return f64::NEG_INFINITY;
    }
    let mut ld = 0.0;
    let mut sign = 1.0;
    for i in 0..a.nrows() {
        let d = lu[(i, i)];
        if d < 0.0 {
            sign = -sign;
        }
        ld += d.abs().ln();
        if ipiv[i] as usize != i + 1 {
            sign = -sign;
        }
    }
    if sign < 0.0 { f64::NAN } else { ld }
}

/// State of the multivariate derivative recursion (`computeDLIK.m`).
struct MvState {
    da: Mat,            // mm x k
    dk: Vec<Mat>,       // mm x pp
    df: Vec<Mat>,       // pp x pp
    d2a: Vec<Vec<f64>>, // (i,j) at i + j*k
    d2k: Vec<Mat>,
    d2f: Vec<Mat>,
}

/// Port of `computeDKalman` (Zflag = 0): DK, DF and the next-period DP (in place).
fn compute_dkalman(
    t: &Mat,
    info: &DerivInfo,
    p: &Mat,
    dp: &mut [Mat],
    z: &[usize],
    i_f: &Mat,
    kg: &Mat,
    st: &mut MvState,
) {
    let k = info.k;
    let pz_rows = Mat::from_fn(z.len(), p.ncols(), |i, j| p[(z[i], j)]); // P(Z,:)
    let tmp = p - mul(kg, false, &pz_rows, false);
    let pz_cols = Mat::from_fn(p.nrows(), z.len(), |i, j| p[(i, z[j])]); // P(:,Z)
    for ii in 0..k {
        let dpi = &dp[ii];
        let df = Mat::from_fn(z.len(), z.len(), |a, b| dpi[(z[a], z[b])]) + &info.dh[ii];
        let dif = -mul(&mul(i_f, false, &df, false), false, i_f, false);
        let dpz = Mat::from_fn(p.nrows(), z.len(), |a, b| dpi[(a, z[b])]);
        let dk = mul(&dpz, false, i_f, false) + mul(&pz_cols, false, &dif, false);
        let dpzr = Mat::from_fn(z.len(), p.ncols(), |a, b| dpi[(z[a], b)]);
        let dtmp = dpi - mul(&dk, false, &pz_rows, false) - mul(kg, false, &dpzr, false);
        let dti = &info.dt[ii];
        let new = mul(&mul(dti, false, &tmp, false), false, t, true)
            + mul(&mul(t, false, &dtmp, false), false, t, true)
            + mul(&mul(t, false, &tmp, false), false, dti, true)
            + &info.dom[ii];
        st.df[ii] = df;
        st.dk[ii] = dk;
        dp[ii] = new;
    }
}

/// Port of `computeD2Kalman` (Zflag = 0): D2K, D2F and the next-period D2P (in place);
/// `dp0` is the current-period DP.
fn compute_d2kalman(
    t: &Mat,
    info: &DerivInfo,
    sec: &SecondDerivs,
    p0: &Mat,
    dp0: &[Mat],
    d2p: &mut Mat,
    z: &[usize],
    i_f: &Mat,
    k0: &Mat,
    st: &mut MvState,
) {
    let k = info.k;
    let ns = p0.nrows();
    let pz_rows = Mat::from_fn(z.len(), ns, |i, j| p0[(z[i], j)]);
    let tmp = p0 - mul(k0, false, &pz_rows, false);
    let pz_cols = Mat::from_fn(ns, z.len(), |i, j| p0[(i, z[j])]);
    let sel_zz = |m: &Mat| Mat::from_fn(z.len(), z.len(), |a, b| m[(z[a], z[b])]);
    let sel_cz = |m: &Mat| Mat::from_fn(ns, z.len(), |a, b| m[(a, z[b])]);
    let sel_zr = |m: &Mat| Mat::from_fn(z.len(), ns, |a, b| m[(z[a], b)]);
    let mut jcount = 0;
    for ii in 0..k {
        let dai = &info.dt[ii];
        let dfi = sel_zz(&dp0[ii]) + &info.dh[ii];
        let difi = -mul(&mul(i_f, false, &dfi, false), false, i_f, false);
        let dki = &st.dk[ii];
        for jj in 0..=ii {
            let daj = &info.dt[jj];
            let dfj = sel_zz(&dp0[jj]) + &info.dh[jj];
            let difj = -mul(&mul(i_f, false, &dfj, false), false, i_f, false);
            let dkj = &st.dk[jj];
            let d2aij = unvec(&sec.d2t, jcount, ns);
            let d2pij = col_vech(d2p, jcount);
            let d2omij = col_vech(&sec.d2om, jcount);
            let d2fij = sel_zz(&d2pij) + &sec.d2h[jj + ii * k];
            let d2if = -mul(&mul(&difi, false, &dfj, false), false, i_f, false)
                - mul(&mul(i_f, false, &d2fij, false), false, i_f, false)
                - mul(&mul(i_f, false, &dfj, false), false, &difi, false);
            let d2kij = mul(&sel_cz(&d2pij), false, i_f, false)
                + mul(&pz_cols, false, &d2if, false)
                + mul(&sel_cz(&dp0[jj]), false, &difi, false)
                + mul(&sel_cz(&dp0[ii]), false, &difj, false);
            let d2kcp = mul(&d2kij, false, &pz_rows, false)
                + mul(k0, false, &sel_zr(&d2pij), false)
                + mul(dki, false, &sel_zr(&dp0[jj]), false)
                + mul(dkj, false, &sel_zr(&dp0[ii]), false);
            let dtmpi = &dp0[ii] - mul(dki, false, &pz_rows, false) - mul(k0, false, &sel_zr(&dp0[ii]), false);
            let dtmpj = &dp0[jj] - mul(dkj, false, &pz_rows, false) - mul(k0, false, &sel_zr(&dp0[jj]), false);
            let d2tmp = &d2pij - &d2kcp;
            let m3 = |a: &Mat, b: &Mat, c: &Mat| mul(&mul(a, false, b, false), false, c, true);
            let d2atmpa = m3(&d2aij, &tmp, t)
                + m3(t, &d2tmp, t)
                + m3(t, &tmp, &d2aij)
                + m3(dai, &dtmpj, t)
                + m3(daj, &dtmpi, t)
                + m3(t, &dtmpj, dai)
                + m3(t, &dtmpi, daj)
                + m3(dai, &tmp, daj)
                + m3(daj, &tmp, dai);
            st.d2k[ii + jj * k] = d2kij.clone();
            set_col(d2p, jcount, &vech(&(d2atmpa + d2omij)));
            st.d2f[ii + jj * k] = d2fij.clone();
            st.d2k[jj + ii * k] = d2kij;
            st.d2f[jj + ii * k] = d2fij;
            jcount += 1;
        }
    }
}

/// `getHesst_ij` of computeDLIK.m.
fn hesst_ij(
    e: &Mat,
    dei: &Mat,
    dej: &Mat,
    d2eij: &Mat,
    i_s: &Mat,
    disi: &Mat,
    disj: &Mat,
    d2isij: &Mat,
    dsj: &Mat,
    d2sij: &Mat,
) -> f64 {
    let tr = (mul(disi, false, dsj, false) + mul(i_s, false, d2sij, false)).trace();
    let q = |a: &Mat, m: &Mat, b: &Mat| mul(&mul(a, true, m, false), false, b, false)[(0, 0)];
    tr + q(e, d2isij, e) + 2.0 * (q(dei, disj, e) + q(dei, i_s, dej) + q(e, disi, dej) + q(e, i_s, d2eij))
}

/// Port of `computeDLIK.m` (Zflag = 0). Returns (DLIKt, Hesst); updates Da (and D2a).
fn compute_dlik(
    info: &DerivInfo,
    tmp: &Mat,
    z: &[usize],
    v: &Mat,
    t: &Mat,
    kg: &Mat,
    i_f: &Mat,
    full: bool,
    st: &mut MvState,
) -> (Vec<f64>, Mat) {
    let k = info.k;
    let pp = z.len();
    let mm = t.nrows();
    let dv = Mat::from_fn(pp, k, |a, ii| -st.da[(z[a], ii)] - info.dyss[(z[a], ii)]);
    let mut hesst = Mat::zeros(k, k);
    let mut dlik = vec![0.0; k];
    let mut dtmp = Mat::zeros(mm, k);
    let mut jcount = 0;
    let col = |m: &Mat, c: usize| m.columns(c, 1).into_owned();
    for ii in 0..k {
        let dki = &st.dk[ii];
        let dvi = col(&dv, ii);
        let d = col(&st.da, ii) + mul(dki, false, v, false) + mul(kg, false, &dvi, false);
        dtmp.set_column(ii, &d.column(0));
        if full {
            let sec = info.second.as_ref().expect("second derivatives");
            let difi = -mul(&mul(i_f, false, &st.df[ii], false), false, i_f, false);
            for jj in 0..=ii {
                let dfj = &st.df[jj];
                let difj = -mul(&mul(i_f, false, dfj, false), false, i_f, false);
                let dkj = &st.dk[jj];
                let d2kij = &st.d2k[jj + ii * k];
                let d2fij = &st.d2f[jj + ii * k];
                let d2ifij = -mul(&mul(&difi, false, dfj, false), false, i_f, false)
                    - mul(&mul(i_f, false, d2fij, false), false, i_f, false)
                    - mul(&mul(i_f, false, dfj, false), false, &difi, false);
                let d2a_old = &st.d2a[jj + ii * k];
                let d2vij = Mat::from_fn(pp, 1, |a, _| -sec.d2yss[z[a]][(jj, ii)] - d2a_old[z[a]]);
                let dvj = col(&dv, jj);
                let d2a_col = Mat::from_column_slice(mm, 1, d2a_old);
                let d2tmpij = &d2a_col
                    + mul(d2kij, false, v, false)
                    + mul(dkj, false, &dvi, false)
                    + mul(dki, false, &dvj, false)
                    + mul(kg, false, &d2vij, false);
                let new = mul(&unvec(&sec.d2t, jcount, mm), false, tmp, false)
                    + mul(&info.dt[jj], false, &col(&dtmp, ii), false)
                    + mul(&info.dt[ii], false, &col(&dtmp, jj), false)
                    + mul(t, false, &d2tmpij, false);
                let nv: Vec<f64> = new.column(0).iter().copied().collect();
                st.d2a[jj + ii * k] = nv.clone();
                st.d2a[ii + jj * k] = nv;
                hesst[(ii, jj)] = hesst_ij(v, &dvi, &dvj, &d2vij, i_f, &difi, &difj, &d2ifij, dfj, d2fij);
                jcount += 1;
            }
        }
        let na = mul(&info.dt[ii], false, tmp, false) + mul(t, false, &col(&dtmp, ii), false);
        st.da.set_column(ii, &na.column(0));
        let idf = mul(i_f, false, &st.df[ii], false);
        let q = |a: &Mat, m: &Mat, b: &Mat| mul(&mul(a, true, m, false), false, b, false)[(0, 0)];
        dlik[ii] = idf.trace() + 2.0 * q(&dvi, i_f, v) - q(v, &mul(&idf, false, i_f, false), v);
    }
    if !full {
        // 2*Dv'*iF*Dv + vec(DF)' kron(iF,iF) vec(DF) = trace(DF_i iF DF_j iF)
        let a = mul(&mul(&dv, true, i_f, false), false, &dv, false) * 2.0;
        let g: Vec<Mat> = st.df.iter().map(|d| mul(i_f, false, d, false)).collect();
        for i in 0..k {
            for j in 0..k {
                hesst[(i, j)] = a[(i, j)] + mul(&g[i], false, &g[j], false).trace();
            }
        }
    }
    (dlik, hesst)
}

/// Port of `kalman_filter.m` with `analytic_derivation` (Zflag = 0) and its switch to
/// `kalman_filter_ss.m`. `a`: a_0_given_tm1, `p`: Pstar, `qq = R Q R'`. Periods
/// `0..nobs`; `info.dp`/`d2p` are the initial derivatives (copied).
pub fn kalman_filter(
    y: &Mat,
    a0: &[f64],
    p0: &Mat,
    kalman_tol: f64,
    riccati_tol: f64,
    rescale0: bool,
    presample: usize,
    t: &Mat,
    qq: &Mat,
    h: &Mat,
    z: &[usize],
    info: &DerivInfo,
    hess_kind: AnalyticHessian,
) -> FilterOut {
    let (mm, pp, k) = (t.nrows(), z.len(), info.k);
    let smpl = y.ncols();
    let full = hess_kind == AnalyticHessian::Full;
    let asy = hess_kind == AnalyticHessian::Asymptotic;
    let mut a = Mat::from_column_slice(mm, 1, a0);
    let mut p = p0.clone();
    let mut likk = vec![0.0; smpl];
    let mut dlik = vec![0.0; k];
    let mut dlikk = Mat::zeros(smpl, k);
    let mut hess = Mat::zeros(k, k);
    let mut dp = info.dp.clone();
    let mut d2p = info.second.as_ref().map(|s| s.d2p.clone());
    let mut st = MvState {
        da: Mat::zeros(mm, k),
        dk: vec![Mat::zeros(mm, pp); k],
        df: vec![Mat::zeros(pp, pp); k],
        d2a: if full { vec![vec![0.0; mm]; k * k] } else { Vec::new() },
        d2k: if full { vec![Mat::zeros(mm, pp); k * k] } else { Vec::new() },
        d2f: if full { vec![Mat::zeros(pp, pp); k * k] } else { Vec::new() },
    };
    let fail = |lik: f64| FilterOut { lik, dlik: vec![0.0; k], dlikk: Mat::zeros(smpl, k), hess: None };
    let mut old_k: Option<Mat> = None;
    let mut notsteady = true;
    let mut tt = 0;
    let mut rescale = rescale0;
    let mut last_kif: Option<(Mat, Mat, f64)> = None;
    let mut s = 0;
    while notsteady && tt < smpl {
        s = tt;
        let v = Mat::from_fn(pp, 1, |i, _| y[(i, tt)] - a[(z[i], 0)]);
        let f = Mat::from_fn(pp, pp, |i, j| p[(z[i], z[j])]) + h;
        let sig: Vec<f64> = (0..pp).map(|i| f[(i, i)].sqrt()).collect();
        let scaled = || Mat::from_fn(pp, pp, |i, j| f[(i, j)] / (sig[i] * sig[j]));
        let mut bad = false;
        if rescale {
            if (0..pp).any(|i| f[(i, i)] < kalman_tol) || rcond(&scaled()) < kalman_tol {
                bad = true;
            }
        } else if rcond(&f) < kalman_tol {
            if (0..pp).any(|i| f[(i, i)] < kalman_tol) || rcond(&scaled()) < kalman_tol {
                bad = true;
            } else {
                rescale = true;
            }
        }
        if bad {
            return fail(f64::INFINITY);
        }
        let (log_df, i_f) = if rescale {
            let fs = scaled();
            let ld = log_det(&fs) + 2.0 * sig.iter().map(|v| v.ln()).sum::<f64>();
            let ifs = inverse(&fs);
            rescale = rescale0;
            (ld, Mat::from_fn(pp, pp, |i, j| ifs[(i, j)] / (sig[i] * sig[j])))
        } else {
            (log_det(&f), inverse(&f))
        };
        likk[s] = log_df + mul(&mul(&v, true, &i_f, false), false, &v, false)[(0, 0)];
        let pzc = Mat::from_fn(mm, pp, |i, j| p[(i, z[j])]);
        let kg = mul(&pzc, false, &i_f, false);
        let pzr = Mat::from_fn(pp, mm, |i, j| p[(z[i], j)]);
        let ptmp = mul(&mul(t, false, &(&p - mul(&kg, false, &pzr, false)), false), false, t, true) + qq;
        let tmp = &a + mul(&kg, false, &v, false);
        // computeDLIK (notsteady = 1)
        let dp_old = dp.clone();
        compute_dkalman(t, info, &p, &mut dp, z, &i_f, &kg, &mut st);
        if full {
            compute_d2kalman(
                t,
                info,
                info.second.as_ref().unwrap(),
                &p,
                &dp_old,
                d2p.as_mut().unwrap(),
                z,
                &i_f,
                &kg,
                &mut st,
            );
        }
        let (dlikt, hesst) = compute_dlik(info, &tmp, z, &v, t, &kg, &i_f, full, &mut st);
        if tt >= presample {
            for i in 0..k {
                dlik[i] += dlikt[i];
            }
            if full || asy {
                hess += &hesst;
            }
        }
        for i in 0..k {
            dlikk[(s, i)] = dlikt[i];
        }
        a = mul(t, false, &tmp, false);
        p = ptmp;
        notsteady = match &old_k {
            None => true,
            Some(ok) => (&kg - ok).amax() > riccati_tol,
        };
        old_k = Some(kg.clone());
        last_kif = Some((kg, i_f, log_df));
        tt += 1;
    }
    let pi2 = (2.0 * std::f64::consts::PI).ln();
    for v in likk.iter_mut().take(s + 1) {
        *v = 0.5 * (*v + pp as f64 * pi2);
    }
    for v in &mut dlik {
        *v /= 2.0;
    }
    dlikk /= 2.0;
    if full || asy {
        if !asy {
            for i in 0..k {
                for j in 0..i {
                    hess[(j, i)] += hess[(i, j)];
                }
            }
        }
        hess *= -0.5;
    }
    if tt < smpl {
        // kalman_filter_ss.m with derivatives (notsteady = 0: DK, DF, D2K, D2F frozen)
        let (kg, i_f, log_df) = last_kif.expect("at least one period");
        let mut dl = vec![0.0; k];
        let mut hs = Mat::zeros(k, k);
        let s0 = tt;
        while tt < smpl {
            let v = Mat::from_fn(pp, 1, |i, _| y[(i, tt)] - a[(z[i], 0)]);
            let tmp = &a + mul(&kg, false, &v, false);
            let (dlikt, hesst) = compute_dlik(info, &tmp, z, &v, t, &kg, &i_f, full, &mut st);
            for i in 0..k {
                dl[i] += dlikt[i];
                dlikk[(tt, i)] = dlikt[i] / 2.0;
            }
            if full || asy {
                hs += &hesst;
            }
            a = mul(t, false, &tmp, false);
            likk[tt] = mul(&mul(&v, true, &i_f, false), false, &v, false)[(0, 0)];
            tt += 1;
        }
        for v in likk.iter_mut().skip(s0) {
            *v = 0.5 * (*v + log_df + pp as f64 * pi2);
        }
        for i in 0..k {
            dlik[i] += dl[i] / 2.0;
        }
        if full || asy {
            if !asy {
                for i in 0..k {
                    for j in 0..i {
                        hs[(j, i)] += hs[(i, j)];
                    }
                }
            }
            hess += hs * -0.5;
        }
    }
    let lik = likk.iter().skip(presample).sum();
    FilterOut { lik, dlik, dlikk, hess: (full || asy).then_some(hess) }
}

// ---------------------------------------------------------------------------------------
// Univariate filter

/// The measurement equation of the univariate filter: observed state rows (`Zflag = 0`) or
/// the selection matrix of the expanded state (`Zflag = 1`).
pub enum Meas<'a> {
    Index(&'a [usize]),
    Matrix(&'a Mat),
}

impl Meas<'_> {
    fn row(&self, i: usize, mm: usize) -> Mat {
        match self {
            Meas::Index(z) => {
                let mut r = Mat::zeros(1, mm);
                r[(0, z[i])] = 1.0;
                r
            }
            Meas::Matrix(m) => m.rows(i, 1).into_owned(),
        }
    }
}

struct UvState {
    da: Mat,             // mm x k
    ddk: Vec<Mat>,       // per observable index: mm x k (DDK(:,indx,:))
    ddf: Mat,            // pp x k
    d2a: Vec<Vec<f64>>,  // (i,j) at i + j*k
    dd2k: Vec<Vec<Mat>>, // per observable: mm x (k*k) as Vec over (i + j*k) of mm-vectors -> stored as Mat mm x 1
    dd2f: Vec<Mat>,      // per observable: k x k
}

/// Port of `univariate_computeDLIK.m`. `zr`: the 1 x mm measurement row. Returns
/// (DLIK, Hesst); updates Da, DP (when notsteady), D2a, D2P.
fn univariate_compute_dlik(
    k: usize,
    indx: usize,
    zr: &Mat,
    v: f64,
    kg: &Mat,
    pz: &Mat,
    f: f64,
    dyss: &Mat,
    dp: &mut Vec<Mat>,
    dh_row: &[f64],
    notsteady: bool,
    full: bool,
    st: &mut UvState,
    d2yss: Option<&[Mat]>,
    d2h: Option<&Mat>,
    d2p: Option<&mut Mat>,
) -> (Vec<f64>, Mat) {
    let mm = kg.nrows();
    let zda = mul(zr, false, &st.da, false); // 1 x k
    let zdy = mul(zr, false, dyss, false);
    let dv: Vec<f64> = (0..k).map(|j| -zda[(0, j)] - zdy[(0, j)]).collect();
    let zrt = zr.transpose();
    let mut d2v = Mat::zeros(k, k);
    if full {
        let d2y = d2yss.unwrap();
        for j in 0..k {
            for i in 0..k {
                let mut s = 0.0;
                for r in 0..mm {
                    let zv = zr[(0, r)];
                    if zv != 0.0 {
                        s += zv * (st.d2a[i + j * k][r] + d2y[r][(i, j)]);
                    }
                }
                d2v[(i, j)] = -s;
            }
        }
    }
    let (dk, df, d2k, d2f) = if notsteady {
        let mut dfv = vec![0.0; k];
        let mut dkm = Mat::zeros(mm, k);
        let dpz: Vec<Mat> = dp.iter().map(|d| mul(d, false, &zrt, false)).collect();
        for j in 0..k {
            dfv[j] = mul(zr, false, &dpz[j], false)[(0, 0)] + dh_row[j];
            for r in 0..mm {
                dkm[(r, j)] = dpz[j][(r, 0)] / f - pz[(r, 0)] * dfv[j] / (f * f);
            }
        }
        let mut d2fm = Mat::zeros(k, k);
        let mut d2km: Vec<Mat> = Vec::new();
        if full {
            d2km = vec![Mat::zeros(mm, 1); k * k];
            let d2pm = d2p.as_deref().unwrap();
            let d2hm = d2h.unwrap();
            let mut jcount = 0;
            for j in 0..k {
                for i in 0..=j {
                    let tmp = col_vech(d2pm, jcount);
                    let tz = mul(&tmp, false, &zrt, false);
                    let v2 = mul(zr, false, &tz, false)[(0, 0)] + d2hm[(j, i)];
                    d2fm[(j, i)] = v2;
                    d2fm[(i, j)] = v2;
                    let mut c = Mat::zeros(mm, 1);
                    for r in 0..mm {
                        c[(r, 0)] = tz[(r, 0)] / f
                            - dpz[j][(r, 0)] * dfv[i] / (f * f)
                            - dpz[i][(r, 0)] * dfv[j] / (f * f)
                            - pz[(r, 0)] * v2 / (f * f)
                            + 2.0 * pz[(r, 0)] / f.powi(3) * dfv[i] * dfv[j];
                    }
                    d2km[j + i * k] = c.clone();
                    d2km[i + j * k] = c;
                    jcount += 1;
                }
            }
        }
        if full {
            st.dd2k[indx] = d2km.clone();
            st.dd2f[indx] = d2fm.clone();
        }
        st.ddk[indx] = dkm.clone();
        for j in 0..k {
            st.ddf[(indx, j)] = dfv[j];
        }
        (dkm, dfv, d2km, d2fm)
    } else {
        let dkm = st.ddk[indx].clone();
        let dfv: Vec<f64> = (0..k).map(|j| st.ddf[(indx, j)]).collect();
        let (d2km, d2fm) =
            if full { (st.dd2k[indx].clone(), st.dd2f[indx].clone()) } else { (Vec::new(), Mat::zeros(k, k)) };
        (dkm, dfv, d2km, d2fm)
    };
    let dlik: Vec<f64> = (0..k).map(|j| df[j] / f + 2.0 * dv[j] / f * v - v * v / (f * f) * df[j]).collect();
    let mut hesst = Mat::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            hesst[(i, j)] = if full {
                d2f[(i, j)] / f - df[i] * df[j] / (f * f) + 2.0 * d2v[(i, j)] / f * v + 2.0 * dv[i] * dv[j] / f
                    - 2.0 * df[i] * dv[j] * v / (f * f)
                    - d2f[(i, j)] * v * v / (f * f)
                    - 2.0 * v / (f * f) * dv[i] * df[j]
                    + 2.0 * v * v / f.powi(3) * df[i] * df[j]
            } else {
                df[i] * df[j] / (f * f) + 2.0 * dv[i] * dv[j] / f
            };
        }
    }
    // Da = Da + DK*v + K*Dv
    for j in 0..k {
        for r in 0..mm {
            st.da[(r, j)] += dk[(r, j)] * v + kg[(r, 0)] * dv[j];
        }
    }
    if full {
        for idx in 0..k * k {
            for r in 0..mm {
                st.d2a[idx][r] += d2k[idx][(r, 0)] * v;
            }
        }
        for j in 0..k {
            for i in 0..=j {
                let mut c = st.d2a[j + i * k].clone();
                for r in 0..mm {
                    c[r] += dk[(r, i)] * dv[j] + dk[(r, j)] * dv[i] + kg[(r, 0)] * d2v[(j, i)];
                }
                st.d2a[i + j * k] = c.clone();
                st.d2a[j + i * k] = c;
            }
        }
    }
    if notsteady {
        let kt = kg.transpose();
        let pzt = pz.clone();
        let dpz_old: Vec<Mat> = dp.iter().map(|d| mul(d, false, &zrt, false)).collect();
        if full {
            let d2pm = d2p.unwrap();
            let mut jcount = 0;
            for j in 0..k {
                for i in 0..=j {
                    let tmp = col_vech(d2pm, jcount);
                    let d2pz = mul(&tmp, false, &zrt, false);
                    let dki = dk.columns(i, 1).into_owned();
                    let dkj = dk.columns(j, 1).into_owned();
                    let nt = &tmp
                        - mul(&d2pz, false, &kt, false)
                        - mul(&dpz_old[j], false, &dki.transpose(), false)
                        - mul(&dpz_old[i], false, &dkj.transpose(), false)
                        - mul(&pzt, false, &d2k[j + i * k].transpose(), false);
                    set_col(d2pm, jcount, &vech(&nt));
                    jcount += 1;
                }
            }
        }
        for j in 0..k {
            let dkj = dk.columns(j, 1).into_owned();
            let n = &dp[j] - mul(&dpz_old[j], false, &kt, false) - mul(&pzt, false, &dkj.transpose(), false);
            dp[j] = n;
        }
    }
    (dlik, hesst)
}

/// Port of `univariate_computeDstate.m`.
fn univariate_compute_dstate(
    k: usize,
    a: &Mat,
    p: &Mat,
    t: &Mat,
    info_dt: &[Mat],
    dom: Option<&[Mat]>,
    notsteady: bool,
    st: &mut UvState,
    dp: &mut Vec<Mat>,
    full: bool,
    d2t: Option<&Mat>,
    d2om: Option<&Mat>,
    d2p: Option<&mut Mat>,
) {
    let mm = t.nrows();
    let da_old = st.da.clone();
    let mut dp1: Vec<Mat> = if notsteady { vec![Mat::zeros(mm, mm); k] } else { dp.clone() };
    for j in 0..k {
        let c = mul(t, false, &da_old.columns(j, 1).into_owned(), false) + mul(&info_dt[j], false, a, false);
        st.da.set_column(j, &c.column(0));
        if notsteady {
            dp1[j] = mul(&mul(t, false, &dp[j], false), false, t, true)
                + mul(&mul(&info_dt[j], false, p, false), false, t, true)
                + mul(&mul(t, false, p, false), false, &info_dt[j], true);
        }
    }
    if notsteady {
        let dom = dom.unwrap();
        for j in 0..k {
            dp1[j] += &dom[j];
        }
    }
    if full {
        let d2t = d2t.unwrap();
        let mut d2p = d2p;
        let mut jcount = 0;
        for j in 0..k {
            for i in 0..=j {
                let d2tm = unvec(d2t, jcount, mm);
                let c = mul(&info_dt[i], false, &da_old.columns(j, 1).into_owned(), false)
                    + mul(&info_dt[j], false, &da_old.columns(i, 1).into_owned(), false)
                    + mul(t, false, &Mat::from_column_slice(mm, 1, &st.d2a[j + i * k]), false)
                    + mul(&d2tm, false, a, false);
                let cv: Vec<f64> = c.column(0).iter().copied().collect();
                st.d2a[j + i * k] = cv.clone();
                st.d2a[i + j * k] = cv;
                if notsteady {
                    let d2pm = d2p.as_deref_mut().unwrap();
                    let tmp = col_vech(d2pm, jcount);
                    let m3 = |x: &Mat, y: &Mat, z: &Mat| mul(&mul(x, false, y, false), false, z, true);
                    let nt = m3(t, &tmp, t)
                        + m3(&info_dt[i], &dp[j], t)
                        + m3(t, &dp[j], &info_dt[i])
                        + m3(&info_dt[j], &dp[i], t)
                        + m3(t, &dp[i], &info_dt[j])
                        + m3(&info_dt[j], p, &info_dt[i])
                        + m3(&info_dt[i], p, &info_dt[j])
                        + m3(&d2tm, p, t)
                        + m3(t, p, &d2tm)
                        + col_vech(d2om.unwrap(), jcount);
                    set_col(d2pm, jcount, &vech(&nt));
                }
                jcount += 1;
            }
        }
    }
    *dp = dp1;
}

/// Port of `univariate_kalman_filter.m` with `analytic_derivation` and its switch to
/// `univariate_kalman_filter_ss.m`. `h1`: the diagonal of H (zeros for the expanded state),
/// `dh`: `DH` as pp x k (one row per observable), `d2h`: `D2H(j,:,:)` per observable.
/// `data_index[t]`: observed rows at t; `nmmo`: 0-based `no_more_missing_observations`.
pub fn univariate_kalman_filter(
    data_index: &[Vec<usize>],
    nmmo: usize,
    y: &Mat,
    a0: &[f64],
    p0: &Mat,
    kalman_tol: f64,
    riccati_tol: f64,
    presample: usize,
    t: &Mat,
    qq: &Mat,
    h1: &[f64],
    meas: &Meas,
    info: &DerivInfo,
    dh: &Mat,
    d2h: Option<&[Mat]>,
    hess_kind: AnalyticHessian,
) -> FilterOut {
    let (mm, k) = (t.nrows(), info.k);
    let pp = y.nrows();
    let smpl = y.ncols();
    let full = hess_kind == AnalyticHessian::Full;
    let asy = hess_kind == AnalyticHessian::Asymptotic;
    let sec = info.second.as_ref();
    let mut a = Mat::from_column_slice(mm, 1, a0);
    let mut p = p0.clone();
    let mut lik = Mat::zeros(smpl, pp);
    let mut dlik = vec![0.0; k];
    let mut dlikm = Mat::zeros(smpl, k);
    let mut hess = Mat::zeros(k, k);
    let mut dp = info.dp.clone();
    let mut d2p = sec.map(|s| s.d2p.clone());
    let mut st = UvState {
        da: Mat::zeros(mm, k),
        ddk: vec![Mat::zeros(mm, k); pp],
        ddf: Mat::zeros(pp, k),
        d2a: if full { vec![vec![0.0; mm]; k * k] } else { Vec::new() },
        dd2k: if full { vec![vec![Mat::zeros(mm, 1); k * k]; pp] } else { Vec::new() },
        dd2f: if full { vec![Mat::zeros(k, k); pp] } else { Vec::new() },
    };
    let l2pi = (2.0 * std::f64::consts::PI).ln();
    let mut kmat = Mat::from_element(mm, pp, f64::NAN);
    let mut old_k: Option<Mat> = None;
    let mut notsteady = true;
    let mut tt = 0;
    let mut s = 0;
    let nan_out = |dlik: Vec<f64>, k: usize| FilterOut { lik: f64::NAN, dlik, dlikk: Mat::zeros(smpl, k), hess: None };
    while notsteady && tt < smpl {
        s = tt;
        let d_index = &data_index[tt];
        p = (&p + p.transpose()) * 0.5; // Dynare 7.2: P = (P+P')/2
        for (i, &di) in d_index.iter().enumerate() {
            let zr = meas.row(di, mm);
            let (pe, pz, fi) = match meas {
                Meas::Index(z) => {
                    (y[(di, tt)] - a[(z[di], 0)], p.columns(z[di], 1).into_owned(), p[(z[di], z[di])] + h1[di])
                }
                Meas::Matrix(_) => {
                    let pz = mul(&p, false, &zr.transpose(), false);
                    (
                        y[(di, tt)] - mul(&zr, false, &a, false)[(0, 0)],
                        pz.clone(),
                        mul(&zr, false, &pz, false)[(0, 0)] + h1[di],
                    )
                }
            };
            if fi > kalman_tol {
                let ki = &pz / fi;
                if tt >= nmmo {
                    kmat.set_column(i, &ki.column(0));
                }
                lik[(s, i)] = fi.ln() + pe * pe / fi + l2pi;
                let dh_row: Vec<f64> = dh.row(di).iter().copied().collect();
                let (dl, hs) = univariate_compute_dlik(
                    k,
                    i,
                    &zr,
                    pe,
                    &ki,
                    &pz,
                    fi,
                    &info.dyss,
                    &mut dp,
                    &dh_row,
                    notsteady,
                    full,
                    &mut st,
                    sec.map(|s| s.d2yss.as_slice()),
                    d2h.map(|d| &d[di]),
                    d2p.as_mut(),
                );
                if tt >= presample {
                    for j in 0..k {
                        dlik[j] += dl[j];
                    }
                    if full || asy {
                        hess += &hs;
                    }
                }
                for j in 0..k {
                    dlikm[(s, j)] += dl[j];
                }
                a += &ki * pe;
                p -= mul(&pz, false, &ki, true);
                p = (&p + p.transpose()) * 0.5; // Dynare 7.2
            } else if fi < 0.0 {
                return nan_out(dlik, k);
            }
        }
        univariate_compute_dstate(
            k,
            &a,
            &p,
            t,
            &info.dt,
            Some(&info.dom),
            notsteady,
            &mut st,
            &mut dp,
            full,
            sec.map(|s| &s.d2t),
            sec.map(|s| &s.d2om),
            d2p.as_mut(),
        );
        a = mul(t, false, &a, false);
        p = mul(&mul(t, false, &p, false), false, t, true) + qq;
        if tt >= nmmo {
            notsteady = match &old_k {
                None => true,
                Some(ok) => {
                    (0..mm * pp).map(|i| (kmat[i] - ok[i]).abs()).fold(f64::NEG_INFINITY, f64::max) > riccati_tol
                }
            };
            old_k = Some(kmat.clone());
        }
        tt += 1;
    }
    lik.rows_mut(0, s + 1).iter_mut().for_each(|v| *v *= 0.5);
    for v in &mut dlik {
        *v /= 2.0;
    }
    dlikm /= 2.0;
    if full || asy {
        hess *= -0.5;
    }
    if tt < smpl {
        // univariate_kalman_filter_ss.m with derivatives (notsteady = 0)
        let mut dl_ss = vec![0.0; k];
        let mut hs_ss = Mat::zeros(k, k);
        let s0 = tt;
        while tt < smpl {
            let ss = tt;
            let mut pp_ = p.clone();
            let mut dpp = dp.clone();
            let mut d2pp = d2p.clone();
            for i in 0..pp {
                let zr = meas.row(i, mm);
                let (pe, ppz, fi) = match meas {
                    Meas::Index(z) => {
                        (y[(i, tt)] - a[(z[i], 0)], pp_.columns(z[i], 1).into_owned(), pp_[(z[i], z[i])] + h1[i])
                    }
                    Meas::Matrix(_) => {
                        let pz = mul(&pp_, false, &zr.transpose(), false);
                        (
                            y[(i, tt)] - mul(&zr, false, &a, false)[(0, 0)],
                            pz.clone(),
                            mul(&zr, false, &pz, false)[(0, 0)] + h1[i],
                        )
                    }
                };
                if fi > kalman_tol {
                    let ki = &ppz / fi;
                    a += &ki * pe;
                    pp_ -= mul(&ppz, false, &ki, true);
                    lik[(ss, i)] = 0.5 * (fi.ln() + pe * pe / fi + l2pi);
                    let dh_row: Vec<f64> = dh.row(i).iter().copied().collect();
                    let (dl, hs) = univariate_compute_dlik(
                        k,
                        i,
                        &zr,
                        pe,
                        &ki,
                        &ppz,
                        fi,
                        &info.dyss,
                        &mut dpp,
                        &dh_row,
                        false,
                        full,
                        &mut st,
                        sec.map(|s| s.d2yss.as_slice()),
                        d2h.map(|d| &d[i]),
                        d2pp.as_mut(),
                    );
                    for j in 0..k {
                        dl_ss[j] += dl[j];
                        dlikm[(ss, j)] += dl[j] / 2.0;
                    }
                    if full || asy {
                        hs_ss += &hs;
                    }
                }
            }
            univariate_compute_dstate(
                k,
                &a,
                &p,
                t,
                &info.dt,
                None,
                false,
                &mut st,
                &mut dp,
                full,
                sec.map(|s| &s.d2t),
                None,
                None,
            );
            a = mul(t, false, &a, false);
            tt += 1;
        }
        let _ = s0;
        for j in 0..k {
            dlik[j] += dl_ss[j] / 2.0;
        }
        if full || asy {
            hess += hs_ss * -0.5;
        }
    }
    let total: f64 = lik.rows(presample, smpl - presample.min(smpl)).iter().sum();
    FilterOut { lik: total, dlik, dlikk: dlikm, hess: (full || asy).then_some(hess) }
}
