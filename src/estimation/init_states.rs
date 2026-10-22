// SPDX-License-Identifier: GPL-3.0-or-later
//! `estimation(estimate_initial_states_endogenous_prior)` (estimation completion): the joint
//! estimation of the initial states with the endogenous prior of Dynare 7.2.
//!
//! * dynare_estimation_init.m: one estimated parameter `init <var>` per state variable
//!   (`oo_.dr.state_var` order), [`EstimatedKind::InitState`], ordered sixth in `xparam1`
//!   (before the structural parameters), initial value the steady state, uniform prior on
//!   `[-1e8, 1e8]`; `lik_init = 2` with `Harvey_scale_factor = 0`.
//! * dsge_likelihood.m: `a_0|0` from the initial states (set_Kalman_starting_values.m),
//!   `Pstar = R Q R'`, and `fval = likelihood - lnpriorendoinitstate - (lnprior -
//!   lnpriorinitstate)` (init_state_endogenous_prior.m: the Gaussian log density of the initial
//!   states under the unconditional covariance of the states, on its non-singular subspace).
//! * set_init_state.m: projection of the initial states onto the support of that covariance.
//! * draw_init_state_from_smoother.m: the slice sampler's Metropolis step for the initial
//!   states, proposing from the smoothed initial state (`DsgeSmoother.m`'s `alphahat0` and
//!   `state_uncertainty0`, computed here by a Durbin-Koopman backward pass of the stationary
//!   filter, `lik_init = 1`) or, at initialization, from the unconditional distribution.
//!   The draws use the chain's own random stream (Dynare uses MATLAB's global stream).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)] // index-for-index port of the MATLAB loops

use anyhow::{Result, anyhow, bail};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::dsge::{DsgeModel, DsgePosterior, EvalFailure, Resolved};
use crate::estimation::{Bounds, Objective};
use crate::kalman::KalmanOptions;
use crate::linalg::{Mat, mul, sym_eig};
use crate::modfile::{EstimatedKind, EstimatedParam, ModFile, PriorShape};

/// dynare_estimation_init.m: `estim_params_.endo_init_vals` for every state variable (the
/// steady state of the `steady` command as initial value, uniform prior on [-1e8, 1e8]);
/// returns the mod file with these estimated parameters added.
pub fn augment(mf: &ModFile) -> Result<ModFile> {
    let dsge = DsgeModel::new(mf, None)?;
    let x0 = dsge.init_vector()?;
    let res = dsge.resolve(&x0).map_err(|e| {
        anyhow!("estimate_initial_states_endogenous_prior: no solution at the initial values (info {})", e.info)
    })?;
    let mut out = mf.clone();
    for &k in &res.dr.state_var {
        out.estimated_params.push(EstimatedParam {
            kind: EstimatedKind::InitState(k),
            init: Some(dsge.ys_calib[k]),
            lb: None,
            ub: None,
            prior: Some(PriorShape::Uniform),
            mean: f64::NAN,
            sd: f64::NAN,
            p3: Some(-1e8),
            p4: Some(1e8),
            jscale: None,
        });
    }
    Ok(out)
}

/// The estimated initial states of `dsge`: (position in `xparam1`, endogenous index).
pub fn positions(dsge: &DsgeModel) -> Vec<(usize, usize)> {
    dsge.estimated
        .iter()
        .enumerate()
        .filter_map(|(i, e)| if let EstimatedKind::InitState(k) = e.kind { Some((i, k)) } else { None })
        .collect()
}

/// What dsge_likelihood.m changes with the option.
pub struct KalmanInit {
    /// `Pstar = T*0*T' + R Q R'` (lik_init = 2, Harvey_scale_factor = 0).
    pub pstar: Mat,
    /// `a_0_given_tm1 = T a_0_given_tm0`.
    pub a_tm1: Vec<f64>,
    /// `lnpriorendoinitstate - lnpriorinitstate`, subtracted from `fval`.
    pub fval_shift: f64,
}

/// `a_0|0` of set_Kalman_starting_values.m (restricted state space): the initial states minus
/// the steady state (their logs with `loglinear`) at the state rows `mf0`.
fn a_tm0(dsge: &DsgeModel, res: &Resolved, x: &[f64]) -> Vec<f64> {
    let idx = &dsge.ss_index;
    let mut values = vec![0.0; dsge.model.n_endo()];
    for &(i, k) in &dsge.init_states {
        values[k] = x[i];
    }
    let mut a = vec![0.0; res.t.nrows()];
    for &p in &idx.mf0 {
        let s = res.dr.order_var[idx.restrict_var_list[p]];
        let ys = res.steady_state[s];
        a[p] = if dsge.loglinear { values[s].ln() - ys.ln() } else { values[s] - ys };
    }
    a
}

/// `[U, X] = svd(0.5*(P + P'))` of a symmetric positive semi-definite matrix: eigenvectors
/// sorted by decreasing |eigenvalue| (MATLAB's singular values), and those singular values.
fn sym_svd(p: &Mat) -> (Mat, Vec<f64>) {
    let s = (p + p.transpose()) * 0.5;
    let (w, v) = sym_eig(&s);
    let mut order: Vec<usize> = (0..w.len()).collect();
    order.sort_by(|&a, &b| w[b].abs().partial_cmp(&w[a].abs()).unwrap());
    let u = Mat::from_fn(v.nrows(), order.len(), |i, j| v[(i, order[j])]);
    (u, order.iter().map(|&j| w[j].abs()).collect())
}

/// `isp`: the leading singular values above `kalman_tol`, cut where `dd/dd(1) <= eps`.
fn isp_count(x: &[f64], kalman_tol: f64) -> usize {
    let isp: Vec<usize> = (0..x.len()).filter(|&i| x[i] > kalman_tol).collect();
    if isp.is_empty() {
        return 0;
    }
    let d1 = x[isp[0]];
    isp.iter().filter(|&&i| x[i] / d1 > f64::EPSILON).count()
}

/// Pstar(mf0, mf0) of the stationary filter (lyapunov_solver.m).
fn pstar_states(dsge: &DsgeModel, res: &Resolved, opts: &KalmanOptions) -> Result<Mat, EvalFailure> {
    let p = crate::kalman::lyapunov::lyapunov_solver_opts(&res.t, &res.r, &res.q, opts)
        .map_err(|_| EvalFailure { info: 50, penalty: 0.1 })?;
    let mf0 = &dsge.ss_index.mf0;
    Ok(Mat::from_fn(mf0.len(), mf0.len(), |i, j| p[(mf0[i], mf0[j])]))
}

/// dsge_likelihood.m's initial condition and `fval` correction (init_state_endogenous_prior.m).
pub fn kalman_init(
    dsge: &DsgeModel,
    res: &Resolved,
    x: &[f64],
    opts: &KalmanOptions,
) -> Result<KalmanInit, EvalFailure> {
    let a0 = a_tm0(dsge, res, x);
    let a_tm1 = (&res.t * crate::linalg::Vector::from_column_slice(&a0)).iter().copied().collect();
    let pstar = mul(&mul(&res.r, false, &res.q, false), false, &res.r, true);
    // init_state_endogenous_prior.m
    let ps = pstar_states(dsge, res, opts)?;
    let (up, xs) = sym_svd(&ps);
    let ns = xs.len();
    let k = isp_count(&xs, opts.kalman_tol);
    let mf0 = &dsge.ss_index.mf0;
    let am: Vec<f64> = mf0.iter().map(|&p| a0[p]).collect();
    let mut quad = 0.0;
    let mut log_ds = 0.0;
    for j in 0..k {
        let v: f64 = (0..ns).map(|i| up[(i, j)] * am[i]).sum();
        quad += v * v / xs[j];
        log_ds += xs[j].ln();
    }
    let lnprior_endo = -(log_ds + quad + ns as f64 * (2.0 * std::f64::consts::PI).ln()) / 2.0;
    let lnprior_init: f64 = dsge.init_states.iter().map(|&(i, _)| dsge.priors[i].logpdf(x[i])).sum();
    Ok(KalmanInit { pstar, a_tm1, fval_shift: lnprior_endo - lnprior_init })
}

/// Port of set_init_state.m: when the initial states (minus the steady state `ys0`) leave the
/// support of the unconditional covariance of the states, the states of largest weight in
/// its null space are set so that they lie in it. Returns (`xparam1`, `icheck`).
pub fn set_init_state(post: &DsgePosterior, x: &[f64], ys0: &[f64]) -> (Vec<f64>, bool) {
    let dsge = post.dsge;
    let opts = &post.data.kalman;
    let Ok(res) = dsge.resolve(x) else { return (x.to_vec(), false) };
    let Ok(ps) = pstar_states(dsge, &res, opts) else { return (x.to_vec(), false) };
    let (up, xs) = sym_svd(&ps);
    let ns = xs.len();
    let k = isp_count(&xs, opts.kalman_tol);
    if k == ns {
        return (x.to_vec(), false);
    }
    let isn: Vec<usize> = (k..ns).collect();
    let nn = isn.len();
    let upn = Mat::from_fn(ns, nn, |i, j| up[(i, isn[j])]);
    let argmax = |c: usize| (0..ns).fold(0, |b, i| if upn[(i, c)].abs() > upn[(b, c)].abs() { i } else { b });
    let mut im: Vec<usize> = (0..nn).map(argmax).collect();
    let mut uniq = im.clone();
    uniq.sort_unstable();
    uniq.dedup();
    if uniq.len() < im.len() {
        let mut inn: Vec<Option<usize>> = vec![None; nn];
        for c in 0..nn {
            let t = argmax(c);
            if !inn.contains(&Some(t)) {
                inn[c] = Some(t);
            } else {
                let mut srt: Vec<usize> = (0..ns).collect();
                srt.sort_by(|&a, &b| upn[(a, c)].abs().partial_cmp(&upn[(b, c)].abs()).unwrap());
                let mut off = 0;
                loop {
                    off += 1;
                    let cand = srt[ns - 1 - off];
                    if !inn.contains(&Some(cand)) {
                        inn[c] = Some(cand);
                        break;
                    }
                }
            }
        }
        im = inn.into_iter().map(|v| v.unwrap()).collect();
    }
    // a = get_init_state(...) + dr.ys(order_var) - ys0(order_var), DR order
    let dr = &res.dr;
    let idx = &dsge.ss_index;
    let mut values = vec![0.0; dsge.model.n_endo()];
    for &(i, kk) in &dsge.init_states {
        values[kk] = x[i];
    }
    let ys = &res.steady_state;
    let n = dsge.model.n_endo();
    let mut a_decl = vec![0.0; n];
    for &s in &dr.state_var {
        a_decl[s] = if dsge.loglinear { values[s].ln() - ys[s].ln() } else { values[s] - ys[s] };
    }
    let mut a: Vec<f64> = (0..n).map(|j| a_decl[dr.order_var[j]]).collect();
    for j in 0..n {
        let v = dr.order_var[j];
        a[j] += if dsge.loglinear { ys[v].ln() - ys0[v].ln() } else { ys[v] - ys0[v] };
    }
    let rows: Vec<usize> = idx.mf0.iter().map(|&p| idx.restrict_var_list[p]).collect();
    let mycheck = (0..nn).map(|c| (0..ns).map(|i| upn[(i, c)] * a[rows[i]]).sum::<f64>().abs()).fold(0.0, f64::max);
    if mycheck <= opts.kalman_tol {
        return (x.to_vec(), false);
    }
    let nstates = dr.state_var.len();
    let inn: Vec<usize> = (0..nstates).filter(|i| !im.contains(i)).collect();
    // a1(im) = -UPN(im,:)' \ (UPN(in,:)' * a(in))
    let lhs = Mat::from_fn(nn, im.len(), |c, r| upn[(im[r], c)]);
    let rhs = Mat::from_fn(nn, 1, |c, _| inn.iter().map(|&i| upn[(i, c)] * a[rows[i]]).sum());
    let Some(sol) = crate::linalg::solve(&lhs, &rhs) else { return (x.to_vec(), true) };
    let mut a1 = a.clone();
    for (r, &i) in im.iter().enumerate() {
        a1[rows[i]] = -sol[(r, 0)];
    }
    let mut xo = x.to_vec();
    for &(i, kk) in &dsge.init_states {
        let dev = a1[dr.inv_order_var[kk]];
        xo[i] = if dsge.loglinear { dev.exp() * ys[kk] } else { dev + ys[kk] };
    }
    (xo, true)
}

/// DsgeSmoother.m with `lik_init = 1` (no initial state): the smoothed initial state
/// `alphahat0` and its variance `state_uncertainty0` on the state rows `mf0` (Durbin and
/// Koopman 2012, eqs. 4.35-4.43, period 0: `L0 = T`).
pub fn smoothed_initial_state(post: &DsgePosterior, x: &[f64]) -> Result<(Vec<f64>, Mat), EvalFailure> {
    let dsge = post.dsge;
    let opts = &post.data.kalman;
    let res = dsge.resolve(x)?;
    let idx = &dsge.ss_index;
    let p0 = crate::kalman::lyapunov::lyapunov_solver_opts(&res.t, &res.r, &res.q, opts)
        .map_err(|_| EvalFailure { info: 50, penalty: 0.1 })?;
    let (t, h) = (&res.t, &res.h);
    let mm = t.nrows();
    let qq = mul(&mul(&res.r, false, &res.q, false), false, &res.r, true);
    let mut y = post.data.y.clone();
    for (i, &v) in idx.mfys.iter().enumerate() {
        let c = if dsge.loglinear { res.steady_state[v].ln() } else { res.steady_state[v] };
        for j in 0..y.ncols() {
            y[(i, j)] -= c;
        }
    }
    if let Some(tr) =
        dsge.trend_addition(&res.params, post.data.first_obs, y.ncols(), post.data.prefilter_mean.is_some())
    {
        y -= tr;
    }
    let nobs = y.ncols();
    let mut a = crate::linalg::Vector::zeros(mm);
    let mut p = p0.clone();
    // forward pass: (observed rows, Z' iF v, Z' iF Z, L)
    let mut store: Vec<(crate::linalg::Vector, Mat, Mat)> = Vec::with_capacity(nobs);
    for tt in 0..nobs {
        let di: Vec<usize> = (0..y.nrows()).filter(|&i| !y[(i, tt)].is_nan()).collect();
        if di.is_empty() {
            store.push((crate::linalg::Vector::zeros(mm), Mat::zeros(mm, mm), t.clone()));
            a = t * &a;
            p = mul(&mul(t, false, &p, false), false, t, true) + &qq;
            continue;
        }
        let z: Vec<usize> = di.iter().map(|&i| idx.mf[i]).collect();
        let nd = di.len();
        let v = crate::linalg::Vector::from_fn(nd, |i, _| y[(di[i], tt)] - a[z[i]]);
        let f = Mat::from_fn(nd, nd, |i, j| p[(z[i], z[j])] + h[(di[i], di[j])]);
        let i_f = crate::linalg::solve(&f, &Mat::identity(nd, nd)).ok_or(EvalFailure { info: 50, penalty: 0.1 })?;
        let pz = Mat::from_fn(mm, nd, |i, j| p[(i, z[j])]);
        let k = mul(&pz, false, &i_f, false);
        let mut zmat = Mat::zeros(nd, mm);
        for (i, &zi) in z.iter().enumerate() {
            zmat[(i, zi)] = 1.0;
        }
        let ifv = &i_f * &v;
        let ztifv = zmat.transpose() * &ifv;
        let ztifz = mul(&mul(&zmat, true, &i_f, false), false, &zmat, false);
        let l = t * (Mat::identity(mm, mm) - &k * &zmat);
        let tmp = &a + &k * &v;
        let ptmp = &p - &k * pz.transpose();
        store.push((ztifv, ztifz, l));
        a = t * tmp;
        p = mul(&mul(t, false, &ptmp, false), false, t, true) + &qq;
    }
    let mut r = crate::linalg::Vector::zeros(mm);
    let mut nn = Mat::zeros(mm, mm);
    for (ztifv, ztifz, l) in store.iter().rev() {
        r = ztifv + l.transpose() * &r;
        nn = ztifz + mul(&mul(l, true, &nn, false), false, l, false);
    }
    let r0 = t.transpose() * &r;
    let n0 = mul(&mul(t, true, &nn, false), false, t, false);
    let alpha0 = &p0 * &r0;
    let v0 = &p0 - mul(&mul(&p0, false, &n0, false), false, &p0, false);
    let mf0 = &idx.mf0;
    Ok((mf0.iter().map(|&i| alpha0[i]).collect(), Mat::from_fn(mf0.len(), mf0.len(), |i, j| v0[(mf0[i], mf0[j])])))
}

/// The slice sampler's interface to the initial states ([`Objective::init_state_ops`]).
pub trait InitStateOps: Sync {
    /// `index_init_state`: positions of the initial states in `xparam1`.
    fn index_init_state(&self) -> Vec<usize>;
    /// slice_sampler.m's `get_steady_state` (None if the steady state fails).
    fn steady_state_at(&self, x: &[f64]) -> Option<Vec<f64>>;
    /// set_init_state.m.
    fn set_init_state(&self, x: &[f64], ys0: &[f64]) -> (Vec<f64>, bool);
    /// draw_init_state_from_smoother.m: (`xparam1`, `logpost0`, `neval`).
    fn draw_init_state(
        &self,
        init: bool,
        target: usize,
        x: &[f64],
        logpost0: f64,
        bounds: &Bounds,
        rng: &mut ChaCha8Rng,
    ) -> (Vec<f64>, f64, usize);
}

impl InitStateOps for DsgePosterior<'_> {
    fn index_init_state(&self) -> Vec<usize> {
        self.dsge.init_states.iter().map(|p| p.0).collect()
    }

    fn steady_state_at(&self, x: &[f64]) -> Option<Vec<f64>> {
        let (params, _) = self.dsge.set_all_parameters(x);
        self.dsge.steady_state(&params).ok()
    }

    fn set_init_state(&self, x: &[f64], ys0: &[f64]) -> (Vec<f64>, bool) {
        set_init_state(self, x, ys0)
    }

    fn draw_init_state(
        &self,
        init: bool,
        target: usize,
        x: &[f64],
        logpost0: f64,
        bounds: &Bounds,
        rng: &mut ChaCha8Rng,
    ) -> (Vec<f64>, f64, usize) {
        draw_init_state_from_smoother(self, init, target, x, logpost0, bounds, rng)
    }
}

/// Port of draw_init_state_from_smoother.m (`target_accepted` 1 at initialization, else the
/// caller's 5 or 1); `bounds`: the sampler's `thetaprior` (MCMC bounds).
pub fn draw_init_state_from_smoother(
    post: &DsgePosterior,
    init: bool,
    target: usize,
    x: &[f64],
    logpost0: f64,
    bounds: &Bounds,
    rng: &mut ChaCha8Rng,
) -> (Vec<f64>, f64, usize) {
    let dsge = post.dsge;
    let opts = &post.data.kalman;
    let target = if init { 1 } else { target };
    let mut neval = 0;
    let mut xparam1 = x.to_vec();
    let mut logpost0 = logpost0;
    let Some(ys0) = post.steady_state_at(x) else { return (xparam1, logpost0, 0) };
    let Ok(res) = dsge.resolve(x) else { return (xparam1, logpost0, 0) };
    let idx = &dsge.ss_index;
    let dr = &res.dr;
    let ys = &res.steady_state;
    let rows: Vec<usize> = idx.mf0.iter().map(|&p| idx.restrict_var_list[p]).collect();
    let n = dsge.model.n_endo();
    let ns = rows.len();
    // alphahat0 (DR order) and state_uncertainty0(rows, rows)
    let (mut alphahat0, su0) = if init {
        let Ok(ps) = pstar_states(dsge, &res, opts) else { return (xparam1, logpost0, 0) };
        (vec![0.0; n], ps)
    } else {
        match smoothed_initial_state(post, x) {
            Ok((a, v)) => {
                let mut full = vec![0.0; n];
                for (i, &r) in rows.iter().enumerate() {
                    full[r] = a[i];
                }
                (full, v)
            }
            Err(_) => return (xparam1, logpost0, 0),
        }
    };
    let (u, xs) = sym_svd(&su0);
    let is: Vec<usize> = (0..ns).filter(|&i| xs[i] > opts.kalman_tol).collect();
    let rank = is.len();
    let mut sqrt: Vec<f64> = is.iter().map(|&i| xs[i].sqrt()).collect();
    let ib: Vec<(usize, usize)> = dsge.init_states.clone();
    let within = |c: &[f64]| c.iter().zip(&bounds.lb).zip(&bounds.ub).all(|((v, l), u)| v >= l && v <= u);
    let clamp = |c: &mut Vec<f64>| {
        for i in 0..c.len() {
            if c[i] < bounds.lb[i] {
                c[i] = bounds.lb[i] + f64::EPSILON.sqrt();
            }
            if c[i] > bounds.ub[i] {
                c[i] = bounds.ub[i] - f64::EPSILON.sqrt();
            }
        }
    };
    let dev_to_x = |alpha_dr: &[f64], xv: &mut Vec<f64>| {
        for &(i, k) in &ib {
            let dev = alpha_dr[dr.inv_order_var[k]];
            xv[i] = if dsge.loglinear { (ys[k].ln() + dev).exp() } else { ys[k] + dev };
        }
    };
    // the stored initial states minus the steady state, DR order
    let store_dev = |xv: &[f64]| -> Vec<f64> {
        let mut values = vec![0.0; n];
        for &(i, k) in &ib {
            values[k] = xv[i];
        }
        (0..n)
            .map(|j| {
                let v = dr.order_var[j];
                if dsge.loglinear { values[v].ln() - ys[v].ln() } else { values[v] - ys[v] }
            })
            .collect()
    };
    let f = |c: &[f64]| -post.eval(c).0;
    let mut logpost_smo = f64::NEG_INFINITY;
    let mut is_smoothed_state_optimal = true;
    if !init {
        let mut xproposal = xparam1.clone();
        dev_to_x(&alphahat0, &mut xproposal);
        if !within(&xproposal) {
            clamp(&mut xproposal);
        }
        let (xc, icheck) = set_init_state(post, &xproposal, &ys0);
        if icheck {
            xproposal = xc;
        }
        let logpost1 = f(&xproposal);
        neval += 1;
        if logpost1 < logpost0 {
            is_smoothed_state_optimal = false;
        }
        logpost_smo = logpost1;
        if !is_smoothed_state_optimal {
            alphahat0 = store_dev(&xparam1);
        }
    }
    let (mut naccepted, mut nattempts) = (0, 0);
    while naccepted < target && nattempts < 10 {
        let mut niter = 0;
        nattempts += 1;
        while naccepted < target && niter < 20 {
            niter += 1;
            let mut xproposal = xparam1.clone();
            let mut out_of_bounds = true;
            let mut icount = 0;
            while out_of_bounds && icount < 10 {
                icount += 1;
                let draw: Vec<f64> = (0..rank).map(|_| StandardNormal.sample(rng)).collect();
                let mut alpha01 = alphahat0.clone();
                for (r, &row) in rows.iter().enumerate() {
                    let mut yv = alphahat0[row];
                    for (c, &j) in is.iter().enumerate() {
                        yv += u[(r, j)] * sqrt[c] * draw[c];
                    }
                    alpha01[row] = yv;
                }
                xproposal = xparam1.clone();
                dev_to_x(&alpha01, &mut xproposal);
                if within(&xproposal) {
                    out_of_bounds = false;
                }
            }
            if out_of_bounds {
                clamp(&mut xproposal);
            }
            let (xc, icheck) = set_init_state(post, &xproposal, &ys0);
            if icheck {
                xproposal = xc;
            }
            if init {
                naccepted = 1;
                xparam1 = xproposal;
                break;
            }
            let lnrand = rng.r#gen::<f64>().ln();
            let mut logpost1 = f(&xproposal);
            neval += 1;
            let accepted = if logpost1 < logpost0 {
                let r = logpost1 - logpost0;
                if logpost1 > f64::NEG_INFINITY && lnrand < r {
                    xparam1 = xproposal;
                    true
                } else {
                    logpost1 = logpost0;
                    false
                }
            } else {
                xparam1 = xproposal;
                true
            };
            if accepted {
                logpost0 = logpost1;
                naccepted += 1;
                if logpost_smo < logpost0 {
                    is_smoothed_state_optimal = false;
                }
                if !is_smoothed_state_optimal {
                    alphahat0 = store_dev(&xparam1);
                }
            }
        }
        if init {
            break;
        }
        if naccepted == 0 {
            sqrt.iter_mut().for_each(|s| *s *= 0.66);
        }
    }
    (xparam1, logpost0, neval)
}

/// Is `estimate_initial_states_endogenous_prior` usable here (Dynare's checks are in
/// [`crate::estimation::est_completion::check_initial_states`])?
pub fn check(dsge: &DsgeModel) -> Result<()> {
    if dsge.ml {
        bail!(
            "estimate_initial_states_endogenous_prior: maximum likelihood is not possible (the initial states have priors)"
        );
    }
    Ok(())
}
