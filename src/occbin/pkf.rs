// SPDX-License-Identifier: GPL-3.0-or-later
//! OccBin's piecewise Kalman filter (PKF): ports of `+occbin/kalman_update_algo_1.m`
//! (with its `occbin_kalman_update0` sub-function), `+occbin/kalman_update_engine.m`
//! (multivariate branch) and of the OccBin branch of
//! `kalman/likelihood/missing_observations_kalman_filter.m` (Dynare 7.2).
//!
//! Each period, the update step smooths the state of the previous period and the current shock
//! given the current observation, simulates the piecewise-linear model from there with the
//! OccBin solver, and iterates on the regime of the current period until it is consistent
//! with the state-space matrices used in the update. [`PointModel`] holds what the solver
//! needs at one parameter point (and the solver's regime dictionary, Dynare's `persistent`
//! state, which only caches results).

use nalgebra::DVector;

use super::constraints::Constraints;
use super::regimes::{self, Regimes};
use super::solver::{self, Dictionary, Dm, OccbinError, Problem, SimulOptions, SolveOutput, StateSpace};
use crate::linalg::{self, Mat};
use crate::solver::DecisionRule;

type Vector = DVector<f64>;

/// The OccBin model at one parameter point, with the restricted state space of the filter.
pub struct PointModel<'a> {
    pub dm: Dm,
    pub constraints: &'a Constraints,
    pub params: Vec<f64>,
    /// Steady state (declaration order).
    pub ys: Vec<f64>,
    /// Number of constraints (1 or 2).
    pub nc: usize,
    /// Declaration index of each state of the filter (`dr.order_var(dr.restrict_var_list)`,
    /// or `dr.order_var` without restriction).
    pub my_order_var: Vec<usize>,
    /// The baseline (slack) restricted transition: a_{t+1} = T a_t + R e_t.
    pub t: Mat,
    pub r: Mat,
    dict: Option<Dictionary>,
}

impl<'a> PointModel<'a> {
    /// `restrict`: DR-order indices of the filter's states (`dr.restrict_var_list`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: &crate::model::Model,
        constraints: &'a Constraints,
        pswitch: &[usize],
        params: Vec<f64>,
        ys: Vec<f64>,
        dr: &DecisionRule,
        restrict: &[usize],
        t: Mat,
        r: Mat,
    ) -> Self {
        let dm = Dm::new(m, &ys, &params, pswitch, dr);
        let my_order_var = restrict.iter().map(|&k| dr.order_var[k]).collect();
        PointModel { dm, constraints, params, ys, nc: pswitch.len(), my_order_var, t, r, dict: None }
    }

    /// `occbin.solver` at this parameter point (paths in deviations from the steady state).
    pub fn solve(&mut self, o: &SimulOptions) -> Result<SolveOutput, OccbinError> {
        let pb = Problem { dm: &self.dm, constraints: self.constraints, params: &self.params, ys: &self.ys };
        if self.nc == 1 {
            solver::solve_one_constraint(&pb, &mut self.dict, o)
        } else {
            solver::solve_two_constraints(&pb, &mut self.dict, o)
        }
    }

    /// One period's state space restricted to the filter's states.
    pub fn restrict(&self, s: &StateSpace) -> (Mat, Mat, Vector) {
        let iv = &self.my_order_var;
        let n = iv.len();
        let t = Mat::from_fn(n, n, |i, j| s.t[(iv[i], iv[j])]);
        let r = Mat::from_fn(n, s.r.ncols(), |i, j| s.r[(iv[i], j)]);
        let c = Vector::from_fn(n, |i, _| s.c[iv[i]]);
        (t, r, c)
    }

    /// `occbin.dynare_resolve(..., regime, 'restrict', T0, R0)` -> `check_regimes`: the state
    /// space of the first period of a guessed regime (solver with `maxit = 1`).
    pub fn regime_state_space(&mut self, reg: &Regimes, base_opts: &SimulOptions) -> (Mat, Mat, Vector) {
        let n = self.t.nrows();
        if regimes::is_base(reg) {
            return (self.t.clone(), self.r.clone(), Vector::zeros(n));
        }
        let mut o = base_opts.clone();
        o.init_regime = Some(vec![reg.clone()]);
        o.init_binding_indicator = Vec::new();
        o.endo_init = Vec::new();
        o.shocks = Mat::zeros(1, self.dm.n_exo);
        o.exo_pos = (0..self.dm.n_exo).collect();
        o.maxit = 1;
        o.periods = regimes::max_start(std::slice::from_ref(reg));
        match self.solve(&o) {
            Ok(out) => self.restrict(&out.ss[0]),
            Err(_) => (
                Mat::from_element(n, n, f64::NAN),
                Mat::from_element(n, self.r.ncols(), f64::NAN),
                Vector::from_element(n, f64::NAN),
            ),
        }
    }

    /// Declaration-order initial state of the solver from a vector of filter states
    /// (`tmp(dr.restrict_var_list) = alpha; endo_init = tmp(dr.inv_order_var)`).
    fn endo_init(&self, alpha: &Vector) -> Vec<f64> {
        let mut e = vec![0.0; self.dm.n_vars];
        for (k, &v) in self.my_order_var.iter().enumerate() {
            e[v] = alpha[k];
        }
        e
    }
}

/// Options of the PKF (`options_.occbin.likelihood`, `.filter`).
#[derive(Clone, Debug)]
pub struct PkfOptions {
    /// `occbin_options.opts_simul` (the solver options of the likelihood).
    pub simul: SimulOptions,
    pub max_number_of_iterations: usize,
    pub periodic_solution: bool,
    pub use_relaxation: bool,
    pub use_relaxation_tol_period: i64,
    pub brute_force_regime_guess: bool,
    pub brute_force_extra_regime_guess: bool,
    pub number_of_initial_periods_with_extra_regime_guess: usize,
    pub first_period_binding_regime_allowed: usize,
    pub rescale_prediction_error_covariance: bool,
}

/// Inputs of one update (`kalman_update_algo_1`): two periods, t-1 (index 0) and t (1).
pub struct UpdateInput<'b> {
    /// `a(:,1)`: updated state of t-1.
    pub a0: &'b Vector,
    /// `a1(:,1)`, `P1(:,:,1)`: predicted state and covariance of t-1.
    pub a1_0: &'b Vector,
    pub p0: &'b Mat,
    pub p1_0: &'b Mat,
    pub data_index: [&'b [usize]; 2],
    /// Observed rows of the state vector (Z selection).
    pub mf: &'b [usize],
    /// `v(:,1)`: innovation of t-1; `y`: observations of t.
    pub v0: &'b Vector,
    pub y: &'b Vector,
    pub h: &'b Mat,
    pub q: &'b Mat,
    pub tt: [&'b Mat; 2],
    pub rr: [&'b Mat; 2],
    pub cc: [&'b Vector; 2],
    /// The univariate update of `kalman_update_algo_3.m` (OccBin smoother) instead of the
    /// multivariate one of `kalman_update_algo_1.m`.
    pub uni: Option<Univariate<'b>>,
}

/// Inputs of the univariate update (`kalman_update_algo_3.m`): `Fi(:,t-1)`, `Ki(:,:,t-1)`.
#[derive(Clone, Copy)]
pub struct Univariate<'b> {
    pub fi0: &'b Vector,
    pub ki0: &'b Mat,
    pub kalman_tol: f64,
}

/// Outputs of `kalman_update_algo_1` (state fields are only meaningful when `info == 0`).
#[derive(Clone, Debug)]
pub struct Update {
    /// `a`: updated state of t, predicted state of t+1.
    pub a: [Vector; 2],
    pub a1: [Vector; 2],
    pub p: [Mat; 2],
    pub p1: [Mat; 2],
    pub v: [Vector; 2],
    pub t: [Mat; 2],
    pub r: [Mat; 2],
    pub c: [Vector; 2],
    /// Regimes of periods t, t+1, t+2.
    pub regimes: Vec<Regimes>,
    pub info: i32,
    pub lik: f64,
    /// Smoothed shock of t.
    pub etahat: Vector,
    /// Univariate update: `Fi(:,t)`, `Ki(:,:,t)`.
    pub fi: Vector,
    pub ki: Mat,
}

impl Update {
    fn failed(nc: usize, info: i32, regimes: Vec<Regimes>) -> Update {
        let e = Vector::zeros(0);
        let m = Mat::zeros(0, 0);
        let _ = nc;
        Update {
            a: [e.clone(), e.clone()],
            a1: [e.clone(), e.clone()],
            p: [m.clone(), m.clone()],
            p1: [m.clone(), m.clone()],
            v: [e.clone(), e.clone()],
            t: [m.clone(), m.clone()],
            r: [m.clone(), m.clone()],
            c: [e.clone(), e.clone()],
            regimes,
            info,
            lik: f64::INFINITY,
            etahat: e.clone(),
            fi: e,
            ki: m,
        }
    }
}

/// MATLAB `rank`: singular values above `max(size) * eps(max(s))`.
fn rank(f: &Mat) -> usize {
    if f.nrows() == 0 {
        return 0;
    }
    let s = linalg::singular_values(f);
    let smax = s.iter().cloned().fold(0.0, f64::max);
    let tol = f.nrows().max(f.ncols()) as f64 * eps(smax);
    s.iter().filter(|&&x| x > tol).count()
}

/// MATLAB `eps(x)`.
fn eps(x: f64) -> f64 {
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let e = x.abs().log2().floor();
    2f64.powf(e - 52.0)
}

/// MATLAB `inv` (LU with partial pivoting).
fn inv(a: &Mat) -> Mat {
    let n = a.nrows();
    let (f, ipiv, _) = linalg::lu(a);
    let mut x = Mat::identity(n, n);
    linalg::lu_solve(&f, &ipiv, false, &mut x);
    x
}

/// MATLAB `det` (LU).
fn det(a: &Mat) -> f64 {
    let n = a.nrows();
    if n == 0 {
        return 1.0;
    }
    let (f, ipiv, _) = linalg::lu(a);
    let mut d = 1.0;
    for i in 0..n {
        d *= f[(i, i)];
        if ipiv[i] as usize != i + 1 {
            d = -d;
        }
    }
    d
}

/// `inv(F./(sig*sig'))./(sig*sig')`.
fn inv_rescaled(f: &Mat) -> Mat {
    let n = f.nrows();
    let sig: Vec<f64> = (0..n).map(|i| f[(i, i)].sqrt()).collect();
    let fs = Mat::from_fn(n, n, |i, j| f[(i, j)] / (sig[i] * sig[j]));
    let mut x = inv(&fs);
    for i in 0..n {
        for j in 0..n {
            x[(i, j)] /= sig[i] * sig[j];
        }
    }
    x
}

/// `F = Z(di,:) P Z(di,:)' + H(di,di)`.
fn forecast_cov(p: &Mat, mf: &[usize], di: &[usize], h: &Mat) -> Mat {
    let n = di.len();
    Mat::from_fn(n, n, |i, j| p[(mf[di[i]], mf[di[j]])] + h[(di[i], di[j])])
}

/// `iF(di,di,1)` and `L(:,:,1)` of period t-1 (top of `kalman_update_algo_1`).
struct Lagged {
    if1: Mat,
    l1: Mat,
}

fn lagged(inp: &UpdateInput) -> Result<Lagged, i32> {
    let mm = inp.a0.len();
    if inp.uni.is_some() {
        return Ok(Lagged { if1: Mat::zeros(0, 0), l1: Mat::zeros(0, 0) });
    }
    let di = inp.data_index[0];
    let f = forecast_cov(inp.p1_0, inp.mf, di, inp.h);
    if f.iter().any(|v| v.is_nan()) {
        return Err(325);
    }
    if rank(&f) < f.nrows() {
        return Err(326);
    }
    let if1 = inv_rescaled(&f);
    // L = I - P1 Z' iF Z
    let mut l1 = Mat::identity(mm, mm);
    if !di.is_empty() {
        let pzi =
            Mat::from_fn(mm, di.len(), |i, j| (0..di.len()).map(|k| inp.p1_0[(i, inp.mf[di[k]])] * if1[(k, j)]).sum());
        for (j, &d) in di.iter().enumerate() {
            let c = inp.mf[d];
            for i in 0..mm {
                l1[(i, c)] -= pzi[(i, j)];
            }
        }
    }
    Ok(Lagged { if1, l1 })
}

/// Output of `occbin_kalman_update0`.
struct Upd0 {
    fi: Vector,
    ki: Mat,
    a1: Vector,
    p: Mat,
    p1: Mat,
    v: Vector,
    alphahat0: Vector,
    etahat1: Vector,
    lik: f64,
}

/// Port of `occbin_kalman_update0` (sub-function of kalman_update_algo_1.m): update of
/// period t with the matrices `TT(:,:,2), RR(:,:,2), CC(:,2)`, then the two-period smoother.
fn update0(inp: &UpdateInput, lg: &Lagged, tt: &Mat, rr: &Mat, cc: &Vector, rescale: bool) -> Result<Upd0, i32> {
    if let Some(u) = &inp.uni {
        return Ok(update_univariate(inp, u, tt, rr, cc));
    }
    let mm = inp.a0.len();
    let q = inp.q;
    let qq = rr * q * rr.transpose();
    let a1 = tt * inp.a0 + cc;
    let mut a = a1.clone();
    let p1 = tt * inp.p0 * tt.transpose() + &qq;
    let mut p = p1.clone();
    let di = inp.data_index[1];
    let mut v = Vector::zeros(inp.y.len());
    let lik;
    let mut r2 = Vector::zeros(mm);
    if di.is_empty() {
        lik = 0.0;
    } else {
        for (k, &d) in di.iter().enumerate() {
            let _ = k;
            v[d] = inp.y[d] - a[inp.mf[d]];
        }
        let f = forecast_cov(&p, inp.mf, di, inp.h);
        if f.iter().any(|x| x.is_nan()) {
            return Err(325);
        }
        if rank(&f) < f.nrows() {
            return Err(326);
        }
        let n = di.len();
        let (log_df, i_f) = if rescale {
            let sig: Vec<f64> = (0..n).map(|i| f[(i, i)].sqrt()).collect();
            let fs = Mat::from_fn(n, n, |i, j| f[(i, j)] / (sig[i] * sig[j]));
            (det(&fs).ln() + 2.0 * sig.iter().map(|s| s.ln()).sum::<f64>(), inv_rescaled(&f))
        } else {
            (det(&f).ln(), inv(&f))
        };
        let vd = Vector::from_fn(n, |i, _| v[di[i]]);
        let ifv = &i_f * &vd;
        lik = log_df + vd.dot(&ifv) + n as f64 * (2.0 * std::f64::consts::PI).ln();
        // PZI = P Z' iF; a = a + PZI v; K = PZI Z; L = I - K; P = P - K P
        let pzi = Mat::from_fn(mm, n, |i, j| (0..n).map(|k| p[(i, inp.mf[di[k]])] * i_f[(k, j)]).sum());
        a += &pzi * &vd;
        let mut k = Mat::zeros(mm, mm);
        for (j, &d) in di.iter().enumerate() {
            let c = inp.mf[d];
            for i in 0..mm {
                k[(i, c)] += pzi[(i, j)];
            }
        }
        p = &p - &k * &p;
        // r_1 = Z' iF v (+ L' r_2 with r_2 = 0)
        for (j, &d) in di.iter().enumerate() {
            r2[inp.mf[d]] += ifv[j];
        }
    }
    let etahat1 = q * rr.transpose() * &r2;
    let r1t = tt.transpose() * &r2; // r(:,2) = T' r(:,2)
    // t = 1 (period t-1)
    let di0 = inp.data_index[0];
    let mut r1 = lg.l1.transpose() * &r1t;
    if !di0.is_empty() {
        let n0 = di0.len();
        let v0 = Vector::from_fn(n0, |i, _| inp.v0[di0[i]]);
        let x = &lg.if1 * &v0;
        for (j, &d) in di0.iter().enumerate() {
            r1[inp.mf[d]] += x[j];
        }
    }
    let alphahat0 = inp.a1_0 + inp.p1_0 * &r1;
    Ok(Upd0 { fi: Vector::zeros(0), ki: Mat::zeros(0, 0), a1, p, p1, v, alphahat0, etahat1, lik })
}

/// Univariate backward step over the observations of one period (`flipud(di)`):
/// `r = Z_i' v_i / F_i + L_i' r`, `L_i = I - K_i Z_i / F_i`.
fn univariate_r(r: &mut Vector, di: &[usize], mf: &[usize], fi: &Vector, ki: &Mat, v: &Vector, tol: f64) {
    for &i in di.iter().rev() {
        let f = fi[i];
        if f > tol {
            // L' r = r - Z_i' (K_i' r) / F_i
            let kr: f64 = (0..r.len()).map(|j| ki[(j, i)] * r[j]).sum();
            r[mf[i]] += (v[i] - kr) / f;
        }
    }
}

/// Port of `occbin_kalman_update` (sub-function of kalman_update_algo_3.m): univariate update
/// of period t, then the two-period univariate smoother.
fn update_univariate(inp: &UpdateInput, u: &Univariate, tt: &Mat, rr: &Mat, cc: &Vector) -> Upd0 {
    let mm = inp.a0.len();
    let pp = inp.y.len();
    let q = inp.q;
    let qq = rr * q * rr.transpose();
    let a1 = tt * inp.a0 + cc;
    let mut a = a1.clone();
    let p1 = tt * inp.p0 * tt.transpose() + &qq;
    let mut p = p1.clone();
    let di = inp.data_index[1];
    let mut v = Vector::zeros(pp);
    let mut fi = Vector::zeros(pp);
    let mut ki = Mat::zeros(mm, pp);
    let mut lik = 0.0;
    let n = di.len();
    let (mut vv, mut f) = (Vector::zeros(n), Mat::zeros(n, n));
    if n > 0 {
        vv = Vector::from_fn(n, |i, _| inp.y[di[i]] - a[inp.mf[di[i]]]);
        f = Mat::from_fn(n, n, |i, j| {
            p[(inp.mf[di[i]], inp.mf[di[j]])] + if i == j { inp.h[(di[i], di[i])] } else { 0.0 }
        });
    }
    for &i in di {
        let zi = inp.mf[i];
        v[i] = inp.y[i] - a[zi];
        fi[i] = p[(zi, zi)] + inp.h[(i, i)];
        let k = p.column(zi).into_owned();
        ki.set_column(i, &k);
        if fi[i] > u.kalman_tol {
            a += &k * (v[i] / fi[i]);
            p -= &k * k.transpose() / fi[i];
        }
    }
    if n > 0 {
        let sig: Vec<f64> = (0..n).map(|i| f[(i, i)].sqrt()).collect();
        let fs = Mat::from_fn(n, n, |i, j| f[(i, j)] / (sig[i] * sig[j]));
        let log_df = det(&fs).ln() + 2.0 * sig.iter().map(|s| s.ln()).sum::<f64>();
        let i_f = inv_rescaled(&f);
        lik = log_df + vv.dot(&(&i_f * &vv)) + n as f64 * (2.0 * std::f64::consts::PI).ln();
    }
    let mut r = Vector::zeros(mm);
    univariate_r(&mut r, di, inp.mf, &fi, &ki, &v, u.kalman_tol);
    let etahat1 = q * rr.transpose() * &r;
    r = tt.transpose() * &r;
    univariate_r(&mut r, inp.data_index[0], inp.mf, u.fi0, u.ki0, inp.v0, u.kalman_tol);
    let alphahat0 = inp.a1_0 + inp.p1_0 * &r;
    Upd0 { fi, ki, a1, p, p1, v, alphahat0, etahat1, lik }
}

/// Port of `kalman_update_algo_1.m`. `regimes0`: `None` for `struct()`, else the guessed
/// regimes of periods t (and t+1); `guess_regime` = `options_.occbin.filter.guess_regime`.
pub fn kalman_update_algo_1(
    pm: &mut PointModel,
    inp: &UpdateInput,
    regimes0: Option<&[Regimes]>,
    guess_regime: bool,
    o: &PkfOptions,
) -> Update {
    let nc = pm.nc;
    let base = regimes::base(nc);
    let base3 = vec![base.clone(), base.clone(), base.clone()];
    let lg = match lagged(inp) {
        Ok(l) => l,
        Err(e) => return Update::failed(nc, e, base3),
    };
    let mut tt2: Mat;
    let mut rr2: Mat;
    let mut cc2: Vector;
    let mut regimes0: Vec<Regimes> = match regimes0 {
        Some(r) => r.to_vec(),
        None => Vec::new(),
    };
    if !regimes0.is_empty() {
        if guess_regime {
            let (t, r, c) = pm.regime_state_space(&regimes0[0], &o.simul);
            // regimes0(2:3): the guess shifted by one and two periods
            let r1: Regimes = regimes0[0].iter().map(regimes::shift).collect();
            let r2: Regimes = r1.iter().map(regimes::shift).collect();
            regimes0.resize(3, base.clone());
            regimes0[1] = r1;
            regimes0[2] = r2;
            tt2 = t;
            rr2 = r;
            cc2 = c;
        } else {
            tt2 = inp.tt[1].clone();
            rr2 = inp.rr[1].clone();
            cc2 = inp.cc[1].clone();
        }
    } else {
        tt2 = pm.t.clone();
        rr2 = pm.r.clone();
        cc2 = Vector::zeros(pm.t.nrows());
        regimes0 = base3.clone();
    }
    let mut u = match update0(inp, &lg, &tt2, &rr2, &cc2, o.rescale_prediction_error_covariance) {
        Ok(u) => u,
        Err(e) => return Update::failed(nc, e, base3),
    };
    let n_exo = pm.dm.n_exo;
    let mut opts = o.simul.clone();
    opts.shocks = Mat::zeros(3, n_exo);
    opts.exo_pos = (0..n_exo).collect();
    let set_shock = |opts: &mut SimulOptions, u: &Upd0, pm: &PointModel| {
        for k in 0..n_exo {
            opts.shocks[(0, k)] = u.etahat1[k];
        }
        opts.endo_init = pm.endo_init(&u.alphahat0);
    };
    set_shock(&mut opts, &u, pm);
    let mut out = if guess_regime {
        let mut og = opts.clone();
        og.init_regime = Some(regimes0.clone());
        match pm.solve(&og) {
            Ok(x) => Ok(x),
            Err(_) => pm.solve(&opts),
        }
    } else {
        match pm.solve(&opts) {
            Ok(x) => Ok(x),
            Err(_) => {
                let mut og = opts.clone();
                og.init_regime = Some(regimes0.clone());
                pm.solve(&og)
            }
        }
    };
    let mut res = match out {
        Ok(x) => x,
        Err(e) => return Update::failed(nc, e.code, base3),
    };
    let mut regimes_ = res.regime_history.clone();
    let mut regime_hist: Vec<Regimes> = vec![regimes0[0].clone()];
    let mut lik_hist = vec![u.lik];
    let mut niter = 1usize;
    let mut is_periodic = false;
    if regimes::any_binding(&regimes_) || regimes_[0] != regimes0[0] {
        while regimes_[0] != regimes0[0] && !is_periodic && niter <= o.max_number_of_iterations {
            niter += 1;
            (tt2, rr2, cc2) = pm.restrict(&res.ss[0]);
            regime_hist.push(regimes_[0].clone());
            u = match update0(inp, &lg, &tt2, &rr2, &cc2, o.rescale_prediction_error_covariance) {
                Ok(x) => x,
                Err(e) => return Update::failed(nc, e, regimes_),
            };
            lik_hist.push(u.lik);
            set_shock(&mut opts, &u, pm);
            opts.init_regime = Some(vec![regimes_[0].clone()]);
            opts.periods = opts.periods.max(regimes::max_start(&regimes_));
            out = match pm.solve(&opts) {
                Ok(x) => Ok(x),
                Err(_) if inp.uni.is_none() => {
                    let mut og = opts.clone();
                    og.init_regime = None;
                    pm.solve(&og)
                }
                Err(e) => Err(e),
            };
            res = match out {
                Ok(x) => x,
                Err(e) => return Update::failed(nc, e.code, regimes_),
            };
            regimes0 = std::mem::replace(&mut regimes_, res.regime_history.clone());
            let periodic: Vec<usize> = (0..niter - 1).filter(|&k| regime_hist[k] == regimes_[0]).collect();
            is_periodic = !periodic.is_empty();
            if is_periodic {
                if o.periodic_solution {
                    let first = periodic[0];
                    let mut im = first;
                    for k in first..lik_hist.len() {
                        if lik_hist[k] < lik_hist[im] {
                            im = k;
                        }
                    }
                    opts.init_regime = Some(vec![regime_hist[im].clone()]);
                    opts.periods = opts.periods.max(regimes::max_start(&regimes0));
                    opts.maxit = 1;
                    res = match pm.solve(&opts) {
                        Ok(x) => x,
                        Err(e) => return Update::failed(nc, e.code, regimes_),
                    };
                    regimes_ = res.regime_history.clone();
                    (tt2, rr2, cc2) = pm.restrict(&res.ss[0]);
                    u = match update0(inp, &lg, &tt2, &rr2, &cc2, o.rescale_prediction_error_covariance) {
                        Ok(x) => x,
                        Err(e) => return Update::failed(nc, e, regimes_),
                    };
                } else {
                    return Update::failed(nc, 330, regimes_);
                }
            }
        }
    }
    let _ = (&tt2, &rr2, &cc2);
    if niter > o.max_number_of_iterations && regimes_[0] != regimes0[0] {
        return Update::failed(nc, 331, regimes_);
    }
    let a: [Vector; 2] =
        std::array::from_fn(|k| Vector::from_fn(pm.my_order_var.len(), |i, _| res.piecewise[(k, pm.my_order_var[i])]));
    regimes_.truncate(3);
    let (t0, r0, c0) = pm.restrict(&res.ss[0]);
    let (t1, r1, c1) = pm.restrict(&res.ss[1]);
    let qq = &r1 * inp.q * r1.transpose();
    let p_next = &t1 * &u.p * t1.transpose() + qq;
    Update {
        a,
        a1: [inp.a1_0.clone(), u.a1],
        p: [u.p, p_next],
        p1: [inp.p1_0.clone(), u.p1],
        v: [inp.v0.clone(), u.v],
        t: [t0, t1],
        r: [r0, r1],
        c: [c0, c1],
        regimes: regimes_,
        info: 0,
        lik: u.lik,
        etahat: u.etahat1,
        fi: u.fi,
        ki: u.ki,
    }
}

/// Port of `kalman_update_engine.m` (multivariate filter): the update from no guess, from the
/// regimes expected in the previous period, the optional relaxation of the regime duration
/// (`filter_use_relaxation`) and the brute-force regime guesses; several solutions are mixed
/// in the likelihood.
pub fn kalman_update_engine(
    pm: &mut PointModel,
    inp: &UpdateInput,
    t: usize,
    regimes_in: &[Regimes],
    o: &PkfOptions,
) -> Update {
    let nc = pm.nc;
    let base = regimes::base(nc);
    let base2 = regimes::is_base(&regimes_in[0]) && regimes::is_base(&regimes_in[1]);
    let mut x = kalman_update_algo_1(pm, inp, None, false, o);
    let mut likvec: Vec<f64> = Vec::new();
    let mut regvec: Vec<Regimes> = Vec::new();
    if x.info == 0 {
        likvec.push(x.lik);
        regvec.push(x.regimes[0].clone());
    } else {
        x.lik = f64::INFINITY;
    }
    let info0 = x.info;
    let info1;
    let mut brute_force_regime_guess = o.brute_force_regime_guess;
    if x.info != 0 {
        if !base2 {
            x = kalman_update_algo_1(pm, inp, Some(&regimes_in[..2]), false, o);
        }
        if x.info == 0 {
            likvec = vec![x.lik];
            regvec = vec![x.regimes[0].clone()];
        } else {
            x.lik = f64::INFINITY;
        }
        info1 = x.info;
    } else if !base2 {
        let x1 = kalman_update_algo_1(pm, inp, Some(&regimes_in[..2]), false, o);
        info1 = x1.info;
        if x1.info == 0 && x1.regimes != x.regimes {
            likvec.push(x1.lik);
            regvec.push(x1.regimes[0].clone());
        }
        if x1.info == 0 && x1.lik < x.lik {
            x = x1;
        }
    } else if t > o.number_of_initial_periods_with_extra_regime_guess {
        info1 = 0;
    } else {
        // Dynare also sets loss_function_regime_guess here (occbin.findmin, not ported); the
        // option that leads here cannot be set from the .mod file.
        info1 = 1;
        brute_force_regime_guess = true;
    }
    if t < o.first_period_binding_regime_allowed {
        return x;
    }
    // relaxation of the start of the last regime (filter_use_relaxation)
    let mut diffstart: i64 = 0;
    let mut which = 0usize;
    if x.info == 0 && regvec.len() == 1 {
        let d: Vec<i64> = (0..nc)
            .map(|k| *x.regimes[0][k].start.last().unwrap() as i64 - *regimes_in[0][k].start.last().unwrap() as i64)
            .collect();
        // MATLAB max: first maximum
        which = 0;
        for k in 1..nc {
            if d[k] > d[which] {
                which = k;
            }
        }
        diffstart = d[which];
    }
    if o.use_relaxation && diffstart > o.use_relaxation_tol_period {
        let mut guess = vec![x.regimes[0].clone(), base.clone()];
        let mut reg_r = x.regimes.clone();
        while guess[0] == reg_r[0] {
            let last_r = *reg_r[0][which].start.last().unwrap();
            if last_r <= 1 {
                break;
            }
            let g = &mut guess[0][which];
            let n = g.start.len();
            g.start[n - 1] = last_r - 1;
            if g.start[n - 1] == 1 {
                g.start = vec![1];
                g.regime = vec![false];
            } else if g.start[n - 2] == g.start[n - 1] {
                g.start[n - 2] -= 1;
            }
            let xr = kalman_update_algo_1(pm, inp, Some(&guess), true, o);
            reg_r = xr.regimes.clone();
            if !regvec.contains(&xr.regimes[0]) {
                likvec.push(xr.lik);
                regvec.push(xr.regimes[0].clone());
            }
            if xr.info == 0 && xr.lik < x.lik {
                x = xr;
            }
        }
        brute_force_regime_guess = false;
    }
    // brute-force regime guesses
    if (brute_force_regime_guess && info0 != 0 && info1 != 0)
        || (o.brute_force_extra_regime_guess && (info0 != 0 || info1 != 0))
    {
        let mut guess = vec![base.clone(), base.clone()];
        let mut found: Option<Update> = None;
        let same = |x: &Update, y: &Update| x.info == 0 && y.regimes == x.regimes;
        if nc == 1 {
            for k in 1..=5usize {
                let mut use_index: Option<Update> = None;
                guess[0][0] = regimes::regime(&[0, 1, 0], &[1, 5, 5 + 4 * k]);
                let x2 = kalman_update_algo_1(pm, inp, Some(&guess), true, o);
                if x2.info == 0 {
                    let stop = !same(&x, &x2);
                    use_index = Some(x2);
                    if stop {
                        found = use_index;
                        break;
                    }
                }
                guess[0][0] = regimes::regime(&[1, 0], &[1, 1 + 4 * k]);
                let x2 = kalman_update_algo_1(pm, inp, Some(&guess), true, o);
                if x2.info == 0 {
                    use_index = Some(x2);
                }
                if use_index.is_some() {
                    found = use_index;
                    break;
                }
            }
        } else {
            'jk: for jk in 0..=1usize {
                for k in 1..=5usize {
                    let mut use_index: Option<Update> = None;
                    let mut info2 = 1;
                    let mut last_same = false;
                    'jr: for jr in 1..=2usize {
                        let (c1, c2) = if jr == 1 { (0, 1) } else { (1, 0) };
                        for kk in 1..=2usize {
                            guess[0][c1] = if kk == 1 {
                                regimes::regime(&[0, 1, 0], &[1, 5, 5 + 4 * k])
                            } else {
                                regimes::regime(&[1, 0], &[1, 1 + 4 * k])
                            };
                            for kj in 1..=(1 + (jk > 0) as usize) {
                                guess[0][c2] = if jk == 0 {
                                    regimes::regime(&[0], &[1])
                                } else if kj == 1 {
                                    regimes::regime(&[1, 0], &[1, 1 + 4 * jk])
                                } else {
                                    regimes::regime(&[0, 1, 0], &[1, 5, 5 + 4 * jk])
                                };
                                let x2 = kalman_update_algo_1(pm, inp, Some(&guess), true, o);
                                info2 = x2.info;
                                if x2.info == 0 {
                                    last_same = same(&x, &x2);
                                    use_index = Some(x2);
                                    if !last_same {
                                        break;
                                    }
                                }
                            }
                            if info2 == 0 && !last_same {
                                break;
                            }
                        }
                        if info2 == 0 && !last_same {
                            break 'jr;
                        }
                    }
                    if use_index.is_some() {
                        found = use_index;
                        break 'jk;
                    }
                }
            }
        }
        if let Some(x2) = found {
            if !regvec.contains(&x2.regimes[0]) {
                regvec.push(x2.regimes[0].clone());
                likvec.push(x2.lik);
            }
            if x2.lik < x.lik {
                x = x2;
            }
        }
    }
    if likvec.len() > 1 {
        let m = likvec.iter().cloned().filter(|v| !v.is_nan()).fold(f64::INFINITY, f64::min);
        let s: f64 = likvec.iter().map(|l| (-(l - m) / 2.0).exp()).sum();
        x.lik = -2.0 * s.ln() + m;
    }
    if x.info == 0 {
        if x.lik.is_nan() {
            x.info = 323;
        } else if x.a.iter().any(|v| v.iter().any(|z| z.is_nan())) {
            x.info = 324;
        }
    }
    x
}

/// Result of the piecewise Kalman filter.
#[derive(Clone, Debug)]
pub struct PkfOutput {
    /// `LIK`: minus the log likelihood (presample periods excluded); +inf on failure.
    pub minus_loglik: f64,
    /// Per-period contributions (halved, as `lik` of the filter).
    pub lik: Vec<f64>,
    /// Filter's regimes (`regimes_`, periods 1..nobs+2).
    pub regimes: Vec<Regimes>,
    /// Updated states a_{t|t} (mm x nobs, `a0`).
    pub a_updated: Mat,
    /// 1-based period of the failure, `info` code of the update (0 = success).
    pub fail_period: usize,
    pub info: i32,
}

/// The OccBin branch of `missing_observations_kalman_filter.m` from period 1
/// (`first_period_occbin_update = 1`, no diffuse periods): `y` the demeaned data (pp x nobs),
/// `a`, `p` the initial predicted state and covariance.
#[allow(clippy::too_many_arguments)]
pub fn pkf(
    pm: &mut PointModel,
    y: &Mat,
    data_index: &[Vec<usize>],
    mf: &[usize],
    q: &Mat,
    h: &Mat,
    a: Vector,
    p: &Mat,
    presample: usize,
    o: &PkfOptions,
) -> PkfOutput {
    let (pp, last) = (y.nrows(), y.ncols());
    let mm = a.len();
    let nc = pm.nc;
    // P = tril(P) + tril(P,-1)'
    let mut p = Mat::from_fn(mm, mm, |i, j| if i >= j { p[(i, j)] } else { p[(j, i)] });
    let mut a = a;
    let mut regimes_: Vec<Regimes> = vec![regimes::base(nc); last + 1];
    let mut tt: Vec<Mat> = vec![pm.t.clone(); last + 1];
    let mut rr: Vec<Mat> = vec![pm.r.clone(); last + 1];
    let mut cc: Vec<Vector> = vec![Vector::zeros(mm); last + 1];
    let mut a0 = Mat::zeros(mm, last);
    let mut a1 = vec![Vector::zeros(mm); last];
    let mut p0 = vec![Mat::zeros(mm, mm); last];
    let mut p1 = vec![Mat::zeros(mm, mm); last];
    let mut vv = vec![Vector::zeros(pp); last];
    let mut lik = vec![0.0; last];
    let empty: Vec<usize> = Vec::new();
    let (mut ainit, mut pinit) = (Vector::zeros(mm), Mat::zeros(mm, mm));
    let fail = |lik: Vec<f64>, regimes: Vec<Regimes>, a0: Mat, t: usize, info: i32| PkfOutput {
        minus_loglik: f64::INFINITY,
        lik,
        regimes,
        a_updated: a0,
        fail_period: t,
        info,
    };
    for t in 1..=last {
        a1[t - 1] = a.clone();
        p1[t - 1] = p.clone();
        if t == 1 {
            pinit = p1[0].clone();
            ainit = a1[0].clone();
        }
        // (with no observation, MATLAB first predicts a and P with TT(:,:,t+1); the OccBin
        // update below replaces both)
        let d_index = &data_index[t - 1];
        let yt = Vector::from_fn(pp, |i, _| y[(i, t - 1)]);
        let x = if t == 1 {
            let v0 = Vector::zeros(pp);
            let inp = UpdateInput {
                a0: &ainit,
                a1_0: &ainit,
                p0: &pinit,
                p1_0: &p1[0],
                data_index: [&empty, &data_index[0]],
                mf,
                v0: &v0,
                y: &yt,
                h,
                q,
                tt: [&tt[1], &tt[0]],
                rr: [&rr[1], &rr[0]],
                cc: [&Vector::zeros(mm), &cc[0]],
                uni: None,
            };
            kalman_update_engine(pm, &inp, t, &regimes_[0..2], o)
        } else {
            let inp = UpdateInput {
                a0: &a0.column(t - 2).into_owned(),
                a1_0: &a1[t - 2],
                p0: &p0[t - 2],
                p1_0: &p1[t - 2],
                data_index: [&data_index[t - 2], &data_index[t - 1]],
                mf,
                v0: &vv[t - 2],
                y: &yt,
                h,
                q,
                tt: [&tt[t - 2], &tt[t - 1]],
                rr: [&rr[t - 2], &rr[t - 1]],
                cc: [&cc[t - 2], &cc[t - 1]],
                uni: None,
            };
            kalman_update_engine(pm, &inp, t, &regimes_[t - 1..t + 1], o)
        };
        if x.info != 0 {
            return fail(lik, regimes_, a0, t, x.info);
        }
        if x.lik.is_infinite() {
            return fail(lik, regimes_, a0, t, 0);
        }
        lik[t - 1] = x.lik;
        if regimes_.len() < t + 2 {
            regimes_.resize(t + 2, regimes::base(nc));
        }
        for (k, r) in x.regimes.iter().enumerate().take(3) {
            regimes_[t - 1 + k] = r.clone();
        }
        a0.set_column(t - 1, &x.a[0]);
        a1[t - 1] = x.a1[1].clone();
        a = x.a[1].clone();
        for &d in d_index {
            vv[t - 1][d] = x.v[1][d];
        }
        tt[t - 1] = x.t[0].clone();
        tt[t] = x.t[1].clone();
        rr[t - 1] = x.r[0].clone();
        rr[t] = x.r[1].clone();
        cc[t - 1] = x.c[0].clone();
        cc[t] = x.c[1].clone();
        p0[t - 1] = x.p[0].clone();
        p1[t - 1] = x.p1[1].clone();
        p = x.p[1].clone();
    }
    for l in lik.iter_mut() {
        *l *= 0.5;
    }
    let minus_loglik = lik[presample.min(last)..].iter().sum();
    PkfOutput { minus_loglik, lik, regimes: regimes_, a_updated: a0, fail_period: 0, info: 0 }
}
