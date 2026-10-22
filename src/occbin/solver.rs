// SPDX-License-Identifier: GPL-3.0-or-later
//! The piecewise-linear OccBin solver: ports of `+occbin/get_deriv.m`, `get_pq.m`,
//! `map_regime.m`, `mkdata.m`, `mkdatap_anticipated_dyn.m`,
//! `mkdatap_anticipated_2constraints_dyn.m`, `solve_one_constraint.m` and
//! `solve_two_constraints.m` (Dynare 7.2), with the same guess-and-verify iterations,
//! periodic-solution handling, endogenous extension of the check-ahead horizon and
//! error codes (310 periodic solution rejected, 311 maxit, 312 constraint binding at the end of
//! the horizon, 313 infinite loop of guesses).
//!
//! Like Dynare's `persistent` variables, the regime dictionary of `mkdatap_*` ([`Dictionary`])
//! lives as long as the decision rule it was built from (see `super::Session`).

use anyhow::Result;

use super::constraints::Constraints;
use crate::linalg::{self, Mat};
use crate::model::Model;
use crate::solver::DecisionRule;
use nalgebra::DVector;

/// First derivatives of one regime's dynamic model at the steady state
/// (`get_deriv.m`: `h_minus_1, h, h_plus_1, h_exo, resid`).
#[derive(Clone, Debug)]
pub struct RegimeMats {
    /// d/d y_{t-1} (`Cbarmat`).
    pub c: Mat,
    /// d/d y_t (`Bbarmat`).
    pub b: Mat,
    /// d/d y_{t+1} (`Abarmat`).
    pub a: Mat,
    /// d/d x_t (`Jbarmat`).
    pub j: Mat,
    /// Residuals at the steady state (`Dbarmat`).
    pub d: Vec<f64>,
}

/// Port of `get_deriv.m`.
pub fn get_deriv(m: &Model, ys: &[f64], params: &[f64]) -> RegimeMats {
    let n = m.n_endo();
    let ne = m.n_exo();
    let x = vec![0.0; ne];
    let g1 = m.dynamic_jacobian(ys, &x, params);
    let mut y3 = Vec::with_capacity(3 * n);
    for _ in 0..3 {
        y3.extend_from_slice(ys);
    }
    let mut d = vec![0.0; m.n_equations()];
    m.dynamic_residuals(&y3, &x, params, &mut d);
    RegimeMats {
        c: g1.columns(0, n).into_owned(),
        b: g1.columns(n, n).into_owned(),
        a: g1.columns(2 * n, n).into_owned(),
        j: g1.columns(3 * n, ne).into_owned(),
        d,
    }
}

/// Port of `get_pq.m`: the decision rule in declaration order, y_t = p y_{t-1} + q e_t.
pub fn get_pq(dr: &DecisionRule) -> (Mat, Mat) {
    let n = dr.ghx.nrows();
    let mut p = Mat::zeros(n, n);
    for (r, &i) in dr.order_var.iter().enumerate() {
        for (c, &j) in dr.state_var.iter().enumerate() {
            p[(i, j)] = dr.ghx[(r, c)];
        }
    }
    let mut q = Mat::zeros(n, dr.ghu.ncols());
    for i in 0..n {
        for k in 0..dr.ghu.ncols() {
            q[(i, k)] = dr.ghu[(dr.inv_order_var[i], k)];
        }
    }
    (p, q)
}

/// Dynare's `DM` structure: the regimes' derivatives and the baseline decision rule.
/// `regimes[k]`, `k = b1 + 2*b2` (b_i = constraint i binding): 0 is the baseline
/// (`*barmat`), 1 constraint 1 binding (`*starbarmat` / `*barmat10`), 2 constraint 2
/// binding (`*barmat01`), 3 both (`*barmat11`).
#[derive(Clone, Debug)]
pub struct Dm {
    pub n_vars: usize,
    pub n_exo: usize,
    pub regimes: Vec<RegimeMats>,
    pub decrulea: Mat,
    pub decruleb: Mat,
}

impl Dm {
    /// `pswitch`: parameter indices of the constraints (`M_.occbin.pswitch`, block order).
    pub fn new(m: &Model, ys: &[f64], params: &[f64], pswitch: &[usize], dr: &DecisionRule) -> Dm {
        let mut regimes = Vec::new();
        for k in 0..(1usize << pswitch.len()) {
            let mut p = params.to_vec();
            for (i, &ip) in pswitch.iter().enumerate() {
                if k & (1 << i) != 0 {
                    p[ip] = 1.0;
                }
            }
            regimes.push(get_deriv(m, ys, &p));
        }
        let (decrulea, decruleb) = get_pq(dr);
        Dm { n_vars: m.n_endo(), n_exo: m.n_exo(), regimes, decrulea, decruleb }
    }
}

/// One period's piecewise state space: y_t = T y_{t-1} + R e_t + C.
#[derive(Clone, Debug)]
pub struct StateSpace {
    pub t: Mat,
    pub r: Mat,
    pub c: Vec<f64>,
}

/// MATLAB `X \ [..]` for square X (LU with partial pivoting; like MATLAB, a singular X gives
/// Inf/NaN instead of an error).
fn solve_split(x: &Mat, rhs: &Mat, n: usize, ne: usize) -> StateSpace {
    let (f, ipiv, _) = linalg::lu(x);
    let mut s = rhs.clone();
    linalg::lu_solve(&f, &ipiv, false, &mut s);
    StateSpace {
        t: s.columns(0, n).into_owned(),
        r: s.columns(n, ne).into_owned(),
        c: s.column(n + ne).iter().copied().collect(),
    }
}

fn hcat(c: &Mat, j: &Mat, v: &[f64]) -> Mat {
    let n = c.nrows();
    let mut m = Mat::zeros(n, c.ncols() + j.ncols() + 1);
    m.columns_mut(0, c.ncols()).copy_from(c);
    m.columns_mut(c.ncols(), j.ncols()).copy_from(j);
    for i in 0..n {
        m[(i, c.ncols() + j.ncols())] = v[i];
    }
    m
}

/// Last binding period followed by the baseline rule:
/// `-(A*decrulea + B) \ [C J D]`.
fn first_step(m: &RegimeMats, decrulea: &Mat, ne: usize) -> StateSpace {
    let n = m.b.nrows();
    let x = -(&m.a * decrulea + &m.b);
    solve_split(&x, &hcat(&m.c, &m.j, &m.d), n, ne)
}

/// Backward step: `-(B + A*T_next) \ [C J A*C_next(+D)]`.
fn step(m: &RegimeMats, next: &StateSpace, with_d: bool, ne: usize) -> StateSpace {
    let n = m.b.nrows();
    let x = -(&m.b + &m.a * &next.t);
    let mut v: Vec<f64> = (0..n).map(|i| (0..n).map(|k| m.a[(i, k)] * next.c[k]).sum()).collect();
    if with_d {
        for i in 0..n {
            v[i] += m.d[i];
        }
    }
    solve_split(&x, &hcat(&m.c, &m.j, &v), n, ne)
}

/// The `persistent dictionary` of `mkdatap_anticipated_*`: regime-sequence keys (columns of
/// `dictionary.binding_indicator`; for two constraints, both columns stacked) and their
/// state space.
#[derive(Clone, Debug, Default)]
pub struct Dictionary {
    pub keys: Vec<Vec<bool>>,
    pub ss: Vec<StateSpace>,
}

/// Output of `mkdatap_anticipated_*`.
pub struct MkData {
    /// (n_periods+1) x n_vars: the expected path from the current period on.
    pub zdata: Mat,
    pub t: Mat,
    pub r: Mat,
    pub c: Vec<f64>,
    /// The state space of periods 1..T_max.
    pub ss: Vec<StateSpace>,
}

fn simulate(
    dm: &Dm,
    n_periods: usize,
    t_max: usize,
    seq: &[&StateSpace],
    exo_pos: &[usize],
    shock: &[f64],
    init: &[f64],
) -> MkData {
    let n = dm.n_vars;
    let mut hist = Mat::zeros(n, n_periods + 2);
    for i in 0..n {
        hist[(i, 0)] = init[i];
    }
    let mut err = vec![0.0; dm.n_exo];
    for (k, &e) in exo_pos.iter().enumerate() {
        err[e] = shock[k];
    }
    let err = DVector::from_column_slice(&err);
    let (t, r, c);
    if t_max >= 1 {
        let s = seq[0];
        let h = &s.t * hist.column(0) + DVector::from_column_slice(&s.c) + &s.r * &err;
        hist.column_mut(1).copy_from(&h);
        t = s.t.clone();
        c = s.c.clone();
        r = s.r.clone();
    } else {
        let h = &dm.decrulea * hist.column(0) + &dm.decruleb * &err;
        hist.column_mut(1).copy_from(&h);
        t = dm.decrulea.clone();
        c = vec![0.0; n];
        r = dm.decruleb.clone();
    }
    for irfpos in 2..=n_periods + 1 {
        let h = if irfpos <= t_max {
            let s = seq[irfpos - 1];
            &s.t * hist.column(irfpos - 1) + DVector::from_column_slice(&s.c)
        } else {
            &dm.decrulea * hist.column(irfpos - 1)
        };
        hist.column_mut(irfpos).copy_from(&h);
    }
    let zdata = hist.columns(1, n_periods + 1).transpose();
    MkData { zdata, t, r, c, ss: seq.iter().map(|s| (*s).clone()).collect() }
}

/// Port of `mkdatap_anticipated_dyn.m` (one constraint).
#[allow(clippy::too_many_arguments)]
pub fn mkdatap_anticipated(
    dict: &mut Option<Dictionary>,
    n_periods: usize,
    dm: &Dm,
    t_max: usize,
    binding_indicator: &[bool],
    exo_pos: &[usize],
    shock: &[f64],
    init: &[f64],
) -> MkData {
    let ne = dm.n_exo;
    let mut bi = binding_indicator.to_vec();
    if let Some(d) = dict.as_mut() {
        let l = d.keys[0].len();
        if bi.len() > l {
            for k in &mut d.keys {
                k.resize(bi.len(), false);
            }
        }
        if bi.len() < l {
            bi.resize(l, false);
        }
    }
    let mut ireg = vec![0usize; t_max];
    if t_max > 0 {
        if dict.is_none() {
            let mut key = vec![false; n_periods + 1];
            key[0] = true;
            let ss = first_step(&dm.regimes[1], &dm.decrulea, ne);
            *dict = Some(Dictionary { keys: vec![key], ss: vec![ss] });
        }
        let d = dict.as_mut().unwrap();
        ireg[t_max - 1] = 0;
        let len = bi.len();
        for i in (1..t_max).rev() {
            let mut tmp = vec![false; len];
            tmp[..len - i + 1].copy_from_slice(&bi[i - 1..]);
            if let Some(k) = d.keys.iter().position(|k| *k == tmp) {
                ireg[i - 1] = k;
            } else {
                let ss = if bi[i - 1] {
                    step(&dm.regimes[1], &d.ss[ireg[i]], true, ne)
                } else {
                    step(&dm.regimes[0], &d.ss[ireg[i]], false, ne)
                };
                d.keys.push(tmp);
                d.ss.push(ss);
                ireg[i - 1] = d.ss.len() - 1;
            }
        }
    }
    let seq: Vec<&StateSpace> = match dict.as_ref() {
        Some(d) => ireg.iter().map(|&k| &d.ss[k]).collect(),
        None => Vec::new(),
    };
    simulate(dm, n_periods, t_max, &seq, exo_pos, shock, init)
}

fn stack(rows: &[[bool; 2]]) -> Vec<bool> {
    rows.iter().map(|r| r[0]).chain(rows.iter().map(|r| r[1])).collect()
}

/// Regime of a row for the backward steps (`(1,0)`: 10, `(0,1)`: 01, `(1,1)`: 11, else base).
fn regime_index(r: [bool; 2]) -> usize {
    r[0] as usize + 2 * r[1] as usize
}

/// Regime of the last binding period (`T_max`): 10, 11, otherwise 01.
fn first_regime_index(r: [bool; 2]) -> usize {
    match r {
        [true, false] => 1,
        [true, true] => 3,
        _ => 2,
    }
}

/// Port of `mkdatap_anticipated_2constraints_dyn.m`. `binding_indicator` rows are periods.
#[allow(clippy::too_many_arguments)]
pub fn mkdatap_anticipated_2constraints(
    dict: &mut Option<Dictionary>,
    n_periods: usize,
    dm: &Dm,
    t_max: usize,
    binding_indicator: &[[bool; 2]],
    exo_pos: &[usize],
    shock: &[f64],
    init: &[f64],
) -> MkData {
    let ne = dm.n_exo;
    let mut bi = binding_indicator.to_vec();
    if let Some(d) = dict.as_mut() {
        let l2 = d.keys[0].len();
        let rows = bi.len().max(2); // MATLAB length() of a rows x 2 matrix
        if 2 * bi.len() > l2 {
            let old = l2 / 2;
            let pad = rows - old;
            for k in &mut d.keys {
                let mut nk = Vec::with_capacity(2 * (old + pad));
                nk.extend_from_slice(&k[..old]);
                nk.extend(std::iter::repeat_n(false, pad));
                nk.extend_from_slice(&k[old..2 * old]);
                nk.extend(std::iter::repeat_n(false, pad));
                *k = nk;
            }
        }
        if 2 * bi.len() < l2 {
            bi.resize(l2 / 2, [false, false]);
        }
    }
    let rows = bi.len();
    let suffix = |i: usize| -> Vec<bool> {
        // tmp = 0*bi; tmp(1:end-i+1,:) = bi(i:end,:)  (i 1-based)
        let mut tmp = vec![[false, false]; rows];
        tmp[..rows - i + 1].copy_from_slice(&bi[i - 1..]);
        stack(&tmp)
    };
    let mut ireg = vec![0usize; t_max];
    if t_max > 0 {
        let row = bi[t_max - 1];
        if dict.is_none() {
            let mut tmp = vec![[false, false]; n_periods + 1];
            tmp[0] = row;
            let ss = first_step(&dm.regimes[first_regime_index(row)], &dm.decrulea, ne);
            *dict = Some(Dictionary { keys: vec![stack(&tmp)], ss: vec![ss] });
            ireg[t_max - 1] = 0;
        } else {
            let d = dict.as_mut().unwrap();
            let key = suffix(t_max);
            if let Some(k) = d.keys.iter().position(|k| k[..key.len().min(k.len())] == key[..]) {
                ireg[t_max - 1] = k;
            } else {
                let mut tmp = vec![[false, false]; rows];
                tmp[0] = row;
                let ss = first_step(&dm.regimes[first_regime_index(row)], &dm.decrulea, ne);
                d.keys.push(stack(&tmp));
                d.ss.push(ss);
                ireg[t_max - 1] = d.ss.len() - 1;
            }
        }
        let d = dict.as_mut().unwrap();
        for i in (1..t_max).rev() {
            let key = suffix(i);
            if let Some(k) = d.keys.iter().position(|k| k[..key.len().min(k.len())] == key[..]) {
                ireg[i - 1] = k;
            } else {
                let ri = regime_index(bi[i - 1]);
                let ss = step(&dm.regimes[ri], &d.ss[ireg[i]], ri != 0, ne);
                d.keys.push(key);
                d.ss.push(ss);
                ireg[i - 1] = d.ss.len() - 1;
            }
        }
    }
    let seq: Vec<&StateSpace> = match dict.as_ref() {
        Some(d) => ireg.iter().map(|&k| &d.ss[k]).collect(),
        None => Vec::new(),
    };
    simulate(dm, n_periods, t_max, &seq, exo_pos, shock, init)
}

/// Port of `map_regime.m` for a logical vector: the regime values, their 1-based starting
/// periods, and the error flag (constraint binding at the end of the horizon).
pub fn map_regime(bi: &[bool]) -> (Vec<bool>, Vec<usize>, bool) {
    let first = bi.first().copied().unwrap_or(false);
    let mut regime = vec![first];
    let mut start = vec![1usize];
    for (i, &b) in bi.iter().enumerate().skip(1) {
        if b != *regime.last().unwrap() {
            regime.push(b);
            start.push(i + 1);
        }
    }
    let err = (regime[0] && start.len() == 1) || *regime.last().unwrap();
    (regime, start, err)
}

/// Port of `mkdata.m`: the linear path y_t = A y_{t-1} + B e_t, rows = periods.
pub fn mkdata(n_periods: usize, a: &Mat, b: &Mat, exo_pos: &[usize], shocks: &Mat, init: &[f64]) -> Mat {
    let n = a.nrows();
    let mut out = Mat::zeros(n_periods, n);
    let mut h = Mat::from_column_slice(n, 1, init);
    let mut err = Mat::zeros(b.ncols(), 1);
    for i in 0..n_periods {
        h = if i < shocks.nrows() {
            for (k, &e) in exo_pos.iter().enumerate() {
                err[e] = shocks[(i, k)];
            }
            a * &h + b * &err
        } else {
            a * &h
        };
        for j in 0..n {
            out[(i, j)] = h[j];
        }
    }
    out
}

/// `options_.occbin.simul` (defaults of `occbin.set_default_options(...,'simul')`).
#[derive(Clone, Debug)]
pub struct SimulOptions {
    pub periods: usize,
    pub maxit: usize,
    pub check_ahead_periods: usize,
    /// None = Inf.
    pub max_check_ahead_periods: Option<usize>,
    pub curb_retrench: bool,
    pub debug: bool,
    pub periodic_solution: bool,
    pub periodic_solution_threshold: f64,
    pub periodic_solution_strict: bool,
    pub piecewise_only: bool,
    pub reset_check_ahead_periods_in_new_period: bool,
    pub reset_regime_in_new_period: bool,
    pub algo_truncation: usize,
    /// Initial state, deviations from the steady state (declaration order); empty = zeros.
    pub endo_init: Vec<f64>,
    /// Initial guess of the binding periods (rows = periods, one column per constraint).
    pub init_binding_indicator: Vec<Vec<bool>>,
    /// Guessed regime history (as `init_regime`), one entry per shock period.
    pub init_regime: Option<RegimeHistory>,
    /// `SHOCKS`: shock periods x shocks in `exo_pos`.
    pub shocks: Mat,
    /// 0-based exogenous indices of the columns of `shocks`.
    pub exo_pos: Vec<usize>,
}

impl SimulOptions {
    pub fn new(n_exo: usize) -> Self {
        SimulOptions {
            periods: 100,
            maxit: 30,
            check_ahead_periods: 200,
            max_check_ahead_periods: None,
            curb_retrench: false,
            debug: false,
            periodic_solution: false,
            periodic_solution_threshold: 1.0,
            periodic_solution_strict: true,
            piecewise_only: false,
            reset_check_ahead_periods_in_new_period: false,
            reset_regime_in_new_period: false,
            algo_truncation: 1,
            endo_init: Vec::new(),
            init_binding_indicator: Vec::new(),
            init_regime: None,
            shocks: Mat::zeros(1, n_exo),
            exo_pos: (0..n_exo).collect(),
        }
    }
}

/// `regime` / `regimestart` of one constraint after the shock of a period.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Regime {
    pub regime: Vec<bool>,
    /// 1-based expected starting periods of each regime.
    pub start: Vec<usize>,
}

/// `oo_.occbin.simul.regime_history`: per shock period, one [`Regime`] per constraint
/// (`regime`/`regimestart`, or `regime1`, `regimestart1`, `regime2`, `regimestart2`).
pub type RegimeHistory = Vec<Vec<Regime>>;

/// Result of `solve_one_constraint` / `solve_two_constraints` (deviations from the steady state).
#[derive(Clone, Debug)]
pub struct SolveOutput {
    /// periods x n_vars.
    pub piecewise: Mat,
    pub linear: Option<Mat>,
    pub regime_history: RegimeHistory,
    /// Per shock period: `SS_out.T/R/C`.
    pub ss: Vec<StateSpace>,
}

/// A failure with Dynare's error code (`print_info` texts).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("OccBin error {code}: {}", describe(*.code))]
pub struct OccbinError {
    pub code: i32,
    /// 1-based shock period where it happened.
    pub period: usize,
}

pub fn describe(code: i32) -> &'static str {
    match code {
        310 => "the solver loops between two regimes (periodic solution; use simul_periodic_solution to accept it)",
        311 => "the solver did not converge (increase maxit)",
        312 => "the constraint is binding at the end of the horizon (increase check_ahead_periods / periods)",
        313 => "the solver did not converge: infinite loop of guess regimes",
        _ => "OccBin solver error",
    }
}

fn disp(debug: bool, s: &str) {
    if debug {
        println!("{s}");
    }
}

/// Fills `bi` from a guessed regime history entry (`binding_indicator(:,k) = regime(end)` then
/// the earlier regimes over their spans); MATLAB grows the vector if the spans exceed it.
fn apply_guess(col: &mut Vec<bool>, g: &Regime) {
    let last = *g.regime.last().unwrap_or(&false);
    col.iter_mut().for_each(|b| *b = last);
    for ir in 0..g.regime.len().saturating_sub(1) {
        let (a, b) = (g.start[ir], g.start[ir + 1]);
        if b > 1 && b - 1 > col.len() {
            col.resize(b - 1, false);
        }
        for t in a..b {
            col[t - 1] = g.regime[ir];
        }
    }
}

fn row_has_shock(shocks: &Mat, sp: usize) -> bool {
    (0..shocks.ncols()).any(|k| shocks[(sp - 1, k)] != 0.0)
}

fn shock_row(shocks: &Mat, sp: usize) -> Vec<f64> {
    (0..shocks.ncols()).map(|k| shocks[(sp - 1, k)]).collect()
}

/// Inputs shared by both solvers.
pub struct Problem<'a> {
    pub dm: &'a Dm,
    pub constraints: &'a Constraints,
    pub params: &'a [f64],
    pub ys: &'a [f64],
}

fn shift_no_shock(z: &mut Mat, a: &Mat) {
    let nr = z.nrows();
    for t in 0..nr - 1 {
        for j in 0..z.ncols() {
            z[(t, j)] = z[(t + 1, j)];
        }
    }
    let last: Vec<f64> = (0..z.ncols()).map(|j| z[(nr - 1, j)]).collect();
    let v = a * Mat::from_column_slice(last.len(), 1, &last);
    for j in 0..z.ncols() {
        z[(nr - 1, j)] = v[j];
    }
}

fn ss_after_shift(ss: &mut Vec<StateSpace>, dm: &Dm) -> StateSpace {
    if ss.len() > 1 {
        ss.remove(0);
    } else {
        ss.clear();
    }
    match ss.first() {
        None => StateSpace { t: dm.decrulea.clone(), r: dm.decruleb.clone(), c: vec![0.0; dm.n_vars] },
        Some(s) => s.clone(),
    }
}

/// MATLAB `max` of a vector: NaN ignored unless all are NaN.
fn nanmax(v: impl Iterator<Item = f64>) -> f64 {
    let mut m = f64::NAN;
    for x in v {
        if m.is_nan() || x > m {
            if !x.is_nan() {
                m = x;
            }
        }
    }
    m
}

fn argmin_first(v: impl Iterator<Item = (usize, f64)>) -> usize {
    // MATLAB min: first minimum, NaN ignored
    let mut best: Option<(usize, f64)> = None;
    for (k, x) in v {
        if x.is_nan() {
            continue;
        }
        if best.is_none_or(|(_, b)| x < b) {
            best = Some((k, x));
        }
    }
    best.map(|b| b.0).unwrap_or(0)
}

/// Port of `solve_one_constraint.m`.
pub fn solve_one_constraint(
    pb: &Problem,
    dict: &mut Option<Dictionary>,
    o: &SimulOptions,
) -> Result<SolveOutput, OccbinError> {
    let dm = pb.dm;
    let shocks = &o.shocks;
    let n_sp = shocks.nrows();
    let n_periods = o.periods.max(n_sp);
    let mut nperiods_0 = o.check_ahead_periods.max(n_periods - n_sp);
    let maxit = o.maxit;
    let mut endo_init = if o.endo_init.is_empty() { vec![0.0; dm.n_vars] } else { o.endo_init.clone() };
    let init_orig = endo_init.clone();
    let mut piecewise = Mat::zeros(n_periods, dm.n_vars);
    let mut bi: Vec<bool> = o.init_binding_indicator.iter().map(|r| r.first().copied().unwrap_or(false)).collect();
    if bi.len() < nperiods_0 + 1 {
        bi.resize(nperiods_0 + 1, false);
    }
    let guess = o.init_regime.as_ref();
    let mut regime_history: RegimeHistory = guess.cloned().unwrap_or_default();
    let mut ss_out: Vec<Option<StateSpace>> = vec![None; n_sp];
    let mut zdatalinear = Mat::zeros(0, dm.n_vars);
    let mut ss_seq: Vec<StateSpace> = Vec::new();
    let mut ecp = false;
    let maxc = o.max_check_ahead_periods;

    for sp in 1..=n_sp {
        let mut regime_change = true;
        let mut iter = 0usize;
        let mut increased = false;
        let guess_it = guess.is_some_and(|g| sp <= g.len());
        let mut is_periodic = false;
        let mut is_periodic_loop = false;
        let mut history: Vec<Vec<bool>> = Vec::new();
        let mut max_err = vec![f64::NAN; maxit.max(1)];
        let mut viol_exp = vec![false; maxit.max(1)];
        let mut nviol = vec![0.0f64; maxit.max(1)];
        let mut merr = f64::NAN;
        let set_hist = |rh: &mut RegimeHistory, sp: usize, r: Regime| {
            if rh.len() < sp {
                rh.resize(sp, vec![Regime::default()]);
            }
            rh[sp - 1] = vec![r];
        };
        while regime_change && iter < maxit && !is_periodic && !is_periodic_loop {
            iter += 1;
            if *bi.last().unwrap() && maxc.is_none_or(|m| nperiods_0 < m) {
                bi.push(false);
                nperiods_0 += 1;
                increased = true;
                disp(o.debug, &format!("nperiods has been endogenously increased up to {nperiods_0}."));
            } else if let Some(m) = maxc.filter(|&m| nperiods_0 >= m) {
                bi.truncate(m + 1);
                *bi.last_mut().unwrap() = false;
            }
            if bi.len() < nperiods_0 + 1 {
                bi.resize(nperiods_0 + 1, false);
            }
            if iter == 1 && guess_it {
                apply_guess(&mut bi, &guess.unwrap()[sp - 1][0]);
                nperiods_0 = bi.len() - 1;
            }
            history.push(bi.clone());
            let (regime, start, e) = map_regime(&bi);
            ecp = e;
            set_hist(&mut regime_history, sp, Regime { regime, start: start.clone() });
            let mut start = start;
            if sp == 1 || row_has_shock(shocks, sp) {
                if iter == 1 && o.reset_regime_in_new_period {
                    if o.reset_check_ahead_periods_in_new_period {
                        nperiods_0 = o.check_ahead_periods.max(n_periods - n_sp);
                        bi = vec![false; nperiods_0 + 1];
                    } else {
                        bi.iter_mut().for_each(|b| *b = false);
                    }
                    history[iter - 1] = bi.clone();
                    let (regime, s, e) = map_regime(&bi);
                    ecp = e;
                    set_hist(&mut regime_history, sp, Regime { regime, start: s.clone() });
                    start = s;
                }
                let mk = mkdatap_anticipated(
                    dict,
                    nperiods_0,
                    dm,
                    start.last().unwrap() - 1,
                    &bi,
                    &o.exo_pos,
                    &shock_row(shocks, sp),
                    &endo_init,
                );
                zdatalinear = mk.zdata;
                ss_out[sp - 1] = Some(StateSpace { t: mk.t, r: mk.r, c: mk.c });
                ss_seq = mk.ss;
                let diff = pb.constraints.difference(&zdatalinear, pb.params, pb.ys);
                let (b, r) = (&diff.binding[0], &diff.relax[0]);
                let len = bi.len();
                let (end, n_last) = match maxc {
                    Some(m) if m < len => (m, len - m),
                    _ => (len, 0),
                };
                let viol: Vec<usize> = (0..end).filter(|&t| b[t] && !bi[t]).collect();
                let rel: Vec<usize> = (0..end).filter(|&t| r[t] && bi[t]).collect();
                if !viol.is_empty() || !rel.is_empty() {
                    let m = viol
                        .iter()
                        .map(|&t| diff.err_binding[0][t])
                        .chain(rel.iter().map(|&t| diff.err_relax[0][t]))
                        .map(f64::abs);
                    let m = nanmax(m);
                    max_err[iter - 1] = m;
                    regime_change = true;
                    viol_exp[iter - 1] = !viol.is_empty();
                } else {
                    regime_change = false;
                    max_err[iter - 1] = 0.0;
                }
                let bcn: Vec<bool> = (0..end).map(|t| b[t]).chain(std::iter::repeat_n(false, n_last)).collect();
                let rcn: Vec<bool> = (0..end).map(|t| r[t]).chain(std::iter::repeat_n(true, n_last)).collect();
                let new_bi: Vec<bool> = (0..len).map(|t| (bi[t] || bcn[t]) && !(bi[t] && rcn[t])).collect();
                nviol[iter - 1] = (0..len).map(|t| bi[t] as i32 as f64 - new_bi[t] as i32 as f64).sum();
                if o.curb_retrench {
                    let mut retrench = vec![false; len];
                    let max_relax = (0..end).rev().find(|&t| r[t] && bi[t]);
                    if let Some(mr) = max_relax {
                        let lr = (0..end).rev().find(|&t| r[t]);
                        let lb = (0..end).rev().find(|&t| bi[t]);
                        if lr >= lb {
                            retrench[mr] = true;
                        }
                    }
                    bi = (0..len).map(|t| (bi[t] || bcn[t]) && !retrench[t]).collect();
                } else {
                    bi = new_bi;
                }
                if iter > 1 && regime_change && !increased {
                    let loop_all: Vec<bool> = history[..iter - 1].iter().map(|h| *h == bi).collect();
                    is_periodic_loop = loop_all.iter().any(|&x| x);
                    let mut per_all = vec![false; iter - 1];
                    {
                        let k = iter - 2;
                        let h = &history[k];
                        let s: f64 =
                            history[iter - 1].iter().zip(&bi).map(|(&a, &b)| a as i32 as f64 - b as i32 as f64).sum();
                        per_all[k] = *h == bi && s <= o.periodic_solution_threshold;
                    }
                    is_periodic = per_all.iter().any(|&x| x);
                    let strict = is_periodic;
                    if is_periodic_loop
                        && !is_periodic
                        && !o.periodic_solution_strict
                        && (0..iter).any(|t| !viol_exp[t] && nviol[t] <= o.periodic_solution_threshold)
                    {
                        is_periodic = true;
                        per_all = vec![false; loop_all.len()];
                        per_all[0] = true;
                    }
                    if is_periodic && o.periodic_solution {
                        let first = per_all.iter().position(|&x| x).unwrap() + 1;
                        let mut inx: Vec<usize> = (first..=iter).collect();
                        let inx1: Vec<usize> = inx.iter().copied().filter(|&t| !viol_exp[t - 1]).collect();
                        let inx1_empty = inx1.is_empty();
                        if !inx1_empty {
                            inx = inx1;
                        }
                        let k = if strict || inx1_empty {
                            argmin_first(inx.iter().enumerate().map(|(k, &t)| (k, max_err[t - 1])))
                        } else {
                            argmin_first(inx.iter().enumerate().map(|(k, &t)| (k, nviol[t - 1])))
                        };
                        let sel = inx[k];
                        merr = max_err[sel - 1];
                        bi = history[sel - 1].clone();
                        if sel < iter {
                            let (regime, s, e) = map_regime(&bi);
                            ecp = e;
                            set_hist(&mut regime_history, sp, Regime { regime, start: s.clone() });
                            let mk = mkdatap_anticipated(
                                dict,
                                nperiods_0,
                                dm,
                                s.last().unwrap() - 1,
                                &bi,
                                &o.exo_pos,
                                &shock_row(shocks, sp),
                                &endo_init,
                            );
                            zdatalinear = mk.zdata;
                            ss_out[sp - 1] = Some(StateSpace { t: mk.t, r: mk.r, c: mk.c });
                            ss_seq = mk.ss;
                        }
                    }
                }
            } else {
                regime_change = false;
                shift_no_shock(&mut zdatalinear, &dm.decrulea);
                ss_out[sp - 1] = Some(ss_after_shift(&mut ss_seq, dm));
                history[iter - 1] = bi.clone();
            }
        }
        if regime_change {
            if maxit > o.algo_truncation {
                disp(o.debug, &format!("occbin solver:: period {sp}::"));
                if is_periodic {
                    disp(o.debug, "OccBin solver loops between two regimes.");
                    if o.periodic_solution {
                        disp(o.debug, &format!("Max error:{merr}."));
                    } else {
                        return Err(OccbinError { code: 310, period: sp });
                    }
                } else {
                    let code = if is_periodic_loop { 313 } else { 311 };
                    return Err(OccbinError { code, period: sp });
                }
            } else {
                bi = history.last().unwrap().clone();
            }
        }
        if ecp {
            disp(o.debug, "Increase nperiods.");
            return Err(OccbinError { code: 312, period: sp });
        }
        for j in 0..dm.n_vars {
            endo_init[j] = zdatalinear[(0, j)];
            piecewise[(sp - 1, j)] = endo_init[j];
        }
        bi.remove(0);
        bi.push(false);
    }
    for t in n_sp..n_periods {
        for j in 0..dm.n_vars {
            piecewise[(t, j)] = zdatalinear[(t - n_sp + 1, j)];
        }
    }
    let linear =
        (!o.piecewise_only).then(|| mkdata(n_periods, &dm.decrulea, &dm.decruleb, &o.exo_pos, shocks, &init_orig));
    Ok(SolveOutput { piecewise, linear, regime_history, ss: ss_out.into_iter().map(Option::unwrap).collect() })
}

/// Port of `solve_two_constraints.m`.
pub fn solve_two_constraints(
    pb: &Problem,
    dict: &mut Option<Dictionary>,
    o: &SimulOptions,
) -> Result<SolveOutput, OccbinError> {
    let dm = pb.dm;
    let shocks = &o.shocks;
    let n_sp = shocks.nrows();
    let n_periods = o.periods.max(n_sp);
    let mut nperiods_0 = o.check_ahead_periods.max(n_periods - n_sp);
    let maxit = o.maxit;
    let mut endo_init = if o.endo_init.is_empty() { vec![0.0; dm.n_vars] } else { o.endo_init.clone() };
    let init_orig = endo_init.clone();
    let mut piecewise = Mat::zeros(n_periods, dm.n_vars);
    let mut bi: Vec<[bool; 2]> = o
        .init_binding_indicator
        .iter()
        .map(|r| [r.first().copied().unwrap_or(false), r.get(1).copied().unwrap_or(false)])
        .collect();
    if bi.len() < nperiods_0 + 1 {
        bi.resize(nperiods_0 + 1, [false, false]);
    }
    let guess = o.init_regime.as_ref();
    let mut regime_history: RegimeHistory = guess.cloned().unwrap_or_default();
    let mut ss_out: Vec<Option<StateSpace>> = vec![None; n_sp];
    let mut zdatalinear = Mat::zeros(0, dm.n_vars);
    let mut ss_seq: Vec<StateSpace> = Vec::new();
    let mut ecp = [false, false];
    let maxc = o.max_check_ahead_periods;
    let set_hist = |rh: &mut RegimeHistory, sp: usize, r: Vec<Regime>| {
        if rh.len() < sp {
            rh.resize(sp, vec![Regime::default(); 2]);
        }
        rh[sp - 1] = r;
    };
    let map2 = |bi: &[[bool; 2]], ecp: &mut [bool; 2]| -> (Vec<Regime>, usize) {
        let mut out = Vec::new();
        let mut tmax = 0;
        for k in 0..2 {
            let col: Vec<bool> = bi.iter().map(|r| r[k]).collect();
            let (regime, start, e) = map_regime(&col);
            ecp[k] = e;
            tmax = tmax.max(*start.last().unwrap());
            out.push(Regime { regime, start });
        }
        (out, tmax - 1)
    };

    for sp in 1..=n_sp {
        let mut regime_change = true;
        let mut increased = false;
        let mut iter = 0usize;
        let guess_it = guess.is_some_and(|g| sp <= g.len());
        let mut is_periodic = false;
        let mut is_periodic_loop = false;
        let mut history: Vec<Vec<[bool; 2]>> = Vec::new();
        let mut max_err = vec![f64::NAN; maxit.max(1)];
        let mut viol_exp = vec![false; maxit.max(1)];
        let mut nviol = vec![0.0f64; maxit.max(1)];
        let mut min_err = f64::NAN;
        while regime_change && iter < maxit && !is_periodic && !is_periodic_loop {
            iter += 1;
            let last = *bi.last().unwrap();
            if (last[0] || last[1]) && maxc.is_none_or(|m| nperiods_0 < m) {
                bi.push([false, false]);
                nperiods_0 += 1;
                increased = true;
                disp(o.debug, &format!("nperiods has been endogenously increased up to {nperiods_0}."));
            } else if let Some(m) = maxc.filter(|&m| nperiods_0 >= m) {
                bi.truncate(m + 1);
                *bi.last_mut().unwrap() = [false, false];
            }
            if bi.len() < nperiods_0 + 1 {
                bi.resize(nperiods_0 + 1, [false, false]);
            }
            if iter == 1 && guess_it {
                let g = &guess.unwrap()[sp - 1];
                for k in 0..2 {
                    let mut col: Vec<bool> = bi.iter().map(|r| r[k]).collect();
                    apply_guess(&mut col, &g[k]);
                    if col.len() > bi.len() {
                        bi.resize(col.len(), [false, false]);
                    }
                    for (t, &v) in col.iter().enumerate() {
                        bi[t][k] = v;
                    }
                }
                nperiods_0 = bi.len() - 1;
            }
            history.push(bi.clone());
            let (rh, mut tmax) = map2(&bi, &mut ecp);
            set_hist(&mut regime_history, sp, rh);
            if sp == 1 || row_has_shock(shocks, sp) {
                if iter == 1 && o.reset_regime_in_new_period {
                    if o.reset_check_ahead_periods_in_new_period {
                        nperiods_0 = o.check_ahead_periods.max(n_periods - n_sp);
                        bi = vec![[false, false]; nperiods_0 + 1];
                    } else {
                        bi.iter_mut().for_each(|b| *b = [false, false]);
                    }
                    history[iter - 1] = bi.clone();
                    let (rh, t) = map2(&bi, &mut ecp);
                    set_hist(&mut regime_history, sp, rh);
                    tmax = t;
                }
                let mk = mkdatap_anticipated_2constraints(
                    dict,
                    nperiods_0,
                    dm,
                    tmax,
                    &bi,
                    &o.exo_pos,
                    &shock_row(shocks, sp),
                    &endo_init,
                );
                zdatalinear = mk.zdata;
                ss_out[sp - 1] = Some(StateSpace { t: mk.t, r: mk.r, c: mk.c });
                ss_seq = mk.ss;
                let diff = pb.constraints.difference(&zdatalinear, pb.params, pb.ys);
                let len = bi.len().max(2);
                let rows = bi.len();
                let (end, n_last) = match maxc {
                    Some(m) if m < len => (m, len - m),
                    _ => (len, 0),
                };
                // stacked [constraint 1 (1:end); constraint 2 (1:end)]
                let mut viol_err = Vec::new();
                let mut any_viol = false;
                let mut any_rel = false;
                for k in 0..2 {
                    for t in 0..end {
                        if diff.binding[k][t] && !bi[t][k] {
                            any_viol = true;
                            viol_err.push(diff.err_binding[k][t]);
                        }
                    }
                }
                let mut rel_err = Vec::new();
                for k in 0..2 {
                    for t in 0..end {
                        if diff.relax[k][t] && bi[t][k] {
                            any_rel = true;
                            rel_err.push(diff.err_relax[k][t]);
                        }
                    }
                }
                if any_viol || any_rel {
                    let m = nanmax(viol_err.iter().chain(&rel_err).map(|x| x.abs()));
                    max_err[iter - 1] = m;
                    regime_change = true;
                    viol_exp[iter - 1] = any_viol;
                } else {
                    regime_change = false;
                    max_err[iter - 1] = 0.0;
                }
                let col = |v: &Vec<bool>, fill: bool| -> Vec<bool> {
                    (0..end).map(|t| v[t]).chain(std::iter::repeat_n(fill, n_last)).collect()
                };
                let bcn = [col(&diff.binding[0], false), col(&diff.binding[1], false)];
                let rcn = [col(&diff.relax[0], true), col(&diff.relax[1], true)];
                let mut tmp_nper = [0.0f64; 2];
                for k in 0..2 {
                    // sum(bi - (bi | b) & ~(bi & r)): '-' binds tighter than '&'
                    tmp_nper[k] = (0..rows)
                        .filter(|&t| {
                            let d = bi[t][k] as i32 - (bi[t][k] || bcn[k][t]) as i32;
                            d != 0 && !(bi[t][k] && rcn[k][t])
                        })
                        .count() as f64;
                }
                nviol[iter - 1] = tmp_nper[0].max(tmp_nper[1]);
                if o.curb_retrench {
                    let mut retrench = [vec![false; rows], vec![false; rows]];
                    for k in 0..2 {
                        let r = &diff.relax[k];
                        if let Some(mr) = (0..end).rev().find(|&t| r[t] && bi[t][k]) {
                            let lr = (0..end).rev().find(|&t| r[t]);
                            let lb = (0..end).rev().find(|&t| bi[t][k]);
                            if lr >= lb {
                                retrench[k][mr] = true;
                            }
                        }
                    }
                    for t in 0..rows {
                        for k in 0..2 {
                            bi[t][k] = (bi[t][k] || bcn[k][t]) && !retrench[k][t];
                        }
                    }
                } else {
                    for t in 0..rows {
                        for k in 0..2 {
                            bi[t][k] = (bi[t][k] || bcn[k][t]) && !(bi[t][k] && rcn[k][t]);
                        }
                    }
                }
                if iter > 1 && regime_change && !increased {
                    let loop_all: Vec<bool> = history[..iter - 1].iter().map(|h| *h == bi).collect();
                    is_periodic_loop = loop_all.iter().any(|&x| x);
                    let mut per_all = vec![false; iter - 1];
                    {
                        let k = iter - 2;
                        let cur = &history[iter - 1];
                        let s = |c: usize| -> f64 {
                            cur.iter().zip(&bi).map(|(a, b)| a[c] as i32 as f64 - b[c] as i32 as f64).sum()
                        };
                        per_all[k] = history[k] == bi
                            && s(0) <= o.periodic_solution_threshold
                            && s(1) <= o.periodic_solution_threshold;
                    }
                    is_periodic = per_all.iter().any(|&x| x);
                    let strict = is_periodic;
                    if is_periodic_loop
                        && !is_periodic
                        && !o.periodic_solution_strict
                        && (0..iter).any(|t| !viol_exp[t] && nviol[t] < o.periodic_solution_threshold)
                    {
                        is_periodic = true;
                        per_all = vec![false; loop_all.len()];
                        per_all[0] = true;
                    }
                    if is_periodic && o.periodic_solution {
                        let first = per_all.iter().position(|&x| x).unwrap() + 1;
                        let mut inx: Vec<usize> = (first..=iter).collect();
                        let inx1: Vec<usize> = inx.iter().copied().filter(|&t| !viol_exp[t - 1]).collect();
                        let inx1_empty = inx1.is_empty();
                        if !inx1_empty {
                            inx = inx1;
                        }
                        let k = if strict || inx1_empty {
                            argmin_first(inx.iter().enumerate().map(|(k, &t)| (k, max_err[t - 1])))
                        } else {
                            argmin_first(inx.iter().enumerate().map(|(k, &t)| (k, nviol[t - 1])))
                        };
                        let sel = inx[k];
                        min_err = max_err[sel - 1];
                        bi = history[sel - 1].clone();
                        if sel < iter {
                            let (rh, t) = map2(&bi, &mut ecp);
                            set_hist(&mut regime_history, sp, rh);
                            let mk = mkdatap_anticipated_2constraints(
                                dict,
                                nperiods_0,
                                dm,
                                t,
                                &bi,
                                &o.exo_pos,
                                &shock_row(shocks, sp),
                                &endo_init,
                            );
                            zdatalinear = mk.zdata;
                            ss_out[sp - 1] = Some(StateSpace { t: mk.t, r: mk.r, c: mk.c });
                            ss_seq = mk.ss;
                        }
                    }
                }
            } else {
                regime_change = false;
                shift_no_shock(&mut zdatalinear, &dm.decrulea);
                ss_out[sp - 1] = Some(ss_after_shift(&mut ss_seq, dm));
                history[iter - 1] = bi.clone();
            }
        }
        if regime_change {
            if maxit > o.algo_truncation {
                disp(o.debug, &format!("occbin solver: period {sp}:"));
                if is_periodic {
                    disp(o.debug, "OccBin solver loops between two regimes.");
                    if o.periodic_solution {
                        disp(o.debug, &format!("Max error:{min_err}."));
                    } else {
                        return Err(OccbinError { code: 310, period: sp });
                    }
                } else {
                    let code = if is_periodic_loop { 313 } else { 311 };
                    return Err(OccbinError { code, period: sp });
                }
            } else {
                bi = history.last().unwrap().clone();
            }
        }
        if ecp[0] || ecp[1] {
            disp(o.debug, "Increase nperiods.");
            return Err(OccbinError { code: 312, period: sp });
        }
        for j in 0..dm.n_vars {
            endo_init[j] = zdatalinear[(0, j)];
            piecewise[(sp - 1, j)] = endo_init[j];
        }
        bi.remove(0);
        bi.push([false, false]);
    }
    for t in n_sp..n_periods {
        for j in 0..dm.n_vars {
            piecewise[(t, j)] = zdatalinear[(t - n_sp + 1, j)];
        }
    }
    let linear =
        (!o.piecewise_only).then(|| mkdata(n_periods, &dm.decrulea, &dm.decruleb, &o.exo_pos, shocks, &init_orig));
    Ok(SolveOutput { piecewise, linear, regime_history, ss: ss_out.into_iter().map(Option::unwrap).collect() })
}
