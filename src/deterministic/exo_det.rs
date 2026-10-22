// SPDX-License-Identifier: GPL-3.0-or-later
//! deterministic exogenous variables (`varexo_det`), as Dynare 7.2 handles them.
//!
//! The `varexo_det` variables are out of `M_.exo_names` / `M_.Sigma_e` and enter the model at
//! their steady-state value `oo_.exo_det_steady_state`: dynare-rust makes them parameters of
//! the model (`ModelBuilder::q1_exo_det`), so the steady state, `check`, the decision rules
//! (`ghx`, `ghu`, `ghxx`, ...), the IRFs, the `periods` simulations (`simult_.m` ignores them)
//! and the estimation (a constant of the state space) are Dynare's. Their paths matter only for
//! forecasts (`forecasts.run.m` -> `simult_varexo_det.m`), through
//! - `dr.ghud` (order 1) and `dr.ghxud`, `dr.ghuud`, `dr.ghudud` (order 2), the
//!   "exogenous deterministic variables" section of `stochastic_solvers.m`;
//! - `oo_.exo_det_simul` (`make_ex_.m`, then the steady-state extension of `forecasts.run`).
//!
//! The errors Dynare raises are here too: purely forward / backward models, the k-order
//! solver (order 3), `extended_path`; the preprocessor's are in `modfile::parser_q1`.

use crate::forecast::Forecast;
use crate::linalg::Mat;
use crate::model::Model;
use crate::model::expr::{DiffVar, Node};
use crate::modfile::{DetShockKind, ModFile};
use crate::solver::DecisionRule;
use crate::solver::korder::KOrderDr;
use std::collections::HashMap;

/// The errors Dynare raises on a model with `varexo_det` before running command `c`
/// (`stochastic_solvers.m`: purely forward / purely backward models; `extended_path`).
pub fn command_check(mf: &ModFile, c: &crate::modfile::Command) -> anyhow::Result<()> {
    use crate::modfile::Command;
    if mf.model.exo_det_names.is_empty() {
        return Ok(());
    }
    // stochastic_solvers.m: `max(dr.exo_det_length, max(M_.det_shocks(i).periods))` fails on
    // shocks given as dates (every solve but check's)
    let dates = mf.q1.exo_det_shocks.iter().chain(&mf.det_shocks).any(|d| !d.dates.is_empty());
    let solves = matches!(
        c,
        Command::StochSimul(..)
            | Command::Estimation(..)
            | Command::Forecast(..)
            | Command::ConditionalForecast(..)
            | Command::ShockDecomposition(..)
            | Command::RealtimeShockDecomposition(..)
            | Command::CalibSmoother(..)
    );
    if dates && solves {
        solver_check(&mf.model)?;
        anyhow::bail!("All input arguments must be dates objects.");
    }
    match c {
        Command::Other(n, _) if n == "extended_path" => anyhow::bail!("Extended path does not support varexo_det."),
        Command::StochSimul(..)
        | Command::Check(..)
        | Command::Estimation(..)
        | Command::Forecast(..)
        | Command::ConditionalForecast(..)
        | Command::ShockDecomposition(..)
        | Command::RealtimeShockDecomposition(..)
        | Command::CalibSmoother(..) => solver_check(&mf.model),
        Command::Other(n, _) if n == "identification" || n == "dynare_sensitivity" || n == "sensitivity" => {
            solver_check(&mf.model)
        }
        _ => Ok(()),
    }
}

/// `stochastic_solvers.m`: no `varexo_det` in purely forward or purely backward models.
pub fn solver_check(model: &Model) -> anyhow::Result<()> {
    if model.exo_det_names.is_empty() {
        return Ok(());
    }
    let lli = &model.lead_lag_incidence;
    if !lli.iter().any(|r| r[0] > 0) {
        println!("\nSTOCHASTIC_SOLVER: Dynare does not solve purely forward models with var_exo_det.");
        println!(
            "STOCHASTIC_SOLVER: To circumvent this restriction, you can add a backward-looking dummy equation of the form:"
        );
        println!("STOCHASTIC_SOLVER: junk=0.9*junk(-1);");
        anyhow::bail!("var_exo_det not implemented for purely forward models");
    }
    if !lli.iter().any(|r| r[2] > 0) {
        println!("\nSTOCHASTIC_SOLVER: Dynare does not solve purely backward models with var_exo_det.");
        println!(
            "STOCHASTIC_SOLVER: To circumvent this restriction, you can add a foward-looking dummy equation of the form:"
        );
        println!("STOCHASTIC_SOLVER: junk=0.9*junk(+1);");
        anyhow::bail!("var_exo_det not implemented for purely backwards models");
    }
    Ok(())
}

/// `k_order_pert.m` (order 3, or `k_order_solver`): the k-order MEX fails on a model with
/// `varexo_det` (it indexes the exogenous steady state past `M_.exo_nbr`); `print_info`
/// then stops with info 9.
pub fn k_order_check(model: &Model, order: usize, k_order_solver: bool) -> anyhow::Result<()> {
    if model.exo_det_names.is_empty() || (order < 3 && !k_order_solver) {
        return Ok(());
    }
    println!("Index exceeds the number of array elements. Index must not exceed {}.", model.n_exo());
    anyhow::bail!("k_order_pert was unable to compute the solution")
}

/// `dr.exo_det_length`: the last period of a deterministic shock (`stochastic_solvers.m` loops
/// over every entry of `M_.det_shocks`, the ones on ordinary exogenous variables included).
pub fn exo_det_length(mf: &ModFile) -> usize {
    mf.q1.exo_det_shocks.iter().chain(&mf.det_shocks).flat_map(|d| d.periods.iter()).copied().max().unwrap_or(0).max(0)
        as usize
}

/// Leaf values at the deterministic steady state (exogenous shocks at zero).
fn leaf_value<'a>(ys: &'a [f64], params: &'a [f64]) -> impl Fn(Node) -> f64 + 'a {
    move |n: Node| -> f64 {
        match n {
            Node::Endo(j, _) | Node::SteadyState(j, _) => ys[j as usize],
            Node::Param(q) => params[q as usize],
            _ => 0.0,
        }
    }
}

/// `fudet` at the steady state: `neq x n_exo_det` (the `varexo_det` columns of `g1`).
pub fn fudet(model: &Model, ys: &[f64], params: &[f64]) -> Mat {
    let neq = model.n_equations();
    let nd = model.exo_det_params.len();
    let mut out = Mat::zeros(neq, nd);
    if neq == 0 || nd == 0 {
        return out;
    }
    let mut dag = model.residual_node(0).0.clone();
    let leaf = leaf_value(ys, params);
    for (k, &p) in model.exo_det_params.iter().enumerate() {
        let mut memo = HashMap::new();
        for i in 0..neq {
            let r = model.residual_node(i).1;
            let d = dag.diff(r, DiffVar::Param(p as u32), &mut memo);
            out[(i, k)] = dag.eval(d, &leaf);
        }
    }
    out
}

/// The second derivatives of the dynamic residuals that involve a `varexo_det` variable (the
/// `varexo_det` rows and columns of `g2`): `(eq, dynamic column, k, value)` with the
/// endogenous columns of `Model::dynamic_jacobian`'s layout, and `(eq, k1, k2, value)`.
struct ExoDetHessian {
    cross: Vec<(usize, usize, usize, f64)>,
    dd: Vec<(usize, usize, usize, f64)>,
}

fn exo_det_hessian(model: &Model, ys: &[f64], params: &[f64]) -> ExoDetHessian {
    let n = model.n_endo();
    let mut out = ExoDetHessian { cross: Vec::new(), dd: Vec::new() };
    if model.n_equations() == 0 {
        return out;
    }
    let mut dag = model.residual_node(0).0.clone();
    let leaf = leaf_value(ys, params);
    let mut memo: HashMap<DiffVar, HashMap<crate::model::expr::NodeId, crate::model::expr::NodeId>> = HashMap::new();
    let mut buf = Vec::new();
    for i in 0..model.n_equations() {
        let r = model.residual_node(i).1;
        for (k, &p) in model.exo_det_params.iter().enumerate() {
            let v1 = DiffVar::Param(p as u32);
            let d1 = dag.diff(r, v1, memo.entry(v1).or_default());
            buf.clear();
            dag.collect_leaves(d1, &mut buf);
            let mut cols: Vec<(u32, i32)> = buf
                .iter()
                .filter_map(|l| match *l {
                    Node::Endo(j, l) if (-1..=1).contains(&l) => Some((j, l)),
                    _ => None,
                })
                .collect();
            cols.sort_unstable();
            cols.dedup();
            for (j, l) in cols {
                let v2 = DiffVar::Endo(j, l);
                let d2 = dag.diff(d1, v2, memo.entry(v2).or_default());
                let v = dag.eval(d2, &leaf);
                if v != 0.0 {
                    out.cross.push((i, (l + 1) as usize * n + j as usize, k, v));
                }
            }
            for (k2, &p2) in model.exo_det_params.iter().enumerate() {
                let v2 = DiffVar::Param(p2 as u32);
                let d2 = dag.diff(d1, v2, memo.entry(v2).or_default());
                let v = dag.eval(d2, &leaf);
                if v != 0.0 {
                    out.dd.push((i, k, k2, v));
                }
            }
        }
    }
    out
}

/// `M1 = inv(f0 + [0 f1*gx 0])`, `f1` of `stochastic_solvers.m` (DR order).
fn m1_f1(model: &Model, dr: &DecisionRule, ys: &[f64], params: &[f64]) -> (Mat, Mat, Mat) {
    let n = model.n_endo();
    let x = vec![0.0; model.n_exo()];
    let pe = model.params_with_steady_state(params, ys);
    let g1 = model.dynamic_jacobian(ys, &x, &pe);
    let (ns, np, nsf) = (dr.nstatic, dr.npred, dr.nsfwrd);
    let ov = &dr.order_var;
    let f1 = Mat::from_fn(n, nsf, |i, j| g1[(i, 2 * n + ov[ns + np + j])]);
    let f0 = Mat::from_fn(n, n, |i, j| g1[(i, n + ov[j])]);
    let gx = Mat::from_fn(nsf, dr.nspred, |i, j| dr.ghx[(ns + np + i, j)]);
    let f1gx = &f1 * &gx;
    let mut a = f0;
    for i in 0..n {
        for j in 0..dr.nspred {
            a[(i, ns + j)] += f1gx[(i, j)];
        }
    }
    let m1 = a.try_inverse().unwrap_or_else(|| Mat::from_element(n, n, f64::NAN));
    (m1, f1, g1)
}

/// `dr.ghud{1..length}` (DR row order): `M1 = inv(f0 + [0 f1*gx 0])`, `M2 = M1*f1`,
/// `ghud{1} = -M1*fudet`, `ghud{i} = -M2*ghud{i-1}(end-nsfwrd+1:end, :)`. As in MATLAB,
/// `ghud{1}` exists even when `length = 0`.
pub fn ghud(model: &Model, dr: &DecisionRule, ys: &[f64], params: &[f64], length: usize) -> Vec<Mat> {
    let n = model.n_endo();
    let nsf = dr.nsfwrd;
    let (m1, f1, _) = m1_f1(model, dr, ys, params);
    let m2 = &m1 * &f1;
    let fu = fudet(model, ys, params);
    let mut out = Vec::with_capacity(length.max(1));
    out.push(-(&m1 * &fu));
    for _ in 1..length {
        let prev = out.last().unwrap();
        let tail = prev.rows(n - nsf, nsf).clone_owned();
        out.push(-(&m2 * &tail));
    }
    out
}

/// The order-2 decision rule terms of the `varexo_det` variables (`stochastic_solvers.m`,
/// `local_order > 1`), DR row order: `ghud{i}`, `ghxud{i}` (columns `kron(x, ud)`),
/// `ghuud{i}` (`kron(u, ud)`) and `ghudud{j,i}` for `j <= i` (`kron(ud_j, ud_i)`), `i, j` in
/// `1..exo_det_length` (at least 1).
#[derive(Clone, Debug)]
pub struct ExoDetRule2 {
    pub ghud: Vec<Mat>,
    pub ghxud: Vec<Mat>,
    pub ghuud: Vec<Mat>,
    /// `ghudud[j][i]` = `dr.ghudud{j+1,i+1}`, filled for `j <= i`.
    pub ghudud: Vec<Vec<Mat>>,
}

/// Port of the order-2 part of `stochastic_solvers.m`'s "exogenous deterministic variables"
/// section, with Dynare's own expressions (`zu` without the shock rows, the `ghudud{i,i}`
/// state block with `ghud{i-1}(kf,:)`), on dynare-rust's order-2 rule `dr2` (`ghxx`).
pub fn second_order(model: &Model, dr2: &KOrderDr, ys: &[f64], params: &[f64], length: usize) -> ExoDetRule2 {
    let dr = &dr2.first;
    let n = model.n_endo();
    let nd = model.exo_det_params.len();
    let (ns, nsp, nsf) = (dr.nstatic, dr.nspred, dr.nsfwrd);
    let ov = &dr.order_var;
    let lli = &model.lead_lag_incidence;
    let ghud = ghud(model, dr, ys, params, length);
    let len = ghud.len();
    let (m1, f1, _) = m1_f1(model, dr, ys, params);
    let m2 = &m1 * &f1;
    // z = [lagged states; current; leads; shocks; varexo_det] (kk1 of stochastic_solvers.m)
    let kp: Vec<usize> = (ns..ns + nsp).collect();
    let k0: Vec<usize> = (0..n).filter(|&k| lli[ov[k]][1] != 0).collect();
    let kf: Vec<usize> = (n - nsf..n).collect();
    let (nc, ne) = (k0.len(), model.n_exo());
    let nz = nsp + nc + nsf + ne + nd;
    let zd0 = nsp + nc + nsf + ne;
    let mut zmap = vec![usize::MAX; 3 * n + ne];
    for (p, &k) in kp.iter().enumerate() {
        zmap[ov[k]] = p;
    }
    for (q, &k) in k0.iter().enumerate() {
        zmap[n + ov[k]] = nsp + q;
    }
    for (p, &k) in kf.iter().enumerate() {
        zmap[2 * n + ov[k]] = nsp + nc + p;
    }
    for e in 0..ne {
        zmap[3 * n + e] = nsp + nc + nsf + e;
    }
    // the Hessian over z (both orders of every pair)
    let x = vec![0.0; ne];
    let pe = model.params_with_steady_state(params, ys);
    let deriv = model.dynamic_derivatives(2);
    let vals = deriv.eval(ys, &x, &pe);
    let mut h: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut push = |eq: usize, a: usize, b: usize, v: f64| {
        if a == usize::MAX || b == usize::MAX || v == 0.0 {
            return;
        }
        h.push((eq, a, b, v));
        if a != b {
            h.push((eq, b, a, v));
        }
    };
    for (&(eq, a, b), &v) in deriv.g2_index.iter().zip(&vals.g2) {
        push(eq, zmap[a], zmap[b], v);
    }
    let xh = exo_det_hessian(model, ys, params);
    for &(eq, c, k, v) in &xh.cross {
        push(eq, zmap[c], zd0 + k, v);
    }
    for &(eq, k1, k2, v) in &xh.dd {
        if k1 <= k2 {
            push(eq, zd0 + k1, zd0 + k2, v);
        }
    }
    // g2_reordered * kron(A, B)
    let hk = |a: &Mat, b: &Mat| -> Mat {
        let (na, nb) = (a.ncols(), b.ncols());
        let mut r = Mat::zeros(n, na * nb);
        for &(eq, za, zb, v) in &h {
            for p in 0..na {
                let x = a[(za, p)];
                if x == 0.0 {
                    continue;
                }
                for q in 0..nb {
                    r[(eq, p * nb + q)] += v * x * b[(zb, q)];
                }
            }
        }
        r
    };
    let rows = |m: &Mat, idx: &[usize]| Mat::from_fn(idx.len(), m.ncols(), |i, j| m[(idx[i], j)]);
    let gx = rows(&dr.ghx, &kf);
    let gy = rows(&dr.ghx, &kp);
    let hu = rows(&dr.ghu, &kp);
    let hx = rows(&dr.ghx, &kp);
    let ghxx_f = rows(&dr2.ghxx, &kf);
    let eud = Mat::identity(nd, nd);
    // z-vectors: zx, zu, zud{i} (with the extra lead term for the ghudud{i,i} recursion)
    let zvec = |cur: &Mat, lead: &Mat, ncol: usize, det_eye: bool| -> Mat {
        let mut z = Mat::zeros(nz, ncol);
        for (q, &k) in k0.iter().enumerate() {
            for j in 0..ncol {
                z[(nsp + q, j)] = cur[(k, j)];
            }
        }
        for p in 0..nsf {
            for j in 0..ncol {
                z[(nsp + nc + p, j)] = lead[(p, j)];
            }
        }
        if det_eye {
            for k in 0..nd {
                z[(zd0 + k, k)] = 1.0;
            }
        }
        z
    };
    let mut zx = zvec(&dr.ghx, &(&gx * &gy), nsp, false);
    for p in 0..nsp {
        zx[(p, p)] = 1.0;
    }
    let zu = zvec(&dr.ghu, &(&gx * &hu), ne, false);
    let hud: Vec<Mat> = ghud.iter().map(|g| rows(g, &kp)).collect();
    let zud = zvec(&ghud[0], &(&gx * &hud[0]), nd, true);
    let zudi = |i: usize| zvec(&ghud[i], &(&gx * &hud[i]), nd, false);
    let f1ghxx = &f1 * &ghxx_f;
    // ghxud
    let mut ghxud: Vec<Mat> = Vec::with_capacity(len);
    ghxud.push(-(&m1 * (hk(&zx, &zud) + &f1ghxx * hx.kronecker(&hud[0]))));
    for i in 1..len {
        let r2 = hk(&zx, &zudi(i));
        let prev = rows(&ghxud[i - 1], &kf);
        let t = &prev * gy.kronecker(&eud) + &ghxx_f * hx.kronecker(&hud[i]);
        ghxud.push(-(&m2 * t) - &m1 * r2);
    }
    // ghuud
    let mut ghuud: Vec<Mat> = Vec::with_capacity(len);
    ghuud.push(-(&m1 * (hk(&zu, &zud) + &f1ghxx * hu.kronecker(&hud[0]))));
    for i in 1..len {
        let r2 = hk(&zu, &zudi(i));
        let prev = rows(&ghxud[i - 1], &kf);
        ghuud.push(-(&m2 * (&prev * hu.kronecker(&eud))) - &m1 * r2);
    }
    // ghudud{j,i}, j <= i
    let mut ghudud: Vec<Vec<Mat>> = vec![vec![Mat::zeros(0, 0); len]; len];
    ghudud[0][0] = -(&m1 * hk(&zud, &zud)) - &m2 * (&ghxx_f * hud[0].kronecker(&hud[0]));
    for i in 1..len {
        let hudi = &hud[i];
        let mut zi = zudi(i);
        let extra = rows(&ghud[i - 1], &kf);
        for p in 0..nsf {
            for j in 0..nd {
                zi[(nsp + nc + p, j)] += extra[(p, j)];
            }
        }
        let r2 = hk(&zi, &zi);
        let t = rows(&ghudud[i - 1][i - 1], &kf)
            + 2.0 * (rows(&ghxud[i - 1], &kf) * hudi.kronecker(&eud))
            + &ghxx_f * hudi.kronecker(hudi);
        ghudud[i][i] = -(&m2 * t) - &m1 * r2;
        let r2 = hk(&zud, &zi);
        let t = rows(&ghxud[i - 1], &kf) * hud[0].kronecker(&eud) + &ghxx_f * hud[0].kronecker(hudi);
        ghudud[0][i] = -(&m2 * t) - &m1 * r2;
        for j in 1..i {
            let r2 = hk(&zudi(j), &zi);
            let t = rows(&ghudud[j - 1][i - 1], &kf)
                + rows(&ghxud[j - 1], &kf) * hudi.kronecker(&eud)
                + rows(&ghxud[i - 1], &kf) * hud[j].kronecker(&eud)
                + &ghxx_f * hud[j].kronecker(hudi);
            ghudud[j][i] = -(&m2 * t) - &m1 * r2;
        }
    }
    ExoDetRule2 { ghud, ghxud, ghuud, ghudud }
}

/// `oo_.exo_det_simul` as `forecasts.run` sees it after `make_ex_.m` with `options_.periods =
/// periods` (`periods + maximum_lag + maximum_lead` rows at least; MATLAB grows the matrix
/// with zeros when a shock lies beyond them), extended with the steady state `xs` up to
/// `horizon` forecast periods.
pub fn exo_det_simul(mf: &ModFile, xs: &[f64], periods: usize, horizon: usize, ml: usize, mlead: usize) -> Mat {
    let nd = xs.len();
    let mut rows: Vec<Vec<f64>> = (0..ml + periods + mlead).map(|_| xs.to_vec()).collect();
    for d in &mf.q1.exo_det_shocks {
        if !d.dates.is_empty() {
            // ---- shocks/shock_paths: make_ex_.m needs options_.simul.first_simulation_period
            println!(
                "WARNING: forecast: the shocks on {} are given as dates, which a stochastic forecast cannot place; ignored.",
                mf.model.exo_det_names[d.exo]
            );
        }
        for &p in &d.periods {
            let r = (p + ml as i64 - 1) as usize;
            while rows.len() <= r {
                rows.push(vec![0.0; nd]);
            }
            rows[r][d.exo] = match d.kind {
                DetShockKind::Level => d.value,
                _ => xs[d.exo] * d.value,
            };
        }
    }
    let len = rows.len() - ml;
    if horizon > len {
        for _ in 0..horizon - len {
            rows.push(xs.to_vec());
        }
    }
    Mat::from_fn(rows.len(), nd, |r, k| rows[r][k])
}

/// `maximum_lag`, `maximum_lead` of the model (`M_.maximum_lag` / `M_.maximum_lead`).
fn max_lag_lead(model: &Model) -> (usize, usize) {
    let lli = &model.lead_lag_incidence;
    (lli.iter().any(|r| r[0] > 0) as usize, lli.iter().any(|r| r[2] > 0) as usize)
}

/// The `varexo_det` input of `simult_varexo_det.m` for a forecast over `horizon` periods with
/// `options_.periods = periods`: `ex_det` (rows from period `1 - maximum_lag`), the number of
/// simulated periods (`size(ex, 1)`), `maximum_lag` and the steady state.
pub struct ExoDetPath {
    pub ex_det: Mat,
    pub sim_periods: usize,
    pub ml: usize,
    pub xs: Vec<f64>,
    pub length: usize,
}

impl ExoDetPath {
    pub fn new(mf: &ModFile, params: &[f64], periods: usize, horizon: usize) -> ExoDetPath {
        let (ml, mlead) = max_lag_lead(&mf.model);
        let xs = mf.model.exo_det_values(params);
        let ex_det = exo_det_simul(mf, &xs, periods, horizon, ml, mlead);
        let sim_periods = ex_det.nrows() - ml;
        ExoDetPath { ex_det, sim_periods, ml, xs, length: exo_det_length(mf) }
    }

    /// `ex_det(row, :)' - exo_det_steady_state` (0-based row).
    fn dev(&self, row: usize) -> Vec<f64> {
        (0..self.xs.len()).map(|k| self.ex_det[(row, k)] - self.xs[k]).collect()
    }
}

/// The deterministic part of `simult_varexo_det.m` (order 1) added to a forecast computed
/// without it (the bands are unchanged): `y_i += ghx*dev_{i-1} + sum_j ghud{j}*(ex_det(i+j-1) - xs)`.
pub fn add_to_forecast(
    mf: &ModFile,
    dr: &DecisionRule,
    ys: &[f64],
    params: &[f64],
    periods: usize,
    horizon: usize,
    f: &mut Forecast,
) {
    let model = &mf.model;
    if model.exo_det_names.is_empty() {
        return;
    }
    let path = ExoDetPath::new(mf, params, periods, horizon);
    let (ml, len) = (path.ml, path.length);
    let g = ghud(model, dr, ys, params, len);
    let n = dr.ghx.nrows();
    let k2: Vec<usize> = dr.state_var.iter().map(|&s| dr.inv_order_var[s]).collect();
    let mut dev = vec![0.0; n];
    for i in ml + 1..=horizon + ml {
        let s: Vec<f64> = k2.iter().map(|&k| dev[k]).collect();
        let mut y: Vec<f64> = (0..n).map(|r| (0..k2.len()).map(|j| dr.ghx[(r, j)] * s[j]).sum()).collect();
        let jmax = (ml + len + 1).saturating_sub(i).min(len);
        for j in 1..=jmax {
            let u = path.dev(i + j - 2); // ex_det(i+j-1)
            for (k, &uk) in u.iter().enumerate() {
                if uk != 0.0 {
                    for r in 0..n {
                        y[r] += g[j - 1][(r, k)] * uk;
                    }
                }
            }
        }
        dev = y;
        let h = i - ml - 1;
        for (c, &v) in f.vars.iter().enumerate() {
            let d = dev[dr.inv_order_var[v]];
            f.mean[(h, c)] += d;
            f.lower[(h, c)] += d;
            f.upper[(h, c)] += d;
        }
    }
}

fn kron(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().flat_map(|&x| b.iter().map(move |&y| x * y)).collect()
}

fn mv_add(y: &mut [f64], m: &Mat, v: &[f64], a: f64) {
    for (j, &x) in v.iter().enumerate() {
        if x != 0.0 {
            for (i, yi) in y.iter_mut().enumerate() {
                *yi += a * m[(i, j)] * x;
            }
        }
    }
}

/// `simulate_varexo_det_order_2` of `simult_varexo_det.m`: from `y0` (declaration order,
/// levels; `maximum_lag <= 1`) with the shocks `ex` (`sim_periods x n_exo`). Returns
/// `n x (sim_periods + 1)`, declaration order, column 0 = `y0`.
pub fn simulate_order2(dr2: &KOrderDr, xd: &ExoDetRule2, path: &ExoDetPath, y0: &[f64], ex: &Mat) -> Mat {
    let d = &dr2.first;
    let n = d.ghx.nrows();
    let ov = &d.order_var;
    let ml = path.ml;
    let len = path.length;
    let ysd: Vec<f64> = ov.iter().map(|&j| d.ys[j]).collect();
    let periods = ex.nrows();
    let mut out = Mat::zeros(n, periods + 1);
    for i in 0..n {
        out[(i, 0)] = y0[i];
    }
    for t in 1..=periods {
        let i = t + ml; // MATLAB column of y_ (1-based, maximum_lag = 1)
        let x: Vec<f64> = (d.nstatic..d.nstatic + d.nspred).map(|k| out[(ov[k], t - 1)] - ysd[k]).collect();
        let u: Vec<f64> = (0..ex.ncols()).map(|e| ex[(t - 1, e)]).collect();
        let mut y: Vec<f64> = (0..n).map(|k| ysd[k] + dr2.ghs2[(k, 0)] / 2.0).collect();
        mv_add(&mut y, &d.ghx, &x, 1.0);
        mv_add(&mut y, &d.ghu, &u, 1.0);
        mv_add(&mut y, &dr2.ghxx, &kron(&x, &x), 0.5);
        mv_add(&mut y, &dr2.ghuu, &kron(&u, &u), 0.5);
        mv_add(&mut y, &dr2.ghxu, &kron(&x, &u), 1.0);
        let jmax = (ml + len + 1).saturating_sub(i).min(len);
        for j in 1..=jmax {
            let ud = path.dev(i + j - 2);
            mv_add(&mut y, &xd.ghud[j - 1], &ud, 1.0);
            mv_add(&mut y, &xd.ghxud[j - 1], &kron(&x, &ud), 1.0);
            mv_add(&mut y, &xd.ghuud[j - 1], &kron(&u, &ud), 1.0);
            mv_add(&mut y, &xd.ghudud[j - 1][j - 1], &kron(&ud, &ud), 0.5);
            for k in 1..j {
                let udk = path.dev(i + k - 2);
                mv_add(&mut y, &xd.ghudud[k - 1][j - 1], &kron(&udk, &ud), 1.0);
            }
        }
        for (k, &j) in ov.iter().enumerate() {
            out[(j, t)] = y[k];
        }
    }
    out
}
