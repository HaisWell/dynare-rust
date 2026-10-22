// SPDX-License-Identifier: GPL-3.0-or-later
//! Estimation of a PAC equation: `pac.estimate.nls` (Gauss-Newton on the sum of squared
//! residuals, `gauss_newton.m`) and `pac.estimate.iterative_ols` (Brayton et al., 2000).
//!
//! In Dynare these commands take a `dseries` and a structure of initial values from the MATLAB
//! workspace; here the data is a [`PacData`] (named columns of equal length, the rows are the
//! periods) and `range` the first and last rows (inclusive) of the estimation sample. The
//! equation is the one decomposed by the parser (`PacModel::eq_lhs/eq_rhs`, with the
//! auxiliary variables of the PAC expectation substituted by their definitions), so the data
//! only needs the variables of the model (not the auxiliary ones); the residual of the
//! equation (the exogenous variable that is not in the data, or has only NaN over the sample,
//! `pac.estimate.init`) is zero. Every other variable of the equation, observed exogenous
//! regressors included, is read from the data; a missing one is an error.
//!
//! Every optimizer name of `pac.estimate.nls` is mapped to the Gauss-Newton iteration of
//! `GaussNewton` (projected on the bounds that `nls.m` gives the optimizer); an unknown name
//! is an error, as in `nls.m`.

use super::VarModels;
use super::pac::PacModel;
use crate::linalg::Mat;
use crate::modfile::{Ast, SpecialOp};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

/// Named series (same length), the period of row `t` is the same in all of them.
#[derive(Clone, Debug, Default)]
pub struct PacData {
    pub names: Vec<String>,
    pub cols: Vec<Vec<f64>>,
    /// `M_.exo_names`: the candidates for the residual of the equation. Empty: every variable
    /// of the equation that is not in the data (or only NaN) is a candidate.
    pub exo: Vec<String>,
}

impl PacData {
    pub fn new(names: Vec<String>, cols: Vec<Vec<f64>>) -> Self {
        PacData { names, cols, exo: Vec::new() }
    }

    /// [`PacData::new`] with the exogenous variables of the model (`M_.exo_names`).
    pub fn with_exo(mut self, exo: Vec<String>) -> Self {
        self.exo = exo;
        self
    }

    fn col(&self, n: &str) -> Option<&Vec<f64>> {
        self.names.iter().position(|x| x == n).map(|i| &self.cols[i])
    }
}

/// `oo_.pac.<name>` after an estimation.
#[derive(Clone, Debug)]
pub struct PacEstimate {
    pub parnames: Vec<String>,
    pub estimator: Vec<f64>,
    /// NLS only.
    pub covariance: Option<Mat>,
    pub student: Vec<f64>,
    pub ssr: f64,
    pub s2: f64,
    pub r2: f64,
    pub residual: Vec<f64>,
    pub lhs: Vec<f64>,
    /// Number of iterations (iterative OLS).
    pub iterations: usize,
}

struct Ev<'a> {
    names: &'a [String],
    params: &'a [f64],
    data: &'a PacData,
    defs: HashMap<&'a str, &'a Ast>,
    /// The residual of the equation (zero).
    residual: Option<&'a str>,
}

impl Ev<'_> {
    fn ev(&self, a: &Ast, t: isize) -> Result<f64> {
        Ok(match a {
            Ast::Num(v) => *v,
            Ast::Var { name, lag, .. } => {
                if let Some(i) = self.names.iter().position(|x| x == name) {
                    return Ok(self.params[i]);
                }
                if let Some(d) = self.defs.get(name.as_str()) {
                    return self.ev(d, t + *lag as isize);
                }
                if self.residual == Some(name.as_str()) {
                    return Ok(0.0);
                }
                match self.data.col(name) {
                    Some(c) => {
                        let k = t + *lag as isize;
                        if k < 0 || k as usize >= c.len() {
                            bail!(
                                "the sample does not contain enough observations before the first period ({name}({lag}))"
                            );
                        }
                        let v = c[k as usize];
                        if v.is_nan() {
                            bail!("Some variable values are missing in the database ({name}).");
                        }
                        v
                    }
                    None => bail!("Variable {name} is not in the database."),
                }
            }
            Ast::Un(op, x) => op.apply(self.ev(x, t)?),
            Ast::Bin(op, x, y) => op.apply(self.ev(x, t)?, self.ev(y, t)?),
            Ast::Op(SpecialOp::Diff, v) => self.ev(&v[0], t)? - self.ev(&v[0], t - 1)?,
            Ast::Op(o, _) => bail!("{} is not supported in a PAC equation", o.name()),
        })
    }
}

/// Variables of the PAC equation (not the parameters, nor the auxiliary variables of `defs`).
fn eq_vars(pac: &PacModel, names: &[String]) -> Vec<String> {
    fn walk(a: &Ast, out: &mut Vec<String>) {
        match a {
            Ast::Var { name, .. } => {
                if !out.contains(name) {
                    out.push(name.clone())
                }
            }
            Ast::Un(_, x) => walk(x, out),
            Ast::Bin(_, x, y) => {
                walk(x, out);
                walk(y, out)
            }
            Ast::Op(_, v) => v.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(&pac.eq_lhs, &mut out);
    walk(&pac.eq_rhs, &mut out);
    for (_, d) in &pac.defs {
        walk(d, &mut out);
    }
    out.retain(|n| !names.contains(n) && !pac.defs.iter().any(|(d, _)| d == n));
    out
}

/// `pac.estimate.init`: the residual of the equation, the exogenous variable that is not
/// observed (not in the data, or only NaN over `range`). `required`: estimation, where there
/// must be exactly one.
fn residual_name(
    pac: &PacModel,
    names: &[String],
    data: &PacData,
    range: (usize, usize),
    required: bool,
) -> Result<Option<String>> {
    let vars = eq_vars(pac, names);
    let unobserved: Vec<&String> = vars
        .iter()
        .filter(|n| data.exo.is_empty() || data.exo.contains(n))
        .filter(|n| match data.col(n) {
            None => true,
            Some(c) => {
                c.get(range.0..=range.1.min(c.len().saturating_sub(1))).is_none_or(|s| s.iter().all(|v| v.is_nan()))
            }
        })
        .collect();
    match unobserved.len() {
        0 if required => bail!("There is no residual in this equation, all the exogenous variables are observed."),
        0 => Ok(None),
        1 => Ok(Some(unobserved[0].clone())),
        _ => bail!(
            "It is not allowed to have more than one residual in a PAC equation ({} are not observed).",
            unobserved.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn pac_by_eq<'a>(vms: &'a VarModels, eqname: &str) -> Result<&'a PacModel> {
    let p = vms
        .pacs
        .iter()
        .find(|p| p.eq_name == eqname)
        .ok_or_else(|| anyhow!("This is not a PAC equation ({eqname})."))?;
    Ok(p)
}

/// Residuals `lhs - rhs` of the PAC equation over `range` (rows, inclusive) and the left hand
/// side, for the parameters `params` (the `h` parameters are *not* updated here).
fn residuals(
    pac: &PacModel,
    names: &[String],
    params: &[f64],
    data: &PacData,
    range: (usize, usize),
    residual: Option<&str>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let e = Ev { names, params, data, defs: pac.defs.iter().map(|(n, a)| (n.as_str(), a)).collect(), residual };
    let mut r = Vec::new();
    let mut l = Vec::new();
    for t in range.0..=range.1 {
        let lhs = e.ev(&pac.eq_lhs, t as isize)?;
        let rhs = e.ev(&pac.eq_rhs, t as isize)?;
        l.push(lhs);
        r.push(lhs - rhs);
    }
    Ok((r, l))
}

/// Residuals `lhs - rhs` of the PAC equation `eqname` over `range` (rows, inclusive) for the
/// parameters `params` (which must contain the current `h` parameters). The residual of the
/// equation, if not observed (not in `data` or only NaN), counts as zero.
pub fn pac_residuals(
    vms: &VarModels,
    eqname: &str,
    names: &[String],
    params: &[f64],
    data: &PacData,
    range: (usize, usize),
) -> Result<Vec<f64>> {
    let pac = pac_by_eq(vms, eqname)?;
    let rname = residual_name(pac, names, data, range, false)?;
    Ok(residuals(pac, names, params, data, range, rname.as_deref())?.0)
}

fn set_params(
    vms: &VarModels,
    pac: &PacModel,
    names: &[String],
    params: &mut [f64],
    est: &[usize],
    values: &[f64],
) -> Result<()> {
    for (&i, &v) in est.iter().zip(values) {
        params[i] = v;
    }
    let upd = if pac.is_mce() { pac.mce_values(names, params)? } else { pac.update_values(vms, names, params)? };
    for (n, v) in upd {
        let k = names.iter().position(|x| *x == n).ok_or_else(|| anyhow!("unknown parameter {n}"))?;
        params[k] = v;
    }
    Ok(())
}

fn indices(names: &[String], guess: &[(String, f64)]) -> Result<Vec<usize>> {
    guess
        .iter()
        .map(|(n, _)| names.iter().position(|x| x == n).ok_or_else(|| anyhow!("{n} is not a parameter of the model")))
        .collect()
}

fn var_of(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0)
}

/// `pac.estimate.nls(eqname, guess, data, range, algo)`. `names`/`params` are `M_.param_names`
/// / `M_.params` (updated: the estimates and the `h` parameters).
pub fn estimate_nls(
    vms: &VarModels,
    eqname: &str,
    names: &[String],
    params: &mut [f64],
    guess: &[(String, f64)],
    data: &PacData,
    range: (usize, usize),
) -> Result<PacEstimate> {
    estimate_nls_with(vms, eqname, names, params, guess, data, range, None, &[])
}

/// Optimizers known to `nls.m` (in the order of its error message, then `particleswarm`).
pub const NLS_OPTIMIZERS: [&str; 9] = [
    "fmincon",
    "fminunc",
    "csminwel",
    "fminsearch",
    "simplex",
    "annealing",
    "lsqnonlin",
    "GaussNewton",
    "particleswarm",
];

/// Bounds that `nls.m` puts on the error correction parameter for `optimizer` (with the
/// optional arguments `args` that follow it), `None` if unbounded; an unknown name is an error.
/// `lsqnonlin`, `fmincon` and `annealing`: [0, 1], except `lsqnonlin` with
/// `'levenberg-marquardt'` (which does not handle bounds); `particleswarm`: nls.m assigns the
/// lower bound twice (0, then 1), so the bounds are [1, Inf].
pub fn nls_bounds(optimizer: Option<&str>, args: &[String]) -> Result<Option<(f64, f64)>> {
    let Some(o) = optimizer else { return Ok(None) }; // csminwel
    Ok(match o {
        "lsqnonlin" if args.iter().any(|a| a == "levenberg-marquardt") => None,
        "lsqnonlin" | "fmincon" | "annealing" => Some((0.0, 1.0)),
        "particleswarm" => Some((1.0, f64::INFINITY)),
        "GaussNewton" | "fminunc" | "csminwel" | "fminsearch" | "simplex" => None,
        _ => {
            let mut msg = format!("{o} is not an implemented optimization routine.\nAvailable algorithms are:\n");
            for a in &NLS_OPTIMIZERS[..8] {
                msg.push_str(&format!(" - {a}\n"));
            }
            bail!("{msg}")
        }
    })
}

/// [`estimate_nls`] with the optimizer name of `nls.m` and its optional arguments: the error
/// correction parameter is bounded as [`nls_bounds`] says (projected Gauss-Newton here, not
/// the MATLAB algorithms); every known name runs Gauss-Newton.
#[allow(clippy::too_many_arguments)]
pub fn estimate_nls_with(
    vms: &VarModels,
    eqname: &str,
    names: &[String],
    params: &mut [f64],
    guess: &[(String, f64)],
    data: &PacData,
    range: (usize, usize),
    optimizer: Option<&str>,
    opt_args: &[String],
) -> Result<PacEstimate> {
    let pac = pac_by_eq(vms, eqname)?;
    let est = indices(names, guess)?;
    let ec_bounds = nls_bounds(optimizer, opt_args)?;
    if range.0 == 0 || range.1 < range.0 || range.1 >= data.cols.first().map_or(0, |c| c.len()) {
        bail!("invalid estimation range");
    }
    let rname = residual_name(pac, names, data, range, true)?;
    let rname = rname.as_deref();
    let base = params.to_vec();
    let resfun = |p: &[f64]| -> Result<Vec<f64>> {
        let mut q = base.clone();
        set_params(vms, pac, names, &mut q, &est, p)?;
        Ok(residuals(pac, names, &q, data, range, rname)?.0)
    };
    let x0: Vec<f64> = guess.iter().map(|(_, v)| *v).collect();
    let r0 = resfun(&x0)?;
    let s0: f64 = r0.iter().map(|v| v * v).sum();
    if s0.is_nan() || s0.is_infinite() {
        bail!("Cannot evaluate the Sum of Squared Residuals on the initial guess.");
    }
    let bounds: Vec<(f64, f64)> = if let Some(b) = ec_bounds {
        guess.iter().map(|(n, _)| if *n == pac.ec_param { b } else { (f64::NEG_INFINITY, f64::INFINITY) }).collect()
    } else {
        Vec::new()
    };
    let (x1, ssr) = gauss_newton(&resfun, x0, &bounds)?;
    // covariance (Newey-West)
    let (r, jac) = jacobian(&resfun, &x1, 1e-6)?;
    let t = r.len();
    let p = x1.len();
    let tf = t as f64;
    let a = (jac.transpose() * &jac) * (2.0 / tf);
    let mut jr = jac.clone();
    for i in 0..t {
        for j in 0..p {
            jr[(i, j)] *= r[i];
        }
    }
    let mut b = jr.transpose() * &jr;
    let l = tf.powf(0.25).round() as usize;
    for tau in 1..=l {
        let w = 1.0 - tau as f64 / (l as f64 + 1.0);
        let up = jr.rows(tau, t - tau).into_owned();
        let lo = jr.rows(0, t - tau).into_owned();
        b += (up.transpose() * &lo + lo.transpose() * &up) * w;
    }
    let b = b * (4.0 / tf);
    let ainv = a.clone().try_inverse().ok_or_else(|| anyhow!("singular matrix in the NLS covariance"))?;
    let c = (&ainv * b * &ainv) / tf;
    let student: Vec<f64> = (0..p).map(|i| x1[i] / c[(i, i)].sqrt()).collect();
    let (rr, lhs) = residuals_at(vms, pac, names, params, &est, &x1, data, range, rname)?;
    set_params(vms, pac, names, params, &est, &x1)?;
    Ok(PacEstimate {
        parnames: guess.iter().map(|(n, _)| n.clone()).collect(),
        estimator: x1,
        covariance: Some(c),
        student,
        ssr,
        s2: ssr / tf,
        r2: 1.0 - var_of(&rr) / var_of(&lhs),
        residual: rr,
        lhs,
        iterations: 0,
    })
}

#[allow(clippy::too_many_arguments)]
fn residuals_at(
    vms: &VarModels,
    pac: &PacModel,
    names: &[String],
    params: &[f64],
    est: &[usize],
    x: &[f64],
    data: &PacData,
    range: (usize, usize),
    rname: Option<&str>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut q = params.to_vec();
    set_params(vms, pac, names, &mut q, est, x)?;
    residuals(pac, names, &q, data, range, rname)
}

/// `jacobian.m`: forward differences of `r` (note the sign: `(r(x) - r(x+h))/h`).
fn jacobian(f: &dyn Fn(&[f64]) -> Result<Vec<f64>>, x: &[f64], h: f64) -> Result<(Vec<f64>, Mat)> {
    let r = f(x)?;
    let mut j = Mat::zeros(r.len(), x.len());
    for i in 0..x.len() {
        let mut z = x.to_vec();
        z[i] += h;
        let ri = f(&z)?;
        for k in 0..r.len() {
            j[(k, i)] = (r[k] - ri[k]) / h;
        }
    }
    Ok((r, j))
}

/// `optimization/gauss_newton.m`.
fn gauss_newton(
    f: &dyn Fn(&[f64]) -> Result<Vec<f64>>,
    x0: Vec<f64>,
    bounds: &[(f64, f64)],
) -> Result<(Vec<f64>, f64)> {
    let clip = |mut x: Vec<f64>| -> Vec<f64> {
        for (v, b) in x.iter_mut().zip(bounds) {
            *v = v.clamp(b.0, b.1);
        }
        x
    };
    let x0 = clip(x0);
    let max_iter = 100;
    let h = f64::EPSILON.sqrt();
    let xtol = 1e-6;
    let phi = 0.5 * (5f64.sqrt() + 1.0);
    let ssr = |x: &[f64]| -> Result<f64> { Ok(f(x)?.iter().map(|v| v * v).sum()) };
    let mut x0 = x0;
    let mut counter = 0;
    let mut s1;
    loop {
        let (r0, jac) = jacobian(f, &x0, h)?;
        let pinv = jac
            .clone()
            .svd(true, true)
            .pseudo_inverse(f64::EPSILON * jac.nrows().max(jac.ncols()) as f64 * jac.norm())
            .map_err(|e| anyhow!("{e}"))?;
        let d = pinv * Mat::from_column_slice(r0.len(), 1, &r0);
        let d: Vec<f64> = d.iter().copied().collect();
        let x1: Vec<f64> = clip(x0.iter().zip(&d).map(|(a, b)| a + b).collect());
        let r1 = f(&x1)?;
        let s0: f64 = r0.iter().map(|v| v * v).sum();
        s1 = r1.iter().map(|v| v * v).sum();
        let x1 = if s1 > s0 {
            let (mut l1, mut l2) = (0.0, 1.0);
            let mut ll1 = l2 - (l2 - l1) / phi;
            let mut ll2 = l1 + (l2 - l1) / phi;
            while (ll1 - ll2).abs() > 1e-6 {
                let a: Vec<f64> = clip(x0.iter().zip(&d).map(|(a, b)| a + ll1 * b).collect());
                let b: Vec<f64> = clip(x0.iter().zip(&d).map(|(a, b)| a + ll2 * b).collect());
                if ssr(&a)? < ssr(&b)? {
                    l2 = ll2;
                } else {
                    l1 = ll1;
                }
                ll1 = l2 - (l2 - l1) / phi;
                ll2 = l1 + (l2 - l1) / phi;
            }
            let scale = 0.5 * (l1 + l2);
            clip(x0.iter().zip(&d).map(|(a, b)| a + scale * b).collect())
        } else {
            x1
        };
        let noconv = x1.iter().zip(&x0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) > xtol;
        counter += 1;
        x0 = x1;
        if !noconv || counter > max_iter {
            break;
        }
    }
    Ok((x0, s1))
}

/// `pac.estimate.iterative_ols(eqname, guess, data, range)`: all the error correction and
/// autoregressive parameters of the equation must be in `guess` (the order of the result is
/// `[ec, ar_1, ...]`).
pub fn estimate_iterative_ols(
    vms: &VarModels,
    eqname: &str,
    names: &[String],
    params: &mut Vec<f64>,
    guess: &[(String, f64)],
    data: &PacData,
    range: (usize, usize),
) -> Result<PacEstimate> {
    let pac = pac_by_eq(vms, eqname)?;
    if pac.is_mce() {
        bail!("pac.estimate.iterative_ols: model consistent expectations are not supported");
    }
    let mut order = vec![pac.ec_param.clone()];
    order.extend(pac.ar_params.iter().cloned());
    for n in &order {
        if !guess.iter().any(|(g, _)| g == n) {
            bail!("pac.estimate.iterative_ols: the parameter {n} must be in the initial values");
        }
    }
    if guess.len() != order.len() {
        bail!(
            "pac.estimate.iterative_ols: only the parameters of the error correction and autoregressive terms can be estimated"
        );
    }
    if range.0 == 0 || range.1 < range.0 || range.1 >= data.cols.first().map_or(0, |c| c.len()) {
        bail!("invalid estimation range");
    }
    let rname = residual_name(pac, names, data, range, true)?;
    let rname = rname.as_deref();
    let est = indices(names, &order.iter().map(|n| (n.clone(), 0.0)).collect::<Vec<_>>())?;
    let mut x0: Vec<f64> = order.iter().map(|n| guess.iter().find(|(g, _)| g == n).unwrap().1).collect();
    // regressors
    let defs: HashMap<&str, &Ast> = pac.defs.iter().map(|(n, a)| (n.as_str(), a)).collect();
    let t_n = range.1 - range.0 + 1;
    let ncol = 1 + pac.ar_params.len();
    let mut xdata = Mat::zeros(t_n, ncol);
    {
        let e = Ev { names, params, data, defs: defs.clone(), residual: rname };
        let lag = |n: &str, k: i32| Ast::Var {
            name: n.to_string(),
            lag: k,
            loc: crate::modfile::lexer::Loc { file: std::sync::Arc::from("<pac>"), line: 0 },
        };
        let ect = Ast::Bin(
            crate::model::expr::BinaryOp::Sub,
            Box::new(lag(&pac.target, -1)),
            Box::new(lag(&pac.lhs_var, -1)),
        );
        for (i, t) in (range.0..=range.1).enumerate() {
            xdata[(i, 0)] = e.ev(&ect, t as isize)?;
            for k in 1..=pac.ar_params.len() {
                xdata[(i, k)] = e.ev(&Ast::Op(SpecialOp::Diff, vec![lag(&pac.lhs_var, -(k as i32))]), t as isize)?;
            }
        }
    }
    set_params(vms, pac, names, params, &est, &x0)?;
    let mut counter = 0;
    let (mut ssr, mut resid, mut lhs);
    loop {
        counter += 1;
        let (r0, l) = residuals(pac, names, params, data, range, rname)?;
        // y = lhs - (rhs - X*p) = r0 + X*p
        let xp = &xdata * Mat::from_column_slice(ncol, 1, &x0);
        let y: Vec<f64> = (0..t_n).map(|i| r0[i] + xp[(i, 0)]).collect();
        let ym = Mat::from_column_slice(t_n, 1, &y);
        let p1 = xdata.clone().svd(true, true).solve(&ym, 1e-14).map_err(|e| anyhow!("{e}"))?;
        let p1: Vec<f64> = p1.iter().copied().collect();
        let fit = &xdata * Mat::from_column_slice(ncol, 1, &p1);
        resid = (0..t_n).map(|i| y[i] - fit[(i, 0)]).collect::<Vec<f64>>();
        ssr = resid.iter().map(|v| v * v).sum();
        lhs = l;
        let crit = p1.iter().zip(&x0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        x0 = p1;
        set_params(vms, pac, names, params, &est, &x0)?;
        if crit <= 1e-6 || counter > 1000 {
            break;
        }
    }
    let tf = t_n as f64;
    Ok(PacEstimate {
        parnames: order,
        estimator: x0,
        covariance: None,
        student: vec![f64::NAN; ncol],
        ssr,
        s2: ssr / tf,
        r2: 1.0 - var_of(&resid) / var_of(&lhs),
        residual: resid,
        lhs,
        iterations: counter,
    })
}
