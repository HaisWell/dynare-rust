// SPDX-License-Identifier: GPL-3.0-or-later
//! The `epilogue` block: port of the preprocessor's generated
//! `+<fname>/epilogue_dynamic.m` and `epilogue_static.m` (`EpilogueStatement`) and of
//! `shock_decomposition/epilogue_shock_decomposition.m`.
//!
//! `epilogue_dynamic(params, ds)` evaluates the equations in order on a dseries: for each
//! `NAME = EXPR`, the series `NAME` is created (NaN) if absent, then computed recursively
//! (`from d1 to d2 do ...`) from the latest `firstobservedperiod` of a variable of `EXPR` plus
//! that variable's maximum lag (Dynare 7.2) to the last period. Any MATLAB error inside (no variable, no overlapping
//! observations, a lead: the 7.2 preprocessor prints `x(t1)` for `x(+1)`) is caught and leaves
//! the series NaN, which is reproduced here.
//!
//! `epilogue_static(params, ds)` evaluates the equations with the lags removed (and the
//! DataTree simplifications `x/x = 1`, `x-x = 0` that dropping the lags exposes) on the
//! steady state.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};

use crate::model::expr::BinaryOp;
use crate::modfile::{Ast, ModFile, SpecialOp};

/// `NAME = EXPR;` of the epilogue block.
#[derive(Clone, Debug)]
pub struct EpilogueEq {
    pub name: String,
    pub ast: Ast,
}

/// Variables (not parameters) of `a` with their lags, in order of first appearance.
fn vars_of(a: &Ast, params: &[String], out: &mut Vec<(String, i32)>) {
    match a {
        Ast::Num(_) => {}
        Ast::Var { name, lag, .. } => {
            if !params.contains(name) {
                out.push((name.clone(), *lag));
            }
        }
        Ast::Un(_, x) => vars_of(x, params, out),
        Ast::Bin(_, x, y) => {
            vars_of(x, params, out);
            vars_of(y, params, out);
        }
        Ast::Op(_, args) => args.iter().for_each(|x| vars_of(x, params, out)),
    }
}

/// A named collection of series of equal length (a dseries without dates).
pub type Series = BTreeMap<String, Vec<f64>>;

/// `M_.epilogue_var_list_`: the endogenous variables used by the epilogue, in order.
pub fn var_list(mf: &ModFile) -> Vec<String> {
    let names: Vec<&String> = mf.q4.epilogue.iter().map(|e| &e.name).collect();
    let mut out: Vec<String> = Vec::new();
    for e in &mf.q4.epilogue {
        let mut v = Vec::new();
        vars_of(&e.ast, &mf.model.param_names, &mut v);
        for (n, _) in v {
            if !names.contains(&&n) && !out.contains(&n) && mf.model.endo_index(&n).is_some() {
                out.push(n);
            }
        }
    }
    out
}

fn eval(a: &Ast, p: &BTreeMap<&str, f64>, var: &dyn Fn(&str, i32) -> f64) -> f64 {
    match a {
        Ast::Num(v) => *v,
        Ast::Var { name, lag, .. } => match p.get(name.as_str()) {
            Some(v) => *v,
            None => var(name, *lag),
        },
        Ast::Un(op, x) => op.apply(eval(x, p, var)),
        Ast::Bin(op, x, y) => op.apply(eval(x, p, var), eval(y, p, var)),
        Ast::Op(op, args) => {
            let v: Vec<f64> = args.iter().map(|x| eval(x, p, var)).collect();
            let (mu, sigma) = if v.len() == 3 { (v[1], v[2]) } else { (0.0, 1.0) };
            match op {
                SpecialOp::Normcdf => {
                    0.5 * crate::model::expr::UnaryOp::Erfc.apply(-(v[0] - mu) / (sigma * std::f64::consts::SQRT_2))
                }
                SpecialOp::Normpdf => {
                    let z = (v[0] - mu) / sigma;
                    (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
                }
                SpecialOp::External(f) => crate::model::external::get(*f).eval(0, 0, &v),
                _ => f64::NAN, // rejected by the parser
            }
        }
    }
}

fn param_map<'a>(mf: &'a ModFile, params: &[f64]) -> BTreeMap<&'a str, f64> {
    mf.model.param_names.iter().map(String::as_str).zip(params.iter().copied()).collect()
}

/// `epilogue_dynamic(params, ds)`: adds (or updates) the epilogue series in `ds` (all series
/// of length `t`).
pub fn dynamic(mf: &ModFile, params: &[f64], ds: &mut Series, t: usize) {
    let p = param_map(mf, params);
    for e in &mf.q4.epilogue {
        ds.entry(e.name.clone()).or_insert_with(|| vec![f64::NAN; t]);
        let mut vars = Vec::new();
        vars_of(&e.ast, &mf.model.param_names, &mut vars);
        // MATLAB errors (caught by the generated try/catch): no variable, unknown series, a lead
        if vars.is_empty() || vars.iter().any(|(n, l)| *l > 0 || !ds.contains_key(n)) {
            continue;
        }
        // 7.2: simul_begin_date = max_i(firstobservedperiod(ds{x_i}) + maxlag_i), the maximum
        // lag taken per variable (7.1 used the first period where all variables are observed
        // plus the global maximum lag). firstobservedperiod errors on an all-NaN series.
        let mut per_var: Vec<(&str, usize)> = Vec::new();
        for (n, l) in &vars {
            let lag = (-l).max(0) as usize;
            match per_var.iter_mut().find(|(m, _)| *m == n.as_str()) {
                Some((_, m)) => *m = (*m).max(lag),
                None => per_var.push((n, lag)),
            }
        }
        let begin = per_var
            .iter()
            .map(|(n, lag)| (0..t).find(|&s| !ds[*n][s].is_nan()).map(|f| f + lag))
            .collect::<Option<Vec<_>>>();
        let Some(begin) = begin.and_then(|b| b.into_iter().max()) else { continue };
        if begin >= t {
            continue;
        }
        // from simul_begin_date to simul_end_date do NAME(t) = EXPR(t): sequential in t (the
        // expression may use NAME's own lags)
        let mut out = ds[&e.name].clone();
        for s in begin..t {
            let v = {
                let look = |n: &str, l: i32| -> f64 {
                    let k = s as i64 + l as i64;
                    if n == e.name { out[k as usize] } else { ds[n][k as usize] }
                };
                eval(&e.ast, &p, &look)
            };
            out[s] = v;
        }
        // dseries' `from` assigns `ds.NAME` back through sprintf('%f ', data): every value of
        // the series is rounded to 6 decimals (and later equations see the rounded values)
        for v in out.iter_mut() {
            if v.is_finite() {
                *v = format!("{:.6}", *v).parse().unwrap();
            }
        }
        ds.insert(e.name.clone(), out);
    }
}

/// The static form of an expression: lags dropped, then the DataTree rules `x/x = 1`,
/// `x-x = 0` (the only simplifications that change values: the others are exact).
fn static_ast(a: &Ast) -> Ast {
    match a {
        Ast::Num(_) => a.clone(),
        Ast::Var { name, loc, .. } => Ast::Var { name: name.clone(), lag: 0, loc: loc.clone() },
        Ast::Un(op, x) => Ast::Un(*op, Box::new(static_ast(x))),
        Ast::Bin(op, x, y) => {
            let (x, y) = (static_ast(x), static_ast(y));
            if x == y {
                match op {
                    BinaryOp::Div => return Ast::Num(1.0),
                    BinaryOp::Sub => return Ast::Num(0.0),
                    _ => {}
                }
            }
            Ast::Bin(*op, Box::new(x), Box::new(y))
        }
        Ast::Op(op, args) => Ast::Op(*op, args.iter().map(static_ast).collect()),
    }
}

/// `epilogue_static(params, ds)` on one period: the epilogue variables at `ys` (declaration
/// order of the endogenous variables).
pub fn static_values(mf: &ModFile, params: &[f64], ys: &[f64]) -> Vec<f64> {
    let p = param_map(mf, params);
    let mut vals: BTreeMap<String, f64> = BTreeMap::new();
    let mut out = Vec::new();
    for e in &mf.q4.epilogue {
        let a = static_ast(&e.ast);
        let v = {
            let look = |n: &str, _l: i32| -> f64 {
                vals.get(n).copied().or_else(|| mf.model.endo_index(n).map(|j| ys[j])).unwrap_or(f64::NAN)
            };
            eval(&a, &p, &look)
        };
        vals.insert(e.name.clone(), v);
        out.push(v);
    }
    out
}

/// The epilogue variables over endogenous paths: `paths` is n_endo x T (levels, declaration
/// order); returns n_epilogue x T.
pub fn on_paths(mf: &ModFile, params: &[f64], paths: &crate::linalg::Mat) -> crate::linalg::Mat {
    let t = paths.ncols();
    let mut ds: Series = Series::new();
    for v in var_list(mf) {
        let j = mf.model.endo_index(&v).unwrap();
        ds.insert(v, paths.row(j).iter().copied().collect());
    }
    dynamic(mf, params, &mut ds, t);
    let n = mf.q4.epilogue.len();
    crate::linalg::Mat::from_fn(n, t, |k, s| ds[&mf.q4.epilogue[k].name][s])
}

/// Port of `epilogue_shock_decomposition.m`: `z` holds `nvar` rows (declaration order of the
/// endogenous variables) of (ncomp x T) decompositions in deviation from `ys`; returns the
/// epilogue rows to append and their steady state (`epilogue_steady_state`).
pub fn shock_decomposition(
    mf: &ModFile,
    params: &[f64],
    ys: &[f64],
    z: &[crate::linalg::Mat],
) -> Result<(Vec<crate::linalg::Mat>, Vec<f64>)> {
    let ivar: Vec<usize> = var_list(mf)
        .iter()
        .map(|v| mf.model.endo_index(v).ok_or_else(|| anyhow!("epilogue: unknown variable {v}")))
        .collect::<Result<_>>()?;
    let nepi = mf.q4.epilogue.len();
    let (ncomp, t) = z.first().map_or((0, 0), |m| (m.nrows(), m.ncols()));
    let yss = static_values(mf, params, ys);
    // y(k, comp, t)
    let mut y = vec![crate::linalg::Mat::from_element(ncomp, t, f64::NAN); nepi];
    for comp in 0..ncomp {
        let mut ds = Series::new();
        for &j in &ivar {
            ds.insert(mf.model.endo_names[j].clone(), (0..t).map(|s| z[j][(comp, s)] + ys[j]).collect());
        }
        dynamic(mf, params, &mut ds, t);
        for (k, e) in mf.q4.epilogue.iter().enumerate() {
            for s in 0..t {
                y[k][(comp, s)] = ds[&e.name][s] - yss[k];
            }
        }
    }
    // the components do not add up to the total (nonlinear epilogue): the residual is spread
    // over the components in proportion to their absolute values
    for m in y.iter_mut() {
        for s in 0..t {
            let sum: f64 = (0..ncomp - 1).map(|c| m[(c, s)]).sum();
            let sabs: f64 = (0..ncomp - 1).map(|c| m[(c, s)].abs()).sum();
            let res = m[(ncomp - 1, s)] - sum;
            for c in 0..ncomp - 1 {
                let w = m[(c, s)].abs() / sabs;
                m[(c, s)] += res * w;
            }
        }
    }
    Ok((y, yss))
}

/// dynare-rust's own output (Dynare evaluates the epilogue only inside the shock
/// decompositions; `<fname>.epilogue_dynamic(M_.params, ds)` is left to the user):
/// `<fname>_epilogue_<what>.csv` with the epilogue variables over endogenous paths `paths`
/// (n_endo x T levels, declaration order), one row per period labelled by `labels`.
pub fn write_csv(
    mf: &ModFile,
    params: &[f64],
    paths: &crate::linalg::Mat,
    labels: &[String],
    out_dir: &std::path::Path,
    fname: &str,
    what: &str,
) -> Result<()> {
    if mf.q4.epilogue.is_empty() {
        return Ok(());
    }
    let y = on_paths(mf, params, paths);
    let mut w = String::from("date");
    for e in &mf.q4.epilogue {
        w.push(',');
        w.push_str(&e.name);
    }
    w.push('\n');
    for t in 0..y.ncols() {
        w += labels.get(t).map_or(&(t + 1).to_string(), |s| s);
        for k in 0..y.nrows() {
            w += &format!(",{}", y[(k, t)]);
        }
        w.push('\n');
    }
    std::fs::create_dir_all(out_dir)?;
    let p = out_dir.join(format!("{fname}_epilogue_{what}.csv"));
    std::fs::write(&p, w)?;
    println!("epilogue: {} variable(s) over the {what} written to {}", y.nrows(), p.display());
    Ok(())
}

/// Index of an epilogue variable (rows after the endogenous variables in decompositions).
pub fn index(mf: &ModFile, name: &str) -> Option<usize> {
    mf.q4.epilogue.iter().position(|e| e.name == name)
}
