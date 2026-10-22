// SPDX-License-Identifier: GPL-3.0-or-later
//! The `shock_paths` block (Dynare 7.2): temporary, permanent (`end`) and controlled
//! (`exogenize` / `endogenize`) shocks of a perfect-foresight scenario , with dates,
//! databases and expectation errors (`learnt_in`, the `prev.` and `learnt_in(...)` scopes).
//!
//! As the preprocessor, each block becomes an evaluation function (`<fname>.shock_paths_<k>`,
//! `exo_paths = f(exo_paths, p, periods, first_simulation_period, M_, oo_)`): at period `p`
//! every stanza whose periods contain `p` sets its exogenous variable, in block order;
//! `INTEGER:end` and `end` also cover the terminal period `periods+1` (a permanent shock).
//! `make_ex_.m` calls the (`learnt_in = 1`) functions for `p = 1..periods(+1)` on the
//! `n_exo x (periods(+1))` path; `perfect_foresight_with_expectation_errors_setup.m` keeps
//! one path per informational period (`exo_paths(:, :, info_period)`, copied from the previous
//! one) and calls the functions of the blocks learnt in `info_period` for
//! `p = info_period..periods(+1)`.
//!
//! Scopes of the `values` expressions: `initval.x` / `init.x` (`oo_.steady_state` /
//! `oo_.exo_steady_state` at evaluation time), `self.x(lag)` (the current informational
//! period's path), `prev.x(lag)` (the previous informational period's), `learnt_in(INTEGER |
//! DATE).x(lag)` (that informational period's) and `DATABASE.x(lag)` (`extract_from_database.m`:
//! a dseries at date `first_simulation_period + p + lag - 1`, a table at row `p + lag`, a
//! single-row table whatever the period). Controlled stanzas become
//! `M_.perfect_foresight_controlled_paths` entries evaluated where the block stands
//! (`initval` scope only, as Dynare).

use anyhow::{Result, anyhow, bail};

use crate::dates::Date;
use crate::ep::controlled::ControlledPath;
use crate::linalg::Mat;
use crate::model::{Model, SymbolKind};
use crate::modfile::{Ast, ModFile};

/// A bound of a `periods` entry: an integer simulation period or a date.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PeriodPoint {
    Int(usize),
    Date(Date),
}

impl PeriodPoint {
    fn resolve(self, fsp: Option<Date>) -> Result<i64> {
        match self {
            PeriodPoint::Int(p) => Ok(p as i64),
            PeriodPoint::Date(d) => super::periods::date_to_period(d, fsp),
        }
    }
}

/// Periods of one `periods` entry: `from..=to`, `to = None` for `FROM:end` (through the
/// terminal period), `from = None` for `end` alone.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodRange {
    pub from: Option<PeriodPoint>,
    pub to: Option<PeriodPoint>,
}

impl PeriodRange {
    /// The generated `if` condition (`p == a`, `p >= a && p <= b`, `p >= a`, `p == periods+1`).
    fn contains(&self, p: usize, periods: usize, fsp: Option<Date>) -> Result<bool> {
        let p = p as i64;
        Ok(match (self.from, self.to) {
            (None, _) => p == periods as i64 + 1,
            (Some(a), None) => p >= a.resolve(fsp)?,
            (Some(a), Some(b)) => p >= a.resolve(fsp)? && p <= b.resolve(fsp)?,
        })
    }
    pub fn is_endval(&self) -> bool {
        self.to.is_none()
    }
    fn has_date(&self) -> bool {
        matches!(self.from, Some(PeriodPoint::Date(_))) || matches!(self.to, Some(PeriodPoint::Date(_)))
    }
}

/// `var x; periods ...; values ...;`
#[derive(Clone, Debug, PartialEq)]
pub struct ShockStanza {
    pub exo: usize,
    pub items: Vec<(PeriodRange, Ast)>,
}

/// `exogenize y; periods ...; values ...; endogenize e;` (periods: integers or dates).
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledStanza {
    pub exogenize: usize,
    pub endogenize: usize,
    pub items: Vec<(Vec<i64>, Vec<Date>, Ast)>,
}

/// One `shock_paths` block.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ShockPathsBlock {
    pub learnt_in: usize,
    /// `learnt_in = DATE` (`learnt_in` is then 0).
    pub learnt_in_date: Option<Date>,
    pub overwrite: bool,
    pub stanzas: Vec<ShockStanza>,
    pub controlled: Vec<ControlledStanza>,
    /// `contains_date`: dates in `periods`, in `learnt_in` or in a `learnt_in(DATE)` scope.
    pub contains_date: bool,
    /// False once a later `shock_paths(overwrite)` block with the same `learnt_in` removed it
    /// from `M_.shock_paths` (blocks keep their index: the `shock_paths` commands refer to it).
    pub active: bool,
}

impl ShockPathsBlock {
    /// `contains_endval`.
    pub fn contains_endval(&self) -> bool {
        self.stanzas.iter().any(|s| s.items.iter().any(|(r, _)| r.is_endval()))
    }
    /// Whether the block's `learnt_in` is the same option value as `(int, date)` (a date never
    /// matches an integer, as the generated `overwrite` filter).
    pub fn same_learnt_in(&self, int: usize, date: Option<Date>) -> bool {
        match (self.learnt_in_date, date) {
            (Some(a), Some(b)) => a == b,
            (None, None) => self.learnt_in == int,
            _ => false,
        }
    }
    /// Sets `contains_date` from the periods (the parser adds the scopes and `learnt_in`).
    pub fn periods_have_dates(&self) -> bool {
        self.stanzas.iter().any(|s| s.items.iter().any(|(r, _)| r.has_date()))
            || self.controlled.iter().any(|c| c.items.iter().any(|(_, d, _)| !d.is_empty()))
    }
}

/// declarations kept in `ModFile::q1`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Q1Decl {
    /// `M_.shock_paths` (blocks in file order; see [`ShockPathsBlock::active`]).
    pub shock_paths: Vec<ShockPathsBlock>,
    /// `varexo_det` of a stochastic model (see `Model::exo_det_names`): their entries of
    /// `M_.det_shocks` (`exo` = index among them) and `oo_.exo_det_steady_state`.
    pub exo_det_shocks: Vec<crate::modfile::DetShock>,
    pub exo_det_steady_state: Vec<f64>,
    // ---- shocks/shock_paths ----
    /// `M_.database` (`database db1, db2;`).
    pub databases: Vec<String>,
    /// The MATLAB workspace variables of the .mod file that dynare-rust evaluated.
    pub workspace: super::workspace::Workspace,
}

/// Scope prefixes of the parser's rewritten names (`initval.x` -> `__init__x`, ...).
pub const INIT_PREFIX: &str = "__init__";
pub const SELF_PREFIX: &str = "__self__";
pub const PREV_PREFIX: &str = "__prev__";

/// A scope of a `values` expression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scope {
    Init,
    SelfPath,
    Prev,
    LearntIn(usize, Option<Date>),
    Database(usize),
}

/// The rewritten name of `scope.var`.
pub fn scoped_name(scope: Scope, var: &str) -> String {
    match scope {
        Scope::Init => format!("{INIT_PREFIX}{var}"),
        Scope::SelfPath => format!("{SELF_PREFIX}{var}"),
        Scope::Prev => format!("{PREV_PREFIX}{var}"),
        Scope::LearntIn(n, None) => format!("__li{n}__{var}"),
        Scope::LearntIn(_, Some(d)) => format!("__lid{}_{}__{var}", d.freq.number(), d.time),
        Scope::Database(k) => format!("__db{k}__{var}"),
    }
}

/// Inverse of [`scoped_name`] (None: an unscoped name).
pub fn parse_scoped(name: &str) -> Option<(Scope, &str)> {
    if let Some(v) = name.strip_prefix(INIT_PREFIX) {
        return Some((Scope::Init, v));
    }
    if let Some(v) = name.strip_prefix(SELF_PREFIX) {
        return Some((Scope::SelfPath, v));
    }
    if let Some(v) = name.strip_prefix(PREV_PREFIX) {
        return Some((Scope::Prev, v));
    }
    let split = |rest: &'_ str| rest.find("__").map(|k| (rest[..k].to_string(), k + 2));
    if let Some(rest) = name.strip_prefix("__lid") {
        let (code, k) = split(rest)?;
        let (f, t) = code.split_once('_')?;
        let d = Date::new(crate::dates::Freq::from_number(f.parse().ok()?)?, t.parse().ok()?);
        return Some((Scope::LearntIn(0, Some(d)), &rest[k..]));
    }
    if let Some(rest) = name.strip_prefix("__li") {
        let (code, k) = split(rest)?;
        return Some((Scope::LearntIn(code.parse().ok()?, None), &rest[k..]));
    }
    if let Some(rest) = name.strip_prefix("__db") {
        let (code, k) = split(rest)?;
        return Some((Scope::Database(code.parse().ok()?), &rest[k..]));
    }
    None
}

/// What the evaluation of a value reads.
struct Ctx<'a> {
    mf: &'a ModFile,
    params: &'a [f64],
    ys: &'a [f64],
    xs: &'a [f64],
    fsp: Option<Date>,
}

impl Ctx<'_> {
    fn model(&self) -> &Model {
        &self.mf.model
    }

    fn exo(&self, v: &str, scope: &str) -> Result<usize> {
        match self.model().symbol(v) {
            Some(s) if s.kind == SymbolKind::Exo => Ok(s.index),
            _ => bail!("shock_paths: {scope}.{v}: '{v}' is not an exogenous variable"),
        }
    }

    /// The value of `name(lag)` at period `p` of informational period `info` (1-based) over
    /// the paths `paths[i]` (informational period `i + 1`; `None`: no paths, controlled stanzas).
    fn value(&self, name: &str, lag: i32, p: usize, info: i64, paths: Option<&[Mat]>) -> Result<f64> {
        let Some((scope, v)) = parse_scoped(name) else {
            return match self.model().symbol(name) {
                Some(s) if s.kind == SymbolKind::Param => Ok(self.params[s.index]),
                Some(_) => bail!("shock_paths: variable '{name}' must have a scope (initval., self., ...)"),
                None => match name {
                    "pi" => Ok(std::f64::consts::PI),
                    "inf" | "Inf" => Ok(f64::INFINITY),
                    "nan" | "NaN" => Ok(f64::NAN),
                    _ => bail!("shock_paths: unknown symbol '{name}'"),
                },
            };
        };
        let q = p as i64 + lag as i64;
        let path_value = |ip: i64, k: usize, what: &str| -> Result<f64> {
            let Some(paths) = paths else {
                bail!("shock_paths: only the initval scope is supported in exogenize stanzas")
            };
            if ip < 1 || ip as usize > paths.len() {
                bail!("shock_paths: {what}.{v}: informational period {ip} does not exist");
            }
            let m = &paths[ip as usize - 1];
            if q < 1 || q as usize > m.ncols() {
                bail!("shock_paths: {what}.{v} at period {q} is outside the simulation");
            }
            Ok(m[(k, q as usize - 1)])
        };
        match scope {
            Scope::Init => {
                if lag != 0 {
                    bail!("shock_paths: initval.{v} cannot have a lead or lag");
                }
                match self.model().symbol(v) {
                    Some(s) if s.kind == SymbolKind::Endo => Ok(self.ys[s.index]),
                    Some(s) if s.kind == SymbolKind::Exo => Ok(self.xs[s.index]),
                    _ => bail!("shock_paths: '{v}' is not a variable"),
                }
            }
            Scope::SelfPath => path_value(info, self.exo(v, "self")?, "self"),
            Scope::Prev => path_value(info - 1, self.exo(v, "prev")?, "prev"),
            Scope::LearntIn(n, d) => {
                let ip = super::periods::learnt_in_period(n, d, self.fsp)?;
                path_value(ip as i64, self.exo(v, "learnt_in")?, "learnt_in")
            }
            Scope::Database(k) => {
                if paths.is_none() {
                    bail!("shock_paths: only the initval scope is supported in exogenize stanzas");
                }
                let db = self.mf.q1.databases.get(k).ok_or_else(|| anyhow!("shock_paths: unknown database"))?;
                super::workspace::extract_from_database(&self.mf.q1.workspace, db, v, q, self.fsp)
            }
        }
    }
}

/// The generated `shock_paths_<k>` function: sets, at period `p`, the stanzas of `b` on the
/// path of informational period `info` (`paths[info - 1]`).
fn evaluate_block(
    c: &Ctx,
    b: &ShockPathsBlock,
    p: usize,
    periods: usize,
    info: usize,
    paths: &mut [Mat],
) -> Result<()> {
    for s in &b.stanzas {
        for (r, ast) in &s.items {
            if !r.contains(p, periods, c.fsp)? {
                continue;
            }
            let v = crate::modfile::eval_ast(ast, &|n, l, _loc| c.value(n, l, p, info as i64, Some(&*paths)))?;
            paths[info - 1][(s.exo, p - 1)] = v;
        }
    }
    Ok(())
}

/// `make_ex_.m`'s `shock_paths` part: evaluates the blocks over `exo_paths` (`n_exo x
/// (periods + contains_endval)`, column p-1 = period p), in place.
pub fn evaluate_paths(
    mf: &ModFile,
    params: &[f64],
    ys: &[f64],
    xs: &[f64],
    periods: usize,
    fsp: Option<Date>,
    exo_paths: &mut Mat,
) -> Result<()> {
    let c = Ctx { mf, params, ys, xs, fsp };
    let mut paths = vec![std::mem::replace(exo_paths, Mat::zeros(0, 0))];
    let r = (|| -> Result<()> {
        for b in mf.q1.shock_paths.iter().filter(|b| b.active) {
            if b.learnt_in_date.is_some() || b.learnt_in != 1 {
                bail!(
                    "make_ex_: shock_paths(learnt_in=...) blocks need perfect_foresight_with_expectation_errors_setup (assertion M_.shock_paths(i).learnt_in == 1 failed)"
                );
            }
            for p in 1..=periods + b.contains_endval() as usize {
                evaluate_block(&c, b, p, periods, 1, &mut paths)?;
            }
        }
        Ok(())
    })();
    *exo_paths = paths.pop().unwrap();
    r
}

/// `perfect_foresight_with_expectation_errors_setup.m` with `shock_paths` blocks: the
/// `n_exo x (periods + 1)` path anticipated in each informational period (index 0 = period 1).
pub fn information_paths(
    mf: &ModFile,
    params: &[f64],
    ys: &[f64],
    xs: &[f64],
    periods: usize,
    fsp: Option<Date>,
) -> Result<Vec<Mat>> {
    let c = Ctx { mf, params, ys, xs, fsp };
    let nx = mf.model.n_exo();
    let mut paths: Vec<Mat> = Vec::with_capacity(periods);
    for info in 1..=periods {
        let m = if info == 1 { Mat::from_fn(nx, periods + 1, |k, _| xs[k]) } else { paths[info - 2].clone() };
        paths.push(m);
        let blocks = mf.q1.shock_paths.iter().filter(|b| b.active);
        // numeric learnt_in first, then dates (as the MATLAB block_idx)
        let ints: Vec<&ShockPathsBlock> =
            blocks.clone().filter(|b| b.learnt_in_date.is_none() && b.learnt_in == info).collect();
        let mut dated: Vec<&ShockPathsBlock> = Vec::new();
        if let Some(f) = fsp {
            dated = blocks.filter(|b| b.learnt_in_date == Some(f.add(info as i64 - 1))).collect();
        }
        for b in ints.into_iter().chain(dated) {
            for p in info..=periods + b.contains_endval() as usize {
                evaluate_block(&c, b, p, periods, info, &mut paths)?;
            }
        }
    }
    Ok(paths)
}

/// `perfect_foresight_setup.m`'s checks on `M_.shock_paths` and `M_.database`.
pub fn check_setup(mf: &ModFile, last_simulation_period: Option<Date>) -> Result<()> {
    let active: Vec<&ShockPathsBlock> = mf.q1.shock_paths.iter().filter(|b| b.active).collect();
    if active.iter().any(|b| b.contains_date) && last_simulation_period.is_none() {
        bail!(
            "perfect_foresight_setup: the shock_paths block contains dates but neither first_simulation_period nor last_simulation_period option was passed"
        );
    }
    if !active.is_empty() {
        check_databases(mf, "perfect_foresight_setup")?;
    }
    Ok(())
}

/// `evalin('base', "exist('db', 'var')")` for every database.
pub fn check_databases(mf: &ModFile, caller: &str) -> Result<()> {
    for d in &mf.q1.databases {
        if !mf.q1.workspace.vars.contains_key(d) {
            bail!(
                "{caller}: database {d} does not exist in the base workspace (dynare-rust evaluates dseries(...), table(...) and array2table(...) assignments of the .mod file)"
            );
        }
    }
    Ok(())
}

/// The controlled stanzas of `b` as `M_.perfect_foresight_controlled_paths` entries,
/// evaluated at the block's position (`initval` scope only).
pub fn controlled_paths(
    mf: &ModFile,
    b: &ShockPathsBlock,
    params: &[f64],
    ys: &[f64],
    xs: &[f64],
) -> Result<Vec<ControlledPath>> {
    let c = Ctx { mf, params, ys, xs, fsp: None };
    let mut out = Vec::new();
    for cs in &b.controlled {
        for (periods, dates, ast) in &cs.items {
            let value = crate::modfile::eval_ast(ast, &|n, l, _loc| c.value(n, l, 1, 1, None))?;
            out.push(ControlledPath {
                learnt_in: b.learnt_in,
                exogenize: cs.exogenize,
                endogenize: cs.endogenize,
                periods: periods.clone(),
                value,
                learnt_in_date: b.learnt_in_date,
                dates: dates.clone(),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_names_round_trip() {
        let d = Date::parse("2000Q3").unwrap();
        for s in [
            Scope::Init,
            Scope::SelfPath,
            Scope::Prev,
            Scope::LearntIn(4, None),
            Scope::LearntIn(0, Some(d)),
            Scope::Database(2),
        ] {
            let n = scoped_name(s, "x_1__y");
            assert_eq!(parse_scoped(&n), Some((s, "x_1__y")), "{n}");
        }
        assert_eq!(parse_scoped("alpha"), None);
    }
}
