// SPDX-License-Identifier: GPL-3.0-or-later
//! The `occbin_constraints` block, compiled: Dynare's generated `<fname>.occbin_difference.m`.
//!
//! The expressions are evaluated on the levels of the simulated paths (one row per period),
//! with `STEADY_STATE(x)` reading the steady state. As the preprocessor does, a missing
//! `relax` condition is `~bind`, a missing `error_bind` is `abs(lhs - rhs)` of the `bind`
//! inequality, and a missing `error_relax` is `abs(lhs - rhs)` of `relax` (or `error_bind`
//! when `relax` is missing too).

use anyhow::{Result, anyhow, bail};

use super::OccbinConstraint;
use crate::linalg::Mat;
use crate::model::expr::{BinaryOp, UnaryOp};
use crate::model::{Model, SymbolKind};
use crate::modfile::Ast;

#[derive(Clone, Debug)]
enum CExpr {
    Num(f64),
    Endo(usize),
    SteadyState(usize),
    Param(usize),
    Un(UnaryOp, Box<CExpr>),
    Bin(BinaryOp, Box<CExpr>, Box<CExpr>),
}

impl CExpr {
    fn eval(&self, y: &[f64], p: &[f64], ys: &[f64]) -> f64 {
        match self {
            CExpr::Num(v) => *v,
            CExpr::Endo(j) => y[*j],
            CExpr::SteadyState(j) => ys[*j],
            CExpr::Param(k) => p[*k],
            CExpr::Un(op, a) => op.apply(a.eval(y, p, ys)),
            CExpr::Bin(op, a, b) => op.apply(a.eval(y, p, ys), b.eval(y, p, ys)),
        }
    }
}

fn is_comparison(op: BinaryOp) -> bool {
    matches!(op, BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge | BinaryOp::Eq | BinaryOp::Ne)
}

fn compile(ast: &Ast, m: &Model) -> Result<CExpr> {
    Ok(match ast {
        Ast::Num(v) => CExpr::Num(*v),
        Ast::Var { name, lag, loc } => {
            if *lag != 0 {
                bail!(
                    "{loc}: occbin_constraints may only contain contemporaneous variables ('{name}' has lead/lag {lag}); define an auxiliary variable"
                );
            }
            match m.symbol(name) {
                Some(s) if s.kind == SymbolKind::Endo => CExpr::Endo(s.index),
                Some(s) if s.kind == SymbolKind::Param => CExpr::Param(s.index),
                Some(_) => bail!("{loc}: occbin_constraints: '{name}' must be an endogenous variable or a parameter"),
                None => match name.as_str() {
                    "inf" | "Inf" => CExpr::Num(f64::INFINITY),
                    "nan" | "NaN" => CExpr::Num(f64::NAN),
                    "pi" => CExpr::Num(std::f64::consts::PI),
                    _ => bail!("{loc}: unknown symbol '{name}'"),
                },
            }
        }
        Ast::Un(op, a) => CExpr::Un(*op, Box::new(compile(a, m)?)),
        Ast::Bin(op, a, b) => CExpr::Bin(*op, Box::new(compile(a, m)?), Box::new(compile(b, m)?)),
        // stream PP: STEADY_STATE(expr) reads the steady state of every variable in expr
        Ast::Op(crate::modfile::SpecialOp::SteadyState, args) => to_steady_state(compile(&args[0], m)?),
        Ast::Op(op, _) => bail!("{}() is not supported in occbin_constraints", op.name()),
    })
}

fn to_steady_state(e: CExpr) -> CExpr {
    match e {
        CExpr::Endo(j) => CExpr::SteadyState(j),
        CExpr::Un(op, a) => CExpr::Un(op, Box::new(to_steady_state(*a))),
        CExpr::Bin(op, a, b) => CExpr::Bin(op, Box::new(to_steady_state(*a)), Box::new(to_steady_state(*b))),
        other => other,
    }
}

/// `abs((lhs)-(rhs))` of an inequality (`abs(expr)` if it is not one).
fn default_error(e: &CExpr) -> CExpr {
    match e {
        CExpr::Bin(op, a, b) if is_comparison(*op) => {
            CExpr::Un(UnaryOp::Abs, Box::new(CExpr::Bin(BinaryOp::Sub, a.clone(), b.clone())))
        }
        other => CExpr::Un(UnaryOp::Abs, Box::new(other.clone())),
    }
}

#[derive(Clone, Debug)]
struct Compiled {
    bind: CExpr,
    /// None: `relax = ~bind`.
    relax: Option<CExpr>,
    err_bind: CExpr,
    /// None: `err_relax = err_bind`.
    err_relax: Option<CExpr>,
}

/// The compiled block (`M_.occbin.constraint_nbr` constraints, block order).
#[derive(Clone, Debug)]
pub struct Constraints {
    c: Vec<Compiled>,
    pub names: Vec<String>,
}

/// Output of `occbin_difference` for one path: per constraint, per period.
#[derive(Clone, Debug, Default)]
pub struct Difference {
    pub binding: Vec<Vec<bool>>,
    pub relax: Vec<Vec<bool>>,
    pub err_binding: Vec<Vec<f64>>,
    pub err_relax: Vec<Vec<f64>>,
}

impl Constraints {
    pub fn compile(decl: &[OccbinConstraint], m: &Model) -> Result<Constraints> {
        let mut c = Vec::new();
        for d in decl {
            let bind = compile(
                d.bind
                    .as_ref()
                    .ok_or_else(|| anyhow!("occbin_constraints: constraint '{}' has no bind condition", d.name))?,
                m,
            )?;
            let relax = d.relax.as_ref().map(|a| compile(a, m)).transpose()?;
            let err_bind = match &d.error_bind {
                Some(a) => compile(a, m)?,
                None => default_error(&bind),
            };
            let err_relax = match (&d.error_relax, &relax) {
                (Some(a), _) => Some(compile(a, m)?),
                (None, Some(r)) => Some(default_error(r)),
                (None, None) => None,
            };
            c.push(Compiled { bind, relax, err_bind, err_relax });
        }
        Ok(Constraints { c, names: decl.iter().map(|d| d.name.clone()).collect() })
    }

    pub fn len(&self) -> usize {
        self.c.len()
    }
    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    /// `occbin_difference(zdatalinear + ys', params, ys)`: `z` holds deviations from the
    /// steady state (rows = periods, columns = endogenous variables in declaration order).
    pub fn difference(&self, z: &Mat, p: &[f64], ys: &[f64]) -> Difference {
        let (nr, n) = z.shape();
        let mut d = Difference::default();
        let mut y = vec![0.0; n];
        for c in &self.c {
            let mut b = Vec::with_capacity(nr);
            let mut r = Vec::with_capacity(nr);
            let mut eb = Vec::with_capacity(nr);
            let mut er = Vec::with_capacity(nr);
            for t in 0..nr {
                for j in 0..n {
                    y[j] = z[(t, j)] + ys[j];
                }
                let bt = c.bind.eval(&y, p, ys) != 0.0;
                b.push(bt);
                r.push(match &c.relax {
                    Some(e) => e.eval(&y, p, ys) != 0.0,
                    None => !bt,
                });
                let e = c.err_bind.eval(&y, p, ys);
                eb.push(e);
                er.push(match &c.err_relax {
                    Some(x) => x.eval(&y, p, ys),
                    None => e,
                });
            }
            d.binding.push(b);
            d.relax.push(r);
            d.err_binding.push(eb);
            d.err_relax.push(er);
        }
        d
    }
}
