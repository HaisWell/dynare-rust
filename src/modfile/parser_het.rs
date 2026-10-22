// SPDX-License-Identifier: GPL-3.0-or-later
// ---- heterogeneity ----
//! Heterogeneous-agent models (Dynare 7): `heterogeneity_dimension`, `var/varexo/parameters
//! (heterogeneity=NAME)`, `model(heterogeneity=NAME)`, `shocks(heterogeneity=NAME)` and the
//! `SUM()` aggregation operator of the aggregate model, as `ParsingDriver` /
//! `SymbolTable` / `DynamicModel::substituteAggregationOperators` build them.
//!
//! The aggregate model is the ordinary model block. `SUM(x)` (`x` a heterogeneous variable) is
//! replaced by the auxiliary endogenous variable `SUM_x`, whose equation `SUM_x = <aggregate
//! of x>` is appended after the written equations; here the equation is stored as
//! `SUM_x = 0` and the aggregate is subtracted by [`crate::heterogeneity`] when evaluating
//! (it is a constant of the aggregate model, `yagg` in Dynare's `dynamic_resid`).

use super::{Ast, Parser, src_join};
use crate::heterogeneity::{HetAggregate, HetDecl, HetModel, HetSymbolDecl};
use crate::linalg::Mat;
use crate::model::dtree::ModelItem;
use crate::model::{AuxVar, Model, SymbolKind};
use crate::modfile::lexer::{Tok, Token};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HetKind {
    Endo,
    Exo,
    Param,
}

#[derive(Clone, Debug)]
pub(super) enum HetShock {
    Var(String, f64),
    Std(String, f64),
    Cov(String, String, f64),
    Corr(String, String, f64),
}

/// Parser state of the heterogeneity statements.
#[derive(Default)]
pub(super) struct HetParse {
    pub dim: Option<String>,
    pub endo: Vec<HetSymbolDecl>,
    pub exo: Vec<HetSymbolDecl>,
    pub params: Vec<HetSymbolDecl>,
    names: HashMap<String, HetKind>,
    items: Vec<ModelItem>,
    /// Heterogeneous variables aggregated by `SUM()`, in order of first appearance, with the
    /// aggregate auxiliary variable `SUM_x` (declared endogenous index).
    sums: Vec<(String, usize)>,
    shocks: Vec<HetShock>,
}

/// The value of `heterogeneity = NAME` in the option list starting at the cursor, if any.
fn het_option(toks: &[Token]) -> Option<String> {
    // `( heterogeneity = NAME ... )`
    if toks.len() >= 4 && toks[0].is("(") && toks[1].is_ident("heterogeneity") && toks[2].is("=") {
        return toks[3].ident().map(str::to_string);
    }
    None
}

impl<'a> Parser<'a> {
    /// Is the statement at the cursor `kw(heterogeneity = NAME)`? Returns NAME.
    pub(super) fn het_header(&self) -> Option<String> {
        let (st, mut k) = (self.c.i + 1, 0);
        let toks = &self.c.t[st..];
        while k < toks.len() && !toks[k].is(")") && !toks[k].is(";") {
            k += 1;
        }
        het_option(&toks[..k.min(toks.len())])
    }

    fn het_check_dim(&self, name: &str) -> Result<()> {
        match &self.het.dim {
            Some(d) if d == name => Ok(()),
            _ => bail!("{}: Unknown heterogeneity dimension: {name}", self.c.loc()),
        }
    }

    /// `heterogeneity_dimension NAME;`
    pub(super) fn het_dimension(&mut self) -> Result<()> {
        self.c.i += 1;
        let toks = self.c.statement()?;
        for t in toks {
            if t.is(",") {
                continue;
            }
            let Some(n) = t.ident() else { bail!("{}: unexpected '{}' in heterogeneity_dimension", t.loc, t.raw) };
            if self.het.dim.is_some() {
                bail!("{}: Heterogeneity dimension '{n}': only one heterogeneity dimension is supported", t.loc);
            }
            self.het.dim = Some(n.to_string());
            self.c.het_dim = true;
        }
        Ok(())
    }

    /// `var(heterogeneity=NAME) ...;`, `varexo(...)`, `parameters(...)`.
    pub(super) fn het_declaration(&mut self, kind: HetKind) -> Result<()> {
        let dim = self.het_header().unwrap_or_default();
        self.c.i += 1;
        self.c.skip_parens()?;
        self.het_check_dim(&dim)?;
        loop {
            if self.c.eat(";") {
                return Ok(());
            }
            if self.c.eat(",") {
                continue;
            }
            let (name, loc) = self.c.ident()?;
            if self.b.symbol(name).is_some() || self.het.names.contains_key(name) {
                bail!("{loc}: symbol '{name}' declared twice");
            }
            let mut tex = name.replace('_', "\\_");
            let mut long = name.to_string();
            while let Some(t) = self.c.peek() {
                if let Tok::Tex(s) = &t.tok {
                    tex = s.clone();
                    self.c.i += 1;
                } else if t.is("(") {
                    let st = self.c.i;
                    self.c.skip_parens()?;
                    for item in super::split_commas(&self.c.t[st + 1..self.c.i - 1]) {
                        if let [k, eq, v] = item
                            && k.is_ident("long_name")
                            && eq.is("=")
                            && let Tok::Str(s) = &v.tok
                        {
                            long = s.clone();
                        }
                    }
                } else {
                    break;
                }
            }
            self.het.names.insert(name.to_string(), kind);
            self.b.reserve_symbol_id(); // Dynare's symbol ids count the heterogeneous symbols
            let d = HetSymbolDecl { name: name.to_string(), tex, long };
            match kind {
                HetKind::Endo => self.het.endo.push(d),
                HetKind::Exo => self.het.exo.push(d),
                HetKind::Param => self.het.params.push(d),
            }
        }
    }

    /// `model(heterogeneity=NAME); equations end;`
    pub(super) fn het_model_block(&mut self) -> Result<()> {
        let dim = self.het_header().unwrap_or_default();
        self.c.i += 1;
        self.c.skip_parens()?;
        self.c.expect(";")?;
        self.het_check_dim(&dim)?;
        self.model_seen = true;
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            let mut tags = Vec::new();
            if self.c.eat("[") {
                loop {
                    if self.c.eat("]") {
                        break;
                    }
                    if self.c.eat(",") {
                        continue;
                    }
                    let key = self.c.next()?.raw.to_ascii_lowercase();
                    let mut val = String::new();
                    if self.c.eat("=") {
                        let v = self.c.next()?;
                        val = match &v.tok {
                            Tok::Str(s) => s.clone(),
                            _ => v.raw.clone(),
                        };
                    }
                    tags.push((key, val));
                }
            }
            self.c.nums.clear();
            self.c.sums.clear();
            let line = self.c.peek().map_or(0, |t| t.loc.line);
            let lhs = self.c.expr()?;
            let has_rhs = self.c.eat("=");
            let rhs = if has_rhs { self.c.expr()? } else { Ast::Num(0.0) };
            if !self.c.sums.is_empty() {
                bail!("{}: The SUM() operator cannot be used inside a model(heterogeneity=...) block", self.c.loc());
            }
            if self.c.eat("⟂") {
                let st = self.c.i;
                while self.c.peek().is_some_and(|t| !t.is(";")) {
                    self.c.i += 1;
                }
                tags.push((crate::model::MCP_TAG.to_string(), src_join(&self.c.t[st..self.c.i])));
            }
            self.c.expect(";")?;
            let nums = std::mem::take(&mut self.c.nums);
            self.het.items.push(ModelItem::Equation { lhs, rhs: has_rhs.then_some(rhs), nums, tags, line });
        }
    }

    /// Called by the aggregate model block after each equation: declares the aggregate
    /// auxiliary variables `SUM_x` for the `SUM(x)` found in it.
    pub(super) fn het_register_sums(&mut self) -> Result<()> {
        let sums = std::mem::take(&mut self.c.sums);
        for x in sums {
            match self.het.names.get(&x) {
                Some(HetKind::Endo) => {}
                _ => bail!(
                    "{}: The argument of SUM() must be a heterogeneous endogenous variable ('{x}' is not one)",
                    self.c.loc()
                ),
            }
            if self.het.sums.iter().any(|(n, _)| *n == x) {
                continue;
            }
            let name = format!("SUM_{x}");
            let k = self
                .b
                .declare(&name, SymbolKind::Endo)
                .map_err(|_| anyhow!("{}: the variable/parameter '{name}' conflicts with a variable that will be generated for an aggregation operator. Please rename it.", self.c.loc()))?;
            self.labels.push(SymbolKind::Endo, name.replace('_', "\\_"), name.clone());
            self.het.sums.push((x, k));
        }
        Ok(())
    }

    /// `shocks(heterogeneity=NAME[, overwrite]); var e; stderr s; ... end;`
    pub(super) fn het_shocks_block(&mut self, dim: &str, overwrite: bool) -> Result<()> {
        self.het_check_dim(dim)?;
        if overwrite {
            self.het.shocks.clear();
        }
        let check = |p: &Self, n: &str, loc: &dyn std::fmt::Display| -> Result<()> {
            if p.het.names.get(n) != Some(&HetKind::Exo) {
                bail!(
                    "{loc}: shocks: setting a moment on '{n}' is not allowed, because it is not a heterogeneous exogenous variable"
                );
            }
            Ok(())
        };
        loop {
            if self.c.at_end() {
                self.c.i += 2;
                return Ok(());
            }
            let corr = self.c.at_ident("corr");
            if corr || self.c.at_ident("var") {
                self.c.i += 1;
                let (a, la) = self.c.ident()?;
                check(self, a, la)?;
                let b = if self.c.eat(",") {
                    let (b, lb) = self.c.ident()?;
                    check(self, b, lb)?;
                    Some(b.to_string())
                } else {
                    None
                };
                if self.c.eat("=") {
                    let toks = self.c.statement()?;
                    let v = self.eval_tokens(toks)?;
                    self.het.shocks.push(match (b, corr) {
                        (None, false) => HetShock::Var(a.to_string(), v),
                        (Some(b), false) => HetShock::Cov(a.to_string(), b, v),
                        (Some(b), true) => HetShock::Corr(a.to_string(), b, v),
                        (None, true) => bail!("{la}: corr requires two variables"),
                    });
                    continue;
                }
                self.c.expect(";")?;
                if self.c.at_ident("stderr") {
                    self.c.i += 1;
                    let toks = self.c.statement()?;
                    let v = self.eval_tokens(toks)?;
                    self.het.shocks.push(HetShock::Std(a.to_string(), v));
                    continue;
                }
                bail!("{la}: shocks(heterogeneity=...): expected 'stderr' after 'var {a};'");
            }
            let t = self.c.next()?;
            bail!("{}: unexpected '{}' in a shocks(heterogeneity=...) block", t.loc, t.raw);
        }
    }

    /// Adds the `SUM_x = 0` equations after all the model blocks (Dynare appends the
    /// aggregation equations to the written ones).
    pub(super) fn het_finish_equations(&mut self) -> Result<()> {
        let sums = self.het.sums.clone();
        for (x, _) in sums {
            let lhs = Ast::Var {
                name: format!("SUM_{x}"),
                lag: 0,
                loc: crate::modfile::lexer::Loc { file: std::sync::Arc::from("<heterogeneity>"), line: 0 },
            };
            let rhs = Ast::Num(0.0);
            let l = self.resolve_ast(&lhs)?;
            let r = self.resolve_ast(&rhs)?;
            self.b.items.push(ModelItem::Equation {
                lhs,
                rhs: Some(rhs),
                nums: vec!["0".into()],
                tags: vec![("__het_agg".into(), String::new())],
                line: 0,
            });
            self.b.add_equation(l, r, vec![("__het_agg".into(), String::new())]);
        }
        Ok(())
    }
}

impl HetParse {
    /// Builds the heterogeneous model once the aggregate `model` is built.
    pub(super) fn het_build(&mut self, model: &mut Model) -> Result<HetDecl> {
        let Some(dim) = self.dim.clone() else { return Ok(HetDecl::default()) };
        let n_sum = self.sums.len();
        // the aggregate auxiliary variables are not original endogenous variables
        model.orig_endo_nbr -= n_sum;
        model.orig_eq_nbr -= n_sum;
        // (the aggregation equations are added after the equations got their names)
        for k in 0..n_sum {
            model.equation_tags[model.orig_eq_nbr + k].clear();
        }
        let mut aggregates = Vec::new();
        let mut agg_aux = Vec::new();
        for (x, k) in &self.sums {
            let het_endo = self.endo.iter().position(|d| d.name == *x).unwrap();
            aggregates.push(HetAggregate { op: "sum".into(), dimension: 0, het_endo });
            agg_aux.push(AuxVar {
                endo_index: *k,
                ty: 14,
                orig_index: None,
                orig_lead_lag: None,
                orig_expr: format!("sum({x})"),
            });
        }
        let mut sigma = Mat::zeros(self.exo.len(), self.exo.len());
        let pos = |p: &Self, n: &str| p.exo.iter().position(|d| d.name == n).unwrap();
        let mut corrs = Vec::new();
        for s in &self.shocks {
            match s {
                HetShock::Var(a, v) => {
                    let i = pos(self, a);
                    sigma[(i, i)] = *v;
                }
                HetShock::Std(a, v) => {
                    let i = pos(self, a);
                    sigma[(i, i)] = v * v;
                }
                HetShock::Cov(a, b, v) => {
                    let (i, j) = (pos(self, a), pos(self, b));
                    sigma[(i, j)] = *v;
                    sigma[(j, i)] = *v;
                }
                HetShock::Corr(a, b, v) => corrs.push((pos(self, a), pos(self, b), *v)),
            }
        }
        for (i, j, v) in corrs {
            let c = v * (sigma[(i, i)] * sigma[(j, j)]).sqrt();
            sigma[(i, j)] = c;
            sigma[(j, i)] = c;
        }
        let items = std::mem::take(&mut self.items);
        let decl = crate::heterogeneity::HetModelDecl {
            dimension_name: dim,
            endo: self.endo.clone(),
            exo: self.exo.clone(),
            params: self.params.clone(),
            sigma_e: sigma,
        };
        let hm = HetModel::build(decl, &items, model)?;
        Ok(HetDecl { model: Some(hm), aggregates, aggregate_aux: agg_aux })
    }
}
