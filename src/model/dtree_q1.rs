// SPDX-License-Identifier: GPL-3.0-or-later
//! balanced-growth detrending and log-transformed variables in the preprocessor's
//! transform pass (`DynamicModel::{detrendEquations, testTrendDerivativesEqualToZero,
//! removeTrendVariableFromEquations, substituteLogTransform}`, `VariableNode::{detrend,
//! removeTrendLeadLag, replaceTrendVar, substituteLogTransform}`).

use super::{BgContext, Id, SymType, Tree};
use crate::model::expr::{Dag, DiffVar, Node, UnaryOp};
use anyhow::{Result, bail};
use std::collections::{BTreeSet, HashMap};

impl Tree {
    /// `VariableNode::detrend` through `recurseTransform` for one nonstationary variable:
    /// `x(l)` becomes `x(l)*deflator(l)` (`x(l)+deflator(l)` for a log deflator).
    fn detrend(&mut self, id: Id, sym: u32, log: bool, deflator: Id) -> Result<Id> {
        let mut leaf = |t: &mut Tree, id: Id, s: u32, lag: i32| -> Result<Option<Id>> {
            if s != sym {
                return Ok(None);
            }
            let tr = if lag == 0 { deflator } else { t.decrease(deflator, -lag) };
            Ok(Some(if log { t.add_plus(id, tr) } else { t.add_times(id, tr) }))
        };
        let mut memo = HashMap::new();
        self.recurse(id, &mut memo, &mut leaf)
    }

    /// `VariableNode::removeTrendLeadLag`: `A(+k) = A*g(+1)*...*g(+k)`, `A(-k) = A/(g*...*g(-k+1))`
    /// (sums / differences for log trends).
    fn remove_trend_lead_lag(&mut self, id: Id) -> Result<Id> {
        let map = self.trend_map.clone();
        let mut leaf = |t: &mut Tree, _id: Id, s: u32, lag: i32| -> Result<Option<Id>> {
            let ty = t.sym_ty(s);
            if !matches!(ty, SymType::Trend | SymType::LogTrend) || lag == 0 {
                return Ok(None);
            }
            let Some(&trend) = map.get(&s) else {
                bail!("trend variable '{}' has no growth factor", t.names[s as usize])
            };
            let no_lag = t.add_var(s, 0);
            let log = ty == SymType::LogTrend;
            Ok(Some(if lag > 0 {
                let mut seq = t.decrease(trend, -1);
                for i in 1..lag {
                    let g = t.decrease(trend, -(i + 1));
                    seq = if log { t.add_plus(seq, g) } else { t.add_times(seq, g) };
                }
                if log { t.add_plus(no_lag, seq) } else { t.add_times(no_lag, seq) }
            } else {
                let mut seq = trend;
                for i in 1..lag.abs() {
                    let g = t.decrease(trend, i);
                    seq = if log { t.add_plus(seq, g) } else { t.add_times(seq, g) };
                }
                if log { t.add_minus(no_lag, seq) } else { t.add_divide(no_lag, seq)? }
            }))
        };
        let mut memo = HashMap::new();
        self.recurse(id, &mut memo, &mut leaf)
    }

    /// `DynamicModel::detrendEquations` (backwards over the nonstationary variables, "to deal
    /// correctly with I(2) processes"), then `removeTrendLeadLag`.
    pub(super) fn detrend_equations(&mut self) -> Result<()> {
        let ns: Vec<(u32, (bool, Id))> = self.nonstationary.iter().rev().map(|(&s, &v)| (s, v)).collect();
        for (sym, (log, defl)) in ns {
            self.for_all_eqs(true, &mut |t, e| t.detrend(e, sym, log, defl))?;
        }
        self.for_all_eqs(true, &mut |t, e| t.remove_trend_lead_lag(e))
    }

    /// `DynamicModel::removeTrendVariableFromEquations` (`replaceTrendVar`: trend -> 1, log
    /// trend -> 0).
    pub(super) fn remove_trend_variables(&mut self) -> Result<()> {
        let mut leaf = |t: &mut Tree, _id: Id, s: u32, _lag: i32| -> Result<Option<Id>> {
            Ok(match t.sym_ty(s) {
                SymType::Trend => Some(t.one),
                SymType::LogTrend => Some(t.zero),
                _ => None,
            })
        };
        let mut memo = HashMap::new();
        self.for_all_eqs(true, &mut |t, e| t.recurse(e, &mut memo, &mut leaf))
    }

    /// `DynamicModel::testTrendDerivativesEqualToZero` on the detrended model (before the trend
    /// variables are removed): for every equation whose residual is not zero at the context,
    /// `d^2 log(lhs - rhs) / d trend d y` must not exceed `balanced_growth_test_tol` for any
    /// endogenous `y` (each lead/lag, in Dynare's deriv-id order: symbol id, then lag).
    pub(super) fn trend_test(&mut self, bg: &BgContext) -> Result<()> {
        let n_params = bg.params.len();
        let mut dag = Dag::new();
        self.trend_param_offset = Some(n_params);
        let mut memo = HashMap::new();
        let eqs = self.equations.clone();
        let mut resid = Vec::with_capacity(eqs.len());
        for e in &eqs {
            let (a, b) = self.eq_args(e.eq);
            let a = self.convert(a, &mut dag, &mut memo);
            let b = self.convert(b, &mut dag, &mut memo);
            let (a, b) = match (a, b) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(e), _) | (_, Err(e)) => {
                    self.trend_param_offset = None;
                    return Err(e);
                }
            };
            resid.push(dag.sub(a, b));
        }
        self.trend_param_offset = None;
        let mut trends = BTreeSet::new();
        let mut endo = BTreeSet::new();
        let mut leaves = Vec::new();
        for &r in &resid {
            leaves.clear();
            dag.collect_leaves(r, &mut leaves);
            for l in &leaves {
                match *l {
                    Node::Endo(j, lag) => {
                        endo.insert((j, lag));
                    }
                    Node::Param(k) if k as usize >= n_params => {
                        trends.insert(k);
                    }
                    _ => {}
                }
            }
        }
        let name_of = |t: &Tree, ty: &[SymType], idx: usize| -> String {
            t.syms
                .iter()
                .zip(&t.names)
                .find(|(s, _)| ty.contains(&s.ty) && s.index == idx)
                .map(|x| x.1.clone())
                .unwrap_or_default()
        };
        let leaf = |nd: Node| -> f64 {
            match nd {
                Node::Endo(j, _) => bg.endo.get(j as usize).copied().unwrap_or(0.0),
                Node::Exo(k, _) => bg.exo.get(k as usize).copied().unwrap_or(0.0),
                Node::Param(k) if (k as usize) < n_params => {
                    let v = bg.params[k as usize];
                    if v.is_nan() { 0.0 } else { v }
                }
                Node::Param(_) => 2.0,
                _ => 0.0,
            }
        };
        const ZERO_BAND: f64 = 1e-8;
        for &tk in &trends {
            for (eq, &h) in resid.iter().enumerate() {
                if dag.eval(h, &leaf).abs() <= ZERO_BAND {
                    continue;
                }
                let f = dag.unary(UnaryOp::Log, h);
                let ft = dag.diff(f, DiffVar::Param(tk), &mut HashMap::new());
                for &(j, lag) in &endo {
                    let d = dag.diff(ft, DiffVar::Endo(j, lag), &mut HashMap::new());
                    let v = dag.eval(d, &leaf);
                    if v.abs() > bg.tol {
                        bail!(
                            "ERROR: trends not compatible with balanced growth path; the second-order cross partial of equation {} (line {}) w.r.t. trend variable {} and endogenous variable {} is not null (abs. value = {}). If you are confident that your trends are correctly specified, you can raise the value of option 'balanced_growth_test_tol' in the 'model' block.",
                            eq + 1,
                            eqs[eq].line,
                            name_of(self, &[SymType::Trend, SymType::LogTrend], tk as usize - n_params),
                            name_of(self, &[SymType::Endo], j as usize),
                            crate::deterministic::fmt_g(v.abs())
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// `DynamicModel::substituteLogTransform`: for each `var(log)` variable `X`, a new
    /// variable `LOG_X` (aux type 7, defined as `log(X)`) replaces `X` by `exp(LOG_X)` in the
    /// model-local variables and the equations, and the equation `X = exp(LOG_X)` is appended.
    pub(super) fn substitute_log_transform(&mut self, log_vars: &[u32]) -> Result<()> {
        for &sym in log_vars {
            let v = self.add_var(sym, 0);
            let aux_def = self.add_fn(UnaryOp::Log, v);
            let name = format!("LOG_{}", self.names[sym as usize]);
            let aux = self.add_aux(name, 7, Some(sym), Some(0), aux_def);
            let mut leaf = move |t: &mut Tree, _id: Id, s: u32, lag: i32| -> Result<Option<Id>> {
                if s != sym {
                    return Ok(None);
                }
                let a = t.add_var(aux, lag);
                Ok(Some(t.add_fn(UnaryOp::Exp, a)))
            };
            let keys: Vec<u32> = self.locals.keys().copied().collect();
            for k in keys {
                let d = self.locals[&k];
                let mut memo = HashMap::new();
                let r = self.recurse(d, &mut memo, &mut leaf)?;
                self.set_local(k, r);
            }
            let mut memo = HashMap::new();
            self.for_all_eqs(true, &mut |t, e| t.recurse(e, &mut memo, &mut leaf))?;
            let x = self.add_var(sym, 0);
            let la = self.add_var(aux, 0);
            let ex = self.add_fn(UnaryOp::Exp, la);
            let eq = self.add_equal(x, ex);
            self.add_equation(eq, Vec::new());
            // addAuxEquation(LOG_X = log(X)): only in aux_equations (set_auxiliary_variables),
            // but the node counts in the numbering
            let la = self.add_var(aux, 0);
            self.add_equal(la, aux_def);
        }
        Ok(())
    }
}
