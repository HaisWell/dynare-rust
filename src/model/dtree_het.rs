// SPDX-License-Identifier: GPL-3.0-or-later
// ---- heterogeneity ----
//! The `model(heterogeneity = NAME)` block through the DataTree emulation: port of the parts of
//! `HeterogeneousModel` (`HeterogeneousModel.cc`) and of `ExprNode` that shape the model the
//! preprocessor writes to `dynamic_het1_*.m`:
//!
//! * `checkPass` (lead/lag limits, non-separable lead/lag expressions),
//! * `substituteHetEndoNonLinearLead` (auxiliary variables `AUX_HET_ENDO_LEAD_<idx>` for
//!   non-linear expressions of leaded heterogeneous variables, named after the internal
//!   node index like Dynare does),
//! * `transformPass` (unfolding of complementarity conditions into multipliers
//!   `MULT_L_<var>` / `MULT_U_<var>`, ordering of the auxiliary equations),
//! * the topological levels of the auxiliary variables (`computeHetAuxTopologicalLevels`).

use super::{BOp, Id, N, SubstTable, SymType, Tree};
use super::{ModelItem, Sym};
use crate::model::expr::{Dag, NodeId};
use crate::modfile::parse_expression;
use anyhow::{Result, bail};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Input of [`het_transform`]. Symbol ids index `syms`/`names`: the heterogeneous endogenous
/// variables of the dimension (`SymType::Endo`, index = position), its heterogeneous exogenous
/// ones (`Exo`), the aggregate endogenous / exogenous ones (`AggEndo` / `AggExo`) and the
/// parameters (`Param`).
pub(crate) struct HetInput<'a> {
    pub items: &'a [ModelItem],
    pub syms: Vec<Sym>,
    pub names: Vec<String>,
    pub n_het_endo: usize,
    pub n_het_exo: usize,
    pub lookup: &'a dyn Fn(&str) -> Option<u32>,
    pub dim_name: &'a str,
}

/// An auxiliary heterogeneous variable (`M_.heterogeneity(1).aux_vars`).
#[derive(Clone, Debug)]
pub struct HetAuxOut {
    pub name: String,
    /// Dynare's `AuxVarType`: 15 multiplier, 16 non-linear lead.
    pub ty: u8,
    /// 1-based `equation_number_for_multiplier + 1` (the equation of the multiplier, else 1).
    pub eq_nbr: usize,
    /// 0-based index among the heterogeneous endogenous variables.
    pub endo_index: usize,
}

/// Result of [`het_transform`]; expressions live in the `Dag` given to it. In the `Dag`,
/// endogenous index `j < n_het_endo` is heterogeneous variable `j` and `n_het_endo + j` is
/// aggregate endogenous variable `j`; exogenous index `k < n_het_exo` is heterogeneous shock
/// `k` and `n_het_exo + k` is aggregate shock `k`.
pub struct HetOut {
    pub equations: Vec<(NodeId, NodeId)>,
    pub tags: Vec<Vec<(String, String)>>,
    /// All heterogeneous endogenous names: declared ones then the auxiliary variables.
    pub aux: Vec<HetAuxOut>,
    pub n_het_endo: usize,
    /// Per equation: complementarity condition (variable index, lower bound, upper bound).
    pub mcp: Vec<Option<(usize, Option<NodeId>, Option<NodeId>)>>,
    /// Auxiliary equations `endo_index = rhs` in the order the preprocessor writes them
    /// (`dynamic_het1_set_auxiliary_variables.m`).
    pub aux_equations: Vec<(usize, NodeId)>,
    /// `het_aux_levels` (0-based het endo indices per level).
    pub aux_levels: Vec<Vec<usize>>,
}

type Mcp = Option<(u32, Option<Id>, Option<Id>)>;

impl Tree {
    /// `findNonSeparableHetLeadLagSubexpr`.
    fn find_non_separable(&mut self, id: Id) -> Option<Id> {
        match self.node(id) {
            N::Num(_) => None,
            N::Var(s, lag) => {
                if self.sym_ty(s) == SymType::Local {
                    let v = self.get_local(s, lag);
                    self.find_non_separable(v)
                } else {
                    None
                }
            }
            N::Un(_, a, _) => {
                if let Some(x) = self.find_non_separable(a) {
                    return Some(x);
                }
                (self.max_endo_lead(a) >= 1 && self.max_endo_lag(a) >= 1).then_some(id)
            }
            N::Bin(op, a, b) => {
                if let Some(x) = self.find_non_separable(a) {
                    return Some(x);
                }
                if let Some(x) = self.find_non_separable(b) {
                    return Some(x);
                }
                let (l1, l2) = (self.max_endo_lead(a), self.max_endo_lead(b));
                let (g1, g2) = (self.max_endo_lag(a), self.max_endo_lag(b));
                if l1 < 1 && l2 < 1 {
                    return None;
                }
                match op {
                    BOp::Plus | BOp::Minus | BOp::Equal => None,
                    BOp::Times | BOp::Divide => {
                        if l1 >= 1 && l2 == 0 && self.max_exo_lead(b) == 0 {
                            return None;
                        }
                        if l1 == 0 && self.max_exo_lead(a) == 0 && l2 >= 1 && op == BOp::Times {
                            return None;
                        }
                        ((l1 >= 1 || l2 >= 1) && g1.max(g2) >= 1).then_some(id)
                    }
                    _ => ((l1 >= 1 || l2 >= 1) && g1.max(g2) >= 1).then_some(id),
                }
            }
            N::Tri(_, a, b, c) => {
                let ls = self.max_endo_lead(a).max(self.max_endo_lead(b)).max(self.max_endo_lead(c));
                let gs = self.max_endo_lag(a).max(self.max_endo_lag(b)).max(self.max_endo_lag(c));
                (ls >= 1 && gs >= 1).then_some(id)
            }
            N::Ext(..) => (self.max_endo_lead(id) >= 1 && self.max_endo_lag(id) >= 1).then_some(id),
        }
    }

    /// `createHetEndoLeadAuxiliaryVarForMyself`.
    fn create_het_lead_aux(&mut self, id: Id, table: &mut SubstTable, neweqs: &mut Vec<Id>) -> Id {
        let n = self.max_endo_lead(id);
        assert!(n >= 1);
        if let Some(&r) = table.get(&id) {
            return r;
        }
        let mut substexpr = self.decrease(id, n);
        let mut lag = n - 1;
        while lag >= 0 {
            let orig = self.decrease(id, lag);
            if let Some(&r) = table.get(&orig) {
                substexpr = r;
            } else {
                let name = format!("AUX_HET_ENDO_LEAD_{orig}");
                let symb = self.add_aux(name, 16, None, None, substexpr);
                let v0 = self.add_var(symb, 0);
                neweqs.push(self.add_equal(v0, substexpr));
                substexpr = self.add_var(symb, 1);
                table.insert(orig, substexpr);
            }
            lag -= 1;
        }
        substexpr
    }

    /// `substituteHetEndoNonLinearLead`.
    fn subst_het_lead(&mut self, id: Id, table: &mut SubstTable, neweqs: &mut Vec<Id>) -> Result<Id> {
        match self.node(id) {
            N::Num(_) => Ok(id),
            N::Var(s, lag) => match self.sym_ty(s) {
                SymType::Endo if lag > 1 => Ok(self.create_het_lead_aux(id, table, neweqs)),
                SymType::Local => {
                    let v = self.get_local(s, lag);
                    if self.max_endo_lead(v) <= 1 { Ok(id) } else { self.subst_het_lead(v, table, neweqs) }
                }
                _ => Ok(id),
            },
            N::Un(op, a, info) => {
                if op == super::UOp::Uminus {
                    let a2 = self.subst_het_lead(a, table, neweqs)?;
                    Ok(self.build_un(op, info, a2))
                } else if self.max_endo_lead(a) >= 1 {
                    Ok(self.create_het_lead_aux(id, table, neweqs))
                } else {
                    Ok(id)
                }
            }
            N::Bin(op, a, b) => {
                let (l1, l2) = (self.max_endo_lead(a), self.max_endo_lead(b));
                if l1 < 1 && l2 < 1 {
                    return Ok(id);
                }
                match op {
                    BOp::Plus | BOp::Minus | BOp::Equal => {
                        let a2 = if l1 >= 1 { self.subst_het_lead(a, table, neweqs)? } else { a };
                        let b2 = if l2 >= 1 { self.subst_het_lead(b, table, neweqs)? } else { b };
                        self.build_bin(op, a2, b2)
                    }
                    BOp::Times | BOp::Divide => {
                        if l1 >= 1 && l2 == 0 && self.max_exo_lead(b) == 0 {
                            let a2 = self.subst_het_lead(a, table, neweqs)?;
                            return self.build_bin(op, a2, b);
                        }
                        if l1 == 0 && self.max_exo_lead(a) == 0 && l2 >= 1 && op == BOp::Times {
                            let b2 = self.subst_het_lead(b, table, neweqs)?;
                            return self.build_bin(op, a, b2);
                        }
                        Ok(self.create_het_lead_aux(id, table, neweqs))
                    }
                    _ => Ok(self.create_het_lead_aux(id, table, neweqs)),
                }
            }
            N::Tri(..) | N::Ext(..) => {
                if self.max_endo_lead(id) < 1 {
                    Ok(id)
                } else {
                    Ok(self.create_het_lead_aux(id, table, neweqs))
                }
            }
        }
    }

    /// `matchComplementarityCondition(het_dim)` for `var >= lb`, `lb <= var`, ..., and the
    /// double inequality `lb < var < ub`.
    fn match_complementarity(&mut self, c: Id) -> Result<(u32, Option<Id>, Option<Id>)> {
        let N::Bin(op, a, b) = self.node(c) else { bail!("This expression is not an inequality") };
        let greater = match op {
            BOp::Less | BOp::LessEqual => false,
            BOp::Greater | BOp::GreaterEqual => true,
            _ => bail!("This expression is not an inequality"),
        };
        let het_endo = |t: &Tree, e: Id| match t.node(e) {
            N::Var(s, 0) if t.sym_ty(s) == SymType::Endo => Some(s),
            _ => None,
        };
        let is_const = |t: &mut Tree, e: Id| -> Result<()> {
            let mut vars = BTreeSet::new();
            for ty in [SymType::Endo, SymType::Exo, SymType::AggEndo, SymType::AggExo] {
                t.collect_vars(e, ty, &mut vars, &mut BTreeSet::new());
            }
            if !vars.is_empty() {
                bail!("Bounds must not contain any endogenous or exogenous variable");
            }
            Ok(())
        };
        if let Some(s) = het_endo(self, a) {
            is_const(self, b)?;
            return Ok((s, greater.then_some(b), (!greater).then_some(b)));
        }
        if let Some(s) = het_endo(self, b) {
            is_const(self, a)?;
            return Ok((s, (!greater).then_some(a), greater.then_some(a)));
        }
        if let N::Bin(op1, x, y) = self.node(a) {
            let ok = if greater {
                matches!(op1, BOp::Greater | BOp::GreaterEqual)
            } else {
                matches!(op1, BOp::Less | BOp::LessEqual)
            };
            if ok && let Some(s) = het_endo(self, y) {
                is_const(self, x)?;
                is_const(self, b)?;
                // "lb < var < ub" or "ub > var > lb"
                return Ok(if greater { (s, Some(b), Some(x)) } else { (s, Some(x), Some(b)) });
            }
        }
        bail!("Complementarity condition has an incorrect form")
    }
}

/// `HeterogeneousModel` pipeline up to (not including) the derivatives.
pub(crate) fn het_transform(inp: HetInput<'_>, dag: &mut Dag, agg_endo_offset_hint: usize) -> Result<HetOut> {
    let HetInput { items, syms, names, n_het_endo, n_het_exo, lookup, dim_name } = inp;
    let mut t = Tree::new(syms, names, n_het_endo);
    let mut mcp_raw: Vec<Mcp> = Vec::new();
    for item in items {
        match item {
            ModelItem::Equation { lhs, rhs, nums, tags, line } => {
                let mut it = nums.iter();
                let l = t.replay(lhs, &mut it, lookup)?;
                let r = match rhs {
                    Some(r) => t.replay(r, &mut it, lookup)?,
                    None => t.zero,
                };
                let mut tags = tags.clone();
                let mut cond = None;
                if let Some(k) = tags.iter().position(|(k, _)| k == crate::model::MCP_TAG) {
                    let text = tags.remove(k).1;
                    let ast = parse_expression(&text)?;
                    let c = t.replay(&ast, &mut Vec::<String>::new().iter(), lookup)?;
                    cond = Some(
                        t.match_complementarity(c)
                            .map_err(|e| anyhow::anyhow!("Complementarity condition has an incorrect form: {e}"))?,
                    );
                }
                let id = t.add_equal(l, r);
                t.add_equation(id, tags);
                t.equations.last_mut().unwrap().line = *line;
                mcp_raw.push(cond);
            }
            ModelItem::Local { .. } => {
                bail!("model-local variables are not supported in a model(heterogeneity={dim_name}) block")
            }
            _ => bail!("unsupported statement in a model(heterogeneity={dim_name}) block"),
        }
    }
    let n_user = t.equations.len();
    if n_user == 0 {
        bail!("model(heterogeneity={dim_name}) has no equation");
    }
    // ---- checkPass
    for (i, k) in (0..n_user).enumerate() {
        let e = t.equations[k].eq;
        let mut het_exo = BTreeSet::new();
        t.collect_vars(e, SymType::Exo, &mut het_exo, &mut BTreeSet::new());
        let mut het_endo = BTreeSet::new();
        t.collect_vars(e, SymType::Endo, &mut het_endo, &mut BTreeSet::new());
        for &(s, l) in &het_exo {
            if l < 0 {
                bail!(
                    "In model(heterogeneity={dim_name}), equation {}: lagged heterogeneous exogenous variable '{}' is not supported.",
                    i + 1,
                    t.names[s as usize]
                );
            }
            if l > 0 {
                bail!(
                    "In model(heterogeneity={dim_name}), equation {}: lead on heterogeneous exogenous variable '{}({l:+})' is not supported.",
                    i + 1,
                    t.names[s as usize]
                );
            }
        }
        for &(s, l) in &het_endo {
            if l < -1 {
                bail!(
                    "In model(heterogeneity={dim_name}), equation {}: heterogeneous endogenous variable '{}' with lag {l} is not supported (maximum lag is -1).",
                    i + 1,
                    t.names[s as usize]
                );
            }
            if l > 1 {
                bail!(
                    "In model(heterogeneity={dim_name}), equation {}: heterogeneous endogenous variable '{}' with lead {l} is not supported (maximum lead is +1).",
                    i + 1,
                    t.names[s as usize]
                );
            }
        }
        let mut agg = BTreeSet::new();
        t.collect_vars(e, SymType::AggEndo, &mut agg, &mut BTreeSet::new());
        for &(s, l) in &agg {
            if l.abs() > 1 {
                bail!(
                    "In model(heterogeneity={dim_name}), equation {}: aggregate variable '{}' with lead/lag {l} is not supported by dynare-rust.",
                    i + 1,
                    t.names[s as usize]
                );
            }
        }
        let mut aggx = BTreeSet::new();
        t.collect_vars(e, SymType::AggExo, &mut aggx, &mut BTreeSet::new());
        if let Some(&(s, l)) = aggx.iter().find(|&&(_, l)| l != 0) {
            bail!(
                "In model(heterogeneity={dim_name}), equation {}: aggregate shock '{}' with lead/lag {l} is not supported by dynare-rust.",
                i + 1,
                t.names[s as usize]
            );
        }
        if let Some(sub) = t.find_non_separable(e) {
            let txt = t.json_str(sub);
            bail!(
                "In model(heterogeneity={dim_name}), equation {}: Non-separable expression '{txt}' combines forward-looking variables with lagged states and is not supported.",
                i + 1
            );
        }
    }
    // ---- substituteEndoLead (heterogeneous non-linear leads)
    let mut table = SubstTable::new();
    let mut neweqs = Vec::new();
    for k in 0..t.equations.len() {
        let e = t.equations[k].eq;
        t.equations[k].eq = t.subst_het_lead(e, &mut table, &mut neweqs)?;
    }
    let mut het_aux_eq_idx = Vec::new();
    for ne in neweqs {
        t.add_equation(ne, Vec::new());
        het_aux_eq_idx.push(t.equations.len() - 1);
        mcp_raw.push(None);
    }
    // ---- transformPass: complementarity conditions
    let mut aux_out: Vec<HetAuxOut> = t
        .aux
        .iter()
        .map(|a| HetAuxOut { name: a.name.clone(), ty: a.ty, eq_nbr: 1, endo_index: t.syms[a.symb as usize].index })
        .collect();
    let mut aux_eqs: Vec<Id> = Vec::new(); // `aux_equations` (the equations `aux = rhs`)
    let mut bounds: Vec<Mcp> = mcp_raw.clone();
    let mut i = 0;
    while i < t.equations.len() {
        if let Some((var, lb, ub)) = mcp_raw.get(i).copied().flatten() {
            let vname = t.names[var as usize].clone();
            let vnode = t.add_var(var, 0);
            if let Some(lb) = lb {
                let sym = t.add_aux(format!("MULT_L_{vname}"), 15, None, None, 0);
                aux_out.push(HetAuxOut {
                    name: format!("MULT_L_{vname}"),
                    ty: 15,
                    eq_nbr: i + 1,
                    endo_index: t.syms[sym as usize].index,
                });
                let mu = t.add_var(sym, 0);
                let (a1, a2) = t.eq_args(t.equations[i].eq);
                let orig = t.add_minus(a1, a2);
                let lhs = t.add_minus(a1, mu);
                t.equations[i].eq = t.add_equal(lhs, a2);
                let d = t.add_minus(vnode, lb);
                let m = t.add_times(mu, d);
                let z = t.zero;
                let e = t.add_equal(m, z);
                t.add_equation(e, Vec::new());
                mcp_raw.push(None);
                let ind = t.add_binary_op(vnode, BOp::LessEqual, lb);
                let prod = t.add_times(orig, ind);
                aux_eqs.push(t.add_equal(mu, prod));
            }
            if let Some(ub) = ub {
                let sym = t.add_aux(format!("MULT_U_{vname}"), 15, None, None, 0);
                aux_out.push(HetAuxOut {
                    name: format!("MULT_U_{vname}"),
                    ty: 15,
                    eq_nbr: i + 1,
                    endo_index: t.syms[sym as usize].index,
                });
                let mu = t.add_var(sym, 0);
                let (a1, a2) = t.eq_args(t.equations[i].eq);
                let orig = t.add_minus(a2, a1);
                let lhs = t.add_plus(a1, mu);
                t.equations[i].eq = t.add_equal(lhs, a2);
                let d = t.add_minus(ub, vnode);
                let m = t.add_times(mu, d);
                let z = t.zero;
                let e = t.add_equal(m, z);
                t.add_equation(e, Vec::new());
                mcp_raw.push(None);
                let ind = t.add_binary_op(vnode, BOp::GreaterEqual, ub);
                let prod = t.add_times(orig, ind);
                aux_eqs.push(t.add_equal(mu, prod));
            }
        }
        i += 1;
    }
    for &k in &het_aux_eq_idx {
        aux_eqs.push(t.equations[k].eq);
    }
    // ---- reorderHetAuxiliaryEquations
    let n_aux_eq = aux_eqs.len();
    let aux_lhs_sym = |t: &Tree, e: Id| -> u32 {
        match t.node(t.eq_args(e).0) {
            N::Var(s, 0) => s,
            _ => unreachable!("auxiliary equation without a variable on the left"),
        }
    };
    let mut eq_of: BTreeMap<u32, usize> = BTreeMap::new();
    for (k, &e) in aux_eqs.iter().enumerate() {
        eq_of.insert(aux_lhs_sym(&t, e), k);
    }
    if n_aux_eq > 0 {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_aux_eq];
        for (k, &e) in aux_eqs.iter().enumerate() {
            let mut vars = BTreeSet::new();
            t.collect_vars(e, SymType::Endo, &mut vars, &mut BTreeSet::new());
            let syms: BTreeSet<u32> = vars.iter().map(|&(s, _)| s).collect();
            for s in syms {
                if let Some(&j) = eq_of.get(&s)
                    && j != k
                {
                    adj[k].push(j);
                }
            }
        }
        // boost::topological_sort: DFS finishing order
        let mut state = vec![0u8; n_aux_eq];
        let mut order = Vec::with_capacity(n_aux_eq);
        fn dfs(v: usize, adj: &[Vec<usize>], state: &mut [u8], order: &mut Vec<usize>) {
            state[v] = 1;
            for &w in &adj[v] {
                if state[w] == 0 {
                    dfs(w, adj, state, order);
                }
            }
            state[v] = 2;
            order.push(v);
        }
        for v in 0..n_aux_eq {
            if state[v] == 0 {
                dfs(v, &adj, &mut state, &mut order);
            }
        }
        let old = aux_eqs.clone();
        for k in 0..n_aux_eq {
            aux_eqs[k] = old[order[k]];
        }
    }
    // ---- computeHetAuxTopologicalLevels
    let mut eq_of: BTreeMap<u32, usize> = BTreeMap::new();
    for (k, &e) in aux_eqs.iter().enumerate() {
        eq_of.insert(aux_lhs_sym(&t, e), k);
    }
    let mut deps: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_aux_eq];
    for (k, &e) in aux_eqs.iter().enumerate() {
        let rhs = t.eq_args(e).1;
        let mut vars = BTreeSet::new();
        t.collect_vars(rhs, SymType::Endo, &mut vars, &mut BTreeSet::new());
        for (s, lag) in vars {
            if lag != 1 && lag != -1 {
                continue;
            }
            if let Some(&j) = eq_of.get(&s)
                && j != k
            {
                deps[k].insert(j);
            }
        }
    }
    let mut levels = vec![-1i32; n_aux_eq];
    let mut aux_levels: Vec<Vec<usize>> = Vec::new();
    let (mut cur, mut assigned) = (0i32, 0usize);
    while assigned < n_aux_eq {
        let mut lv = Vec::new();
        for k in 0..n_aux_eq {
            if levels[k] >= 0 {
                continue;
            }
            if deps[k].iter().all(|&d| levels[d] >= 0 && levels[d] < cur) {
                levels[k] = cur;
                lv.push(t.syms[aux_lhs_sym(&t, aux_eqs[k]) as usize].index);
                assigned += 1;
            }
        }
        if lv.is_empty() {
            bail!("Circular dependency detected in heterogeneous auxiliary variables");
        }
        aux_levels.push(lv);
        cur += 1;
    }
    // ---- conversion to the Dag
    let n_het_final = t.n_endo;
    t.agg_offsets = (n_het_final, n_het_exo);
    let _ = agg_endo_offset_hint;
    let mut memo: HashMap<Id, NodeId> = HashMap::new();
    let eqs = t.equations.clone();
    let mut equations = Vec::with_capacity(eqs.len());
    let mut tags = Vec::with_capacity(eqs.len());
    for e in &eqs {
        let (a, b) = t.eq_args(e.eq);
        equations.push((t.convert(a, dag, &mut memo)?, t.convert(b, dag, &mut memo)?));
        tags.push(e.tags.clone());
    }
    let mut aux_equations = Vec::with_capacity(n_aux_eq);
    for &e in &aux_eqs {
        let (a, b) = t.eq_args(e);
        let s = match t.node(a) {
            N::Var(s, _) => s,
            _ => unreachable!(),
        };
        aux_equations.push((t.syms[s as usize].index, t.convert(b, dag, &mut memo)?));
    }
    let mut mcp = Vec::with_capacity(eqs.len());
    for k in 0..eqs.len() {
        mcp.push(match bounds.get_mut(k).copied().flatten() {
            Some((var, lb, ub)) => {
                let lb = lb.map(|x| t.convert(x, dag, &mut memo)).transpose()?;
                let ub = ub.map(|x| t.convert(x, dag, &mut memo)).transpose()?;
                Some((t.syms[var as usize].index, lb, ub))
            }
            None => None,
        });
    }
    // M_.heterogeneity.aux_vars in the order of symbol creation: lead auxiliaries, multipliers
    // (aux_out was filled in that order).
    Ok(HetOut { equations, tags, aux: aux_out, n_het_endo: n_het_final, mcp, aux_equations, aux_levels })
}
