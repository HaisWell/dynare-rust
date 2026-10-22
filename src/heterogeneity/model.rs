// SPDX-License-Identifier: GPL-3.0-or-later
// ---- heterogeneity ----
//! The heterogeneous model of a dimension (`M_.heterogeneity(1)`): names, auxiliary variables,
//! and compiled evaluators of what the preprocessor writes to `dynamic_het1_resid.m`,
//! `dynamic_het1_g1.m`, `dynamic_het1_set_auxiliary_variables.m` and
//! `dynamic_het1_complementarity_conditions.m`.
//!
//! Layout of the evaluation arrays, as in the generated MATLAB code: `yh` has `3 * endo_nbr`
//! entries (`[yh_{t-1}; yh_t; yh_{t+1}]`), `xh` has `exo_nbr`, the aggregate `y` has
//! `3 * n_agg_endo` and `x` has `n_agg_exo`. The Jacobian columns are
//! `[yh (3 n_het); xh; y (3 n_agg); x]`.

use crate::linalg::Mat;
use crate::model::dtree::{HetAuxOut, HetInput, ModelItem, Sym, SymType, het_transform};
use crate::model::expr::{Dag, DiffVar, Node, NodeId};
use crate::model::tape::Tape;
use crate::model::{Model, SymbolKind};
use anyhow::{Result, bail};
use std::collections::HashMap;

/// A heterogeneous symbol as declared (`var(heterogeneity=...)` and friends).
#[derive(Clone, Debug, PartialEq)]
pub struct HetSymbolDecl {
    pub name: String,
    pub tex: String,
    pub long: String,
}

/// What the parser hands over to [`HetModel::build`].
#[derive(Clone, Debug)]
pub struct HetModelDecl {
    pub dimension_name: String,
    pub endo: Vec<HetSymbolDecl>,
    pub exo: Vec<HetSymbolDecl>,
    pub params: Vec<HetSymbolDecl>,
    /// `M_.heterogeneity(1).Sigma_e`.
    pub sigma_e: Mat,
}

/// `M_.heterogeneity(1).aux_vars(k)`.
#[derive(Clone, Debug, PartialEq)]
pub struct HetAuxVar {
    /// 0-based position among the heterogeneous endogenous variables.
    pub endo_index: usize,
    /// 15 multiplier of a bound, 16 non-linear lead.
    pub ty: u8,
    /// 1-based equation number (the equation of the multiplier).
    pub eq_nbr: usize,
}

/// `M_.heterogeneity(1)`.
#[derive(Clone, Debug)]
pub struct HetModel {
    pub dimension_name: String,
    /// Declared heterogeneous endogenous variables, then the auxiliary ones.
    pub endo_names: Vec<String>,
    pub endo_names_tex: Vec<String>,
    pub endo_names_long: Vec<String>,
    pub orig_endo_nbr: usize,
    pub exo_names: Vec<String>,
    pub exo_names_tex: Vec<String>,
    pub exo_names_long: Vec<String>,
    pub param_names: Vec<String>,
    pub param_names_tex: Vec<String>,
    pub param_names_long: Vec<String>,
    pub aux_vars: Vec<HetAuxVar>,
    /// 0-based indices of the (original) heterogeneous variables that appear lagged
    /// (`state_var`): the states of the individual problem.
    pub state_var: Vec<usize>,
    /// Tags of each equation (written equations first, then the auxiliary ones).
    pub equations_tags: Vec<Vec<(String, String)>>,
    /// Number of written equations (`orig_endo_nbr`).
    pub sigma_e: Mat,
    pub n_agg_endo: usize,
    pub n_agg_exo: usize,
    /// `dynamic_mcp_equations_reordering` (0-based): equation solving for variable `i`.
    pub mcp_equations_reordering: Vec<usize>,
    pub set_auxiliary_variables: bool,
    /// `het_aux_levels`: 0-based heterogeneous endogenous indices per level.
    pub het_aux_levels: Vec<Vec<usize>>,
    /// MATLAB-style 1-based sparse structure of `dynamic_het1_g1`
    /// (`dynamic_g1_sparse_rowval/colval/colptr`).
    pub dynamic_g1_sparse_rowval: Vec<i32>,
    pub dynamic_g1_sparse_colval: Vec<i32>,
    pub dynamic_g1_sparse_colptr: Vec<i32>,
    dag: Dag,
    bounds: Vec<(Option<NodeId>, Option<NodeId>)>,
    resid_tape: Tape,
    jac_tape: Tape,
    /// Structural non-zeros (row, column), column-major, 0-based.
    jac_pattern: Vec<(usize, usize)>,
    /// One tape per auxiliary equation, in write order, with the target index (0-based) and level.
    aux_tapes: Vec<(usize, Tape)>,
    aux_level_of: Vec<usize>,
    scratch_len: usize,
}

/// Evaluation buffers of a [`HetModel`]: fill `y`/`x` (flat `[yh; y]` / `[xh; x]`) then call the
/// evaluators.
#[derive(Clone, Debug)]
pub struct HetWorkspace {
    n_het: usize,
    n_het_exo: usize,
    n_agg: usize,
    /// `[yh (3 n_het); y (3 n_agg)]`.
    pub yf: Vec<f64>,
    /// `[xh; x]`.
    pub xf: Vec<f64>,
    /// `M_.params`.
    pub params: Vec<f64>,
    scratch: Vec<f64>,
}

impl HetWorkspace {
    pub fn yh(&self) -> &[f64] {
        &self.yf[..3 * self.n_het]
    }
    pub fn yh_mut(&mut self) -> &mut [f64] {
        &mut self.yf[..3 * self.n_het]
    }
    pub fn y_agg(&self) -> &[f64] {
        &self.yf[3 * self.n_het..]
    }
    pub fn y_agg_mut(&mut self) -> &mut [f64] {
        &mut self.yf[3 * self.n_het..]
    }
    pub fn xh_mut(&mut self) -> &mut [f64] {
        &mut self.xf[..self.n_het_exo]
    }
    pub fn x_agg_mut(&mut self) -> &mut [f64] {
        &mut self.xf[self.n_het_exo..]
    }
    pub fn n_agg(&self) -> usize {
        self.n_agg
    }
}

impl HetModel {
    /// Transforms the heterogeneous model block (`items`, as parsed) into the model of the
    /// dimension; `model` is the (built) aggregate model.
    pub(crate) fn build(decl: HetModelDecl, items: &[ModelItem], model: &Model) -> Result<HetModel> {
        let n0 = decl.endo.len();
        let nx = decl.exo.len();
        let n_agg_orig = model.orig_endo_nbr;
        let n_agg = model.endo_names.len();
        let n_agg_exo = model.exo_names.len();
        // symbol ids of the heterogeneous tree: het endo, het exo, aggregate endo, aggregate
        // exo, parameters
        let mut syms = Vec::new();
        let mut names = Vec::new();
        let mut ids: HashMap<String, u32> = HashMap::new();
        let mut add = |name: &str, ty: SymType, index: usize, syms: &mut Vec<Sym>, names: &mut Vec<String>| {
            ids.insert(name.to_string(), syms.len() as u32);
            syms.push(Sym { ty, index });
            names.push(name.to_string());
        };
        for (i, d) in decl.endo.iter().enumerate() {
            add(&d.name, SymType::Endo, i, &mut syms, &mut names);
        }
        for (i, d) in decl.exo.iter().enumerate() {
            add(&d.name, SymType::Exo, i, &mut syms, &mut names);
        }
        for (i, nm) in model.endo_names.iter().take(n_agg_orig + 0).enumerate() {
            if model.symbol(nm).is_some_and(|s| s.kind == SymbolKind::Endo) {
                add(nm, SymType::AggEndo, i, &mut syms, &mut names);
            }
        }
        for (i, nm) in model.exo_names.iter().enumerate() {
            add(nm, SymType::AggExo, i, &mut syms, &mut names);
        }
        for (i, nm) in model.param_names.iter().enumerate() {
            add(nm, SymType::Param, i, &mut syms, &mut names);
        }
        let lookup = |n: &str| ids.get(n).copied();
        let mut dag = Dag::new();
        let out = het_transform(
            HetInput {
                items,
                syms,
                names,
                n_het_endo: n0,
                n_het_exo: nx,
                lookup: &lookup,
                dim_name: &decl.dimension_name,
            },
            &mut dag,
            0,
        )?;
        let nh = out.n_het_endo;
        if out.equations.len() != nh {
            bail!(
                "There are {} equations but {} endogenous variables in the model for heterogeneity dimension '{}'!",
                out.equations.len(),
                nh,
                decl.dimension_name
            );
        }
        let neq = nh;
        // names
        let mut endo_names: Vec<String> = decl.endo.iter().map(|d| d.name.clone()).collect();
        let mut endo_tex: Vec<String> = decl.endo.iter().map(|d| d.tex.clone()).collect();
        let mut endo_long: Vec<String> = decl.endo.iter().map(|d| d.long.clone()).collect();
        let mut aux_vars = Vec::new();
        for a in &out.aux {
            endo_names.push(a.name.clone());
            endo_tex.push(a.name.replace('_', "\\_"));
            endo_long.push(a.name.clone());
            aux_vars.push(HetAuxVar { endo_index: a.endo_index, ty: a.ty, eq_nbr: a.eq_nbr });
        }
        let _: &[HetAuxOut] = &out.aux;
        // residuals and Jacobian
        let resid: Vec<NodeId> = out.equations.iter().map(|&(l, r)| dag.sub(l, r)).collect();
        let nhx = nx;
        let ycol = |j: u32, l: i32| -> usize {
            let j = j as usize;
            if j < nh { (l + 1) as usize * nh + j } else { 3 * nh + nhx + (l + 1) as usize * n_agg + (j - nh) }
        };
        let mut present_lag = vec![false; n0];
        let mut diff_memo: HashMap<DiffVar, HashMap<NodeId, NodeId>> = HashMap::new();
        let mut entries: Vec<(usize, usize, NodeId)> = Vec::new();
        let mut leaves = Vec::new();
        for (i, &r) in resid.iter().enumerate() {
            leaves.clear();
            dag.collect_leaves(r, &mut leaves);
            for &leaf in &leaves {
                let (v, col) = match leaf {
                    Node::Endo(j, l) => {
                        if !(-1..=1).contains(&l) {
                            bail!("internal error: lead/lag {l} left in the heterogeneous model");
                        }
                        if (j as usize) < n0 && l == -1 {
                            present_lag[j as usize] = true;
                        }
                        (DiffVar::Endo(j, l), ycol(j, l))
                    }
                    Node::Exo(k, _) => {
                        let k = k as usize;
                        let col = if k < nhx { 3 * nh + k } else { 3 * nh + nhx + 3 * n_agg + (k - nhx) };
                        (DiffVar::Exo(k as u32, 0), col)
                    }
                    _ => continue,
                };
                let d = dag.diff(r, v, diff_memo.entry(v).or_default());
                if d != Dag::ZERO {
                    entries.push((i, col, d));
                }
            }
        }
        entries.sort_by_key(|&(i, c, _)| (c, i));
        let ncols = 3 * nh + nhx + 3 * n_agg + n_agg_exo;
        let mut colptr = vec![1i32; ncols + 1];
        for &(_, c, _) in &entries {
            for p in colptr.iter_mut().skip(c + 1) {
                *p += 1;
            }
        }
        let state_var: Vec<usize> = (0..n0).filter(|&j| present_lag[j]).collect();
        // flat y/x -> tape slots
        let yidx = move |j: u32, l: i32| -> usize {
            let j = j as usize;
            if j < nh { (l + 1) as usize * nh + j } else { 3 * nh + (l + 1) as usize * n_agg + (j - nh) }
        };
        let xidx = |k: u32, _l: i32| k as usize;
        let resid_tape = Tape::compile(&dag, &resid, &yidx, &xidx);
        let jac_nodes: Vec<NodeId> = entries.iter().map(|e| e.2).collect();
        let jac_tape = Tape::compile(&dag, &jac_nodes, &yidx, &xidx);
        let aux_tapes: Vec<(usize, Tape)> =
            out.aux_equations.iter().map(|&(idx, rhs)| (idx, Tape::compile(&dag, &[rhs], &yidx, &xidx))).collect();
        let mut aux_level_of = vec![0; aux_tapes.len()];
        for (lv, vars) in out.aux_levels.iter().enumerate() {
            for v in vars {
                if let Some(k) = aux_tapes.iter().position(|(i, _)| i == v) {
                    aux_level_of[k] = lv;
                }
            }
        }
        // MCP reordering and bounds
        let mut reorder: Vec<usize> = (0..neq).collect();
        let mut bounds = vec![(None, None); nh];
        let mut seen = vec![false; nh];
        for (eq, c) in out.mcp.iter().enumerate() {
            if let Some((var, lb, ub)) = c {
                if std::mem::replace(&mut seen[*var], true) {
                    bail!("variable {} appears in two complementarity conditions", endo_names[*var]);
                }
                let pos = reorder.iter().position(|&e| e == eq).unwrap();
                reorder.swap(*var, pos);
                bounds[*var] = (*lb, *ub);
            }
        }
        let scratch_len = [resid_tape.scratch_len(), jac_tape.scratch_len()]
            .into_iter()
            .chain(aux_tapes.iter().map(|a| a.1.scratch_len()))
            .max()
            .unwrap_or(0);
        let mut tags = out.tags.clone();
        tags.truncate(neq);
        Ok(HetModel {
            dimension_name: decl.dimension_name,
            endo_names,
            endo_names_tex: endo_tex,
            endo_names_long: endo_long,
            orig_endo_nbr: n0,
            exo_names: decl.exo.iter().map(|d| d.name.clone()).collect(),
            exo_names_tex: decl.exo.iter().map(|d| d.tex.clone()).collect(),
            exo_names_long: decl.exo.iter().map(|d| d.long.clone()).collect(),
            param_names: decl.params.iter().map(|d| d.name.clone()).collect(),
            param_names_tex: decl.params.iter().map(|d| d.tex.clone()).collect(),
            param_names_long: decl.params.iter().map(|d| d.long.clone()).collect(),
            aux_vars,
            state_var,
            equations_tags: tags,
            sigma_e: decl.sigma_e,
            n_agg_endo: n_agg,
            n_agg_exo,
            mcp_equations_reordering: reorder,
            set_auxiliary_variables: !aux_tapes.is_empty(),
            het_aux_levels: out.aux_levels.clone(),
            dynamic_g1_sparse_rowval: entries.iter().map(|e| e.0 as i32 + 1).collect(),
            dynamic_g1_sparse_colval: entries.iter().map(|e| e.1 as i32 + 1).collect(),
            dynamic_g1_sparse_colptr: colptr,
            dag,
            bounds,
            resid_tape,
            jac_tape,
            jac_pattern: entries.iter().map(|e| (e.0, e.1)).collect(),
            aux_tapes,
            aux_level_of,
            scratch_len,
        })
    }

    pub fn endo_nbr(&self) -> usize {
        self.endo_names.len()
    }

    pub fn exo_nbr(&self) -> usize {
        self.exo_names.len()
    }

    /// Number of columns of `dynamic_het1_g1`: `3 (n_het + n_agg) + n_het_exo + n_agg_exo`.
    pub fn jacobian_ncols(&self) -> usize {
        3 * (self.endo_nbr() + self.n_agg_endo) + self.exo_nbr() + self.n_agg_exo
    }

    /// Structural non-zeros `(row, column)` of `dynamic_het1_g1`, 0-based, column-major.
    pub fn jacobian_pattern(&self) -> &[(usize, usize)] {
        &self.jac_pattern
    }

    /// A workspace sized for this model and `n_params` parameters (`M_.params`).
    pub fn workspace(&self, params: &[f64]) -> HetWorkspace {
        HetWorkspace {
            n_het: self.endo_nbr(),
            n_het_exo: self.exo_nbr(),
            n_agg: self.n_agg_endo,
            yf: vec![0.0; 3 * (self.endo_nbr() + self.n_agg_endo)],
            xf: vec![0.0; self.exo_nbr() + self.n_agg_exo],
            params: params.to_vec(),
            scratch: vec![0.0; self.scratch_len],
        }
    }

    /// `dynamic_het1_resid(y, x, params, ss, yh, xh, [])` for the arrays held by `ws`.
    pub fn resid(&self, ws: &mut HetWorkspace, out: &mut [f64]) {
        self.resid_tape.eval_into(&ws.yf, &ws.xf, &ws.params, &mut ws.scratch, out);
    }

    /// Values of the non-zeros of `dynamic_het1_g1` (order of [`HetModel::jacobian_pattern`]).
    pub fn jacobian_values(&self, ws: &mut HetWorkspace, out: &mut [f64]) {
        self.jac_tape.eval_into(&ws.yf, &ws.xf, &ws.params, &mut ws.scratch, out);
    }

    /// Dense `dynamic_het1_g1` (`endo_nbr x jacobian_ncols`).
    pub fn jacobian_dense(&self, ws: &mut HetWorkspace) -> Mat {
        let mut v = vec![0.0; self.jac_pattern.len()];
        self.jacobian_values(ws, &mut v);
        let mut m = Mat::zeros(self.endo_nbr(), self.jacobian_ncols());
        for (k, &(i, j)) in self.jac_pattern.iter().enumerate() {
            m[(i, j)] = v[k];
        }
        m
    }

    /// `dynamic_het1_set_auxiliary_variables(..., step)`: sets the auxiliary variables of
    /// level `step` in the current block of `yh`.
    pub fn set_auxiliary_variables(&self, ws: &mut HetWorkspace, step: usize) {
        let nh = self.endo_nbr();
        for (k, (idx, tape)) in self.aux_tapes.iter().enumerate() {
            if self.aux_level_of[k] != step {
                continue;
            }
            tape.eval(&ws.yf, &ws.xf, &ws.params, &mut ws.scratch);
            ws.yf[nh + idx] = tape.output(0, &ws.scratch);
        }
    }

    /// `[lower_bounds, upper_bounds] = dynamic_het1_complementarity_conditions(params)`
    /// (`-Inf` / `+Inf` where a variable has no bound).
    pub fn complementarity_bounds(&self, params: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let leaf = |n: Node| match n {
            Node::Param(i) => params[i as usize],
            _ => f64::NAN,
        };
        let lo = self.bounds.iter().map(|b| b.0.map_or(f64::NEG_INFINITY, |id| self.dag.eval(id, &leaf))).collect();
        let hi = self.bounds.iter().map(|b| b.1.map_or(f64::INFINITY, |id| self.dag.eval(id, &leaf))).collect();
        (lo, hi)
    }

    /// Equation number (1-based) whose `name` tag is `name`, among the heterogeneous equations.
    pub fn equation_by_name(&self, name: &str) -> Option<usize> {
        self.equations_tags.iter().position(|t| t.iter().any(|(k, v)| k == "name" && v == name)).map(|i| i + 1)
    }
}
