// SPDX-License-Identifier: GPL-3.0-or-later
//! complementarity conditions of mixed complementarity problems (`eq ⟂ x > lb;`,
//! `eq _|_ x < ub;`, the obsolete `[mcp = 'x > lb']` tag): the preprocessor's
//! `<fname>.{static,dynamic}_complementarity_conditions(params)` (bounds per endogenous
//! variable) and `M_.{static,dynamic}_mcp_equations_reordering` (the equation complementary to
//! each variable, the residual order the solver sees).

use crate::model::{Model, SymbolKind};
use anyhow::{Result, anyhow, bail};

/// Bounds and equation order of the MCP formulation.
#[derive(Clone, Debug)]
pub struct Complementarity {
    /// Lower / upper bound of each endogenous variable (`-Inf` / `Inf` when free).
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    /// `eq_index[j]`: the equation whose residual is the j-th component (0-based).
    pub eq_index: Vec<usize>,
}

impl Complementarity {
    pub fn is_empty_model(model: &Model) -> bool {
        model.mcp.is_empty()
    }

    /// Evaluates the conditions at `params`.
    pub fn new(model: &Model, params: &[f64]) -> Result<Complementarity> {
        let n = model.n_endo();
        let neq = model.n_equations();
        let mut lb = vec![f64::NEG_INFINITY; n];
        let mut ub = vec![f64::INFINITY; n];
        let mut eq_index: Vec<usize> = (0..neq).collect();
        for (eq, cond) in &model.mcp {
            let (var, op, rhs) = split_condition(cond)?;
            let j = match model.symbol(var) {
                Some(s) if s.kind == SymbolKind::Endo => s.index,
                _ => bail!("complementarity condition '{cond}': '{var}' is not an endogenous variable"),
            };
            let e = model.compile_expr(rhs).map_err(|e| anyhow!("complementarity condition '{cond}': {e}"))?;
            let v = e.eval(&vec![0.0; n], &vec![0.0; model.n_exo()], params);
            if op.starts_with('>') {
                lb[j] = v;
            } else {
                ub[j] = v;
            }
            // the equation goes to the variable's position (DynamicModel's reordering)
            let pos = eq_index.iter().position(|&k| k == *eq).unwrap();
            eq_index.swap(pos, j);
        }
        Ok(Complementarity { lb, ub, eq_index })
    }

    /// Residuals in MCP order (`res(eq_index)`).
    pub fn reorder(&self, res: &[f64]) -> Vec<f64> {
        self.eq_index.iter().map(|&k| res[k]).collect()
    }
}

/// `x > 0` -> ("x", ">", "0").
fn split_condition(c: &str) -> Result<(&str, &str, &str)> {
    for op in [">=", "<=", ">", "<"] {
        if let Some(k) = c.find(op) {
            return Ok((c[..k].trim(), op, c[k + op.len()..].trim()));
        }
    }
    bail!("complementarity condition '{c}': expected 'variable > bound' or 'variable < bound'")
}
