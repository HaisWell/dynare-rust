// SPDX-License-Identifier: GPL-3.0-or-later
//! The stacked perfect-foresight problem (Dynare's `perfect_foresight_problem` MEX and
//! `linear_perfect_foresight_problem.m`) and its sparse Jacobian.
//!
//! Unknowns are `y = [y_1; ...; y_T]` (period-major, `n` endogenous per period, as in
//! `sim1.m`); the residuals of period `t` are the dynamic equations at
//! `[y_{t-1}; y_t; y_{t+1}]` and the exogenous row `t + maximum_lag` of `exo_simul`, with
//! `y_0` and `y_{T+1}` the initial and terminal conditions.
//!
//! The Jacobian (block tridiagonal: lag / current / lead blocks of the dynamic Jacobian per
//! period, the lag block of period 1 and the lead block of period T dropped) is assembled in
//! compressed-column form and factorized by a sparse LU with partial pivoting
//! ([`super::sparse_lu`]), where MATLAB's `A\b` calls UMFPACK.

use super::sparse_lu::{Csc, LuWork, NONE, SparseLu};
use crate::model::{Model, Workspace};

/// Structure of the dynamic Jacobian: for each endogenous variable, its pattern entries in
/// the lag / current / lead columns, used to assemble the stacked matrix column by column.
pub struct StackedStructure {
    pub n: usize,
    npat: usize,
    /// For variable j: (row i, pattern index k) of its lag, current and lead entries, by row.
    by_var: Vec<[Vec<(usize, usize)>; 3]>,
    /// Column order of the variables within each period (minimum degree on the period graph).
    pub order: Vec<usize>,
    /// Equation matched to each variable in the current-period block (preferred pivot row),
    /// `NONE` if the current block has no perfect matching.
    pub match_row: Vec<usize>,
}

/// Maximum bipartite matching variables -> equations on the current-period entries
/// (augmenting paths); `None` if not perfect.
fn current_matching(n: usize, by_var: &[[Vec<(usize, usize)>; 3]]) -> Option<Vec<usize>> {
    let mut eq_of_var = vec![NONE; n];
    let mut var_of_eq = vec![NONE; n];
    fn augment(
        j: usize,
        by_var: &[[Vec<(usize, usize)>; 3]],
        seen: &mut [bool],
        eq_of_var: &mut [usize],
        var_of_eq: &mut [usize],
    ) -> bool {
        for &(i, _) in &by_var[j][1] {
            if seen[i] {
                continue;
            }
            seen[i] = true;
            if var_of_eq[i] == NONE || augment(var_of_eq[i], by_var, seen, eq_of_var, var_of_eq) {
                var_of_eq[i] = j;
                eq_of_var[j] = i;
                return true;
            }
        }
        false
    }
    for j in 0..n {
        let mut seen = vec![false; n];
        if !augment(j, by_var, &mut seen, &mut eq_of_var, &mut var_of_eq) {
            return None;
        }
    }
    Some(eq_of_var)
}

/// Greedy minimum-degree ordering of a symmetric graph (explicit elimination graph; the
/// per-period graphs are small).
fn minimum_degree(adj: Vec<std::collections::BTreeSet<usize>>) -> Vec<usize> {
    let n = adj.len();
    let mut adj = adj;
    let mut done = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for _ in 0..n {
        let v = (0..n).filter(|&v| !done[v]).min_by_key(|&v| (adj[v].len(), v)).unwrap();
        done[v] = true;
        order.push(v);
        let nb: Vec<usize> = adj[v].iter().copied().collect();
        for &a in &nb {
            adj[a].remove(&v);
            for &b in &nb {
                if a != b {
                    adj[a].insert(b);
                }
            }
        }
        adj[v].clear();
    }
    order
}

impl StackedStructure {
    pub fn new(model: &Model) -> Self {
        let n = model.n_endo();
        let pattern = model.dynamic_jacobian_pattern();
        let mut by_var: Vec<[Vec<(usize, usize)>; 3]> = (0..n).map(|_| [vec![], vec![], vec![]]).collect();
        for (k, &(i, c)) in pattern.iter().enumerate() {
            if c < 3 * n {
                by_var[c % n][c / n].push((i, k));
            }
        }
        for v in &mut by_var {
            for b in v.iter_mut() {
                b.sort();
            }
        }
        let (order, match_row) = match current_matching(n, &by_var) {
            Some(m) => {
                let mut var_of_eq = vec![NONE; n];
                for (j, &i) in m.iter().enumerate() {
                    var_of_eq[i] = j;
                }
                let mut adj = vec![std::collections::BTreeSet::new(); n];
                for (j, v) in by_var.iter().enumerate() {
                    for blk in v {
                        for &(i, _) in blk {
                            let r = var_of_eq[i];
                            if r != j {
                                adj[r].insert(j);
                                adj[j].insert(r);
                            }
                        }
                    }
                }
                (minimum_degree(adj), m)
            }
            None => ((0..n).collect(), vec![NONE; n]),
        };
        StackedStructure { n, npat: pattern.len(), by_var, order, match_row }
    }
}

/// Threshold of the preferred (matched) pivot relative to the largest candidate, as KLU's
/// default `tol = 0.001`.
const PIVOT_TOL: f64 = 0.001;

/// The stacked Jacobian over `t` periods: dynamic Jacobian values per period.
pub struct StackedJacobian {
    pub n: usize,
    pub t: usize,
    npat: usize,
    vals: Vec<f64>,
    csc: Csc,
    /// Preferred pivot row of each (permuted) column.
    prefer: Vec<usize>,
    work: LuWork,
}

impl StackedJacobian {
    pub fn new(s: &StackedStructure, t: usize) -> Self {
        let n = s.n;
        // pattern of the stacked matrix (fixed): column (period q, var j) holds the lead
        // entries of period q-1, the current entries of q and the lag entries of q+1
        let mut csc = Csc { n: n * t, colptr: Vec::with_capacity(n * t + 1), ..Default::default() };
        csc.colptr.push(0);
        for q in 0..t {
            for &j in &s.order {
                let v = &s.by_var[j];
                if q > 0 {
                    csc.rowind.extend(v[2].iter().map(|&(i, _)| (q - 1) * n + i));
                }
                csc.rowind.extend(v[1].iter().map(|&(i, _)| q * n + i));
                if q + 1 < t {
                    csc.rowind.extend(v[0].iter().map(|&(i, _)| (q + 1) * n + i));
                }
                csc.colptr.push(csc.rowind.len());
            }
        }
        csc.vals = vec![0.0; csc.rowind.len()];
        let mut prefer = vec![NONE; n * t];
        for q in 0..t {
            for (pos, &j) in s.order.iter().enumerate() {
                if s.match_row[j] != NONE {
                    prefer[q * n + pos] = q * n + s.match_row[j];
                }
            }
        }
        StackedJacobian { n, t, npat: s.npat, vals: vec![0.0; s.npat * t], csc, prefer, work: LuWork::default() }
    }

    /// Stores period `t`'s dynamic Jacobian values (pattern order).
    pub fn set_period(&mut self, t: usize, vals: &[f64]) {
        self.vals[t * self.npat..(t + 1) * self.npat].copy_from_slice(vals);
    }

    fn assemble(&mut self, s: &StackedStructure) {
        let (t, np) = (self.t, self.npat);
        let mut p = 0;
        for q in 0..t {
            for &j in &s.order {
                let v = &s.by_var[j];
                if q > 0 {
                    for &(_, k) in &v[2] {
                        self.csc.vals[p] = self.vals[(q - 1) * np + k];
                        p += 1;
                    }
                }
                for &(_, k) in &v[1] {
                    self.csc.vals[p] = self.vals[q * np + k];
                    p += 1;
                }
                if q + 1 < t {
                    for &(_, k) in &v[0] {
                        self.csc.vals[p] = self.vals[(q + 1) * np + k];
                        p += 1;
                    }
                }
            }
        }
    }

    /// `A x = b` in place on `b` (length n*T). Returns false if the matrix is singular.
    pub fn solve(&mut self, s: &StackedStructure, b: &mut [f64]) -> bool {
        self.assemble(s);
        match SparseLu::factor(&self.csc, Some(&self.prefer), PIVOT_TOL, &mut self.work) {
            Some(f) => {
                f.solve(b);
                // b holds the solution in column order: unknown (q, order[pos]) = b[q*n + pos]
                let n = self.n;
                let xs = b.to_vec();
                for q in 0..self.t {
                    for (pos, &j) in s.order.iter().enumerate() {
                        b[q * n + j] = xs[q * n + pos];
                    }
                }
                true
            }
            None => false,
        }
    }

    /// `A v` (diagnostics and tests; `v` in period-major variable order).
    pub fn mul_vec(&mut self, s: &StackedStructure, v: &[f64]) -> Vec<f64> {
        self.assemble(s);
        let n = self.n;
        let mut vp = vec![0.0; v.len()];
        for q in 0..self.t {
            for (pos, &j) in s.order.iter().enumerate() {
                vp[q * n + pos] = v[q * n + j];
            }
        }
        self.csc.mul_vec(&vp)
    }
}

/// Evaluates the stacked problem: residuals (and, with `jac`, the Jacobian).
pub struct StackedProblem<'a> {
    pub model: &'a Model,
    pub s: StackedStructure,
    ws: Workspace,
    y3: Vec<f64>,
    vals: Vec<f64>,
    out: Vec<f64>,
}

impl<'a> StackedProblem<'a> {
    pub fn new(model: &'a Model) -> Self {
        let s = StackedStructure::new(model);
        StackedProblem {
            model,
            ws: model.workspace(),
            y3: vec![0.0; 3 * s.n],
            vals: vec![0.0; s.npat],
            out: vec![0.0; s.n],
            s,
        }
    }

    /// Residuals `res` (length n*T) at `y` (length n*T) with conditions `y0`, `yt`;
    /// `exo(t)` gives the exogenous row of simulation period t (0-based). Fills `jac` if given.
    #[allow(clippy::too_many_arguments)]
    pub fn eval(
        &mut self,
        y: &[f64],
        y0: &[f64],
        yt: &[f64],
        exo: &dyn Fn(usize) -> Vec<f64>,
        params: &[f64],
        res: &mut [f64],
        mut jac: Option<&mut StackedJacobian>,
    ) {
        let n = self.s.n;
        let tt = y.len() / n;
        for t in 0..tt {
            let prev = if t == 0 { y0 } else { &y[(t - 1) * n..t * n] };
            let next = if t + 1 == tt { yt } else { &y[(t + 1) * n..(t + 2) * n] };
            self.y3[..n].copy_from_slice(prev);
            self.y3[n..2 * n].copy_from_slice(&y[t * n..(t + 1) * n]);
            self.y3[2 * n..].copy_from_slice(next);
            let x = exo(t);
            self.model.dynamic_residuals(&self.y3, &x, params, &mut self.out);
            res[t * n..(t + 1) * n].copy_from_slice(&self.out);
            if let Some(j) = jac.as_deref_mut() {
                self.model.dynamic_jacobian_values_with(&self.y3, &x, params, &mut self.ws, &mut self.vals);
                j.set_period(t, &self.vals);
            }
        }
    }
}
