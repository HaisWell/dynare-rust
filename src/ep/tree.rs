// SPDX-License-Identifier: GPL-3.0-or-later
//! The stacked problems solved at every period of the (stochastic) extended path: port of
//! `ep/ep_problem_0.m` (full tree of future histories, `ep.stochastic.algo = 0`, with the
//! index bookkeeping of `solve_stochastic_perfect_foresight_model_0.m`) and
//! `ep/ep_problem_1.m` (sparse "fishbone" tree, `algo = 1`,
//! `solve_stochastic_perfect_foresight_model_1.m`). With `order = 0` the full tree is a single
//! path and the problem is `perfect_foresight_problem` (the deterministic extended path).
//!
//! The unknowns are blocks of `ny` endogenous variables, one per (period, node), ordered by
//! period then node as in Dynare (`i_upd_y`), and the residual block `b` holds the dynamic
//! equations of block `b` (`i_upd_r`). The residual of a block is a weighted sum of
//! *terms*, the dynamic equations at `[y_prev; y_b; y_next]` and one exogenous row: a node
//! of period `i <= order` integrates over the next-period nodes with the quadrature weights.
//! Terms are evaluated in parallel by contiguous ranges of blocks, so the results do not
//! depend on the number of threads.

use anyhow::{Result, bail};
use rayon::prelude::*;

use super::sparse::SparseJac;
use super::trust_region::System;
use crate::linalg::Mat;
use crate::model::Model;

/// A neighbour of a block: another block, the initial condition (period 0) or the terminal
/// condition (period `periods + 1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ref {
    Block(usize),
    Init,
    Terminal,
}

/// One term of a block's residual.
#[derive(Clone, Copy, Debug)]
pub struct Term {
    pub prev: Ref,
    pub next: Ref,
    /// Quadrature weight (1 for single-term blocks).
    pub weight: f64,
    /// Integration node replacing the shocks with positive variance (`None`: row `period` of
    /// the exogenous matrix, i.e. the realized shock in period 1 and the steady state later).
    pub node: Option<usize>,
    /// Adds the hybrid correction (`pfm.h_correction`) to the next-period values.
    pub hybrid: bool,
}

#[derive(Clone, Debug)]
pub struct Block {
    /// 1-based period.
    pub period: usize,
    pub terms: Vec<Term>,
}

/// Shape of the tree of future histories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeKind {
    /// `ep.stochastic.algo = 0` (`tree=perfect`).
    Full,
    /// `ep.stochastic.algo = 1` (`tree=sparse`).
    Sparse,
}

/// Port of `get_block_world_nbr.m`'s `world_nbr` (number of paths after period `order`).
pub fn world_nbr(kind: TreeKind, nnodes: usize, order: usize) -> usize {
    match kind {
        TreeKind::Full => nnodes.pow(order as u32),
        TreeKind::Sparse => 1 + (nnodes - 1) * order,
    }
}

/// The blocks of the tree, in Dynare's unknown order.
pub fn build_tree(
    kind: TreeKind,
    order: usize,
    nnodes: usize,
    periods: usize,
    weights: &[f64],
    hybrid: bool,
) -> Result<Vec<Block>> {
    if periods < order + 1 || periods == 0 {
        bail!("extended_path: the number of solver periods ({periods}) must be at least order + 1 = {}", order + 1);
    }
    let wn = world_nbr(kind, nnodes, order);
    let cnt = |i: usize| -> usize {
        let m = (i - 1).min(order);
        match kind {
            TreeKind::Full => nnodes.pow(m as u32),
            TreeKind::Sparse => 1 + (nnodes - 1) * m,
        }
    };
    let mut base = vec![0usize; periods + 2];
    for i in 1..=periods {
        base[i + 1] = base[i] + cnt(i);
    }
    let total = base[periods + 1];
    let blk = |i: usize, j: usize| -> Ref {
        if i > periods {
            Ref::Terminal
        } else {
            debug_assert!(j < cnt(i));
            Ref::Block(base[i] + j)
        }
    };
    let mut blocks: Vec<Option<Block>> = vec![None; total];
    let single = |prev: Ref, next: Ref, node: Option<usize>| Term { prev, next, weight: 1.0, node, hybrid: false };
    for i in 1..=order + 1 {
        let nj = match kind {
            TreeKind::Full => nnodes.pow((i - 1) as u32),
            TreeKind::Sparse => 1 + (nnodes - 1) * (i - 1),
        };
        for j in 0..nj {
            let terms = match kind {
                TreeKind::Full => {
                    let prev = if i == 1 { Ref::Init } else { blk(i - 1, j / nnodes) };
                    let node = if i > 1 { Some(j % nnodes) } else { None };
                    if i <= order {
                        (0..nnodes)
                            .map(|k| Term {
                                prev,
                                next: blk(i + 1, j * nnodes + k),
                                weight: weights[k],
                                node,
                                hybrid: hybrid && i == order,
                            })
                            .collect()
                    } else {
                        vec![single(prev, blk(i + 1, j), node)]
                    }
                }
                TreeKind::Sparse => {
                    let j1 = j + 1; // 1-based as in ep_problem_1.m
                    if i <= order && j1 == 1 {
                        let prev = if i == 1 { Ref::Init } else { blk(i - 1, 0) };
                        (1..=nnodes)
                            .map(|k| {
                                let k1 = if k == 1 { 1 } else { (nnodes - 1) * (i - 1) + k };
                                Term {
                                    prev,
                                    next: blk(i + 1, k1 - 1),
                                    weight: weights[k - 1],
                                    node: if i > 1 { Some(k - 1) } else { None },
                                    hybrid: hybrid && (k > 1 || i == order),
                                }
                            })
                            .collect()
                    } else if i >= 2 && j1 > 1 + (nnodes - 1) * (i - 2) {
                        let k = j1 - (nnodes - 1) * (i - 2);
                        vec![single(blk(i - 1, 0), blk(i + 1, j), Some(k - 1))]
                    } else {
                        vec![single(blk(i - 1, j), blk(i + 1, j), None)]
                    }
                }
            };
            let Ref::Block(b) = blk(i, j) else { unreachable!() };
            blocks[b] = Some(Block { period: i, terms });
        }
    }
    for j in 0..wn {
        for i in order + 2..=periods {
            let Ref::Block(b) = blk(i, j) else { unreachable!() };
            blocks[b] = Some(Block { period: i, terms: vec![single(blk(i - 1, j), blk(i + 1, j), None)] });
        }
    }
    Ok(blocks.into_iter().map(|b| b.expect("every block of the tree is set")).collect())
}

/// The stacked problem of one extended-path period on a tree.
pub struct TreeProblem<'a> {
    model: &'a Model,
    pub params: Vec<f64>,
    ny: usize,
    pub blocks: Vec<Block>,
    /// Initial condition (period 0) and terminal condition (period `periods + 1`).
    pub init: Vec<f64>,
    pub terminal: Vec<f64>,
    /// `(periods + 2) x n_exo`: row 0 = period 0, row 1 = the realized shock, then steady state.
    pub exo: Mat,
    /// Shocks with positive variance (`pfm.positive_var_indx`) and the nodes (`nnodes x neff`).
    pos: Vec<usize>,
    nodes: Mat,
    h_correction: Vec<f64>,
    rows: Vec<u32>,
    cols: Vec<u32>,
    vals: Vec<f64>,
    /// Offset of each block's triplets.
    trip_off: Vec<usize>,
    chunk: usize,
}

impl<'a> TreeProblem<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &'a Model,
        params: Vec<f64>,
        blocks: Vec<Block>,
        exo: Mat,
        pos: Vec<usize>,
        nodes: Mat,
        h_correction: Option<Vec<f64>>,
    ) -> Self {
        let ny = model.n_endo();
        let pattern = model.dynamic_jacobian_pattern();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut trip_off = vec![0usize; blocks.len() + 1];
        for (b, blk) in blocks.iter().enumerate() {
            for t in &blk.terms {
                for &(i, c) in pattern {
                    if let Some(col) = column(t, b, c, ny) {
                        rows.push((b * ny + i) as u32);
                        cols.push(col as u32);
                    }
                }
            }
            trip_off[b + 1] = rows.len();
        }
        let nb = blocks.len();
        let threads = rayon::current_num_threads().max(1);
        // ~0.4 us per term: a chunk must clearly outweigh rayon's wake-up and scheduling cost,
        // which on a loaded machine reaches milliseconds; only large trees run in parallel
        let chunk = nb.div_ceil(threads).max(4096);
        TreeProblem {
            model,
            params,
            ny,
            vals: vec![0.0; rows.len()],
            rows,
            cols,
            trip_off,
            blocks,
            init: vec![0.0; ny],
            terminal: vec![0.0; ny],
            exo,
            pos,
            nodes,
            h_correction: h_correction.unwrap_or_else(|| vec![0.0; ny]),
            chunk,
        }
    }

    /// An empty Jacobian with the problem's pattern.
    pub fn new_jacobian(&self) -> SparseJac {
        SparseJac::from_triplets(self.blocks.len() * self.ny, &self.rows, &self.cols)
    }

    fn eval_blocks(&self, b0: usize, x: &[f64], res: &mut [f64], vals: Option<&mut [f64]>, want_res: bool) {
        let (ny, model) = (self.ny, self.model);
        let pattern = model.dynamic_jacobian_pattern();
        let nx = self.exo.ncols();
        let mut ws = model.workspace();
        let mut y3 = vec![0.0; 3 * ny];
        let mut xr = vec![0.0; nx];
        let mut d1 = vec![0.0; ny];
        let mut jv = vec![0.0; pattern.len()];
        let mut vals = vals;
        let mut q = 0;
        let nblk = res.len() / ny;
        for bb in 0..nblk {
            let b = b0 + bb;
            let blk = &self.blocks[b];
            let r = &mut res[bb * ny..(bb + 1) * ny];
            r.iter_mut().for_each(|v| *v = 0.0);
            for t in &blk.terms {
                match t.prev {
                    Ref::Block(p) => y3[..ny].copy_from_slice(&x[p * ny..(p + 1) * ny]),
                    Ref::Init => y3[..ny].copy_from_slice(&self.init),
                    Ref::Terminal => unreachable!(),
                }
                y3[ny..2 * ny].copy_from_slice(&x[b * ny..(b + 1) * ny]);
                match t.next {
                    Ref::Block(p) => y3[2 * ny..].copy_from_slice(&x[p * ny..(p + 1) * ny]),
                    Ref::Terminal => y3[2 * ny..].copy_from_slice(&self.terminal),
                    Ref::Init => unreachable!(),
                }
                if t.hybrid {
                    for v in 0..ny {
                        y3[2 * ny + v] += self.h_correction[v];
                    }
                }
                for (k, v) in xr.iter_mut().enumerate() {
                    *v = self.exo[(blk.period, k)];
                }
                if let Some(nd) = t.node {
                    for (c, &k) in self.pos.iter().enumerate() {
                        xr[k] = self.nodes[(nd, c)];
                    }
                }
                if want_res {
                    model.dynamic_residuals(&y3, &xr, &self.params, &mut d1);
                    for v in 0..ny {
                        r[v] += t.weight * d1[v];
                    }
                }
                if let Some(vals) = vals.as_deref_mut() {
                    model.dynamic_jacobian_values_with(&y3, &xr, &self.params, &mut ws, &mut jv);
                    for (k, &(_, c)) in pattern.iter().enumerate() {
                        if column(t, b, c, ny).is_some() {
                            vals[q] = t.weight * jv[k];
                            q += 1;
                        }
                    }
                }
            }
        }
    }
}

/// Stacked column of dynamic-Jacobian column `c` in a term of block `b` (`None` for the
/// initial/terminal conditions and the exogenous columns: `i_cols_1`, `i_cols_T`).
#[inline]
fn column(t: &Term, b: usize, c: usize, ny: usize) -> Option<usize> {
    if c < ny {
        match t.prev {
            Ref::Block(p) => Some(p * ny + c),
            _ => None,
        }
    } else if c < 2 * ny {
        Some(b * ny + c - ny)
    } else if c < 3 * ny {
        match t.next {
            Ref::Block(p) => Some(p * ny + c - 2 * ny),
            _ => None,
        }
    } else {
        None
    }
}

impl System for TreeProblem<'_> {
    fn dim(&self) -> usize {
        self.blocks.len() * self.ny
    }

    fn eval(&mut self, x: &[f64], res: &mut [f64], jac: Option<&mut SparseJac>) {
        self.eval_impl(x, res, jac, true)
    }

    fn jacobian(&mut self, x: &[f64], jac: &mut SparseJac) {
        let mut res = vec![0.0; self.dim()];
        self.eval_impl(x, &mut res, Some(jac), false)
    }
}

impl TreeProblem<'_> {
    fn eval_impl(&mut self, x: &[f64], res: &mut [f64], jac: Option<&mut SparseJac>, want_res: bool) {
        let ny = self.ny;
        let nb = self.blocks.len();
        let chunk = self.chunk;
        let want_jac = jac.is_some();
        let mut vals = std::mem::take(&mut self.vals);
        {
            // contiguous block ranges with their residual and triplet slices
            let mut parts: Vec<(usize, &mut [f64], &mut [f64])> = Vec::new();
            let mut rrest: &mut [f64] = res;
            let mut vrest: &mut [f64] = &mut vals;
            let mut b0 = 0;
            while b0 < nb {
                let b1 = (b0 + chunk).min(nb);
                let (r, rr) = rrest.split_at_mut((b1 - b0) * ny);
                let (v, vr) = vrest.split_at_mut(self.trip_off[b1] - self.trip_off[b0]);
                parts.push((b0, r, v));
                rrest = rr;
                vrest = vr;
                b0 = b1;
            }
            let this = &*self;
            if parts.len() > 1 {
                parts
                    .into_par_iter()
                    .for_each(|(b0, r, v)| this.eval_blocks(b0, x, r, if want_jac { Some(v) } else { None }, want_res));
            } else {
                for (b0, r, v) in parts {
                    this.eval_blocks(b0, x, r, if want_jac { Some(v) } else { None }, want_res);
                }
            }
        }
        if let Some(j) = jac {
            j.fill(&self.rows, &self.cols, &vals);
        }
        self.vals = vals;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(kind: TreeKind, order: usize, nnodes: usize, periods: usize) -> usize {
        build_tree(kind, order, nnodes, periods, &vec![1.0 / nnodes as f64; nnodes], false).unwrap().len()
    }

    #[test]
    fn block_counts_match_dynare() {
        // ep_problem_0: dimension/ny = sum(nnodes.^(0:order-1)) + (periods-order)*world_nbr
        assert_eq!(count(TreeKind::Full, 0, 1, 10), 10);
        assert_eq!(count(TreeKind::Full, 1, 3, 10), 1 + 9 * 3);
        assert_eq!(count(TreeKind::Full, 2, 3, 10), 1 + 3 + 8 * 9);
        // get_block_world_nbr (algo 1): order + (nnodes-1)*(order-1)*order/2 + (periods-order)*world_nbr
        assert_eq!(count(TreeKind::Sparse, 2, 3, 10), 2 + 2 + 8 * 5);
        assert_eq!(count(TreeKind::Sparse, 3, 5, 12), 3 + 4 * 3 + 9 * 13);
    }

    #[test]
    fn full_tree_links() {
        let b = build_tree(TreeKind::Full, 1, 3, 4, &[0.2, 0.6, 0.2], false).unwrap();
        // period 1: expectation over the 3 nodes of period 2
        assert_eq!(b[0].terms.len(), 3);
        assert_eq!(b[0].terms[2].next, Ref::Block(3));
        // period 2 node 1 -> period 3 world 1
        assert_eq!(b[2].terms[0].prev, Ref::Block(0));
        assert_eq!(b[2].terms[0].next, Ref::Block(5));
        assert_eq!(b[2].terms[0].node, Some(1));
        // last period -> terminal
        assert_eq!(b[b.len() - 1].terms[0].next, Ref::Terminal);
    }
}
