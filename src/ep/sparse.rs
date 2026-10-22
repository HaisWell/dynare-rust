// SPDX-License-Identifier: GPL-3.0-or-later
//! Sparse Jacobians of the (stochastic) extended-path problems: a fixed-pattern
//! compressed-column matrix assembled from triplets, its products, and a sparse LU
//! (`deterministic::sparse_lu`) with a block-level elimination order.
//!
//! The unknowns and the residuals of the extended-path problems are grouped in blocks of
//! `ny` (one block = one period of one node of the tree), and the residual block `b` holds the
//! equations of the unknown block `b`. The LU eliminates the blocks in a minimum-degree order
//! of the block graph (a tree of paths: leaves first, so no fill spreads across worlds, where
//! MATLAB's UMFPACK uses COLAMD), the variables of a block in the order of
//! `deterministic::stacked::StackedStructure` and with its matched equation as the preferred
//! pivot (threshold partial pivoting, as KLU).

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

use crate::deterministic::sparse_lu::{Csc, LuWork, NONE, SparseLu};
use crate::deterministic::stacked::StackedStructure;

/// Threshold of the preferred pivot relative to the largest candidate (KLU's default).
const PIVOT_TOL: f64 = 0.001;

/// Square sparse matrix with a fixed pattern, filled from a triplet stream whose
/// `(row, col)` sequence is the same at every evaluation (duplicates are summed in stream order).
#[derive(Clone, Debug, Default)]
pub struct SparseJac {
    pub csc: Csc,
    /// Slot in `csc.vals` of each triplet of the stream.
    map: Vec<usize>,
    /// The `(row, col)` stream the pattern was built from (checked in debug builds).
    rows: Vec<u32>,
    cols: Vec<u32>,
}

impl SparseJac {
    /// Builds the pattern from a triplet stream (values ignored).
    pub fn from_triplets(n: usize, rows: &[u32], cols: &[u32]) -> Self {
        let m = rows.len();
        let mut idx: Vec<usize> = (0..m).collect();
        idx.sort_by_key(|&k| (cols[k], rows[k]));
        let mut colptr = vec![0usize; n + 1];
        let mut rowind = Vec::with_capacity(m);
        let mut map = vec![0usize; m];
        let mut last: Option<(u32, u32)> = None;
        for &k in &idx {
            let key = (cols[k], rows[k]);
            if last != Some(key) {
                rowind.push(rows[k] as usize);
                colptr[cols[k] as usize + 1] += 1;
                last = Some(key);
            }
            map[k] = rowind.len() - 1;
        }
        for j in 0..n {
            colptr[j + 1] += colptr[j];
        }
        let nnz = rowind.len();
        SparseJac {
            csc: Csc { n, colptr, rowind, vals: vec![0.0; nnz] },
            map,
            rows: rows.to_vec(),
            cols: cols.to_vec(),
        }
    }

    /// Fills the values from a triplet stream with the pattern's `(row, col)` sequence.
    pub fn fill(&mut self, rows: &[u32], cols: &[u32], vals: &[f64]) {
        debug_assert!(rows == &self.rows[..] && cols == &self.cols[..], "triplet stream changed");
        let _ = (rows, cols);
        self.csc.vals.iter_mut().for_each(|v| *v = 0.0);
        for (k, &v) in vals.iter().enumerate() {
            self.csc.vals[self.map[k]] += v;
        }
    }

    pub fn n(&self) -> usize {
        self.csc.n
    }

    /// `A x`.
    pub fn mul(&self, x: &[f64]) -> Vec<f64> {
        self.csc.mul_vec(x)
    }

    /// `A' x`.
    pub fn tmul(&self, x: &[f64]) -> Vec<f64> {
        let a = &self.csc;
        (0..a.n).map(|j| (a.colptr[j]..a.colptr[j + 1]).map(|p| a.vals[p] * x[a.rowind[p]]).sum()).collect()
    }

    /// Euclidean norms of the columns (`sqrt(sum(A.*A))'`).
    pub fn col_norms(&self) -> Vec<f64> {
        let a = &self.csc;
        (0..a.n).map(|j| (a.colptr[j]..a.colptr[j + 1]).map(|p| a.vals[p] * a.vals[p]).sum::<f64>().sqrt()).collect()
    }

    pub fn all_finite(&self) -> bool {
        self.csc.vals.iter().all(|v| v.is_finite())
    }
}

/// Elimination order of a block-structured square matrix (see the module doc).
pub struct LuPlan {
    /// Column `k` of the permuted matrix is column `colperm[k]` of `A`.
    colperm: Vec<usize>,
    /// Preferred pivot row of each permuted column (`NONE`: plain partial pivoting).
    prefer: Vec<usize>,
    work: LuWork,
    perm: Csc,
}

impl LuPlan {
    /// `ny` variables per block; `s` gives the within-block order and preferred rows.
    pub fn new(a: &SparseJac, ny: usize, s: &StackedStructure) -> Self {
        let n = a.n();
        assert_eq!(n % ny, 0);
        let nb = n / ny;
        // block graph
        let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nb];
        let c = &a.csc;
        for j in 0..n {
            for p in c.colptr[j]..c.colptr[j + 1] {
                let (bi, bj) = (c.rowind[p] / ny, j / ny);
                if bi != bj {
                    adj[bi].insert(bj);
                    adj[bj].insert(bi);
                }
            }
        }
        let border = minimum_degree(adj);
        let mut colperm = Vec::with_capacity(n);
        let mut prefer = Vec::with_capacity(n);
        for &b in &border {
            for &v in &s.order {
                colperm.push(b * ny + v);
                prefer.push(if s.match_row[v] == NONE { NONE } else { b * ny + s.match_row[v] });
            }
        }
        let mut perm =
            Csc { n, colptr: Vec::with_capacity(n + 1), rowind: Vec::with_capacity(c.rowind.len()), vals: vec![] };
        perm.colptr.push(0);
        for &j in &colperm {
            perm.rowind.extend_from_slice(&c.rowind[c.colptr[j]..c.colptr[j + 1]]);
            perm.colptr.push(perm.rowind.len());
        }
        perm.vals = vec![0.0; perm.rowind.len()];
        LuPlan { colperm, prefer, work: LuWork::default(), perm }
    }

    /// Factorizes `a` (same pattern as at construction); `None` if singular.
    pub fn factor(&mut self, a: &SparseJac) -> Option<Factor> {
        let c = &a.csc;
        let mut q = 0;
        for &j in &self.colperm {
            let len = c.colptr[j + 1] - c.colptr[j];
            self.perm.vals[q..q + len].copy_from_slice(&c.vals[c.colptr[j]..c.colptr[j + 1]]);
            q += len;
        }
        SparseLu::factor(&self.perm, Some(&self.prefer), PIVOT_TOL, &mut self.work)
            .map(|lu| Factor { lu, colperm: self.colperm.clone() })
    }
}

/// An LU factorization of `A` with permuted columns.
pub struct Factor {
    lu: SparseLu,
    colperm: Vec<usize>,
}

impl Factor {
    /// `A \ b`.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let mut z = b.to_vec();
        self.lu.solve(&mut z);
        let mut x = vec![0.0; z.len()];
        for (k, &j) in self.colperm.iter().enumerate() {
            x[j] = z[k];
        }
        x
    }
}

/// Minimum-degree ordering of a symmetric graph (explicit elimination graph, lazy heap).
pub fn minimum_degree(mut adj: Vec<BTreeSet<usize>>) -> Vec<usize> {
    let n = adj.len();
    let mut done = vec![false; n];
    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = (0..n).map(|v| Reverse((adj[v].len(), v))).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(Reverse((d, v))) = heap.pop() {
        if done[v] || d != adj[v].len() {
            continue;
        }
        done[v] = true;
        order.push(v);
        let nb: Vec<usize> = std::mem::take(&mut adj[v]).into_iter().collect();
        for &a in &nb {
            adj[a].remove(&v);
        }
        for &a in &nb {
            for &b in &nb {
                if a != b {
                    adj[a].insert(b);
                }
            }
        }
        for &a in &nb {
            heap.push(Reverse((adj[a].len(), a)));
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_degree_eliminates_leaves_first() {
        // path 0-1-2 with a leaf 3 on 1
        let mut adj = vec![BTreeSet::new(); 4];
        for (a, b) in [(0, 1), (1, 2), (1, 3)] {
            adj[a].insert(b);
            adj[b].insert(a);
        }
        let o = minimum_degree(adj);
        assert_eq!(o.len(), 4);
        assert_ne!(o[0], 1);
    }

    #[test]
    fn triplets_sum_duplicates() {
        let rows = [0u32, 1, 0, 1, 0];
        let cols = [0u32, 0, 1, 1, 0];
        let mut a = SparseJac::from_triplets(2, &rows, &cols);
        a.fill(&rows, &cols, &[1.0, 2.0, 3.0, 4.0, 0.5]);
        assert_eq!(a.mul(&[1.0, 0.0]), vec![1.5, 2.0]);
        assert_eq!(a.tmul(&[1.0, 0.0]), vec![1.5, 3.0]);
    }
}
