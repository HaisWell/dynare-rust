// SPDX-License-Identifier: GPL-3.0-or-later
//! a small compressed-row sparse matrix used by the mixed-complementarity solver
//! (`lmmcp.m`) and the iterative linear solvers of the perfect-foresight solver
//! (`stack_solve_algo` 2, 3, 10), with the few operations they need.

use super::sparse_lu::{Csc, LuWork, SparseLu};
use crate::linalg::Mat;

/// Compressed sparse rows (`nrows x ncols`); column indices sorted within each row.
#[derive(Clone, Debug, Default)]
pub struct Csr {
    pub nrows: usize,
    pub ncols: usize,
    pub rowptr: Vec<usize>,
    pub colind: Vec<usize>,
    pub vals: Vec<f64>,
}

impl Csr {
    /// From triplets (duplicates summed).
    pub fn from_triplets(nrows: usize, ncols: usize, mut t: Vec<(usize, usize, f64)>) -> Csr {
        t.sort_unstable_by_key(|&(i, j, _)| (i, j));
        let mut m = Csr { nrows, ncols, rowptr: Vec::with_capacity(nrows + 1), ..Default::default() };
        m.rowptr.push(0);
        let mut k = 0;
        for i in 0..nrows {
            while k < t.len() && t[k].0 == i {
                let j = t[k].1;
                let mut v = 0.0;
                while k < t.len() && t[k].0 == i && t[k].1 == j {
                    v += t[k].2;
                    k += 1;
                }
                m.colind.push(j);
                m.vals.push(v);
            }
            m.rowptr.push(m.colind.len());
        }
        m
    }

    /// From a dense matrix (exact zeros dropped).
    pub fn from_dense(a: &Mat) -> Csr {
        let mut t = Vec::new();
        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                if a[(i, j)] != 0.0 {
                    t.push((i, j, a[(i, j)]));
                }
            }
        }
        Csr::from_triplets(a.nrows(), a.ncols(), t)
    }

    pub fn to_dense(&self) -> Mat {
        let mut a = Mat::zeros(self.nrows, self.ncols);
        for i in 0..self.nrows {
            for p in self.rowptr[i]..self.rowptr[i + 1] {
                a[(i, self.colind[p])] += self.vals[p];
            }
        }
        a
    }

    /// `A x`.
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        (0..self.nrows)
            .map(|i| (self.rowptr[i]..self.rowptr[i + 1]).map(|p| self.vals[p] * x[self.colind[p]]).sum())
            .collect()
    }

    /// `A' x`.
    pub fn tmul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.ncols];
        for i in 0..self.nrows {
            let xi = x[i];
            if xi != 0.0 {
                for p in self.rowptr[i]..self.rowptr[i + 1] {
                    y[self.colind[p]] += self.vals[p] * xi;
                }
            }
        }
        y
    }

    /// `A' diag(d) A + diag(e) A + A' diag(e) + diag(f)` (square `A`; `e`, `f` may be empty),
    /// in compressed-column form (for [`SparseLu`]).
    pub fn normal_matrix(&self, d: &[f64], e: &[f64], f: &[f64]) -> Csc {
        let n = self.ncols;
        let mut t: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..self.nrows {
            let r = self.rowptr[i]..self.rowptr[i + 1];
            let di = d[i];
            if di != 0.0 {
                for p in r.clone() {
                    for q in r.clone() {
                        t.push((self.colind[p], self.colind[q], di * self.vals[p] * self.vals[q]));
                    }
                }
            }
            if !e.is_empty() && e[i] != 0.0 {
                // (diag(e) A)(i, :) and its transpose
                for p in r {
                    t.push((i, self.colind[p], e[i] * self.vals[p]));
                    t.push((self.colind[p], i, e[i] * self.vals[p]));
                }
            }
        }
        for (i, &v) in f.iter().enumerate() {
            if v != 0.0 {
                t.push((i, i, v));
            }
        }
        // compressed columns
        t.sort_unstable_by_key(|&(i, j, _)| (j, i));
        let mut c = Csc { n, colptr: Vec::with_capacity(n + 1), ..Default::default() };
        c.colptr.push(0);
        let mut k = 0;
        for j in 0..n {
            while k < t.len() && t[k].1 == j {
                let i = t[k].0;
                let mut v = 0.0;
                while k < t.len() && t[k].1 == j && t[k].0 == i {
                    v += t[k].2;
                    k += 1;
                }
                c.rowind.push(i);
                c.vals.push(v);
            }
            c.colptr.push(c.rowind.len());
        }
        c
    }

    /// Compressed columns of a square matrix.
    pub fn to_csc(&self) -> Csc {
        let n = self.nrows;
        let mut t: Vec<(usize, usize, f64)> = Vec::with_capacity(self.vals.len());
        for i in 0..n {
            for p in self.rowptr[i]..self.rowptr[i + 1] {
                t.push((i, self.colind[p], self.vals[p]));
            }
        }
        t.sort_unstable_by_key(|&(i, j, _)| (j, i));
        let mut c = Csc { n, colptr: Vec::with_capacity(n + 1), ..Default::default() };
        c.colptr.push(0);
        let mut k = 0;
        for j in 0..n {
            while k < t.len() && t[k].1 == j {
                c.rowind.push(t[k].0);
                c.vals.push(t[k].2);
                k += 1;
            }
            c.colptr.push(c.rowind.len());
        }
        c
    }

    /// `A \ b` for a square matrix (sparse LU with partial pivoting); `None` if singular.
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        let lu = SparseLu::factor(&self.to_csc(), None, 1.0, &mut LuWork::default())?;
        let mut x = b.to_vec();
        lu.solve(&mut x);
        Some(x)
    }
}

/// `norm(v)` (2-norm).
pub fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `norm(v, Inf)` (NaN if any entry is NaN).
pub fn norm_inf(v: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for &x in v {
        if x.is_nan() {
            return f64::NAN;
        }
        m = m.max(x.abs());
    }
    m
}
