// SPDX-License-Identifier: GPL-3.0-or-later
//! Sparse LU with partial pivoting: a port of CSparse's `cs_lu` / `cs_spsolve` / `cs_reach`
//! / `cs_dfs` / `cs_lusol` (T. A. Davis, *Direct Methods for Sparse Linear Systems*, 2006),
//! the left-looking Gilbert–Peierls algorithm. Used for the stacked perfect-foresight
//! Jacobian, where MATLAB's `A\b` calls UMFPACK.

/// Compressed-column matrix.
#[derive(Clone, Debug, Default)]
pub struct Csc {
    pub n: usize,
    pub colptr: Vec<usize>,
    pub rowind: Vec<usize>,
    pub vals: Vec<f64>,
}

impl Csc {
    /// `y = A x`.
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n];
        for j in 0..self.n {
            for p in self.colptr[j]..self.colptr[j + 1] {
                y[self.rowind[p]] += self.vals[p] * x[j];
            }
        }
        y
    }
}

/// `L U = A(p, :)` (natural column order), `L` unit lower triangular with the diagonal first
/// in each column, `U` with the diagonal last.
pub struct SparseLu {
    n: usize,
    lp: Vec<usize>,
    li: Vec<usize>,
    lx: Vec<f64>,
    up: Vec<usize>,
    ui: Vec<usize>,
    ux: Vec<f64>,
    /// pinv[i] = k: row i of A is the k-th pivot row.
    pinv: Vec<usize>,
}

pub const NONE: usize = usize::MAX;

/// Workspace of the factorization (reused across factorizations of the same size).
#[derive(Default)]
pub struct LuWork {
    x: Vec<f64>,
    xi: Vec<usize>,
    pstack: Vec<usize>,
    mark: Vec<u32>,
    stamp: u32,
}

impl SparseLu {
    /// Factorizes `a` with threshold partial pivoting (`cs_lu`): in column k the preferred
    /// row `prefer[k]` is the pivot if its magnitude is at least `tol` times the largest
    /// candidate, else the largest candidate (`prefer = None`: plain partial pivoting).
    /// `None` if a pivot is zero or not finite (singular matrix).
    pub fn factor(a: &Csc, prefer: Option<&[usize]>, tol: f64, w: &mut LuWork) -> Option<SparseLu> {
        let n = a.n;
        if w.x.len() < n {
            w.x = vec![0.0; n];
            w.xi = vec![0; n];
            w.pstack = vec![0; n];
            w.mark = vec![0; n];
            w.stamp = 0;
        }
        let guess = 4 * a.vals.len() + n;
        let mut f = SparseLu {
            n,
            lp: vec![0; n + 1],
            li: Vec::with_capacity(guess),
            lx: Vec::with_capacity(guess),
            up: vec![0; n + 1],
            ui: Vec::with_capacity(guess),
            ux: Vec::with_capacity(guess),
            pinv: vec![NONE; n],
        };
        for k in 0..n {
            f.lp[k] = f.li.len();
            f.up[k] = f.ui.len();
            let top = f.spsolve(a, k, w);
            // pivot: largest entry among the not-yet-pivotal rows
            let mut ipiv = NONE;
            let mut amax = -1.0f64;
            for p in top..n {
                let i = w.xi[p];
                if f.pinv[i] == NONE {
                    let t = w.x[i].abs();
                    if t > amax {
                        amax = t;
                        ipiv = i;
                    }
                } else {
                    f.ui.push(f.pinv[i]);
                    f.ux.push(w.x[i]);
                }
            }
            if ipiv == NONE || !(amax > 0.0) || !amax.is_finite() {
                for p in top..n {
                    w.x[w.xi[p]] = 0.0;
                }
                return None;
            }
            if let Some(pr) = prefer {
                let c = pr[k];
                if c != NONE && f.pinv[c] == NONE && w.x[c].abs() >= amax * tol && w.x[c] != 0.0 {
                    // only rows in the reach have a defined x; others were never touched (0)
                    ipiv = c;
                }
            }
            let pivot = w.x[ipiv];
            f.ui.push(k);
            f.ux.push(pivot);
            f.pinv[ipiv] = k;
            f.li.push(ipiv);
            f.lx.push(1.0);
            for p in top..n {
                let i = w.xi[p];
                if f.pinv[i] == NONE {
                    f.li.push(i);
                    f.lx.push(w.x[i] / pivot);
                }
                w.x[i] = 0.0;
            }
        }
        f.lp[n] = f.li.len();
        f.up[n] = f.ui.len();
        for i in f.li.iter_mut() {
            *i = f.pinv[*i];
        }
        Some(f)
    }

    /// `x = L \ A(:, k)` on the reach of column k; returns `top` (pattern in `xi[top..n]`).
    fn spsolve(&self, a: &Csc, k: usize, w: &mut LuWork) -> usize {
        let n = self.n;
        let top = self.reach(a, k, w);
        for p in top..n {
            w.x[w.xi[p]] = 0.0;
        }
        for p in a.colptr[k]..a.colptr[k + 1] {
            w.x[a.rowind[p]] = a.vals[p];
        }
        for px in top..n {
            let j = w.xi[px];
            let jj = self.pinv[j];
            if jj == NONE {
                continue;
            }
            // unit diagonal first: skip it
            let xj = w.x[j];
            for p in self.lp[jj] + 1..self.lp[jj + 1] {
                w.x[self.li[p]] -= self.lx[p] * xj;
            }
        }
        top
    }

    /// `cs_reach`: topologically ordered nonzero pattern of `L \ A(:, k)`.
    fn reach(&self, a: &Csc, k: usize, w: &mut LuWork) -> usize {
        let n = self.n;
        w.stamp = w.stamp.wrapping_add(1);
        if w.stamp == 0 {
            w.mark.fill(0);
            w.stamp = 1;
        }
        let mut top = n;
        for p in a.colptr[k]..a.colptr[k + 1] {
            let i = a.rowind[p];
            if w.mark[i] != w.stamp {
                top = self.dfs(i, top, w);
            }
        }
        top
    }

    /// `cs_dfs` (non-recursive) through the graph of `L` (row indices still unpermuted).
    fn dfs(&self, j0: usize, mut top: usize, w: &mut LuWork) -> usize {
        let mut head: isize = 0;
        w.xi[0] = j0;
        while head >= 0 {
            let h = head as usize;
            let j = w.xi[h];
            let jnew = self.pinv[j];
            if w.mark[j] != w.stamp {
                w.mark[j] = w.stamp;
                w.pstack[h] = if jnew == NONE { 0 } else { self.lp[jnew] + 1 };
            }
            let mut done = true;
            let p2 = if jnew == NONE { 0 } else { self.lp[jnew + 1] };
            let mut p = w.pstack[h];
            while p < p2 {
                let i = self.li[p];
                if w.mark[i] != w.stamp {
                    w.pstack[h] = p + 1;
                    head += 1;
                    w.xi[head as usize] = i;
                    done = false;
                    break;
                }
                p += 1;
            }
            if done {
                head -= 1;
                top -= 1;
                // xi[top] = j: top > head always holds (n slots, at most n distinct nodes)
                w.xi[top] = j;
            }
        }
        top
    }

    /// Solves `A x = b` in place.
    pub fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[self.pinv[i]] = b[i];
        }
        for j in 0..n {
            let xj = x[j];
            if xj != 0.0 {
                for p in self.lp[j] + 1..self.lp[j + 1] {
                    x[self.li[p]] -= self.lx[p] * xj;
                }
            }
        }
        for j in (0..n).rev() {
            let d = self.up[j + 1] - 1;
            x[j] /= self.ux[d];
            let xj = x[j];
            if xj != 0.0 {
                for p in self.up[j]..d {
                    x[self.ui[p]] -= self.ux[p] * xj;
                }
            }
        }
        b.copy_from_slice(&x);
    }

    /// Number of non-zeros of L + U.
    pub fn nnz(&self) -> usize {
        self.li.len() + self.ui.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_random_sparse_system_needing_pivoting() {
        let n = 60;
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        // zero diagonal (forces row interchanges) + a few random entries per column
        let mut dense = vec![0.0; n * n];
        for j in 0..n {
            dense[(j + 1) % n + j * n] = 1.0 + rnd();
            for _ in 0..3 {
                let i = (rnd() * n as f64) as usize % n;
                if i != j {
                    dense[i + j * n] += rnd() - 0.5;
                }
            }
        }
        let mut a = Csc { n, colptr: vec![0], ..Default::default() };
        for j in 0..n {
            for i in 0..n {
                if dense[i + j * n] != 0.0 {
                    a.rowind.push(i);
                    a.vals.push(dense[i + j * n]);
                }
            }
            a.colptr.push(a.rowind.len());
        }
        let xt: Vec<f64> = (0..n).map(|k| (k as f64 * 0.37).cos()).collect();
        let mut b = a.mul_vec(&xt);
        let mut w = LuWork::default();
        let f = SparseLu::factor(&a, None, 1.0, &mut w).unwrap();
        f.solve(&mut b);
        for k in 0..n {
            assert!((b[k] - xt[k]).abs() < 1e-10, "{k}: {} vs {}", b[k], xt[k]);
        }
        // singular: a zero column
        let mut s = a.clone();
        for p in s.colptr[3]..s.colptr[4] {
            s.vals[p] = 0.0;
        }
        assert!(SparseLu::factor(&s, None, 1.0, &mut w).is_none());
    }
}
