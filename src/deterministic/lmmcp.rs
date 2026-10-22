// SPDX-License-Identifier: GPL-3.0-or-later
//! port of `lmmcp/lmmcp.m` (C. Kanzow and S. Petra's Levenberg–Marquardt method
//! for mixed complementarity problems, as shipped with Dynare): find `lb <= x <= ub` with
//! `F_i(x) = 0` if `lb_i < x_i < ub_i`, `F_i(x) >= 0` if `x_i = lb_i`, `F_i(x) <= 0` if
//! `x_i = ub_i`. Used by `solve_algo = 10` (steady state), `perfect_foresight_solver(lmmcp)`
//! (`stack_solve_algo = 7`) and `extended_path(lmmcp)`.
//!
//! Same options and defaults as Dynare (`options_.lmmcp` merged with lmmcp's `defaultopt`:
//! `MaxIter` 500, `TolFun` sqrt(eps), preprocessor with 20 projected Levenberg–Marquardt steps,
//! non-monotone Armijo line search, watchdog). The regularized least-squares step
//! `[DPhi; sqrt(mu) I] \ [-Phi; 0]` (MATLAB's sparse QR) is computed from the normal equations
//! `(DPhi'DPhi + mu I) d = -DPhi'Phi` with the sparse LU (an LU of the 2n x n system's normal
//! matrix: same solution up to rounding when DPhi has full column rank); `condest` is the exact
//! 1-norm condition number for n < 100, where MATLAB estimates it.

use super::sparse_lu::{LuWork, SparseLu};
use super::spmat::{Csr, norm_inf, norm2};
use crate::linalg::{Mat, lu, lu_solve, norm1};

/// Options of `lmmcp.m` (its `defaultopt`).
#[derive(Clone, Debug)]
pub struct LmmcpOptions {
    pub beta: f64,
    pub big: f64,
    pub delta: f64,
    pub deltamin: f64,
    pub epsilon1: f64,
    pub eta: f64,
    pub kwatch: usize,
    pub lambda1: f64,
    pub m: usize,
    pub max_iter: usize,
    pub null: f64,
    pub preprocess: bool,
    pub presteps: usize,
    pub sigma: f64,
    pub sigma1: f64,
    pub sigma2: f64,
    pub tmin: f64,
    pub tol_fun: f64,
    pub watchdog: bool,
    /// `Display`: 0 none, 1 final, 2 iter.
    pub verbosity: u8,
}

impl Default for LmmcpOptions {
    fn default() -> Self {
        LmmcpOptions {
            beta: 0.55,
            big: 1e10,
            delta: 5.0,
            deltamin: 1.0,
            epsilon1: 1e-6,
            eta: 0.95,
            kwatch: 20,
            lambda1: 0.1,
            m: 10,
            max_iter: 500,
            null: 1e-10,
            preprocess: true,
            presteps: 20,
            sigma: 1e-4,
            sigma1: 0.5,
            sigma2: 2.0,
            tmin: 1e-12,
            tol_fun: f64::EPSILON.sqrt(),
            watchdog: true,
            verbosity: 0,
        }
    }
}

/// Result of [`lmmcp`]: `EXITFLAG` 1 solution, 0 iteration limit, -1 failure.
#[derive(Clone, Debug)]
pub struct LmmcpResult {
    pub x: Vec<f64>,
    pub fval: Vec<f64>,
    pub exitflag: i32,
    pub iterations: usize,
    pub psix: f64,
    pub norm_dpsix: f64,
}

/// The problem: `f(x, jacobian)` returns `F(x)` and, when asked, its (square) Jacobian.
pub type McpFn<'a> = dyn FnMut(&[f64], bool) -> (Vec<f64>, Option<Csr>) + 'a;

/// `Indexset`: 0 free, 1 lower bound only, 2 upper bound only, 3 both.
fn index_set(lb: &[f64], ub: &[f64], big: f64) -> Vec<u8> {
    lb.iter()
        .zip(ub)
        .map(|(&l, &u)| {
            if l > -big && u > big {
                1
            } else if l < -big && u < big {
                2
            } else if l > -big && u < big {
                3
            } else {
                0
            }
        })
        .collect()
}

/// `Phi` (length 2n).
fn phi(x: &[f64], fx: &[f64], lb: &[f64], ub: &[f64], l1: f64, l2: f64, ix: &[u8]) -> Vec<f64> {
    let n = x.len();
    let mut y = vec![0.0; 2 * n];
    for i in 0..n {
        let phi_u = ((ub[i] - x[i]).powi(2) + fx[i] * fx[i]).sqrt() - ub[i] + x[i] + fx[i];
        match ix[i] {
            0 => {
                y[i] = -l1 * fx[i];
                y[n + i] = -l2 * fx[i];
            }
            1 => {
                y[i] = l1 * (-x[i] + lb[i] - fx[i] + ((x[i] - lb[i]).powi(2) + fx[i] * fx[i]).sqrt());
                y[n + i] = l2 * (x[i] - lb[i]).max(0.0) * fx[i].max(0.0);
            }
            2 => {
                y[i] = -l1 * phi_u;
                y[n + i] = l2 * (ub[i] - x[i]).max(0.0) * (-fx[i]).max(0.0);
            }
            _ => {
                y[i] = l1 * (((x[i] - lb[i]).powi(2) + phi_u * phi_u).sqrt() - x[i] + lb[i] - phi_u);
                y[n + i] =
                    l2 * ((x[i] - lb[i]).max(0.0) * fx[i].max(0.0) + (ub[i] - x[i]).max(0.0) * (-fx[i]).max(0.0));
            }
        }
    }
    y
}

/// `DPhi` as row scalings of the Jacobian: `H = [diag(a1) J + diag(b1); diag(a2) J + diag(b2)]`.
struct DPhi {
    a1: Vec<f64>,
    b1: Vec<f64>,
    a2: Vec<f64>,
    b2: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn dphi(x: &[f64], fx: &[f64], j: &Csr, lb: &[f64], ub: &[f64], l1: f64, l2: f64, ix: &[u8]) -> DPhi {
    let n = x.len();
    let null = 1e-8;
    let mut beta_l = vec![false; n];
    let mut beta_u = vec![false; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        if (x[i] - lb[i]).abs() <= null && fx[i].abs() <= null {
            beta_l[i] = true;
            z[i] = 1.0;
        }
        if (ub[i] - x[i]).abs() <= null && fx[i].abs() <= null {
            beta_u[i] = true;
            z[i] = 1.0;
        }
    }
    let alpha_l: Vec<bool> = (0..n).map(|i| x[i] - lb[i] >= -null && fx[i] >= -null).collect();
    let alpha_u: Vec<bool> = (0..n).map(|i| ub[i] - x[i] >= -null && fx[i] <= null).collect();
    let jz = j.mul_vec(&z);
    let (mut da, mut db) = (vec![0.0; n], vec![0.0; n]);
    let (mut c2, mut d2) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        match ix[i] {
            0 => {
                da[i] = 0.0;
                db[i] = -1.0;
                c2[i] = -1.0;
            }
            1 => {
                let denom1 = null.max(((x[i] - lb[i]).powi(2) + fx[i] * fx[i]).sqrt());
                let denom2 = null.max((z[i] * z[i] + jz[i] * jz[i]).sqrt());
                if !beta_l[i] {
                    da[i] = (x[i] - lb[i]) / denom1 - 1.0;
                    db[i] = fx[i] / denom1 - 1.0;
                } else {
                    da[i] = z[i] / denom2 - 1.0;
                    db[i] = jz[i] / denom2 - 1.0;
                }
                if alpha_l[i] {
                    c2[i] = x[i] - lb[i];
                    d2[i] = fx[i];
                }
            }
            2 => {
                let denom1 = null.max(((ub[i] - x[i]).powi(2) + fx[i] * fx[i]).sqrt());
                let denom2 = null.max((z[i] * z[i] + jz[i] * jz[i]).sqrt());
                if !beta_u[i] {
                    da[i] = (ub[i] - x[i]) / denom1 - 1.0;
                    db[i] = -fx[i] / denom1 - 1.0;
                } else {
                    da[i] = -z[i] / denom2 - 1.0;
                    db[i] = -jz[i] / denom2 - 1.0;
                }
                if alpha_u[i] {
                    c2[i] = x[i] - ub[i];
                    d2[i] = fx[i];
                }
            }
            _ => {
                let ph = -ub[i] + x[i] + fx[i] + ((ub[i] - x[i]).powi(2) + fx[i] * fx[i]).sqrt();
                let denom1 = null.max(((x[i] - lb[i]).powi(2) + ph * ph).sqrt());
                let denom2 = null.max((z[i] * z[i] + jz[i] * jz[i]).sqrt());
                let denom3 = null.max(((ub[i] - x[i]).powi(2) + fx[i] * fx[i]).sqrt());
                let denom4 = null.max(z[i].abs());
                let (ci, di) = if !beta_u[i] {
                    ((x[i] - ub[i]) / denom3 + 1.0, fx[i] / denom3 + 1.0)
                } else {
                    (1.0 + z[i] / denom2, 1.0 + jz[i] / denom2)
                };
                let (ai, bi) = if !beta_l[i] {
                    ((x[i] - lb[i]) / denom1 - 1.0, ph / denom1 - 1.0)
                } else {
                    (z[i] / denom4 - 1.0, (ci * z[i] + di * jz[i]) / denom4 - 1.0)
                };
                da[i] = ai + bi * ci;
                db[i] = bi * di;
                match (alpha_l[i], alpha_u[i]) {
                    (true, true) => {
                        c2[i] = -lb[i] - ub[i] + 2.0 * x[i];
                        d2[i] = 2.0 * fx[i];
                    }
                    (true, false) => {
                        c2[i] = x[i] - lb[i];
                        d2[i] = fx[i];
                    }
                    (false, true) => {
                        c2[i] = x[i] - ub[i];
                        d2[i] = fx[i];
                    }
                    (false, false) => {}
                }
            }
        }
    }
    DPhi {
        a1: db.iter().map(|v| l1 * v).collect(),
        b1: da.iter().map(|v| l1 * v).collect(),
        a2: c2.iter().map(|v| l2 * v).collect(),
        b2: d2.iter().map(|v| l2 * v).collect(),
    }
}

impl DPhi {
    /// `DPhi' * v` (v of length 2n).
    fn tmul(&self, j: &Csr, v: &[f64]) -> Vec<f64> {
        let n = self.a1.len();
        let w: Vec<f64> = (0..n).map(|i| self.a1[i] * v[i] + self.a2[i] * v[n + i]).collect();
        let mut r = j.tmul_vec(&w);
        for i in 0..n {
            r[i] += self.b1[i] * v[i] + self.b2[i] * v[n + i];
        }
        r
    }

    /// `[DPhi; sqrt(mu) I] \ [-Phi; 0]`: the least-squares step by a row-wise Givens QR of the
    /// (banded, once rows are sorted by their first column) sparse matrix, as MATLAB's sparse QR
    /// solves it.
    fn step(&self, j: &Csr, phix: &[f64], mu: f64) -> Option<Vec<f64>> {
        let n = self.a1.len();
        let mut rows: Vec<(Vec<(usize, f64)>, f64)> = Vec::with_capacity(3 * n);
        for (blk, (a, b)) in [(&self.a1, &self.b1), (&self.a2, &self.b2)].into_iter().enumerate() {
            for i in 0..n {
                let mut r: Vec<(usize, f64)> = Vec::new();
                if a[i] != 0.0 {
                    for p in j.rowptr[i]..j.rowptr[i + 1] {
                        r.push((j.colind[p], a[i] * j.vals[p]));
                    }
                }
                if b[i] != 0.0 {
                    match r.iter_mut().find(|e| e.0 == i) {
                        Some(e) => e.1 += b[i],
                        None => r.push((i, b[i])),
                    }
                }
                r.retain(|e| e.1 != 0.0);
                r.sort_unstable_by_key(|e| e.0);
                if !r.is_empty() {
                    rows.push((r, -phix[blk * n + i]));
                }
            }
        }
        if mu > 0.0 {
            let s = mu.sqrt();
            for i in 0..n {
                rows.push((vec![(i, s)], 0.0));
            }
        }
        let d = lstsq_givens(rows, n);
        d.iter().all(|v| v.is_finite()).then_some(d)
    }

    /// Fallback: `(DPhi'DPhi + mu I) d = -DPhi'Phi` with the sparse LU.
    fn step_normal(&self, j: &Csr, phix: &[f64], mu: f64) -> Option<Vec<f64>> {
        let n = self.a1.len();
        let d: Vec<f64> = (0..n).map(|i| self.a1[i] * self.a1[i] + self.a2[i] * self.a2[i]).collect();
        let e: Vec<f64> = (0..n).map(|i| self.a1[i] * self.b1[i] + self.a2[i] * self.b2[i]).collect();
        let f: Vec<f64> = (0..n).map(|i| self.b1[i] * self.b1[i] + self.b2[i] * self.b2[i] + mu).collect();
        let m = j.normal_matrix(&d, &e, &f);
        let lu = SparseLu::factor(&m, None, 1.0, &mut LuWork::default())?;
        let mut rhs: Vec<f64> = self.tmul(j, phix).iter().map(|v| -v).collect();
        lu.solve(&mut rhs);
        rhs.iter().all(|v| v.is_finite()).then_some(rhs)
    }

    fn condest(&self, j: &Csr) -> f64 {
        let n = self.a1.len();
        let d: Vec<f64> = (0..n).map(|i| self.a1[i] * self.a1[i] + self.a2[i] * self.a2[i]).collect();
        let e: Vec<f64> = (0..n).map(|i| self.a1[i] * self.b1[i] + self.a2[i] * self.b2[i]).collect();
        let f: Vec<f64> = (0..n).map(|i| self.b1[i] * self.b1[i] + self.b2[i] * self.b2[i]).collect();
        let c = j.normal_matrix(&d, &e, &f);
        let mut a = Mat::zeros(n, n);
        for col in 0..n {
            for p in c.colptr[col]..c.colptr[col + 1] {
                a[(c.rowind[p], col)] = c.vals[p];
            }
        }
        let an = norm1(&a);
        let (fct, ipiv, info) = lu(&a);
        if info != 0 {
            return f64::INFINITY;
        }
        let mut inv = Mat::identity(n, n);
        lu_solve(&fct, &ipiv, false, &mut inv);
        let r = an * norm1(&inv);
        if r.is_finite() { r } else { f64::INFINITY }
    }

    /// The Levenberg–Marquardt step of the preprocessor (`mu_big`: 1e-6/(k+1)) or of the
    /// main loop (1e-1/(k+1)).
    fn lm_step(&self, j: &Csr, phix: &[f64], mu_big: f64) -> Vec<f64> {
        let n = self.a1.len();
        let mu = if n < 100 { if self.condest(j) > 1e25 { mu_big } else { 1e-16 } } else { 0.0 };
        if let Some(d) = self.step(j, phix, mu) {
            return d;
        }
        self.step_normal(j, phix, mu.max(1e-12)).unwrap_or_else(|| vec![f64::NAN; n])
    }
}

/// Least squares `min ||A x - b||` for sparse rows `(entries sorted by column, rhs)`: row-wise
/// Givens QR (rows taken by increasing first column, so `R` keeps the envelope of `A`), then
/// back substitution; columns without a usable pivot get 0 (a basic solution, as a
/// rank-deficient sparse QR).
pub fn lstsq_givens(mut rows: Vec<(Vec<(usize, f64)>, f64)>, ncols: usize) -> Vec<f64> {
    rows.sort_by_key(|r| r.0.first().map_or(usize::MAX, |e| e.0));
    // R row k: dense values over columns k..k+len, and its rhs
    let mut r: Vec<Option<(Vec<f64>, f64)>> = vec![None; ncols];
    let mut w: Vec<f64> = Vec::new();
    for (entries, rhs) in rows {
        let Some(&(c0, _)) = entries.first() else { continue };
        let mut end = entries.last().unwrap().0 + 1;
        w.clear();
        w.resize(end - c0, 0.0);
        for &(c, v) in &entries {
            w[c - c0] += v;
        }
        let mut beta = rhs;
        let mut k = c0;
        while k < end {
            let a = w[k - c0];
            if a == 0.0 {
                k += 1;
                continue;
            }
            match &mut r[k] {
                None => {
                    r[k] = Some((w[k - c0..end - c0].to_vec(), beta));
                    break;
                }
                Some((rk, rb)) => {
                    let rend = k + rk.len();
                    if rend > end {
                        w.resize(rend - c0, 0.0);
                        end = rend;
                    } else if end > rend {
                        rk.resize(end - k, 0.0);
                    }
                    let (p, q) = (rk[0], a);
                    let h = p.hypot(q);
                    let (c, s) = (p / h, q / h);
                    for m in 0..rk.len() {
                        let (x, y) = (rk[m], w[k - c0 + m]);
                        rk[m] = c * x + s * y;
                        w[k - c0 + m] = -s * x + c * y;
                    }
                    let (x, y) = (*rb, beta);
                    *rb = c * x + s * y;
                    beta = -s * x + c * y;
                    w[k - c0] = 0.0;
                    k += 1;
                }
            }
        }
    }
    // back substitution with a relative pivot tolerance
    let rmax = r.iter().flatten().map(|(v, _)| v[0].abs()).fold(0.0, f64::max);
    let tol = 20.0 * (ncols as f64) * f64::EPSILON * rmax;
    let mut x = vec![0.0; ncols];
    for k in (0..ncols).rev() {
        if let Some((rk, rb)) = &r[k] {
            if rk[0].abs() <= tol {
                continue;
            }
            let mut s = *rb;
            for (m, v) in rk.iter().enumerate().skip(1) {
                s -= v * x[k + m];
            }
            x[k] = s / rk[0];
        }
    }
    x
}

/// Port of `lmmcp.m`.
pub fn lmmcp(fun: &mut McpFn<'_>, x0: &[f64], lb: &[f64], ub: &[f64], o: &LmmcpOptions) -> LmmcpResult {
    let eps1 = o.epsilon1;
    let eps2 = 0.5 * o.tol_fun * o.tol_fun;
    let null = o.null;
    let kmax = o.max_iter;
    let l1 = o.lambda1;
    let l2 = 1.0 - l1;
    let n = x0.len();
    let mut delta = o.delta;
    let mut k = 0usize;
    let mut k_main = 0usize;
    let mut x: Vec<f64> = (0..n).map(|i| lb[i].max(x0[i].min(ub[i]))).collect();
    let ix = index_set(lb, ub, o.big);
    let (mut fx, jac) = fun(&x, true);
    let mut dfx = jac.expect("lmmcp: the Jacobian is required");
    let mut phix = phi(&x, &fx, lb, ub, l1, l2, &ix);
    let mut norm_phix = norm2(&phix);
    let mut psix = 0.5 * phix.iter().map(|v| v * v).sum::<f64>();
    let mut dph = dphi(&x, &fx, &dfx, lb, ub, l1, l2, &ix);
    let mut dpsix = dph.tmul(&dfx, &phix);
    let mut norm_dpsix = norm2(&dpsix);
    let (x_init, fx_init, dfx_init, phix0, psix0, dpsix0, norm_dpsix0) =
        (x.clone(), fx.clone(), dfx.clone(), phix.clone(), psix, dpsix.clone(), norm_dpsix);
    let mut aux = vec![0.0; o.m];
    aux[0] = psix;
    let mut max_psi = psix;
    let v = o.verbosity;
    if v > 1 {
        println!("   k               Psi(x)                || DPsi(x) ||    stepsize");
        println!("====================================================================");
        println!("********************* Output at starting point *********************");
        println!("{:4} {:24.5e} {:24.5e}", k, psix, norm_dpsix);
    }
    if o.preprocess {
        if v > 1 {
            println!("************************** Preprocessor ****************************");
        }
        let mut norm_plm = 1.0;
        while k < o.presteps && psix > eps2 && norm_plm > null {
            k += 1;
            let plm = dph.lm_step(&dfx, &phix, 1e-6 / (k + 1) as f64);
            norm_plm = norm2(&plm);
            let d: Vec<f64> = (0..n)
                .map(|i| {
                    let lbn = (lb[i] - x[i]).min(0.0).max(-delta);
                    let ubn = (ub[i] - x[i]).max(0.0).min(delta);
                    lbn.max(plm[i].min(ubn))
                })
                .collect();
            let xnew: Vec<f64> = (0..n).map(|i| x[i] + d[i]).collect();
            let (fxnew, jnew) = fun(&xnew, true);
            let phixnew = phi(&xnew, &fxnew, lb, ub, l1, l2, &ix);
            let psixnew = 0.5 * phixnew.iter().map(|v| v * v).sum::<f64>();
            let norm_phixnew = norm2(&phixnew);
            if norm_phixnew <= o.eta * norm_phix {
                delta = o.deltamin.max(o.sigma2 * delta);
            } else if norm_phixnew > 5.0 * o.eta * norm_phix {
                delta = o.deltamin.max(o.sigma1 * delta);
            }
            x = xnew;
            fx = fxnew;
            dfx = jnew.expect("lmmcp: the Jacobian is required");
            phix = phixnew;
            psix = psixnew;
            norm_phix = norm_phixnew;
            dph = dphi(&x, &fx, &dfx, lb, ub, l1, l2, &ix);
            dpsix = dph.tmul(&dfx, &phix);
            norm_dpsix = norm_inf(&dpsix);
            if v > 1 {
                println!("{:4} {:24.5e} {:24.5e} {:11.7}", k, psix, norm_dpsix, 1.0);
            }
        }
    }
    if o.preprocess && psix < eps2 {
        if v > 0 {
            println!("Psix = {:.4e}\nnormDPsix = {:.4e}", psix, norm_dpsix);
            println!("Approximate solution found.");
        }
        return LmmcpResult { x, fval: fx, exitflag: 1, iterations: k, psix, norm_dpsix };
    } else if o.preprocess {
        // restart from the initial point (MATLAB keeps Fx/DFx of the last preprocessor step
        // here, but recomputes DFx before using it; Fx is only returned)
        x = x_init;
        fx = fx_init;
        dfx = dfx_init;
        phix = phix0;
        psix = psix0;
        dph = dphi(&x, &fx, &dfx, lb, ub, l1, l2, &ix);
        dpsix = dpsix0;
        let _ = norm_dpsix0;
        if v > 1 {
            println!("******************** Restart with initial point ********************");
            println!("{:4} {:24.5e} {:24.5e}", k_main, psix0, norm_dpsix0);
        }
    }
    if v > 1 {
        println!("************************** Main program ****************************");
    }
    // watchdog (initialized before the preprocessor in lmmcp.m: k = 0 at the initial point,
    // which is where the main loop restarts)
    let mut kbest = 0;
    let mut xbest = x.clone();
    let mut phibest = phix.clone();
    let mut psibest = psix;
    let mut dfxbest = dfx.clone();
    let mut fxbest = fx.clone();
    let mut dpsibest = dpsix.clone();
    let mut norm_dpsibest = norm_dpsix;
    while k < kmax && psix > eps2 {
        let d = dph.lm_step(&dfx, &phix, 1e-1 / (k + 1) as f64);
        let mut t = 1.0;
        let mut xnew: Vec<f64> = (0..n).map(|i| x[i] + d[i]).collect();
        let (mut fxnew, _) = fun(&xnew, false);
        let mut phixnew = phi(&xnew, &fxnew, lb, ub, l1, l2, &ix);
        let mut psixnew = 0.5 * phixnew.iter().map(|v| v * v).sum::<f64>();
        let cst = o.sigma * dpsix.iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
        while psixnew > max_psi + cst * t && t > o.tmin {
            t *= o.beta;
            xnew = (0..n).map(|i| x[i] + t * d[i]).collect();
            fxnew = fun(&xnew, false).0;
            phixnew = phi(&xnew, &fxnew, lb, ub, l1, l2, &ix);
            psixnew = 0.5 * phixnew.iter().map(|v| v * v).sum::<f64>();
        }
        x = xnew;
        fx = fxnew;
        phix = phixnew;
        psix = psixnew;
        dfx = fun(&x, true).1.expect("lmmcp: the Jacobian is required");
        dph = dphi(&x, &fx, &dfx, lb, ub, l1, l2, &ix);
        dpsix = dph.tmul(&dfx, &phix);
        norm_dpsix = norm2(&dpsix);
        k += 1;
        k_main += 1;
        aux[k_main % o.m] = psix;
        max_psi = if k_main <= 5 { psix } else { aux.iter().copied().fold(f64::NEG_INFINITY, f64::max) };
        if o.watchdog {
            if psix < psibest {
                kbest = k;
                xbest = x.clone();
                phibest = phix.clone();
                psibest = psix;
                dfxbest = dfx.clone();
                fxbest = fx.clone();
                dpsibest = dpsix.clone();
                norm_dpsibest = norm_dpsix;
            } else if k - kbest > o.kwatch {
                x = xbest.clone();
                phix = phibest.clone();
                psix = psibest;
                // MATLAB restores DPhix (computed at xbest); Fx/DFx are only used through it
                dfx = dfxbest.clone();
                fx = fxbest.clone();
                dph = dphi(&x, &fx, &dfx, lb, ub, l1, l2, &ix);
                dpsix = dpsibest.clone();
                norm_dpsix = norm_dpsibest;
                max_psi = psix;
            }
        }
        if v > 1 {
            println!("{:4} {:24.5e} {:24.5e} {:11.7}", k, psix, norm_dpsix, t);
        }
    }
    let exitflag = if psix <= eps2 {
        if v > 0 {
            println!("Approximate solution found.");
        }
        1
    } else if k >= kmax {
        if v > 0 {
            println!("Maximum iteration number reached.");
        }
        0
    } else {
        if v > 0 {
            println!(
                "{}",
                if norm_dpsix <= eps1 { "Approximate stationary point found." } else { "No solution found." }
            );
        }
        -1
    };
    LmmcpResult { x, fval: fx, exitflag, iterations: k, psix, norm_dpsix }
}

/// `dynare_solve`'s post-processing for `solve_algo = 10`: residuals of the equations whose
/// variable sits at a finite bound are set to 0 (`eq_to_ignore`).
pub fn ignore_bound_residuals(x: &[f64], fvec: &mut [f64], lb: &[f64], ub: &[f64]) {
    for i in 0..x.len() {
        if (lb[i].is_finite() || ub[i].is_finite()) && (x[i] <= lb[i] + f64::EPSILON || x[i] >= ub[i] - f64::EPSILON) {
            fvec[i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn givens_least_squares() {
        // 12 x 5 overdetermined banded system
        let mut rows = Vec::new();
        let mut dense = Mat::zeros(12, 5);
        let mut b = vec![0.0; 12];
        for i in 0..12usize {
            let mut r = Vec::new();
            for j in 0..5 {
                let v = ((i * 7 + j * 3) % 11) as f64 - 5.0;
                if (i / 2).abs_diff(j) <= 1 && v != 0.0 {
                    r.push((j, v));
                    dense[(i, j)] = v;
                }
            }
            b[i] = (i as f64 * 0.3).cos();
            rows.push((r, b[i]));
        }
        let x = lstsq_givens(rows, 5);
        // normal equations
        let ata = dense.transpose() * &dense;
        let atb = dense.transpose() * crate::linalg::Vector::from_column_slice(&b);
        let xr = ata.lu().solve(&atb).unwrap();
        for j in 0..5 {
            assert!((x[j] - xr[j]).abs() < 1e-10, "{j}: {} vs {}", x[j], xr[j]);
        }
    }

    /// `DPhi` against central differences of `Phi` at a generic point (all index sets).
    #[test]
    fn dphi_is_the_jacobian_of_phi() {
        let fun = |x: &[f64]| -> Vec<f64> {
            vec![x[0] * x[1] - 0.3, x[1] + x[2] * x[2] - 1.1, x[0] - x[2] + 0.2, x[3].exp() - 2.0 + x[0]]
        };
        let jac = |x: &[f64]| -> Csr {
            Csr::from_dense(&Mat::from_row_slice(
                4,
                4,
                &[
                    x[1],
                    x[0],
                    0.0,
                    0.0, //
                    0.0,
                    1.0,
                    2.0 * x[2],
                    0.0, //
                    1.0,
                    0.0,
                    -1.0,
                    0.0, //
                    1.0,
                    0.0,
                    0.0,
                    x[3].exp(),
                ],
            ))
        };
        let lb = [0.1, f64::NEG_INFINITY, -1.0, f64::NEG_INFINITY];
        let ub = [f64::INFINITY, 2.0, 1.5, f64::INFINITY];
        let ix = index_set(&lb, &ub, 1e10);
        for x in [[0.7, 0.4, 0.3, 0.2], [0.15, 1.9, -0.9, 0.5], [0.4, 1.0, 1.2, -0.3]] {
            let fx = fun(&x);
            let d = dphi(&x, &fx, &jac(&x), &lb, &ub, 0.1, 0.9, &ix);
            let j = jac(&x).to_dense();
            for k in 0..4 {
                let h = 1e-6;
                let (mut xp, mut xm) = (x, x);
                xp[k] += h;
                xm[k] -= h;
                let pp = phi(&xp, &fun(&xp), &lb, &ub, 0.1, 0.9, &ix);
                let pm = phi(&xm, &fun(&xm), &lb, &ub, 0.1, 0.9, &ix);
                for i in 0..4 {
                    let h1 = d.a1[i] * j[(i, k)] + if i == k { d.b1[i] } else { 0.0 };
                    let h2 = d.a2[i] * j[(i, k)] + if i == k { d.b2[i] } else { 0.0 };
                    let n1 = (pp[i] - pm[i]) / (2.0 * h);
                    let n2 = (pp[4 + i] - pm[4 + i]) / (2.0 * h);
                    assert!((h1 - n1).abs() < 1e-6, "x={x:?} row {i} col {k}: {h1} vs {n1}");
                    assert!((h2 - n2).abs() < 1e-6, "x={x:?} row {} col {k}: {h2} vs {n2}", 4 + i);
                }
            }
        }
    }

    /// x >= 0 complementary to F(x) = x - 1 (interior solution) and to F(x) = x + 1 (bound).
    #[test]
    fn simple_mcp() {
        let mut f = |x: &[f64], j: bool| {
            let fx = vec![x[0] - 1.0, x[1] + 1.0, x[2] * x[2] - 4.0];
            let jac = j.then(|| Csr::from_triplets(3, 3, vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 2.0 * x[2])]));
            (fx, jac)
        };
        let r = lmmcp(
            &mut f,
            &[0.5, 0.5, 1.0],
            &[0.0, 0.0, f64::NEG_INFINITY],
            &[f64::INFINITY; 3],
            &LmmcpOptions::default(),
        );
        assert_eq!(r.exitflag, 1);
        assert!((r.x[0] - 1.0).abs() < 1e-7);
        assert!(r.x[1].abs() < 1e-7);
        assert!((r.x[2] - 2.0).abs() < 1e-7);
    }
}
