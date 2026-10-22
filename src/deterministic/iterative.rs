// SPDX-License-Identifier: GPL-3.0-or-later
//! Krylov solvers used by Dynare's perfect-foresight and steady-state code through
//! MATLAB built-ins: `gmres` (left preconditioning, restarted, Householder-free modified
//! Gram–Schmidt Arnoldi with Givens rotations), `bicgstab` (van der Vorst, preconditioner
//! applied to the search directions as MATLAB does, convergence checked at half and full
//! iterations) and `cgs` (Sonneveld's conjugate gradient squared, for Pardiso's
//! `solve_cgs` of `stack_solve_algo = 10`). The algorithms are the textbook ones that MATLAB
//! documents; iterates agree with MATLAB's up to rounding, the converged Newton steps to the
//! tolerance.
//!
//! Return values follow MATLAB: `flag` 0 converged, 1 maximum iterations, 2 ill-conditioned
//! preconditioner, 3 stagnation, 4 breakdown; `relres` the relative residual.

use super::spmat::norm2;

/// Result of an iterative solve.
#[derive(Clone, Debug)]
pub struct IterResult {
    pub x: Vec<f64>,
    pub flag: i32,
    pub relres: f64,
    pub iter: usize,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn resid(a: &dyn Fn(&[f64]) -> Vec<f64>, b: &[f64], x: &[f64]) -> Vec<f64> {
    let ax = a(x);
    b.iter().zip(&ax).map(|(u, v)| u - v).collect()
}

/// `gmres(A, b, restart, tol, maxit, M)` with `x0` (zero if `None`); `restart = None` is
/// MATLAB's `[]` (no restarts, at most `min(n, 10)` iterations unless `maxit` says otherwise:
/// the caller passes the MATLAB defaults).
#[allow(clippy::too_many_arguments)]
pub fn gmres(
    a: &dyn Fn(&[f64]) -> Vec<f64>,
    m: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
    b: &[f64],
    restart: usize,
    tol: f64,
    maxit: usize,
    x0: Option<&[f64]>,
) -> IterResult {
    let n = b.len();
    let prec = |v: &[f64]| -> Vec<f64> { m.map_or_else(|| v.to_vec(), |f| f(v)) };
    let mut x = x0.map_or_else(|| vec![0.0; n], |v| v.to_vec());
    let nb = norm2(b);
    if nb == 0.0 {
        return IterResult { x: vec![0.0; n], flag: 0, relres: 0.0, iter: 0 };
    }
    let mb = prec(b);
    let n2mb = norm2(&mb);
    let target = tol * n2mb;
    let restart = restart.max(1).min(n.max(1));
    let mut total = 0;
    let r0 = resid(a, b, &x);
    let mut relres = norm2(&prec(&r0)) / n2mb;
    if norm2(&r0) <= tol * nb {
        return IterResult { x, flag: 0, relres: norm2(&r0) / nb, iter: 0 };
    }
    for _outer in 0..maxit {
        let r = prec(&resid(a, b, &x));
        let beta = norm2(&r);
        if !beta.is_finite() {
            return IterResult { x, flag: 2, relres, iter: total };
        }
        let mut v: Vec<Vec<f64>> = vec![r.iter().map(|z| z / beta).collect()];
        let mut h = vec![vec![0.0; restart]; restart + 1];
        let (mut cs, mut sn) = (vec![0.0; restart], vec![0.0; restart]);
        let mut g = vec![0.0; restart + 1];
        g[0] = beta;
        let mut k_done = 0;
        let mut converged = false;
        for k in 0..restart {
            total += 1;
            let mut w = prec(&a(&v[k]));
            for i in 0..=k {
                h[i][k] = dot(&w, &v[i]);
                for (wj, vj) in w.iter_mut().zip(&v[i]) {
                    *wj -= h[i][k] * vj;
                }
            }
            h[k + 1][k] = norm2(&w);
            for i in 0..k {
                let t = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
                h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
                h[i][k] = t;
            }
            let den = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            if den == 0.0 {
                k_done = k;
                break;
            }
            cs[k] = h[k][k] / den;
            sn[k] = h[k + 1][k] / den;
            let hk1 = h[k + 1][k];
            h[k][k] = cs[k] * h[k][k] + sn[k] * hk1;
            h[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] *= cs[k];
            k_done = k + 1;
            if g[k + 1].abs() <= target {
                converged = true;
                break;
            }
            let nw = norm2(&w);
            if nw == 0.0 {
                break;
            }
            v.push(w.iter().map(|z| z / nw).collect());
        }
        // y = H \ g, x += V y
        let kk = k_done;
        let mut y = vec![0.0; kk];
        for i in (0..kk).rev() {
            let mut s = g[i];
            for j in i + 1..kk {
                s -= h[i][j] * y[j];
            }
            y[i] = s / h[i][i];
        }
        for (j, yj) in y.iter().enumerate() {
            for (xi, vi) in x.iter_mut().zip(&v[j]) {
                *xi += yj * vi;
            }
        }
        let rt = resid(a, b, &x);
        relres = norm2(&prec(&rt)) / n2mb;
        if converged || relres <= tol {
            return IterResult { x, flag: 0, relres, iter: total };
        }
        if kk == 0 {
            return IterResult { x, flag: 3, relres, iter: total };
        }
    }
    IterResult { x, flag: 1, relres, iter: total }
}

/// `bicgstab(A, b, tol, maxit, M, [], x0)`.
pub fn bicgstab(
    a: &dyn Fn(&[f64]) -> Vec<f64>,
    m: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
    b: &[f64],
    tol: f64,
    maxit: usize,
    x0: Option<&[f64]>,
) -> IterResult {
    let n = b.len();
    let prec = |v: &[f64]| -> Vec<f64> { m.map_or_else(|| v.to_vec(), |f| f(v)) };
    let nb = norm2(b);
    if nb == 0.0 {
        return IterResult { x: vec![0.0; n], flag: 0, relres: 0.0, iter: 0 };
    }
    let mut x = x0.map_or_else(|| vec![0.0; n], |v| v.to_vec());
    let mut r = resid(a, b, &x);
    let mut normr = norm2(&r);
    if normr <= tol * nb {
        return IterResult { x, flag: 0, relres: normr / nb, iter: 0 };
    }
    let rt = r.clone();
    let (mut rho, mut alpha, mut omega) = (1.0, 1.0, 1.0);
    let mut p = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut best = (x.clone(), normr);
    for it in 1..=maxit {
        let rho1 = rho;
        rho = dot(&rt, &r);
        if rho == 0.0 || !rho.is_finite() {
            return IterResult { x: best.0, flag: 4, relres: best.1 / nb, iter: it };
        }
        if it == 1 {
            p = r.clone();
        } else {
            let beta = (rho / rho1) * (alpha / omega);
            if beta == 0.0 || !beta.is_finite() {
                return IterResult { x: best.0, flag: 4, relres: best.1 / nb, iter: it };
            }
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }
        let ph = prec(&p);
        if ph.iter().any(|z| !z.is_finite()) {
            return IterResult { x: best.0, flag: 2, relres: best.1 / nb, iter: it };
        }
        v = a(&ph);
        let rtv = dot(&rt, &v);
        if rtv == 0.0 || !rtv.is_finite() {
            return IterResult { x: best.0, flag: 4, relres: best.1 / nb, iter: it };
        }
        alpha = rho / rtv;
        let xh: Vec<f64> = x.iter().zip(&ph).map(|(xi, pi)| xi + alpha * pi).collect();
        let s: Vec<f64> = r.iter().zip(&v).map(|(ri, vi)| ri - alpha * vi).collect();
        let norms = norm2(&resid(a, b, &xh));
        if norms <= tol * nb {
            return IterResult { x: xh, flag: 0, relres: norms / nb, iter: it };
        }
        if norms < best.1 {
            best = (xh.clone(), norms);
        }
        let sh = prec(&s);
        let t = a(&sh);
        let tt = dot(&t, &t);
        if tt == 0.0 || !tt.is_finite() {
            return IterResult { x: best.0, flag: 4, relres: best.1 / nb, iter: it };
        }
        omega = dot(&t, &s) / tt;
        if omega == 0.0 {
            return IterResult { x: best.0, flag: 4, relres: best.1 / nb, iter: it };
        }
        x = xh.iter().zip(&sh).map(|(xi, si)| xi + omega * si).collect();
        r = s.iter().zip(&t).map(|(si, ti)| si - omega * ti).collect();
        normr = norm2(&resid(a, b, &x));
        if normr <= tol * nb {
            return IterResult { x, flag: 0, relres: normr / nb, iter: it };
        }
        if normr < best.1 {
            best = (x.clone(), normr);
        }
    }
    IterResult { x: best.0, flag: 1, relres: best.1 / nb, iter: maxit }
}

/// `cgs(A, b, tol, maxit, M)` (preconditioned conjugate gradient squared).
pub fn cgs(
    a: &dyn Fn(&[f64]) -> Vec<f64>,
    m: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
    b: &[f64],
    tol: f64,
    maxit: usize,
) -> IterResult {
    let n = b.len();
    let prec = |v: &[f64]| -> Vec<f64> { m.map_or_else(|| v.to_vec(), |f| f(v)) };
    let nb = norm2(b);
    if nb == 0.0 {
        return IterResult { x: vec![0.0; n], flag: 0, relres: 0.0, iter: 0 };
    }
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let rt = r.clone();
    let mut rho = 1.0;
    let (mut p, mut q) = (vec![0.0; n], vec![0.0; n]);
    let mut normr = nb;
    for it in 1..=maxit {
        let rho1 = rho;
        rho = dot(&rt, &r);
        if rho == 0.0 || !rho.is_finite() {
            return IterResult { x, flag: 4, relres: normr / nb, iter: it };
        }
        let u: Vec<f64>;
        if it == 1 {
            u = r.clone();
            p = u.clone();
        } else {
            let beta = rho / rho1;
            u = r.iter().zip(&q).map(|(ri, qi)| ri + beta * qi).collect();
            p = (0..n).map(|i| u[i] + beta * (q[i] + beta * p[i])).collect();
        }
        let ph = prec(&p);
        let vh = a(&ph);
        let rtv = dot(&rt, &vh);
        if rtv == 0.0 || !rtv.is_finite() {
            return IterResult { x, flag: 4, relres: normr / nb, iter: it };
        }
        let alpha = rho / rtv;
        q = u.iter().zip(&vh).map(|(ui, vi)| ui - alpha * vi).collect();
        let uq: Vec<f64> = u.iter().zip(&q).map(|(ui, qi)| ui + qi).collect();
        let uh = prec(&uq);
        for i in 0..n {
            x[i] += alpha * uh[i];
        }
        let qh = a(&uh);
        for i in 0..n {
            r[i] -= alpha * qh[i];
        }
        normr = norm2(&resid(a, b, &x));
        if normr <= tol * nb {
            return IterResult { x, flag: 0, relres: normr / nb, iter: it };
        }
    }
    IterResult { x, flag: 1, relres: normr / nb, iter: maxit }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn system() -> (Vec<Vec<f64>>, Vec<f64>) {
        let n: usize = 30;
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            4.0 + i as f64 * 0.1
                        } else if i.abs_diff(j) == 1 {
                            -1.0
                        } else if j == i + 3 {
                            0.3
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        (a, b)
    }

    #[test]
    fn krylov_solvers_converge() {
        let (a, b) = system();
        let mv = |x: &[f64]| -> Vec<f64> { a.iter().map(|r| dot(r, x)).collect() };
        let jac = |x: &[f64]| -> Vec<f64> { x.iter().enumerate().map(|(i, v)| v / a[i][i]).collect() };
        for r in [
            gmres(&mv, None, &b, 30, 1e-10, 1, None),
            gmres(&mv, Some(&jac), &b, 5, 1e-10, 50, None),
            bicgstab(&mv, None, &b, 1e-10, 100, None),
            bicgstab(&mv, Some(&jac), &b, 1e-10, 100, None),
            cgs(&mv, Some(&jac), &b, 1e-10, 100),
        ] {
            assert_eq!(r.flag, 0, "{r:?}");
            let res = resid(&mv, &b, &r.x);
            assert!(norm2(&res) <= 1e-9 * norm2(&b), "{}", norm2(&res));
        }
    }
}
