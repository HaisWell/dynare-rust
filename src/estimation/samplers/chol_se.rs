// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `estimation/chol_SE.m` (generalized Cholesky of Schnabel and
//! Eskow 1990/1999, without pivoting as TaRB calls it: `chol_SE(A, 0)`).

use anyhow::{Result, bail};

use crate::linalg::Mat;

/// `gersh_nested(A, j, n)`: Gerschgorin bounds of the trailing submatrix
/// (lower triangle of `a` holds the working matrix). 0-based `j`.
fn gersh_nested(a: &Mat, j: usize, n: usize) -> Vec<f64> {
    let mut g = vec![0.0; n];
    for ii in j..n {
        let up: f64 = if ii == 0 { 0.0 } else { (j..ii).map(|k| a[(ii, k)].abs()).sum() };
        let after: f64 = ((ii + 1)..n).map(|k| a[(k, ii)].abs()).sum();
        g[ii] = up + after - a[(ii, ii)];
    }
    g
}

/// `[R, indef] = chol_SE(A, 0)`: upper triangular `R` with
/// `R'*R = A + diag(E)`, `E >= 0` (zero when `A` is safely positive definite).
pub fn chol_se(a0: &Mat) -> Result<(Mat, bool)> {
    let n = a0.nrows();
    let asym: f64 =
        (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (a0[(i, j)] - a0[(j, i)]).abs()).sum();
    if asym > 1e-8 {
        bail!("chol_SE: A is not symmetric");
    }
    let mut a = if asym > 0.0 { (a0 + a0.transpose()) * 0.5 } else { a0.clone() };
    let tau1 = f64::EPSILON.powf(1.0 / 3.0);
    let tau2 = tau1;
    let mut phase1 = true;
    let mut delta: f64 = 0.0;
    let mut g = vec![0.0; n];
    let gammma = (0..n).map(|i| a[(i, i)]).fold(f64::NEG_INFINITY, f64::max);
    if (0..n).any(|i| a[(i, i)] < 0.0) {
        phase1 = false;
    }
    let taugam = tau1 * gammma;
    if !phase1 {
        g = gersh_nested(&a, 0, n);
    }
    if n == 1 {
        let mut e = 0.0;
        let d = tau2 * a[(0, 0)].abs() - a[(0, 0)];
        if d > 0.0 {
            e = d;
        }
        if a[(0, 0)] == 0.0 {
            e = tau2;
        }
        a[(0, 0)] = (a[(0, 0)] + e).sqrt();
    }
    for j in 0..n.saturating_sub(1) {
        if phase1 {
            let jp1 = j + 1;
            if a[(j, j)] > 0.0 {
                let m = (jp1..n).map(|i| a[(i, i)] - a[(i, j)].powi(2) / a[(j, j)]).fold(f64::INFINITY, f64::min);
                if m < taugam {
                    phase1 = false;
                }
            } else {
                phase1 = false;
            }
            if phase1 {
                a[(j, j)] = a[(j, j)].sqrt();
                let tjj = a[(j, j)];
                for ii in jp1..n {
                    a[(ii, j)] /= tjj;
                }
                for ii in jp1..n {
                    let temp = a[(ii, j)];
                    for k in jp1..=ii {
                        a[(ii, k)] -= temp * a[(k, j)];
                    }
                }
                if j == n - 2 {
                    a[(n - 1, n - 1)] = a[(n - 1, n - 1)].sqrt();
                }
            } else {
                g = gersh_nested(&a, j, n);
            }
        }
        if !phase1 {
            if j != n - 2 {
                let normj: f64 = ((j + 1)..n).map(|ii| a[(ii, j)].abs()).sum();
                delta =
                    [0.0, delta, -a[(j, j)] + normj, -a[(j, j)] + taugam].into_iter().fold(f64::NEG_INFINITY, f64::max);
                a[(j, j)] += delta;
                if a[(j, j)] != normj {
                    let temp = normj / a[(j, j)] - 1.0;
                    for ii in (j + 1)..n {
                        g[ii] += a[(ii, j)].abs() * temp;
                    }
                }
                a[(j, j)] = a[(j, j)].sqrt();
                let tjj = a[(j, j)];
                for ii in (j + 1)..n {
                    a[(ii, j)] /= tjj;
                }
                for ii in (j + 1)..n {
                    let temp = a[(ii, j)];
                    for k in (j + 1)..=ii {
                        a[(ii, k)] -= temp * a[(k, j)];
                    }
                }
            } else {
                let (p, q) = (n - 2, n - 1);
                a[(p, q)] = a[(q, p)];
                // eig of the symmetric 2x2 block
                let (x, y, z) = (a[(p, p)], a[(q, p)], a[(q, q)]);
                let mid = 0.5 * (x + z);
                let rad = (0.25 * (x - z).powi(2) + y * y).sqrt();
                let (emin, emax) = (mid - rad, mid + rad);
                delta = [0.0, delta, -emin + tau2 * ((emax - emin) / (1.0 - tau1)).max(gammma)]
                    .into_iter()
                    .fold(f64::NEG_INFINITY, f64::max);
                if delta > 0.0 {
                    a[(p, p)] += delta;
                    a[(q, q)] += delta;
                }
                a[(p, p)] = a[(p, p)].sqrt();
                a[(q, p)] /= a[(p, p)];
                a[(q, q)] -= a[(q, p)].powi(2);
                a[(q, q)] = a[(q, q)].sqrt();
            }
        }
    }
    let _ = g;
    // R = tril(A)'
    let mut r = Mat::zeros(n, n);
    for j in 0..n {
        for i in j..n {
            r[(j, i)] = a[(i, j)];
        }
    }
    Ok((r, !phase1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: &[&[f64]]) -> Mat {
        Mat::from_fn(rows.len(), rows.len(), |i, j| rows[i][j])
    }

    #[test]
    fn dynare_unit_tests() {
        let a = m(&[&[4.0, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);
        let (r, indef) = chol_se(&a).unwrap();
        assert!(!indef);
        assert!((r.transpose() * &r - &a).amax() < 1e-12);
        for (a, pd) in [
            (m(&[&[1.0, 1.0, 0.0], &[1.0, 1.0, 1.0], &[0.0, 1.0, 1.0]]), false),
            (m(&[&[1.0, 2.0], &[2.0, 1.0]]), false),
            (-m(&[&[4.0, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]), false),
            (
                m(&[
                    &[2.0, -1.0, 0.0, 0.0, 0.0],
                    &[-1.0, 2.0, -1.0, 0.0, 0.0],
                    &[0.0, -1.0, 0.0, -1.0, 0.0],
                    &[0.0, 0.0, -1.0, 2.0, -1.0],
                    &[0.0, 0.0, 0.0, -1.0, 2.0],
                ]),
                false,
            ),
        ] {
            let (r, indef) = chol_se(&a).unwrap();
            assert!(indef != pd);
            let rr = r.transpose() * &r;
            // R'R - A is diagonal and nonnegative
            let d = &rr - &a;
            for i in 0..a.nrows() {
                for j in 0..a.nrows() {
                    if i != j {
                        assert!(d[(i, j)].abs() < 1e-10, "{d}");
                    } else {
                        assert!(d[(i, i)] >= -1e-10);
                    }
                }
            }
            let (ev, _) = crate::linalg::sym_eig(&rr);
            assert!(ev[0] > 0.0);
        }
        let (r, _) = chol_se(&m(&[&[0.0]])).unwrap();
        assert!((r[(0, 0)] - f64::EPSILON.powf(1.0 / 3.0).sqrt()).abs() < 1e-14);
        assert!(chol_se(&m(&[&[1.0, 2.0], &[3.0, 4.0]])).is_err());
    }
}
