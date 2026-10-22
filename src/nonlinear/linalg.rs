// SPDX-License-Identifier: GPL-3.0-or-later
//! Small dense helpers with MATLAB semantics used by the nonlinear filters: `chol`,
//! `reduced_rank_cholesky.m`, `qr2.m`, `\` and `/`, `probability2.m`, `neff.m`.

use crate::linalg::{Mat, lu, norm1, rcond_from_lu, sym_eig};

/// MATLAB `[R, p] = chol(A)` (upper triangle of `A` only, as LAPACK `dpotrf('U')`):
/// `Some(R)` with `R'R = A`, or `None` when `A` is not positive definite.
pub fn chol_upper(a: &Mat) -> Option<Mat> {
    let n = a.nrows();
    let mut r = Mat::zeros(n, n);
    for j in 0..n {
        let mut d = a[(j, j)];
        for k in 0..j {
            d -= r[(k, j)] * r[(k, j)];
        }
        if d.is_nan() || d <= 0.0 {
            return None;
        }
        let d = d.sqrt();
        r[(j, j)] = d;
        for i in j + 1..n {
            let mut s = a[(j, i)];
            for k in 0..j {
                s -= r[(k, j)] * r[(k, i)];
            }
            r[(j, i)] = s / d;
        }
    }
    Some(r)
}

/// MATLAB `chol(A)'` (lower factor).
pub fn chol_lower(a: &Mat) -> Option<Mat> {
    chol_upper(a).map(|r| r.transpose())
}

/// MATLAB `eps(x)`.
pub fn eps_of(x: f64) -> f64 {
    let x = x.abs();
    if x == 0.0 || !x.is_finite() {
        return f64::MIN_POSITIVE * f64::EPSILON;
    }
    let e = x.log2().floor() as i32;
    // guard against log2 rounding at powers of two
    let e = if 2f64.powi(e) > x {
        e - 1
    } else if 2f64.powi(e + 1) <= x {
        e + 1
    } else {
        e
    };
    2f64.powi(e - 52)
}

/// Port of `reduced_rank_cholesky.m`: `T` with `T'T = X` (`chol` when `X` is positive
/// definite, else from the eigen-decomposition, keeping the eigenvalues above
/// `sqrt(eps(max(D)) * n * 10)`; `T` is then rank x n). `None` when `X` is not positive
/// semidefinite (Dynare returns NaN).
pub fn reduced_rank_cholesky(x: &Mat) -> Option<Mat> {
    if let Some(t) = chol_upper(x) {
        return Some(t);
    }
    let n = x.nrows();
    let (d, mut u) = sym_eig(x);
    // flip the eigenvectors whose largest (first maximal) |entry| is negative
    for j in 0..n {
        let mut imax = 0;
        let mut vmax = -1.0;
        for i in 0..n {
            if u[(i, j)].abs() > vmax {
                vmax = u[(i, j)].abs();
                imax = i;
            }
        }
        if u[(imax, j)] < 0.0 {
            for i in 0..n {
                u[(i, j)] = -u[(i, j)];
            }
        }
    }
    let dmax = d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tol = (eps_of(dmax) * n as f64 * 10.0).sqrt();
    let keep: Vec<usize> = (0..n).filter(|&i| d[i].abs() > tol).collect();
    if keep.iter().any(|&i| d[i] < 0.0) {
        println!("reduced_rank_cholesky:: Input matrix is not semidefinite positive!");
        return None;
    }
    let mut t = Mat::zeros(keep.len(), n);
    for (r, &i) in keep.iter().enumerate() {
        let s = d[i].sqrt();
        for c in 0..n {
            t[(r, c)] = s * u[(c, i)];
        }
    }
    Some(t)
}

/// `[~, R] = qr2(A, 0)`: the economy R factor with a nonnegative diagonal.
pub fn qr2_r(a: &Mat) -> Mat {
    let mut r = a.clone().qr().r();
    for i in 0..r.nrows().min(r.ncols()) {
        if r[(i, i)] < 0.0 {
            for j in 0..r.ncols() {
                r[(i, j)] = -r[(i, j)];
            }
        }
    }
    r
}

fn is_lower(s: &Mat) -> bool {
    (0..s.ncols()).all(|j| (0..j.min(s.nrows())).all(|i| s[(i, j)] == 0.0))
}

fn is_upper(s: &Mat) -> bool {
    (0..s.ncols()).all(|j| (j + 1..s.nrows()).all(|i| s[(i, j)] == 0.0))
}

/// MATLAB `S \ X` for square `S` (triangular solves when `S` is triangular, else LU).
pub fn mldivide(s: &Mat, x: &Mat) -> Mat {
    let n = s.nrows();
    assert_eq!(n, s.ncols(), "mldivide: square matrices only");
    let mut out = x.clone();
    if is_lower(s) {
        for c in 0..x.ncols() {
            for i in 0..n {
                let mut v = out[(i, c)];
                for k in 0..i {
                    v -= s[(i, k)] * out[(k, c)];
                }
                out[(i, c)] = v / s[(i, i)];
            }
        }
    } else if is_upper(s) {
        for c in 0..x.ncols() {
            for i in (0..n).rev() {
                let mut v = out[(i, c)];
                for k in i + 1..n {
                    v -= s[(i, k)] * out[(k, c)];
                }
                out[(i, c)] = v / s[(i, i)];
            }
        }
    } else {
        let (f, ipiv, _) = lu(s);
        crate::linalg::lu_solve(&f, &ipiv, false, &mut out);
    }
    out
}

/// MATLAB `A / B` (= `(B' \ A')'`).
pub fn mrdivide(a: &Mat, b: &Mat) -> Mat {
    mldivide(&b.transpose(), &a.transpose()).transpose()
}

/// MATLAB `rcond(A)` (1-norm estimate, from the LU factorization).
pub fn rcond(a: &Mat) -> f64 {
    if a.nrows() == 0 {
        return f64::INFINITY;
    }
    let (f, _, info) = lu(a);
    if info > 0 {
        return 0.0;
    }
    rcond_from_lu(&f, norm1(a))
}

/// MATLAB `det(A)` (from the LU factorization).
pub fn det(a: &Mat) -> f64 {
    let n = a.nrows();
    if n == 0 {
        return 1.0;
    }
    let (f, ipiv, _) = lu(a);
    let mut d = 1.0;
    for i in 0..n {
        d *= f[(i, i)];
        if ipiv[i] as usize != i + 1 {
            d = -d;
        }
    }
    d
}

/// Port of `probability2.m`: Gaussian densities `N(mu, S S')` at the columns of `X`
/// (`exp(-0.5*sum(foo.*foo))' ./ abs(normfact*prod(diag(S))) + 1e-99`, `foo = S \ (X - mu)`).
/// As in MATLAB, when `X` has a single row `sum` runs over the particles and every entry
/// gets the same value.
pub fn probability2(mu: &[f64], s: &Mat, x: &Mat) -> Vec<f64> {
    let dim = x.nrows();
    let n = x.ncols();
    let normfact = (2.0 * std::f64::consts::PI).powf(dim as f64 / 2.0);
    let mut d = x.clone();
    for c in 0..n {
        for i in 0..dim {
            d[(i, c)] -= mu[i];
        }
    }
    let foo = mldivide(s, &d);
    let pd: f64 = (0..s.nrows().min(s.ncols())).map(|i| s[(i, i)]).product();
    let den = (normfact * pd).abs();
    if dim == 1 {
        let q: f64 = foo.iter().map(|v| v * v).sum();
        let v = (-0.5 * q).exp() / den + 1e-99;
        return vec![v; n];
    }
    (0..n)
        .map(|c| {
            let q: f64 = (0..dim).map(|i| foo[(i, c)] * foo[(i, c)]).sum();
            (-0.5 * q).exp() / den + 1e-99
        })
        .collect()
}

/// `neff.m`: `1 / dot(w, w)`.
pub fn neff(w: &[f64]) -> f64 {
    1.0 / w.iter().map(|v| v * v).sum::<f64>()
}

/// `A(rows, :)`.
pub fn select_rows(a: &Mat, rows: &[usize]) -> Mat {
    Mat::from_fn(rows.len(), a.ncols(), |i, j| a[(rows[i], j)])
}

/// `A(:, cols)`.
pub fn select_cols(a: &Mat, cols: &[usize]) -> Mat {
    Mat::from_fn(a.nrows(), cols.len(), |i, j| a[(i, cols[j])])
}

/// `M * w` for a column-weight vector (`tmp * weights`).
pub fn weighted_mean(m: &Mat, w: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; m.nrows()];
    for (c, &wc) in w.iter().enumerate() {
        for i in 0..m.nrows() {
            out[i] += m[(i, c)] * wc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chol_and_reduced_rank() {
        let a = Mat::from_row_slice(3, 3, &[4.0, 2.0, 0.4, 2.0, 5.0, 1.0, 0.4, 1.0, 3.0]);
        let r = chol_upper(&a).unwrap();
        assert!((r.transpose() * &r - &a).abs().max() < 1e-14);
        // rank 2
        let v = Mat::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 0.0, 2.0]);
        let x = &v * v.transpose();
        assert!(chol_upper(&x).is_none());
        let t = reduced_rank_cholesky(&x).unwrap();
        assert_eq!(t.nrows(), 2);
        assert!((t.transpose() * &t - &x).abs().max() < 1e-12);
    }

    #[test]
    fn qr2_positive_diagonal() {
        let a = Mat::from_row_slice(4, 2, &[1.0, -2.0, 3.0, 1.0, -1.0, 0.5, 2.0, 2.0]);
        let r = qr2_r(&a);
        assert!(r[(0, 0)] > 0.0 && r[(1, 1)] > 0.0);
        assert!((r.transpose() * &r - a.transpose() * &a).abs().max() < 1e-12);
    }

    #[test]
    fn eps_matches_matlab() {
        assert_eq!(eps_of(1.0), f64::EPSILON);
        assert_eq!(eps_of(2.0), 2.0 * f64::EPSILON);
        assert_eq!(eps_of(1.5), f64::EPSILON);
    }
}
