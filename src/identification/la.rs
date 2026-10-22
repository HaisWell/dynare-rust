// SPDX-License-Identifier: GPL-3.0-or-later
//! Small dense linear-algebra helpers with MATLAB semantics (`svd(X,0)`, `rank`, `cond`,
//! `eps(x)`, `inv`, `\` for rectangular systems) used by the identification checks.

use std::os::raw::{c_char, c_int};

use crate::linalg::{Mat, lapack};

/// MATLAB `eps(x)`: the spacing of doubles at |x|.
pub fn eps_of(x: f64) -> f64 {
    let x = x.abs();
    if x == 0.0 {
        return f64::from_bits(1); // smallest subnormal, MATLAB eps(0)
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    let e = x.log2().floor() as i32;
    // guard against log2 rounding at exact powers of two
    let e = if 2f64.powi(e) > x {
        e - 1
    } else if 2f64.powi(e + 1) <= x {
        e + 1
    } else {
        e
    };
    2f64.powi(e - 52)
}

/// Economy SVD `[U, S, V] = svd(A, 0)` (LAPACK dgesvd): U m x k, s (descending), V n x k,
/// k = min(m, n). (MATLAB returns V n x n when m < n; callers only use its first k columns
/// except [`svd_v_full`].)
pub fn svd_econ(a: &Mat) -> (Mat, Vec<f64>, Mat) {
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);
    if k == 0 {
        return (Mat::zeros(m, 0), vec![], Mat::zeros(n, 0));
    }
    let mut f = a.clone();
    let mut s = vec![0.0; k];
    let mut u = Mat::zeros(m, k);
    let mut vt = Mat::zeros(k, n);
    let mut info: c_int = 0;
    let (mi, ni, ki) = (m as c_int, n as c_int, k as c_int);
    let mut wq = 0.0;
    let job = b'S' as c_char;
    unsafe {
        lapack::dgesvd_(
            &job,
            &job,
            &mi,
            &ni,
            f.as_mut_ptr(),
            &mi,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &mi,
            vt.as_mut_ptr(),
            &ki,
            &mut wq,
            &-1,
            &mut info,
        )
    };
    let lwork = (wq as usize).max(5 * (m + n)).max(1);
    let mut work = vec![0.0; lwork];
    unsafe {
        lapack::dgesvd_(
            &job,
            &job,
            &mi,
            &ni,
            f.as_mut_ptr(),
            &mi,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &mi,
            vt.as_mut_ptr(),
            &ki,
            work.as_mut_ptr(),
            &(lwork as c_int),
            &mut info,
        )
    };
    (u, s, vt.transpose())
}

/// The full right singular basis V (n x n) of `svd(A, 0)` (needed when m < n, where MATLAB's
/// economy SVD still returns all n columns of V).
pub fn svd_v_full(a: &Mat) -> (Vec<f64>, Mat) {
    let (m, n) = (a.nrows(), a.ncols());
    if m >= n {
        let (_, s, v) = svd_econ(a);
        return (s, v);
    }
    let k = m;
    let mut f = a.clone();
    let mut s = vec![0.0; k];
    let mut vt = Mat::zeros(n, n);
    let mut info: c_int = 0;
    let (mi, ni) = (m as c_int, n as c_int);
    let one: c_int = 1;
    let mut wq = 0.0;
    unsafe {
        lapack::dgesvd_(
            &(b'N' as c_char),
            &(b'A' as c_char),
            &mi,
            &ni,
            f.as_mut_ptr(),
            &mi.max(1),
            s.as_mut_ptr(),
            std::ptr::null_mut(),
            &one,
            vt.as_mut_ptr(),
            &ni,
            &mut wq,
            &-1,
            &mut info,
        )
    };
    let lwork = (wq as usize).max(5 * (m + n)).max(1);
    let mut work = vec![0.0; lwork];
    unsafe {
        lapack::dgesvd_(
            &(b'N' as c_char),
            &(b'A' as c_char),
            &mi,
            &ni,
            f.as_mut_ptr(),
            &mi.max(1),
            s.as_mut_ptr(),
            std::ptr::null_mut(),
            &one,
            vt.as_mut_ptr(),
            &ni,
            work.as_mut_ptr(),
            &(lwork as c_int),
            &mut info,
        )
    };
    (s, vt.transpose())
}

pub fn singular_values(a: &Mat) -> Vec<f64> {
    if a.nrows().min(a.ncols()) == 0 {
        return vec![];
    }
    crate::linalg::singular_values(a).iter().copied().collect()
}

/// MATLAB `rank(A, tol)`; `tol = None`: `max(size(A)) * eps(norm(A))`.
pub fn rank(a: &Mat, tol: Option<f64>) -> usize {
    let s = singular_values(a);
    if s.is_empty() {
        return 0;
    }
    let tol = tol.unwrap_or_else(|| a.nrows().max(a.ncols()) as f64 * eps_of(s[0]));
    s.iter().filter(|&&v| v > tol).count()
}

/// MATLAB `cond(A)` (2-norm): largest over smallest singular value.
pub fn cond(a: &Mat) -> f64 {
    let s = singular_values(a);
    if s.is_empty() {
        return 0.0;
    }
    let smin = *s.last().unwrap();
    if smin == 0.0 { f64::INFINITY } else { s[0] / smin }
}

/// MATLAB `inv(A)`: LU inverse; an exactly singular matrix gives Inf entries (with MATLAB's
/// warning).
pub fn inv(a: &Mat) -> Mat {
    let n = a.nrows();
    crate::linalg::solve(a, &Mat::identity(n, n)).unwrap_or_else(|| Mat::from_element(n, n, f64::INFINITY))
}

/// MATLAB `A \ B` for square A (LU; an exactly singular A gives Inf).
pub fn solve(a: &Mat, b: &Mat) -> Mat {
    crate::linalg::solve(a, b).unwrap_or_else(|| Mat::from_element(a.ncols(), b.ncols(), f64::INFINITY))
}

/// Least-squares fit `X * (X \ y)` (MATLAB's QR-based `\` for a rectangular X; the fitted
/// values do not depend on which basic solution a rank-deficient X gives).
pub fn ls_fit(x: &Mat, y: &[f64]) -> Option<Vec<f64>> {
    let (m, n) = (x.nrows(), x.ncols());
    if n == 0 {
        return Some(vec![0.0; m]);
    }
    let (u, s, _) = svd_econ(x);
    if s.is_empty() || !s[0].is_finite() {
        return None;
    }
    let tol = m.max(n) as f64 * eps_of(s[0]);
    let mut fit = vec![0.0; m];
    for (k, &sk) in s.iter().enumerate() {
        if sk > tol {
            let c: f64 = (0..m).map(|i| u[(i, k)] * y[i]).sum();
            for i in 0..m {
                fit[i] += u[(i, k)] * c;
            }
        }
    }
    Some(fit)
}

/// Position of (i, j) in `find(tril(ones(n)))` (column-wise lower triangle, as in
/// `duplication.m` and `dvechSig` of get_jacobians.m).
pub fn tril_index(n: usize, i: usize, j: usize) -> usize {
    let (i, j) = if i >= j { (i, j) } else { (j, i) };
    j * n - j * j.saturating_sub(1) / 2 + (i - j)
}

/// Position of (i, j) in `dyn_vech` of a symmetric matrix: `dyn_vech` stacks the upper
/// triangle column by column, (1,1), (1,2), (2,2), (1,3), ...
pub fn vech_index(i: usize, j: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    j * (j + 1) / 2 + i
}

/// Inverse of [`vech_index`] as `display.m` reads it (`find(dyn_unvech(1:n)==k)`, first hit):
/// (larger index, smaller index).
pub fn vech_pair(k: usize) -> (usize, usize) {
    let mut j = 0;
    while (j + 1) * (j + 2) / 2 <= k {
        j += 1;
    }
    (j, k - j * (j + 1) / 2)
}
