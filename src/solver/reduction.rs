// SPDX-License-Identifier: GPL-3.0-or-later
//! the iterative first-order solvers of `dr=cycle_reduction` and
//! `dr=logarithmic_reduction` (ports of `missing/mex/cycle_reduction/cycle_reduction.m` and
//! `missing/mex/logarithmic_reduction/logarithmic_reduction.m`, the MATLAB versions of the
//! MEX files Dynare 7.2 ships). Both solve the quadratic matrix equation
//! `A0 + A1 X + A2 X^2 = 0` for its minimal solvent `X`.

use crate::linalg::{Mat, lu, lu_solve, mul, norm1};

/// `options_.dr_cycle_reduction` / `options_.dr_logarithmic_reduction` with their tolerances
/// (defaults of `default_option_values.m`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FirstOrderMethod {
    /// `mjdgges` (Dynare's default).
    #[default]
    Qz,
    /// `dr=cycle_reduction` (`dr_cycle_reduction_tol` 1e-7, `dr_cycle_reduction_maxiter` 100).
    CycleReduction { tol: f64, maxiter: usize },
    /// `dr=logarithmic_reduction` (`dr_logarithmic_reduction_tol` 1e-12, `..._maxiter` 100).
    LogarithmicReduction { tol: f64, maxiter: usize },
}

/// `B / A` (right division, `B * inv(A)`) via the LU of `A'`.
fn rdiv(b: &Mat, a: &Mat) -> Mat {
    // X A = B  <=>  A' X' = B'
    let (f, p, _) = lu(&a.transpose());
    let mut xt = b.transpose();
    lu_solve(&f, &p, false, &mut xt);
    xt.transpose()
}

/// `A \ B`.
fn ldiv(a: &Mat, b: &Mat) -> Mat {
    let (f, p, _) = lu(a);
    let mut x = b.clone();
    lu_solve(&f, &p, false, &mut x);
    x
}

/// Port of `cycle_reduction.m` (`[X, info] = cycle_reduction(A0, A1, A2, cvg_tol, max_it)`):
/// `Err((401 | 402, info(2)))` when the iterations do not converge / give NaN.
pub fn cycle_reduction(a0: &Mat, a1: &Mat, a2: &Mat, cvg_tol: f64, max_it: usize) -> Result<Mat, (i32, f64)> {
    let n = a0.nrows();
    let a0_0 = a0.clone();
    let (mut a0, mut a1, mut a2) = (a0.clone(), a1.clone(), a2.clone());
    let mut ahat1 = a1.clone();
    let mut it = 0usize;
    loop {
        // tmp = ([A0; A2] / A1) * [A0 A2]
        let mut st = Mat::zeros(2 * n, n);
        st.rows_mut(0, n).copy_from(&a0);
        st.rows_mut(n, n).copy_from(&a2);
        let mut lr = Mat::zeros(n, 2 * n);
        lr.columns_mut(0, n).copy_from(&a0);
        lr.columns_mut(n, n).copy_from(&a2);
        let tmp = mul(&rdiv(&st, &a1), false, &lr, false);
        let t00 = tmp.view((0, 0), (n, n)).into_owned();
        let t02 = tmp.view((0, n), (n, n)).into_owned();
        let t20 = tmp.view((n, 0), (n, n)).into_owned();
        let t22 = tmp.view((n, n), (n, n)).into_owned();
        a1 = &a1 - &t02 - &t20;
        a0 = -t00;
        a2 = -t22;
        ahat1 -= &t20;
        let crit = norm1(&a0);
        if crit < cvg_tol {
            if norm1(&a2) < cvg_tol {
                break;
            }
        } else if it == max_it {
            return Err((401, norm1(&a1).ln()));
        } else if crit.is_nan() {
            return Err((402, norm1(&a1).ln()));
        }
        it += 1;
    }
    // X = -Ahat1 \ A0_0
    let mut x = ldiv(&ahat1, &a0_0);
    x.neg_mut();
    Ok(x)
}

/// Port of `logarithmic_reduction.m` (`[X, info] = logarithmic_reduction(A, B, C, tol, maxit)`,
/// solving `A X^2 + B X + C = 0`): `Err((411 | 412, info(2)))` on failure.
pub fn logarithmic_reduction(a: &Mat, b: &Mat, c: &Mat, tol: f64, maxit: usize) -> Result<Mat, (i32, f64)> {
    let n = a.nrows();
    // tmp0 = -B \ [A, C]
    let mut ac = Mat::zeros(n, 2 * n);
    ac.columns_mut(0, n).copy_from(a);
    ac.columns_mut(n, n).copy_from(c);
    let mut tmp0 = ldiv(b, &ac);
    tmp0.neg_mut();
    let mut x0 = tmp0.columns(n, n).into_owned();
    let mut u0 = tmp0.columns(0, n).into_owned();
    let mut x1 = x0.clone();
    let mut kk = 0usize;
    let mut cc = 1.0f64;
    while cc > tol && kk <= maxit {
        let t_i = tmp0.columns(0, n).into_owned();
        let t_n = tmp0.columns(n, n).into_owned();
        // (eye(n) - tmp0 * [tmp0(:, n+i); tmp0(:, i)]) \ [tmp0(:,i)*tmp0(:,i), tmp0(:,n+i)*tmp0(:,n+i)]
        let mut stack = Mat::zeros(2 * n, n);
        stack.rows_mut(0, n).copy_from(&t_n);
        stack.rows_mut(n, n).copy_from(&t_i);
        let lhs = Mat::identity(n, n) - mul(&tmp0, false, &stack, false);
        let mut rhs = Mat::zeros(n, 2 * n);
        rhs.columns_mut(0, n).copy_from(&mul(&t_i, false, &t_i, false));
        rhs.columns_mut(n, n).copy_from(&mul(&t_n, false, &t_n, false));
        let tmp1 = ldiv(&lhs, &rhs);
        x1 = &x0 + mul(&u0, false, &tmp1.columns(n, n).into_owned(), false);
        let u1 = mul(&u0, false, &tmp1.columns(0, n).into_owned(), false);
        cc = norm1(&(&x1 - &x0));
        if cc.is_nan() {
            return Err((412, 1.0));
        }
        x0 = x1.clone();
        u0 = u1;
        tmp0 = tmp1;
        kk += 1;
    }
    if kk == maxit {
        return Err((411, cc.ln()));
    }
    Ok(x1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The unit test of cycle_reduction.m (n = 50 here): both algorithms agree.
    #[test]
    fn cycle_and_logarithmic_reduction_agree() {
        let n = 50;
        let a = Mat::identity(n, n);
        let mut b = Mat::from_diagonal_element(n, n, 30.0);
        b[(0, 0)] = 20.0;
        b[(n - 1, n - 1)] = 20.0;
        let mut c = Mat::from_diagonal_element(n, n, 15.0);
        for i in 0..n - 1 {
            b[(i + 1, i)] -= 10.0;
            b[(i, i + 1)] -= 10.0;
            c[(i + 1, i)] -= 5.0;
            c[(i, i + 1)] -= 5.0;
        }
        let x1 = cycle_reduction(&c, &b, &a, 1e-7, 100).unwrap();
        let x2 = logarithmic_reduction(&a, &b, &c, 1e-16, 100).unwrap();
        let d = (&x1 - &x2).abs().max();
        assert!(d < 1e-12, "{d}");
        // residual of A X^2 + B X + C
        let r = &a * &x1 * &x1 + &b * &x1 + &c;
        assert!(r.abs().max() < 1e-10);
    }
}
