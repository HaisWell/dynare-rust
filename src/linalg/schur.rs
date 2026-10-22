// SPDX-License-Identifier: GPL-3.0-or-later
//! Real Schur decomposition with ordering (MATLAB `schur` / `ordschur` / `ordeig`).
//! Owned by the Kalman work (used by `compute_Pinf_Pstar.m` and `lyapunov_symm.m`).
//!
//! `schur` calls LAPACK `dgees` without sorting (as MATLAB's `schur`), `ordschur` calls
//! `dtrsen` (as MATLAB's `ordschur`).

use super::{Mat, lapack};
use std::os::raw::{c_char, c_int};

/// A = U T U' with U orthogonal and T quasi-upper-triangular (1x1 and standardised 2x2 blocks).
#[derive(Clone, Debug)]
pub struct RealSchur {
    pub t: Mat,
    pub u: Mat,
    /// Real and imaginary parts of the eigenvalues, in the order of the diagonal of `t`
    /// (MATLAB `ordeig(T)`).
    pub wr: Vec<f64>,
    pub wi: Vec<f64>,
}

impl RealSchur {
    /// |eigenvalues| in diagonal order: `abs(ordeig(T))`.
    pub fn eig_abs(&self) -> Vec<f64> {
        self.wr.iter().zip(&self.wi).map(|(r, i)| r.hypot(*i)).collect()
    }
}

/// MATLAB `[U, T] = schur(A)` for a real square matrix (LAPACK dgees, no sorting).
pub fn schur(a: &Mat) -> anyhow::Result<RealSchur> {
    assert!(a.is_square(), "schur: matrix must be square");
    let n = a.nrows();
    let mut t = a.clone();
    let mut u = Mat::zeros(n, n);
    let mut wr = vec![0.0; n];
    let mut wi = vec![0.0; n];
    if n == 0 {
        return Ok(RealSchur { t, u, wr, wi });
    }
    let ni = n as c_int;
    let mut sdim: c_int = 0;
    let mut info: c_int = 0;
    let mut bwork = vec![0 as c_int; n];
    let jobvs = b'V' as c_char;
    let sort = b'N' as c_char;
    // workspace query
    let mut wq = 0.0;
    let lq: c_int = -1;
    unsafe {
        lapack::dgees_(
            &jobvs,
            &sort,
            None,
            &ni,
            t.as_mut_ptr(),
            &ni,
            &mut sdim,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            u.as_mut_ptr(),
            &ni,
            &mut wq,
            &lq,
            bwork.as_mut_ptr(),
            &mut info,
        )
    };
    let lwork = (wq as usize).max(3 * n).max(1);
    let mut work = vec![0.0; lwork];
    unsafe {
        lapack::dgees_(
            &jobvs,
            &sort,
            None,
            &ni,
            t.as_mut_ptr(),
            &ni,
            &mut sdim,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            u.as_mut_ptr(),
            &ni,
            work.as_mut_ptr(),
            &(lwork as c_int),
            bwork.as_mut_ptr(),
            &mut info,
        )
    };
    anyhow::ensure!(info == 0, "schur: dgees failed (info = {info})");
    Ok(RealSchur { t, u, wr, wi })
}

/// MATLAB `[U, T] = ordschur(U, T, select)`: moves the eigenvalues with `select[k]` true
/// (k in the current diagonal order) to the leading block (LAPACK dtrsen). For a complex
/// pair, selecting either member selects the pair (as LAPACK; MATLAB's callers here always
/// select pairs consistently since they test |eigenvalue|).
pub fn ordschur(s: &mut RealSchur, select: &[bool]) -> anyhow::Result<()> {
    let n = s.t.nrows();
    assert_eq!(select.len(), n, "ordschur: select length");
    if n == 0 {
        return Ok(());
    }
    let sel: Vec<c_int> = select.iter().map(|&b| b as c_int).collect();
    let ni = n as c_int;
    let mut m: c_int = 0;
    let (mut sc, mut sep) = (0.0, 0.0);
    let lwork = n.max(1) as c_int;
    let mut work = vec![0.0; n.max(1)];
    let liwork: c_int = 1;
    let mut iwork = [0 as c_int; 1];
    let mut info: c_int = 0;
    unsafe {
        lapack::dtrsen_(
            &(b'N' as c_char),
            &(b'V' as c_char),
            sel.as_ptr(),
            &ni,
            s.t.as_mut_ptr(),
            &ni,
            s.u.as_mut_ptr(),
            &ni,
            s.wr.as_mut_ptr(),
            s.wi.as_mut_ptr(),
            &mut m,
            &mut sc,
            &mut sep,
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &liwork,
            &mut info,
        )
    };
    anyhow::ensure!(info == 0, "ordschur: dtrsen failed (info = {info}, ill-conditioned reordering)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schur_and_reorder() {
        // eigenvalues 0.5, 1.0, complex pair 0.3 +- 0.4i, 1.2
        let a = Mat::from_row_slice(
            5,
            5,
            &[
                0.5, 1.0, 0.2, 0.0, 0.3, //
                0.0, 1.0, 0.1, 0.4, 0.0, //
                0.0, 0.0, 0.3, 0.4, 0.2, //
                0.0, 0.0, -0.4, 0.3, 0.1, //
                0.0, 0.0, 0.0, 0.0, 1.2,
            ],
        );
        let p = Mat::from_fn(5, 5, |i, j| ((i * 7 + j * 3) % 5) as f64 + if i == j { 3.0 } else { 0.0 });
        let pi = p.clone().try_inverse().unwrap();
        let b = &p * &a * &pi;
        let mut s = schur(&b).unwrap();
        let rec = &s.u * &s.t * s.u.transpose();
        assert!((rec - &b).amax() < 1e-12);
        let sel: Vec<bool> = s.eig_abs().iter().map(|&x| x > 1.0 - 1e-6).collect();
        ordschur(&mut s, &sel).unwrap();
        let rec = &s.u * &s.t * s.u.transpose();
        assert!((rec - &b).amax() < 1e-11);
        let e = s.eig_abs();
        assert!(e[0] > 0.999 && e[1] > 0.999, "{e:?}");
        assert!(e[2..].iter().all(|&x| x < 0.999));
        for i in 1..5usize {
            for j in 0..i.saturating_sub(1) {
                assert_eq!(s.t[(i, j)], 0.0);
            }
        }
    }
}
