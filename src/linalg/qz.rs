// SPDX-License-Identifier: GPL-3.0-or-later
//! Generalised Schur (QZ) decomposition with eigenvalue ordering, as Dynare's
//! `mjdgges` MEX (`mex/sources/mjdgges/mjdgges.F08`; the MATLAB fallback is
//! `matlab/missing/mex/mjdgges/mjdgges.m`). Owned by the solver work.
//!
//! Semantics reproduced from the MEX (checked against its disassembly):
//! * `dgges('N', 'V', 'S', select, ...)` on the pencil `(E, D)` with
//!   `lwork = 16 n + 16`; only the right Schur vectors `Z` are formed.
//! * `select(alphar, alphai, beta) = alphar^2 + alphai^2 < crit^2 * beta^2`,
//!   i.e. the generalized eigenvalues with modulus strictly below
//!   `qz_criterium` go to the top-left block (infinite ones never do).
//! * eigenvalues `(alphar + i alphai) / beta`; when `alphai == 0` and
//!   `beta == 0` the result is `alphar / 0 + 0i` (an infinite real value).
//! * `info` is dgges' `info`; if it is 0 and some eigenvalue has
//!   `|alphar| <= zhreshold` and `|beta| <= zhreshold` (a 0/0 eigenvalue),
//!   `info = -30`.

use super::{Mat, lapack};
use nalgebra::Complex;
use std::cell::Cell;
use std::os::raw::{c_char, c_int};

thread_local! {
    /// `qz_criterium` seen by the dgges selection callback (dgges calls it on
    /// the calling thread, so a thread-local keeps concurrent solves apart).
    static CRITERIUM: Cell<f64> = const { Cell::new(1.0 + 1e-6) };
}

extern "C" fn select_stable(alphar: *const f64, alphai: *const f64, beta: *const f64) -> c_int {
    // SAFETY: dgges passes valid pointers to scalars.
    let (ar, ai, b) = unsafe { (*alphar, *alphai, *beta) };
    let c = CRITERIUM.with(|c| c.get());
    (ar * ar + ai * ai < c * c * (b * b)) as c_int
}

/// Result of [`mjdgges`]: `Q' E Z = S`, `Q' D Z = T` with `S` quasi upper
/// triangular, `T` upper triangular, the `sdim` stable eigenvalues first.
#[derive(Debug, Clone)]
pub struct Qz {
    pub s: Mat,
    pub t: Mat,
    pub z: Mat,
    /// Left Schur vectors (`Q' E Z = S`); only formed by [`qz_ordered`] with
    /// `want_q`, otherwise 0 x 0 (mjdgges never forms them).
    pub q: Mat,
    /// Number of eigenvalues with modulus `< qz_criterium` (top-left block).
    pub sdim: usize,
    pub alphar: Vec<f64>,
    pub alphai: Vec<f64>,
    pub beta: Vec<f64>,
    /// Generalized eigenvalues in the reordered order, as mjdgges returns them.
    pub eigval: Vec<Complex<f64>>,
    /// dgges info, or -30 for a 0/0 eigenvalue (see module docs).
    pub info: i32,
}

/// Port of Dynare's `mjdgges(e, d, qz_criterium, zhreshold)`.
pub fn mjdgges(e: &Mat, d: &Mat, qz_criterium: f64, zhreshold: f64) -> Qz {
    qz_ordered(e, d, qz_criterium, zhreshold, false)
}

/// [`mjdgges`], optionally also forming the left Schur vectors `Q`
/// (`dgges('V', 'V', 'S', ...)`).
pub fn qz_ordered(e: &Mat, d: &Mat, qz_criterium: f64, zhreshold: f64, want_q: bool) -> Qz {
    qz_impl(e, d, Some(qz_criterium), zhreshold, want_q)
}

/// Generalized real Schur form without eigenvalue ordering
/// (`dgges('V', 'V', 'N', ...)`, `sdim = 0`), `Q` and `Z` formed; the 0/0
/// check of [`mjdgges`] still sets `info = -30`. The ordering is done by
/// `solver::block_schur` with Sylvester equations instead of `dtgsen`
/// swaps, which dominate dgges' cost on pencils with long zero/infinite
/// Jordan chains.
pub fn qz_unordered(e: &Mat, d: &Mat, zhreshold: f64) -> Qz {
    qz_impl(e, d, None, zhreshold, true)
}

fn qz_impl(e: &Mat, d: &Mat, sort: Option<f64>, zhreshold: f64, want_q: bool) -> Qz {
    let n = e.nrows();
    assert!(e.is_square() && d.is_square() && d.nrows() == n, "mjdgges: square matrices of equal size");
    let mut s = e.clone();
    let mut t = d.clone();
    let mut z = Mat::zeros(n, n);
    let mut q = if want_q { Mat::zeros(n, n) } else { Mat::zeros(0, 0) };
    let mut alphar = vec![0.0; n];
    let mut alphai = vec![0.0; n];
    let mut beta = vec![0.0; n];
    let mut info: c_int = 0;
    let mut sdim: c_int = 0;
    if n > 0 {
        let ni = n as c_int;
        let lwork = (16 * n + 16) as c_int;
        let mut work = vec![0.0; lwork as usize];
        let mut bwork = vec![0 as c_int; n];
        let mut vsl = [0.0f64; 1];
        let one: c_int = 1;
        let (jobvsl, vsl_ptr, ldvsl) = if want_q { (b'V', q.as_mut_ptr(), ni) } else { (b'N', vsl.as_mut_ptr(), one) };
        if let Some(c) = sort {
            CRITERIUM.with(|x| x.set(c));
        }
        let (sort_c, select) = match sort {
            Some(_) => (b'S', Some(select_stable as lapack::SelectFn3)),
            None => (b'N', None),
        };
        unsafe {
            lapack::dgges_(
                &(jobvsl as c_char),
                &(b'V' as c_char),
                &(sort_c as c_char),
                select,
                &ni,
                s.as_mut_ptr(),
                &ni,
                t.as_mut_ptr(),
                &ni,
                &mut sdim,
                alphar.as_mut_ptr(),
                alphai.as_mut_ptr(),
                beta.as_mut_ptr(),
                vsl_ptr,
                &ldvsl,
                z.as_mut_ptr(),
                &ni,
                work.as_mut_ptr(),
                &lwork,
                bwork.as_mut_ptr(),
                &mut info,
            );
        }
    }
    let eigval = alphar.iter().zip(&alphai).zip(&beta).map(|((&ar, &ai), &b)| eigenvalue(ar, ai, b)).collect();

    if info == 0 && alphar.iter().zip(&beta).any(|(ar, b)| ar.abs() <= zhreshold && b.abs() <= zhreshold) {
        info = -30;
    }
    Qz { s, t, z, q, sdim: sdim.max(0) as usize, alphar, alphai, beta, eigval, info }
}

/// `complex(alphar, alphai) / beta` as computed by the mjdgges MEX (gfortran
/// complex division by a real-valued complex).
pub fn eigenvalue(ar: f64, ai: f64, b: f64) -> Complex<f64> {
    if ai == 0.0 && b == 0.0 {
        Complex::new(ar / b, 0.0)
    } else {
        // Smith's division by (b + 0i): ratio = 0 / b (NaN when b == 0).
        let r = 0.0 / b;
        let den = r * 0.0 + b;
        Complex::new((ai * r + ar) / den, (ai - r * ar) / den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::mul;

    fn check_decomposition(e: &Mat, d: &Mat, q: &Qz) {
        // Z orthogonal, and E Z = Q S, D Z = Q T for some orthogonal Q: check
        // that (E Z) S^+ and (D Z) T^+ agree through the eigenvalue structure
        // indirectly via the invariant ||E Z||_F = ||S||_F.
        let ztz = mul(&q.z, true, &q.z, false);
        assert!((ztz - Mat::identity(e.nrows(), e.nrows())).amax() < 1e-12);
        let ez = mul(e, false, &q.z, false);
        let dz = mul(d, false, &q.z, false);
        assert!((ez.norm() - q.s.norm()).abs() < 1e-10 * (1.0 + q.s.norm()));
        assert!((dz.norm() - q.t.norm()).abs() < 1e-10 * (1.0 + q.t.norm()));
        // T upper triangular
        for j in 0..q.t.ncols() {
            for i in (j + 1)..q.t.nrows() {
                assert!(q.t[(i, j)].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn diagonal_pencil_orders_stable_first() {
        // eigenvalues 2, 0.5, inf (beta = 0), 0.9
        let e = Mat::from_diagonal(&nalgebra::DVector::from_vec(vec![2.0, 0.5, 1.0, 0.9]));
        let d = Mat::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 1.0, 0.0, 1.0]));
        let q = mjdgges(&e, &d, 1.000001, 1e-6);
        assert_eq!(q.info, 0);
        assert_eq!(q.sdim, 2);
        let mut stable: Vec<f64> = q.eigval[..2].iter().map(|z| z.norm()).collect();
        stable.sort_by(f64::total_cmp);
        assert!((stable[0] - 0.5).abs() < 1e-14 && (stable[1] - 0.9).abs() < 1e-14);
        let unstable: Vec<f64> = q.eigval[2..].iter().map(|z| z.norm()).collect();
        assert!(unstable.iter().any(|&x| x.is_infinite()));
        assert!(unstable.iter().any(|&x| (x - 2.0).abs() < 1e-14));
        check_decomposition(&e, &d, &q);
    }

    #[test]
    fn complex_pair_and_unit_root() {
        // rotation scaled by 0.8 (|lambda| = 0.8, complex pair), a unit root
        // (|lambda| = 1 < 1.000001: stable) and an explosive root 1.5.
        let (c, s) = (0.8 * 0.6, 0.8 * 0.8);
        let e = Mat::from_row_slice(
            4,
            4,
            &[
                c, -s, 0.0, 0.0, //
                s, c, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.5,
            ],
        );
        // a non-trivial similarity so the pencil is not already triangular
        let p = Mat::from_row_slice(
            4,
            4,
            &[1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0],
        );
        let e2 = mul(&p, false, &e, false);
        let d2 = p.clone();
        let q = mjdgges(&e2, &d2, 1.000001, 1e-6);
        assert_eq!(q.info, 0);
        assert_eq!(q.sdim, 3);
        let mods: Vec<f64> = q.eigval.iter().map(|z| z.norm()).collect();
        assert!((mods[0] - 0.8).abs() < 1e-12 && (mods[1] - 0.8).abs() < 1e-12, "{mods:?}");
        assert!((mods[2] - 1.0).abs() < 1e-12);
        assert!((mods[3] - 1.5).abs() < 1e-12);
        assert!(q.eigval[0].im.abs() > 0.1);
        check_decomposition(&e2, &d2, &q);
    }

    #[test]
    fn zero_over_zero_eigenvalue_flags_minus_30() {
        let e = Mat::from_diagonal(&nalgebra::DVector::from_vec(vec![0.0, 0.5]));
        let d = Mat::from_diagonal(&nalgebra::DVector::from_vec(vec![0.0, 1.0]));
        let q = mjdgges(&e, &d, 1.000001, 1e-6);
        assert_eq!(q.info, -30);
    }

    #[test]
    fn dynare_unit_test_pencil() {
        // matlab/missing/mex/mjdgges/mjdgges.m, @test:1 (expects sdim = 5)
        #[rustfmt::skip]
        let e = Mat::from_row_slice(10, 10, &[
            0.,0.,0.,0.,0.,0.,0.,-1.,0.,0., 0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.,
            0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.990099009900990, 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.0990099009900990,
            0.,0.,0.,-1.01010101010101,0.,0.0427672955974843,0.,0.,0.,0., 0.,0.,0.,0.,0.,0.128301886792453,-1.,0.,0.,0.,
            0.8,0.,0.,0.,0.,0.,0.,0.,0.,0., 0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,
            0.,0.,0.9,0.,0.,0.,0.,0.,0.,0., 0.,0.,0.,-1.01010101010101,-1.,0.,2.,0.,-1.,0.]);
        #[rustfmt::skip]
        let d = Mat::from_row_slice(10, 10, &[
            0.,0.,0.,0.,-1.,0.,0.,-0.792,0.,0., 0.,0.,0.,0.,0.,0.,0.,0.,-0.99,0.,
            0.,0.,0.,-0.000493818030899887,0.,0.,0.,0.,0.,-0.882178217821782, 0.,0.,0.,-1.00493818030900,0.,0.,0.,0.,0.,0.0882178217821782,
            0.,0.,0.,-1.,0.128301886792453,0.,0.,0.,0.,0., -1.,0.,0.,0.,0.,0.,-0.99,0.,0.,0.,
            1.,0.,0.,0.,0.,0.,0.,0.,0.,0., 0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,
            0.,0.,1.,0.,0.,0.,0.,0.,0.,0., 0.,0.,0.,0.,-1.,0.,0.,0.,0.,0.]);
        let q = mjdgges(&e, &d, 1.000001, 1e-6);
        assert_eq!(q.info, 0);
        assert_eq!(q.sdim, 5);
        check_decomposition(&e, &d, &q);
    }
}
