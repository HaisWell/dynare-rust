// SPDX-License-Identifier: GPL-3.0-or-later
//! The generalized Sylvester equation of the k-order perturbation,
//! `Y + S Y (C ⊗ ... ⊗ C) = F` (`r` factors), as Dynare's `gensylv` (Kamenik 2005).
//!
//! Instead of Kamenik's real quasi-triangular recursion we use the complex Schur form
//! `C = U T U^H`: with `Z = Y U^{⊗r}` the equation becomes `Z + S Z T^{⊗r} = F U^{⊗r}`, and
//! `T^{⊗r}` is upper triangular, so the column blocks of `Z` (first Kronecker index) are
//! solved in order, each one a Sylvester equation with `r - 1` factors; at `r = 0` it is the
//! linear system `(I + s S) Z = F`.

use super::tensor::kron_power_apply;
use crate::linalg::Mat;
use nalgebra::{Complex, DMatrix};

type CMat = DMatrix<Complex<f64>>;

fn to_c(a: &Mat) -> CMat {
    a.map(|v| Complex::new(v, 0.0))
}

/// Solves `Y + S Y C^{⊗r} = F` (`S`: p x p, `C`: m x m, `F`: p x m^r).
pub fn gensylv(r: usize, s: &Mat, c: &Mat, f: &Mat) -> anyhow::Result<Mat> {
    let p = s.nrows();
    if r == 0 || c.nrows() == 0 {
        let a = Mat::identity(p, p) + s;
        return a.lu().solve(f).ok_or_else(|| anyhow::anyhow!("gensylv: singular (I + S)"));
    }
    let m = c.nrows();
    let schur = nalgebra::linalg::Schur::try_new(to_c(c), 1e-15 * (1.0 + c.abs().max()), 10_000)
        .ok_or_else(|| anyhow::anyhow!("gensylv: Schur decomposition did not converge"))?;
    let (u, t) = schur.unpack();
    let fc = kron_power_apply(&to_c(f), &u, r);
    let sc = to_c(s);
    let z = solve_rec(&sc, &t, m, r, Complex::new(1.0, 0.0), &fc)?;
    let uh = u.adjoint();
    let y = kron_power_apply(&z, &uh, r);
    Ok(y.map(|v| v.re))
}

/// `Z + scale S Z T^{⊗r} = F` with `T` upper triangular.
fn solve_rec(s: &CMat, t: &CMat, m: usize, r: usize, scale: Complex<f64>, f: &CMat) -> anyhow::Result<CMat> {
    let p = s.nrows();
    if r == 0 {
        let a = CMat::identity(p, p) + s * scale;
        return a.lu().solve(f).ok_or_else(|| anyhow::anyhow!("gensylv: singular system"));
    }
    let bs = f.ncols() / m;
    let mut z = CMat::zeros(p, f.ncols());
    for j in 0..m {
        let mut rhs = f.columns(j * bs, bs).into_owned();
        if j > 0 {
            let mut g = CMat::zeros(p, bs);
            for i in 0..j {
                let tij = t[(i, j)];
                if tij != Complex::new(0.0, 0.0) {
                    g += z.columns(i * bs, bs) * tij;
                }
            }
            let gw = kron_power_apply(&g, t, r - 1);
            rhs -= s * gw * scale;
        }
        let zj = solve_rec(s, t, m, r - 1, scale * t[(j, j)], &rhs)?;
        z.columns_mut(j * bs, bs).copy_from(&zj);
    }
    Ok(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gensylv_solves_the_equation() {
        let s = Mat::from_fn(3, 3, |i, j| 0.3 * (i as f64 - j as f64) + if i == j { 0.5 } else { 0.1 });
        let c = Mat::from_row_slice(2, 2, &[0.9, -0.4, 0.3, 0.7]); // complex eigenvalues
        for r in 0..=3 {
            let f = Mat::from_fn(3, 2usize.pow(r as u32), |i, j| (i as f64 + 1.0) * 0.2 - j as f64 * 0.13);
            let y = gensylv(r, &s, &c, &f).unwrap();
            let res = &y + &s * kron_power_apply(&y, &c, r) - &f;
            assert!(res.abs().max() < 1e-13, "r={r}: {}", res.abs().max());
        }
    }
}
