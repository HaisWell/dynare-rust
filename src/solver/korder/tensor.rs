// SPDX-License-Identifier: GPL-3.0-or-later
//! Dense unfolded tensors (MATLAB `kron` column ordering) and the products the
//! k-order solver needs.
//!
//! A k-tensor with `p` rows over index sets of sizes `m1..mk` is a `p x (m1*...*mk)`
//! column-major matrix whose column for the tuple `(i1, ..., ik)` is
//! `((i1*m2 + i2)*m3 + i3)...` (first index slowest), so that `X * kron(a, b)` is the
//! contraction of `X` with `a` on the first index and `b` on the second, as in Dynare's
//! `A_times_B_kronecker_C`.

use nalgebra::{ComplexField, DMatrix};

/// Multiplies the index `mode` of the tensor `x` (`p` rows, index sizes `dims`) by `a`
/// (`dims[mode] x r`): `y[.., j, ..] = sum_i x[.., i, ..] a[i, j]`. Returns the new data.
fn mode_product<T: ComplexField + Copy>(x: &[T], p: usize, dims: &[usize], mode: usize, a: &DMatrix<T>) -> Vec<T> {
    let m = dims[mode];
    assert_eq!(a.nrows(), m);
    let r = a.ncols();
    let inner = p * dims[mode + 1..].iter().product::<usize>();
    let outer: usize = dims[..mode].iter().product();
    let mut y = vec![T::zero(); outer * r * inner];
    for o in 0..outer {
        let xo = &x[o * m * inner..(o + 1) * m * inner];
        let yo = &mut y[o * r * inner..(o + 1) * r * inner];
        for j in 0..r {
            let yj = &mut yo[j * inner..(j + 1) * inner];
            for i in 0..m {
                let aij = a[(i, j)];
                if aij == T::zero() {
                    continue;
                }
                let xi = &xo[i * inner..(i + 1) * inner];
                for (yv, &xv) in yj.iter_mut().zip(xi) {
                    *yv += aij * xv;
                }
            }
        }
    }
    y
}

/// `x * (a1 ⊗ a2 ⊗ ... ⊗ ak)` for `x` with `prod(rows(ai))` columns.
pub fn kron_apply<T: ComplexField + Copy>(x: &DMatrix<T>, factors: &[&DMatrix<T>]) -> DMatrix<T> {
    let p = x.nrows();
    let mut dims: Vec<usize> = factors.iter().map(|a| a.nrows()).collect();
    assert_eq!(dims.iter().product::<usize>(), x.ncols(), "kron_apply: dimension mismatch");
    let mut data = x.as_slice().to_vec();
    for (l, a) in factors.iter().enumerate() {
        data = mode_product(&data, p, &dims, l, a);
        dims[l] = a.ncols();
    }
    DMatrix::from_vec(p, dims.iter().product(), data)
}

/// `x * a^{⊗k}`.
pub fn kron_power_apply<T: ComplexField + Copy>(x: &DMatrix<T>, a: &DMatrix<T>, k: usize) -> DMatrix<T> {
    if k == 0 {
        return x.clone();
    }
    let f: Vec<&DMatrix<T>> = (0..k).map(|_| a).collect();
    kron_apply(x, &f)
}

/// Column index of the tuple `idx` over `k` indices of size `m` each.
#[inline]
pub fn col_of(idx: &[usize], m: usize) -> usize {
    idx.iter().fold(0, |c, &i| c * m + i)
}

/// Decodes a column index into its tuple.
#[inline]
pub fn tuple_of(mut c: usize, m: usize, k: usize, out: &mut [usize]) {
    for l in (0..k).rev() {
        out[l] = c % m;
        c /= m;
    }
}

/// Sum over the 3 cyclic placements of the first index of a 3-tensor `t[a,(b,c)]`
/// (`m` per index): `h[a,b,c] += t[a,b,c] + t[b,a,c] + t[c,a,b]`.
pub fn add_sym3_first(h: &mut DMatrix<f64>, t: &DMatrix<f64>, m: usize) {
    let p = h.nrows();
    let (hs, ts) = (h.as_mut_slice(), t.as_slice());
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                let dst = ((a * m + b) * m + c) * p;
                let s1 = ((a * m + b) * m + c) * p;
                let s2 = ((b * m + a) * m + c) * p;
                let s3 = ((c * m + a) * m + b) * p;
                for r in 0..p {
                    hs[dst + r] += ts[s1 + r] + ts[s2 + r] + ts[s3 + r];
                }
            }
        }
    }
}

/// `h[a,b,c] += sum over the 6 permutations pi of t[pi(a,b,c)]`.
pub fn add_sym6(h: &mut DMatrix<f64>, t: &DMatrix<f64>, m: usize) {
    let p = h.nrows();
    let (hs, ts) = (h.as_mut_slice(), t.as_slice());
    let c3 = |a: usize, b: usize, c: usize| ((a * m + b) * m + c) * p;
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                let dst = c3(a, b, c);
                let src = [c3(a, b, c), c3(a, c, b), c3(b, a, c), c3(b, c, a), c3(c, a, b), c3(c, b, a)];
                for r in 0..p {
                    let mut s = 0.0;
                    for &q in &src {
                        s += ts[q + r];
                    }
                    hs[dst + r] += s;
                }
            }
        }
    }
}

/// `h[a,b] += t[a,b] + t[b,a]`.
pub fn add_sym2(h: &mut DMatrix<f64>, t: &DMatrix<f64>, m: usize) {
    let p = h.nrows();
    let (hs, ts) = (h.as_mut_slice(), t.as_slice());
    for a in 0..m {
        for b in 0..m {
            let (d, s1, s2) = ((a * m + b) * p, (a * m + b) * p, (b * m + a) * p);
            for r in 0..p {
                hs[d + r] += ts[s1 + r] + ts[s2 + r];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kron(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
        a.kronecker(b)
    }

    #[test]
    fn kron_apply_matches_explicit_kron() {
        let x = DMatrix::from_fn(3, 6, |i, j| (i as f64 + 1.0) * 0.3 - j as f64 * 0.17 + (i * j) as f64 * 0.05);
        let a = DMatrix::from_fn(2, 4, |i, j| 0.1 + i as f64 - 0.2 * j as f64);
        let b = DMatrix::from_fn(3, 2, |i, j| (i + 2 * j) as f64 * 0.7 - 1.0);
        let direct = &x * kron(&a, &b);
        let fast = kron_apply(&x, &[&a, &b]);
        assert!((direct - fast).abs().max() < 1e-13);
        let c = DMatrix::from_fn(2, 2, |i, j| 1.0 + i as f64 * 0.5 - j as f64);
        let x3 = DMatrix::from_fn(2, 8, |i, j| (i * 8 + j) as f64 * 0.11 - 0.4);
        let d3 = &x3 * kron(&kron(&c, &c), &c);
        assert!((d3 - kron_power_apply(&x3, &c, 3)).abs().max() < 1e-12);
    }
}
