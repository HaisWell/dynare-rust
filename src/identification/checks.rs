// SPDX-License-Identifier: GPL-3.0-or-later
//! Rank and collinearity checks: ports of `+identification/checks.m`, `cosn.m`, `vnorm.m` and
//! `bruteforce.m` (ident_bruteforce).

use crate::linalg::Mat;

use super::la::{cond, eps_of, inv, ls_fit, rank, svd_v_full};

/// Output of `identification.checks`.
#[derive(Clone, Debug, Default)]
pub struct CheckResult {
    pub cond: f64,
    pub rank: usize,
    /// `ind0`: parameters with a non-zero Jacobian column (not singularly unidentified).
    pub ind0: Vec<bool>,
    /// `indno`: one row per problematic parameter set.
    pub indno: Vec<Vec<bool>>,
    /// `ino` (`ixno`): number of problematic sets.
    pub ino: usize,
    /// Multicorrelation coefficients.
    pub mco: Vec<f64>,
    /// Pairwise correlation (cosn) coefficients, NaN where undefined.
    pub pco: Mat,
    pub jweak: Vec<bool>,
    /// `dyn_vech` of the weak-pair matrix.
    pub jweak_pair: Vec<bool>,
}

/// `vnorm(A)`: Euclidean norms of the columns.
pub fn vnorm(a: &Mat) -> Vec<f64> {
    (0..a.ncols()).map(|j| a.column(j).iter().map(|v| v * v).sum::<f64>().sqrt()).collect()
}

/// `cosn(H)`: cosine between `H(:,1)` and its least-squares fit on the other columns.
pub fn cosn(y: &[f64], x: &Mat) -> f64 {
    let fit = match ls_fit(x, y) {
        Some(f) if f.iter().all(|v| v.is_finite()) => f,
        _ => return 0.0,
    };
    let nf: f64 = fit.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nf == 0.0 || nf <= fit.len() as f64 * eps_of(nf) {
        return 0.0;
    }
    let yy: f64 = y.iter().map(|v| v * v).sum();
    let yf: f64 = y.iter().zip(&fit).map(|(a, b)| a * b).sum();
    let ff: f64 = fit.iter().map(|v| v * v).sum();
    (yf / (yy * ff).sqrt()).abs()
}

fn hcat(cols: &[&Mat]) -> Mat {
    // empty blocks (no Xrest) do not count: Qu-Tkachenko's square Xparnonzero has fewer rows
    let m = cols
        .iter()
        .filter(|c| c.ncols() > 0)
        .map(|c| c.nrows())
        .max()
        .unwrap_or_else(|| cols.first().map_or(0, |c| c.nrows()));
    let n: usize = cols.iter().map(|c| c.ncols()).sum();
    let mut out = Mat::zeros(m, n);
    let mut k = 0;
    for c in cols {
        for j in 0..c.ncols() {
            out.set_column(k, &c.column(j));
            k += 1;
        }
    }
    out
}

fn select_cols(a: &Mat, cols: &[usize]) -> Mat {
    Mat::from_fn(a.nrows(), cols.len(), |i, j| a[(i, cols[j])])
}

fn select(a: &Mat, rows: &[usize], cols: &[usize]) -> Mat {
    Mat::from_fn(rows.len(), cols.len(), |i, j| a[(rows[i], cols[j])])
}

/// Port of `identification.checks(X, test_flag, tol_rank, tol_sv, param_nbr)`.
/// `test_flag`: 0 information matrix (Gram), 1 Jacobian, 2 Komunjer–Ng (first `param_nbr`
/// columns are parameters), 3 Qu–Tkachenko G (Gram). `tol_rank = None` is `'robust'`.
pub fn checks(x: &Mat, test_flag: u8, tol_rank: Option<f64>, tol_sv: f64, param_nbr: usize) -> CheckResult {
    let tol_rank = tol_rank.unwrap_or_else(|| {
        let s = super::la::singular_values(x);
        x.nrows().max(x.ncols()) as f64 * eps_of(s.first().copied().unwrap_or(0.0))
    });
    let rankrequired = x.ncols();
    let (xpar, xrest) = if test_flag == 2 {
        (x.columns(0, param_nbr).into_owned(), x.columns(param_nbr, x.ncols() - param_nbr).into_owned())
    } else {
        (x.clone(), Mat::zeros(x.nrows(), 0))
    };
    let norms: Vec<f64> =
        if xpar.nrows() > 1 { vnorm(&xpar) } else { (0..xpar.ncols()).map(|j| xpar[(0, j)].abs()).collect() };
    let mut ind1: Vec<usize> = (0..xpar.ncols()).filter(|&j| norms[j] >= f64::EPSILON).collect();
    let nonzero =
        |ind1: &[usize]| -> Mat { if test_flag == 3 { select(&xpar, ind1, ind1) } else { select_cols(&xpar, ind1) } };
    let mut xpnz = nonzero(&ind1);
    let mut full = hcat(&[&xpnz, &xrest]);
    let (_, mut ee1) = svd_v_full(&full);
    let mut cond_x = cond(&full);
    let rank_x = rank(x, Some(tol_rank));
    let mut icheck = false;
    if param_nbr > 0 && rank_x < rankrequired {
        let mut ind11 = vec![true; ind1.len()];
        for j in rank_x..full.ncols() {
            let col: Vec<f64> = ee1.column(j).iter().map(|v| v.abs()).collect();
            if col.iter().filter(|&&v| v > tol_sv).count() == 1 {
                icheck = true;
                for (k, f) in ind11.iter_mut().enumerate() {
                    *f &= col[k] <= tol_sv;
                }
            }
        }
        // Dynare's test above depends on the basis LAPACK picks inside a degenerate null space
        // (e.g. two parameters with numerically zero columns: MATLAB happens to return the unit
        // vectors, another LAPACK a rotation of them). Also flag every parameter whose unit
        // vector lies in the null space (row norm of the null basis ~ 1), which is
        // basis-independent and agrees with Dynare whenever its basis is aligned.
        let nnull = full.ncols().saturating_sub(rank_x);
        if nnull > 0 {
            for (k, f) in ind11.iter_mut().enumerate() {
                let r2: f64 = (rank_x..full.ncols()).map(|j| ee1[(k, j)] * ee1[(k, j)]).sum();
                if r2 >= 1.0 - tol_sv * tol_sv {
                    icheck = true;
                    *f = false;
                }
            }
        }
        ind1 = ind1.iter().zip(&ind11).filter(|(_, f)| **f).map(|(i, _)| *i).collect();
    }
    if icheck {
        xpnz = nonzero(&ind1);
        full = hcat(&[&xpnz, &xrest]);
        ee1 = svd_v_full(&full).1;
        cond_x = cond(&full);
    }
    let mut ind0 = vec![false; param_nbr];
    for &i in &ind1 {
        ind0[i] = true;
    }
    let mut mco;
    let mut pco = Mat::from_element(param_nbr, param_nbr, f64::NAN);
    if test_flag == 0 || test_flag == 3 {
        mco = vec![if test_flag == 3 { f64::NAN } else { 0.0 }; param_nbr];
        let xs = select(x, &ind1, &ind1);
        let delta: Vec<f64> = (0..ind1.len()).map(|i| xs[(i, i)].sqrt()).collect();
        let tilda = Mat::from_fn(ind1.len(), ind1.len(), |i, j| xs[(i, j)] / (delta[i] * delta[j]));
        let it = inv(&tilda);
        for (k, &i) in ind1.iter().enumerate() {
            mco[i] = 1.0 - 1.0 / it[(k, k)];
        }
        let ix = inv(&xs);
        for (a, &i) in ind1.iter().enumerate() {
            for (b, &j) in ind1.iter().enumerate() {
                pco[(i, j)] = ix[(a, b)];
            }
        }
        let sd: Vec<f64> = (0..param_nbr).map(|i| pco[(i, i)].sqrt()).collect();
        for i in 0..param_nbr {
            for j in 0..param_nbr {
                pco[(i, j)] = (pco[(i, j)] / (sd[i] * sd[j])).abs();
            }
        }
    } else {
        mco = vec![f64::NAN; param_nbr];
        for ii in 0..xpnz.ncols() {
            let y: Vec<f64> = xpnz.column(ii).iter().copied().collect();
            let others: Vec<usize> = (0..xpnz.ncols()).filter(|&k| k != ii).collect();
            let xo = hcat(&[&select_cols(&xpnz, &others), &xrest]);
            mco[ind1[ii]] = cosn(&y, &xo);
        }
    }
    let mut indno: Vec<Vec<bool>> = Vec::new();
    let min_1_mco = mco.iter().filter(|v| !v.is_nan()).map(|v| 1.0 - v).fold(f64::INFINITY, f64::min);
    if param_nbr > 0 && (rank_x < rankrequired || min_1_mco < tol_rank) {
        if ind1.len() < param_nbr {
            indno.push((0..param_nbr).map(|i| !ind1.contains(&i)).collect());
        }
        for j in rank_x..full.ncols() {
            let mut row = vec![false; param_nbr];
            for (k, &i) in ind1.iter().enumerate() {
                row[i] = ee1[(k, j)].abs() > tol_sv;
            }
            indno.push(row);
        }
    }
    let ino = indno.len();
    let mut jweak = vec![false; param_nbr];
    let mut wp = vec![vec![false; param_nbr]; param_nbr];
    if test_flag != 0 {
        pco = Mat::from_element(param_nbr, param_nbr, f64::NAN);
        for ii in 0..xpnz.ncols() {
            pco[(ind1[ii], ind1[ii])] = 1.0;
            let y: Vec<f64> = xpnz.column(ii).iter().copied().collect();
            for jj in ii + 1..xpnz.ncols() {
                let xo = hcat(&[&select_cols(&xpnz, &[jj]), &xrest]);
                let c = cosn(&y, &xo);
                pco[(ind1[ii], ind1[jj])] = c;
                pco[(ind1[jj], ind1[ii])] = c;
            }
        }
        for j in 0..param_nbr {
            if mco[j] > 1.0 - tol_rank {
                jweak[j] = true;
                for k in j + 1..param_nbr {
                    if pco[(j, k)] > 1.0 - tol_rank {
                        wp[j][k] = true;
                        wp[k][j] = true;
                    }
                }
            }
        }
    }
    let mut jweak_pair = Vec::with_capacity(param_nbr * (param_nbr + 1) / 2);
    for j in 0..param_nbr {
        for i in 0..=j {
            jweak_pair.push(wp[i][j]);
        }
    }
    CheckResult { cond: cond_x, rank: rank_x, ind0, indno, ino, mco, pco, jweak, jweak_pair }
}

/// All `k`-subsets of `items`, in `nchoosek` order (lexicographic).
pub fn nchoosek(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    let n = items.len();
    let mut out = Vec::new();
    if k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.iter().map(|&i| items[i]).collect());
        let Some(i) = (0..k).rev().find(|&i| idx[i] < i + n - k) else {
            return out;
        };
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Port of `identification.bruteforce`: for each parameter and each group size `ll`, the
/// maximum cosn with any `ll` other parameters (`cosnJ`) and the group attaining it (`pars`,
/// `None` = `NaN`).
pub fn bruteforce(j: &Mat, max_dim_cova_group: usize, tol_deriv: f64) -> (Vec<Vec<Vec<Option<usize>>>>, Mat) {
    let np = j.ncols();
    let mut cosnj = Mat::zeros(np, max_dim_cova_group);
    let mut pars: Vec<Vec<Vec<Option<usize>>>> = vec![vec![Vec::new(); max_dim_cova_group]; np];
    for ll in 1..=max_dim_cova_group {
        for ii in 0..np {
            let tmp: Vec<usize> = (0..np).filter(|&k| k != ii).collect();
            let combos = nchoosek(&tmp, ll);
            let y: Vec<f64> = j.column(ii).iter().copied().collect();
            let vals: Vec<f64> = combos.iter().map(|c| cosn(&y, &select_cols(j, c))).collect();
            let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            cosnj[(ii, ll - 1)] = mx;
            if mx > tol_deriv {
                if ll > 1 && (mx - cosnj[(ii, ll - 2)]) < tol_deriv {
                    let mut p = pars[ii][ll - 2].clone();
                    p.push(None);
                    pars[ii][ll - 1] = p;
                    cosnj[(ii, ll - 1)] = cosnj[(ii, ll - 2)];
                } else {
                    // first maximum (MATLAB: `== max`); exact ties such as the two members of a
                    // non-identified pair differ by rounding only, so compare with a tolerance
                    let k = vals.iter().position(|&v| v >= mx - 1e-12 * mx.abs().max(1.0)).unwrap_or(0);
                    pars[ii][ll - 1] = combos[k].iter().map(|&c| Some(c)).collect();
                }
            } else {
                pars[ii][ll - 1] = vec![None; ll];
            }
        }
    }
    (pars, cosnj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nchoosek_order() {
        assert_eq!(
            nchoosek(&[1, 2, 3, 4], 2),
            vec![vec![1, 2], vec![1, 3], vec![1, 4], vec![2, 3], vec![2, 4], vec![3, 4]]
        );
        assert_eq!(nchoosek(&[5, 7], 1), vec![vec![5], vec![7]]);
        assert_eq!(nchoosek(&[5, 7], 2), vec![vec![5, 7]]);
    }

    #[test]
    fn checks_detects_collinear_pair() {
        // columns 1 and 2 proportional
        let x = Mat::from_row_slice(4, 3, &[1.0, 2.0, 0.0, 2.0, 4.0, 1.0, 3.0, 6.0, 0.5, -1.0, -2.0, 2.0]);
        let r = checks(&x, 1, None, 1e-3, 3);
        assert_eq!(r.rank, 2);
        assert_eq!(r.ino, 1);
        assert_eq!(r.indno[0], vec![true, true, false]);
        assert!(r.jweak_pair[1]); // (2,1)
        assert!((r.pco[(0, 1)] - 1.0).abs() < 1e-12);
    }
}
