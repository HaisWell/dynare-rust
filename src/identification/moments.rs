// SPDX-License-Identifier: GPL-3.0-or-later
//! Theoretical first and second moments of the observables and their parameter derivatives
//! at order 1: port of the `order = 1` path of `+pruned_SS/pruned_state_space_system.m`, and
//! of the moment and spectrum parts of `+identification/get_jacobians.m`
//! (`analytic_derivation_mode` 0: Qu–Tkachenko's G from the analytic derivative of the spectral
//! density on `grid_nbr` frequencies).

use anyhow::Result;
use nalgebra::Complex;
use rayon::prelude::*;

use crate::kalman::lyapunov_symm;
use crate::linalg::schur::{ordschur, schur};
use crate::linalg::{Mat, mul};
use crate::solver::DecisionRule;

use super::derivs::PerturbationDerivs;

type C64 = Complex<f64>;
type CMat = nalgebra::DMatrix<C64>;

const LYAPUNOV_COMPLEX_THRESHOLD: f64 = 1e-15;

/// The pruned state space at order 1 (`z = x`: the state variables) with its moments.
#[derive(Clone, Debug)]
pub struct PrunedStateSpace {
    /// DR positions of the states (`indx`) and of the selected variables (`indy`).
    pub indx: Vec<usize>,
    pub indy: Vec<usize>,
    pub a: Mat,
    pub b: Mat,
    pub c: Mat,
    pub d: Mat,
    pub varinov: Mat,
    pub da: Vec<Mat>,
    pub db: Vec<Mat>,
    pub dc: Vec<Mat>,
    pub dd: Vec<Mat>,
    pub dvarinov: Vec<Mat>,
    pub var_z: Mat,
    pub dvar_z: Vec<Mat>,
    /// Selected variables whose variance is finite (`stationary_vars`).
    pub stationary: Vec<bool>,
    pub e_y: Vec<f64>,
    /// y_nbr x tot.
    pub de_y: Mat,
    pub var_y: Mat,
    pub dvar_y: Vec<Mat>,
    /// `Var_yi(:,:,i)` = Cov(y_{t+i}, y_t), i = 1..nlags.
    pub var_yi: Vec<Mat>,
    /// `[lag][param]`.
    pub dvar_yi: Vec<Vec<Mat>>,
    pub corr_yi: Option<Vec<Mat>>,
    pub dcorr_yi: Option<Vec<Vec<Mat>>>,
}

/// `[X, Schur_u] = lyapunov_symm(A, B, ...)`: X solves X = A X A' + B on the stationary part,
/// `Schur_u` spans the unit roots (|eig| > 2 - qz_criterium).
pub fn lyapunov_with_unit_roots(a: &Mat, b: &Mat, qz_criterium: f64) -> Result<(Mat, Mat)> {
    let x = lyapunov_symm(a, b, qz_criterium, LYAPUNOV_COMPLEX_THRESHOLD)?;
    let n = a.nrows();
    if n <= 1 {
        return Ok((x, Mat::zeros(n, 0)));
    }
    let mut sch = schur(a)?;
    let e1: Vec<bool> = sch.eig_abs().iter().map(|&e| e > 2.0 - qz_criterium).collect();
    let k = e1.iter().filter(|&&s| s).count();
    if k == 0 {
        return Ok((x, Mat::zeros(n, 0)));
    }
    ordschur(&mut sch, &e1)?;
    Ok((x, sch.u.columns(0, k).into_owned()))
}

fn sub(m: &Mat, rows: &[usize], cols: &[usize]) -> Mat {
    Mat::from_fn(rows.len(), cols.len(), |i, j| m[(rows[i], cols[j])])
}

fn zero_small(m: &mut Mat) {
    for v in m.iter_mut() {
        if v.abs() < 1e-12 {
            *v = 0.0;
        }
    }
}

/// Embeds the stationary block `s` (rows/cols `st` of a y_nbr x y_nbr matrix), NaN elsewhere.
fn embed(s: &Mat, st: &[usize], ny: usize) -> Mat {
    let mut m = Mat::from_element(ny, ny, f64::NAN);
    for (i, &a) in st.iter().enumerate() {
        for (j, &b) in st.iter().enumerate() {
            m[(a, b)] = s[(i, j)];
        }
    }
    m
}

/// Port of `pruned_state_space_system(M_, options_, dr, indy, nlags, useautocorr, 1)` at order 1.
#[allow(clippy::too_many_arguments)]
pub fn pruned_state_space(
    dr: &DecisionRule,
    derivs: &PerturbationDerivs,
    sigma_e: &Mat,
    indy: &[usize],
    nlags: usize,
    useautocorr: bool,
    n_shock: usize,
    qz_criterium: f64,
    schur_vec_tol: f64,
) -> Result<PrunedStateSpace> {
    let indx: Vec<usize> = (dr.nstatic..dr.nstatic + dr.nspred).collect();
    let all: Vec<usize> = (0..dr.ghx.ncols()).collect();
    let alle: Vec<usize> = (0..dr.ghu.ncols()).collect();
    let tot = derivs.dghx.len();
    let ny = indy.len();
    let a = sub(&dr.ghx, &indx, &all);
    let c = sub(&dr.ghx, indy, &all);
    let b = sub(&dr.ghu, &indx, &alle);
    let d = sub(&dr.ghu, indy, &alle);
    let varinov = sigma_e.clone();
    let om_z = mul(&mul(&b, false, &varinov, false), false, &b, true);
    let (var_z, schur_u) = lyapunov_with_unit_roots(&a, &om_z, qz_criterium)?;
    let stationary: Vec<bool> = if schur_u.ncols() > 0 {
        let x = mul(&c, false, &schur_u, false);
        (0..ny).map(|i| (0..x.ncols()).all(|j| x[(i, j)].abs() < schur_vec_tol)).collect()
    } else {
        vec![true; ny]
    };
    let st: Vec<usize> = (0..ny).filter(|&i| stationary[i]).collect();

    let da: Vec<Mat> = derivs.dghx.iter().map(|m| sub(m, &indx, &all)).collect();
    let dc: Vec<Mat> = derivs.dghx.iter().map(|m| sub(m, indy, &all)).collect();
    let db: Vec<Mat> = derivs.dghu.iter().map(|m| sub(m, &indx, &alle)).collect();
    let dd: Vec<Mat> = derivs.dghu.iter().map(|m| sub(m, indy, &alle)).collect();
    let dvarinov = derivs.dsigma_e.clone();
    let mut dvar_z = Vec::with_capacity(tot);
    for j in 0..tot {
        let rhs = if j < n_shock {
            mul(&mul(&b, false, &dvarinov[j], false), false, &b, true)
        } else {
            let t1 = mul(&mul(&db[j], false, &varinov, false), false, &b, true);
            let t2 = mul(&mul(&da[j], false, &var_z, false), false, &a, true);
            &t1 + t1.transpose() + &t2 + t2.transpose()
        };
        dvar_z.push(lyapunov_symm(&a, &rhs, qz_criterium, LYAPUNOV_COMPLEX_THRESHOLD)?);
    }

    let cs = sub(&c, &st, &all);
    let ds = sub(&d, &st, &alle);
    let var_y = {
        let m0 = mul(&mul(&cs, false, &var_z, false), false, &cs, true)
            + mul(&mul(&ds, false, &varinov, false), false, &ds, true);
        let mut m = embed(&m0, &st, ny);
        zero_small(&mut m);
        m
    };
    let sdy: Vec<f64> = st.iter().map(|&i| var_y[(i, i)].sqrt()).collect();
    let sy = Mat::from_fn(st.len(), st.len(), |i, j| sdy[i] * sdy[j]);

    let mut dvar_y = Vec::with_capacity(tot);
    for j in 0..tot {
        let dcs = sub(&dc[j], &st, &all);
        let dds = sub(&dd[j], &st, &alle);
        let t1 = mul(&mul(&dcs, false, &var_z, false), false, &cs, true);
        let t3 = mul(&mul(&dds, false, &varinov, false), false, &ds, true);
        let mut m = &t1
            + mul(&mul(&cs, false, &dvar_z[j], false), false, &cs, true)
            + t1.transpose()
            + &t3
            + mul(&mul(&ds, false, &dvarinov[j], false), false, &ds, true)
            + t3.transpose();
        zero_small(&mut m);
        dvar_y.push(embed(&m, &st, ny));
    }

    // autocovariances
    let tmp =
        mul(&mul(&a, false, &var_z, false), false, &cs, true) + mul(&mul(&b, false, &varinov, false), false, &ds, true);
    let nz = a.nrows();
    let mut var_yi = Vec::with_capacity(nlags);
    let mut ai = Mat::identity(nz, nz);
    for _ in 0..nlags {
        let v = mul(&mul(&cs, false, &ai, false), false, &tmp, false);
        var_yi.push(embed(&v, &st, ny));
        ai = mul(&ai, false, &a, false);
    }
    let mut dvar_yi = vec![Vec::with_capacity(tot); nlags];
    for j in 0..tot {
        let dcs = sub(&dc[j], &st, &all);
        let dds = sub(&dd[j], &st, &alle);
        let dtmp = mul(&mul(&da[j], false, &var_z, false), false, &cs, true)
            + mul(&mul(&a, false, &dvar_z[j], false), false, &cs, true)
            + mul(&mul(&a, false, &var_z, false), false, &dcs, true)
            + mul(&mul(&db[j], false, &varinov, false), false, &ds, true)
            + mul(&mul(&b, false, &dvarinov[j], false), false, &ds, true)
            + mul(&mul(&b, false, &varinov, false), false, &dds, true);
        let mut ai = Mat::identity(nz, nz);
        let mut dai = Mat::zeros(nz, nz);
        for lag in dvar_yi.iter_mut() {
            let v = mul(&mul(&dcs, false, &ai, false), false, &tmp, false)
                + mul(&mul(&cs, false, &dai, false), false, &tmp, false)
                + mul(&mul(&cs, false, &ai, false), false, &dtmp, false);
            lag.push(embed(&v, &st, ny));
            dai = mul(&dai, false, &a, false) + mul(&ai, false, &da[j], false);
            ai = mul(&ai, false, &a, false);
        }
    }
    let (corr_yi, dcorr_yi) = if useautocorr {
        let corr: Vec<Mat> = var_yi
            .iter()
            .map(|v| {
                let vs = sub(v, &st, &st);
                embed(&vs.component_div(&sy), &st, ny)
            })
            .collect();
        let mut dcorr = vec![Vec::with_capacity(tot); nlags];
        for j in 0..tot {
            let dsd: Vec<f64> = st.iter().enumerate().map(|(k, &i)| 0.5 / sdy[k] * dvar_y[j][(i, i)]).collect();
            let dsy = Mat::from_fn(st.len(), st.len(), |a, b| dsd[a] * sdy[b] + sdy[a] * dsd[b]);
            for (l, lag) in dcorr.iter_mut().enumerate() {
                let dv = sub(&dvar_yi[l][j], &st, &st);
                let v = sub(&var_yi[l], &st, &st);
                let m = Mat::from_fn(st.len(), st.len(), |a, b| {
                    (dv[(a, b)] * sy[(a, b)] - dsy[(a, b)] * v[(a, b)]) / (sy[(a, b)] * sy[(a, b)])
                });
                lag.push(embed(&m, &st, ny));
            }
        }
        (Some(corr), Some(dcorr))
    } else {
        (None, None)
    };

    // means (E_z = 0 at order 1)
    let yss: Vec<f64> = indy.iter().map(|&i| dr.ys[dr.order_var[i]]).collect();
    let mut e_y = yss;
    let mut de_y = Mat::zeros(ny, tot);
    for j in n_shock..tot {
        for (r, &i) in indy.iter().enumerate() {
            de_y[(r, j)] = derivs.d_yss[(i, j - n_shock)];
        }
    }
    for r in 0..ny {
        if !stationary[r] {
            e_y[r] = f64::NAN;
            for j in 0..tot {
                de_y[(r, j)] = f64::NAN;
            }
        }
    }
    Ok(PrunedStateSpace {
        indx,
        indy: indy.to_vec(),
        a,
        b,
        c,
        d,
        varinov,
        da,
        db,
        dc,
        dd,
        dvarinov,
        var_z,
        dvar_z,
        stationary,
        e_y,
        de_y,
        var_y,
        dvar_y,
        var_yi,
        dvar_yi,
        corr_yi,
        dcorr_yi,
    })
}

/// `dyn_vech`: the upper triangle, column by column.
pub fn vech(m: &Mat) -> Vec<f64> {
    let mut v = Vec::with_capacity(m.nrows() * (m.nrows() + 1) / 2);
    for j in 0..m.ncols() {
        for i in 0..=j.min(m.nrows().saturating_sub(1)) {
            v.push(m[(i, j)]);
        }
    }
    v
}

/// `M(find(tril(ones(n))))`: the lower triangle, column by column.
pub fn vech_lower(m: &Mat) -> Vec<f64> {
    let mut v = Vec::with_capacity(m.nrows() * (m.nrows() + 1) / 2);
    for j in 0..m.ncols() {
        for i in j..m.nrows() {
            v.push(m[(i, j)]);
        }
    }
    v
}

/// `MOMENTS = [E_y; vech(Var_y); vec(E_yyi(:,:,1)); ...]` and its Jacobian `dMOMENTS`
/// (get_jacobians.m, `analytic_derivation_mode >= 0`).
pub fn moments(ps: &PrunedStateSpace, nlags: usize) -> (Vec<f64>, Mat) {
    let ny = ps.indy.len();
    let tot = ps.dvar_y.len();
    let (lags, dlags) = match (&ps.corr_yi, &ps.dcorr_yi) {
        (Some(c), Some(dc)) => (c, dc),
        _ => (&ps.var_yi, &ps.dvar_yi),
    };
    let mut mom = ps.e_y.clone();
    mom.extend(vech(&ps.var_y));
    for l in lags.iter().take(nlags) {
        mom.extend(l.iter().copied());
    }
    let nrow = ny + ny * (ny + 1) / 2 + nlags * ny * ny;
    let mut dm = Mat::zeros(nrow, tot);
    for j in 0..tot {
        let mut col = ps.de_y.column(j).iter().copied().collect::<Vec<_>>();
        col.extend(vech(&ps.dvar_y[j]));
        for l in dlags.iter().take(nlags) {
            col.extend(l[j].iter().copied());
        }
        for (r, v) in col.into_iter().enumerate() {
            dm[(r, j)] = v;
        }
    }
    (mom, dm)
}

fn to_c(m: &Mat) -> CMat {
    m.map(|v| C64::new(v, 0.0))
}

fn cmul(a: &CMat, b: &CMat) -> CMat {
    a * b
}

/// `dSPECTRUM_NO_MEAN` (Qu and Tkachenko's G without the mean part) for
/// `analytic_derivation_mode = 0`: `2*pi/(2*nfreq-1) * sum_w w_k dOmega(w)' dOmega(w)` over
/// `w = 0, pi/(grid_nbr/2), ..., pi` (weight 1 at w = 0, 2 elsewhere).
pub fn spectrum_gram(ps: &PrunedStateSpace, n_shock: usize, grid_nbr: usize) -> Mat {
    let tot = ps.dvar_y.len();
    let nfreq = grid_nbr / 2 + 1;
    let nx = ps.a.nrows();
    let ny = ps.c.nrows();
    let a = to_c(&ps.a);
    let b = to_c(&ps.b);
    let c = to_c(&ps.c);
    let d = to_c(&ps.d);
    let sig = to_c(&ps.varinov);
    let da: Vec<CMat> = ps.da.iter().map(to_c).collect();
    let db: Vec<CMat> = ps.db.iter().map(to_c).collect();
    let dc: Vec<CMat> = ps.dc.iter().map(to_c).collect();
    let dd: Vec<CMat> = ps.dd.iter().map(to_c).collect();
    let dsig: Vec<CMat> = ps.dvarinov.iter().map(to_c).collect();
    let step = std::f64::consts::PI / (grid_nbr as f64 / 2.0);
    let freq_term = |ig: usize| -> CMat {
        let w = ig as f64 * step;
        let z = C64::new(w.cos(), w.sin());
        let mut m = -a.clone();
        for i in 0..nx {
            m[(i, i)] += z;
        }
        let inv = m.lu().try_inverse().unwrap_or_else(|| CMat::from_element(nx, nx, C64::new(f64::NAN, 0.0)));
        let inv_b = cmul(&inv, &b);
        let c_inv = cmul(&c, &inv);
        let tf = &d + cmul(&c, &inv_b);
        let tf_h = tf.adjoint();
        let mut domega = CMat::zeros(ny * ny, tot);
        let scale = C64::new(1.0 / (2.0 * std::f64::consts::PI), 0.0);
        for j in 0..tot {
            let om = if j < n_shock {
                cmul(&cmul(&tf, &dsig[j]), &tf_h)
            } else {
                let dtf = &dd[j] + cmul(&dc[j], &inv_b) + cmul(&cmul(&c_inv, &da[j]), &inv_b) + cmul(&c_inv, &db[j]);
                let t1 = cmul(&cmul(&dtf, &sig), &tf_h);
                t1.clone() + cmul(&cmul(&tf, &dsig[j]), &tf_h) + t1.adjoint()
            };
            for (r, v) in om.iter().enumerate() {
                domega[(r, j)] = *v * scale;
            }
        }
        let g = domega.adjoint() * &domega;
        if ig == 0 { g } else { g * C64::new(2.0, 0.0) }
    };
    // fixed chunks, summed in order: the result does not depend on the thread count
    let nchunk = 32.min(nfreq).max(1);
    let partial: Vec<CMat> = (0..nchunk)
        .into_par_iter()
        .map(|ch| {
            let lo = ch * nfreq / nchunk;
            let hi = (ch + 1) * nfreq / nchunk;
            let mut acc = CMat::zeros(tot, tot);
            for ig in lo..hi {
                acc += freq_term(ig);
            }
            acc
        })
        .collect();
    let mut sum = CMat::zeros(tot, tot);
    for p in partial {
        sum += p;
    }
    let f = 2.0 * std::f64::consts::PI / (2.0 * nfreq as f64 - 1.0);
    sum.map(|v| v.re * f)
}
