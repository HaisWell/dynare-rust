// SPDX-License-Identifier: GPL-3.0-or-later
//! Komunjer and Ng (2011): the minimal state space and the Jacobian `Deltabar` of the
//! minimal system (`+identification/get_minimal_state_representation.m` and the
//! `dMINIMAL` part of `get_jacobians.m`).

use crate::linalg::{Mat, mul};

use super::la::{rank, svd_v_full};
use super::moments::PrunedStateSpace;

pub struct MinimalSystem {
    pub a: Mat,
    pub b: Mat,
    pub c: Mat,
    pub d: Mat,
    pub da: Vec<Mat>,
    pub db: Vec<Mat>,
    pub dc: Vec<Mat>,
    pub dd: Vec<Mat>,
}

fn kron(a: &Mat, b: &Mat) -> Mat {
    a.kronecker(b)
}

/// `check_minimality`: full rank of the controllability and observability matrices.
fn check_minimality(a: &Mat, b: &Mat, c: &Mat) -> bool {
    let n = a.nrows();
    let mut cc = b.clone();
    let mut oo = c.clone();
    let mut ap = Mat::identity(n, n);
    for _ in 1..n.max(1) {
        ap = mul(&ap, false, a, false);
        let ab = mul(&ap, false, b, false);
        let ca = mul(c, false, &ap, false);
        let mut ncc = Mat::zeros(n, cc.ncols() + ab.ncols());
        ncc.columns_mut(0, cc.ncols()).copy_from(&cc);
        ncc.columns_mut(cc.ncols(), ab.ncols()).copy_from(&ab);
        cc = ncc;
        let mut noo = Mat::zeros(oo.nrows() + ca.nrows(), n);
        noo.rows_mut(0, oo.nrows()).copy_from(&oo);
        noo.rows_mut(oo.nrows(), ca.nrows()).copy_from(&ca);
        oo = noo;
    }
    rank(&cc, None) == n && rank(&oo, None) == n
}

/// `ControllabilityStaircaseRosenbrock(a, b, c, tol)`: returns (abar, bbar, cbar, t, k).
fn controllability_staircase(a: &Mat, b: &Mat, c: &Mat, tol: f64) -> (Mat, Mat, Mat, Mat, Vec<usize>) {
    let ra = a.nrows();
    let mut ptjn1 = Mat::identity(ra, ra);
    let mut ajn1 = a.clone();
    let mut bjn1 = b.clone();
    let mut deltajn1 = 0;
    let mut k = vec![0usize; ra];
    for kj in k.iter_mut() {
        // [uj, sj] = svd(bjn1) (full U), columns reversed
        let (_, u_full) = svd_v_full(&bjn1.transpose());
        let rs = bjn1.nrows();
        let uj = Mat::from_fn(rs, rs, |i, j| u_full[(i, rs - 1 - j)]);
        let bb = mul(&uj, true, &bjn1, false);
        let roj = rank(&bb, Some(tol));
        let sigmaj = rs - roj;
        *kj = roj;
        if roj == 0 || sigmaj == 0 {
            break;
        }
        let abxy = mul(&mul(&uj, true, &ajn1, false), false, &uj, false);
        ajn1 = abxy.view((0, 0), (sigmaj, sigmaj)).into_owned();
        bjn1 = abxy.view((0, sigmaj), (sigmaj, roj)).into_owned();
        let mut blk = Mat::identity(rs + deltajn1, rs + deltajn1);
        blk.view_mut((0, 0), (rs, rs)).copy_from(&uj);
        ptjn1 = mul(&ptjn1, false, &blk, false);
        deltajn1 += roj;
    }
    let t = ptjn1.transpose();
    let abar = mul(&mul(&t, false, a, false), false, &t, true);
    let bbar = mul(&t, false, b, false);
    let cbar = mul(c, false, &t, true);
    (abar, bbar, cbar, t, k)
}

/// `minrealold(SYS, tol)`: controllable then observable part; `U` is the controllability
/// transformation.
fn minrealold(a: &Mat, b: &Mat, c: &Mat, tol: f64) -> (Mat, Mat, Mat, Mat) {
    let ns0 = b.nrows();
    let (am, bm, cm, u, k) = controllability_staircase(a, b, c, tol);
    let kk: usize = k.iter().sum();
    let nu = ns0 - kk;
    let mut am = am.view((nu, nu), (ns0 - nu, ns0 - nu)).into_owned();
    let mut bm = bm.rows(nu, ns0 - nu).into_owned();
    let mut cm = cm.columns(nu, ns0 - nu).into_owned();
    let ns = ns0 - nu;
    if ns > 0 {
        let (aa, bb, cc, _, k) = controllability_staircase(&am.transpose(), &cm.transpose(), &bm.transpose(), tol);
        let (am2, bm2, cm2) = (aa.transpose(), cc.transpose(), bb.transpose());
        let kk: usize = k.iter().sum();
        let nu = ns - kk;
        am = am2.view((nu, nu), (ns - nu, ns - nu)).into_owned();
        bm = bm2.rows(nu, ns - nu).into_owned();
        cm = cm2.columns(nu, ns - nu).into_owned();
    }
    (am, bm, cm, u)
}

/// Port of `get_minimal_state_representation(SYS, 1)`; `None` when no minimal system was found
/// (`CheckCO == 0`).
pub fn minimal_state_representation(ps: &PrunedStateSpace) -> Option<MinimalSystem> {
    let (a, b, c, d) = (&ps.a, &ps.b, &ps.c, &ps.d);
    if check_minimality(a, b, c) {
        return Some(MinimalSystem {
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
            d: d.clone(),
            da: ps.da.clone(),
            db: ps.db.clone(),
            dc: ps.dc.clone(),
            dd: ps.dd.clone(),
        });
    }
    let (am, bm, cm, u) = minrealold(a, b, c, 1e-7);
    let minns = am.nrows();
    if !check_minimality(&am, &bm, &cm) {
        return None;
    }
    let tot = ps.da.len();
    let mut da = Vec::with_capacity(tot);
    let mut db = Vec::with_capacity(tot);
    let mut dc = Vec::with_capacity(tot);
    for j in 0..tot {
        let t = mul(&mul(&u, false, &ps.da[j], false), false, &u, true);
        da.push(t.view((0, 0), (minns, minns)).into_owned());
        let t = mul(&u, false, &ps.db[j], false);
        db.push(t.rows(0, minns).into_owned());
        let t = mul(&ps.dc[j], false, &u, true);
        dc.push(t.columns(0, minns).into_owned());
    }
    Some(MinimalSystem { a: am, b: bm, c: cm, d: d.clone(), da, db, dc, dd: ps.dd.clone() })
}

/// `[~, DpMPinv] = duplication(p)`: Moore–Penrose inverse of the duplication matrix.
fn duplication_mp_inverse(p: usize) -> Mat {
    let m = p * (p + 1) / 2;
    let mut dp = Mat::zeros(p * p, m);
    for j in 0..p {
        for i in 0..p {
            let (r, c) = if i >= j { (i, j) } else { (j, i) };
            dp[(i + j * p, super::la::tril_index(p, r, c))] = 1.0;
        }
    }
    let dtd = mul(&dp, true, &dp, false);
    super::la::solve(&dtd, &dp.transpose())
}

/// `dMINIMAL = [dMEAN zeros; KomunjerNg_DL KomunjerNg_DT KomunjerNg_DU]`.
pub fn d_minimal(ms: &MinimalSystem, de_y: &Mat, dsigma_e: &[Mat], sigma_e: &Mat) -> Mat {
    let nx = ms.a.nrows();
    let ny = ms.c.nrows();
    let nu = ms.b.ncols();
    let tot = ms.da.len();
    let nvech = nu * (nu + 1) / 2;
    let rows_dl = nx * nx + nx * nu + ny * nx + ny * nu + nvech;
    let mut dl = Mat::zeros(rows_dl, tot);
    for j in 0..tot {
        let mut col: Vec<f64> = ms.da[j].iter().copied().collect();
        col.extend(ms.db[j].iter().copied());
        col.extend(ms.dc[j].iter().copied());
        col.extend(ms.dd[j].iter().copied());
        col.extend(super::moments::vech_lower(&dsigma_e[j]));
        for (r, v) in col.into_iter().enumerate() {
            dl[(r, j)] = v;
        }
    }
    let ix = Mat::identity(nx, nx);
    let iu = Mat::identity(nu, nu);
    let mut dt = Mat::zeros(rows_dl, nx * nx);
    let blk1 = kron(&ms.a.transpose(), &ix) - kron(&ix, &ms.a);
    dt.view_mut((0, 0), (nx * nx, nx * nx)).copy_from(&blk1);
    dt.view_mut((nx * nx, 0), (nx * nu, nx * nx)).copy_from(&kron(&ms.b.transpose(), &ix));
    dt.view_mut((nx * nx + nx * nu, 0), (ny * nx, nx * nx)).copy_from(&(-kron(&ix, &ms.c)));
    let mut du = Mat::zeros(rows_dl, nu * nu);
    du.view_mut((nx * nx, 0), (nx * nu, nu * nu)).copy_from(&kron(&iu, &ms.b));
    du.view_mut((nx * nx + nx * nu + ny * nx, 0), (ny * nu, nu * nu)).copy_from(&kron(&iu, &ms.d));
    let enu = duplication_mp_inverse(nu);
    let last = mul(&enu, false, &kron(sigma_e, &iu), false) * -2.0;
    du.view_mut((rows_dl - nvech, 0), (nvech, nu * nu)).copy_from(&last);
    let ncol = tot + nx * nx + nu * nu;
    let mut out = Mat::zeros(ny + rows_dl, ncol);
    out.view_mut((0, 0), (ny, tot)).copy_from(de_y);
    out.view_mut((ny, 0), (rows_dl, tot)).copy_from(&dl);
    out.view_mut((ny, tot), (rows_dl, nx * nx)).copy_from(&dt);
    out.view_mut((ny, tot + nx * nx), (rows_dl, nu * nu)).copy_from(&du);
    out
}
