// SPDX-License-Identifier: GPL-3.0-or-later
//! `heterogeneity_solve`: port of `+heterogeneity/solve.m`, the sequence-space Jacobian method of
//! Auclert, Bardoczy, Rognlie and Straub (2021) for the first-order solution of a model with rich
//! heterogeneity around its steady state. The four MEX functions it calls are re-implemented here
//! (their sources are not distributed with the binaries; the semantics were pinned down against
//! Dynare 7.2 and are checked in `tests/heterogeneity.rs`):
//!
//! * `compute_expected_dx`: derivative of the expectation of next period's policy with respect to
//!   the (continuous) states chosen today, [`expected_dx`];
//! * `compute_curlyDs`: derivative of the next-period distribution with respect to the
//!   policies of the states, [`curly_ds`];
//! * `compute_expected_y`: expectation operator of the transition (the transpose of the
//!   forward iteration) on the distribution grid, [`expected_y`];
//! * `compute_transition_matrices`: the fake-news matrices `F`.

#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;

use anyhow::{Result, bail};

use super::HetDecl;
use super::model::HetModel;
use super::steady::{GridMat, LoadedSteadyState, Sparse};
use crate::linalg::{Mat, mul, solve};
use crate::model::Model;

/// `dr` of `heterogeneity.solve`. Index order of the vectors: `g[i * n_exo + j]` is
/// `dr.G.(endo_i).(exo_j)`; `f[k][y]`, `j_ha[k][y]` are `dr.F.(SUM_x_k).(input_y)`;
/// `curly_ys[k][y]` is `dr.curlyYs.(input_y).(x_k)`; `curly_ds[y]` is `dr.curlyDs.(input_y)`
/// (`T x N_om`, column-major, i.e. the array of dimensions `curly_ds_dims`).
#[derive(Clone, Debug, Default)]
pub struct HetDr {
    pub truncation_horizon: usize,
    pub endo_names: Vec<String>,
    pub exo_names: Vec<String>,
    pub g: Vec<Mat>,
    /// Names of the aggregates `SUM_x` (outputs of `J_ha` and `F`).
    pub output_names: Vec<String>,
    /// Names of the heterogeneous variables aggregated (outputs of `curlyYs`).
    pub curly_output_names: Vec<String>,
    /// Names of the inputs, in the order lag, current, lead, exo (`x_lag`, `x`, `x_lead`, `e`).
    pub input_names: Vec<String>,
    pub j_ha: Vec<Vec<Mat>>,
    pub f: Vec<Vec<Mat>>,
    pub curly_ys: Vec<Vec<Vec<f64>>>,
    pub curly_ds: Vec<Mat>,
    pub curly_ds_dims: Vec<usize>,
    /// `dr.J.(equation).(variable)`: rows `[timing value]`.
    pub j: BTreeMap<(String, String), Vec<[f64; 2]>>,
}

impl HetDr {
    /// `dr.G.(var).(shock)`.
    pub fn g_block(&self, var: &str, shock: &str) -> Option<&Mat> {
        let i = self.endo_names.iter().position(|n| n == var)?;
        let j = self.exo_names.iter().position(|n| n == shock)?;
        self.g.get(i * self.exo_names.len() + j)
    }
}

/// Multilinear interpolation on the state grids of a [`GridMat`]: the corners of the bracket of
/// the next-period state chosen at point `q` (point = shock index fastest, then the states).
struct Interp<'a> {
    gm: &'a GridMat,
    strides: Vec<usize>,
    nst: usize,
}

impl<'a> Interp<'a> {
    fn new(gm: &'a GridMat) -> Self {
        let nst = gm.dims.len();
        let mut strides = vec![1usize; nst];
        for k in 1..nst {
            strides[k] = strides[k - 1] * gm.dims[k - 1];
        }
        Interp { gm, strides, nst }
    }

    /// Linear index (without the shock) and weight of corner `c` (bit `k` set: upper point of
    /// state `k`). With `dk = Some(k)` the weight of state `k` is replaced by the derivative of
    /// the weight with respect to the position of the state chosen.
    fn corner(&self, q: usize, c: usize, dk: Option<usize>) -> (usize, f64) {
        let np = self.gm.n_points;
        let mut lin = 0usize;
        let mut wt = 1.0;
        for k in 0..self.nst {
            let idx = q + k * np;
            let w = self.gm.w[idx];
            let up = c >> k & 1 == 1;
            lin += self.strides[k] * (self.gm.ind[idx] as usize - 1 + up as usize);
            let f = if dk == Some(k) {
                let ih = self.gm.inv_h[idx];
                if up { ih } else { -ih }
            } else if up {
                1.0 - w
            } else {
                w
            };
            wt *= f;
        }
        (lin, wt)
    }
}

/// `compute_expected_dx(x_bar, ind, w, inv_h, Mu, dims)`: `n x nst x N_sp` (column-major pages),
/// `Ex(:, k, p) = sum_{e'} Mu(e, e') d/d a_k x_bar(e', a)` at the state chosen at point `p`.
fn expected_dx(x_bar: &Mat, gm: &GridMat, mu: &Mat) -> Vec<f64> {
    let n = x_bar.nrows();
    let n_e = mu.nrows();
    let it = Interp::new(gm);
    let np = gm.n_points;
    let nst = it.nst;
    let mut out = vec![0.0; n * nst * np];
    for p in 0..np {
        let e = p % n_e;
        for k in 0..nst {
            for c in 0..(1usize << nst) {
                let (lin, wt) = it.corner(p, c, Some(k));
                if wt == 0.0 {
                    continue;
                }
                for e2 in 0..n_e {
                    let m = mu[(e, e2)] * wt;
                    let col = lin * n_e + e2;
                    for r in 0..n {
                        out[r + n * (k + nst * p)] += m * x_bar[(r, col)];
                    }
                }
            }
        }
    }
    out
}

/// `compute_expected_y(ind, w, y, Mu, dims)` for the columns of `y` (`N_om x m`): the expected value
/// of `y` next period given the point today.
fn expected_y(gm: &GridMat, mu: &Mat, y: &Mat) -> Mat {
    let n_e = mu.nrows();
    let it = Interp::new(gm);
    let np = gm.n_points;
    let mut out = Mat::zeros(np, y.ncols());
    for q in 0..np {
        let e = q % n_e;
        for c in 0..(1usize << it.nst) {
            let (lin, wt) = it.corner(q, c, None);
            if wt == 0.0 {
                continue;
            }
            for e2 in 0..n_e {
                let m = mu[(e, e2)] * wt;
                let col = lin * n_e + e2;
                for j in 0..y.ncols() {
                    out[(q, j)] += m * y[(col, j)];
                }
            }
        }
    }
    out
}

/// `compute_curlyDs(ind, w, inv_h, px_hat, hist, Mu, dims)` for one `(t, input)`: `dp[q * nst + k]`
/// is the perturbation of state `k`'s policy at point `q`; returns the change of the distribution.
fn curly_ds(gm: &GridMat, mu: &Mat, hist: &[f64], dp: &[f64]) -> Vec<f64> {
    let n_e = mu.nrows();
    let it = Interp::new(gm);
    let np = gm.n_points;
    let nst = it.nst;
    let mut out = vec![0.0; np];
    for q in 0..np {
        let m0 = hist[q];
        if m0 == 0.0 {
            continue;
        }
        let e = q % n_e;
        for k in 0..nst {
            let d = dp[q * nst + k];
            if d == 0.0 {
                continue;
            }
            for c in 0..(1usize << nst) {
                let (lin, wt) = it.corner(q, c, Some(k));
                if wt == 0.0 {
                    continue;
                }
                for e2 in 0..n_e {
                    out[lin * n_e + e2] += m0 * d * wt * mu[(e, e2)];
                }
            }
        }
    }
    out
}

/// `P * v` with `P` sparse, column `j` of the result = `sum_i P(i, j) * a(:, i)`:
/// `a * P` for dense `a` (`m x P.nrows()`).
fn dense_times_sparse(a: &Mat, p: &Sparse) -> Mat {
    let m = a.nrows();
    let mut out = Mat::zeros(m, p.ncols);
    for j in 0..p.ncols {
        for idx in p.colptr[j]..p.colptr[j + 1] {
            let (i, v) = (p.rowind[idx], p.vals[idx]);
            for r in 0..m {
                out[(r, j)] += v * a[(r, i)];
            }
        }
    }
    out
}

/// `compute_impulse_responses`: `T` matrices of `(n * N_Y) x N_sp` (row `i + n * y`).
pub fn impulse_responses(h: &HetModel, ss: &LoadedSteadyState, t: usize) -> Result<Vec<Mat>> {
    let mat = &ss.mat;
    let n = h.endo_nbr();
    let n_sp = ss.sizes.n_sp;
    let idx = &ss.indices;
    let ny = idx.y_all.len();
    let nst = h.state_var.len();
    let (dn, ncols, dp) = mat.df_dims;
    if dn != n || dp != n_sp {
        bail!("heterogeneity_solve: inconsistent dimensions of mat.dF");
    }
    let phi_e =
        mat.pol_phi_e.as_ref().ok_or_else(|| anyhow::anyhow!("heterogeneity_solve: mat.pol.Phi_e is missing"))?;
    let ex = expected_dx(&mat.x_bar, &mat.pol, &mat.mu);
    let mut c_fx_next: Vec<Mat> = Vec::with_capacity(n_sp);
    let mut first = Mat::zeros(n * ny, n_sp);
    for p in 0..n_sp {
        let page = |c0: usize, nc: usize| Mat::from_fn(n, nc, |i, j| mat.df[i + n * (c0 + j) + n * ncols * p]);
        let dfxe = page(2 * n, n);
        let mut f3 = page(n, n);
        let exp = Mat::from_fn(n, nst, |i, k| ex[i + n * (k + nst * p)]);
        let ae3 = &dfxe * &exp;
        for (k, &s) in h.state_var.iter().enumerate() {
            for i in 0..n {
                f3[(i, s)] += ae3[(i, k)];
            }
        }
        let mut rhs = Mat::zeros(n, n + ny);
        for i in 0..n {
            for j in 0..n {
                rhs[(i, j)] = dfxe[(i, j)];
            }
            for (y, &c) in idx.y_ind_in_df.iter().enumerate() {
                rhs[(i, n + y)] = mat.df[i + n * (c - 1) + n * ncols * p];
            }
        }
        let z = match f3.clone().lu().solve(&rhs) {
            Some(z) => -z,
            None => bail!("heterogeneity_solve: singular Jacobian of the heterogeneous block at point {}", p + 1),
        };
        c_fx_next.push(z.columns(0, n).into_owned());
        for y in 0..ny {
            for i in 0..n {
                first[(i + n * y, p)] = z[(i, n + y)];
            }
        }
    }
    let mut irs = Vec::with_capacity(t);
    irs.push(first);
    for s in 1..t {
        let ex_dash = dense_times_sparse(&irs[s - 1], phi_e);
        let mut cur = Mat::zeros(n * ny, n_sp);
        for p in 0..n_sp {
            for y in 0..ny {
                for i in 0..n {
                    let mut acc = 0.0;
                    for k in 0..n {
                        acc += c_fx_next[p][(i, k)] * ex_dash[(k + n * y, p)];
                    }
                    cur[(i + n * y, p)] = acc;
                }
            }
        }
        irs.push(cur);
    }
    Ok(irs)
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect()
}

/// `heterogeneity.solve(M_, options_solve, oo_het)`.
pub fn solve_hetero(model: &Model, het: &HetDecl, ss: &LoadedSteadyState, t: usize) -> Result<HetDr> {
    let Some(h) = het.model.as_ref() else {
        bail!("heterogeneity_solve: the model has no heterogeneous block");
    };
    if t < 2 {
        bail!("heterogeneity_solve: truncation_horizon must be at least 2");
    }
    let mat = &ss.mat;
    let idx = &ss.indices;
    let sizes = &ss.sizes;
    let (n, nst) = (h.endo_nbr(), h.state_var.len());
    let (ne, nx) = (model.n_endo(), model.n_exo());
    let n_om = sizes.n_om;
    let n_ix = idx.ix_in_agg.len();
    let ny = idx.y_all.len();
    let (n_lag, n_cur, n_lead, n_exo_y) = (idx.y_lag.len(), idx.y_current.len(), idx.y_lead.len(), idx.y_exo.len());
    let n_e = sizes.n_e_total;
    let dg = mat.dg.as_ref().ok_or_else(|| anyhow::anyhow!("heterogeneity_solve: mat.dG is missing"))?;
    let d_phi = mat.d_phi.as_ref().ok_or_else(|| anyhow::anyhow!("heterogeneity_solve: mat.d.Phi is missing"))?;
    let d_x_bar = mat.d_x_bar.as_ref().ok_or_else(|| anyhow::anyhow!("heterogeneity_solve: mat.d.x_bar is missing"))?;
    if mat.d.dims.len() != nst || mat.d_hist.len() != n_om || mat.mu.nrows() != n_e {
        bail!("heterogeneity_solve: inconsistent steady-state matrices");
    }

    // impulse responses of the policies, distributed on the distribution grid
    let irs = impulse_responses(h, ss, t)?;
    let x_hat: Vec<Mat> = irs.iter().map(|ir| dense_times_sparse(ir, d_phi)).collect();
    drop(irs);

    // curlyDs (T x N_om per input) and curlyYs (T per input and aggregate)
    let hist = &mat.d_hist;
    let mut curly_ds_m: Vec<Mat> = (0..ny).map(|_| Mat::zeros(t, n_om)).collect();
    let mut curly_ys = vec![vec![vec![0.0; t]; ny]; n_ix];
    for (s, xh) in x_hat.iter().enumerate() {
        for y in 0..ny {
            let mut dp = vec![0.0; n_om * nst];
            for q in 0..n_om {
                for (k, &sv) in h.state_var.iter().enumerate() {
                    dp[q * nst + k] = xh[(sv + n * y, q)];
                }
            }
            let cd = curly_ds(&mat.d, &mat.mu, hist, &dp);
            for q in 0..n_om {
                curly_ds_m[y][(s, q)] = cd[q];
            }
            for (j, &ih) in idx.ix_in_het.iter().enumerate() {
                let row = ih + n * y;
                curly_ys[j][y][s] = (0..n_om).map(|q| hist[q] * xh[(row, q)]).sum();
            }
        }
    }
    drop(x_hat);

    // expectations (N_om x N_Ix per horizon)
    let mut expectations: Vec<Mat> = Vec::with_capacity(t - 1);
    let mut e1 = Mat::from_fn(n_om, n_ix, |q, i| d_x_bar[(idx.ix_in_het[i], q)]);
    demean(&mut e1);
    expectations.push(e1);
    for s in 1..t - 1 {
        let mut e = expected_y(&mat.d, &mat.mu, &expectations[s - 1]);
        demean(&mut e);
        expectations.push(e);
    }

    // transition matrices F and heterogeneous Jacobian
    let mut f_all: Vec<Vec<Mat>> = Vec::with_capacity(n_ix);
    let mut jha_all: Vec<Vec<Mat>> = Vec::with_capacity(n_ix);
    for j in 0..n_ix {
        // E_j: N_om x (T-1)
        let ej = Mat::from_fn(n_om, t - 1, |q, s| expectations[s][(q, j)]);
        let mut fs = Vec::with_capacity(ny);
        let mut js = Vec::with_capacity(ny);
        for y in 0..ny {
            let mut f = Mat::zeros(t, t);
            for s in 0..t {
                f[(0, s)] = curly_ys[j][y][s];
            }
            if t > 1 {
                let sub = mul(&ej, true, &curly_ds_m[y], true); // (T-1) x T
                for s in 1..t {
                    for c in 0..t {
                        f[(s, c)] = sub[(s - 1, c)];
                    }
                }
            }
            let mut jm = f.clone();
            for c in 1..t {
                for r in 1..t {
                    jm[(r, c)] = f[(r, c)] + jm[(r - 1, c - 1)];
                }
            }
            fs.push(f);
            js.push(jm);
        }
        f_all.push(fs);
        jha_all.push(js);
    }
    // state and shock Jacobians (dense)
    let state_var: Vec<usize> = (0..ne).filter(|&i| model.lead_lag_incidence[i][0] != 0).collect();
    let nt = ne * t;
    let mut sj = Mat::zeros(nt, nt);
    let mut shj = Mat::zeros(nt, nx * t);
    for r in 0..ne {
        for c in 0..ne {
            let b = dg[(r, c)];
            let a = dg[(r, ne + c)];
            let l = dg[(r, 2 * ne + c)];
            for s in 0..t {
                if a != 0.0 {
                    sj[(r * t + s, c * t + s)] += a;
                }
                if s >= 1 && b != 0.0 {
                    sj[(r * t + s, c * t + s - 1)] += b;
                }
                if s + 1 < t && l != 0.0 {
                    sj[(r * t + s, c * t + s + 1)] += l;
                }
            }
        }
        for c in 0..nx {
            let d = dg[(r, 3 * ne + c)];
            if d != 0.0 {
                for s in 0..t {
                    shj[(r * t + s, c * t + s)] += d;
                }
            }
        }
    }
    for i in 0..n_ix {
        let rb = idx.ix_in_agg[i] * t;
        let mut iy = 0usize;
        for j in 0..n_lag {
            let v = idx.y_lag[j] - 1;
            if state_var.contains(&v) {
                for r in 0..t {
                    for s in 0..t - 1 {
                        sj[(rb + r, v * t + s)] -= jha_all[i][iy][(r, s + 1)];
                    }
                }
            }
            iy += 1;
        }
        for j in 0..n_cur {
            let v = idx.y_current[j] - 1;
            for r in 0..t {
                for s in 0..t {
                    sj[(rb + r, v * t + s)] -= jha_all[i][iy][(r, s)];
                }
            }
            iy += 1;
        }
        for j in 0..n_lead {
            let v = idx.y_lead[j] - 1;
            for r in 0..t {
                for s in 1..t {
                    sj[(rb + r, v * t + s)] -= jha_all[i][iy][(r, s - 1)];
                }
            }
            iy += 1;
        }
        for j in 0..n_exo_y {
            let v = idx.y_exo[j] - 1;
            for r in 0..t {
                for s in 0..t {
                    shj[(rb + r, v * t + s)] -= jha_all[i][iy][(r, s)];
                }
            }
            iy += 1;
        }
    }
    let Some(gfull) = solve(&sj, &shj) else {
        bail!("heterogeneity_solve: the state Jacobian is singular");
    };
    let gfull = -gfull;
    let mut g = Vec::with_capacity(ne * nx);
    for i in 0..ne {
        for j in 0..nx {
            g.push(gfull.view((i * t, j * t), (t, t)).into_owned());
        }
    }

    // names
    let mut input_names = Vec::new();
    for &v in &idx.y_lag {
        input_names.push(format!("{}_lag", model.endo_names[v - 1]));
    }
    for &v in &idx.y_current {
        input_names.push(model.endo_names[v - 1].clone());
    }
    for &v in &idx.y_lead {
        input_names.push(format!("{}_lead", model.endo_names[v - 1]));
    }
    for &v in &idx.y_exo {
        input_names.push(model.exo_names[v - 1].clone());
    }
    let output_names: Vec<String> = het
        .aggregates
        .iter()
        .filter(|a| a.op == "sum")
        .map(|a| format!("{}_{}", a.op.to_uppercase(), h.endo_names[a.het_endo]))
        .collect();
    let curly_output_names: Vec<String> = idx.ix_in_het.iter().map(|&i| h.endo_names[i].clone()).collect();

    // dr.J
    let mut tags: Vec<String> = Vec::new();
    for eq in &model.equation_tags {
        for (_, v) in eq {
            tags.push(v.clone());
        }
    }
    let mut jmap: BTreeMap<(String, String), Vec<[f64; 2]>> = BTreeMap::new();
    let mut eq = String::new();
    let blocks: [(usize, f64, bool); 4] =
        [(ne, 0.0, false), (0, -1.0, false), (2 * ne, 1.0, false), (3 * ne, 0.0, true)];
    for (c0, timing, is_exo) in blocks {
        let ncb = if is_exo { nx } else { ne };
        for c in 0..ncb {
            for r in 0..dg.nrows() {
                let v = dg[(r, c0 + c)];
                if v == 0.0 {
                    continue;
                }
                if r < model.orig_endo_nbr {
                    eq = sanitize(&format!("res_{}", tags.get(r).cloned().unwrap_or_default()));
                }
                let var = if is_exo { model.exo_names[c].clone() } else { model.endo_names[c].clone() };
                let key = (eq.clone(), var);
                if timing == -1.0 {
                    jmap.entry(key).or_default().push([timing, v]);
                } else {
                    jmap.insert(key, vec![[timing, v]]);
                }
            }
        }
    }
    let mut dims = vec![t];
    for s in &idx.base.shocks {
        dims.push(sizes.shock(s));
    }
    for s in &idx.base.states {
        dims.push(sizes.d_state(s));
    }
    Ok(HetDr {
        truncation_horizon: t,
        endo_names: model.endo_names.clone(),
        exo_names: model.exo_names.clone(),
        g,
        output_names,
        curly_output_names,
        input_names,
        j_ha: jha_all,
        f: f_all,
        curly_ys,
        curly_ds: curly_ds_m,
        curly_ds_dims: dims,
        j: jmap,
    })
}

fn demean(m: &mut Mat) {
    let n = m.nrows() as f64;
    for j in 0..m.ncols() {
        let s: f64 = (0..m.nrows()).map(|i| m[(i, j)]).sum::<f64>() / n;
        for i in 0..m.nrows() {
            m[(i, j)] -= s;
        }
    }
}
