// SPDX-License-Identifier: GPL-3.0-or-later
//! `ms_estimation`: posterior mode of the Markov-switching SBVAR of Sims-Waggoner-Zha
//! (the `ms_sbvar_command_line -estimate` mex of Dynare 7.2, whose C sources are not shipped; the
//! model was reconstructed from `init_*.dat` / `est_*.out` files and checked numerically, see
//! README "MS-SBVAR, part 2a").
//!
//! Model (regimes `s = 1..S` are the product of the chains' regimes, chain 1 slowest):
//! `y_t' A0_j(s) - x_t' A+_j(s) = e_tj / sqrt(zeta_j(s))`, `e_t ~ N(0, I)`, with
//! `A0_j(s) = U_j b_{j,c_j(s)}`, `A+_j(s) = V_j g_{j,c_j(s)} + [A0_j(s); 0]` (the random-walk prior mean,
//! `specification = 2`), `zeta_j` the variance regime parameters (`zeta = 1` in the first regime of each
//! equation), `c_j(s)` / `v_j(s)` the coefficient / variance regime of equation `j` (the chains
//! controlling it through `svar`), and `Q = kron(Q_1, ..., Q_C)` with `Q[i, j] = P(s_t = i | s_{t-1} = j)`.
//! The initial regime is uniform. Prior: `b ~ N(0, (U' H0inv U)^-1)`, `g ~ N(0, (V' Hpinv V)^-1)`,
//! `zeta ~ Gamma(alpha, beta)` (rate), each column of `Q_c` is `M_c d` with `d` Dirichlet; every
//! density is normalised (the reported prior values include the constants).

use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use super::hamilton::{self, Filtered};
use super::identification::Restrictions;
use super::sbvar::{self, SbvarSettings};
use super::{Equations, MsChain, MsDecl};
use crate::estimation::special::gammaln;
use crate::linalg::Mat;

const LN2PI: f64 = 1.8378770664093453;

#[derive(Clone, Debug)]
pub struct ChainSpec {
    pub n: usize,
    /// Dirichlet prior matrix `Alpha(i, j)` of `ms_write_markov_file.m`.
    pub alpha: Mat,
}

/// Offsets of the blocks of the free-parameter vector (`est_free_*.out` order).
#[derive(Clone, Debug)]
pub struct Layout {
    pub b: Vec<Vec<usize>>,
    pub g: Vec<Vec<usize>>,
    /// Sims-Zha specification: `lam[j][r-1]` offset of the `nvar` scale factors of coefficient regime `r >= 1`
    /// (regime 0 has `lambda = 1`) and `psi[j]` offset of the `ncoef - 1` lag coefficients followed by one
    /// constant per coefficient regime (both empty otherwise).
    pub lam: Vec<Vec<usize>>,
    pub psi: Vec<usize>,
    pub z: Vec<Vec<usize>>,
    pub q: Vec<Vec<usize>>,
    pub len: usize,
}

#[derive(Clone, Debug)]
pub struct MsSetup {
    pub nvar: usize,
    pub nlags: usize,
    pub ncoef: usize,
    pub nobs: usize,
    pub y: Mat,
    pub x: Mat,
    pub ui: Vec<Mat>,
    pub vi: Vec<Mat>,
    pub n0: Vec<usize>,
    pub np: Vec<usize>,
    /// Prior precisions of the `b_j` / `g_j` blocks (`U' H0inv U`, `V' Hpinv V`) and their log determinants.
    pub pb: Vec<Mat>,
    pub pg: Vec<Mat>,
    pub pb_ld: Vec<f64>,
    pub pg_ld: Vec<f64>,
    pub chains: Vec<ChainSpec>,
    pub nstates: usize,
    /// `[j][s]`: coefficient / variance regime index of equation `j` in regime `s`.
    pub coef_idx: Vec<Vec<usize>>,
    pub var_idx: Vec<Vec<usize>>,
    pub n_coef_reg: Vec<usize>,
    pub n_var_reg: Vec<usize>,
    pub zeta_a: f64,
    pub zeta_b: f64,
    pub layout: Layout,
    pub freq: usize,
    pub initial_year: usize,
    pub initial_subperiod: usize,
    /// `A+_j = V_j g_j - W_j A0_j` (`W_j = -[I; 0]` for the random-walk and Sims-Zha specifications).
    pub w: Vec<Mat>,
    /// 0 = default, 1 = Sims-Zha, 2 = random walk (`options_.ms.specification`).
    pub specification: usize,
    /// Prior variances of the columns of `A0` / `A+` before the restrictions (the matrices of `init_*.dat`).
    pub h0var: Vec<Mat>,
    pub hpvar: Vec<Mat>,
    /// Variance of the Gaussian prior on the Sims-Zha `lambda`.
    pub lambda_var: f64,
    /// Dynare scales the prior precision of the coefficients of equation `j` by (number of distinct coefficient
    /// regime tuples of all equations) / (coefficient regimes of `j`), i.e. it sums the precision over the
    /// combined coefficient states (1 with a single chain); pinned down with hand-edited init files of
    /// two-chain models (Dynare 7.2).
    pub coef_mult: Vec<f64>,
}

/// The parameters of every regime.
pub struct Regimes {
    pub a0: Vec<Mat>,
    pub aplus: Vec<Mat>,
    /// `[s][j]`.
    pub zeta: Vec<Vec<f64>>,
    /// Combined transition matrix.
    pub q: Mat,
    /// Transition matrix of each chain.
    pub qc: Vec<Mat>,
}

/// Rows of the Dirichlet design `M_c` of column `i` (`ms_write_markov_file.m`): `(rows of d1 with weight, rows of d2)`.
pub fn chain_columns(n: usize, i: usize) -> ([(usize, f64); 2], usize, usize) {
    // returns d1 contributions (up to two rows), then (d2 row, number of valid d1 rows)
    if n == 1 {
        return ([(0, 1.0), (0, 0.0)], 0, 1);
    }
    if i == 0 {
        ([(0, 1.0), (0, 0.0)], 1, 1)
    } else if i == n - 1 {
        ([(n - 2, 1.0), (0, 0.0)], n - 1, 1)
    } else {
        ([(i - 1, 0.5), (i + 1, 0.5)], i, 2)
    }
}

pub fn chain_nfree(n: usize) -> usize {
    if n <= 1 { 0 } else { n }
}

/// `Q_c` from the first Dirichlet component `d1` of every column.
pub fn chain_q(n: usize, d: &[f64]) -> Mat {
    let mut q = Mat::zeros(n, n);
    if n == 1 {
        q[(0, 0)] = 1.0;
        return q;
    }
    for i in 0..n {
        let (d1rows, d2row, nr) = chain_columns(n, i);
        for &(r, w) in d1rows.iter().take(nr) {
            q[(r, i)] += w * d[i];
        }
        q[(d2row, i)] += 1.0 - d[i];
    }
    q
}

/// Dirichlet prior parameters `(a1, a2)` of column `i` of chain `c`.
pub fn chain_alpha(ch: &ChainSpec, i: usize) -> (f64, f64) {
    let (d1rows, d2row, nr) = chain_columns(ch.n, i);
    let a1: f64 = d1rows.iter().take(nr).map(|&(r, _)| ch.alpha[(r, i)]).sum::<f64>() / nr as f64;
    (a1, ch.alpha[(d2row, i)])
}

fn logabsdet(m: &Mat) -> f64 {
    let lu = m.clone().lu();
    lu.u().diagonal().iter().map(|x| x.abs().ln()).sum()
}

fn kron_q(qs: &[Mat]) -> Mat {
    let mut acc = Mat::from_element(1, 1, 1.0);
    for q in qs {
        let (r, c) = (acc.nrows() * q.nrows(), acc.ncols() * q.ncols());
        acc = Mat::from_fn(r, c, |i, j| acc[(i / q.nrows(), j / q.ncols())] * q[(i % q.nrows(), j % q.ncols())]);
    }
    acc
}

impl MsSetup {
    fn make_layout(n0: &[usize], np: &[usize], ncr: &[usize], nvr: &[usize], chains: &[ChainSpec], sz: bool) -> Layout {
        let nvar = n0.len();
        let mut pos = 0;
        let mut b = vec![];
        for j in 0..nvar {
            let mut v = vec![];
            for _ in 0..ncr[j] {
                v.push(pos);
                pos += n0[j];
            }
            b.push(v);
        }
        let (mut g, mut lam, mut psi) = (vec![], vec![], vec![]);
        if sz {
            for j in 0..nvar {
                let mut v = vec![];
                for _ in 1..ncr[j] {
                    v.push(pos);
                    pos += nvar;
                }
                lam.push(v);
                g.push(vec![]);
            }
            for j in 0..nvar {
                psi.push(pos);
                pos += np[j] - 1 + ncr[j];
            }
        } else {
            for j in 0..nvar {
                let mut v = vec![];
                for _ in 0..ncr[j] {
                    v.push(pos);
                    pos += np[j];
                }
                g.push(v);
                lam.push(vec![]);
            }
        }
        let mut z = vec![];
        for j in 0..nvr.len() {
            let mut v = vec![];
            for _ in 1..nvr[j] {
                v.push(pos);
                pos += 1;
            }
            z.push(v);
        }
        let mut q = vec![];
        for c in chains {
            let mut v = vec![];
            for _ in 0..chain_nfree(c.n) {
                v.push(pos);
                pos += 1;
            }
            q.push(v);
        }
        Layout { b, g, lam, psi, z, q, len: pos }
    }

    pub fn sims_zha(&self) -> bool {
        !self.layout.psi.is_empty()
    }

    /// Deviation `A+_j + W_j A0_j` of equation `j` in coefficient regime `r` (`V_j g`, or the Sims-Zha form).
    pub fn aplus_dev(&self, th: &[f64], j: usize, r: usize) -> Mat {
        let lay = &self.layout;
        if self.sims_zha() {
            let k = self.ncoef - 1;
            let p = lay.psi[j];
            let mut d = Mat::zeros(self.ncoef, 1);
            for c in 0..k {
                let l = if r == 0 { 1.0 } else { th[lay.lam[j][r - 1] + c % self.nvar] };
                d[(c, 0)] = l * th[p + c];
            }
            d[(k, 0)] = th[p + k + r];
            d
        } else {
            let g = Mat::from_column_slice(self.np[j], 1, &th[lay.g[j][r]..lay.g[j][r] + self.np[j]]);
            &self.vi[j] * g
        }
    }

    /// Regime parameters from the free vector; `None` outside the parameter space.
    pub fn regimes(&self, th: &[f64]) -> Option<Regimes> {
        let n = self.nvar;
        let lay = &self.layout;
        let mut zeta_j: Vec<Vec<f64>> = Vec::new(); // [j][k]
        for j in 0..n {
            let mut v = vec![1.0];
            for &o in &lay.z[j] {
                if !(th[o] > 0.0) || !th[o].is_finite() {
                    return None;
                }
                v.push(th[o]);
            }
            zeta_j.push(v);
        }
        let mut qc = vec![];
        for (c, ch) in self.chains.iter().enumerate() {
            let d: Vec<f64> = lay.q[c].iter().map(|&o| th[o]).collect();
            if d.iter().any(|x| !(*x >= 0.0 && *x <= 1.0)) {
                return None;
            }
            qc.push(chain_q(ch.n, &d));
        }
        let q = kron_q(&qc);
        let (mut a0s, mut aps, mut zs) = (vec![], vec![], vec![]);
        for s in 0..self.nstates {
            let mut a0 = Mat::zeros(n, n);
            let mut ap = Mat::zeros(self.ncoef, n);
            let mut zv = vec![];
            for j in 0..n {
                let r = self.coef_idx[j][s];
                let b = Mat::from_column_slice(self.n0[j], 1, &th[lay.b[j][r]..lay.b[j][r] + self.n0[j]]);
                let a0j = &self.ui[j] * b;
                let apj = self.aplus_dev(th, j, r) - &self.w[j] * &a0j;
                a0.set_column(j, &a0j.column(0));
                ap.set_column(j, &apj.column(0));
                zv.push(zeta_j[j][self.var_idx[j][s]]);
            }
            a0s.push(a0);
            aps.push(ap);
            zs.push(zv);
        }
        Some(Regimes { a0: a0s, aplus: aps, zeta: zs, q, qc })
    }

    /// `T x S` conditional log densities.
    pub fn log_densities(&self, r: &Regimes) -> Option<Mat> {
        let n = self.nvar as f64;
        let mut out = Mat::zeros(self.nobs, self.nstates);
        for s in 0..self.nstates {
            let ld = logabsdet(&r.a0[s]);
            if !ld.is_finite() {
                return None;
            }
            let c0 = ld + 0.5 * r.zeta[s].iter().map(|z| z.ln()).sum::<f64>() - 0.5 * n * LN2PI;
            let e = &self.y * &r.a0[s] - &self.x * &r.aplus[s];
            for t in 0..self.nobs {
                let mut q = 0.0;
                for j in 0..self.nvar {
                    q += r.zeta[s][j] * e[(t, j)] * e[(t, j)];
                }
                out[(t, s)] = c0 - 0.5 * q;
            }
        }
        Some(out)
    }

    pub fn filter(&self, th: &[f64]) -> Option<(Regimes, Filtered)> {
        let r = self.regimes(th)?;
        let l = self.log_densities(&r)?;
        let f = hamilton::filter(&l, &r.q);
        Some((r, f))
    }

    pub fn log_likelihood(&self, th: &[f64]) -> f64 {
        match self.filter(th) {
            Some((_, f)) if f.loglik.is_finite() => f.loglik,
            _ => f64::NEG_INFINITY,
        }
    }

    pub fn log_prior(&self, th: &[f64]) -> f64 {
        let lay = &self.layout;
        let mut lp = 0.0;
        for j in 0..self.nvar {
            // precision multiplied by the multiplicity, normalised
            let (m, lm) = (self.coef_mult[j], self.coef_mult[j].ln());
            for &o in &lay.b[j] {
                let b = Mat::from_column_slice(self.n0[j], 1, &th[o..o + self.n0[j]]);
                lp += -0.5 * m * (b.transpose() * &self.pb[j] * &b)[(0, 0)]
                    + 0.5 * (self.pb_ld[j] + self.n0[j] as f64 * lm)
                    - 0.5 * self.n0[j] as f64 * LN2PI;
            }
            for &o in &lay.g[j] {
                let g = Mat::from_column_slice(self.np[j], 1, &th[o..o + self.np[j]]);
                lp += -0.5 * m * (g.transpose() * &self.pg[j] * &g)[(0, 0)]
                    + 0.5 * (self.pg_ld[j] + self.np[j] as f64 * lm)
                    - 0.5 * self.np[j] as f64 * LN2PI;
            }
            if self.sims_zha() {
                lp += self.sz_log_prior(th, j);
            }
            for &o in &lay.z[j] {
                let z = th[o];
                if !(z > 0.0) {
                    return f64::NEG_INFINITY;
                }
                lp += self.zeta_a * self.zeta_b.ln() - gammaln(self.zeta_a) + (self.zeta_a - 1.0) * z.ln()
                    - self.zeta_b * z;
            }
        }
        for (c, ch) in self.chains.iter().enumerate() {
            for (i, &o) in lay.q[c].iter().enumerate() {
                let d = th[o];
                if !(d > 0.0 && d < 1.0) {
                    if d == 0.0 || d == 1.0 {
                        // boundary: finite only when the exponents vanish
                        let (a1, a2) = chain_alpha(ch, i);
                        if (d == 0.0 && a1 > 1.0) || (d == 1.0 && a2 > 1.0) {
                            return f64::NEG_INFINITY;
                        }
                        lp += gammaln(a1 + a2) - gammaln(a1) - gammaln(a2);
                        continue;
                    }
                    return f64::NEG_INFINITY;
                }
                let (a1, a2) = chain_alpha(ch, i);
                lp += gammaln(a1 + a2) - gammaln(a1) - gammaln(a2) + (a1 - 1.0) * d.ln() + (a2 - 1.0) * (1.0 - d).ln();
            }
        }
        lp
    }

    pub fn log_posterior(&self, th: &[f64]) -> f64 {
        let ll = self.log_likelihood(th);
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
        let lp = self.log_prior(th);
        if !lp.is_finite() { f64::NEG_INFINITY } else { ll + lp }
    }

    /// Free-vector entries that are `ln zeta` / logit `d` (for optimisation and sampling).
    fn unconstrained_kind(&self) -> Vec<u8> {
        let mut k = vec![0u8; self.layout.len];
        for v in &self.layout.z {
            for &o in v {
                k[o] = 1;
            }
        }
        for v in &self.layout.q {
            for &o in v {
                k[o] = 2;
            }
        }
        k
    }

    pub fn to_phi(&self, th: &[f64]) -> Vec<f64> {
        let kind = self.unconstrained_kind();
        th.iter()
            .enumerate()
            .map(|(i, &x)| match kind[i] {
                1 => x.ln(),
                2 => (x / (1.0 - x)).ln(),
                _ => x,
            })
            .collect()
    }

    pub fn from_phi(&self, phi: &[f64]) -> Vec<f64> {
        let kind = self.unconstrained_kind();
        phi.iter()
            .enumerate()
            .map(|(i, &x)| match kind[i] {
                1 => x.exp(),
                2 => 1.0 / (1.0 + (-x).exp()),
                _ => x,
            })
            .collect()
    }

    /// `ln |d theta / d phi|`.
    pub fn log_jacobian(&self, th: &[f64]) -> f64 {
        let kind = self.unconstrained_kind();
        th.iter()
            .enumerate()
            .map(|(i, &x)| match kind[i] {
                1 => x.ln(),
                2 => x.ln() + (1.0 - x).ln(),
                _ => 0.0,
            })
            .sum()
    }

    /// Sign normalisation: the first non-zero entry of `A0_j` (first regime) is positive.
    pub fn normalise_signs(&self, th: &mut [f64]) {
        for j in 0..self.nvar {
            let r0 = self.layout.b[j][0];
            let a0j = &self.ui[j] * Mat::from_column_slice(self.n0[j], 1, &th[r0..r0 + self.n0[j]]);
            let lead = (0..self.nvar).map(|i| a0j[(i, 0)]).find(|x| x.abs() > 1e-12).unwrap_or(1.0);
            if lead < 0.0 {
                for &o in &self.layout.b[j] {
                    for k in 0..self.n0[j] {
                        th[o + k] = -th[o + k];
                    }
                }
                for &o in &self.layout.g[j] {
                    for k in 0..self.np[j] {
                        th[o + k] = -th[o + k];
                    }
                }
                if self.sims_zha() {
                    let o = self.layout.psi[j];
                    for k in 0..self.ncoef - 1 + self.n_coef_reg[j] {
                        th[o + k] = -th[o + k];
                    }
                }
            }
        }
    }

    /// Precision of the Sims-Zha `psi_j = (lag coefficients, constant of every coefficient regime)`:
    /// `sum_r E_r' Hp^-1 E_r` with `E_r` picking the lags and the constant of regime `r` (pinned down by
    /// Dynare 7.2 prior values at hand-edited initial points).
    pub fn sz_psi_precision(&self, j: usize) -> Mat {
        let k = self.ncoef - 1;
        let h = self.n_coef_reg[j];
        let p = &self.pg[j];
        let mut m = Mat::zeros(k + h, k + h);
        for r in 0..h {
            for a in 0..=k {
                for b in 0..=k {
                    let ia = if a < k { a } else { k + r };
                    let ib = if b < k { b } else { k + r };
                    m[(ia, ib)] += p[(a, b)];
                }
            }
        }
        m
    }

    /// Sims-Zha prior of equation `j`: `psi_j ~ N(0, sz_psi_precision^-1)`, the scale factors
    /// `lambda_j(r)` (`r >= 1`) `N(0, lambda_var)` element by element.
    fn sz_log_prior(&self, th: &[f64], j: usize) -> f64 {
        let k = self.ncoef - 1;
        let h = self.n_coef_reg[j];
        let o = self.layout.psi[j];
        let x = Mat::from_column_slice(k + h, 1, &th[o..o + k + h]);
        // (multi-chain scaling as for b / g: not checked against Dynare)
        let m = self.coef_mult[j];
        let pm = self.sz_psi_precision(j) * m;
        let mut lp = -0.5 * (x.transpose() * &pm * &x)[(0, 0)] + 0.5 * logabsdet(&pm) - 0.5 * (k + h) as f64 * LN2PI;
        let lv = self.lambda_var / m;
        for &o in &self.layout.lam[j] {
            for i in 0..self.nvar {
                let x = th[o + i];
                lp += -0.5 * x * x / lv - 0.5 * (LN2PI + lv.ln());
            }
        }
        lp
    }

    /// Label of a free-vector entry (`est_flat_header` style is separate).
    pub fn kind_of_param(&self, i: usize) -> &'static str {
        let kind = self.unconstrained_kind();
        for j in 0..self.nvar {
            for &o in &self.layout.b[j] {
                if i >= o && i < o + self.n0[j] {
                    return "b";
                }
            }
            for &o in &self.layout.g[j] {
                if i >= o && i < o + self.np[j] {
                    return "g";
                }
            }
            for &o in &self.layout.lam[j] {
                if i >= o && i < o + self.nvar {
                    return "lambda";
                }
            }
            if let Some(&o) = self.layout.psi.get(j) {
                if i >= o && i < o + self.ncoef - 1 + self.n_coef_reg[j] {
                    return "psi";
                }
            }
        }
        match kind[i] {
            1 => "zeta",
            _ => "q",
        }
    }
}

// ---------------------------------------------------------------------------------------
// setup from the .mod declarations
// ---------------------------------------------------------------------------------------

fn controlled(eqs: &Option<super::SvarSpec>, j: usize) -> bool {
    match eqs {
        None => false,
        Some(s) => match &s.equations {
            Equations::All => true,
            Equations::List(l) => l.contains(&(j + 1)),
        },
    }
}

/// Options of `ms_estimation` besides the data options (`initialize_ms_sbvar_options.m`).
#[derive(Clone, Debug)]
pub struct MsOptions {
    pub alpha: f64,
    pub beta: f64,
    /// 0 = `none` (default prior), 1 = `sims_zha`, 2 = random walk (Dynare's default).
    pub specification: usize,
    pub no_create_init: bool,
    /// `gsig2_lmdm`: variance of the Sims-Zha lambda prior (Dynare default 50^2).
    pub gsig2_lmdm: f64,
    pub optim: super::ms_optim::OptimOptions,
}

impl MsOptions {
    pub fn from_options(o: &crate::modfile::Options) -> Result<MsOptions> {
        use crate::modfile::OptValue;
        let mut m = MsOptions {
            alpha: 1.0,
            beta: 1.0,
            specification: 2,
            no_create_init: false,
            gsig2_lmdm: 2500.0,
            optim: Default::default(),
        };
        for (k, v) in o {
            if k == "specification" {
                m.specification = match v {
                    OptValue::Num(x) => *x as usize,
                    OptValue::Str(s) => match s.trim_matches('\'').to_ascii_lowercase().as_str() {
                        "none" | "0" => 0,
                        "sims_zha" | "1" => 1,
                        "random_walk" | "2" => 2,
                        x => bail!("ms_estimation: unknown specification {x} (sims_zha or none)"),
                    },
                    _ => bail!("ms_estimation: specification expects sims_zha or none"),
                };
                continue;
            }
            let x = match v {
                OptValue::Num(x) => Some(*x),
                OptValue::Str(s) => s.parse().ok(),
                _ => None,
            };
            let op = &mut m.optim;
            match (k.as_str(), x) {
                ("alpha", Some(x)) => m.alpha = x,
                ("beta", Some(x)) => m.beta = x,
                ("gsig2_lmdm", Some(x)) => m.gsig2_lmdm = x,
                ("no_create_init", _) => m.no_create_init = !matches!(v, OptValue::Num(x) if *x == 0.0),
                ("convergence_starting_value", Some(x)) => op.cb = x,
                ("convergence_ending_value", Some(x)) => op.ce = x,
                ("convergence_increment_value", Some(x)) => op.ci = x,
                ("max_iterations_starting_value", Some(x)) => op.ib = x as usize,
                ("max_iterations_increment_value", Some(x)) => op.ii = x,
                ("max_block_iterations", Some(x)) => op.mb = x as usize,
                ("max_repeated_optimization_runs", Some(x)) => op.repeat_max = x as usize,
                ("function_convergence_criterion", Some(x)) => op.repeat_tol_obj = x,
                ("parameter_convergence_criterion", Some(x)) => op.repeat_tol_parms = x,
                ("number_of_large_perturbations", Some(x)) => op.random = x as usize,
                ("number_of_small_perturbations", Some(x)) => op.random_small = x as usize,
                ("number_of_posterior_draws_after_perturbation", Some(x)) => op.random_small_ndraws = x as usize,
                ("max_number_of_stages", Some(x)) => op.random_max = x as usize,
                ("random_function_convergence_criterion", Some(x)) => op.random_tol_obj = x,
                ("random_parameter_convergence_criterion", Some(x)) => op.random_tol_parms = x,
                _ => {}
            }
        }
        Ok(m)
    }
}

fn mixed_index(chains: &[MsChain], ctrl: &[bool], s: usize) -> (usize, usize) {
    // s = mixed radix over all chains (chain 1 slowest); returns (index over controlling chains, count)
    let n: Vec<usize> = chains.iter().map(|c| c.number_of_regimes).collect();
    let mut digits = vec![0; n.len()];
    let mut r = s;
    for k in (0..n.len()).rev() {
        digits[k] = r % n[k];
        r /= n[k];
    }
    let (mut idx, mut cnt) = (0, 1);
    for k in 0..n.len() {
        if ctrl[k] {
            idx = idx * n[k] + digits[k];
            cnt *= n[k];
        }
    }
    (idx, cnt)
}

/// The restrictions `ms_sbvar_setup.m` uses: identity bases under `contemp_reduced_form`.
pub fn effective_restrictions(restr: &Restrictions, settings: &SbvarSettings, nvar: usize) -> Restrictions {
    if settings.contemp_reduced_form {
        let ncoef = nvar * settings.nlags + 1;
        Restrictions {
            ui: vec![Mat::identity(nvar, nvar); nvar],
            vi: vec![Mat::identity(ncoef, ncoef); nvar],
            n0: vec![nvar; nvar],
            np: vec![ncoef; nvar],
            cross: vec![],
        }
    } else {
        restr.clone()
    }
}

impl MsSetup {
    pub fn build(
        z: &Mat,
        decl: &MsDecl,
        restr: &Restrictions,
        settings: &SbvarSettings,
        mo: &MsOptions,
        freq: (usize, usize, usize),
    ) -> Result<MsSetup> {
        let nvar = z.ncols();
        let lags = settings.nlags;
        if decl.chains.is_empty() {
            bail!("ms_estimation: no markov_switching chain declared");
        }
        for ch in &decl.chains {
            for spec in [&ch.svar_variances, &ch.svar_coefficients].into_iter().flatten() {
                if let Equations::List(l) = &spec.equations {
                    if let Some(e) = l.iter().find(|&&e| e < 1 || e > nvar) {
                        bail!(
                            "svar: equation {e} of chain {} is outside 1..{nvar} (the number of observed variables)",
                            ch.number
                        );
                    }
                }
            }
        }
        let restr = effective_restrictions(restr, settings, nvar);
        let ncoef = nvar * lags + 1;
        let nobs = z.nrows() - lags;
        let y = Mat::from_fn(nobs, nvar, |t, v| z[(lags + t, v)]);
        let x = Mat::from_fn(
            nobs,
            ncoef,
            |t, c| {
                if c == ncoef - 1 { 1.0 } else { z[(lags + t - (c / nvar + 1), c % nvar)] }
            },
        );
        let (_pi, h0i, hpi) = sbvar::rnrprior(nvar, settings.freq, lags, z, &settings.mu)?;
        // the printed prior is a variance matrix
        let h0 = h0i.try_inverse().ok_or_else(|| anyhow!("ms_estimation: singular A0 prior"))?;
        let hp = hpi.clone().try_inverse().ok_or_else(|| anyhow!("ms_estimation: singular A+ prior"))?;
        let mut chains = vec![];
        for c in &decl.chains {
            let n = c.number_of_regimes;
            let mut alpha = Mat::from_element(n, n, 1.0);
            for i in 0..n {
                let p = 1.0 - 1.0 / c.durations[i];
                alpha[(i, i)] = p * (n as f64 - 1.0) / (1.0 - p);
            }
            chains.push(ChainSpec { n, alpha });
        }
        let nstates: usize = decl.chains.iter().map(|c| c.number_of_regimes).product();
        let (mut coef_idx, mut var_idx) = (vec![], vec![]);
        for j in 0..nvar {
            let cc: Vec<bool> = decl.chains.iter().map(|c| controlled(&c.svar_coefficients, j)).collect();
            let vc: Vec<bool> = decl.chains.iter().map(|c| controlled(&c.svar_variances, j)).collect();
            coef_idx.push((0..nstates).map(|s| mixed_index(&decl.chains, &cc, s).0).collect::<Vec<_>>());
            var_idx.push((0..nstates).map(|s| mixed_index(&decl.chains, &vc, s).0).collect::<Vec<_>>());
        }
        // A+_j = V_j g - W_j A0_j: W = -[I; 0] (random walk, Sims-Zha); the default specification uses
        // W_j = -V_j (V_j' Hp^-1 V_j)^-1 V_j' Hp [I; 0] with Hp the prior variance (what Dynare 7.2's mex writes
        // to init_*.dat: checked against it, Hp enters where the conditional prior mean would have Hp^-1).
        let s = Mat::from_fn(ncoef, nvar, |i, k| if i == k { 1.0 } else { 0.0 });
        let mut w = vec![];
        for j in 0..nvar {
            if mo.specification == 0 {
                let v = &restr.vi[j];
                let hpinv = &hpi;
                let m = (v.transpose() * hpinv * v)
                    .try_inverse()
                    .ok_or_else(|| anyhow!("ms_estimation: singular A+ prior"))?;
                w.push(-(v * m * v.transpose() * &hp * &s));
            } else {
                w.push(-s.clone());
            }
        }
        if mo.specification == 1 {
            for j in 0..nvar {
                if restr.np[j] != ncoef || (&restr.vi[j] - Mat::identity(ncoef, ncoef)).abs().max() > 1e-12 {
                    bail!(
                        "ms_estimation: specification=sims_zha with restrictions on A+ (Dynare 7.2's mex crashes on this case)"
                    );
                }
            }
        }
        Self::assemble(MsParts {
            y,
            x,
            restr,
            w,
            h0var: vec![h0; nvar],
            hpvar: vec![hp; nvar],
            chains,
            coef_idx,
            var_idx,
            zeta_a: mo.alpha,
            zeta_b: mo.beta,
            specification: mo.specification,
            lambda_var: mo.gsig2_lmdm,
            nlags: lags,
            freq,
        })
    }

    /// Completes a setup from its primitive parts (from the .mod file or an `init_*.dat` file).
    pub fn assemble(p: MsParts) -> Result<MsSetup> {
        let nvar = p.y.ncols();
        let ncoef = p.x.ncols();
        let restr = p.restr;
        let (mut pb, mut pg, mut pb_ld, mut pg_ld) = (vec![], vec![], vec![], vec![]);
        for j in 0..nvar {
            // precisions U' H0^-1 U, V' Hp^-1 V (conditional form: the second derivatives of Dynare's prior
            // at hand-edited initial points)
            let h0i = p.h0var[j].clone().try_inverse().ok_or_else(|| anyhow!("ms_estimation: singular A0 prior"))?;
            let hpi = p.hpvar[j].clone().try_inverse().ok_or_else(|| anyhow!("ms_estimation: singular A+ prior"))?;
            let b = restr.ui[j].transpose() * h0i * &restr.ui[j];
            let g = restr.vi[j].transpose() * hpi * &restr.vi[j];
            pb_ld.push(logabsdet(&b));
            pg_ld.push(logabsdet(&g));
            pb.push(b);
            pg.push(g);
        }
        let nstates = p.chains.iter().map(|c| c.n).product::<usize>();
        let ncr: Vec<usize> = p.coef_idx.iter().map(|v| v.iter().max().map_or(1, |m| m + 1)).collect();
        let nvr: Vec<usize> = p.var_idx.iter().map(|v| v.iter().max().map_or(1, |m| m + 1)).collect();
        let layout = Self::make_layout(&restr.n0, &restr.np, &ncr, &nvr, &p.chains, p.specification == 1);
        let combos = |t: &Vec<Vec<usize>>| {
            let mut seen: Vec<Vec<usize>> = (0..nstates).map(|s| t.iter().map(|r| r[s]).collect()).collect();
            seen.sort();
            seen.dedup();
            seen.len() as f64
        };
        let cc = combos(&p.coef_idx);
        let coef_mult: Vec<f64> = ncr.iter().map(|&n| cc / n as f64).collect();
        Ok(MsSetup {
            nvar,
            nlags: p.nlags,
            ncoef,
            nobs: p.y.nrows(),
            y: p.y,
            x: p.x,
            ui: restr.ui,
            vi: restr.vi,
            n0: restr.n0,
            np: restr.np,
            pb,
            pg,
            pb_ld,
            pg_ld,
            chains: p.chains,
            nstates,
            coef_idx: p.coef_idx,
            var_idx: p.var_idx,
            n_coef_reg: ncr,
            n_var_reg: nvr,
            zeta_a: p.zeta_a,
            zeta_b: p.zeta_b,
            layout,
            freq: p.freq.0,
            initial_year: p.freq.1,
            initial_subperiod: p.freq.2,
            w: p.w,
            specification: p.specification,
            h0var: p.h0var,
            hpvar: p.hpvar,
            lambda_var: p.lambda_var,
            coef_mult,
        })
    }

    /// Starting point of `ms_estimation`: the constant-parameter posterior mode in every regime,
    /// `zeta = 1`, `Q` from the durations.
    pub fn start_value(
        &self,
        z: &Mat,
        settings: &SbvarSettings,
        restr: &Restrictions,
        _decl: &MsDecl,
    ) -> Result<Vec<f64>> {
        let res = sbvar::sbvar(z, settings, restr)?;
        let mut a0 = Mat::zeros(self.nvar, self.nvar);
        let mut off = 0;
        for j in 0..self.nvar {
            let b = Mat::from_column_slice(self.n0[j], 1, &res.xhat[off..off + self.n0[j]]);
            off += self.n0[j];
            a0.set_column(j, &(&self.ui[j] * b).column(0));
        }
        let q: Vec<Mat> = self
            .chains
            .iter()
            .map(|ch| {
                // Dynare starts from the prior mean of the Dirichlet distributions
                let d: Vec<f64> = (0..chain_nfree(ch.n))
                    .map(|i| {
                        let (a1, a2) = chain_alpha(ch, i);
                        a1 / (a1 + a2)
                    })
                    .collect();
                chain_q(ch.n, &d)
            })
            .collect();
        let a0s = vec![a0; self.nstates];
        let aps = vec![res.aplus.clone(); self.nstates];
        let zs = vec![vec![1.0; self.nvar]; self.nstates];
        Ok(self.theta_from_regimes(&a0s, &aps, &zs, &q))
    }

    /// Free parameters from the regime matrices (inverse of `regimes`; least squares where the
    /// matrices do not satisfy the restrictions exactly).
    pub fn theta_from_regimes(&self, a0: &[Mat], aplus: &[Mat], zeta: &[Vec<f64>], qc: &[Mat]) -> Vec<f64> {
        let lay = &self.layout;
        let mut th = vec![0.0; lay.len];
        for j in 0..self.nvar {
            let k = self.ncoef - 1;
            let mut psi_set = false;
            for r in 0..self.n_coef_reg[j] {
                let Some(s) = (0..self.nstates).find(|&s| self.coef_idx[j][s] == r) else { continue };
                let a0j = a0[s].column(j).into_owned();
                let b = self.ui[j].transpose() * &a0j;
                th[lay.b[j][r]..lay.b[j][r] + self.n0[j]].copy_from_slice(b.as_slice());
                let a0j = &self.ui[j] * b;
                let dev = aplus[s].column(j).into_owned() + &self.w[j] * &a0j;
                if self.sims_zha() {
                    let p = lay.psi[j];
                    if !psi_set {
                        for c in 0..k {
                            th[p + c] = dev[c];
                        }
                        psi_set = true;
                    }
                    th[p + k + r] = dev[k];
                    if r > 0 {
                        for i in 0..self.nvar {
                            let (mut num, mut den) = (0.0, 0.0);
                            for c in (i..k).step_by(self.nvar) {
                                num += th[p + c] * dev[c];
                                den += th[p + c] * th[p + c];
                            }
                            th[lay.lam[j][r - 1] + i] = if den > 0.0 { num / den } else { 1.0 };
                        }
                    }
                } else {
                    let g = self.vi[j].transpose() * dev;
                    th[lay.g[j][r]..lay.g[j][r] + self.np[j]].copy_from_slice(g.as_slice());
                }
            }
            for (v, &o) in lay.z[j].iter().enumerate() {
                if let Some(s) = (0..self.nstates).find(|&s| self.var_idx[j][s] == v + 1) {
                    th[o] = zeta[s][j];
                }
            }
        }
        for (c, ch) in self.chains.iter().enumerate() {
            for (i, &o) in lay.q[c].iter().enumerate() {
                let (d1rows, _, nr) = chain_columns(ch.n, i);
                th[o] = d1rows
                    .iter()
                    .take(nr)
                    .map(|&(r, wgt)| if wgt > 0.0 { qc[c][(r, i)] / wgt } else { 0.0 })
                    .sum::<f64>()
                    / nr as f64;
            }
        }
        th
    }
}

/// Primitive parts of a [`MsSetup`].
pub struct MsParts {
    pub y: Mat,
    pub x: Mat,
    pub restr: Restrictions,
    pub w: Vec<Mat>,
    pub h0var: Vec<Mat>,
    pub hpvar: Vec<Mat>,
    pub chains: Vec<ChainSpec>,
    pub coef_idx: Vec<Vec<usize>>,
    pub var_idx: Vec<Vec<usize>>,
    pub zeta_a: f64,
    pub zeta_b: f64,
    pub specification: usize,
    pub lambda_var: f64,
    pub nlags: usize,
    pub freq: (usize, usize, usize),
}

// ---------------------------------------------------------------------------------------
// optimisation
// ---------------------------------------------------------------------------------------

fn num_grad(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let mut g = vec![0.0; x.len()];
    let mut xp = x.to_vec();
    for i in 0..x.len() {
        let h = 1e-6 * (1.0 + x[i].abs());
        xp[i] = x[i] + h;
        let fp = f(&xp);
        xp[i] = x[i] - h;
        let fm = f(&xp);
        xp[i] = x[i];
        g[i] = if fp.is_finite() && fm.is_finite() { (fp - fm) / (2.0 * h) } else { 0.0 };
    }
    g
}

/// BFGS with a backtracking line search and a central-difference gradient; minimises `f`.
pub fn bfgs(f: &dyn Fn(&[f64]) -> f64, x0: Vec<f64>, max_iter: usize) -> (Vec<f64>, f64, usize) {
    bfgs_crit(f, x0, max_iter, 0.0)
}

/// [`bfgs`] that also stops (csminwel-like) once an iteration improves `f` by less than `crit`.
pub fn bfgs_crit(f: &dyn Fn(&[f64]) -> f64, x0: Vec<f64>, max_iter: usize, crit: f64) -> (Vec<f64>, f64, usize) {
    let n = x0.len();
    if n == 0 {
        let v = f(&x0);
        return (x0, v, 0);
    }
    let mut x = x0;
    let mut fx = f(&x);
    let mut gx = num_grad(f, &x);
    let mut h = Mat::identity(n, n);
    let mut small = 0;
    let mut slow = 0;
    let mut it = 0;
    while it < max_iter {
        let gn = gx.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if gn < 1e-8 || !fx.is_finite() {
            break;
        }
        let gv = Mat::from_column_slice(n, 1, &gx);
        let mut d = -(&h * &gv);
        let mut slope = (gv.transpose() * &d)[(0, 0)];
        if slope >= 0.0 {
            h = Mat::identity(n, n);
            d = -gv.clone();
            slope = (gv.transpose() * &d)[(0, 0)];
        }
        let mut t = 1.0;
        let maxstep = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if maxstep > 2.0 {
            t = 2.0 / maxstep;
        }
        let (mut ok, mut xn, mut fnew) = (false, x.clone(), fx);
        for _ in 0..60 {
            xn = (0..n).map(|i| x[i] + t * d[(i, 0)]).collect();
            fnew = f(&xn);
            if fnew.is_finite() && fnew <= fx + 1e-4 * t * slope {
                ok = true;
                break;
            }
            t *= 0.5;
        }
        it += 1;
        if !ok {
            if h != Mat::identity(n, n) {
                h = Mat::identity(n, n);
                continue;
            }
            break;
        }
        let gnew = num_grad(f, &xn);
        let s = Mat::from_fn(n, 1, |i, _| xn[i] - x[i]);
        let y = Mat::from_fn(n, 1, |i, _| gnew[i] - gx[i]);
        let sy = (s.transpose() * &y)[(0, 0)];
        if sy > 1e-12 {
            let rho = 1.0 / sy;
            if it == 1 {
                let yy = (y.transpose() * &y)[(0, 0)];
                h = Mat::identity(n, n) * (sy / yy);
            }
            let l = Mat::identity(n, n) - &s * y.transpose() * rho;
            h = &l * &h * l.transpose() + &s * s.transpose() * rho;
        } else {
            h = Mat::identity(n, n);
        }
        if (fx - fnew).abs() < 1e-14 * fx.abs().max(1.0) {
            small += 1;
        } else {
            small = 0;
        }
        let gain = fx - fnew;
        x = xn;
        fx = fnew;
        gx = gnew;
        slow = if crit > 0.0 && gain < crit { slow + 1 } else { 0 };
        if small >= 3 || slow >= 3 {
            break;
        }
    }
    (x, fx, it)
}

/// The result of `ms_estimation`, with everything the later `ms_*` commands need.
#[derive(Clone)]
pub struct MsEstimate {
    pub setup: Arc<MsSetup>,
    pub theta: Vec<f64>,
    pub log_posterior: f64,
    pub log_likelihood: f64,
    pub log_prior: f64,
    pub iterations: usize,
}

impl MsEstimate {
    pub fn regimes(&self) -> Regimes {
        self.setup.regimes(&self.theta).expect("mode inside the parameter space")
    }
    pub fn filtered(&self) -> (Regimes, Filtered) {
        self.setup.filter(&self.theta).expect("mode inside the parameter space")
    }
}

/// Maximises the posterior from `start` (several restarts of BFGS in unconstrained coordinates).
pub fn estimate(setup: Arc<MsSetup>, start: Vec<f64>) -> Result<MsEstimate> {
    let obj = |phi: &[f64]| {
        let th = setup.from_phi(phi);
        let v = setup.log_posterior(&th);
        if v.is_finite() { -v } else { f64::INFINITY }
    };
    let mut phi = setup.to_phi(&start);
    if !obj(&phi).is_finite() {
        bail!("ms_estimation: the starting value is outside the parameter space");
    }
    let mut total = 0;
    let mut last = f64::INFINITY;
    for _ in 0..6 {
        let (p, f, it) = bfgs(&obj, phi, 3000);
        phi = p;
        total += it;
        if (last - f).abs() < 1e-11 * f.abs().max(1.0) {
            last = f;
            break;
        }
        last = f;
    }
    let _ = last;
    let mut theta = setup.from_phi(&phi);
    setup.normalise_signs(&mut theta);
    let ll = setup.log_likelihood(&theta);
    let lp = setup.log_prior(&theta);
    Ok(MsEstimate { setup, theta, log_posterior: ll + lp, log_likelihood: ll, log_prior: lp, iterations: total })
}

// ---------------------------------------------------------------------------------------
// output files
// ---------------------------------------------------------------------------------------

fn fmt_row_e(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x:.6e}")).collect::<Vec<_>>().join(" ")
}

/// Writes `est_free_<tag>.out`, `est_flat_header_<tag>.out`, `est_flat_<tag>.out` (the files
/// `set_ms_estimation_file.m` reads) next to the .mod file.
pub fn write_estimation_files(dir: &std::path::Path, tag: &str, e: &MsEstimate) -> Result<()> {
    let mut free = vec![e.log_posterior, e.log_likelihood];
    free.extend_from_slice(&e.theta);
    std::fs::write(dir.join(format!("est_free_{tag}.out")), fmt_row_e(&free) + " \n")?;
    let (head, vals) = flat_row(&e.setup, &e.theta).expect("mode inside the parameter space");
    std::fs::write(dir.join(format!("est_flat_header_{tag}.out")), head.join(" ") + " \n")?;
    std::fs::write(
        dir.join(format!("est_flat_{tag}.out")),
        vals.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(" ") + " \n",
    )?;
    Ok(())
}

/// Header labels and values of one `est_flat` / `draws` line (Q, then per regime A0, A+, Zeta).
pub fn flat_row(s: &MsSetup, theta: &[f64]) -> Option<(Vec<String>, Vec<f64>)> {
    let r = s.regimes(theta)?;
    let multi = s.chains.len() > 1;
    let (mut head, mut vals) = (vec![], vec![]);
    for (c, q) in r.qc.iter().enumerate() {
        for j in 0..q.ncols() {
            for i in 0..q.nrows() {
                head.push(if multi {
                    format!("Q[{}]({},{})", c + 1, i + 1, j + 1)
                } else {
                    format!("Q({},{})", i + 1, j + 1)
                });
                vals.push(q[(i, j)]);
            }
        }
    }
    for st in 0..s.nstates {
        for j in 0..s.nvar {
            for i in 0..s.nvar {
                head.push(format!("A0[{}]({},{})", st + 1, i + 1, j + 1));
                vals.push(r.a0[st][(i, j)]);
            }
        }
        for j in 0..s.nvar {
            for i in 0..s.ncoef {
                head.push(format!("Aplus[{}]({},{})", st + 1, i + 1, j + 1));
                vals.push(r.aplus[st][(i, j)]);
            }
        }
        for j in 0..s.nvar {
            head.push(format!("Zeta[{}]({},{})", st + 1, j + 1, j + 1));
            vals.push(r.zeta[st][j]);
        }
    }
    Some((head, vals))
}

pub fn mat_rows(m: &Mat) -> Value {
    json!((0..m.nrows()).map(|i| m.row(i).iter().copied().collect::<Vec<f64>>()).collect::<Vec<_>>())
}

/// `<fname>/Output/ms_estimation.json`: the structures later `ms_*` commands (and users) need.
pub fn estimate_json(e: &MsEstimate) -> Value {
    let r = e.regimes();
    let s = &e.setup;
    json!({
        "nvar": s.nvar, "nlags": s.nlags, "ncoef": s.ncoef, "nobs": s.nobs, "nstates": s.nstates,
        "log_posterior": e.log_posterior, "log_likelihood": e.log_likelihood, "log_prior": e.log_prior,
        "free_parameters": e.theta,
        "A0": r.a0.iter().map(mat_rows).collect::<Vec<_>>(),
        "Aplus": r.aplus.iter().map(mat_rows).collect::<Vec<_>>(),
        "zeta": r.zeta,
        "Q": mat_rows(&r.q),
        "Q_chains": r.qc.iter().map(mat_rows).collect::<Vec<_>>(),
        "coefficient_regime": s.coef_idx, "variance_regime": s.var_idx,
        "layout": {"b": s.layout.b, "g": s.layout.g, "zeta": s.layout.z, "q": s.layout.q},
    })
}
