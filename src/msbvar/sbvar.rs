// SPDX-License-Identifier: GPL-3.0-or-later
//! `sbvar`: the constant-parameter Bayesian structural VAR with the Sims-Zha prior
//! (`ms_mardd.m` -> `msstart2.m` with `fn_dataxy`, `fn_rnrprior_covres_dobs`, `fn_rlrprior`,
//! `fn_rlrpostr` / `fn_dlrpostr`, `fn_a0freefun` / `fn_a0freegrad` maximised by `csminwel`,
//! `fn_gfmean`, `fn_impulse`, `fn_forecast`).
//!
//! Model: `y_t' A0 = sum_l y_{t-l}' A_l + c + e_t'`, `e_t ~ N(0, I)`; column `j` of `A0` and of
//! `A+ = [A_1; ...; A_p; c]` belongs to equation `j`, restricted as `a_j = U_j b_j`,
//! `a+_j = V_j g_j`. The posterior mode of `A0` maximises
//! `-fss*log|det A0| + 1/2 sum_j b_j' H0inv_j b_j` (reported as `fhat`), `A+` is the
//! conditional posterior mean `g_j = P_j b_j`.

use anyhow::{Result, bail};

use super::identification::Restrictions;
use crate::linalg::Mat;

/// Everything `sbvar` takes besides the restrictions.
#[derive(Clone, Debug)]
pub struct SbvarSettings {
    pub freq: usize,
    pub nlags: usize,
    /// `coefficients_prior_hyperparameters` (`mu`).
    pub mu: [f64; 6],
    /// `false` for `no_bayesian_prior`.
    pub bayesian_prior: bool,
    pub contemp_reduced_form: bool,
    /// Impulse response horizon (`imstp = 4*freq`).
    pub irf_steps: usize,
    /// Number of forecast periods (`nfqm = forecast_years*freq`).
    pub forecast_steps: usize,
}

#[derive(Clone, Debug)]
pub struct SbvarResult {
    pub nvar: usize,
    pub ncoef: usize,
    /// Effective sample size (`fss`).
    pub fss: usize,
    pub a0: Mat,
    pub a0_inv: Mat,
    /// `Fhat`: `A+` (`ncoef x nvar`).
    pub aplus: Mat,
    /// Reduced-form lag coefficients `Bhat = A+ inv(A0)`.
    pub bhat: Mat,
    /// Free parameters of `A0` at the mode (`xhat`, in this implementation's unit-vector basis).
    pub xhat: Vec<f64>,
    /// `fn_a0freefun` at the mode.
    pub fhat: f64,
    pub grad: Vec<f64>,
    pub iterations: usize,
    pub fcount: usize,
    /// Per-equation posterior matrices in the restriction bases: `P` (`np x n0`), `H0inv`
    /// (`n0 x n0`), `Hpinv` (`np x np`). With cross restrictions `P` and `H0inv` are the
    /// restricted ones (`g = P b` on the restriction set; `H0inv` includes the added quadratic).
    pub pmat: Vec<Mat>,
    pub h0inv: Vec<Mat>,
    pub hpinv: Vec<Mat>,
    pub ui: Vec<Mat>,
    pub vi: Vec<Mat>,
    /// `imfhat`: `irf_steps x nvar^2`, row `t`, column `r + nvar*c` is element `(r, c)` of
    /// `Phi_t' ` as `fn_impulse` (`Phi_0 = inv(A0)`).
    pub imf: Mat,
    /// Unconditional point forecast, `forecast_steps x nvar`.
    pub forecast: Mat,
    /// Bayesian prior only: `log p(a0hat, a+hat)`, `log p(Y|a0hat, a+hat)`, `logMarLHres`,
    /// `log p(a+hat | Y, a0hat)`. (The Gibbs-sampled `p(a0_k|Y, a_others)` terms of
    /// `ms_mardd.m` are in `sbvar_extra`.) `None` with cross restrictions.
    pub vlog_a0p: Option<f64>,
    pub vlog_y_a: Option<f64>,
    pub log_mar_lh_res: Option<f64>,
    pub vlog_ap_ya0: Option<f64>,
}

fn diag(v: &[f64]) -> Mat {
    Mat::from_fn(v.len(), v.len(), |i, j| if i == j { v[i] } else { 0.0 })
}

fn inv(m: &Mat) -> Result<Mat> {
    m.clone().try_inverse().ok_or_else(|| anyhow::anyhow!("sbvar: singular matrix"))
}

fn logabsdet(m: &Mat) -> f64 {
    let lu = m.clone().lu();
    lu.u().diagonal().iter().map(|x| x.abs().ln()).sum()
}

/// `fn_dataxy(nvar, lags, z, mu, 0, 1)`: `(xtx, xty, yty, phi, y)`.
fn dataxy(nvar: usize, lags: usize, z: &Mat) -> (Mat, Mat, Mat, Mat, Mat) {
    let ns = z.nrows();
    let fss = ns - lags;
    let ncoef = nvar * lags + 1;
    let phi = Mat::from_fn(fss, ncoef, |t, c| {
        if c == ncoef - 1 {
            1.0
        } else {
            let (l, v) = (c / nvar, c % nvar); // lag l+1
            z[(lags + t - (l + 1), v)]
        }
    });
    let y = Mat::from_fn(fss, nvar, |t, v| z[(lags + t, v)]);
    (phi.transpose() * &phi, phi.transpose() * &y, y.transpose() * &y, phi, y)
}

/// Residual variance of the univariate AR(`lags`) with constant of every series (`sye`).
fn ar_residual_variances(z: &Mat, lags: usize) -> Vec<f64> {
    let ns = z.nrows();
    (0..z.ncols())
        .map(|k| {
            let zk = z.columns(k, 1).into_owned();
            let (xtx, xty, _, phi, y) = dataxy(1, lags, &zk);
            let b = xtx.lu().solve(&xty).expect("AR normal equations");
            let e = &y - &phi * b;
            (e.transpose() * &e)[(0, 0)] / (ns - lags) as f64
        })
        .collect()
}

/// `fn_rnrprior_covres_dobs` (no asymmetric weights, `nexo = 1`, `indxmsmdeqn = 0`):
/// `(Pi, H0inv_multi, Hpinv_multi)` (the same matrices for every equation).
pub fn rnrprior(nvar: usize, q_m: usize, lags: usize, z: &Mat, mu: &[f64; 6]) -> Result<(Mat, Mat, Mat)> {
    let ncoef = nvar * lags + 1;
    let pi = Mat::from_fn(ncoef, nvar, |i, j| if i == j { 1.0 } else { 0.0 });
    let (mut a, mut b) = (0.0, 0.0);
    if q_m == 12 {
        let xx1 = 1.0f64;
        let xx2 = 1.0 / ((lags as f64 / 3.0).ceil()).powf(mu[3]);
        b = if lags == 1 { 0.0 } else { (xx1.ln() - xx2.ln()) / (1.0 - lags as f64) };
        a = xx1 * (-b).exp();
    } else if q_m != 4 {
        bail!("sbvar: incompatibility with lags: freq must be 4 or 12 for the Bayesian prior");
    }
    let sgsh = ar_residual_variances(z, lags);
    let sg0bid: Vec<f64> = sgsh.iter().map(|s| 1.0 / s).collect();
    let mut sgpbid = vec![0.0; ncoef];
    for i in 1..=lags {
        let lagdecay = if q_m == 12 { a * (b * i as f64 * mu[3]).exp() } else { 1.0 / (i as f64).powf(mu[3]) };
        for j in 0..nvar {
            sgpbid[(i - 1) * nvar + j] = lagdecay * lagdecay / sgsh[j];
        }
    }
    // dummy observations (explicit prior)
    let ndobs = nvar + 1;
    let mut phibar = Mat::zeros(ndobs, ncoef);
    phibar[(ndobs - 1, ncoef - 1)] = 1.0;
    let init: Vec<f64> = (0..nvar).map(|k| (0..lags).map(|t| z[(t, k)]).sum::<f64>() / lags as f64).collect();
    for k in 0..nvar {
        for m in 0..lags {
            phibar[(ndobs - 1, nvar * m + k)] = init[k];
            phibar[(k, nvar * m + k)] = init[k];
        }
    }
    for c in 0..ncoef {
        for k in 0..nvar {
            phibar[(k, c)] *= mu[4];
        }
        phibar[(ndobs - 1, c)] *= mu[5];
    }
    let xtxbar = phibar.transpose() * &phibar;
    let sg0bida: Vec<f64> = sg0bid.iter().map(|s| mu[0] * mu[0] * s).collect();
    let mut sgpbida: Vec<f64> = sgpbid.iter().map(|s| mu[0] * mu[0] * mu[1] * mu[1] * s).collect();
    sgpbida[ncoef - 1] = mu[0] * mu[0] * mu[2] * mu[2];
    // Hptdinv: diag(1/sgpbida) for the lag blocks (A1 block uses sgpbida, deeper lags sgppbd =
    // sgpbida(nvar+1:end), which are the same numbers shifted), constant 1/sgppbd(end).
    let mut hptdinv = Mat::zeros(ncoef, ncoef);
    for i in 0..nvar.min(ncoef) {
        hptdinv[(i, i)] = 1.0 / sgpbida[i];
    }
    if lags > 1 {
        for i in nvar..ncoef - 1 {
            hptdinv[(i, i)] = 1.0 / sgpbida[i]; // sgppbd(1:ncoef-nvar-1) = sgpbida(nvar+1:ncoef-1)
        }
    }
    hptdinv[(ncoef - 1, ncoef - 1)] = 1.0 / sgpbida[ncoef - 1];
    let hpinv = hptdinv + xtxbar;
    let h0inv = diag(&sg0bida.iter().map(|s| 1.0 / s).collect::<Vec<_>>());
    Ok((pi, h0inv, hpinv))
}

struct A0Problem<'a> {
    ui: &'a [Mat],
    n0: &'a [usize],
    nvar: usize,
    fss: f64,
    h0inv: &'a [Mat],
}

impl A0Problem<'_> {
    fn a0(&self, b: &[f64]) -> Mat {
        let mut a0 = Mat::zeros(self.nvar, self.nvar);
        let mut off = 0;
        for k in 0..self.nvar {
            let bj = Mat::from_column_slice(self.n0[k], 1, &b[off..off + self.n0[k]]);
            a0.set_column(k, &(&self.ui[k] * bj).column(0));
            off += self.n0[k];
        }
        a0
    }

    /// `fn_a0freefun`
    fn f(&self, b: &[f64]) -> f64 {
        let a0 = self.a0(b);
        let mut tra = 0.0;
        let mut off = 0;
        for k in 0..self.nvar {
            let bj = Mat::from_column_slice(self.n0[k], 1, &b[off..off + self.n0[k]]);
            tra += 0.5 * (bj.transpose() * &self.h0inv[k] * &bj)[(0, 0)];
            off += self.n0[k];
        }
        -self.fss * logabsdet(&a0) + tra
    }

    /// `fn_a0freegrad`
    fn g(&self, b: &[f64]) -> Vec<f64> {
        let a0 = self.a0(b);
        let bmat = a0.transpose().try_inverse().unwrap_or_else(|| Mat::zeros(self.nvar, self.nvar));
        let mut g = Vec::with_capacity(b.len());
        let mut off = 0;
        for k in 0..self.nvar {
            let bj = Mat::from_column_slice(self.n0[k], 1, &b[off..off + self.n0[k]]);
            let hb = &self.h0inv[k] * &bj;
            let corr = self.ui[k].transpose() * bmat.column(k);
            for i in 0..self.n0[k] {
                g.push(hb[(i, 0)] - self.fss * corr[i]);
            }
            off += self.n0[k];
        }
        g
    }
}

/// BFGS with backtracking and the analytic gradient; returns `(x, f, grad, iterations, fcount)`.
fn minimize(p: &A0Problem, x0: Vec<f64>) -> (Vec<f64>, f64, Vec<f64>, usize, usize) {
    let n = x0.len();
    let mut x = x0;
    let mut fx = p.f(&x);
    let mut gx = p.g(&x);
    let mut h = Mat::identity(n, n);
    let (mut it, mut fcount) = (0, 1);
    while it < 20000 {
        let gn = gx.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if gn < 1e-11 || !fx.is_finite() {
            break;
        }
        let gv = Mat::from_column_slice(n, 1, &gx);
        let d = -(&h * &gv);
        let slope = (gv.transpose() * &d)[(0, 0)];
        let (mut t, mut ok) = (1.0, false);
        let mut xn = x.clone();
        let mut fnew = fx;
        for _ in 0..80 {
            xn = (0..n).map(|i| x[i] + t * d[(i, 0)]).collect();
            fnew = p.f(&xn);
            fcount += 1;
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
        let gnew = p.g(&xn);
        let s = Mat::from_fn(n, 1, |i, _| xn[i] - x[i]);
        let y = Mat::from_fn(n, 1, |i, _| gnew[i] - gx[i]);
        let sy = (s.transpose() * &y)[(0, 0)];
        if sy > 1e-14 {
            let rho = 1.0 / sy;
            let i_n = Mat::identity(n, n);
            let l = &i_n - &s * y.transpose() * rho;
            h = &l * &h * l.transpose() + &s * s.transpose() * rho;
        } else {
            h = Mat::identity(n, n);
        }
        let done = (fx - fnew).abs() < 1e-15 * fx.abs().max(1.0) && gn < 1e-8;
        x = xn;
        fx = fnew;
        gx = gnew;
        if done {
            break;
        }
    }
    (x, fx, gx, it, fcount)
}

/// Deterministic starting value: `A0 = I` projected on the free entries, perturbed until
/// invertible.
fn start_value(ui: &[Mat], n0: &[usize], nvar: usize) -> Vec<f64> {
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64) / ((1u64 << 31) as f64)
    };
    for attempt in 0..50 {
        let mut x = Vec::new();
        for k in 0..nvar {
            for i in 0..n0[k] {
                let proj = ui[k][(k, i)]; // I(:,k)' * Ui(:,i)
                x.push(proj + if attempt == 0 { 0.0 } else { 0.5 * rnd() });
            }
        }
        let p = A0Problem { ui, n0, nvar, fss: 1.0, h0inv: &[] };
        if logabsdet(&p.a0(&x)).is_finite() && p.a0(&x).determinant().abs() > 1e-8 {
            return x;
        }
    }
    vec![1.0; n0.iter().sum()]
}

/// `fn_impulse`: `steps x nvar^2`.
fn impulse(bh: &Mat, swish: &Mat, nvar: usize, lags: usize, steps: usize) -> Mat {
    let ah = bh.transpose(); // nvar x ncoef
    let mut ms: Vec<Mat> = vec![swish.transpose()]; // M_0
    for t in 1..steps {
        let mut m = Mat::zeros(nvar, nvar);
        for s in 1..=t.min(lags) {
            m += ah.columns((s - 1) * nvar, nvar) * &ms[t - s];
        }
        ms.push(m);
    }
    Mat::from_fn(steps, nvar * nvar, |t, c| ms[t][(c % nvar, c / nvar)])
}

/// Runs `sbvar` on the sample `z` (`nobs x nvar`, columns in `varobs` order).
pub fn sbvar(z: &Mat, s: &SbvarSettings, r: &Restrictions) -> Result<SbvarResult> {
    let nvar = z.ncols();
    let lags = s.nlags;
    if lags < 1 {
        bail!("sbvar: nlags must be a positive integer");
    }
    if s.freq < 1 {
        bail!("sbvar: freq must be a positive integer");
    }
    if z.nrows() <= lags + 1 {
        bail!("sbvar: too few observations");
    }
    let ncoef = nvar * lags + 1;
    let (xtx, xty, yty, phi, y) = dataxy(nvar, lags, z);
    let fss = phi.nrows();
    let (mut ui, mut vi, mut n0, mut np) = (r.ui.clone(), r.vi.clone(), r.n0.clone(), r.np.clone());
    if s.contemp_reduced_form {
        ui = vec![Mat::identity(nvar, nvar); nvar];
        vi = vec![Mat::identity(ncoef, ncoef); nvar];
        n0 = vec![nvar; nvar];
        np = vec![ncoef; nvar];
    }
    let (mut ptld, mut h0invtld, mut hpinvtld) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pmat, mut h0inv, mut hpinv) = (Vec::new(), Vec::new(), Vec::new());
    if s.bayesian_prior {
        let (pi, h0i, hpi) = rnrprior(nvar, s.freq, lags, z, &s.mu)?;
        // fn_rlrprior (Hmulti \ X = Hinvmulti * X)
        for k in 0..nvar {
            let hpt = vi[k].transpose() * &hpi * &vi[k];
            let pt = inv(&hpt)? * vi[k].transpose() * &hpi * &pi * &ui[k];
            let h0t = ui[k].transpose() * &h0i * &ui[k] + ui[k].transpose() * pi.transpose() * &hpi * &pi * &ui[k]
                - pt.transpose() * &hpt * &pt;
            ptld.push(pt);
            h0invtld.push(h0t);
            hpinvtld.push(hpt);
        }
        // fn_rlrpostr
        for k in 0..nvar {
            let hp = vi[k].transpose() * &xtx * &vi[k] + &hpinvtld[k];
            let p1 = vi[k].transpose() * &xty * &ui[k] + &hpinvtld[k] * &ptld[k];
            let p = inv(&hp)? * &p1;
            let h0 = ui[k].transpose() * &yty * &ui[k] + &h0invtld[k] + ptld[k].transpose() * &hpinvtld[k] * &ptld[k]
                - p1.transpose() * &p;
            pmat.push(p);
            h0inv.push(h0);
            hpinv.push(hp);
        }
    } else {
        // fn_dlrpostr
        for k in 0..nvar {
            let hp = vi[k].transpose() * &xtx * &vi[k];
            let p1 = vi[k].transpose() * &xty * &ui[k];
            let p = inv(&hp)? * &p1;
            let h0 = ui[k].transpose() * &yty * &ui[k] - p1.transpose() * &p;
            pmat.push(p);
            h0inv.push(h0);
            hpinv.push(hp);
        }
    }
    // Cross A0-A+ restrictions `R a+_j + Q a_j = 0`, i.e. `L g_j = M b_j` with `L = R V_j`,
    // `M = -Q U_j`. Given b_j, g_j | Y ~ N(P b_j, Hpinv^-1); on the affine set the conditional
    // mode is `P b + S L'(L S L')^-1 D b` (S = Hpinv^-1, D = M - L P; the formula of
    // msstart2.m's cross-restriction branch), and integrating g_j over it adds
    // `1/2 b' D'(L S L')^-1 D b` to the A0 objective (exact joint posterior mode, whereas
    // msstart2.m would only adjust A+ after the unrestricted A0 mode).
    let cross_restr = !s.contemp_reduced_form && !r.cross.is_empty();
    if cross_restr {
        for c in &r.cross {
            let j = c.equation;
            let l = &c.r * &vi[j];
            let m = -(&c.q * &ui[j]);
            let sigma = inv(&hpinv[j])?;
            let lsl = &l * &sigma * l.transpose();
            let lsl_inv = lsl
                .try_inverse()
                .ok_or_else(|| anyhow::anyhow!("sbvar: the cross restrictions of equation {} are redundant or only involve excluded A+ coefficients", j + 1))?;
            let d = &m - &l * &pmat[j];
            h0inv[j] += d.transpose() * &lsl_inv * &d;
            pmat[j] = &pmat[j] + &sigma * l.transpose() * &lsl_inv * &d;
        }
    }
    // posterior mode of A0
    let (a0, a0_inv, xhat, fhat, grad, iterations, fcount, aplus, bhat);
    if s.contemp_reduced_form {
        let c = (&h0inv[0] / fss as f64)
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("sbvar: H0inv is not positive definite"))?;
        a0_inv = c.l().transpose(); // chol(.) upper triangular
        a0 = inv(&a0_inv)?;
        bhat = pmat[0].clone();
        aplus = &bhat * &a0;
        xhat = a0.iter().copied().collect();
        (fhat, grad, iterations, fcount) = (f64::NAN, Vec::new(), 0, 0);
    } else {
        let prob = A0Problem { ui: &ui, n0: &n0, nvar, fss: fss as f64, h0inv: &h0inv };
        let (x, f, g, it, fc) = minimize(&prob, start_value(&ui, &n0, nvar));
        a0 = prob.a0(&x);
        a0_inv = inv(&a0)?;
        // fn_gfmean
        let mut ap = Mat::zeros(ncoef, nvar);
        let mut off = 0;
        for k in 0..nvar {
            let bj = Mat::from_column_slice(n0[k], 1, &x[off..off + n0[k]]);
            ap.set_column(k, &(&vi[k] * &pmat[k] * bj).column(0));
            off += n0[k];
        }
        bhat = &ap * &a0_inv;
        aplus = ap;
        xhat = x;
        (fhat, grad, iterations, fcount) = (f, g, it, fc);
    }
    let imf = impulse(&bhat, &a0_inv, nvar, lags, s.irf_steps);
    // fn_forecast from the last observation
    let mut ph: Vec<f64> = phi.row(fss - 1).iter().copied().collect();
    let tcwx = nvar * lags;
    for i in (nvar..tcwx).rev() {
        ph[i] = ph[i - nvar];
    }
    for v in 0..nvar {
        ph[v] = y[(fss - 1, v)];
    }
    let mut forecast = Mat::zeros(s.forecast_steps, nvar);
    for k in 0..s.forecast_steps {
        let row = Mat::from_row_slice(1, ncoef, &ph) * &bhat;
        for v in 0..nvar {
            forecast[(k, v)] = row[(0, v)];
        }
        for i in (nvar..tcwx).rev() {
            ph[i] = ph[i - nvar];
        }
        for v in 0..nvar {
            ph[v] = row[(0, v)];
        }
    }
    // pieces of the marginal data density available without the Gibbs sampler
    let (mut vlog_a0p, mut vlog_y_a, mut lmlr, mut vlog_ap_ya0) = (None, None, None, None);
    if s.bayesian_prior && !s.contemp_reduced_form && !cross_restr {
        let two_pi = (2.0 * std::f64::consts::PI).ln();
        let (mut a0p, mut yexpt, mut apexpt, mut totp, mut hsum) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for k in 0..nvar {
            let a0k = a0.column(k).into_owned();
            let apk = aplus.column(k).into_owned();
            let bk = ui[k].transpose() * &a0k;
            let gk = vi[k].transpose() * &apk;
            let gbar = &ptld[k] * &bk;
            yexpt -= 0.5 * (a0k.dot(&(&yty * &a0k)) - 2.0 * apk.dot(&(&xty * &a0k)) + apk.dot(&(&xtx * &apk)));
            let dg = &gk - &gbar;
            a0p += -0.5 * (n0[k] + np[k]) as f64 * two_pi
                + 0.5 * h0invtld[k].determinant().abs().ln()
                + 0.5 * hpinvtld[k].determinant().abs().ln()
                - 0.5 * (bk.dot(&(&h0invtld[k] * &bk)) + dg.dot(&(&hpinvtld[k] * &dg)));
            let tmpd = &gk - &pmat[k] * &bk;
            apexpt -= 0.5 * tmpd.dot(&(&hpinv[k] * &tmpd));
            totp += np[k] as f64;
            hsum += 0.5 * hpinv[k].determinant().abs().ln();
        }
        let ldet = logabsdet(&a0);
        vlog_a0p = Some(a0p);
        vlog_y_a = Some(-0.5 * (nvar * fss) as f64 * two_pi + fss as f64 * ldet + yexpt);
        let mut l = 0.0;
        for t in 0..fss {
            let e = y.row(t) * &a0 - phi.row(t) * &aplus;
            l += -0.5 * nvar as f64 * two_pi + ldet - 0.5 * e.norm_squared();
        }
        lmlr = Some(l);
        vlog_ap_ya0 = Some(-0.5 * totp * two_pi + hsum + apexpt);
    }
    Ok(SbvarResult {
        nvar,
        ncoef,
        fss,
        a0,
        a0_inv,
        aplus,
        bhat,
        xhat,
        fhat,
        grad,
        iterations,
        fcount,
        pmat,
        h0inv,
        hpinv,
        ui,
        vi,
        imf,
        forecast,
        vlog_a0p,
        vlog_y_a,
        log_mar_lh_res: lmlr,
        vlog_ap_ya0,
    })
}
