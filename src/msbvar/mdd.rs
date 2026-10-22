// SPDX-License-Identifier: GPL-3.0-or-later
//! Posterior sampling and the log marginal data density of the Markov-switching SBVAR
//! (`ms_compute_mdd`): Müller, Meng-Wong bridge and Waggoner-Zha estimates of `ln p(Y)`, ported from
//! `dw_mdd_switch.c`, `dw_elliptical.c`, `dw_dirichlet_restrictions.c` and `sbvar_mdd.c` of Dynare 7.2.
//!
//! Proposal (`dw_marginal_data_density_command_line`): the first `percentage_draws = 20%` of the posterior
//! draws fit it, the remaining 80% are the posterior sample of the estimators. It is the product of an
//! elliptical density on `theta` (every free parameter but the transition probabilities) and independent
//! Dirichlet densities on the free transition probabilities `q`:
//! - centre = the draw with the highest log posterior among the fitting draws (Dynare's `-use_mean` flag never
//!   reaches the C code, which looks for `-mean`, so `use_mean_center` has no effect in Dynare either);
//!   scale `S` = the second moment about the centre (divisor n);
//! - radii `r = sqrt((theta - c)' S^-1 (theta - c))` of the fitting draws, bounds
//!   `0.5 (R[i] + R[i+1])` with `i = floor(p (n - 1))` for `p = proposal_lower/upper_bound`;
//! - `proposal_type` 1 Gaussian, 2 power (`[0, b]`), 3 truncated power, 4 step (equal-probability table of
//!   at most 30 steps), 5 truncated Gaussian (whose density, as in Dynare, is not set to zero outside its
//!   bounds), 6 uniform on the annulus; the power of 2 and 3 makes `E[r^2] / dim = 1`
//!   (`GetPower_truncatedpower`);
//! - Dirichlet parameters `alpha_j = 1.5 m mu_j + max_j(1.1 - m mu_j)` (`PosteriorDistributionInfo`), with
//!   `mu` the mean of the fitting draws and `m = max_j(0, mu_j (1 - mu_j) / (s_j - mu_j^2)) - 1`, where
//!   Dynare's `s_j` is the last fitting draw divided by n (`ProductVS(b_square, b, 1/n)`, a slip for the
//!   mean square that makes `m = -1` in practice); reproduced as is.
//!
//! Normalisation (`sbvar_mdd.c` switches on the Waggoner-Zha sign normalisation with the `A0` of the mode as
//! target): the log posterior of every *proposal* draw is `-inf` outside the normalised region and carries
//! `nvars ln 2` (`VARbase.c`), while the *posterior* side takes column 0 of `simulation_<tag>.out`, written
//! without normalisation. Dynare's estimate thus mixes two kernels and lands between the one-mode integral and
//! the true mdd (hence its run-to-run spread). `consistent = true` applies the normalisation on both sides,
//! which gives the true `ln p(Y)` of the sign-symmetric posterior.
//!
//! With Dynare's own draws, its proposal parameters are reproduced to the printed digits and, given its
//! proposal matrix, its Müller / bridge / Waggoner-Zha values to 1e-4 (tests/msbvar_ms.rs); only the random
//! numbers of the proposal draws differ.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};

use super::ms_estimation::MsSetup;
use crate::estimation::special::{chi2inv, gammainc, gammaln};
use crate::linalg::Mat;

const LN2PI: f64 = 1.8378770664093453;

fn randn(rng: &mut ChaCha8Rng) -> f64 {
    rng.sample(StandardNormal)
}

/// Random-walk Metropolis sampler for the posterior (in unconstrained coordinates `ln zeta`,
/// logit `d`, with the log-Jacobian in the target, so the draws are posterior draws of the free
/// parameters). As in Dynare's C sampler, the proposal is tuned first: `adaptive` calibration draws starting
/// from the inverse Hessian at `start`, adapting the covariance on their history; then the proposal is fixed
/// for the `drop` burn-in draws and the `replic * thin` kept ones. Returns `(draws, log posterior)` after
/// `drop` and thinning.
pub fn rwmh(
    setup: &MsSetup,
    start: &[f64],
    replic: usize,
    drop: usize,
    thin: usize,
    adaptive: usize,
    seed: u64,
) -> (Mat, Vec<f64>) {
    let n = start.len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let target = |phi: &[f64]| {
        let th = setup.from_phi(phi);
        let v = setup.log_posterior(&th);
        if v.is_finite() { v + setup.log_jacobian(&th) } else { f64::NEG_INFINITY }
    };
    let mut phi = setup.to_phi(start);
    // inverse Hessian of -target
    let nt = |p: &[f64]| -target(p);
    let mut hess = Mat::zeros(n, n);
    for i in 0..n {
        let h = 1e-4 * (1.0 + phi[i].abs());
        let (mut pp, mut pm) = (phi.clone(), phi.clone());
        pp[i] += h;
        pm[i] -= h;
        let gp = grad(&nt, &pp);
        let gm = grad(&nt, &pm);
        for j in 0..n {
            hess[(i, j)] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    let hess = (&hess + hess.transpose()) * 0.5;
    let mut cov = hess
        .clone()
        .try_inverse()
        .filter(|c| c.clone().cholesky().is_some())
        .unwrap_or_else(|| Mat::identity(n, n) * 1e-2);
    let scale = 2.38 * 2.38 / n as f64;
    let mut chol = (&cov * scale).cholesky().map(|c| c.l()).unwrap_or_else(|| Mat::identity(n, n) * 0.1);
    let mut cur = target(&phi);
    let mut hist: Vec<Vec<f64>> = Vec::new();
    let total = adaptive + drop + replic * thin;
    let mut draws = Mat::zeros(replic, n);
    let mut lps = Vec::with_capacity(replic);
    let mut kept = 0;
    for it in 0..total {
        let z: Vec<f64> = (0..n).map(|_| randn(&mut rng)).collect();
        let step = &chol * Mat::from_column_slice(n, 1, &z);
        let cand: Vec<f64> = (0..n).map(|i| phi[i] + step[(i, 0)]).collect();
        let tc = target(&cand);
        if tc.is_finite() && rng.r#gen::<f64>().ln() < tc - cur {
            phi = cand;
            cur = tc;
        }
        if it < adaptive {
            hist.push(phi.clone());
            if (it + 1) % 500 == 0 && hist.len() >= 200 {
                let m = hist.len() / 2;
                let h = &hist[m..];
                let mean: Vec<f64> = (0..n).map(|i| h.iter().map(|r| r[i]).sum::<f64>() / h.len() as f64).collect();
                let mut c = Mat::zeros(n, n);
                for r in h {
                    for i in 0..n {
                        for j in 0..n {
                            c[(i, j)] += (r[i] - mean[i]) * (r[j] - mean[j]) / h.len() as f64;
                        }
                    }
                }
                let c = c + Mat::identity(n, n) * 1e-10;
                if let Some(l) = (&c * scale).cholesky() {
                    cov = c;
                    chol = l.l();
                }
            }
            continue;
        }
        let it = it - adaptive;
        if it >= drop && (it - drop) % thin == thin - 1 && kept < replic {
            let th = setup.from_phi(&phi);
            for i in 0..n {
                draws[(kept, i)] = th[i];
            }
            lps.push(setup.log_posterior(&th));
            kept += 1;
        }
    }
    let _ = cov;
    (draws, lps)
}

fn grad(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let mut g = vec![0.0; x.len()];
    let mut xp = x.to_vec();
    for i in 0..x.len() {
        let h = 1e-6 * (1.0 + x[i].abs());
        xp[i] = x[i] + h;
        let fp = f(&xp);
        xp[i] = x[i] - h;
        let fm = f(&xp);
        xp[i] = x[i];
        g[i] = (fp - fm) / (2.0 * h);
    }
    g
}

/// Waggoner-Zha sign normalisation of `sbvar_mdd.c` (`IsNormalized_VAR`, `SetFlipped_VAR`) with the
/// target `A0` of the mode: for each equation `j` and coefficient regime `k`, the signs of
/// `row_j(A0(s)^-1) . target_j(s)` summed over the regimes `s` with that `A0` column must be positive
/// (or zero and `target_j . A0_j >= 0`).
pub fn wz_normalized(setup: &MsSetup, target: &[Mat], th: &[f64]) -> bool {
    let Some(r) = setup.regimes(th) else { return false };
    let n = setup.nvar;
    let mut flipped: Vec<Vec<i64>> = (0..n).map(|j| vec![0; setup.n_coef_reg[j]]).collect();
    let mut dots: Vec<Vec<f64>> = (0..n).map(|j| vec![0.0; setup.n_coef_reg[j]]).collect();
    for s in 0..setup.nstates {
        let Some(inv) = r.a0[s].clone().try_inverse() else { return false };
        for j in 0..n {
            let k = setup.coef_idx[j][s];
            let x: f64 = (0..n).map(|m| inv[(j, m)] * target[s][(m, j)]).sum();
            if x < 0.0 {
                flipped[j][k] -= 1;
            } else if x > 0.0 {
                flipped[j][k] += 1;
            }
            dots[j][k] = (0..n).map(|m| target[s][(m, j)] * r.a0[s][(m, j)]).sum();
        }
    }
    (0..n).all(|j| (0..setup.n_coef_reg[j]).all(|k| flipped[j][k] > 0 || (flipped[j][k] == 0 && dots[j][k] >= 0.0)))
}

#[derive(Clone, Debug)]
pub struct MddOptions {
    /// `-pt`: 1 Gaussian, 2 power, 3 truncated power, 4 step, 5 truncated Gaussian, 6 uniform.
    pub proposal_type: usize,
    pub proposal_draws: usize,
    pub lower: f64,
    pub upper: f64,
    /// Kept for the option of the same name; Dynare 7.2 ignores it (see the module docs), so do we.
    pub use_mean_center: bool,
    /// Share of the posterior draws (the first ones) that fit the proposal (`-percentage_draws`, 0.2).
    pub percentage_draws: f64,
    /// Apply the sign normalisation to the posterior draws too (true mdd) instead of only to the
    /// proposal draws as Dynare does.
    pub consistent: bool,
}

impl Default for MddOptions {
    fn default() -> Self {
        MddOptions {
            proposal_type: 3,
            proposal_draws: 100000,
            lower: 0.1,
            upper: 0.9,
            use_mean_center: false,
            percentage_draws: 0.2,
            consistent: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MddResult {
    pub muller: f64,
    pub bridge: f64,
    /// Waggoner-Zha estimate without cut (first line of Dynare's list).
    pub wz: f64,
    /// Dimension of the elliptical part (theta).
    pub dim: usize,
    pub radius_bounds: (f64, f64),
    pub n_posterior: usize,
    pub n_proposal: usize,
    /// proposal draws with finite posterior kernel
    pub n_valid: usize,
    /// Müller's `in_P1`, `in_P2`.
    pub in_p: (usize, usize),
    /// `PrintEllipticalInfo` of the proposal (header of `mdd_t<type>_<tag>.out`).
    pub proposal_info: String,
    pub alpha: Vec<f64>,
}

fn logaddexp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if !m.is_finite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Radial law of the elliptical proposal (`dw_elliptical.c`).
#[derive(Clone, Debug)]
pub enum Radial {
    /// Gaussian (`hi = inf`, `lo = 0`) and truncated Gaussian: chi-square cdf at `lo^2` and the mass.
    Gaussian {
        lo: f64,
        hi: f64,
        cdf_lo: f64,
        prob: f64,
        truncated: bool,
    },
    /// Truncated power (power: `lo = 0`).
    Power {
        lo: f64,
        hi: f64,
        power: f64,
        lower_only: bool,
    },
    Uniform {
        lo: f64,
        hi: f64,
    },
    Step {
        t: Vec<f64>,
    },
}

/// Elliptical density `Gamma(d/2) / (2 pi^(d/2) |det scale|) f(r) / r^(d-1)`, `r` the Mahalanobis radius.
#[derive(Clone, Debug)]
pub struct Elliptical {
    pub dim: usize,
    pub center: Vec<f64>,
    /// `L L' = S`.
    pub chol: Mat,
    /// `-ln |det scale| = -ln det L`.
    pub logabsdet: f64,
    pub radial: Radial,
}

fn chi2_cdf(x: f64, d: usize) -> f64 {
    gammainc(x / 2.0, d as f64 / 2.0)
}

fn variance_scale_truncated_power(dim: usize, lo: f64, hi: f64, p: f64) -> f64 {
    p * (hi.powf(p + 2.0) - lo.powf(p + 2.0)) / (dim as f64 * (p + 2.0) * (hi.powf(p) - lo.powf(p)))
}

/// `GetPower_truncatedpower`: the power with `E[r^2] = dim`.
pub fn power_truncated_power(dim: usize, lo_b: f64, hi_b: f64) -> f64 {
    let vs = |p: f64| variance_scale_truncated_power(dim, lo_b, hi_b, p);
    let (mut lo, mut hi) = (0.0001, 1.0);
    if vs(lo) > 1.0 {
        return lo;
    }
    for _ in 0..20 {
        if vs(hi) < 1.0 {
            lo = hi;
            hi *= 2.0;
        } else {
            let mut mid = 0.0;
            for _ in 0..30 {
                mid = 0.5 * (lo + hi);
                if vs(mid) < 1.0 { lo = mid } else { hi = mid }
            }
            return mid;
        }
    }
    hi
}

impl Elliptical {
    pub fn radius(&self, th: &[f64]) -> f64 {
        let z = Mat::from_fn(self.dim, 1, |i, _| th[i] - self.center[i]);
        self.chol.solve_lower_triangular(&z).unwrap().norm()
    }

    fn ln_surface(&self) -> f64 {
        let d = self.dim as f64;
        (0.5f64).ln() + gammaln(0.5 * d) - 0.5 * d * std::f64::consts::PI.ln() + self.logabsdet
    }

    pub fn ln_density_radius(&self, r: f64) -> f64 {
        let d = self.dim as f64;
        match &self.radial {
            // no truncation test: logdensity_radius_truncated_gaussian
            Radial::Gaussian { prob, .. } => -0.5 * r * r - 0.5 * d * LN2PI + self.logabsdet - prob.ln(),
            Radial::Power { lo, hi, power, .. } => {
                if r <= *lo || r > *hi {
                    f64::NEG_INFINITY
                } else {
                    let mult = hi.powf(*power) - lo.powf(*power);
                    (power - d) * r.ln() + (0.5 * power).ln() + gammaln(0.5 * d)
                        - 0.5 * d * std::f64::consts::PI.ln()
                        - mult.ln()
                        + self.logabsdet
                }
            }
            Radial::Uniform { lo, hi } => {
                if r < *lo || r > *hi {
                    f64::NEG_INFINITY
                } else {
                    let mult = hi.powf(d) - lo.powf(d);
                    (0.5 * d).ln() + gammaln(0.5 * d) - 0.5 * d * std::f64::consts::PI.ln() - mult.ln() + self.logabsdet
                }
            }
            Radial::Step { t } => {
                let m = t.len() - 1;
                if r < t[0] || r > t[m] {
                    return f64::NEG_INFINITY;
                }
                let (mut min, mut max) = (0, m);
                while max - min > 1 {
                    let mid = (min + max) / 2;
                    if r > t[mid] { min = mid } else { max = mid }
                }
                (1.0 - d) * r.ln() - (m as f64 * (t[max] - t[min])).ln() + self.ln_surface()
            }
        }
    }

    /// A draw and its radius.
    pub fn draw(&self, rng: &mut ChaCha8Rng) -> (Vec<f64>, f64) {
        let d = self.dim;
        let (mut z, mut s);
        loop {
            z = (0..d).map(|_| randn(rng)).collect::<Vec<f64>>();
            s = z.iter().map(|x| x * x).sum::<f64>().sqrt();
            if s > 0.0 {
                break;
            }
        }
        let r = match &self.radial {
            Radial::Gaussian { truncated: false, .. } => s,
            Radial::Gaussian { cdf_lo, prob, .. } => chi2inv(cdf_lo + rng.r#gen::<f64>() * prob, d as f64).sqrt(),
            Radial::Power { lo, hi, power, .. } => {
                let shift = lo.powf(*power);
                ((hi.powf(*power) - shift) * rng.r#gen::<f64>() + shift).powf(1.0 / power)
            }
            Radial::Uniform { lo, hi } => {
                let shift = lo.powf(d as f64);
                ((hi.powf(d as f64) - shift) * rng.r#gen::<f64>() + shift).powf(1.0 / d as f64)
            }
            Radial::Step { t } => {
                let m = t.len() - 1;
                let i = (rng.r#gen::<f64>() * m as f64).floor() as usize;
                if i < m { t[i] + rng.r#gen::<f64>() * (t[i + 1] - t[i]) } else { t[m] }
            }
        };
        let w = Mat::from_fn(d, 1, |i, _| z[i] * r / s);
        let lz = &self.chol * w;
        ((0..d).map(|i| self.center[i] + lz[(i, 0)]).collect(), r)
    }

    pub fn bounds(&self) -> (f64, f64) {
        match &self.radial {
            Radial::Gaussian { lo, hi, .. } | Radial::Power { lo, hi, .. } | Radial::Uniform { lo, hi } => (*lo, *hi),
            Radial::Step { t } => (t[0], t[t.len() - 1]),
        }
    }

    /// `PrintEllipticalInfo`.
    pub fn info(&self) -> String {
        use super::ms_optim::fmt_g;
        let d = self.dim;
        match &self.radial {
            Radial::Gaussian { truncated: false, .. } => format!("gaussian\n dim={d}\n"),
            Radial::Gaussian { lo, hi, .. } => {
                format!("truncated gaussian\n dim={d}\n lower bound={}\n upper bound={}\n", fmt_g(*lo), fmt_g(*hi))
            }
            Radial::Power { hi, power, lower_only: true, .. } => {
                format!("power\n dim={d}\n upper bound={}\n power={}\n", fmt_g(*hi), fmt_g(*power))
            }
            Radial::Power { lo, hi, power, .. } => {
                format!(
                    "truncated power\n dim={d}\n lower bound={}\n upper bound={}\n power={}\n",
                    fmt_g(*lo),
                    fmt_g(*hi),
                    fmt_g(*power)
                )
            }
            Radial::Uniform { lo, hi } => {
                format!("uniform\n dim={d}\n lower bound={}\n upper bound={}\n", fmt_g(*lo), fmt_g(*hi))
            }
            Radial::Step { t } => {
                let tab: String = t.iter().map(|x| format!("{} ", fmt_g(*x))).collect();
                format!(
                    "step\n dim={d}\n lower bound={}\n upper bound={}\n table size={}\n table={tab}\n",
                    fmt_g(t[0]),
                    fmt_g(t[t.len() - 1]),
                    t.len() - 1
                )
            }
        }
    }
}

/// `LogDirichlet_pdf` at the `n - 1` free elements `q`.
fn ln_dirichlet_pdf(q: &[f64], alpha: &[f64]) -> f64 {
    let n = alpha.len();
    let (mut sum_q, mut sum_alpha, mut log_pdf) = (1.0, alpha[n - 1], -gammaln(alpha[n - 1]));
    let mut i = n - 1;
    while i > 0 {
        i -= 1;
        if q[i] > 0.0 {
            sum_q -= q[i];
            sum_alpha += alpha[i];
            log_pdf += (alpha[i] - 1.0) * q[i].ln() - gammaln(alpha[i]);
        } else if alpha[i] != 1.0 {
            let mut lk = alpha[i] - 1.0;
            loop {
                if q[i] > 0.0 {
                    sum_q -= q[i];
                } else {
                    lk += alpha[i] - 1.0;
                }
                if i == 0 {
                    break;
                }
                i -= 1;
            }
            if sum_q <= 0.0 {
                lk += alpha[n - 1] - 1.0;
            }
            return if lk > 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
        }
    }
    if sum_q > 0.0 {
        log_pdf + (alpha[n - 1] - 1.0) * sum_q.ln() + gammaln(sum_alpha)
    } else if alpha[n - 1] == 1.0 {
        log_pdf + gammaln(sum_alpha)
    } else if alpha[n - 1] > 1.0 {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    }
}

/// The proposal of `dw_marginal_data_density_command_line`: elliptical on theta x independent Dirichlet on q.
#[derive(Clone, Debug)]
pub struct Proposal {
    pub theta_idx: Vec<usize>,
    pub q_idx: Vec<usize>,
    /// Dimensions of the independent Dirichlet distributions (each free `q` is the first element of a
    /// 2-dimensional one in this port's Markov-chain design).
    pub dirichlet_dims: Vec<usize>,
    pub alpha: Vec<f64>,
    pub elliptical: Elliptical,
}

impl Proposal {
    /// `PosteriorDistributionInfo` + `PosteriorRadius` + `CreateEllipticalFromPosterior_*` on the first
    /// `ninfo` rows of `draws` (`lps`: their log posterior, column 0 of the simulation file).
    pub fn fit(setup: &MsSetup, draws: &Mat, lps: &[f64], ninfo: usize, opts: &MddOptions) -> anyhow::Result<Proposal> {
        let n = ninfo;
        let q_idx: Vec<usize> = setup.layout.q.iter().flatten().copied().collect();
        let theta_idx: Vec<usize> = (0..draws.ncols()).filter(|i| !q_idx.contains(i)).collect();
        let dim = theta_idx.len();
        if n < 2 || dim == 0 {
            anyhow::bail!("ms_compute_mdd: too few posterior draws ({n}) to form the proposal");
        }
        let th = |r: usize| -> Vec<f64> { theta_idx.iter().map(|&j| draws[(r, j)]).collect() };
        // centre: the mode of the fitting draws (USE_MODE; see module docs for USE_MEAN)
        let (mut best, mut imax) = (f64::NEG_INFINITY, 0);
        for (r, &lp) in lps.iter().enumerate().take(n) {
            if lp > best {
                best = lp;
                imax = r;
            }
        }
        let center = th(imax);
        let mut s = Mat::zeros(dim, dim);
        for r in 0..n {
            let t = th(r);
            for i in 0..dim {
                for j in 0..dim {
                    s[(i, j)] += (t[i] - center[i]) * (t[j] - center[j]);
                }
            }
        }
        let s = s / n as f64;
        let chol = s
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("Not enough variation in the posterior draws to form proposal"))?
            .l();
        let logabsdet = -(0..dim).map(|i| chol[(i, i)].ln()).sum::<f64>();
        let mut ell = Elliptical {
            dim,
            center,
            chol,
            logabsdet,
            radial: Radial::Gaussian { lo: 0.0, hi: f64::INFINITY, cdf_lo: 0.0, prob: 1.0, truncated: false },
        };
        let mut rad: Vec<f64> = (0..n).map(|r| ell.radius(&th(r))).collect();
        rad.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| -> f64 {
            let i = (p * (n - 1) as f64).floor() as usize;
            if i < n - 1 { 0.5 * (rad[i + 1] + rad[i]) } else { rad[i] }
        };
        ell.radial = match opts.proposal_type {
            1 => ell.radial.clone(),
            2 => {
                let b = q(opts.upper);
                Radial::Power { lo: 0.0, hi: b, power: power_truncated_power(dim, 0.0, b), lower_only: true }
            }
            3 => {
                let (a, b) = (q(opts.lower), q(opts.upper));
                if b <= a {
                    anyhow::bail!("ms_compute_mdd: empty radius band [{a}, {b}]");
                }
                Radial::Power { lo: a, hi: b, power: power_truncated_power(dim, a, b), lower_only: false }
            }
            4 => {
                let (pl, ph) = (opts.lower, opts.upper);
                let mut table = vec![];
                let mut k = 30;
                while k >= 1 {
                    let inc = (ph - pl) / k as f64;
                    table.clear();
                    let mut x = pl;
                    for _ in 0..=k {
                        let y = x * (n - 1) as f64;
                        let i = y.floor() as usize;
                        table.push(if i >= n - 1 {
                            rad[n - 1]
                        } else {
                            (1.0 - y + i as f64) * rad[i] + (y - i as f64) * rad[i + 1]
                        });
                        x += inc;
                    }
                    if (1..=k).all(|j| table[j] > table[j - 1]) {
                        break;
                    }
                    k -= 1;
                }
                if k == 0 {
                    anyhow::bail!("Not enough variation in the posterior draws to form proposal");
                }
                Radial::Step { t: table }
            }
            5 => {
                let (a, b) = (q(opts.lower), q(opts.upper));
                let cdf_lo = if a > 0.0 { chi2_cdf(a * a, dim) } else { 0.0 };
                let prob = chi2_cdf(b * b, dim) - cdf_lo;
                if prob <= 0.0 {
                    anyhow::bail!("Truncated gaussian defined only on positive probability regions");
                }
                Radial::Gaussian { lo: a, hi: b, cdf_lo, prob, truncated: true }
            }
            6 => {
                let (a, b) = (q(opts.lower), q(opts.upper));
                Radial::Uniform { lo: a, hi: b }
            }
            t => anyhow::bail!("ms_compute_mdd: unknown proposal type {t} (1 to 6)"),
        };
        // Dirichlet parameters (alpha_scale = 1.5)
        let dirichlet_dims = vec![2; q_idx.len()];
        let mut alpha = vec![0.0; 2 * q_idx.len()];
        for (k, &o) in q_idx.iter().enumerate() {
            let b = |r: usize| -> [f64; 2] {
                let x = draws[(r, o)];
                [x, if 1.0 - x > 0.0 { 1.0 - x } else { 0.0 }]
            };
            let mut mean = [0.0; 2];
            for r in 0..n {
                let v = b(r);
                mean[0] += v[0];
                mean[1] += v[1];
            }
            let mean = [mean[0] / n as f64, mean[1] / n as f64];
            // Dynare: b_square = (last fitting draw) / n
            let last = b(n - 1);
            let sq = [last[0] / n as f64, last[1] / n as f64];
            let mut mx = 0.0;
            for j in (0..2).rev() {
                let t = mean[j] * (1.0 - mean[j]) / (sq[j] - mean[j] * mean[j]);
                if t > mx {
                    mx = t;
                }
            }
            mx -= 1.0;
            let mut inc = 0.0;
            for j in (0..2).rev() {
                let t = 1.1 - mx * mean[j];
                if t > inc {
                    inc = t;
                }
            }
            for j in 0..2 {
                alpha[2 * k + j] = 1.5 * mx * mean[j] + inc;
            }
        }
        Ok(Proposal { theta_idx, q_idx, dirichlet_dims, alpha, elliptical: ell })
    }

    fn ln_dirichlet(&self, x: &[f64]) -> f64 {
        let (mut lp, mut j, mut k) = (0.0, 0, 0);
        for &m in &self.dirichlet_dims {
            let q: Vec<f64> = self.q_idx[k..k + m - 1].iter().map(|&o| x[o]).collect();
            lp += ln_dirichlet_pdf(&q, &self.alpha[j..j + m]);
            j += m;
            k += m - 1;
        }
        lp
    }

    /// Log proposal density at a full free-parameter vector.
    pub fn ln_density(&self, x: &[f64]) -> f64 {
        let t: Vec<f64> = self.theta_idx.iter().map(|&o| x[o]).collect();
        self.elliptical.ln_density_radius(self.elliptical.radius(&t)) + self.ln_dirichlet(x)
    }

    /// A proposal draw and its log density (`CreateProposal`).
    pub fn draw(&self, rng: &mut ChaCha8Rng) -> (Vec<f64>, f64) {
        let (t, r) = self.elliptical.draw(rng);
        let mut x = vec![0.0; self.theta_idx.len() + self.q_idx.len()];
        for (i, &o) in self.theta_idx.iter().enumerate() {
            x[o] = t[i];
        }
        let (mut j, mut k) = (0, 0);
        for &m in &self.dirichlet_dims {
            // DrawDirichlet_free
            let g = |a: f64, rng: &mut ChaCha8Rng| Gamma::new(a, 1.0).unwrap().sample(rng);
            let mut scale = g(self.alpha[j + m - 1], rng);
            let mut q = vec![0.0; m - 1];
            for i in (0..m - 1).rev() {
                q[i] = g(self.alpha[j + i], rng);
                scale += q[i];
            }
            for i in 0..m - 1 {
                x[self.q_idx[k + i]] = if scale > 0.0 { q[i] / scale } else { q[i] };
            }
            j += m;
            k += m - 1;
        }
        let lg = self.elliptical.ln_density_radius(r) + self.ln_dirichlet(&x);
        (x, lg)
    }
}

/// `ComputeLogMarginalDensity_WaggonerZha` without cut; rows are `(ln h, ln p)`.
pub fn waggoner_zha(proposal: &[(f64, f64)], posterior: &[(f64, f64)]) -> f64 {
    let _ = proposal;
    let mut i = f64::NEG_INFINITY;
    for &(h, p) in posterior.iter().rev() {
        i = logaddexp(i, h - p);
    }
    -(i - (posterior.len() as f64).ln())
}

fn bridge_difference(proposal: &[(f64, f64)], posterior: &[(f64, f64)], logr: f64) -> f64 {
    let (n1, n2) = (posterior.len() as f64, proposal.len() as f64);
    let term = |x: f64, r: f64| if x < 0.0 { -(1.0 + r * x.exp()).ln() } else { -x - ((-x).exp() + r).ln() };
    let (mut s1, mut s2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(h, p) in proposal.iter().rev() {
        s2 = logaddexp(s2, term(h + logr - p, n2 / n1));
    }
    for &(h, p) in posterior.iter().rev() {
        s1 = logaddexp(s1, term(p - h - logr, n1 / n2));
    }
    s2 - s1
}

/// `ComputeLogMarginalDensity_Bridge`; rows are `(ln h, ln p)`.
pub fn bridge(proposal: &[(f64, f64)], posterior: &[(f64, f64)]) -> f64 {
    const MAX_C: f64 = 1e50;
    const TOL: f64 = 1e-7;
    let f = |c: f64| bridge_difference(proposal, posterior, c);
    let (mut min_c, mut max_c);
    if f(0.0) < 0.0 {
        max_c = 0.0;
        min_c = -1.0;
        while min_c > -MAX_C {
            if f(min_c) > 0.0 {
                break;
            }
            max_c = min_c;
            min_c *= 10.0;
        }
        if min_c <= -MAX_C {
            return min_c;
        }
    } else {
        min_c = 0.0;
        max_c = 1.0;
        while max_c < MAX_C {
            if f(max_c) < 0.0 {
                break;
            }
            min_c = max_c;
            max_c *= 10.0;
        }
        if max_c >= MAX_C {
            return max_c;
        }
    }
    let mut mid = 0.5 * (min_c + max_c);
    let mut diff = f(mid);
    for _ in 0..50 {
        if diff > 0.0 {
            min_c = mid
        } else {
            max_c = mid
        }
        mid = 0.5 * (min_c + max_c);
        diff = f(mid);
        if diff.abs() < TOL {
            break;
        }
    }
    mid
}

/// `ComputeLogMarginalDensity_Mueller`; rows are `(ln h, ln p)`. Returns `(ln mdd, in_P1, in_P2)`.
pub fn mueller(proposal: &[(f64, f64)], posterior: &[(f64, f64)]) -> (f64, usize, usize) {
    const MAX_C: f64 = 1e50;
    let (n1, n2) = (1.0 / proposal.len() as f64, 1.0 / posterior.len() as f64);
    let lin = |lc: f64| -> (f64, usize) {
        let (mut s, mut k) = (0.0, 0);
        for &(h, p) in proposal.iter().rev() {
            let t = h - p;
            if lc <= t {
                k += 1;
                s += (lc - t).exp();
            }
        }
        (s, k)
    };
    let inv = |lc: f64| -> (f64, usize) {
        let (mut s, mut k) = (0.0, 0);
        for &(h, p) in posterior.iter().rev() {
            let t = h - p;
            if lc >= t {
                k += 1;
                s += (t - lc).exp();
            }
        }
        (s, k)
    };
    let diff = |lc: f64| -> (f64, usize, usize) {
        let ((s1, i1), (s2, i2)) = (lin(lc), inv(lc));
        (n1 * (i1 as f64 - s1) - n2 * (i2 as f64 - s2), i1, i2)
    };
    let mut lc = 0.0;
    let (mut d, mut i1, mut i2) = diff(lc);
    let (mut min_lc, mut max_lc, mut min_i, mut max_i);
    if d < 0.0 {
        loop {
            if lc < -MAX_C {
                return (lc, i1, i2);
            }
            max_i = (i1, i2);
            max_lc = lc;
            lc = 10.0 * (lc - 1.0);
            (d, i1, i2) = diff(lc);
            if d >= 0.0 {
                break;
            }
        }
        min_i = (i1, i2);
        min_lc = lc;
    } else {
        loop {
            if lc > MAX_C {
                return (lc, i1, i2);
            }
            min_i = (i1, i2);
            min_lc = lc;
            lc = 10.0 * (lc + 1.0);
            (d, i1, i2) = diff(lc);
            if d < 0.0 {
                break;
            }
        }
        max_i = (i1, i2);
        max_lc = lc;
    }
    while min_i != max_i {
        lc = 0.5 * (min_lc + max_lc);
        if lc == min_lc || lc == max_lc {
            break;
        }
        (d, i1, i2) = diff(lc);
        if d > 0.0 {
            min_i = (i1, i2);
            min_lc = lc;
        } else {
            max_i = (i1, i2);
            max_lc = lc;
        }
    }
    let (s1, k1) = lin(min_lc);
    let (s2, k2) = inv(min_lc);
    let (slope1, icpt1, slope2, icpt2) = (n1 * s1, n1 * k1 as f64, n2 * s2, n2 * k2 as f64);
    let mut tmp = icpt1 - icpt2;
    let v = if slope1 > 0.0 {
        tmp += (tmp * tmp + 4.0 * slope1 * slope2).sqrt();
        if tmp >= 2.0 * slope1 {
            tmp = min_lc + tmp.ln() - (2.0 * slope1).ln();
            if tmp > max_lc { -max_lc } else { -tmp }
        } else {
            -min_lc
        }
    } else if tmp > 0.0 {
        if slope2 > tmp {
            tmp = min_lc + slope2.ln() - tmp.ln();
            if tmp > max_lc { -max_lc } else { -tmp }
        } else {
            -min_lc
        }
    } else {
        -max_lc
    };
    (v, k1, k2)
}

/// `ms_compute_mdd` from posterior draws (`draws`: free parameters, `lps`: their log posterior as in column 0
/// of `simulation_<tag>.out`, recomputed when `None`). `mode` gives the target of the sign normalisation.
pub fn mdd(
    setup: &MsSetup,
    mode: &[f64],
    draws: &Mat,
    lps: Option<&[f64]>,
    opts: &MddOptions,
    seed: u64,
) -> anyhow::Result<MddResult> {
    let np = draws.nrows();
    let lps: Vec<f64> = match lps {
        Some(l) => l.to_vec(),
        None => (0..np).map(|r| setup.log_posterior(&draws.row(r).iter().copied().collect::<Vec<_>>())).collect(),
    };
    let ninfo = ((opts.percentage_draws * np as f64) as usize).min(np);
    let prop = Proposal::fit(setup, draws, &lps, ninfo, opts)?;
    // Waggoner-Zha normalisation of sbvar_mdd.c: target A0 = the mode, constant nvars ln 2
    let target = setup.regimes(mode).ok_or_else(|| anyhow::anyhow!("ms_compute_mdd: invalid mode"))?.a0;
    let ln_norm = setup.nvar as f64 * std::f64::consts::LN_2;
    let normalized_post = |th: &[f64]| -> f64 {
        if wz_normalized(setup, &target, th) { setup.log_posterior(th) + ln_norm } else { f64::NEG_INFINITY }
    };
    // posterior: the draws after the fitting ones (all of them when they all fit)
    let first = if ninfo == np { 0 } else { ninfo };
    let posterior: Vec<(f64, f64)> = (first..np)
        .map(|r| {
            let th: Vec<f64> = draws.row(r).iter().copied().collect();
            let lp = if opts.consistent { normalized_post(&th) } else { lps[r] };
            (prop.ln_density(&th), lp)
        })
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut valid = 0;
    let proposal: Vec<(f64, f64)> = (0..opts.proposal_draws)
        .map(|_| {
            let (th, lg) = prop.draw(&mut rng);
            let lp = normalized_post(&th);
            if lp.is_finite() {
                valid += 1;
            }
            (lg, lp)
        })
        .collect();
    if proposal.is_empty() || posterior.is_empty() {
        anyhow::bail!("No proposal or posterior draws");
    }
    let (muller, in1, in2) = mueller(&proposal, &posterior);
    Ok(MddResult {
        muller,
        bridge: bridge(&proposal, &posterior),
        wz: waggoner_zha(&proposal, &posterior),
        dim: prop.elliptical.dim,
        radius_bounds: prop.elliptical.bounds(),
        n_posterior: posterior.len(),
        n_proposal: proposal.len(),
        n_valid: valid,
        in_p: (in1, in2),
        proposal_info: prop.elliptical.info(),
        alpha: prop.alpha.clone(),
    })
}
