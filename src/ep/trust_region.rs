// SPDX-License-Identifier: GPL-3.0-or-later
//! `dynare_solve.m` with `solve_algo = 9` (the extended path's default `ep.solve_algo`):
//! `optimization/trust_region.m` (Powell's dogleg, MINPACK's `hybrj` scaling) on a sparse
//! Jacobian. The dense version used by the steady state is `solver::steady_state`.

use super::sparse::{LuPlan, SparseJac};

/// A square nonlinear system with a fixed-pattern sparse Jacobian.
pub trait System {
    fn dim(&self) -> usize;
    /// Residuals at `x` into `res`; with `jac`, the Jacobian values too.
    fn eval(&mut self, x: &[f64], res: &mut [f64], jac: Option<&mut SparseJac>);
    /// The Jacobian alone (default: [`System::eval`], discarding the residuals).
    fn jacobian(&mut self, x: &[f64], jac: &mut SparseJac) {
        let mut res = vec![0.0; self.dim()];
        self.eval(x, &mut res, Some(jac));
    }
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn wnorm(d: &[f64], v: &[f64]) -> f64 {
    d.iter().zip(v).map(|(a, b)| (a * b) * (a * b)).sum::<f64>().sqrt()
}

/// MATLAB `eps(x)`.
fn eps_of(x: f64) -> f64 {
    let x = x.abs();
    if !x.is_finite() {
        return f64::NAN;
    }
    if x < f64::MIN_POSITIVE {
        return f64::from_bits(1);
    }
    let e = ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    2f64.powi(e - 52)
}

/// Outcome of [`dynare_solve`]: `errorflag` and Dynare's `errorcode` (-11: the initial guess
/// solves the system; -10: non-finite residuals or Jacobian at the initial guess (Dynare
/// would randomize the guess); `trust_region.m`'s `info` otherwise: 1 converged, 2 maxit,
/// 3 trust region too small, 4 slow progress, 5 step too small, 0 no iteration).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveStatus {
    pub errorflag: bool,
    pub errorcode: i32,
    pub iterations: usize,
}

/// Port of `dynare_solve.m` (`jacobian_flag`, `solve_algo = 9`): checks the initial guess,
/// then `trust_region.m`. `x` is updated in place (left at the initial guess on failure, as
/// `trust_region.m`).
#[allow(clippy::too_many_arguments)]
pub fn dynare_solve(
    sys: &mut dyn System,
    jac: &mut SparseJac,
    plan: &mut LuPlan,
    x: &mut [f64],
    maxit: usize,
    tolf: f64,
    tolx: f64,
    factor: f64,
) -> SolveStatus {
    let n = sys.dim();
    let mut fvec = vec![0.0; n];
    sys.eval(x, &mut fvec, Some(jac));
    let finite = fvec.iter().all(|v| v.is_finite());
    if finite && fvec.iter().fold(0.0f64, |m, v| m.max(v.abs())) < tolf {
        return SolveStatus { errorflag: false, errorcode: -11, iterations: 0 };
    }
    if !finite || !jac.all_finite() {
        return SolveStatus { errorflag: true, errorcode: -10, iterations: 0 };
    }
    trust_region(sys, jac, plan, x, fvec, maxit, tolf, tolx, factor)
}

/// Port of `optimization/trust_region.m` (Jacobian supplied, `j1 = j2 = 1:n`); `fval`/`jac`
/// hold the residuals and the Jacobian at `x`.
#[allow(clippy::too_many_arguments)]
fn trust_region(
    sys: &mut dyn System,
    fjac: &mut SparseJac,
    plan: &mut LuPlan,
    x: &mut [f64],
    mut fval: Vec<f64>,
    maxiter: usize,
    tolf: f64,
    tolx: f64,
    factor: f64,
) -> SolveStatus {
    let radiusfactor = 0.5;
    let actualreductionthreshold = 0.001;
    let relativereductionthreshold = 0.1;
    let n = x.len();
    let mut errorflag = true;
    let mut iter = 1usize;
    let mut info = 0;
    let mut delta = 0.0;
    let (mut ncsucc, mut ncslow) = (0usize, 0usize);
    let xinit = x.to_vec();
    let mut fnorm = norm2(&fval);
    if fnorm < tolf {
        return SolveStatus { errorflag: false, errorcode: -1, iterations: 0 };
    }
    let mut fjacnorm_ = vec![0.0; n];
    let mut xnorm;
    let mut fval1 = vec![0.0; n];
    let mut xx = vec![0.0; n];
    while iter <= maxiter && info == 0 {
        let fjacnorm = fjac.col_norms();
        if iter == 1 {
            fjacnorm_ = fjacnorm.iter().map(|&v| if v < f64::EPSILON { 1.0 } else { v }).collect();
            xnorm = wnorm(&fjacnorm_, x);
            delta = if xnorm > 0.0 { xnorm * factor } else { factor };
        } else {
            for (a, b) in fjacnorm_.iter_mut().zip(&fjacnorm) {
                *a = (0.1 * *a).max(*b);
            }
        }
        let p = dogleg(fjac, plan, &fval, &fjacnorm_, delta);
        let pnorm = wnorm(&fjacnorm_, &p);
        let x0 = x.to_vec();
        for k in 0..n {
            xx[k] = x[k] - p[k];
        }
        let jp = fjac.mul(&p);
        let ww: Vec<f64> = (0..n).map(|k| fval[k] - jp[k]).collect();
        sys.eval(&xx, &mut fval1, None);
        if fval1.iter().any(|v| !v.is_finite()) {
            delta *= radiusfactor;
            iter += 1;
            continue;
        }
        let fnorm1 = norm2(&fval1);
        if fnorm1 < tolf {
            x.copy_from_slice(&xx);
            errorflag = false;
            info = 1;
            continue;
        }
        let actualreduction = if fnorm1 < fnorm { 1.0 - (fnorm1 / fnorm).powi(2) } else { -1.0 };
        let tt = norm2(&ww);
        let predictedreduction = if tt < fnorm { 1.0 - (tt / fnorm).powi(2) } else { 0.0 };
        let ratio = if predictedreduction > 0.0 { actualreduction / predictedreduction } else { 0.0 };
        if iter == 1 {
            delta = delta.min(pnorm);
        }
        if ratio < relativereductionthreshold {
            ncsucc = 0;
            delta *= radiusfactor;
        } else {
            ncsucc += 1;
            if (ratio - 1.0).abs() < relativereductionthreshold {
                delta = pnorm / radiusfactor;
            } else if ratio >= radiusfactor || ncsucc > 1 {
                delta = delta.max(pnorm / radiusfactor);
            }
        }
        if ratio > 1e-4 {
            x.copy_from_slice(&xx);
            std::mem::swap(&mut fval, &mut fval1);
            fnorm = fnorm1;
        }
        xnorm = wnorm(&fjacnorm_, x);
        ncslow += 1;
        if actualreduction >= actualreductionthreshold {
            ncslow = 0;
        }
        iter += 1;
        if iter == maxiter {
            info = 2;
            x.copy_from_slice(&xinit);
            continue;
        }
        if delta < tolx * xnorm {
            info = 3;
            x.copy_from_slice(&xinit);
            errorflag = true;
            continue;
        }
        if tolx > 0.0 && (0.1 * delta).max(pnorm) <= 10.0 * eps_of(xnorm) * xnorm {
            info = 5;
            x.copy_from_slice(&xinit);
            errorflag = true;
            continue;
        }
        if ncslow == 15 {
            info = 4;
            x.copy_from_slice(&xinit);
            errorflag = true;
            continue;
        }
        let fjac0 = fjac.csc.vals.clone();
        sys.jacobian(x, fjac);
        if !fjac.all_finite() {
            x.copy_from_slice(&x0);
            fjac.csc.vals = fjac0;
            delta *= radiusfactor;
        }
    }
    SolveStatus { errorflag, errorcode: info, iterations: iter }
}

/// `dogleg` subfunction of `trust_region.m` (`r \ b` by sparse LU; a singular `r` gives a
/// NaN Newton direction, whose MATLAB arithmetic falls back to the scaled gradient step).
fn dogleg(r: &SparseJac, plan: &mut LuPlan, b: &[f64], d: &[f64], delta: f64) -> Vec<f64> {
    let n = b.len();
    let mut x = match plan.factor(r) {
        Some(f) => f.solve(b),
        None => vec![f64::NAN; n],
    };
    let qnorm = wnorm(d, &x);
    if qnorm <= delta {
        return x;
    }
    let mut s: Vec<f64> = r.tmul(b).iter().zip(d).map(|(a, b)| a / b).collect();
    let gnorm = norm2(&s);
    let (alpha, sgnorm);
    if gnorm > 0.0 {
        for (v, dd) in s.iter_mut().zip(d) {
            *v = (*v / gnorm) / dd;
        }
        let temp0 = norm2(&r.mul(&s));
        sgnorm = gnorm / (temp0 * temp0);
        if sgnorm < delta {
            let bnorm = norm2(b);
            let temp1 = delta / qnorm;
            let temp2 = sgnorm / delta;
            let mut t0 = bnorm * bnorm * temp2 / (gnorm * qnorm);
            t0 = t0 - temp1 * temp2 * temp2
                + ((t0 - temp1).powi(2) + (1.0 - temp1 * temp1) * (1.0 - temp2 * temp2)).sqrt();
            alpha = temp1 * (1.0 - temp2 * temp2) / t0;
        } else {
            alpha = 0.0;
        }
    } else {
        alpha = delta / qnorm;
        sgnorm = 0.0;
    }
    let w = sgnorm.min(delta);
    if alpha > 0.0 {
        for k in 0..n {
            x[k] = alpha * x[k] + (1.0 - alpha) * w * s[k];
        }
    } else {
        for k in 0..n {
            x[k] = w * s[k];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eps_matches_matlab() {
        assert_eq!(eps_of(1.0), f64::EPSILON);
        assert_eq!(eps_of(2.0), 2.0 * f64::EPSILON);
        assert_eq!(eps_of(1.5), f64::EPSILON);
    }
}
