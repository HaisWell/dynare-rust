// SPDX-License-Identifier: GPL-3.0-or-later
//! First and second derivatives of the log prior density: the `nargout >= 2` branches of
//! `estimation/priordens.m` and of the `lpdf*.m` functions it calls (`lpdfgbeta`, `lpdfgam`,
//! `lpdfnorm`, `lpdfig1`, `lpdfig2`, `lpdfgweibull`; the uniform has zero derivatives).

use crate::estimation::{Prior, PriorShape};

/// `[~, dlprior, d2lprior] = priordens(x, ...)`: the gradient and the diagonal of the Hessian
/// of the log prior density (Dynare returns `diag(d2lprior)`). Entries outside the support
/// are `-Inf` as in the `lpdf*.m` functions (the caller has already rejected such points).
pub fn log_prior_derivatives(x: &[f64], priors: &[Prior]) -> (Vec<f64>, Vec<f64>) {
    let mut d1 = vec![0.0; x.len()];
    let mut d2 = vec![0.0; x.len()];
    for (k, (&xi, p)) in x.iter().zip(priors).enumerate() {
        let (a, b) = match p.shape {
            PriorShape::Beta => dlpdfgbeta(xi, p.p6, p.p7, p.p3, p.p4),
            PriorShape::Gamma => dlpdfgam(xi - p.p3, p.p6, p.p7),
            PriorShape::Normal => dlpdfnorm(xi, p.p6, p.p7),
            PriorShape::InvGamma1 => dlpdfig1(xi - p.p3, p.p6, p.p7),
            PriorShape::Uniform => (0.0, 0.0),
            PriorShape::InvGamma2 => dlpdfig2(xi - p.p3, p.p6, p.p7),
            PriorShape::Weibull => dlpdfgweibull(xi, p.p6, p.p7, p.p3),
        };
        d1[k] = a;
        d2[k] = b;
    }
    (d1, d2)
}

/// `lpdfgbeta.m`: `(a-1)/(x-aa) - (b-1)/(bb-x)`, `-(a-1)/(x-aa)^2 - (b-1)/(bb-x)^2`.
fn dlpdfgbeta(x: f64, a: f64, b: f64, aa: f64, bb: f64) -> (f64, f64) {
    if x - aa > 0.0 && x - bb < 0.0 {
        ((a - 1.0) / (x - aa) - (b - 1.0) / (bb - x), -(a - 1.0) / (x - aa).powi(2) - (b - 1.0) / (bb - x).powi(2))
    } else {
        (f64::NEG_INFINITY, f64::NEG_INFINITY)
    }
}

/// `lpdfgam.m` (shape `a`, scale `b`).
fn dlpdfgam(x: f64, a: f64, b: f64) -> (f64, f64) {
    if x > 0.0 { ((a - 1.0) / x - 1.0 / b, -(a - 1.0) / (x * x)) } else { (f64::NEG_INFINITY, f64::NEG_INFINITY) }
}

/// `lpdfnorm.m`.
fn dlpdfnorm(x: f64, a: f64, b: f64) -> (f64, f64) {
    (-(1.0 / b) * ((x - a) / b), -(1.0 / b).powi(2))
}

/// `lpdfig1.m`.
fn dlpdfig1(x: f64, s: f64, nu: f64) -> (f64, f64) {
    if x > 0.0 {
        (-(nu + 1.0) / x + s / x.powi(3), (nu + 1.0) / (x * x) - 3.0 * s / x.powi(4))
    } else {
        (f64::NEG_INFINITY, f64::NEG_INFINITY)
    }
}

/// `lpdfig2.m`.
fn dlpdfig2(x: f64, s: f64, nu: f64) -> (f64, f64) {
    if x > 0.0 {
        (-0.5 * (nu + 2.0) / x + 0.5 * s / (x * x), 0.5 * (nu + 2.0) / (x * x) - s / x.powi(3))
    } else {
        (f64::NEG_INFINITY, f64::NEG_INFINITY)
    }
}

/// `lpdfgweibull.m` (shape `a`, scale `b`, location `c`): NaN outside the support and at
/// the exponential's origin (where Dynare's index `idx` excludes the point).
fn dlpdfgweibull(x: f64, a: f64, b: f64, c: f64) -> (f64, f64) {
    if x - c < 0.0 || ((a - 1.0).abs() < 1e-12 && (x - c) < 1e-12) {
        return (f64::NAN, f64::NAN);
    }
    let x0 = x - c;
    let x1 = x0 / b;
    let x2 = x1.powf(a);
    let x3 = (a - 1.0) / x0;
    let x4 = a * x2 / x1 / b;
    (x3 - x4, -x3 / x0 - (a - 1.0) * x4 / x1 / b)
}
