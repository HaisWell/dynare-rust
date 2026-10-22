// SPDX-License-Identifier: GPL-3.0-or-later
//! Special functions and the quantile functions Dynare ships in
//! `missing/stats/` (used by `prior_bounds.m` when `prior_trunc > 0` and by
//! `marginal_density.m` through `chi2inv`).

#![allow(clippy::neg_cmp_op_on_partial_ord)] // `~(a > 0)` is MATLAB's NaN-aware test

use std::os::raw::c_int;

unsafe extern "C" {
    /// Reentrant `lgamma` from the C math library (thread-safe, ~1 ulp).
    fn lgamma_r(x: f64, sign: *mut c_int) -> f64;
}

/// MATLAB `gammaln` (log of the absolute value of the gamma function).
#[inline]
pub fn gammaln(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let mut s: c_int = 0;
    unsafe { lgamma_r(x, &mut s) }
}

/// MATLAB `betaln(a, b) = gammaln(a) + gammaln(b) - gammaln(a+b)`.
#[inline]
pub fn betaln(a: f64, b: f64) -> f64 {
    gammaln(a) + gammaln(b) - gammaln(a + b)
}

/// Regularized incomplete beta function (MATLAB `betainc(x, a, b)`).
pub fn betainc(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        statrs::function::beta::beta_reg(a, b, x)
    }
}

/// Regularized lower incomplete gamma function (MATLAB `gammainc(x, a)`).
pub fn gammainc(x: f64, a: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x.is_infinite() {
        1.0
    } else {
        statrs::function::gamma::gamma_lr(a, x)
    }
}

fn betapdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - betaln(a, b)).exp()
}

fn gampdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() - x / b - gammaln(a) - a * b.ln()).exp()
}

/// Port of `missing/stats/betainv.m` (Newton iterations from the mean).
pub fn betainv(x: f64, a: f64, b: f64) -> f64 {
    let eps = f64::EPSILON;
    if x < 0.0 || x > 1.0 || !(a > 0.0) || !(b > 0.0) || x.is_nan() {
        return f64::NAN;
    }
    if x == 1.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    let mut y = a / (a + b);
    if y < eps {
        y = eps.sqrt();
    }
    if y > 1.0 - eps {
        y = 1.0 - eps.sqrt();
    }
    let mut y_new = y;
    let mut h = f64::INFINITY;
    let mut loopcnt = 0;
    while h.abs() >= eps.sqrt() && loopcnt < 40 {
        let y_old = y_new;
        h = (betainc(y_old, a, b) - x) / betapdf(y_old, a, b);
        y_new = y_old - h;
        if y_new <= eps {
            y_new = y_old / 10.0;
        }
        if y_new >= 1.0 - eps {
            y_new = 1.0 - (1.0 - y_old) / 10.0;
        }
        h = y_old - y_new;
        loopcnt += 1;
    }
    y_new
}

/// Port of `missing/stats/gaminv.m` (shape `a`, scale `b`).
pub fn gaminv(x: f64, a: f64, b: f64) -> f64 {
    let eps = f64::EPSILON;
    let ok_ab = a > 0.0 && a < f64::INFINITY && b > 0.0 && b < f64::INFINITY;
    if x < 0.0 || x > 1.0 || x.is_nan() || !ok_ab {
        return f64::NAN;
    }
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return 0.0;
    }
    let mut y = a * b;
    if x < eps {
        y = eps.sqrt();
    }
    let mut y_new = y;
    let mut h = f64::INFINITY;
    let mut loopcnt = 0;
    while h.abs() >= eps.sqrt() && loopcnt < 40 {
        let y_old = y_new;
        h = (gammainc(y_old / b, a) - x) / gampdf(y_old, a, b);
        y_new = y_old - h;
        if y_new <= eps {
            y_new = y_old / 10.0;
        }
        h = y_old - y_new;
        loopcnt += 1;
    }
    y_new
}

/// Port of `missing/stats/norminv.m` (`m + s*sqrt(2)*erfinv(2x-1)`).
pub fn norminv(x: f64, m: f64, s: f64) -> f64 {
    if m.is_infinite() || m.is_nan() || !(s > 0.0) || !(s < f64::INFINITY) {
        return f64::NAN;
    }
    m + s * (2f64.sqrt() * statrs::function::erf::erf_inv(2.0 * x - 1.0))
}

/// Port of `missing/stats/wblinv.m`.
pub fn wblinv(p: f64, scale: f64, shape: f64) -> f64 {
    let eps = f64::EPSILON;
    if p > 1.0 - 2.0 * eps {
        f64::INFINITY
    } else if p >= 2.0 * eps {
        (scale.ln() + (-(1.0 - p).ln()).ln() / shape).exp()
    } else {
        0.0
    }
}

/// Port of `missing/stats/chi2inv.m` (`gaminv(x, n/2, 2)`).
pub fn chi2inv(x: f64, n: f64) -> f64 {
    gaminv(x, n / 2.0, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gammaln_values() {
        assert!((gammaln(5.0) - 24f64.ln()).abs() < 1e-14);
        assert!((gammaln(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-15);
    }

    #[test]
    fn inverses_roundtrip() {
        let x = betainv(1e-10, 7.2896, 14.1504);
        assert!((betainc(x, 7.2896, 14.1504) - 1e-10).abs() < 1e-20);
        let g = gaminv(0.3, 3.0, 2.0);
        assert!((gammainc(g / 2.0, 3.0) - 0.3).abs() < 1e-14);
        assert!((norminv(0.975, 0.0, 1.0) - 1.959963984540054).abs() < 1e-12);
        let c = chi2inv(0.5, 73.0);
        assert!((gammainc(c / 2.0, 36.5) - 0.5).abs() < 1e-14);
    }
}
