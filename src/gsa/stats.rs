// SPDX-License-Identifier: GPL-3.0-or-later
//! Statistics of the GSA toolbox: two-sample Smirnov test (`+gsa/smirnov_test.m`),
//! MATLAB's `corrcoef` with p-values, sample moments, and Dynare's `dyntable` printout.

use crate::estimation::special::betainc;

/// Port of `+gsa/smirnov_test.m`: two-sample Kolmogorov–Smirnov statistic and its
/// asymptotic p-value. `iflag` 0 = two-sided, 1 = `max(cum1 - cum2)`, -1 = `max(cum2 - cum1)`.
/// Returns `(H, prob, d)` with `H = alpha >= prob`.
pub fn smirnov_test(x1: &[f64], x2: &[f64], alpha: f64, iflag: i32) -> (bool, f64, f64) {
    let n1 = x1.len();
    let n2 = x2.len();
    // histcounts(x, [-inf; sort([x1;x2]); inf]) then cumsum/sum, last bin dropped: the CDF
    // just below each sorted pooled value, cum(k) = #(x < s_k) / n.
    let mut s: Vec<f64> = x1.iter().chain(x2).copied().collect();
    s.sort_by(f64::total_cmp);
    let mut a = x1.to_vec();
    a.sort_by(f64::total_cmp);
    let mut b = x2.to_vec();
    b.sort_by(f64::total_cmp);
    let (mut ia, mut ib) = (0usize, 0usize);
    let mut d = f64::NEG_INFINITY;
    let (f1, f2) = (n1 as f64, n2 as f64);
    for &sk in &s {
        while ia < n1 && a[ia] < sk {
            ia += 1;
        }
        while ib < n2 && b[ib] < sk {
            ib += 1;
        }
        let c1 = ia as f64 / f1;
        let c2 = ib as f64 / f2;
        let v = match iflag {
            0 => (c1 - c2).abs(),
            1 => c1 - c2,
            _ => c2 - c1,
        };
        // MATLAB max ignores NaN (empty sample -> all NaN -> NaN)
        if v > d || d == f64::NEG_INFINITY && !v.is_nan() {
            d = v;
        }
    }
    if n1 == 0 || n2 == 0 || s.is_empty() {
        d = f64::NAN;
    }
    let n = f1 * f2 / (f1 + f2);
    let lam = ((n.sqrt() + 0.12 + 0.11 / n.sqrt()) * d).max(0.0);
    let prob = if d.is_nan() {
        f64::NAN
    } else if iflag == 0 {
        let mut p = 0.0;
        for j in 1..=101 {
            let jf = j as f64;
            let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
            p += sign * (-2.0 * lam * lam * jf * jf).exp();
        }
        (2.0 * p).clamp(0.0, 1.0)
    } else {
        (-2.0 * lam * lam).exp()
    };
    (alpha >= prob, prob, d)
}

/// `[c, p] = corrcoef(x)` (MATLAB R2025b): `x` is n x m column data (`cols[j]` = column j).
/// Returns m x m matrices (row-major `Vec<Vec>`).
pub fn corrcoef(cols: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = cols.len();
    let n = cols.first().map_or(0, Vec::len);
    let means: Vec<f64> = cols.iter().map(|c| mean(c)).collect();
    let mut cov = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..=i {
            let s: f64 = (0..n).map(|k| (cols[i][k] - means[i]) * (cols[j][k] - means[j])).sum();
            cov[i][j] = s / (n as f64 - 1.0);
            cov[j][i] = cov[i][j];
        }
    }
    let d: Vec<f64> = (0..m).map(|i| cov[i][i].sqrt()).collect();
    let mut r = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            r[i][j] = cov[i][j] / d[i] / d[j];
        }
    }
    for i in 0..m {
        for j in 0..i {
            let v = (r[i][j] + r[j][i]) / 2.0;
            let v = if v.abs() > 1.0 { v.signum() } else { v };
            r[i][j] = v;
            r[j][i] = v;
        }
        r[i][i] = if r[i][i].is_nan() { f64::NAN } else { r[i][i].signum() };
    }
    let nv = n as f64;
    let mut p = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..i {
            let rv = r[i][j];
            let t = rv * ((nv - 2.0) / (1.0 - rv * rv)).sqrt();
            let pv = 2.0 * tpvalue(-t.abs(), nv - 2.0);
            p[i][j] = pv;
            p[j][i] = pv;
        }
        p[i][i] = r[i][i];
    }
    (r, p)
}

/// `tpvalue` of MATLAB's corrcoef.m: Student-t CDF at `x` with `v` degrees of freedom.
fn tpvalue(x: f64, v: f64) -> f64 {
    if x.is_nan() || !(v > 0.0) {
        return f64::NAN;
    }
    if v == 1.0 {
        return 0.5 + x.atan() / std::f64::consts::PI;
    }
    if v > 1e7 {
        return 0.5 * libm_erfc(-x / std::f64::consts::SQRT_2);
    }
    if x == 0.0 {
        return 0.5;
    }
    let mut p = betainc(v / (v + x * x), v / 2.0, 0.5) / 2.0;
    if x > 0.0 {
        p = 1.0 - p;
    }
    p
}

fn libm_erfc(x: f64) -> f64 {
    statrs::function::erf::erfc(x)
}

pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// MATLAB `std` (n-1 normalization).
pub fn std(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)).sqrt()
}

/// MATLAB `var` (n-1 normalization).
pub fn var(x: &[f64]) -> f64 {
    std(x).powi(2)
}

/// `[~, i] = sort(x)` (ascending, stable, NaN last), 0-based.
pub fn argsort(x: &[f64]) -> Vec<usize> {
    let mut i: Vec<usize> = (0..x.len()).collect();
    i.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
    // total_cmp orders -NaN first; MATLAB puts every NaN last
    let (nan, mut ok): (Vec<usize>, Vec<usize>) = i.into_iter().partition(|&k| x[k].is_nan());
    ok.extend(nan);
    ok
}

/// Dynare's `dyntable.m` (console part): title, headers, one row per label.
pub fn dyntable(
    title: &str,
    headers: &[&str],
    labels: &[String],
    values: &[Vec<f64>],
    label_width: usize,
    val_width: usize,
    val_precis: usize,
) -> String {
    let maxlab = labels.iter().map(|l| l.len()).chain(std::iter::once(headers[0].len())).max().unwrap_or(0);
    let label_width = (maxlab + 2).max(label_width);
    let finite: Vec<f64> = values.iter().flatten().copied().filter(|v| v.is_finite()).collect();
    let mut values_length = if finite.is_empty() {
        4
    } else {
        let mx = finite.iter().map(|v| v.abs().log10()).fold(f64::NEG_INFINITY, f64::max);
        (mx.ceil().max(1.0) as usize) + val_precis + 1
    };
    if values.iter().flatten().any(|&v| v < 0.0) {
        values_length += 1;
    }
    let headers_length = headers[1..].iter().map(|h| h.len()).max().unwrap_or(0);
    let val_width = (headers_length.max(values_length) + 2).max(val_width);
    let mut s = String::new();
    if !title.is_empty() {
        s += &format!("\n\n{title}\n");
    }
    s += &format!("{:<label_width$}", headers[0]);
    for h in &headers[1..] {
        s += &format!("{h:>val_width$}");
    }
    s.push('\n');
    for (l, row) in labels.iter().zip(values) {
        s += &format!("{l:<label_width$}");
        for v in row {
            s += &fmt_f(*v, val_width, val_precis);
        }
        s.push('\n');
    }
    s
}

/// MATLAB `sprintf('%W.Pf', v)` (NaN/Inf as MATLAB prints them).
pub fn fmt_f(v: f64, w: usize, p: usize) -> String {
    if v.is_nan() {
        format!("{:>w$}", "NaN")
    } else if v.is_infinite() {
        format!("{:>w$}", if v > 0.0 { "Inf" } else { "-Inf" })
    } else {
        format!("{v:>w$.p$}")
    }
}

/// MATLAB `num2str(x)` for a scalar (`%.4g`-like with up to 4 significant digits after
/// trimming; integers printed as integers).
pub fn num2str(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        return format!("{}", x as i64);
    }
    if !x.is_finite() {
        return if x.is_nan() {
            "NaN".into()
        } else if x > 0.0 {
            "Inf".into()
        } else {
            "-Inf".into()
        };
    }
    // num2str: '%.{max(ceil(log10(|x|)),1)+4}g'
    let digits = ((x.abs().log10().ceil() as i32).max(1) + 4) as usize;
    fmt_g(x, digits)
}

/// C `%.{p}g`.
pub fn fmt_g(x: f64, p: usize) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let p = p.max(1);
    let e = x.abs().log10().floor() as i32;
    if e < -4 || e >= p as i32 {
        let s = format!("{:.*e}", p - 1, x);
        // Rust: 1.2346e-5 -> C: 1.2346e-05
        let (mant, exp) = s.split_once('e').unwrap();
        let mant = if mant.contains('.') { mant.trim_end_matches('0').trim_end_matches('.') } else { mant };
        let ev: i32 = exp.parse().unwrap();
        format!("{mant}e{}{:02}", if ev < 0 { '-' } else { '+' }, ev.abs())
    } else {
        let dec = (p as i32 - 1 - e).max(0) as usize;
        let s = format!("{x:.dec$}");
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smirnov_identical_and_disjoint() {
        let a: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let b: Vec<f64> = (100..150).map(|i| i as f64).collect();
        let (_, p, d) = smirnov_test(&a, &b, 0.05, 0);
        assert_eq!(d, 1.0);
        assert!(p < 1e-10);
        let (_, p, d) = smirnov_test(&a, &a, 0.05, 0);
        assert_eq!(d, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn formats() {
        assert_eq!(fmt_g(1e-5, 5), "1e-05");
        assert_eq!(fmt_g(0.123456, 5), "0.12346");
        assert_eq!(num2str(0.001), "0.001");
        assert_eq!(num2str(1e-5), "1e-05");
        assert_eq!(num2str(2.0), "2");
    }
}
