// SPDX-License-Identifier: GPL-3.0-or-later
//! Console tables in Dynare's format: port of `dyntable.m`, plus MATLAB-compatible
//! `sprintf` conversions (`%f`, `%e`, `%g` with NaN/Inf spelled as MATLAB does).

use crate::linalg::Mat;

/// MATLAB `sprintf('%W.Pf', x)`.
pub fn fmt_f(x: f64, width: usize, prec: usize) -> String {
    if !x.is_finite() {
        return format!("{:>width$}", nonfinite(x));
    }
    format!("{x:>width$.prec$}")
}

/// MATLAB `sprintf('%W.Pe', x)` (C style: `1.234560e+05`).
pub fn fmt_e(x: f64, width: usize, prec: usize) -> String {
    if !x.is_finite() {
        return format!("{:>width$}", nonfinite(x));
    }
    format!("{:>width$}", c_exp(x, prec))
}

/// MATLAB `sprintf('%W.Pg', x)` (C `%g`: P significant digits, trailing zeros removed,
/// exponent form when the exponent is < -4 or >= P).
pub fn fmt_g(x: f64, width: usize, prec: usize) -> String {
    if !x.is_finite() {
        return format!("{:>width$}", nonfinite(x));
    }
    let p = prec.max(1);
    if x == 0.0 {
        return format!("{:>width$}", if x.is_sign_negative() { "-0" } else { "0" });
    }
    // exponent after rounding to p significant digits
    let e = c_exp(x, p - 1);
    let exp: i32 = e[e.find('e').unwrap() + 1..].parse().unwrap();
    let s = if exp < -4 || exp >= p as i32 {
        let (mant, ex) = e.split_at(e.find('e').unwrap());
        format!("{}{}", strip_zeros(mant), ex)
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        strip_zeros(&format!("{x:.decimals$}")).to_string()
    };
    format!("{s:>width$}")
}

fn strip_zeros(s: &str) -> &str {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { s }
}

fn c_exp(x: f64, prec: usize) -> String {
    let s = format!("{x:.prec$e}"); // e.g. 1.234560e5, -1.5e-7
    let (mant, ex) = s.split_at(s.find('e').unwrap());
    let ex: i32 = ex[1..].parse().unwrap();
    format!("{mant}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs())
}

fn nonfinite(x: f64) -> &'static str {
    if x.is_nan() {
        "NaN"
    } else if x > 0.0 {
        "Inf"
    } else {
        "-Inf"
    }
}

/// MATLAB `num2str` of a scalar as used in titles (integers without decimals).
pub fn num2str(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        let s = fmt_g(x, 0, 5);
        s.trim().to_string()
    }
}

/// Port of `dyntable.m`: returns the text (title preceded by two blank lines, header, rows).
/// `label_width` / `val_width` are the minimum widths passed by the caller.
pub fn dyntable(
    title: &str,
    headers: &[String],
    labels: &[String],
    values: &Mat,
    label_width: usize,
    val_width: usize,
    val_precis: usize,
) -> String {
    let maxlen = |v: &[String]| v.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    let mut first = vec![headers.first().cloned().unwrap_or_default()];
    first.extend(labels.iter().cloned());
    let label_width = (maxlen(&first) + 2).max(label_width);
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let mut values_length = if finite.is_empty() {
        4
    } else {
        let m = finite.iter().map(|v| v.abs().log10()).fold(f64::NEG_INFINITY, f64::max).ceil();
        (m.max(1.0) as usize) + val_precis + 1
    };
    if values.iter().any(|&v| v < 0.0) {
        values_length += 1;
    }
    let headers_length = if headers.len() > 1 { maxlen(&headers[1..]) } else { 0 };
    let val_width = (headers_length.max(values_length) + 2).max(val_width);
    let mut out = String::new();
    if !title.is_empty() {
        out += &format!("\n\n{title}\n");
    }
    if !headers.is_empty() {
        out += &format!("{:<label_width$}", headers[0]);
        for h in &headers[1..] {
            out += &format!("{h:>val_width$}");
        }
        out.push('\n');
    }
    for (i, l) in labels.iter().enumerate() {
        out += &format!("{l:<label_width$}");
        for j in 0..values.ncols() {
            out += &fmt_f(values[(i, j)], val_width, val_precis);
        }
        out.push('\n');
    }
    out
}

/// `cellofchararraymaxlength`.
pub fn max_len(v: &[String]) -> usize {
    v.iter().map(|s| s.chars().count()).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matlab_formats() {
        assert_eq!(fmt_g(0.4326, 16, 4), "          0.4326");
        assert_eq!(fmt_g(0.8, 0, 4), "0.8");
        assert_eq!(fmt_g(1.0, 0, 4), "1");
        assert_eq!(fmt_g(1.18346, 0, 4), "1.183");
        assert_eq!(fmt_g(-0.05262, 0, 4), "-0.05262");
        assert_eq!(fmt_g(6.6214e16, 0, 4), "6.621e+16");
        assert_eq!(fmt_g(1.5e-7, 0, 4), "1.5e-07");
        assert_eq!(fmt_g(12345.0, 0, 4), "1.234e+04");
        assert_eq!(fmt_g(9999.5, 0, 4), "1e+04");
        assert_eq!(fmt_g(f64::INFINITY, 5, 4), "  Inf");
        assert_eq!(fmt_e(123456.0, 16, 6), "    1.234560e+05");
        assert_eq!(fmt_f(f64::NAN, 11, 4), "        NaN");
        assert_eq!(fmt_f(-0.0, 8, 2), "   -0.00");
        assert_eq!(num2str(1600.0), "1600");
    }
}
