//! Weekly and daily `dates` against dseries 7.2 (reference/dates/dump_dates.m ->
//! ref_dates.json): parsing, `char`, `year`, `subperiod` (ISO weeks, long years, leap
//! days), shifts, differences, ranges, the `dates('W', y, w)` constructor,
//! `sameperiodpreviousyear` and weekly/daily `ydiff`/`ygrowth`.
mod common;

use dynare::dates::{Date, Freq, same_period_previous_year, ydiff, ygrowth};
use serde_json::Value;

fn reference() -> Value {
    common::json_ref("reference/dates/ref_dates.json")
}

fn d(s: &str) -> Date {
    Date::parse(s).unwrap_or_else(|| panic!("parse {s}"))
}

fn ints(v: &Value) -> Vec<i64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_i64().unwrap()).collect(),
        x => vec![x.as_i64().unwrap()],
    }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()
}

#[test]
fn parse_label_year_subperiod() {
    let r = reference();
    for c in r["parse"].as_array().unwrap() {
        let s = c["s"].as_str().unwrap();
        // MATLAB prints datenum 0 as '0000-01-00'; not reproduced.
        if s == "-0001-12-31" {
            continue;
        }
        let p = Date::parse(s);
        assert_eq!(p.is_some(), c["ok"].as_bool().unwrap(), "{s}");
        let Some(p) = p else { continue };
        assert_eq!(p.freq.number(), c["freq"].as_i64().unwrap(), "{s}");
        assert_eq!(p.time, c["time"].as_i64().unwrap(), "{s}");
        assert_eq!(p.label(), c["label"].as_str().unwrap(), "{s}");
        assert_eq!(p.year(), c["year"].as_i64().unwrap(), "{s}");
        if let Some(sp) = c["subperiod"].as_i64() {
            assert_eq!(p.subperiod().unwrap(), sp, "{s}");
        }
    }
}

#[test]
fn daily_subperiod_errors() {
    let e = d("2020-03-01").subperiod().unwrap_err();
    assert_eq!(e.to_string(), "dates::subperiod: Method not available for daily frequency!");
    assert_eq!(d("2020W10").subperiod().unwrap(), 10);
}

#[test]
fn shifts_and_differences() {
    let r = reference();
    for c in r["shifts"].as_array().unwrap() {
        let s = c["s"].as_str().unwrap();
        let x = d(s).add(c["k"].as_i64().unwrap());
        assert_eq!(x.time, c["time"].as_i64().unwrap(), "{s}");
        assert_eq!(x.label(), c["label"].as_str().unwrap(), "{s}");
        assert_eq!(x.year(), c["year"].as_i64().unwrap(), "{s}");
    }
    for c in r["diffs"].as_array().unwrap() {
        let (a, b) = (c["a"].as_str().unwrap(), c["b"].as_str().unwrap());
        assert_eq!(d(a).diff(d(b)), Some(c["n"].as_i64().unwrap()), "{a} {b}");
    }
}

#[test]
fn ranges() {
    let r = reference();
    for c in r["ranges"].as_array().unwrap() {
        let (a, b) = (c["a"].as_str().unwrap(), c["b"].as_str().unwrap());
        let v = Date::range(d(a), d(b)).unwrap();
        let labels: Vec<String> = v.iter().map(|x| x.label()).collect();
        let want: Vec<&str> = c["labels"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
        assert_eq!(labels, want, "{a}:{b}");
        if v[0].freq == Freq::Weekly {
            // (MATLAB's year() of a daily range returns only the first year.)
            assert_eq!(v.iter().map(|x| x.year()).collect::<Vec<_>>(), ints(&c["years"]), "{a}:{b}");
            assert_eq!(v.iter().map(|x| x.subperiod().unwrap()).collect::<Vec<_>>(), ints(&c["subperiods"]), "{a}:{b}");
        }
    }
}

#[test]
fn weekly_constructor() {
    let r = reference();
    for c in r["ctor"].as_array().unwrap() {
        let (y, w) = (c["y"].as_i64().unwrap(), c["w"].as_i64().unwrap());
        let x = Date::weekly(y, w);
        assert_eq!(x.is_ok(), c["ok"].as_bool().unwrap(), "{y}W{w}");
        if let Ok(x) = x {
            assert_eq!(x.time, c["time"].as_i64().unwrap());
            assert_eq!(x.label(), c["label"].as_str().unwrap());
        }
    }
}

#[test]
fn same_period_previous_year_and_ydiff_ygrowth() {
    let r = reference();
    for c in r["spy"].as_array().unwrap() {
        let (a, b) = (c["a"].as_str().unwrap(), c["b"].as_str().unwrap());
        let n = (d(a).diff(d(b)).unwrap() + 1) as usize;
        let got: Vec<i64> = same_period_previous_year(d(a), n).iter().map(|x| x.map_or(0, |j| j as i64 + 1)).collect();
        assert_eq!(got, ints(&c["id"]), "{a}:{b}");
    }
    let close = |got: &[f64], want: &[f64], tag: &str| {
        assert_eq!(got.len(), want.len(), "{tag}");
        for (t, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12 * (1.0 + w.abs()), "{tag}[{t}]: {g} vs {w}");
        }
    };
    let wy = d("2019W1");
    let x: Vec<f64> = (1..=157).map(|k| 1.01f64.powi(k)).collect();
    close(&ydiff(wy, &x), &floats(&r["weekly_ydiff"]), "weekly ydiff");
    close(&ygrowth(wy, &x), &floats(&r["weekly_ygrowth"]), "weekly ygrowth");
    let dy = d("2019-01-01");
    let n = (dy.diff(d("2021-12-31")).unwrap() + 1) as i32;
    let x: Vec<f64> = (1..=n).map(|k| 1.0001f64.powi(k)).collect();
    close(&ydiff(dy, &x), &floats(&r["daily_ydiff"]), "daily ydiff");
    close(&ygrowth(dy, &x), &floats(&r["daily_ygrowth"]), "daily ygrowth");
}
