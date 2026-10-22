// SPDX-License-Identifier: GPL-3.0-or-later
//! `load_ident_files` (re-display a saved point analysis, `<fname>_identif.json`,
//! the counterpart of Dynare's `<fname>_identif.mat`) and `gsa_sample_file` (the Monte Carlo
//! sample of a previous `sensitivity` run: `<fname>/gsa/<fname>_prior.mat`, `_mc.mat`, or a
//! named file with `lpmat`, `lpmat0`, `istable`).

use std::path::Path;

use anyhow::{Context as _, Result, anyhow};
use serde_json::Value;

use super::checks::CheckResult;
use super::display::{Criterion, display};
use super::run::IdentOptions;
use crate::linalg::Mat;
use crate::modfile::OptValue;

fn bools(v: &Value) -> Vec<bool> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_bool().unwrap_or(x.as_f64().is_some_and(|f| f != 0.0))).collect(),
        _ => Vec::new(),
    }
}

fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect(),
        _ => Vec::new(),
    }
}

fn check_from_json(v: &Value) -> CheckResult {
    let pco = match &v["Pco"] {
        Value::Array(rows) if !rows.is_empty() => {
            let r: Vec<Vec<f64>> = rows.iter().map(nums).collect();
            let nc = r[0].len();
            Mat::from_fn(r.len(), nc, |i, j| r[i][j])
        }
        _ => Mat::zeros(0, 0),
    };
    CheckResult {
        cond: v["cond"].as_f64().unwrap_or(f64::NAN),
        rank: v["rank"].as_u64().unwrap_or(0) as usize,
        ind0: bools(&v["ind0"]),
        indno: match &v["indno"] {
            Value::Array(rows) => rows.iter().map(bools).collect(),
            _ => Vec::new(),
        },
        ino: v["ino"].as_u64().unwrap_or(0) as usize,
        mco: nums(&v["Mco"]),
        pco,
        jweak: bools(&v["jweak"]),
        jweak_pair: bools(&v["jweak_pair"]),
    }
}

/// `load_ident_files = 1`: `Testing <parameters>` and `identification.display` of the saved
/// point analysis. Returns false when there is no saved file.
pub fn display_saved(ident_dir: &Path, fname: &str, names: &[String], o: &IdentOptions) -> Result<bool> {
    let p = ident_dir.join(format!("{fname}_identif.json"));
    let Ok(s) = std::fs::read_to_string(&p) else { return Ok(false) };
    let j: Value = serde_json::from_str(&s).with_context(|| format!("reading {}", p.display()))?;
    let get = |k: &str| j.get(k).filter(|v| !v.is_null()).map(|b| check_from_json(&b["checks"]));
    let (rf, mi, sp, mo) = (get("reducedform"), get("minimal"), get("spectrum"), get("moments"));
    let mss = j["minimal_state_space"].as_bool().unwrap_or(false);
    fn crit(c: &Option<CheckResult>) -> Option<Criterion<'_>> {
        c.as_ref().map(|c| Criterion { checks: vec![c], minimal_state_space: None, subsets: None })
    }
    let minimal = if o.no_minimal || !mss {
        None
    } else {
        mi.as_ref().map(|c| Criterion { checks: vec![c], minimal_state_space: Some(vec![true]), subsets: None })
    };
    println!("Testing {}", j["parameter_set"].as_str().unwrap_or(&o.parameter_set));
    display(1, [crit(&rf), minimal, crit(&sp), crit(&mo)], names, o, None);
    Ok(true)
}

/// `gsa_sample_file`: `pdraws0 = [lpmat0(istable,:) lpmat(istable,:)]`.
pub fn gsa_sample(dir: &Path, fname: &str, spec: &OptValue) -> Result<Vec<Vec<f64>>> {
    let gsa = dir.join(fname).join("gsa");
    let path = match spec {
        OptValue::Num(v) if *v == 1.0 => gsa.join(format!("{fname}_prior.mat")),
        OptValue::Num(v) if *v == 2.0 => gsa.join(format!("{fname}_mc.mat")),
        OptValue::Str(s) => {
            let f = if s.ends_with(".mat") { s.clone() } else { format!("{s}.mat") };
            gsa.join(f)
        }
        other => return Err(anyhow!("identification: bad gsa_sample_file {other:?}")),
    };
    let m = crate::matfile::read(&path).with_context(|| format!("gsa_sample_file: reading {}", path.display()))?;
    let lpmat = m.get("lpmat").and_then(|v| v.as_matrix()).ok_or_else(|| anyhow!("{}: no lpmat", path.display()))?;
    let lpmat0 = m
        .get("lpmat0")
        .and_then(|v| v.as_matrix())
        .filter(|x| x.nrows() == lpmat.nrows())
        .unwrap_or_else(|| Mat::zeros(lpmat.nrows(), 0));
    let mut istable: Vec<usize> =
        m.get("istable").and_then(|v| v.as_vec()).unwrap_or_default().iter().map(|&x| x as usize - 1).collect();
    if istable.is_empty() {
        istable = (0..lpmat.nrows()).collect();
    }
    Ok(istable.iter().map(|&k| lpmat0.row(k).iter().chain(lpmat.row(k).iter()).copied().collect()).collect())
}
