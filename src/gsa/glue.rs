// SPDX-License-Identifier: GPL-3.0-or-later
//! `sensitivity(glue=1)` — the GLUE file of `+gsa/run.m`
//! (`<fname>_glue_prior.mat` / `_glue_mc.mat`: `Out`, `Sam`, `Lik`, `Obs`, `Rem`, `Info`, `Exo`)
//! built from the RMSE analysis of the same run.

use std::path::Path;

use anyhow::Result;

use super::rmse::RmseResults;
use super::stability::StabMap;
use super::{Ctx, GsaOptions};
use crate::dsge::DsgeModel;
use crate::linalg::Mat;
use crate::matfile::{MatValue, write_values};
use crate::modfile::{ModFile, Options};

fn s(x: &str) -> MatValue {
    MatValue::Char(x.to_string())
}

fn n(x: f64) -> MatValue {
    MatValue::scalar(x)
}

fn row_vec(v: &[f64]) -> MatValue {
    MatValue::row(v)
}

fn mat(m: &Mat) -> MatValue {
    MatValue::matrix(m)
}

/// A 1 x k struct array from per-element field lists (same fields in every element).
fn struct_array(fields: &[&str], elems: Vec<Vec<MatValue>>) -> MatValue {
    MatValue::Struct { dims: vec![1, elems.len()], fields: fields.iter().map(|f| f.to_string()).collect(), elems }
}

/// Writes the GLUE file; prints Dynare's message when there is no RMSE analysis.
#[allow(clippy::too_many_arguments)]
pub fn write_glue(
    ctx: &Ctx,
    mf: &ModFile,
    dir: &Path,
    dsge: &DsgeModel,
    o: &GsaOptions,
    eo: &Options,
    stab: &StabMap,
    rmse: Option<&RmseResults>,
) -> Result<()> {
    let Some(r) = rmse else {
        println!("No RMSE analysis is available for current options");
        println!("No GLUE file prepared");
        return Ok(());
    };
    let Some(datafile) = o.datafile.as_deref() else {
        println!("No RMSE analysis is available for current options");
        println!("No GLUE file prepared");
        return Ok(());
    };
    let data = super::rmse::likelihood_data(mf, dir, eo, datafile)?;
    let gend = data.y.ncols();
    let dr = crate::solver::DrStructure::new(&dsge.model.lead_lag_incidence, dsge.model.n_exo());
    let cwd = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default();
    let outdir = format!("{cwd}/{}", ctx.out_dir.display());
    let varobs: Vec<String> = dsge.varobs.iter().map(|&v| dsge.model.endo_names[v].clone()).collect();
    let mut out = Vec::new();
    let mut lik = Vec::new();
    let mut ismoo = Vec::new();
    for (j, vj) in varobs.iter().enumerate() {
        let jxj = dr.inv_order_var[dsge.varobs[j]] + 1;
        ismoo.push(jxj);
        out.push(vec![n(jxj as f64), s(&outdir), s(vj), s("yes")]);
        let col: Vec<f64> = r.rmse_mc.iter().map(|row| row.get(j).copied().unwrap_or(f64::NAN)).collect();
        lik.push(vec![s(&format!("rmse_{vj}")), s("yes"), n(1.0), row_vec(&col)]);
    }
    for k in 1..=dsge.model.n_endo() {
        if !ismoo.contains(&k) {
            let vj = &dsge.model.endo_names[dr.order_var[k - 1]];
            out.push(vec![n(k as f64), s(&outdir), s(vj), s("yes")]);
        }
    }
    let logpo: Vec<f64> = r.logpo2.iter().map(|v| -v).collect();
    lik.push(vec![s("logpo"), s("yes"), n(1.0), row_vec(&logpo)]);
    let x: Vec<Vec<f64>> = stab.istable.iter().map(|&k| stab.point(k)).collect();
    let xm = Mat::from_fn(x.len(), x.first().map_or(0, Vec::len), |i, j| x[i][j]);
    let names = MatValue::Cell {
        dims: vec![dsge.estimated_names.len(), 1],
        items: dsge.estimated_names.iter().map(|v| s(v)).collect(),
    };
    let sam = MatValue::structure(vec![
        ("name", names),
        ("dim", row_vec(&[xm.nrows() as f64, xm.ncols() as f64, 0.0])),
        ("data", mat(&xm)),
    ]);
    let obs = MatValue::structure(vec![
        ("data", mat(&data.y)),
        ("time", row_vec(&(1..=gend).map(|t| t as f64).collect::<Vec<_>>())),
        ("num", n(gend as f64)),
        ("name", MatValue::Cell { dims: vec![1, varobs.len()], items: varobs.iter().map(|v| s(v)).collect() }),
    ]);
    let rem = MatValue::structure(vec![
        ("id", s("Original")),
        ("ind", row_vec(&(1..=xm.nrows()).map(|t| t as f64).collect::<Vec<_>>())),
    ]);
    let ty = if o.pprior { "prior" } else { "mc" };
    let info = MatValue::structure(vec![
        ("dynare", s(&ctx.fname)),
        ("order_var", MatValue::column(&dr.order_var.iter().map(|&v| (v + 1) as f64).collect::<Vec<_>>())),
        ("TypeofSample", s(ty)),
    ]);
    let exo = struct_array(&["name"], dsge.model.exo_names.iter().map(|e| vec![s(e)]).collect());
    let out = struct_array(&["data", "time", "name", "ini"], out);
    let lik = struct_array(&["name", "ini", "isam", "data"], lik);
    let p = ctx.out_dir.join(format!("{}_glue_{ty}.mat", ctx.fname));
    write_values(
        &p,
        &[("Out", &out), ("Sam", &sam), ("Lik", &lik), ("Obs", &obs), ("Rem", &rem), ("Info", &info), ("Exo", &exo)],
    )?;
    println!("GLUE file written to {}", p.display());
    Ok(())
}
