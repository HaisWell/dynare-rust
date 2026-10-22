//! `.xlsx` (xls_sheet, xls_range) and legacy `.mat` datafiles against what Dynare 7.1 reads
//! from the same files (reference/data/make_datafiles.m -> ref_datafiles.json): the SW07 data
//! written by MATLAB to sw07_obs.xlsx (sheet 2 'data', from B3, quarterly dates as text) and
//! sw07_obs.mat (one vector per variable).
mod common;

use std::path::Path;

use dynare::data::{DataOptions, Table};
use serde_json::Value;

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => vec![f64::NAN],
    }
}

/// Equal up to the last bit (the xlsx cells are decimal strings, which MATLAB's reader
/// rounds 1 ulp differently in a few cells).
fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 4.0 * f64::EPSILON * a.abs().max(b.abs())
}

#[test]
fn xlsx_and_mat_datafiles_match_dynare() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/data");
    let text = common::read_ref(dir.join("ref_datafiles.json"));
    let r: Value = serde_json::from_str(&text).unwrap();
    let names: Vec<String> = r["names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().into()).collect();
    let rows = r["data"].as_array().unwrap();
    let range = r["xls_range"].as_str().unwrap().to_string();

    for sheet in ["data", "2"] {
        let opts = DataOptions { xls_sheet: Some(sheet.into()), xls_range: Some(range.clone()) };
        let t = Table::read(&dir.join("sw07_obs.xlsx"), &opts).unwrap();
        assert_eq!(t.names, names);
        assert_eq!(t.values.nrows(), rows.len());
        assert_eq!(t.dates.as_ref().unwrap()[0], r["first_date"].as_str().unwrap());
        for (i, row) in rows.iter().enumerate() {
            for (j, v) in vecf(row).iter().enumerate() {
                assert!(same(t.values[(i, j)], *v), "xlsx ({i}, {j}): {} vs {v}", t.values[(i, j)]);
            }
        }
    }
    // the same numbers as the CSV the file was written from
    let csv = Table::read(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07/sw07_obs.csv"),
        &DataOptions::default(),
    )
    .unwrap();
    let x = Table::read(
        &dir.join("sw07_obs.xlsx"),
        &DataOptions { xls_sheet: Some("data".into()), xls_range: Some(range) },
    )
    .unwrap();
    let a = csv.sample(&names, 1, None).unwrap();
    let b = x.sample(&names, 1, None).unwrap();
    assert!(a.iter().zip(b.iter()).all(|(p, q)| same(*p, *q)));

    let m = Table::read(&dir.join("sw07_obs.mat"), &DataOptions::default()).unwrap();
    let mat_names: Vec<String> =
        r["mat_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().into()).collect();
    let s = m.sample(&mat_names, 1, None).unwrap();
    for (i, row) in r["mat_data"].as_array().unwrap().iter().enumerate() {
        for (j, v) in vecf(row).iter().enumerate() {
            assert!(same(s[(i, j)], *v), "mat ({i}, {j})");
        }
    }
}
