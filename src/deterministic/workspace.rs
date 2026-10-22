// SPDX-License-Identifier: GPL-3.0-or-later
//! The part of the MATLAB workspace that deterministic statements read: numeric variables
//! assigned in the .mod file (`xx = [1.2; 1.3; 1];`, used by `shocks; ... values (xx);`) and
//! the databases of the `shock_paths` block (`database db;`), which Dynare takes from the base
//! workspace (`evalin('base', ...)`): `dseries` objects and `table`s.
//!
//! The parser evaluates the assignments it understands (`dseries(MATRIX, DATE, NAMES)`,
//! `dseries('file')`, `table(...)`, `array2table(...)` and numeric expressions with
//! `[...]`, `linspace`, `transpose`, `ones`, `zeros`, `repmat`, colon ranges and elementwise
//! arithmetic, see `modfile::parser_shocks`); [`extract_from_database`] ports
//! `perfect-foresight-models/extract_from_database.m`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow, bail};

use crate::dates::Date;
use crate::linalg::Mat;

/// A workspace value.
#[derive(Clone, Debug, PartialEq)]
pub enum WsValue {
    /// A numeric matrix (scalars are 1x1).
    Num(Mat),
    /// A string (`'2000Q1'`).
    Str(String),
    /// A cell array of strings (`{'a', 'b'}`).
    Cell(Vec<String>),
    /// `dseries(data, init, names)`: nobs x nvars.
    Dseries { init: Date, names: Vec<String>, data: Mat },
    /// `table(...)` / `array2table(...)`: nrows x nvars.
    Table { names: Vec<String>, data: Mat },
}

/// Variables assigned in the .mod file that dynare-rust could evaluate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Workspace {
    pub vars: HashMap<String, WsValue>,
}

impl Workspace {
    /// A numeric variable as a vector (column-major).
    pub fn vector(&self, name: &str) -> Option<Vec<f64>> {
        match self.vars.get(name)? {
            WsValue::Num(m) => Some(m.iter().copied().collect()),
            _ => None,
        }
    }
}

/// `dseries(file)`: a csv/xls(x)/mat/m file whose first column holds the dates (relative
/// paths from `dir`, the .mod file's folder).
pub fn load_dseries_file(file: &str, dir: &Path) -> Result<(Date, Vec<String>, Mat)> {
    let p = if Path::new(file).is_absolute() { Path::new(file).to_path_buf() } else { dir.join(file) };
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let t = match ext.as_str() {
        "csv" => crate::data::Table::read_csv(&p)?,
        "xls" | "xlsx" => crate::data::Table::read_xls(&p, &Default::default())?,
        "mat" => crate::data::Table::read_mat(&p)?,
        "m" => crate::data_m::read_m_file(&p)?,
        _ => bail!("dseries('{file}'): unsupported file type"),
    };
    let init = crate::dataset::table_init(&t).unwrap_or(Date::new(crate::dates::Freq::Annual, 1));
    Ok((init, t.names, t.values))
}

/// Port of `extract_from_database.m`: the value of `varname` in database `dbname` at
/// simulation period `period` (dseries: date `first_simulation_period + period - 1`; table:
/// row `period`, or the only row of a single-row table).
pub fn extract_from_database(
    ws: &Workspace,
    dbname: &str,
    varname: &str,
    period: i64,
    first_simulation_period: Option<Date>,
) -> Result<f64> {
    let db = ws
        .vars
        .get(dbname)
        .ok_or_else(|| anyhow!("database {dbname} does not exist in the base workspace (dynare-rust: it must be assigned in the .mod file as a dseries or a table)"))?;
    let col = |names: &[String]| -> Result<usize> {
        names.iter().position(|n| n == varname).ok_or_else(|| anyhow!("database {dbname}: no variable named {varname}"))
    };
    let from_dseries = |init: Date, names: &[String], data: &Mat| -> Result<f64> {
        let Some(f) = first_simulation_period else {
            bail!(
                "{dbname} is a dseries database is used but neither first_simulation_period nor last_simulation_period option was passed"
            )
        };
        let d = f.add(period - 1);
        let j = col(names)?;
        let Some(k) = init.diff(d) else {
            bail!("database {dbname}: the frequency differs from the simulation dates'")
        };
        if k < 0 || k as usize >= data.nrows() {
            bail!("database {dbname}: no observation of {varname} at {}", d.label());
        }
        Ok(data[(k as usize, j)])
    };
    match db {
        WsValue::Dseries { init, names, data } => from_dseries(*init, names, data),
        WsValue::Table { names, data } => {
            let j = col(names)?;
            if data.nrows() == 1 {
                return Ok(data[(0, j)]);
            }
            if period < 1 || period as usize > data.nrows() {
                bail!("database {dbname}: row {period} does not exist (the table has {} rows)", data.nrows());
            }
            Ok(data[(period as usize - 1, j)])
        }
        _ => bail!("Unsupported database type for {dbname}"),
    }
}
