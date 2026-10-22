// SPDX-License-Identifier: GPL-3.0-or-later
//! Data loading for `estimation(datafile=...)`: `.csv`, `.xls`/`.xlsx` (calamine;
//! `xls_sheet`, `xls_range`) and legacy `.mat` files ([`Table::read`]).
//!
//! Mirrors what Dynare's `makedataset.m` does with a CSV file: the first row
//! holds the variable names, an optional first column holds the dates (its
//! header is empty, as written by pandas), empty cells are missing values
//! (NaN). `first_obs` (1-based) and `nobs` then select the estimation sample,
//! and the observed variables are picked in `varobs` order.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

use crate::linalg::Mat;

/// A rectangular table of series read from a CSV file.
#[derive(Debug, Clone)]
pub struct Table {
    pub names: Vec<String>,
    /// Date labels from the first column, if it had no header.
    pub dates: Option<Vec<String>>,
    /// nobs x nvars, NaN = missing.
    pub values: Mat,
}

/// `xls_sheet` / `xls_range` options of `estimation` (spreadsheets only).
#[derive(Debug, Clone, Default)]
pub struct DataOptions {
    /// Sheet name, or 1-based index as a number string; default: the first sheet.
    pub xls_sheet: Option<String>,
    /// Excel range such as `A1:D100`; default: the whole used range.
    pub xls_range: Option<String>,
}

/// Datafile extensions in the order makedataset.m tries them for a name without extension.
pub const DATAFILE_EXTENSIONS: &[&str] = &["m", "mat", "csv", "xls", "xlsx"];

impl Table {
    /// Reads a datafile by extension: `.csv`, `.mat` (`load_mat_file_data_legacy.m`: one
    /// vector per variable) or `.xls`/`.xlsx` (`load_xls_file_data.m`).
    pub fn read(path: &Path, opts: &DataOptions) -> Result<Table> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        match ext.as_str() {
            "csv" => Self::read_csv(path),
            "mat" => Self::read_mat(path),
            "xls" | "xlsx" | "xlsm" | "ods" => Self::read_xls(path, opts),
            "m" => crate::data_m::read_m_file(path), // [stream X]
            _ => bail!("{}: unknown datafile extension (expected csv, mat, xls or xlsx)", path.display()),
        }
    }

    /// Port of `load_mat_file_data_legacy.m`: every real numeric vector of the file is a
    /// series (`vec()` of the variable); no dates. Shorter series are padded with NaN.
    pub fn read_mat(path: &Path) -> Result<Table> {
        let vars = crate::matfile::read(path)?;
        let mut names = Vec::new();
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for (name, v) in &vars {
            if let crate::matfile::MatValue::Numeric { dims, data } = v {
                if dims.iter().filter(|&&d| d > 1).count() <= 1 && !data.is_empty() {
                    names.push(name.clone());
                    cols.push(data.clone());
                }
            }
        }
        if names.is_empty() {
            bail!("{}: no numeric vectors in the file", path.display());
        }
        let n = cols.iter().map(Vec::len).max().unwrap();
        let values = Mat::from_fn(n, names.len(), |i, j| cols[j].get(i).copied().unwrap_or(f64::NAN));
        Ok(Table { names, dates: None, values })
    }

    /// Port of `load_xls_file_data.m` (dseries): the first row holds the names, the first
    /// column the dates if it holds text (or Excel dates) or if its header is empty; empty or
    /// non-numeric cells are missing values.
    pub fn read_xls(path: &Path, opts: &DataOptions) -> Result<Table> {
        use calamine::{Data, Reader, open_workbook_auto};
        let mut wb = open_workbook_auto(path).with_context(|| format!("opening {}", path.display()))?;
        let sheet_names = wb.sheet_names();
        let sheet = match opts.xls_sheet.as_deref() {
            None => sheet_names.first().cloned().ok_or_else(|| anyhow!("{}: no sheets", path.display()))?,
            Some(s) => match s.trim().parse::<usize>() {
                Ok(k) if !sheet_names.contains(&s.to_string()) => sheet_names
                    .get(k.wrapping_sub(1))
                    .cloned()
                    .ok_or_else(|| anyhow!("{}: no sheet number {k}", path.display()))?,
                _ => s.to_string(),
            },
        };
        let mut range = wb.worksheet_range(&sheet).with_context(|| format!("{}: sheet {sheet}", path.display()))?;
        if let Some(r) = opts.xls_range.as_deref() {
            let (a, b) = parse_excel_range(r)?;
            range = range.range(a, b);
        }
        let rows: Vec<Vec<Data>> = range.rows().map(|r| r.to_vec()).collect();
        let Some((head, body)) = rows.split_first() else { bail!("{}: empty sheet {sheet}", path.display()) };
        let text = |c: &Data| match c {
            Data::String(s) | Data::DateTimeIso(s) => s.trim().to_string(),
            Data::Float(f) => f.to_string(),
            Data::Int(i) => i.to_string(),
            Data::DateTime(d) => excel_serial_to_date(d.as_f64()),
            _ => String::new(),
        };
        let first_col_text =
            body.iter().any(|r| matches!(r.first(), Some(Data::String(_) | Data::DateTime(_) | Data::DateTimeIso(_))));
        let has_dates = first_col_text || head.first().is_none_or(|c| text(c).is_empty());
        let skip = has_dates as usize;
        let ncols = head.len();
        let names: Vec<String> = head.iter().skip(skip).map(text).collect();
        if names.iter().any(String::is_empty) {
            bail!("{}: the data set must not contain empty/non-named columns", path.display());
        }
        let num = |c: &Data| match c {
            Data::Float(f) => *f,
            Data::Int(i) => *i as f64,
            Data::String(s) => s.trim().parse::<f64>().unwrap_or(f64::NAN),
            _ => f64::NAN,
        };
        let body: Vec<&Vec<Data>> = body.iter().filter(|r| r.iter().any(|c| !matches!(c, Data::Empty))).collect();
        let values = Mat::from_fn(body.len(), ncols - skip, |i, j| body[i].get(j + skip).map_or(f64::NAN, num));
        let dates = has_dates.then(|| body.iter().map(|r| r.first().map_or(String::new(), text)).collect());
        Ok(Table { names, dates, values })
    }

    pub fn read_csv(path: &Path) -> Result<Table> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse_csv(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse_csv(text: &str) -> Result<Table> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty()).peekable();
        let header: Vec<String> = split_row(lines.next().ok_or_else(|| anyhow!("empty file"))?);
        // load_csv_file_data.m: a first column of text (dates) is the time
        // column, with or without a header
        let first_cell_date = lines
            .peek()
            .and_then(|l| split_row(l).into_iter().next())
            .is_some_and(|c| c.parse::<f64>().is_err() && crate::dates::Date::parse(&c).is_some());
        let has_dates = header.first().is_some_and(|h| h.is_empty()) || first_cell_date;
        let names: Vec<String> = header.iter().skip(has_dates as usize).cloned().collect();
        let mut dates = Vec::new();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for (k, line) in lines.enumerate() {
            let cells = split_row(line);
            if cells.len() != header.len() {
                bail!("row {} has {} cells, header has {}", k + 2, cells.len(), header.len());
            }
            let mut it = cells.into_iter();
            if has_dates {
                dates.push(it.next().unwrap());
            }
            let row =
                it.map(|c| parse_cell(&c)).collect::<Result<Vec<f64>>>().with_context(|| format!("row {}", k + 2))?;
            rows.push(row);
        }
        let values = Mat::from_fn(rows.len(), names.len(), |i, j| rows[i][j]);
        Ok(Table { names, dates: has_dates.then_some(dates), values })
    }

    pub fn column(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// The estimation sample: rows `first_obs..first_obs+nobs` (1-based
    /// `first_obs`, `nobs = None` = to the end), columns in `varobs` order.
    /// Returns nobs x nvobs.
    pub fn sample(&self, varobs: &[String], first_obs: usize, nobs: Option<usize>) -> Result<Mat> {
        if first_obs == 0 || first_obs > self.values.nrows() {
            bail!("first_obs = {first_obs} is outside the data (1..={})", self.values.nrows());
        }
        let start = first_obs - 1;
        let available = self.values.nrows() - start;
        let n = nobs.unwrap_or(available);
        if n > available {
            bail!("nobs = {n} with first_obs = {first_obs} runs past the {} rows of data", self.values.nrows());
        }
        let cols = varobs
            .iter()
            .map(|v| self.column(v).ok_or_else(|| anyhow!("observed variable {v} is not in the datafile")))
            .collect::<Result<Vec<usize>>>()?;
        Ok(Mat::from_fn(n, cols.len(), |i, j| self.values[(start + i, cols[j])]))
    }
}

/// `A1:D100` -> ((row, col), (row, col)), 0-based.
fn parse_excel_range(r: &str) -> Result<((u32, u32), (u32, u32))> {
    let cell = |s: &str| -> Result<(u32, u32)> {
        let s = s.trim().replace('$', "").to_ascii_uppercase();
        let split = s.find(|c: char| c.is_ascii_digit()).ok_or_else(|| anyhow!("bad cell {s:?} in xls_range"))?;
        let (letters, digits) = s.split_at(split);
        if letters.is_empty() || letters.len() > 3 || !letters.chars().all(|c| c.is_ascii_uppercase()) {
            bail!("bad cell {s:?} in xls_range");
        }
        let col = letters.bytes().fold(0u32, |a, b| a * 26 + (b - b'A' + 1) as u32) - 1;
        let row: u32 = digits.parse().map_err(|_| anyhow!("bad cell {s:?} in xls_range"))?;
        Ok((row.checked_sub(1).ok_or_else(|| anyhow!("bad cell {s:?} in xls_range"))?, col))
    };
    let (a, b) = r.split_once(':').ok_or_else(|| anyhow!("xls_range {r:?}: expected e.g. A1:D100"))?;
    let (a, b) = (cell(a)?, cell(b)?);
    if a.0 > b.0 || a.1 > b.1 {
        bail!(
            "xls_range {r:?}: the first cell must be the upper-left corner and the second the lower-right one (e.g. A1:D100)"
        );
    }
    Ok((a, b))
}

/// Excel serial date (1900 system) -> `YYYY-MM-DD`.
fn excel_serial_to_date(serial: f64) -> String {
    // days since 1970-01-01: Excel day 25569 = 1970-01-01 (serials > 60, past the 1900 leap bug)
    let z = serial.floor() as i64 - 25569 + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02}")
}

fn split_row(line: &str) -> Vec<String> {
    line.trim_end_matches('\r').split(',').map(|c| c.trim().trim_matches('"').to_string()).collect()
}

fn parse_cell(c: &str) -> Result<f64> {
    if c.is_empty() || c.eq_ignore_ascii_case("nan") || c.eq_ignore_ascii_case("na") {
        return Ok(f64::NAN);
    }
    c.parse::<f64>().map_err(|e| anyhow!("bad number {c:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_missing_and_sample() {
        let t = Table::parse_csv(",a,b\n2003-03-31,1.5,\n2003-06-30,2,3\n2003-09-30,4,5\n").unwrap();
        assert_eq!(t.names, ["a", "b"]);
        assert_eq!(t.dates.as_ref().unwrap()[1], "2003-06-30");
        assert!(t.values[(0, 1)].is_nan());
        let s = t.sample(&["b".into(), "a".into()], 2, None).unwrap();
        assert_eq!((s.nrows(), s.ncols()), (2, 2));
        assert_eq!(s[(0, 0)], 3.0);
        assert_eq!(s[(1, 1)], 4.0);
        assert!(t.sample(&["a".into()], 3, Some(2)).is_err());
    }
}
