// SPDX-License-Identifier: GPL-3.0-or-later
//! `.xlsx` workbooks (rust_xlsxwriter) for `write_xls`: MATLAB's `writetable` /
//! `xlswrite` targets (`.xls` in Dynare) become real Office Open XML workbooks.

use std::path::Path;

use anyhow::{Context, Result};
use rust_xlsxwriter::Workbook;

use super::decomp::{XlsCell, XlsSheet};

/// Excel sheet names: at most 31 characters, none of `[]:*?/\`.
fn sheet_name(s: &str) -> String {
    let t: String = s.chars().map(|c| if "[]:*?/\\".contains(c) { '_' } else { c }).collect();
    t.chars().take(31).collect()
}

pub fn write(path: &Path, sheets: &[XlsSheet]) -> Result<()> {
    let mut wb = Workbook::new();
    for sh in sheets {
        let ws = wb.add_worksheet();
        ws.set_name(sheet_name(&sh.name)).with_context(|| format!("sheet name {}", sh.name))?;
        for (r, row) in sh.rows.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                let (r, c) = (r as u32, c as u16);
                match cell {
                    XlsCell::Empty => {}
                    XlsCell::Num(v) if v.is_finite() => {
                        ws.write_number(r, c, *v)?;
                    }
                    XlsCell::Num(v) => {
                        // NaN / Inf: Excel has no such numbers; MATLAB's writetable leaves NaN cells empty
                        let _ = v;
                    }
                    XlsCell::Text(s) => {
                        ws.write_string(r, c, s)?;
                    }
                }
            }
        }
    }
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    wb.save(path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
