// SPDX-License-Identifier: GPL-3.0-or-later
//! Figures of the MS-SBVAR commands (`plot_ms_probabilities`, `plot_ms_irf`, `plot_ms_forecast`,
//! `plot_ms_variance_decomposition` of Dynare's matlab/ms-sbvar), written through `crate::graphics`
//! (honour `nograph` and `graph_format`).

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::graphics::{self, Color, Figure, GraphOptions, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::modfile::Options;

fn options(o: &Options, dir: &Path, fname: &str, target: PathBuf) -> GraphOptions {
    let mut g = GraphOptions::from_options(o, dir, fname);
    g.dir = target;
    g
}

fn report(files: &[PathBuf]) {
    if !files.is_empty() {
        println!(
            "MS-SBVAR: {} figure file(s) written to {}",
            files.len(),
            files[0].parent().map(|p| p.display().to_string()).unwrap_or_default()
        );
    }
}

fn line(x: Vec<f64>, y: Vec<f64>, style: LineStyle, color: Color, label: Option<String>) -> Series {
    Series::Line { x, y, color, style, width: 1.2, label }
}

/// One line per percentile; with several, the outer two as a shaded band and the median line.
fn percentile_series(x: &[f64], cols: &[Vec<f64>]) -> Vec<Series> {
    let k = cols.len();
    if k >= 3 {
        vec![
            Series::Band {
                x: x.to_vec(),
                lower: cols[0].clone(),
                upper: cols[k - 1].clone(),
                color: Color::Gray,
                alpha: 0.4,
                label: None,
            },
            line(x.to_vec(), cols[k / 2].clone(), LineStyle::Solid, Color::Blue, None),
        ]
    } else {
        cols.iter().map(|c| line(x.to_vec(), c.clone(), LineStyle::Solid, Color::Blue, None)).collect()
    }
}

/// `plot_ms_probabilities`: marginal regime probabilities of each chain (`chain_sizes`, chain 1 slowest).
pub fn probabilities(m: &Mat, chain_sizes: &[usize], o: &Options, dir: &Path, fname: &str, tag: &str) -> Result<()> {
    let g = options(o, dir, fname, dir.join(tag).join("Output").join("Probabilities"));
    if !g.enabled {
        return Ok(());
    }
    let t: Vec<f64> = (1..=m.nrows()).map(|v| v as f64).collect();
    let mut files = vec![];
    for (c, &nr) in chain_sizes.iter().enumerate() {
        let inner: usize = chain_sizes[c + 1..].iter().product();
        let mut series = vec![];
        for r in 0..nr {
            let y: Vec<f64> = (0..m.nrows())
                .map(|i| (0..m.ncols()).filter(|s| (s / inner) % nr == r).map(|s| m[(i, s)]).sum())
                .collect();
            let col = [Color::Blue, Color::Red, Color::Green, Color::Black, Color::Gray][r % 5];
            series.push(line(t.clone(), y, LineStyle::Solid, col, Some(format!("Regime {}", r + 1))));
        }
        let fig = Figure {
            name: format!("MS-Probabilities-Chain-{}", c + 1),
            title: format!("MS-Probabilities, Chain {}", c + 1),
            rows: 1,
            cols: 1,
            panels: vec![Panel { title: format!("Chain {}", c + 1), series, legend: true, ..Default::default() }],
        };
        files.extend(graphics::save_as(&fig, &fig.name.clone(), &g)?);
    }
    report(&files);
    Ok(())
}

/// IRF / variance decomposition: `m` has `npct` blocks of `horizon` rows, columns `a*n + b`
/// (IRF: shock `a`, variable `b`; variance decomposition: variable `a`, shock `b`).
pub fn blocks(
    kind: &str,
    label: &str,
    m: &Mat,
    n: usize,
    horizon: usize,
    names: &[String],
    o: &Options,
    dir: &Path,
    fname: &str,
    target: PathBuf,
) -> Result<()> {
    let g = options(o, dir, fname, target);
    if !g.enabled || horizon == 0 {
        return Ok(());
    }
    let npct = m.nrows() / horizon;
    let x: Vec<f64> = (1..=horizon).map(|v| v as f64).collect();
    let irf = kind == "IRF";
    let mut panels = vec![];
    for v in 0..n {
        for s in 0..n {
            // row: variable, column: shock
            let col = if irf { s * n + v } else { v * n + s };
            let cols: Vec<Vec<f64>> =
                (0..npct).map(|k| (0..horizon).map(|h| m[(k * horizon + h, col)]).collect()).collect();
            let mut series = percentile_series(&x, &cols);
            if irf {
                series.push(Series::HLine { y: 0.0, color: Color::Red, style: LineStyle::Dotted });
            }
            panels.push(Panel {
                title: if v == 0 {
                    format!("shock {}", names.get(s).cloned().unwrap_or_default())
                } else {
                    String::new()
                },
                series,
                y_label: if s == 0 { names.get(v).cloned() } else { None },
                ..Default::default()
            });
        }
    }
    let stem = format!("MS-{kind}-{label}");
    let fig = Figure { name: stem.clone(), title: format!("MS-SBVAR {kind}, {label}"), rows: n, cols: n, panels };
    report(&graphics::save_as(&fig, &stem, &g)?);
    Ok(())
}

/// `plot_ms_forecast`: `full` has `npct` blocks of `rows` rows (data history, then the forecast), `n` columns.
pub fn forecast(
    label: &str,
    full: &Mat,
    rows: usize,
    dobs: usize,
    names: &[String],
    o: &Options,
    dir: &Path,
    fname: &str,
    target: PathBuf,
) -> Result<()> {
    let g = options(o, dir, fname, target);
    if !g.enabled || rows == 0 {
        return Ok(());
    }
    let n = full.ncols();
    let npct = full.nrows() / rows;
    let x: Vec<f64> = (1..=rows).map(|v| v as f64).collect();
    let mut panels = vec![];
    for v in 0..n {
        let cols: Vec<Vec<f64>> = (0..npct).map(|k| (0..rows).map(|r| full[(k * rows + r, v)]).collect()).collect();
        let mut series = vec![];
        if dobs > 0 {
            series.push(line(x[..dobs].to_vec(), cols[0][..dobs].to_vec(), LineStyle::Solid, Color::Black, None));
        }
        let xf = x[dobs.saturating_sub(1).min(rows - 1)..].to_vec();
        let fc: Vec<Vec<f64>> = cols.iter().map(|c| c[dobs.saturating_sub(1).min(rows - 1)..].to_vec()).collect();
        series.extend(percentile_series(&xf, &fc));
        panels.push(Panel { title: names.get(v).cloned().unwrap_or_default(), series, ..Default::default() });
    }
    let stem = format!("MS-Forecast-{label}");
    let fig = Figure { name: stem.clone(), title: format!("MS-SBVAR forecast, {label}"), rows: n, cols: 1, panels };
    report(&graphics::save_as(&fig, &stem, &g)?);
    Ok(())
}
