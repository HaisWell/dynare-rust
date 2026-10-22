// SPDX-License-Identifier: GPL-3.0-or-later
//! Figures: Dynare's `dyn_figure`/`dyn_saveas` equivalent. Modules build a [`Figure`] (a grid
//! of panels holding plain data) and call [`save`]; rendering is this module's job alone.
//!
//! Files go to `<fname>/graphics/<fname>_<name>.<ext>` like Dynare's `graphs/` folder
//! (see [`graph_dir`]), one file per format in `graph_format` (pdf, eps, fig -> svg, png
//! (rasterized SVG), svg).

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub mod canvas;
pub mod font;
pub mod layout;
pub mod png; // [stream X] PNG via resvg

// ---- a colormap for the categorical palette (`colormap` option of the decompositions)
thread_local! {
    static PALETTE_OVERRIDE: std::cell::RefCell<Option<Vec<(f64, f64, f64)>>> = const { std::cell::RefCell::new(None) };
}

/// Renders with `Color::Palette(i)` = `colors[i]` while `f` runs (figures are rendered on the
/// calling thread).
pub fn with_palette<R>(colors: Option<Vec<(f64, f64, f64)>>, f: impl FnOnce() -> R) -> R {
    let old = PALETTE_OVERRIDE.with(|p| p.replace(colors));
    let r = f();
    PALETTE_OVERRIDE.with(|p| *p.borrow_mut() = old);
    r
}

pub(crate) fn palette_override(i: usize) -> Option<canvas::Rgb> {
    PALETTE_OVERRIDE.with(|p| {
        p.borrow().as_ref().filter(|c| !c.is_empty()).map(|c| {
            let (r, g, b) = c[i % c.len()];
            canvas::Rgb(r, g, b)
        })
    })
}

/// One output format; `graph_format` values that have no counterpart are mapped by
/// [`GraphFormat::parse`] (`fig` -> svg) or dropped with a warning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphFormat {
    Pdf,
    Eps,
    Svg,
    Png,
}

impl GraphFormat {
    pub fn parse(s: &str) -> Option<GraphFormat> {
        match s.to_ascii_lowercase().as_str() {
            "pdf" => Some(GraphFormat::Pdf),
            "eps" => Some(GraphFormat::Eps),
            "svg" | "fig" => Some(GraphFormat::Svg),
            "png" => Some(GraphFormat::Png),
            _ => None,
        }
    }
    pub fn ext(self) -> &'static str {
        match self {
            GraphFormat::Pdf => "pdf",
            GraphFormat::Eps => "eps",
            GraphFormat::Svg => "svg",
            GraphFormat::Png => "png",
        }
    }
}

/// What the run asked for: `nograph` disables everything, `graph_format` picks the formats
/// (Dynare's default is eps; dynare-rust defaults to pdf).
#[derive(Clone, Debug)]
pub struct GraphOptions {
    pub enabled: bool,
    pub formats: Vec<GraphFormat>,
    /// `nodisplay` is implied: figures are only written to files.
    pub dir: PathBuf,
}

/// `<dir>/<fname>/graphs`, created on first save.
pub fn graph_dir(dir: &Path, fname: &str) -> PathBuf {
    dir.join(fname).join("graphs")
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
}

/// Colors as Dynare uses them (`'-k'`, `'-r'`, gray bands...), resolved by the renderer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Color {
    Black,
    Red,
    Blue,
    Green,
    Gray,
    /// i-th color of a categorical palette (stacked bars of a shock decomposition...).
    Palette(usize),
    Rgb(u8, u8, u8),
}

#[derive(Clone, Debug)]
pub enum Series {
    /// y against x (x = 1..n when it is a time index).
    Line { x: Vec<f64>, y: Vec<f64>, color: Color, style: LineStyle, width: f64, label: Option<String> },
    /// Shaded area between lower and upper (HPD intervals, forecast bands).
    Band { x: Vec<f64>, lower: Vec<f64>, upper: Vec<f64>, color: Color, alpha: f64, label: Option<String> },
    /// Stacked bars: one Vec per component (same length as x); positive and negative parts
    /// stack separately as in Dynare's `graph_decomp`.
    StackedBars { x: Vec<f64>, components: Vec<Vec<f64>>, labels: Vec<String> },
    /// Single points (the mode marker in mode_check plots...).
    Points { x: Vec<f64>, y: Vec<f64>, color: Color, label: Option<String> },
    /// Vertical line at x (mode in prior/posterior plots).
    VLine { x: f64, color: Color, style: LineStyle },
    /// Horizontal line at y (zero line of IRFs).
    HLine { y: f64, color: Color, style: LineStyle },
    /// [stream X] MATLAB's `imagesc(z, [vmin vmax])` with the `jet` colormap whose lowest
    /// color is light gray (identification's collinearity plots): cell (i, j) at x = j+1,
    /// y = i+1 with row 1 on top, NaN cells gray, a colorbar on the right. `labels` name the
    /// rows (on the y axis) and the columns (as the panel's `x_ticks` would).
    HeatMap { z: Vec<Vec<f64>>, vmin: f64, vmax: f64, labels: Vec<String> },
}

#[derive(Clone, Debug, Default)]
pub struct Panel {
    pub title: String,
    pub series: Vec<Series>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    /// Tick labels for a time axis (dates), aligned with x = 1..n; the renderer thins them.
    pub x_ticks: Option<Vec<(f64, String)>>,
    pub legend: bool,
}

#[derive(Clone, Debug)]
pub struct Figure {
    /// File stem without fname prefix, e.g. "PriorsAndPosteriors1", "CheckPlots2", "IRF_ea".
    pub name: String,
    /// Figure title (Dynare's figure name), may be empty.
    pub title: String,
    pub rows: usize,
    pub cols: usize,
    /// Row-major, at most rows*cols panels.
    pub panels: Vec<Panel>,
}

/// Dynare's `pltorg`: number of panels per figure (nstar, at most 9: 3x3) and the grid.
pub fn pltorg(number: usize) -> (usize /*nbplt*/, usize /*nr*/, usize /*nc*/, usize /*per figure*/) {
    let nstar = number.min(9);
    let (nr, nc) = match nstar {
        0 | 1 => (1, 1),
        2 => (2, 1),
        3 => (3, 1),
        4 => (2, 2),
        5 | 6 => (3, 2),
        _ => (3, 3),
    };
    (number.div_ceil(9).max(1), nr, nc, nstar.max(1))
}

/// Dynare's `pltorg.m` with all its outputs: `(nbplt, nr, nc, lr, lc, nstar)`. `lr`/`lc` (the
/// grid of the last figure when `number > 9`) are 0 when `number <= 9`, as in MATLAB; the
/// Dynare plotting functions ignore them and keep `nr x nc` for the last figure too.
pub fn pltorg_full(number: usize) -> (usize, usize, usize, usize, usize, usize) {
    let small = |n: usize| match n {
        1 => (1, 1),
        2 => (2, 1),
        3 => (3, 1),
        4 => (2, 2),
        5 | 6 => (3, 2),
        7..=9 => (3, 3),
        _ => (0, 0),
    };
    if number == 0 {
        return (0, 0, 0, 0, 0, 9);
    }
    if number <= 9 {
        let (nr, nc) = small(number);
        return (1, nr, nc, 0, 0, 9);
    }
    let nbplt = number.div_ceil(9);
    let reste = number - (nbplt - 1) * 9;
    let (lr, lc) = small(reste);
    (nbplt, 3, 3, lr, lc, 9)
}

impl GraphOptions {
    /// `nograph` / `graph_format` of a command's options (`graph_format=pdf`,
    /// `graph_format=(pdf,eps)`, `none`); figures go to [`graph_dir`].
    pub fn from_options(o: &crate::modfile::Options, dir: &Path, fname: &str) -> GraphOptions {
        use crate::modfile::{OptValue, opt};
        let mut names: Vec<String> = Vec::new();
        match opt(o, "graph_format") {
            Some(OptValue::Str(s)) => names.push(s.clone()),
            Some(OptValue::List(l)) => {
                for v in l {
                    if let OptValue::Str(s) = v {
                        names.push(s.clone());
                    }
                }
            }
            _ => names.push("pdf".into()),
        }
        let mut formats = Vec::new();
        let mut none = false;
        for n in &names {
            match GraphFormat::parse(n) {
                Some(f) if !formats.contains(&f) => formats.push(f),
                Some(_) => {}
                None if n.eq_ignore_ascii_case("none") => none = true,
                None => println!("WARNING: graph_format `{n}` is not supported; ignored."),
            }
        }
        if formats.is_empty() && !none {
            formats.push(GraphFormat::Pdf);
        }
        let enabled = opt(o, "nograph").is_none() && !formats.is_empty();
        GraphOptions { enabled, formats, dir: graph_dir(dir, fname) }
    }
}

/// Time-axis ticks from the date column of a data table: ISO dates three months apart become
/// Dynare's `2003Q1`, other `yyyy-mm-..` dates `2003M7`; anything else is kept as is. `x = 1..n`.
pub fn date_ticks(dates: &[String]) -> Vec<(f64, String)> {
    let parsed: Vec<Option<(i32, u32)>> = dates
        .iter()
        .map(|d| {
            let mut it = d.split(['-', '/']);
            let y: i32 = it.next()?.trim().parse().ok()?;
            let m: u32 = it.next()?.trim().parse().ok()?;
            (1..=12).contains(&m).then_some((y, m))
        })
        .collect();
    let all = !parsed.is_empty() && parsed.iter().all(Option::is_some);
    let quarterly = all
        && parsed.windows(2).all(|w| {
            let (a, b) = (w[0].unwrap(), w[1].unwrap());
            (b.0 * 12 + b.1 as i32) - (a.0 * 12 + a.1 as i32) == 3
        });
    dates
        .iter()
        .enumerate()
        .map(|(k, d)| {
            let label = match parsed[k] {
                Some((y, m)) if quarterly => format!("{y}Q{}", (m - 1) / 3 + 1),
                Some((y, m)) if all => format!("{y}M{m}"),
                _ => d.clone(),
            };
            ((k + 1) as f64, label)
        })
        .collect()
}

/// Writes `fig` to `opts.dir/<stem>.<ext>` for each format (no fname prefix: Dynare's
/// `TracePlot_*` files). PNG is the SVG rasterized by [`png::svg_to_png`].
pub fn save_as(fig: &Figure, stem: &str, opts: &GraphOptions) -> Result<Vec<PathBuf>> {
    if !opts.enabled {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(&opts.dir).with_context(|| format!("creating {}", opts.dir.display()))?;
    let canvas = layout::render(fig);
    let mut out = Vec::new();
    for f in &opts.formats {
        let (bytes, ext) = match f {
            GraphFormat::Pdf => (canvas.to_pdf(), "pdf"),
            GraphFormat::Eps => (canvas.to_eps(), "eps"),
            GraphFormat::Svg => (canvas.to_svg().into_bytes(), "svg"),
            GraphFormat::Png => (png::svg_to_png(&canvas.to_svg())?, "png"), // [stream X]
        };
        let p = opts.dir.join(format!("{stem}.{ext}"));
        std::fs::write(&p, bytes).with_context(|| format!("writing {}", p.display()))?;
        out.push(p);
    }
    Ok(out)
}

/// Writes `fig` in every requested format to `opts.dir/<fname>_<fig.name>.<ext>`; returns the
/// paths written (empty when graphs are disabled).
pub fn save(fig: &Figure, fname: &str, opts: &GraphOptions) -> Result<Vec<PathBuf>> {
    save_as(fig, &format!("{fname}_{}", fig.name), opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pltorg_matches_dynare() {
        // [nbplt,nr,nc,lr,lc,nstar] = pltorg(n) in MATLAB
        assert_eq!(pltorg_full(1), (1, 1, 1, 0, 0, 9));
        assert_eq!(pltorg_full(5), (1, 3, 2, 0, 0, 9));
        assert_eq!(pltorg_full(8), (1, 3, 3, 0, 0, 9));
        assert_eq!(pltorg_full(18), (2, 3, 3, 3, 3, 9));
        assert_eq!(pltorg_full(73), (9, 3, 3, 1, 1, 9));
        assert_eq!(pltorg_full(14), (2, 3, 3, 3, 2, 9));
        for n in 1..40 {
            let (a, b, c, _, _, _) = pltorg_full(n);
            let (a2, b2, c2, s) = pltorg(n);
            assert_eq!((a, b, c), (a2, b2, c2), "{n}");
            assert_eq!(s, n.min(9));
        }
    }

    #[test]
    fn quarterly_dates() {
        let d: Vec<String> = ["2003-03-31", "2003-06-30", "2003-09-30", "2003-12-31", "2004-03-31"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let t = date_ticks(&d);
        assert_eq!(t[0], (1.0, "2003Q1".to_string()));
        assert_eq!(t[3].1, "2003Q4");
        assert_eq!(t[4].1, "2004Q1");
    }
}
