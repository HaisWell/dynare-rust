// SPDX-License-Identifier: GPL-3.0-or-later
//! Figure -> display list: the subplot grid, axes with MATLAB-like boxed axes and inward
//! ticks, nice tick values, date ticks, the series, panel and shared legends.

use super::canvas::{Anchor, Canvas, Op, Rgb, Stroke};
use super::font::{CAP_HEIGHT, text_width};
use super::{Color, Figure, LineStyle, Panel, Series};

/// Page size in points (10 x 7.5 in, MATLAB's figure aspect ratio).
pub const PAGE_W: f64 = 720.0;
pub const PAGE_H: f64 = 540.0;

const TITLE_SIZE: f64 = 10.0;
const FIG_TITLE_SIZE: f64 = 12.0;
const TICK_SIZE: f64 = 7.5;
const LABEL_SIZE: f64 = 8.5;
const LEGEND_SIZE: f64 = 8.0;
const AXIS_GRAY: Rgb = Rgb(0.15, 0.15, 0.15);
const GRID_GRAY: Rgb = Rgb(0.9, 0.9, 0.9);

pub fn resolve(c: Color) -> Rgb {
    match c {
        Color::Black => Rgb(0.0, 0.0, 0.0),
        Color::Red => Rgb(0.85, 0.1, 0.1),
        Color::Blue => Rgb(0.0, 0.447, 0.741),
        Color::Green => Rgb(0.0, 0.6, 0.0),
        Color::Gray => Rgb(0.7, 0.7, 0.7),
        Color::Palette(i) => super::palette_override(i).unwrap_or(PALETTE[i % PALETTE.len()]), // colormap
        Color::Rgb(r, g, b) => Rgb(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0),
    }
}

/// MATLAB's `lines` colors, then the rest of a 12-color categorical set.
const PALETTE: [Rgb; 12] = [
    Rgb(0.0, 0.447, 0.741),
    Rgb(0.85, 0.325, 0.098),
    Rgb(0.929, 0.694, 0.125),
    Rgb(0.494, 0.184, 0.556),
    Rgb(0.466, 0.674, 0.188),
    Rgb(0.301, 0.745, 0.933),
    Rgb(0.635, 0.078, 0.184),
    Rgb(0.5, 0.5, 0.5),
    Rgb(0.737, 0.741, 0.133),
    Rgb(0.09, 0.745, 0.812),
    Rgb(0.89, 0.467, 0.761),
    Rgb(0.549, 0.337, 0.294),
];

fn dash(style: LineStyle, width: f64) -> Vec<f64> {
    let w = width.max(0.8);
    match style {
        LineStyle::Solid => vec![],
        LineStyle::Dashed => vec![4.0 * w, 2.5 * w],
        LineStyle::Dotted => vec![w, 1.8 * w],
    }
}

fn stroke(color: Color, style: LineStyle, width: f64) -> Stroke {
    let width = if width > 0.0 { width } else { 0.5 };
    Stroke { color: resolve(color), width, dash: dash(style, width) }
}

/// One legend entry: how to draw its key.
#[derive(Clone, Debug, PartialEq)]
enum Key {
    Line(Rgb, f64, Vec<f64>),
    Patch(Rgb, f64),
    Dot(Rgb),
}

fn legend_entries(p: &Panel) -> Vec<(String, Key)> {
    let mut v = Vec::new();
    for s in &p.series {
        match s {
            Series::Line { color, style, width, label: Some(l), .. } => {
                let st = stroke(*color, *style, *width);
                v.push((l.clone(), Key::Line(st.color, st.width, st.dash)))
            }
            Series::Band { color, alpha, label: Some(l), .. } => {
                v.push((l.clone(), Key::Patch(resolve(*color), *alpha)))
            }
            Series::Points { color, label: Some(l), .. } => v.push((l.clone(), Key::Dot(resolve(*color)))),
            Series::StackedBars { labels, components, .. } => {
                for (k, l) in labels.iter().enumerate().take(components.len()) {
                    v.push((l.clone(), Key::Patch(resolve(Color::Palette(k)), 1.0)));
                }
            }
            _ => {}
        }
    }
    v
}

/// Data range of a panel: (xmin, xmax, ymin, ymax), NaN-free; None when empty.
fn data_range(p: &Panel) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
    let mut xr: Option<(f64, f64)> = None;
    let mut yr: Option<(f64, f64)> = None;
    let add = |r: &mut Option<(f64, f64)>, v: f64| {
        if v.is_finite() {
            *r = Some(match *r {
                None => (v, v),
                Some((a, b)) => (a.min(v), b.max(v)),
            });
        }
    };
    for s in &p.series {
        match s {
            Series::Line { x, y, .. } | Series::Points { x, y, .. } => {
                for (a, b) in x.iter().zip(y) {
                    if a.is_finite() && b.is_finite() {
                        add(&mut xr, *a);
                        add(&mut yr, *b);
                    }
                }
            }
            Series::Band { x, lower, upper, .. } => {
                for k in 0..x.len().min(lower.len()).min(upper.len()) {
                    if x[k].is_finite() && lower[k].is_finite() && upper[k].is_finite() {
                        add(&mut xr, x[k]);
                        add(&mut yr, lower[k]);
                        add(&mut yr, upper[k]);
                    }
                }
            }
            Series::StackedBars { x, components, .. } => {
                for (t, xv) in x.iter().enumerate() {
                    let (mut pos, mut neg) = (0.0, 0.0);
                    for c in components {
                        let v = c.get(t).copied().unwrap_or(0.0);
                        if v.is_finite() {
                            if v >= 0.0 { pos += v } else { neg += v }
                        }
                    }
                    add(&mut xr, *xv - 0.5);
                    add(&mut xr, *xv + 0.5);
                    add(&mut yr, pos);
                    add(&mut yr, neg);
                }
            }
            Series::VLine { x, .. } => add(&mut xr, *x),
            Series::HLine { y, .. } => add(&mut yr, *y),
            // [stream X]
            Series::HeatMap { z, .. } => {
                let nc = z.iter().map(Vec::len).max().unwrap_or(0);
                xr = Some((0.5, nc as f64 + 0.5));
                yr = Some((0.5, z.len() as f64 + 0.5));
            }
        }
    }
    (xr, yr)
}

/// [stream X] MATLAB's `jet` colormap at `v` in [0, 1].
pub fn jet(v: f64) -> Rgb {
    let f = |c: f64| (1.5 - (4.0 * v - c).abs()).clamp(0.0, 1.0);
    Rgb(f(3.0), f(2.0), f(1.0))
}

/// [stream X] Color of a heat-map cell (`imagesc` + `jet` with the first entry light gray).
fn heat_color(v: f64, vmin: f64, vmax: f64) -> Rgb {
    if !v.is_finite() {
        return Rgb(0.9, 0.9, 0.9);
    }
    let t = ((v - vmin) / (vmax - vmin)).clamp(0.0, 1.0);
    // 64-entry colormap as MATLAB's default length: index 1 is the gray entry
    let k = (t * 63.0).round();
    if k == 0.0 { Rgb(0.9, 0.9, 0.9) } else { jet(k / 63.0) }
}

/// Nice tick step for `range` with about `target` intervals (1, 2, 2.5, 5 x 10^k).
pub fn nice_step(range: f64, target: f64) -> f64 {
    let raw = range / target.max(1.0);
    let mag = 10f64.powf(raw.log10().floor());
    let r = raw / mag;
    let m = if r <= 1.0 {
        1.0
    } else if r <= 2.0 {
        2.0
    } else if r <= 2.5 {
        2.5
    } else if r <= 5.0 {
        5.0
    } else {
        10.0
    };
    m * mag
}

/// Ticks inside [lo, hi].
pub fn ticks(lo: f64, hi: f64, target: f64) -> (Vec<f64>, f64) {
    let step = nice_step(hi - lo, target);
    let first = (lo / step - 1e-9).ceil();
    let mut v = Vec::new();
    let mut k = first;
    while k * step <= hi + 1e-9 * step && v.len() < 100 {
        let t = k * step;
        v.push(if t.abs() < 1e-12 * step { 0.0 } else { t });
        k += 1.0;
    }
    (v, step)
}

/// Tick labels with a common number of decimals and, for very large/small magnitudes, a
/// common power of ten (returned separately, like MATLAB's `x10^k` axis exponent).
pub fn tick_labels(ticks: &[f64], step: f64) -> (Vec<String>, Option<i32>) {
    let maxabs = ticks.iter().fold(0.0f64, |m, t| m.max(t.abs()));
    let mut exp = None;
    let mut scale = 1.0;
    if maxabs >= 1e5 || (maxabs > 0.0 && maxabs < 1e-3) {
        let e = maxabs.log10().floor() as i32;
        exp = Some(e);
        scale = 10f64.powi(e);
    }
    let s = step / scale;
    let decimals = (0..=8usize)
        .find(|&d| {
            ticks.iter().all(|t| {
                let v = t / scale;
                let r: f64 = format!("{v:.d$}").parse().unwrap_or(v);
                (r - v).abs() <= 1e-6 * s
            })
        })
        .unwrap_or(8);
    let labels = ticks
        .iter()
        .map(|t| {
            let v = t / scale;
            let mut l = format!("{v:.decimals$}");
            if l.contains('.') {
                l = l.trim_end_matches('0').trim_end_matches('.').to_string();
            }
            if l == "-0" {
                l = "0".into();
            }
            l
        })
        .collect();
    (labels, exp)
}

/// Axis limits from a data range: tight in x; in y padded by 5% (0 kept as the floor when the
/// data are non-negative and start close to it, e.g. densities).
fn limits(r: Option<(f64, f64)>, pad: bool) -> (f64, f64) {
    let (a, b) = r.unwrap_or((0.0, 1.0));
    if (b - a).abs() <= 1e-300 || (b - a).abs() <= 1e-12 * a.abs().max(b.abs()) {
        let d = if a == 0.0 { 1.0 } else { 0.1 * a.abs() };
        return (a - d, b + d);
    }
    if !pad {
        return (a, b);
    }
    let d = 0.05 * (b - a);
    let lo = if a >= 0.0 && a <= 0.1 * b { 0.0 } else { a - d };
    (lo, b + d)
}

struct Axes {
    x0: f64,
    y0: f64,
    w: f64,
    h: f64,
    xl: (f64, f64),
    yl: (f64, f64),
}

impl Axes {
    fn px(&self, x: f64) -> f64 {
        self.x0 + (x - self.xl.0) / (self.xl.1 - self.xl.0) * self.w
    }
    fn py(&self, y: f64) -> f64 {
        self.y0 + self.h - (y - self.yl.0) / (self.yl.1 - self.yl.0) * self.h
    }
}

/// Pick the date labels to show: every k-th, k chosen so the labels do not overlap; with
/// quarterly/monthly labels k is a multiple of 4/12 when possible and the shown labels
/// start on a year (Q1 / M1).
fn thin_date_ticks(t: &[(f64, String)], axis_w: f64, xl: (f64, f64)) -> Vec<(f64, String)> {
    let inside: Vec<&(f64, String)> = t.iter().filter(|(x, _)| *x >= xl.0 - 1e-9 && *x <= xl.1 + 1e-9).collect();
    if inside.is_empty() {
        return vec![];
    }
    let maxw = inside.iter().map(|(_, s)| text_width(s, TICK_SIZE, false)).fold(0.0, f64::max) + 8.0;
    let per_tick = if inside.len() > 1 { (inside[1].0 - inside[0].0).abs() / (xl.1 - xl.0) * axis_w } else { axis_w };
    let need = (maxw / per_tick.max(1e-9)).ceil().max(1.0) as usize;
    let quarterly = inside.iter().any(|(_, s)| s.contains('Q'));
    let monthly = inside.iter().any(|(_, s)| s.contains('M'));
    let candidates: &[usize] = if quarterly {
        &[1, 2, 4, 8, 12, 16, 20, 40, 80, 120, 200]
    } else if monthly {
        &[1, 2, 3, 6, 12, 24, 36, 60, 120, 240]
    } else {
        &[1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
    };
    // at most ~8 labels
    let need = need.max(inside.len().div_ceil(8));
    let k = candidates.iter().copied().find(|&c| c >= need).unwrap_or(need);
    let start = if quarterly && k % 4 == 0 {
        inside.iter().position(|(_, s)| s.ends_with("Q1")).unwrap_or(0)
    } else if monthly && k % 12 == 0 {
        inside.iter().position(|(_, s)| s.ends_with("M1")).unwrap_or(0)
    } else {
        0
    };
    inside.iter().skip(start % k.max(1)).step_by(k.max(1)).map(|t| (*t).clone()).collect()
}

/// Shrinks (down to 6.5 pt) then truncates `s` to fit `w`.
fn fit_text(s: &str, size: f64, bold: bool, w: f64) -> (String, f64) {
    let mut sz = size;
    while text_width(s, sz, bold) > w && sz > 6.5 {
        sz -= 0.5;
    }
    if text_width(s, sz, bold) <= w {
        return (s.to_string(), sz);
    }
    let mut t: String = s.to_string();
    while !t.is_empty() && text_width(&format!("{t}..."), sz, bold) > w {
        t.pop();
    }
    (format!("{t}..."), sz)
}

/// Series of the panel split at NaNs (lines) into drawable pieces.
fn segments(x: &[f64], y: &[f64]) -> Vec<Vec<(f64, f64)>> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for (a, b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            cur.push((*a, *b));
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Extra room the panel needs below the axes for x tick labels and x label, and left of it.
struct Margins {
    left: f64,
    bottom: f64,
    top: f64,
}

fn draw_panel(c: &mut Canvas, p: &Panel, cell: (f64, f64, f64, f64), shared_legend: bool) {
    let (cx, cy, cw, ch) = cell;
    let (xr, yr) = data_range(p);
    // [stream X] a heat map fixes the limits, labels the rows and needs room for a colorbar
    let heat = p.series.iter().find_map(|s| match s {
        Series::HeatMap { z, vmin, vmax, labels } => Some((z, *vmin, *vmax, labels)),
        _ => None,
    });
    let (xl, yl) = match (&heat, xr, yr) {
        (Some(_), Some(x), Some(y)) => (x, y),
        _ => (limits(xr, false), limits(yr, true)),
    };
    let ny = (ch / 38.0).clamp(3.0, 7.0);
    let (mut yt, ystep) = ticks(yl.0, yl.1, ny);
    let (mut ylab, yexp) = tick_labels(&yt, ystep);
    if let Some((z, _, _, labels)) = &heat {
        // row i is drawn at y = n - i (row 1 on top)
        yt = (0..z.len()).map(|i| (z.len() - i) as f64).collect();
        ylab = (0..z.len()).map(|i| labels.get(i).cloned().unwrap_or_default()).collect();
    }
    let colorbar_w = if heat.is_some() { 36.0 } else { 0.0 };
    let ylab_w = ylab.iter().map(|s| text_width(s, TICK_SIZE, false)).fold(0.0, f64::max);
    let m = Margins {
        left: ylab_w + 6.0 + if p.y_label.is_some() { LABEL_SIZE + 4.0 } else { 0.0 },
        bottom: TICK_SIZE + 6.0 + if p.x_label.is_some() { LABEL_SIZE + 4.0 } else { 0.0 },
        top: if p.title.is_empty() { 6.0 } else { TITLE_SIZE + 6.0 } + if yexp.is_some() { 2.0 } else { 0.0 },
    };
    let ax = Axes {
        x0: cx + m.left,
        y0: cy + m.top,
        w: (cw - m.left - 6.0 - colorbar_w).max(10.0),
        h: (ch - m.top - m.bottom).max(10.0),
        xl,
        yl,
    };

    // x ticks: dates or numbers
    let xticks: Vec<(f64, String)> = match &p.x_ticks {
        Some(t) => thin_date_ticks(t, ax.w, xl),
        None if heat.is_some() => {
            let (z, _, _, labels) = heat.as_ref().unwrap();
            let nc = z.iter().map(Vec::len).max().unwrap_or(0);
            thin_date_ticks(
                &(0..nc).map(|j| ((j + 1) as f64, labels.get(j).cloned().unwrap_or_default())).collect::<Vec<_>>(),
                ax.w,
                xl,
            )
        }
        None => {
            let nx = (ax.w / 40.0).clamp(3.0, 8.0);
            let (xt, xstep) = ticks(xl.0, xl.1, nx);
            let (lab, exp) = tick_labels(&xt, xstep);
            let mut v: Vec<(f64, String)> = xt.into_iter().zip(lab).collect();
            if let Some(e) = exp {
                // put the exponent on the last label's right
                if let Some(last) = v.last_mut() {
                    last.1 = format!("{} \u{d7}1e{e}", last.1);
                }
            }
            v
        }
    };

    // background grid
    let grid = Stroke { color: GRID_GRAY, width: 0.4, dash: vec![] };
    for t in &yt {
        let y = ax.py(*t);
        c.line(vec![(ax.x0, y), (ax.x0 + ax.w, y)], grid.clone());
    }
    for (t, _) in &xticks {
        let x = ax.px(*t);
        c.line(vec![(x, ax.y0), (x, ax.y0 + ax.h)], grid.clone());
    }

    // series
    c.clip(ax.x0, ax.y0, ax.w, ax.h);
    // bands and bars first, lines on top
    for s in &p.series {
        match s {
            Series::Band { x, lower, upper, color, alpha, .. } => {
                let n = x.len().min(lower.len()).min(upper.len());
                let mut run: Vec<usize> = Vec::new();
                let flush = |run: &mut Vec<usize>, c: &mut Canvas| {
                    if run.len() >= 2 {
                        let mut pts: Vec<(f64, f64)> = run.iter().map(|&k| (ax.px(x[k]), ax.py(upper[k]))).collect();
                        pts.extend(run.iter().rev().map(|&k| (ax.px(x[k]), ax.py(lower[k]))));
                        c.polygon(pts, resolve(*color), *alpha, None);
                    }
                    run.clear();
                };
                for k in 0..n {
                    if x[k].is_finite() && lower[k].is_finite() && upper[k].is_finite() {
                        run.push(k);
                    } else {
                        flush(&mut run, c);
                    }
                }
                flush(&mut run, c);
            }
            // [stream X]
            Series::HeatMap { z, vmin, vmax, .. } => {
                let n = z.len();
                for (i, row) in z.iter().enumerate() {
                    for (j, &v) in row.iter().enumerate() {
                        let (x0, x1) = (ax.px(j as f64 + 0.5), ax.px(j as f64 + 1.5));
                        let yc = (n - i) as f64;
                        let (y0, y1) = (ax.py(yc + 0.5), ax.py(yc - 0.5));
                        c.rect(x0, y0, x1 - x0, y1 - y0, Some((heat_color(v, *vmin, *vmax), 1.0)), None);
                    }
                }
            }
            Series::StackedBars { x, components, .. } => {
                let dx = x.windows(2).map(|w| (w[1] - w[0]).abs()).filter(|d| *d > 0.0).fold(f64::INFINITY, f64::min);
                let bw = if dx.is_finite() { 0.8 * dx } else { 0.8 };
                for (t, xv) in x.iter().enumerate() {
                    let (mut pos, mut neg) = (0.0, 0.0);
                    for (k, comp) in components.iter().enumerate() {
                        let v = comp.get(t).copied().unwrap_or(0.0);
                        if !v.is_finite() || v == 0.0 {
                            continue;
                        }
                        let (a, b) = if v > 0.0 {
                            pos += v;
                            (pos - v, pos)
                        } else {
                            neg += v;
                            (neg, neg - v)
                        };
                        let (x0, x1) = (ax.px(xv - bw / 2.0), ax.px(xv + bw / 2.0));
                        c.rect(
                            x0,
                            ax.py(b),
                            x1 - x0,
                            ax.py(a) - ax.py(b),
                            Some((resolve(Color::Palette(k)), 1.0)),
                            None,
                        );
                    }
                }
            }
            _ => {}
        }
    }
    for s in &p.series {
        match s {
            Series::Line { x, y, color, style, width, .. } => {
                for seg in segments(x, y) {
                    let pts: Vec<(f64, f64)> = seg.iter().map(|(a, b)| (ax.px(*a), ax.py(*b))).collect();
                    if pts.len() == 1 {
                        c.ops.push(Op::Circle {
                            cx: pts[0].0,
                            cy: pts[0].1,
                            r: 1.2,
                            fill: resolve(*color),
                            stroke: None,
                        });
                    } else {
                        c.line(pts, stroke(*color, *style, *width));
                    }
                }
            }
            Series::VLine { x, color, style } => {
                let px = ax.px(*x);
                c.line(vec![(px, ax.y0), (px, ax.y0 + ax.h)], stroke(*color, *style, 1.0));
            }
            Series::HLine { y, color, style } => {
                let py = ax.py(*y);
                c.line(vec![(ax.x0, py), (ax.x0 + ax.w, py)], stroke(*color, *style, 0.6));
            }
            _ => {}
        }
    }
    c.unclip();
    // points are drawn unclipped so markers on the axis edge stay whole
    for s in &p.series {
        if let Series::Points { x, y, color, .. } = s {
            for (a, b) in x.iter().zip(y) {
                if a.is_finite()
                    && b.is_finite()
                    && *a >= xl.0
                    && *a <= xl.1
                    && *b >= yl.0 - 1e-12
                    && *b <= yl.1 + 1e-12
                {
                    c.ops.push(Op::Circle {
                        cx: ax.px(*a),
                        cy: ax.py(*b),
                        r: 2.4,
                        fill: resolve(*color),
                        stroke: None,
                    });
                }
            }
        }
    }

    // box and ticks
    let axis = Stroke { color: AXIS_GRAY, width: 0.6, dash: vec![] };
    // [stream X] colorbar of a heat map
    if let Some((_, vmin, vmax, _)) = &heat {
        let (bx, bw) = (ax.x0 + ax.w + 8.0, 10.0);
        let steps = 64;
        for k in 0..steps {
            let v = vmin + (vmax - vmin) * (k as f64 + 0.5) / steps as f64;
            let (ya, yb) =
                (ax.y0 + ax.h * (1.0 - (k + 1) as f64 / steps as f64), ax.y0 + ax.h * (1.0 - k as f64 / steps as f64));
            c.rect(bx, ya, bw, yb - ya, Some((heat_color(v, *vmin, *vmax), 1.0)), None);
        }
        c.rect(bx, ax.y0, bw, ax.h, None, Some(axis.clone()));
        for k in 0..=5 {
            let v = vmin + (vmax - vmin) * k as f64 / 5.0;
            let y = ax.y0 + ax.h * (1.0 - k as f64 / 5.0);
            c.text(
                bx + bw + 2.0,
                y + CAP_HEIGHT * TICK_SIZE / 2.0,
                &crate::deterministic::fmt_g(v),
                TICK_SIZE,
                false,
                Anchor::Start,
                false,
            );
        }
    }
    c.rect(ax.x0, ax.y0, ax.w, ax.h, None, Some(axis.clone()));
    let tl = 3.0;
    for (t, lab) in yt.iter().zip(&ylab) {
        let y = ax.py(*t);
        c.line(vec![(ax.x0, y), (ax.x0 + tl, y)], axis.clone());
        c.line(vec![(ax.x0 + ax.w, y), (ax.x0 + ax.w - tl, y)], axis.clone());
        c.text(ax.x0 - 3.0, y + CAP_HEIGHT * TICK_SIZE / 2.0, lab, TICK_SIZE, false, Anchor::End, false);
    }
    if let Some(e) = yexp {
        c.text(ax.x0, ax.y0 - 2.5, &format!("\u{d7}1e{e}"), TICK_SIZE, false, Anchor::Start, false);
    }
    let base = ax.y0 + ax.h + 3.0 + CAP_HEIGHT * TICK_SIZE;
    let mut last_right = f64::NEG_INFINITY;
    for (t, lab) in &xticks {
        let x = ax.px(*t);
        c.line(vec![(x, ax.y0 + ax.h), (x, ax.y0 + ax.h - tl)], axis.clone());
        c.line(vec![(x, ax.y0), (x, ax.y0 + tl)], axis.clone());
        let w = text_width(lab, TICK_SIZE, false);
        // keep labels inside the cell and apart from each other
        let left = (x - w / 2.0).max(cx).min(cx + cw - w);
        if left > last_right + 3.0 {
            c.text(left, base, lab, TICK_SIZE, false, Anchor::Start, false);
            last_right = left + w;
        }
    }
    if let Some(l) = &p.x_label {
        c.text(ax.x0 + ax.w / 2.0, base + LABEL_SIZE + 3.0, l, LABEL_SIZE, false, Anchor::Middle, false);
    }
    if let Some(l) = &p.y_label {
        c.text(cx + LABEL_SIZE, ax.y0 + ax.h / 2.0, l, LABEL_SIZE, false, Anchor::Middle, true);
    }
    if !p.title.is_empty() {
        let (t, sz) = fit_text(&p.title, TITLE_SIZE, true, cw - 4.0);
        c.text(ax.x0 + ax.w / 2.0, ax.y0 - 5.0, &t, sz, true, Anchor::Middle, false);
    }
    if p.legend && !shared_legend {
        let entries = legend_entries(p);
        if !entries.is_empty() {
            draw_legend_box(c, &entries, &ax);
        }
    }
}

fn draw_key(c: &mut Canvas, key: &Key, x: f64, y: f64, len: f64) {
    match key {
        Key::Line(col, w, d) => c.line(vec![(x, y), (x + len, y)], Stroke { color: *col, width: *w, dash: d.clone() }),
        Key::Patch(col, a) => c.rect(x, y - 3.5, len, 7.0, Some((*col, *a)), None),
        Key::Dot(col) => c.ops.push(Op::Circle { cx: x + len / 2.0, cy: y, r: 2.4, fill: *col, stroke: None }),
    }
}

/// Legend in the upper right corner of the axes (MATLAB's NorthEast).
fn draw_legend_box(c: &mut Canvas, entries: &[(String, Key)], ax: &Axes) {
    let key_len = 16.0;
    let row = LEGEND_SIZE + 3.5;
    let tw = entries.iter().map(|(s, _)| text_width(s, LEGEND_SIZE, false)).fold(0.0, f64::max);
    let w = 6.0 + key_len + 4.0 + tw + 6.0;
    let h = 4.0 + row * entries.len() as f64 + 1.0;
    let x = ax.x0 + ax.w - w - 4.0;
    let y = ax.y0 + 4.0;
    c.rect(x, y, w, h, Some((Rgb::WHITE, 0.9)), Some(Stroke { color: Rgb(0.6, 0.6, 0.6), width: 0.4, dash: vec![] }));
    for (k, (label, key)) in entries.iter().enumerate() {
        let yy = y + 3.0 + row * (k as f64 + 0.5);
        draw_key(c, key, x + 6.0, yy, key_len);
        c.text(
            x + 6.0 + key_len + 4.0,
            yy + CAP_HEIGHT * LEGEND_SIZE / 2.0,
            label,
            LEGEND_SIZE,
            false,
            Anchor::Start,
            false,
        );
    }
}

/// One-row legend centred at the bottom of the page (mode_check's shared legend).
fn draw_shared_legend(c: &mut Canvas, entries: &[(String, Key)], y: f64) {
    let key_len = 22.0;
    let widths: Vec<f64> =
        entries.iter().map(|(s, _)| key_len + 5.0 + text_width(s, LEGEND_SIZE + 1.0, false)).collect();
    let total: f64 = widths.iter().sum::<f64>() + 18.0 * (entries.len() as f64 - 1.0) + 16.0;
    let x0 = (PAGE_W - total) / 2.0;
    c.rect(x0, y - 8.0, total, 16.0, None, Some(Stroke { color: Rgb(0.6, 0.6, 0.6), width: 0.4, dash: vec![] }));
    let mut x = x0 + 8.0;
    for ((label, key), w) in entries.iter().zip(&widths) {
        draw_key(c, key, x, y, key_len);
        c.text(
            x + key_len + 5.0,
            y + CAP_HEIGHT * (LEGEND_SIZE + 1.0) / 2.0,
            label,
            LEGEND_SIZE + 1.0,
            false,
            Anchor::Start,
            false,
        );
        x += w + 18.0;
    }
}

/// Lays the figure out on one page.
pub fn render(fig: &Figure) -> Canvas {
    let mut c = Canvas::new(PAGE_W, PAGE_H);
    let rows = fig.rows.max(1);
    let cols = fig.cols.max(1);
    // A legend shared by every panel that asks for one (same entries) goes once at the bottom.
    let with_legend: Vec<Vec<(String, Key)>> = fig.panels.iter().filter(|p| p.legend).map(legend_entries).collect();
    let shared =
        with_legend.len() > 1 && !with_legend[0].is_empty() && with_legend.iter().all(|e| *e == with_legend[0]);
    let top = if fig.title.is_empty() { 14.0 } else { 34.0 };
    let bottom = if shared { 36.0 } else { 14.0 };
    let (left, right) = (14.0, 14.0);
    if !fig.title.is_empty() {
        let (t, sz) = fit_text(&fig.title, FIG_TITLE_SIZE, true, PAGE_W - 40.0);
        c.text(PAGE_W / 2.0, 22.0, &t, sz, true, Anchor::Middle, false);
    }
    let gap = 8.0;
    let cw = (PAGE_W - left - right - gap * (cols as f64 - 1.0)) / cols as f64;
    let ch = (PAGE_H - top - bottom - gap * (rows as f64 - 1.0)) / rows as f64;
    for (k, p) in fig.panels.iter().enumerate().take(rows * cols) {
        let (r, col) = (k / cols, k % cols);
        let cell = (left + col as f64 * (cw + gap), top + r as f64 * (ch + gap), cw, ch);
        draw_panel(&mut c, p, cell, shared);
    }
    if shared {
        draw_shared_legend(&mut c, &with_legend[0], PAGE_H - 18.0);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_ticks() {
        let (t, s) = ticks(0.0, 1.0, 5.0);
        assert_eq!(s, 0.2);
        assert_eq!(t.len(), 6);
        let (t, s) = ticks(-0.013, 0.027, 4.0);
        assert_eq!(s, 0.01);
        assert_eq!(t, vec![-0.01, 0.0, 0.01, 0.02]);
        let (l, e) = tick_labels(&t, s);
        assert_eq!(l, vec!["-0.01", "0", "0.01", "0.02"]);
        assert_eq!(e, None);
        let (l, _) = tick_labels(&[0.0, 2.5, 5.0], 2.5);
        assert_eq!(l, vec!["0", "2.5", "5"]);
        let (l, e) = tick_labels(&[0.0, 2e-5, 4e-5], 2e-5);
        assert_eq!(e, Some(-5));
        assert_eq!(l, vec!["0", "2", "4"]);
    }

    #[test]
    fn date_thinning_starts_on_q1() {
        let t: Vec<(f64, String)> =
            (0..94).map(|k| ((k + 1) as f64, format!("{}Q{}", 2003 + (k + 1) / 4, (k + 1) % 4 + 1))).collect();
        let v = thin_date_ticks(&t, 200.0, (1.0, 94.0));
        assert!(v.len() >= 3 && v.len() <= 8, "{v:?}");
        assert!(v.iter().all(|(_, s)| s.ends_with("Q1")), "{v:?}");
    }
}
