// SPDX-License-Identifier: GPL-3.0-or-later
//! Monte Carlo filtering: `+gsa/stability_mapping_univariate.m` (Smirnov tests per
//! parameter), `+gsa/stability_mapping_bivariate.m` (significant correlations inside a
//! subset), `+gsa/monte_carlo_filtering_analysis.m` (both, the printed tables and the
//! `scatter_mcf` figure).

use super::Ctx;
use super::stats::{corrcoef, dyntable, num2str, smirnov_test};
use crate::graphics::{Color, Figure, LineStyle, Panel, Series};

/// Column `j` of `x` restricted to `rows`.
pub fn column(x: &[Vec<f64>], rows: &[usize], j: usize) -> Vec<f64> {
    rows.iter().map(|&r| x[r][j]).collect()
}

/// `[proba, dproba] = stability_mapping_univariate(lpmat, ibeha, inobeha, ..., iplot=0)`.
pub fn univariate(x: &[Vec<f64>], ibeha: &[usize], inobeha: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let npar = x.first().map_or(0, Vec::len);
    (0..npar)
        .map(|j| {
            let (_, p, d) = smirnov_test(&column(x, ibeha, j), &column(x, inobeha, j), 0.05, 0);
            (p, d)
        })
        .unzip()
}

/// `stability_mapping_bivariate.m`: pairs `(j, i)` (i > j, 0-based) of columns of `x` whose
/// correlation exceeds `alpha2` in absolute value with p-value below `pvalue_crit`; prints
/// the table unless `noprint` (the stability map runs with `options_.noprint = 1`).
pub fn bivariate(
    x: &[Vec<f64>],
    alpha2: f64,
    pvalue_crit: f64,
    names: &[String],
    case_name: &str,
    noprint: bool,
    out: &mut String,
) -> Vec<(usize, usize, f64)> {
    let npar = x.first().map_or(0, Vec::len);
    let cols: Vec<Vec<f64>> = (0..npar).map(|j| x.iter().map(|r| r[j]).collect()).collect();
    let (c0, pv) = corrcoef(&cols);
    let mut pairs = Vec::new();
    for j in 0..npar {
        for i in j + 1..npar {
            if c0[i][j].abs() > alpha2 && pv[j][i] < pvalue_crit {
                pairs.push((j, i, c0[i][j]));
            }
        }
    }
    if pairs.is_empty() {
        out.push('\n');
        out.push_str(&format!(
            "No correlation term with pvalue <{} and |corr. coef.| >{} found for {}\n",
            num2str(pvalue_crit),
            num2str(alpha2),
            case_name
        ));
    } else if !noprint {
        out.push('\n');
        let labels: Vec<String> = pairs.iter().map(|&(j, i, _)| format!("[{},{}]", names[j], names[i])).collect();
        let vals: Vec<Vec<f64>> = pairs.iter().map(|p| vec![p.2]).collect();
        out.push_str(&dyntable(
            &format!("Correlation analysis for {case_name}"),
            &["Parameters", "corrcoef"],
            &labels,
            &vals,
            0,
            7,
            3,
        ));
    }
    pairs
}

/// Labels and thresholds of one `monte_carlo_filtering_analysis` call (`options_mcf`).
pub struct McfSpec<'a> {
    pub pvalue_ks: f64,
    pub pvalue_corr: f64,
    pub alpha2: f64,
    pub param_names: &'a [String],
    /// File stem of the figure (`amcf_name`) and its title.
    pub amcf_name: String,
    pub amcf_title: String,
    pub beha_title: String,
    pub nobeha_title: String,
    pub title: String,
    /// Marker position (`xparam1`), if any.
    pub xparam1: Option<Vec<f64>>,
    /// `options_.noprint` (correlation tables).
    pub noprint: bool,
}

/// Result of one analysis: the Smirnov statistics and the parameters that drive the split.
#[derive(Debug, Clone)]
pub struct McfResult {
    pub proba: Vec<f64>,
    pub dproba: Vec<f64>,
    /// Parameters with `proba < pvalue_ks` (sorted by p-value), then those in significant
    /// correlations (`indmcf`), 0-based.
    pub indmcf: Vec<usize>,
    pub corr_beha: Vec<(usize, usize, f64)>,
    pub corr_nobeha: Vec<(usize, usize, f64)>,
}

/// Port of `+gsa/monte_carlo_filtering_analysis.m`.
pub fn analysis(ctx: &Ctx, x: &[Vec<f64>], ibeha: &[usize], inobeha: &[usize], spec: &McfSpec) -> McfResult {
    let mut out = String::new();
    let (proba, dproba) = univariate(x, ibeha, inobeha);
    let mut indmcf: Vec<usize> = (0..proba.len()).filter(|&j| proba[j] < spec.pvalue_ks).collect();
    indmcf.sort_by(|&a, &b| proba[a].total_cmp(&proba[b]).then(a.cmp(&b)));
    if !indmcf.is_empty() {
        out.push('\n');
        let labels: Vec<String> = indmcf.iter().map(|&j| spec.param_names[j].clone()).collect();
        let vals: Vec<Vec<f64>> = indmcf.iter().map(|&j| vec![dproba[j], proba[j]]).collect();
        out.push_str(&dyntable(
            &format!("Smirnov statistics in driving {}", spec.title),
            &["Parameter", "d-stat", "p-value"],
            &labels,
            &vals,
            3,
            16,
            3,
        ));
    }
    let (mut corr_beha, mut corr_nobeha) = (Vec::new(), Vec::new());
    if ibeha.len() > 10 && inobeha.len() > 10 {
        let xb: Vec<Vec<f64>> = ibeha.iter().map(|&r| x[r].clone()).collect();
        let xn: Vec<Vec<f64>> = inobeha.iter().map(|&r| x[r].clone()).collect();
        corr_beha =
            bivariate(&xb, spec.alpha2, spec.pvalue_corr, spec.param_names, &spec.beha_title, spec.noprint, &mut out);
        corr_nobeha =
            bivariate(&xn, spec.alpha2, spec.pvalue_corr, spec.param_names, &spec.nobeha_title, spec.noprint, &mut out);
        // indcorr = union(indcorr1(:), indcorr2(:)) minus indmcf, sorted
        let mut indcorr: Vec<usize> = corr_beha.iter().chain(&corr_nobeha).flat_map(|&(a, b, _)| [a, b]).collect();
        indcorr.sort_unstable();
        indcorr.dedup();
        indcorr.retain(|k| !indmcf.contains(k));
        indmcf.extend(indcorr);
    }
    if !indmcf.is_empty() && ctx.graphs_enabled() {
        out.push('\n');
        let xb: Vec<Vec<f64>> = ibeha.iter().map(|&r| indmcf.iter().map(|&j| x[r][j]).collect()).collect();
        let xn: Vec<Vec<f64>> = inobeha.iter().map(|&r| indmcf.iter().map(|&j| x[r][j]).collect()).collect();
        let names: Vec<String> = indmcf.iter().map(|&j| spec.param_names[j].clone()).collect();
        let xp = spec.xparam1.as_ref().map(|v| indmcf.iter().map(|&j| v[j]).collect::<Vec<_>>());
        let fig = scatter_mcf(
            &xb,
            &xn,
            &names,
            &spec.amcf_name,
            &spec.amcf_title,
            xp.as_deref(),
            &spec.beha_title,
            &spec.nobeha_title,
        );
        ctx.save(&fig);
    }
    print!("{out}");
    McfResult { proba, dproba, indmcf, corr_beha, corr_nobeha }
}

/// Step coordinates of `+gsa/cumplot.m` (`stairs([-inf; sort(x); inf], [0:n n]/n)`), with the
/// infinite ends clipped to the data range.
pub fn cumplot(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut s: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    let (mut xs, mut ys) = (Vec::with_capacity(2 * n + 2), Vec::with_capacity(2 * n + 2));
    if n == 0 {
        return (xs, ys);
    }
    let nf = n as f64;
    xs.push(s[0]);
    ys.push(0.0);
    for (k, &v) in s.iter().enumerate() {
        xs.push(v);
        ys.push(k as f64 / nf);
        xs.push(v);
        ys.push((k + 1) as f64 / nf);
    }
    (xs, ys)
}

pub fn cdf_series(x: &[f64], color: Color, style: LineStyle, width: f64, label: Option<String>) -> Series {
    let (x, y) = cumplot(x);
    Series::Line { x, y, color, style, width, label }
}

/// `+gsa/scatter_mcf.m`: p x p matrix of panels, CDFs of both subsets on the diagonal
/// (blue = behavioural, red = non-behavioural), scatter plots off the diagonal.
#[allow(clippy::too_many_arguments)]
pub fn scatter_mcf(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    names: &[String],
    fname: &str,
    title: &str,
    xparam1: Option<&[f64]>,
    beha: &str,
    nobeha: &str,
) -> Figure {
    let p = names.len();
    let col = |m: &[Vec<f64>], j: usize| m.iter().map(|r| r[j]).collect::<Vec<f64>>();
    let mut panels = Vec::with_capacity(p * p);
    for j in 0..p {
        for i in 0..p {
            let mut panel = Panel::default();
            if i == j {
                panel.series.push(cdf_series(&col(x, j), Color::Blue, LineStyle::Solid, 1.5, Some(beha.to_string())));
                panel.series.push(cdf_series(&col(y, j), Color::Red, LineStyle::Solid, 1.5, Some(nobeha.to_string())));
                if let Some(xp) = xparam1 {
                    panel.series.push(Series::VLine { x: xp[j], color: Color::Black, style: LineStyle::Dashed });
                }
            } else {
                let pts =
                    |m: &[Vec<f64>], c: Color| Series::Points { x: col(m, i), y: col(m, j), color: c, label: None };
                if j > i {
                    panel.series.push(pts(x, Color::Blue));
                    panel.series.push(pts(y, Color::Red));
                } else {
                    panel.series.push(pts(y, Color::Red));
                    panel.series.push(pts(x, Color::Blue));
                }
                if let Some(xp) = xparam1 {
                    panel.series.push(Series::Points {
                        x: vec![xp[i]],
                        y: vec![xp[j]],
                        color: Color::Green,
                        label: None,
                    });
                }
            }
            if j == 0 {
                panel.title = names[i].clone();
            }
            if i == 0 {
                panel.y_label = Some(names[j].clone());
            }
            panels.push(panel);
        }
    }
    Figure { name: fname.to_string(), title: title.to_string(), rows: p, cols: p, panels }
}

/// `+gsa/scatter_plots.m`: scatter-plot matrix of the columns of `x` (CDFs on the diagonal).
pub fn scatter_plots(x: &[Vec<f64>], names: &[String], fname: &str, title: &str) -> Figure {
    let p = names.len();
    let col = |j: usize| x.iter().map(|r| r[j]).collect::<Vec<f64>>();
    let mut panels = Vec::with_capacity(p * p);
    for j in 0..p {
        for i in 0..p {
            let mut panel = Panel::default();
            if i == j {
                panel.series.push(cdf_series(&col(j), Color::Blue, LineStyle::Solid, 1.5, None));
            } else {
                panel.series.push(Series::Points { x: col(i), y: col(j), color: Color::Blue, label: None });
            }
            if j == 0 {
                panel.title = names[i].clone();
            }
            if i == 0 {
                panel.y_label = Some(names[j].clone());
            }
            panels.push(panel);
        }
    }
    Figure { name: fname.to_string(), title: title.to_string(), rows: p, cols: p, panels }
}
