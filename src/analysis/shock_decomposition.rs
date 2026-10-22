// SPDX-License-Identifier: GPL-3.0-or-later
//! Historical shock decomposition: port of `shock_decomposition/shock_decomposition.m`
//! (order 1, no forecast extension, no epilogue), the shock grouping of
//! `plot_shock_decomposition.m` (`make_the_groups`) and the stacked-bar figure of
//! `graph_decomp.m`.
//!
//! `z[v]` is a (nshocks+2) x gend matrix per endogenous variable v (declaration order):
//! rows 0..nshocks are the contributions of each shock, row nshocks the initial conditions
//! (smoothed deviation minus the sum of the shock contributions), row nshocks+1 the smoothed
//! deviation from the steady state.

use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::graphics::{Color, Figure, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::solver::DecisionRule;

#[derive(Clone, Debug)]
pub struct ShockDecomposition {
    /// One (nshocks+2) x gend matrix per endogenous variable (declaration order).
    pub z: Vec<Mat>,
    pub nshocks: usize,
}

/// `shock_decomposition.m`'s recursion. `smoothed_dev`: n_endo x gend smoothed deviations
/// (declaration order, `Smoothed_Variables_deviation_from_mean`), `eps`: nshocks x gend
/// smoothed shocks, `init_state` = `options_.shock_decomp.init_state` (shocks contribute only
/// from period init_state+1 on).
pub fn shock_decomposition(dr: &DecisionRule, smoothed_dev: &Mat, eps: &Mat, init_state: usize) -> ShockDecomposition {
    let n = smoothed_dev.nrows();
    let gend = smoothed_dev.ncols();
    let nshocks = eps.nrows();
    // i_state = dr.order_var(nstatic + (1:nspred)): declaration indices of the states.
    let i_state: Vec<usize> = (dr.nstatic..dr.nstatic + dr.nspred).map(|k| dr.order_var[k]).collect();
    // ghx(inv_order_var,:), ghu(inv_order_var,:): declaration-order rows
    let ghx = Mat::from_fn(n, dr.ghx.ncols(), |i, j| dr.ghx[(dr.inv_order_var[i], j)]);
    let ghu = Mat::from_fn(n, nshocks, |i, j| dr.ghu[(dr.inv_order_var[i], j)]);
    let mut prev = Mat::zeros(n, nshocks);
    let mut z: Vec<Mat> = (0..n).map(|_| Mat::zeros(nshocks + 2, gend)).collect();
    for t in 0..gend {
        let mut cur = Mat::zeros(n, nshocks);
        if t > 0 {
            let tempx = Mat::from_fn(i_state.len(), nshocks, |a, k| prev[(i_state[a], k)]);
            cur = &ghx * tempx;
        }
        if t + 1 > init_state {
            for v in 0..n {
                for k in 0..nshocks {
                    cur[(v, k)] += ghu[(v, k)] * eps[(k, t)];
                }
            }
        }
        for v in 0..n {
            let total = smoothed_dev[(v, t)];
            let mut s = 0.0;
            for k in 0..nshocks {
                z[v][(k, t)] = cur[(v, k)];
                s += cur[(v, k)];
            }
            z[v][(nshocks + 1, t)] = total;
            z[v][(nshocks, t)] = total - s;
        }
        prev = cur;
    }
    ShockDecomposition { z, nshocks }
}

/// A `shock_groups(name=...)` block: (label, shock indices) per group.
#[derive(Clone, Debug)]
pub struct ShockGroups {
    pub name: String,
    pub groups: Vec<(String, Vec<usize>)>,
}

/// `make_the_groups` of plot_shock_decomposition.m: components = one row per group, then
/// "Others + Initial Values" (shocks in no group plus initial conditions; labelled
/// "Initial values" when no shock is left over), then the smoothed total. Returns the
/// grouped z for one variable and the component labels (without the total).
pub fn group(z: &Mat, nshocks: usize, groups: &ShockGroups) -> (Mat, Vec<String>) {
    let gend = z.ncols();
    let ng = groups.groups.len();
    let mut zz = Mat::zeros(ng + 2, gend);
    let mut used = vec![false; nshocks];
    for (g, (_, shocks)) in groups.groups.iter().enumerate() {
        for &k in shocks {
            if !used[k] {
                used[k] = true;
                for t in 0..gend {
                    zz[(g, t)] += z[(k, t)];
                }
            }
        }
    }
    let mut others_nonzero = false;
    for t in 0..gend {
        let mut o = z[(nshocks, t)];
        for k in (0..nshocks).filter(|&k| !used[k]) {
            o += z[(k, t)];
            others_nonzero |= z[(k, t)] != 0.0;
        }
        zz[(ng, t)] = o;
        zz[(ng + 1, t)] = z[(nshocks + 1, t)];
    }
    let mut labels: Vec<String> = groups.groups.iter().map(|(l, _)| l.clone()).collect();
    labels.push(if others_nonzero { "Others + Initial Values".into() } else { "Initial values".into() });
    (zz, labels)
}

/// Dynare's `get_parameter_set_name`.
pub fn parameter_set_name(parameter_set: &str) -> &'static str {
    match parameter_set {
        "posterior_mode" => "Posterior mode",
        "posterior_mean" => "Posterior mean",
        "posterior_median" => "Posterior median",
        "mle_mode" => "ML mode",
        "prior_mode" => "Prior mode",
        "prior_mean" => "Prior mean",
        "calibration" => "Calibration",
        _ => "",
    }
}

/// `graph_decomp.m`'s figure for one variable: stacked bars of the components (positive and
/// negative parts stacked separately) and the smoothed deviation as a black line. File
/// `<fname>_shock_decomposition_<var>[_group_<g>]_<parameter_set>` (Dynare's naming), or
/// None when the bars are flat (`ymax - ymin < 1e-6`, Dynare skips the figure).
pub fn decomposition_figure(
    var: &str,
    zv: &Mat,
    labels: &[String],
    parameter_set: &str,
    group: Option<&str>,
    x_ticks: Option<Vec<(f64, String)>>,
) -> Option<Figure> {
    let comp = zv.nrows() - 1;
    let gend = zv.ncols();
    let (mut ymax, mut ymin) = (f64::NEG_INFINITY, f64::INFINITY);
    for t in 0..gend {
        let (p, m): (f64, f64) =
            (0..comp).map(|k| zv[(k, t)]).fold((0.0, 0.0), |(p, m), v| (p + v.max(0.0), m + v.min(0.0)));
        ymax = ymax.max(p * 1.1);
        ymin = ymin.min(m * 1.1);
    }
    if ymax - ymin < 1e-6 {
        return None;
    }
    let x: Vec<f64> = (1..=gend).map(|t| t as f64).collect();
    let row = |k: usize| zv.row(k).iter().copied().collect::<Vec<f64>>();
    let series = vec![
        Series::StackedBars { x: x.clone(), components: (0..comp).map(row).collect(), labels: labels.to_vec() },
        Series::Line { x, y: row(comp), color: Color::Black, style: LineStyle::Solid, width: 2.0, label: None },
    ];
    let fig_name_long = group.map(|g| format!(" group {g}")).unwrap_or_default();
    let fig_name = fig_name_long.replace(' ', "_");
    Some(Figure {
        name: format!("shock_decomposition_{var}{fig_name}_{parameter_set}"),
        title: format!("Historical shock decomposition{fig_name_long}: {var} ({}).", parameter_set_name(parameter_set)),
        rows: 1,
        cols: 1,
        panels: vec![Panel { title: var.into(), series, x_ticks, legend: true, ..Default::default() }],
    })
}

/// `<fname>_shock_decomposition.csv`: `date,variable,component,value` (long format; the
/// components are the shocks (or groups), `Initial values` and `Smoothed`).
pub fn write_csv(path: &Path, rows: &[(String, Mat, Vec<String>)], dates: &[String]) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut w = String::from("date,variable,component,value\n");
    for (var, zv, labels) in rows {
        if labels.len() + 1 != zv.nrows() {
            bail!("shock decomposition CSV: {} labels for {} rows", labels.len(), zv.nrows());
        }
        for t in 0..zv.ncols() {
            let date = dates.get(t).cloned().unwrap_or_else(|| (t + 1).to_string());
            for k in 0..zv.nrows() {
                let c = labels.get(k).map_or("Smoothed", String::as_str);
                w += &format!("{date},{var},{c},{}\n", zv[(k, t)]);
            }
        }
    }
    std::fs::write(path, w).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouping_sums_and_others() {
        // 3 shocks + init + total, 2 periods
        let z = Mat::from_row_slice(5, 2, &[1., 2., 3., 4., 5., 6., 0.5, 0.5, 9.5, 12.5]);
        let g = ShockGroups { name: "g".into(), groups: vec![("A".into(), vec![0, 2])] };
        let (zz, labels) = group(&z, 3, &g);
        assert_eq!(labels, ["A", "Others + Initial Values"]);
        assert_eq!(zz.row(0).iter().copied().collect::<Vec<_>>(), [6., 8.]);
        assert_eq!(zz.row(1).iter().copied().collect::<Vec<_>>(), [3.5, 4.5]);
        assert_eq!(zz.row(2).iter().copied().collect::<Vec<_>>(), [9.5, 12.5]);
    }
}
