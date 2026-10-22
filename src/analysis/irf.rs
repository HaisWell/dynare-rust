// SPDX-License-Identifier: GPL-3.0-or-later
//! Impulse responses at order 1: port of `stochastic_solver/irf.m` (via `simult_.m`), the
//! IRF part of `stoch_simul.m` (orthogonalized shocks `cs(:,i)`, `relative_irf`, plot
//! threshold, `pltorg` figures `IRF_<shock>[k]`) and CSV output.

use std::path::Path;

use anyhow::{Context, Result};

use crate::graphics::{Color, Figure, LineStyle, Panel, Series};
use crate::linalg::Mat;
use crate::solver::DecisionRule;

/// `irf(M_, options_, dr, e1, long, ...)` at order 1: n_endo x `periods` deviations from the
/// steady state, declaration order; period 1 is the impact `ghu * e1`.
pub fn irf(dr: &DecisionRule, e1: &[f64], periods: usize) -> Mat {
    let n = dr.ghx.nrows();
    let (nstatic, nspred) = (dr.nstatic, dr.nspred);
    let mut out = Mat::zeros(n, periods);
    if periods == 0 {
        return out;
    }
    let mut y: Vec<f64> = (0..n).map(|r| (0..e1.len()).map(|j| dr.ghu[(r, j)] * e1[j]).sum()).collect();
    for t in 0..periods {
        if t > 0 {
            let st: Vec<f64> = y[nstatic..nstatic + nspred].to_vec();
            for (r, yr) in y.iter_mut().enumerate() {
                *yr = (0..nspred).map(|k| dr.ghx[(r, k)] * st[k]).sum();
            }
        }
        for r in 0..n {
            out[(dr.order_var[r], t)] = y[r];
        }
    }
    out
}

/// IRFs of the requested variables to one orthogonalized shock (`oo_.irfs`).
#[derive(Clone, Debug)]
pub struct ShockIrf {
    pub shock: usize,
    /// nvar x periods, rows in `ivar` order.
    pub y: Mat,
}

/// The IRF loop of `stoch_simul.m` at order 1: for each shock of `shocks` with
/// `cs(i,i) > 5e-7` (`cs` the lower Cholesky factor of Sigma_e), `irf(cs(:,i))`, scaled by
/// `100 / cs(i,i)` under `relative_irf`.
pub fn stoch_simul_irfs(
    dr: &DecisionRule,
    cs: &Mat,
    ivar: &[usize],
    shocks: &[usize],
    periods: usize,
    relative: bool,
) -> Vec<ShockIrf> {
    shocks
        .iter()
        .filter(|&&i| cs[(i, i)] > 5e-7)
        .map(|&i| {
            let e: Vec<f64> = cs.column(i).iter().copied().collect();
            let mut y = irf(dr, &e, periods);
            if relative {
                y *= 100.0 / cs[(i, i)];
            }
            ShockIrf { shock: i, y: y.select_rows(ivar) }
        })
        .collect()
}

/// `getIrfShocksIndx.m`: the listed shocks (unique, sorted) or all of them.
pub fn irf_shock_indices(exo_names: &[String], irf_shocks: Option<&[String]>) -> Result<Vec<usize>> {
    match irf_shocks {
        None => Ok((0..exo_names.len()).collect()),
        Some(list) => {
            let mut v = list
                .iter()
                .map(|s| {
                    exo_names
                        .iter()
                        .position(|e| e == s)
                        .ok_or_else(|| anyhow::anyhow!("irf_shocks: {s} is not an exogenous variable"))
                })
                .collect::<Result<Vec<_>>>()?;
            v.sort_unstable();
            v.dedup();
            Ok(v)
        }
    }
}

/// `<out_dir>/<fname>_IRF_<shock>.csv`: one column per variable, one row per period.
pub fn write_irf_csv(
    out_dir: &Path,
    fname: &str,
    shock_name: &str,
    var_names: &[String],
    y: &Mat,
) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let path = out_dir.join(format!("{fname}_IRF_{shock_name}.csv"));
    let mut w = String::from("period");
    for v in var_names {
        w.push(',');
        w += v;
    }
    w.push('\n');
    for t in 0..y.ncols() {
        w += &(t + 1).to_string();
        for i in 0..y.nrows() {
            w += &format!(",{}", y[(i, t)]);
        }
        w.push('\n');
    }
    std::fs::write(&path, w).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Dynare's `pltorg` including the layout of the last figure (`lr`, `lc`):
/// (nbplt, nr, nc, lr, lc, nstar).
pub fn pltorg(number: usize) -> (usize, usize, usize, usize, usize, usize) {
    let small = |k: usize| match k {
        1 => (1, 1),
        2 => (2, 1),
        3 => (3, 1),
        4 => (2, 2),
        5 | 6 => (3, 2),
        _ => (3, 3),
    };
    match number {
        0 => (0, 0, 0, 0, 0, 9),
        1..=9 => {
            let (r, c) = small(number);
            (1, r, c, 0, 0, 9)
        }
        _ => {
            let nbplt = number.div_ceil(9);
            let reste = number - (nbplt - 1) * 9;
            let (lr, lc) = small(reste);
            (nbplt, 3, 3, lr, lc, 9)
        }
    }
}

/// The figures `stoch_simul.m` draws for one shock: variables whose IRF reaches
/// `plot_threshold` (in absolute value), `pltorg` grids, black response and red zero line.
/// Names: `IRF_<shock>` for a single figure, `IRF_<shock><k>` otherwise.
pub fn irf_figures(
    shock_name: &str,
    var_names: &[String],
    y: &Mat,
    relative: bool,
    plot_threshold: f64,
) -> Vec<Figure> {
    let keep: Vec<usize> =
        (0..y.nrows()).filter(|&i| y.row(i).iter().fold(0.0f64, |m, v| m.max(v.abs())) >= plot_threshold).collect();
    let (nbplt, nr, nc, lr, lc, nstar) = pltorg(keep.len());
    let periods = y.ncols();
    let x: Vec<f64> = (1..=periods).map(|t| t as f64).collect();
    let panel = |i: usize| Panel {
        title: var_names[i].clone(),
        series: vec![
            Series::Line {
                x: x.clone(),
                y: y.row(i).iter().copied().collect(),
                color: Color::Black,
                style: LineStyle::Solid,
                width: 1.0,
                label: None,
            },
            Series::HLine { y: 0.0, color: Color::Red, style: LineStyle::Solid },
        ],
        ..Default::default()
    };
    let what = if relative { "Relative response to orthogonalized shock to" } else { "Orthogonalized shock to" };
    let mut figs = Vec::new();
    for f in 0..nbplt {
        let idx = &keep[f * nstar..(f * nstar + nstar).min(keep.len())];
        let last = f + 1 == nbplt && nbplt > 1;
        let (rows, cols) = if last && lr > 0 { (lr, lc) } else { (nr, nc) };
        let (name, title) = if nbplt == 1 {
            (format!("IRF_{shock_name}"), format!("{what} {shock_name}"))
        } else {
            (format!("IRF_{shock_name}{}", f + 1), format!("{what} {shock_name} figure {}", f + 1))
        };
        figs.push(Figure { name, title, rows, cols, panels: idx.iter().map(|&i| panel(i)).collect() });
    }
    figs
}
