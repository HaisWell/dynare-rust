// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `+mom/matched_irfs_blocks.m`: the empirical IRFs to match, their weighting
//! matrix and their positions in the `irf x obs_nbr x exo_nbr` array of model IRFs.

use anyhow::{Result, bail};

use super::{IrfWeight, MatchedIrf};
use crate::linalg::Mat;
use crate::model::Model;

/// What `matched_irfs_blocks` returns.
#[derive(Clone, Debug)]
pub struct IrfTargets {
    /// `data_irfs(irf_index)`.
    pub data: Vec<f64>,
    /// `weight_mat(irf_index, irf_index)`.
    pub w: Mat,
    /// 0-based linear (column-major) indices into `horizon x obs_nbr x exo_nbr`, ascending.
    pub index: Vec<usize>,
    /// `max_irf_horizon` (becomes `options_mom_.irf`).
    pub horizon: usize,
    /// Labels `var shock (h)` in block order (`display_comparison_moments_irfs.m`).
    pub labels: Vec<String>,
}

/// `varobs_id`: endogenous indices of the observed variables (varobs order).
pub fn matched_irfs_blocks(
    irfs: &[MatchedIrf],
    weights: &[IrfWeight],
    varobs_id: &[usize],
    model: &Model,
) -> Result<IrfTargets> {
    let obs_nbr = varobs_id.len();
    let exo_nbr = model.n_exo();
    let horizon = irfs.iter().flat_map(|e| e.periods.iter().flatten().copied()).max().unwrap_or(0);
    if horizon == 0 {
        bail!("method_of_moments: You need to provide a 'matched_irfs' block for 'mom_method=IRF_MATCHING'!");
    }
    let n = horizon * obs_nbr * exo_nbr;
    let mut data = vec![f64::NAN; n];
    let mut w = std::collections::BTreeMap::<(usize, usize), f64>::new();
    let lin = |h: usize, o: usize, e: usize| (h - 1) + horizon * (o + obs_nbr * e);
    // Dynare's preprocessor stores M_.matched_irfs sorted by (variable, shock) name; the labels
    // of display_comparison_moments_irfs.m follow that order (the values follow `index`)
    let mut sorted: Vec<&MatchedIrf> = irfs.iter().collect();
    sorted.sort_by_key(|e| (e.endo.map(|v| model.endo_names[v].clone()), e.exo.map(|x| model.exo_names[x].clone())));
    let mut labels = Vec::new();
    for e in &sorted {
        if let (Some(v), Some(x)) = (e.endo, e.exo) {
            for per in &e.periods {
                for h in per {
                    labels.push(format!("{} {} ({h})", model.endo_names[v], model.exo_names[x]));
                }
            }
        }
    }
    for e in irfs {
        let (Some(v), Some(x)) = (e.endo, e.exo) else {
            bail!("method_of_moments: matched_irfs: each entry needs var and varexo")
        };
        let Some(o) = varobs_id.iter().position(|&i| i == v) else {
            bail!(
                "method_of_moments: You specified an IRF matching involving variable {}, but it is not declared as a varobs!",
                model.endo_names[v]
            )
        };
        let (vn, xn) = (&model.endo_names[v], &model.exo_names[x]);
        for (k, per) in e.periods.iter().enumerate() {
            let mut sorted = per.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() < per.len() {
                bail!(
                    "method_of_moments: You specified an IRF matching involving variable {vn} and shock {xn}, but there were duplicate 'periods' in the specification!"
                );
            }
            let vals = e.values.get(k).or(if e.values.len() == 1 { e.values.first() } else { None });
            let Some(vals) = vals else {
                bail!(
                    "method_of_moments: You specified an IRF matching involving variable {vn} and shock {xn}, but the length of 'periods' does not match the length of 'values'!"
                )
            };
            let vals: Vec<f64> = if vals.len() == 1 { vec![vals[0]; per.len()] } else { vals.clone() };
            if vals.len() != per.len() {
                bail!(
                    "method_of_moments: You specified an IRF matching involving variable {vn} and shock {xn}, but the length of 'periods' does not match the length of 'values'!"
                );
            }
            let wts: Vec<f64> = match e.weights.get(k).or(if e.weights.len() == 1 { e.weights.first() } else { None }) {
                None => vec![1.0; per.len()],
                Some(w) if w.len() == 1 => vec![w[0]; per.len()],
                Some(w) => w.clone(),
            };
            if wts.len() != per.len() {
                bail!(
                    "method_of_moments: You specified an IRF matching involving variable {vn} and shock {xn}, but the length of 'periods' does not match the length of 'weights'!"
                );
            }
            for (j, &h) in per.iter().enumerate() {
                let i = lin(h, o, x);
                data[i] = vals[j];
                if wts[j] != 1.0 {
                    w.insert((i, i), wts[j]);
                }
            }
        }
    }
    for e in weights {
        let o1 = varobs_id.iter().position(|&i| i == e.endo1);
        let o2 = varobs_id.iter().position(|&i| i == e.endo2);
        let (Some(o1), Some(o2)) = (o1, o2) else {
            bail!(
                "method_of_moments: You specified a weight for an IRF matching involving a variable that is not a varobs!"
            )
        };
        if e.periods1.len() != e.periods2.len() {
            bail!(
                "method_of_moments: You specified a 'matched_irfs_weights' entry whose horizons do not have the same length!"
            );
        }
        if e.periods1.iter().chain(&e.periods2).any(|&h| h > horizon) {
            bail!(
                "method_of_moments: You specified a 'matched_irfs_weights' entry whose horizon is larger than the maximum one declared in the 'matched_irfs' block!"
            );
        }
        let vals: Vec<f64> = if e.weights.len() == 1 && e.periods1.len() > 1 {
            vec![e.weights[0]; e.periods1.len()]
        } else {
            e.weights.clone()
        };
        if vals.len() != e.periods1.len() {
            bail!(
                "method_of_moments: You specified a 'matched_irfs_weights' entry whose horizons do not match the length of 'weights'!"
            );
        }
        for k in 0..vals.len() {
            let i1 = lin(e.periods1[k], o1, e.exo1);
            let i2 = lin(e.periods2[k], o2, e.exo2);
            w.insert((i1, i2), vals[k]);
            w.insert((i2, i1), vals[k]);
        }
    }
    let index: Vec<usize> = (0..n).filter(|&i| !data[i].is_nan()).collect();
    let pos = |i: usize| index.binary_search(&i).ok();
    let mut wm = Mat::identity(index.len(), index.len());
    for (&(a, b), &v) in &w {
        if let (Some(i), Some(j)) = (pos(a), pos(b)) {
            wm[(i, j)] = v;
        }
    }
    // labels follow the declaration order of the block (display_comparison_moments_irfs.m),
    // the data follow the linear index order
    Ok(IrfTargets { data: index.iter().map(|&i| data[i]).collect(), w: wm, index, horizon, labels })
}
