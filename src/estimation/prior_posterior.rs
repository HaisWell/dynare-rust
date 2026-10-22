// SPDX-License-Identifier: GPL-3.0-or-later
//! Posterior distributions of smoothed objects and forecasts: port of
//! `prior_posterior_statistics.m` / `prior_posterior_statistics_core.m` (type 'posterior',
//! order 1, no measurement error, no OccBin) and the statistics part of `pm3.m`.
//!
//! For each of `sub_draws` parameter vectors taken from the MCMC chains after `mh_drop`
//! (`get_posterior_subsample.m`), the smoother is rerun (`DsgeSmoother`), and the smoothed
//! variables, smoothed shocks, updated variables, one-step-ahead filtered variables and the
//! forecasts without (`MeanForecast`) and with simulated shocks (`PointForecast`) are stored;
//! `pm3` then reports, per variable and period, the mean, median, variance, HPD interval at
//! `mh_conf_sig` and deciles (`posterior_moments.m`). Draws run in parallel (rayon); each
//! draw's shocks come from its own ChaCha stream, so results do not depend on scheduling.

use std::path::Path;

use anyhow::{Context, Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::mcmc::Chain;
use super::posterior::posterior_moments;
use crate::dsge::{DsgeModel, LikelihoodData};
use crate::forecast::{ShockLoading, dsge_smoother, simulate_posterior_forecasts};
use crate::graphics::{Color, Figure, LineStyle, Panel, Series};
use crate::linalg::Mat;

/// pm3's summary of one object: rows = periods (n2), columns = variables.
#[derive(Clone, Debug)]
pub struct PosteriorSummary {
    /// Dynare's `name3` (`SmoothedVariables`, `MeanForecast`, ...).
    pub name: String,
    /// pm3's `tit1` (console/figure title).
    pub title: String,
    pub names: Vec<String>,
    pub mean: Mat,
    pub median: Mat,
    pub var: Mat,
    pub hpd_inf: Mat,
    pub hpd_sup: Mat,
    /// deciles[k] = (k+1)*10 % quantile (`post_deciles` / `oo_.X.deciles`), each n2 x nvar.
    pub deciles: Vec<Mat>,
    /// ---- estimation completion: pm3's `Density(:,:,:,i)` per variable (512 x 2*n2, see
    /// [`crate::estimation::est_completion::pm3_density`]); empty with `no_posterior_kernel_density`.
    pub density: Vec<Mat>,
}

/// Port of pm3.m's three-dimensional branch: `stock[b]` is draw b's n2 x nvar matrix.
pub fn summarize(name: &str, title: &str, names: &[String], stock: &[Mat], mh_conf_sig: f64) -> PosteriorSummary {
    let (n2, nvar) = stock.first().map_or((0, names.len()), |m| (m.nrows(), m.ncols()));
    let cells: Vec<_> = (0..n2 * nvar)
        .into_par_iter()
        .map(|c| {
            let (j, i) = (c % n2, c / n2);
            let mut xx: Vec<f64> = stock.iter().map(|m| m[(j, i)]).collect();
            posterior_moments(&mut xx, mh_conf_sig)
        })
        .collect();
    let get = |f: &dyn Fn(&super::ParamMoments) -> f64| Mat::from_fn(n2, nvar, |j, i| f(&cells[i * n2 + j]));
    PosteriorSummary {
        name: name.into(),
        title: title.into(),
        names: names.to_vec(),
        mean: get(&|p| p.mean),
        median: get(&|p| p.median),
        var: get(&|p| p.var),
        hpd_inf: get(&|p| p.hpd.0),
        hpd_sup: get(&|p| p.hpd.1),
        deciles: (0..9).map(|k| get(&|p| p.deciles[k])).collect(),
        density: Vec::new(),
    }
}

/// ---- estimation completion: [`summarize`] plus pm3's kernel density estimates
/// (`options_.estimation.moments_posterior_density.indicator`).
pub fn summarize_with_density(
    name: &str,
    title: &str,
    names: &[String],
    stock: &[Mat],
    mh_conf_sig: f64,
    density: bool,
) -> PosteriorSummary {
    let mut s = summarize(name, title, names, stock, mh_conf_sig);
    if density {
        s.density = (0..names.len())
            .into_par_iter()
            .map(|i| crate::estimation::est_completion::pm3_density(stock, i))
            .collect();
    }
    s
}

/// What prior_posterior_statistics computes (`options_.smoother`, `filtered_vars`, `forecast`).
#[derive(Clone, Debug)]
pub struct PosteriorSmootherOptions {
    pub smoother: bool,
    pub filtered_vars: bool,
    /// `options_.forecast` (0 = none).
    pub horizon: usize,
    pub mh_conf_sig: f64,
    /// Declaration-order endogenous variables reported (and their names).
    pub vars: Vec<usize>,
    pub var_names: Vec<String>,
    pub exo_names: Vec<String>,
    /// Seed of the forecast shocks (draw b uses stream b of this seed).
    pub seed: u64,
    /// ---- estimation completion: pm3's densities (not `no_posterior_kernel_density`).
    pub kernel_density: bool,
}

#[derive(Clone, Debug, Default)]
pub struct PosteriorObjects {
    /// Number of draws used (failed draws are dropped, as `is_successful_draw` does).
    pub draws: usize,
    pub failed: usize,
    pub smoothed_variables: Option<PosteriorSummary>,
    pub smoothed_shocks: Option<PosteriorSummary>,
    pub updated_variables: Option<PosteriorSummary>,
    pub filtered_variables: Option<PosteriorSummary>,
    pub mean_forecast: Option<PosteriorSummary>,
    pub point_forecast: Option<PosteriorSummary>,
}

impl PosteriorObjects {
    pub fn all(&self) -> Vec<&PosteriorSummary> {
        [
            &self.smoothed_variables,
            &self.smoothed_shocks,
            &self.updated_variables,
            &self.filtered_variables,
            &self.mean_forecast,
            &self.point_forecast,
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

/// `set_number_of_subdraws.m`: `sub_draws` or min(posterior_max_subsample_draws = 1200,
/// number of kept draws over all chains).
pub fn number_of_subdraws(sub_draws: Option<usize>, kept_per_chain: usize, nblck: usize) -> Result<usize> {
    let total = kept_per_chain * nblck;
    match sub_draws {
        None => Ok(crate::estimation::mh_session::posterior_max_subsample_draws().min(total)),
        Some(b) if b > total => bail!(
            "The value of option sub_draws ({b}) is greater than the number of available draws in the MCMC ({total})!"
        ),
        Some(b) => Ok(b),
    }
}

/// `get_posterior_subsample.m` (draws `first..first+nkeep` of each chain are kept): for each
/// of the `b` subdraws a chain and a kept draw are picked uniformly at random (with
/// replacement); when `b` equals the number of kept draws all of them are used
/// (`GetAllPosteriorDraws`, chain by chain).
pub fn posterior_subsample(chains: &[Chain], first: usize, nkeep: usize, b: usize, seed: u64) -> Vec<Vec<f64>> {
    if b == nkeep * chains.len() {
        return chains.iter().flat_map(|c| (first..first + nkeep).map(move |t| c.draw(t).to_vec())).collect();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(u64::MAX - 7);
    let mut addr: Vec<(usize, usize)> =
        (0..b).map(|_| (rng.gen_range(0..chains.len()), first + rng.gen_range(0..nkeep))).collect();
    // sortrows(SampleAddress, [3 2]): grouped by chain (file order irrelevant here)
    addr.sort_by_key(|&(c, _)| c);
    addr.into_iter().map(|(c, t)| chains[c].draw(t).to_vec()).collect()
}

struct DrawResult {
    smooth: Mat,
    update: Mat,
    filter: Mat,
    innov: Mat,
    fc_mean: Mat,
    fc_point: Mat,
}

/// The loop of `prior_posterior_statistics_core.m` over `draws`, then `pm3` per object.
pub fn prior_posterior_statistics(
    dsge: &DsgeModel,
    data: &LikelihoodData,
    draws: &[Vec<f64>],
    o: &PosteriorSmootherOptions,
) -> Result<PosteriorObjects> {
    let vars = &o.vars;
    let results: Vec<Option<DrawResult>> = draws
        .par_iter()
        .enumerate()
        .map(|(b, x)| {
            let sm = dsge_smoother(dsge, data, x, false).ok()?;
            let dr = sm.dr();
            let ys = sm.ys();
            let y0 = sm.last_state();
            let pick = |m: &Mat| Mat::from_fn(m.nrows(), vars.len(), |h, j| m[(h, vars[j])] + ys[vars[j]]);
            let (fc_mean, fc_point) = if o.horizon > 0 {
                let mean = simulate_posterior_forecasts::<ChaCha8Rng>(dr, &y0, o.horizon, None);
                let mut rng = ChaCha8Rng::seed_from_u64(o.seed);
                rng.set_stream(b as u64 + 1);
                let load = ShockLoading::new(dr, &sm.resolved.q).ok()?;
                let point = simulate_posterior_forecasts(dr, &y0, o.horizon, Some((&load, &mut rng)));
                (pick(&mean), pick(&point))
            } else {
                (Mat::zeros(0, 0), Mat::zeros(0, 0))
            };
            Some(DrawResult {
                smooth: sm.levels(&sm.out.alphahat, vars),
                update: sm.levels(&sm.out.ahat, vars),
                filter: sm.levels(&sm.out.filtered, vars),
                innov: sm.out.etahat.transpose(),
                fc_mean,
                fc_point,
            })
        })
        .collect();
    let failed = results.iter().filter(|r| r.is_none()).count();
    let ok: Vec<DrawResult> = results.into_iter().flatten().collect();
    if ok.is_empty() {
        bail!("prior_posterior_statistics: the smoother failed for every draw");
    }
    if failed > 0 {
        println!(
            "prior_posterior_statistics: {failed} of the {} draws failed (no solution) and were dropped.",
            draws.len()
        );
    }
    let c = o.mh_conf_sig;
    let collect = |f: &dyn Fn(&DrawResult) -> Mat| ok.iter().map(f).collect::<Vec<Mat>>();
    let run = |name: &str, title: &str, names: &[String], f: &dyn Fn(&DrawResult) -> Mat| {
        println!("Estimation::mcmc: {title}");
        let s = summarize_with_density(name, title, names, &collect(f), c, o.kernel_density); // ---- estimation completion
        println!("Estimation::mcmc: {title}, done!");
        Some(s)
    };
    let mut out = PosteriorObjects { draws: ok.len(), failed, ..Default::default() };
    if o.smoother {
        out.smoothed_variables = run("SmoothedVariables", "Smoothed variables", &o.var_names, &|d| d.smooth.clone());
        out.smoothed_shocks = run("SmoothedShocks", "Smoothed shocks", &o.exo_names, &|d| d.innov.clone());
        out.updated_variables = run("UpdatedVariables", "Updated Variables", &o.var_names, &|d| d.update.clone());
    }
    if o.filtered_vars {
        out.filtered_variables =
            run("FilteredVariables", "One step ahead forecast (filtered variables)", &o.var_names, &|d| {
                d.filter.clone()
            });
    }
    if o.horizon > 0 {
        out.mean_forecast = run("MeanForecast", "Forecasted variables (mean)", &o.var_names, &|d| d.fc_mean.clone());
        out.point_forecast =
            run("PointForecast", "Forecasted variables (point)", &o.var_names, &|d| d.fc_point.clone());
    }
    Ok(out)
}

/// `<fname>_posterior_<object>.csv`, long format:
/// `date,variable,mean,median,sd,hpd_lower,hpd_upper,d10..d90`.
pub fn write_summary_csv(path: &Path, s: &PosteriorSummary, dates: &[String]) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut w = String::from("date,variable,mean,median,sd,hpd_lower,hpd_upper,d10,d20,d30,d40,d50,d60,d70,d80,d90\n");
    for (i, name) in s.names.iter().enumerate() {
        for t in 0..s.mean.nrows() {
            w += &format!(
                "{},{},{},{},{},{},{}",
                dates.get(t).cloned().unwrap_or_else(|| (t + 1).to_string()),
                name,
                s.mean[(t, i)],
                s.median[(t, i)],
                s.var[(t, i)].sqrt(),
                s.hpd_inf[(t, i)],
                s.hpd_sup[(t, i)]
            );
            for d in &s.deciles {
                w += &format!(",{}", d[(t, i)]);
            }
            w.push('\n');
        }
    }
    std::fs::write(path, w).with_context(|| format!("writing {}", path.display()))
}

/// pm3_core.m's figures: 3x3 panels, the deciles as thin green lines and the mean in black;
/// file `<fname>_<name3>_<last variable of the figure>`. `select`: the variables to plot
/// (Dynare plots `options_.varlist`), all if empty. Panels whose mean is all NaN are skipped.
pub fn pm3_figures(s: &PosteriorSummary, select: &[String]) -> Vec<Figure> {
    let idx: Vec<usize> = if select.is_empty() {
        (0..s.names.len()).collect()
    } else {
        select.iter().filter_map(|v| s.names.iter().position(|n| n == v)).collect()
    };
    let idx: Vec<usize> = idx.into_iter().filter(|&i| s.mean.column(i).iter().any(|v| !v.is_nan())).collect();
    let n2 = s.mean.nrows();
    let x: Vec<f64> = (1..=n2).map(|t| t as f64).collect();
    let mut figs = Vec::new();
    for (k, chunk) in idx.chunks(9).enumerate() {
        let panels = chunk
            .iter()
            .map(|&i| {
                let mut series: Vec<Series> = s
                    .deciles
                    .iter()
                    .map(|d| Series::Line {
                        x: x.clone(),
                        y: d.column(i).iter().copied().collect(),
                        color: Color::Green,
                        style: LineStyle::Solid,
                        width: 0.5,
                        label: None,
                    })
                    .collect();
                series.push(Series::Line {
                    x: x.clone(),
                    y: s.mean.column(i).iter().copied().collect(),
                    color: Color::Black,
                    style: LineStyle::Solid,
                    width: 1.0,
                    label: None,
                });
                Panel { title: s.names[i].clone(), series, ..Default::default() }
            })
            .collect();
        figs.push(Figure {
            name: format!("{}_{}", s.name, s.names[*chunk.last().unwrap()]),
            title: format!("{} {}", s.title, k + 1),
            rows: 3,
            cols: 3,
            panels,
        });
    }
    figs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_matches_posterior_moments() {
        let stock: Vec<Mat> = (0..50).map(|b| Mat::from_fn(2, 1, |t, _| (b as f64) * (t as f64 + 1.0))).collect();
        let s = summarize("X", "x", &["a".into()], &stock, 0.9);
        assert!((s.mean[(0, 0)] - 24.5).abs() < 1e-12);
        assert!((s.mean[(1, 0)] - 49.0).abs() < 1e-12);
        assert_eq!(s.deciles.len(), 9);
        assert!((s.deciles[4][(0, 0)] - 24.0).abs() < 1e-12);
    }

    #[test]
    fn subdraws() {
        assert_eq!(number_of_subdraws(None, 500, 2).unwrap(), 1000);
        assert_eq!(number_of_subdraws(None, 5000, 2).unwrap(), 1200);
        assert!(number_of_subdraws(Some(2000), 500, 2).is_err());
    }
}
