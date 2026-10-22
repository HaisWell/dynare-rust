// SPDX-License-Identifier: GPL-3.0-or-later
//! Reduced-form mapping: which parameters drive the entries of the decision rule
//! `[ghx ghu]` (`+gsa/reduced_form_mapping.m` with `threshold_redform`,
//! `+gsa/reduced_form_screening.m` + `+gsa/Morris_Measure_Groups.m` for Morris), and a
//! regression substitute for the SS-ANOVA mapping (Dynare needs the external SS-ANOVA-R
//! toolbox for it).

use anyhow::Result;

use super::mcf::{McfSpec, analysis, cdf_series, univariate};
use super::sampling::priorcdf;
use super::stability::StabMap;
use super::stats::{dyntable, fmt_f, mean, num2str, var};
use super::{Ctx, GsaOptions};
use crate::dsge::DsgeModel;
use crate::graphics::{Color, Figure, LineStyle, Panel, Series, pltorg};
use crate::solver::DrStructure;

/// One entry of the reduced form: `(label for messages, column of [ghx ghu], kind)`.
enum Entry {
    Shock(usize),
    Lag(usize),
}

struct Target {
    namendo: String,
    /// row of [ghx ghu] (DR position)
    row: usize,
}

fn entries(dsge: &DsgeModel, o: &GsaOptions) -> (DrStructure, Vec<Target>) {
    let s = DrStructure::new(&dsge.model.lead_lag_incidence, dsge.model.n_exo());
    let targets = o
        .namendo
        .iter()
        .filter_map(|n| {
            let i = dsge.model.endo_index(n)?;
            Some(Target { namendo: n.clone(), row: s.inv_order_var[i] })
        })
        .collect();
    (s, targets)
}

/// Column of `[ghx ghu]` for `namexo` / lagged `namlagendo` (`None` if not a shock / state).
fn column_of(dsge: &DsgeModel, s: &DrStructure, e: &Entry) -> Option<usize> {
    match *e {
        Entry::Shock(k) => Some(s.nspred + k),
        Entry::Lag(i) => {
            let pos = s.inv_order_var[i];
            (pos >= s.nstatic && pos < s.nstatic + s.nspred)
                .then(|| pos - s.nstatic)
                .filter(|_| dsge.model.n_endo() > i)
        }
    }
}

/// MATLAB `num2str` of a row vector of non-integers (`'%{w}.{p}g'` columns, leading blanks
/// trimmed), as printed in the threshold messages.
fn num2str_vec(v: &[f64]) -> String {
    if v.iter().all(|x| x.fract() == 0.0) {
        let w = v.iter().map(|x| format!("{}", *x as i64).len()).max().unwrap_or(1) + 2;
        return v.iter().map(|x| format!("{:>w$}", *x as i64)).collect::<String>().trim_start().to_string();
    }
    let mx = v.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let ndgt = ((mx.log10().floor() as i32) + 5).max(5) as usize;
    let w = ndgt + 7;
    v.iter().map(|&x| format!("{:>w$}", super::stats::fmt_g(x, ndgt))).collect::<String>().trim_start().to_string()
}

/// `reduced_form_mapping.m`.
/// Returns the first-order indices of every entry (`label`, `si`) of the regression mapping.
pub fn mapping(ctx: &Ctx, dsge: &DsgeModel, o: &GsaOptions, stab: &StabMap) -> Result<Vec<(String, Vec<f64>)>> {
    let (s, targets) = entries(dsge, o);
    let nshock = ctx.nshock;
    let pnames: Vec<String> = ctx.param_names[nshock..].to_vec();
    let typ = if o.pprior { "prior" } else { "mc" };
    let adir = ctx.out_dir.join(format!("redform_{typ}"));
    let x0: Vec<Vec<f64>> = stab.istable.iter().map(|&k| stab.lpmat[k].clone()).collect();
    let np = pnames.len();
    let mut si_all: Vec<(String, Vec<f64>)> = Vec::new();
    for t in &targets {
        let mut bars: Vec<Panel> = Vec::new();
        let mut lag_bars: Vec<Panel> = Vec::new();
        let mut list: Vec<(String, Option<Entry>, bool)> = Vec::new();
        for nx in &o.namexo {
            list.push((nx.clone(), dsge.model.exo_names.iter().position(|e| e == nx).map(Entry::Shock), false));
        }
        for nl in &o.namlagendo {
            list.push((nl.clone(), dsge.model.endo_index(nl).map(Entry::Lag), true));
        }
        for (name, entry, lag) in list {
            println!("\n[{} vs {}{}]", t.namendo, if lag { "lagged " } else { "" }, name);
            let Some(col) = entry.as_ref().and_then(|e| column_of(dsge, &s, e)) else { continue };
            let y0: Vec<f64> = stab.t.iter().map(|m| m[(t.row, col)]).collect();
            let (mn, mx) = y0.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
            if !(mx - mn > 1e-10) {
                println!(
                    "This entry in the {} matrix is CONSTANT = {}",
                    if lag { "transition" } else { "shock" },
                    num2str_sig(mean(&y0), 3)
                );
                continue;
            }
            let label = format!("{} vs {}", t.namendo, name);
            if let Some((lo, hi)) = o.threshold_redform {
                let iy: Vec<usize> = (0..y0.len()).filter(|&k| y0[k] > lo && y0[k] < hi).collect();
                let iyc: Vec<usize> = (0..y0.len()).filter(|&k| y0[k] <= lo || y0[k] >= hi).collect();
                let xdir = adir.join(format!("{}_vs_{}_threshold", t.namendo, name));
                let sub = ctx.with_dir(&xdir);
                if ctx.graphs_enabled() {
                    let mut p = Panel {
                        title: format!(
                            "{}{} - threshold [{} {}]",
                            label.replace(" vs ", if lag { " vs lagged " } else { " vs " }),
                            "",
                            num2str(lo),
                            num2str(hi)
                        ),
                        ..Default::default()
                    };
                    p.series.push(Series::Band {
                        x: vec![lo.max(mn), hi.min(mx)],
                        lower: vec![0.0, 0.0],
                        upper: vec![1.0, 1.0],
                        color: Color::Rgb(178, 204, 255),
                        alpha: 1.0,
                        label: None,
                    });
                    p.series.push(cdf_series(&y0, Color::Black, LineStyle::Solid, 2.0, None));
                    sub.save(&Figure {
                        name: format!("{typ}_{}_vs_{}", t.namendo, name),
                        title: format!("Reduced Form Mapping (Monte Carlo Filtering): {label}"),
                        rows: 1,
                        cols: 1,
                        panels: vec![p],
                    });
                }
                let atitle0 = format!("Reduced Form Mapping (Monte Carlo Filtering) for {label}");
                let spec = McfSpec {
                    pvalue_ks: o.ksstat_redform,
                    pvalue_corr: o.alpha2_redform,
                    alpha2: o.alpha2_redform,
                    param_names: &pnames,
                    amcf_name: format!("{typ}_{}_vs_{}_threshold", t.namendo, name),
                    amcf_title: format!(
                        "{typ} Reduced Form Mapping (Monte Carlo Filtering): Parameter(s) driving {label}"
                    ),
                    beha_title: "inside threshold".into(),
                    nobeha_title: "outside threshold".into(),
                    title: atitle0.clone(),
                    xparam1: None,
                    noprint: false,
                };
                let mut icheck_empty = true;
                if !iy.is_empty() && !iyc.is_empty() {
                    println!(
                        "{}% of the {typ} support matches {atitle0}",
                        fmt_f(iy.len() as f64 / y0.len() as f64 * 100.0, 4, 1)
                    );
                    let r = analysis(&sub, &x0, &iy, &iyc, &spec);
                    icheck_empty = r.indmcf.is_empty() || iy.len() <= 10 || iyc.len() <= 10;
                }
                if icheck_empty {
                    let th = num2str_vec(&[lo, hi]);
                    if iy.len() <= 10 {
                        if iy.is_empty() {
                            println!("There are NO MC samples in the desired range [{th}]!");
                        } else {
                            println!("There are TOO FEW (<=10) MC samples  in the desired range [{th}]!");
                        }
                    } else if iyc.len() <= 10 {
                        if iyc.is_empty() {
                            println!("ALL MC samples are in the desired range [{th}]!");
                        } else {
                            println!("Almost ALL MC samples are in the desired range [{th}]!");
                            println!("There are TOO FEW (<=10) MC samples OUTSIDE the desired range!");
                        }
                    }
                    if ctx.graphs_enabled() {
                        let spec2 = McfSpec { title: format!("Monte Carlo Filtering for {label}"), ..spec };
                        sub.save(&redform_mcf(&y0, &x0, &spec2));
                    }
                }
            } else {
                let aname = format!("{typ}_{}_vs_{}{name}", t.namendo, if lag { "lag_" } else { "" });
                let xdir0 = adir.join(format!("{}_vs_{name}", t.namendo));
                let si = regression_si(&y0, &x0, dsge, ctx, o, &xdir0, &aname)?;
                let mut iso: Vec<usize> = (0..np).collect();
                iso.sort_by(|&a, &b| si[b].total_cmp(&si[a]));
                let top: Vec<usize> = iso.into_iter().take(np.min(10)).collect();
                let panel = Panel {
                    title: format!(
                        "{}{}{}",
                        if o.logtrans_redform { "log " } else { " " },
                        label,
                        if lag { "(-1)" } else { "" }
                    ),
                    series: vec![Series::StackedBars {
                        x: (1..=top.len()).map(|k| k as f64).collect(),
                        components: vec![top.iter().map(|&k| si[k]).collect()],
                        labels: vec!["si".into()],
                    }],
                    x_ticks: Some(top.iter().enumerate().map(|(k, &j)| ((k + 1) as f64, pnames[j].clone())).collect()),
                    ..Default::default()
                };
                if lag {
                    lag_bars.push(panel)
                } else {
                    bars.push(panel)
                }
                println!(
                    "{}",
                    dyntable(
                        "",
                        &["Parameter", "S_i"],
                        &top.iter().map(|&j| pnames[j].clone()).collect::<Vec<_>>(),
                        &top.iter().map(|&j| vec![si[j]]).collect::<Vec<_>>(),
                        0,
                        10,
                        3
                    )
                    .trim_end()
                );
                si_all.push((label, si));
            }
        }
        if o.threshold_redform.is_none() && ctx.graphs_enabled() {
            let logflag = if o.logtrans_redform { "log" } else { "" };
            for (panels, what) in [(bars, "shocks"), (lag_bars, "lags")] {
                for (ifig, chunk) in panels.chunks(9).enumerate() {
                    let (_, nr, nc, _) = pltorg(9);
                    ctx.save(&Figure {
                        name: format!("redform_{}_vs_{what}_{logflag}{}", t.namendo, ifig + 1),
                        title: format!("Reduced Form Mapping: {} vs {what} {}", t.namendo, ifig + 1),
                        rows: nr,
                        cols: nc,
                        panels: chunk.to_vec(),
                    });
                }
            }
        }
    }
    if o.threshold_redform.is_none() && !si_all.is_empty() {
        // `Reduced form GSA` summary: distribution over entries of each parameter's index
        let means: Vec<Vec<f64>> =
            (0..np).map(|j| vec![mean(&si_all.iter().map(|(_, s)| s[j]).collect::<Vec<_>>())]).collect();
        print!(
            "{}",
            dyntable(
                "Reduced form GSA (regression substitute for SS-ANOVA): mean first-order index over the entries",
                &["Parameter", "mean S_i"],
                &pnames,
                &means,
                0,
                10,
                3
            )
        );
        if ctx.graphs_enabled() {
            let p = Panel {
                title: if o.logtrans_redform {
                    "Reduced form GSA - Log-transformed elements".into()
                } else {
                    "Reduced form GSA".into()
                },
                series: si_all
                    .iter()
                    .map(|(_, s)| Series::Points {
                        x: (1..=np).map(|k| k as f64).collect(),
                        y: s.clone(),
                        color: Color::Blue,
                        label: None,
                    })
                    .collect(),
                x_ticks: Some(pnames.iter().enumerate().map(|(k, n)| ((k + 1) as f64, n.clone())).collect()),
                ..Default::default()
            };
            ctx.save(&Figure {
                name: if o.logtrans_redform { "redform_gsa_log".into() } else { "redform_gsa".into() },
                title: "Reduced Form GSA".into(),
                rows: 1,
                cols: 1,
                panels: vec![p],
            });
        }
    }
    Ok(si_all)
}

/// `num2str(x, n)` (n significant digits).
fn num2str_sig(x: f64, n: usize) -> String {
    super::stats::fmt_g(x, n)
}

/// `redform_mcf` of reduced_form_mapping.m: CDFs of the driving parameters in each decile of
/// the reduced-form coefficient.
fn redform_mcf(y0: &[f64], x0: &[Vec<f64>], spec: &McfSpec) -> Figure {
    let mut sorted = y0.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    // posterior_moments' deciles: x(round(n*[0.1:0.1:0.9]))
    let mut dec: Vec<f64> = vec![f64::NEG_INFINITY];
    for k in 1..=9 {
        let i = ((n as f64 * k as f64 / 10.0).round() as usize).clamp(1, n);
        dec.push(sorted[i - 1]);
    }
    dec.push(f64::INFINITY);
    let indy: Vec<Vec<usize>> =
        (0..10).map(|jt| (0..n).filter(|&k| y0[k] > dec[jt] && y0[k] <= dec[jt + 1]).collect()).collect();
    let (proba, _) = univariate(x0, &indy[0], &indy[9]);
    let mut indmcf: Vec<usize> = (0..proba.len()).filter(|&j| proba[j] < spec.pvalue_ks).collect();
    if indmcf.is_empty() {
        let mut all: Vec<usize> = (0..proba.len()).collect();
        all.sort_by(|&a, &b| proba[a].total_cmp(&proba[b]));
        indmcf = all.into_iter().take(1).collect();
    }
    indmcf.sort_by(|&a, &b| proba[a].total_cmp(&proba[b]));
    let nbr = indmcf.len();
    let nrow = ((nbr + 1) as f64).sqrt().ceil() as usize;
    let ncol = if nrow * (nrow - 1) > nbr { nrow - 1 } else { nrow };
    let panels = indmcf
        .iter()
        .map(|&j| Panel {
            title: spec.param_names[j].clone(),
            series: (0..10)
                .map(|jt| {
                    let x: Vec<f64> = indy[jt].iter().map(|&k| x0[k][j]).collect();
                    cdf_series(&x, Color::Palette(jt), LineStyle::Solid, 2.0, Some(format!("{}-dec", jt + 1)))
                })
                .collect(),
            legend: true,
            ..Default::default()
        })
        .collect();
    Figure { name: spec.amcf_name.clone(), title: spec.amcf_title.clone(), rows: nrow, cols: ncol.max(1), panels }
}

/// First-order sensitivity indices of `y` w.r.t. each parameter, by cubic regression of `y`
/// on the parameter's prior-CDF (or prior-range) transform: `var(E[y|x_j]) / var(y)`.
/// Substitute for Dynare's SS-ANOVA estimate (`ss_anova` of the SS-ANOVA-R toolbox), which
/// estimates the same first-order indices with smoothing splines.
fn regression_si(
    y0: &[f64],
    x0: &[Vec<f64>],
    dsge: &DsgeModel,
    ctx: &Ctx,
    o: &GsaOptions,
    xdir: &std::path::Path,
    aname: &str,
) -> Result<Vec<f64>> {
    // redform_private: `<xdir>/<fname>_<aname>[_log]_map.mat` holds `gsa_` (with the indices `si`)
    let mapfile = xdir.join(format!("{}_{}{}_map.mat", ctx.fname, aname, if o.logtrans_redform { "_log" } else { "" }));
    if o.load_redform {
        let saved = crate::matfile::read(&mapfile)
            .ok()
            .and_then(|m| m.get("gsa_").and_then(|g| g.field("si")).and_then(|v| v.as_vec()));
        match saved {
            Some(si) if si.len() == x0.first().map_or(0, Vec::len) => return Ok(si),
            _ => println!(
                "WARNING: load_redform: {} not found or incompatible; computing the mapping.",
                mapfile.display()
            ),
        }
    }
    let si = regression_si_compute(y0, x0, dsge, ctx, o)?;
    if std::fs::create_dir_all(xdir).is_ok() {
        use crate::matfile::MatValue;
        let g = MatValue::structure(vec![("si", MatValue::column(&si))]);
        let _ = crate::matfile::write_values(&mapfile, &[("gsa_", &g)]);
    }
    Ok(si)
}

fn regression_si_compute(y0: &[f64], x0: &[Vec<f64>], dsge: &DsgeModel, ctx: &Ctx, o: &GsaOptions) -> Result<Vec<f64>> {
    let nshock = ctx.nshock;
    let np = x0.first().map_or(0, Vec::len);
    let y: Vec<f64> = if o.logtrans_redform {
        let (yy, _, _) = log_transform(y0);
        yy
    } else {
        y0.to_vec()
    };
    let vy = var(&y);
    let mut si = vec![f64::NAN; np];
    if y.len() < 20 || !(vy > 0.0) {
        println!("WARNING: the number of samples is too small for the reduced-form mapping");
        return Ok(si);
    }
    for (j, s) in si.iter_mut().enumerate() {
        let u: Vec<f64> = x0
            .iter()
            .map(|r| {
                if o.prior_range {
                    (r[j] - ctx.bounds.lb[j + nshock]) / (ctx.bounds.ub[j + nshock] - ctx.bounds.lb[j + nshock])
                } else {
                    priorcdf(&dsge.priors[j + nshock], r[j])
                }
            })
            .collect();
        let fit = poly_fit(&u, &y, 3);
        *s = (var(&fit) / vy).clamp(0.0, 1.0);
    }
    Ok(si)
}

/// Least-squares polynomial fit of degree `deg` (Legendre-like centring for conditioning);
/// returns the fitted values.
fn poly_fit(u: &[f64], y: &[f64], deg: usize) -> Vec<f64> {
    use crate::linalg::Mat;
    let n = u.len();
    let k = deg + 1;
    let x = Mat::from_fn(n, k, |i, d| (2.0 * u[i] - 1.0).powi(d as i32));
    let xt = x.transpose();
    let xtx = &xt * &x;
    let xty = &xt * Mat::from_column_slice(n, 1, y);
    match xtx.clone().lu().solve(&xty) {
        Some(beta) => (0..n).map(|i| (0..k).map(|d| x[(i, d)] * beta[(d, 0)]).sum()).collect(),
        None => vec![mean(y); n],
    }
}

/// `+gsa/log_transform.m`: `log(y*isig + lam)` with `lam` chosen to remove the skewness.
/// Returns `(yy, isig, lam)`.
pub fn log_transform(y0: &[f64]) -> (Vec<f64>, f64, f64) {
    let skew = |y: &[f64]| {
        let m = mean(y);
        let m2 = y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / y.len() as f64;
        let m3 = y.iter().map(|v| (v - m).powi(3)).sum::<f64>() / y.len() as f64;
        m3 / m2.powf(1.5)
    };
    let f = |lam: f64, y: &[f64]| skew(&y.iter().map(|v| (v + lam).ln()).collect::<Vec<_>>());
    let mut y = y0.to_vec();
    let mut isig = 1.0;
    let (mn, mx) = y.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let median = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        let n = s.len();
        if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
    };
    let solve = |a: f64, b: f64, y: &[f64]| -> f64 {
        let (fa, fb) = (f(a, y), f(b, y));
        if fa.is_finite() && fb.is_finite() && fa * fb < 0.0 {
            let (mut a, mut b, mut fa) = (a, b, fa);
            for _ in 0..200 {
                let c = 0.5 * (a + b);
                let fc = f(c, y);
                if fc * fa > 0.0 {
                    a = c;
                    fa = fc;
                } else {
                    b = c;
                }
            }
            0.5 * (a + b)
        } else if fa.abs() < fb.abs() {
            a - 9.0 * f64::EPSILON
        } else {
            b
        }
    };
    if !(mx < 0.0 || mn > 0.0) {
        if skew(&y) < 0.0 {
            isig = -1.0;
            y.iter_mut().for_each(|v| *v = -*v);
        }
        // histcounts(y0, 10)
        let (mn, mx) = y.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let w = (mx - mn) / 10.0;
        let first = y.iter().filter(|&&v| v < mn + w).count();
        let last = y.iter().filter(|&&v| v >= mx - w).count();
        if first > 20 * last {
            let ymin = y.iter().copied().fold(f64::INFINITY, f64::min);
            let lam = solve(-ymin + 10.0 * f64::EPSILON, -ymin + median(&y).abs(), &y);
            let yy = y.iter().map(|v| (v + lam).ln()).collect();
            (yy, isig, lam)
        } else {
            (y.iter().map(|v| (v * v).ln()).collect(), 0.0, 0.0)
        }
    } else {
        if mx < 0.0 {
            isig = -1.0;
            y.iter_mut().for_each(|v| *v = -*v);
        }
        let ymin = y.iter().copied().fold(f64::INFINITY, f64::min);
        let lam = solve(-ymin + 10.0 * f64::EPSILON, -ymin + median(&y), &y).max(0.0);
        let yy = y.iter().map(|v| (v + lam).ln()).collect();
        (yy, isig, lam)
    }
}

/// Port of `+gsa/Morris_Measure_Groups.m` (no groups): for each output column, the absolute
/// mean `mu*`, mean `mu` and standard error of the elementary effects of each factor.
/// `sample`: trajectories of `nfact + 1` rows; `output[k]` NaN for unsolved points.
pub fn morris_measures(nfact: usize, sample: &[Vec<f64>], output: &[f64], p: usize) -> Vec<[f64; 3]> {
    let delt = p as f64 / (2.0 * p as f64 - 2.0);
    let sizea = nfact;
    let r = sample.len() / (sizea + 1);
    let mut ee = vec![vec![f64::NAN; r]; nfact];
    for i in 0..r {
        let base = i * (sizea + 1);
        for j in 0..sizea {
            let (a, b) = (&sample[base + j], &sample[base + j + 1]);
            let Some(cf) = (0..nfact).find(|&f| b[f] - a[f] != 0.0) else { continue };
            let delta = b[cf] - a[cf];
            let (o0, o1) = (output[base + j], output[base + j + 1]);
            ee[cf][i] = if delta > 0.0 { (o1 - o0) / delt } else { (o0 - o1) / delt };
        }
    }
    ee.iter()
        .map(|e| {
            let v: Vec<f64> = e.iter().copied().filter(|x| !x.is_nan()).collect();
            let rr = v.len() as f64;
            let absmu = v.iter().map(|x| x.abs()).sum::<f64>() / rr;
            let mu = v.iter().sum::<f64>() / rr;
            let sd = (v.iter().map(|x| (x - mu).powi(2) / (rr * (rr - 1.0))).sum::<f64>()).sqrt();
            [absmu, mu, sd]
        })
        .collect()
}

/// `reduced_form_screening.m`: Morris elementary effects of each requested reduced-form entry
/// (`SA = mu*/max(mu*)` over the structural parameters).
pub fn screening(ctx: &Ctx, dsge: &DsgeModel, o: &GsaOptions, stab: &StabMap) -> Result<()> {
    let (s, targets) = entries(dsge, o);
    let nshock = ctx.nshock;
    let pnames: Vec<String> = ctx.param_names[nshock..].to_vec();
    let np = pnames.len();
    let kn = stab.lpmat.len();
    let sample: Vec<Vec<f64>> = (0..kn).map(|k| stab.point(k)).collect();
    let mut sa_cols: Vec<(String, Vec<f64>)> = Vec::new();
    for t in &targets {
        let mut panels: Vec<Panel> = Vec::new();
        let mut lag_panels: Vec<Panel> = Vec::new();
        let mut list: Vec<(String, Option<Entry>, bool)> = Vec::new();
        for nx in &o.namexo {
            list.push((nx.clone(), dsge.model.exo_names.iter().position(|e| e == nx).map(Entry::Shock), false));
        }
        for nl in &o.namlagendo {
            list.push((nl.clone(), dsge.model.endo_index(nl).map(Entry::Lag), true));
        }
        for (name, entry, lag) in list {
            let Some(col) = entry.as_ref().and_then(|e| column_of(dsge, &s, e)) else { continue };
            // teff: NaN for the unstable points, skipped when the entry is constant
            let vals: Vec<f64> = stab.t.iter().map(|m| m[(t.row, col)]).collect();
            let (mn, mx) = vals.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
            if !(mx - mn > 1e-8) {
                continue;
            }
            let mut y0 = vec![f64::NAN; kn];
            for (&k, &v) in stab.istable.iter().zip(&vals) {
                y0[k] = v;
            }
            let meas = morris_measures(np + nshock, &sample, &y0, o.morris_nliv);
            let sam: Vec<f64> = meas[nshock..].iter().map(|m| m[0]).collect();
            let mx = sam.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sa: Vec<f64> = sam.iter().map(|v| v / (mx + f64::EPSILON)).collect();
            let mut iso: Vec<usize> = (0..np).collect();
            iso.sort_by(|&a, &b| sa[b].total_cmp(&sa[a]));
            let top: Vec<usize> = iso.into_iter().take(np.min(10)).collect();
            let panel = Panel {
                title: format!("{} vs. {}{}", t.namendo, name, if lag { "(-1)" } else { "" }),
                series: vec![Series::StackedBars {
                    x: (1..=top.len()).map(|k| k as f64).collect(),
                    components: vec![top.iter().map(|&k| sa[k]).collect()],
                    labels: vec!["EE".into()],
                }],
                x_ticks: Some(top.iter().enumerate().map(|(k, &j)| ((k + 1) as f64, pnames[j].clone())).collect()),
                ..Default::default()
            };
            if lag {
                lag_panels.push(panel)
            } else {
                panels.push(panel)
            }
            sa_cols.push((format!("{} vs {}{}", t.namendo, name, if lag { "(-1)" } else { "" }), sa));
        }
        if ctx.graphs_enabled() {
            for (ps, what, title) in [(panels, "shock", "shocks"), (lag_panels, "lags", "lagged endogenous")] {
                let n = ps.len();
                for (ifig, chunk) in ps.chunks(9).enumerate() {
                    let (_, nr, nc, _) = pltorg(n.min(9));
                    // Dynare names the last (partial) figure `_vs_shocks_`, full ones `_vs_shock_`
                    let full = chunk.len() == 9;
                    let stem = if what == "shock" && !full { "shocks" } else { what };
                    ctx.save(&Figure {
                        name: format!("{}_vs_{stem}_{}", t.namendo, ifig + 1),
                        title: format!("{} vs. {title} {}", t.namendo, ifig + 1),
                        rows: nr,
                        cols: nc,
                        panels: chunk.to_vec(),
                    });
                }
            }
        }
    }
    if sa_cols.is_empty() {
        return Ok(());
    }
    // Not printed by Dynare (figures only): the normalized elementary effects per entry.
    let labels: Vec<String> = sa_cols.iter().map(|(l, _)| l.clone()).collect();
    println!("\nReduced form screening (Morris elementary effects, mu* / max mu*):");
    let headers: Vec<&str> = std::iter::once("Entry").chain(pnames.iter().map(String::as_str)).collect();
    let vals: Vec<Vec<f64>> = sa_cols.iter().map(|(_, s)| s.clone()).collect();
    print!("{}", dyntable("", &headers, &labels, &vals, 0, 8, 3));
    if ctx.graphs_enabled() {
        let p = Panel {
            title: "Reduced form screening".into(),
            y_label: Some("Elementary Effects".into()),
            series: sa_cols
                .iter()
                .map(|(_, s)| Series::Points {
                    x: (1..=np).map(|k| k as f64).collect(),
                    y: s.clone(),
                    color: Color::Blue,
                    label: None,
                })
                .collect(),
            x_ticks: Some(pnames.iter().enumerate().map(|(k, n)| ((k + 1) as f64, n.clone())).collect()),
            ..Default::default()
        };
        ctx.save(&Figure {
            name: "redform_screen".into(),
            title: "Reduced form screening".into(),
            rows: 1,
            cols: 1,
            panels: vec![p],
        });
    }
    Ok(())
}
