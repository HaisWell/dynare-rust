// SPDX-License-Identifier: GPL-3.0-or-later
//! The rest of the `sbvar` command: the Gibbs-sampled terms of the log marginal data density
//! (`ms_mardd.m` with `fn_gibbsrvar_setup` / `fn_gibbsrvar`, Waggoner-Zha), the growth-rate files
//! `outactqmygdata.prn` / `outyrqm.prn` (`fn_datana`) and the impulse response and forecast
//! graphs (`fn_imcgraph`, `fn_foregraph`).
//!
//! The Gibbs sampler is random: Dynare's and this port's `log mdd` agree to Monte Carlo accuracy
//! only. `ms_mardd.m` hard-codes `nbuffer = 1000`, 1000 tossed-away and 10000 used draws.

use std::path::Path;

use anyhow::{Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use super::sbvar::SbvarResult;
use crate::estimation::special::gammaln;
use crate::graphics::{self, Color, Figure, GraphOptions, LineStyle, Panel, Series};
use crate::linalg::Mat;

/// `log p(a0_k | Y, a_others)` for each equation and the log marginal data density
/// `vlog_a0p + vlog_Y_a - sum(vlog_a0_Yao) - vlog_ap_Ya0`.
#[derive(Clone, Debug)]
pub struct GibbsMdd {
    pub vlog_a0_yao: Vec<f64>,
    pub log_mdd: f64,
}

fn randn(rng: &mut ChaCha8Rng) -> f64 {
    rng.sample(StandardNormal)
}

/// Unit vector `w1` (in `U_k T_k`-coordinates) orthogonal to the other columns of `A0`: the first
/// column of `W` of `fn_gibbsrvar`.
fn null_direction(a0: &Mat, ki: usize, ut: &Mat) -> Result<Vec<f64>> {
    let nvar = a0.ncols();
    let mut x = a0.clone();
    x.column_mut(ki).fill(0.0);
    let svd = x.transpose().svd(false, true);
    let vt = svd.v_t.ok_or_else(|| anyhow::anyhow!("sbvar: SVD failed"))?;
    let imin = (0..nvar).min_by(|&a, &b| svd.singular_values[a].total_cmp(&svd.singular_values[b])).unwrap();
    let w: Vec<f64> = (0..nvar).map(|j| vt[(imin, j)]).collect();
    let w0 = ut.transpose() * Mat::from_column_slice(nvar, 1, &w);
    let nrm = w0.norm();
    Ok(w0.iter().map(|v| v / nrm).collect())
}

/// Householder matrix whose first column is `-+w1`.
fn basis(w1: &[f64]) -> Mat {
    let n = w1.len();
    let mut v = w1.to_vec();
    v[0] += if w1[0] >= 0.0 { 1.0 } else { -1.0 };
    let vv: f64 = v.iter().map(|x| x * x).sum();
    let mut h = Mat::identity(n, n);
    if vv > 0.0 {
        for i in 0..n {
            for j in 0..n {
                h[(i, j)] -= 2.0 * v[i] * v[j] / vv;
            }
        }
    }
    h
}

/// `fn_gibbsrvar` over the columns `from..nvar`; returns the `w1` of column `from`.
fn gibbs_sweep(
    a0: &mut Mat,
    ut: &[Mat],
    n0: &[usize],
    fss: usize,
    from: usize,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<f64>> {
    let nvar = a0.ncols();
    let mut first = Vec::new();
    let jstd = (1.0 / fss as f64).sqrt();
    for ki in from..nvar {
        let w1 = null_direction(a0, ki, &ut[ki])?;
        let w = basis(&w1);
        let mut beta = vec![0.0; n0[ki]];
        for b in beta.iter_mut().skip(1) {
            *b = jstd * randn(rng);
        }
        let ss: f64 = (0..fss + 1).map(|_| (jstd * randn(rng)).powi(2)).sum();
        beta[0] = if rng.gen_range(0.0..1.0) < 0.5 { ss.sqrt() } else { -ss.sqrt() };
        let col = &ut[ki] * (&w * Mat::from_column_slice(n0[ki], 1, &beta));
        a0.set_column(ki, &col.column(0));
        if ki == from {
            first = w1;
        }
    }
    Ok(first)
}

/// Log marginal data density of the constant-parameter SBVAR (Bayesian prior, free `A0`).
pub fn gibbs_mdd(r: &SbvarResult, seed: u64, ndraws1: usize, ndraws2: usize) -> Result<GibbsMdd> {
    let (Some(a0p), Some(ya), Some(apya0)) = (r.vlog_a0p, r.vlog_y_a, r.vlog_ap_ya0) else {
        bail!("sbvar: the marginal data density needs the Bayesian prior and a free A0");
    };
    let nvar = r.nvar;
    let fss = r.fss;
    let n0: Vec<usize> = r.ui.iter().map(|u| u.ncols()).collect();
    // fn_gibbsrvar_setup: R = chol(H0inv/fss) (upper), UT = U inv(R)
    let mut rmat = Vec::new();
    let mut ut = Vec::new();
    for k in 0..nvar {
        let l = (&r.h0inv[k] / fss as f64)
            .cholesky()
            .ok_or_else(|| anyhow::anyhow!("sbvar: H0inv is not positive definite"))?;
        let rk = l.l().transpose();
        let rinv = rk.clone().try_inverse().ok_or_else(|| anyhow::anyhow!("sbvar: singular chol factor"))?;
        ut.push(&r.ui[k] * rinv);
        rmat.push(rk);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let f = fss as f64;
    let term = |k: usize, w1: &[f64]| -> f64 {
        let nk = n0[k] as f64;
        let bk = r.ui[k].transpose() * r.a0.column(k);
        let rb = &rmat[k] * bk;
        let logdet: f64 = (0..n0[k]).map(|i| rmat[k][(i, i)].abs().ln()).sum();
        let g1: f64 = w1.iter().zip(rb.iter()).map(|(a, b)| a * b).sum();
        0.5 * (f + nk) * f.ln() + logdet
            - 0.5 * (nk - 1.0) * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * (f + 1.0) * 2f64.ln()
            - gammaln(0.5 * (f + 1.0))
            + f * g1.abs().ln()
            - 0.5 * f * rb.norm_squared()
    };
    let mut yao = vec![0.0; nvar];
    for k in 0..nvar {
        let mut a = r.a0.clone();
        if k + 1 < nvar {
            for _ in 0..ndraws1 {
                gibbs_sweep(&mut a, &ut, &n0, fss, k, &mut rng)?;
            }
            let mut v = Vec::with_capacity(ndraws2);
            for _ in 0..ndraws2 {
                let w1 = gibbs_sweep(&mut a, &ut, &n0, fss, k, &mut rng)?;
                v.push(term(k, &w1));
            }
            let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            yao[k] = m - (ndraws2 as f64).ln() + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        } else {
            let w1 = gibbs_sweep(&mut a, &ut, &n0, fss, k, &mut rng)?;
            yao[k] = term(k, &w1);
        }
    }
    let log_mdd = a0p + ya - yao.iter().sum::<f64>() - apya0;
    Ok(GibbsMdd { vlog_a0_yao: yao, log_mdd })
}

// ---------------------------------------------------------------------------------------
// fn_datana
// ---------------------------------------------------------------------------------------

/// `[year subperiod data...]` rows of consecutive periods starting at `(y0, s0)`.
pub fn with_dates(data: &Mat, y0: i64, s0: i64, freq: usize) -> Mat {
    let mut m = Mat::zeros(data.nrows(), data.ncols() + 2);
    for i in 0..data.nrows() {
        let (y, s) = date_at(y0, s0, freq, i);
        m[(i, 0)] = y as f64;
        m[(i, 1)] = s as f64;
        for j in 0..data.ncols() {
            m[(i, 2 + j)] = data[(i, j)];
        }
    }
    m
}

/// Rows of `x` between the two dates (inclusive), `fn_dataext`.
pub fn data_ext(x: &Mat, from: (i64, i64), to: (i64, i64), freq: i64) -> Option<Mat> {
    let (y0, s0) = (x[(0, 0)] as i64, x[(0, 1)] as i64);
    let idx = |d: (i64, i64)| (d.0 - y0) * freq + d.1 - s0;
    let (a, b) = (idx(from), idx(to));
    if a < 0 || b < a || b >= x.nrows() as i64 {
        return None;
    }
    Some(x.rows(a as usize, (b - a + 1) as usize).into_owned())
}

/// `fn_datana`: `yactyrge` (annual-average growth rates), `yactqmyge` (year-on-year rates).
pub struct Datana {
    pub yactyrge: Mat,
    pub yactqmyge: Mat,
}

/// `range` is `fn_datana`'s `Byrqm, Eyrqm`.
pub fn fn_datana(
    x: &Mat,
    q: usize,
    vlistlog: &[usize],
    vlistper: &[usize],
    range: Option<((i64, i64), (i64, i64))>,
) -> Result<Datana> {
    if x.nrows() < 2 * q {
        bail!("We need at least two years of xdatae to get annual rates.  Check xdatae!!");
    }
    let (y0, s0) = (x[(0, 0)] as i64, x[(0, 1)] as i64);
    let (brow, erow, yr0) = match range {
        None => (0usize, x.nrows() - 1, x[(q, 0)] as i64),
        Some((b, e)) => {
            let row = |d: (i64, i64)| (d.0 - y0) * q as i64 + d.1 - s0;
            let (br, er) = (row(b), row(e));
            if br < 0 || er < br || er >= x.nrows() as i64 {
                bail!("fn_datana: date range outside the data");
            }
            (br as usize, er as usize, b.0 + 1)
        }
    };
    let n = erow - brow + 1;
    if n < q + 1 {
        bail!("fn_datana: sample too short");
    }
    let nyr = n / q;
    let nv = x.ncols() - 2;
    let lvl = |t: usize, j: usize| x[(brow + t, 2 + j)];
    let is_log = |j: usize| vlistlog.contains(&j);
    let is_per = |j: usize| vlistper.contains(&j);
    let mut qy = Mat::zeros(n - q, nv + 2);
    for t in q..n {
        qy[(t - q, 0)] = x[(brow + t, 0)];
        qy[(t - q, 1)] = x[(brow + t, 1)];
        for j in 0..nv {
            qy[(t - q, 2 + j)] = if is_log(j) {
                100.0 * ((lvl(t, j) - lvl(t - q, j)).exp() - 1.0)
            } else if is_per(j) {
                100.0 * lvl(t, j)
            } else {
                lvl(t, j)
            };
        }
    }
    let mut yg = Mat::zeros(nyr.saturating_sub(1), nv + 2);
    for yi in 1..nyr {
        yg[(yi - 1, 0)] = (yr0 + yi as i64 - 1) as f64;
        for j in 0..nv {
            let avg = |y: usize| (0..q).map(|i| lvl(y * q + i, j)).sum::<f64>() / q as f64;
            yg[(yi - 1, 2 + j)] = if is_log(j) {
                100.0 * ((avg(yi) - avg(yi - 1)).exp() - 1.0)
            } else if is_per(j) {
                100.0 * avg(yi)
            } else {
                avg(yi)
            };
        }
    }
    Ok(Datana { yactyrge: yg, yactqmyge: qy })
}

/// `%16.7e` as C prints it (`-9.3072871e+01`).
fn c_exp(v: f64) -> String {
    let s = format!("{v:.7e}");
    let (m, e) = s.split_once('e').unwrap();
    let e: i32 = e.parse().unwrap();
    format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

/// Calendar position `(year, subperiod)` of row `i` of a sample starting at `(y0, s0)`.
pub fn date_at(y0: i64, s0: i64, freq: usize, i: usize) -> (i64, i64) {
    let k = s0 - 1 + i as i64;
    (y0 + k.div_euclid(freq as i64), k.rem_euclid(freq as i64) + 1)
}

/// Writes `outactqmygdata.prn` and `outyrqm.prn` of `msstart2.m` to `dir`.
pub fn write_prn(dir: &Path, z: &Mat, y0: i64, s0: i64, freq: usize, vlistlog: &[usize]) -> Result<()> {
    let nvar = z.ncols();
    let x = with_dates(z, y0, s0, freq);
    let vper: Vec<usize> = (0..nvar).filter(|j| !vlistlog.contains(j)).collect();
    let (yb, qb) = if s0 == 1 { (y0, s0) } else { (y0 + 1, 1) };
    let last = date_at(y0, s0, freq, z.nrows() - 1);
    let Ok(d) = fn_datana(&x, freq, vlistlog, &vper, Some(((yb, qb), last))) else {
        println!("sbvar: sample too short for the growth-rate files");
        return Ok(());
    };
    let mut s = String::new();
    for i in 0..d.yactqmyge.nrows() {
        for j in 0..d.yactqmyge.ncols() {
            s += &format!("{:>16}", c_exp(d.yactqmyge[(i, j)]));
        }
        s.push('\n');
    }
    std::fs::write(dir.join("outactqmygdata.prn"), s)?;
    let mut s = String::new();
    for i in 0..d.yactyrge.nrows() {
        s += &format!("{:4.0} {:2.0}", d.yactyrge[(i, 0)], d.yactyrge[(i, 1)]);
        for j in 0..nvar {
            s += &format!(" {:8.3}", d.yactyrge[(i, 2 + j)]);
        }
        s.push('\n');
    }
    std::fs::write(dir.join("outyrqm.prn"), s)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------
// graphs
// ---------------------------------------------------------------------------------------

fn line(x: Vec<f64>, y: Vec<f64>, style: LineStyle, color: Color) -> Series {
    Series::Line { x, y, color, style, width: 1.0, label: None }
}

/// Impulse responses (`fn_imcgraph`: variable `i` in row `i`, shock `j` in column `j`), the
/// unconditional forecast in annual growth rates (`fn_foregraph`) and the year-on-year growth
/// rates of the data. Returns the files written.
#[allow(clippy::too_many_arguments)]
pub fn graphs(
    r: &SbvarResult,
    names: &[String],
    z: &Mat,
    y0: i64,
    s0: i64,
    freq: usize,
    vlistlog: &[usize],
    forecast_steps: usize,
    fname: &str,
    opts: &GraphOptions,
) -> Result<Vec<std::path::PathBuf>> {
    let nvar = r.nvar;
    let mut out = Vec::new();
    let steps = r.imf.nrows();
    let t: Vec<f64> = (1..=steps).map(|v| v as f64).collect();
    let mut panels = Vec::new();
    for i in 0..nvar {
        for j in 0..nvar {
            let y: Vec<f64> = (0..steps).map(|k| r.imf[(k, j * nvar + i)]).collect();
            panels.push(Panel {
                title: if i == 0 { format!("shock {}", names[j]) } else { String::new() },
                series: vec![
                    line(t.clone(), y, LineStyle::Solid, Color::Blue),
                    Series::HLine { y: 0.0, color: Color::Red, style: LineStyle::Dotted },
                ],
                y_label: if j == 0 { Some(names[i].clone()) } else { None },
                ..Default::default()
            });
        }
    }
    out.extend(graphics::save(
        &Figure { name: "sbvar_irf".into(), title: "Impulse responses".into(), rows: nvar, cols: nvar, panels },
        fname,
        opts,
    )?);
    let x = with_dates(z, y0, s0, freq);
    let vper: Vec<usize> = (0..nvar).filter(|j| !vlistlog.contains(j)).collect();
    let end = date_at(y0, s0, freq, z.nrows() - 1);
    let start = ((end.0 - 4).max(y0), 1);
    if let Some(act) = data_ext(&x, start, end, freq as i64) {
        let mut af = Mat::zeros(act.nrows() + forecast_steps, nvar + 2);
        af.rows_mut(0, act.nrows()).copy_from(&act);
        for k in 0..forecast_steps {
            let (yy, ss) = date_at(y0, s0, freq, z.nrows() + k);
            af[(act.nrows() + k, 0)] = yy as f64;
            af[(act.nrows() + k, 1)] = ss as f64;
            for v in 0..nvar {
                af[(act.nrows() + k, 2 + v)] = r.forecast[(k, v)];
            }
        }
        if let (Ok(a), Ok(f)) =
            (fn_datana(&act, freq, vlistlog, &vper, None), fn_datana(&af, freq, vlistlog, &vper, None))
        {
            let col = |m: &Mat, c: usize| -> Vec<f64> { (0..m.nrows()).map(|i| m[(i, c)]).collect() };
            let mut panels = Vec::new();
            for v in 0..nvar {
                let sc = |c: Vec<f64>| c.into_iter().map(|x| x / 100.0).collect::<Vec<_>>();
                panels.push(Panel {
                    title: if v == 0 { format!("{}:{} Forecast", end.0, end.1) } else { String::new() },
                    series: vec![
                        line(col(&a.yactyrge, 0), sc(col(&a.yactyrge, 2 + v)), LineStyle::Solid, Color::Blue),
                        line(col(&f.yactyrge, 0), sc(col(&f.yactyrge, 2 + v)), LineStyle::Dashed, Color::Green),
                    ],
                    y_label: Some(names[v].clone()),
                    x_label: if v + 1 == nvar { Some("unconditional".into()) } else { None },
                    ..Default::default()
                });
            }
            out.extend(graphics::save(
                &Figure {
                    name: "sbvar_forecast".into(),
                    title: "Unconditional forecast".into(),
                    rows: nvar,
                    cols: 1,
                    panels,
                },
                fname,
                opts,
            )?);
        }
    }
    let (yb, qb) = if s0 == 1 { (y0, s0) } else { (y0 + 1, 1) };
    if let Ok(d) = fn_datana(&x, freq, vlistlog, &vper, Some(((yb, qb), end))) {
        let tt: Vec<f64> =
            (0..d.yactqmyge.nrows()).map(|i| d.yactqmyge[(i, 0)] + (d.yactqmyge[(i, 1)] - 1.0) / freq as f64).collect();
        let panels: Vec<Panel> = (0..nvar)
            .map(|v| Panel {
                title: names[v].clone(),
                series: vec![line(
                    tt.clone(),
                    (0..tt.len()).map(|i| d.yactqmyge[(i, 2 + v)] / 100.0).collect(),
                    LineStyle::Solid,
                    Color::Blue,
                )],
                ..Default::default()
            })
            .collect();
        out.extend(graphics::save(
            &Figure { name: "sbvar_growth".into(), title: "Year-on-year growth".into(), rows: nvar, cols: 1, panels },
            fname,
            opts,
        )?);
    }
    Ok(out)
}
