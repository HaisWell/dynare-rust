// SPDX-License-Identifier: GPL-3.0-or-later
//! the `prior` command's `moments(distribution)`, `irfs(distribution)` and
//! `optimize` options (`matlab/cli/prior.m`, `estimation/prior_sampler.m`,
//! `cli/print_moments_implied_prior.m`, `estimation/optimize_prior.m`,
//! `estimation/minus_logged_prior_density.m`).
//!
//! The prior draws come from ChaCha8 (seeded with `--seed`), not MATLAB's generator: the
//! implied prior moments agree with Dynare's statistically (Monte Carlo error of `prior_mc`
//! draws); every draw is processed exactly as Dynare does.

use anyhow::Result;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use super::Ctx;
use crate::analysis::StochSimulOptions;
use crate::dsge::DsgeModel;
use crate::estimation::Objective;
use crate::estimation::prior::{log_prior_density, prior_bounds, prior_draw};
use crate::linalg::Mat;
use crate::matfile::{MatValue, write_values};
use crate::modfile::{Command, ModFile, OptValue, opt};

/// `options_` as the `prior` command at index `ci` sees them: every `stoch_simul` /
/// `estimation` option of the earlier commands (Dynare's options are global), and
/// `options_.prior_mc` (20000), `options_.mode_compute` (4).
pub struct PriorOptions {
    pub ss: StochSimulOptions,
    pub prior_mc: usize,
    pub mode_compute: i32,
    pub optim: Option<OptValue>,
}

impl PriorOptions {
    pub fn at(mf: &ModFile, ci: usize) -> PriorOptions {
        let mut ss = StochSimulOptions::default();
        let mut prior_mc = 20000;
        let mut mode_compute = 4;
        let mut optim = None;
        for c in mf.commands.iter().take(ci) {
            let o = match c {
                Command::StochSimul(o, _) | Command::Estimation(o, _) => {
                    let _ = ss.apply(o);
                    o
                }
                Command::Other(_, o) => o,
                _ => continue,
            };
            if let Some(OptValue::Num(v)) = opt(o, "prior_mc") {
                prior_mc = *v as usize;
            }
            if let Some(OptValue::Num(v)) = opt(o, "mode_compute") {
                mode_compute = *v as i32;
            }
            if let Some(v) = opt(o, "optim") {
                optim = Some(v.clone());
            }
        }
        // prior.m: options_.order = 1 for everything below
        ss.order = 1;
        PriorOptions { ss, prior_mc, mode_compute, optim }
    }
}

/// MATLAB `num2str` of a scalar (`%.{max(floor(log10|x|)+5, 5)}g`, integers as integers).
fn num2str(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Inf".into() } else { "-Inf".into() };
    }
    if x == x.round() && x.abs() < 1e15 {
        return format!("{}", x as i64);
    }
    let dgt = if x == 0.0 { 0 } else { x.abs().log10().floor() as i32 };
    let p = (dgt + 5).max(5) as usize;
    crate::analysis::table::fmt_g(x, 0, p).trim().to_string()
}

/// One accepted prior draw (`pdraws{j,1..3}` with `INFO(1) == 0`).
struct Draw {
    q: Mat,
    dr: crate::solver::DecisionRule,
}

/// `prior_sampler(1, ...)`: `prior_mc` draws, solved; `None` where `dynare_resolve` fails.
fn sample(
    dsge: &DsgeModel,
    n: usize,
    seed: u64,
    restrictions: Option<&crate::gsa::calibration::Restrictions>,
) -> Vec<Option<Draw>> {
    // dprior(bayestopt_, options_.prior_trunc): draws truncated at the 1e-10 quantiles
    let bounds = prior_bounds(&dsge.priors, 1e-10);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed ^ 0x5052_494f_525f_4d43);
    let draws: Vec<Vec<f64>> = (0..n).map(|_| prior_draw(&dsge.priors, &bounds, &mut rng)).collect();
    draws
        .par_iter()
        .map(|x| {
            let r = dsge.resolve(x).ok()?;
            // prior_sampler.m: endogenous_prior_restrictions (info 49) reject the draw
            if let Some(rs) = restrictions {
                if !rs.is_empty() {
                    let v = crate::gsa::calibration::endogenous_prior_restrictions(dsge, &r.dr, &r.q, rs).ok()?;
                    if v.info.0 != 0 {
                        return None;
                    }
                }
            }
            Some(Draw { q: r.q, dr: r.dr })
        })
        .collect()
}

/// `prior moments(distribution)` and / or `prior irfs(distribution)`.
pub fn distribution(mf: &ModFile, dsge: &DsgeModel, ctx: &Ctx, seed: u64, moments: bool, irfs: bool) -> Result<()> {
    let po = PriorOptions::at(mf, ctx.ci);
    let o = &po.ss;
    let model = &mf.model;
    let n = model.orig_endo_nbr;
    let nmc = po.prior_mc;
    let rs = crate::gsa::calibration::restrictions_before(mf, ctx.ci)?;
    let draws = sample(dsge, nmc, seed, Some(&rs));
    let ivar: Vec<usize> = (0..n).collect();
    let mut mo = o.moment_options();
    mo.nodecomposition = true;
    let shocks = crate::analysis::irf::irf_shock_indices(&model.exo_names, o.irf_shocks.as_deref())?;
    // per draw: (mean, var) and the IRFs per shock (n x irf)
    type PerDraw = (Option<(Vec<f64>, Mat)>, Vec<Option<Mat>>);
    let per: Vec<PerDraw> = draws
        .par_iter()
        .map(|d| {
            let Some(d) = d else { return (None, vec![None; shocks.len()]) };
            let m = if moments {
                crate::analysis::moments::th_autocovariances(&d.dr, &d.q, &ivar, &mo)
                    .ok()
                    .map(|th| (th.mean(&d.dr), th.gamma0))
            } else {
                None
            };
            let mut ir = vec![None; shocks.len()];
            if irfs {
                let cs = crate::analysis::vardec::lower_cholesky(&d.q, o.add_tiny_number_to_cholesky);
                for (k, &s) in shocks.iter().enumerate() {
                    if cs[(s, s)] > 5e-7 {
                        let e1: Vec<f64> = cs.column(s).iter().copied().collect();
                        let mut y = crate::analysis::irf::irf(&d.dr, &e1, o.irf);
                        if o.relative_irf {
                            y *= 100.0 / cs[(s, s)];
                        }
                        ir[k] = Some(y.rows(0, n).into_owned());
                    }
                }
            }
            (m, ir)
        })
        .collect();
    let prior_dir = ctx.dir.join(ctx.fname).join("prior");
    std::fs::create_dir_all(&prior_dir)?;
    let path = prior_dir.join(format!("{}_endogenous_variables_prior_draws.mat", ctx.fname));
    let mut vars: Vec<(String, MatValue)> = Vec::new();
    if moments {
        let mut first = vec![f64::NAN; n * nmc];
        let mut second = vec![f64::NAN; n * n * nmc];
        for (j, (m, _)) in per.iter().enumerate() {
            if let Some((mean, var)) = m {
                first[j * n..(j + 1) * n].copy_from_slice(mean);
                second[j * n * n..(j + 1) * n * n].copy_from_slice(var.as_slice());
            }
        }
        // FirstOrderMoments(:, any(isnan(.), 1)) = []; same for the second-order moments
        let keep1: Vec<usize> = (0..nmc).filter(|&j| first[j * n..(j + 1) * n].iter().all(|v| !v.is_nan())).collect();
        let keep2: Vec<usize> =
            (0..nmc).filter(|&j| second[j * n * n..(j + 1) * n * n].iter().all(|v| !v.is_nan())).collect();
        let mm: Vec<f64> =
            (0..n).map(|i| keep1.iter().map(|&j| first[j * n + i]).sum::<f64>() / keep1.len() as f64).collect();
        let vm: Vec<f64> = (0..n)
            .map(|i| keep1.iter().map(|&j| (first[j * n + i] - mm[i]).powi(2)).sum::<f64>() / keep1.len() as f64)
            .collect();
        let mv = Mat::from_fn(n, n, |a, b| {
            keep2.iter().map(|&j| second[j * n * n + b * n + a]).sum::<f64>() / keep2.len() as f64
        });
        let vv = Mat::from_fn(n, n, |a, b| {
            keep2.iter().map(|&j| (second[j * n * n + b * n + a] - mv[(a, b)]).powi(2)).sum::<f64>()
                / keep2.len() as f64
        });
        print!("\n\n");
        // print_moments_implied_prior.m uses disp (printed whatever noprint)
        print!("{}", moments_implied_prior_text(&model.endo_names[..n], &mm, &vm, &mv, &vv));
        vars.push(("FirstOrderMoments".into(), MatValue::Numeric { dims: vec![n, nmc], data: first }));
        vars.push(("SecondOrderMoments".into(), MatValue::Numeric { dims: vec![n, n, nmc], data: second }));
    }
    if irfs {
        let mut fields = Vec::new();
        let mut elems = Vec::new();
        for (k, &s) in shocks.iter().enumerate() {
            let mut data = vec![f64::NAN; n * o.irf * nmc];
            for (j, (_, ir)) in per.iter().enumerate() {
                if let Some(y) = &ir[k] {
                    data[j * n * o.irf..(j + 1) * n * o.irf].copy_from_slice(y.as_slice());
                }
            }
            fields.push(model.exo_names[s].clone());
            elems.push(MatValue::Numeric { dims: vec![n, o.irf, nmc], data });
        }
        vars.push((
            "ImpulseResponseFunctions".into(),
            MatValue::Struct { dims: vec![1, 1], fields, elems: vec![elems] },
        ));
    }
    let refs: Vec<(&str, &MatValue)> = vars.iter().map(|(k, v)| (k.as_str(), v)).collect();
    write_values(&path, &refs)?;
    let ok = draws.iter().filter(|d| d.is_some()).count();
    println!(
        "prior: {} of {nmc} prior draws solved (dynare-rust's own random stream); implied prior draws written to {}.",
        ok,
        path.display()
    );
    Ok(())
}

/// `print_moments_implied_prior.m` (`strvcat` blocks padded to a common width).
pub fn moments_implied_prior_text(names: &[String], mm: &[f64], vm: &[f64], mv: &Mat, vv: &Mat) -> String {
    let n = names.len();
    let block = |cols: Vec<Vec<String>>| -> String {
        // [T1, T0, T2, ...]: each column padded (strvcat) then joined with two spaces
        let widths: Vec<usize> = cols.iter().map(|c| c.iter().map(|s| s.chars().count()).max().unwrap_or(0)).collect();
        let rows: Vec<String> = (0..cols[0].len())
            .map(|r| cols.iter().zip(&widths).map(|(c, &w)| format!("{:<w$}", c[r])).collect::<Vec<_>>().join("  "))
            .collect();
        let width = rows[0].chars().count() + 1;
        let l0 = "-".repeat(width);
        let mut out = String::new();
        let pad = |s: &str| format!("{s:<width$}");
        out += &format!("{l0}\n{}\n{l0}\n", pad(&rows[0]));
        for r in &rows[1..] {
            out += &format!("{}\n", pad(r));
        }
        out += &format!("{l0}\n");
        out
    };
    let mut s = String::from("(Implied) Prior moments of the endogenous variables' expectation\n");
    s += &"-".repeat(64);
    s.push('\n');
    let mut t1 = vec!["VARIABLE ".to_string()];
    let mut t2 = vec!["Prior mean \t Prior st. dev.".to_string()];
    for i in 0..n {
        t1.push(names[i].clone());
        t2.push(format!(" {:6.4} \t {:6.4}", mm[i], vm[i].sqrt()));
    }
    s += "\n\n";
    s += &block(vec![t1, t2]);
    s += "\n\n";
    s += "(Implied) Prior moments of the endogenous variables' (co)variance\n";
    s += &"-".repeat(61);
    s.push('\n');
    let (mut a, mut b, mut c, mut d) = (
        vec!["VARIABLE-1".to_string()],
        vec!["VARIABLE-2".to_string()],
        vec!["Prior mean".to_string()],
        vec!["Prior st.dev.".to_string()],
    );
    for i in 0..n {
        for j in i..n {
            a.push(names[i].clone());
            b.push(names[j].clone());
            c.push(format!("{:12.8}", mv[(i, j)]));
            d.push(format!("{:12.8}", vv[(i, j)]));
        }
    }
    s += "\n\n";
    s += &block(vec![a, b, c, d]);
    s += "\n\n";
    s
}

/// `minus_logged_prior_density.m` (mode_compute != 1): `-log p(x)`, +inf off the prior
/// support or where the model has no solution.
struct MinusLogPrior<'a> {
    dsge: &'a DsgeModel,
}

impl Objective for MinusLogPrior<'_> {
    fn dim(&self) -> usize {
        self.dsge.priors.len()
    }
    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let p = &self.dsge.priors;
        if x.iter().zip(p).any(|(v, q)| *v < q.p3) {
            return (f64::INFINITY, 41);
        }
        if x.iter().zip(p).any(|(v, q)| *v > q.p4) {
            return (f64::INFINITY, 42);
        }
        if let Err(e) = self.dsge.resolve(x) {
            return (f64::INFINITY, e.info);
        }
        let f = -log_prior_density(x, p);
        if f.is_infinite() {
            (f64::INFINITY, 40)
        } else if f.is_nan() {
            (f64::INFINITY, 47)
        } else {
            (f, 0)
        }
    }
}

/// `prior optimize` (`optimize_prior.m`).
pub fn optimize(mf: &ModFile, dsge: &DsgeModel, ctx: &Ctx, seed: u64) -> Result<()> {
    let po = PriorOptions::at(mf, ctx.ci);
    let p = &dsge.priors;
    let xparam1: Vec<f64> = p.iter().map(|q| q.mean).collect();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed ^ 0x4f50_5449_4d5f_5052);
    let (mut scale, mut iter) = (1.0f64, 0usize);
    let xinit = loop {
        let x: Vec<f64> = xparam1
            .iter()
            .map(|m| {
                let z: f64 = StandardNormal.sample(&mut rng);
                m + scale * z
            })
            .collect();
        if x.iter().zip(p).all(|(v, q)| *v > q.p3 && *v < q.p4) {
            if dsge.resolve(&x).is_ok() {
                break x;
            }
        } else if iter == 2000 {
            scale /= 1.1;
            iter = 0;
        } else {
            iter += 1;
        }
    };
    let names: Vec<String> =
        dsge.estimated_names.iter().map(|n| crate::estimation::plots::dynare_param_name(n)).collect();
    let bounds =
        crate::estimation::prior::Bounds { lb: p.iter().map(|q| q.p3).collect(), ub: p.iter().map(|q| q.p4).collect() };
    let sd = dsge.prior_sd();
    let mctx = crate::estimation::optim::MinimizeContext {
        bounds: &bounds,
        names: &names,
        prior_sd: &sd,
        optim: po.optim.as_ref(),
        verbose: true,
        seed,
        stream: 0x5052_494f_525f_4f50,
    };
    let obj = MinusLogPrior { dsge };
    let res = crate::estimation::optim::minimize(po.mode_compute, &obj, &xinit, &mctx)?;
    print!("\n\n");
    println!("------------------");
    println!("PRIOR OPTIMIZATION");
    println!("------------------");
    println!();
    for (i, name) in names.iter().enumerate() {
        println!("parameter {}: {}.", i + 1, name);
        println!("  Initial condition ........ {}.", num2str(xinit[i]));
        println!("  Prior mode ............... {}.", num2str(p[i].p5));
        println!("  Optimized prior mode ..... {}.", num2str(res.x[i]));
        println!();
    }
    println!();
    let out = ctx.out_dir.join(format!("{}_prior_optimize.json", ctx.fname));
    std::fs::create_dir_all(ctx.out_dir)?;
    std::fs::write(
        &out,
        serde_json::to_string_pretty(
            &serde_json::json!({ "names": names, "xinit": xinit, "prior_mode": p.iter().map(|q| q.p5).collect::<Vec<_>>(), "optimized": res.x, "fval": res.fval }),
        )?,
    )?;
    Ok(())
}
