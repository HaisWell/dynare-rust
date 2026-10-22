// SPDX-License-Identifier: GPL-3.0-or-later
//! `init_<tag>.dat` / `est_final_<tag>.out`: the files through which Dynare's `ms_*` commands pass
//! the model and its estimate from one command (or one Dynare run) to the next.
//!
//! The writer follows the layout of the files Dynare 7.2's mex writes (state-variable specification,
//! VAR specification with `U[j]`, `V[j]`, `W[j]`, prior variances, specification flag, translation
//! tables, data, then `Initial: ` / `Posterior mode: ` parameter blocks); the reader accepts both our
//! files and Dynare's (the free parameters are then in Dynare's `U`/`V` bases, which are read along).
//! One extra block, `Dates (frequency, initial year, initial subperiod)`, keeps the sample dates.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};

use super::identification::Restrictions;
use super::ms_estimation::{ChainSpec, MsEstimate, MsParts, MsSetup, chain_columns};
use super::ms_optim::{Trace, fmt_g};
use crate::linalg::Mat;

/// C `%.<prec>e`.
pub fn fmt_e(x: f64, prec: usize) -> String {
    let s = format!("{:.*e}", prec, x);
    let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
    let e: i32 = e.parse().unwrap_or(0);
    format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

fn mat_text(out: &mut String, m: &Mat) {
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            let _ = write!(out, "{:>22} ", fmt_e(m[(i, j)], 14));
        }
        out.push('\n');
    }
    out.push('\n');
}

fn ints_line(out: &mut String, v: &[usize]) {
    for x in v {
        let _ = write!(out, "{x} ");
    }
    out.push('\n');
}

fn chain_title(s: &MsSetup, t: &str, c: usize) -> String {
    if s.chains.len() > 1 { format!("{t} for state_variable[{}]", c + 1) } else { t.to_string() }
}

/// The model part (everything but the parameter blocks).
pub fn spec_text(s: &MsSetup) -> String {
    let mut o = String::new();
    o.push_str("//*********************** State Variable Specification ************************//\n");
    let nch = if s.chains.len() > 1 { s.chains.len() } else { 0 };
    let _ = write!(o, "//== Number independent state variables ==//\n{nch}\n");
    if nch > 0 {
        o.push_str("//*****************************************************************************//\n\n");
    } else {
        o.push('\n');
    }
    for (c, ch) in s.chains.iter().enumerate() {
        if nch > 0 {
            let _ = write!(
                o,
                "//********************** Specification for state_variable[{}] **********************//\n\n",
                c + 1
            );
            let _ = write!(o, "//== {} ==//\n0\n\n", chain_title(s, "Number independent state variables", c));
        }
        let _ = write!(o, "//== {} ==//\n{}\n\n", chain_title(s, "Number states", c), ch.n);
        let _ = write!(o, "//== {} ==//\n0\n\n", chain_title(s, "Number of lags encoded", c));
        let _ = write!(o, "//== {} ==//\n{}\n\n", chain_title(s, "Number of base states", c), ch.n);
        let _ = writeln!(o, "//== {} ==//", chain_title(s, "Prior", c));
        mat_text(&mut o, &ch.alpha);
        let nd = if ch.n > 1 { ch.n } else { 0 };
        let _ = write!(o, "//== {} ==//\n{nd}\n\n", chain_title(s, "Number Dirichlet variables", c));
        let _ = writeln!(o, "//== {} ==//", chain_title(s, "Dirichlet dimensions", c));
        ints_line(&mut o, &vec![2; nd]);
        o.push('\n');
        let _ = writeln!(o, "//== {} ==//", chain_title(s, "Dirichlet index (total)", c));
        for i in 0..ch.n {
            for j in 0..ch.n {
                let (d1, d2, nr) = chain_columns(ch.n, j);
                let k: i64 = if ch.n == 1 {
                    0
                } else if d1.iter().take(nr).any(|&(r, _)| r == i) {
                    2 * j as i64
                } else if d2 == i {
                    2 * j as i64 + 1
                } else {
                    -1
                };
                let _ = write!(o, "{k} ");
            }
            o.push('\n');
        }
        o.push('\n');
        let _ = writeln!(o, "//== {} ==//", chain_title(s, "Dirichlet multipliers", c));
        let mult = Mat::from_fn(ch.n, ch.n, |i, j| {
            let (d1, _, nr) = chain_columns(ch.n, j);
            d1.iter().take(nr).find(|&&(r, _)| r == i).map_or(1.0, |&(_, w)| w)
        });
        mat_text(&mut o, &mult);
        o.pop();
        o.push_str("//*****************************************************************************//\n\n");
    }
    let (n, k) = (s.nvar, s.ncoef);
    let _ = write!(
        o,
        "//== Number Variables ==//\n{n}\n\n//== Number Lags ==//\n{}\n\n//== Exogenous Variables ==//\n1\n\n",
        s.nlags
    );
    let _ = write!(o, "//== Number States ==//\n{}\n\n//== Number Observations ==//\n{}\n\n", s.nstates, s.nobs);
    o.push_str("//== Number of free parameters in each column of A0 ==//\n");
    ints_line(&mut o, &s.n0);
    o.push('\n');
    for j in 0..n {
        let _ = writeln!(o, "//== U[{}] ==//", j + 1);
        mat_text(&mut o, &s.ui[j]);
    }
    o.push_str("//== Number of free parameters in each column of Aplus ==//\n");
    ints_line(&mut o, &s.np);
    o.push('\n');
    for j in 0..n {
        let _ = writeln!(o, "//== V[{}] ==//", j + 1);
        mat_text(&mut o, &s.vi[j]);
    }
    o.push_str("//== Non-zero W[j] ==//\n");
    ints_line(&mut o, &s.w.iter().map(|w| (w.abs().max() > 0.0) as usize).collect::<Vec<_>>());
    o.push('\n');
    for j in 0..n {
        let _ = writeln!(o, "//== W[{}] ==//", j + 1);
        mat_text(&mut o, &s.w[j]);
    }
    o.push_str("//== Gamma prior on zeta - a ==//\n");
    mat_text(&mut o, &Mat::from_element(1, n, s.zeta_a));
    o.push_str("//== Gamma prior on zeta - b ==//\n");
    mat_text(&mut o, &Mat::from_element(1, n, s.zeta_b));
    for j in 0..n {
        let _ = writeln!(o, "//== Variance of Gaussian prior on column {} of A0 ==//", j + 1);
        mat_text(&mut o, &s.h0var[j]);
    }
    for j in 0..n {
        let _ = writeln!(o, "//== Variance of Gaussian prior on column {} of Aplus ==//", j + 1);
        mat_text(&mut o, &s.hpvar[j]);
    }
    let _ = write!(o, "//== Specification (0=default  1=Sims-Zha  2=Random Walk) ==//\n{}\n\n", s.specification);
    if s.specification == 1 {
        o.push_str("//== Variance of Gaussian prior on lambda ==//\n");
        mat_text(&mut o, &Mat::from_element(1, 1, s.lambda_var));
    }
    for (t, tab) in [("coefficient", &s.coef_idx), ("variance", &s.var_idx)] {
        let _ = writeln!(o, "//== Translation table for {t} states ==//");
        for row in tab.iter() {
            for x in row {
                let _ = write!(o, "{x:>4} ");
            }
            o.push('\n');
        }
        o.push('\n');
    }
    o.push_str("//== Metropolis kernel scales for A0 ==//\n");
    for j in 0..n {
        for _ in 0..s.n_coef_reg[j] {
            let _ = write!(o, "{:>22} ", fmt_e(1.0, 14));
        }
        o.push('\n');
    }
    o.push('\n');
    o.push_str("//== Data Y (nobs x nvars) ==//\n");
    mat_text(&mut o, &s.y);
    o.push_str("//== Data X (nobs x npre) ==//\n");
    mat_text(&mut o, &s.x);
    let _ = writeln!(
        o,
        "//== Dates (frequency, initial year, initial subperiod) ==//\n{} {} {}\n",
        s.freq, s.initial_year, s.initial_subperiod
    );
    let _ = k;
    o
}

/// `//== <header>Base transition matrix ==//`, `A0[s]`, `Aplus[s]`, `Zeta[s]` blocks.
pub fn params_text(s: &MsSetup, theta: &[f64], header: &str) -> Result<String> {
    let r = s.regimes(theta).ok_or_else(|| anyhow!("ms_estimation: parameters outside the parameter space"))?;
    let mut o = String::new();
    for (c, q) in r.qc.iter().enumerate() {
        if s.chains.len() > 1 {
            let _ = writeln!(o, "//== {header}Base transition matrix[{}] ==//", c + 1);
        } else {
            let _ = writeln!(o, "//== {header}Base transition matrix ==//");
        }
        for i in 0..q.nrows() {
            for j in 0..q.ncols() {
                let _ = write!(o, "{} ", fmt_e(q[(i, j)], 14));
            }
            o.push('\n');
        }
        o.push('\n');
    }
    for st in 0..s.nstates {
        let _ = writeln!(o, "//== {header}A0[{}] ==//", st + 1);
        mat_text(&mut o, &r.a0[st]);
        let _ = writeln!(o, "//== {header}Aplus[{}] ==//", st + 1);
        mat_text(&mut o, &r.aplus[st]);
        let _ = writeln!(o, "//== {header}Zeta[{}] ==//", st + 1);
        mat_text(&mut o, &Mat::from_fn(s.nvar, s.nvar, |i, j| if i == j { r.zeta[st][i] } else { 0.0 }));
    }
    Ok(o)
}

pub fn write_init(path: &Path, s: &MsSetup, theta: &[f64]) -> Result<()> {
    let text = spec_text(s) + &params_text(s, theta, "Initial: ")?;
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

/// `est_final_<tag>.out`: the model, the search log, the values at the start and at the mode, and the mode.
pub fn write_est_final(path: &Path, init_name: &str, e: &MsEstimate, start: &[f64], trace: &Trace) -> Result<()> {
    let s = &e.setup;
    let mut o = spec_text(s);
    let nq: usize = s.layout.q.iter().map(|v| v.len()).sum();
    let _ = write!(
        o,
        "Specification file: {init_name}\nInitialization file: {init_name}\nInitialization header: \"Initial: \"\n"
    );
    let _ = write!(
        o,
        "Number free parameters in transition matrix: {nq}\nNumber free parameters in theta: {}\n",
        s.layout.len - nq
    );
    let (l0, p0) = (s.log_likelihood(start), s.log_prior(start));
    let _ = write!(
        o,
        "Initial likelihood Value: {}\nInitial prior Value: {}\nInitial posterior Value: {}\n",
        fmt_g(l0),
        fmt_g(p0),
        fmt_g(l0 + p0)
    );
    for l in &trace.lines {
        o.push_str(l);
        o.push('\n');
    }
    let _ = write!(
        o,
        "Final likelihood Value: {}\nFinal prior Value: {}\nFinal posterior Value: {}\n\n",
        fmt_g(e.log_likelihood),
        fmt_g(e.log_prior),
        fmt_g(e.log_posterior)
    );
    o.push_str("//== Posterior mode: Free parameters ==//\n");
    for x in &e.theta {
        let _ = write!(o, "{} ", fmt_e(*x, 16));
    }
    o.push_str("\n\n");
    o.push_str(&params_text(s, &e.theta, "Posterior mode: ")?);
    std::fs::write(path, o).with_context(|| format!("writing {}", path.display()))
}

// ---------------------------------------------------------------------------------------
// reader
// ---------------------------------------------------------------------------------------

/// The `//== title ==//` blocks of a file: rows of numbers up to the next blank or comment line.
pub struct Sections(Vec<(String, Vec<Vec<f64>>)>);

impl Sections {
    pub fn parse(text: &str) -> Sections {
        let lines: Vec<&str> = text.lines().collect();
        let mut out = vec![];
        let mut i = 0;
        while i < lines.len() {
            let l = lines[i].trim();
            if l.starts_with("//==") && l.ends_with("==//") && l.len() >= 8 {
                let title = l[4..l.len() - 4].trim().to_string();
                let mut rows = vec![];
                let mut k = i + 1;
                while k < lines.len() {
                    let t = lines[k].trim();
                    if t.is_empty() || t.starts_with("//") {
                        break;
                    }
                    let row: Vec<f64> = t.split_whitespace().filter_map(|x| x.parse::<f64>().ok()).collect();
                    if row.is_empty() {
                        break;
                    }
                    rows.push(row);
                    k += 1;
                }
                out.push((title, rows));
                i = k;
            } else {
                i += 1;
            }
        }
        Sections(out)
    }

    pub fn get(&self, title: &str) -> Option<&Vec<Vec<f64>>> {
        self.0.iter().find(|(t, _)| t == title).map(|(_, r)| r)
    }

    fn req(&self, title: &str) -> Result<&Vec<Vec<f64>>> {
        self.get(title).ok_or_else(|| anyhow!("MS-SBVAR file: missing block '{title}'"))
    }

    fn scalar(&self, title: &str) -> Result<f64> {
        self.req(title)?
            .first()
            .and_then(|r| r.first().copied())
            .ok_or_else(|| anyhow!("MS-SBVAR file: empty block '{title}'"))
    }

    fn mat(&self, title: &str) -> Result<Mat> {
        let rows = self.req(title)?;
        let nc = rows.first().map_or(0, |r| r.len());
        if rows.iter().any(|r| r.len() != nc) {
            bail!("MS-SBVAR file: ragged block '{title}'");
        }
        Ok(Mat::from_fn(rows.len(), nc, |i, j| rows[i][j]))
    }
}

/// A [`MsSetup`] from the model part of an `init_*.dat` / `est_final_*.out` file.
pub fn read_setup(sec: &Sections) -> Result<MsSetup> {
    let nch = sec.scalar("Number independent state variables")? as usize;
    let mut chains = vec![];
    for c in 0..nch.max(1) {
        let t = |x: &str| if nch > 0 { format!("{x} for state_variable[{}]", c + 1) } else { x.to_string() };
        let n = sec.scalar(&t("Number states"))? as usize;
        let alpha = sec.mat(&t("Prior"))?;
        if alpha.nrows() != n || alpha.ncols() != n {
            bail!("MS-SBVAR file: the Dirichlet prior of chain {} is not {n} x {n}", c + 1);
        }
        chains.push(ChainSpec { n, alpha });
    }
    let nvar = sec.scalar("Number Variables")? as usize;
    let nlags = sec.scalar("Number Lags")? as usize;
    let nstates = sec.scalar("Number States")? as usize;
    if nstates != chains.iter().map(|c| c.n).product::<usize>() {
        bail!("MS-SBVAR file: Number States does not match the chains");
    }
    let ncoef = nvar * nlags + 1;
    let n0: Vec<usize> =
        sec.req("Number of free parameters in each column of A0")?[0].iter().map(|&x| x as usize).collect();
    let np: Vec<usize> =
        sec.req("Number of free parameters in each column of Aplus")?[0].iter().map(|&x| x as usize).collect();
    let mut ui = vec![];
    let mut vi = vec![];
    let mut w = vec![];
    let mut h0var = vec![];
    let mut hpvar = vec![];
    let nonzero_w: Vec<f64> =
        sec.get("Non-zero W[j]").and_then(|r| r.first().cloned()).unwrap_or_else(|| vec![1.0; nvar]);
    for j in 0..nvar {
        let u = sec.mat(&format!("U[{}]", j + 1))?;
        let v = sec.mat(&format!("V[{}]", j + 1))?;
        if u.nrows() != nvar || u.ncols() != n0[j] || v.nrows() != ncoef || v.ncols() != np[j] {
            bail!("MS-SBVAR file: dimensions of U[{0}] / V[{0}]", j + 1);
        }
        ui.push(u);
        vi.push(v);
        w.push(if nonzero_w.get(j).copied().unwrap_or(1.0) != 0.0 {
            sec.mat(&format!("W[{}]", j + 1))?
        } else {
            Mat::zeros(ncoef, nvar)
        });
        h0var.push(sec.mat(&format!("Variance of Gaussian prior on column {} of A0", j + 1))?);
        hpvar.push(sec.mat(&format!("Variance of Gaussian prior on column {} of Aplus", j + 1))?);
    }
    let zeta_a = sec.scalar("Gamma prior on zeta - a")?;
    let zeta_b = sec.scalar("Gamma prior on zeta - b")?;
    let specification = sec
        .get("Specification (0=default  1=Sims-Zha  2=Random Walk)")
        .and_then(|r| r.first())
        .and_then(|r| r.first())
        .map_or(2, |&x| x as usize);
    let lambda_var = sec
        .get("Variance of Gaussian prior on lambda")
        .and_then(|r| r.first())
        .and_then(|r| r.first())
        .copied()
        .unwrap_or(2500.0);
    let table = |t: &str| -> Result<Vec<Vec<usize>>> {
        let rows = sec.req(t)?;
        if rows.len() != nvar || rows.iter().any(|r| r.len() != nstates) {
            bail!("MS-SBVAR file: '{t}' is not {nvar} x {nstates}");
        }
        Ok(rows.iter().map(|r| r.iter().map(|&x| x as usize).collect()).collect())
    };
    let coef_idx = table("Translation table for coefficient states")?;
    let var_idx = table("Translation table for variance states")?;
    let y = sec.mat("Data Y (nobs x nvars)")?;
    let x = sec.mat("Data X (nobs x npre)")?;
    if y.ncols() != nvar || x.ncols() != ncoef || x.nrows() != y.nrows() {
        bail!("MS-SBVAR file: dimensions of the data blocks");
    }
    let freq = sec
        .get("Dates (frequency, initial year, initial subperiod)")
        .and_then(|r| r.first())
        .filter(|r| r.len() == 3)
        .map_or((4, 1, 1), |r| (r[0] as usize, r[1] as usize, r[2] as usize));
    MsSetup::assemble(MsParts {
        y,
        x,
        restr: Restrictions { ui, vi, n0, np, cross: vec![] },
        w,
        h0var,
        hpvar,
        chains,
        coef_idx,
        var_idx,
        zeta_a,
        zeta_b,
        specification,
        lambda_var,
        nlags,
        freq,
    })
}

/// Free parameters from the `<header>A0[s]`, ... blocks.
pub fn read_regime_params(sec: &Sections, s: &MsSetup, header: &str) -> Result<Vec<f64>> {
    let mut qc = vec![];
    for c in 0..s.chains.len() {
        let t = if s.chains.len() > 1 {
            format!("{header}Base transition matrix[{}]", c + 1)
        } else {
            format!("{header}Base transition matrix")
        };
        qc.push(sec.mat(&t)?);
    }
    let (mut a0, mut ap, mut z) = (vec![], vec![], vec![]);
    for st in 0..s.nstates {
        a0.push(sec.mat(&format!("{header}A0[{}]", st + 1))?);
        ap.push(sec.mat(&format!("{header}Aplus[{}]", st + 1))?);
        let zm = sec.mat(&format!("{header}Zeta[{}]", st + 1))?;
        z.push((0..s.nvar).map(|j| zm[(j, j)]).collect::<Vec<f64>>());
    }
    Ok(s.theta_from_regimes(&a0, &ap, &z, &qc))
}

/// `init_<tag>.dat` -> model and starting values.
pub fn read_init(path: &Path) -> Result<(MsSetup, Vec<f64>)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let sec = Sections::parse(&text);
    let s = read_setup(&sec)?;
    let th = read_regime_params(&sec, &s, "Initial: ")?;
    Ok((s, th))
}

/// `est_final_<tag>.out` -> the estimate (`Posterior mode: Free parameters`).
pub fn read_est_final(path: &Path) -> Result<MsEstimate> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let sec = Sections::parse(&text);
    let s = read_setup(&sec)?;
    let theta = match sec.get("Posterior mode: Free parameters").and_then(|r| r.first()) {
        Some(v) if v.len() == s.layout.len => v.clone(),
        Some(v) => bail!("{}: {} free parameters, the model has {}", path.display(), v.len(), s.layout.len),
        None => read_regime_params(&sec, &s, "Posterior mode: ")?,
    };
    let ll = s.log_likelihood(&theta);
    let lp = s.log_prior(&theta);
    Ok(MsEstimate {
        setup: Arc::new(s),
        theta,
        log_posterior: ll + lp,
        log_likelihood: ll,
        log_prior: lp,
        iterations: 0,
    })
}

/// The estimate the `ms_*` commands use: `est_final_<file_tag>.out` next to the .mod file
/// (`set_ms_estimation_file.m`: an error when it is missing).
pub fn load_estimate(dir: &Path, file_tag: &str) -> Result<MsEstimate> {
    let p = dir.join(format!("est_final_{file_tag}.out"));
    if !p.is_file() {
        bail!("ERROR: Could not find free parameter file: est_final_{file_tag}.out");
    }
    read_est_final(&p)
}
