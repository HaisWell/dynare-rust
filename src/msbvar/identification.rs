// SPDX-License-Identifier: GPL-3.0-or-later
//! `svar_identification` block: the restrictions on `A0` / `A+` that the preprocessor turns
//! into `options_.ms.{upper_cholesky,lower_cholesky,constants_exclusion,Qi,Ri,nlags}` and that
//! `exclusions.m` / `upper_cholesky.m` / `lower_cholesky.m` turn into the bases `Ui`, `Vi`.
//!
//! Rows of `Qi{eq}` / `Ri{eq}` follow the preprocessor (`ParsingDriver::combine_lag_and_restriction`,
//! `add_restriction_element`): one counter per equation and matrix shared by exclusions and
//! `restriction equation` lines, in the order of the block (an exclusion group is closed when
//! the next statement that is not an `equation` line arrives); `exclusion lag L` is an error
//! when an earlier row already has an element at lag `L`; a restriction line must be
//! homogeneous and touch either `Qi` (lag 0) or `Ri` (lags > 0). `Ui = null(Qi)`,
//! `Vi = null(Ri)` (orthonormal bases; the results are basis independent).

use anyhow::{Result, anyhow, bail};

use crate::linalg::Mat;

/// One `exclusion lag L; equation E, V1, V2;` entry (`equation` lines after the most recent
/// `exclusion lag`). `variables` are 1-based endogenous-declaration indices (the preprocessor
/// uses the type-specific id, not the `varobs` position).
#[derive(Clone, Debug, PartialEq)]
pub struct Exclusion {
    pub lag: usize,
    pub equation: usize,
    pub variables: Vec<usize>,
}

/// One row of `options_.ms.Qi{equation}` (all entries at lag 0) or `Ri{equation}` (all lagged):
/// `sum value * coeff(variable, lag) = 0`. Exclusions give one row per excluded variable with
/// value 1; `restriction equation N, LHS = RHS;` gives one row (right-hand side negated, as the
/// preprocessor does). A row mixing lag 0 and lags > 0 is a cross `A0`-`A+` restriction
/// (rejected by the preprocessor; only accepted together with `cross_restrictions`).
#[derive(Clone, Debug, PartialEq)]
pub struct RestrictionRow {
    pub equation: usize,
    /// `(variable (1-based declaration index), lag, value)`.
    pub entries: Vec<(usize, usize, f64)>,
    /// `true` for a `restriction equation` line, `false` for an exclusion.
    pub from_line: bool,
}

impl RestrictionRow {
    pub fn is_cross(&self) -> bool {
        self.entries.iter().any(|e| e.1 == 0) && self.entries.iter().any(|e| e.1 > 0)
    }
    pub fn is_a0(&self) -> bool {
        self.entries.iter().all(|e| e.1 == 0)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SvarIdentification {
    pub upper_cholesky: bool,
    pub lower_cholesky: bool,
    /// `exclusion constants;` (`options_.ms.constants_exclusion`).
    pub constants_exclusion: bool,
    pub exclusions: Vec<Exclusion>,
    /// Every row in the preprocessor's order.
    pub rows: Vec<RestrictionRow>,
}

/// Which of the three restriction functions of `ms-sbvar/identification` applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestrictionKind {
    Exclusions,
    UpperCholesky,
    LowerCholesky,
}

/// Cross `A0`-`A+` restrictions of one equation: `R a+_j + Q a_j = 0`.
#[derive(Clone, Debug)]
pub struct CrossRestriction {
    /// 0-based equation.
    pub equation: usize,
    /// `m x ncoef`.
    pub r: Mat,
    /// `m x nvar`.
    pub q: Mat,
}

/// The linear restrictions `a_j = U_j b_j`, `a+_j = V_j g_j` of every equation.
#[derive(Clone, Debug)]
pub struct Restrictions {
    pub ui: Vec<Mat>,
    pub vi: Vec<Mat>,
    pub n0: Vec<usize>,
    pub np: Vec<usize>,
    /// Cross restrictions (mixed-lag restriction lines, `cross_restrictions` only).
    pub cross: Vec<CrossRestriction>,
}

/// Orthonormal basis of the null space of `a` (`null(a)` of MATLAB up to an orthogonal change
/// of basis); unit vectors when every row has a single non-zero entry.
pub fn null_basis(a: &Mat) -> Mat {
    let n = a.ncols();
    let single = (0..a.nrows()).all(|i| a.row(i).iter().filter(|v| **v != 0.0).count() <= 1);
    if single {
        let free: Vec<usize> = (0..n).filter(|&j| a.column(j).iter().all(|v| *v == 0.0)).collect();
        return Mat::from_fn(n, free.len(), |r, c| if r == free[c] { 1.0 } else { 0.0 });
    }
    // pad to at least n x n so that the SVD returns the whole right singular basis
    let m = a.nrows().max(n);
    let mut p = Mat::zeros(m, n);
    p.rows_mut(0, a.nrows()).copy_from(a);
    let svd = p.svd(false, true);
    let vt = svd.v_t.expect("svd: V");
    let smax = svd.singular_values.iter().fold(0.0f64, |x, y| x.max(*y));
    let tol = m as f64 * f64::EPSILON * smax;
    let free: Vec<usize> = (0..n).filter(|&i| svd.singular_values[i] <= tol).collect();
    Mat::from_fn(n, free.len(), |r, c| vt[(free[c], r)])
}

/// Parses one side of a restriction line: `[+|-][EXPR*]coeff(VAR,LAG) ...` or a constant
/// expression that must be zero. Returns `(variable, lag, value)` with values multiplied by `sign`.
fn parse_side(
    side: &str,
    sign: f64,
    endo_names: &[String],
    params: &dyn Fn(&str) -> Option<f64>,
) -> Result<Vec<(usize, usize, f64)>> {
    let eval = |e: &str| -> Result<f64> {
        let ast = crate::modfile::parse_expression(e)?;
        crate::modfile::eval_ast(&ast, &|n, _, _| {
            params(n).ok_or_else(|| anyhow!("svar_identification: {n} is not a parameter with a value"))
        })
    };
    let s: String = side.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = Vec::new();
    let mut rest = s.as_str();
    while let Some(p) = rest.find("coeff(") {
        let prefix = &rest[..p];
        let close = rest[p..].find(')').ok_or_else(|| anyhow!("svar_identification: unbalanced coeff("))? + p;
        let inner: Vec<&str> = rest[p + 6..close].split(',').collect();
        rest = &rest[close + 1..];
        let [var, lag] = inner[..] else { bail!("svar_identification: coeff(VARIABLE, LAG) expected") };
        let v = endo_names
            .iter()
            .position(|n| n == var)
            .ok_or_else(|| anyhow!("svar_identification: unknown endogenous variable {var}"))?;
        let lag: usize = lag.parse().map_err(|_| {
            anyhow!("svar_identification: the lag of coeff({var}, {lag}) must be a non-negative integer")
        })?;
        let (sg, body) = match prefix.chars().next() {
            Some('+') => (1.0, &prefix[1..]),
            Some('-') => (-1.0, &prefix[1..]),
            _ if out.is_empty() => (1.0, prefix),
            _ => bail!("svar_identification: syntax error in `{side}`"),
        };
        let val = if body.is_empty() {
            1.0
        } else {
            let e = body.strip_suffix('*').ok_or_else(|| anyhow!("svar_identification: syntax error in `{side}`"))?;
            eval(e)?
        };
        out.push((v + 1, lag, sign * sg * val));
    }
    if out.is_empty() {
        if eval(&s)? != 0.0 {
            bail!("SVAR_INDENTIFICATION restrictions must be homogenous");
        }
    } else if !rest.is_empty() {
        if eval(rest).is_ok() {
            bail!("SVAR_INDENTIFICATION restrictions must be homogenous");
        }
        bail!("svar_identification: syntax error in `{side}`");
    }
    Ok(out)
}

impl SvarIdentification {
    /// Parses the statements of the block (one string per `;`-terminated statement, tokens
    /// joined by blanks, as the parser stores them under `BLOCK_BODY_KEY`). Parameters in the
    /// coefficients of restriction lines take their calibrated values (`params`).
    pub fn parse(stmts: &[String], endo_names: &[String]) -> Result<Self> {
        Self::parse_with_params(stmts, endo_names, &|_| None)
    }

    pub fn parse_with_params(
        stmts: &[String],
        endo_names: &[String],
        params: &dyn Fn(&str) -> Option<f64>,
    ) -> Result<Self> {
        let mut s = SvarIdentification::default();
        let mut lag: Option<usize> = None;
        let mut group: Vec<Exclusion> = Vec::new();
        // combine_lag_and_restriction
        let close = |s: &mut SvarIdentification, lag: Option<usize>, group: &mut Vec<Exclusion>| -> Result<()> {
            let Some(l) = lag else { return Ok(()) };
            if s.rows.iter().any(|r| r.entries.iter().any(|e| e.1 == l)) {
                bail!("svar_identification: lag {l} used more than once.");
            }
            // the preprocessor keeps the group in a map ordered by equation
            group.sort_by_key(|e| e.equation);
            for e in group.drain(..) {
                for &v in &e.variables {
                    s.rows.push(RestrictionRow { equation: e.equation, entries: vec![(v, l, 1.0)], from_line: false });
                }
                s.exclusions.push(e);
            }
            Ok(())
        };
        for st in stmts {
            let toks: Vec<String> = st.replace(',', " , ").split_whitespace().map(str::to_string).collect();
            let kw: Vec<String> = toks.iter().map(|t| t.to_lowercase()).collect();
            if kw.first().map(String::as_str) != Some("equation") {
                close(&mut s, lag, &mut group)?;
                lag = None;
            }
            match kw.first().map(String::as_str) {
                Some("upper_cholesky") => s.upper_cholesky = true,
                Some("lower_cholesky") => s.lower_cholesky = true,
                Some("exclusion") => match kw.get(1).map(String::as_str) {
                    Some("constants") => s.constants_exclusion = true,
                    Some("lag") => {
                        lag = Some(
                            kw.get(2)
                                .and_then(|t| t.parse().ok())
                                .ok_or_else(|| anyhow!("svar_identification: `exclusion lag` needs an integer"))?,
                        )
                    }
                    _ => bail!("svar_identification: unexpected `{st}`"),
                },
                Some("equation") => {
                    let Some(l) = lag else { bail!("svar_identification: `equation` before any `exclusion lag`") };
                    let eq: usize = kw
                        .get(1)
                        .and_then(|t| t.parse().ok())
                        .ok_or_else(|| anyhow!("svar_identification: `equation` needs an integer"))?;
                    let mut vars = Vec::new();
                    for t in toks.iter().skip(2).filter(|t| *t != ",") {
                        match endo_names.iter().position(|n| n == t) {
                            Some(i) => vars.push(i + 1),
                            None => bail!("svar_identification: unknown endogenous variable {t}"),
                        }
                    }
                    group.push(Exclusion { lag: l, equation: eq, variables: vars });
                }
                Some("restriction") => {
                    if kw.get(1).map(String::as_str) != Some("equation") {
                        bail!("svar_identification: unexpected `{st}`");
                    }
                    let eq: usize = kw
                        .get(2)
                        .and_then(|t| t.parse().ok())
                        .ok_or_else(|| anyhow!("svar_identification: `restriction equation` needs an integer"))?;
                    let body = st
                        .split_once(',')
                        .map(|x| x.1)
                        .ok_or_else(|| anyhow!("svar_identification: `restriction equation {eq},` expected"))?;
                    let sides: Vec<&str> = body.split('=').collect();
                    if sides.len() != 2 {
                        bail!("svar_identification: a restriction needs exactly one `=`: `{st}`");
                    }
                    let mut entries = parse_side(sides[0], 1.0, endo_names, params)?;
                    entries.extend(parse_side(sides[1], -1.0, endo_names, params)?);
                    if entries.is_empty() {
                        bail!("svar_identification: restriction without coefficient: `{st}`");
                    }
                    s.rows.push(RestrictionRow { equation: eq, entries, from_line: true });
                }
                _ => bail!("svar_identification: unexpected `{st}`"),
            }
        }
        close(&mut s, lag, &mut group)?;
        if s.upper_cholesky && s.lower_cholesky {
            bail!(
                "Within the svar_identification statement, you may only have one of upper_cholesky and lower_cholesky"
            );
        }
        if (s.upper_cholesky || s.lower_cholesky) && !s.rows.is_empty() {
            bail!(
                "Within the svar_identification statement, you may only have one of the Cholesky decompositions or exclusion/restriction lines"
            );
        }
        Ok(s)
    }

    /// `options_.ms.nlags = ...` as emitted by the preprocessor: the largest lag of any
    /// exclusion or restriction (`None` when there is none).
    pub fn max_lag(&self) -> Option<usize> {
        self.rows.iter().flat_map(|r| r.entries.iter().map(|e| e.1)).max()
    }

    /// Restriction lines mixing lag 0 and lags > 0.
    pub fn has_cross(&self) -> bool {
        self.rows.iter().any(RestrictionRow::is_cross)
    }

    /// `options_.ms.Qi{eq}` / `Ri{eq}` as `(row, column)` pairs of the entries the preprocessor
    /// assigns: lag-0 rows go to `Qi` at column `variable`, lagged ones to `Ri` at column
    /// `(lag-1)*neq + variable`; rows count the rows of the equation (cross rows are left out).
    pub fn qi_ri(&self, neq: usize) -> (Vec<Vec<(usize, usize)>>, Vec<Vec<(usize, usize)>>) {
        let (q, r) = self.qi_ri_values(neq);
        let strip = |v: Vec<Vec<(usize, usize, f64)>>| {
            v.into_iter().map(|e| e.into_iter().map(|(a, b, _)| (a, b)).collect()).collect()
        };
        (strip(q), strip(r))
    }

    /// As [`Self::qi_ri`] with the values: `(row, column, value)`.
    #[allow(clippy::type_complexity)]
    pub fn qi_ri_values(&self, neq: usize) -> (Vec<Vec<(usize, usize, f64)>>, Vec<Vec<(usize, usize, f64)>>) {
        let (mut qi, mut ri) = (vec![Vec::new(); neq], vec![Vec::new(); neq]);
        let (mut nq, mut nr) = (vec![0usize; neq], vec![0usize; neq]);
        for row in &self.rows {
            let e = row.equation;
            if e == 0 || e > neq || row.is_cross() {
                continue;
            }
            if row.is_a0() {
                nq[e - 1] += 1;
                for &(v, _, x) in &row.entries {
                    qi[e - 1].push((nq[e - 1], v, x));
                }
            } else {
                nr[e - 1] += 1;
                for &(v, l, x) in &row.entries {
                    ri[e - 1].push((nr[e - 1], (l - 1) * neq + v, x));
                }
            }
        }
        (qi, ri)
    }

    /// The restriction function `sbvar.m` selects.
    pub fn kind(&self) -> Option<RestrictionKind> {
        if self.upper_cholesky {
            Some(RestrictionKind::UpperCholesky)
        } else if self.lower_cholesky {
            Some(RestrictionKind::LowerCholesky)
        } else if !self.rows.is_empty() {
            Some(RestrictionKind::Exclusions)
        } else {
            None
        }
    }

    /// `[Ui, Vi, n0, np] = exclusions / upper_cholesky / lower_cholesky (nvar, nexo, ms)`, plus
    /// the cross restrictions when `cross` is set (otherwise a mixed-lag row is the
    /// preprocessor's error).
    pub fn restrictions_for(
        &self,
        kind: RestrictionKind,
        nvar: usize,
        nlags: usize,
        nexo: usize,
    ) -> Result<Restrictions> {
        self.restrictions_for_cross(kind, nvar, nlags, nexo, false)
    }

    pub fn restrictions_for_cross(
        &self,
        kind: RestrictionKind,
        nvar: usize,
        nlags: usize,
        nexo: usize,
        cross: bool,
    ) -> Result<Restrictions> {
        let ncoef = nlags * nvar + nexo;
        for row in &self.rows {
            if row.equation == 0 || row.equation > nvar {
                bail!("svar_identification: equation {} out of range", row.equation);
            }
            for &(v, l, _) in &row.entries {
                if v == 0 || v > nvar {
                    bail!("svar_identification: variable index {v} out of range");
                }
                if l > nlags {
                    bail!("svar_identification: lag {l} exceeds nlags = {nlags}");
                }
            }
            if row.is_cross() && !cross {
                bail!(
                    "SVAR_IDENTIFICATION: a single restrictions must affect either Qi or Ri, but not both (cross A0-A+ restrictions need the cross_restrictions option)"
                );
            }
        }
        // exclusions.m: Qi{n} padded into nvar x nvar, Ri{n} into k x k, constants in row sR(1)+1
        let (mut qm, mut rm): (Vec<Mat>, Vec<Mat>) =
            (vec![Mat::zeros(nvar, nvar); nvar], vec![Mat::zeros(ncoef, ncoef); nvar]);
        match kind {
            RestrictionKind::UpperCholesky => (0..nvar).for_each(|n| (0..n).for_each(|i| qm[n][(i, i)] = 1.0)),
            RestrictionKind::LowerCholesky => (0..nvar).for_each(|n| (n + 1..nvar).for_each(|i| qm[n][(i, i)] = 1.0)),
            RestrictionKind::Exclusions => {
                let (qv, rv) = self.qi_ri_values(nvar);
                for n in 0..nvar {
                    let nq = qv[n].iter().map(|e| e.0).max().unwrap_or(0);
                    if nq > nvar {
                        qm[n] = qm[n].clone().resize(nq, nvar, 0.0);
                    }
                    for &(r, c, x) in &qv[n] {
                        qm[n][(r - 1, c - 1)] = x;
                    }
                    let nr = rv[n].iter().map(|e| e.0).max().unwrap_or(0);
                    if nr + 1 > ncoef {
                        rm[n] = rm[n].clone().resize(nr + 1, ncoef, 0.0);
                    }
                    for &(r, c, x) in &rv[n] {
                        rm[n][(r - 1, c - 1)] = x;
                    }
                }
            }
        }
        if self.constants_exclusion {
            // `Ri{n}(sR(1)+1, k) = 1`: after the rows of Ri{n} (Cholesky: Ri is empty)
            let (_, rv) = self.qi_ri_values(nvar);
            for n in 0..nvar {
                let nr =
                    if kind == RestrictionKind::Exclusions { rv[n].iter().map(|e| e.0).max().unwrap_or(0) } else { 0 };
                if nr + 1 > rm[n].nrows() {
                    rm[n] = rm[n].clone().resize(nr + 1, ncoef, 0.0);
                }
                rm[n][(nr, ncoef - 1)] = 1.0;
            }
        }
        let ui: Vec<Mat> = qm.iter().map(null_basis).collect();
        let vi: Vec<Mat> = rm.iter().map(null_basis).collect();
        let n0: Vec<usize> = ui.iter().map(|m| m.ncols()).collect();
        let np: Vec<usize> = vi.iter().map(|m| m.ncols()).collect();
        if n0.contains(&0) {
            bail!("A0: restrictions give no free parameter in one of the equations");
        }
        if np.contains(&0) {
            bail!("A+: restrictions give no free parameter in one of the equations");
        }
        let mut cr: Vec<CrossRestriction> = Vec::new();
        for row in self.rows.iter().filter(|r| r.is_cross()) {
            let j = row.equation - 1;
            let (mut r, mut q) = (Mat::zeros(1, ncoef), Mat::zeros(1, nvar));
            for &(v, l, x) in &row.entries {
                if l == 0 {
                    q[(0, v - 1)] = x;
                } else {
                    r[(0, (l - 1) * nvar + v - 1)] = x;
                }
            }
            match cr.iter_mut().find(|c| c.equation == j) {
                Some(c) => {
                    let m = c.r.nrows();
                    c.r = c.r.clone().resize_vertically(m + 1, 0.0);
                    c.q = c.q.clone().resize_vertically(m + 1, 0.0);
                    c.r.row_mut(m).copy_from(&r.row(0));
                    c.q.row_mut(m).copy_from(&q.row(0));
                }
                None => cr.push(CrossRestriction { equation: j, r, q }),
            }
        }
        Ok(Restrictions { ui, vi, n0, np, cross: cr })
    }
}
