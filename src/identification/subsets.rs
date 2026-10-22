// SPDX-License-Identifier: GPL-3.0-or-later
//! `checks_via_subsets` — port of `+identification/checks_via_subsets.m` (rank
//! conditions on every parameter subset up to `max_dim_subsets_groups`) and of its branch of
//! `identification.display`.

use super::checks::nchoosek;
use super::la::rank;
use super::{Analysis, IdeBlock};
use crate::linalg::Mat;

/// One criterion's result: `ide_*.rank`, `ide_*.problpars` (`problpars[j-1]` = the
/// problematic subsets with `j` parameters, 0-based parameter indices) and the number of
/// columns of the Jacobian `display.m` compares the rank with (`size(Jacob, 2)`).
#[derive(Clone, Debug, Default)]
pub struct SubsetResult {
    pub rank: usize,
    pub ncols: usize,
    pub problpars: Vec<Vec<Vec<usize>>>,
}

/// The five criteria (`ide_dynamic`, `ide_reducedform`, `ide_moments`, `ide_spectrum`,
/// `ide_minimal`); `None` where the criterion is skipped.
#[derive(Clone, Debug, Default)]
pub struct Subsets {
    pub dynamic: Option<SubsetResult>,
    pub reducedform: Option<SubsetResult>,
    pub moments: Option<SubsetResult>,
    pub spectrum: Option<SubsetResult>,
    pub minimal: Option<SubsetResult>,
}

/// `dX(ind, :) = dX(ind, :) ./ norm` on the full Jacobian.
fn normalized_full(b: &IdeBlock) -> Mat {
    let mut m = b.jacobian.clone();
    for (k, &i) in b.ind.iter().enumerate() {
        let nv = b.norm[k];
        for c in 0..m.ncols() {
            m[(i, c)] /= nv;
        }
    }
    m
}

fn cols(m: &Mat, c: &[usize]) -> Mat {
    Mat::from_fn(m.nrows(), c.len(), |i, j| m[(i, c[j])])
}

/// How a criterion's subset rank is computed.
enum Kind {
    /// rank of the columns of the subset.
    Columns(Mat),
    /// rank of the principal submatrix (`dSPECTRUM` is the Gram matrix).
    Gram(Mat),
    /// rank of `[dMINIMAL_par(:, subset) dMINIMAL_rest]`, `fixed` = columns of the rest.
    Minimal { par: Mat, rest: Mat },
}

struct Crit {
    kind: Kind,
    rank: usize,
    active: bool,
    indparam: Vec<usize>,
    problpars: Vec<Vec<Vec<usize>>>,
    ncols: usize,
}

impl Crit {
    fn fixed(&self) -> usize {
        if let Kind::Minimal { rest, .. } = &self.kind { rest.ncols() } else { 0 }
    }
    fn subset_rank(&self, s: &[usize], tol: Option<f64>) -> usize {
        match &self.kind {
            Kind::Columns(m) => rank(&cols(m, s), tol),
            Kind::Gram(g) => rank(&Mat::from_fn(s.len(), s.len(), |a, b| g[(s[a], s[b])]), tol),
            Kind::Minimal { par, rest } => {
                let p = cols(par, s);
                let mut m = Mat::zeros(par.nrows(), s.len() + rest.ncols());
                m.columns_mut(0, s.len()).copy_from(&p);
                m.columns_mut(s.len(), rest.ncols()).copy_from(rest);
                rank(&m, tol)
            }
        }
    }
}

/// `RemoveProblematicParameterSets`: drops the subsets that contain a smaller problematic set.
fn remove_problematic(idx: &mut Vec<Vec<usize>>, probl: &[Vec<Vec<usize>>]) {
    let iset = idx.first().map(|r| r.len()).unwrap_or(0);
    for (iii, sets) in probl.iter().enumerate().take(iset.saturating_sub(1)) {
        for row in sets {
            idx.retain(|r| r.iter().filter(|p| row.contains(p)).count() != iii + 1);
        }
    }
}

/// Port of `identification.checks_via_subsets` on the point analysis.
pub fn checks_via_subsets(an: &Analysis, tot: usize, nm: usize, tol_rank: Option<f64>, max_dim: usize) -> Subsets {
    let mut crits: Vec<Option<Crit>> = Vec::new();
    let new = |kind: Kind, ncols: usize, full: usize| -> Crit {
        let r = match &kind {
            Kind::Columns(m) | Kind::Gram(m) => rank(m, tol_rank),
            Kind::Minimal { par, rest } => {
                let mut m = Mat::zeros(par.nrows(), par.ncols() + rest.ncols());
                m.columns_mut(0, par.ncols()).copy_from(par);
                m.columns_mut(par.ncols(), rest.ncols()).copy_from(rest);
                rank(&m, tol_rank)
            }
        };
        let active = r != full;
        Crit {
            kind,
            rank: r,
            active,
            indparam: if active { (0..tot).collect() } else { vec![] },
            problpars: vec![Vec::new(); max_dim],
            ncols,
        }
    };
    // dynamic: zeros for the stderr / corr parameters
    let d = normalized_full(&an.dynamic);
    let mut dd = Mat::zeros(d.nrows(), tot);
    dd.columns_mut(tot - nm, nm).copy_from(&d);
    crits.push(Some(new(Kind::Columns(dd), tot, tot)));
    crits.push(an.reducedform.as_ref().map(|b| new(Kind::Columns(normalized_full(b)), b.jacobian.ncols(), tot)));
    crits.push(an.moments.as_ref().map(|b| new(Kind::Columns(normalized_full(b)), b.si.ncols(), tot)));
    crits.push(an.spectrum.as_ref().map(|b| new(Kind::Gram(b.si.clone()), b.jacobian.ncols(), tot)));
    crits.push(an.minimal.as_ref().map(|b| {
        let m = normalized_full(b);
        let par = m.columns(0, tot).into_owned();
        let rest = m.columns(tot, m.ncols() - tot).into_owned();
        let fixed = rest.ncols();
        new(Kind::Minimal { par, rest }, b.jacobian.ncols(), tot + fixed)
    }));
    // single parameters
    for c in crits.iter_mut().flatten() {
        if !c.active {
            continue;
        }
        let fixed = c.fixed();
        for j in 0..tot {
            let bad = match &c.kind {
                Kind::Columns(m) => rank(&cols(m, &[j]), tol_rank) == 0,
                // abs(dSPECTRUM(j,j)) < tol_rank; with tol_rank = 'robust' MATLAB compares with
                // the character codes of 'robust' (all must hold: < 98)
                Kind::Gram(g) => g[(j, j)].abs() < tol_rank.unwrap_or(98.0),
                Kind::Minimal { .. } => c.subset_rank(&[j], tol_rank) == fixed,
            };
            if bad {
                c.problpars[0].push(vec![j]);
            }
        }
        let nbad = c.problpars[0].len();
        if nbad + c.rank == tot + fixed {
            c.active = false;
        } else {
            let bad: Vec<usize> = c.problpars[0].iter().map(|r| r[0]).collect();
            c.indparam.retain(|p| !bad.contains(p));
        }
    }
    let mut indtot: Vec<usize> = crits.iter().flatten().flat_map(|c| c.indparam.iter().copied()).collect();
    indtot.sort_unstable();
    indtot.dedup();
    for j in 2..=indtot.len().min(max_dim) {
        if !crits.iter().flatten().any(|c| c.active) {
            continue;
        }
        let indexj = nchoosek(&indtot, j);
        for c in crits.iter_mut().flatten() {
            if !c.active {
                continue;
            }
            let mut idx = indexj.clone();
            remove_problematic(&mut idx, &c.problpars);
            let fixed = c.fixed();
            let bad: Vec<Vec<usize>> = idx.into_iter().filter(|s| c.subset_rank(s, tol_rank) < j + fixed).collect();
            c.problpars[j - 1] = bad;
        }
    }
    let mut it = crits.into_iter();
    let out = |c: Option<Crit>| c.map(|c| SubsetResult { rank: c.rank, ncols: c.ncols, problpars: c.problpars });
    let dynamic = it.next().unwrap().map(|mut c| {
        // get rid of stderr and corr variables for dynamic
        c.problpars[0].retain(|r| r[0] >= tot - nm);
        c
    });
    Subsets {
        dynamic: out(dynamic),
        reducedform: out(it.next().unwrap()),
        moments: out(it.next().unwrap()),
        spectrum: out(it.next().unwrap()),
        minimal: out(it.next().unwrap()),
    }
}

/// The `checks_via_subsets` branch of `identification.display` for one criterion (point
/// analysis); `false` when the rank is full (the caller prints "All parameters are identified").
pub fn display_subsets(r: &SubsetResult, names: &[String], str_test: &str, str_jac: &str, str_meaning: &str) -> bool {
    if r.rank >= r.ncols {
        return false;
    }
    println!();
    println!("{}:", str_test.to_uppercase());
    println!("    !!!WARNING!!!");
    println!("    The rank of {str_jac} ({str_meaning}) is deficient!");
    let maxdim = r.problpars.len();
    if r.problpars.iter().all(|p| p.is_empty()) {
        println!(
            "    No problematic parameter combinations with maximum dimension {maxdim} were found. Increase max_dim_subsets_groups."
        );
        println!();
    } else {
        println!("    Displaying problematic parameter combinations (with maximum dimension {maxdim}):");
        println!();
        for sets in &r.problpars {
            for s in sets {
                let txt: Vec<&str> = s.iter().map(|&j| names[j].as_str()).collect();
                if s.len() == 1 {
                    println!("    {} is not identified!", txt.join(","));
                } else {
                    println!("    [{}] are collinear!", txt.join(","));
                }
            }
        }
    }
    true
}
