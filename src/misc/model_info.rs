// SPDX-License-Identifier: GPL-3.0-or-later
//! `M_.block_structure` / `M_.block_structure_stat` of a .mod file (the preprocessor's
//! `computingPassBlock`, see [`crate::model::blocks`]) and the block part of `model_info.m`
//! (`model_info(block_static|block_dynamic, incidence)`).

use crate::model::blocks::{self, BlockSettings, BlockStructure, EvalContext};
use crate::model::expr::Node;
use crate::modfile::{Command, ModFile};
use std::fmt::Write;

/// `ModFile::evalAllExpressions`: parameters, `initval`/`endval` values, then the auxiliary
/// variables whose definitions only use known values; everything else is 0.
pub fn eval_context(mf: &ModFile) -> EvalContext {
    let m = &mf.model;
    let (n, nx) = (m.n_endo(), m.n_exo());
    let mut y = vec![0.0; n];
    let mut ky = vec![false; n];
    for &(i, v) in mf.initval.iter().chain(&mf.endval) {
        y[i] = v;
        ky[i] = true;
    }
    let mut x = vec![0.0; nx];
    let mut kx = vec![false; nx];
    for &(i, v) in &mf.initval_exo {
        x[i] = v;
        kx[i] = true;
    }
    let unknown = &mf.block_opts.unknown_params;
    let fin = &mf.block_opts.final_params;
    let params: Vec<f64> = m.params.iter().enumerate().map(|(i, v)| fin.get(i).copied().unwrap_or(*v)).collect();
    let kp: Vec<bool> = params.iter().enumerate().map(|(i, v)| !v.is_nan() && !unknown.contains(&i)).collect();
    let p: Vec<f64> = params.iter().zip(&kp).map(|(v, k)| if *k { *v } else { 0.0 }).collect();
    let dag = m.dag();
    for eq in m.orig_eq_nbr..m.n_equations() {
        let (l, r) = m.equations()[eq];
        let Node::Endo(a, 0) = dag.node(l) else { continue };
        let mut leaves = Vec::new();
        dag.collect_leaves(r, &mut leaves);
        let known = leaves.iter().all(|lf| match *lf {
            Node::Endo(j, _) | Node::SteadyState(j, _) => ky[j as usize],
            Node::Exo(k, _) => kx[k as usize],
            Node::Param(i) => kp.get(i as usize).copied().unwrap_or(false),
            _ => true,
        });
        if known {
            let v = dag.eval(r, &|lf| match lf {
                Node::Endo(j, _) | Node::SteadyState(j, _) => y[j as usize],
                Node::Exo(k, _) => x[k as usize],
                Node::Param(i) => p[i as usize],
                _ => 0.0,
            });
            y[a as usize] = v;
            ky[a as usize] = true;
        }
    }
    EvalContext { y, x, p }
}

fn has(mf: &ModFile, f: &dyn Fn(&Command) -> bool) -> bool {
    mf.commands.iter().any(f)
}

/// `mod_file_struct.{stoch_simul,estimation,osr,ramsey_model,identification,calib_smoother,
/// mom_estimation}_present`: the commands that set the normalization cutoff to 0.
fn stochastic_tasks(mf: &ModFile) -> bool {
    has(mf, &|c| match c {
        Command::StochSimul(..) | Command::Estimation(..) | Command::CalibSmoother(..) => true,
        Command::Policy(n, ..) => matches!(n.as_str(), "osr" | "ramsey_model" | "ramsey_policy"),
        Command::Mom(crate::mom::MomCommand::MethodOfMoments(_)) => true,
        Command::Other(n, o) => {
            n == "identification"
                || (n == "dynare_sensitivity"
                    && crate::modfile::opt(o, "identification")
                        .is_some_and(|v| matches!(v, crate::modfile::OptValue::Num(x) if *x == 1.0)))
        }
        _ => false,
    })
}

fn perfect_foresight_tasks(mf: &ModFile) -> bool {
    has(mf, &|c| match c {
        Command::Deterministic(n, ..) => matches!(n.as_str(), "perfect_foresight_solver" | "simul"),
        Command::Other(n, _) => {
            matches!(n.as_str(), "perfect_foresight_with_expectation_errors_solver" | "extended_path")
        }
        _ => false,
    })
}

/// The settings of `StaticModel::computingPass` / `DynamicModel::computingPass`.
pub fn settings(mf: &ModFile, dynamic: bool) -> BlockSettings {
    let o = &mf.block_opts;
    let mut cutoff = o.cutoff.unwrap_or(1e-15);
    let stoch = stochastic_tasks(mf);
    if dynamic {
        if stoch && !perfect_foresight_tasks(mf) {
            cutoff = 0.0;
        }
        let m = &mf.model;
        let max_lag = m.lead_lag_incidence.iter().any(|r| r[0] > 0);
        let max_lead = m.lead_lag_incidence.iter().any(|r| r[2] > 0);
        let (mut mfs, mut time_recursive) = (o.mfs.unwrap_or(1), false);
        if !o.block && (!max_lag || !max_lead) {
            time_recursive = true;
            mfs = 3;
        }
        BlockSettings { dynamic, mfs, cutoff, time_recursive }
    } else {
        if stoch {
            cutoff = 0.0;
        }
        BlockSettings { dynamic, mfs: o.static_mfs.unwrap_or(0), cutoff, time_recursive: false }
    }
}

/// `M_.block_structure_stat` (`None` with `no_static`, or if the static model cannot be
/// normalized).
pub fn static_block_structure(mf: &ModFile) -> Option<BlockStructure> {
    if mf.block_opts.no_static {
        return None;
    }
    blocks::compute(&mf.model, &settings(mf, false), &eval_context(mf))
}

/// `M_.block_structure` (`None` if the dynamic model cannot be normalized).
pub fn dynamic_block_structure(mf: &ModFile) -> Option<BlockStructure> {
    blocks::compute(&mf.model, &settings(mf, true), &eval_context(mf))
}

/// `Sym_type` of `model_info.m`.
fn sym_type(t: u8) -> &'static str {
    match t {
        1 | 9 => "EVALUATE FORWARD            ",
        2 | 10 => "EVALUATE BACKWARD            ",
        3 => "SOLVE FORWARD SIMPLE        ",
        4 => "SOLVE BACKWARD SIMPLE        ",
        5 => "SOLVE TWO BOUNDARIES SIMPLE  ",
        6 => "SOLVE FORWARD COMPLETE      ",
        7 => "SOLVE BACKWARD COMPLETE      ",
        8 => "SOLVE TWO BOUNDARIES COMPLETE",
        _ => "UNKNOWN                     ",
    }
}

/// MATLAB `int2str` of a column vector: right-justified, width of the widest entry.
fn int2str_col(v: &[usize]) -> Vec<String> {
    let w = v.iter().map(|x| x.to_string().len()).max().unwrap_or(1);
    v.iter().map(|x| format!("{x:>w$}")).collect()
}

/// `model_info.m` with `block_static` or `block_dynamic` (and `incidence`), after the
/// "Information about ..." header line `header` (as printed, newlines included).
pub fn block_text(mf: &ModFile, bs: &BlockStructure, statik: bool, incidence: bool, header_len: usize) -> String {
    let m = &mf.model;
    let n = m.n_endo();
    let names = super::equation_names(mf);
    let mut out = String::new();
    let _ = writeln!(out, "{}", "=".repeat(header_len));
    let _ =
        writeln!(out, "The model has {} equations and is decomposed into {} blocks as follows:", n, bs.blocks.len());
    let bar = "================================================================================================================================";
    let _ = writeln!(out, "{bar}");
    let _ = writeln!(
        out,
        "| {:>10} | {:>10} | {:>30} | {:>31} | {:>31} |",
        "Block no", "Size", "Block Type", "Equation", "Dependent variable"
    );
    let _ = writeln!(
        out,
        "|============|============|================================|=================================|=================================|"
    );
    for (i, b) in bs.blocks.iter().enumerate() {
        if i > 0 {
            let _ = writeln!(
                out,
                "|------------|------------|--------------------------------|---------------------------------|---------------------------------|"
            );
        }
        for (j, (&e, &v)) in b.equations.iter().zip(&b.variables).enumerate() {
            let ename = names.get(e).cloned().flatten().unwrap_or_default();
            let vname = &m.endo_names[v];
            let (c1, c2, c3) = if j == 0 {
                ((i + 1).to_string(), b.equations.len().to_string(), sym_type(b.simulation_type))
            } else {
                (String::new(), String::new(), "")
            };
            let _ = writeln!(
                out,
                "| {c1:>10} | {c2:>10} | {c3:>30} | {:<6} {ename:>24} | {:<6} {vname:>24} |",
                e + 1,
                v + 1
            );
        }
    }
    let _ = writeln!(out, "{bar}");
    let _ = writeln!(out);
    let print_im = |out: &mut String, im: &[(usize, usize)]| {
        let mut im = im.to_vec();
        im.sort_by_key(|&(_, v)| v); // sortrows(IM, 2) (stable)
        let mut last = usize::MAX;
        for &(e, v) in &im {
            if last != v {
                let _ = write!(out, "\n{:<30}", m.endo_names[v]);
            }
            let _ = write!(out, " {:>5}", e + 1);
            last = v;
        }
        out.push_str("\n\n");
    };
    let lag_of = |l: i32| bs.incidence.iter().find(|(k, _)| *k == l).map(|(_, s)| s.as_slice()).unwrap_or(&[]);
    if statik {
        let _ = write!(out, "{:<30} is used contemporaneously in the following equations:", "The variable");
        print_im(&mut out, lag_of(0));
    } else {
        let max_lag = m.lead_lag_incidence.iter().any(|r| r[0] > 0) as i32;
        let max_lead = m.lead_lag_incidence.iter().any(|r| r[2] > 0) as i32;
        for k in 1..=max_lag + max_lead + 1 {
            if k == max_lag + 1 {
                let _ = write!(out, "{:<30} is used in the following equations contemporaneously", "The variable");
            } else if k < max_lag + 1 {
                let _ = write!(
                    out,
                    "{:<30} is used in the following equations with lag  {}",
                    "The variable",
                    max_lag + 1 - k
                );
            } else {
                let _ = write!(out, "{:<30} is used in equations with lead  {}", "The variable", k - (max_lag + 1));
            }
            print_im(&mut out, lag_of(k - max_lag - 1));
        }
    }
    if incidence {
        incidence_text(mf, bs, statik, &mut out);
    }
    out
}

fn incidence_text(mf: &ModFile, bs: &BlockStructure, statik: bool, out: &mut String) {
    let m = &mf.model;
    let n = m.n_endo();
    // set_state_space: state_var = [pred_var; both_var]
    let state: Vec<bool> = m.lead_lag_incidence.iter().map(|r| r[0] > 0).collect();
    let nb_leadlag = if statik { 1 } else { bs.incidence.len() };
    let width = 3 * (n - 1) + 1;
    let entries: Vec<(usize, usize)> =
        bs.incidence[..nb_leadlag.min(bs.incidence.len())].iter().flat_map(|(_, s)| s.iter().copied()).collect();
    let mark = |v: usize| if state[v] { b'X' } else { b'1' };
    let l = m.endo_names.iter().map(String::len).max().unwrap_or(0);
    let lead = if n > 1 { (n as f64).log10().ceil() as usize } else { 0 };
    // topp: L rows; column 3k (1-based) holds name k vertically, preceded by `sep(k)` rows
    let topp = |sep: &dyn Fn(usize) -> u8, order: &dyn Fn(usize) -> usize| -> Vec<String> {
        (0..l)
            .map(|r| {
                let mut row = vec![b' '; lead + 3 * n];
                for k in 0..n {
                    row[lead + 3 * k] = sep(k);
                    let name = m.endo_names[order(k)].as_bytes();
                    row[lead + 3 * k + 2] = *name.get(r).unwrap_or(&b' ');
                }
                if sep(0) != b' ' {
                    row[lead] = b' ';
                }
                String::from_utf8(row).unwrap()
            })
            .collect()
    };
    let disp = |out: &mut String, rows: &[String]| {
        for r in rows {
            out.push_str(r);
            out.push('\n');
        }
    };
    // gross incidence matrix
    let mut im = vec![vec![b' '; width]; n];
    for &(e, v) in &entries {
        im[e][3 * v] = mark(v);
    }
    let seq: Vec<usize> = (1..=n).collect();
    let bott = |labels: &[usize], im: &[Vec<u8>]| -> Vec<String> {
        int2str_col(labels)
            .into_iter()
            .zip(im)
            .map(|(s, row)| format!("{s}  {}", String::from_utf8(row.clone()).unwrap()))
            .collect()
    };
    out.push_str("\nGross incidence matrix\n=======================\n");
    disp(out, &topp(&|_| b' ', &|k| k));
    out.push('\n');
    disp(out, &bott(&seq, &im));
    // reordered incidence matrix
    let mut eqpos = vec![0; n];
    let mut vapos = vec![0; n];
    for i in 0..n {
        eqpos[bs.equation_reordered[i]] = i;
        vapos[bs.variable_reordered[i]] = i;
    }
    let blk_of: Vec<usize> = bs.variable_reordered.iter().map(|&v| bs.block_of_variable(v).unwrap_or(0)).collect();
    let mut im = vec![vec![b' '; width]; n];
    for &(e, v) in &entries {
        im[eqpos[e]][3 * vapos[v]] = mark(v);
    }
    let _ = write!(out, "\n1: non-null element, X: non-null element related to a state variable\n");
    for i in 1..n {
        if blk_of[i] != blk_of[i - 1] {
            for row in im.iter_mut().take(i) {
                row[3 * i - 2] = b'|';
            }
        }
    }
    let sep = |k: usize| if k > 0 && blk_of[k] != blk_of[k - 1] { b'|' } else { b' ' };
    out.push_str("\nReordered incidence matrix\n==========================\n");
    disp(out, &topp(&sep, &|k| bs.variable_reordered[k]));
    out.push('\n');
    let labels: Vec<usize> = bs.equation_reordered.iter().map(|e| e + 1).collect();
    disp(out, &bott(&labels, &im));
    out.push_str("\n1: non-null element, X: non-null element related to a state variable\n");
}
