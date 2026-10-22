// SPDX-License-Identifier: GPL-3.0-or-later
//! port of `model_diagnostics.m` (non-block, non-bytecode, no Ramsey instruments):
//! variables absent at the current period, uncalibrated parameters, the steady state, Inf/NaN
//! in the static and dynamic Jacobians (`display_problematic_vars_Jacobian.m`), rank of the
//! static, dynamic and contemporaneous dynamic Jacobians with the collinear variables and
//! equations (MATLAB's `rank` and `null(A, "rational")` = the `rref` basis), the unit-root
//! hint, Inf/NaN in the dynamic Hessian (`options_.order >= 2`).

use super::{SteadyStateModel, evaluate_steady_state, fmt_g};
use crate::linalg::Mat;
use crate::model::Model;
use crate::solver::SteadyOptions;

/// MATLAB `rank(A)`: singular values above `max(size(A)) * eps(max(s))`.
pub fn matlab_rank(a: &Mat) -> usize {
    if a.nrows() == 0 || a.ncols() == 0 {
        return 0;
    }
    let s = crate::linalg::singular_values(a);
    let smax = s.iter().cloned().fold(0.0, f64::max);
    let tol = a.nrows().max(a.ncols()) as f64 * eps_of(smax);
    s.iter().filter(|&&v| v > tol).count()
}

fn eps_of(x: f64) -> f64 {
    if x == 0.0 { f64::MIN_POSITIVE * f64::EPSILON } else { 2f64.powi(x.abs().log2().floor() as i32 - 52) }
}

/// MATLAB `rref(A)`: (R, pivot columns).
pub fn rref(a: &Mat) -> (Mat, Vec<usize>) {
    let (m, n) = (a.nrows(), a.ncols());
    let mut r = a.clone();
    let norm_inf = (0..m).map(|i| (0..n).map(|j| a[(i, j)].abs()).sum::<f64>()).fold(0.0, f64::max);
    let tol = m.max(n) as f64 * f64::EPSILON * norm_inf;
    let (mut i, mut j) = (0, 0);
    let mut jb = Vec::new();
    while i < m && j < n {
        let (mut p, mut k) = (-1.0, i);
        for q in i..m {
            if r[(q, j)].abs() > p {
                p = r[(q, j)].abs();
                k = q;
            }
        }
        if p <= tol {
            for q in i..m {
                r[(q, j)] = 0.0;
            }
            j += 1;
        } else {
            jb.push(j);
            for c in j..n {
                let t = r[(i, c)];
                r[(i, c)] = r[(k, c)];
                r[(k, c)] = t;
            }
            let piv = r[(i, j)];
            for c in j..n {
                r[(i, c)] /= piv;
            }
            for q in (0..m).filter(|&q| q != i) {
                let f = r[(q, j)];
                if f != 0.0 {
                    for c in j..n {
                        r[(q, c)] -= f * r[(i, c)];
                    }
                }
            }
            i += 1;
            j += 1;
        }
    }
    (r, jb)
}

/// `null(A, "rational")`: one column per non-pivot column of `rref(A)`.
pub fn null_rational(a: &Mat) -> Vec<Vec<f64>> {
    let n = a.ncols();
    let (r, piv) = rref(a);
    let nopiv: Vec<usize> = (0..n).filter(|j| !piv.contains(j)).collect();
    nopiv
        .iter()
        .map(|&f| {
            let mut z = vec![0.0; n];
            z[f] = 1.0;
            for (row, &pc) in piv.iter().enumerate() {
                z[pc] = -r[(row, f)];
            }
            z
        })
        .collect()
}

/// The variables (columns) of relation `z` of `a`: `find(abs(z) > 10^-j)` for the first `j`
/// with `max(abs(a(:, k) * z(k))) < 1e-6`.
fn relation_support(a: &Mat, z: &[f64]) -> Vec<usize> {
    let mut k = Vec::new();
    for j in 1..=10 {
        k = (0..z.len()).filter(|&q| z[q].abs() > 10f64.powi(-j)).collect();
        let worst = (0..a.nrows()).map(|r| k.iter().map(|&q| a[(r, q)] * z[q]).sum::<f64>().abs()).fold(0.0, f64::max);
        if worst < 1e-6 {
            break;
        }
    }
    k
}

fn eq_label(model: &Model, eq: usize) -> String {
    match model.equation_tags.get(eq).and_then(|t| t.iter().find(|(k, _)| k == "name")) {
        Some((_, v)) => format!("    Equation {}: {v}", eq + 1),
        None => format!("    Equation {}", eq + 1),
    }
}

fn print_vars(names: &[&str]) {
    if names.len() > 5 {
        for chunk in names.chunks(10) {
            println!("    {}", chunk.join(", "));
        }
    } else {
        for v in names {
            println!("    {v}");
        }
    }
}

fn collinear_vars(model: &Model, a: &Mat, title: &str) {
    let ncol = null_rational(a);
    let n_rel = ncol.len();
    for (i, z) in ncol.iter().enumerate() {
        println!();
        println!("--- {title}: Collinear variables (relation {} of {n_rel}) ---", i + 1);
        let k = relation_support(a, z);
        let names: Vec<&str> = k.iter().map(|&q| model.endo_names[q].as_str()).collect();
        print_vars(&names);
    }
}

fn collinear_eqs(model: &Model, a: &Mat, title: &str) {
    let at = a.transpose();
    let neq = null_rational(&at);
    let n_rel = neq.len();
    for (i, z) in neq.iter().enumerate() {
        println!();
        println!("--- {title}: Collinear equations (relation {} of {n_rel}) ---", i + 1);
        for q in relation_support(&at, z) {
            println!("{}", eq_label(model, q));
        }
    }
}

/// `display_problematic_vars_Jacobian.m` for the static (`dynamic = false`) or dynamic model.
fn problematic(model: &Model, cells: &[(usize, usize)], x: &[f64], dynamic: bool) {
    println!();
    let n = model.n_endo();
    let orig = model.orig_endo_nbr;
    for &(row, col) in cells {
        let eq = row + 1;
        let (endo, var, ty) = if dynamic {
            if col < 3 * n { (true, col % n, ["lag of", "", "lead of"][col / n]) } else { (false, col - 3 * n, "") }
        } else {
            (true, col, "")
        };
        if endo {
            let v = if var < orig {
                var
            } else {
                model.aux_vars.iter().find(|a| a.endo_index == var).and_then(|a| a.orig_index).unwrap_or(var)
            };
            let xv = if dynamic || var < orig { x[v] } else { x[var] };
            println!(
                "Derivative of Equation {eq} with respect to {ty} Variable {}  (initial value of {}: {}) ",
                model.endo_names[v],
                model.endo_names[v],
                fmt_g(xv)
            );
        } else {
            println!("Derivative of Equation {eq} with respect to {ty} shock {} ", model.exo_names[var]);
        }
    }
    let c = "MODEL_DIAGNOSTICS: ";
    println!("\n{c}  The problem most often occurs, because a variable with");
    println!(
        "{c}  exponent smaller than {} has been initialized to 0. Taking the derivative",
        if dynamic { 0 } else { 1 }
    );
    println!("{c}  and evaluating it at the steady state then results in a division by 0.");
    println!("{c}  If you are using model-local variables (# operator), check their values as well.");
}

/// Port of `model_diagnostics.m`.
#[allow(clippy::too_many_arguments)]
pub fn model_diagnostics(
    model: &Model,
    ssm: Option<&SteadyStateModel>,
    ys0: &[f64],
    xs: &[f64],
    params: &[f64],
    steady_opts: &SteadyOptions,
    nocheck: bool,
    order: usize,
) {
    let mut problem = false;
    let n = model.n_endo();
    let absent: Vec<&str> =
        (0..n).filter(|&j| model.lead_lag_incidence[j][1] == 0).map(|j| model.endo_names[j].as_str()).collect();
    if !absent.is_empty() {
        problem = true;
        println!(
            "MODEL_DIAGNOSTICS: The following endogenous variables aren't present at the current period in the model:"
        );
        for v in absent {
            println!("{v}");
        }
    }
    let nan: Vec<&str> =
        (0..params.len()).filter(|&k| params[k].is_nan()).map(|k| model.param_names[k].as_str()).collect();
    if !nan.is_empty() {
        problem = true;
        println!(
            "Warning: Some of the parameters have no value ({}) when using model_diagnostics. If these parameters are not initialized in a steadystate file or a steady_state_model-block, Dynare may not be able to solve the model. Note that simul, perfect_foresight_setup, and perfect_foresight_solver do not automatically call the steady state file.",
            nan.join(", ")
        );
    }
    // options_.debug is set locally: dynare_solve (solve_algo 2|4) reports its block count
    if ssm.is_none() && !steady_opts.linear && matches!(steady_opts.solve_algo, 2 | 4) {
        let mut g = ys0.to_vec();
        model.set_auxiliary_variables(&mut g, xs, params);
        let mut r = vec![0.0; model.n_equations()];
        model.static_residuals(&g, xs, params, &mut r);
        let j = model.static_jacobian(&g, xs, params);
        let finite = r.iter().all(|v| v.is_finite()) && j.iter().all(|v| v.is_finite());
        if finite && r.iter().map(|v| v.abs()).fold(0.0, f64::max) >= steady_opts.tolf {
            println!("DYNARE_SOLVE (solve_algo=2|4): number of blocks = {}", crate::solver::dmperm(&j).r.len() - 1);
        }
    }
    let (ys, p) = match evaluate_steady_state(model, ssm, ys0, xs, params, steady_opts, !nocheck) {
        Ok(r) => r,
        Err(_) => {
            println!("MODEL_DIAGNOSTICS: The steady state cannot be computed");
            let mut g = ys0.to_vec();
            model.set_auxiliary_variables(&mut g, xs, params);
            if g.iter().any(|v| v.is_nan()) {
                println!("MODEL_DIAGNOSTICS: Steady state contains NaNs");
            }
            if g.iter().any(|v| v.is_infinite()) {
                println!("MODEL_DIAGNOSTICS: Steady state contains Inf");
            }
            return;
        }
    };
    // static Jacobian
    let jacob = model.static_jacobian(&ys, xs, &p);
    let bad: Vec<(usize, usize)> = cells(&jacob, |v| !v.is_finite());
    if !bad.is_empty() {
        problem = true;
        println!(
            "\nMODEL_DIAGNOSTICS: The Jacobian of the static model contains Inf or NaN. The problem arises from: \n"
        );
        problematic(model, &bad, &ys, false);
    }
    let rank = if bad.is_empty() { matlab_rank(&jacob) } else { jacob.nrows() };
    let mut singular = false;
    if rank < jacob.nrows() {
        problem = true;
        singular = true;
        println!();
        println!("{}", "=".repeat(80));
        println!("MODEL_DIAGNOSTICS: Singularity in Static Jacobian");
        println!("{}", "=".repeat(80));
        println!("The Jacobian of the static model is singular.");
        println!("There are {} collinear relationship(s) between the variables and the equations.", n - rank);
        collinear_vars(model, &jacob, "Static Jacobian");
        collinear_eqs(model, &jacob, "Static Jacobian");
        println!();
    }
    if singular {
        let pe = model.params_with_steady_state(&p, &ys);
        let g1 = model.dynamic_jacobian(&ys, xs, &pe);
        if let Ok(dr) =
            crate::solver::first_order(&g1, &model.lead_lag_incidence, model.n_exo(), &ys, &Default::default())
            && dr.eigval_abs.iter().any(|e| (e - 1.0).abs() < 1e-6)
        {
            println!("MODEL_DIAGNOSTICS:  The singularity seems to be (partly) caused by the presence of a unit root");
            println!("MODEL_DIAGNOSTICS:  as the absolute value of one eigenvalue is in the range of +-1e-6 to 1.");
            println!(
                "MODEL_DIAGNOSTICS:  If the model is actually supposed to feature unit root behavior, such a warning is expected,"
            );
            println!(
                "MODEL_DIAGNOSTICS:  but you should nevertheless check whether there is an additional singularity problem."
            );
        }
        println!("MODEL_DIAGNOSTICS:  The presence of a singularity problem typically indicates that there is one");
        println!(
            "MODEL_DIAGNOSTICS:  redundant equation entered in the model block, while another non-redundant equation"
        );
        println!("MODEL_DIAGNOSTICS:  is missing. The problem often derives from Walras Law.");
    }
    // dynamic Jacobian at the steady state
    let pe = model.params_with_steady_state(&p, &ys);
    let g1 = model.dynamic_jacobian(&ys, xs, &pe);
    let bad = cells(&g1, |v| !v.is_finite());
    if !bad.is_empty() {
        problem = true;
        println!(
            "\nMODEL_DIAGNOSTICS: The Jacobian of the dynamic model contains Inf or NaN. The problem arises from: \n"
        );
        problematic(model, &bad, &ys, true);
    }
    let rank_dyn = if bad.is_empty() { matlab_rank(&g1) } else { g1.nrows() };
    if rank_dyn < n {
        problem = true;
        println!();
        println!("{}", "=".repeat(80));
        println!("MODEL_DIAGNOSTICS: Singularity in Dynamic Jacobian");
        println!("{}", "=".repeat(80));
        println!("The Jacobian of the dynamic model is singular.");
        println!("There are {} redundant equation(s) in the model.", n - rank_dyn);
        collinear_eqs(model, &g1, "Dynamic Jacobian");
        println!();
    }
    let contemp = Mat::from_fn(g1.nrows(), n, |i, j| g1[(i, n + j)]);
    let rank_c = if contemp.iter().all(|v| v.is_finite()) { matlab_rank(&contemp) } else { n };
    if rank_c < n {
        problem = true;
        println!();
        println!("{}", "=".repeat(80));
        println!("MODEL_DIAGNOSTICS: Singularity in Contemporaneous Dynamic Jacobian");
        println!("{}", "=".repeat(80));
        println!("The contemporaneous part of the Jacobian of the dynamic model is singular.");
        println!("There are {} collinear relationship(s) between the variables and the equations.", n - rank_c);
        collinear_vars(model, &contemp, "Contemporaneous Dynamic Jacobian");
        collinear_eqs(model, &contemp, "Contemporaneous Dynamic Jacobian");
        println!();
    }
    if order >= 2 {
        let d = model.dynamic_derivatives(2);
        let v = d.eval(&ys, xs, &pe);
        if v.g2.iter().any(|x| !x.is_finite()) {
            problem = true;
            println!("\nMODEL_DIAGNOSTICS: The Hessian of the dynamic model contains Inf or NaN.");
        }
    }
    if !problem {
        println!("MODEL_DIAGNOSTICS:  No obvious problems with this mod-file were detected.");
    }
}

/// `find(pred(A))` (row, col), column-major order.
fn cells(a: &Mat, pred: impl Fn(f64) -> bool) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            if pred(a[(i, j)]) {
                v.push((i, j));
            }
        }
    }
    v
}
