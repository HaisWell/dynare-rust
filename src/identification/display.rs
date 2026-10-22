// SPDX-License-Identifier: GPL-3.0-or-later
//! Console output of the identification analysis: port of `+identification/display.m`.

use super::checks::CheckResult;
use super::la::vech_pair;
use super::run::IdentOptions;

/// C/MATLAB `%.{prec}g`.
pub fn fmt_g(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Inf".into() } else { "-Inf".into() };
    }
    if x == 0.0 {
        return "0".into();
    }
    let p = prec.max(1);
    let e = x.abs().log10().floor() as i32;

    if e < -4 || e >= p as i32 {
        let s = format!("{:.*e}", p - 1, x);
        // mantissa: strip trailing zeros; exponent: sign and two digits
        let (m, ex) = s.split_once('e').unwrap();
        let m = if m.contains('.') { m.trim_end_matches('0').trim_end_matches('.') } else { m };
        let ev: i32 = ex.parse().unwrap();
        format!("{m}e{}{:02}", if ev < 0 { '-' } else { '+' }, ev.abs())
    } else {
        let dec = (p as i32 - 1 - e).max(0) as usize;
        let s = format!("{:.*}", dec, x);
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
    }
}

/// MATLAB `num2str(x)` for a scalar.
pub fn num2str(x: f64) -> String {
    if x.is_finite() && x == x.round() && x.abs() < 1e15 {
        return format!("{}", x as i64);
    }
    let d = if x != 0.0 && x.is_finite() { (x.abs().log10().floor() as i64).max(0) as usize } else { 0 };
    fmt_g(x, d + 5)
}

/// MATLAB `sprintf('%.0d', x)` for a non-integer (falls back to `%.0e`): e.g. `1e-08`.
pub fn fmt_0d(x: f64) -> String {
    if x == x.round() && x.abs() < 1e15 {
        return format!("{}", x as i64);
    }
    let s = format!("{:.0e}", x);
    let (m, ex) = s.split_once('e').unwrap();
    let ev: i32 = ex.parse().unwrap();
    format!("{m}e{}{:02}", if ev < 0 { '-' } else { '+' }, ev.abs())
}

pub fn tol_rank_str(t: Option<f64>) -> String {
    match t {
        None => "robust".into(),
        Some(v) => num2str(v),
    }
}

/// One criterion over the draws (a single draw for the point analysis).
pub struct Criterion<'a> {
    pub checks: Vec<&'a CheckResult>,
    /// Komunjer–Ng only: per draw, whether the minimal state space could be computed.
    pub minimal_state_space: Option<Vec<bool>>,
    /// the `checks_via_subsets` result (point analysis).
    pub subsets: Option<&'a super::subsets::SubsetResult>,
}

/// Port of `identification.display(pdraws, ide_reducedform, ide_moments, ide_spectrum,
/// ide_minimal, name, options_ident)`. `criteria` in display order: reduced form, minimal
/// system, spectrum, moments (`None` = skipped). `bruteforce`: the `advanced` collinearity
/// patterns `(pars, cosn)` of the point analysis.
pub fn display(
    sample_size: usize,
    criteria: [Option<Criterion>; 4],
    names: &[String],
    o: &IdentOptions,
    bruteforce: Option<&(Vec<Vec<Vec<Option<usize>>>>, crate::linalg::Mat)>,
) {
    let tot = names.len();
    println!("  ");
    println!("Note that differences in the criteria could be due to numerical settings,");
    println!("numerical errors or the method used to find problematic parameter sets.");
    println!("Settings:");
    if o.analytic_derivation_mode == 0 {
        println!("    Derivation mode for Jacobians:                         Analytic using sylvester equations");
    } else if o.analytic_derivation_mode == 1 {
        println!("    Derivation mode for Jacobians:                         Analytic using kronecker products");
    } else {
        println!("    Derivation mode for Jacobians:                         Numerical");
    }
    if o.checks_via_subsets {
        println!("    Method to find problematic parameters:                 Rank condition on all possible subsets");
    } else {
        println!(
            "    Method to find problematic parameters:                 Nullspace and multicorrelation coefficients"
        );
    }
    println!(
        "    Normalize Jacobians:                                   {}",
        if o.normalize_jacobians { "Yes" } else { "No" }
    );
    println!("    Tolerance level for rank computations:                 {}", tol_rank_str(o.tol_rank));
    println!("    Tolerance level for selecting nonzero columns:         {}", fmt_0d(o.tol_deriv));
    println!("    Tolerance level for selecting nonzero singular values: {}", fmt_0d(o.tol_sv));

    let tests = [
        ("REDUCED-FORM", "Tau", "Jacobian of steady state and reduced-form solution matrices"),
        ("MINIMAL SYSTEM (Komunjer and Ng, 2011)", "Deltabar", "Jacobian of steady state and minimal system"),
        ("SPECTRUM (Qu and Tkachenko, 2012)", "Gbar", "Jacobian of mean and spectrum"),
        ("MOMENTS (Iskrev, 2010)", "J", "Jacobian of first two moments"),
    ];
    for (jide, crit) in criteria.iter().enumerate() {
        let (mut str_test, mut str_jac, mut str_meaning) = tests[jide];
        // ---- display.m's titles at order > 1 ----
        match (jide, o.order) {
            (1, 2) => str_meaning = "Jacobian of first-order minimal system and second-order accurate mean",
            (1, 3) => str_meaning = "Jacobian of first-order minimal system and third-order accurate mean",
            (2, k) if k > 1 => str_test = "SPECTRUM (Mutschler, 2015)",
            (3, k) if k > 1 => (str_test, str_jac) = ("MOMENTS (Mutschler, 2015)", "Mbar"),
            _ => {}
        }
        let Some(crit) = crit else {
            continue;
        };
        // ---- display of identification.checks_via_subsets ----
        if let Some(sr) = crit.subsets {
            if !super::subsets::display_subsets(sr, names, str_test, str_jac, str_meaning) {
                println!();
                println!("{}:", str_test.to_uppercase());
                println!(
                    "    All parameters are identified in the {str_meaning} (rank({str_jac}) is full with tol = {}).",
                    tol_rank_str(o.tol_rank)
                );
            }
            continue;
        }
        let mut no_warning = true;
        let mut ck: Vec<&CheckResult> = crit.checks.clone();
        let mut effective = sample_size;
        let mut non_minimal_error = false;
        let mut mss: Vec<bool> = crit.minimal_state_space.clone().unwrap_or_else(|| vec![true; ck.len()]);
        if sample_size > 1 && jide == 1 && mss.iter().any(|m| !m) {
            effective = sample_size - mss.iter().filter(|m| !**m).count();
            non_minimal_error = true;
            // checks of the failed draws are absent from `ck` already
            mss.retain(|m| *m);
        }
        let any_ino = ck.iter().any(|c| c.ino > 0);
        let any_ind0 = ck.iter().any(|c| c.ind0.iter().any(|b| !b));
        let any_pair = ck.iter().any(|c| c.jweak_pair.iter().any(|b| *b));
        if any_ino || any_ind0 || any_pair || non_minimal_error {
            no_warning = false;
            println!();
            println!("{}:", str_test.to_uppercase());
            println!("    !!!WARNING!!!");
            if sample_size > 1 {
                if non_minimal_error {
                    println!(
                        "\n    The minimal state space could not be computed for {} out of {} cases.",
                        sample_size - effective,
                        sample_size
                    );
                }
                let ndef = ck.iter().filter(|c| c.ino > 0).count();
                if jide == 1 {
                    if ndef > 0 {
                        println!(
                            "    The rank of {str_jac} ({str_meaning}) is deficient for {ndef} out of {effective} effective MC runs!"
                        );
                    }
                } else {
                    println!(
                        "    The rank of {str_jac} ({str_meaning}) is deficient for {ndef} out of {effective} effective MC runs!"
                    );
                }
            } else {
                println!("    The rank of {str_jac} ({str_meaning}) is deficient!");
            }
            println!();
            for (j, name) in names.iter().enumerate() {
                let nz = ck.iter().filter(|c| !c.ind0[j]).count();
                if nz > 0 {
                    if sample_size > 1 {
                        let pno = 100.0 * nz as f64 / effective as f64;
                        println!("    {name} is not identified for {}% of MC runs!", num2str(pno));
                    } else {
                        println!("    {name} is not identified!");
                    }
                }
            }
            let npairs = tot * (tot + 1) / 2;
            let mut jstore = Vec::new();
            for k in 0..npairs {
                let iweak = ck.iter().filter(|c| c.jweak_pair[k]).count();
                if iweak > 0 {
                    let (jx, jy) = vech_pair(k);
                    jstore.push(jx);
                    jstore.push(jy);
                    if sample_size > 1 {
                        println!(
                            "    [{},{}] are PAIRWISE collinear for {}% of MC runs!",
                            names[jx],
                            names[jy],
                            num2str(iweak as f64 / effective as f64 * 100.0)
                        );
                    } else {
                        println!("    [{},{}] are PAIRWISE collinear!", names[jx], names[jy]);
                    }
                }
            }
            for (j, name) in names.iter().enumerate() {
                let iweak = ck.iter().filter(|c| c.jweak[j]).count();
                if iweak > 0 && !jstore.contains(&j) {
                    if sample_size > 1 {
                        println!(
                            "    {name} is collinear w.r.t. all other parameters for {}% of MC runs!",
                            num2str(iweak as f64 / effective as f64 * 100.0)
                        );
                    } else {
                        println!("    {name} is collinear w.r.t. all other parameters!");
                    }
                }
            }
        }
        ck.clear();
        if no_warning {
            println!();
            println!("{}:", str_test.to_uppercase());
            println!(
                "    All parameters are identified in the {str_meaning} (rank({str_jac}) is full with tol = {}).",
                tol_rank_str(o.tol_rank)
            );
        }
    }

    if sample_size == 1 && o.advanced {
        if let Some((pars, cosn)) = bruteforce {
            println!();
            for j in 0..cosn.ncols() {
                println!();
                println!("Collinearity patterns with {} parameter(s)", j + 1);
                println!("{:<15} [{:<w$}] {:>10}", "Parameter", " Expl. params ", "cosn", w = 16 * (j + 1));
                for i in 0..tot {
                    let mut namx = String::new();
                    for p in &pars[i][j] {
                        namx.push(' ');
                        match p {
                            None => namx.push_str(&format!("{:<15}", "--")),
                            Some(k) => namx.push_str(&format!("{:<15}", names[*k])),
                        }
                    }
                    println!("{:<15} [{}] {:14.7}", names[i], namx, cosn[(i, j)]);
                }
            }
        }
    }
}
