//! LaTeX output and model_comparison against Dynare 7.2 (reference/latex/, written by
//! reference/latex/dump_latex.m): runs latex_nk_alt.mod then latex_nk.mod (write_latex_*,
//! estimation with TeX, model_comparison, collect_latex_files) in a temporary copy and compares
//! every file with Dynare's.
mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dynare::driver::{RunOptions, run};
use dynare::estimation::model_comparison::{compare, parse_models};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/latex")
}

fn setup(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["nk_model.inc", "latex_nk.mod", "latex_nk_alt.mod", "nk_data.csv"] {
        std::fs::copy(root().join(f), dir.join(f)).unwrap();
    }
    dir
}

/// Drops the `% <date>` lines and returns the rest.
fn undated(s: &str) -> Vec<String> {
    s.lines()
        .filter(|l| !(l.starts_with("% ") && l.chars().nth(4) == Some('-') && l.contains(':')))
        .map(str::to_string)
        .collect()
}

/// Character multiset of an equation line (operand order is not Dynare's in the transformed
/// model; see src/latex/mod.rs).
fn chars(s: &str) -> BTreeMap<char, usize> {
    let mut m = BTreeMap::new();
    for c in s.chars().filter(|c| !c.is_whitespace()) {
        *m.entry(c).or_insert(0) += 1;
    }
    m
}

fn check_file(ours: &Path, theirs: &Path, structural: bool) {
    let a = undated(&std::fs::read_to_string(ours).unwrap_or_else(|_| panic!("missing {}", ours.display())));
    let b = undated(&std::fs::read_to_string(theirs).unwrap());
    assert_eq!(a.len(), b.len(), "{}: number of lines", ours.display());
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        if structural && x.contains('=') && !x.starts_with('%') && !x.starts_with("\\noindent") {
            let (lx, rx) = x.split_once('=').unwrap();
            let (ly, ry) = y.split_once('=').unwrap();
            assert_eq!(lx, ly, "{} line {}: lhs", ours.display(), k + 1);
            assert_eq!(chars(rx), chars(ry), "{} line {}:\n ours   {x}\n Dynare {y}", ours.display(), k + 1);
        } else {
            assert_eq!(x, y, "{} line {}", ours.display(), k + 1);
        }
    }
}

#[test]
fn latex_outputs_and_model_comparison_match_dynare() {
    let dir = setup("latex_nk");
    let ro = RunOptions { quiet: true, ..Default::default() };
    run(&dir.join("latex_nk_alt.mod"), &ro).unwrap();
    run(&dir.join("latex_nk.mod"), &ro).unwrap();
    let exp = root().join("expected");
    let ours = dir.join("latex_nk/latex");
    // identical files (up to the date comments)
    for f in [
        "original.tex",
        "dynamic.tex",
        "static.tex",
        "original_content.tex",
        "latex_nk_latex_definitions.tex",
        "latex_nk_latex_parameters.tex",
        "latex_nk_priors_table.tex",
        "latex_nk_Posterior_Mode_1.tex",
        "latex_nk_Posterior_Mode_2.tex",
    ] {
        check_file(&ours.join(f), &exp.join(f), false);
    }
    check_file(&dir.join("latex_nk_TeX_binder.tex"), &exp.join("latex_nk_TeX_binder.tex"), false);
    // same equations, operands of commutative operators possibly in another order
    for f in ["dynamic_content.tex", "static_content.tex"] {
        check_file(&ours.join(f), &exp.join(f), true);
    }
    // model comparison: our Laplace approximations agree with Dynare's to ~1e-11 relative (the
    // csminwel modes differ by ~2e-9, see laplace_pieces_match_dynare), so the table is compared
    // line by line with the numbers checked to 1e-9 (the table prints 6 digits).
    let mc = std::fs::read_to_string(ours.join("latex_nk_model_comparisonLaplaceApproximation.tex")).unwrap();
    let mc_ref = std::fs::read_to_string(exp.join("latex_nk_model_comparisonLaplaceApproximation.tex")).unwrap();
    let (a, b) = (undated(&mc), undated(&mc_ref));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        let nx: Vec<f64> = x.split('&').filter_map(|c| c.trim().trim_end_matches("\\\\").trim().parse().ok()).collect();
        let ny: Vec<f64> = y.split('&').filter_map(|c| c.trim().trim_end_matches("\\\\").trim().parse().ok()).collect();
        assert_eq!(nx.len(), ny.len(), "{x} / {y}");
        for (p, q) in nx.iter().zip(&ny) {
            assert!((p - q).abs() < 1e-9 * q.abs().max(1.0), "{x} / {y}");
        }
        if nx.is_empty() {
            assert_eq!(x, y);
        }
    }
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(root().join("ref_model_comparison.json")).unwrap()).unwrap();
    let ours: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("latex_nk/Output/latex_nk_model_comparison.json")).unwrap(),
    )
    .unwrap();
    for m in ["latex_nk", "latex_nk_alt"] {
        for k in ["Prior", "Log_Marginal_Density", "Bayes_Ratio", "Posterior_Model_Probability"] {
            let (p, q) = (ours["models"][m][k].as_f64().unwrap(), r["model_comparison"][m][k].as_f64().unwrap());
            // Bayes ratio and model probabilities exponentiate a difference of log marginal
            // densities: 3e-8 with OpenBLAS and another libm (Linux, Windows).
            let tol = if cfg!(target_os = "macos") { 1e-8 } else { 2e-7 };
            assert!((p - q).abs() < tol * q.abs().max(1.0), "{m}.{k}: {p} vs {q}");
        }
    }
}

#[test]
fn model_comparison_formula_matches_dynare_exactly() {
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(root().join("ref_model_comparison.json")).unwrap()).unwrap();
    let names = vec!["latex_nk".to_string(), "latex_nk_alt".to_string()];
    let d: Vec<f64> = names.iter().map(|n| r["laplace"][n].as_f64().unwrap()).collect();
    // `model_comparison(marginal_density = laplace) latex_nk latex_nk_alt;` (priors 1, 1)
    let c = compare(&names, &[1.0, 1.0], &d);
    for (i, n) in names.iter().enumerate() {
        let w = &r["model_comparison"][n];
        for (k, v) in [
            ("Prior", c.prior[i]),
            ("Bayes_Ratio", c.bayes_ratio[i]),
            ("Posterior_Model_Probability", c.posterior_probability[i]),
        ] {
            let q = w[k].as_f64().unwrap();
            assert!((v - q).abs() < 1e-12, "{n}.{k}: {v} vs {q}");
        }
    }
    // the first call of latex_nk.mod: priors 0.4/0.6 (printed by Dynare: 3.122864, 0.242550)
    let c = compare(&["latex_nk_alt".into(), "latex_nk".into()], &[0.4, 0.6], &[d[1], d[0]]);
    assert!((c.bayes_ratio[1] - 3.122864).abs() < 1e-6);
    assert!((c.posterior_probability[0] - 0.242550).abs() < 1e-6);
}

#[test]
fn model_comparison_arguments() {
    let t = |s: &[&str]| parse_models(&s.iter().map(|x| x.to_string()).collect::<Vec<_>>()).unwrap();
    assert_eq!(t(&["m1", "(", "0.7", ")", "m2", "(", "0.3", ")"]), vec![("m1".into(), 0.7), ("m2".into(), 0.3)]);
    assert_eq!(t(&["m1", "m2"]), vec![("m1".into(), 1.0), ("m2".into(), 1.0)]);
    assert_eq!(t(&["m1", ",", "m2", "(", "2", ")"]), vec![("m1".into(), 1.0), ("m2".into(), 2.0)]);
}

/// The pieces of the Laplace approximation (reference/latex/dump_laplace.m: Dynare's csminwel
/// mode, fval and hessian.m Hessian from `<fname>_mode.mat`, and its Laplace value): the
/// formula reproduces Dynare's value from Dynare's hh and fval to all digits, our hessian.m at
/// Dynare's mode agrees with Dynare's hh to ~3e-9 (finite differences of fval, rounding), and
/// the whole run (our csminwel endpoint differs by ~2e-9) to ~1e-11 relative.
#[test]
fn laplace_pieces_match_dynare() {
    use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
    use dynare::estimation::posterior::laplace_approximation;
    use dynare::estimation::{Objective, hessian};
    use dynare::linalg::Mat;
    let text = common::read_ref(root().join("ref_laplace.json"));
    let r: Value = serde_json::from_str(&text).unwrap();
    let num = |v: &Value| v.as_f64().unwrap();
    let vec = |v: &Value| v.as_array().unwrap().iter().map(num).collect::<Vec<f64>>();
    let dir = setup("latex_laplace");
    let ro = RunOptions { quiet: true, ..Default::default() };
    for name in ["latex_nk_alt", "latex_nk"] {
        let w = &r[name];
        let (xd, fd, lap) = (vec(&w["xparam1"]), num(&w["fval"]), num(&w["laplace"]));
        let n = xd.len();
        let hd = Mat::from_fn(n, n, |i, j| num(&w["hh"][i][j]));
        // Dynare's formula on Dynare's mode and Hessian
        assert!((laplace_approximation(&hd, fd) - lap).abs() < 1e-12 * lap.abs(), "{name}");
        // hessian.m at Dynare's mode
        let mf = dynare::modfile::parse_mod_file(&dir.join(format!("{name}.mod")), &[]).unwrap();
        let m = DsgeModel::new(&mf, Some(&xd)).unwrap();
        let table = dynare::data::Table::read_csv(&dir.join("nk_data.csv")).unwrap();
        let data = LikelihoodData::new(&table.sample(&mf.varobs, 1, None).unwrap(), Default::default());
        let post = DsgePosterior { dsge: &m, data: &data, bounds: &m.bounds };
        let f = post.eval(&xd).0;
        assert!((f - fd).abs() < 1e-13 * fd.abs(), "{name}: fval {f} vs {fd}");
        let h = hessian(&post, &xd, [num(&w["gstep"][0]), num(&w["gstep"][1])]);
        let dh = (&h - &hd).amax() / hd.amax();
        let dl = (laplace_approximation(&h, f) - lap).abs() / lap.abs();
        eprintln!("{name}: hessian at Dynare's mode {dh:.1e}, Laplace {dl:.1e} (relative)");
        assert!(dh < 1e-8 && dl < 1e-10, "{name}: {dh} {dl}");
        // the whole estimation
        run(&dir.join(format!("{name}.mod")), &ro).unwrap();
        let res: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(format!("{name}/Output/{name}_results.json"))).unwrap(),
        )
        .unwrap();
        let ours = num(&res["log_data_density_laplace"]);
        eprintln!("{name}: Laplace {ours:.10} vs Dynare {lap:.10}");
        assert!((ours - lap).abs() < 1e-10 * lap.abs(), "{name}: {ours} vs {lap}");
    }
}

/// `write_latex_steady_state_model`: steady_state.tex / steady_state_content.tex identical to
/// the Dynare 7.1 preprocessor's on reference/latex/latex_ssm.mod.
#[test]
fn steady_state_model_latex_matches_dynare() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("latex_ssm");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(root().join("latex_ssm.mod"), dir.join("latex_ssm.mod")).unwrap();
    run(&dir.join("latex_ssm.mod"), &RunOptions { quiet: true, ..Default::default() }).unwrap();
    for f in ["steady_state.tex", "steady_state_content.tex"] {
        check_file(&dir.join("latex_ssm/latex").join(f), &root().join("expected").join(f), false);
    }
}
