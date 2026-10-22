//! `model_options` equivalence with the `model` block options; `steady` fsolve_options / non_zero.
use dynare::deterministic::DetState;
use dynare::modfile::{Command, ModFile, parse_mod_str};

const HEAD: &str = "var y c; varexo e; parameters a b; a=0.5; b=0.9;\n";
const BODY: &str = "y = a*y(-1) + e;\nc = b*c(-1) + y;\n";

fn parse(model_hdr: &str, extra: &str) -> dynare::modfile::ModFile {
    parse_mod_str(&format!("{HEAD}model{model_hdr};\n{BODY}end;\n{extra}shocks; var e; stderr 1; end;\n"), &[]).unwrap()
}

#[test]
fn model_options_linear_matches_model_linear() {
    let a = parse("(linear)", "");
    let b = parse("", "model_options(linear);\n");
    assert!(a.model.linear && b.model.linear);
    assert!(!parse("", "").model.linear);
}

#[test]
fn model_options_block_options_match_model_block() {
    for (hdr, opt) in [
        ("(block)", "block"),
        ("(block, mfs=2)", "block, mfs=2"),
        ("(block, no_static, cutoff=0)", "block, no_static, cutoff=0"),
        ("(block, static_mfs=1, bytecode)", "block, static_mfs=1, bytecode"),
        ("(use_dll)", "use_dll"),
    ] {
        let a = parse(hdr, "");
        let b = parse("", &format!("model_options({opt});\n"));
        assert_eq!(a.block_opts.block, b.block_opts.block, "{opt}");
        assert_eq!(a.block_opts.mfs, b.block_opts.mfs, "{opt}");
        assert_eq!(a.block_opts.static_mfs, b.block_opts.static_mfs, "{opt}");
        assert_eq!(a.block_opts.cutoff, b.block_opts.cutoff, "{opt}");
        assert_eq!(a.block_opts.no_static, b.block_opts.no_static, "{opt}");
        assert_eq!(a.model.block_solve.is_some(), b.model.block_solve.is_some(), "{opt}");
        assert_eq!(a.model.linear, b.model.linear, "{opt}");
    }
}

const NL: &str = "var y z; varexo e; parameters a; a=0.5;\nmodel;\n[name='cubic'] y^3 + y = 10 + e;\nz = a*y;\nend;\ninitval; y=50; z=1; e=0; end;\n";

fn steady_result(cmd: &str) -> (bool, DetState, ModFile) {
    let mf = parse_mod_str(&format!("{NL}{cmd}"), &[]).unwrap();
    let mut det = DetState::new(&mf).unwrap();
    let mut ok = true;
    for c in &mf.commands {
        if let Command::Steady(o) = c {
            let r = det.steady(&mf.model, o);
            if let Err(e) = &r {
                eprintln!("ERR {e}");
            }
            ok &= r.is_ok();
        }
    }
    (ok, det, mf)
}

#[test]
fn fsolve_options_map_onto_solve_algo_0() {
    let (ok, d, _) = steady_result("steady(solve_algo=0, noprint);\n");
    assert!(ok && (d.steady_state[0].powi(3) + d.steady_state[0] - 10.0).abs() < 1e-5);
    // MaxIter=1 binds (one trust-region iteration from y=50 cannot converge)
    let (ok, _, _) = steady_result("steady(solve_algo=0, noprint, fsolve_options=('MaxIter',1));\n");
    assert!(!ok, "MaxIter=1 must make the solver give up");
    // TolFun loosened to 1e3 accepts the starting point; Display/MaxFunEvals are accepted
    let (ok, d, _) = steady_result(
        "steady(solve_algo=0, noprint, fsolve_options=('TolFun',1e6,'Display','iter','MaxFunEvals',10));\n",
    );
    assert!(ok && d.steady_state[0] > 5.0, "loose TolFun stops at the initial point");
}

#[test]
fn non_zero_residual_listing() {
    let (_, _, mf) = steady_result("");
    let names = dynare::misc::equation_names(&mf);
    let z = [3.0, 0.0];
    let all = dynare::misc::residuals_text(&mf.model, &z, &names, 1e-5, false);
    assert!(all.starts_with("\n\nResiduals of the static equations:\n"), "{all:?}");
    assert!(all.contains("Equation number 1: cubic :"), "{all}");
    assert!(all.contains("Equation number 2: z     :"), "{all}");
    let nz = dynare::misc::residuals_text(&mf.model, &z, &names, 1e-5, true);
    assert!(nz.contains("Equation number 1") && !nz.contains("Equation number 2"), "{nz}");
    let none = dynare::misc::residuals_text(&mf.model, &[0.0, 1e-9], &names, 1e-5, true);
    assert!(none.contains("All residuals are zero") && !none.contains("Equation number"), "{none}");
}
