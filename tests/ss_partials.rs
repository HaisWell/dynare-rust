//! steady_state_model multi-output / external steady-state file errors and the accessor setters.
use dynare::modfile::{Command, parse_mod_file, parse_mod_str};

const BASE: &str = "var y; varexo e; parameters a b; a=0.5; b=2;\nmodel; y = a*y(-1) + e; end;\n";

#[test]
fn multi_output_assignment_gives_clear_error() {
    let src = format!("{BASE}steady_state_model; [y, z] = f(a); end;\n");
    let e = parse_mod_str(&src, &[]).unwrap_err().to_string();
    assert!(e.contains("multiple-output") && e.contains("one assignment per output"), "{e}");
}

#[test]
fn external_steadystate_file_is_refused() {
    let d = std::env::temp_dir().join(format!("ss_partials_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("m.mod"), format!("{BASE}shocks; var e; stderr 1; end;\n")).unwrap();
    assert!(parse_mod_file(&d.join("m.mod"), &[]).is_ok());
    std::fs::write(d.join("m_steadystate.m"), "function [ys,params,check]=m_steadystate(ys,exo,params)\n").unwrap();
    let e = parse_mod_file(&d.join("m.mod"), &[]).unwrap_err().to_string();
    assert!(e.contains("steady_state_model;") && e.contains("m_steadystate.m"), "{e}");
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn setters_before_and_after_commands() {
    let src = format!(
        "{BASE}shocks; var e; stderr 1; end;\nset_param_value('a', 0.3*2);\nset_shock_stderr_value('e', 3);\n\
         set_shock_skew_value('e', 0.1);\ncheck;\nset_param_value('b', 7);\nset_shock_stderr_value('e', 4);\n"
    );
    let mf = parse_mod_str(&src, &[]).unwrap();
    let ia = mf.model.param_names.iter().position(|n| n == "a").unwrap();
    assert!((mf.model.params[ia] - 0.6).abs() < 1e-14);
    assert!((mf.sigma_e[(0, 0)] - 9.0).abs() < 1e-14);
    let others = mf.commands.iter().filter(|c| matches!(c, Command::Other(..))).count();
    assert_eq!(others, 2, "later setters become commands");
}

#[test]
fn setter_errors_match_dynare() {
    let e = parse_mod_str(&format!("{BASE}set_param_value('zz', 1);\n"), &[]).unwrap_err().to_string();
    assert!(e.contains("Parameter name zz doesn't exist"), "{e}");
    let e = parse_mod_str(&format!("{BASE}set_shock_stderr_value('zz', 1);\n"), &[]).unwrap_err().to_string();
    assert!(e.contains("Shock name zz doesn't exist"), "{e}");
}
