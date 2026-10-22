// SPDX-License-Identifier: GPL-3.0-or-later
//! External `<fname>_steadystate.m` files written on Dynare's template (e.g.
//! examples/optimal_policy/nk_ramsey_steady_file_steadystate.m): the function copies the
//! parameters (and the Ramsey instruments) into variables with `eval` loops, runs plain
//! assignments, and copies the endogenous variables back into `ys` with another `eval` loop.
//! Such a file is equivalent to a `steady_state_model` block holding its assignments, which is
//! what [`translate`] returns (as .mod source lines). Anything else in the file (conditionals,
//! function calls outside expressions, assignments to parameters) is refused.

use anyhow::{Result, bail};

/// The `steady_state_model` statements equivalent to the template file `src`, given the
/// parameter names; `Err` when the file does not follow the template.
pub fn translate(src: &str, param_names: &[String]) -> Result<Vec<String>> {
    // strip comments (outside strings) and join continuation lines
    let mut text = String::new();
    for line in src.lines() {
        let mut out = String::new();
        let mut in_str = false;
        let mut prev = ' ';
        for c in line.chars() {
            if c == '\'' {
                // a quote after a value character is a transpose
                if in_str || !(prev.is_alphanumeric() || prev == ')' || prev == ']' || prev == '_' || prev == '.') {
                    in_str = !in_str;
                }
            }
            if c == '%' && !in_str {
                break;
            }
            out.push(c);
            if !c.is_whitespace() {
                prev = c;
            }
        }
        let t = out.trim_end();
        if let Some(stripped) = t.strip_suffix("...") {
            text.push_str(stripped);
            text.push(' ');
        } else {
            text.push_str(t);
            text.push('\n');
        }
    }
    let mut stmts: Vec<String> = Vec::new();
    for line in text.lines() {
        // split on ';' outside brackets/strings
        let mut depth = 0i32;
        let mut cur = String::new();
        let mut in_str = false;
        for c in line.chars() {
            match c {
                '\'' => in_str = !in_str,
                '(' | '[' | '{' if !in_str => depth += 1,
                ')' | ']' | '}' if !in_str => depth -= 1,
                _ => {}
            }
            if c == ';' && depth == 0 && !in_str {
                stmts.push(std::mem::take(&mut cur));
            } else {
                cur.push(c);
            }
        }
        stmts.push(cur);
    }
    let stmts: Vec<String> = stmts.into_iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let Some(first) = stmts.first() else { bail!("empty file") };
    let header = first.strip_prefix("function").map(str::trim).unwrap_or("");
    if !header.replace(' ', "").starts_with("[ys,params,check]=") {
        bail!("not a steady-state function [ys,params,check] = f(ys,exo,M_,options_)");
    }
    let mut loop_depth = 0;
    let mut writes_back = false;
    let mut out = Vec::new();
    for s in &stmts[1..] {
        let compact = s.replace(' ', "");
        if compact.starts_with("for") && (s.starts_with("for ") || s.starts_with("for\t")) {
            loop_depth += 1;
            continue;
        }
        if compact == "end" {
            if loop_depth == 0 {
                // the function's end
                continue;
            }
            loop_depth -= 1;
            continue;
        }
        if loop_depth > 0 {
            // template loop bodies: only the eval copies
            if compact.starts_with("eval(") || compact.starts_with("paramname=") || compact.starts_with("varname=") {
                if compact.contains("'ys('") || compact.contains("['ys(") {
                    writes_back = true;
                }
                continue;
            }
            bail!("unsupported statement in a loop: {s}");
        }
        if compact.contains("M_.")
            || compact.contains("options_.")
            || compact.starts_with("check=")
            || compact.starts_with("params=")
        {
            continue;
        }
        let Some((lhs, rhs)) = s.split_once('=') else { bail!("unsupported statement: {s}") };
        let (lhs, rhs) = (lhs.trim(), rhs.trim());
        if !lhs.chars().all(|c| c.is_alphanumeric() || c == '_') || rhs.starts_with('=') {
            bail!("unsupported statement: {s}");
        }
        if param_names.iter().any(|p| p == lhs) {
            // `beta = NaN;` placeholders that shadow MATLAB functions are harmless
            if rhs == "NaN" || rhs == "nan" {
                continue;
            }
            bail!("assignment to the parameter {lhs} (parameters set in a steady-state file are not supported)");
        }
        if rhs.contains('\'') || rhs.contains('{') || rhs.contains('[') {
            bail!("unsupported expression: {s}");
        }
        out.push(format!("{lhs} = {rhs};"));
    }
    if !writes_back {
        bail!("the file does not copy the endogenous variables back into ys with Dynare's template loop");
    }
    if out.is_empty() {
        bail!("no steady-state assignments");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::translate;

    #[test]
    fn template_file() {
        let src = "function [ys,params,check] = m_steadystate(ys,exo,M_,options_)\n% doc\nbeta=NaN; %make it known\nNumberOfParameters = M_.param_nbr;\nfor ii = 1:NumberOfParameters\n  paramname = M_.param_names{ii};\n  eval([ paramname ' = M_.params(' int2str(ii) ');']);\nend\ncheck = 0;\n    Z=0;\n    pi=(R+1)*beta; % comment\n    C=sqrt(1+ ...\n  pi);\nparams=NaN(NumberOfParameters,1);\nfor iter = 1:length(M_.params)\n  eval([ 'params(' num2str(iter) ') = ' M_.param_names{iter} ';' ])\nend\nNumberOfEndogenousVariables = M_.orig_endo_nbr;\nfor ii = 1:NumberOfEndogenousVariables\n  varname = M_.endo_names{ii};\n  eval(['ys(' int2str(ii) ') = ' varname ';']);\nend\nend\n";
        let out = translate(src, &["beta".to_string()]).unwrap();
        assert_eq!(out, vec!["Z = 0;", "pi = (R+1)*beta;", "C = sqrt(1+    pi);"]);
        assert!(translate("function [ys,params,check]=m_steadystate(ys,exo,params)\n", &[]).is_err());
        assert!(translate("function x\n", &[]).is_err());
        let with_if = src.replace("Z=0;", "if R > 0, Z = 0; end");
        assert!(translate(&with_if, &["beta".to_string()]).is_err());
    }
}
