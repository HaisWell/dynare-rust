//! Ramsey models go through the preprocessor's DataTree emulation (`model::dtree`), so the
//! auxiliary variables are named after the node indices of Dynare 7.1's DataTree
//! (`AUX_ENDO_LEAD_<idx>`) and the structure matches the preprocessor's `driver.m`.
//! Reference values: `dynare-preprocessor reference/policy/ramsey_cmr2.mod json=compute`.

use std::path::Path;

fn model(f: &str) -> dynare::model::Model {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/policy").join(f);
    let src = std::fs::read_to_string(&p).unwrap();
    dynare::modfile::parse_mod_str(&src, &[]).unwrap().model
}

#[test]
fn ramsey_cmr2_names_and_incidence_match_preprocessor() {
    let m = model("ramsey_cmr2.mod");
    let names: Vec<&str> = m.endo_names.iter().map(String::as_str).collect();
    assert_eq!(names.len(), 32);
    assert_eq!(names[15], "MULT_1");
    assert_eq!(names[27], "MULT_13");
    assert_eq!(names[28..], ["AUX_ENDO_LEAD_222", "AUX_ENDO_LEAD_95", "AUX_ENDO_LAG_34_1", "AUX_ENDO_LAG_3_1"]);
    let aux: Vec<&str> = m.aux_vars.iter().map(|a| a.orig_expr.as_str()).collect();
    assert_eq!(aux, ["MULT_13(1)", "log(pi(1))", "MULT_12(-1)", "Z(-1)"]);
    // M_.lead_lag_incidence of the preprocessor (transposed in driver.m: rows = endogenous)
    let lli: [[usize; 3]; 32] = [
        [1, 10, 42],
        [0, 11, 43],
        [0, 12, 0],
        [2, 13, 0],
        [0, 14, 0],
        [0, 15, 0],
        [0, 16, 0],
        [0, 17, 0],
        [0, 18, 0],
        [0, 19, 0],
        [0, 20, 0],
        [0, 21, 0],
        [0, 22, 0],
        [3, 23, 0],
        [0, 24, 0],
        [4, 25, 0],
        [5, 26, 0],
        [0, 27, 0],
        [0, 28, 44],
        [0, 29, 0],
        [0, 30, 0],
        [0, 31, 0],
        [0, 32, 0],
        [6, 33, 0],
        [0, 34, 0],
        [0, 35, 0],
        [7, 36, 0],
        [0, 37, 45],
        [0, 38, 46],
        [0, 39, 47],
        [8, 40, 0],
        [9, 41, 0],
    ];
    assert_eq!(m.lead_lag_incidence, lli);
    // equations: 15 FOCs, 13 original equations, 4 auxiliary equations
    assert_eq!(m.n_endo(), 32);
}

#[test]
fn ramsey_models_have_consecutive_multipliers() {
    for f in ["gali_ramsey.mod", "ramsey_cmr.mod", "ramsey_cmr_ssm.mod", "ramsey_cmr_o2.mod"] {
        let m = model(f);
        let ri = m.planner().unwrap().ramsey.as_ref().unwrap();
        let n_mult = m.endo_names.iter().filter(|n| n.starts_with("MULT_")).count();
        assert_eq!(n_mult, ri.orig_eq_nbr, "{f}");
        for k in 0..n_mult {
            assert_eq!(m.endo_names[ri.orig_endo_nbr + k], format!("MULT_{}", k + 1), "{f}");
        }
    }
}
