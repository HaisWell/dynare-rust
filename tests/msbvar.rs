//! MS-SBVAR part 1: `svar_identification` / `markov_switching` / `svar` declarations and the
//! constant-parameter `sbvar` command against Dynare 7.2 (reference/msbvar/ref_msbvar.json,
//! written by reference/msbvar/dump_msbvar.m). MATLAB starts the posterior-mode search from
//! `10*rand`, so its `A0` is only determined up to the sign of each column (the likelihood is
//! invariant); the comparison aligns the column signs first.

use std::path::Path;

use dynare::linalg::Mat;
use dynare::modfile::{Command, parse_mod_str};
use dynare::msbvar::{self, Equations, MsDecl, identification::SvarIdentification};
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn mat(v: &Value) -> Mat {
    let rows: Vec<Vec<f64>> = match v {
        Value::Array(a) if a.first().is_some_and(|x| x.is_array()) => {
            a.iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect()
        }
        Value::Array(a) => a.iter().map(|x| vec![x.as_f64().unwrap()]).collect(),
        Value::Number(n) => vec![vec![n.as_f64().unwrap()]],
        _ => panic!("not a matrix"),
    };
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    let d = (a - b).abs() / b.abs().max(1.0);
    assert!(d < tol, "{what}: {a} vs {b} (rel {d:e})");
}

fn close_mat(a: &Mat, b: &Mat, tol: f64, what: &str) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    let scale = b.iter().fold(1.0f64, |m, x| m.max(x.abs()));
    let d = (a - b).iter().fold(0.0f64, |m, x| m.max(x.abs())) / scale;
    assert!(d < tol, "{what}: max rel diff {d:e}");
}

fn run_case(name: &str, refs: &Value) {
    let path = root().join(format!("reference/msbvar/{name}.mod"));
    let mf = dynare::modfile::parse_mod_file(&path, &[]).unwrap();
    let Some(Command::Other(_, o)) = mf.commands.iter().find(|c| matches!(c, Command::Other(n, _) if n == "sbvar"))
    else {
        panic!("no sbvar command")
    };
    let (r, _) = msbvar::run_sbvar(&mf, o, &root().join("reference/msbvar")).unwrap();
    let m = &refs[name];
    let n = r.nvar;
    // column signs of A0 (and the matching signs of A0^{-1} rows, A+ columns, shocks)
    let a0m = mat(&m["A0hat"]);
    let s: Vec<f64> = (0..n).map(|j| if r.a0.column(j).dot(&a0m.column(j)) >= 0.0 { 1.0 } else { -1.0 }).collect();
    let sc = |x: &Mat| Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] * s[j]);
    let sr = |x: &Mat| Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)] * s[i]);
    close(r.fhat, m["fhat"].as_f64().unwrap(), 1e-9, "fhat");
    close_mat(&sc(&r.a0), &a0m, 1e-5, "A0hat");
    close_mat(&sr(&r.a0_inv), &mat(&m["A0hatinv"]), 1e-5, "A0hatinv");
    close_mat(&sc(&r.aplus), &mat(&m["Fhat"]), 1e-5, "Fhat");
    close_mat(&r.bhat, &mat(&m["Bhat"]), 1e-5, "Bhat");
    close_mat(&r.forecast, &mat(&m["yforehat"]), 1e-5, "forecast");
    // impulse responses: element (r,c) at column r + n*c, sign of shock c
    let imf_m = mat(&m["imfhat"]);
    let imf = Mat::from_fn(imf_m.nrows(), imf_m.ncols(), |t, k| r.imf[(t, k)] * s[k / n]);
    close_mat(&imf, &imf_m, 1e-5, "imfhat");
    assert_eq!(r.fss, m["fss"].as_f64().unwrap() as usize);
    // basis-independent embeddings of the posterior matrices
    for k in 0..n {
        let (um, vm) = (mat(&m["Uiconst"][k]), mat(&m["Viconst"][k]));
        let h0m = mat(&m["H0inv"][k]);
        let hpm = mat(&m["Hpinv"][k]);
        let pm = mat(&m["Pmat"][k]);
        // the restriction subspaces (projectors are basis independent)
        close_mat(&(&r.ui[k] * r.ui[k].transpose()), &(&um * um.transpose()), 1e-10, &format!("Ui[{k}]"));
        close_mat(&(&r.vi[k] * r.vi[k].transpose()), &(&vm * vm.transpose()), 1e-10, &format!("Vi[{k}]"));
        let h0r = &r.ui[k] * &r.h0inv[k] * r.ui[k].transpose();
        let hpr = &r.vi[k] * &r.hpinv[k] * r.vi[k].transpose();
        close_mat(&h0r, &(&um * h0m * um.transpose()), 1e-7, &format!("H0inv[{k}]"));
        close_mat(&hpr, &(&vm * hpm * vm.transpose()), 1e-7, &format!("Hpinv[{k}]"));
        // P maps b to g: only the Uiconst-embedded value P*U' a is basis independent
        let pr = &r.vi[k] * &r.pmat[k] * r.ui[k].transpose();
        close_mat(&pr, &(&vm * pm * um.transpose()), 1e-7, &format!("Pmat[{k}]"));
    }
    if let Some(v) = m.get("vlog_a0p") {
        close(r.vlog_a0p.unwrap(), v.as_f64().unwrap(), 1e-5, "vlog_a0p");
        close(r.vlog_y_a.unwrap(), m["vlog_Y_a"].as_f64().unwrap(), 1e-5, "vlog_Y_a");
        close(r.log_mar_lh_res.unwrap(), m["logMarLHres"].as_f64().unwrap(), 1e-5, "logMarLHres");
        close(r.vlog_ap_ya0.unwrap(), m["vlog_ap_Ya0"].as_f64().unwrap(), 1e-5, "vlog_ap_Ya0");
    } else {
        assert!(r.vlog_a0p.is_none() || name != "msbvar_flat");
    }
}

fn refs() -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join("reference/msbvar/ref_msbvar.json")).unwrap()).unwrap()
}

/// `restriction equation` cases (reference/msbvar/dump_msbvar_lin.m).
fn refs_lin() -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join("reference/msbvar/ref_msbvar_lin.json")).unwrap())
        .unwrap()
}

#[test]
fn sbvar_exclusions() {
    run_case("msbvar_excl", &refs());
}

#[test]
fn sbvar_upper_cholesky_hyperparameters() {
    run_case("msbvar_chol", &refs());
}

#[test]
fn sbvar_no_bayesian_prior() {
    run_case("msbvar_flat", &refs());
}

#[test]
fn declarations_parse_as_preprocessor() {
    let mf =
        parse_mod_str(&std::fs::read_to_string(root().join("reference/msbvar/msbvar_excl.mod")).unwrap(), &[]).unwrap();
    let d = MsDecl::from_modfile(&mf).unwrap();
    let id = d.identification.as_ref().unwrap();
    assert!(!id.upper_cholesky && !id.lower_cholesky && !id.constants_exclusion);
    assert_eq!(id.max_lag(), Some(2));
    // options_.ms.Qi{1}(1,3), Qi{3}(1,1), Ri{2}(1,3*? ) = lag 2 -> column (2-1)*3+3 = 6
    let (qi, ri) = id.qi_ri(3);
    assert_eq!(qi[0], vec![(1, 3)]);
    assert_eq!(qi[2], vec![(1, 1)]);
    assert_eq!(ri[1], vec![(1, 6)]);
    assert_eq!(d.chains.len(), 2);
    assert_eq!(d.chains[0].number_of_regimes, 2);
    assert_eq!(d.chains[0].durations, vec![2.5, 2.5]);
    assert_eq!(d.chains[1].number_of_regimes, 3);
    assert_eq!(d.chains[1].restrictions, vec![[1.0, 3.0, 0.0], [3.0, 1.0, 0.0]]);
    assert_eq!(d.chains[0].svar_variances.as_ref().unwrap().equations, Equations::All);
    assert_eq!(d.chains[1].svar_coefficients.as_ref().unwrap().equations, Equations::List(vec![1, 2]));
    assert!(d.chains[1].svar_variances.is_none());
}

#[test]
fn declaration_variants_and_errors() {
    let base = "var y1 y2 y3; varexo e; model; y1=y1(-1)+e; y2=y1; y3=y2; end; varobs y1 y2 y3;\n";
    let parse = |extra: &str| MsDecl::from_modfile(&parse_mod_str(&format!("{base}{extra}"), &[]).unwrap());
    // vector durations (accepted here though the 7.1 preprocessor aborts on them) and Cholesky
    assert!(parse("svar_identification; lower_cholesky; exclusion lag 1; equation 1, y1; end;").is_err());
    let d = parse("svar_identification; lower_cholesky; end;\nmarkov_switching(chain=1, duration=[0.5, 2.5, 2.5]);")
        .unwrap();
    assert!(d.identification.unwrap().lower_cholesky);
    assert_eq!(d.chains[0].number_of_regimes, 3);
    assert_eq!(d.chains[0].durations, vec![0.5, 2.5, 2.5]);
    // non-consecutive chain, bad probability rows, unknown chain in svar
    assert!(parse("markov_switching(chain=2, number_of_regimes=2, duration=2);").is_err());
    assert!(
        parse("markov_switching(chain=1, number_of_regimes=3, duration=2, restrictions=[[1,3,0],[3,3,1]]);").is_err()
    );
    assert!(
        parse("markov_switching(chain=1, number_of_regimes=2, duration=2, restrictions=[[1,2,0.3],[1,2,0.1]]);")
            .is_err()
    );
    assert!(parse("svar(variances, chain=3);").is_err());
    assert!(
        parse("markov_switching(chain=1, number_of_regimes=2, duration=2); svar(variances, coefficients, chain=1);")
            .is_err()
    );
    // unknown variable in an exclusion
    let e = SvarIdentification::parse(&["exclusion lag 1".into(), "equation 1 , zz".into()], &["y1".into()]);
    assert!(e.is_err());
}

// ---------------------------------------------------------------------------------------
// `restriction equation` lines (linear restrictions within A0 / within A+), against Dynare
// (reference/msbvar/ref_msbvar_lin.json from dump_msbvar_lin.m).

#[test]
fn sbvar_linear_restrictions() {
    run_case("msbvar_lin", &refs_lin());
}

#[test]
fn sbvar_linear_restrictions_constants_hyperparameters() {
    run_case("msbvar_lin2", &refs_lin());
}

fn ident_of(text: &str) -> anyhow::Result<SvarIdentification> {
    let mf = parse_mod_str(text, &[])?;
    Ok(MsDecl::from_modfile(&mf)?.identification.unwrap())
}

#[test]
fn restriction_lines_give_the_preprocessor_qi_ri() {
    // options_.ms.Qi / Ri written by the 7.1 preprocessor for msbvar_lin.mod (a = 0.5)
    let id = ident_of(&std::fs::read_to_string(root().join("reference/msbvar/msbvar_lin.mod")).unwrap()).unwrap();
    let (qi, ri) = id.qi_ri_values(3);
    assert_eq!(qi[0], vec![(1, 3, 1.0), (2, 1, 1.0), (2, 2, -2.0)]);
    assert!(qi[1].is_empty());
    assert_eq!(qi[2], vec![(1, 2, -0.5), (1, 1, -1.0), (1, 3, 1.0)]);
    assert!(ri[0].is_empty());
    assert_eq!(ri[1], vec![(1, 1, 1.0), (1, 2, 1.0), (2, 6, 1.0)]);
    assert_eq!(ri[2], vec![(1, 6, 0.5), (1, 3, -1.0)]);
    assert_eq!(id.max_lag(), Some(2));
    let id2 = ident_of(&std::fs::read_to_string(root().join("reference/msbvar/msbvar_lin2.mod")).unwrap()).unwrap();
    let (qi, ri) = id2.qi_ri_values(3);
    assert_eq!(qi[1], vec![(1, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0)]);
    assert_eq!(qi[2], vec![(1, 1, 1.0)]);
    assert_eq!(ri[0], vec![(1, 2, 1.0), (1, 5, -0.3)]);
    assert_eq!(ri[2], vec![(1, 1, 2.0), (1, 3, -1.0)]);
    assert!(id2.constants_exclusion);
    // more preprocessor forms: signs on both sides
    let base = "var y1 y2 y3; varexo e; model; y1=y1(-1)+e; y2=y1; y3=y2; end; varobs y1 y2 y3;\n";
    let id = ident_of(&format!(
        "{base}svar_identification; exclusion lag 0; equation 1, y3; restriction equation 1, - 2*coeff(y1,0) = + 3*coeff(y2,0) - coeff(y3,0); end;"
    ))
    .unwrap();
    assert_eq!(id.qi_ri_values(3).0[0], vec![(1, 3, 1.0), (2, 1, -2.0), (2, 2, -3.0), (2, 3, 1.0)]);
    // preprocessor errors: lag reused by a later exclusion group, inhomogeneous, Cholesky mix
    assert!(ident_of(&format!("{base}svar_identification; restriction equation 1, coeff(y1,0) = 0.5*coeff(y2,0); exclusion lag 0; equation 1, y3; end;")).is_err());
    assert!(
        ident_of(&format!(
            "{base}svar_identification; exclusion lag 0; equation 1, y3; exclusion lag 0; equation 2, y2; end;"
        ))
        .is_err()
    );
    assert!(ident_of(&format!("{base}svar_identification; restriction equation 1, coeff(y1,1) = 1; end;")).is_err());
    assert!(
        ident_of(&format!("{base}svar_identification; upper_cholesky; restriction equation 1, coeff(y1,1) = 0; end;"))
            .is_err()
    );
    // Qi and Ri mixed: kept as a cross restriction (sbvar rejects it without cross_restrictions)
    assert!(
        ident_of(&format!("{base}svar_identification; restriction equation 1, coeff(y1,1) = coeff(y1,0); end;"))
            .unwrap()
            .has_cross()
    );
}

/// The msbvar_lin model plus two cross A0-A+ restriction lines.
const CROSS_LINES: &str = "restriction equation 2, coeff(y1,1) = 0.5*coeff(y2,0);\n\
                           restriction equation 3, coeff(y3,1) - coeff(y1,2) = 0.2*coeff(y3,0) + coeff(y1,0);\n";

fn lin_mod(extra_lines: &str, sbvar_opts: &str) -> String {
    let src = std::fs::read_to_string(root().join("reference/msbvar/msbvar_lin.mod")).unwrap();
    src.replace("end;\n\nsbvar(", &format!("{extra_lines}end;\n\nsbvar("))
        .replace("nlags=2);", &format!("nlags=2{sbvar_opts});"))
}

fn run_text(text: &str) -> anyhow::Result<dynare::msbvar::sbvar::SbvarResult> {
    let mf = parse_mod_str(text, &[])?;
    let Some(Command::Other(_, o)) = mf.commands.iter().find(|c| matches!(c, Command::Other(n, _) if n == "sbvar"))
    else {
        panic!("no sbvar command")
    };
    Ok(msbvar::run_sbvar(&mf, o, &root().join("reference/msbvar"))?.0)
}

fn null_space(c: &Mat) -> Mat {
    let n = c.ncols();
    let mut p = Mat::zeros(c.nrows().max(n), n);
    p.rows_mut(0, c.nrows()).copy_from(c);
    let svd = p.svd(false, true);
    let vt = svd.v_t.unwrap();
    let free: Vec<usize> = (0..n).filter(|&i| svd.singular_values[i] < 1e-10).collect();
    Mat::from_fn(n, free.len(), |r, k| vt[(free[k], r)])
}

/// `f_j(i)` weights `rf` (row of A+), `a_j(h)` weights `ra`: `sum rf f_j = sum ra a_j`.
type CrossRow = (usize, Vec<(usize, f64)>, Vec<(usize, f64)>);

/// msstart2.m's cross-restriction step: `A+_j` given `A0_j` from the unrestricted posterior.
fn msstart2_step(r0: &dynare::msbvar::sbvar::SbvarResult, row: &CrossRow, a0: &Mat) -> Mat {
    let (j, rf, ra) = row;
    let j = *j;
    let b = r0.ui[j].transpose() * a0.columns(j, 1);
    let fur = &r0.vi[j] * &r0.pmat[j] * &b; // Fhatur0P(:,ixeq)
    let lit = Mat::from_fn(1, r0.vi[j].ncols(), |_, c| rf.iter().map(|(i, x)| x * r0.vi[j][(*i, c)]).sum());
    let ci = ra.iter().map(|(h, s)| s * a0[(*h, j)]).sum::<f64>();
    let lth = &lit * r0.hpinv[j].clone().try_inverse().unwrap(); // LtH = Lit/Hpinv
    let hlv = lth.transpose() / (&lth * lit.transpose())[(0, 0)];
    let gihat = r0.vi[j].transpose() * &fur;
    &r0.vi[j] * (&gihat + &hlv * (ci - (&lit * &gihat)[(0, 0)]))
}

/// Dynare 7.1 errors on `cross_restrictions` (msstart2.m line 415, `ixmC0Pres{ki}` on the NaN
/// that exclusions.m returns: "Brace indexing is not supported for variables of this type"),
/// so the restricted mode is checked against a direct computation: the joint posterior of
/// (b_j, g_j) built from the unrestricted posterior matrices, maximised on the restriction set.
#[test]
fn sbvar_cross_restrictions_maximise_the_restricted_posterior() {
    // mixed lines need the option (the preprocessor rejects them outright)
    assert!(run_text(&lin_mod(CROSS_LINES, "")).is_err());
    // the option without mixed lines changes nothing
    let r0 = run_text(&lin_mod("", "")).unwrap();
    let r0c = run_text(&lin_mod("", ", cross_restrictions")).unwrap();
    assert!((&r0.a0 - &r0c.a0).abs().max() < 1e-12 && (&r0.aplus - &r0c.aplus).abs().max() < 1e-12);
    let r = run_text(&lin_mod(CROSS_LINES, ", cross_restrictions")).unwrap();
    assert!(r.vlog_a0p.is_none());
    let n = r.nvar;
    for k in 0..n {
        assert!((&r.ui[k] - &r0.ui[k]).abs().max() < 1e-14 && (&r.vi[k] - &r0.vi[k]).abs().max() < 1e-14);
    }
    // 1. the restrictions hold at the mode
    let (a0, ap) = (&r.a0, &r.aplus);
    let c2 = ap[(0, 1)] - 0.5 * a0[(1, 1)];
    let c3 = ap[(2, 2)] - ap[(3, 2)] - 0.2 * a0[(2, 2)] - a0[(0, 2)];
    assert!(c2.abs() < 1e-10 && c3.abs() < 1e-10, "{c2} {c3}");
    assert!(ap[(0, 1)].abs() > 1e-3, "the restriction should bind on a non-trivial value");
    // 2. A+ given A0 is msstart2.m's cross-restriction formula on the unrestricted posterior
    let rows: Vec<CrossRow> =
        vec![(1, vec![(0, 1.0)], vec![(1, 0.5)]), (2, vec![(2, 1.0), (3, -1.0)], vec![(2, 0.2), (0, 1.0)])];
    for row in &rows {
        close_mat(&msstart2_step(&r0, row, a0), &ap.columns(row.0, 1).into_owned(), 1e-9, "msstart2 formula");
    }
    // 3. stationarity and maximum of the joint log posterior on the restriction set
    let n0: Vec<usize> = r0.ui.iter().map(|u| u.ncols()).collect();
    let np: Vec<usize> = r0.vi.iter().map(|v| v.ncols()).collect();
    let off: Vec<usize> = (0..=n).map(|j| (0..j).map(|i| n0[i] + np[i]).sum()).collect();
    let dim = off[n];
    let pack = |a0: &Mat, ap: &Mat| -> Vec<f64> {
        let mut th = vec![0.0; dim];
        for j in 0..n {
            let b = r0.ui[j].transpose() * a0.column(j);
            let g = r0.vi[j].transpose() * ap.column(j);
            th[off[j]..off[j] + n0[j]].copy_from_slice(b.as_slice());
            th[off[j] + n0[j]..off[j + 1]].copy_from_slice(g.as_slice());
        }
        th
    };
    let logpost = |th: &[f64]| -> f64 {
        let mut a = Mat::zeros(n, n);
        let mut l = 0.0;
        for j in 0..n {
            let b = Mat::from_column_slice(n0[j], 1, &th[off[j]..off[j] + n0[j]]);
            let g = Mat::from_column_slice(np[j], 1, &th[off[j] + n0[j]..off[j + 1]]);
            a.set_column(j, &(&r0.ui[j] * &b).column(0));
            let d = &g - &r0.pmat[j] * &b;
            l -= 0.5 * ((b.transpose() * &r0.h0inv[j] * &b)[(0, 0)] + (d.transpose() * &r0.hpinv[j] * &d)[(0, 0)]);
        }
        l + r0.fss as f64 * a.determinant().abs().ln()
    };
    let mut cm = Mat::zeros(rows.len(), dim);
    for (k, (j, rf, ra)) in rows.iter().enumerate() {
        let j = *j;
        for (i, x) in rf {
            for c in 0..np[j] {
                cm[(k, off[j] + n0[j] + c)] += x * r0.vi[j][(*i, c)];
            }
        }
        for (h, s) in ra {
            for c in 0..n0[j] {
                cm[(k, off[j] + c)] -= s * r0.ui[j][(*h, c)];
            }
        }
    }
    let theta = pack(a0, ap);
    assert!((&cm * Mat::from_column_slice(dim, 1, &theta)).abs().max() < 1e-10);
    let ns = null_space(&cm);
    assert_eq!(ns.ncols(), dim - rows.len());
    let l0 = logpost(&theta);
    let h = 1e-4;
    for c in 0..ns.ncols() {
        let tp: Vec<f64> = (0..dim).map(|i| theta[i] + h * ns[(i, c)]).collect();
        let tm: Vec<f64> = (0..dim).map(|i| theta[i] - h * ns[(i, c)]).collect();
        let (lp, lm) = (logpost(&tp), logpost(&tm));
        let d = (lp - lm) / (2.0 * h);
        assert!(d.abs() < 1e-5, "directional derivative {c}: {d}");
        assert!(lp < l0 && lm < l0, "not a maximum along {c}");
    }
    // msstart2.m's shortcut (unrestricted A0 mode, then the A+ step) is feasible but lower
    let mut ap_dyn = &r0.aplus * 1.0;
    for row in &rows {
        ap_dyn.set_column(row.0, &msstart2_step(&r0, row, &r0.a0).column(0));
    }
    let th_dyn = pack(&r0.a0, &ap_dyn);
    assert!((&cm * Mat::from_column_slice(dim, 1, &th_dyn)).abs().max() < 1e-10);
    assert!(logpost(&th_dyn) < l0);
}
