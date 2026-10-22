//! : numerical checks of the transformed models of reference/pp/*.mod against Dynare 7.2
//! (reference/pp/dump_pp.m -> ref_<name>.json): dynamic residuals and Jacobian at a perturbed
//! point with the steady state given to `STEADY_STATE()`, static residuals and Jacobian, the
//! first-order decision rules and a perfect-foresight simulation. The structures (names, aux
//! variables, incidence, sparsity) are checked in tests/preprocessor_sw07.rs.

use dynare::linalg::Mat;
use dynare::modfile::parse_mod_file;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::Null => vec![f64::NAN],
        _ => panic!("not numeric: {v}"),
    }
}

/// A MATLAB matrix as jsonencode writes it (array of rows; a vector may be flat).
fn mat(v: &Value, nr: usize, nc: usize) -> Mat {
    let f = vecf(v);
    assert_eq!(f.len(), nr * nc);
    Mat::from_row_slice(nr, nc, &f)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

fn load_in(dir: &str, name: &str) -> Option<Value> {
    let p = root().join(format!("reference/{dir}/ref_{name}.json"));
    std::fs::read_to_string(p).ok().map(|s| serde_json::from_str(&s).unwrap())
}

/// Checks the models `names` of `reference/<dir>/` against their `ref_<name>.json` (/// reuses it for reference/q1); returns how many were checked.
pub fn check_numerics(dir: &str, names: &[&str]) -> usize {
    let mut checked = 0;
    for &name in names {
        let Some(r) = load_in(dir, name) else { continue };
        let mf = parse_mod_file(&root().join(format!("reference/{dir}/{name}.mod")), &[]).unwrap();
        let m = &mf.model;
        let (n, nx, neq) = (m.n_endo(), m.n_exo(), m.n_equations());
        let names: Vec<String> =
            r["endo_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        assert_eq!(m.endo_names, names, "{name}");
        let params = vecf(&r["params"]);
        let ys = vecf(&r["ys"]);
        let y3 = vecf(&r["y3"]);
        let x = vecf(&r["x"]);
        let pe = m.params_with_steady_state(&params, &ys);
        // dynamic residuals
        let mut res = vec![0.0; neq];
        m.dynamic_residuals(&y3, &x, &pe, &mut res);
        for (i, (a, b)) in res.iter().zip(vecf(&r["dynamic_resid"])).enumerate() {
            assert!(close(*a, b, 1e-12), "{name}: dynamic residual {} = {a} vs {b}: {}", i + 1, m.equation_string(i));
        }
        // dynamic Jacobian
        let g1r = mat(&r["dynamic_g1"], neq, 3 * n + nx);
        let mut g1 = Mat::zeros(neq, 3 * n + nx);
        m.dynamic_jacobian_y3_with(&y3, &x, &pe, &mut m.workspace(), &mut g1);
        for i in 0..neq {
            for j in 0..3 * n + nx {
                assert!(
                    close(g1[(i, j)], g1r[(i, j)], 1e-11),
                    "{name}: g1({},{}) = {} vs {}",
                    i + 1,
                    j + 1,
                    g1[(i, j)],
                    g1r[(i, j)]
                );
            }
        }
        // static model
        let yst = vecf(&r["ystat"]);
        let mut sres = vec![0.0; neq];
        m.static_residuals(&yst, &x, &params, &mut sres);
        for (i, (a, b)) in sres.iter().zip(vecf(&r["static_resid"])).enumerate() {
            assert!(close(*a, b, 1e-12), "{name}: static residual {} = {a} vs {b}", i + 1);
        }
        let sg = m.static_jacobian(&yst, &x, &params);
        let sgr = mat(&r["static_g1"], neq, n);
        for i in 0..neq {
            for j in 0..n {
                assert!(close(sg[(i, j)], sgr[(i, j)], 1e-11), "{name}: static g1({},{})", i + 1, j + 1);
            }
        }
        // first-order decision rules
        if let Some(ghx) = r.get("ghx") {
            let sol = dynare::analysis::solve_calibration(&mf, 1.0 + 1e-6).unwrap().unwrap();
            for (a, b) in sol.dr.ys.iter().zip(&ys) {
                assert!(close(*a, *b, 1e-9), "{name}: steady state {a} vs {b}");
            }
            let nsp = sol.dr.ghx.ncols();
            let ghxr = mat(ghx, n, nsp);
            let ghur = mat(&r["ghu"], n, nx);
            for i in 0..n {
                for j in 0..nsp {
                    assert!(close(sol.dr.ghx[(i, j)], ghxr[(i, j)], 1e-9), "{name}: ghx({},{})", i + 1, j + 1);
                }
                for j in 0..nx {
                    assert!(close(sol.dr.ghu[(i, j)], ghur[(i, j)], 1e-9), "{name}: ghu({},{})", i + 1, j + 1);
                }
            }
        }
        // second order (STEADY_STATE terms in the second derivatives)
        if let Some(ghxx) = r.get("ghxx") {
            let k2 = dynare::solver::korder::solve_model(m, &ys, &params, &mf.sigma_e, 2, Default::default()).unwrap();
            for (key, ours) in [("ghxx", &k2.ghxx), ("ghxu", &k2.ghxu), ("ghuu", &k2.ghuu), ("ghs2", &k2.ghs2)] {
                let theirs = mat(if key == "ghxx" { ghxx } else { &r[key] }, ours.nrows(), ours.ncols());
                for i in 0..ours.nrows() {
                    for j in 0..ours.ncols() {
                        assert!(
                            close(ours[(i, j)], theirs[(i, j)], 1e-8),
                            "{name}: {key}({},{}) {} vs {}",
                            i + 1,
                            j + 1,
                            ours[(i, j)],
                            theirs[(i, j)]
                        );
                    }
                }
            }
        }
        // perfect foresight
        if let Some(es) = r.get("endo_simul") {
            let tmp = root().join(format!("target/tmp/{dir}_det"));
            std::fs::create_dir_all(&tmp).unwrap();
            std::fs::copy(root().join(format!("reference/{dir}/{name}.mod")), tmp.join(format!("{name}.mod"))).unwrap();
            dynare::driver::run(
                &tmp.join(format!("{name}.mod")),
                &dynare::driver::RunOptions { quiet: true, ..Default::default() },
            )
            .unwrap();
            let csv = std::fs::read_to_string(tmp.join(format!("{name}/Output/{name}_endo_simul.csv"))).unwrap();
            let rows: Vec<Vec<f64>> = csv
                .lines()
                .skip(1)
                .map(|l| l.split(',').skip(1).map(|v| v.trim().parse().unwrap()).collect())
                .collect();
            let t = rows.len();
            let er = mat(es, n, vecf(es).len() / n);
            assert!(er.ncols() >= t, "{name}: {} periods vs {t}", er.ncols());
            let off = er.ncols() - t;
            let mut worst: f64 = 0.0;
            for (k, row) in rows.iter().enumerate() {
                for j in 0..row.len().min(n) {
                    worst = worst.max((row[j] - er[(j, k + off)]).abs());
                }
            }
            assert!(worst < 1e-8, "{name}: perfect foresight paths differ by {worst}");
        }
        checked += 1;
    }
    checked
}

#[test]
fn pp_models_numerics_match_dynare() {
    let checked = check_numerics(
        "pp",
        &["pp_ss", "pp_leads", "ppdet_leads", "pp_expect", "pp_const", "pp_static", "pp_edit", "pp_ss2"],
    );
    assert!(checked >= 8, "reference/pp/ref_*.json missing");
}

// ---- var(log), deflators, trend_var (reference/q1)
#[test]
fn q1_models_numerics_match_dynare() {
    let checked = check_numerics("q1", &["q1_log", "q1_logexp", "q1_ns2", "q1_fsns"]);
    assert_eq!(checked, 4, "reference/q1/ref_*.json missing");
}

/// `STEADY_STATE()` in estimation: the operator is re-evaluated at every draw's steady state.
/// `r = phi*STEADY_STATE(y)*y(-1)` with `STEADY_STATE(y) = mu` must give the likelihood of
/// `r = phi*mu*y(-1)` at every parameter vector (mu moves the steady state).
#[test]
fn steady_state_operator_in_estimation() {
    use dynare::data::Table;
    use dynare::dsge::{DsgeModel, DsgePosterior, LikelihoodData};
    use dynare::estimation::Objective;
    use dynare::kalman::KalmanOptions;
    let dir = root().join("target/tmp/pp_est");
    std::fs::create_dir_all(&dir).unwrap();
    let mut csv = String::from("y,r\n");
    let mut s: u64 = 12345;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    for _ in 0..60 {
        csv.push_str(&format!("{},{}\n", 1.5 + 0.3 * rnd(), 1.0 + 0.4 * rnd()));
    }
    std::fs::write(dir.join("data.csv"), csv).unwrap();
    let model = |eq: &str| {
        format!(
            "var y r;\nvarexo e u;\nparameters rho mu phi;\nrho = 0.8; mu = 1.5; phi = 0.3;\nmodel;\n\
             y = mu*(1-rho) + rho*y(-1) + e;\n{eq}\nend;\n\
             steady_state_model;\ny = mu;\nr = phi*mu*mu/(1-0.5);\nend;\n\
             shocks; var e; stderr 0.1; var u; stderr 0.2; end;\nvarobs y r;\n\
             estimated_params;\nrho, 0.8, 0, 0.99;\nmu, 1.5, -10, 10;\nphi, 0.3, -5, 5;\nend;\n\
             estimation(datafile='data.csv', mode_compute=0);\n"
        )
    };
    let mut f = Vec::new();
    for (k, eq) in
        ["r = phi*STEADY_STATE(y)*y(-1) + 0.5*r(-1) + u;", "r = phi*mu*y(-1) + 0.5*r(-1) + u;"].iter().enumerate()
    {
        let p = dir.join(format!("m{k}.mod"));
        std::fs::write(&p, model(eq)).unwrap();
        let mf = parse_mod_file(&p, &[]).unwrap();
        assert_eq!(mf.model.uses_steady_state(), k == 0);
        let dsge = DsgeModel::new(&mf, None).unwrap();
        let table = Table::read_csv(&dir.join("data.csv")).unwrap();
        let data = LikelihoodData::new(&table.sample(&mf.varobs, 1, None).unwrap(), KalmanOptions::default());
        let post = DsgePosterior { dsge: &dsge, data: &data, bounds: &dsge.bounds };
        f.push(
            [[0.8, 1.5, 0.3], [0.5, 2.5, -0.4], [0.9, -1.0, 1.2]].iter().map(|x| post.eval(x).0).collect::<Vec<f64>>(),
        );
    }
    for (a, b) in f[0].iter().zip(&f[1]) {
        assert!(a.is_finite() && close(*a, *b, 1e-12), "{a} vs {b}");
    }
}
