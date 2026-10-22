//! Heterogeneous-agent models (Dynare 7), part 1, against Dynare 7.2
//! (reference/heterogeneity/ref_ks.json and ref_ksl.json, written by
//! reference/heterogeneity/dump.m from ks.mod and ksl.mod):
//! - the preprocessor structures `M_.heterogeneity(1)`, `M_.heterogeneity_aggregates`, the
//!   aggregation auxiliary variable of `SUM()`, the multiplier and non-linear-lead auxiliary
//!   variables, the sparse Jacobian structure, the complementarity reordering, the levels of the
//!   auxiliary variables, and the values of the generated `dynamic_het1_*` / `dynamic_*` code;
//! - `heterogeneity_load_steady_state` (sizes, indices, `mat`, residuals) on the Krusell-Smith
//!   example's MAT file;
//! - `heterogeneity_compute_steady_state` with a calibrated discount factor: parameter,
//!   policy functions, distribution, aggregates;
//! - `rouwenhorst`.

use std::path::{Path, PathBuf};

use dynare::heterogeneity::{
    HetOptions, SteadyStateInput, rouwenhorst, run_compute_steady_state, run_load_steady_state,
};
use dynare::matfile::MatValue;
use dynare::modfile::{ModFile, parse_mod_file};
use serde_json::Value;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/heterogeneity")
}

fn reference(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join(name)).unwrap()).unwrap()
}

/// Flattens nested JSON arrays (row-major as `jsonencode`) and scalars.
fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Null => vec![f64::NAN],
        x => vec![x.as_f64().unwrap()],
    }
}

/// A MATLAB matrix from `jsonencode` as rows (a vector is a column or a row of one element).
fn rows(v: &Value) -> Vec<Vec<f64>> {
    match v {
        Value::Array(a) if a.iter().all(|x| x.is_array()) => a.iter().map(flat).collect(),
        Value::Array(a) => a.iter().map(|x| vec![x.as_f64().unwrap_or(f64::NAN)]).collect(),
        x => vec![vec![x.as_f64().unwrap()]],
    }
}

fn strings(v: &Value) -> Vec<String> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
        Value::String(s) => vec![s.clone()],
        _ => vec![],
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs())
}

fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w, tol), "{what}[{k}]: got {g}, want {w}");
    }
}

fn ints(v: &[i32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn model(name: &str) -> ModFile {
    parse_mod_file(&dir().join(name), &[]).unwrap_or_else(|e| panic!("{name}: {e:#}"))
}

fn check_preprocessor(mf: &ModFile, r: &Value) {
    let m = &mf.model;
    assert_eq!(m.endo_names, strings(&r["endo_names"]));
    assert_eq!(m.orig_endo_nbr as f64, r["orig_endo_nbr"].as_f64().unwrap());
    assert_eq!(m.exo_names, strings(&r["exo_names"]));
    assert_eq!(m.param_names, strings(&r["param_names"]));
    // lead_lag_incidence: MATLAB has [lag; current; lead] x endo
    let want = rows(&r["lead_lag_incidence"]);
    for (j, l) in m.lead_lag_incidence.iter().enumerate() {
        for t in 0..3 {
            // Dynare drops the lead row when no variable has a lead
            let w = if t < want.len() { want[t][j] } else { 0.0 };
            assert_eq!(l[t] as f64, w, "lead_lag_incidence[{t}][{j}]");
        }
    }
    // M_.heterogeneity_aggregates and the aggregation auxiliary variables
    // (an n x 3 cell is written column-major: n ops, n dimensions, n indices; a single
    // row as ['sum', 1, 2])
    let agg: Vec<&Value> = r["heterogeneity_aggregates"].as_array().unwrap().iter().collect();
    let na = mf.het.aggregates.len();
    assert_eq!(agg.len(), 3 * na);
    for (k, a) in mf.het.aggregates.iter().enumerate() {
        assert_eq!(agg[k].as_str().unwrap(), a.op);
        assert_eq!(agg[na + k].as_f64().unwrap() as usize, a.dimension + 1);
        assert_eq!(agg[2 * na + k].as_f64().unwrap() as usize, a.het_endo + 1);
    }
    // M_.aux_vars: the aggregation variables (type 14) come first in Dynare
    let aux: Vec<&Value> = match &r["aux_vars"] {
        Value::Array(a) => a.iter().collect(),
        x => vec![x],
    };
    for (k, a) in mf.het.aggregate_aux.iter().enumerate() {
        assert_eq!(aux[k]["endo_index"].as_f64().unwrap() as usize, a.endo_index + 1);
        assert_eq!(aux[k]["type"].as_f64().unwrap() as u8, a.ty);
        assert_eq!(aux[k]["orig_expr"].as_str().unwrap(), a.orig_expr);
    }
    // the other auxiliary variables of the aggregate model follow
    for (k, a) in m.aux_vars.iter().enumerate() {
        let w = aux[na + k];
        assert_eq!(w["endo_index"].as_f64().unwrap() as usize, a.endo_index + 1, "aux_vars({})", na + k + 1);
        assert_eq!(w["type"].as_f64().unwrap() as u8, a.ty, "aux_vars({}) type", na + k + 1);
    }
    assert_eq!(aux.len(), na + m.aux_vars.len());
    // aggregate equation tags and sparse structure of the aggregate model
    let tags = flat_tags(&r["equations_tags"]);
    let got: Vec<String> = m
        .equation_tags
        .iter()
        .enumerate()
        .flat_map(|(i, t)| t.iter().filter(|(k, _)| k == "name").map(move |(_, v)| format!("{}:{v}", i + 1)))
        .collect();
    assert_eq!(got, tags);
    // M_.heterogeneity(1)
    let h = mf.het.model.as_ref().expect("heterogeneous model");
    let hr = &r["het"];
    assert_eq!(hr["dimension_name"].as_str().unwrap(), h.dimension_name);
    assert_eq!(h.endo_names, strings(&hr["endo_names"]));
    assert_eq!(h.endo_names_tex, strings(&hr["endo_names_tex"]));
    assert_eq!(h.endo_names_long, strings(&hr["endo_names_long"]));
    assert_eq!(h.orig_endo_nbr as f64, hr["orig_endo_nbr"].as_f64().unwrap());
    assert_eq!(h.exo_names, strings(&hr["exo_names"]));
    assert_eq!(h.exo_names_tex, strings(&hr["exo_names_tex"]));
    assert_eq!(h.exo_names_long, strings(&hr["exo_names_long"]));
    assert_eq!(h.param_names.len() as f64, hr["param_nbr"].as_f64().unwrap());
    let hav: Vec<&Value> = match &hr["aux_vars"] {
        Value::Array(a) => a.iter().collect(),
        x => vec![x],
    };
    assert_eq!(h.aux_vars.len(), hav.len());
    for (a, w) in h.aux_vars.iter().zip(hav) {
        assert_eq!(a.endo_index + 1, w["endo_index"].as_f64().unwrap() as usize);
        assert_eq!(a.ty as f64, w["type"].as_f64().unwrap());
        assert_eq!(a.eq_nbr as f64, w["eq_nbr"].as_f64().unwrap());
    }
    assert_close("Sigma_e", h.sigma_e.as_slice(), &flat(&hr["Sigma_e"]), 1e-14);
    assert_eq!(h.state_var.iter().map(|&s| (s + 1) as f64).collect::<Vec<_>>(), flat(&hr["state_var"]));
    assert_eq!(ints(&h.dynamic_g1_sparse_rowval), flat(&hr["dynamic_g1_sparse_rowval"]));
    assert_eq!(ints(&h.dynamic_g1_sparse_colval), flat(&hr["dynamic_g1_sparse_colval"]));
    assert_eq!(ints(&h.dynamic_g1_sparse_colptr), flat(&hr["dynamic_g1_sparse_colptr"]));
    assert_eq!(
        h.mcp_equations_reordering.iter().map(|&s| (s + 1) as f64).collect::<Vec<_>>(),
        flat(&hr["dynamic_mcp_equations_reordering"])
    );
    assert_eq!(h.set_auxiliary_variables, hr["set_auxiliary_variables"].as_bool().unwrap());
    assert_eq!(h.het_aux_levels.len() as f64, hr["n_aux_levels"].as_f64().unwrap());
    let levels: Vec<f64> = h.het_aux_levels.iter().flatten().map(|&x| (x + 1) as f64).collect();
    assert_eq!(levels, flat(&hr["het_aux_levels"]));
    let htags = flat_tags(&hr["equations_tags"]);
    let got: Vec<String> = h
        .equations_tags
        .iter()
        .enumerate()
        .flat_map(|(i, t)| t.iter().filter(|(k, _)| k == "name").map(move |(_, v)| format!("{}:{v}", i + 1)))
        .collect();
    assert_eq!(got, htags);
}

/// `{[1],[2],'name','name','a','b'}` as written by `jsonencode`: first the equation numbers, then
/// the tag keys, then the values; returns `"1:a"` strings.
fn flat_tags(v: &Value) -> Vec<String> {
    let a: Vec<&Value> = match v {
        Value::Array(a) => a.iter().collect(),
        _ => return vec![],
    };
    if a.is_empty() {
        return vec![];
    }
    // jsonencode of an n x 3 cell is column-major: n numbers, n keys, n values
    let n = a.len() / 3;
    (0..n)
        .filter(|&i| a[n + i].as_str() == Some("name"))
        .map(|i| format!("{}:{}", a[i].as_f64().unwrap() as usize, a[2 * n + i].as_str().unwrap()))
        .collect()
}

fn check_generated_code(mf: &ModFile, r: &Value) {
    let h = mf.het.model.as_ref().unwrap();
    let m = &mf.model;
    let e = &r["eval"];
    let (nh, na) = (h.endo_nbr(), m.n_endo());
    // (the reference was written after the calibration: M_.params at the end of the run)
    let params = flat(&r["params_after"]);
    let mut ws = h.workspace(&params);
    let yh = flat(&e["yh"]);
    let y = flat(&e["y"]);
    let xh = flat(&e["xh"]);
    let x = flat(&e["x"]);
    ws.yh_mut().copy_from_slice(&yh);
    ws.y_agg_mut().copy_from_slice(&y);
    ws.xh_mut().copy_from_slice(&xh);
    ws.x_agg_mut().copy_from_slice(&x);
    // dynamic_het1_resid
    let mut resid = vec![0.0; nh];
    h.resid(&mut ws, &mut resid);
    assert_close("dynamic_het1_resid", &resid, &flat(&e["het_resid"]), 1e-13);
    // dynamic_het1_g1 (dense, row-major rows)
    let g1 = h.jacobian_dense(&mut ws);
    let want = rows(&e["het_g1"]);
    assert_eq!(g1.ncols(), want[0].len());
    for i in 0..nh {
        for j in 0..g1.ncols() {
            assert!(
                close(g1[(i, j)], want[i][j], 1e-13),
                "dynamic_het1_g1[{i}][{j}]: {} vs {}",
                g1[(i, j)],
                want[i][j]
            );
        }
    }
    // dynamic_het1_set_auxiliary_variables
    for step in 0..h.het_aux_levels.len() {
        let mut w2 = h.workspace(&params);
        w2.yf.copy_from_slice(&ws.yf);
        w2.xf.copy_from_slice(&ws.xf);
        // the generated function is applied to the original yh at every step
        h.set_auxiliary_variables(&mut w2, step);
        assert_close(&format!("set_aux step {step}"), w2.yh(), &flat(&e[format!("het_aux_step{step}")]), 1e-13);
    }
    // bounds
    let (lb, ub) = h.complementarity_bounds(&params);
    let wl = flat(&e["lb"]);
    let wu = flat(&e["ub"]);
    for i in 0..nh {
        // (jsonencode writes +-Inf as null)
        if wl[i].is_nan() {
            assert!(lb[i].is_infinite() && lb[i] < 0.0, "lb[{i}] = {}", lb[i]);
        } else {
            assert!(close(lb[i], wl[i], 1e-14), "lb[{i}] {} vs {}", lb[i], wl[i]);
        }
        if wu[i].is_nan() {
            assert!(ub[i].is_infinite() && ub[i] > 0.0, "ub[{i}] = {}", ub[i]);
        } else {
            assert!(close(ub[i], wu[i], 1e-14), "ub[{i}] {} vs {}", ub[i], wu[i]);
        }
    }
    // aggregate model with yagg
    let yagg = flat(&e["yagg"]);
    let mut ar = vec![0.0; m.n_equations()];
    mf.het.agg_residuals(m, &y, &x, &params, &yagg, &mut ar);
    assert_close("dynamic_resid", &ar, &flat(&e["agg_resid"]), 1e-13);
    // set_auxiliary_variables
    let mut ys = flat(&e["set_aux_in"]);
    mf.het.set_auxiliary_variables(m, &mut ys, &vec![0.0; m.n_exo()], &params, &yagg);
    assert_close("set_auxiliary_variables", &ys, &flat(&e["set_aux_out"]), 1e-13);
    let _ = na;
}

#[test]
fn ks_preprocessor_structures() {
    let r = reference("ref_ks.json");
    let mf = model("ks.mod");
    check_preprocessor(&mf, &r);
    check_generated_code(&mf, &r);
}

#[test]
fn ksl_preprocessor_structures() {
    let r = reference("ref_ksl.json");
    let mf = model("ksl.mod");
    check_preprocessor(&mf, &r);
    check_generated_code(&mf, &r);
}

fn mat_input(path: &Path, var: &str) -> SteadyStateInput {
    let all = dynare::matfile::read(path).unwrap();
    let v: &MatValue = all.get(var).expect("variable in MAT file");
    SteadyStateInput::from_mat(v).unwrap()
}

fn check_steady_state(s: &dynare::heterogeneity::LoadedSteadyState, r: &Value, tol: f64) {
    let rs = &r["sizes"];
    assert_eq!(s.sizes.n_e_total as f64, rs["N_e"].as_f64().unwrap());
    assert_eq!(s.sizes.n_sp as f64, rs["N_sp"].as_f64().unwrap());
    assert_eq!(s.sizes.n_om as f64, rs["N_om"].as_f64().unwrap());
    assert_eq!(s.sizes.pol_n_a as f64, rs["pol"]["N_a"].as_f64().unwrap());
    assert_eq!(s.sizes.d_n_a as f64, rs["d"]["N_a"].as_f64().unwrap());
    assert_eq!(s.sizes.n_pol as f64, rs["n_pol"].as_f64().unwrap());
    let ri = &r["indices"];
    assert_eq!(s.indices.base.states, strings(&ri["states"]));
    assert_eq!(s.indices.base.shocks, strings(&ri["shocks"]));
    assert_eq!(s.indices.base.order, strings(&ri["order"]));
    assert_eq!(s.indices.ix_in_agg.iter().map(|&x| (x + 1) as f64).collect::<Vec<_>>(), flat(&ri["Ix_in_agg"]));
    assert_eq!(s.indices.ix_in_het.iter().map(|&x| (x + 1) as f64).collect::<Vec<_>>(), flat(&ri["Ix_in_het"]));
    assert_eq!(s.indices.mult_in_het.iter().map(|&x| (x + 1) as f64).collect::<Vec<_>>(), flat(&ri["mult_in_het"]));
    let to_f = |v: &[usize]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    assert_eq!(to_f(&s.indices.y_all), flat(&ri["Y_all"]));
    assert_eq!(to_f(&s.indices.y_lead), flat(&ri["Y_lead"]));
    assert_eq!(to_f(&s.indices.y_current), flat(&ri["Y_current"]));
    assert_eq!(to_f(&s.indices.y_lag), flat(&ri["Y_lag"]));
    assert_eq!(to_f(&s.indices.y_exo), flat(&ri["Y_exo"]));
    // steady_state structure
    let rss = &r["ss"];
    for (n, v) in &s.steady_state.agg {
        assert!(close(*v, rss["agg"][n].as_f64().unwrap(), 1e-14), "agg {n}");
    }
    for (n, a) in &s.steady_state.pol_values {
        let w = &rss["pol_values"][n];
        assert_eq!(a.dims.iter().map(|&d| d as f64).collect::<Vec<_>>(), flat(&w["size"]), "pol.values.{n} size");
        assert_close(&format!("pol.values.{n}"), &a.data, &flat(&w["data"]), tol);
    }
    let h = s.steady_state.d_hist.as_ref().unwrap();
    assert_close("d.hist", &h.data, &flat(&rss["d_hist"]["data"]), tol);
    for (n, g) in &s.steady_state.pol_grids {
        assert_close(&format!("pol.grids.{n}"), g, &flat(&rss["pol_grids"][n]), 1e-14);
    }
    // mat
    let rm = &r["mat"];
    let mt = &s.mat;
    let want_mu = rows(&rm["Mu"]);
    for i in 0..mt.mu.nrows() {
        for j in 0..mt.mu.ncols() {
            assert!(close(mt.mu[(i, j)], want_mu[i][j], 1e-14), "Mu");
        }
    }
    let cmp_mat = |what: &str, m: &dynare::linalg::Mat, w: &Value, tol: f64| {
        let w = rows(w);
        assert_eq!((m.nrows(), m.ncols()), (w.len(), w[0].len()), "{what} shape");
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                assert!(close(m[(i, j)], w[i][j], tol), "{what}[{i}][{j}]: {} vs {}", m[(i, j)], w[i][j]);
            }
        }
    };
    cmp_mat("sm", &mt.sm, &rm["sm"], 1e-14);
    cmp_mat("x_bar", &mt.x_bar, &rm["x_bar"], tol);
    cmp_mat("d.x_bar", mt.d_x_bar.as_ref().unwrap(), &rm["d_x_bar"], tol);
    assert_eq!(mt.pol.ind.iter().map(|&x| x as f64).collect::<Vec<_>>(), flat_colmajor(&rm["pol_ind"]));
    assert_close("pol.w", &mt.pol.w, &flat_colmajor(&rm["pol_w"]), tol.max(1e-13));
    assert_close("pol.inv_h", &mt.pol.inv_h, &flat_colmajor(&rm["pol_inv_h"]), tol.max(1e-13));
    assert_eq!(mt.d.ind.iter().map(|&x| x as f64).collect::<Vec<_>>(), flat_colmajor(&rm["d_ind"]));
    assert_close("d.w", &mt.d.w, &flat_colmajor(&rm["d_w"]), tol);
    assert_close("d.inv_h", &mt.d.inv_h, &flat_colmajor(&rm["d_inv_h"]), tol);
    assert_close("d.hist", &mt.d_hist, &flat(&rm["d_hist"]), tol);
    let phi_e = mt.pol_phi_e.as_ref().unwrap();
    assert_eq!(phi_e.nnz() as f64, rm["Phi_e_nnz"].as_f64().unwrap());
    assert!(close(phi_e.vals.iter().sum::<f64>(), rm["Phi_e_sum"].as_f64().unwrap(), tol.max(1e-12)));
    for row in rows(&rm["Phi_e_samples"]) {
        let (i, j, v) = (row[0] as usize - 1, row[1] as usize - 1, row[2]);
        let p = (phi_e.colptr[j]..phi_e.colptr[j + 1]).find(|&p| phi_e.rowind[p] == i).expect("Phi_e entry");
        assert!(close(phi_e.vals[p], v, tol.max(1e-13)));
    }
    let phi = mt.d_phi.as_ref().unwrap();
    assert_eq!(phi.nnz() as f64, rm["Phi_nnz"].as_f64().unwrap());
    assert!(close(phi.vals.iter().sum::<f64>(), rm["Phi_sum"].as_f64().unwrap(), 1e-12));
    // aggregate residuals, heterogeneous residuals, Jacobians
    assert_close("G", &mt.g, &flat(&rm["G"]), tol.max(1e-12));
    cmp_mat("dG", mt.dg.as_ref().unwrap(), &rm["dG"], tol.max(1e-12));
    assert_close("mat.y", &mt.y, &flat(&rm["y"]), tol.max(1e-13));
    assert_close("Ix", &mt.ix, &flat(&rm["Ix"]), tol);
    cmp_mat("F", mt.f.as_ref().unwrap(), &rm["F"], tol.max(1e-9));
    let (nh, nc, nsp) = mt.df_dims;
    let want_size = flat(&rm["dF_size"]);
    assert_eq!(vec![nh as f64, nc as f64, nsp as f64], want_size);
    // samples of dF at 5 points: JSON is [row][col][sample]
    let idx = flat(&rm["dF_sample_idx"]);
    let sam = &rm["dF_samples"];
    for i in 0..nh {
        for c in 0..nc {
            for (k, p) in idx.iter().enumerate() {
                let want = sam[i][c][k].as_f64().unwrap();
                let got = mt.df[i + nh * c + nh * nc * (*p as usize - 1)];
                assert!(close(got, want, tol.max(1e-9)), "dF[{i}][{c}][{p}]: {got} vs {want}");
            }
        }
    }
}

/// Column-major flattening of a JSON matrix given as rows.
fn flat_colmajor(v: &Value) -> Vec<f64> {
    let r = rows(v);
    let (m, n) = (r.len(), r[0].len());
    let mut out = Vec::with_capacity(m * n);
    for j in 0..n {
        for row in &r {
            out.push(row[j]);
        }
    }
    out
}

#[test]
fn ksl_load_steady_state() {
    let r = reference("ref_ksl.json");
    let mf = model("ksl.mod");
    let o = HetOptions::default();
    let input = mat_input(&dir().join("krusell_smith_1998.mat"), "steady_state");
    let s = run_load_steady_state(&mf, &o, input).unwrap();
    check_steady_state(&s, &r, 1e-12);
}

#[test]
fn ks_compute_steady_state() {
    let r = reference("ref_ks.json");
    let mf = model("ks.mod");
    let mut o = HetOptions::default();
    o.time_iteration.verbosity = 0;
    o.forward.verbosity = 0;
    o.calibration.verbosity = 0;
    let input = mat_input(&dir().join("ks_guess.mat"), "initial_guess");
    let c = run_compute_steady_state(&mf, &o, input).unwrap();
    // the calibrated discount factor
    let ib = mf.model.param_names.iter().position(|n| n == "beta").unwrap();
    let want = flat(&r["params_after"]);
    assert_close("params", &c.params, &want, 1e-12);
    assert!(close(c.params[ib], want[ib], 1e-9));
    check_steady_state(&c.steady_state, &r, 1e-11);
}

#[test]
fn rouwenhorst_matches_dynare() {
    let r = reference("ref_ks.json");
    for (key, (rho, sigma, n, tol, maxit)) in
        [("rouwenhorst", (0.966, 0.5, 7, 1e-12, 100000)), ("rouwenhorst2", (0.9, 0.3, 4, 1e-10, 1000))]
    {
        let (y, p, pm) = rouwenhorst(rho, sigma, n, tol, maxit).unwrap();
        let w = &r[key];
        // p comes from an iterated stationary distribution (tol up to 1e-10): allow 1e-11;
        // y and P are closed-form but go through exp/pow/binomials, so allow libm noise.
        assert_close("y", &y, &flat(&w["y"]), 1e-13);
        assert_close("p", &p, &flat(&w["p"]), 1e-11);
        let wp = rows(&w["P"]);
        for i in 0..n {
            for j in 0..n {
                assert!(close(pm[(i, j)], wp[i][j], 1e-13), "P[{i}][{j}]");
            }
        }
    }
}

#[test]
fn hank1_preprocessor_structures() {
    // examples/heterogeneity/hank_one_asset_steady_state.mod: two aggregates (`SUM`/`sum`), an
    // exogenous lag in the aggregate model, labor supply and effective labor as policies
    let r = reference("ref_hank1.json");
    let mf = model("hank1.mod");
    check_preprocessor(&mf, &r);
    check_generated_code(&mf, &r);
}

#[test]
fn hank1_compute_steady_state() {
    // two calibrated parameters (beta, vphi) matching two named target equations
    let r = reference("ref_hank1.json");
    let mf = model("hank1.mod");
    let mut o = HetOptions::default();
    o.time_iteration.verbosity = 0;
    o.forward.verbosity = 0;
    o.calibration.verbosity = 0;
    o.time_iteration.solver_stop_on_error = true;
    o.calibration.target_equations = vec![
        dynare::heterogeneity::TargetEquation::Name("Asset market clearing".into()),
        dynare::heterogeneity::TargetEquation::Name("Labor market clearing".into()),
    ];
    let input = mat_input(&dir().join("hank1_guess.mat"), "initial_guess");
    let c = run_compute_steady_state(&mf, &o, input).unwrap();
    assert_close("params", &c.params, &flat(&r["params_after"]), 1e-10);
    check_steady_state(&c.steady_state, &r, 1e-9);
}

// ---- declarations, shocks and error messages ----------------------------------------------

const BASE: &str = "
heterogeneity_dimension hh;
var(heterogeneity=hh) c a;
varexo(heterogeneity=hh) e1 e2;
var K r;
varexo Z;
parameters beta alpha;
beta = 0.9; alpha = 0.3;
";

fn parse(body: &str) -> anyhow::Result<ModFile> {
    dynare::modfile::parse_mod_str(&format!("{BASE}{body}"), &[])
}

const OK_HET: &str = "
model(heterogeneity=hh);
  c + a = (1+r)*a(-1) + e1 + e2;
  c^(-1) = beta*(1+r(+1))*c(+1)^(-1) ⟂ a >= 0;
end;
model;
  K = SUM(a);
  r = alpha*K(-1)^(alpha-1)*exp(Z);
end;
";

#[test]
fn declarations_and_shocks() {
    let mf = parse(&format!(
        "{OK_HET} shocks(heterogeneity=hh); var e1 = 0.04; var e2; stderr 0.1; var e1, e2 = 0.002; end;
         shocks; var Z; stderr 0.01; end;"
    ))
    .unwrap();
    let h = mf.het.model.as_ref().unwrap();
    assert_eq!(h.exo_names, ["e1", "e2"]);
    assert_eq!(h.orig_endo_nbr, 2);
    assert_eq!(h.state_var, vec![1]);
    assert_eq!(h.endo_names[..2], ["c", "a"]);
    assert_eq!(h.endo_names.last().unwrap(), "MULT_L_a");
    // Sigma_e: variance, stderr (squared), covariance
    assert_eq!(h.sigma_e[(0, 0)], 0.04);
    assert!(close(h.sigma_e[(1, 1)], 0.01, 1e-15));
    assert_eq!((h.sigma_e[(0, 1)], h.sigma_e[(1, 0)]), (0.002, 0.002));
    // the aggregate model: SUM_a is the aggregation variable (type 14), after the declared ones
    assert_eq!(mf.model.endo_names[..3], ["K", "r", "SUM_a"]);
    assert_eq!(mf.model.orig_endo_nbr, 2);
    assert_eq!(mf.het.aggregate_aux[0].ty, 14);
    assert_eq!(mf.het.aggregates[0].het_endo, 1);
    // shocks(heterogeneity) on an aggregate shock is an error
    let e = parse(&format!("{OK_HET} shocks(heterogeneity=hh); var Z; stderr 0.1; end;")).unwrap_err().to_string();
    assert!(e.contains("not a heterogeneous exogenous variable"), "{e}");
    // corr
    let mf = parse(&format!(
        "{OK_HET} shocks(heterogeneity=hh); var e1; stderr 0.1; var e2; stderr 0.2; corr e1, e2 = 0.5; end;"
    ))
    .unwrap();
    assert!(close(mf.het.model.as_ref().unwrap().sigma_e[(0, 1)], 0.5 * 0.1 * 0.2, 1e-15));
}

#[test]
fn model_errors() {
    let err = |body: &str| parse(body).unwrap_err().to_string();
    // undeclared dimension, SUM of an aggregate variable, SUM in the heterogeneous block
    assert!(err("var(heterogeneity=zz) q;").contains("Unknown heterogeneity dimension: zz"));
    let e = err("model(heterogeneity=hh); c + a = a(-1) + e1 + e2; c = a; end; model; K = SUM(r); r = Z; end;");
    assert!(e.contains("must be a heterogeneous endogenous variable"), "{e}");
    let e = err("model(heterogeneity=hh); c + a = SUM(a); c = a; end; model; K = r; r = Z; end;");
    assert!(e.contains("cannot be used inside a model(heterogeneity=...) block"), "{e}");
    // lead/lag limits of the heterogeneous block
    let e = err("model(heterogeneity=hh); c + a = a(-2) + e1 + e2; c = a; end; model; K = SUM(a); r = Z; end;");
    assert!(e.contains("with lag -2 is not supported (maximum lag is -1)"), "{e}");
    let e = err("model(heterogeneity=hh); c + a = a(-1) + e1(-1) + e2; c = a; end; model; K = SUM(a); r = Z; end;");
    assert!(e.contains("lagged heterogeneous exogenous variable 'e1' is not supported"), "{e}");
    // non-separable expression of a lead and a lagged state
    let e = err(
        "model(heterogeneity=hh); c + a = log(a(-1) + c(+1)) + e1 + e2; c = a; end; model; K = SUM(a); r = Z; end;",
    );
    assert!(e.contains("Non-separable expression"), "{e}");
    // a separable one is fine
    assert!(
        parse("model(heterogeneity=hh); c + a = a(-1) + c(+1) + e1 + e2; c = a; end; model; K = SUM(a); r = Z; end;")
            .is_ok()
    );
    // equations / variables
    let e = err("model(heterogeneity=hh); c + a = a(-1) + e1 + e2; end; model; K = SUM(a); r = Z; end;");
    assert!(
        e.contains("There are 1 equations but 2 endogenous variables in the model for heterogeneity dimension 'hh'"),
        "{e}"
    );
    // two complementarity conditions on one variable
    let e = err(
        "model(heterogeneity=hh); c + a = a(-1) + e1 + e2 ⟂ a >= 0; c = a ⟂ a >= 1; end; model; K = SUM(a); r = Z; end;",
    );
    assert!(e.contains("appears in two complementarity conditions"), "{e}");
    // an upper and a lower bound make two multipliers
    let mf = parse(
        "model(heterogeneity=hh); c + a = a(-1) + e1 + e2 ⟂ 0 <= a <= 5; c = a; end; model; K = SUM(a); r = Z; end;",
    );
    let e = mf.err().map(|e| e.to_string()).unwrap_or_default();
    // 2 equations + 2 bound equations for 2 variables + 2 multipliers: the equation count is right
    assert!(!e.contains("There are"), "{e}");
}

#[test]
fn double_bound_makes_two_multipliers() {
    let mf = parse(
        "model(heterogeneity=hh); c + a = a(-1) + e1 + e2 ⟂ 0 < a < 5; c = a; end; model; K = SUM(a); r = Z; end;",
    );
    let mf = mf.unwrap();
    let h = mf.het.model.as_ref().unwrap();
    assert_eq!(h.endo_names[2..], ["MULT_L_a", "MULT_U_a"]);
    let (lo, hi) = h.complementarity_bounds(&mf.model.params);
    assert_eq!((lo[1], hi[1]), (0.0, 5.0));
}

#[test]
fn steady_state_input_errors() {
    let mf = model("ks.mod");
    let o = HetOptions::default();
    let good = || mat_input(&dir().join("ks_guess.mat"), "initial_guess");
    // an initial guess has no distribution: loading it as a steady state fails
    let e = run_load_steady_state(&mf, &o, good()).unwrap_err().to_string();
    assert!(e.contains("the `steady_state.d` field is missing"), "{e}");
    let ss = || mat_input(&dir().join("krusell_smith_1998.mat"), "steady_state");
    let ksl = model("ksl.mod");
    let mut s = ss();
    s.agg.as_mut().unwrap().retain(|(n, _)| n != "K");
    let e = run_load_steady_state(&ksl, &o, s).unwrap_err().to_string();
    assert!(e.contains("The following fields are missing in `steady_state.agg`: K"), "{e}");
    let mut s = ss();
    s.shocks.as_mut().unwrap().pi.as_mut().unwrap()[0].1[(0, 0)] += 0.1;
    let e = run_load_steady_state(&ksl, &o, s).unwrap_err().to_string();
    assert!(e.contains("has row sums different from 1"), "{e}");
    let mut s = ss();
    s.pol.as_mut().unwrap().grids.as_mut().unwrap()[0].1.swap(3, 4);
    let e = run_load_steady_state(&ksl, &o, s).unwrap_err().to_string();
    assert!(e.contains("are not strictly increasing"), "{e}");
    let mut s = ss();
    s.d.as_mut().unwrap().hist.as_mut().unwrap().dims = vec![500, 7];
    let e = run_load_steady_state(&ksl, &o, s).unwrap_err().to_string();
    assert!(e.contains("incompatible sizes"), "{e}");
    // residuals on the aggregate equations
    let mut s = ss();
    s.agg.as_mut().unwrap().iter_mut().find(|(n, _)| n == "K").unwrap().1 = 5.0;
    let e = run_load_steady_state(&ksl, &o, s).unwrap_err().to_string();
    assert!(e.contains("residuals too large on aggregate equations"), "{e}");
    let mut o2 = HetOptions::default();
    o2.check.tolf = f64::INFINITY;
    let mut s = ss();
    s.agg.as_mut().unwrap().iter_mut().find(|(n, _)| n == "K").unwrap().1 = 5.0;
    assert!(run_load_steady_state(&ksl, &o2, s).is_ok());
    // free parameters: unknown name, number of target equations
    let fp = |v: f64| dynare::heterogeneity::check::FreeParameter {
        initial_guess: Some(v),
        lower_bound: None,
        upper_bound: None,
    };
    let mut g = good();
    g.free_parameters.push(("nope".into(), fp(1.0)));
    let e = run_compute_steady_state(&mf, &o, g).err().unwrap().to_string();
    assert!(e.contains("missing in `M_.param_names`: nope"), "{e}");
    let mut o3 = HetOptions::default();
    o3.calibration.verbosity = 0;
    o3.time_iteration.verbosity = 0;
    o3.forward.verbosity = 0;
    let mut g = good();
    g.free_parameters.clear();
    // Dynare 7.2: without free parameters the aggregate residuals are only evaluated, and
    // `‖G‖∞ > calibration_tolf` errors (the aggregate values are not recomputed)
    let e = run_compute_steady_state(&mf, &o3, g.clone()).err().unwrap().to_string();
    assert!(e.contains("heterogeneity_compute_steady_state: residuals too large on aggregate equations"), "{e}");
    o3.calibration.tolf = f64::INFINITY;
    let c = run_compute_steady_state(&mf, &o3, g).unwrap();
    assert!(c.output.converged); // no free parameter: the residuals are only evaluated
    o3.calibration.tolf = HetOptions::default().calibration.tolf;
    let mut g = good();
    g.free_parameters.push(("alpha".into(), fp(0.1)));
    g.free_parameters.push(("delta".into(), fp(0.02)));
    let e = run_compute_steady_state(&mf, &o3, g).err().unwrap().to_string();
    assert!(e.contains("must match number of unknown parameters"), "{e}");
}

#[test]
fn rouwenhorst_properties() {
    let (y, p, pm) = rouwenhorst(0.9, 0.3, 5, 1e-14, 100000).unwrap();
    assert!(close(y.iter().zip(&p).map(|(a, b)| a * b).sum::<f64>(), 1.0, 1e-14)); // normalised mean
    for i in 0..5 {
        assert!(close((0..5).map(|j| pm[(i, j)]).sum::<f64>(), 1.0, 1e-14));
    }
    assert!(rouwenhorst(0.9, 0.3, 1, 1e-14, 10).is_err());
}

// ---- heterogeneity_solve / heterogeneity_simulate (part 2) -------------------------------
// reference/heterogeneity/ref_solve_{ks,hank1,ksnews}.json, written by dump_solve.m from the
// models of this folder followed by `heterogeneity_solve(truncation_horizon = 40)` and
// `heterogeneity_simulate(irf = 30, periods = 25)`; the steady state is the MATLAB one
// (`<model>_ss.mat`), so the inputs are identical.

use dynare::heterogeneity::{HetDr, SimOptions, simulate, solve_hetero};
use dynare::linalg::Mat;

fn load_ss(mf: &mut ModFile, name: &str, r: &Value) -> dynare::heterogeneity::LoadedSteadyState {
    mf.model.params = flat(&r["params_after"]);
    let input = mat_input(&dir().join(format!("{name}_ss.mat")), "steady_state");
    let mut o = HetOptions::default();
    o.check.tolf = f64::INFINITY; // the calibrated steady state meets the target equations to 1e-5 only
    run_load_steady_state(mf, &o, input).unwrap()
}

fn mat_flat(m: &Mat) -> Vec<f64> {
    (0..m.nrows()).flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)])).collect()
}

fn colmajor(m: &Mat) -> Vec<f64> {
    (0..m.ncols()).flat_map(|j| (0..m.nrows()).map(move |i| m[(i, j)])).collect()
}

fn nd_data(v: &Value) -> Vec<f64> {
    flat(&v["data"])
}

fn check_solution(mf: &ModFile, ss: &dynare::heterogeneity::LoadedSteadyState, dr: &HetDr, r: &Value, tol: f64) {
    let rd = &r["dr"];
    for (i, v) in dr.endo_names.iter().enumerate() {
        for (j, s) in dr.exo_names.iter().enumerate() {
            let want = flat(&rd["G"][v][s]);
            assert_close(&format!("G.{v}.{s}"), &mat_flat(&dr.g[i * dr.exo_names.len() + j]), &want, tol);
        }
    }
    for (k, out) in dr.output_names.iter().enumerate() {
        for (y, inp) in dr.input_names.iter().enumerate() {
            assert_close(&format!("F.{out}.{inp}"), &mat_flat(&dr.f[k][y]), &flat(&rd["F"][out][inp]), tol);
            assert_close(&format!("J_ha.{out}.{inp}"), &mat_flat(&dr.j_ha[k][y]), &flat(&rd["J_ha"][out][inp]), tol);
        }
    }
    for (y, inp) in dr.input_names.iter().enumerate() {
        for (k, out) in dr.curly_output_names.iter().enumerate() {
            assert_close(&format!("curlyYs.{inp}.{out}"), &dr.curly_ys[k][y], &flat(&rd["curlyYs"][inp][out]), tol);
        }
        let want = &r["curlyDs"][inp];
        let dims: Vec<usize> = flat(&want["size"]).iter().map(|&x| x as usize).collect();
        assert_eq!(dims, dr.curly_ds_dims, "curlyDs.{inp} dims");
        assert_close(&format!("curlyDs.{inp}"), &colmajor(&dr.curly_ds[y]), &nd_data(want), tol);
    }
    // dr.J: [timing value] rows (a single entry is a vector)
    let mut n = 0;
    for ((eq, var), rows_) in &dr.j {
        let got: Vec<f64> = rows_.iter().flat_map(|r| r.iter().copied()).collect();
        assert_close(&format!("J.{eq}.{var}"), &got, &flat(&rd["J"][eq][var]), tol);
        n += 1;
    }
    let total: usize = rd["J"].as_object().unwrap().values().map(|o| o.as_object().unwrap().len()).sum();
    assert_eq!(n, total, "number of dr.J entries");
    // intermediate policy impulse responses
    let h = mf.het.model.as_ref().unwrap();
    let irs = dynare::heterogeneity::solve::impulse_responses(h, ss, dr.truncation_horizon).unwrap();
    for (key, s) in [("ir_s1", 0usize), ("ir_s2", 1), ("ir_s7", 6)] {
        assert_close(key, &colmajor(&irs[s]), &nd_data(&r[key]), tol);
    }
}

fn check_simulation(mf: &ModFile, dr: &HetDr, ss: &dynare::heterogeneity::LoadedSteadyState, r: &Value, tol: f64) {
    let o = SimOptions { irf: Some(30), periods: Some(25), ..Default::default() };
    let y = &ss.mat.y[mf.model.n_endo()..2 * mf.model.n_endo()];
    let res = simulate(&mf.model, dr, y, &mf.sigma_e, &[], &o, 1).unwrap();
    let want = r["irfs"].as_object().unwrap();
    assert_eq!(res.irfs.len(), want.len());
    for (name, v) in &res.irfs {
        assert_close(&format!("irf {name}"), v, &flat(&want[name]), tol);
    }
    // the stochastic path given MATLAB's draws
    let nx = mf.model.exo_names.len();
    let ev = flat(&r["eps_draws"]); // row-major, nx rows
    let nc = ev.len() / nx;
    let eps = Mat::from_fn(nx, nc, |i, j| ev[i * nc + j]);
    let shocks: Vec<usize> = (0..mf.model.exo_names.len()).collect();
    let path = dynare::heterogeneity::simulate::stochastic_paths(&mf.model, dr, y, &eps, 25, &shocks).unwrap();
    assert_close("endo_simul", &mat_flat(&path), &rows(&r["endo_simul"]).concat(), tol);
    // our own draws: shapes
    let res2 = simulate(&mf.model, dr, y, &mf.sigma_e, &[], &o, 7).unwrap();
    let sim = res2.endo_simul.unwrap();
    assert_eq!((sim.nrows(), sim.ncols()), (mf.model.n_endo(), 25));
}

#[test]
fn ks_solve_and_simulate() {
    let r = reference("ref_solve_ks.json");
    let mut mf = model("ks.mod");
    let ss = load_ss(&mut mf, "ks", &r);
    let t = r["T"].as_f64().unwrap() as usize;
    let dr = solve_hetero(&mf.model, &mf.het, &ss, t).unwrap();
    check_solution(&mf, &ss, &dr, &r, 1e-12);
    check_simulation(&mf, &dr, &ss, &r, 1e-9);
}

#[test]
fn hank1_solve_and_simulate() {
    // two aggregates, an exogenous lag, labor supply and effective labor as policies
    let r = reference("ref_solve_hank1.json");
    let mut mf = model("hank1.mod");
    let ss = load_ss(&mut mf, "hank1", &r);
    let t = r["T"].as_f64().unwrap() as usize;
    let dr = solve_hetero(&mf.model, &mf.het, &ss, t).unwrap();
    check_solution(&mf, &ss, &dr, &r, 1e-12);
    check_simulation(&mf, &dr, &ss, &r, 1e-9);
}

#[test]
fn ks_news_shocks() {
    // shocks block with `periods`/`values`: anticipated shocks known at date 0
    let r = reference("ref_solve_ksnews.json");
    let txt = std::fs::read_to_string(dir().join("ks.mod"))
        .unwrap()
        .replace("var Z; stderr 0.01;", "var Z; stderr 0.01; var Z; periods 2:6; values 0.01;");
    let path = std::env::temp_dir().join("dynare_rust_ks_news.mod");
    std::fs::write(&path, txt).unwrap();
    let mut mf = parse_mod_file(&path, &[]).unwrap();
    let ss = load_ss(&mut mf, "ks", &r);
    let t = r["T"].as_f64().unwrap() as usize;
    let dr = solve_hetero(&mf.model, &mf.het, &ss, t).unwrap();
    let y = &ss.mat.y[mf.model.n_endo()..2 * mf.model.n_endo()];
    let res = simulate(&mf.model, &dr, y, &mf.sigma_e, &mf.det_shocks, &SimOptions::default(), 1).unwrap();
    assert_close("exo_simul", &mat_flat(&res.exo_simul.unwrap()), &rows(&r["exo_simul"]).concat(), 1e-14);
    assert_close("endo_simul", &mat_flat(&res.endo_simul.unwrap()), &rows(&r["endo_simul"]).concat(), 1e-9);
    let bad = SimOptions { periods: Some(10), ..Default::default() };
    let e = simulate(&mf.model, &dr, y, &mf.sigma_e, &mf.det_shocks, &bad, 1).unwrap_err().to_string();
    assert!(e.contains("incompatible with stochastic IRF options") && e.contains("periods=10"), "{e}");
}

#[test]
fn simulate_errors() {
    let r = reference("ref_solve_ks.json");
    let mut mf = model("ks.mod");
    let ss = load_ss(&mut mf, "ks", &r);
    let dr = solve_hetero(&mf.model, &mf.het, &ss, 20).unwrap();
    let y = &ss.mat.y[mf.model.n_endo()..2 * mf.model.n_endo()];
    let o = SimOptions { irf: Some(30), ..Default::default() };
    let e = simulate(&mf.model, &dr, y, &mf.sigma_e, &[], &o, 1).unwrap_err().to_string();
    assert!(e.contains("exceeds Jacobian truncation horizon 20"), "{e}");
    let o = SimOptions { irf_shocks: Some(vec!["nope".into()]), ..Default::default() };
    assert!(simulate(&mf.model, &dr, y, &mf.sigma_e, &[], &o, 1).is_err());
    let o = SimOptions { irf: Some(10), relative_irf: Some(true), ..Default::default() };
    let res = simulate(&mf.model, &dr, y, &mf.sigma_e, &[], &o, 1).unwrap();
    let g = dr.g_block("K", "Z").unwrap();
    let k = &res.irfs.iter().find(|(n, _)| n == "K_Z").unwrap().1;
    assert!((k[3] - 100.0 * g[(3, 0)]).abs() < 1e-14);
}

#[test]
fn ks_simulate_graphs() {
    use dynare::graphics::{GraphFormat, GraphOptions};
    let r = reference("ref_solve_ks.json");
    let mut mf = model("ks.mod");
    let ss = load_ss(&mut mf, "ks", &r);
    let dr = solve_hetero(&mf.model, &mf.het, &ss, r["T"].as_f64().unwrap() as usize).unwrap();
    let o = SimOptions { irf: Some(30), periods: Some(25), ..Default::default() };
    let y = &ss.mat.y[mf.model.n_endo()..2 * mf.model.n_endo()];
    let res = simulate(&mf.model, &dr, y, &mf.sigma_e, &[], &o, 1).unwrap();
    let d = std::env::temp_dir().join("dynare_rust_het_graphs");
    let _ = std::fs::remove_dir_all(&d);
    let g = GraphOptions { enabled: true, formats: vec![GraphFormat::Pdf, GraphFormat::Svg], dir: d.clone() };
    let f1 = dynare::heterogeneity::simulate::plot_irfs(&mf.model, &res.irfs, &o, &g, "ks").unwrap();
    assert!(!f1.is_empty() && f1.iter().all(|p| p.exists()));
    assert!(d.join("ks_IRF_Z.pdf").exists() || d.join("ks_IRF_Z1.pdf").exists());
    let f2 = dynare::heterogeneity::simulate::plot_simulation(
        &mf.model,
        res.endo_simul.as_ref().unwrap(),
        &[],
        false,
        &g,
        "ks",
    )
    .unwrap();
    assert!(f2.iter().any(|p| p.file_name().unwrap().to_string_lossy().starts_with("ks_simulation")));
    let off = GraphOptions { enabled: false, ..g };
    assert!(dynare::heterogeneity::simulate::plot_irfs(&mf.model, &res.irfs, &o, &off, "ks").unwrap().is_empty());
}
