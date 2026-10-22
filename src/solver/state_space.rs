// SPDX-License-Identifier: GPL-3.0-or-later
//! Restricted state space used by the likelihood.
//!
//! Port of the "restricted state space" block of
//! `estimation/dynare_estimation_init.m` (restrict_var_list, restrict_columns,
//! mf0, mf1, mf2, mfys, smoother_*), of `stochastic_solver/kalman_transition_matrix.m`
//! and of the `'restrict'` branch of `stochastic_solver/dynare_resolve.m`.

use super::first_order::DecisionRule;
use crate::linalg::Mat;

/// Index sets of the restricted state space (all 0-based).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StateSpaceIndex {
    /// `oo_.dr.restrict_var_list`: sorted DR positions of observed + state variables.
    pub restrict_var_list: Vec<usize>,
    /// `oo_.dr.restrict_columns`: positions in `restrict_var_list` of the state variables
    /// (columns of T filled by `ghx`).
    pub restrict_columns: Vec<usize>,
    /// `bayestopt_.mf1`: positions of the observed variables in `restrict_var_list` (varobs order).
    pub mf: Vec<usize>,
    /// `bayestopt_.mf0`: positions of the state variables in `restrict_var_list`.
    pub mf0: Vec<usize>,
    /// `bayestopt_.mf2`: DR positions of the observed variables (varobs order).
    pub mf2: Vec<usize>,
    /// `bayestopt_.mfys`: declaration indices of the observed variables (rows of `ys`).
    pub mfys: Vec<usize>,
    /// `bayestopt_.smoother_var_list` (all variables, `selected_variables_only = false`).
    pub smoother_var_list: Vec<usize>,
    /// `bayestopt_.smoother_saved_var_list`.
    pub smoother_saved_var_list: Vec<usize>,
    /// `bayestopt_.smoother_restrict_columns`.
    pub smoother_restrict_columns: Vec<usize>,
    /// `bayestopt_.smoother_mf`.
    pub smoother_mf: Vec<usize>,
}

/// `[~, loc] = ismember(a, b)` for `b` sorted and `a` known to be in `b`.
fn positions(a: &[usize], b: &[usize]) -> Vec<usize> {
    a.iter().map(|x| b.binary_search(x).expect("ismember: element not found")).collect()
}

/// Port of the restricted-state-space part of `dynare_estimation_init.m`.
/// `varobs` are declaration indices of the observed variables, in `varobs` order.
pub fn state_space_index(dr: &DecisionRule, varobs: &[usize]) -> StateSpaceIndex {
    let n = dr.order_var.len();
    let (nstatic, nspred) = (dr.nstatic, dr.nspred);
    // dynare_estimation_init.m sets npred = M_.nspred, so nspred - npred = 0 there.
    let var_obs_index_dr: Vec<usize> = varobs.iter().map(|&v| dr.inv_order_var[v]).collect();
    let states: Vec<usize> = (nstatic..nstatic + nspred).collect();
    let mut k2: Vec<usize> = var_obs_index_dr.iter().chain(&states).copied().collect();
    k2.sort_unstable();
    k2.dedup();
    let mf0 = positions(&states, &k2);
    let mf = positions(&var_obs_index_dr, &k2);
    // [~, ic] = intersect(k2, nstatic+(1:npred)'): k2 sorted, so ic = positions of the states
    let restrict_columns = mf0.clone();
    let smoother_var_list: Vec<usize> = (0..n).collect(); // union(k2, k3) with k3 = 1:endo_nbr
    let smoother_saved_var_list: Vec<usize> = (0..n).collect();
    let smoother_restrict_columns = states.clone();
    let smoother_mf = var_obs_index_dr.clone();
    StateSpaceIndex {
        restrict_var_list: k2,
        restrict_columns,
        mf,
        mf0,
        mf2: var_obs_index_dr,
        mfys: varobs.to_vec(),
        smoother_var_list,
        smoother_saved_var_list,
        smoother_restrict_columns,
        smoother_mf,
    }
}

impl StateSpaceIndex {
    /// The unrestricted state space of `dynare_resolve` called without `'restrict'`
    /// (`iv = 1:endo_nbr`, `ic = nstatic + (1:nspred)`); only `restrict_var_list`
    /// and `restrict_columns` are meaningful.
    pub fn full(dr: &DecisionRule) -> Self {
        let n = dr.order_var.len();
        StateSpaceIndex {
            restrict_var_list: (0..n).collect(),
            restrict_columns: (dr.nstatic..dr.nstatic + dr.nspred).collect(),
            mf: vec![],
            mf0: vec![],
            mf2: vec![],
            mfys: vec![],
            smoother_var_list: (0..n).collect(),
            smoother_saved_var_list: (0..n).collect(),
            smoother_restrict_columns: (dr.nstatic..dr.nstatic + dr.nspred).collect(),
            smoother_mf: vec![],
        }
    }
}

/// Port of `kalman_transition_matrix.m` with `iv = restrict_var_list`,
/// `ic = restrict_columns`: `T(:, ic) = ghx(iv, :)`, `R = ghu(iv, :)`.
pub fn kalman_transition_matrix(dr: &DecisionRule, idx: &StateSpaceIndex) -> (Mat, Mat) {
    let m = idx.restrict_var_list.len();
    let mut t = Mat::zeros(m, m);
    let mut r = Mat::zeros(m, dr.ghu.ncols());
    kalman_transition_matrix_into(dr, idx, &mut t, &mut r);
    (t, r)
}

/// Allocation-free [`kalman_transition_matrix`] into preallocated `t` (m x m) and `r` (m x n_exo).
pub fn kalman_transition_matrix_into(dr: &DecisionRule, idx: &StateSpaceIndex, t: &mut Mat, r: &mut Mat) {
    let iv = &idx.restrict_var_list;
    let m = iv.len();
    assert_eq!((t.nrows(), t.ncols()), (m, m), "kalman_transition_matrix: T shape");
    assert_eq!((r.nrows(), r.ncols()), (m, dr.ghu.ncols()), "kalman_transition_matrix: R shape");
    t.fill(0.0);
    for (k, &c) in idx.restrict_columns.iter().enumerate() {
        for (i, &v) in iv.iter().enumerate() {
            t[(i, c)] = dr.ghx[(v, k)];
        }
    }
    for j in 0..dr.ghu.ncols() {
        for (i, &v) in iv.iter().enumerate() {
            r[(i, j)] = dr.ghu[(v, j)];
        }
    }
}
