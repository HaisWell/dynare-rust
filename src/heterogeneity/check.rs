// SPDX-License-Identifier: GPL-3.0-or-later
// ---- heterogeneity ----
//! Port of `+heterogeneity/check_steady_state_input.m`: validates the user's `steady_state`
//! structure, normalises it (column grids, arrays permuted to the order `[shocks; states]`,
//! distribution grids defaulting to the policy grids) and computes the sizes and indices.

use super::model::HetModel;
use crate::linalg::Mat;
use crate::matfile::MatValue;
use anyhow::{Result, bail};

/// A multidimensional array, column-major (first dimension fastest), MATLAB `size` semantics:
/// trailing singleton dimensions are dropped except that there are at least two dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct NdArray {
    pub dims: Vec<usize>,
    pub data: Vec<f64>,
}

impl NdArray {
    pub fn new(dims: Vec<usize>, data: Vec<f64>) -> NdArray {
        NdArray { dims, data }
    }

    /// `size(x, d)` (1-based `d`; 1 beyond the number of dimensions).
    pub fn size(&self, d: usize) -> usize {
        self.dims.get(d - 1).copied().unwrap_or(1)
    }

    /// `ndims(x)`.
    pub fn ndims(&self) -> usize {
        let mut n = self.dims.len();
        while n > 2 && self.dims[n - 1] == 1 {
            n -= 1;
        }
        n.max(2)
    }

    /// `permute(x, order)` with 0-based `order` (`order[k]` = old dimension of new dimension k).
    pub fn permute(&self, order: &[usize]) -> NdArray {
        let nd = order.len();
        let old: Vec<usize> = (0..nd).map(|d| self.dims.get(d).copied().unwrap_or(1)).collect();
        let new_dims: Vec<usize> = order.iter().map(|&o| old[o]).collect();
        let mut old_strides = vec![1usize; nd];
        for d in 1..nd {
            old_strides[d] = old_strides[d - 1] * old[d - 1];
        }
        let total: usize = new_dims.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; nd];
        for _ in 0..total {
            let off: usize = (0..nd).map(|k| idx[k] * old_strides[order[k]]).sum();
            out.push(self.data[off]);
            for k in 0..nd {
                idx[k] += 1;
                if idx[k] < new_dims[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        NdArray { dims: new_dims, data: out }
    }

    fn from_mat(v: &MatValue) -> Option<NdArray> {
        match v {
            MatValue::Numeric { dims, data } => Some(NdArray { dims: dims.clone(), data: data.clone() }),
            _ => None,
        }
    }
}

/// `steady_state.free_parameters.NAME`.
#[derive(Clone, Debug, PartialEq)]
pub struct FreeParameter {
    pub initial_guess: Option<f64>,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShocksIn {
    pub grids: Option<Vec<(String, Vec<f64>)>>,
    pub pi: Option<Vec<(String, Mat)>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PolIn {
    pub values: Option<Vec<(String, NdArray)>>,
    pub grids: Option<Vec<(String, Vec<f64>)>>,
    pub order: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DIn {
    pub hist: Option<NdArray>,
    pub grids: Option<Vec<(String, Vec<f64>)>>,
    pub order: Option<Vec<String>>,
}

/// The `steady_state` structure provided by the user (MAT file or in code). `None` marks a
/// missing field.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SteadyStateInput {
    pub agg: Option<Vec<(String, f64)>>,
    pub shocks: Option<ShocksIn>,
    pub pol: Option<PolIn>,
    pub d: Option<DIn>,
    pub free_parameters: Vec<(String, FreeParameter)>,
}

fn fields(v: &MatValue) -> Option<Vec<(String, &MatValue)>> {
    match v {
        MatValue::Struct { fields, elems, .. } if elems.len() == 1 => {
            Some(fields.iter().cloned().zip(elems[0].iter()).collect())
        }
        _ => None,
    }
}

fn vec_fields(v: &MatValue, what: &str) -> Result<Vec<(String, Vec<f64>)>> {
    let Some(f) = fields(v) else {
        bail!("Misspecified steady-state input `steady_state`: the `{what}` field should be a structure.")
    };
    f.into_iter()
        .map(|(k, x)| match x {
            MatValue::Numeric { dims, data } if dims.iter().filter(|&&d| d != 1).count() <= 1 => Ok((k, data.clone())),
            _ => bail!("Misspecified steady-state input `steady_state`: `{what}.{k}` is not a dense real vector."),
        })
        .collect()
}

fn cellstr(v: &MatValue, what: &str) -> Result<Vec<String>> {
    v.as_strings().ok_or_else(|| anyhow::anyhow!("Misspecified steady-state input `steady_state`: the `{what}` field should be a cell array of character vectors or a string array."))
}

impl SteadyStateInput {
    /// Reads the structure `v` (a variable of a MAT file).
    pub fn from_mat(v: &MatValue) -> Result<SteadyStateInput> {
        let Some(top) = fields(v) else {
            bail!("Misspecified steady-state input: the steady-state input `steady_state` is not a structure.")
        };
        let get = |name: &str| top.iter().find(|(k, _)| k == name).map(|(_, x)| *x);
        let mut out = SteadyStateInput::default();
        if let Some(a) = get("agg") {
            let Some(f) = fields(a) else {
                bail!(
                    "Misspecified steady-state input `steady_state`: the `steady_state.agg` field should be a structure."
                )
            };
            out.agg = Some(
                f.into_iter()
                    .map(|(k, x)| match x {
                        MatValue::Numeric { data, .. } if data.len() == 1 => Ok((k, data[0])),
                        _ => bail!("Misspecified steady-state input `steady_state`: the following fields in `steady_state.agg` are not real scalars: {k}."),
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        if let Some(s) = get("shocks") {
            let Some(f) = fields(s) else {
                bail!(
                    "Misspecified steady-state input `steady_state`: the `steady_state.shocks` field should be a structure."
                )
            };
            let mut sh = ShocksIn::default();
            for (k, x) in f {
                match k.as_str() {
                    "grids" => sh.grids = Some(vec_fields(x, "steady_state.shocks.grids")?),
                    "Pi" => {
                        let Some(pf) = fields(x) else {
                            bail!(
                                "Misspecified steady-state input `steady_state`: the `steady_state.shocks.Pi` field should be a structure."
                            )
                        };
                        sh.pi = Some(
                            pf.into_iter()
                                .map(|(n, m)| match m.as_matrix() {
                                    Some(m) => Ok((n, m)),
                                    None => bail!("Misspecified steady-state input `steady_state`: `steady_state.shocks.Pi.{n}` is not a dense real matrix."),
                                })
                                .collect::<Result<Vec<_>>>()?,
                        );
                    }
                    _ => {}
                }
            }
            out.shocks = Some(sh);
        }
        if let Some(p) = get("pol") {
            let Some(f) = fields(p) else {
                bail!(
                    "Misspecified steady-state input `steady_state`: the `steady_state.pol` field should be a structure."
                )
            };
            let mut pol = PolIn::default();
            for (k, x) in f {
                match k.as_str() {
                    "values" => {
                        let Some(vf) = fields(x) else {
                            bail!(
                                "Misspecified steady-state input `steady_state`: the `steady_state.pol.values` field should be a structure."
                            )
                        };
                        pol.values = Some(
                            vf.into_iter()
                                .map(|(n, a)| match NdArray::from_mat(a) {
                                    Some(a) => Ok((n, a)),
                                    None => bail!("Misspecified steady-state input `steady_state`: the following fields in `steady_state.pol.values` are not dense real arrays: {n}."),
                                })
                                .collect::<Result<Vec<_>>>()?,
                        );
                    }
                    "grids" => pol.grids = Some(vec_fields(x, "steady_state.pol.grids")?),
                    "order" => pol.order = Some(cellstr(x, "steady_state.pol.order")?),
                    _ => {}
                }
            }
            out.pol = Some(pol);
        }
        if let Some(d) = get("d") {
            let Some(f) = fields(d) else {
                bail!(
                    "Misspecified steady-state input `steady_state`: the `steady_state.d` field should be a structure."
                )
            };
            let mut dd = DIn::default();
            for (k, x) in f {
                match k.as_str() {
                    "hist" => {
                        dd.hist = Some(NdArray::from_mat(x).ok_or_else(|| anyhow::anyhow!("Misspecified steady-state input `steady_state`: `steady_state.d.hist` is not a dense real array."))?)
                    }
                    "grids" => dd.grids = Some(vec_fields(x, "steady_state.d.grids")?),
                    "order" => dd.order = Some(cellstr(x, "steady_state.d.order")?),
                    _ => {}
                }
            }
            out.d = Some(dd);
        }
        if let Some(fp) = get("free_parameters") {
            let Some(f) = fields(fp) else {
                bail!(
                    "Misspecified steady-state input `steady_state`: `steady_state.free_parameters` should be a structure."
                )
            };
            for (k, x) in f {
                let Some(pf) = fields(x) else {
                    bail!(
                        "Misspecified steady-state input `steady_state`: `steady_state.free_parameters.{k}` should be a structure."
                    )
                };
                let g = |n: &str| pf.iter().find(|(a, _)| a == n).and_then(|(_, v)| v.as_scalar());
                out.free_parameters.push((
                    k,
                    FreeParameter {
                        initial_guess: g("initial_guess"),
                        lower_bound: g("lower_bound"),
                        upper_bound: g("upper_bound"),
                    },
                ));
            }
        }
        Ok(out)
    }
}

/// `options_.heterogeneity.check`.
#[derive(Clone, Debug)]
pub struct CheckOptions {
    pub tol_check_sum: f64,
    pub tolf: f64,
    pub no_warning_redundant: bool,
    pub no_warning_d_grids: bool,
}

impl Default for CheckOptions {
    fn default() -> Self {
        CheckOptions { tol_check_sum: 1e-6, tolf: 1e-6, no_warning_redundant: true, no_warning_d_grids: true }
    }
}

/// `sizes`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sizes {
    pub agg: usize,
    pub n_e: usize,
    pub n_e_total: usize,
    /// Number of points of each shock (`sizes.shocks.NAME`), in `indices.shocks` order.
    pub shocks: Vec<(String, usize)>,
    pub n_a: usize,
    /// `sizes.pol.states.NAME`, in `indices.states` order, and `sizes.pol.N_a`.
    pub pol_states: Vec<(String, usize)>,
    pub pol_n_a: usize,
    pub n_sp: usize,
    pub n_pol: usize,
    pub d_states: Vec<(String, usize)>,
    pub d_n_a: usize,
    pub n_om: usize,
}

impl Sizes {
    pub fn shock(&self, name: &str) -> usize {
        self.shocks.iter().find(|(n, _)| n == name).unwrap().1
    }
    pub fn pol_state(&self, name: &str) -> usize {
        self.pol_states.iter().find(|(n, _)| n == name).unwrap().1
    }
    pub fn d_state(&self, name: &str) -> usize {
        self.d_states.iter().find(|(n, _)| n == name).unwrap().1
    }
}

/// `indices` (the part built by `check_steady_state_input`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Indices {
    pub states: Vec<String>,
    pub shocks: Vec<String>,
    /// `[shocks; states]`.
    pub order: Vec<String>,
}

/// The normalised `steady_state`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SteadyStateData {
    pub agg: Vec<(String, f64)>,
    pub shocks_grids: Vec<(String, Vec<f64>)>,
    pub shocks_pi: Vec<(String, Mat)>,
    pub pol_grids: Vec<(String, Vec<f64>)>,
    /// Policy functions as arrays with dimensions `[shocks; states]` (`pol.order`).
    pub pol_values: Vec<(String, NdArray)>,
    pub pol_order: Vec<String>,
    pub d_grids: Vec<(String, Vec<f64>)>,
    /// Distribution (absent for an initial guess) with dimensions `d.order = [shocks; states]`.
    pub d_hist: Option<NdArray>,
    pub d_order: Vec<String>,
    pub free_parameters: Vec<(String, FreeParameter)>,
}

impl SteadyStateData {
    pub fn agg(&self, name: &str) -> Option<f64> {
        self.agg.iter().find(|(n, _)| n == name).map(|x| x.1)
    }
    pub fn pol_grid(&self, name: &str) -> &[f64] {
        &self.pol_grids.iter().find(|(n, _)| n == name).unwrap().1
    }
    pub fn d_grid(&self, name: &str) -> &[f64] {
        &self.d_grids.iter().find(|(n, _)| n == name).unwrap().1
    }
    pub fn shock_grid(&self, name: &str) -> &[f64] {
        &self.shocks_grids.iter().find(|(n, _)| n == name).unwrap().1
    }
    pub fn shock_pi(&self, name: &str) -> &Mat {
        &self.shocks_pi.iter().find(|(n, _)| n == name).unwrap().1
    }
    pub fn pol_value(&self, name: &str) -> Option<&NdArray> {
        self.pol_values.iter().find(|(n, _)| n == name).map(|x| &x.1)
    }
    pub fn set_pol_value(&mut self, name: &str, a: NdArray) {
        match self.pol_values.iter_mut().find(|(n, _)| n == name) {
            Some(x) => x.1 = a,
            None => self.pol_values.push((name.to_string(), a)),
        }
    }
}

fn err<T>(msg: String) -> Result<T> {
    bail!("Misspecified steady-state input `steady_state`: {msg}")
}

fn find<'a, T>(v: &'a [(String, T)], n: &str) -> Option<&'a T> {
    v.iter().find(|(k, _)| k == n).map(|x| &x.1)
}

fn check_missing_redundant(have: &[String], what: &str, symbs: &[String], no_warning_redundant: bool) -> Result<()> {
    let missing: Vec<&str> = symbs.iter().filter(|s| !have.contains(s)).map(String::as_str).collect();
    if !missing.is_empty() {
        bail!(
            "Misspecified steady-state input `steady_state`. The following fields are missing in `{what}`: {}.",
            missing.join(" ")
        );
    }
    if !no_warning_redundant {
        let red: Vec<&str> = have.iter().filter(|s| !symbs.contains(s)).map(String::as_str).collect();
        if !red.is_empty() {
            eprintln!(
                "Warning: Steady-state input `steady_state`. The following fields are redundant in `{what}`: {}.",
                red.join(" ")
            );
        }
    }
    Ok(())
}

fn check_permutation(order: &Option<Vec<String>>, name: &str, symbs: &[String]) -> Result<()> {
    if let Some(f) = order {
        let bad: Vec<&str> = f.iter().filter(|s| !symbs.contains(s)).map(String::as_str).collect();
        if !bad.is_empty() {
            return err(format!(
                "the set of variables of the `{name}` field is not consistent with the information in M_ and the other fields in the steady-state structure `steady_state`. Problematic variables: {}.",
                bad.join(" ")
            ));
        }
    }
    Ok(())
}

fn strictly_increasing(x: &[f64]) -> bool {
    x.windows(2).all(|w| w[1] - w[0] > 0.0)
}

/// Port of `heterogeneity.check_steady_state_input(M_, options_het.check, steady_state,
/// flag_initial_guess)`; `agg_symbs` is `M_.endo_names(1:M_.orig_endo_nbr)`.
pub fn check_steady_state_input(
    h: &HetModel,
    agg_symbs: &[String],
    opts: &CheckOptions,
    input: SteadyStateInput,
    flag_initial_guess: bool,
) -> Result<(SteadyStateData, Sizes, Indices)> {
    let orig_pol_symbs: Vec<String> = h.endo_names[..h.orig_endo_nbr].to_vec();
    let mut sizes = Sizes::default();
    let indices = Indices {
        states: h.state_var.iter().map(|&i| h.endo_names[i].clone()).collect(),
        shocks: h.exo_names.clone(),
        order: h.exo_names.iter().cloned().chain(h.state_var.iter().map(|&i| h.endo_names[i].clone())).collect(),
    };
    let mut data = SteadyStateData::default();
    // ---- agg
    let agg = if agg_symbs.is_empty() { Some(Vec::new()) } else { input.agg };
    let Some(agg) = agg else { return err("the `agg` field is missing.".into()) };
    let have: Vec<String> = agg.iter().map(|x| x.0.clone()).collect();
    check_missing_redundant(&have, "steady_state.agg", agg_symbs, opts.no_warning_redundant)?;
    sizes.agg = agg.len();
    let nan: Vec<&str> =
        agg_symbs.iter().filter(|s| find(&agg, s).is_some_and(|v| v.is_nan())).map(String::as_str).collect();
    if !nan.is_empty() {
        return err(format!("the following fields in `steady_state.agg` contain NaN values: {}.", nan.join(" ")));
    }
    data.agg = agg;
    // ---- shocks
    let Some(shocks) = input.shocks else { return err("the `steady_state.shocks` field is missing.".into()) };
    let Some(pi) = shocks.pi else { return err("the `Pi` field is missing.".into()) };
    let Some(grids) = shocks.grids else { return err("the `grids` field is missing.".into()) };
    for s in &h.exo_names {
        let (g, p) = (find(&grids, s), find(&pi, s));
        if g.is_none() || p.is_none() {
            return err(format!(
                "shock {s} requires both `steady_state.shocks.grids.{s}` and `steady_state.shocks.Pi.{s}` to be provided."
            ));
        }
        let g = g.unwrap();
        if g.iter().any(|v| v.is_nan()) {
            return err(format!("the following fields in `steady_state.shocks.grids` contain NaN elements: {s}."));
        }
        if !strictly_increasing(g) {
            return err(format!(
                "the following fields in `steady_state.shocks.grids` are not strictly increasing: {s}."
            ));
        }
        let p = p.unwrap();
        if p.nrows() != g.len() {
            return err(format!(
                "`steady_state.shocks.Pi.{s}` row count does not match `steady_state.shocks.grids.{s}` size."
            ));
        }
        if p.ncols() != g.len() {
            return err(format!(
                "`steady_state.shocks.Pi.{s}` column count does not match `steady_state.shocks.grids.{s}` size."
            ));
        }
        if p.iter().any(|v| v.is_nan()) {
            return err(format!("`steady_state.shocks.Pi.{s}` contains NaN elements."));
        }
        if p.iter().any(|&v| v < 0.0) {
            return err(format!("`steady_state.shocks.Pi.{s}` contains negative elements."));
        }
        for i in 0..p.nrows() {
            let r: f64 = (0..p.ncols()).map(|j| p[(i, j)]).sum();
            if (r - 1.0).abs() > opts.tol_check_sum {
                return err(format!("`steady_state.shocks.Pi.{s}` has row sums different from 1."));
            }
        }
        data.shocks_grids.push((s.clone(), g.clone()));
        data.shocks_pi.push((s.clone(), p.clone()));
    }
    if !opts.no_warning_redundant {
        let red: Vec<&str> =
            grids.iter().map(|x| x.0.as_str()).filter(|s| !h.exo_names.iter().any(|e| e == s)).collect();
        if !red.is_empty() {
            eprintln!(
                "Warning: Steady-state input `steady_state`. The following fields are redundant in `steady_state.shocks.grids`: {}.",
                red.join(" ")
            );
        }
    }
    sizes.n_e = h.exo_nbr();
    sizes.n_e_total = 1;
    for s in &h.exo_names {
        let n = data.shock_grid(s).len();
        sizes.shocks.push((s.clone(), n));
        sizes.n_e_total *= n;
    }
    // ---- pol
    let Some(pol) = input.pol else { return err("the `steady_state.pol` field is missing.".into()) };
    let Some(values) = pol.values else { return err("the `steady_state.pol.values` field is missing.".into()) };
    let have: Vec<String> = values.iter().map(|x| x.0.clone()).collect();
    check_missing_redundant(&have, "steady_state.pol.values", &orig_pol_symbs, true)?;
    if !opts.no_warning_redundant {
        let red: Vec<&str> = have.iter().filter(|s| !orig_pol_symbs.contains(s)).map(String::as_str).collect();
        if !red.is_empty() {
            eprintln!(
                "Warning: Steady-state input `steady_state`. The following fields are redundant in `steady_state.pol.values`: {}.",
                red.join(" ")
            );
        }
    }
    let nan: Vec<&str> = orig_pol_symbs
        .iter()
        .filter(|s| find(&values, s).is_some_and(|a| a.data.iter().any(|v| v.is_nan())))
        .map(String::as_str)
        .collect();
    if !nan.is_empty() {
        return err(format!(
            "the following fields in `steady_state.pol.values` contain NaN elements: {}.",
            nan.join(" ")
        ));
    }
    let Some(pgrids) = pol.grids else { return err("the `steady_state.pol.grids` field is missing.".into()) };
    let have: Vec<String> = pgrids.iter().map(|x| x.0.clone()).collect();
    check_missing_redundant(&have, "steady_state.pol.grids", &indices.states, opts.no_warning_redundant)?;
    for s in &indices.states {
        let g = find(&pgrids, s).unwrap();
        if g.iter().any(|v| v.is_nan()) {
            return err(format!("the following fields in `steady_state.pol.grids` contain NaN elements: {s}."));
        }
        if !strictly_increasing(g) {
            return err(format!("the following fields in `steady_state.pol.grids` are not strictly increasing: {s}."));
        }
    }
    sizes.n_a = indices.states.len();
    sizes.pol_n_a = 1;
    for s in &indices.states {
        let n = find(&pgrids, s).unwrap().len();
        sizes.pol_states.push((s.clone(), n));
        sizes.pol_n_a *= n;
        data.pol_grids.push((s.clone(), find(&pgrids, s).unwrap().clone()));
    }
    sizes.n_sp = sizes.n_e_total * sizes.pol_n_a;
    sizes.n_pol = h.endo_nbr();
    let expected_nd = sizes.n_a + sizes.n_e;
    let bad: Vec<&str> = orig_pol_symbs
        .iter()
        .filter(|s| find(&values, s).unwrap().ndims() != expected_nd)
        .map(String::as_str)
        .collect();
    if !bad.is_empty() {
        return err(format!(
            "the following fields in `steady_state.pol.values` have a number of dimensions that is not consistent with the number of states and shocks: {}.",
            bad.join(" ")
        ));
    }
    check_permutation(&pol.order, "steady_state.pol.order", &indices.order)?;
    let pol_order_given: Vec<String> = pol.order.clone().unwrap_or_else(|| indices.order.clone());
    // (MATLAB errors in `find` if the order is not a full permutation)
    let dim_of = |name: &str| -> Result<usize> {
        match pol_order_given.iter().position(|x| x == name) {
            Some(p) => Ok(p),
            None => bail!(
                "Misspecified steady-state input `steady_state`: `steady_state.pol.order` does not contain {name}."
            ),
        }
    };
    let order: Vec<usize> = indices.order.iter().map(|x| dim_of(x)).collect::<Result<_>>()?;
    for s in &indices.shocks {
        let d = dim_of(s)?;
        let bad: Vec<&str> = orig_pol_symbs
            .iter()
            .filter(|v| find(&values, v).unwrap().size(d + 1) != sizes.shock(s))
            .map(String::as_str)
            .collect();
        if !bad.is_empty() {
            return err(format!(
                "the following fields in `steady_state.pol.values` have a dimension-{} size that is not compatible with the size steady_state.shocks.grids.{s}.: {}.",
                d + 1,
                bad.join(" ")
            ));
        }
    }
    for s in &indices.states {
        let d = dim_of(s)?;
        let bad: Vec<&str> = orig_pol_symbs
            .iter()
            .filter(|v| find(&values, v).unwrap().size(d + 1) != sizes.pol_state(s))
            .map(String::as_str)
            .collect();
        if !bad.is_empty() {
            return err(format!(
                "the following fields in `steady_state.pol.values` have a dimension-{} size that is not compatible with the size steady_state.pol.grids.{s}.: {}.",
                d + 1,
                bad.join(" ")
            ));
        }
    }
    for (n, a) in values {
        if orig_pol_symbs.contains(&n) {
            data.pol_values.push((n, a.permute(&order)));
        }
    }
    // ---- d
    let dgrids_in = input.d.as_ref().and_then(|d| d.grids.clone());
    let (d_states, d_grids) = validate_d_grids(dgrids_in, &sizes, &indices, &data, opts, !flag_initial_guess)?;
    sizes.d_states = d_states;
    data.d_grids = d_grids;
    if !flag_initial_guess {
        let Some(d) = input.d else { return err("the `steady_state.d` field is missing.".into()) };
        let Some(hist) = d.hist else { return err("the `steady_state.d.hist` field is missing.".into()) };
        if hist.data.iter().any(|v| v.is_nan()) {
            return err("`steady_state.d.hist` contains NaN elements.".into());
        }
        check_permutation(&d.order, "steady_state.d", &indices.order)?;
        let d_order_given = d.order.clone().unwrap_or_else(|| data_pol_order_default(&pol.order, &indices));
        let ddim = |name: &str| -> Result<usize> {
            match d_order_given.iter().position(|x| x == name) {
                Some(p) => Ok(p),
                None => bail!(
                    "Misspecified steady-state input `steady_state`: `steady_state.d.order` does not contain {name}."
                ),
            }
        };
        for s in &indices.shocks {
            let dim = ddim(s)?;
            if hist.size(dim + 1) != sizes.shock(s) {
                return err(format!(
                    "dimension {} of `steady_state.d.hist` and `steady_state.shocks.grids.{s}` have incompatible sizes.",
                    dim + 1
                ));
            }
        }
        for s in &indices.states {
            let dim = ddim(s)?;
            if hist.size(dim + 1) != sizes.d_state(s) {
                return err(format!(
                    "dimension {} of `steady_state.d.hist` and `steady_state.pol.grids.{s}` have incompatible sizes.",
                    dim + 1
                ));
            }
        }
        let order: Vec<usize> = indices.order.iter().map(|x| ddim(x)).collect::<Result<_>>()?;
        data.d_hist = Some(hist.permute(&order));
    }
    sizes.d_n_a = 1;
    for s in &indices.states {
        sizes.d_n_a *= sizes.d_state(s);
    }
    sizes.n_om = sizes.d_n_a * sizes.n_e_total;
    data.pol_order = indices.order.clone();
    data.d_order = indices.order.clone();
    Ok((data, sizes, indices))
}

fn data_pol_order_default(pol_order: &Option<Vec<String>>, indices: &Indices) -> Vec<String> {
    pol_order.clone().unwrap_or_else(|| indices.order.clone())
}

/// `validate_d_grids`: returns `sizes.d.states` and the distribution grids.
fn validate_d_grids(
    d_grids: Option<Vec<(String, Vec<f64>)>>,
    sizes: &Sizes,
    indices: &Indices,
    data: &SteadyStateData,
    opts: &CheckOptions,
    d_grids_required: bool,
) -> Result<(Vec<(String, usize)>, Vec<(String, Vec<f64>)>)> {
    let default = || (sizes.pol_states.clone(), data.pol_grids.clone());
    let warn_pol = |msg: &str| eprintln!("Warning: {msg}");
    let Some(dg) = d_grids else {
        if d_grids_required && !opts.no_warning_d_grids {
            warn_pol(&format!(
                "In the steady-state input `steady_state.d.grids`, no distribution-specific grid is set for states {}. The policy grids in `steady_state.pol.grids` shall be used.",
                indices.states.join(" ")
            ));
        }
        return Ok(default());
    };
    if dg.is_empty() {
        if !opts.no_warning_redundant {
            warn_pol(&format!(
                "In the steady-state input `steady_state.d.grids`, no distribution-specific grid is set for states {}. The policy grids in `steady_state.pol.grids` shall be used.",
                indices.states.join(" ")
            ));
        }
        return Ok(default());
    }
    let used: Vec<&(String, Vec<f64>)> = dg.iter().filter(|(n, _)| indices.states.contains(n)).collect();
    for (n, g) in &used {
        if g.iter().any(|v| v.is_nan()) {
            return err(format!("the following fields in `steady_state.d.grids` contain NaN elements: {n}."));
        }
        if !strictly_increasing(g) {
            return err(format!("the following fields in `steady_state.d.grids` are not strictly increasing: {n}."));
        }
    }
    for (n, g) in &used {
        let p = data.pol_grid(n);
        if g[0] != p[0] {
            return err(format!(
                "`steady_state.d.grids.{n}` has minimum value {:.6} which differs from `steady_state.pol.grids.{n}` minimum {:.6}.",
                g[0], p[0]
            ));
        }
        if g[g.len() - 1] != p[p.len() - 1] {
            return err(format!(
                "`steady_state.d.grids.{n}` has maximum value {:.6} which differs from `steady_state.pol.grids.{n}` maximum {:.6}.",
                g[g.len() - 1],
                p[p.len() - 1]
            ));
        }
    }
    // MATLAB records the states of `d.grids` first, then adds the missing ones from `pol`
    let mut st = Vec::new();
    let mut gr = Vec::new();
    for (n, g) in &used {
        st.push((n.clone(), g.len()));
        gr.push((n.clone(), g.clone()));
    }
    for s in &indices.states {
        if !used.iter().any(|(n, _)| n == s) {
            st.push((s.clone(), sizes.pol_state(s)));
            gr.push((s.clone(), data.pol_grid(s).to_vec()));
        }
    }
    // keep the order of the states for lookups by name (names are unique)
    Ok((st, gr))
}
