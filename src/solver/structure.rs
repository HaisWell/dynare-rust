// SPDX-License-Identifier: GPL-3.0-or-later
//! Decision-rule ordering of the endogenous variables.

/// The DR ordering of Dynare: `dr.order_var`, `dr.inv_order_var`, `dr.state_var`
/// and `M_.nstatic/npred/nboth/nfwrd/nspred/nsfwrd/ndynamic`.
///
/// Port of `stochastic_solver/set_state_space.m` (the preprocessor computes
/// the same counts into `M_`): static variables (no lag, no lead), purely
/// predetermined (lag only), both (lag and lead), purely forward (lead only),
/// each group in declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrStructure {
    pub n_endo: usize,
    pub n_exo: usize,
    /// DR position -> declaration index.
    pub order_var: Vec<usize>,
    /// Declaration index -> DR position.
    pub inv_order_var: Vec<usize>,
    /// State variables (declaration indices): purely predetermined then both,
    /// as `dyn_first_order_solver.m`'s `[no_both_lag_id, both_id]`.
    pub state_var: Vec<usize>,
    pub nstatic: usize,
    pub npred: usize,
    pub nboth: usize,
    pub nfwrd: usize,
    pub nspred: usize,
    pub nsfwrd: usize,
    pub ndynamic: usize,
    /// `M_.maximum_endo_lag` / `M_.maximum_endo_lead` (0 or 1 after aux substitution).
    pub maximum_endo_lag: usize,
    pub maximum_endo_lead: usize,
    /// Whether variable `i` (declaration order) appears at `t` (`lead_lag_incidence(cur, i) != 0`).
    pub current: Vec<bool>,
}

impl DrStructure {
    /// `lli[i] = [lag, current, lead]` for endogenous variable `i` (declaration
    /// order): nonzero iff the variable appears with that timing (as
    /// `M_.lead_lag_incidence'`).
    pub fn new(lli: &[[usize; 3]], n_exo: usize) -> Self {
        let n = lli.len();
        let lag: Vec<bool> = lli.iter().map(|r| r[0] != 0).collect();
        let lead: Vec<bool> = lli.iter().map(|r| r[2] != 0).collect();
        let current: Vec<bool> = lli.iter().map(|r| r[1] != 0).collect();
        let maximum_endo_lag = usize::from(lag.iter().any(|&b| b));
        let maximum_endo_lead = usize::from(lead.iter().any(|&b| b));
        let (mut stat, mut pred, mut both, mut fwrd) = (vec![], vec![], vec![], vec![]);
        for i in 0..n {
            match (lag[i], lead[i]) {
                (false, false) => stat.push(i),
                (true, false) => pred.push(i),
                (true, true) => both.push(i),
                (false, true) => fwrd.push(i),
            }
        }
        let (nstatic, npred, nboth, nfwrd) = (stat.len(), pred.len(), both.len(), fwrd.len());
        let state_var: Vec<usize> = pred.iter().chain(&both).copied().collect();
        let order_var: Vec<usize> = stat.into_iter().chain(pred).chain(both).chain(fwrd).collect();
        let mut inv_order_var = vec![0; n];
        for (k, &v) in order_var.iter().enumerate() {
            inv_order_var[v] = k;
        }
        DrStructure {
            n_endo: n,
            n_exo,
            order_var,
            inv_order_var,
            state_var,
            nstatic,
            npred,
            nboth,
            nfwrd,
            nspred: npred + nboth,
            nsfwrd: nfwrd + nboth,
            ndynamic: n - nstatic,
            maximum_endo_lag,
            maximum_endo_lead,
            current,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_ordering() {
        // x0: static, x1: lag+lead (both), x2: lag only, x3: lead only, x4: static
        let lli = [[0, 1, 0], [1, 2, 4], [2, 3, 0], [0, 4, 5], [0, 5, 0]];
        let s = DrStructure::new(&lli, 1);
        assert_eq!(s.order_var, vec![0, 4, 2, 1, 3]);
        assert_eq!(s.inv_order_var, vec![0, 3, 2, 4, 1]);
        assert_eq!(s.state_var, vec![2, 1]);
        assert_eq!((s.nstatic, s.npred, s.nboth, s.nfwrd, s.nspred, s.nsfwrd), (2, 1, 1, 1, 2, 2));
    }
}
