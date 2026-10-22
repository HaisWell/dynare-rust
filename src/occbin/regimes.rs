// SPDX-License-Identifier: GPL-3.0-or-later
//! Regime structs as the OccBin filter and smoother handle them: one [`Regime`]
//! (`regime`/`regimestart`) per constraint for one period (MATLAB's `regime_history(t)` with
//! fields `regime, regimestart` or `regime1, regimestart1, regime2, regimestart2`).

use super::solver::{Regime, map_regime};

/// The regimes of all constraints in one period.
pub type Regimes = Vec<Regime>;

/// `base_regime`: every constraint slack from period 1 on.
pub fn base(nc: usize) -> Regimes {
    vec![Regime { regime: vec![false], start: vec![1] }; nc]
}

pub fn is_base(r: &Regimes) -> bool {
    r.iter().all(|g| g.regime == [false] && g.start == [1])
}

/// Any constraint binding at any expected period.
pub fn any_binding(r: &[Regimes]) -> bool {
    r.iter().flatten().any(|g| g.regime.iter().any(|&b| b))
}

/// Largest expected regime start over the periods and constraints (`max([regimes_.regimestart])`).
pub fn max_start(r: &[Regimes]) -> usize {
    r.iter().flatten().flat_map(|g| g.start.iter().copied()).max().unwrap_or(1)
}

/// Port of `backward_map_regime.m` (two-argument form): the binding indicator of length
/// `regime_start(end)` (the last entry is the slack terminal period).
pub fn backward_map_regime(g: &Regime) -> Vec<bool> {
    let n = *g.start.last().unwrap_or(&1);
    let mut bi = vec![false; n];
    if g.regime.len() > 1 {
        for ir in 0..g.regime.len() - 1 {
            for t in g.start[ir]..g.start[ir + 1] {
                bi[t - 1] = g.regime[ir];
            }
        }
    }
    bi
}

/// The regime expected one period later: `map_regime(backward_map_regime(g)(2:end))`.
pub fn shift(g: &Regime) -> Regime {
    let bi = backward_map_regime(g);
    let (regime, start, _) = map_regime(if bi.len() > 1 { &bi[1..] } else { &[] });
    Regime { regime, start }
}

/// `[regime, regimestart]` of a guess built from literal values.
pub fn regime(regime: &[u8], start: &[usize]) -> Regime {
    Regime { regime: regime.iter().map(|&b| b != 0).collect(), start: start.to_vec() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_and_shift() {
        let g = regime(&[0, 1, 0], &[1, 3, 5]);
        assert_eq!(backward_map_regime(&g), vec![false, false, true, true, false]);
        assert_eq!(shift(&g), regime(&[0, 1, 0], &[1, 2, 4]));
        assert_eq!(shift(&regime(&[1, 0], &[1, 2])), regime(&[0], &[1]));
        assert_eq!(shift(&regime(&[0], &[1])), regime(&[0], &[1]));
    }
}
