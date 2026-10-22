// SPDX-License-Identifier: GPL-3.0-or-later
//! Dates in deterministic statements (`shocks`, `mshocks`, `shocks(learnt_in=DATE)`,
//! `endval(learnt_in=DATE)`, `perfect_foresight_controlled_paths`, `shock_paths`): port of
//! `perfect-foresight-models/get_simulation_periods.m` and of the `dates -
//! first_simulation_period + 1` conversions of `make_ex_.m`,
//! `perfect_foresight_with_expectation_errors_setup.m` and `controlled_paths_by_period.m`.
//!
//! `perfect_foresight_setup(first_simulation_period = DATE | last_simulation_period = DATE,
//! periods = INTEGER)` (or both dates without `periods`) sets `options_.simul.
//! first_simulation_period` / `last_simulation_period`; a date `d` in a statement is then the
//! simulation period `d - first_simulation_period + 1`.

use anyhow::{Result, bail};

use crate::dates::Date;

/// `options_.periods`, `options_.simul.first_simulation_period` and `last_simulation_period`
/// (`get_simulation_periods.m`): the number of periods and the first/last simulation dates.
pub fn get_simulation_periods(
    periods: usize,
    first: Option<Date>,
    last: Option<Date>,
) -> Result<(usize, Option<Date>, Option<Date>)> {
    if periods > 0 {
        let (mut f, mut l) = (first, last);
        match (first, last) {
            (Some(a), Some(b)) => {
                if a.diff(b).map(|d| d + 1) != Some(periods as i64) {
                    bail!("Options first_simulation_period, last_simulation_period and periods are inconsistent");
                }
            }
            (Some(a), None) => l = Some(a.add(periods as i64 - 1)),
            (None, Some(b)) => f = Some(b.add(-(periods as i64) + 1)),
            (None, None) => {}
        }
        return Ok((periods, f, l));
    }
    let (Some(a), Some(b)) = (first, last) else {
        bail!("Options first_simulation_period and last_simulation_period must be set when periods is not provided");
    };
    let Some(d) = a.diff(b) else {
        bail!("first_simulation_period and last_simulation_period must have the same frequency");
    };
    if d < 0 {
        bail!("last_simulation_period must not be before first_simulation_period");
    }
    Ok((d as usize + 1, first, last))
}

/// `d - first_simulation_period + 1`.
pub fn date_to_period(d: Date, first: Option<Date>) -> Result<i64> {
    let Some(f) = first else {
        bail!(
            "a period is given as a date ({}) but neither first_simulation_period nor last_simulation_period option was passed",
            d.label()
        );
    };
    match f.diff(d) {
        Some(k) => Ok(k + 1),
        None => bail!("the date {} does not have the frequency of first_simulation_period ({})", d.label(), f.label()),
    }
}

/// Integer periods as they are, or dates converted with [`date_to_period`].
pub fn to_sim_periods(periods: &[i64], dates: &[Date], first: Option<Date>) -> Result<Vec<i64>> {
    if dates.is_empty() {
        return Ok(periods.to_vec());
    }
    dates.iter().map(|&d| date_to_period(d, first)).collect()
}

/// A `learnt_in` option: the integer, or the date converted with [`date_to_period`].
pub fn learnt_in_period(int: usize, date: Option<Date>, first: Option<Date>) -> Result<usize> {
    match date {
        None => Ok(int),
        Some(d) => {
            let p = date_to_period(d, first)?;
            if p < 1 {
                bail!("learnt_in = {}: the date is before the first simulation period", d.label());
            }
            Ok(p as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_periods() {
        let q = |s: &str| Date::parse(s).unwrap();
        assert_eq!(
            get_simulation_periods(10, Some(q("2000Q1")), None).unwrap(),
            (10, Some(q("2000Q1")), Some(q("2002Q2")))
        );
        assert_eq!(
            get_simulation_periods(10, None, Some(q("2002Q2"))).unwrap(),
            (10, Some(q("2000Q1")), Some(q("2002Q2")))
        );
        assert_eq!(get_simulation_periods(0, Some(q("2000Q1")), Some(q("2002Q2"))).unwrap().0, 10);
        assert!(get_simulation_periods(9, Some(q("2000Q1")), Some(q("2002Q2"))).is_err());
        assert!(get_simulation_periods(0, Some(q("2000Q1")), None).is_err());
        assert_eq!(to_sim_periods(&[], &[q("2000Q3"), q("2000Q4")], Some(q("2000Q1"))).unwrap(), vec![3, 4]);
        assert!(to_sim_periods(&[], &[q("2000Q3")], None).is_err());
        assert_eq!(learnt_in_period(0, Some(q("2001Q1")), Some(q("2000Q1"))).unwrap(), 5);
    }
}
