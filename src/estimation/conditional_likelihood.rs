// SPDX-License-Identifier: GPL-3.0-or-later
//! `conditional_likelihood`: port of `estimation/dsge_conditional_likelihood_1.m`.
//!
//! With as many shocks as observables and no measurement errors, the shocks are recovered
//! exactly from the data, `eps_t = R(Z,:)^{-1} (y_t - T(Z,:) S_{t-1})`, `S_t = T S_{t-1} + R eps_t`,
//! from `S_0 = 0`, and the likelihood is the product of their Gaussian densities times the
//! Jacobian `|det R(Z,:)|^{-1}` (conditional on the initial state).

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if
)] // index loops mirror the MATLAB code

use anyhow::{Result, bail};

use super::{Bounds, Evaluation, Objective};
use crate::dsge::{DsgeModel, EvalFailure, LikelihoodData};
use crate::linalg::{Mat, Vector};
use crate::modfile::EstimatedKind;

pub struct ConditionalPosterior<'a> {
    pub dsge: &'a DsgeModel,
    pub data: &'a LikelihoodData,
    pub bounds: &'a Bounds,
}

impl ConditionalPosterior<'_> {
    /// The checks of dsge_conditional_likelihood_1.m that do not depend on the parameters.
    pub fn check(dsge: &DsgeModel, data: &LikelihoodData) -> Result<()> {
        if dsge.base_h.iter().any(|&v| v != 0.0)
            || dsge.estimated.iter().any(|e| {
                matches!(e.kind, EstimatedKind::MeStdErr(_) | EstimatedKind::MeCorr(..) | EstimatedKind::Corr(..))
            })
        {
            bail!(
                "Option conditional_likelihood does not support declaration of measurement errors. You can specify the measurement errors in the model block directly by adding measurement equations."
            );
        }
        if data.missing.any_missing {
            bail!("Option conditional_likelihood is not compatible with missing observations.");
        }
        if data.y.nrows() != dsge.model.n_exo() {
            bail!(
                "With conditional_likelihood the number of innovations must be equal to the number of observed varilables!"
            );
        }
        if data.endogenous_prior {
            bail!("conditional_likelihood: endogenous_prior is not supported (Dynare fails on it as well)");
        }
        Ok(())
    }

    pub fn evaluate(&self, x: &[f64]) -> Result<(f64, f64, f64), EvalFailure> {
        let b = self.bounds;
        if x.iter().zip(&b.lb).any(|(v, l)| v < l) {
            let p = x.iter().zip(&b.lb).filter(|(v, l)| v < l).map(|(v, l)| (l - v).powi(2)).sum();
            return Err(EvalFailure { info: 41, penalty: p });
        }
        if x.iter().zip(&b.ub).any(|(v, u)| v > u) {
            let p = x.iter().zip(&b.ub).filter(|(v, u)| v > u).map(|(v, u)| (v - u).powi(2)).sum();
            return Err(EvalFailure { info: 42, penalty: p });
        }
        let res = self.dsge.resolve(x)?;
        // ---- calibration restrictions ----
        if let Some(rs) = &self.data.calib_restrictions {
            crate::gsa::calibration::likelihood_check(self.dsge, &res, rs)?;
        }
        // ---- end calibration restrictions ----
        let idx = &self.dsge.ss_index;
        let rr = res.q.nrows();
        // iQ_upper_chol = chol(inv(Q))
        let iq = crate::linalg::solve(&res.q, &Mat::identity(rr, rr)).ok_or(EvalFailure { info: 50, penalty: 0.1 })?;
        let iq = (&iq + iq.transpose()) * 0.5;
        let l = iq.cholesky().ok_or(EvalFailure { info: 43, penalty: 0.1 })?.l();
        let uq = l.transpose();
        let mut y = self.data.y.clone();
        for (i, &v) in idx.mfys.iter().enumerate() {
            let c = if self.dsge.loglinear { res.steady_state[v].ln() } else { res.steady_state[v] };
            for j in 0..y.ncols() {
                y[(i, j)] -= c;
            }
        }
        if let Some(tr) =
            self.dsge.trend_addition(&res.params, self.data.first_obs, y.ncols(), self.data.prefilter_mean.is_some())
        {
            y -= tr;
        }
        let z = &idx.mf;
        let (mm, pp) = (res.t.nrows(), z.len());
        let rz = Mat::from_fn(pp, rr, |i, j| res.r[(z[i], j)]);
        let tz = Mat::from_fn(pp, mm, |i, j| res.t[(z[i], j)]);
        let (lu, ipiv, _) = crate::linalg::lu(&rz);
        let log_abs_det: f64 = (0..pp).map(|i| lu[(i, i)].abs().ln()).sum();
        let mut ytild = y.clone();
        crate::linalg::lu_solve(&lu, &ipiv, false, &mut ytild);
        let mut ttild = tz;
        crate::linalg::lu_solve(&lu, &ipiv, false, &mut ttild);
        let constant = -0.5 * rr as f64 * (2.0 * std::f64::consts::PI).ln() - log_abs_det
            + (0..rr).map(|i| uq[(i, i)].ln()).sum::<f64>();
        let mut s = Vector::zeros(mm);
        let presample = self.data.kalman.presample;
        let mut lik = 0.0;
        for t in 0..y.ncols() {
            let eps = ytild.column(t) - &ttild * &s;
            s = &res.t * &s + &res.r * &eps;
            if t >= presample {
                let ups = &uq * &eps;
                lik += constant - 0.5 * ups.dot(&ups);
            }
        }
        let likelihood = -lik;
        if likelihood.is_infinite() {
            return Err(EvalFailure { info: 50, penalty: 0.1 });
        }
        if likelihood.is_nan() {
            return Err(EvalFailure { info: 45, penalty: 0.1 });
        }
        let lp = self.dsge.log_prior(x);
        if lp.is_infinite() {
            return Err(EvalFailure { info: 40, penalty: 0.1 });
        }
        if lp.is_nan() {
            return Err(EvalFailure { info: 47, penalty: 0.1 });
        }
        Ok((likelihood - lp, likelihood, lp))
    }
}

impl Objective for ConditionalPosterior<'_> {
    fn dim(&self) -> usize {
        self.dsge.n_estimated()
    }

    fn eval(&self, x: &[f64]) -> (f64, i32) {
        let r = match self.evaluate(x) {
            Ok(e) => (e.0, 0),
            Err(f) => (f64::INFINITY, f.info),
        };
        crate::progress::eval(r.0);
        r
    }

    fn eval_full(&self, x: &[f64], _contributions: bool) -> Evaluation {
        let r = self.evaluate(x);
        crate::progress::eval(r.as_ref().map_or(f64::INFINITY, |e| e.0));
        match r {
            Ok(e) => Evaluation { fval: e.0, info: 0, info4: 0.0, contributions: None },
            Err(f) => Evaluation { fval: f64::INFINITY, info: f.info, info4: f.penalty, contributions: None },
        }
    }
}
