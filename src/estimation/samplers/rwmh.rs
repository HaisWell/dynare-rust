// SPDX-License-Identifier: GPL-3.0-or-later
//! `random_walk_metropolis_hastings` branch of `posterior_sampler_iteration.m`
//! with a general proposal (`rand_multivariate_student`). The Gaussian random
//! walk keeps using [`crate::estimation::rw_metropolis`] (with pre-fetching);
//! this one is sequential.

use anyhow::Result;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use super::{ProposalDist, Stepper, strictly_inside};
use crate::estimation::{Bounds, Objective};

pub(crate) struct RwStudent<'a> {
    obj: &'a dyn Objective,
    bounds: &'a Bounds,
    rng: ChaCha8Rng,
    last: Vec<f64>,
    last_lp: f64,
    prop: ProposalDist,
    par: Vec<f64>,
}

impl<'a> RwStudent<'a> {
    pub fn new(
        obj: &'a dyn Objective,
        bounds: &'a Bounds,
        rng: ChaCha8Rng,
        x: &[f64],
        lp: f64,
        prop: ProposalDist,
    ) -> Self {
        RwStudent { obj, bounds, rng, last: x.to_vec(), last_lp: lp, prop, par: vec![0.0; x.len()] }
    }
}

impl Stepper for RwStudent<'_> {
    fn rng(&self) -> &ChaCha8Rng {
        &self.rng
    }

    fn advance(&mut self, _max_steps: usize, on_step: &mut dyn FnMut(&[f64], f64, f64, usize)) -> Result<usize> {
        self.prop.draw(&mut self.rng, &self.last, &mut self.par);
        let lp = if strictly_inside(&self.par, self.bounds) { -self.obj.eval(&self.par).0 } else { f64::NEG_INFINITY };
        let acc = lp > f64::NEG_INFINITY && self.rng.r#gen::<f64>().ln() < lp - self.last_lp;
        if acc {
            self.last.copy_from_slice(&self.par);
            self.last_lp = lp;
        }
        on_step(&self.last, self.last_lp, acc as u8 as f64, 1);
        Ok(1)
    }
}
