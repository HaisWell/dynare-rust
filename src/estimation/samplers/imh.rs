// SPDX-License-Identifier: GPL-3.0-or-later
//! `independent_metropolis_hastings` branch of `posterior_sampler_iteration.m`:
//! proposal `xparam1 + randn*d*diag(jscale)` (or Student), accepted when
//! `log(rand) < logpost - last_logpost + log q(last) - log q(par)`
//! (`rand` drawn only for a finite posterior).
//!
//! The proposals do not depend on the current state, so the next `width`
//! iterations are drawn ahead from a copy of the stream (assuming finite
//! posteriors: one uniform after each proposal inside the bounds) and
//! evaluated in parallel; the walk consumes them in order and restarts the
//! look-ahead after an infinite posterior. The chain is the sequential one.

use anyhow::Result;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::{ProposalDist, Stepper, strictly_inside};
use crate::estimation::{Bounds, Objective};

pub(crate) struct Imh<'a> {
    obj: &'a dyn Objective,
    bounds: &'a Bounds,
    rng: ChaCha8Rng,
    last: Vec<f64>,
    last_lp: f64,
    /// `log q(last)`.
    last_lq: f64,
    center: Vec<f64>,
    prop: ProposalDist,
    width: usize,
}

struct Ahead {
    par: Vec<f64>,
    inside: bool,
    rng_after_normals: ChaCha8Rng,
    rng_after_uniform: ChaCha8Rng,
    uniform: f64,
}

impl<'a> Imh<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        obj: &'a dyn Objective,
        bounds: &'a Bounds,
        rng: ChaCha8Rng,
        x: &[f64],
        lp: f64,
        xparam1: &[f64],
        prop: ProposalDist,
        width: usize,
    ) -> Self {
        let last_lq = prop.logpdf(x, xparam1);
        Imh {
            obj,
            bounds,
            rng,
            last: x.to_vec(),
            last_lp: lp,
            last_lq,
            center: xparam1.to_vec(),
            prop,
            width: width.max(1),
        }
    }
}

impl Stepper for Imh<'_> {
    fn rng(&self) -> &ChaCha8Rng {
        &self.rng
    }

    fn advance(&mut self, max_steps: usize, on_step: &mut dyn FnMut(&[f64], f64, f64, usize)) -> Result<usize> {
        let n = self.last.len();
        // Look ahead: up to `width` evaluations, at most `max_steps` iterations.
        let mut ahead: Vec<Ahead> = Vec::new();
        let mut r = self.rng.clone();
        let mut evals = 0;
        while ahead.len() < max_steps {
            let mut par = vec![0.0; n];
            self.prop.draw(&mut r, &self.center, &mut par);
            let inside = strictly_inside(&par, self.bounds);
            if inside && evals == self.width {
                break;
            }
            let rng_after_normals = r.clone();
            let uniform = if inside {
                evals += 1;
                r.r#gen::<f64>()
            } else {
                0.0
            };
            ahead.push(Ahead { par, inside, rng_after_normals, rng_after_uniform: r.clone(), uniform });
        }
        let obj = self.obj;
        let lps: Vec<f64> = if evals > 1 {
            ahead.par_iter().map(|a| if a.inside { -obj.eval(&a.par).0 } else { f64::NEG_INFINITY }).collect()
        } else {
            ahead.iter().map(|a| if a.inside { -obj.eval(&a.par).0 } else { f64::NEG_INFINITY }).collect()
        };
        let mut done = 0;
        for (a, lp) in ahead.iter().zip(lps) {
            let finite = lp > f64::NEG_INFINITY;
            let acc = if finite {
                let lq = self.prop.logpdf(&a.par, &self.center);
                let ratio = lp - self.last_lp + self.last_lq - lq;
                self.rng = a.rng_after_uniform.clone();
                let acc = a.uniform.ln() < ratio;
                if acc {
                    self.last.copy_from_slice(&a.par);
                    self.last_lp = lp;
                    self.last_lq = lq;
                }
                acc
            } else {
                self.rng = a.rng_after_normals.clone();
                false
            };
            done += 1;
            on_step(&self.last, self.last_lp, acc as u8 as f64, 1);
            if a.inside && !finite {
                break; // the look-ahead assumed a uniform here
            }
        }
        Ok(done)
    }
}
