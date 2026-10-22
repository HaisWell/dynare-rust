// SPDX-License-Identifier: GPL-3.0-or-later
//! Random draws of the filters: MATLAB's `randn(r, c)` / `rand(r, c)` (column-major fill).
//!
//! Dynare resets the generator to its default state at the start of every filter call
//! (`set_dynare_seed_local_options([],false,'default')`); [`SeededDraws::new`] plays that role
//! here (one fresh stream per evaluation, so the likelihood is a pure function of the
//! parameters). [`ReplayDraws`] feeds back the draws MATLAB logged (`reference/nonlinear/`).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

/// A source of standard normal and uniform(0,1) draws.
pub trait Draws: Send {
    /// Fills `out` with N(0,1) draws (MATLAB `randn`, column-major).
    fn randn(&mut self, out: &mut [f64]);
    /// Fills `out` with U(0,1) draws (MATLAB `rand`, column-major).
    fn rand(&mut self, out: &mut [f64]);
    /// One uniform draw (`rand` / `rand(1,1)`).
    fn rand1(&mut self) -> f64 {
        let mut v = [0.0];
        self.rand(&mut v);
        v[0]
    }
    /// `randn(r, c)` as a matrix.
    fn randn_mat(&mut self, r: usize, c: usize) -> crate::linalg::Mat {
        let mut m = crate::linalg::Mat::zeros(r, c);
        self.randn(m.as_mut_slice());
        m
    }
}

/// ChaCha8 stream (one per likelihood evaluation).
pub struct SeededDraws {
    rng: ChaCha8Rng,
}

impl SeededDraws {
    pub fn new(seed: u64) -> Self {
        SeededDraws { rng: ChaCha8Rng::seed_from_u64(seed) }
    }
}

impl Draws for SeededDraws {
    fn randn(&mut self, out: &mut [f64]) {
        for v in out {
            *v = self.rng.sample(StandardNormal);
        }
    }
    fn rand(&mut self, out: &mut [f64]) {
        for v in out {
            *v = self.rng.r#gen::<f64>();
        }
    }
}

/// Replays logged MATLAB draws (normals and uniforms are separate sequences, each in call
/// order). Running out yields NaN and sets `exhausted`.
#[derive(Debug, Clone)]
pub struct ReplayDraws {
    pub normals: Vec<f64>,
    pub uniforms: Vec<f64>,
    pn: usize,
    pu: usize,
    pub exhausted: bool,
}

impl ReplayDraws {
    pub fn new(normals: Vec<f64>, uniforms: Vec<f64>) -> Self {
        ReplayDraws { normals, uniforms, pn: 0, pu: 0, exhausted: false }
    }
    /// (normals used, uniforms used).
    pub fn used(&self) -> (usize, usize) {
        (self.pn, self.pu)
    }
}

impl Draws for ReplayDraws {
    fn randn(&mut self, out: &mut [f64]) {
        for v in out {
            *v = match self.normals.get(self.pn) {
                Some(&x) => x,
                None => {
                    self.exhausted = true;
                    f64::NAN
                }
            };
            self.pn += 1;
        }
    }
    fn rand(&mut self, out: &mut [f64]) {
        for v in out {
            *v = match self.uniforms.get(self.pu) {
                Some(&x) => x,
                None => {
                    self.exhausted = true;
                    f64::NAN
                }
            };
            self.pu += 1;
        }
    }
}
