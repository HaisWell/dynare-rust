// SPDX-License-Identifier: GPL-3.0-or-later
//! Port of `iterate_law_of_motion.m`: one step of the order-1/2/3 decision rule for many
//! particles at once (`local_state_space_iteration_2` / `_3`, pruned or not), parallel over
//! the particles (each column is independent, so the result does not depend on threads).

use super::ReducedForm;
use crate::linalg::Mat;
use rayon::prelude::*;

/// `out += s * G * kron(a, b)` (MATLAB `kron` column order).
fn acc_kron2(out: &mut [f64], g: &Mat, a: &[f64], b: &[f64], s: f64) {
    let nb = b.len();
    let m = g.nrows();
    let gs = g.as_slice();
    for (j, &aj) in a.iter().enumerate() {
        if aj == 0.0 {
            continue;
        }
        for (k, &bk) in b.iter().enumerate() {
            let w = s * aj * bk;
            if w == 0.0 {
                continue;
            }
            let col = &gs[(j * nb + k) * m..(j * nb + k + 1) * m];
            for (o, &gv) in out.iter_mut().zip(col) {
                *o += gv * w;
            }
        }
    }
}

/// `out += s * G * kron(kron(a, b), c)`.
fn acc_kron3(out: &mut [f64], g: &Mat, a: &[f64], b: &[f64], c: &[f64], s: f64) {
    let (nb, nc) = (b.len(), c.len());
    let m = g.nrows();
    let gs = g.as_slice();
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            let ab = ai * bj;
            if ab == 0.0 {
                continue;
            }
            for (k, &ck) in c.iter().enumerate() {
                let w = s * ab * ck;
                if w == 0.0 {
                    continue;
                }
                let col0 = ((i * nb + j) * nc + k) * m;
                for (o, &gv) in out.iter_mut().zip(&gs[col0..col0 + m]) {
                    *o += gv * w;
                }
            }
        }
    }
}

/// `out += s * G * a`.
fn acc_mv(out: &mut [f64], g: &Mat, a: &[f64], s: f64) {
    let m = g.nrows();
    let gs = g.as_slice();
    for (j, &aj) in a.iter().enumerate() {
        let w = s * aj;
        if w == 0.0 {
            continue;
        }
        for (o, &gv) in out.iter_mut().zip(&gs[j * m..(j + 1) * m]) {
            *o += gv * w;
        }
    }
}

/// Result of [`iterate`]: `y` (m x N) and, with pruning, the pruned state space
/// (m x N at order 2, 3m x N at order 3).
pub struct Iterated {
    pub y: Mat,
    pub pruned: Option<Mat>,
}

/// `iterate_law_of_motion(states, epsilon, ReducedForm, M_, options_, false, pruning, pruned)`:
/// `states` ns x N (or ns x 1, repeated for every shock column), `eps` nu x N; `pruned` the
/// pruned state space (ns x N at order 2, 3ns x N at order 3) when pruning.
pub fn iterate(rf: &ReducedForm, states: &Mat, eps: &Mat, pruned: Option<&Mat>) -> Iterated {
    let m = rf.m();
    let ns = rf.ns();
    let nu = rf.nu();
    let n = eps.ncols().max(states.ncols());
    let state_col = |c: usize| if states.ncols() == 1 { 0 } else { c };
    let mut y = Mat::zeros(m, n);
    let prows = match (pruned.is_some(), rf.order) {
        (false, _) => 0,
        (true, 3) => 3 * m,
        (true, _) => m,
    };
    let mut p = Mat::zeros(prows, n);
    let ss = &rf.state_ss;
    let kernel = |buf: &mut (Vec<f64>, Vec<f64>), c: usize, out: &mut [f64], pout: &mut [f64]| {
        let sc = state_col(c);
        let (yhat, u) = (&mut buf.0, &mut buf.1);
        for i in 0..ns {
            yhat[i] = states[(i, sc)] - ss[i];
        }
        for i in 0..nu {
            u[i] = eps[(i, c)];
        }
        let (yhat, u): (&[f64], &[f64]) = (yhat, u);
        match (rf.order, pruned) {
            (1, _) => {
                out.copy_from_slice(&rf.constant);
                acc_mv(out, &rf.ghx, yhat, 1.0);
                acc_mv(out, &rf.ghu, u, 1.0);
            }
            (2, None) => {
                out.copy_from_slice(&rf.constant);
                acc_mv(out, &rf.ghx, yhat, 1.0);
                acc_mv(out, &rf.ghu, u, 1.0);
                acc_kron2(out, &rf.ghxx, yhat, yhat, 0.5);
                acc_kron2(out, &rf.ghuu, u, u, 0.5);
                acc_kron2(out, &rf.ghxu, yhat, u, 1.0);
            }
            (2, Some(pr)) => {
                // local_state_space_iteration_2 with pruning
                let yp: Vec<f64> = (0..ns).map(|i| pr[(i, c)] - ss[i]).collect();
                out.copy_from_slice(&rf.constant);
                acc_mv(out, &rf.ghx, yhat, 1.0);
                acc_mv(out, &rf.ghu, u, 1.0);
                acc_kron2(out, &rf.ghxx, &yp, &yp, 0.5);
                acc_kron2(out, &rf.ghuu, u, u, 0.5);
                acc_kron2(out, &rf.ghxu, &yp, u, 1.0);
                pout.copy_from_slice(&rf.steadystate);
                acc_mv(pout, &rf.ghx, &yp, 1.0);
                acc_mv(pout, &rf.ghu, u, 1.0);
            }
            (3, None) => {
                // local_state_space_iteration_3 without pruning: the full order-3 Taylor
                // expansion around the steady state
                out.copy_from_slice(&rf.steadystate);
                acc_mv(out, &rf.ghx, yhat, 1.0);
                acc_mv(out, &rf.ghu, u, 1.0);
                acc_kron2(out, &rf.ghxx, yhat, yhat, 0.5);
                acc_kron2(out, &rf.ghuu, u, u, 0.5);
                acc_kron2(out, &rf.ghxu, yhat, u, 1.0);
                for (o, g) in out.iter_mut().zip(&rf.ghs2) {
                    *o += 0.5 * g;
                }
                acc_kron3(out, &rf.ghxxx, yhat, yhat, yhat, 1.0 / 6.0);
                acc_kron3(out, &rf.ghuuu, u, u, u, 1.0 / 6.0);
                acc_kron3(out, &rf.ghxxu, yhat, yhat, u, 0.5);
                acc_kron3(out, &rf.ghxuu, yhat, u, u, 0.5);
                acc_mv(out, &rf.ghxss, yhat, 0.5);
                acc_mv(out, &rf.ghuss, u, 0.5);
            }
            (3, Some(pr)) => {
                // pruned (Andreasen et al.): components y1, y2, y3 of the states
                let comp = |k: usize| -> Vec<f64> { (0..ns).map(|i| pr[(k * ns + i, c)] - ss[i]).collect() };
                let (a1, a2, a3) = (comp(0), comp(1), comp(2));
                let mut y1 = vec![0.0; m];
                acc_mv(&mut y1, &rf.ghx, &a1, 1.0);
                acc_mv(&mut y1, &rf.ghu, u, 1.0);
                let mut y2 = vec![0.0; m];
                acc_mv(&mut y2, &rf.ghx, &a2, 1.0);
                acc_kron2(&mut y2, &rf.ghxx, &a1, &a1, 0.5);
                acc_kron2(&mut y2, &rf.ghuu, u, u, 0.5);
                acc_kron2(&mut y2, &rf.ghxu, &a1, u, 1.0);
                for (o, g) in y2.iter_mut().zip(&rf.ghs2) {
                    *o += 0.5 * g;
                }
                let mut y3 = vec![0.0; m];
                acc_mv(&mut y3, &rf.ghx, &a3, 1.0);
                acc_kron2(&mut y3, &rf.ghxx, &a1, &a2, 1.0);
                acc_kron2(&mut y3, &rf.ghxu, &a2, u, 1.0);
                acc_kron3(&mut y3, &rf.ghxxx, &a1, &a1, &a1, 1.0 / 6.0);
                acc_kron3(&mut y3, &rf.ghuuu, u, u, u, 1.0 / 6.0);
                acc_kron3(&mut y3, &rf.ghxxu, &a1, &a1, u, 0.5);
                acc_kron3(&mut y3, &rf.ghxuu, &a1, u, u, 0.5);
                acc_mv(&mut y3, &rf.ghxss, &a1, 0.5);
                acc_mv(&mut y3, &rf.ghuss, u, 0.5);
                for i in 0..m {
                    out[i] = rf.steadystate[i] + y1[i] + y2[i] + y3[i];
                    pout[i] = rf.steadystate[i] + y1[i];
                    pout[m + i] = rf.steadystate[i] + y2[i];
                    pout[2 * m + i] = rf.steadystate[i] + y3[i];
                }
            }
            (k, _) => panic!("iterate_law_of_motion: order {k} is not supported (use 1, 2 or 3)"),
        }
    };
    let init = || (vec![0.0; ns], vec![0.0; nu]);
    let ys = y.as_mut_slice();
    if prows == 0 {
        ys.par_chunks_mut(m).with_min_len(64).enumerate().for_each_init(init, |b, (c, out)| kernel(b, c, out, &mut []));
    } else {
        let ps = p.as_mut_slice();
        ys.par_chunks_mut(m)
            .zip(ps.par_chunks_mut(prows))
            .with_min_len(64)
            .enumerate()
            .for_each_init(init, |b, (c, (out, pout))| kernel(b, c, out, pout));
    }
    Iterated { y, pruned: (prows > 0).then_some(p) }
}

/// `measurement_equations.m`: the observables (`mf1` rows) at zero shocks.
pub fn measurement(rf: &ReducedForm, states: &Mat) -> Mat {
    let eps = Mat::zeros(rf.nu(), states.ncols());
    let y = iterate(rf, states, &eps, None).y;
    super::linalg::select_rows(&y, &rf.mf1)
}
