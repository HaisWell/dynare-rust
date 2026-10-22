// SPDX-License-Identifier: GPL-3.0-or-later
//! Dynare's `qmc_sequence(d, seed, 0, n)` (uniform Sobol / LPTAU points, Gray-code order,
//! up to 1111 dimensions), as `stability_mapping.m` calls it: `qmc_sequence(np, int64(1), 0, Nsam)'`.
//!
//! The direction integers come from Dynare's own sequence (`reference/gsa/dump_sobol.m`
//! reads them off the points at seeds `2^(j+1)-1`, which hold one direction number each),
//! so the points are Dynare's bit for bit for samples of fewer than 2^24 points.

use std::sync::OnceLock;

pub const DIM_MAX: usize = 1111;
/// Bits (= log2 of the longest sequence) of the direction table.
pub const NBITS: usize = 24;

static DIRECTIONS: OnceLock<Vec<[u32; NBITS]>> = OnceLock::new();

/// `v[d][j]` = direction number j of dimension d, scaled to 2^NBITS (`m(d,j) * 2^(NBITS-j-1)`).
fn directions() -> &'static [[u32; NBITS]] {
    DIRECTIONS.get_or_init(|| {
        let txt = include_str!("sobol_directions.txt");
        let mut out = Vec::with_capacity(DIM_MAX);
        for line in txt.lines().filter(|l| !l.starts_with('%') && !l.trim().is_empty()) {
            let mut row = [0u32; NBITS];
            for (j, tok) in line.split_whitespace().enumerate().take(NBITS) {
                let m: u32 = tok.parse().expect("sobol_directions.txt: bad integer");
                row[j] = m << (NBITS - j - 1);
            }
            out.push(row);
        }
        assert_eq!(out.len(), DIM_MAX, "sobol_directions.txt: expected {DIM_MAX} dimensions");
        out
    })
}

/// Points with seeds `seed..seed+n` (Dynare's seed s is the point whose Gray code is s),
/// returned as `n` rows of `d` coordinates (i.e. `qmc_sequence(d, seed, 0, n)'`).
pub fn qmc_sequence(d: usize, seed: u64, n: usize) -> anyhow::Result<Vec<Vec<f64>>> {
    if d == 0 || d > DIM_MAX {
        anyhow::bail!("qmc_sequence: dimension {d} outside 1..={DIM_MAX}");
    }
    let last = seed + n as u64;
    if last > (1u64 << NBITS) {
        anyhow::bail!("qmc_sequence: at most 2^{NBITS} points are supported");
    }
    let v = &directions()[..d];
    let scale = 1.0 / (1u64 << NBITS) as f64;
    // state at `seed`: XOR of the direction numbers of gray(seed)
    let mut x = vec![0u32; d];
    let g = seed ^ (seed >> 1);
    for j in (0..NBITS).filter(|j| g >> j & 1 == 1) {
        for k in 0..d {
            x[k] ^= v[k][j];
        }
    }
    let mut out = Vec::with_capacity(n);
    let mut s = seed;
    for i in 0..n {
        out.push(x.iter().map(|&xi| xi as f64 * scale).collect());
        if i + 1 == n {
            break;
        }
        // next: flip the direction number of the lowest zero bit of s
        let c = (!s).trailing_zeros() as usize;
        for k in 0..d {
            x[k] ^= v[k][c];
        }
        s += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_points() {
        let p = qmc_sequence(3, 1, 8).unwrap();
        let want = [
            [0.5, 0.5, 0.5],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.375, 0.375, 0.625],
            [0.875, 0.875, 0.125],
            [0.625, 0.125, 0.375],
            [0.125, 0.625, 0.875],
            [0.1875, 0.3125, 0.3125],
        ];
        for (r, w) in p.iter().zip(want) {
            assert_eq!(r.as_slice(), w.as_slice());
        }
        let p0 = qmc_sequence(2, 0, 2).unwrap();
        assert_eq!(p0[0], vec![0.0, 0.0]);
        assert_eq!(p0[1], vec![0.5, 0.5]);
    }
}
