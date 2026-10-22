// SPDX-License-Identifier: GPL-3.0-or-later
//! Dulmage–Mendelsohn decomposition, as MATLAB's `dmperm` (CSparse
//! `cs_dmperm` with seed 0: `cs_maxtrans`, `cs_bfs`, `cs_matched`,
//! `cs_unmatched`, `cs_scc`, `cs_dfs`). Used by `dynare_solve.m` for
//! `solve_algo = 2, 4`.

use crate::linalg::Mat;

/// Compressed-column pattern (row indices sorted within each column).
struct Csc {
    m: usize,
    n: usize,
    p: Vec<usize>,
    i: Vec<usize>,
}

impl Csc {
    fn from_dense(a: &Mat) -> Self {
        let (m, n) = (a.nrows(), a.ncols());
        let mut p = Vec::with_capacity(n + 1);
        let mut idx = Vec::new();
        p.push(0);
        for j in 0..n {
            for i in 0..m {
                if a[(i, j)] != 0.0 {
                    idx.push(i);
                }
            }
            p.push(idx.len());
        }
        Csc { m, n, p, i: idx }
    }

    /// cs_transpose (pattern only); row indices come out sorted.
    fn transpose(&self) -> Csc {
        let mut cnt = vec![0usize; self.m + 1];
        for &r in &self.i {
            cnt[r + 1] += 1;
        }
        for k in 0..self.m {
            cnt[k + 1] += cnt[k];
        }
        let p = cnt.clone();
        let mut w = cnt;
        let mut idx = vec![0; self.i.len()];
        for j in 0..self.n {
            for k in self.p[j]..self.p[j + 1] {
                let r = self.i[k];
                idx[w[r]] = j;
                w[r] += 1;
            }
        }
        Csc { m: self.n, n: self.m, p, i: idx }
    }
}

/// Result of [`dmperm`]: `A(p, q)` is block upper triangular, block `k`
/// spans rows `r[k]..r[k+1]` and columns `s[k]..s[k+1]` (0-based).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dmperm {
    pub p: Vec<usize>,
    pub q: Vec<usize>,
    pub r: Vec<usize>,
    pub s: Vec<usize>,
}

/// cs_augment: augmenting path from column k.
#[allow(clippy::too_many_arguments)]
fn augment(
    k: usize,
    a: &Csc,
    jmatch: &mut [isize],
    cheap: &mut [usize],
    w: &mut [isize],
    js: &mut [usize],
    is: &mut [usize],
    ps: &mut [usize],
) {
    let mut found = false;
    let mut i: isize = -1;
    let mut head: isize = 0;
    js[0] = k;
    while head >= 0 {
        let h = head as usize;
        let j = js[h];
        if w[j] != k as isize {
            w[j] = k as isize;
            let mut p = cheap[j];
            while p < a.p[j + 1] && !found {
                i = a.i[p] as isize;
                found = jmatch[i as usize] == -1;
                p += 1;
            }
            cheap[j] = p;
            if found {
                is[h] = i as usize;
                break;
            }
            ps[h] = a.p[j];
        }
        let mut p = ps[h];
        while p < a.p[j + 1] {
            let ii = a.i[p];
            let jm = jmatch[ii];
            if w[jm as usize] == k as isize {
                p += 1;
                continue;
            }
            ps[h] = p + 1;
            is[h] = ii;
            head += 1;
            js[head as usize] = jm as usize;
            break;
        }
        if p == a.p[j + 1] {
            head -= 1;
        }
    }
    if found {
        let mut p = head;
        while p >= 0 {
            jmatch[is[p as usize]] = js[p as usize] as isize;
            p -= 1;
        }
    }
}

/// cs_maxtrans(A, 0): returns (jmatch [m], imatch [n]).
fn maxtrans(a: &Csc) -> (Vec<isize>, Vec<isize>) {
    let (m, n) = (a.m, a.n);
    let mut w = vec![0isize; m];
    let mut n2 = 0;
    let mut k = 0;
    for j in 0..n {
        n2 += usize::from(a.p[j] < a.p[j + 1]);
        for p in a.p[j]..a.p[j + 1] {
            w[a.i[p]] = 1;
            k += usize::from(j == a.i[p]);
        }
    }
    if k == m.min(n) {
        let jmatch = (0..m).map(|i| if i < k { i as isize } else { -1 }).collect();
        let imatch = (0..n).map(|j| if j < k { j as isize } else { -1 }).collect();
        return (jmatch, imatch);
    }
    let m2: usize = w.iter().map(|&x| x as usize).sum();
    let transposed = m2 < n2;
    let ct;
    let c = if transposed {
        ct = a.transpose();
        &ct
    } else {
        a
    };
    let (cm, cn) = (c.m, c.n);
    let mut cjmatch = vec![-1isize; cm];
    let mut cheap: Vec<usize> = c.p[..cn].to_vec();
    let mut wv = vec![-1isize; cn];
    let (mut js, mut is, mut ps) = (vec![0usize; cn], vec![0usize; cn], vec![0usize; cn]);
    for kk in 0..cn {
        augment(kk, c, &mut cjmatch, &mut cheap, &mut wv, &mut js, &mut is, &mut ps);
    }
    let mut cimatch = vec![-1isize; cn];
    for i in 0..cm {
        if cjmatch[i] >= 0 {
            cimatch[cjmatch[i] as usize] = i as isize;
        }
    }
    if transposed {
        // jmatch of C' is imatch of A
        (cimatch, cjmatch)
    } else {
        (cjmatch, cimatch)
    }
}

/// cs_bfs
#[allow(clippy::too_many_arguments)]
fn bfs(
    a: &Csc,
    n: usize,
    wi: &mut [isize],
    wj: &mut [isize],
    queue: &mut [usize],
    imatch: &[isize],
    jmatch: &[isize],
    mark: isize,
) {
    let (mut head, mut tail) = (0, 0);
    for j in 0..n {
        if imatch[j] >= 0 {
            continue;
        }
        wj[j] = 0;
        queue[tail] = j;
        tail += 1;
    }
    if tail == 0 {
        return;
    }
    let ct;
    let c = if mark == 1 {
        a
    } else {
        ct = a.transpose();
        &ct
    };
    while head < tail {
        let j = queue[head];
        head += 1;
        for p in c.p[j]..c.p[j + 1] {
            let i = c.i[p];
            if wi[i] >= 0 {
                continue;
            }
            wi[i] = mark;
            let j2 = jmatch[i] as usize;
            if wj[j2] >= 0 {
                continue;
            }
            wj[j2] = mark;
            queue[tail] = j2;
            tail += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn matched(
    n: usize,
    wj: &[isize],
    imatch: &[isize],
    p: &mut [usize],
    q: &mut [usize],
    cc: &mut [usize; 5],
    rr: &mut [usize; 5],
    set: usize,
    mark: isize,
) {
    let mut kc = cc[set];
    let mut kr = rr[set - 1];
    for j in 0..n {
        if wj[j] != mark {
            continue;
        }
        p[kr] = imatch[j] as usize;
        kr += 1;
        q[kc] = j;
        kc += 1;
    }
    cc[set + 1] = kc;
    rr[set] = kr;
}

fn unmatched(m: usize, wi: &[isize], p: &mut [usize], rr: &mut [usize; 5], set: usize) {
    let mut kr = rr[set];
    for (i, _) in wi.iter().enumerate().take(m).filter(|(_, w)| **w == 0) {
        p[kr] = i;
        kr += 1;
    }
    rr[set + 1] = kr;
}

/// cs_dfs on a graph given by adjacency lists, with a visited mask.
fn dfs(
    j0: usize,
    g: &Csc,
    marked: &mut [bool],
    top: usize,
    out: &mut [usize],
    stack: &mut [usize],
    pstack: &mut [usize],
) -> usize {
    let mut top = top;
    let mut head: isize = 0;
    stack[0] = j0;
    while head >= 0 {
        let h = head as usize;
        let j = stack[h];
        if !marked[j] {
            marked[j] = true;
            pstack[h] = g.p[j];
        }
        let mut done = true;
        let p2 = g.p[j + 1];
        let mut p = pstack[h];
        while p < p2 {
            let i = g.i[p];
            if marked[i] {
                p += 1;
                continue;
            }
            pstack[h] = p;
            head += 1;
            stack[head as usize] = i;
            done = false;
            break;
        }
        if done {
            head -= 1;
            top -= 1;
            out[top] = j;
        }
    }
    top
}

/// cs_scc: returns (p, r) with blocks r[b]..r[b+1] of p.
fn scc(a: &Csc) -> (Vec<usize>, Vec<usize>) {
    let n = a.n;
    let at = a.transpose();
    let mut xi = vec![0usize; n];
    let mut stack = vec![0usize; n + 1];
    let mut pstack = vec![0usize; n + 1];
    let mut marked = vec![false; n];
    let mut top = n;
    for i in 0..n {
        if !marked[i] {
            top = dfs(i, a, &mut marked, top, &mut xi, &mut stack, &mut pstack);
        }
    }
    let mut marked_t = vec![false; n];
    let mut p = vec![0usize; n];
    let mut r = vec![0usize; n + 1];
    top = n;
    let mut nb = n;
    for &i in xi.iter().take(n) {
        if marked_t[i] {
            continue;
        }
        r[nb] = top;
        nb -= 1;
        top = dfs(i, &at, &mut marked_t, top, &mut p, &mut stack, &mut pstack);
    }
    r[nb] = 0;
    for k in nb..=n {
        r[k - nb] = r[k];
    }
    let nblocks = n - nb;
    r.truncate(nblocks + 1);
    let mut blk = vec![0usize; n];
    for b in 0..nblocks {
        for k in r[b]..r[b + 1] {
            blk[p[k]] = b;
        }
    }
    let mut rcopy = r.clone();
    for i in 0..n {
        p[rcopy[blk[i]]] = i;
        rcopy[blk[i]] += 1;
    }
    (p, r)
}

/// MATLAB `[p, q, r, s] = dmperm(A)` on the nonzero pattern of a dense matrix.
pub fn dmperm(a: &Mat) -> Dmperm {
    let csc = Csc::from_dense(a);
    let (m, n) = (csc.m, csc.n);
    let (jmatch, imatch) = maxtrans(&csc);
    let mut p = vec![0usize; m];
    let mut q = vec![0usize; n];
    let mut wi = vec![-1isize; m];
    let mut wj = vec![-1isize; n];
    let mut cc = [0usize; 5];
    let mut rr = [0usize; 5];
    bfs(&csc, n, &mut wi, &mut wj, &mut q, &imatch, &jmatch, 1);
    bfs(&csc, m, &mut wj, &mut wi, &mut p, &jmatch, &imatch, 3);
    unmatched(n, &wj, &mut q, &mut cc, 0);
    matched(n, &wj, &imatch, &mut p, &mut q, &mut cc, &mut rr, 1, 1);
    matched(n, &wj, &imatch, &mut p, &mut q, &mut cc, &mut rr, 2, -1);
    matched(n, &wj, &imatch, &mut p, &mut q, &mut cc, &mut rr, 3, 3);
    unmatched(m, &wi, &mut p, &mut rr, 3);

    // C = A(p, q) restricted to rows R2 and columns C2 (entries keep A's row order)
    let mut pinv = vec![0usize; m];
    for (k, &i) in p.iter().enumerate() {
        pinv[i] = k;
    }
    let nc = cc[3] - cc[2];
    let mut cp = vec![0usize; nc + 1];
    let mut ci = Vec::new();
    for k in 0..nc {
        let j = q[cc[2] + k];
        for t in csc.p[j]..csc.p[j + 1] {
            let i = pinv[csc.i[t]];
            if i >= rr[1] && i < rr[2] {
                ci.push(i - rr[1]);
            }
        }
        cp[k + 1] = ci.len();
    }
    let c = Csc { m: nc, n: nc, p: cp, i: ci };
    let (ps, rs) = scc(&c);
    let nb1 = rs.len() - 1;
    let qs: Vec<usize> = (0..nc).map(|k| q[ps[k] + cc[2]]).collect();
    q[cc[2]..cc[2] + nc].copy_from_slice(&qs);
    let pps: Vec<usize> = (0..nc).map(|k| p[ps[k] + rr[1]]).collect();
    p[rr[1]..rr[1] + nc].copy_from_slice(&pps);
    // fine block partitions
    let cap = m.max(n) + 3;
    let (mut r, mut s) = (vec![0usize; cap], vec![0usize; cap]);
    let mut nb2 = 0;
    if cc[2] > 0 {
        nb2 += 1; // leading coarse block A(R1, [C0 C1])
    }
    for &rk in rs.iter().take(nb1) {
        r[nb2] = rk + rr[1];
        s[nb2] = rk + cc[2];
        nb2 += 1;
    }
    if rr[2] < m {
        r[nb2] = rr[2]; // trailing coarse block A([R3 R0], C3)
        s[nb2] = cc[3];
        nb2 += 1;
    }
    r[nb2] = m;
    s[nb2] = n;
    r.truncate(nb2 + 1);
    s.truncate(nb2 + 1);
    Dmperm { p, q, r, s }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_block_upper_triangular(a: &Mat, d: &Dmperm) -> bool {
        let nb = d.r.len() - 1;
        for b in 0..nb {
            for bb in 0..b {
                // block (b, bb) below the diagonal must be zero
                for ii in d.r[b]..d.r[b + 1] {
                    for jj in d.s[bb]..d.s[bb + 1] {
                        if a[(d.p[ii], d.q[jj])] != 0.0 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    #[test]
    fn recursive_and_simultaneous_blocks() {
        // x0 = 1; x1 + x2 = x0; x2 - x1 = 0; x3 = x2
        let a = Mat::from_row_slice(
            4,
            4,
            &[
                1., 0., 0., 0., //
                -1., 1., 1., 0., //
                0., -1., 1., 0., //
                0., 0., -1., 1.,
            ],
        );
        let d = dmperm(&a);
        assert_eq!(d.r.len() - 1, 3, "{d:?}");
        assert!(is_block_upper_triangular(&a, &d));
        let mut p = d.p.clone();
        p.sort();
        assert_eq!(p, vec![0, 1, 2, 3]);
    }

    #[test]
    fn structurally_singular() {
        // a zero row (unit root equation in the static model) and a free column
        let a = Mat::from_row_slice(3, 3, &[1., 1., 0., 0., 0., 0., 0., 1., 1.]);
        let d = dmperm(&a);
        assert!(is_block_upper_triangular(&a, &d));
        assert_eq!(*d.r.last().unwrap(), 3);
        assert_eq!(*d.s.last().unwrap(), 3);
        let nonsquare = (0..d.r.len() - 1).any(|k| d.r[k + 1] - d.r[k] != d.s[k + 1] - d.s[k]);
        assert!(nonsquare);
    }
}
