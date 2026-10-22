// SPDX-License-Identifier: GPL-3.0-or-later
//! Graph algorithms of the preprocessor's block decomposition, ported so that ties resolve as in
//! the Dynare 7.1 binary (GCC 15 / libstdc++ and Boost.Graph; the 7.2 references agree):
//!
//! * [`edmonds_matching`]: `boost::edmonds_maximum_cardinality_matching` (extra-greedy initial
//!   matching with libstdc++'s `std::sort` + `std::stable_sort`, then Edmonds' augmenting paths),
//! * [`DepGraph`]: `VariableDependencyGraph` (Boost `adjacency_list<listS, listS,
//!   bidirectionalS>`, iteration order and the iterator quirks of its elimination rules included),
//!   with `sortedStronglyConnectedComponents` (Boost `strong_components` = Tarjan, then
//!   `topological_sort` of the condensation), `minimalSetOfFeedbackVertices` and
//!   `reorderRecursiveVariables`.

use std::collections::BTreeSet;

const NULL: usize = usize::MAX;

// ---- libstdc++ std::sort ---------------------------------------------------------------------

/// `std::sort` as implemented by libstdc++ (introsort, median-of-three pivot, final insertion
/// sort with threshold 16, heap sort fallback). Not stable: the permutation of equivalent
/// elements is the one libstdc++ produces.
pub fn std_sort<T: Copy>(v: &mut [T], less: &dyn Fn(&T, &T) -> bool) {
    let n = v.len();
    if n < 2 {
        return;
    }
    let lg = (usize::BITS - 1 - n.leading_zeros()) as usize;
    introsort_loop(v, 0, n, 2 * lg, less);
    final_insertion_sort(v, 0, n, less);
}

const THRESHOLD: usize = 16;

fn introsort_loop<T: Copy>(
    v: &mut [T],
    first: usize,
    mut last: usize,
    mut depth: usize,
    less: &dyn Fn(&T, &T) -> bool,
) {
    while last - first > THRESHOLD {
        if depth == 0 {
            // __partial_sort(first, last, last): heap select + sort heap
            make_heap(v, first, last, less);
            let mut l = last;
            while l - first > 1 {
                l -= 1;
                pop_heap(v, first, l, l, less);
            }
            return;
        }
        depth -= 1;
        let cut = unguarded_partition_pivot(v, first, last, less);
        introsort_loop(v, cut, last, depth, less);
        last = cut;
    }
}

fn move_median_to_first<T: Copy>(
    v: &mut [T],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    less: &dyn Fn(&T, &T) -> bool,
) {
    if less(&v[a], &v[b]) {
        if less(&v[b], &v[c]) {
            v.swap(result, b);
        } else if less(&v[a], &v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if less(&v[a], &v[c]) {
        v.swap(result, a);
    } else if less(&v[b], &v[c]) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

fn unguarded_partition<T: Copy>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    less: &dyn Fn(&T, &T) -> bool,
) -> usize {
    loop {
        while less(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while less(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

fn unguarded_partition_pivot<T: Copy>(v: &mut [T], first: usize, last: usize, less: &dyn Fn(&T, &T) -> bool) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, less);
    unguarded_partition(v, first + 1, last, first, less)
}

fn unguarded_linear_insert<T: Copy>(v: &mut [T], mut last: usize, less: &dyn Fn(&T, &T) -> bool) {
    let val = v[last];
    let mut next = last - 1;
    while less(&val, &v[next]) {
        v[last] = v[next];
        last = next;
        next = next.wrapping_sub(1);
    }
    v[last] = val;
}

fn insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, less: &dyn Fn(&T, &T) -> bool) {
    if first == last {
        return;
    }
    for i in first + 1..last {
        if less(&v[i], &v[first]) {
            let val = v[i];
            v.copy_within(first..i, first + 1);
            v[first] = val;
        } else {
            unguarded_linear_insert(v, i, less);
        }
    }
}

fn final_insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, less: &dyn Fn(&T, &T) -> bool) {
    if last - first > THRESHOLD {
        insertion_sort(v, first, first + THRESHOLD, less);
        for i in first + THRESHOLD..last {
            unguarded_linear_insert(v, i, less);
        }
    } else {
        insertion_sort(v, first, last, less);
    }
}

fn adjust_heap<T: Copy>(
    v: &mut [T],
    first: usize,
    mut hole: usize,
    len: usize,
    value: T,
    less: &dyn Fn(&T, &T) -> bool,
) {
    let top = hole;
    let mut second = hole;
    while second < (len.saturating_sub(1)) / 2 {
        second = 2 * (second + 1);
        if less(&v[first + second], &v[first + second - 1]) {
            second -= 1;
        }
        v[first + hole] = v[first + second];
        hole = second;
    }
    if len & 1 == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[first + hole] = v[first + second - 1];
        hole = second - 1;
    }
    // __push_heap
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && less(&v[first + parent], &value) {
        v[first + hole] = v[first + parent];
        hole = parent;
        parent = hole.wrapping_sub(1) / 2;
    }
    v[first + hole] = value;
}

fn make_heap<T: Copy>(v: &mut [T], first: usize, last: usize, less: &dyn Fn(&T, &T) -> bool) {
    let len = last - first;
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[first + parent];
        adjust_heap(v, first, parent, len, value, less);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

fn pop_heap<T: Copy>(v: &mut [T], first: usize, last: usize, result: usize, less: &dyn Fn(&T, &T) -> bool) {
    let value = v[result];
    v[result] = v[first];
    adjust_heap(v, first, 0, last - first, value, less);
}

/// `std::stable_sort`.
fn stable_sort<T: Copy>(v: &mut [T], less: &dyn Fn(&T, &T) -> bool) {
    v.sort_by(|a, b| {
        if less(a, b) {
            std::cmp::Ordering::Less
        } else if less(b, a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
}

// ---- Boost maximum cardinality matching -------------------------------------------------------

/// Simple union-find (the representative chosen does not matter for the algorithm, which only
/// reads `origin[find(x)]`).
struct Dsu(Vec<usize>);

impl Dsu {
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.0[r] != r {
            r = self.0[r];
        }
        let mut y = x;
        while self.0[y] != r {
            let nx = self.0[y];
            self.0[y] = r;
            y = nx;
        }
        r
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[ra] = rb;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum St {
    Even,
    Odd,
    Unreached,
}

struct Edmonds<'a> {
    adj: &'a [Vec<usize>],
    mate: Vec<usize>,
    anc_v: Vec<usize>,
    anc_w: Vec<usize>,
    state: Vec<St>,
    origin: Vec<usize>,
    pred: Vec<usize>,
    bridge: Vec<(usize, usize)>,
    ds: Dsu,
    aug_path: std::collections::VecDeque<usize>,
    even_edges: Vec<(usize, usize)>,
}

impl Edmonds<'_> {
    fn parent(&mut self, x: usize) -> usize {
        if self.state[x] == St::Even && self.mate[x] != NULL {
            self.mate[x]
        } else if self.state[x] == St::Odd {
            let r = self.ds.find(self.pred[x]);
            self.origin[r]
        } else {
            x
        }
    }

    fn push_edges(&mut self, u: usize) {
        for &t in &self.adj[u] {
            if t != u {
                self.even_edges.push((u, t));
            }
        }
    }

    fn link_and_set_bridges(&mut self, x: usize, stop: usize, the_bridge: (usize, usize)) {
        let mut v = x;
        while v != stop {
            self.ds.union(v, stop);
            let r = self.ds.find(stop);
            self.origin[r] = stop;
            if self.state[v] == St::Odd {
                self.bridge[v] = the_bridge;
                self.push_edges(v);
            }
            v = self.parent(v);
        }
    }

    fn retrieve(&mut self, v: usize, w: usize) {
        if v == w {
            self.aug_path.push_back(v);
        } else if self.state[v] == St::Even {
            self.aug_path.push_back(v);
            let m = self.mate[v];
            self.aug_path.push_back(m);
            self.retrieve(self.pred[m], w);
        } else {
            self.aug_path.push_back(v);
            let (b0, b1) = self.bridge[v];
            self.reversed_retrieve(b0, self.mate[v]);
            self.retrieve(b1, w);
        }
    }

    fn reversed_retrieve(&mut self, v: usize, w: usize) {
        if v == w {
            self.aug_path.push_back(v);
        } else if self.state[v] == St::Even {
            let m = self.mate[v];
            self.reversed_retrieve(self.pred[m], w);
            self.aug_path.push_back(m);
            self.aug_path.push_back(v);
        } else {
            let (b0, b1) = self.bridge[v];
            self.reversed_retrieve(b1, w);
            self.retrieve(b0, self.mate[v]);
            self.aug_path.push_back(v);
        }
    }

    fn augment(&mut self) -> bool {
        let n = self.adj.len();
        let mut timestamp = 0usize;
        self.even_edges.clear();
        for u in 0..n {
            self.origin[u] = u;
            self.pred[u] = u;
            self.anc_v[u] = 0;
            self.anc_w[u] = 0;
            self.ds.0[u] = u;
            if self.mate[u] == NULL {
                self.state[u] = St::Even;
                self.push_edges(u);
            } else {
                self.state[u] = St::Unreached;
            }
        }
        let (mut v, mut w) = (NULL, NULL);
        let (mut w_free, mut v_free) = (NULL, NULL);
        let mut found = false;
        while !found {
            let Some((a, b)) = self.even_edges.pop() else { break };
            v = a;
            w = b;
            let rv = self.ds.find(v);
            let mut vp = self.origin[rv];
            let rw = self.ds.find(w);
            let mut wp = self.origin[rw];
            if self.state[vp] != St::Even {
                std::mem::swap(&mut vp, &mut wp);
                std::mem::swap(&mut v, &mut w);
            }
            if self.state[wp] == St::Unreached {
                self.state[wp] = St::Odd;
                let wm = self.mate[wp];
                self.state[wm] = St::Even;
                self.push_edges(wm);
                self.pred[wp] = v;
            } else if self.state[wp] == St::Even && wp != vp {
                let (mut w_up, mut v_up) = (wp, vp);
                let mut nca = NULL;
                w_free = NULL;
                v_free = NULL;
                timestamp += 1;
                while nca == NULL && (v_free == NULL || w_free == NULL) {
                    self.anc_w[w_up] = timestamp;
                    self.anc_v[v_up] = timestamp;
                    if w_free == NULL {
                        w_up = self.parent(w_up);
                    }
                    if v_free == NULL {
                        v_up = self.parent(v_up);
                    }
                    if self.mate[v_up] == NULL {
                        v_free = v_up;
                    }
                    if self.mate[w_up] == NULL {
                        w_free = w_up;
                    }
                    if self.anc_w[v_up] == timestamp {
                        nca = v_up;
                    } else if self.anc_v[w_up] == timestamp {
                        nca = w_up;
                    } else if v_free == w_free && v_free != NULL {
                        nca = v_up;
                    }
                }
                if nca == NULL {
                    found = true;
                } else {
                    self.link_and_set_bridges(wp, nca, (w, v));
                    self.link_and_set_bridges(vp, nca, (v, w));
                }
            }
        }
        if !found {
            return false;
        }
        self.aug_path.clear();
        self.reversed_retrieve(v, v_free);
        self.retrieve(w, w_free);
        while let (Some(a), Some(b)) = (self.aug_path.pop_front(), self.aug_path.pop_front()) {
            self.mate[a] = b;
            self.mate[b] = a;
        }
        true
    }
}

/// `boost::edmonds_maximum_cardinality_matching` on the undirected graph with `nv` vertices
/// (`adjacency_list<vecS, vecS, undirectedS>`) whose edges were added in `edges` order. Returns
/// the mate of each vertex (`None` if unmatched).
pub fn edmonds_matching(nv: usize, edges: &[(usize, usize)]) -> Vec<Option<usize>> {
    let mut adj = vec![Vec::new(); nv];
    for &(u, v) in edges {
        adj[u].push(v);
        if u != v {
            adj[v].push(u);
        }
    }
    // extra_greedy_matching
    let mut mate = vec![NULL; nv];
    let mut el: Vec<(usize, usize)> = Vec::with_capacity(2 * edges.len());
    for &(u, v) in edges {
        if u == v {
            continue;
        }
        el.push((u, v));
        el.push((v, u));
    }
    let deg: Vec<usize> = adj.iter().map(Vec::len).collect();
    std_sort(&mut el, &|x, y| deg[x.1] < deg[y.1]);
    stable_sort(&mut el, &|x, y| deg[x.0] < deg[y.0]);
    for &(a, b) in &el {
        if mate[a] == mate[b] {
            mate[a] = b;
            mate[b] = a;
        }
    }
    let mut e = Edmonds {
        adj: &adj,
        mate,
        anc_v: vec![0; nv],
        anc_w: vec![0; nv],
        state: vec![St::Unreached; nv],
        origin: vec![0; nv],
        pred: vec![0; nv],
        bridge: vec![(NULL, NULL); nv],
        ds: Dsu((0..nv).collect()),
        aug_path: Default::default(),
        even_edges: Vec::new(),
    };
    while e.augment() {}
    e.mate.into_iter().map(|m| (m != NULL).then_some(m)).collect()
}

// ---- VariableDependencyGraph ------------------------------------------------------------------

/// `VariableDependencyGraph`: a directed graph whose vertices keep their `vertex_index`
/// (`id`) and `vertex_index1` (`index1`) properties; `verts` is Boost's vertex list (`vertex(i, G)`
/// is `verts[i]`), `out`/`inn` the per-vertex edge lists in insertion order.
#[derive(Clone, Debug)]
pub struct DepGraph {
    verts: Vec<usize>,
    pub index1: Vec<usize>,
    out: Vec<Vec<usize>>,
    inn: Vec<Vec<usize>>,
}

impl DepGraph {
    pub fn new(n: usize) -> Self {
        DepGraph { verts: (0..n).collect(), index1: vec![0; n], out: vec![Vec::new(); n], inn: vec![Vec::new(); n] }
    }

    pub fn num_vertices(&self) -> usize {
        self.verts.len()
    }

    /// `add_edge(vertex(u, G), vertex(v, G), G)` (positions in the vertex list).
    pub fn add_edge_pos(&mut self, u: usize, v: usize) {
        let (a, b) = (self.verts[u], self.verts[v]);
        self.add_edge(a, b);
    }

    fn add_edge(&mut self, a: usize, b: usize) {
        self.out[a].push(b);
        self.inn[b].push(a);
    }

    fn has_edge(&self, a: usize, b: usize) -> bool {
        self.out[a].contains(&b)
    }

    fn suppress(&mut self, v: usize) {
        for t in std::mem::take(&mut self.out[v]) {
            if t != v {
                self.inn[t].retain(|&s| s != v);
            }
        }
        for s in std::mem::take(&mut self.inn[v]) {
            if s != v {
                self.out[s].retain(|&t| t != v);
            }
        }
        self.verts.retain(|&x| x != v);
    }

    fn eliminate(&mut self, v: usize) {
        if !self.inn[v].is_empty() && !self.out[v].is_empty() {
            let (ins, outs) = (self.inn[v].clone(), self.out[v].clone());
            for &s in &ins {
                for &t in &outs {
                    if !self.has_edge(s, t) {
                        self.add_edge(s, t);
                    }
                }
            }
        }
        self.suppress(v);
    }

    /// The vertex-list sweep used by the elimination rules: `act` returns true when it removed
    /// the current vertex; the iterator then moves as the C++ code does (`it = ita`, or back to
    /// `begin()` when `i == 0`, which skips the new first vertex).
    fn sweep(&mut self, act: &mut dyn FnMut(&mut Self, usize) -> bool) -> bool {
        let mut done = false;
        let (mut pos, mut i) = (0usize, 0isize);
        while pos < self.verts.len() {
            let v = self.verts[pos];
            if act(self, v) {
                done = true;
                if i > 0 {
                    pos -= 1;
                } else {
                    pos = 0;
                    i -= 1;
                }
            }
            pos += 1;
            i += 1;
        }
        done
    }

    fn has_cycle(&self) -> bool {
        // 0 white, 1 gray, 2 black
        let mut color = vec![0u8; self.out.len()];
        fn dfs(g: &DepGraph, u: usize, color: &mut [u8]) -> bool {
            color[u] = 1;
            for &t in &g.out[u] {
                if color[t] == 0 && dfs(g, t, color) {
                    return true;
                } else if color[t] == 1 {
                    return true;
                }
            }
            color[u] = 2;
            false
        }
        for &v in &self.verts {
            if color[v] == 0 && dfs(self, v, &mut color) {
                return true;
            }
        }
        false
    }

    fn belongs_to_clique(&self, v: usize) -> bool {
        let (ins, outs) = (&self.inn[v], &self.out[v]);
        let mut liste = Vec::new();
        let mut agree = true;
        let mut k = 0;
        while k < ins.len() && k < outs.len() && agree {
            agree = ins[k] == outs[k] && ins[k] != v;
            liste.push(ins[k]);
            k += 1;
        }
        if agree {
            if k < ins.len() || k < outs.len() {
                agree = false;
            }
            let mut i = 1;
            while i < liste.len() && agree {
                let mut j = i + 1;
                while j < liste.len() && agree {
                    agree = self.has_edge(liste[i], liste[j]) && self.has_edge(liste[j], liste[i]);
                    j += 1;
                }
                i += 1;
            }
        }
        agree
    }

    fn elim_low_degree(&mut self) -> bool {
        self.sweep(&mut |g, v| {
            let (di, dout) = (g.inn[v].len(), g.out[v].len());
            if di <= 1 || dout <= 1 {
                let mut not_a_loop = true;
                if di >= 1 && dout >= 1 && g.inn[v].contains(&v) {
                    not_a_loop = false;
                }
                if not_a_loop {
                    g.eliminate(v);
                    return true;
                }
            }
            false
        })
    }

    fn elim_clique(&mut self) -> bool {
        self.sweep(&mut |g, v| {
            if g.belongs_to_clique(v) {
                g.eliminate(v);
                true
            } else {
                false
            }
        })
    }

    fn suppress_loops(&mut self, fb: &mut BTreeSet<usize>) -> bool {
        self.sweep(&mut |g, v| {
            if g.has_edge(v, v) {
                fb.insert(v);
                g.suppress(v);
                true
            } else {
                false
            }
        })
    }

    /// `minimalSetOfFeedbackVertices` (vertex indices).
    pub fn minimal_feedback_set(&self) -> BTreeSet<usize> {
        let mut fb = BTreeSet::new();
        let mut g = self.clone();
        while g.num_vertices() > 0 {
            let mut sth = true;
            while sth && g.num_vertices() > 0 {
                sth = g.elim_low_degree();
                sth = g.elim_clique() || sth;
                sth = g.suppress_loops(&mut fb) || sth;
            }
            if !g.has_cycle() {
                return fb;
            }
            if g.num_vertices() > 0 {
                let mut max_degree = 0;
                let mut best = NULL;
                for &v in &g.verts {
                    let d = g.inn[v].len() + g.out[v].len();
                    if d > max_degree {
                        max_degree = d;
                        best = v;
                    }
                }
                fb.insert(best);
                g.suppress(best);
            }
        }
        fb
    }

    /// `reorderRecursiveVariables`: the non-feedback vertices in recursive order.
    pub fn reorder_recursive(&self, fb: &BTreeSet<usize>) -> Vec<usize> {
        let mut g = self.clone();
        for &f in fb.iter().rev() {
            let v = g.verts[f];
            g.suppress(v);
        }
        let mut order = Vec::new();
        let mut sth = true;
        while sth {
            sth = g.sweep(&mut |g, v| {
                if g.inn[v].is_empty() {
                    order.push(v);
                    g.suppress(v);
                    true
                } else {
                    false
                }
            });
        }
        if g.num_vertices() > 0 {
            eprintln!("Error in the computation of feedback vertex set");
        }
        order
    }

    /// `extractSubgraph`: vertices `sel` (positions = indices of the unmodified graph).
    pub fn extract_subgraph(&self, sel: &[usize]) -> DepGraph {
        let n = sel.len();
        let mut g = DepGraph::new(n);
        let mut rev = std::collections::BTreeMap::new();
        for (i, &s) in sel.iter().enumerate() {
            rev.insert(s, i);
            g.index1[i] = s;
        }
        for (i, &s) in sel.iter().enumerate() {
            let vi = self.verts[s];
            for &t in &self.out[vi] {
                if let Some(&j) = rev.get(&t) {
                    g.add_edge(i, j);
                }
            }
        }
        g
    }

    /// `sortedStronglyConnectedComponents`: number of SCCs and the (topologically sorted) SCC of
    /// each vertex (graph without removals, so positions = indices).
    pub fn sorted_sccs(&self) -> (usize, Vec<usize>) {
        let n = self.out.len();
        // Boost strong_components (Tarjan as a DFS visitor), iterative DFS in vertex order
        let inf = usize::MAX;
        let mut comp = vec![inf; n];
        let mut root = vec![0usize; n];
        let mut dtime = vec![0usize; n];
        let mut color = vec![0u8; n];
        let mut stack: Vec<usize> = Vec::new();
        let (mut time, mut c) = (0usize, 0usize);
        let finish = |v: usize,
                      comp: &mut Vec<usize>,
                      root: &mut Vec<usize>,
                      stack: &mut Vec<usize>,
                      c: &mut usize,
                      dtime: &Vec<usize>| {
            for &w in &self.out[v] {
                if comp[w] == inf {
                    let (rv, rw) = (root[v], root[w]);
                    root[v] = if dtime[rv] < dtime[rw] { rv } else { rw };
                }
            }
            if root[v] == v {
                loop {
                    let w = stack.pop().unwrap();
                    comp[w] = *c;
                    root[w] = v;
                    if w == v {
                        break;
                    }
                }
                *c += 1;
            }
        };
        for &s in &self.verts {
            if color[s] != 0 {
                continue;
            }
            let mut dfs: Vec<(usize, usize)> = Vec::new();
            color[s] = 1;
            root[s] = s;
            comp[s] = inf;
            dtime[s] = time;
            time += 1;
            stack.push(s);
            dfs.push((s, 0));
            while let Some(&mut (u, ref mut k)) = dfs.last_mut() {
                if *k < self.out[u].len() {
                    let v = self.out[u][*k];
                    *k += 1;
                    if color[v] == 0 {
                        color[v] = 1;
                        root[v] = v;
                        comp[v] = inf;
                        dtime[v] = time;
                        time += 1;
                        stack.push(v);
                        dfs.push((v, 0));
                    }
                } else {
                    color[u] = 2;
                    finish(u, &mut comp, &mut root, &mut stack, &mut c, &dtime);
                    dfs.pop();
                }
            }
        }
        let num = c;
        // condensation DAG (vecS, edges in insertion order) and its topological sort
        let mut dag = vec![Vec::new(); num];
        for i in 0..n {
            let vi = self.verts[i];
            for &t in &self.out[vi] {
                let (tb, sb) = (comp[t], comp[vi]);
                if sb != tb {
                    dag[sb].push(tb);
                }
            }
        }
        let mut order = Vec::with_capacity(num); // finish order = reverse topological order
        let mut col = vec![0u8; num];
        for s in 0..num {
            if col[s] != 0 {
                continue;
            }
            let mut st = vec![(s, 0usize)];
            col[s] = 1;
            while let Some(&mut (u, ref mut k)) = st.last_mut() {
                if *k < dag[u].len() {
                    let v = dag[u][*k];
                    *k += 1;
                    if col[v] == 0 {
                        col[v] = 1;
                        st.push((v, 0));
                    }
                } else {
                    col[u] = 2;
                    order.push(u);
                    st.pop();
                }
            }
        }
        let mut unordered2ordered = vec![0; num];
        for j in 0..num {
            unordered2ordered[order[num - j - 1]] = j;
        }
        (num, (0..n).map(|i| unordered2ordered[comp[i]]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn std_sort_sorts() {
        let mut seed = 12345u64;
        for n in [0usize, 1, 5, 16, 17, 40, 300, 1000] {
            let mut v: Vec<(u32, u32)> = (0..n)
                .map(|i| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (((seed >> 33) % 7) as u32, i as u32)
                })
                .collect();
            std_sort(&mut v, &|a, b| a.0 < b.0);
            assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
            let mut ids: Vec<u32> = v.iter().map(|x| x.1).collect();
            ids.sort();
            assert_eq!(ids, (0..n as u32).collect::<Vec<_>>());
        }
    }

    #[test]
    fn matching_is_perfect_on_bipartite() {
        // 3 endos (0..3), 3 eqs (3..6)
        let edges = [(3, 0), (3, 1), (4, 1), (5, 1), (5, 2)];
        let m = edmonds_matching(6, &edges);
        assert!(m.iter().all(Option::is_some));
    }
}
