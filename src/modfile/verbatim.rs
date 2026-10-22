// SPDX-License-Identifier: GPL-3.0-or-later
//! `verbatim; ... end;` blocks: Dynare copies them into the driver as MATLAB code, run in the
//! workspace where every parameter assigned so far is a variable (`name = M_.params(k);`).
//! dynare-rust interprets the subset of MATLAB that such blocks use to build inputs, e.g. the
//! initial guess structure of `heterogeneity_compute_steady_state(variable = initial_guess)`:
//!
//! - values: N-D real arrays (column-major), character strings, cell arrays, scalar structs;
//! - statements: `x = expr`, `s.a.b = expr`, `x(i, j) = expr`, `[a, ~, b] = f(...)`,
//!   expression statements (`disp`, `fprintf`... are ignored); `%` comments, `...` continuations;
//! - expressions: `+ - * / .* ./ .^ ^` with implicit expansion, unary `-`/`+`/`~`, comparisons,
//!   `&&`/`||`, transposes, ranges `a:b`/`a:s:b`, `[...]` concatenation, `{...}` cells,
//!   indexing `x(i)`, `x(i, :)`, field access;
//! - functions: `struct`, `zeros`, `ones`, `nan`, `size`, `numel`, `length`, `reshape`,
//!   `linspace`, `logspace`, `log`, `log10`, `exp`, `sqrt`, `abs`, `floor`, `ceil`, `round`,
//!   `max`/`min` (element-wise with two arguments, column reduction with one), `sum`,
//!   `cumsum`, `isempty`, `repmat`, `kron`, `eye`, `rouwenhorst`.
//!
//! The blocks run when they are parsed, with the parameter values assigned above them (Dynare
//! runs them at the same point of the driver; no computing command precedes them in practice).
//! Numeric scalars they define can be used by later parameter assignments, as in the driver.

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};

use crate::matfile::MatValue;

/// A MATLAB value.
#[derive(Clone, Debug, PartialEq)]
pub enum V {
    /// Real array, column-major; `dims.len() >= 2`.
    Num {
        dims: Vec<usize>,
        data: Vec<f64>,
    },
    Str(String),
    Cell {
        dims: Vec<usize>,
        items: Vec<V>,
    },
    /// Scalar struct, fields in creation order.
    Struct(Vec<(String, V)>),
}

fn norm_dims(mut d: Vec<usize>) -> Vec<usize> {
    while d.len() > 2 && d.last() == Some(&1) {
        d.pop();
    }
    while d.len() < 2 {
        d.push(1);
    }
    d
}

impl V {
    pub fn scalar(x: f64) -> V {
        V::Num { dims: vec![1, 1], data: vec![x] }
    }
    fn num(dims: Vec<usize>, data: Vec<f64>) -> V {
        V::Num { dims: norm_dims(dims), data }
    }
    fn row(data: Vec<f64>) -> V {
        V::Num { dims: vec![1, data.len()], data }
    }
    fn col(data: Vec<f64>) -> V {
        V::Num { dims: vec![data.len(), 1], data }
    }
    fn nums(&self) -> Result<(&[usize], &[f64])> {
        match self {
            V::Num { dims, data } => Ok((dims, data)),
            V::Str(s) => bail!("'{s}': a number was expected"),
            _ => bail!("a numeric array was expected"),
        }
    }
    fn as_f64(&self) -> Result<f64> {
        let (_, d) = self.nums()?;
        if d.len() != 1 {
            bail!("a scalar was expected (got {} elements)", d.len());
        }
        Ok(d[0])
    }
    fn data(&self) -> Result<Vec<f64>> {
        Ok(self.nums()?.1.to_vec())
    }
    fn dims(&self) -> Vec<usize> {
        match self {
            V::Num { dims, .. } | V::Cell { dims, .. } => dims.clone(),
            V::Str(s) => vec![if s.is_empty() { 0 } else { 1 }, s.chars().count()],
            V::Struct(_) => vec![1, 1],
        }
    }
    fn numel(&self) -> usize {
        self.dims().iter().product()
    }
    fn truthy(&self) -> Result<bool> {
        let (_, d) = self.nums()?;
        Ok(!d.is_empty() && d.iter().all(|&x| x != 0.0))
    }

    /// The MAT-file representation.
    pub fn to_mat(&self) -> MatValue {
        match self {
            V::Num { dims, data } => MatValue::Numeric { dims: dims.clone(), data: data.clone() },
            V::Str(s) => MatValue::Char(s.clone()),
            V::Cell { dims, items } => {
                MatValue::Cell { dims: dims.clone(), items: items.iter().map(V::to_mat).collect() }
            }
            V::Struct(f) => MatValue::Struct {
                dims: vec![1, 1],
                fields: f.iter().map(|(n, _)| n.clone()).collect(),
                elems: vec![f.iter().map(|(_, v)| v.to_mat()).collect()],
            },
        }
    }
}

// ---------------------------------------------------------------------------------------
// lexer

#[derive(Clone, Debug, PartialEq)]
enum T {
    Num(f64),
    Id(String),
    Str(String),
    Op(&'static str),
    /// Statement separator (`;`, `,` or a new line outside brackets); `true` when it is `;`.
    Sep,
}

const OPS: [&str; 30] = [
    "...", ".*", "./", ".^", ".'", "==", "~=", "<=", ">=", "&&", "||", "+", "-", "*", "/", "^", "(", ")", "[", "]",
    "{", "}", ",", ";", "=", "<", ">", "&", "|", "~",
];

fn lex(src: &str) -> Result<Vec<T>> {
    let b: Vec<char> = src.chars().collect();
    let mut out: Vec<T> = Vec::new();
    // bracket stack: '(' , '[' , '{'
    let mut stack: Vec<char> = Vec::new();
    let mut i = 0;
    let ends_value = |t: Option<&T>| match t {
        Some(T::Id(s)) => s != ":",
        _ => matches!(t, Some(T::Num(_) | T::Str(_) | T::Op(")" | "]" | "}" | "'" | ".'"))),
    };
    while i < b.len() {
        let c = b[i];
        let in_mat = matches!(stack.last(), Some('[' | '{'));
        if c == ':' {
            out.push(T::Id(":".into()));
            i += 1;
            continue;
        }
        if c == '%' || (c == '.' && b[i..].starts_with(&['.', '.', '.'])) {
            // comment or continuation: skip to the end of the line (a continuation also drops the newline)
            let cont = c == '.';
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
            if cont {
                i += 1;
            }
            continue;
        }
        if c == '\n' {
            if stack.is_empty() {
                out.push(T::Sep);
            } else if in_mat {
                out.push(T::Op(";"));
            }
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            // inside [ ] / { }, whitespace separates elements: [a -b] has two, [a - b] one
            let st = i;
            while i < b.len() && b[i].is_whitespace() && b[i] != '\n' {
                i += 1;
            }
            if in_mat && ends_value(out.last()) && i < b.len() {
                let n = b[i];
                let next_starts = n.is_alphanumeric()
                    || n == '_'
                    || n == '('
                    || n == '['
                    || n == '{'
                    || n == '\''
                    || n == '"'
                    || (n == '.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
                    || ((n == '-' || n == '+' || n == '~')
                        && b.get(i + 1).is_some_and(|d| !d.is_whitespace())
                        && b.get(i + 1) != Some(&'='));
                if next_starts && st < i {
                    out.push(T::Op(","));
                }
            }
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let st = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < b.len() && b[i] == '.' && !matches!(b.get(i + 1), Some('*' | '/' | '^' | '\'')) {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < b.len() && (b[i] == 'e' || b[i] == 'E') {
                let mut k = i + 1;
                if k < b.len() && (b[k] == '+' || b[k] == '-') {
                    k += 1;
                }
                if k < b.len() && b[k].is_ascii_digit() {
                    i = k;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let raw: String = b[st..i].iter().collect();
            out.push(T::Num(raw.parse().map_err(|_| anyhow!("invalid number '{raw}'"))?));
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            out.push(T::Id(b[st..i].iter().collect()));
            continue;
        }
        if c == '"' || (c == '\'' && !(ends_value(out.last()) && (i > 0 && !b[i - 1].is_whitespace() || !in_mat))) {
            let q = c;
            let mut s = String::new();
            i += 1;
            loop {
                if i >= b.len() || b[i] == '\n' {
                    bail!("unterminated string");
                }
                if b[i] == q {
                    if b.get(i + 1) == Some(&q) {
                        s.push(q);
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                s.push(b[i]);
                i += 1;
            }
            out.push(T::Str(s));
            continue;
        }
        if c == '\'' {
            out.push(T::Op("'"));
            i += 1;
            continue;
        }
        if c == '.' && !b.get(i + 1).is_some_and(|d| matches!(d, '*' | '/' | '^' | '\'')) {
            out.push(T::Op("."));
            i += 1;
            continue;
        }
        let rest: String = b[i..b.len().min(i + 3)].iter().collect();
        let Some(op) = OPS.iter().find(|o| rest.starts_with(**o)) else {
            bail!("unexpected character '{c}'");
        };
        match *op {
            "(" | "[" | "{" => stack.push(c),
            ")" | "]" | "}" => {
                stack.pop();
            }
            _ => {}
        }
        i += op.len();
        if stack.is_empty() && (*op == ";" || *op == ",") {
            out.push(T::Sep);
        } else {
            out.push(T::Op(op));
        }
    }
    out.push(T::Sep);
    Ok(out)
}

// ---------------------------------------------------------------------------------------
// parser (to an AST) and evaluation

#[derive(Clone, Debug)]
enum E {
    Num(f64),
    Str(String),
    Id(String),
    Colon,
    End,
    /// `f(args)` or `x(args)`
    Call(Box<E>, Vec<E>),
    /// `c{args}`
    Brace(Box<E>, Vec<E>),
    Field(Box<E>, String),
    Bin(&'static str, Box<E>, Box<E>),
    Un(&'static str, Box<E>),
    Range(Box<E>, Option<Box<E>>, Box<E>),
    Matrix(Vec<Vec<E>>),
    CellLit(Vec<Vec<E>>),
}

struct P<'a> {
    t: &'a [T],
    i: usize,
    /// > 0 inside an index list (where `end` and a bare `:` are allowed)
    in_index: usize,
}

impl<'a> P<'a> {
    fn peek(&self) -> Option<&'a T> {
        self.t.get(self.i)
    }
    fn is_op(&self, o: &str) -> bool {
        matches!(self.peek(), Some(T::Op(p)) if *p == o)
    }
    fn eat(&mut self, o: &str) -> bool {
        if self.is_op(o) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, o: &str) -> Result<()> {
        if !self.eat(o) {
            bail!("'{o}' expected, found {:?}", self.peek());
        }
        Ok(())
    }

    fn expr(&mut self) -> Result<E> {
        self.binary(0)
    }

    /// Binary operators by increasing precedence; ranges sit between the comparisons (level 4)
    /// and the additions (level 5).
    fn binary(&mut self, level: usize) -> Result<E> {
        const LEVELS: [&[&str]; 7] = [
            &["||"],
            &["&&"],
            &["|"],
            &["&"],
            &["==", "~=", "<", "<=", ">", ">="],
            &["+", "-"],
            &["*", "/", ".*", "./"],
        ];
        if level == LEVELS.len() {
            return self.unary();
        }
        let mut a = if level == 4 { self.range()? } else { self.binary(level + 1)? };
        loop {
            let Some(T::Op(o)) = self.peek() else { break };
            if !LEVELS[level].contains(o) {
                break;
            }
            self.i += 1;
            let b = if level == 4 { self.range()? } else { self.binary(level + 1)? };
            a = E::Bin(o, Box::new(a), Box::new(b));
        }
        Ok(a)
    }

    /// `a:b`, `a:s:b` (`:` is lexed as the identifier-like token `T::Id(":")`)
    fn range(&mut self) -> Result<E> {
        let a = self.binary(5)?;
        if !self.is_colon() {
            return Ok(a);
        }
        self.i += 1;
        let b = self.binary(5)?;
        if self.is_colon() {
            self.i += 1;
            let c = self.binary(5)?;
            return Ok(E::Range(Box::new(a), Some(Box::new(b)), Box::new(c)));
        }
        Ok(E::Range(Box::new(a), None, Box::new(b)))
    }

    fn is_colon(&self) -> bool {
        matches!(self.peek(), Some(T::Id(s)) if s == ":")
    }

    fn unary(&mut self) -> Result<E> {
        for o in ["-", "+", "~"] {
            if self.eat(o) {
                let a = self.unary()?;
                return Ok(E::Un(o, Box::new(a)));
            }
        }
        self.power()
    }

    fn power(&mut self) -> Result<E> {
        let mut a = self.postfix()?;
        loop {
            let o = if self.is_op("^") {
                "^"
            } else if self.is_op(".^") {
                ".^"
            } else {
                break;
            };
            self.i += 1;
            // the exponent may carry a unary sign: 2^-1
            let b = if self.eat("-") {
                E::Un("-", Box::new(self.postfix()?))
            } else {
                self.eat("+");
                self.postfix()?
            };
            a = E::Bin(o, Box::new(a), Box::new(b));
        }
        Ok(a)
    }

    fn postfix(&mut self) -> Result<E> {
        let mut a = self.primary()?;
        loop {
            if self.is_op("(") {
                self.i += 1;
                let args = self.args(")")?;
                a = E::Call(Box::new(a), args);
            } else if self.is_op("{") && matches!(a, E::Id(_) | E::Field(..) | E::Call(..)) {
                self.i += 1;
                let args = self.args("}")?;
                a = E::Brace(Box::new(a), args);
            } else if self.is_op(".") {
                self.i += 1;
                match self.peek() {
                    Some(T::Id(f)) => {
                        self.i += 1;
                        a = E::Field(Box::new(a), f.clone());
                    }
                    other => bail!("field name expected after '.', found {other:?}"),
                }
            } else if self.eat("'") || self.eat(".'") {
                a = E::Un("'", Box::new(a));
            } else {
                break;
            }
        }
        Ok(a)
    }

    fn args(&mut self, close: &str) -> Result<Vec<E>> {
        let mut v = Vec::new();
        self.in_index += 1;
        if !self.eat(close) {
            loop {
                if self.is_colon() && matches!(self.t.get(self.i + 1), Some(T::Op(o)) if *o == "," || *o == close) {
                    self.i += 1;
                    v.push(E::Colon);
                } else {
                    v.push(self.expr()?);
                }
                if self.eat(close) {
                    break;
                }
                self.expect(",")?;
            }
        }
        self.in_index -= 1;
        Ok(v)
    }

    fn rows(&mut self, close: &str) -> Result<Vec<Vec<E>>> {
        let saved = self.in_index;
        self.in_index = 0;
        let mut rows = vec![Vec::new()];
        loop {
            if self.eat(close) {
                break;
            }
            if self.eat(";") {
                rows.push(Vec::new());
                continue;
            }
            if self.eat(",") {
                continue;
            }
            let e = self.expr()?;
            rows.last_mut().unwrap().push(e);
        }
        self.in_index = saved;
        rows.retain(|r| !r.is_empty());
        Ok(rows)
    }

    fn primary(&mut self) -> Result<E> {
        let Some(t) = self.peek() else { bail!("unexpected end of statement") };
        self.i += 1;
        Ok(match t {
            T::Num(x) => E::Num(*x),
            T::Str(s) => E::Str(s.clone()),
            T::Id(s) if s == "end" && self.in_index > 0 => E::End,
            T::Id(s) => E::Id(s.clone()),
            T::Op("(") => {
                let saved = self.in_index;
                self.in_index = 0;
                let e = self.expr()?;
                self.in_index = saved;
                self.expect(")")?;
                // (x) is a value, not a call target
                E::Un("()", Box::new(e))
            }
            T::Op("[") => E::Matrix(self.rows("]")?),
            T::Op("{") => E::CellLit(self.rows("}")?),
            other => bail!("unexpected {other:?}"),
        })
    }
}

/// The interpreter: a workspace of variables.
#[derive(Clone, Debug, Default)]
pub struct Workspace {
    pub vars: HashMap<String, V>,
    /// Creation order of the variables.
    pub order: Vec<String>,
}

impl Workspace {
    pub fn set(&mut self, name: &str, v: V) {
        if !self.vars.contains_key(name) {
            self.order.push(name.to_string());
        }
        self.vars.insert(name.to_string(), v);
    }

    pub fn get(&self, name: &str) -> Option<&V> {
        self.vars.get(name)
    }

    /// Runs MATLAB source.
    pub fn run(&mut self, src: &str) -> Result<()> {
        let toks = lex(src)?;
        let mut st: Vec<&[T]> = Vec::new();
        let mut s = 0;
        for (k, t) in toks.iter().enumerate() {
            if *t == T::Sep {
                if k > s {
                    st.push(&toks[s..k]);
                }
                s = k + 1;
            }
        }
        for s in st {
            self.statement(s)?;
        }
        Ok(())
    }

    fn statement(&mut self, t: &[T]) -> Result<()> {
        // [a, ~, b] = f(...)
        if t.first() == Some(&T::Op("[")) {
            if let Some(close) = t.iter().position(|x| *x == T::Op("]")) {
                if t.get(close + 1) == Some(&T::Op("=")) {
                    let mut outs = Vec::new();
                    for x in &t[1..close] {
                        match x {
                            T::Id(n) => outs.push(Some(n.clone())),
                            T::Op("~") => outs.push(None),
                            T::Op(",") => {}
                            other => bail!("unsupported output list element {other:?}"),
                        }
                    }
                    let mut p = P { t: &t[close + 2..], i: 0, in_index: 0 };
                    let e = p.expr()?;
                    if p.i != p.t.len() {
                        bail!("unexpected {:?}", p.peek());
                    }
                    let vals = self.eval_multi(&e, outs.len())?;
                    for (o, v) in outs.into_iter().zip(vals) {
                        if let Some(n) = o {
                            self.set(&n, v);
                        }
                    }
                    return Ok(());
                }
            }
        }
        // lvalue = expr
        let eq = t.iter().position(|x| *x == T::Op("="));
        if let (Some(eq), Some(T::Id(_))) = (eq, t.first()) {
            let mut p = P { t: &t[..eq], i: 0, in_index: 0 };
            let lhs = p.postfix()?;
            if p.i != eq {
                bail!("invalid assignment target");
            }
            let mut p = P { t: &t[eq + 1..], i: 0, in_index: 0 };
            let rhs = p.expr()?;
            if p.i != p.t.len() {
                bail!("unexpected {:?}", p.peek());
            }
            let v = self.eval(&rhs)?;
            return self.assign(&lhs, v);
        }
        // statements acting outside the workspace (output, files, figures): no effect here
        if let Some(T::Id(n)) = t.first()
            && matches!(
                n.as_str(),
                "disp"
                    | "fprintf"
                    | "clear"
                    | "clc"
                    | "close"
                    | "format"
                    | "warning"
                    | "display"
                    | "figure"
                    | "hold"
                    | "plot"
                    | "save"
                    | "copyfile"
                    | "movefile"
                    | "mkdir"
                    | "delete"
                    | "rmdir"
                    | "drawnow"
                    | "print"
                    | "saveas"
                    | "diary"
                    | "pause"
                    | "dbstop"
                    | "tic"
                    | "toc"
            )
        {
            return Ok(());
        }
        let mut p = P { t, i: 0, in_index: 0 };
        let e = p.expr()?;
        if p.i != t.len() {
            bail!("unexpected {:?}", p.peek());
        }
        let v = self.eval(&e)?;
        self.set("ans", v);
        Ok(())
    }

    fn assign(&mut self, lhs: &E, v: V) -> Result<()> {
        // flatten the target into a variable and a path of field names / indices
        enum Step {
            Field(String),
            Index(Vec<E>),
        }
        let mut path = Vec::new();
        let mut e = lhs;
        let name = loop {
            match e {
                E::Id(n) => break n.clone(),
                E::Field(b, f) => {
                    path.push(Step::Field(f.clone()));
                    e = b;
                }
                E::Call(b, a) => {
                    path.push(Step::Index(a.clone()));
                    e = b;
                }
                _ => bail!("invalid assignment target"),
            }
        };
        path.reverse();
        fn set_in(ws: &Workspace, cur: Option<V>, path: &[Step], v: V) -> Result<V> {
            let Some((first, rest)) = path.split_first() else { return Ok(v) };
            match first {
                Step::Field(f) => {
                    let mut fields = match cur {
                        Some(V::Struct(f)) => f,
                        None => Vec::new(),
                        Some(V::Num { data, .. }) if data.is_empty() => Vec::new(),
                        Some(_) => bail!("field assignment to a non-struct"),
                    };
                    let old = fields.iter().position(|(n, _)| n == f);
                    let inner = set_in(ws, old.map(|k| fields[k].1.clone()), rest, v)?;
                    match old {
                        Some(k) => fields[k].1 = inner,
                        None => fields.push((f.clone(), inner)),
                    }
                    Ok(V::Struct(fields))
                }
                Step::Index(args) => {
                    if !rest.is_empty() {
                        bail!("indexed assignment followed by a field is not supported");
                    }
                    let cur = cur.unwrap_or(V::Num { dims: vec![0, 0], data: Vec::new() });
                    ws.assign_index(cur, args, v)
                }
            }
        }
        let cur = self.vars.get(&name).cloned();
        let nv = set_in(self, cur, &path, v)?;
        self.set(&name, nv);
        Ok(())
    }

    fn index_lists(&self, target: &V, args: &[E]) -> Result<Vec<Vec<usize>>> {
        let dims = target.dims();
        let n = args.len();
        let mut out = Vec::new();
        for (k, a) in args.iter().enumerate() {
            // extent of this subscript ("end"): the last one absorbs the trailing dimensions
            let ext = if n == 1 {
                dims.iter().product()
            } else if k == n - 1 {
                if k < dims.len() { dims[k..].iter().product() } else { 1 }
            } else {
                dims.get(k).copied().unwrap_or(1)
            };
            let idx: Vec<usize> = match a {
                E::Colon => (0..ext).collect(),
                _ => {
                    let v = self.eval_end(a, ext as f64)?;
                    let (_, d) = v.nums()?;
                    let logical = matches!(a, E::Bin("==" | "~=" | "<" | "<=" | ">" | ">=", ..));
                    if logical {
                        d.iter().enumerate().filter(|(_, x)| **x != 0.0).map(|(i, _)| i).collect()
                    } else {
                        d.iter()
                            .map(|&x| {
                                if x < 1.0 || x.fract() != 0.0 {
                                    bail!("Index must be a positive integer (got {x})")
                                }
                                Ok(x as usize - 1)
                            })
                            .collect::<Result<_>>()?
                    }
                }
            };
            out.push(idx);
        }
        Ok(out)
    }

    fn index(&self, v: &V, args: &[E]) -> Result<V> {
        let lists = self.index_lists(v, args)?;
        let dims = v.dims();
        let get = |lin: usize| -> Result<V> {
            match v {
                V::Num { data, .. } => data
                    .get(lin)
                    .map(|&x| V::scalar(x))
                    .ok_or_else(|| anyhow!("Index exceeds the number of array elements ({})", data.len())),
                V::Cell { items, .. } => items
                    .get(lin)
                    .map(|x| V::Cell { dims: vec![1, 1], items: vec![x.clone()] })
                    .ok_or_else(|| anyhow!("Index exceeds the number of cell elements")),
                V::Str(s) => s
                    .chars()
                    .nth(lin)
                    .map(|c| V::Str(c.to_string()))
                    .ok_or_else(|| anyhow!("Index exceeds the string length")),
                V::Struct(_) if lin == 0 => Ok(v.clone()),
                V::Struct(_) => bail!("Index exceeds the struct dimensions"),
            }
        };
        let (out_dims, lins): (Vec<usize>, Vec<usize>) = if lists.len() == 1 {
            let l = &lists[0];
            // x(idx) keeps the orientation of a vector source / of the index
            let od = if dims.len() == 2 && dims[0] == 1 {
                vec![1, l.len()]
            } else if dims.len() == 2 && dims[1] == 1 {
                vec![l.len(), 1]
            } else {
                match &args[0] {
                    E::Colon => vec![l.len(), 1],
                    a => {
                        let iv = self.eval_end(a, v.numel() as f64)?;
                        let d = iv.dims();
                        if d.len() == 2 && d[0] == 1 { vec![1, l.len()] } else { vec![l.len(), 1] }
                    }
                }
            };
            (od, l.clone())
        } else {
            // effective dims: the last subscript absorbs the remaining dimensions
            let n = lists.len();
            let mut ed: Vec<usize> = (0..n).map(|k| dims.get(k).copied().unwrap_or(1)).collect();
            if n - 1 < dims.len() {
                ed[n - 1] = dims[n - 1..].iter().product();
            }
            let mut lins = Vec::new();
            let counts: Vec<usize> = lists.iter().map(Vec::len).collect();
            let total: usize = counts.iter().product();
            for t in 0..total {
                let mut r = t;
                let mut lin = 0;
                let mut stride = 1;
                for k in 0..n {
                    let ik = lists[k][r % counts[k]];
                    r /= counts[k];
                    if ik >= ed[k] {
                        bail!("Index in position {} exceeds array bounds ({})", k + 1, ed[k]);
                    }
                    lin += ik * stride;
                    stride *= ed[k];
                }
                lins.push(lin);
            }
            (counts, lins)
        };
        match v {
            V::Num { data, .. } => {
                let d: Result<Vec<f64>> = lins
                    .iter()
                    .map(|&l| {
                        data.get(l)
                            .copied()
                            .ok_or_else(|| anyhow!("Index exceeds the number of array elements ({})", data.len()))
                    })
                    .collect();
                Ok(V::num(out_dims, d?))
            }
            V::Cell { items, .. } => {
                let d: Result<Vec<V>> = lins
                    .iter()
                    .map(|&l| items.get(l).cloned().ok_or_else(|| anyhow!("Index exceeds the number of cell elements")))
                    .collect();
                Ok(V::Cell { dims: norm_dims(out_dims), items: d? })
            }
            V::Str(_) => {
                let s: Result<String> =
                    lins.iter().map(|&l| get(l).map(|c| if let V::Str(c) = c { c } else { String::new() })).collect();
                Ok(V::Str(s?))
            }
            V::Struct(_) => get(lins.first().copied().unwrap_or(0)),
        }
    }

    fn assign_index(&self, cur: V, args: &[E], v: V) -> Result<V> {
        let (dims, mut data) = match cur {
            V::Num { dims, data } => (dims, data),
            _ => bail!("indexed assignment is only supported on numeric arrays"),
        };
        let src = v.data()?;
        let lists = self.index_lists(&V::Num { dims: dims.clone(), data: data.clone() }, args)?;
        let total: usize = lists.iter().map(Vec::len).product();
        if src.len() != 1 && src.len() != total {
            bail!("Unable to perform assignment: {total} elements on the left, {} on the right", src.len());
        }
        let val = |t: usize| if src.len() == 1 { src[0] } else { src[t] };
        if lists.len() == 1 {
            let mut dims = dims;
            let need = lists[0].iter().max().map_or(0, |m| m + 1);
            if need > data.len() {
                // grow a vector (a 0x0 or row grows as a row)
                if dims[1] == 1 && dims[0] > 1 {
                    dims = vec![need, 1];
                } else if dims.iter().product::<usize>() == data.len() && (dims[0] <= 1) {
                    dims = vec![1, need];
                } else {
                    bail!("Index exceeds the array bounds in an assignment");
                }
                data.resize(need, 0.0);
            }
            for (t, &l) in lists[0].iter().enumerate() {
                data[l] = val(t);
            }
            return Ok(V::num(dims, data));
        }
        // N subscripts: grow each dimension as needed
        let n = lists.len();
        let mut nd: Vec<usize> = (0..n.max(dims.len())).map(|k| dims.get(k).copied().unwrap_or(1)).collect();
        if n < dims.len() {
            bail!("assignment with fewer subscripts than dimensions is not supported");
        }
        for k in 0..n {
            let m = lists[k].iter().max().map_or(0, |m| m + 1);
            nd[k] = nd[k].max(m);
        }
        let mut out = vec![0.0; nd.iter().product()];
        // copy the old data
        let old_total: usize = dims.iter().product();
        for lin in 0..old_total {
            let mut r = lin;
            let mut nl = 0;
            let mut stride = 1;
            for k in 0..nd.len() {
                let dk = dims.get(k).copied().unwrap_or(1);
                nl += (r % dk) * stride;
                r /= dk;
                stride *= nd[k];
            }
            out[nl] = data[lin];
        }
        let counts: Vec<usize> = lists.iter().map(Vec::len).collect();
        for t in 0..total {
            let mut r = t;
            let mut lin = 0;
            let mut stride = 1;
            for k in 0..n {
                lin += lists[k][r % counts[k]] * stride;
                r /= counts[k];
                stride *= nd[k];
            }
            out[lin] = val(t);
        }
        Ok(V::num(nd, out))
    }

    fn eval_end(&self, e: &E, end: f64) -> Result<V> {
        // substitute `end` in an index expression
        fn has_end(e: &E) -> bool {
            match e {
                E::End => true,
                E::Bin(_, a, b) => has_end(a) || has_end(b),
                E::Un(_, a) => has_end(a),
                E::Range(a, s, b) => has_end(a) || s.as_ref().is_some_and(|s| has_end(s)) || has_end(b),
                _ => false,
            }
        }
        fn subst(e: &E, end: f64) -> E {
            match e {
                E::End => E::Num(end),
                E::Bin(o, a, b) => E::Bin(o, Box::new(subst(a, end)), Box::new(subst(b, end))),
                E::Un(o, a) => E::Un(o, Box::new(subst(a, end))),
                E::Range(a, s, b) => E::Range(
                    Box::new(subst(a, end)),
                    s.as_ref().map(|s| Box::new(subst(s, end))),
                    Box::new(subst(b, end)),
                ),
                other => other.clone(),
            }
        }
        if has_end(e) { self.eval(&subst(e, end)) } else { self.eval(e) }
    }

    fn eval(&self, e: &E) -> Result<V> {
        let mut v = self.eval_multi(e, 1)?;
        v.truncate(1);
        v.pop().ok_or_else(|| anyhow!("the expression has no value"))
    }

    /// Evaluates `e` with `nargout` outputs (only function calls can have several).
    fn eval_multi(&self, e: &E, nargout: usize) -> Result<Vec<V>> {
        Ok(match e {
            E::Num(x) => vec![V::scalar(*x)],
            E::Str(s) => vec![V::Str(s.clone())],
            E::End => bail!("'end' outside an index"),
            E::Colon => bail!("':' outside an index"),
            E::Id(n) => match self.vars.get(n) {
                Some(v) => vec![v.clone()],
                None => call(n, &[], nargout)?,
            },
            E::Call(f, args) => {
                if let E::Id(n) = f.as_ref()
                    && !self.vars.contains_key(n)
                {
                    let vals: Result<Vec<V>> = args.iter().map(|a| self.eval(a)).collect();
                    return call(n, &vals?, nargout);
                }
                let base = self.eval(f)?;
                vec![self.index(&base, args)?]
            }
            E::Brace(c, args) => {
                let base = self.eval(c)?;
                match self.index(&base, args)? {
                    V::Cell { items, .. } if items.len() == 1 => items,
                    V::Cell { items, .. } => items,
                    _ => bail!("brace indexing of a non-cell"),
                }
            }
            E::Field(b, f) => match self.eval(b)? {
                V::Struct(fs) => vec![
                    fs.into_iter()
                        .find(|(n, _)| n == f)
                        .map(|(_, v)| v)
                        .ok_or_else(|| anyhow!("Unrecognized field name \"{f}\""))?,
                ],
                _ => bail!("Dot indexing is not supported for variables of this type (field {f})"),
            },
            E::Un(o, a) => {
                let a = self.eval(a)?;
                vec![match *o {
                    "()" => a,
                    "-" => map(&a, |x| -x)?,
                    "+" => a,
                    "~" => map(&a, |x| f64::from(x == 0.0))?,
                    "'" => transpose(&a)?,
                    _ => unreachable!(),
                }]
            }
            E::Bin(o, a, b) => {
                if *o == "&&" || *o == "||" {
                    let x = self.eval(a)?.truthy()?;
                    let r = if *o == "&&" { x && self.eval(b)?.truthy()? } else { x || self.eval(b)?.truthy()? };
                    return Ok(vec![V::scalar(f64::from(r))]);
                }
                let (x, y) = (self.eval(a)?, self.eval(b)?);
                vec![binop(o, &x, &y)?]
            }
            E::Range(a, s, b) => {
                let a = self.eval(a)?.as_f64()?;
                let s = match s {
                    Some(s) => self.eval(s)?.as_f64()?,
                    None => 1.0,
                };
                let b = self.eval(b)?.as_f64()?;
                if s == 0.0 || (b - a) / s < 0.0 {
                    return Ok(vec![V::num(vec![1, 0], Vec::new())]);
                }
                let n = ((b - a) / s + 1e-10).floor() as usize + 1;
                vec![V::row((0..n).map(|k| a + k as f64 * s).collect())]
            }
            E::Matrix(rows) => vec![self.concat(rows)?],
            E::CellLit(rows) => {
                let mut items = Vec::new();
                let nr = rows.len();
                let nc = rows.first().map_or(0, Vec::len);
                for r in rows {
                    if r.len() != nc {
                        bail!("Dimensions of cell array rows are not consistent");
                    }
                }
                for j in 0..nc {
                    for r in rows {
                        items.push(self.eval(&r[j])?);
                    }
                }
                vec![V::Cell { dims: vec![nr, nc], items }]
            }
        })
    }

    fn concat(&self, rows: &[Vec<E>]) -> Result<V> {
        let mut blocks: Vec<V> = Vec::new();
        for r in rows {
            let vals: Vec<V> = r.iter().map(|e| self.eval(e)).collect::<Result<_>>()?;
            if vals.iter().all(|v| matches!(v, V::Str(_))) && !vals.is_empty() {
                blocks.push(V::Str(
                    vals.iter().map(|v| if let V::Str(s) = v { s.clone() } else { String::new() }).collect(),
                ));
                continue;
            }
            // horizontal concatenation of 2-D numeric blocks
            let mut nr = None;
            let mut data = Vec::new();
            let mut nc = 0;
            for v in &vals {
                let (d, x) = v.nums()?;
                if x.is_empty() {
                    continue;
                }
                if d.len() > 2 {
                    bail!("concatenation of N-D arrays is not supported");
                }
                if nr.is_some_and(|n| n != d[0]) {
                    bail!("Dimensions of arrays being concatenated are not consistent");
                }
                nr = Some(d[0]);
                nc += d[1];
                data.extend_from_slice(x);
            }
            blocks.push(V::num(vec![nr.unwrap_or(0), nc], data));
        }
        if blocks.len() == 1 {
            return Ok(blocks.pop().unwrap());
        }
        if !blocks.is_empty() && blocks.iter().all(|b| matches!(b, V::Str(_))) {
            bail!("vertical concatenation of strings is not supported");
        }
        // vertical concatenation
        let blocks: Vec<&V> = blocks.iter().filter(|b| b.numel() > 0).collect();
        let Some(first) = blocks.first() else { return Ok(V::num(vec![0, 0], Vec::new())) };
        let nc = first.dims()[1];
        let nr: usize = blocks.iter().map(|b| b.dims()[0]).sum();
        let mut out = vec![0.0; nr * nc];
        let mut r0 = 0;
        for b in &blocks {
            let (d, x) = b.nums()?;
            if d[1] != nc {
                bail!("Dimensions of arrays being concatenated are not consistent");
            }
            for j in 0..nc {
                for i in 0..d[0] {
                    out[j * nr + r0 + i] = x[j * d[0] + i];
                }
            }
            r0 += d[0];
        }
        Ok(V::num(vec![nr, nc], out))
    }
}

// ---------------------------------------------------------------------------------------
// numeric helpers

fn map(a: &V, f: impl Fn(f64) -> f64) -> Result<V> {
    let (d, x) = a.nums()?;
    Ok(V::num(d.to_vec(), x.iter().map(|&v| f(v)).collect()))
}

fn transpose(a: &V) -> Result<V> {
    let (d, x) = a.nums()?;
    if d.len() > 2 {
        bail!("Transpose on ND array is not defined");
    }
    let (r, c) = (d[0], d[1]);
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[i * c + j] = x[j * r + i];
        }
    }
    Ok(V::num(vec![c, r], out))
}

/// Element-wise operation with implicit expansion.
fn broadcast(a: &V, b: &V, f: impl Fn(f64, f64) -> f64) -> Result<V> {
    let (da, xa) = a.nums()?;
    let (db, xb) = b.nums()?;
    let n = da.len().max(db.len());
    let dim = |d: &[usize], k: usize| d.get(k).copied().unwrap_or(1);
    let mut dims = Vec::with_capacity(n);
    for k in 0..n {
        let (p, q) = (dim(da, k), dim(db, k));
        dims.push(if p == q {
            p
        } else if p == 1 {
            q
        } else if q == 1 {
            p
        } else {
            bail!("Arrays have incompatible sizes for this operation ({da:?} and {db:?})")
        });
    }
    let total: usize = dims.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut sub = vec![0usize; n];
    for _ in 0..total {
        let (mut la, mut lb, mut sa, mut sb) = (0, 0, 1, 1);
        for k in 0..n {
            let (p, q) = (dim(da, k), dim(db, k));
            la += if p == 1 { 0 } else { sub[k] } * sa;
            lb += if q == 1 { 0 } else { sub[k] } * sb;
            sa *= p;
            sb *= q;
        }
        out.push(f(xa[la], xb[lb]));
        for k in 0..n {
            sub[k] += 1;
            if sub[k] < dims[k] {
                break;
            }
            sub[k] = 0;
        }
    }
    Ok(V::num(dims, out))
}

fn binop(o: &str, a: &V, b: &V) -> Result<V> {
    let scalar = |v: &V| v.numel() == 1;
    match o {
        "+" => broadcast(a, b, |x, y| x + y),
        "-" => broadcast(a, b, |x, y| x - y),
        ".*" => broadcast(a, b, |x, y| x * y),
        "./" => broadcast(a, b, |x, y| x / y),
        ".^" => broadcast(a, b, f64::powf),
        "==" => broadcast(a, b, |x, y| f64::from(x == y)),
        "~=" => broadcast(a, b, |x, y| f64::from(x != y)),
        "<" => broadcast(a, b, |x, y| f64::from(x < y)),
        "<=" => broadcast(a, b, |x, y| f64::from(x <= y)),
        ">" => broadcast(a, b, |x, y| f64::from(x > y)),
        ">=" => broadcast(a, b, |x, y| f64::from(x >= y)),
        "&" => broadcast(a, b, |x, y| f64::from(x != 0.0 && y != 0.0)),
        "|" => broadcast(a, b, |x, y| f64::from(x != 0.0 || y != 0.0)),
        "*" if scalar(a) || scalar(b) => broadcast(a, b, |x, y| x * y),
        "*" => {
            let (da, xa) = a.nums()?;
            let (db, xb) = b.nums()?;
            if da.len() > 2 || db.len() > 2 || da[1] != db[0] {
                bail!("Incorrect dimensions for matrix multiplication ({da:?} * {db:?})");
            }
            let (m, k, n) = (da[0], da[1], db[1]);
            let mut out = vec![0.0; m * n];
            for j in 0..n {
                for l in 0..k {
                    let y = xb[j * k + l];
                    for i in 0..m {
                        out[j * m + i] += xa[l * m + i] * y;
                    }
                }
            }
            Ok(V::num(vec![m, n], out))
        }
        "/" if scalar(b) => broadcast(a, b, |x, y| x / y),
        "/" => bail!("matrix right division is not supported"),
        "^" if scalar(a) && scalar(b) => broadcast(a, b, f64::powf),
        "^" => bail!("matrix power is not supported"),
        _ => bail!("operator {o} is not supported"),
    }
}

/// Dimensions from `f(n)`, `f(m, n, ...)` or `f([m n ...])`.
fn dims_args(args: &[V]) -> Result<Vec<usize>> {
    let d: Vec<f64> =
        if args.len() == 1 { args[0].data()? } else { args.iter().map(V::as_f64).collect::<Result<_>>()? };
    let mut d: Vec<usize> = d.iter().map(|&x| x.max(0.0) as usize).collect();
    if d.is_empty() {
        d = vec![1, 1];
    } else if d.len() == 1 {
        d.push(d[0]);
    }
    Ok(norm_dims(d))
}

fn reduce_cols(a: &V, f: impl Fn(&[f64]) -> f64) -> Result<V> {
    let (d, x) = a.nums()?;
    if d.len() == 2 && (d[0] == 1 || d[1] == 1) {
        return Ok(V::scalar(f(x)));
    }
    let r = d[0];
    let rest: usize = d[1..].iter().product();
    let mut dims = d.to_vec();
    dims[0] = 1;
    Ok(V::num(dims, (0..rest).map(|j| f(&x[j * r..(j + 1) * r])).collect()))
}

fn call(name: &str, a: &[V], nargout: usize) -> Result<Vec<V>> {
    let n = a.len();
    let arg = |k: usize| a.get(k).ok_or_else(|| anyhow!("{name}: not enough input arguments"));
    let f1 = |f: fn(f64) -> f64| -> Result<Vec<V>> { Ok(vec![map(arg(0)?, f)?]) };
    Ok(match name {
        "pi" => vec![V::scalar(std::f64::consts::PI)],
        "Inf" | "inf" => vec![V::scalar(f64::INFINITY)],
        "NaN" | "nan" if n == 0 => vec![V::scalar(f64::NAN)],
        "eps" => vec![V::scalar(f64::EPSILON)],
        "true" => vec![V::scalar(1.0)],
        "false" => vec![V::scalar(0.0)],
        "struct" => {
            if !n.is_multiple_of(2) {
                bail!("struct: field/value pairs expected");
            }
            let mut f = Vec::new();
            for k in (0..n).step_by(2) {
                let V::Str(s) = &a[k] else { bail!("struct: field names must be strings") };
                f.push((s.clone(), a[k + 1].clone()));
            }
            vec![V::Struct(f)]
        }
        "zeros" | "ones" | "nan" | "NaN" => {
            let d = dims_args(a)?;
            let v = match name {
                "zeros" => 0.0,
                "ones" => 1.0,
                _ => f64::NAN,
            };
            vec![V::num(d.clone(), vec![v; d.iter().product()])]
        }
        "eye" => {
            let d = dims_args(a)?;
            let mut x = vec![0.0; d[0] * d[1]];
            for i in 0..d[0].min(d[1]) {
                x[i * d[0] + i] = 1.0;
            }
            vec![V::num(d, x)]
        }
        "size" => {
            let d = arg(0)?.dims();
            if n == 2 {
                let k = a[1].as_f64()? as usize;
                vec![V::scalar(d.get(k - 1).copied().unwrap_or(1) as f64)]
            } else if nargout > 1 {
                let mut out: Vec<V> = (0..nargout).map(|k| V::scalar(d.get(k).copied().unwrap_or(1) as f64)).collect();
                if nargout < d.len() {
                    out[nargout - 1] = V::scalar(d[nargout - 1..].iter().product::<usize>() as f64);
                }
                out
            } else {
                vec![V::row(d.iter().map(|&x| x as f64).collect())]
            }
        }
        "numel" => vec![V::scalar(arg(0)?.numel() as f64)],
        "length" => {
            let d = arg(0)?.dims();
            vec![V::scalar(if d.contains(&0) { 0.0 } else { *d.iter().max().unwrap() as f64 })]
        }
        "isempty" => vec![V::scalar(f64::from(arg(0)?.numel() == 0))],
        "reshape" => {
            let x = arg(0)?.data()?;
            let mut dims: Vec<Option<usize>> = if n == 2 {
                a[1].data()?.iter().map(|&v| Some(v as usize)).collect()
            } else {
                a[1..]
                    .iter()
                    .map(|v| if v.numel() == 0 { Ok(None) } else { v.as_f64().map(|x| Some(x as usize)) })
                    .collect::<Result<_>>()?
            };
            let known: usize = dims.iter().flatten().product();
            if let Some(k) = dims.iter().position(Option::is_none) {
                if known == 0 || x.len() % known != 0 {
                    bail!("reshape: size is not divisible");
                }
                dims[k] = Some(x.len() / known);
            }
            let d: Vec<usize> = dims.into_iter().map(|v| v.unwrap()).collect();
            if d.iter().product::<usize>() != x.len() {
                bail!("reshape: number of elements must not change");
            }
            vec![V::num(d, x)]
        }
        "linspace" | "logspace" => {
            let (lo, hi) = (arg(0)?.as_f64()?, arg(1)?.as_f64()?);
            let m = if n > 2 {
                a[2].as_f64()? as usize
            } else if name == "linspace" {
                100
            } else {
                50
            };
            let v: Vec<f64> = (0..m)
                .map(|k| {
                    if m == 1 {
                        hi
                    } else if k + 1 == m {
                        hi
                    } else {
                        lo + k as f64 * (hi - lo) / (m - 1) as f64
                    }
                })
                .collect();
            // MATLAB's linspace: lo + (0:n-1)*(hi-lo)/(n-1), the last point exactly hi
            let v = if name == "logspace" { v.iter().map(|&e| 10f64.powf(e)).collect() } else { v };
            vec![V::row(v)]
        }
        "log" => f1(f64::ln)?,
        "log10" => f1(f64::log10)?,
        "log2" => f1(f64::log2)?,
        "exp" => f1(f64::exp)?,
        "sqrt" => f1(f64::sqrt)?,
        "abs" => f1(f64::abs)?,
        "floor" => f1(f64::floor)?,
        "ceil" => f1(f64::ceil)?,
        "round" => f1(f64::round)?,
        "max" | "min" if n >= 2 => {
            let mx = name == "max";
            vec![broadcast(&a[0], &a[1], |x, y| {
                if x.is_nan() {
                    y
                } else if y.is_nan() {
                    x
                } else if mx {
                    x.max(y)
                } else {
                    x.min(y)
                }
            })?]
        }
        "max" => vec![reduce_cols(arg(0)?, |c| c.iter().copied().fold(f64::NEG_INFINITY, f64::max))?],
        "min" => vec![reduce_cols(arg(0)?, |c| c.iter().copied().fold(f64::INFINITY, f64::min))?],
        "sum" if n == 1 => vec![reduce_cols(arg(0)?, |c| c.iter().sum())?],
        "mean" if n == 1 => vec![reduce_cols(arg(0)?, |c| c.iter().sum::<f64>() / c.len() as f64)?],
        "cumsum" => {
            let (d, x) = arg(0)?.nums()?;
            let r = if d.len() == 2 && d[0] == 1 { d[1] } else { d[0] };
            let mut y = x.to_vec();
            for c in y.chunks_mut(r.max(1)) {
                for k in 1..c.len() {
                    c[k] += c[k - 1];
                }
            }
            vec![V::num(d.to_vec(), y)]
        }
        "repmat" => {
            let (d, x) = arg(0)?.nums()?;
            if d.len() > 2 {
                bail!("repmat: N-D arrays are not supported");
            }
            let rep = dims_args(&a[1..])?;
            let (r, c) = (d[0], d[1]);
            let (nr, nc) = (r * rep[0], c * rep[1]);
            let mut out = vec![0.0; nr * nc];
            for j in 0..nc {
                for i in 0..nr {
                    out[j * nr + i] = x[(j % c) * r + i % r];
                }
            }
            vec![V::num(vec![nr, nc], out)]
        }
        "kron" => {
            let (da, xa) = arg(0)?.nums()?;
            let (db, xb) = arg(1)?.nums()?;
            if da.len() > 2 || db.len() > 2 {
                bail!("kron: N-D arrays are not supported");
            }
            let (nr, nc) = (da[0] * db[0], da[1] * db[1]);
            let mut out = vec![0.0; nr * nc];
            for ja in 0..da[1] {
                for ia in 0..da[0] {
                    for jb in 0..db[1] {
                        for ib in 0..db[0] {
                            out[(ja * db[1] + jb) * nr + ia * db[0] + ib] = xa[ja * da[0] + ia] * xb[jb * db[0] + ib];
                        }
                    }
                }
            }
            vec![V::num(vec![nr, nc], out)]
        }
        "rouwenhorst" => {
            let rho = arg(0)?.as_f64()?;
            let sigma = arg(1)?.as_f64()?;
            let m = arg(2)?.as_f64()? as usize;
            let tol = if n > 3 { a[3].as_f64()? } else { 1e-12 };
            let maxiter = if n > 4 { a[4].as_f64()? as usize } else { 100_000 };
            let (y, p, pm) = crate::heterogeneity::rouwenhorst(rho, sigma, m, tol, maxiter)?;
            let pmv: Vec<f64> = pm.as_slice().to_vec();
            vec![V::col(y), V::col(p), V::num(vec![pm.nrows(), pm.ncols()], pmv)]
        }
        _ => bail!("Unrecognized function or variable '{name}'"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> Workspace {
        let mut w = Workspace::default();
        w.run(src).unwrap();
        w
    }

    fn num(w: &Workspace, n: &str) -> (Vec<usize>, Vec<f64>) {
        match w.get(n).unwrap() {
            V::Num { dims, data } => (dims.clone(), data.clone()),
            other => panic!("{n}: {other:?}"),
        }
    }

    #[test]
    fn arithmetic_and_precedence() {
        let w = run(
            "a = -2^2; b = 2^-1; c = 1 + 2*3 - 4/2; d = [1 2; 3 4]'; e = [1 -2]; f = [1 - 2];\n% comment\ng = 3 ... continued\n + 1;",
        );
        assert_eq!(num(&w, "a").1, vec![-4.0]);
        assert_eq!(num(&w, "b").1, vec![0.5]);
        assert_eq!(num(&w, "c").1, vec![5.0]);
        assert_eq!(num(&w, "d"), (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(num(&w, "e").1, vec![1.0, -2.0]);
        assert_eq!(num(&w, "f").1, vec![-1.0]);
        assert_eq!(num(&w, "g").1, vec![4.0]);
    }

    #[test]
    fn broadcasting_reshape_and_structs() {
        let w = run("x = [1 2 3]; y = [10; 20]; z = x + y; s = struct; s.a.b = z; s.c = {'e', 'a'};\n\
             r = reshape(1:6, 1, [], 2); q = max(r - 3, 0); n = ones(size(z)); k = z(2, :); m = z(end);\n\
             [g, ~, P] = rouwenhorst(0.9, 0.5, 3, 1e-12, 1e5); t = x'; v = 0:0.5:1; u = 'it''s';");
        assert_eq!(num(&w, "z"), (vec![2, 3], vec![11.0, 21.0, 12.0, 22.0, 13.0, 23.0]));
        assert_eq!(num(&w, "r").0, vec![1, 3, 2]);
        assert_eq!(num(&w, "q").1, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        assert_eq!(num(&w, "n"), (vec![2, 3], vec![1.0; 6]));
        assert_eq!(num(&w, "k"), (vec![1, 3], vec![21.0, 22.0, 23.0]));
        assert_eq!(num(&w, "m").1, vec![23.0]);
        assert_eq!(num(&w, "g").0, vec![3, 1]);
        assert_eq!(num(&w, "P").0, vec![3, 3]);
        assert_eq!(num(&w, "t").0, vec![3, 1]);
        assert_eq!(num(&w, "v").1, vec![0.0, 0.5, 1.0]);
        assert_eq!(w.get("u"), Some(&V::Str("it's".into())));
        let V::Struct(f) = w.get("s").unwrap() else { panic!() };
        assert_eq!(f[0].0, "a");
        assert_eq!(f[1].1, V::Cell { dims: vec![1, 2], items: vec![V::Str("e".into()), V::Str("a".into())] });
        assert!(matches!(w.get("s").unwrap().to_mat(), MatValue::Struct { .. }));
    }

    #[test]
    fn logspace_matches_matlab() {
        let w = run("g = logspace(log10(0.25), log10(200.25), 30)-0.25;");
        let (d, x) = num(&w, "g");
        assert_eq!(d, vec![1, 30]);
        assert_eq!(x[0], 0.0);
        assert!((x[29] - 200.0).abs() < 1e-12);
    }

    #[test]
    fn errors() {
        let mut w = Workspace::default();
        assert!(w.run("a = undefined_thing + 1;").unwrap_err().to_string().contains("undefined_thing"));
        assert!(w.run("a = [1 2] + [1 2 3];").is_err());
    }
}
