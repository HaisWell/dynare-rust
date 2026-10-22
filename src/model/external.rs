// SPDX-License-Identifier: GPL-3.0-or-later
//! `external_function`: functions of the model that are not written in the .mod file.
//!
//! Dynare calls a MATLAB function (`ExternalFunctionNode` in the preprocessor). Here a model
//! calls an entry of a process-wide registry, filled either
//!
//! * from Rust with [`register_external_function`] /
//!   [`register_external_function_with_derivatives`] (closures, looked up by name when the
//!   `.mod` declares `external_function(name = NAME, ...)`), or
//! * by loading `NAME.m` next to the `.mod` file: a small interpreter for simple MATLAB
//!   function files (scalar arithmetic, elementary functions, `if`/`elseif`/`else`, vector and
//!   matrix literals and indexed assignments for derivative outputs; see [`MFunction`]).
//!
//! Derivatives follow the preprocessor: with `first_deriv_provided` (no name: second output of
//! the function; `= NAME`: a separate function returning the gradient) and
//! `second_deriv_provided` (third output / separate function returning the Hessian) the
//! provided values are used; otherwise Dynare's `jacob_element.m` / `hess_element.m` central
//! differences with `h = 1e-6`. Third and higher derivatives are not available (the
//! preprocessor refuses them too).

use anyhow::{Context, Result, anyhow, bail};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Value of an external function at its arguments.
pub type ScalarFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;
/// Gradient (length `nargs`) or Hessian (`nargs x nargs`, column-major) at the arguments.
pub type VecFn = Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;

/// One registered external function.
pub struct ExternalFunction {
    pub name: &'static str,
    pub nargs: usize,
    pub value: ScalarFn,
    pub gradient: Option<VecFn>,
    pub hessian: Option<VecFn>,
}

impl std::fmt::Debug for ExternalFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExternalFunction({}, nargs={})", self.name, self.nargs)
    }
}

/// Step of `jacob_element.m` / `hess_element.m`.
const H: f64 = 1e-6;

impl ExternalFunction {
    /// The function (`d1 = d2 = 0`), its derivative w.r.t. argument `d1 - 1` (`d2 = 0`) or its
    /// second derivative w.r.t. arguments `d1 - 1`, `d2 - 1`.
    pub fn eval(&self, d1: u16, d2: u16, args: &[f64]) -> f64 {
        let n = args.len();
        match (d1 as usize, d2 as usize) {
            (0, _) => (self.value)(args),
            (i, 0) => match &self.gradient {
                Some(g) => g(args).get(i - 1).copied().unwrap_or(f64::NAN),
                None => {
                    // jacob_element.m
                    let (mut p, mut m) = (args.to_vec(), args.to_vec());
                    p[i - 1] += H;
                    m[i - 1] -= H;
                    ((self.value)(&p) - (self.value)(&m)) / (2.0 * H)
                }
            },
            (i, j) => match &self.hessian {
                Some(h) => h(args).get((j - 1) * n + i - 1).copied().unwrap_or(f64::NAN),
                None => hess_element(&*self.value, i - 1, j - 1, args),
            },
        }
    }
}

/// `hess_element.m` (the same formulas, in the same order).
fn hess_element(f: &dyn Fn(&[f64]) -> f64, e1: usize, e2: usize, args: &[f64]) -> f64 {
    let (mut p10, mut p01, mut m10, mut m01, mut p11, mut m11) =
        (args.to_vec(), args.to_vec(), args.to_vec(), args.to_vec(), args.to_vec(), args.to_vec());
    p10[e1] += H;
    m10[e1] -= H;
    p11[e1] += H;
    m11[e1] -= H;
    p01[e2] += H;
    m01[e2] -= H;
    p11[e2] += H;
    m11[e2] -= H;
    if e1 == e2 {
        (16.0 * f(&p10) + 16.0 * f(&m10) - 30.0 * f(args) - f(&p11) - f(&m11)) / (12.0 * H * H)
    } else {
        (f(&p10) + f(&m10) + f(&p01) + f(&m01) - 2.0 * f(args) - f(&p11) - f(&m11)) / (-2.0 * H * H)
    }
}

// ---- runtime errors ---------------------------------------------------------------------------

/// A runtime error in an external function (e.g. a MATLAB function file that leaves an output
/// unassigned or indexes out of bounds). The model evaluation cannot return it, so it unwinds
/// (without the panic hook) to [`catch_external_errors`].
#[derive(Debug)]
pub struct ExternalError(pub String);

pub(crate) fn fail(msg: String) -> ! {
    std::panic::resume_unwind(Box::new(ExternalError(msg)))
}

/// Runs `f`, turning a runtime error of an external function ([`ExternalError`]) into an
/// error; other panics are propagated.
pub fn catch_external_errors<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => match p.downcast::<ExternalError>() {
            Ok(e) => Err(anyhow!("external function: {}", e.0)),
            Err(p) => std::panic::resume_unwind(p),
        },
    }
}

// ---- registry ---------------------------------------------------------------------------------

/// Functions referenced by models (append-only: a model's ids stay valid).
static FUNCS: RwLock<Vec<Arc<ExternalFunction>>> = RwLock::new(Vec::new());

struct UserFn {
    value: ScalarFn,
    gradient: Option<VecFn>,
    hessian: Option<VecFn>,
}

/// Functions registered from Rust, by name.
static USER: RwLock<Option<HashMap<String, Arc<UserFn>>>> = RwLock::new(None);

/// Registers a Rust closure as the external function `name`: a later
/// `external_function(name = NAME, ...)` uses it instead of looking for `NAME.m`.
/// Without derivatives, the model uses Dynare's numerical derivatives.
pub fn register_external_function(name: &str, value: impl Fn(&[f64]) -> f64 + Send + Sync + 'static) {
    register_impl(name, Arc::new(value), None, None);
}

/// [`register_external_function`] with an optional gradient (vector of length `nargs`) and
/// Hessian (`nargs x nargs`, column-major). They are used when the `.mod` declares
/// `first_deriv_provided` / `second_deriv_provided` (without a name).
pub fn register_external_function_with_derivatives(
    name: &str,
    value: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    gradient: Option<VecFn>,
    hessian: Option<VecFn>,
) {
    register_impl(name, Arc::new(value), gradient, hessian);
}

/// Registers a vector-valued function under `name` (used as `first_deriv_provided = NAME` or
/// `second_deriv_provided = NAME`).
pub fn register_external_vector_function(name: &str, f: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static) {
    let f: VecFn = Arc::new(f);
    let g = f.clone();
    let value: ScalarFn = Arc::new(move |a: &[f64]| g(a).first().copied().unwrap_or(f64::NAN));
    register_impl(name, value, Some(f), None);
}

fn register_impl(name: &str, value: ScalarFn, gradient: Option<VecFn>, hessian: Option<VecFn>) {
    let mut u = USER.write().unwrap();
    u.get_or_insert_with(HashMap::new).insert(name.to_string(), Arc::new(UserFn { value, gradient, hessian }));
}

fn user(name: &str) -> Option<Arc<UserFn>> {
    USER.read().unwrap().as_ref().and_then(|m| m.get(name).cloned())
}

/// The function with id `fid` (as stored in `Node::External`).
pub fn get(fid: u32) -> Arc<ExternalFunction> {
    FUNCS.read().unwrap()[fid as usize].clone()
}

/// Name of the function with id `fid`.
pub fn name(fid: u32) -> &'static str {
    get(fid).name
}

fn push(f: ExternalFunction) -> u32 {
    let mut v = FUNCS.write().unwrap();
    v.push(Arc::new(f));
    (v.len() - 1) as u32
}

// ---- declarations of the file being parsed ------------------------------------------------------

thread_local! {
    /// `external_function` declarations of the .mod file being parsed on this thread:
    /// name -> (function id, nargs).
    static DECLARED: RefCell<HashMap<String, (u32, usize)>> = RefCell::new(HashMap::new());
}

/// Forgets the declarations (start of a new .mod file).
pub(crate) fn clear_declared() {
    DECLARED.with(|d| d.borrow_mut().clear());
}

/// `(id, nargs)` if `name` was declared by `external_function` in the file being parsed.
pub(crate) fn declared(name: &str) -> Option<(u32, usize)> {
    DECLARED.with(|d| d.borrow().get(name).copied())
}

/// A derivative option of `external_function`.
#[derive(Clone, Debug, PartialEq)]
pub enum DerivOpt {
    /// Not given: numerical derivatives.
    Numerical,
    /// Given without a name: an extra output of the function itself.
    SameFunction,
    /// `= NAME`: a separate function.
    Named(String),
}

/// `external_function(name = NAME, nargs = N, first_deriv_provided [= F1], second_deriv_provided [= F2])`:
/// resolves the implementation (Rust registry first, then `NAME.m` in `dir`) and declares it
/// for the rest of the file. Returns the function id.
pub(crate) fn declare(name: &str, nargs: usize, first: DerivOpt, second: DerivOpt, dir: Option<&Path>) -> Result<u32> {
    let mut mcache: HashMap<String, Arc<MFunction>> = HashMap::new();
    let mut load_m = |fname: &str| -> Result<Arc<MFunction>> {
        if let Some(f) = mcache.get(fname) {
            return Ok(f.clone());
        }
        let file = dir.map(|d| d.join(format!("{fname}.m"))).unwrap_or_else(|| format!("{fname}.m").into());
        if !file.exists() {
            bail!(
                "external_function: no Rust function registered under '{fname}' \
                 (dynare::model::external::register_external_function) and no file {}",
                file.display()
            );
        }
        let f = Arc::new(MFunction::load(&file)?);
        mcache.insert(fname.to_string(), f.clone());
        Ok(f)
    };
    let check_nargs = |f: &MFunction| -> Result<()> {
        if f.nargin() != nargs {
            bail!(
                "{}: function {} takes {} arguments, external_function declares nargs = {nargs}",
                f.file,
                f.name,
                f.nargin()
            );
        }
        Ok(())
    };
    let (value, gradient, hessian): (ScalarFn, Option<VecFn>, Option<VecFn>) = if let Some(u) = user(name) {
        let mut pick = |opt: &DerivOpt, own: &Option<VecFn>, what: &str| -> Result<Option<VecFn>> {
            Ok(match opt {
                DerivOpt::Numerical => None,
                DerivOpt::SameFunction => Some(own.clone().ok_or_else(|| {
                    anyhow!("external_function(name = {name}): {what}_deriv_provided, but the registered Rust function has no {what} derivative")
                })?),
                DerivOpt::Named(n) => Some(named_vec(n, &mut load_m, nargs, what == "second")?),
            })
        };
        (u.value.clone(), pick(&first, &u.gradient, "first")?, pick(&second, &u.hessian, "second")?)
    } else {
        let f = load_m(name)?;
        check_nargs(&f)?;
        let pick = |opt: &DerivOpt,
                    out: usize,
                    load_m: &mut dyn FnMut(&str) -> Result<Arc<MFunction>>|
         -> Result<Option<VecFn>> {
            Ok(match opt {
                DerivOpt::Numerical => None,
                DerivOpt::SameFunction => {
                    if f.nargout() <= out {
                        bail!(
                            "{}: {} has {} output(s); {}_deriv_provided without a name needs output {}",
                            f.file,
                            f.name,
                            f.nargout(),
                            if out == 1 { "first" } else { "second" },
                            out + 1
                        );
                    }
                    let g = f.clone();
                    Some(Arc::new(move |a: &[f64]| {
                        g.call(a, out + 1)
                            .map(|mut o| o.swap_remove(out).data)
                            .unwrap_or_else(|e| fail(format!("{e:#}")))
                    }))
                }
                DerivOpt::Named(n) => Some(named_vec(n, load_m, nargs, out == 2)?),
            })
        };
        let g = pick(&first, 1, &mut load_m)?;
        let h = pick(&second, 2, &mut load_m)?;
        let v = f.clone();
        let value: ScalarFn = Arc::new(move |a: &[f64]| match v.call(a, 1) {
            Ok(o) if o[0].data.len() == 1 => o[0].data[0],
            Ok(o) => {
                fail(format!("{}: {} returned a {}x{} value instead of a scalar", v.file, v.name, o[0].rows, o[0].cols))
            }
            Err(e) => fail(format!("{e:#}")),
        });
        (value, g, h)
    };
    let fid =
        push(ExternalFunction { name: Box::leak(name.to_string().into_boxed_str()), nargs, value, gradient, hessian });
    DECLARED.with(|d| d.borrow_mut().insert(name.to_string(), (fid, nargs)));
    Ok(fid)
}

/// A separately named derivative function (`second`: the Hessian): Rust registry, else `NAME.m`.
/// From the registry: a vector function ([`register_external_vector_function`]), else the
/// Hessian (`second`) or gradient registered with it, else its scalar value.
fn named_vec(
    n: &str,
    load_m: &mut dyn FnMut(&str) -> Result<Arc<MFunction>>,
    nargs: usize,
    second: bool,
) -> Result<VecFn> {
    if let Some(u) = user(n) {
        let own = if second { u.hessian.as_ref().or(u.gradient.as_ref()) } else { u.gradient.as_ref() };
        return Ok(match own {
            Some(g) => g.clone(),
            None => {
                let v = u.value.clone();
                Arc::new(move |a: &[f64]| vec![v(a)])
            }
        });
    }
    let f = load_m(n)?;
    if f.nargin() != nargs {
        bail!(
            "{}: function {} takes {} arguments, external_function declares nargs = {nargs}",
            f.file,
            f.name,
            f.nargin()
        );
    }
    Ok(Arc::new(move |a: &[f64]| {
        f.call(a, 1).map(|mut o| o.swap_remove(0).data).unwrap_or_else(|e| fail(format!("{e:#}")))
    }))
}

// ---- MATLAB function files ---------------------------------------------------------------------

/// A MATLAB value: a real matrix (column-major); scalars are 1x1.
#[derive(Clone, Debug, PartialEq)]
pub struct Val {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Val {
    fn s(v: f64) -> Val {
        Val { rows: 1, cols: 1, data: vec![v] }
    }
    fn scalar(&self) -> Option<f64> {
        (self.data.len() == 1).then(|| self.data[0])
    }
    fn truthy(&self) -> bool {
        !self.data.is_empty() && self.data.iter().all(|&x| x != 0.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tk {
    Num(f64),
    Id(String),
    Op(&'static str),
    /// Statement separator: `;`, `,` (outside brackets) or a newline.
    Sep,
}

#[derive(Clone, Debug)]
struct Token {
    t: Tk,
    line: usize,
}

const OPS: [&str; 27] = [
    ":", "...", ".*", "./", ".^", "==", "~=", "!=", "<=", ">=", "&&", "||", "+", "-", "*", "/", "^", "<", ">", "&",
    "|", "~", "!", "(", ")", "=", "'",
];

fn tokenize(src: &str, file: &str) -> Result<Vec<Token>> {
    let mut out: Vec<Token> = Vec::new();
    let mut brackets: Vec<char> = Vec::new();
    let mut in_block_comment = false;
    for (ln, raw) in src.lines().enumerate() {
        let line = ln + 1;
        let trimmed = raw.trim();
        if in_block_comment {
            if trimmed == "%}" {
                in_block_comment = false;
            }
            continue;
        }
        if trimmed == "%{" {
            in_block_comment = true;
            continue;
        }
        let ch: Vec<char> = raw.chars().collect();
        let mut i = 0;
        let mut continued = false;
        let ends_value = |out: &Vec<Token>| {
            matches!(out.last().map(|t| &t.t), Some(Tk::Num(_) | Tk::Id(_) | Tk::Op(")") | Tk::Op("]") | Tk::Op("'")))
        };
        while i < ch.len() {
            let c = ch[i];
            if c == '%' || c == '#' {
                break;
            }
            if c.is_whitespace() {
                // whitespace separates the elements of a matrix literal: [a b], [a -b]
                let mut j = i;
                while j < ch.len() && ch[j].is_whitespace() {
                    j += 1;
                }
                if brackets.last() == Some(&'[') && ends_value(&out) && j < ch.len() {
                    let n = ch[j];
                    let starts = n.is_ascii_alphanumeric()
                        || n == '_'
                        || n == '('
                        || n == '['
                        || n == '.' && ch.get(j + 1).is_some_and(|d| d.is_ascii_digit());
                    let signed = (n == '-' || n == '+') && ch.get(j + 1).is_some_and(|d| !d.is_whitespace());
                    if starts || signed {
                        out.push(Token { t: Tk::Op(","), line });
                    }
                }
                i = j;
                continue;
            }
            if c.is_ascii_digit() || (c == '.' && ch.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
                let st = i;
                while i < ch.len() && ch[i].is_ascii_digit() {
                    i += 1;
                }
                if i < ch.len() && ch[i] == '.' && !ch.get(i + 1).is_some_and(|d| matches!(d, '*' | '/' | '^' | '\'')) {
                    i += 1;
                    while i < ch.len() && ch[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < ch.len() && (ch[i] == 'e' || ch[i] == 'E') {
                    let mut j = i + 1;
                    if j < ch.len() && (ch[j] == '+' || ch[j] == '-') {
                        j += 1;
                    }
                    if j < ch.len() && ch[j].is_ascii_digit() {
                        i = j;
                        while i < ch.len() && ch[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let s: String = ch[st..i].iter().collect();
                let v = s.parse::<f64>().map_err(|_| anyhow!("{file}:{line}: bad number '{s}'"))?;
                out.push(Token { t: Tk::Num(v), line });
                continue;
            }
            if c.is_ascii_alphabetic() || c == '_' {
                let st = i;
                while i < ch.len() && (ch[i].is_ascii_alphanumeric() || ch[i] == '_') {
                    i += 1;
                }
                out.push(Token { t: Tk::Id(ch[st..i].iter().collect()), line });
                continue;
            }
            match c {
                '[' | '{' => {
                    if c == '{' {
                        bail!("{file}:{line}: cell arrays are not supported in external function files");
                    }
                    brackets.push('[');
                    out.push(Token { t: Tk::Op("["), line });
                    i += 1;
                    continue;
                }
                ']' => {
                    brackets.pop();
                    out.push(Token { t: Tk::Op("]"), line });
                    i += 1;
                    continue;
                }
                ';' => {
                    let t = if brackets.last() == Some(&'[') { Tk::Op(";") } else { Tk::Sep };
                    out.push(Token { t, line });
                    i += 1;
                    continue;
                }
                ',' => {
                    let t = if brackets.is_empty() { Tk::Sep } else { Tk::Op(",") };
                    out.push(Token { t, line });
                    i += 1;
                    continue;
                }
                '"' => bail!("{file}:{line}: strings are not supported in external function files"),
                _ => {}
            }
            let rest: String = ch[i..].iter().take(3).collect();
            let Some(op) = OPS.iter().find(|o| rest.starts_with(**o)) else {
                bail!("{file}:{line}: unexpected character '{c}'");
            };
            if *op == "..." {
                continued = true;
                break;
            }
            if *op == "'" {
                if ends_value(&out) {
                    bail!("{file}:{line}: the transpose operator is not supported in external function files");
                }
                bail!("{file}:{line}: strings are not supported in external function files");
            }
            match *op {
                "(" => brackets.push('('),
                ")" => {
                    brackets.pop();
                }
                _ => {}
            }
            out.push(Token { t: Tk::Op(op), line });
            i += op.len();
        }
        if !continued {
            if brackets.last() == Some(&'[') {
                out.push(Token { t: Tk::Op(";"), line });
            } else if brackets.is_empty() {
                out.push(Token { t: Tk::Sep, line });
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Bop {
    Add,
    Sub,
    Mul,
    Div,
    /// `*` / `/`: matrix product / right division, supported when one operand is a scalar.
    MatMul,
    MatDiv,
    Pow,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    AndAnd,
    OrOr,
}

#[derive(Clone, Debug)]
enum Builtin {
    Un(super::expr::UnaryOp),
    Max,
    Min,
    Power,
    Zeros,
    Ones,
    Numel,
}

#[derive(Clone, Debug)]
enum Ex {
    Num(f64),
    Var(usize),
    Index(usize, Vec<Ex>),
    Call(Builtin, Vec<Ex>),
    Neg(Box<Ex>),
    Not(Box<Ex>),
    Bin(Bop, Box<Ex>, Box<Ex>),
    /// Matrix literal: rows of elements.
    Mat(Vec<Vec<Ex>>),
    Nargin,
    Nargout,
}

#[derive(Clone, Debug)]
enum St {
    Assign(usize, Vec<Ex>, Ex, usize),
    If(Vec<(Ex, Vec<St>)>, Vec<St>),
    Return,
}

/// A parsed MATLAB function file (`function [o1, ...] = name(a1, ...)`), restricted to what an
/// external function of a model needs: assignments of scalar expressions (`+ - * / ^`, their
/// element-wise forms, comparisons, `&& || ~`, the elementary functions of the .mod language,
/// `max`, `min`, `power`), `if`/`elseif`/`else`, matrix literals `[a, b; c, d]`, `zeros`,
/// `ones`, indexed assignments `g(i) = ...`, `H(i, j) = ...`, `nargin`, `nargout`, `return`.
/// Anything else (loops, other function calls, cell arrays, strings, local functions) is a
/// load error naming the file and line.
#[derive(Debug)]
pub struct MFunction {
    pub name: String,
    pub file: String,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    names: Vec<String>,
    body: Vec<St>,
}

struct P<'a> {
    t: &'a [Token],
    i: usize,
    file: &'a str,
    names: Vec<String>,
    assigned: std::collections::HashSet<String>,
}

impl<'a> P<'a> {
    fn peek(&self) -> Option<&'a Tk> {
        self.t.get(self.i).map(|t| &t.t)
    }
    fn line(&self) -> usize {
        self.t.get(self.i).or(self.t.last()).map(|t| t.line).unwrap_or(0)
    }
    fn err<T>(&self, msg: impl std::fmt::Display) -> Result<T> {
        bail!("{}:{}: {msg}", self.file, self.line())
    }
    fn is_op(&self, o: &str) -> bool {
        matches!(self.peek(), Some(Tk::Op(x)) if *x == o)
    }
    fn is_id(&self, s: &str) -> bool {
        matches!(self.peek(), Some(Tk::Id(x)) if x == s)
    }
    fn eat_op(&mut self, o: &str) -> bool {
        if self.is_op(o) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect_op(&mut self, o: &str) -> Result<()> {
        if self.eat_op(o) { Ok(()) } else { self.err(format!("expected '{o}'")) }
    }
    fn ident(&mut self) -> Result<String> {
        match self.peek() {
            Some(Tk::Id(s)) => {
                self.i += 1;
                Ok(s.clone())
            }
            _ => self.err("expected a name"),
        }
    }
    fn skip_seps(&mut self) {
        while matches!(self.peek(), Some(Tk::Sep)) {
            self.i += 1;
        }
    }
    fn slot(&mut self, n: &str) -> usize {
        match self.names.iter().position(|x| x == n) {
            Some(k) => k,
            None => {
                self.names.push(n.to_string());
                self.names.len() - 1
            }
        }
    }

    fn block(&mut self, stops: &[&str]) -> Result<Vec<St>> {
        let mut v = Vec::new();
        loop {
            self.skip_seps();
            match self.peek() {
                None => return Ok(v),
                Some(Tk::Id(s)) if stops.contains(&s.as_str()) => return Ok(v),
                _ => v.push(self.statement()?),
            }
        }
    }

    fn statement(&mut self) -> Result<St> {
        let line = self.line();
        let name = match self.peek() {
            Some(Tk::Id(s)) => s.clone(),
            _ => return self.err("expected a statement"),
        };
        match name.as_str() {
            "if" => {
                self.i += 1;
                let mut arms = Vec::new();
                let c = self.expr()?;
                let b = self.block(&["elseif", "else", "end"])?;
                arms.push((c, b));
                let mut els = Vec::new();
                loop {
                    match self.peek() {
                        Some(Tk::Id(s)) if s == "elseif" => {
                            self.i += 1;
                            let c = self.expr()?;
                            let b = self.block(&["elseif", "else", "end"])?;
                            arms.push((c, b));
                        }
                        Some(Tk::Id(s)) if s == "else" => {
                            self.i += 1;
                            els = self.block(&["end"])?;
                        }
                        Some(Tk::Id(s)) if s == "end" => {
                            self.i += 1;
                            break;
                        }
                        _ => return self.err("'if' not closed by 'end'"),
                    }
                }
                Ok(St::If(arms, els))
            }
            "return" => {
                self.i += 1;
                Ok(St::Return)
            }
            "for" | "while" | "switch" | "try" | "parfor" | "global" | "persistent" | "function" | "break"
            | "continue" => self.err(format!(
                "'{name}' is not supported in external function files (only assignments and if/elseif/else)"
            )),
            _ => {
                self.i += 1;
                let mut idx = Vec::new();
                if self.eat_op("(") {
                    idx.push(self.expr()?);
                    while self.eat_op(",") {
                        idx.push(self.expr()?);
                    }
                    self.expect_op(")")?;
                    if idx.len() > 2 {
                        return self.err("only one or two indices are supported");
                    }
                }
                if !self.eat_op("=") {
                    return self.err(format!(
                        "expected an assignment '{name} = ...' (function calls as statements are not supported)"
                    ));
                }
                let e = self.expr()?;
                if !matches!(self.peek(), None | Some(Tk::Sep)) {
                    return self.err("unexpected token after the expression");
                }
                self.assigned.insert(name.clone());
                let s = self.slot(&name);
                Ok(St::Assign(s, idx, e, line))
            }
        }
    }

    // precedence (MATLAB): || < && < | < & < comparison < +- < */ < unary < ^
    fn expr(&mut self) -> Result<Ex> {
        let mut a = self.and_and()?;
        while self.eat_op("||") {
            a = Ex::Bin(Bop::OrOr, Box::new(a), Box::new(self.and_and()?));
        }
        Ok(a)
    }
    fn and_and(&mut self) -> Result<Ex> {
        let mut a = self.or()?;
        while self.eat_op("&&") {
            a = Ex::Bin(Bop::AndAnd, Box::new(a), Box::new(self.or()?));
        }
        Ok(a)
    }
    fn or(&mut self) -> Result<Ex> {
        let mut a = self.and()?;
        while self.eat_op("|") {
            a = Ex::Bin(Bop::Or, Box::new(a), Box::new(self.and()?));
        }
        Ok(a)
    }
    fn and(&mut self) -> Result<Ex> {
        let mut a = self.cmp()?;
        while self.eat_op("&") {
            a = Ex::Bin(Bop::And, Box::new(a), Box::new(self.cmp()?));
        }
        Ok(a)
    }
    fn cmp(&mut self) -> Result<Ex> {
        let mut a = self.add()?;
        loop {
            let op = match self.peek() {
                Some(Tk::Op("<")) => Bop::Lt,
                Some(Tk::Op(">")) => Bop::Gt,
                Some(Tk::Op("<=")) => Bop::Le,
                Some(Tk::Op(">=")) => Bop::Ge,
                Some(Tk::Op("==")) => Bop::Eq,
                Some(Tk::Op("~=" | "!=")) => Bop::Ne,
                _ => return Ok(a),
            };
            self.i += 1;
            a = Ex::Bin(op, Box::new(a), Box::new(self.add()?));
        }
    }
    fn add(&mut self) -> Result<Ex> {
        let mut a = self.mul()?;
        loop {
            let op = if self.eat_op("+") {
                Bop::Add
            } else if self.eat_op("-") {
                Bop::Sub
            } else {
                return Ok(a);
            };
            a = Ex::Bin(op, Box::new(a), Box::new(self.mul()?));
        }
    }
    fn mul(&mut self) -> Result<Ex> {
        let mut a = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Tk::Op(".*")) => Bop::Mul,
                Some(Tk::Op("./")) => Bop::Div,
                Some(Tk::Op("*")) => Bop::MatMul,
                Some(Tk::Op("/")) => Bop::MatDiv,
                _ => return Ok(a),
            };
            self.i += 1;
            a = Ex::Bin(op, Box::new(a), Box::new(self.unary()?));
        }
    }
    fn unary(&mut self) -> Result<Ex> {
        if self.eat_op("-") {
            return Ok(Ex::Neg(Box::new(self.unary()?)));
        }
        if self.eat_op("+") {
            return self.unary();
        }
        if self.eat_op("~") || self.eat_op("!") {
            return Ok(Ex::Not(Box::new(self.unary()?)));
        }
        self.power()
    }
    fn power(&mut self) -> Result<Ex> {
        let mut a = self.primary()?;
        // left-associative, binds tighter than unary minus: -2^2 = -4, 2^-1 = 0.5
        while self.is_op("^") || self.is_op(".^") {
            self.i += 1;
            let b = if self.eat_op("-") {
                Ex::Neg(Box::new(self.primary()?))
            } else {
                self.eat_op("+");
                self.primary()?
            };
            a = Ex::Bin(Bop::Pow, Box::new(a), Box::new(b));
        }
        Ok(a)
    }
    fn args(&mut self) -> Result<Vec<Ex>> {
        let mut v = Vec::new();
        if self.eat_op(")") {
            return Ok(v);
        }
        v.push(self.expr()?);
        while self.eat_op(",") {
            v.push(self.expr()?);
        }
        self.expect_op(")")?;
        Ok(v)
    }
    fn primary(&mut self) -> Result<Ex> {
        let line_err = |p: &Self, m: String| p.err::<Ex>(m);
        match self.peek().cloned() {
            Some(Tk::Num(v)) => {
                self.i += 1;
                Ok(Ex::Num(v))
            }
            Some(Tk::Op("(")) => {
                self.i += 1;
                let e = self.expr()?;
                self.expect_op(")")?;
                Ok(e)
            }
            Some(Tk::Op("[")) => {
                self.i += 1;
                let mut rows: Vec<Vec<Ex>> = vec![Vec::new()];
                loop {
                    if self.eat_op("]") {
                        break;
                    }
                    if self.eat_op(";") {
                        rows.push(Vec::new());
                        continue;
                    }
                    if self.eat_op(",") {
                        continue;
                    }
                    if self.peek().is_none() {
                        return self.err("'[' not closed");
                    }
                    let e = self.expr()?;
                    rows.last_mut().unwrap().push(e);
                }
                rows.retain(|r| !r.is_empty());
                Ok(Ex::Mat(rows))
            }
            Some(Tk::Id(name)) => {
                self.i += 1;
                if self.names.contains(&name) || self.assigned.contains(&name) {
                    let s = self.slot(&name);
                    if self.eat_op("(") {
                        let idx = self.args()?;
                        if idx.is_empty() || idx.len() > 2 {
                            return self.err("only one or two indices are supported");
                        }
                        return Ok(Ex::Index(s, idx));
                    }
                    return Ok(Ex::Var(s));
                }
                let constant = match name.as_str() {
                    "pi" => Some(std::f64::consts::PI),
                    "Inf" | "inf" => Some(f64::INFINITY),
                    "NaN" | "nan" => Some(f64::NAN),
                    "eps" => Some(f64::EPSILON),
                    "true" => Some(1.0),
                    "false" => Some(0.0),
                    _ => None,
                };
                if let Some(c) = constant {
                    if self.eat_op("(") && !self.eat_op(")") {
                        return self.err(format!("'{name}(...)' is not supported"));
                    }
                    return Ok(Ex::Num(c));
                }
                match name.as_str() {
                    "nargin" => return Ok(Ex::Nargin),
                    "nargout" => return Ok(Ex::Nargout),
                    _ => {}
                }
                let b = match name.as_str() {
                    "max" => Builtin::Max,
                    "min" => Builtin::Min,
                    "power" => Builtin::Power,
                    "zeros" => Builtin::Zeros,
                    "ones" => Builtin::Ones,
                    "numel" | "length" => Builtin::Numel,
                    "log" | "ln" => Builtin::Un(super::expr::UnaryOp::Log),
                    n => match super::expr::UnaryOp::from_name(n) {
                        Some(op) => Builtin::Un(op),
                        None => {
                            return line_err(
                                self,
                                format!(
                                    "'{name}' is neither a variable assigned in this function nor a supported function"
                                ),
                            );
                        }
                    },
                };
                if !self.eat_op("(") {
                    return self.err(format!("'{name}' must be called with arguments"));
                }
                let a = self.args()?;
                let ok = match b {
                    Builtin::Un(_) | Builtin::Numel => a.len() == 1,
                    Builtin::Max | Builtin::Min | Builtin::Power => a.len() == 2,
                    Builtin::Zeros | Builtin::Ones => a.len() == 1 || a.len() == 2,
                };
                if !ok {
                    return self.err(format!("wrong number of arguments to {name}()"));
                }
                Ok(Ex::Call(b, a))
            }
            Some(t) => self.err(format!("unexpected {t:?} in an expression")),
            None => self.err("unexpected end of file"),
        }
    }
}

struct Env<'a> {
    vars: Vec<Option<Val>>,
    nargin: usize,
    nargout: usize,
    f: &'a MFunction,
}

impl MFunction {
    /// Loads and parses a `.m` function file.
    pub fn load(path: &Path) -> Result<MFunction> {
        let src = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&src, &path.display().to_string())
    }

    /// Parses the text of a function file (`file` is used in messages).
    pub fn parse(src: &str, file: &str) -> Result<MFunction> {
        let toks = tokenize(src, file)?;
        let mut p = P { t: &toks, i: 0, file, names: Vec::new(), assigned: Default::default() };
        p.skip_seps();
        if !p.is_id("function") {
            return p.err("expected 'function' (scripts are not supported as external functions)");
        }
        p.i += 1;
        // outputs
        let mut outs = Vec::new();
        if p.eat_op("[") {
            while !p.eat_op("]") {
                if p.eat_op(",") {
                    continue;
                }
                outs.push(p.ident()?);
            }
            p.expect_op("=")?;
        } else if matches!(p.t.get(p.i + 1).map(|t| &t.t), Some(Tk::Op("="))) {
            outs.push(p.ident()?);
            p.expect_op("=")?;
        }
        if outs.is_empty() {
            return p.err("the function must have at least one output");
        }
        let name = p.ident()?;
        let mut ins = Vec::new();
        if p.eat_op("(") {
            while !p.eat_op(")") {
                if p.eat_op(",") {
                    continue;
                }
                ins.push(p.ident()?);
            }
        }
        let inputs: Vec<usize> = ins.iter().map(|n| p.slot(n)).collect();
        for n in &ins {
            p.assigned.insert(n.clone());
        }
        let outputs: Vec<usize> = outs.iter().map(|n| p.slot(n)).collect();
        let body = p.block(&["end", "function"])?;
        if p.is_id("end") {
            p.i += 1;
        }
        p.skip_seps();
        if p.is_id("function") {
            return p.err("local functions are not supported in external function files");
        }
        if p.peek().is_some() {
            return p.err("unexpected statement after the end of the function");
        }
        let names = p.names;
        Ok(MFunction { name, file: file.to_string(), inputs, outputs, names, body })
    }

    pub fn nargin(&self) -> usize {
        self.inputs.len()
    }
    pub fn nargout(&self) -> usize {
        self.outputs.len()
    }

    /// Calls the function with scalar arguments, asking for `nargout` outputs.
    pub fn call(&self, args: &[f64], nargout: usize) -> Result<Vec<Val>> {
        if args.len() > self.inputs.len() {
            bail!("{}: {} called with {} arguments (takes {})", self.file, self.name, args.len(), self.inputs.len());
        }
        if nargout > self.outputs.len() {
            bail!("{}: {} called with {} outputs (has {})", self.file, self.name, nargout, self.outputs.len());
        }
        let mut env = Env { vars: vec![None; self.names.len()], nargin: args.len(), nargout, f: self };
        for (&s, &a) in self.inputs.iter().zip(args) {
            env.vars[s] = Some(Val::s(a));
        }
        env.block(&self.body)?;
        let mut out = Vec::with_capacity(nargout);
        for &s in &self.outputs[..nargout.max(1)] {
            match env.vars[s].take() {
                Some(v) => out.push(v),
                None => bail!("{}: output '{}' of {} is not assigned", self.file, self.names[s], self.name),
            }
        }
        Ok(out)
    }
}

/// `true` when a `return` was executed.
type Flow = bool;

impl Env<'_> {
    fn block(&mut self, b: &[St]) -> Result<Flow> {
        for s in b {
            if self.st(s)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn st(&mut self, s: &St) -> Result<Flow> {
        match s {
            St::Return => Ok(true),
            St::If(arms, els) => {
                for (c, b) in arms {
                    if self.ex(c)?.truthy() {
                        return self.block(b);
                    }
                }
                self.block(els)
            }
            St::Assign(slot, idx, e, line) => {
                let v = self.ex(e)?;
                if idx.is_empty() {
                    self.vars[*slot] = Some(v);
                    return Ok(false);
                }
                let x = v
                    .scalar()
                    .ok_or_else(|| anyhow!("{}:{line}: indexed assignment of a non-scalar value", self.f.file))?;
                let ix: Vec<usize> = idx.iter().map(|i| self.index(i, *line)).collect::<Result<_>>()?;
                let cur = self.vars[*slot].take().unwrap_or(Val { rows: 0, cols: 0, data: Vec::new() });
                let (r, c) = if ix.len() == 1 {
                    // linear index: a row vector grows along its columns, a column along its rows
                    if cur.cols == 1 && cur.rows > 1 {
                        (ix[0], 1)
                    } else if cur.rows <= 1 {
                        (1, ix[0])
                    } else {
                        let k = ix[0] - 1;
                        if k >= cur.data.len() {
                            bail!(
                                "{}:{line}: linear index {} out of the bounds of a {}x{} matrix",
                                self.f.file,
                                ix[0],
                                cur.rows,
                                cur.cols
                            );
                        }
                        let mut cur = cur;
                        cur.data[k] = x;
                        self.vars[*slot] = Some(cur);
                        return Ok(false);
                    }
                } else {
                    (ix[0], ix[1])
                };
                let (nr, nc) = (cur.rows.max(r), cur.cols.max(c));
                let mut m = Val { rows: nr, cols: nc, data: vec![0.0; nr * nc] };
                for j in 0..cur.cols {
                    for i in 0..cur.rows {
                        m.data[j * nr + i] = cur.data[j * cur.rows + i];
                    }
                }
                m.data[(c - 1) * nr + r - 1] = x;
                self.vars[*slot] = Some(m);
                Ok(false)
            }
        }
    }

    fn index(&mut self, e: &Ex, line: usize) -> Result<usize> {
        let v = self.ex(e)?.scalar().ok_or_else(|| anyhow!("{}:{line}: non-scalar index", self.f.file))?;
        if v < 1.0 || v.fract() != 0.0 {
            bail!("{}:{line}: index {v} is not a positive integer", self.f.file);
        }
        Ok(v as usize)
    }

    fn ex(&mut self, e: &Ex) -> Result<Val> {
        Ok(match e {
            Ex::Num(v) => Val::s(*v),
            Ex::Nargin => Val::s(self.nargin as f64),
            Ex::Nargout => Val::s(self.nargout as f64),
            Ex::Var(s) => match &self.vars[*s] {
                Some(v) => v.clone(),
                None => bail!(
                    "{}: {}: variable '{}' used before it is assigned",
                    self.f.file,
                    self.f.name,
                    self.f.names[*s]
                ),
            },
            Ex::Index(s, idx) => {
                let ix: Vec<usize> = idx.iter().map(|i| self.index(i, 0)).collect::<Result<_>>()?;
                let Some(v) = &self.vars[*s] else {
                    bail!(
                        "{}: {}: variable '{}' used before it is assigned",
                        self.f.file,
                        self.f.name,
                        self.f.names[*s]
                    );
                };
                let k = if ix.len() == 1 {
                    ix[0] - 1
                } else {
                    if ix[0] > v.rows || ix[1] > v.cols {
                        bail!(
                            "{}: index ({}, {}) out of the bounds of '{}'",
                            self.f.file,
                            ix[0],
                            ix[1],
                            self.f.names[*s]
                        );
                    }
                    (ix[1] - 1) * v.rows + ix[0] - 1
                };
                match v.data.get(k) {
                    Some(&x) => Val::s(x),
                    None => bail!("{}: index out of the bounds of '{}'", self.f.file, self.f.names[*s]),
                }
            }
            Ex::Neg(a) => {
                let mut v = self.ex(a)?;
                v.data.iter_mut().for_each(|x| *x = -*x);
                v
            }
            Ex::Not(a) => {
                let mut v = self.ex(a)?;
                v.data.iter_mut().for_each(|x| *x = if *x == 0.0 { 1.0 } else { 0.0 });
                v
            }
            Ex::Bin(Bop::AndAnd, a, b) => Val::s(if self.ex(a)?.truthy() && self.ex(b)?.truthy() { 1.0 } else { 0.0 }),
            Ex::Bin(Bop::OrOr, a, b) => Val::s(if self.ex(a)?.truthy() || self.ex(b)?.truthy() { 1.0 } else { 0.0 }),
            Ex::Bin(op, a, b) => {
                let (x, y) = (self.ex(a)?, self.ex(b)?);
                if matches!(op, Bop::MatMul | Bop::MatDiv | Bop::Pow) && x.scalar().is_none() && y.scalar().is_none() {
                    bail!(
                        "{}: {}: matrix products, divisions and powers are not supported (use .*, ./, .^)",
                        self.f.file,
                        self.f.name
                    );
                }
                let f = |u: f64, v: f64| -> f64 {
                    let t = |c: bool| if c { 1.0 } else { 0.0 };
                    match op {
                        Bop::Add => u + v,
                        Bop::Sub => u - v,
                        Bop::Mul | Bop::MatMul => u * v,
                        Bop::Div | Bop::MatDiv => u / v,
                        Bop::Pow => super::expr::BinaryOp::Pow.apply(u, v),
                        Bop::Lt => t(u < v),
                        Bop::Gt => t(u > v),
                        Bop::Le => t(u <= v),
                        Bop::Ge => t(u >= v),
                        Bop::Eq => t(u == v),
                        Bop::Ne => t(u != v),
                        Bop::And => t(u != 0.0 && v != 0.0),
                        Bop::Or => t(u != 0.0 || v != 0.0),
                        Bop::AndAnd | Bop::OrOr => unreachable!(),
                    }
                };
                self.elementwise(&x, &y, f)?
            }
            Ex::Call(b, args) => {
                let v: Vec<Val> = args.iter().map(|a| self.ex(a)).collect::<Result<_>>()?;
                match b {
                    Builtin::Un(op) => {
                        let mut r = v[0].clone();
                        r.data.iter_mut().for_each(|x| *x = op.apply(*x));
                        r
                    }
                    Builtin::Max => self.elementwise(&v[0], &v[1], |a, b| super::expr::BinaryOp::Max.apply(a, b))?,
                    Builtin::Min => self.elementwise(&v[0], &v[1], |a, b| super::expr::BinaryOp::Min.apply(a, b))?,
                    Builtin::Power => self.elementwise(&v[0], &v[1], |a, b| super::expr::BinaryOp::Pow.apply(a, b))?,
                    Builtin::Numel => Val::s(v[0].data.len() as f64),
                    Builtin::Zeros | Builtin::Ones => {
                        let dim = |x: &Val| x.scalar().filter(|d| *d >= 0.0 && d.fract() == 0.0).map(|d| d as usize);
                        let (Some(r), Some(c)) = (dim(&v[0]), dim(v.get(1).unwrap_or(&v[0]))) else {
                            bail!("{}: zeros/ones: dimensions must be non-negative integers", self.f.file);
                        };
                        let fill = if matches!(b, Builtin::Ones) { 1.0 } else { 0.0 };
                        Val { rows: r, cols: c, data: vec![fill; r * c] }
                    }
                }
            }
            Ex::Mat(rows) => {
                let mut vals: Vec<Vec<f64>> = Vec::new();
                for r in rows {
                    let mut row = Vec::new();
                    for e in r {
                        let v = self.ex(e)?;
                        if v.rows > 1 {
                            bail!("{}: matrix literals may only contain scalars and row vectors", self.f.file);
                        }
                        row.extend(v.data);
                    }
                    vals.push(row);
                }
                let nr = vals.len();
                let nc = vals.first().map_or(0, |r| r.len());
                if vals.iter().any(|r| r.len() != nc) {
                    bail!("{}: rows of a matrix literal have different lengths", self.f.file);
                }
                let mut data = vec![0.0; nr * nc];
                for (i, r) in vals.iter().enumerate() {
                    for (j, &x) in r.iter().enumerate() {
                        data[j * nr + i] = x;
                    }
                }
                Val { rows: nr, cols: nc, data }
            }
        })
    }

    fn elementwise(&self, x: &Val, y: &Val, f: impl Fn(f64, f64) -> f64) -> Result<Val> {
        Ok(if let Some(a) = x.scalar() {
            Val { rows: y.rows, cols: y.cols, data: y.data.iter().map(|&b| f(a, b)).collect() }
        } else if let Some(b) = y.scalar() {
            Val { rows: x.rows, cols: x.cols, data: x.data.iter().map(|&a| f(a, b)).collect() }
        } else if x.rows == y.rows && x.cols == y.cols {
            Val { rows: x.rows, cols: x.cols, data: x.data.iter().zip(&y.data).map(|(&a, &b)| f(a, b)).collect() }
        } else {
            bail!(
                "{}: operation on a {}x{} and a {}x{} matrix (only scalar and element-wise arithmetic is supported)",
                self.f.file,
                x.rows,
                x.cols,
                y.rows,
                y.cols
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_function_interpreter() {
        let src = "function [y, g, h] = f(a, b) % comment\n\
                   %{\n block comment\n %}\n\
                   y = a^2*b + exp(-b) - 2^-1;\n\
                   if nargout > 1\n g = [2*a*b, a^2 - exp(-b)];\n end\n\
                   if nargout > 2, h = zeros(2); h(1,1) = 2*b; h(1, 2) = 2*a; h(2,1) = h(1,2); h(2,2) = exp(-b); end\n\
                   if a > 10 && b ~= 0\n y = -y;\n elseif a < -10\n y = 0;\n else\n y = y + 0;\n end\n\
                   end\n";
        let f = MFunction::parse(src, "f.m").unwrap();
        assert_eq!((f.nargin(), f.nargout()), (2, 3));
        let o = f.call(&[1.5, 0.5], 3).unwrap();
        assert!((o[0].data[0] - (1.5f64.powi(2) * 0.5 + (-0.5f64).exp() - 0.5)).abs() < 1e-15);
        assert_eq!(o[1].data, vec![1.5, 2.25 - (-0.5f64).exp()]);
        assert_eq!((o[2].rows, o[2].cols), (2, 2));
        assert_eq!(o[2].data[2], 3.0);
        let o = f.call(&[11.0, 1.0], 1).unwrap();
        assert!(o[0].data[0] < 0.0);
        // matrix literal with spaces, unary minus, and continuation
        let g = MFunction::parse("function m = g(x)\nm = [x -x; 2*x ...\n 3];\nend", "g.m").unwrap();
        assert_eq!(g.call(&[2.0], 1).unwrap()[0].data, vec![2.0, 4.0, -2.0, 3.0]);
        // errors
        assert!(MFunction::parse("function y = f(x)\nfor i=1:2\n y = x;\nend\nend", "f.m").is_err());
        assert!(MFunction::parse("function y = f(x)\ny = foo(x);\nend", "f.m").is_err());
        assert!(MFunction::parse("y = 1;", "f.m").is_err());
    }

    #[test]
    fn numerical_derivatives_as_dynare() {
        let f = ExternalFunction {
            name: "t",
            nargs: 2,
            value: Arc::new(|a: &[f64]| a[0].powi(3) * a[1] + a[1] * a[1]),
            gradient: None,
            hessian: None,
        };
        let a = [2.0, 3.0];
        assert!((f.eval(1, 0, &a) - 36.0).abs() < 1e-6);
        assert!((f.eval(2, 0, &a) - 14.0).abs() < 1e-6);
        assert!((f.eval(1, 2, &a) - 12.0).abs() < 1e-2);
    }
}
