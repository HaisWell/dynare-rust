// SPDX-License-Identifier: GPL-3.0-or-later
//! Dynare's MCMC files : the draws of every chain in
//! `<dname>/metropolis/<fname>_mh<k>_blck<b>.mat` (`x2`, `logpo2`, `LastSeeds`,
//! `accepted_draws_this_chain`, `accepted_draws_this_file`, `feval_this_chain`,
//! `feval_this_file`), at most `MAX_nruns = ceil(MaxNumberOfBytes/(npar+2)/8)` draws per
//! file, and the session records `<fname>_mh_history_<i>.mat` (`record`, `version = 2`),
//! as `posterior_sampler_initialization.m`, `posterior_sampler_core.m`,
//! `write_mh_history_file.m`, `update_last_mh_history_file.m`,
//! `load_last_mh_history_file.m`, `CutSample.m` and `compute_mh_covariance_matrix.m`
//! write and read them. Files written here load in MATLAB/Dynare and Dynare's files load
//! here, so either program can post-process or continue the other's chains.
//!
//! Random streams: Dynare stores MATLAB's generator state (`Unifor`, `Normal`, 625
//! `uint32` for `mt19937ar`). dynare-rust stores its ChaCha8 state in the same fields
//! (seed, stream and word position as 14 numbers, `algo = 'chacha8 (dynare-rust)'`), so a
//! continued chain is bit-identical to one longer run. MATLAB cannot set that state and
//! reseeds the chain (`computeMHBlock`'s `catch`); a MATLAB state read here seeds ChaCha8
//! with a hash of the state (reproducible, not MATLAB's draws).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::linalg::Mat;
use crate::matfile::{self, MatValue};

/// `options_.MaxNumberOfBytes` (default_option_values.m).
pub const MAX_NUMBER_OF_BYTES: f64 = 1e8;

/// The `algo` written with dynare-rust's random-stream states.
pub const CHACHA_ALGO: &str = "chacha8 (dynare-rust)";

/// `MAX_nruns = ceil(MaxNumberOfBytes/(npar+2)/8)`: draws per mh file.
pub fn max_nruns(npar: usize, max_bytes: f64) -> usize {
    (max_bytes / (npar as f64 + 2.0) / 8.0).ceil() as usize
}

/// File names in `<dname>/metropolis/`.
#[derive(Clone, Debug)]
pub struct MhPaths {
    pub folder: PathBuf,
    pub model: String,
}

impl MhPaths {
    /// `CheckPath('metropolis', M_.dname)`; `dname` is `<dir of the .mod>/<fname>`.
    pub fn new(dname: &Path, fname: &str) -> Self {
        MhPaths { folder: dname.join("metropolis"), model: fname.to_string() }
    }

    /// `<fname>_mh<file>_blck<blck>.mat` (both 1-based).
    pub fn mh_file(&self, file: usize, blck: usize) -> PathBuf {
        self.folder.join(format!("{}_mh{file}_blck{blck}.mat", self.model))
    }

    pub fn history(&self, i: usize) -> PathBuf {
        self.folder.join(format!("{}_mh_history_{i}.mat", self.model))
    }

    pub fn log(&self) -> PathBuf {
        self.folder.join("metropolis.log")
    }

    fn list(&self) -> Vec<String> {
        std::fs::read_dir(&self.folder)
            .map(|rd| rd.filter_map(|e| e.ok()?.file_name().into_string().ok()).collect())
            .unwrap_or_default()
    }

    /// Number of `<fname>_mh_history_*.mat` files.
    pub fn history_count(&self) -> usize {
        let pre = format!("{}_mh_history_", self.model);
        self.list().iter().filter(|f| f.starts_with(&pre) && f.ends_with(".mat")).count()
    }

    /// File numbers `k` of the `<fname>_mh<k>_blck<blck>.mat` present (sorted; no tmp files).
    pub fn mh_files_of_block(&self, blck: usize) -> Vec<usize> {
        let pre = format!("{}_mh", self.model);
        let suf = format!("_blck{blck}.mat");
        let mut v: Vec<usize> =
            self.list().iter().filter_map(|f| f.strip_prefix(&pre)?.strip_suffix(&suf)?.parse().ok()).collect();
        v.sort_unstable();
        v
    }

    /// Deletes the mh files and history files (fresh run: `posterior_sampler_initialization.m`).
    pub fn delete_all(&self) -> Result<usize> {
        let pre_mh = format!("{}_mh", self.model);
        let mut n = 0;
        for f in self.list() {
            let is_mh = f.starts_with(&pre_mh) && f.contains("_blck") && f.ends_with(".mat");
            let is_hist = f.starts_with(&format!("{}_mh_history", self.model)) && f.ends_with(".mat");
            if is_mh || is_hist {
                std::fs::remove_file(self.folder.join(&f)).with_context(|| format!("deleting {f}"))?;
                n += is_mh as usize;
            }
        }
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// random-stream states (record.InitialSeeds / LastSeeds, mh files' LastSeeds.file<k>)
// ---------------------------------------------------------------------------

/// One generator state as Dynare stores it.
#[derive(Clone, Debug, PartialEq)]
pub struct RngState {
    pub unifor: Vec<f64>,
    pub normal: Vec<f64>,
    pub algo: String,
}

impl RngState {
    /// dynare-rust's ChaCha8 state: 8 seed words, the stream (2 words) and the word
    /// position (4 words), each a 32-bit integer.
    pub fn from_chacha(r: &ChaCha8Rng) -> Self {
        let seed = r.get_seed();
        let mut v: Vec<f64> = seed.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f64).collect();
        let s = r.get_stream();
        v.push((s & 0xffff_ffff) as f64);
        v.push((s >> 32) as f64);
        let w = r.get_word_pos();
        for k in 0..4 {
            v.push(((w >> (32 * k)) & 0xffff_ffff) as f64);
        }
        RngState { unifor: v.clone(), normal: v, algo: CHACHA_ALGO.to_string() }
    }

    /// Whether this is a dynare-rust state.
    pub fn is_chacha(&self) -> bool {
        self.algo == CHACHA_ALGO && self.unifor.len() == 14
    }

    /// The generator to continue from: the exact ChaCha8 state for dynare-rust's files;
    /// for MATLAB's (or missing) states, ChaCha8 seeded with a hash of the state and the
    /// chain number (reproducible, but not MATLAB's draws).
    pub fn to_chacha(&self, chain: usize) -> ChaCha8Rng {
        if self.is_chacha() {
            let u = |k: usize| self.unifor[k] as u32;
            let mut seed = [0u8; 32];
            for k in 0..8 {
                seed[4 * k..4 * k + 4].copy_from_slice(&u(k).to_le_bytes());
            }
            let mut r = ChaCha8Rng::from_seed(seed);
            r.set_stream(u(8) as u64 | ((u(9) as u64) << 32));
            let w = (0..4).fold(0u128, |a, k| a | ((u(10 + k) as u128) << (32 * k)));
            r.set_word_pos(w);
            return r;
        }
        // FNV-1a over the state's values
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in self.unifor.iter().chain(&self.normal) {
            for b in v.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        let mut r = ChaCha8Rng::seed_from_u64(h);
        r.set_stream(chain as u64 + 1);
        r
    }

    fn fields(&self) -> Vec<MatValue> {
        vec![MatValue::column(&self.unifor), MatValue::column(&self.normal), MatValue::Char(self.algo.clone())]
    }

    fn from_struct(v: &MatValue, k: usize) -> RngState {
        let num = |f: &str| v.field_of(k, f).and_then(MatValue::as_vec).unwrap_or_default();
        RngState {
            unifor: num("Unifor"),
            normal: num("Normal"),
            algo: v.field_of(k, "algo").and_then(MatValue::as_str).unwrap_or("").to_string(),
        }
    }
}

const SEED_FIELDS: [&str; 3] = ["Unifor", "Normal", "algo"];

fn seeds_value(s: &[RngState]) -> MatValue {
    MatValue::Struct {
        dims: vec![1, s.len()],
        fields: SEED_FIELDS.iter().map(|f| f.to_string()).collect(),
        elems: s.iter().map(RngState::fields).collect(),
    }
}

fn seeds_of(v: Option<&MatValue>) -> Vec<RngState> {
    match v {
        Some(v @ MatValue::Struct { .. }) => (0..v.struct_len()).map(|k| RngState::from_struct(v, k)).collect(),
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// the history record
// ---------------------------------------------------------------------------

/// `record.KeepedDraws` (CutSample.m).
#[derive(Clone, Debug, PartialEq)]
pub struct KeepedDraws {
    pub first_mh_file: usize,
    pub first_line: usize,
    pub distribution: Vec<f64>,
}

/// The `record` structure of `<fname>_mh_history_<i>.mat`.
#[derive(Clone, Debug, PartialEq)]
pub struct MhRecord {
    pub sampler: String,
    pub nblck: usize,
    /// One row per session: draws per chain, files per chain, lines in the last file.
    pub mh_draws: Vec<[f64; 3]>,
    /// `MAX_nruns` of each session (NaN when unknown).
    pub max_nruns: Vec<f64>,
    pub acceptance_ratio: Vec<f64>,
    pub function_eval_per_iteration: Vec<f64>,
    pub initial_seeds: Vec<RngState>,
    pub last_seeds: Vec<RngState>,
    /// `nblck x npar`.
    pub initial_parameters: Mat,
    pub initial_log_post: Vec<f64>,
    pub last_parameters: Mat,
    pub last_log_post: Vec<f64>,
    pub last_file_number: usize,
    pub last_line_number: usize,
    /// 0, 1 or NaN (unknown, old files).
    pub mcmc_concluded_successfully: f64,
    pub proposal_scale_vec: Option<Vec<f64>>,
    /// Upper Cholesky factor `d` of the proposal covariance.
    pub proposal_covariance: Option<Mat>,
    pub keeped_draws: Option<KeepedDraws>,
}

fn nums(v: Option<&MatValue>) -> Vec<f64> {
    v.and_then(MatValue::as_vec).unwrap_or_default()
}

fn num(v: Option<&MatValue>) -> Option<f64> {
    v.and_then(|x| x.as_vec()).and_then(|d| d.first().copied())
}

impl MhRecord {
    pub fn npar(&self) -> usize {
        self.last_parameters.ncols()
    }

    /// `sum(record.MhDraws(:,1))`.
    pub fn total_draws(&self) -> usize {
        self.mh_draws.iter().map(|r| r[0] as usize).sum()
    }

    /// `sum(record.MhDraws(:,2))`.
    pub fn total_files(&self) -> usize {
        self.mh_draws.iter().map(|r| r[1] as usize).sum()
    }

    pub fn to_mat(&self) -> MatValue {
        let k = self.mh_draws.len();
        let mut md = vec![0.0; 3 * k];
        for (i, r) in self.mh_draws.iter().enumerate() {
            for j in 0..3 {
                md[j * k + i] = r[j];
            }
        }
        let mut f: Vec<(&str, MatValue)> = vec![
            ("Sampler", MatValue::Char(self.sampler.clone())),
            ("Nblck", MatValue::scalar(self.nblck as f64)),
            ("MhDraws", MatValue::Numeric { dims: vec![k, 3], data: md }),
            ("MAX_nruns", MatValue::column(&self.max_nruns)),
            ("AcceptanceRatio", MatValue::row(&self.acceptance_ratio)),
            ("FunctionEvalPerIteration", MatValue::row(&self.function_eval_per_iteration)),
            ("InitialSeeds", seeds_value(&self.initial_seeds)),
            ("InitialParameters", MatValue::matrix(&self.initial_parameters)),
            ("InitialLogPost", MatValue::column(&self.initial_log_post)),
            ("LastParameters", MatValue::matrix(&self.last_parameters)),
            ("LastLogPost", MatValue::column(&self.last_log_post)),
            ("LastFileNumber", MatValue::scalar(self.last_file_number as f64)),
            ("LastLineNumber", MatValue::scalar(self.last_line_number as f64)),
            ("MCMCConcludedSuccessfully", MatValue::scalar(self.mcmc_concluded_successfully)),
        ];
        if let Some(s) = &self.proposal_scale_vec {
            f.push(("ProposalScaleVec", MatValue::column(s)));
        }
        if let Some(d) = &self.proposal_covariance {
            f.push(("ProposalCovariance", MatValue::matrix(d)));
        }
        if !self.last_seeds.is_empty() {
            f.push(("LastSeeds", seeds_value(&self.last_seeds)));
        }
        if let Some(kd) = &self.keeped_draws {
            f.push((
                "KeepedDraws",
                MatValue::structure(vec![
                    ("FirstMhFile", MatValue::scalar(kd.first_mh_file as f64)),
                    ("FirstLine", MatValue::scalar(kd.first_line as f64)),
                    ("Distribution", MatValue::column(&kd.distribution)),
                ]),
            ));
        }
        MatValue::structure(f)
    }

    pub fn from_mat(r: &MatValue) -> Result<Self> {
        let g = |k: &str| r.field(k);
        let mat = |k: &str| g(k).and_then(MatValue::as_matrix);
        let mhd = mat("MhDraws").ok_or_else(|| anyhow!("mh history record without MhDraws"))?;
        let mh_draws: Vec<[f64; 3]> = (0..mhd.nrows()).map(|i| [mhd[(i, 0)], mhd[(i, 1)], mhd[(i, 2)]]).collect();
        let nblck = num(g("Nblck")).ok_or_else(|| anyhow!("mh history record without Nblck"))? as usize;
        let last_parameters =
            mat("LastParameters").ok_or_else(|| anyhow!("mh history record without LastParameters"))?;
        let mut max_nruns = nums(g("MAX_nruns"));
        if max_nruns.len() < mh_draws.len() {
            max_nruns.resize(mh_draws.len(), f64::NAN);
        }
        let kd = g("KeepedDraws").and_then(|k| {
            Some(KeepedDraws {
                first_mh_file: num(k.field("FirstMhFile"))? as usize,
                first_line: num(k.field("FirstLine"))? as usize,
                distribution: nums(k.field("Distribution")),
            })
        });
        Ok(MhRecord {
            sampler: g("Sampler").and_then(MatValue::as_str).unwrap_or("random_walk_metropolis_hastings").to_string(),
            nblck,
            mh_draws,
            max_nruns,
            acceptance_ratio: nums(g("AcceptanceRatio")),
            function_eval_per_iteration: nums(g("FunctionEvalPerIteration")),
            initial_seeds: seeds_of(g("InitialSeeds")),
            last_seeds: seeds_of(g("LastSeeds")),
            initial_parameters: mat("InitialParameters").unwrap_or_else(|| Mat::zeros(0, 0)),
            initial_log_post: nums(g("InitialLogPost")),
            last_log_post: nums(g("LastLogPost")),
            last_parameters,
            last_file_number: num(g("LastFileNumber")).unwrap_or(0.0) as usize,
            last_line_number: num(g("LastLineNumber")).unwrap_or(0.0) as usize,
            mcmc_concluded_successfully: num(g("MCMCConcludedSuccessfully")).unwrap_or(f64::NAN),
            proposal_scale_vec: g("ProposalScaleVec").and_then(MatValue::as_vec),
            proposal_covariance: mat("ProposalCovariance").filter(|m| m.nrows() > 0),
            keeped_draws: kd,
        })
    }
}

/// `load_last_mh_history_file.m`: the record of the last session and its index.
pub fn load_last_record(paths: &MhPaths) -> Result<(MhRecord, usize)> {
    let n = paths.history_count();
    if n == 0 {
        bail!("Estimation::load_mh_file: I cannot find any mh-history file in {}!", paths.folder.display());
    }
    let p = paths.history(n - 1);
    let vars = matfile::read(&p)?;
    let r = vars.get("record").ok_or_else(|| anyhow!("{}: no variable record", p.display()))?;
    Ok((MhRecord::from_mat(r).with_context(|| format!("reading {}", p.display()))?, n - 1))
}

/// `load_first_mh_history_file.m`.
pub fn load_record(paths: &MhPaths, i: usize) -> Result<MhRecord> {
    let p = paths.history(i);
    let vars = matfile::read(&p)?;
    MhRecord::from_mat(vars.get("record").ok_or_else(|| anyhow!("{}: no variable record", p.display()))?)
}

/// `write_mh_history_file.m`: a new history file; returns its index.
pub fn write_new_record(paths: &MhPaths, rec: &MhRecord) -> Result<usize> {
    std::fs::create_dir_all(&paths.folder)?;
    let i = paths.history_count();
    save_record(&paths.history(i), rec)?;
    Ok(i)
}

/// `update_last_mh_history_file.m`.
pub fn update_last_record(paths: &MhPaths, rec: &MhRecord) -> Result<()> {
    let n = paths.history_count();
    if n == 0 {
        bail!("update_last_mh_history_file: I cannot find any mh-history file in {}!", paths.folder.display());
    }
    save_record(&paths.history(n - 1), rec)
}

fn save_record(p: &Path, rec: &MhRecord) -> Result<()> {
    matfile::write_values(p, &[("record", &rec.to_mat()), ("version", &MatValue::scalar(2.0))])
}

/// `compute_overall_acceptance_ratio.m` over all history files.
pub fn overall_acceptance_ratio(paths: &MhPaths) -> Result<Vec<f64>> {
    let n = paths.history_count();
    let mut draws = 0.0;
    let mut acc: Vec<f64> = vec![];
    for i in 0..n {
        let r = load_record(paths, i)?;
        let m = r.mh_draws.last().map_or(0.0, |x| x[0]);
        draws += m;
        if acc.is_empty() {
            acc = vec![0.0; r.acceptance_ratio.len()];
        }
        for (a, v) in acc.iter_mut().zip(&r.acceptance_ratio) {
            *a += v * m;
        }
    }
    Ok(acc.into_iter().map(|a| a / draws).collect())
}

// ---------------------------------------------------------------------------
// mh files
// ---------------------------------------------------------------------------

/// Contents of one `<fname>_mh<k>_blck<b>.mat`.
#[derive(Clone, Debug)]
pub struct MhFile {
    /// Row-major draws (`rows x npar`).
    pub x2: Vec<f64>,
    pub logpo2: Vec<f64>,
    pub npar: usize,
    /// `LastSeeds.file<k>` fields.
    pub last_seeds: Vec<(String, RngState)>,
    pub accepted_draws_this_chain: f64,
    pub accepted_draws_this_file: f64,
    pub feval_this_chain: f64,
    pub feval_this_file: f64,
}

impl MhFile {
    pub fn rows(&self) -> usize {
        self.logpo2.len()
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.x2[i * self.npar..(i + 1) * self.npar]
    }

    /// `LastSeeds.file<k>`.
    pub fn seed(&self, file: usize) -> Option<&RngState> {
        let key = format!("file{file}");
        self.last_seeds.iter().find(|(k, _)| *k == key).map(|(_, s)| s)
    }

    pub fn read(p: &Path) -> Result<Self> {
        Self::read_parts(p, true)
    }

    /// Reads a file; `with_draws = false` skips `x2` (kept empty).
    pub fn read_parts(p: &Path, with_draws: bool) -> Result<Self> {
        let vars = matfile::read(p)?;
        let logpo2 =
            vars.get("logpo2").and_then(MatValue::as_vec).ok_or_else(|| anyhow!("{}: no logpo2", p.display()))?;
        let (x2, npar) = if with_draws {
            let m = vars.get("x2").and_then(MatValue::as_matrix).ok_or_else(|| anyhow!("{}: no x2", p.display()))?;
            if m.nrows() != logpo2.len() {
                bail!("{}: x2 has {} rows and logpo2 {}", p.display(), m.nrows(), logpo2.len());
            }
            let t = m.transpose();
            (t.as_slice().to_vec(), m.ncols())
        } else {
            (vec![], 0)
        };
        let last_seeds = match vars.get("LastSeeds") {
            Some(MatValue::Struct { fields, .. }) => {
                let s = vars.get("LastSeeds").unwrap();
                fields
                    .iter()
                    .filter_map(|f| {
                        let sub = s.field(f)?;
                        Some((f.clone(), RngState::from_struct(sub, 0)))
                    })
                    .collect()
            }
            _ => vec![],
        };
        let sc = |k: &str| vars.get(k).and_then(MatValue::as_scalar).unwrap_or(f64::NAN);
        Ok(MhFile {
            x2,
            logpo2,
            npar,
            last_seeds,
            accepted_draws_this_chain: sc("accepted_draws_this_chain"),
            accepted_draws_this_file: sc("accepted_draws_this_file"),
            feval_this_chain: sc("feval_this_chain"),
            feval_this_file: sc("feval_this_file"),
        })
    }

    pub fn write(&self, p: &Path) -> Result<()> {
        let rows = self.rows();
        let mut col = vec![0.0; rows * self.npar];
        for i in 0..rows {
            for j in 0..self.npar {
                col[j * rows + i] = self.x2[i * self.npar + j];
            }
        }
        let x2 = MatValue::Numeric { dims: vec![rows, self.npar], data: col };
        let seeds = MatValue::Struct {
            dims: vec![1, 1],
            fields: self.last_seeds.iter().map(|(k, _)| k.clone()).collect(),
            elems: vec![
                self.last_seeds
                    .iter()
                    .map(|(_, s)| MatValue::Struct {
                        dims: vec![1, 1],
                        fields: SEED_FIELDS.iter().map(|f| f.to_string()).collect(),
                        elems: vec![s.fields()],
                    })
                    .collect(),
            ],
        };
        matfile::write_values(
            p,
            &[
                ("x2", &x2),
                ("logpo2", &MatValue::column(&self.logpo2)),
                ("LastSeeds", &seeds),
                ("accepted_draws_this_chain", &MatValue::scalar(self.accepted_draws_this_chain)),
                ("accepted_draws_this_file", &MatValue::scalar(self.accepted_draws_this_file)),
                ("feval_this_chain", &MatValue::scalar(self.feval_this_chain)),
                ("feval_this_file", &MatValue::scalar(self.feval_this_file)),
            ],
        )
    }
}

/// Where a chain of the current session starts writing: file `new_file`, line `first_line`
/// (1-based; `first_line > 1` appends to a file of the previous session), and how many
/// draws it runs.
#[derive(Clone, Copy, Debug)]
pub struct WriterStart {
    pub new_file: usize,
    pub first_line: usize,
    pub nruns: usize,
}

/// The file bookkeeping of one chain in `posterior_sampler_core.m`'s `computeMHBlock`:
/// draws are buffered and a file is saved when it is full or the chain is done, with the
/// generator state at that point in `LastSeeds.file<k>`. Owned by the chain's thread.
pub struct ChainWriter {
    paths: Arc<MhPaths>,
    blck: usize,
    max_nruns: usize,
    cur: MhFile,
    file: usize,
    /// `InitSizeArray`: rows of the current file when it is saved.
    target: usize,
    nruns: usize,
    done: usize,
    accepted_chain: f64,
    feval_chain: f64,
    pub(crate) error: Option<anyhow::Error>,
}

impl ChainWriter {
    /// `blck` is 1-based. Appending to a partially filled file (`first_line > 1`) loads it.
    pub fn new(paths: Arc<MhPaths>, blck: usize, npar: usize, max_nruns: usize, s: WriterStart) -> Result<Self> {
        let cur = if s.first_line > 1 {
            let p = paths.mh_file(s.new_file, blck);
            let f = MhFile::read(&p).with_context(|| format!("loading {} to append the new draws", p.display()))?;
            if f.rows() != s.first_line - 1 {
                bail!("{}: {} draws, expected {}", p.display(), f.rows(), s.first_line - 1);
            }
            f
        } else {
            MhFile {
                x2: vec![],
                logpo2: vec![],
                npar,
                last_seeds: vec![],
                accepted_draws_this_chain: 0.0,
                accepted_draws_this_file: 0.0,
                feval_this_chain: 0.0,
                feval_this_file: 0.0,
            }
        };
        let mut cur = cur;
        cur.accepted_draws_this_file = 0.0;
        cur.feval_this_file = 0.0;
        Ok(ChainWriter {
            paths,
            blck,
            max_nruns,
            target: max_nruns.min(s.first_line + s.nruns - 1),
            file: s.new_file,
            cur,
            nruns: s.nruns,
            done: 0,
            accepted_chain: 0.0,
            feval_chain: 0.0,
            error: None,
        })
    }

    /// Draws that can still be pushed before the current file must be saved.
    #[inline]
    pub fn until_flush(&self) -> usize {
        self.target.saturating_sub(self.cur.rows()).max(1)
    }

    #[inline]
    pub fn push(&mut self, x: &[f64], lp: f64, accepted: f64, nevals: f64) {
        self.cur.x2.extend_from_slice(x);
        self.cur.logpo2.push(lp);
        self.cur.accepted_draws_this_file += accepted;
        self.cur.feval_this_file += nevals;
        self.accepted_chain += accepted;
        self.feval_chain += nevals;
        self.done += 1;
    }

    /// After a batch of iterations: saves the current file when it is full or the chain is
    /// done (`rng` is the chain's generator after the last pushed draw).
    pub fn after(&mut self, rng: &ChaCha8Rng) {
        if self.error.is_some() || (self.cur.rows() < self.target && self.done < self.nruns) {
            return;
        }
        let key = format!("file{}", self.file);
        let st = RngState::from_chacha(rng);
        match self.cur.last_seeds.iter_mut().find(|(k, _)| *k == key) {
            Some(e) => e.1 = st,
            None => self.cur.last_seeds.push((key, st)),
        }
        self.cur.accepted_draws_this_chain = self.accepted_chain;
        self.cur.feval_this_chain = self.feval_chain;
        if let Err(e) = self.cur.write(&self.paths.mh_file(self.file, self.blck)) {
            self.error = Some(e);
            return;
        }
        self.log_file();
        self.cur.accepted_draws_this_file = 0.0;
        self.cur.feval_this_file = 0.0;
        let next = (self.nruns - self.done).min(self.max_nruns);
        if next > 0 {
            self.cur.x2.clear();
            self.cur.logpo2.clear();
            self.file += 1;
            self.target = next;
        }
    }

    fn log_file(&self) {
        use std::io::Write;
        let n = self.cur.rows().max(1) as f64;
        let line = format!(
            "\n% Mh{}Blck{}\n \n  Number of simulations.: {}\n  Acceptance ratio......: {}\n  Feval per iteration...: {}\n",
            self.file,
            self.blck,
            self.cur.rows(),
            self.cur.accepted_draws_this_file / n,
            self.cur.feval_this_file / n
        );
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(self.paths.log()) {
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// Accepted proposals and function evaluations of this session.
    pub fn totals(&self) -> (f64, f64, usize) {
        (self.accepted_chain, self.feval_chain, self.done)
    }

    /// Last file number written or being written.
    pub fn file(&self) -> usize {
        self.file
    }
}

/// How a chain of the sampler starts (continuations: `load_mh_file`, `mh_recover`,
/// `mh_initialize_from_previous_mcmc`).
#[derive(Clone, Debug)]
pub struct ChainStart {
    pub x: Vec<f64>,
    pub lp: f64,
    pub rng: ChaCha8Rng,
    pub nruns: usize,
}

/// End state of a chain (`record.LastParameters`, `LastLogPost`, `LastSeeds`, ratios).
#[derive(Clone, Debug)]
pub struct ChainEnd {
    pub x: Vec<f64>,
    pub lp: f64,
    pub seed: RngState,
    pub accepted: f64,
    pub fevals: f64,
    pub draws: usize,
}

/// Shared between [`crate::estimation::rw_metropolis`] / the samplers and the session
/// driver ([`crate::estimation::mh_session`]): per-chain starts, writers and end states.
pub struct SessionIo {
    /// Starting points (else the samplers draw them, `posterior_sampler_initialization.m`).
    pub starts: Option<Vec<ChainStart>>,
    /// The proposal's Cholesky factor `d` (`record.ProposalCovariance` on reload), else
    /// `chol(invhess)`.
    pub proposal_chol: Option<Mat>,
    /// Seconds per posterior evaluation measured during the initialization (pre-fetching).
    pub eval_secs: f64,
    writers: Mutex<Vec<Option<ChainWriter>>>,
    ends: Mutex<Vec<Option<ChainEnd>>>,
}

impl std::fmt::Debug for SessionIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionIo").field("chains", &self.starts.as_ref().map(Vec::len)).finish()
    }
}

impl SessionIo {
    pub fn new(
        starts: Option<Vec<ChainStart>>,
        proposal_chol: Option<Mat>,
        writers: Vec<Option<ChainWriter>>,
        eval_secs: f64,
    ) -> Self {
        let n = writers.len();
        SessionIo {
            starts,
            proposal_chol,
            eval_secs,
            writers: Mutex::new(writers),
            ends: Mutex::new((0..n).map(|_| None).collect()),
        }
    }

    /// Chain `b`'s (0-based) generator and number of draws (default: stream `b+1` of
    /// `seed`, `replic` draws).
    pub fn chain_rng(&self, b: usize, seed: u64, replic: usize) -> (ChaCha8Rng, usize) {
        match self.starts.as_ref().and_then(|s| s.get(b)) {
            Some(s) => (s.rng.clone(), s.nruns),
            None => (crate::estimation::mcmc::rng_stream(seed, b as u64 + 1), replic),
        }
    }

    /// The starting points, if given.
    pub fn start_points(&self) -> Option<(Vec<Vec<f64>>, Vec<f64>)> {
        self.starts.as_ref().map(|s| (s.iter().map(|c| c.x.clone()).collect(), s.iter().map(|c| c.lp).collect()))
    }

    pub fn take_writer(&self, b: usize) -> Option<ChainWriter> {
        self.writers.lock().unwrap().get_mut(b).and_then(Option::take)
    }

    /// Records the end of chain `b`; returns the writer's I/O error, if any.
    pub fn finish_chain(
        &self,
        b: usize,
        writer: Option<ChainWriter>,
        rng: &ChaCha8Rng,
        x: &[f64],
        lp: f64,
        accepted: f64,
        fevals: f64,
        draws: usize,
    ) {
        let err = writer.and_then(|w| w.error);
        if let Some(e) = err {
            eprintln!("ERROR: MCMC chain {}: could not write its mh file: {e:#}", b + 1);
        }
        let end = ChainEnd { x: x.to_vec(), lp, seed: RngState::from_chacha(rng), accepted, fevals, draws };
        if let Some(slot) = self.ends.lock().unwrap().get_mut(b) {
            *slot = Some(end);
        }
    }

    pub fn ends(&self) -> Vec<Option<ChainEnd>> {
        self.ends.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// reading the draws back (CutSample.m, GetAllPosteriorDraws.m)
// ---------------------------------------------------------------------------

/// `CutSample.m`: the first kept draw `floor(mh_drop*Total)+1` as (file, line), with the
/// files' `MAX_nruns` of the current options (`npar` from the record).
pub fn cut_sample(rec: &MhRecord, mh_drop: f64, max_bytes: f64) -> KeepedDraws {
    let npar = rec.npar();
    let total_files = rec.total_files();
    let total = rec.total_draws();
    let mx = max_nruns(npar, max_bytes);
    let first_draw = 1usize.max((mh_drop * total as f64).floor() as usize + 1);
    let first_mh_file = first_draw.div_ceil(mx);
    let first_line = first_draw - (first_mh_file - 1) * mx;
    let last3 = rec.mh_draws.last().map_or(0.0, |r| r[2]);
    let distribution = if total_files as i64 - 1 - (first_mh_file as i64 + 1) + 1 > 0 {
        let mut d = vec![(mx - first_line + 1) as f64];
        d.extend(std::iter::repeat_n(mx as f64, total_files - 1 - (first_mh_file + 1) + 1));
        d.push(last3);
        d
    } else if total_files == 2 && first_mh_file > 1 {
        vec![(mx - first_line + 1) as f64, last3]
    } else {
        vec![]
    };
    KeepedDraws { first_mh_file, first_line, distribution }
}

/// Reads draws `from..total` (0-based, per chain) of every chain from the mh files, in
/// file order (every file but the last of a chain holds `MAX_nruns` draws, whatever the
/// sessions were). Returns one [`crate::estimation::Chain`] per block with
/// `first_stored = from`.
pub fn read_chains(
    paths: &MhPaths,
    rec: &MhRecord,
    from: usize,
    accepted: &[f64],
) -> Result<Vec<crate::estimation::Chain>> {
    let npar = rec.npar();
    let total = rec.total_draws();
    let nfiles = rec.total_files();
    let read_block = |b: usize| -> Result<crate::estimation::Chain> {
        let mut draws = Vec::with_capacity(total.saturating_sub(from) * npar);
        let mut logpost = Vec::with_capacity(total.saturating_sub(from));
        let mut pos = 0usize; // index of the first draw of the current file
        let mut init = vec![f64::NAN; npar];
        let mut init_lp = f64::NAN;
        for k in 1..=nfiles {
            let p = paths.mh_file(k, b + 1);
            // Files entirely before `from` only need their size.
            let f = MhFile::read(&p).with_context(|| format!("reading {}", p.display()))?;
            if f.npar != npar {
                bail!("{}: {} parameters, the record has {npar}", p.display(), f.npar);
            }
            if k == 1 && f.rows() > 0 {
                init = f.row(0).to_vec();
                init_lp = f.logpo2[0];
            }
            let n = f.rows();
            let lo = from.saturating_sub(pos).min(n);
            for i in lo..n {
                draws.extend_from_slice(f.row(i));
                logpost.push(f.logpo2[i]);
            }
            pos += n;
        }
        if pos != total {
            bail!("chain {}: {} draws in the mh files, the history records {total}", b + 1, pos);
        }
        let acc = accepted.get(b).copied().unwrap_or(f64::NAN);
        Ok(crate::estimation::Chain {
            npar,
            draws,
            logpost,
            first_stored: from.min(total),
            total_draws: total,
            accepted: (acc * total as f64).round().max(0.0) as usize,
            init: if rec.initial_parameters.nrows().gt(&b) {
                rec.initial_parameters.row(b).iter().copied().collect()
            } else {
                init
            },
            init_logpost: rec.initial_log_post.get(b).copied().unwrap_or(init_lp),
        })
    };
    use rayon::prelude::*;
    (0..rec.nblck).into_par_iter().map(read_block).collect()
}

/// `compute_mh_covariance_matrix.m`: mean and covariance (`recursive_moments`, divided by
/// the number of draws) of the kept draws `record.KeepedDraws` of all chains, and the draw
/// with the highest posterior kernel in the files `FirstMhFile..` (all their lines).
pub fn compute_mh_covariance_matrix(paths: &MhPaths, rec: &MhRecord) -> Result<(Vec<f64>, Mat, Vec<f64>, f64)> {
    let kd = rec.keeped_draws.clone().ok_or_else(|| {
        anyhow!("compute_mh_covariance_matrix: the mh history has no KeepedDraws (run CutSample first)")
    })?;
    let n = rec.npar();
    let nfiles = rec.total_files();
    let mut kernel = f64::NEG_INFINITY;
    let mut mode = vec![f64::NAN; n];
    let mut mean = vec![0.0; n];
    let mut cov = Mat::zeros(n, n);
    let mut offset = 0usize;
    for b in 1..=rec.nblck {
        let mut first_line = kd.first_line;
        for k in kd.first_mh_file..=nfiles {
            let f = MhFile::read(&paths.mh_file(k, b))?;
            if let Some((idx, &v)) =
                f.logpo2.iter().enumerate().fold(None, |best: Option<(usize, &f64)>, (i, v)| match best {
                    Some((_, bv)) if bv >= v => best,
                    _ => Some((i, v)),
                })
                && v > kernel
            {
                kernel = v;
                mode = f.row(idx).to_vec();
            }
            // recursive_moments.m
            for t in (first_line - 1)..f.rows() {
                let x = f.row(t);
                let tt = (t - (first_line - 1) + 1 + offset) as f64;
                let m1: Vec<f64> = (0..n).map(|i| mean[i] + (x[i] - mean[i]) / tt).collect();
                for i in 0..n {
                    for j in 0..n {
                        let qq = m1[i] * m1[j];
                        cov[(i, j)] += ((x[i] * x[j] - qq - cov[(i, j)]) + (tt - 1.0) * (mean[i] * mean[j] - qq)) / tt;
                    }
                }
                mean = m1;
            }
            offset += f.rows() - (first_line - 1);
            first_line = 1;
        }
    }
    Ok((mean, cov, mode, kernel))
}

/// `<dname>/prior/definition.mat` with `bayestopt_` (`name`, `pshape`, `p1`..`p7`,
/// `jscale`), as `set_prior.m` saves it; read by `mh_initialize_from_previous_mcmc`.
pub fn write_prior_definition(
    dname: &Path,
    names: &[String],
    priors: &[crate::estimation::Prior],
    jscale: &[f64],
) -> Result<()> {
    let dir = dname.join("prior");
    std::fs::create_dir_all(&dir)?;
    let col =
        |f: &dyn Fn(&crate::estimation::Prior) -> f64| MatValue::column(&priors.iter().map(f).collect::<Vec<_>>());
    let b = MatValue::structure(vec![
        (
            "name",
            MatValue::Cell {
                dims: vec![names.len(), 1],
                items: names.iter().map(|s| MatValue::Char(s.clone())).collect(),
            },
        ),
        ("pshape", col(&|p| p.shape.code() as f64)),
        ("p1", col(&|p| p.mean)),
        ("p2", col(&|p| p.sd)),
        ("p3", col(&|p| p.p3)),
        ("p4", col(&|p| p.p4)),
        ("p5", col(&|p| p.p5)),
        ("p6", col(&|p| p.p6)),
        ("p7", col(&|p| p.p7)),
        ("jscale", MatValue::column(jscale)),
    ]);
    matfile::write_values(&dir.join("definition.mat"), &[("bayestopt_", &b)])
}

/// `bayestopt_.name` of a `prior/definition.mat`.
pub fn read_prior_names(p: &Path) -> Result<Vec<String>> {
    let vars = matfile::read(p)?;
    vars.get("bayestopt_")
        .and_then(|b| b.field("name"))
        .and_then(MatValue::as_strings)
        .ok_or_else(|| anyhow!("{}: no bayestopt_.name", p.display()))
}

/// Writes `<fname>_mh_mode.mat` (`compute_posterior_covariance_matrix.m`): `xparam1` the
/// MCMC mode, `hh = inv(covariance)`, `fval` the posterior kernel there.
pub fn write_mh_mode(
    out_dir: &Path,
    fname: &str,
    mode: &[f64],
    cov: &Mat,
    kernel: f64,
    names: &[String],
) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let p = out_dir.join(format!("{fname}_mh_mode.mat"));
    let hh = cov.clone().try_inverse().unwrap_or_else(|| Mat::from_element(cov.nrows(), cov.ncols(), f64::NAN));
    let vals: BTreeMap<&str, MatValue> = BTreeMap::from([
        ("xparam1", MatValue::column(mode)),
        ("hh", MatValue::matrix(&hh)),
        ("fval", MatValue::scalar(kernel)),
        (
            "parameter_names",
            MatValue::Cell {
                dims: vec![names.len(), 1],
                items: names.iter().map(|s| MatValue::Char(s.clone())).collect(),
            },
        ),
    ]);
    let order = ["xparam1", "hh", "fval", "parameter_names"];
    let vars: Vec<(&str, &MatValue)> = order.iter().map(|k| (*k, &vals[k])).collect();
    matfile::write_values(&p, &vars)?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn chacha_state_roundtrip() {
        let mut r = crate::estimation::mcmc::rng_stream(12345, 3);
        for _ in 0..17 {
            let _: f64 = r.r#gen();
        }
        let _: u32 = r.r#gen();
        let st = RngState::from_chacha(&r);
        let mut r2 = st.to_chacha(0);
        for _ in 0..100 {
            assert_eq!(r.r#gen::<u64>(), r2.r#gen::<u64>());
        }
    }

    #[test]
    fn cut_sample_matches_dynare() {
        let rec = MhRecord {
            sampler: "random_walk_metropolis_hastings".into(),
            nblck: 1,
            mh_draws: vec![[2500.0, 3.0, 500.0]],
            max_nruns: vec![1000.0],
            acceptance_ratio: vec![0.3],
            function_eval_per_iteration: vec![1.0],
            initial_seeds: vec![],
            last_seeds: vec![],
            initial_parameters: Mat::zeros(1, 2),
            initial_log_post: vec![0.0],
            last_parameters: Mat::zeros(1, 2),
            last_log_post: vec![0.0],
            last_file_number: 3,
            last_line_number: 500,
            mcmc_concluded_successfully: 1.0,
            proposal_scale_vec: None,
            proposal_covariance: None,
            keeped_draws: None,
        };
        // MAX_nruns = ceil(32000/4/8) = 1000; FirstDraw = 1251 -> file 2, line 251
        let kd = cut_sample(&rec, 0.5, 32000.0);
        assert_eq!((kd.first_mh_file, kd.first_line), (2, 251));
        assert!(kd.distribution.is_empty()); // CutSample.m leaves it unset for 3 files from file 2
    }
}
