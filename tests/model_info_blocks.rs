//! model_info: the preprocessor's block decomposition against Dynare 7.2.
//!
//! * `M_.block_structure` / `M_.block_structure_stat` of every .mod file under `reference/`
//!   against the 7.1 preprocessor's driver output (`reference/blocks/oracle/<dir>__<name>.json`,
//!   written by `python3 reference/blocks/extract_blocks.py --out reference/blocks/oracle ...`):
//!   identical blocks, types, orderings, feedback sets, `is_linear` and incidence;
//! * the text of `model_info(block_static|block_dynamic[, incidence])` against MATLAB
//!   (`reference/blocks/model_info/<name>.txt`, Dynare 7.1 under MATLAB R2025b), including the
//!   SW07, QPM and GM05 benchmark models.

use std::path::{Path, PathBuf};

use dynare::misc::model_info::{dynamic_block_structure, static_block_structure};
use dynare::modfile::{Command, ModFile, parse_mod_file, parse_mod_str};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference")
}

fn mod_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            mod_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "mod") {
            out.push(p);
        }
    }
}

#[test]
fn block_structures_match_the_preprocessor() {
    let mut mods = Vec::new();
    mod_files(&root(), &mut mods);
    mods.sort();
    let (mut n, mut bad) = (0, Vec::new());
    for p in &mods {
        let parent = p.parent().unwrap().file_name().unwrap().to_string_lossy();
        let stem = p.file_stem().unwrap().to_string_lossy();
        let oracle = root().join("blocks/oracle").join(format!("{parent}__{stem}.json"));
        let Ok(text) = std::fs::read_to_string(&oracle) else { continue };
        let want: Value = serde_json::from_str(&text).unwrap();
        let mf = parse_mod_file(p, &[]).unwrap_or_else(|e| panic!("{}: {e:#}", p.display()));
        let got = serde_json::json!({
            "static": static_block_structure(&mf).map(|b| b.to_json(false)),
            "dynamic": dynamic_block_structure(&mf).map(|b| b.to_json(true)),
        });
        n += 1;
        if got != want {
            bad.push(p.display().to_string());
        }
    }
    assert!(n >= 90, "only {n} models compared");
    assert!(bad.is_empty(), "block structures differ from the preprocessor's: {bad:?}");
}

const CMDS: &str = "\nmodel_info(block_static);\nmodel_info(block_static, incidence);\nmodel_info(block_dynamic);\nmodel_info(block_dynamic, incidence);\nstoch_simul(order=1, irf=0, noprint, nomoments, nocorr, nofunctions, nograph);\n";

const END: &str = "1: non-null element, X: non-null element related to a state variable";

/// The `model_info` outputs of the file, from the first header to the last legend line (as
/// the reference files are cut from the MATLAB log).
fn model_info_output(mf: &ModFile, fname: &str) -> String {
    let mut out = String::new();
    for c in &mf.commands {
        if let Command::Other(name, o) = c
            && name == "model_info"
        {
            out += &dynare::misc::model_info_text(mf, o, fname).unwrap();
        }
    }
    let lines: Vec<&str> = out.split('\n').collect();
    let a = lines.iter().position(|l| l.starts_with("Information about")).unwrap();
    let b = lines.iter().rposition(|l| *l == END).unwrap();
    lines[a..=b].join("\n") + "\n"
}

fn check_text(mf: &ModFile, name: &str) {
    let want = std::fs::read_to_string(root().join("blocks/model_info").join(format!("{name}.txt"))).unwrap();
    let got = model_info_output(mf, name);
    if got != want {
        let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
        let k = g.iter().zip(&w).position(|(a, b)| a != b).unwrap_or(g.len().min(w.len()));
        panic!(
            "{name}: model_info output differs at line {}:\n  ours:   {:?}\n  matlab: {:?}",
            k + 1,
            g.get(k),
            w.get(k)
        );
    }
}

#[test]
fn model_info_text_matches_matlab() {
    let dir = root().join("blocks/model_info");
    let mut n = 0;
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let p = e.path();
        if p.extension().is_none_or(|x| x != "mod") {
            continue;
        }
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        let mf = parse_mod_file(&p, &[]).unwrap();
        check_text(&mf, &name);
        n += 1;
    }
    assert!(n >= 15);
}

/// SW07, QPM and GM05: the model up to `varobs`, then the `model_info` commands and a
/// `stoch_simul` (which sets the normalization cutoff to 0, as `estimation` does).
#[test]
fn model_info_text_matches_matlab_semi_structural() {
    for (name, path) in
        [("mi_sw07", "sw07/sw07.mod"), ("mi_qpm", "qpm/model_qpm.mod"), ("mi_gm05", "gm05/model_gm05.mod")]
    {
        let src = std::fs::read_to_string(root().join(path)).unwrap();
        let cut = src.find("\nvarobs").unwrap() + 1;
        let text = format!("{}\n{CMDS}", &src[..cut]);
        let mf = parse_mod_str(&text, &[]).unwrap();
        check_text(&mf, name);
    }
}
