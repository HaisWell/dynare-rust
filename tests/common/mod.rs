//! Shared test helpers. Reference files under `reference/` are committed; a
//! missing or corrupt one is a test failure (never a silent skip). Tests whose
//! reference needs a local MATLAB run and is not committed are `#[ignore]`d.
#![allow(dead_code)]

use std::path::Path;

/// Read a reference file, panicking with its path if it is missing.
pub fn read_ref(p: impl AsRef<Path>) -> String {
    let p = p.as_ref();
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("missing reference {}: {e}", p.display()))
}

/// Read and parse a JSON reference file, panicking with its path if it is missing or corrupt.
pub fn json_ref<T: serde::de::DeserializeOwned>(p: impl AsRef<Path>) -> T {
    let p = p.as_ref();
    serde_json::from_str(&read_ref(p)).unwrap_or_else(|e| panic!("corrupt reference {}: {e}", p.display()))
}

/// A fresh scratch directory unique to this process and `tag`, so tests running
/// in parallel threads of one test binary never share a directory.
pub fn temp_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("dynare_rs_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("create {}: {e}", d.display()));
    d
}
