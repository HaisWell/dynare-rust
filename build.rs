// SPDX-License-Identifier: GPL-3.0-or-later
fn main() {
    // LAPACK/BLAS: Apple's Accelerate framework on macOS (AMX-backed on Apple silicon), the
    // reference platform of the numerical tests. Elsewhere a system LAPACK/BLAS that exports the
    // Fortran `d*_` symbols and `cblas_dnrm2`: OpenBLAS by default (Debian/Ubuntu:
    // libopenblas-dev; Windows: the prebuilt OpenBLAS-x64 release, whose import library is
    // `libopenblas.lib`, see scripts/setup-windows.ps1), or the libraries named in
    // DYNARE_RS_LAPACK (comma-separated, e.g. "lapack,blas"). DYNARE_RS_LAPACK_DIR adds a
    // directory to the library search path.
    println!("cargo:rerun-if-env-changed=DYNARE_RS_LAPACK");
    println!("cargo:rerun-if-env-changed=DYNARE_RS_LAPACK_DIR");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else {
        if let Ok(dir) = std::env::var("DYNARE_RS_LAPACK_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        }
        let default = if target_os == "windows" { "libopenblas" } else { "openblas" };
        let libs = std::env::var("DYNARE_RS_LAPACK").unwrap_or_else(|_| default.into());
        for lib in libs.split(',').map(str::trim).filter(|l| !l.is_empty()) {
            println!("cargo:rustc-link-lib={lib}");
        }
        // MSVC: dynare-rs loads libopenblas.dll at the first BLAS call, not at process start,
        // so that `--openblas-threads` can restart it with OPENBLAS_NUM_THREADS set before
        // OpenBLAS ever commits its threads' buffers (see main.rs). Only that binary: the
        // linker warns about a delay-load in one that imports nothing from the DLL.
        let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
        if target_os == "windows" && msvc && libs.split(',').any(|l| l.trim() == "libopenblas") {
            println!("cargo:rustc-link-arg-bin=dynare-rs=/DELAYLOAD:libopenblas.dll");
            println!("cargo:rustc-link-arg-bin=dynare-rs=delayimp.lib");
        }
    }
}
