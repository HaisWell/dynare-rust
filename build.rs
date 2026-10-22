// SPDX-License-Identifier: GPL-3.0-or-later
fn main() {
    // LAPACK/BLAS: Apple's Accelerate framework on macOS (AMX-backed on Apple silicon), the
    // reference platform of the numerical tests. Elsewhere a system LAPACK/BLAS that exports the
    // Fortran `d*_` symbols and `cblas_dnrm2`: OpenBLAS by default (Debian/Ubuntu:
    // libopenblas-dev), or the libraries named in DYNARE_RS_LAPACK (comma-separated, e.g.
    // "lapack,blas").
    println!("cargo:rerun-if-env-changed=DYNARE_RS_LAPACK");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else {
        let libs = std::env::var("DYNARE_RS_LAPACK").unwrap_or_else(|_| "openblas".into());
        for lib in libs.split(',').map(str::trim).filter(|l| !l.is_empty()) {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}
