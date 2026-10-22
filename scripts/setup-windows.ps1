# SPDX-License-Identifier: GPL-3.0-or-later
# Windows setup for dynare-rust: downloads the prebuilt OpenBLAS (LP64, 32-bit integers like
# LAPACK's reference interface) and points the build at it.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1 [-Prefix DIR] [-Version X.Y.Z]
#
# Requirements not handled here: Rust (https://rustup.rs, x86_64-pc-windows-msvc toolchain) and
# the Visual Studio C++ Build Tools (MSVC linker).
#
# Sets, for the current user: DYNARE_RS_LAPACK=libopenblas, DYNARE_RS_LAPACK_DIR=<prefix>\lib
# (read by build.rs), and adds <prefix>\bin to PATH so dynare-rs.exe finds libopenblas.dll.
param(
    [string]$Prefix = "$env:LOCALAPPDATA\OpenBLAS",
    [string]$Version = "0.3.34"
)
$ErrorActionPreference = "Stop"

$dll = Join-Path $Prefix "bin\libopenblas.dll"
if (Test-Path $dll) {
    Write-Host "OpenBLAS already present in $Prefix"
} else {
    $url = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$Version/OpenBLAS-$Version-x64.zip"
    $zip = Join-Path $env:TEMP "OpenBLAS-$Version-x64.zip"
    Write-Host "Downloading $url"
    Invoke-WebRequest $url -OutFile $zip
    New-Item -ItemType Directory -Force $Prefix | Out-Null
    Expand-Archive $zip -DestinationPath $Prefix -Force
    Remove-Item $zip
}

$lib = Join-Path $Prefix "lib"
$bin = Join-Path $Prefix "bin"
[Environment]::SetEnvironmentVariable("DYNARE_RS_LAPACK", "libopenblas", "User")
[Environment]::SetEnvironmentVariable("DYNARE_RS_LAPACK_DIR", $lib, "User")
$path = [Environment]::GetEnvironmentVariable("Path", "User")
if (($path -split ";") -notcontains $bin) {
    [Environment]::SetEnvironmentVariable("Path", ($path.TrimEnd(";") + ";" + $bin), "User")
}
$env:DYNARE_RS_LAPACK = "libopenblas"
$env:DYNARE_RS_LAPACK_DIR = $lib
if (($env:Path -split ";") -notcontains $bin) { $env:Path = "$env:Path;$bin" }

# CI: export the same settings to later steps.
if ($env:GITHUB_ENV) {
    # AppendAllText writes no BOM (Out-File -Encoding utf8 does in Windows PowerShell 5.1).
    [IO.File]::AppendAllText($env:GITHUB_ENV, "DYNARE_RS_LAPACK=libopenblas`nDYNARE_RS_LAPACK_DIR=$lib`n")
    [IO.File]::AppendAllText($env:GITHUB_PATH, "$bin`n")
}

Write-Host "OpenBLAS $Version ready: DYNARE_RS_LAPACK_DIR=$lib, $bin on PATH."
Write-Host "Open a new terminal, then: cargo build --release"
