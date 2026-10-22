"""Builds ref_ms2.json and rts_small.out from the files Dynare's ms_irf / ms_variance_decomposition / ms_forecast /
ms_compute_probabilities(real_time_smoothed) wrote on ms_small.mod (run as m2.mod: ms_small2.mod plus the p* and rts
commands below; each command's output directory is copied to keep/cN, keep/pN by a verbatim copyfile block; the p*
runs use 19 / 3 percentiles).

Usage:
  python3 gen_ref2.py          runs MATLAB + Dynare $DYNARE_VERSION (default 7.2; $DYNARE_ROOT, default /Applications/Dynare/<v>-arm64) in a
                               temporary directory, then writes ref_ms2.json and rts_small.out next to this script
  python3 gen_ref2.py WORKDIR  only parses an existing run (WORKDIR holds keep/ and the rts file)

Reproducibility: ms_small's mode search is deterministic (Dynare 7.1 and 7.2 give the same mode to the last printed
digit), so the per-regime IRFs / variance decompositions, the filtered and ergodic IRF percentiles and the real-time
smoothed probabilities, which the tests compare at Dynare's mode, are the same in every run. The regime-mixture
variance decompositions and the forecasts are simulations whose draws are not seeded by the mex (two runs differ);
the tests compare them statistically."""
import json, sys, glob, os, shutil, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = os.environ.get('DYNARE_VERSION', '7.2')
DYNARE_ROOT = os.environ.get('DYNARE_ROOT', f'/Applications/Dynare/{VERSION}-arm64')
MATLAB = os.environ.get('MATLAB', '/Applications/MATLAB_R2025b.app/bin/matlab')

PCT19 = ' '.join(f'{0.05 * i:.2f}' for i in range(1, 20))
EXTRA = f"""
ms_irf(horizon=3, filtered_probabilities, error_band_percentiles=[{PCT19}]);
verbatim;
copyfile('m2/IRF','keep/p1');
end;
ms_irf(horizon=3, error_band_percentiles=[{PCT19}]);
verbatim;
copyfile('m2/IRF','keep/p2');
end;
ms_forecast(regime=1, horizon=4, error_band_percentiles=[0.1 0.5 0.9]);
verbatim;
copyfile('m2/Forecast','keep/p3');
end;
ms_forecast(horizon=4, error_band_percentiles=[0.1 0.5 0.9]);
verbatim;
copyfile('m2/Forecast','keep/p4');
end;
ms_forecast(regime=2, horizon=4, error_band_percentiles=[0.1 0.5 0.9]);
verbatim;
copyfile('m2/Forecast','keep/p5');
end;
ms_variance_decomposition(horizon=6, filtered_probabilities, error_band_percentiles=[0.16 0.5 0.84]);
verbatim;
copyfile('m2/Variance_Decomposition','keep/p6');
end;
ms_compute_probabilities(real_time_smoothed);
"""


def run_dynare():
    work = tempfile.mkdtemp(prefix='ms2_', dir=os.environ.get('GEN_WORKDIR'))
    shutil.copy(f'{HERE}/d2.csv', work)
    with open(f'{work}/m2.mod', 'w') as f:
        f.write(open(f'{HERE}/ms_small2.mod').read() + EXTRA)
    os.makedirs(f'{work}/keep')
    dyn = f'{DYNARE_ROOT}/matlab'
    subprocess.run([MATLAB, '-sd', work, '-batch', f"addpath('{dyn}'); dynare m2 nolog nograph"], check=True)
    return work


keep_root = sys.argv[1] if len(sys.argv) > 1 else run_dynare()
keep = f'{keep_root}/keep'


def rd(pat):
    f = glob.glob(f'{keep}/{pat}')[0]
    return [[float(x) for x in l.split()] for l in open(f) if l.strip()]


out = {}
for k, pat in {
    'irf_r0': 'c1/ir_percentiles_regime_0_*', 'irf_r1': 'c1/ir_percentiles_regime_1_*',
    'irf_filt3': 'c2/ir_percentiles_filtered_*', 'irf_erg_med': 'c3/ir_percentiles_ergodic_*',
    'vd_r0': 'c5/var_decomp_mean_regime_0_*', 'vd_r1': 'c5/var_decomp_mean_regime_1_*',
    'vd_erg': 'c6/var_decomp_mean_ergodic_*', 'vd_filt': 'c7/var_decomp_mean_filtered_*',
    'fc_r0': 'c8/forecasts_percentiles_regime_0_*', 'fc_r1': 'c8/forecasts_percentiles_regime_1_*',
    'fc_all_med': 'c9/forecasts_percentiles_*',
    'irf_filt19': 'p1/ir_percentiles_filtered_*', 'irf_erg19': 'p2/ir_percentiles_ergodic_*',
    'fc_reg1_3': 'p3/forecasts_percentiles_regime_0_*', 'fc_all_3': 'p4/forecasts_percentiles_*',
    'fc_reg2_3': 'p5/forecasts_percentiles_regime_1_*', 'vd_filt_pct': 'p6/var_decomp_percentiles_filtered_*',
}.items():
    try:
        out[k] = rd(pat)
    except IndexError:
        print('missing', k)
json.dump(out, open(f'{HERE}/ref_ms2.json', 'w'))
rts = glob.glob(f'{keep_root}/real_time_smoothed_*.out') + glob.glob(f'{keep_root}/smoothed_*.out')
if rts:
    shutil.copy(rts[0], f'{HERE}/rts_small.out')
else:
    print('missing real-time smoothed probabilities file')
