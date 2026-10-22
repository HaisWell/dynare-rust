"""Builds ref_est.json and copies the est_final_*.out / probability files Dynare wrote when run on the est_*.mod files
of this directory (`dynare est_none; dynare est_none_chains; dynare est_sz; dynare est_rf; dynare est_chains2;
dynare est_default` in one directory with d2.csv).

Usage:
  python3 gen_ref_est.py             runs MATLAB + Dynare $DYNARE_VERSION (default 7.2; $DYNARE_ROOT, default /Applications/Dynare/<v>-arm64)
                                     in a temporary directory ($GEN_WORKDIR if set), then writes the references here
  python3 gen_ref_est.py DYNARE_DIR  only parses an existing run

Reproducibility: these references cannot be regenerated bit for bit. The mode search (csminwel with its random
search, the block / perturbation stages) is not seeded reproducibly by the mex: two runs of the same .mod under
Dynare 7.2 end at modes up to 0.07 apart (only est_sz came out identical). The tests are built for that: the kernel
(initial values, model matrices, the posterior and probabilities at the mode Dynare wrote to est_final_*.out) is
compared tightly at Dynare's own mode, and our mode search is compared with the attained objective with a tolerance.
The checked-in files are one Dynare 7.2 run."""
import json, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = os.environ.get('DYNARE_VERSION', '7.2')
DYNARE_ROOT = os.environ.get('DYNARE_ROOT', f'/Applications/Dynare/{VERSION}-arm64')
MATLAB = os.environ.get('MATLAB', '/Applications/MATLAB_R2025b.app/bin/matlab')
MODS = ['est_none', 'est_none_chains', 'est_sz', 'est_rf', 'est_chains2', 'est_default']


def run_dynare():
    work = tempfile.mkdtemp(prefix='ms_est_', dir=os.environ.get('GEN_WORKDIR'))
    for f in [f'{m}.mod' for m in MODS] + ['d2.csv']:
        shutil.copy(f'{HERE}/{f}', work)
    script = f"addpath('{DYNARE_ROOT}/matlab');" + ''.join(
        f" dynare {m} nolog nograph; clearvars;" for m in MODS)
    subprocess.run([MATLAB, '-sd', work, '-batch', script], check=True)
    return work


d = sys.argv[1] if len(sys.argv) > 1 else run_dynare()
cases = {'none': 'none_out', 'none_chains': 'est_none_chains', 'sz': 'est_sz', 'rf': 'est_rf',
         'chains2': 'est_chains2', 'default': 'est_default'}
out = {}
for key, tag in cases.items():
    inter = open(f'{d}/est_intermediate_{tag}.out').read()
    fin = open(f'{d}/est_final_{tag}.out').read()
    r = {'tag': tag}
    r['init_lik'] = float(re.search(r'Likelihood value:\s+(\S+)', inter).group(1))
    r['init_prior'] = float(re.search(r'Prior value:\s+(\S+)', inter).group(1))
    objs = re.findall(r'Objective function \(run \d+\): (\S+)', fin)
    best = re.findall(r'maximum objective = (\S+)', fin)
    r['objective'] = float(best[-1]) if best else float(objs[-1])
    r['free'] = [float(x) for x in open(f'{d}/est_free_{tag}.out').read().split()]
    out[key] = r
    shutil.copy(f'{d}/est_final_{tag}.out', f'{HERE}/est_final_{tag}.out')
shutil.copy(f'{d}/smoothed_none_out.out', f'{HERE}/smoothed_none_out.out')
shutil.copy(f'{d}/filtered_est_none_chains.out', f'{HERE}/filtered_est_none_chains.out')
json.dump(out, open(f'{HERE}/ref_est.json', 'w'), indent=1)
