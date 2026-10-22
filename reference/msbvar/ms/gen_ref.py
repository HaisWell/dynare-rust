"""Builds ref_ms.json (and sim_small.out) from the files Dynare's ms_estimation / ms_compute_probabilities wrote when
run on ms_small.mod, ms_chains.mod, ms_three.mod (as msv, msb, msc), and ms_simulation + ms_compute_mdd on ms_small
(mdd/msd.mod with mh_replic=20000, as msm).

Usage:
  python3 gen_ref.py               runs MATLAB + Dynare $DYNARE_VERSION (default 7.2; $DYNARE_ROOT, default /Applications/Dynare/<v>-arm64)
                                   in a temporary directory ($GEN_WORKDIR if set), then writes the references
  python3 gen_ref.py DIR_SMALL DIR_CHAINS DIR_THREE DIR_MDD   only parses existing Dynare working directories

Reproducibility: ms_small's mode search is deterministic (7.1 and 7.2 give the same mode, likelihoods, lnp and
probabilities to the last printed digit). The mode searches of ms_chains / ms_three and the Metropolis draws of
ms_simulation are not: the mex does not seed them reproducibly (two runs of the same .mod differ), so a
regeneration gives another local mode / another chain. The tests stay meaningful because they evaluate our
likelihood, probabilities and posterior at whatever mode or draws Dynare wrote ("at mode" checks are exact up to
printing precision), and compare our own mode search and mdd estimates with a tolerance."""
import json, os, re, shutil, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = os.environ.get('DYNARE_VERSION', '7.2')
DYNARE_ROOT = os.environ.get('DYNARE_ROOT', f'/Applications/Dynare/{VERSION}-arm64')
MATLAB = os.environ.get('MATLAB', '/Applications/MATLAB_R2025b.app/bin/matlab')


def run_dynare():
    work = tempfile.mkdtemp(prefix='ms_', dir=os.environ.get('GEN_WORKDIR'))
    small = open(f'{HERE}/ms_small.mod').read()
    # ms_small calls ms_compute_probabilities twice; the second call deletes filtered_msv.out, so keep a copy
    call = 'ms_compute_probabilities(filtered_probabilities);\n'
    assert call in small
    small = small.replace(call, call + "verbatim;\ncopyfile('filtered_msv.out','filtered_keep.out');\nend;\n")
    mdd = open(f'{HERE}/mdd/msd.mod').read().replace('mh_replic=5000', 'mh_replic=20000')
    mods = {'small': ('msv', small), 'chains': ('msb', open(f'{HERE}/ms_chains.mod').read()),
            'three': ('msc', open(f'{HERE}/ms_three.mod').read()), 'mdd': ('msm', mdd)}
    script = f"addpath('{DYNARE_ROOT}/matlab'); root = pwd;"
    for d, (name, text) in mods.items():
        os.makedirs(f'{work}/{d}')
        shutil.copy(f'{HERE}/d2.csv', f'{work}/{d}')
        open(f'{work}/{d}/{name}.mod', 'w').write(text)
        script += f" cd(fullfile(root, '{d}')); dynare {name} nolog nograph; cd(root); clearvars -except root;"
    subprocess.run([MATLAB, '-sd', work, '-batch', script], check=True)
    shutil.move(f'{work}/small/filtered_keep.out', f'{work}/small/filtered_msv.out')
    return [f'{work}/{d}' for d in mods]


def load(d, tag):
    r = {}
    free = np.loadtxt(f'{d}/est_free_{tag}.out')
    r['log_posterior'], r['log_likelihood'], r['theta'] = free[0], free[1], free[2:].tolist()
    f = open(f'{d}/est_final_{tag}.out').read()
    for key, name in [('Initial likelihood Value', 'init_lik'), ('Initial prior Value', 'init_prior'),
                      ('Final likelihood Value', 'final_lik'), ('Final prior Value', 'final_prior')]:
        r[name] = float(re.search(key + r': (\S+)', f).group(1))
    try:
        r['smoothed'] = np.loadtxt(f'{d}/smoothed_{tag}.out').tolist()
    except OSError:
        pass
    try:
        r['filtered'] = np.loadtxt(f'{d}/filtered_{tag}.out').tolist()
    except OSError:
        pass
    aux = np.loadtxt(f'{d}/est_aux_{tag}.out', delimiter=',', skiprows=1, usecols=(0,))
    r['lnp'] = aux.tolist()
    L = open(f'{d}/init_{tag}.dat').read().split('\n')
    def blocks(prefix):
        out = []
        for k, l in enumerate(L):
            if l.startswith('//== ' + prefix + '['):
                rows = []
                i = k + 1
                while L[i].strip():
                    rows.append([float(x) for x in L[i].split()])
                    i += 1
                out.append(rows)
        return out
    r['U'], r['V'] = blocks('U'), blocks('V')
    return r


d_small, d_chains, d_three, d_mdd = sys.argv[1:5] if len(sys.argv) > 4 else run_dynare()
out = {'small': load(d_small, 'msv'), 'chains': load(d_chains, 'msb'), 'three': load(d_three, 'msc')}
m = load(d_mdd, 'msm')
txt = open(f'{d_mdd}/mdd_t3_msm.out').read()
m['muller'] = float(re.search(r'Muller log\(mdd\) = (\S+)', txt).group(1))
m['bridge'] = float(re.search(r'Bridge log\(mdd\) = (\S+)', txt).group(1))
wz = re.findall(r'log\(mdd\) = (\S+)\s+log\(cut\)', txt)
m['wz_first'] = float(wz[0])
del m['lnp']
out['mdd'] = m
json.dump(out, open(f'{HERE}/ref_ms.json', 'w'))
sim = np.loadtxt(f'{d_mdd}/simulation_msm.out')[::5]
np.savetxt(f'{HERE}/sim_small.out', sim, fmt='%.7e')
