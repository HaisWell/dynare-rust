"""Regenerates this directory's Dynare outputs: go.m (msd.mod: ms_estimation + ms_simulation of 5000 draws +
ms_compute_mdd; then the mdd of proposal types 1-6 with -print, which writes mdd_t<pt>_msd.out and the
posterior_t<pt>/proposal_t<pt> density files) and seeds.m (Muller / bridge of 20 seeded runs per type on the same
draws, dynare_seeds.txt).

Usage: python3 gen_mdd.py   (MATLAB + Dynare $DYNARE_VERSION, default 7.2, $DYNARE_ROOT, default /Applications/Dynare/<v>-arm64; the work
directory is a temporary directory, under $GEN_WORKDIR if set)

Reproducibility: the mode of msd.mod is deterministic, and `ms_sbvar_command_line -mdd -seed N` with N > 0 is
reproducible (7.1 and 7.2 give identical dynare_seeds.txt rows on the same simulation file). The ms_simulation draws
are not (two runs of the same .mod differ), and go.m uses -seed 0 (time seeded), so a regeneration gives a new
chain and new proposal draws. The tests stay meaningful: they check our proposal against Dynare's own draws and
densities in these files (deterministic), and compare mdd means over seeds statistically."""
import glob, gzip, os, shutil, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = os.environ.get('DYNARE_VERSION', '7.2')
DYNARE_ROOT = os.environ.get('DYNARE_ROOT', f'/Applications/Dynare/{VERSION}-arm64')
MATLAB = os.environ.get('MATLAB', '/Applications/MATLAB_R2025b.app/bin/matlab')

work = tempfile.mkdtemp(prefix='mdd_', dir=os.environ.get('GEN_WORKDIR'))
for f in ['msd.mod', 'go.m', 'seeds.m']:
    shutil.copy(f'{HERE}/{f}', work)
shutil.copy(f'{HERE}/../d2.csv', work)
# seeds.m rewrites mdd_t<pt>_msd.out and proposal_t<pt>_msd.out: keep go.m's files aside meanwhile
script = (f"addpath('{DYNARE_ROOT}/matlab'); go; mkdir('keep');"
          " copyfile('mdd_t*_msd.out', 'keep'); copyfile('proposal_t*_msd.out', 'keep');"
          " copyfile('posterior_t*_msd.out', 'keep'); seeds; copyfile('keep/*', '.');")
subprocess.run([MATLAB, '-sd', work, '-batch', script], check=True)
for f in glob.glob(f'{work}/mdd_t*_msd.out'):
    shutil.copy(f, HERE)
shutil.copy(f'{work}/seeds.txt', f'{HERE}/dynare_seeds.txt')
for f in glob.glob(f'{work}/posterior_t*_msd.out') + glob.glob(f'{work}/proposal_t*_msd.out') + [f'{work}/simulation_msd.out']:
    with open(f, 'rb') as src, open(f'{HERE}/{os.path.basename(f)}.gz', 'wb') as raw, \
            gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as dst:
        shutil.copyfileobj(src, dst)
