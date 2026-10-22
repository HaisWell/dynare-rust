#!/bin/sh
# timing.sh -- the README speed-table protocol on sw07.mod (MATLAB + Dynare 7.2 vs dynare-rs).
#   1. one posterior evaluation at the mode, 1 thread: time_lik.m (MATLAB, best of 3 x 300)
#      and `cargo test --release --test kalman_perf` style timing (dynare-rs prints ms per
#      evaluation at "Initial value of the log posterior" in the STAGE 9 run);
#   2. full STAGE 9 run: wall time of `matlab -batch "dynare sw07 -DSTAGE=9"` / `dynare-rs
#      sw07.mod -DSTAGE=9 nograph`;
#   3. MCMC: the stage-3 estimation with mh_replic=10000 and no posterior objects, once with
#      mh_tune_jscale (sw07_timing_tune.mod) and once with the tuned mh_jscale
#      (sw07_timing_fixed.mod, JSCALE below).
# Run on a quiet machine: sh timing.sh [JSCALE]
MATLAB=${MATLAB:-/Applications/MATLAB_R2025b.app/bin/matlab}
RS=${RS:-../../target/release/dynare-rs}
JSCALE=${1:-0.3}
cd "$(dirname "$0")" || exit 1
DYNARE_ROOT=${DYNARE_ROOT:-/Applications/Dynare/${DYNARE_VERSION:-7.2}-arm64}
DYN="addpath('$DYNARE_ROOT/matlab');"
# the MCMC variants: stage-3 call without filtered_vars/smoother/forecast/state uncertainty
python3 - "$JSCALE" <<'EOF'
import re, sys
src = open('sw07.mod').read()
old = """           mh_replic=1000000, mh_nblocks=6,
           filtered_vars, smoother, forecast=12,
           first_obs=5, presample=8, diffuse_filter, smoothed_state_uncertainty)"""
new = """           mh_replic=10000, mh_nblocks=6,
           first_obs=5, presample=8, diffuse_filter)"""
assert old in src
t = src.replace(old, new)
open('sw07_timing_tune.mod', 'w').write(t)
assert '           mh_tune_jscale,\n' in t
open('sw07_timing_fixed.mod', 'w').write(t.replace('           mh_tune_jscale,\n', '           mh_jscale=%s,\n' % sys.argv[1]))
EOF
echo "== MATLAB one evaluation (1 thread, best of 3 x 300)"
"$MATLAB" -batch "run('time_lik.m')" 2>&1 | tail -1
echo "== STAGE 9 wall time"
/usr/bin/time -p "$MATLAB" -batch "${DYN} dynare sw07 -DSTAGE=9 nointeractive" > timing_stage9_matlab.log 2>&1; tail -3 timing_stage9_matlab.log
/usr/bin/time -p "$RS" sw07.mod -DSTAGE=9 nograph > timing_stage9_rs.log 2>&1; grep 'per evaluation' timing_stage9_rs.log; tail -3 timing_stage9_rs.log
echo "== MCMC with mh_tune_jscale"
/usr/bin/time -p "$MATLAB" -batch "${DYN} dynare sw07_timing_tune -DSTAGE=3 nointeractive" > timing_tune_matlab.log 2>&1; grep -i 'jscale\|real' timing_tune_matlab.log | tail -4
/usr/bin/time -p "$RS" sw07_timing_tune.mod -DSTAGE=3 nograph > timing_tune_rs.log 2>&1; grep -i 'jscale\|real\|mcmc: done' timing_tune_rs.log | tail -4
echo "== MCMC 6 x 10,000 with mh_jscale=$JSCALE"
/usr/bin/time -p "$MATLAB" -batch "${DYN} dynare sw07_timing_fixed -DSTAGE=3 nointeractive" > timing_fixed_matlab.log 2>&1; grep -i 'real\|acceptance' timing_fixed_matlab.log | tail -3
/usr/bin/time -p "$RS" sw07_timing_fixed.mod -DSTAGE=3 nograph > timing_fixed_rs.log 2>&1; grep -i 'real\|mcmc: done' timing_fixed_rs.log | tail -3
