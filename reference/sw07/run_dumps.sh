#!/bin/sh
# run_dumps.sh -- regenerate every reference/sw07/ref*.json with MATLAB + Dynare (default 7.2),
# in dependency order: ref.json first (the points of dump_solver/dump_kalman), ref_mode_check
# before its Schur follow-up, points_wide before point 15. Logs go to <script>.log.
#   sh run_dumps.sh                 # all
#   sh run_dumps.sh dump_kalman     # some
MATLAB=${MATLAB:-/Applications/MATLAB_R2025b.app/bin/matlab}
cd "$(dirname "$0")" || exit 1
ALL="dump_reference dump_prior_trunc dump_solver dump_kalman dump_stoch_simul dump_forecast dump_shock_decomp \
dump_mode_check dump_mode_check_schur dump_points_wide dump_point15 dump_posterior dump_conditional"
for s in ${*:-$ALL}; do
    echo "== $s"
    "$MATLAB" -batch "run('$s.m')" > "$s.log" 2>&1
    echo "exit $?"
    tail -n 2 "$s.log"
done
