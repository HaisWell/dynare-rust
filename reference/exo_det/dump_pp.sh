#!/bin/bash
# dump_pp.sh -- the Dynare preprocessor's (DYNARE_VERSION env, default 7.2) messages on the varexo_det error files
# (<name>.pp.txt) and its JSON output of xdet_json.mod (json/). Run from a PRIVATE copy of
# reference/exo_det.
PP=${DYNARE_ROOT:-/Applications/Dynare/${DYNARE_VERSION:-7.2}-arm64}/preprocessor/dynare-preprocessor
for f in xdet_e_*.mod; do
  n=${f%.mod}
  $PP $f > $n.pp.txt 2>&1
done
$PP xdet_json.mod json=compute onlyjson > /dev/null 2>&1
mkdir -p json
cp xdet_json/model/json/modfile.json json/xdet_json_modfile.json
cp xdet_json/model/json/modfile-original.json json/xdet_json_modfile-original.json
