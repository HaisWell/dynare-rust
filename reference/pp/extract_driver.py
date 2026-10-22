#!/usr/bin/env python3
"""Runs Dynare's preprocessor (DYNARE_VERSION, default 7.2) on reference/pp/*.mod and extracts the model structure from the
generated driver.m into reference/pp/<name>.json (format of tests/data/preprocessor_*.json, read
by tests/preprocessor_sw07.rs::pp_models_match_dynare).

    python3 reference/pp/extract_driver.py [name ...]
    python3 reference/pp/extract_driver.py --out OUT.json [-DNAME=VALUE ...] path/to/file.mod

The second form extracts any .mod file (e.g. reference/sw07/sw07.mod -DSTAGE=9 ->
tests/data/preprocessor_sw07.json, read by tests/preprocessor_sw07.rs).

The preprocessor runs in target/pp_oracle/<name>/ (it writes +<name>/ next to the .mod file).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# DYNARE_VERSION=7.1 in the environment runs the 7.1 preprocessor; the default is 7.2
PP = os.environ.get('DYNARE_ROOT', f"/Applications/Dynare/{os.environ.get('DYNARE_VERSION', '7.2')}-arm64") + "/preprocessor/dynare-preprocessor"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WORK = ROOT / "target" / "pp_oracle"


def ints(s):
    return [int(x) for x in s.split()]


def num(s):
    s = s.strip()
    if s in ("NaN", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract(name, src=None, out=None, defines=()):
    src = src or HERE / f"{name}.mod"
    out = out or HERE / f"{name}.json"
    wd = WORK / name
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    shutil.copy(src, wd / src.name)
    for inc in src.parent.glob(f"{name}_*.mod"):  # included files
        shutil.copy(inc, wd / inc.name)
    r = subprocess.run([PP, src.name, *defines], cwd=wd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"{name}: preprocessor failed")
    drv = (wd / f"+{name}" / "driver.m").read_text()

    def cell(field):
        out = {}
        for m in re.finditer(rf"^M_\.{field}\((\d+)\) = \{{'(.*)'\}};$", drv, re.M):
            out[int(m.group(1))] = m.group(2)
        return [out[k] for k in sorted(out)]

    def scalar(field):
        m = re.search(rf"^M_\.{re.escape(field)} = (.*);$", drv, re.M)
        return m.group(1) if m else None

    d = {
        "endo_names": cell("endo_names"),
        "exo_names": cell("exo_names"),
        "param_names": cell("param_names"),
        "orig_endo_nbr": int(scalar("orig_endo_nbr")),
    }
    aux = {}
    for m in re.finditer(r"^M_\.aux_vars\((\d+)\)\.(\w+) = (.*);$", drv, re.M):
        k, f, v = int(m.group(1)), m.group(2), m.group(3)
        e = aux.setdefault(k, {})
        if v.startswith("'"):
            e[f] = v[1:-1]
        else:
            try:
                e[f] = int(v)
            except ValueError:
                e[f] = v
    d["aux_vars"] = [aux[k] for k in sorted(aux)]
    m = re.search(r"^M_\.lead_lag_incidence = \[\n(.*?)\]';", drv, re.M | re.S)
    d["lead_lag_incidence"] = [ints(row) for row in m.group(1).replace(";", "\n").split("\n") if row.strip()]
    for f in ["dynamic_g1_sparse_rowval", "dynamic_g1_sparse_colval", "static_g1_sparse_rowval", "static_g1_sparse_colval"]:
        m = re.search(rf"^M_\.{f} = int32\(\[(.*)\]\);$", drv, re.M)
        d[f] = ints(m.group(1))
    params = {}
    for m in re.finditer(r"^M_\.params\((\d+)\) = (.*);$", drv, re.M):
        params[int(m.group(1))] = num(m.group(2))
    d["params"] = [params.get(k + 1) for k in range(len(d["param_names"]))]
    d["orig_maximum_lag"] = int(scalar("orig_maximum_lag"))
    d["orig_maximum_lead"] = int(scalar("orig_maximum_lead"))
    tags = []
    m = re.search(r"^M_\.equations_tags = \{\n(.*?)\};", drv, re.M | re.S)
    if m:
        for row in m.group(1).strip().split("\n"):
            parts = [p.strip() for p in row.rstrip(";").split(",")]
            tags.append([int(parts[0]), parts[1].strip("'"), parts[2].strip("'")])
    d["equations_tags"] = tags
    out.write_text(json.dumps(d, indent=1) + "\n")
    print(f"{name}: {len(d['endo_names'])} endogenous, {len(d['aux_vars'])} aux")


if __name__ == "__main__":
    if "--out" in sys.argv:
        args = sys.argv[1:]
        out = Path(args.pop(args.index("--out") + 1)).resolve()
        args.remove("--out")
        defines = [a for a in args if a.startswith("-D")]
        (path,) = [Path(a).resolve() for a in args if not a.startswith("-D")]
        extract(path.stem, path, out, defines)
        raise SystemExit
    names = sys.argv[1:] or sorted(p.stem for p in HERE.glob("*.mod") if "_" not in p.stem or p.stem.startswith("pp_"))
    for n in names:
        extract(n)
