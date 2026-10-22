#!/usr/bin/env python3
"""Runs the Dynare preprocessor (DYNARE_VERSION env, default 7.2) on .mod files and extracts M_.block_structure /
M_.block_structure_stat from the generated driver.m (1-based, layout of
`BlockStructure::to_json`); with --compare, also runs dynare-blocks (target/release) and reports
the differences.

    python3 reference/blocks/extract_blocks.py [--compare] [--out DIR] file.mod ...

The preprocessor runs in target/blocks_oracle/<dir>__<name>/ (the .mod and the files next to it
that it may include are copied there).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PP = os.environ.get('DYNARE_ROOT', f"/Applications/Dynare/{os.environ.get('DYNARE_VERSION', '7.2')}-arm64") + "/preprocessor/dynare-preprocessor"
ROOT = Path(__file__).resolve().parent.parent.parent
WORK = ROOT / "target" / "blocks_oracle"
BIN = ROOT / "target" / "release" / "dynare-blocks"


def ints(s):
    return [int(x) for x in s.split()]


def parse(drv, prefix, dynamic):
    p = re.escape(prefix)
    if not re.search(rf"^{p}\.block\(", drv, re.M):
        return None
    blocks = {}
    for m in re.finditer(rf"^{p}\.block\((\d+)\)\.(\w+) = (.*);$", drv, re.M):
        k, f, v = int(m.group(1)), m.group(2), m.group(3)
        b = blocks.setdefault(k, {})
        if f in ("Simulation_Type", "endo_nbr", "mfs"):
            b[f] = int(v)
        elif f in ("equation", "variable"):
            b[f] = ints(v.strip("[]"))
        elif f == "is_linear" and dynamic:
            b[f] = v == "true"
    out = {"block": [blocks[k] for k in sorted(blocks)]}
    for f in ("variable_reordered", "equation_reordered"):
        m = re.search(rf"^{p}\.{f} = \[(.*)\];$", drv, re.M)
        out[f] = ints(m.group(1))
    inc = []
    if dynamic:
        out["time_recursive"] = re.search(rf"^{p}\.time_recursive = (\w+);", drv, re.M).group(1) == "true"
        pat = rf"^{p}\.incidence\((\d+)\)\.lead_lag = (-?\d+);\n{p}\.incidence\(\d+\)\.sparse_IM = \[\n(.*?)\];$"
        for m in re.finditer(pat, drv, re.M | re.S):
            rows = [ints(r.strip().rstrip(";")) for r in m.group(3).strip().split("\n") if r.strip()]
            inc.append({"lead_lag": int(m.group(2)), "sparse_IM": rows})
    else:
        m = re.search(rf"^{p}\.incidence\.sparse_IM = \[\n(.*?)\];$", drv, re.M | re.S)
        rows = [ints(r.strip().rstrip(";")) for r in m.group(1).strip().split("\n") if r.strip()]
        inc.append({"lead_lag": 0, "sparse_IM": rows})
    out["incidence"] = inc
    return out


def extract(src, defines=()):
    src = Path(src).resolve()
    name = src.stem
    wd = WORK / f"{src.parent.name}__{name}__{zlib.crc32(str(src).encode()) % 10**6}"
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    for f in src.parent.iterdir():
        if f.is_file() and f.suffix in (".mod", ".inc", ".m", ".mat", ".csv", ".xlsx", ".modfile", ".json"):
            shutil.copy(f, wd / f.name)
    r = subprocess.run([PP, src.name, "nolog", *[f"-D{d}" for d in defines]], cwd=wd, capture_output=True, text=True)
    drv_path = wd / f"+{name}" / "driver.m"
    # with use_dll the preprocessor fails after writing driver.m (no MEX compilation setup)
    if r.returncode != 0 and not ("use_dll" in r.stdout + r.stderr and drv_path.exists()):
        return {"error": (r.stdout + r.stderr)[-2000:]}
    drv = drv_path.read_text()
    return {"static": parse(drv, "M_.block_structure_stat", False), "dynamic": parse(drv, "M_.block_structure", True)}


def ours(src):
    r = subprocess.run([str(BIN), str(src)], capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr[-2000:]}
    return json.loads(r.stdout)


def diff(a, b, path=""):
    if type(a) is not type(b):
        return [f"{path}: {json.dumps(a)[:200]} != {json.dumps(b)[:200]}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), f"{path}.{k}")
        return out
    if isinstance(a, list) and a and isinstance(a[0], dict):
        if len(a) != len(b):
            return [f"{path}: {len(a)} blocks != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i + 1}]")
        return out
    return [] if a == b else [f"{path}: {json.dumps(a)[:300]} != {json.dumps(b)[:300]}"]


def main():
    args = sys.argv[1:]
    compare = "--compare" in args
    args = [a for a in args if a != "--compare"]
    outdir = None
    if "--out" in args:
        i = args.index("--out")
        outdir = Path(args[i + 1])
        del args[i:i + 2]
    jobs = 1
    if "-j" in args:
        i = args.index("-j")
        jobs = int(args[i + 1])
        del args[i:i + 2]

    def run(f):
        ref = extract(f)
        return f, ref, (ours(f) if compare and "error" not in ref else None)

    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(run, args))
    nok = 0
    for f, ref, got in results:
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{Path(f).resolve().parent.name}__{Path(f).stem}.json").write_text(json.dumps(ref, separators=(",", ":")) + "\n")
        if not compare:
            print(json.dumps(ref))
            continue
        if "error" in ref:
            print(f"SKIP {f}: preprocessor failed")
            continue
        if "error" in got:
            lines = got["error"].strip().splitlines()
            print(f"ERR  {f}: {lines[-1] if lines else ''}")
            continue
        d = diff(ref, got)
        if d:
            print(f"DIFF {f}")
            for x in d[:6]:
                print("     " + x)
        else:
            nok += 1
            print(f"OK   {f}")
    if compare:
        print(f"{nok}/{len(args)} identical")


if __name__ == "__main__":
    main()
