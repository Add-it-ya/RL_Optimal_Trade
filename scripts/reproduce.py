#!/usr/bin/env python
"""Re-score the committed policies and check the published tables.

    python scripts/reproduce.py --list            # what can be checked
    python scripts/reproduce.py --quick           # E1, tuned VecPPO (a few minutes)
    python scripts/reproduce.py e1 e3 --workers 4 # chosen results
    python scripts/reproduce.py --all --workers 4 # everything, plus both E2 sweeps

Every committed ``results/<name>/`` was produced by ``rlx evaluate`` from the committed run
folders under ``runs/``; its ``meta.json`` records which runs, regimes and test seeds. This
script repeats that evaluation into ``results/_reproduce/<name>/`` (ignored by git), writes
the report, and compares the new ``comparison.csv`` with the committed one, row by row.
``load_run`` re-hashes every weight file against its ``provenance.json`` on the way, so a
changed weight fails loudly before anything is scored. The E2 sweeps are rebuilt from the
re-scored experiments and compared the same way.

On the same machine the numbers come out bit-identical. Across machines the policies'
float32 inference differs in the last bits (measured: <= 4.2e-6 bps per episode between the
lab and a laptop), so the default tolerance, 1e-3 bps, is far below the precision anything
is reported at. Exit status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT  # where runs/ and results/ live (tests point it elsewhere)
RESULTS = ROOT / "results"
OUT = RESULTS / "_reproduce"

#: Columns compared, in bps unless named otherwise.
BPS_COLUMNS = ["J_bps", "IS_bps", "IS_std", "IS_p95", "IS_cvar95"]
GAP_COLUMNS = ["ci_low", "ci_high"]
OTHER = {"eff_steps": 1e-3, "L1_vs_AC": 1e-4}
EXACT = ["n_seeds", "n_degenerate"]

#: E2 sweeps: the sweep folder and the experiments it combines.
SWEEPS: Dict[str, List[str]] = {
    "e2": ["e2_ic000", "e2_ic025", "e2_ic050", "e2_ic100", "e2_ic200"],
    "e2_tuned": ["e2t_ic000", "e2t_ic025", "e2t_ic050", "e2t_ic100", "e2t_ic200"],
}
QUICK = ["e1", "e1t_C"]


def committed() -> List[str]:
    """Every committed result folder that ``rlx evaluate`` wrote (it has a ``meta.json``)."""
    return sorted(p.parent.name for p in RESULTS.glob("*/meta.json") if p.parent.name != OUT.name)


def evaluate_command(name: str, meta: dict, out: Path, jobs: int) -> List[str]:
    runs_dir = Path(meta["runs_dir"])
    cmd = [sys.executable, "-m", "rl_execution.cli", "evaluate"]
    cmd += ["--experiment", runs_dir.name, "--runs-dir", str(ROOT / runs_dir.parent)]
    cmd += ["--regimes", *meta["regimes"], "--episodes", str(meta["episodes"])]
    cmd += ["--base-seed", str(meta["base_seed"]), "--out", str(out), "--jobs", str(jobs)]
    return cmd + (["--antithetic"] if meta["antithetic"] else [])


def compare_tables(
    new: pd.DataFrame, old: pd.DataFrame, keys: Sequence[str], columns: Dict[str, float]
) -> Tuple[bool, float, List[str]]:
    """``(ok, worst deviation, problems)``: ``new`` against ``old`` within ``columns``' tolerances.

    Rows are matched on ``keys``; a missing or extra row is a problem, NaN equals NaN, and a
    column absent from both tables is skipped.
    """
    problems: List[str] = []
    a, b = new.set_index(list(keys)), old.set_index(list(keys))
    for missing in b.index.difference(a.index):
        problems.append(f"row {missing} missing")
    for extra in a.index.difference(b.index):
        problems.append(f"unexpected row {extra}")
    worst = 0.0
    shared = a.index.intersection(b.index)
    for col, tol in columns.items():
        if col not in a.columns and col not in b.columns:
            continue
        if col not in a.columns or col not in b.columns:
            problems.append(f"column {col} only in one table")
            continue
        x = pd.to_numeric(a.loc[shared, col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(b.loc[shared, col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.where(both_nan, 0.0, np.abs(x - y))
        if np.isnan(diff).any():
            problems.append(f"{col}: NaN in one table only")
            diff = np.nan_to_num(diff, nan=np.inf)
        worst = max(worst, float(diff.max(initial=0.0)))
        for key, d in zip(shared, diff):
            if d > tol:
                problems.append(f"{col} {key}: off by {d:.3g} (tolerance {tol:g})")
    for col in EXACT:
        if col in a.columns and col in b.columns:
            for key in shared:
                if str(a.loc[key, col]) != str(b.loc[key, col]):
                    problems.append(f"{col} {key}: {a.loc[key, col]} != {b.loc[key, col]}")
    return not problems, worst, problems


def comparison_tolerances(table: pd.DataFrame, tol_bps: float) -> Dict[str, float]:
    gap = [c for c in table.columns if c.startswith("vs_")]
    return {c: tol_bps for c in BPS_COLUMNS + GAP_COLUMNS + gap} | OTHER


def run(cmd: List[str], log: Path) -> None:
    path = os.pathsep.join(p for p in (str(PACKAGE_ROOT), os.environ.get("PYTHONPATH")) if p)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": path}
    with log.open("w", encoding="utf-8") as f:
        done = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[2:5])} failed; see {log}")


def runs_mismatch(meta: dict) -> str:
    """Why the runs folder no longer holds exactly the runs ``meta`` was scored on ("" if it does).

    ``rlx evaluate`` scores every run in the folder, so a seed added (or retrained) after the
    evaluation would silently change what is being re-checked.
    """
    runs_dir = ROOT / meta["runs_dir"]
    present = sorted(
        json.loads(p.read_text(encoding="utf-8"))["weights_sha256"]
        for p in runs_dir.glob("*/seed_*/provenance.json")
    )
    recorded = sorted(r["weights_sha256"] for r in meta["runs"])
    if present == recorded:
        return ""
    return (
        f"{meta['runs_dir']} holds {len(present)} run(s), {len(set(present) & set(recorded))} of "
        f"them among the {len(recorded)} this result was scored on; not re-scoring"
    )


def rescore(name: str, jobs: int) -> None:
    """Repeat ``results/<name>``'s evaluation and report into ``OUT/<name>``."""
    meta = json.loads((RESULTS / name / "meta.json").read_text(encoding="utf-8"))
    mismatch = runs_mismatch(meta)
    if mismatch:
        raise RuntimeError(mismatch)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    run(evaluate_command(name, meta, out, jobs), out / "evaluate.log")
    report = [sys.executable, "-m", "rl_execution.cli", "report", "--experiment", name]
    run(report + ["--results", str(out)], out / "report.log")


def check(name: str, tol_bps: float) -> Tuple[bool, str]:
    """Compare the re-scored tables (``comparison.csv``, ``per_seed.csv``) with the committed."""
    keys = ["regime", "strategy"]
    old = pd.read_csv(RESULTS / name / "comparison.csv")
    new = pd.read_csv(OUT / name / "comparison.csv")
    ok, worst, problems = compare_tables(new, old, keys, comparison_tolerances(old, tol_bps))
    rows = len(old)
    if (RESULTS / name / "per_seed.csv").exists():
        old = pd.read_csv(RESULTS / name / "per_seed.csv")
        new = pd.read_csv(OUT / name / "per_seed.csv")
        seed_ok, seed_worst, seed_problems = compare_tables(
            new, old, keys, comparison_tolerances(old, tol_bps)
        )
        ok, worst, problems = ok and seed_ok, max(worst, seed_worst), problems + seed_problems
        rows += len(old)
    note = f"{rows} rows, worst deviation {worst:.2g}"
    return ok, note + "".join(f"\n      {p}" for p in problems[:10])


def reproduce_one(name: str, jobs: int, tol_bps: float) -> Tuple[str, bool, str]:
    t0 = time.time()
    try:
        rescore(name, jobs)
    except RuntimeError as err:
        return name, False, str(err)
    ok, note = check(name, tol_bps)
    return name, ok, f"{note}, {time.time() - t0:.0f}s"


def reproduce_sweep(sweep: str, tol_bps: float) -> Tuple[str, bool, str]:
    out = OUT / sweep
    cmd = [sys.executable, "-m", "rl_execution.cli", "sweep", "--experiments", *SWEEPS[sweep]]
    try:
        run(cmd + ["--results-root", str(OUT), "--out", str(out)], OUT / f"{sweep}.log")
    except RuntimeError as err:
        return sweep, False, str(err)
    old, new = pd.read_csv(RESULTS / sweep / "sweep.csv"), pd.read_csv(out / "sweep.csv")
    cols = {"J_bps": tol_bps, "improvement": tol_bps, "ci_low": tol_bps, "ci_high": tol_bps}
    ok, worst, problems = compare_tables(new, old, ["experiment", "benchmark", "strategy"], cols)
    note = f"{len(old)} rows, worst deviation {worst:.2g}"
    return sweep, ok, note + "".join(f"\n      {p}" for p in problems[:10])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="committed results/<name> folders to check")
    parser.add_argument("--all", action="store_true", help="every committed result, both sweeps")
    parser.add_argument("--quick", action="store_true", help=f"only {', '.join(QUICK)}")
    parser.add_argument("--list", action="store_true", help="list what can be checked")
    parser.add_argument("--workers", type=int, default=1, help="experiments at a time")
    parser.add_argument("--jobs", type=int, default=1, help="regimes at a time within one")
    parser.add_argument("--tol-bps", type=float, default=1e-3)
    args = parser.parse_args(argv)

    available = committed()
    if args.list or not (args.names or args.all or args.quick):
        for name in available:
            meta = json.loads((RESULTS / name / "meta.json").read_text(encoding="utf-8"))
            print(
                f"{name:16s} runs {meta['runs_dir']:18s} {len(meta['runs']):3d} runs, "
                f"{len(meta['regimes']):2d} regime(s), {meta['episodes']:,} episodes "
                f"from {meta['base_seed']:,}"
            )
        print(f"sweeps: {', '.join(SWEEPS)}")
        return 0
    names = available if args.all else (QUICK if args.quick else list(args.names))
    unknown = [n for n in names if n not in available]
    if unknown:
        parser.error(f"not a committed result: {', '.join(unknown)}")

    print(f"Re-scoring {len(names)} result(s) into {OUT.relative_to(ROOT)}/ ...")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        outcomes = list(pool.map(lambda n: reproduce_one(n, args.jobs, args.tol_bps), names))
    for sweep, members in SWEEPS.items():
        if all(m in names for m in members):
            outcomes.append(reproduce_sweep(sweep, args.tol_bps))
    failed = 0
    for name, ok, note in outcomes:
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:16s} {note}")
    print(f"{len(outcomes) - failed} passed, {failed} failed (tolerance {args.tol_bps:g} bps)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
