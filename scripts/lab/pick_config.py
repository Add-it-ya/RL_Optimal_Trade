#!/usr/bin/env python
"""Pick the tuning study's winner from its tuning-block evaluations.

    python scripts/lab/pick_config.py --out results/tuning \\
        A=results/e1_vec_tuning C=results/e1t_C_tuning D=results/e1t_D_tuning B=results/e1t_B_tuning

Candidates are given **cheapest first**. Each folder is an ``rlx evaluate`` + ``rlx report``
output on the tuning block (seeds from 1,500,000: inside the validation range, disjoint from
the episodes checkpoint selection used, and never the test block). The rule, fixed before any
result:

1. Configs with a degenerate seed (fewer than 5 effective steps) are not eligible.
2. Among the rest, the lowest across-seed mean gap to Almgren-Chriss wins...
3. ...except that a cheaper config within ``--tie-bps`` (default 0.05) of the best is
   preferred: a difference that small is noise, not a reason to pay for more compute.

Writes ``choice.json`` (the winner, its config file and every candidate's numbers) and
``TUNING.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
AGENT = "VecPPO"


def candidate_row(label: str, results: Path) -> Dict:
    """The VecPPO row of one tuning-block evaluation."""
    table = pd.read_csv(results / "comparison.csv")
    row = table[table["strategy"] == AGENT]
    if row.empty:
        raise ValueError(f"{results}: no {AGENT} row in comparison.csv")
    r = row.iloc[0]
    return {
        "label": label,
        "results": str(results),
        "n_seeds": int(r["n_seeds"]),
        "n_degenerate": int(r["n_degenerate"]),
        "J_bps": float(r["J_bps"]),
        "gap_vs_AC": float(r["vs_AlmgrenChriss"]),
        "ci_low": float(r["ci_low"]),
        "ci_high": float(r["ci_high"]),
        "L1_vs_AC": float(r["L1_vs_AC"]),
    }


def choose(rows: Sequence[Dict], tie_bps: float = 0.05) -> str:
    """The winning label; ``rows`` must be ordered cheapest first."""
    eligible = [r for r in rows if r["n_degenerate"] == 0]
    if not eligible:
        raise ValueError("every config has a degenerate seed; nothing to pick")
    best = min(r["gap_vs_AC"] for r in eligible)
    return next(r["label"] for r in eligible if r["gap_vs_AC"] <= best + tie_bps)


def config_file(label: str) -> str:
    matches = sorted(CONFIG_DIR.glob(f"{label}_*.yaml"))
    if len(matches) != 1:
        raise ValueError(f"expected one config file for {label!r} in {CONFIG_DIR}, got {matches}")
    return str(matches[0].relative_to(CONFIG_DIR.parents[2])).replace("\\", "/")


def parse(pairs: Sequence[str]) -> List[Tuple[str, Path]]:
    out = []
    for pair in pairs:
        label, _, path = pair.partition("=")
        if not label or not path:
            raise ValueError(f"expected LABEL=RESULTS_DIR, got {pair!r}")
        out.append((label, Path(path)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("candidates", nargs="+", help="LABEL=RESULTS_DIR, cheapest first")
    parser.add_argument("--out", required=True, help="folder for choice.json and TUNING.md")
    parser.add_argument("--tie-bps", type=float, default=0.05)
    args = parser.parse_args()

    rows = [candidate_row(label, path) for label, path in parse(args.candidates)]
    winner = choose(rows, args.tie_bps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    choice = {
        "winner": winner,
        "config": config_file(winner),
        "rule": (
            "no degenerate seed; lowest mean gap to AC on the tuning block (seeds from "
            f"1,500,000); a cheaper config within {args.tie_bps} bps of the best wins"
        ),
        "candidates_cheapest_first": rows,
    }
    (out / "choice.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")

    cols = ["label", "n_seeds", "n_degenerate", "J_bps", "gap_vs_AC", "ci_low", "ci_high"]
    cols += ["L1_vs_AC"]
    table = pd.DataFrame(rows)[cols]

    def cell(v) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    markdown = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    markdown += ["| " + " | ".join(cell(v) for v in r) + " |" for r in table.itertuples(False)]
    lines = [
        "# Tuning study on E1: choice of config",
        "",
        f"Rule (fixed in advance): {choice['rule']}. Candidates listed cheapest first. These",
        "numbers are on the tuning block, not the test block; the winner is scored on the test",
        "block afterwards, once.",
        "",
        *markdown,
        "",
        f"**Winner: {winner}** (`{choice['config']}`)",
        "",
    ]
    (out / "TUNING.md").write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwinner: {winner} ({choice['config']}); wrote {out / 'choice.json'} and TUNING.md")


if __name__ == "__main__":
    main()
