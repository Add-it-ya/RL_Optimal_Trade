"""The committed results: each evaluation points at committed runs whose weights match.

Fast consistency checks only; ``scripts/reproduce.py`` re-scores the runs and checks the
numbers themselves.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = sorted(
    p.parent for p in (ROOT / "results").glob("*/meta.json") if p.parent.name != "_reproduce"
)


@pytest.mark.skipif(not RESULTS, reason="no committed results in this checkout")
@pytest.mark.parametrize("result", RESULTS, ids=lambda p: p.name)
def test_each_result_points_at_runs_whose_weights_match(result):
    meta = json.loads((result / "meta.json").read_text(encoding="utf-8"))
    assert not Path(meta["runs_dir"]).is_absolute()  # portable, no machine paths
    for run in meta["runs"]:
        folder = ROOT / run["run_dir"]
        prov = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        weights = hashlib.sha256((folder / prov["model_file"]).read_bytes()).hexdigest()
        assert weights == prov["weights_sha256"] == run["weights_sha256"], run["run_dir"]
    table = pd.read_csv(result / "comparison.csv")
    seeds = pd.read_csv(result / "per_seed.csv")
    assert {"IS_cvar95", "L1_vs_AC", "vs_AlmgrenChriss"} <= set(table.columns)
    assert len(seeds) == len(meta["runs"]) * len(meta["regimes"])
    for (regime, agent), group in seeds.groupby(["regime", "agent"]):
        row = table[(table["regime"] == regime) & (table["strategy"] == agent)].iloc[0]
        assert group["vs_AlmgrenChriss"].mean() == pytest.approx(row["vs_AlmgrenChriss"])


@pytest.mark.skipif(not RESULTS, reason="no committed results in this checkout")
def test_results_figures_draw_from_the_committed_tables(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "make_figures", ROOT / "scripts" / "make_figures.py"
    )
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    monkeypatch.setattr(figures, "FIGURES", tmp_path)
    figures.setup(figures.THEMES["dark"])
    for draw in figures.FIGS:
        assert draw(figures.THEMES["dark"], "dark").stat().st_size > 10_000
    assert len(list(tmp_path.glob("*-dark.png"))) == len(figures.FIGS)
