#!/usr/bin/env python3
"""Run lightweight consistency checks on this migration snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    csv_path = ROOT / "results/tables/results_population.csv"
    split_path = ROOT / "data/splits/test_proteins_from_population_results.txt"
    summary_path = ROOT / "results/metrics/evaluation_summary.json"

    data = pd.read_csv(csv_path)
    ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text())

    assert len(data) == 72, f"Expected 72 result rows, found {len(data)}"
    assert data["pdb_id"].tolist() == ids
    assert summary["n_test_proteins"] == len(data)
    assert data["pdb_id"].is_unique
    print("Repository snapshot checks passed.")


if __name__ == "__main__":
    main()
