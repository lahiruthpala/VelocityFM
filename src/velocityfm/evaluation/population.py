"""Helpers for summarising protein-level evaluation tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_COLUMNS = {
    "tm_score": "gen_tm_mean",
    "ca_rmsd_angstrom": "gen_rmsd_mean",
    "gdt_ts": "gen_gdt_ts",
    "valid_ca_ca_fraction": "gen_valid_pct",
    "steric_clash_fraction": "gen_clash_pct",
    "ramachandran_favoured_fraction": "gen_rama_fav",
    "rmsf_pearson": "rmsf_pearson",
    "rmsf_ratio": "rmsf_ratio",
    "helix_preservation_ratio": "helix_pres_ratio",
    "diversity_rmsd_angstrom": "gen_div_rmsd",
}


def summarize_population(path: str | Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    return {
        "n_proteins": int(len(df)),
        "median": {name: float(df[column].median()) for name, column in DEFAULT_COLUMNS.items()},
    }
