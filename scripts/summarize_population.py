#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from velocityfm.evaluation.population import summarize_population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=Path("results/tables/results_population.csv"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize_population(args.csv), indent=2))


if __name__ == "__main__":
    main()
