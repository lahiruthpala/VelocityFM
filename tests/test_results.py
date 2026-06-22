from pathlib import Path

import pandas as pd


def test_population_result_table():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "results/tables/results_population.csv")
    assert len(df) == 72
    assert df["pdb_id"].is_unique
    assert (df["gen_tm_gt70"] == 1.0).all()
