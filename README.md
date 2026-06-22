# VelocityFM

VelocityFM is a short-horizon protein-trajectory research prototype that applies
rectified flow matching to translational, rotational, and torsional velocity
sequences over residue frames. The model combines an IPA spatial trunk with
per-residue temporal self-attention.

This repository is a **curated migration snapshot** assembled from the linked
Google Drive research resources. It preserves the current notebooks, population
results, paper PDF, exact Drive-source metadata, configuration snapshots, and a
small importable core package.

## Important status

The complete model and trainer are still notebook-based and include Colab,
Google Drive, FoldFlow, and OpenFold assumptions. The `src/velocityfm` package
contains portable mathematical components, but it is not yet a complete rewrite
of the full trainer. See `docs/migration_status.md` and `docs/audit_notes.md`.

## Repository layout

```text
configs/       Model, training, inference, and evaluation snapshots
src/           Portable SO(3), AR(1), integration, loss, and temporal modules
notebooks/     Curated original Drive notebooks and searchable exports
data/          Dataset metadata and evidenced test-protein manifest
results/       Population CSV, derived summaries, and evaluation figures
checkpoints/   Metadata and links only; large checkpoint binaries are excluded
artifacts/     Drive-source and repository manifests
paper/         Supplied paper PDF
scripts/       Notebook export and repository validation utilities
tests/         Lightweight unit and consistency tests
```

## Install the portable core

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[test]"
pytest -q
python scripts/validate_repository.py
```

For notebook dependencies:

```bash
pip install -e ".[notebooks,test]"
```

OpenFold and FoldFlow require separate installation; consult
`docs/dependencies.md`.

## Main notebooks

- Training: `notebooks/02_training/01_velocityfm_training_current.ipynb`
- Inference: `notebooks/03_inference/01_velocityfm_inference_strictpdb.ipynb`
- Evaluation: `notebooks/04_evaluation/01_protein_trajectory_validation.ipynb`
- OpenFold feature extraction: `notebooks/01_data/05_extract_openfold_features_v4.ipynb`

## Results included

`results/tables/results_population.csv` contains 72 protein-level rows. The
summary in `results/metrics/evaluation_summary.json` is derived directly from
that CSV rather than manually copied from the manuscript.

## Large files deliberately excluded

The complete raw/processed ATLAS data, training checkpoints, logs, and generated
trajectories are not embedded in Git history. Their Drive identifiers are in:

- `artifacts/drive_source_manifest.json`
- `checkpoints/checkpoint_manifest.json`
- `data/README.md`

Before public release, place immutable large artifacts in Zenodo or another
research archive and replace the Drive links with DOI-backed records.

## Reproducibility warning

The current training notebook was modified after the population validation
artifact and currently resumes from a later training run. The paper-reported
settings also differ from the notebook snapshot in several places. These are
listed explicitly in `docs/audit_notes.md`; reconcile them before submission.
