# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% cell 0
# NOTEBOOK_MAGIC: !pip install -q MDAnalysis biopython pandas numpy scipy tqdm

# %% cell 1
import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD
from scipy.stats import median_abs_deviation, zscore, wasserstein_distance

# %% cell 2
from google.colab import drive
drive.mount('/content/drive')
BASE_FOLDER = "/content/drive/MyDrive/af_native_dynamics_predictor/data"

# %% cell 3
# ===============================
# Config
# ===============================
METADATA_DIR = f"{BASE_FOLDER}/raw/metadata"
METADATA_TSV = f"{METADATA_DIR}/ATLAS/2023_03_09_ATLAS_info.tsv"
BASE_DIR     = f"{BASE_FOLDER}/raw/MD_Simulation/proteins"

# Filtering hyperparameters (feel free to adjust)
COIL_MAX_PERCENT        = 50.0         # Step 1: exclude chains with coil% > this
POLY_DEGREE             = 2            # Step 2: degree for Rg ~ poly(length)
RESID_Z_THR             = 3.0          # Step 2: |standardized residual| threshold to flag Rg outliers
RMSD_SLOPE_Z_THR        = 3.0          # Step 3: z-threshold on RMSD slope (positive drift)
RMSD_MEAN_Z_THR         = 3.0          # Step 3: z-threshold on mean RMSD outliers
MIN_FRAMES              = 200          # Step 4: minimum frames per replicate
MISS_FRAC_MAX           = 0.01         # Step 5: max fraction of residues missing backbone atoms
SIM_WASSERSTEIN_THR     = 0.05         # Step 6: cull replicates whose RMSD distributions are almost identical (Å)
PDB_REGEX               = r"^[0-9][a-z0-9]{3}$"  # sanity check for PDB code

# %% cell 4
# ===============================
# Helpers: chain/rep discovery
# ===============================

# %% cell 5
def list_chain_dirs(base_dir):
    """
    Return sorted list of chain directory names like '16pk_A'.
    A valid chain dir must contain a PDB named '16pk_A.pdb'.
    """
    out = []
    for name in os.listdir(base_dir):
        d = os.path.join(base_dir, name)
        if not os.path.isdir(d):
            continue
        pdb_path = os.path.join(d, f"{name}.pdb")
        if os.path.exists(pdb_path):
            out.append(name)
    return sorted(out)

# %% cell 6
def parse_pdb_chain_from_dirname(chain_id):
    """
    '16pk_A' -> ('16pk', 'A')
    """
    if "_" not in chain_id:
        return None, None
    pdb, chain = chain_id.split("_", 1)
    return pdb.lower(), chain

# %% cell 7
def list_replicates(chain_dir, chain_id):
    """
    Return available replicate tags for this chain, e.g., ['R1','R2','R3'] subset.
    """
    reps = []
    for rep in ["R1","R2","R3"]:
        if os.path.exists(os.path.join(chain_dir, f"{chain_id}_{rep}.xtc")):
            reps.append(rep)
    return reps

# %% cell 8
# ===============================
# Load metadata TSV
# ===============================
meta = pd.read_csv(METADATA_TSV, sep="\t")
# Standardize useful columns; adapt here if your column names differ
REQUIRED_COLS = ["PDB", "length", "coil%", "avg_gyration"]
missing_cols = [c for c in REQUIRED_COLS if c not in meta.columns]
if missing_cols:
    raise ValueError(f"Your TSV is missing required columns: {missing_cols}")

# Normalize PDB id in metadata
meta["PDB_norm"] = meta["PDB"].astype(str).str.strip()

# Map from (pdb, chain_letter) to metadata row(s).
# If your TSV encodes chain letter elsewhere, you can extend this mapping.
meta_index = {row["PDB_norm"]: i for i, row in meta.iterrows()}
print(f"Metadata rows: {len(meta)}")
print(meta_index)

# %% cell 9
meta.

# %% cell 10
# ===============================
# Logging structures
# ===============================
details_rows = []   # per-step, per-entity decisions: chain or replicate
summary_counts = defaultdict(lambda: {"checked": 0, "kept": 0, "removed": 0})

def log_decision(step, entity_type, chain_id, replicate, decision, reason):
    """
    entity_type: 'chain' or 'replicate'
    decision: 'kept' or 'removed'
    """
    details_rows.append({
        "step": step,
        "entity_type": entity_type,
        "chain_id": chain_id,
        "replicate": replicate if replicate is not None else "",
        "decision": decision,
        "reason": reason
    })
    summary_counts[(step, entity_type)]["checked"] += 1
    if decision == "kept":
        summary_counts[(step, entity_type)]["kept"] += 1
    else:
        summary_counts[(step, entity_type)]["removed"] += 1

# %% cell 11
# ===============================
# Scan filesystem for chains & replicates
# ===============================
chain_dirs = list_chain_dirs(BASE_DIR)
print(f"Discovered {len(chain_dirs)} chain directories under {BASE_DIR}.")

# Build initial chain table
chains_df = pd.DataFrame({"chain_id": chain_dirs})
chains_df[["pdb","chain_letter"]] = chains_df["chain_id"].str.split("_", n=1, expand=True)
chains_df["pdb"] = chains_df["pdb"].str.lower()
chains_df["meta_idx"] = chains_df["chain_id"].map(meta_index)
chains_df["has_meta"] = chains_df["meta_idx"].notna()
print(f"Chains with matching metadata rows: {chains_df['has_meta'].sum()} / {len(chains_df)}")
print(len(chains_df))

# %% cell 12
df.head()

# %% cell 13
# ===============================
# Step 1: exclude high coil content (> 50%)
# (chain-level)
# ===============================
kept_chains_step1 = []
for _, row in chains_df.iterrows():
    chain_id = row["chain_id"]
    if not row["has_meta"]:
        log_decision(1, "chain", chain_id, None, "removed",
                     "No metadata row for PDB; cannot evaluate coil%")
        continue
        print(f"{chain_id} has no metadata")
    meta_row = meta.iloc[int(row["meta_idx"])]
    coil = meta_row["coil%"]
    if pd.isna(coil):
        log_decision(1, "chain", chain_id, None, "removed",
                     "coil% is NaN in metadata")
        print(f"{chain_id} has no metadata")
        continue
    if float(coil) > COIL_MAX_PERCENT:
        log_decision(1, "chain", chain_id, None, "removed",
                     f"coil%={coil:.1f} > {COIL_MAX_PERCENT}")
        # print(f"{chain_id} has coil%={coil:.1f} > {COIL_MAX_PERCENT}")
    else:
        log_decision(1, "chain", chain_id, None, "kept",
                     f"coil%={coil:.1f} ≤ {COIL_MAX_PERCENT}")
        kept_chains_step1.append(chain_id)

print(f"[After Step 1] Chains kept: {len(kept_chains_step1)}")

# Restrict to chains that passed Step 1
chains_df = chains_df[chains_df["chain_id"].isin(kept_chains_step1)].reset_index(drop=True)

# %% cell 14
# ===============================
# Step 2: Rg polynomial outliers vs length
# (chain-level)
# Fit avg_gyration ~ poly(length) and remove large standardized residuals
# ===============================
if len(chains_df) == 0:
    raise SystemExit("No chains remain after Step 1.")

# Build regression table from metadata for the remaining chains
rg_rows = []
for _, row in chains_df.iterrows():
    meta_row = meta.iloc[int(row["meta_idx"])]
    length = meta_row["length"]
    rg     = meta_row["avg_gyration"]
    if pd.isna(length) or pd.isna(rg):
        continue
    rg_rows.append((row["chain_id"], float(length), float(rg)))

rg_df = pd.DataFrame(rg_rows, columns=["chain_id","length","avg_gyration"])
if len(rg_df) < POLY_DEGREE + 2:
    print("Warning: too few chains for a stable polynomial fit; Step 2 will be lenient.")

# Fit polynomial
x = rg_df["length"].values
y = rg_df["avg_gyration"].values
coef = np.polyfit(x, y, POLY_DEGREE)
yhat = np.polyval(coef, x)
resid = y - yhat

# Robust standardization of residuals
resid_mad = median_abs_deviation(resid, scale='normal')
if resid_mad == 0:
    resid_z = np.zeros_like(resid)
else:
    resid_z = (resid - np.median(resid)) / resid_mad

kept_chains_step2 = []
for cid, rz in zip(rg_df["chain_id"], resid_z):
    if abs(rz) > RESID_Z_THR:
        log_decision(2, "chain", cid, None, "removed",
                     f"Rg residual z={rz:.2f} > {RESID_Z_THR}")
    else:
        log_decision(2, "chain", cid, None, "kept",
                     f"Rg residual z={rz:.2f} ≤ {RESID_Z_THR}")
        kept_chains_step2.append(cid)

print(f"[After Step 2] Chains kept: {len(kept_chains_step2)}")

chains_df = chains_df[chains_df["chain_id"].isin(kept_chains_step2)].reset_index(drop=True)
if len(chains_df) == 0:
    raise SystemExit("No chains remain after Step 2.")

# %% cell 15
# ===============================
# Step 5: missing residues or ambiguities (chain-level)
# (We do Step 5 before 3/4 because it is cheap and chain-level.)
# Define: fraction of residues missing backbone atoms (N, CA, C, O) must be <= MISS_FRAC_MAX
# ===============================
def backbone_missing_fraction(pdb_path):
    u = mda.Universe(pdb_path)
    prot = u.select_atoms("protein")
    miss = 0
    total = 0
    for res in prot.residues:
        names = set(a.name.upper() for a in res.atoms)
        total += 1
        if not {"N","CA","C","O"}.issubset(names):
            miss += 1
    return (miss / total) if total > 0 else 1.0

kept_chains_step5 = []
for _, row in tqdm(chains_df.iterrows(), total=len(chains_df)):
    chain_id = row["chain_id"]
    pdb_path = os.path.join(BASE_DIR, chain_id, f"{chain_id}.pdb")
    try:
        miss_frac = backbone_missing_fraction(pdb_path)
    except Exception as e:
        log_decision(5, "chain", chain_id, None, "removed",
                     f"PDB parse error: {e}")
        continue
    if miss_frac > MISS_FRAC_MAX:
        log_decision(5, "chain", chain_id, None, "removed",
                     f"Missing-backbone fraction {miss_frac:.3f} > {MISS_FRAC_MAX}")
    else:
        log_decision(5, "chain", chain_id, None, "kept",
                     f"Missing-backbone fraction {miss_frac:.3f} ≤ {MISS_FRAC_MAX}")
        kept_chains_step5.append(chain_id)

print(f"[After Step 5] Chains kept: {len(kept_chains_step5)}")

chains_df = chains_df[chains_df["chain_id"].isin(kept_chains_step5)].reset_index(drop=True)
if len(chains_df) == 0:
    raise SystemExit("No chains remain after Step 5.")

# %% cell 16
# ===============================
# Build replicate table for remaining chains
# ===============================
rep_rows = []
for _, row in chains_df.iterrows():
    chain_id = row["chain_id"]
    cdir = os.path.join(BASE_DIR, chain_id)
    reps = list_replicates(cdir, chain_id)
    for rep in reps:
        xtc = os.path.join(cdir, f"{chain_id}_{rep}.xtc")
        rep_rows.append((chain_id, rep, cdir, xtc))
rep_df = pd.DataFrame(rep_rows, columns=["chain_id","replicate","chain_dir","xtc_path"])

print(f"Replicates discovered after chain-level filtering: {len(rep_df)}")

if len(rep_df) == 0:
    raise SystemExit("No replicates remain after chain-level filters.")

# %% cell 17
# ===============================
# Utilities for RMSD and frames
# ===============================
def replicate_num_frames(pdb_path, xtc_path):
    u = mda.Universe(pdb_path, xtc_path)
    return len(u.trajectory)

def replicate_rmsd_series(pdb_path, xtc_path):
    """
    Cα RMSD to first frame (Å). Returns np.array shape (frames,)
    """
    u = mda.Universe(pdb_path, xtc_path)
    R = RMSD(u, u, select="protein and name CA", ref_frame=0)
    R.run()
    # R.rmsd columns: [frame, time(ps), RMSD(Å)]
    return R.rmsd[:, 2]

# %% cell 18
# Pre-compute frames and RMSD series for all replicates (used in Steps 3,4,6)
rep_df["pdb_path"] = rep_df.apply(lambda r: os.path.join(r["chain_dir"], f"{r['chain_id']}.pdb"), axis=1)
rep_meta = []
print("Precomputing frames and RMSD for replicate-level filters (Steps 3,4,6)...")
for i, r in tqdm(rep_df.iterrows(), total=len(rep_df)):
    cid, rep, pdbp, xtc = r["chain_id"], r["replicate"], r["pdb_path"], r["xtc_path"]
    try:
        nF = replicate_num_frames(pdbp, xtc)
        rmsd = replicate_rmsd_series(pdbp, xtc)
        rep_meta.append((cid, rep, nF, rmsd))
    except Exception as e:
        # Will be handled as "removed" in step 3 with reason
        rep_meta.append((cid, rep, 0, f"ERROR::{e}"))

rep_meta_df = pd.DataFrame(rep_meta, columns=["chain_id","replicate","num_frames","rmsd_series"])

# %% cell 19
# ===============================
# Step 3: exclude replicates failing RMSD stability (outliers)
# We use two robust signals across all replicates:
#   - RMSD mean z-score
#   - RMSD slope z-score (positive drift)
# A replicate is removed if mean_z > RMSD_MEAN_Z_THR or slope_z > RMSD_SLOPE_Z_THR.
# ===============================
def slope_of_series(y):
    x = np.arange(len(y))
    if len(y) < 2:
        return np.nan
    coef = np.polyfit(x, y, 1)
    return coef[0]  # Å per frame

# Build stats arrays where possible
means, slopes, valid_mask = [], [], []
for _, r in rep_meta_df.iterrows():
    rs = r["rmsd_series"]
    if isinstance(rs, str) and rs.startswith("ERROR::"):
        means.append(np.nan); slopes.append(np.nan); valid_mask.append(False)
    else:
        means.append(float(np.mean(rs)))
        slopes.append(float(slope_of_series(rs)))
        valid_mask.append(True)

means = np.array(means, dtype=float)
slopes = np.array(slopes, dtype=float)
valid_mask = np.array(valid_mask, dtype=bool)

# Robust z-scores (use MAD; fall back to standard z-score if MAD==0)
def robust_z(v):
    v = v.copy()
    m = np.nanmedian(v)
    mad = median_abs_deviation(v[~np.isnan(v)], scale='normal') if np.any(~np.isnan(v)) else 0.0
    if mad == 0 or np.isnan(mad):
        return zscore(v, nan_policy='omit')
    else:
        return (v - m) / mad

mean_z  = robust_z(means)
slope_z = robust_z(slopes)

kept_reps_step3 = []
for i, r in rep_meta_df.iterrows():
    cid, rep = r["chain_id"], r["replicate"]
    rs = r["rmsd_series"]
    if isinstance(rs, str) and rs.startswith("ERROR::"):
        log_decision(3, "replicate", cid, rep, "removed", f"RMSD computation failed: {rs[7:]}")
        continue
    if (not np.isfinite(mean_z[i])) or (not np.isfinite(slope_z[i])):
        log_decision(3, "replicate", cid, rep, "removed", "Non-finite RMSD stats")
        continue
    reason_parts = []
    remove = False
    if mean_z[i] > RMSD_MEAN_Z_THR:
        reason_parts.append(f"mean RMSD z={mean_z[i]:.2f} > {RMSD_MEAN_Z_THR}")
        remove = True
    if slope_z[i] > RMSD_SLOPE_Z_THR:
        reason_parts.append(f"slope z={slope_z[i]:.2f} > {RMSD_SLOPE_Z_THR}")
        remove = True
    if remove:
        log_decision(3, "replicate", cid, rep, "removed", "; ".join(reason_parts))
    else:
        log_decision(3, "replicate", cid, rep, "kept",
                     f"mean_z={mean_z[i]:.2f}, slope_z={slope_z[i]:.2f}")
        kept_reps_step3.append((cid, rep))

print(f"[After Step 3] Replicates kept: {len(kept_reps_step3)}")

rep_meta_df = rep_meta_df.merge(
    pd.DataFrame(kept_reps_step3, columns=["chain_id","replicate"]),
    on=["chain_id","replicate"], how="inner"
)

# %% cell 20
# ===============================
# Step 4: exclude short replicates (min frames)
# ===============================
kept_reps_step4 = []
for _, r in rep_meta_df.iterrows():
    cid, rep, nF = r["chain_id"], r["replicate"], int(r["num_frames"])
    if nF < MIN_FRAMES:
        log_decision(4, "replicate", cid, rep, "removed",
                     f"frames={nF} < MIN_FRAMES={MIN_FRAMES}")
    else:
        log_decision(4, "replicate", cid, rep, "kept",
                     f"frames={nF} ≥ MIN_FRAMES={MIN_FRAMES}")
        kept_reps_step4.append((cid, rep))

print(f"[After Step 4] Replicates kept: {len(kept_reps_step4)}")

rep_meta_df = rep_meta_df.merge(
    pd.DataFrame(kept_reps_step4, columns=["chain_id","replicate"]),
    on=["chain_id","replicate"], how="inner"
)

# %% cell 21
# ===============================
# Step 6: cull highly similar replicates (within each chain)
# Criterion: pairwise Wasserstein distance between RMSD distributions < SIM_WASSERSTEIN_THR
# Keep a minimal diverse subset, preferring lexicographic order (R1 > R2 > R3).
# ===============================
kept_reps_step6 = []

for cid, grp in rep_meta_df.groupby("chain_id"):
    # Order replicates R1, R2, R3 if present
    grp_sorted = grp.sort_values("replicate")
    reps = list(grp_sorted["replicate"].values)
    rmsd_series = {rep: grp_sorted[grp_sorted["replicate"] == rep]["rmsd_series"].values[0] for rep in reps}

    kept = []
    for rep in reps:
        if not kept:
            kept.append(rep)
            continue
        # Check similarity against all kept reps
        is_similar = False
        for rkept in kept:
            wd = wasserstein_distance(rmsd_series[rep], rmsd_series[rkept])
            if wd < SIM_WASSERSTEIN_THR:
                is_similar = True
                break
        if is_similar:
            log_decision(6, "replicate", cid, rep, "removed",
                         f"Wasserstein distance to kept replicate < {SIM_WASSERSTEIN_THR}")
        else:
            kept.append(rep)
            log_decision(6, "replicate", cid, rep, "kept",
                         f"Not redundant; sufficiently different RMSD distribution")
    for rep in kept:
        kept_reps_step6.append((cid, rep))

print(f"[After Step 6] Replicates kept: {len(kept_reps_step6)}")

# Final kept list (chain, replicate)
final_kept = pd.DataFrame(kept_reps_step6, columns=["chain_id","replicate"]).drop_duplicates()

# %% cell 22
# ===============================
# Summaries & CSV output
# ===============================
os.makedirs(f"{METADATA_DIR}", exist_ok=True)

details_df = pd.DataFrame(details_rows)
details_df.to_csv(f"{METADATA_DIR}/filter_details.csv", index=False)

summary_rows = []
for (step, etype), counts in summary_counts.items():
    summary_rows.append({
        "step": step,
        "entity_type": etype,
        "checked": counts["checked"],
        "kept": counts["kept"],
        "removed": counts["removed"]
    })
summary_df = pd.DataFrame(summary_rows).sort_values(["step","entity_type"])
summary_df.to_csv(f"{METADATA_DIR}/filter_summary_by_step.csv", index=False)

final_kept.to_csv(f"{METADATA_DIR}/final_kept_chain_replicates.csv", index=False)

print("\n=== SUMMARY BY STEP ===")
display(summary_df)

print("\n=== SAMPLE OF DECISIONS ===")
display(details_df.head(20))

print(f"\nFiles written to {METADATA_DIR}:")
print(" - filter_summary_by_step.csv")
print(" - filter_details.csv")
print(" - final_kept_chain_replicates.csv")

