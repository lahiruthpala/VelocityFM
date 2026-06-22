# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% cell 0
# ============================================================
# ATLAS Protein Contact Stability Analysis (Debug Version)
# Author: Lahiru
# ============================================================

# --- 1. Setup and dependencies ---
# NOTEBOOK_MAGIC: !pip install MDAnalysis seaborn matplotlib scipy tqdm pandas --quiet

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import MDAnalysis as mda
import traceback

# --- 2. User configuration ---
BASE_PATH = "/content/drive/MyDrive/af_native_dynamics_predictor/data"
TSV_PATH = f"{BASE_PATH}/ATLAS_protein_list.tsv"
RESULTS_DIR = "/content/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("[INFO] Base path:", BASE_PATH)
print("[INFO] TSV file:", TSV_PATH)
print("[INFO] Results directory:", RESULTS_DIR)

# --- 3. Load metadata ---
print("[INFO] Loading metadata...")
try:
    meta = pd.read_csv(TSV_PATH, sep="\t")
    pdb_ids = meta["PDB"].dropna().unique()
    print(f"[INFO] Loaded {len(pdb_ids)} PDB IDs from TSV.")
except Exception as e:
    print("[ERROR] Failed to read TSV:", e)
    raise

# --- 4. Helper function ---
def compute_avg_contact_matrix(universe, cutoff=8.0, sample_step=10, fraction=0.1):
    try:
        ca = universe.select_atoms("name CA")
        if len(ca) == 0:
            print("[WARN] No CA atoms found in", universe.filename)
            return None

        n = len(ca)
        n_frames = len(universe.trajectory)
        contact_sum = np.zeros((n, n), dtype=float)

        n_sample = max(1, int(n_frames * fraction))
        frames = range(0, n_sample, sample_step)
        if len(frames) == 0:
            frames = [0]

        for frame in frames:
            universe.trajectory[frame]
            if frame % 50 == 0:
                print(f"[DEBUG] Processing frame {frame}/{n_frames}")

            dist = np.linalg.norm(
                ca.positions[:, None, :] - ca.positions[None, :, :],
                axis=-1
            )
            contacts = (dist < cutoff).astype(float)
            np.fill_diagonal(contacts, 0.0)
            contact_sum += contacts

        return contact_sum / len(frames)

    except Exception as e:
        print("[ERROR] In compute_avg_contact_matrix:", e)
        traceback.print_exc()
        return None

# --- 6. Main analysis loop (with debug logging and robust plotting) ---
summary = []

for pdb_id in tqdm(pdb_ids):
    pdb_dir = os.path.join(BASE_PATH, "raw/MD_Simulation/proteins", pdb_id)
    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    xtc_file = os.path.join(pdb_dir, f"{pdb_id}_R1.xtc")

    # print(f"\n[INFO] Processing {pdb_id}")
    # print(f"   PDB: {pdb_file}")
    # print(f"   XTC: {xtc_file}")

    if not (os.path.exists(pdb_file) and os.path.exists(xtc_file)):
        print(f"   [WARN] Missing files for {pdb_id}, skipping.")
        continue

    try:
        u = mda.Universe(pdb_file, xtc_file)
        n_frames = len(u.trajectory)
        # print(f"   [INFO] Loaded {n_frames} frames")

        if n_frames < 20:
            print("   [WARN] Too few frames, skipping.")
            continue

        # --- Compute averages for first and last 10% ---
        # print("   [DEBUG] Computing initial contact matrix...")
        C_init = compute_avg_contact_matrix(u, fraction=0.1)
        # print("   [DEBUG] Computing final contact matrix...")
        u.trajectory[-1]  # Ensure end loaded
        C_final = compute_avg_contact_matrix(u, fraction=0.1)

        # --- Correlation and variance check ---
        mask = np.triu_indices_from(C_init, k=1)
        init_vals = C_init[mask]
        final_vals = C_final[mask]

        var_init = np.var(init_vals)
        var_final = np.var(final_vals)
        corr, _ = pearsonr(init_vals, final_vals)
        # print(f"   [RESULT] Correlation for {pdb_id} = {corr:.4f}, Var(init)={var_init:.3e}, Var(final)={var_final:.3e}")
        print(corr)
        if corr > 0.5:
            # print("[WARN] Low correlation, skipping.")
            continue

        # --- Robust density/scatter plotting ---
        if var_init < 1e-8 or var_final < 1e-8:
            print(f"   [WARN] Low variance data — using scatter instead of KDE.")
            plt.figure(figsize=(6,5))
            plt.scatter(init_vals, final_vals, s=2, alpha=0.5)
        else:
            try:
                plt.figure(figsize=(6,5))
                sns.kdeplot(x=init_vals, y=final_vals, fill=True, cmap="viridis", levels=80)
            except Exception as e:
                print(f"   [ERROR] KDE failed ({e}), using scatter plot.")
                plt.figure(figsize=(6,5))
                plt.scatter(init_vals, final_vals, s=2, alpha=0.5)

        plt.xlabel("Initial Contact Probability")
        plt.ylabel("Final Contact Probability")
        plt.title(f"{pdb_id} — Contact Correlation (r = {corr:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{pdb_id}_density.png"))
        plt.close()

        # --- Heatmap difference ---
        diff = C_final - C_init
        plt.figure(figsize=(6,5))
        sns.heatmap(diff, cmap="coolwarm", center=0)
        plt.title(f"{pdb_id} — ΔContact Probability (Final − Initial)")
        plt.xlabel("Residue Index")
        plt.ylabel("Residue Index")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{pdb_id}_heatmap.png"))
        plt.close()

        summary.append({
            "PDB_ID": pdb_id,
            "Correlation": corr,
            "Frames": n_frames,
            "Var_Init": var_init,
            "Var_Final": var_final
        })

    except Exception as e:
        print(f"[ERROR] Failed processing {pdb_id}: {e}")
        continue

# --- 7. Save summary ---
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(RESULTS_DIR, "contact_stability_summary.csv"), index=False)
print("\n[INFO] Analysis complete!")
print(f"[INFO] Results saved in: {RESULTS_DIR}")

