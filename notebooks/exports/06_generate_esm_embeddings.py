# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% cell 0
# -*- coding: utf-8 -*-
"""Data_Extraction_V5_Grouped_Split.ipynb"""

# ============================================
# 1. Setup & Installation
# ============================================
# NOTEBOOK_MAGIC: !pip -q install mdtraj biopython fair-esm joblib scikit-learn

import os
import zipfile
import shutil
import gc
import numpy as np
import pandas as pd
import torch
import mdtraj as md
from Bio.SeqUtils import seq1
from tqdm import tqdm
from joblib import Parallel, delayed
import esm
from sklearn.model_selection import train_test_split

# ============================================
# 2. Configuration
# ============================================
DRIVE_BASE     = "/content/drive/MyDrive/af_native_dynamics_predictor"
INPUT_ZIP      = os.path.join(DRIVE_BASE, "data/raw/MD_Simulation/proteins_filtered.zip")
CSV_PATH       = os.path.join(DRIVE_BASE, "data/processed/filtered_chain_ids_with_256_cutoff.csv")
FINAL_OUT_DIR  = os.path.join(DRIVE_BASE, "data/processed/MD_Simulation/V3_Model_Data/with_esm_650M")

LOCAL_ZIP_TEMP  = "/content/raw_data.zip"
LOCAL_UNZIP_DIR = "/content/unzipped_raw"
LOCAL_NPZ_DIR   = "/content/processed_npz"

ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM_LAYER      = 33
BATCH_SIZE_ESM = 16
CHUNK_FRAMES   = 1000

from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

os.makedirs(LOCAL_UNZIP_DIR, exist_ok=True)
os.makedirs(LOCAL_NPZ_DIR, exist_ok=True)
os.makedirs(FINAL_OUT_DIR, exist_ok=True)

# ============================================
# 3. Helpers
# ============================================
def copy_with_progress(src, dst):
    size = os.path.getsize(src)
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        with tqdm(total=size, unit='B', unit_scale=True, desc=f"Copying {os.path.basename(src)}") as pbar:
            while True:
                chunk = fsrc.read(1024 * 1024)
                if not chunk: break
                fdst.write(chunk)
                pbar.update(len(chunk))

def _normalize(v, eps=1e-8):
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)

@torch.no_grad()
def frames_from_n_ca_c(n_xyz, ca_xyz, c_xyz):
    e1 = _normalize(ca_xyz - n_xyz)
    e2 = _normalize(c_xyz - ca_xyz)
    e3 = _normalize(torch.cross(e1, e2, dim=-1))
    e2p = torch.cross(e3, e1, dim=-1)
    return torch.stack([e1, e2p, e3], dim=-1), ca_xyz

def extract_metadata_from_pdb(pdb_path):
    try:
        top = md.load(pdb_path).topology
        n_idx, ca_idx, c_idx, seq, mask = [], [], [], [], []
        for res in top.residues:
            aa = seq1(res.name, custom_map={})
            seq.append(aa if len(aa) == 1 else "X")
            atoms = {a.name: a.index for a in res.atoms}
            has_all = ("N" in atoms and "CA" in atoms and "C" in atoms)
            mask.append(1.0 if has_all else 0.0)
            n_idx.append(atoms.get("N", -1))
            ca_idx.append(atoms.get("CA", -1))
            c_idx.append(atoms.get("C", -1))
        return n_idx, ca_idx, c_idx, "".join(seq), np.asarray(mask, dtype=np.float32)
    except: return None

# ============================================
# 4. ESM Inference
# ============================================
def run_batch_esm(unique_sequences, device="cuda"):
    print(f"Loading {ESM_MODEL_NAME} to {device}...")
    model, alphabet = getattr(esm.pretrained, ESM_MODEL_NAME)()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    embeddings_dict = {}

    for i in tqdm(range(0, len(unique_sequences), BATCH_SIZE_ESM), desc="Batch ESM Inference"):
        batch_seqs = unique_sequences[i : i + BATCH_SIZE_ESM]
        _, _, batch_tokens = batch_converter([(str(j), s) for j, s in enumerate(batch_seqs)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                results = model(batch_tokens, repr_layers=[ESM_LAYER])
                token_embeddings = results["representations"][ESM_LAYER]
                for j, seq in enumerate(batch_seqs):
                    embeddings_dict[seq] = token_embeddings[j, 1 : len(seq) + 1].cpu().numpy().astype(np.float16)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return embeddings_dict

# ============================================
# 5. Worker
# ============================================
def process_single_protein(task_data, esm_embedding):
    sample_id, xtc_path, pdb_path = task_data
    out_file = os.path.join(LOCAL_NPZ_DIR, f"{sample_id}.npz")

    try:
        meta = extract_metadata_from_pdb(pdb_path)
        if meta is None: return False
        n_idx, ca_idx, c_idx, seq_str, mask = meta
        N = len(seq_str)

        coords_list, rots_list = [], []
        for chunk in md.iterload(xtc_path, top=pdb_path, chunk=CHUNK_FRAMES):
            xyz_A = chunk.xyz * 10.0
            def gather(idxs):
                out = np.zeros((xyz_A.shape[0], N, 3), dtype=np.float32)
                valid = np.array(idxs) >= 0
                out[:, valid, :] = xyz_A[:, np.array(idxs)[valid], :]
                return torch.from_numpy(out)

            R, t = frames_from_n_ca_c(gather(n_idx), gather(ca_idx), gather(c_idx))
            coords_list.append(t.numpy().astype(np.float16))
            rots_list.append(R.numpy().astype(np.float16))

        np.savez_compressed(
            out_file,
            coords=np.concatenate(coords_list, axis=0),
            rotations=np.concatenate(rots_list, axis=0),
            mask=mask, seq_str=np.array(seq_str), esm=esm_embedding
        )
        return True
    except Exception: return False

# ============================================
# 6. Main Flow
# ============================================
def main():
    # PHASE A: Fetch & Unzip
    if not os.path.exists(LOCAL_ZIP_TEMP):
        copy_with_progress(INPUT_ZIP, LOCAL_ZIP_TEMP)

    if not os.listdir(LOCAL_UNZIP_DIR):
        with zipfile.ZipFile(LOCAL_ZIP_TEMP, 'r') as z:
            for member in tqdm(z.infolist(), desc="Unzipping Raw Data"):
                z.extract(member, LOCAL_UNZIP_DIR)

    # PHASE B: Scan
    df = pd.read_csv(CSV_PATH)
    valid_tasks, unique_seqs, seq_map = [], set(), {}

    print("Scanning PDBs and Replicates...")
    for pid in tqdm(df['chain_id'].tolist()):
        pdb_p = os.path.join(LOCAL_UNZIP_DIR, pid, f"{pid}.pdb")
        found_meta = False
        for i in [1, 2, 3]:
            xtc_p = os.path.join(LOCAL_UNZIP_DIR, pid, f"{pid}_R{i}.xtc")
            if os.path.exists(pdb_p) and os.path.exists(xtc_p):
                if not found_meta:
                    meta = extract_metadata_from_pdb(pdb_p)
                    if meta:
                        seq_map[pid] = meta[3]
                        unique_seqs.add(meta[3])
                        found_meta = True
                if found_meta:
                    valid_tasks.append((f"{pid}_R{i}", xtc_p, pdb_p, pid))

    esm_cache = run_batch_esm(list(unique_seqs))

    # PHASE C: Parallel Process
    print(f"Processing {len(valid_tasks)} samples...")
    Parallel(n_jobs=4, backend="threading")(
    delayed(process_single_protein)(t[:3], esm_cache[seq_map[t[3]]])
        for t in tqdm(valid_tasks)
    )

    import re

    def base_from_npz_filename(fname: str) -> str:
      # fname examples: "6ovk_R_R1.npz", "1abc_R2.npz"
      stem = fname[:-4] if fname.endswith(".npz") else fname
      return re.sub(r"_R[1-3]$", "", stem)

    # PHASE D: Grouped Split & Zip (NO SPLIT FOLDERS)
    print("Performing Grouped Split to prevent data leakage...")

    processed_files = [f for f in os.listdir(LOCAL_NPZ_DIR) if f.endswith(".npz")]

    # 1) Identify unique protein bases (group)
    unique_bases = list(set(base_from_npz_filename(f) for f in processed_files))

    # 2) Split the bases
    train_bases, test_bases = train_test_split(unique_bases, test_size=0.2, random_state=42)
    val_bases, test_bases   = train_test_split(test_bases, test_size=0.5, random_state=42)

    base_split_map = {
        "train": set(train_bases),
        "val":   set(val_bases),
        "test":  set(test_bases),
    }

    # 3) Create zips directly in FINAL_OUT_DIR
    for split, base_set in base_split_map.items():
        files_to_zip = [f for f in processed_files if base_from_npz_filename(f) in base_set]

        z_name  = f"{split}.zip"
        z_local = os.path.join("/content", z_name)
        z_final = os.path.join(FINAL_OUT_DIR, z_name)

        print(f"Creating {z_name} with {len(files_to_zip)} files...")
        with zipfile.ZipFile(z_local, "w", zipfile.ZIP_DEFLATED) as z:
            for f in tqdm(files_to_zip, desc=f"Zipping {split}"):
                z.write(os.path.join(LOCAL_NPZ_DIR, f), arcname=f)

        copy_with_progress(z_local, z_final)

    print("\n--- ALL TASKS COMPLETE ---")

if __name__ == "__main__":
    main()

