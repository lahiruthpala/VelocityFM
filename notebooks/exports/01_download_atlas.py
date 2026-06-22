# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% cell 0
# NOTEBOOK_MAGIC: !pip install MDAnalysis numpy panda

# %% cell 1
from google.colab import drive
import os
import requests
import json
import time
from typing import List
from pathlib import Path
import pandas as pd, os, glob
import subprocess
import numpy as np
import MDAnalysis as mda
import io
import zipfile
import shutil
drive.mount('/content/drive')

# %% cell 2
ATLAS_API_BASE= "https://www.dsimb.inserm.fr/ATLAS/api"
BASE_FOLDER = "/content/drive/MyDrive/af_native_dynamics_predictor/data"

# %% cell 3
def fetch_atlas_data(pdb_chain: str):
    pdb_id   = pdb_chain[:4].lower()
    chain_id = pdb_chain[4:]
    url      = f"{ATLAS_API_BASE}/ATLAS/analysis/{pdb_chain}"

    resp = requests.get(url, stream=True)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"[ATLAS] {pdb_chain} → HTTP {resp.status_code}: {e}")
        return

    # Look for a filename in Content-Disposition
    cd = resp.headers.get('Content-Disposition', '') or resp.headers.get('content-disposition', '')
    if 'attachment' in cd and 'filename=' in cd:
        # Extract “16pk_A_analysis.zip” → folder “16pk_A_analysis”
        filename   = cd.split('filename=')[1].strip('"').rsplit('_', 1)[0]
        foldername = os.path.splitext(filename)[0]
        folderpath = os.path.join(f"{BASE_FOLDER}/raw/MD_Simulation/proteins", foldername)
        os.makedirs(folderpath, exist_ok=True)

        # Stream ZIP into memory, then extract
        buf = io.BytesIO()
        for chunk in resp.iter_content(chunk_size=8192):
            buf.write(chunk)
        buf.seek(0)
        with zipfile.ZipFile(buf) as z:
            z.extractall(folderpath)

        print(f"[ATLAS] extracted ZIP to folder → {foldername}")

    else:
        # Fallback: assume JSON
        try:
            data = resp.json()
        except ValueError:
            print(f"[ATLAS] Unexpected non-JSON response for {pdb_chain}")
            return

        out_path = os.path.join(ATLAS_DIR, f"{pdb_chain}_ATLAS.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[ATLAS] saved JSON data → {os.path.basename(out_path)}")

# %% cell 4
# Paths
INPUT_CSV = os.path.join(BASE_FOLDER, 'ATLAS_protein_list.tsv')

# 1) Read the CSV (assumes first column is pdb_chain)
df = pd.read_csv(INPUT_CSV, sep='\t')

# 2) Add a status column (empty by default)
if 'ATLAS_Download' not in df.columns:
    df['ATLAS_Download'] = ''

# 3) Process each entry, updating status on success/failure
for idx, row in df.iterrows():
    pdb = row['PDB']
    print(f"\nProcessing {pdb} ...")
    if df.at[idx, 'ATLAS_Download'] != 'done':
        print(f"→ downloading from ATLAS")
        try:
            fetch_atlas_data(pdb)
            df.at[idx, 'ATLAS_Download'] = 'done'
        except Exception as e:
            print(f"ATLAS fetch error for {pdb}: {e}")
            df.at[idx, 'ATLAS_Download'] = 'error'
    else:
        print(f"→ already done, skipping")
    df.to_csv(INPUT_CSV, index=False, sep='\t')
    time.sleep(1)
# Write once after loop completes
print("\nAll entries processed and TSV updated")

