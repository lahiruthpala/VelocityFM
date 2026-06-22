# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% cell 0
from google.colab import drive
drive.mount('/content/drive')
BASE_FOLDER = "/content/drive/MyDrive/af_native_dynamics_predictor"
DATA_DIR = f"{BASE_FOLDER}/data/processed/MD_Simulation/V3_Model_Data/proteins"
CHECK_POINTS = f"{BASE_FOLDER}/models/Model_T3_V3/Traning_1/checkpoints"

# %% cell 1
import os
import glob
import shutil
import random
import csv
import numpy as np
from tqdm.auto import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_DATA_DIR = DATA_DIR  # Defined in your previous cells
BASE_OUTPUT_DIR = f"{BASE_FOLDER}/data/processed/MD_Simulation/V3_Model_Data/dataset_splits"

# Split Ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# CRITICAL: Set a fixed seed so the shuffle is the same every time you restart!
RANDOM_SEED = 42

# ==========================================
# SPLITTING LOGIC
# ==========================================
def split_dataset_resumable(source_dir, output_dir):
    # 1. Setup Fixed Randomness
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 2. Get all .npz files
    all_files = sorted(glob.glob(os.path.join(source_dir, "*.npz")))
    if len(all_files) == 0:
        print(f"Error: No .npz files found in {source_dir}")
        return

    print(f"Found {len(all_files)} total files.")

    # 3. Group files by Protein ID (PDB)
    pdb_groups = {}
    for f in all_files:
        filename = os.path.basename(f)
        pdb_id = filename.split('_')[0]
        if pdb_id not in pdb_groups:
            pdb_groups[pdb_id] = []
        pdb_groups[pdb_id].append(f)

    unique_pdbs = list(pdb_groups.keys())

    # SHUFFLE: This will now be the SAME order every time due to the seed
    random.shuffle(unique_pdbs)

    print(f"Found {len(unique_pdbs)} unique proteins.")

    # 4. Calculate Split Indices
    n_total = len(unique_pdbs)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    splits = {
        "train": unique_pdbs[:n_train],
        "val":   unique_pdbs[n_train : n_train + n_val],
        "test":  unique_pdbs[n_train + n_val:]
    }

    # 5. Process Splits
    print("\nStarting resumable copy process...")

    total_files_processed = 0
    total_files_skipped = 0

    # We iterate through the splits (train, val, test)
    for split_name, pdb_list in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # CSV Path
        csv_file_path = os.path.join(output_dir, f"{split_name}_files.csv")

        # A. Load Existing Progress
        copied_files_set = set()
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None) # Skip header
                for row in reader:
                    if row: copied_files_set.add(row[0]) # Assuming filename is col 0

        print(f"\nProcessing {split_name.upper()} split...")
        print(f" -> Found {len(copied_files_set)} files already copied in previous runs.")

        # B. Prepare file list for this split
        files_for_split = [f for pdb in pdb_list for f in pdb_groups[pdb]]

        # C. Open CSV in Append Mode
        # We use 'a' to append new files as we copy them
        # buffering=1 ensures lines are written to disk frequently
        with open(csv_file_path, 'a', newline='', buffering=1) as csvfile:
            writer = csv.writer(csvfile)

            # Write header only if file is empty/new
            if os.path.getsize(csv_file_path) == 0:
                writer.writerow(["filename", "pdb_id", "original_path"])

            # D. Iterate and Copy
            with tqdm(total=len(files_for_split), desc=f"Copying {split_name}") as pbar:
                for src_path in files_for_split:
                    filename = os.path.basename(src_path)

                    # CHECK: Have we copied this already?
                    if filename in copied_files_set:
                        total_files_skipped += 1
                        pbar.update(1)
                        continue # SKIP COPY

                    # DO COPY
                    try:
                        dst_path = os.path.join(split_dir, filename)
                        shutil.copy(src_path, dst_path)

                        # LOG TO CSV
                        pdb_id = filename.split('_')[0]
                        writer.writerow([filename, pdb_id, src_path])

                        total_files_processed += 1
                        pbar.update(1)

                    except OSError as e:
                        print(f"\nCRITICAL ERROR copying {filename}: {e}")
                        print("Run the script again to resume from this point.")
                        return # Stop script immediately on error

    print(f"\nDone! Processed {total_files_processed} new files. Skipped {total_files_skipped} existing files.")

# Run the split
split_dataset_resumable(SOURCE_DATA_DIR, BASE_OUTPUT_DIR)

