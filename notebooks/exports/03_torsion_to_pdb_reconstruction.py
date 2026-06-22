# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% cell 0
from google.colab import drive
drive.mount('/content/drive')
BASE_FOLDER = "/content/drive/MyDrive/af_native_dynamics_predictor"

# %% cell 1
import numpy as np
import torch
import math
import os

# -------------------------
# Configuration
# -------------------------
Model = "Model_T2"
Traning = 1

# Path to your generated file
INPUT_NPZ = f"{BASE_FOLDER}/models/{Model}/Traning_{1}/Predictions/final_long_trajectory.npz"
OUTPUT_PDB = INPUT_NPZ.replace(".npz", ".pdb")

# -------------------------
# Constants for Backbone Geometry
# -------------------------
# Idealized bond lengths (Angstroms)
LEN_N_CA = 1.46
LEN_CA_C = 1.52
LEN_C_N  = 1.33

# Idealized bond angles (Radians)
# N-CA-C
ANG_N_CA_C = math.radians(111.0)
# CA-C-N
ANG_CA_C_N = math.radians(116.0)
# C-N-CA
ANG_C_N_CA = math.radians(122.0)

def extend_arm(prev_a, prev_b, prev_c, length, bond_angle, torsion_angle):
    """
    Places atom D based on atoms A, B, C and internal coords.
    Implements the NeRF (Natural Extension Reference Frame) algorithm.
    """
    # Normalize vectors
    bc = prev_c - prev_b
    bc = bc / np.linalg.norm(bc)

    n = np.cross(prev_b - prev_a, bc)
    n = n / np.linalg.norm(n)

    # Create D in local frame
    m = np.array([
        bc,
        np.cross(n, bc),
        n
    ]).T # Transpose to get rotation matrix

    d_local = np.array([
        length * np.cos(bond_angle),
        length * np.sin(bond_angle) * np.cos(torsion_angle),
        length * np.sin(bond_angle) * np.sin(torsion_angle)
    ])

    return prev_c + m @ d_local

def angles_from_sincos(sincos_array):
    """
    Converts (..., 2) sin/cos array to radians.
    """
    # sincos shape: (..., 2) -> (sin, cos)
    # arctan2(y, x) -> arctan2(sin, cos)
    return np.arctan2(sincos_array[..., 0], sincos_array[..., 1])

def reconstruct_backbone(torsions):
    """
    Reconstructs (N, CA, C) coordinates from phi, psi, omega angles.
    torsions: (Time, Residues, 5) -> Radians
    Returns: coords (Time, Residues * 3, 3) -> Interleaved N, CA, C
    """
    T, N_res, _ = torsions.shape

    # Extract backbone angles:
    # Index 0: Phi (N-CA)
    # Index 1: Psi (CA-C)
    # Index 2: Omega (C-N next)
    phi   = torsions[:, :, 0]
    psi   = torsions[:, :, 1]
    omega = torsions[:, :, 2]

    # Initialize container for one frame
    # We will build residues one by one
    # Order: N, CA, C, N, CA, C...

    all_frames_coords = []

    print(f"Reconstructing {T} frames...")

    for t in range(T):
        # Initial 3 atoms to start the chain (Arbitrary placement)
        # N at origin
        n_0  = np.array([0.0, 0.0, 0.0])
        # CA on X axis
        ca_0 = np.array([LEN_N_CA, 0.0, 0.0])
        # C in XY plane
        c_x = LEN_N_CA + LEN_CA_C * np.cos(np.pi - ANG_N_CA_C)
        c_y = LEN_CA_C * np.sin(np.pi - ANG_N_CA_C)
        c_0  = np.array([c_x, c_y, 0.0])

        frame_atoms = [n_0, ca_0, c_0]

        for i in range(1, N_res):
            prev_n  = frame_atoms[-3]
            prev_ca = frame_atoms[-2]
            prev_c  = frame_atoms[-1]

            # 1. Place Next N (using psi of previous, omega of previous)
            # Torsion for N depends on C-N bond, which rotates by psi of prev residue?
            # Actually standard NeRF chain:
            # To place N_i: torsion is psi_{i-1} (rotation around CA_{i-1}-C_{i-1})?
            # Standard definition:
            # Omega: CA(i-1) - C(i-1) - N(i) - CA(i)
            # Psi:   N(i)    - CA(i)  - C(i) - N(i+1)
            # Phi:   C(i-1)  - N(i)   - CA(i)- C(i)

            # Place N_i
            # Bond: C_{i-1} -> N_i
            # Angle: CA_{i-1} - C_{i-1} - N_i
            # Torsion: N_{i-1} - CA_{i-1} - C_{i-1} - N_i (Psi of prev)
            n_i = extend_arm(prev_n, prev_ca, prev_c, LEN_C_N, np.pi - ANG_CA_C_N, psi[t, i-1])

            # Place CA_i
            # Bond: N_i -> CA_i
            # Angle: C_{i-1} - N_i - CA_i
            # Torsion: CA_{i-1} - C_{i-1} - N_i - CA_i (Omega of prev)
            ca_i = extend_arm(prev_ca, prev_c, n_i, LEN_N_CA, np.pi - ANG_C_N_CA, omega[t, i-1])

            # Place C_i
            # Bond: CA_i -> C_i
            # Angle: N_i - CA_i - C_i
            # Torsion: C_{i-1} - N_i - CA_i - C_i (Phi of curr)
            c_i = extend_arm(prev_c, n_i, ca_i, LEN_CA_C, np.pi - ANG_N_CA_C, phi[t, i])

            frame_atoms.extend([n_i, ca_i, c_i])

        all_frames_coords.append(np.array(frame_atoms))

    return np.array(all_frames_coords)

def write_pdb(coords, output_path):
    """
    Writes a multi-model PDB file.
    coords: (Time, N_atoms, 3)
    """
    T, N_atoms, _ = coords.shape
    N_res = N_atoms // 3

    print(f"Writing PDB to {output_path}...")

    with open(output_path, 'w') as f:
        for t in range(T):
            f.write(f"MODEL     {t+1}\n")
            atom_idx = 1
            for r in range(N_res):
                # We have 3 atoms per residue in order: N, CA, C
                base = r * 3

                # Atom N
                x, y, z = coords[t, base]
                f.write(f"ATOM  {atom_idx:5d}  N   ALA A{r+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           N\n")
                atom_idx += 1

                # Atom CA
                x, y, z = coords[t, base+1]
                f.write(f"ATOM  {atom_idx:5d}  CA  ALA A{r+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                atom_idx += 1

                # Atom C
                x, y, z = coords[t, base+2]
                f.write(f"ATOM  {atom_idx:5d}  C   ALA A{r+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                atom_idx += 1

            f.write("ENDMDL\n")
    print("Done.")

# -------------------------
# Main Execution
# -------------------------
def main():
    if not os.path.exists(INPUT_NPZ):
        print(f"Error: {INPUT_NPZ} does not exist.")
        return

    # 1. Load Data
    d = np.load(INPUT_NPZ)
    sincos = d['torsions'] # Shape (T, N, 5, 2)

    # 2. Convert to Radians
    print("Converting sin/cos to radians...")
    angles = angles_from_sincos(sincos) # Shape (T, N, 5)

    # 3. Reconstruct Backbone (N, CA, C)
    coords = reconstruct_backbone(angles)

    # 4. Save
    write_pdb(coords, OUTPUT_PDB)

if __name__ == "__main__":
    main()

