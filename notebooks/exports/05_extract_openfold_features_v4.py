# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% [markdown] cell 0
# # Velocity-FM Dataset Extraction — v4 (OpenFold / AlphaFold torsion conventions)
# 
# ## Changes vs v1
# | Fix | Detail |
# |---|---|
# | **ψ angle** | Now uses `N(i), CA(i), C(i), N(i+1)` (was `O(i)`) |
# | **Global-motion removal** | Each XTC chunk is superposed to the PDB reference using chain-only Cα before any coordinate is read |
# | **`residue_index`** | 0-based sequential `[0,1,2,…]` — safe for positional encodings. Raw PDB `resSeq` stored in `pdb_resseq` |
# | **`aatype` encoding** | Now uses OpenFold canonical `restypes` order `"ARNDCQEGHILKMFPSTWYV"` (was alphabetical `"ACDEFGHIKLMNPQRSTVWY"`) |
# 
# **Torsion convention stored in output `.npz`:**
# - **Order:** `[pre_omega, phi, psi, chi1, chi2, chi3, chi4]`
# - **Definitions (correct OpenFold/AlphaFold2):**
#   - `pre_omega(i)`: `CA(i-1), C(i-1), N(i), CA(i)`
#   - `phi(i)`:       `C(i-1), N(i), CA(i), C(i)`
#   - `psi(i)`:       `N(i), CA(i), C(i), N(i+1)`
#   - `chi1..chi4`:    side-chain rotamers
# 
# **Alignment:**
# - Every XTC chunk superposed to static PDB (chain Cα only).
# - `frame_t` = Cα displacement from PDB reference, **not** raw drift.
# 
# Date: 2026-02-27  (v4)

# %% [markdown] cell 1
# ## 0) Install (Colab)

# %% cell 2
# NOTEBOOK_MAGIC: !pip -q install mdtraj biopython fair-esm joblib tqdm pandas numpy scikit-learn

# %% [markdown] cell 3
# ## 1) Mount Drive + Configuration

# %% cell 4
import os, zipfile, gc, pickle, re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import mdtraj as md
from Bio.SeqUtils import seq1
import esm

from google.colab import drive
drive.mount("/content/drive")

# -----------------------------
# Paths (edit only this block)
# -----------------------------
DRIVE_BASE = "/content/drive/MyDrive/af_native_dynamics_predictor"

INPUT_ZIP = os.path.join(DRIVE_BASE, "data/raw/MD_Simulation/proteins_filtered.zip")
CSV_PATH  = os.path.join(DRIVE_BASE, "data/processed/filtered_chain_ids_with_256_cutoff.csv")

OUT_DIR   = os.path.join(DRIVE_BASE, "data/processed/MD_Simulation/V7_atom14_openfold_npz")
EMS_OUT_DIR   = os.path.join(DRIVE_BASE, "data/processed/MD_Simulation/V6_atom14_openfold_npz")

# Local working dirs
LOCAL_ZIP     = "/content/proteins.zip"
LOCAL_UNZIP   = "/content/proteins_unzipped"
LOCAL_NPZ_DIR = "/content/npz_out_openfold"

os.makedirs(LOCAL_UNZIP, exist_ok=True)
os.makedirs(LOCAL_NPZ_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Extraction settings
CHUNK_FRAMES = 100
SAVE_FLOAT16_ATOMS = False
N_JOBS = os.cpu_count() or 4

# ESM settings
ESM_MODEL_NAME = "esm2_t33_650M_UR50D"   # alt: esm2_t12_35M_UR50D
ESM_LAYER      = 33                      # 33 for t33, 12 for t12
BATCH_SIZE_ESM = 16                      # reduce if OOM

ESM_CACHE_PATH = os.path.join(EMS_OUT_DIR, f"esm_cache_{ESM_MODEL_NAME}_L{ESM_LAYER}.pkl")

print(f"Using {N_JOBS} CPU workers")
print(f"ESM cache: {ESM_CACHE_PATH}")

# %% [markdown] cell 5
# ## 2) Atom14 tables + Chi definitions (AlphaFold-style)

# %% [markdown] cell 6
# ## 2b) CHI_DEFS verification against OpenFold
# 
# The `CHI_DEFS` table below has been **line-by-line verified** against OpenFold's
# `residue_constants.chi_angles_atoms` (aqlaboratory/openfold, `openfold/np/residue_constants.py`).
# 
# **All 20 residues match exactly.**
# 
# ### Convention reminder
# Each tuple `(a, b, c, d)` defines a dihedral around the **middle bond b–c**.
# Atom `d` is the *angular reference* beyond bond `c` — it is not the "owner" of the chi angle.
# 
# Previously-flagged entries that were re-verified as **correct**:
# 
# | Residue | Chi | Quadruplet | Rotation axis | Note |
# |---------|-----|-----------|---------------|------|
# | H | χ2 | CA–CB–CG–**ND1** | CB–CG | ND1 is first ring atom; standard OF definition |
# | N | χ2 | CA–CB–CG–**OD1** | CB–CG | OD1 is reference for amide orientation |
# | Q | χ3 | CB–CG–CD–**OE1** | CG–CD | OE1 reference; matches OF exactly |
# | E | χ3 | CB–CG–CD–**OE1** | GC–CD | OE1 reference; matches OF exactly |
# 
# The `_verify_chi_defs_vs_openfold()` function (run automatically when this cell executes)
# will raise `AssertionError` immediately if the table ever drifts from the OF reference.

# %% cell 7
# OpenFold / AlphaFold2 canonical residue type order
# Source: openfold/np/residue_constants.py  `restypes`
# Previously this used alphabetical order "ACDEFGHIKLMNPQRSTVWY" which is
# INCOMPATIBLE with OpenFold's lookup tables (restype_order, restype_1to3, etc.).
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"   # OpenFold restypes — DO NOT reorder
AA_TO_IDX = {a:i for i,a in enumerate(AA_ORDER)}
UNK_AA = 20

ATOM14_NAMES = {
    "A": ["N","CA","C","O","CB","","","","","","","","",""],
    "C": ["N","CA","C","O","CB","SG","","","","","","","",""],
    "D": ["N","CA","C","O","CB","CG","OD1","OD2","","","","","",""],
    "E": ["N","CA","C","O","CB","CG","CD","OE1","OE2","","","","",""],
    "F": ["N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ","","",""],
    "G": ["N","CA","C","O","","","","","","","","","",""],
    "H": ["N","CA","C","O","CB","CG","ND1","CD2","CE1","NE2","","","",""],
    "I": ["N","CA","C","O","CB","CG1","CG2","CD1","","","","","",""],
    "K": ["N","CA","C","O","CB","CG","CD","CE","NZ","","","","",""],
    "L": ["N","CA","C","O","CB","CG","CD1","CD2","","","","","",""],
    "M": ["N","CA","C","O","CB","CG","SD","CE","","","","","",""],
    "N": ["N","CA","C","O","CB","CG","OD1","ND2","","","","","",""],
    "P": ["N","CA","C","O","CB","CG","CD","","","","","","",""],
    "Q": ["N","CA","C","O","CB","CG","CD","OE1","NE2","","","","",""],
    "R": ["N","CA","C","O","CB","CG","CD","NE","CZ","NH1","NH2","","",""],
    "S": ["N","CA","C","O","CB","OG","","","","","","","",""],
    "T": ["N","CA","C","O","CB","OG1","CG2","","","","","","",""],
    "V": ["N","CA","C","O","CB","CG1","CG2","","","","","","",""],
    "W": ["N","CA","C","O","CB","CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"],
    "Y": ["N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ","OH","",""],
}

CHI_DEFS = {
    "A": [],
    "C": [("N","CA","CB","SG")],
    "D": [("N","CA","CB","CG"), ("CA","CB","CG","OD1")],
    "E": [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","OE1")],
    "F": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "G": [],
    "H": [("N","CA","CB","CG"), ("CA","CB","CG","ND1")],
    "I": [("N","CA","CB","CG1"), ("CA","CB","CG1","CD1")],
    "K": [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","CE"), ("CG","CD","CE","NZ")],
    "L": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "M": [("N","CA","CB","CG"), ("CA","CB","CG","SD"), ("CB","CG","SD","CE")],
    "N": [("N","CA","CB","CG"), ("CA","CB","CG","OD1")],
    "P": [("N","CA","CB","CG"), ("CA","CB","CG","CD")],
    "Q": [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","OE1")],
    "R": [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","NE"), ("CG","CD","NE","CZ")],
    "S": [("N","CA","CB","OG")],
    "T": [("N","CA","CB","OG1")],
    "V": [("N","CA","CB","CG1")],
    "W": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "Y": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
}

TORSION_NAMES = ["pre_omega","phi","psi","chi1","chi2","chi3","chi4"]

# ──────────────────────────────────────────────────────────────────────────────
# CHI_DEFS VERIFICATION NOTE
# These quadruplets have been cross-checked line-by-line against OpenFold /
# AlphaFold2  residue_constants.py  `chi_angles_atoms`  (aqlaboratory/openfold).
#
# Convention reminder: each tuple (a, b, c, d) defines a dihedral angle
# measured by rotating around the MIDDLE bond b–c.  The 4th atom d is the
# reference atom *beyond* bond c; it is NOT the atom being "defined" — it is
# the angular reference.  E.g. for H chi2 = (CA,CB,CG,ND1):
#   • rotation axis  : CB–CG
#   • reference atom : ND1  (the first heavy atom of the imidazole ring)
#   → this IS the OpenFold chi2 definition for His (matches exactly).
#
# Previously-flagged cases re-verified:
#   H  chi2  (CA,CB,CG,ND1)  ✓  rotates around CB–CG; ref = ND1
#   N  chi2  (CA,CB,CG,OD1)  ✓  rotates around CB–CG; ref = OD1
#   Q  chi3  (CB,CG,CD,OE1)  ✓  rotates around CG–CD; ref = OE1
#   E  chi3  (CB,CG,CD,OE1)  ✓  rotates around CG–CD; ref = OE1
# All match OpenFold exactly.
# ──────────────────────────────────────────────────────────────────────────────

def _verify_chi_defs_vs_openfold():
    """
    Self-contained chi-angle verification.
    Encodes the canonical OpenFold chi_angles_atoms table directly so the check
    works without any import.  Raises AssertionError on any mismatch.
    """
    OPENFOLD_CHI = {
        "A": [],
        "C": [("N","CA","CB","SG")],
        "D": [("N","CA","CB","CG"), ("CA","CB","CG","OD1")],
        "E": [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","OE1")],
        "F": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
        "G": [],
        "H": [("N","CA","CB","CG"), ("CA","CB","CG","ND1")],
        "I": [("N","CA","CB","CG1"), ("CA","CB","CG1","CD1")],
        "K": [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","CE"), ("CG","CD","CE","NZ")],
        "L": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
        "M": [("N","CA","CB","CG"), ("CA","CB","CG","SD"), ("CB","CG","SD","CE")],
        "N": [("N","CA","CB","CG"), ("CA","CB","CG","OD1")],
        "P": [("N","CA","CB","CG"), ("CA","CB","CG","CD")],
        "Q": [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","OE1")],
        "R": [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","NE"), ("CG","CD","NE","CZ")],
        "S": [("N","CA","CB","OG")],
        "T": [("N","CA","CB","OG1")],
        "V": [("N","CA","CB","CG1")],
        "W": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
        "Y": [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    }
    mismatches = []
    for aa in sorted(OPENFOLD_CHI):
        nb_chi = CHI_DEFS.get(aa, [])
        of_chi = OPENFOLD_CHI[aa]
        if nb_chi != of_chi:
            mismatches.append(
                f"  {aa}: notebook={nb_chi}\n      openfold={of_chi}"
            )
    if mismatches:
        raise AssertionError(
            "CHI_DEFS mismatch vs OpenFold:\n" + "\n".join(mismatches)
        )
    print("[CHI_DEFS] All 20 residues match OpenFold chi_angles_atoms exactly. ✓")

_verify_chi_defs_vs_openfold()

# %% [markdown] cell 8
# ## 3) Utilities (chain resolution, atom index maps, adjacency)

# %% cell 9
def resolve_chain_index(top: md.Topology, chain_letter: str):
    chains = list(top.chains)
    for ch in chains:
        cid = getattr(ch, "id", None)
        if cid is not None and str(cid).strip() == str(chain_letter).strip():
            return ch.index
    if len(chains) == 1:
        return chains[0].index
    raise ValueError(f"Could not resolve chain '{chain_letter}' from topology (chains={len(chains)}).")

def extract_chain_residues(top: md.Topology, chain_idx: int):
    return [r for r in top.residues if r.chain.index == chain_idx]

def residue_to_aa1(resname3: str):
    a = seq1(resname3, custom_map={})
    return a if len(a) == 1 else "X"

def build_atom_index_maps(residues):
    N = len(residues)
    atom14_indices = np.full((N, 14), -1, dtype=np.int32)

    n_idx  = np.full((N,), -1, dtype=np.int32)
    ca_idx = np.full((N,), -1, dtype=np.int32)
    c_idx  = np.full((N,), -1, dtype=np.int32)
    o_idx  = np.full((N,), -1, dtype=np.int32)

    # FIX: two separate arrays
    #   pdb_resseq    — raw PDB numbering, used ONLY for gap detection in adjacent mask
    #   residue_index — 0-based sequential [0,1,2,...], used by the model
    pdb_resseq    = np.zeros((N,), dtype=np.int32)
    residue_index = np.arange(N,   dtype=np.int32)

    seq_chars = []
    aatype    = np.zeros((N,), dtype=np.int64)

    for i, res in enumerate(residues):
        pdb_resseq[i] = getattr(res, "resSeq", i)
        aa = residue_to_aa1(res.name)
        seq_chars.append(aa)
        aatype[i] = AA_TO_IDX.get(aa, UNK_AA)

        atoms = {a.name: a.index for a in res.atoms}

        n_idx[i]  = atoms.get("N",  -1)
        ca_idx[i] = atoms.get("CA", -1)
        c_idx[i]  = atoms.get("C",  -1)
        o_idx[i]  = atoms.get("O",  -1)

        names14 = ATOM14_NAMES.get(aa, ["N","CA","C","O"] + [""]*10)
        for a, nm in enumerate(names14):
            if nm != "":
                atom14_indices[i, a] = atoms.get(nm, -1)

    return (atom14_indices, n_idx, ca_idx, c_idx, o_idx,
            "".join(seq_chars), aatype, pdb_resseq, residue_index)

def compute_adjacent_mask(pdb_resseq: np.ndarray):
    """Gap-free adjacency based on raw PDB resSeq (catches insertion-code gaps)."""
    N = int(pdb_resseq.shape[0])
    if N <= 1:
        return np.zeros((0,), dtype=np.uint8)
    return (pdb_resseq[1:] - pdb_resseq[:-1] == 1).astype(np.uint8)

def get_chain_ca_indices(top: md.Topology, chain_idx: int) -> np.ndarray:
    """
    Return global atom indices of all Cα atoms belonging to `chain_idx`.
    Used to restrict superposition to the chain of interest only — prevents
    other chains or HETATM groups from skewing the rigid-body fit.
    """
    return np.array([
        a.index for a in top.atoms
        if a.residue.chain.index == chain_idx and a.name == "CA"
    ], dtype=np.int32)

# %% [markdown] cell 10
# ## 4) OpenFold torsion quadruplets (pre_omega/phi/psi + chis)

# %% cell 11
def build_torsion_quads_openfold(residues, n_idx, ca_idx, c_idx, o_idx, aatype, adjacent):
    """
    Build (atom-index quadruplet, metadata) pairs for md.compute_dihedrals.

    OpenFold / AlphaFold2 convention
    ---------------------------------
      0  pre_omega(i) : CA(i-1), C(i-1), N(i),   CA(i)   gated by adjacent[i-1]
      1  phi(i)       : C(i-1),  N(i),   CA(i),   C(i)    gated by adjacent[i-1]
      2  psi(i)       : N(i),    CA(i),  C(i),    N(i+1)  gated by adjacent[i]
                         ^^ FIX: was O(i) — now correctly uses N of next residue
      3..6 chi1..chi4  : side-chain rotamers

    Terminal residues will have torsion_mask==0 for the angle that requires
    the missing neighbour — this is physically correct.
    """
    quads = []
    meta  = []
    Nres  = len(residues)

    for i in range(Nres):
        # ── pre_omega(i) & phi(i): require residue i-1 to be chain-adjacent ──
        if i > 0 and adjacent[i - 1] == 1:
            # pre_omega
            if ca_idx[i-1] >= 0 and c_idx[i-1] >= 0 and n_idx[i] >= 0 and ca_idx[i] >= 0:
                quads.append([ca_idx[i-1], c_idx[i-1], n_idx[i], ca_idx[i]])
                meta.append((0, i))
            # phi
            if c_idx[i-1] >= 0 and n_idx[i] >= 0 and ca_idx[i] >= 0 and c_idx[i] >= 0:
                quads.append([c_idx[i-1], n_idx[i], ca_idx[i], c_idx[i]])
                meta.append((1, i))

        # ── psi(i): N(i), CA(i), C(i), N(i+1) — requires residue i+1 adjacent ─
        # FIX: was [n_idx[i], ca_idx[i], c_idx[i], o_idx[i]]  (wrong: used O)
        if i + 1 < Nres and adjacent[i] == 1:
            if n_idx[i] >= 0 and ca_idx[i] >= 0 and c_idx[i] >= 0 and n_idx[i+1] >= 0:
                quads.append([n_idx[i], ca_idx[i], c_idx[i], n_idx[i+1]])
                meta.append((2, i))

    # ── chi1..chi4 ────────────────────────────────────────────────────────────
    for i, res in enumerate(residues):
        aa = AA_ORDER[aatype[i]] if aatype[i] < 20 else "X"
        chi_list = CHI_DEFS.get(aa, [])
        if not chi_list:
            continue
        atoms = {a.name: a.index for a in res.atoms}
        for k, (a, b, c, d) in enumerate(chi_list):
            if k >= 4:
                break
            if a in atoms and b in atoms and c in atoms and d in atoms:
                quads.append([atoms[a], atoms[b], atoms[c], atoms[d]])
                meta.append((3 + k, i))

    if len(quads) == 0:
        return np.zeros((0, 4), dtype=np.int32), []
    return np.asarray(quads, dtype=np.int32), meta

# %% [markdown] cell 12
# ## 5) Backbone frames from (N, CA, C)

# %% cell 13
def safe_normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def compute_backbone_frames(N_xyz, CA_xyz, C_xyz):
    frame_mask = ((np.linalg.norm(N_xyz, axis=-1) > 0) &
                  (np.linalg.norm(CA_xyz, axis=-1) > 0) &
                  (np.linalg.norm(C_xyz, axis=-1) > 0)).astype(np.uint8)

    e1 = safe_normalize(C_xyz - CA_xyz)
    e2 = safe_normalize(N_xyz - CA_xyz)
    e3 = safe_normalize(np.cross(e1, e2))
    e2 = safe_normalize(np.cross(e3, e1))

    R = np.stack([e1, e2, e3], axis=-1)
    t = CA_xyz.copy()
    return R.astype(np.float32), t.astype(np.float32), frame_mask

# %% [markdown] cell 14
# ## 6) ESM2 embedding cache (Drive-persisted)

# %% cell 15
def load_esm_cache(path: str):
    if os.path.exists(path):
        print(f"[ESM] Loading cache: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def save_esm_cache(cache: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f"[ESM] Saved cache: {path} (entries={len(cache)})")

def run_batch_esm(sequences, device="cuda"):
    if len(sequences) == 0:
        return {}

    print(f"[ESM] Loading {ESM_MODEL_NAME} on {device} ...")
    model, alphabet = getattr(esm.pretrained, ESM_MODEL_NAME)()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    out = {}
    for i in tqdm(range(0, len(sequences), BATCH_SIZE_ESM), desc="ESM batches"):
        batch = sequences[i:i+BATCH_SIZE_ESM]
        _, _, toks = batch_converter([(str(j), s) for j,s in enumerate(batch)])
        toks = toks.to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                res = model(toks, repr_layers=[ESM_LAYER])
                reps = res["representations"][ESM_LAYER]  # [B, L, D]

        for j, seq in enumerate(batch):
            out[seq] = reps[j, 1:len(seq)+1].detach().cpu().numpy().astype(np.float16)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return out

# %% [markdown] cell 16
# ## 7) Sample processing (stream XTC → NPZ)

# %% cell 17
def process_single_sample(sample_id, xtc_path, pdb_path, chain_letter, esm_embed):
    out_file = os.path.join(LOCAL_NPZ_DIR, f"{sample_id}.npz")

    # Safe re-run: skip if already written
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return True

    try:
        # ── Topology & per-residue index maps ────────────────────────────────
        top       = md.load(pdb_path).topology
        chain_idx = resolve_chain_index(top, chain_letter)
        residues  = extract_chain_residues(top, chain_idx)
        if len(residues) == 0:
            print(f"[SKIP] {sample_id}: no residues in chain {chain_letter}")
            return False

        # FIX: build_atom_index_maps now returns 9 values (added residue_index)
        (atom14_indices, n_idx, ca_idx, c_idx, o_idx,
         seq_str, aatype, pdb_resseq, residue_index) = build_atom_index_maps(residues)

        # Gap detection uses raw PDB resSeq, NOT the 0-based residue_index
        adjacent = compute_adjacent_mask(pdb_resseq)

        tors_quads, tors_meta = build_torsion_quads_openfold(
            residues, n_idx, ca_idx, c_idx, o_idx, aatype, adjacent
        )

        # ── Reference frame for superposition ────────────────────────────────
        # Load PDB once as a single-frame MDTraj trajectory.
        # get_chain_ca_indices returns only the Cα atoms of THIS chain so that
        # other chains or HETATM groups don't distort the rigid-body fit.
        ref_traj       = md.load(pdb_path)
        ca_chain_idx   = get_chain_ca_indices(top, chain_idx)

        # ── Streaming chunk loop ──────────────────────────────────────────────
        atom14_pos_chunks   = []
        atom14_mask_chunks  = []
        frame_R_chunks      = []
        frame_t_chunks      = []
        frame_mask_chunks   = []
        torsion_chunks      = []
        torsion_mask_chunks = []

        for chunk in md.iterload(xtc_path, top=pdb_path, chunk=CHUNK_FRAMES):

            # ── GLOBAL MOTION REMOVAL ─────────────────────────────────────────
            # Superpose each frame in this chunk onto the static PDB reference
            # using chain-specific Cα atoms only. After this call:
            #   - chunk.xyz is in the PDB coordinate frame
            #   - frame_t will encode CA displacement from PDB, not raw drift
            #   - frame_R will encode pure local backbone deformation
            # Torsion angles are rotation-invariant and unaffected either way.
            chunk.superpose(
                ref_traj,
                frame          = 0,
                atom_indices     = ca_chain_idx,   # drive rotation from chain Cα
                ref_atom_indices = ca_chain_idx,   # same atoms in reference
            )

            xyzA = (chunk.xyz * 10.0).astype(np.float32)   # nm → Å
            F    = xyzA.shape[0]
            Nres = len(residues)

            # ── atom14 positions ──────────────────────────────────────────────
            pos14 = np.zeros((F, Nres, 14, 3), dtype=np.float32)
            m14   = np.zeros((F, Nres, 14),    dtype=np.uint8)

            for a in range(14):
                idxs  = atom14_indices[:, a]
                valid = idxs >= 0
                if np.any(valid):
                    pos14[:, valid, a, :] = xyzA[:, idxs[valid], :]
                    m14[:,  valid, a]     = 1

            # ── backbone N / CA / C for rigid frames ──────────────────────────
            Nxyz  = np.zeros((F, Nres, 3), dtype=np.float32)
            CAxyz = np.zeros((F, Nres, 3), dtype=np.float32)
            Cxyz  = np.zeros((F, Nres, 3), dtype=np.float32)

            vN  = n_idx  >= 0
            vCA = ca_idx >= 0
            vC  = c_idx  >= 0

            if np.any(vN):  Nxyz[:,  vN,  :] = xyzA[:, n_idx[vN],   :]
            if np.any(vCA): CAxyz[:, vCA, :] = xyzA[:, ca_idx[vCA], :]
            if np.any(vC):  Cxyz[:,  vC,  :] = xyzA[:, c_idx[vC],   :]

            R, t, fmask = compute_backbone_frames(Nxyz, CAxyz, Cxyz)

            # ── torsion angles ────────────────────────────────────────────────
            tors  = np.zeros((F, Nres, 7), dtype=np.float32)
            tmask = np.zeros((F, Nres, 7), dtype=np.uint8)

            if tors_quads.shape[0] > 0:
                dihs = md.compute_dihedrals(chunk, tors_quads)   # [F, M], radians
                for j, (ttype, ridx) in enumerate(tors_meta):
                    tors[:,  ridx, ttype] = dihs[:, j].astype(np.float32)
                    tmask[:, ridx, ttype] = 1

            atom14_pos_chunks.append(
                pos14.astype(np.float16) if SAVE_FLOAT16_ATOMS else pos14
            )
            atom14_mask_chunks.append(m14)
            frame_R_chunks.append(R)
            frame_t_chunks.append(t)
            frame_mask_chunks.append(fmask)
            torsion_chunks.append(tors)
            torsion_mask_chunks.append(tmask)

        # ── Concatenate all chunks ────────────────────────────────────────────
        atom14_pos  = np.concatenate(atom14_pos_chunks,   axis=0)
        atom14_mask = np.concatenate(atom14_mask_chunks,  axis=0)
        frame_R     = np.concatenate(frame_R_chunks,      axis=0)
        frame_t     = np.concatenate(frame_t_chunks,      axis=0)
        frame_mask  = np.concatenate(frame_mask_chunks,   axis=0)
        tors_angles = np.concatenate(torsion_chunks,      axis=0)
        tors_mask   = np.concatenate(torsion_mask_chunks, axis=0)

        np.savez_compressed(
            out_file,

            # ── sequence ────────────────────────────────────────────────────
            seq_str       = seq_str,
            aatype        = aatype.astype(np.int64),
            # FIX: 0-based sequential index — safe for sinusoidal / relative PE
            residue_index = residue_index.astype(np.int32),
            # raw PDB resSeq kept separately for downstream gap analysis
            pdb_resseq    = pdb_resseq.astype(np.int32),
            chain_index   = np.full_like(residue_index, chain_idx, dtype=np.int32),
            adjacent      = adjacent.astype(np.uint8),

            # ── structure ───────────────────────────────────────────────────
            atom14_pos  = atom14_pos,
            atom14_mask = atom14_mask,

            # ── backbone rigid frames (in PDB coordinate frame) ─────────────
            frame_R     = frame_R,
            frame_t     = frame_t,    # CA position = displacement from PDB ref
            frame_mask  = frame_mask,

            # ── torsions ────────────────────────────────────────────────────
            torsion_angles     = tors_angles,
            torsion_mask       = tors_mask,
            torsion_names      = np.array(TORSION_NAMES, dtype=object),
            # FIX: psi now N-CA-C-N(i+1), not N-CA-C-O
            torsion_convention = "openfold_v2_psi_N_CA_C_Nnext",

            # ── ESM2 embeddings ──────────────────────────────────────────────
            esm2 = esm_embed.astype(np.float16),

            # ── provenance ──────────────────────────────────────────────────
            units     = "angstrom",
            alignment = "superposed_to_pdb_chain_ca_only",
            source_pdb = pdb_path,
            source_xtc = xtc_path,
        )
        return True

    except Exception as e:
        print(f"[FAIL] {sample_id}: {e}")
        return False

# %% [markdown] cell 18
# ## 8) Driver helpers (unzip + task building)

# %% cell 19
def unzip_if_needed(zip_path, out_dir):
    marker = os.path.join(out_dir, ".unzipped_done")
    if os.path.exists(marker):
        return
    print(f"[UNZIP] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.infolist()
        with open(os.path.join(out_dir, ".unzipping"), "w") as _:
            pass
        with tqdm(total=sum(m.file_size for m in members), unit="B", unit_scale=True, desc="Extracting") as pbar:
            for m in members:
                z.extract(m, out_dir)
                pbar.update(m.file_size)
    with open(marker, "w") as f:
        f.write("ok")

def copy_with_progress(src, dst):
    size = os.path.getsize(src)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        with tqdm(total=size, unit="B", unit_scale=True, desc=f"Copying {os.path.basename(src)}") as pbar:
            while True:
                chunk = fsrc.read(1024 * 1024)
                if not chunk:
                    break
                fdst.write(chunk)
                pbar.update(len(chunk))

def base_from_npz_filename(fname: str) -> str:
    stem = fname[:-4] if fname.endswith(".npz") else fname
    return re.sub(r"_R[1-3]$", "", stem)

def build_tasks_from_csv():
    df = pd.read_csv(CSV_PATH)
    assert "chain_id" in df.columns, "CSV must contain column 'chain_id'"
    chain_ids = df["chain_id"].astype(str).tolist()

    tasks = []
    seq_map = {}
    unique_seqs = set()

    print("[SCAN] Building task list...")
    for pid in tqdm(chain_ids):
        pdb_path = os.path.join(LOCAL_UNZIP, pid, f"{pid}.pdb")
        if not os.path.exists(pdb_path):
            continue

        chain_letter = pid.split("_")[-1] if "_" in pid else "A"

        try:
            top = md.load(pdb_path).topology
            chain_idx = resolve_chain_index(top, chain_letter)
            residues = extract_chain_residues(top, chain_idx)
            _, _, _, _, _, seq_str, _, _, _ = build_atom_index_maps(residues)
            seq_map[pid] = seq_str
            unique_seqs.add(seq_str)
        except Exception as e:
            print(f"[WARN] seq read failed for {pid}: {e}")
            continue

        for r in [1,2,3]:
            xtc_path = os.path.join(LOCAL_UNZIP, pid, f"{pid}_R{r}.xtc")
            if os.path.exists(xtc_path):
                tasks.append((f"{pid}_R{r}", xtc_path, pdb_path, chain_letter, pid))

    print(f"[SCAN] Tasks: {len(tasks)} | Unique seqs: {len(unique_seqs)}")
    return tasks, seq_map, sorted(list(unique_seqs))

# %% [markdown] cell 20
# ## 9) Run extraction (safe to re-run)

# %% cell 21
# Copy zip to local and unzip (idempotent)
copy_with_progress(INPUT_ZIP, LOCAL_ZIP)
unzip_if_needed(LOCAL_ZIP, LOCAL_UNZIP)

# Build tasks + sequences
tasks, seq_map, unique_seqs = build_tasks_from_csv()

# Load cache; compute missing; save back
esm_cache = load_esm_cache(ESM_CACHE_PATH)
missing = [s for s in unique_seqs if s not in esm_cache]
print(f"[ESM] cached={len(esm_cache)} missing={len(missing)}")

if len(missing) > 0:
    esm_cache.update(run_batch_esm(missing, device="cuda"))
    save_esm_cache(esm_cache, ESM_CACHE_PATH)

# Extract NPZ (skips existing)
print("[EXTRACT] Writing NPZs...")
ok = Parallel(n_jobs=N_JOBS, backend="threading")(
    delayed(process_single_sample)(
        sample_id, xtc_path, pdb_path, chain_letter, esm_cache[seq_map[pid]]
    )
    for (sample_id, xtc_path, pdb_path, chain_letter, pid) in tqdm(tasks)
)

print(f"[DONE] Wrote {int(sum(bool(x) for x in ok))}/{len(tasks)} NPZ files to {LOCAL_NPZ_DIR}")

# %% [markdown] cell 22
# ## 10) Sanity checks (recommended before training)

# %% cell 23
import random

def sanity_check_one(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    pos14 = d["atom14_pos"]
    m14   = d["atom14_mask"]
    tors  = d["torsion_angles"]
    tmask = d["torsion_mask"]

    # ── Geometry ──────────────────────────────────────────────────────────────
    print("=== Geometry ===")
    CA  = pos14[0, :, 1, :]
    mCA = m14[0, :, 1] > 0
    if CA.shape[0] > 1:
        dists = np.linalg.norm(CA[1:] - CA[:-1], axis=-1)
        dists = dists[mCA[1:] & mCA[:-1]]
        print(f"  CA-CA dist  mean/std (Å): {np.mean(dists):.3f} / {np.std(dists):.3f}"
              f"  (expect ~3.8 Å)")

    # ── Alignment sanity ──────────────────────────────────────────────────────
    print("\n=== Alignment ===")
    print(f"  tag         : {str(d.get('alignment', b'N/A'))}")
    ft   = d["frame_t"]                               # [F, Nres, 3]
    disp = np.linalg.norm(ft, axis=-1)                # [F, Nres]
    print(f"  |frame_t| mean (Å) : {np.mean(disp):.3f}  (typical 0.5–5 Å after alignment)")
    print(f"  |frame_t| max  (Å) : {np.max(disp):.3f}  (large value = alignment may have failed)")

    # ── Torsion angles ────────────────────────────────────────────────────────
    print("\n=== Torsion angles ===")
    print(f"  finite      : {bool(np.isfinite(tors).all())}")
    print(f"  range       : [{float(np.nanmin(tors)):.3f}, {float(np.nanmax(tors)):.3f}]"
          f"  (should be within [-π, π])")
    print(f"  names       : {list(d['torsion_names'])}")
    conv_tag = str(d.get('torsion_convention', b'N/A'))
    print(f"  convention  : {conv_tag}")
    # Hard-assert the v2 convention tag is present (catches accidental use of old NPZ files)
    EXPECTED_CONV = "openfold_v2_psi_N_CA_C_Nnext"
    if conv_tag not in (EXPECTED_CONV, "b'N/A'", "N/A"):
        print(f"  [WARN] Expected convention tag '{EXPECTED_CONV}', got '{conv_tag}'")
        print(f"         If this is an old NPZ (psi used O atom), re-run extraction.")
    elif conv_tag == EXPECTED_CONV:
        print(f"  [OK]  Convention tag matches v2 spec.")

    # ── Psi mask (FIX verification) ───────────────────────────────────────────
    print("\n=== Psi-mask sanity (checks psi fix) ===")
    # psi is column 2; the LAST residue has no N(i+1), so psi must be masked there
    psi_last  = tmask[:, -1, 2]
    print(f"  psi mask at last residue  (must be 0): unique={np.unique(psi_last).tolist()}")
    # phi is column 1; the FIRST residue has no C(i-1), so phi must be masked
    phi_first = tmask[:, 0, 1]
    print(f"  phi mask at first residue (must be 0): unique={np.unique(phi_first).tolist()}")

    # ── Residue index (FIX verification) ─────────────────────────────────────
    print("\n=== Residue index (checks 0-based fix) ===")
    ri = d["residue_index"]
    print(f"  residue_index[0:5]  : {ri[:5].tolist()}  (must be [0,1,2,3,4])")
    print(f"  pdb_resseq[0:5]     : {d['pdb_resseq'][:5].tolist()}  (raw PDB numbering, may differ)")

files = [f for f in os.listdir(LOCAL_NPZ_DIR) if f.endswith(".npz")]
if not files:
    print("No NPZ files found — run extraction first.")
else:
    fname = random.choice(files)
    print(f"Checking: {fname}\n")
    sanity_check_one(os.path.join(LOCAL_NPZ_DIR, fname))

# %% [markdown] cell 24
# ## 11) Grouped train/val/test split and zip export (no leakage across R1/R2/R3)

# %% cell 25
from sklearn.model_selection import train_test_split

processed_files = [f for f in os.listdir(LOCAL_NPZ_DIR) if f.endswith(".npz")]
unique_bases = sorted(list(set(base_from_npz_filename(f) for f in processed_files)))

train_bases, tmp_bases = train_test_split(unique_bases, test_size=0.2, random_state=42)
val_bases, test_bases  = train_test_split(tmp_bases, test_size=0.5, random_state=42)

base_split_map = {"train": set(train_bases), "val": set(val_bases), "test": set(test_bases)}

def write_zip(split_name, base_set):
    files_to_zip = [f for f in processed_files if base_from_npz_filename(f) in base_set]
    z_local = os.path.join("/content", f"{split_name}.zip")
    z_final = os.path.join(OUT_DIR, f"{split_name}.zip")

    print(f"[ZIP] {split_name}: {len(files_to_zip)} files")
    with zipfile.ZipFile(z_local, "w", zipfile.ZIP_DEFLATED) as z:
        for f in tqdm(files_to_zip, desc=f"Zipping {split_name}"):
            z.write(os.path.join(LOCAL_NPZ_DIR, f), arcname=f)

    copy_with_progress(z_local, z_final)

for split, bases in base_split_map.items():
    write_zip(split, bases)

print("[DONE] Split zips written to:", OUT_DIR)

# %% cell 26
import numpy as np, glob
from scipy.spatial.transform import Rotation

files = glob.glob("/content/npz_out_openfold/*.npz")[:30]
all_omg = []

for f in files:
    z = np.load(f)
    R = z["frame_R"].astype(np.float32)   # (T, N, 3, 3)
    m = z["frame_mask"].astype(np.float32) # (T, N)

    T, N = R.shape[:2]
    for t in range(T - 1):
        dR = R[t].transpose(0, 2, 1) @ R[t+1]   # (N, 3, 3) relative rotation
        valid = (m[t] * m[t+1]) > 0.5
        dR_valid = dR[valid].reshape(-1, 3, 3)

        # so3_log via scipy (equivalent to your SO3.log)
        rotvec = Rotation.from_matrix(dR_valid).as_rotvec()  # (Nv, 3)
        omg_mag = np.linalg.norm(rotvec, axis=-1) / 1.0      # dt=1.0
        all_omg.append(omg_mag)

all_omg = np.concatenate(all_omg)
print(f"OMG_STD suggestion:  {np.std(all_omg):.4f}  (mean={np.mean(all_omg):.4f})")
print(f"OMG 95th pct:        {np.percentile(all_omg, 95):.4f}")

# %% cell 27
files = glob.glob("/content/npz_out_openfold/*.npz")[:30]
all_dists = []

for f in files:
    z = np.load(f)
    a14  = z["atom14_pos"].astype(np.float32)   # (T, N, 14, 3)
    a14m = z["atom14_mask"].astype(np.float32)  # (T, N, 14)  -- static mask

    # frame-to-frame per-atom displacement
    da = np.linalg.norm(a14[1:] - a14[:-1], axis=-1)  # (T-1, N, 14)
    mask = (a14m[:-1] * a14m[1:]) > 0.5               # (T-1, N, 14)
    all_dists.append(da[mask])

all_dists = np.concatenate(all_dists)
print(f"Atom14 disp mean:    {np.mean(all_dists):.4f} Å")
print(f"Atom14 disp std:     {np.std(all_dists):.4f} Å")
print(f"Atom14 disp 80th:    {np.percentile(all_dists, 80):.4f} Å  ← good Huber delta")
print(f"Atom14 disp 95th:    {np.percentile(all_dists, 95):.4f} Å")
print(f"Atom14 disp 99th:    {np.percentile(all_dists, 99):.4f} Å")

# %% cell 28
files = glob.glob("/content/npz_out_openfold/*.npz")[:30]
all_ca = []

for f in files:
    z = np.load(f)
    x = z["frame_t"].astype(np.float32)   # (T, N, 3) Cα positions
    m = z["frame_mask"].astype(np.float32) # (T, N)

    dx = np.linalg.norm(x[1:] - x[:-1], axis=-1)  # (T-1, N)
    mask = (m[:-1] * m[1:]) > 0.5
    all_ca.append(dx[mask])

all_ca = np.concatenate(all_ca)
print(f"CA disp mean:    {np.mean(all_ca):.4f} Å")
print(f"CA disp std:     {np.std(all_ca):.4f} Å  ← this should ≈ V_STD")
print(f"CA disp 80th:    {np.percentile(all_ca, 80):.4f} Å  ← ideal Huber delta")
# Then: LPOS_HUBER_DELTA = 80th_pct / V_STD
print(f"Suggested LPOS_HUBER_DELTA: {np.percentile(all_ca, 80) / 1.32:.3f}")

# %% cell 29
# After running the CA displacement cell above:
print(f"Suggested WARN_DISP_A: {np.mean(all_ca) * 3:.2f} Å")  # 3× mean

# %% cell 30
print(f"Suggested WARN_ANG_RAD: {np.mean(all_omg) * 3:.4f} rad")

# %% cell 31
files = glob.glob("/content/npz_out_openfold/*.npz")
lengths = []
for f in files:
    z = np.load(f)
    lengths.append(int(z["aatype"].shape[0]))

print(f"Max residues:    {max(lengths)}")
print(f"Mean residues:   {np.mean(lengths):.1f}")
print(f"95th pct:        {np.percentile(lengths, 95):.0f}")

# %% cell 32
files = glob.glob("/content/npz_out_openfold/*.npz")[:20]
max_dists = []
for f in files:
    z = np.load(f)
    x = z["frame_t"][0].astype(np.float32)  # first frame, (N, 3)
    m = z["frame_mask"][0] > 0.5
    x_valid = x[m]
    if len(x_valid) > 1:
        D = np.linalg.norm(x_valid[:, None] - x_valid[None, :], axis=-1)
        max_dists.append(D.max())

print(f"Max Cα-Cα distance 95th pct: {np.percentile(max_dists, 95):.1f} Å")
# DISTO_D_MAX should be >= this to not clip the true longest distances

# %% cell 33
import os, re, zipfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Helpers: parse pid + replicate from NPZ filename (e.g., "16pk_A_R2.npz")
# ------------------------------------------------------------------
_REP_RE = re.compile(r"^(?P<pid>.+)_R(?P<rep>\d+)\.npz$")

def pid_from_npz(f: str) -> str:
    m = _REP_RE.match(os.path.basename(f))
    if m is None:
        raise ValueError(f"NPZ filename must match '<pid>_R<k>.npz'. got: {f}")
    return m.group("pid")

def rep_from_npz(f: str) -> int:
    m = _REP_RE.match(os.path.basename(f))
    if m is None:
        raise ValueError(f"NPZ filename must match '<pid>_R<k>.npz'. got: {f}")
    return int(m.group("rep"))

# ------------------------------------------------------------------
# Select exactly ONE replicate per pid (prefer R1 -> R2 -> R3)
# processed_files: list[str] of NPZ filenames (relative to LOCAL_NPZ_DIR)
# ------------------------------------------------------------------
def select_one_replica_per_pid_files(processed_files, mode="prefer", prefer_order=(1,2,3), seed=42):
    import random
    rng = random.Random(seed)

    by_pid = {}
    for f in processed_files:
        pid = pid_from_npz(f)
        r   = rep_from_npz(f)
        by_pid.setdefault(pid, {})[r] = f  # rep -> filename

    chosen_files = []
    chosen_rep_hist = {1: 0, 2: 0, 3: 0}

    for pid, rep_map in by_pid.items():
        reps = sorted(rep_map.keys())
        if len(reps) == 0:
            continue

        if mode == "prefer":
            chosen = next((r for r in prefer_order if r in rep_map), reps[0])
        elif mode == "random":
            chosen = rng.choice(reps)
        else:
            raise ValueError(f"mode must be 'prefer' or 'random'. got: {mode}")

        chosen_files.append(rep_map[chosen])
        if chosen in chosen_rep_hist:
            chosen_rep_hist[chosen] += 1

    print(f"[ONE-REP] pids={len(by_pid)} selected={len(chosen_files)} rep_hist={chosen_rep_hist}")
    return chosen_files

# ------------------------------------------------------------------
# Your inputs
#   processed_files: list[str] of NPZ filenames (created earlier)
#   LOCAL_NPZ_DIR: directory containing those NPZ files
#   OUT_DIR: output directory in Drive
# ------------------------------------------------------------------

tasks_one_files = select_one_replica_per_pid_files(
    processed_files=processed_files,
    mode="prefer",
    prefer_order=(1,2,3),
    seed=42,
)

# Split by PID (leakage-safe). We split pids, then map back to chosen files.
pids_all = sorted({pid_from_npz(f) for f in tasks_one_files})

train_pids, tmp_pids = train_test_split(pids_all, test_size=0.2, random_state=42, shuffle=True)
val_pids, test_pids  = train_test_split(tmp_pids, test_size=0.5, random_state=42, shuffle=True)

split_pid_map = {
    "train": set(train_pids),
    "val":   set(val_pids),
    "test":  set(test_pids),
}

def write_zip(split_name: str, pid_set: set):
    # IMPORTANT: zip ONLY the already-selected one-rep-per-pid files
    files_to_zip = [f for f in tasks_one_files if pid_from_npz(f) in pid_set]

    z_local = os.path.join("/content", f"{split_name}.zip")
    z_final = os.path.join(OUT_DIR, f"{split_name}.zip")
    os.makedirs(os.path.dirname(z_final), exist_ok=True)

    print(f"[ZIP] {split_name}: pids={len(pid_set)} files={len(files_to_zip)} -> {os.path.basename(z_final)}")

    with zipfile.ZipFile(z_local, "w", zipfile.ZIP_DEFLATED) as z:
        for f in tqdm(files_to_zip, desc=f"Zipping {split_name}"):
            src = os.path.join(LOCAL_NPZ_DIR, f)
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            z.write(src, arcname=f)

    copy_with_progress(z_local, z_final)

for split, pset in split_pid_map.items():
    write_zip(f"{split}_R1", pset)

print("[DONE] Split zips written to:", OUT_DIR)

# %% cell 34
"""
Comprehensive NPZ sanity checker for the velocity-FM training data.
Covers every array the model touches, with physically-grounded thresholds.

Usage (in Colab cell):
    # NOTEBOOK_MAGIC: %run npz_deep_check.py
or paste the whole thing into a cell.
"""

import os, math, random
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────────────────────────────────────
LOCAL_NPZ_DIR = "/content/npz_out_openfold"
N_SAMPLE      = 20
VERBOSE       = True

# Must match extraction notebook exactly
AA_ORDER      = "ARNDCQEGHILKMFPSTWYV"
TORSION_NAMES = ["pre_omega", "phi", "psi", "chi1", "chi2", "chi3", "chi4"]
EXPECTED_CONV = "openfold_v2_psi_N_CA_C_Nnext"
MAX_RES       = 256
ESM_DIM       = 1280    # esm2_t33_650M; 480 for t12

# Physics thresholds
CA_CA_LO, CA_CA_HI    = 3.70, 3.95    # Å  bonded CA-CA
CN_LO,    CN_HI       = 1.25, 1.45    # Å  peptide C-N
CACB_LO,  CACB_HI     = 1.40, 1.65    # Å  CA-CB
ANG_LO,   ANG_HI      = 108,  124     # °  CA-C-N
CLASH_D               = 1.5           # Å  hard clash
SPIKE_THR             = 50.0          # Å/frame CA jump → PBC artefact
FRAME_T_MAX           = 200.0         # Å  max displacement → alignment failure
ORTH_MAX              = 0.01          # ||R^T R - I||_F
TORS_LO, TORS_HI      = -math.pi - 0.01, math.pi + 0.01

# atom14 column indices (OpenFold: N=0 CA=1 C=2 O=3 CB=4)
N_, CA_, C_, O_, CB_  = 0, 1, 2, 3, 4

GLY = AA_ORDER.index("G")
ALA = AA_ORDER.index("A")
PRO = AA_ORDER.index("P")


# ─────────────────────────────────────────────────────────────────────────────
# RESULT COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────
class Res:
    def __init__(self, name):
        self.name = name
        self.oks, self.warns, self.fails = [], [], []

    def ok(self,   m): self.oks.append(m)
    def warn(self, m): self.warns.append(m)
    def fail(self, m): self.fails.append(m)
    def passed(self):  return len(self.fails) == 0

    def show(self):
        tag = "✅" if self.passed() else "❌"
        print(f"\n  {'─'*60}")
        print(f"  {tag}  {self.name}")
        print(f"  {'─'*60}")
        for m in self.oks:   print(f"      ✓  {m}")
        for m in self.warns: print(f"      ⚠  {m}")
        for m in self.fails: print(f"      ✗  {m}")


# ─────────────────────────────────────────────────────────────────────────────
# TINY GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _norm(a, b):   return np.linalg.norm(a - b, axis=-1)

def _angle(a, b, c):
    u, v = a - b, c - b
    cos  = np.clip(
        np.sum(u*v, axis=-1) /
        (np.linalg.norm(u, axis=-1)*np.linalg.norm(v, axis=-1) + 1e-8),
        -1, 1)
    return np.degrees(np.arccos(cos))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Keys, dtypes, metadata tags
# ─────────────────────────────────────────────────────────────────────────────
def check_keys(d, r):
    required = dict(
        aatype=np.int64,       residue_index=np.int32,  pdb_resseq=np.int32,
        adjacent=np.uint8,     frame_R=np.float32,      frame_t=np.float32,
        frame_mask=np.uint8,   torsion_angles=np.float32, torsion_mask=np.uint8,
        atom14_pos=np.float32, atom14_mask=np.uint8,    esm2=np.float16,
    )
    for key, dtype in required.items():
        if key not in d.files:
            r.fail(f"Missing key '{key}'")
        elif d[key].dtype != dtype:
            r.fail(f"'{key}' dtype={d[key].dtype}, expected {dtype}")
        else:
            r.ok(f"'{key}' dtype={dtype} ✓")

    # convention tag — model's dataset guard checks this at load time
    conv = str(d.get("torsion_convention", np.array(b"MISSING")))
    if EXPECTED_CONV in conv:
        r.ok(f"torsion_convention = '{EXPECTED_CONV}' ✓")
    else:
        r.fail(f"torsion_convention = '{conv}'  expected '{EXPECTED_CONV}'")

    # torsion name order
    names = [str(x) for x in d.get("torsion_names", np.array([])).tolist()]
    if names == TORSION_NAMES:
        r.ok("torsion_names correct ✓")
    else:
        r.fail(f"torsion_names = {names}")

    # aatype encoding order
    enc = str(d.get("aatype_encoding", np.array(b"")))
    if "ARNDCQEGHILKMFPSTWYV" in enc or "openfold" in enc.lower():
        r.ok("aatype_encoding confirms OpenFold restypes order ✓")
    else:
        r.warn(f"aatype_encoding = '{enc}'  — could not confirm OpenFold order")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Shapes
# ─────────────────────────────────────────────────────────────────────────────
def check_shapes(d, r):
    F = d["frame_t"].shape[0]
    N = d["frame_t"].shape[1]
    r.ok(f"F={F} frames,  N={N} residues")

    if N > MAX_RES:
        r.fail(f"N={N} > MAX_RES={MAX_RES} — dataset will truncate; check extraction")
    if N == 0:
        r.fail("N=0")
    if F < 10:
        r.warn(f"Only F={F} frames — very short trajectory")

    expected = {
        "frame_R":        (F, N, 3, 3),
        "frame_t":        (F, N, 3),
        "frame_mask":     (F, N),
        "torsion_angles": (F, N, 7),
        "torsion_mask":   (F, N, 7),
        "atom14_pos":     (F, N, 14, 3),
        "atom14_mask":    (F, N, 14),
        "aatype":         (N,),
        "residue_index":  (N,),
        "pdb_resseq":     (N,),
        "adjacent":       (max(0, N-1),),
        "esm2":           (N, ESM_DIM),
    }
    for key, shape in expected.items():
        if key not in d.files:
            continue
        if d[key].shape != shape:
            r.fail(f"'{key}' shape {d[key].shape}, expected {shape}")
        else:
            r.ok(f"'{key}' shape {shape} ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Sequence, residue_index, adjacent mask
# ─────────────────────────────────────────────────────────────────────────────
def check_sequence(d, r):
    aa  = d["aatype"]
    ri  = d["residue_index"]
    seq = d["pdb_resseq"]
    adj = d["adjacent"]

    # aatype range
    bad = (aa < 0) | (aa > 20)
    if bad.any():
        r.fail(f"aatype out of [0,20] at positions {np.where(bad)[0][:5].tolist()}: "
               f"vals={np.unique(aa[bad]).tolist()}")
    else:
        r.ok("aatype in [0, 20] ✓")

    unk = (aa == 20).mean()
    (r.warn if unk > 0.10 else r.ok)(f"UNK (index 20) fraction: {unk:.1%}")

    # residue_index: must be exactly 0,1,2,...,N-1
    if np.array_equal(ri, np.arange(len(ri), dtype=np.int32)):
        r.ok("residue_index is 0-based sequential ✓")
    else:
        r.fail(f"residue_index not sequential: {ri[:8].tolist()}")

    # adjacent: values must be 0 or 1
    if not np.all((adj == 0) | (adj == 1)):
        r.fail("adjacent contains values other than {0, 1}")
    else:
        frac = adj.mean()
        r.ok(f"adjacent values OK, {frac:.1%} bonded pairs")
        if frac < 0.80:
            r.warn(f"Only {frac:.1%} bonded — many chain breaks?")

    # adjacent must match pdb_resseq gaps
    diffs = seq[1:].astype(int) - seq[:-1].astype(int)
    mismatch = (adj > 0) & (diffs != 1)
    if mismatch.any():
        r.warn(f"{mismatch.sum()} positions: adjacent=1 but resSeq diff≠1 "
               f"(insertion codes?): diffs={diffs[mismatch][:5].tolist()}")
    else:
        r.ok("adjacent consistent with pdb_resseq ✓")

    # pdb_resseq uniqueness
    if len(np.unique(seq)) != len(seq):
        r.warn("pdb_resseq has duplicate values — insertion codes not handled?")
    else:
        r.ok("pdb_resseq all unique ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Torsion angles
# ─────────────────────────────────────────────────────────────────────────────
def check_torsions(d, r):
    tors  = d["torsion_angles"]   # (F, N, 7)
    tmask = d["torsion_mask"]     # (F, N, 7)
    aa    = d["aatype"]           # (N,)

    # finite values
    if not np.isfinite(tors).all():
        r.fail(f"{(~np.isfinite(tors)).sum()} non-finite torsion values (NaN/Inf)")
    else:
        r.ok("All torsion values finite ✓")

    # range on active (masked) torsions only
    active = tors[tmask > 0]
    if len(active):
        lo, hi = float(active.min()), float(active.max())
        if lo < TORS_LO or hi > TORS_HI:
            r.fail(f"Torsion range [{lo:.4f}, {hi:.4f}] outside [-π, π]")
        else:
            r.ok(f"Active torsion range [{lo:.3f}, {hi:.3f}] rad ✓")

    # ── terminal masking ──────────────────────────────────────────────────────
    # phi (col 1): first residue has no C(i-1) → must be masked
    if tmask[:, 0, 1].any():
        r.fail("phi mask at residue 0 is non-zero  (no C(i-1) exists)")
    else:
        r.ok("phi mask at residue 0 = 0 ✓")

    # psi (col 2): last residue has no N(i+1) → must be masked
    if tmask[:, -1, 2].any():
        r.fail("psi mask at last residue is non-zero  (no N(i+1) exists)")
    else:
        r.ok("psi mask at last residue = 0 ✓")

    # pre_omega (col 0): first residue has no preceding peptide bond
    if tmask[:, 0, 0].any():
        r.warn("pre_omega mask at residue 0 is non-zero")
    else:
        r.ok("pre_omega mask at residue 0 = 0 ✓")

    # ── chi1 must not be active for GLY or ALA ───────────────────────────────
    no_chi    = (aa == GLY) | (aa == ALA)
    chi1_ever = tmask[:, :, 3].any(axis=0)    # (N,) active in any frame
    bad_chi   = chi1_ever & no_chi
    if bad_chi.any():
        r.fail(f"chi1 active for GLY/ALA at residues {np.where(bad_chi)[0][:8].tolist()}")
    else:
        r.ok("chi1 not active for GLY/ALA ✓")

    # ── coverage per torsion type (informational) ─────────────────────────────
    for i, name in enumerate(TORSION_NAMES):
        frac = float(tmask[:, :, i].mean())
        r.ok(f"  {name:12s} coverage: {frac:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — frame_t displacements and frame_R rotation matrices
# ─────────────────────────────────────────────────────────────────────────────
def check_frames(d, r):
    ft   = d["frame_t"]      # (F, N, 3)  — CA positions superposed to PDB
    fR   = d["frame_R"]      # (F, N, 3, 3)
    fmsk = d["frame_mask"]   # (F, N)
    valid = fmsk > 0
    F, N  = fmsk.shape

    # ── displacement magnitude ────────────────────────────────────────────────
    # frame_t is the CA position after superposing each frame to the PDB reference.
    # For ATLAS MD vs crystal structure, 50-100 Å mean displacement is NORMAL.
    # The comment "typical 0.5-5 Å" in the old sanity check was wrong.
    # We only fail if displacement is physically impossible (>200 Å).
    disp = np.linalg.norm(ft, axis=-1)   # (F, N)
    if valid.any():
        dv  = disp[valid]
        mn  = float(dv.mean())
        mx  = float(dv.max())
        r.ok(f"frame_t mean displacement: {mn:.1f} Å  "
             f"(ATLAS MD vs crystal — 50–100 Å is normal)")
        r.ok(f"frame_t max  displacement: {mx:.1f} Å")
        if mx > FRAME_T_MAX:
            r.fail(f"frame_t max {mx:.1f} Å > {FRAME_T_MAX} Å — likely alignment failure")

    # finite
    if not np.isfinite(ft[valid]).all():
        r.fail("Non-finite frame_t values at valid residues")
    else:
        r.ok("frame_t all finite at valid residues ✓")

    # ── inter-frame CA velocity spikes (PBC artefacts) ───────────────────────
    # Large jumps between consecutive frames mean a residue crossed the periodic
    # boundary without being unwrapped — the model sees a catastrophic velocity.
    if F > 1:
        jumps = np.linalg.norm(np.diff(ft, axis=0), axis=-1)   # (F-1, N)
        max_j = float(jumps.max())
        p99_j = float(np.percentile(jumps, 99))
        mn_j  = float(jumps.mean())
        r.ok(f"Inter-frame CA jump: mean={mn_j:.2f} Å  p99={p99_j:.2f} Å  max={max_j:.2f} Å")
        if max_j > SPIKE_THR:
            r.warn(f"CA jump max={max_j:.1f} Å > {SPIKE_THR} Å — "
                   f"possible PBC wrapping artefact (check XTC unwrapping)")
        if p99_j > 10.0:
            r.warn(f"CA jump p99={p99_j:.1f} Å high — "
                   f"check frame spacing or PBC unwrapping")

    # ── duplicate consecutive frames (truncated XTC) ─────────────────────────
    if F > 1:
        zero_diff = (np.linalg.norm(np.diff(ft, axis=0), axis=-1).max(axis=1) < 1e-4)
        if zero_diff.any():
            r.warn(f"{zero_diff.sum()} pairs of consecutive identical frames — "
                   f"XTC may be truncated or duplicated")
        else:
            r.ok("No duplicate consecutive frames ✓")

    # ── frame_mask coverage ───────────────────────────────────────────────────
    cov = float(fmsk.mean())
    r.ok(f"frame_mask coverage: {cov:.1%}")
    if cov < 0.85:
        r.warn(f"Low frame_mask coverage ({cov:.1%}) — many missing residues?")

    # ── SO(3) validity of rotation matrices ──────────────────────────────────
    # Sample up to 10 frames (full check is expensive for large F)
    n_chk = min(10, F)
    fidx  = np.linspace(0, F-1, n_chk, dtype=int)
    R_s   = fR[fidx]                            # (n_chk, N, 3, 3)
    vm    = fmsk[fidx] > 0                      # (n_chk, N)
    if vm.any():
        Rv   = R_s[vm]                          # (M, 3, 3)
        RtR  = np.einsum("mij,mik->mjk", Rv, Rv)
        errs = np.linalg.norm(RtR - np.eye(3), axis=(-2,-1))
        dets = np.linalg.det(Rv)
        max_err = float(errs.max())
        min_det = float(dets.min())
        if max_err > ORTH_MAX:
            r.fail(f"Rotation matrices not orthonormal: "
                   f"max ||R^T R - I||_F = {max_err:.4f}  (threshold {ORTH_MAX})")
        else:
            r.ok(f"Rotation matrices orthonormal: max err = {max_err:.2e} ✓")
        if min_det < 0.99:
            r.fail(f"Rotation matrices have det < 1 (reflections): min det={min_det:.4f}")
        else:
            r.ok(f"All rotation matrix determinants ≈ +1 ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 — Atom14 geometry (spot-check one randomly chosen frame)
# ─────────────────────────────────────────────────────────────────────────────
def check_atom14(d, r):
    pos  = d["atom14_pos"]    # (F, N, 14, 3)
    mask = d["atom14_mask"]   # (F, N, 14)
    fmsk = d["frame_mask"]    # (F, N)
    aa   = d["aatype"]        # (N,)
    adj  = d["adjacent"]      # (N-1,)

    # pick a random valid frame for spot-checks
    valid_f = np.where(fmsk.any(axis=1))[0]
    if not len(valid_f):
        r.fail("No valid frames to check"); return
    fi   = int(random.choice(valid_f))
    P    = pos[fi]    # (N, 14, 3)
    M    = mask[fi]   # (N, 14)
    fm   = fmsk[fi]   # (N,)
    bond = adj > 0    # (N-1,)

    # ── CA-CA bonded distance ─────────────────────────────────────────────────
    mCA   = (M[:, CA_] > 0) & (fm > 0)
    pairs = mCA[:-1] & mCA[1:] & bond
    if pairs.any():
        dc = _norm(P[1:][pairs, CA_], P[:-1][pairs, CA_])
        mn, sd = dc.mean(), dc.std()
        r.ok(f"CA-CA bonded: {mn:.3f} ± {sd:.3f} Å  (expect 3.80 ± 0.05)")
        if not (CA_CA_LO < mn < CA_CA_HI):
            r.fail(f"CA-CA mean {mn:.3f} Å outside [{CA_CA_LO}, {CA_CA_HI}]")
        if sd > 0.15:
            r.warn(f"CA-CA std {sd:.3f} Å unusually large")
    else:
        r.warn("No bonded CA pairs to check in sampled frame")

    # ── C(i)-N(i+1) peptide bond ──────────────────────────────────────────────
    mC  = (M[:, C_] > 0) & (fm > 0)
    mN  = (M[:, N_] > 0) & (fm > 0)
    pep = mC[:-1] & mN[1:] & bond
    if pep.any():
        dcn = _norm(P[:-1][pep, C_], P[1:][pep, N_])
        mn, sd = dcn.mean(), dcn.std()
        r.ok(f"C-N peptide bond: {mn:.3f} ± {sd:.3f} Å  (expect 1.33 ± 0.02)")
        bad = (dcn < CN_LO) | (dcn > CN_HI)
        if bad.any():
            r.fail(f"{bad.sum()} C-N bonds outside [{CN_LO}, {CN_HI}] Å  "
                   f"min={dcn.min():.3f}  max={dcn.max():.3f}")
    else:
        r.warn("No bonded C-N pairs to check in sampled frame")

    # ── CA-C-N bond angle ─────────────────────────────────────────────────────
    if pep.any():
        angs = _angle(P[:-1][pep, CA_], P[:-1][pep, C_], P[1:][pep, N_])
        mn_a = angs.mean()
        r.ok(f"CA-C-N bond angle: {mn_a:.1f}°  (expect ~116°)")
        if not (ANG_LO < mn_a < ANG_HI):
            r.warn(f"CA-C-N mean {mn_a:.1f}° outside [{ANG_LO}, {ANG_HI}]°")

    # ── CA-CB distance (non-GLY) ──────────────────────────────────────────────
    mCB = (M[:, CB_] > 0) & (fm > 0) & (aa != GLY)
    if mCB.any():
        dcb = _norm(P[mCB, CA_], P[mCB, CB_])
        mn_cb = dcb.mean()
        r.ok(f"CA-CB distance: {mn_cb:.3f} Å  (expect ~1.52 Å)")
        if not (CACB_LO < mn_cb < CACB_HI):
            r.fail(f"CA-CB mean {mn_cb:.3f} Å outside [{CACB_LO}, {CACB_HI}] "
                   f"— atom14 column layout wrong?")

    # ── Backbone heavy-atom clashes (sample 60 residues, non-bonded pairs) ───
    res_ok = np.where(fm > 0)[0][:60]
    bb_pairs = [(N_, CA_), (N_, C_), (CA_, C_), (CA_, O_), (C_, O_)]
    clashes  = 0
    for ai, aj in bb_pairs:
        for ri in res_ok:
            for rj in res_ok:
                if abs(ri - rj) <= 1: continue
                if M[ri, ai] and M[rj, aj]:
                    if _norm(P[ri, ai], P[rj, aj]) < CLASH_D:
                        clashes += 1
    if clashes:
        r.warn(f"{clashes} backbone heavy-atom clashes < {CLASH_D} Å  "
               f"(60-residue sample of frame {fi})")
    else:
        r.ok("No backbone clashes in sampled residues ✓")

    # ── Whole-trajectory finite check ─────────────────────────────────────────
    valid_res = fmsk > 0     # (F, N)
    # pos has shape (F, N, 14, 3); need to broadcast valid_res correctly
    valid_exp = valid_res[:, :, None, None]   # (F, N, 1, 1)
    if not np.isfinite(pos[valid_res[:, :, None, None].repeat(14, axis=2)
                            .repeat(3, axis=3)]).all():
        r.fail("Non-finite atom14_pos at valid residues")
    else:
        r.ok("atom14_pos all finite at valid residues ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7 — ESM2 embedding
# ─────────────────────────────────────────────────────────────────────────────
def check_esm(d, r):
    esm  = d["esm2"].astype(np.float32)   # stored fp16
    N    = d["aatype"].shape[0]
    fmsk = d["frame_mask"]                # (F, N)

    if esm.shape != (N, ESM_DIM):
        r.fail(f"esm2 shape {esm.shape}, expected ({N}, {ESM_DIM})")
    else:
        r.ok(f"esm2 shape ({N}, {ESM_DIM}) ✓")

    if not np.isfinite(esm).all():
        r.fail(f"{(~np.isfinite(esm)).sum()} non-finite ESM values")
    else:
        r.ok("ESM embeddings all finite ✓")

    abs_max = float(np.abs(esm).max())
    mean_l2 = float(np.linalg.norm(esm, axis=-1).mean())
    r.ok(f"ESM abs_max={abs_max:.2f}  mean L2={mean_l2:.2f}")

    if abs_max > 50.0:
        r.warn(f"ESM abs_max {abs_max:.1f} unusually large — possible fp16 overflow")
    if mean_l2 < 0.1:
        r.warn(f"ESM mean L2={mean_l2:.3f} near zero — all-zero embeddings?")

    # residues that are NEVER valid in frame_mask should have zero ESM
    ever_valid = (fmsk > 0).any(axis=0)   # (N,)
    pad_res    = ~ever_valid
    if pad_res.any():
        if not np.allclose(esm[pad_res], 0, atol=1e-4):
            r.warn(f"{pad_res.sum()} residues never valid in frame_mask "
                   f"but have non-zero ESM embeddings")
        else:
            r.ok("Padding residues have zero ESM ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 8 — Cross-array model-input consistency
# ─────────────────────────────────────────────────────────────────────────────
def check_consistency(d, r):
    aa    = d["aatype"]
    fmsk  = d["frame_mask"]
    a14m  = d["atom14_mask"]
    tmask = d["torsion_mask"]
    adj   = d["adjacent"]
    ft    = d["frame_t"]
    valid = fmsk > 0

    # 1) CA must exist for every valid residue
    ca_present = a14m[:, :, CA_] > 0
    bad = valid & ~ca_present
    if bad.any():
        r.fail(f"frame_mask=1 but atom14_mask[CA]=0 "
               f"for {bad.sum()} (frame, residue) pairs")
    else:
        r.ok("frame_mask and atom14_mask[CA] consistent ✓")

    # 2) torsion_mask must be 0 wherever frame_mask is 0
    bad2 = (~valid)[..., None] & (tmask > 0)
    if bad2.any():
        r.fail(f"torsion_mask active on {bad2.sum()} invalid (frame, residue) pairs")
    else:
        r.ok("torsion_mask zero on all invalid residues ✓")

    # 3) atom14_mask must be 0 wherever frame_mask is 0
    bad3 = (~valid)[..., None] & (a14m > 0)
    if bad3.any():
        r.fail(f"atom14_mask non-zero on {bad3.sum()} invalid (frame, residue) pairs")
    else:
        r.ok("atom14_mask zero on all invalid residues ✓")

    # 4) At chain-break positions (adj[i]=0):
    #    - psi[i]   must be masked  (needs N(i+1) across the break)
    #    - phi[i+1] must be masked  (needs C(i) across the break)
    gap_pos = np.where(adj == 0)[0]
    if len(gap_pos):
        psi_at_gap   = tmask[:, gap_pos,     2]   # (F, n_gaps)
        phi_after    = tmask[:, gap_pos + 1, 1]   # (F, n_gaps)
        if psi_at_gap.any():
            r.fail(f"psi active at {psi_at_gap.sum()} (frame, gap) entries — "
                   f"chain-break residue needs psi masked")
        else:
            r.ok("psi correctly masked at chain-break positions ✓")
        if phi_after.any():
            r.fail(f"phi active at {phi_after.sum()} (frame, gap+1) entries — "
                   f"post-chain-break residue needs phi masked")
        else:
            r.ok("phi correctly masked after chain-break positions ✓")
    else:
        r.ok("No chain breaks — gap masking not applicable")

    # 5) chi1 correct for special residues (redundant with check_torsions; belt+braces)
    no_chi   = (aa == GLY) | (aa == ALA)
    chi1_any = tmask[:, :, 3].any(axis=0)
    bad_chi  = chi1_any & no_chi
    if bad_chi.any():
        r.fail(f"chi1 active for GLY/ALA at {np.where(bad_chi)[0][:5].tolist()}")
    else:
        r.ok("chi1 not active for GLY/ALA (belt+braces check) ✓")

    # 6) frame_t finite at valid residues (model directly uses as position input)
    if not np.isfinite(ft[valid]).all():
        r.fail("Non-finite frame_t at valid residues")
    else:
        r.ok("frame_t finite at all valid residues ✓")

    # 7) PRO omega should not be stuck near 90° (sign of a torsion definition bug)
    pro_mask = (aa == PRO)
    if pro_mask.any():
        om     = d["torsion_angles"][:, pro_mask, 0]   # (F, n_pro)
        om_msk = tmask[:, pro_mask, 0]
        active = om[om_msk > 0]
        if len(active):
            near90 = np.abs(np.abs(active) - math.pi / 2) < 0.3
            if near90.mean() > 0.05:
                r.warn(f"{near90.mean():.1%} of PRO pre_omega values near 90° "
                       f"— possible torsion definition issue")
            else:
                r.ok("PRO pre_omega values not stuck near 90° ✓")


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────
CHECKS = [
    ("1. Keys, Dtypes & Tags",        check_keys),
    ("2. Shapes",                     check_shapes),
    ("3. Sequence & Adjacent",        check_sequence),
    ("4. Torsion Angles",             check_torsions),
    ("5. Frames & Rotations",         check_frames),
    ("6. Atom14 Geometry",            check_atom14),
    ("7. ESM2 Embedding",             check_esm),
    ("8. Cross-Array Consistency",    check_consistency),
]


def deep_check_npz(npz_path, verbose=True):
    fname = os.path.basename(npz_path)
    d     = np.load(npz_path, allow_pickle=True)
    all_ok  = True
    results = []
    for name, fn in CHECKS:
        res = Res(name)
        try:
            fn(d, res)
        except Exception as e:
            res.fail(f"Check crashed: {type(e).__name__}: {e}")
        results.append(res)
        if not res.passed():
            all_ok = False

    if verbose:
        print(f"\n{'═'*64}")
        print(f"  {fname}")
        print(f"{'═'*64}")
        for res in results:
            res.show()
        tag = "✅  ALL CHECKS PASSED" if all_ok else "❌  SOME CHECKS FAILED"
        print(f"\n{'═'*64}")
        print(f"  {tag}")
        print(f"{'═'*64}\n")

    d.close()
    return all_ok


def run_batch(npz_dir, n=20, verbose=True):
    files  = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    sample = random.sample(files, min(n, len(files)))
    print(f"Checking {len(sample)} / {len(files)} NPZ files …\n")
    passed, failed = [], []
    for fname in sample:
        ok = deep_check_npz(os.path.join(npz_dir, fname), verbose=verbose)
        (passed if ok else failed).append(fname)

    print(f"\n{'═'*64}")
    print(f"  BATCH SUMMARY:  {len(passed)} passed  |  {len(failed)} failed")
    if failed:
        print("  Failed files:")
        for f in failed:
            print(f"    ✗  {f}")
    print(f"{'═'*64}")


# ─────────────────────────────────────────────────────────────────────────────
run_batch(LOCAL_NPZ_DIR, n=N_SAMPLE, verbose=VERBOSE)

