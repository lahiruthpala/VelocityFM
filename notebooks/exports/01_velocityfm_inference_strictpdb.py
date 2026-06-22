# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% [markdown] cell 0
# # Velocity-FM Protein Trajectory Inference (Colab)
# 
# This notebook performs **inference-only** sampling from your **velocity-space Flow Matching** trajectory model.
# 
# **Inputs**
# - A single-frame **PDB** (starting structure)
# - A trained **checkpoint** (`best.pt` / `last.pt`)
# - `TOTAL_FRAMES` (total frames written, including the input as frame 0)
# 
# **Output**
# - A **multi-model PDB** written via **MDTraj** (`Trajectory.save_pdb`), i.e., standard `MODEL/ENDMDL` blocks.
# 
# **Path policy**
# - A **static base path** (`DRIVE_BASE`) is used.
# - For each file, you only edit the **final segment** (e.g., filename like `7z1k_A.pdb`, `best.pt`, `traj_200f.pdb`).

# %% cell 1
# =========================
# 0) Install dependencies
# =========================
# NOTEBOOK_MAGIC: !pip -q install mdtraj biopython fair-esm

import os, sys, math, inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, List

import mdtraj as md
from Bio.SeqUtils import seq1

# %% cell 2
# NOTEBOOK_MAGIC: !pip -q install einops dm-tree ml-collections biopython scipy pandas tqdm modelcif geomstats pot

# %% cell 3

# NEW: FoldFlow clone (vendored openfold; no CUDA build)
import os, sys, shutil

FOLDFLOW_DIR = "/content/FoldFlow"

if not os.path.exists(FOLDFLOW_DIR):
    # NOTEBOOK_MAGIC: !git clone https://github.com/DreamFold/FoldFlow.git {FOLDFLOW_DIR}

# Make sure Python resolves FoldFlow's vendored `openfold/` first
if FOLDFLOW_DIR not in sys.path:
    sys.path.insert(0, FOLDFLOW_DIR)

# Optional: if FoldFlow has a nested package layout, also add these defensively
for p in [os.path.join(FOLDFLOW_DIR, "FoldFlow"), os.path.join(FOLDFLOW_DIR, "src")]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import types, sys
sys.modules.setdefault("ipdb", types.SimpleNamespace(set_trace=lambda *a, **k: None))
print("sys.path[0:5] =", sys.path[:5])

# %% cell 4
# Minimal FoldFlow so3_flow_matching patch for Python 3.12 (no verbose logs)

import os, re, pathlib, sys, subprocess

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")

def _find_so3_flow_matching():
    cands = [
        "/content/FoldFlow/FoldFlow/so3/so3_flow_matching.py",
        "/content/FoldFlow-main/FoldFlow/so3/so3_flow_matching.py",
    ]
    for c in cands:
        p = pathlib.Path(c)
        if p.exists():
            return p
    for p in pathlib.Path("/content").rglob("so3_flow_matching.py"):
        return p
    return None

p = _find_so3_flow_matching()
if p is None:
    raise FileNotFoundError("so3_flow_matching.py not found under /content (check unzip/clone path).")

txt = p.read_text()

# 1) Guard private geomstats import (Python 3.12 + geomstats variations)
if ("from geomstats._backend import _backend_config as _config" in txt) and \
   ("try:\n    from geomstats._backend import _backend_config as _config" not in txt):
    txt = re.sub(
        r"from\s+geomstats\._backend\s+import\s+_backend_config\s+as\s+_config\s*\n"
        r"(#?_config\.DEFAULT_DTYPE.*\n)?",
        "try:\n    from geomstats._backend import _backend_config as _config  # optional\n"
        "except Exception:\n    _config = None\n",
        txt,
    )

# 2) Robust vmap import
if "from functorch import vmap" in txt and "from torch.func import vmap" not in txt:
    txt = txt.replace(
        "from functorch import vmap",
        "try:\n    from torch.func import vmap\nexcept Exception:\n    from functorch import vmap",
    )

p.write_text(txt)

# Import test (raises immediately if still broken)
import FoldFlow.so3.so3_flow_matching as sfm

# Optional: hard-assert the symbol you actually need
assert hasattr(sfm, "SO3ConditionalFlowMatcher"), "SO3ConditionalFlowMatcher missing after import."

# %% cell 5
# =========================
# 1) Mount Google Drive
# =========================
from google.colab import drive
drive.mount("/content/drive")

# %% cell 6
# =========================
# 2) Paths (static prefix + final segment inputs)
# =========================

# ---- Static root (do NOT change frequently)
DRIVE_BASE = "/content/drive/MyDrive/af_native_dynamics_predictor"

# ---- Final path segments (YOU edit only these)
Frames = 64
Model = "Model_T4_V6"
PDB_NAME        = "8htw_B"         # e.g., "7z1k_A.pdb"
TRAINING_RUN    = "Traning_192"         # e.g., "Traning_12"
CKPT_NAME       = "best.pt"           # e.g., "best.pt" or "last.pt"
OUT_PDB_NAME    = f"{PDB_NAME}_traj_{Frames}f.pdb"     # output multi-frame PDB

# ---- Static subfolders (edit once if your layout differs)
PDB_DIR   = os.path.join(DRIVE_BASE, "data", "raw", "MD_Simulation", "proteins")                 # where input PDBs live
MODEL_ROOT= os.path.join(DRIVE_BASE, "models", Model)                  # model version root
OUT_DIR   = os.path.join(MODEL_ROOT, TRAINING_RUN,"inference_outputs")                      # where outputs are written
os.makedirs(OUT_DIR, exist_ok=True)

PDB_PATH   = os.path.join(PDB_DIR, PDB_NAME, f"{PDB_NAME}.pdb")
CKPT_DIR   = os.path.join(MODEL_ROOT, TRAINING_RUN, "checkpoints")
CKPT_PATH  = os.path.join(CKPT_DIR, CKPT_NAME)
OUT_PATH   = os.path.join(OUT_DIR, OUT_PDB_NAME)

print("PDB_PATH :", PDB_PATH)
print("CKPT_PATH:", CKPT_PATH)
print("OUT_PATH :", OUT_PATH)

# %% cell 7
# =========================
# 3) Inference configuration (editable)
# =========================
# NOTE: When a checkpoint is loaded, its saved `cfg` is used as the base.
# Keys present here then OVERRIDE the checkpoint cfg.
# Architecture keys (C_S, IPA_BLOCKS, etc.) are verified to match the
# checkpoint so you get an immediate error if they're out of sync.

INF_CFG = dict(
    # ── Output ──────────────────────────────────────────────────────────────
    # TOTAL_FRAMES  = Frames,   # total frames to write (frame 0 = input structure)
    CHAIN_ID      = None,     # "A" etc. If None → first chain

    # ── Sampling / Flow ──────────────────────────────────────────────────────
    FLOW_STEPS    = 32,       # Euler steps in flow-time s∈[0,1]
    SEED          = 0,
    AMP           = True,     # BF16 autocast (recommended on GPU)
    AMP_DTYPE     = "bf16",   # must match training AMP_DTYPE

    # ── Windowing (MUST match training) ─────────────────────────────────────
    WINDOW_SIZE   = 16,       # W future frames per window  ← CFG["WINDOW_SIZE"]
    WINDOW_STRIDE = 1,        # default stride S in frames  ← CFG["WINDOW_STRIDE"]
    MULTI_STRIDE  = [1, 2, 4],# used only to pick dt at inference; first entry used

    # ── Physical time ────────────────────────────────────────────────────────
    DT_PHYS       = None,     # None → dt = WINDOW_STRIDE (in frames). Set to ps if needed.
    DELTA_T_SCALE = 4.0,      # ← CFG["DELTA_T_SCALE"]  (delta_t MLP input = delta_t/scale)
    COORD_SCALE   = 1.0,      # ← CFG["COORD_SCALE"] keep 1.0 for Å data

    # ── Velocity normalisation (MUST match training) ─────────────────────────
    NORMALIZE_VELOCITIES = True,
    V_STD         = 0.28471,   # ← CFG["V_STD"]
    OMG_STD       = 0.07205,   # ← CFG["OMG_STD"]
    THDOT_STD     = 0.21562,   # ← CFG["THDOT_STD"]

    # ── Noise in velocity space ──────────────────────────────────────────────
    SIGMA_V       = 0.3701,    # ← CFG["SIGMA_V"]
    SIGMA_OMG     = 0.0937,    # ← CFG["SIGMA_OMG"]
    SIGMA_THDOT   = 0.2803,    # ← CFG["SIGMA_THDOT"]
    NOISE_AR1_RHO = 0.7,      # ← CFG["NOISE_AR1_RHO"]  temporal AR(1) correlation
    SIGMA_STEP_ALPHA = 0.5,   # ← CFG["SIGMA_STEP_ALPHA"] sigma(k) = sigma0*(1+0.5*k/(W-1))

    # ── Flow-time sampling ───────────────────────────────────────────────────
    FLOW_S_PER_STEP = True,   # ← CFG["FLOW_S_PER_STEP"]
    FLOW_S_RHO      = 0.7,    # ← CFG["FLOW_S_RHO"]
    FLOW_S_NOISE    = 0.15,   # ← CFG["FLOW_S_NOISE"]

    # ── Model architecture (MUST match training exactly) ─────────────────────
    C_S           = 384,      # ← CFG["C_S"]   single embedding dim
    C_Z           = 128,      # ← CFG["C_Z"]   pair   embedding dim
    IPA_BLOCKS    = 6,        # ← CFG["IPA_BLOCKS"]
    HEADS         = 8,        # ← CFG["HEADS"]
    IPA_C_HIDDEN  = 16,       # ← CFG["IPA_C_HIDDEN"]
    NO_QK_POINTS  = 4,        # ← CFG["NO_QK_POINTS"]
    NO_V_POINTS   = 8,        # ← CFG["NO_V_POINTS"]
    AUX_EVERY     = 2,        # ← CFG["AUX_EVERY"]  aux head every N IPA blocks
    IPA_MASK_INF  = 50,       # ← CFG["IPA_MASK_INF"] (training default via config)

    # ── Relative positional bias ──────────────────────────────────────────────
    RELPOS_MAX    = 155.0,     # ← CFG["RELPOS_MAX"]  max |i-j| bucket

    # ── Temporal attention ───────────────────────────────────────────────────
    USE_TEMPORAL_ATTENTION = True,   # ← CFG["USE_TEMPORAL_ATTENTION"]
    TEMP_ATTN_LAYERS       = 2,      # ← CFG["TEMP_ATTN_LAYERS"]
    TEMP_ATTN_HEADS        = 8,      # ← CFG["TEMP_ATTN_HEADS"]
    TEMP_ATTN_DROPOUT      = 0.0,    # ← CFG["TEMP_ATTN_DROPOUT"] (0 at inference)

    # ── Task / conditioning ───────────────────────────────────────────────────
    USE_TASK_INTERP  = True,  # ← CFG["USE_TASK_INTERP"]  (only affects interpolation mode)
    USE_ANCHOR_EMB   = True,  # ← CFG["USE_ANCHOR_EMB"]   (must match training to load weights)
    ANCHOR_ALPHA     = 0.0,   # ← set 0.0 for free generation; >0 anchors toward start frame

    # ── Pair feature sharing ─────────────────────────────────────────────────
    USE_SHARED_PAIR_Z = True, # ← CFG["USE_SHARED_PAIR_Z"]  precompute z_pair once per window
    PAIR_Z_SOURCE     = "cond", # ← CFG["PAIR_Z_SOURCE"]  "cond" | "step0"
    DETACH_PAIR_X     = True, # ← CFG["DETACH_PAIR_X"]

    # ── SO(3) orthonormalisation ──────────────────────────────────────────────
    ORTHO_INPUT_R = True,     # ← CFG["ORTHO_INPUT_R"]  re-orthonormalise input R each step
    ORTHO_PRED_R  = True,     # ← CFG["ORTHO_PRED_R"]   re-orthonormalise predicted R
    ORTHO_R_FP32  = True,     # ← CFG["ORTHO_R_FP32"]   run ortho in FP32 for safety

    # ── iGSO3 noise (set to match training; can lower SIGMA_* for less noise) ─
    USE_IGSO3      = False,   # ← CFG["USE_IGSO3"]
    IGSO3_STRICT   = True,    # ← CFG["IGSO3_STRICT"]
    IGSO3_USE_AR1  = True,    # ← CFG["IGSO3_USE_AR1"]
    IGSO3_EPS_FACTOR = 1.0,   # ← CFG["IGSO3_EPS_FACTOR"]

    # ── Memory / compute ──────────────────────────────────────────────────────
    STEP_CHUNK            = 1,     # ← CFG["STEP_CHUNK"]  process N window steps at once
    GRAD_CHECKPOINT_IPA   = False, # ← CFG["GRAD_CHECKPOINT_IPA"]  no checkpointing at inf
    CAST_Z_FP32_AFTER_SLICE = True,# ← CFG["CAST_Z_FP32_AFTER_SLICE"]

    # ── Data shape ───────────────────────────────────────────────────────────
    MAX_RES = 180,            # ← CFG["MAX_RES"]  used to size padding / masks

    # ── ESM embeddings (compute on-the-fly from PDB sequence) ────────────────
    ESM_MODEL_NAME = "esm2_t33_650M_UR50D",
    ESM_LAYER      = 33,
    ESM_DIM        = 1280,    # ← must match esm_proj.weight in checkpoint

    # ── Reconstruction / output ───────────────────────────────────────────────
    PREFER_ATOM14  = True,    # use all-atom14 output if OpenFold available; else CA-only

    # ── Optional per-run overrides (keys here win over checkpoint cfg) ────────
    # Use this to safely override non-architectural settings without editing above.
    # e.g. OVERRIDE = dict(WINDOW_STRIDE=2, ANCHOR_ALPHA=0.1)
    OVERRIDE = {},

    # ── Diversity / Langevin sampling ────────────────────────────────────────
    # TEMPERATURE > 1.0 scales up initial noise in flow ODE → broader basin sampling
    # Start at 1.5–2.0; go up to 3.0 if still collapsed
    TEMPERATURE      = 2.0,

    # Langevin noise injected into conditioning state between windows
    # Breaks deterministic trajectory stalling between rollout windows
    # Units: Å for translation, radians for rotation/torsion
    LANGEVIN_NOISE_X    = 0.15,   # Å per window step
    LANGEVIN_NOISE_TORS = 0.05,   # rad per window step
)

# %% cell 8
# =========================
# 4) Environment: add FoldFlow/OpenFold to PYTHONPATH (static prefix + final segment)
# =========================
# Try importing OpenFold rigid utils (required for IPA rigids)
try:
    from openfold.utils import rigid_utils
    from openfold.utils import feats as of_feats
    from openfold.np import residue_constants as of_rc
    HAS_OPENFOLD = True
    print("OpenFold import: OK")
except Exception as e:
    HAS_OPENFOLD = False
    print("OpenFold import: FAILED -> atom14 output may fall back to CA-only.")
    print("Error:", repr(e))

# %% cell 9
# =========================
# 4b) FoldFlow wrapper refresh (defines FF namespace used by the model/builder)
# =========================
import importlib

# --------------------------------------------------------------------------------------
# External module refresh (FoldFlow wrappers + vendored OpenFold)
# --------------------------------------------------------------------------------------

class _FFNamespace:
    """Namespace for optional FoldFlow components. Never modify the FoldFlow repo; wrap it here."""
    def __init__(self):
        self.ipa_pytorch = None
        self.so3_helpers = None
        self.all_atom = None
        self.igso3 = None
        self.condflowmatcher = None
        self.so3_flow_matching = None

FF = _FFNamespace()

# Vendored OpenFold modules (provided by FoldFlow clone). These are pure-Python.
rc = None
rigid_utils = None
feats = None
of_loss = None

def refresh_external_modules(foldflow_dir: str = "/content/FoldFlow") -> None:
    """(Re)import FoldFlow + vendored OpenFold modules after cloning / path changes."""
    global rc, rigid_utils, feats, of_loss, FF

    # Best-effort FoldFlow wrappers
    for modattr, modname in [
        ("ipa_pytorch", "foldflow.models.components.ipa_pytorch"),
        ("so3_helpers", "foldflow.utils.so3_helpers"),
        ("all_atom", "foldflow.data.all_atom"),
        ("igso3", "foldflow.utils.igso3"),
        ("condflowmatcher", "foldflow.utils.condflowmatcher"),
        ("so3_flow_matching", "FoldFlow.so3.so3_flow_matching"),  # heavy deps; optional
    ]:
        try:
            setattr(FF, modattr, importlib.import_module(modname))
        except Exception:
            print(f"Failed to import {modname}")
            setattr(FF, modattr, None)

    # Vendored OpenFold (from FoldFlow clone) – required for residue constants / Rigid types / FAPE utilities.
    try:
        rc = importlib.import_module("openfold.np.residue_constants")
        rigid_utils = importlib.import_module("openfold.utils.rigid_utils")
        feats = importlib.import_module("openfold.utils.feats")
        try:
            of_loss = importlib.import_module("openfold.utils.loss")
        except Exception:
            print("Failed to import openfold.utils.loss")
            of_loss = None
    except Exception:
        # In a non-Colab / non-clone environment these may be missing.
        print("Failed to import OpenFold vendored modules")
        rc = None
        rigid_utils = None
        feats = None
        of_loss = None

# Try a light refresh at import-time (no hard failure).
refresh_external_modules()
torch.set_default_dtype(torch.float32)

# Re-run refresh with the configured folder (if present)
try:
    refresh_external_modules(foldflow_dir=FOLDFLOW_DIR)
except Exception as e:
    print("refresh_external_modules failed:", repr(e))

# Update OpenFold availability flags used later
HAS_OPENFOLD = (rc is not None) and (rigid_utils is not None) and (feats is not None)
if HAS_OPENFOLD:
    from openfold.np import residue_constants as of_rc
    from openfold.utils import rigid_utils as of_rigid_utils
    from openfold.utils import feats as of_feats

# %% cell 10
# =========================
# 5) PDB -> model inputs (sequence, aatype, backbone frames, torsions, atom14 indices)
# =========================

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {a:i for i,a in enumerate(AA_ORDER)}
UNK_AA = 20

# AF-style atom14 ordering (heavy atoms)
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

RES3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y","MSE":"M",
}

def residue_to_aa1(res3: str) -> str:
    return RES3_TO_1.get(res3.upper(), "X")

def safe_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)

def compute_backbone_frames(N_xyz: np.ndarray, CA_xyz: np.ndarray, C_xyz: np.ndarray):
    frame_mask = ((np.linalg.norm(N_xyz, axis=-1) > 0) &
                  (np.linalg.norm(CA_xyz, axis=-1) > 0) &
                  (np.linalg.norm(C_xyz, axis=-1) > 0)).astype(np.uint8)
    e1 = safe_normalize(C_xyz - CA_xyz)   # CA->C
    e2 = safe_normalize(N_xyz - CA_xyz)   # CA->N
    e3 = safe_normalize(np.cross(e1, e2))
    e2 = safe_normalize(np.cross(e3, e1))
    R = np.stack([e1, e2, e3], axis=-1).astype(np.float32)
    t = CA_xyz.astype(np.float32)
    return R, t, frame_mask

def build_atom_index_maps(residues):
    N = len(residues)
    atom14_indices = np.full((N, 14), -1, dtype=np.int32)
    n_idx   = np.full((N,), -1, dtype=np.int32)
    ca_idx  = np.full((N,), -1, dtype=np.int32)
    c_idx   = np.full((N,), -1, dtype=np.int32)
    pdb_resseq = np.zeros((N,), dtype=np.int32)   # ← NEW: raw PDB numbering for gap detection

    seq_chars = []
    aatype = np.full((N,), UNK_AA, dtype=np.int64)

    for i, res in enumerate(residues):
        pdb_resseq[i] = getattr(res, "resSeq", i)  # ← NEW
        aa = residue_to_aa1(res.name)
        seq_chars.append(aa)
        aatype[i] = AA_TO_IDX.get(aa, UNK_AA)

        atoms = {a.name: a.index for a in res.atoms}
        n_idx[i]  = atoms.get("N",  -1)
        ca_idx[i] = atoms.get("CA", -1)
        c_idx[i]  = atoms.get("C",  -1)

        names14 = ATOM14_NAMES.get(aa, ["N","CA","C","O"] + [""]*10)
        for j, nm in enumerate(names14):
            if nm:
                atom14_indices[i, j] = atoms.get(nm, -1)

    # returns 7 values now — update the call site in Run Inference cell too
    return atom14_indices, n_idx, ca_idx, c_idx, "".join(seq_chars), aatype, pdb_resseq

def compute_adjacent_mask(pdb_resseq: np.ndarray) -> np.ndarray:
    """adjacent[i] == 1 means residue i and i+1 are bonded (no PDB gap)."""
    if len(pdb_resseq) <= 1:
        return np.zeros(max(0, len(pdb_resseq) - 1), dtype=np.uint8)
    return (pdb_resseq[1:] - pdb_resseq[:-1] == 1).astype(np.uint8)

def build_torsion_quads(residues, n_idx, ca_idx, c_idx, aatype, pdb_resseq):
    """
    Produces quads for md.compute_dihedrals and meta=(slot_index, residue_i) pairs.

    Slot layout matches training data exactly:
      0  pre_omega(i) : CA[i-1], C[i-1], N[i],   CA[i]    gated by adjacent[i-1]
      1  phi(i)       : C[i-1],  N[i],   CA[i],  C[i]     gated by adjacent[i-1]
      2  psi(i)       : N[i],    CA[i],  C[i],   N[i+1]   gated by adjacent[i]
      3-6 chi1..chi4  : side-chain rotamers
    """
    adjacent = compute_adjacent_mask(pdb_resseq)
    quads = []
    meta  = []
    Nres  = len(residues)

    for i in range(Nres):
        # pre_omega(i) and phi(i) both require residue i-1 to be adjacent
        if i > 0 and adjacent[i - 1] == 1:
            # pre_omega: CA[i-1], C[i-1], N[i], CA[i]
            if ca_idx[i-1] >= 0 and c_idx[i-1] >= 0 and n_idx[i] >= 0 and ca_idx[i] >= 0:
                quads.append([ca_idx[i-1], c_idx[i-1], n_idx[i], ca_idx[i]])
                meta.append((0, i))
            # phi: C[i-1], N[i], CA[i], C[i]
            if c_idx[i-1] >= 0 and n_idx[i] >= 0 and ca_idx[i] >= 0 and c_idx[i] >= 0:
                quads.append([c_idx[i-1], n_idx[i], ca_idx[i], c_idx[i]])
                meta.append((1, i))

        # psi(i) requires residue i+1 to be adjacent
        if i + 1 < Nres and adjacent[i] == 1:
            # psi: N[i], CA[i], C[i], N[i+1]  ← OpenFold v2 convention
            if n_idx[i] >= 0 and ca_idx[i] >= 0 and c_idx[i] >= 0 and n_idx[i+1] >= 0:
                quads.append([n_idx[i], ca_idx[i], c_idx[i], n_idx[i+1]])
                meta.append((2, i))

    # chi1..chi4 (slots 3-6)
    for i, res in enumerate(residues):
        aa = AA_ORDER[aatype[i]] if aatype[i] < 20 else "X"
        chi_list = CHI_DEFS.get(aa, [])
        if not chi_list:
            continue
        atoms = {a.name: a.index for a in res.atoms}
        for k, (a, b, c, d) in enumerate(chi_list[:4]):
            if a in atoms and b in atoms and c in atoms and d in atoms:
                quads.append([atoms[a], atoms[b], atoms[c], atoms[d]])
                meta.append((3 + k, i))

    if len(quads) == 0:
        return np.zeros((0, 4), dtype=np.int32), []
    return np.asarray(quads, dtype=np.int32), meta

def compute_initial_torsions(traj0, residues, n_idx, ca_idx, c_idx, aatype, pdb_resseq):
    quads, meta = build_torsion_quads(residues, n_idx, ca_idx, c_idx, aatype, pdb_resseq)
    Nres = len(residues)
    tors = np.zeros((Nres, 7), dtype=np.float32)
    mask = np.zeros((Nres, 7), dtype=np.float32)
    if quads.shape[0] == 0:
        return tors, mask
    ang = md.compute_dihedrals(traj0, quads)[0]   # shape (n_quads,), radians
    for j, (t_type, i) in enumerate(meta):
        tors[i, t_type] = float(ang[j])
        mask[i, t_type] = 1.0
    return tors, mask

def resolve_chain(top: md.Topology, chain_id: str | None):
    chains = list(top.chains)
    if len(chains) == 0:
        raise ValueError("No chains found in PDB topology.")
    if chain_id is None:
        return chains[0]
    for ch in chains:
        cid = getattr(ch, "chain_id", None) or getattr(ch, "id", None)
        if cid is not None and str(cid).strip() == str(chain_id).strip():
            return ch
    # fallback: numeric
    try:
        return chains[int(chain_id)]
    except Exception:
        return chains[0]

# %% cell 11
# =========================
# 6) Geometry utilities (from your training notebook)
# =========================
# The code below is copied from your training notebook (Geometry cell).

# --------------------------------------------------------------------------------------
# 4) Geometry utilities (SO(3), wrapping, rigid transforms)
# --------------------------------------------------------------------------------------

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


# ==========================================
# 1. GEOMETRY UTILS (SO3)
# ==========================================

class SO3:
    @staticmethod
    def hat(w: torch.Tensor) -> torch.Tensor:
        """(...,3) -> (...,3,3)"""
        wx, wy, wz = w.unbind(-1)
        O = torch.zeros_like(wx)
        return torch.stack([
            O,  -wz,  wy,
            wz,  O,  -wx,
           -wy, wx,  O
        ], dim=-1).reshape(w.shape[:-1] + (3, 3))

    @staticmethod
    def exp(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """SO(3) exponential map with stable small-angle handling.
        w: (...,3) rotation-vector (axis-angle) in radians.
        """
        theta = torch.linalg.norm(w, dim=-1, keepdim=True)
        theta2 = theta * theta
        theta_safe = torch.clamp(theta, min=eps)

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        small = theta < 1e-4
        A = torch.where(
            small,
            1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0,
            sin_t / theta_safe,
        )
        B = torch.where(
            small,
            0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0,
            (1.0 - cos_t) / (theta_safe * theta_safe),
        )

        W = SO3.hat(w)
        W2 = W @ W
        I = torch.eye(3, device=w.device, dtype=w.dtype).expand(W.shape)
        return I + A.unsqueeze(-1) * W + B.unsqueeze(-1) * W2

    @staticmethod
    def log(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """SO(3) logarithm map with stable handling near 0 and near pi.
        R: (...,3,3) rotation matrices
        Returns w: (...,3) rotation-vectors (axis-angle) in radians.
        """
        tr = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta)  # (...)

        # general branch: w = vee( (R-R^T) * theta / (2 sin theta) )
        sin_theta = torch.sin(theta).clamp(min=eps)
        factor = theta / (2.0 * sin_theta)
        A = R - R.transpose(-1, -2)
        w_hat = factor[..., None, None] * A
        w = torch.stack([w_hat[..., 2, 1], w_hat[..., 0, 2], w_hat[..., 1, 0]], dim=-1)

        # small-angle refinement (theta ~ 0): use first-order approximation
        small = theta < 1e-6
        if small.any():
            w_small = 0.5 * torch.stack([A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]], dim=-1)
            w = torch.where(small[..., None], w_small, w)

        # near-pi handling: use (R+I)/2 to extract axis stably
        near_pi = (math.pi - theta).abs() < 1e-3
        if near_pi.any():
            I = torch.eye(3, device=R.device, dtype=R.dtype).expand(R.shape)
            Rp = 0.5 * (R + I)
            diag = torch.stack([Rp[..., 0, 0], Rp[..., 1, 1], Rp[..., 2, 2]], dim=-1)
            max_i = diag.argmax(dim=-1)  # (...)

            axis = torch.zeros_like(w)
            # i=0
            m0 = near_pi & (max_i == 0)
            if m0.any():
                a0 = torch.sqrt(torch.clamp(Rp[..., 0, 0], min=eps))
                axis0 = torch.zeros_like(w)
                axis0[..., 0] = a0
                axis0[..., 1] = Rp[..., 0, 1] / (a0 + eps)
                axis0[..., 2] = Rp[..., 0, 2] / (a0 + eps)
                axis = torch.where(m0[..., None], axis0, axis)

            # i=1
            m1 = near_pi & (max_i == 1)
            if m1.any():
                a1 = torch.sqrt(torch.clamp(Rp[..., 1, 1], min=eps))
                axis1 = torch.zeros_like(w)
                axis1[..., 1] = a1
                axis1[..., 0] = Rp[..., 0, 1] / (a1 + eps)
                axis1[..., 2] = Rp[..., 1, 2] / (a1 + eps)
                axis = torch.where(m1[..., None], axis1, axis)

            # i=2
            m2 = near_pi & (max_i == 2)
            if m2.any():
                a2 = torch.sqrt(torch.clamp(Rp[..., 2, 2], min=eps))
                axis2 = torch.zeros_like(w)
                axis2[..., 2] = a2
                axis2[..., 0] = Rp[..., 0, 2] / (a2 + eps)
                axis2[..., 1] = Rp[..., 1, 2] / (a2 + eps)
                axis = torch.where(m2[..., None], axis2, axis)

            axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=eps))
            w_pi = axis * theta.unsqueeze(-1)
            w = torch.where(near_pi[..., None], w_pi, w)

        return w

    @staticmethod
    def orthonormalize_safe(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Project a batch of matrices to the closest proper rotation (det=+1) via SVD."""
        U, S, Vh = torch.linalg.svd(R)
        UV = U @ Vh
        det = torch.linalg.det(UV)
        # fix reflections (det<0) by flipping last column of U
        if det.ndim == 0:
            if det < 0:
                U = U.clone()
                U[..., :, -1] *= -1.0
        else:
            neg = det < 0
            if neg.any():
                U = U.clone()
                U[neg, :, -1] *= -1.0
        return U @ Vh


def so3_exp(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation-vector exponential map.

    Uses FoldFlow's numerically-stabilized implementation if available; otherwise falls back to local SO3.exp.
    """
    if (FF.so3_helpers is not None) and hasattr(FF.so3_helpers, "so3_exp_map"):
        return FF.so3_helpers.so3_exp_map(w)
    return SO3.exp(w, eps=eps)


def so3_log(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SO(3) log map: (...,3,3) -> (...,3)
    Uses FoldFlow helper if available. That helper hardcodes float64 internally,
    so we upcast inputs to float64 for the call, then cast back.
    """
    if (FF.so3_helpers is not None) and hasattr(FF.so3_helpers, "rotmat_to_rotvec"):
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"so3_log expected (...,3,3), got {tuple(R.shape)}")

        lead = R.shape[:-2]
        Rf = R.reshape(-1, 3, 3).contiguous()

        # Critical: FoldFlow helper uses float64 internally -> feed float64 to avoid index_put mismatch
        with torch.autocast(device_type=R.device.type, enabled=False):
            vf64 = FF.so3_helpers.rotmat_to_rotvec(Rf.to(torch.float64))  # (prod,3), float64

        return vf64.to(R.dtype).reshape(*lead, 3)

    # fallback if FoldFlow helper isn't available
    return SO3.log(R, eps=eps)



def make_rigid(R: torch.Tensor, t: torch.Tensor) -> rigid_utils.Rigid:
    """
    R: (...,3,3), t: (...,3)
    """
    return rigid_utils.Rigid(
        rigid_utils.Rotation(rot_mats=R),
        t,
    )


def apply_global_se3(Q, t, x, R):
    """
    Apply x' = x * Q^T + t,  R' = Q * R

    Q: (B,3,3), t: (B,3)
    x: (B,W,N,3) or (B,N,3) or None
    R: (B,W,N,3,3) or (B,N,3,3) or None

    Returns:
      if R is None: x2
      else: (x2, R2)
    """
    x2 = None

    # --- positions ---
    if x is not None:
        Qx = Q.to(dtype=x.dtype, device=x.device)
        tx = t.to(dtype=x.dtype, device=x.device)

        if x.ndim == 4:  # (B,W,N,3)
            # Make Q broadcast over W,N: (B,1,3,3)
            Qb = Qx[:, None, :, :]  # (B,1,3,3)
            x2 = x @ Qb.transpose(-1, -2) + tx[:, None, None, :]
        elif x.ndim == 3:  # (B,N,3)
            x2 = x @ Qx.transpose(-1, -2) + tx[:, None, :]
        else:
            raise ValueError(f"Unsupported x.ndim={x.ndim} for x shape {tuple(x.shape)}")

    # --- rotations ---
    if R is None:
        return x2

    QR = Q.to(dtype=R.dtype, device=R.device)

    if R.ndim == 5:  # (B,W,N,3,3)
        # Broadcast Q over W,N: (B,1,1,3,3)
        Qb = QR[:, None, None, :, :]
        R2 = Qb @ R
    elif R.ndim == 4:  # (B,N,3,3)
        # Broadcast Q over N: (B,1,3,3)
        Qb = QR[:, None, :, :]
        R2 = Qb @ R
    else:
        raise ValueError(f"Unsupported R.ndim={R.ndim} for R shape {tuple(R.shape)}")

    return x2, R2

def kabsch_align(P: torch.Tensor, Qgt: torch.Tensor, w: torch.Tensor, eps: float = 1e-8):
    """
    Weighted Kabsch alignment.
    P, Qgt: (B,M,3)
    w: (B,M)
    Returns (R,t) such that R@P + t aligns to Qgt

    NOTE: SVD is not supported for float16 on CUDA. We force FP32 here.
    """
    with torch.autocast(device_type=P.device.type, enabled=False):
        P32   = P.to(torch.float32)
        Q32   = Qgt.to(torch.float32)
        w32   = w.to(torch.float32)

        B, M, _ = P32.shape
        w32 = w32.clamp(min=0.0)
        ws  = w32.sum(dim=1, keepdim=True).clamp(min=eps)

        Pc = (P32 * w32[..., None]).sum(dim=1, keepdim=True) / ws[..., None]
        Qc = (Q32 * w32[..., None]).sum(dim=1, keepdim=True) / ws[..., None]

        P0 = P32 - Pc
        Q0 = Q32 - Qc

        C = (P0 * w32[..., None]).transpose(1, 2) @ Q0  # (B,3,3) float32

        U, S, Vh = torch.linalg.svd(C)                  # OK in float32
        V = Vh.transpose(-1, -2)

        d = torch.det(V @ U.transpose(-1, -2))
        D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))

        R = V @ D @ U.transpose(-1, -2)                 # (B,3,3) float32
        t = Qc.squeeze(1) - (R @ Pc.squeeze(1)[..., None]).squeeze(-1)  # (B,3) float32

        return R, t

# =========================
# 6b) STRICT PDB writer (fixed-width ATOM records)
# =========================
# Rationale:
# - PDB ATOM/HETATM lines are **fixed-column**. Space-separated formatting can confuse
#   polymer perception / secondary-structure assignment in ChimeraX.
# - The writer below emits compliant ATOM records and multi-model (MODEL/ENDMDL) blocks.

import re as _re

def _pdb_atom_name_4(name: str) -> str:
    """Return a 4-character PDB atom name field with conservative alignment."""
    name = (name or "").strip()
    if not name:
        return "    "
    # If atom name begins with a digit, left-justify; otherwise right-justify (typical for protein atoms).
    if name[0].isdigit():
        return name.ljust(4)[:4]
    return name.rjust(4)[:4]

def _pdb_element_2(atom) -> str:
    """Return a 2-character element field (cols 77-78). Prefer topology element; fall back safely."""
    try:
        el = getattr(atom, "element", None)
        if el is not None and getattr(el, "symbol", None):
            sym = str(el.symbol).strip()
            return sym.rjust(2)[:2]
    except Exception:
        print(f"Error _pdb_element_2")
    nm = (getattr(atom, "name", None) or "").strip()
    # Use first alphabetic character from atom name: CA->C, OD1->O, SD->S
    m = _re.search(r"[A-Za-z]", nm)
    sym = m.group(0).upper() if m else ""
    return sym.rjust(2)[:2]

def write_multimodel_pdb_fixedwidth(
    out_xyz: np.ndarray,
    top: md.Topology,
    out_path: str,
    *,
    model_start: int = 1,
    occupancy: float = 1.00,
    temp_factor: float = 0.00,
    assume_nm: bool = True,
) -> None:
    """Write a multi-model PDB with strict fixed-width formatting.

    Parameters
    ----------
    out_xyz : (T, n_atoms, 3)
        Coordinates for each frame. If assume_nm=True, out_xyz is interpreted as **nm** (MDTraj convention)
        and written in **Å**. If assume_nm=False, out_xyz is interpreted as Å.
    top : md.Topology
        Topology matching n_atoms.
    out_path : str
        Output PDB file path.
    model_start : int
        First model index (ChimeraX is happiest with MODEL 1..N).
    """
    out_xyz = np.asarray(out_xyz)
    if out_xyz.ndim != 3 or out_xyz.shape[-1] != 3:
        raise ValueError(f"out_xyz must be (T,n_atoms,3); got {out_xyz.shape}")
    T, n_atoms, _ = out_xyz.shape
    if n_atoms != top.n_atoms:
        raise ValueError(f"Topology atom count {top.n_atoms} != coords atom count {n_atoms}")

    scale = 10.0 if assume_nm else 1.0

    with open(out_path, "w") as f:
        for t in range(T):
            f.write(f"MODEL     {model_start + t:4d}\n")
            serial = 1

            for atom in top.atoms:
                res = atom.residue
                chain_id = getattr(res.chain, "chain_id", None) or "A"
                chain_id = (str(chain_id).strip() or "A")[0]

                resname = (res.name or "UNK")[:3].upper()
                resseq  = int(getattr(res, "resSeq", res.index + 1))
                altloc  = " "
                icode   = " "

                x, y, z = (out_xyz[t, atom.index] * scale).tolist()

                atom_name4 = _pdb_atom_name_4(getattr(atom, "name", ""))
                element2   = _pdb_element_2(atom)

                # Use HETATM for waters/ligands; ATOM otherwise (optional but improves interoperability)
                record = "ATOM"
                if getattr(res, "is_water", False) or resname in {"HOH", "WAT"}:
                    record = "HETATM"

                line = (
                    f"{record:<6s}"
                    f"{serial:5d} "
                    f"{atom_name4}"
                    f"{altloc:1s}"
                    f"{resname:>3s} "
                    f"{chain_id:1s}"
                    f"{resseq:4d}"
                    f"{icode:1s}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{occupancy:6.2f}{temp_factor:6.2f}          "
                    f"{element2:>2s}"
                    f"\n"
                )
                f.write(line)
                serial += 1

            f.write("ENDMDL\n")
        f.write("END\n")

    print(f"[PDB] Wrote strict fixed-width multi-model PDB: {out_path} (models={T})")

# %% cell 12
# =========================
# 7) Model definition (from your training notebook)
# =========================
# The code below is copied from your training notebook (Model cell).

# --------------------------------------------------------------------------------------
# 7) Model: IPA dynamics vector field in velocity-space + optional temporal attention
# --------------------------------------------------------------------------------------

# IPA is imported lazily inside IPABlock to avoid any compiled/CUDA dependencies.


def construct(cls, **kwargs):
    sig = inspect.signature(cls.__init__)
    return cls(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def rbf(dist, D_min=2.0, D_max=20.0, D_count=64):
    centers = torch.linspace(D_min, D_max, D_count, device=dist.device).view(1, 1, 1, -1)
    widths = (D_max - D_min) / D_count
    return torch.exp(-((dist.unsqueeze(-1) - centers) / widths) ** 2)


class TemporalEncoder(nn.Module):
    """
    Per-residue temporal attention across W steps.
    Input: (B,W,N,C)
    Output: (B,W,N,C)
    """
    def __init__(self, c: int, n_layers: int, n_heads: int, dropout: float = 0.0, cast_fp32: bool = True):
        super().__init__()
        self.cast_z_fp32_after_slice = cast_fp32
        self.n_layers = int(n_layers)
        if self.n_layers <= 0:
            self.enc = None
            return
        layer = nn.TransformerEncoderLayer(
            d_model=c, nhead=n_heads, dim_feedforward=4*c, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=self.n_layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_layers <= 0:
            return x
        # x: (B,W,N,C) -> (B*N, W, C)
        B, W, N, C = x.shape
        y = x.permute(0, 2, 1, 3).contiguous().view(B * N, W, C)
        y = self.enc(y)
        y = y.view(B, N, W, C).permute(0, 2, 1, 3).contiguous()
        return y


class IPABlock(nn.Module):
    def __init__(self, c_s, c_z, cfg):
        super().__init__()
        use_ff_ipa = bool(cfg.get("USE_FOLDFLOW_IPA", True)) and (FF.ipa_pytorch is not None)
        self.cast_z_fp32_after_slice = bool(cfg.get("CAST_Z_FP32_AFTER_SLICE", True))
        inf_mask = float(cfg.get("IPA_MASK_INF", 50.0))
        if use_ff_ipa:
            ipa_conf = SimpleNamespace(
                c_s=c_s,
                c_z=c_z,
                c_hidden=cfg["IPA_C_HIDDEN"],
                no_heads=cfg["HEADS"],
                no_qk_points=cfg["NO_QK_POINTS"],
                no_v_points=cfg["NO_V_POINTS"],
            )
            self.ipa = FF.ipa_pytorch.InvariantPointAttention(ipa_conf, inf=inf_mask)
        else:
            try:
                from openfold.model.structure_module import InvariantPointAttention as OF_IPA
            except Exception as e:
                raise RuntimeError(
                    "InvariantPointAttention unavailable. Either enable FoldFlow IPA (USE_FOLDFLOW_IPA=True) "
                    "with a FoldFlow clone on sys.path, or install a pure-Python OpenFold that provides IPA."
                ) from e
            self.ipa = construct(
                OF_IPA,
                c_s=c_s,
                c_z=c_z,
                c_hidden=cfg["IPA_C_HIDDEN"],
                no_heads=cfg["HEADS"],
                no_qk_points=cfg["NO_QK_POINTS"],
                no_v_points=cfg["NO_V_POINTS"],
                inf=inf_mask,
            )
        self.ln1 = nn.LayerNorm(c_s)
        self.ff = nn.Sequential(nn.Linear(c_s, 4 * c_s), nn.GELU(), nn.Linear(4 * c_s, c_s))
        self.ln2 = nn.LayerNorm(c_s)

    def forward(self, s, z, rigids, mask):
        with torch.autocast(device_type=s.device.type, enabled=False):
            s_f = s.float()
            m_bool = (mask > 0.5) if mask.dtype != torch.bool else mask
            m_bool = m_bool.to(torch.bool)
            B, N, C = s_f.shape
            z_in = z
            ds_out = torch.zeros_like(s_f)

            def _slice_z(zin, b, idx):
                if isinstance(zin, list):
                    out = []
                    for zi in zin:
                        if zi.ndim == 4:
                            zib = zi[b:b+1].index_select(1, idx).index_select(2, idx)
                        elif zi.ndim == 3:
                            zib = zi[b:b+1].index_select(1, idx)
                        else:
                            zib = zi[b:b+1]
                        if self.cast_z_fp32_after_slice and zib.dtype != torch.float32:
                            zib = zib.float()
                        out.append(zib)
                    return out
                else:
                    if zin.ndim == 4:
                        zib = zin[b:b+1].index_select(1, idx).index_select(2, idx)
                    elif zin.ndim == 3:
                        zib = zin[b:b+1].index_select(1, idx)
                    else:
                        zib = zin[b:b+1]
                    if self.cast_z_fp32_after_slice and zib.dtype != torch.float32:
                        zib = zib.float()
                    return zib

            def _slice_rigids(r, b, idx):
                try:
                    return r[b:b+1, idx]
                except Exception:
                    print(f"_slice_rigids")
                try:
                    Rm = r.get_rots().get_rot_mats()[b:b+1, idx]
                    tm = r.get_trans()[b:b+1, idx]
                    return make_rigid(Rm, tm)
                except Exception:
                    print("_slice_rigids_2")
                rots  = getattr(r, 'rots',  None) or getattr(r, '_rots',  None) or getattr(r, 'rot', None)
                trans = getattr(r, 'trans', None) or getattr(r, '_trans', None) or getattr(r, 't',   None)
                if rots is None or trans is None:
                    raise RuntimeError("Unable to slice rigids: unknown rigid_utils.Rigid API")
                Rm_full = rots.get_rot_mats() if hasattr(rots, 'get_rot_mats') else rots
                return make_rigid(Rm_full[b:b+1, idx], trans[b:b+1, idx])

            for b in range(B):
                valid_idx = m_bool[b].nonzero(as_tuple=False).squeeze(1)
                if valid_idx.numel() == 0:
                    continue
                sb = s_f[b:b+1].index_select(1, valid_idx)
                zb = _slice_z(z_in, b, valid_idx)
                r_b = _slice_rigids(rigids, b, valid_idx)
                mask_ipa = torch.ones((1, valid_idx.numel()), device=s.device, dtype=torch.float32)
                ds_b = self.ipa(sb, zb, r_b, mask_ipa)
                if not torch.isfinite(ds_b).all():
                    raise RuntimeError(f"[IPA] Non-finite ds: b={b}, Nv={valid_idx.numel()}")
                ds_out[b:b+1].index_copy_(1, valid_idx, ds_b)

        ds = ds_out.to(dtype=s.dtype)
        s = self.ln1(s + ds)
        s = self.ln2(s + self.ff(s))
        m3 = ((mask > 0.5) if mask.dtype != torch.bool else mask).to(torch.bool)[..., None]
        s = torch.where(m3, s, torch.zeros_like(s))
        return s


def sanitize_aatype(aatype: torch.Tensor, res_mask: torch.Tensor, unk: int = 20):
    if aatype.dtype != torch.long:
        aa = aatype.long()
    else:
        aa = aatype
    valid = res_mask > 0.5
    aa = torch.where(valid, aa, aa.new_full(aa.shape, int(unk)))
    try:
        compiling = torch._dynamo.is_compiling()
    except Exception:
        compiling = False
    if not compiling:
        bad = valid & ((aa < 0) | (aa > int(unk)))
        if bad.any():
            raise ValueError(f"[AATYPE] out-of-range values: {aa[bad][:20].tolist()}")
    num_valid = valid.sum()
    num_unk   = ((aa == int(unk)) & valid).sum()
    unk_frac  = num_unk.float() / num_valid.clamp(min=1).float()
    return aa, unk_frac

class VelocityFlowIPADynamics(nn.Module):
    """
    Flow-time vector field in velocity space y = (v_local, omega_body, thdot).

    This is a *denoising*-style conditioning: for each window step k we feed the
    "previous" pose state (x_{k-1},R_{k-1},tors_{k-1}) derived from integrating the
    current FM state y_s (noisy velocities). This materially improves accuracy vs
    conditioning only on (x_t,R_t,tors_t).
    """
    def __init__(self, esm_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        c_s, c_z = int(cfg["C_S"]), int(cfg["C_Z"])

        self.aa_emb = nn.Embedding(21, c_s)
        self.esm_proj = nn.Linear(esm_dim, c_s, bias=False)

        # time embedding (flow time s) and delta embedding (step index)
        self.t_mlp = nn.Sequential(nn.Linear(1, c_s), nn.SiLU(), nn.Linear(c_s, c_s))
        self.delta_mlp = nn.Sequential(nn.Linear(1, c_s), nn.SiLU(), nn.Linear(c_s, c_s))

        # state input: pose + tors + noisy velocities
        # pose: x(3) + rot6(6) + tors_sincos(14) = 23
        # y_s: v(3) + omg(3) + thdot(7) = 13
        in_dim = 23 + 13
        self.state_mlp = nn.Sequential(nn.Linear(in_dim, c_s), nn.GELU(), nn.Linear(c_s, c_s))
        self.in_ln = nn.LayerNorm(c_s)
        # MDGen-inspired endpoint anchor conditioning (for interpolation tasks):
        # anchor_feat per residue encodes relative transform from conditioning frame to endpoint:
        #   dx_local = R0^T (x_end - x0)   (3)
        #   dR = log(R0^T R_end)          (3)
        #   dθ = wrap(θ_end - θ0)         (7) represented as sin/cos (14)
        # Total anchor feature dim = 3 + 3 + 14 = 20
        self.anchor_mlp = nn.Sequential(nn.Linear(20, c_s), nn.GELU(), nn.Linear(c_s, c_s))

        # ── Stride conditioning embed ─────────────────────────────────────────
        self.stride_embed = nn.Embedding(9, c_s)
        # ── Velocity output head(s) ──────────────────────────────────────────
        self.use_stride_heads = bool(cfg.get("USE_STRIDE_HEADS", False))
        if self.use_stride_heads:
            _strides = cfg.get("MULTI_STRIDE", [1, 2, 4])
            self.vel_heads = nn.ModuleDict({
                str(s): nn.Linear(c_s, 3 + 3 + 7) for s in _strides
            })
            self.head = None   # kept for load_state_dict compatibility; never used
        else:
            self.vel_heads = None
            self.head = nn.Linear(c_s, 3 + 3 + 7)

        self.pair_in = nn.Linear(64, c_z)

        # Relative positional bias for pair features (sequence separation)
        self.relpos_k = int(cfg.get("RELPOS_MAX", 32))
        self.relpos_emb = nn.Embedding(2 * self.relpos_k + 1, c_z)

        self.blocks = nn.ModuleList([IPABlock(c_s, c_z, cfg) for _ in range(int(cfg["IPA_BLOCKS"]))])

        n_blocks  = int(cfg.get("IPA_BLOCKS", 6))
        aux_every = int(cfg.get("AUX_EVERY", 2))
        self.aux_heads = nn.ModuleList([
            nn.Linear(c_s, 13)
            for i in range(n_blocks)
            if (i + 1) % aux_every == 0 and (i + 1) < n_blocks
        ])

        # optional temporal attention on single features before IPA
        self.use_temp = bool(cfg.get("USE_TEMPORAL_ATTENTION", False))
        if self.use_temp:
            self.temp = TemporalEncoder(
                c=c_s,
                n_layers=int(cfg.get("TEMP_ATTN_LAYERS", 0)),
                n_heads=int(cfg.get("TEMP_ATTN_HEADS", 8)),
                dropout=float(cfg.get("TEMP_ATTN_DROPOUT", 0.0)),
                cast_fp32=bool(cfg.get("CAST_Z_FP32_AFTER_SLICE", True)),
            )
        else:
            self.temp = None

    @staticmethod
    def rot_to_6d(R: torch.Tensor) -> torch.Tensor:
        return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)  # (...,6)

    def pair_embed(self, x: torch.Tensor, detach_x: bool = True) -> torch.Tensor:
        # x: (B?, N, 3)
        xg = x.detach() if detach_x else x

        dist = torch.cdist(xg, xg)  # (B?,N,N)

        # keep trainable params (pair_in, relpos_emb) fully differentiable
        phi = rbf(dist.to(dtype=self.pair_in.weight.dtype))  # (B?,N,N,D)
        z   = self.pair_in(phi)                              # (B?,N,N,Cz)

        N = x.shape[-2]
        ridx = torch.arange(N, device=x.device)
        rel  = (ridx[None, :] - ridx[:, None]).clamp(-self.relpos_k, self.relpos_k) + self.relpos_k
        rel_z = self.relpos_emb(rel).unsqueeze(0).to(dtype=z.dtype)

        return z + rel_z

    def forward(
        self,
        aatype: torch.Tensor,   # (B,N)
        esm: torch.Tensor,      # (B,N,D)
        x_prev: torch.Tensor,   # (B,W,N,3)
        R_prev: torch.Tensor,   # (B,W,N,3,3)
        tors_prev: torch.Tensor,# (B,W,N,7)
        mask_prev: torch.Tensor,# (B,W,N)
        v_s: torch.Tensor,      # (B,W,N,3)
        omg_s: torch.Tensor,    # (B,W,N,3)
        thdot_s: torch.Tensor,  # (B,W,N,7)
        flow_s: torch.Tensor,   # (B,W,1) or (B,1)
        delta_t: Optional[torch.Tensor] = None,   # (W,) or (B,W) in physical units
        delta_idx: Optional[torch.Tensor] = None, # DEPRECATED: step index (used if delta_t is None)
        anchor_feat: Optional[torch.Tensor] = None,        # (B,N,20) endpoint anchor features (interp only)
        z_pair: Optional[torch.Tensor] = None,     # (B,W,N,N,Cz) or (B,1,N,N,Cz)
        stride: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, W, N, _ = x_prev.shape
        m = mask_prev.unsqueeze(-1)

        res_mask_bool = (mask_prev[:, 0, :] > 0.5)
        aa_idx, _ = sanitize_aatype(aatype, res_mask_bool, unk=20)
        aa = self.aa_emb(aa_idx)[:, None, :, :].expand(B, W, N, -1)
        esm = esm.to(dtype=self.esm_proj.weight.dtype)  # match projection weight dtype
        e = self.esm_proj(esm)[:, None, :, :].expand(B, W, N, -1)


        tors_sc = torch.cat([torch.sin(tors_prev), torch.cos(tors_prev)], dim=-1)  # (B,W,N,14)
        st = torch.cat([x_prev, self.rot_to_6d(R_prev), tors_sc, v_s, omg_s, thdot_s], dim=-1)
        st = self.state_mlp(st)

        # If endpoint anchor is provided (interpolation task), inject it as an additive conditioning term.
        # This conditions the entire window on the desired endpoint relative transform, discouraging drift.
        if (anchor_feat is not None) and bool(self.cfg.get("USE_ANCHOR_EMB", True)):
            # anchor_feat: (B,N,20) -> (B,W,N,C)
            a = self.anchor_mlp(anchor_feat.float())[:, None, :, :].expand(B, W, N, -1)
            st = st + a

        # flow time embedding
        if flow_s.ndim == 2:  # (B,1)
            t_in = flow_s[:, None, :].expand(B, W, 1)  # (B,W,1)
        else:
            t_in = flow_s  # (B,W,1)
        t_emb = self.t_mlp(t_in.reshape(B * W, 1)).view(B, W, 1, -1).expand(B, W, N, -1)

        # delta (physical) embedding
        if delta_t is None:
            # Fallback to step indices if no physical delta provided
            if delta_idx is None:
                delta_idx = torch.arange(1, W + 1, device=x_prev.device).view(1, W).expand(B, W)
            elif delta_idx.ndim == 1:
                delta_idx = delta_idx.view(1, W).expand(B, W)
            delta_t = delta_idx.to(dtype=x_prev.dtype)
        else:
            if delta_t.ndim == 1:
                delta_t = delta_t.view(1, W).expand(B, W)
            else:
                delta_t = delta_t.to(dtype=x_prev.dtype)

        dt_scale = float(self.cfg.get("DELTA_T_SCALE", 1.0))
        delta_in = (delta_t / max(dt_scale, 1e-8)).reshape(B * W, 1)
        d_emb = self.delta_mlp(delta_in).view(B, W, 1, -1).expand(B, W, N, -1)

        stride_idx = torch.tensor(
            min(int(stride), self.stride_embed.num_embeddings - 1),
            device=x_prev.device, dtype=torch.long
        )
        str_emb = self.stride_embed(stride_idx).view(1, 1, 1, -1).expand(B, W, N, -1)
        s_feat = self.in_ln(aa + e + st + t_emb + d_emb + str_emb) * m

        # temporal attention (per residue) before IPA
        if self.use_temp and self.temp is not None:
            s_feat = self.temp(s_feat)

        # flatten steps for IPA blocks
        # s = s_feat.reshape(B * W, N, -1)
        # mask_flat = mask_prev.reshape(B * W, N)
        # x_flat = x_prev.reshape(B * W, N, 3)
        # R_flat = R_prev.reshape(B * W, N, 3, 3)

        step_chunk = int(self.cfg.get("STEP_CHUNK", 1))
        outs = []

        if self.use_stride_heads and self.vel_heads is not None:
            _stride_key = str(stride)
            if _stride_key not in self.vel_heads:
                _stride_key = str(min(self.vel_heads.keys(), key=lambda k: abs(int(k) - stride)))
            _out_head = self.vel_heads[_stride_key]
        else:
            _out_head = self.head


        for k0 in range(0, W, step_chunk):
            k1 = min(W, k0 + step_chunk)
            C = k1 - k0  # chunk size

            # chunk views: (B,C,N,·) -> (B*C,N,·)
            s_chunk   = s_feat[:, k0:k1].reshape(B * C, N, -1)
            mask_flat = mask_prev[:, k0:k1].reshape(B * C, N)
            x_flat    = x_prev[:, k0:k1].reshape(B * C, N, 3)
            R_flat    = R_prev[:, k0:k1].reshape(B * C, N, 3, 3)
            rigids    = make_rigid(R_flat, x_flat)

            # pair features for this chunk only
            if z_pair is None:
                z = self.pair_embed(x_flat, detach_x=bool(self.cfg.get("DETACH_PAIR_X", True)))
            else:
                # allow shared (B,1,N,N,Cz) or per-step (B,W,N,N,Cz)
                if z_pair.ndim == 5 and z_pair.shape[1] == 1:
                    z = z_pair[:, 0].unsqueeze(1).expand(B, C, N, N, -1).reshape(B * C, N, N, -1)
                else:
                    z = z_pair[:, k0:k1].reshape(B * C, N, N, -1)

            for blk in self.blocks:
                s_chunk = blk(s_chunk, z, rigids, mask_flat)

            out_chunk = _out_head(s_chunk).view(B, C, N, -1)
            outs.append(out_chunk)

        out = torch.cat(outs, dim=1)   # (B,W,N,13)
        u_v     = out[..., :3]
        u_omg   = out[..., 3:6]
        u_thdot = out[..., 6:]
        return u_v, u_omg, u_thdot

# %% cell 13
# =========================
# 8) OpenFold atom14 builder (from your training notebook)
# =========================
# Used to reconstruct atom14 coordinates from (x, R, torsions).
# If OpenFold is not importable, inference still runs but output falls back to CA-only updates.

def first_non_none_attr(obj, names: List[str]):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None


class OpenFoldAtom14Builder:
    """
    A robust wrapper around OpenFold's torsion->frames->atom14 pipeline that tolerates
    minor signature differences between forks/versions.
    """
    def __init__(self, residue_constants, rigid_utils_mod, feats_mod, device="cuda"):
        self.rc = residue_constants
        self.rigid_utils = rigid_utils_mod
        self.feats = feats_mod
        self.device = device

        def T(x, dtype=None):
            if torch.is_tensor(x):
                t = x
                if dtype is not None and t.dtype != dtype:
                    t = t.to(dtype)
                return t.to(device)
            return torch.as_tensor(x, dtype=dtype, device=device)

        # FULL tables
        self.rrgdf = T(self.rc.restype_rigid_group_default_frame, dtype=torch.float32)   # (21,8,4,4)
        self.atom14_mask_table = T(self.rc.restype_atom14_mask, dtype=torch.float32)    # (21,14)
        self.atom14_to_group_table = T(self.rc.restype_atom14_to_rigid_group, dtype=torch.long)  # (21,14)
        self.atom14_litpos_table   = T(self.rc.restype_atom14_rigid_group_positions, dtype=torch.float32)  # (21,14,3)

        # Optional mapping tables
        self.atom14_to_group = None
        if hasattr(self.rc, "restype_atom14_to_rigid_group"):
            self.atom14_to_group = T(getattr(self.rc, "restype_atom14_to_rigid_group"), dtype=torch.long)  # (21,14)
        elif hasattr(self.rc, "restype_atom14_to_rigid_group_idx"):
            self.atom14_to_group = T(getattr(self.rc, "restype_atom14_to_rigid_group_idx"), dtype=torch.long)

        self.group_pos = first_non_none_attr(self.rc, [
            "restype_atom14_rigid_group_positions",
            "restype_atom14_rigid_group_positions",
        ])
        self.group_pos = T(self.group_pos, dtype=torch.float32)  # (21,14,3)

        # Ambiguity mask (optional)
        amb_mask = first_non_none_attr(self.rc, [
            "restype_atom14_ambiguous_atoms",
            "restype_atom14_atom_is_ambiguous",
        ])
        self.ambig_mask = T(amb_mask, torch.float32) if amb_mask is not None else None

    def _call_torsion_angles_to_frames(self, fn, aatype, backb_rigid, torsion_sc):
        sig = inspect.signature(fn)
        p = sig.parameters
        kw = {}

        # rigid arg name variants
        if "r" in p:
            kw["r"] = backb_rigid
        elif "backb_rigid" in p:
            kw["backb_rigid"] = backb_rigid
        elif "rigid" in p:
            kw["rigid"] = backb_rigid

        # torsion sin/cos arg name variants
        if "torsion_angles_sin_cos" in p:
            kw["torsion_angles_sin_cos"] = torsion_sc
        elif "torsions" in p:
            kw["torsions"] = torsion_sc
        elif "alpha" in p:
            kw["alpha"] = torsion_sc

        if "aatype" in p:
            kw["aatype"] = aatype

        # IMPORTANT: FoldFlow variant expects `rrgdf`
        if "rrgdf" in p:
            kw["rrgdf"] = self.rrgdf
        if "restype_rigid_group_default_frame" in p:
            kw["restype_rigid_group_default_frame"] = self.rrgdf
        if "default_frames" in p:
            kw["default_frames"] = self.rrgdf

        # robust call (keyword first, positional fallback if needed)
        try:
            return fn(**kw)
        except TypeError:
            args = []
            for name in p.keys():
                if name in kw:
                    args.append(kw[name])
                else:
                    raise
            return fn(*args)

    def _call_frames_to_atom14(self, fn, aatype, all_frames):
        p = inspect.signature(fn).parameters
        kw = {}

        # Rigid frames argument
        if "r" in p:
            kw["r"] = all_frames
        if "all_frames" in p:
            kw["all_frames"] = all_frames
        if "frames" in p:
            kw["frames"] = all_frames

        # residue types
        if "aatype" in p:
            kw["aatype"] = aatype

        # default frames table
        if "default_frames" in p:
            kw["default_frames"] = self.rrgdf
        if "rrgdf" in p:
            kw["rrgdf"] = self.rrgdf
        if "restype_rigid_group_default_frame" in p:
            kw["restype_rigid_group_default_frame"] = self.rrgdf

        # atom14 -> rigid-group mapping table
        if "group_idx" in p:
            kw["group_idx"] = self.atom14_to_group_table
        if "atom14_to_rigid_group" in p:
            kw["atom14_to_rigid_group"] = self.atom14_to_group_table
        if "restype_atom14_to_rigid_group" in p:
            kw["restype_atom14_to_rigid_group"] = self.atom14_to_group_table

        # atom mask table
        if "atom_mask" in p:
            kw["atom_mask"] = self.atom14_mask_table
        if "atom14_mask" in p:
            kw["atom14_mask"] = self.atom14_mask_table
        if "restype_atom14_mask" in p:
            kw["restype_atom14_mask"] = self.atom14_mask_table

        # literature positions table
        if "lit_positions" in p:
            kw["lit_positions"] = self.atom14_litpos_table
        if "restype_atom14_rigid_group_positions" in p:
            kw["restype_atom14_rigid_group_positions"] = self.atom14_litpos_table

        return fn(**kw)

    def build_atom14(self, aatype, R, x, torsion_angles) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        aatype: (B,N)
        R: (B,N,3,3)
        x: (B,N,3)
        torsion_angles: (B,N,7) radians
        Returns:
          atom14_pos: (B,N,14,3)
          atom14_mask: (B,N,14)
        """
        B, N = aatype.shape
        device = x.device

        # sin/cos torsions
        tors_sc = torch.stack([torch.sin(torsion_angles), torch.cos(torsion_angles)], dim=-1)  # (B,N,7,2)

        backb_rigid = make_rigid(R, x)  # Rigid per residue
        # Prefer FoldFlow's all_atom implementation if present (robust atom14 + consistent supervision).
        if (FF.all_atom is not None) and hasattr(FF.all_atom, "torsion_angles_to_frames") and hasattr(FF.all_atom, "frames_to_atom14_pos"):
            try:
                # Use signature-safe call instead of positional args.
                # FoldFlow variants differ in arg order; _call_torsion_angles_to_frames
                # inspects the signature and passes keyword args, which is safe across forks.
                fn_t2f = FF.all_atom.torsion_angles_to_frames
                fn_f2a = FF.all_atom.frames_to_atom14_pos
                aatype_cpu = aatype.cpu()

                all_frames = self._call_torsion_angles_to_frames(
                    FF.all_atom.torsion_angles_to_frames,
                    aatype_cpu, backb_rigid, tors_sc
                )

                atom14_pos  = FF.all_atom.frames_to_atom14_pos(all_frames, aatype_cpu)
                atom14_mask = self.atom14_mask_table[aatype_cpu].to(x.device)

                return atom14_pos, atom14_mask
            except Exception as _e:
                print(f"[WARN] FoldFlow atom14 fast-path failed ({_e!r}); falling back to OpenFold feats path.")


        torsion_angles_to_frames = getattr(self.feats, "torsion_angles_to_frames", None)
        frames_to_atom14 = getattr(self.feats, "frames_and_literature_positions_to_atom14_pos", None)
        if torsion_angles_to_frames is None or frames_to_atom14 is None:
            raise RuntimeError("OpenFold feats missing torsion->frames or frames->atom14 functions")

        all_frames = self._call_torsion_angles_to_frames(torsion_angles_to_frames, aatype, backb_rigid, tors_sc)
        atom14_pos = self._call_frames_to_atom14(frames_to_atom14, aatype, all_frames)

        # mask table lookup
        atom14_mask = self.atom14_mask_table[aatype.clamp(0, 20)]
        return atom14_pos, atom14_mask


def call_with_sig(fn, args: Dict[str, Any]):
    sig = inspect.signature(fn)
    kw = {k: v for k, v in args.items() if k in sig.parameters}
    return fn(**kw)


def openfold_fape_atom14(
    pred_R: torch.Tensor,
    pred_x: torch.Tensor,
    pred_a14: torch.Tensor,
    pred_a14m: torch.Tensor,
    gt_R: torch.Tensor,
    gt_x: torch.Tensor,
    gt_a14: torch.Tensor,
    gt_a14m: torch.Tensor,
    res_mask: torch.Tensor,
    length_scale: float = 10.0,
    clamp_distance: float = 10.0,
    eps: float = 1e-4,
    return_stats: bool = False,
) -> torch.Tensor:
    """OpenFold compute_fape on atom14 points averaged over batch.

    Frames: residue frames (pred_R/pred_x vs gt_R/gt_x)
    Points: atom14 positions (flattened)

    If return_stats=True, returns (fape_clamped_mean, stats_dict) where stats_dict includes:
      - fape_clamp_frac: fraction of points that hit the L1 clamp
      - fape_raw_p95 / p99: high quantiles of the *unclamped* per-point error (Å)
    """
    if (of_loss is None) or (not hasattr(of_loss, "compute_fape")):
        out = torch.tensor(0.0, device=pred_x.device)
        return (out, dict(fape_clamp_frac=0.0, fape_raw_p95=0.0, fape_raw_p99=0.0)) if return_stats else out
    try:
        Rot = rigid_utils.Rotation
        Rigid = rigid_utils.Rigid
    except Exception:
        out = torch.tensor(0.0, device=pred_x.device)
        return (out, dict(fape_clamp_frac=0.0, fape_raw_p95=0.0, fape_raw_p99=0.0)) if return_stats else out

    B, N, _, _ = pred_R.shape
    res_mask = res_mask.clamp(0, 1)

    # Effective point masks (B, N*14)
    p_mask = (pred_a14m * gt_a14m * res_mask[..., None]).reshape(B, N * 14)
    pred_pts = pred_a14.reshape(B, N * 14, 3)
    gt_pts = gt_a14.reshape(B, N * 14, 3)

    pred_frame = Rigid(Rot(rot_mats=pred_R, quats=None), pred_x)
    gt_frame = Rigid(Rot(rot_mats=gt_R, quats=None), gt_x)

    # Clamped FAPE (training objective)
    f_clamped = of_loss.compute_fape(
        pred_frames=pred_frame,
        target_frames=gt_frame,
        frames_mask=res_mask,
        pred_positions=pred_pts,
        target_positions=gt_pts,
        positions_mask=p_mask,
        length_scale=float(length_scale),
        l1_clamp_distance=float(clamp_distance),
        eps=float(eps),
    )
    fape_mean = torch.mean(f_clamped)

    if not return_stats:
        return fape_mean

    # Unclamped FAPE for diagnostics (use a huge clamp)
    big = float(clamp_distance) * 1e6
    f_raw = of_loss.compute_fape(
        pred_frames=pred_frame,
        target_frames=gt_frame,
        frames_mask=res_mask,
        pred_positions=pred_pts,
        target_positions=gt_pts,
        positions_mask=p_mask,
        length_scale=float(length_scale),
        l1_clamp_distance=float(big),
        eps=float(eps),
    )

    # Estimate clamp-hit fraction approximately: compare raw distance vs clamp_distance
    # f_raw is scaled by length_scale inside OF; we undo scaling to get Å.
    # NOTE: This is approximate but robust enough as a monitoring signal.
    with torch.no_grad():
        fr = f_raw.detach().reshape(-1)
        finite = torch.isfinite(fr)
        if finite.any():
            fr = fr[finite].to(torch.float32)
            # raw per-point error in Å (approx)
            # compute_fape returns error / length_scale; so multiply back
            fr_A = fr * float(length_scale)
            clamp_frac = float((fr_A > float(clamp_distance)).float().mean().item())
            p95 = float(torch.quantile(fr_A, 0.95).item())
            p99 = float(torch.quantile(fr_A, 0.99).item())
        else:
            clamp_frac, p95, p99 = 0.0, 0.0, 0.0

    stats = dict(fape_clamp_frac=clamp_frac, fape_raw_p95=p95, fape_raw_p99=p99)
    return fape_mean, stats


def steric_clash_loss(atom14_pos, atom14_mask, res_mask, cutoff=2.0, exclude_k=2):
    """
    Simple steric clash penalty between atom14 positions, excluding near neighbors.
    atom14_pos: (B,N,14,3)
    atom14_mask: (B,N,14)
    res_mask: (B,N)
    """
    B, N, A, _ = atom14_pos.shape
    M = (res_mask > 0.5).float()
    atomM = (atom14_mask > 0.5).float()
    M = (M.unsqueeze(-1) * atomM).reshape(B, N * A)  # (B, NA)
    P = atom14_pos.reshape(B, N * A, 3)
    dist = torch.cdist(P, P)

    # exclude same residue and neighbors within exclude_k
    ridx = torch.arange(N, device=dist.device)
    neigh = (torch.abs(ridx[None, :] - ridx[:, None]) <= exclude_k).unsqueeze(0).repeat(B, 1, 1)
    neighA = neigh.repeat_interleave(A, dim=1).repeat_interleave(A, dim=2)

    valid = (M.unsqueeze(-1) > 0.5) & (M.unsqueeze(-2) > 0.5)
    clash = (dist < cutoff) & valid & (~neighA)

    pen = F.softplus((cutoff - dist) * 2.0)
    return (pen * clash.float()).sum() / (clash.float().sum() + 1e-6)


def bond_loss(pA, pB, maskAB, target):
    d = torch.linalg.norm(pA - pB, dim=-1)
    return ((d - target) ** 2 * maskAB).sum() / (maskAB.sum() + 1e-6)


def angle_loss(pA, pB, pC, maskABC, target_rad):
    v1 = F.normalize(pA - pB, dim=-1, eps=1e-6)
    v2 = F.normalize(pC - pB, dim=-1, eps=1e-6)
    ang = torch.acos((v1 * v2).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    return ((ang - target_rad) ** 2 * maskABC).sum() / (maskABC.sum() + 1e-6)


def dihedral(p1, p2, p3, p4):
    b0 = p2 - p1
    b1 = p3 - p2
    b2 = p4 - p3
    b1n = F.normalize(b1, dim=-1, eps=1e-6)
    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


# %% cell 14
# =========================
# 9) Noise + integration helpers (from your training notebook)
# =========================

def sample_ar1_noise(shape: Tuple[int, ...], rho: float, device, dtype):
    """
    shape: (B,W,N,C)
    AR(1) along W dimension for each (B,N,C).
    """
    assert len(shape) == 4
    B, W, N, C = shape
    eps = torch.randn((B, W, N, C), device=device, dtype=dtype)
    if rho <= 0.0:
        return eps
    out = torch.zeros_like(eps)
    out[:, 0] = eps[:, 0]
    scale = math.sqrt(max(1e-8, 1.0 - rho * rho))
    for k in range(1, W):
        out[:, k] = rho * out[:, k - 1] + scale * eps[:, k]
    return out


def integrate_velocities_to_window(
    x0: torch.Tensor, R0: torch.Tensor, tors0: torch.Tensor,
    v_local: torch.Tensor, omg: torch.Tensor, thdot: torch.Tensor,
    dt: float,
    anchor_alpha: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Anchor-corrected Euler integration.
    anchor_alpha=0.0 → pure Euler (no change from previous behaviour).
    anchor_alpha>0.0 → linearly-decaying SO(3)-blended pull back toward
                       the conditioning frame (x0, R0, tors0) at each step.
    """
    B, W, N, _ = v_local.shape
    x    = x0
    R    = R0
    tors = tors0
    xs, Rs, torss = [], [], []

    for k in range(W):
        v_k   = v_local[:, k]
        omg_k = omg[:, k]
        th_k  = thdot[:, k]

        # Standard Euler step
        dx_g       = (R @ (v_k * dt).unsqueeze(-1)).squeeze(-1)
        x_euler    = x + dx_g
        R_euler    = R @ so3_exp((omg_k * dt).reshape(-1, 3)).reshape(B, N, 3, 3)
        tors_euler = wrap_to_pi(tors + th_k * dt)

        if anchor_alpha > 0.0:
            # alpha decays linearly: anchor_alpha at k=0, 0 at k=W-1
            den     = max(W - 1, 1)
            alpha_k = anchor_alpha * (1.0 - float(k) / float(den))

            # Translation: linear blend toward x0
            x = (1.0 - alpha_k) * x_euler + alpha_k * x0

            # Rotation: blend in SO(3) tangent space
            #   R_blend = R0 @ exp((1 - alpha_k) * log(R0^T @ R_euler))
            dR     = R0.transpose(-1, -2) @ R_euler        # (B,N,3,3)
            log_dR = so3_log(dR.reshape(-1, 3, 3))          # (B*N,3)
            R = R0 @ so3_exp(
                ((1.0 - alpha_k) * log_dR).reshape(-1, 3)
            ).reshape(B, N, 3, 3)

            # Torsions: linear blend toward tors0
            tors = wrap_to_pi((1.0 - alpha_k) * tors_euler + alpha_k * tors0)
        else:
            x    = x_euler
            R    = R_euler
            tors = tors_euler

        xs.append(x)
        Rs.append(R)
        torss.append(tors)

    return torch.stack(xs, dim=1), torch.stack(Rs, dim=1), torch.stack(torss, dim=1)


def integrate_prev_states_from_velocities(
    x0: torch.Tensor, R0: torch.Tensor, tors0: torch.Tensor,
    v_local: torch.Tensor, omg: torch.Tensor, thdot: torch.Tensor,
    dt: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Produce per-step "previous" states (x_{k-1},R_{k-1},tors_{k-1}) for k=1..W by integrating
    velocities up to k-1.

    Returns:
      x_prev, R_prev, tors_prev: (B,W,N,...) aligned with velocities at step k
    """
    B, W, N, _ = v_local.shape
    x = x0
    R = R0
    tors = tors0
    xs = []
    Rs = []
    torss = []
    for k in range(W):
        xs.append(x)
        Rs.append(R)
        torss.append(tors)
        v_k = v_local[:, k]
        omg_k = omg[:, k]
        th_k = thdot[:, k]
        dx_g = (R @ (v_k * dt).unsqueeze(-1)).squeeze(-1)
        x = x + dx_g
        R = R @ so3_exp((omg_k * dt).reshape(-1, 3)).reshape(B, N, 3, 3)
        tors = wrap_to_pi(tors + th_k * dt)
    return torch.stack(xs, dim=1), torch.stack(Rs, dim=1), torch.stack(torss, dim=1)


# --------------------------------------------------------------------------------------
# 10) EMA helper
# --------------------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

# %% cell 15
# =========================
# 10) ESM embedding (on-the-fly)
# =========================
import esm

@torch.no_grad()
def compute_esm_embedding(seq: str, model_name: str, layer: int, device: str):
    # fair-esm pretrained loader by attribute name
    loader = getattr(esm.pretrained, model_name)
    model, alphabet = loader()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    _, _, toks = batch_converter([("p", seq)])
    toks = toks.to(device)
    out = model(toks, repr_layers=[layer], return_contacts=False)
    reps = out["representations"][layer][0]  # [L+2, D]
    emb = reps[1:1+len(seq)].detach().cpu().numpy().astype(np.float32)
    return emb

# %% cell 16
# =========================
# 11) Inference sampler (rectified-flow/FM ODE in velocity space)
# =========================

def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict produced by your trainer.")
    cfg = ckpt.get("cfg", {})
    # prefer EMA weights if present
    if isinstance(ckpt.get("ema", None), dict):
        state = ckpt["ema"]
    else:
        state = ckpt.get("model", ckpt.get("state_dict", None))
    if state is None:
        raise ValueError(f"Could not find model weights in checkpoint keys: {list(ckpt.keys())}")
    return cfg, state


def canonicalize_state_dict(state: dict) -> dict:
    """Normalize checkpoint key prefixes (DDP/torch.compile wrappers) for inference."""
    if not isinstance(state, dict):
        raise TypeError(f"state must be a dict, got {type(state)}")
    if len(state) == 0:
        return state

    def strip_prefix_if_majority(sd: dict, prefix: str, frac: float = 0.8) -> dict:
        keys = list(sd.keys())
        n = len(keys)
        n_pref = sum(1 for k in keys if isinstance(k, str) and k.startswith(prefix))
        if n > 0 and (n_pref / n) >= frac:
            out = {}
            for k, v in sd.items():
                nk = k[len(prefix):] if isinstance(k, str) and k.startswith(prefix) else k
                # avoid accidental collisions: keep last write deterministic
                out[nk] = v
            return out
        return sd

    # Common wrappers:
    # - DDP: "module."
    # - torch.compile: "_orig_mod."
    # - Some trainers: "model."
    for pref in ("module.", "_orig_mod.", "model."):
        state = strip_prefix_if_majority(state, pref)

    return state

def infer_esm_dim_from_state(state: dict):
    """Best-effort infer ESM dim from projection weight; returns int or None."""
    # Typical key: "esm_proj.weight" (out_dim, esm_dim)
    cand = []
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        lk = str(k).lower()
        if ("esm" in lk) and ("proj" in lk) and lk.endswith("weight") and v.ndim == 2:
            cand.append((k, v.shape))
    if len(cand) == 0:
        return None
    # choose the largest second dimension as esm_dim
    cand.sort(key=lambda x: int(x[1][1]), reverse=True)
    return int(cand[0][1][1])


def normalize_velocities(v, omg, thdot, cfg, stride: int = None):
    if not bool(cfg.get("NORMALIZE_VELOCITIES", True)):
        return v, omg, thdot
    if stride is not None:
        v_std  = float(cfg.get("V_STD_BY_STRIDE",     {}).get(stride, cfg.get("V_STD",     1.0)))
        o_std  = float(cfg.get("OMG_STD_BY_STRIDE",   {}).get(stride, cfg.get("OMG_STD",   1.0)))
        th_std = float(cfg.get("THDOT_STD_BY_STRIDE", {}).get(stride, cfg.get("THDOT_STD", 1.0)))
    else:
        v_std  = float(cfg.get("V_STD",     1.0))
        o_std  = float(cfg.get("OMG_STD",   1.0))
        th_std = float(cfg.get("THDOT_STD", 1.0))
    return v / v_std, omg / o_std, thdot / th_std

def denormalize_velocities(v, omg, thdot, cfg, stride: int = None):
    if not bool(cfg.get("NORMALIZE_VELOCITIES", True)):
        return v, omg, thdot
    if stride is not None:
        v_std  = float(cfg.get("V_STD_BY_STRIDE",     {}).get(stride, cfg.get("V_STD",     1.0)))
        o_std  = float(cfg.get("OMG_STD_BY_STRIDE",   {}).get(stride, cfg.get("OMG_STD",   1.0)))
        th_std = float(cfg.get("THDOT_STD_BY_STRIDE", {}).get(stride, cfg.get("THDOT_STD", 1.0)))
    else:
        v_std  = float(cfg.get("V_STD",     1.0))
        o_std  = float(cfg.get("OMG_STD",   1.0))
        th_std = float(cfg.get("THDOT_STD", 1.0))
    return v * v_std, omg * o_std, thdot * th_std

@torch.no_grad()
def sample_window_velocities(
    model,
    cfg,
    aatype, esm,
    x_c, R_c, tors_c,
    res_mask,
    dt: float,
    W: int,
    n_flow_steps: int,
    use_amp: bool,
    stride: int = 1,
):
    device = x_c.device
    dtype  = x_c.dtype
    B, N = aatype.shape

    # ── Temperature-scaled initial noise ─────────────────────────────────────
    temp = float(cfg.get("TEMPERATURE", 1.0))
    rho = float(cfg.get("NOISE_AR1_RHO", 0.0))
    v = sample_ar1_noise((B, W, N, 3), rho, device, dtype) * float(cfg.get("SIGMA_V", 1.0)) * temp
    o = sample_ar1_noise((B, W, N, 3), rho, device, dtype) * float(cfg.get("SIGMA_OMG", 1.0)) * temp
    t = sample_ar1_noise((B, W, N, 7), rho, device, dtype) * float(cfg.get("SIGMA_THDOT", 1.0)) * temp

    alpha = float(cfg.get("SIGMA_STEP_ALPHA", 0.0))
    if alpha != 0.0 and W > 1:
        k = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, W, 1, 1)
        scale = 1.0 + alpha * k
        v *= scale
        o *= scale
        t *= scale

    delta_t = torch.arange(1, W + 1, device=device, dtype=dtype) * float(dt)

    z_pair = None
    if bool(cfg.get("USE_SHARED_PAIR_Z", True)):
        z_pair = model.pair_embed(x_c, detach_x=bool(cfg.get("DETACH_PAIR_X", True))).unsqueeze(1)

    mask_prev = res_mask[:, None, :].expand(B, W, -1).contiguous()

    for i in range(n_flow_steps):
        s = (i + 0.5) / float(n_flow_steps)
        flow_s = torch.full((B, 1), s, device=device, dtype=dtype)

        v_phys, o_phys, t_phys = denormalize_velocities(v, o, t, cfg, stride=stride)
        x_prev, R_prev, tors_prev = integrate_prev_states_from_velocities(
            x_c, R_c, tors_c, v_phys, o_phys, t_phys, dt=dt
        )

        amp_enabled = bool(use_amp)
        amp_dtype = torch.bfloat16
        if device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                print("[WARN] CUDA bf16 not supported on this GPU; disabling autocast (running fp32).")
                amp_enabled = False
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            u_v, u_o, u_t = model(
                aatype=aatype,
                esm=esm,
                x_prev=x_prev,
                R_prev=R_prev,
                tors_prev=tors_prev,
                mask_prev=mask_prev,
                v_s=v,
                omg_s=o,
                thdot_s=t,
                flow_s=flow_s,
                delta_t=delta_t,
                z_pair=z_pair,
                stride=stride,
            )[:3]  # drop aux_outputs if present

        u_v = u_v.to(dtype)
        u_o = u_o.to(dtype)
        u_t = u_t.to(dtype)

        ds = 1.0 / float(n_flow_steps)
        v = v + ds * u_v
        o = o + ds * u_o
        t = t + ds * u_t

    return denormalize_velocities(v, o, t, cfg, stride=stride)

def update_xyz_with_atom14(xyz_nm, atom14_pos_A, atom14_indices):
    # xyz_nm: (n_atoms,3) in nm
    for i in range(atom14_indices.shape[0]):
        for j in range(14):
            aidx = int(atom14_indices[i, j])
            if aidx >= 0:
                xyz_nm[aidx, :] = atom14_pos_A[i, j, :] / 10.0

def update_xyz_with_ca_only(xyz_nm, x_A, ca_idx):
    for i, aidx in enumerate(ca_idx):
        aidx = int(aidx)
        if aidx >= 0:
            xyz_nm[aidx, :] = x_A[i] / 10.0

# %% cell 17
# =========================
# 12) Run inference and write multi-frame PDB (MDTraj)
# =========================

# Reproducibility
torch.manual_seed(int(INF_CFG["SEED"]))
np.random.seed(int(INF_CFG["SEED"]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load input structure and strip H atoms in memory only
_traj_raw = md.load(PDB_PATH)
_heavy_idx = _traj_raw.topology.select('not element H')

if len(_heavy_idx) < _traj_raw.n_atoms:
    _n_h = _traj_raw.n_atoms - len(_heavy_idx)
    traj0 = _traj_raw.atom_slice(_heavy_idx)   # in-memory only, no file written
    print(f"[INFO] Stripped {_n_h} H atoms in memory. Continuing with heavy atoms only.")
else:
    traj0 = _traj_raw
    print("[INFO] No H atoms found.")

top      = traj0.topology
chain    = resolve_chain(top, INF_CFG["CHAIN_ID"])
residues = list(chain.residues)

atom14_indices, n_idx, ca_idx, c_idx, seq, aatype_np, pdb_resseq = build_atom_index_maps(residues)

# coords in Å for frame 0
xyz0_A = traj0.xyz[0] * 10.0

N_xyz  = np.zeros((1, len(residues), 3), dtype=np.float32)
CA_xyz = np.zeros((1, len(residues), 3), dtype=np.float32)
C_xyz  = np.zeros((1, len(residues), 3), dtype=np.float32)
for i in range(len(residues)):
    if n_idx[i] >= 0:
        N_xyz[0, i] = xyz0_A[n_idx[i]]
    if ca_idx[i] >= 0:
        CA_xyz[0, i] = xyz0_A[ca_idx[i]]
    if c_idx[i] >= 0:
        C_xyz[0, i] = xyz0_A[c_idx[i]]

R0_np, x0_np, frame_mask = compute_backbone_frames(N_xyz, CA_xyz, C_xyz)
R0_np[frame_mask == 0] = np.eye(3, dtype=np.float32)

tors0_np, _ = compute_initial_torsions(traj0, residues, n_idx, ca_idx, c_idx, aatype_np, pdb_resseq)

# ---- Build tensors
aatype = torch.from_numpy(aatype_np[None, :]).to(device=device, dtype=torch.long)
res_mask = torch.from_numpy(frame_mask[0][None, :].astype(np.float32)).to(device=device, dtype=torch.float32)
x_c   = torch.from_numpy(x0_np[0][None, :, :]).to(device=device, dtype=torch.float32)
R_c   = torch.from_numpy(R0_np[0][None, :, :, :]).to(device=device, dtype=torch.float32)
tors_c= torch.from_numpy(tors0_np[None, :, :]).to(device=device, dtype=torch.float32)

# ---- ESM embedding (map unknown residues to A for ESM only)
esm_seq = seq.replace("X", "A")
esm_np  = compute_esm_embedding(
    esm_seq,
    model_name=str(INF_CFG["ESM_MODEL_NAME"]),
    layer=int(INF_CFG["ESM_LAYER"]),
    device=str(device),
)
esm = torch.from_numpy(esm_np[None, :, :]).to(device=device, dtype=torch.float32)

# ---- Load checkpoint and build model
ckpt_cfg, state = load_checkpoint(CKPT_PATH)
state = canonicalize_state_dict(state)

# Merge configs: checkpoint cfg -> override
cfg = dict(ckpt_cfg) if isinstance(ckpt_cfg, dict) else {}
cfg.update(INF_CFG.get("OVERRIDE", {}))

required = ["NORMALIZE_VELOCITIES","V_STD","OMG_STD","THDOT_STD","SIGMA_V","SIGMA_OMG","SIGMA_THDOT"]
missing_cfg = [k for k in required if k not in cfg]
if missing_cfg:
    raise KeyError(f"Inference cfg missing keys from checkpoint: {missing_cfg}. Do NOT default these; copy them from training CFG.")

print("[CFG] V_STD/OMG_STD/THDOT_STD =", cfg["V_STD"], cfg["OMG_STD"], cfg["THDOT_STD"])
print("[CFG] SIGMA_V/OMG/THDOT      =", cfg["SIGMA_V"], cfg["SIGMA_OMG"], cfg["SIGMA_THDOT"])
print("[CFG] DELTA_T_SCALE          =", cfg.get("DELTA_T_SCALE", None))

# Best-effort: infer ESM_DIM from checkpoint (helps when config got out-of-sync)
esm_dim_ckpt = infer_esm_dim_from_state(state)
if esm_dim_ckpt is not None and int(INF_CFG.get("ESM_DIM", esm_dim_ckpt)) != int(esm_dim_ckpt):
    print(f"[WARN] INF_CFG['ESM_DIM']={int(INF_CFG['ESM_DIM'])} != checkpoint-inferred ESM dim {esm_dim_ckpt}. Using checkpoint value.")
    INF_CFG["ESM_DIM"] = int(esm_dim_ckpt)

# Instantiate model using your training notebook class
model = VelocityFlowIPADynamics(esm_dim=int(INF_CFG["ESM_DIM"]), cfg=cfg).to(device)
missing, unexpected = model.load_state_dict(state, strict=True)
if len(missing) > 0:
    print("[WARN] missing keys (showing up to 20):", missing[:20])
if len(unexpected) > 0:
    print("[WARN] unexpected keys (showing up to 20):", unexpected[:20])
model.eval()

# ---- Optional: OpenFold atom14 builder
atom14_builder = None
if bool(INF_CFG["PREFER_ATOM14"]) and HAS_OPENFOLD:
    try:
        atom14_builder = OpenFoldAtom14Builder(of_rc, rigid_utils, of_feats, device=str(device))
        print("[INFO] atom14 builder enabled.")
    except Exception as e:
        atom14_builder = None
        print("[WARN] atom14 builder init failed; CA-only fallback.", repr(e))
else:
    print("[INFO] atom14 builder disabled or OpenFold missing; CA-only output.")

# ---- Output buffer (MDTraj uses nm units)
T = int(INF_CFG["TOTAL_FRAMES"])
out_xyz = np.zeros((T, traj0.n_atoms, 3), dtype=np.float32)
out_xyz[0] = traj0.xyz[0].astype(np.float32)

W = int(cfg["WINDOW_SIZE"])
stride = int(cfg.get("WINDOW_STRIDE", 1))
dt = float(stride) if cfg.get("DT_PHYS", None) is None else float(cfg["DT_PHYS"]) * float(stride)

cur_x, cur_R, cur_tors = x_c, R_c, tors_c
out_i = 1

while out_i < T:
    start_out_i = out_i
    v_w, o_w, t_w = sample_window_velocities(
        model=model, cfg=cfg,
        aatype=aatype, esm=esm,
        x_c=cur_x, R_c=cur_R, tors_c=cur_tors,
        res_mask=res_mask,
        dt=dt, W=W,
        n_flow_steps=int(INF_CFG["FLOW_STEPS"]),
        use_amp=bool(INF_CFG["AMP"]),
        stride=stride,
    )
    x_w, R_w, tors_w = integrate_velocities_to_window(cur_x, cur_R, cur_tors, v_w, o_w, t_w, dt=dt, anchor_alpha=float(cfg.get("ANCHOR_ALPHA", 0.0)),)

    if bool(cfg.get("ORTHO_PRED_R", True)):
        B_, W_, N_ = R_w.shape[:3]
        R_w = SO3.orthonormalize_safe(
            R_w.float().reshape(-1, 3, 3)
        ).reshape(B_, W_, N_, 3, 3).to(cur_R.dtype)

    _atom14_covered = set()
    for _i in range(atom14_indices.shape[0]):
        for _j in range(14):
            _aidx = int(atom14_indices[_i, _j])
            if _aidx >= 0:
                _atom14_covered.add(_aidx)
    _n_atoms_total = traj0.n_atoms
    _uncovered_mask = np.array(
        [i not in _atom14_covered for i in range(_n_atoms_total)], dtype=bool
    )
    _has_uncovered = _uncovered_mask.any()
    if _has_uncovered:
        _n_uncovered = int(_uncovered_mask.sum())
        print(f"[INFO] {_n_uncovered} atoms not in atom14 (likely H). "
              f"These will be frozen at frame 0 in output — expected if input has no H.")

    for k in range(W):
        if out_i >= T:
            break
        xyz_k = traj0.xyz[0].copy()   # base: frame 0 for any uncovered atoms (H etc.)

        if atom14_builder is not None:
            atom14_pos, atom14_mask = atom14_builder.build_atom14(
                aatype=aatype,
                R=R_w[:, k],
                x=x_w[:, k],
                torsion_angles=tors_w[:, k],
            )
            a14_A = atom14_pos[0].detach().cpu().numpy()
            update_xyz_with_atom14(xyz_k, a14_A, atom14_indices)
        else:
            xA = x_w[0, k].detach().cpu().numpy()
            update_xyz_with_ca_only(xyz_k, xA, ca_idx)

        out_xyz[out_i] = xyz_k.astype(np.float32)
        out_i += 1

    # advance conditioning to the last frame produced this window
    produced = out_i - start_out_i
    last_k = 0 if produced <= 0 else min(W, produced) - 1
    cur_x    = x_w[:, last_k].detach()
    cur_R    = R_w[:, last_k].detach()
    cur_tors = tors_w[:, last_k].detach()

    # ── Langevin noise injection: break deterministic trajectory stalling ─────
    # Injects small Gaussian perturbations into the conditioning state between
    # windows. This is the primary fix for mode collapse (D5: intra-TM ~1.0).
    # Scale: LANGEVIN_NOISE_X in Å, LANGEVIN_NOISE_TORS in radians.
    _lx   = float(cfg.get("LANGEVIN_NOISE_X",    0.0))
    _ltors = float(cfg.get("LANGEVIN_NOISE_TORS", 0.0))
    if _lx > 0.0:
        # Translate x in local frame: noise in global space is fine here
        cur_x = cur_x + torch.randn_like(cur_x) * _lx
    if _ltors > 0.0:
        # Perturb torsions directly (already in wrapped space)
        dth = torch.randn_like(cur_tors) * _ltors
        cur_tors = wrap_to_pi(cur_tors + dth)
        # Perturb R via small random rotation (SO3 exp of small rotvec)
        rotvec = torch.randn(cur_R.shape[0], cur_R.shape[1], 3,
                             device=device, dtype=cur_R.dtype) * _ltors
        dR = so3_exp(rotvec.reshape(-1, 3)).reshape(*cur_R.shape)
        cur_R = cur_R @ dR
        if bool(cfg.get("ORTHO_PRED_R", True)):
            cur_R = SO3.orthonormalize_safe(
                cur_R.float().reshape(-1, 3, 3)
            ).reshape(*cur_R.shape[:3], 3, 3).to(x_w.dtype)
    # ─────────────────────────────────────────────────────────────────────────

# ---- Write multi-frame PDB (strict fixed-width PDB writer for ChimeraX)
write_multimodel_pdb_fixedwidth(out_xyz, top, OUT_PATH, model_start=1, assume_nm=True)
print(f"[DONE] Wrote {T} frames -> {OUT_PATH}")

