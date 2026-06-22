# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% [markdown] cell 0
# ## Patch v4 — Critical Fixes + Optimized CFG
# 
# ### Bug Fixes (from v3)
# - **P0 CRITICAL — `self.opt.step()` was missing.** Model weights now update.
# - **P1 — EMA `.update()` now called** after `opt.step()`.
# - **P1 — Peptide losses ramped** with `r_a14` during warmup.
# - **P2 — Dead `l_atom` computation removed.**
# - **P2 — `l_end_pos` returns scalar.**
# - **P3 — `grad_clip_for_epoch` renamed to `grad_clip_for_step`.**
# - **P0 — `TRAIN_ZIP` was pointing to `test.zip`!** Now points to `train.zip`.
# 
# ### Optimized Hyperparameters (v4)
# - **LR: 1e-4 → 3e-4** with **200-step linear warmup** + step-level cosine decay
# - **Weight decay: 1e-4 → 1e-2** (standard for transformers)
# - **EMA enabled** with decay 0.9999
# - **W_DISTO: 0.0 → 0.1** (long-range distance supervision was disabled!)
# - **MULTI_STRIDE: [1] → [1,2,4]** for multi-rate dynamics learning
# - **Warmup: 54 → 100 steps**, ramp: 270 → 400 steps (slower, safer)
# - **Grad clip: 80/20/10 → 50/10/5** (tighter)
# - **ORTHO_INPUT_R & ORTHO_PRED_R enabled** (SO(3) projection)
# - **torch.compile disabled** initially for debugging
# - **Debug frequency reduced** for faster training

# %% [markdown] cell 1
# ## Patch notes (OpenFold-convention dataset)
# 
# This notebook has been updated for the **new extraction** where torsions are already stored in **OpenFold/AlphaFold convention**:
# 
# - `reorder_torsions_to_openfold` is now a **no-op** (identity).
# - OpenFold atom14 supervision uses torsions **as-is** (and sanitizes non-finite angles).
# - Dataset performs a **fail-fast** check on the first NPZ (`torsion_convention` / `torsion_names`) to prevent mixing datasets.
# 
# After switching datasets, avoid resuming checkpoints trained with the old torsion semantics unless you reinitialize torsion-related heads.

# %% [markdown] cell 2
# ## Added peptide-bond stabilizers (recommended for ramped OpenFold losses)
# 
# This notebook adds two continuous stabilizers to reduce divergence when ramping OpenFold/atom14/FAPE losses:
# 
# 1. **Smooth peptide geometry loss** (`W_PEPTIDE_GEO`): continuous springs on C–N bond length, bond angles, and a small omega planarity term (cis allowed for Pro).
# 2. **Velocity-space bond constraint** (`W_PEPTIDE_VEL`): penalizes the instantaneous bond-length rate `d/dt ||C–N||` computed from predicted rigid-body velocities.
# 
# Both are gated by the `adjacent` mask from the new NPZ extraction to avoid enforcing bonds across chain breaks.

# %% cell 3
# NOTEBOOK_MAGIC: !python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
# NOTEBOOK_MAGIC: !nvcc -V
# NOTEBOOK_MAGIC: !python --version
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"

# %% cell 4
# NOTEBOOK_MAGIC: !pip -q install einops dm-tree ml-collections biopython scipy pandas tqdm modelcif geomstats pot

# %% cell 5

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

# %% cell 6
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

# %% cell 7
from __future__ import annotations

import os
import sys
import math
import glob
import time
import json
import random
import inspect
import subprocess
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
from torch._dynamo import utils as dynamo_utils
import torch._dynamo as dynamo
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import importlib
from types import SimpleNamespace

# %% cell 8
import importlib
import inspect
import types

def debug_openfold_import(of_loss):
    print("=== OpenFold import diagnostics ===")
    if of_loss is None:
        print("[FAIL] of_loss is None (not imported).")
        return False

    mod = importlib.import_module(of_loss.__name__) if isinstance(of_loss, types.ModuleType) else None

    print("[OK] of_loss object:", of_loss)
    print("[TYPE]", type(of_loss))

    # If it's a module
    if isinstance(of_loss, types.ModuleType):
        print("[MODULE NAME]", of_loss.__name__)
        print("[MODULE FILE]", getattr(of_loss, "__file__", None))
        has_fsv = hasattr(of_loss, "find_structural_violations")
        has_vl  = hasattr(of_loss, "violation_loss")
        print("[HAS find_structural_violations]", has_fsv)
        print("[HAS violation_loss]", has_vl)

        if has_fsv:
            try:
                sig = inspect.signature(of_loss.find_structural_violations)
                print("[SIG] find_structural_violations:", sig)
            except Exception as e:
                print("[WARN] cannot read signature:", e)

        if has_vl:
            try:
                sig = inspect.signature(of_loss.violation_loss)
                print("[SIG] violation_loss:", sig)
            except Exception as e:
                print("[WARN] cannot read signature:", e)

        return has_fsv and has_vl

    # If it's some other object wrapper
    print("[WARN] of_loss is not a module; checking attributes anyway.")
    has_fsv = hasattr(of_loss, "find_structural_violations")
    has_vl  = hasattr(of_loss, "violation_loss")
    print("[HAS find_structural_violations]", has_fsv)
    print("[HAS violation_loss]", has_vl)
    return has_fsv and has_vl

# Example usage (whatever your code sets as of_loss):
# ok = debug_openfold_import(of_loss)

# %% cell 9
# --------------------------------------------------------------------------------------
# Diagnostics / logging helpers (JSONL + optional TensorBoard) + invariant checks
# --------------------------------------------------------------------------------------
import os, json, math, time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _is_compiling() -> bool:
    """True when executing under torch.compile graph capture (avoid debug side-effects)."""
    try:
        import torch._dynamo
        return torch._dynamo.is_compiling()
    except Exception:
        return False


@torch.no_grad()
def _flat_sample(x: torch.Tensor, k: int = 8192) -> torch.Tensor:
    if x is None:
        return None
    if not torch.is_tensor(x):
        return None
    if x.numel() <= k:
        return x.reshape(-1)
    idx = torch.randint(0, x.numel(), (k,), device=x.device)
    return x.reshape(-1).gather(0, idx)


def _require(cond: bool, msg: str, strict: bool):
    if cond:
        return
    if strict:
        raise RuntimeError(msg)
    print(f"[DBG][WARN] {msg}")



@dataclass
class DebugCfg:
    enabled: bool = False
    strict: bool = True
    every: int = 50
    closure_every: int = 200
    roundtrip_every: int = 200
    so3_every: int = 2000
    grad_every: int = 200
    save_bad_batches: bool = True
    bad_batch_dir: str = "debug_bad_batches"
    max_saved: int = 20


class TrainDiagnostics:
    """Write training diagnostics to JSONL (and optional TensorBoard)."""
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.log_dir = str(cfg.get("LOG_DIR", "logs"))
        self.log_every = int(cfg.get("LOG_EVERY", 20))
        self.sample_k = int(cfg.get("DIAG_SAMPLE_K", 8192))
        self.use_tb = bool(cfg.get("USE_TENSORBOARD", False)) and (SummaryWriter is not None)
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.log_dir, "train_diag.jsonl")
        self._fh = open(self.jsonl_path, "a", encoding="utf-8")
        self._dense_path = os.path.join(self.log_dir, "train_dense.jsonl")
        self._fh_dense = open(self._dense_path, "a", encoding="utf-8")
        self._dense_mirror_path = "/content/train_dense.jsonl"
        self._fh_dense_mirror = open(self._dense_mirror_path, "a", encoding="utf-8")
        self._writer = SummaryWriter(self.log_dir) if self.use_tb else None
        self.warn_disp_A = float(cfg.get("WARN_DISP_A", 10.0))
        self.warn_ang_rad = float(cfg.get("WARN_ANG_RAD", math.pi * 0.95))
        self.warn_tors_rad = float(cfg.get("WARN_TORS_RAD", math.pi * 0.95))

    def maybe_log(self, step: int, payload: Dict[str, Any]):
        if (step % self.log_every) != 0:
            return
        payload = dict(payload)
        payload["step"] = int(step)
        payload["time"] = time.time()
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()
        if self._writer is not None:
            for k, v in payload.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    self._writer.add_scalar(k, float(v), step)

    def log_always(self, step: int, payload: Dict[str, Any]):
        """Write every step to train_dense.jsonl — bypasses LOG_EVERY."""
        row = dict(payload)
        row["step"] = int(step)
        row["time"] = time.time()
        line = json.dumps(row) + "\n"
        self._fh_dense.write(line)
        self._fh_dense.flush()
        self._fh_dense_mirror.write(line)
        self._fh_dense_mirror.flush()

    def close(self):
        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            for fh in (self._fh, self._fh_dense, self._fh_dense_mirror):
                try:
                    fh.close()
                except Exception:
                    pass

@torch.no_grad()
def _sample_pairs_per_batch(
    res_mask: torch.Tensor,   # (B, N) float/bool
    K: int,
    k_min: int = 4,
    k_max: int | None = 64,
    mix_band: float = 0.5,    # fraction of K from band sampling
    max_tries: int = 8,
):
    """
    Returns:
      idx_i: (B, K) long
      idx_j: (B, K) long
      pair_mask: (B, K) float in {0,1}  (1 if a valid pair was found)
      valid_pair_counts: (B,) long      (#valid sampled pairs per sample)
    """
    device = res_mask.device
    B, N = res_mask.shape
    res_mask_bool = res_mask > 0.5

    idx_i = torch.zeros((B, K), dtype=torch.long, device=device)
    idx_j = torch.zeros((B, K), dtype=torch.long, device=device)
    pair_mask = torch.zeros((B, K), dtype=torch.float32, device=device)

    K_band = int(round(K * mix_band))
    K_uni  = K - K_band

    for b in range(B):
        valid = torch.nonzero(res_mask_bool[b], as_tuple=False).squeeze(-1)  # (Nv,)
        Nv = int(valid.numel())
        if Nv < 2:
            continue  # leave zeros, mask=0

        # ---- helper: sample uniformly from valid residues
        def sample_valid(num):
            ridx = torch.randint(low=0, high=Nv, size=(num,), device=device)
            return valid[ridx]

        # ---- 1) Uniform pairs
        if K_uni > 0:
            i_u = sample_valid(K_uni)
            j_u = sample_valid(K_uni)

            # rejection to enforce constraints
            for _ in range(max_tries):
                bad = (i_u == j_u) | ((i_u - j_u).abs() < k_min)
                if not bad.any():
                    break
                j_u[bad] = sample_valid(int(bad.sum().item()))

            # If still bad (rare), just force j != i (relax k_min for leftovers)
            bad = (i_u == j_u)
            if bad.any():
                j_u[bad] = sample_valid(int(bad.sum().item()))
                # still could coincide; last resort shift by 1 in valid-index space
                bad2 = (i_u == j_u)
                if bad2.any() and Nv > 2:
                    # pick a different residue deterministically
                    tmp = sample_valid(int(bad2.sum().item()))
                    j_u[bad2] = tmp

        else:
            i_u = j_u = None

        # ---- 2) Band (near/mid-range) pairs
        if K_band > 0:
            i_b = sample_valid(K_band)
            if k_max is None:
                k_max_eff = max(k_min, 1)
            else:
                k_max_eff = max(k_max, k_min)

            # sample delta in [k_min, k_max_eff]
            delta = torch.randint(low=k_min, high=k_max_eff + 1, size=(K_band,), device=device)
            sign = torch.randint(low=0, high=2, size=(K_band,), device=device) * 2 - 1  # {-1, +1}
            j_b = i_b + sign * delta

            # rejection: in-bounds + valid + constraints
            for _ in range(max_tries):
                bad = (
                    (j_b < 0) | (j_b >= N) |
                    (~res_mask_bool[b, j_b.clamp(0, N - 1)]) |
                    (i_b == j_b) | ((i_b - j_b).abs() < k_min)
                )
                if not bad.any():
                    break

                # resample bad elements
                i_b[bad] = sample_valid(int(bad.sum().item()))
                delta_bad = torch.randint(low=k_min, high=k_max_eff + 1, size=(int(bad.sum().item()),), device=device)
                sign_bad  = torch.randint(low=0, high=2, size=(int(bad.sum().item()),), device=device) * 2 - 1
                j_b[bad]  = i_b[bad] + sign_bad * delta_bad

            # If still invalid, fall back to uniform sampling for remaining bad
            bad = (
                (j_b < 0) | (j_b >= N) |
                (~res_mask_bool[b, j_b.clamp(0, N - 1)]) |
                (i_b == j_b) | ((i_b - j_b).abs() < k_min)
            )
            if bad.any():
                j_b[bad] = sample_valid(int(bad.sum().item()))
        else:
            i_b = j_b = None

        # ---- Concatenate
        if K_uni > 0 and K_band > 0:
            i = torch.cat([i_u, i_b], dim=0)
            j = torch.cat([j_u, j_b], dim=0)
        elif K_uni > 0:
            i, j = i_u, j_u
        else:
            i, j = i_b, j_b

        # ---- Final validity check (should be almost all valid)
        valid_pair = (
            res_mask_bool[b, i] &
            res_mask_bool[b, j] &
            (i != j) &
            ((i - j).abs() >= k_min)
        )

        idx_i[b] = i
        idx_j[b] = j
        pair_mask[b] = valid_pair.float()

    valid_pair_counts = pair_mask.sum(dim=1).long()
    return idx_i, idx_j, pair_mask, valid_pair_counts


def distogram_loss_sampled_per_sample(
    x_pred: torch.Tensor,   # (B, N, 3)
    x_gt: torch.Tensor,     # (B, N, 3)
    res_mask: torch.Tensor, # (B, N)
    K: int = 1024,
    k_min: int = 4,
    k_max: int | None = 64,
    mix_band: float = 0.5,
    d_max: float | None = None,
    eps: float = 1e-6,
):
    """
    Returns:
      loss_per_sample: (B,)
      valid_pair_counts: (B,) long
    """
    idx_i, idx_j, pair_mask, valid_pair_counts = _sample_pairs_per_batch(
        res_mask=res_mask, K=K, k_min=k_min, k_max=k_max, mix_band=mix_band
    )

    # Gather coordinates: (B, K, 3)
    gather_i = idx_i.unsqueeze(-1).expand(-1, -1, 3)
    gather_j = idx_j.unsqueeze(-1).expand(-1, -1, 3)

    xi_p = x_pred.gather(dim=1, index=gather_i)
    xj_p = x_pred.gather(dim=1, index=gather_j)
    xi_g = x_gt.gather(dim=1, index=gather_i)
    xj_g = x_gt.gather(dim=1, index=gather_j)

    dp = torch.linalg.norm(xi_p - xj_p, dim=-1)  # (B, K)
    dg = torch.linalg.norm(xi_g - xj_g, dim=-1)  # (B, K)

    if d_max is not None:
        dp = dp.clamp(max=d_max)
        dg = dg.clamp(max=d_max)

    err = (dp - dg) ** 2  # (B, K)

    # masked mean over sampled pairs (only relevant when masks are sparse)
    denom = pair_mask.sum(dim=1).clamp_min(1.0)  # (B,)
    loss_per_sample = (err * pair_mask).sum(dim=1) / (denom + eps)  # (B,)

    return loss_per_sample, valid_pair_counts

# %% cell 10
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
            print(debug_openfold_import(of_loss))
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

# %% cell 11
import torch

def move_all_tensors_in_module(mod, device):
    moved = []
    for name, val in vars(mod).items():
        if torch.is_tensor(val) and val.device != device:
            setattr(mod, name, val.to(device))
            moved.append(name)
    return moved

device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

print("FF.all_atom:", FF.all_atom)
print("DEFAULT_FRAMES before:", FF.all_atom.DEFAULT_FRAMES.device)
print("GROUP_IDX before:", getattr(FF.all_atom, "GROUP_IDX").device)

moved = move_all_tensors_in_module(FF.all_atom, device)

print("Moved tensors:", moved)
print("DEFAULT_FRAMES after:", FF.all_atom.DEFAULT_FRAMES.device)
print("GROUP_IDX after:", getattr(FF.all_atom, "GROUP_IDX").device)

assert FF.all_atom.DEFAULT_FRAMES.device.type == "cuda"
assert getattr(FF.all_atom, "GROUP_IDX").device.type == "cuda"

# %% cell 12
import importlib, inspect, sys

def openfold_healthcheck():
    print("=== OpenFold healthcheck ===")
    print("FoldFlow in sys.path?", any("FoldFlow" in p for p in sys.path))
    try:
        import openfold
        print("openfold.__path__ =", list(openfold.__path__))
    except Exception as e:
        print("[FAIL] import openfold:", e)
        return False

    # residue_constants
    try:
        rc_mod = importlib.import_module("openfold.np.residue_constants")
        print("[OK] residue_constants file:", getattr(rc_mod, "__file__", None))
    except Exception as e:
        print("[FAIL] openfold.np.residue_constants:", e)
        return False

    # loss module
    try:
        loss_mod = importlib.import_module("openfold.utils.loss")
        print("[OK] openfold.utils.loss file:", getattr(loss_mod, "__file__", None))
        print("has find_structural_violations:", hasattr(loss_mod, "find_structural_violations"))
        print("has violation_loss:", hasattr(loss_mod, "violation_loss"))
        if hasattr(loss_mod, "find_structural_violations"):
            print("sig find_structural_violations:", inspect.signature(loss_mod.find_structural_violations))
        if hasattr(loss_mod, "violation_loss"):
            print("sig violation_loss:", inspect.signature(loss_mod.violation_loss))
        return True
    except Exception as e:
        print("[WARN] openfold.utils.loss not importable:", e)
        return False

ok = openfold_healthcheck()
print("OpenFold loss available:", ok)

# %% cell 13
from google.colab import drive  # type: ignore
drive.mount("/content/drive")

DRIVE_BASE = "/content/drive/MyDrive/af_native_dynamics_predictor"
ACTIVE_SMOOTHING_WINDOW = 5   # 7 | 5 | 3 | "raw"

# >>> EDIT: point to your Option-A zips
TRAIN_ZIP = os.path.join(DRIVE_BASE, f"data/processed/MD_Simulation/V7_atom14_openfold_npz/train_smooth_{ACTIVE_SMOOTHING_WINDOW}.zip")
VAL_ZIP   = os.path.join(DRIVE_BASE, f"data/processed/MD_Simulation/V7_atom14_openfold_npz/val_smooth_{ACTIVE_SMOOTHING_WINDOW}.zip")
LOCAL_ZIP = "/content/zip"
WORKDIR = "/content/Data"
TRAIN_DIR = os.path.join(WORKDIR, "train_npz")
VAL_DIR   = os.path.join(WORKDIR, "val_npz")
TEST_DIR  = os.path.join(WORKDIR, "test_npz")
os.makedirs(WORKDIR, exist_ok=True)
os.makedirs(LOCAL_ZIP, exist_ok=True)

def unzip_if_needed(zip_path, out_dir):
    marker = os.path.join(out_dir, ".unzipped_done")
    if os.path.exists(marker):
        return

    print(f"[UNZIP] Extracting {zip_path} -> {out_dir}")

    with zipfile.ZipFile(zip_path, "r") as z:
        # Get the list of files to extract
        members = z.infolist()

        # specific TQDM settings for bytes
        with tqdm(total=sum(m.file_size for m in members), unit='B', unit_scale=True, desc="Extracting") as pbar:
            for member in members:
                z.extract(member, out_dir)
                pbar.update(member.file_size)

    with open(marker, "w") as f:
        f.write("ok")

def copy_with_progress(src, dst):
    size = os.path.getsize(src)
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        with tqdm(total=size, unit='B', unit_scale=True, desc=f"Copying {os.path.basename(src)}") as pbar:
            while True:
                chunk = fsrc.read(1024 * 1024)
                if not chunk: break
                fdst.write(chunk)
                pbar.update(len(chunk))


copy_with_progress(TRAIN_ZIP, LOCAL_ZIP + "/train.zip")
copy_with_progress(VAL_ZIP, LOCAL_ZIP + "/val.zip")

unzip_if_needed(LOCAL_ZIP + "/train.zip", TRAIN_DIR)
unzip_if_needed(LOCAL_ZIP + "/val.zip", VAL_DIR)

# %% cell 14
CURRENT_NOTEBOOK_NAME = "velocity_fm_v5_6_anchor_v2.ipynb"

# 2. Define Base Paths
BASE_FOLDER = "/content/drive/MyDrive/af_native_dynamics_predictor"

# The root folder where all training versions are stored
TRAINING_ROOT = f"{BASE_FOLDER}/models/Model_T4_V6"

# 3. Find the last training folder
# Check if the root model directory exists, create if not
if not os.path.exists(TRAINING_ROOT):
    os.makedirs(TRAINING_ROOT)
    print(f"Created root model directory: {TRAINING_ROOT}")

# List existing "Traning_X" folders
existing_folders = glob.glob(os.path.join(TRAINING_ROOT, "Traning_*"))
indices = []

for folder in existing_folders:
    # Extract the folder name (e.g., "Traning_4")
    folder_name = os.path.basename(folder)
    try:
        # Split by '_' and get the last part as an integer
        idx = int(folder_name.split('_')[-1])
        indices.append(idx)
    except ValueError:
        pass # Ignore folders that don't match the format

# Determine the next index
if indices:
    next_index = max(indices) + 1
else:
    next_index = 1 # Start at 1 if no folders exist

# 4. Create the new paths
NEW_TRAINING_FOLDER = os.path.join(TRAINING_ROOT, f"Traning_{next_index}")
CHECK_POINTS = os.path.join(NEW_TRAINING_FOLDER, "checkpoints")

# Create directories
os.makedirs(CHECK_POINTS, exist_ok=True)

# 5. Save a copy of the current Code
# We assume your notebooks are in the default "Colab Notebooks" folder
# If they are elsewhere, update 'source_path' below.
source_path = f"{BASE_FOLDER}/notebooks/modeling/T4_V7/velocity_fm_T4_V7_anchor_v2.ipynb"
destination_path = os.path.join(NEW_TRAINING_FOLDER, f"{CURRENT_NOTEBOOK_NAME}_v{next_index}_copy.ipynb")

if os.path.exists(source_path):
    shutil.copy2(source_path, destination_path)
    print(f" Backup of code saved to: {destination_path}")
else:
    print(f" WARNING: Could not find notebook at {source_path}")
    print("   -> Please check 'CURRENT_NOTEBOOK_NAME' or manually save a copy.")

# %% cell 15
# ──────────────────────────────────────────────────────────────────────────────
# 2) Configuration
# ──────────────────────────────────────────────────────────────────────────────

WINDOW_CFGS: Dict[Any, Dict[str, Any]] = {

    7: dict(
        # Measured from R1+R2+R3 NPZ, SG window=7
        V_STD    = 0.28471,
        OMG_STD  = 0.07205,
        THDOT_STD= 0.21562,

        V_STD_BY_STRIDE     = {1: 0.28471, 2: 0.25033, 4: 0.19308},
        OMG_STD_BY_STRIDE   = {1: 0.07205, 2: 0.06106, 4: 0.04548},
        THDOT_STD_BY_STRIDE = {1: 0.21562, 2: 0.17218, 4: 0.12124},

        # Noise stds — derived from velocity distribution at this smoothing level
        SIGMA_V    = 0.3701,
        SIGMA_OMG  = 0.0937,
        SIGMA_THDOT= 0.2803,

        # Huber delta at stride=4: displacement is larger at coarser stride
        # Rule of thumb: delta[4] ≈ delta[1] × (V_STD[4]/V_STD[1]) × dt × 3
        A14_HUBER_DELTA_BY_STRIDE = {1: 1.4, 2: 2.0, 4: 4.0},
    ),

    5: dict(
        V_STD     = 0.36148,
        OMG_STD   = 0.09518,
        THDOT_STD = 0.27686,

        V_STD_BY_STRIDE     = {1: 0.36148, 2: 0.29533, 4: 0.1932},
        OMG_STD_BY_STRIDE   = {1: 0.09518, 2: 0.0751, 4: 0.04722},
        THDOT_STD_BY_STRIDE = {1: 0.27686, 2: 0.20992, 4: 0.12483},

        SIGMA_V      = 0.4699,   # 1.3x stride-1 V_STD
        SIGMA_OMG    = 0.1237,
        SIGMA_THDOT  = 0.3599,

        A14_HUBER_DELTA_BY_STRIDE = {1: 1.5, 2: 2.2, 4: 4.5},  # ← tune after first run
    ),

    "raw": dict(
        V_STD    = 0.33500,          # ← MEASURE
        OMG_STD  = 0.08400,          # ← MEASURE
        THDOT_STD= 0.25500,          # ← MEASURE

        V_STD_BY_STRIDE     = {1: 0.33500, 2: 0.29500, 4: 0.23000},  # ← MEASURE
        OMG_STD_BY_STRIDE   = {1: 0.08400, 2: 0.07100, 4: 0.05300},  # ← MEASURE
        THDOT_STD_BY_STRIDE = {1: 0.25500, 2: 0.20300, 4: 0.14300},  # ← MEASURE

        SIGMA_V    = 0.4400,         # ← MEASURE
        SIGMA_OMG  = 0.1100,         # ← MEASURE
        SIGMA_THDOT= 0.3300,         # ← MEASURE

        # Raw trajectories have much larger displacement variance at stride=4
        A14_HUBER_DELTA_BY_STRIDE = {1: 1.8, 2: 2.8, 4: 6.0},  # ← tune after first run
    ),
}

# ── Common configuration (unchanged across smoothing levels) ──────────────────
COMMON_CFG: Dict[str, Any] = dict(

    # ── Infrastructure ────────────────────────────────────────────────────────
    DEVICE="cuda" if torch.cuda.is_available() else "cpu",
    SEED=42,
    VAL_EMA_ALPHA=0.3,

    # ── Data loading ──────────────────────────────────────────────────────────
    PRELOAD_TO_RAM=True,
    MAX_PRELOAD_GB=0.0,
    NUM_WORKERS=2,
    PREFETCH_FACTOR=6,

    # ── Data shaping ──────────────────────────────────────────────────────────
    MAX_RES=160,
    TRAJ_LEN=1000,
    COORD_SCALE=1.0,

    # ── Windowing ─────────────────────────────────────────────────────────────
    WINDOW_SIZE=16,
    WINDOW_STRIDE=1,
    MULTI_STRIDE=[1, 2, 4],

    # ── Stride-aware training ─────────────────────────────────────────────────
    STRIDE_EMBED_DIM=None,
    STRIDE_FM_WEIGHTS={1: 0.3, 2: 0.6, 4: 0.75},
    LPOS_HUBER_DELTA_BY_STRIDE={1: 0.8, 2: 1.2, 4: 1.7},
    USE_STRIDE_HEADS=True,

    # ── t0 sampling ───────────────────────────────────────────────────────────
    RANDOM_T0_TRAIN=True,
    STRATIFIED_T0_TRAIN=True,
    T0_STRAT_BINS=12,
    VAL_DETERMINISTIC_T0_COVERAGE=True,
    VAL_T0_FRACS=[0.0, 0.25, 0.50, 0.75, 1.0],
    VAL_STRIDE=None,
    DT_PHYS=None,

    # ── Task mixing ───────────────────────────────────────────────────────────
    USE_TASK_INTERP=True,
    TASK_INTERP_PROB=0.25,
    DISABLE_KABSCH_INTERP=True,
    USE_ANCHOR_EMB=True,

    # ── Embeddings / conditioning ─────────────────────────────────────────────
    DELTA_T_SCALE=4.0,
    RELPOS_MAX=155.0,

    # ── Model architecture ────────────────────────────────────────────────────
    C_S=384,
    C_Z=128,
    IPA_BLOCKS=6,
    HEADS=8,
    IPA_C_HIDDEN=16,
    NO_QK_POINTS=4,
    NO_V_POINTS=8,
    STEP_CHUNK=1,
    DETACH_PAIR_X=True,
    GRAD_CHECKPOINT_IPA=False,
    CAST_Z_FP32_AFTER_SLICE=True,
    USE_TEMPORAL_ATTENTION=True,
    TEMP_ATTN_LAYERS=2,
    TEMP_ATTN_HEADS=8,
    TEMP_ATTN_DROPOUT=0.0,

    # ── Velocity normalisation ────────────────────────────────────────────────
    # NOTE: V_STD / OMG_STD / THDOT_STD and the BY_STRIDE dicts are injected
    # from WINDOW_CFGS[ACTIVE_SMOOTHING_WINDOW] below — do NOT set them here.
    NORMALIZE_VELOCITIES=True,

    # ── Flow noise ────────────────────────────────────────────────────────────
    # NOTE: SIGMA_V / SIGMA_OMG / SIGMA_THDOT come from WINDOW_CFGS.
    NOISE_AR1_RHO=0.7,
    SIGMA_STEP_ALPHA=0.5,
    FLOW_S_PER_STEP=True,
    FLOW_S_RHO=0.7,
    FLOW_S_NOISE=0.15,

    # ── Optimiser ─────────────────────────────────────────────────────────────
    BATCH_SIZE=8,
    LR=3e-4,
    LR_MIN=3e-5,
    WD=0.01,
    EPOCHS=100,
    LR_WARMUP_STEPS=100,
    GRAD_CLIP=1.0,
    AMP_ENABLED=True,
    AMP_DTYPE='bf16',
    COMPILE=True,
    EMA=True,
    EMA_DECAY=0.999,
    IPA_LR_MULT=1.0,
    HEAD_LR_MULT=2.0,
    OTHER_LR_MULT=0.1,
    W_AUX=0.1,
    AUX_EVERY=2,
    HEAD_GRAD_CLIP=40.0,
    IPA_GRAD_CLIP=50.0,
    END_POS_CLAMP=5.0,
    GRAD_ACCUM_STEPS=4,
    PEPGEO_LOSS_CLAMP=20.0,
    FAPE_ONLY_LAST=True,

    # ── Loss schedule ─────────────────────────────────────────────────────────
    SCHED_ENABLED=True,
    SCHED_WARMUP_STEPS=150,
    SCHED_A14_RAMP_STEPS=600,
    SCHED_OF_RAMP_STEPS=1200,
    SCHED_FAPE_RAMP_STEPS=600,
    SCHED_PEPGEO_RAMP_STEPS=1200,
    SCHED_DISTO_RAMP_STEPS=600,
    SCHED_DISTO_STRIDE_EARLY=8,
    SCHED_DISTO_STRIDE_LATE=4,
    SCHED_DISTO_STRIDE_SWITCH=800,
    SCHED_DISTO_SAMPLE_K_EARLY=1024,
    SCHED_DISTO_SAMPLE_K_LATE=2048,
    SCHED_DISTO_K_SWITCH=800,
    SCHED_GEOM_STRIDE_MID=4,
    SCHED_GEOM_STRIDE_LATE=2,
    SCHED_GEOM_MID_START=0,
    SCHED_GEOM_LATE_START=3000,
    SCHED_INTERP_START=1200,
    SCHED_INTERP_RAMP_STEPS=600,
    GRAD_CLIP_SCHEDULE=[
        {"until_step": 100,  "max_norm": 500.0},
        {"until_step": 500,  "max_norm": 50.0},
        {"until_step": 9999, "max_norm": 15.0},
    ],
    SCHED_POSE_DECAY_START=300,
    SCHED_POSE_DECAY_STEPS=600,
    SCHED_POSE_FLOOR=0.02,

    # ── Huber deltas ──────────────────────────────────────────────────────────
    # NOTE: A14_HUBER_DELTA_BY_STRIDE comes from WINDOW_CFGS (scales with V_STD).
    LPOS_HUBER_DELTA=1.7,
    A14_HUBER_DELTA=1.4,

    # ── Checkpoints ───────────────────────────────────────────────────────────
    CKPT_DIR=CHECK_POINTS,

    # ── Loss weights ──────────────────────────────────────────────────────────
    W_FM=1.0,
    W_DISTO=0.1,
    W_ATOM14_SUP=0.203,
    W_RMSF_MATCH=0.003,
    W_BACKBONE_SUP=0.0,
    SCHED_BACKBONE_RAMP_STEPS=300,
    ANCHOR_ALPHA=0.05,
    ANCHOR_DECAY_STEPS=200,
    W_VEL_ENTROPY=0.02,
    VEL_ENTROPY_MIN_VAR=0.05,
    W_OF_VIOL=0.07,
    OF_REQUIRED=True,
    OF_STRIDE=1,
    W_PEPTIDE_GEO=0.1,
    W_PEPTIDE_VEL=0.15,
    PEP_GEO_W_LEN=2.5,
    PEP_GEO_W_ANG=0.2,
    PEP_GEO_W_OMEGA=0.05,
    PEP_ALLOW_CIS_PRO=True,
    PEP_CN_LEN_A=1.329,
    PEP_ANG_CA_C_N_DEG=116.2,
    PEP_ANG_C_N_CA_DEG=121.7,
    PEP_GEO_EPS=1e-6,
    PEP_VEL_EPS=1e-6,
    W_TEMP_DX=0.015,
    W_TEMP_DR=0.008,
    W_TEMP_DTH=0.008,
    W_END_POS=0.119,
    W_END_ROT=0.024,
    W_END_TOR=0.098,
    W_END_A14=0.099,
    W_END_OF=0.049,
    W_POS=0.07,
    W_ROT=0.0004,
    W_TOR=0.004,
    FM_TORSION_WEIGHT=1.0,
    POSE_USE_ALIGNED=True,
    EXCLUDE_LAST_FROM_WINDOW_LOSSES_IN_INTERP=False,
    OF_VIOL_TOL=12.0,
    OF_CLASH_TOL=1.5,
    W_FAPE=0.05,
    FAPE_LENGTH_SCALE=10.0,
    FAPE_CLAMP=10.0,
    FAPE_EPS=1e-4,

    # ── Sparse losses ─────────────────────────────────────────────────────────
    PROFILE_ONE_BATCH=False,
    USE_SHARED_PAIR_Z=True,
    PAIR_Z_SOURCE="cond",
    DISTO_STRIDE=2,
    DISTO_ONLY_LAST=False,
    GEOM_STRIDE=4,
    GEOM_ONLY_LAST=False,
    OF_VIOL_ONLY_LAST=False,
    DISTO_USE_SAMPLED=True,
    DISTO_SAMPLE_K=1024,
    DISTO_K_MIN_SEP=4,
    DISTO_K_MAX_SEP=64,
    DISTO_MIX_BAND=0.7,
    DISTO_D_MAX=80.0,
    DISTO_AVG_ONLY_ACTIVE_STEPS=True,
    USE_IGSO3=False,
    IGSO3_STRICT=True,
    IGSO3_USE_AR1=True,
    IGSO3_EPS_FACTOR=1.0,
    ORTHO_INPUT_R=True,
    ORTHO_PRED_R=True,
    ORTHO_R_FP32=True,
    ORTHO_DEBUG=False,

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_DIR=os.path.join(CHECK_POINTS, "logs"),
    LOG_EVERY=5,
    USE_TENSORBOARD=True,
    DIAG_SAMPLE_K=8192,
    DEBUG=False,
    DEBUG_PRINT_FIRST=50,
    DEBUG_PRINT_EVERY=50,
    DEBUG_JSONL=True,
    DEBUG_JSONL_PATH=None,
    DEBUG_TOPK_GRADS=10,
    DEBUG_PRINT_MEM=True,
    DEBUG_ANOMALY_STEPS=1,
    DEBUG_ASSERT_FINITE=True,
    WARN_DISP_A=2.59,
    WARN_ANG_RAD=0.6872,
    WARN_TORS_RAD=float(math.pi * 0.95),
    LOG_FAPE_CLAMP_STATS=True,
    DEBUG_STRICT=True,
    DEBUG_EVERY=50,
    DEBUG_CLOSURE_EVERY=100,
    DEBUG_ROUNDTRIP_EVERY=100,
    DEBUG_SO3_EVERY=500,
    DEBUG_GRAD_EVERY=100,
    DEBUG_SAVE_BAD_BATCHES=True,
    DEBUG_BAD_BATCH_DIR=os.path.join(CHECK_POINTS, "bad_batches"),
    DEBUG_MAX_SAVED=20,
    DEBUG_ISOLATE_LOSS_TERM=False,
    DETECT_ANOMALY=False,
    CHECKS_ENABLED=False,
    NAN_TRACK=False,

    # ── Resume ────────────────────────────────────────────────────────────────
    RESUME_FROM="/content/drive/MyDrive/af_native_dynamics_predictor/models/Model_T4_V6/Traning_195/checkpoints/last.pt",
    RESUME_STRICT=False,
)

# ── Merge: window-specific values override common values ──────────────────────
assert ACTIVE_SMOOTHING_WINDOW in WINDOW_CFGS, (
    f"ACTIVE_SMOOTHING_WINDOW={ACTIVE_SMOOTHING_WINDOW!r} not in WINDOW_CFGS. "
    f"Valid options: {list(WINDOW_CFGS.keys())}"
)
CFG: Dict[str, Any] = {**COMMON_CFG, **WINDOW_CFGS[ACTIVE_SMOOTHING_WINDOW]}

# ── Sanity print so you always know what's active ────────────────────────────
_w = WINDOW_CFGS[ACTIVE_SMOOTHING_WINDOW]
print(f"[CFG] Smoothing window : {ACTIVE_SMOOTHING_WINDOW}")
print(f"[CFG] V_STD            : {_w['V_STD']}  (stride=1 baseline)")
print(f"[CFG] V_STD_BY_STRIDE  : {_w['V_STD_BY_STRIDE']}")
print(f"[CFG] RESUME_FROM      : {COMMON_CFG['RESUME_FROM']}")

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

os.makedirs(CFG["CKPT_DIR"], exist_ok=True)
seed_all(int(CFG["SEED"]))

# %% [markdown] cell 16
# ## Ramp + sparsity schedule (memory/time efficient)
# 
# This notebook uses a **soft curriculum**:
# 
# - Start with **FM + pose** (pos/rot/tor) to stabilize dynamics.
# - Gradually ramp in **distogram**, then **atom14/FAPE**, then **physics/violations**.
# - Compute heavy losses **sparsely** early (last step only / large stride), then increase frequency later.
# 
# All ramps are defined in **global steps** via `CFG["SCHED_*"]` keys.

# %% cell 17
import os, time

RUN_TAG = time.strftime("%Y%m%d_%H%M%S")
# LOG_DIR = os.path.join(NEW_TRAINING_FOLDER, "train_logs", RUN_TAG)  # Colab-safe
LOG_DIR = os.path.join(NEW_TRAINING_FOLDER, "train_logs")
os.makedirs(LOG_DIR, exist_ok=True)

CFG.update({
    # logging
    "LOG_DIR": LOG_DIR,           # per N batches (console + file)
    "JSONL_METRICS": True,      # write metrics.jsonl
    "SAVE_BAD_BATCH": True,     # dump batch if NaN/Inf
    "SAVE_BAD_BATCH_MAX_MB": 300,  # avoid accidental huge dumps

    # sanity checks (low overhead if limited to first steps)
    "SANITY_STEPS": 3,          # run expensive checks only for first K steps of each epoch
    "ASSERT_SHAPES": True,
    "ASSERT_FINITE": True,
    "GRAD_FINITE_CHECK": True,

    # timing/memory instrumentation
    "SYNC_TIMING": False,       # True = accurate GPU timings but slower (syncs)
    "RESET_CUDA_PEAK_EVERY_STEP": True,

    # anomaly detection (very slow; keep 0 unless chasing NaN source)
    "ANOMALY_STEPS": 0,         # set 1 to enable detect_anomaly on first step only
})
print("Logging to:", CFG["LOG_DIR"])

# %% cell 18
import torch

@torch.no_grad()
def openfold_violation_sanity(of_loss, pred_a14, atom14_exists, aatype, residx_atom14_to_atom37, violation_tolerance_factor=12.0, clash_overlap_tolerance=1.5):
    """
    pred_a14: (B,N,14,3)
    atom14_exists: (B,N,14) in {0,1}
    aatype: (B,N)
    """
    assert of_loss is not None
    assert pred_a14.ndim == 4 and pred_a14.shape[-2] == 14 and pred_a14.shape[-1] == 3
    assert atom14_exists.shape == pred_a14.shape[:-1]
    B, N = pred_a14.shape[0], pred_a14.shape[1]

    residue_index = torch.arange(N, device=pred_a14.device).view(1, N).expand(B, N)

    batch_v = dict(
        atom14_atom_exists=atom14_exists,
        residue_index=residue_index,
        aatype=aatype,
        residx_atom14_to_atom37=residx_atom14_to_atom37,
    )

    viol_good = call_with_sig(
        of_loss.find_structural_violations,
        dict(
            batch=batch_v,
            atom14_pred_positions=pred_a14,
            violation_tolerance_factor=float(violation_tolerance_factor),
            clash_overlap_tolerance=float(clash_overlap_tolerance),
        ),
    )
    loss_good = call_with_sig(
        of_loss.violation_loss,
        dict(violations=viol_good, atom14_atom_exists=atom14_exists),
    )
    loss_good = torch.mean(loss_good) if torch.is_tensor(loss_good) else torch.tensor(loss_good, device=pred_a14.device)

    # Construct a "bad" structure by displacing atoms by ~5Å
    noise = 5.0 * torch.randn_like(pred_a14)
    pred_bad = pred_a14 + noise

    viol_bad = call_with_sig(
        of_loss.find_structural_violations,
        dict(
            batch=batch_v,
            atom14_pred_positions=pred_bad,
            violation_tolerance_factor=float(violation_tolerance_factor),
            clash_overlap_tolerance=float(clash_overlap_tolerance),
        ),
    )
    loss_bad = call_with_sig(
        of_loss.violation_loss,
        dict(violations=viol_bad, atom14_atom_exists=atom14_exists),
    )
    loss_bad = torch.mean(loss_bad) if torch.is_tensor(loss_bad) else torch.tensor(loss_bad, device=pred_a14.device)

    print("=== OpenFold violation sanity ===")
    print("loss_good:", float(loss_good))
    print("loss_bad :", float(loss_bad))
    print("finite?  :", torch.isfinite(loss_good).item(), torch.isfinite(loss_bad).item())
    print("sensitive (bad>good)?", (loss_bad > loss_good).item())

    return loss_good, loss_bad

# Usage example (inside a debug step):
# out = a14_loss(...); pred_a14 = out["pred_a14"]; atom14_exists = out["pred_a14_mask"]; aatype = batch["aatype"]
# openfold_violation_sanity(of_loss, pred_a14, atom14_exists, aatype)

# %% cell 19
import pkgutil, sys

def find_openfold_like():
    hits = []
    for m in pkgutil.iter_modules():
        if "openfold" in m.name.lower() or "alphafold" in m.name.lower():
            hits.append(m.name)
    print("modules containing 'openfold'/'alphafold':")
    for h in sorted(hits)[:50]:
        print(" -", h)

find_openfold_like()

# %% cell 20
def sanitize_aatype(aatype: torch.Tensor, res_mask: torch.Tensor, unk: int = 20):
    if aatype.dtype != torch.long:
        aa = aatype.long()
    else:
        aa = aatype

    valid = res_mask > 0.5  # (B,N) bool

    # padding -> UNK (tensor-only)
    aa = torch.where(valid, aa, aa.new_full(aa.shape, int(unk)))

    # eager-only range check (avoid graph breaks / exceptions inside compile)
    try:
        compiling = torch._dynamo.is_compiling()
    except Exception:
        compiling = False

    if not compiling:
        bad = valid & ((aa < 0) | (aa > int(unk)))
        if bad.any():
            bad_vals = aa[bad][:20].tolist()
            raise ValueError(
                f"[AATYPE] out-of-range values on valid residues: {bad_vals} (showing up to 20)"
            )

    # unk fraction among valid residues (tensor scalar; no .item, no empty-index mean)
    num_valid = valid.sum()
    num_unk   = ((aa == int(unk)) & valid).sum()
    unk_frac  = num_unk.float() / num_valid.clamp(min=1).float()

    return aa, unk_frac

# %% cell 21
@torch.no_grad()
def debug_check_atom14_masks(aa, res_mask, pred_mask, gt_a14m, step_tag=""):
    """
    aa:        (B,N)
    res_mask:  (B,N)
    pred_mask: (B,N,14) from builder existence table
    gt_a14m:   (B,N,14) from dataset
    """
    valid = (res_mask > 0.5).unsqueeze(-1)  # (B,N,1)

    pred_sup = pred_mask * valid
    gt_sup   = gt_a14m   * valid

    den_pred = pred_sup.sum().item()
    den_gt   = gt_sup.sum().item()

    # mismatch only on valid residues
    mismatch = ((pred_mask > 0.5) != (gt_a14m > 0.5)) & valid.bool()
    mismatch_frac = mismatch.float().sum().item() / max(gt_sup.numel(), 1)

    # also catch the “everything became UNK” symptom
    aa_valid = aa[res_mask > 0.5]
    aa_min = int(aa_valid.min().item()) if aa_valid.numel() else -999
    aa_max = int(aa_valid.max().item()) if aa_valid.numel() else -999
    unk_frac = (aa_valid == 20).float().mean().item() if aa_valid.numel() else 0.0

    print(
        f"[MASKCHK]{step_tag} aa[min,max]=[{aa_min},{aa_max}] unk_frac={unk_frac:.3f} "
        f"den_pred={den_pred:.1f} den_gt={den_gt:.1f} mismatch_frac={mismatch_frac:.6f}"
    )

    # fail-fast thresholds (tune as needed)
    if den_gt < 100:  # usually should be large for any real protein batch
        print("[MASKCHK] WARNING: very small GT denominator -> supervision almost empty")
    if mismatch_frac > 1e-4:
        print("[MASKCHK] WARNING: pred_mask != gt_a14m on valid residues (aatype/indexing mismatch likely)")

# %% cell 22
def build_restype_atom14_to_atom37_from_names(rc, device):
    import torch

    # atom37 name -> index
    atom_order = getattr(rc, "atom_order", None)
    if atom_order is None:
        atom_types = getattr(rc, "atom_types", None)
        if atom_types is None:
            raise AttributeError("rc must provide atom_order or atom_types to build atom14->atom37 mapping")
        atom_order = {name: i for i, name in enumerate(atom_types)}

    # restype letters in canonical index order (0..19), then X/UNK at 20
    restypes = getattr(rc, "restypes", None)
    if restypes is None:
        ro = getattr(rc, "restype_order", None)
        if ro is None:
            raise AttributeError("rc must provide restypes or restype_order")
        restypes = [aa for aa, _ in sorted(ro.items(), key=lambda kv: kv[1])]

    restype_1to3 = getattr(rc, "restype_1to3", None)
    if restype_1to3 is None:
        raise AttributeError("rc must provide restype_1to3 (1-letter to 3-letter)")

    name_to_atom14 = getattr(rc, "restype_name_to_atom14_names", None)
    if name_to_atom14 is None:
        raise AttributeError("rc must provide restype_name_to_atom14_names")

    # Build (21,14) mapping
    out = torch.full((21, 14), -1, dtype=torch.long)

    # 0..19
    for i, aa1 in enumerate(restypes[:20]):
        aa3 = restype_1to3[aa1]
        a14_names = name_to_atom14.get(aa3, None)
        if a14_names is None:
            raise KeyError(f"Missing atom14 name list for residue {aa3}")
        for a14i, aname in enumerate(a14_names):
            if aname and (aname in atom_order):
                out[i, a14i] = int(atom_order[aname])

    # 20: unknown
    unk3 = restype_1to3.get("X", "UNK")
    a14_names = name_to_atom14.get(unk3, [""] * 14)
    for a14i, aname in enumerate(a14_names):
        if aname and (aname in atom_order):
            out[20, a14i] = int(atom_order[aname])

    return out.to(device)

# %% cell 23
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

    @staticmethod
    def orthonormalize_safe_ste(R: torch.Tensor) -> torch.Tensor:
        """
        Straight-through SO(3) projection:
          - forward uses SVD projection
          - backward is identity (avoids SVD gradient singularities)
        """
        with torch.no_grad():
            R_proj = SO3.orthonormalize_safe(R)
        return R + (R_proj - R).detach()

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

        return vf64.to(torch.float32).reshape(*lead, 3)

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

# %% cell 24
def _pad_to(x: np.ndarray, target_len: int, pad_value: float = 0.0, axis: int = 0) -> np.ndarray:
    if x.shape[axis] >= target_len:
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(0, target_len)
        return x[tuple(sl)]
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, target_len - x.shape[axis])
    return np.pad(x, pad_width, mode="constant", constant_values=pad_value)


def _assert_npz_openfold_convention(npz_path: str):
    """Fail-fast guard: ensure we are training on OpenFold-convention torsions."""
    import numpy as _np
    with _np.load(npz_path, allow_pickle=True) as z:
        conv = z.get("torsion_convention", None)
        names = z.get("torsion_names", None)

    if conv is not None:
        conv_s = str(conv)
        if conv_s != "openfold_v2_psi_N_CA_C_Nnext":
            raise RuntimeError(
                f"[TORSION CONVENTION] expected openfold_v2_psi_N_CA_C_Nnext , got {conv_s} in {npz_path}"
            )

    if names is not None:
        names_l = [str(x) for x in names.tolist()]
        exp = ["pre_omega","phi","psi","chi1","chi2","chi3","chi4"]
        if names_l != exp:
            raise RuntimeError(f"[TORSION NAMES] expected {exp}, got {names_l} in {npz_path}")

    with np.load(npz_path, allow_pickle=True) as _z2:
        _enc = str(_z2.get("aatype_encoding", b""))
    if _enc and "ARNDCQEGHILKMFPSTWYV" not in _enc and "openfold_restypes" not in _enc:
        import warnings
        warnings.warn(
            f"[AATYPE ENCODING] {npz_path}: aatype_encoding='{_enc}'. "
            "Likely uses old alphabetical AA_ORDER — rc table lookups will be wrong. "
            "Re-extract or run the remap utility in the extraction notebook.",
            stacklevel=2,
        )


class WindowVelocityFlowDataset(Dataset):
    def __init__(self, data_dir: str, cfg: Dict[str, Any], train: bool = True):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if len(all_files) == 0:
            raise RuntimeError(f"No NPZ files found in {data_dir}")
        self.cfg = cfg
        self.train = train
        self.max_res = int(cfg["MAX_RES"])
        self.W = int(cfg["WINDOW_SIZE"])
        self.multi_stride = list(cfg.get("MULTI_STRIDE", [int(cfg["WINDOW_STRIDE"])]))

        # Fail-fast: verify torsion convention on first file
        if bool(cfg.get('ASSERT_TORSION_CONVENTION', True)) and len(all_files) > 0:
            _assert_npz_openfold_convention(all_files[0])

        # t0 sampling modes
        self.strat_train = bool(cfg.get("STRATIFIED_T0_TRAIN", False)) and self.train
        self.strat_bins  = max(1, int(cfg.get("T0_STRAT_BINS", 5)))
        self.train_repeat = self.strat_bins if self.strat_train else 1

        self.val_cover = (not self.train) and bool(cfg.get("VAL_DETERMINISTIC_T0_COVERAGE", True))
        self.val_t0_fracs = list(cfg.get("VAL_T0_FRACS", [0.0])) if self.val_cover else [0.0]
        if len(self.val_t0_fracs) == 0:
            self.val_t0_fracs = [0.0]
        self.val_repeat = len(self.val_t0_fracs)

        # RAM preload controls
        self.preload = bool(cfg.get("PRELOAD_TO_RAM", False))
        self.max_preload_gb = float(cfg.get("MAX_PRELOAD_GB", 0.0))
        self._ram: Dict[str, Dict[str, np.ndarray]] = {}

        keep_keys = [
            "aatype", "esm2", "frame_R", "frame_t", "frame_mask",
            "torsion_angles", "torsion_mask", "atom14_pos", "atom14_mask",
            "residue_index", "adjacent",
        ]

        # ── FIX: FILTER not clamp ─────────────────────────────────────────
        # Scan every file ONCE at startup. Keep only those where N <= max_res.
        # This is the right place — we're already opening every file for
        # preloading anyway, so there's zero extra I/O cost.
        # Proteins longer than max_res are excluded entirely; their ESM
        # embeddings, geometry, and dynamics are all internally consistent
        # and should not be truncated mid-chain.
        # ──────────────────────────────────────────────────────────────────
        cap = None if self.max_preload_gb <= 0 else int(self.max_preload_gb * (1024 ** 3))
        used = 0
        kept = 0
        filtered_long = 0
        filtered_cap = 0
        self.files = []
        self._file_T = []

        for fpath in all_files:
            with np.load(fpath, allow_pickle=False) as z:
                N = int(z["aatype"].shape[0])

                # ── FILTER: skip proteins longer than max_res ──────────────
                if N > self.max_res:
                    filtered_long += 1
                    continue
                # ──────────────────────────────────────────────────────────
                T_traj = int(z["frame_R"].shape[0])
                # Preload into RAM if requested and cap not exceeded
                if self.preload:
                    item = {k: z[k] for k in keep_keys if k in z.files}
                    item_bytes = sum(v.nbytes for v in item.values() if isinstance(v, np.ndarray))
                    if cap is not None and (used + item_bytes) > cap:
                        filtered_cap += 1
                        continue
                    self._ram[fpath] = item
                    used += item_bytes

            self.files.append(fpath)
            self._file_T.append(T_traj)
            kept += 1

        # Pre-assign a fixed stride to every virtual index so the sampler can group them.
        # Round-robin assignment ensures balanced stride coverage regardless of trajectory length.
        # Feasible strides per file are those where the trajectory is long enough for a full window.
        W_ = self.W
        cand_ = list(self.multi_stride) if self.multi_stride else [int(cfg.get("WINDOW_STRIDE", 1))]

        self._stride_for_idx: List[int] = []
        for fi, T_fi in enumerate(self._file_T):
            feasible_fi = [s for s in cand_ if (T_fi - W_ * s - 1) >= 0]
            if not feasible_fi:
                feasible_fi = [cand_[0]]   # fallback (will raise in __getitem__, same as before)
            n_repeats = self.train_repeat if self.train else self.val_repeat
            for rep in range(n_repeats):
                # Round-robin: each repeat gets the next stride in the feasible list
                assigned = feasible_fi[rep % len(feasible_fi)]
                self._stride_for_idx.append(assigned)

        if kept == 0:
            raise RuntimeError(
                f"[Dataset] No trajectories remain after filtering! "
                f"All {len(all_files)} files had N > MAX_RES={self.max_res}. "
                f"Increase MAX_RES or use a different dataset."
            )

        print(
            f"[Dataset] {'train' if train else 'val'}: "
            f"{kept} kept  |  {filtered_long} filtered (N > {self.max_res})  |  "
            f"{filtered_cap} filtered (RAM cap)"
            + (f"  |  RAM used: {used / 1e9:.2f} GB" if self.preload else "")
        )
        # ── END FIX ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self.train:
            return len(self.files) * int(self.train_repeat)
        else:
            return len(self.files) * int(self.val_repeat)

    def _load(self, fpath: str):
        if self.preload:
            return self._ram[fpath]
        return np.load(fpath, allow_pickle=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Map dataset index -> (file_idx, bin_idx/frac_idx)
        if self.train:
            file_idx = int(idx) // int(self.train_repeat)
            bin_idx  = int(idx) %  int(self.train_repeat)
            frac_idx = 0
        else:
            file_idx = int(idx) // int(self.val_repeat)
            bin_idx  = 0
            frac_idx = int(idx) % int(self.val_repeat)

        fpath = self.files[file_idx]
        z = self._load(fpath)

        # ---- load arrays (no padding yet) ----
        aatype = z["aatype"].astype(np.int64, copy=False)       # (N,)

        _enc = str(z.get("aatype_encoding", b""))
        if _enc and "ARNDCQEGHILKMFPSTWYV" not in _enc and "openfold_restypes" not in _enc:
            import warnings
            warnings.warn(
                f"[AATYPE] {os.path.basename(fpath)}: aatype_encoding='{_enc}' "
                "does not confirm OpenFold restypes order. "
                "If this NPZ was extracted with the old alphabetical AA_ORDER, "
                "rc table lookups will be wrong for 17/20 residues. "
                "Re-extract or run the remap utility.",
                stacklevel=2,
            )

        res_idx = z.get("residue_index", None)
        if res_idx is None:
            res_idx = np.arange(int(aatype.shape[0]), dtype=np.int32)
        else:
            res_idx = res_idx.astype(np.int32, copy=False)
        esm = z["esm2"].astype(np.float32, copy=False)          # (N,D)

        R = z["frame_R"].astype(np.float32, copy=False)         # (T,N,3,3)
        x = z["frame_t"].astype(np.float32, copy=False)         # (T,N,3)
        m = z["frame_mask"].astype(np.float32, copy=False)      # (T,N)

        tors  = z.get("torsion_angles", None)
        torsm = z.get("torsion_mask", None)
        if tors is None:
            tors = np.zeros((R.shape[0], R.shape[1], 7), np.float32)
        else:
            tors = tors.astype(np.float32, copy=False)
        if torsm is None:
            torsm = np.zeros((R.shape[0], R.shape[1], 7), np.float32)
        else:
            torsm = torsm.astype(np.float32, copy=False)

        a14  = z.get("atom14_pos", None)
        a14m = z.get("atom14_mask", None)
        if a14 is None:
            a14 = np.zeros((R.shape[0], R.shape[1], 14, 3), np.float32)
        else:
            a14 = a14.astype(np.float32, copy=False)
        if a14m is None:
            a14m = np.zeros((R.shape[0], R.shape[1], 14), np.float32)
        else:
            a14m = a14m.astype(np.float32, copy=False)

        adjacent = z.get("adjacent", None)
        if adjacent is None:
            adjacent = np.ones((max(0, int(aatype.shape[0]) - 1),), np.float32)
        else:
            adjacent = adjacent.astype(np.float32, copy=False)

        if not self.preload:
            z.close()

        # ── FIX: no clamp — N is already guaranteed <= max_res by __init__ ──
        # The old block:
        #   N0 = min(int(aatype.shape[0]), self.max_res)
        #   aatype = aatype[:N0]  ...etc...
        # is removed. We just read the true length.
        N0 = int(aatype.shape[0])   # guaranteed <= max_res
        # ─────────────────────────────────────────────────────────────────────

        T = int(R.shape[0])

        # ---- choose stride ----
        cand = [int(s) for s in (self.multi_stride if len(self.multi_stride) > 0
                                  else [int(self.cfg.get("WINDOW_STRIDE", 1))])]
        feasible = [s for s in cand if (T - (self.W * s) - 1) >= 0]
        if len(feasible) == 0:
            raise RuntimeError(
                f"[WINDOW] trajectory too short for any stride. file={os.path.basename(fpath)} T={T} "
                f"need >= {self.W * min(cand) + 1} (W={self.W}, strides={cand})"
            )

        if self.train:
            stride = self._stride_for_idx[idx]
            # Fallback: if pre-assigned stride isn't feasible for this file (short traj),
            # take the largest feasible stride instead of crashing.
            if stride not in feasible:
                stride = feasible[-1]
        else:
            val_stride = self.cfg.get("VAL_STRIDE", None)
            if (val_stride is not None) and (int(val_stride) in feasible):
                stride = int(val_stride)
            else:
                stride = self._stride_for_idx[idx] if self._stride_for_idx[idx] in feasible else feasible[0]

        max_t0 = T - (self.W * stride) - 1

        # ---- t0 sampling ----
        if self.train:
            if bool(self.cfg.get("RANDOM_T0_TRAIN", True)) and (max_t0 > 0):
                if self.strat_train and (self.strat_bins > 1):
                    bins = int(self.strat_bins)
                    M = int(max_t0) + 1
                    b = int(bin_idx)
                    lo = (b * M) // bins
                    hi = ((b + 1) * M) // bins - 1
                    if hi < lo:
                        hi = lo
                    if hi > max_t0:
                        hi = max_t0
                    t0 = random.randint(int(lo), int(hi))
                else:
                    t0 = random.randint(0, int(max_t0))
            else:
                t0 = 0
        else:
            if self.val_cover:
                frac = float(self.val_t0_fracs[int(frac_idx)])
                t0 = int(round(frac * float(max_t0)))
            else:
                t0 = 0

        t0 = min(max(int(t0), 0), int(max_t0))

        idxs = np.array([t0] + [t0 + (k * stride) for k in range(1, self.W + 1)], dtype=np.int64)

        # ---- slice window ----
        R_seq     = R[idxs]
        x_seq     = x[idxs] * float(self.cfg["COORD_SCALE"])
        m_seq     = m[idxs]
        tors_seq  = tors[idxs]
        torsm_seq = torsm[idxs]
        a14_seq   = a14[idxs] * float(self.cfg["COORD_SCALE"])
        a14m_seq  = a14m[idxs]

        # ---- pad to max_res (only pads proteins SHORTER than max_res) ----
        aatype = _pad_to(aatype, self.max_res, pad_value=0, axis=0)
        pad_val = int(res_idx[-1]) + 1000 if res_idx.shape[0] > 0 else 0
        res_idx = _pad_to(res_idx, self.max_res, pad_value=pad_val, axis=0).astype(np.int32, copy=False)
        esm     = _pad_to(esm,    self.max_res, pad_value=0.0, axis=0)

        R_seq     = _pad_to(R_seq,     self.max_res, pad_value=0.0, axis=1)
        x_seq     = _pad_to(x_seq,     self.max_res, pad_value=0.0, axis=1)
        m_seq     = _pad_to(m_seq,     self.max_res, pad_value=0.0, axis=1)
        tors_seq  = _pad_to(tors_seq,  self.max_res, pad_value=0.0, axis=1)
        torsm_seq = _pad_to(torsm_seq, self.max_res, pad_value=0.0, axis=1)
        a14_seq   = _pad_to(a14_seq,   self.max_res, pad_value=0.0, axis=1)
        a14m_seq  = _pad_to(a14m_seq,  self.max_res, pad_value=0.0, axis=1)
        adjacent  = _pad_to(adjacent,  max(0, self.max_res - 1), pad_value=0.0, axis=0)

        # ---- return ----
        out = dict(
            aatype=torch.from_numpy(aatype),
            residue_index=torch.from_numpy(res_idx).long(),
            esm=torch.from_numpy(esm),

            R_c=torch.from_numpy(R_seq[0]),
            x_c=torch.from_numpy(x_seq[0]),
            mask_c=torch.from_numpy(m_seq[0]),
            tors_c=torch.from_numpy(tors_seq[0]),
            torsm_c=torch.from_numpy(torsm_seq[0]),
            a14_c=torch.from_numpy(a14_seq[0]),
            a14m_c=torch.from_numpy(a14m_seq[0]),

            R_w=torch.from_numpy(R_seq[1:]),
            x_w=torch.from_numpy(x_seq[1:]),
            mask_w=torch.from_numpy(m_seq[1:]),
            tors_w=torch.from_numpy(tors_seq[1:]),
            torsm_w=torch.from_numpy(torsm_seq[1:]),
            a14_w=torch.from_numpy(a14_seq[1:]),
            a14m_w=torch.from_numpy(a14m_seq[1:]),
            adjacent=torch.from_numpy(adjacent),

            stride=torch.tensor(stride, dtype=torch.long),
            t0=torch.tensor(t0, dtype=torch.int64),
            max_t0=torch.tensor(max_t0, dtype=torch.int64),
            t0_frac=torch.tensor(0.0 if max_t0 <= 0 else (float(t0) / float(max_t0)), dtype=torch.float32),
            file_idx=torch.tensor(file_idx, dtype=torch.int64),
        )
        return out

# %% cell 25
# --------------------------------------------------------------------------------------
# 6) OpenFold atom14 builder + violation regularizer + all-atom/physics loss (ported)
# --------------------------------------------------------------------------------------

import torch


def reorder_torsions_to_openfold(tors: torch.Tensor) -> torch.Tensor:
    """Backward-compat shim.

    New extraction already stores torsions in OpenFold/AlphaFold convention:
      [pre_omega, phi, psi(O), chi1, chi2, chi3, chi4]

    Therefore **no reordering** should be applied.
    """
    return tors

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
        from openfold.utils import rigid_utils as rigid_utils_mod
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
                all_frames = FF.all_atom.torsion_angles_to_frames(backb_rigid, tors_sc, aatype)
                atom14_pos = FF.all_atom.frames_to_atom14_pos(all_frames, aatype)
                atom14_mask = self.atom14_mask_table[aatype]
                return atom14_pos, atom14_mask

            except Exception as e:
                print(f"[atom14 fallback] {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc()  # full stack trace to stderr
                pass


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

class AllAtom14Loss(nn.Module):
    """
    Atom14 supervision + OpenFold violation regularizer (if available).
    Notes:
      - No physics terms (l_phys) and no FAPE.
      - If OF_REQUIRED=True and OpenFold violation loss is unavailable, initialization fails fast.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.global_step = 0
        self.builder = OpenFoldAtom14Builder(rc, rigid_utils, feats, device=cfg["DEVICE"])

        # Mapping needed for OpenFold violation loss
        if hasattr(rc, "restype_atom14_to_atom37"):
            print("Map is avalabel restype_atom14_to_atom37")
            self.atom14_to_atom37_table = torch.as_tensor(
                rc.restype_atom14_to_atom37, dtype=torch.long, device=cfg["DEVICE"]
            )
        elif hasattr(rc, "restype_atom37_to_atom14"):
            print("Map is avalable restype_atom37_to_atom14")
            a37_to_a14 = torch.as_tensor(rc.restype_atom37_to_atom14, dtype=torch.long)
            inv = torch.full((a37_to_a14.shape[0], 14), -1, dtype=torch.long)
            for r in range(a37_to_a14.shape[0]):
                for a37 in range(a37_to_a14.shape[1]):
                    a14 = int(a37_to_a14[r, a37].item())
                    if 0 <= a14 < 14:
                        inv[r, a14] = a37
            self.atom14_to_atom37_table = inv.to(device=cfg["DEVICE"])
        else:
            self.atom14_to_atom37_table = build_restype_atom14_to_atom37_from_names(rc, cfg["DEVICE"])
            print("[OF] built restype_atom14_to_atom37 from atom names (fallback)")

        self.of_required = bool(cfg.get("OF_REQUIRED", False))
        self.of_available = (
            (of_loss is not None)
            and hasattr(of_loss, "find_structural_violations")
            and hasattr(of_loss, "violation_loss")
            and (self.atom14_to_atom37_table is not None)
        )

        if self.of_required and (not self.of_available):
            raise RuntimeError(
                "[OF] OF_REQUIRED=True but OpenFold violation loss is unavailable. "
                "Ensure your vendored OpenFold/foldflow import exposes find_structural_violations + violation_loss."
            )

    def forward(
        self,
        aatype: torch.Tensor,          # (B,N)
        R: torch.Tensor,               # (B,N,3,3)
        x: torch.Tensor,               # (B,N,3)
        tors: torch.Tensor,            # (B,N,7)
        gt_a14: torch.Tensor,          # (B,N,14,3)
        gt_a14m: torch.Tensor,         # (B,N,14)
        res_mask: torch.Tensor,        # (B,N)
        compute_of: bool = True,
        residue_index: Optional[torch.Tensor] = None,
        huber_delta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:

        # Torsions already OpenFold convention in new NPZ: [pre_omega, phi, psi(O), chi1..4]
        nan_tors = ~torch.isfinite(tors)          # (B,N,7)  True where NaN/Inf
        tors_of  = torch.where(~nan_tors, tors, torch.zeros_like(tors))
        pred_a14, pred_mask = self.builder.build_atom14(aatype, R, x, tors_of)

        # debug_check_atom14_masks(aatype, res_mask, pred_mask, gt_a14m)

        # ── Bug 3 fix: exclude atoms built from NaN torsions from supervision ───
        # nan_tors positions were replaced with 0.0 above so build_atom14 won't
        # crash, but the resulting atom positions are garbage (built from a fake
        # angle). We must zero the corresponding entries in gt_a14m before
        # building m_sup — otherwise l_a14 computes a real loss against a
        # position the model never actually predicted.
        #
        # OpenFold torsion convention → which atom14 index it places:
        #   pre_omega (idx 0) → O        (atom14 idx 4)
        #   phi       (idx 1) → N        (atom14 idx 0)
        #   psi       (idx 2) → O        (atom14 idx 4)
        #   chi1..4   (idx 3-6) → CB + all sidechain atoms (atom14 idx 3..13)
        #              conservative: if ANY chi is NaN, unsupervise the whole
        #              sidechain for that residue (correct, never over-penalises)
        nan_atom_mask = torch.zeros(
            gt_a14m.shape[0], gt_a14m.shape[1], 14,
            device=gt_a14m.device, dtype=torch.bool,
        )
        nan_atom_mask[:, :, 4] |= nan_tors[:, :, 0]            # pre_omega → O
        nan_atom_mask[:, :, 0] |= nan_tors[:, :, 1]            # phi       → N
        nan_atom_mask[:, :, 4] |= nan_tors[:, :, 2]            # psi       → O
        any_chi_nan = nan_tors[:, :, 3:].any(dim=-1)           # (B,N)
        nan_atom_mask[:, :, 3:] |= any_chi_nan.unsqueeze(-1)   # chi*      → CB + sidechain
        gt_a14m_safe = gt_a14m * (~nan_atom_mask).float()      # (B,N,14)
        # ────────────────────────────────────────────────────────────────────────

        # Masking:
        #  - pred_exists: structural existence (do NOT gate by gt_a14m)
        #  - sup mask: requires GT atom exists AND was not built from NaN torsion
        res_m = (res_mask > 0.5).float()  # (B,N)
        pred_exists = (pred_mask * res_m[..., None]).clamp(0.0, 1.0)  # (B,N,14)
        sup_exists = (pred_exists * gt_a14m_safe).clamp(0.0, 1.0)     # ← was gt_a14m
        m_sup = (sup_exists > 0.5).float()

        dbg = (self.global_step < 10) or (self.global_step % 200 == 0)

        # IMPORTANT: compute geometry losses in FP32 (disable autocast)
        with torch.autocast(device_type="cuda", enabled=False):
            # A14_HUBER_DELTA: configurable Å threshold (fixed; does NOT scale with V_STD)
            _a14_delta = huber_delta if huber_delta is not None else float(self.cfg.get("A14_HUBER_DELTA", 1.0))
            loss_a14, den_atoms, mean_dist_A = atom14_huber_loss(
                pred_a14=pred_a14.float(),
                gt_a14=gt_a14.float(),
                m_sup=m_sup.float(),
                delta=_a14_delta,
                eps=1e-8,
            )

        # (dead l_atom code removed — l_a14 from atom14_huber_loss is used for training)

        # OpenFold-style structural violation loss
        # initialize as FP32 scalar on the right device

        l_of = pred_a14.new_zeros((), dtype=torch.float32)

        if compute_of:
            if (not self.of_available):
                if self.of_required:
                    raise RuntimeError("[OF] compute_of=True but OpenFold violation loss is unavailable.")
            else:
                # ---- enforce dtype policy for OF loss ----
                # Run OF structural violations in FP32 (even if the rest of the model is autocast BF16)
                with torch.autocast(device_type="cuda", enabled=False):
                    pred_a14_f = pred_a14.float()          # (B,N,14,3)
                    pred_exists_f = pred_exists.float()    # (B,N,14) 0/1
                    aatype_l = aatype.long()               # (B,N)

                    B, N, _, _ = pred_a14_f.shape
                    if residue_index is None:
                        residue_index_l = torch.arange(N, device=pred_a14_f.device, dtype=torch.long).view(1, N).expand(B, N)
                    else:
                        residue_index_l = residue_index.to(device=pred_a14_f.device, dtype=torch.long, non_blocking=True)
                        if residue_index_l.shape != (B, N):
                            raise RuntimeError(f"[OF] residue_index shape mismatch: expected {(B,N)}, got {tuple(residue_index_l.shape)}")

                    # mapping table must be on GPU and long
                    residx_atom14_to_atom37 = self.atom14_to_atom37_table[aatype_l].long()  # (B,N,14)

                    batch_v = dict(
                        atom14_atom_exists=pred_exists_f,
                        residue_index=residue_index_l,
                        aatype=aatype_l,
                        residx_atom14_to_atom37=residx_atom14_to_atom37,
                    )

                    # NOTE (cleanup): Some earlier debug copies had a corrupted line merge inside this try/except
                    # (e.g., a SyntaxError around 'raise' / 'l_of'). This block is the corrected reference implementation.
                    # OpenFold violation loss (OF loss) is intentionally kept enabled; failures are surfaced when of_required=True.
                    try:
                        viol = call_with_sig(
                            of_loss.find_structural_violations,
                            dict(
                                batch=batch_v,
                                atom14_pred_positions=pred_a14_f,
                                violation_tolerance_factor=float(self.cfg.get("OF_VIOL_TOL", 12.0)),
                                clash_overlap_tolerance=float(self.cfg.get("OF_CLASH_TOL", 1.5)),
                            ),
                        )
                        l_of_raw = call_with_sig(
                            of_loss.violation_loss,
                            dict(
                                violations=viol,
                                atom14_atom_exists=pred_exists_f,
                            ),
                        )

                        # ---- robust reduction ----
                        if torch.is_tensor(l_of_raw):
                            l_of_raw = l_of_raw.float()
                            # if it's per-example or per-residue, reduce explicitly to scalar
                            l_of = l_of_raw.mean()
                        else:
                            l_of = pred_a14_f.new_zeros((), dtype=torch.float32)

                    except Exception as e:
                        # If OF is optional, return 0 but do not silently hide issues during debugging
                        if self.of_required:
                            raise
                        if (self.global_step < 10) or (self.global_step % 200 == 0):
                            print("[OF] warning: violation loss failed:", repr(e))
                        l_of = pred_a14_f.new_zeros((), dtype=torch.float32)

                # Optional one-time sanity (never inside compiled forward)
                if bool(self.cfg.get("OF_SANITY_ONCE", False)) and (not hasattr(self, "_did_of_sanity")):
                    self._did_of_sanity = True
                    with torch.no_grad():
                        openfold_violation_sanity(
                            of_loss, pred_a14, pred_exists, aatype,
                            residx_atom14_to_atom37=residx_atom14_to_atom37,
                            violation_tolerance_factor=float(self.cfg.get("OF_VIOL_TOL", 12.0)),
                            clash_overlap_tolerance=float(self.cfg.get("OF_CLASH_TOL", 1.5)),
                        )

        with torch.autocast(device_type="cuda", enabled=False):
          l_atom_mse = ((pred_a14.float() - gt_a14.float()) ** 2).sum(dim=-1)   # (B,N,14) squared dist
          l_atom_mse = (l_atom_mse * m_sup.float()).sum() / (m_sup.sum().clamp_min(1.0) + 1e-8)

        return {
            "l_a14": loss_a14,                      # <-- use this for training
            "den_a14": den_atoms,                   # scalar
            "mean_dist_A": mean_dist_A,             # scalar
            "l_atom_mse": l_atom_mse,               # optional debug
            "l_of": l_of,
            "pred_a14": pred_a14,
            "pred_a14_mask": pred_exists,
            "gt_a14": gt_a14,
            "m_sup": m_sup,                         # NaN-torsion atoms excluded via gt_a14m_safe
        }


def openfold_fape_atom14(
    pred_R: torch.Tensor,     # (B,N,3,3)
    pred_x: torch.Tensor,     # (B,N,3)
    pred_a14: torch.Tensor,   # (B,N,14,3)
    pred_a14m: torch.Tensor,  # (B,N,14)    (predicted existence mask)
    gt_R: torch.Tensor,       # (B,N,3,3)
    gt_x: torch.Tensor,       # (B,N,3)
    gt_a14: torch.Tensor,     # (B,N,14,3)
    gt_a14m: torch.Tensor,    # (B,N,14)
    res_mask: torch.Tensor,   # (B,N)
    length_scale: float = 10.0,
    clamp_distance: float = 10.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    OpenFold compute_fape on atom14 points.
    Returns a scalar (FP32) mean over the batch.
    """

    # Fallback if vendored OpenFold loss is missing
    if (of_loss is None) or (not hasattr(of_loss, "compute_fape")) or (rigid_utils is None):
        return pred_x.new_zeros((), dtype=torch.float32)

    try:
        Rot = rigid_utils.Rotation
        Rigid = rigid_utils.Rigid
    except Exception:
        return pred_x.new_zeros((), dtype=torch.float32)

    # Force FP32 math for stability
    pred_Rf = pred_R.float()
    pred_xf = pred_x.float()
    gt_Rf   = gt_R.float()
    gt_xf   = gt_x.float()

    B, N = pred_xf.shape[:2]
    rm = res_mask.float().clamp(0.0, 1.0)  # (B,N)

    # Point mask: pred-exists AND gt-exists AND residue-exists
    # pred_a14m in your notebook is already "pred_exists" (includes residue gating),
    # but multiply again for safety/clarity.
    p_mask = (pred_a14m.float() * gt_a14m.float() * rm[..., None]).reshape(B, N * 14)  # (B,N*14)

    pred_pts = pred_a14.float().reshape(B, N * 14, 3)
    gt_pts   = gt_a14.float().reshape(B, N * 14, 3)

    pred_frame = Rigid(Rot(rot_mats=pred_Rf, quats=None), pred_xf)
    gt_frame   = Rigid(Rot(rot_mats=gt_Rf,   quats=None), gt_xf)

    fape = call_with_sig(
        of_loss.compute_fape,
        dict(
            pred_frames=pred_frame,
            target_frames=gt_frame,
            frames_mask=rm,
            pred_positions=pred_pts,
            target_positions=gt_pts,
            positions_mask=p_mask,
            length_scale=float(length_scale),
            l1_clamp_distance=float(clamp_distance),
            eps=float(eps),
        )
    )

    if torch.is_tensor(fape):
        return fape.float().mean()

    return pred_x.new_zeros((), dtype=torch.float32)

# %% cell 26
import torch

def atom14_huber_loss(
    pred_a14: torch.Tensor,   # (B,N,14,3) float
    gt_a14:   torch.Tensor,   # (B,N,14,3) float
    m_sup:    torch.Tensor,   # (B,N,14)   float {0,1}
    delta: float = 1.0,       # Å
    eps: float = 1e-8,
):
    """
    Huber loss over per-atom Euclidean distances, normalized by # supervised atoms.
    Returns: (loss, den_atoms, mean_dist_A)
    """
    # (B,N,14)
    m = (m_sup > 0.5).float()

    # per-atom distance in Å
    d = torch.linalg.norm(pred_a14 - gt_a14, dim=-1)  # (B,N,14)
    d = d * m

    # Huber on distances
    absd = d.abs()
    delta_t = torch.as_tensor(delta, device=d.device, dtype=d.dtype)
    quad = torch.minimum(absd, delta_t)
    lin  = absd - quad
    huber = 0.5 * quad * quad + delta_t * lin

    den_atoms = m.sum().clamp_min(1.0)
    loss = huber.sum() / (den_atoms + eps)

    mean_dist_A = (d.sum() / (den_atoms + eps))  # mean distance over supervised atoms

    return loss, den_atoms, mean_dist_A

# %% cell 27
# --------------------------------------------------------------------------------------
# 7) Model: IPA dynamics vector field in velocity-space + optional temporal attention
# --------------------------------------------------------------------------------------

# IPA is imported lazily inside IPABlock to avoid any compiled/CUDA dependencies.
from torch.utils.checkpoint import checkpoint


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
        """
        IPA block with FP32 island + PACKED valid-only execution.
        This eliminates NaNs from IPA softmax on fully-masked query rows.
        """
        with torch.autocast(device_type=s.device.type, enabled=False):
            s_f = s.float()

            # mask -> bool [B,N]
            m_bool = (mask > 0.5) if mask.dtype != torch.bool else mask
            m_bool = m_bool.to(torch.bool)
            B, N, C = s_f.shape

            # Prepare z for FP32 computation *after slicing* to avoid allocating a full FP32 pair tensor on GPU.
            # NOTE: Casting the full (B,N,N,Cz) to FP32 here (e.g., z.float()) can add multiple GB of VRAM.
            z_in = z

            # Output buffer (FP32)
            ds_out = torch.zeros_like(s_f)

            # --- helpers ---
            def _slice_z(zin, b, idx):
                # zin: tensor or list; slice to (1, Nv, Nv, Cz) when pairwise, else appropriate
                if isinstance(zin, list):
                    out = []
                    for zi in zin:
                        if zi.ndim == 4:  # (B,N,N,Cz)
                            zib = zi[b:b+1].index_select(1, idx).index_select(2, idx)
                        elif zi.ndim == 3:  # (B,N,Cz)
                            zib = zi[b:b+1].index_select(1, idx)
                        else:
                            zib = zi[b:b+1]
                        if self.cast_z_fp32_after_slice and zib.dtype != torch.float32: zib = zib.float()
                        out.append(zib)
                    return out
                else:
                    if zin.ndim == 4:     # (B,N,N,Cz)
                        zib = zin[b:b+1].index_select(1, idx).index_select(2, idx)
                        if self.cast_z_fp32_after_slice and zib.dtype != torch.float32: zib = zib.float()
                        return zib
                    elif zin.ndim == 3:   # (B,N,Cz)
                        zib = zin[b:b+1].index_select(1, idx)
                        if self.cast_z_fp32_after_slice and zib.dtype != torch.float32: zib = zib.float()
                        return zib
                    else:
                        zib = zin[b:b+1]
                        if self.cast_z_fp32_after_slice and zib.dtype != torch.float32: zib = zib.float()
                        return zib

            def _slice_rigids(r, b, idx):
                """
                Try common OpenFold rigid APIs.
                Falls back to slicing underlying tensors and rebuilding via make_rigid().
                """
                # Most OpenFold Rigid implementations support tensor-like indexing
                try:
                    return r[b:b+1, idx]
                except Exception:
                    pass

                # Try method-based access
                try:
                    Rm = r.get_rots().get_rot_mats()[b:b+1, idx]   # (1,Nv,3,3)
                    tm = r.get_trans()[b:b+1, idx]                # (1,Nv,3)
                    return make_rigid(Rm, tm)
                except Exception:
                    pass

                # Try attribute-based access
                rots = getattr(r, "rots", None) or getattr(r, "_rots", None) or getattr(r, "rot", None)
                trans = getattr(r, "trans", None) or getattr(r, "_trans", None) or getattr(r, "t", None)
                if rots is None or trans is None:
                    raise RuntimeError("Unable to slice rigids: unknown rigid_utils.Rigid API")

                # rots may be Rotation wrapper
                if hasattr(rots, "get_rot_mats"):
                    Rm_full = rots.get_rot_mats()
                elif hasattr(rots, "rot_mats"):
                    Rm_full = rots.rot_mats
                else:
                    Rm_full = rots

                Rm = Rm_full[b:b+1, idx]
                tm = trans[b:b+1, idx]
                return make_rigid(Rm, tm)

            # --- run IPA per sample on valid residues only ---
            for b in range(B):
                idx = torch.nonzero(m_bool[b], as_tuple=False).squeeze(-1)  # (Nv,)
                if idx.numel() == 0:
                    continue

                s_b = s_f[b:b+1].index_select(1, idx)   # (1,Nv,C)
                z_b = _slice_z(z_in, b, idx)            # (1,Nv,Nv,Cz) or list
                r_b = _slice_rigids(rigids, b, idx)     # rigid for (1,Nv)

                # mask inside IPA is all-ones because everything here is valid
                mask_ipa = torch.ones((1, idx.numel()), device=s.device, dtype=torch.float32)

                ds_b = self.ipa(s_b, z_b, r_b, mask_ipa)  # (1,Nv,C)
                if not torch.isfinite(ds_b).all():
                  raise RuntimeError(f"[IPA] Non-finite ds in packed IPA: b={b}, Nv={idx.numel()}")

                # Scatter back
                ds_out[b:b+1].index_copy_(1, idx, ds_b)

            ds = ds_out

        # back to outer dtype
        ds = ds.to(dtype=s.dtype)

        # residual + norms
        s = self.ln1(s + ds)
        s = self.ln2(s + self.ff(s))

        # enforce masked residues are exactly zero
        m3 = ((mask > 0.5) if mask.dtype != torch.bool else mask).to(torch.bool)[..., None]
        s = torch.where(m3, s, torch.zeros_like(s))
        return s

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

        self.pair_in = nn.Linear(64, c_z)

        # Relative positional bias for pair features (sequence separation)
        self.relpos_k = int(cfg.get("RELPOS_MAX", 32))
        self.relpos_emb = nn.Embedding(2 * self.relpos_k + 1, c_z)

        self.blocks = nn.ModuleList([IPABlock(c_s, c_z, cfg) for _ in range(int(cfg["IPA_BLOCKS"]))])

        n_blocks = int(cfg.get("IPA_BLOCKS", 6))
        aux_every = int(cfg.get("AUX_EVERY", 2))    # add aux head every 2 blocks
        self.aux_heads = nn.ModuleList([
            nn.Linear(c_s, 13)
            for i in range(n_blocks)
            if (i + 1) % aux_every == 0 and (i + 1) < n_blocks  # not on the final block
        ])

        # optional temporal attention on single features before IPA
        self.use_temp = bool(cfg.get("USE_TEMPORAL_ATTENTION", False))
        if self.use_temp:
            self.temp = TemporalEncoder(
                c=c_s,
                n_layers=int(cfg.get("TEMP_ATTN_LAYERS", 0)),
                n_heads=int(cfg.get("TEMP_ATTN_HEADS", 8)),
                dropout=float(cfg.get("TEMP_ATTN_DROPOUT", 0.0)),
                cast_fp32=bool(cfg.get("CAST_Z_FP32_AFTER_SLICE", True))
            )
        else:
            self.temp = None

        # ── Stride conditioning embed ─────────────────────────────────────────
        # Allows every IPA layer to know the temporal resolution of the batch.
        # Supports strides 0-8; embed dim matches c_s so no projection needed.
        self.stride_embed = nn.Embedding(9, c_s)   # strides 0-8
        # ─────────────────────────────────────────────────────────────────────

        # ── Velocity output head(s) ───────────────────────────────────────────
        # USE_STRIDE_HEADS=True: separate linear per stride — each specialises.
        #   stride=1 head learns small, smooth velocities (thermal regime).
        #   stride=4 head learns larger conformational shifts.
        # USE_STRIDE_HEADS=False (default): single shared head, current behaviour.
        self.use_stride_heads = bool(cfg.get("USE_STRIDE_HEADS", False))
        if self.use_stride_heads:
            _strides = cfg.get("MULTI_STRIDE", [1, 2, 4])
            self.vel_heads = nn.ModuleDict({
                str(s): nn.Linear(c_s, 3 + 3 + 7) for s in _strides
            })
            self.head = None   # disabled; slot kept for load_state_dict compatibility
        else:
            self.vel_heads = None
            self.head = nn.Linear(c_s, 3 + 3 + 7)
        # ─────────────────────────────────────────────────────────────────────

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
        stride: int = 1,                           # window stride (1/2/4); used for stride embed + head routing
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        B, W, N, _ = x_prev.shape
        m = mask_prev.unsqueeze(-1)
        res_mask = (mask_prev[:, 0, :] > 0.5)     # bool, (B,N)
        aa_idx, unk_frac = sanitize_aatype(aatype, res_mask, unk=20)
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

        # ── Change 4b: stride conditioning token ─────────────────────────────
        # stride_embed maps the integer stride (1/2/4) to a c_s-dim vector,
        # broadcast over all window steps and residues as an additive bias.
        # Clamp to embedding range so an unseen stride value degrades gracefully.
        stride_idx = torch.tensor(
            min(int(stride), self.stride_embed.num_embeddings - 1),
            device=x_prev.device, dtype=torch.long
        )
        str_emb = self.stride_embed(stride_idx).view(1, 1, 1, -1).expand(B, W, N, -1)
        s_feat = self.in_ln(aa + e + st + t_emb + d_emb + str_emb) * m
        # ─────────────────────────────────────────────────────────────────────

        # temporal attention (per residue) before IPA
        if self.use_temp and self.temp is not None:
            s_feat = self.temp(s_feat)

        # flatten steps for IPA blocks
        s = s_feat.reshape(B * W, N, -1)
        mask_flat = mask_prev.reshape(B * W, N)
        x_flat = x_prev.reshape(B * W, N, 3)
        R_flat = R_prev.reshape(B * W, N, 3, 3)

        step_chunk = int(self.cfg.get("STEP_CHUNK", 1))
        outs = []

        # Build a map: block_index -> aux_head_index for intermediate supervision
        aux_every = int(self.cfg.get("AUX_EVERY", 2))
        n_blks = len(self.blocks)
        # qualifying block indices: every aux_every-th block, excluding the final block
        _aux_blk_indices = [i for i in range(n_blks)
                            if (i + 1) % aux_every == 0 and (i + 1) < n_blks]
        _aux_head_map = {blk_i: head_i for head_i, blk_i in enumerate(_aux_blk_indices)}
        # aux_preds[head_i] accumulates (B, C, N, 13) chunks across step_chunks
        aux_chunk_lists: List[List[torch.Tensor]] = [[] for _ in self.aux_heads]

        # ── Change 4c: resolve which output head to use for this stride ──────
        if self.use_stride_heads and self.vel_heads is not None:
            # fall back to the nearest available stride key if exact match missing
            _stride_key = str(stride)
            if _stride_key not in self.vel_heads:
                _stride_key = str(min(self.vel_heads.keys(), key=lambda k: abs(int(k) - stride)))
            _out_head = self.vel_heads[_stride_key]
        else:
            _out_head = self.head
        # ─────────────────────────────────────────────────────────────────────

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

            for blk_i, blk in enumerate(self.blocks):
                if self.training and bool(self.cfg.get('GRAD_CHECKPOINT_IPA', True)):
                    # Activation checkpointing through the IPA block to reduce VRAM (recompute in backward).
                    s_chunk = checkpoint(lambda _s: blk(_s, z, rigids, mask_flat), s_chunk, use_reentrant=False)
                else:
                    s_chunk = blk(s_chunk, z, rigids, mask_flat)

                # Auxiliary head: collect intermediate velocity prediction for deep supervision
                if blk_i in _aux_head_map:
                    head_i = _aux_head_map[blk_i]
                    aux_pred = self.aux_heads[head_i](s_chunk)          # (B*C, N, 13)
                    aux_chunk_lists[head_i].append(aux_pred.view(B, C, N, -1))

            out_chunk = _out_head(s_chunk).view(B, C, N, -1)           # uses stride-resolved head
            outs.append(out_chunk)

        out = torch.cat(outs, dim=1)   # (B,W,N,13)
        u_v     = out[..., :3]
        u_omg   = out[..., 3:6]
        u_thdot = out[..., 6:]

        # Concatenate aux predictions across step_chunks -> list of (B,W,N,13) tensors
        aux_outputs: Optional[List[torch.Tensor]] = (
            [torch.cat(chunks, dim=1) for chunks in aux_chunk_lists]
            if aux_chunk_lists and aux_chunk_lists[0]
            else None
        )

        return u_v, u_omg, u_thdot, aux_outputs

# %% cell 28
# --------------------------------------------------------------------------------------
# 8) Loss helpers
# --------------------------------------------------------------------------------------

def rotation_geodesic_mse(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mask-safe SO(3) geodesic loss.

    Key properties:
      - Forces masked residues to identity BEFORE any trig/log math (prevents NaN*0 propagation).
      - Uses atan2(sin, cos) instead of acos(cos) (better gradient behavior near cos≈±1).
      - Reduces in FP32.
    """
    # Promote to FP32 for stability (even if caller is under autocast)
    Rp = R_pred.to(torch.float32)
    Rg = R_gt.to(torch.float32)

    # Boolean mask
    m = (mask > 0.5) if mask.dtype != torch.bool else mask
    m = m.to(torch.bool)

    # Build broadcastable identity
    # Shape: (1,1,...,3,3) matching Rp.ndim
    I = torch.eye(3, device=Rp.device, dtype=torch.float32)
    I = I.view((1,) * (Rp.ndim - 2) + (3, 3))

    # CRITICAL: eliminate masked residues BEFORE computing relative rotation
    Rp = torch.where(m[..., None, None], Rp, I)
    Rg = torch.where(m[..., None, None], Rg, I)

    # Relative rotation
    R = Rp.transpose(-1, -2) @ Rg

    # cos(theta) = (trace(R)-1)/2
    tr  = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = 0.5 * (tr - 1.0)
    cos = cos.clamp(-1.0 + eps, 1.0 - eps)

    # sin(theta) from skew-symmetric part:
    # v = [R32-R23, R13-R31, R21-R12],  ||v|| = 2*sin(theta)
    wx = R[..., 2, 1] - R[..., 1, 2]
    wy = R[..., 0, 2] - R[..., 2, 0]
    wz = R[..., 1, 0] - R[..., 0, 1]
    s2  = (wx * wx + wy * wy + wz * wz)
    sin = 0.5 * torch.sqrt(s2.clamp_min(eps))   # eps MUST be > 0

    # theta in [0, pi]
    theta = torch.atan2(sin, cos)

    # squared angle ~ squared rotvec norm for small angles; robust and bounded
    l = theta * theta

    # Ensure masked entries are exactly zero (no NaN*0)
    l = torch.where(m, l, torch.zeros_like(l))

    denom = m.to(torch.float32).sum().clamp_min(1.0)
    return l.sum() / denom

def torsion_loss(
    t_pred: torch.Tensor,
    t_gt: torch.Tensor,
    t_mask: torch.Tensor,
    res_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    # boolean mask: [B,N,T]
    m = (t_mask > 0.5)
    m = m & (res_mask.unsqueeze(-1) > 0.5)
    m = m.to(torch.bool)

    # FP32 compute for stability
    tp = t_pred.to(torch.float32)
    tg = t_gt.to(torch.float32)

    # CRITICAL: gate BEFORE wrap/cos so masked NaNs never enter trig
    dt = torch.where(m, tp - tg, torch.zeros_like(tp))
    d  = wrap_to_pi(dt)
    l  = 1.0 - torch.cos(d)

    # ensure masked entries are exactly zero (no NaN*0)
    l = torch.where(m, l, torch.zeros_like(l))

    denom = m.to(torch.float32).sum().clamp_min(1.0)
    return l.sum() / denom

def distogram_loss(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    res_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    # m: [B,N] boolean
    m = (res_mask > 0.5) if res_mask.dtype != torch.bool else res_mask
    m = m.to(torch.bool)

    # FP32 compute
    xp = x_pred.to(torch.float32)
    xg = x_gt.to(torch.float32)

    # CRITICAL: gate coords so masked residues cannot inject NaNs into cdist
    xp = torch.where(m[..., None], xp, torch.zeros_like(xp))
    xg = torch.where(m[..., None], xg, torch.zeros_like(xg))

    Dp = torch.cdist(xp, xp)   # [B,N,N]
    Dg = torch.cdist(xg, xg)   # [B,N,N]

    mm = (m.unsqueeze(-1) & m.unsqueeze(-2))  # [B,N,N] bool
    err = (Dp - Dg) ** 2

    # mask-safe reduction (no NaN*0)
    err = torch.where(mm, err, torch.zeros_like(err))

    denom = mm.to(torch.float32).sum().clamp_min(1.0)
    return err.sum() / denom

def distogram_loss_per_sample(x_pred, x_gt, res_mask):
    m = (res_mask > 0.5) if res_mask.dtype != torch.bool else res_mask
    m = m.to(torch.bool)

    xp = x_pred.to(torch.float32)
    xg = x_gt.to(torch.float32)

    xp = torch.where(m[..., None], xp, torch.zeros_like(xp))
    xg = torch.where(m[..., None], xg, torch.zeros_like(xg))

    Dp = torch.cdist(xp, xp)
    Dg = torch.cdist(xg, xg)

    mm  = (m.unsqueeze(-1) & m.unsqueeze(-2))
    err = (Dp - Dg) ** 2
    return masked_mse_per_sample(err, mm, reduce_dims=(1, 2))


def masked_mse(err: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (err * mask).sum() / (mask.sum() + 1e-6)

def masked_huber_pos(
    xP: torch.Tensor,       # (..., 3) predicted positions
    xG: torch.Tensor,       # (..., 3) GT positions
    mask: torch.Tensor,     # (...,) float mask {0,1}
    delta: float = 5.0,     # Huber threshold in Angstrom
) -> torch.Tensor:
    """
    Huber loss on per-residue L2 displacement in Angstrom (not squared MSE).

    WHY: raw MSE on squared displacements (d^2) lets a single hard batch with
    d=10A contribute 4x more gradient than a normal batch with d=5A, and 16x
    more than a good prediction at d=2.5A. This creates the l_pos stdev=12.34
    / max=113 problem seen in train_dense.jsonl — all t0_frac quintiles have
    the same mean (~28) but wildly different maxima.

    Huber on d (not d^2):
      d < delta  ->  0.5 * d^2      (quadratic; same gradient direction as MSE)
      d >= delta ->  delta*(d-delta/2)  (linear; gradient capped at delta)

    With delta=5 A (~1 stdev of per-residue displacement):
      d=3 A (good)   : loss=4.5   vs MSE=9    (same order)
      d=5 A (mean)   : loss=12.5  vs MSE=25   (half — use W_POS*2 or accept new scale)
      d=10 A (spike) : loss=37.5  vs MSE=100  (2.7x reduction — key fix)
      d=20 A (blow-up): loss=87.5 vs MSE=400  (4.6x reduction)

    Note: switching from d^2 to d halves the typical loss value. W_POS has been
    reduced from 0.05->0.02 separately, so the effective gradient contribution
    is also reduced; re-tune W_POS if the position signal becomes too weak.
    """
    d = torch.linalg.norm(xP - xG, dim=-1)          # (...,) displacement in A
    delta_t = d.new_tensor(delta)
    quad = torch.minimum(d, delta_t)                 # min(d, delta)
    lin  = d - quad                                  # max(d - delta, 0)
    h    = 0.5 * quad * quad + delta_t * lin         # Huber value per residue
    return (h * mask).sum() / (mask.sum() + 1e-6)


def masked_mse_per_sample(err: torch.Tensor, mask: torch.Tensor, reduce_dims=(1,)):
    """
    err, mask: shape [B, ...] (same)
    reduce_dims: tuple of dims to reduce over (not including batch dim 0)
    returns: scalar (equal weight per sample)
    """
    m = (mask > 0.5) if mask.dtype != torch.bool else mask
    m = m.to(torch.bool)
    e = err.to(torch.float32)
    e = torch.where(m, e, torch.zeros_like(e))
    denom = m.to(torch.float32).sum(dim=reduce_dims).clamp_min(1.0)
    num   = e.sum(dim=reduce_dims)
    return num / denom


# --------------------------------------------------------------------------------------
# 9) Velocity helpers: normalization, AR(1) noise, integration
# --------------------------------------------------------------------------------------

def normalize_velocities(v, omg, thdot, cfg, stride: int = None):
    if not bool(cfg.get("NORMALIZE_VELOCITIES", True)):
        return v, omg, thdot

    # Per-stride stds take priority when stride is known
    if stride is not None:
        v_by_s   = cfg.get("V_STD_BY_STRIDE",     {})
        o_by_s   = cfg.get("OMG_STD_BY_STRIDE",   {})
        th_by_s  = cfg.get("THDOT_STD_BY_STRIDE", {})
        v_std   = float(v_by_s.get(stride,   cfg.get("V_STD",     1.0)))
        o_std   = float(o_by_s.get(stride,   cfg.get("OMG_STD",   1.0)))
        th_std  = float(th_by_s.get(stride,  cfg.get("THDOT_STD", 1.0)))
    else:
        v_std  = float(cfg.get("V_STD",     1.0))
        o_std  = float(cfg.get("OMG_STD",   1.0))
        th_std = float(cfg.get("THDOT_STD", 1.0))

    return v / v_std, omg / o_std, thdot / th_std


def denormalize_velocities(v, omg, thdot, cfg, stride: int = None):
    if not bool(cfg.get("NORMALIZE_VELOCITIES", True)):
        return v, omg, thdot

    if stride is not None:
        v_by_s   = cfg.get("V_STD_BY_STRIDE",     {})
        o_by_s   = cfg.get("OMG_STD_BY_STRIDE",   {})
        th_by_s  = cfg.get("THDOT_STD_BY_STRIDE", {})
        v_std   = float(v_by_s.get(stride,   cfg.get("V_STD",     1.0)))
        o_std   = float(o_by_s.get(stride,   cfg.get("OMG_STD",   1.0)))
        th_std  = float(th_by_s.get(stride,  cfg.get("THDOT_STD", 1.0)))
    else:
        v_std  = float(cfg.get("V_STD",     1.0))
        o_std  = float(cfg.get("OMG_STD",   1.0))
        th_std = float(cfg.get("THDOT_STD", 1.0))

    return v * v_std, omg * o_std, thdot * th_std


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

    At each step k, after the standard Euler update we apply a soft pull
    back toward the conditioning frame (x0, R0):

        alpha_k  = anchor_alpha * (1 - k / W)   # linearly decays to 0 at k=W-1
        x_k      = (1 - alpha_k) * x_euler  +  alpha_k * x0
        R_k      = R0 @ exp((1 - alpha_k) * log(R0^T @ R_euler))

    alpha_k is largest at k=0 (where errors start accumulating) and reaches
    zero at the window endpoint, so the endpoint prediction is unconstrained.

    Setting anchor_alpha=0.0 recovers the original pure Euler integration.

    Args:
        x0, R0, tors0    : (B,N,...) conditioning frame at t=0
        v_local, omg,
        thdot            : (B,W,N,...) predicted velocities for steps 0..W-1
        dt               : timestep (physical units)
        anchor_alpha     : strength of anchor at k=0 (0 = pure Euler)

    Returns:
        x_w, R_w, tors_w : (B,W,N,...) integrated frames at t+1 .. t+W
    """
    B, W, N, _ = v_local.shape
    x   = x0
    R   = R0
    tors = tors0
    xs, Rs, torss = [], [], []

    for k in range(W):
        v_k   = v_local[:, k]   # (B,N,3) local-frame velocity
        omg_k = omg[:, k]       # (B,N,3) angular velocity
        th_k  = thdot[:, k]     # (B,N,7) torsion rates

        # ── Standard Euler step ──────────────────────────────────────────────
        dx_g    = (R @ (v_k * dt).unsqueeze(-1)).squeeze(-1)  # (B,N,3)
        x_euler = x + dx_g
        R_euler = R @ so3_exp((omg_k * dt).reshape(-1, 3)).reshape(B, N, 3, 3)
        tors_euler = wrap_to_pi(tors + th_k * dt)

        # ── Anchor correction ────────────────────────────────────────────────
        # alpha decays linearly: full at k=0, zero at k=W (endpoint unconstrained)
        if anchor_alpha > 0.0:
            den = max(W - 1, 1)
            alpha_k = anchor_alpha * (1.0 - float(k) / float(den))

            # Translation: linear blend toward conditioning position x0
            x = (1.0 - alpha_k) * x_euler + alpha_k * x0

            # Rotation: blend in SO(3) tangent space (log map)
            #   R_blended = R0 @ exp((1-alpha_k) * log(R0^T @ R_euler))
            #   At alpha_k=1: R_blended = R0 (full anchor)
            #   At alpha_k=0: R_blended = R_euler (pure Euler)
            dR   = R0.transpose(-1, -2) @ R_euler          # (B*N,3,3) relative rotation
            log_dR = so3_log(dR.reshape(-1, 3, 3))          # (B*N,3) tangent vector
            R = R0 @ so3_exp(
                ((1.0 - alpha_k) * log_dR).reshape(-1, 3)
            ).reshape(B, N, 3, 3)

            # Torsions: linear blend toward conditioning torsion tors0
            tors = wrap_to_pi(
                (1.0 - alpha_k) * tors_euler + alpha_k * tors0
            )
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
        model.load_state_dict(self.shadow, strict=False)


# --------------------------------------------------------------------------------------
# Peptide-bond preconditioning (continuous geometry + velocity constraint)
# --------------------------------------------------------------------------------------

def _safe_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=-1) + eps)

def _safe_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Angle at vertex b formed by (a-b) and (c-b), in radians. Stable via atan2."""
    u = a - b
    v = c - b
    u_norm = _safe_norm(u, eps=eps)
    v_norm = _safe_norm(v, eps=eps)
    u_hat = u / u_norm.unsqueeze(-1)
    v_hat = v / v_norm.unsqueeze(-1)
    cross = torch.cross(u_hat, v_hat, dim=-1)
    sin = _safe_norm(cross, eps=eps)
    cos = (u_hat * v_hat).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.atan2(sin, cos)

def _safe_dihedral(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Dihedral angle (radians) for points p0-p1-p2-p3, stable via atan2."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1n = b1 / _safe_norm(b1, eps=eps).unsqueeze(-1)

    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n

    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)

def peptide_geom_loss_atom14(
    pred_a14: torch.Tensor,          # (B,N,14,3)
    pred_a14m: torch.Tensor,         # (B,N,14)
    res_mask: torch.Tensor,          # (B,N)
    adjacent: torch.Tensor,          # (B,N-1) or (N-1)
    aatype: torch.Tensor,            # (B,N) int
    cfg: Dict[str, Any],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Smooth peptide geometry preconditioner using atom14 positions.

    Penalizes:
      - C(i)-N(i+1) bond length (spring)
      - Two bond angles: CA(i)-C(i)-N(i+1) and C(i)-N(i+1)-CA(i+1)
      - Optional omega planarity: dihedral CA(i)-C(i)-N(i+1)-CA(i+1)

    Returns (loss_mean, denom_pairs) where denom_pairs is #valid adjacent pairs.
    """
    P = pred_a14.to(torch.float32)
    M = pred_a14m.to(torch.float32)
    rm = res_mask.to(torch.float32)

    if adjacent.dim() == 1:
        adj = adjacent.unsqueeze(0).expand(P.shape[0], -1).to(torch.float32)
    else:
        adj = adjacent.to(torch.float32)

    IDX_N, IDX_CA, IDX_C = 0, 1, 2

    C_i  = P[:, :-1, IDX_C, :]
    CA_i = P[:, :-1, IDX_CA, :]
    N_j  = P[:, 1:,  IDX_N, :]
    CA_j = P[:, 1:,  IDX_CA, :]

    m_pair = (rm[:, :-1] * rm[:, 1:] * adj).clamp(0.0, 1.0)

    mC   = M[:, :-1, IDX_C]
    mCAi = M[:, :-1, IDX_CA]
    mN   = M[:, 1:,  IDX_N]
    mCAj = M[:, 1:,  IDX_CA]
    m_pair = m_pair * mC * mN * mCAi * mCAj

    denom = m_pair.sum().clamp_min(0.0)
    if denom.item() <= 0:
        z = torch.zeros((), device=P.device, dtype=torch.float32)
        return z, z

    d0   = float(cfg.get("PEP_CN_LEN_A", 1.329))
    ang1 = float(cfg.get("PEP_ANG_CA_C_N_DEG", 116.2)) * math.pi / 180.0
    ang2 = float(cfg.get("PEP_ANG_C_N_CA_DEG", 121.7)) * math.pi / 180.0

    w_len = float(cfg.get("PEP_GEO_W_LEN", 1.0))
    w_ang = float(cfg.get("PEP_GEO_W_ANG", 0.2))
    w_omg = float(cfg.get("PEP_GEO_W_OMEGA", 0.05))
    allow_cis_pro = bool(cfg.get("PEP_ALLOW_CIS_PRO", True))

    d = _safe_norm(C_i - N_j, eps=eps)
    l_len = ((d - d0) ** 2) * m_pair

    a1 = _safe_angle(CA_i, C_i, N_j, eps=eps)
    a2 = _safe_angle(C_i, N_j, CA_j, eps=eps)
    l_ang = ((a1 - ang1) ** 2 + (a2 - ang2) ** 2) * m_pair

    if w_omg > 0.0:
        omega = _safe_dihedral(CA_i, C_i, N_j, CA_j, eps=eps)
        trans = 1.0 - torch.cos(omega - math.pi)

        if allow_cis_pro:
            pro_idx = 14  # 'P' in OpenFold restypes 'ARNDCQEGHILKMFPSTWYV' (was 12 under old alphabetical AA_ORDER — now corrected)
            is_pro = (aatype[:, 1:] == pro_idx).to(torch.float32)
            cis = 1.0 - torch.cos(omega)
            omg_pen = torch.minimum(cis, trans) * is_pro + trans * (1.0 - is_pro)
        else:
            omg_pen = trans

        l_omg = omg_pen * m_pair
    else:
        l_omg = torch.zeros_like(l_len)

    num = w_len * l_len.sum() + w_ang * l_ang.sum() + w_omg * l_omg.sum()
    loss = num / denom.clamp_min(eps)
    return loss, denom

def peptide_bond_velocity_constraint(
    x_c: torch.Tensor,       # (B,N,3) CA positions (global)
    R_c: torch.Tensor,       # (B,N,3,3) residue frames at conditioning time (local->global)
    a14_c: torch.Tensor,     # (B,N,14,3) atom14 positions at conditioning frame (global)
    a14m_c: torch.Tensor,    # (B,N,14)
    res_mask: torch.Tensor,  # (B,N)
    adjacent: torch.Tensor,  # (B,N-1) or (N-1)
    v_phys: torch.Tensor,    # (B,N,3) translational velocity in **local/body** frame
    w_phys: torch.Tensor,    # (B,N,3) angular velocity in **local/body** frame
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Velocity-space constraint enforcing d/dt ||C(i)-N(i+1)|| = 0.

    IMPORTANT FRAME CONVENTION:
      - This model defines v_phys and w_phys in the **residue local/body frame**.
      - Atom positions (a14_c) and origins (x_c) are in the **global/world frame**.
      - Therefore we must rotate velocities into the global frame using R_c before forming rigid point velocities.

    Constraint:
      Let r = C_i - N_{i+1}. For constant bond length, we require:
        d/dt ||r|| = (r · (v_C - v_N)) / ||r|| = 0
      We penalize (r · Δv)^2 / (||r||^2 + eps).
    """
    # Promote to FP32 for stability
    x  = x_c.to(torch.float32)
    R  = R_c.to(torch.float32)
    P  = a14_c.to(torch.float32)
    M  = a14m_c.to(torch.float32)
    rm = res_mask.to(torch.float32)
    vL = v_phys.to(torch.float32)
    wL = w_phys.to(torch.float32)

    if vL.dim() != 3 or wL.dim() != 3:
        raise RuntimeError(f"[PEPVEL] expected v_phys/w_phys shape (B,N,3), got v={tuple(vL.shape)} w={tuple(wL.shape)}")
    if R.dim() != 4 or R.shape[-2:] != (3,3):
        raise RuntimeError(f"[PEPVEL] expected R_c shape (B,N,3,3), got {tuple(R.shape)}")

    B, N = x.shape[:2]
    if R.shape[0] != B or R.shape[1] != N:
        raise RuntimeError(f"[PEPVEL] R_c shape mismatch: x_c {(B,N)} vs R_c {tuple(R.shape[:2])}")

    if adjacent.dim() == 1:
        adj = adjacent.unsqueeze(0).expand(B, -1).to(torch.float32)
    else:
        adj = adjacent.to(torch.float32)

    IDX_N, IDX_C = 0, 2

    # Bond endpoints in global frame
    C_i = P[:, :-1, IDX_C, :]        # (B,N-1,3)
    N_j = P[:,  1:, IDX_N, :]        # (B,N-1,3)
    x_i = x[:, :-1, :]               # (B,N-1,3) origin (CA) for residue i
    x_j = x[:,  1:, :]               # (B,N-1,3) origin (CA) for residue i+1

    # Pair gating: residue mask, adjacency, atom existence
    m_pair = (rm[:, :-1] * rm[:, 1:] * adj).clamp(0.0, 1.0)
    mC = M[:, :-1, IDX_C]
    mN = M[:,  1:, IDX_N]
    m_pair = m_pair * mC * mN

    denom = m_pair.sum().clamp_min(0.0)
    if denom.item() <= 0:
        z = torch.zeros((), device=x.device, dtype=torch.float32)
        return z, z

    # Local velocities for residues i and i+1
    v_iL = vL[:, :-1, :]
    v_jL = vL[:,  1:, :]
    w_iL = wL[:, :-1, :]
    w_jL = wL[:,  1:, :]

    # Rotate local->global using conditioning frames
    R_i = R[:, :-1, :, :]            # (B,N-1,3,3)
    R_j = R[:,  1:, :, :]            # (B,N-1,3,3)

    v_iG = (R_i @ v_iL.unsqueeze(-1)).squeeze(-1)
    v_jG = (R_j @ v_jL.unsqueeze(-1)).squeeze(-1)
    w_iG = (R_i @ w_iL.unsqueeze(-1)).squeeze(-1)
    w_jG = (R_j @ w_jL.unsqueeze(-1)).squeeze(-1)

    # Rigid point velocities in global frame
    vC = v_iG + torch.cross(w_iG, (C_i - x_i), dim=-1)
    vN = v_jG + torch.cross(w_jG, (N_j - x_j), dim=-1)

    r  = C_i - N_j
    r2 = torch.sum(r * r, dim=-1) + eps
    c  = torch.sum(r * (vC - vN), dim=-1)

    l = (c * c) / r2
    loss = (l * m_pair).sum() / denom.clamp_min(eps)
    return loss, denom

# %% cell 29
#----------------Helpers----------------------
def grad_clip_for_step(cfg, step: int) -> float:
    sched = cfg.get("GRAD_CLIP_SCHEDULE", None)
    if not sched:
        return float(cfg.get("GRAD_CLIP", 0.0))
    for s in sched:
        if step <= int(s.get("until_step", s.get("until_epoch", 0))):
            return float(s["max_norm"])
    return float(sched[-1]["max_norm"])

# %% cell 30
def grads_are_finite(model):
    for p in model.parameters():
        if p.grad is not None and (not torch.isfinite(p.grad).all()):
            return False
    return True

def assert_finite(t: torch.Tensor, name: str, fail: bool = True):
    if not torch.is_tensor(t):
        return
    ok = torch.isfinite(t).all().item()
    if not ok:
        msg = f"[NONFINITE] {name}: found NaN/Inf | shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
        if fail:
            raise FloatingPointError(msg)
        else:
            print(msg)

# %% cell 31
import torch

def _iter_tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _iter_tensors(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _iter_tensors(v)

def _tensor_summary(t: torch.Tensor):
    tt = t.detach()
    if tt.is_cuda:
        tt = tt.float()
    finite = torch.isfinite(tt)
    if finite.all():
        return f"shape={tuple(t.shape)} dtype={t.dtype} min={tt.min().item():.3e} max={tt.max().item():.3e} mean={tt.mean().item():.3e}"
    idx = (~finite).nonzero(as_tuple=False)
    # show first few bad indices
    show = idx[:5].tolist()
    return f"NON-FINITE! shape={tuple(t.shape)} dtype={t.dtype} bad_idx[:5]={show}"

class NanInfTracker:
    def __init__(self, enabled=True, raise_on_first=True, max_reports=5):
        self.enabled = enabled
        self.raise_on_first = raise_on_first
        self.max_reports = max_reports
        self._reports = 0
        self._seen = 0

    def reset(self):
        self.hit = False
        self.last_msg = None

    def _report(self, where: str, name: str, t: torch.Tensor):
        if self._reports >= self.max_reports:
            return
        self._reports += 1
        msg = f"[NAN-TRACK] {where} :: {name} :: {_tensor_summary(t)}"
        self.hit = True
        self.last_msg = msg
        print(msg, flush=True)
        if self.raise_on_first:
            raise RuntimeError(msg)

    def make_forward_hook(self, mod_name: str):
      def hook(module, inputs, output):
          if not self.enabled:
              return

          # sanity: confirm at least one hook is executing
          if self._seen < 1:
              print(f"[NAN-TRACK] forward hook active (example): {mod_name}", flush=True)
              self._seen += 1

          # Forward check (NaN/Inf in forward outputs)
          for t in _iter_tensors(output):
              if torch.is_tensor(t) and t.is_floating_point():
                  if not torch.isfinite(t.detach()).all():
                      self._report("FWD_OUT", mod_name, t)

          # Backward check (NaN/Inf in gradients of this module's outputs)
          self.attach_grad_hook(mod_name, output)

      return hook

    def attach_grad_hook(self, mod_name: str, output):
        if not self.enabled:
            return
        # Attach gradient hooks to module outputs (common way to localize BmmBackward NaNs)
        for t in _iter_tensors(output):
            if torch.is_tensor(t) and t.requires_grad and t.is_floating_point():
                def _g_hook(grad, n=mod_name):
                    if grad is not None and not torch.isfinite(grad.detach()).all():
                        self._report("BWD_GRAD", n, grad)
                    return grad
                t.register_hook(_g_hook)

# %% cell 32
import torch

def grad_norm_from_loss(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = 0.0
    for g in grads:
        if g is None:
            continue
        sq += (g.float().norm(2) ** 2)
    return float(torch.sqrt(sq + 1e-12))

@torch.no_grad()
def tensor_stats(name, x):
    x = x.detach()
    return {
        "name": name,
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "min": float(x.min().float()),
        "max": float(x.max().float()),
        "mean": float(x.mean().float()),
        "std": float(x.std().float()),
        "finite": bool(torch.isfinite(x).all()),
    }

def probe_loss_terms(loss_dict, model):
    # loss_dict: {"fm": l_fm, "end_pos": l_end_pos, ...}  each is scalar tensor
    params = [p for p in model.parameters() if p.requires_grad]
    out = []
    for k, l in loss_dict.items():
        if l is None:
            continue
        out.append((k, float(l.detach()), grad_norm_from_loss(l, params)))
    out.sort(key=lambda x: x[2], reverse=True)
    return out

# %% cell 33
import torch

def _safe_scalar(x: torch.Tensor) -> torch.Tensor:
    # ensure scalar float32 tensor (autograd-friendly)
    if x is None:
        return None
    if not torch.is_tensor(x):
        return None
    if x.numel() == 0:
        return None
    return x.float().mean()

def _grad_l2_norm_from_grads(grads) -> float:
    # grads: iterable of tensors or None
    s = 0.0
    for g in grads:
        if g is None:
            continue
        # use float32 for stable norm accumulation
        gg = g.detach().float()
        s += float(gg.pow(2).sum().item())
    return (s ** 0.5)

def grad_norm_wrt_tensors(loss: torch.Tensor, tensors: dict) -> dict:
    """
    loss: scalar tensor
    tensors: dict{name: tensor} where tensor participates in the graph
    returns: dict{name: grad_l2_norm}
    """
    loss = _safe_scalar(loss)
    if loss is None:
        return {k: 0.0 for k in tensors.keys()}

    names = list(tensors.keys())
    tlist = [tensors[k] for k in names]

    grads = torch.autograd.grad(
        loss,
        tlist,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    return {n: _grad_l2_norm_from_grads([g]) for n, g in zip(names, grads)}

def grad_norm_wrt_params(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> float:
    """
    loss: scalar tensor
    params: parameter list (keep this small for speed)
    returns: L2 norm of grads wrt these params
    """
    loss = _safe_scalar(loss)
    if loss is None or len(params) == 0:
        return 0.0

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    return _grad_l2_norm_from_grads(grads)

# %% cell 34
# --------------------------------------------------------------------------------------
# 11) Trainer: velocity-space FM + full auxiliary losses + window temporal alignment/coherence
# --------------------------------------------------------------------------------------

class VelocityFMTrainer:
    def __init__(self, model: VelocityFlowIPADynamics, cfg: Dict[str, Any]):
        self.model = model
        self.cfg = cfg
        self.device = cfg["DEVICE"]

        base_lr = float(cfg["LR"])
        wd = float(cfg["WD"])

        # Separate parameters into groups
        head_params, backbone_params, other_params = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "head" in name:
                head_params.append(p)
            elif "blocks" in name or "stride_embed" in name:  # stride_embed at backbone LR
                backbone_params.append(p)
            else:
                other_params.append(p)

        self.opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": base_lr * float(cfg.get("IPA_LR_MULT", 5.0)), "weight_decay": wd},
            {"params": head_params,     "lr": base_lr * float(cfg.get("HEAD_LR_MULT", 0.1)), "weight_decay": wd},
            {"params": other_params,    "lr": base_lr * float(cfg.get("OTHER_LR_MULT", 0.1)), "weight_decay": wd},
        ], lr=base_lr, weight_decay=wd)


        # Step-level LR schedule: linear warmup + cosine decay over TOTAL_STEPS optimizer steps.
        # NOTE: main_train() must set cfg["TOTAL_STEPS"] = len(train_loader) * EPOCHS.
        warmup_steps = int(cfg.get("LR_WARMUP_STEPS", 0))
        self._batch_count = 0
        total_steps  = int(cfg.get("TOTAL_STEPS", cfg.get("_TOTAL_STEPS", 0)))

        if total_steps <= 0:
            # Fallback (avoids div-by-zero). Prefer setting TOTAL_STEPS in main_train().
            total_steps = int(cfg.get("EPOCHS", 1)) * 1

        accum = int(cfg.get("GRAD_ACCUM_STEPS", 1))
        total_steps = total_steps // accum   # optimizer steps, not batch steps

        min_lr = float(cfg.get("LR_MIN", 0.0))
        base_lr = float(cfg.get("LR", 1e-4))
        min_factor = (min_lr / base_lr) if (base_lr > 0.0) else 0.0

        def _lr_factor(step: int) -> float:
            step = int(step)
            # Warmup: 0 -> 1
            if warmup_steps > 0 and step < warmup_steps:
                return max(0.0, float(step) / float(max(1, warmup_steps)))

            # Cosine: 1 -> min_factor over remaining steps
            denom = float(max(1, total_steps - warmup_steps))
            prog = float(step - warmup_steps) / denom
            prog = min(max(prog, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))
            return max(min_factor, float(cos))

        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lr_lambda=[_lr_factor, _lr_factor, _lr_factor]  # same schedule shape, different base LRs
        )
        # ---------------------------------------------------------------------
        # Precision policy (BF16 autocast on A100) - centralized and auditable
        #   - Geometry/state tensors remain FP32
        #   - Neural network matmul/attention runs under BF16 autocast
        # ---------------------------------------------------------------------
        self.use_amp = bool(self.cfg.get("AMP_ENABLED", self.cfg.get("AMP", True))) and (self.device == "cuda")
        self.amp_dtype = torch.bfloat16  # force BF16 on Ampere/Hopper
        self.use_scaler = False
        from openfold.utils import rigid_utils as rigid_utils_mod
        self.rigid_utils = rigid_utils_mod
        self.builder = OpenFoldAtom14Builder(rc, rigid_utils, feats, device=cfg["DEVICE"])


        # Optional performance knobs on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        print(f"[PRECISION] autocast={self.use_amp} dtype={self.amp_dtype} scaler={self.use_scaler}")
        self.global_step = 0
        self._profile_printed = False

        self.ema = EMA(model, decay=float(cfg["EMA_DECAY"])) if bool(cfg.get("EMA", True)) else None

        self.allatom_loss = AllAtom14Loss(cfg).to(self.device)

        # diagnostics / debug (JSONL + optional TensorBoard + invariant checks)
        self.diag = TrainDiagnostics(cfg)
        # self.diag = None
        self.dbg = DebugCfg(
            enabled=bool(cfg.get("DEBUG", False)),
            strict=bool(cfg.get("DEBUG_STRICT", True)),
            every=int(cfg.get("DEBUG_EVERY", 50)),
            closure_every=int(cfg.get("DEBUG_CLOSURE_EVERY", 200)),
            roundtrip_every=int(cfg.get("DEBUG_ROUNDTRIP_EVERY", 200)),
            so3_every=int(cfg.get("DEBUG_SO3_EVERY", 2000)),
            grad_every=int(cfg.get("DEBUG_GRAD_EVERY", 200)),
            save_bad_batches=bool(cfg.get("DEBUG_SAVE_BAD_BATCHES", True)),
            bad_batch_dir=str(cfg.get("DEBUG_BAD_BATCH_DIR", "debug_bad_batches")),
            max_saved=int(cfg.get("DEBUG_MAX_SAVED", 20)),
        )

    def _dt_from_stride(self, stride: int) -> float:
        if self.cfg.get("DT_PHYS", None) is None:
            return float(stride)
        return float(stride) * float(self.cfg["DT_PHYS"])

    def _dt_eff_tensor(self, dt, B: int, W: int, device, dtype) -> torch.Tensor:
        """
        Convert dt (float | scalar tensor | (B,) | (W,) | (B,W)) into a (B,W) tensor.
        Required for IGSO3 sampling, which expects per-(B,W) time steps.
        """
        if torch.is_tensor(dt):
            dt_t = dt.to(device=device, dtype=dtype)
            if dt_t.ndim == 0:
                dt_eff = dt_t.view(1, 1).expand(B, W)
            elif dt_t.ndim == 1:
                if dt_t.shape[0] == B:
                    dt_eff = dt_t.view(B, 1).expand(B, W)
                elif dt_t.shape[0] == W:
                    dt_eff = dt_t.view(1, W).expand(B, W)
                else:
                    raise ValueError(f"dt shape {tuple(dt_t.shape)} incompatible with (B={B}, W={W})")
            elif dt_t.ndim == 2:
                if dt_t.shape == (B, W):
                    dt_eff = dt_t
                else:
                    dt_eff = dt_t.expand(B, W)
            else:
                raise ValueError(f"dt has too many dims: {dt_t.ndim}")
        else:
            dt_eff = torch.full((B, W), float(dt), device=device, dtype=dtype)

        return dt_eff.clamp_min(1e-6)


    def _compute_gt_velocities(
        self,
        x_c: torch.Tensor, R_c: torch.Tensor, tors_c: torch.Tensor, mask_c: torch.Tensor, torsm_c: torch.Tensor,
        x_w: torch.Tensor, R_w: torch.Tensor, tors_w: torch.Tensor, mask_w: torch.Tensor, torsm_w: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute GT velocities between [x_c,R_c,tors_c] and window frames [x_w,R_w,tors_w].
        Returns:
          v_gt, omg_gt, thdot_gt: (B,W,N,*)
          m_vel: (B,W,N) mask for v/omg (both frames exist)
          m_th:  (B,W,N,7) torsion-rate mask (both torsions exist)
        """
        B, W, N, _ = x_w.shape
        # previous states sequence: (B,W,N,...) is [cond, w[:-1]]
        x_prev = torch.cat([x_c[:, None], x_w[:, :-1]], dim=1)
        R_prev = torch.cat([R_c[:, None], R_w[:, :-1]], dim=1)
        tors_prev = torch.cat([tors_c[:, None], tors_w[:, :-1]], dim=1)

        m_prev = torch.cat([mask_c[:, None], mask_w[:, :-1]], dim=1)  # (B,W,N)
        m_vel = (m_prev * mask_w).clamp(0, 1)

        # local translation velocity: R_prev^T (x_next - x_prev) / dt
        dx = (x_w - x_prev) / dt  # global
        v_local = (R_prev.transpose(-1, -2) @ dx.unsqueeze(-1)).squeeze(-1)  # local

        # body angular velocity: log(R_prev^T R_next) / dt
        dR = R_prev.transpose(-1, -2) @ R_w
        omg = so3_log(dR) / dt

        # torsion rate (wrapped)
        dth = wrap_to_pi(tors_w - tors_prev) / dt
        tmask_prev = torch.cat([torsm_c[:, None], torsm_w[:, :-1]], dim=1)
        m_th = (tmask_prev * torsm_w).clamp(0, 1)

        return v_local, omg, dth, m_vel, m_th

    def _sample_base_noise(self, B: int, W: int, N: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rho = float(self.cfg.get("NOISE_AR1_RHO", 0.0))
        eps_v = sample_ar1_noise((B, W, N, 3), rho, device, dtype)
        eps_o = sample_ar1_noise((B, W, N, 3), rho, device, dtype)
        eps_t = sample_ar1_noise((B, W, N, 7), rho, device, dtype)
        v0 = eps_v * float(self.cfg["SIGMA_V"])
        o0 = eps_o * float(self.cfg["SIGMA_OMG"])
        t0 = eps_t * float(self.cfg["SIGMA_THDOT"])

        # Step-dependent scaling: later k can have higher uncertainty
        alpha = float(self.cfg.get("SIGMA_STEP_ALPHA", 0.0))
        if (alpha != 0.0) and (W > 1):
            k = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, W, 1, 1)
            scale = 1.0 + alpha * k
            v0 = v0 * scale
            o0 = o0 * scale
            t0 = t0 * scale
        return v0, o0, t0

    def _sample_flow_time(self, B: int, W: int, device, dtype) -> torch.Tensor:
        """Sample flow time s.

        If FLOW_S_PER_STEP=True: returns (B,W,1) with mild AR(1) correlation along W.
        Else: returns (B,1) shared across steps (classic rectified-flow/FM).
        """
        if not bool(self.cfg.get("FLOW_S_PER_STEP", False)) or W == 1:
            return torch.rand((B, 1), device=device, dtype=dtype)

        base = torch.rand((B, 1, 1), device=device, dtype=dtype)  # shared base
        sigma = float(self.cfg.get("FLOW_S_NOISE", 0.0))
        if sigma <= 0.0:
            # deterministic across k given base (still per-step shaped)
            return base.expand(B, W, 1).contiguous()

        rho = float(self.cfg.get("FLOW_S_RHO", 0.0))
        eps = torch.randn((B, W, 1), device=device, dtype=dtype)
        if rho <= 0.0:
            noise = eps
        else:
            noise = torch.zeros_like(eps)
            noise[:, 0] = eps[:, 0]
            scale = math.sqrt(max(1e-8, 1.0 - rho * rho))
            for k in range(1, W):
                noise[:, k] = rho * noise[:, k - 1] + scale * eps[:, k]

        s = (base.expand_as(noise) + sigma * noise).clamp(0.0, 1.0)
        return s


    from contextlib import contextmanager

    @contextmanager
    def autocast_ctx(self):
        """BF16 autocast context for neural network compute."""
        if self.use_amp and self.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                yield
        else:
            yield

    @contextmanager
    def fp32_ctx(self):
        """Disable autocast (force FP32) for geometry/reductions/float64 islands."""
        if self.device == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                yield
        else:
            yield

    def _ramp01(self, step_i: int, start: int, duration: int) -> float:
        """Clamp((step-start)/duration, 0, 1)."""
        if duration <= 0:
            return 1.0 if step_i >= start else 0.0
        x = (step_i - start) / float(duration)
        return float(max(0.0, min(1.0, x)))

    def _schedule_val_full(self) -> Dict[str, Any]:
        """Validation always uses full loss weights — gives a consistent metric across all epochs."""
        cfg = self.cfg
        # Get a fully-ramped schedule by passing a very large step number
        # This ensures r_a14=1, r_fape=1, r_of=1, r_disto=1 regardless of training progress
        return self._schedule(step_i=999999)

    def _schedule(self, step_i: int) -> Dict[str, Any]:
        """Per-step dynamic weights + sparsity knobs (de-overlapped objective)."""
        cfg = self.cfg

        # ── Static path (SCHED_ENABLED=False) ────────────────────────────────────
        if not bool(cfg.get("SCHED_ENABLED", True)):
            w_fm       = float(cfg.get("W_FM", 0.0))
            w_disto    = float(cfg.get("W_DISTO", 0.0))
            w_a14      = float(cfg.get("W_ATOM14_SUP", 0.0))
            w_of       = float(cfg.get("W_OF_VIOL", 0.0))
            w_dx       = float(cfg.get("W_TEMP_DX", 0.0))
            w_dR       = float(cfg.get("W_TEMP_DR", 0.0))
            w_dth      = float(cfg.get("W_TEMP_DTH", 0.0))
            w_end_pos  = float(cfg.get("W_END_POS", 0.0))
            w_end_rot  = float(cfg.get("W_END_ROT", 0.0))
            w_end_tor  = float(cfg.get("W_END_TOR", 0.0))
            w_end_a14  = float(cfg.get("W_END_A14", 0.0))
            w_end_of   = float(cfg.get("W_END_OF",  0.0))
            w_fape     = float(cfg.get("W_FAPE", 0.0))
            w_pepgeo   = float(cfg.get("W_PEPTIDE_GEO", 0.0))
            w_pepvel   = float(cfg.get("W_PEPTIDE_VEL", 0.0))
            w_pos      = float(cfg.get("W_POS", 0.0))
            w_rot      = float(cfg.get("W_ROT", 0.0))
            w_tor      = float(cfg.get("W_TOR", 0.0))
            w_backbone = float(cfg.get("W_BACKBONE_SUP", 0.0))  # static: full weight

            # ─── Dynamic normalization: balance all physical-space losses vs l_fm ──────
            # l_fm is in normalized velocity space (O(1)). Physical losses are in Å or rad.
            # Divide each weight by the matching std² so the balance is maintained
            # automatically when V_STD / OMG_STD / THDOT_STD are changed in CFG.
            # All W_* values in CFG are CANONICAL (calibrated at std=1.0).
            if bool(cfg.get("NORMALIZE_VELOCITIES", True)):
                _v2  = float(cfg.get("V_STD",     1.0)) ** 2
                _o2  = float(cfg.get("OMG_STD",   1.0)) ** 2
                _th2 = float(cfg.get("THDOT_STD", 1.0)) ** 2
                # Å-scale
                w_pos      /= _v2;  w_disto  /= _v2;  w_a14    /= _v2
                w_of       /= _v2;  w_fape   /= _v2;  w_pepgeo /= _v2
                w_pepvel   /= _v2;  w_dx     /= _v2;  w_backbone /= _v2
                w_end_pos  /= _v2;  w_end_a14 /= _v2; w_end_of /= _v2
                # rad rotation
                w_rot /= _o2;  w_dR /= _o2;  w_end_rot /= _o2
                # rad torsion
                w_tor /= _th2;  w_dth /= _th2;  w_end_tor /= _th2

            # ──────────────────────────────────────────────────────────────────────────
            return dict(
                w_fm=w_fm,
                w_disto=w_disto, w_a14=w_a14, w_of=w_of,
                w_dx=w_dx, w_dR=w_dR, w_dth=w_dth,
                w_end_pos=w_end_pos, w_end_rot=w_end_rot, w_end_tor=w_end_tor,
                w_end_a14=w_end_a14, w_end_of=w_end_of,
                w_fape=w_fape,
                w_pepgeo=w_pepgeo, w_pepvel=w_pepvel,
                r_pepgeo=1.0,
                w_pos=w_pos, w_rot=w_rot, w_tor=w_tor,
                w_backbone=w_backbone, r_backbone=1.0,
                disto_only_last=bool(cfg.get("DISTO_ONLY_LAST", False)),
                disto_stride=int(cfg.get("DISTO_STRIDE", 0) or 0),
                disto_sample_k=int(cfg.get("DISTO_SAMPLE_K", 2048)),
                geom_only_last=bool(cfg.get("GEOM_ONLY_LAST", False)),
                geom_stride=int(cfg.get("GEOM_STRIDE", 0) or 0),
                interp_prob=float(cfg.get("TASK_INTERP_PROB", 0.0)),
                r_disto=1.0, r_a14=1.0, r_of=1.0, r_pose=1.0,
                r_interp=1.0 if bool(cfg.get("USE_TASK_INTERP", True)) else 0.0,
            )

        # ── Ramped path ───────────────────────────────────────────────────────────
        warmup    = int(cfg.get("SCHED_WARMUP_STEPS", 2000))
        a14_ramp  = int(cfg.get("SCHED_A14_RAMP_STEPS", 10000))
        of_ramp   = int(cfg.get("SCHED_OF_RAMP_STEPS", a14_ramp))
        disto_ramp = int(cfg.get("SCHED_DISTO_RAMP_STEPS", 4000))

        r_disto = self._ramp01(step_i, warmup, disto_ramp)
        # ① l_a14 from step 0: ramp starts at 0, not at warmup.
        # The model always has the input structure x_c, so atom14 supervision
        # is meaningful from the very first step — prevents unphysical frame drift.
        r_a14   = self._ramp01(step_i, 0, a14_ramp)
        r_of    = self._ramp01(step_i, warmup, of_ramp)

        fape_ramp = cfg.get("SCHED_FAPE_RAMP_STEPS", None)
        fape_ramp = int(a14_ramp if (fape_ramp is None) else fape_ramp)
        r_fape = self._ramp01(step_i, warmup, fape_ramp)

        pose_start = int(cfg.get("SCHED_POSE_DECAY_START",
            warmup if cfg.get("SCHED_POSE_DECAY_START", None) is None else cfg["SCHED_POSE_DECAY_START"]))
        pose_decay = int(cfg.get("SCHED_POSE_DECAY_STEPS",
            a14_ramp if cfg.get("SCHED_POSE_DECAY_STEPS", None) is None else cfg["SCHED_POSE_DECAY_STEPS"]))

        r_pose = 1.0 - self._ramp01(step_i, pose_start, pose_decay)
        floor  = float(cfg.get("SCHED_POSE_FLOOR", 0.0))
        r_pose = float(floor + (1.0 - floor) * r_pose)

        # ── All ramped weights as locals (required for /= normalization below) ───
        w_fm    = float(cfg.get("W_FM",   0.0))
        w_pos   = float(cfg.get("W_POS",  0.0)) * r_pose
        w_rot   = float(cfg.get("W_ROT",  0.0)) * r_pose
        w_tor   = float(cfg.get("W_TOR",  0.0)) * r_pose
        w_fape  = float(cfg.get("W_FAPE", 0.0)) * r_fape
        w_disto = float(cfg.get("W_DISTO", 0.0)) * r_disto
        w_a14   = float(cfg.get("W_ATOM14_SUP", 0.0)) * r_a14
        w_of    = float(cfg.get("W_OF_VIOL", 0.0)) * (r_of * r_of)
        w_dx    = float(cfg.get("W_TEMP_DX", 0.0))
        w_dR    = float(cfg.get("W_TEMP_DR", 0.0))
        w_dth   = float(cfg.get("W_TEMP_DTH", 0.0))

        # disto sparsity schedule
        if step_i < warmup:
            disto_only_last = True
            disto_stride    = 0
        else:
            switch = int(cfg.get("SCHED_DISTO_STRIDE_SWITCH", 20000))
            disto_only_last = False
            disto_stride = (int(cfg.get("SCHED_DISTO_STRIDE_EARLY", 8))
                            if step_i < switch else
                            int(cfg.get("SCHED_DISTO_STRIDE_LATE", 4)))

        k_switch     = int(cfg.get("SCHED_DISTO_K_SWITCH", 20000))
        disto_sample_k = (int(cfg.get("SCHED_DISTO_SAMPLE_K_EARLY", 1024))
                          if step_i < k_switch else
                          int(cfg.get("SCHED_DISTO_SAMPLE_K_LATE", 2048)))

        # geom sparsity schedule
        mid_start  = int(cfg.get("SCHED_GEOM_MID_START", 0))
        late_start = int(cfg.get("SCHED_GEOM_LATE_START", 40000))
        if step_i < mid_start:
            geom_only_last = True
            geom_stride    = 0
        elif step_i < late_start:
            geom_only_last = False
            geom_stride    = int(cfg.get("SCHED_GEOM_STRIDE_MID", 4))
        else:
            geom_only_last = False
            geom_stride    = int(cfg.get("SCHED_GEOM_STRIDE_LATE", 2))

        # interpolation prob + endpoint loss ramps
        if bool(cfg.get("USE_TASK_INTERP", True)):
            interp_start = int(cfg.get("SCHED_INTERP_START", warmup))
            interp_ramp  = int(cfg.get("SCHED_INTERP_RAMP_STEPS", 5000))
            r_interp     = self._ramp01(step_i, interp_start, interp_ramp)
            interp_prob  = float(cfg.get("TASK_INTERP_PROB", 0.0)) * r_interp
        else:
            r_interp    = 0.0
            interp_prob = 0.0

        w_end_pos = float(cfg.get("W_END_POS", 0.0)) * r_interp
        w_end_rot = float(cfg.get("W_END_ROT", 0.0)) * r_interp
        w_end_tor = float(cfg.get("W_END_TOR", 0.0)) * r_interp
        w_end_a14 = float(cfg.get("W_END_A14", 0.0)) * r_interp
        w_end_of  = float(cfg.get("W_END_OF",  0.0)) * r_interp

        # ── Dedicated pepgeo ramp (independent of r_a14) ─────────────────────────
        # pepgeo previously hit full weight the instant warmup ended, causing
        # l_pepgeo to spike 30× and shock the model. Own ramp avoids this.
        pepgeo_ramp = int(cfg.get("SCHED_PEPGEO_RAMP_STEPS", a14_ramp))
        r_pepgeo    = self._ramp01(step_i, warmup, pepgeo_ramp)
        w_pepgeo    = float(cfg.get("W_PEPTIDE_GEO", 0.0)) * r_pepgeo   # own ramp
        w_pepvel    = float(cfg.get("W_PEPTIDE_VEL", 0.0)) * r_a14      # ramps with atom14
        # ─────────────────────────────────────────────────────────────────────────

        # ── ② Backbone supervision from step 0 ───────────────────────────────────
        # N, CA, C, O atoms are purely determined by IPA frame outputs (R_hat, x_hat).
        # Ramping from step 0 anchors frame integration from the very first step,
        # preventing the l_pepgeo=1500 explosion seen when frames drift unconstrained.
        backbone_ramp = int(cfg.get("SCHED_BACKBONE_RAMP_STEPS", 400))
        r_backbone    = self._ramp01(step_i, 0, backbone_ramp)   # starts at step 0
        w_backbone    = float(cfg.get("W_BACKBONE_SUP", 0.0)) * r_backbone
        # ─────────────────────────────────────────────────────────────────────────

        # ─── Dynamic normalization: balance all physical-space losses vs l_fm ──────
        # l_fm is in normalized velocity space (O(1)). Physical losses are in Å or rad.
        # Divide each weight by the matching std² so the balance is maintained
        # automatically when V_STD / OMG_STD / THDOT_STD are changed in CFG.
        # All W_* values in CFG are CANONICAL (calibrated at std=1.0).
        if bool(cfg.get("NORMALIZE_VELOCITIES", True)):
            _v2  = float(cfg.get("V_STD",     1.0)) ** 2
            _o2  = float(cfg.get("OMG_STD",   1.0)) ** 2
            _th2 = float(cfg.get("THDOT_STD", 1.0)) ** 2
            # Å-scale
            w_pos      /= _v2;  w_disto  /= _v2;  w_a14    /= _v2
            w_of       /= _v2;  w_fape   /= _v2;  w_pepgeo /= _v2
            w_pepvel   /= _v2;  w_dx     /= _v2;  w_backbone /= _v2
            w_end_pos  /= _v2;  w_end_a14 /= _v2; w_end_of /= _v2
            # rad rotation
            w_rot /= _o2;  w_dR /= _o2;  w_end_rot /= _o2
            # rad torsion
            w_tor /= _th2;  w_dth /= _th2;  w_end_tor /= _th2
        # ──────────────────────────────────────────────────────────────────────────
        return dict(
            w_fm=w_fm,
            w_disto=w_disto, w_a14=w_a14, w_of=w_of,
            w_dx=w_dx, w_dR=w_dR, w_dth=w_dth,
            w_end_pos=w_end_pos, w_end_rot=w_end_rot, w_end_tor=w_end_tor,
            w_end_a14=w_end_a14, w_end_of=w_end_of,
            w_fape=w_fape,
            w_pepgeo=w_pepgeo, w_pepvel=w_pepvel,
            r_pepgeo=r_pepgeo,
            r_fape=r_fape,
            w_pos=w_pos, w_rot=w_rot, w_tor=w_tor,
            r_pose=r_pose,
            w_backbone=w_backbone,   # ② backbone supervision from step 0
            r_backbone=r_backbone,   # expose for logging
            disto_only_last=disto_only_last,
            disto_stride=disto_stride,
            disto_sample_k=disto_sample_k,
            geom_only_last=geom_only_last,
            geom_stride=geom_stride,
            interp_prob=interp_prob,
            r_disto=r_disto, r_a14=r_a14, r_of=r_of, r_interp=r_interp,
        )

    def _select_steps(self, W: int, kind: str, only_last: bool = None, stride: int = None) -> List[int]:
        """Select a subset of step indices (0-based) to compute expensive losses."""
        cfg = self.cfg
        kind = kind.lower()

        if W <= 0:
            return []

        if kind in ("disto", "distogram"):
            if bool(cfg.get("DISTO_ONLY_LAST", False)) if only_last is None else bool(only_last):
                return [W - 1]
            stride = int(cfg.get("DISTO_STRIDE", 0) or 0) if stride is None else int(stride)
            if stride <= 1:
                return list(range(W))
            idx = list(range(stride - 1, W, stride))
            if (W - 1) not in idx:
                idx.append(W - 1)
            return idx

        if kind in ("geom", "geometry", "atom14"):
            if bool(cfg.get("GEOM_ONLY_LAST", False)) if only_last is None else bool(only_last):
                return [W - 1]
            stride = int(cfg.get("GEOM_STRIDE", 0) or 0) if stride is None else int(stride)
            if stride <= 1:
                return list(range(W))
            idx = list(range(stride - 1, W, stride))
            if (W - 1) not in idx:
                idx.append(W - 1)
            return idx

        return list(range(W))


    def step(self, batch: Dict[str, torch.Tensor], train: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        device = self.device
        self.allatom_loss.global_step = int(self.global_step)

        do_checks = bool(cfg.get("CHECKS_ENABLED", False)) and (self.global_step % int(cfg.get("CHECK_EVERY", 1)) == 0)

        if train:
            sched = self._schedule(self.global_step)
        else:
            sched = self._schedule_val_full()

        w_pepgeo = float(sched.get("w_pepgeo", 0.0))
        w_pepvel = float(sched.get("w_pepvel", 0.0))
        fail_nonfinite = bool(cfg.get("FAIL_ON_NONFINITE", True))

        diag = getattr(self, "diag", None)
        dbg = getattr(self, "dbg", None)
        step_i = int(self.global_step)
        profile = bool(cfg.get("PROFILE_ONE_BATCH", False)) and (train is True) and (not self._profile_printed)
        if profile:
            import time
            if device == "cuda":
                torch.cuda.synchronize()
            t_step0 = time.perf_counter()

        aatype = batch["aatype"].to(device, non_blocking=True)
        esm = batch["esm"].to(device, non_blocking=True)

        x_c = batch["x_c"].to(device, non_blocking=True)
        R_c = batch["R_c"].to(device, non_blocking=True)
        tors_c = batch["tors_c"].to(device, non_blocking=True)
        mask_c = batch["mask_c"].to(device, non_blocking=True)
        torsm_c = batch["torsm_c"].to(device, non_blocking=True)
        a14_c = batch["a14_c"].to(device, non_blocking=True)
        a14m_c = batch["a14m_c"].to(device, non_blocking=True)
        adjacent = batch.get("adjacent", None)
        residue_index = batch.get("residue_index", None)
        if residue_index is not None:
            residue_index = residue_index.to(device, non_blocking=True).long()
        else:
            residue_index = None

        x_w = batch["x_w"].to(device, non_blocking=True)
        R_w = batch["R_w"].to(device, non_blocking=True)
        tors_w = batch["tors_w"].to(device, non_blocking=True)
        mask_w = batch["mask_w"].to(device, non_blocking=True)
        torsm_w = batch["torsm_w"].to(device, non_blocking=True)

        a14_w = batch["a14_w"].to(device, non_blocking=True)
        a14m_w = batch["a14m_w"].to(device, non_blocking=True)

        # --------------------------------------------------------------------------------------
        # DTYPE POLICY ENFORCEMENT (device boundary): geometry/state in FP32, ids in int64
        # --------------------------------------------------------------------------------------
        aatype = aatype.long()
        esm = esm.float()

        x_c = x_c.float()
        R_c = R_c.float()
        tors_c = tors_c.float()
        tors_c = torch.where(torch.isfinite(tors_c), tors_c, torch.zeros_like(tors_c))
        mask_c = mask_c.float()
        torsm_c = torsm_c.float() if torch.is_tensor(torsm_c) else torsm_c

        x_w = x_w.float()
        R_w = R_w.float()
        tors_w = tors_w.float()
        tors_w = torch.where(torch.isfinite(tors_w), tors_w, torch.zeros_like(tors_w))
        mask_w = mask_w.float()
        torsm_w = torsm_w.float() if torch.is_tensor(torsm_w) else torsm_w

        a14_w = a14_w.float()
        a14m_w = a14m_w.float()
        a14_c = a14_c.float()
        a14m_c = a14m_c.float()
        if adjacent is None:
            adjacent = torch.ones((aatype.shape[0], max(0, int(aatype.shape[1]) - 1)), device=device, dtype=torch.float32)
        else:
            adjacent = adjacent.to(device, non_blocking=True).float()

        def _pick_t(x, t):
          """
          Accepts either per-step tensors or windowed tensors.
          Returns a per-step tensor.
          """
          if not torch.is_tensor(x):
              return x
          # Common windowed layouts:
          # (B,W,N,3), (B,W,N,3,3), (B,W,N,14,3), (B,W,N,14), (B,W,N,7)...
          if x.dim() >= 3 and x.shape[1] == self.cfg.get("W", x.shape[1]):
              return x[:, t]
          return x  # already per-step


        # -----------------------------
        # DEBUG: GT closure / bond checks
        # -----------------------------
        debug_every = int(self.cfg.get("DBG_CLOSURE_EVERY", 1))
        debug_first = int(self.cfg.get("DBG_CLOSURE_FIRST", 1))

        do_dbg = (self.global_step < debug_first) or (self.global_step % debug_every == 0)
        if do_dbg:
            # rotation_quality_stats(R_w, mask_w, name="R_w[t]_valid")
            # with torch.cuda.amp.autocast(enabled=False):
            #     rotation_det_stats(R_w, name="R_w[t]")
            # Choose a deterministic mid-window time index
            # (If tensors are not windowed, _pick_t will just return them as-is.)
            if torch.is_tensor(x_w) and x_w.dim() >= 4:
                W = x_w.shape[1]
                t = W // 2
            else:
                W = 1
                t = 0

            # Per-step tensors (B,N,...) for the closure
            R_gt   = _pick_t(R_w,    t)      # (B,N,3,3)
            x_gt   = _pick_t(x_w,    t)      # (B,N,3)
            tors_gt= _pick_t(tors_w, t)      # (B,N,...)  per-step tors
            res_m  = _pick_t(mask_w, t)      # (B,N)

            gt_a14 = _pick_t(a14_w,  t)      # (B,N,14,3)
            gt_a14m= _pick_t(a14m_w, t)      # (B,N,14)
            gt_mask_c= _pick_t(mask_c, t)      # (B,N,14)
            gt_mask_w= _pick_t(mask_w, t)      # (B,N,14)
            gt_torsm_c= _pick_t(torsm_c, t)      # (B,N,14)

            # IMPORTANT: do not let autocast/bf16 perturb debug geometry
            with torch.cuda.amp.autocast(enabled=False):
                tag = f" step={self.global_step} t={t}/{W}"

                # (1) quick bond sanity (especially C-O)
                # gt_backbone_bond_checks(gt_a14, gt_a14m, res_m, tag=tag)

                # (2) closure check
                # Pick the module that actually owns .builder
                # In your run you used self.allatom_loss successfully.
                atom14_mod = getattr(self, "allatom_loss", None)
                if atom14_mod is None:
                    atom14_mod = getattr(self, "loss_a14", None)  # fallback if your trainer uses this name

                # rigid utils module: you must pass the SAME rigid_utils class family used by FoldFlow/OpenFold
                # Many implementations keep it on the trainer; otherwise import it where you build the builder.
                ru = getattr(self, "rigid_utils", None)
                if ru is None:
                    ru = self.model.rigid_utils if hasattr(self.model, "rigid_utils") else None

                # if ru is None:
                #     print(f"[GT-A14-CLOSURE]{tag} SKIP: rigid_utils not found on trainer/model")
                # else:
                #     debug_atom14_closure(self.builder, aatype, R_gt, x_gt, tors_gt, torsm_c, gt_a14, gt_a14m, gt_mask_c, gt_mask_w,
                #          tag=f"[GT-A14-CLOSURE] t={t} AS-IS tors")
                #     tors_re = reorder_torsions_to_openfold(tors_gt)
                #     torsm_re = reorder_torsions_to_openfold(gt_torsm_c) if gt_torsm_c is not None else None
                #     debug_atom14_closure(self.builder, aatype, R_gt, x_gt, tors_re, torsm_re, gt_a14, gt_a14m, gt_mask_c, gt_mask_w,
                #                         tag=f"[GT-A14-CLOSURE] t={t} REORDER tors (omega,phi,psi)")

        # --------------------------------------------------------------------------------------
        # PADDED INPUT SANITIZATION  (prevents NaNs inside IPA / rigid math)
        # --------------------------------------------------------------------------------------
        if bool(self.cfg.get("SANITIZE_PADDED_INPUTS", True)):
            # --- conditioning frame (B,N,...) ---
            mc = (mask_c > 0.5).to(dtype=x_c.dtype).unsqueeze(-1)        # (B,N,1)
            mR_c = (mask_c > 0.5).to(dtype=R_c.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,N,1,1)

            # padded coords -> 0
            x_c = x_c * mc

            # padded torsions -> 0 (use torsm_c if available, else use mask_c)
            if torsm_c is not None:
                # ensure broadcast (B,N,...) -> (B,N,1,1,...) not needed; rely on broadcasting
                tors_c = tors_c * torsm_c.to(dtype=tors_c.dtype)
            else:
                tors_c = tors_c * mc

            # padded rotations -> identity
            I_c = torch.eye(3, device=R_c.device, dtype=R_c.dtype).view(1, 1, 3, 3)  # (1,1,3,3)
            R_c = R_c * mR_c + I_c * (1.0 - mR_c)

            # --- window frame (B,W,N,...) ---
            mw = (mask_w > 0.5).to(dtype=x_w.dtype).unsqueeze(-1)        # (B,W,N,1)
            mR_w = (mask_w > 0.5).to(dtype=R_w.dtype).unsqueeze(-1).unsqueeze(-1)   # (B,W,N,1,1)

            # padded coords -> 0
            x_w = x_w * mw

            # padded torsions -> 0 (use torsm_w if available, else use mask_w)
            if torsm_w is not None:
                tors_w = tors_w * torsm_w.to(dtype=tors_w.dtype)
            else:
                tors_w = tors_w * mw

            # padded rotations -> identity
            I_w = torch.eye(3, device=R_w.device, dtype=R_w.dtype).view(1, 1, 1, 3, 3)  # (1,1,1,3,3)
            R_w = R_w * mR_w + I_w * (1.0 - mR_w)

        if do_checks:
            # Shape + finite checks
            for name, t in [
                ("x_c", x_c), ("R_c", R_c), ("tors_c", tors_c), ("mask_c", mask_c), ("m_c", torsm_c),
                ("x_w", x_w), ("R_w", R_w), ("tors_w", tors_w), ("mask_w", mask_w), ("m_w", torsm_w),
                ("aatype", aatype), ("esm", esm),
                ("a14m_w", a14m_w),
            ]:
                if torch.is_tensor(t):
                    assert_finite(t, name, fail=fail_nonfinite)

            # Dtype contract checks (catch BF16/FP16/FP64 leaks into geometry lane)
            for name, t in [("x_c", x_c), ("R_c", R_c), ("tors_c", tors_c), ("x_w", x_w), ("R_w", R_w), ("tors_w", tors_w)]:
                if torch.is_tensor(t) and (t.dtype != torch.float32):
                    raise RuntimeError(f"[DTYPE] {name} must be fp32, got {t.dtype}")
            if aatype.dtype != torch.long:
                raise RuntimeError(f"[DTYPE] aatype must be int64/long, got {aatype.dtype}")
            if esm.dtype not in (torch.float32, torch.bfloat16):
                raise RuntimeError(f"[DTYPE] esm unexpected dtype {esm.dtype}")

        # ------------------------------------------------------------
        # Enforce valid SO(3) rotations for GT/input (prevents log-map / IPA / atom builder issues)
        # ------------------------------------------------------------
        if bool(cfg.get("ORTHO_INPUT_R", True)):
            use_fp32 = bool(cfg.get("ORTHO_R_FP32", True))
            with self.fp32_ctx():
                Rc_in = R_c.float() if use_fp32 else R_c
                Rw_in = R_w.float() if use_fp32 else R_w

                Rc_proj = SO3.orthonormalize_safe(Rc_in)
                Rw_proj = SO3.orthonormalize_safe(Rw_in)

                # cast back to original dtype (typically fp16/bf16 under AMP)
                R_c = Rc_proj  # keep FP32
                R_w = Rw_proj  # keep FP32

            if bool(cfg.get("ORTHO_DEBUG", False)) and (self.global_step == 0) and train:
                # one-time diagnostics (FP32)
                with torch.no_grad():
                    Rc = R_c.float()
                    Rw = R_w.float()
                    I = torch.eye(3, device=Rc.device, dtype=Rc.dtype)

                    err_c = (Rc.transpose(-1, -2) @ Rc - I).pow(2).sum(dim=(-2, -1)).sqrt().max().item()
                    err_w = (Rw.transpose(-1, -2) @ Rw - I).pow(2).sum(dim=(-2, -1)).sqrt().max().item()

                    det_c = torch.linalg.det(Rc).min().item()
                    det_w = torch.linalg.det(Rw).min().item()

                print(f"[ORTHO] max||R^T R - I||_F: cond={err_c:.3e} win={err_w:.3e} | min det: cond={det_c:.6f} win={det_w:.6f}")

        if profile:
            if device == "cuda":
                torch.cuda.synchronize()
            t_to_device = time.perf_counter()

        _strides_in_batch = batch["stride"].reshape(-1)
        assert _strides_in_batch.eq(_strides_in_batch[0]).all(), (
            f"[BUG] Mixed-stride batch detected: {_strides_in_batch.tolist()}. "
            f"StrideGroupedBatchSampler contract violated."
        )
        stride = int(_strides_in_batch[0].item())
        dt = self._dt_from_stride(stride)

        B, W, N, _ = x_w.shape

        # -----------------------------
        # MDGen-inspired task mixing:
        #   - Forward task: condition on X_t only (your current default)
        #   - Interpolation task: condition on X_t and X_{t+W} (endpoint anchor)
        # We sample per-batch for simplicity (all samples in batch share the same task).
        use_interp = False
        if train and bool(cfg.get("USE_TASK_INTERP", True)):
            p_interp = float(sched.get("interp_prob", 0.0))
            use_interp = (torch.rand((), device=device) < p_interp).item()
        anchor_feat = None
        exclude_last = bool(cfg.get("EXCLUDE_LAST_FROM_WINDOW_LOSSES_IN_INTERP", True)) and bool(use_interp)
        if use_interp:
            # endpoint pose at the end of the window (k = W)
            x_end = x_w[:, -1]     # (B,N,3)
            R_end = R_w[:, -1]     # (B,N,3,3)
            tors_end = tors_w[:, -1]  # (B,N,7)

            # build anchor feature in the *conditioning* local frame
            # dx_local = R0^T (x_end - x0)
            dx = x_end - x_c
            dx_local = torch.matmul(R_c.transpose(-1, -2), dx.unsqueeze(-1)).squeeze(-1)

            # dR = log(R0^T R_end)
            dR = so3_log(torch.matmul(R_c.transpose(-1, -2), R_end))

            # dθ wrapped, represented as sin/cos
            dth = wrap_to_pi(tors_end - tors_c)
            dth_sc = torch.cat([torch.sin(dth), torch.cos(dth)], dim=-1)  # (B,N,14)

            anchor_feat = torch.cat([dx_local, dR, dth_sc], dim=-1)  # (B,N,20)


        # Compute GT velocities in physical units
        v1, o1, t1, m_vel, m_th = self._compute_gt_velocities(
            x_c, R_c, tors_c, mask_c, torsm_c,
            x_w, R_w, tors_w, mask_w, torsm_w,
            dt=dt
        )


        # Normalize velocities for FM stability
        v1n, o1n, t1n = normalize_velocities(v1, o1, t1, cfg, stride=stride)

        # Sample base noise y0 in same (normalized) space
        v0, o0, t0 = self._sample_base_noise(B, W, N, device, v1n.dtype)

        # Optional: sample rotational base-noise from the IGSO(3) heat kernel (better than Euclidean Gaussian in rotvec).
        # NOTE: fail loudly by default if USE_IGSO3=True but dt / shapes / helpers are wrong (avoid silent no-op).
        if bool(cfg.get("USE_IGSO3", False)) and (FF.igso3 is not None) and (FF.so3_helpers is not None):
          try:
              dt_eff = self._dt_eff_tensor(dt, B, W, device, v1n.dtype)  # (B,W)

              # If you normalize omega by OMG_STD, then sigma_omg is in *normalized* units.
              # To sample SO(3) increments (rotvec in radians), convert sigma to physical:
              if bool(cfg.get("NORMALIZE_VELOCITIES", True)):
                  _omg_by_s = cfg.get("OMG_STD_BY_STRIDE", {})
                  omg_std = float(_omg_by_s.get(stride, cfg.get("OMG_STD", 1.0)))
              else:
                  omg_std = 1.0

              sigma_omg = float(cfg["SIGMA_OMG"])  # normalized omega std (dimensionless)
              alpha = float(cfg.get("SIGMA_STEP_ALPHA", 0.0))

              # Step-dependent sigma schedule (normalized space)
              if (alpha != 0.0) and (W > 1):
                  k = torch.linspace(0.0, 1.0, W, device=device, dtype=v1n.dtype).view(1, W)  # (1,W)
                  sigma_eff_norm = sigma_omg * (1.0 + alpha * k)  # (1,W)
              else:
                  sigma_eff_norm = torch.full((1, W), sigma_omg, device=device, dtype=v1n.dtype)  # (1,W)

              # Convert to physical omega std (rad/time), then to SO(3) increment variance:
              sigma_eff_phys = sigma_eff_norm * omg_std  # (1,W), physical omega std
              eps = (sigma_eff_phys.expand(B, W) * dt_eff) ** 2
              eps = eps * float(cfg.get("IGSO3_EPS_FACTOR", 1.0))  # (B,W)

              # Sample one SO(3) perturbation per (B,W,N)
              ig_dtype = torch.float64
              eps_vec = eps.unsqueeze(-1).expand(B, W, N).reshape(-1).to(dtype=ig_dtype)  # (B*W*N,)
              mu = torch.eye(3, device=device, dtype=ig_dtype).expand(eps_vec.shape[0], 3, 3).contiguous()

              R_noise = FF.igso3._batch_sample(mu, eps_vec, 1)  # (B*W*N,3,3)
              rotvec = FF.so3_helpers.rotmat_to_rotvec(R_noise).reshape(B, W, N, 3).to(dtype=v1n.dtype)  # radians

              # Optional: impose AR(1) temporal correlation in rotvec space (approximate)
              if bool(cfg.get("IGSO3_USE_AR1", True)):
                  rho = float(cfg.get("NOISE_AR1_RHO", 0.0))
                  if (rho != 0.0) and (W > 1):
                      coeff = math.sqrt(max(0.0, 1.0 - rho * rho))
                      for kk in range(1, W):
                          rotvec[:, kk] = rho * rotvec[:, kk - 1] + coeff * rotvec[:, kk]

              # Convert SO(3) increment to omega noise in *normalized* space:
              # omega_phys = rotvec / dt ; omega_norm = omega_phys / OMG_STD
              o0 = rotvec / (dt_eff[:, :, None, None] * omg_std)  # (B,W,N,3), normalized

          except Exception as e:
              if bool(cfg.get("IGSO3_STRICT", True)):
                  raise RuntimeError(
                      f"IGSO3 sampling failed (check dt type/shape and FoldFlow igso3/so3_helpers). Error: {repr(e)}"
                  ) from e
              if not getattr(self, "_warned_igso3", False):
                  print(f"[WARN] IGSO3 sampling failed; falling back to Gaussian omega noise. Error: {repr(e)}")
                  self._warned_igso3 = True

        # Flow-time s ~ U(0,1)
        # Optionally sample per-step s_k to reflect increasing difficulty with horizon.
        s = self._sample_flow_time(B, W, device, v1n.dtype)
        if s.ndim == 2:  # (B,1)
            sW = s[:, None, None, :]  # (B,1,1,1)
        else:            # (B,W,1)
            sW = s[:, :, None, :]     # (B,W,1,1)

        # Interpolate y_s and target u = y1 - y0
        v_s = (1 - sW) * v0 + sW * v1n
        o_s = (1 - sW) * o0 + sW * o1n
        t_s = (1 - sW) * t0 + sW * t1n

        u_v = (v1n - v0)
        u_o = (o1n - o0)
        u_t = (t1n - t0)

        # Integrate y_s to get per-step previous states (noisy pose conditioning)
        v_s_phys, o_s_phys, t_s_phys = denormalize_velocities(v_s, o_s, t_s, cfg, stride=stride)
        if profile:
            if device == "cuda":
                torch.cuda.synchronize()
            t_noise_int0 = time.perf_counter()

        # ── Anchor-corrected Euler: safe defaults (always defined even if skipped) ──
        anchor_alpha_cur = 0.0   # overwritten below if ANCHOR_ALPHA > 0
        frame_drift_A    = 0.0   # mean per-residue ||x_hat - x_c|| after correction

        # ── Anchor-corrected Euler: compute current alpha from annealing schedule ──
        # alpha starts at ANCHOR_ALPHA and anneals to 0 over ANCHOR_DECAY_STEPS.
        # Once alpha=0 the integration is identical to pure Euler.
        _anchor_alpha_init  = float(cfg.get("ANCHOR_ALPHA", 0.0))
        _anchor_decay_steps = int(cfg.get("ANCHOR_DECAY_STEPS", 0))
        if _anchor_alpha_init > 0.0 and _anchor_decay_steps > 0:
            # Linear anneal: alpha = alpha_init * max(0, 1 - step/decay_steps)
            _anneal_frac = max(0.0, 1.0 - float(step_i) / float(_anchor_decay_steps))
            anchor_alpha_cur = _anchor_alpha_init * _anneal_frac
        else:
            anchor_alpha_cur = _anchor_alpha_init  # constant (or 0)

        # 1) Build the anchored "s-trajectory" next-states (x_{1..W}) consistently
        x_s_next, R_s_next, tors_s_next = integrate_velocities_to_window(
            x_c, R_c, tors_c,
            v_s_phys, o_s_phys, t_s_phys,
            dt=dt,
            anchor_alpha=anchor_alpha_cur,
        )

        # 2) Convert to per-step "previous" states aligned with step k velocities
        #    prev[k] corresponds to state at time k (k=0 is the conditioning frame)
        x_prev    = torch.cat([x_c[:, None],    x_s_next[:, :-1]],   dim=1)   # (B,W,N,3)
        R_prev    = torch.cat([R_c[:, None],    R_s_next[:, :-1]],   dim=1)   # (B,W,N,3,3)
        tors_prev = torch.cat([tors_c[:, None], tors_s_next[:, :-1]], dim=1)  # (B,W,N,7)

        # ── Anchor-correct x_prev so IPA rigids match x_hat coordinate system ──
        if anchor_alpha_cur > 0.0:
            x_prev_a = x_prev.clone()
            R_prev_a = R_prev.clone()
            for k in range(W):
                den = max(W - 1, 1)
                alpha_k = anchor_alpha_cur * (1.0 - float(k) / float(den))
                if alpha_k <= 0.0:
                    break
                x_prev_a[:, k] = (1.0 - alpha_k) * x_prev[:, k] + alpha_k * x_c
                dR = R_c.transpose(-1, -2) @ R_prev[:, k]
                log_dR = so3_log(dR.reshape(-1, 3, 3))
                R_prev_a[:, k] = (
                    R_c @ so3_exp(
                        ((1.0 - alpha_k) * log_dR).reshape(-1, 3)
                    ).reshape(B, N, 3, 3)
                )
            x_prev = x_prev_a
            R_prev = R_prev_a

        mask_prev = torch.cat([mask_c[:, None], mask_w[:, :-1]], dim=1).clamp(0, 1)
        if profile:
            if device == "cuda":
                torch.cuda.synchronize()
            t_noise_int1 = time.perf_counter()

        # physical horizon offsets (used by delta_t embedding)
        delta_t = (torch.arange(1, W + 1, device=device, dtype=v1n.dtype) * float(dt))
        # -------------------------------------------------------------------------
        # BF16 region: model forward only (no geometry, no reductions, no SVD)
        # -------------------------------------------------------------------------
        # Precompute pair features (optional) under autocast for performance.
        z_pair = None
        if bool(cfg.get("USE_SHARED_PAIR_Z", True)):
            src = str(cfg.get("PAIR_Z_SOURCE", "cond")).lower()
            if src.startswith("step"):
                x_pair = x_prev[:, 0]  # (B,N,3)
            else:
                x_pair = x_c  # (B,N,3)
            with self.autocast_ctx():
                z_pair = self.model.pair_embed(x_pair, detach_x=bool(cfg.get("DETACH_PAIR_X", True))).unsqueeze(1)

        # Model predicts u_hat at (s, y_s)
        with self.autocast_ctx():
            if profile and device == "cuda":
                torch.cuda.synchronize()
            if profile:
                t_fwd0 = time.perf_counter()

            uhat_v, uhat_o, uhat_t, aux_outputs = self.model(
                aatype=aatype,
                esm=esm,
                x_prev=x_prev,
                R_prev=R_prev,
                tors_prev=tors_prev,
                mask_prev=mask_prev,
                v_s=v_s,
                omg_s=o_s,
                thdot_s=t_s,
                flow_s=s,
                delta_t=delta_t,
                delta_idx=None,
                anchor_feat=anchor_feat,
                z_pair=z_pair,
                stride=stride,
            )

            if profile and device == "cuda":
                torch.cuda.synchronize()
            if profile:
                t_fwd1 = time.perf_counter()

        # -------------------------------------------------------------------------
        # FP32 region: FM loss reductions + geometry integration + all geometry losses
        # -------------------------------------------------------------------------
        with self.fp32_ctx():
            uhat_v_f = uhat_v.float()
            uhat_o_f = uhat_o.float()
            uhat_t_f = uhat_t.float()

            # FM loss in velocity space (FP32 reductions)
            l_fm_v = masked_mse(((uhat_v_f - u_v) ** 2).sum(dim=-1), m_vel)
            l_fm_o = masked_mse(((uhat_o_f - u_o) ** 2).sum(dim=-1), m_vel)
            l_fm_t = masked_mse(((uhat_t_f - u_t) ** 2), m_th)
            _fm_tor_w = float(cfg.get("FM_TORSION_WEIGHT", 1.0))
            l_fm = l_fm_v + l_fm_o + _fm_tor_w * l_fm_t

            # ── CHANGE 3: stride FM loss weight ──────────────────────────────────────
            if train:
                _l_fm_raw = l_fm.detach()
                _stride_fm_w = float(cfg.get("STRIDE_FM_WEIGHTS", {}).get(stride, 1.0))
                l_fm = l_fm * _stride_fm_w
            else:
                _l_fm_raw = l_fm.detach()   # val: unweighted (stride weight = 1.0)
                _stride_fm_w = 1.0           # val: no stride weighting applied
            # ─────────────────────────────────────────────────────────────────────────

            # Endpoint estimate y_hat1 (normalized) via straight-line correction
            v1_hat = v_s.detach() + (1 - sW) * uhat_v_f
            o1_hat = o_s.detach() + (1 - sW) * uhat_o_f
            t1_hat = t_s.detach() + (1 - sW) * uhat_t_f

            # Integrate predicted velocities (denorm to physical units) - FP32 geometry lane
            v1_hat_phys, o1_hat_phys, t1_hat_phys = denormalize_velocities(v1_hat, o1_hat, t1_hat, cfg, stride=stride)

            x_hat, R_hat, tors_hat = integrate_velocities_to_window(
                x_c, R_c, tors_c, v1_hat_phys, o1_hat_phys, t1_hat_phys,
                dt=dt,
                anchor_alpha=anchor_alpha_cur,
            )

            # frame_drift_A: mean per-residue ||x_hat - x_c|| AFTER anchor correction.
            # With pure Euler (anchor_alpha=0) this is the raw integration drift (~25 Å early).
            # With anchor active this shows the corrected drift — should be much smaller.
            # Watch this fall toward <5 Å as FM learns better velocities.
            with torch.no_grad():
                frame_drift_A = float((x_hat - x_c.unsqueeze(1)).norm(dim=-1).mean().item())

            # Optional: enforce valid SO(3) for predicted rotations (FP32)
            if bool(cfg.get("ORTHO_PRED_R", True)):
                R_hat = SO3.orthonormalize_safe_ste(R_hat.float())

            # Shared-window SE(3) alignment (Kabsch) to remove global gauge drift.
            if use_interp and bool(cfg.get("DISABLE_KABSCH_INTERP", True)):
                x_hat_a, R_hat_a = x_hat, R_hat
            else:
                P   = x_hat.reshape(B, W * N, 3).to(torch.float32)
                Qgt = x_w.reshape(B, W * N, 3).to(torch.float32)
                wgt = mask_w.reshape(B, W * N).to(torch.float32).clamp(0, 1)
                Qalign, talign = kabsch_align(P.detach(), Qgt, wgt)
                x_hat_a, R_hat_a = apply_global_se3(Qalign, talign, x_hat, R_hat)

            # ------------------------------
            # Cheap per-step pose losses
            # ------------------------------
            l_pos = torch.zeros((), device=device, dtype=torch.float32)
            l_rot = torch.zeros((), device=device, dtype=torch.float32)
            l_tor = torch.zeros((), device=device, dtype=torch.float32)

            w_pos = float(sched.get("w_pos", 0.0))
            w_rot = float(sched.get("w_rot", 0.0))
            w_tor = float(sched.get("w_tor", 0.0))

            # ── dt² weight correction for pose losses ─────────────────────────────
            # _schedule() normalises w_pos, w_rot, w_tor by V_STD[1]², OMG_STD[1]²,
            # THDOT_STD[1]² (stride-1 values). At stride>1 the predicted trajectory
            # spans dt× more time, so expected position/rotation/torsion errors are
            # naturally larger (same over-penalty as l_a14 / l_rmsf_match).
            # Correction: multiply by (std[1]² / (dt² × std[stride]²)).
            # stride=1 → correction = 1.0 (no change).
            # stride=2 → w_pos×0.32, w_rot×0.35, w_tor×0.39
            # stride=4 → w_pos×0.14, w_rot×0.16, w_tor×0.20
            # NOT corrected:
            #   l_pepgeo / l_of  — absolute chemistry (bond length is 1.329Å at any stride)
            #   l_dx / l_dR / l_dth — smoothness regularizers; kept strict at all strides
            if stride > 1 and bool(cfg.get("NORMALIZE_VELOCITIES", True)):
                _v_std_s  = float(cfg.get("V_STD_BY_STRIDE",    {}).get(stride, cfg.get("V_STD",     1.0)))
                _o_std_s  = float(cfg.get("OMG_STD_BY_STRIDE",  {}).get(stride, cfg.get("OMG_STD",   1.0)))
                _th_std_s = float(cfg.get("THDOT_STD_BY_STRIDE",{}).get(stride, cfg.get("THDOT_STD", 1.0)))
                _v_std_1  = float(cfg.get("V_STD",     1.0))
                _o_std_1  = float(cfg.get("OMG_STD",   1.0))
                _th_std_1 = float(cfg.get("THDOT_STD", 1.0))
                _dt2_v    = (dt ** 2) * (_v_std_s  / _v_std_1)  ** 2
                _dt2_o    = (dt ** 2) * (_o_std_s  / _o_std_1)  ** 2
                _dt2_th   = (dt ** 2) * (_th_std_s / _th_std_1) ** 2
                w_pos /= _dt2_v
                w_rot /= _dt2_o
                w_tor /= _dt2_th
            # ─────────────────────────────────────────────────────────────────────

            if (w_pos > 0.0) or (w_rot > 0.0) or (w_tor > 0.0):
                # Choose aligned vs raw prediction for pos/rot
                if bool(cfg.get("POSE_USE_ALIGNED", True)):
                    xP, RP = x_hat_a, R_hat_a
                else:
                    xP, RP = x_hat, R_hat

                # Optionally exclude last step in interpolation windows (consistent with your dx/dR/dth logic)
                if exclude_last:
                    xP, RP = xP[:, :-1], RP[:, :-1]
                    xG, RG = x_w[:, :-1], R_w[:, :-1]
                    mR = mask_w[:, :-1].clamp(0, 1)

                    tP, tG = tors_hat[:, :-1], tors_w[:, :-1]
                    tM = (torsm_w[:, :-1] if (torsm_w is not None) else mR.unsqueeze(-1))
                else:
                    xG, RG = x_w, R_w
                    mR = mask_w.clamp(0, 1)

                    tP, tG = tors_hat, tors_w
                    tM = (torsm_w if (torsm_w is not None) else mR.unsqueeze(-1))

                # 1) Position loss: MSE on per-residue translation vectors
                if w_pos > 0.0:
                    # LPOS_HUBER_DELTA scales with V_STD: the threshold should match
                    # the typical per-residue displacement, which scales with V_STD.
                    # CFG value is a MULTIPLIER: effective_delta = LPOS_HUBER_DELTA * V_STD.
                    # ── CHANGE 4: per-stride Huber delta ─────────────────────────────────────
                    _huber_mult = float(
                        cfg.get("LPOS_HUBER_DELTA_BY_STRIDE", {}).get(
                            stride, cfg.get("LPOS_HUBER_DELTA", 0.8)
                        )
                    )
                    _v_std_eff = float(
                        cfg.get("V_STD_BY_STRIDE", {}).get(stride, cfg.get("V_STD", 1.0))
                    )
                    _lpos_delta = (
                        _huber_mult * _v_std_eff
                        if bool(cfg.get("NORMALIZE_VELOCITIES", True))
                        else _huber_mult
                    )
                    # ─────────────────────────────────────────────────────────────────────────
                    l_pos = masked_huber_pos(
                        xP, xG, mR,
                        delta=_lpos_delta,
                    )

                # 2) Rotation loss: SO(3) geodesic squared angle (your helper is already mask-safe)
                if w_rot > 0.0:
                    l_rot = rotation_geodesic_mse(RP, RG, mR)

                # 3) Torsion loss: wrapped angular loss (1 - cos(Δθ)), mask-safe
                # IMPORTANT: this compares torsions in whatever convention tors_hat/tors_w are stored.
                # It is independent of reorder_torsions_to_openfold() (which is only for OpenFold atom placement).
                if w_tor > 0.0:
                    l_tor = torsion_loss(tP, tG, tM, mR)

            # Distogram loss (sparse steps; optional)
            l_disto = torch.tensor(0.0, device=device)
            disto_cnt = torch.tensor(0.0, device=device)
            k_disto = []

            if float(sched.get("w_disto", 0.0)) > 0.0:
                k_disto = self._select_steps(
                    W, "disto", only_last=bool(sched.get("disto_only_last")), stride=int(sched.get("disto_stride"))
                )
                if exclude_last:
                    k_disto = [k for k in k_disto if k != (W - 1)]

                for k in k_disto:
                    mk = mask_w[:, k].clamp(0, 1)
                    if self.cfg.get("DISTO_USE_SAMPLED", True):
                        disto_k_vec, pair_counts = distogram_loss_sampled_per_sample(
                            x_pred=x_hat_a[:, k].float(),
                            x_gt=x_w[:, k].float(),
                            res_mask=mk,
                            K=int(sched.get("disto_sample_k", self.cfg.get("DISTO_SAMPLE_K", 2048))),
                            k_min=self.cfg.get("DISTO_K_MIN_SEP", 4),
                            k_max=self.cfg.get("DISTO_K_MAX_SEP", 64),
                            mix_band=self.cfg.get("DISTO_MIX_BAND", 0.7),
                            d_max=self.cfg.get("DISTO_D_MAX", 30.0),
                        )
                        valid = (pair_counts > 0).to(disto_k_vec.dtype)
                        has_pairs_any = (valid.sum() > 0).to(disto_k_vec.dtype)
                        disto_k = (disto_k_vec * valid).sum() / valid.sum().clamp_min(1.0)
                    else:
                        disto_k_any = distogram_loss_per_sample(x_hat_a[:, k], x_w[:, k], mk)
                        disto_k = disto_k_any.mean() if disto_k_any.ndim > 0 else disto_k_any
                        has_pairs_any = ((mk.sum(dim=1) >= 2).any()).float()

                    if self.cfg.get("DISTO_AVG_ONLY_ACTIVE_STEPS", True):
                        l_disto = l_disto + disto_k * has_pairs_any
                        disto_cnt = disto_cnt + has_pairs_any
                    else:
                        l_disto = l_disto + disto_k
                        disto_cnt = disto_cnt + 1.0

                l_disto = l_disto / disto_cnt.clamp_min(1.0)
            # Atom14 supervision + OpenFold violations (averaged over window; sparse steps)
            l_a14     = torch.tensor(0.0, device=device)
            l_backbone = torch.tensor(0.0, device=device)  # ② backbone-only (N,CA,C,O)
            l_of      = torch.tensor(0.0, device=device)
            l_fape    = torch.tensor(0.0, device=device)
            l_pepgeo  = torch.tensor(0.0, device=device)
            l_pepvel  = torch.tensor(0.0, device=device)
            k_geom = []
            n_of = 0

            # ---- A14 / OF losses (k_of built separately; membership-based compute_of) ----
            w_a14 = float(sched.get("w_a14", 0.0))
            w_of  = float(sched.get("w_of", 0.0))
            w_fape = float(sched.get("w_fape", 0.0))


            w_backbone = float(sched.get("w_backbone", 0.0))  # ② backbone from step 0
            if (w_a14 > 0.0) or (w_of > 0.0) or (w_fape > 0.0) or (w_pepgeo > 0.0) or (w_backbone > 0.0):
                k_geom = self._select_steps(
                    W, "geom",
                    only_last=bool(sched.get("geom_only_last")),
                    stride=int(sched.get("geom_stride"))
                )

                # For forward task, ensure last step is included (stability / endpoint)
                if (not use_interp) and ((W - 1) not in k_geom):
                    k_geom = list(k_geom) + [W - 1]

                # For interpolation, exclude endpoint from window-averaged losses
                if exclude_last:
                    k_geom = [k for k in k_geom if k != (W - 1)]

                k_geom = sorted(set(k_geom))

                # -----------------------------
                # Build k_of separately
                # -----------------------------
                k_of = []
                of_stride = int(sched.get("of_stride", sched.get("OF_STRIDE", 0)))  # support either key

                if w_of > 0.0:
                    if not use_interp:
                        # Always compute OF at endpoint in forward mode
                        k_of.append(W - 1)

                        # Optional: compute OF periodically inside window (only if stride >= 2)
                        # Example: stride=8 => k=7,15,23,... (endpoint already included)
                        if of_stride >= 2:
                            k_of += list(range(of_stride - 1, W - 1, of_stride))
                    else:
                        # In interp mode, typically rely on endpoint losses (W_END_OF).
                        # If you still want OF here, keep it extremely sparse.
                        if (not exclude_last) and ((W - 1) in k_geom):
                            k_of.append(W - 1)

                k_of = sorted(set(k_of))

                # ---- denom-weighted aggregation across time steps (A14) ----
                device = aatype.device
                acc_a14_num = torch.zeros((), device=device, dtype=torch.float32)
                acc_a14_den = torch.zeros((), device=device, dtype=torch.float32)
                acc_pep_num = torch.zeros((), device=device, dtype=torch.float32)
                acc_pep_den = torch.zeros((), device=device, dtype=torch.float32)
                acc_bb_num  = torch.zeros((), device=device, dtype=torch.float32)
                acc_bb_den  = torch.zeros((), device=device, dtype=torch.float32)
                acc_pepvel_num = torch.zeros((), device=device, dtype=torch.float32)
                acc_pepvel_den = torch.zeros((), device=device, dtype=torch.float32)

                # ---- OF aggregation (mean over evaluated frames) ----
                acc_of = torch.zeros((), device=device, dtype=torch.float32)
                n_of = 0

                # Ensure loss module has correct step for debug gating
                if hasattr(self.allatom_loss, "global_step"):
                    self.allatom_loss.global_step = int(self.global_step)

                acc_fape = torch.zeros((), device=device, dtype=torch.float32)
                n_fape = 0

                # ── Stride-scaled A14 Huber delta ─────────────────────────────
                # A14_HUBER_DELTA is in Å. At stride>1, predicted atom positions
                # are dt× further from conditioning frame. Scale delta linearly
                # with dt so the Huber threshold stays physically meaningful.
                # A14_HUBER_DELTA_BY_STRIDE overrides if set; otherwise dt-scaling.
                _a14_delta_base = float(cfg.get("A14_HUBER_DELTA", 1.4))
                _a14_delta_by_stride = cfg.get("A14_HUBER_DELTA_BY_STRIDE", {})
                _a14_delta_stride = float(
                    _a14_delta_by_stride.get(stride,
                    _a14_delta_by_stride.get(str(stride),
                    _a14_delta_base * dt))        # default: linear dt-scaling
                )
                # ──────────────────────────────────────────────────────────────

                # ── dt² weight correction for geometry losses ─────────────────────────
                # w_a14 and w_fape from _schedule() use V_STD[1]² normalization.
                # At stride>1 atom14 / FAPE errors scale with dt²×V_STD_stride²,
                # so we apply the same correction as for w_pos.
                # NOT corrected: l_of (absolute geometry), l_backbone (Huber delta
                # already stride-aware via _a14_delta_stride — correcting weight too
                # would double-count the fix).
                _w_a14_corr  = float(sched.get("w_a14",  0.0))
                _w_fape_corr = float(sched.get("w_fape", 0.0))
                if stride > 1 and bool(cfg.get("NORMALIZE_VELOCITIES", True)):
                    _v_std_s_g = float(cfg.get("V_STD_BY_STRIDE", {}).get(stride, cfg.get("V_STD", 1.0)))
                    _v_std_1_g = float(cfg.get("V_STD", 1.0))
                    _dt2_v_g   = (dt ** 2) * (_v_std_s_g / _v_std_1_g) ** 2
                    _w_a14_corr  /= _dt2_v_g
                    _w_fape_corr /= _dt2_v_g
                # ─────────────────────────────────────────────────────────────────────

                for k in k_geom:
                    mk = mask_w[:, k].clamp(0, 1)

                    # membership-based OF selection
                    compute_of = (k in k_of)

                    out_a = self.allatom_loss(
                        aatype=aatype,
                        R=R_hat_a[:, k],
                        x=x_hat_a[:, k],
                        tors=tors_hat[:, k],
                        gt_a14=a14_w[:, k],
                        gt_a14m=a14m_w[:, k],
                        res_mask=mk,
                        compute_of=compute_of,
                        residue_index=residue_index,
                        huber_delta=_a14_delta_stride,
                    )
                    # Smooth peptide geometry preconditioner on predicted atom14 (distance + angles + optional omega)
                    if (w_pepgeo > 0.0):
                        pep_k, den_pep_k = peptide_geom_loss_atom14(
                            pred_a14=out_a["pred_a14"], pred_a14m=out_a["pred_a14_mask"],
                            res_mask=mk, adjacent=adjacent, aatype=aatype, cfg=cfg,
                            eps=float(cfg.get("PEP_GEO_EPS", 1e-6)),
                        )
                        acc_pep_num = acc_pep_num + pep_k.float() * den_pep_k.float()
                        acc_pep_den = acc_pep_den + den_pep_k.float()

                    if (w_pepvel > 0.0):
                        if k == 0:
                            ref_x    = x_c
                            ref_R    = R_c
                            ref_a14  = a14_c
                            ref_a14m = a14m_c
                        elif float(sched.get("r_backbone", 0.0)) < 0.5:
                            ref_x = None          # gate: skip k>0 until frames are trustworthy
                        else:
                            ref_x    = x_hat_a[:, k].detach()
                            ref_R    = R_hat_a[:, k].detach()
                            ref_a14  = out_a["pred_a14"].detach()
                            ref_a14m = out_a["pred_a14_mask"].detach()

                        if ref_x is not None:     # only call when ref geometry is actually defined
                            pv_k, den_pv_k = peptide_bond_velocity_constraint(
                                x_c=ref_x, R_c=ref_R,
                                a14_c=ref_a14, a14m_c=ref_a14m,
                                res_mask=mk, adjacent=adjacent,
                                v_phys=v1_hat_phys[:, k],
                                w_phys=o1_hat_phys[:, k],
                                eps=float(cfg.get("PEP_VEL_EPS", 1e-6)),
                            )
                            acc_pepvel_num = acc_pepvel_num + pv_k.float() * den_pv_k.float()
                            acc_pepvel_den = acc_pepvel_den + den_pv_k.float()

                    if w_fape > 0.0:
                      # Compute FAPE in FP32 (disable autocast BF16)
                      _fape_only_last = bool(cfg.get("FAPE_ONLY_LAST", True))  # default True
                      if (not _fape_only_last) or (k == k_geom[-1]):
                          fape_k = openfold_fape_atom14(
                              pred_R=R_hat_a[:, k],
                              pred_x=x_hat_a[:, k],
                              pred_a14=out_a["pred_a14"],
                              pred_a14m=out_a["pred_a14_mask"],
                              gt_R=R_w[:, k],
                              gt_x=x_w[:, k],
                              gt_a14=a14_w[:, k],
                              gt_a14m=out_a["m_sup"],
                              res_mask=mk,
                              length_scale=float(cfg.get("FAPE_LENGTH_SCALE", 10.0)),
                              clamp_distance=float(cfg.get("FAPE_CLAMP", 10.0)),
                              eps=float(cfg.get("FAPE_EPS", 1e-4)),
                          )
                          acc_fape = acc_fape + fape_k.float()
                          n_fape += 1

                    # Expecting AllAtom14Loss to return:
                    #   out_a["l_a14"]   : scalar (Huber mean over atoms at this frame)
                    #   out_a["den_a14"] : scalar (# supervised atoms at this frame)
                    l_a14_k = out_a["l_a14"].float()
                    den_k   = out_a["den_a14"].float()

                    # ② Backbone-only supervision (N=0, CA=1, C=2, O=4)
                    # Uses out_a["gt_a14"] and out_a["m_sup"] — same tensors used
                    # internally by allatom_loss for l_a14, so mask and GT are consistent.
                    # Accumulated with denom-weighting (same pattern as l_a14) so that
                    # variable atom counts across geom steps don't inflate the average.
                    if w_backbone > 0.0:
                        _BB_IDX  = [0, 1, 2, 4]                              # N, CA, C, O
                        _pred_bb = out_a["pred_a14"][:, :, _BB_IDX, :].float()  # (B,N,4,3)
                        _gt_bb   = out_a["gt_a14"][:, :, _BB_IDX, :].float()    # (B,N,4,3)
                        _m_bb    = out_a["m_sup"][:, :, _BB_IDX].float()         # (B,N,4)
                        _delta   = float(cfg.get("A14_HUBER_DELTA", 1.0))
                        _diff    = (_pred_bb - _gt_bb).norm(dim=-1)              # (B,N,4)
                        _huber   = torch.where(
                            _diff < _delta,
                            0.5 * _diff ** 2,
                            _delta * (_diff - 0.5 * _delta),
                        )
                        _l_bb_k  = (_huber * _m_bb).sum()   # weighted numerator
                        _d_bb_k  = _m_bb.sum()               # denominator (supervised atoms)
                        acc_bb_num = acc_bb_num + _l_bb_k
                        acc_bb_den = acc_bb_den + _d_bb_k

                    # denom-weighted sum: (mean_k * den_k) / sum(den_k)
                    if den_k.item() > 0:
                        acc_a14_num = acc_a14_num + l_a14_k * den_k
                        acc_a14_den = acc_a14_den + den_k

                    if compute_of:
                        # OF is already a scalar mean (or reduced) inside AllAtom14Loss
                        acc_of = acc_of + out_a.get("l_of", torch.zeros((), device=device)).float()
                        n_of += 1

                # Final aggregated losses
                if acc_a14_den.item() > 0:
                    l_a14 = acc_a14_num / acc_a14_den.clamp_min(1.0)
                else:
                    l_a14 = torch.zeros((), device=device, dtype=torch.float32)

                # Backbone: denom-weighted average over geom steps (fixes ~4× inflation)
                if acc_bb_den.item() > 0:
                    l_backbone = acc_bb_num / acc_bb_den.clamp_min(1.0)
                else:
                    l_backbone = torch.zeros((), device=device, dtype=torch.float32)

                if n_of > 0:
                    l_of = acc_of / float(n_of)
                else:
                    l_of = torch.zeros((), device=device, dtype=torch.float32)

                if n_fape > 0:
                    l_fape = acc_fape / float(n_fape)
                else:
                    l_fape = torch.zeros((), device=device, dtype=torch.float32)

                # Finalize smooth peptide geometry loss (denom-weighted over valid adjacent pairs)
                if (w_pepgeo > 0.0) and (acc_pep_den.item() > 0):
                    l_pepgeo = acc_pep_num / acc_pep_den.clamp_min(1.0)
                else:
                    l_pepgeo = torch.zeros((), device=device, dtype=torch.float32)

                if (w_pepvel > 0.0) and (acc_pepvel_den.item() > 0):
                    l_pepvel = acc_pepvel_num / acc_pepvel_den.clamp_min(1.0)
                else:
                    l_pepvel = torch.zeros((), device=device, dtype=torch.float32)


            l_end_pos = torch.tensor(0.0, device=device)
            l_end_rot = torch.tensor(0.0, device=device)
            l_end_tor = torch.tensor(0.0, device=device)
            l_end_a14 = torch.tensor(0.0, device=device)
            l_end_of = torch.tensor(0.0, device=device)

            if use_interp:
                m_end = mask_w[:, -1].clamp(0, 1)
                l_end_pos = masked_mse(((x_hat_a[:, -1] - x_w[:, -1]) ** 2).sum(dim=-1), m_end)
                l_end_rot = rotation_geodesic_mse(R_hat_a[:, -1], R_w[:, -1], m_end)
                m_end_t = (m_end.unsqueeze(-1) * torsm_w[:, -1]).clamp(0, 1)
                dth_end = wrap_to_pi(tors_hat[:, -1] - tors_w[:, -1])
                l_end_tor = masked_mse((dth_end ** 2), m_end_t)

                # Endpoint geometry supervision (interp only)
                if (float(sched.get("w_end_a14", 0.0)) > 0.0) or (float(sched.get("w_end_of", 0.0)) > 0.0):
                    compute_of_end = (float(sched.get("w_end_of", 0.0)) > 0.0)
                    out_end = self.allatom_loss(
                        aatype=aatype,
                        R=R_hat_a[:, -1],
                        x=x_hat_a[:, -1],
                        tors=tors_hat[:, -1],
                        gt_a14=a14_w[:, -1],
                        gt_a14m=a14m_w[:, -1],
                        res_mask=m_end,
                        compute_of=compute_of_end,
                        residue_index=residue_index,
                        huber_delta=_a14_delta_stride,
                    )
                    l_end_a14 = out_end["l_a14"]
                    if compute_of_end:
                        l_end_of = out_end.get("l_of", torch.tensor(0.0, device=device))

                # ── Endpoint loss dt² normalisation ──────────────────────────────
                # l_end_pos/a14/rot/tor are in physical space. For stride>1 the
                # endpoint sits dt× further from the start, so displacement variance
                # scales as dt²×V_STD_stride². Without this correction stride=4 is
                # over-penalised by 7.4× vs stride=1, causing the late-phase spikes.
                if stride > 1:
                    _vstd_ep = float(
                        cfg.get("V_STD_BY_STRIDE", {}).get(stride, cfg.get("V_STD", 1.0))
                    )
                    _dt2_pos = (dt ** 2) * (_vstd_ep ** 2) / (float(cfg.get("V_STD", 1.0)) ** 2)
                    l_end_pos = l_end_pos / _dt2_pos
                    l_end_a14 = l_end_a14 / _dt2_pos

                    _omg_ep = float(
                        cfg.get("OMG_STD_BY_STRIDE", {}).get(stride, cfg.get("OMG_STD", 1.0))
                    )
                    _dt2_rot = (dt ** 2) * (_omg_ep ** 2) / (float(cfg.get("OMG_STD", 1.0)) ** 2)
                    l_end_rot = l_end_rot / _dt2_rot

                    _th_ep = float(
                        cfg.get("THDOT_STD_BY_STRIDE", {}).get(stride, cfg.get("THDOT_STD", 1.0))
                    )
                    _dt2_tor = (dt ** 2) * (_th_ep ** 2) / (float(cfg.get("THDOT_STD", 1.0)) ** 2)
                    l_end_tor = l_end_tor / _dt2_tor

                # ── Endpoint soft-clamp (safety net for outlier batches) ──────────
                _ep_clamp = float(cfg.get("END_POS_CLAMP", float("inf")))
                if _ep_clamp < float("inf"):
                    l_end_pos = _ep_clamp * torch.tanh(l_end_pos / _ep_clamp)
                    l_end_a14 = _ep_clamp * torch.tanh(l_end_a14 / _ep_clamp)
                # ─────────────────────────────────────────────────────────────────

            if W > 1:
                # Optional smoothing on increments (exclude last step for interpolation windows if configured)
                if exclude_last:
                    if W > 2:
                        m_inc = (mask_w[:, :-2] * mask_w[:, 1:-1]).clamp(0, 1)  # k=0..W-3
                        dx_hat = x_hat_a[:, 1:-1] - x_hat_a[:, :-2]
                        dx_gt = x_w[:, 1:-1] - x_w[:, :-2]
                        l_dx = masked_mse(((dx_hat - dx_gt) ** 2).sum(dim=-1), m_inc)

                        dR_hat = R_hat_a[:, :-2].transpose(-1, -2) @ R_hat_a[:, 1:-1]
                        dR_gt = R_w[:, :-2].transpose(-1, -2) @ R_w[:, 1:-1]
                        l_dR = rotation_geodesic_mse(dR_hat, dR_gt, m_inc)

                        dth_hat = wrap_to_pi(tors_hat[:, 1:-1] - tors_hat[:, :-2])
                        dth_gt = wrap_to_pi(tors_w[:, 1:-1] - tors_w[:, :-2])
                        m_th_inc = (torsm_w[:, :-2] * torsm_w[:, 1:-1] * m_inc.unsqueeze(-1)).clamp(0, 1)
                        l_dth = masked_mse((wrap_to_pi(dth_hat - dth_gt) ** 2), m_th_inc)
                    else:
                        l_dx = torch.tensor(0.0, device=device)
                        l_dR = torch.tensor(0.0, device=device)
                        l_dth = torch.tensor(0.0, device=device)
                else:
                    m_inc = (mask_w[:, :-1] * mask_w[:, 1:]).clamp(0, 1)
                    dx_hat = x_hat_a[:, 1:] - x_hat_a[:, :-1]
                    dx_gt = x_w[:, 1:] - x_w[:, :-1]
                    l_dx = masked_mse(((dx_hat - dx_gt) ** 2).sum(dim=-1), m_inc)

                    dR_hat = R_hat_a[:, :-1].transpose(-1, -2) @ R_hat_a[:, 1:]
                    dR_gt = R_w[:, :-1].transpose(-1, -2) @ R_w[:, 1:]
                    l_dR = rotation_geodesic_mse(dR_hat, dR_gt, m_inc)

                    dth_hat = wrap_to_pi(tors_hat[:, 1:] - tors_hat[:, :-1])
                    dth_gt = wrap_to_pi(tors_w[:, 1:] - tors_w[:, :-1])
                    m_th_inc = (torsm_w[:, :-1] * torsm_w[:, 1:] * m_inc.unsqueeze(-1)).clamp(0, 1)
                    l_dth = masked_mse((wrap_to_pi(dth_hat - dth_gt) ** 2), m_th_inc)
            else:
                l_dx = torch.tensor(0.0, device=device)
                l_dR = torch.tensor(0.0, device=device)
                l_dth = torch.tensor(0.0, device=device)

            # ── Soft-clamp the correctly-accumulated l_pepgeo ────────────────
            # NOTE: the stale draft block that recomputed l_pepgeo from
            # undefined variables (d, d0, mask) has been removed.
            # l_pepgeo is already correct from the acc_pep_num/acc_pep_den
            # accumulation loop above.
            clamp = float(cfg.get("PEPGEO_LOSS_CLAMP", 50.0))
            l_pepgeo_clamped = clamp * torch.tanh(l_pepgeo / clamp)  # soft clamp

            # ── Velocity diversity regularizer ────────────────────────────────
            # v1_hat: (B,W,N,3) normalized FM endpoint estimate.
            # Penalizes when variance across W steps is below VEL_ENTROPY_MIN_VAR.
            # SAFE: operates on v1_hat, NOT on the FM interpolant (no (1-s) factor).
            _w_ent = float(cfg.get("W_VEL_ENTROPY", 0.0))
            l_vel_entropy = torch.tensor(0.0, device=device, dtype=torch.float32)
            if _w_ent > 0.0 and W > 1:
                with self.fp32_ctx():
                    _min_var = float(cfg.get("VEL_ENTROPY_MIN_VAR", 0.01))
                    _v_var = uhat_v_f.float().var(dim=1)            # (B,N,3)
                    _o_var = uhat_o_f.float().var(dim=1)            # (B,N,3)
                    _t_var = uhat_t_f.float().var(dim=1)            # (B,N,7)
                    _v_deficit = (_min_var - _v_var).clamp_min(0.0)
                    _o_deficit = (_min_var - _o_var).clamp_min(0.0)
                    _t_deficit = (_min_var - _t_var).clamp_min(0.0)
                    _m_ent = mask_w[:, 0].unsqueeze(-1).float()  # (B,N,1)
                    l_vel_entropy = (
                        (_v_deficit * _m_ent).mean() +
                        (_o_deficit * _m_ent).mean() +
                        (_t_deficit * _m_ent.expand_as(_t_var)).mean()
                    )

            # ── Per-residue RMSF supervision ──────────────────────────────────
            # Penalizes ||var(x_hat_a, W) − var(x_gt, W)|| per residue.
            # Requires W > 2 to have meaningful variance.
            # SAFE: uses x_hat_a (aligned predicted), not FM interpolant.
            _w_rmsf = float(cfg.get("W_RMSF_MATCH", 0.0))
            l_rmsf_match = torch.tensor(0.0, device=device, dtype=torch.float32)
            if _w_rmsf > 0.0 and W > 2:
                with self.fp32_ctx():
                    var_hat = x_hat_a.float().var(dim=1)           # (B,N,3)
                    var_gt  = x_w.float().var(dim=1)               # (B,N,3)
                    _m_rmsf = mask_w[:, 0].unsqueeze(-1).float()   # (B,N,1)
                    _vstd_rmsf = float(
                        cfg.get("V_STD_BY_STRIDE", {}).get(stride, cfg.get("V_STD", 1.0))
                    )
                    # displacement variance scales as dt² × V_STD² — must divide by both
                    _v2_rmsf = (dt ** 2) * (_vstd_rmsf ** 2)
                    l_rmsf_match = (
                        ((var_hat - var_gt).abs() * _m_rmsf).sum()
                        / (_m_rmsf.sum() * 3.0).clamp_min(1.0)
                        / _v2_rmsf
                    )

            loss = (
                float(sched["w_fm"]) * l_fm +
                float(sched.get("w_pos", 0.0)) * l_pos +
                float(sched.get("w_rot", 0.0)) * l_rot +
                float(sched.get("w_tor", 0.0)) * l_tor +
                float(sched.get("w_disto", 0.0)) * l_disto +
                _w_a14_corr * l_a14 +  # dt²-corrected; stride=4: weight×0.14
                float(sched.get("w_of", 0.0)) * l_of +
                float(sched.get("w_dx", 0.0)) * l_dx +
                float(sched.get("w_dR", 0.0)) * l_dR +
                float(sched.get("w_dth", 0.0)) * l_dth +
                float(sched.get("w_end_pos", 0.0)) * l_end_pos +
                float(sched.get("w_end_rot", 0.0)) * l_end_rot +
                float(sched.get("w_end_tor", 0.0)) * l_end_tor +
                float(sched.get("w_end_a14", 0.0)) * l_end_a14 +
                float(sched.get("w_end_of", 0.0)) * l_end_of +
                _w_fape_corr * l_fape  # dt²-corrected; stride=4: weight×0.14
                + float(sched.get("w_backbone", 0.0)) * l_backbone
                + float(sched.get("w_pepgeo", 0.0)) * l_pepgeo_clamped
                + float(sched.get("w_pepvel", 0.0)) * l_pepvel
                + _w_ent  * l_vel_entropy    # diversity: penalize intra-window collapse
                + _w_rmsf * l_rmsf_match     # RMSF: match per-residue fluctuation magnitude
            )

            # Ensure scalar loss for backward/logging when batch_size>1
            if isinstance(loss, torch.Tensor) and (loss.ndim != 0):
                loss = loss.mean()

            w_aux = float(cfg.get("W_AUX", 0.1))  # small weight — just for gradient flow
            if w_aux > 0 and aux_outputs is not None:
                for aux_pred in aux_outputs:
                    # same velocity loss but lightweight
                    l_aux = (
                        masked_mse(((aux_pred[..., :3] - u_v)**2).sum(-1), m_vel) +
                        masked_mse(((aux_pred[..., 3:6] - u_o)**2).sum(-1), m_vel) +
                        masked_mse(((aux_pred[..., 6:] - u_t)**2), m_th)
                    )
                    l_aux = l_aux * _stride_fm_w
                    loss = loss + w_aux * l_aux

        if profile:
            if device == "cuda":
                torch.cuda.synchronize()
            t_loss_done = time.perf_counter()

        log_every = int(cfg.get("LOG_EVERY", 20))
        log_now = (not train) or (step_i % log_every == 0) or profile

        _loss_for_log = float(loss.detach().float().mean().item())
        logs = {"loss": _loss_for_log}

        if log_now:
            logs.update({
                "interp": float(1.0 if use_interp else 0.0),
                "fm": float(l_fm.detach().float().mean().item()),
                "pos": float(l_pos.detach().float().mean().item()),
                "stride_fm_w": float(_stride_fm_w),
                "l_fm_raw": float(_l_fm_raw.float().mean().item()),
                "dt": float(dt),
                "rot": float(l_rot.detach().float().mean().item()),
                "tor": float(l_tor.detach().float().mean().item()),
                "a14": float(l_a14.detach().float().mean().item()),
                "of": float(l_of.detach().float().mean().item()),
                "dx": float(l_dx.detach().float().mean().item()),
                "dR": float(l_dR.detach().float().mean().item()),
                "dth": float(l_dth.detach().float().mean().item()),
                "end_pos": float(l_end_pos.detach().float().mean().item()),
                "end_rot": float(l_end_rot.detach().float().mean().item()),
                "end_tor": float(l_end_tor.detach().float().mean().item()),
                "end_a14": float(l_end_a14.detach().float().mean().item()),
                "end_of": float(l_end_of.detach().float().mean().item()),
                "fape": float(l_fape.detach().float().mean().item()),
                "pepgeo": float(l_pepgeo.detach().float().mean().item()),
                "pepvel": float(l_pepvel.detach().float().mean().item()),
                "vel_entropy": float(l_vel_entropy.detach().float().item()),
                "rmsf_match": float(l_rmsf_match.detach().float().item()),
            })

        # ------------------------- BACKWARD + OPT STEP (CLEAN) -------------------------
        t_bwd0 = None
        t_bwd1 = None

        if train and profile and device == "cuda":
            torch.cuda.synchronize()
        if train and profile:
            t_bwd0 = time.perf_counter()

        if train:
            # ---- backward + optimizer step (BF16-safe; no GradScaler) ----
            # ---- one-time check: loss must be a scalar ----
            if train and (step_i == 0) and bool(self.cfg.get("ASSERT_SCALAR_LOSS_ONCE", True)):
                if not torch.is_tensor(loss):
                    raise RuntimeError(f"[LOSS] expected tensor, got {type(loss)}")
                if loss.numel() != 1:
                    raise RuntimeError(
                        f"[LOSS] loss must be scalar. got shape={tuple(loss.shape)} numel={loss.numel()} dtype={loss.dtype}"
                    )
                if not torch.isfinite(loss).all():
                    raise RuntimeError(f"[LOSS] loss is non-finite at step 0: {loss}")


            # Optional: autograd anomaly detection for the first few steps (VERY slow)
            an_steps = int(cfg.get("DEBUG_ANOMALY_STEPS", 0))
            use_anom = (an_steps > 0) and (step_i < an_steps) and (not _is_compiling())

            max_norm = float("nan")
            pre_clip_norm = None
            clip_coef = None
            post_clip_norm = None
            per_group_pre  = {"backbone": None, "head": None, "other": None}
            per_group_coef = {"backbone": None, "head": None, "other": None}
            _group_clips   = {"backbone": float("nan"), "head": float("nan"), "other": float("nan")}


            loss = loss / float(cfg.get("GRAD_ACCUM_STEPS", 1))
            if use_anom:
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()

            accum = int(cfg.get("GRAD_ACCUM_STEPS", 1))
            self._batch_count += 1
            _is_opt_step = (self._batch_count % accum == 0)
            if _is_opt_step:
                max_norm = grad_clip_for_step(cfg, self.global_step)
                eps = float(cfg.get("GRAD_CLIP_EPS", 1e-6))

                pre_clip_norm = None
                clip_coef = None
                post_clip_norm = None

                # ── Per-group gradient clipping (Solution B) ─────────────────────────────────
                head_grad_clip    = float(cfg.get("HEAD_GRAD_CLIP", 3.0))
                ipa_grad_clip     = float(cfg.get("IPA_GRAD_CLIP", 50.0))
                default_grad_clip = max_norm  # fallback for other_params group

                # Build per-group param lists (only params that have gradients)
                _group_clips = {
                    "backbone": ipa_grad_clip,
                    "head":     head_grad_clip,
                    "other":    default_grad_clip,
                }

                # Map optimizer param groups to clip values by group name
                # param_groups order must match what was set in __init__:
                #   [0] = backbone (IPA blocks + aux_heads)
                #   [1] = head
                #   [2] = other (state_mlp, embeddings, etc.)
                _group_order = ["backbone", "head", "other"]

                per_group_pre  = {}   # group_name -> pre-clip norm
                per_group_coef = {}   # group_name -> clip coef

                _ar1_rho = float(cfg.get("NOISE_AR1_RHO", 0.0))

                # Same formula as AR(1) variance inflation used in your noise sampling:
                # scale = sqrt(W) × sqrt((1+rho)/(1-rho))
                _W_for_scale = int(cfg.get("WINDOW_SIZE", W))
                if _ar1_rho > 0.0:
                    _grad_norm_scale = math.sqrt(float(_W_for_scale)) * math.sqrt((1.0 + _ar1_rho) / (1.0 - _ar1_rho))
                else:
                    _grad_norm_scale = math.sqrt(float(_W_for_scale))

                for group_name, pg in zip(_group_order, self.opt.param_groups):
                    clip_max = _group_clips[group_name]
                    if clip_max <= 0.0:
                        per_group_pre[group_name]  = None
                        per_group_coef[group_name] = None
                        continue

                    params_with_grad = [p for p in pg["params"] if p.grad is not None]
                    if not params_with_grad:
                        per_group_pre[group_name]  = 0.0
                        per_group_coef[group_name] = 1.0
                        continue

                    # ── Normalize gradient values before clipping ─────────────────────────
                    # Exactly like: loss = raw_loss / V_STD²
                    # Here:         grad = raw_grad / _grad_norm_scale  (only for "other")
                    if group_name == "other":
                        for p in params_with_grad:
                            p.grad.data.div_(_grad_norm_scale)   # in-place normalize

                    norm_t = torch.nn.utils.clip_grad_norm_(params_with_grad, clip_max)
                    norm_f = float(norm_t)
                    coef   = min(1.0, clip_max / (norm_f + eps))

                    per_group_pre[group_name]  = norm_f
                    per_group_coef[group_name] = coef

                # Expose the backbone norm as the "main" pre_clip_norm for existing logging
                pre_clip_norm  = per_group_pre.get("backbone")
                clip_coef      = per_group_coef.get("backbone")
                post_clip_norm = (pre_clip_norm * clip_coef) if (pre_clip_norm is not None and clip_coef is not None) else None
                # ─────────────────────────────────────────────────────────────────────────────

                # ── Layer norm scan HERE (gradients still exist, before zero_grad) ──
                _layer_norms_for_dense = {}
                try:
                    for _n, _p in self.model.named_parameters():
                        if _p.grad is not None:
                            _layer_norms_for_dense[_n] = float(_p.grad.detach().float().norm().item())
                except Exception as _eg:
                    _layer_norms_for_dense = {}

                # ----optimizer step (was missing — weights never updated!) ----
                self.opt.step()
                self.sched.step()  #  step-level LR schedule (was per-epoch)
                self.opt.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.ema is not None:
                    self.ema.update(self.model)

            # ---- structured diagnostics (JSONL + optional TensorBoard) ----
            if (diag is not None) and (not _is_compiling()):
                try:
                    lr_now = float(self.opt.param_groups[0]["lr"])
                except Exception:
                    lr_now = float("nan")
                payload = dict(logs)
                payload.update(dict(
                    lr=lr_now,
                    epoch=int(getattr(self, "_epoch", 0)),
                    stride=int(batch["stride"].reshape(-1)[0].item()) if ("stride" in batch) else 0,
                    t0=int(batch["t0"].reshape(-1)[0].item()) if ("t0" in batch) else 0,
                    t0_frac=float(batch["t0_frac"].reshape(-1)[0].item()) if ("t0_frac" in batch) else 0.0,
                    grad_norm_pre=float(pre_clip_norm) if pre_clip_norm is not None else float("nan"),
                    grad_clip_max=float(max_norm),
                    grad_clip_coef=float(clip_coef) if clip_coef is not None else float("nan"),
                    # schedule ramps (debugging loss-ramp interactions)
                    r_disto=float(sched.get("r_disto", 0.0)),
                    r_a14=float(sched.get("r_a14", 0.0)),
                    r_of=float(sched.get("r_of", 0.0)),
                    r_pose=float(sched.get("r_pose", 0.0)),
                    r_interp=float(sched.get("r_interp", 0.0)),
                ))
                diag.maybe_log(step_i, payload)

                _bb = per_group_pre.get("backbone")
                _hd = per_group_pre.get("head")

                # ── DENSE per-step log (every step -> train_dense.jsonl) ────────────
                dense = dict(
                    epoch=int(getattr(self, "_epoch", 0)),
                    train=int(train),
                    interp=int(1 if use_interp else 0),
                    stride=int(batch["stride"].reshape(-1)[0].item()) if "stride" in batch else -1,
                    t0_frac=float(batch["t0_frac"].reshape(-1)[0].item()) if "t0_frac" in batch else float("nan"),

                    # ── Loss components ───────────────────────────────────────────────────────
                    loss=_loss_for_log,
                    loss_finite=int(torch.isfinite(loss.detach()).all().item()),
                    l_fm=float(l_fm.detach().float().mean().item()),
                    l_pos=float(l_pos.detach().float().mean().item()),
                    l_pos_mse=float(masked_mse(((xP - xG) ** 2).sum(dim=-1), mR).detach().item()) if (w_pos > 0.0) else 0.0,
                    l_rot=float(l_rot.detach().float().mean().item()),
                    l_tor=float(l_tor.detach().float().mean().item()),
                    l_a14=float(l_a14.detach().float().mean().item()),
                    l_of=float(l_of.detach().float().mean().item()),
                    l_fape=float(l_fape.detach().float().mean().item()),
                    l_backbone=float(l_backbone.detach().float().mean().item()),
                    l_pepgeo=float(l_pepgeo.detach().float().mean().item()),
                    l_pepvel=float(l_pepvel.detach().float().mean().item()),
                    l_end_pos=float(l_end_pos.detach().float().mean().item()),
                    l_end_rot=float(l_end_rot.detach().float().mean().item()),
                    l_end_a14=float(l_end_a14.detach().float().mean().item()),
                    l_vel_entropy=float(l_vel_entropy.detach().float().item()),
                    l_rmsf_match=float(l_rmsf_match.detach().float().item()),
                    l_dx=float(l_dx.detach().float().mean().item()),
                    l_dR=float(l_dR.detach().float().mean().item()),
                    l_dth=float(l_dth.detach().float().mean().item()),

                    # ── Gradient norms — PRE-clip (what arrived before clipping) ─────────────
                    # These tell you how large the raw gradients are.
                    # Large values = unstable region; should shrink as training stabilises.
                    grad_norm_pre_backbone=float(per_group_pre.get("backbone") or 0.0) if _is_opt_step else float("nan"),
                    grad_norm_pre_head=float(per_group_pre.get("head")     or 0.0) if _is_opt_step else float("nan"),
                    grad_norm_pre_other=float(per_group_pre.get("other")   or 0.0) if _is_opt_step else float("nan"),
                    # Global pre-clip norm: sqrt(sum of squared norms across all groups).
                    # This is what a single clip_grad_norm_ over all params would see.
                    grad_norm_pre_global=float(
                        (
                            (per_group_pre.get("backbone") or 0.0) ** 2 +
                            (per_group_pre.get("head")     or 0.0) ** 2 +
                            (per_group_pre.get("other")    or 0.0) ** 2
                        ) ** 0.5
                    ) if _is_opt_step else float("nan"),

                    # ── Gradient norms — POST-clip (what actually hits the optimizer) ─────────
                    # post = min(pre, clip_threshold).
                    # If post << pre: clipping was active and truncated the update.
                    # If post ≈ pre: gradient was within budget; no clipping occurred.
                    grad_norm_post_backbone=float(min(
                        per_group_pre.get("backbone") or 0.0,
                        _group_clips.get("backbone", float("inf"))
                    )) if (self._batch_count % accum == 0) else float("nan"),
                    grad_norm_post_head=float(min(
                        per_group_pre.get("head")     or 0.0,
                        _group_clips.get("head",     float("inf"))
                    )) if (self._batch_count % accum == 0) else float("nan"),
                    grad_norm_post_other=float(min(
                        per_group_pre.get("other")    or 0.0,
                        _group_clips.get("other",    float("inf"))
                    )) if (self._batch_count % accum == 0) else float("nan"),
                    # Global post-clip norm: same sqrt-sum but using post values.
                    grad_norm_post_global=float(
                        (
                            min(per_group_pre.get("backbone") or 0.0, _group_clips.get("backbone", float("inf"))) ** 2 +
                            min(per_group_pre.get("head")     or 0.0, _group_clips.get("head",     float("inf"))) ** 2 +
                            min(per_group_pre.get("other")    or 0.0, _group_clips.get("other",    float("inf"))) ** 2
                        ) ** 0.5
                    )if (self._batch_count % accum == 0) else float("nan"),

                    # ── Clip coefficients (pre/post ratio per group) ──────────────────────────
                    # coef = 1.0  → not clipped (gradient was within budget)
                    # coef < 1.0  → clipped; coef=0.25 means gradient was 4x too large
                    grad_coef_backbone=float(per_group_coef.get("backbone") or 1.0) if _is_opt_step else float("nan"),
                    grad_coef_head=float(per_group_coef.get("head")         or 1.0) if _is_opt_step else float("nan"),
                    grad_coef_other=float(per_group_coef.get("other")       or 1.0) if _is_opt_step else float("nan"),

                    # ── Clip thresholds actually used this step ───────────────────────────────
                    # Lets you verify the schedule is applying the right threshold per step.
                    grad_clip_thresh_backbone=float(_group_clips.get("backbone", 0.0)),
                    grad_clip_thresh_head=float(_group_clips.get("head",         0.0)),
                    grad_clip_thresh_other=float(_group_clips.get("other",       0.0)),  # = max_norm from schedule

                    # ── Legacy fields (kept for backward compat with existing plots) ──────────
                    grad_norm_pre=float(pre_clip_norm)  if pre_clip_norm  is not None else float("nan"),
                    grad_clip_max=float(max_norm),
                    grad_clip_coef=float(clip_coef)    if clip_coef      is not None else float("nan"),
                    grad_clipped=int((clip_coef is not None) and (clip_coef < 1.0)) if _is_opt_step else -1,
                    grad_norm_backbone=float(_bb)      if _bb            is not None else float("nan"),
                    grad_norm_head=float(_hd)          if _hd            is not None else float("nan"),
                    grad_coef_backbone_legacy=per_group_coef.get("backbone", float("nan")) or float("nan"),
                    grad_coef_head_legacy=per_group_coef.get("head",         float("nan")) or float("nan"),

                    # ── Optimizer / schedule ──────────────────────────────────────────────────
                    lr=lr_now,
                    r_disto=float(sched.get("r_disto", 0.0)),
                    r_a14=float(sched.get("r_a14",     0.0)),
                    r_of=float(sched.get("r_of",       0.0)),
                    r_fape = float(sched.get("r_fape", 0.0)),
                    r_pose=float(sched.get("r_pose",   0.0)),
                    r_interp=float(sched.get("r_interp", 0.0)),

                    # ── Normalization scale diagnostics ──────────────────────────────────────
                    norm_v_std=float(cfg.get("V_STD",     1.0)),
                    norm_omg_std=float(cfg.get("OMG_STD", 1.0)),
                    norm_thdot_std=float(cfg.get("THDOT_STD", 1.0)),
                    norm_v_std2=float(cfg.get("V_STD",    1.0)) ** 2,

                    # Per-stride stds actually used this step (after stride-grouped sampler fix)
                    norm_v_std_eff=float(cfg.get("V_STD_BY_STRIDE", {}).get(stride, cfg.get("V_STD", 1.0))),
                    norm_omg_std_eff=float(cfg.get("OMG_STD_BY_STRIDE", {}).get(stride, cfg.get("OMG_STD", 1.0))),
                    norm_thdot_std_eff=float(cfg.get("THDOT_STD_BY_STRIDE", {}).get(stride, cfg.get("THDOT_STD", 1.0))),

                    # ── Stride-aware FM loss diagnostics ─────────────────────────────────────
                    # l_fm_raw: unweighted FM loss — use this to compare quality across strides.
                    # l_fm (already in dense above) is the stride-weighted value that enters the grad.
                    l_fm_raw=float(_l_fm_raw.float().mean().item()),
                    stride_fm_w=float(_stride_fm_w),          # 0.4 for stride=1 / 0.7 for stride=2 / 1.0 for stride=4
                    dt=float(dt),                             # physical dt used for GT velocity computation this step
                    # l_fm component breakdown (raw, before stride weighting)
                    l_fm_v_raw=float(l_fm_v.detach().float().mean().item()),
                    l_fm_o_raw=float(l_fm_o.detach().float().mean().item()),
                    l_fm_t_raw=float(l_fm_t.detach().float().mean().item()),

                    # ── Effective loss weights ────────────────────────────────────────────────
                    w_pepgeo_eff=float(sched.get("w_pepgeo",   0.0)),
                    w_a14_eff=float(sched.get("w_a14",         0.0)),
                    w_of_eff=float(sched.get("w_of",           0.0)),
                    w_backbone_eff=float(sched.get("w_backbone", 0.0)),
                    w_fape_eff = float(sched.get("w_fape", 0.0)),

                    # ── Ramp / anchor diagnostics ─────────────────────────────────────────────
                    r_backbone=float(sched.get("r_backbone",   0.0)),
                    anchor_alpha=float(anchor_alpha_cur),
                    frame_drift_A=float(frame_drift_A),

                    # ── Effective loss contributions (weight × raw loss) ─────────────────────
                    contrib_backbone=float(sched.get("w_backbone", 0.0)) * float(l_backbone.detach().float().mean().item()),
                    contrib_a14=float(sched.get("w_a14",           0.0)) * float(l_a14.detach().float().mean().item()),
                    contrib_pepgeo=float(sched.get("w_pepgeo",     0.0)) * float(l_pepgeo_clamped.detach().float().mean().item()),
                    contrib_fm=float(sched["w_fm"]) * float(l_fm.detach().float().mean().item()),

                    # ── Structural health ratios ──────────────────────────────────────────────
                    ratio_bb_to_a14=float(l_backbone.detach().float().mean().item()) /
                                    max(float(l_a14.detach().float().mean().item()), 1e-6),
                    l_pepgeo_clamped=float(l_pepgeo_clamped.detach().float().mean().item()),
                    pepgeo_clamp_ratio=float(l_pepgeo_clamped.detach().float().mean().item()) /
                                      max(float(l_pepgeo.detach().float().mean().item()), 1e-6),

                    # ── Per-block gradient diagnostics (first, second, last IPA block) ────────
                    # Detects vanishing/exploding gradients through depth.
                    # Healthy: roughly similar magnitude across blocks.
                    # block0_dominance ≈ 0.33 means block0 carries 1/3 of gradient energy (balanced).
                    # block0_dominance → 1.0 means only the first block is learning (vanishing in depth).
                    grad_block0_max=float(max(
                        (p.grad.abs().max().item() for p in self.model.parameters()
                        if p.grad is not None and hasattr(p, '_block_idx') and p._block_idx == 0),
                        default=float("nan")
                    )) if False else float("nan"),   # placeholder — wire up if block params are tagged
                )
                # top-5 exploding layers + NaN grad detection
                if self._batch_count % accum == 0 and _layer_norms_for_dense:
                    top5 = sorted(_layer_norms_for_dense.items(), key=lambda x: x[1], reverse=True)[:5]
                    _b0_norms = [v for k, v in _layer_norms_for_dense.items() if "blocks.0" in k]
                    _b1_norms = [v for k, v in _layer_norms_for_dense.items() if "blocks.1" in k]
                    _b5_norms = [v for k, v in _layer_norms_for_dense.items() if "blocks.5" in k]
                    dense["grad_block0_max"] = float(max(_b0_norms)) if _b0_norms else 0.0
                    dense["grad_block1_max"] = float(max(_b1_norms)) if _b1_norms else 0.0
                    dense["grad_block5_max"] = float(max(_b5_norms)) if _b5_norms else 0.0
                    _total_grad = sum(_layer_norms_for_dense.values())
                    _b0_total = sum(_b0_norms)
                    dense["grad_block0_dominance"] = float(_b0_total / max(_total_grad, 1e-8))
                    dense["top_grad_layers"] = [[n, round(v, 4)] for n, v in top5]
                    nan_layers = [n for n, v in _layer_norms_for_dense.items() if not math.isfinite(v)]
                    if nan_layers:
                        dense["nan_grad_layers"] = nan_layers[:10]

                diag.log_always(step_i, dense)
                # ─────────────────────────────────────────────────────────────────────

            # ---- logging example (adapt to your logger) ----
            # if train and (step_i % int(cfg.get("LOG_EVERY", 10)) == 0):
            #     if pre_clip_norm is not None:
            #         print(f"[GRAD] pre={pre_clip_norm:.4g} max={max_norm:.4g} coef={clip_coef:.4g} post≈{post_clip_norm:.4g}")


        if train and profile and device == "cuda":
            torch.cuda.synchronize()
        if train and profile:
            t_bwd1 = time.perf_counter()

        # ------------------------- PROFILE LOGGING (KEEP) -------------------------
        if profile:
            if device == "cuda":
                torch.cuda.synchronize()
            t_step1 = time.perf_counter()

            t_total = (t_step1 - t_step0) * 1000.0
            t_to = (t_to_device - t_step0) * 1000.0
            t_noise_int = (t_noise_int1 - t_noise_int0) * 1000.0
            t_fwd = (t_fwd1 - t_fwd0) * 1000.0

            if t_bwd0 is not None:
                t_losses = (t_bwd0 - t_fwd1) * 1000.0
            else:
                t_losses = (t_loss_done - t_fwd1) * 1000.0

            t_bwd = ((t_bwd1 - t_bwd0) * 1000.0) if (t_bwd0 is not None and t_bwd1 is not None) else 0.0
            logs.update(dict(
                t_to_ms=t_to,
                t_noiseint_ms=t_noise_int,
                t_fwd_ms=t_fwd,
                t_losses_ms=t_losses,
                t_bwdopt_ms=t_bwd,
                t_total_ms=t_total,
            ))
            print(f"[PROFILE] to(device)={t_to:.1f}ms | noise+int={t_noise_int:.1f}ms | fwd={t_fwd:.1f}ms | losses={t_losses:.1f}ms | bwd+opt={t_bwd:.1f}ms | total={t_total:.1f}ms")
            self._profile_printed = True

        return loss.detach(), logs

    def run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> Dict[str, float]:
        self._epoch = int(epoch)
        self.model.train() if train else self.model.eval()
        totals = {}
        n = 0
        pbar = tqdm(loader, desc=("train" if train else "val") + f" ep{epoch:03d}")
        import time
        t_prev = time.perf_counter()
        for batch in pbar:
                t_iter = time.perf_counter()
                t_load = (t_iter - t_prev) * 1000.0
                with torch.set_grad_enabled(train):
                    loss, logs = self.step(batch, train=train)
                # attach DataLoader+collate time for the very first batch (ms)
                if bool(self.cfg.get("PROFILE_ONE_BATCH", False)) and train and (n == 0) and (epoch == 1):
                    logs["t_load_ms"] = float(t_load)
                    print(f"[PROFILE] dataloader={t_load:.1f}ms")
                t_prev = time.perf_counter()
                for k, v in logs.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                n += 1
                avg_loss = totals["loss"] / max(1, n)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

                if train:
                    _total = int(self.cfg.get("TOTAL_STEPS", 0))
                    if _total > 0 and int(self.global_step) >= _total:
                        break

        for k in totals:
            totals[k] /= max(1, n)
        if train:
            pass  # sched.step() moved to per-batch (after opt.step)
        return totals

# %% cell 35
from torch.utils.data import Sampler
import random as _random

class StrideGroupedBatchSampler(Sampler):
    """
    Yields batches where every item shares the same stride.

    Guarantees that batch['stride'] is uniform, so the trainer can safely
    use batch['stride'][0] for dt, normalize_velocities, and stride_embed.

    Args:
        dataset   : WindowVelocityFlowDataset instance (must have _stride_for_idx)
        batch_size: items per batch
        shuffle   : shuffle within each stride bucket each epoch (default True)
        drop_last : drop final incomplete batch per stride bucket (default True)
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = True):
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last

        # Build stride -> [idx, ...] buckets from pre-assigned strides
        from collections import defaultdict
        buckets = defaultdict(list)
        for idx, s in enumerate(dataset._stride_for_idx):
            buckets[s].append(idx)
        self.buckets = dict(buckets)   # {stride_int: [idx, ...]}

    def __iter__(self):
        all_batches = []
        for stride, indices in self.buckets.items():
            idx_pool = list(indices)
            if self.shuffle:
                _random.shuffle(idx_pool)
            # Slice into batches
            for start in range(0, len(idx_pool), self.batch_size):
                batch = idx_pool[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        # Shuffle batch order across strides so training sees all strides interleaved
        if self.shuffle:
            _random.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.buckets.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

# %% cell 36
import torch
from collections import deque

def _iter_tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _iter_tensors(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _iter_tensors(v)

def _tensor_summary(t: torch.Tensor, max_bad_idx=5):
    tt = t.detach()
    # avoid bf16 min/max issues; keep on device, but cast to fp32 for summary
    if tt.is_floating_point():
        tt = tt.float()

    finite = torch.isfinite(tt)
    if finite.all():
        amax = tt.abs().max()
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={tt.min().item():.3e} max={tt.max().item():.3e} mean={tt.mean().item():.3e} "
            f"absmax={amax.item():.3e}"
        )

    idx = (~finite).nonzero(as_tuple=False)
    show = idx[:max_bad_idx].tolist()
    return f"NON-FINITE! shape={tuple(t.shape)} dtype={t.dtype} bad_idx[:{max_bad_idx}]={show}"

class NanInfTracker:
    """
    Centralized numeric-stability tracker:
      - forward output checks
      - (optional) forward input checks
      - gradient checks via register_hook on outputs
      - absmax threshold to catch explosions BEFORE NaN/Inf
    """
    def __init__(
        self,
        enabled=True,
        raise_on_first=True,
        max_reports=5,
        absmax_thr=1e4,
        grad_absmax_thr=1e4,
        check_inputs=False,
        recent=40,
    ):
        self.enabled = bool(enabled)
        self.raise_on_first = bool(raise_on_first)
        self.max_reports = int(max_reports)
        self.absmax_thr = float(absmax_thr)
        self.grad_absmax_thr = float(grad_absmax_thr)
        self.check_inputs = bool(check_inputs)

        self._reports = 0
        self._seen = 0
        self.hit = False
        self.last_msg = None
        self.recent_msgs = deque(maxlen=int(recent))

    def reset_step(self):
        self.hit = False
        self.last_msg = None

    def _emit(self, msg: str):
        self.hit = True
        self.last_msg = msg
        self.recent_msgs.append(msg)
        print(msg, flush=True)
        if self.raise_on_first:
            raise RuntimeError(msg)

    def _report_tensor(self, where: str, name: str, t: torch.Tensor):
        if self._reports >= self.max_reports:
            return
        self._reports += 1
        self._emit(f"[NAN-TRACK] {where} :: {name} :: {_tensor_summary(t)}")

    def _check_tensor(self, where: str, name: str, t: torch.Tensor, thr: float):
        if not torch.is_tensor(t) or (not t.is_floating_point()):
            return

        td = t.detach()
        if not torch.isfinite(td).all():
            self._report_tensor(where + "::NONFINITE", name, t)
            return

        # catch pre-NaN explosion
        amax = td.abs().max()
        # avoid device sync unless over threshold
        if amax.isfinite() and amax > thr:
            self._report_tensor(where + f"::ABSMAX>{thr:g}", name, t)

    def make_forward_hook(self, mod_name: str):
        def hook(module, inputs, output):
            if not self.enabled:
                return

            # sanity: confirm hooks are executing
            if self._seen < 1:
                print(f"[NAN-TRACK] forward hook active (example): {mod_name}", flush=True)
                self._seen += 1

            if self.check_inputs:
                for t in _iter_tensors(inputs):
                    self._check_tensor("FWD_IN", mod_name, t, self.absmax_thr)

            for t in _iter_tensors(output):
                self._check_tensor("FWD_OUT", mod_name, t, self.absmax_thr)

            # attach grad hooks to outputs (localizes bad gradient emergence)
            self.attach_grad_hook(mod_name, output)

        return hook

    def attach_grad_hook(self, mod_name: str, output):
        if not self.enabled:
            return
        for t in _iter_tensors(output):
            if torch.is_tensor(t) and t.requires_grad and t.is_floating_point():
                def _g_hook(grad, n=mod_name):
                    if grad is None:
                        return grad
                    if not torch.isfinite(grad.detach()).all():
                        self._report_tensor("BWD_GRAD::NONFINITE", n, grad)
                    else:
                        # pre-NaN grad explosion
                        gmax = grad.detach().abs().max()
                        if gmax.isfinite() and gmax > self.grad_absmax_thr:
                            self._report_tensor(f"BWD_GRAD::ABSMAX>{self.grad_absmax_thr:g}", n, grad)
                    return grad
                t.register_hook(_g_hook)

    def report(self, tail=40):
        if not self.recent_msgs:
            return "[NAN-TRACK] no events recorded."
        tail = min(int(tail), len(self.recent_msgs))
        lines = list(self.recent_msgs)[-tail:]
        return "[NAN-TRACK] recent events:\n" + "\n".join(lines)

# %% cell 37
# --------------------------------------------------------------------------------------
# 12) Training entry point  — EPOCHS-anchored, step-controlled termination
# --------------------------------------------------------------------------------------
#
# Design principles:
#   • EPOCHS in CFG is the single user-facing control. It means
#     "how many full passes over the training data". Never changes.
#   • TOTAL_STEPS is computed dynamically each run from:
#         current_global_step + (EPOCHS - ckpt_epoch) * steps_per_epoch
#     This means the cosine schedule always spans exactly the remaining
#     epochs at the current dataset size, regardless of whether you switched
#     datasets, changed batch size, or filtered proteins.
#   • run_epoch() stops mid-batch the instant global_step >= TOTAL_STEPS.
#     Training therefore ends exactly at the right step — never overshoots
#     into a wasteful "LR stuck at LR_MIN" tail.
#   • TOTAL_STEPS is NEVER set in CFG. It is always derived here.
# --------------------------------------------------------------------------------------

def build_model(cfg: Dict[str, Any], esm_dim: int, do_compile: bool = True) -> nn.Module:
    model = VelocityFlowIPADynamics(esm_dim=esm_dim, cfg=cfg).to(cfg["DEVICE"])
    model = model.to(dtype=torch.float32)
    if do_compile and bool(cfg.get("COMPILE", False)):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print("torch.compile failed:", e)
    return model


def main_train(train_dir: str, val_dir: str, cfg: Dict[str, Any]) -> None:
    import os, math, inspect
    import torch
    from torch.utils.data import DataLoader
    import torch._dynamo as dynamo

    def unwrap_model(m):
        return getattr(m, "_orig_mod", m)

    def strip_prefix(sd, prefix):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}

    def normalize_state_dict_keys(sd):
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = strip_prefix(sd, "_orig_mod.")
        if any(k.startswith("module.")   for k in sd):
            sd = strip_prefix(sd, "module.")
        return sd

    # ── 1. Datasets and loaders ──────────────────────────────────────────────
    train_ds = WindowVelocityFlowDataset(train_dir, cfg, train=True)
    val_ds   = WindowVelocityFlowDataset(val_dir,   cfg, train=False)
    dynamo.reset()

    sample  = train_ds[0]
    esm_dim = int(sample["esm"].shape[-1])
    print("Training Folder:", NEW_TRAINING_FOLDER)
    print("ESM dim:", esm_dim)

    nw          = 0 if cfg.get("PRELOAD_TO_RAM", False) else int(cfg.get("NUM_WORKERS", 2))
    _batch_size = int(cfg["BATCH_SIZE"])

    train_sampler = StrideGroupedBatchSampler(train_ds, batch_size=_batch_size,
                                              shuffle=True, drop_last=True)
    val_sampler   = StrideGroupedBatchSampler(val_ds,   batch_size=_batch_size,
                                              shuffle=False, drop_last=False)

    dl_kw = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        dl_kw.update(persistent_workers=True,
                     prefetch_factor=int(cfg.get("PREFETCH_FACTOR", 4)))

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **dl_kw)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,   **dl_kw)

    steps_per_epoch = len(train_loader)        # actual value for THIS dataset
    cfg["STEPS_PER_EPOCH"] = int(steps_per_epoch)

    # ── 2. Read checkpoint EARLY — needed to compute TOTAL_STEPS ─────────────
    resume_path   = cfg.get("RESUME_FROM", None)
    resume_strict = bool(cfg.get("RESUME_STRICT", True))

    start_epoch      = 1
    best             = float("inf")
    ckpt_epoch       = 0
    ckpt             = None
    global_step_ckpt = 0

    if resume_path:
        ckpt  = torch.load(resume_path, map_location=cfg["DEVICE"])
        if isinstance(ckpt, dict):
            ckpt_epoch       = int(ckpt.get("epoch", 0))
            start_epoch      = ckpt_epoch + 1
            best             = float(ckpt.get("best_val", best))
            global_step_ckpt = int(ckpt.get("global_step",
                                             max(0, (ckpt_epoch - 1) * steps_per_epoch)))
        print(f"[RESUME] loaded: {resume_path}")
        print(f"[RESUME] ckpt_epoch={ckpt_epoch} -> start_epoch={start_epoch}")

    # ── 3. Compute TOTAL_STEPS dynamically (EPOCHS-anchored) ─────────────────
    #
    #   TOTAL_STEPS = current_global_step
    #               + (EPOCHS_cfg - ckpt_epoch) * steps_per_epoch
    #
    # Meaning: "run for exactly (EPOCHS_cfg - ckpt_epoch) more full dataset
    # passes, at whatever step rate this dataset produces per pass."
    #
    # This is always correct because:
    #   • Same dataset, continuing   → same total_steps as before (no change)
    #   • Larger dataset (R1+R2+R3)  → larger total_steps (cosine stretched
    #                                   proportionally over more steps)
    #   • Smaller dataset            → smaller total_steps (cosine compressed)
    #   • Fresh run (ckpt_epoch=0)   → total_steps = EPOCHS * steps_per_epoch
    #
    epochs_cfg       = int(cfg["EPOCHS"])
    epochs_remaining = epochs_cfg - ckpt_epoch
    if epochs_remaining <= 0:
        raise ValueError(
            f"[ABORT] cfg EPOCHS={epochs_cfg} <= ckpt_epoch={ckpt_epoch}. "
            f"Nothing to train. Increase EPOCHS in CFG."
        )

    total_steps             = global_step_ckpt + epochs_remaining * steps_per_epoch
    cfg["TOTAL_STEPS"]      = int(total_steps)   # write back for scheduler + trainer

    print(f"[STEPS] steps/epoch={steps_per_epoch}  "
          f"epochs_remaining={epochs_remaining}  "
          f"global_step_at_resume={global_step_ckpt}  "
          f"total_steps={total_steps}  "
          f"lr_warmup_steps={int(cfg.get('LR_WARMUP_STEPS', 0))}")
    print(f"[SCHEDULE] cosine: step {global_step_ckpt} → {total_steps}  "
          f"(spans {total_steps - global_step_ckpt} steps = {epochs_remaining} epochs × {steps_per_epoch} steps/epoch)")

    # ── 4. Build model, load weights, compile ────────────────────────────────
    build_sig = None
    try:
        build_sig = inspect.signature(build_model)
    except Exception:
        pass

    model = (build_model(cfg, esm_dim, do_compile=False)
             if (build_sig and "do_compile" in build_sig.parameters)
             else build_model(cfg, esm_dim))

    if resume_path and ckpt is not None:
        state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
        unwrap_model(model).load_state_dict(normalize_state_dict_keys(state),
                                            strict=resume_strict)

    if bool(cfg.get("COMPILE", False)) and not hasattr(model, "_orig_mod"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print("torch.compile failed:", e)

    # ── 5. Trainer — restore opt / EMA / scheduler / global_step ─────────────
    trainer = VelocityFMTrainer(model, cfg)

    if not hasattr(trainer, "run_epoch"):
        raise AttributeError(
            "VelocityFMTrainer.run_epoch is missing. "
            "Restart runtime and run all cells."
        )

    if resume_path and isinstance(ckpt, dict):
        if "opt" in ckpt:
            trainer.opt.load_state_dict(ckpt["opt"])
        if trainer.ema is not None and "ema" in ckpt:
            trainer.ema.shadow = ckpt["ema"]
        if hasattr(trainer, "sched") and "sched" in ckpt:
            try:
                trainer.sched.load_state_dict(ckpt["sched"])
            except Exception as e:
                print("[RESUME] sched load failed (fresh sched):", e)
        if hasattr(trainer, "global_step"):
            trainer.global_step = global_step_ckpt
        if "best_val" in ckpt:
            best = float(ckpt["best_val"])
        print(f"[RESUME] restored opt/ema/sched/global_step={global_step_ckpt}  "
              f"best_val={best:.6f}")

    print("[MODEL TYPE]", type(model))
    print("[COMPILED?]",  hasattr(model, "_orig_mod"))
    print("Params:", sum(p.numel() for p in unwrap_model(model).parameters()))

    # ── 6. Training loop ─────────────────────────────────────────────────────
    VAL_EMA_ALPHA = float(cfg.get("VAL_EMA_ALPHA", 0.3))
    val_ema_loss  = None

    for epoch in range(start_epoch, epochs_cfg + 1):

        # run_epoch honours cfg["TOTAL_STEPS"]: it breaks the inner batch loop
        # the moment global_step >= TOTAL_STEPS, so we never overshoot.
        tr = trainer.run_epoch(train_loader, train=True,  epoch=epoch)
        va = trainer.run_epoch(val_loader,   train=False, epoch=epoch)
        lr = trainer.opt.param_groups[0]["lr"]

        va_loss_raw = float(va["loss"])
        val_ema_loss = (va_loss_raw if val_ema_loss is None
                        else (1 - VAL_EMA_ALPHA) * val_ema_loss + VAL_EMA_ALPHA * va_loss_raw)

        print(f"[E{epoch:03d}] lr={lr:.3e} train={tr['loss']:.6f} "
              f"val={va_loss_raw:.6f} val_ema={val_ema_loss:.6f}")

        if hasattr(trainer, "diag") and trainer.diag is not None:
            val_log = dict(va)
            val_log.update(epoch=epoch, train=0,
                           val_ema_loss=round(val_ema_loss, 6),
                           val_raw_loss=round(va_loss_raw, 6),
                           lr=lr, step=int(trainer.global_step))
            trainer.diag.log_always(trainer.global_step, val_log)

        ckpt_out = dict(
            epoch=epoch, cfg=cfg,
            model=unwrap_model(model).state_dict(),
            opt=trainer.opt.state_dict(),
            best_val=float(best),
        )
        if trainer.ema is not None:
            ckpt_out["ema"] = trainer.ema.shadow
        if hasattr(trainer, "sched"):
            ckpt_out["sched"] = trainer.sched.state_dict()
        if hasattr(trainer, "global_step"):
            ckpt_out["global_step"] = int(trainer.global_step)

        torch.save(ckpt_out, os.path.join(cfg["CKPT_DIR"], "last.pt"))
        torch.save(ckpt_out, os.path.join(cfg["CKPT_DIR"], f"epoch_{epoch:04d}.pt"))

        if val_ema_loss < best:
            best = val_ema_loss
            ckpt_out["best_val"] = float(best)
            torch.save(ckpt_out, os.path.join(cfg["CKPT_DIR"], "best.pt"))
            print(f"  -> saved best.pt  (val_ema={val_ema_loss:.6f})")

        # Stop if we have consumed all planned steps (run_epoch broke early)
        if hasattr(trainer, "global_step") and int(trainer.global_step) >= total_steps:
            print(f"[DONE] global_step={trainer.global_step} reached "
                  f"total_steps={total_steps}. Stopping.")
            break

    try:
        if hasattr(trainer, "diag") and trainer.diag is not None:
            trainer.diag.close()
    except Exception as e:
        print("[WARN] diag.close failed:", repr(e))

    print("DONE. best_val =", best)

# %% cell 38
# --------------------------------------------------------------------------------------
# 14) Example usage
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    main_train(TRAIN_DIR, VAL_DIR, CFG)
    print("Loaded full velocity-FM windowed script. Edit paths under __main__ to train.")

