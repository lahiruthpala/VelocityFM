# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% cell 0
# ============================================
# GPU Round-Trip Reconstruction Validator (Option A) - FIXED
# Equivalent semantics to prior "all atom14" CPU validator
# ============================================

# NOTEBOOK_MAGIC: !pip -q install numpy pandas tqdm matplotlib torch

import os, glob, math, random, zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

# ----------------------------
# 0) CONFIG
# ----------------------------
from google.colab import drive
drive.mount("/content/drive")

NPZ_DIR = "/content/drive/MyDrive/af_native_dynamics_predictor/data/processed/MD_Simulation/V4_atom14_optionA_npz"
LOCAL_TRAIN = "/content/train.zip"
LOCAL_TEST = "/content/test.zip"
LOCAL_VAL = "/content/val.zip"

LOCAL_UNZIP_DIR = "/content/unzipped"
os.makedirs(LOCAL_UNZIP_DIR, exist_ok=True)

LIB_MAX_FILES   = 300
LIB_MAX_FRAMES  = 50

VAL_MAX_FILES   = None      # None => all (can be slow)
VAL_MAX_FRAMES  = None       # None => all (can be slow)
VAL_RESIDUES_PER_FRAME = None  # e.g., 64 for speed; None => all residues

PLOT_HISTS      = True
HIST_BINS       = 120
MAX_ERR_SAMPLES = 200_000  # bounded histogram sample

RANDOM_SEED     = 42
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE           = torch.float32

LIB_CACHE_PATH  = os.path.join(NPZ_DIR, "_atom14_geom_library_cache.npz")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# speed knobs (safe for this validator)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("DEVICE:", DEVICE)

# ----------------------------
# 1) Definitions
# ----------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
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

def aa_idx_to_char(aatype_i: int) -> str:
    return AA_ORDER[aatype_i] if 0 <= aatype_i < 20 else "X"

def atom14_index_map(aa: str):
    names = ATOM14_NAMES.get(aa, ["N","CA","C","O"] + [""]*10)
    mp = {}
    for i,nm in enumerate(names):
        if nm != "":
            mp[nm] = i
    return mp

def build_specs_for_aa(aa: str):
    # same spec list as CPU validator
    if aa not in ATOM14_NAMES:
        return [("O", "N","CA","C", "const", None)]

    instr = []
    instr.append(("O", "N","CA","C", "const", None))
    if aa != "G":
        instr.append(("CB", "N","C","CA", "const", None))

    if aa == "A" or aa == "G": return instr
    if aa == "S": instr.append(("OG","N","CA","CB","chi1",0.0)); return instr
    if aa == "C": instr.append(("SG","N","CA","CB","chi1",0.0)); return instr
    if aa == "T":
        instr.append(("OG1","N","CA","CB","chi1",0.0))
        instr.append(("CG2","N","CA","CB","chi1",None)); return instr
    if aa == "V":
        instr.append(("CG1","N","CA","CB","chi1",0.0))
        instr.append(("CG2","N","CA","CB","chi1",None)); return instr
    if aa == "I":
        instr.append(("CG1","N","CA","CB","chi1",0.0))
        instr.append(("CG2","N","CA","CB","chi1",None))
        instr.append(("CD1","CA","CB","CG1","chi2",0.0)); return instr
    if aa == "L":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("CD1","CA","CB","CG","chi2",0.0))
        instr.append(("CD2","CA","CB","CG","chi2",None)); return instr
    if aa == "P":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("CD","CA","CB","CG","chi2",0.0)); return instr
    if aa == "M":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("SD","CA","CB","CG","chi2",0.0))
        instr.append(("CE","CB","CG","SD","chi3",0.0)); return instr
    if aa == "K":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("CD","CA","CB","CG","chi2",0.0))
        instr.append(("CE","CB","CG","CD","chi3",0.0))
        instr.append(("NZ","CG","CD","CE","chi4",0.0)); return instr
    if aa == "R":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("CD","CA","CB","CG","chi2",0.0))
        instr.append(("NE","CB","CG","CD","chi3",0.0))
        instr.append(("CZ","CG","CD","NE","chi4",0.0))
        instr.append(("NH1","CD","NE","CZ","const",None))
        instr.append(("NH2","CD","NE","CZ","const",None)); return instr
    if aa == "D":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("OD1","CA","CB","CG","chi2",0.0))
        instr.append(("OD2","CA","CB","CG","chi2",None)); return instr
    if aa == "N":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("OD1","CA","CB","CG","chi2",0.0))
        instr.append(("ND2","CA","CB","CG","chi2",None)); return instr
    if aa == "E":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("CD","CA","CB","CG","chi2",0.0))
        instr.append(("OE1","CB","CG","CD","chi3",0.0))
        instr.append(("OE2","CB","CG","CD","chi3",None)); return instr
    if aa == "Q":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("CD","CA","CB","CG","chi2",0.0))
        instr.append(("OE1","CB","CG","CD","chi3",0.0))
        instr.append(("NE2","CB","CG","CD","chi3",None)); return instr
    if aa == "F":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("CD1","CA","CB","CG","chi2",0.0))
        instr.append(("CD2","CA","CB","CG","chi2",None))
        instr.append(("CE1","CB","CG","CD1","const",None))
        instr.append(("CE2","CB","CG","CD2","const",None))
        instr.append(("CZ","CG","CD1","CE1","const",None)); return instr
    if aa == "Y":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("CD1","CA","CB","CG","chi2",0.0))
        instr.append(("CD2","CA","CB","CG","chi2",None))
        instr.append(("CE1","CB","CG","CD1","const",None))
        instr.append(("CE2","CB","CG","CD2","const",None))
        instr.append(("CZ","CG","CD1","CE1","const",None))
        instr.append(("OH","CD1","CE1","CZ","const",None)); return instr
    if aa == "W":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("CD1","CA","CB","CG","chi2",0.0))
        instr.append(("CD2","CA","CB","CG","chi2",None))
        instr.append(("NE1","CB","CG","CD1","const",None))
        instr.append(("CE2","CG","CD2","NE1","const",None))
        instr.append(("CE3","CB","CG","CD2","const",None))
        instr.append(("CZ2","CD2","CE2","CE3","const",None))
        instr.append(("CZ3","CD2","CE3","CZ2","const",None))
        instr.append(("CH2","CE2","CZ2","CZ3","const",None)); return instr
    if aa == "H":
        instr.append(("CG","N","CA","CB","chi1",0.0))
        instr.append(("ND1","CA","CB","CG","chi2",0.0))
        instr.append(("CD2","CA","CB","CG","chi2",None))
        instr.append(("CE1","CB","CG","ND1","const",None))
        instr.append(("NE2","CG","CD2","CE1","const",None)); return instr

    return instr

# ----------------------------
# 2) Fit / Cache geometry library (same as earlier; sampled)
# ----------------------------
def wrap_to_pi_np(x): return np.arctan2(np.sin(x), np.cos(x))
def normalize_np(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def angle_np(p2, p3, p4, eps=1e-8):
    v1 = normalize_np(p2 - p3, eps)
    v2 = normalize_np(p4 - p3, eps)
    c = np.clip(np.sum(v1*v2, axis=-1), -1+1e-7, 1-1e-7)
    return np.arccos(c)

def dihedral_np(p1, p2, p3, p4, eps=1e-8):
    b0 = p1 - p2
    b1 = p3 - p2
    b2 = p4 - p3
    b1n = normalize_np(b1, eps)
    v = b0 - np.sum(b0*b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2*b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return np.arctan2(y, x)

def circ_mean_np(a):
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))

def sample_indices(T, max_frames):
    if max_frames is None or T <= max_frames:
        return np.arange(T, dtype=np.int64)
    return np.linspace(0, T-1, max_frames).astype(np.int64)

def fit_library(npz_files, max_files, max_frames_per_file):
    files = npz_files[:]
    random.shuffle(files)
    if max_files is not None:
        files = files[:max_files]

    n_local_stats = {aa: [] for aa in AA_ORDER}
    cac_len_stats = {aa: [] for aa in AA_ORDER}
    stats = {}

    for path in tqdm(files, desc="[LIB] fitting from GT"):
        d = np.load(path, allow_pickle=True)
        aatype = d["aatype"].astype(np.int64)
        A = d["atom14_pos"].astype(np.float32)
        M = d["atom14_mask"].astype(np.uint8)
        R = d["frame_R"].astype(np.float32)
        FM = d["frame_mask"].astype(np.uint8)
        tors = d["torsion_angles"].astype(np.float32)
        tors_mask = d["torsion_mask"].astype(np.uint8)

        T, N = A.shape[0], A.shape[1]
        fidx = sample_indices(T, max_frames_per_file)

        aa_chars = [aa_idx_to_char(int(aatype[i])) for i in range(N)]
        mp_list  = [atom14_index_map(aa) for aa in aa_chars]

        for fi in fidx:
            for i in range(N):
                aa = aa_chars[i]
                if aa not in AA_ORDER:  # matches CPU: skip unknowns
                    continue
                mp = mp_list[i]
                if FM[fi, i] != 1:
                    continue
                if not ("N" in mp and "CA" in mp and "C" in mp):
                    continue
                iN,iCA,iC = mp["N"], mp["CA"], mp["C"]
                if not (M[fi,i,iN] and M[fi,i,iCA] and M[fi,i,iC]):
                    continue

                Nxyz, CAxyz, Cxyz = A[fi,i,iN], A[fi,i,iCA], A[fi,i,iC]
                v = (Nxyz - CAxyz).astype(np.float32)
                nloc = (R[fi,i].T @ v).astype(np.float32)
                n_local_stats[aa].append(nloc)
                cac_len_stats[aa].append(float(np.linalg.norm(Cxyz - CAxyz)))

                instr = build_specs_for_aa(aa)
                for atom,p1n,p2n,p3n,dsrc,fixed_off in instr:
                    if atom not in mp: continue
                    ai = mp[atom]
                    if M[fi,i,ai] != 1: continue
                    if p1n not in mp or p2n not in mp or p3n not in mp: continue
                    i1,i2,i3 = mp[p1n], mp[p2n], mp[p3n]
                    if not (M[fi,i,i1] and M[fi,i,i2] and M[fi,i,i3]): continue

                    p1,p2,p3,p4 = A[fi,i,i1], A[fi,i,i2], A[fi,i,i3], A[fi,i,ai]
                    L = float(np.linalg.norm(p4-p3))
                    ang = float(angle_np(p2,p3,p4))
                    dih = float(dihedral_np(p1,p2,p3,p4))

                    key=(aa,atom)
                    if key not in stats:
                        stats[key]={"L":[], "ang":[], "dih_const":[], "dih_off":[], "dsrc":dsrc, "fixed_off":fixed_off}
                    stats[key]["L"].append(L)
                    stats[key]["ang"].append(ang)

                    if dsrc=="const":
                        stats[key]["dih_const"].append(dih)
                    else:
                        chi_k=int(dsrc.replace("chi",""))
                        chi_idx=2+chi_k
                        if tors_mask[fi,i,chi_idx]!=1:
                            stats[key]["dih_const"].append(dih)
                        else:
                            chi=float(tors[fi,i,chi_idx])
                            if fixed_off is None:
                                stats[key]["dih_off"].append(float(wrap_to_pi_np(dih-chi)))

    backbone={}
    for aa in AA_ORDER:
        nlocs=n_local_stats[aa]
        if len(nlocs)==0:
            continue
        nloc=np.stack(nlocs,0)
        backbone[aa]={
            "N_local_mean": nloc.mean(0).astype(np.float32),
            "CA_C_len_med": float(np.median(np.array(cac_len_stats[aa],dtype=np.float32)))
        }

    instr_params={}
    for (aa,atom),st in stats.items():
        L=np.array(st["L"],dtype=np.float32)
        ang=np.array(st["ang"],dtype=np.float32)
        dih_const = circ_mean_np(np.array(st["dih_const"],dtype=np.float32)) if len(st["dih_const"])>0 else None
        dih_off   = circ_mean_np(np.array(st["dih_off"],dtype=np.float32))   if len(st["dih_off"])>0 else None
        instr_params[(aa,atom)]={
            "L": float(np.median(L)),
            "ang": float(np.median(ang)),
            "dsrc": st["dsrc"],
            "dih_const": dih_const,
            "dih_off": dih_off,
            "fixed_off": st["fixed_off"],
        }
    return backbone,instr_params

def save_library_cache(path, backbone, instr_params):
    # tensors over [21,...] so aatype=20 is safe (unknown)
    N_local_mean = np.zeros((21,3), dtype=np.float32)
    CA_C_len     = np.zeros((21,), dtype=np.float32)

    fallback = "A" if "A" in backbone else next(iter(backbone.keys()))
    N_local_mean[:] = backbone[fallback]["N_local_mean"][None,:]
    CA_C_len[:]     = backbone[fallback]["CA_C_len_med"]

    for i,aa in enumerate(AA_ORDER):
        if aa in backbone:
            N_local_mean[i]=backbone[aa]["N_local_mean"]
            CA_C_len[i]=backbone[aa]["CA_C_len_med"]

    parent1 = np.full((21,14), -1, dtype=np.int16)
    parent2 = np.full((21,14), -1, dtype=np.int16)
    parent3 = np.full((21,14), -1, dtype=np.int16)

    Ltab   = np.full((21,14), np.nan, dtype=np.float32)
    Angtab = np.full((21,14), np.nan, dtype=np.float32)
    mode   = np.full((21,14), -1, dtype=np.int8)     # -1 none, 0 const, 1..4 chiK
    dihC   = np.full((21,14), np.nan, dtype=np.float32)
    dihOff = np.full((21,14), np.nan, dtype=np.float32)
    fixedM = np.zeros((21,14), dtype=np.uint8)
    fixedV = np.zeros((21,14), dtype=np.float32)

    atom_exists = np.zeros((21,14), dtype=np.uint8)
    for ai in range(21):
        aa = aa_idx_to_char(ai)
        names = ATOM14_NAMES.get(aa, ["N","CA","C","O"] + [""]*10)
        for j,nm in enumerate(names):
            if nm!="":
                atom_exists[ai,j]=1
    atom_exists[:,0:4]=1  # backbone always exists

    for ai in range(20):
        aa=AA_ORDER[ai]
        mp=atom14_index_map(aa)
        for atom,p1n,p2n,p3n,dsrc,fixed_off in build_specs_for_aa(aa):
            if atom not in mp or p1n not in mp or p2n not in mp or p3n not in mp:
                continue
            j=mp[atom]
            parent1[ai,j]=mp[p1n]
            parent2[ai,j]=mp[p2n]
            parent3[ai,j]=mp[p3n]
            key=(aa,atom)
            if key not in instr_params:
                continue
            Ltab[ai,j]=instr_params[key]["L"]
            Angtab[ai,j]=instr_params[key]["ang"]
            if dsrc=="const":
                mode[ai,j]=0
                if instr_params[key]["dih_const"] is not None:
                    dihC[ai,j]=float(instr_params[key]["dih_const"])
            else:
                k=int(dsrc.replace("chi",""))
                mode[ai,j]=k
                if instr_params[key]["dih_const"] is not None:
                    dihC[ai,j]=float(instr_params[key]["dih_const"])
                if instr_params[key]["dih_off"] is not None:
                    dihOff[ai,j]=float(instr_params[key]["dih_off"])
                if fixed_off is not None:
                    fixedM[ai,j]=1
                    fixedV[ai,j]=float(fixed_off)

    np.savez_compressed(
        path,
        N_local_mean=N_local_mean, CA_C_len=CA_C_len,
        atom_exists=atom_exists,
        parent1=parent1, parent2=parent2, parent3=parent3,
        L=Ltab, Ang=Angtab, mode=mode,
        dihC=dihC, dihOff=dihOff,
        fixedM=fixedM, fixedV=fixedV
    )

def load_library_cache(path):
    z=np.load(path, allow_pickle=True)
    return {k:z[k] for k in z.files}

# ----------------------------
# 3) Torch placement (GPU)
# ----------------------------
def wrap_to_pi_t(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

def normalize_t(v, eps=1e-8):
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)

def place_atom_t(p1,p2,p3,length,ang,dih,eps=1e-8):
    b1=p3-p2
    e1=normalize_t(b1,eps)
    b0=p2-p1
    n=torch.cross(b0,b1,dim=-1)
    nn=torch.linalg.norm(n,dim=-1,keepdim=True)
    bad=(nn[...,0]<1e-6)
    n=torch.where(bad.unsqueeze(-1), torch.tensor([0.0,0.0,1.0],device=n.device,dtype=n.dtype), n)
    en=normalize_t(n,eps)
    e2=torch.cross(en,e1,dim=-1)

    ct=torch.cos(ang); st=torch.sin(ang)
    cp=torch.cos(dih); sp=torch.sin(dih)
    v=(-ct).unsqueeze(-1)*e1 + st.unsqueeze(-1)*(cp.unsqueeze(-1)*e2 + sp.unsqueeze(-1)*en)
    return p3 + length.unsqueeze(-1)*v

# ----------------------------
# 4) GPU reconstructor (FIXED: safe gather + correct masking semantics)
# ----------------------------
class Atom14Reconstructor(torch.nn.Module):
    def __init__(self, lib_npz):
        super().__init__()
        self.register_buffer("N_local_mean", torch.tensor(lib_npz["N_local_mean"], dtype=DTYPE), persistent=False) # [21,3]
        self.register_buffer("CA_C_len",     torch.tensor(lib_npz["CA_C_len"], dtype=DTYPE),     persistent=False) # [21]
        self.register_buffer("atom_exists",  torch.tensor(lib_npz["atom_exists"].astype(np.uint8), dtype=torch.bool), persistent=False) # [21,14]

        self.register_buffer("parent1", torch.tensor(lib_npz["parent1"].astype(np.int64), dtype=torch.long), persistent=False)
        self.register_buffer("parent2", torch.tensor(lib_npz["parent2"].astype(np.int64), dtype=torch.long), persistent=False)
        self.register_buffer("parent3", torch.tensor(lib_npz["parent3"].astype(np.int64), dtype=torch.long), persistent=False)

        self.register_buffer("L",   torch.tensor(lib_npz["L"], dtype=DTYPE),   persistent=False)
        self.register_buffer("Ang", torch.tensor(lib_npz["Ang"], dtype=DTYPE), persistent=False)
        self.register_buffer("mode", torch.tensor(lib_npz["mode"].astype(np.int64), dtype=torch.long), persistent=False)

        self.register_buffer("dihC",   torch.tensor(lib_npz["dihC"], dtype=DTYPE),   persistent=False)
        self.register_buffer("dihOff", torch.tensor(lib_npz["dihOff"], dtype=DTYPE), persistent=False)
        self.register_buffer("fixedM", torch.tensor(lib_npz["fixedM"].astype(np.uint8), dtype=torch.bool), persistent=False)
        self.register_buffer("fixedV", torch.tensor(lib_npz["fixedV"], dtype=DTYPE), persistent=False)

        self.Ni, self.CAi, self.Ci, self.Oi = 0, 1, 2, 3

    @torch.no_grad()
    def forward(self, aatype, frame_R, frame_t, frame_mask, torsions, torsion_mask):
        """
        aatype: [N] long
        frame_R: [F,N,3,3]  (columns are basis vectors)
        frame_t: [F,N,3]    (CA)
        frame_mask: [F,N] bool
        torsions: [F,N,7]
        torsion_mask: [F,N,7] bool
        returns coords [F,N,14,3] and pred_mask [F,N,14] bool
        """
        device = frame_t.device
        F, N = frame_t.shape[0], frame_t.shape[1]

        aatype = aatype.to(device=device)
        # safety: detect invalid aatype early (prevents out-of-bounds indexing)
        if torch.any((aatype < 0) | (aatype > 20)):
            bad = torch.where((aatype < 0) | (aatype > 20))[0][:10].detach().cpu().numpy()
            raise ValueError(f"aatype out of [0,20] at indices {bad}")

        frame_mask = frame_mask.to(device=device).bool()
        torsion_mask = torsion_mask.to(device=device).bool()

        aa_ok = (aatype < 20)  # matches CPU semantics: unknown residues excluded

        coords = torch.zeros((F,N,14,3), device=device, dtype=frame_t.dtype)
        pmask  = torch.zeros((F,N,14),   device=device, dtype=torch.bool)

        # backbone params
        nloc = self.N_local_mean[aatype] # [N,3] (fallback exists even for 20)
        cac  = self.CA_C_len[aatype]     # [N]

        # CA
        coords[:,:,self.CAi,:] = frame_t
        pmask[:,:,self.CAi] = frame_mask & aa_ok.view(1,N)

        # C
        e1 = frame_R[:,:,:,0]  # [F,N,3]
        coords[:,:,self.Ci,:] = frame_t + e1 * cac.view(1,N,1)
        pmask[:,:,self.Ci] = frame_mask & aa_ok.view(1,N)

        # N
        nxyz = torch.einsum("fnij,nj->fni", frame_R, nloc)
        coords[:,:,self.Ni,:] = frame_t + nxyz
        pmask[:,:,self.Ni] = frame_mask & aa_ok.view(1,N)

        # IMPORTANT: do NOT pre-mark O or others as present.
        # We only set pmask for atoms we successfully place (CPU-equivalent).

        for j in range(3,14):
            # which residues should have this atom at all?
            exists_res = self.atom_exists[aatype, j] & aa_ok  # [N] bool

            p1 = self.parent1[aatype, j]
            p2 = self.parent2[aatype, j]
            p3 = self.parent3[aatype, j]

            valid_instr = (p1 >= 0) & (p2 >= 0) & (p3 >= 0) & exists_res
            if not torch.any(valid_instr):
                continue

            # FIX 1: clamp negative indices before gather to avoid CUDA asserts
            p1s = p1.clamp(min=0)
            p2s = p2.clamp(min=0)
            p3s = p3.clamp(min=0)

            p1i = p1s.view(1,N,1,1).expand(F,N,1,3)
            p2i = p2s.view(1,N,1,1).expand(F,N,1,3)
            p3i = p3s.view(1,N,1,1).expand(F,N,1,3)

            P1 = torch.gather(coords, 2, p1i).squeeze(2)
            P2 = torch.gather(coords, 2, p2i).squeeze(2)
            P3 = torch.gather(coords, 2, p3i).squeeze(2)

            p1m = torch.gather(pmask, 2, p1s.view(1,N,1).expand(F,N,1)).squeeze(2)
            p2m = torch.gather(pmask, 2, p2s.view(1,N,1).expand(F,N,1)).squeeze(2)
            p3m = torch.gather(pmask, 2, p3s.view(1,N,1).expand(F,N,1)).squeeze(2)

            L   = self.L[aatype, j].view(1,N).expand(F,N)
            Ang = self.Ang[aatype, j].view(1,N).expand(F,N)
            has_geom = torch.isfinite(L) & torch.isfinite(Ang)

            mode = self.mode[aatype, j]              # [N]
            modeF = mode.view(1,N).expand(F,N)

            # const dihedral fallback
            dih_const = self.dihC[aatype, j].view(1,N).expand(F,N)
            dih = dih_const.clone()

            # dynamic chi
            dyn = (modeF >= 1) & (modeF <= 4)
            if torch.any(dyn):
                chi_idx = (2 + mode).clamp(0,6)  # [N]
                chi = torsions.gather(2, chi_idx.view(1,N,1).expand(F,N,1)).squeeze(2)
                chi_ok = torsion_mask.gather(2, chi_idx.view(1,N,1).expand(F,N,1)).squeeze(2)

                fixedM = self.fixedM[aatype, j].view(1,N).expand(F,N)
                fixedV = self.fixedV[aatype, j].view(1,N).expand(F,N)

                off = self.dihOff[aatype, j].view(1,N).expand(F,N)
                has_off = torch.isfinite(off)

                dih_dyn = chi
                dih_dyn = torch.where(fixedM, wrap_to_pi_t(chi + fixedV), dih_dyn)
                dih_dyn = torch.where((~fixedM) & has_off, wrap_to_pi_t(chi + off), dih_dyn)

                # use dynamic only if chi is defined; else keep const
                dih = torch.where(dyn & chi_ok, dih_dyn, dih)

            # place condition
            cond = (frame_mask &
                    valid_instr.view(1,N).expand(F,N) &
                    p1m & p2m & p3m &
                    has_geom &
                    torch.isfinite(dih))

            if not torch.any(cond):
                continue

            P4 = place_atom_t(P1,P2,P3,L,Ang,dih)

            coords[:,:,j,:] = torch.where(cond.unsqueeze(-1), P4, coords[:,:,j,:])
            pmask[:,:,j] = pmask[:,:,j] | cond

        return coords, pmask

# ----------------------------
# 5) Load NPZ list
# ----------------------------

def copy_with_progress(src, dst):
    size = os.path.getsize(src)
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        with tqdm(total=size, unit='B', unit_scale=True, desc=f"Copying {os.path.basename(src)}") as pbar:
            while True:
                chunk = fsrc.read(1024 * 1024)
                if not chunk: break
                fdst.write(chunk)
                pbar.update(len(chunk))

def unzip_if_needed(zip_path, out_dir):
    marker_name = f".unzipped_{os.path.basename(zip_path)}"
    marker = os.path.join(out_dir, marker_name)
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

# copy_with_progress(f"{NPZ_DIR}/train.zip", LOCAL_TRAIN)
# copy_with_progress(f"{NPZ_DIR}/test.zip", LOCAL_TEST)
# copy_with_progress(f"{NPZ_DIR}/val.zip", LOCAL_VAL)

# unzip_if_needed(LOCAL_TRAIN, LOCAL_UNZIP_DIR)
# unzip_if_needed(LOCAL_TEST, LOCAL_UNZIP_DIR)
# unzip_if_needed(LOCAL_VAL, LOCAL_UNZIP_DIR)

npz_files = sorted(glob.glob(os.path.join(LOCAL_UNZIP_DIR, "*.npz")))
assert len(npz_files) > 0, f"No NPZ files found: {LOCAL_UNZIP_DIR}"
print("NPZ count:", len(npz_files))

# ----------------------------
# 6) Fit/load library cache
# ----------------------------
if os.path.exists(LIB_CACHE_PATH):
    print(f"[LIB] Loading cached geometry: {LIB_CACHE_PATH}")
    lib_npz = load_library_cache(LIB_CACHE_PATH)
else:
    print("[LIB] Cache not found; fitting (sampled)...")
    backbone, instr_params = fit_library(npz_files, LIB_MAX_FILES, LIB_MAX_FRAMES)
    print("[LIB] Saving cache...")
    save_library_cache(LIB_CACHE_PATH, backbone, instr_params)
    lib_npz = load_library_cache(LIB_CACHE_PATH)

recon = Atom14Reconstructor(lib_npz).to(DEVICE).eval()

# ----------------------------
# 7) Validation loop (GPU)
# ----------------------------
def validate_files(files, max_frames, max_res_per_frame):
    rows = []
    all_err, bb_err, sc_err = [], [], []

    for path in tqdm(files, desc="[VAL] files"):
        d = np.load(path, allow_pickle=True)

        aatype_np = d["aatype"].astype(np.int64)
        if aatype_np.min() < 0 or aatype_np.max() > 20:
            raise ValueError(f"aatype out of [0,20] in file: {path}  min={aatype_np.min()} max={aatype_np.max()}")

        aatype = torch.tensor(aatype_np, dtype=torch.long, device=DEVICE)  # [N]

        gt_pos = d["atom14_pos"].astype(np.float32)       # [T,N,14,3]
        gt_msk = d["atom14_mask"].astype(np.uint8)        # [T,N,14]
        frame_R = d["frame_R"].astype(np.float32)         # [T,N,3,3]
        frame_t = d["frame_t"].astype(np.float32)         # [T,N,3]
        frame_mask = d["frame_mask"].astype(np.uint8)     # [T,N]
        tors = d["torsion_angles"].astype(np.float32)     # [T,N,7]
        tors_mask = d["torsion_mask"].astype(np.uint8)    # [T,N,7]

        T, N = gt_pos.shape[0], gt_pos.shape[1]
        fidx = sample_indices(T, max_frames)
        all_res_idx = np.arange(N, dtype=np.int64)

        se_sum_all = 0.0; cnt_all = 0
        se_sum_bb  = 0.0; cnt_bb  = 0
        se_sum_sc  = 0.0; cnt_sc  = 0

        for fi in fidx:
            if max_res_per_frame is not None:
                valid_res = np.where(frame_mask[fi].astype(bool))[0]
                if valid_res.size > max_res_per_frame:
                    ridx = np.random.choice(valid_res, max_res_per_frame, replace=False)
                else:
                    ridx = valid_res
            else:
                ridx = all_res_idx

            # frame tensors on GPU (F=1)
            gt = torch.tensor(gt_pos[fi, ridx], dtype=DTYPE, device=DEVICE).unsqueeze(0)
            gm = torch.tensor(gt_msk[fi, ridx].astype(np.bool_), dtype=torch.bool, device=DEVICE).unsqueeze(0)
            R  = torch.tensor(frame_R[fi, ridx], dtype=DTYPE, device=DEVICE).unsqueeze(0)
            t  = torch.tensor(frame_t[fi, ridx], dtype=DTYPE, device=DEVICE).unsqueeze(0)
            fm = torch.tensor(frame_mask[fi, ridx].astype(np.bool_), dtype=torch.bool, device=DEVICE).unsqueeze(0)
            ta = torch.tensor(tors[fi, ridx], dtype=DTYPE, device=DEVICE).unsqueeze(0)
            tm = torch.tensor(tors_mask[fi, ridx].astype(np.bool_), dtype=torch.bool, device=DEVICE).unsqueeze(0)
            aa = aatype[ridx]

            pred, pm = recon(aa, R, t, fm, ta, tm)

            mask = gm & pm & fm.unsqueeze(-1)  # CPU-equivalent masking
            diff = pred - gt
            se = (diff*diff).sum(dim=-1)  # [1,Nsel,14]

            se_sum_all += float(se[mask].sum().item())
            cnt_all    += int(mask.sum().item())

            bb_mask = mask.clone()
            bb_mask[..., 4:] = False
            se_sum_bb += float(se[bb_mask].sum().item())
            cnt_bb    += int(bb_mask.sum().item())

            sc_mask = mask.clone()
            sc_mask[..., :4] = False
            se_sum_sc += float(se[sc_mask].sum().item())
            cnt_sc    += int(sc_mask.sum().item())

            # bounded histogram sampling
            if PLOT_HISTS and MAX_ERR_SAMPLES > 0:
                def sample_err(dst_list, m):
                    if len(dst_list) >= MAX_ERR_SAMPLES:
                        return
                    e = torch.sqrt(se[m]).detach().flatten()
                    if e.numel() == 0:
                        return
                    take = min(2000, MAX_ERR_SAMPLES - len(dst_list), e.numel())
                    idx = torch.randperm(e.numel(), device=e.device)[:take]
                    dst_list.extend(e[idx].float().cpu().tolist())
                sample_err(all_err, mask)
                sample_err(bb_err, bb_mask)
                sample_err(sc_err, sc_mask)

        rms_all = math.sqrt(se_sum_all / max(cnt_all, 1))
        rms_bb  = math.sqrt(se_sum_bb  / max(cnt_bb, 1)) if cnt_bb > 0 else float("nan")
        rms_sc  = math.sqrt(se_sum_sc  / max(cnt_sc, 1)) if cnt_sc > 0 else float("nan")

        rows.append({
            "file": os.path.basename(path),
            "N": int(N),
            "T_total": int(T),
            "T_checked": int(len(fidx)),
            "rmsd_all_atom14": float(rms_all),
            "rmsd_backbone": float(rms_bb),
            "rmsd_sidechain": float(rms_sc),
            "atom_count_used": int(cnt_all),
        })

    return pd.DataFrame(rows), np.array(all_err,np.float32), np.array(bb_err,np.float32), np.array(sc_err,np.float32)

# choose validation set
val_files = npz_files[:]
random.shuffle(val_files)
if VAL_MAX_FILES is not None:
    val_files = val_files[:VAL_MAX_FILES]

print(f"[VAL] Files={len(val_files)}  frames/file={VAL_MAX_FRAMES}  residues/frame={VAL_RESIDUES_PER_FRAME}")
val_df, all_err, bb_err, sc_err = validate_files(val_files, VAL_MAX_FRAMES, VAL_RESIDUES_PER_FRAME)

print("\nPer-file RMSD summary (Å):")
print(val_df.describe(include="all").to_string())

print("\nWorst files by all-atom RMSD:")
print(val_df.sort_values("rmsd_all_atom14", ascending=False).head(10).to_string(index=False))

OUT_CSV = os.path.join(NPZ_DIR, "_roundtrip_recon_report_gpu.csv")
val_df.to_csv(OUT_CSV, index=False)
print("\nSaved:", OUT_CSV)

if PLOT_HISTS:
    def hist_plot(vals, title, xlabel):
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        plt.figure()
        plt.hist(vals, bins=HIST_BINS)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.show()

    hist_plot(all_err, "Atom14 per-atom reconstruction error distribution (Å) [sampled]", "abs error (Å)")
    hist_plot(bb_err,  "Backbone (N/CA/C/O) per-atom error distribution (Å) [sampled]", "abs error (Å)")
    hist_plot(sc_err,  "Sidechain per-atom error distribution (Å) [sampled]", "abs error (Å)")

