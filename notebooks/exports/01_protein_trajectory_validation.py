# Auto-exported from a Jupyter/Colab notebook.
# This is a searchable reference export, not guaranteed to run as one script.
# The original notebook remains the execution source of truth.

# %% [markdown] cell 0
# # 🧬 Protein Trajectory Validation: Generated vs Ground Truth
# 
# **Inputs:**
# - **Ground Truth**: `.xtc` trajectory + reference `.pdb` topology file
# - **Generated**: Multi-model `.pdb` file (one MODEL block per sample)
# 
# ---
# ## 📋 Table of Contents
# 1. [Setup & Installation](#section-1)
# 2. [Load Trajectories](#section-2)
# 3. [Section A — Structural Similarity (TM-score, RMSD, GDT)](#section-a)
# 4. [Section B — Geometry & Validity (Bonds, Clashes)](#section-b)
# 5. [Section C — Secondary Structure & Ramachandran](#section-c)
# 6. [Section D — Ensemble Diversity & Distance Maps](#section-d)
# 7. [Summary Dashboard](#summary)
# 
# ---

# %% [markdown] cell 1
# ## 1. Setup & Installation

# %% cell 2
# Install required packages
# NOTEBOOK_MAGIC: !pip install -q "OpenMM>=8.2" mdtraj tmtools biopython matplotlib seaborn numpy scipy
print("✅ All packages installed.")

# %% cell 3
import numpy as np
import openmm
import openmm.unit as openmm_unit
import mdtraj as md
import mdtraj.utils.unit as md_unit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from Bio.SVDSuperimposer import SVDSuperimposer
import warnings
import os

# Repair mdtraj's OpenMM unit binding for this runtime
md_unit.openmm_unit = openmm_unit

warnings.filterwarnings('ignore')

# ── Plot style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 110,
})

GT_COLOR  = '#2E86AB'
GEN_COLOR = '#E84855'
ACC_COLOR = '#3BB273'

print("✅ Imports complete.")
print("✅ mdtraj ↔ OpenMM unit binding repaired.")

# %% [markdown] cell 4
# ## 2. Load Trajectories
# 
# **Upload your files:**
# - `gt_topology.pdb` — reference PDB for the XTC ground truth
# - `gt_trajectory.xtc` — ground truth MD trajectory
# - `generated.pdb` — multi-MODEL PDB file from your model

# %% cell 5
from google.colab import drive  # type: ignore
drive.mount("/content/drive")

DRIVE_BASE = "/content/drive/MyDrive/af_native_dynamics_predictor"
GT_DIR = f"{DRIVE_BASE}/data/raw/MD_Simulation/proteins"

# %% cell 6
# ── ✏️  SET YOUR FILE PATHS HERE ────────────────────────────────────────────
PDB_ID = "1qw2_A"
TRAINING_RUN = "Traning_192"
MODEL_ROOT= os.path.join(DRIVE_BASE, "models", "Model_T4_V6")                  # model version root
INF_DIR   = os.path.join(MODEL_ROOT, TRAINING_RUN,"inference_outputs")

GT_PDB_PATH  = f'{GT_DIR}/{PDB_ID}/{PDB_ID}.pdb'   # reference topology for XTC
GT_XTC_PATH  = f'{GT_DIR}/{PDB_ID}/{PDB_ID}_R1.xtc' # ground truth trajectory
GEN_PDB_PATH = f'{INF_DIR}/1qw2_A_traj_128f.pdb'     # your multi-MODEL generated file

# Subsample GT frames for speed (set to None to use all frames)
GT_MAX_FRAMES = 128
# ────────────────────────────────────────────────────────────────────────────

# %% cell 7
# ── 🔬 DIAGNOSTICS — Run this first if you get atom count errors ─────────────
import struct

def count_xtc_atoms(xtc_path):
    """Read atom count from XTC header without loading the whole file."""
    with open(xtc_path, 'rb') as f:
        f.read(4)   # magic
        n_atoms = struct.unpack('>i', f.read(4))[0]
    return n_atoms

def count_pdb_atoms(pdb_path, model_only=True):
    """Count ATOM/HETATM lines in the first MODEL block of a PDB."""
    count = 0
    in_model = False
    found_model = False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('MODEL'):
                if found_model and model_only:
                    break  # stop after first model
                in_model = True
                found_model = True
            if line.startswith('ENDMDL') and model_only:
                break
            if line.startswith(('ATOM', 'HETATM')):
                count += 1
    return count

xtc_atoms = count_xtc_atoms(GT_XTC_PATH)
pdb_atoms = count_pdb_atoms(GT_PDB_PATH)
gen_atoms = count_pdb_atoms(GEN_PDB_PATH)

print("=" * 55)
print(" TOPOLOGY DIAGNOSTICS")
print("=" * 55)
print(f"  GT XTC  atom count : {xtc_atoms:>6}")
print(f"  GT PDB  atom count : {pdb_atoms:>6}")
print(f"  Gen PDB atom count : {gen_atoms:>6}  (first MODEL)")
print()

if xtc_atoms == pdb_atoms:
    print("✅ GT XTC and PDB topology match — safe to load directly.")
else:
    print(f"⚠️  MISMATCH: XTC has {xtc_atoms} atoms, topology PDB has {pdb_atoms} atoms.")
    print()
    print("   Likely cause & fix:")
    if xtc_atoms > pdb_atoms:
        print("   → Your topology PDB is probably Cα-only or backbone-only.")
        print("   → Use the FULL all-atom or all-heavy-atom PDB as topology.")
        print("   → If you only have Cα PDB, enable AUTO_FIX below.")
    else:
        print("   → Your PDB has more atoms than the XTC (check for waters/ions).")
        print("   → Enable AUTO_FIX below to strip to matching atoms.")

print("=" * 55)

# %% cell 8
# NOTEBOOK_MAGIC: !pip install pdbfixer -q

from pdbfixer import PDBFixer
from openmm.app import PDBFile

fixer = PDBFixer(filename=GT_PDB_PATH)
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()        # adds N, C, O, CB, sidechains
fixer.addMissingHydrogens(7.0) # optional — skip if you want heavy atoms only

with open(GT_PDB_PATH, 'w') as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f)

print("Saved gt_topology_full.pdb")

# %% cell 9
# ── Load Ground Truth (XTC + PDB) ───────────────────────────────────────────
# AUTO_FIX: if True, the notebook will try strategies to reconcile atom mismatches
AUTO_FIX = True

def load_gt_robust(xtc_path, pdb_path, auto_fix=True):
    """Load XTC with automatic topology mismatch recovery."""

    # ── Strategy 0: direct load ──────────────────────────────────────────────
    try:
        traj = md.load(xtc_path, top=pdb_path)
        print("  Direct load succeeded.")
        return traj
    except ValueError as e:
        if not auto_fix:
            raise
        print(f"  Direct load failed: {e}")
        print("  Trying AUTO_FIX strategies...")

    xtc_n = count_xtc_atoms(xtc_path)
    pdb_n = count_pdb_atoms(pdb_path)

    # ── Strategy 1: XTC > PDB  →  PDB is stripped; try loading raw XTC with
    #    a gro/pdb generated from XTC first frame via mdtraj ─────────────────
    if xtc_n > pdb_n:
        print(f"  XTC({xtc_n}) > PDB({pdb_n}): topology is missing atoms.")
        print("  Trying: load XTC first frame directly as PDB topology...")
        try:
            # Use mdtraj to load raw XTC without topology — requires a GRO or
            # matching full PDB. Try building a minimal topology from atom count.
            import tempfile, shutil
            # Attempt with mdtraj's XTCTrajectoryFile to get first frame
            from mdtraj.formats import XTCTrajectoryFile
            with XTCTrajectoryFile(xtc_path) as xf:
                xyz_first, _, _, _ = xf.read(n_frames=1)
            print(f"  XTC first frame shape: {xyz_first.shape}")
            print("  ❌ Cannot auto-create topology from atom count alone.")
            print("  ➡  Please provide the correct full-atom topology PDB.")
            print("     (e.g. the PDB you used to run your MD simulation)")
            raise ValueError("Topology PDB has fewer atoms than XTC. Supply the correct topology.")
        except Exception as e2:
            raise ValueError(
                f"AUTO_FIX failed. The topology PDB ({pdb_n} atoms) does not match "
                f"the XTC ({xtc_n} atoms).\n"
                "SOLUTION: Use the original full-atom PDB (or GRO) that was used "
                "to run the MD simulation as your GT_PDB_PATH."
            ) from e2

    # ── Strategy 2: PDB > XTC  →  PDB has extra atoms (waters/ions/H) ────────
    if pdb_n > xtc_n:
        print(f"  PDB({pdb_n}) > XTC({xtc_n}): topology has extra atoms. Stripping to match...")
        pdb_top = md.load(pdb_path)
        # Try heavy protein atoms first, then all protein atoms
        for selection in [
            'protein and not element H',
            'protein',
            'not (resname HOH or resname WAT or resname SOL or resname NA or resname CL)',
        ]:
            sel_idx = pdb_top.topology.select(selection)
            if len(sel_idx) == xtc_n:
                print(f"  ✅ Selection '{selection}' gives {len(sel_idx)} atoms — matches XTC.")
                stripped_top = pdb_top.atom_slice(sel_idx)
                tmp_top = '/tmp/_stripped_topology.pdb'
                stripped_top.save_pdb(tmp_top)
                traj = md.load(xtc_path, top=tmp_top)
                return traj
        print(f"  No simple selection matched {xtc_n} atoms. Trying brute-force index slice...")
        traj = md.load(xtc_path, top=pdb_path, atom_indices=list(range(xtc_n)))
        return traj

    raise ValueError("Unexpected topology mismatch — could not auto-fix.")


# ── Load GT ─────────────────────────────────────────────────────────────────
print("Loading Ground Truth trajectory...")
gt_traj = load_gt_robust(GT_XTC_PATH, GT_PDB_PATH, auto_fix=AUTO_FIX)

if GT_MAX_FRAMES and len(gt_traj) > GT_MAX_FRAMES:
    stride = max(1, len(gt_traj) // GT_MAX_FRAMES)
    gt_traj = gt_traj[::stride]
    print(f"  Subsampled to {len(gt_traj)} frames (stride={stride})")

# Keep only protein heavy atoms for analysis
prot_sel = gt_traj.topology.select('protein and not element H')
if len(prot_sel) > 0:
    gt_traj = gt_traj.atom_slice(prot_sel)
print(f"  GT: {len(gt_traj)} frames | {gt_traj.n_residues} residues | {gt_traj.n_atoms} atoms")

# ── Load Generated (multi-MODEL PDB) ────────────────────────────────────────
print("\nLoading Generated trajectory...")
gen_traj = md.load(GEN_PDB_PATH)
gen_prot_sel = gen_traj.topology.select('protein and not element H')
if len(gen_prot_sel) > 0:
    gen_traj = gen_traj.atom_slice(gen_prot_sel)
print(f"  Gen: {len(gen_traj)} frames | {gen_traj.n_residues} residues | {gen_traj.n_atoms} atoms")

# ── Align on shared residues ─────────────────────────────────────────────────
n_res = min(gt_traj.n_residues, gen_traj.n_residues)
print(f"\nShared residues for comparison: {n_res}")
if gt_traj.n_residues != gen_traj.n_residues:
    print(f"  ⚠️  GT has {gt_traj.n_residues} residues, Gen has {gen_traj.n_residues}. "
          f"Using first {n_res} residues.")

# ── Extract Cα positions (nm → Å) ───────────────────────────────────────────
def get_ca_positions(traj, max_res=None):
    ca_idx = traj.topology.select('name CA')
    if max_res:
        ca_idx = ca_idx[:max_res]
    return traj.xyz[:, ca_idx, :] * 10.0  # nm → Å

gt_ca  = get_ca_positions(gt_traj,  n_res)
gen_ca = get_ca_positions(gen_traj, n_res)

# GT reference = mean structure
gt_ref_ca = np.mean(gt_ca, axis=0)  # (n_res, 3)

print(f"\nCα array shapes — GT: {gt_ca.shape}, Gen: {gen_ca.shape}")
print("✅ Trajectories loaded successfully. Ready for analysis.")

# %% [markdown] cell 10
# ---
# ## Section A — Structural Similarity
# ### Metrics: TM-score, Aligned RMSD, GDT-TS, Per-residue RMSD

# %% cell 11
# ── Utility functions ────────────────────────────────────────────────────────

def kabsch_rmsd(P, Q):
    """Kabsch-aligned RMSD between two Cα sets (N,3)."""
    sup = SVDSuperimposer()
    sup.set(Q, P)
    sup.run()
    return float(sup.get_rms())

def per_residue_rmsd(P, Q):
    """Per-residue RMSD after Kabsch alignment."""
    sup = SVDSuperimposer()
    sup.set(Q, P)
    sup.run()
    P_rot = sup.get_transformed()
    return np.linalg.norm(P_rot - Q, axis=-1)

def tm_score(P, Q):
    """Approximate TM-score (Cα only)."""
    L = len(Q)
    d0 = 1.24 * (L - 15) ** (1/3) - 1.8 if L > 21 else 0.5
    sup = SVDSuperimposer()
    sup.set(Q, P)
    sup.run()
    P_rot = sup.get_transformed()
    d = np.linalg.norm(P_rot - Q, axis=-1)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))

def gdt_score(P, Q, thresholds):
    """GDT score at given distance thresholds (Å)."""
    sup = SVDSuperimposer()
    sup.set(Q, P)
    sup.run()
    P_rot = sup.get_transformed()
    d = np.linalg.norm(P_rot - Q, axis=-1)
    return np.mean([np.mean(d <= t) for t in thresholds])

def gdt_ts(P, Q): return gdt_score(P, Q, [1.0, 2.0, 4.0, 8.0])
def gdt_ha(P, Q): return gdt_score(P, Q, [0.5, 1.0, 2.0, 4.0])

print("✅ Structural similarity helpers defined.")

# %% cell 12
# ── Compute metrics for every GT and Generated frame vs GT reference ─────────
print("Computing structural metrics (this may take a minute)...")

gt_rmsds, gt_tms, gt_gdt_ts_vals, gt_gdt_ha_vals = [], [], [], []
for frame in gt_ca:
    gt_rmsds.append(kabsch_rmsd(frame, gt_ref_ca))
    gt_tms.append(tm_score(frame, gt_ref_ca))
    gt_gdt_ts_vals.append(gdt_ts(frame, gt_ref_ca))
    gt_gdt_ha_vals.append(gdt_ha(frame, gt_ref_ca))

gen_rmsds, gen_tms, gen_gdt_ts_vals, gen_gdt_ha_vals = [], [], [], []
for frame in gen_ca:
    gen_rmsds.append(kabsch_rmsd(frame, gt_ref_ca))
    gen_tms.append(tm_score(frame, gt_ref_ca))
    gen_gdt_ts_vals.append(gdt_ts(frame, gt_ref_ca))
    gen_gdt_ha_vals.append(gdt_ha(frame, gt_ref_ca))

# Per-residue RMSD (mean over all frames)
gt_perres  = np.mean([per_residue_rmsd(f, gt_ref_ca) for f in gt_ca], axis=0)
gen_perres = np.mean([per_residue_rmsd(f, gt_ref_ca) for f in gen_ca], axis=0)

print(f"GT   — mean TM-score: {np.mean(gt_tms):.3f}  |  mean RMSD: {np.mean(gt_rmsds):.2f} Å")
print(f"Gen  — mean TM-score: {np.mean(gen_tms):.3f}  |  mean RMSD: {np.mean(gen_rmsds):.2f} Å")

# %% cell 13
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Section A — Structural Similarity', fontsize=15, fontweight='bold', y=1.01)

# ── A1: TM-score histogram ───────────────────────────────────────────────────
ax = axes[0, 0]
ax.hist(gt_tms,  bins=30, alpha=0.6, color=GT_COLOR,  label='Ground Truth', density=True)
ax.hist(gen_tms, bins=30, alpha=0.6, color=GEN_COLOR, label='Generated',    density=True)
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.2, label='Fold threshold (0.5)')
ax.set_xlabel('TM-score vs GT reference')
ax.set_ylabel('Density')
ax.set_title('A1 — TM-score Distribution')
ax.legend()
ax.text(0.97, 0.95, f"GT μ={np.mean(gt_tms):.3f}\nGen μ={np.mean(gen_tms):.3f}",
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# ── A2: RMSD histogram ──────────────────────────────────────────────────────
ax = axes[0, 1]
ax.hist(gt_rmsds,  bins=30, alpha=0.6, color=GT_COLOR,  label='Ground Truth', density=True)
ax.hist(gen_rmsds, bins=30, alpha=0.6, color=GEN_COLOR, label='Generated',    density=True)
ax.set_xlabel('Cα RMSD vs GT reference (Å)')
ax.set_ylabel('Density')
ax.set_title('A2 — Cα RMSD Distribution')
ax.legend()
ax.text(0.97, 0.95, f"GT μ={np.mean(gt_rmsds):.2f} Å\nGen μ={np.mean(gen_rmsds):.2f} Å",
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# ── A3: GDT-TS vs GDT-HA bar chart ──────────────────────────────────────────
ax = axes[1, 0]
labels = ['GDT-TS', 'GDT-HA']
gt_vals  = [np.mean(gt_gdt_ts_vals),  np.mean(gt_gdt_ha_vals)]
gen_vals = [np.mean(gen_gdt_ts_vals), np.mean(gen_gdt_ha_vals)]
x = np.arange(len(labels))
w = 0.35
bars1 = ax.bar(x - w/2, gt_vals,  w, color=GT_COLOR,  label='Ground Truth', alpha=0.85)
bars2 = ax.bar(x + w/2, gen_vals, w, color=GEN_COLOR, label='Generated',    alpha=0.85)
ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=9)
ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('A3 — GDT-TS & GDT-HA')
ax.legend()

# ── A4: Per-residue RMSD line plot ──────────────────────────────────────────
ax = axes[1, 1]
res_ids = np.arange(1, n_res + 1)
ax.plot(res_ids, gt_perres,  color=GT_COLOR,  alpha=0.8, linewidth=1.5, label='Ground Truth')
ax.plot(res_ids, gen_perres, color=GEN_COLOR, alpha=0.8, linewidth=1.5, label='Generated')
ax.fill_between(res_ids, gen_perres, alpha=0.15, color=GEN_COLOR)
ax.set_xlabel('Residue index')
ax.set_ylabel('Mean RMSD (Å)')
ax.set_title('A4 — Per-residue RMSD')
ax.legend()

plt.tight_layout()
plt.savefig('section_A_structural_similarity.png', bbox_inches='tight', dpi=150)
plt.show()
print("Section A saved.")

# %% [markdown] cell 14
# ---
# ## Section B — Geometry & Validity
# ### Metrics: Cα–Cα bond deviation, valid bond %, steric clashes

# %% cell 15
IDEAL_CA_CA = 3.8  # Å
CA_CA_TOL   = 0.1  # Å
CLASH_TOL   = 1.5  # Å

def ca_ca_stats(ca_pos):
    """Given (N,3) Cα positions, return bond devs and clash info."""
    bonds = np.linalg.norm(ca_pos[1:] - ca_pos[:-1], axis=-1)
    dev   = np.abs(bonds - IDEAL_CA_CA)
    valid = np.mean(bonds < (IDEAL_CA_CA + CA_CA_TOL))
    # steric clashes — all pairwise Cα distances
    dists = cdist(ca_pos, ca_pos)
    upper = dists[np.triu_indices_from(dists, k=2)]  # skip i,i+1 bonded pairs
    clash_pct = np.mean(upper < CLASH_TOL)
    return np.mean(dev), valid, clash_pct, bonds

def batch_geometry(ca_frames):
    devs, valids, clashes, all_bonds = [], [], [], []
    for frame in ca_frames:
        d, v, c, bonds = ca_ca_stats(frame)
        devs.append(d); valids.append(v); clashes.append(c)
        all_bonds.extend(bonds.tolist())
    return np.array(devs), np.array(valids), np.array(clashes), np.array(all_bonds)

print("Computing geometry metrics...")
gt_devs,  gt_valids,  gt_clashes,  gt_bonds  = batch_geometry(gt_ca)
gen_devs, gen_valids, gen_clashes, gen_bonds = batch_geometry(gen_ca)
print(f"GT   — mean bond dev: {np.mean(gt_devs):.3f} Å | valid%: {np.mean(gt_valids)*100:.1f}% | clash%: {np.mean(gt_clashes)*100:.2f}%")
print(f"Gen  — mean bond dev: {np.mean(gen_devs):.3f} Å | valid%: {np.mean(gen_valids)*100:.1f}% | clash%: {np.mean(gen_clashes)*100:.2f}%")

# %% cell 16
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Section B — Geometry & Validity', fontsize=15, fontweight='bold', y=1.01)

# ── B1: Cα–Cα bond length distribution ─────────────────────────────────────
ax = axes[0, 0]
ax.hist(gt_bonds,  bins=60, alpha=0.6, color=GT_COLOR,  label='Ground Truth', density=True)
ax.hist(gen_bonds, bins=60, alpha=0.6, color=GEN_COLOR, label='Generated',    density=True)
ax.axvline(IDEAL_CA_CA, color='black', linestyle='--', linewidth=1.5, label=f'Ideal ({IDEAL_CA_CA} Å)')
ax.set_xlabel('Cα–Cα bond length (Å)')
ax.set_ylabel('Density')
ax.set_title('B1 — Cα–Cα Bond Length Distribution')
ax.legend()

# ── B2: Bond deviation violin ────────────────────────────────────────────────
ax = axes[0, 1]
vp = ax.violinplot([gt_devs, gen_devs], positions=[1, 2], showmedians=True, showextrema=True)
vp['bodies'][0].set_facecolor(GT_COLOR);  vp['bodies'][0].set_alpha(0.7)
vp['bodies'][1].set_facecolor(GEN_COLOR); vp['bodies'][1].set_alpha(0.7)
for pc in ['cmedians','cbars','cmins','cmaxes']:
    vp[pc].set_edgecolor('black')
ax.set_xticks([1, 2])
ax.set_xticklabels(['Ground Truth', 'Generated'])
ax.set_ylabel('Mean Cα–Cα Deviation (Å)')
ax.set_title('B2 — Bond Deviation per Frame')

# ── B3: Valid bond % bar ─────────────────────────────────────────────────────
ax = axes[1, 0]
categories = ['Valid Bond %', 'Clash-Free %']
gt_vals  = [np.mean(gt_valids)*100,  (1-np.mean(gt_clashes))*100]
gen_vals = [np.mean(gen_valids)*100, (1-np.mean(gen_clashes))*100]
x = np.arange(len(categories))
w = 0.35
b1 = ax.bar(x - w/2, gt_vals,  w, color=GT_COLOR,  label='Ground Truth', alpha=0.85)
b2 = ax.bar(x + w/2, gen_vals, w, color=GEN_COLOR, label='Generated',    alpha=0.85)
ax.bar_label(b1, fmt='%.1f%%', padding=3, fontsize=9)
ax.bar_label(b2, fmt='%.1f%%', padding=3, fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 115)
ax.set_ylabel('Percentage (%)')
ax.set_title('B3 — Bond Validity & Clash-Free Percentage')
ax.legend()

# ── B4: Steric clash % histogram ─────────────────────────────────────────────
ax = axes[1, 1]
ax.hist(gt_clashes*100,  bins=25, alpha=0.6, color=GT_COLOR,  label='Ground Truth', density=True)
ax.hist(gen_clashes*100, bins=25, alpha=0.6, color=GEN_COLOR, label='Generated',    density=True)
ax.set_xlabel('Cα Steric Clash % per frame')
ax.set_ylabel('Density')
ax.set_title('B4 — Steric Clash Distribution')
ax.legend()

plt.tight_layout()
plt.savefig('section_B_geometry_validity.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Section B saved.")

# %% [markdown] cell 17
# ---
# ## Section C — Secondary Structure & Ramachandran
# ### Metrics: DSSP SS%, radius of gyration, backbone dihedral φ/ψ angles

# %% cell 18
print("Computing secondary structure (DSSP)...")

def compute_ss(traj):
    dssp = md.compute_dssp(traj, simplified=True)
    helix  = np.mean(dssp == 'H', axis=1)
    strand = np.mean(dssp == 'E', axis=1)
    coil   = np.mean(dssp == 'C', axis=1)
    return helix, strand, coil

gt_helix,  gt_strand,  gt_coil  = compute_ss(gt_traj)
gen_helix, gen_strand, gen_coil = compute_ss(gen_traj)

# Radius of gyration
gt_rg  = md.compute_rg(gt_traj)  * 10.0  # nm → Å
gen_rg = md.compute_rg(gen_traj) * 10.0

# Backbone dihedrals
print("Computing backbone dihedrals...")
_, gt_phi_vals  = md.compute_phi(gt_traj)
_, gt_psi_vals  = md.compute_psi(gt_traj)
_, gen_phi_vals = md.compute_phi(gen_traj)
_, gen_psi_vals = md.compute_psi(gen_traj)

gt_phi_flat  = np.degrees(gt_phi_vals.flatten())
gt_psi_flat  = np.degrees(gt_psi_vals.flatten())
gen_phi_flat = np.degrees(gen_phi_vals.flatten())
gen_psi_flat = np.degrees(gen_psi_vals.flatten())

print(f"GT  — mean helix: {np.mean(gt_helix)*100:.1f}%  strand: {np.mean(gt_strand)*100:.1f}%  Rg: {np.mean(gt_rg):.2f} Å")
print(f"Gen — mean helix: {np.mean(gen_helix)*100:.1f}%  strand: {np.mean(gen_strand)*100:.1f}%  Rg: {np.mean(gen_rg):.2f} Å")

# %% cell 19
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Section C — Secondary Structure & Ramachandran', fontsize=15, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

# ── C1: SS composition bar chart ────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ss_labels = ['Helix', 'Strand', 'Coil']
gt_ss  = [np.mean(gt_helix)*100,  np.mean(gt_strand)*100,  np.mean(gt_coil)*100]
gen_ss = [np.mean(gen_helix)*100, np.mean(gen_strand)*100, np.mean(gen_coil)*100]
x = np.arange(len(ss_labels))
w = 0.35
b1 = ax.bar(x - w/2, gt_ss,  w, color=GT_COLOR,  alpha=0.85, label='GT')
b2 = ax.bar(x + w/2, gen_ss, w, color=GEN_COLOR, alpha=0.85, label='Gen')
ax.bar_label(b1, fmt='%.1f%%', padding=3, fontsize=8)
ax.bar_label(b2, fmt='%.1f%%', padding=3, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(ss_labels)
ax.set_ylim(0, max(max(gt_ss), max(gen_ss)) * 1.25)
ax.set_ylabel('%'); ax.set_title('C1 — SS Composition'); ax.legend()

# ── C2: Helix % distribution ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.hist(gt_helix*100,  bins=25, alpha=0.6, color=GT_COLOR,  density=True, label='GT')
ax.hist(gen_helix*100, bins=25, alpha=0.6, color=GEN_COLOR, density=True, label='Gen')
ax.set_xlabel('Helix % per frame')
ax.set_ylabel('Density')
ax.set_title('C2 — Helix Content Distribution')
ax.legend()

# ── C3: Radius of gyration ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
ax.hist(gt_rg,  bins=30, alpha=0.6, color=GT_COLOR,  density=True, label='GT')
ax.hist(gen_rg, bins=30, alpha=0.6, color=GEN_COLOR, density=True, label='Gen')
ax.axvline(np.mean(gt_rg),  color=GT_COLOR,  linestyle='--', linewidth=1.5)
ax.axvline(np.mean(gen_rg), color=GEN_COLOR, linestyle='--', linewidth=1.5)
ax.set_xlabel('Radius of Gyration (Å)')
ax.set_ylabel('Density')
ax.set_title('C3 — Radius of Gyration')
ax.legend()

# ── C4: Ramachandran — Ground Truth ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
mask = np.isfinite(gt_phi_flat) & np.isfinite(gt_psi_flat)
ax.hexbin(gt_phi_flat[mask], gt_psi_flat[mask], gridsize=60, cmap='Blues', mincnt=1)
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('φ (°)'); ax.set_ylabel('ψ (°)')
ax.set_title('C4 — Ramachandran: Ground Truth')

# ── C5: Ramachandran — Generated ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
mask = np.isfinite(gen_phi_flat) & np.isfinite(gen_psi_flat)
ax.hexbin(gen_phi_flat[mask], gen_psi_flat[mask], gridsize=60, cmap='Reds', mincnt=1)
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('φ (°)'); ax.set_ylabel('ψ (°)')
ax.set_title('C5 — Ramachandran: Generated')

# ── C6: Ramachandran overlay (KDE contours) ──────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
m1 = np.isfinite(gt_phi_flat) & np.isfinite(gt_psi_flat)
m2 = np.isfinite(gen_phi_flat) & np.isfinite(gen_psi_flat)

def kde_contour(phi, psi, color, label, ax, alpha=0.6):
    subsample = np.random.choice(len(phi), min(3000, len(phi)), replace=False)
    xy = np.vstack([phi[subsample], psi[subsample]])
    kde = gaussian_kde(xy)
    xi = np.linspace(-180, 180, 80)
    yi = np.linspace(-180, 180, 80)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    ax.contour(Xi, Yi, Zi, levels=5, colors=[color], alpha=alpha, linewidths=1.5)
    ax.contourf(Xi, Yi, Zi, levels=5, colors=[color], alpha=0.08)

kde_contour(gt_phi_flat[m1],  gt_psi_flat[m1],  GT_COLOR,  'GT',  ax)
kde_contour(gen_phi_flat[m2], gen_psi_flat[m2], GEN_COLOR, 'Gen', ax)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0], color=GT_COLOR,  lw=2, label='GT'),
    Line2D([0],[0], color=GEN_COLOR, lw=2, label='Gen'),
])
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('φ (°)'); ax.set_ylabel('ψ (°)')
ax.set_title('C6 — Ramachandran Overlay (KDE)')

plt.savefig('section_C_secondary_structure.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Section C saved.")

# %% [markdown] cell 20
# ---
# ## Section D — Ensemble Diversity & Distance Maps
# ### Metrics: Cα contact/distance maps, pairwise TM-score heatmap, RMSF

# %% cell 21
print("Computing distance maps and ensemble diversity...")

# Mean Cα distance maps
gt_mean_ca  = np.mean(gt_ca, axis=0)   # (n_res, 3)
gen_mean_ca = np.mean(gen_ca, axis=0)  # (n_res, 3)

gt_dmap  = cdist(gt_mean_ca,  gt_mean_ca)   # (n_res, n_res)
gen_dmap = cdist(gen_mean_ca, gen_mean_ca)  # (n_res, n_res)
diff_dmap = gen_dmap - gt_dmap              # signed difference

# RMSF (root-mean-square fluctuation per residue)
gt_rmsf  = np.sqrt(np.mean(np.sum((gt_ca  - gt_mean_ca[None])**2,  axis=-1), axis=0))
gen_rmsf = np.sqrt(np.mean(np.sum((gen_ca - gen_mean_ca[None])**2, axis=-1), axis=0))

# Pairwise TM-score within generated ensemble (subsample for speed)
MAX_PAIR = min(30, len(gen_ca))
gen_sub = gen_ca[:MAX_PAIR]
print(f"Computing {MAX_PAIR}×{MAX_PAIR} pairwise TM-score within generated ensemble...")
pw_tm = np.zeros((MAX_PAIR, MAX_PAIR))
for i in range(MAX_PAIR):
    for j in range(i, MAX_PAIR):
        s = tm_score(gen_sub[i], gen_sub[j])
        pw_tm[i, j] = s
        pw_tm[j, i] = s

print(f"Mean intra-ensemble TM-score: {np.mean(pw_tm[np.triu_indices(MAX_PAIR, k=1)]):.3f}")

# %% cell 22
fig = plt.figure(figsize=(16, 13))
fig.suptitle('Section D — Ensemble Diversity & Distance Maps', fontsize=15, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

vmax = max(gt_dmap.max(), gen_dmap.max())

# ── D1: GT distance map ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(gt_dmap, cmap='viridis', vmin=0, vmax=vmax, origin='lower')
plt.colorbar(im, ax=ax, fraction=0.046, label='Distance (Å)')
ax.set_title('D1 — GT Cα Distance Map')
ax.set_xlabel('Residue'); ax.set_ylabel('Residue')

# ── D2: Generated distance map ───────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(gen_dmap, cmap='viridis', vmin=0, vmax=vmax, origin='lower')
plt.colorbar(im, ax=ax, fraction=0.046, label='Distance (Å)')
ax.set_title('D2 — Gen Cα Distance Map')
ax.set_xlabel('Residue'); ax.set_ylabel('Residue')

# ── D3: Difference map ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
abs_max = np.percentile(np.abs(diff_dmap), 98)
im = ax.imshow(diff_dmap, cmap='bwr', vmin=-abs_max, vmax=abs_max, origin='lower')
plt.colorbar(im, ax=ax, fraction=0.046, label='Δ Distance (Å)')
ax.set_title('D3 — Difference Map (Gen − GT)')
ax.set_xlabel('Residue'); ax.set_ylabel('Residue')

# ── D4: RMSF per residue ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
res_ids = np.arange(1, n_res + 1)
ax.plot(res_ids, gt_rmsf,  color=GT_COLOR,  linewidth=1.5, label='Ground Truth')
ax.plot(res_ids, gen_rmsf, color=GEN_COLOR, linewidth=1.5, label='Generated')
ax.fill_between(res_ids, gt_rmsf,  alpha=0.15, color=GT_COLOR)
ax.fill_between(res_ids, gen_rmsf, alpha=0.15, color=GEN_COLOR)
ax.set_xlabel('Residue index')
ax.set_ylabel('RMSF (Å)')
ax.set_title('D4 — Per-residue RMSF')
ax.legend()

# ── D5: Pairwise TM-score within generated ensemble ──────────────────────────
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(pw_tm, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
plt.colorbar(im, ax=ax, fraction=0.046, label='TM-score')
ax.set_title(f'D5 — Intra-Ensemble TM-score\n(first {MAX_PAIR} generated frames)')
ax.set_xlabel('Frame'); ax.set_ylabel('Frame')

# ── D6: TM-score to GT — cumulative distribution (CDF) ──────────────────────
ax = fig.add_subplot(gs[1, 2])
for vals, color, label in [(gt_tms, GT_COLOR, 'Ground Truth'), (gen_tms, GEN_COLOR, 'Generated')]:
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, color=color, linewidth=2, label=label)
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.2, label='0.5 threshold')
ax.set_xlabel('TM-score vs GT reference')
ax.set_ylabel('Cumulative fraction')
ax.set_title('D6 — TM-score CDF')
ax.legend()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.savefig('section_D_ensemble_diversity.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Section D saved.")

# %% [markdown] cell 23
# ---
# ## Summary Dashboard
# All key metrics in one printout and one consolidated figure.

# %% cell 24
from IPython.display import display, HTML

summary = {
    'Metric': [
        'TM-score (mean)', 'TM-score (std)',
        'Cα RMSD (mean, Å)', 'Cα RMSD (std, Å)',
        'GDT-TS', 'GDT-HA',
        'Ca-Ca Bond Dev (Å)', 'Valid Bond %', 'Steric Clash %',
        'Helix %', 'Strand %', 'Coil %',
        'Radius of Gyration (Å)',
        'RMSF mean (Å)',
    ],
    'Ground Truth': [
        f"{np.mean(gt_tms):.3f}", f"{np.std(gt_tms):.3f}",
        f"{np.mean(gt_rmsds):.2f}", f"{np.std(gt_rmsds):.2f}",
        f"{np.mean(gt_gdt_ts_vals):.3f}", f"{np.mean(gt_gdt_ha_vals):.3f}",
        f"{np.mean(gt_devs):.3f}", f"{np.mean(gt_valids)*100:.1f}%", f"{np.mean(gt_clashes)*100:.2f}%",
        f"{np.mean(gt_helix)*100:.1f}%", f"{np.mean(gt_strand)*100:.1f}%", f"{np.mean(gt_coil)*100:.1f}%",
        f"{np.mean(gt_rg):.2f}",
        f"{np.mean(gt_rmsf):.2f}",
    ],
    'Generated': [
        f"{np.mean(gen_tms):.3f}", f"{np.std(gen_tms):.3f}",
        f"{np.mean(gen_rmsds):.2f}", f"{np.std(gen_rmsds):.2f}",
        f"{np.mean(gen_gdt_ts_vals):.3f}", f"{np.mean(gen_gdt_ha_vals):.3f}",
        f"{np.mean(gen_devs):.3f}", f"{np.mean(gen_valids)*100:.1f}%", f"{np.mean(gen_clashes)*100:.2f}%",
        f"{np.mean(gen_helix)*100:.1f}%", f"{np.mean(gen_strand)*100:.1f}%", f"{np.mean(gen_coil)*100:.1f}%",
        f"{np.mean(gen_rg):.2f}",
        f"{np.mean(gen_rmsf):.2f}",
    ]
}

html = "<table style='border-collapse:collapse;font-family:monospace;font-size:13px'>"
html += "<tr style='background:#2E86AB;color:white'>"
for col in summary:
    html += f"<th style='padding:8px 14px;text-align:left'>{col}</th>"
html += "</tr>"
for i, metric in enumerate(summary['Metric']):
    bg = '#f7f7f7' if i % 2 == 0 else '#ffffff'
    html += f"<tr style='background:{bg}'>"
    html += f"<td style='padding:6px 14px'>{metric}</td>"
    html += f"<td style='padding:6px 14px;color:{GT_COLOR}'>{summary['Ground Truth'][i]}</td>"
    html += f"<td style='padding:6px 14px;color:{GEN_COLOR}'>{summary['Generated'][i]}</td>"
    html += "</tr>"
html += "</table>"

display(HTML("<h3>📊 Summary Metrics Table</h3>" + html))

# %% cell 25
# ── Radar chart / summary bar ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Summary Dashboard', fontsize=15, fontweight='bold')

# Left: normalised metric comparison
ax = axes[0]
metrics_labels = ['TM-score', 'GDT-TS', 'GDT-HA', 'Valid Bond', 'Clash-Free', 'Helix match', 'Strand match']

def safe_match(a, b):
    """How close gen is to GT — 1 = perfect match."""
    return 1.0 - min(abs(a - b) / (abs(a) + 1e-6), 1.0)

gen_scores = [
    np.mean(gen_tms),
    np.mean(gen_gdt_ts_vals),
    np.mean(gen_gdt_ha_vals),
    np.mean(gen_valids),
    1 - np.mean(gen_clashes),
    safe_match(np.mean(gen_helix), np.mean(gt_helix)),
    safe_match(np.mean(gen_strand), np.mean(gt_strand)),
]
gt_scores = [
    np.mean(gt_tms),
    np.mean(gt_gdt_ts_vals),
    np.mean(gt_gdt_ha_vals),
    np.mean(gt_valids),
    1 - np.mean(gt_clashes),
    1.0, 1.0,
]

x = np.arange(len(metrics_labels))
w = 0.35
b1 = ax.bar(x - w/2, gt_scores,  w, color=GT_COLOR,  alpha=0.85, label='Ground Truth')
b2 = ax.bar(x + w/2, gen_scores, w, color=GEN_COLOR, alpha=0.85, label='Generated')
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, rotation=30, ha='right')
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score (higher = better)')
ax.set_title('Summary: Key Metrics Comparison')
ax.legend()

# Right: box plots of TM-score and RMSD side by side
ax = axes[1]
data_to_plot = [gt_tms, gen_tms]
colors = [GT_COLOR, GEN_COLOR]
bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.4,
                medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for whisker in bp['whiskers']:
    whisker.set(color='gray', linewidth=1.2)
for cap in bp['caps']:
    cap.set(color='gray', linewidth=1.2)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.2, label='Fold threshold')
ax.set_xticks([1, 2])
ax.set_xticklabels(['Ground Truth', 'Generated'])
ax.set_ylabel('TM-score')
ax.set_title('TM-score Distribution (Box Plot)')
ax.legend()

plt.tight_layout()
plt.savefig('summary_dashboard.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Summary dashboard saved.")

# %% cell 26
# ── Download all saved figures ───────────────────────────────────────────────
import zipfile

figures = [
    'section_A_structural_similarity.png',
    'section_B_geometry_validity.png',
    'section_C_secondary_structure.png',
    'section_D_ensemble_diversity.png',
    'summary_dashboard.png',
]

with zipfile.ZipFile(f'{PDB_ID}_protein_validation_figures.zip', 'w') as zf:
    for f in figures:
        if os.path.exists(f):
            zf.write(f)
            print(f"  Added {f}")

try:
    from google.colab import files
    files.download(f'{PDB_ID}_protein_validation_figures.zip')
    print("\n✅ Figures ZIP downloaded.")
except ImportError:
    print(f"\n✅ Figures saved locally as {PDB_ID}_protein_validation_figures.zip")

# %% cell 27
# PASTE THIS AS NEW CELLS IN YOUR NOTEBOOK BEFORE THE SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
# SECTION E — Trajectory-Level Quality Metrics
# ═══════════════════════════════════════════════════════════════════════

# ── E0: Extra installs ───────────────────────────────────────────────────────
# Run this first, then continue
# NOTEBOOK_MAGIC: !pip install scikit-learn scipy -q
print("✅ sklearn and scipy ready")


# ════════════════════════════════════════════════════════════════════════════
# CELL E1 — PCA: Conformational Space Coverage
# ════════════════════════════════════════════════════════════════════════════
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# Flatten Cα positions per frame: (n_frames, n_res*3)
gt_flat  = gt_ca.reshape(len(gt_ca),  -1)   # (n_gt_frames,  n_res*3)
gen_flat = gen_ca.reshape(len(gen_ca), -1)   # (n_gen_frames, n_res*3)

# Fit PCA on GT only, then project both onto the same space
pca = PCA(n_components=5)
scaler = StandardScaler()
gt_scaled  = scaler.fit_transform(gt_flat)
gen_scaled = scaler.transform(gen_flat)

gt_pca  = pca.fit_transform(gt_scaled)   # (n_gt_frames,  5)
gen_pca = pca.transform(gen_scaled)      # (n_gen_frames, 5)

explained = pca.explained_variance_ratio_ * 100
print(f"PC1 explains {explained[0]:.1f}% variance")
print(f"PC2 explains {explained[1]:.1f}% variance")
print(f"PC1+PC2 total: {explained[0]+explained[1]:.1f}%")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Section E1 — PCA Conformational Space Coverage',
             fontsize=14, fontweight='bold')

# E1a: PC1 vs PC2 scatter
ax = axes[0]
ax.scatter(gt_pca[:, 0],  gt_pca[:, 1],  alpha=0.3, s=8,
           color=GT_COLOR,  label=f'Ground Truth (n={len(gt_pca)})')
ax.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.5, s=12,
           color=GEN_COLOR, label=f'Generated (n={len(gen_pca)})', marker='x')
ax.set_xlabel(f'PC1 ({explained[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({explained[1]:.1f}%)')
ax.set_title('E1a — PC1 vs PC2')
ax.legend(markerscale=2)

# E1b: Free Energy Landscape — GT
ax = axes[1]
h_gt = ax.hist2d(gt_pca[:, 0], gt_pca[:, 1], bins=40,
                  cmap='Blues', density=True)
fe_gt = -np.log(h_gt[0].T + 1e-10)
fe_gt -= fe_gt.min()
im = ax.imshow(fe_gt, origin='lower', aspect='auto',
               extent=[h_gt[1][0], h_gt[1][-1], h_gt[2][0], h_gt[2][-1]],
               cmap='Blues_r', vmin=0, vmax=5)
plt.colorbar(im, ax=ax, label='Free Energy (kT)')
ax.set_xlabel(f'PC1 ({explained[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({explained[1]:.1f}%)')
ax.set_title('E1b — FEL: Ground Truth')

# E1c: Free Energy Landscape — Generated
ax = axes[2]
h_gen = ax.hist2d(gen_pca[:, 0], gen_pca[:, 1], bins=40,
                   cmap='Reds', density=True)
fe_gen = -np.log(h_gen[0].T + 1e-10)
fe_gen -= fe_gen.min()
im = ax.imshow(fe_gen, origin='lower', aspect='auto',
               extent=[h_gen[1][0], h_gen[1][-1], h_gen[2][0], h_gen[2][-1]],
               cmap='Reds_r', vmin=0, vmax=5)
plt.colorbar(im, ax=ax, label='Free Energy (kT)')
ax.set_xlabel(f'PC1 ({explained[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({explained[1]:.1f}%)')
ax.set_title('E1c — FEL: Generated')

plt.tight_layout()
plt.savefig('section_E1_pca_fel.png', bbox_inches='tight', dpi=150)
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# CELL E2 — Distribution Overlap: JSD, Wasserstein, MMD
# ════════════════════════════════════════════════════════════════════════════

def histogram_1d(data, bins=50, range_=None):
    """Return normalised histogram as probability distribution."""
    counts, edges = np.histogram(data, bins=bins, range=range_, density=False)
    probs = counts / counts.sum()
    return probs, edges

def mmd_linear(X, Y):
    """Linear-time Maximum Mean Discrepancy estimator."""
    n, m = len(X), len(Y)
    XXT = np.dot(X, X.T)
    YYT = np.dot(Y, Y.T)
    XYT = np.dot(X, Y.T)
    mmd = (np.sum(XXT) / (n*n)
           + np.sum(YYT) / (m*m)
           - 2 * np.sum(XYT) / (n*m))
    return float(max(mmd, 0.0)) ** 0.5

print("Computing distribution metrics...\n")
results = {}

for pc_idx in range(3):
    gt_vals  = gt_pca[:,  pc_idx]
    gen_vals = gen_pca[:, pc_idx]

    # Shared range
    lo = min(gt_vals.min(), gen_vals.min())
    hi = max(gt_vals.max(), gen_vals.max())

    gt_hist,  _ = histogram_1d(gt_vals,  range_=(lo, hi))
    gen_hist, _ = histogram_1d(gen_vals, range_=(lo, hi))

    jsd = jensenshannon(gt_hist, gen_hist)           # 0=identical, 1=max diff
    w2  = wasserstein_distance(gt_vals, gen_vals)    # Earth mover's distance

    results[f'PC{pc_idx+1}'] = {'JSD': jsd, 'W2': w2}
    print(f"PC{pc_idx+1}:  JSD={jsd:.4f}  |  Wasserstein={w2:.3f} Å")

# MMD on full PC space (top 5 PCs)
mmd = mmd_linear(gt_pca[:, :5], gen_pca[:, :5])
print(f"\nMMD (top-5 PCs): {mmd:.4f}  (0 = identical distributions)")

# ── Plot JSD and Wasserstein ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Section E2 — Distribution Overlap Metrics',
             fontsize=14, fontweight='bold')

pc_labels = [f'PC{i+1}' for i in range(3)]

# JSD bar
ax = axes[0]
jsd_vals = [results[p]['JSD'] for p in pc_labels]
colors_bar = [ACC_COLOR if v < 0.1 else '#F4A261' if v < 0.3 else GEN_COLOR
              for v in jsd_vals]
bars = ax.bar(pc_labels, jsd_vals, color=colors_bar, alpha=0.85)
ax.bar_label(bars, fmt='%.3f', padding=3)
ax.axhline(0.1, color=ACC_COLOR,  linestyle='--', lw=1.2, label='Good (<0.1)')
ax.axhline(0.3, color='#F4A261',  linestyle='--', lw=1.2, label='OK (<0.3)')
ax.set_ylabel('Jensen-Shannon Divergence')
ax.set_title('E2a — JSD per PC\n(lower = better)')
ax.set_ylim(0, max(jsd_vals) * 1.3 + 0.05)
ax.legend(fontsize=8)

# Wasserstein bar
ax = axes[1]
w2_vals = [results[p]['W2'] for p in pc_labels]
bars = ax.bar(pc_labels, w2_vals, color=GT_COLOR, alpha=0.75)
ax.bar_label(bars, fmt='%.2f', padding=3)
ax.set_ylabel('Wasserstein Distance (Å)')
ax.set_title('E2b — Wasserstein Distance per PC\n(lower = better)')

# PC1 distribution overlay
ax = axes[2]
lo = min(gt_pca[:,0].min(), gen_pca[:,0].min())
hi = max(gt_pca[:,0].max(), gen_pca[:,0].max())
ax.hist(gt_pca[:,  0], bins=40, alpha=0.6, color=GT_COLOR,
        density=True, label='GT',  range=(lo, hi))
ax.hist(gen_pca[:, 0], bins=40, alpha=0.6, color=GEN_COLOR,
        density=True, label='Gen', range=(lo, hi))
ax.set_xlabel('PC1 coordinate')
ax.set_ylabel('Density')
ax.set_title(f'E2c — PC1 Distribution Overlap\nJSD={results["PC1"]["JSD"]:.3f}')
ax.legend()

plt.tight_layout()
plt.savefig('section_E2_distribution_overlap.png', bbox_inches='tight', dpi=150)
plt.show()
print(f"\n✅ Summary: MMD={mmd:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# CELL E3 — RMSF Correlation & Contact Frequency
# ════════════════════════════════════════════════════════════════════════════

# ── RMSF Pearson correlation ─────────────────────────────────────────────────
gt_mean  = np.mean(gt_ca,  axis=0)
gen_mean = np.mean(gen_ca, axis=0)

gt_rmsf  = np.sqrt(np.mean(np.sum((gt_ca  - gt_mean[None])**2,  axis=-1), axis=0))
gen_rmsf = np.sqrt(np.mean(np.sum((gen_ca - gen_mean[None])**2, axis=-1), axis=0))

rmsf_r, rmsf_p = pearsonr(gt_rmsf, gen_rmsf)
print(f"RMSF Pearson r = {rmsf_r:.3f}  (1.0 = perfect, >0.7 = good)")

# ── Contact frequency maps ───────────────────────────────────────────────────
CONTACT_CUTOFF = 8.0  # Å — residues within this distance are "in contact"

def contact_frequency(ca_frames, cutoff=8.0):
    """Fraction of frames where each residue pair is within cutoff."""
    n_frames, n_res, _ = ca_frames.shape
    freq = np.zeros((n_res, n_res))
    for frame in ca_frames:
        dmat = np.linalg.norm(frame[:, None, :] - frame[None, :, :], axis=-1)
        freq += (dmat < cutoff).astype(float)
    return freq / n_frames

print("Computing contact frequencies (may take ~30s)...")
gt_contacts  = contact_frequency(gt_ca)
gen_contacts = contact_frequency(gen_ca)
contact_diff = gen_contacts - gt_contacts

# Overlap score: mean absolute difference in contact frequencies
contact_overlap = 1.0 - np.mean(np.abs(contact_diff))
print(f"Contact frequency overlap score = {contact_overlap:.3f}  (1.0 = perfect)")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Section E3 — RMSF Correlation & Contact Frequency',
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# E3a: RMSF scatter (GT vs Gen per residue)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(gt_rmsf, gen_rmsf, s=12, alpha=0.6, color=GT_COLOR)
lim = max(gt_rmsf.max(), gen_rmsf.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y=x (perfect)')
ax.set_xlabel('GT RMSF (Å)')
ax.set_ylabel('Generated RMSF (Å)')
ax.set_title(f'E3a — RMSF Correlation\nPearson r={rmsf_r:.3f}')
ax.legend()

# E3b: RMSF line comparison
ax = fig.add_subplot(gs[0, 1:])
res_ids = np.arange(1, n_res + 1)
ax.plot(res_ids, gt_rmsf,  color=GT_COLOR,  lw=1.5, label='Ground Truth')
ax.plot(res_ids, gen_rmsf, color=GEN_COLOR, lw=1.5, label='Generated')
ax.fill_between(res_ids, gt_rmsf,  alpha=0.15, color=GT_COLOR)
ax.fill_between(res_ids, gen_rmsf, alpha=0.15, color=GEN_COLOR)
ax.set_xlabel('Residue index')
ax.set_ylabel('RMSF (Å)')
ax.set_title(f'E3b — Per-residue RMSF  |  Pearson r={rmsf_r:.3f}')
ax.legend()

# E3c: GT contact frequency map
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(gt_contacts, cmap='Blues', vmin=0, vmax=1, origin='lower')
plt.colorbar(im, ax=ax, fraction=0.046, label='Contact freq.')
ax.set_title('E3c — GT Contact Frequency')
ax.set_xlabel('Residue'); ax.set_ylabel('Residue')

# E3d: Gen contact frequency map
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(gen_contacts, cmap='Reds', vmin=0, vmax=1, origin='lower')
plt.colorbar(im, ax=ax, fraction=0.046, label='Contact freq.')
ax.set_title('E3d — Gen Contact Frequency')
ax.set_xlabel('Residue'); ax.set_ylabel('Residue')

# E3e: Difference map
ax = fig.add_subplot(gs[1, 2])
im = ax.imshow(contact_diff, cmap='bwr', vmin=-0.5, vmax=0.5, origin='lower')
plt.colorbar(im, ax=ax, fraction=0.046, label='Δ Freq (Gen−GT)')
ax.set_title(f'E3e — Contact Diff\nOverlap={contact_overlap:.3f}')
ax.set_xlabel('Residue'); ax.set_ylabel('Residue')

plt.savefig('section_E3_rmsf_contacts.png', bbox_inches='tight', dpi=150)
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# CELL E4 — Autocorrelation & Ensemble Diversity Summary
# ════════════════════════════════════════════════════════════════════════════

def autocorrelation(signal, max_lag=None):
    """Normalised autocorrelation function of a 1D signal."""
    signal = signal - signal.mean()
    n = len(signal)
    if max_lag is None:
        max_lag = n // 2
    acf = np.array([
        np.dot(signal[:n-lag], signal[lag:]) / (np.dot(signal, signal))
        for lag in range(max_lag)
    ])
    return acf

def mean_pairwise_rmsd(ca_frames, subsample=40):
    """Mean RMSD between all pairs of frames — measures diversity."""
    idx = np.random.choice(len(ca_frames), min(subsample, len(ca_frames)),
                           replace=False)
    frames = ca_frames[idx]
    rmsds = []
    for i in range(len(frames)):
        for j in range(i+1, len(frames)):
            diff = frames[i] - frames[j]
            rmsds.append(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))
    return np.mean(rmsds), np.std(rmsds)

print("Computing autocorrelation and diversity metrics...")

# Autocorrelation of PC1 (captures slow conformational modes)
max_lag = min(100, len(gt_pca)//2, len(gen_pca)//2)
gt_acf  = autocorrelation(gt_pca[:, 0],  max_lag=max_lag)
gen_acf = autocorrelation(gen_pca[:, 0], max_lag=max_lag)

# Mean pairwise RMSD (ensemble diversity)
gt_div_mean,  gt_div_std  = mean_pairwise_rmsd(gt_ca)
gen_div_mean, gen_div_std = mean_pairwise_rmsd(gen_ca)
print(f"GT  mean pairwise RMSD: {gt_div_mean:.2f} ± {gt_div_std:.2f} Å")
print(f"Gen mean pairwise RMSD: {gen_div_mean:.2f} ± {gen_div_std:.2f} Å")
print(f"Diversity ratio (Gen/GT): {gen_div_mean/gt_div_mean:.2f}  (1.0 = same diversity)")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Section E4 — Autocorrelation & Ensemble Diversity',
             fontsize=14, fontweight='bold')

# E4a: Autocorrelation of PC1
ax = axes[0]
lags = np.arange(max_lag)
ax.plot(lags, gt_acf,  color=GT_COLOR,  lw=1.8, label='Ground Truth')
ax.plot(lags, gen_acf, color=GEN_COLOR, lw=1.8, label='Generated')
ax.axhline(0, color='black', lw=0.8, linestyle='--')
ax.axhline(1/np.e, color='gray', lw=1, linestyle=':', label='1/e decay')
ax.set_xlabel('Lag (frames)')
ax.set_ylabel('Autocorrelation')
ax.set_title('E4a — PC1 Autocorrelation\n(how fast conformations change)')
ax.legend()
ax.set_ylim(-0.3, 1.05)

# E4b: Pairwise RMSD distribution (diversity)
ax = axes[1]
# Sample pairwise RMSDs for histogram
def pairwise_rmsd_samples(ca_frames, n_pairs=300):
    idx_i = np.random.randint(0, len(ca_frames), n_pairs)
    idx_j = np.random.randint(0, len(ca_frames), n_pairs)
    rmsds = []
    for i, j in zip(idx_i, idx_j):
        if i != j:
            diff = ca_frames[i] - ca_frames[j]
            rmsds.append(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))
    return np.array(rmsds)

gt_pw  = pairwise_rmsd_samples(gt_ca)
gen_pw = pairwise_rmsd_samples(gen_ca)
ax.hist(gt_pw,  bins=30, alpha=0.6, color=GT_COLOR,  density=True, label='GT')
ax.hist(gen_pw, bins=30, alpha=0.6, color=GEN_COLOR, density=True, label='Gen')
ax.set_xlabel('Pairwise Cα RMSD (Å)')
ax.set_ylabel('Density')
ax.set_title(f'E4b — Pairwise RMSD Distribution\nGT: {gt_div_mean:.2f} Å  |  Gen: {gen_div_mean:.2f} Å')
ax.legend()

# E4c: Summary radar of all E-section scores
ax = axes[2]
metric_names = [
    'RMSF\ncorrelation',
    'Contact\noverlap',
    'PC1 JSD\n(inverted)',
    'Diversity\nratio',
    'TM-score\n(mean)',
]
jsd_inv = max(0, 1 - results['PC1']['JSD'] * 3)  # scale so 0 JSD → 1
div_ratio_clipped = min(gen_div_mean / gt_div_mean, 1.0)
gen_scores_e = [
    max(0, rmsf_r),
    contact_overlap,
    jsd_inv,
    div_ratio_clipped,
    np.mean(gen_tms),
]
gt_scores_e = [1.0, 1.0, 1.0, 1.0, np.mean(gt_tms)]

x = np.arange(len(metric_names))
w = 0.35
b1 = ax.bar(x - w/2, gt_scores_e,  w, color=GT_COLOR,  alpha=0.85, label='GT ref')
b2 = ax.bar(x + w/2, gen_scores_e, w, color=GEN_COLOR, alpha=0.85, label='Generated')
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=8)
ax.set_ylim(0, 1.2)
ax.set_ylabel('Score (higher = better)')
ax.set_title('E4c — Trajectory Quality Summary')
ax.legend()
for b, v in zip(b2, gen_scores_e):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}',
            ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig('section_E4_autocorr_diversity.png', bbox_inches='tight', dpi=150)
plt.show()

print("\n" + "="*55)
print(" SECTION E — TRAJECTORY QUALITY SUMMARY")
print("="*55)
print(f"  RMSF Pearson r         : {rmsf_r:.3f}  (>0.7 = good)")
print(f"  Contact overlap        : {contact_overlap:.3f}  (>0.85 = good)")
print(f"  PC1 JSD                : {results['PC1']['JSD']:.3f}  (<0.1 = good)")
print(f"  MMD (top-5 PCs)        : {mmd:.3f}  (<0.1 = good)")
print(f"  GT  pairwise RMSD      : {gt_div_mean:.2f} Å")
print(f"  Gen pairwise RMSD      : {gen_div_mean:.2f} Å")
print(f"  Diversity ratio        : {gen_div_mean/gt_div_mean:.2f}  (1.0 = same diversity)")
print("="*55)


# ── Download all Section E figures ───────────────────────────────────────────
import zipfile, os

e_figures = [
    'section_E1_pca_fel.png',
    'section_E2_distribution_overlap.png',
    'section_E3_rmsf_contacts.png',
    'section_E4_autocorr_diversity.png',
]

with zipfile.ZipFile(f'{PDB_ID}_section_E_figures.zip', 'w') as zf:
    for f in e_figures:
        if os.path.exists(f):
            zf.write(f)
            print(f"  Added {f}")

try:
    from google.colab import files
    files.download(f'{PDB_ID}_section_E_figures.zip')
    print("✅ Section E figures downloaded.")
except ImportError:
    print("✅ Saved as section_E_figures.zip")

