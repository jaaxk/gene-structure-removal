"""Centralized filesystem paths for the ESM Gene Structure Removal project.

All heavy artifacts (data, scores, embeddings, checkpoints, wandb, eval outputs)
MUST live under scratch -- NEVER under the code repo in /home (30K inode limit).
This module is the single source of truth for where things go, so no other
module hardcodes a scratch path.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Code repo (home) -------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Scratch root for all outputs/data -------------------------------------
# Overridable via env var so the repo is portable across users.
SCRATCH_ROOT = Path(
    os.environ.get("GSR_SCRATCH_ROOT", "/scratch/jv2807/gene_structure_removal")
)

# Working-set inputs (downloaded UniRef, DMS CSV copies/links, splits).
DATA_DIR = SCRATCH_ROOT / "data"
# Cached wild-type / variant LL/PLL scores + per-variant metadata (parquet).
SCORES_DIR = SCRATCH_ROOT / "scores"
# Cached per-variant embeddings (HDF5 shards + manifest).
EMBEDDINGS_DIR = SCRATCH_ROOT / "embeddings"
# Per-run outputs: checkpoints, resolved args, logs, wandb.
RUNS_DIR = SCRATCH_ROOT / "runs"
# Standalone evaluation outputs (figures, metrics tables).
EVAL_DIR = SCRATCH_ROOT / "eval"

# --- External datasets already on disk (read-only; CSVs only) --------------
PROTEINGYM_DIR = Path("/scratch/jv2807/proteingym")
PROTEINGYM_REFERENCE_CSV = PROTEINGYM_DIR / "reference_files" / "DMS_substitutions.csv"
# Per-selection-type DMS CSVs (sequences + DMS scores). We use ONLY the CSVs
# here; the sibling h5 embedding caches are intentionally NOT touched.
DMS_DATASETS_DIR = Path("/scratch/jv2807/dms_data/datasets")

# --- Singularity overlay (shared Python env) -------------------------------
OVERLAY = Path("/scratch/jv2807/dms_singularity/dms_contrastive.ext3")
BASE_IMAGE = Path(
    "/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
)


def run_dir(run_name: str) -> Path:
    """Directory holding all outputs for a single training run."""
    return RUNS_DIR / run_name


def ensure_dirs(*dirs: Path) -> None:
    """Create the given scratch directories if they do not exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
