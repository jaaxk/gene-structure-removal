# Interactive / smoke runs

The login node is shared: only cheap commands run there directly. Everything that
touches ESM, GPUs, or real data goes through SLURM. For quick one-offs use `srun`
inside the overlay; for real runs use the `sbatch` scripts in this directory.

## Overlay invocation

```bash
OVERLAY=/scratch/jv2807/dms_singularity/dms_contrastive.ext3
BASE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

srun --account=torch_pr_800_cds --gres=gpu:1 --time=00:30:00 --mem=32G --cpus-per-task=4 \
  singularity exec --nv --overlay ${OVERLAY}:ro ${BASE} /bin/bash -c \
  "source /ext3/env.sh && export HDF5_USE_FILE_LOCKING=FALSE && cd /home/jv2807/gene_structure && \
   PYTHONPATH=src python scripts/build_dataset.py --dataset_name smoke --limit_genes 20 --variants_per_gene 40"
```

Drop `--gres=gpu:1` and `--nv` for CPU-only work (e.g. `pytest tests`).

## Installing a package into the overlay

Mount `:rw` (needs the overlay free of other users), e.g. to add `umap-learn`:

```bash
srun --account=torch_pr_800_cds --time=00:30:00 --mem=16G \
  singularity exec --overlay ${OVERLAY}:rw ${BASE} /bin/bash -c \
  "source /ext3/env.sh && pip install umap-learn"
```

If you hit `overlay image ... currently in use` (stale NFS lock), copy the overlay
to a fresh file and use that: `cp --sparse=always ${OVERLAY} ${OVERLAY}.new && mv ...`.
