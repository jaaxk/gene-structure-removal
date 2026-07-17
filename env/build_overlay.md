# Python environment (Singularity overlay)

This project reuses the existing overlay from the sibling repo:

```
/scratch/jv2807/dms_singularity/dms_contrastive.ext3
```

It already provides everything the pipeline needs: torch 2.5.1+cu121,
transformers 4.48.1, fair-esm, peft, h5py, pyarrow, pandas, scikit-learn,
matplotlib, seaborn, tqdm, wandb. ESM-C 600M (`Synthyra/ESMplusplus_large`) is in
its HuggingFace cache.

## Run pattern

```bash
OVERLAY=/scratch/jv2807/dms_singularity/dms_contrastive.ext3
BASE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
singularity exec --nv --overlay ${OVERLAY}:ro ${BASE} /bin/bash -c \
  "source /ext3/env.sh && export HDF5_USE_FILE_LOCKING=FALSE && \
   cd /home/jv2807/gene_structure && PYTHONPATH=src python ..."
```

## Optional: umap-learn

UMAP figures require `umap-learn`, which may not be in the overlay. The
dim-reduction code degrades gracefully (PCA + t-SNE still run; UMAP is skipped
with a message). To enable UMAP, either:

- install into the shared overlay (mount `:rw` -- **note this mutates the env the
  sibling repo also uses**):
  ```bash
  srun --account=torch_pr_800_cds --time=00:30:00 --mem=16G \
    singularity exec --overlay ${OVERLAY}:rw ${BASE} /bin/bash -c \
    "source /ext3/env.sh && pip install umap-learn"
  ```
- or, to avoid touching the shared env, build a dedicated overlay for this project
  following the `environments` skill (`apptainer overlay create --size 5000 --sparse
  --create-dir ext3 ...`, Miniforge into `/ext3/miniforge3`, then `pip install -r
  requirements.txt`). See the skill for cluster gotchas (no `--fakeroot`; stale NFS
  lock -> `cp --sparse=always`).

## Reproducibility

`requirements.txt` pins minimum versions. To capture an exact freeze from the
overlay: `pip freeze > env/requirements.lock.txt`.
```
