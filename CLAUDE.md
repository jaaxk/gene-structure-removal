# Repo rules -- gene_structure

Brief, evolving rules for this repo. Keep it short.

## Storage
- Code only here (`/home/jv2807/gene_structure`; 30K inode / 50GB limit).
- All data/outputs/checkpoints/wandb → `/scratch/jv2807/gene_structure_removal/`
  (`data/ scores/ embeddings/ runs/ eval/`). Paths live in `src/gsr/paths.py` -- do
  not hardcode scratch paths elsewhere.

## Running
- No heavy compute on the login node. Use `srun`/`sbatch` inside the overlay.
- Env is the Singularity overlay `dms_contrastive.ext3`:
  `singularity exec --nv --overlay <ovl>.ext3:ro <base>.sif /bin/bash -c "source /ext3/env.sh && python ..."`
- One place to change any hyperparameter: the `run/*.sbatch` scripts (UPPERCASE
  vars at the top → CLI flags in `src/gsr/args.py`).
- Commit to `main` before submitting a job.

## Conventions
- Config = argparse (`src/gsr/args.py`); validated at startup.
- Print dataset stats on every run (`gsr.utils.stats`).
- Storage format: per-variant scalars → Parquet; embeddings → HDF5 shards + manifest.
- Losses live in their own module (`src/gsr/losses/`), swappable via `--loss_type`.
- Reproducibility: seed everything; resolved args saved per run.
- Whenever a significant change is made that affects default behavior, update the README.
