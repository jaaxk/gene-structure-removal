#!/bin/bash
# Launch one frozen (CPU-only) tuning run on the allocated GPU node's spare
# CPU/mem, via SSH (adopted into the job's cgroup by pam_slurm_adopt so it's
# capped by the job's own allocation). Forces --device cpu so it never
# competes with the LoRA run for the GPU.
# Usage: tune_launch_gpunode_frozen.sh <gpu_jobid> <run_name> <extra train.py args...>
set -uo pipefail
GPU_JOBID="$1"; shift
RUN_NAME="$1"; shift
LOGDIR=/scratch/jv2807/gene_structure_removal/tune/logs
mkdir -p "$LOGDIR"

NODE=$(squeue -j "$GPU_JOBID" -h -o "%N")
if [ -z "$NODE" ]; then
  echo "ERROR: could not resolve node for jobid $GPU_JOBID" >&2
  exit 1
fi

FIXED_ARGS=(
  --fasta_path /scratch/jv2807/gene_structure_removal/data/human_uniref90.fasta
  --limit_genes 5000 --max_seq_len 400 --esm_model esmc_600m --embedding_layer -1
  --pooling concat --scorer masked_marginal --variants_per_gene 150
  --min_variants_per_gene 10 --quartile_low 0.25 --quartile_high 0.25
  --dms_selection_types Activity Binding Expression Stability OrganismalFitness
  --dms_max_per_assay 200 --eval_distance cosine --held_out_gene_frac 0.2 --seed 0
  --embeddings_mode stream --num_workers 2 --no_save_checkpoints
  --device cpu --wandb_project gene-structure-removal --wandb_mode online
  --run_name "$RUN_NAME"
)

ssh -o BatchMode=yes "$NODE" "singularity exec \
  --overlay /scratch/jv2807/dms_singularity/dms_contrastive.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c \
  'source /ext3/env.sh && export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES= \
   OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 && \
   cd /home/jv2807/gene_structure && PYTHONPATH=src python scripts/train.py ${FIXED_ARGS[*]} $* '" \
  > "$LOGDIR/$RUN_NAME.log" 2>&1 &

echo "launched $RUN_NAME (pid $!) on $NODE (cpu) -> $LOGDIR/$RUN_NAME.log"
