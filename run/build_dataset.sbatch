#!/bin/bash
#SBATCH --job-name=gsr_build
#SBATCH --account=torch_pr_800_cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-0            # <-- set to 0-(NUM_SHARDS-1); keep NUM_SHARDS in sync

# =====================================================================
# Build the score+embedding store for training. Edit the variables below;
# this is the single place to change any dataset-build option.
# =====================================================================
DATASET_NAME=uniref90_dev5000    # store subdir
FASTA_PATH=/scratch/jv2807/gene_structure_removal/data/human_uniref90.fasta
ESM_MODEL=esmc_600m
EMBEDDING_LAYER=-1
POOLING=concat              # mean | mutated_position | concat
MAX_SEQ_LEN=1024
SCORER=masked_marginal       # masked_marginal | wt_marginal | pll
SCORE_BATCH_SIZE=16
VARIANTS_PER_GENE=200
MIN_VARIANTS_PER_GENE=10
QUARTILE_LOW=0.25
QUARTILE_HIGH=0.25
NUM_SHARDS=1                  # keep in sync with --array range above
LIMIT_GENES=0                # 0 = all; set small (e.g. 20) for a smoke test
SEED=0

OVERLAY=/scratch/jv2807/dms_singularity/dms_contrastive.ext3
BASE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
REPO=/home/jv2807/gene_structure

singularity exec --nv --overlay ${OVERLAY}:ro ${BASE} /bin/bash -c "
  source /ext3/env.sh
  export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
  cd ${REPO}
  PYTHONPATH=src python scripts/build_dataset.py \
    --dataset_name ${DATASET_NAME} \
    --fasta_path ${FASTA_PATH} \
    --esm_model ${ESM_MODEL} \
    --embedding_layer ${EMBEDDING_LAYER} \
    --pooling ${POOLING} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --scorer ${SCORER} \
    --score_batch_size ${SCORE_BATCH_SIZE} \
    --variants_per_gene ${VARIANTS_PER_GENE} \
    --min_variants_per_gene ${MIN_VARIANTS_PER_GENE} \
    --quartile_low ${QUARTILE_LOW} \
    --quartile_high ${QUARTILE_HIGH} \
    --num_shards ${NUM_SHARDS} \
    --shard_id ${SLURM_ARRAY_TASK_ID} \
    --limit_genes ${LIMIT_GENES} \
    --seed ${SEED}
"
