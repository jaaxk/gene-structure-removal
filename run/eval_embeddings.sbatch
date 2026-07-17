#!/bin/bash
#SBATCH --job-name=gsr_eval
#SBATCH --account=torch_pr_800_cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# =====================================================================
# Standalone evaluation of embeddings. Leave CHECKPOINT empty to get the
# raw-backbone baseline; set it to a run's best.pt/final.pt to evaluate the
# trained projection. Edit the variables below.
# =====================================================================
OUT_NAME=backbone_baseline_concat
CHECKPOINT=                 # e.g. /scratch/.../runs/<run>/best.pt ; empty = backbone
EVALUATORS="centroid dimreduction regression"

ESM_MODEL=esmc_600m
EMBEDDING_LAYER=-1
POOLING=concat              # mean | mutated_position | concat
SCORE_BATCH_SIZE=16

DMS_SELECTION_TYPES="Activity Binding Expression Stability OrganismalFitness"
DMS_MAX_PER_ASSAY=200

EVAL_DISTANCE=cosine
HELD_OUT_GENE_FRAC=0.2
CENTROID_SUBSAMPLE=0        # 0 = use all centroid variants (standalone)
DIMRED_METHODS="pca tsne umap"
DIMRED_MAX_POINTS=5000
SEED=0

OVERLAY=/scratch/jv2807/dms_singularity/dms_contrastive.ext3
BASE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
REPO=/home/jv2807/gene_structure

CKPT_ARG=""
if [ -n "${CHECKPOINT}" ]; then CKPT_ARG="--checkpoint ${CHECKPOINT}"; fi

singularity exec --nv --overlay ${OVERLAY}:ro ${BASE} /bin/bash -c "
  source /ext3/env.sh
  export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
  cd ${REPO}
  PYTHONPATH=src python scripts/eval_embeddings.py \
    --out_name ${OUT_NAME} ${CKPT_ARG} \
    --evaluators ${EVALUATORS} \
    --esm_model ${ESM_MODEL} \
    --embedding_layer ${EMBEDDING_LAYER} \
    --pooling ${POOLING} \
    --score_batch_size ${SCORE_BATCH_SIZE} \
    --dms_selection_types ${DMS_SELECTION_TYPES} \
    --dms_max_per_assay ${DMS_MAX_PER_ASSAY} \
    --eval_distance ${EVAL_DISTANCE} \
    --held_out_gene_frac ${HELD_OUT_GENE_FRAC} \
    --centroid_subsample ${CENTROID_SUBSAMPLE} \
    --dimred_methods ${DIMRED_METHODS} \
    --dimred_max_points ${DIMRED_MAX_POINTS} \
    --seed ${SEED}
"
