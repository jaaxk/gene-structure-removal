#!/bin/bash
#SBATCH --job-name=gsr_train
#SBATCH --account=torch_pr_800_cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# =====================================================================
# Train the projection head. Edit the variables below -- this is the
# single place to change any training hyperparameter.
# =====================================================================
RUN_NAME=gsr_wtanchored_concat
DATASET_NAME=uniref90_dev5000               # must match a built store
POOLING=concat             # mean | mutated_position | concat (must match the store)

# --- head ---
HEAD_HIDDEN_DIMS="1024 512"
HEAD_OUT_DIM=256
HEAD_DROPOUT=0.1
HEAD_ACTIVATION=gelu       # relu | gelu | silu | tanh
HEAD_NORM=layernorm        # none | layernorm | batchnorm

# --- loss ---
LOSS_TYPE=wt_anchored_bce  # wt_anchored_bce | contrastive_ce | ntxent | triplet
DISTANCE_METRIC=cosine     # cosine | euclidean

# --- LoRA (backbone finetuning; off by default) ---
USE_LORA=""                # set to "--use_lora" to finetune the backbone
ESM_MODEL=esmc_600m
ESM_LR=1e-5
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# --- batching / diversity ---
BATCH_MODE=gene_diverse    # gene_diverse | cross_gene
BATCH_SIZE=64
GENES_PER_BATCH=2          # cross_gene only

# --- optimization ---
LR=1e-3
WEIGHT_DECAY=0.0
EPOCHS=20
GRAD_CLIP=1.0

# --- eval ---
EVAL_EVERY_STEPS=500       # 0 = disable during-training eval
CENTROID_SUBSAMPLE=2000
EVAL_DISTANCE=cosine
HELD_OUT_GENE_FRAC=0.2

# --- tracking ---
WANDB_PROJECT=gene-structure-removal
WANDB_MODE=online          # online | offline | disabled
SEED=0

OVERLAY=/scratch/jv2807/dms_singularity/dms_contrastive.ext3
BASE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
REPO=/home/jv2807/gene_structure

singularity exec --nv --overlay ${OVERLAY}:ro ${BASE} /bin/bash -c "
  source /ext3/env.sh
  export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
  cd ${REPO}
  PYTHONPATH=src python scripts/train.py \
    --run_name ${RUN_NAME} \
    --dataset_name ${DATASET_NAME} \
    --pooling ${POOLING} \
    --esm_model ${ESM_MODEL} \
    --head_hidden_dims ${HEAD_HIDDEN_DIMS} \
    --head_out_dim ${HEAD_OUT_DIM} \
    --head_dropout ${HEAD_DROPOUT} \
    --head_activation ${HEAD_ACTIVATION} \
    --head_norm ${HEAD_NORM} \
    --loss_type ${LOSS_TYPE} \
    --distance_metric ${DISTANCE_METRIC} \
    ${USE_LORA} \
    --esm_lr ${ESM_LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --batch_mode ${BATCH_MODE} \
    --batch_size ${BATCH_SIZE} \
    --genes_per_batch ${GENES_PER_BATCH} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --epochs ${EPOCHS} \
    --grad_clip ${GRAD_CLIP} \
    --eval_every_steps ${EVAL_EVERY_STEPS} \
    --centroid_subsample ${CENTROID_SUBSAMPLE} \
    --eval_distance ${EVAL_DISTANCE} \
    --held_out_gene_frac ${HELD_OUT_GENE_FRAC} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_mode ${WANDB_MODE} \
    --seed ${SEED}
"
