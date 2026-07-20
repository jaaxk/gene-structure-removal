#!/bin/bash
#SBATCH --job-name=gsr_pooling_sweep
#SBATCH --account=torch_pr_800_cds
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=jv2807@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# =====================================================================
# Compare all 3 pooling modes (concat, wt_mut_mean_concat,
# wt_subtracted_mean) under identical hyperparameters -- the best non-LoRA
# config from tune/results.jsonl (tune_h3_bs32, macro_spearman_mean=0.3104).
# One job, one GPU allocation, 3 sequential full (5-epoch) runs so all 3
# comparable llr_projection_figure.png + final_full_metrics.json outputs
# land in the same session. Keep this in sync with run/train.sbatch --
# same hyperparameter block, just looped over POOLING_MODES.
# =====================================================================
RUN_PREFIX=pooling_sweep
# concat already completed successfully (job 14422106); the other two failed
# on a since-fixed WtMeanCache bug (h5py rejected the duplicate/broadcast row
# indices used to read cached WT means) -- rerunning just those two.
POOLING_MODES="wt_mut_mean_concat wt_subtracted_mean"

# --- data / wild-types ---
FASTA_PATH=/scratch/jv2807/gene_structure_removal/data/human_uniref90.fasta
LIMIT_GENES=5000            # 0 = all length-capped genes
MAX_SEQ_LEN=400

# --- backbone / pooling ---
ESM_MODEL=esmc_600m
EMBEDDING_LAYER=-1

# --- labels (frozen-ESM likelihood) ---
SCORER=masked_marginal     # masked_marginal | wt_marginal | pll
SCORE_BATCH_SIZE=64        # embedding/scoring forward batch (H200 handles 64+)
VARIANTS_PER_GENE=150      # variants embedded + trained on per gene
MIN_VARIANTS_PER_GENE=10
QUARTILE_LOW=0.25
QUARTILE_HIGH=0.25

# --- head ---
HEAD_HIDDEN_DIMS="1024 512"
HEAD_OUT_DIM=256
HEAD_DROPOUT=0.1
HEAD_ACTIVATION=gelu       # relu | gelu | silu | tanh
HEAD_NORM=layernorm        # none | layernorm | batchnorm

# --- loss (tune_h3_bs32 winner) ---
LOSS_TYPE=contrastive_ce   # wt_anchored_bce | contrastive_ce | ntxent | triplet
DISTANCE_METRIC=cosine     # cosine | euclidean

# --- LoRA (off -- best non-LoRA hparams) ---
USE_LORA=""
ESM_LR=1e-5
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# --- batching / diversity ---
BATCH_MODE=gene_diverse    # gene_diverse | cross_gene
BATCH_SIZE=32
GENES_PER_BATCH=2          # cross_gene only

# --- optimization ---
LR=1e-3
WEIGHT_DECAY=0.0
EPOCHS=5
GRAD_CLIP=1.0

# --- embedding read mode ---
DEVICE=auto
EMBEDDINGS_MODE=stream
MAX_RESIDENT_GB=32
NUM_WORKERS=8
PREFETCH_FACTOR=4

# --- eval ---
EVAL_PER_EPOCH=8
CENTROID_SUBSAMPLE=2000
EVAL_DISTANCE=cosine
HELD_OUT_GENE_FRAC=0.2
DMS_PER_GENE_SUBSAMPLE=200
PRIMARY_METRIC=llr_projection/spearman

# --- tracking ---
WANDB_PROJECT=gene-structure-removal
WANDB_MODE=online
SEED=0

OVERLAY=/scratch/jv2807/dms_singularity/dms_contrastive.ext3
BASE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
REPO=/home/jv2807/gene_structure

for POOLING in ${POOLING_MODES}; do
  echo "=== [pooling_sweep] starting pooling=${POOLING} ==="
  singularity exec --nv --overlay ${OVERLAY}:ro ${BASE} /bin/bash -c "
    source /ext3/env.sh
    export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
    cd ${REPO}
    PYTHONPATH=src python scripts/train.py \
      --run_name ${RUN_PREFIX}_${POOLING} \
      --device ${DEVICE} \
      --embeddings_mode ${EMBEDDINGS_MODE} \
      --max_resident_gb ${MAX_RESIDENT_GB} \
      --num_workers ${NUM_WORKERS} \
      --prefetch_factor ${PREFETCH_FACTOR} \
      --fasta_path ${FASTA_PATH} \
      --limit_genes ${LIMIT_GENES} \
      --max_seq_len ${MAX_SEQ_LEN} \
      --esm_model ${ESM_MODEL} \
      --embedding_layer ${EMBEDDING_LAYER} \
      --pooling ${POOLING} \
      --scorer ${SCORER} \
      --score_batch_size ${SCORE_BATCH_SIZE} \
      --variants_per_gene ${VARIANTS_PER_GENE} \
      --min_variants_per_gene ${MIN_VARIANTS_PER_GENE} \
      --quartile_low ${QUARTILE_LOW} \
      --quartile_high ${QUARTILE_HIGH} \
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
      --eval_per_epoch ${EVAL_PER_EPOCH} \
      --centroid_subsample ${CENTROID_SUBSAMPLE} \
      --eval_distance ${EVAL_DISTANCE} \
      --held_out_gene_frac ${HELD_OUT_GENE_FRAC} \
      --dms_per_gene_subsample ${DMS_PER_GENE_SUBSAMPLE} \
      --primary_metric ${PRIMARY_METRIC} \
      --wandb_project ${WANDB_PROJECT} \
      --wandb_mode ${WANDB_MODE} \
      --seed ${SEED}
  "
  echo "=== [pooling_sweep] finished pooling=${POOLING} (exit $?) ==="
done
