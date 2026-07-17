#!/bin/bash
#SBATCH --job-name=gsr_warm
#SBATCH --account=torch_pr_800_cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

# =====================================================================
# GPU cache-warming (sustained-high-util ESM fill), then exit. Fills the
# per-gene score cache, the (mut,wt) embedding cache, and the DMS eval cache
# for the given config. Optional: a normal GPU train run self-warms too; use
# this to pre-fill a large set so head-training / HP sweeps can then run cheaply
# on a CPU node. Keep these vars in sync with run/train.sbatch.
# =====================================================================
FASTA_PATH=/scratch/jv2807/gene_structure_removal/data/human_uniref90.fasta
LIMIT_GENES=5000
MAX_SEQ_LEN=400
ESM_MODEL=esmc_600m
EMBEDDING_LAYER=-1
POOLING=concat
SCORER=masked_marginal
SCORE_BATCH_SIZE=16
VARIANTS_PER_GENE=150
MIN_VARIANTS_PER_GENE=10
QUARTILE_LOW=0.25
QUARTILE_HIGH=0.25
SEED=0

OVERLAY=/scratch/jv2807/dms_singularity/dms_contrastive.ext3
BASE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
REPO=/home/jv2807/gene_structure

singularity exec --nv --overlay ${OVERLAY}:ro ${BASE} /bin/bash -c "
  source /ext3/env.sh
  export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
  cd ${REPO}
  PYTHONPATH=src python scripts/train.py \
    --run_name warm --warm_only --device auto \
    --fasta_path ${FASTA_PATH} --limit_genes ${LIMIT_GENES} \
    --max_seq_len ${MAX_SEQ_LEN} --esm_model ${ESM_MODEL} \
    --embedding_layer ${EMBEDDING_LAYER} --pooling ${POOLING} \
    --scorer ${SCORER} --score_batch_size ${SCORE_BATCH_SIZE} \
    --variants_per_gene ${VARIANTS_PER_GENE} \
    --min_variants_per_gene ${MIN_VARIANTS_PER_GENE} \
    --quartile_low ${QUARTILE_LOW} --quartile_high ${QUARTILE_HIGH} \
    --wandb_mode disabled --seed ${SEED}
"
