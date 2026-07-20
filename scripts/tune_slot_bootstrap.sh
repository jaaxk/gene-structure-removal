#!/bin/bash
# Wait for an already-running run's PID, log its result, then hand this slot
# off to tune_slot_runner.sh to keep consuming a shared queue independently.
# Usage: tune_slot_bootstrap.sh <existing_pid> <existing_run_name> <existing_hyp> \
#          <launcher_script> <queue_file> [launcher_extra_args...]
set -uo pipefail
cd /home/jv2807/gene_structure
PID="$1"; shift
RUN_NAME="$1"; shift
HYP="$1"; shift
LAUNCHER="$1"; shift
QUEUE="$1"; shift

while kill -0 "$PID" 2>/dev/null; do
  sleep 15
done

metrics="/scratch/jv2807/gene_structure_removal/runs/$RUN_NAME/final_full_metrics.json"
if [ -f "$metrics" ]; then
  singularity exec --overlay /scratch/jv2807/dms_singularity/dms_contrastive.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c \
    "source /ext3/env.sh && python scripts/tune_log_result.py $RUN_NAME $HYP"
  echo "[bootstrap] DONE $RUN_NAME"
else
  echo "[bootstrap] FAILED $RUN_NAME (no final_full_metrics.json)"
fi

exec bash scripts/tune_slot_runner.sh "$LAUNCHER" "$QUEUE" "$@"
