#!/bin/bash
# One independent slot: pop a config off a shared queue, launch it, wait ONLY
# for that run to finish, log it, then pop the next one. Runs standalone
# under Monitor so slots never wait on each other.
#
# Usage: tune_slot_runner.sh <launcher_script> <queue_file> [launcher_extra_args...]
#   launcher_script: scripts/tune_launch.sh | tune_launch_gpunode_frozen.sh | tune_launch_lora.sh
#   launcher_extra_args: e.g. a GPU jobid, prepended before run_name for the LoRA/gpu-frozen launchers
set -uo pipefail
cd /home/jv2807/gene_structure
LAUNCHER="$1"; shift
QUEUE="$1"; shift
LAUNCHER_EXTRA=("$@")

SLOT_TAG="slot_$$"

while true; do
  line=$(python3 scripts/tune_queue_pop.py "$QUEUE")
  if [ -z "$line" ]; then
    echo "[$SLOT_TAG] queue $QUEUE empty, exiting"
    break
  fi
  run_name=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['run_name'])")
  hyp=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('hyp',''))")
  args_str=$(echo "$line" | python3 -c "import json,sys,shlex; print(' '.join(shlex.quote(a) for a in json.load(sys.stdin)['args']))")

  echo "[$SLOT_TAG] launching $run_name ($hyp)"
  launch_out=$(bash "$LAUNCHER" "${LAUNCHER_EXTRA[@]}" "$run_name" $args_str)
  echo "[$SLOT_TAG] $launch_out"
  pid=$(echo "$launch_out" | grep -oP '\(pid \K[0-9]+')

  if [ -n "$pid" ]; then
    while kill -0 "$pid" 2>/dev/null; do
      sleep 15
    done
  fi

  metrics="/scratch/jv2807/gene_structure_removal/runs/$run_name/final_full_metrics.json"
  if [ -f "$metrics" ]; then
    singularity exec --overlay /scratch/jv2807/dms_singularity/dms_contrastive.ext3:ro \
      /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c \
      "source /ext3/env.sh && python scripts/tune_log_result.py $run_name $hyp"
    echo "[$SLOT_TAG] DONE $run_name"
  else
    echo "[$SLOT_TAG] FAILED $run_name (no final_full_metrics.json -- check log)"
  fi
done
