#!/usr/bin/env bash
# Creates the three W&B sweeps for Q2 and runs an agent for each, one after another.
# Usage: bash run_sweeps.sh [project-name]
set -euo pipefail
PROJECT=${1:-CS6886W-SEDL-Assignment1-VGG6}

run_sweep () {
  local yaml=$1 count=$2
  local id
  id=$(wandb sweep --project "$PROJECT" "$yaml" 2>&1 | tee /dev/stderr | sed 's/\x1b\[[0-9;]*m//g' | grep -o 'wandb agent [^ ]*' | awk '{print $3}')
  echo ">>> running sweep $id ($yaml, $count runs)"
  wandb agent --count "$count" "$id"
}

run_sweep sweeps/activation.yaml 5   # Q2(a)
run_sweep sweeps/optimizer.yaml 6    # Q2(b)
run_sweep sweeps/hparams.yaml 6      # Q2(c)
