#!/bin/bash

MODEL_NAME=${MODEL_NAME:-your-affiliation/your-ckpt}

# Split MODEL_NAME into AFFILIATION and CKPT_NAME
AFFILIATION=$(cut -d'/' -f1 <<< "$MODEL_NAME")
CKPT_NAME=$(cut -d'/' -f2 <<< "$MODEL_NAME")

# Set and export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

export OPENAI_API_KEY=${OPENAI_API_KEY:-your_api_key}

LOG_DIR="./log"
LOG_FILE="$LOG_DIR/alpaca_api.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"; }

set -euo pipefail

sparsities=(0.7 0.8 0.9)
scenarios=(ideal prac)
methods=(cats m-countdown d-countdown)

total_runs=$(( ${#models[@]} * ${#sparsities[@]} * ${#scenarios[@]} * ${#methods[@]} ))
run=0

for SPARSITY_RATIO in "${sparsities[@]}"; do
  for SCENARIO in "${scenarios[@]}"; do
    for METHOD in "${methods[@]}"; do
      run=$(( run + 1 ))
      log "Run ${run}/${total_runs}"
      log "Evaluating ${MODEL_NAME} with ${METHOD} and sparsity ${SPARSITY_RATIO} in ${SCENARIO} scenario"
      alpaca_eval evaluate \
        --model_outputs "./alpaca_eval/results/${CKPT_NAME}/${METHOD}/${SPARSITY_RATIO}_${SCENARIO}/model_outputs.json" \
        --reference_outputs "./alpaca_eval/results/${CKPT_NAME}/dense/model_outputs.json" \
        --annotators_config "weighted_alpaca_eval_gpt4o"
    done
  done
done



