#!/bin/bash

MODEL_NAME=${MODEL_NAME:-your-affiliation/your-ckpt}

# Split MODEL_NAME into AFFILIATION and CKPT_NAME
AFFILIATION=$(cut -d'/' -f1 <<< "$MODEL_NAME")
CKPT_NAME=$(cut -d'/' -f2 <<< "$MODEL_NAME")

# Set and export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

export OPENAI_API_KEY=${OPENAI_API_KEY:-your_api_key}

LOG_DIR="./log/$CKPT_NAME"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/alpaca_gen.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"; }

# Count total tasks
total_task_method_times_sparsityratio_times_scenario () {
    local dir_count=0
    for METHOD in d-countdown cats m-countdown; do
        for SPARSITY_RATIO in 0.7 0.8 0.9; do
            for SCENARIO in ideal prac; do
                ((dir_count++))
            done
        done
    done
    ((dir_count++))
    echo $dir_count
}

TOTAL_TASKS=$(total_task_method_times_sparsityratio_times_scenario)
CURRENT_TASK=0

for METHOD in d-countdown cats m-countdown; do
    for SPARSITY_RATIO in 0.7 0.8 0.9; do
        for SCENARIO in ideal prac; do
            CURRENT_TASK=$((CURRENT_TASK + 1))
            START_TIME=$(date +%s)

            YAML_PATH="./alpaca_eval/src/alpaca_eval/models_configs/${CKPT_NAME}/${METHOD}/${SPARSITY_RATIO}_${SCENARIO}.yaml"
            OUTPUT_PATH="./alpaca_eval/results/${CKPT_NAME}/${METHOD}/${SPARSITY_RATIO}_${SCENARIO}"

            log "[Progress: $CURRENT_TASK / $TOTAL_TASKS] DEVICE=0, Starting METHOD=${METHOD}, SPARSITY_RATIO=${SPARSITY_RATIO}, SCENARIO=${SCENARIO}"

            alpaca_eval evaluate_from_model --model_configs "$YAML_PATH" \
                                            --output_path "$OUTPUT_PATH" \
                                            --chunksize 64


            END_TIME=$(date +%s)
            RUNTIME=$((END_TIME - START_TIME))
            log "[Progress: $CURRENT_TASK / $TOTAL_TASKS] Completed in ${RUNTIME} seconds"
        done
    done
done

alpaca_eval evaluate_from_model --model_configs "./alpaca_eval/src/alpaca_eval/models_configs/${CKPT_NAME}/dense/config.yaml" \
                                --output_path "./alpaca_eval/results/${CKPT_NAME}/dense" \
                                --chunksize 64



