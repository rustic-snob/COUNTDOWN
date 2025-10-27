#!/bin/bash

MODEL_NAME=${MODEL_NAME:-your-affiliation/your-ckpt}

# Split MODEL_NAME into AFFILIATION and CKPT_NAME
AFFILIATION=$(cut -d'/' -f1 <<< "$MODEL_NAME")
CKPT_NAME=$(cut -d'/' -f2 <<< "$MODEL_NAME")

# Set and export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

LOG_DIR="./log/$CKPT_NAME"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/bench_prac.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"; }

# === Directory Variables ===
PREDICTORS_DIR="./results/countdown_train/${AFFILIATION}_${CKPT_NAME}"
TRACES_DIR="./DATA/user/countdown/${AFFILIATION}__${CKPT_NAME}"

# === Task-Specific Configurations ===
declare -A TASK_BATCH_SIZE=(
    ["gsm8k_cot"]=16
    ["arc_easy"]=32
    ["arc_challenge"]=32
    ["winogrande"]=32
    ["piqa"]=32
    ["openbookqa"]=32
    ["hellaswag"]=32
    ["truthfulqa_mc1"]=32
)

declare -A TASK_YAML_SUFFIX=(
    ["gsm8k_cot"]="_gen"
    ["arc_easy"]=""
    ["arc_challenge"]=""
    ["winogrande"]=""
    ["piqa"]=""
    ["openbookqa"]=""
    ["hellaswag"]=""
    ["truthfulqa_mc1"]=""
)

# === Progress and Runtime Tracking ===
TOTAL_TASKS=0
CURRENT_TASK=0
PREDICTORS_DIR_LIST=()
TRACES_DIR_LIST=()
FILTERED_PREDICTORS_DIR_LIST=()
FILTERED_TRACES_DIR_LIST=()

# === Function: Calculate Total Tasks and Extract Directories ===
calculate_total_tasks() {
    local dir_count=0
    local -n dir_list_ref=$2  # Pass array by reference
    
    if [[ "$1" == "PREDICTORS_DIR" ]]; then
        while read -r dir; do
            dir_list_ref+=("$dir")
        done < <(find "${PREDICTORS_DIR}" -mindepth 5 -maxdepth 5 -type d)
    elif [[ "$1" == "TRACES_DIR" ]]; then
        for METHOD in m-countdown cats; do
            while read -r dir; do
                dir_list_ref+=("$dir")
            done < <(find "${TRACES_DIR}/${METHOD}" -mindepth 2 -maxdepth 2 -type d)
        done
    fi
    
    local task_count=$(( ${#dir_list_ref[@]} * ${#TASK_BATCH_SIZE[@]} ))
    TOTAL_TASKS=$((TOTAL_TASKS + task_count))
    log "Total tasks for $1: $task_count"
}

# === Function: Filter Directories by Pattern ===
filter_dirs_by_pattern() {
    local -n input_dirs=$1  # Array passed by reference
    local -n output_dirs=$2 # Filtered array passed by reference
    local pattern="$3"      # Pattern to filter (e.g., 'foo')

    output_dirs=()  # Clear the output array

    for dir in "${input_dirs[@]}"; do
        if [[ "$dir" != *"$pattern"* ]]; then
            output_dirs+=("$dir")
        else
            log "🛑 Skipping directory (matched pattern '$pattern'): $dir"
        fi
    done
}

update_total_tasks() {
    TOTAL_TASKS=0  # Reset the total task count
    
    local predictors_task_count=$(( ${#FILTERED_PREDICTORS_DIR_LIST[@]} * ${#TASK_BATCH_SIZE[@]} ))
    local traces_task_count=$(( ${#FILTERED_TRACES_DIR_LIST[@]} * ${#TASK_BATCH_SIZE[@]} ))
    
    TOTAL_TASKS=$(( predictors_task_count + traces_task_count ))
    
    log "🔄 Updated TOTAL_TASKS after filtering: $TOTAL_TASKS"
    log "Filtered PREDICTORS_DIR_LIST: ${#FILTERED_PREDICTORS_DIR_LIST[@]} directories, Total tasks: $predictors_task_count"
    log "Filtered TRACES_DIR_LIST: ${#FILTERED_TRACES_DIR_LIST[@]} directories, Total tasks: $traces_task_count"
}

# === Function: Run Tasks ===
run_tasks() {
    local METHOD="${1:-default_method}"
    local RANK="${2:-}"
    local SPARSITY_RATIO="${3:-}"
    local COUNTDOWN_LEARN="${4:-}"
    local TIMECODE="${5:-}"
    local PREDICTOR_DIR="$6"

    for TASK in "${!TASK_BATCH_SIZE[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        local START_TIME=$(date +%s)

        local BATCH_SIZE=${TASK_BATCH_SIZE[$TASK]}
        local YAML_SUFFIX=${TASK_YAML_SUFFIX[$TASK]}
        local YAML_PATH="./config/${METHOD}/prac/${SPARSITY_RATIO}${YAML_SUFFIX}.yaml"
        
        local OUTPUT_PATH="sp_prac/${CKPT_NAME}/${METHOD}"
        if [[ -n $COUNTDOWN_LEARN && -n $TIMECODE && -n $RANK ]]; then
            OUTPUT_PATH+="_${TIMECODE}"
        fi
        OUTPUT_PATH+="_pred_eval_${SPARSITY_RATIO}"

        log "[Progress: $CURRENT_TASK / $TOTAL_TASKS] DEVICE=2, Starting TASK=${TASK}, METHOD=${METHOD}, SPARSITY_RATIO=${SPARSITY_RATIO}, TIMECODE=${TIMECODE}"

        
        lm_eval --model hf \
                --model_args pretrained=${MODEL_NAME},cache_dir="./MODELS/",attn_implementation="flash_attention_2" \
                --tasks ${TASK} \
                --device cuda:0 \
                --output_path ${OUTPUT_PATH} \
                --batch_size ${BATCH_SIZE} \
                --trust_remote_code \
                --log_samples \
                --yaml_dir ${YAML_PATH} \
                --prior_dataset Eval \
                --predictor_dir ${PREDICTOR_DIR} \
                --trace_base_dir ${OUTPUT_PATH} \
        

        local END_TIME=$(date +%s)
        local RUNTIME=$((END_TIME - START_TIME))
        log "[Progress: $CURRENT_TASK / $TOTAL_TASKS] Completed TASK=${TASK} in ${RUNTIME} seconds"
    done
}

# === Calculate and Filter Directories ===
calculate_total_tasks "PREDICTORS_DIR" PREDICTORS_DIR_LIST

calculate_total_tasks "TRACES_DIR" TRACES_DIR_LIST

filter_dirs_by_pattern TRACES_DIR_LIST FILTERED_TRACES_DIR_LIST "prac"

# === Update Total Tasks ===
update_total_tasks

# === Phase 1: Countdown Train Directories ===
for PREDICTOR_DIR in "${FILTERED_PREDICTORS_DIR_LIST[@]}"; do
    IFS='/' read -r _ _ _ _ _ _ METHOD RANK SPARSITY_RATIO COUNTDOWN_LEARN TIMECODE <<<"${PREDICTOR_DIR}"
    run_tasks "${METHOD}" "${RANK}" "${SPARSITY_RATIO}" "${COUNTDOWN_LEARN}" "${TIMECODE}" "${PREDICTOR_DIR}"
done

# === Phase 2: Countdown Trace Directories ===
for PREDICTOR_DIR in "${FILTERED_TRACES_DIR_LIST[@]}"; do
    IFS='/' read -r _ _ _ _ _ _ METHOD _ SPARSITY_RATIO <<<"${PREDICTOR_DIR}"
    run_tasks "${METHOD}" "" "${SPARSITY_RATIO}" "" "" "${PREDICTOR_DIR}"
done