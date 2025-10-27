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
LOG_FILE="$LOG_DIR/prob.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"; }

for TASK in arc_easy arc_challenge winogrande piqa openbookqa hellaswag truthfulqa_mc1; do
    log "Running DENSE_PROB, TASK=${TASK}"
    lm_eval --model hf \
            --model_args pretrained=${MODEL_NAME}, attn_implementation="flash_attention_2" \
            --tasks ${TASK} \
            --device cuda:0 \
            --output_path sp_ideal/${CKPT_NAME}/dense_eval \
            --batch_size 16 \
            --trust_remote_code \
            --log_samples \
            --yaml_dir ./config/trace.yaml \
            --prior_dataset Eval
done


log "Running DENSE_GEN, TASK=gsm8k_cot"
lm_eval --model hf \
        --model_args pretrained=${MODEL_NAME}, attn_implementation="flash_attention_2" \
        --tasks gsm8k_cot \
        --device cuda:0 \
        --output_path sp_ideal/${CKPT_NAME}/dense_eval \
        --batch_size 8 \
        --trust_remote_code \
        --log_samples \
        --yaml_dir ./config/trace_gen.yaml \
        --prior_dataset Eval 


for TASK in arc_easy arc_challenge winogrande piqa openbookqa hellaswag truthfulqa_mc1; do
    log "Running TRACE_TRAIN_DSET, TASK=${TASK}"
    lm_eval --model hf \
            --model_args pretrained=${MODEL_NAME}, attn_implementation="flash_attention_2" \
            --tasks ${TASK} \
            --device cuda:0 \
            --output_path sp_ideal/${CKPT_NAME}/dense_train \
            --batch_size 16 \
            --trust_remote_code \
            --log_samples \
            --yaml_dir ./config/trace.yaml \
            --prior_dataset Train \
            --save_tensors 
done

for METHOD in dejavu cats m-countdown d-countdown; do
    for SPARSITY_RATIO in 0.7 0.8 0.9; do
        for TASK in arc_easy arc_challenge winogrande piqa openbookqa hellaswag truthfulqa_mc1; do
            log "Running ARTIFACTS_TRAIN_DSET, METHOD=${METHOD}, TASK=${TASK}, SPARSITY_RATIO=${SPARSITY_RATIO}"
            lm_eval --model hf \
                    --model_args pretrained=${MODEL_NAME}, attn_implementation="flash_attention_2" \
                    --tasks ${TASK} \
                    --device cuda:0 \
                    --output_path sp_ideal/${CKPT_NAME}/${METHOD}_ideal_train_${SPARSITY_RATIO} \
                    --batch_size 16 \
                    --trust_remote_code \
                    --log_samples \
                    --yaml_dir ./config/${METHOD}/ideal/${SPARSITY_RATIO}.yaml \
                    --prior_dataset Train
        done
    done
done

for METHOD in dejavu cats m-countdown d-countdown; do
    for SPARSITY_RATIO in 0.7 0.8 0.9; do
        for TASK in arc_easy arc_challenge winogrande piqa openbookqa hellaswag truthfulqa_mc1; do
            log "Running METHOD=${METHOD}, TASK=${TASK}, SPARSITY_RATIO=${SPARSITY_RATIO}"
            lm_eval --model hf \
                    --model_args pretrained=${MODEL_NAME}, attn_implementation="flash_attention_2" \
                    --tasks ${TASK} \
                    --device cuda:0 \
                    --output_path sp_ideal/${CKPT_NAME}/${METHOD}_ideal_eval_${SPARSITY_RATIO} \
                    --batch_size 16 \
                    --trust_remote_code \
                    --log_samples \
                    --yaml_dir ./config/${METHOD}/ideal/${SPARSITY_RATIO}.yaml \
                    --prior_dataset Eval
        done
    done
done

for METHOD in dejavu cats m-countdown d-countdown; do
    for SPARSITY_RATIO in 0.7 0.8 0.9; do
        for TASK in gsm8k_cot; do
            log "Running METHOD=${METHOD}, TASK=${TASK}, SPARSITY_RATIO=${SPARSITY_RATIO}"
            lm_eval --model hf \
                    --model_args pretrained=${MODEL_NAME}, attn_implementation="flash_attention_2" \
                    --tasks ${TASK} \
                    --device cuda:0 \
                    --output_path sp_ideal/${CKPT_NAME}/${METHOD}_ideal_eval_${SPARSITY_RATIO} \
                    --batch_size 8 \
                    --trust_remote_code \
                    --log_samples \
                    --yaml_dir ./config/${METHOD}/ideal/${SPARSITY_RATIO}_gen.yaml \
                    --prior_dataset Eval
        done
    done
done