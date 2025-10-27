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
LOG_FILE="$LOG_DIR/train.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }


for EPOCHS in 10 20 40 80; do
    for LR in 0.001 0.0005; do
        for SPARSITY_RATIO in 0.7 0.8 0.9; do
            log "- TRAIN Running METHOD=d-countdown, SPARSITY_RATIO=${SPARSITY_RATIO}, EPOCHS=${EPOCHS}"
            python scripts/train.py --model_name ${MODEL_NAME} \
                                    --epochs ${EPOCHS} \
                                    --lr ${LR} \
                                    --countdown_method d-countdown \
                                    --countdown_rank 512 \
                                    --countdown_sparsity_ratio ${SPARSITY_RATIO}
        done
    done
done


for EPOCHS in 10 20 40 80; do
    for LR in 0.001 0.0005; do
        for SPARSITY_RATIO in 0.7 0.8 0.9; do
            log "- TRAIN Running METHOD=d-countdown, SPARSITY_RATIO=${SPARSITY_RATIO}, EPOCHS=${EPOCHS}"
            python scripts/train.py --model_name ${MODEL_NAME} \
                                    --epochs ${EPOCHS} \
                                    --lr ${LR} \
                                    --countdown_method d-countdown \
                                    --countdown_sparsity_ratio ${SPARSITY_RATIO} \
                                    --predictor_shape bitlinear
        done
    done
done