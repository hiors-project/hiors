#!/bin/bash

export HYDRA_FULL_ERROR=1

UNIQUE_IDENTIFIER=reward

# Use "1" for single GPU, "1,2,3" for multi GPU
CUDA_VISIBLE_DEVICES=7

CHECKPOINT_PATH="$HIORS_PATH/examples/experiments/dobot_pnp/ckpt/$UNIQUE_IDENTIFIER"
DEEPSPEED_CONFIG_PATH="$HIORS_PATH/config/zero2.json"

# Automatically infer GPU mode from CUDA_VISIBLE_DEVICES
GPU_NUMS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$GPU_NUMS" -eq 1 ]; then
    GPU_MODE="single"
else
    GPU_MODE="multi"
fi

echo "EXP: $UNIQUE_IDENTIFIER"
echo "GPU_NUMS: $GPU_NUMS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU_MODE: $GPU_MODE"

COMMON_ARGS="../../train_reward_classifier.py \
    --config-path="$HIORS_PATH/config" \
    --config-name=reward_classifier \
    task=dobot_pnp \
    unique_identifier=$UNIQUE_IDENTIFIER \
    checkpoint_path=$CHECKPOINT_PATH \
    training.gradient_accumulation_steps=1 \
    training.deepspeed_config=$DEEPSPEED_CONFIG_PATH \
    training.num_gpus=$GPU_NUMS \
    logging.logger=wandb"

# logger: wandb, tensorboard, debug
# --use_deepspeed is needed for single GPU training, ref: https://github.com/axolotl-ai-cloud/axolotl/issues/1211

# Run based on inferred GPU mode
if [ "$GPU_MODE" = "single" ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch \
        --num_processes 1 \
        --use_deepspeed \
        --gradient_accumulation_steps 1 \
        --main_process_port 29502 \
        --machine_rank 0 \
        $COMMON_ARGS "$@"
else
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch \
        --num_processes $GPU_NUMS \
        --multi_gpu \
        --gradient_accumulation_steps 1 \
        --main_process_port 29502 \
        --machine_rank 0 \
        $COMMON_ARGS "$@"
fi
