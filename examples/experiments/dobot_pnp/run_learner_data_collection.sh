#!/bin/bash

export HYDRA_FULL_ERROR=1

UNIQUE_IDENTIFIER=data_collection

# Use "1" for single GPU, "1,2,3" for multi GPU
# CUDA_VISIBLE_DEVICES=3
CUDA_VISIBLE_DEVICES=3,4,5,6
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

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


COMMON_ARGS="../../train_rlpd.py \
    --config-path=$HIORS_PATH/config \
    --config-name=base \
    algo=sac_pi0 \
    task=dobot_pnp \
    unique_identifier=$UNIQUE_IDENTIFIER \
    checkpoint_path=$CHECKPOINT_PATH \
    learner=true \
    training.mixed_precision=bf16 \
    training.gradient_accumulation_steps=1 \
    training.deepspeed_config=$DEEPSPEED_CONFIG_PATH \
    training.num_gpus=$GPU_NUMS \
    logging.logger=wandb \
    environment.max_episode_length=100 \
    environment.wait_for_after_done_time=1 \
    environment.wait_for_after_reset_time=2 \
    steps_per_update=1000000 \
    buffer_period=100"

# logger: wandb, tensorboard, debug
# use_deepspeed is needed for single GPU training, ref: https://github.com/axolotl-ai-cloud/axolotl/issues/1211

# Run based on inferred GPU mode
if [ "$GPU_MODE" = "single" ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch \
        --num_processes 1 \
        --use_deepspeed \
        --mixed_precision bf16 \
        --gradient_accumulation_steps 1 \
        --main_process_port 29501 \
        --machine_rank 0 \
        $COMMON_ARGS "$@"
else
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch \
        --num_processes $GPU_NUMS \
        --multi_gpu \
        --mixed_precision bf16 \
        --gradient_accumulation_steps 1 \
        --main_process_port 29501 \
        --machine_rank 0 \
        $COMMON_ARGS "$@"
fi
