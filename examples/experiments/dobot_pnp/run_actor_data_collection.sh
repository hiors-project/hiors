#!/bin/bash

export HYDRA_FULL_ERROR=1

UNIQUE_IDENTIFIER=data_collection

CHECKPOINT_PATH="$HIORS_PATH/examples/experiments/dobot_pnp/ckpt/$UNIQUE_IDENTIFIER"

echo "EXP: $UNIQUE_IDENTIFIER"

CUDA_VISIBLE_DEVICES=2 python ../../train_rlpd.py \
    --config-path="$HIORS_PATH/config" \
    --config-name=base \
    algo=sac_pi0 \
    task=dobot_pnp \
    unique_identifier=$UNIQUE_IDENTIFIER \
    checkpoint_path=$CHECKPOINT_PATH \
    actor=true \
    environment.save_video=true \
    environment.max_episode_length=100 \
    environment.wait_for_after_done_time=1 \
    environment.wait_for_after_reset_time=2 \
    steps_per_update=1000000 \
    buffer_period=100 \
    "$@"
    # eval_n_trajs=100 \
    # eval_checkpoint_step=1 \
