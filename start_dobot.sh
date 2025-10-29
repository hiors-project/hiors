#!/bin/bash

# Define the base directory
export HIORS_PATH=/mnt/disk_3/guanxing/hiors
export HF_LEROBOT_HOME=$HIORS_PATH/cache_huggingface/lerobot
export HF_DATASETS_CACHE=$HIORS_PATH/cache_hf_datasets

TMUX_NAME="dobot_torch_rl"

tmux kill-session -t $TMUX_NAME
tmux new-session -d -s $TMUX_NAME -n "dobot_workspace"

# Split the window into left and right panes
tmux split-window -h -t $TMUX_NAME:0

# Configure left pane (actor)
tmux send-keys -t $TMUX_NAME:0.0 "cd $HIORS_PATH" C-m
tmux send-keys -t $TMUX_NAME:0.0 "source .venv/bin/activate" C-m
tmux send-keys -t $TMUX_NAME:0.0 "export HF_LEROBOT_HOME=$HF_LEROBOT_HOME" C-m
tmux send-keys -t $TMUX_NAME:0.0 "export HF_DATASETS_CACHE=$HF_DATASETS_CACHE" C-m
tmux send-keys -t $TMUX_NAME:0.0 "export HIORS_PATH=$HIORS_PATH" C-m
tmux send-keys -t $TMUX_NAME:0.0 "cd examples/experiments/dobot_pnp" C-m
tmux send-keys -t $TMUX_NAME:0.0 "echo 'ACTOR'; bash run_actor.sh" C-m

# Configure right pane (learner)
tmux send-keys -t $TMUX_NAME:0.1 "cd $HIORS_PATH" C-m
tmux send-keys -t $TMUX_NAME:0.1 "source .venv/bin/activate" C-m
tmux send-keys -t $TMUX_NAME:0.1 "export HF_LEROBOT_HOME=$HF_LEROBOT_HOME" C-m
tmux send-keys -t $TMUX_NAME:0.1 "export HF_DATASETS_CACHE=$HF_DATASETS_CACHE" C-m
tmux send-keys -t $TMUX_NAME:0.1 "export HIORS_PATH=$HIORS_PATH" C-m
tmux send-keys -t $TMUX_NAME:0.1 "cd examples/experiments/dobot_pnp" C-m
tmux send-keys -t $TMUX_NAME:0.1 "echo 'LEARNER'; bash run_learner.sh" C-m

tmux a -t $TMUX_NAME

echo "Started $TMUX_NAME tmux session with left/right split layout"
