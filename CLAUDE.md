# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hi-ORS (Human-in-the-loop Online Rejection Sampling) is a reinforcement learning framework for robotic manipulation. The codebase integrates PyTorch-based RL algorithms with vision-language models (PI0 policy using PaliGemma backbone) for robot control. Built upon hil-serl and lerobot, it uses a distributed actor-learner architecture where actors collect rollouts from real robots while learners train policies on GPUs.

## Development Setup

### Environment Setup
```bash
# Install dependencies (requires uv package manager)
uv sync
source .venv/bin/activate

# Optional fixes
uv pip install numpy==1.26.4
uv pip install rerun-sdk==0.22.1  # For visualization

# Set required environment variables (add to ~/.bashrc or ~/.zshrc)
export HIORS_PATH=<your_path>/hiors
export HF_LEROBOT_HOME=$HIORS_PATH/../cache_huggingface/lerobot
export HF_DATASETS_CACHE=$HIORS_PATH/../cache_hf_datasets

# Login to wandb for experiment tracking
wandb login
```

### Testing
```bash
# Run unit tests
pytest tests/test_replay_buffer.py -s

# Run specific test markers
pytest -m manual  # Run manual tests
```

### Linting
```bash
# Run ruff linter/formatter
ruff check .
ruff format .
```

## Training Commands

### Actor-Learner Training (Standard Workflow)

The training process uses a distributed architecture with separate actor and learner processes:

**Actor process** (runs on machine with robot access):
```bash
cd examples/experiments/dobot_pnp/
bash run_actor.sh  # Collects rollouts from robot
```

**Learner process** (runs on machine with GPUs):
```bash
cd examples/experiments/dobot_pnp/
bash run_learner.sh  # Trains policy on collected data
```

**Quick start with tmux** (runs both actor and learner):
```bash
bash start_dobot.sh  # Starts both processes in tmux split panes
```

### Data Collection (Optional Pre-training)
```bash
cd examples/experiments/dobot_pnp/
bash run_actor_data_collection.sh  # Actor side
bash run_learner_data_collection.sh  # Learner side
```

### Reward Model Training (Optional)
```bash
cd examples/experiments/dobot_pnp/
bash train_reward.sh  # Check reward_classifier.yaml first
```

### Other Training Modes
```bash
# Rejection sampling
bash run_actor_reject_sampling.sh
bash run_learner_reject_sampling.sh

# Supervised fine-tuning
bash run_learner_sft.sh

# Evaluation
bash run_actor_eval.sh  # Update checkpoint_path first
```

### Training Configuration

All training scripts use Hydra configuration system:
- Base config: `config/base.yaml`
- Algorithm configs: `config/algo/*.yaml` (sac_pi0, reject_sampling_pi0)
- Task configs: `config/task/*.yaml` (dobot_pnp)
- Main training script: `examples/train_rlpd.py`

Common Hydra overrides in shell scripts:
```bash
python train_rlpd.py \
    algo=sac_pi0 \
    task=dobot_pnp \
    unique_identifier=experiment_name \
    checkpoint_path=/path/to/ckpt \
    learner=true/false \
    actor=true/false \
    training.num_gpus=4 \
    training.mixed_precision=bf16 \
    logging.logger=wandb
```

## Architecture Overview

### Core Components

**Workspace Structure:**
- `serl_launcher/`: RL algorithms, networks, data handling, and utilities
  - `agents/continuous/`: SAC-based agents (sac_pi0.py implements SAC with PI0 policy)
  - `networks/`: Neural network implementations (critics, MLPs, transformers, PI0 policy)
  - `data/`: Replay buffers, datasets, data stores (LeRobot dataset integration)
  - `utils/`: Training, logging, timer utilities
- `serl_robot_infra/`: Robot environment implementations
  - `dobot_env/`: Dobot robot environment and deployment code
- `packages/openpi-client/`: Client library for OpenPI vision-language model inference
- `examples/`: Training scripts and experiment configurations
  - `train_rlpd.py`: Main training entry point with distributed training support
  - `train_reward_classifier.py`: Reward model training
  - `experiments/dobot_pnp/`: Dobot pick-and-place experiment scripts

**Key Design Patterns:**
- **Actor-Learner Architecture**: Uses `agentlace` library for distributed RL with TrainerServer/TrainerClient pattern. Actors collect environment rollouts, learners train on GPUs, communicating via network.
- **LeRobot Integration**: Replay buffers are stored as LeRobot datasets for compatibility and visualization. Data is stored in HF_LEROBOT_HOME directory.
- **PI0 Policy**: The agent uses a pre-trained vision-language model (PI0 with PaliGemma backbone) as the policy, which is fine-tuned during SAC training. Requires Google PaliGemma access: https://huggingface.co/google/paligemma-3b-pt-224
- **Multi-GPU Training**: Uses Accelerate library with DeepSpeed for distributed training across multiple GPUs (see config/zero2.json).

### Data Flow

1. **Actor Process**:
   - Connects to robot environment (real or simulated)
   - Queries learner for action predictions via TrainerClient
   - Collects transitions (observation, action, reward, next_observation)
   - Sends transitions to learner's replay buffer via data store

2. **Learner Process**:
   - Receives transitions from actors via QueuedDataStore
   - Stores in ReplayBuffer (backed by LeRobot dataset)
   - Samples batches for training (mixed from replay buffer and demo buffer)
   - Updates PI0 policy and critic networks using SAC algorithm
   - Serves actor requests for action predictions via TrainerServer
   - Periodically saves checkpoints and replay buffers

3. **Observation Format**:
   - Images: cam_high, cam_left_wrist, cam_right_wrist (transformed from [B,T,H,W,C] to [B,T,C,H,W] for PI0)
   - State: robot proprioceptive state (joint positions, gripper state)
   - Task: language prompt string (e.g., "put the objects in the box")

### Important Implementation Details

- **Mixed Precision Training**: Uses bfloat16 by default for memory efficiency
- **Gradient Accumulation**: Supported for effective larger batch sizes
- **Replay Buffer**: Capacity set to 200k transitions, backed by LeRobot dataset format
- **Environment Variables**: HIORS_PATH, HF_LEROBOT_HOME, HF_DATASETS_CACHE must be set
- **Port Configuration**: TrainerConfig uses ports 5588 (requests) and 5589 (broadcast) for actor-learner communication
- **IP Configuration**: Set `ip=<learner_ip>` in config when running actor/learner on different machines

## Dataset Visualization

Visualize collected datasets using LeRobot's visualization tools:

```bash
# On remote machine where dataset is stored
python -m lerobot.scripts.visualize_dataset \
    --repo-id "replay_buffer" \
    --root "$HF_LEROBOT_HOME/dobot/replay_buffer" \
    --local-files-only 1 \
    --mode distant \
    --ws-port 9087 \
    --episode-index 0

# On local machine (after port forwarding)
rerun ws://localhost:9087
```

Merge training timelapse videos:
```bash
python tools/merge_video.py \
    --video_folder examples/experiments/dobot_pnp/videos \
    --output_path examples/experiments/dobot_pnp/output.mp4 \
    --fps 5
```

## Configuration Notes

- Python version: 3.11 (required)
- UV workspace: Packages in `packages/`, `serl_robot_infra/`, `serl_launcher/` installed as editable
- Ruff config: Line length 120, extensive linting rules (see pyproject.toml)
- Pytest markers: Use `@pytest.mark.manual` for tests requiring manual execution

## When Modifying Training Code

- Training loop is in `examples/train_rlpd.py` with Accelerate integration
- Agent implementations are in `serl_launcher/serl_launcher/agents/continuous/`
- Network architectures in `serl_launcher/serl_launcher/networks/`
- Replay buffer logic in `serl_launcher/serl_launcher/data/replay_buffer.py`
- Robot environments in `serl_robot_infra/dobot_env/envs/`
- Always test with unit tests before running full training
- Consider disk space: full installation ~10GB, datasets can be large
