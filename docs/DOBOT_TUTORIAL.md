# Hi-ORS Dobot Tutorial

## Usage

Note that both the actor and learner run on the remote machine.

1. [Optional] **Collect offline data**
Collect a set of human trajectoris for rlpd demo buffer and reward model.
```bash
cd examples/experiments/dobot_pnp/
bash run_actor_data_collection.sh

# in another terminal:
cd examples/experiments/dobot_pnp/
bash run_learner_data_collection.sh
```

2. [Optional] **Train reward model**
```bash
# Check reward_classifier.yaml first, then:
cd examples/experiments/dobot_pnp
bash train_reward.sh
```

3. **Train the policy**
```bash
# if you don't have real robots to connect, set this in dobot_env.py:
# MAX_TIMEOUT = 0.1

cd examples/experiments/dobot_pnp/
bash actor.sh

# in another terminal:
cd examples/experiments/dobot_pnp/
bash learner.sh
```
The default use HIL-SERL as the RL algorithm. You can check other scripts for different algorithms.

If you have real robot, you need to start your specific robot controller on the robot's local computer, the process is basically:
```bash
# window 1
# Run your specific robot controller (e.g., a moveit wrapper), 
# You can also incorporate human interventions in your controller,
# as Hi-ORS records actions directly from the pose of the robot.
# We omit this step here.

# window 2
python <your_path>/inference_xtrainer.py
```

5. **Eval the policy**
```bash
# Remember to change the checkpoint_path in run_actor_eval.sh
bash run_actor_eval.sh
```

## Debugging

1. **Unit test**:
```bash
pytest tests/test_replay_buffer.py -s
```

2. **Check collected dataset**:

This check requires `rerun` for visualization. Run this on the remote machine where the dataset is stored:
```bash
# Check replay buffer (both successful and unsuccessful)
python -m lerobot.scripts.visualize_dataset \
    --repo-id "replay_buffer" \
    --root "<your_path>/cache_huggingface/lerobot/dobot/replay_buffer" \
    --local-files-only 1 \
    --mode distant \
    --ws-port 9087 \
    --episode-index 0

# Check successful buffer
python -m lerobot.scripts.visualize_dataset \
    --repo-id "successful_buffer" \
    --root "<your_path>/cache_huggingface/lerobot/dobot/successful_buffer" \
    --local-files-only 1 \
    --mode distant \
    --ws-port 9087 \
    --episode-index 0
```

Then run on local machine:
```bash
# forward port 9087 first
rerun ws://localhost:9087
```
If the rerun GUI is blank, restart port forwarding

3. **Check training timelapse video**:
Run this to merge seperate videos into a single video:
```bash
python tools/merge_video.py \
    --video_folder examples/experiments/dobot_pnp/videos \
    --output_path examples/experiments/dobot_pnp/output.mp4 \
    --fps 5
```
