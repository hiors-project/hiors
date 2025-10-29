### Installation Guide of Hi-ORS

1. **Clone the repository**:

```bash
git clone this repo url
cd hiors
```

2. [Optional] **Setup uv Python Package Manager**:

You can skip this step if you already have uv installed. 
We use uv to manage Python environments and dependencies for its efficiency and simplicity. Perhaps the simplest way to install uv is:
```bash
pip install uv
```

| NOTE: If you want to use mirror:
```bash
mkdir ~/.config/uv
vim ~/.config/uv/uv.toml
# [[index]]
# url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
# default = true
```

3. **Create a virtual env and install dependencies**:

```bash
uv sync
# active the environment
source .venv/bin/activate

# [Optional] Fix some possible issues
uv pip install numpy==1.26.4

# [Optional] For lerobot visualization:
uv pip install rerun-sdk==0.22.1

# Maybe you should add comment out the following line in lerobot_dataset.py
# self.stats = aggregate_stats(self._datasets)
```
It may take a while to install all the dependencies. All packages (`serl_launcher`, `serl_robot_infra`, `openpi-client`) are automatically installed in editable mode. It will takes up to 10GB of disk space.

4. **Set Environment Variables**:

We use wandb to track experiments, run the following command to login:
```bash
wandb login
```

We use lerobot dataset for training, please set the following environment variables in your `~/.bashrc` or `~/.zshrc` file:
```bash
export HIORS_PATH=<your_path>/hiors
export HF_LEROBOT_HOME=$HIORS_PATH/../cache_huggingface/lerobot
export HF_DATASETS_CACHE=$HIORS_PATH/../cache_hf_datasets
```

As pi0 uses Google's PaliGemma bachbone, you may need to first get granted to it at this website: https://huggingface.co/google/paligemma-3b-pt-2. The `modeling_pi0.py` will automatically download the model weights to `~/.cache/huggingface/hub` folder. If you have limited bandwidth, you can manually download the model weights using git lfs:
```bash
git lfs clone https://huggingface.co/google/paligemma-3b-pt-2
```
Remember to change the loading path in `modeling_pi0.py` if you want to use local weights.


