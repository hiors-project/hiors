import datetime
import tempfile
import os
from copy import copy
from socket import gethostname
from typing import Dict, Any, Optional

import absl.flags as flags
import ml_collections
import numpy as np


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


class TensorBoardLogger(object):
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.project = "hilserl"  # TensorBoard log directory name
        config.exp_descriptor = ""  # Experiment name
        config.unique_identifier = ""  # Unique identifier for the run
        config.log_dir = None  # Base log directory (will be created if None)
        return config

    def __init__(
        self,
        tensorboard_config,
        variant,
        tensorboard_output_dir=None,
        debug=False,
    ):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard logging requires torch to be installed. "
                "Install with: pip install torch tensorboard"
            )
        
        self.config = tensorboard_config
        self.config.unique_identifier += datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.config.experiment_id = (
            self.experiment_id
        ) = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"

        print("TensorBoard Config:", self.config)

        if tensorboard_output_dir is None:
            if self.config.log_dir is not None:
                tensorboard_output_dir = self.config.log_dir
            else:
                tensorboard_output_dir = tempfile.mkdtemp()

        # Create the full log directory path
        self.log_dir = os.path.join(
            tensorboard_output_dir, 
            self.config.project, 
            self.experiment_id
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        self.debug = debug
        if not debug:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"TensorBoard logging to: {self.log_dir}")
            print(f"Start TensorBoard with: tensorboard --logdir {tensorboard_output_dir}")
        else:
            self.writer = None
            print("TensorBoard logging disabled (debug mode)")

        # Log configuration and variant as text
        if not debug:
            self._log_config()

    def _log_config(self):
        """Log configuration and variant information as text."""
        if self.writer is None:
            return
            
        # Log variant (hyperparameters) as text
        variant_text = self._dict_to_text(self._variant, "Variant/Hyperparameters")
        self.writer.add_text("config/variant", variant_text, 0)
        
        # Log flags if available
        if flags.FLAGS.is_parsed():
            flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
            flag_text = self._dict_to_text(flag_dict, "Command Line Flags")
            self.writer.add_text("config/flags", flag_text, 0)

    def _dict_to_text(self, d: dict, title: str) -> str:
        """Convert dictionary to formatted text for TensorBoard."""
        lines = [f"## {title}", ""]
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"**{key}:**")
                for subkey, subvalue in value.items():
                    lines.append(f"  - {subkey}: {subvalue}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def log(self, data: dict, step: int = None):
        """Log data to TensorBoard."""
        if self.debug or self.writer is None:
            return
            
        data_flat = _recursive_flatten_dict(data)
        
        for key, value in zip(*data_flat):
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.writer.add_scalar(key, float(value), step)
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:  # scalar array
                    self.writer.add_scalar(key, float(value), step)
                elif value.ndim == 1:  # histogram
                    self.writer.add_histogram(key, value, step)
                else:
                    # For higher dimensional arrays, log as histogram
                    self.writer.add_histogram(key, value.flatten(), step)
            elif hasattr(value, 'item'):  # JAX arrays, torch tensors, etc.
                try:
                    scalar_value = float(value.item() if hasattr(value, 'item') else value)
                    self.writer.add_scalar(key, scalar_value, step)
                except (ValueError, TypeError):
                    # If conversion fails, try to log as histogram
                    try:
                        array_value = np.asarray(value)
                        if array_value.ndim == 0:
                            self.writer.add_scalar(key, float(array_value), step)
                        else:
                            self.writer.add_histogram(key, array_value.flatten(), step)
                    except Exception:
                        print(f"Warning: Could not log {key} with value {value} (type: {type(value)})")
            else:
                print(f"Warning: Unsupported data type for key '{key}': {type(value)}")

        # Flush to ensure data is written
        self.writer.flush()

    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
