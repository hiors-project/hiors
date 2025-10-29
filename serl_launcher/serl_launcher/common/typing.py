from typing import Any, Callable, Dict, Sequence, Union, TypedDict

import torch
import torch.nn as nn
import numpy as np

PRNGKey = Any  # PyTorch uses different random state management
Params = nn.Module  # Using torch.nn.Module instead of nnx.Module
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
Array = Union[np.ndarray, torch.Tensor]  # Using torch.Tensor instead of jnp.ndarray
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
# A method to be passed into TrainState.__call__
ModuleMethod = Union[str, Callable, None]
