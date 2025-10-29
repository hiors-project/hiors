from functools import partial
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LagrangeMultiplier(nn.Module):
    def __init__(
        self,
        init_value: float = 1.0,
        constraint_shape: Sequence[int] = (),
        constraint_type: str = "eq",  # One of ("eq", "leq", "geq")
        parameterization: Optional[str] = None,  # One of ("softplus", "exp"), or None for equality constraints
    ):
        super().__init__()
            
        self.constraint_shape = constraint_shape
        self.constraint_type = constraint_type
        self.parameterization = parameterization
        
        # Validate parameterization based on constraint type
        if constraint_type != "eq":
            assert init_value > 0, "Inequality constraints must have non-negative initial multiplier values"
            
            if parameterization == "softplus":
                transformed_init = torch.log(torch.exp(torch.tensor(init_value)) - 1)
            elif parameterization == "exp":
                transformed_init = torch.log(torch.tensor(init_value))
            elif parameterization == "none":
                transformed_init = torch.tensor(init_value)
            else:
                raise ValueError(f"Invalid multiplier parameterization {parameterization}")
        else:
            assert parameterization is None, "Equality constraints must have no parameterization"
            transformed_init = torch.tensor(init_value)
            
        # Create the parameter
        if len(constraint_shape) == 0:
            # Scalar case
            self.lagrange_param = nn.Parameter(transformed_init.clone().detach().unsqueeze(0))
        else:
            # Vector case
            self.lagrange_param = nn.Parameter(torch.full(constraint_shape, transformed_init.item()))
    
    def forward(self, *, lhs: Optional[torch.Tensor] = None, rhs: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Apply parameterization transformation
        if self.constraint_type != "eq":
            if self.parameterization == "softplus":
                multiplier = F.softplus(self.lagrange_param)
            elif self.parameterization == "exp":
                multiplier = torch.exp(self.lagrange_param)
            elif self.parameterization == "none":
                multiplier = self.lagrange_param
            else:
                raise ValueError(f"Invalid multiplier parameterization {self.parameterization}")
        else:
            multiplier = self.lagrange_param
        
        # Return raw multiplier if no lhs provided
        if lhs is None:
            return multiplier
            
        # Compute Lagrange penalty
        if rhs is None:
            rhs = torch.zeros_like(lhs)
            
        diff = lhs - rhs
        
        # if multiplier is [1] and diff is scalar, squeeze multiplier
        # multiplier = multiplier.squeeze(0)
        multiplier = multiplier[0]

        if self.constraint_type == "eq":
            return multiplier * diff
        elif self.constraint_type == "geq":
            return multiplier * diff
        elif self.constraint_type == "leq":
            return -multiplier * diff
        else:
            raise ValueError(f"Invalid constraint type {self.constraint_type}")



GeqLagrangeMultiplier = partial(
    LagrangeMultiplier, constraint_type="geq", parameterization="softplus"
)

LeqLagrangeMultiplier = partial(
    LagrangeMultiplier, constraint_type="leq", parameterization="softplus"
)

BetterLeqLagrangeMultiplier = partial(
    LagrangeMultiplier, constraint_type="leq", parameterization="none"
)
