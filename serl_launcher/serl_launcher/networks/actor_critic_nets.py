from typing import Optional
import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np


class ValueCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        network: nn.Module,
        network_output_dim: int,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.network = network
        self.network_output_dim = network_output_dim
        self.init_final = init_final
        
        # Create the value layer
        self.value_layer = nn.Linear(network_output_dim, 1)
        if self.init_final is not None:
            nn.init.uniform_(self.value_layer.weight, -self.init_final, self.init_final)
            nn.init.uniform_(self.value_layer.bias, -self.init_final, self.init_final)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        outputs = self.network(self.encoder(observations))
        value = self.value_layer(outputs)
        return value.squeeze(-1)


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions):
        if actions.dim() == 3:
            # Handle multiple actions per state
            batch_size = observations.shape[0]
            num_actions = actions.shape[1]
            
            # Expand observations to match actions
            obs_expanded = observations.unsqueeze(1).expand(-1, num_actions, -1)
            obs_flat = obs_expanded.reshape(batch_size * num_actions, -1)
            actions_flat = actions.reshape(batch_size * num_actions, -1)
            
            # Forward pass
            q_values_flat = forward(self, obs_flat, actions_flat)
            q_values = q_values_flat.reshape(batch_size, num_actions)
        else:
            q_values = forward(self, observations, actions)
        return q_values

    return wrapped


class Critic(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        network_output_dim: int,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.network = network
        self.network_output_dim = network_output_dim
        self.init_final = init_final
        
        # Create the value layer
        self.value_layer = nn.Linear(network_output_dim, 1)
        if self.init_final is not None:
            nn.init.uniform_(self.value_layer.weight, -self.init_final, self.init_final)
            nn.init.uniform_(self.value_layer.bias, -self.init_final, self.init_final)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        observations: [batch_size, obs_dim]
        actions: [batch_size, action_horizon, action_dim]
        """
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)    # [batch_size, encoder_output_dim]

        # [B, T, A] -> [B, T*A]
        B, T, A = actions.shape

        # mlp
        actions = actions.reshape(B, T*A)
        inputs = torch.cat([obs_enc, actions], dim=-1)    # [B, encoder_output_dim + T*A]

        # transformer
        # inputs = torch.cat([actions, obs_enc.unsqueeze(1).expand(-1, T, -1)], dim=-1)    # [B, T, A + encoder_output_dim]

        outputs = self.network(inputs)  # [B, out_dim]
        value = self.value_layer(outputs)
        return value.squeeze(-1)

    
class GraspCritic(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        network_output_dim: int,
        init_final: Optional[float] = None,
        output_dim: Optional[int] = 3,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.network = network
        self.network_output_dim = network_output_dim
        self.init_final = init_final
        self.output_dim = output_dim
        
        # Create the value layer
        self.value_layer = nn.Linear(network_output_dim, self.output_dim)
        if self.init_final is not None:
            nn.init.uniform_(self.value_layer.weight, -self.init_final, self.init_final)
            nn.init.uniform_(self.value_layer.bias, -self.init_final, self.init_final)
    
    def forward(
        self, 
        observations: torch.Tensor, 
    ) -> torch.Tensor:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)
        
        outputs = self.network(obs_enc)
        value = self.value_layer(outputs)
        return value # (batch_size, self.output_dim)


def ensemblize(cls, num_qs, out_axes=0):
    class EnsembleModule(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            
            # Create ensemble of modules
            self.modules_list = nn.ModuleList()
            for i in range(num_qs):
                # Create the module with only keyword arguments (excluding rngs)
                module_kwargs = {k: v for k, v in kwargs.items() if k != 'rngs'}
                self.modules_list.append(cls(**module_kwargs))

        def forward(self, *args, **kwargs):
            # Apply each module and stack results
            outputs = []
            for module in self.modules_list:
                outputs.append(module(*args, **kwargs))
            
            if out_axes == 0:
                return torch.stack(outputs, dim=0)
            else:
                return torch.stack(outputs, dim=out_axes)

    return EnsembleModule


class Policy(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        network_output_dim: int,
        action_dim: int,
        action_horizon: int = 1,
        init_final: Optional[float] = None,
        std_parameterization: str = "exp",  # "exp", "softplus", "fixed", or "uniform"
        std_min: Optional[float] = 1e-5,
        std_max: Optional[float] = 10.0,
        tanh_squash_distribution: bool = False,
        fixed_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.network = network
        self.network_output_dim = network_output_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.init_final = init_final
        self.std_parameterization = std_parameterization
        self.std_min = std_min
        self.std_max = std_max
        self.tanh_squash_distribution = tanh_squash_distribution
        self.fixed_std = fixed_std
        
        # Create output layers
        if self.action_horizon > 1:
            self.means_layer = nn.Linear(network_output_dim, self.action_dim * self.action_horizon)
            if self.fixed_std is None:
                if self.std_parameterization in ["exp", "softplus"]:
                    self.stds_layer = nn.Linear(network_output_dim, self.action_dim * self.action_horizon)
                elif self.std_parameterization == "uniform":
                    self.log_stds = nn.Parameter(torch.zeros((self.action_horizon, self.action_dim)))
        else:
            self.means_layer = nn.Linear(network_output_dim, self.action_dim)
            if self.fixed_std is None:
                if self.std_parameterization in ["exp", "softplus"]:
                    self.stds_layer = nn.Linear(network_output_dim, self.action_dim)
                elif self.std_parameterization == "uniform":
                    self.log_stds = nn.Parameter(torch.zeros((self.action_dim,)))

    def forward(
        self, observations: torch.Tensor, temperature: float = 1.0, non_squash_distribution: bool = False,
    ) -> distributions.Distribution:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)

        outputs = self.network(obs_enc)

        # Reshape outputs to predict action_horizon actions at once
        if self.action_horizon > 1:
            means = self.means_layer(outputs)
            means = means.reshape(-1, self.action_horizon, self.action_dim)
            
            if self.fixed_std is None:
                if self.std_parameterization == "exp":
                    log_stds = self.stds_layer(outputs)
                    log_stds = log_stds.reshape(-1, self.action_horizon, self.action_dim)
                    stds = torch.exp(log_stds)
                elif self.std_parameterization == "softplus":
                    stds = self.stds_layer(outputs)
                    stds = stds.reshape(-1, self.action_horizon, self.action_dim)
                    stds = torch.nn.functional.softplus(stds)
                elif self.std_parameterization == "uniform":
                    stds = torch.exp(self.log_stds)
                    stds = stds.unsqueeze(0).expand(means.shape[0], -1, -1)
                else:
                    raise ValueError(
                        f"Invalid std_parameterization: {self.std_parameterization}"
                    )
            else:
                assert self.std_parameterization == "fixed"
                stds = self.fixed_std
                if stds.dim() == 1:
                    stds = stds.unsqueeze(0).unsqueeze(0).expand(means.shape[0], self.action_horizon, -1)
                elif stds.dim() == 2:
                    stds = stds.unsqueeze(0).expand(means.shape[0], -1, -1)
        else:
            means = self.means_layer(outputs)  # [batch_size, action_dim]
            if self.fixed_std is None:
                if self.std_parameterization == "exp":
                    log_stds = self.stds_layer(outputs)
                    stds = torch.exp(log_stds)
                elif self.std_parameterization == "softplus":
                    stds = self.stds_layer(outputs)
                    stds = torch.nn.functional.softplus(stds)
                elif self.std_parameterization == "uniform":
                    stds = torch.exp(self.log_stds)
                    stds = stds.unsqueeze(0).expand(means.shape[0], -1)
                else:
                    raise ValueError(
                        f"Invalid std_parameterization: {self.std_parameterization}"
                    )
            else:
                assert self.std_parameterization == "fixed"
                stds = self.fixed_std.unsqueeze(0).expand(means.shape[0], -1)

        # Clip stds to avoid numerical instability
        # For a normal distribution under MaxEnt, optimal std scales with sqrt(temperature)
        stds = torch.clamp(stds, self.std_min, self.std_max) * np.sqrt(temperature)

        if self.action_horizon > 1:
            # For action sequences, flatten batch and horizon dimensions for distribution
            batch_size = means.shape[0]
            means_flat = means.reshape(batch_size * self.action_horizon, self.action_dim)
            stds_flat = stds.reshape(batch_size * self.action_horizon, self.action_dim)
            
            if self.tanh_squash_distribution and not non_squash_distribution:
                flat_dist = TanhMultivariateNormal(
                    loc=means_flat,
                    scale=stds_flat,
                )
            else:
                flat_dist = distributions.MultivariateNormal(
                    loc=means_flat,
                    covariance_matrix=torch.diag_embed(stds_flat**2),
                )
            
            # Create a wrapper that reshapes samples back to [batch, horizon, action_dim]
            distribution = ActionSequenceDistribution(flat_dist, batch_size, self.action_horizon, self.action_dim)
        else:
            if self.tanh_squash_distribution and not non_squash_distribution:
                distribution = TanhMultivariateNormal(
                    loc=means,
                    scale=stds,
                )
            else:
                distribution = distributions.MultivariateNormal(
                    loc=means,
                    covariance_matrix=torch.diag_embed(stds**2),
                )

        return distribution
    
    def get_features(self, observations):
        return self.encoder(observations)


class TanhMultivariateNormal(distributions.TransformedDistribution):
    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ):
        base_dist = distributions.MultivariateNormal(
            loc=loc, 
            covariance_matrix=torch.diag_embed(scale**2)
        )

        transforms = []

        if low is not None and high is not None:
            # First apply tanh, then rescale
            transforms.append(distributions.TanhTransform())
            transforms.append(distributions.AffineTransform(
                loc=(high + low) / 2,
                scale=(high - low) / 2
            ))
        else:
            transforms.append(distributions.TanhTransform())

        super().__init__(base_dist, transforms)

    def mode(self) -> torch.Tensor:
        mode = self.base_dist.loc
        for transform in self.transforms:
            mode = transform(mode)
        return mode


class ActionSequenceDistribution:
    """Wrapper for distributions that handles reshaping for action sequences."""
    
    def __init__(self, base_dist, batch_size, action_horizon, action_dim):
        self.base_dist = base_dist
        self.batch_size = batch_size
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        
    def sample(self, sample_shape=torch.Size()):
        samples = self.base_dist.sample(sample_shape)  # [batch*horizon, action_dim]
        if sample_shape == torch.Size():
            return samples.reshape(self.batch_size, self.action_horizon, self.action_dim)
        else:
            return samples.reshape(*sample_shape, self.batch_size, self.action_horizon, self.action_dim)
    
    def mode(self):
        if hasattr(self.base_dist, 'mode'):
            mode = self.base_dist.mode()  # [batch*horizon, action_dim]
        else:
            mode = self.base_dist.mean  # [batch*horizon, action_dim]
        return mode.reshape(self.batch_size, self.action_horizon, self.action_dim)
    
    def log_prob(self, value):
        # value shape: [batch, horizon, action_dim]
        batch_size = value.shape[0]
        value_flat = value.reshape(batch_size * self.action_horizon, self.action_dim)
        log_probs = self.base_dist.log_prob(value_flat)  # [batch*horizon]
        log_probs = log_probs.reshape(batch_size, self.action_horizon)
        # Sum over horizon dimension to get log_prob of the entire sequence
        return log_probs.sum(dim=1)  # [batch]

    @property
    def mean(self):
        mean = self.base_dist.mean  # [batch*horizon, action_dim]
        return mean.reshape(self.batch_size, self.action_horizon, self.action_dim)
