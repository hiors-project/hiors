import functools as ft
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import Resize

from serl_launcher.vision.data_augmentations import resize

ModuleDef = Any
Array = Any
Shape = Tuple[int]
Dtype = Any


class AddSpatialCoordinates(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        batch_size = x.shape[0] if x.dim() == 4 else 1
        height, width = x.shape[-2:]
        
        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, height, device=x.device, dtype=self.dtype)
        x_coords = torch.linspace(-1, 1, width, device=x.device, dtype=self.dtype)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack coordinates and add batch dimension
        coords = torch.stack([x_grid, y_grid], dim=0)  # [2, H, W]
        
        if x.dim() == 4:
            coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, 2, H, W]
        
        return torch.cat([x, coords], dim=-3 if x.dim() == 4 else 0)


class SpatialSoftmax(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        channel: int,
        temperature: Optional[float] = None,
        log_heatmap: bool = False,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.log_heatmap = log_heatmap
        
        # Create position grids
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing='ij'
        )
        self.register_buffer('pos_x', pos_x.reshape(-1))
        self.register_buffer('pos_y', pos_y.reshape(-1))
        
        if temperature == -1:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.temperature = temperature or 1.0

    def forward(self, features):
        # Handle temperature
        if isinstance(self.temperature, nn.Parameter):
            temp = self.temperature
        else:
            temp = self.temperature

        # Add batch dim if missing
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features.unsqueeze(0)

        batch_size = features.shape[0]
        num_featuremaps = features.shape[1]
        
        # Reshape features for softmax
        features_flat = features.view(batch_size, num_featuremaps, -1)
        
        # Apply softmax
        softmax_attention = F.softmax(features_flat / temp, dim=2)
        
        # Compute expected positions
        expected_x = torch.sum(
            self.pos_x * softmax_attention, dim=2
        )
        expected_y = torch.sum(
            self.pos_y * softmax_attention, dim=2
        )
        
        expected_xy = torch.cat([expected_x, expected_y], dim=1)
        
        if no_batch_dim:
            expected_xy = expected_xy.squeeze(0)
            
        return expected_xy


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        channel: int,
        num_features: int = 5,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features
        
        # Learnable spatial embeddings
        self.embeddings = nn.Parameter(
            torch.randn(height, width, channel, num_features) * 0.02
        )

    def forward(self, features):
        # Add batch dim if missing
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features.unsqueeze(0)

        batch_size = features.shape[0]
        
        # features: [B, C, H, W] -> [B, H, W, C]
        features = features.permute(0, 2, 3, 1)
        
        # Compute spatial embeddings
        # features: [B, H, W, C], embeddings: [H, W, C, num_features]
        features_expanded = features.unsqueeze(-1)  # [B, H, W, C, 1]
        embeddings_expanded = self.embeddings.unsqueeze(0)  # [1, H, W, C, num_features]
        
        # Element-wise multiplication and sum over spatial dimensions
        result = (features_expanded * embeddings_expanded).sum(dim=(1, 2))  # [B, C, num_features]
        result = result.view(batch_size, -1)  # [B, C * num_features]
        
        if no_batch_dim:
            result = result.squeeze(0)
            
        return result


class ResNetBlock(nn.Module):
    """Basic ResNet block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.GroupNorm(4, out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.GroupNorm(4, out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.GroupNorm(4, out_channels * 4)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out


class ResNetEncoder(nn.Module):
    """ResNet encoder with various pooling options."""
    
    def __init__(
        self,
        stage_sizes: Sequence[int],
        block_cls: type,
        num_filters: int = 64,
        add_spatial_coordinates: bool = False,
        pooling_method: str = "avg",
        softmax_temperature: float = 1.0,
        num_spatial_blocks: int = 8,
        bottleneck_dim: Optional[int] = None,
        pre_pooling: bool = False,
        image_size: tuple = (128, 128),
        **kwargs
    ):
        super().__init__()
        
        self.stage_sizes = stage_sizes
        self.num_filters = num_filters
        self.add_spatial_coordinates = add_spatial_coordinates
        self.pooling_method = pooling_method
        self.softmax_temperature = softmax_temperature
        self.num_spatial_blocks = num_spatial_blocks
        self.bottleneck_dim = bottleneck_dim
        self.pre_pooling = pre_pooling
        self.image_size = image_size
        
        # Add spatial coordinates if needed
        if add_spatial_coordinates:
            self.spatial_coords = AddSpatialCoordinates()
            input_channels = 5  # RGB + 2 spatial coordinates
        else:
            input_channels = 3
            
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, num_filters, 7, 2, 3, bias=False)
        self.bn1 = nn.GroupNorm(4, num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Build ResNet stages
        self.stages = nn.ModuleList()
        in_channels = num_filters
        
        for i, num_blocks in enumerate(stage_sizes):
            stage_channels = num_filters * (2 ** i)
            stride = 2 if i > 0 else 1
            
            stage = self._make_stage(
                block_cls, in_channels, stage_channels, num_blocks, stride
            )
            self.stages.append(stage)
            
            if block_cls == BottleneckResNetBlock:
                in_channels = stage_channels * 4
            else:
                in_channels = stage_channels
                
        self.final_channels = in_channels
        
        # Post-processing layers
        if not pre_pooling:
            self._setup_pooling_layers()
            
    def _make_stage(self, block_cls, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * (4 if block_cls == BottleneckResNetBlock else 1):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * (4 if block_cls == BottleneckResNetBlock else 1), 1, stride, bias=False),
                nn.GroupNorm(4, out_channels * (4 if block_cls == BottleneckResNetBlock else 1))
            )
            
        layers = []
        layers.append(block_cls(in_channels, out_channels, stride, downsample))
        
        in_channels = out_channels * (4 if block_cls == BottleneckResNetBlock else 1)
        for _ in range(1, num_blocks):
            layers.append(block_cls(in_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def _setup_pooling_layers(self):
        if self.pooling_method == "spatial_learned_embeddings":
            # Calculate expected feature map size
            h, w = self.image_size
            h, w = h // 4, w // 4  # Initial conv + maxpool
            for i in range(len(self.stage_sizes)):
                if i > 0:
                    h, w = h // 2, w // 2
                    
            self.spatial_embeddings = SpatialLearnedEmbeddings(
                h, w, self.final_channels, self.num_spatial_blocks
            )
            
        if self.bottleneck_dim is not None:
            if self.pooling_method == "spatial_learned_embeddings":
                in_features = self.final_channels * self.num_spatial_blocks
            else:
                in_features = self.final_channels
                
            self.bottleneck = nn.Sequential(
                nn.Linear(in_features, self.bottleneck_dim),
                nn.LayerNorm(self.bottleneck_dim),
                nn.Tanh()
            )
            
    def forward(self, x, **kwargs):
        # Normalize input
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
            
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        # x = (x / 255.0 - mean) / std
        x = (x - mean) / std
        
        if self.add_spatial_coordinates:
            x = self.spatial_coords(x)
            
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet stages
        for stage in self.stages:
            x = stage(x)
            
        if self.pre_pooling:
            return x
            
        # Pooling
        if self.pooling_method == "spatial_learned_embeddings":
            x = self.spatial_embeddings(x)
        elif self.pooling_method == "spatial_softmax":
            height, width = x.shape[-2:]
            spatial_softmax = SpatialSoftmax(
                height, width, x.shape[1], self.softmax_temperature
            )
            x = spatial_softmax(x)
        elif self.pooling_method == "avg":
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif self.pooling_method == "max":
            x = F.adaptive_max_pool2d(x, 1).flatten(1)
        elif self.pooling_method == "none":
            pass
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
            
        # Bottleneck
        if self.bottleneck_dim is not None:
            x = self.bottleneck(x)
            
        return x


class PreTrainedResNetEncoder(nn.Module):
    """Wrapper for pretrained ResNet encoders."""
    
    def __init__(
        self,
        pooling_method: str = "avg",
        softmax_temperature: float = 1.0,
        num_spatial_blocks: int = 8,
        bottleneck_dim: Optional[int] = None,
        pretrained_model: str = "resnet18",
        **kwargs
    ):
        super().__init__()
        
        self.pooling_method = pooling_method
        self.softmax_temperature = softmax_temperature
        self.num_spatial_blocks = num_spatial_blocks
        self.bottleneck_dim = bottleneck_dim
        
        # Load pretrained ResNet
        if pretrained_model == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
        elif pretrained_model == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
        elif pretrained_model == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported pretrained model: {pretrained_model}")
            
        # Remove final layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Get number of features from the backbone
        self.feature_channels = 512
        self.feature_size = (4, 4)

        # Setup pooling layers
        if self.pooling_method == "spatial_learned_embeddings":
            h, w = self.feature_size
            self.spatial_embeddings = SpatialLearnedEmbeddings(
                h, w, self.feature_channels, self.num_spatial_blocks
            )
            
        if self.bottleneck_dim is not None:
            if self.pooling_method == "spatial_learned_embeddings":
                in_features = self.feature_channels * self.num_spatial_blocks
            else:
                in_features = self.feature_channels
                
            self.bottleneck = nn.Sequential(
                nn.Linear(in_features, self.bottleneck_dim),
                nn.LayerNorm(self.bottleneck_dim),
                nn.Tanh()
            )
        
    def forward(self, x, encode: bool = True, **kwargs):
        """
        Resnet encoder for image inputs.

        Args:
            x: [B, C, H, W]
        """
        # maybe (B, H, W, C), should be (B, C, H, W)
        # print(f"{x.shape=}")
        # if x.shape[-1] == 3:
        #     x = x.permute(0, 3, 1, 2).contiguous()
            
        if encode:
            x = self.backbone(x)
            
        # Pooling
        if self.pooling_method == "spatial_learned_embeddings":
            x = self.spatial_embeddings(x)
        elif self.pooling_method == "spatial_softmax":
            height, width = x.shape[-2:]
            spatial_softmax = SpatialSoftmax(
                height, width, x.shape[1], self.softmax_temperature
            )
            x = spatial_softmax(x)
        elif self.pooling_method == "avg":
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif self.pooling_method == "max":
            x = F.adaptive_max_pool2d(x, 1).flatten(1)
        elif self.pooling_method == "none":
            pass
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
            
        # Bottleneck
        if self.bottleneck_dim is not None:
            x = self.bottleneck(x)
            
        return x


# Configuration dictionary
resnetv1_configs = {
    "resnetv1-10": ft.partial(
        ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock
    ),
    "resnetv1-10-frozen": ft.partial(
        ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock, pre_pooling=True
    ),
    "resnetv1-18": ft.partial(
        ResNetEncoder, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock
    ),
    "resnetv1-18-frozen": ft.partial(
        ResNetEncoder, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock, pre_pooling=True
    ),
    "resnetv1-34": ft.partial(
        ResNetEncoder, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock
    ),
    "resnetv1-50": ft.partial(
        ResNetEncoder, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock
    ),
    "resnetv1-50-frozen": ft.partial(
        ResNetEncoder, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock, pre_pooling=True
    ),
    "resnetv1-18-deeper": ft.partial(
        ResNetEncoder, stage_sizes=(3, 3, 3, 3), block_cls=ResNetBlock
    ),
    "resnetv1-18-deepest": ft.partial(
        ResNetEncoder, stage_sizes=(4, 4, 4, 4), block_cls=ResNetBlock
    ),
    "resnetv1-18-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(2, 2, 2, 2),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    "resnetv1-34-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    "resnetv1-50-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=BottleneckResNetBlock,
        num_spatial_blocks=8,
    ),
    # Pretrained versions
    "pretrained-resnet18": ft.partial(
        PreTrainedResNetEncoder, pretrained_model="resnet18"
    ),
    "pretrained-resnet34": ft.partial(
        PreTrainedResNetEncoder, pretrained_model="resnet34"
    ),
    "pretrained-resnet50": ft.partial(
        PreTrainedResNetEncoder, pretrained_model="resnet50"
    ),
}
