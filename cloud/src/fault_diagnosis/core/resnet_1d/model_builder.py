"""
故障诊断 ResNet 1D - 模型构建器
定义用于故障分类的1D残差网络模型结构
支持二分类（正常-故障）和多分类（正常-内圈故障-外圈故障等）

重构说明：从 MindSpore 迁移到 PyTorch
"""

from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """
    1D 残差块
    
    结构：
    x -> Conv1D -> BN -> ReLU -> Conv1D -> BN -> (+) -> ReLU
    |_____________________ shortcut ___________________|
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: bool = False,
    ):
        super(ResidualBlock1D, self).__init__()
        
        actual_stride = 2 if downsample else stride
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=actual_stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.need_shortcut = (in_channels != out_channels) or downsample
        if self.need_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=actual_stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.need_shortcut and self.shortcut is not None:
            identity = self.shortcut(x)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class ResNet1DClassifier(nn.Module):
    """
    1D ResNet 故障分类器
    
    架构设计：
    - 输入：(batch_size, seq_len, n_features)
    - 初始卷积层：调整通道数
    - 残差块组：多层残差块堆叠
    - 全局平均池化：压缩时序维度
    - 全连接层：分类输出
    - 输出：(batch_size, num_classes)
    """
    
    BLOCK_CONFIGS = {
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3],
        'resnet_small': [1, 1, 1, 1],
        'resnet_tiny': [1, 1],
    }
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        base_channels: int = 64,
        block_config: str = 'resnet_small',
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super(ResNet1DClassifier, self).__init__()
        
        seq_len, n_features = input_shape
        self.sequence_length = seq_len
        self.input_dim = n_features
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.block_config = block_config
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        
        if isinstance(block_config, str):
            self.num_blocks = self.BLOCK_CONFIGS.get(block_config, [1, 1, 1, 1])
        else:
            self.num_blocks = block_config
        
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=base_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        self.layer1 = self._make_layer(
            base_channels, base_channels, self.num_blocks[0], downsample=False
        )
        
        layers = [self.layer1]
        in_channels = base_channels
        
        for i, num_block in enumerate(self.num_blocks[1:], 1):
            out_channels = base_channels * (2 ** i)
            layer = self._make_layer(in_channels, out_channels, num_block, downsample=True)
            layers.append(layer)
            in_channels = out_channels
        
        self.res_layers = nn.ModuleList(layers)
        self.final_channels = in_channels
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.final_channels, self.final_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(self.final_channels // 2, num_classes),
        )
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        downsample: bool = False,
    ) -> nn.Sequential:
        blocks = []
        
        blocks.append(
            ResidualBlock1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                downsample=downsample,
            )
        )
        
        for _ in range(1, num_blocks):
            blocks.append(
                ResidualBlock1D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    downsample=False,
                )
            )
        
        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        # 转换输入格式: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        x = self.stem(x)
        
        for layer in self.res_layers:
            x = layer(x)
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        if return_probs:
            return F.softmax(logits, dim=-1)
        
        return logits


class ModelBuilder:
    """ResNet 1D 分类器模型工厂"""
    
    DEFAULT_CONFIG = {
        "base_channels": 64,
        "block_config": "resnet_small",
        "kernel_size": 3,
        "dropout": 0.3,
    }
    
    @staticmethod
    def get_default_config(model_type: str = "resnet_1d_classifier") -> Dict[str, Any]:
        if model_type == "resnet_1d_classifier":
            return ModelBuilder.DEFAULT_CONFIG.copy()
        return {}
    
    @staticmethod
    def build_resnet_1d_classifier(
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        base_channels: int = 64,
        block_config: str = "resnet_small",
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> ResNet1DClassifier:
        return ResNet1DClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            base_channels=base_channels,
            block_config=block_config,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    
    @staticmethod
    def create_model(
        model_type: str,
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        **kwargs
    ) -> nn.Module:
        if model_type != "resnet_1d_classifier":
            raise ValueError(
                f"Unsupported model_type: {model_type}"
            )
        
        config = ModelBuilder.get_default_config(model_type)
        
        if 'num_filters' in kwargs:
            kwargs['base_channels'] = kwargs.pop('num_filters')
        if 'num_layers' in kwargs:
            num_layers = kwargs.pop('num_layers')
            if num_layers <= 2:
                kwargs['block_config'] = 'resnet_tiny'
            elif num_layers <= 4:
                kwargs['block_config'] = 'resnet_small'
            else:
                kwargs['block_config'] = 'resnet18'
        
        config.update(kwargs)
        return ModelBuilder.build_resnet_1d_classifier(
            input_shape=input_shape,
            num_classes=num_classes,
            **config
        )
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        if isinstance(model, ResNet1DClassifier):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {
                "model_type": "resnet_1d_classifier",
                "input_dim": model.input_dim,
                "sequence_length": model.sequence_length,
                "num_classes": model.num_classes,
                "base_channels": model.base_channels,
                "block_config": model.block_config,
                "num_blocks": model.num_blocks,
                "final_channels": model.final_channels,
                "kernel_size": model.kernel_size,
                "dropout": model.dropout_rate,
                "trainable_params": total_params,
            }
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "model_type": "unknown",
            "trainable_params": total_params,
        }


def create_model(model_type: str, **kwargs) -> nn.Module:
    """向后兼容接口"""
    if model_type != "resnet_1d_classifier":
        raise ValueError(
            "create_model only supports resnet_1d_classifier"
        )
    
    if "input_dim" not in kwargs:
        raise ValueError("create_model requires 'input_dim'")
    
    input_dim = kwargs.pop("input_dim")
    seq_len = kwargs.pop("seq_len", 50)
    num_classes = kwargs.pop("num_classes", 3)
    input_shape = (seq_len, input_dim)
    
    return ModelBuilder.create_model(
        model_type, input_shape, num_classes=num_classes, **kwargs
    )


def get_default_config(model_type: str = "resnet_1d_classifier") -> Dict[str, Any]:
    return ModelBuilder.get_default_config(model_type)


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    model_type = config.get("model_type", "resnet_1d_classifier")
    input_dim = config.get("input_dim")
    if input_dim is None:
        raise ValueError("config must include 'input_dim'")
    
    seq_len = config.get("seq_len", config.get("sequence_length", 50))
    num_classes = config.get("num_classes", 3)
    input_shape = (seq_len, input_dim)
    
    model_kwargs = {k: v for k, v in config.items() if k in {
        "base_channels",
        "block_config",
        "kernel_size",
        "dropout",
    }}
    
    return ModelBuilder.create_model(
        model_type, input_shape, num_classes=num_classes, **model_kwargs
    )
