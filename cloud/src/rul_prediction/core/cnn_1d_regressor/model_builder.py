"""RUL预测 1D-CNN 回归器（PyTorch版本）。"""

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    mapping = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leakyrelu": nn.LeakyReLU(0.2),
        "gelu": nn.GELU(),
    }
    return mapping.get((name or "relu").lower(), nn.ReLU())


class CNN1DRegressor(nn.Module):
    """一维卷积RUL回归器（批量优先）。"""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        conv_channels: Sequence[int] = (64, 128, 256),
        kernel_sizes: Sequence[int] = (7, 5, 3),
        activation: str = "relu",
        dropout: float = 0.3,
        pooling: str = "avg",
        use_batch_norm: bool = True,
        fc_units: int = 256,
    ):
        super().__init__()

        seq_len, n_features = input_shape
        self.sequence_length = seq_len
        self.input_dim = n_features
        self.conv_channels = list(conv_channels)
        self.kernel_sizes = list(kernel_sizes)
        self.activation_name = activation
        self.dropout = dropout
        self.pooling = (pooling or "avg").lower()
        self.use_batch_norm = use_batch_norm
        self.fc_units = fc_units

        blocks: List[nn.Module] = []
        in_channels = n_features
        for idx, out_channels in enumerate(self.conv_channels):
            kernel_size = self.kernel_sizes[idx] if idx < len(self.kernel_sizes) else self.kernel_sizes[-1]
            padding = max(kernel_size // 2, 0)

            layers: List[nn.Module] = [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
            ]
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(_get_activation(self.activation_name))
            layers.append(nn.Dropout(p=min(dropout + idx * 0.05, 0.6)))

            blocks.append(nn.Sequential(*layers))
            in_channels = out_channels

        self.conv_blocks = nn.ModuleList(blocks)

        if self.pooling in ("avg", "max"):
            reg_input_dim = in_channels
        else:
            reg_input_dim = in_channels * self.sequence_length

        hidden_dim = max(fc_units, 32)
        second_hidden = max(hidden_dim // 2, 32)

        self.regressor = nn.Sequential(
            nn.Linear(reg_input_dim, hidden_dim),
            _get_activation(self.activation_name),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, second_hidden),
            _get_activation(self.activation_name),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(second_hidden, 1),
        )

    def _global_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "avg":
            return torch.mean(x, dim=2)
        if self.pooling == "max":
            return torch.max(x, dim=2).values
        return x.reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        for block in self.conv_blocks:
            x = block(x)
        features = self._global_pool(x)
        return self.regressor(features)


class ModelBuilder:
    """1D-CNN 回归器模型工厂（PyTorch版）。"""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "conv_channels": [64, 128, 256],
        "kernel_sizes": [7, 5, 3],
        "activation": "relu",
        "dropout": 0.3,
        "pooling": "avg",
        "use_batch_norm": True,
        "fc_units": 256,
    }

    @staticmethod
    def get_default_config(_: str = "cnn_1d_regressor") -> Dict[str, Any]:
        return ModelBuilder.DEFAULT_CONFIG.copy()

    @staticmethod
    def create_model(model_type: str, input_shape: Tuple[int, int], **kwargs) -> nn.Module:
        if model_type != "cnn_1d_regressor":
            raise ValueError(f"Unsupported model_type for CNN1D builder: {model_type}")

        config = ModelBuilder.get_default_config(model_type)
        config.update(kwargs)
        return CNN1DRegressor(input_shape=input_shape, **config)

    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        if isinstance(model, CNN1DRegressor):
            return {
                "model_type": "cnn_1d_regressor",
                "input_dim": model.input_dim,
                "sequence_length": model.sequence_length,
                "conv_channels": model.conv_channels,
                "kernel_sizes": model.kernel_sizes,
                "activation": model.activation_name,
                "dropout": model.dropout,
                "pooling": model.pooling,
                "use_batch_norm": model.use_batch_norm,
                "fc_units": model.fc_units,
                "trainable_params": sum(p.numel() for p in model.parameters()),
            }
        return {
            "model_type": "unknown",
            "trainable_params": sum(p.numel() for p in model.parameters()),
        }


__all__ = ["CNN1DRegressor", "ModelBuilder"]
