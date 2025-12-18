"""
RUL预测 1D-CNN 回归器 - 模型构建器
基于一维卷积的回归网络，用于处理时序滑动窗口特征。
"""

from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class CNN1DRegressor(nn.Cell):
    """一维卷积RUL回归器"""

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

        # 构建卷积模块
        self.conv_blocks = nn.CellList()
        in_channels = n_features
        for idx, out_channels in enumerate(self.conv_channels):
            kernel_size = self.kernel_sizes[idx] if idx < len(self.kernel_sizes) else self.kernel_sizes[-1]
            padding = max(kernel_size // 2, 0)

            block_layers: List[nn.Cell] = [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pad_mode="pad",
                    padding=padding,
                    has_bias=True,
                ),
            ]

            if self.use_batch_norm:
                block_layers.append(nn.BatchNorm1d(out_channels))

            block_layers.append(self._get_activation(self.activation_name))
            block_layers.append(nn.Dropout(p=min(dropout + idx * 0.05, 0.6)))

            self.conv_blocks.append(nn.SequentialCell(block_layers))
            in_channels = out_channels

        # 全局池化
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.reduce_max = ops.ReduceMax(keep_dims=False)

        # 回归头
        if self.pooling in ("avg", "max"):
            reg_input_dim = in_channels
        else:
            reg_input_dim = in_channels * self.sequence_length

        hidden_dim = max(fc_units, 32)
        second_hidden = max(hidden_dim // 2, 32)

        self.regressor = nn.SequentialCell([
            nn.Dense(reg_input_dim, hidden_dim),
            self._get_activation(self.activation_name),
            nn.Dropout(p=dropout),
            nn.Dense(hidden_dim, second_hidden),
            self._get_activation(self.activation_name),
            nn.Dropout(p=dropout * 0.5),
            nn.Dense(second_hidden, 1),
        ])

    def _get_activation(self, name: str) -> nn.Cell:
        mapping = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(alpha=0.2),
            "gelu": nn.GELU(),
        }
        return mapping.get((name or "relu").lower(), nn.ReLU())

    def _global_pool(self, x: ms.Tensor) -> ms.Tensor:
        if self.pooling == "avg":
            return self.reduce_mean(x, 2)
        if self.pooling == "max":
            return self.reduce_max(x, 2)
        # 展平所有时间维度
        return ops.reshape(x, (x.shape[0], -1))

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # 输入形状: (batch, seq_len, features)
        x = ops.transpose(x, (0, 2, 1))  # 转为 (batch, channels, seq_len)
        for block in self.conv_blocks:
            x = block(x)
        features = self._global_pool(x)
        return self.regressor(features)


class ModelBuilder:
    """1D-CNN 回归器模型工厂"""

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
    def create_model(model_type: str, input_shape: Tuple[int, int], **kwargs) -> nn.Cell:
        if model_type != "cnn_1d_regressor":
            raise ValueError(f"Unsupported model_type for CNN1D builder: {model_type}")

        config = ModelBuilder.get_default_config(model_type)
        config.update(kwargs)
        return CNN1DRegressor(input_shape=input_shape, **config)

    @staticmethod
    def get_model_info(model: nn.Cell) -> Dict[str, Any]:
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
                "trainable_params": sum(p.size for p in model.trainable_params()),
            }
        return {
            "model_type": "unknown",
            "trainable_params": sum(p.size for p in model.trainable_params()),
        }


__all__ = ["CNN1DRegressor", "ModelBuilder"]
