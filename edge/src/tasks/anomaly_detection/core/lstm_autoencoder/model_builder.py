"""
LSTM自编码器 - 模型构建器
定义用于工业时序重构的LSTM Autoencoder结构


"""

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder，用于学习正常序列并重构输入窗口"""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        hidden_units: int = 128,
        bottleneck_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "tanh",
    ):
        super().__init__()
        seq_len, n_features = input_shape
        self.sequence_length = seq_len
        self.input_dim = n_features
        self.hidden_units = hidden_units
        self.bottleneck_size = bottleneck_size
        self.num_layers = num_layers
        self.dropout = dropout

        # 编码器：提取序列潜在表示
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 潜在空间映射
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_units, bottleneck_size),
            self._get_activation(activation),
        )

        # 解码器：根据潜在向量重建序列
        self.decoder = nn.LSTM(
            input_size=bottleneck_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_units, hidden_units // 2),
            self._get_activation(activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units // 2, n_features),
        )

    def _get_activation(self, name: str) -> nn.Module:
        """根据名称返回激活函数"""
        mapping = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(),
            "gelu": nn.GELU(),
        }
        return mapping.get(name.lower(), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回重构序列
        
        Args:
            x: 输入序列 (batch_size, seq_len, n_features)
        
        Returns:
            重构序列 (batch_size, seq_len, n_features)
        """
        batch_size = x.shape[0]

        # 编码
        _, (hidden, cell) = self.encoder(x)
        latent_vector = hidden[-1]  # (batch, hidden_units)
        latent_vector = self.latent_projection(latent_vector)  # (batch, bottleneck_size)

        # 重复潜在向量到序列长度
        repeated_latent = latent_vector.unsqueeze(1).expand(-1, self.sequence_length, -1)
        # (batch, seq_len, bottleneck_size)

        # 解码
        decoder_output, _ = self.decoder(repeated_latent)
        # (batch, seq_len, hidden_units)

        # 重构
        reconstruction = self.reconstruction_head(decoder_output)
        # (batch, seq_len, n_features)
        
        return reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码，返回潜在向量
        
        Args:
            x: 输入序列 (batch_size, seq_len, n_features)
        
        Returns:
            潜在向量 (batch_size, bottleneck_size)
        """
        _, (hidden, _) = self.encoder(x)
        latent_vector = hidden[-1]
        latent_vector = self.latent_projection(latent_vector)
        return latent_vector


class ModelBuilder:
    """LSTM Autoencoder 模型工厂"""

    DEFAULT_CONFIG = {
        "hidden_units": 128,
        "bottleneck_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "activation": "tanh",
    }

    @staticmethod
    def get_default_config(model_type: str = "lstm_autoencoder") -> Dict[str, Any]:
        if model_type == "lstm_autoencoder":
            return ModelBuilder.DEFAULT_CONFIG.copy()
        return {}

    @staticmethod
    def build_lstm_autoencoder(
        input_shape: Tuple[int, int],
        hidden_units: int = 128,
        bottleneck_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "tanh",
    ) -> LSTMAutoencoder:
        return LSTMAutoencoder(
            input_shape=input_shape,
            hidden_units=hidden_units,
            bottleneck_size=bottleneck_size,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )

    @staticmethod
    def create_model(model_type: str, input_shape: Tuple[int, int], **kwargs) -> nn.Module:
        if model_type != "lstm_autoencoder":
            raise ValueError(f"Unsupported model_type for autoencoder builder: {model_type}")

        config = ModelBuilder.get_default_config(model_type)
        
        # 处理参数名称兼容性
        if 'hidden_size' in kwargs:
            kwargs['hidden_units'] = kwargs.pop('hidden_size')
        if 'bottleneck_dim' in kwargs:
            kwargs['bottleneck_size'] = kwargs.pop('bottleneck_dim')
            
        config.update(kwargs)
        return ModelBuilder.build_lstm_autoencoder(input_shape, **config)

    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        if isinstance(model, LSTMAutoencoder):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {
                "model_type": "lstm_autoencoder",
                "input_dim": model.input_dim,
                "sequence_length": model.sequence_length,
                "hidden_units": model.hidden_units,
                "bottleneck_size": model.bottleneck_size,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
                "trainable_params": total_params,
            }
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "model_type": "unknown",
            "trainable_params": total_params,
        }


def create_model(model_type: str, **kwargs) -> nn.Module:
    """向后兼容接口"""
    if model_type != "lstm_autoencoder":
        raise ValueError("create_model from autoencoder module only supports lstm_autoencoder")

    if "input_dim" not in kwargs:
        raise ValueError("Autoencoder create_model requires 'input_dim'")

    input_dim = kwargs.pop("input_dim")
    seq_len = kwargs.pop("seq_len", 50)
    input_shape = (seq_len, input_dim)
    return ModelBuilder.create_model(model_type, input_shape, **kwargs)


def get_default_config(model_type: str = "lstm_autoencoder") -> Dict[str, Any]:
    return ModelBuilder.get_default_config(model_type)


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    model_type = config.get("model_type", "lstm_autoencoder")
    input_dim = config.get("input_dim")
    if input_dim is None:
        raise ValueError("config must include 'input_dim'")

    seq_len = config.get("seq_len", config.get("sequence_length", 50))
    model_kwargs = {k: v for k, v in config.items() if k in {
        "hidden_units",
        "bottleneck_size",
        "num_layers",
        "dropout",
        "activation",
    }}

    return create_model(model_type, input_dim=input_dim, seq_len=seq_len, **model_kwargs)
