"""
RUL预测 BiLSTM/GRU 回归器 - 模型构建器（PyTorch版，边端推理）。
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    mapping = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "gelu": nn.GELU(),
        "leakyrelu": nn.LeakyReLU(0.2),
    }
    return mapping.get((name or "relu").lower(), nn.ReLU())


class AttentionPool(nn.Module):
    """简单的加性注意力池化。"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)  # (batch, seq, 1)
        return (x * weights).sum(dim=1)  # (batch, dim)


class BiLSTMGRURegressor(nn.Module):
    """
    BiLSTM/GRU RUL回归器（与云端保持一致，便于权重复用）。
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        hidden_units: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = "relu",
        bidirectional: bool = True,
        use_attention: bool = True,
        use_layer_norm: bool = True,
        rnn_type: str = "lstm",  # "lstm" 或 "gru"
        output_activation: str = None,
    ):
        super().__init__()

        seq_len, n_features = input_shape
        self.sequence_length = seq_len
        self.input_dim = n_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_units,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=n_features,
                hidden_size=hidden_units,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}，支持 'lstm' 或 'gru'")

        rnn_output_dim = hidden_units * 2 if bidirectional else hidden_units
        self.attention = AttentionPool(rnn_output_dim) if use_attention else None

        reg_layers = [
            nn.Linear(rnn_output_dim, rnn_output_dim // 2),
        ]
        if use_layer_norm:
            reg_layers.append(nn.LayerNorm(rnn_output_dim // 2))
        reg_layers.append(_get_activation(activation))
        reg_layers.append(nn.Dropout(p=dropout))

        reg_layers.append(nn.Linear(rnn_output_dim // 2, rnn_output_dim // 4))
        if use_layer_norm:
            reg_layers.append(nn.LayerNorm(rnn_output_dim // 4))
        reg_layers.append(_get_activation(activation))
        reg_layers.append(nn.Dropout(p=dropout * 0.5))

        reg_layers.append(nn.Linear(rnn_output_dim // 4, 1))
        self.output_activation = (
            nn.Sigmoid() if output_activation and output_activation.lower() == "sigmoid" else None
        )
        self.regressor = nn.Sequential(*reg_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_output, _ = self.rnn(x)
        if self.attention is not None:
            features = self.attention(rnn_output)
        else:
            features = rnn_output[:, -1, :]
        out = self.regressor(features)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class ModelBuilder:
    """BiLSTM/GRU 回归器模型工厂"""

    DEFAULT_CONFIG = {
        "hidden_units": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "activation": "relu",
        "bidirectional": True,
        "use_attention": True,
        "use_layer_norm": True,
        "rnn_type": "lstm",  # "lstm" 或 "gru"
    }

    @staticmethod
    def get_default_config(model_type: str = "bilstm_gru_regressor") -> Dict[str, Any]:
        if model_type == "bilstm_gru_regressor":
            return ModelBuilder.DEFAULT_CONFIG.copy()
        return {}

    @staticmethod
    def build_regressor(
        input_shape: Tuple[int, int],
        hidden_units: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = "relu",
        bidirectional: bool = True,
        use_attention: bool = True,
        use_layer_norm: bool = True,
        rnn_type: str = "lstm",
    ) -> BiLSTMGRURegressor:
        return BiLSTMGRURegressor(
            input_shape=input_shape,
            hidden_units=hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            bidirectional=bidirectional,
            use_attention=use_attention,
            use_layer_norm=use_layer_norm,
            rnn_type=rnn_type,
        )

    @staticmethod
    def create_model(
        model_type: str,
        input_shape: Tuple[int, int],
        **kwargs
    ) -> nn.Module:
        if model_type != "bilstm_gru_regressor":
            raise ValueError(
                f"Unsupported model_type for BiLSTM/GRU regressor builder: {model_type}"
            )

        config = ModelBuilder.get_default_config(model_type)
        if 'hidden_size' in kwargs:
            kwargs['hidden_units'] = kwargs.pop('hidden_size')
        config.update(kwargs)
        return ModelBuilder.build_regressor(
            input_shape=input_shape,
            **config
        )

    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        if isinstance(model, BiLSTMGRURegressor):
            return {
                "model_type": "bilstm_gru_regressor",
                "input_dim": model.input_dim,
                "sequence_length": model.sequence_length,
                "hidden_units": model.hidden_units,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
                "bidirectional": model.bidirectional,
                "use_attention": model.use_attention,
                "rnn_type": model.rnn_type,
                "trainable_params": sum(p.numel() for p in model.parameters()),
            }
        return {
            "model_type": "unknown",
            "trainable_params": sum(p.numel() for p in model.parameters()),
        }