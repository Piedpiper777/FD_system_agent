"""Transformer Encoder RUL回归器（PyTorch版，边端推理复用）。"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    mapping = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "leakyrelu": nn.LeakyReLU(0.2),
    }
    return mapping.get((name or "gelu").lower(), nn.GELU())


class TransformerEncoderLayer(nn.Module):
    """单层 Transformer Encoder。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.activation = _get_activation(activation)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            self.activation,
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class TransformerRegressor(nn.Module):
    """用于RUL预测的 Transformer Encoder 回归器。"""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        pooling: str = "avg",
        use_positional_encoding: bool = True,
        output_activation: str = None,
    ):
        super().__init__()
        seq_len, input_dim = input_shape
        self.sequence_length = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.pooling = (pooling or "avg").lower()
        self.use_positional_encoding = use_positional_encoding
        self.output_activation = (output_activation or "").lower()

        self.input_projection = nn.Linear(input_dim, embed_dim)

        if self.use_positional_encoding:
            self.positional_embedding = nn.Parameter(
                torch.empty(seq_len, embed_dim)
            )
            nn.init.xavier_uniform_(self.positional_embedding)
        else:
            self.positional_embedding = None

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(max(1, num_layers))
        ])

        head_hidden = max(embed_dim, 64)
        head_mid = max(head_hidden // 2, 32)
        layers = [
            nn.Linear(embed_dim, head_hidden),
            _get_activation(activation),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, head_mid),
            _get_activation(activation),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(head_mid, 1),
        ]
        if self.output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        self.regressor = nn.Sequential(*layers)

    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional_embedding is None:
            return x
        pos_embed = self.positional_embedding.unsqueeze(0)  # (1, seq, dim)
        return x + pos_embed

    def _pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "max":
            return torch.max(x, dim=1).values
        if self.pooling == "last":
            return x[:, -1, :]
        return torch.mean(x, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        if self.use_positional_encoding:
            x = self._apply_positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        features = self._pool_sequence(x)
        out = self.regressor(features)
        return out


class ModelBuilder:
    """Transformer Encoder 回归器模型工厂（PyTorch版）。"""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 3,
        "ffn_dim": 256,
        "dropout": 0.1,
        "activation": "gelu",
        "pooling": "avg",
        "use_positional_encoding": True,
    }

    @staticmethod
    def get_default_config(_: str = "transformer_encoder_regressor") -> Dict[str, Any]:
        return ModelBuilder.DEFAULT_CONFIG.copy()

    @staticmethod
    def create_model(model_type: str, input_shape: Tuple[int, int], **kwargs) -> nn.Module:
        if model_type != "transformer_encoder_regressor":
            raise ValueError(
                f"Unsupported model_type for Transformer builder: {model_type}"
            )
        config = ModelBuilder.get_default_config(model_type)
        config.update(kwargs)
        return TransformerRegressor(input_shape=input_shape, **config)

    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        if isinstance(model, TransformerRegressor):
            return {
                "model_type": "transformer_encoder_regressor",
                "sequence_length": model.sequence_length,
                "input_dim": model.input_dim,
                "embed_dim": model.embed_dim,
                "pooling": model.pooling,
                "use_positional_encoding": model.use_positional_encoding,
                "trainable_params": sum(p.numel() for p in model.parameters()),
            }
        return {
            "model_type": "unknown",
            "trainable_params": sum(p.numel() for p in model.parameters()),
        }


__all__ = [
    "TransformerRegressor",
    "ModelBuilder",
]
