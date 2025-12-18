"""Transformer Encoder RUL回归器模型构建器。"""

from typing import Any, Dict, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import XavierUniform, initializer


class MultiHeadSelfAttention(nn.Cell):
    """简化的多头自注意力模块，支持时间序列输入。"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Dense(embed_dim, embed_dim * 3)
        self.out_proj = nn.Dense(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = ops.reshape(qkv, (batch, seq_len, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch, seq_len, self.embed_dim))
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        return attn_output


class TransformerEncoderLayer(nn.Cell):
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
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm((embed_dim,))
        self.norm2 = nn.LayerNorm((embed_dim,))
        self.dropout = nn.Dropout(p=dropout)

        self.activation = self._get_activation(activation)
        self.ffn = nn.SequentialCell([
            nn.Dense(embed_dim, ffn_dim),
            self.activation,
            nn.Dropout(p=dropout),
            nn.Dense(ffn_dim, embed_dim),
        ])

    @staticmethod
    def _get_activation(name: str) -> nn.Cell:
        mapping = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leakyrelu": nn.LeakyReLU(alpha=0.2),
        }
        return mapping.get(name.lower(), nn.GELU())

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class TransformerRegressor(nn.Cell):
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
    ):
        super().__init__()
        seq_len, input_dim = input_shape
        self.sequence_length = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.pooling = (pooling or "avg").lower()
        self.use_positional_encoding = use_positional_encoding

        self.input_projection = nn.Dense(input_dim, embed_dim)

        if self.use_positional_encoding:
            self.positional_embedding = ms.Parameter(
                initializer(
                    XavierUniform(),
                    [self.sequence_length, embed_dim],
                    dtype=ms.float32,
                ),
                name="positional_embedding",
            )
        else:
            self.positional_embedding = None

        self.encoder_layers = nn.CellList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(max(1, num_layers))
        ])

        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.reduce_max = ops.ReduceMax(keep_dims=False)

        head_hidden = max(embed_dim, 64)
        head_mid = max(head_hidden // 2, 32)
        self.regressor = nn.SequentialCell([
            nn.Dense(embed_dim, head_hidden),
            self._get_activation(activation),
            nn.Dropout(p=dropout),
            nn.Dense(head_hidden, head_mid),
            self._get_activation(activation),
            nn.Dropout(p=dropout * 0.5),
            nn.Dense(head_mid, 1),
        ])

    @staticmethod
    def _get_activation(name: str) -> nn.Cell:
        mapping = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leakyrelu": nn.LeakyReLU(alpha=0.2),
        }
        return mapping.get(name.lower(), nn.GELU())

    def _apply_positional_encoding(self, x: ms.Tensor) -> ms.Tensor:
        if self.positional_embedding is None:
            return x
        pos_embed = ops.expand_dims(self.positional_embedding, 0)
        return x + pos_embed

    def _pool_sequence(self, x: ms.Tensor) -> ms.Tensor:
        if self.pooling == "max":
            return self.reduce_max(x, 1)
        if self.pooling == "last":
            return x[:, -1, :]
        return self.reduce_mean(x, 1)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # 输入映射到embedding空间
        x = self.input_projection(x)
        if self.use_positional_encoding:
            x = self._apply_positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        features = self._pool_sequence(x)
        return self.regressor(features)


class ModelBuilder:
    """Transformer Encoder 回归器模型工厂。"""

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
    def create_model(model_type: str, input_shape: Tuple[int, int], **kwargs) -> nn.Cell:
        if model_type != "transformer_encoder_regressor":
            raise ValueError(
                f"Unsupported model_type for Transformer builder: {model_type}"
            )
        config = ModelBuilder.get_default_config(model_type)
        config.update(kwargs)
        return TransformerRegressor(input_shape=input_shape, **config)

    @staticmethod
    def get_model_info(model: nn.Cell) -> Dict[str, Any]:
        if isinstance(model, TransformerRegressor):
            return {
                "model_type": "transformer_encoder_regressor",
                "sequence_length": model.sequence_length,
                "input_dim": model.input_dim,
                "embed_dim": model.embed_dim,
                "pooling": model.pooling,
                "use_positional_encoding": model.use_positional_encoding,
                "trainable_params": sum(p.size for p in model.trainable_params()),
            }
        return {
            "model_type": "unknown",
            "trainable_params": sum(p.size for p in model.trainable_params()),
        }


__all__ = [
    "TransformerRegressor",
    "ModelBuilder",
]
