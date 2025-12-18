"""
1D CNN自编码器 - 模型构建器
定义用于工业时序重构的1D CNN Autoencoder结构
"""

from typing import Dict, Any, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class CNN1DAutoencoder(nn.Cell):
    """1D CNN Autoencoder，用于学习正常序列并重构输入窗口"""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_filters: int = 64,
        kernel_size: int = 3,
        bottleneck_size: int = 64,
        num_conv_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        seq_len, n_features = input_shape
        self.sequence_length = seq_len
        self.input_dim = n_features
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.bottleneck_size = bottleneck_size
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout

        # 编码器：使用1D卷积提取特征
        encoder_layers = []
        in_channels = n_features
        
        for i in range(num_conv_layers):
            encoder_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    pad_mode='same',
                    has_bias=True
                ),
                self._get_activation(activation),
                nn.Dropout(p=dropout),
            ])
            in_channels = num_filters
        
        # 添加池化层以压缩序列长度
        encoder_layers.append(nn.AdaptiveAvgPool1d(1))  # 将序列压缩到长度为1
        self.encoder = nn.SequentialCell(encoder_layers)
        
        # 潜在空间映射：将压缩后的特征映射到瓶颈层
        # 池化后形状为 (batch, num_filters, 1)，展平后为 (batch, num_filters)
        self.latent_projection = nn.SequentialCell([
            nn.Dense(num_filters, bottleneck_size),
            self._get_activation(activation),
        ])
        
        # 解码器：从瓶颈层重构序列
        # 首先将瓶颈向量扩展回序列长度
        self.decoder_projection = nn.SequentialCell([
            nn.Dense(bottleneck_size, num_filters),
            self._get_activation(activation),
        ])
        
        # 解码器卷积层：重构序列
        decoder_layers = []
        for i in range(num_conv_layers):
            decoder_layers.extend([
                nn.Conv1dTranspose(
                    in_channels=num_filters if i == 0 else num_filters,
                    out_channels=num_filters if i < num_conv_layers - 1 else n_features,
                    kernel_size=kernel_size,
                    stride=1,
                    pad_mode='same',
                    has_bias=True
                ),
                self._get_activation(activation) if i < num_conv_layers - 1 else nn.Identity(),
                nn.Dropout(p=dropout) if i < num_conv_layers - 1 else nn.Identity(),
            ])
        
        self.decoder = nn.SequentialCell(decoder_layers)

    def _get_activation(self, name: str):
        """根据名称返回激活函数"""
        mapping = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(),
        }
        return mapping.get(name.lower(), nn.ReLU())

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """前向传播，返回重构序列
        
        Args:
            x: 输入序列 (batch_size, seq_len, n_features)
        
        Returns:
            重构序列 (batch_size, seq_len, n_features)
        """
        batch_size = x.shape[0]
        
        # 转换输入格式: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(0, 2, 1)  # (batch, n_features, seq_len)
        
        # 编码器：提取特征并压缩
        encoded = self.encoder(x)  # (batch, num_filters, 1)
        
        # 展平: (batch, num_filters, 1) -> (batch, num_filters)
        encoded_flat = encoded.squeeze(-1)  # (batch, num_filters)
        
        # 潜在空间映射
        latent = self.latent_projection(encoded_flat)  # (batch, bottleneck_size)
        
        # 解码器投影：扩展回特征维度
        decoded_flat = self.decoder_projection(latent)  # (batch, num_filters)
        
        # 扩展回序列长度: (batch, num_filters) -> (batch, num_filters, seq_len)
        decoded_flat = decoded_flat.expand_dims(-1)  # (batch, num_filters, 1)
        decoded_flat = ops.tile(decoded_flat, (1, 1, self.sequence_length))  # (batch, num_filters, seq_len)
        
        # 解码器：重构序列
        reconstruction = self.decoder(decoded_flat)  # (batch, n_features, seq_len)
        
        # 转换回原始格式: (batch, n_features, seq_len) -> (batch, seq_len, n_features)
        reconstruction = reconstruction.transpose(0, 2, 1)  # (batch, seq_len, n_features)
        
        return reconstruction


class ModelBuilder:
    """1D CNN Autoencoder 模型工厂"""

    DEFAULT_CONFIG = {
        "num_filters": 64,
        "kernel_size": 3,
        "bottleneck_size": 64,
        "num_conv_layers": 3,
        "dropout": 0.1,
        "activation": "relu",
    }

    @staticmethod
    def get_default_config(model_type: str = "cnn_1d_autoencoder") -> Dict[str, Any]:
        if model_type == "cnn_1d_autoencoder":
            return ModelBuilder.DEFAULT_CONFIG.copy()
        return {}

    @staticmethod
    def build_cnn_1d_autoencoder(
        input_shape: Tuple[int, int],
        num_filters: int = 64,
        kernel_size: int = 3,
        bottleneck_size: int = 64,
        num_conv_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> CNN1DAutoencoder:
        return CNN1DAutoencoder(
            input_shape=input_shape,
            num_filters=num_filters,
            kernel_size=kernel_size,
            bottleneck_size=bottleneck_size,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            activation=activation,
        )

    @staticmethod
    def create_model(model_type: str, input_shape: Tuple[int, int], **kwargs) -> nn.Cell:
        if model_type != "cnn_1d_autoencoder":
            raise ValueError(f"Unsupported model_type for CNN autoencoder builder: {model_type}")

        config = ModelBuilder.get_default_config(model_type)
        
        # 处理参数名称兼容性
        if 'bottleneck_dim' in kwargs:
            kwargs['bottleneck_size'] = kwargs.pop('bottleneck_dim')
        if 'num_layers' in kwargs:
            kwargs['num_conv_layers'] = kwargs.pop('num_layers')
            
        config.update(kwargs)
        return ModelBuilder.build_cnn_1d_autoencoder(input_shape, **config)

    @staticmethod
    def get_model_info(model: nn.Cell) -> Dict[str, Any]:
        if isinstance(model, CNN1DAutoencoder):
            return {
                "model_type": "cnn_1d_autoencoder",
                "input_dim": model.input_dim,
                "sequence_length": model.sequence_length,
                "num_filters": model.num_filters,
                "kernel_size": model.kernel_size,
                "bottleneck_size": model.bottleneck_size,
                "num_conv_layers": model.num_conv_layers,
                "dropout": model.dropout,
                "trainable_params": sum(p.size for p in model.trainable_params()),
            }
        return {
            "model_type": "unknown",
            "trainable_params": sum(p.size for p in model.trainable_params()),
        }


def create_model(model_type: str, **kwargs) -> nn.Cell:
    """向后兼容接口"""
    if model_type != "cnn_1d_autoencoder":
        raise ValueError("create_model from CNN autoencoder module only supports cnn_1d_autoencoder")

    if "input_dim" not in kwargs:
        raise ValueError("Autoencoder create_model requires 'input_dim'")

    input_dim = kwargs.pop("input_dim")
    seq_len = kwargs.pop("seq_len", 50)
    input_shape = (seq_len, input_dim)
    return ModelBuilder.create_model(model_type, input_shape, **kwargs)


def get_default_config(model_type: str = "cnn_1d_autoencoder") -> Dict[str, Any]:
    return ModelBuilder.get_default_config(model_type)


def create_model_from_config(config: Dict[str, Any]) -> nn.Cell:
    model_type = config.get("model_type", "cnn_1d_autoencoder")
    input_dim = config.get("input_dim")
    if input_dim is None:
        raise ValueError("config must include 'input_dim'")

    seq_len = config.get("seq_len", config.get("sequence_length", 50))
    model_kwargs = {k: v for k, v in config.items() if k in {
        "num_filters",
        "kernel_size",
        "bottleneck_size",
        "num_conv_layers",
        "dropout",
        "activation",
    }}

    return create_model(model_type, input_dim=input_dim, seq_len=seq_len, **model_kwargs)

