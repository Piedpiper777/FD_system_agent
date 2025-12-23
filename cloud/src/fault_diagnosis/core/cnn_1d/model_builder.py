"""
故障诊断 CNN 1D - 模型构建器
定义用于故障分类的1D CNN模型结构
支持二分类（正常-故障）和三分类（正常-内圈故障-外圈故障）

重构说明：从 MindSpore 迁移到 PyTorch
"""

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DClassifier(nn.Module):
    """
    1D CNN 故障分类器
    
    架构设计：
    - 输入：(batch_size, seq_len, n_features) - 时序窗口
    - 1D卷积层：提取时序特征
    - 全局池化：压缩时序维度
    - 全连接层：分类输出
    - 输出：(batch_size, num_classes) - 类别概率
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        """
        初始化CNN 1D分类器
        
        Args:
            input_shape: 输入形状 (seq_len, n_features)
            num_classes: 分类数量（2或3）
            num_filters: 卷积核数量
            kernel_size: 卷积核大小
            num_conv_layers: 卷积层数量
            dropout: Dropout率
            activation: 激活函数
            use_batch_norm: 是否使用批归一化
        """
        super(CNN1DClassifier, self).__init__()
        
        seq_len, n_features = input_shape
        self.sequence_length = seq_len
        self.input_dim = n_features
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # 编码器：使用1D卷积提取特征
        encoder_layers = []
        in_channels = n_features
        
        for i in range(num_conv_layers):
            # 卷积层，使用 padding='same' 保持序列长度
            encoder_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # 'same' padding
                    bias=not use_batch_norm,  # 如果使用BN，可以不用bias
                )
            )
            
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(num_filters))
            
            encoder_layers.append(self._get_activation(activation))
            encoder_layers.append(nn.Dropout(p=dropout))
            
            in_channels = num_filters
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 全局平均池化：将时序维度压缩为1
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头：全连接层
        # 池化后形状为 (batch, num_filters, 1)，展平后为 (batch, num_filters)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            self._get_activation(activation),
            nn.Dropout(p=dropout),
            nn.Linear(num_filters // 2, num_classes),
        )
    
    def _get_activation(self, name: str) -> nn.Module:
        """根据名称返回激活函数"""
        mapping = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(negative_slope=0.2),
            "gelu": nn.GELU(),
        }
        return mapping.get(name.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, seq_len, n_features)
            return_probs: 是否返回概率（应用softmax），默认返回logits
            
        Returns:
            分类输出 (batch_size, num_classes)
        """
        # 转换输入格式: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        
        # 编码器：提取特征
        encoded = self.encoder(x)  # (batch, num_filters, seq_len)
        
        # 全局平均池化：压缩时序维度
        pooled = self.global_pool(encoded)  # (batch, num_filters, 1)
        
        # 展平: (batch, num_filters, 1) -> (batch, num_filters)
        pooled_flat = pooled.squeeze(-1)  # (batch, num_filters)
        
        # 分类头：输出类别logits
        logits = self.classifier(pooled_flat)  # (batch, num_classes)
        
        # 如果需要概率，应用softmax
        if return_probs:
            return F.softmax(logits, dim=-1)
        
        return logits


class ModelBuilder:
    """CNN 1D 分类器模型工厂"""
    
    DEFAULT_CONFIG = {
        "num_filters": 64,
        "kernel_size": 3,
        "num_conv_layers": 3,
        "dropout": 0.3,
        "activation": "relu",
        "use_batch_norm": True,
    }
    
    @staticmethod
    def get_default_config(model_type: str = "cnn_1d_classifier") -> Dict[str, Any]:
        """获取默认配置"""
        if model_type == "cnn_1d_classifier":
            return ModelBuilder.DEFAULT_CONFIG.copy()
        return {}
    
    @staticmethod
    def build_cnn_1d_classifier(
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ) -> CNN1DClassifier:
        """构建CNN 1D分类器"""
        return CNN1DClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )
    
    @staticmethod
    def create_model(
        model_type: str,
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        **kwargs
    ) -> nn.Module:
        """
        创建模型
        
        Args:
            model_type: 模型类型（目前仅支持 "cnn_1d_classifier"）
            input_shape: 输入形状 (seq_len, n_features)
            num_classes: 分类数量（2或3）
            **kwargs: 其他模型参数
            
        Returns:
            模型实例
        """
        if model_type != "cnn_1d_classifier":
            raise ValueError(
                f"Unsupported model_type for CNN classifier builder: {model_type}"
            )
        
        if num_classes not in [2, 3]:
            raise ValueError(f"num_classes 必须是 2 或 3，当前为 {num_classes}")
        
        config = ModelBuilder.get_default_config(model_type)
        
        # 处理参数名称兼容性
        if 'num_layers' in kwargs:
            kwargs['num_conv_layers'] = kwargs.pop('num_layers')
        
        config.update(kwargs)
        return ModelBuilder.build_cnn_1d_classifier(
            input_shape=input_shape,
            num_classes=num_classes,
            **config
        )
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """获取模型信息"""
        if isinstance(model, CNN1DClassifier):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {
                "model_type": "cnn_1d_classifier",
                "input_dim": model.input_dim,
                "sequence_length": model.sequence_length,
                "num_classes": model.num_classes,
                "num_filters": model.num_filters,
                "kernel_size": model.kernel_size,
                "num_conv_layers": model.num_conv_layers,
                "dropout": model.dropout,
                "use_batch_norm": model.use_batch_norm,
                "trainable_params": total_params,
            }
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "model_type": "unknown",
            "trainable_params": total_params,
        }


def create_model(model_type: str, **kwargs) -> nn.Module:
    """向后兼容接口"""
    if model_type != "cnn_1d_classifier":
        raise ValueError(
            "create_model from CNN classifier module only supports cnn_1d_classifier"
        )
    
    if "input_dim" not in kwargs:
        raise ValueError("Classifier create_model requires 'input_dim'")
    
    input_dim = kwargs.pop("input_dim")
    seq_len = kwargs.pop("seq_len", 50)
    num_classes = kwargs.pop("num_classes", 3)
    input_shape = (seq_len, input_dim)
    
    return ModelBuilder.create_model(
        model_type, input_shape, num_classes=num_classes, **kwargs
    )


def get_default_config(model_type: str = "cnn_1d_classifier") -> Dict[str, Any]:
    """获取默认配置"""
    return ModelBuilder.get_default_config(model_type)


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """从配置字典创建模型"""
    model_type = config.get("model_type", "cnn_1d_classifier")
    input_dim = config.get("input_dim")
    if input_dim is None:
        raise ValueError("config must include 'input_dim'")
    
    seq_len = config.get("seq_len", config.get("sequence_length", 50))
    num_classes = config.get("num_classes", 3)
    input_shape = (seq_len, input_dim)
    
    model_kwargs = {k: v for k, v in config.items() if k in {
        "num_filters",
        "kernel_size",
        "num_conv_layers",
        "dropout",
        "activation",
        "use_batch_norm",
    }}
    
    return ModelBuilder.create_model(
        model_type, input_shape, num_classes=num_classes, **model_kwargs
    )
