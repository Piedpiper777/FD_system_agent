"""
故障诊断 LSTM - 模型构建器
定义用于故障分类的LSTM模型结构
支持二分类（正常-故障）和三分类（正常-内圈故障-外圈故障）


"""

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """
    LSTM 故障分类器
    
    架构设计：
    - 输入：(batch_size, seq_len, n_features) - 时序窗口
    - LSTM层：提取时序特征
    - 全连接层：分类输出
    - 输出：(batch_size, num_classes) - 类别概率
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        hidden_units: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = "tanh",
        bidirectional: bool = False,
        use_attention: bool = False,
    ):
        """
        初始化LSTM分类器
        
        Args:
            input_shape: 输入形状 (seq_len, n_features)
            num_classes: 分类数量（2或3）
            hidden_units: LSTM隐藏单元数
            num_layers: LSTM层数
            dropout: Dropout率
            activation: 激活函数
            bidirectional: 是否使用双向LSTM
            use_attention: 是否使用注意力机制
        """
        super(LSTMClassifier, self).__init__()
        
        seq_len, n_features = input_shape
        self.sequence_length = seq_len
        self.input_dim = n_features
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # 计算LSTM输出维度（双向时是2倍）
        lstm_output_dim = hidden_units * 2 if bidirectional else hidden_units
        
        # 注意力机制（可选）
        if use_attention:
            self.attention = nn.Linear(lstm_output_dim, 1)
        else:
            self.attention = None
        
        # 分类头：全连接层
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            self._get_activation(activation),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_output_dim // 2, num_classes),
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
        return mapping.get(name.lower(), nn.Tanh())
    
    def _apply_attention(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        应用注意力机制
        
        Args:
            lstm_output: LSTM输出 (batch_size, seq_len, hidden_units)
            
        Returns:
            加权后的特征向量 (batch_size, hidden_units)
        """
        # 计算注意力权重
        attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # 归一化
        
        # 加权求和
        weighted_output = torch.sum(
            lstm_output * attention_weights, dim=1
        )  # (batch_size, hidden_units)
        
        return weighted_output
    
    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, seq_len, n_features)
            return_probs: 是否返回概率（应用softmax），默认返回logits
            
        Returns:
            分类输出 (batch_size, num_classes)
        """
        # LSTM编码：提取时序特征
        lstm_output, (hidden, cell) = self.lstm(x)
        # lstm_output: (batch_size, seq_len, hidden_units * num_directions)
        
        if self.use_attention and self.attention is not None:
            # 使用注意力机制聚合序列信息
            features = self._apply_attention(lstm_output)
        else:
            # 使用最后一个时刻的隐藏状态
            features = lstm_output[:, -1, :]  # (batch_size, hidden_units * num_directions)
        
        # 分类头：输出类别logits
        logits = self.classifier(features)  # (batch_size, num_classes)
        
        # 如果需要概率，应用softmax
        if return_probs:
            return F.softmax(logits, dim=-1)
        
        return logits


class ModelBuilder:
    """LSTM 分类器模型工厂"""
    
    DEFAULT_CONFIG = {
        "hidden_units": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "activation": "tanh",
        "bidirectional": False,
        "use_attention": False,
    }
    
    @staticmethod
    def get_default_config(model_type: str = "lstm_classifier") -> Dict[str, Any]:
        """获取默认配置"""
        if model_type == "lstm_classifier":
            return ModelBuilder.DEFAULT_CONFIG.copy()
        return {}
    
    @staticmethod
    def build_lstm_classifier(
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        hidden_units: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = "tanh",
        bidirectional: bool = False,
        use_attention: bool = False,
    ) -> LSTMClassifier:
        """构建LSTM分类器"""
        return LSTMClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_units=hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            bidirectional=bidirectional,
            use_attention=use_attention,
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
            model_type: 模型类型（目前仅支持 "lstm_classifier"）
            input_shape: 输入形状 (seq_len, n_features)
            num_classes: 分类数量（2或3）
            **kwargs: 其他模型参数
            
        Returns:
            模型实例
        """
        if model_type != "lstm_classifier":
            raise ValueError(
                f"Unsupported model_type for LSTM classifier builder: {model_type}"
            )
        
        if num_classes not in [2, 3]:
            raise ValueError(f"num_classes 必须是 2 或 3，当前为 {num_classes}")
        
        config = ModelBuilder.get_default_config(model_type)
        
        # 处理参数名称兼容性
        if 'hidden_size' in kwargs:
            kwargs['hidden_units'] = kwargs.pop('hidden_size')
        
        config.update(kwargs)
        return ModelBuilder.build_lstm_classifier(
            input_shape=input_shape,
            num_classes=num_classes,
            **config
        )
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """获取模型信息"""
        if isinstance(model, LSTMClassifier):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {
                "model_type": "lstm_classifier",
                "input_dim": model.input_dim,
                "sequence_length": model.sequence_length,
                "num_classes": model.num_classes,
                "hidden_units": model.hidden_units,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
                "bidirectional": model.bidirectional,
                "use_attention": model.use_attention,
                "trainable_params": total_params,
            }
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "model_type": "unknown",
            "trainable_params": total_params,
        }


def create_model(model_type: str, **kwargs) -> nn.Module:
    """向后兼容接口"""
    if model_type != "lstm_classifier":
        raise ValueError(
            "create_model from LSTM classifier module only supports lstm_classifier"
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


def get_default_config(model_type: str = "lstm_classifier") -> Dict[str, Any]:
    """获取默认配置"""
    return ModelBuilder.get_default_config(model_type)


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """从配置字典创建模型"""
    model_type = config.get("model_type", "lstm_classifier")
    input_dim = config.get("input_dim")
    if input_dim is None:
        raise ValueError("config must include 'input_dim'")
    
    seq_len = config.get("seq_len", config.get("sequence_length", 50))
    num_classes = config.get("num_classes", 3)
    input_shape = (seq_len, input_dim)
    
    model_kwargs = {k: v for k, v in config.items() if k in {
        "hidden_units",
        "num_layers",
        "dropout",
        "activation",
        "bidirectional",
        "use_attention",
    }}
    
    return ModelBuilder.create_model(
        model_type, input_shape, num_classes=num_classes, **model_kwargs
    )
