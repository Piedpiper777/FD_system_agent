"""
RUL预测 BiLSTM/GRU 回归器 - 模型构建器
定义用于RUL预测的回归模型结构
支持BiLSTM和GRU，可选Attention机制
"""

from typing import Dict, Any, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class BiLSTMGRURegressor(nn.Cell):
    """
    BiLSTM/GRU RUL回归器
    
    架构设计：
    - 输入：(batch_size, seq_len, n_features) - 时序窗口
    - LSTM/GRU层：提取时序特征（支持双向）
    - Attention层（可选）：关注重要时间步
    - 全连接层：回归输出
    - 输出：(batch_size, 1) - RUL值
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
    ):
        """
        初始化BiLSTM/GRU回归器
        
        Args:
            input_shape: 输入形状 (seq_len, n_features)
            hidden_units: RNN隐藏单元数
            num_layers: RNN层数
            dropout: Dropout率
            activation: 激活函数
            bidirectional: 是否使用双向RNN
            use_attention: 是否使用注意力机制
            use_layer_norm: 是否使用层归一化
            rnn_type: RNN类型，"lstm" 或 "gru"
        """
        super(BiLSTMGRURegressor, self).__init__()
        
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
        
        # RNN层（LSTM或GRU）
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
        
        # 计算RNN输出维度（双向时是2倍）
        rnn_output_dim = hidden_units * 2 if bidirectional else hidden_units
        
        # 注意力机制（可选）
        if use_attention:
            self.attention = nn.Dense(rnn_output_dim, 1)
        else:
            self.attention = None
        
        # 回归头：全连接层
        regressor_layers = []
        # 第一层
        regressor_layers.append(nn.Dense(rnn_output_dim, rnn_output_dim // 2))
        if use_layer_norm:
            regressor_layers.append(nn.LayerNorm((rnn_output_dim // 2,)))
        regressor_layers.append(self._get_activation(activation))
        regressor_layers.append(nn.Dropout(p=dropout))
        
        # 第二层
        regressor_layers.append(nn.Dense(rnn_output_dim // 2, rnn_output_dim // 4))
        if use_layer_norm:
            regressor_layers.append(nn.LayerNorm((rnn_output_dim // 4,)))
        regressor_layers.append(self._get_activation(activation))
        regressor_layers.append(nn.Dropout(p=dropout * 0.5))
        
        # 输出层
        regressor_layers.append(nn.Dense(rnn_output_dim // 4, 1))  # 输出单个RUL值
        
        self.regressor = nn.SequentialCell(regressor_layers)
    
    def _get_activation(self, name: str):
        """根据名称返回激活函数"""
        mapping = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(alpha=0.2),
            "gelu": nn.GELU(),
        }
        return mapping.get(name.lower(), nn.ReLU())
    
    def _apply_attention(self, rnn_output: ms.Tensor) -> ms.Tensor:
        """
        应用注意力机制
        
        Args:
            rnn_output: RNN输出 (batch_size, seq_len, hidden_units)
            
        Returns:
            加权后的特征向量 (batch_size, hidden_units)
        """
        # 计算注意力权重
        attention_weights = self.attention(rnn_output)  # (batch_size, seq_len, 1)
        attention_weights = ops.Softmax(axis=1)(attention_weights)  # 归一化
        
        # 加权求和
        weighted_output = ops.ReduceSum()(
            rnn_output * attention_weights, axis=1
        )  # (batch_size, hidden_units)
        
        return weighted_output
    
    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, seq_len, n_features)
            
        Returns:
            RUL预测值 (batch_size, 1)
        """
        # RNN编码：提取时序特征
        rnn_output, _ = self.rnn(x)
        # rnn_output: (batch_size, seq_len, hidden_units * num_directions)
        
        if self.use_attention and self.attention is not None:
            # 使用注意力机制聚合序列信息
            features = self._apply_attention(rnn_output)
        else:
            # 使用最后一个时刻的隐藏状态
            features = rnn_output[:, -1, :]  # (batch_size, hidden_units * num_directions)
        
        # 回归头：输出RUL值
        rul_pred = self.regressor(features)  # (batch_size, 1)
        
        return rul_pred


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
        """获取默认配置"""
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
        """构建BiLSTM/GRU回归器"""
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
    ) -> nn.Cell:
        """
        创建模型
        
        Args:
            model_type: 模型类型（目前仅支持 "bilstm_gru_regressor"）
            input_shape: 输入形状 (seq_len, n_features)
            **kwargs: 其他模型参数
            
        Returns:
            模型实例
        """
        if model_type != "bilstm_gru_regressor":
            raise ValueError(
                f"Unsupported model_type for BiLSTM/GRU regressor builder: {model_type}"
            )
        
        config = ModelBuilder.get_default_config(model_type)
        
        # 处理参数名称兼容性
        if 'hidden_size' in kwargs:
            kwargs['hidden_units'] = kwargs.pop('hidden_size')
        
        config.update(kwargs)
        return ModelBuilder.build_regressor(
            input_shape=input_shape,
            **config
        )
    
    @staticmethod
    def get_model_info(model: nn.Cell) -> Dict[str, Any]:
        """获取模型信息"""
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
                "trainable_params": sum(p.size for p in model.trainable_params()),
            }
        return {
            "model_type": "unknown",
            "trainable_params": sum(p.size for p in model.trainable_params()),
        }

