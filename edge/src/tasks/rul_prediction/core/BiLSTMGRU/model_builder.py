"""
RUL预测 BiLSTM/GRU 回归器 - 模型构建器（边端版本）
与云端版本保持一致，用于边端推理
"""

from typing import Dict, Any, Tuple
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class BiLSTMGRURegressor(nn.Cell):
    """
    BiLSTM/GRU RUL回归器
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
        rnn_type: str = "lstm",
    ):
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
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, n_features)
            
        Returns:
            rul_pred: RUL预测值 (batch_size, 1)
        """
        # RNN前向传播
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, rnn_output_dim)
        
        # 注意力机制（可选）
        if self.attention is not None:
            # 计算注意力权重
            attention_weights = self.attention(rnn_out)  # (batch_size, seq_len, 1)
            attention_weights = ops.Softmax(axis=1)(attention_weights)  # 归一化
            
            # 加权求和
            context = ops.ReduceSum()(rnn_out * attention_weights, axis=1)  # (batch_size, rnn_output_dim)
        else:
            # 不使用注意力，使用最后一个时间步的输出
            context = rnn_out[:, -1, :]  # (batch_size, rnn_output_dim)
        
        # 回归预测
        rul_pred = self.regressor(context)  # (batch_size, 1)
        
        return rul_pred


class ModelBuilder:
    """BiLSTM/GRU 回归器模型工厂"""
    
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
            model_type: 模型类型
            input_shape: 输入形状 (seq_len, n_features)
            **kwargs: 其他模型参数
            
        Returns:
            模型实例
        """
        if model_type == 'bilstm_gru_regressor':
            return ModelBuilder.build_regressor(
                input_shape=input_shape,
                hidden_units=kwargs.get('hidden_units', 128),
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.3),
                activation=kwargs.get('activation', 'relu'),
                bidirectional=kwargs.get('bidirectional', True),
                use_attention=kwargs.get('use_attention', True),
                use_layer_norm=kwargs.get('use_layer_norm', True),
                rnn_type=kwargs.get('rnn_type', 'lstm'),
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

