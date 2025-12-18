"""
LSTM预测异常检测模块 - 模型构建器
定义LSTM预测模型的结构，封装模型初始化和预测逻辑
"""

import mindspore as ms
import mindspore.nn as nn
from typing import Optional, Dict, Any, Tuple
import numpy as np


class LSTMPredictor(nn.Cell):
    """
    基于LSTM的工业异常检测预测器

    核心逻辑：
    - 通过LSTM学习设备正常运行时的时序数据规律
    - 利用"预测偏差"识别异常（残差超过阈值）
    - 支持多维特征同时预测（传感器数据、设备参数等）

    架构设计：
    - 输入：(batch_size, seq_len, n_features) - 时序窗口
    - LSTM层：1-3层，64-256个隐藏单元
    - 输出：(batch_size, n_features) - 下一时刻的特征预测值
    """

    def __init__(self,
                 input_shape: Tuple[int, int],
                 hidden_units: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 activation: str = 'tanh'):
        """
        初始化LSTM异常检测预测器

        Args:
            input_shape: 输入形状 (seq_len, n_features)
            hidden_units: LSTM隐藏单元数
            num_layers: LSTM层数
            dropout: Dropout率
            activation: 激活函数
        """
        super(LSTMPredictor, self).__init__()

        seq_len, n_features = input_shape
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout

        # LSTM编码器
        self.lstm_encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 预测头
        self.predictor = nn.SequentialCell([
            nn.Dense(hidden_units, hidden_units // 2),
            self._get_activation(activation),
            nn.Dropout(p=dropout),
            nn.Dense(hidden_units // 2, n_features)
        ])

        print(f"✅ LSTM预测器构建完成")
        print(f"  - 输入形状: {input_shape}")
        print(f"  - 隐藏单元: {hidden_units}")
        print(f"  - LSTM层数: {num_layers}")

    def _get_activation(self, name: str):
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leakyrelu': nn.LeakyReLU()
        }
        return activations.get(name.lower(), nn.Tanh())

    def construct(self, x):
        """
        前向传播

        Args:
            x: 输入时序 (batch_size, seq_len, n_features)

        Returns:
            预测值 (batch_size, n_features)
        """
        # LSTM编码
        lstm_output, (hidden, cell) = self.lstm_encoder(x)

        # 使用最后一个时刻的输出
        last_hidden = lstm_output[:, -1, :]  # (batch_size, hidden_units)

        # 预测
        prediction = self.predictor(last_hidden)  # (batch_size, n_features)

        return prediction

    def predict(self, x: ms.Tensor) -> ms.Tensor:
        """
        预测接口

        Args:
            x: 输入数据

        Returns:
            预测结果
        """
        return self.construct(x)


class ModelBuilder:
    """
    模型构建器

    提供标准化的模型构建接口，支持不同类型的模型
    """

    @staticmethod
    def build_lstm_predictor(input_shape: Tuple[int, int],
                           hidden_units: int = 128,
                           num_layers: int = 2,
                           dropout: float = 0.1,
                           activation: str = 'tanh') -> LSTMPredictor:
        """
        构建LSTM预测器

        Args:
            input_shape: 输入形状 (seq_len, n_features)
            hidden_units: 隐藏单元数
            num_layers: LSTM层数
            dropout: Dropout率
            activation: 激活函数

        Returns:
            LSTM预测器实例
        """
        return LSTMPredictor(
            input_shape=input_shape,
            hidden_units=hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )

    @staticmethod
    def get_default_config(model_type: str = 'lstm_predictor') -> Dict[str, Any]:
        """
        获取默认模型配置

        Args:
            model_type: 模型类型

        Returns:
            默认配置字典
        """
        configs = {
            'lstm_predictor': {
                'hidden_units': 128,
                'num_layers': 2,
                'dropout': 0.1,
                'activation': 'tanh'
            }
        }
        return configs.get(model_type, {})

    @staticmethod
    def create_model(model_type: str, input_shape: Tuple[int, int], **kwargs) -> nn.Cell:
        """
        创建模型的统一接口

        Args:
            model_type: 模型类型
            input_shape: 输入形状
            **kwargs: 模型参数

        Returns:
            模型实例
        """
        if model_type == 'lstm_predictor':
            config = ModelBuilder.get_default_config(model_type)
            config.update(kwargs)
            return ModelBuilder.build_lstm_predictor(input_shape, **config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    @staticmethod
    def get_model_info(model: nn.Cell) -> Dict[str, Any]:
        """
        获取模型信息

        Args:
            model: 模型实例

        Returns:
            模型信息字典
        """
        if isinstance(model, LSTMPredictor):
            return {
                'model_type': 'lstm_predictor',
                'input_shape': model.input_shape,
                'hidden_units': model.hidden_units,
                'num_layers': model.num_layers,
                'dropout': model.dropout,
                'trainable_params': sum(p.size for p in model.trainable_params())
            }
        else:
            return {
                'model_type': 'unknown',
                'trainable_params': sum(p.size for p in model.trainable_params())
            }


# 向后兼容性函数
def create_model(model_type: str, **kwargs) -> nn.Cell:
    """
    模型工厂函数 - 向后兼容接口

    Args:
        model_type: 模型类型
        **kwargs: 模型参数 (必须包含 input_dim)

    Returns:
        模型实例

    Raises:
        ValueError: 不支持的模型类型或缺少必需参数
    """
    if 'input_dim' not in kwargs:
        raise ValueError("create_model 需要 'input_dim' 参数")

    input_dim = kwargs.pop('input_dim')

    # 使用默认的序列长度
    seq_len = kwargs.pop('seq_len', 100)
    input_shape = (seq_len, input_dim)

    return ModelBuilder.create_model(model_type, input_shape, **kwargs)


def get_default_config(model_type: str) -> dict:
    """
    获取模型默认配置 - 向后兼容接口

    Args:
        model_type: 模型类型

    Returns:
        默认配置字典
    """
    return ModelBuilder.get_default_config(model_type)


def create_model_from_config(config: dict) -> nn.Cell:
    """
    从配置创建模型 - 向后兼容接口

    Args:
        config: 包含model_type和其他参数的配置字典

    Returns:
        模型实例
    """
    model_type = config.get('model_type', 'lstm_predictor')
    input_dim = config.get('input_dim')

    if input_dim is None:
        raise ValueError("配置中必须包含 'input_dim'")

    # 定义模型参数白名单
    model_param_whitelist = {
        'hidden_dim', 'num_layers', 'dropout', 'activation', 'seq_len'
    }

    # 获取模型参数
    model_kwargs = {k: v for k, v in config.items() if k in model_param_whitelist}

    # 合并默认配置
    default_config = get_default_config(model_type)
    final_config = {**default_config, **model_kwargs}

    return create_model(model_type, input_dim=input_dim, **final_config)