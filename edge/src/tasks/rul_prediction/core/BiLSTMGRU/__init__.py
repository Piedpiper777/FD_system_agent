"""
BiLSTM/GRU 回归器模块
用于RUL预测的回归模型
"""

from .model_builder import ModelBuilder, BiLSTMGRURegressor

__all__ = [
    'ModelBuilder',
    'BiLSTMGRURegressor',
]

