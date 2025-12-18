"""
异常检测模块

基于LSTM预测的时序异常检测系统
支持多模型架构和灵活的训练配置
"""

# 核心组件
from .core.lstm_predicton import LSTMPredictor, ModelBuilder, Trainer

# API蓝图
from .api import anomaly_detection_bp

__version__ = "1.0.0"
__author__ = "喻家山飞昇者"

# 模块级别的配置
SUPPORTED_MODELS = ['lstm_predictor', 'lstm_autoencoder', 'cnn_1d_autoencoder']
DEFAULT_CONFIG = {
    'model_type': 'lstm_predictor',
    'input_dim': None,  # 必须指定
    'seq_len': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 50
}

__all__ = [
    # 核心组件
    'LSTMPredictor', 'ModelBuilder', 'Trainer',
    
    # API蓝图
    'anomaly_detection_bp',
    
    # 常量
    'SUPPORTED_MODELS', 'DEFAULT_CONFIG'
]
