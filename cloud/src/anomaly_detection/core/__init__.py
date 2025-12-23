"""
异常检测核心模块初始化


"""

# 导入新的模块化架构
from . import lstm_predicton
from . import lstm_autoencoder
from . import cnn_1d_autoencoder

# LSTM Predictor 组件
from .lstm_predicton import (
    LSTMPredictor,
    create_model,
    get_default_config,
    create_model_from_config,
    Trainer,
    Evaluator,
    create_evaluator_from_config,
    get_default_evaluator_config,
    DataProcessor,
    ModelBuilder,
    ThresholdCalculator,
)

# LSTM Autoencoder 组件
from .lstm_autoencoder import (
    LSTMAutoencoder,
    ModelBuilder as LSTMAutoencoderModelBuilder,
    Trainer as LSTMAutoencoderTrainer,
    DataProcessor as LSTMAutoencoderDataProcessor,
    ThresholdCalculator as LSTMAutoencoderThresholdCalculator,
)

# CNN 1D Autoencoder 组件
from .cnn_1d_autoencoder import (
    CNN1DAutoencoder,
    ModelBuilder as CNN1DAutoencoderModelBuilder,
    Trainer as CNN1DAutoencoderTrainer,
)

# 向后兼容别名
LSTMPredictorEvaluator = Evaluator
LSTMPredictionEvaluator = Evaluator

__all__ = [
    # 新架构模块
    'lstm_predicton',
    'lstm_autoencoder',
    'cnn_1d_autoencoder',

    # LSTM Predictor
    'LSTMPredictor',
    'create_model',
    'get_default_config',
    'create_model_from_config',

    # LSTM Autoencoder
    'LSTMAutoencoder',
    'LSTMAutoencoderModelBuilder',
    'LSTMAutoencoderTrainer',
    'LSTMAutoencoderDataProcessor',
    'LSTMAutoencoderThresholdCalculator',

    # CNN 1D Autoencoder
    'CNN1DAutoencoder',
    'CNN1DAutoencoderModelBuilder',
    'CNN1DAutoencoderTrainer',

    # 训练器
    'Trainer',

    # 评估器
    'Evaluator',
    'LSTMPredictorEvaluator',
    'LSTMPredictionEvaluator',
    'create_evaluator_from_config',
    'get_default_evaluator_config',

    # 数据处理和阈值
    'DataProcessor',
    'ModelBuilder',
    'ThresholdCalculator',
]
