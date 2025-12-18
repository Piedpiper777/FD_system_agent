"""
异常检测核心模块初始化
"""

# 导入新的模块化架构
from . import lstm_predicton
from . import lstm_autoencoder

# 保持向后兼容性 - 从 lstm_predictor 模块导入
from .lstm_predicton import (
    LSTMPredictor,
    create_model,
    get_default_config,
    create_model_from_config,
    Trainer,  # 更新：使用新的Trainer类
    Evaluator,  # 更新：使用新的Evaluator类
    create_evaluator_from_config,
    get_default_evaluator_config,
    LSTMPredictorInference,  # 保留但可能需要重构
    create_inference_engine,
    get_default_inference_config,
    create_inference_from_config
)

# 导入新的模块化组件
from .lstm_predicton import (
    DataProcessor,
    ModelBuilder,
    Trainer,
    ThresholdCalculator,
    AnomalyDetector
)

# LSTM Autoencoder 组件
from .lstm_autoencoder import (
    LSTMAutoencoder,
    ModelBuilder as LSTMAutoencoderModelBuilder,
    Trainer as LSTMAutoencoderTrainer,
    DataProcessor as LSTMAutoencoderDataProcessor,
    ThresholdCalculator as LSTMAutoencoderThresholdCalculator,
    AnomalyDetector as LSTMAutoencoderDetector
)

# 为了向后兼容，保留旧的导入名称
from .lstm_predicton import Evaluator as LSTMPredictorEvaluator  # 向后兼容
from .lstm_predicton import Evaluator as LSTMPredictionEvaluator  # 向后兼容
from .lstm_predicton import LSTMPredictorInference as AnomalyDetectorLegacy  # 向后兼容

__all__ = [
    # 新架构模块
    'lstm_predicton',
    'lstm_autoencoder',

    # 模型
    'LSTMPredictor', 'create_model', 'get_default_config', 'create_model_from_config',

    # LSTM Autoencoder 模块
    'LSTMAutoencoder', 'LSTMAutoencoderModelBuilder', 'LSTMAutoencoderTrainer',
    'LSTMAutoencoderDataProcessor', 'LSTMAutoencoderThresholdCalculator',
    'LSTMAutoencoderDetector',

    # 训练器
    'Trainer',

    # 评估器
    'Evaluator', 'LSTMPredictorEvaluator', 'LSTMPredictionEvaluator', 'create_evaluator_from_config', 'get_default_evaluator_config',

    # 推理器（保留向后兼容）
    'LSTMPredictorInference', 'AnomalyDetectorLegacy', 'create_inference_engine', 'get_default_inference_config', 'create_inference_from_config',

    # 新模块化组件
    'DataProcessor', 'ModelBuilder', 'Trainer', 'ThresholdCalculator', 'AnomalyDetector'
]