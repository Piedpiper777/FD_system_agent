"""
LSTM预测异常检测模块
基于LSTM的时间序列预测异常检测方法
"""

from .model_builder import LSTMPredictor, ModelBuilder, create_model, get_default_config, create_model_from_config
from .trainer import Trainer  # 新架构：简化的训练器
from .evaluator import Evaluator, get_default_evaluator_config, create_evaluator_from_config  # 新架构：评估器
# 导入新的模块化架构组件
from .data_processor import DataProcessor
from .model_builder import ModelBuilder
from .threshold_calculator import ThresholdCalculator
from .anomaly_detector import AnomalyDetector

# 向后兼容性别名
LSTMPredictorInference = AnomalyDetector  # 新的AnomalyDetector替代旧的LSTMPredictorInference

# 向后兼容性推理函数
def create_inference_engine(model_type: str, model, **kwargs):
    """创建推理引擎 - 向后兼容接口"""
    return AnomalyDetector(model, **kwargs)

def get_default_inference_config(model_type: str = 'lstm_predictor'):
    """获取默认推理配置 - 向后兼容接口"""
    return {
        'threshold': 0.8,
        'window_size': 100,
        'step_size': 1
    }

def create_inference_from_config(config, model):
    """从配置创建推理引擎 - 向后兼容接口"""
    return AnomalyDetector(model, **config)

__all__ = [
    # 模型
    'LSTMPredictor', 'ModelBuilder', 'create_model', 'get_default_config', 'create_model_from_config',

    # 训练器（新架构）
    'Trainer',

    # 评估器（新架构）
    'Evaluator', 'LSTMPredictorEvaluator', 'get_default_evaluator_config', 'create_evaluator_from_config',

    # 推理器（保留向后兼容）
    'LSTMPredictorInference', 'create_inference_engine', 'get_default_inference_config', 'create_inference_from_config',

    # 新模块化架构组件
    'DataProcessor', 'ModelBuilder', 'Trainer', 'ThresholdCalculator', 'AnomalyDetector', 'AnomalyDetectorLegacy'
]