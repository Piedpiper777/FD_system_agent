"""
故障诊断 CNN 1D 模块
提供数据处理器、模型构建器和训练器
"""

from .data_processor import (
    DataProcessor,
    ClassificationData,
)
from .model_builder import (
    CNN1DClassifier,
    ModelBuilder,
    create_model,
    get_default_config,
    create_model_from_config,
)
from .trainer import Trainer

__all__ = [
    # 数据处理器
    "DataProcessor",
    "ClassificationData",
    # 模型构建器
    "CNN1DClassifier",
    "ModelBuilder",
    "create_model",
    "get_default_config",
    "create_model_from_config",
    # 训练器
    "Trainer",
]

