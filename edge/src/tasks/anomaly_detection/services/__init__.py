"""
异常检测服务层初始化
"""

from .inferencer import LocalAnomalyDetector
from .trainer import AnomalyDetectionTrainer

__all__ = [
    'LocalAnomalyDetector',
    'AnomalyDetectionTrainer'
]
