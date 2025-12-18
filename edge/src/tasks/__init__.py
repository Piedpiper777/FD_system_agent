"""
任务模块 - 三大主要任务
"""

from .anomaly_detection import ad_inference_bp, ad_training_bp, ad_models_bp
from .fault_diagnosis import fd_inference_bp, fd_training_bp, fd_models_bp
from .rul_prediction import rup_inference_bp, rup_training_bp

__all__ = [
    # 异常检测
    'ad_inference_bp',
    'ad_training_bp',
    'ad_models_bp',
    # 故障诊断
    'fd_inference_bp',
    'fd_training_bp',
    'fd_models_bp',
    # RUL预测
    'rup_inference_bp',
    'rup_training_bp'
]
