"""
RUL预测任务模块
"""

from .routes import rup_inference_bp, rup_training_bp, rup_models_bp

__all__ = [
    'rup_inference_bp',
    'rup_training_bp',
    'rup_models_bp'
]
