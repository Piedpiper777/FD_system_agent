"""
RUL预测任务路由模块
"""

from .inference import rup_inference_bp
from .training import rup_training_bp
from .models import rup_models_bp

__all__ = [
    'rup_inference_bp',
    'rup_training_bp',
    'rup_models_bp'
]
