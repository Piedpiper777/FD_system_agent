"""
异常检测任务路由模块
"""

from .inference import ad_inference_bp
from .training import ad_training_bp
from .models import ad_models_bp

__all__ = [
    'ad_inference_bp',
    'ad_training_bp',
    'ad_models_bp'
]
