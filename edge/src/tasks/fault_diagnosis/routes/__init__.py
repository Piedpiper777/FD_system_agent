"""
故障诊断任务路由模块
"""

from .inference import fd_inference_bp
from .training import fd_training_bp
from .models import fd_models_bp

__all__ = [
    'fd_inference_bp',
    'fd_training_bp',
    'fd_models_bp'
]
