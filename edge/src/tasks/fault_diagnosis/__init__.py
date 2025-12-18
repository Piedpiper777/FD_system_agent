"""
故障诊断任务模块
"""

from .routes import fd_inference_bp, fd_training_bp, fd_models_bp

__all__ = [
    'fd_inference_bp',
    'fd_training_bp',
    'fd_models_bp'
]
