"""
异常检测任务模块
"""

from .routes import ad_inference_bp, ad_training_bp, ad_models_bp

__all__ = [
    'ad_inference_bp',
    'ad_training_bp',
    'ad_models_bp'
]
