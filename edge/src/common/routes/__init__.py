"""
通用功能路由模块
"""

from .file_upload import file_upload_bp
from .connection import connection_bp
from .health import health_bp
from .training_status import training_bp
from .chat_stub import chat_stub_bp

__all__ = [
    'file_upload_bp',
    'connection_bp',
    'health_bp',
    'training_bp',
    'chat_stub_bp'
]
