"""
配置文件 - 云端训练服务
"""

import os
from pathlib import Path

# 加载环境变量
try:
    from dotenv import load_dotenv
    # 优先加载 cloud/.env 文件
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        # 如果 cloud/.env 不存在，尝试加载项目根目录的 .env
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment variables from {env_path}")
        else:
            print(f"Warning: .env file not found. Using system environment variables or defaults.")
except ImportError:
    print("python-dotenv not installed, using system environment variables")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# Flask配置
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production-immediately'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 500 * 1024 * 1024))  # 默认500MB
    
    # 云端服务配置
    CLOUD_HOST = os.environ.get('CLOUD_HOST') or '0.0.0.0'
    CLOUD_PORT = int(os.environ.get('CLOUD_PORT') or 5001)
    
    # 优先使用 CLOUD_BASE_URL 环境变量，否则从 CLOUD_HOST 和 CLOUD_PORT 构建
    CLOUD_BASE_URL = os.environ.get('CLOUD_BASE_URL') or None
    if not CLOUD_BASE_URL:
        # 如果未设置 CLOUD_BASE_URL，尝试从环境变量获取实际IP
        # 如果 CLOUD_HOST 是 0.0.0.0，使用环境变量中的实际IP或默认值
        cloud_host_for_url = os.environ.get('CLOUD_PUBLIC_HOST') or os.environ.get('CLOUD_HOST') or 'localhost'
        if cloud_host_for_url == '0.0.0.0':
            cloud_host_for_url = 'localhost'
        CLOUD_BASE_URL = f"http://{cloud_host_for_url}:{CLOUD_PORT}"
    else:
        # 如果提供了 CLOUD_BASE_URL，也解析出 HOST 和 PORT（用于兼容）
        from urllib.parse import urlparse
        parsed = urlparse(CLOUD_BASE_URL)
        cloud_host_for_url = parsed.hostname or 'localhost'
        if cloud_host_for_url == '0.0.0.0':
            cloud_host_for_url = 'localhost'
    
    # 边缘端配置（用于通信）
    EDGE_HOST = os.environ.get('EDGE_HOST') or 'localhost'
    EDGE_PORT = int(os.environ.get('EDGE_PORT') or 5000)
    
    # PyTorch配置
    PYTORCH_DEVICE = os.environ.get('PYTORCH_DEVICE') or 'cpu'
    PYTORCH_CUDA_AVAILABLE = os.environ.get('PYTORCH_CUDA_AVAILABLE', '').lower() == 'true'
    
    # 数据目录配置
    MODELS_DIR = Path(os.environ.get('MODELS_DIR') or PROJECT_ROOT / 'models')
    TRAINING_DATA_DIR = Path(os.environ.get('TRAINING_DATA_DIR') or PROJECT_ROOT / 'data' / 'training')
    LOGS_DIR = Path(os.environ.get('LOGS_DIR') or PROJECT_ROOT / 'logs')
    
    # 日志配置
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or str(LOGS_DIR / 'cloud.log')
    
    # 通信配置
    REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT') or 30)
    LONG_REQUEST_TIMEOUT = int(os.environ.get('LONG_REQUEST_TIMEOUT') or 300)
    TRAINING_TIMEOUT = int(os.environ.get('TRAINING_TIMEOUT') or 3600)
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES') or 3)
    RETRY_DELAY = float(os.environ.get('RETRY_DELAY') or 1.0)
    
    @staticmethod
    def init_app(app):
        """初始化应用配置"""
        # 确保必要的目录存在
        config = Config()
        for folder in [config.MODELS_DIR, config.TRAINING_DATA_DIR, config.LOGS_DIR]:
            folder.mkdir(parents=True, exist_ok=True)

# 开发环境配置
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

# 生产环境配置
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key'

# 配置映射
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

