"""
配置文件 - 异常检测系统Edge服务
"""

import os
from pathlib import Path

# 加载环境变量
try:
    from dotenv import load_dotenv
    # 优先加载 edge/.env 文件
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        # 如果 edge/.env 不存在，尝试加载项目根目录的 .env
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
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

    # 上传文件配置
    UPLOAD_FOLDER = PROJECT_ROOT / 'data' / 'uploaded'
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'npy', 'pkl'}

    # 模型保存配置
    MODEL_FOLDER = PROJECT_ROOT / 'models'

    # 配置保存配置
    CONFIG_FOLDER = PROJECT_ROOT / 'static' / 'configs'

    # 对话占位文件配置（悬浮窗临时中转）
    CHAT_LOG_DIR = PROJECT_ROOT / 'data' / 'chat'
    CHAT_INPUT_FILE = CHAT_LOG_DIR / 'input.txt'
    CHAT_OUTPUT_FILE = CHAT_LOG_DIR / 'output.txt'

    # Cloud模块路径
    CLOUD_PATH = PROJECT_ROOT.parent / 'cloud'

    # 云端服务器配置
    # 优先使用 CLOUD_BASE_URL 环境变量，否则从 CLOUD_HOST 和 CLOUD_PORT 构建
    CLOUD_BASE_URL = os.environ.get('CLOUD_BASE_URL') or None
    if not CLOUD_BASE_URL:
        CLOUD_HOST = os.environ.get('CLOUD_HOST') or 'localhost'
        CLOUD_PORT = int(os.environ.get('CLOUD_PORT') or 5001)
        CLOUD_BASE_URL = f"http://{CLOUD_HOST}:{CLOUD_PORT}"
    else:
        # 如果提供了 CLOUD_BASE_URL，也解析出 HOST 和 PORT（用于兼容）
        from urllib.parse import urlparse
        parsed = urlparse(CLOUD_BASE_URL)
        CLOUD_HOST = parsed.hostname or 'localhost'
        CLOUD_PORT = parsed.port or 5001

    # 边缘端自身配置（用于告诉云端如何下载文件）
    EDGE_HOST = os.environ.get('EDGE_HOST') or 'localhost'  # 边缘端自己的实际IP地址
    EDGE_PORT = int(os.environ.get('EDGE_PORT') or 5000)   # 边缘端自己的端口

    @staticmethod
    def init_app(app):
        """初始化应用配置"""
        # 确保必要的目录存在
        for folder in [Config.UPLOAD_FOLDER, Config.MODEL_FOLDER, Config.CONFIG_FOLDER, Config.CHAT_LOG_DIR]:
            folder.mkdir(parents=True, exist_ok=True)

        # 确保占位文件存在
        for file_path in [Config.CHAT_INPUT_FILE, Config.CHAT_OUTPUT_FILE]:
            if not file_path.exists():
                file_path.touch()

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