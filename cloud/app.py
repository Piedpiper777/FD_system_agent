"""
云端主应用
整合各个功能模块的API
"""

from flask import Flask, request, jsonify
import os
import json
import logging
from pathlib import Path
from config import Config

# 设置项目路径
project_root = Path(__file__).parent.parent
cloud_src_path = Path(__file__).parent / 'src'

# 添加到Python路径
import sys
if str(cloud_src_path) not in sys.path:
    sys.path.insert(0, str(cloud_src_path))

# 导入各模块API蓝图
from anomaly_detection.api import anomaly_detection_bp
from fault_diagnosis.api import fault_diagnosis_bp  
from rul_prediction.api import rul_prediction_bp
from common.model_api import model_management_bp
from common.health_api import health_bp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_class=Config):
    """Flask应用工厂"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # 基础配置
    app.config['JSON_AS_ASCII'] = False
    
    # 注册蓝图
    app.register_blueprint(anomaly_detection_bp)
    app.register_blueprint(fault_diagnosis_bp)
    app.register_blueprint(rul_prediction_bp)
    app.register_blueprint(model_management_bp)
    app.register_blueprint(health_bp)
    
    @app.route('/')
    def index():
        """API根路径"""
        return jsonify({
            'service': 'ICT Cloud Training API',
            'version': '1.0.0',
            'modules': [
                'anomaly_detection',
                'fault_diagnosis', 
                'rul_prediction'
            ],
            'endpoints': {
                'health': '/api/health',
                'models': '/api/models',
                'anomaly_detection': '/api/anomaly_detection',
                'fault_diagnosis': '/api/fault_diagnosis',
                'rul_prediction': '/api/rul_prediction'
            }
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'API endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

# 创建应用实例
app = create_app()

if __name__ == '__main__':
    # 初始化配置
    Config.init_app(app)
    
    # 从配置获取服务地址和端口
    cloud_host = app.config.get('CLOUD_HOST', '0.0.0.0')
    cloud_port = app.config.get('CLOUD_PORT', 5001)
    cloud_base_url = app.config.get('CLOUD_BASE_URL', f'http://localhost:{cloud_port}')
    
    print("🚀 启动云端训练服务...")
    print(f"📡 异常检测API: {cloud_base_url}/api/anomaly_detection")
    print(f"🔧 故障诊断API: {cloud_base_url}/api/fault_diagnosis") 
    print(f"📈 RUL预测API: {cloud_base_url}/api/rul_prediction")
    print(f"🏥 健康检查: {cloud_base_url}/api/health")
    
    app.run(
        host=cloud_host,
        port=cloud_port,
        debug=app.config.get('DEBUG', True),
        threaded=True
    )
