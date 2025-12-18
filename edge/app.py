"""
异常检测系统主应用
模块化架构：按任务组织蓝图
"""

import os
import sys
import threading
import subprocess
from pathlib import Path
from flask import Flask, render_template
from config import Config

# 导入三大任务的蓝图
from src.tasks.anomaly_detection import ad_inference_bp, ad_training_bp, ad_models_bp
from src.tasks.fault_diagnosis import fd_inference_bp, fd_training_bp, fd_models_bp
from src.tasks.rul_prediction import rup_inference_bp, rup_training_bp, rup_models_bp

# 导入通用功能的蓝图
from src.common import file_upload_bp, connection_bp, health_bp, training_bp, chat_stub_bp
from src.common.model_sync import model_sync_bp

# 导入设备监测蓝图
from src.tasks.device_monitoring import device_monitoring_bp

# 导入数据管理蓝图
from src.tasks.data_management import data_management_bp


def create_app(config_class=Config):
    """应用工厂函数"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # 添加安全头
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # 改进的 Content Security Policy
        csp = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com",
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com", 
            "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com data:",
            "img-src 'self' data: https:",
            "connect-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'"
        ]
        response.headers['Content-Security-Policy'] = "; ".join(csp)
        
        return response

    # 注册异常检测蓝图
    app.register_blueprint(ad_training_bp)
    app.register_blueprint(ad_inference_bp)
    app.register_blueprint(ad_models_bp)

    # 注册故障诊断蓝图
    app.register_blueprint(fd_training_bp)
    app.register_blueprint(fd_inference_bp)
    app.register_blueprint(fd_models_bp)

    # 注册RUL预测蓝图
    app.register_blueprint(rup_training_bp)
    app.register_blueprint(rup_inference_bp)
    app.register_blueprint(rup_models_bp)

    # 注册通用功能蓝图
    app.register_blueprint(file_upload_bp)
    app.register_blueprint(connection_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(chat_stub_bp)
    app.register_blueprint(model_sync_bp)

    # 注册设备监测蓝图
    app.register_blueprint(device_monitoring_bp)

    # 注册数据管理蓝图
    app.register_blueprint(data_management_bp)

    # 注册模型管理蓝图
    from edge.src.tasks.model_management import model_management_bp
    app.register_blueprint(model_management_bp)

    # 初始化配置
    Config.init_app(app)

    # 添加根路由
    @app.route('/')
    def index():
        """主页"""
        return render_template('index.html')

    # 添加 task 主页面路由
    @app.route('/anomaly_detection')
    def anomaly_detection_index():
        """异常检测主页"""
        return render_template('anomaly_detection/index.html')

    @app.route('/fault_diagnosis')
    def fault_diagnosis_index():
        """故障诊断主页"""
        return render_template('fault_diagnosis/index.html')

    @app.route('/rul_prediction')
    def rul_prediction_index():
        """RUL预测主页"""
        return render_template('rul_prediction/index.html')

    @app.route('/device_monitoring')
    def device_monitoring_index():
        """设备监测主页"""
        return render_template('device_monitoring/index.html')

    @app.route('/data_management')
    def data_management_index():
        """数据管理主页"""
        return render_template('data_management/index.html')

    # 添加favicon路由
    @app.route('/favicon.ico')
    def favicon():
        # 返回一个简单的ICO数据，避免404错误
        import base64
        # 这是一个16x16的空透明ICO文件
        ico_data = base64.b64decode(
            'AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=='
        )
        from flask import Response
        return Response(ico_data, mimetype='image/x-icon')

    # 添加测试验证页面路由
    @app.route('/test-validation')
    def test_validation_page():
        """参数验证测试页面"""
        return render_template('test_validation.html')

    return app


# 创建应用实例（用于直接运行）
app = create_app()


if __name__ == '__main__':
    # 从配置获取服务地址和端口
    # 注意：EDGE_HOST 用于告诉云端如何访问边缘端，但服务器监听地址应该使用 0.0.0.0
    # 这样可以监听所有网络接口，允许从任何IP访问
    listen_host = '0.0.0.0'  # 监听所有网络接口
    edge_port = app.config.get('EDGE_PORT', 5000)
    edge_host = app.config.get('EDGE_HOST', 'localhost')  # 用于显示的地址
    
    # 用于显示的地址
    display_host = edge_host if edge_host != '0.0.0.0' else 'localhost'
    
    print("Edge服务器启动中...")
    print(f"服务器监听地址: {listen_host}:{edge_port}")
    print(f"访问地址: http://{display_host}:{edge_port}")
    print("按 Ctrl+C 停止服务器")
    print()
    print("[提示] 如需启动数据采集，请运行：")
    print("       python data_collection/start_data_collection.py")

    try:
        app.run(
            host=listen_host,  # 使用 0.0.0.0 监听所有网络接口
            port=edge_port,
            debug=app.config.get('DEBUG', True),
            use_reloader=False  # 禁用reloader以避免重复启动
        )
    except KeyboardInterrupt:
        print("\n服务器正在停止...")
        print("服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
        exit(1)