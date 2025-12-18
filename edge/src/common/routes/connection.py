"""
连接管理路由
"""

from flask import Blueprint, jsonify
from src.common.services.cloud_communication import CloudCommunicationService

connection_bp = Blueprint('connection', __name__, url_prefix='/api')

# 创建通信服务实例
comm_service = CloudCommunicationService('auto')


@connection_bp.route('/connection/status', methods=['GET'])
def get_connection_status():
    """获取连接状态"""
    try:
        status = comm_service.get_communication_status()
        is_connected = comm_service._check_cloud_availability()

        if is_connected:
            message = "✅ 云端高性能服务器连接正常"
            service_type = "云端高性能计算服务"
        else:
            message = "❌ 未连接到云端服务器"
            service_type = "未连接"

        return jsonify({
            'success': True,
            'connected': is_connected,
            'mode': status.get('mode', 'unknown'),
            'service_type': service_type,
            'cloud_host': comm_service.cloud_base_url,
            'message': message,
            'retry_config': {
                'max_retries': comm_service.max_retries,
                'retry_delay': comm_service.retry_delay,
                'timeout': comm_service.request_timeout
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'connected': False,
            'message': f'检查连接状态失败: {str(e)}'
        }), 500


@connection_bp.route('/connection/connect', methods=['POST'])
def connect_to_server():
    """连接到服务器"""
    try:
        global comm_service
        comm_service = CloudCommunicationService('http')

        if comm_service._check_cloud_availability():
            return jsonify({
                'success': True,
                'message': '✅ 成功连接到云端服务器！高性能计算资源已就绪',
                'mode': 'http',
                'service_type': '云端高性能计算服务'
            })
        else:
            return jsonify({
                'success': False,
                'message': '❌ 无法连接到云端服务器'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'连接失败: {str(e)}'
        }), 500


@connection_bp.route('/connection/disconnect', methods=['POST'])
def disconnect_from_server():
    """断开服务器连接"""
    try:
        global comm_service
        comm_service = CloudCommunicationService('auto')

        return jsonify({
            'success': True,
            'message': '已断开连接，切换到自动模式'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'断开连接失败: {str(e)}'
        }), 500
