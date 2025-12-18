"""
训练状态查询路由
转发到云端的训练状态查询
"""

from flask import Blueprint, jsonify, current_app
import requests

training_bp = Blueprint('training', __name__, url_prefix='/api')


@training_bp.route('/training/<task_id>', methods=['GET'])
def get_training_status(task_id):
    """获取训练状态"""
    try:
        # 使用统一的配置项 CLOUD_BASE_URL
        cloud_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
        response = requests.get(f'{cloud_url}/api/training/{task_id}')
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500