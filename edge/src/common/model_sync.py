"""
边侧模型同步路由
处理与云端的模型同步
"""

from flask import Blueprint, request, jsonify, current_app, current_app
import json
import logging
from pathlib import Path
from src.common.model_manager import edge_model_manager

model_sync_bp = Blueprint('model_sync', __name__, url_prefix='/api/models')
logger = logging.getLogger(__name__)

@model_sync_bp.route('/sync_from_cloud', methods=['POST'])
def sync_from_cloud():
    """从云端同步模型到边侧"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        module = data.get('module')
        model_type = data.get('model_type')
        task_id = data.get('task_id')
        # 优先使用请求中的 cloud_url，否则从配置中获取
        cloud_url = data.get('cloud_url')
        if not cloud_url:
            cloud_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
        
        if not all([module, model_type, task_id]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # 使用模型管理器下载模型
        success = edge_model_manager.download_trained_model(
            module=module,
            model_type=model_type, 
            task_id=task_id.replace('task_', ''),  # 移除task_前缀
            cloud_url=cloud_url
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Model {task_id} downloaded successfully',
                'local_path': str(edge_model_manager.models_dir / module / model_type / task_id)
            })
        else:
            return jsonify({
                'success': False, 
                'error': 'Failed to download model from cloud'
            }), 500
            
    except Exception as e:
        logger.error(f"Model sync failed: {e}")
        return jsonify({'error': str(e)}), 500


@model_sync_bp.route('/list_local', methods=['GET'])
def list_local_models():
    """列出边侧本地模型"""
    try:
        module = request.args.get('module')
        models = edge_model_manager.list_local_models(module=module)
        
        return jsonify({
            'success': True,
            'models': models,
            'total_count': len(models)
        })
        
    except Exception as e:
        logger.error(f"Failed to list local models: {e}")
        return jsonify({'error': str(e)}), 500


@model_sync_bp.route('/cleanup', methods=['POST'])
def cleanup_models():
    """清理旧模型"""
    try:
        data = request.get_json() or {}
        module = data.get('module')
        keep_recent = data.get('keep_recent', 5)
        
        edge_model_manager.cleanup_old_models(module=module, keep_recent=keep_recent)
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up old models, keeping {keep_recent} most recent'
        })
        
    except Exception as e:
        logger.error(f"Model cleanup failed: {e}")
        return jsonify({'error': str(e)}), 500


@model_sync_bp.route('/check_availability', methods=['POST'])
def check_model_availability():
    """检查指定模型是否在边侧可用"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        module = data.get('module')
        model_type = data.get('model_type')
        task_id = data.get('task_id')
        
        if not all([module, model_type, task_id]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # 检查模型是否存在
        model_dir = edge_model_manager.models_dir / module / model_type / f"task_{task_id}"
        required_files = ['config.json', 'model.ckpt']
        
        availability = {
            'available': model_dir.exists(),
            'path': str(model_dir),
            'files': {}
        }
        
        if model_dir.exists():
            for filename in ['config.json', 'model.ckpt', 'scaler.pkl', 'scaler.pkl.npz', 
                           'threshold.json', 'threshold.npz', 'download_record.json']:
                file_path = model_dir / filename
                availability['files'][filename] = file_path.exists()
                
            # 检查是否有必要文件
            availability['ready_for_inference'] = all(
                availability['files'].get(f, False) for f in required_files
            )
            
        return jsonify({
            'success': True,
            'availability': availability
        })
        
    except Exception as e:
        logger.error(f"Failed to check model availability: {e}")
        return jsonify({'error': str(e)}), 500