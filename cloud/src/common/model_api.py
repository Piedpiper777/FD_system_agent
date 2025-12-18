"""
模型管理API模块
处理模型下载、模型信息查询等共享功能
"""

from flask import Blueprint, request, jsonify, send_file
import json
import logging
import shutil
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# 创建蓝图
model_management_bp = Blueprint('model_management', __name__, url_prefix='/api/model_management')

def _find_model_artifacts(task_id: str, module: str = None) -> dict:
    """查找模型训练产物"""
    # 如果提供了模块，优先搜索该模块
    search_modules = [module] if module else ['anomaly_detection', 'fault_diagnosis', 'rul_prediction']
    
    for mod in search_modules:
        models_dir = Path(f'models/{mod}')
        if not models_dir.exists():
            continue
            
        # 搜索所有模型类型目录
        for model_type_dir in models_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
                
            # 查找指定的task_id
            task_dir = model_type_dir / task_id
            if task_dir.exists() and task_dir.is_dir():
                # 找到匹配的任务目录，收集模型产物
                artifacts = {
                    'module': mod,
                    'model_type': model_type_dir.name,
                    'task_id': task_id,
                    'model_dir': task_dir
                }
                
                # 查找模型文件
                for ext in ['.ckpt', '.pth', '.h5', '.pkl']:
                    model_files = list(task_dir.glob(f'*{ext}'))
                    if model_files:
                        artifacts['model_path'] = model_files[0]
                        break
                
                # 查找其他相关文件
                scaler_path = task_dir / 'scaler.pkl'
                if scaler_path.exists():
                    artifacts['scaler_path'] = scaler_path
                
                threshold_path = task_dir / 'threshold.json'
                if threshold_path.exists():
                    artifacts['threshold_path'] = threshold_path
                
                config_path = task_dir / 'config.json'
                if config_path.exists():
                    artifacts['config_path'] = config_path
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        artifacts['config'] = config
                        
                        # 提取有用的配置信息
                        artifacts['sequence_length'] = config.get('sequence_length')
                        
                        # 如果有阈值信息，提取出来
                        if 'threshold_value' in config:
                            artifacts['threshold_value'] = config['threshold_value']
                            artifacts['threshold_meta'] = config.get('threshold_meta', {})
                            
                    except Exception as e:
                        logger.warning(f"Failed to load config from {config_path}: {e}")
                
                return artifacts
    
    return None

def _create_model_download_package(artifacts: dict, temp_dir: Path) -> Path:
    """创建模型下载包"""
    # 创建下载包目录结构
    package_dir = temp_dir / f"{artifacts['task_id']}_model_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_package = []
    
    # 复制模型文件
    if 'model_path' in artifacts:
        model_file = package_dir / artifacts['model_path'].name
        shutil.copy2(artifacts['model_path'], model_file)
        files_to_package.append(('model', artifacts['model_path'].name))
    
    # 复制标准化器
    if 'scaler_path' in artifacts:
        scaler_file = package_dir / artifacts['scaler_path'].name
        shutil.copy2(artifacts['scaler_path'], scaler_file)
        files_to_package.append(('scaler', artifacts['scaler_path'].name))
    
    # 复制阈值文件
    if 'threshold_path' in artifacts:
        threshold_file = package_dir / artifacts['threshold_path'].name
        shutil.copy2(artifacts['threshold_path'], threshold_file)
        files_to_package.append(('threshold', artifacts['threshold_path'].name))
    
    # 创建包信息文件
    package_info = {
        'package_created_at': datetime.now().isoformat(),
        'source_module': artifacts['module'],
        'source_model_type': artifacts['model_type'],
        'source_task_id': artifacts['task_id'],
        'files': dict(files_to_package)
    }
    
    # 包含原始配置信息
    if 'config' in artifacts:
        package_info['original_config'] = artifacts['config']
    
    info_file = package_dir / 'package_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(package_info, f, indent=2, ensure_ascii=False)
    
    # 创建ZIP文件
    zip_path = temp_dir / f"{artifacts['task_id']}_model.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.iterdir():
            if file_path.is_file():
                zipf.write(file_path, file_path.name)
    
    return zip_path

# API路由
@model_management_bp.route('/download/<task_id>', methods=['GET'])
def download_model(task_id):
    """下载训练好的模型"""
    module = request.args.get('module')
    
    try:
        # 查找模型产物
        artifacts = _find_model_artifacts(task_id, module)
        if not artifacts:
            return jsonify({
                'success': False,
                'error': f'Model for task {task_id} not found'
            }), 404
        
        if 'model_path' not in artifacts or not artifacts['model_path'].exists():
            return jsonify({
                'success': False,
                'error': f'Model file not found for task {task_id}'
            }), 404
        
        # 创建临时目录打包模型
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = _create_model_download_package(artifacts, temp_path)
            
            # 发送文件
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f"{task_id}_model.zip",
                mimetype='application/zip'
            )
            
    except Exception as e:
        logger.error(f"Failed to download model {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@model_management_bp.route('/info/<task_id>', methods=['GET'])
def get_model_info(task_id):
    """获取模型信息"""
    module = request.args.get('module')
    
    try:
        artifacts = _find_model_artifacts(task_id, module)
        if not artifacts:
            return jsonify({
                'success': False,
                'error': f'Model for task {task_id} not found'
            }), 404
        
        # 构建模型信息响应
        model_info = {
            'task_id': task_id,
            'module': artifacts['module'],
            'model_type': artifacts['model_type'],
            'model_dir': str(artifacts['model_dir'])
        }
        
        # 添加文件信息
        files_info = {}
        if 'model_path' in artifacts:
            model_path = artifacts['model_path']
            files_info['model'] = {
                'filename': model_path.name,
                'size': model_path.stat().st_size,
                'exists': model_path.exists()
            }
        
        if 'scaler_path' in artifacts:
            scaler_path = artifacts['scaler_path']
            files_info['scaler'] = {
                'filename': scaler_path.name,
                'size': scaler_path.stat().st_size,
                'exists': scaler_path.exists()
            }
        
        if 'threshold_path' in artifacts:
            threshold_path = artifacts['threshold_path']
            files_info['threshold'] = {
                'filename': threshold_path.name,
                'size': threshold_path.stat().st_size,
                'exists': threshold_path.exists()
            }
        
        model_info['files'] = files_info
        
        # 添加配置信息（如果有）
        if 'config' in artifacts:
            model_info['config'] = artifacts['config']
        
        # 添加一些有用的元数据
        if 'sequence_length' in artifacts:
            model_info['sequence_length'] = artifacts['sequence_length']
        
        if 'threshold_value' in artifacts:
            model_info['threshold_value'] = artifacts['threshold_value']
            model_info['threshold_meta'] = artifacts.get('threshold_meta', {})
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
        
    except Exception as e:
        logger.error(f"Failed to get model info for {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@model_management_bp.route('/list', methods=['GET'])
def list_all_models():
    """列出所有可用的训练模型"""
    module_filter = request.args.get('module')
    
    try:
        all_models = []
        
        # 搜索的模块列表
        search_modules = [module_filter] if module_filter else ['anomaly_detection', 'fault_diagnosis', 'rul_prediction']
        
        for module in search_modules:
            models_dir = Path(f'models/{module}')
            if not models_dir.exists():
                continue
                
            for model_type_dir in models_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                    
                for task_dir in model_type_dir.iterdir():
                    if not task_dir.is_dir() or not task_dir.name.startswith('task_'):
                        continue
                        
                    config_path = task_dir / 'config.json'
                    if not config_path.exists():
                        continue
                        
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        # 检查模型文件是否存在
                        model_exists = False
                        for ext in ['.ckpt', '.pth', '.h5', '.pkl']:
                            if list(task_dir.glob(f'*{ext}')):
                                model_exists = True
                                break
                        
                        model_info = {
                            'task_id': task_dir.name,
                            'module': module,
                            'model_type': model_type_dir.name,
                            'created_at': config.get('created_at'),
                            'training_status': config.get('training_status', 'unknown'),
                            'model_exists': model_exists,
                            'has_scaler': (task_dir / 'scaler.pkl').exists(),
                            'has_threshold': (task_dir / 'threshold.json').exists()
                        }
                        
                        # 添加一些训练配置信息
                        if 'sequence_length' in config:
                            model_info['sequence_length'] = config['sequence_length']
                        if 'batch_size' in config:
                            model_info['batch_size'] = config['batch_size']
                        if 'epochs' in config:
                            model_info['epochs'] = config['epochs']
                        
                        all_models.append(model_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load config for {task_dir}: {e}")
                        continue
        
        # 按创建时间降序排列
        all_models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'models': all_models,
            'total_count': len(all_models),
            'modules': list(set(m['module'] for m in all_models))
        })
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@model_management_bp.route('/cleanup', methods=['POST'])
def cleanup_models():
    """清理旧的或无效的模型文件"""
    data = request.get_json() or {}
    
    # 可以根据需要添加清理逻辑
    # 比如删除超过一定时间的模型，或者删除没有配置文件的模型目录等
    
    return jsonify({
        'success': True,
        'message': 'Model cleanup completed',
        'cleaned_items': 0
    })