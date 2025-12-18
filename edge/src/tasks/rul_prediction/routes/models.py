"""
RUL预测模型管理路由
"""

from flask import Blueprint, request, jsonify, current_app
import os
import json
from datetime import datetime
from pathlib import Path

rup_models_bp = Blueprint('rup_models', __name__, url_prefix='/rul_prediction')
SUPPORTED_MODEL_TYPES = [
    'bilstm_gru_regressor',
    'cnn_1d_regressor',
    'transformer_encoder_regressor',
]


@rup_models_bp.route('/api/models', methods=['GET'])
def list_models():
    """获取RUL预测模型列表"""
    try:
        models = _get_local_models()
        
        return jsonify({
            'success': True,
            'models': models,
            'total_count': len(models)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _get_local_models():
    """获取本地RUL预测模型列表"""
    models = []
    
    # 获取 edge/models/rul_prediction 目录
    edge_dir = Path(__file__).resolve().parents[4]  # 回到edge目录
    models_dir = edge_dir / 'models' / 'rul_prediction'
    
    if not models_dir.exists():
        return models
    
    # 定义模型类型子目录
    model_type_dirs = SUPPORTED_MODEL_TYPES
    
    # 遍历模型类型子目录
    for model_type in model_type_dirs:
        type_dir = models_dir / model_type
        if not type_dir.exists():
            continue
        
        # 遍历该类型下的所有任务目录
        for task_dir in type_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            task_id = task_dir.name
            
            # 跳过非任务目录（如 __pycache__）
            if task_id.startswith('_') or task_id.startswith('.'):
                continue
            
            try:
                model_info = _parse_model_info(task_dir, task_id, model_type)
                if model_info:
                    models.append(model_info)
            except Exception as e:
                print(f"读取模型 {task_id} 失败: {e}")
                continue
    
    # 向后兼容：也检查直接在 rul_prediction 下的模型（旧格式）
    for task_dir in models_dir.iterdir():
        if not task_dir.is_dir():
            continue
        
        task_id = task_dir.name
        
        # 跳过模型类型子目录和特殊目录
        if task_id in model_type_dirs or task_id.startswith('_') or task_id.startswith('.'):
            continue
        
        try:
            model_info = _parse_model_info(task_dir, task_id, 'unknown')
            if model_info:
                models.append(model_info)
        except Exception as e:
            print(f"读取模型 {task_id} 失败: {e}")
            continue
    
    # 按创建时间降序排列
    models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return models


def _parse_model_info(task_dir: Path, task_id: str, model_type_dir: str = 'unknown') -> dict:
    """解析单个模型的信息"""
    model_info = {
        'task_id': task_id,
        'model_type': 'unknown',
        'model_type_dir': model_type_dir,  # 模型所在的子目录
        'config': {},
        'files': {},
        'created_at': None,
        'modified_at': None,
    }
    
    # 读取模型配置文件
    config_path = task_dir / 'model_config.json'
    if not config_path.exists():
        # 如果没有配置文件，跳过
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_info['model_type'] = config.get('model_type', 'bilstm_gru_regressor')
        model_info['config'] = config
        model_info['created_at'] = config.get('created_at')
        
        # 训练指标
        if 'final_train_loss' in config:
            model_info['metrics'] = {
                'train_loss': config.get('final_train_loss'),
                'val_loss': config.get('final_val_loss'),
                'train_rmse': config.get('final_train_rmse'),
                'val_rmse': config.get('final_val_rmse'),
                'train_mae': config.get('final_train_mae'),
                'val_mae': config.get('final_val_mae'),
                'train_r2': config.get('final_train_r2'),
                'val_r2': config.get('final_val_r2'),
            }
        
    except Exception as e:
        print(f"读取模型配置失败 {config_path}: {e}")
        return None
    
    # 检查文件存在性
    model_info['files'] = {
        'has_model': (task_dir / 'model.ckpt').exists(),
        'has_config': config_path.exists(),
        'has_scaler': (task_dir / 'scaler.pkl').exists(),
        'has_best_model': (task_dir / 'best_model.ckpt').exists(),
        'has_final_model': (task_dir / 'final_model.ckpt').exists(),
    }
    
    # 获取文件大小和修改时间
    model_path = task_dir / 'model.ckpt'
    if model_path.exists():
        stat = model_path.stat()
        model_info['size'] = stat.st_size
        model_info['modified_at'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
    else:
        # 如果没有model.ckpt，尝试使用其他模型文件
        for alt_name in ['best_model.ckpt', 'final_model.ckpt']:
            alt_path = task_dir / alt_name
            if alt_path.exists():
                stat = alt_path.stat()
                model_info['size'] = stat.st_size
                model_info['modified_at'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                break
    
    # 路径信息
    model_info['path'] = str(task_dir)
    
    return model_info


@rup_models_bp.route('/api/models/<task_id>', methods=['DELETE'])
def delete_model(task_id):
    """删除指定的模型"""
    try:
        edge_dir = Path(__file__).resolve().parents[4]
        models_dir = edge_dir / 'models' / 'rul_prediction'
        
        # 在所有模型类型子目录中查找
        model_dir = None
        for model_type_dir in SUPPORTED_MODEL_TYPES:
            potential_dir = models_dir / model_type_dir / task_id
            if potential_dir.exists():
                model_dir = potential_dir
                break
        
        # 如果在子目录中未找到，尝试直接查找（向后兼容）
        if model_dir is None:
            direct_dir = models_dir / task_id
            if direct_dir.exists():
                model_dir = direct_dir
        
        if model_dir is None or not model_dir.exists():
            return jsonify({
                'success': False,
                'error': f'模型 {task_id} 不存在'
            }), 404
        
        # 删除模型目录
        import shutil
        shutil.rmtree(model_dir)
        
        return jsonify({
            'success': True,
            'message': f'模型 {task_id} 已删除'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'删除模型失败: {str(e)}'
        }), 500


@rup_models_bp.route('/api/models/<task_id>/info', methods=['GET'])
def get_model_info(task_id):
    """获取指定模型的详细信息"""
    try:
        edge_dir = Path(__file__).resolve().parents[4]
        models_dir = edge_dir / 'models' / 'rul_prediction'
        
        # 在所有模型类型子目录中查找
        model_dir = None
        for model_type_dir in SUPPORTED_MODEL_TYPES:
            potential_dir = models_dir / model_type_dir / task_id
            if potential_dir.exists():
                model_dir = potential_dir
                break
        
        # 如果在子目录中未找到，尝试直接查找（向后兼容）
        if model_dir is None:
            direct_dir = models_dir / task_id
            if direct_dir.exists():
                model_dir = direct_dir
        
        if model_dir is None or not model_dir.exists():
            return jsonify({
                'success': False,
                'error': f'模型 {task_id} 不存在'
            }), 404
        
        # 解析模型信息
        model_info = _parse_model_info(model_dir, task_id, model_dir.parent.name)
        
        if not model_info:
            return jsonify({
                'success': False,
                'error': '无法读取模型信息'
            }), 500
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取模型信息失败: {str(e)}'
        }), 500

