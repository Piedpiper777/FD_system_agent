"""
故障诊断模型管理路由
"""

from flask import Blueprint, request, jsonify, current_app
import os
import json
from datetime import datetime
from pathlib import Path

fd_models_bp = Blueprint('fd_models', __name__, url_prefix='/fault_diagnosis')


@fd_models_bp.route('/api/models', methods=['GET'])
def list_models():
    """获取故障诊断模型列表"""
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
    """获取本地故障诊断模型列表"""
    models = []
    
    # 获取 edge/models/fault_diagnosis 目录
    edge_dir = Path(__file__).resolve().parents[4]  # 回到edge目录
    models_dir = edge_dir / 'models' / 'fault_diagnosis'
    
    if not models_dir.exists():
        return models
    
    # 定义模型类型子目录
    model_type_dirs = ['cnn_1d', 'lstm', 'resnet_1d']
    
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
    
    # 向后兼容：也检查直接在 fault_diagnosis 下的模型（旧格式）
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
        'size': 0,
        'created_at': None,
        'modified_at': None
    }
    
    # 读取 model_config.json
    config_file = task_dir / 'model_config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                model_info['config'] = config
                model_info['model_type'] = config.get('model_type', 'unknown')
            model_info['files']['config'] = True
        except Exception as e:
            print(f"读取配置文件失败 {config_file}: {e}")
    
    # 读取 training_metrics.json
    metrics_file = task_dir / 'training_metrics.json'
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                model_info['metrics'] = metrics
            model_info['files']['metrics'] = True
        except Exception as e:
            print(f"读取指标文件失败 {metrics_file}: {e}")
    
    # 读取评估结果文件（*_evaluation_results.json）
    for eval_file in task_dir.glob('*_evaluation_results.json'):
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                evaluation = json.load(f)
                model_info['evaluation'] = evaluation
            model_info['files']['evaluation'] = True
            break  # 只读取第一个评估结果文件
        except Exception as e:
            print(f"读取评估结果文件失败 {eval_file}: {e}")
    
    # 检查模型文件
    model_ckpt = task_dir / 'model.ckpt'
    if model_ckpt.exists():
        model_info['files']['model'] = True
        model_info['size'] += model_ckpt.stat().st_size
    
    # 计算总大小和获取时间戳
    total_size = 0
    earliest_time = None
    latest_time = None
    
    for file in task_dir.iterdir():
        if file.is_file():
            stat = file.stat()
            total_size += stat.st_size
            
            file_time = datetime.fromtimestamp(stat.st_mtime)
            if earliest_time is None or file_time < earliest_time:
                earliest_time = file_time
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
    
    model_info['size'] = total_size
    if earliest_time:
        model_info['created_at'] = earliest_time.isoformat()
    if latest_time:
        model_info['modified_at'] = latest_time.isoformat()
    
    return model_info


@fd_models_bp.route('/api/models/<task_id>', methods=['DELETE'])
def delete_model(task_id):
    """删除本地模型"""
    try:
        import shutil
        
        edge_dir = Path(__file__).resolve().parents[4]
        models_dir = edge_dir / 'models' / 'fault_diagnosis'
        
        # 在所有模型类型子目录中查找
        task_dir = None
        for model_type_dir in ['cnn_1d', 'lstm', 'resnet_1d']:
            potential_dir = models_dir / model_type_dir / task_id
            if potential_dir.exists():
                task_dir = potential_dir
                break
        
        # 向后兼容：直接在 fault_diagnosis 下查找
        if task_dir is None:
            direct_dir = models_dir / task_id
            if direct_dir.exists():
                task_dir = direct_dir
        
        if task_dir is None or not task_dir.exists():
            return jsonify({
                'success': False,
                'error': f'模型 {task_id} 不存在'
            }), 404
        
        # 删除整个目录
        shutil.rmtree(task_dir)
        
        return jsonify({
            'success': True,
            'message': f'模型 {task_id} 已删除'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'删除失败: {str(e)}'}), 500


@fd_models_bp.route('/api/models/<task_id>/info', methods=['GET'])
def get_model_info(task_id):
    """获取模型详细信息"""
    try:
        edge_dir = Path(__file__).resolve().parents[4]
        models_dir = edge_dir / 'models' / 'fault_diagnosis'
        
        # 在所有模型类型子目录中查找
        task_dir = None
        for model_type_dir in ['cnn_1d', 'lstm', 'resnet_1d']:
            potential_dir = models_dir / model_type_dir / task_id
            if potential_dir.exists():
                task_dir = potential_dir
                break
        
        # 向后兼容：直接在 fault_diagnosis 下查找
        if task_dir is None:
            direct_dir = models_dir / task_id
            if direct_dir.exists():
                task_dir = direct_dir
        
        if task_dir is None or not task_dir.exists():
            return jsonify({
                'success': False,
                'error': f'模型 {task_id} 不存在'
            }), 404
        
        model_info = {
            'task_id': task_id,
            'model_type': 'unknown',
            'config': {},
            'files': {},
            'evaluation': None,
            'created_at': None
        }
        
        # 读取配置文件
        config_file = task_dir / 'model_config.json'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                model_info['config'] = json.load(f)
                model_info['model_type'] = model_info['config'].get('model_type', 'unknown')
        
        # 读取评估结果文件（*_evaluation_results.json）
        for eval_file in task_dir.glob('*_evaluation_results.json'):
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    model_info['evaluation'] = json.load(f)
                break  # 只读取第一个评估结果文件
            except Exception as e:
                print(f"读取评估结果文件失败 {eval_file}: {e}")
        
        # 获取所有文件信息
        files_info = {}
        for file in task_dir.iterdir():
            if file.is_file():
                stat = file.stat()
                files_info[file.name] = {
                    'size': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
        
        model_info['files'] = files_info
        
        # 获取创建时间（使用最早的文件时间）
        if files_info:
            earliest = min(f['created_at'] for f in files_info.values())
            model_info['created_at'] = earliest
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取模型信息失败: {str(e)}'
        }), 500

