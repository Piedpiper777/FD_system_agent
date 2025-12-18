"""
统一模型管理模块
整合异常检测、故障诊断、RUL预测的模型管理功能
"""

from flask import Blueprint, render_template, jsonify, request, current_app
from pathlib import Path
import os
import json
from datetime import datetime

# 创建蓝图
model_management_bp = Blueprint('model_management', __name__, url_prefix='/model_management')

# 模块配置
MODULES = {
    'anomaly_detection': {
        'name': '异常检测',
        'icon': 'chart-line',
        'color': 'primary'
    },
    'fault_diagnosis': {
        'name': '故障诊断',
        'icon': 'tools',
        'color': 'warning'
    },
    'rul_prediction': {
        'name': 'RUL预测',
        'icon': 'clock',
        'color': 'info'
    }
}


@model_management_bp.route('/')
def index():
    """模型管理主页"""
    cloud_base_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
    return render_template('model_management/index.html', 
                         modules=MODULES,
                         cloud_base_url=cloud_base_url)


@model_management_bp.route('/api/models', methods=['GET'])
def list_all_models():
    """获取所有模块的模型列表"""
    try:
        module_filter = request.args.get('module')  # 可选：按模块筛选
        
        all_models = []
        
        # 遍历所有模块
        modules_to_search = [module_filter] if module_filter else list(MODULES.keys())
        
        for module in modules_to_search:
            if module not in MODULES:
                continue
                
            # 获取该模块的模型列表
            module_models = _get_module_models(module)
            
            # 为每个模型添加模块信息
            for model in module_models:
                model['module'] = module
                model['module_name'] = MODULES[module]['name']
            
            all_models.extend(module_models)
        
        # 按创建时间降序排列
        all_models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'models': all_models,
            'total_count': len(all_models),
            'modules': {k: v['name'] for k, v in MODULES.items()}
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _get_module_models(module):
    """获取指定模块的模型列表"""
    models = []
    
    # 获取edge目录
    edge_dir = Path(__file__).resolve().parents[3]
    models_dir = edge_dir / 'models' / module
    
    if not models_dir.exists():
        return models
    
    # 遍历模型类型目录
    for model_type_dir in models_dir.iterdir():
        if not model_type_dir.is_dir():
            continue
        
        # 遍历任务目录
        for task_dir in model_type_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            task_id = task_dir.name
            
            # 检查必要文件
            config_path = task_dir / 'config.json'
            if not config_path.exists():
                # 故障诊断可能使用 model_config.json
                config_path = task_dir / 'model_config.json'
                if not config_path.exists():
                    continue
            
            try:
                # 加载配置
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 查找模型文件
                model_files = [f for f in task_dir.iterdir() 
                             if f.is_file() and f.suffix in ('.ckpt', '.pth', '.h5')]
                
                if not model_files:
                    continue
                
                model_file = model_files[0]
                model_stat = model_file.stat()
                
                # 构建模型信息
                model_info = {
                    'task_id': task_id,
                    'model_type': model_type_dir.name,
                    'filename': model_file.name,
                    'size': model_stat.st_size,
                    'created_at': datetime.fromtimestamp(model_stat.st_ctime).isoformat(),
                    'modified_at': datetime.fromtimestamp(model_stat.st_mtime).isoformat(),
                    'config': config,
                    'path': str(task_dir)
                }
                
                # 异常检测特有：检查阈值文件
                if module == 'anomaly_detection':
                    threshold_path = task_dir / 'threshold.json'
                    if threshold_path.exists():
                        try:
                            with open(threshold_path, 'r', encoding='utf-8') as f:
                                threshold_info = json.load(f)
                            model_info['threshold'] = threshold_info
                        except:
                            pass
                
                models.append(model_info)
                
            except Exception as e:
                print(f"读取模型配置失败 {task_dir}: {e}")
                continue
    
    return models


@model_management_bp.route('/api/models/<module>/<task_id>', methods=['DELETE'])
def delete_model(module, task_id):
    """删除指定模块的模型"""
    try:
        import shutil
        
        if module not in MODULES:
            return jsonify({
                'success': False,
                'error': f'无效的模块: {module}'
            }), 400
        
        edge_dir = Path(__file__).resolve().parents[3]
        models_dir = edge_dir / 'models' / module
        
        # 在所有模型类型子目录中查找
        task_dir = None
        if models_dir.exists():
            for model_type_dir in models_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                potential_task_dir = model_type_dir / task_id
                if potential_task_dir.exists():
                    task_dir = potential_task_dir
                    break
        
        if not task_dir or not task_dir.exists():
            return jsonify({
                'success': False,
                'error': f'模型 {task_id} 不存在'
            }), 404
        
        # 删除模型目录
        shutil.rmtree(task_dir)
        
        return jsonify({
            'success': True,
            'message': f'模型 {task_id} 已成功删除'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'删除失败: {str(e)}'
        }), 500


@model_management_bp.route('/api/models/<module>/<task_id>/info', methods=['GET'])
def get_model_info(module, task_id):
    """获取模型详细信息"""
    try:
        if module not in MODULES:
            return jsonify({
                'success': False,
                'error': f'无效的模块: {module}'
            }), 400
        
        edge_dir = Path(__file__).resolve().parents[3]
        models_dir = edge_dir / 'models' / module
        
        # 查找模型目录
        task_dir = None
        if models_dir.exists():
            for model_type_dir in models_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                potential_task_dir = model_type_dir / task_id
                if potential_task_dir.exists():
                    task_dir = potential_task_dir
                    break
        
        if not task_dir or not task_dir.exists():
            return jsonify({
                'success': False,
                'error': f'模型 {task_id} 不存在'
            }), 404
        
        # 收集模型信息
        model_info = {
            'task_id': task_id,
            'module': module,
            'module_name': MODULES[module]['name'],
            'model_type': task_dir.parent.name if task_dir.parent else 'unknown',
            'config': {},
            'files': {},
            'path': str(task_dir)
        }
        
        # 读取配置文件
        config_path = task_dir / 'config.json'
        if not config_path.exists():
            config_path = task_dir / 'model_config.json'
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                model_info['config'] = json.load(f)
        
        # 收集文件信息
        for file_path in task_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                model_info['files'][file_path.name] = {
                    'size': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取模型信息失败: {str(e)}'
        }), 500

