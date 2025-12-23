"""
健康检查API模块
提供系统状态检查和服务监控功能
"""

from flask import Blueprint, jsonify
import json
import logging
import psutil
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# 创建蓝图
health_bp = Blueprint('health', __name__, url_prefix='/api/health')

def _check_pytorch_status():
    """检查PyTorch状态"""
    try:
        import torch
        return {
            'available': True,
            'version': torch.__version__ if hasattr(torch, '__version__') else 'unknown',
            'backend': 'PyTorch',
            'cuda_available': torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False,
            'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') and torch.cuda.is_available() else None
        }
    except ImportError:
        return {
            'available': False,
            'error': 'PyTorch not installed',
            'backend': None
        }

def _check_disk_usage():
    """检查磁盘使用情况"""
    try:
        # 检查模型目录的磁盘使用
        models_dir = Path('models')
        if models_dir.exists():
            disk_usage = psutil.disk_usage(str(models_dir))
            total_gb = disk_usage.total / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            
            return {
                'total_gb': round(total_gb, 2),
                'used_gb': round(used_gb, 2),
                'free_gb': round(free_gb, 2),
                'usage_percent': round((used_gb / total_gb) * 100, 2)
            }
        else:
            return {'error': 'Models directory not found'}
    except Exception as e:
        return {'error': str(e)}

def _check_models_directory():
    """检查模型目录状态"""
    try:
        models_dir = Path('models')
        status = {
            'exists': models_dir.exists(),
            'modules': {}
        }
        
        if models_dir.exists():
            for module in ['anomaly_detection', 'fault_diagnosis', 'rul_prediction']:
                module_dir = models_dir / module
                module_status = {
                    'exists': module_dir.exists(),
                    'model_types': [],
                    'total_tasks': 0
                }
                
                if module_dir.exists():
                    for item in module_dir.iterdir():
                        if item.is_dir():
                            model_type = item.name
                            task_count = len([t for t in item.iterdir() if t.is_dir() and t.name.startswith('task_')])
                            
                            module_status['model_types'].append({
                                'name': model_type,
                                'task_count': task_count
                            })
                            module_status['total_tasks'] += task_count
                
                status['modules'][module] = module_status
        
        return status
    except Exception as e:
        return {'error': str(e)}

def _get_system_info():
    """获取系统信息"""
    try:
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'percent': psutil.virtual_memory().percent
            },
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    except Exception as e:
        return {'error': str(e)}

# API路由
@health_bp.route('/', methods=['GET'])
@health_bp.route('/status', methods=['GET'])
def health_status():
    """获取系统健康状态"""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'service': 'Cloud Training Service',
            'version': '1.0.0'
        }
        
        # 检查PyTorch状态
        pytorch_status = _check_pytorch_status()
        status['pytorch'] = pytorch_status
        
        # 检查模型目录状态
        models_status = _check_models_directory()
        status['models_directory'] = models_status
        
        # 获取系统信息
        system_info = _get_system_info()
        status['system'] = system_info
        
        # 检查磁盘使用
        disk_info = _check_disk_usage()
        status['disk'] = disk_info
        
        # 根据各个组件状态确定整体状态
        if not pytorch_status['available']:
            status['status'] = 'degraded'
            status['warnings'] = ['PyTorch not available']
        
        if 'error' in models_status:
            status['status'] = 'unhealthy'
            status['errors'] = [f"Models directory error: {models_status['error']}"]
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@health_bp.route('/models', methods=['GET'])
def models_health():
    """获取模型相关的健康信息"""
    try:
        models_info = _check_models_directory()
        
        # 统计信息
        summary = {
            'total_modules': len(models_info.get('modules', {})),
            'total_tasks': sum(
                module.get('total_tasks', 0) 
                for module in models_info.get('modules', {}).values()
            ),
            'modules_detail': models_info.get('modules', {})
        }
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'models_summary': summary,
            'disk_usage': _check_disk_usage()
        })
        
    except Exception as e:
        logger.error(f"Models health check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@health_bp.route('/system', methods=['GET'])
def system_health():
    """获取系统资源信息"""
    try:
        system_info = _get_system_info()
        pytorch_info = _check_pytorch_status()
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'system_info': system_info,
            'pytorch_info': pytorch_info
        })
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@health_bp.route('/ready', methods=['GET'])
def readiness_check():
    """就绪检查 - 用于K8s等容器编排"""
    try:
        pytorch_status = _check_pytorch_status()
        models_status = _check_models_directory()
        
        # 简单的就绪检查
        ready = (
            pytorch_status['available'] and 
            models_status.get('exists', False)
        )
        
        if ready:
            return jsonify({'status': 'ready'}), 200
        else:
            return jsonify({
                'status': 'not_ready',
                'reasons': [
                    'PyTorch not available' if not pytorch_status['available'] else None,
                    'Models directory not found' if not models_status.get('exists') else None
                ]
            }), 503
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@health_bp.route('/live', methods=['GET'])
def liveness_check():
    """存活检查 - 用于K8s等容器编排"""
    try:
        # 简单的存活检查，只要服务能响应就认为是存活的
        return jsonify({
            'status': 'alive',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500