"""
异常检测推理路由
"""

from flask import Blueprint, request, jsonify, render_template, current_app, flash
import os
import json
import shutil
import requests
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from ..services.inferencer import LocalAnomalyDetector
from src.common.utils.file_utils import allowed_file
from src.common.utils.path_utils import to_relative_path, resolve_edge_path

ad_inference_bp = Blueprint('ad_inference', __name__, url_prefix='/anomaly_detection')

# 推理服务：使用本地 LocalAnomalyDetector（完全在Edge端执行，不依赖Cloud）
logger = logging.getLogger(__name__)

# 本地目录配置 (Edge端独立运行，不依赖Cloud)
EDGE_DIR = Path(__file__).resolve().parents[4]
EDGE_MODELS_ROOT = EDGE_DIR / 'models' / 'anomaly_detection'
MODEL_OUTPUT_DIRS = {
    'lstm_predictor': 'lstm_prediction',
    'lstm_autoencoder': 'lstm_autoencoder',
    'cnn_1d_autoencoder': 'cnn_1d_autoencoder'
}



@ad_inference_bp.route('/api/inference_history', methods=['GET'])
def get_inference_history():
    """获取本地推理历史列表"""
    try:
        results_base_dir = Path(__file__).resolve().parents[4] / 'inference_tasks' / 'anomaly_detection'
        
        if not results_base_dir.exists():
            return jsonify({
                'success': True,
                'tasks': [],
                'total': 0
            })
        
        inference_tasks = []
        
        # 遍历所有推理结果目录
        for inference_dir in sorted(results_base_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not inference_dir.is_dir():
                continue
            
            result_file = inference_dir / 'inference_result.json'
            if not result_file.exists():
                continue
            
            try:
                # 读取推理结果
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # 提取关键信息
                inference_id = result_data.get('inference_id', inference_dir.name)
                task_id = result_data.get('task_id', '')
                model_type = result_data.get('model_type', result_data.get('model_config', {}).get('model_type', 'unknown'))
                
                # 获取推理参数
                inference_params = result_data.get('inference_params', {})
                model_info = result_data.get('model_info', {})
                
                # 获取文件修改时间
                try:
                    modified_at = datetime.fromtimestamp(result_file.stat().st_mtime).isoformat()
                except Exception:
                    modified_at = result_data.get('created_at', '')
                
                inference_tasks.append({
                    'inference_id': inference_id,  # 使用JSON中的inference_id
                    'inference_dir_name': inference_dir.name,  # 同时保存目录名，用于查找
                    'task_id': task_id,
                    'model_type': model_type,
                    'model_name': f"{task_id} ({model_type})",
                    'sequence_length': inference_params.get('sequence_length', model_info.get('trained_sequence_length', 50)),
                    'batch_size': inference_params.get('batch_size', 32),
                    'data_file': inference_params.get('data_file', ''),
                    'remark': inference_params.get('remark', ''),
                    'total_samples': result_data.get('total_samples', 0),
                    'anomalies_detected': result_data.get('anomalies_detected', 0),
                    'anomaly_percentage': result_data.get('anomaly_percentage', 0),
                    'threshold_value': result_data.get('threshold_value'),
                    'created_at': result_data.get('created_at', modified_at),
                    'modified_at': modified_at,
                    'result_file': str(result_file),
                    'result_dir': str(inference_dir)
                })
                
            except Exception as e:
                logger.warning(f"读取推理结果失败 {result_file}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'tasks': inference_tasks,
            'total': len(inference_tasks)
        })
        
    except Exception as e:
        logger.error(f"获取推理历史失败: {e}")
        return jsonify({
            'success': False,
            'tasks': [],
            'error': str(e)
        }), 500

@ad_inference_bp.route('/api/inference_result/<inference_id>', methods=['GET'])
def get_inference_result(inference_id):
    """获取指定推理ID的完整推理结果"""
    try:
        results_base_dir = Path(__file__).resolve().parents[4] / 'inference_tasks' / 'anomaly_detection'
        
        if not results_base_dir.exists():
            return jsonify({
                'success': False,
                'error': f'推理结果目录不存在'
            }), 404
        
        # 查找推理结果目录
        inference_dir = None
        result_file = None
        
        # 方法1: 先尝试精确匹配目录名
        potential_dir = results_base_dir / inference_id
        if potential_dir.exists() and potential_dir.is_dir():
            potential_file = potential_dir / 'inference_result.json'
            if potential_file.exists():
                inference_dir = potential_dir
                result_file = potential_file
        
        # 方法2: 如果精确匹配失败，遍历所有目录，匹配目录名或JSON中的inference_id
        if not inference_dir:
            for dir_path in results_base_dir.iterdir():
                if not dir_path.is_dir():
                    continue
                
                # 检查目录名是否包含inference_id
                if inference_id in dir_path.name or dir_path.name == inference_id:
                    potential_file = dir_path / 'inference_result.json'
                    if potential_file.exists():
                        inference_dir = dir_path
                        result_file = potential_file
                        break
                
                # 检查JSON文件中的inference_id是否匹配
                potential_file = dir_path / 'inference_result.json'
                if potential_file.exists():
                    try:
                        with open(potential_file, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                        if result_data.get('inference_id') == inference_id:
                            inference_dir = dir_path
                            result_file = potential_file
                            break
                    except Exception:
                        continue
        
        if not inference_dir or not result_file:
            return jsonify({
                'success': False,
                'error': f'未找到推理结果: {inference_id}'
            }), 404
        
        # 读取推理结果
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        return jsonify({
            'success': True,
            'result': result_data
        })
        
    except Exception as e:
        logger.error(f"获取推理结果失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ad_inference_bp.route('/api/inference_tasks')
def api_inference_tasks():
    """获取推理任务列表的API端点（兼容旧接口，重定向到推理历史）"""
    return get_inference_history()


@ad_inference_bp.route('/inference')
def inference_page():
    """异常检测推理页面"""
    # 简化页面初始化，模型列表通过AJAX动态加载
    return render_template('anomaly_detection/inference.html')


@ad_inference_bp.route('/api/run_inference', methods=['POST'])
def run_inference():
    """执行异常检测推理 - 使用本地模型"""
    try:
        # 解析JSON数据（前端现在发送JSON而不是FormData）
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '无效的请求数据'
            }), 400
        
        task_id = data.get('task_id', '').strip()
        data_file_path = data.get('data_file_path', '').strip()
        sequence_length = data.get('sequence_length', 50)
        batch_size = data.get('batch_size', 32)
        remark = data.get('remark', '').strip()
        
        # 参数验证和转换
        try:
            sequence_length = int(sequence_length) if sequence_length else 50
            batch_size = int(batch_size) if batch_size else 32
        except ValueError:
            return jsonify({
                'success': False,
                'error': '序列长度和批次大小必须是有效的整数'
            }), 400
        
        # 验证task_id
        if not task_id:
            return jsonify({
                'success': False,
                'error': '请选择一个模型进行推理'
            }), 400
        
        # 验证数据文件路径
        if not data_file_path:
            return jsonify({
                'success': False,
                'error': '请选择数据文件'
            }), 400

        try:
            data_path = resolve_edge_path(data_file_path)
        except ValueError as exc:
            return jsonify({
                'success': False,
                'error': f'数据文件路径无效: {exc}'
            }), 400

        # 验证文件路径是否存在且可访问
        if not data_path.exists():
            return jsonify({
                'success': False,
                'error': f'数据文件不存在: {data_file_path}'
            }), 400
        
        if not data_path.is_file():
            return jsonify({
                'success': False,
                'error': f'指定的路径不是文件: {data_file_path}'
            }), 400
        
        # 验证文件扩展名
        if not allowed_file(data_path.name):
            return jsonify({
                'success': False,
                'error': '不支持的文件格式，请选择CSV文件'
            }), 400
        
        # 查找本地模型
        model_dir = _find_model_dir_by_task_id(task_id)
        if not model_dir:
            return jsonify({
                'success': False,
                'error': f'未找到模型 {task_id}，请确保模型已下载到本地'
            }), 404
        
        # 执行推理
        result = _run_inference_with_local_model(
            task_id=task_id,
            model_dir=model_dir,
            data_path=data_path,
            sequence_length=sequence_length,
            batch_size=batch_size,
            remark=remark
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"推理过程失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'推理过程出现错误: {str(e)}'
        }), 500


@ad_inference_bp.route('/api/data_files', methods=['GET'])
def get_data_files():
    """获取edge/data/processed/AnomalyDetection目录下的可用数据文件列表"""
    try:
        from pathlib import Path
        
        # 数据目录路径：专门使用异常检测的预处理数据目录
        edge_dir = Path(__file__).resolve().parents[4]  # 回到edge目录
        data_dir = edge_dir / 'data' / 'processed' / 'AnomalyDetection'
        
        available_files = []
        
        if data_dir.exists():
            # 查找所有CSV文件（不递归，只查找当前目录）
            for csv_file in data_dir.glob('*.csv'):
                try:
                    # 获取相对路径
                    relative_path = csv_file.relative_to(data_dir)
                    
                    # 获取文件信息
                    stat = csv_file.stat()
                    file_size = stat.st_size
                    
                    available_files.append({
                        'name': csv_file.name,
                        'path': str(csv_file),
                        'relative_path': str(relative_path),
                        'size': file_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                    
                except Exception as e:
                    logger.warning(f"处理文件 {csv_file} 时出错: {e}")
                    continue
        
        # 按修改时间倒序排序
        available_files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': available_files,
            'total': len(available_files)
        })
        
    except Exception as e:
        logger.error(f"获取数据文件列表失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'files': []
        }), 500


@ad_inference_bp.route('/api/available_models', methods=['GET'])
def get_available_models():
    """获取可用于推理的本地模型列表（仅本地模型，不连接云端）"""
    try:
        # 直接获取本地模型，不连接云端
        from pathlib import Path
        import json
        
        edge_dir = Path(__file__).resolve().parents[4]  # 回到edge目录
        models_dir = edge_dir / 'models' / 'anomaly_detection'
        
        inference_models = []
        
        if models_dir.exists():
            # 遍历所有模型类型目录
            for model_type in os.listdir(models_dir):
                model_type_dir = models_dir / model_type
                if not model_type_dir.is_dir():
                    continue
                
                # 遍历每个任务目录
                for task_id in os.listdir(model_type_dir):
                    task_dir = model_type_dir / task_id
                    if not task_dir.is_dir():
                        continue
                    
                    # 检查必要的文件是否存在
                    config_file = task_dir / 'config.json'
                    scaler_file = task_dir / 'scaler.pkl'
                    threshold_file = task_dir / 'threshold.json'
                    
                    # 至少需要配置文件
                    if not config_file.exists():
                        continue
                    
                    # 查找模型文件（支持多种格式，优先 .pth）
                    model_file = None
                    for ext in ['.pth', '.ckpt', '.h5']:
                        potential_file = task_dir / f'model{ext}'
                        if potential_file.exists():
                            model_file = potential_file
                            break
                    
                    # 如果没找到标准命名，查找任何模型文件
                    if not model_file:
                        model_files = [f for f in os.listdir(task_dir) 
                                     if f.endswith(('.pth', '.ckpt', '.h5'))]
                        if model_files:
                            model_file = task_dir / model_files[0]
                    
                    # 至少需要模型文件和配置文件
                    if not model_file or not model_file.exists():
                        continue
                    
                    # 读取配置
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                    except Exception:
                        config = {}
                    
                    # 读取阈值信息
                    threshold_value = None
                    if threshold_file.exists():
                        try:
                            with open(threshold_file, 'r', encoding='utf-8') as f:
                                threshold_data = json.load(f)
                                threshold_value = threshold_data.get('threshold_value')
                        except Exception:
                            pass
                    
                    # 获取文件修改时间
                    try:
                        modified_at = datetime.fromtimestamp(config_file.stat().st_mtime).isoformat()
                    except Exception:
                        modified_at = None
                    
                    inference_models.append({
                        'task_id': task_id,
                        'model_type': model_type,
                        'sequence_length': config.get('sequence_length', 50),
                        'hidden_units': config.get('hidden_units', 128),
                        'epochs': config.get('epochs'),
                        'created_at': config.get('created_at'),
                        'modified_at': modified_at or config.get('modified_at'),
                        'has_model': model_file.exists(),
                        'has_scaler': scaler_file.exists(),
                        'has_threshold': threshold_file.exists(),
                        'threshold_value': threshold_value,
                        'status': 'local'
                    })
        
        # 按修改时间倒序排序
        inference_models.sort(key=lambda x: x.get('modified_at') or x.get('created_at') or '', reverse=True)
        
        return jsonify({
            'success': True,
            'models': inference_models,
            'total': len(inference_models)
        })
        
    except Exception as e:
        logger.error(f"获取可用模型失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'models': []
        }), 500


def _find_model_dir_by_task_id(task_id):
    """根据task_id查找本地模型目录"""
    try:
        models_base_dir = Path(__file__).resolve().parents[4] / 'models' / 'anomaly_detection'
        
        # 在所有模型类型目录中查找
        for model_type_dir in ['lstm_prediction', 'lstm_autoencoder', 'cnn_1d_autoencoder']:
            model_type_path = models_base_dir / model_type_dir
            if not model_type_path.exists():
                continue
                
            task_dir = model_type_path / task_id
            if task_dir.exists() and (task_dir / 'config.json').exists():
                return task_dir
        
        return None
    except Exception as e:
        logger.error(f"查找模型目录失败: {e}")
        return None


def _run_inference_with_local_model(task_id, model_dir, data_path, sequence_length, batch_size, remark):
    """使用本地模型执行推理"""
    try:
        # 生成推理任务ID
        inference_id = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_id}"

        # 创建推理结果目录 (保存到 edge/inference_tasks/anomaly_detection)
        results_dir = Path(__file__).resolve().parents[4] / 'inference_tasks' / 'anomaly_detection' / inference_id
        results_dir.mkdir(parents=True, exist_ok=True)

        relative_data_path = to_relative_path(data_path)

        # 读取模型配置
        config_path = model_dir / 'config.json'
        model_config = {}
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)

        # 检查必要的模型文件（支持 .pth 和 .ckpt 格式）
        model_path = model_dir / 'model.pth'
        if not model_path.exists():
            model_path = model_dir / 'model.ckpt'  # 兼容旧格式
        threshold_path = model_dir / 'threshold.json'
        scaler_path = model_dir / 'scaler.pkl'

        if not model_path.exists():
            return {
                'success': False,
                'error': f'模型文件不存在: {model_dir}/model.pth 或 model.ckpt'
            }

        if not threshold_path.exists():
            return {
                'success': False,
                'error': f'阈值文件不存在: {threshold_path}'
            }

        if not scaler_path.exists():
            return {
                'success': False,
                'error': f'标准化器文件不存在: {scaler_path}'
            }

        # 获取模型类型和序列长度（优先使用模型配置中的值）
        model_type = model_config.get('model_type', 'lstm_predictor')
        # 从模型配置中读取训练时使用的sequence_length，确保推理时使用相同的序列长度
        model_sequence_length = model_config.get('sequence_length', sequence_length)
        
        # 如果表单传入的sequence_length与模型配置不一致，使用模型配置的值并记录警告
        if sequence_length != model_sequence_length:
            logger.warning(
                f"推理时传入的序列长度({sequence_length})与模型训练时的序列长度({model_sequence_length})不一致，"
                f"将使用模型训练时的序列长度({model_sequence_length})"
            )
        
        # 初始化本地异常检测器（使用模型配置中的sequence_length）
        detector = LocalAnomalyDetector(
            model_path=model_path,
            threshold_path=threshold_path,
            scaler_path=scaler_path,
            sequence_length=model_sequence_length,  # 使用模型配置中的序列长度
            model_type=model_type
        )

        # 执行推理
        result = detector.run_inference(data_path, batch_size=batch_size)

        # 添加推理参数信息
        result['task_id'] = task_id
        result['model_type'] = model_type
        result['model_config'] = model_config
        result['inference_params'] = {
            'sequence_length': model_sequence_length,  # 使用实际使用的序列长度
            'batch_size': batch_size,
            'data_file': data_path.name,
            'data_file_path': relative_data_path,
            'remark': remark
        }
        result['model_info'] = {
            'task_id': task_id,
            'model_type': model_type,
            'trained_sequence_length': model_config.get('sequence_length', 50),
            'trained_hidden_units': model_config.get('hidden_units', 128),
            'trained_epochs': model_config.get('epochs'),
            'threshold_value': result.get('threshold_value')
        }

        # 保存推理结果
        result_file = results_dir / 'inference_result.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 读取原始数据用于生成详细CSV
        df = pd.read_csv(data_path)

        # 生成详细数据CSV
        # 确保所有数组长度一致
        n_results = len(result['anomaly_scores'])
        detail_df = pd.DataFrame({
            'timestamp': result['timestamps'][:n_results] if len(result['timestamps']) > n_results else result['timestamps'],
            'anomaly_score': result['anomaly_scores'],
            'is_anomaly': result['anomaly_flags'],
            'threshold': [result['threshold_value']] * n_results
        })

        # 如果原数据有特征列，也包含进来
        feature_cols = [col for col in df.columns if col != 'timestamp']
        if len(feature_cols) > 0 and len(df) >= n_results:
            # 对齐数据长度（使用模型配置中的序列长度）
            # 对于autoencoder模型，start_idx应该是sequence_length - 1
            start_idx = model_sequence_length - 1 if model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder'] else model_sequence_length
            for i, col in enumerate(feature_cols[:5]):  # 最多保留5个特征列
                if start_idx + n_results <= len(df):
                    detail_df[f'feature_{i+1}'] = df[col].iloc[start_idx:start_idx + n_results].values
                else:
                    # 如果数据不够，只取能取到的部分
                    available_len = len(df) - start_idx
                    if available_len > 0:
                        detail_df[f'feature_{i+1}'] = list(df[col].iloc[start_idx:].values) + [None] * (n_results - available_len)
                    else:
                        detail_df[f'feature_{i+1}'] = [None] * n_results

        detail_csv_path = results_dir / 'inference_details.csv'
        detail_df.to_csv(detail_csv_path, index=False)

        return {
            'success': True,
            'inference_id': inference_id,
            'result': result,
            'result_file': to_relative_path(result_file),
            'detail_file': to_relative_path(detail_csv_path)
        }

    except Exception as e:
        logger.error(f"执行推理失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


# 注意：异常检测推理完全在Edge端本地执行
# 模型从 edge/models/anomaly_detection 加载
# 数据从 edge/data/processed/AnomalyDetection 选择
# 结果保存到 edge/inference_tasks/anomaly_detection
