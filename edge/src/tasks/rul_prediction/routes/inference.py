"""
RUL预测推理路由
"""

from flask import Blueprint, request, jsonify, render_template
from pathlib import Path
import json
import logging
import math
import statistics
import time
import shutil
from datetime import datetime
from ..services.inferencer import RULPredictionInferencer

rup_inference_bp = Blueprint('rup_inference', __name__, url_prefix='/rul_prediction')
logger = logging.getLogger(__name__)

EDGE_ROOT = Path(__file__).resolve().parents[4]
MODELS_ROOT = EDGE_ROOT / 'models' / 'rul_prediction'
PROCESSED_DATA_ROOT = EDGE_ROOT / 'data' / 'processed' / 'RULPrediction'
INFERENCE_RESULTS_DIR = EDGE_ROOT / 'inference_tasks' / 'rul_prediction'

# 全局推理器缓存
_inferencer_cache = {}
SUPPORTED_MODEL_TYPES = [
    'bilstm_gru_regressor',
    'cnn_1d_regressor',
    'transformer_encoder_regressor',
]


def get_inferencer(model_id: str) -> RULPredictionInferencer:
    """获取或创建推理器实例"""
    if model_id in _inferencer_cache:
        return _inferencer_cache[model_id]
    
    # 在所有模型类型子目录中查找
    model_dir = None
    for model_type_dir in SUPPORTED_MODEL_TYPES:
        model_type_path = MODELS_ROOT / model_type_dir
        if not model_type_path.exists():
            continue

        prefixed_dir = model_type_path / f'task_{model_id}'
        plain_dir = model_type_path / model_id

        if prefixed_dir.exists():
            model_dir = prefixed_dir
            break
        if plain_dir.exists():
            model_dir = plain_dir
            break
    
    # 如果在子目录中未找到，尝试直接查找（向后兼容）
    if model_dir is None:
        direct_dir_prefixed = MODELS_ROOT / f'task_{model_id}'
        direct_dir_plain = MODELS_ROOT / model_id
        if direct_dir_prefixed.exists():
            model_dir = direct_dir_prefixed
        elif direct_dir_plain.exists():
            model_dir = direct_dir_plain
    
    if model_dir is None or not model_dir.exists():
        raise FileNotFoundError(f'模型 {model_id} 不存在')
    
    # 检查必要文件（优先使用PyTorch权重）
    candidate_model_paths = [
        model_dir / 'best_model.pt',
        model_dir / 'model.pt',
        model_dir / 'model.ckpt',  # 兼容旧版
    ]
    model_path = next((p for p in candidate_model_paths if p.exists()), None)
    config_path = model_dir / 'model_config.json'
    scaler_path = model_dir / 'scaler.pkl'  # 特征scaler（可选）
    label_scaler_path = model_dir / 'label_scaler.pkl'  # 标签scaler（必需）
    
    if model_path is None:
        raise FileNotFoundError(f'模型文件不存在（期望 best_model.pt/model.pt 或兼容的ckpt）: {model_dir}')
    if not config_path.exists():
        raise FileNotFoundError(f'模型配置文件不存在: {config_path}')
    if not label_scaler_path.exists():
        raise FileNotFoundError(f'标签归一化器文件不存在: {label_scaler_path}')
    
    # 读取配置获取序列长度
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    sequence_length = config.get('sequence_length', 50)
    
    # 创建推理器
    inferencer = RULPredictionInferencer(
        model_path=model_path,
        scaler_path=scaler_path if scaler_path.exists() else None,
        label_scaler_path=label_scaler_path,
        config_path=config_path,
        sequence_length=sequence_length
    )
    
    # 缓存推理器
    _inferencer_cache[model_id] = inferencer
    
    return inferencer


@rup_inference_bp.route('/inference')
def inference_page():
    """RUL预测推理页面"""
    return render_template('rul_prediction/inference.html')


@rup_inference_bp.route('/api/inference/data_files', methods=['GET'])
def get_data_files():
    """获取可用于推理的数据文件列表（限定 processed 目录）"""
    try:
        files = []

        if PROCESSED_DATA_ROOT.exists():
            for csv_file in PROCESSED_DATA_ROOT.glob('*.csv'):
                files.append({
                    'filename': csv_file.name,
                    'path': str(csv_file),
                    'type': 'processed',
                    'size': csv_file.stat().st_size
                })

        return jsonify({
            'success': True,
            'files': files
        })

    except Exception as e:
        logger.error(f"获取数据文件列表失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@rup_inference_bp.route('/api/inference/models', methods=['GET'])
def get_models():
    """获取可用的模型列表"""
    try:
        models = []
        
        if MODELS_ROOT.exists():
            for model_type_dir in MODELS_ROOT.iterdir():
                if not model_type_dir.is_dir():
                    continue
                
                for task_dir in model_type_dir.iterdir():
                    if not task_dir.is_dir():
                        continue

                    dir_name = task_dir.name
                    if dir_name.startswith('task_'):
                        model_id = dir_name.replace('task_', '', 1)
                    else:
                        model_id = dir_name
                    
                    config_path = task_dir / 'model_config.json'
                    if not config_path.exists():
                        continue
                    
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        models.append({
                            'model_id': model_id,
                            'model_type': config.get('model_type', 'bilstm_gru_regressor'),
                            'sequence_length': config.get('sequence_length', 50),
                            'input_dim': config.get('input_dim'),
                            'created_at': config.get('created_at'),
                            'path': str(task_dir)
                        })
                    except Exception as e:
                        logger.warning(f"读取模型配置失败 {task_dir}: {e}")
                        continue
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@rup_inference_bp.route('/api/inference/predict', methods=['POST'])
def predict_rul():
    """RUL预测推理API"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        data_file = data.get('data_file')
        batch_size = data.get('batch_size', 32)
        condition_values = data.get('condition_values', {})  # 工况值
        start_time = time.time()
        
        if not model_id:
            return jsonify({'success': False, 'error': '请选择模型'}), 400
        if not data_file:
            return jsonify({'success': False, 'error': '请选择数据文件'}), 400
        
        # 校验数据文件路径（必须位于 processed/RULPrediction 下）
        processed_root = PROCESSED_DATA_ROOT.resolve()
        data_path = Path(data_file).resolve()

        if processed_root not in data_path.parents:
            return jsonify({'success': False, 'error': '数据文件仅支持 edge/data/processed/RULPrediction 目录下的内容'}), 400

        # 使用规范化路径（避免相对路径）
        data_file = str(data_path)

        # 获取推理器
        try:
            inferencer = get_inferencer(model_id)
        except FileNotFoundError as e:
            return jsonify({'success': False, 'error': str(e)}), 404
        
        # 执行预测
        result = inferencer.predict_rul({
            'data_file': data_file,
            'batch_size': batch_size,
            'condition_values': condition_values
        })
        
        if result.get('success'):
            sanitized = _sanitize_predictions(result.get('predictions'))
            result['predictions'] = sanitized['predictions']
            if sanitized['invalid_count']:
                warnings = result.setdefault('warnings', [])
                warnings.append(
                    f"检测到 {sanitized['invalid_count']} 个无效预测值，已用 null 占位，建议检查输入工况/数据"
                )
            enhanced_result = _augment_inference_result(
                base_result=result,
                model_id=model_id,
                batch_size=batch_size,
                condition_values=condition_values,
                duration=time.time() - start_time
            )
            storage_info = _save_inference_result(enhanced_result)
            if storage_info.get('storage_path'):
                enhanced_result['storage_path'] = storage_info['storage_path']
            return jsonify(enhanced_result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"RUL预测失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@rup_inference_bp.route('/api/inference/history', methods=['GET'])
def get_rul_inference_history():
    """获取RUL推理历史"""
    try:
        tasks = _load_inference_history_entries()
        return jsonify({'success': True, 'tasks': tasks, 'total': len(tasks)})
    except Exception as e:
        logger.error(f"获取RUL推理历史失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@rup_inference_bp.route('/api/inference/result/<inference_id>', methods=['GET'])
def get_rul_inference_result(inference_id: str):
    """读取指定推理任务的完整结果"""
    try:
        inference_dir = _find_inference_dir(inference_id)
        if inference_dir is None:
            return jsonify({'success': False, 'error': f'推理结果不存在: {inference_id}'}), 404
        result = _read_json_file(inference_dir / 'inference_result.json')
        if result is None:
            return jsonify({'success': False, 'error': '推理结果文件缺失'}), 404
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        logger.error(f"读取RUL推理结果失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@rup_inference_bp.route('/api/inference/delete/<inference_id>', methods=['DELETE'])
def delete_rul_inference_result(inference_id: str):
    """删除指定推理任务的结果"""
    try:
        inference_dir = _find_inference_dir(inference_id)
        if inference_dir is None:
            return jsonify({'success': False, 'error': f'推理结果不存在: {inference_id}'}), 404
        shutil.rmtree(inference_dir, ignore_errors=False)
        return jsonify({'success': True, 'message': f'已删除推理结果: {inference_id}'})
    except FileNotFoundError:
        return jsonify({'success': False, 'error': f'推理结果不存在: {inference_id}'}), 404
    except Exception as e:
        logger.error(f"删除RUL推理结果失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _augment_inference_result(base_result: dict, model_id: str, batch_size: int,
                              condition_values: dict, duration: float) -> dict:
    """为推理结果补充元数据，便于落盘与展示"""
    inference_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions = base_result.get('predictions') or []
    enhanced_result = {
        **base_result,
        'inference_id': inference_id,
        'model_id': model_id,
        'batch_size': batch_size,
        'condition_values': condition_values,
        'prediction_stats': _compute_prediction_stats(predictions),
        'created_at': datetime.now().isoformat(),
        'inference_time': duration,
    }
    return enhanced_result


def _compute_prediction_stats(predictions):
    """计算预测值的统计信息"""
    if not predictions:
        return {}
    try:
        values = []
        for p in predictions:
            if p is None:
                continue
            try:
                value = float(p)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    except (TypeError, ValueError):
        return {}
    if not values:
        return {}
    stats = {
        'min': min(values),
        'max': max(values),
        'mean': sum(values) / len(values),
        'median': statistics.median(values)
    }
    if len(values) > 1:
        stats['std'] = statistics.pstdev(values)
    stats['latest'] = values[-1]
    return stats


def _sanitize_predictions(predictions):
    """将预测数组中非有限数字替换为 None 以保证JSON合法"""
    if not isinstance(predictions, list):
        try:
            predictions = list(predictions)
        except TypeError:
            return {'predictions': [], 'invalid_count': 0}
    cleaned = []
    invalid = 0
    for value in predictions:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            cleaned.append(None)
            invalid += 1
            continue
        if math.isfinite(numeric):
            cleaned.append(numeric)
        else:
            cleaned.append(None)
            invalid += 1
    return {
        'predictions': cleaned,
        'invalid_count': invalid
    }


def _save_inference_result(result: dict) -> dict:
    """将推理结果保存到 edge/inference_tasks/rul_prediction"""
    inference_id = result.get('inference_id')
    if not inference_id:
        return {}
    try:
        INFERENCE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        inference_dir = INFERENCE_RESULTS_DIR / inference_id
        inference_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整结果
        result_file = inference_dir / 'inference_result.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 保存摘要，避免携带完整数组
        summary = {
            'inference_id': inference_id,
            'model_id': result.get('model_id'),
            'data_file': result.get('data_file'),
            'data_filename': Path(result.get('data_file', '')).name,
            'num_samples': result.get('num_samples'),
            'batch_size': result.get('batch_size'),
            'condition_values': result.get('condition_values', {}),
            'prediction_stats': result.get('prediction_stats', {}),
            'prediction_preview': (result.get('predictions') or [])[:20],
            'created_at': result.get('created_at'),
            'inference_time': result.get('inference_time'),
            'model_config': result.get('model_config')
        }

        summary_file = inference_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return {'storage_path': str(inference_dir)}
    except Exception as exc:
        logger.error(f"保存RUL推理结果失败: {exc}")
        return {}


def _load_inference_history_entries():
    """遍历本地推理结果目录，构建历史记录列表"""
    if not INFERENCE_RESULTS_DIR.exists():
        return []
    entries = []
    for inference_dir in sorted(INFERENCE_RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not inference_dir.is_dir():
            continue
        summary = _read_json_file(inference_dir / 'summary.json')
        if summary is None:
            summary = _read_json_file(inference_dir / 'inference_result.json')
        if summary is None:
            continue
        summary.setdefault('inference_id', inference_dir.name)
        summary['result_dir'] = str(inference_dir)
        entries.append(summary)
    return entries


def _find_inference_dir(inference_id: str):
    """根据ID查找推理结果目录，支持目录名或JSON中的inference_id"""
    if not INFERENCE_RESULTS_DIR.exists():
        return None
    direct_dir = INFERENCE_RESULTS_DIR / inference_id
    if direct_dir.exists() and direct_dir.is_dir():
        return direct_dir
    for candidate in INFERENCE_RESULTS_DIR.iterdir():
        if not candidate.is_dir():
            continue
        result = _read_json_file(candidate / 'inference_result.json')
        if result and result.get('inference_id') == inference_id:
            return candidate
    return None


def _read_json_file(path: Path):
    """安全读取JSON文件"""
    try:
        if not path.exists():
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"读取JSON失败 {path}: {exc}")
        return None
