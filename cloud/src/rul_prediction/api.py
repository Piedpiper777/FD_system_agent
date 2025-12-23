"""
RUL预测API模块
处理RUL预测相关的训练和推理请求
"""

from flask import Blueprint, request, jsonify
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import threading
import sys
import os
import shutil
from typing import Dict, Any, List, Tuple, Optional, Union
import random

# 创建RUL预测Blueprint
rul_prediction_bp = Blueprint('rul_prediction', __name__, url_prefix='/api/rul_prediction')

# 添加项目路径以导入训练模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

AVAILABLE_MODEL_TYPES: List[str] = []

# 导入训练组件（PyTorch）
logger = logging.getLogger(__name__)
try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from rul_prediction.core import (
        RULPredictionEvaluator,
        get_available_model_types,
        get_model_builder,
        get_trainer_class,
    )

    training_available = True
    AVAILABLE_MODEL_TYPES = get_available_model_types()
    logger.info(
        "RUL预测训练模块加载成功，已注册模型: %s",
        ", ".join(AVAILABLE_MODEL_TYPES)
    )
except ImportError as e:
    training_available = False
    AVAILABLE_MODEL_TYPES = [
        'bilstm_gru_regressor',
        'cnn_1d_regressor',
        'transformer_encoder_regressor',
    ]
    logger.warning(f"训练模块不可用: {e}")

def _normalize_device_target(value):
    normalized = str(value or 'cuda:0').strip().lower()
    if normalized in ('gpu', 'cuda'):
        return 'cuda:0'
    if normalized.startswith('cuda'):
        return normalized
    return 'cpu'

# 导入任务管理器
training_tasks = {}

try:
    from ..common.task_manager import get_task_manager, TrainingTask, TrainingStatus
except ImportError:
    try:
        from common.task_manager import get_task_manager, TrainingTask, TrainingStatus
    except ImportError:
        def get_task_manager():
            class SimpleTaskManager:
                def __init__(self):
                    self.tasks = {}
                def create_task(self, config):
                    task_id = config.get('task_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
                    training_tasks[task_id] = {'status': 'queued', 'config': config, 'task_id': task_id}
                    self.tasks[task_id] = type('Task', (), training_tasks[task_id])()
                    return self.tasks[task_id]
                def get_task(self, task_id):
                    if task_id in training_tasks:
                        return type('Task', (), training_tasks[task_id])()
                    return None
                def update_task_status(self, task_id, status, message=''):
                    if task_id in training_tasks:
                        training_tasks[task_id]['status'] = status
                        training_tasks[task_id]['message'] = message
                def add_log(self, task_id, log_message):
                    if task_id in training_tasks:
                        if 'logs' not in training_tasks[task_id]:
                            training_tasks[task_id]['logs'] = []
                        training_tasks[task_id]['logs'].append({
                            'timestamp': datetime.now().isoformat(),
                            'level': 'info',
                            'message': log_message
                        })
                def start_training(self, task_id, training_func):
                    threading.Thread(target=training_func, args=(task_id,), daemon=True).start()
            return SimpleTaskManager()

# 数据文件存储
uploaded_data_files = {}  # 存储上传的数据文件信息

@rul_prediction_bp.route('/upload_data', methods=['POST'])
def upload_training_data():
    """接收边端上传的训练数据，保存到云端训练数据目录"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # 获取task_id和file_type
        task_id = request.form.get('task_id', '').strip()
        file_type = request.form.get('file_type', 'train').strip()
        
        if not task_id:
            return jsonify({'success': False, 'error': 'task_id is required'}), 400
        
        # 保存到task_id对应的目录: cloud/data/rul/<task_id>/raw/
        training_data_dir = Path('data') / 'rul' / task_id / 'raw'
        training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用原始文件名
        filename = file.filename
        file_path = training_data_dir / filename
        
        # 保存文件
        file.save(str(file_path))
        
        # 记录文件信息
        uploaded_data_files[file.filename] = {
            'original_name': file.filename,
            'saved_name': filename,
            'path': str(file_path),
            'uploaded_at': datetime.now().isoformat(),
            'size': file_path.stat().st_size,
            'source': 'edge_upload',
            'task_id': task_id,
            'file_type': file_type
        }
        
        logger.debug(f"数据文件上传: {filename} ({file_path.stat().st_size} bytes) to {file_path}")
        
        return jsonify({
            'success': True,
            'original_filename': file.filename,
            'saved_filename': filename,
            'message': 'Training data uploaded successfully to cloud'
        })
        
    except Exception as e:
        logger.error(f"Data upload failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@rul_prediction_bp.route('/train', methods=['POST'])
def create_training():
    """创建RUL预测训练任务"""
    if not training_available:
        return jsonify({
            'success': False,
            'error': 'Training functionality not available'
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    logger.info(
        "接收到RUL训练请求: task_id=%s, model_type=%s, 提供参数=%s",
        data.get('task_id', 'auto'),
        data.get('model_type', 'bilstm_gru_regressor'),
        sorted(list(data.keys()))
    )

    # 验证RUL预测模型类型
    valid_models = AVAILABLE_MODEL_TYPES or ['bilstm_gru_regressor']
    model_type = data.get('model_type', 'bilstm_gru_regressor')
    if model_type not in valid_models:
        return jsonify({
            'success': False,
            'error': f'Invalid model_type for rul_prediction. Must be one of: {", ".join(valid_models)}'
        }), 400

    data['model_type'] = model_type
    data['module'] = 'rul_prediction'

    try:
        # 使用任务管理器创建任务
        task_manager = get_task_manager()
        
        # 如果配置中已有task_id，使用它（Edge端生成的）
        # 否则让任务管理器生成新的task_id
        if 'task_id' in data and data['task_id']:
            # 手动创建任务，使用Edge端提供的task_id
            task_id = data['task_id']
            
            # 创建TrainingTask对象，参考fault_diagnosis的实现
            try:
                from ..common.task_manager import TrainingTask
            except ImportError:
                from common.task_manager import TrainingTask
            
            # 准备任务参数（只包含TrainingTask支持的字段）
            def _to_int(value, default):
                try:
                    if value is None or value == '':
                        return default
                    return int(value)
                except (TypeError, ValueError):
                    return default

            def _to_float(value, default):
                try:
                    if value is None or value == '':
                        return default
                    return float(value)
                except (TypeError, ValueError):
                    return default

            def _to_bool(value, default=False):
                if value is None:
                    return default
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.strip().lower() in ('true', '1', 'yes', 'y')
                return bool(value)
            
            # 创建任务对象
            task = TrainingTask(
                task_id=task_id,
                module=data.get('module', 'rul_prediction'),
                model_type=data.get('model_type', 'bilstm_gru_regressor'),
                output_path=data.get('output_path', f'models/rul_prediction/{data.get("model_type", "bilstm_gru_regressor")}/{task_id}'),
                input_dim=_to_int(data.get('input_dim'), 10),  # 默认值，实际会在训练时更新
                dataset_mode=data.get('dataset_mode', 'condition_filtered'),
                epochs=_to_int(data.get('epochs'), 50),
                batch_size=_to_int(data.get('batch_size'), 32),
                learning_rate=_to_float(data.get('learning_rate'), 0.001),
                validation_split=_to_float(data.get('validation_split'), None),  # RUL预测使用文件级验证集，不需要validation_split
                sequence_length=_to_int(data.get('sequence_length'), 50),
                hidden_units=_to_int(data.get('hidden_units'), 128),
                num_layers=_to_int(data.get('num_layers'), 2),
                dropout=_to_float(data.get('dropout'), 0.3),
                activation=data.get('activation', 'relu'),
                bidirectional=_to_bool(data.get('bidirectional'), True),
                train_file=data.get('train_file'),
                val_file=data.get('val_file'),
                test_file=data.get('test_file'),
                status='queued'
            )
            
            # 保存完整的原始配置（包括train_files, test_files, conditions等）
            # 移除不需要的validation_split字段（RUL预测使用文件级验证集）
            raw_config = data.copy()
            if 'validation_split' in raw_config:
                del raw_config['validation_split']
            
            task._raw_config = raw_config
            logger.debug(
                "任务%s使用_edge提供的配置，关键模型参数: rnn_type=%s, hidden_units=%s, bidirectional=%s, use_layer_norm=%s",
                task_id,
                task._raw_config.get('rnn_type'),
                task._raw_config.get('hidden_units'),
                task._raw_config.get('bidirectional'),
                task._raw_config.get('use_layer_norm')
            )
            
            # 将任务添加到任务管理器
            with task_manager.lock:
                task_manager.tasks[task_id] = task
            
            # 同时保存到training_tasks字典（如果使用SimpleTaskManager）
            training_tasks[task_id] = {
                'task_id': task_id,
                'status': 'queued',
                'config': data
            }
        else:
            # 使用任务管理器生成task_id
            task = task_manager.create_task(data)
        
        # 启动异步训练
        task_manager.start_training(task.task_id, _run_real_training)

        logger.info(f"训练任务已创建: {task.task_id}")
        return jsonify({
            'success': True,
            'message': 'RUL prediction training task created',
            'task_id': task.task_id,
            'model_type': model_type,
            'module': 'rul_prediction'
        })

    except Exception as e:
        logger.error(f"Failed to create training task: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Failed to create training task: {str(e)}'
        }), 500


def _process_condition_filtered_data(config, task_id, task_manager, model_type):
    """
    处理工况筛选模式的数据（RUL预测专用）：
    
    重要原则 - Unit级别独立性：
    ============================
    1. 每个文件代表一个独立的unit（运行序列）
    2. 每个unit是完整的run-to-failure轨迹，不可截断
    3. 数据划分基于unit级别：train units, val units, test units
    4. 不使用比例划分（如70% train, 30% val），这会破坏退化轨迹完整性
    5. unit之间必须完全独立，避免信息泄漏
    
    处理流程：
    =========
    1. 读取训练units、验证units、测试units（完整文件）
    2. 从元数据文件读取工况信息和RUL配置
    3. 根据用户选择的工况key，补全工况列
    4. 计算RUL标签（最后一个点就是故障点）
    5. 对每个unit创建滑动窗口
    6. 分别合并train/val/test units的窗口
    7. 保存为train.npz, dev.npz, test.npz
    """
    import pickle
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # 获取文件列表
    train_files = config.get('train_files', [])
    validation_files = config.get('validation_files', [])  # 文件级验证集（必需）
    test_files = config.get('test_files', [])
    
    # 获取用户选择的工况key列表
    condition_keys = config.get('condition_keys', [])  # 例如: ['转速', '负载']
    conditions = config.get('conditions', [])  # 完整的工况配置 [{name: '转速', values: [20, 30]}, ...]
    
    # 构建工况key到值的映射（从conditions配置中）
    condition_key_to_values = {}
    for cond in conditions:
        if isinstance(cond, dict) and 'name' in cond:
            condition_key_to_values[cond['name']] = cond.get('values', [])
    
    # RUL预测任务必须使用文件级（unit级）验证集
    if not validation_files or len(validation_files) == 0:
        raise ValueError(
            "RUL预测任务必须提供validation_files（验证集units）。"
            "数据划分应基于unit级别，不能使用比例划分。"
            "每个unit是完整的run-to-failure轨迹，不可截断。"
        )
    
    sequence_length = config.get('sequence_length', 50)
    stride = config.get('stride', 1)
    random_seed = config.get('random_seed', 42)
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 查找数据文件目录
    training_data_dir = Path('data') / 'rul' / task_id
    raw_data_dir = training_data_dir / 'raw'
    
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {raw_data_dir}")
    
    task_manager.add_log(
        task_id, 
        f'使用Unit级别数据划分: {len(train_files)} 个训练units, {len(validation_files)} 个验证units, {len(test_files)} 个测试units'
    )
    task_manager.add_log(
        task_id,
        '重要：每个unit是完整的run-to-failure轨迹，unit之间完全独立'
    )
    
    # Step 1 & 2: 读取数据，补全工况列，计算RUL标签
    train_data_list = []  # 存储每个文件的(数据, RUL标签, 工况值)
    val_data_list = []    # 新增：存储验证集文件
    test_data_list = []
    
    for filename in train_files:
        csv_path = raw_data_dir / filename
        meta_path = raw_data_dir / filename.replace('.csv', '.json')
        
        if not csv_path.exists():
            task_manager.add_log(task_id, f'警告: 训练文件不存在: {filename}')
            continue
        
        if not meta_path.exists():
            task_manager.add_log(task_id, f'警告: 元文件不存在: {meta_path.name}')
            continue
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        if df.empty:
            task_manager.add_log(task_id, f'警告: 训练文件为空: {filename}')
            continue
        
        # 读取元文件
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        # 获取RUL配置
        rul_config = meta_data.get('rul_config', {})
        rul_unit = rul_config.get('rul_unit', 'cycle')
        max_rul = rul_config.get('max_rul', 200)
        
        # 获取工况信息（从元文件）
        tags_condition = meta_data.get('tags_condition', [])
        file_conditions = {}
        for cond in tags_condition:
            if isinstance(cond, dict) and 'key' in cond and 'value' in cond:
                file_conditions[cond['key']] = cond['value']
        
        # 补全工况列（只添加用户选择的工况key）
        for key in condition_keys:
            if key in file_conditions:
                # 使用文件中的工况值
                df[key] = file_conditions[key]
            else:
                # 如果文件中没有该工况，使用默认值（取第一个值）
                if key in condition_key_to_values and condition_key_to_values[key]:
                    df[key] = condition_key_to_values[key][0]
                else:
                    df[key] = 0  # 默认值
        
        # Step 2: 计算RUL标签
        # 因为文件已经截断到故障点，最后一个点就是故障点
        N = len(df)
        
        # 数据在标注前已经截断到故障点，所以最后一行就是故障点(RUL=0)
        # 不再使用元文件中的failure_row_index，而是直接使用数据实际长度
        failure_row_index = N - 1  # 最后一个点的索引（故障点）
        
        task_manager.add_log(
            task_id,
            f'文件 {filename}: 数据长度={N}, 故障点索引={failure_row_index}, RUL单位={rul_unit}'
        )
        
        if rul_unit == 'cycle':
            # 按采样点计算：RUL[i] = (N-1) - i
            # 第一个点(i=0)的RUL = N-1，最后一个点(i=N-1)的RUL = 0
            rul_labels = np.array([N - 1 - i for i in range(N)], dtype=np.float32)
        else:
            # 按时间戳计算（需要时间戳列）
            # 这里简化处理，假设有timestamp列
            timestamp_col = None
            for col in df.columns:
                if 'time' in col.lower() or 'timestamp' in col.lower():
                    timestamp_col = col
                    break
            
            if timestamp_col:
                timestamps = pd.to_datetime(df[timestamp_col]).values
                failure_timestamp = timestamps[failure_row_index]
                
                if rul_unit == 'second':
                    time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 's')
                elif rul_unit == 'minute':
                    time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 'm')
                else:
                    time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 's')
                
                rul_labels = np.maximum(0, time_diffs).astype(np.float32)
            else:
                # 没有时间戳列，回退到cycle模式
                rul_labels = np.array([N - 1 - i for i in range(N)], dtype=np.float32)
        
        # 截断到最大RUL值
        # 注意：这会将所有超过max_rul的RUL值截断为max_rul
        # 这可能导致早期数据点的真实RUL信息丢失
        original_max_rul = rul_labels.max()
        rul_labels = np.minimum(rul_labels, max_rul)
        
        # 记录截断信息（如果发生了截断）
        if original_max_rul > max_rul:
            truncated_count = np.sum(rul_labels == max_rul)
            task_manager.add_log(
                task_id,
                f'文件 {filename}: RUL标签被截断。原始最大RUL={original_max_rul:.1f}, '
                f'截断值={max_rul}, 被截断的样本数={truncated_count} ({truncated_count/N*100:.1f}%)'
            )
        
        # 准备特征数据：工况也会作为特征的一部分
        # 先分离传感器特征和工况特征，然后合并（便于后续统一处理）
        timestamp_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        # 传感器特征列：排除工况列和时间戳列（工况会单独处理后再合并）
        exclude_cols = set(condition_keys)
        if timestamp_col:
            exclude_cols.add(timestamp_col)
        sensor_feature_cols = [col for col in df.columns if col not in exclude_cols]
        condition_cols = condition_keys
        
        # 提取传感器特征和工况特征
        sensor_features = df[sensor_feature_cols].values.astype(np.float32)
        condition_features = df[condition_cols].values.astype(np.float32) if condition_cols else None
        
        # 合并传感器特征和工况特征（工况作为特征的一部分）
        # 最终特征 = [传感器特征, 工况特征]
        if condition_features is not None:
            data_with_conditions = np.hstack([sensor_features, condition_features])
        else:
            data_with_conditions = sensor_features
        
        # 记录RUL标签统计信息
        task_manager.add_log(
            task_id,
            f'文件 {filename}: RUL标签统计 - 最小值={rul_labels.min():.1f}, '
            f'最大值={rul_labels.max():.1f}, 均值={rul_labels.mean():.1f}, '
            f'中位数={np.median(rul_labels):.1f}, 标准差={rul_labels.std():.1f}'
        )
        
        # 从元数据中提取unit_id（如果存在）
        unit_id = meta_data.get('unit_id', filename)
        
        train_data_list.append({
            'filename': filename,
            'unit_id': unit_id,  # 保存unit_id用于日志
            'data': data_with_conditions,
            'rul_labels': rul_labels,
            'conditions': file_conditions
        })
    
    # 处理测试文件（类似流程，但不划分验证集）
    for filename in test_files:
        csv_path = raw_data_dir / filename
        meta_path = raw_data_dir / filename.replace('.csv', '.json')
        
        if not csv_path.exists() or not meta_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        rul_config = meta_data.get('rul_config', {})
        rul_unit = rul_config.get('rul_unit', 'cycle')
        max_rul = rul_config.get('max_rul', 200)
        
        tags_condition = meta_data.get('tags_condition', [])
        file_conditions = {}
        for cond in tags_condition:
            if isinstance(cond, dict) and 'key' in cond and 'value' in cond:
                file_conditions[cond['key']] = cond['value']
        
        # 补全工况列
        for key in condition_keys:
            if key in file_conditions:
                df[key] = file_conditions[key]
            else:
                if key in condition_key_to_values and condition_key_to_values[key]:
                    df[key] = condition_key_to_values[key][0]
                else:
                    df[key] = 0
        
        # 计算RUL标签
        N = len(df)
        
        # 数据在标注前已经截断到故障点，所以最后一行就是故障点(RUL=0)
        failure_row_index = N - 1  # 最后一个点的索引（故障点）
        
        task_manager.add_log(
            task_id,
            f'测试文件 {filename}: 数据长度={N}, 故障点索引={failure_row_index}, RUL单位={rul_unit}'
        )
        
        if rul_unit == 'cycle':
            # 按采样点计算：RUL[i] = (N-1) - i
            # 第一个点(i=0)的RUL = N-1，最后一个点(i=N-1)的RUL = 0
            rul_labels = np.array([N - 1 - i for i in range(N)], dtype=np.float32)
        else:
            timestamp_col = None
            for col in df.columns:
                if 'time' in col.lower() or 'timestamp' in col.lower():
                    timestamp_col = col
                    break
            
            if timestamp_col:
                timestamps = pd.to_datetime(df[timestamp_col]).values
                failure_timestamp = timestamps[failure_row_index]
                
                if rul_unit == 'second':
                    time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 's')
                elif rul_unit == 'minute':
                    time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 'm')
                else:
                    time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 's')
                
                rul_labels = np.maximum(0, time_diffs).astype(np.float32)
            else:
                rul_labels = np.array([N - 1 - i for i in range(N)], dtype=np.float32)
        
        # 截断到最大RUL值
        original_max_rul = rul_labels.max()
        rul_labels = np.minimum(rul_labels, max_rul)
        
        # 记录截断信息（如果发生了截断）
        if original_max_rul > max_rul:
            truncated_count = np.sum(rul_labels == max_rul)
            task_manager.add_log(
                task_id,
                f'测试文件 {filename}: RUL标签被截断。原始最大RUL={original_max_rul:.1f}, '
                f'截断值={max_rul}, 被截断的样本数={truncated_count} ({truncated_count/N*100:.1f}%)'
            )
        
        # 准备特征数据：工况也会作为特征的一部分
        # 先分离传感器特征和工况特征，然后合并（便于后续统一处理）
        timestamp_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        # 传感器特征列：排除工况列和时间戳列（工况会单独处理后再合并）
        exclude_cols = set(condition_keys)
        if timestamp_col:
            exclude_cols.add(timestamp_col)
        sensor_feature_cols = [col for col in df.columns if col not in exclude_cols]
        condition_cols = condition_keys
        
        # 提取传感器特征和工况特征
        sensor_features = df[sensor_feature_cols].values.astype(np.float32)
        condition_features = df[condition_cols].values.astype(np.float32) if condition_cols else None
        
        # 合并传感器特征和工况特征（工况作为特征的一部分）
        # 最终特征 = [传感器特征, 工况特征]
        if condition_features is not None:
            data_with_conditions = np.hstack([sensor_features, condition_features])
        else:
            data_with_conditions = sensor_features
        
        # 从元数据中提取unit_id（如果存在）
        unit_id = meta_data.get('unit_id', filename)
        
        test_data_list.append({
            'filename': filename,
            'unit_id': unit_id,  # 保存unit_id用于日志
            'data': data_with_conditions,
            'rul_labels': rul_labels,
            'conditions': file_conditions
        })
    
    # 读取验证集文件（必需，RUL任务不使用比例划分）
    for filename in validation_files:
        csv_path = raw_data_dir / filename
        meta_path = raw_data_dir / filename.replace('.csv', '.json')
        
        if not csv_path.exists():
            task_manager.add_log(task_id, f'警告: 验证文件不存在: {filename}')
            continue
        
        if not meta_path.exists():
            task_manager.add_log(task_id, f'警告: 验证文件元数据不存在: {meta_path.name}')
            continue
        
        # 读取CSV文件和元数据（使用和训练文件相同的逻辑）
        df = pd.read_csv(csv_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            val_meta_data = json.load(f)
        
        rul_config = val_meta_data.get('rul_config', {})
        max_rul = rul_config.get('max_rul', 200)
        rul_unit = rul_config.get('rul_unit', 'cycle')
        tags_condition = val_meta_data.get('tags_condition', [])
        
        # 补全工况列
        file_conditions = {}
        for cond in tags_condition:
            if isinstance(cond, dict) and 'key' in cond and 'value' in cond:
                file_conditions[cond['key']] = cond['value']
        
        for key in condition_keys:
            if key in file_conditions:
                df[key] = file_conditions[key]
            else:
                if key in condition_key_to_values and condition_key_to_values[key]:
                    df[key] = condition_key_to_values[key][0]
                else:
                    df[key] = 0
        
        # 计算RUL标签
        N = len(df)
        failure_row_index = N - 1
        
        if rul_unit == 'cycle':
            rul_labels = np.array([N - 1 - i for i in range(N)], dtype=np.float32)
        else:
            timestamp_col = None
            for col in df.columns:
                if 'time' in col.lower() or 'timestamp' in col.lower():
                    timestamp_col = col
                    break
            
            if timestamp_col:
                timestamps = pd.to_datetime(df[timestamp_col]).values
                failure_timestamp = timestamps[failure_row_index]
                
                if rul_unit == 'second':
                    time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 's')
                elif rul_unit == 'minute':
                    time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 'm')
                else:
                    time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 's')
                
                rul_labels = np.maximum(0, time_diffs).astype(np.float32)
            else:
                rul_labels = np.array([N - 1 - i for i in range(N)], dtype=np.float32)
        
        # 截断RUL标签
        original_max_rul = rul_labels.max()
        rul_labels = np.minimum(rul_labels, max_rul)
        
        if original_max_rul > max_rul:
            truncated_count = np.sum(rul_labels == max_rul)
            task_manager.add_log(
                task_id,
                f'验证文件 {filename}: RUL标签被截断。原始最大RUL={original_max_rul:.1f}, '
                f'截断值={max_rul}, 被截断的样本数={truncated_count} ({truncated_count/N*100:.1f}%)'
            )
        
        # 提取特征
        timestamp_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        exclude_cols = set(condition_keys)
        if timestamp_col:
            exclude_cols.add(timestamp_col)
        sensor_feature_cols = [col for col in df.columns if col not in exclude_cols]
        condition_cols = condition_keys
        
        sensor_features = df[sensor_feature_cols].values.astype(np.float32)
        condition_features = df[condition_cols].values.astype(np.float32) if condition_cols else None
        
        if condition_features is not None:
            data_with_conditions = np.hstack([sensor_features, condition_features])
        else:
            data_with_conditions = sensor_features
        
        task_manager.add_log(
            task_id,
            f'验证文件 {filename}: 数据形状={data_with_conditions.shape}, RUL范围=[{rul_labels.min():.1f}, {rul_labels.max():.1f}]'
        )
        
        # 从元数据中提取unit_id（如果存在）
        unit_id = val_meta_data.get('unit_id', filename)
        
        val_data_list.append({
            'filename': filename,
            'unit_id': unit_id,  # 保存unit_id用于日志
            'data': data_with_conditions,
            'rul_labels': rul_labels,
            'conditions': file_conditions
        })
    
    if not train_data_list:
        error_msg = f"没有有效的训练数据文件。已检查 {len(train_files)} 个训练文件。"
        task_manager.add_log(task_id, error_msg)
        raise ValueError(error_msg)
    
    if not val_data_list:
        error_msg = f"没有有效的验证数据文件。已检查 {len(validation_files)} 个验证文件。RUL任务必须提供独立的验证units。"
        task_manager.add_log(task_id, error_msg)
        raise ValueError(error_msg)
    
    task_manager.add_log(
        task_id, 
        f'成功读取 {len(train_data_list)} 个训练units, {len(val_data_list)} 个验证units, {len(test_data_list)} 个测试units'
    )
    
    # Step 3: Unit级别数据划分（每个unit完整处理，不截断）
    all_train_sequences = []
    all_train_labels = []
    all_val_sequences = []
    all_val_labels = []
    all_test_sequences = []
    all_test_labels = []
    
    task_manager.add_log(task_id, '使用Unit级别数据划分，每个unit保持完整')
    
    # 处理训练units（完整unit用于训练）
    for file_data in train_data_list:
        data = file_data['data']
        rul_labels = file_data['rul_labels']
        filename = file_data['filename']
        unit_id = file_data.get('unit_id', 'unknown')
        
        # 创建滑动窗口
        train_sequences, train_seq_labels = _create_sequences(data, rul_labels, sequence_length, stride)
        
        if len(train_sequences) > 0:
            all_train_sequences.append(train_sequences)
            all_train_labels.append(train_seq_labels)
            task_manager.add_log(
                task_id,
                f'训练unit {unit_id} ({filename}): 生成 {len(train_sequences)} 个训练序列'
            )
    
    # 处理验证units（完整unit用于验证）
    for file_data in val_data_list:
        data = file_data['data']
        rul_labels = file_data['rul_labels']
        filename = file_data['filename']
        unit_id = file_data.get('unit_id', 'unknown')
        
        # 创建滑动窗口
        val_sequences, val_seq_labels = _create_sequences(data, rul_labels, sequence_length, stride)
        
        if len(val_sequences) > 0:
            all_val_sequences.append(val_sequences)
            all_val_labels.append(val_seq_labels)
            task_manager.add_log(
                task_id,
                f'验证unit {unit_id} ({filename}): 生成 {len(val_sequences)} 个验证序列'
            )
    
    # 处理测试units（完整unit用于测试）
    for file_data in test_data_list:
        data = file_data['data']
        rul_labels = file_data['rul_labels']
        filename = file_data['filename']
        unit_id = file_data.get('unit_id', 'unknown')
        
        if len(data) >= sequence_length:
            test_seqs, test_seq_labels = _create_sequences(data, rul_labels, sequence_length, stride)
            all_test_sequences.append(test_seqs)
            all_test_labels.append(test_seq_labels)
            task_manager.add_log(
                task_id,
                f'测试unit {unit_id} ({filename}): 生成 {len(test_seqs)} 个测试序列'
            )
    
    # 合并所有文件的序列
    if all_train_sequences:
        train_sequences = np.vstack(all_train_sequences)
        train_labels = np.concatenate(all_train_labels)
    else:
        error_msg = "没有生成训练序列。可能原因：1) 数据文件太短；2) sequence_length设置过大；3) 数据文件为空。"
        task_manager.add_log(task_id, error_msg)
        raise ValueError(error_msg)
    
    if all_val_sequences:
        val_sequences = np.vstack(all_val_sequences)
        val_labels = np.concatenate(all_val_labels)
    else:
        val_sequences = np.array([]).reshape(0, sequence_length, train_sequences.shape[2])
        val_labels = np.array([])
    
    if all_test_sequences:
        test_sequences = np.vstack(all_test_sequences)
        test_labels = np.concatenate(all_test_labels)
    else:
        test_sequences = np.array([]).reshape(0, sequence_length, train_sequences.shape[2])
        test_labels = np.array([])
    
    task_manager.add_log(task_id, f'生成训练序列: {len(train_sequences)}, 验证序列: {len(val_sequences)}, 测试序列: {len(test_sequences)}')
    
    # 对RUL标签进行MinMax归一化（基于训练标签）
    label_scaler = MinMaxScaler()
    train_labels_reshaped = train_labels.reshape(-1, 1)
    label_scaler.fit(train_labels_reshaped)
    
    # 归一化所有标签
    train_labels_normalized = label_scaler.transform(train_labels_reshaped).flatten()
    if len(val_labels) > 0:
        val_labels_normalized = label_scaler.transform(val_labels.reshape(-1, 1)).flatten()
    else:
        val_labels_normalized = val_labels
    
    if len(test_labels) > 0:
        test_labels_normalized = label_scaler.transform(test_labels.reshape(-1, 1)).flatten()
    else:
        test_labels_normalized = test_labels
    
    # 记录标签归一化信息
    original_label_min = train_labels.min()
    original_label_max = train_labels.max()
    normalized_label_min = train_labels_normalized.min()
    normalized_label_max = train_labels_normalized.max()
    
    task_manager.add_log(
        task_id,
        f'RUL标签归一化完成: 原始范围=[{original_label_min:.3f}, {original_label_max:.3f}], '
        f'归一化后范围=[{normalized_label_min:.3f}, {normalized_label_max:.3f}]'
    )
    
    # 检查数据是否已经归一化（通过检查数据范围）
    # 如果数据已经在[0,1]或[-1,1]范围内，可能已经归一化过
    train_sequences_2d = train_sequences.reshape(-1, train_sequences.shape[2])
    data_min = train_sequences_2d.min()
    data_max = train_sequences_2d.max()
    data_mean = train_sequences_2d.mean()
    data_std = train_sequences_2d.std()
    
    # 判断数据是否已经归一化：
    # 1. 如果数据范围在[0,1]或[-1,1]附近，且标准差较小，可能已经归一化
    # 2. 或者通过配置选项明确指定
    skip_normalization = config.get('skip_normalization', False) or config.get('data_already_normalized', False)
    
    # 自动检测：如果数据范围在[-1.5, 1.5]且均值接近0，标准差接近1，可能是StandardScaler的结果
    # 如果数据范围在[0, 1]或[-1, 1]，可能是MinMax归一化的结果
    if not skip_normalization:
        if (data_min >= -1.5 and data_max <= 1.5 and abs(data_mean) < 0.5 and 0.5 < data_std < 2.0) or \
           (data_min >= -0.1 and data_max <= 1.1):
            task_manager.add_log(
                task_id,
                f'检测到数据可能已经归一化: 范围=[{data_min:.3f}, {data_max:.3f}], '
                f'均值={data_mean:.3f}, 标准差={data_std:.3f}。跳过标准化处理。'
            )
            skip_normalization = True
    
    if skip_normalization:
        # 数据已经归一化，直接使用
        task_manager.add_log(task_id, '数据已归一化，跳过标准化处理')
        train_sequences_scaled = train_sequences
        val_sequences_scaled = val_sequences
        test_sequences_scaled = test_sequences
        scaler = None  # 不保存scaler
    else:
        # 标准化数据（只基于训练数据）
        task_manager.add_log(
            task_id,
            f'对数据进行标准化: 原始范围=[{data_min:.3f}, {data_max:.3f}], '
            f'均值={data_mean:.3f}, 标准差={data_std:.3f}'
        )
        scaler = StandardScaler()
        scaler.fit(train_sequences_2d)
        
        # 标准化所有数据
        train_sequences_2d_scaled = scaler.transform(train_sequences_2d)
        train_sequences_scaled = train_sequences_2d_scaled.reshape(train_sequences.shape)
        
        if len(val_sequences) > 0:
            val_sequences_2d = val_sequences.reshape(-1, val_sequences.shape[2])
            val_sequences_2d_scaled = scaler.transform(val_sequences_2d)
            val_sequences_scaled = val_sequences_2d_scaled.reshape(val_sequences.shape)
        else:
            val_sequences_scaled = val_sequences
        
        if len(test_sequences) > 0:
            test_sequences_2d = test_sequences.reshape(-1, test_sequences.shape[2])
            test_sequences_2d_scaled = scaler.transform(test_sequences_2d)
            test_sequences_scaled = test_sequences_2d_scaled.reshape(test_sequences.shape)
        else:
            test_sequences_scaled = test_sequences
        
        # 记录标准化后的统计信息
        train_scaled_2d = train_sequences_scaled.reshape(-1, train_sequences_scaled.shape[2])
        task_manager.add_log(
            task_id,
            f'标准化完成: 新范围=[{train_scaled_2d.min():.3f}, {train_scaled_2d.max():.3f}], '
            f'均值={train_scaled_2d.mean():.3f}, 标准差={train_scaled_2d.std():.3f}'
        )
    
    # 保存数据集
    processed_dir = training_data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        processed_dir / 'train.npz',
        sequences=train_sequences_scaled,
        labels=train_labels_normalized
    )
    
    if len(val_sequences_scaled) > 0:
        np.savez_compressed(
            processed_dir / 'dev.npz',
            sequences=val_sequences_scaled,
            labels=val_labels_normalized
        )
    
    if len(test_sequences_scaled) > 0:
        np.savez_compressed(
            processed_dir / 'test.npz',
            sequences=test_sequences_scaled,
            labels=test_labels_normalized
        )
    
    # 保存特征scaler（如果进行了标准化）
    if scaler is not None:
        with open(processed_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        task_manager.add_log(task_id, '特征标准化器已保存')
    else:
        task_manager.add_log(task_id, '特征未标准化，未保存标准化器')
    
    # 保存标签scaler（始终保存，用于预测时反归一化）
    with open(processed_dir / 'label_scaler.pkl', 'wb') as f:
        pickle.dump(label_scaler, f)
    task_manager.add_log(task_id, '标签归一化器已保存')
    
    task_manager.add_log(task_id, f'数据集已保存到: {processed_dir}')
    
    return {
        'train_file': str(processed_dir / 'train.npz'),
        'val_file': str(processed_dir / 'dev.npz') if len(val_sequences_scaled) > 0 else None,
        'test_file': str(processed_dir / 'test.npz') if len(test_sequences_scaled) > 0 else None,
        'scaler_file': str(processed_dir / 'scaler.pkl') if scaler is not None else None,
        'label_scaler_file': str(processed_dir / 'label_scaler.pkl'),
        'input_dim': train_sequences_scaled.shape[2],  # 特征维度（包含工况）
        'num_train_samples': len(train_sequences_scaled),
        'num_val_samples': len(val_sequences_scaled),
        'num_test_samples': len(test_sequences_scaled)
    }


def _evaluate_rul_model(
    model: torch.nn.Module,
    test_file: Union[str, Path],
    label_scaler_file: Optional[Union[str, Path]],
    config: dict,
    task_id: str
) -> Dict[str, Any]:
    """
    使用测试集评估RUL预测模型
    
    Args:
        model: 训练好的模型
        test_file: 测试数据文件路径（.npz格式）
        label_scaler_file: 标签归一化器文件路径（用于反归一化）
        config: 训练配置
        task_id: 任务ID
        
    Returns:
        评估结果字典，包含RMSE, MAE, R², MAPE等指标
    """
    try:
        task_manager = get_task_manager()
        task_manager.add_log(task_id, f'加载测试集: {test_file}')
        
        # 加载测试数据
        test_data = np.load(test_file)
        test_sequences = test_data['sequences']
        test_labels_normalized = test_data['labels']
        
        task_manager.add_log(task_id, f'测试集大小: {len(test_sequences)} 个样本')
        
        # 加载标签scaler（用于反归一化）
        label_scaler = None
        if label_scaler_file and Path(label_scaler_file).exists():
            import pickle
            with open(label_scaler_file, 'rb') as f:
                label_scaler = pickle.load(f)
            task_manager.add_log(task_id, '标签归一化器已加载')
        else:
            task_manager.add_log(task_id, '警告：未找到标签归一化器，预测值可能未反归一化')
        
        # 创建评估器
        evaluator = RULPredictionEvaluator(
            model=model,
            label_scaler=label_scaler,
            device=config.get('device', 'cuda'),
        )
        
        # 执行评估
        batch_size = config.get('batch_size', 32)
        task_manager.add_log(task_id, f'开始评估，批次大小: {batch_size}')
        
        evaluation_results = evaluator.evaluate(
            test_sequences=test_sequences,
            test_labels_normalized=test_labels_normalized,
            batch_size=batch_size
        )
        
        # 添加评估时间
        evaluation_results['evaluated_at'] = datetime.now().isoformat()
        
        # 保存评估结果到文件
        model_type = config.get('model_type', 'bilstm_gru_regressor')
        model_dir = Path('models') / 'rul_prediction' / model_type / task_id
        try:
            evaluator.save_evaluation_results(
                metrics=evaluation_results,
                save_dir=model_dir,
                task_id=task_id
            )
            task_manager.add_log(task_id, f'评估结果已保存到: {model_dir}')
        except Exception as save_error:
            task_manager.add_log(task_id, f'保存评估结果失败: {save_error}')
        
        task_manager.add_log(
            task_id,
            f'评估完成 - RMSE: {evaluation_results["rmse"]:.6f}, '
            f'MAE: {evaluation_results["mae"]:.6f}, '
            f'R²: {evaluation_results["r2"]:.6f}'
        )
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"RUL模型评估失败: {e}", exc_info=True)
        raise


def _create_sequences(data: np.ndarray, labels: np.ndarray, sequence_length: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建滑动窗口序列
    
    Args:
        data: 数据数组 (N, features)
        labels: 标签数组 (N,)
        sequence_length: 序列长度
        stride: 步长
        
    Returns:
        sequences: (n_samples, sequence_length, features)
        sequence_labels: (n_samples,)
    """
    sequences = []
    sequence_labels = []
    
    for i in range(0, len(data) - sequence_length + 1, stride):
        sequences.append(data[i:i + sequence_length])
        # 使用序列最后一个点的RUL标签
        sequence_labels.append(labels[i + sequence_length - 1])
    
    if len(sequences) == 0:
        return np.array([]).reshape(0, sequence_length, data.shape[1]), np.array([])
    
    return np.array(sequences), np.array(sequence_labels)


def _run_real_training(task_id):
    """执行真实的训练过程"""
    try:
        # 辅助函数：转换布尔值
        def _to_bool(value, default=False):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ('true', '1', 'yes', 'y')
            return bool(value)

        def _to_int(value, default):
            try:
                if value is None or value == '':
                    return default
                return int(value)
            except (TypeError, ValueError):
                return default

        def _to_float(value, default):
            try:
                if value is None or value == '':
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default

        def _parse_int_list(value, fallback):
            if value is None:
                return fallback
            if isinstance(value, (list, tuple)):
                parsed = []
                for item in value:
                    try:
                        parsed.append(int(item))
                    except (TypeError, ValueError):
                        continue
                return parsed or fallback
            if isinstance(value, str):
                parts = [part.strip() for part in value.split(',') if part.strip()]
                parsed = []
                for part in parts:
                    try:
                        parsed.append(int(part))
                    except ValueError:
                        continue
                return parsed or fallback
            return fallback

        def _prepare_model_params(config, model_type):
            if model_type == 'bilstm_gru_regressor':
                params = {
                    'rnn_type': (config.get('rnn_type') or 'lstm').lower(),
                    'hidden_units': _to_int(config.get('hidden_units'), 128),
                    'num_layers': _to_int(config.get('num_layers'), 2),
                    'dropout': _to_float(config.get('dropout'), 0.3),
                    'activation': config.get('activation', 'relu'),
                    'bidirectional': _to_bool(config.get('bidirectional'), True),
                    'use_attention': _to_bool(config.get('use_attention'), True),
                    'use_layer_norm': _to_bool(config.get('use_layer_norm'), True),
                }
            elif model_type == 'cnn_1d_regressor':
                conv_channels = _parse_int_list(config.get('conv_channels'), None)
                if not conv_channels:
                    base_filters = _to_int(config.get('num_filters'), 128)
                    num_layers = _to_int(config.get('num_conv_layers'), 3)
                    conv_channels = [max(base_filters * (2 ** i), 16) for i in range(max(num_layers, 1))]
                kernel_sizes = _parse_int_list(config.get('kernel_sizes'), None)
                if not kernel_sizes:
                    base_kernel = _to_int(config.get('kernel_size'), 5)
                    kernel_sizes = [max(base_kernel, 1)] * len(conv_channels)
                params = {
                    'conv_channels': conv_channels,
                    'kernel_sizes': kernel_sizes,
                    'activation': config.get('activation', 'relu'),
                    'dropout': _to_float(config.get('dropout'), 0.3),
                    'pooling': (config.get('pooling') or 'avg').lower(),
                    'use_batch_norm': _to_bool(config.get('use_batch_norm'), True),
                    'fc_units': _to_int(config.get('fc_units'), 256),
                }
            elif model_type == 'transformer_encoder_regressor':
                embed_dim = _to_int(config.get('embed_dim'), 128)
                num_heads = max(1, _to_int(config.get('num_heads'), 4))
                if embed_dim < num_heads:
                    embed_dim = num_heads * 2
                if embed_dim % num_heads != 0:
                    embed_dim = ((embed_dim // num_heads) + 1) * num_heads
                params = {
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'num_layers': max(1, _to_int(config.get('num_layers'), 3)),
                    'ffn_dim': max(embed_dim * 2, _to_int(config.get('ffn_dim'), 256)),
                    'dropout': _to_float(config.get('dropout'), 0.1),
                    'activation': config.get('activation', 'gelu'),
                    'pooling': (config.get('pooling') or 'avg').lower(),
                    'use_positional_encoding': _to_bool(config.get('use_positional_encoding'), True),
                }
            else:
                params = {}
            # 将归一化后的参数写回config，方便后续日志与任务查询
            for key, value in params.items():
                config[key] = value
            return params
        
        task_manager = get_task_manager()
        task = task_manager.get_task(task_id)
        
        if not task:
            logger.error(f"任务不存在: {task_id}")
            # 尝试更新任务状态为失败（如果任务管理器中有记录）
            try:
                task_manager = get_task_manager()
                task_manager.update_task_status(task_id, 'failed', f'任务不存在: {task_id}')
            except:
                pass
            return
        
        # 优先使用_raw_config（它包含完整的原始数据，包括所有前端发送的参数）
        if hasattr(task, '_raw_config') and task._raw_config is not None:
            # 直接使用_raw_config，它应该包含所有前端发送的参数
            config = task._raw_config.copy()  # 使用副本，避免修改原始数据
            task_manager.add_log(task_id, f'加载训练配置: 来源=_raw_config, 参数数={len(config)}')
        else:
            # 如果没有_raw_config，使用config属性
            config = task.config.copy() if hasattr(task, 'config') else {}
            task_manager.add_log(task_id, f'加载训练配置: 来源=task.config, 参数数={len(config)}')
        
        model_type = config.get('model_type') or getattr(task, 'model_type', 'bilstm_gru_regressor')
        config['model_type'] = model_type
        
        # 检查关键参数是否缺失（仅记录缺失项）
        critical_params = [
            'sequence_length', 'stride', 'random_seed', 'dataset_mode',
            'epochs', 'batch_size', 'learning_rate', 'weight_decay',
            'clip_grad_norm', 'patience', 'early_stop_mode', 'device', 'loss_type'
        ]
        missing_params = [p for p in critical_params if config.get(p) is None]
        if missing_params:
            task_manager.add_log(task_id, f'配置缺少参数: {missing_params}')

        device_target = _normalize_device_target(
            config.get('device') or config.get('device_target') or 'cuda:0'
        )
        config['device'] = device_target
        config['device_target'] = device_target
        # 解析 device_ids（优先使用显式传入，其次从device字符串推断）
        config['device_ids'] = _parse_int_list(config.get('device_ids'), [])
        if config.get('device_ids') is None and isinstance(device_target, str) and ',' in device_target:
            try:
                parsed_ids = []
                for part in device_target.split(','):
                    token = part.strip()
                    if not token:
                        continue
                    if token.startswith('cuda:'):
                        token = token.split(':', 1)[1]
                    if token.isdigit():
                        parsed_ids.append(int(token))
                if parsed_ids:
                    config['device_ids'] = parsed_ids
            except Exception:
                config['device_ids'] = None
        if 'ms' in globals():
            try:
                ms.set_context(mode=ms.GRAPH_MODE, device_target=device_target)
                task_manager.add_log(task_id, f'PyTorch device 已设置: device={device_target}')
            except Exception as ctx_err:
                task_manager.add_log(task_id, f'警告：设置PyTorch设备失败，继续使用默认: {ctx_err}')
        
        task_manager.update_task_status(task_id, 'running', '开始数据处理...')
        task_manager.add_log(task_id, f'开始处理RUL预测训练任务: {task_id}')
        
        # 处理数据
        dataset_info = _process_condition_filtered_data(config, task_id, task_manager, model_type)
        
        # 更新配置中的input_dim（同时更新任务对象的_raw_config）
        config['input_dim'] = dataset_info['input_dim']
        if task and hasattr(task, '_raw_config') and task._raw_config is not None:
            task._raw_config['input_dim'] = dataset_info['input_dim']
        # 同时更新任务对象的input_dim字段
        if task:
            task.input_dim = dataset_info['input_dim']
        
        # 加载数据集
        train_file = dataset_info['train_file']
        val_file = dataset_info.get('val_file')
        test_file = dataset_info.get('test_file')
        
        train_data = np.load(train_file)
        train_sequences = train_data['sequences']
        train_labels = train_data['labels']
        
        if val_file:
            val_data = np.load(val_file)
            val_sequences = val_data['sequences']
            val_labels = val_data['labels']
        else:
            val_sequences = np.array([]).reshape(0, train_sequences.shape[1], train_sequences.shape[2])
            val_labels = np.array([])
        
        task_manager.add_log(task_id, f'数据集加载完成: 训练={len(train_sequences)}, 验证={len(val_sequences)}')
        
        # 构建模型
        seq_len, input_dim = train_sequences.shape[1], train_sequences.shape[2]
        
        model_params = _prepare_model_params(config, model_type)
        task_manager.add_log(task_id, f'{model_type} 模型参数: {model_params}')

        model_specific_required = {
            'bilstm_gru_regressor': ['rnn_type', 'hidden_units', 'num_layers'],
            'cnn_1d_regressor': ['conv_channels', 'kernel_sizes', 'pooling'],
            'transformer_encoder_regressor': ['embed_dim', 'num_heads', 'num_layers'],
        }
        missing_specific = [p for p in model_specific_required.get(model_type, []) if not config.get(p)]
        if missing_specific:
            task_manager.add_log(task_id, f'模型特定参数缺失: {missing_specific}')
        
        model_builder_cls = get_model_builder(model_type)
        model = model_builder_cls.create_model(
            model_type=model_type,
            input_shape=(seq_len, input_dim),
            **model_params
        )
        model_for_save = model
        
        # 验证模型实际使用的RNN类型
        if hasattr(model_for_save, 'rnn_type'):
            task_manager.add_log(task_id, f'模型实际使用的RNN类型: {model_for_save.rnn_type}')
        elif hasattr(model_for_save, 'rnn'):
            rnn_type_name = type(model_for_save.rnn).__name__.lower()
            task_manager.add_log(task_id, f'模型实际使用的RNN类型: {rnn_type_name}')
        
        # 创建训练器
        trainer_cls = get_trainer_class(model_type)
        trainer = trainer_cls(
            model=model,
            learning_rate=_to_float(config.get('learning_rate'), 0.001),
            weight_decay=_to_float(config.get('weight_decay'), 0.0001),
            clip_grad_norm=_to_float(config.get('clip_grad_norm'), 5.0),
            loss_type=config.get('loss_type', 'mse'),
            device=config.get('device', 'cuda:0'),
            device_ids=config.get('device_ids'),
        )
        model_for_save = trainer.get_base_model()
        
        # 记录训练参数
        task_manager.add_log(
            task_id,
            f'训练参数: epochs={config.get("epochs", 50)}, '
            f'batch_size={config.get("batch_size", 32)}, '
            f'learning_rate={config.get("learning_rate", 0.001)}, '
            f'weight_decay={config.get("weight_decay", 0.0001)}, '
            f'clip_grad_norm={config.get("clip_grad_norm", 5.0)}, '
            f'loss_type={config.get("loss_type", "mse")}, '
            f'patience={config.get("patience", 10)}, '
            f'early_stop_mode={config.get("early_stop_mode", "loss")}, '
            f'device={config.get("device", "gpu")}, '
            f'random_seed={config.get("random_seed", 42)}'
        )
        
        # 创建PyTorch DataLoader
        def create_torch_dataloader(sequences, labels, batch_size, shuffle=True, dataset_name=''):
            total_samples = len(sequences)
            drop_last = (total_samples % batch_size) != 0
            if drop_last:
                dropped_samples = total_samples % batch_size
                task_manager.add_log(
                    task_id,
                    f'{dataset_name}数据集: 总样本数={total_samples}, batch_size={batch_size}, '
                    f'将被丢弃的样本数={dropped_samples} ({dropped_samples/total_samples*100:.1f}%)'
                )
            tensor_x = torch.tensor(sequences, dtype=torch.float32)
            tensor_y = torch.tensor(labels, dtype=torch.float32)
            ds = TensorDataset(tensor_x, tensor_y)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        
        batch_size = _to_int(config.get('batch_size'), 32)
        config['batch_size'] = batch_size

        train_dataset = create_torch_dataloader(
            train_sequences, train_labels,
            batch_size=batch_size,
            shuffle=True,
            dataset_name='训练'
        )
        
        val_dataset = None
        if len(val_sequences) > 0:
            val_dataset = create_torch_dataloader(
                val_sequences, val_labels,
                batch_size=batch_size,
                shuffle=False,
                dataset_name='验证'
            )
        
        # 训练模型
        epochs = _to_int(config.get('epochs'), 50)
        patience = _to_int(config.get('patience'), 10)
        early_stop_mode = config.get('early_stop_mode', 'loss')
        config['epochs'] = epochs
        config['patience'] = patience
        
        task_manager.update_task_status(task_id, 'running', '开始模型训练...')
        task_manager.add_log(task_id, f'开始训练，共 {epochs} 个epoch')
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_metrics = trainer.train_epoch(train_dataset)
            
            # 验证
            if val_dataset:
                val_metrics = trainer.validate(val_dataset)
                val_loss = val_metrics['loss']
            else:
                val_loss = train_metrics['loss']
                val_metrics = train_metrics
            
            # 更新任务状态
            progress = int((epoch + 1) / epochs * 100)
            task_manager.update_task_status(task_id, 'running', f'Epoch {epoch+1}/{epochs}')
            
            # 更新任务对象的属性（用于状态查询）
            task = task_manager.get_task(task_id)
            if task:
                task.current_epoch = epoch + 1
                task.current_train_loss = train_metrics['loss']
                task.current_val_loss = val_loss
                task.train_rmse = train_metrics.get('rmse', 0)
                task.train_mae = train_metrics.get('mae', 0)
                task.train_r2 = train_metrics.get('r2', 0)
                task.val_rmse = val_metrics.get('rmse', 0)
                task.val_mae = val_metrics.get('mae', 0)
                task.val_r2 = val_metrics.get('r2', 0)
                task.rmse = val_metrics.get('rmse', 0)
            
            # 同时更新training_tasks字典
            if task_id in training_tasks:
                training_tasks[task_id]['current_epoch'] = epoch + 1
                training_tasks[task_id]['current_train_loss'] = train_metrics['loss']
                training_tasks[task_id]['current_val_loss'] = val_loss
                training_tasks[task_id]['train_rmse'] = train_metrics.get('rmse', 0)
                training_tasks[task_id]['train_mae'] = train_metrics.get('mae', 0)
                training_tasks[task_id]['train_r2'] = train_metrics.get('r2', 0)
                training_tasks[task_id]['val_rmse'] = val_metrics.get('rmse', 0)
                training_tasks[task_id]['val_mae'] = val_metrics.get('mae', 0)
                training_tasks[task_id]['val_r2'] = val_metrics.get('r2', 0)
                training_tasks[task_id]['rmse'] = val_metrics.get('rmse', 0)
            
            # 记录训练损失（评估模式）和训练损失（训练模式，如果可用）
            train_loss_msg = f'Train Loss: {train_metrics["loss"]:.6f}'
            if 'train_loss' in train_metrics:
                train_loss_msg += f' (训练模式: {train_metrics["train_loss"]:.6f})'
            
            task_manager.add_log(
                task_id,
                f'Epoch {epoch+1}/{epochs} - {train_loss_msg}, '
                f'Val Loss: {val_loss:.6f}, Train RMSE: {train_metrics.get("rmse", 0):.6f}, '
                f'Val RMSE: {val_metrics.get("rmse", 0):.6f}'
            )
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                model_dir = Path('models') / 'rul_prediction' / model_type / task_id
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model_for_save.state_dict(), str(model_dir / 'best_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    task_manager.add_log(task_id, f'早停触发，在第 {epoch+1} 个epoch停止训练')
                    break
        
        # 保存最终模型
        model_dir = Path('models') / 'rul_prediction' / model_type / task_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果存在最佳模型，优先使用最佳模型
        best_model_path = model_dir / 'best_model.pt'
        model_path = model_dir / 'model.pt'
        if best_model_path.exists():
            import shutil
            if model_path.exists():
                try:
                    model_path.unlink()
                except Exception as e:
                    task_manager.add_log(task_id, f'警告：删除旧模型文件失败: {e}')
            try:
                shutil.copy2(best_model_path, model_path)
                task_manager.add_log(task_id, '使用最佳模型作为最终模型')
                model_for_save.load_state_dict(torch.load(model_path, map_location='cpu'))
            except Exception as e:
                task_manager.add_log(task_id, f'警告：复制最佳模型失败: {e}，将保存当前模型')
                torch.save(model_for_save.state_dict(), str(model_path))
        else:
            torch.save(model_for_save.state_dict(), str(model_path))
        
        # 保存特征scaler（如果存在）
        import shutil
        scaler_file = dataset_info.get('scaler_file')
        if scaler_file and Path(scaler_file).exists():
            shutil.copy(scaler_file, model_dir / 'scaler.pkl')
            task_manager.add_log(task_id, '特征标准化器已复制到模型目录')
        else:
            task_manager.add_log(task_id, '未找到特征标准化器文件（数据可能已归一化）')
        
        # 保存标签scaler（必须存在）
        label_scaler_file = dataset_info.get('label_scaler_file')
        if label_scaler_file and Path(label_scaler_file).exists():
            shutil.copy(label_scaler_file, model_dir / 'label_scaler.pkl')
            task_manager.add_log(task_id, '标签归一化器已复制到模型目录')
        else:
            task_manager.add_log(task_id, '警告：未找到标签归一化器文件')
        
        # 保存模型配置文件（包含所有训练参数）
        model_config = {
            'model_type': model_type,
            'module': 'rul_prediction',
            'task_id': task_id,
            # 数据参数
            'sequence_length': config.get('sequence_length', 50),
            'stride': config.get('stride', 1),
            'random_seed': config.get('random_seed', 42),
            'dataset_mode': config.get('dataset_mode', 'file_based_validation'),
            'input_dim': config.get('input_dim', dataset_info['input_dim']),
            'conditions': config.get('conditions', []),
            'condition_keys': config.get('condition_keys', []),
            # 模型参数
            'rnn_type': config.get('rnn_type', 'lstm'),
            'hidden_units': config.get('hidden_units', 128),
            'num_layers': config.get('num_layers', 2),
            'dropout': config.get('dropout', 0.3),
            'activation': config.get('activation', 'relu'),
            'bidirectional': config.get('bidirectional', True),
            'use_attention': config.get('use_attention', True),
            'use_layer_norm': config.get('use_layer_norm', True),
            # 训练参数
            'learning_rate': config.get('learning_rate', 0.001),
            'weight_decay': config.get('weight_decay', 0.0001),
            'clip_grad_norm': config.get('clip_grad_norm', 5.0),
            'loss_type': config.get('loss_type', 'mse'),
            'patience': config.get('patience', 10),
            'early_stop_mode': config.get('early_stop_mode', 'loss'),
            'device': config.get('device', 'gpu'),
            'device_target': config.get('device_target', config.get('device', 'CPU')),
            'batch_size': config.get('batch_size', 32),
            # 训练结果
            'epochs_trained': epoch + 1,
            'final_train_loss': train_metrics.get('loss'),
            'final_val_loss': val_loss,
            'final_train_rmse': train_metrics.get('rmse'),
            'final_val_rmse': val_metrics.get('rmse'),
            'final_train_mae': train_metrics.get('mae'),
            'final_val_mae': val_metrics.get('mae'),
            'final_train_r2': train_metrics.get('r2'),
            'final_val_r2': val_metrics.get('r2'),
            'created_at': datetime.now().isoformat()
        }
        config_file = model_dir / 'model_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        
        task_manager.update_task_status(task_id, 'running', '训练完成，正在准备评估...')
        task_manager.add_log(task_id, f'训练完成，模型已保存到: {model_dir}')
        
        # 使用测试集评估模型（如果存在测试集）
        test_file = dataset_info.get('test_file')
        evaluation_done = False
        if test_file and Path(test_file).exists():
            try:
                task_manager.add_log(task_id, '开始使用测试集评估模型...')
                evaluation_results = _evaluate_rul_model(
                    model=model_for_save,
                    test_file=test_file,
                    label_scaler_file=dataset_info.get('label_scaler_file'),
                    config=config,
                    task_id=task_id
                )
                
                # 保存评估结果到任务对象
                task = task_manager.get_task(task_id)
                if task:
                    task.evaluation_results = evaluation_results
                
                # 同时更新training_tasks字典
                if task_id in training_tasks:
                    training_tasks[task_id]['evaluation_results'] = evaluation_results
                    training_tasks[task_id]['evaluation'] = evaluation_results
                
                task_manager.add_log(
                    task_id,
                    f'评估完成 - RMSE: {evaluation_results["rmse"]:.6f}, '
                    f'MAE: {evaluation_results["mae"]:.6f}, '
                    f'R²: {evaluation_results["r2"]:.6f}'
                )
                evaluation_done = True
            except Exception as eval_error:
                logger.error(f"评估过程异常: {eval_error}", exc_info=True)
                task_manager.add_log(task_id, f'评估失败: {str(eval_error)}')
        else:
            task_manager.add_log(task_id, '未找到测试集文件，跳过评估')
            evaluation_done = True

        # 评估完成或被跳过后再更新任务状态为完成
        completion_message = '训练完成'
        if test_file and Path(test_file).exists():
            if evaluation_done:
                completion_message = '训练与评估完成'
            else:
                completion_message = '训练完成（评估失败，详见日志）'
        task_manager.update_task_status(task_id, 'completed', completion_message)
        
    except Exception as e:
        logger.error(f"训练过程异常: {e}", exc_info=True)
        task_manager.update_task_status(task_id, 'failed', f'训练失败: {str(e)}')
        task_manager.add_log(task_id, f'训练失败: {str(e)}')


@rul_prediction_bp.route('/task/<task_id>/status', methods=['GET'])
def get_training_status(task_id):
    """获取训练任务状态"""
    try:
        task_manager = get_task_manager()
        task = task_manager.get_task(task_id)
        
        # 如果从任务管理器获取不到，尝试从training_tasks字典获取
        if not task and task_id in training_tasks:
            task_data = training_tasks[task_id]
            
            # 处理日志格式：确保是列表格式
            task_logs = task_data.get('logs', [])
            if isinstance(task_logs, str):
                # 如果是字符串，按行分割并转换为列表格式
                if task_logs.strip():
                    logs_list = []
                    log_lines = task_logs.strip().split('\n')
                    base_time = None
                    created_at = task_data.get('created_at')
                    if created_at:
                        try:
                            base_time = datetime.fromisoformat(created_at)
                        except (ValueError, TypeError):
                            base_time = None
                    
                    import re
                    for idx, line in enumerate(log_lines):
                        if line.strip():
                            # 尝试从日志消息中解析时间戳
                            log_timestamp = None
                            log_message = line.strip()
                            
                            # 尝试解析标准Python logging格式的时间戳
                            # 格式: YYYY-MM-DD HH:MM:SS,mmm
                            timestamp_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d{3})?)'
                            match = re.search(timestamp_pattern, log_message)
                            if match:
                                try:
                                    timestamp_str = match.group(1).replace(',', '.')
                                    log_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                                    # 移除时间戳部分，保留消息
                                    log_message = re.sub(timestamp_pattern, '', log_message).strip()
                                    # 移除可能的日志级别和模块名
                                    log_message = re.sub(r'^-\s+[\w\.]+\s+-\s+\w+\s+-\s*', '', log_message).strip()
                                except (ValueError, AttributeError):
                                    pass
                            
                            # 如果没有解析到时间戳，使用基准时间+偏移
                            if log_timestamp is None:
                                if base_time:
                                    log_timestamp = base_time + timedelta(seconds=idx)
                                else:
                                    log_timestamp = datetime.now() - timedelta(seconds=len(log_lines) - idx)
                            
                            logs_list.append({
                                'timestamp': log_timestamp.isoformat(),
                                'level': 'info',
                                'message': log_message
                            })
                    task_logs = logs_list
                else:
                    task_logs = []
            elif not isinstance(task_logs, list):
                task_logs = []
            
            # 限制日志数量（只返回最近50条）
            if isinstance(task_logs, list):
                task_logs = task_logs[-50:]
            
            return jsonify({
                'success': True,
                'task': {
                    'id': task_id,
                    'task_id': task_id,
                    'status': task_data.get('status', 'unknown'),
                    'current_epoch': task_data.get('current_epoch', 0),
                    'epoch': task_data.get('current_epoch', 0),
                    'total_epochs': task_data.get('config', {}).get('epochs', 50),
                    'current_train_loss': task_data.get('current_train_loss'),
                    'current_val_loss': task_data.get('current_val_loss'),
                    'loss': task_data.get('current_train_loss'),
                    'val_loss': task_data.get('current_val_loss'),
                    'train_rmse': task_data.get('train_rmse'),
                    'train_mae': task_data.get('train_mae'),
                    'train_r2': task_data.get('train_r2'),
                    'val_rmse': task_data.get('val_rmse'),
                    'val_mae': task_data.get('val_mae'),
                    'val_r2': task_data.get('val_r2'),
                    'rmse': task_data.get('rmse'),
                    'message': task_data.get('message', ''),
                    'logs': task_logs,
                    'evaluation_results': task_data.get('evaluation_results'),
                    'evaluation': task_data.get('evaluation_results'),  # 兼容字段
                    'config': task_data.get('config', {}),
                    'created_at': task_data.get('created_at')
                }
            })
        
        if not task:
            return jsonify({
                'success': False,
                'message': '任务不存在'
            }), 404
        
        # 处理日志格式：将字符串转换为列表
        task_logs = getattr(task, 'logs', [])
        if isinstance(task_logs, str):
            # 如果是字符串，按行分割并转换为列表格式
            if task_logs.strip():
                logs_list = []
                log_lines = task_logs.strip().split('\n')
                base_time = None
                if task.created_at:
                    try:
                        base_time = datetime.fromisoformat(task.created_at)
                    except (ValueError, TypeError):
                        base_time = None
                
                for idx, line in enumerate(log_lines):
                    if line.strip():
                        # 尝试从日志消息中解析时间戳
                        # 格式可能是: "2025-12-02 20:03:41,349 - ..." 或类似格式
                        log_timestamp = None
                        log_message = line.strip()
                        
                        # 尝试解析标准Python logging格式的时间戳
                        # 格式: YYYY-MM-DD HH:MM:SS,mmm
                        import re
                        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d{3})?)'
                        match = re.search(timestamp_pattern, log_message)
                        if match:
                            try:
                                timestamp_str = match.group(1).replace(',', '.')
                                # 尝试解析时间戳
                                log_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                                # 移除时间戳部分，保留消息
                                log_message = re.sub(timestamp_pattern, '', log_message).strip()
                                # 移除可能的日志级别和模块名（如 "- common.task_manager - INFO -"）
                                log_message = re.sub(r'^-\s+[\w\.]+\s+-\s+\w+\s+-\s*', '', log_message).strip()
                            except (ValueError, AttributeError):
                                pass
                        
                        # 如果没有解析到时间戳，使用基准时间+偏移
                        if log_timestamp is None:
                            if base_time:
                                log_timestamp = base_time + timedelta(seconds=idx)
                            else:
                                log_timestamp = datetime.now() - timedelta(seconds=len(log_lines) - idx)
                        
                        logs_list.append({
                            'timestamp': log_timestamp.isoformat(),
                            'level': 'info',
                            'message': log_message
                        })
                task_logs = logs_list
            else:
                task_logs = []
        elif not isinstance(task_logs, list):
            task_logs = []
        
        # 限制日志数量（只返回最近50条）
        if isinstance(task_logs, list):
            task_logs = task_logs[-50:]
        
        return jsonify({
            'success': True,
            'task': {
                'id': task.task_id,
                'task_id': task.task_id,
                'status': task.status,
                'current_epoch': getattr(task, 'current_epoch', 0),
                'epoch': getattr(task, 'current_epoch', 0),
                'total_epochs': task.config.get('epochs', 50),
                'current_train_loss': getattr(task, 'current_train_loss', None),
                'current_val_loss': getattr(task, 'current_val_loss', None),
                'loss': getattr(task, 'current_train_loss', None),
                'val_loss': getattr(task, 'current_val_loss', None),
                'train_rmse': getattr(task, 'train_rmse', None),
                'train_mae': getattr(task, 'train_mae', None),
                'train_r2': getattr(task, 'train_r2', None),
                'val_rmse': getattr(task, 'val_rmse', None),
                'val_mae': getattr(task, 'val_mae', None),
                'val_r2': getattr(task, 'val_r2', None),
                'rmse': getattr(task, 'rmse', None),
                'message': getattr(task, 'message', ''),
                'logs': task_logs,
                'evaluation_results': getattr(task, 'evaluation_results', None),
                'evaluation': getattr(task, 'evaluation_results', None),  # 兼容字段
                'config': task.config,
                'created_at': getattr(task, 'created_at', None)
            }
        })
        
    except Exception as e:
        logger.error(f"获取训练状态失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'获取训练状态失败: {str(e)}'
        }), 500


@rul_prediction_bp.route('/download_model/<task_id>', methods=['GET'])
def download_model(task_id):
    """下载训练好的模型（整个文件夹打包为zip）"""
    try:
        from flask import send_file
        import zipfile
        import tempfile

        # 扫描模型类型目录，找到包含该 task_id 的目录
        model_type_dirs = AVAILABLE_MODEL_TYPES or ['bilstm_gru_regressor']
        models_base = Path('models') / 'rul_prediction'
        models_dir = None
        model_dir_name = None
        
        for type_dir in model_type_dirs:
            potential_dir = models_base / type_dir / task_id
            if potential_dir.exists():
                models_dir = potential_dir
                model_dir_name = type_dir
                break
        
        # 如果扫描未找到，尝试从 training_tasks 获取
        if models_dir is None:
            model_type = 'bilstm_gru_regressor'  # 默认值
            
            if training_tasks and task_id in training_tasks:
                config = training_tasks[task_id].get('config', {})
                model_type = config.get('model_type', 'bilstm_gru_regressor')
            else:
                try:
                    task_manager = get_task_manager()
                    task = task_manager.get_task(task_id)
                    if task and task.config:
                        model_type = task.config.get('model_type', 'bilstm_gru_regressor')
                except Exception as e:
                    logger.warning(f"无法从任务管理器获取模型类型: {e}")

            model_dir_name = model_type
            models_dir = models_base / model_dir_name / task_id
        
        logger.info(f"尝试下载模型目录: {models_dir}, 目录名: {model_dir_name}")

        if not models_dir.exists():
            logger.error(f"模型目录不存在: {models_dir}")
            return jsonify({
                'success': False,
                'message': f'模型目录不存在: {models_dir}'
            }), 404

        # 创建临时zip文件
        zip_filename = f'rul_prediction_model_{task_id}.zip'
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip_path = temp_zip.name
        temp_zip.close()
        
        # 将整个模型目录打包为zip
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in models_dir.rglob('*'):
                if file_path.is_file():
                    # 保持目录结构，使用 task_id 作为根目录名
                    arcname = str(file_path.relative_to(models_dir.parent))
                    zipf.write(file_path, arcname)
                    logger.info(f"已添加文件到zip: {arcname}")
        
        logger.info(f"模型打包完成: {temp_zip_path}")

        # 返回zip文件
        return send_file(
            temp_zip_path,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )

    except Exception as e:
        logger.error(f"下载模型失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'下载模型失败: {str(e)}'
        }), 500

