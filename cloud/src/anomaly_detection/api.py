"""
异常检测API模块
处理异常检测相关的训练和推理请求
"""

from flask import Blueprint, request, jsonify, send_file
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
import sys
import os

# 创建异常检测Blueprint
anomaly_detection_bp = Blueprint('anomaly_detection', __name__, url_prefix='/api/anomaly_detection')

# 添加项目路径以导入训练模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入真实的训练组件
try:
    # LSTM Predictor模块（原有）
    from anomaly_detection.core.lstm_predicton.data_processor import DataProcessor as LSTMPredictorDataProcessor
    from anomaly_detection.core.lstm_predicton.model_builder import ModelBuilder as LSTMPredictorModelBuilder
    from anomaly_detection.core.lstm_predicton.trainer import Trainer as LSTMPredictorTrainer
    from anomaly_detection.core.lstm_predicton.threshold_calculator import ThresholdCalculator as LSTMPredictorThresholdCalculator
    
    # LSTM Autoencoder模块（新增）
    from anomaly_detection.core.lstm_autoencoder.data_processor import DataProcessor as LSTMAutoencoderDataProcessor
    from anomaly_detection.core.lstm_autoencoder.model_builder import ModelBuilder as LSTMAutoencoderModelBuilder
    from anomaly_detection.core.lstm_autoencoder.trainer import Trainer as LSTMAutoencoderTrainer
    from anomaly_detection.core.lstm_autoencoder.threshold_calculator import ThresholdCalculator as LSTMAutoencoderThresholdCalculator
    
    # 1D CNN Autoencoder模块（新增）
    from anomaly_detection.core.cnn_1d_autoencoder.data_processor import DataProcessor as CNN1DAutoencoderDataProcessor
    from anomaly_detection.core.cnn_1d_autoencoder.model_builder import ModelBuilder as CNN1DAutoencoderModelBuilder
    from anomaly_detection.core.cnn_1d_autoencoder.trainer import Trainer as CNN1DAutoencoderTrainer
    from anomaly_detection.core.cnn_1d_autoencoder.threshold_calculator import ThresholdCalculator as CNN1DAutoencoderThresholdCalculator
    
    import torch
    training_available = True
    logger = logging.getLogger(__name__)
    logger.info("Real training modules loaded successfully (LSTM Predictor + LSTM Autoencoder + 1D CNN Autoencoder) [PyTorch]")
except ImportError as e:
    training_available = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Training modules not available: {e}")

# 导入任务管理器
try:
    # 尝试相对导入
    from ..common.task_manager import get_task_manager, TrainingTask, TrainingStatus
except ImportError:
    # 相对导入失败时使用绝对导入
    from common.task_manager import get_task_manager, TrainingTask, TrainingStatus

# 数据文件存储
uploaded_data_files = {}  # 存储上传的数据文件信息


def _normalize_device_target(device_value):
    """标准化设备类型字符串，确保PyTorch识别"""
    if not device_value:
        return 'cpu'
    normalized = str(device_value).strip().lower()
    if normalized in ('gpu', 'cuda'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cpu'


def _get_torch_device(device_target: str = None) -> torch.device:
    """获取PyTorch设备"""
    if device_target is None:
        device_target = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_target)

@anomaly_detection_bp.route('/upload_data', methods=['POST'])
def upload_training_data():
    """接收边端上传的训练数据，保存到云端训练数据目录"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # 获取task_id（如果提供）
        task_id = request.form.get('task_id', '').strip()
        
        if task_id:
            # 保存到task_id对应的目录
            training_data_dir = Path('data') / 'ad' / task_id
            training_data_dir.mkdir(parents=True, exist_ok=True)
            # 使用原始文件名（不添加时间戳，因为已经在task_id目录下）
            filename = file.filename
            file_path = training_data_dir / filename
        else:
            # 兼容旧模式：保存到通用目录
            training_data_dir = Path('data') / 'ad'
            training_data_dir.mkdir(parents=True, exist_ok=True)
            # 生成带时间戳的文件名以避免冲突
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_name = Path(file.filename)
            filename = f"{timestamp}_{original_name.stem}{original_name.suffix}"
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
            'task_id': task_id if task_id else None
        }
        
        logger.debug(f"数据文件上传: {filename} ({file_path.stat().st_size} bytes)")
        
        return jsonify({
            'success': True,
            'original_filename': file.filename,
            'saved_filename': filename,
            'message': 'Training data uploaded successfully to cloud'
        })
        
    except Exception as e:
        logger.error(f"Data upload failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 辅助函数
def _notify_edge_model_ready(task_id: str, model_type: str):
    """通知Edge端模型已就绪，可以下载"""
    try:
        import requests
        
        # 这里可以通过API通知Edge端，或者简单地记录日志
        # 在实际部署中，可以配置Edge端地址进行主动通知
        # 或者Edge端定期轮询Cloud端获取新模型
        
        logger.info(f"模型已就绪，等待Edge端下载: {task_id} (类型: {model_type})")
        
        # 如果知道Edge端地址，可以主动通知：
        # edge_url = os.getenv('EDGE_SERVICE_URL')  # 例如 http://edge-device:5000
        # if edge_url:
        #     try:
        #         response = requests.post(f"{edge_url}/api/models/notification", 
        #                                json={'task_id': task_id, 'model_type': model_type},
        #                                timeout=5)
        #         logger.info(f"已通知Edge端: {response.status_code}")
        #     except Exception as e:
        #         logger.warning(f"通知Edge端失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"通知Edge端模型就绪失败: {e}")
        return False

def _create_inference_task_dir(model_type: str) -> Path:
    """创建推理任务目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    counter = 1
    
    base_dir = Path(f'models/anomaly_detection/inference_tasks')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    while True:
        task_id = f"{timestamp}_{counter:03d}"
        inference_dir = base_dir / f'inference_{task_id}'
        if not inference_dir.exists():
            inference_dir.mkdir(parents=True, exist_ok=True)
            return inference_dir
        counter += 1

def _save_inference_config(inference_dir: Path, data: dict, model_artifacts: dict):
    """保存推理任务配置"""
    config = {
        'task_id': inference_dir.name.replace('inference_', ''),
        'task_type': 'inference',
        'model_type': data.get('model_type', 'lstm_predictor'),
        'module': 'anomaly_detection',
        'created_at': datetime.now().isoformat(),
        
        # 模型信息
        'source_model_dir': str(model_artifacts['model_dir']),
        'source_task_id': model_artifacts.get('task_id'),
        'model_path': str(model_artifacts['model_path']),
        'scaler_path': str(model_artifacts.get('scaler_path', '')),
        'threshold_path': str(model_artifacts.get('threshold_path', '')),
        
        # 推理配置
        'sequence_length': model_artifacts.get('sequence_length'),
        'batch_size': data.get('batch_size', 32),
        
        # 数据信息
        'data_path': data.get('data_path', ''),
        'label_column': data.get('label_column'),
        
        # 阈值信息
        'threshold_value': model_artifacts.get('threshold_value'),
        'threshold_meta': model_artifacts.get('threshold_meta', {})
    }
    
    config_path = inference_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config_path

def _save_inference_results(inference_dir: Path, result: dict, data_info: dict = None):
    """保存推理结果"""
    # 保存主要结果
    results_summary = {
        'success': result['success'],
        'model_type': result['model_type'],
        'total_samples': result['total_samples'],
        'anomalies_detected': result['anomalies_detected'],
        'anomaly_percentage': result['anomaly_percentage'],
        'threshold': result['threshold'],
        'threshold_source': result.get('threshold_source'),
        'statistics': result['statistics'],
        'sequence_length': result['sequence_length'],
        'input_dim': result['input_dim'],
        'completed_at': datetime.now().isoformat()
    }
    
    # 如果有性能指标，包含进去
    if 'performance_metrics' in result:
        results_summary['performance_metrics'] = result['performance_metrics']
    
    # 保存摘要结果
    summary_path = inference_dir / 'results_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    # 保存详细结果（大数据量）
    detailed_results = {
        'residual_scores': result['residual_scores'],
        'anomaly_mask': result['anomaly_mask'],
        'predictions': result['predictions']
    }
    
    # 使用numpy格式保存大数据
    np.savez_compressed(
        inference_dir / 'detailed_results.npz',
        **detailed_results
    )
    
    # 保存数据集信息（如果提供）
    if data_info:
        data_info_path = inference_dir / 'data_info.json'
        with open(data_info_path, 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=2, ensure_ascii=False)
    
    print(f"📁 推理结果已保存到: {inference_dir}")
    print(f"  - 配置文件: config.json")
    print(f"  - 结果摘要: results_summary.json")
    print(f"  - 详细结果: detailed_results.npz")
    
    return inference_dir

def _load_model_config(model_dir: Path) -> dict:
    """加载模型配置文件"""
    config_path = model_dir / 'config.json'
    if not config_path.exists():
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load model config from {config_path}: {exc}")
        return {}

# API路由

@anomaly_detection_bp.route('/training', methods=['POST'])
def create_training():
    """创建异常检测训练任务"""
    if not training_available:
        return jsonify({
            'success': False,
            'error': 'Training functionality not available'
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    # 验证异常检测模型类型
    valid_models = ['lstm_predictor', 'cnn_autoencoder', 'cnn_1d_autoencoder', 'lstm_autoencoder']
    model_type = data.get('model_type', 'lstm_predictor')
    if model_type not in valid_models:
        return jsonify({
            'success': False,
            'error': f'Invalid model_type for anomaly_detection. Must be one of: {", ".join(valid_models)}'
        }), 400

    data['model_type'] = model_type
    data['module'] = 'anomaly_detection'

    # 详细日志：显示接收到的数据
    train_files = data.get('train_files', [])
    test_files = data.get('test_files', [])
    logger.info(f"Cloud端create_training接收到的数据 - 训练文件列表: {train_files}")
    logger.info(f"Cloud端create_training接收到的数据 - 测试文件列表: {test_files}")
    print(f"🔍 Cloud端create_training接收到的数据 - 训练文件列表: {train_files}")
    print(f"🔍 Cloud端create_training接收到的数据 - 测试文件列表: {test_files}")

    try:
        # 使用任务管理器创建任务
        task_manager = get_task_manager()
        task = task_manager.create_task(data)
        
        # 验证任务保存的config
        saved_train_files = task.config.get('train_files', [])
        saved_test_files = task.config.get('test_files', [])
        logger.info(f"Cloud端任务保存的config - 训练文件列表: {saved_train_files}")
        logger.info(f"Cloud端任务保存的config - 测试文件列表: {saved_test_files}")
        print(f"🔍 Cloud端任务保存的config - 训练文件列表: {saved_train_files}")
        print(f"🔍 Cloud端任务保存的config - 测试文件列表: {saved_test_files}")

        # 启动异步训练
        task_manager.start_training(task.task_id, _run_real_training)

        logger.info(f"训练任务已创建: {task.task_id}")
        return jsonify({
            'success': True,
            'message': 'Anomaly detection training task created',
            'task_id': task.task_id,
            'model_type': model_type,
            'module': 'anomaly_detection'
        })

    except Exception as e:
        logger.error(f"Failed to create training task: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to create training task: {str(e)}'
        }), 500

def _process_condition_filtered_data(config, task_id, task_manager, model_type):
    """
    处理工况筛选模式的数据：
    1. 读取多个训练文件
    2. 从元数据文件（.json）读取工况信息（tags_condition），而不是从文件名
    3. 将工况信息添加到特征维度（每个样本都添加相同的工况值）
    4. 对每个文件划分训练/验证集（验证集取最后val_ratio比例，保持时间连续性）
    5. 收集所有文件的训练数据，统一fit scaler（只基于训练数据）
    6. 对每个文件的训练集和验证集分别标准化
    7. 对每个文件的训练集和验证集分别创建滑动窗口
    8. 合并所有文件的训练窗口和验证窗口
    9. 保存为train.npz, dev.npz, test.npz
    """
    import pickle
    from sklearn.preprocessing import StandardScaler
    
    train_files = config.get('train_files', [])
    test_files = config.get('test_files', [])
    conditions = config.get('conditions', {})  # {key: [value1, value2, ...]}
    validation_split = config.get('validation_split', 0.2)
    sequence_length = config.get('sequence_length', 50)
    stride = config.get('stride', 1)
    
    # 详细日志：显示接收到的文件列表
    logger.info(f"接收到的训练文件列表: {train_files}")
    logger.info(f"接收到的测试文件列表: {test_files}")
    
    # 如果文件同时出现在train_files和test_files中，优先将其视为测试文件，从train_files中移除
    if train_files and test_files:
        train_files_set = set(train_files)
        test_files_set = set(test_files)
        overlap = train_files_set & test_files_set
        if overlap:
            logger.warning(f"发现文件同时出现在训练和测试列表中，将从训练列表中移除: {overlap}")
            task_manager.add_log(task_id, f'警告: 发现 {len(overlap)} 个文件同时出现在训练和测试列表中，将从训练列表中移除: {list(overlap)}')
            train_files = [f for f in train_files if f not in overlap]
            config['train_files'] = train_files
    
    task_manager.add_log(task_id, f'工况筛选模式: {len(train_files)} 个训练文件, {len(test_files)} 个测试文件')
    task_manager.add_log(task_id, f'训练文件列表: {train_files}')
    task_manager.add_log(task_id, f'测试文件列表: {test_files}')
    
    # 查找数据文件目录
    training_data_dir = Path('data') / 'ad' / task_id
    training_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取工况key列表（用于添加特征）
    # 如果没有选择工况，则不添加工况特征
    condition_keys = sorted(list(conditions.keys())) if conditions else []
    if condition_keys:
        task_manager.add_log(task_id, f'工况特征: {", ".join(condition_keys)}')
    else:
        task_manager.add_log(task_id, '未选择工况，将不添加工况特征')
    
    # 第一步：处理每个训练文件（添加工况、划分训练/验证集）
    all_train_raw_data = []  # 收集所有训练数据（用于fit scaler）
    file_data_list = []  # 保存每个文件的处理信息（包含train_data和val_data）
    
    for filename in train_files:
        # task_manager.add_log(task_id, f'读取训练文件: {filename}')  # 减少日志
        
        # 查找文件
        file_path = training_data_dir / filename
        if not file_path.exists():
            if filename in uploaded_data_files:
                file_path = Path(uploaded_data_files[filename]['path'])
            else:
                task_manager.add_log(task_id, f'警告: 文件未找到 {filename}，跳过')
                continue
        
        # 读取数据文件
        df = pd.read_csv(file_path)
        
        # 按时间排序（如果存在时间戳列）
        time_col = None
        for col in df.columns:
            if col.lower() in ['timestamp', 'time', '时间']:
                time_col = col
                break
        
        if time_col:
            try:
                # 尝试将时间列转换为数值或datetime类型
                if pd.api.types.is_numeric_dtype(df[time_col]):
                    df = df.sort_values(by=time_col)
                else:
                    # 尝试转换为datetime
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    df = df.sort_values(by=time_col)
                task_manager.add_log(task_id, f'文件 {filename} 已按时间排序')
            except Exception as e:
                task_manager.add_log(task_id, f'警告: 时间排序失败 {filename}: {e}，使用原始顺序')
        
        # 获取数值列（排除时间戳列）
        numeric_cols = [col for col in df.columns 
                       if pd.api.types.is_numeric_dtype(df[col]) 
                       and col.lower() not in ['timestamp', 'time', '时间']]
        
        if not numeric_cols:
            task_manager.add_log(task_id, f'警告: 文件 {filename} 没有数值列，跳过')
            continue
        
        # 从元数据文件读取工况信息（不从文件名提取）
        meta_file_path = training_data_dir / (filename.replace('.csv', '.json'))
        condition_values = {}
        
        if meta_file_path.exists():
            try:
                with open(meta_file_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    tags_condition = meta_data.get('tags_condition', [])
                    for cond in tags_condition:
                        if isinstance(cond, dict) and 'key' in cond and 'value' in cond:
                            key = cond['key']
                            if key in condition_keys:
                                condition_values[key] = float(cond['value'])
                # task_manager.add_log(task_id, f'文件 {filename} 工况值: {condition_values}')  # 减少日志
            except Exception as e:
                task_manager.add_log(task_id, f'警告: 读取元数据失败 {filename}: {e}，工况值将使用默认值0.0')
        else:
            task_manager.add_log(task_id, f'警告: 未找到元数据文件 {meta_file_path}，工况值将使用默认值0.0')
        
        # 提取特征数据
        feature_data = df[numeric_cols].values.astype(np.float32)
        
        # 添加工况特征（如果选择了工况）
        if condition_keys:
            for key in condition_keys:
                value = condition_values.get(key, 0.0)
                condition_feature = np.full((len(feature_data), 1), value, dtype=np.float32)
                feature_data = np.hstack([feature_data, condition_feature])
        
        # 划分训练/验证集（验证集取最后val_ratio比例，保持时间连续性）
        n_samples = len(feature_data)
        val_len = int(n_samples * validation_split)
        train_data = feature_data[:-val_len] if val_len > 0 else feature_data
        val_data = feature_data[-val_len:] if val_len > 0 else np.array([]).reshape(0, feature_data.shape[1])
        
        task_manager.add_log(task_id, f'文件 {filename}: 总样本数={n_samples}, 训练集={len(train_data)}, 验证集={len(val_data)}')
        
        # 保存文件信息
        file_data_list.append({
            'filename': filename,
            'train_data': train_data,
            'val_data': val_data
        })
        
        # 收集所有训练数据（用于fit scaler）
        all_train_raw_data.append(train_data)
    
    if not all_train_raw_data:
        raise ValueError("没有训练数据")
    
    # 第二步：统一fit scaler（只基于所有文件的训练数据）
    all_train_raw = np.vstack(all_train_raw_data)
    scaler = StandardScaler()
    scaler.fit(all_train_raw)
    task_manager.add_log(task_id, f'Scaler已fit（基于所有训练数据），特征维度: {all_train_raw.shape[1]}')
    
    # 第三步：对每个文件的训练集和验证集分别标准化、创建窗口
    all_train_sequences = []
    all_val_sequences = []
    all_train_targets = []
    all_val_targets = []
    
    # 根据模型类型选择不同的窗口创建函数
    if model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']:
        # 自编码器：targets = inputs（重构自己）
        def create_sequences(data):
            if len(data) < sequence_length:
                return np.array([]).reshape(0, sequence_length, data.shape[1]), np.array([]).reshape(0, sequence_length, data.shape[1])
            sequences = []
            targets = []
            for start in range(0, len(data) - sequence_length + 1, stride):
                end = start + sequence_length
                seq = data[start:end]
                sequences.append(seq)
                targets.append(seq.copy())  # 自编码器：target = input
            return (np.stack(sequences) if sequences else np.array([]).reshape(0, sequence_length, data.shape[1]),
                    np.stack(targets) if targets else np.array([]).reshape(0, sequence_length, data.shape[1]))
    else:
        # LSTM预测模型：targets = future（预测未来值）
        prediction_horizon = config.get('prediction_horizon', 1)
        def create_sequences(data):
            if len(data) < sequence_length + prediction_horizon:
                return np.array([]).reshape(0, sequence_length, data.shape[1]), np.array([]).reshape(0, prediction_horizon, data.shape[1])
            sequences = []
            targets = []
            for start in range(0, len(data) - sequence_length - prediction_horizon + 1, stride):
                end = start + sequence_length
                seq = data[start:end]  # past: [x_t, ..., x_{t+L-1}]
                future = data[end:end + prediction_horizon]  # future: [x_{t+L}, ..., x_{t+L+H-1}]
                sequences.append(seq)
                targets.append(future)
            return (np.stack(sequences) if sequences else np.array([]).reshape(0, sequence_length, data.shape[1]),
                    np.stack(targets) if targets else np.array([]).reshape(0, prediction_horizon, data.shape[1]))
    
    for file_info in file_data_list:
        # task_manager.add_log(task_id, f'处理文件窗口: {file_info["filename"]}')  # 减少日志
        
        # 对训练集和验证集分别标准化
        train_data_scaled = scaler.transform(file_info['train_data'])
        val_data_scaled = scaler.transform(file_info['val_data']) if len(file_info['val_data']) > 0 else np.array([]).reshape(0, train_data_scaled.shape[1])
        
        # 对训练集和验证集分别创建滑动窗口
        train_seqs, train_tgts = create_sequences(train_data_scaled)
        val_seqs, val_tgts = create_sequences(val_data_scaled)
        
        if len(train_seqs) > 0:
            all_train_sequences.append(train_seqs)
            all_train_targets.append(train_tgts)
        if len(val_seqs) > 0:
            all_val_sequences.append(val_seqs)
            all_val_targets.append(val_tgts)
    
    # 合并所有文件的窗口
    if not all_train_sequences:
        raise ValueError("没有生成任何训练序列")
    
    train_sequences = np.vstack(all_train_sequences)
    train_targets = np.vstack(all_train_targets)
    
    # 验证集的targets维度需要根据模型类型确定
    if all_val_sequences:
        val_sequences = np.vstack(all_val_sequences)
        val_targets = np.vstack(all_val_targets)
    else:
        # 根据模型类型创建空的验证集
        if model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']:
            val_sequences = np.array([]).reshape(0, sequence_length, train_sequences.shape[2])
            val_targets = np.array([]).reshape(0, sequence_length, train_targets.shape[2])
        else:
            prediction_horizon = config.get('prediction_horizon', 1)
            val_sequences = np.array([]).reshape(0, sequence_length, train_sequences.shape[2])
            val_targets = np.array([]).reshape(0, prediction_horizon, train_targets.shape[2])
    
    task_manager.add_log(task_id, f'训练集: {len(train_sequences)} 个序列')
    task_manager.add_log(task_id, f'验证集: {len(val_sequences)} 个序列')
    
    # 保存数据（统一使用npz格式）
    train_data_path = training_data_dir / 'train.npz'
    dev_data_path = training_data_dir / 'dev.npz'
    test_data_path = training_data_dir / 'test.npz'
    
    np.savez(train_data_path, sequences=train_sequences, targets=train_targets)
    np.savez(dev_data_path, sequences=val_sequences, targets=val_targets)
    
    # 保存scaler
    scaler_path = training_data_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 处理测试集文件（如果有）
    if test_files:
        all_test_sequences = []
        all_test_targets = []
        all_test_labels = []  # 保存每个样本的标签（0=正常，1=异常）
        
        task_manager.add_log(task_id, f'开始处理测试文件列表: {test_files}')
        for filename in test_files:
            task_manager.add_log(task_id, f'开始处理测试文件: {filename}')
            
            file_path = training_data_dir / filename
            if not file_path.exists():
                if filename in uploaded_data_files:
                    file_path = Path(uploaded_data_files[filename]['path'])
                    task_manager.add_log(task_id, f'测试文件 {filename} 在上传文件列表中找到: {file_path}')
                else:
                    task_manager.add_log(task_id, f'警告: 测试文件未找到 {filename}，跳过')
                    logger.warning(f"测试文件未找到: {filename}, 训练数据目录: {training_data_dir}, 上传文件列表: {list(uploaded_data_files.keys())}")
                    continue
            
            task_manager.add_log(task_id, f'测试文件 {filename} 路径: {file_path}')
            
            df = pd.read_csv(file_path)
            
            # 按时间排序（如果存在时间戳列）
            time_col = None
            for col in df.columns:
                if col.lower() in ['timestamp', 'time', '时间']:
                    time_col = col
                    break
            
            if time_col:
                try:
                    if pd.api.types.is_numeric_dtype(df[time_col]):
                        df = df.sort_values(by=time_col)
                    else:
                        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                        df = df.sort_values(by=time_col)
                except Exception as e:
                    task_manager.add_log(task_id, f'警告: 测试文件时间排序失败 {filename}: {e}')
            
            numeric_cols = [col for col in df.columns 
                           if pd.api.types.is_numeric_dtype(df[col]) 
                           and col.lower() not in ['timestamp', 'time', '时间']]
            
            if not numeric_cols:
                task_manager.add_log(task_id, f'警告: 测试文件 {filename} 没有数值列，跳过')
                continue
            
            # 从元数据文件读取工况信息和标签信息
            meta_file_path = training_data_dir / (filename.replace('.csv', '.json'))
            condition_values = {}
            file_label = 0  # 默认正常（0=正常，1=异常）
            
            if meta_file_path.exists():
                try:
                    with open(meta_file_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                        
                        # 读取工况信息
                        tags_condition = meta_data.get('tags_condition', [])
                        for cond in tags_condition:
                            if isinstance(cond, dict) and 'key' in cond and 'value' in cond:
                                key = cond['key']
                                if key in condition_keys:
                                    condition_values[key] = float(cond['value'])
                        
                        # 读取标签信息（从tags_label判断是正常还是异常）
                        tags_label = meta_data.get('tags_label', [])
                        task_manager.add_log(task_id, f'测试文件 {filename} 元数据tags_label: {tags_label}')
                        label_found = False
                        for label_tag in tags_label:
                            if isinstance(label_tag, dict) and 'value' in label_tag:
                                label_value_raw = label_tag['value']
                                label_value = str(label_value_raw).strip().lower()
                                task_manager.add_log(task_id, f'测试文件 {filename} 标签值: "{label_value_raw}" (处理后: "{label_value}")')
                                # 判断是否为异常（可以根据实际标签值调整）
                                if label_value in ['异常', 'anomaly', 'abnormal', '故障', 'fault', '1', 'true']:
                                    file_label = 1  # 异常
                                    label_found = True
                                    task_manager.add_log(task_id, f'测试文件 {filename} 标签: 异常 (从元数据: {label_value_raw})')
                                    break
                                elif label_value in ['正常', 'normal', '健康', 'healthy', '0', 'false']:
                                    file_label = 0  # 正常
                                    label_found = True
                                    task_manager.add_log(task_id, f'测试文件 {filename} 标签: 正常 (从元数据: {label_value_raw})')
                                    break
                        
                        # 如果没有找到标签，尝试从文件名判断（作为后备方案）
                        if not label_found:
                            task_manager.add_log(task_id, f'测试文件 {filename} 未在元数据中找到有效标签，尝试从文件名推断')
                            if '异常' in filename or 'abnormal' in filename.lower() or 'anomaly' in filename.lower():
                                file_label = 1
                                task_manager.add_log(task_id, f'测试文件 {filename} 标签: 异常 (从文件名推断)')
                            else:
                                file_label = 0
                                task_manager.add_log(task_id, f'测试文件 {filename} 标签: 正常 (默认值)')
                except Exception as e:
                    task_manager.add_log(task_id, f'警告: 读取测试元数据失败 {filename}: {e}，工况值和标签将使用默认值')
                    # 如果读取失败，尝试从文件名判断
                    if '异常' in filename or 'abnormal' in filename.lower() or 'anomaly' in filename.lower():
                        file_label = 1
            else:
                task_manager.add_log(task_id, f'警告: 未找到测试元数据文件 {meta_file_path}，尝试从文件名推断标签')
                # 如果元数据文件不存在，尝试从文件名判断
                if '异常' in filename or 'abnormal' in filename.lower() or 'anomaly' in filename.lower():
                    file_label = 1
                    task_manager.add_log(task_id, f'测试文件 {filename} 标签: 异常 (从文件名推断)')
                else:
                    file_label = 0
                    task_manager.add_log(task_id, f'测试文件 {filename} 标签: 正常 (默认值)')
            
            feature_data = df[numeric_cols].values.astype(np.float32)
            n_samples = len(feature_data)
            
            # 为整个文件创建标签数组（所有样本使用相同的文件标签）
            file_labels = np.full(n_samples, file_label, dtype=np.int32)
            
            # 添加工况特征（如果选择了工况）
            if condition_keys:
                for key in condition_keys:
                    value = condition_values.get(key, 0.0)
                    condition_feature = np.full((len(feature_data), 1), value, dtype=np.float32)
                    feature_data = np.hstack([feature_data, condition_feature])
            
            # 在特征数据最后添加标签列（1=异常，0=正常）
            # 注意：标签列不参与标准化，在标准化后再添加
            label_column = file_labels.reshape(-1, 1).astype(np.float32)
            
            # 标准化特征数据（不包括标签列）
            test_data_scaled = scaler.transform(feature_data)
            
            # 标准化后，将标签列添加到特征数据的最后一列
            test_data_scaled = np.hstack([test_data_scaled, label_column])
            
            # 创建滑动窗口（整个文件，不划分训练/验证集）
            # 根据模型类型创建不同的窗口
            if model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']:
                # 自编码器：targets = inputs
                test_seqs, test_tgts = create_sequences(test_data_scaled)
            else:
                # LSTM预测模型：targets = future
                prediction_horizon = config.get('prediction_horizon', 1)
                if len(test_data_scaled) < sequence_length + prediction_horizon:
                    test_seqs = np.array([]).reshape(0, sequence_length, test_data_scaled.shape[1])
                    test_tgts = np.array([]).reshape(0, prediction_horizon, test_data_scaled.shape[1])
                else:
                    test_seqs_list = []
                    test_tgts_list = []
                    for start in range(0, len(test_data_scaled) - sequence_length - prediction_horizon + 1, stride):
                        end = start + sequence_length
                        seq = test_data_scaled[start:end]
                        future = test_data_scaled[end:end + prediction_horizon]
                        test_seqs_list.append(seq)
                        test_tgts_list.append(future)
                    test_seqs = np.stack(test_seqs_list) if test_seqs_list else np.array([]).reshape(0, sequence_length, test_data_scaled.shape[1])
                    test_tgts = np.stack(test_tgts_list) if test_tgts_list else np.array([]).reshape(0, prediction_horizon, test_data_scaled.shape[1])
            
            if len(test_seqs) > 0:
                # 从序列的最后一列提取标签（标签列在标准化后添加到了最后一列）
                # 每个序列的标签取该序列最后一个时间步的标签值
                seq_labels = test_seqs[:, -1, -1].astype(np.int32)  # 取每个序列最后一个时间步的最后一列（标签列）
                
                # 调试信息：检查标签提取是否正确
                normal_count = int(np.sum(seq_labels == 0))
                anomaly_count = int(np.sum(seq_labels == 1))
                task_manager.add_log(task_id, f'测试文件 {filename}: file_label={file_label}, 生成序列数={len(test_seqs)}, 标签统计: 正常={normal_count}, 异常={anomaly_count}')
                logger.info(f"测试文件 {filename}: file_label={file_label}, 序列数={len(test_seqs)}, 正常={normal_count}, 异常={anomaly_count}")
                
                # 从序列和targets中移除标签列（最后一列），因为标签不应该参与模型预测
                # 序列形状: (n_seqs, seq_len, feature_dim+1)，需要移除最后一列
                test_seqs_no_label = test_seqs[:, :, :-1]
                # targets形状也需要移除标签列（如果是自编码器，targets=sequences；如果是预测器，targets=future）
                if model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']:
                    test_tgts_no_label = test_tgts[:, :, :-1]  # 自编码器：targets = sequences
                else:
                    test_tgts_no_label = test_tgts[:, :, :-1]  # 预测器：targets = future（也包含标签列）
                
                all_test_sequences.append(test_seqs_no_label)
                all_test_targets.append(test_tgts_no_label)
                all_test_labels.append(seq_labels)
            else:
                task_manager.add_log(task_id, f'警告: 测试文件 {filename} 没有生成任何序列（数据长度不足）')
                logger.warning(f"测试文件 {filename} 没有生成任何序列")
        
        if all_test_sequences:
            test_sequences = np.vstack(all_test_sequences)
            test_targets = np.vstack(all_test_targets)
            test_labels = np.concatenate(all_test_labels) if all_test_labels else np.array([], dtype=np.int32)
            
            # 计算标签统计
            normal_count = int(np.sum(test_labels == 0))
            anomaly_count = int(np.sum(test_labels == 1))
            total_count = len(test_labels)
            
            # 在终端输出标签统计（使用logger和print双重输出确保可见）
            logger.info(f"📊 测试集标签统计: 总序列数={total_count}, 正常样本={normal_count}, 异常样本={anomaly_count}")
            print(f"📊 测试集标签统计: 总序列数={total_count}, 正常样本={normal_count}, 异常样本={anomaly_count}")
            
            # 调试信息：检查标签统计
            task_manager.add_log(task_id, f'测试集合并前统计: 文件数={len(all_test_labels)}, 总序列数={total_count}')
            for i, labels in enumerate(all_test_labels):
                file_normal = int(np.sum(labels == 0))
                file_anomaly = int(np.sum(labels == 1))
                task_manager.add_log(task_id, f'  文件{i+1}: 序列数={len(labels)}, 正常={file_normal}, 异常={file_anomaly}')
            
            # 保存测试数据，包括标签信息
            np.savez(test_data_path, sequences=test_sequences, targets=test_targets, labels=test_labels)
            task_manager.add_log(task_id, f'测试集: {total_count} 个序列, 正常样本: {normal_count}, 异常样本: {anomaly_count}')
            
            # 额外检查：验证标签值
            unique_labels = np.unique(test_labels)
            task_manager.add_log(task_id, f'测试集标签唯一值: {unique_labels.tolist()}, 标签数据类型: {test_labels.dtype}')
            logger.info(f"测试集标签唯一值: {unique_labels.tolist()}, 标签数据类型: {test_labels.dtype}")
    
    # 创建TimeSeriesData对象
    from anomaly_detection.core.lstm_autoencoder.data_processor import TimeSeriesData
    
    train_data_obj = TimeSeriesData(sequences=train_sequences, targets=train_targets)
    val_data_obj = TimeSeriesData(sequences=val_sequences, targets=val_targets) if len(val_sequences) > 0 else None
    
    # 获取特征维度
    feature_dim = train_sequences.shape[2]
    
    return train_data_obj, val_data_obj, feature_dim


def _run_real_training(task_id):
    """执行真实的训练过程"""
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)
    
    if task is None:
        logger.error(f"Task {task_id} not found in task manager")
        return
    
    config = task.config
    
    # 详细日志：显示从任务获取的config
    train_files_from_task = config.get('train_files', [])
    test_files_from_task = config.get('test_files', [])
    logger.info(f"Cloud端_run_real_training从任务获取的config - 训练文件列表: {train_files_from_task}")
    logger.info(f"Cloud端_run_real_training从任务获取的config - 测试文件列表: {test_files_from_task}")
    print(f"🔍 Cloud端_run_real_training从任务获取的config - 训练文件列表: {train_files_from_task}")
    print(f"🔍 Cloud端_run_real_training从任务获取的config - 测试文件列表: {test_files_from_task}")
    
    try:
        # 获取模型类型（提前获取，用于日志输出）
        model_type = config.get('model_type', 'lstm_predictor')
        
        # 更新任务状态为训练中
        task_manager.update_task_status(task_id, 'training', 'Initializing training pipeline...')
        task_manager.add_log(task_id, f'Starting {model_type} training task')
        
        # 记录关键参数
        logger.info(f"开始训练 {model_type} | 序列长度={config.get('sequence_length', 50)} | "
                   f"隐藏层={config.get('hidden_units', 64)} | 学习率={config.get('learning_rate', 0.001)} | "
                   f"训练轮数={config.get('epochs', 50)}")
        
        # 设置PyTorch设备
        device_target = _normalize_device_target(
            config.get('device_target') or config.get('device') or 'cpu'
        )
        device = _get_torch_device(device_target)
        logger.info(f"PyTorch device initialized: {device}")
        
        # 1. 数据处理
        task_manager.update_task_status(task_id, 'training', 'Loading and preprocessing data...')
        task_manager.add_log(task_id, 'Data preprocessing started')
        
        dataset_mode = config.get('dataset_mode', 'processed_file')
        
        # 工况筛选模式：处理多文件、工况信息、按文件划分
        if dataset_mode == 'condition_filtered':
            # 从config获取train_files和test_files（Edge端发送的）
            train_files = config.get('train_files', [])
            test_files = config.get('test_files', [])
            training_data_dir = Path('data') / 'ad' / task_id
            training_data_dir.mkdir(parents=True, exist_ok=True)
            
            # 详细日志：显示从config获取的文件列表
            logger.info(f"Cloud端_run_real_training从config获取 - 训练文件列表: {train_files}")
            logger.info(f"Cloud端_run_real_training从config获取 - 测试文件列表: {test_files}")
            print(f"🔍 Cloud端_run_real_training从config获取 - 训练文件列表: {train_files}")
            print(f"🔍 Cloud端_run_real_training从config获取 - 测试文件列表: {test_files}")
            
            # 如果Edge端已经发送了train_files和test_files，需要等待文件上传完成
            # 因为Edge端是先创建任务，然后上传文件
            if train_files and len(train_files) > 0:
                task_manager.add_log(task_id, f'等待Edge端上传文件... (训练文件: {len(train_files)} 个, 测试文件: {len(test_files)} 个)')
                logger.info(f"等待Edge端上传文件... (训练文件: {len(train_files)} 个, 测试文件: {len(test_files)} 个)")
                
                # 等待文件上传完成（最多等待60秒，每2秒检查一次）
                import time
                max_wait_time = 60
                wait_interval = 2
                waited_time = 0
                last_file_count = 0
                stable_count = 0  # 文件数量稳定的次数（连续3次不变认为上传完成）
                
                while waited_time < max_wait_time:
                    time.sleep(wait_interval)
                    waited_time += wait_interval
                    
                    # 检查目录中是否有CSV文件
                    if training_data_dir.exists():
                        uploaded_csv_files = [f.name for f in training_data_dir.glob('*.csv') if f.is_file()]
                        uploaded_json_files = [f.name for f in training_data_dir.glob('*.json') if f.is_file()]
                        
                        current_file_count = len(uploaded_csv_files)
                        
                        # 检查所有train_files和test_files是否都已上传
                        all_expected_files = set(train_files + test_files)
                        uploaded_files_set = set(uploaded_csv_files)
                        
                        if all_expected_files.issubset(uploaded_files_set):
                            # 所有文件都已上传，检查元数据文件
                            expected_meta_count = len(all_expected_files)
                            if len(uploaded_json_files) >= expected_meta_count:
                                task_manager.add_log(task_id, f'所有文件已上传完成: {len(uploaded_csv_files)} 个CSV文件, {len(uploaded_json_files)} 个元数据文件')
                                logger.info(f"所有文件已上传完成: {len(uploaded_csv_files)} 个CSV文件, {len(uploaded_json_files)} 个元数据文件")
                                break
                        
                        # 如果文件数量发生变化，重置稳定计数器
                        if current_file_count != last_file_count:
                            stable_count = 0
                            last_file_count = current_file_count
                            if current_file_count > 0:
                                task_manager.add_log(task_id, f'检测到 {current_file_count} 个数据文件已上传，等待上传完成...')
                        else:
                            stable_count += 1
                        
                        # 如果文件数量稳定了3次检查（6秒），认为上传完成
                        if stable_count >= 3 and current_file_count > 0:
                            task_manager.add_log(task_id, f'文件上传完成: {current_file_count} 个CSV文件, {len(uploaded_json_files)} 个元数据文件')
                            logger.info(f"文件上传完成: {current_file_count} 个CSV文件, {len(uploaded_json_files)} 个元数据文件")
                            break
                    
                    if waited_time % 10 == 0:  # 每10秒记录一次
                        task_manager.add_log(task_id, f'等待文件上传中... ({waited_time}/{max_wait_time}秒)')
                
                if waited_time >= max_wait_time:
                    task_manager.add_log(task_id, f'警告: 等待文件上传超时，继续处理已上传的文件')
                    logger.warning(f"等待文件上传超时，继续处理已上传的文件")
            
            # 如果train_files为空，等待文件上传并自动分配
            elif not train_files or len(train_files) == 0:
                task_manager.add_log(task_id, '等待训练文件上传...')
                logger.info(f"等待训练文件上传...")
                
                # 等待文件上传（最多等待60秒，每2秒检查一次）
                import time
                max_wait_time = 60
                wait_interval = 2
                waited_time = 0
                last_file_count = 0
                stable_count = 0  # 文件数量稳定的次数（连续3次不变认为上传完成）
                
                while waited_time < max_wait_time:
                    time.sleep(wait_interval)
                    waited_time += wait_interval
                    
                    # 检查目录中是否有CSV文件
                    if training_data_dir.exists():
                        uploaded_csv_files = [f for f in training_data_dir.glob('*.csv') if f.is_file()]
                        uploaded_json_files = [f for f in training_data_dir.glob('*.json') if f.is_file()]
                        
                        current_file_count = len(uploaded_csv_files)
                        
                        # 如果文件数量发生变化，重置稳定计数器
                        if current_file_count != last_file_count:
                            stable_count = 0
                            last_file_count = current_file_count
                            if current_file_count > 0:
                                task_manager.add_log(task_id, f'检测到 {current_file_count} 个数据文件已上传，等待上传完成...')
                        else:
                            stable_count += 1
                        
                        # 如果有文件，等待元数据文件上传完成
                        if current_file_count > 0:
                            # 等待元数据文件上传（最多再等10秒）
                            if len(uploaded_json_files) < len(uploaded_csv_files) and waited_time < max_wait_time - 10:
                                task_manager.add_log(task_id, f'等待元数据文件上传... (已上传 {len(uploaded_json_files)}/{len(uploaded_csv_files)})')
                                continue
                            
                            # 如果文件数量稳定了3次检查（6秒），认为上传完成
                            if stable_count >= 3:
                                # 从上传的文件中提取文件名
                                all_files = [f.name for f in uploaded_csv_files]
                                potential_train_files = []
                                potential_test_files = []
                                
                                for fname in all_files:
                                    # 检查对应的元数据文件
                                    meta_file = training_data_dir / fname.replace('.csv', '.json')
                                    if meta_file.exists():
                                        try:
                                            with open(meta_file, 'r', encoding='utf-8') as mf:
                                                meta_data = json.load(mf)
                                                tags_label = meta_data.get('tags_label', [])
                                                # 检查标签：正常文件用于训练，异常文件用于测试
                                                is_normal = False
                                                for label_tag in tags_label:
                                                    if isinstance(label_tag, dict) and 'value' in label_tag:
                                                        label_value = label_tag['value']
                                                        if label_value in ['正常', 'normal', '健康', 'healthy']:
                                                            is_normal = True
                                                            break
                                                
                                                if is_normal:
                                                    potential_train_files.append(fname)
                                                else:
                                                    potential_test_files.append(fname)
                                        except Exception as e:
                                            logger.warning(f"读取元数据文件失败 {fname}: {e}")
                                            # 如果无法读取元数据，根据文件名判断
                                            if 'test' not in fname.lower():
                                                potential_train_files.append(fname)
                                            else:
                                                potential_test_files.append(fname)
                                    else:
                                        # 没有元数据文件，根据文件名判断
                                        logger.warning(f"未找到元数据文件: {fname.replace('.csv', '.json')}")
                                        if 'test' not in fname.lower():
                                            potential_train_files.append(fname)
                                        else:
                                            potential_test_files.append(fname)
                                
                                # 只有在train_files为空时才自动分配
                                # 如果Edge端已经发送了train_files和test_files，应该使用Edge端发送的列表
                                if potential_train_files and (not train_files or len(train_files) == 0):
                                    train_files = potential_train_files
                                    config['train_files'] = train_files
                                    task_manager.add_log(task_id, f'自动检测到 {len(train_files)} 个训练文件已上传（根据标签自动分配）')
                                
                                if potential_test_files and (not test_files or len(test_files) == 0):
                                    test_files = potential_test_files
                                    config['test_files'] = test_files
                                    if test_files:
                                        task_manager.add_log(task_id, f'自动检测到 {len(test_files)} 个测试文件已上传（根据标签自动分配）')
                                
                                # 如果Edge端已经发送了train_files和test_files，记录它们
                                if train_files and len(train_files) > 0:
                                    task_manager.add_log(task_id, f'使用Edge端发送的训练文件列表: {len(train_files)} 个文件')
                                if test_files and len(test_files) > 0:
                                    task_manager.add_log(task_id, f'使用Edge端发送的测试文件列表: {len(test_files)} 个文件')
                                
                                logger.info(f"文件上传完成: {len(train_files)} 个训练文件, {len(uploaded_json_files)} 个元数据文件")
                                break
                    
                    if waited_time % 10 == 0:  # 每10秒记录一次
                        task_manager.add_log(task_id, f'等待文件上传中... ({waited_time}/{max_wait_time}秒)')
                
                if not train_files or len(train_files) == 0:
                    raise ValueError("等待超时：训练文件未上传完成，请检查文件上传是否成功")
            
            # 如果 conditions 为空，尝试从元数据文件中提取
            if not config.get('conditions') or len(config.get('conditions', {})) == 0:
                task_manager.add_log(task_id, '从元数据文件中提取工况信息...')
                conditions = {}
                
                # 从所有训练文件的元数据中提取工况key
                for filename in train_files:
                    meta_file_path = training_data_dir / filename.replace('.csv', '.json')
                    if meta_file_path.exists():
                        try:
                            with open(meta_file_path, 'r', encoding='utf-8') as f:
                                meta_data = json.load(f)
                                tags_condition = meta_data.get('tags_condition', [])
                                for cond in tags_condition:
                                    if isinstance(cond, dict) and 'key' in cond:
                                        key = cond['key']
                                        if key not in conditions:
                                            conditions[key] = []
                                        value = cond.get('value', '')
                                        if value and value not in conditions[key]:
                                            conditions[key].append(value)
                        except Exception as e:
                            logger.warning(f"读取元数据文件失败 {filename}: {e}")
                
                if conditions:
                    config['conditions'] = conditions
                    task_manager.add_log(task_id, f'从元数据提取到工况: {list(conditions.keys())}')
                    logger.info(f"从元数据提取到工况: {conditions}")
                else:
                    task_manager.add_log(task_id, '警告: 未找到工况信息，将不使用工况特征')
                    logger.warning("未找到工况信息，将不使用工况特征")
            
            train_data, val_data, feature_dim = _process_condition_filtered_data(
                config, task_id, task_manager, model_type
            )
            
            # 创建processor对象（用于后续保存scaler等操作）
            if model_type == 'lstm_autoencoder':
                processor = LSTMAutoencoderDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    stride=config.get('stride', 1),
                    normalize=True
                )
            elif model_type == 'cnn_1d_autoencoder':
                processor = CNN1DAutoencoderDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    stride=config.get('stride', 1),
                    normalize=True
                )
            else:
                processor = LSTMPredictorDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    prediction_horizon=config.get('prediction_horizon', 1),
                    normalize=True
                )
        else:
            # 原有模式：单个文件处理
            # 获取数据文件路径
            dataset_file = config.get('dataset_file')
            if not dataset_file:
                raise ValueError("No dataset file provided")
            
            # 1. 查找训练数据文件
            data_path = None
            
            # 首先检查是否已经上传到云端训练数据目录
            if dataset_file in uploaded_data_files:
                data_path = Path(uploaded_data_files[dataset_file]['path'])
                task_manager.add_log(task_id, f'Using uploaded training data: {uploaded_data_files[dataset_file]["saved_name"]}')
            else:
                # 尝试在训练数据目录中查找文件 (异常检测: cloud/data/ad)
                training_data_dir = Path('data') / 'ad'
                possible_paths = [
                    training_data_dir / dataset_file,  # 云端异常检测训练数据目录
                    Path('data') / dataset_file,  # 云端通用data目录
                    Path(dataset_file)  # 相对路径
                ]
                
                for path in possible_paths:
                    if path.exists():
                        data_path = path
                        task_manager.add_log(task_id, f'Found training data at: {path}')
                        break
            
            if data_path is None or not data_path.exists():
                # 数据文件未找到，提示需要上传
                task_manager.update_task_status(task_id, 'failed', 'Training data not found, please upload from edge')
                task_manager.add_log(task_id, f'Training data file not found: {dataset_file}')
                task_manager.add_log(task_id, 'Available files: ' + ', '.join(uploaded_data_files.keys()))
                
                raise FileNotFoundError(
                    f"Training data file '{dataset_file}' not available on cloud server. "
                    f"Please upload the data from edge server first. "
                    f"Available files: {list(uploaded_data_files.keys())}"
                )
            
            # 处理数据（根据模型类型选择对应的数据处理器）
            if model_type == 'lstm_autoencoder':
                # 使用LSTM Autoencoder数据处理器
                processor = LSTMAutoencoderDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    stride=config.get('stride', 1),
                    normalize=True
                )
                
                # 处理数据（自编码器输入输出相同）
                train_data, val_data = processor.process_pipeline(
                    str(data_path),
                    train_ratio=config.get('train_ratio', 0.8)
                )
            elif model_type == 'cnn_1d_autoencoder':
                # 使用1D CNN Autoencoder数据处理器
                processor = CNN1DAutoencoderDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    stride=config.get('stride', 1),
                    normalize=True
                )
                
                # 处理数据（自编码器输入输出相同）
                train_data, val_data = processor.process_pipeline(
                    str(data_path),
                    train_ratio=config.get('train_ratio', 0.8)
                )
            else:
                # 使用LSTM Predictor数据处理器（默认）
                processor = LSTMPredictorDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    prediction_horizon=config.get('prediction_horizon', 1),
                    normalize=True
                )
                
                # 计算train_ratio：从validation_split转换为train_ratio
                validation_split = config.get('validation_split', 0.2)  # 默认验证集20%
                train_ratio = 1.0 - validation_split
                logger.info(f"使用validation_split={validation_split}, 计算train_ratio={train_ratio}")
                
                # 处理数据（预测器输入输出不同）
                train_data, val_data = processor.process_pipeline(
                    str(data_path),
                    train_ratio=train_ratio
                )
            
            # 获取特征维度
            feature_dim = train_data.sequences.shape[2]
            # 创建processor对象（用于后续保存scaler等操作）
            if model_type == 'lstm_autoencoder':
                processor = LSTMAutoencoderDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    stride=config.get('stride', 1),
                    normalize=True
                )
            elif model_type == 'cnn_1d_autoencoder':
                processor = CNN1DAutoencoderDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    stride=config.get('stride', 1),
                    normalize=True
                )
            else:
                processor = LSTMPredictorDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    prediction_horizon=config.get('prediction_horizon', 1),
                    normalize=True
                )
        
        task_manager.update_task_status(task_id, 'training', f'Data loaded: {len(train_data.sequences)} training samples')
        task_manager.add_log(task_id, f'Training dataset: {len(train_data.sequences)} samples')
        if val_data:
            task_manager.add_log(task_id, f'Validation dataset: {len(val_data.sequences)} samples')
        
        # 2. 构建模型（根据模型类型选择对应的构建器）
        task_manager.update_task_status(task_id, 'training', 'Building model architecture...')
        task_manager.add_log(task_id, f'Building {model_type} model')
        
        # 获取序列长度（用于日志和模型构建）
        sequence_length = config.get('sequence_length', 50)
        input_shape = (sequence_length, feature_dim)
        
        if model_type == 'lstm_autoencoder':
            # 构建LSTM Autoencoder模型
            model = LSTMAutoencoderModelBuilder.create_model(
                'lstm_autoencoder',
                input_shape=input_shape,
                hidden_size=config.get('hidden_units', 128),
                num_layers=config.get('num_layers', 2),
                bottleneck_dim=config.get('bottleneck_dim', 64),
                dropout=config.get('dropout', 0.1)
            )
        elif model_type == 'cnn_1d_autoencoder':
            # 构建1D CNN Autoencoder模型
            model = CNN1DAutoencoderModelBuilder.create_model(
                'cnn_1d_autoencoder',
                input_shape=input_shape,
                num_filters=config.get('num_filters', 64),
                kernel_size=config.get('kernel_size', 3),
                bottleneck_dim=config.get('bottleneck_dim', 64),
                num_conv_layers=config.get('num_conv_layers', config.get('num_layers', 3)),
                dropout=config.get('dropout', 0.1),
                activation=config.get('activation', 'relu')
            )
        else:
            # 构建LSTM Predictor模型（默认）
            hidden_units = config.get('hidden_units', 128)
            num_layers = config.get('num_layers', 2)
            dropout = config.get('dropout', 0.1)
            activation = config.get('activation', 'tanh')
            
            model = LSTMPredictorModelBuilder.build_lstm_predictor(
                input_shape=input_shape,
                hidden_units=hidden_units,
                num_layers=num_layers,
                dropout=dropout,
                activation=activation
            )
        
        # 3. 训练模型（根据模型类型选择对应的训练器）
        # 获取训练轮数（提前获取，用于日志输出）
        num_epochs = config.get('epochs', 50)
        
        task_manager.update_task_status(task_id, 'training', 'Starting model training...')
        task_manager.add_log(task_id, f'Starting {model_type} model training')
        logger.info(f"开始模型训练，共 {num_epochs} 个epoch")
        
        if model_type == 'lstm_autoencoder':
            trainer = LSTMAutoencoderTrainer(
                model=model,
                learning_rate=config.get('learning_rate', 0.001)
            )
        elif model_type == 'cnn_1d_autoencoder':
            trainer = CNN1DAutoencoderTrainer(
                model=model,
                learning_rate=config.get('learning_rate', 0.001)
            )
        else:
            learning_rate = config.get('learning_rate', 0.001)
            weight_decay = config.get('weight_decay', 0.0001)
            
            trainer = LSTMPredictorTrainer(
                model=model,
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
        
        # 创建PyTorch数据加载器
        from torch.utils.data import TensorDataset, DataLoader
        
        def create_dataloader(sequences, targets, batch_size, shuffle=True):
            """创建PyTorch DataLoader"""
            sequences_tensor = torch.from_numpy(sequences.astype(np.float32))
            targets_tensor = torch.from_numpy(targets.astype(np.float32))
            dataset = TensorDataset(sequences_tensor, targets_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        
        train_dataset = create_dataloader(
            train_data.sequences,
            train_data.targets,
            batch_size=config.get('batch_size', 32),
            shuffle=True
        )
        
        val_dataset = create_dataloader(
            val_data.sequences,
            val_data.targets,
            batch_size=config.get('batch_size', 32),
            shuffle=False
        ) if val_data is not None else None
        
        # 自定义训练循环以支持进度回调
        for epoch in range(num_epochs):
            # 检查是否被取消
            current_task = task_manager.get_task(task_id)
            if current_task and current_task.status == 'cancelled':
                task_manager.update_task_status(task_id, 'cancelled', 'Training was cancelled')
                return
            
            # 训练一个epoch
            train_loss = trainer.train_epoch(train_dataset, epoch)
            
            # 验证（如果有验证集）
            val_loss = None
            if val_dataset is not None:
                val_loss = trainer.validate(val_dataset)
            
            # 记录epoch完成日志（合并为一行）
            epoch_log = f'Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.6f}'
            if val_loss is not None:
                epoch_log += f', Val: {val_loss:.6f}'
            # 每10个epoch或最后一个epoch记录一次
            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.6f}" + (f", Val: {val_loss:.6f}" if val_loss is not None else ""))
            
            # 更新训练进度
            progress = ((epoch + 1) / num_epochs) * 100
            task_manager.update_task_status(
                task_id, 
                'training',
                epoch_log,
                round(progress, 2),
                epoch + 1,
                train_loss,
                val_loss
            )
            task_manager.add_log(task_id, epoch_log)
        
        # 4. 保存模型
        task_manager.update_task_status(task_id, 'training', 'Saving trained model...', current_epoch=num_epochs)
        task_manager.add_log(task_id, f'Saving {model_type} model and artifacts')
        # 保存模型
        
        # 创建模型保存目录 - 根据模型类型选择子目录
        if model_type == 'lstm_autoencoder':
            model_dir = Path(f'models/anomaly_detection/lstm_autoencoder/{task_id}')
        elif model_type == 'cnn_1d_autoencoder':
            model_dir = Path(f'models/anomaly_detection/cnn_1d_autoencoder/{task_id}')
        else:
            model_dir = Path(f'models/anomaly_detection/lstm_prediction/{task_id}')
            
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_path = model_dir / 'model.pth'
        trainer.save_model(str(model_path))
        
        # 保存标准化器（从训练数据目录复制，或从processor获取）
        scaler_path = model_dir / 'scaler.pkl'
        training_data_dir = Path('data') / 'ad' / task_id
        training_scaler_path = training_data_dir / 'scaler.pkl'
        
        if training_scaler_path.exists():
            # 从训练数据目录复制scaler（工况筛选模式）
            import shutil
            shutil.copy2(training_scaler_path, scaler_path)
            logger.info(f"已复制scaler: {training_scaler_path} -> {scaler_path}")
        elif 'processor' in locals() and hasattr(processor, 'scaler') and processor.scaler is not None:
            # 从processor保存scaler（兼容旧模式）
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(processor.scaler, f)
        
        # 保存配置
        config_path = model_dir / 'config.json'
        # 获取sequence_length（从processor或config）
        seq_len = config.get('sequence_length', 50)
        if 'processor' in locals() and hasattr(processor, 'sequence_length'):
            seq_len = processor.sequence_length
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                **config,
                'feature_dim': feature_dim,
                'sequence_length': seq_len,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'trained_at': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        # 5. 在验证集上计算阈值（训练完成后）
        task_manager.update_task_status(task_id, 'training', 'Calculating threshold on validation set...', 90)
        task_manager.add_log(task_id, '开始在验证集上计算阈值')
        
        threshold = None
        threshold_file = model_dir / 'threshold.json'
        
        try:
            # 加载验证集数据
            dev_data_path = training_data_dir / 'dev.npz'
            if dev_data_path.exists():
                dev_data = np.load(dev_data_path, allow_pickle=True)
                dev_sequences = dev_data['sequences']
                dev_targets = dev_data['targets']
                
                if len(dev_sequences) > 0:
                    # 在验证集上进行预测
                    model.eval()
                    dev_predictions = []
                    batch_size = config.get('batch_size', 32)
                    
                    with torch.no_grad():
                        for i in range(0, len(dev_sequences), batch_size):
                            batch_sequences = dev_sequences[i:i+batch_size]
                            batch_tensor = torch.from_numpy(batch_sequences.astype(np.float32)).to(device)
                            pred = model(batch_tensor)
                            dev_predictions.append(pred.cpu().numpy())
                    
                    dev_predictions = np.vstack(dev_predictions)
                    
                    # 计算验证集误差（逐样本计算，避免内存溢出）
                    dev_errors = []
                    for i in range(len(dev_targets)):
                        error = np.mean(np.square(dev_targets[i] - dev_predictions[i]))
                        dev_errors.append(error)
                    dev_errors = np.array(dev_errors)
                    
                    # 根据配置的阈值方法计算阈值
                    threshold_method = config.get('threshold_method', 'percentile')
                    
                    if threshold_method == 'percentile':
                        percentile = config.get('threshold_percentile', 95.0)
                        threshold = float(np.percentile(dev_errors, percentile))
                        task_manager.add_log(task_id, f'使用{percentile}分位数计算阈值: {threshold:.6f}')
                    elif threshold_method == '3-sigma':
                        mean_error = np.mean(dev_errors)
                        std_error = np.std(dev_errors)
                        threshold = float(mean_error + 3 * std_error)
                        task_manager.add_log(task_id, f'使用3-sigma方法计算阈值: {threshold:.6f} (mean={mean_error:.6f}, std={std_error:.6f})')
                    elif threshold_method == 'contamination':
                        contamination = config.get('threshold_contamination', 0.01)
                        threshold = float(np.percentile(dev_errors, (1 - contamination) * 100))
                        task_manager.add_log(task_id, f'使用contamination方法计算阈值: {threshold:.6f} (contamination={contamination})')
                    else:
                        # 默认使用95分位数
                        threshold = float(np.percentile(dev_errors, 95.0))
                        task_manager.add_log(task_id, f'使用默认95分位数计算阈值: {threshold:.6f}')
                    
                    # 保存阈值
                    threshold_data = {
                        'threshold': threshold,
                        'threshold_method': threshold_method,
                        'threshold_params': {
                            'percentile': config.get('threshold_percentile', 95.0) if threshold_method == 'percentile' else None,
                            'contamination': config.get('threshold_contamination', 0.01) if threshold_method == 'contamination' else None,
                        },
                        'validation_error_stats': {
                            'mean': float(np.mean(dev_errors)),
                            'std': float(np.std(dev_errors)),
                            'min': float(np.min(dev_errors)),
                            'max': float(np.max(dev_errors)),
                            'percentiles': {
                                'p50': float(np.percentile(dev_errors, 50)),
                                'p75': float(np.percentile(dev_errors, 75)),
                                'p90': float(np.percentile(dev_errors, 90)),
                                'p95': float(np.percentile(dev_errors, 95)),
                                'p99': float(np.percentile(dev_errors, 99))
                            }
                        },
                        'calculated_at': datetime.now().isoformat()
                    }
                    
                    with open(threshold_file, 'w', encoding='utf-8') as f:
                        json.dump(threshold_data, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f'阈值已保存: {threshold}')
                    task_manager.add_log(task_id, f'✅ 阈值计算完成: {threshold:.6f}')
                else:
                    task_manager.add_log(task_id, '⚠️ 验证集为空，跳过阈值计算')
            else:
                task_manager.add_log(task_id, '⚠️ 验证集文件不存在，跳过阈值计算')
        except Exception as e:
            logger.warning(f'阈值计算失败: {e}', exc_info=True)
            task_manager.add_log(task_id, f'⚠️ 阈值计算失败: {str(e)}')
        
        # 6. 如果提供了测试集，进行评估
        evaluation_results = None
        test_files = config.get('test_files', [])
        test_file = config.get('test_file')  # 兼容旧模式
        training_data_dir = Path('data') / 'ad' / task_id
        
        # 工况筛选模式：从保存的test.npz文件加载
        if dataset_mode == 'condition_filtered' and test_files:
            task_manager.update_task_status(task_id, 'training', 'Evaluating model on test set...', 95)
            task_manager.add_log(task_id, f'开始使用测试集评估模型（从保存的test.npz文件加载）')
            
            try:
                test_data_path = training_data_dir / 'test.npz'
                if test_data_path.exists():
                    evaluation_results = _evaluate_from_npz(
                        model=model,
                        test_data_path=test_data_path,
                        model_type=model_type,
                        config=config,
                        model_dir=model_dir,
                        task_id=task_id
                    )
                    
                    if evaluation_results:
                        task_manager.add_log(task_id, f'✅ 测试集评估完成')
                        task_manager.add_log(task_id, f'平均重构误差: {evaluation_results.get("mean_error", "N/A"):.6f}')
                        task_manager.add_log(task_id, f'误差标准差: {evaluation_results.get("std_error", "N/A"):.6f}')
                        logger.info(f'测试集评估完成: mean_error={evaluation_results.get("mean_error")}')
                else:
                    task_manager.add_log(task_id, f'⚠️ 测试集文件不存在: {test_data_path}')
            except Exception as e:
                logger.warning(f'测试集评估失败: {e}')
                task_manager.add_log(task_id, f'⚠️ 测试集评估失败: {str(e)}')
        elif test_file:
            # 旧模式：从CSV文件加载
            task_manager.update_task_status(task_id, 'training', 'Evaluating model on test set...', 95)
            task_manager.add_log(task_id, f'开始使用测试集评估模型')
            
            try:
                evaluation_results = _evaluate_anomaly_detection_model(
                    model=model,
                    processor=processor,
                    test_file=test_file,
                    model_type=model_type,
                    config=config,
                    model_dir=model_dir,
                    task_id=task_id
                )
                
                if evaluation_results:
                    task_manager.add_log(task_id, f'✅ 测试集评估完成')
                    task_manager.add_log(task_id, f'平均重构误差: {evaluation_results.get("mean_error", "N/A"):.6f}')
                    task_manager.add_log(task_id, f'误差标准差: {evaluation_results.get("std_error", "N/A"):.6f}')
                    logger.info(f'测试集评估完成: mean_error={evaluation_results.get("mean_error")}')
            except Exception as e:
                logger.warning(f'测试集评估失败: {e}')
                task_manager.add_log(task_id, f'⚠️ 测试集评估失败: {str(e)}')
        
        # 训练完成后通知Edge端下载模型
        try:
            _notify_edge_model_ready(task_id, model_dir.parent.name)
            logger.info(f"已通知Edge端模型就绪: {task_id}")
        except Exception as e:
            logger.warning(f"通知Edge端失败: {e}")
        
        # 训练完成 - 不自动计算阈值，等待用户点击
        completion_message = 'Training completed successfully'
        if evaluation_results:
            completion_message += f' (Test MAE: {evaluation_results.get("mean_error", 0):.6f})'
        
        task_manager.update_task_status(
            task_id, 
            'completed', 
            completion_message,
            100,
            num_epochs  # 传递总epoch数作为current_epoch
        )
        task_manager.update_model_save_path(task_id, str(model_path))
        task_manager.update_scaler_path(task_id, str(scaler_path))
        
        # 保存评估结果到任务
        if evaluation_results:
            task = task_manager.get_task(task_id)
            if task:
                task.evaluation_results = evaluation_results
        
        logger.info(f'✅ {model_type} 训练完成 | 模型: {model_path} | 标准化器: {scaler_path}')
        
        # 返回成功结果
        return {
            'success': True,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'task_id': task_id,
            'message': 'Training completed successfully'
        }
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(f"Training error for task {task_id}: {e}", exc_info=True)
        
        task_manager.update_task_status(task_id, 'failed', error_msg)
        logger.error(f'ERROR: {error_msg}')
        
        # 返回失败结果
        return {
            'success': False,
            'error': error_msg,
            'task_id': task_id
        }


def _evaluate_from_npz(
    model,
    test_data_path: Path,
    model_type: str,
    config: dict,
    model_dir: Path,
    task_id: str
) -> dict:
    """
    从保存的test.npz文件加载测试数据并评估模型
    
    测试数据包含标签信息（从元数据文件读取）：
    - labels: 每个序列的标签（0=正常，1=异常）
    
    Args:
        model: 训练好的模型
        test_data_path: test.npz文件路径
        model_type: 模型类型
        config: 配置字典
        model_dir: 模型保存目录
        task_id: 任务ID
    
    Returns:
        评估结果字典，包含误差统计和分类性能指标（如果有标签）
    """
    try:
        logger.info(f"从NPZ文件加载测试数据: {test_data_path}")
        
        # 加载测试数据
        test_data = np.load(test_data_path, allow_pickle=True)
        test_sequences = test_data['sequences']
        test_targets = test_data['targets']
        
        # 检查是否有标签信息（优先从npz的labels字段读取，如果没有则从序列最后一列提取）
        has_labels = 'labels' in test_data
        if has_labels:
            test_labels = test_data['labels']
            logger.info(f"从NPZ文件加载标签: labels形状={test_labels.shape}, 标签值范围=[{test_labels.min()}, {test_labels.max()}]")
            logger.info(f"标签统计: 0的数量={np.sum(test_labels == 0)}, 1的数量={np.sum(test_labels == 1)}, 其他值={np.sum((test_labels != 0) & (test_labels != 1))}")
        else:
            # 如果npz中没有labels字段，尝试从序列的最后一列提取标签
            # 注意：如果序列已经移除了标签列，这里会失败，所以优先使用npz中的labels字段
            test_labels = None
            logger.warning("NPZ文件中没有labels字段")
        
        if test_labels is not None:
            logger.info(f"测试数据: {len(test_sequences)} 个序列, 正常样本: {np.sum(test_labels == 0)}, 异常样本: {np.sum(test_labels == 1)}")
        else:
            logger.info(f"测试数据: {len(test_sequences)} 个序列（无标签信息）")
        
        # 进行预测
        model.eval()
        predictions = []
        batch_size = config.get('batch_size', 32)
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for i in range(0, len(test_sequences), batch_size):
                batch_sequences = test_sequences[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch_sequences.astype(np.float32)).to(device)
                
                if model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']:
                    # 自编码器：预测重构结果
                    pred = model(batch_tensor)
                    predictions.append(pred.cpu().numpy())
                else:
                    # 预测器：预测未来值
                    pred = model(batch_tensor)
                    predictions.append(pred.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # 确保predictions和test_targets的形状匹配
        logger.info(f"测试数据形状: sequences={test_sequences.shape}, targets={test_targets.shape}, predictions={predictions.shape}")
        
        # 计算重构误差（自编码器）或预测误差（预测器）
        if model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']:
            # 自编码器：计算重构误差
            # test_targets和predictions都是 (n_samples, sequence_length, feature_dim)
            # 需要逐样本计算MSE，避免内存溢出
            errors = []
            for i in range(len(test_targets)):
                error = np.mean(np.square(test_targets[i] - predictions[i]))
                errors.append(error)
            errors = np.array(errors)
        else:
            # 预测器：计算预测误差
            # test_targets和predictions都是 (n_samples, prediction_horizon, feature_dim)
            # 需要逐样本计算MSE，避免内存溢出
            errors = []
            for i in range(len(test_targets)):
                error = np.mean(np.square(test_targets[i] - predictions[i]))
                errors.append(error)
            errors = np.array(errors)
        
        # 计算统计信息
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        min_error = float(np.min(errors))
        max_error = float(np.max(errors))
        
        percentiles = {
            'p50': float(np.percentile(errors, 50)),
            'p75': float(np.percentile(errors, 75)),
            'p90': float(np.percentile(errors, 90)),
            'p95': float(np.percentile(errors, 95)),
            'p99': float(np.percentile(errors, 99))
        }
        
        evaluation_results = {
            'mean_error': mean_error,
            'std_error': std_error,
            'min_error': min_error,
            'max_error': max_error,
            'percentiles': percentiles,
            'test_samples': len(errors),
            'has_labels': has_labels
        }
        
        # 如果有标签，计算分类性能指标
        if has_labels and test_labels is not None:
            # 加载阈值（如果存在）
            threshold = None
            threshold_file = model_dir / 'threshold.json'
            if threshold_file.exists():
                try:
                    with open(threshold_file, 'r', encoding='utf-8') as f:
                        threshold_data = json.load(f)
                        threshold = threshold_data.get('threshold')
                        logger.info(f"使用已保存的阈值: {threshold}")
                except Exception as e:
                    logger.warning(f"读取阈值文件失败: {e}")
            
            # 如果没有阈值，使用95分位数作为阈值
            if threshold is None:
                threshold = percentiles['p95']
                logger.info(f"使用95分位数作为阈值: {threshold}")
            
            # 基于阈值预测异常
            predicted_labels = (errors > threshold).astype(int)
            
            # 计算分类性能指标
            true_positive = np.sum((predicted_labels == 1) & (test_labels == 1))
            true_negative = np.sum((predicted_labels == 0) & (test_labels == 0))
            false_positive = np.sum((predicted_labels == 1) & (test_labels == 0))
            false_negative = np.sum((predicted_labels == 0) & (test_labels == 1))
            
            accuracy = (true_positive + true_negative) / len(test_labels) if len(test_labels) > 0 else 0
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 正常样本和异常样本的误差统计
            normal_mask = test_labels == 0
            anomaly_mask = test_labels == 1
            
            normal_errors = errors[normal_mask] if np.any(normal_mask) else np.array([])
            anomaly_errors = errors[anomaly_mask] if np.any(anomaly_mask) else np.array([])
            
            evaluation_results.update({
                'threshold': float(threshold),
                'classification': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1_score),
                    'confusion_matrix': {
                        'true_positive': int(true_positive),
                        'true_negative': int(true_negative),
                        'false_positive': int(false_positive),
                        'false_negative': int(false_negative)
                    }
                },
                'normal_samples': int(np.sum(test_labels == 0)),
                'anomaly_samples': int(np.sum(test_labels == 1)),
                'error_by_label': {
                    'normal_error_mean': float(np.mean(normal_errors)) if len(normal_errors) > 0 else None,
                    'normal_error_std': float(np.std(normal_errors)) if len(normal_errors) > 0 else None,
                    'anomaly_error_mean': float(np.mean(anomaly_errors)) if len(anomaly_errors) > 0 else None,
                    'anomaly_error_std': float(np.std(anomaly_errors)) if len(anomaly_errors) > 0 else None
                }
            })
        
        # 保存评估结果
        eval_result_path = model_dir / f'{task_id}_evaluation_results.json'
        with open(eval_result_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存: {eval_result_path}")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"从NPZ文件评估模型失败: {e}", exc_info=True)
        raise


def _evaluate_anomaly_detection_model(
    model,
    processor,
    test_file: str,
    model_type: str,
    config: dict,
    model_dir: Path,
    task_id: str
) -> dict:
    """
    评估异常检测模型（旧模式：从CSV文件加载）
    
    测试数据格式：最后一列是标签列，0=正常，1=异常
    
    评估指标包括：
    - 重构误差统计（均值、标准差、最小值、最大值、分位数）
    - 分类性能指标（准确率、精确率、召回率、F1分数）- 如果测试集有标签
    - 正常/异常样本的误差分布对比
    """
    try:
        import pandas as pd
        
        logger.info(f"开始评估异常检测模型: {model_type}, 测试文件: {test_file}")
        
        # 查找测试数据文件
        test_data_path = None
        
        # 检查是否在上传的数据文件中
        if test_file in uploaded_data_files:
            test_data_path = Path(uploaded_data_files[test_file]['path'])
        else:
            # 尝试在训练数据目录中查找
            training_data_dir = Path('data') / 'ad'
            possible_paths = [
                training_data_dir / test_file,
                Path('data') / test_file,
                Path(test_file)
            ]
            
            for path in possible_paths:
                if path.exists():
                    test_data_path = path
                    break
        
        if not test_data_path or not test_data_path.exists():
            logger.warning(f"测试数据文件未找到: {test_file}")
            return None
        
        logger.info(f"找到测试数据文件: {test_data_path}")
        
        # 读取原始测试数据，提取标签列
        test_df = pd.read_csv(test_data_path)
        logger.info(f"测试数据形状: {test_df.shape}, 列: {list(test_df.columns)}")
        
        # 检测标签列（最后一列，0=正常，1=异常）
        has_labels = False
        labels = None
        label_column = None
        
        # 检查最后一列是否是标签列
        last_col = test_df.columns[-1]
        last_col_values = test_df[last_col].unique()
        
        # 如果最后一列只包含0和1，认为是标签列
        if set(last_col_values).issubset({0, 1, 0.0, 1.0}):
            has_labels = True
            label_column = last_col
            labels = test_df[last_col].values.astype(int)
            # 从特征中移除标签列
            feature_df = test_df.iloc[:, :-1]
            logger.info(f"检测到标签列: {last_col}, 正常样本: {np.sum(labels == 0)}, 异常样本: {np.sum(labels == 1)}")
        else:
            feature_df = test_df
            logger.info("未检测到标签列，将只计算误差统计")
        
        # 保存特征数据到临时文件用于数据处理器
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            feature_df.to_csv(tmp.name, index=False)
            temp_feature_path = tmp.name
        
        try:
            # 处理测试数据（使用与训练相同的处理器）
            if model_type == 'lstm_autoencoder':
                test_processor = LSTMAutoencoderDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    stride=config.get('stride', 1),
                    normalize=True
                )
            elif model_type == 'cnn_1d_autoencoder':
                test_processor = CNN1DAutoencoderDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    stride=config.get('stride', 1),
                    normalize=True
                )
            else:
                test_processor = LSTMPredictorDataProcessor(
                    sequence_length=config.get('sequence_length', 50),
                    prediction_horizon=config.get('prediction_horizon', 1),
                    normalize=True
                )
            
            # 加载已有的scaler
            scaler_path = model_dir / 'scaler.pkl'
            if scaler_path.exists():
                import pickle
                with open(scaler_path, 'rb') as f:
                    test_processor.scaler = pickle.load(f)
                logger.info(f"已加载标准化器: {scaler_path}")
            
            # 处理测试数据 - 只需要测试集，不需要划分
            test_data, _ = test_processor.process_pipeline(
                temp_feature_path,
                train_ratio=1.0  # 全部作为测试数据
            )
        finally:
            # 清理临时文件
            import os
            if os.path.exists(temp_feature_path):
                os.unlink(temp_feature_path)
        
        test_sequences = test_data.sequences
        test_targets = test_data.targets
        
        logger.info(f"测试数据: {len(test_sequences)} 个序列")
        
        # 进行预测
        model.eval()
        predictions = []
        batch_size = config.get('batch_size', 32)
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for i in range(0, len(test_sequences), batch_size):
                batch_seq = test_sequences[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch_seq.astype(np.float32)).to(device)
                batch_pred = model(batch_tensor)
                predictions.extend(batch_pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # 计算重构误差
        if model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']:
            # 自编码器：计算输入与重构之间的误差
            errors = np.mean((test_sequences - predictions) ** 2, axis=(1, 2))  # MSE per sample
        else:
            # 预测器：计算预测值与真实值之间的误差
            errors = np.mean((test_targets - predictions) ** 2, axis=-1)  # MSE per sample
            if len(errors.shape) > 1:
                errors = np.mean(errors, axis=1)
        
        # 计算基础统计指标
        evaluation_results = {
            'total_samples': len(test_sequences),
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'median_error': float(np.median(errors)),
            'percentile_90': float(np.percentile(errors, 90)),
            'percentile_95': float(np.percentile(errors, 95)),
            'percentile_99': float(np.percentile(errors, 99)),
            'test_file': test_file,
            'model_type': model_type,
            'has_labels': has_labels,
            'evaluated_at': datetime.now().isoformat()
        }
        
        # 如果有标签，计算分类性能指标
        if has_labels and labels is not None:
            sequence_length = config.get('sequence_length', 50)
            stride = config.get('stride', 1)
            
            # 将原始标签映射到序列标签（每个序列的标签取该序列中的主要标签）
            sequence_labels = []
            for i in range(len(test_sequences)):
                # 计算该序列对应的原始数据索引范围
                start_idx = i * stride
                end_idx = start_idx + sequence_length
                if end_idx <= len(labels):
                    # 如果序列中任何一个点是异常，则标记该序列为异常
                    seq_label = 1 if np.any(labels[start_idx:end_idx] == 1) else 0
                else:
                    seq_label = 0
                sequence_labels.append(seq_label)
            
            sequence_labels = np.array(sequence_labels)
            
            # 使用95%分位数作为默认阈值
            threshold = evaluation_results['percentile_95']
            
            # 基于阈值预测异常
            predicted_labels = (errors > threshold).astype(int)
            
            # 计算分类指标
            true_positive = np.sum((predicted_labels == 1) & (sequence_labels == 1))
            true_negative = np.sum((predicted_labels == 0) & (sequence_labels == 0))
            false_positive = np.sum((predicted_labels == 1) & (sequence_labels == 0))
            false_negative = np.sum((predicted_labels == 0) & (sequence_labels == 1))
            
            accuracy = (true_positive + true_negative) / len(sequence_labels) if len(sequence_labels) > 0 else 0
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 正常样本和异常样本的误差统计
            normal_mask = sequence_labels == 0
            anomaly_mask = sequence_labels == 1
            
            normal_errors = errors[normal_mask] if np.any(normal_mask) else np.array([])
            anomaly_errors = errors[anomaly_mask] if np.any(anomaly_mask) else np.array([])
            
            evaluation_results.update({
                'normal_samples': int(np.sum(sequence_labels == 0)),
                'anomaly_samples': int(np.sum(sequence_labels == 1)),
                'threshold_used': float(threshold),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'true_positive': int(true_positive),
                'true_negative': int(true_negative),
                'false_positive': int(false_positive),
                'false_negative': int(false_negative),
                'normal_error_mean': float(np.mean(normal_errors)) if len(normal_errors) > 0 else None,
                'normal_error_std': float(np.std(normal_errors)) if len(normal_errors) > 0 else None,
                'anomaly_error_mean': float(np.mean(anomaly_errors)) if len(anomaly_errors) > 0 else None,
                'anomaly_error_std': float(np.std(anomaly_errors)) if len(anomaly_errors) > 0 else None,
            })
            
            logger.info(f"分类性能 - 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1_score:.4f}")
        
        # 保存评估结果到文件
        eval_file = model_dir / f'{task_id}_evaluation_results.json'
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        # 保存误差分布数据（用于可视化）
        errors_file = model_dir / f'{task_id}_error_distribution.npz'
        save_data = {'errors': errors}
        if has_labels:
            save_data['labels'] = sequence_labels
        np.savez_compressed(errors_file, **save_data)
        
        logger.info(f"评估结果已保存: {eval_file}")
        logger.info(f"评估完成 - 平均误差: {evaluation_results['mean_error']:.6f}, 标准差: {evaluation_results['std_error']:.6f}")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"评估异常检测模型失败: {e}", exc_info=True)
        return None


@anomaly_detection_bp.route('/training_status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    """获取训练任务状态"""
    try:
        def safe_load_json(file_path):
            """Load JSON content from disk without raising."""
            if not file_path:
                return None
            try:
                path_obj = Path(file_path)
                if not path_obj.exists():
                    return None
                with open(path_obj, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(f"无法解析文件 {file_path}: {exc}")
                return None

        # 使用任务管理器获取任务状态
        task_manager = get_task_manager()
        task = task_manager.get_task(task_id)
        
        if task is None:
            # 如果任务管理器中没有找到，尝试从文件系统恢复
            models_dir = Path(project_root) / 'cloud' / 'models' / 'anomaly_detection'
            
            # 查找所有可能的任务目录（遍历所有模型类型目录）
            task_dirs = []
            if models_dir.exists():
                # 遍历所有模型类型目录：lstm_prediction, lstm_autoencoder, cnn_1d_autoencoder
                for model_type_dir in models_dir.iterdir():
                    if not model_type_dir.is_dir():
                        continue
                    
                    # 首先查找精确匹配的任务ID
                    exact_match = model_type_dir / task_id
                    if exact_match.exists() and exact_match.is_dir():
                        task_dirs.append(exact_match)
                    else:
                        # 查找包含任务ID的目录
                        for item in model_type_dir.iterdir():
                            if item.is_dir() and (task_id in item.name or item.name.endswith('001')):
                                task_dirs.append(item)
            
            # 如果找到任务目录，恢复状态
            if task_dirs:
                # 使用最新的目录
                task_dir = max(task_dirs, key=lambda p: p.stat().st_mtime)
                logger.info(f"从文件系统恢复任务状态: {task_dir}")
                
                # 检查任务完成状态
                threshold_path = task_dir / 'threshold.json'
                model_path = task_dir / 'model.pth'
                
                # 构建状态信息
                status = 'completed'
                threshold_value = None
                threshold_metadata = None
                
                # 如果存在阈值文件，标记为阈值已计算
                if threshold_path.exists():
                    try:
                        with open(threshold_path, 'r', encoding='utf-8') as f:
                            threshold_data = json.load(f)
                        
                        status = 'threshold_completed'
                        threshold_value = threshold_data.get('threshold', threshold_data.get('threshold_value'))
                        threshold_metadata = threshold_data
                        logger.info(f"恢复阈值信息: {threshold_value}")
                    except Exception as e:
                        logger.warning(f"无法读取阈值文件: {e}")
                
                # 尝试加载评估结果
                evaluation_results = None
                eval_file = task_dir / f'{task_id}_evaluation_results.json'
                if eval_file.exists():
                    try:
                        with open(eval_file, 'r', encoding='utf-8') as f:
                            evaluation_results = json.load(f)
                        logger.info(f"恢复评估结果: {eval_file}")
                    except Exception as e:
                        logger.warning(f"无法读取评估结果文件: {e}")
                
                config_path = task_dir / 'config.json'
                training_config_path = task_dir / 'training_config.json'
                metadata_path = task_dir / 'metadata.json'

                config_data = safe_load_json(config_path) or {}
                if not isinstance(config_data, dict):
                    config_data = {}

                training_config = safe_load_json(training_config_path) or config_data
                if not isinstance(training_config, dict):
                    training_config = config_data

                metadata = safe_load_json(metadata_path) or config_data.get('metadata') or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                dataset_config = {}
                if isinstance(config_data.get('dataset_config'), dict):
                    dataset_config = config_data.get('dataset_config')
                elif isinstance(training_config, dict) and isinstance(training_config.get('dataset_config'), dict):
                    dataset_config = training_config.get('dataset_config')

                # 返回edge端trainer期望的格式
                return jsonify({
                    'success': True,
                    'task': {
                        'id': task_id,
                        'status': status,
                        'current_epoch': 100,
                        'epoch': 100,
                        'completed_epochs': 100,
                        'config': config_data,
                        'training_config': training_config,
                        'dataset_config': dataset_config,
                        'metadata': metadata,
                        'total_epochs': 100,
                        'current_train_loss': None,  # 文件恢复时没有当前损失
                        'current_val_loss': None,
                        'loss': None,                # 兼容前端
                        'val_loss': None,
                        'final_train_loss': None,
                        'final_val_loss': None,
                        'message': f'任务已完成（从文件恢复）',
                        'logs': [f'从文件系统恢复任务状态: {task_dir.name}'],
                        'threshold_value': threshold_value,
                        'threshold_path': str(threshold_path) if threshold_path.exists() else None,
                        'threshold_metadata': threshold_metadata,
                        'learning_rate': config_data.get('learning_rate'),
                        'dataset_mode': config_data.get('dataset_mode'),
                        'model_type': config_data.get('model_type'),
                        'model_save_path': str(model_path) if model_path.exists() else None,
                        'created_at': task_dir.stat().st_ctime,
                        'updated_at': task_dir.stat().st_mtime,
                        'progress': 100,
                        'evaluation': evaluation_results,
                        'evaluation_results': evaluation_results
                    }
                })
            
            # 如果完全找不到任务
            return jsonify({
                'success': False,
                'error': f'Task {task_id} not found'
            }), 404
        
        # 任务存在，使用任务管理器的数据
        # 获取评估结果（如果存在）
        evaluation_results = getattr(task, 'evaluation_results', None)
        task_config = task.config if isinstance(task.config, dict) else {}
        dataset_config = task_config.get('dataset_config') if isinstance(task_config.get('dataset_config'), dict) else {}
        metadata = task_config.get('metadata') if isinstance(task_config.get('metadata'), dict) else {}
        
        # 转换为edge端trainer期望的格式 - 包装在task字段中
        return jsonify({
            'success': True,
            'task': {
                'id': task.task_id,
                'status': task.status,
                'current_epoch': task.current_epoch,
                'epoch': task.current_epoch,  # 兼容两个字段
                'completed_epochs': task.current_epoch,
            'config': task_config,
            'training_config': task_config,
            'dataset_config': dataset_config,
            'metadata': metadata,
                'total_epochs': task.config.get('epochs', 100),
                'current_train_loss': task.current_train_loss,  # 当前训练损失
                'current_val_loss': task.current_val_loss,    # 当前验证损失
                'loss': task.current_train_loss,              # 兼容前端期望的loss字段
                'val_loss': task.current_val_loss,            # 兼容前端期望的val_loss字段
                'final_train_loss': task.final_train_loss,
                'final_val_loss': task.final_val_loss,
                'message': task.message,
                'logs': task.logs.split('\n')[-20:] if isinstance(task.logs, str) and len(task.logs.split('\n')) > 20 else (task.logs.split('\n') if isinstance(task.logs, str) else []),  # 最后20条日志或全部
                'threshold_value': task.threshold_value,
                'threshold_path': task.threshold_path,
                'threshold_metadata': task.threshold_metadata,
                'learning_rate': getattr(task, 'learning_rate', None),
                'dataset_mode': getattr(task, 'dataset_mode', None),
                'model_type': getattr(task, 'model_type', None),
                'progress': task.progress,
                'created_at': task.created_at if isinstance(task.created_at, str) else None,
                'updated_at': task.updated_at if isinstance(task.updated_at, str) else None,
                'model_save_path': task.model_save_path,
                'scaler_path': task.scaler_path,
                'evaluation': evaluation_results,
                'evaluation_results': evaluation_results
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get training status for {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 旧的模型列表路由已移除，使用下面的 list_anomaly_detection_models() 替代
# @anomaly_detection_bp.route('/models', methods=['GET'])
# def list_models():
#     """列出异常检测模型（已废弃，使用 list_anomaly_detection_models）"""
#     ...

@anomaly_detection_bp.route('/inference_tasks', methods=['GET'])
def list_inference_tasks():
    """获取推理任务列表"""
    try:
        base_dir = Path('models/anomaly_detection/inference_tasks')
        if not base_dir.exists():
            return jsonify({'success': True, 'tasks': []})
        
        tasks = []
        for task_dir in sorted(base_dir.glob('inference_*'), reverse=True):
            config_path = task_dir / 'config.json'
            summary_path = task_dir / 'results_summary.json'
            
            if config_path.exists() and summary_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    task_info = {
                        'task_id': config['task_id'],
                        'created_at': config['created_at'],
                        'completed_at': summary.get('completed_at'),
                        'model_type': config['model_type'],
                        'source_task_id': config.get('source_task_id'),
                        'total_samples': summary['total_samples'],
                        'anomalies_detected': summary['anomalies_detected'],
                        'anomaly_percentage': summary['anomaly_percentage'],
                        'threshold': summary['threshold'],
                        'has_performance_metrics': 'performance_metrics' in summary,
                        'task_dir': str(task_dir)
                    }
                    tasks.append(task_info)
                    
                except Exception as e:
                    logger.warning(f"Failed to load task info for {task_dir}: {e}")
                    continue
        
        return jsonify({
            'success': True,
            'tasks': tasks,
            'total_count': len(tasks)
        })
        
    except Exception as e:
        logger.error(f"Failed to list inference tasks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@anomaly_detection_bp.route('/inference_tasks/<task_id>', methods=['GET'])
def get_inference_task_detail(task_id):
    """获取推理任务详细信息"""
    try:
        task_dir = Path(f'models/anomaly_detection/inference_tasks/inference_{task_id}')
        if not task_dir.exists():
            return jsonify({
                'success': False,
                'error': f'Inference task {task_id} not found'
            }), 404
        
        result = {'task_id': task_id}
        
        # 加载配置
        config_path = task_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                result['config'] = json.load(f)
        
        # 加载结果摘要
        summary_path = task_dir / 'results_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                result['results_summary'] = json.load(f)
        
        # 加载数据信息
        data_info_path = task_dir / 'data_info.json'
        if data_info_path.exists():
            with open(data_info_path, 'r', encoding='utf-8') as f:
                result['data_info'] = json.load(f)
        
        # 检查详细结果文件
        detailed_path = task_dir / 'detailed_results.npz'
        result['has_detailed_results'] = detailed_path.exists()
        
        return jsonify({
            'success': True,
            'task': result
        })
        
    except Exception as e:
        logger.error(f"Failed to get inference task detail: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@anomaly_detection_bp.route('/calculate_threshold/<task_id>', methods=['POST'])
def calculate_threshold(task_id):
    """计算训练完成后的异常检测阈值"""
    try:
        logger.info(f"开始计算任务 {task_id} 的阈值")
        
        # 获取请求中的阈值参数
        request_data = request.get_json() or {}
        threshold_method = request_data.get('threshold_method', 'percentile')
        percentile = float(request_data.get('percentile', 95.0))
        residual_metric = request_data.get('residual_metric', 'rmse')
        contamination = request_data.get('contamination')  # 获取contamination参数
        
        # 如果contamination是字符串，转换为浮点数
        if contamination is not None:
            if isinstance(contamination, str):
                contamination = float(contamination)
            else:
                contamination = float(contamination)
            # 如果contamination是百分比形式（如20.0表示20%），转换为小数（0.2）
            if contamination > 1.0:
                contamination = contamination / 100.0
        
        logger.info(f"使用阈值参数: method={threshold_method}, percentile={percentile}, metric={residual_metric}, contamination={contamination}")
        
        # 使用任务管理器获取任务
        task_manager = get_task_manager()
        task = task_manager.get_task(task_id)
        
        if task is None:
            return jsonify({
                'success': False,
                'error': f'训练任务 {task_id} 不存在'
            }), 404
        
        # 检查训练状态
        if task.status != 'completed':
            return jsonify({
                'success': False,
                'error': '训练尚未完成，无法计算阈值'
            }), 400
        
        # 更新任务状态为计算阈值中
        task_manager.update_task_status(task_id, 'calculating_threshold', '正在计算异常检测阈值...')
        
        # 获取训练数据文件路径 - 支持多种可能的模型存储位置
        config = task.config
        model_type = config.get('model_type', 'lstm_predictor')
        
        # 根据模型类型确定模型目录
        if model_type == 'lstm_autoencoder':
            model_type_dir = 'lstm_autoencoder'
        elif model_type == 'cnn_1d_autoencoder':
            model_type_dir = 'cnn_1d_autoencoder'
        else:
            model_type_dir = 'lstm_prediction'
        
        possible_model_dirs = [
            Path(project_root) / 'cloud' / 'models' / 'anomaly_detection' / model_type_dir,
            Path('models') / 'anomaly_detection' / model_type_dir,  # 相对路径（云端服务运行时）
            Path.cwd() / 'models' / 'anomaly_detection' / model_type_dir,  # 当前工作目录
        ]
        
        model_dir = None
        
        # 尝试在所有可能的位置查找模型目录
        for models_dir in possible_model_dirs:
            if models_dir.exists():
                task_model_dir = models_dir / task_id
                if task_model_dir.exists():
                    model_dir = task_model_dir
                    logger.info(f'找到模型目录: {model_dir}')
                    break
                else:
                    # 尝试模糊匹配
                    for item in models_dir.iterdir():
                        if item.is_dir() and (task_id in item.name or item.name.startswith(task_id[:10])):
                            model_dir = item
                            logger.info(f'找到匹配的模型目录: {model_dir}')
                            break
                    if model_dir:
                        break
        
        if not model_dir:
            logger.error(f'无法找到模型目录，任务ID: {task_id}')
            logger.error(f'尝试的路径: {[str(d) for d in possible_model_dirs]}')
            task_manager.update_task_status(task_id, 'completed', f'阈值计算失败: 找不到模型目录')
            return jsonify({
                'success': False,
                'error': f'模型文件不存在，任务ID: {task_id}'
            }), 400
        
        config = task.config
        data_file = config.get('dataset_file')
        
        # 查找数据文件
        if data_file and data_file in uploaded_data_files:
            # 使用上传的训练数据
            data_file_path = Path(uploaded_data_files[data_file]['path'])
        elif data_file:
            # 尝试在训练数据目录中查找 (异常检测: cloud/data/ad)
            training_data_dir = Path('data') / 'ad'
            data_file_path = training_data_dir / data_file
            if not data_file_path.exists():
                data_file_path = Path('data') / data_file
        else:
            data_file_path = None
        
        if not data_file_path or not data_file_path.exists():
            task_manager.update_task_status(task_id, 'completed', '阈值计算失败: 找不到训练数据文件')
            return jsonify({
                'success': False,
                'error': '训练数据文件不存在，请确保数据已从边端上传到云端'
            }), 400
        
        # model_dir已经在前面验证过存在，这里不需要再检查
        
        # 使用已训练的模型计算阈值
        if training_available:
            try:
                # 从任务中获取保存的配置信息
                config_path = model_dir / 'config.json'
                if not config_path.exists():
                    task_manager.update_task_status(task_id, 'completed', '阈值计算失败: 找不到训练配置文件')
                    return jsonify({
                        'success': False,
                        'error': '找不到训练配置文件'
                    }), 400
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                
                # 重新加载数据处理器和数据（根据模型类型选择）
                model_type = model_config.get('model_type', 'lstm_predictor')
                
                if model_type == 'lstm_autoencoder':
                    data_processor = LSTMAutoencoderDataProcessor(
                        sequence_length=model_config.get('sequence_length', 50),
                        stride=model_config.get('stride', 1),
                        normalize=True
                    )
                elif model_type == 'cnn_1d_autoencoder':
                    data_processor = CNN1DAutoencoderDataProcessor(
                        sequence_length=model_config.get('sequence_length', 50),
                        stride=model_config.get('stride', 1),
                        normalize=True
                    )
                else:
                    data_processor = LSTMPredictorDataProcessor(
                        sequence_length=model_config.get('sequence_length', 50),
                        prediction_horizon=model_config.get('prediction_horizon', 1),
                        normalize=True
                    )
                
                # 加载训练数据
                train_data, _ = data_processor.process_pipeline(
                    str(data_file_path),
                    train_ratio=0.8  # 只需要训练数据用于计算阈值
                )
                
                # 加载训练好的模型
                model_path = model_dir / 'model.pth'
                if not model_path.exists():
                    task_manager.update_task_status(task_id, 'completed', '阈值计算失败: 找不到训练好的模型文件')
                    return jsonify({
                        'success': False,
                        'error': '找不到训练好的模型文件'
                    }), 400
                
                # 重建模型架构（根据模型类型选择）
                feature_dim = model_config.get('feature_dim')
                input_shape = (model_config.get('sequence_length', 50), feature_dim)
                
                if model_type == 'lstm_autoencoder':
                    model = LSTMAutoencoderModelBuilder.create_model(
                        'lstm_autoencoder',
                        input_shape=input_shape,
                        hidden_size=model_config.get('hidden_units', 128),
                        num_layers=model_config.get('num_layers', 2),
                        bottleneck_dim=model_config.get('bottleneck_dim', 64),
                        dropout=model_config.get('dropout', 0.1)
                    )
                elif model_type == 'cnn_1d_autoencoder':
                    model = CNN1DAutoencoderModelBuilder.create_model(
                        'cnn_1d_autoencoder',
                        input_shape=input_shape,
                        num_filters=model_config.get('num_filters', 64),
                        kernel_size=model_config.get('kernel_size', 3),
                        bottleneck_dim=model_config.get('bottleneck_dim', 64),
                        num_conv_layers=model_config.get('num_conv_layers', model_config.get('num_layers', 3)),
                        dropout=model_config.get('dropout', 0.1),
                        activation=model_config.get('activation', 'relu')
                    )
                else:
                    model = LSTMPredictorModelBuilder.build_lstm_predictor(
                        input_shape=input_shape,
                        hidden_units=model_config.get('hidden_units', 128),
                        num_layers=model_config.get('num_layers', 2),
                        dropout=model_config.get('dropout', 0.1),
                        activation=model_config.get('activation', 'tanh')
                    )
                
                # 加载模型参数
                device = _get_torch_device()
                model = model.to(device)
                state_dict = torch.load(str(model_path), map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                logger.info(f"成功加载模型: {model_path}")
                
                # 使用训练数据计算阈值
                sample_size = min(1000, len(train_data.sequences))
                sample_sequences = train_data.sequences[:sample_size]
                sample_targets = train_data.targets[:sample_size]
                
                # 批量预测
                predictions = []
                batch_size = 32
                with torch.no_grad():
                    for i in range(0, len(sample_sequences), batch_size):
                        batch_seq = sample_sequences[i:i+batch_size]
                        batch_tensor = torch.from_numpy(batch_seq.astype(np.float32)).to(device)
                        batch_pred = model(batch_tensor)
                        predictions.extend(batch_pred.cpu().numpy())
                
                predictions = np.array(predictions)
                actuals = sample_targets
                
                # 计算阈值（根据模型类型选择阈值计算器）
                if model_type == 'lstm_autoencoder':
                    threshold_calc = LSTMAutoencoderThresholdCalculator(
                        method=threshold_method,
                        residual_metric=residual_metric
                    )
                    # 对于自编码器，用重构误差计算阈值
                    threshold_value = threshold_calc.fit(
                        predictions, 
                        actuals,
                        percentile=percentile,
                        contamination=contamination
                    )
                elif model_type == 'cnn_1d_autoencoder':
                    threshold_calc = CNN1DAutoencoderThresholdCalculator(
                        method=threshold_method,
                        residual_metric=residual_metric
                    )
                    # 对于自编码器，用重构误差计算阈值
                    threshold_value = threshold_calc.fit(
                        predictions, 
                        actuals,
                        percentile=percentile,
                        contamination=contamination
                    )
                else:
                    threshold_calc = LSTMPredictorThresholdCalculator(residual_method='l2_norm')
                    threshold_value = threshold_calc.fit_threshold(
                        predictions, 
                        actuals,
                        method=threshold_method,
                        percentile=percentile,
                        contamination=contamination
                    )
                
                # 判断是否为自编码器类型
                is_autoencoder = model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']
                threshold_metadata = {
                    'method': threshold_method if is_autoencoder else threshold_method,
                    'percentile': percentile if is_autoencoder else percentile,
                    'residual_metric': residual_metric if is_autoencoder else 'l2_norm',
                    'contamination': contamination,  # 添加contamination信息
                    'sample_size': sample_size,
                    'statistics': threshold_calc.stats if hasattr(threshold_calc, 'stats') else getattr(threshold_calc, 'residual_stats', {})
                }
                
                # 保存阈值文件
                threshold_path = model_dir / 'threshold.json'
                threshold_data = {
                    'threshold_value': float(threshold_value),
                    'method': threshold_method if is_autoencoder else 'percentile',
                    'percentile': percentile if is_autoencoder else 95.0,
                    'residual_metric': residual_metric if is_autoencoder else 'l2_norm',
                    'sample_size': sample_size,
                    'calculated_at': datetime.now().isoformat(),
                    'task_id': task_id,
                    'model_type': model_type,
                    'statistics': threshold_calc.stats if hasattr(threshold_calc, 'stats') else getattr(threshold_calc, 'residual_stats', {})
                }
                
                with open(threshold_path, 'w', encoding='utf-8') as f:
                    json.dump(threshold_data, f, indent=2, ensure_ascii=False)
                
                # 更新任务状态为阈值计算完成
                task_manager.update_threshold_info(
                    task_id, 
                    str(threshold_path), 
                    float(threshold_value), 
                    threshold_metadata
                )
                task_manager.update_task_status(task_id, 'threshold_completed', f'阈值计算完成: {threshold_value:.6f}')
                
                # 阈值计算完成后通知Edge端下载更新的模型
                try:
                    _notify_edge_model_ready(task_id, model_dir.parent.name)
                    logger.info(f"已通知Edge端阈值文件就绪: {task_id}")
                except Exception as e:
                    logger.warning(f"通知Edge端阈值更新失败: {e}")
                
                logger.info(f"阈值计算完成: {threshold_value}")
                
                return jsonify({
                    'success': True,
                    'threshold_value': float(threshold_value),
                    'threshold_path': str(threshold_path),
                    'metadata': threshold_metadata
                })
                
            except Exception as e:
                logger.error(f"阈值计算失败: {e}")
                task_manager.update_task_status(task_id, 'completed', f'阈值计算失败: {str(e)}')
                
                return jsonify({
                    'success': False,
                    'error': f'阈值计算失败: {str(e)}'
                }), 500
        else:
            # 训练模块不可用，无法计算真实阈值
            error_msg = '训练模块不可用，无法计算异常检测阈值。请确保PyTorch等依赖已正确安装。'
            task_manager.update_task_status(task_id, 'failed', error_msg)
            logger.error(f"阈值计算失败: {error_msg}")
            
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    except Exception as e:
        logger.error(f"阈值计算异常: {e}")
        return jsonify({
            'success': False,
            'error': f'阈值计算异常: {str(e)}'
        }), 500


@anomaly_detection_bp.route('/models', methods=['GET'])
def list_anomaly_detection_models():
    """获取异常检测模型列表"""
    print("=" * 80)
    print("[模型列表API] ===== 开始处理模型列表请求 =====")
    print("=" * 80)
    try:
        # 尝试多个可能的模型目录路径
        from pathlib import Path
        import os
        
        # 获取项目根目录（假设api.py在cloud/src/anomaly_detection/）
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]  # 从 cloud/src/anomaly_detection/api.py 回到项目根目录
        
        # 使用 print 确保日志输出（因为 logger 可能级别设置问题）
        print(f'[模型列表API] 当前文件路径: {current_file}')
        print(f'[模型列表API] 计算的项目根目录: {project_root}')
        print(f'[模型列表API] 当前工作目录: {Path.cwd()}')
        logger.info(f'当前文件路径: {current_file}')
        logger.info(f'计算的项目根目录: {project_root}')
        logger.info(f'当前工作目录: {Path.cwd()}')
        
        # 构建可能的模型目录路径
        # 优先使用环境变量或配置文件中的路径
        models_dir_from_env = os.environ.get('MODELS_DIR')
        if models_dir_from_env:
            env_models_dir = Path(models_dir_from_env) / 'anomaly_detection'
        else:
            env_models_dir = None
        
        possible_model_dirs = []
        
        # 1. 环境变量指定的路径（最高优先级）
        if env_models_dir:
            possible_model_dirs.append(env_models_dir)
        
        # 2. 标准项目路径
        possible_model_dirs.extend([
            project_root / 'cloud' / 'models' / 'anomaly_detection',  # 标准路径：项目根/cloud/models/anomaly_detection
            Path('models/anomaly_detection'),  # 相对路径（如果从cloud目录运行）
            Path.cwd() / 'models' / 'anomaly_detection',  # 当前工作目录
            Path.cwd() / 'cloud' / 'models' / 'anomaly_detection',  # 当前工作目录下的cloud/models
        ])
        
        print(f'[模型列表API] 尝试查找模型目录，可能的路径:')
        logger.info(f'尝试查找模型目录，可能的路径:')
        for i, possible_dir in enumerate(possible_model_dirs, 1):
            exists = possible_dir.exists()
            print(f'  {i}. {possible_dir} - 存在: {exists}')
            logger.info(f'  {i}. {possible_dir} - 存在: {exists}')
            if exists:
                # 列出目录内容以确认
                try:
                    subdirs = [d.name for d in possible_dir.iterdir() if d.is_dir()]
                    print(f'     子目录: {subdirs}')
                    logger.info(f'     子目录: {subdirs}')
                except Exception as e:
                    print(f'     无法列出子目录: {e}')
                    logger.warning(f'     无法列出子目录: {e}')
        
        models_dir = None
        for possible_dir in possible_model_dirs:
            if possible_dir.exists():
                models_dir = possible_dir
                print(f'[模型列表API] ✓ 找到模型目录: {models_dir}')
                logger.info(f'✓ 找到模型目录: {models_dir}')
                break
        
        models = []
        
        if models_dir and models_dir.exists():
            print(f'[模型列表API] 开始扫描模型目录: {models_dir}')
            logger.info(f'开始扫描模型目录: {models_dir}')
            # 遍历所有模型类型目录
            for model_type_dir in models_dir.iterdir():
                if not model_type_dir.is_dir():
                    print(f'[模型列表API] 跳过非目录: {model_type_dir.name}')
                    logger.debug(f'跳过非目录: {model_type_dir.name}')
                    continue
                
                print(f'[模型列表API] 扫描模型类型目录: {model_type_dir.name}')
                logger.info(f'扫描模型类型目录: {model_type_dir.name}')
                task_count = 0
                    
                # 遍历所有任务目录
                for task_dir in model_type_dir.iterdir():
                    if not task_dir.is_dir():
                        print(f'[模型列表API] 跳过非目录任务: {task_dir.name}')
                        logger.debug(f'跳过非目录任务: {task_dir.name}')
                        continue
                    
                    task_count += 1
                    print(f'[模型列表API] 检查任务目录: {task_dir.name}')
                    logger.debug(f'检查任务目录: {task_dir.name}')
                    
                    # 检查必要文件
                    config_path = task_dir / 'config.json'
                    # 尝试多种模型文件扩展名
                    model_files = (
                        list(task_dir.glob('*.pth')) + 
                        list(task_dir.glob('*.pth')) + 
                        list(task_dir.glob('*.mindir')) +
                        list(task_dir.glob('model.*'))  # 匹配任何以model.开头的文件
                    )
                    
                    # 列出目录中的所有文件，用于调试
                    all_files = list(task_dir.iterdir())
                    print(f'[模型列表API]   目录中的文件: {[f.name for f in all_files]}')
                    print(f'[模型列表API]   config.json 存在: {config_path.exists()}')
                    print(f'[模型列表API]   模型文件数量: {len(model_files)}, 文件: {[f.name for f in model_files]}')
                    logger.debug(f'  目录中的文件: {[f.name for f in all_files]}')
                    logger.debug(f'  config.json 存在: {config_path.exists()}')
                    logger.debug(f'  模型文件数量: {len(model_files)}, 文件: {[f.name for f in model_files]}')
                    
                    if not config_path.exists():
                        print(f'[模型列表API]   跳过 {task_dir.name}: 缺少 config.json')
                        logger.warning(f'  跳过 {task_dir.name}: 缺少 config.json')
                        continue
                    
                    if not model_files:
                        print(f'[模型列表API]   跳过 {task_dir.name}: 缺少模型文件 (.pth)')
                        logger.warning(f'  跳过 {task_dir.name}: 缺少模型文件 (.pth)')
                        continue
                    
                    try:
                        # 加载配置
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        model_path = model_files[0]
                        model_stat = model_path.stat()
                        
                        # 检查阈值文件
                        threshold_info = None
                        threshold_path = task_dir / 'threshold.json'
                        if threshold_path.exists():
                            try:
                                with open(threshold_path, 'r', encoding='utf-8') as f:
                                    threshold_info = json.load(f)
                            except:
                                pass
                        
                        model_info = {
                            'task_id': task_dir.name,
                            'model_type': model_type_dir.name,
                            'filename': model_path.name,
                            'size': model_stat.st_size,
                            'created_at': datetime.fromtimestamp(model_stat.st_ctime).isoformat(),
                            'modified_at': datetime.fromtimestamp(model_stat.st_mtime).isoformat(),
                            'config': {
                                'sequence_length': config.get('sequence_length', 50),
                                'hidden_units': config.get('hidden_units', 128),
                                'num_layers': config.get('num_layers', 2),
                                'epochs': config.get('epochs', 50),
                                'batch_size': config.get('batch_size', 32),
                                'feature_dim': config.get('feature_dim'),
                            },
                            'files': {
                                'model': {
                                    'exists': True,
                                    'size': model_stat.st_size,
                                    'filename': model_path.name
                                },
                                'scaler': {
                                    'exists': (task_dir / 'scaler.pkl').exists()
                                },
                                'threshold': {
                                    'exists': threshold_path.exists(),
                                    'value': threshold_info.get('threshold_value') if threshold_info else None,
                                    'method': threshold_info.get('method') if threshold_info else None
                                }
                            }
                        }
                        
                        models.append(model_info)
                        print(f'[模型列表API] ✓ 添加模型: {task_dir.name} (类型: {model_type_dir.name})')
                        logger.info(f'✓ 添加模型: {task_dir.name} (类型: {model_type_dir.name})')
                        
                    except Exception as e:
                        print(f'[模型列表API] ❌ 读取模型配置失败 {task_dir}: {e}')
                        logger.error(f"读取模型配置失败 {task_dir}: {e}", exc_info=True)
                        continue
                
                print(f'[模型列表API] 模型类型 {model_type_dir.name} 共扫描了 {len(list(model_type_dir.iterdir()))} 个项目')
                logger.info(f'模型类型 {model_type_dir.name} 共扫描了 {len(list(model_type_dir.iterdir()))} 个项目')
        else:
            print(f'[模型列表API] ❌ 模型目录不存在！尝试的路径:')
            logger.error(f'❌ 模型目录不存在！尝试的路径:')
            for i, path in enumerate(possible_model_dirs, 1):
                print(f'  {i}. {path} (存在: {path.exists()})')
                logger.error(f'  {i}. {path} (存在: {path.exists()})')
        
        # 按创建时间降序排列
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        print(f'[模型列表API] 📊 总共找到 {len(models)} 个模型')
        logger.info(f'📊 总共找到 {len(models)} 个模型')
        if len(models) > 0:
            print(f'[模型列表API] 模型列表: {[m["task_id"] for m in models]}')
            logger.info(f'模型列表: {[m["task_id"] for m in models]}')
        
        return jsonify({
            'success': True,
            'models': models,
            'total_count': len(models)
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[模型列表API] ❌ 异常发生: {e}")
        print(f"[模型列表API] 异常堆栈:\n{error_trace}")
        logger.error(f"获取模型列表失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'获取模型列表失败: {str(e)}'
        }), 500
    finally:
        print("[模型列表API] ===== 模型列表请求处理完成 =====")
        print("=" * 80)


@anomaly_detection_bp.route('/models/<task_id>/download', methods=['GET'])
def download_model(task_id):
    """下载指定模型的所有文件（打包为ZIP）"""
    try:
        models_dir = Path('models/anomaly_detection')
        task_dir = None
        
        # 查找模型目录 - 遍历所有模型类型目录
        if models_dir.exists():
            for model_type_dir in models_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                potential_task_dir = model_type_dir / task_id
                if potential_task_dir.exists():
                    task_dir = potential_task_dir
                    break
        
        if not task_dir:
            return jsonify({'error': '模型不存在'}), 404
        
        # 创建临时ZIP文件
        import tempfile
        import zipfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加所有模型文件到ZIP
                for file_path in task_dir.glob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)
            
            return send_file(
                tmp_file.name,
                as_attachment=True,
                download_name=f"{task_id}_model.zip",
                mimetype='application/zip'
            )
    
    except Exception as e:
        logger.error(f"下载模型失败 {task_id}: {e}")
        return jsonify({'error': f'下载失败: {str(e)}'}), 500


@anomaly_detection_bp.route('/models/<task_id>/info', methods=['GET'])
def get_model_info(task_id):
    """获取指定模型的详细信息"""
    try:
        models_dir = Path('models/anomaly_detection')
        task_dir = None
        
        # 查找模型目录 - 遍历所有模型类型目录
        if models_dir.exists():
            for model_type_dir in models_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                potential_task_dir = model_type_dir / task_id
                if potential_task_dir.exists():
                    task_dir = potential_task_dir
                    break
        
        if not task_dir:
            return jsonify({'error': '模型不存在'}), 404
        
        # 收集所有信息
        model_info = {
            'task_id': task_id,
            'model_type': task_dir.parent.name,
            'files': {},
            'config': {},
            'training_logs': []
        }
        
        # 配置文件
        config_path = task_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                model_info['config'] = json.load(f)
        
        # 文件信息
        for file_path in task_dir.glob('*'):
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
        logger.error(f"获取模型信息失败 {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': f'获取模型信息失败: {str(e)}'
        }), 500


@anomaly_detection_bp.route('/models/<task_id>/download_package', methods=['GET'])
def download_model_package(task_id):
    """下载指定模型的完整包（供Edge端调用）"""
    try:
        models_dir = Path('models/anomaly_detection')
        task_dir = None
        
        # 查找模型目录 - 遍历所有模型类型目录
        if models_dir.exists():
            for model_type_dir in models_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                potential_task_dir = model_type_dir / task_id
                if potential_task_dir.exists():
                    task_dir = potential_task_dir
                    break
        
        if not task_dir:
            return jsonify({'error': '模型不存在'}), 404
        
        # 创建临时ZIP文件
        import tempfile
        import zipfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加所有模型文件到ZIP
                for file_path in task_dir.glob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)
            
            return send_file(
                tmp_file.name,
                as_attachment=True,
                download_name=f"{task_id}_model_package.zip",
                mimetype='application/zip'
            )
    
    except Exception as e:
        logger.error(f"下载模型包失败 {task_id}: {e}")
        return jsonify({'error': f'下载失败: {str(e)}'}), 500