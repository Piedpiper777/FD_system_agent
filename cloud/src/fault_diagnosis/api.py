"""
故障诊断API模块
处理故障诊断相关的训练和推理请求
"""

from flask import Blueprint, request, jsonify
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
import sys
import os
import shutil
from sklearn.model_selection import train_test_split
from typing import Dict, Any

# 导入评估器
from .core.evaluator import FaultDiagnosisEvaluator

# 创建故障诊断Blueprint
fault_diagnosis_bp = Blueprint('fault_diagnosis', __name__, url_prefix='/api/fault_diagnosis')

# 添加项目路径以导入训练模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入训练组件
try:
    # CNN 1D 分类器模块
    from fault_diagnosis.core.cnn_1d.data_processor import DataProcessor as CNN1DDataProcessor
    from fault_diagnosis.core.cnn_1d.model_builder import ModelBuilder as CNN1DModelBuilder
    from fault_diagnosis.core.cnn_1d.trainer import Trainer as CNN1DTrainer
    
    # LSTM 分类器模块
    from fault_diagnosis.core.lstm.data_processor import DataProcessor as LSTMDataProcessor
    from fault_diagnosis.core.lstm.model_builder import ModelBuilder as LSTMModelBuilder
    from fault_diagnosis.core.lstm.trainer import Trainer as LSTMTrainer
    
    # ResNet 1D 分类器模块
    from fault_diagnosis.core.resnet_1d.data_processor import DataProcessor as ResNet1DDataProcessor
    from fault_diagnosis.core.resnet_1d.model_builder import ModelBuilder as ResNet1DModelBuilder
    from fault_diagnosis.core.resnet_1d.trainer import Trainer as ResNet1DTrainer
    
    import torch
    training_available = True
    logger = logging.getLogger(__name__)
    logger.info("故障诊断训练模块加载成功 (CNN 1D + LSTM + ResNet 1D) - PyTorch版本")
except ImportError as e:
    training_available = False
    logger = logging.getLogger(__name__)
    logger.warning(f"训练模块不可用: {e}")

def _normalize_device_target(value):
    """标准化设备标识（PyTorch风格）"""
    normalized = str(value or 'cpu').strip().lower()
    if normalized in ('gpu', 'cuda'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if normalized in ('ascend', 'npu', 'atlas'):
        # Ascend/NPU 目前不支持，回退到CPU
        logger.warning("Ascend/NPU设备暂不支持，使用CPU")
        return 'cpu'
    return 'cpu'


def _get_torch_device(device_target: str = None) -> torch.device:
    """获取PyTorch设备对象"""
    if device_target is None:
        device_target = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_target = _normalize_device_target(device_target)
    return torch.device(device_target)

# 导入任务管理器
# 始终初始化 training_tasks 字典作为备用存储
training_tasks = {}

try:
    from ..common.task_manager import get_task_manager, TrainingTask, TrainingStatus
except ImportError:
    try:
        from common.task_manager import get_task_manager, TrainingTask, TrainingStatus
    except ImportError:
        # 如果任务管理器不存在，使用简单的任务字典
        def get_task_manager():
            class SimpleTaskManager:
                def __init__(self):
                    self.tasks = {}
                def create_task(self, task_id, config):
                    training_tasks[task_id] = {'status': 'running', 'config': config, 'task_id': task_id}
                    self.tasks[task_id] = type('Task', (), training_tasks[task_id])()
                def get_task(self, task_id):
                    # 直接从全局字典获取最新数据，每次都创建新对象以确保数据同步
                    if task_id in training_tasks:
                        # 使用字典中的最新数据创建对象
                        task_obj = type('Task', (), training_tasks[task_id])()
                        self.tasks[task_id] = task_obj
                        return task_obj
                    # 如果全局字典没有，尝试从self.tasks获取
                    if task_id in self.tasks:
                        return self.tasks[task_id]
                    return None
                def update_task_status(self, task_id, status, message=''):
                    if task_id in training_tasks:
                        training_tasks[task_id]['status'] = status
                        training_tasks[task_id]['message'] = message
                    if task_id in self.tasks:
                        self.tasks[task_id].status = status
                        self.tasks[task_id].message = message
                def add_log(self, task_id, log_message):
                    from datetime import datetime
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'level': 'info',
                        'message': log_message
                    }
                    if task_id in training_tasks:
                        if 'logs' not in training_tasks[task_id]:
                            training_tasks[task_id]['logs'] = []
                        training_tasks[task_id]['logs'].append(log_entry)
                    if task_id in self.tasks:
                        if not hasattr(self.tasks[task_id], 'logs'):
                            self.tasks[task_id].logs = []
                        self.tasks[task_id].logs.append(log_entry)
            return SimpleTaskManager()
        
        # 定义简单的TrainingTask类
        class TrainingTask:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        TrainingStatus = type('TrainingStatus', (), {
            'QUEUED': 'queued',
            'RUNNING': 'running',
            'COMPLETED': 'completed',
            'FAILED': 'failed'
        })


@fault_diagnosis_bp.route('/train', methods=['POST'])
def train_model():
    """故障诊断模型训练API"""
    global training_tasks  # 在函数开头声明global
    
    if not training_available:
        return jsonify({
            'success': False,
            'error': '训练模块不可用，请检查PyTorch安装'
        }), 500
    
    try:
        config = request.get_json()
        if not config:
            return jsonify({
                'success': False,
            'error': '无效的配置数据'
            }), 400
        
        # 获取任务ID
        task_id = config.get('task_id')
        if not task_id:
            return jsonify({
                'success': False,
                'error': '缺少任务ID'
            }), 400
        
        # 创建任务到任务管理器（如果可用）
        try:
            task_manager = get_task_manager()
            # 创建任务对象
            # 只保留 TrainingTask dataclass 支持的字段
            allowed_keys = {
                'data_path', 'dataset_mode', 'epochs', 'batch_size', 'learning_rate',
                'validation_split', 'train_ratio', 'val_ratio', 'test_ratio', 'val_ratio_from_train',
                'sequence_length', 'prediction_horizon', 'hidden_units', 'num_layers', 'dropout',
                'activation', 'bidirectional', 'preprocess_method', 'status',
                'dataset_file', 'train_file', 'val_file', 'test_file',
                'threshold_method', 'percentile', 'residual_metric'
            }
            task_kwargs = {k: v for k, v in config.items() if k in allowed_keys}
            
            task = TrainingTask(
                task_id=task_id,
                module='fault_diagnosis',
                model_type=config.get('model_type', 'cnn_1d_classifier'),
                output_path=str(Path('models') / 'fault_diagnosis' / config.get('model_type', 'cnn_1d_classifier').split('_')[0] / task_id),
                input_dim=5,  # 故障诊断默认5维（RPM, 三轴加速度, 温度），实际会根据工况数动态调整
                status='queued',
                **task_kwargs
            )
            task_manager.tasks[task_id] = task
            # 同时保存到简单字典（作为备用）
            training_tasks[task_id] = {
                'task_id': task_id,
                'status': 'queued',
                'config': config,
                'created_at': datetime.now().isoformat(),
                'progress': 0,
                'message': '训练任务已创建',
                'logs': [{
                    'timestamp': datetime.now().isoformat(),
                    'level': 'info',
                    'message': '正在初始化训练环境...'
                }]
            }
            logger.info(f"任务已创建到任务管理器: {task_id}")
        except Exception as e:
            logger.warning(f"任务管理器不可用，使用简单存储: {e}")
            # 如果任务管理器不可用，使用简单的字典存储
            training_tasks[task_id] = {
                'task_id': task_id,
                'status': 'queued',
                'config': config,
                'created_at': datetime.now().isoformat(),
                'progress': 0,
                'message': '训练任务已创建',
                'logs': [{
                    'timestamp': datetime.now().isoformat(),
                    'level': 'info',
                    'message': '正在初始化训练环境...'
                }]
            }
        
        # 在后台线程中执行训练
        training_thread = threading.Thread(
            target=_execute_training,
            args=(task_id, config),
            daemon=True
        )
        training_thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '训练任务已启动'
        })
        
    except Exception as e:
        logger.error(f"训练任务启动失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'训练任务启动失败: {str(e)}'
        }), 500


def _execute_training(task_id: str, config: dict):
    """执行训练任务（在后台线程中运行）"""
    global training_tasks  # 在函数开头声明global
    
    try:
        logger.info(f"开始执行故障诊断训练任务: {task_id}")
        
        # 更新任务状态为运行中
        try:
            task_manager = get_task_manager()
            task = task_manager.get_task(task_id)
            if task:
                task_manager.update_task_status(task_id, 'running', '训练任务已启动')
                task_manager.add_log(task_id, f'开始执行故障诊断训练任务: {task_id}')
        except Exception as e:
            logger.warning(f"无法更新任务状态: {e}")
            # 更新简单字典
            if 'training_tasks' in globals() and training_tasks and task_id in training_tasks:
                training_tasks[task_id]['status'] = 'running'
                training_tasks[task_id]['message'] = '训练任务已启动'
                if 'logs' not in training_tasks[task_id]:
                    training_tasks[task_id]['logs'] = []
                training_tasks[task_id]['logs'].append({
                    'timestamp': datetime.now().isoformat(),
                    'level': 'info',
                    'message': f'开始执行故障诊断训练任务: {task_id}'
                })
        
        device_target = _normalize_device_target(
            config.get('device_target') or config.get('device') or 'cpu'
        )
        config['device_target'] = device_target
        config['device'] = device_target
        # PyTorch 设备设置
        device = _get_torch_device(device_target)
        logger.info(f"PyTorch 设备已设置: {device}")

        # 1. 从Edge获取数据文件
        logger.info("步骤1: 从Edge获取数据文件")
        edge_data_dir = _fetch_data_from_edge(config, task_id)
        if not edge_data_dir:
            logger.error("从Edge获取数据失败")
            return
        
        # 2. 处理数据（与异常检测一致的方式）
        logger.info("步骤2: 处理数据（工况筛选模式）")
        model_type = config.get('model_type', 'cnn_1d_classifier')
        try:
            task_manager = get_task_manager()
        except:
            task_manager = None
        
        # 处理数据：从元文件读取信息，划分训练/验证集，统一标准化，创建窗口，保存为npz
        _process_condition_filtered_data(config, task_id, task_manager, model_type)
        
        # 3. 从npz文件加载数据
        logger.info("步骤3: 从npz文件加载数据")
        cloud_data_dir = Path('data') / 'fd' / task_id
        train_data_path = cloud_data_dir / 'train.npz'
        dev_data_path = cloud_data_dir / 'dev.npz'
        test_data_path = cloud_data_dir / 'test.npz'
        
        if not train_data_path.exists():
            logger.error(f"训练数据文件不存在: {train_data_path}")
            return
        
        train_npz = np.load(train_data_path)
        train_data = {
            'sequences': train_npz['sequences'],
            'labels': train_npz['labels'] if 'labels' in train_npz else None
        }
        
        val_data = None
        if dev_data_path.exists():
            val_npz = np.load(dev_data_path)
            val_data = {
                'sequences': val_npz['sequences'],
                'labels': val_npz['labels'] if 'labels' in val_npz else None
            }
        
        test_data = None
        if test_data_path.exists():
            test_npz = np.load(test_data_path)
            test_data = {
                'sequences': test_npz['sequences'],
                'labels': test_npz['labels'] if 'labels' in test_npz else None
            }
        
        logger.info(f"训练集: {len(train_data['sequences'])} 个序列")
        if val_data:
            logger.info(f"验证集: {len(val_data['sequences'])} 个序列")
        if test_data:
            logger.info(f"测试集: {len(test_data['sequences'])} 个序列")
        
        # 4. 开始训练
        logger.info("步骤4: 开始模型训练")
        
        if model_type == 'cnn_1d_classifier':
            model = _train_cnn1d_model(config, train_data, val_data, task_id)
        elif model_type == 'lstm_classifier':
            model = _train_lstm_model(config, train_data, val_data, task_id)
        elif model_type == 'resnet_1d_classifier':
            model = _train_resnet1d_model(config, train_data, val_data, task_id)
        else:
            logger.error(f"不支持的模型类型: {model_type}")
            return
        
        # 5. 使用测试集评估模型
        logger.info("步骤5: 使用测试集评估模型")
        
        # 检查测试数据是否存在，如果不存在则使用验证集或跳过评估
        eval_data = test_data
        if eval_data is None:
            logger.warning("测试集不存在，尝试使用验证集进行评估")
            eval_data = val_data
        
        if eval_data is None:
            logger.warning("验证集也不存在，跳过模型评估步骤")
            evaluation_results = {
                'overall': {'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0},
                'per_class': {'precision': [], 'recall': [], 'f1': []},
                'confusion_matrix': [],
                'num_samples': 0,
                'num_classes': len(config.get('labels', ['正常', '内圈故障', '外圈故障'])),
                'note': '无测试集或验证集，评估结果为默认值'
            }
        else:
            evaluation_results = _evaluate_model(
                model=model,
                test_data=eval_data,
                config=config,
                task_id=task_id
            )
        
        logger.info(f"训练和评估任务完成: {task_id}")
        logger.info(f"测试集准确率: {evaluation_results['overall']['accuracy']:.4f}")
        
        # 更新任务状态为已完成（同时更新任务管理器和简单字典）
        try:
            task_manager = get_task_manager()
            task = task_manager.get_task(task_id)
            if task:
                task_manager.update_task_status(task_id, 'completed', '训练任务已完成')
                task_manager.add_log(task_id, f'训练任务完成，测试集准确率: {evaluation_results["overall"]["accuracy"]:.4f}')
                # 保存评估结果到任务对象
                task.evaluation_results = evaluation_results
        except Exception as e:
            logger.warning(f"任务管理器更新失败: {e}")
        
        # 始终更新简单字典（作为备用数据源）
        if 'training_tasks' in globals() and training_tasks and task_id in training_tasks:
            training_tasks[task_id]['status'] = 'completed'
            training_tasks[task_id]['message'] = '训练任务已完成'
            training_tasks[task_id]['test_accuracy'] = evaluation_results['overall']['accuracy']
            training_tasks[task_id]['evaluation_results'] = evaluation_results
            training_tasks[task_id]['evaluation'] = evaluation_results  # 同时添加evaluation字段
            # 添加完成日志
            if 'logs' not in training_tasks[task_id]:
                training_tasks[task_id]['logs'] = []
            training_tasks[task_id]['logs'].append({
                'timestamp': datetime.now().isoformat(),
                'level': 'info',
                'message': f'训练任务完成，测试集准确率: {evaluation_results["overall"]["accuracy"]:.4f}'
            })
            logger.info(f"已保存评估结果到 training_tasks[{task_id}]")
        
    except Exception as e:
        logger.error(f"训练任务执行失败: {str(e)}", exc_info=True)
        
        # 更新任务状态为失败
        try:
            task_manager = get_task_manager()
            task = task_manager.get_task(task_id)
            if task:
                task_manager.update_task_status(task_id, 'failed', f'训练任务失败: {str(e)}')
                task_manager.add_log(task_id, f'训练任务失败: {str(e)}')
        except Exception as e2:
            logger.warning(f"无法更新任务状态: {e2}")
            # 更新简单字典
            if 'training_tasks' in globals() and training_tasks and task_id in training_tasks:
                training_tasks[task_id]['status'] = 'failed'
                training_tasks[task_id]['message'] = f'训练任务失败: {str(e)}'
                training_tasks[task_id]['error'] = str(e)
                # 添加失败日志
                if 'logs' not in training_tasks[task_id]:
                    training_tasks[task_id]['logs'] = []
                training_tasks[task_id]['logs'].append({
                    'timestamp': datetime.now().isoformat(),
                    'level': 'error',
                    'message': f'训练任务失败: {str(e)}'
                })


def _fetch_data_from_edge(config: dict, task_id: str) -> Path:
    """从Edge获取数据文件
    
    Args:
        config: 训练配置
        task_id: 任务ID
        
    Returns:
        云端数据目录路径
    """
    try:
        # 获取Edge端信息
        edge_host = config.get('edge_host', 'localhost')
        edge_port = config.get('edge_port', 5000)
        training_dir = config.get('training_dir', f'edge/data/training/FaultDiagnosis/{task_id}')
        
        # 创建云端数据目录
        cloud_data_dir = Path('data') / 'fd' / task_id
        cloud_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取文件选择配置
        file_selections = config.get('file_selections', [])
        
        # 通过HTTP从Edge下载文件
        import requests
        edge_base_url = f"http://{edge_host}:{edge_port}"
        
        copied_files = []
        seen_files = set()  # 用于去重，避免重复下载同一个文件
        
        # 遍历文件选择配置（新格式：列表）
        if isinstance(file_selections, list):
            # 新格式：列表，每个元素包含 filename, condition_combo, label_index 等
            for file_selection in file_selections:
                filename = file_selection.get('filename') if isinstance(file_selection, dict) else None
                
                if not filename:
                    continue
                
                # 如果文件已经下载过，跳过（避免重复）
                if filename in seen_files:
                    continue
                
                # 从Edge下载数据文件
                download_url = f"{edge_base_url}/fault_diagnosis/train/download_file"
                response = requests.get(
                    download_url,
                    params={'task_id': task_id, 'filename': filename, 'file_type': 'data'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    # 保存数据文件到云端目录
                    dest_file = cloud_data_dir / filename
                    with open(dest_file, 'wb') as f:
                        f.write(response.content)
                    copied_files.append(filename)
                    seen_files.add(filename)
                    logger.info(f"已下载数据文件: {filename}")
                    
                    # 下载对应的元文件（如果存在）
                    meta_filename = filename.replace('.csv', '.json')
                    meta_response = requests.get(
                        download_url,
                        params={'task_id': task_id, 'filename': meta_filename, 'file_type': 'meta'},
                        timeout=30
                    )
                    
                    if meta_response.status_code == 200:
                        # 保存元文件到云端目录
                        meta_dest_file = cloud_data_dir / meta_filename
                        with open(meta_dest_file, 'wb') as f:
                            f.write(meta_response.content)
                        logger.info(f"已下载元文件: {meta_filename}")
                    else:
                        logger.warning(f"元文件不存在或下载失败: {meta_filename}, HTTP {meta_response.status_code}")
                else:
                    logger.warning(f"下载数据文件失败: {filename}, HTTP {response.status_code}")
        elif isinstance(file_selections, dict):
            # 兼容旧格式：字典 {condition_id: {label_index: filename}}
            for condition_id, label_files in file_selections.items():
                for label_index, filename in label_files.items():
                    if not filename:
                        continue
                    
                    # 如果文件已经下载过，跳过
                    if filename in seen_files:
                        continue
                    
                    # 从Edge下载数据文件
                    download_url = f"{edge_base_url}/fault_diagnosis/train/download_file"
                    response = requests.get(
                        download_url,
                        params={'task_id': task_id, 'filename': filename, 'file_type': 'data'},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        # 保存数据文件到云端目录
                        dest_file = cloud_data_dir / filename
                        with open(dest_file, 'wb') as f:
                            f.write(response.content)
                        copied_files.append(filename)
                        seen_files.add(filename)
                        logger.info(f"已下载数据文件: {filename}")
                        
                        # 下载对应的元文件（如果存在）
                        meta_filename = filename.replace('.csv', '.json')
                        meta_response = requests.get(
                            download_url,
                            params={'task_id': task_id, 'filename': meta_filename, 'file_type': 'meta'},
                            timeout=30
                        )
                        
                        if meta_response.status_code == 200:
                            # 保存元文件到云端目录
                            meta_dest_file = cloud_data_dir / meta_filename
                            with open(meta_dest_file, 'wb') as f:
                                f.write(meta_response.content)
                            logger.info(f"已下载元文件: {meta_filename}")
                        else:
                            logger.warning(f"元文件不存在或下载失败: {meta_filename}, HTTP {meta_response.status_code}")
                    else:
                        logger.warning(f"下载数据文件失败: {filename}, HTTP {response.status_code}")
        else:
            logger.error(f"file_selections 格式错误，应为列表或字典，当前类型: {type(file_selections)}")
            return None
        
        if not copied_files:
            logger.error("没有成功下载任何文件")
            return None
        
        logger.info(f"已从Edge获取 {len(copied_files)} 个文件到 {cloud_data_dir}")
        return cloud_data_dir
        
    except Exception as e:
        logger.error(f"从Edge获取数据失败: {str(e)}", exc_info=True)
        return None


def _process_condition_filtered_data(config, task_id, task_manager, model_type):
    """
    处理工况筛选模式的数据（与异常检测一致的方式）：
    1. 读取多个训练文件
    2. 从元数据文件（.json）读取工况信息和标签信息
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
    
    # 获取文件列表（从file_selections中提取）
    file_selections = config.get('file_selections', [])
    train_files = []
    test_files = config.get('test_files', [])
    
    # 从file_selections中提取训练文件列表
    for file_selection in file_selections:
        if isinstance(file_selection, dict):
            filename = file_selection.get('filename')
            if filename:
                train_files.append(filename)
    
    conditions = config.get('conditions', {})  # {key: [value1, value2, ...]} 或 [{name: '转速', values: [...]}, ...]
    validation_split = config.get('validation_split', 0.2)
    sequence_length = config.get('sequence_length', 50)
    stride = config.get('stride', 1)
    
    # 获取标签列表
    labels = config.get('labels', ['正常', '内圈故障', '外圈故障'])
    num_classes = len(labels) if isinstance(labels, list) else config.get('num_classes', 3)
    
    # 转换conditions格式（如果是列表格式，转换为字典格式）
    if isinstance(conditions, list):
        conditions_dict = {}
        for cond in conditions:
            if isinstance(cond, dict) and 'name' in cond:
                conditions_dict[cond['name']] = cond.get('values', [])
        conditions = conditions_dict
    
    # 详细日志
    logger.info(f"接收到的训练文件列表: {train_files}")
    logger.info(f"接收到的测试文件列表: {test_files}")
    
    # 如果文件同时出现在train_files和test_files中，优先将其视为测试文件，从train_files中移除
    if train_files and test_files:
        train_files_set = set(train_files)
        test_files_set = set(test_files)
        overlap = train_files_set & test_files_set
        if overlap:
            logger.warning(f"发现文件同时出现在训练和测试列表中，将从训练列表中移除: {overlap}")
            if task_manager:
                task_manager.add_log(task_id, f'警告: 发现 {len(overlap)} 个文件同时出现在训练和测试列表中，将从训练列表中移除: {list(overlap)}')
            train_files = [f for f in train_files if f not in overlap]
            config['train_files'] = train_files
    
    if task_manager:
        task_manager.add_log(task_id, f'工况筛选模式: {len(train_files)} 个训练文件, {len(test_files)} 个测试文件')
        task_manager.add_log(task_id, f'训练文件列表: {train_files}')
        task_manager.add_log(task_id, f'测试文件列表: {test_files}')
    
    # 查找数据文件目录
    training_data_dir = Path('data') / 'fd' / task_id
    training_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取工况key列表（用于添加特征）
    condition_keys = sorted(list(conditions.keys())) if conditions else []
    if condition_keys:
        if task_manager:
            task_manager.add_log(task_id, f'工况特征: {", ".join(condition_keys)}')
    else:
        if task_manager:
            task_manager.add_log(task_id, '未选择工况，将不添加工况特征')
    
    # 第一步：处理每个训练文件（添加工况、划分训练/验证集）
    all_train_raw_data = []  # 收集所有训练数据（用于fit scaler）
    file_data_list = []  # 保存每个文件的处理信息（包含train_data和val_data）
    
    for filename in train_files:
        # 查找文件
        file_path = training_data_dir / filename
        if not file_path.exists():
            if task_manager:
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
                if pd.api.types.is_numeric_dtype(df[time_col]):
                    df = df.sort_values(by=time_col)
                else:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    df = df.sort_values(by=time_col)
                if task_manager:
                    task_manager.add_log(task_id, f'文件 {filename} 已按时间排序')
            except Exception as e:
                if task_manager:
                    task_manager.add_log(task_id, f'警告: 时间排序失败 {filename}: {e}，使用原始顺序')
        
        # 获取数值列（排除时间戳列）
        numeric_cols = [col for col in df.columns 
                       if pd.api.types.is_numeric_dtype(df[col]) 
                       and col.lower() not in ['timestamp', 'time', '时间']]
        
        if not numeric_cols:
            if task_manager:
                task_manager.add_log(task_id, f'警告: 文件 {filename} 没有数值列，跳过')
            continue
        
        # 从元数据文件读取工况信息和标签信息
        meta_file_path = training_data_dir / (filename.replace('.csv', '.json'))
        condition_values = {}
        file_label_index = 0  # 默认标签索引为0
        
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
                    
                    # 读取标签信息（从tags_label获取标签值，然后在labels列表中找到对应的索引）
                    tags_label = meta_data.get('tags_label', [])
                    if tags_label and len(tags_label) > 0:
                        label_value = tags_label[0].get('value', '') if isinstance(tags_label[0], dict) else str(tags_label[0])
                        # 在labels列表中找到对应的索引
                        try:
                            file_label_index = labels.index(label_value)
                        except ValueError:
                            logger.warning(f"文件 {filename} 的标签 '{label_value}' 不在标签列表中，使用默认标签0")
                            file_label_index = 0
            except Exception as e:
                if task_manager:
                    task_manager.add_log(task_id, f'警告: 读取元数据失败 {filename}: {e}，工况值将使用默认值0.0，标签将使用默认值0')
        else:
            if task_manager:
                task_manager.add_log(task_id, f'警告: 未找到元数据文件 {meta_file_path}，工况值将使用默认值0.0，标签将使用默认值0')
        
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
        
        if task_manager:
            task_manager.add_log(task_id, f'文件 {filename}: 总样本数={n_samples}, 训练集={len(train_data)}, 验证集={len(val_data)}, 标签={labels[file_label_index]}')
        
        # 保存文件信息
        file_data_list.append({
            'filename': filename,
            'train_data': train_data,
            'val_data': val_data,
            'label_index': file_label_index
        })
        
        # 收集所有训练数据（用于fit scaler）
        all_train_raw_data.append(train_data)
    
    if not all_train_raw_data:
        raise ValueError("没有训练数据")
    
    # 第二步：统一fit scaler（只基于所有文件的训练数据）
    all_train_raw = np.vstack(all_train_raw_data)
    scaler = StandardScaler()
    scaler.fit(all_train_raw)
    if task_manager:
        task_manager.add_log(task_id, f'Scaler已fit（基于所有训练数据），特征维度: {all_train_raw.shape[1]}')
    
    # 第三步：对每个文件的训练集和验证集分别标准化、创建窗口
    all_train_sequences = []
    all_val_sequences = []
    all_train_labels = []
    all_val_labels = []
    
    # 创建序列函数（分类任务：只需要sequences，labels单独存储）
    def create_sequences(data):
        if len(data) < sequence_length:
            return np.array([]).reshape(0, sequence_length, data.shape[1])
        sequences = []
        for start in range(0, len(data) - sequence_length + 1, stride):
            end = start + sequence_length
            seq = data[start:end]
            sequences.append(seq)
        return np.stack(sequences) if sequences else np.array([]).reshape(0, sequence_length, data.shape[1])
    
    for file_info in file_data_list:
        # 对训练集和验证集分别标准化
        train_data_scaled = scaler.transform(file_info['train_data'])
        val_data_scaled = scaler.transform(file_info['val_data']) if len(file_info['val_data']) > 0 else np.array([]).reshape(0, train_data_scaled.shape[1])
        
        # 对训练集和验证集分别创建滑动窗口
        train_seqs = create_sequences(train_data_scaled)
        val_seqs = create_sequences(val_data_scaled)
        
        # 为每个序列创建标签（文件级别的标签）
        label_index = file_info['label_index']
        
        if len(train_seqs) > 0:
            all_train_sequences.append(train_seqs)
            train_labels = np.full(len(train_seqs), label_index, dtype=np.int32)
            all_train_labels.append(train_labels)
        if len(val_seqs) > 0:
            all_val_sequences.append(val_seqs)
            val_labels = np.full(len(val_seqs), label_index, dtype=np.int32)
            all_val_labels.append(val_labels)
    
    # 合并所有文件的窗口
    if not all_train_sequences:
        raise ValueError("没有生成任何训练序列")
    
    train_sequences = np.vstack(all_train_sequences)
    train_labels = np.concatenate(all_train_labels)
    
    if all_val_sequences:
        val_sequences = np.vstack(all_val_sequences)
        val_labels = np.concatenate(all_val_labels)
    else:
        val_sequences = np.array([]).reshape(0, sequence_length, train_sequences.shape[2])
        val_labels = np.array([], dtype=np.int32)
    
    if task_manager:
        task_manager.add_log(task_id, f'训练集: {len(train_sequences)} 个序列, 标签分布: {np.bincount(train_labels, minlength=num_classes).tolist()}')
        task_manager.add_log(task_id, f'验证集: {len(val_sequences)} 个序列, 标签分布: {np.bincount(val_labels, minlength=num_classes).tolist()}')
    
    # 保存数据（统一使用npz格式）
    train_data_path = training_data_dir / 'train.npz'
    dev_data_path = training_data_dir / 'dev.npz'
    test_data_path = training_data_dir / 'test.npz'
    
    np.savez(train_data_path, sequences=train_sequences, labels=train_labels)
    np.savez(dev_data_path, sequences=val_sequences, labels=val_labels)
    
    # 保存scaler
    scaler_path = training_data_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 处理测试集文件（如果有）
    if test_files:
        all_test_sequences = []
        all_test_labels = []
        
        if task_manager:
            task_manager.add_log(task_id, f'开始处理测试文件列表: {test_files}')
        
        for filename in test_files:
            if task_manager:
                task_manager.add_log(task_id, f'开始处理测试文件: {filename}')
            
            file_path = training_data_dir / filename
            if not file_path.exists():
                if task_manager:
                    task_manager.add_log(task_id, f'警告: 测试文件未找到 {filename}，跳过')
                continue
            
            df = pd.read_csv(file_path)
            
            # 按时间排序
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
                    if task_manager:
                        task_manager.add_log(task_id, f'警告: 测试文件时间排序失败 {filename}: {e}')
            
            numeric_cols = [col for col in df.columns 
                           if pd.api.types.is_numeric_dtype(df[col]) 
                           and col.lower() not in ['timestamp', 'time', '时间']]
            
            if not numeric_cols:
                if task_manager:
                    task_manager.add_log(task_id, f'警告: 测试文件 {filename} 没有数值列，跳过')
                continue
            
            # 从元数据文件读取工况信息和标签信息
            meta_file_path = training_data_dir / (filename.replace('.csv', '.json'))
            condition_values = {}
            file_label_index = 0
            
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
                        
                        # 读取标签信息
                        tags_label = meta_data.get('tags_label', [])
                        if tags_label and len(tags_label) > 0:
                            label_value = tags_label[0].get('value', '') if isinstance(tags_label[0], dict) else str(tags_label[0])
                            try:
                                file_label_index = labels.index(label_value)
                            except ValueError:
                                logger.warning(f"测试文件 {filename} 的标签 '{label_value}' 不在标签列表中，使用默认标签0")
                                file_label_index = 0
                except Exception as e:
                    if task_manager:
                        task_manager.add_log(task_id, f'警告: 读取测试文件元数据失败 {filename}: {e}')
            
            # 提取特征数据
            feature_data = df[numeric_cols].values.astype(np.float32)
            
            # 添加工况特征
            if condition_keys:
                for key in condition_keys:
                    value = condition_values.get(key, 0.0)
                    condition_feature = np.full((len(feature_data), 1), value, dtype=np.float32)
                    feature_data = np.hstack([feature_data, condition_feature])
            
            # 标准化（使用训练集的scaler）
            test_data_scaled = scaler.transform(feature_data)
            
            # 创建窗口
            test_seqs = create_sequences(test_data_scaled)
            
            if len(test_seqs) > 0:
                all_test_sequences.append(test_seqs)
                test_labels = np.full(len(test_seqs), file_label_index, dtype=np.int32)
                all_test_labels.append(test_labels)
        
        # 合并测试集
        if all_test_sequences:
            test_sequences = np.vstack(all_test_sequences)
            test_labels = np.concatenate(all_test_labels)
            np.savez(test_data_path, sequences=test_sequences, labels=test_labels)
            if task_manager:
                task_manager.add_log(task_id, f'测试集: {len(test_sequences)} 个序列, 标签分布: {np.bincount(test_labels, minlength=num_classes).tolist()}')
    else:
        test_sequences = None
        test_labels = None
    
    # 保存数据集信息
    dataset_info = {
        'task_id': task_id,
        'labels': labels,
        'num_classes': num_classes,
        'conditions': config.get('conditions', []),
        'sequence_length': sequence_length,
        'stride': stride,
        'train_samples': len(train_sequences),
        'val_samples': len(val_sequences),
        'test_samples': len(test_sequences) if test_sequences is not None else 0,
        'train_label_distribution': np.bincount(train_labels, minlength=num_classes).tolist(),
        'val_label_distribution': np.bincount(val_labels, minlength=num_classes).tolist(),
        'test_label_distribution': np.bincount(test_labels, minlength=num_classes).tolist() if test_labels is not None else [],
    }
    
    with open(training_data_dir / 'dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)


def _preprocess_data(config: dict, data_dir: Path, task_id: str) -> list:
    """预处理数据：窗口截取、特征提取、标签解析
    
    Args:
        config: 训练配置
        data_dir: 数据目录
        task_id: 任务ID
        
    Returns:
        处理后的数据列表，每个元素为 (sequences, labels, label_names)
    """
    try:
        sequence_length = config.get('sequence_length', 50)
        stride = config.get('stride', 1)
        # 从标签列表获取分类数量
        labels = config.get('labels', ['正常', '内圈故障', '外圈故障'])
        num_classes = len(labels) if isinstance(labels, list) else config.get('num_classes', 3)
        
        # 获取模型类型
        model_type = config.get('model_type', 'cnn_1d_classifier')
        
        # 获取文件选择配置（新格式：列表，每个元素包含filename, condition_combo, label_index等）
        file_selections = config.get('file_selections', [])
        
        # 获取工况配置
        conditions_config = config.get('conditions', [])  # [{name: '转速', values: [300, 600, 900]}, ...]
        num_conditions = len(conditions_config) if conditions_config else 0
        
        processed_results = []
        
        # 处理每个文件
        for file_selection in file_selections:
            filename = file_selection.get('filename')
            condition_combo = file_selection.get('condition_combo', {})  # {转速: 300, 负载: 50}
            label_index = file_selection.get('label_index', 1)
            
            if not filename:
                continue
            
            file_path = data_dir / filename
            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}")
                continue
            
            try:
                # 为每个文件创建新的数据处理器（避免feature_columns被污染）
                if model_type == 'cnn_1d_classifier':
                    processor = CNN1DDataProcessor(
                        sequence_length=sequence_length,
                        stride=stride,
                        normalize=True,
                        num_classes=num_classes
                    )
                elif model_type == 'resnet_1d_classifier':
                    processor = ResNet1DDataProcessor(
                        sequence_length=sequence_length,
                        stride=stride,
                        normalize=True,
                        num_classes=num_classes
                    )
                else:
                    processor = LSTMDataProcessor(
                        sequence_length=sequence_length,
                        stride=stride,
                        normalize=True,
                        num_classes=num_classes
                    )
                
                # 加载原始数据（返回DataFrame，只包含数值列，排除时间戳等）
                raw_data_df, _ = processor.load_data(file_path)
                original_feature_count = len(processor.feature_columns)
                
                # 在窗口切割之前，将工况值作为特征列添加到数据中
                # 使用通用列名（condition_0, condition_1等），而不是工况名称
                if conditions_config and condition_combo:
                    # 按照conditions_config的顺序获取工况值
                    for idx, cond_config in enumerate(conditions_config):
                        cond_name = cond_config.get('name', '')
                        cond_value = condition_combo.get(cond_name, 0.0)
                        # 使用通用列名：condition_0, condition_1, ...
                        condition_col_name = f'condition_{idx}'
                        raw_data_df[condition_col_name] = cond_value
                    
                    # 更新feature_columns以包含工况列（使用通用列名）
                    processor.feature_columns = list(raw_data_df.columns)
                    logger.info(f"文件 {filename}: 原始特征数={original_feature_count}, 添加工况后特征数={len(processor.feature_columns)}")
                
                # 预处理数据（标准化等）
                processed_data = processor.preprocess(raw_data_df)
                
                # 创建序列（标签由 label_index 决定，不再从文件名提取）
                sequences, _ = processor.create_sequences(processed_data)
                
                # 根据label_index确定标签（label_index从1开始，转换为0开始）
                label = int(label_index) - 1
                
                # 确保标签在有效范围内
                if label < 0 or label >= num_classes:
                    logger.warning(f"标签索引 {label_index} 超出范围（有效范围: 0-{num_classes-1}），使用默认标签0")
                    label = 0
                
                # 为每个序列创建标签数组
                labels = np.full(len(sequences), label, dtype=np.int32)
                
                processed_results.append({
                    'sequences': sequences,
                    'labels': labels,
                    'filename': filename,
                    'condition_combo': condition_combo if condition_combo else {},  # 如果没有工况，使用空字典
                    'label_index': label_index
                })
                
                if condition_combo:
                    logger.info(f"已处理文件: {filename}, 生成 {len(sequences)} 个样本, 工况: {condition_combo}")
                else:
                    logger.info(f"已处理文件: {filename}, 生成 {len(sequences)} 个样本, 无工况")
                
            except Exception as e:
                logger.error(f"处理文件失败 {filename}: {str(e)}", exc_info=True)
                continue
        
        return processed_results
        
    except Exception as e:
        logger.error(f"数据预处理失败: {str(e)}", exc_info=True)
        return []


def _merge_samples(processed_data: list) -> dict:
    """合并所有样本
    
    Args:
        processed_data: 处理后的数据列表
        
    Returns:
        合并后的数据字典
    """
    try:
        all_sequences = []
        all_labels = []
        
        for item in processed_data:
            all_sequences.append(item['sequences'])
            all_labels.append(item['labels'])
        
        # 合并所有序列和标签
        merged_sequences = np.concatenate(all_sequences, axis=0)
        merged_labels = np.concatenate(all_labels, axis=0)
        
        logger.info(f"合并完成: 总样本数 {len(merged_sequences)}, 标签分布: {np.bincount(merged_labels)}")
        
        return {
            'sequences': merged_sequences,
            'labels': merged_labels
        }
        
    except Exception as e:
        logger.error(f"合并样本失败: {str(e)}", exc_info=True)
        return None


def _split_dataset_stratified(
    data: dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> tuple:
    """使用分层抽样划分数据集
    
    Args:
        data: 合并后的数据字典
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        
    Returns:
        (train_data, val_data, test_data)
    """
    try:
        sequences = data['sequences']
        labels = data['labels']
        
        # 验证比例
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            logger.warning(f"比例之和不为1.0 ({total_ratio})，将自动调整")
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 第一次划分：训练集 vs (验证集+测试集)
        # 使用分层抽样确保各类别比例一致
        try:
            train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(
                sequences,
                labels,
                test_size=(1 - train_ratio),
                stratify=labels,
                random_state=random_seed
            )
        except ValueError as e:
            # 如果某个类别样本太少，无法分层抽样，使用普通划分
            logger.warning(f"分层抽样失败，使用普通划分: {str(e)}")
            indices = np.arange(len(sequences))
            np.random.shuffle(indices)
            split_idx = int(len(sequences) * train_ratio)
            train_indices = indices[:split_idx]
            temp_indices = indices[split_idx:]
            train_sequences = sequences[train_indices]
            train_labels = labels[train_indices]
            temp_sequences = sequences[temp_indices]
            temp_labels = labels[temp_indices]
        
        # 第二次划分：验证集 vs 测试集
        val_size = val_ratio / (val_ratio + test_ratio)
        try:
            val_sequences, test_sequences, val_labels, test_labels = train_test_split(
                temp_sequences,
                temp_labels,
                test_size=(1 - val_size),
                stratify=temp_labels,
                random_state=random_seed
            )
        except ValueError as e:
            # 如果某个类别样本太少，使用普通划分
            logger.warning(f"验证集/测试集分层抽样失败，使用普通划分: {str(e)}")
            indices = np.arange(len(temp_sequences))
            np.random.shuffle(indices)
            split_idx = int(len(temp_sequences) * val_size)
            val_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            val_sequences = temp_sequences[val_indices]
            val_labels = temp_labels[val_indices]
            test_sequences = temp_sequences[test_indices]
            test_labels = temp_labels[test_indices]
        
        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {len(train_sequences)} 样本 (标签分布: {np.bincount(train_labels)})")
        logger.info(f"  验证集: {len(val_sequences)} 样本 (标签分布: {np.bincount(val_labels)})")
        logger.info(f"  测试集: {len(test_sequences)} 样本 (标签分布: {np.bincount(test_labels)})")
        
        return (
            {'sequences': train_sequences, 'labels': train_labels},
            {'sequences': val_sequences, 'labels': val_labels},
            {'sequences': test_sequences, 'labels': test_labels}
        )
        
    except Exception as e:
        logger.error(f"数据集划分失败: {str(e)}", exc_info=True)
        return None, None, None


def _save_dataset(data: dict, file_path: Path):
    """保存数据集到NPZ文件（使用numpy格式，更高效）
    
    Args:
        data: 数据字典，包含sequences和labels
        file_path: 保存路径（会自动改为.npz扩展名）
    """
    try:
        sequences = data['sequences']  # (n_samples, seq_len, n_features)
        labels = data['labels']  # (n_samples,)
        
        # 保存为NPZ格式（numpy压缩格式，更高效）
        npz_path = file_path.with_suffix('.npz')
        np.savez_compressed(
            npz_path,
            sequences=sequences,
            labels=labels
        )
        
        logger.info(f"数据集已保存: {npz_path} ({len(sequences)} 样本)")
        
    except Exception as e:
        logger.error(f"保存数据集失败: {str(e)}", exc_info=True)


def _update_training_progress(
    task_id: str,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    learning_rate: float
):
    """更新训练进度到任务管理器"""
    global training_tasks  # 声明global变量
    
    try:
        # 计算进度百分比
        progress = int((epoch / total_epochs) * 100)
        
        # 尝试更新任务管理器
        try:
            task_manager = get_task_manager()
            task = task_manager.get_task(task_id)
            if task:
                # 更新任务属性
                task.current_epoch = epoch
                task.progress = progress
                task.train_loss = train_loss
                task.train_accuracy = train_acc
                task.val_loss = val_loss
                task.val_accuracy = val_acc
                task.learning_rate = learning_rate
                task.current_loss = train_loss
                task.current_accuracy = train_acc
                
                # 添加日志
                task_manager.add_log(
                    task_id,
                    f'Epoch [{epoch}/{total_epochs}] Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}'
                )
        except Exception as e:
            logger.debug(f"任务管理器更新失败: {e}")
        
        # 更新简单字典（作为备用）
        if 'training_tasks' in globals() and training_tasks and task_id in training_tasks:
            training_tasks[task_id].update({
                'current_epoch': epoch,
                'epoch': epoch,
                'progress': progress,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': learning_rate,
                'loss': train_loss,
                'accuracy': train_acc,
            })
            # 确保logs字段存在并添加日志
            if 'logs' not in training_tasks[task_id]:
                training_tasks[task_id]['logs'] = []
            log_message = f'Epoch [{epoch}/{total_epochs}] Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}'
            training_tasks[task_id]['logs'].append({
                'timestamp': datetime.now().isoformat(),
                'level': 'info',
                'message': log_message
            })
            # 只保留最近50条日志
            if len(training_tasks[task_id]['logs']) > 50:
                training_tasks[task_id]['logs'] = training_tasks[task_id]['logs'][-50:]
            
    except Exception as e:
        logger.warning(f"更新训练进度失败: {e}")


def _train_cnn1d_model(config: dict, train_data: dict, val_data: dict, task_id: str):
    """训练CNN 1D模型"""
    try:
        logger.info("开始训练CNN 1D分类器")
        
        # 构建模型
        sequence_length = config.get('sequence_length', 50)
        # 从标签列表获取分类数量
        labels = config.get('labels', ['正常', '内圈故障', '外圈故障'])
        num_classes = len(labels) if isinstance(labels, list) else config.get('num_classes', 3)
        n_features = train_data['sequences'].shape[2]  # 从数据中获取特征数
        
        model_builder = CNN1DModelBuilder()
        model = model_builder.create_model(
            model_type='cnn_1d_classifier',
            input_shape=(sequence_length, n_features),
            num_classes=num_classes,
            num_filters=config.get('num_filters', 64),
            kernel_size=config.get('kernel_size', 3),
            num_conv_layers=config.get('num_conv_layers', 3),
            dropout=config.get('dropout', 0.3),
            activation=config.get('activation', 'relu'),
            use_batch_norm=config.get('use_batch_norm', True),
        )
        
        # 创建训练器
        trainer = CNN1DTrainer(
            model=model,
            learning_rate=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0001),
            clip_grad_norm=config.get('clip_grad_norm', 5.0),
        )
        
        # 创建PyTorch数据加载器
        train_loader = _create_dataloader(
            train_data['sequences'],
            train_data['labels'],
            batch_size=config.get('batch_size', 32),
            shuffle=True
        )
        
        val_loader = _create_dataloader(
            val_data['sequences'],
            val_data['labels'],
            batch_size=config.get('batch_size', 32),
            shuffle=False
        )
        
        # 开始训练（带进度更新）
        num_epochs = config.get('epochs', 50)
        patience = config.get('patience', 10)
        early_stop_mode = 'loss'
        
        # 自定义训练循环，以便更新任务状态
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        final_epoch = 0  # 记录最终训练的epoch数
        
        for epoch in range(num_epochs):
            final_epoch = epoch + 1  # 更新最终epoch数
            
            # 训练一个epoch
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc = trainer.validate(val_loader)
            
            # 更新任务状态
            _update_training_progress(
                task_id=task_id,
                epoch=final_epoch,
                total_epochs=num_epochs,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                learning_rate=config.get('learning_rate', 0.001)
            )
            
            logger.info(
                f"Epoch [{final_epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}"
            )
            
            # 早停检查
            if patience > 0:
                if early_stop_mode == 'loss':
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:  # early_stop_mode == 'acc'
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"⏹️ 早停于第 {final_epoch} 轮")
                    break
        
        # 保存训练指标
        training_metrics = trainer.get_training_metrics()
        training_metrics['epochs_trained'] = final_epoch
        
        # 保存模型
        models_dir = Path('models') / 'fault_diagnosis' / 'cnn_1d' / task_id
        models_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(models_dir / 'model.pth')
        
        # 保存训练指标
        metrics_file = models_dir / 'training_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(training_metrics, f, indent=2, ensure_ascii=False)
        
        # 保存模型配置（方便用户调用）
        model_config = {
            'model_type': 'cnn_1d_classifier',
            'sequence_length': sequence_length,
            'num_classes': num_classes,
            'n_features': n_features,
            'labels': config.get('labels', []),
            'conditions': config.get('conditions', []),
            'num_filters': config.get('num_filters', 64),
            'kernel_size': config.get('kernel_size', 3),
            'num_conv_layers': config.get('num_conv_layers', 3),
            'dense_units': config.get('dense_units', 128),
            'dropout': config.get('dropout', 0.3),
            'learning_rate': config.get('learning_rate', 0.001),
            'batch_size': config.get('batch_size', 32),
            'device_target': config.get('device_target', 'CPU'),
            'epochs_trained': final_epoch,
            'created_at': datetime.now().isoformat()
        }
        config_file = models_dir / 'model_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"CNN 1D模型训练完成，已保存到: {models_dir}")
        
        return model
        
    except Exception as e:
        logger.error(f"CNN 1D模型训练失败: {str(e)}", exc_info=True)
        raise


def _train_lstm_model(config: dict, train_data: dict, val_data: dict, task_id: str):
    """训练LSTM模型
    
    Returns:
        训练好的模型
    """
    try:
        logger.info("开始训练LSTM分类器")
        
        # 构建模型
        sequence_length = config.get('sequence_length', 50)
        # 从标签列表获取分类数量
        labels = config.get('labels', ['正常', '内圈故障', '外圈故障'])
        num_classes = len(labels) if isinstance(labels, list) else config.get('num_classes', 3)
        n_features = train_data['sequences'].shape[2]
        
        model_builder = LSTMModelBuilder()
        model = model_builder.create_model(
            model_type='lstm_classifier',
            input_shape=(sequence_length, n_features),
            num_classes=num_classes,
            hidden_units=config.get('hidden_units', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3),
            activation=config.get('activation', 'tanh'),
            bidirectional=config.get('bidirectional', False),
            use_attention=config.get('use_attention', False),
        )
        
        # 创建训练器
        trainer = LSTMTrainer(
            model=model,
            learning_rate=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0001),
            clip_grad_norm=config.get('clip_grad_norm', 5.0),
        )
        
        # 创建PyTorch数据加载器
        train_loader = _create_dataloader(
            train_data['sequences'],
            train_data['labels'],
            batch_size=config.get('batch_size', 32),
            shuffle=True
        )
        
        val_loader = _create_dataloader(
            val_data['sequences'],
            val_data['labels'],
            batch_size=config.get('batch_size', 32),
            shuffle=False
        )
        
        # 开始训练（带进度更新）
        num_epochs = config.get('epochs', 50)
        patience = config.get('patience', 10)
        early_stop_mode = 'loss'
        
        logger.info(f"训练配置: epochs={num_epochs}, patience={patience}, early_stop_mode=loss")
        
        # 自定义训练循环，以便更新任务状态
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        final_epoch = 0  # 记录最终训练的epoch数
        
        for epoch in range(num_epochs):
            final_epoch = epoch + 1  # 更新最终epoch数
            
            # 训练一个epoch
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc = trainer.validate(val_loader)
            
            # 更新任务状态
            _update_training_progress(
                task_id=task_id,
                epoch=final_epoch,
                total_epochs=num_epochs,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                learning_rate=config.get('learning_rate', 0.001)
            )
            
            logger.info(
                f"Epoch [{final_epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}"
            )
            
            # 早停检查
            if patience > 0:
                if early_stop_mode == 'loss':
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:  # early_stop_mode == 'acc'
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"⏹️ 早停于第 {final_epoch} 轮")
                    break
        
        # 保存训练指标
        training_metrics = trainer.get_training_metrics()
        training_metrics['epochs_trained'] = final_epoch
        
        # 保存模型
        models_dir = Path('models') / 'fault_diagnosis' / 'lstm' / task_id
        models_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(models_dir / 'model.pth')
        
        # 保存训练指标
        metrics_file = models_dir / 'training_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(training_metrics, f, indent=2, ensure_ascii=False)
        
        # 保存模型配置（方便用户调用）
        model_config = {
            'model_type': 'lstm_classifier',
            'sequence_length': sequence_length,
            'num_classes': num_classes,
            'n_features': n_features,
            'labels': config.get('labels', []),
            'conditions': config.get('conditions', []),
            'hidden_units': config.get('hidden_units', 128),
            'num_layers': config.get('num_layers', 2),
            'dropout': config.get('dropout', 0.3),
            'activation': config.get('activation', 'tanh'),
            'bidirectional': config.get('bidirectional', False),
            'use_attention': config.get('use_attention', False),
            'learning_rate': config.get('learning_rate', 0.001),
            'batch_size': config.get('batch_size', 32),
            'device_target': config.get('device_target', 'CPU'),
            'epochs_trained': final_epoch,
            'created_at': datetime.now().isoformat()
        }
        config_file = models_dir / 'model_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"LSTM模型训练完成，已保存到: {models_dir}")
        logger.info(f"训练轮数: {training_metrics.get('epochs_trained', 'N/A')}")
        
        return model
        
    except Exception as e:
        logger.error(f"LSTM模型训练失败: {str(e)}", exc_info=True)
        raise


def _train_resnet1d_model(config: dict, train_data: dict, val_data: dict, task_id: str):
    """
    训练 ResNet 1D 分类模型
    
    Args:
        config: 训练配置
        train_data: 训练数据 {'sequences': np.ndarray, 'labels': np.ndarray}
        val_data: 验证数据 {'sequences': np.ndarray, 'labels': np.ndarray}
        task_id: 任务ID
        
    Returns:
        训练好的模型
    """
    try:
        device_target = _normalize_device_target(
            config.get('device_target') or config.get('device') or 'cpu'
        )
        device = _get_torch_device(device_target)
        logger.info(f"PyTorch 设备已设置: {device}")
        
        logger.info("===== 开始 ResNet 1D 模型训练 =====")
        
        # 获取配置参数
        sequence_length = config.get('sequence_length', 50)
        num_classes = config.get('num_classes', 3)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.001)
        num_epochs = config.get('epochs', 50)
        patience = config.get('patience', 10)
        early_stop_mode = config.get('early_stop_mode', 'loss')
        
        # 从训练数据获取特征维度
        n_features = train_data['sequences'].shape[2]
        logger.info(f"输入形状: (seq_len={sequence_length}, n_features={n_features})")
        logger.info(f"分类数: {num_classes}")
        
        # 创建数据加载器
        train_loader = _create_dataloader(
            train_data['sequences'], 
            train_data['labels'], 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = _create_dataloader(
            val_data['sequences'], 
            val_data['labels'], 
            batch_size=batch_size, 
            shuffle=False
        )
        
        logger.info(f"训练样本数: {len(train_data['sequences'])}")
        logger.info(f"验证样本数: {len(val_data['sequences'])}")
        
        # 构建模型
        model_builder = ResNet1DModelBuilder()
        model = model_builder.create_model(
            model_type='resnet_1d_classifier',
            input_shape=(sequence_length, n_features),
            num_classes=num_classes,
            base_channels=config.get('base_channels', 64),
            block_config=config.get('block_config', 'resnet_small'),
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.3)
        )
        
        model_info = model_builder.get_model_info(model)
        logger.info(f"模型信息: {model_info}")
        
        # 创建训练器
        trainer = ResNet1DTrainer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=config.get('weight_decay', 1e-4),
            clip_grad_norm=config.get('clip_grad_norm', 5.0),
        )
        
        # 训练循环（手动实现以便添加日志）
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        final_epoch = 0
        
        logger.info(f"开始训练，共 {num_epochs} 轮")
        
        for epoch in range(num_epochs):
            final_epoch = epoch + 1
            
            # 训练一个epoch
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc = trainer.validate(val_loader)
            
            # 使用统一的进度更新函数（与CNN 1D和LSTM保持一致）
            _update_training_progress(
                task_id=task_id,
                epoch=epoch + 1,
                total_epochs=num_epochs,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                learning_rate=config.get('learning_rate', 0.001)
            )
            
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}"
            )
            
            # 早停检查
            if patience > 0:
                if early_stop_mode == 'loss':
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:  # early_stop_mode == 'acc'
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"⏹️ 早停于第 {final_epoch} 轮")
                    break
        
        # 保存训练指标
        training_metrics = trainer.get_training_metrics()
        training_metrics['epochs_trained'] = final_epoch
        
        # 保存模型
        models_dir = Path('models') / 'fault_diagnosis' / 'resnet_1d' / task_id
        models_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(models_dir / 'model.pth')
        
        # 保存训练指标
        metrics_file = models_dir / 'training_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(training_metrics, f, indent=2, ensure_ascii=False)
        
        # 保存模型配置（方便用户调用）
        model_config = {
            'model_type': 'resnet_1d_classifier',
            'sequence_length': sequence_length,
            'num_classes': num_classes,
            'n_features': n_features,
            'labels': config.get('labels', []),
            'conditions': config.get('conditions', []),
            'base_channels': config.get('base_channels', 64),
            'block_config': config.get('block_config', 'resnet_small'),
            'kernel_size': config.get('kernel_size', 3),
            'dropout': config.get('dropout', 0.3),
            'learning_rate': config.get('learning_rate', 0.001),
            'batch_size': config.get('batch_size', 32),
            'device_target': config.get('device_target', 'CPU'),
            'epochs_trained': final_epoch,
            'created_at': datetime.now().isoformat()
        }
        config_file = models_dir / 'model_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ResNet 1D模型训练完成，已保存到: {models_dir}")
        logger.info(f"训练轮数: {training_metrics.get('epochs_trained', 'N/A')}")
        
        return model
        
    except Exception as e:
        logger.error(f"ResNet 1D模型训练失败: {str(e)}", exc_info=True)
        raise


def _create_dataloader(sequences: np.ndarray, labels: np.ndarray, batch_size: int = 32, shuffle: bool = True):
    """创建PyTorch数据加载器
    
    Args:
        sequences: 序列数据 (n_samples, seq_len, n_features)
        labels: 标签 (n_samples,)
        batch_size: 批次大小
        shuffle: 是否打乱
        
    Returns:
        PyTorch DataLoader对象
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # 检查输入是否是PyTorch张量，如果是则转换为numpy再重新创建
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # 确保数据类型正确
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    # 创建PyTorch张量
    sequences_tensor = torch.from_numpy(sequences)
    labels_tensor = torch.from_numpy(labels)
    
    # 创建TensorDataset
    dataset = TensorDataset(sequences_tensor, labels_tensor)
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )
    
    return dataloader


def _evaluate_model(
    model: torch.nn.Module,
    test_data: dict,
    config: dict,
    task_id: str
) -> Dict[str, Any]:
    """使用测试集评估模型
    
    Args:
        model: 训练好的模型
        test_data: 测试数据字典
        config: 训练配置
        task_id: 任务ID
        
    Returns:
        评估结果字典
    """
    try:
        logger.info("开始评估模型在测试集上的性能")
        
        # 检查模型是否有效
        if model is None:
            raise ValueError("模型为None，无法进行评估。请检查训练函数是否正确返回了模型。")
        
        # 检查测试数据是否有效
        if test_data is None:
            raise ValueError("测试数据为None，无法进行评估。请确保数据集划分正确生成了测试集。")
        
        if 'sequences' not in test_data or 'labels' not in test_data:
            raise ValueError("测试数据格式错误，需要包含 'sequences' 和 'labels' 字段。")
        
        # 创建测试数据加载器
        test_loader = _create_dataloader(
            test_data['sequences'],
            test_data['labels'],
            batch_size=config.get('batch_size', 32),
            shuffle=False
        )
        
        # 创建评估器
        evaluator = FaultDiagnosisEvaluator(model)
        
        # 执行评估
        # 从标签列表获取分类数量
        labels = config.get('labels', ['正常', '内圈故障', '外圈故障'])
        num_classes = len(labels) if isinstance(labels, list) else config.get('num_classes', 3)
        evaluation_results = evaluator.evaluate(
            test_loader=test_loader,
            num_classes=num_classes
        )
        
        # 保存评估结果（与模型保存在同一目录）
        # 根据模型类型确定目录名：cnn_1d_classifier -> cnn_1d, lstm_classifier -> lstm, resnet_1d_classifier -> resnet_1d
        model_type = config.get('model_type', 'cnn_1d_classifier')
        if model_type == 'cnn_1d_classifier':
            model_dir_name = 'cnn_1d'
        elif model_type == 'lstm_classifier':
            model_dir_name = 'lstm'
        elif model_type == 'resnet_1d_classifier':
            model_dir_name = 'resnet_1d'
        else:
            model_dir_name = model_type.split('_')[0]  # 默认取第一个单词
        
        models_dir = Path('models') / 'fault_diagnosis' / model_dir_name / task_id
        evaluator.save_evaluation_results(
            metrics=evaluation_results,
            save_dir=models_dir,
            task_id=task_id
        )
        
        logger.info(f"评估完成，结果已保存到: {models_dir}")
        logger.info(f"测试集准确率: {evaluation_results['overall']['accuracy']:.4f}")
        logger.info(f"测试集F1分数: {evaluation_results['overall']['f1_macro']:.4f}")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"模型评估失败: {str(e)}", exc_info=True)
        raise


@fault_diagnosis_bp.route('/training_status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    """获取训练状态"""
    global training_tasks  # 确保访问全局变量
    
    try:
        logger.debug(f"获取训练状态: task_id={task_id}")
        logger.debug(f"training_tasks keys: {list(training_tasks.keys()) if training_tasks else 'empty'}")
        
        # 优先从简单字典获取（确保数据最新）
        if training_tasks and task_id in training_tasks:
            task_data = training_tasks[task_id].copy()
            logger.debug(f"从 training_tasks 获取任务: {task_id}")
            logger.debug(f"logs 字段存在: {'logs' in task_data}, 日志数量: {len(task_data.get('logs', []))}")
            
            # 确保包含评估结果
            if 'evaluation_results' in task_data:
                task_data['evaluation'] = task_data['evaluation_results']
            # 确保包含日志
            if 'logs' not in task_data:
                task_data['logs'] = []
            # 确保logs是列表格式
            if not isinstance(task_data.get('logs'), list):
                task_data['logs'] = []
            # 限制日志数量
            task_data['logs'] = task_data['logs'][-50:]
            
            return jsonify({
                'success': True,
                'task': task_data
            })
        
        # 如果简单字典没有，尝试从任务管理器获取
        try:
            task_manager = get_task_manager()
            task = task_manager.get_task(task_id)
            if task:
                logger.debug(f"从任务管理器获取任务: {task_id}")
                # 始终从 training_tasks 字典获取最新的日志（确保数据同步）
                logs = []
                if task_id in training_tasks and 'logs' in training_tasks[task_id]:
                    logs = training_tasks[task_id]['logs'][-50:]
                elif hasattr(task, 'logs') and task.logs:
                    logs = task.logs[-50:] if isinstance(task.logs, list) else []
                
                task_data = {
                    'task_id': task.task_id,
                    'status': task.status.value if hasattr(task.status, 'value') else str(task.status),
                    'config': task.config,
                    'created_at': task.created_at.isoformat() if hasattr(task.created_at, 'isoformat') else str(task.created_at),
                    'progress': getattr(task, 'progress', 0),
                    'epoch': getattr(task, 'current_epoch', 0),
                    'total_epochs': task.config.get('epochs', 50) if task.config else 50,
                    'loss': getattr(task, 'current_loss', None),
                    'accuracy': getattr(task, 'current_accuracy', None),
                    'train_loss': getattr(task, 'train_loss', None),
                    'train_accuracy': getattr(task, 'train_accuracy', None),
                    'val_loss': getattr(task, 'val_loss', None),
                    'val_accuracy': getattr(task, 'val_accuracy', None),
                    'learning_rate': getattr(task, 'learning_rate', None),
                    'logs': logs
                }
                
                # 添加评估结果（如果存在）
                if hasattr(task, 'evaluation_results') and task.evaluation_results:
                    task_data['evaluation_results'] = task.evaluation_results
                    task_data['evaluation'] = task.evaluation_results
                
                return jsonify({
                    'success': True,
                    'task': task_data
                })
        except Exception as e:
            logger.warning(f"任务管理器不可用: {e}")

        logger.warning(f"任务不存在: {task_id}")
        return jsonify({
            'success': False,
            'message': f'任务不存在: {task_id}'
        }), 404

    except Exception as e:
        logger.error(f"获取训练状态失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'获取训练状态失败: {str(e)}'
        }), 500


@fault_diagnosis_bp.route('/pause_training/<task_id>', methods=['POST'])
def pause_training(task_id):
    """暂停训练"""
    try:
        # 尝试使用任务管理器
        try:
            task_manager = get_task_manager()
            task = task_manager.get_task(task_id)
            if task and hasattr(task, 'pause'):
                success = task.pause()
                return jsonify({
                    'success': success,
                    'message': '训练已暂停' if success else '暂停训练失败'
                })
        except Exception as e:
            logger.warning(f"任务管理器暂停功能不可用: {e}")

        # 如果任务管理器不可用，返回不支持
        return jsonify({
            'success': False,
            'message': '暂停功能暂不支持'
        }), 501

    except Exception as e:
        logger.error(f"暂停训练失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'暂停训练失败: {str(e)}'
        }), 500


@fault_diagnosis_bp.route('/stop_training/<task_id>', methods=['POST'])
def stop_training(task_id):
    """停止训练"""
    try:
        # 尝试使用任务管理器
        try:
            task_manager = get_task_manager()
            task = task_manager.get_task(task_id)
            if task and hasattr(task, 'stop'):
                success = task.stop()
                return jsonify({
                    'success': success,
                    'message': '训练已停止' if success else '停止训练失败'
                })
        except Exception as e:
            logger.warning(f"任务管理器停止功能不可用: {e}")

        # 如果任务管理器不可用，返回不支持
        return jsonify({
            'success': False,
            'message': '停止功能暂不支持'
        }), 501

    except Exception as e:
        logger.error(f"停止训练失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'停止训练失败: {str(e)}'
        }), 500


@fault_diagnosis_bp.route('/evaluation_results/<task_id>', methods=['GET'])
def get_evaluation_results(task_id):
    """获取评估结果"""
    try:
        # 尝试从任务管理器获取
        try:
            task_manager = get_task_manager()
            task = task_manager.get_task(task_id)
            if task and hasattr(task, 'evaluation_results') and task.evaluation_results:
                return jsonify({
                    'success': True,
                    'evaluation': task.evaluation_results
                })
        except Exception as e:
            logger.debug(f"任务管理器不可用: {e}")
        
        # 尝试从简单字典获取
        if training_tasks and task_id in training_tasks:
            task = training_tasks[task_id]
            if 'evaluation_results' in task and task['evaluation_results']:
                return jsonify({
                    'success': True,
                    'evaluation': task['evaluation_results']
                })
        
        # 尝试从文件读取评估结果
        try:
            # 扫描所有模型类型目录，找到包含该 task_id 的目录
            model_type_dirs = ['cnn_1d', 'lstm', 'resnet_1d']
            models_base = Path('models') / 'fault_diagnosis'
            
            for model_dir_name in model_type_dirs:
                models_dir = models_base / model_dir_name / task_id
                eval_file = models_dir / f'{task_id}_evaluation_results.json'
                
                if eval_file.exists():
                    with open(eval_file, 'r', encoding='utf-8') as f:
                        evaluation = json.load(f)
                    return jsonify({
                        'success': True,
                        'evaluation': evaluation
                    })
        except Exception as e:
            logger.warning(f"从文件读取评估结果失败: {e}")
        
        return jsonify({
            'success': False,
            'message': '评估结果不可用'
        }), 404
        
    except Exception as e:
        logger.error(f"获取评估结果失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'获取评估结果失败: {str(e)}'
        }), 500


@fault_diagnosis_bp.route('/models/<task_id>/info', methods=['GET'])
def get_model_info(task_id):
    """获取模型信息（包括模型类型）"""
    try:
        # 扫描所有模型类型目录，找到包含该 task_id 的目录
        model_type_dirs = ['cnn_1d', 'lstm', 'resnet_1d']
        models_base = Path('models') / 'fault_diagnosis'
        
        for model_type_dir in model_type_dirs:
            model_dir = models_base / model_type_dir / task_id
            if model_dir.exists():
                # 读取模型配置文件
                config_file = model_dir / 'model_config.json'
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        model_config = json.load(f)
                    return jsonify({
                        'success': True,
                        'model_info': {
                            'task_id': task_id,
                            'model_type': model_config.get('model_type', f'{model_type_dir}_classifier'),
                            'model_type_dir': model_type_dir,
                            'config': model_config
                        }
                    })
                else:
                    # 没有配置文件，根据目录名推断模型类型
                    return jsonify({
                        'success': True,
                        'model_info': {
                            'task_id': task_id,
                            'model_type': f'{model_type_dir}_classifier',
                            'model_type_dir': model_type_dir,
                            'config': {}
                        }
                    })
        
        # 未找到模型
        return jsonify({
            'success': False,
            'error': f'未找到模型: {task_id}'
        }), 404
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'获取模型信息失败: {str(e)}'
        }), 500


@fault_diagnosis_bp.route('/download_model/<task_id>', methods=['GET'])
def download_model(task_id):
    """下载训练好的模型（整个文件夹打包为zip）"""
    try:
        from flask import send_file
        import zipfile
        import tempfile

        # 扫描所有模型类型目录，找到包含该 task_id 的目录
        model_type_dirs = ['cnn_1d', 'lstm', 'resnet_1d']
        models_base = Path('models') / 'fault_diagnosis'
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
            model_type = 'cnn_1d_classifier'  # 默认值
            
            if training_tasks and task_id in training_tasks:
                config = training_tasks[task_id].get('config', {})
                model_type = config.get('model_type', 'cnn_1d_classifier')
            else:
                try:
                    task_manager = get_task_manager()
                    task = task_manager.get_task(task_id)
                    if task and task.config:
                        model_type = task.config.get('model_type', 'cnn_1d_classifier')
                except Exception as e:
                    logger.warning(f"无法从任务管理器获取模型类型: {e}")

            # 根据模型类型确定目录名
            if model_type == 'cnn_1d_classifier':
                model_dir_name = 'cnn_1d'
            elif model_type == 'lstm_classifier':
                model_dir_name = 'lstm'
            elif model_type == 'resnet_1d_classifier':
                model_dir_name = 'resnet_1d'
            else:
                model_dir_name = model_type.split('_')[0]
            
            models_dir = models_base / model_dir_name / task_id
        
        logger.info(f"尝试下载模型目录: {models_dir}, 目录名: {model_dir_name}")

        if not models_dir.exists():
            logger.error(f"模型目录不存在: {models_dir}")
            return jsonify({
                'success': False,
                'message': f'模型目录不存在: {models_dir}'
            }), 404

        # 创建临时zip文件
        zip_filename = f'fault_diagnosis_model_{task_id}.zip'
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

