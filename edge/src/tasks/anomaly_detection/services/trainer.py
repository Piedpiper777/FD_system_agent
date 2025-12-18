"""
异常检测训练服务 - 增强版
支持云端训练，优化数据传输、任务管理和状态监控
"""

import json
import requests
import threading
import time
import uuid
import os
from pathlib import Path
from flask import current_app, request
from werkzeug.utils import secure_filename
import logging
from datetime import datetime


class AnomalyDetectionTrainer:
    """异常检测训练服务 - 增强版
    
    主要功能:
    - 云端训练任务管理和状态监控
    - 文件上传和数据传输优化
    - 训练进度实时跟踪
    - 错误恢复和重试机制
    - 模型下载和本地缓存
    """

    def __init__(self):
        """初始化训练服务"""
        # 延迟获取配置，避免在应用上下文外访问 current_app
        self.cloud_url = None
        self.edge_host = None
        self.edge_port = None
        
        # 尝试从配置中获取，如果不在应用上下文中则使用默认值
        try:
            self.cloud_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
            self.edge_host = current_app.config.get('EDGE_HOST', '10.15.192.149')
            self.edge_port = current_app.config.get('EDGE_PORT', 5000)
        except RuntimeError:
            # 不在 Flask 应用上下文中，使用默认值
            self.cloud_url = 'http://localhost:5001'
            self.edge_host = '10.15.192.149'
            self.edge_port = 5000
        
        # 训练任务状态追踪
        self.training_tasks = {}
        self.task_locks = {}
        
        # 状态监控线程
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # 文件上传缓存
        self.uploaded_files = {}
        
        # 日志记录器
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        print(f"🚀 异常检测训练服务初始化完成")
        print(f"  - 云端地址: {self._get_cloud_url()}")
        print(f"  - 边缘端地址: {self._get_edge_host()}:{self._get_edge_port()}")
    
    def _get_cloud_url(self):
        """获取云端服务URL（延迟获取，确保在应用上下文中）"""
        if self.cloud_url is None:
            try:
                self.cloud_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
            except RuntimeError:
                self.cloud_url = 'http://localhost:5001'
        return self.cloud_url
    
    def _get_edge_host(self):
        """获取边缘端IP（延迟获取，确保在应用上下文中）"""
        if self.edge_host is None:
            try:
                self.edge_host = current_app.config.get('EDGE_HOST', '10.15.192.149')
            except RuntimeError:
                self.edge_host = '10.15.192.149'
        return self.edge_host
    
    def _get_edge_port(self):
        """获取边缘端端口（延迟获取，确保在应用上下文中）"""
        if self.edge_port is None:
            try:
                self.edge_port = current_app.config.get('EDGE_PORT', 5000)
            except RuntimeError:
                self.edge_port = 5000
        return self.edge_port

    def _normalize_device_target(self, value):
        """标准化设备类型，确保MindSpore识别"""
        if not value:
            return 'CPU'
        normalized = str(value).strip().lower()
        if normalized in ('gpu', 'cuda'):
            return 'GPU'
        if normalized in ('ascend', 'npu', 'atlas'):
            return 'Ascend'
        return 'CPU'
        
    def start_monitoring(self):
        """启动训练状态监控线程"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("训练状态监控已启动")
    
    def stop_monitoring(self):
        """停止训练状态监控线程"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            self.logger.info("训练状态监控已停止")
    
    def _monitoring_loop(self):
        """监控循环，定期更新训练状态"""
        while self.monitoring_active:
            try:
                active_tasks = {tid: task for tid, task in self.training_tasks.items() 
                              if task.get('status') in ['training', 'running']}
                
                if active_tasks:
                    self.logger.info(f"监控 {len(active_tasks)} 个活跃训练任务")
                    
                    for task_id, task in active_tasks.items():
                        # 获取云端最新状态
                        cloud_status = self._get_cloud_training_status(task_id)
                        if cloud_status:
                            with self.task_locks.get(task_id, threading.Lock()):
                                self.training_tasks[task_id].update(cloud_status)
                
                time.sleep(10)  # 每10秒更新一次
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(30)  # 错误时等待30秒再重试

    def train(self, model_config):
        """训练模型 - 增强版，支持文件处理和智能参数配置
        
        Args:
            model_config (dict): 训练配置，支持:
                - LSTM预测器专用配置
                - 自动参数推导和验证
                - 智能文件处理
        
        Returns:
            dict: 训练结果
        """
        try:
            self.logger.info(f"开始处理训练请求: {model_config.get('model_type', 'unknown')}")
            
            # 启动监控服务
            self.start_monitoring()
            
            # 参数验证和标准化
            validated_config = self._validate_and_normalize_config(model_config)
            if 'error' in validated_config:
                return validated_config
            
            
            # 处理文件上传
            file_processing_result = self._process_training_files(validated_config)
            if 'error' in file_processing_result:
                return file_processing_result
            
            
            # 更新配置中的文件路径
            validated_config.update(file_processing_result)
            
            
            # 执行云端训练
            return self._execute_cloud_training(validated_config)
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            return {
                'status': 'error',
                'message': f'训练失败: {str(e)}'
            }
    
    def _validate_and_normalize_config(self, config):
        """验证并标准化配置"""
        # 基础验证
        if not isinstance(config, dict):
            return {'status': 'error', 'message': '配置必须是字典格式'}
        
        # 辅助类型转换（改进版：正确处理None和空字符串）
        def to_int(value, default):
            if value is None or value == '':
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def to_float(value, default):
            if value is None or value == '':
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        
        # 设置默认值（优先使用传入的值，只有在参数不存在时才使用默认值）
        device_target = self._normalize_device_target(
            config.get('device_target') or config.get('device') or 'CPU'
        )
        normalized = {
            'module': 'anomaly_detection',
            'model_type': config.get('model_type', 'lstm_predictor'),
            'dataset_mode': config.get('dataset_mode', 'processed_file'),
            'epochs': to_int(config.get('epochs'), 50),
            'batch_size': to_int(config.get('batch_size'), 32),
            'learning_rate': to_float(config.get('learning_rate'), 0.001),
            'weight_decay': to_float(config.get('weight_decay'), 0.0001),
            'validation_split': to_float(config.get('validation_split') or config.get('val_ratio'), 0.2),
            'device_target': device_target,
            'sequence_length': to_int(config.get('sequence_length') or config.get('seq_len'), 50),
            'input_dim': to_int(config.get('input_dim'), 38),
            'output_path': config.get('output_path')
        }
        
        # LSTM特定参数
        if normalized['model_type'] == 'lstm_predictor':
            bidirectional_raw = config.get('bidirectional', False)
            if isinstance(bidirectional_raw, str):
                bidirectional = bidirectional_raw.lower() in ('true', '1', 'yes', 'y')
            else:
                bidirectional = bool(bidirectional_raw)

            # 优先使用传入的值，只有在参数不存在时才使用默认值
            normalized.update({
                'hidden_units': to_int(config.get('hidden_units') or config.get('hidden_dim'), 128),
                'num_layers': to_int(config.get('num_layers'), 2),
                'dropout': to_float(config.get('dropout'), 0.1),
                'activation': config.get('activation', 'tanh'),
                'bidirectional': bidirectional,
                'prediction_horizon': to_int(config.get('prediction_horizon') or config.get('pred_len'), 1)
            })
            
        
        elif normalized['model_type'] == 'lstm_autoencoder':
            # LSTM Autoencoder特定参数
            # 注意：阈值计算相关参数（threshold_method, percentile, residual_metric）在训练完成后单独配置
            normalized.update({
                'hidden_units': to_int(config.get('hidden_units', 128), 128),
                'num_layers': to_int(config.get('num_layers', 2), 2),
                'bottleneck_size': to_int(config.get('bottleneck_size', 64), 64),
                'dropout': to_float(config.get('dropout', 0.1), 0.1),
                'stride': to_int(config.get('stride', 1), 1),
                'random_seed': to_int(config.get('random_seed', 42), 42)
            })
        
        elif normalized['model_type'] == 'cnn_1d_autoencoder':
            # 1D CNN Autoencoder特定参数
            # 注意：阈值计算相关参数（threshold_method, percentile, residual_metric）在训练完成后单独配置
            normalized.update({
                'num_filters': to_int(config.get('num_filters', 64), 64),
                'kernel_size': to_int(config.get('kernel_size', 3), 3),
                'bottleneck_size': to_int(config.get('bottleneck_size', config.get('bottleneck_dim', 64)), 64),
                'num_conv_layers': to_int(config.get('num_conv_layers', config.get('num_layers', 3)), 3),
                'dropout': to_float(config.get('dropout', 0.1), 0.1),
                'activation': config.get('activation', 'relu'),
                'stride': to_int(config.get('stride', 1), 1),
                'random_seed': to_int(config.get('random_seed', 42), 42)
            })
        
        # 数据集拆分参数（根据数据模式调整）
        dataset_mode = normalized['dataset_mode']
        if dataset_mode == 'processed_file':
            # 预处理文件模式：根据validation_split计算比例参数
            validation_split = to_float(config.get('validation_split', 0.2), 0.2)
            normalized['validation_split'] = validation_split
            # 优先使用 dataset_file，如果没有则使用 data_file
            dataset_file = config.get('dataset_file') or config.get('data_file', '')
            normalized['dataset_file'] = dataset_file
            normalized['data_file'] = dataset_file  # 保持兼容性
            
            # 🔧 根据validation_split计算比例参数，而不是使用硬编码值
            # 异常检测模型使用processed_file模式，只需要train_ratio和val_ratio，不需要test_ratio
            train_ratio = 1.0 - validation_split
            normalized['train_ratio'] = train_ratio
            normalized['val_ratio'] = validation_split
            # 不设置test_ratio，因为异常检测模型不使用测试集
        elif dataset_mode == 'condition_filtered':
            # 工况筛选模式：validation_split 在后续处理中设置
            validation_split = to_float(config.get('validation_split', 0.2), 0.2)
            normalized['validation_split'] = validation_split

        # 预处理策略
        if 'preprocess_method' in config:
            normalized['preprocess_method'] = config['preprocess_method']

        # 兼容旧字段，确保 dataset_file / train_file / val_file / test_file 得以保留
        # 但不要在 condition_filtered 模式下设置这些字段，因为该模式使用 train_files 和 test_files
        if dataset_mode != 'condition_filtered':
            for file_key in ['dataset_file', 'train_file', 'val_file', 'test_file']:
                if config.get(file_key):
                    normalized[file_key] = config[file_key]

        # 验证模型类型
        valid_models = ['lstm_predictor', 'cnn_autoencoder', 'cnn_1d_autoencoder', 'lstm_autoencoder']
        if normalized['model_type'] not in valid_models:
            return {
                'status': 'error',
                'message': f'不支持的模型类型: {normalized["model_type"]}，支持: {", ".join(valid_models)}'
            }
        
        # 验证数据集模式（只支持 processed_file 和 condition_filtered）
        valid_modes = ['processed_file', 'condition_filtered']
        if normalized['dataset_mode'] not in valid_modes:
            return {
                'status': 'error',
                'message': f'不支持的数据集模式: {normalized["dataset_mode"]}，支持: {", ".join(valid_modes)}'
            }
        
        # 工况筛选模式：处理train_files和test_files
        if dataset_mode == 'condition_filtered':
            train_files = config.get('train_files', [])
            test_files = config.get('test_files', [])
            conditions = config.get('conditions', {})
            
            if not train_files or len(train_files) == 0:
                return {
                    'status': 'error',
                    'message': '工况筛选模式需要至少选择一个训练集文件'
                }
            
            normalized['train_files'] = train_files
            normalized['test_files'] = test_files if test_files else []
            normalized['conditions'] = conditions
            normalized['validation_split'] = to_float(config.get('validation_split', 0.2), 0.2)
        
        self.logger.debug(f"配置验证通过: {normalized['model_type']} ({normalized['dataset_mode']}模式)")
        return normalized
    
    def _process_training_files(self, config):
        """处理训练数据文件"""
        dataset_mode = config['dataset_mode']
        processed_files = {}
        
        try:
            if dataset_mode == 'condition_filtered':
                # 工况筛选模式：从labeled目录读取文件并上传
                train_files = config.get('train_files', [])
                test_files = config.get('test_files', [])
                
                if not train_files:
                    return {
                        'status': 'error',
                        'message': '工况筛选模式需要至少选择一个训练集文件'
                    }
                
                # 验证文件是否存在（从labeled目录）
                from pathlib import Path
                edge_root = Path(__file__).resolve().parents[4]
                labeled_dir = edge_root / 'data' / 'labeled' / 'AnomalyDetection'
                
                # 验证训练集文件
                for filename in train_files:
                    file_path = labeled_dir / filename
                    if not file_path.exists():
                        return {
                            'status': 'error',
                            'message': f'训练集文件不存在: {filename}'
                        }
                
                # 验证测试集文件（如果有）
                for filename in test_files:
                    file_path = labeled_dir / filename
                    if not file_path.exists():
                        return {
                            'status': 'error',
                            'message': f'测试集文件不存在: {filename}'
                        }
                
                processed_files['train_files'] = train_files
                processed_files['test_files'] = test_files
                processed_files['conditions'] = config.get('conditions', {})
                processed_files['use_condition_filtered'] = True
                self.logger.info(f"工况筛选模式: {len(train_files)} 个训练文件, {len(test_files)} 个测试文件")
                
            elif dataset_mode == 'processed_file':
                # 使用已预处理的数据文件 - LSTM Autoencoder使用
                data_file = config.get('dataset_file') or config.get('data_file')  # 先尝试dataset_file，再尝试data_file
                if not data_file:
                    return {
                        'status': 'error',
                        'message': 'LSTM Autoencoder模式需要选择预处理的数据文件'
                    }
                
                # 验证文件是否存在（使用相对于edge目录的路径）
                from pathlib import Path
                edge_root = Path(__file__).resolve().parents[4]  # 从 trainer.py 到 edge 目录
                processed_dir = edge_root / 'data' / 'processed' / 'AnomalyDetection'
                file_path = processed_dir / data_file
                if not file_path.exists():
                    return {
                        'status': 'error',
                        'message': f'选择的预处理文件不存在: {data_file}'
                    }
                
                processed_files['dataset_file'] = data_file
                processed_files['use_processed_data'] = True  # 标记使用预处理数据
                self.logger.info(f"预处理数据文件: {data_file}")
                
            
            return processed_files
            
        except Exception as e:
            self.logger.error(f"文件处理失败: {e}")
            return {
                'status': 'error',
                'message': f'文件处理失败: {str(e)}'
            }
    
    def _save_uploaded_file(self, file):
        """安全保存上传的文件到edge/data/uploaded目录"""
        if not file or not file.filename:
            raise ValueError("无效的文件")
        
        # 安全文件名处理
        filename = secure_filename(file.filename)
        if not filename:
            filename = f"uploaded_data_{int(time.time())}.csv"
        
        # 使用规范的数据存储结构：edge/data/uploaded
        edge_root = Path(__file__).resolve().parents[4]  # 从 trainer.py 到 edge 目录
        data_uploaded_dir = edge_root / 'data' / 'uploaded'
        data_uploaded_dir.mkdir(parents=True, exist_ok=True)
        
        # 添加时间戳避免文件名冲突
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            timestamp = int(time.time())
            filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            filename = f"{filename}_{int(time.time())}"
        
        file_path = data_uploaded_dir / filename
        file.save(str(file_path))
        
        # 记录文件信息用于后续管理
        self.uploaded_files[filename] = {
            'path': str(file_path),
            'upload_time': time.time(),
            'size': os.path.getsize(file_path),
            'original_name': file.filename,
            'storage_location': 'edge/data/uploaded'
        }
        
        self.logger.info(f"文件保存到边端数据目录: {file_path} ({self.uploaded_files[filename]['size']} bytes)")
        return str(file_path)
    
    def _execute_cloud_training(self, config):
        """执行云端训练 - 增强版"""
        try:
            # 执行云端训练流程
            
            # 1. 先创建训练任务，获取task_id
                # 创建云端训练任务
            cloud_url = self._get_cloud_url()
            
            # 添加边缘端连接信息
            training_config = config.copy()
            training_config.update({
                'edge_host': self._get_edge_host(),
                'edge_port': self._get_edge_port(),
            })

            if not training_config.get('output_path'):
                timestamp = int(time.time())
                training_config['output_path'] = f"models/{training_config['model_type']}_{timestamp}"
            
            # 记录训练请求
            dataset_mode = training_config.get('dataset_mode', 'processed_file')
            self.logger.info(f"创建云端训练任务: {training_config.get('model_type')} ({dataset_mode}模式)")
            
            # 详细日志：显示即将发送给Cloud的文件列表
            train_files = training_config.get('train_files', [])
            test_files = training_config.get('test_files', [])
            self.logger.info(f"Edge端准备发送给Cloud - 训练文件列表: {train_files}")
            self.logger.info(f"Edge端准备发送给Cloud - 测试文件列表: {test_files}")
            print(f"🔍 Edge端准备发送给Cloud - 训练文件列表: {train_files}")
            print(f"🔍 Edge端准备发送给Cloud - 测试文件列表: {test_files}")
            
            # 先创建任务（不启动训练），获取task_id
            create_response = requests.post(
                f"{cloud_url}/api/anomaly_detection/training",
                json=training_config,
                timeout=30
            )
            
            if create_response.status_code != 200:
                error_msg = f"创建训练任务失败: {create_response.status_code}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            result = create_response.json()
            if not result.get('success'):
                error_msg = f"创建训练任务失败: {result.get('error', '未知错误')}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            task_id = result.get('task_id')
            if not task_id:
                error_msg = "创建训练任务成功，但未返回task_id"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # 将task_id添加到config中，用于后续上传文件
            config['task_id'] = task_id
            self.logger.info(f"训练任务已创建: {task_id}")
            
            # 上传数据到云端
            upload_success = self._upload_data_to_cloud(config)
            if not upload_success:
                self.logger.error("数据上传失败，终止训练")
                return {
                    'status': 'error',
                    'message': '数据上传到云端失败'
                }
            
            # 上传完成后，更新training_config中的train_files和test_files（用于后续日志记录）
            if config.get('train_files'):
                training_config['train_files'] = config['train_files']
            if config.get('test_files'):
                training_config['test_files'] = config['test_files']
            if config.get('conditions'):
                training_config['conditions'] = config['conditions']
            
            self.logger.info("数据上传完成，训练已启动")
            
            # 任务已经在创建时启动，直接使用返回的task_id
            # 初始化任务状态
            self.training_tasks[task_id] = {
                'task_id': task_id,
                'cloud_task_id': task_id,
                'status': 'running',
                'config': training_config,
                'start_time': time.time(),
                'epoch': 0,
                'total_epochs': training_config.get('epochs', 50),
                'loss': 0.0,
                'progress': 0,
                'message': '训练任务已启动',
                'logs': [
                    f"✅ 训练任务创建成功 (ID: {task_id})",
                    f"📊 模型类型: {training_config['model_type']}",
                    f"📈 训练轮数: {training_config['epochs']}",
                    f"🔄 批次大小: {training_config['batch_size']}",
                    f"⚡ 学习率: {training_config['learning_rate']}",
                    f"🌐 云端训练已启动..."
                ],
                'paused': False,
                'created_at': datetime.now().isoformat(),
                'model_path': None,
                'performance_metrics': None
            }
            self.task_locks[task_id] = threading.Lock()
            
            self.logger.info(f"训练任务创建成功: {task_id}")
            
            return {
                'status': 'success',
                'success': True,
                'task_id': task_id,
                'cloud_task_id': task_id,
                'message': f'训练任务已提交到云端 (ID: {task_id})',
                'mode': 'cloud',
                'created_at': datetime.now().isoformat(),
                'estimated_duration': self._estimate_training_duration(training_config)
            }
            
        except requests.exceptions.ConnectionError:
            error_msg = f'无法连接到云端: {self._get_cloud_url()}'
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
        except requests.exceptions.Timeout:
            error_msg = '云端请求超时'
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
        except Exception as e:
            error_msg = f'云端训练异常: {str(e)}'
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
    
    def _estimate_training_duration(self, config):
        """估算训练时长"""
        base_time_per_epoch = 30  # 秒
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 32)
        
        # 根据批次大小调整
        if batch_size < 16:
            multiplier = 1.5
        elif batch_size > 64:
            multiplier = 0.8
        else:
            multiplier = 1.0
        
        estimated_seconds = epochs * base_time_per_epoch * multiplier
        return {
            'estimated_seconds': int(estimated_seconds),
            'estimated_minutes': round(estimated_seconds / 60, 1),
            'estimated_hours': round(estimated_seconds / 3600, 2) if estimated_seconds > 3600 else None
        }

    def _train_via_cloud(self, model_config):
        """通过云端进行训练 - 兼容性方法（向后兼容）"""
        # 重定向到新的执行方法
        return self._execute_cloud_training(model_config)

    def get_training_status(self, task_id):
        """获取训练状态 - 增强版，支持实时状态和详细信息"""
        try:
            # 优先从云端获取最新状态
            cloud_status = self._get_cloud_training_status(task_id)
            
            if cloud_status:
                # 更新本地缓存
                if task_id in self.training_tasks:
                    with self.task_locks[task_id]:
                        self.training_tasks[task_id].update(cloud_status)
                
                # 添加边缘端特有的信息
                cloud_status.update({
                    'source': 'cloud',
                    'last_update': time.time(),
                    'edge_cached': task_id in self.training_tasks
                })
                
                return cloud_status
            
            # 云端不可达时使用本地缓存状态
            if task_id not in self.training_tasks:
                return {
                    'status': 'error',
                    'message': '任务不存在且无法连接云端',
                    'source': 'cache',
                    'task_id': task_id
                }
            
            with self.task_locks[task_id]:
                cached_status = self.training_tasks[task_id].copy()
                cached_status.update({
                    'source': 'cache',
                    'last_update': time.time(),
                    'cloud_available': False,
                    'warning': '使用缓存状态，云端连接不可用'
                })
                return cached_status
                
        except Exception as e:
            self.logger.error(f"获取训练状态失败: {e}")
            return {
                'status': 'error',
                'message': f'获取状态失败: {str(e)}',
                'task_id': task_id,
                'source': 'error'
            }
    
    def _get_cloud_training_status(self, task_id):
        """从云端获取真实的训练状态 - 增强版"""
        try:
            # 使用正确的异常检测训练状态API端点
            url = f"{self._get_cloud_url()}/api/anomaly_detection/training_status/{task_id}"
            response = requests.get(
                url, 
                timeout=10,
                headers={'User-Agent': 'EdgeTrainingService/1.0'}
            )
            
            if response.status_code == 200:
                cloud_response = response.json()
                if cloud_response.get('success') and 'task' in cloud_response:
                    cloud_task = cloud_response['task']
                    
                    # 解析云端状态
                    status = cloud_task.get('status', 'unknown')
                    progress = cloud_task.get('progress', 0)
                    message = cloud_task.get('message', '')
                    
                    # 提取训练指标
                    current_epoch = cloud_task.get('current_epoch', 0)
                    completed_epochs = cloud_task.get('completed_epochs', 0)
                    total_epochs = cloud_task.get('config', {}).get('epochs', 50)
                    
                    # 直接从云端获取损失值
                    loss_value = cloud_task.get('loss') or cloud_task.get('train_loss')
                    val_loss = cloud_task.get('val_loss')
                    
                    # 确保损失值是有效的数字
                    if loss_value is not None:
                        try:
                            loss_value = float(loss_value)
                        except (ValueError, TypeError):
                            loss_value = None
                    
                    if val_loss is not None:
                        try:
                            val_loss = float(val_loss)
                        except (ValueError, TypeError):
                            val_loss = None
                    
                    # 如果训练已完成，使用completed_epochs作为当前epoch
                    if status in ['completed', 'threshold_completed'] and completed_epochs > 0:
                        current_epoch = completed_epochs
                    
                    # 智能解析训练消息（作为fallback）
                    if message and 'Epoch' in message and current_epoch == 0:
                        import re
                        # 匹配epoch信息
                        epoch_match = re.search(r'Epoch (\d+)/(\d+)', message)
                        if epoch_match:
                            current_epoch = int(epoch_match.group(1))
                            total_epochs = int(epoch_match.group(2))
                        
                        # 匹配损失值
                        train_loss_match = re.search(r'Train Loss: ([\d.]+)', message)
                        if train_loss_match:
                            loss_value = float(train_loss_match.group(1))
                        
                        val_loss_match = re.search(r'Val Loss: ([\d.]+)', message)
                        if val_loss_match:
                            val_loss = float(val_loss_match.group(1))
                    else:
                        # 从消息中解析损失值（不依赖epoch解析）
                        if message:
                            import re
                            train_loss_match = re.search(r'Train Loss: ([\d.]+)', message)
                            if train_loss_match:
                                loss_value = float(train_loss_match.group(1))
                            
                            val_loss_match = re.search(r'Val Loss: ([\d.]+)', message)
                            if val_loss_match:
                                val_loss = float(val_loss_match.group(1))
                    
                    # 计算训练进度百分比
                    if total_epochs > 0:
                        epoch_progress = (current_epoch / total_epochs) * 100
                        progress = max(progress, epoch_progress)
                    
                    # 获取模型信息
                    model_path = cloud_task.get('model_save_path')
                    config = cloud_task.get('config', {})
                    
                    # 构建详细状态响应
                    detailed_status = {
                        'task_id': task_id,
                        'cloud_task_id': cloud_task.get('id'),
                        'status': 'training' if status == 'running' else status,
                        'epoch': current_epoch,
                        'total_epochs': total_epochs,
                        'progress': round(progress, 2),
                        'message': message,
                        'loss': loss_value,
                        'val_loss': val_loss,
                        'logs': cloud_task.get('logs', []) + ([message] if message and message not in cloud_task.get('logs', []) else []),
                        'created_at': cloud_task.get('created_at'),
                        'updated_at': cloud_task.get('updated_at'),
                        'model_path': model_path,
                        'threshold_value': cloud_task.get('threshold_value'),
                        'threshold_path': cloud_task.get('threshold_path'),
                        'threshold_metadata': cloud_task.get('threshold_metadata'),
                        'scaler_path': cloud_task.get('scaler_path'),
                        'config': config,
                        'performance': {
                            'train_loss': loss_value,
                            'val_loss': val_loss,
                            'epoch_progress': f"{current_epoch}/{total_epochs}",
                            'completion_rate': f"{progress:.1f}%"
                        }
                    }
                    
                    # 如果训练完成，添加额外信息
                    if status in ['completed', 'finished']:
                        detailed_status.update({
                            'completed_at': cloud_task.get('updated_at'),
                            'final_loss': loss_value,
                            'final_val_loss': val_loss,
                            'model_ready': bool(model_path),
                            'download_url': f"/api/models/{os.path.basename(model_path)}/download" if model_path else None
                        })
                    
                    self.logger.debug(f"云端状态更新 - 任务 {task_id}: Epoch {current_epoch}/{total_epochs}, 进度: {progress:.1f}%")
                    return detailed_status
                    
            elif response.status_code == 404:
                self.logger.warning(f"云端任务不存在: {task_id}")
                return {
                    'status': 'not_found',
                    'message': '云端任务不存在',
                    'task_id': task_id
                }
                
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"云端连接失败: {self._get_cloud_url()}")
        except requests.exceptions.Timeout:
            self.logger.warning("云端请求超时")
        except Exception as e:
            self.logger.error(f"获取云端状态异常: {e}")
        
        return None
    
    def download_model(self, task_id, local_path=None):
        """下载训练完成的模型"""
        try:
            # 获取任务状态
            status = self.get_training_status(task_id)
            if not status or status.get('status') != 'completed':
                return {
                    'success': False,
                    'error': '任务未完成或不存在'
                }
            
            model_path = status.get('model_path')
            if not model_path:
                return {
                    'success': False,
                    'error': '模型文件不存在'
                }
            
            # 从云端获取模型信息，确定模型类型
            try:
                info_response = requests.get(f"{self._get_cloud_url()}/api/anomaly_detection/models/{task_id}/info", timeout=10)
                model_type_dir = 'lstm_prediction'  # 默认值
                
                if info_response.status_code == 200:
                    info_data = info_response.json()
                    if info_data.get('success') and 'model_info' in info_data:
                        cloud_model_type = info_data['model_info'].get('model_type', '')
                        # 将云端模型类型映射到本地目录名
                        if cloud_model_type == 'lstm_predictor':
                            model_type_dir = 'lstm_prediction'
                        elif cloud_model_type == 'lstm_autoencoder':
                            model_type_dir = 'lstm_autoencoder'
                        elif cloud_model_type == 'cnn_1d_autoencoder':
                            model_type_dir = 'cnn_1d_autoencoder'
                        else:
                            # 如果云端返回的是目录名格式，直接使用
                            model_type_dir = cloud_model_type if cloud_model_type else 'lstm_prediction'
            except Exception as e:
                # 如果获取模型信息失败，回退到从status中获取
                self.logger.warning(f"无法从云端获取模型类型，使用默认值: {e}")
                config = status.get('config', {})
                model_type = config.get('model_type', 'lstm_predictor')
                if model_type == 'lstm_autoencoder':
                    model_type_dir = 'lstm_autoencoder'
                elif model_type == 'cnn_1d_autoencoder':
                    model_type_dir = 'cnn_1d_autoencoder'
                else:
                    model_type_dir = 'lstm_prediction'
            
            # 下载模型文件（使用download_package接口，下载完整模型包）
            download_url = f"{self._get_cloud_url()}/api/anomaly_detection/models/{task_id}/download_package"
            
            response = requests.get(download_url, stream=True, timeout=60)
            if response.status_code == 200:
                # 确定本地保存目录
                if local_path is None:
                    # 获取模型目录路径（避免在应用上下文外访问）
                    try:
                        model_folder = current_app.config.get('MODEL_FOLDER', './models')
                    except RuntimeError:
                        # 不在应用上下文中，使用默认路径
                        edge_root = Path(__file__).resolve().parents[4]
                        model_folder = edge_root / 'models'
                    local_path = Path(model_folder) / 'anomaly_detection' / model_type_dir / task_id
                    local_path.mkdir(parents=True, exist_ok=True)
                else:
                    local_path = Path(local_path)
                    local_path.mkdir(parents=True, exist_ok=True)
                
                # 保存ZIP文件到临时位置
                import tempfile
                import zipfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                    tmp_zip_path = tmp_file.name
                
                # 解压ZIP文件到本地目标目录
                extracted_files = []
                with zipfile.ZipFile(tmp_zip_path, 'r') as zip_file:
                    zip_file.extractall(local_path)
                    extracted_files = zip_file.namelist()
                
                # 清理临时ZIP文件
                os.unlink(tmp_zip_path)
                
                self.logger.info(f"模型下载成功: {local_path} ({len(extracted_files)} 个文件)")
                return {
                    'success': True,
                    'local_path': str(local_path),
                    'files_count': len(extracted_files),
                    'extracted_files': extracted_files
                }
            else:
                return {
                    'success': False,
                    'error': f'下载失败: HTTP {response.status_code}'
                }
                
        except Exception as e:
            self.logger.error(f"模型下载失败: {e}")
            return {
                'success': False,
                'error': f'下载失败: {str(e)}'
            }
    
    def get_task_logs(self, task_id, lines=50):
        """获取任务详细日志"""
        try:
            response = requests.get(
                f"{self._get_cloud_url()}/api/training/{task_id}/logs",
                params={'lines': lines},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return {
                        'success': True,
                        'logs': result.get('logs', []),
                        'total_lines': len(result.get('logs', []))
                    }
            
            # 回退到本地日志
            if task_id in self.training_tasks:
                local_logs = self.training_tasks[task_id].get('logs', [])
                return {
                    'success': True,
                    'logs': local_logs[-lines:] if len(local_logs) > lines else local_logs,
                    'total_lines': len(local_logs),
                    'source': 'local_cache'
                }
            
            return {
                'success': False,
                'error': '无法获取日志'
            }
            
        except Exception as e:
            self.logger.error(f"获取日志失败: {e}")
            return {
                'success': False,
                'error': f'获取日志失败: {str(e)}'
            }
    
    def cleanup_old_files(self, max_age_hours=24):
        """清理过期的上传文件"""
        try:
            current_time = time.time()
            cleanup_count = 0
            
            for filename, file_info in list(self.uploaded_files.items()):
                age_hours = (current_time - file_info['upload_time']) / 3600
                if age_hours > max_age_hours:
                    try:
                        if os.path.exists(file_info['path']):
                            os.remove(file_info['path'])
                        del self.uploaded_files[filename]
                        cleanup_count += 1
                        self.logger.info(f"清理过期文件: {filename}")
                    except Exception as e:
                        self.logger.error(f"清理文件失败 {filename}: {e}")
            
            if cleanup_count > 0:
                self.logger.info(f"文件清理完成，清理了 {cleanup_count} 个文件")
            
            return {
                'success': True,
                'cleaned_count': cleanup_count,
                'remaining_count': len(self.uploaded_files)
            }
            
        except Exception as e:
            self.logger.error(f"文件清理异常: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    def pause_training(self, task_id):
        """暂停训练 - 增强版"""
        try:
            if task_id in self.training_tasks:
                with self.task_locks[task_id]:
                    self.training_tasks[task_id]['paused'] = True
                    self.training_tasks[task_id]['logs'].append(f'[{time.strftime("%H:%M:%S")}] 训练暂停请求已发送')
                
                # 向云端发送暂停请求
                task = self.training_tasks[task_id]
                cloud_task_id = task.get('cloud_task_id')
                
                if cloud_task_id:
                    try:
                        response = requests.post(
                            f"{self._get_cloud_url()}/api/training/{cloud_task_id}/cancel",
                            timeout=10,
                            headers={'User-Agent': 'EdgeTrainingService/1.0'}
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get('success'):
                                self.logger.info(f"云端训练暂停成功: {task_id}")
                                return {
                                    'status': 'success', 
                                    'message': '训练已暂停',
                                    'cloud_confirmed': True
                                }
                        
                        # 云端暂停失败，但本地标记成功
                        self.logger.warning(f"云端暂停响应异常: {response.status_code}")
                        return {
                            'status': 'partial_success',
                            'message': '本地标记为暂停，但云端确认失败',
                            'cloud_confirmed': False
                        }
                        
                    except Exception as e:
                        self.logger.error(f"云端暂停请求失败: {e}")
                        return {
                            'status': 'partial_success',
                            'message': f'本地标记为暂停，云端请求失败: {str(e)}',
                            'cloud_confirmed': False
                        }
                else:
                    return {
                        'status': 'success', 
                        'message': '训练已暂停（本地）',
                        'cloud_confirmed': False
                    }
            else:
                return {'status': 'error', 'message': '训练任务不存在'}
                
        except Exception as e:
            self.logger.error(f"暂停训练失败: {e}")
            return {'status': 'error', 'message': f'暂停失败: {str(e)}'}

    def stop_training(self, task_id):
        """停止训练 - 增强版"""
        try:
            if task_id in self.training_tasks:
                with self.task_locks[task_id]:
                    self.training_tasks[task_id]['status'] = 'stopped'
                    self.training_tasks[task_id]['logs'].append(f'[{time.strftime("%H:%M:%S")}] 训练停止请求已发送')
                
                # 向云端发送停止请求
                task = self.training_tasks[task_id]
                cloud_task_id = task.get('cloud_task_id')
                
                if cloud_task_id:
                    try:
                        response = requests.post(
                            f"{self._get_cloud_url()}/api/training/{cloud_task_id}/cancel",
                            timeout=10,
                            headers={'User-Agent': 'EdgeTrainingService/1.0'}
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get('success'):
                                self.logger.info(f"云端训练停止成功: {task_id}")
                                
                                # 清理本地文件（如果有）
                                self._cleanup_task_files(task_id)
                                
                                return {
                                    'status': 'success', 
                                    'message': '训练已停止',
                                    'cloud_confirmed': True
                                }
                        
                        self.logger.warning(f"云端停止响应异常: {response.status_code}")
                        return {
                            'status': 'partial_success',
                            'message': '本地标记为停止，但云端确认失败',
                            'cloud_confirmed': False
                        }
                        
                    except Exception as e:
                        self.logger.error(f"云端停止请求失败: {e}")
                        return {
                            'status': 'partial_success',
                            'message': f'本地标记为停止，云端请求失败: {str(e)}',
                            'cloud_confirmed': False
                        }
                else:
                    self._cleanup_task_files(task_id)
                    return {
                        'status': 'success', 
                        'message': '训练已停止（本地）',
                        'cloud_confirmed': False
                    }
            else:
                return {'status': 'error', 'message': '训练任务不存在'}
                
        except Exception as e:
            self.logger.error(f"停止训练失败: {e}")
            return {'status': 'error', 'message': f'停止失败: {str(e)}'}
    
    def _cleanup_task_files(self, task_id):
        """清理任务相关的临时文件"""
        try:
            if task_id in self.training_tasks:
                task = self.training_tasks[task_id]
                config = task.get('config', {})
                
                # 清理上传的数据文件
                for key in ['dataset_file', 'train_file', 'val_file', 'test_file']:
                    filename = config.get(key)
                    if filename and filename in self.uploaded_files:
                        file_info = self.uploaded_files[filename]
                        try:
                            if os.path.exists(file_info['path']):
                                os.remove(file_info['path'])
                            del self.uploaded_files[filename]
                            self.logger.info(f"清理任务文件: {filename}")
                        except Exception as e:
                            self.logger.warning(f"清理文件失败 {filename}: {e}")
                            
        except Exception as e:
            self.logger.error(f"清理任务文件失败: {e}")
    
    def get_training_summary(self):
        """获取训练任务摘要"""
        try:
            summary = {
                'total_tasks': len(self.training_tasks),
                'active_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'paused_tasks': 0,
                'uploaded_files': len(self.uploaded_files),
                'cloud_url': self._get_cloud_url(),
                'monitoring_active': self.monitoring_active
            }
            
            for task in self.training_tasks.values():
                status = task.get('status', 'unknown')
                if status in ['running', 'training']:
                    summary['active_tasks'] += 1
                elif status in ['completed', 'finished']:
                    summary['completed_tasks'] += 1
                elif status in ['failed', 'error']:
                    summary['failed_tasks'] += 1
                elif task.get('paused'):
                    summary['paused_tasks'] += 1
            
            # 计算存储使用情况
            total_storage = sum(file_info['size'] for file_info in self.uploaded_files.values())
            summary['storage_used_mb'] = round(total_storage / (1024 * 1024), 2)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取训练摘要失败: {e}")
            return {
                'error': str(e),
                'total_tasks': 0,
                'cloud_url': self.cloud_url
            }
    
    def calculate_threshold(self, task_id, threshold_params=None):
        """计算异常检测阈值"""
        try:
            if threshold_params is None:
                threshold_params = {}
            
            response = requests.post(
                f"{self._get_cloud_url()}/api/anomaly_detection/calculate_threshold/{task_id}",
                json=threshold_params,
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json() if response.content else {'error': '未知错误'}
                return {
                    'success': False, 
                    'error': error_data.get('error', f'HTTP {response.status_code}')
                }
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f'网络请求失败: {str(e)}'}

    def _upload_data_to_cloud(self, config):
        """上传训练数据和测试数据到云端的训练数据目录"""
        print("="*50)
        print("🔥 _upload_data_to_cloud 函数被调用了！")
        try:
            dataset_mode = config.get('dataset_mode', 'processed_file')
            self.logger.info(f"开始上传数据到云端 ({dataset_mode}模式)")
            
            dataset_mode = config.get('dataset_mode', 'processed_file')
            
            # 工况筛选模式：上传多个文件
            if dataset_mode == 'condition_filtered':
                train_files = config.get('train_files', [])
                test_files = config.get('test_files', [])
                
                if not train_files:
                    self.logger.error("工况筛选模式没有指定训练文件")
                    return False
                
                edge_root = Path(__file__).resolve().parents[4]
                labeled_dir = edge_root / 'data' / 'labeled' / 'AnomalyDetection'
                data_training_dir = edge_root / 'data' / 'training' / 'AnomalyDetection'
                data_training_dir.mkdir(parents=True, exist_ok=True)
                
                # 上传所有训练文件
                uploaded_train_files = []
                for filename in train_files:
                    upload_success = self._upload_single_file_to_cloud(
                        filename, labeled_dir, data_training_dir, config, None, 'train'
                    )
                    if upload_success:
                        uploaded_train_files.append(filename)
                    else:
                        self.logger.warning(f"训练文件上传失败: {filename}")
                
                if not uploaded_train_files:
                    self.logger.error("所有训练文件上传失败")
                    return False
                
                # 上传所有测试文件（如果有）
                uploaded_test_files = []
                for filename in test_files:
                    upload_success = self._upload_single_file_to_cloud(
                        filename, labeled_dir, data_training_dir, config, None, 'test'
                    )
                    if upload_success:
                        uploaded_test_files.append(filename)
                    else:
                        self.logger.warning(f"测试文件上传失败: {filename}")
                
                # 更新config中的文件列表
                config['train_files'] = uploaded_train_files
                config['test_files'] = uploaded_test_files
                
                # 详细日志：显示上传后的文件列表
                self.logger.info(f"Edge端文件上传完成 - 训练文件列表: {uploaded_train_files}")
                self.logger.info(f"Edge端文件上传完成 - 测试文件列表: {uploaded_test_files}")
                print(f"🔍 Edge端文件上传完成 - 训练文件列表: {uploaded_train_files}")
                print(f"🔍 Edge端文件上传完成 - 测试文件列表: {uploaded_test_files}")
                
                return True
            
            # 原有模式：单个文件上传
            dataset_file = config.get('dataset_file')
            test_file = config.get('test_file')  # 测试集文件（可选）
            
            print(f"🔥 dataset_file = {dataset_file}")
            print(f"🔥 test_file = {test_file}")
            self.logger.info(f"从config中获取的dataset_file: {dataset_file}")
            self.logger.info(f"从config中获取的test_file: {test_file}")
            
            if not dataset_file:
                print("🔥 没有指定数据文件 - 这是错误来源！")
                self.logger.error("没有指定数据文件")
                return False

            # 获取路径
            edge_root = Path(__file__).resolve().parents[4]  # 从 trainer.py 到 edge 目录
            data_processed_dir = edge_root / 'data' / 'processed' / 'AnomalyDetection'
            data_training_dir = edge_root / 'data' / 'training' / 'AnomalyDetection'
            data_training_dir.mkdir(parents=True, exist_ok=True)
            
            # 上传训练数据文件
            self.logger.info(f"准备处理训练文件: {dataset_file}")
            upload_success = self._upload_single_file_to_cloud(
                dataset_file, data_processed_dir, data_training_dir, config, 'dataset_file'
            )
            if not upload_success:
                return False
            
            # 上传测试数据文件（如果有）
            if test_file:
                self.logger.info(f"准备处理测试文件: {test_file}")
                test_upload_success = self._upload_single_file_to_cloud(
                    test_file, data_processed_dir, data_training_dir, config, 'test_file'
                )
                if not test_upload_success:
                    self.logger.warning(f"测试文件上传失败，但训练可以继续: {test_file}")
                    # 测试文件上传失败不影响训练，只是不会有评估结果
            
            return True
                
        except Exception as e:
            self.logger.error(f"数据上传异常: {e}")
            import traceback
            self.logger.error(f"异常堆栈: {traceback.format_exc()}")
            return False
    
    def _upload_single_file_to_cloud(self, filename, source_dir, training_dir, config, config_key=None, file_type='train'):
        """上传单个文件到云端
        
        Args:
            filename: 文件名
            source_dir: 源目录（edge/data/processed/AnomalyDetection 或 edge/data/labeled/AnomalyDetection）
            training_dir: 训练目录（edge/data/training/AnomalyDetection）
            config: 配置字典，用于更新云端文件名
            config_key: 配置中的键名（dataset_file 或 test_file），工况筛选模式时为None
            file_type: 文件类型（'train' 或 'test'），用于工况筛选模式
        
        Returns:
            bool: 上传是否成功
        """
        try:
            import shutil
            
            # 第一步：从processed目录复制到edge的training目录
            processed_file_path = source_dir / filename
            self.logger.info(f"源文件路径: {processed_file_path}")
            
            if not processed_file_path.exists():
                self.logger.error(f"预处理文件不存在: {processed_file_path}")
                return False
            
            # 复制文件到training目录
            training_file_path = training_dir / filename
            shutil.copy2(processed_file_path, training_file_path)
            self.logger.info(f"文件已复制到training目录: {training_file_path}")
            
            # 第二步：从edge/data/training/AnomalyDetection上传到cloud/data/ad
            file_size = training_file_path.stat().st_size
            self.logger.info(f"文件大小: {file_size} bytes")
            
            # 上传到云端训练数据目录
            upload_url = f"{self._get_cloud_url()}/api/anomaly_detection/upload_data"
            self.logger.info(f"上传到云端训练数据目录: {upload_url}")
            
            # 获取task_id（如果已创建任务）
            task_id = config.get('task_id', '')
            
            with open(training_file_path, 'rb') as f:
                files = {
                    'file': (filename, f, 'text/csv')
                }
                data = {}
                if task_id:
                    data['task_id'] = task_id
                
                self.logger.info(f"开始上传 {filename} 到云端训练数据目录... [task_id: {task_id or 'N/A'}]")
                response = requests.post(
                    upload_url,
                    files=files,
                    data=data,  # 传递task_id
                    timeout=120,  # 增加超时时间用于大文件上传
                    headers={'User-Agent': 'EdgeTrainingService/1.0'}
                )
                
                self.logger.info(f"云端响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    saved_filename = result.get('saved_filename', filename)
                    self.logger.debug(f"文件上传成功: {filename}")
                    # 更新配置中的文件名为云端保存的文件名（如果提供了config_key）
                    if config_key:
                        config[config_key] = saved_filename
                    
                    # 上传对应的元数据文件（如果存在）
                    # 元数据文件在 meta 目录，不在 source_dir
                    meta_filename = filename.replace('.csv', '.json')
                    edge_root = Path(__file__).resolve().parents[4]
                    meta_dir = edge_root / 'data' / 'meta' / 'AnomalyDetection'
                    meta_file_path = meta_dir / meta_filename
                    
                    if meta_file_path.exists():
                        try:
                            with open(meta_file_path, 'rb') as meta_f:
                                meta_data = {}
                                if task_id:
                                    meta_data['task_id'] = task_id
                                meta_response = requests.post(
                                    self._get_cloud_url() + '/api/anomaly_detection/upload_data',
                                    files={'file': (meta_filename, meta_f, 'application/json')},
                                    data=meta_data,
                                    timeout=300
                                )
                                if meta_response.status_code == 200:
                                    self.logger.debug(f"元数据文件 {meta_filename} 上传成功")
                                else:
                                    self.logger.warning(f"元数据文件 {meta_filename} 上传失败: HTTP {meta_response.status_code}")
                        except Exception as e:
                            self.logger.warning(f"上传元数据文件失败 {meta_filename}: {e}")
                    
                    return True
                else:
                    self.logger.error(f"云端 {config_key} 上传失败: {result.get('error', '未知错误')}")
                    return False
            else:
                self.logger.error(f"上传请求失败: HTTP {response.status_code}")
                self.logger.error(f"错误详情: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"文件 {filename} 上传异常: {e}")
            return False
    
    def __del__(self):
        """析构函数，清理资源"""
        try:
            self.stop_monitoring()
        except:
            pass
