"""
RUL预测训练服务
仅支持云端训练 - 所有训练任务通过 HTTP 转发到云端
"""

import requests
import shutil
import json
from pathlib import Path
from flask import current_app
import logging
from datetime import datetime


class RULPredictionTrainer:
    """RUL预测训练服务 - 仅支持云端训练"""

    def __init__(self):
        """初始化训练服务"""
        # 延迟获取配置，避免在应用上下文外访问 current_app
        self.cloud_url = None
        self.logger = logging.getLogger(__name__)
    
    def _normalize_device_target(self, value):
        """标准化设备标识"""
        normalized = str(value or 'cuda:0').strip().lower()
        if normalized in ('gpu', 'cuda'):
            return 'cuda:0'
        if normalized.startswith('cuda'):
            return normalized
        return 'cpu'
    
    def _get_cloud_url(self):
        """获取云端服务URL（延迟获取，确保在应用上下文中）"""
        if self.cloud_url is None:
            try:
                from flask import current_app
                self.cloud_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
            except RuntimeError:
                # 不在 Flask 应用上下文中，使用默认值
                self.cloud_url = 'http://localhost:5001'
        return self.cloud_url

    def train(self, model_config):
        """训练RUL预测模型 - 使用工况筛选模式
        
        Args:
            model_config (dict): 训练配置，包含:
                - model_type: 模型类型
                - train_files: 训练文件列表
                - test_files: 测试文件列表（可选）
                - conditions: 工况配置列表 [{name: '转速', values: [20, 30]}, ...]
                - file_selections: 文件选择列表
                - sequence_length: 序列长度
                - stride: 窗口步长
                - validation_split: 验证集比例
                - 其他模型参数和训练参数
        
        Returns:
            dict: 训练结果
        """
        try:
            # 生成任务ID
            task_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.logger.info(f"开始RUL预测训练任务: {task_id}")
            
            # 1. 上传数据文件到云端
            upload_result = self._upload_data_to_cloud(model_config, task_id)
            if not upload_result:
                return {
                    'status': 'error',
                    'message': '数据上传到云端失败'
                }
            
            # 2. 构建云端训练配置
            training_config = self._build_training_config(model_config, task_id)
            
            # 3. 发送训练请求到云端
            self.logger.info(f"向云端发送训练请求: {task_id}")
            response = requests.post(
                f'{self._get_cloud_url()}/api/rul_prediction/train',
                json=training_config,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return {
                        'status': 'success',
                        'task_id': result.get('task_id'),
                        'message': '训练任务已提交到云端',
                        'mode': 'cloud'
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f"云端训练失败: {result.get('error', '未知错误')}"
                    }
            else:
                return {
                    'status': 'error',
                    'message': f'云端连接失败: HTTP {response.status_code}'
                }

        except requests.exceptions.ConnectionError:
            return {
                'status': 'error',
                'message': f'无法连接到云端: {self._get_cloud_url()}'
            }
        except Exception as e:
            self.logger.error(f"训练失败: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'训练失败: {str(e)}'
            }
    
    def _upload_data_to_cloud(self, config, task_id):
        """上传训练数据到云端
        
        Args:
            config: 训练配置
            task_id: 任务ID
            
        Returns:
            bool: 上传是否成功
        """
        try:
            train_files = config.get('train_files', [])
            validation_files = config.get('validation_files', [])
            test_files = config.get('test_files', [])
            
            if not train_files:
                self.logger.error("没有指定训练文件")
                return False
            
            edge_root = Path(__file__).resolve().parents[4]
            labeled_dir = edge_root / 'data' / 'labeled' / 'RULPrediction'
            meta_dir = edge_root / 'data' / 'meta' / 'RULPrediction'
            
            # 上传所有训练文件（CSV + meta）
            uploaded_train_files = []
            for filename in train_files:
                if self._upload_single_file_to_cloud(filename, labeled_dir, meta_dir, task_id, 'train'):
                    uploaded_train_files.append(filename)
                else:
                    self.logger.warning(f"训练文件上传失败: {filename}")
            
            if not uploaded_train_files:
                self.logger.error("所有训练文件上传失败")
                return False
            
            # 上传所有验证文件（CSV + meta）
            uploaded_val_files = []
            for filename in validation_files:
                if self._upload_single_file_to_cloud(filename, labeled_dir, meta_dir, task_id, 'val'):
                    uploaded_val_files.append(filename)
                else:
                    self.logger.warning(f"验证文件上传失败: {filename}")
            
            # 上传所有测试文件（如果有）
            uploaded_test_files = []
            for filename in test_files:
                if self._upload_single_file_to_cloud(filename, labeled_dir, meta_dir, task_id, 'test'):
                    uploaded_test_files.append(filename)
                else:
                    self.logger.warning(f"测试文件上传失败: {filename}")
            
            # 更新config中的文件列表
            config['train_files'] = uploaded_train_files
            config['validation_files'] = uploaded_val_files
            config['test_files'] = uploaded_test_files
            
            self.logger.info(f"Edge端文件上传完成 - 训练文件: {uploaded_train_files}, 验证文件: {uploaded_val_files}, 测试文件: {uploaded_test_files}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据上传异常: {e}", exc_info=True)
            return False
    
    def _upload_single_file_to_cloud(self, filename, labeled_dir, meta_dir, task_id, file_type='train'):
        """上传单个文件（CSV + meta）到云端
        
        Args:
            filename: CSV文件名
            labeled_dir: labeled目录
            meta_dir: meta目录
            task_id: 任务ID
            file_type: 文件类型（'train' 或 'test'）
            
        Returns:
            bool: 上传是否成功
        """
        try:
            # 1. 上传CSV文件
            csv_file_path = labeled_dir / filename
            if not csv_file_path.exists():
                self.logger.error(f"CSV文件不存在: {csv_file_path}")
                return False
            
            upload_url = f"{self._get_cloud_url()}/api/rul_prediction/upload_data"
            
            with open(csv_file_path, 'rb') as f:
                files = {'file': (filename, f, 'text/csv')}
                data = {'task_id': task_id, 'file_type': file_type}
                
                response = requests.post(
                    upload_url,
                    files=files,
                    data=data,
                    timeout=120
                )
                
                if response.status_code != 200:
                    self.logger.error(f"CSV文件上传失败: {filename}, HTTP {response.status_code}")
                    return False
            
            # 2. 上传对应的meta文件
            meta_filename = filename.replace('.csv', '.json')
            meta_file_path = meta_dir / meta_filename
            
            if meta_file_path.exists():
                with open(meta_file_path, 'rb') as f:
                    files = {'file': (meta_filename, f, 'application/json')}
                    data = {'task_id': task_id, 'file_type': file_type}
                    
                    response = requests.post(
                        upload_url,
                        files=files,
                        data=data,
                        timeout=120
                    )
                    
                    if response.status_code != 200:
                        self.logger.warning(f"Meta文件上传失败: {meta_filename}, HTTP {response.status_code}")
                        # Meta文件上传失败不影响整体流程
            else:
                self.logger.warning(f"Meta文件不存在: {meta_file_path}")
            
            self.logger.info(f"文件上传成功: {filename} (类型: {file_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"上传文件失败 {filename}: {e}", exc_info=True)
            return False
    
    def _build_training_config(self, config, task_id):
        """构建云端训练配置
        
        Args:
            config: Edge端训练配置
            task_id: 任务ID
            
        Returns:
            dict: 云端训练配置
        """
        # 提取工况key列表（用户选择的工况）
        condition_keys = []
        conditions = config.get('conditions', [])
        for cond in conditions:
            if isinstance(cond, dict) and 'name' in cond:
                condition_keys.append(cond['name'])
        
        def _parse_bool(value, default=True):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ('true', '1', 'yes', 'y', 'on')
            return bool(value)

        def _parse_int_list(value, default=None):
            if value is None:
                return list(default) if default is not None else []
            if isinstance(value, (list, tuple)):
                parsed = []
                for item in value:
                    try:
                        parsed.append(int(item))
                    except (TypeError, ValueError):
                        continue
                if parsed:
                    return parsed
                return list(default) if default is not None else []
            if isinstance(value, str):
                parts = [part.strip() for part in value.split(',') if part.strip()]
                parsed = []
                for part in parts:
                    try:
                        parsed.append(int(part))
                    except ValueError:
                        continue
                if parsed:
                    return parsed
            return list(default) if default is not None else []

        model_type = config.get('model_type') or 'bilstm_gru_regressor'
        device_target = self._normalize_device_target(
            config.get('device') or config.get('device_target') or 'cuda:0'
        )
        config['device'] = device_target
        config['device_target'] = device_target

        training_config = {
            'module': 'rul_prediction',
            'model_type': model_type,
            'task_id': task_id,
            'dataset_mode': 'condition_filtered',
            
            # 文件列表（unit级别）
            'train_files': config.get('train_files', []),
            'validation_files': config.get('validation_files', []),  # 验证集units
            'test_files': config.get('test_files', []),
            
            # 工况配置
            'conditions': conditions,  # 完整的工况配置
            'condition_keys': condition_keys,  # 用户选择的工况key列表
            
            # 数据参数
            'sequence_length': int(config.get('sequence_length', 50)),
            'stride': int(config.get('stride', 1)),
            'validation_split': float(config.get('validation_split', 0.2)),
            'random_seed': int(config.get('random_seed', 42)),
            
            # 训练参数
            'epochs': int(config.get('epochs', 50)),
            'batch_size': int(config.get('batch_size', 32)),
            'learning_rate': float(config.get('learning_rate', 0.001)),
            'weight_decay': float(config.get('weight_decay', 0.0001)),
            'clip_grad_norm': float(config.get('clip_grad_norm', 5.0)),
            'patience': int(config.get('patience', 10)),
            'early_stop_mode': config.get('early_stop_mode', 'loss'),
            'loss_type': config.get('loss_type', 'mse'),
            'device': device_target,
            'device_target': device_target,
        }
        
        # 模型特定参数
        if model_type == 'bilstm_gru_regressor':
            training_config.update({
                'rnn_type': config.get('rnn_type', 'lstm'),
                'hidden_units': int(config.get('hidden_units', 128)),
                'num_layers': int(config.get('num_layers', 2)),
                'dropout': float(config.get('dropout', 0.3)),
                'activation': config.get('activation', 'relu'),
                'bidirectional': _parse_bool(config.get('bidirectional'), True),
                'use_attention': _parse_bool(config.get('use_attention'), True),
                'use_layer_norm': _parse_bool(config.get('use_layer_norm'), True),
            })
        elif model_type == 'cnn_1d_regressor':
            conv_channels = _parse_int_list(
                config.get('conv_channels') or config.get('convChannels'),
                default=[64, 128, 256]
            )
            kernel_sizes = _parse_int_list(
                config.get('kernel_sizes') or config.get('kernelSizes'),
                default=[7, 5, 3]
            )
            pooling = (config.get('pooling') or 'avg').lower()
            if pooling not in ('avg', 'max', 'flatten'):
                pooling = 'avg'

            training_config.update({
                'conv_channels': conv_channels,
                'kernel_sizes': kernel_sizes,
                'activation': config.get('activation', 'relu'),
                'dropout': float(config.get('dropout', 0.3)),
                'pooling': pooling,
                'use_batch_norm': _parse_bool(config.get('use_batch_norm') or config.get('useBatchNorm'), True),
                'fc_units': int(config.get('fc_units', config.get('fcUnits', 256))),
            })
        elif model_type == 'transformer_encoder_regressor':
            embed_dim = int(config.get('embed_dim', config.get('embedDim', 128)))
            num_heads = max(1, int(config.get('num_heads', config.get('numHeads', 4))))
            if embed_dim < num_heads:
                embed_dim = num_heads * 2
            if embed_dim % num_heads != 0:
                embed_dim = ((embed_dim // num_heads) + 1) * num_heads

            pooling = (config.get('pooling') or 'avg').lower()
            if pooling not in ('avg', 'max', 'last'):
                pooling = 'avg'

            training_config.update({
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'num_layers': max(1, int(config.get('num_layers', config.get('numLayers', 3)))) ,
                'ffn_dim': max(embed_dim * 2, int(config.get('ffn_dim', config.get('ffnDim', 256)))),
                'dropout': float(config.get('dropout', 0.1)),
                'activation': config.get('activation', 'gelu'),
                'pooling': pooling,
                'use_positional_encoding': _parse_bool(config.get('use_positional_encoding') or config.get('usePositionalEncoding'), True),
            })
        
        return training_config
    
    def get_training_status(self, task_id):
        """获取训练状态 - 从云端查询"""
        try:
            response = requests.get(
                f'{self._get_cloud_url()}/api/rul_prediction/task/{task_id}/status',
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    task = result.get('task', {}) or {}
                    config = task.get('config', {}) or {}
                    device_target = self._normalize_device_target(
                        config.get('device') or config.get('device_target') or 'CPU'
                    )
                    config['device'] = device_target
                    config['device_target'] = device_target
                    return {
                        'success': True,
                        'task': {
                            'id': task.get('id', task_id),
                            'task_id': task.get('id', task_id),
                            'status': task.get('status', 'unknown'),
                            'model_type': task.get('model_type', config.get('model_type', 'unknown')),
                            'created_at': task.get('created_at', task.get('timestamp')),
                            'start_time': task.get('start_time', task.get('created_at')),
                            'current_epoch': task.get('current_epoch', task.get('epoch', 0)),
                            'epoch': task.get('current_epoch', task.get('epoch', 0)),
                            'total_epochs': task.get('total_epochs', config.get('epochs', 50)),
                            'current_train_loss': task.get('current_train_loss', task.get('loss')),
                            'current_val_loss': task.get('current_val_loss', task.get('val_loss')),
                            'loss': task.get('current_train_loss', task.get('loss')),
                            'train_loss': task.get('current_train_loss', task.get('train_loss')),
                            'val_loss': task.get('current_val_loss', task.get('val_loss')),
                            'train_rmse': task.get('train_rmse', 0),
                            'train_mae': task.get('train_mae', 0),
                            'train_r2': task.get('train_r2', 0),
                            'val_rmse': task.get('val_rmse', 0),
                            'val_mae': task.get('val_mae', 0),
                            'val_r2': task.get('val_r2', 0),
                            'rmse': task.get('rmse', task.get('val_rmse', 0)),
                            'message': task.get('message', ''),
                            'logs': task.get('logs', []),
                            'evaluation_results': task.get('evaluation_results'),
                            'evaluation': task.get('evaluation_results'),
                            'config': config,
                            'device': device_target,
                            'device_target': device_target,
                        }
                    }
                else:
                    return {
                        'success': False,
                        'message': result.get('error', '任务不存在')
                    }
            else:
                return {
                    'success': False,
                    'message': f'云端连接失败: HTTP {response.status_code}'
                }

        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'message': f'无法连接到云端: {self._get_cloud_url()}'
            }
        except Exception as e:
            self.logger.error(f"获取训练状态失败: {e}", exc_info=True)
            return {
                'success': False,
                'message': f'获取训练状态失败: {str(e)}'
            }
