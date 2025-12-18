"""
故障诊断训练服务
支持云端训练 - 处理文件选择和数据传输
"""

import requests
import uuid
import shutil
from pathlib import Path
from flask import current_app
import logging
from datetime import datetime


class FaultDiagnosisTrainer:
    """故障诊断训练服务 - 仅支持云端训练"""

    def __init__(self):
        """初始化训练服务"""
        # 延迟获取配置，避免在应用上下文外访问 current_app
        self.cloud_url = None
        self.edge_host = None
        self.edge_port = None
        self.logger = logging.getLogger(__name__)

    def _normalize_device_target(self, value):
        """标准化设备标识"""
        normalized = str(value or 'CPU').strip().lower()
        if normalized in ('gpu', 'cuda'):
            return 'GPU'
        if normalized in ('ascend', 'npu', 'atlas'):
            return 'Ascend'
        return 'CPU'
    
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
    
    def _get_edge_host(self):
        """获取边缘端主机地址"""
        if self.edge_host is None:
            try:
                from flask import current_app
                self.edge_host = current_app.config.get('EDGE_HOST', 'localhost')
            except RuntimeError:
                self.edge_host = 'localhost'
        return self.edge_host
    
    def _get_edge_port(self):
        """获取边缘端端口"""
        if self.edge_port is None:
            try:
                from flask import current_app
                self.edge_port = current_app.config.get('EDGE_PORT', 5000)
            except RuntimeError:
                self.edge_port = 5000
        return self.edge_port

    def train(self, model_config):
        """训练故障诊断模型
        
        Args:
            model_config (dict): 训练配置，包含:
                - model_type: 'cnn_1d_classifier' 或 'lstm_classifier'
                - num_classes: 分类数量（2或3）
                - num_conditions: 工况数量
                - file_selections: 文件选择字典，格式为 {condition_id: {label_index: filename}}
                - sequence_length: 序列长度
                - stride: 窗口步长
                - train_ratio: 训练集比例
                - val_ratio: 验证集比例
                - test_ratio: 测试集比例
                - 其他模型参数和训练参数
        
        Returns:
            dict: 训练结果
        """
        try:
            # 生成任务ID
            task_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.logger.info(f"开始故障诊断训练任务: {task_id}")
            
            # 1. 处理文件选择，复制文件到训练目录
            file_result = self._prepare_training_files(model_config, task_id)
            if file_result.get('status') != 'success':
                return file_result
            
            # 2. 构建云端训练配置
            training_config = self._build_training_config(model_config, task_id)
            
            # 3. 发送训练请求到云端
            self.logger.info(f"向云端发送训练请求: {task_id}")
            response = requests.post(
                f'{self._get_cloud_url()}/api/fault_diagnosis/train',
                json=training_config,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return {
                        'status': 'success',
                        'task_id': task_id,
                        'cloud_task_id': result.get('task_id'),
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
    
    def _prepare_training_files(self, config, task_id):
        """准备训练文件：从processed/fd复制到training/fd/<任务id>/
        
        Args:
            config: 训练配置
            task_id: 任务ID
            
        Returns:
            dict: 处理结果
        """
        try:
            # 获取edge目录路径
            edge_root = Path(__file__).resolve().parents[4]  # 从 trainer.py 到 edge 目录
            # 故障诊断数据从标注目录读取：edge/data/labeled/FaultDiagnosis
            processed_dir = edge_root / 'data' / 'labeled' / 'FaultDiagnosis'
            training_dir = edge_root / 'data' / 'training' / 'FaultDiagnosis' / task_id
            
            # 创建训练目录
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # 收集所有需要复制的文件
            file_selections = config.get('file_selections', [])
            copied_files = []
            seen_files = set()  # 用于去重，避免重复复制同一个文件
            
            # 遍历文件选择配置（新格式：列表）
            if isinstance(file_selections, list):
                # 新格式：列表，每个元素包含 filename, condition_combo, label_index 等
                for file_selection in file_selections:
                    filename = file_selection.get('filename') if isinstance(file_selection, dict) else None
                    
                    if not filename:
                        continue
                    
                    # 如果文件已经复制过，跳过（避免重复）
                    if filename in seen_files:
                        continue
                    
                    source_file = processed_dir / filename
                    if not source_file.exists():
                        return {
                            'status': 'error',
                            'message': f'文件不存在: {filename}'
                        }
                    
                    # 复制文件到训练目录
                    dest_file = training_dir / filename
                    shutil.copy2(source_file, dest_file)
                    copied_files.append(filename)
                    seen_files.add(filename)
                    self.logger.info(f"已复制文件: {filename} -> {dest_file}")
            elif isinstance(file_selections, dict):
                # 兼容旧格式：字典 {condition_id: {label_index: filename}}
                for condition_id, label_files in file_selections.items():
                    for label_index, filename in label_files.items():
                        if not filename:
                            continue
                        
                        # 如果文件已经复制过，跳过
                        if filename in seen_files:
                            continue
                        
                        source_file = processed_dir / filename
                        if not source_file.exists():
                            return {
                                'status': 'error',
                                'message': f'文件不存在: {filename}'
                            }
                        
                        # 复制文件到训练目录
                        dest_file = training_dir / filename
                        shutil.copy2(source_file, dest_file)
                        copied_files.append(filename)
                        seen_files.add(filename)
                        self.logger.info(f"已复制文件: {filename} -> {dest_file}")
            else:
                return {
                    'status': 'error',
                    'message': f'file_selections 格式错误，应为列表或字典，当前类型: {type(file_selections)}'
                }
            
            if not copied_files:
                return {
                    'status': 'error',
                    'message': '没有选择任何数据文件'
                }
            
            self.logger.info(f"已准备 {len(copied_files)} 个训练文件到 {training_dir}")
            
            return {
                'status': 'success',
                'task_id': task_id,
                'training_dir': str(training_dir),
                'files': copied_files,
                'file_count': len(copied_files)
            }
            
        except Exception as e:
            self.logger.error(f"准备训练文件失败: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'准备训练文件失败: {str(e)}'
            }
    
    def _build_training_config(self, config, task_id):
        """构建云端训练配置
        
        Args:
            config: Edge端训练配置
            task_id: 任务ID
            
        Returns:
            dict: 云端训练配置
        """
        # 获取模型类型
        model_type = config.get('model_type', 'cnn_1d_classifier')
        device_target = self._normalize_device_target(
            config.get('device_target') or config.get('device') or 'CPU'
        )
        config['device_target'] = device_target
        
        # 基础配置
        training_config = {
            'module': 'fault_diagnosis',
            'model_type': model_type,
            'task_id': task_id,
            'edge_host': self._get_edge_host(),
            'edge_port': self._get_edge_port(),
            'training_dir': f'edge/data/training/FaultDiagnosis/{task_id}',
            
            # 数据集配置
            'labels': config.get('labels', ['正常', '内圈故障', '外圈故障']),  # 标签列表
            'conditions': config.get('conditions', []),  # 工况配置列表 [{name: '转速', values: [300, 600, 900]}, ...]
            'num_classes': len(config.get('labels', ['正常', '内圈故障', '外圈故障'])),  # 从标签列表计算
            'num_conditions': len(config.get('conditions', [])),  # 从工况配置列表计算
            'file_selections': config.get('file_selections', []),  # 新格式：列表，每个元素包含filename, condition_combo等
            'sequence_length': int(config.get('sequence_length', 50)),
            'stride': int(config.get('stride', 1)),
            'train_ratio': float(config.get('train_ratio', 0.7)),
            'val_ratio': float(config.get('val_ratio', 0.15)),
            'test_ratio': float(config.get('test_ratio', 0.15)),
            
            # 训练参数
            'epochs': int(config.get('epochs', 50)),
            'batch_size': int(config.get('batch_size', 32)),
            'learning_rate': float(config.get('learning_rate', 0.001)),
            'weight_decay': float(config.get('weight_decay', 0.0001)),
            'clip_grad_norm': float(config.get('clip_grad_norm', 5.0)),
            'patience': int(config.get('patience', 10)),
            'random_seed': int(config.get('random_seed', 42)),
            'device_target': device_target,
        }
        
        # 模型特定参数
        if model_type == 'cnn_1d_classifier':
            training_config.update({
                'num_filters': int(config.get('num_filters', 64)),
                'kernel_size': int(config.get('kernel_size', 3)),
                'num_conv_layers': int(config.get('num_conv_layers', 3)),
                'dropout': float(config.get('dropout', 0.3)),
                'activation': config.get('activation', 'relu'),
                'use_batch_norm': config.get('use_batch_norm', 'true').lower() == 'true',
            })
        elif model_type == 'lstm_classifier':
            training_config.update({
                'hidden_units': int(config.get('hidden_units', 128)),
                'num_layers': int(config.get('num_layers', 2)),
                'dropout': float(config.get('dropout', 0.3)),
                'activation': config.get('activation', 'tanh'),
                'bidirectional': config.get('bidirectional', 'false').lower() == 'true',
                'use_attention': config.get('use_attention', 'false').lower() == 'true',
            })
        
        return training_config

    def get_training_status(self, task_id):
        """获取训练状态"""
        try:
            response = requests.get(
                f'{self._get_cloud_url()}/api/fault_diagnosis/training_status/{task_id}',
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result.get('task')
                else:
                    self.logger.warning(f"获取训练状态失败: {result.get('message', '未知错误')}")
                    return None
            else:
                self.logger.error(f"获取训练状态HTTP错误: {response.status_code}")
                return None

        except requests.exceptions.ConnectionError:
            self.logger.error(f"无法连接到云端获取训练状态: {self._get_cloud_url()}")
            return None
        except Exception as e:
            self.logger.error(f"获取训练状态失败: {str(e)}", exc_info=True)
            return None

    def pause_training(self, task_id):
        """暂停训练"""
        try:
            response = requests.post(
                f'{self._get_cloud_url()}/api/fault_diagnosis/pause_training/{task_id}',
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('success', False)
            else:
                self.logger.error(f"暂停训练HTTP错误: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"暂停训练失败: {str(e)}", exc_info=True)
            return False

    def stop_training(self, task_id):
        """停止训练"""
        try:
            response = requests.post(
                f'{self._get_cloud_url()}/api/fault_diagnosis/stop_training/{task_id}',
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('success', False)
            else:
                self.logger.error(f"停止训练HTTP错误: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"停止训练失败: {str(e)}", exc_info=True)
            return False

    def get_model_path(self, task_id):
        """获取模型文件路径"""
        try:
            # 首先获取训练状态，检查是否完成
            status = self.get_training_status(task_id)
            if not status or status.get('status') != 'completed':
                return None

            # 模型通常保存在云端的models目录中
            # 这里返回云端的下载URL，路由中会处理实际下载
            return f'{self._get_cloud_url()}/api/fault_diagnosis/download_model/{task_id}'

        except Exception as e:
            self.logger.error(f"获取模型路径失败: {str(e)}", exc_info=True)
            return None
