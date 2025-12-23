"""
故障诊断推理服务
提供故障诊断模型的推理功能

重构说明：使用 PyTorch 实现
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class FaultDiagnosisInferencer:
    """
    故障诊断推理服务
    
    负责加载训练好的模型并进行故障诊断推理
    """
    
    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        """
        初始化推理服务
        
        Args:
            model_dir: 模型目录路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.model = None
        self.model_config = None
        self.model_dir = Path(model_dir) if model_dir else None
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"推理服务初始化，设备: {self.device}")
        
        # 如果提供了模型目录，自动加载模型
        if self.model_dir:
            self.load_model(self.model_dir)
    
    def load_model(self, model_dir: Union[str, Path]) -> bool:
        """
        加载模型
        
        Args:
            model_dir: 模型目录路径
            
        Returns:
            是否加载成功
        """
        try:
            model_dir = Path(model_dir)
            
            # 加载模型配置
            config_file = model_dir / 'model_config.json'
            if not config_file.exists():
                raise FileNotFoundError(f'模型配置文件不存在: {config_file}')
            
            with open(config_file, 'r', encoding='utf-8') as f:
                self.model_config = json.load(f)
            
            # 加载模型文件 - 优先使用 .pth，向后兼容 .ckpt
            model_path = model_dir / 'model.pth'
            if not model_path.exists():
                model_path = model_dir / 'model.ckpt'
                if not model_path.exists():
                    raise FileNotFoundError(f'模型文件不存在: {model_dir}/model.pth 或 model.ckpt')
            
            # 获取模型参数
            model_type = self.model_config.get('model_type', 'cnn_1d_classifier')
            num_classes = self.model_config.get('num_classes', 3)
            sequence_length = self.model_config.get('sequence_length', 100)
            n_features = self.model_config.get('n_features', 5)
            input_shape = (sequence_length, n_features)
            
            # 根据模型类型创建网络
            self.model = self._create_model(model_type, input_shape, num_classes)
            
            # 加载模型参数
            state_dict = torch.load(str(model_path), map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_dir = model_dir
            logger.info(f"模型加载成功: {model_dir}, 类型: {model_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None
            self.model_config = None
            return False
    
    def _create_model(self, model_type: str, input_shape: tuple, num_classes: int):
        """
        根据模型类型创建网络
        
        Args:
            model_type: 模型类型
            input_shape: 输入形状 (seq_len, n_features)
            num_classes: 分类数量
            
        Returns:
            模型实例
        """
        config = self.model_config or {}
        
        if 'resnet' in model_type.lower():
            from ..core.resnet_1d.model_builder import ModelBuilder
            
            return ModelBuilder.build_resnet_1d_classifier(
                input_shape=input_shape,
                num_classes=num_classes,
                base_channels=config.get('base_channels', 64),
                block_config=config.get('block_config', 'resnet_small'),
                kernel_size=config.get('kernel_size', 3),
                dropout=config.get('dropout', 0.3)
            )
        elif 'cnn' in model_type.lower():
            from ..core.cnn_1d.model_builder import ModelBuilder
            
            return ModelBuilder.build_cnn_1d_classifier(
                input_shape=input_shape,
                num_classes=num_classes,
                num_filters=config.get('num_filters', 64),
                kernel_size=config.get('kernel_size', 3),
                num_conv_layers=config.get('num_conv_layers', 3),
                dropout=config.get('dropout', 0.3)
            )
        else:
            from ..core.lstm.model_builder import ModelBuilder
            
            return ModelBuilder.build_lstm_classifier(
                input_shape=input_shape,
                num_classes=num_classes,
                hidden_units=config.get('hidden_units', 128),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.3),
                bidirectional=config.get('bidirectional', False),
                use_attention=config.get('use_attention', False)
            )
    
    def predict(
        self,
        sequences: np.ndarray,
        batch_size: int = 32,
        return_probs: bool = False,
    ) -> Dict[str, Any]:
        """
        对输入序列进行预测
        
        Args:
            sequences: 输入序列 (n_samples, seq_len, n_features)
            batch_size: 批次大小
            return_probs: 是否返回完整概率分布
            
        Returns:
            预测结果字典
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        all_predictions = []
        all_confidences = []
        all_probs = [] if return_probs else None
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                batch_tensor = torch.from_numpy(batch.astype(np.float32)).to(self.device)
                
                # 前向传播
                output = self.model(batch_tensor)
                probs = F.softmax(output, dim=1)
                
                # 获取预测结果
                pred = probs.argmax(dim=1).cpu().numpy()
                conf = probs.max(dim=1).values.cpu().numpy()
                
                all_predictions.extend(pred)
                all_confidences.extend(conf)
                
                if return_probs:
                    all_probs.extend(probs.cpu().numpy())
        
        result = {
            'predictions': np.array(all_predictions),
            'confidences': np.array(all_confidences),
        }
        
        if return_probs:
            result['probabilities'] = np.array(all_probs)
        
        return result
    
    def diagnose_fault(
        self,
        data_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        诊断故障（主接口）
        
        Args:
            data_config: 数据配置，包含:
                - data_path: 数据文件路径
                - model_id: 模型ID
                - batch_size: 批次大小（可选）
                - condition_values: 工况值（可选）
                
        Returns:
            诊断结果字典
        """
        try:
            start_time = datetime.now()
            
            # 获取配置参数
            data_path = data_config.get('data_path')
            model_id = data_config.get('model_id')
            batch_size = data_config.get('batch_size', 32)
            condition_values = data_config.get('condition_values', {})
            
            if not data_path:
                raise ValueError("缺少数据文件路径 (data_path)")
            if not model_id and self.model is None:
                raise ValueError("缺少模型ID (model_id) 且模型未预加载")
            
            # 如果指定了新的模型，加载它
            if model_id:
                model_dir = self._find_model_dir(model_id)
                if model_dir and (self.model_dir is None or model_dir != self.model_dir):
                    self.load_model(model_dir)
            
            if self.model is None:
                raise RuntimeError("模型加载失败")
            
            # 加载数据
            sequences = self._load_data(data_path, condition_values)
            
            if len(sequences) == 0:
                raise ValueError("数据序列为空，无法进行推理")
            
            # 执行预测
            result = self.predict(sequences, batch_size, return_probs=True)
            
            # 获取标签名称
            labels = self.model_config.get('labels', [])
            
            # 统计结果
            predictions = result['predictions']
            confidences = result['confidences']
            
            unique, counts = np.unique(predictions, return_counts=True)
            class_distribution = {}
            for idx, count in zip(unique, counts):
                label_name = labels[idx] if idx < len(labels) else f'类别{idx}'
                class_distribution[label_name] = int(count)
            
            # 找出主要预测类别
            main_class_idx = np.argmax(counts)
            main_class = unique[main_class_idx]
            predicted_label = labels[main_class] if main_class < len(labels) else f'类别{main_class}'
            
            # 计算置信度
            main_confidence = float(np.mean([
                confidences[i] for i in range(len(predictions)) 
                if predictions[i] == main_class
            ]))
            avg_confidence = float(np.mean(confidences))
            
            inference_time = (datetime.now() - start_time).total_seconds()
            
            diagnosis_result = {
                'success': True,
                'predicted_label': predicted_label,
                'confidence': main_confidence,
                'avg_confidence': avg_confidence,
                'total_samples': len(sequences),
                'class_distribution': class_distribution,
                'predictions': predictions.tolist(),
                'confidences': confidences.tolist(),
                'inference_time': inference_time,
                'model_config': {
                    'model_type': self.model_config.get('model_type'),
                    'sequence_length': self.model_config.get('sequence_length'),
                    'n_features': self.model_config.get('n_features'),
                    'labels': labels,
                },
                'created_at': datetime.now().isoformat(),
            }
            
            logger.info(f"诊断完成: {predicted_label}, 置信度: {main_confidence:.4f}")
            
            return diagnosis_result
            
        except Exception as e:
            logger.error(f"故障诊断失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'created_at': datetime.now().isoformat(),
            }
    
    def _find_model_dir(self, model_id: str) -> Optional[Path]:
        """
        查找模型目录
        
        Args:
            model_id: 模型ID
            
        Returns:
            模型目录路径
        """
        # 获取边缘端模型目录
        edge_root = Path(__file__).resolve().parents[4]
        models_base = edge_root / 'models' / 'fault_diagnosis'
        
        # 在各个模型类型子目录中查找
        for model_type_dir in ['cnn_1d', 'lstm', 'resnet_1d']:
            potential_dir = models_base / model_type_dir / model_id
            if potential_dir.exists():
                return potential_dir
        
        # 直接查找（向后兼容）
        direct_dir = models_base / model_id
        if direct_dir.exists():
            return direct_dir
        
        return None
    
    def _load_data(
        self,
        data_path: Union[str, Path],
        condition_values: Dict[str, Any] = None,
    ) -> np.ndarray:
        """
        加载并预处理数据
        
        Args:
            data_path: 数据文件路径
            condition_values: 工况值
            
        Returns:
            序列数据 (n_samples, seq_len, n_features)
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 加载CSV数据
        df = pd.read_csv(data_path)
        
        # 检测并排除非特征列
        exclude_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['time', 'timestamp', 'date', 'label', 'class', 'target']):
                exclude_cols.append(col)
        
        # 获取特征列
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 如果第一列是索引或时间戳类型，也排除
        if len(feature_cols) > 0:
            first_col = feature_cols[0]
            if df[first_col].dtype == 'object' or 'unnamed' in first_col.lower():
                feature_cols = feature_cols[1:]
        
        # 获取模型配置中的特征数
        model_conditions = self.model_config.get('conditions', [])
        n_features = self.model_config.get('n_features', 5)
        num_conditions = len(model_conditions)
        original_n_features = n_features - num_conditions
        
        # 提取特征数据
        if len(feature_cols) < original_n_features:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:original_n_features]
        
        features = df[feature_cols[:original_n_features]].values.astype(np.float32)
        
        # 添加工况列（如果有）
        if condition_values and model_conditions:
            for cond in model_conditions:
                cond_name = cond.get('name')
                cond_value = condition_values.get(cond_name, 0)
                cond_col = np.full((len(features), 1), cond_value, dtype=np.float32)
                features = np.hstack([features, cond_col])
        
        # 创建序列
        sequence_length = self.model_config.get('sequence_length', 100)
        sequences = self._create_sequences(features, sequence_length)
        
        return sequences
    
    def _create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int,
        stride: int = 1,
    ) -> np.ndarray:
        """
        创建时间序列窗口
        
        Args:
            data: 原始数据 (n_samples, n_features)
            sequence_length: 序列长度
            stride: 步长
            
        Returns:
            序列数据 (n_sequences, seq_len, n_features)
        """
        if len(data) < sequence_length:
            return np.array([])
        
        sequences = []
        for i in range(0, len(data) - sequence_length + 1, stride):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences, dtype=np.float32)
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        获取当前加载的模型信息
        
        Returns:
            模型信息字典
        """
        if self.model is None or self.model_config is None:
            return None
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_dir': str(self.model_dir) if self.model_dir else None,
            'model_type': self.model_config.get('model_type'),
            'num_classes': self.model_config.get('num_classes'),
            'labels': self.model_config.get('labels', []),
            'sequence_length': self.model_config.get('sequence_length'),
            'n_features': self.model_config.get('n_features'),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'device': str(self.device),
        }
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None
