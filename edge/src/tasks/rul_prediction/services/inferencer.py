"""
RUL预测推理服务
"""

import logging
import numpy as np
import pandas as pd
import mindspore as ms
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import pickle
import json
from datetime import datetime

# 从本地 core 模块导入模型类
from ..core.BiLSTMGRU.model_builder import ModelBuilder as BiLSTMModelBuilder
from ..core.cnn_1d_regressor.model_builder import ModelBuilder as CNN1DModelBuilder
from ..core.transformer_regressor.model_builder import ModelBuilder as TransformerModelBuilder


MODEL_BUILDERS = {
    'bilstm_gru_regressor': BiLSTMModelBuilder,
    'cnn_1d_regressor': CNN1DModelBuilder,
    'transformer_encoder_regressor': TransformerModelBuilder,
}

MODEL_PARAM_KEYS = {
    'bilstm_gru_regressor': {
        'hidden_units', 'num_layers', 'dropout', 'activation', 'bidirectional',
        'use_attention', 'use_layer_norm', 'rnn_type'
    },
    'cnn_1d_regressor': {
        'conv_channels', 'kernel_sizes', 'activation', 'dropout', 'pooling',
        'use_batch_norm', 'fc_units'
    },
    'transformer_encoder_regressor': {
        'embed_dim', 'num_heads', 'num_layers', 'ffn_dim', 'dropout',
        'activation', 'pooling', 'use_positional_encoding'
    }
}


class RULPredictionInferencer:
    """
    本地RUL预测推理服务
    
    负责：
    - 加载训练好的模型和scaler
    - 预处理输入数据
    - 执行批量RUL预测
    - 返回预测结果
    """

    def __init__(self, model_path: Union[str, Path],
                 scaler_path: Optional[Union[str, Path]] = None,
                 label_scaler_path: Optional[Union[str, Path]] = None,
                 config_path: Union[str, Path] = None,
                 sequence_length: int = 50):
        """
        初始化RUL预测推理服务
        
        Args:
            model_path: 模型权重文件路径
            scaler_path: 特征标准化器文件路径（可选，如果数据已归一化可能不存在）
            label_scaler_path: 标签归一化器文件路径（必需，用于反归一化预测结果）
            config_path: 模型配置文件路径
            sequence_length: 序列长度
        """
        # 设置MindSpore上下文
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_device('CPU')

        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.label_scaler = None
        self.config = None

        # 日志记录器
        self.logger = logging.getLogger(__name__)

        # 加载组件（注意顺序：必须先加载config，再加载model）
        # 1. 加载配置（必需，因为模型加载需要配置）
        if config_path:
            self._load_config(config_path)
        else:
            # 如果没有提供config_path，尝试从模型目录自动查找
            model_dir = Path(model_path).parent
            potential_config = model_dir / 'model_config.json'
            if potential_config.exists():
                self._load_config(potential_config)
            else:
                raise FileNotFoundError(f"配置文件不存在，且无法在模型目录中找到: {model_dir}")
        
        # 2. 加载特征scaler（可选）
        if scaler_path:
            self._load_scaler(scaler_path)
        else:
            # 尝试从模型目录自动查找scaler
            model_dir = Path(model_path).parent
            potential_scaler = model_dir / 'scaler.pkl'
            if potential_scaler.exists():
                self._load_scaler(potential_scaler)
        
        # 3. 加载标签scaler（必需，用于反归一化）
        if label_scaler_path:
            self._load_label_scaler(label_scaler_path)
        else:
            # 尝试从模型目录自动查找label_scaler
            model_dir = Path(model_path).parent
            potential_label_scaler = model_dir / 'label_scaler.pkl'
            if potential_label_scaler.exists():
                self._load_label_scaler(potential_label_scaler)
            else:
                raise FileNotFoundError(f"标签归一化器文件不存在，且无法在模型目录中找到: {model_dir}")
        
        # 4. 最后加载模型（需要config）
        self._load_model(model_path)

        print(f"✅ RUL预测推理服务初始化完成")
        print(f"  - 模型: {model_path}")
        if config_path:
            print(f"  - 配置文件: {config_path}")
        if scaler_path:
            print(f"  - 特征标准化器: {scaler_path}")
        if self.label_scaler:
            print(f"  - 标签归一化器: 已加载")
        print(f"  - 序列长度: {sequence_length}")

    def _load_config(self, config_path: Union[str, Path]):
        """加载模型配置"""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 更新序列长度（如果配置中有）
            if 'sequence_length' in self.config:
                self.sequence_length = self.config['sequence_length']
            
            print(f"📂 配置文件加载成功: {config_path}")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise

    def _load_model(self, model_path: Union[str, Path]):
        """加载模型"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            model_type = self.config.get('model_type', 'bilstm_gru_regressor')
            input_dim = self.config.get('input_dim', 1)
            input_shape = (self.sequence_length, input_dim)

            builder = MODEL_BUILDERS.get(model_type)
            if builder is None:
                raise ValueError(f"不支持的模型类型: {model_type}")

            allowed_keys = MODEL_PARAM_KEYS.get(model_type, set())
            builder_kwargs = {
                key: value for key, value in self.config.items()
                if key in allowed_keys
            }

            self.model = builder.create_model(
                model_type=model_type,
                input_shape=input_shape,
                **builder_kwargs
            )
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise

    def _load_scaler(self, scaler_path: Union[str, Path]):
        """加载特征标准化器"""
        try:
            scaler_path = Path(scaler_path)
            if not scaler_path.exists():
                self.logger.warning(f"特征标准化器文件不存在: {scaler_path}，数据可能已归一化")
                return
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print(f"📂 特征标准化器加载成功: {scaler_path}")
        except Exception as e:
            self.logger.error(f"加载特征标准化器失败: {e}")
            raise
    
    def _load_label_scaler(self, label_scaler_path: Union[str, Path]):
        """加载标签归一化器"""
        try:
            label_scaler_path = Path(label_scaler_path)
            if not label_scaler_path.exists():
                raise FileNotFoundError(f"标签归一化器文件不存在: {label_scaler_path}")
            
            with open(label_scaler_path, 'rb') as f:
                self.label_scaler = pickle.load(f)
            
            print(f"📂 标签归一化器加载成功: {label_scaler_path}")
        except Exception as e:
            self.logger.error(f"加载标签归一化器失败: {e}")
            raise

    def _create_sequences(self, data: np.ndarray, sequence_length: int, stride: int = 1) -> np.ndarray:
        """
        创建滑动窗口序列
        
        Args:
            data: 数据数组 (N, features)
            sequence_length: 序列长度
            stride: 步长
            
        Returns:
            sequences: (n_samples, sequence_length, features)
        """
        sequences = []
        
        for i in range(0, len(data) - sequence_length + 1, stride):
            sequences.append(data[i:i + sequence_length])
        
        if len(sequences) == 0:
            return np.array([]).reshape(0, sequence_length, data.shape[1])
        
        return np.array(sequences)

    def predict(self, data: Union[np.ndarray, pd.DataFrame], 
                batch_size: int = 32) -> np.ndarray:
        """
        预测RUL值
        
        Args:
            data: 输入数据 (N, features) 或 DataFrame
            batch_size: 批次大小
            
        Returns:
            rul_predictions: RUL预测值 (N,)
        """
        try:
            # 转换为numpy数组
            if isinstance(data, pd.DataFrame):
                data = data.values
            
            if len(data) < self.sequence_length:
                raise ValueError(f"数据长度 ({len(data)}) 小于序列长度 ({self.sequence_length})")
            
            # 创建序列
            sequences = self._create_sequences(data, self.sequence_length, stride=1)
            
            if len(sequences) == 0:
                return np.array([])
            
            # 标准化（如果scaler存在）
            if self.scaler is not None:
                sequences_2d = sequences.reshape(-1, sequences.shape[2])
                sequences_2d_scaled = self.scaler.transform(sequences_2d)
                sequences_scaled = sequences_2d_scaled.reshape(sequences.shape)
            else:
                # 数据已经归一化，直接使用
                sequences_scaled = sequences
            
            # 批量预测
            all_predictions = []
            
            for i in range(0, len(sequences_scaled), batch_size):
                batch = sequences_scaled[i:i + batch_size]
                batch_tensor = ms.Tensor(batch.astype(np.float32))
                
                # 预测
                with ms._no_grad():
                    predictions = self.model(batch_tensor)
                    predictions_np = predictions.asnumpy()
                
                all_predictions.append(predictions_np.flatten())
            
            # 合并所有预测结果（归一化后的预测值）
            rul_predictions_normalized = np.concatenate(all_predictions)
            
            # 反归一化：将归一化的预测值转换回原始RUL尺度
            if self.label_scaler is not None:
                rul_predictions_normalized_reshaped = rul_predictions_normalized.reshape(-1, 1)
                rul_predictions = self.label_scaler.inverse_transform(rul_predictions_normalized_reshaped).flatten()
            else:
                # 如果没有标签scaler，直接使用预测值（可能是旧模型）
                self.logger.warning("未找到标签归一化器，预测值可能未反归一化")
                rul_predictions = rul_predictions_normalized
            
            # 对于每个原始数据点，使用最后一个包含它的序列的预测值
            # 由于我们使用stride=1，最后一个序列的预测值对应最后一个数据点
            # 对于前面的点，我们需要扩展预测结果
            if len(rul_predictions) < len(data):
                # 扩展：前面的点使用第一个预测值，后面的点使用对应的预测值
                extended_predictions = np.zeros(len(data))
                extended_predictions[:self.sequence_length - 1] = rul_predictions[0]
                extended_predictions[self.sequence_length - 1:] = rul_predictions
                rul_predictions = extended_predictions
            
            return rul_predictions
            
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            raise

    def predict_rul(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测RUL（接口方法）
        
        Args:
            data_config: 数据配置，包含:
                - data_file: 数据文件路径
                - batch_size: 批次大小（可选）
                
        Returns:
            预测结果字典
        """
        try:
            data_file = data_config.get('data_file')
            batch_size = data_config.get('batch_size', 32)
            
            if not data_file:
                raise ValueError("缺少data_file参数")
            
            # 读取数据
            data_path = Path(data_file)
            if not data_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
            df = pd.read_csv(data_path)
            
            # 准备特征数据：工况也会作为特征的一部分（与训练阶段保持一致）
            condition_keys = self.config.get('condition_keys', [])
            
            # 排除时间戳列
            timestamp_col = None
            for col in df.columns:
                if 'time' in col.lower() or 'timestamp' in col.lower():
                    timestamp_col = col
                    break
            
            # 如果配置中有工况，需要补全工况列
            if condition_keys:
                # 从数据配置中获取工况值，或使用默认值
                condition_values = data_config.get('condition_values', {})
                for key in condition_keys:
                    if key not in df.columns:
                        if key in condition_values:
                            df[key] = condition_values[key]
                        else:
                            # 使用默认值（取配置中第一个值）
                            conditions = self.config.get('conditions', [])
                            for cond in conditions:
                                if isinstance(cond, dict) and cond.get('name') == key:
                                    values = cond.get('values', [])
                                    if values:
                                        df[key] = values[0]
                                    else:
                                        df[key] = 0
                                    break
                            else:
                                df[key] = 0
            
            # 分离传感器特征和工况特征（与训练阶段保持一致）
            exclude_cols = set(condition_keys)
            if timestamp_col:
                exclude_cols.add(timestamp_col)
            sensor_feature_cols = [col for col in df.columns if col not in exclude_cols]
            condition_cols = condition_keys
            
            # 提取传感器特征和工况特征
            sensor_features = df[sensor_feature_cols].values.astype(np.float32)
            condition_features = df[condition_cols].values.astype(np.float32) if condition_cols else None
            
            # 合并传感器特征和工况特征（工况作为特征的一部分）
            # 最终特征 = [传感器特征, 工况特征]（与训练阶段保持一致）
            if condition_features is not None:
                data = np.hstack([sensor_features, condition_features])
            else:
                data = sensor_features
            
            # 预测
            rul_predictions = self.predict(data, batch_size=batch_size)
            
            # 构建结果
            result = {
                'success': True,
                'predictions': rul_predictions.tolist(),
                'num_samples': len(rul_predictions),
                'data_file': str(data_file),
                'model_config': {
                    'model_type': self.config.get('model_type'),
                    'sequence_length': self.sequence_length,
                    'input_dim': self.config.get('input_dim'),
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"RUL预测失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
