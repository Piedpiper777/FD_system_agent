"""
异常检测推理服务实现
模型类从本地 core 模块导入（从 Cloud 端复制）
"""

import logging
import numpy as np
import pandas as pd
import mindspore as ms
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
import pickle
import json
from datetime import datetime

# 从本地 core 模块导入模型类（从 Cloud 端复制）
from ..core.lstm_predictor.model_builder import LSTMPredictor
from ..core.lstm_autoencoder.model_builder import LSTMAutoencoder
from ..core.cnn_1d_autoencoder.model_builder import CNN1DAutoencoder


class LocalAnomalyDetector:
    """
    本地异常检测器

    负责：
    - 加载训练好的模型和阈值
    - 预处理输入数据
    - 执行批量异常检测
    - 返回检测结果
    """

    def __init__(self, model_path: Union[str, Path],
                 threshold_path: Union[str, Path],
                 scaler_path: Union[str, Path],
                 sequence_length: int = 50,
                 model_type: str = 'lstm_predictor'):
        """
        初始化本地异常检测器

        Args:
            model_path: 模型权重文件路径
            threshold_path: 阈值文件路径
            scaler_path: 标准化器文件路径
            sequence_length: 序列长度
            model_type: 模型类型 ('lstm_predictor', 'lstm_autoencoder', 'cnn_1d_autoencoder')
        """
        # 设置MindSpore上下文
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_device('CPU')

        self.sequence_length = sequence_length
        self.model_type = model_type
        self.model = None
        self.threshold_value = None
        self.scaler = None
        self._scaler_feature_names = None
        self._scaler_feature_count = None

        # 日志记录器
        self.logger = logging.getLogger(__name__)

        # 加载组件
        self._load_model(model_path)
        self._load_threshold(threshold_path)
        self._load_scaler(scaler_path)

        print(f"✅ 本地异常检测器初始化完成")
        print(f"  - 模型: {model_path}")
        print(f"  - 模型类型: {model_type}")
        print(f"  - 阈值: {threshold_path}")
        print(f"  - 标准化器: {scaler_path}")
        print(f"  - 序列长度: {sequence_length}")

    def _load_model(self, model_path: Union[str, Path]):
        """加载模型"""
        try:
            model_path = Path(model_path)

            # 从配置文件中获取模型参数
            config_path = model_path.parent / 'config.json'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 获取模型类型（优先使用传入的model_type，否则从config读取）
                model_type = self.model_type or config.get('model_type', 'lstm_predictor')
                self.model_type = model_type

                # 重建模型架构（优先使用config中的sequence_length，确保与训练时一致）
                sequence_length = config.get('sequence_length', self.sequence_length)
                # 如果config中有sequence_length，更新self.sequence_length以确保一致性
                if 'sequence_length' in config:
                    self.sequence_length = sequence_length
                feature_dim = config.get('feature_dim', 1)  # 默认1个特征
                input_shape = (sequence_length, feature_dim)

                if model_type == 'lstm_predictor':
                    # LSTM Predictor模型
                    hidden_units = config.get('hidden_units', 128)
                    num_layers = config.get('num_layers', 2)
                    dropout = config.get('dropout', 0.1)
                    activation = config.get('activation', 'tanh')

                    self.model = LSTMPredictor(
                        input_shape=input_shape,
                        hidden_units=hidden_units,
                        num_layers=num_layers,
                        dropout=dropout,
                        activation=activation
                    )
                elif model_type == 'lstm_autoencoder':
                    # LSTM Autoencoder模型
                    hidden_units = config.get('hidden_units', 128)
                    num_layers = config.get('num_layers', 2)
                    bottleneck_size = config.get('bottleneck_size', config.get('bottleneck_dim', 64))
                    dropout = config.get('dropout', 0.1)
                    activation = config.get('activation', 'tanh')

                    self.model = LSTMAutoencoder(
                        input_shape=input_shape,
                        hidden_units=hidden_units,
                        bottleneck_size=bottleneck_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        activation=activation
                    )
                elif model_type == 'cnn_1d_autoencoder':
                    # 1D CNN Autoencoder模型
                    num_filters = config.get('num_filters', 64)
                    kernel_size = config.get('kernel_size', 3)
                    bottleneck_size = config.get('bottleneck_size', config.get('bottleneck_dim', 64))
                    num_conv_layers = config.get('num_conv_layers', config.get('num_layers', 3))
                    dropout = config.get('dropout', 0.1)
                    activation = config.get('activation', 'relu')

                    self.model = CNN1DAutoencoder(
                        input_shape=input_shape,
                        num_filters=num_filters,
                        kernel_size=kernel_size,
                        bottleneck_size=bottleneck_size,
                        num_conv_layers=num_conv_layers,
                        dropout=dropout,
                        activation=activation
                    )
                else:
                    raise ValueError(f"不支持的模型类型: {model_type}")

                # 加载模型权重
                if model_path.exists():
                    ms.load_checkpoint(str(model_path), self.model)
                    print(f"📂 模型权重加载成功: {model_path} (类型: {model_type})")
                else:
                    print(f"⚠️ 模型权重文件不存在: {model_path}")
            else:
                print(f"⚠️ 模型配置文件不存在: {config_path}")

        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            self.model = None

    def _load_threshold(self, threshold_path: Union[str, Path]):
        """加载阈值"""
        try:
            with open(threshold_path, 'r', encoding='utf-8') as f:
                threshold_data = json.load(f)
                self.threshold_value = threshold_data.get(
                    'threshold_value',
                    threshold_data.get('threshold', 0.5)
                )
            print(f"📂 阈值加载成功: {self.threshold_value}")
        except Exception as e:
            print(f"⚠️ 阈值加载失败，使用默认值 0.5: {e}")
            self.threshold_value = 0.5

    def _load_scaler(self, scaler_path: Union[str, Path]):
        """加载标准化器"""
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                self._scaler_feature_names = getattr(self.scaler, 'feature_names_in_', None)
                self._scaler_feature_count = getattr(self.scaler, 'n_features_in_', None)
            print(f"📂 标准化器加载成功")
        except Exception as e:
            print(f"⚠️ 标准化器加载失败: {e}")
            self.scaler = None
            self._scaler_feature_names = None
            self._scaler_feature_count = None

    def _align_features_with_scaler(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """确保输入特征与训练阶段的标准化器维度一致"""
        if self.scaler is None:
            return feature_df

        aligned_df = feature_df.copy()

        if self._scaler_feature_names is not None and len(self._scaler_feature_names) > 0:
            # 根据特征名对齐，缺失的列用0填充，多余的列丢弃
            for name in self._scaler_feature_names:
                if name not in aligned_df.columns:
                    aligned_df[name] = 0.0
            aligned_df = aligned_df[list(self._scaler_feature_names)]
            return aligned_df

        if self._scaler_feature_count is not None:
            current_count = aligned_df.shape[1]
            if current_count < self._scaler_feature_count:
                missing = self._scaler_feature_count - current_count
                for idx in range(missing):
                    aligned_df[f'feature_pad_{idx+1}'] = 0.0
            elif current_count > self._scaler_feature_count:
                aligned_df = aligned_df.iloc[:, :self._scaler_feature_count]

        return aligned_df

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理输入数据，创建序列和对应的目标值

        Args:
            data: 输入数据 DataFrame

        Returns:
            元组 (sequences, targets)
            - sequences: 输入序列 (n_sequences, sequence_length, n_features)
            - targets: 目标值
              - 对于predictor模型: (n_sequences, n_features) - 下一个时间点的值
              - 对于autoencoder模型: (n_sequences, sequence_length, n_features) - 重构目标（与输入相同）
        """
        # 移除时间戳列（如果存在）
        if 'timestamp' in data.columns:
            feature_data = data.drop('timestamp', axis=1)
        else:
            feature_data = data

        # 对齐特征列与训练时的标准化器
        feature_data = self._align_features_with_scaler(feature_data)

        # 标准化
        if self.scaler is not None:
            feature_array = self.scaler.transform(feature_data.values)
        else:
            feature_array = feature_data.values

        # 根据模型类型创建不同的序列和目标
        is_autoencoder = self.model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']
        
        sequences = []
        targets = []
        
        if is_autoencoder:
            # 自编码器：输入和输出都是序列
            for i in range(len(feature_array) - self.sequence_length + 1):
                sequence = feature_array[i:i + self.sequence_length]
                sequences.append(sequence)
                targets.append(sequence)  # 目标与输入相同
        else:
            # 预测器：输入是序列，输出是下一个时间点
            for i in range(len(feature_array) - self.sequence_length):
                sequence = feature_array[i:i + self.sequence_length]
                sequences.append(sequence)
                target = feature_array[i + self.sequence_length]
                targets.append(target)

        if not sequences:
            min_samples = self.sequence_length if is_autoencoder else self.sequence_length + 1
            raise ValueError(f"数据长度不足，至少需要 {min_samples} 个样本")

        return np.array(sequences), np.array(targets)

    def detect_anomalies(self, sequences: np.ndarray, actual_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行异常检测

        根据模型类型使用不同的异常检测逻辑：
        - Predictor模型：根据前sequence_length个时间点预测下一个时间点，计算预测误差
        - Autoencoder模型：重构输入序列，计算重构误差

        Args:
            sequences: 输入序列 (n_sequences, sequence_length, n_features)
            actual_targets: 实际目标值
              - Predictor: (n_sequences, n_features)
              - Autoencoder: (n_sequences, sequence_length, n_features)

        Returns:
            元组 (predictions, anomaly_scores, anomaly_flags)
        """
        try:
            if self.model is None:
                raise ValueError("模型未加载，无法执行推理")

            # 设置推理模式
            self.model.set_train(False)

            n_sequences = len(sequences)
            predictions = []

            # 批量推理
            batch_size = 32
            for i in range(0, n_sequences, batch_size):
                batch_sequences = sequences[i:i + batch_size]
                batch_tensor = ms.Tensor(batch_sequences.astype(np.float32))

                # 模型预测
                batch_predictions = self.model(batch_tensor)
                predictions.extend(batch_predictions.asnumpy())

            predictions = np.array(predictions)

            # 根据模型类型计算异常分数
            is_autoencoder = self.model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']
            
            if is_autoencoder:
                # 自编码器：计算重构误差（RMSE）
                # predictions和actual_targets都是 (n_sequences, sequence_length, n_features)
                residuals = actual_targets - predictions
                # 对每个序列计算RMSE
                anomaly_scores = np.sqrt(np.mean(residuals ** 2, axis=(1, 2)))
            else:
                # 预测器：计算预测误差（L2范数）
                # predictions和actual_targets都是 (n_sequences, n_features)
                anomaly_scores = np.linalg.norm(actual_targets - predictions, axis=1)

            # 根据阈值判断异常
            anomaly_flags = anomaly_scores > self.threshold_value

            return predictions, anomaly_scores, anomaly_flags

        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            raise

    def run_inference(self, data_path: Union[str, Path],
                     batch_size: int = 32) -> Dict[str, Any]:
        """
        执行完整推理流程

        Args:
            data_path: 数据文件路径
            batch_size: 批次大小

        Returns:
            推理结果字典
        """
        try:
            # 读取数据
            data = pd.read_csv(data_path)
            total_samples = len(data)

            # 预处理数据：获取序列和对应的目标值
            sequences, targets = self.preprocess_data(data)
            n_sequences = len(sequences)

            # 执行异常检测
            predictions, anomaly_scores, anomaly_flags = self.detect_anomalies(sequences, targets)

            # 统计结果
            anomalies_detected = int(np.sum(anomaly_flags))
            anomaly_percentage = (anomalies_detected / n_sequences) * 100 if n_sequences > 0 else 0

            # 生成时间戳（对应预测的时间点）
            # 对于autoencoder模型，序列从索引0开始，每个序列对应其最后一个时间点
            # 对于predictor模型，序列从索引0开始，每个序列对应下一个时间点
            if 'timestamp' in data.columns:
                if self.model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']:
                    # Autoencoder: 每个序列对应其最后一个时间点（索引 sequence_length - 1, sequence_length, ..., len-1）
                    start_idx = self.sequence_length - 1
                else:
                    # Predictor: 每个序列对应下一个时间点（索引 sequence_length, sequence_length+1, ..., len-1）
                    start_idx = self.sequence_length
                
                # 确保索引不超出范围
                if start_idx + n_sequences <= len(data):
                    timestamps = data['timestamp'].iloc[start_idx:start_idx + n_sequences].tolist()
                else:
                    # 如果数据不够，只取能取到的部分
                    available_len = len(data) - start_idx
                    timestamps = data['timestamp'].iloc[start_idx:].tolist() if available_len > 0 else []
                    # 如果时间戳数量不够，用最后一个时间戳填充
                    if len(timestamps) < n_sequences and len(timestamps) > 0:
                        last_timestamp = timestamps[-1]
                        timestamps.extend([last_timestamp] * (n_sequences - len(timestamps)))
            else:
                # 生成模拟时间戳
                timestamps = pd.date_range(
                    start='2024-01-01 00:00:00',
                    periods=n_sequences,
                    freq='1H'
                ).strftime('%Y-%m-%d %H:%M:%S').tolist()

            # 构建结果
            result = {
                'inference_id': f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'total_samples': total_samples,
                'sequences_count': n_sequences,
                'anomalies_detected': anomalies_detected,
                'anomaly_percentage': round(anomaly_percentage, 2),
                'threshold_value': self.threshold_value,
                'timestamps': timestamps,
                'anomaly_scores': anomaly_scores.tolist(),
                'anomaly_flags': anomaly_flags.tolist(),
                'predictions': predictions.tolist(),
                'actual_targets': targets.tolist(),
                'statistics': {
                    'mean_score': float(np.mean(anomaly_scores)),
                    'std_score': float(np.std(anomaly_scores)),
                    'min_score': float(np.min(anomaly_scores)),
                    'max_score': float(np.max(anomaly_scores)),
                    'median_score': float(np.median(anomaly_scores))
                },
                'processing_location': 'edge',
                'created_at': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            self.logger.error(f"推理执行失败: {e}")
            raise