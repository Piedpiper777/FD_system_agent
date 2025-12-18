"""
LSTM预测异常检测模块 - 异常检测器
实现实时异常检测功能
"""

import mindspore as ms
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path
import logging
from datetime import datetime

from .model_builder import ModelBuilder
from .threshold_calculator import ThresholdCalculator
from .data_processor import DataProcessor


class AnomalyDetector:
    """
    实时异常检测器

    负责：
    - 加载训练好的模型和阈值
    - 实时预处理输入数据
    - 执行异常检测
    - 记录异常信息
    """

    def __init__(self, model_path: Union[str, Path],
                 threshold_path: Union[str, Path],
                 scaler_params: Dict[str, np.ndarray],
                 sequence_length: int = 50):
        """
        初始化异常检测器

        Args:
            model_path: 模型权重文件路径
            threshold_path: 阈值文件路径
            scaler_params: 标准化参数
            sequence_length: 序列长度
        """
        self.sequence_length = sequence_length
        self.model = None
        self.threshold_calculator = None
        self.scaler_params = scaler_params

        # 历史数据缓冲区（用于构建序列）
        self.data_buffer = []

        # 日志记录器
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 加载组件
        self._load_model(model_path)
        self._load_threshold(threshold_path)

        print(f"✅ 异常检测器初始化完成")
        print(f"  - 模型: {model_path}")
        print(f"  - 阈值: {threshold_path}")
        print(f"  - 序列长度: {sequence_length}")

    def _load_model(self, model_path: Union[str, Path]):
        """加载模型"""
        try:
            self.model = ms.load_checkpoint(str(model_path))
            print(f"📂 模型加载成功: {model_path}")
        except Exception as e:
            raise ValueError(f"模型加载失败: {e}")

    def _load_threshold(self, threshold_path: Union[str, Path]):
        """加载阈值"""
        self.threshold_calculator = ThresholdCalculator()
        self.threshold_calculator.load_threshold(threshold_path)

    def preprocess_online(self, new_data: Union[np.ndarray, List, Dict]) -> np.ndarray:
        """
        对实时数据进行预处理

        Args:
            new_data: 新数据（单个样本或批次）

        Returns:
            预处理后的数据
        """
        # 转换为numpy数组
        if isinstance(new_data, dict):
            # 如果是字典，按特征顺序提取
            feature_names = self.scaler_params.get('feature_names', [])
            if feature_names:
                data_array = np.array([new_data.get(name, 0) for name in feature_names])
            else:
                data_array = np.array(list(new_data.values()))
        elif isinstance(new_data, list):
            data_array = np.array(new_data)
        else:
            data_array = np.array(new_data)

        # 确保是二维数组
        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)

        # 标准化
        if 'mean' in self.scaler_params and 'scale' in self.scaler_params:
            data_array = (data_array - self.scaler_params['mean']) / self.scaler_params['scale']

        return data_array

    def update_buffer(self, new_data: np.ndarray):
        """
        更新数据缓冲区

        Args:
            new_data: 新数据 (n_features,)
        """
        self.data_buffer.append(new_data.flatten())

        # 保持缓冲区大小
        if len(self.data_buffer) > self.sequence_length:
            self.data_buffer.pop(0)

    def create_sequence(self) -> Optional[np.ndarray]:
        """
        从缓冲区创建序列

        Returns:
            序列数据或None（如果数据不足）
        """
        if len(self.data_buffer) < self.sequence_length:
            return None

        # 取最近的sequence_length个样本
        sequence_data = np.array(self.data_buffer[-self.sequence_length:])
        return sequence_data.reshape(1, self.sequence_length, -1)  # (1, seq_len, n_features)

    def detect(self, input_data: Union[np.ndarray, List, Dict]) -> Dict[str, Any]:
        """
        执行异常检测

        Args:
            input_data: 输入数据

        Returns:
            检测结果字典
        """
        # 预处理数据
        processed_data = self.preprocess_online(input_data)

        # 更新缓冲区
        self.update_buffer(processed_data[0])  # 假设每次只处理一个样本

        # 创建序列
        sequence = self.create_sequence()
        if sequence is None:
            return {
                'is_anomaly': False,
                'residual_score': 0.0,
                'confidence': 0.0,
                'status': 'collecting_data',
                'message': f'数据不足，还需要 {self.sequence_length - len(self.data_buffer)} 个样本'
            }

        # 模型预测
        try:
            input_tensor = ms.Tensor(sequence, ms.float32)
            prediction = self.model(input_tensor)
            prediction = prediction.asnumpy()[0]  # 取第一个样本的预测

            # 这里需要实际的下一时刻目标值，但在在线检测中通常不可用
            # 因此我们使用预测值本身作为"实际值"的近似，或者返回预测结果
            # 在实际应用中，可能需要等待下一个时间点的真实值

            # 由于在线检测时没有真实值，我们返回预测信息
            result = {
                'prediction': prediction.tolist(),
                'residual_score': None,  # 在线检测时无法计算残差
                'is_anomaly': None,      # 在线检测时无法判断异常
                'confidence': 0.5,       # 默认置信度
                'status': 'prediction_only',
                'timestamp': datetime.now().isoformat(),
                'message': '在线预测完成，等待实际值进行异常判断'
            }

        except Exception as e:
            result = {
                'is_anomaly': False,
                'residual_score': 0.0,
                'confidence': 0.0,
                'status': 'error',
                'message': f'检测失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

        return result

    def detect_with_actual(self, input_sequence: np.ndarray,
                          actual_next: np.ndarray) -> Dict[str, Any]:
        """
        使用实际值进行异常检测（离线评估或有真实值的场景）

        Args:
            input_sequence: 输入序列 (seq_len, n_features)
            actual_next: 实际的下一时刻值 (n_features,)

        Returns:
            检测结果字典
        """
        try:
            # 模型预测
            input_tensor = ms.Tensor(input_sequence.reshape(1, *input_sequence.shape), ms.float32)
            prediction = self.model(input_tensor)
            prediction = prediction.asnumpy()[0]

            # 计算异常分数
            residual_score, is_anomaly = self.threshold_calculator.detect_anomaly(
                prediction, actual_next
            )

            # 计算置信度（基于阈值的距离）
            threshold = self.threshold_calculator.threshold
            confidence = min(1.0, max(0.0, 1.0 - (residual_score - threshold) / threshold))

            result = {
                'prediction': prediction.tolist(),
                'actual': actual_next.tolist(),
                'residual_score': float(residual_score),
                'is_anomaly': bool(is_anomaly),
                'confidence': float(confidence),
                'threshold': float(threshold),
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            result = {
                'is_anomaly': False,
                'residual_score': 0.0,
                'confidence': 0.0,
                'status': 'error',
                'message': f'检测失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

        return result

    def log_anomaly(self, anomaly_info: Dict[str, Any], log_file: Optional[str] = None):
        """
        记录异常信息

        Args:
            anomaly_info: 异常信息字典
            log_file: 日志文件路径
        """
        if anomaly_info.get('is_anomaly'):
            log_message = (
                f"[{anomaly_info['timestamp']}] 异常检测: "
                f"残差分数={anomaly_info['residual_score']:.4f}, "
                f"阈值={anomaly_info.get('threshold', 'N/A')}, "
                f"置信度={anomaly_info['confidence']:.4f}"
            )

            self.logger.warning(log_message)

            # 如果指定了日志文件，写入文件
            if log_file:
                try:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(log_message + '\n')
                except Exception as e:
                    self.logger.error(f"日志写入失败: {e}")

    def get_detector_info(self) -> Dict[str, Any]:
        """
        获取检测器信息

        Returns:
            检测器信息字典
        """
        return {
            'sequence_length': self.sequence_length,
            'buffer_size': len(self.data_buffer),
            'threshold': self.threshold_calculator.threshold if self.threshold_calculator else None,
            'residual_method': self.threshold_calculator.residual_method if self.threshold_calculator else None,
            'feature_names': self.scaler_params.get('feature_names', []),
            'model_loaded': self.model is not None,
            'threshold_loaded': self.threshold_calculator is not None
        }

    def reset_buffer(self):
        """重置数据缓冲区"""
        self.data_buffer = []
        print("🔄 数据缓冲区已重置")