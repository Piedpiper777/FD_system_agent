"""
LSTM自编码器 - 异常检测器
根据重构误差进行在线或离线异常判定
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import mindspore as ms
import mindspore.ops as ops
import numpy as np

from .model_builder import ModelBuilder
from .threshold_calculator import ThresholdCalculator
from .data_processor import DataProcessor


class AnomalyDetector:
    """加载自编码器模型并执行异常检测"""

    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_params: Dict[str, np.ndarray],
        sequence_length: int,
        input_dim: int,
        threshold_path: Optional[Union[str, Path]] = None,
    ):
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.scaler_params = scaler_params
        self.threshold_calculator = ThresholdCalculator()

        self.model = ModelBuilder.create_model(
            "lstm_autoencoder",
            input_shape=(sequence_length, input_dim),
        )
        param_dict = ms.load_checkpoint(str(model_path))
        ms.load_param_into_net(self.model, param_dict)
        self.model.set_train(False)

        if threshold_path and Path(threshold_path).exists():
            self.threshold_calculator.load(threshold_path)

        self.buffer: List[np.ndarray] = []

    def _normalize(self, sample: np.ndarray) -> np.ndarray:
        mean = self.scaler_params.get("mean")
        scale = self.scaler_params.get("scale")
        if mean is None or scale is None:
            return sample
        return (sample - mean) / scale

    def update_buffer(self, sample: np.ndarray):
        self.buffer.append(sample)
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)

    def ready(self) -> bool:
        return len(self.buffer) >= self.sequence_length

    def detect(self, sample: Union[List[float], np.ndarray]) -> Dict[str, Union[float, bool, str]]:
        sample = np.asarray(sample, dtype=np.float32)
        sample = self._normalize(sample)
        self.update_buffer(sample)

        if not self.ready():
            return {
                "status": "collecting",
                "needed": self.sequence_length - len(self.buffer),
                "is_anomaly": False,
            }

        window = np.stack(self.buffer[-self.sequence_length:])
        window = window.reshape(1, self.sequence_length, self.input_dim)
        input_tensor = ms.Tensor(window, ms.float32)
        reconstruction = self.model(input_tensor).asnumpy()
        residual = window - reconstruction
        score = float(np.sqrt(np.mean(residual ** 2)))
        threshold = self.threshold_calculator.threshold
        is_anomaly = bool(threshold is not None and score > threshold)

        return {
            "status": "ready",
            "reconstruction_error": score,
            "threshold": threshold,
            "is_anomaly": is_anomaly,
        }
