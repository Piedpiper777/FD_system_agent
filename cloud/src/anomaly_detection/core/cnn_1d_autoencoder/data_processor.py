"""
1D CNN自编码器 - 数据处理器
完成数据加载、标准化和滑动窗口序列构建（重构任务）
复用LSTM Autoencoder的数据处理逻辑
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TimeSeriesData:
    """时序序列容器，targets与输入序列相同（重构任务）"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = sequences
        self.targets = targets


class DataProcessor:
    """自编码器数据处理，专注重构"""

    def __init__(
        self,
        sequence_length: int = 50,
        stride: int = 1,
        normalize: bool = True,
    ):
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize

        self.scaler = StandardScaler() if normalize else None
        self.feature_names: List[str] = []

    def load_data(
        self,
        file_path: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        if file_path.suffix.lower() != ".csv":
            raise ValueError("目前仅支持CSV格式数据")

        data = pd.read_csv(file_path)
        if feature_columns is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("数据集中没有可用于训练的数值列")
            self.feature_names = numeric_cols
        else:
            missing = [col for col in feature_columns if col not in data.columns]
            if missing:
                raise ValueError(f"数据缺少以下特征列: {missing}")
            self.feature_names = feature_columns

        return data

    def preprocess(self, raw_data: pd.DataFrame) -> np.ndarray:
        data = raw_data[self.feature_names].copy()
        if data.isnull().any().any():
            data = data.fillna(method="ffill").fillna(method="bfill")

        array = data.values.astype(np.float32)
        if self.normalize and self.scaler is not None:
            array = self.scaler.fit_transform(array)
        return array

    def create_sequences(self, data: np.ndarray) -> TimeSeriesData:
        n_samples = data.shape[0]
        sequences: List[np.ndarray] = []

        for start in range(0, n_samples - self.sequence_length + 1, self.stride):
            end = start + self.sequence_length
            seq = data[start:end]
            sequences.append(seq)

        if not sequences:
            raise ValueError(
                "数据量不足以构建序列，" f"至少需要 {self.sequence_length} 条记录"
            )

        sequences = np.stack(sequences)
        targets = sequences.copy()
        return TimeSeriesData(sequences=sequences, targets=targets)

    def split_dataset(
        self, data: np.ndarray, train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        split_idx = int(len(data) * train_ratio)
        
        # 🔧 确保分割后的数据集足够大，至少能形成一个batch
        min_samples = 32  # 最小batch_size
        if split_idx < min_samples:
            split_idx = min_samples
        if len(data) - split_idx < min_samples:
            split_idx = len(data) - min_samples
        
        return data[:split_idx], data[split_idx:]

    def process_pipeline(
        self,
        file_path: Union[str, Path],
        train_ratio: float = 0.8,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[TimeSeriesData, TimeSeriesData]:
        raw = self.load_data(file_path, feature_columns=feature_columns)
        processed = self.preprocess(raw)
        dataset = self.create_sequences(processed)

        train_seq, val_seq = self.split_dataset(dataset.sequences, train_ratio)
        train_targets, val_targets = self.split_dataset(dataset.targets, train_ratio)

        return (
            TimeSeriesData(train_seq, train_targets),
            TimeSeriesData(val_seq, val_targets),
        )

    def save_scaler_params(self, file_path: Union[str, Path]):
        if not self.scaler:
            return
        params = {
            "mean": self.scaler.mean_,
            "scale": self.scaler.scale_,
            "feature_names": self.feature_names,
        }
        np.savez(file_path, **params)

    @staticmethod
    def load_scaler_params(file_path: Union[str, Path]) -> Dict[str, Any]:
        data = np.load(file_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    def get_scaler_params(self) -> Dict[str, Any]:
        if not self.scaler:
            return {}
        return {
            "mean": getattr(self.scaler, "mean_", None),
            "scale": getattr(self.scaler, "scale_", None),
            "feature_names": self.feature_names,
        }

