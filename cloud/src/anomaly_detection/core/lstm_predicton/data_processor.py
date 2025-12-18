"""
LSTM预测异常检测模块 - 数据处理器
负责数据的加载、清洗、预处理和时序窗口划分
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class TimeSeriesData:
    """时序数据容器"""
    def __init__(self, sequences: np.ndarray, targets: np.ndarray,
                 timestamps: Optional[np.ndarray] = None):
        self.sequences = sequences  # (n_samples, seq_len, n_features)
        self.targets = targets      # (n_samples, n_features)
        self.timestamps = timestamps


class DataProcessor:
    """
    时序数据处理器

    专门针对工业异常检测任务设计：
    - 数据加载和清洗
    - 标准化预处理
    - 滑动窗口序列创建
    - 训练/验证/测试集划分
    """

    def __init__(self, sequence_length: int = 50, prediction_horizon: int = 1,
                 stride: int = 1, normalize: bool = True):
        """
        初始化数据处理器

        Args:
            sequence_length: 序列长度（历史窗口大小）
            prediction_horizon: 预测步长
            stride: 滑动窗口步长
            normalize: 是否标准化
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.normalize = normalize

        # 预处理组件
        self.scaler = StandardScaler() if normalize else None
        self.feature_names: List[str] = []
        self.target_names: List[str] = []

        # 处理后的数据
        self.train_data: Optional[np.ndarray] = None
        self.val_data: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None

        print(f"✅ 数据处理器初始化完成")
        print(f"  - 序列长度: {sequence_length}")
        print(f"  - 预测步长: {prediction_horizon}")
        print(f"  - 标准化: {'启用' if normalize else '禁用'}")

    def load_data(self, file_path: Union[str, Path],
                  feature_columns: Optional[List[str]] = None,
                  target_columns: Optional[List[str]] = None,
                  timestamp_column: Optional[str] = None,
                  label_column: Optional[str] = None) -> pd.DataFrame:
        """
        加载原始数据

        Args:
            file_path: 数据文件路径（CSV格式）
            feature_columns: 特征列名列表
            target_columns: 目标列名列表
            timestamp_column: 时间戳列名
            label_column: 标签列名（用于测试数据）

        Returns:
            加载的DataFrame
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 加载数据
        if file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        print(f"📊 数据加载完成: {file_path}")
        print(f"  - 数据形状: {data.shape}")
        print(f"  - 列名: {list(data.columns)}")

        # 自动识别时间戳列
        if timestamp_column is None:
            timestamp_candidates = ['timestamp', 'time', 'date', 'datetime']
            for col in data.columns:
                if col.lower() in timestamp_candidates or 'time' in col.lower():
                    timestamp_column = col
                    break
            # 如果没找到，假设第一列是时间戳
            if timestamp_column is None and len(data.columns) > 0:
                timestamp_column = data.columns[0]

        # 存储时间戳列名
        self.timestamp_column = timestamp_column

        # 设置特征和目标列
        if feature_columns is None:
            # 自动识别特征列：排除时间戳列和标签列
            exclude_cols = set()
            if timestamp_column and timestamp_column in data.columns:
                exclude_cols.add(timestamp_column)
            if label_column and label_column in data.columns:
                exclude_cols.add(label_column)

            # 选择数值列作为特征，排除指定的列
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_names = [col for col in numeric_cols if col not in exclude_cols]

            # 如果没有找到数值特征列，尝试使用所有非排除列
            if not self.feature_names:
                all_cols = [col for col in data.columns if col not in exclude_cols]
                self.feature_names = all_cols
                print(f"⚠️ 未找到数值特征列，使用所有可用列: {self.feature_names}")
        else:
            self.feature_names = feature_columns

        if target_columns is None:
            self.target_names = self.feature_names
        else:
            self.target_names = target_columns

        # 存储标签列（如果有）
        self.label_column = label_column

        print(f"🔍 列识别完成:")
        print(f"  - 时间戳列: {timestamp_column}")
        print(f"  - 特征列: {self.feature_names}")
        if label_column:
            print(f"  - 标签列: {label_column}")

        return data

    def preprocess_data(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        数据清洗和预处理

        Args:
            raw_data: 原始数据

        Returns:
            处理后的数值数组
        """
        # 选择特征列
        if not all(col in raw_data.columns for col in self.feature_names):
            missing_cols = [col for col in self.feature_names if col not in raw_data.columns]
            raise ValueError(f"数据中缺少特征列: {missing_cols}")

        data = raw_data[self.feature_names].copy()

        # 处理缺失值
        if data.isnull().any().any():
            print(f"⚠️ 发现缺失值，使用前向填充处理")
            data = data.fillna(method='ffill').fillna(method='bfill')

        # 转换为numpy数组
        processed_data = data.values.astype(np.float32)

        # 标准化
        if self.normalize and self.scaler is not None:
            processed_data = self.scaler.fit_transform(processed_data)

        print(f"🔧 数据预处理完成")
        print(f"  - 特征列: {self.feature_names}")
        print(f"  - 数据形状: {processed_data.shape}")
        if self.normalize:
            print(f"  - 标准化参数: mean={self.scaler.mean_}, std={self.scaler.scale_}")

        return processed_data

    def create_sequences(self, data: np.ndarray) -> TimeSeriesData:
        """
        创建滑动窗口序列

        Args:
            data: 输入数据 (n_samples, n_features)

        Returns:
            TimeSeriesData对象
        """
        n_samples, n_features = data.shape
        sequences = []
        targets = []

        for i in range(0, n_samples - self.sequence_length - self.prediction_horizon + 1, self.stride):
            # 输入序列
            seq_end = i + self.sequence_length
            sequence = data[i:seq_end]  # (sequence_length, n_features)

            # 目标值
            target_idx = seq_end + self.prediction_horizon - 1
            target = data[target_idx]  # (n_features,)

            sequences.append(sequence)
            targets.append(target)

        sequences = np.array(sequences)  # (n_sequences, seq_len, n_features)
        targets = np.array(targets)      # (n_sequences, n_features)

        print(f"🔄 序列创建完成")
        print(f"  - 原始样本数: {n_samples}")
        print(f"  - 生成序列数: {len(sequences)}")
        print(f"  - 序列形状: {sequences.shape}")

        return TimeSeriesData(sequences, targets)

    def load_test_data(self, file_path: Union[str, Path],
                      feature_columns: Optional[List[str]] = None,
                      label_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        加载测试数据（包含标签）

        Args:
            file_path: 测试数据文件路径
            feature_columns: 特征列名列表
            label_column: 标签列名

        Returns:
            (测试数据DataFrame, 标签数组)
        """
        # 加载数据
        test_data = self.load_data(file_path, feature_columns=feature_columns, label_column=label_column)

        # 提取标签（如果有）
        labels = None
        if label_column and label_column in test_data.columns:
            labels = test_data[label_column].values
            print(f"🏷️ 提取标签完成: {label_column}, 形状: {labels.shape}")
            print(f"  - 正样本比例: {np.mean(labels):.3f}")

        return test_data, labels

    def process_test_pipeline(self, file_path: Union[str, Path],
                             feature_columns: Optional[List[str]] = None,
                             label_column: Optional[str] = None) -> Tuple[TimeSeriesData, Optional[np.ndarray]]:
        """
        测试数据处理流水线

        Args:
            file_path: 测试数据文件路径
            feature_columns: 特征列名列表
            label_column: 标签列名

        Returns:
            (测试序列数据, 标签数组)
        """
        # 1. 加载测试数据
        test_data, labels = self.load_test_data(file_path, feature_columns, label_column)

        # 2. 预处理（注意：测试数据应该使用训练数据的标准化参数）
        processed_data = self.preprocess_data(test_data)

        # 3. 创建序列
        test_sequences = self.create_sequences(processed_data)

        return test_sequences, labels

    def split_dataset(self, data: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        按时序分割训练集和验证集

        Args:
            data: 输入数据
            train_ratio: 训练集比例

        Returns:
            (train_data, val_data)
        """
        n_samples = len(data)
        train_end = int(n_samples * train_ratio)

        # 🔧 确保分割后的数据集足够大，至少能形成一个batch
        min_samples = 32  # 最小batch_size
        if train_end < min_samples:
            train_end = min_samples
        if n_samples - train_end < min_samples:
            train_end = n_samples - min_samples

        train_data = data[:train_end]
        val_data = data[train_end:]

        print(f"✂️ 数据集划分完成")
        print(f"  - 总样本数: {n_samples}")
        print(f"  - 训练集: {len(train_data)} 样本")
        print(f"  - 验证集: {len(val_data)} 样本")
        print(f"  - train_ratio: {train_ratio}")

        return train_data, val_data

    def process_pipeline(self, file_path: Union[str, Path],
                        train_ratio: float = 0.8,
                        feature_columns: Optional[List[str]] = None,
                        label_column: Optional[str] = None) -> Tuple[TimeSeriesData, TimeSeriesData]:
        """
        完整的数据处理流水线

        Args:
            file_path: 数据文件路径
            train_ratio: 训练集比例
            feature_columns: 特征列名列表
            label_column: 标签列名（用于测试数据）

        Returns:
            (train_data, val_data) TimeSeriesData对象
        """
        # 1. 加载数据
        raw_data = self.load_data(file_path, feature_columns=feature_columns, label_column=label_column)

        # 2. 预处理
        processed_data = self.preprocess_data(raw_data)

        # 3. 创建序列
        all_sequences = self.create_sequences(processed_data)

        # 4. 划分数据集
        train_sequences, val_sequences = self.split_dataset(
            all_sequences.sequences, train_ratio
        )
        train_targets, val_targets = self.split_dataset(
            all_sequences.targets, train_ratio
        )

        # 5. 创建TimeSeriesData对象
        train_data = TimeSeriesData(train_sequences, train_targets)
        val_data = TimeSeriesData(val_sequences, val_targets)

        return train_data, val_data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        反标准化数据

        Args:
            data: 标准化后的数据

        Returns:
            原始尺度的数据
        """
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)

    def get_scaler_params(self) -> Dict[str, np.ndarray]:
        """
        获取标准化参数（用于在线推理）

        Returns:
            标准化参数字典
        """
        if self.scaler is None:
            return {}
        return {
            'mean': self.scaler.mean_,
            'scale': self.scaler.scale_,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }

    def save_scaler_params(self, file_path: str):
        """保存标准化参数"""
        params = self.get_scaler_params()
        np.savez(file_path, **params)
        print(f"💾 标准化参数已保存: {file_path}")

    @classmethod
    def load_scaler_params(cls, file_path: str) -> Dict[str, np.ndarray]:
        """加载标准化参数"""
        data = np.load(file_path, allow_pickle=True)
        params = {key: data[key] for key in data.files}
        print(f"📂 标准化参数已加载: {file_path}")
        return params