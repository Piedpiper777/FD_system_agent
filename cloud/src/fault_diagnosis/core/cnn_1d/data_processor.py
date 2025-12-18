"""
故障诊断 CNN 1D - 数据处理器
负责加载、预处理和构建分类任务的时序数据

标签处理说明：
    标签由前端传递的 label_index 决定，而不是从文件名提取。
    用户在训练页面设置标签列表（如"正常,内圈故障,外圈故障"），
    然后为每个标签选择对应的数据文件，标签与文件的绑定关系在前端完成。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ClassificationData:
    """分类任务数据容器"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, label_names: List[str]):
        """
        Args:
            sequences: 时序序列 (n_samples, seq_len, n_features)
            labels: 标签索引 (n_samples,)
            label_names: 标签名称列表，如 ['normal', 'inner_race', 'outer_race']
        """
        self.sequences = sequences
        self.labels = labels
        self.label_names = label_names


class DataProcessor:
    """
    故障诊断数据处理器
    
    注意：标签由前端传递的 label_index 决定，而不是从文件名提取。
    用户在训练页面设置标签列表（如"正常,内圈故障,外圈故障"），
    然后为每个标签选择对应的数据文件，标签与文件的绑定关系在前端完成。
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        stride: int = 1,
        normalize: bool = True,
        num_classes: int = 3,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        初始化数据处理器
        
        Args:
            sequence_length: 时序窗口长度
            stride: 滑动窗口步长
            normalize: 是否标准化数据
            num_classes: 分类数量（由用户定义的标签数量决定）
            feature_columns: 特征列名，如果为None则自动检测数值列
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.num_classes = num_classes
        
        self.scaler = StandardScaler() if normalize else None
        self.feature_columns = feature_columns or []
    
    def load_data(
        self,
        file_path: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, None]:
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径
            feature_columns: 特征列名，如果为None则自动检测
            
        Returns:
            (数据DataFrame, None) - 标签由前端传递，不再从文件名提取
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        if file_path.suffix.lower() != ".csv":
            raise ValueError("目前仅支持CSV格式数据")
        
        # 加载数据
        data = pd.read_csv(file_path)
        
        # 确定特征列
        if feature_columns is None:
            # 自动检测数值列，排除可能的标签列和时间戳列
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            # 排除可能的标签列和时间戳列
            exclude_cols = ['label', 'Label', 'LABEL', 'class', 'Class', 'CLASS', 
                          'time', 'Time', 'TIME', 'timestamp', 'Timestamp', 'TIMESTAMP',
                          '时间', '时间戳', 'date', 'Date', 'DATE']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # 如果第一列是时间戳（通常是字符串或日期类型），也排除
            if len(data.columns) > 0:
                first_col = data.columns[0]
                first_col_lower = first_col.lower()
                # 检查列名是否包含时间戳相关关键词
                if any(keyword in first_col_lower for keyword in ['time', 'timestamp', 'date', '时间', '时间戳']):
                    if first_col in numeric_cols:
                        numeric_cols.remove(first_col)
                # 如果第一列不是数值类型，也排除
                elif data[first_col].dtype not in [np.number]:
                    if first_col in numeric_cols:
                        numeric_cols.remove(first_col)
                # 如果第一列是数值类型但看起来像时间戳（值很大，可能是Unix时间戳），也排除
                elif data[first_col].dtype in [np.number]:
                    # 检查值是否像时间戳（通常时间戳值很大，如 > 1000000000）
                    sample_values = data[first_col].dropna().head(10)
                    if len(sample_values) > 0 and sample_values.min() > 1000000000:
                        if first_col in numeric_cols:
                            numeric_cols.remove(first_col)
            
            if not numeric_cols:
                raise ValueError("数据集中没有可用于训练的数值列")
            
            # 如果指定了特征列，使用指定的；否则使用检测到的
            if self.feature_columns:
                # 验证指定的列是否存在（排除工况列，因为工况列是动态添加的）
                # 只检查原始数据中存在的列
                original_cols = [col for col in self.feature_columns if not col.startswith('condition_')]
                missing = [col for col in original_cols if col not in data.columns]
                if missing:
                    raise ValueError(f"数据缺少以下特征列: {missing}")
                feature_cols = numeric_cols  # 使用检测到的数值列
            else:
                feature_cols = numeric_cols
                self.feature_columns = feature_cols
        else:
            # 验证指定的列是否存在（排除工况列）
            original_cols = [col for col in feature_columns if not col.startswith('condition_')]
            missing = [col for col in original_cols if col not in data.columns]
            if missing:
                raise ValueError(f"数据缺少以下特征列: {missing}")
            feature_cols = [col for col in feature_columns if col in data.columns] or numeric_cols
            self.feature_columns = feature_cols
        
        return data[feature_cols], None  # 标签由前端传递，不再从文件名提取
    
    def preprocess(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        预处理数据：填充缺失值、标准化
        
        Args:
            raw_data: 原始数据DataFrame
            
        Returns:
            预处理后的numpy数组
        """
        data = raw_data[self.feature_columns].copy()
        
        # 填充缺失值
        if data.isnull().any().any():
            data = data.ffill().bfill()
            # 如果还有缺失值（全列都是NaN），填充0
            data = data.fillna(0)
        
        array = data.values.astype(np.float32)
        
        # 标准化
        if self.normalize and self.scaler is not None:
            array = self.scaler.fit_transform(array)
        
        return array
    
    def create_sequences(
        self,
        data: np.ndarray,
        label: str = None,  # 保留参数以向后兼容，但不再使用
    ) -> Tuple[np.ndarray, None]:
        """
        创建时序序列（使用滑动窗口）
        
        注意：标签由前端传递的 label_index 决定，在 API 层面处理，
        此函数不再处理标签。
        
        Args:
            data: 预处理后的数据 (n_samples, n_features)
            label: 已废弃，保留以向后兼容
            
        Returns:
            (sequences, None) - sequences: (n_sequences, seq_len, n_features)
        """
        n_samples = data.shape[0]
        sequences: List[np.ndarray] = []
        
        # 使用滑动窗口创建序列
        for start in range(0, n_samples - self.sequence_length + 1, self.stride):
            end = start + self.sequence_length
            seq = data[start:end]
            sequences.append(seq)
        
        if not sequences:
            raise ValueError(
                f"数据量不足以构建序列，至少需要 {self.sequence_length} 条记录"
            )
        
        sequences = np.stack(sequences)
        return sequences, None  # 标签由 API 层面根据 label_index 处理
    
    def process_single_file(
        self,
        file_path: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        处理单个文件，返回序列
        
        注意：标签由前端传递，不再从文件名提取
        
        Args:
            file_path: 文件路径
            feature_columns: 特征列名
            
        Returns:
            sequences: (n_sequences, seq_len, n_features)
        """
        raw_data, _ = self.load_data(file_path, feature_columns)
        processed_data = self.preprocess(raw_data)
        sequences, _ = self.create_sequences(processed_data)
        
        return sequences
    
    def split_dataset(
        self,
        data: ClassificationData,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        random_seed: int = 42,
    ) -> Tuple[ClassificationData, ClassificationData]:
        """
        划分训练集和验证集
        
        Args:
            data: ClassificationData对象
            train_ratio: 训练集比例
            shuffle: 是否打乱数据
            random_seed: 随机种子
            
        Returns:
            (train_data, val_data)
        """
        n_samples = len(data.sequences)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        
        # 确保分割后的数据集足够大
        min_samples = 32
        if split_idx < min_samples:
            split_idx = min_samples
        if n_samples - split_idx < min_samples:
            split_idx = n_samples - min_samples
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_data = ClassificationData(
            sequences=data.sequences[train_indices],
            labels=data.labels[train_indices],
            label_names=data.label_names,
        )
        
        val_data = ClassificationData(
            sequences=data.sequences[val_indices],
            labels=data.labels[val_indices],
            label_names=data.label_names,
        )
        
        return train_data, val_data
    
    def save_scaler_params(self, file_path: Union[str, Path]):
        """保存标准化参数"""
        if not self.scaler:
            return
        
        params = {
            "mean": self.scaler.mean_,
            "scale": self.scaler.scale_,
            "feature_names": self.feature_columns,
            "label_names": self.label_names,
            "num_classes": self.num_classes,
        }
        np.savez(file_path, **params)
    
    @staticmethod
    def load_scaler_params(file_path: Union[str, Path]) -> Dict[str, Any]:
        """加载标准化参数"""
        data = np.load(file_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    
    def get_scaler_params(self) -> Dict[str, Any]:
        """获取标准化参数"""
        if not self.scaler:
            return {}
        
        return {
            "mean": getattr(self.scaler, "mean_", None),
            "scale": getattr(self.scaler, "scale_", None),
            "feature_names": self.feature_columns,
            "label_names": self.label_names,
            "num_classes": self.num_classes,
        }

