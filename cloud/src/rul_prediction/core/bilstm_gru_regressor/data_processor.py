"""
RUL预测 BiLSTM/GRU 回归器 - 数据处理器
负责加载、预处理和构建RUL回归任务的时序数据

RUL标签计算说明：
    1. 从元文件中读取失效点索引（failure_row_index）
    2. 根据RUL单位（cycle/second/minute）计算每个时间点的RUL值
    3. RUL值 = max(0, failure_row_index - current_row_index)
    4. 如果RUL值超过max_rul，则截断为max_rul
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class RegressionData:
    """回归任务数据容器"""
    
    def __init__(self, sequences: np.ndarray, rul_labels: np.ndarray):
        """
        Args:
            sequences: 时序序列 (n_samples, seq_len, n_features)
            rul_labels: RUL标签值 (n_samples,)
        """
        self.sequences = sequences
        self.rul_labels = rul_labels


class DataProcessor:
    """
    RUL预测数据处理器
    
    处理流程：
    1. 加载数据文件和元文件
    2. 从元文件读取RUL配置（失效点、单位、最大截断值）
    3. 计算每个时间点的RUL值
    4. 创建滑动窗口序列
    5. 标准化数据
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        stride: int = 1,
        normalize: bool = True,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        初始化数据处理器
        
        Args:
            sequence_length: 时序窗口长度
            stride: 滑动窗口步长
            normalize: 是否标准化数据
            feature_columns: 特征列名，如果为None则自动检测数值列
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.feature_columns = feature_columns or []
    
    def load_meta_file(self, meta_file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载元文件
        
        Args:
            meta_file_path: 元文件路径
            
        Returns:
            元数据字典，包含rul_config等信息
        """
        meta_file_path = Path(meta_file_path)
        if not meta_file_path.exists():
            raise FileNotFoundError(f"元文件不存在: {meta_file_path}")
        
        with open(meta_file_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        return meta_data
    
    def calculate_rul_labels(
        self,
        data_length: int,
        failure_row_index: int,
        rul_unit: str = "cycle",
        max_rul: int = 200,
        timestamp_column: Optional[str] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        计算RUL标签
        
        Args:
            data_length: 数据长度
            failure_row_index: 失效点行索引（从0开始）
            rul_unit: RUL单位，"cycle"（按采样点）或"second"/"minute"（按时间戳）
            max_rul: RUL最大截断值
            timestamp_column: 时间戳列名（如果rul_unit不是cycle）
            timestamps: 时间戳数组（如果rul_unit不是cycle）
            
        Returns:
            RUL标签数组 (data_length,)
        """
        if rul_unit == "cycle":
            # 按采样点计算：RUL = max(0, failure_row_index - current_index)
            rul_labels = np.maximum(0, failure_row_index - np.arange(data_length))
        elif rul_unit in ["second", "minute"]:
            # 按时间戳计算
            if timestamps is None:
                raise ValueError(f"当rul_unit为{rul_unit}时，必须提供timestamps")
            
            # 获取失效点的时间戳
            failure_timestamp = timestamps[failure_row_index]
            
            # 计算每个时间点到失效点的时间差
            if rul_unit == "second":
                time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 's')
            else:  # minute
                time_diffs = (failure_timestamp - timestamps) / np.timedelta64(1, 'm')
            
            rul_labels = np.maximum(0, time_diffs)
        else:
            raise ValueError(f"不支持的RUL单位: {rul_unit}，支持 'cycle', 'second', 'minute'")
        
        # 截断到最大RUL值
        rul_labels = np.minimum(rul_labels, max_rul)
        
        return rul_labels.astype(np.float32)
    
    def load_data(
        self,
        file_path: Union[str, Path],
        meta_file_path: Optional[Union[str, Path]] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """
        加载数据文件和元文件
        
        Args:
            file_path: 数据文件路径
            meta_file_path: 元文件路径，如果为None则尝试自动查找
            feature_columns: 特征列名，如果为None则自动检测
            
        Returns:
            (数据DataFrame, RUL标签数组, 元数据字典)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        if file_path.suffix.lower() != ".csv":
            raise ValueError("目前仅支持CSV格式数据")
        
        # 加载数据
        data = pd.read_csv(file_path)
        
        # 加载元文件
        if meta_file_path is None:
            # 尝试自动查找元文件（在同一目录下，文件名相同但扩展名为.json）
            meta_file_path = file_path.parent / f"{file_path.stem}.json"
            # 如果不在同一目录，尝试在meta目录下查找
            if not meta_file_path.exists():
                meta_dir = file_path.parent.parent.parent / 'meta' / 'RULPrediction'
                meta_file_path = meta_dir / f"{file_path.stem}.json"
        
        meta_data = self.load_meta_file(meta_file_path)
        rul_config = meta_data.get('rul_config', {})
        
        # 获取RUL配置
        failure_row_index = rul_config.get('failure_row_index')
        if failure_row_index is None:
            raise ValueError("元文件中缺少failure_row_index配置")
        
        rul_unit = rul_config.get('rul_unit', 'cycle')
        max_rul = rul_config.get('max_rul', 200)
        
        # 确定特征列
        if feature_columns is None:
            # 自动检测数值列，排除可能的标签列和时间戳列
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['label', 'Label', 'LABEL', 'class', 'Class', 'CLASS', 
                          'rul', 'RUL', 'Rul',
                          'time', 'Time', 'TIME', 'timestamp', 'Timestamp', 'TIMESTAMP',
                          '时间', '时间戳', 'date', 'Date', 'DATE']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # 如果第一列是时间戳，也排除
            if len(data.columns) > 0:
                first_col = data.columns[0]
                first_col_lower = first_col.lower()
                if any(keyword in first_col_lower for keyword in ['time', 'timestamp', 'date', '时间', '时间戳']):
                    if first_col in numeric_cols:
                        numeric_cols.remove(first_col)
                elif data[first_col].dtype not in [np.number]:
                    if first_col in numeric_cols:
                        numeric_cols.remove(first_col)
            
            if not numeric_cols:
                raise ValueError("数据集中没有可用于训练的数值列")
            
            feature_cols = numeric_cols
            self.feature_columns = feature_cols
        else:
            # 验证指定的列是否存在
            missing = [col for col in feature_columns if col not in data.columns]
            if missing:
                raise ValueError(f"数据缺少以下特征列: {missing}")
            feature_cols = feature_columns
            self.feature_columns = feature_cols
        
        # 提取特征数据
        feature_data = data[feature_cols].copy()
        
        # 计算RUL标签
        timestamps = None
        timestamp_col = None
        if rul_unit in ["second", "minute"]:
            # 查找时间戳列
            for col in data.columns:
                if any(keyword in col.lower() for keyword in ['time', 'timestamp', '时间', '时间戳']):
                    timestamp_col = col
                    break
            
            if timestamp_col:
                timestamps = pd.to_datetime(data[timestamp_col]).values
        
        rul_labels = self.calculate_rul_labels(
            data_length=len(data),
            failure_row_index=failure_row_index,
            rul_unit=rul_unit,
            max_rul=max_rul,
            timestamp_column=timestamp_col,
            timestamps=timestamps,
        )
        
        return feature_data, rul_labels, meta_data
    
    def preprocess(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        预处理数据（标准化）
        
        Args:
            raw_data: 原始数据DataFrame
            
        Returns:
            标准化后的数据数组
        """
        data_array = raw_data.values.astype(np.float32)
        
        if self.normalize and self.scaler is not None:
            # 拟合scaler（如果还没有拟合）
            if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                self.scaler.fit(data_array)
            # 标准化
            data_array = self.scaler.transform(data_array)
        
        return data_array
    
    def create_sequences(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口序列
        
        Args:
            data: 数据数组 (n_samples, n_features)
            labels: RUL标签数组 (n_samples,)
            
        Returns:
            (序列数组, 标签数组)
            序列数组: (n_sequences, seq_len, n_features)
            标签数组: (n_sequences,) - 每个序列对应最后一个时间点的RUL值
        """
        sequences = []
        sequence_labels = []
        
        n_samples, n_features = data.shape
        
        # 创建滑动窗口
        for i in range(0, n_samples - self.sequence_length + 1, self.stride):
            # 提取序列
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            # 使用序列最后一个时间点的RUL值作为标签
            # 或者使用序列中所有时间点的平均RUL值（这里使用最后一个时间点）
            label = labels[i + self.sequence_length - 1]
            sequence_labels.append(label)
        
        sequences = np.array(sequences, dtype=np.float32)
        sequence_labels = np.array(sequence_labels, dtype=np.float32)
        
        return sequences, sequence_labels
    
    def process_pipeline(
        self,
        file_path: Union[str, Path],
        meta_file_path: Optional[Union[str, Path]] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> RegressionData:
        """
        完整的数据处理流水线
        
        Args:
            file_path: 数据文件路径
            meta_file_path: 元文件路径
            feature_columns: 特征列名
            
        Returns:
            RegressionData对象
        """
        # 1. 加载数据
        raw_data, rul_labels, meta_data = self.load_data(
            file_path, meta_file_path, feature_columns
        )
        
        # 2. 预处理
        processed_data = self.preprocess(raw_data)
        
        # 3. 创建序列
        sequences, sequence_labels = self.create_sequences(processed_data, rul_labels)
        
        return RegressionData(sequences, sequence_labels)
    
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
        获取标准化器参数（用于保存和恢复）
        
        Returns:
            包含mean_和scale_的字典
        """
        if self.scaler is None:
            return {}
        return {
            'mean_': self.scaler.mean_,
            'scale_': self.scaler.scale_,
        }
    
    def set_scaler_params(self, params: Dict[str, np.ndarray]):
        """
        设置标准化器参数（用于恢复）
        
        Args:
            params: 包含mean_和scale_的字典
        """
        if self.scaler is not None and 'mean_' in params and 'scale_' in params:
            self.scaler.mean_ = params['mean_']
            self.scaler.scale_ = params['scale_']

