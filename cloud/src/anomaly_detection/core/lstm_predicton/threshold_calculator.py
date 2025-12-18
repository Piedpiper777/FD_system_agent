"""
LSTM预测异常检测模块 - 阈值计算器
基于正常数据的残差分布计算异常判定阈值
"""

import numpy as np
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
import json


class ThresholdCalculator:
    """
    异常检测阈值计算器

    基于正常数据的残差分布计算异常判定阈值：
    - 支持多种残差计算方法（L2范数、L1范数等）
    - 支持不同的阈值计算策略（百分位数、统计方法等）
    - 提供阈值保存和加载功能
    """

    def __init__(self, residual_method: str = 'l2_norm'):
        """
        初始化阈值计算器

        Args:
            residual_method: 残差计算方法
                - 'l2_norm': L2范数（默认）
                - 'l1_norm': L1范数
                - 'max_abs': 最大绝对误差
                - 'mean_abs': 平均绝对误差
                - 'rmse': 均方根误差
                - 'relative': 相对误差
        """
        self.residual_method = residual_method
        self.threshold = None
        self.residual_stats = {
            'mean': None,
            'std': None,
            'percentiles': {}
        }

        # 验证残差计算方法
        valid_methods = ['l2_norm', 'l1_norm', 'max_abs', 'mean_abs', 'rmse', 'relative']
        if self.residual_method not in valid_methods:
            raise ValueError(f"不支持的残差计算方法: {self.residual_method}. "
                           f"支持的方法: {valid_methods}")

        print(f"✅ 阈值计算器初始化完成")
        print(f"  - 残差计算方法: {residual_method}")

    def calculate_residuals(self, predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        """
        计算预测残差

        Args:
            predictions: 预测值 (n_samples, n_features)
            actuals: 实际值 (n_samples, n_features)

        Returns:
            残差数组 (n_samples,)
        """
        residuals = actuals - predictions

        if self.residual_method == 'l2_norm':
            # L2范数
            if residuals.ndim > 1:
                l2_residuals = np.sqrt(np.sum(residuals ** 2, axis=1))
                return l2_residuals
            else:
                return np.abs(residuals)

        elif self.residual_method == 'l1_norm':
            # L1范数
            if residuals.ndim > 1:
                l1_residuals = np.sum(np.abs(residuals), axis=1)
                return l1_residuals
            else:
                return np.abs(residuals)

        elif self.residual_method == 'max_abs':
            # 最大绝对误差
            if residuals.ndim > 1:
                max_abs_residuals = np.max(np.abs(residuals), axis=1)
                return max_abs_residuals
            else:
                return np.abs(residuals)

        elif self.residual_method == 'mean_abs':
            # 平均绝对误差
            if residuals.ndim > 1:
                mean_abs_residuals = np.mean(np.abs(residuals), axis=1)
                return mean_abs_residuals
            else:
                return np.abs(residuals)

        elif self.residual_method == 'rmse':
            # 均方根误差
            if residuals.ndim > 1:
                rmse_residuals = np.sqrt(np.mean(residuals ** 2, axis=1))
                return rmse_residuals
            else:
                return np.abs(residuals)

        elif self.residual_method == 'relative':
            # 相对误差
            epsilon = 1e-8
            relative_residuals = np.abs(residuals) / (np.abs(actuals) + epsilon)
            if relative_residuals.ndim > 1:
                mean_relative_residuals = np.mean(relative_residuals, axis=1)
                return mean_relative_residuals
            else:
                return relative_residuals

        else:
            raise ValueError(f"不支持的残差计算方法: {self.residual_method}")

    def fit_threshold(self, predictions: np.ndarray, actuals: np.ndarray,
                     method: str = 'percentile', percentile: float = 99.0,
                     contamination: Optional[float] = None) -> float:
        """
        基于正常数据拟合异常检测阈值

        Args:
            predictions: 预测值
            actuals: 实际值
            method: 阈值计算方法 ('percentile', '3sigma', 'contamination')
            percentile: 百分位数阈值
            contamination: 异常比例（用于contamination方法）

        Returns:
            计算得到的阈值
        """
        # 计算残差
        residuals = self.calculate_residuals(predictions, actuals)

        # 更新残差统计信息
        self.residual_stats['mean'] = float(np.mean(residuals))
        self.residual_stats['std'] = float(np.std(residuals))

        # 计算百分位数
        for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
            self.residual_stats['percentiles'][p] = float(np.percentile(residuals, p))

        # 计算阈值
        if method == 'percentile':
            self.threshold = float(np.percentile(residuals, percentile))
        elif method == '3sigma':
            self.threshold = float(self.residual_stats['mean'] + 3 * self.residual_stats['std'])
        elif method == 'contamination':
            if contamination is None:
                raise ValueError("contamination方法需要指定contamination参数")
            threshold_percentile = 100 * (1 - contamination)
            self.threshold = float(np.percentile(residuals, threshold_percentile))
        else:
            raise ValueError(f"不支持的阈值计算方法: {method}")

        print(f"🎯 异常检测阈值计算完成")
        print(f"  - 计算方法: {method}")
        print(f"  - 阈值: {self.threshold:.6f}")
        print(f"  - 残差统计: 均值={self.residual_stats['mean']:.6f}, 标准差={self.residual_stats['std']:.6f}")

        return self.threshold

    def detect_anomaly(self, prediction: np.ndarray, actual: np.ndarray) -> Tuple[float, bool]:
        """
        检测单个样本是否异常

        Args:
            prediction: 预测值 (n_features,)
            actual: 实际值 (n_features,)

        Returns:
            (residual_score, is_anomaly)
        """
        if self.threshold is None:
            raise ValueError("请先调用fit_threshold()计算阈值")

        # 计算残差分数
        residual = self.calculate_residuals(
            prediction.reshape(1, -1),
            actual.reshape(1, -1)
        )[0]

        # 判断是否异常
        is_anomaly = residual > self.threshold

        return residual, is_anomaly

    def get_threshold_info(self) -> Dict[str, Any]:
        """
        获取阈值相关信息

        Returns:
            阈值信息字典
        """
        return {
            'threshold': self.threshold,
            'residual_method': self.residual_method,
            'residual_stats': self.residual_stats.copy()
        }

    def save_threshold(self, file_path: Union[str, Path]):
        """
        保存阈值到文件（混合格式）

        Args:
            file_path: 保存路径
        """
        threshold_info = self.get_threshold_info()
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存NPZ格式（主要格式，性能优先）
        npz_path = file_path.with_suffix('.npz')
        np.savez(npz_path, **threshold_info)

        # 同时保存JSON格式（调试和展示友好）
        json_path = file_path.with_suffix('.json')
        json_info = threshold_info.copy()
        json_info['created_at'] = datetime.now().isoformat()
        json_info['file_version'] = '1.0'
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_info, f, indent=2, ensure_ascii=False)

        print(f"💾 阈值已保存: {npz_path} (主要格式)")
        print(f"📄 阈值已保存: {json_path} (调试格式)")

    def load_threshold(self, file_path: Union[str, Path]):
        """
        从文件加载阈值（优先NPZ格式）

        Args:
            file_path: 文件路径
        """
        file_path = Path(file_path)
        
        # 优先尝试加载NPZ格式
        npz_path = file_path.with_suffix('.npz')
        json_path = file_path.with_suffix('.json')
        
        if npz_path.exists():
            # 加载numpy格式
            data = np.load(npz_path, allow_pickle=True)
            threshold_info = {key: data[key].item() if data[key].ndim == 0 else data[key]
                            for key in data.files}
            print(f"📂 阈值已加载: {npz_path} (NPZ格式)")
        elif json_path.exists():
            # 回退到JSON格式
            with open(json_path, 'r', encoding='utf-8') as f:
                threshold_info = json.load(f)
            print(f"📂 阈值已加载: {json_path} (JSON格式)")
        else:
            raise FileNotFoundError(f"阈值文件不存在: {npz_path} 或 {json_path}")

        self.threshold = threshold_info['threshold']
        self.residual_method = threshold_info['residual_method']
        self.residual_stats = threshold_info['residual_stats']
        print(f"  - 阈值: {self.threshold:.6f}")
        print(f"  - 残差方法: {self.residual_method}")

    def reset(self):
        """重置计算器状态"""
        self.threshold = None
        self.residual_stats = {
            'mean': None,
            'std': None,
            'percentiles': {}
        }