"""
LSTM自编码器 - 阈值计算器
利用重构误差分布确定异常阈值
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from datetime import datetime

import numpy as np


class ThresholdCalculator:
    """根据重构误差设定阈值"""

    def __init__(self, method: str = "percentile", residual_metric: str = "rmse"):
        self.method = method
        self.residual_metric = residual_metric
        self.threshold: Optional[float] = None
        self.stats: Dict[str, Any] = {"mean": None, "std": None, "percentiles": {}}

    def _reduce_residual(self, residuals: np.ndarray) -> np.ndarray:
        if residuals.ndim == 3:
            residuals = residuals.reshape(residuals.shape[0], -1)
        if self.residual_metric == "l1":
            return np.sum(np.abs(residuals), axis=1)
        if self.residual_metric == "max":
            return np.max(np.abs(residuals), axis=1)
        return np.sqrt(np.mean(residuals ** 2, axis=1))

    def fit(self, reconstructions: np.ndarray, originals: np.ndarray, percentile: float = 99.0,
            contamination: Optional[float] = None) -> float:
        residuals = originals - reconstructions
        scores = self._reduce_residual(residuals)
        self.stats["mean"] = float(np.mean(scores))
        self.stats["std"] = float(np.std(scores))
        for p in [50, 75, 90, 95, 99, 99.5]:
            self.stats["percentiles"][p] = float(np.percentile(scores, p))

        if self.method == "percentile":
            self.threshold = float(np.percentile(scores, percentile))
        elif self.method == "3sigma":
            self.threshold = float(self.stats["mean"] + 3 * self.stats["std"])
        elif self.method == "contamination":
            if contamination is None:
                raise ValueError("contamination method requires contamination value")
            target_percentile = 100 * (1 - contamination)
            self.threshold = float(np.percentile(scores, target_percentile))
        else:
            raise ValueError(f"Unsupported threshold method: {self.method}")
        return self.threshold

    def detect(self, reconstruction: np.ndarray, original: np.ndarray) -> Tuple[float, bool]:
        if self.threshold is None:
            raise ValueError("Threshold not fitted. Call fit() first.")
        residual = original - reconstruction
        score = float(self._reduce_residual(residual.reshape(1, -1))[0])
        return score, score > self.threshold

    def save(self, file_path: Union[str, Path]):
        if self.threshold is None:
            return
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存NPZ格式（主要格式，性能优先）
        npz_path = file_path.with_suffix('.npz')
        np.savez(
            npz_path,
            threshold=self.threshold,
            method=self.method,
            residual_metric=self.residual_metric,
            stats=self.stats,
        )
        
        # 同时保存JSON格式（调试和展示友好）
        json_path = file_path.with_suffix('.json')
        threshold_info = {
            'threshold': float(self.threshold),
            'method': str(self.method),
            'residual_metric': str(self.residual_metric),
            'stats': self._serialize_stats(self.stats),
            'created_at': datetime.now().isoformat(),
            'file_version': '1.0'
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(threshold_info, f, indent=2, ensure_ascii=False)
        
        print(f"💾 阈值已保存: {npz_path} (主要格式)")
        print(f"📄 阈值已保存: {json_path} (调试格式)")
    
    def _serialize_stats(self, stats):
        """序列化统计信息为JSON兼容格式"""
        if stats is None:
            return {}
        if isinstance(stats, dict):
            return {key: self._serialize_stats(value) for key, value in stats.items()}
        if isinstance(stats, (list, tuple)):
            return [self._serialize_stats(value) for value in stats]
        if hasattr(stats, 'item'):
            try:
                return stats.item()
            except Exception:
                return float(stats) if hasattr(stats, '__float__') else str(stats)
        return stats

    def load(self, file_path: Union[str, Path]):
        data = np.load(file_path, allow_pickle=True)
        self.threshold = float(data["threshold"])
        self.method = str(data["method"])
        self.residual_metric = str(data["residual_metric"])
        self.stats = data["stats"].item()
