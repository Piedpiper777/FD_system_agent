"""RUL预测评估器（PyTorch版）。"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


class RULPredictionEvaluator:
    """RUL预测模型评估器（批量推理 + 反归一化 + 指标计算）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        label_scaler: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if model is None:
            raise ValueError("模型不能为None，请确保传入有效的模型")
        self.model = model
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()
        self.label_scaler = label_scaler

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        if isinstance(device, torch.device):
            target = device
        else:
            device_str = str(device or "cuda:0").strip().lower()
            if device_str in ("gpu", "cuda"):
                device_str = "cuda:0"
            if device_str.startswith("cuda"):
                if torch.cuda.is_available():
                    # 解析首个可用GPU编号
                    for part in device_str.split(","):
                        token = part.strip()
                        if not token:
                            continue
                        if token.startswith("cuda:"):
                            token = token.split(":", 1)[1]
                        if token.isdigit():
                            idx = int(token)
                            if idx < torch.cuda.device_count():
                                return torch.device(f"cuda:{idx}")
                    return torch.device("cuda:0")
                return torch.device("cpu")
            target = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

        if target.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return target

    def evaluate(
        self,
        test_sequences: np.ndarray,
        test_labels_normalized: np.ndarray,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        all_predictions_normalized = []

        with torch.no_grad():
            for i in range(0, len(test_sequences), batch_size):
                batch = test_sequences[i : i + batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)
                preds = self.model(batch_tensor).detach().cpu().numpy().flatten()
                all_predictions_normalized.extend(preds)

        all_predictions_normalized = np.array(all_predictions_normalized)

        if self.label_scaler is not None:
            predictions_reshaped = all_predictions_normalized.reshape(-1, 1)
            test_labels_reshaped = test_labels_normalized.reshape(-1, 1)
            all_predictions = self.label_scaler.inverse_transform(predictions_reshaped).flatten()
            test_labels = self.label_scaler.inverse_transform(test_labels_reshaped).flatten()
        else:
            all_predictions = all_predictions_normalized
            test_labels = test_labels_normalized

        # 推理阶段强制非负
        all_predictions = np.maximum(0, all_predictions)

        metrics = self._compute_metrics(all_predictions, test_labels)
        metrics["predictions"] = all_predictions.tolist()
        metrics["true_labels"] = test_labels.tolist()
        metrics["predictions_normalized"] = all_predictions_normalized.tolist()
        metrics["true_labels_normalized"] = test_labels_normalized.tolist()

        return metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        计算回归评估指标
        
        Args:
            predictions: 预测值
            true_labels: 真实值
            
        Returns:
            评估指标字典
        """
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((predictions - true_labels) ** 2))
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(predictions - true_labels))
        
        # MSE (Mean Squared Error)
        mse = np.mean((predictions - true_labels) ** 2)
        
        # R² (Coefficient of Determination)
        ss_res = np.sum((true_labels - predictions) ** 2)
        ss_tot = np.sum((true_labels - np.mean(true_labels)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # MAPE (Mean Absolute Percentage Error)
        # 避免除以0
        non_zero_mask = true_labels != 0
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((true_labels[non_zero_mask] - predictions[non_zero_mask]) / true_labels[non_zero_mask])) * 100
        else:
            mape = np.inf
        
        # 计算误差分位数
        errors = np.abs(predictions - true_labels)
        error_median = np.median(errors)
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        error_p90 = np.percentile(errors, 90)
        error_p95 = np.percentile(errors, 95)
        error_p99 = np.percentile(errors, 99)
        error_min = np.min(errors)
        error_max = np.max(errors)
        
        # 计算预测值的统计信息
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        pred_min = np.min(predictions)
        pred_max = np.max(predictions)
        pred_median = np.median(predictions)
        
        # 计算真实值的统计信息
        true_mean = np.mean(true_labels)
        true_std = np.std(true_labels)
        true_min = np.min(true_labels)
        true_max = np.max(true_labels)
        true_median = np.median(true_labels)
        
        # 构建结果字典
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse),
            'r2': float(r2),
            'mape': float(mape) if mape != np.inf else None,
            'error_stats': {
                'mean': float(error_mean),
                'std': float(error_std),
                'median': float(error_median),
                'min': float(error_min),
                'max': float(error_max),
                'p90': float(error_p90),
                'p95': float(error_p95),
                'p99': float(error_p99),
            },
            'error_median': float(error_median),  # 兼容字段
            'error_p90': float(error_p90),  # 兼容字段
            'error_p95': float(error_p95),  # 兼容字段
            'error_p99': float(error_p99),  # 兼容字段
            'predictions_stats': {
                'mean': float(pred_mean),
                'std': float(pred_std),
                'median': float(pred_median),
                'min': float(pred_min),
                'max': float(pred_max),
            },
            'true_labels_stats': {
                'mean': float(true_mean),
                'std': float(true_std),
                'median': float(true_median),
                'min': float(true_min),
                'max': float(true_max),
            },
            'num_samples': len(predictions),
        }
        
        return metrics
    
    def save_evaluation_results(
        self,
        metrics: Dict[str, Any],
        save_dir: Union[str, Path],
        task_id: str
    ) -> tuple:
        """
        保存评估结果
        
        Args:
            metrics: 评估指标字典
            save_dir: 保存目录
            task_id: 任务ID
            
        Returns:
            (results_file, summary_file) 元组
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整评估结果（JSON格式）
        results_file = save_dir / f'{task_id}_evaluation_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 保存评估摘要（文本格式）
        summary_file = save_dir / f'{task_id}_evaluation_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RUL预测模型评估结果\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"任务ID: {task_id}\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试样本数: {metrics['num_samples']}\n\n")
            
            f.write("主要性能指标:\n")
            f.write("-" * 70 + "\n")
            f.write(f"RMSE (均方根误差): {metrics['rmse']:.6f}\n")
            f.write(f"MAE (平均绝对误差): {metrics['mae']:.6f}\n")
            f.write(f"MSE (均方误差): {metrics['mse']:.6f}\n")
            f.write(f"R² (决定系数): {metrics['r2']:.6f}\n")
            if metrics['mape'] is not None:
                f.write(f"MAPE (平均绝对百分比误差): {metrics['mape']:.2f}%\n")
            else:
                f.write(f"MAPE (平均绝对百分比误差): N/A\n")
            f.write("\n")
            
            f.write("误差统计:\n")
            f.write("-" * 70 + "\n")
            error_stats = metrics['error_stats']
            f.write(f"均值: {error_stats['mean']:.6f}\n")
            f.write(f"标准差: {error_stats['std']:.6f}\n")
            f.write(f"中位数: {error_stats['median']:.6f}\n")
            f.write(f"最小值: {error_stats['min']:.6f}\n")
            f.write(f"最大值: {error_stats['max']:.6f}\n")
            f.write(f"P90: {error_stats['p90']:.6f}\n")
            f.write(f"P95: {error_stats['p95']:.6f}\n")
            f.write(f"P99: {error_stats['p99']:.6f}\n")
            f.write("\n")
            
            f.write("预测值统计:\n")
            f.write("-" * 70 + "\n")
            pred_stats = metrics['predictions_stats']
            f.write(f"均值: {pred_stats['mean']:.6f}\n")
            f.write(f"标准差: {pred_stats['std']:.6f}\n")
            f.write(f"中位数: {pred_stats['median']:.6f}\n")
            f.write(f"最小值: {pred_stats['min']:.6f}\n")
            f.write(f"最大值: {pred_stats['max']:.6f}\n")
            f.write("\n")
            
            f.write("真实值统计:\n")
            f.write("-" * 70 + "\n")
            true_stats = metrics['true_labels_stats']
            f.write(f"均值: {true_stats['mean']:.6f}\n")
            f.write(f"标准差: {true_stats['std']:.6f}\n")
            f.write(f"中位数: {true_stats['median']:.6f}\n")
            f.write(f"最小值: {true_stats['min']:.6f}\n")
            f.write(f"最大值: {true_stats['max']:.6f}\n")
            f.write("\n")
        
        return results_file, summary_file

