"""
工具模块 - 评估指标
提供异常检测的各种评估指标和可视化功能
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class AnomalyMetrics:
    """
    异常检测评估指标计算器
    
    Features:
    - 分类指标（精确度、召回率、F1分数等）
    - ROC/PR曲线分析
    - 阈值优化
    - 时序异常可视化
    - 详细的评估报告
    """

    def __init__(self):
        self.metrics_cache = {}
        self.optimal_threshold = None

    def compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_scores: np.ndarray = None) -> Dict[str, float]:
        """
        计算基础分类指标
        
        Args:
            y_true: 真实标签 (0: 正常, 1: 异常)
            y_pred: 预测标签
            y_scores: 异常分数（可选，用于AUC计算）
            
        Returns:
            指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # 添加AUC（如果有分数）
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            except ValueError:
                metrics['roc_auc'] = np.nan
        
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0.0
        })
        
        return metrics

    def compute_detection_metrics(self, y_true: np.ndarray, y_scores: np.ndarray,
                                threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        计算异常检测指标
        
        Args:
            y_true: 真实标签
            y_scores: 异常分数
            threshold: 检测阈值（如果为None，将自动寻找最优阈值）
            
        Returns:
            详细的评估指标
        """
        if threshold is None:
            threshold = self.find_optimal_threshold(y_true, y_scores)
            self.optimal_threshold = threshold
        
        # 生成预测标签
        y_pred = (y_scores >= threshold).astype(int)
        
        # 计算基础指标
        basic_metrics = self.compute_basic_metrics(y_true, y_pred, y_scores)
        
        # 计算额外指标
        additional_metrics = {
            'threshold': threshold,
            'anomaly_rate_true': np.mean(y_true),
            'anomaly_rate_pred': np.mean(y_pred),
            'score_statistics': {
                'mean': float(np.mean(y_scores)),
                'std': float(np.std(y_scores)),
                'min': float(np.min(y_scores)),
                'max': float(np.max(y_scores)),
                'median': float(np.median(y_scores)),
                'q25': float(np.percentile(y_scores, 25)),
                'q75': float(np.percentile(y_scores, 75))
            }
        }
        
        # 合并所有指标
        all_metrics = {**basic_metrics, **additional_metrics}
        
        # 缓存结果
        self.metrics_cache = all_metrics
        
        return all_metrics

    def find_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray, 
                              metric: str = 'f1') -> float:
        """
        寻找最优阈值
        
        Args:
            y_true: 真实标签
            y_scores: 异常分数
            metric: 优化指标 ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            最优阈值
        """
        # 尝试不同的阈值
        thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)
        best_threshold = thresholds[0]
        best_score = -1
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            else:
                raise ValueError(f"不支持的优化指标: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold

    def compute_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算ROC曲线数据
        
        Args:
            y_true: 真实标签
            y_scores: 异常分数
            
        Returns:
            ROC曲线数据 {'fpr', 'tpr', 'thresholds', 'auc'}
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc
        }

    def compute_pr_curve(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算Precision-Recall曲线数据
        
        Args:
            y_true: 真实标签
            y_scores: 异常分数
            
        Returns:
            PR曲线数据 {'precision', 'recall', 'thresholds', 'auc'}
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # 计算PR AUC
        pr_auc = np.trapz(precision, recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'auc': pr_auc
        }

    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                      save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_scores: 异常分数
            save_path: 保存路径（可选）
            figsize: 图像大小
        """
        roc_data = self.compute_roc_curve(y_true, y_scores)
        
        plt.figure(figsize=figsize)
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存到: {save_path}")
        
        plt.show()

    def plot_pr_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                     save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)):
        """
        绘制Precision-Recall曲线
        
        Args:
            y_true: 真实标签
            y_scores: 异常分数
            save_path: 保存路径（可选）
            figsize: 图像大小
        """
        pr_data = self.compute_pr_curve(y_true, y_scores)
        baseline = np.mean(y_true)  # 随机分类器的基线
        
        plt.figure(figsize=figsize)
        plt.plot(pr_data['recall'], pr_data['precision'], 
                label=f'PR Curve (AUC = {pr_data["auc"]:.3f})', linewidth=2)
        plt.axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Random Classifier (Baseline = {baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR曲线已保存到: {save_path}")
        
        plt.show()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: Optional[str] = None, figsize: Tuple[int, int] = (6, 5)):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径（可选）
            figsize: 图像大小
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()

    def plot_score_distribution(self, y_true: np.ndarray, y_scores: np.ndarray,
                              threshold: Optional[float] = None,
                              save_path: Optional[str] = None, 
                              figsize: Tuple[int, int] = (10, 6)):
        """
        绘制异常分数分布
        
        Args:
            y_true: 真实标签
            y_scores: 异常分数
            threshold: 检测阈值
            save_path: 保存路径（可选）
            figsize: 图像大小
        """
        plt.figure(figsize=figsize)
        
        # 分别绘制正常和异常数据的分数分布
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]
        
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        
        # 添加阈值线
        if threshold is not None:
            plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, 
                       label=f'Threshold = {threshold:.4f}')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分数分布图已保存到: {save_path}")
        
        plt.show()

    def plot_time_series_with_anomalies(self, data: np.ndarray, y_true: np.ndarray,
                                       y_pred: np.ndarray, feature_idx: int = 0,
                                       save_path: Optional[str] = None,
                                       figsize: Tuple[int, int] = (15, 8)):
        """
        绘制时序数据和异常检测结果
        
        Args:
            data: 时序数据
            y_true: 真实异常标签
            y_pred: 预测异常标签
            feature_idx: 要显示的特征索引
            save_path: 保存路径（可选）
            figsize: 图像大小
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # 提取要显示的特征
        if data.ndim > 1:
            feature_data = data[:, feature_idx]
        else:
            feature_data = data
        
        time_indices = np.arange(len(feature_data))
        
        # 上图：原始时序数据
        axes[0].plot(time_indices, feature_data, 'b-', linewidth=1, label='Time Series')
        
        # 标记真实异常
        true_anomaly_indices = np.where(y_true == 1)[0]
        if len(true_anomaly_indices) > 0:
            axes[0].scatter(true_anomaly_indices, feature_data[true_anomaly_indices], 
                          color='red', s=30, label='True Anomalies', alpha=0.7, zorder=5)
        
        axes[0].set_ylabel('Value')
        axes[0].set_title(f'Time Series with True Anomalies (Feature {feature_idx})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 下图：异常检测结果对比
        axes[1].plot(time_indices, y_true, 'r-', linewidth=2, label='True Labels', alpha=0.7)
        axes[1].plot(time_indices, y_pred, 'g--', linewidth=2, label='Predictions', alpha=0.7)
        
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Anomaly Label')
        axes[1].set_title('Anomaly Detection Results')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"时序异常图已保存到: {save_path}")
        
        plt.show()

    def generate_report(self, y_true: np.ndarray, y_scores: np.ndarray,
                       threshold: Optional[float] = None,
                       save_path: Optional[str] = None) -> str:
        """
        生成详细的评估报告
        
        Args:
            y_true: 真实标签
            y_scores: 异常分数
            threshold: 检测阈值
            save_path: 报告保存路径（可选）
            
        Returns:
            报告文本
        """
        # 计算指标
        metrics = self.compute_detection_metrics(y_true, y_scores, threshold)
        
        # 构建报告
        report_lines = [
            "异常检测评估报告",
            "=" * 50,
            "",
            "数据概况:",
            f"  总样本数: {len(y_true)}",
            f"  真实异常数: {np.sum(y_true)} ({metrics['anomaly_rate_true']:.2%})",
            f"  预测异常数: {np.sum(y_scores >= metrics['threshold'])} ({metrics['anomaly_rate_pred']:.2%})",
            "",
            "检测性能:",
            f"  阈值: {metrics['threshold']:.6f}",
            f"  准确率: {metrics['accuracy']:.4f}",
            f"  精确率: {metrics['precision']:.4f}",
            f"  召回率: {metrics['recall']:.4f}",
            f"  F1分数: {metrics['f1_score']:.4f}",
            f"  特异性: {metrics['specificity']:.4f}",
            "",
            "混淆矩阵:",
            f"  真阴性(TN): {metrics['true_negatives']}",
            f"  假阳性(FP): {metrics['false_positives']}",
            f"  假阴性(FN): {metrics['false_negatives']}",
            f"  真阳性(TP): {metrics['true_positives']}",
            "",
            "异常分数统计:",
            f"  均值: {metrics['score_statistics']['mean']:.6f}",
            f"  标准差: {metrics['score_statistics']['std']:.6f}",
            f"  最小值: {metrics['score_statistics']['min']:.6f}",
            f"  最大值: {metrics['score_statistics']['max']:.6f}",
            f"  中位数: {metrics['score_statistics']['median']:.6f}",
            f"  25%分位数: {metrics['score_statistics']['q25']:.6f}",
            f"  75%分位数: {metrics['score_statistics']['q75']:.6f}",
            ""
        ]
        
        # 添加AUC（如果计算了）
        if not np.isnan(metrics.get('roc_auc', np.nan)):
            report_lines.insert(-1, f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        report_text = "\\n".join(report_lines)
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"评估报告已保存到: {save_path}")
        
        return report_text

    def create_comprehensive_evaluation(self, y_true: np.ndarray, y_scores: np.ndarray,
                                       data: Optional[np.ndarray] = None,
                                       output_dir: str = './evaluation_results',
                                       threshold: Optional[float] = None):
        """
        创建全面的评估结果
        
        Args:
            y_true: 真实标签
            y_scores: 异常分数
            data: 原始时序数据（可选，用于可视化）
            output_dir: 输出目录
            threshold: 检测阈值
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成预测标签
        if threshold is None:
            threshold = self.find_optimal_threshold(y_true, y_scores)
        
        y_pred = (y_scores >= threshold).astype(int)
        
        # 1. 生成评估报告
        report = self.generate_report(y_true, y_scores, threshold, 
                                    os.path.join(output_dir, 'evaluation_report.txt'))
        
        # 2. 绘制ROC曲线
        self.plot_roc_curve(y_true, y_scores, 
                          os.path.join(output_dir, 'roc_curve.png'))
        
        # 3. 绘制PR曲线
        self.plot_pr_curve(y_true, y_scores,
                         os.path.join(output_dir, 'pr_curve.png'))
        
        # 4. 绘制混淆矩阵
        self.plot_confusion_matrix(y_true, y_pred,
                                 os.path.join(output_dir, 'confusion_matrix.png'))
        
        # 5. 绘制分数分布
        self.plot_score_distribution(y_true, y_scores, threshold,
                                   os.path.join(output_dir, 'score_distribution.png'))
        
        # 6. 绘制时序图（如果有数据）
        if data is not None:
            self.plot_time_series_with_anomalies(
                data, y_true, y_pred,
                save_path=os.path.join(output_dir, 'time_series_anomalies.png')
            )
        
        # 7. 保存详细指标到CSV
        metrics = self.compute_detection_metrics(y_true, y_scores, threshold)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
        print(f"\\n完整评估结果已保存到: {output_dir}")
        print("包含文件:")
        print("  - evaluation_report.txt: 详细评估报告")
        print("  - roc_curve.png: ROC曲线")
        print("  - pr_curve.png: Precision-Recall曲线")
        print("  - confusion_matrix.png: 混淆矩阵")
        print("  - score_distribution.png: 异常分数分布")
        if data is not None:
            print("  - time_series_anomalies.png: 时序异常可视化")
        print("  - metrics.csv: 详细指标数据")
        
        return report