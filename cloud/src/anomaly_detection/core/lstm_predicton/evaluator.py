"""
LSTM预测异常检测模块 - 评估器
负责模型性能评估和结果可视化
"""

import numpy as np
from typing import Optional, Dict, Any, Union
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt


class Evaluator:
    """
    模型评估器

    专门用于评估模型性能：
    - 计算各种评估指标
    - 生成性能报告
    - 可视化评估结果
    """

    def __init__(self):
        """
        初始化评估器
        """
        # 评估结果存储
        self.evaluation_results = {
            'predictions': [],
            'actuals': [],
            'true_labels': []
        }

        # 性能指标
        self.metrics = {}

        print(f"✅ 评估器初始化完成")

    def evaluate(self, predictions: np.ndarray, actuals: np.ndarray,
                 true_labels: np.ndarray, anomaly_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        评估模型性能

        Args:
            predictions: 模型预测值
            actuals: 实际值
            true_labels: 真实异常标签
            anomaly_scores: 异常分数（可选，用于AUC计算）

        Returns:
            评估指标字典
        """
        # 存储评估结果
        self.evaluation_results['predictions'].extend(predictions.tolist())
        self.evaluation_results['actuals'].extend(actuals.tolist())
        self.evaluation_results['true_labels'].extend(true_labels.tolist())

        # 计算评估指标
        self.metrics = self._compute_metrics(true_labels, anomaly_scores)

        return self.metrics.copy()

    def _compute_metrics(self, true_labels: np.ndarray,
                        anomaly_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        计算详细的评估指标

        Args:
            true_labels: 真实异常标签
            anomaly_scores: 异常分数

        Returns:
            指标字典
        """
        # 基本统计
        total_samples = len(true_labels)
        n_anomalies = np.sum(true_labels)
        anomaly_ratio = n_anomalies / total_samples if total_samples > 0 else 0

        # 如果没有异常分数，假设预测标签就是真实标签（用于基本评估）
        if anomaly_scores is None:
            pred_labels = true_labels  # 自评估模式
            auc = None
        else:
            # 使用异常分数作为预测标签（>0.5为异常）
            pred_labels = (anomaly_scores > 0.5).astype(int)

            # 计算AUC
            auc = None
            try:
                if len(np.unique(true_labels)) > 1:
                    auc = roc_auc_score(true_labels, anomaly_scores)
            except:
                pass

        # 基础分类指标
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division=0
        )

        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

        # 异常检测特有指标
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # 工业应用指标
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        metrics = {
            'total_samples': int(total_samples),
            'n_anomalies': int(n_anomalies),
            'anomaly_ratio': float(anomaly_ratio),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'false_positive_rate': float(false_positive_rate),
            'false_negative_rate': float(false_negative_rate),
            'auc': float(auc) if auc is not None else None,
            'confusion_matrix': {
                'tp': int(tp), 'fp': int(fp),
                'tn': int(tn), 'fn': int(fn)
            }
        }

        return metrics

    def plot_evaluation_results(self, save_path: Optional[str] = None,
                               show_plot: bool = True):
        """
        可视化评估结果

        Args:
            save_path: 保存路径
            show_plot: 是否显示图像
        """
        if not self.evaluation_results['predictions']:
            print("⚠️ 没有评估结果可供可视化")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 预测误差分布
        predictions = np.array(self.evaluation_results['predictions'])
        actuals = np.array(self.evaluation_results['actuals'])
        true_labels = np.array(self.evaluation_results['true_labels'])

        # 计算预测误差
        errors = np.abs(predictions - actuals)
        if errors.ndim > 1:
            errors = np.mean(errors, axis=1)  # 多特征取平均

        axes[0, 0].hist(errors[true_labels == 0], alpha=0.7, label='正常', bins=50, color='blue')
        axes[0, 0].hist(errors[true_labels == 1], alpha=0.7, label='异常', bins=50, color='red')
        axes[0, 0].set_xlabel('预测误差')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('预测误差分布')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 预测值vs实际值散点图
        if predictions.ndim > 1:
            pred_flat = predictions.flatten()
            actual_flat = actuals.flatten()
        else:
            pred_flat = predictions
            actual_flat = actuals

        axes[0, 1].scatter(actual_flat[true_labels == 0], pred_flat[true_labels == 0],
                          alpha=0.6, label='正常', color='blue', s=20)
        axes[0, 1].scatter(actual_flat[true_labels == 1], pred_flat[true_labels == 1],
                          alpha=0.6, label='异常', color='red', s=20)
        # 添加对角线
        min_val = min(np.min(actual_flat), np.min(pred_flat))
        max_val = max(np.max(actual_flat), np.max(pred_flat))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='理想预测')
        axes[0, 1].set_xlabel('实际值')
        axes[0, 1].set_ylabel('预测值')
        axes[0, 1].set_title('预测值 vs 实际值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 残差分析（如果有异常分数）
        # 这里简化处理，暂时用预测误差作为残差
        axes[1, 0].scatter(range(len(errors)), errors, alpha=0.6,
                          c=true_labels, cmap='coolwarm', s=20)
        axes[1, 0].set_xlabel('样本索引')
        axes[1, 0].set_ylabel('残差')
        axes[1, 0].set_title('残差分布')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 性能指标摘要
        axes[1, 1].axis('off')
        metrics_text = f"""
        性能指标摘要:

        样本总数: {self.metrics.get('total_samples', 0)}
        异常样本数: {self.metrics.get('n_anomalies', 0)}
        异常比例: {self.metrics.get('anomaly_ratio', 0):.3f}

        准确率: {self.metrics.get('accuracy', 0):.3f}
        精确率: {self.metrics.get('precision', 0):.3f}
        召回率: {self.metrics.get('recall', 0):.3f}
        F1分数: {self.metrics.get('f1_score', 0):.3f}
        AUC: {self.metrics.get('auc', 'N/A')}

        特异性: {self.metrics.get('specificity', 0):.3f}
        灵敏度: {self.metrics.get('sensitivity', 0):.3f}
        假正率: {self.metrics.get('false_positive_rate', 0):.3f}
        假负率: {self.metrics.get('false_negative_rate', 0):.3f}
        """

        axes[1, 1].text(0.1, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 评估结果图表已保存: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def get_evaluation_report(self) -> Dict[str, Any]:
        """
        获取完整的评估报告

        Returns:
            评估报告字典
        """
        report = {
            'metrics': self.metrics.copy(),
            'evaluation_summary': {
                'total_samples': len(self.evaluation_results['predictions']),
                'total_anomalies': sum(self.evaluation_results['true_labels'])
            }
        }
        return report

    def reset_evaluation(self):
        """
        重置评估结果
        """
        self.evaluation_results = {
            'predictions': [],
            'actuals': [],
            'true_labels': []
        }
        self.metrics = {}


def get_default_evaluator_config(evaluator_type: str = 'evaluator') -> Dict[str, Any]:
    """
    获取默认评估器配置

    Args:
        evaluator_type: 评估器类型

    Returns:
        默认配置字典
    """
    base_config = {
        'evaluator_type': evaluator_type
    }

    return base_config


def create_evaluator_from_config(config: Dict[str, Any]) -> Evaluator:
    """
    从配置创建评估器

    Args:
        config: 配置字典

    Returns:
        评估器实例
    """
    evaluator_type = config.get('evaluator_type', 'evaluator')

    if evaluator_type == 'evaluator':
        return Evaluator()
    else:
        raise ValueError(f"不支持的评估器类型: {evaluator_type}. 支持的类型: ['evaluator']")