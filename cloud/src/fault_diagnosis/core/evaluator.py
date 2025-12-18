"""
故障诊断评估器
用于评估分类模型的性能（准确率、精确率、召回率、F1等）
"""

import numpy as np
import mindspore as ms
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class FaultDiagnosisEvaluator:
    """故障诊断模型评估器"""
    
    def __init__(self, model: ms.nn.Cell):
        """
        初始化评估器
        
        Args:
            model: 训练好的MindSpore模型
        """
        if model is None:
            raise ValueError("模型不能为None，请确保传入有效的MindSpore模型")
        self.model = model
        self.model.set_train(False)  # 设置为评估模式
        
    def evaluate(
        self,
        test_loader,
        num_classes: int = 3,
    ) -> Dict[str, Any]:
        """
        评估模型在测试集上的性能
        
        Args:
            test_loader: 测试数据加载器
            num_classes: 分类数量
            
        Returns:
            评估结果字典
        """
        all_predictions = []
        all_labels = []
        
        # 遍历测试集进行预测
        for batch in test_loader:
            if isinstance(batch, dict):
                sequences = batch['sequences']
                labels = batch['labels']
            else:
                sequences, labels = batch
            
            # 模型预测
            logits = self.model(sequences)
            
            # 获取预测类别
            predictions = ms.ops.argmax(logits, dim=1)
            
            # 转换为numpy
            all_predictions.extend(predictions.asnumpy().tolist())
            all_labels.extend(labels.asnumpy().tolist())
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # 计算评估指标
        metrics = self._compute_metrics(all_predictions, all_labels, num_classes)
        
        # 添加预测结果
        metrics['predictions'] = all_predictions.tolist()
        metrics['true_labels'] = all_labels.tolist()
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        num_classes: int
    ) -> Dict[str, Any]:
        """
        计算评估指标
        
        Args:
            predictions: 预测标签
            true_labels: 真实标签
            num_classes: 分类数量
            
        Returns:
            评估指标字典
        """
        # 基本指标
        accuracy = accuracy_score(true_labels, predictions)
        
        # 多分类指标（使用macro平均）
        precision = precision_score(
            true_labels, predictions,
            average='macro',
            zero_division=0
        )
        recall = recall_score(
            true_labels, predictions,
            average='macro',
            zero_division=0
        )
        f1 = f1_score(
            true_labels, predictions,
            average='macro',
            zero_division=0
        )
        
        # 每个类别的指标
        precision_per_class = precision_score(
            true_labels, predictions,
            average=None,
            zero_division=0
        )
        recall_per_class = recall_score(
            true_labels, predictions,
            average=None,
            zero_division=0
        )
        f1_per_class = f1_score(
            true_labels, predictions,
            average=None,
            zero_division=0
        )
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        
        # 分类报告
        report = classification_report(
            true_labels, predictions,
            output_dict=True,
            zero_division=0
        )
        
        # 构建结果字典
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision),
                'recall_macro': float(recall),
                'f1_macro': float(f1),
            },
            'per_class': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist(),
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'num_samples': len(predictions),
            'num_classes': num_classes,
        }
        
        return metrics
    
    def save_evaluation_results(
        self,
        metrics: Dict[str, Any],
        save_dir: Union[str, Path],
        task_id: str
    ):
        """
        保存评估结果
        
        Args:
            metrics: 评估指标字典
            save_dir: 保存目录
            task_id: 任务ID
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
            f.write("故障诊断模型评估结果\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"任务ID: {task_id}\n")
            f.write(f"测试样本数: {metrics['num_samples']}\n")
            f.write(f"分类数量: {metrics['num_classes']}\n\n")
            
            f.write("整体性能指标:\n")
            f.write("-" * 70 + "\n")
            overall = metrics['overall']
            f.write(f"准确率 (Accuracy): {overall['accuracy']:.4f}\n")
            f.write(f"精确率 (Precision, Macro): {overall['precision_macro']:.4f}\n")
            f.write(f"召回率 (Recall, Macro): {overall['recall_macro']:.4f}\n")
            f.write(f"F1分数 (F1, Macro): {overall['f1_macro']:.4f}\n\n")
            
            f.write("各类别性能指标:\n")
            f.write("-" * 70 + "\n")
            per_class = metrics['per_class']
            for i in range(metrics['num_classes']):
                f.write(f"类别 {i}:\n")
                f.write(f"  精确率: {per_class['precision'][i]:.4f}\n")
                f.write(f"  召回率: {per_class['recall'][i]:.4f}\n")
                f.write(f"  F1分数: {per_class['f1'][i]:.4f}\n")
            f.write("\n")
            
            f.write("混淆矩阵:\n")
            f.write("-" * 70 + "\n")
            cm = metrics['confusion_matrix']
            f.write("真实标签 \\ 预测标签")
            for j in range(len(cm[0])):
                f.write(f"  {j:>6}")
            f.write("\n")
            for i, row in enumerate(cm):
                f.write(f"类别 {i:>2}")
                for val in row:
                    f.write(f"  {val:>6}")
                f.write("\n")
            f.write("\n")
            
            f.write("详细分类报告:\n")
            f.write("-" * 70 + "\n")
            report = metrics['classification_report']
            for key, value in report.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        if isinstance(v, float):
                            f.write(f"  {k}: {v:.4f}\n")
                        else:
                            f.write(f"  {k}: {v}\n")
                else:
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
        
        return results_file, summary_file

