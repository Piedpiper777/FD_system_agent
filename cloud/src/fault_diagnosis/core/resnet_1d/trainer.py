"""
故障诊断 ResNet 1D - 训练器
负责分类任务的训练循环与指标记录
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import mindspore as ms
import mindspore.ops as ops
from mindspore.nn import Adam, CrossEntropyLoss
import numpy as np


class Trainer:
    """ResNet 1D 分类器训练器"""
    
    def __init__(
        self,
        model: ms.nn.Cell,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float = 5.0,
    ):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            learning_rate: 学习率
            weight_decay: 权重衰减
            clip_grad_norm: 梯度裁剪阈值
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        
        # 优化器
        self.optimizer = Adam(
            params=self.model.trainable_params(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 损失函数：交叉熵损失
        self.criterion = CrossEntropyLoss()
        
        # 梯度计算函数
        self.grad_fn = ms.value_and_grad(
            self.forward_fn, None, self.optimizer.parameters
        )
        
        # 训练指标
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.metrics = {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "epochs_trained": 0,
        }
    
    def forward_fn(self, input_seq, labels):
        """前向传播函数"""
        logits = self.model(input_seq)
        loss = self.criterion(logits, labels)
        return loss
    
    def _process_grads(self, grads):
        """处理梯度（梯度裁剪）"""
        if self.clip_grad_norm > 0:
            total_norm = 0.0
            for grad in grads:
                if grad is not None:
                    total_norm += ops.norm(grad) ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > self.clip_grad_norm:
                clip_coef = self.clip_grad_norm / (total_norm + 1e-6)
                grads = [grad * clip_coef if grad is not None else grad for grad in grads]
        
        return tuple(grads)
    
    def compute_accuracy(self, logits: ms.Tensor, labels: ms.Tensor) -> float:
        """计算准确率"""
        predictions = ops.argmax(logits, dim=1)
        correct = ops.equal(predictions, labels)
        accuracy = correct.astype(ms.float32).mean()
        return float(accuracy)
    
    def train_step(self, batch_data):
        """训练一步"""
        if isinstance(batch_data, dict):
            sequences = batch_data["sequences"]
            labels = batch_data["labels"]
        else:
            sequences, labels = batch_data
        
        loss, grads = self.grad_fn(sequences, labels)
        grads = self._process_grads(grads)
        self.optimizer(grads)
        
        with ms._no_grad():
            logits = self.model(sequences)
            accuracy = self.compute_accuracy(logits, labels)
        
        return float(loss), accuracy
    
    def compute_loss_and_accuracy(self, batch_data):
        """计算损失和准确率（用于验证）"""
        if isinstance(batch_data, dict):
            sequences = batch_data["sequences"]
            labels = batch_data["labels"]
        else:
            sequences, labels = batch_data
        
        logits = self.model(sequences)
        loss = self.criterion(logits, labels)
        accuracy = self.compute_accuracy(logits, labels)
        
        return float(loss), accuracy
    
    def train_epoch(self, train_loader, epoch_idx: Optional[int] = None) -> tuple:
        """训练一个epoch"""
        self.model.set_train(True)
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        
        for batch in train_loader:
            loss, acc = self.train_step(batch)
            total_loss += loss
            total_acc += acc
            count += 1
        
        avg_loss = total_loss / max(count, 1)
        avg_acc = total_acc / max(count, 1)
        
        self.metrics["train_losses"].append(avg_loss)
        self.metrics["train_accuracies"].append(avg_acc)
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader) -> tuple:
        """验证"""
        self.model.set_train(False)
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        
        for batch in val_loader:
            loss, acc = self.compute_loss_and_accuracy(batch)
            total_loss += loss
            total_acc += acc
            count += 1
        
        avg_loss = total_loss / max(count, 1)
        avg_acc = total_acc / max(count, 1)
        
        self.metrics["val_losses"].append(avg_loss)
        self.metrics["val_accuracies"].append(avg_acc)
        
        return avg_loss, avg_acc
    
    def check_early_stopping(
        self,
        val_loss: float,
        val_acc: float,
        patience: int,
        mode: str = "loss"
    ) -> bool:
        """检查早停条件"""
        if mode == "loss":
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                return False
        else:
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                return False
        
        self.patience_counter += 1
        return self.patience_counter >= patience
    
    def train(
        self,
        train_loader,
        num_epochs: int = 50,
        val_loader=None,
        patience: Optional[int] = None,
        early_stop_mode: str = "loss",
    ):
        """训练模型"""
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}"
                )
                
                if patience and self.check_early_stopping(
                    val_loss, val_acc, patience, early_stop_mode
                ):
                    print(f"早停于第 {epoch+1} 轮")
                    break
            else:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}"
                )
        
        self.metrics["epochs_trained"] = epoch + 1
        return self.model
    
    def save_model(self, save_path: Union[str, Path]):
        """保存模型"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ms.save_checkpoint(self.model, str(save_path))
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """获取训练指标"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """重置训练指标"""
        self.metrics = {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "epochs_trained": 0,
        }
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0

