"""
故障诊断 LSTM - 训练器
负责分类任务的训练循环与指标记录

重构说明：从 MindSpore 迁移到 PyTorch
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import numpy as np


class Trainer:
    """LSTM 分类器训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float = 5.0,
        device: Optional[torch.device] = None,
    ):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            learning_rate: 学习率
            weight_decay: 权重衰减
            clip_grad_norm: 梯度裁剪阈值
            device: 计算设备 (cuda/cpu)
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        
        # 优化器
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 损失函数：交叉熵损失
        self.criterion = CrossEntropyLoss()
        
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
    
    def compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        计算准确率
        
        Args:
            logits: 模型输出 (batch_size, num_classes)
            labels: 真实标签 (batch_size,)
            
        Returns:
            准确率
        """
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).float()
        accuracy = correct.mean()
        return float(accuracy.item())
    
    def train_step(self, batch_data) -> tuple:
        """
        训练一步
        
        Args:
            batch_data: 批次数据，可以是元组 (sequences, labels) 或字典
            
        Returns:
            (loss, accuracy)
        """
        if isinstance(batch_data, dict):
            sequences = batch_data["sequences"]
            labels = batch_data["labels"]
        else:
            sequences, labels = batch_data
        
        # 将数据移动到设备
        sequences = sequences.to(self.device)
        labels = labels.to(self.device)
        
        # 确保数据类型正确
        if sequences.dtype != torch.float32:
            sequences = sequences.float()
        if labels.dtype != torch.long:
            labels = labels.long()
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        logits = self.model(sequences)
        loss = self.criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.clip_grad_norm
            )
        
        # 优化器更新
        self.optimizer.step()
        
        # 计算准确率（不需要梯度）
        with torch.no_grad():
            accuracy = self.compute_accuracy(logits, labels)
        
        return float(loss.item()), accuracy
    
    def compute_loss_and_accuracy(self, batch_data) -> tuple:
        """
        计算损失和准确率（用于验证）
        
        Args:
            batch_data: 批次数据
            
        Returns:
            (loss, accuracy)
        """
        if isinstance(batch_data, dict):
            sequences = batch_data["sequences"]
            labels = batch_data["labels"]
        else:
            sequences, labels = batch_data
        
        # 将数据移动到设备
        sequences = sequences.to(self.device)
        labels = labels.to(self.device)
        
        # 确保数据类型正确
        if sequences.dtype != torch.float32:
            sequences = sequences.float()
        if labels.dtype != torch.long:
            labels = labels.long()
        
        with torch.no_grad():
            logits = self.model(sequences)
            loss = self.criterion(logits, labels)
            accuracy = self.compute_accuracy(logits, labels)
        
        return float(loss.item()), accuracy
    
    def train_epoch(self, train_loader, epoch_idx: Optional[int] = None) -> tuple:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch_idx: epoch索引（可选）
            
        Returns:
            (平均损失, 平均准确率)
        """
        self.model.train()
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
        """
        验证
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            (平均损失, 平均准确率)
        """
        self.model.eval()
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
        """
        检查早停条件
        
        Args:
            val_loss: 验证损失
            val_acc: 验证准确率
            patience: 耐心值
            mode: 早停模式，"loss"（基于损失）或 "acc"（基于准确率）
            
        Returns:
            是否应该早停
        """
        if mode == "loss":
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                return False
        else:  # mode == "acc"
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
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            val_loader: 验证数据加载器（可选）
            patience: 早停耐心值（可选）
            early_stop_mode: 早停模式，"loss" 或 "acc"
        """
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
                    print(f"⏹️ 早停于第 {epoch+1} 轮")
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
        torch.save(self.model.state_dict(), str(save_path))
    
    def load_model(self, load_path: Union[str, Path]):
        """加载模型"""
        load_path = Path(load_path)
        state_dict = torch.load(str(load_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        return self.model
    
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
