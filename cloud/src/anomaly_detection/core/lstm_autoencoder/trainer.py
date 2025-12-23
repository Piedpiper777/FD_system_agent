"""
LSTM自编码器 - 训练器
负责重构任务的训练循环与指标记录


"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss


class Trainer:
    """标准LSTM Autoencoder训练循环"""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float = 5.0,
        device: Optional[torch.device] = None,
    ):
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm

        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = MSELoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.metrics = {
            "train_losses": [],
            "val_losses": [],
            "epochs_trained": 0,
        }

    def train_step(self, batch_data) -> float:
        """执行一步训练"""
        input_seq, target_seq = batch_data
        
        # 将数据移动到设备
        input_seq = input_seq.to(self.device)
        target_seq = target_seq.to(self.device)
        
        # 确保数据类型正确
        if input_seq.dtype != torch.float32:
            input_seq = input_seq.float()
        if target_seq.dtype != torch.float32:
            target_seq = target_seq.float()
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        reconstruction = self.model(input_seq)
        loss = self.criterion(reconstruction, target_seq)
        
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
        
        return float(loss.item())

    def compute_loss(self, batch_data) -> float:
        """计算损失（不更新参数）"""
        input_seq, target_seq = batch_data
        
        # 将数据移动到设备
        input_seq = input_seq.to(self.device)
        target_seq = target_seq.to(self.device)
        
        # 确保数据类型正确
        if input_seq.dtype != torch.float32:
            input_seq = input_seq.float()
        if target_seq.dtype != torch.float32:
            target_seq = target_seq.float()
        
        with torch.no_grad():
            reconstruction = self.model(input_seq)
            loss = self.criterion(reconstruction, target_seq)
        
        return float(loss.item())

    def train_epoch(self, train_loader, epoch_idx: Optional[int] = None) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        count = 0
        
        for batch in train_loader:
            # 处理不同的批次格式
            if isinstance(batch, dict):
                sequences = batch.get("sequences") or batch.get("data")
                targets = batch.get("targets") or batch.get("target")
            else:
                # 元组格式
                sequences, targets = batch

            if sequences is None or targets is None:
                raise ValueError("训练批次缺少必要的数据/标签字段")

            loss = self.train_step((sequences, targets))
            total_loss += loss
            count += 1

        avg_loss = total_loss / max(count, 1)
        self.metrics["train_losses"].append(avg_loss)
        return avg_loss

    def validate(self, val_loader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        for batch in val_loader:
            # 处理不同的批次格式
            if isinstance(batch, dict):
                sequences = batch.get("sequences") or batch.get("data")
                targets = batch.get("targets") or batch.get("target")
            else:
                # 元组格式
                sequences, targets = batch

            if sequences is None or targets is None:
                raise ValueError("验证批次缺少必要的数据/标签字段")
            
            loss = self.compute_loss((sequences, targets))
            total_loss += loss
            count += 1
        
        avg_loss = total_loss / max(count, 1)
        self.metrics["val_losses"].append(avg_loss)
        return avg_loss

    def check_early_stopping(self, val_loss: float, patience: int) -> bool:
        """检查早停条件"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
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
    ):
        """训练模型"""
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )
                if patience and self.check_early_stopping(val_loss, patience):
                    print(f"⏹️ 早停于第 {epoch+1} 轮")
                    break
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f}")

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
        self.metrics = {"train_losses": [], "val_losses": [], "epochs_trained": 0}
        self.best_val_loss = float("inf")
        self.patience_counter = 0
