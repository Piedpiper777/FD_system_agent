"""
LSTM自编码器 - 训练器
负责重构任务的训练循环与指标记录
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import mindspore as ms
import mindspore.ops as ops
from mindspore.nn import Adam, MSELoss


class Trainer:
    """标准LSTM Autoencoder训练循环"""

    def __init__(
        self,
        model: ms.nn.Cell,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float = 5.0,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm

        self.optimizer = Adam(
            params=self.model.trainable_params(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = MSELoss()
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters)
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.metrics = {
            "train_losses": [],
            "val_losses": [],
            "epochs_trained": 0,
        }

    def forward_fn(self, input_seq, target_seq):
        reconstruction = self.model(input_seq)
        loss = self.criterion(reconstruction, target_seq)
        return loss

    def _process_grads(self, grads):
        # 简化：暂时移除梯度裁剪以避免兼容性问题
        return grads

    def train_step(self, batch_data):
        input_seq, target_seq = batch_data
        loss, grads = self.grad_fn(input_seq, target_seq)
        grads = self._process_grads(grads)
        self.optimizer(grads)
        return float(loss)

    def compute_loss(self, batch_data):
        input_seq, target_seq = batch_data
        reconstruction = self.model(input_seq)
        return float(self.criterion(reconstruction, target_seq))

    def train_epoch(self, train_loader, epoch_idx: Optional[int] = None) -> float:
        self.model.set_train(True)
        total = 0.0
        count = 0
        for batch in train_loader:
            # 处理不同的批次格式
            if isinstance(batch, dict):
                sequences = batch["sequences"]
                targets = batch["targets"]
            else:
                # 元组格式
                sequences, targets = batch
            loss = self.train_step((sequences, targets))
            total += loss
            count += 1
        avg = total / max(count, 1)
        self.metrics["train_losses"].append(avg)
        # 注意：日志输出由调用方（api.py）统一管理，这里不输出日志
        # 如果需要调试，可以取消下面的注释
        # if epoch_idx is not None:
        #     print(f"Epoch [{epoch_idx+1}] Train Loss: {avg:.6f}")
        return avg

    def validate(self, val_loader) -> float:
        self.model.set_train(False)
        total = 0.0
        count = 0
        for batch in val_loader:
            # 处理不同的批次格式
            if isinstance(batch, dict):
                sequences = batch["sequences"]
                targets = batch["targets"]
            else:
                # 元组格式
                sequences, targets = batch
            loss = self.compute_loss((sequences, targets))
            total += loss
            count += 1
        avg = total / max(count, 1)
        self.metrics["val_losses"].append(avg)
        return avg

    def check_early_stopping(self, val_loss: float, patience: int) -> bool:
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
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ms.save_checkpoint(self.model, str(save_path))

    def get_training_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()

    def reset_metrics(self):
        self.metrics = {"train_losses": [], "val_losses": [], "epochs_trained": 0}
        self.best_val_loss = float("inf")
        self.patience_counter = 0
