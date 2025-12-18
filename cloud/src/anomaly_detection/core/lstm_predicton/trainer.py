"""
LSTM预测异常检测模块 - 训练器
负责模型训练、超参数调优和训练过程监控
"""

import mindspore as ms
import mindspore.ops as ops
from mindspore.nn import MSELoss, Adam
import mindspore.numpy as np
from typing import Optional, Any, Dict, Union
from pathlib import Path


class Trainer:
    """
    LSTM模型训练器

    专注于模型训练的核心功能：
    - 执行训练循环
    - 记录训练指标
    - 支持早停机制
    - 模型保存和加载
    """

    def __init__(self, model: ms.nn.Cell, learning_rate: float = 0.001,
                 weight_decay: float = 1e-4, clip_grad_norm: float = 5.0):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            learning_rate: 学习率
            weight_decay: 权重衰减
            clip_grad_norm: 梯度裁剪范数
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm

        # 初始化优化器
        self.optimizer = Adam(
            params=self.model.trainable_params(),
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        # 训练状态
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_metrics = {
            'train_losses': [],
            'val_losses': [],
            'epochs_trained': 0
        }

        # 损失函数
        self.criterion = MSELoss()

        # 梯度函数
        self.grad_fn = ms.value_and_grad(
            self.forward_fn, None, self.optimizer.parameters, has_aux=False
        )

        print(f"✅ 训练器初始化完成")
        print(f"  - 学习率: {learning_rate}")
        print(f"  - 权重衰减: {weight_decay}")

    def process_gradients(self, grads):
        """处理梯度（裁剪等）"""
        if self.clip_grad_norm > 0:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.clip_grad_norm)
        return grads

    def forward_fn(self, input_seq, target):
        """
        前向传播函数

        Args:
            input_seq: 输入序列
            target: 目标值

        Returns:
            损失值
        """
        prediction = self.model(input_seq)
        loss = self.criterion(prediction, target)
        return loss

    def train_step(self, batch_data) -> float:
        """
        单步训练

        Args:
            batch_data: (input_seq, target)

        Returns:
            损失值
        """
        input_seq, target = batch_data

        # 前向传播和梯度计算
        loss, grads = self.grad_fn(input_seq, target)

        # 梯度处理和参数更新
        grads = self.process_gradients(grads)
        self.optimizer(grads)

        return float(loss)

    def compute_loss(self, batch_data) -> float:
        """
        计算批次损失

        Args:
            batch_data: (input_seq, target)

        Returns:
            损失值
        """
        input_seq, target = batch_data
        prediction = self.model(input_seq)
        loss = self.criterion(prediction, target)
        return float(loss)

    def train_epoch(self, train_loader, epoch_idx: Optional[int] = None) -> float:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch_idx: epoch索引

        Returns:
            平均训练损失
        """
        self.model.set_train(True)
        total_loss = 0.0
        batch_count = 0

        for batch_data in train_loader:
            loss = self.train_step(batch_data)
            total_loss += loss
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        self.training_metrics['train_losses'].append(avg_loss)

        # 注意：日志输出由调用方（api.py）统一管理，这里不输出日志
        # 如果需要调试，可以取消下面的注释
        # if epoch_idx is not None:
        #     print(f"Epoch [{epoch_idx+1}] Train Loss: {avg_loss:.6f}")

        return avg_loss

    def validate(self, val_loader) -> float:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            平均验证损失
        """
        self.model.set_train(False)
        total_loss = 0.0
        batch_count = 0

        for batch_data in val_loader:
            loss = self.compute_loss(batch_data)
            total_loss += loss
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        return avg_loss

    def check_early_stopping(self, val_loss: float, patience: int) -> bool:
        """
        检查是否应该早停

        Args:
            val_loss: 当前验证损失
            patience: 耐心值

        Returns:
            是否应该停止训练
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience

    def train(self, train_loader, num_epochs: int = 50, val_loader=None,
              patience: Optional[int] = None) -> ms.nn.Cell:
        """
        训练主循环

        Args:
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            val_loader: 验证数据加载器
            patience: 早停耐心值

        Returns:
            训练好的模型
        """
        print(f"🚀 开始模型训练...")
        print(f"  - 训练轮数: {num_epochs}")

        for epoch in range(num_epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, epoch)

            # 验证
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.training_metrics['val_losses'].append(val_loss)

                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                # 早停检查
                if patience is not None and self.check_early_stopping(val_loss, patience):
                    print(f"⏹️ 早停于第 {epoch+1} 轮")
                    break
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}")

        self.training_metrics['epochs_trained'] = epoch + 1
        print("✅ 训练完成!")

        # 打印训练总结
        self._print_training_summary()

        return self.model

    def _print_training_summary(self):
        """打印训练总结"""
        print("\n📊 训练总结:")
        print(f"  - 训练轮数: {self.training_metrics['epochs_trained']}")
        print(f"  - 最终训练损失: {self.training_metrics['train_losses'][-1]:.6f}")
        if self.training_metrics['val_losses']:
            print(f"  - 最终验证损失: {self.training_metrics['val_losses'][-1]:.6f}")

    def save_model(self, save_path: Union[str, Path]):
        """
        保存模型权重

        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        ms.save_checkpoint(self.model, str(save_path))
        print(f"💾 模型已保存: {save_path}")

    def load_model(self, load_path: Union[str, Path]) -> ms.nn.Cell:
        """
        加载模型权重

        Args:
            load_path: 加载路径

        Returns:
            加载的模型
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {load_path}")

        param_dict = ms.load_checkpoint(str(load_path))
        ms.load_param_into_net(self.model, param_dict)
        print(f"📂 模型已加载: {load_path}")

        return self.model

    def get_training_metrics(self) -> Dict[str, Any]:
        """
        获取训练指标

        Returns:
            训练指标字典
        """
        return self.training_metrics.copy()

    def reset_metrics(self):
        """重置训练指标"""
        self.training_metrics = {
            'train_losses': [],
            'val_losses': [],
            'epochs_trained': 0
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0