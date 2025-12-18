"""
RUL预测 BiLSTM/GRU 回归器 - 训练器
负责回归任务的训练循环与指标记录
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import mindspore as ms
import mindspore.ops as ops
from mindspore.nn import Adam, MSELoss, L1Loss
import numpy as np


class Trainer:
    """BiLSTM/GRU 回归器训练器"""
    
    def __init__(
        self,
        model: ms.nn.Cell,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float = 5.0,
        loss_type: str = "mse",  # "mse" 或 "mae"
    ):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            learning_rate: 学习率
            weight_decay: 权重衰减
            clip_grad_norm: 梯度裁剪阈值
            loss_type: 损失函数类型，"mse" 或 "mae"
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.loss_type = loss_type.lower()
        
        # 优化器
        self.optimizer = Adam(
            params=self.model.trainable_params(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 损失函数：MSE或MAE
        if self.loss_type == "mse":
            self.criterion = MSELoss()
        elif self.loss_type == "mae":
            self.criterion = L1Loss()
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}，支持 'mse' 或 'mae'")
        
        # 梯度计算函数
        self.grad_fn = ms.value_and_grad(
            self.forward_fn, None, self.optimizer.parameters
        )
        
        # 训练指标
        self.best_val_loss = float("inf")
        self.best_val_rmse = float("inf")
        self.best_val_mae = float("inf")
        self.patience_counter = 0
        self.metrics = {
            "train_losses": [],
            "train_rmses": [],
            "train_maes": [],
            "val_losses": [],
            "val_rmses": [],
            "val_maes": [],
            "epochs_trained": 0,
        }
    
    def forward_fn(self, input_seq, labels):
        """
        前向传播函数
        
        Args:
            input_seq: 输入序列 (batch_size, seq_len, n_features)
            labels: RUL标签 (batch_size, 1) 或 (batch_size,)
            
        Returns:
            损失值
        """
        predictions = self.model(input_seq)  # (batch_size, 1)
        
        # 确保labels的形状与predictions一致
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        
        loss = self.criterion(predictions, labels)
        return loss
    
    def _process_grads(self, grads):
        """处理梯度（梯度裁剪）"""
        if self.clip_grad_norm > 0:
            # 计算梯度范数
            total_norm = 0.0
            for grad in grads:
                if grad is not None:
                    total_norm += ops.norm(grad) ** 2
            total_norm = total_norm ** 0.5
            
            # 如果梯度范数超过阈值，进行裁剪
            if total_norm > self.clip_grad_norm:
                clip_coef = self.clip_grad_norm / (total_norm + 1e-6)
                grads = [grad * clip_coef if grad is not None else grad for grad in grads]
        
        # 确保返回元组类型，与MindSpore优化器期望的格式一致
        return tuple(grads)
    
    def compute_metrics(self, predictions: ms.Tensor, labels: ms.Tensor) -> Dict[str, float]:
        """
        计算回归指标
        
        Args:
            predictions: 预测值 (batch_size, 1)
            labels: 真实值 (batch_size, 1) 或 (batch_size,)
            
        Returns:
            包含RMSE和MAE的字典
        """
        # 确保labels的形状与predictions一致
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        
        # 转换为numpy计算指标
        pred_np = predictions.asnumpy().flatten()
        label_np = labels.asnumpy().flatten()
        
        # 计算MSE和RMSE
        mse = np.mean((pred_np - label_np) ** 2)
        rmse = np.sqrt(mse)
        
        # 计算MAE
        mae = np.mean(np.abs(pred_np - label_np))
        
        # 计算R²
        ss_res = np.sum((label_np - pred_np) ** 2)
        ss_tot = np.sum((label_np - np.mean(label_np)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 训练数据加载器
            
        Returns:
            包含训练指标的字典
        """
        self.model.set_train()
        
        # 训练整个epoch，同时累积训练损失和指标
        total_train_loss = 0.0  # 训练模式下的损失（包含dropout）
        total_eval_loss = 0.0   # 评估模式下的损失（用于与验证损失对比）
        all_train_predictions = []
        all_eval_predictions = []
        all_labels = []
        num_batches = 0
        
        for batch_data in dataloader:
            if len(batch_data) == 2:
                input_seq, labels = batch_data
            else:
                raise ValueError("数据加载器应返回 (input_seq, labels) 元组")
            
            # 前向传播和梯度计算（用于参数更新，在训练模式下）
            loss, grads = self.grad_fn(input_seq, labels)
            
            # 累积训练模式下的损失
            total_train_loss += float(loss.asnumpy())
            
            # 处理梯度
            grads = self._process_grads(grads)
            
            # 更新参数
            self.optimizer(grads)
            
            # 在评估模式下计算损失（用于与验证损失对比）
            self.model.set_train(False)
            with ms._no_grad():
                eval_predictions = self.model(input_seq)
                if len(labels.shape) == 1:
                    labels_reshaped = labels.reshape(-1, 1)
                else:
                    labels_reshaped = labels
                eval_loss = self.criterion(eval_predictions, labels_reshaped)
                total_eval_loss += float(eval_loss.asnumpy())
                all_eval_predictions.append(eval_predictions)
            self.model.set_train()  # 恢复训练模式
            
            # 保存标签（用于计算指标）
            all_labels.append(labels)
            
            num_batches += 1
        
        # 计算平均损失
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0.0
        avg_eval_loss = total_eval_loss / num_batches if num_batches > 0 else 0.0
        
        # 计算指标（使用评估模式下的预测值，与验证损失可比）
        if all_eval_predictions:
            all_eval_predictions = ops.concat(all_eval_predictions, axis=0)
            all_labels_tensor = ops.concat(all_labels, axis=0)
            metrics = self.compute_metrics(all_eval_predictions, all_labels_tensor)
        else:
            metrics = {"rmse": 0.0, "mae": 0.0, "r2": 0.0}
        
        # 返回评估模式下的损失（与验证损失可比），但保留训练损失信息
        return {
            "loss": avg_eval_loss,  # 使用评估模式下的损失，与验证损失可比
            "train_loss": avg_train_loss,  # 训练模式下的损失（包含dropout，通常更大）
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            包含验证指标的字典
        """
        self.model.set_train(False)
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        for batch_data in dataloader:
            if len(batch_data) == 2:
                input_seq, labels = batch_data
            else:
                raise ValueError("数据加载器应返回 (input_seq, labels) 元组")
            
            # 前向传播
            predictions = self.model(input_seq)
            
            # 计算损失
            if len(labels.shape) == 1:
                labels_reshaped = labels.reshape(-1, 1)
            else:
                labels_reshaped = labels
            loss = self.criterion(predictions, labels_reshaped)
            
            total_loss += float(loss.asnumpy())
            all_predictions.append(predictions)
            all_labels.append(labels)
            num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 计算指标
        all_predictions = ops.concat(all_predictions, axis=0)
        all_labels = ops.concat(all_labels, axis=0)
        metrics = self.compute_metrics(all_predictions, all_labels)
        
        return {
            "loss": avg_loss,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        patience: int = 10,
        early_stop_mode: str = "loss",  # "loss", "rmse", "mae"
        save_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            patience: 早停耐心值
            early_stop_mode: 早停模式，"loss", "rmse", 或 "mae"
            save_path: 模型保存路径
            progress_callback: 进度回调函数，接收 (epoch, metrics) 参数
            
        Returns:
            训练历史字典
        """
        best_model_state = None
        
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 记录指标
            self.metrics["train_losses"].append(train_metrics["loss"])
            self.metrics["train_rmses"].append(train_metrics["rmse"])
            self.metrics["train_maes"].append(train_metrics["mae"])
            self.metrics["val_losses"].append(val_metrics["loss"])
            self.metrics["val_rmses"].append(val_metrics["rmse"])
            self.metrics["val_maes"].append(val_metrics["mae"])
            self.metrics["epochs_trained"] = epoch + 1
            
            # 早停判断
            if early_stop_mode == "loss":
                current_metric = val_metrics["loss"]
                best_metric = self.best_val_loss
            elif early_stop_mode == "rmse":
                current_metric = val_metrics["rmse"]
                best_metric = self.best_val_rmse
            elif early_stop_mode == "mae":
                current_metric = val_metrics["mae"]
                best_metric = self.best_val_mae
            else:
                current_metric = val_metrics["loss"]
                best_metric = self.best_val_loss
            
            # 更新最佳指标
            if current_metric < best_metric:
                if early_stop_mode == "loss":
                    self.best_val_loss = current_metric
                elif early_stop_mode == "rmse":
                    self.best_val_rmse = current_metric
                elif early_stop_mode == "mae":
                    self.best_val_mae = current_metric
                
                self.best_val_rmse = min(self.best_val_rmse, val_metrics["rmse"])
                self.best_val_mae = min(self.best_val_mae, val_metrics["mae"])
                
                self.patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    best_model_state = self.model.parameters_dict()
            else:
                self.patience_counter += 1
            
            # 进度回调
            if progress_callback:
                progress_callback(
                    epoch + 1,
                    num_epochs,
                    train_metrics,
                    val_metrics,
                )
            
            # 早停
            if self.patience_counter >= patience:
                print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break
        
        # 恢复最佳模型
        if best_model_state and save_path:
            ms.load_param_into_net(self.model, best_model_state)
            ms.save_checkpoint(self.model, save_path)
        
        return self.metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取训练指标"""
        return self.metrics.copy()

