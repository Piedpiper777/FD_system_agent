"""RUL预测 BiLSTM/GRU 回归器 - 训练器（PyTorch版）。"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


class Trainer:
    """BiLSTM/GRU 回归器训练器（兼容 CNN/Transformer 复用）。"""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float = 5.0,
        loss_type: str = "mse",
        device: Optional[Union[str, torch.device]] = None,
        device_ids: Optional[List[int]] = None,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.loss_type = (loss_type or "mse").lower()
        self.device, self.device_ids = self._setup_device(device, device_ids)
        self.model = self._wrap_model(model)

        if self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_type == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}，支持 'mse' 或 'mae'")

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

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

    @staticmethod
    def _parse_device_ids(device_str: str) -> List[int]:
        ids: List[int] = []
        for part in device_str.split(","):
            token = part.strip()
            if not token:
                continue
            if token.startswith("cuda:"):
                token = token.split(":", 1)[1]
            if token.isdigit():
                ids.append(int(token))
        return ids

    def _setup_device(
        self,
        device: Optional[Union[str, torch.device]],
        device_ids: Optional[List[int]],
    ) -> Tuple[torch.device, List[int]]:
        # 设备默认使用 CUDA0/1，可用性优先
        if isinstance(device, torch.device):
            base_device = device
            valid_ids = (
                device_ids
                if device_ids is not None
                else ([base_device.index] if base_device.type == "cuda" and base_device.index is not None else [])
            )
        else:
            device_str = str(device or "cuda:0").strip().lower()
            if device_str in ("gpu", "cuda"):
                device_str = "cuda:0"
            # 提取希望使用的GPU列表
            preferred_ids = device_ids or (
                self._parse_device_ids(device_str) if device_str.startswith("cuda") else []
            )
            available = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            valid_ids = [idx for idx in preferred_ids if idx in available]
            if not valid_ids and available:
                valid_ids = [available[0]]

            if valid_ids:
                base_device = torch.device(f"cuda:{valid_ids[0]}")
            elif device_str.startswith("cuda") and torch.cuda.is_available():
                base_device = torch.device("cuda:0")
                valid_ids = [0]
            else:
                base_device = torch.device("cpu")
                valid_ids = []

        if base_device.type == "cuda" and not torch.cuda.is_available():
            base_device = torch.device("cpu")
            device_ids = []
        else:
            device_ids = valid_ids if base_device.type == "cuda" else []

        return base_device, device_ids

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        if self.device.type == "cuda" and len(self.device_ids) > 1:
            return nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def get_base_model(self) -> nn.Module:
        """返回未包装的基础模型，便于保存/推理。"""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _cpu_state_dict(self) -> Dict[str, torch.Tensor]:
        base_model = self.get_base_model()
        return {k: v.detach().cpu() for k, v in base_model.state_dict().items()}

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        mse = np.mean((preds - labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - labels))
        ss_res = np.sum((labels - preds) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }

    def _forward(self, input_seq: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = self.model(input_seq)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        return self.criterion(preds, labels)

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        preds_list, labels_list = [], []

        for batch in dataloader:
            if len(batch) != 2:
                raise ValueError("数据加载器应返回 (input_seq, labels) 元组")
            input_seq, labels = batch
            input_seq = input_seq.to(self.device).float()
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            loss = self._forward(input_seq, labels)
            loss.backward()

            if self.clip_grad_norm and self.clip_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                preds = self.model(input_seq).detach().cpu().numpy().reshape(-1)
                labels_np = labels.detach().cpu().numpy().reshape(-1)
                preds_list.append(preds)
                labels_list.append(labels_np)

        num_batches = max(1, len(dataloader))
        avg_loss = total_loss / num_batches
        preds_np = np.concatenate(preds_list) if preds_list else np.array([])
        labels_np = np.concatenate(labels_list) if labels_list else np.array([])
        metrics = self._compute_metrics(preds_np, labels_np) if preds_list else {"rmse": 0.0, "mae": 0.0, "r2": 0.0}

        return {
            "loss": avg_loss,
            "train_loss": avg_loss,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        }

    def validate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        preds_list, labels_list = [], []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) != 2:
                    raise ValueError("数据加载器应返回 (input_seq, labels) 元组")
                input_seq, labels = batch
                input_seq = input_seq.to(self.device).float()
                labels = labels.to(self.device).float()

                preds = self.model(input_seq)
                if labels.ndim == 1:
                    labels = labels.unsqueeze(1)
                loss = self.criterion(preds, labels)
                total_loss += loss.item()

                preds_list.append(preds.detach().cpu().numpy().reshape(-1))
                labels_list.append(labels.detach().cpu().numpy().reshape(-1))

        num_batches = max(1, len(dataloader))
        avg_loss = total_loss / num_batches
        preds_np = np.concatenate(preds_list) if preds_list else np.array([])
        labels_np = np.concatenate(labels_list) if labels_list else np.array([])
        metrics = self._compute_metrics(preds_np, labels_np) if preds_list else {"rmse": 0.0, "mae": 0.0, "r2": 0.0}

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
        early_stop_mode: str = "loss",
        save_path: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        best_state = None

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.metrics["train_losses"].append(train_metrics["loss"])
            self.metrics["train_rmses"].append(train_metrics["rmse"])
            self.metrics["train_maes"].append(train_metrics["mae"])
            self.metrics["val_losses"].append(val_metrics["loss"])
            self.metrics["val_rmses"].append(val_metrics["rmse"])
            self.metrics["val_maes"].append(val_metrics["mae"])
            self.metrics["epochs_trained"] = epoch + 1

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
                if save_path:
                    best_state = self._cpu_state_dict()
            else:
                self.patience_counter += 1

            if progress_callback:
                progress_callback(epoch + 1, num_epochs, train_metrics, val_metrics)

            if self.patience_counter >= patience:
                print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break

        if best_state and save_path:
            torch.save(best_state, save_path)

        return self.metrics

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()


__all__ = ["Trainer"]
