"""
RUL预测推理服务（PyTorch版）。
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

# 从本地 core 模块导入模型类（已切换为PyTorch）
from ..core.BiLSTMGRU.model_builder import ModelBuilder as BiLSTMModelBuilder
from ..core.cnn_1d_regressor.model_builder import ModelBuilder as CNN1DModelBuilder
from ..core.transformer_regressor.model_builder import ModelBuilder as TransformerModelBuilder


MODEL_BUILDERS = {
    'bilstm_gru_regressor': BiLSTMModelBuilder,
    'cnn_1d_regressor': CNN1DModelBuilder,
    'transformer_encoder_regressor': TransformerModelBuilder,
}

MODEL_PARAM_KEYS = {
    'bilstm_gru_regressor': {
        'hidden_units', 'num_layers', 'dropout', 'activation', 'bidirectional',
        'use_attention', 'use_layer_norm', 'rnn_type'
    },
    'cnn_1d_regressor': {
        'conv_channels', 'kernel_sizes', 'activation', 'dropout', 'pooling',
        'use_batch_norm', 'fc_units'
    },
    'transformer_encoder_regressor': {
        'embed_dim', 'num_heads', 'num_layers', 'ffn_dim', 'dropout',
        'activation', 'pooling', 'use_positional_encoding'
    }
}


class RULPredictionInferencer:
    """本地RUL预测推理服务（兼容云端PyTorch模型）。"""

    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Optional[Union[str, Path]] = None,
        label_scaler_path: Optional[Union[str, Path]] = None,
        config_path: Union[str, Path] = None,
        sequence_length: int = 50,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.label_scaler = None
        self.config = None
        self.device: torch.device = torch.device("cpu")

        self.logger = logging.getLogger(__name__)

        # 1. 加载配置
        if config_path:
            self._load_config(config_path)
        else:
            model_dir = Path(model_path).parent
            potential_config = model_dir / 'model_config.json'
            if potential_config.exists():
                self._load_config(potential_config)
            else:
                raise FileNotFoundError(f"配置文件不存在，且无法在模型目录中找到: {model_dir}")

        # 设备设置（默认CUDA0/1，取首个可用GPU）
        self.device = self._resolve_device(device or self.config.get('device'))

        # 2. 加载特征scaler（可选）
        if scaler_path:
            self._load_scaler(scaler_path)
        else:
            model_dir = Path(model_path).parent
            potential_scaler = model_dir / 'scaler.pkl'
            if potential_scaler.exists():
                self._load_scaler(potential_scaler)

        # 3. 加载标签scaler（必需）
        if label_scaler_path:
            self._load_label_scaler(label_scaler_path)
        else:
            model_dir = Path(model_path).parent
            potential_label_scaler = model_dir / 'label_scaler.pkl'
            if potential_label_scaler.exists():
                self._load_label_scaler(potential_label_scaler)
            else:
                raise FileNotFoundError(f"标签归一化器文件不存在，且无法在模型目录中找到: {model_dir}")

        # 4. 加载模型
        self._load_model(model_path)

        print(f"✅ RUL预测推理服务初始化完成")
        print(f"  - 模型: {model_path}")
        if config_path:
            print(f"  - 配置文件: {config_path}")
        if scaler_path:
            print(f"  - 特征标准化器: {scaler_path}")
        if self.label_scaler:
            print(f"  - 标签归一化器: 已加载")
        print(f"  - 序列长度: {sequence_length}")
        print(f"  - 推理设备: {self.device}")

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

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        if isinstance(device, torch.device):
            target = device
        else:
            device_str = str(device or "cuda:0").strip().lower()
            if device_str in ("gpu", "cuda"):
                device_str = "cuda:0"
            if device_str.startswith("cuda") and torch.cuda.is_available():
                ids = self._parse_device_ids(device_str) or [0]
                ids = [i for i in ids if i < torch.cuda.device_count()]
                if ids:
                    return torch.device(f"cuda:{ids[0]}")
                return torch.device("cuda:0")
            target = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

        if target.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return target

    def _load_config(self, config_path: Union[str, Path]):
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            if 'sequence_length' in self.config:
                self.sequence_length = self.config['sequence_length']

            print(f"📂 配置文件加载成功: {config_path}")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise

    def _load_model(self, model_path: Union[str, Path]):
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            model_type = self.config.get('model_type', 'bilstm_gru_regressor')
            input_dim = self.config.get('input_dim', 1)
            input_shape = (self.sequence_length, input_dim)

            builder = MODEL_BUILDERS.get(model_type)
            if builder is None:
                raise ValueError(f"不支持的模型类型: {model_type}")

            allowed_keys = MODEL_PARAM_KEYS.get(model_type, set())
            builder_kwargs = {
                key: value for key, value in self.config.items()
                if key in allowed_keys
            }

            model = builder.create_model(
                model_type=model_type,
                input_shape=input_shape,
                **builder_kwargs
            )

            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            self.model = model.to(self.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise

    def _load_scaler(self, scaler_path: Union[str, Path]):
        try:
            scaler_path = Path(scaler_path)
            if not scaler_path.exists():
                self.logger.warning(f"特征标准化器文件不存在: {scaler_path}，数据可能已归一化")
                return

            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            print(f"📂 特征标准化器加载成功: {scaler_path}")
        except Exception as e:
            self.logger.error(f"加载特征标准化器失败: {e}")
            raise

    def _load_label_scaler(self, label_scaler_path: Union[str, Path]):
        try:
            label_scaler_path = Path(label_scaler_path)
            if not label_scaler_path.exists():
                raise FileNotFoundError(f"标签归一化器文件不存在: {label_scaler_path}")

            with open(label_scaler_path, 'rb') as f:
                self.label_scaler = pickle.load(f)

            print(f"📂 标签归一化器加载成功: {label_scaler_path}")
        except Exception as e:
            self.logger.error(f"加载标签归一化器失败: {e}")
            raise

    def _create_sequences(self, data: np.ndarray, sequence_length: int, stride: int = 1) -> np.ndarray:
        sequences = []
        for i in range(0, len(data) - sequence_length + 1, stride):
            sequences.append(data[i:i + sequence_length])
        if len(sequences) == 0:
            return np.array([]).reshape(0, sequence_length, data.shape[1])
        return np.array(sequences, dtype=np.float32)

    def predict(self, data: Union[np.ndarray, pd.DataFrame],
                batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        try:
            if isinstance(data, pd.DataFrame):
                data = data.values

            if len(data) < self.sequence_length:
                raise ValueError(f"数据长度 ({len(data)}) 小于序列长度 ({self.sequence_length})")

            sequences = self._create_sequences(data, self.sequence_length, stride=1)
            if len(sequences) == 0:
                return np.array([])

            if self.scaler is not None:
                sequences_2d = sequences.reshape(-1, sequences.shape[2])
                sequences_2d_scaled = self.scaler.transform(sequences_2d)
                sequences_scaled = sequences_2d_scaled.reshape(sequences.shape)
            else:
                sequences_scaled = sequences

            all_predictions = []
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(sequences_scaled), batch_size):
                    batch = sequences_scaled[i:i + batch_size]
                    batch_tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)
                    predictions = self.model(batch_tensor)
                    preds_np = predictions.detach().cpu().numpy()
                    all_predictions.append(preds_np.flatten())

            rul_predictions_normalized = np.concatenate(all_predictions) if all_predictions else np.array([])

            if self.label_scaler is not None:
                rul_predictions = self.label_scaler.inverse_transform(
                    rul_predictions_normalized.reshape(-1, 1)
                ).flatten()
            else:
                self.logger.warning("未找到标签归一化器，预测值可能未反归一化")
                rul_predictions = rul_predictions_normalized

            # 推理阶段强制非负
            rul_predictions = np.maximum(0, rul_predictions)
            rul_predictions_normalized = np.maximum(0, rul_predictions_normalized)

            if len(rul_predictions) < len(data):
                extended_predictions = np.zeros(len(data))
                extended_predictions[:self.sequence_length - 1] = rul_predictions[0] if len(rul_predictions) else 0
                extended_predictions[self.sequence_length - 1:] = rul_predictions
                rul_predictions = extended_predictions
            if len(rul_predictions_normalized) < len(data):
                extended_norm = np.zeros(len(data))
                if len(rul_predictions_normalized):
                    extended_norm[:self.sequence_length - 1] = rul_predictions_normalized[0]
                    extended_norm[self.sequence_length - 1:] = rul_predictions_normalized
                rul_predictions_normalized = extended_norm

            return rul_predictions, rul_predictions_normalized

        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            raise

    def predict_rul(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data_file = data_config.get('data_file')
            batch_size = data_config.get('batch_size', 32)

            if not data_file:
                raise ValueError("缺少data_file参数")

            data_path = Path(data_file)
            if not data_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {data_path}")

            df = pd.read_csv(data_path)

            condition_keys = self.config.get('condition_keys', [])

            timestamp_col = None
            for col in df.columns:
                if 'time' in col.lower() or 'timestamp' in col.lower():
                    timestamp_col = col
                    break

            if condition_keys:
                condition_values = data_config.get('condition_values', {})
                for key in condition_keys:
                    if key not in df.columns:
                        if key in condition_values:
                            df[key] = condition_values[key]
                        else:
                            conditions = self.config.get('conditions', [])
                            for cond in conditions:
                                if isinstance(cond, dict) and cond.get('name') == key:
                                    values = cond.get('values', [])
                                    if values:
                                        df[key] = values[0]
                                    else:
                                        df[key] = 0
                                    break
                            else:
                                df[key] = 0

            exclude_cols = set(condition_keys)
            if timestamp_col:
                exclude_cols.add(timestamp_col)
            sensor_feature_cols = [col for col in df.columns if col not in exclude_cols]
            condition_cols = condition_keys

            sensor_features = df[sensor_feature_cols].values.astype(np.float32)
            condition_features = df[condition_cols].values.astype(np.float32) if condition_cols else None

            if condition_features is not None:
                data = np.hstack([sensor_features, condition_features])
            else:
                data = sensor_features

            rul_predictions, rul_predictions_normalized = self.predict(data, batch_size=batch_size)

            result = {
                'success': True,
                'predictions': rul_predictions.tolist(),
                'predictions_normalized': rul_predictions_normalized.tolist(),
                'num_samples': len(rul_predictions),
                'data_file': str(data_file),
                'model_config': {
                    'model_type': self.config.get('model_type'),
                    'sequence_length': self.sequence_length,
                    'input_dim': self.config.get('input_dim'),
                    'device': str(self.device),
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"RUL预测失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
