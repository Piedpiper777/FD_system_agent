"""RUL预测核心模块，封装模型注册与评估工具。"""

from typing import Dict, List

from .bilstm_gru_regressor.model_builder import ModelBuilder as BiLSTMModelBuilder
from .bilstm_gru_regressor.trainer import Trainer as BiLSTMTrainer
from .bilstm_gru_regressor.data_processor import DataProcessor as DefaultDataProcessor
from .cnn_1d_regressor.model_builder import ModelBuilder as CNNModelBuilder
from .cnn_1d_regressor.trainer import Trainer as CNNTrainer
from .cnn_1d_regressor.data_processor import DataProcessor as CNNDataProcessor
from .transformer_regressor.model_builder import ModelBuilder as TransformerModelBuilder
from .transformer_regressor.trainer import Trainer as TransformerTrainer
from .transformer_regressor.data_processor import DataProcessor as TransformerDataProcessor
from .evaluator import RULPredictionEvaluator

MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
	'bilstm_gru_regressor': {
		'builder': BiLSTMModelBuilder,
		'trainer': BiLSTMTrainer,
		'data_processor': DefaultDataProcessor,
	},
	'cnn_1d_regressor': {
		'builder': CNNModelBuilder,
		'trainer': CNNTrainer,
		'data_processor': CNNDataProcessor,
	},
	'transformer_encoder_regressor': {
		'builder': TransformerModelBuilder,
		'trainer': TransformerTrainer,
		'data_processor': TransformerDataProcessor,
	}
}


def get_available_model_types() -> List[str]:
	return list(MODEL_REGISTRY.keys())


def get_model_builder(model_type: str):
	entry = MODEL_REGISTRY.get(model_type)
	if not entry:
		raise ValueError(f"Unsupported model_type: {model_type}")
	return entry['builder']


def get_trainer_class(model_type: str):
	entry = MODEL_REGISTRY.get(model_type)
	if not entry:
		raise ValueError(f"Unsupported model_type: {model_type}")
	return entry['trainer']


def get_data_processor_class(model_type: str):
	entry = MODEL_REGISTRY.get(model_type)
	if not entry:
		raise ValueError(f"Unsupported model_type: {model_type}")
	return entry['data_processor']


__all__ = [
	'RULPredictionEvaluator',
	'MODEL_REGISTRY',
	'get_available_model_types',
	'get_model_builder',
	'get_trainer_class',
	'get_data_processor_class',
]

