"""
后端参数验证器
提供服务器端的参数验证，确保数据安全性和一致性
"""

import re
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool = True
    errors: List[str] = None
    warnings: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []


class ParameterValidator:
    """后端参数验证器"""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Dict]]:
        """初始化验证规则"""
        return {
            # LSTM预测器模型参数验证规则
            'lstm_predictor': {
                'input_dim': {
                    'required': False,
                    'type': int,
                    'min': 1,
                    'max': 1000,
                    'message': '输入维度必须是1-1000之间的正整数'
                },
                'hidden_units': {
                    'required': True,
                    'type': int,
                    'min': 8,
                    'max': 1024,
                    'message': '隐藏层维度建议在8-1024之间'
                },
                'num_layers': {
                    'required': True,
                    'type': int,
                    'min': 1,
                    'max': 8,
                    'message': 'LSTM层数建议1-8层'
                },
                'sequence_length': {
                    'required': True,
                    'type': int,
                    'min': 5,
                    'max': 1000,
                    'message': '序列长度建议5-1000'
                },
                'batch_size': {
                    'required': True,
                    'type': int,
                    'min': 1,
                    'max': 512,
                    'power_of_2': True,
                    'message': '批次大小建议为2的幂次'
                },
                'prediction_horizon': {
                    'required': True,
                    'type': int,
                    'min': 1,
                    'max': 100,
                    'message': '预测步长建议1-100'
                }
            },
            # 1D CNN Autoencoder模型参数验证规则
            'cnn_1d_autoencoder': {
                'sequence_length': {
                    'required': True,
                    'type': int,
                    'min': 5,
                    'max': 1000,
                    'message': '序列长度建议5-1000'
                },
                'num_filters': {
                    'required': True,
                    'type': int,
                    'min': 8,
                    'max': 512,
                    'message': '卷积核数量建议8-512'
                },
                'kernel_size': {
                    'required': True,
                    'type': int,
                    'min': 2,
                    'max': 20,
                    'message': '卷积核大小建议2-20'
                },
                'bottleneck_size': {
                    'required': True,
                    'type': int,
                    'min': 4,
                    'max': 512,
                    'message': '瓶颈层维度建议4-512'
                },
                'num_conv_layers': {
                    'required': True,
                    'type': int,
                    'min': 1,
                    'max': 10,
                    'message': '卷积层数建议1-10层'
                },
                'stride': {
                    'required': False,
                    'type': int,
                    'min': 1,
                    'max': 5,
                    'message': '步长建议1-5'
                },
                'batch_size': {
                    'required': True,
                    'type': int,
                    'min': 1,
                    'max': 512,
                    'power_of_2': True,
                    'message': '批次大小建议为2的幂次'
                }
            },
            # 训练参数验证规则
            'training': {
                'learning_rate': {
                    'required': True,
                    'type': float,
                    'min': 1e-6,
                    'max': 1.0,
                    'typical': [0.001, 0.01, 0.1],
                    'message': '学习率建议0.001-0.1'
                },
                'epochs': {
                    'required': True,
                    'type': int,
                    'min': 1,
                    'max': 1000,
                    'message': '训练轮数建议10-200'
                },
                'weight_decay': {
                    'required': False,
                    'type': float,
                    'min': 0,
                    'max': 1.0,
                    'message': '权重衰减建议0-0.01'
                },
                'train_ratio': {
                    'required': True,
                    'type': float,
                    'min': 0.1,
                    'max': 0.95,
                    'message': '训练集比例建议0.6-0.8'
                },
                'val_ratio': {
                    'required': True,
                    'type': float,
                    'min': 0.05,
                    'max': 0.5,
                    'message': '验证集比例建议0.1-0.2'
                },
                'test_ratio': {
                    'required': True,
                    'type': float,
                    'min': 0.05,
                    'max': 0.5,
                    'message': '测试集比例建议0.1-0.2'
                },
                'validation_split': {
                    'required': False,
                    'type': float,
                    'min': 0.05,
                    'max': 0.5,
                    'message': '验证集划分比例建议0.1-0.3'
                }
            },
            # 数据集相关验证规则
            'dataset': {
                'dataset_mode': {
                    'required': True,
                    'type': str,
                    'options': ['processed_file', 'condition_filtered'],
                    'message': '必须选择有效的数据集组织模式'
                },
                'preprocess_method': {
                    'required': True,
                    'type': str,
                    'options': ['standard', 'minmax', 'robust', 'none'],
                    'message': '必须选择有效的预处理方法'
                }
            }
        }
    
    def validate_parameter(self, category: str, param_name: str, 
                         value: Any, all_params: Dict = None) -> ValidationResult:
        """验证单个参数"""
        rule = self.validation_rules.get(category, {}).get(param_name)
        if not rule:
            return ValidationResult()
        
        result = ValidationResult()
        
        # 必填验证
        if rule.get('required', False) and (value is None or value == ''):
            result.is_valid = False
            result.errors.append(f'{self._get_parameter_display_name(param_name)}是必填项')
            return result
        
        if value is None or value == '':
            return result  # 非必填且为空，直接通过
        
        # 类型转换和验证
        try:
            if rule['type'] == int:
                value = int(value)
            elif rule['type'] == float:
                value = float(value)
            elif rule['type'] == str:
                value = str(value).strip()
        except (ValueError, TypeError):
            result.is_valid = False
            result.errors.append(f'{self._get_parameter_display_name(param_name)}类型错误')
            return result
        
        # 范围验证
        if 'min' in rule and value < rule['min']:
            result.is_valid = False
            result.errors.append(f'{self._get_parameter_display_name(param_name)}不能小于{rule["min"]}')
        
        if 'max' in rule and value > rule['max']:
            result.is_valid = False
            result.errors.append(f'{self._get_parameter_display_name(param_name)}不能大于{rule["max"]}')
        
        # 选项验证
        if 'options' in rule and value not in rule['options']:
            result.is_valid = False
            result.errors.append(f'{self._get_parameter_display_name(param_name)}必须是以下选项之一: {", ".join(rule["options"])}')
        
        # 2的幂次验证（警告）
        if rule.get('power_of_2', False) and isinstance(value, int) and value > 0:
            if (value & (value - 1)) != 0:
                result.warnings.append(f'{self._get_parameter_display_name(param_name)}建议设置为2的幂次以优化性能')
        
        # 特殊验证逻辑
        self._apply_special_validations(category, param_name, value, all_params or {}, result)
        
        # 添加建议
        if rule.get('message'):
            result.suggestions.append(rule['message'])
        
        return result
    
    def validate_all_parameters(self, model_type: str, params: Dict[str, Any]) -> ValidationResult:
        """验证所有参数"""
        result = ValidationResult()
        
        # 验证模型特定参数
        if model_type in self.validation_rules:
            for param_name, rule in self.validation_rules[model_type].items():
                param_result = self.validate_parameter(model_type, param_name, 
                                                     params.get(param_name), params)
                if not param_result.is_valid:
                    result.is_valid = False
                    result.errors.extend(param_result.errors)
                result.warnings.extend(param_result.warnings)
                result.suggestions.extend(param_result.suggestions)
        
        # 验证训练参数
        for param_name, rule in self.validation_rules['training'].items():
            # LSTM Predictor 不需要 test_ratio
            if model_type == 'lstm_predictor' and param_name == 'test_ratio':
                continue
            
            # 自编码器模型使用 validation_split，不需要 train_ratio, val_ratio, test_ratio
            if model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']:
                if param_name in ['train_ratio', 'val_ratio', 'test_ratio']:
                    continue
                
            param_result = self.validate_parameter('training', param_name, 
                                                 params.get(param_name), params)
            if not param_result.is_valid:
                result.is_valid = False
                result.errors.extend(param_result.errors)
            result.warnings.extend(param_result.warnings)
            result.suggestions.extend(param_result.suggestions)
        
        # 验证数据集参数
        for param_name, rule in self.validation_rules['dataset'].items():
            # 使用预处理数据文件的模型不需要预处理方法
            if (model_type in ['lstm_predictor', 'lstm_autoencoder', 'cnn_1d_autoencoder']) and param_name == 'preprocess_method':
                continue
                
            param_result = self.validate_parameter('dataset', param_name, 
                                                 params.get(param_name), params)
            if not param_result.is_valid:
                result.is_valid = False
                result.errors.extend(param_result.errors)
            result.warnings.extend(param_result.warnings)
            result.suggestions.extend(param_result.suggestions)
        
        return result
    
    def _apply_special_validations(self, category: str, param_name: str, 
                                 value: Any, all_params: Dict, result: ValidationResult):
        """应用特殊验证逻辑"""
        # 数据分割比例验证
        if category == 'training' and param_name.endswith('_ratio'):
            model_type = all_params.get('model_type', '')
            
            if model_type == 'lstm_predictor':
                # LSTM Predictor 只需要 train_ratio + val_ratio = 1.0
                train_ratio = float(all_params.get('train_ratio', 0) or 0)
                val_ratio = float(all_params.get('val_ratio', 0) or 0)
                ratio_sum = train_ratio + val_ratio
                if abs(ratio_sum - 1.0) > 0.001:
                    result.warnings.append('LSTM预测模型训练集和验证集比例之和应该等于1.0')
            else:
                # 其他模型需要三个比例之和等于1.0
                ratio_sum = sum([
                    float(all_params.get('train_ratio', 0) or 0),
                    float(all_params.get('val_ratio', 0) or 0),
                    float(all_params.get('test_ratio', 0) or 0)
                ])
                if abs(ratio_sum - 1.0) > 0.001:
                    result.warnings.append('训练、验证、测试集比例之和应该等于1.0')
        
        # LSTM参数组合验证
        if category == 'lstm_predictor':
            if param_name == 'hidden_units' and 'sequence_length' in all_params:
                seq_len = int(all_params.get('sequence_length', 0) or 0)
                if value > seq_len * 2:
                    result.warnings.append('隐藏层维度过大，可能导致过拟合')
            
            if param_name == 'batch_size' and 'sequence_length' in all_params:
                seq_len = int(all_params.get('sequence_length', 0) or 0)
                if value * seq_len > 50000:
                    result.warnings.append('批次大小 × 序列长度过大，可能导致内存不足')
        
        # 学习率与网络规模的关系
        if param_name == 'learning_rate' and 'hidden_units' in all_params:
            hidden_dim = int(all_params.get('hidden_units', 0) or 0)
            if value > 0.01 and hidden_dim > 256:
                result.warnings.append('大型网络建议使用较小的学习率(< 0.01)')
    
    def _get_parameter_display_name(self, param_name: str) -> str:
        """获取参数显示名称"""
        display_names = {
            'input_dim': '输入维度',
            'hidden_units': 'LSTM隐藏层维度',
            'num_layers': 'LSTM层数',
            'sequence_length': '序列长度',
            'batch_size': '批次大小',
            'prediction_horizon': '预测步长',
            'learning_rate': '学习率',
            'epochs': '训练轮数',
            'weight_decay': '权重衰减',
            'train_ratio': '训练集比例',
            'val_ratio': '验证集比例',
            'test_ratio': '测试集比例',
            'val_ratio_from_train': '验证集划分比例',
            'validation_split': '验证集划分比例',
            'dataset_mode': '数据集模式',
            'preprocess_method': '预处理方法',
            'num_filters': '卷积核数量',
            'kernel_size': '卷积核大小',
            'bottleneck_size': '瓶颈层维度',
            'num_conv_layers': '卷积层数',
            'stride': '步长'
        }
        
        return display_names.get(param_name, param_name)
    
    def validate_config_dict(self, config: Dict[str, Any]) -> ValidationResult:
        """验证配置字典（兼容现有接口）"""
        model_type = config.get('model_type', 'lstm_predictor')
        if not model_type and isinstance(config.get('model_config'), dict):
            model_type = config['model_config'].get('model_type', 'lstm_predictor')

        all_params: Dict[str, Any] = {}

        for section_key in ('model_config', 'training_config', 'dataset_config'):
            section = config.get(section_key)
            if isinstance(section, dict):
                all_params.update(section)

        meta_keys = {
            'module', 'model_type', 'task_id', 'status', 'message', 'progress',
            'logs', 'error', 'created_at', 'updated_at'
        }

        for key, value in config.items():
            if key in ('model_config', 'training_config', 'dataset_config'):
                continue
            if key in meta_keys:
                continue
            if key not in all_params:
                all_params[key] = value

        return self.validate_all_parameters(model_type or 'lstm_predictor', all_params)


# 创建全局验证器实例
validator = ParameterValidator()


def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """验证训练配置的便捷函数"""
    result = validator.validate_config_dict(config)
    
    return {
        'is_valid': result.is_valid,
        'errors': result.errors,
        'warnings': result.warnings,
        'suggestions': result.suggestions
    }