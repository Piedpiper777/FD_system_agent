# LSTM预测异常检测模块

## 概述

这是基于LSTM的时间序列异常检测模块，采用全新的**模型-方法架构**设计，提供完整的工业异常检测解决方案。

## 架构设计

### 模型-方法架构

与传统的功能分层架构（models.py, trainer.py, dataset.py, evaluator.py, inference.py）不同，新架构采用**模型-方法分层**：

```
core/
├── lstm_predictor/           # LSTM预测异常检测模块
│   ├── __init__.py          # 模块导出
│   ├── model.py             # LSTM模型定义
│   ├── trainer.py           # 专用训练器
│   ├── dataset.py           # 时序数据处理
│   ├── evaluator.py         # 异常检测评估器
│   └── inference.py         # 实时推理引擎
├── autoencoder/             # (计划中) 自编码器异常检测
├── statistical/             # (计划中) 统计方法异常检测
└── ...
```

### 核心优势

1. **模块化设计**: 每个异常检测算法都是独立的模块
2. **即插即用**: 轻松添加新的异常检测方法
3. **统一接口**: 所有模块使用相同的API设计
4. **向后兼容**: 保持与现有代码的兼容性

## 快速开始

### 基本使用

```python
from core.lstm_predictor import (
    LSTMPredictor, create_model, get_default_config,
    LSTMPredictorTrainer, create_trainer,
    LSTMPredictorDataset, create_dataset,
    LSTMPredictorEvaluator, create_evaluator,
    LSTMPredictorInference, create_inference_engine
)

# 1. 创建模型
config = get_default_config('lstm_predictor')
config['input_dim'] = 5  # 5个传感器特征
model = create_model('lstm_predictor', **config)

# 2. 准备数据
dataset = create_dataset('lstm_predictor', data, sequence_length=50)

# 3. 创建训练器并训练
trainer = create_trainer('lstm_predictor', model)
trainer.train(dataset.get_train_loader(), num_epochs=50)

# 4. 创建评估器并设置阈值
evaluator = create_evaluator('lstm_predictor')
evaluator.fit_threshold(predictions, actuals)

# 5. 创建推理引擎进行实时检测
inference_engine = create_inference_engine('lstm_predictor', model, evaluator)

# 6. 异常检测
result = inference_engine.detect_anomaly(input_sequence)
print(f"异常分数: {result['anomaly_score']:.4f}, 是否异常: {result['is_anomaly']}")
```

### 工厂函数模式

```python
from core.lstm_predictor import (
    create_model, create_trainer, create_dataset,
    create_evaluator, create_inference_engine
)

# 使用工厂函数统一创建
model = create_model('lstm_predictor', input_dim=5)
trainer = create_trainer('lstm_predictor', model)
dataset = create_dataset('lstm_predictor', data)
evaluator = create_evaluator('lstm_predictor')
inference = create_inference_engine('lstm_predictor', model, evaluator)
```

### 向后兼容性

```python
# 旧的导入方式仍然有效
from core import LSTMPredictor, LSTMPredictorTrainer, LSTMPredictionEvaluator, AnomalyDetector
```

## 组件详解

### LSTMPredictor 模型

基于LSTM的时间序列预测模型：

- **输入**: `(batch_size, sequence_length, n_features)`
- **输出**: `(batch_size, n_features)` - 下一时刻预测
- **异常检测**: 通过预测误差识别异常

```python
model = LSTMPredictor(
    input_dim=5,      # 特征维度
    hidden_dim=128,   # LSTM隐藏层
    num_layers=2,     # LSTM层数
    dropout=0.1       # Dropout率
)
```

### LSTMPredictorTrainer 训练器

专门为LSTM异常检测优化的训练器：

- MSE损失函数
- 梯度裁剪
- **MindSpore内置ReduceLROnPlateau学习率调度器**
- 残差分析
- 早停机制

#### 学习率调度器特性

使用MindSpore内置的`ReduceLROnPlateau`回调：

```python
trainer = LSTMPredictorTrainer(
    model=model,
    learning_rate=0.001,
    use_lr_scheduler=True,        # 启用学习率调度
    patience_for_scheduler=10     # 耐心值：10轮无改善后衰减
)
```

**调度器参数**:
- `monitor='val_loss'`: 监控验证损失
- `factor=0.5`: 学习率衰减因子
- `patience`: 无改善轮数阈值
- `min_lr=1e-6`: 最小学习率
- `verbose=True`: 打印调度信息

### LSTMPredictorDataset 数据集

时序数据预处理和加载：

- 滑动窗口序列生成
- 数据标准化
- 训练/验证/测试分割
- MindSpore Dataset兼容

### LSTMPredictorEvaluator 评估器

异常检测评估和阈值管理：

- 动态阈值计算
- 多种评估指标
- 可视化分析
- 置信度计算

### LSTMPredictorInference 推理器

实时异常检测引擎：

- 单样本推理
- 批量推理
- 流式检测
- 性能监控

## 配置管理

### 默认配置

```python
from core.lstm_predictor import (
    get_default_config, get_default_dataset_config,
    get_default_evaluator_config, get_default_inference_config
)

# 获取各组件默认配置
model_config = get_default_config('lstm_predictor')
dataset_config = get_default_dataset_config('lstm_predictor')
evaluator_config = get_default_evaluator_config('lstm_predictor')
inference_config = get_default_inference_config('lstm_predictor')
```

### 从配置创建

```python
from core.lstm_predictor import (
    create_model_from_config, create_dataset_from_config,
    create_evaluator_from_config, create_inference_from_config
)

config = {
    'model_type': 'lstm_predictor',
    'input_dim': 5,
    'hidden_dim': 128,
    'sequence_length': 50,
    # ... 其他参数
}

model = create_model_from_config(config)
dataset = create_dataset_from_config(config, data)
evaluator = create_evaluator_from_config(config)
inference = create_inference_from_config(config, model, evaluator)
```

## 工业应用特性

### 实时检测

```python
# 流式异常检测
inference_engine = create_inference_engine('lstm_predictor', model, evaluator)

# 逐个数据点检测
for data_point in streaming_data:
    result = inference_engine.streaming_detect(data_point)
    if result['is_anomaly']:
        print(f"检测到异常! 分数: {result['anomaly_score']:.4f}")
```

### 批量处理

```python
# 批量异常检测
batch_results = inference_engine.batch_detect_anomalies(
    input_sequences, actual_next_values
)

for result in batch_results:
    print(f"样本 {result['batch_idx']}: 异常={result['is_anomaly']}")
```

### 性能监控

```python
# 获取性能统计
stats = inference_engine.get_performance_stats()
print(f"平均推理时间: {stats['avg_inference_time']:.4f}s")
print(f"异常检测率: {stats['anomaly_detection_rate']:.2%}")
```

## 扩展新算法

添加新的异常检测算法非常简单：

1. 创建新的模块目录（如 `core/new_method/`）
2. 实现对应的组件文件
3. 更新 `__init__.py` 导出
4. 添加工厂函数支持

```python
# core/new_method/__init__.py
from .model import NewPredictor, create_model
from .trainer import NewTrainer, create_trainer
# ... 其他组件

__all__ = ['NewPredictor', 'create_model', 'NewTrainer', 'create_trainer', ...]
```

## 测试和验证

运行完整测试：

```bash
cd cloud/src/anomaly_detection
python test_lstm_predictor.py
```

## 迁移指南

### 从旧架构迁移

如果您使用的是旧的功能分层架构，可以逐步迁移：

1. **保持兼容**: 新架构保持了向后兼容性
2. **逐步替换**: 可以逐个组件替换
3. **测试验证**: 每个迁移步骤后运行测试

### 主要变化

- `models.py` → `lstm_predictor/model.py`
- `trainer.py` → `lstm_predictor/trainer.py`
- `dataset.py` → `lstm_predictor/dataset.py`
- `evaluator.py` → `lstm_predictor/evaluator.py`
- `inference.py` → `lstm_predictor/inference.py`

## 版本信息

- **版本**: 1.0.0
- **框架**: MindSpore
- **Python**: 3.7+
- **许可证**: MIT

## 贡献

欢迎提交Issue和Pull Request来改进这个模块！

---

*该模块专为工业物联网异常检测场景优化，支持多维传感器数据、实时检测和高可靠性要求。*