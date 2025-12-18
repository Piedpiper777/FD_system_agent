# LSTM异常检测系统

这是一个基于LSTM预测模型的异常检测系统。该系统专注于使用LSTM网络预测时间序列的未来值，并通过预测误差来识别异常。

## 主要特性

- **LSTM预测模型**: 使用LSTM编码器学习时间序列模式，并预测未来值
- **预测误差检测**: 通过计算预测值与实际值之间的误差来识别异常
- **灵活的训练评估流程**: 支持分离的训练和评估阶段
- **专业的评估系统**: 详细的异常检测指标和可视化
- **MindSpore框架**: 基于华为MindSpore深度学习框架实现
- **模块化设计**: 清晰的模块化架构，便于维护和扩展

## 🚀 新功能：可选测试数据和独立评估

### 核心改进
- **训练阶段可选测试数据**：训练时只需提供正常数据，无需测试数据
- **独立评估系统**：训练完成后单独加载测试数据进行异常检测评估
- **专业的评估工具**：`AnomalyEvaluator`类提供全面的评估功能

### 使用流程
```python
# 阶段1：训练（只需正常数据）
train_loader, val_loader, _ = quick_lstm_dataloaders(
    train_data_path="normal_data.csv"
)
# 训练模型...

# 阶段2：评估（单独加载测试数据）
from evaluation import evaluate_model
metrics = evaluate_model(
    model=trained_model,
    test_data_path="test_data.csv",
    test_labels_path="test_labels.csv"
)
```

## 项目结构

```
anomaly_detection/
├── core/                   # 核心组件
│   ├── models.py          # 模型定义(LSTMPredictor等)
│   ├── trainer.py         # 训练器(LSTMPredictorTrainer)
│   ├── dataset.py         # 数据集类(LSTMPredictorDataset，支持可选测试数据)
│   ├── evaluator.py       # 🆕 评估器(LSTMPredictionEvaluator，专业的LSTM预测异常检测评估)
│   └── inference.py       # 推理器(AnomalyDetector，实时异常检测)
├── data/                   # 数据处理模块
│   ├── multimode_manager.py   # 多模式数据集管理
│   ├── preprocessor.py        # 数据预处理工具
│   └── __init__.py
├── utils/                  # 工具模块
│   ├── config.py          # 配置管理
│   ├── metrics.py         # 评估指标
│   ├── logging.py         # 日志管理
│   └── __init__.py
├── cli/                    # 命令行接口
│   ├── train.py           # 训练命令
│   ├── inference.py       # 推理命令
│   ├── preprocess.py      # 预处理命令
│   └── __init__.py
├── tests/                  # 测试模块
│   ├── test_models.py     # 模型测试
│   ├── test_trainer.py    # 训练器测试
│   ├── test_dataset.py    # 数据集测试
│   ├── test_evaluator.py  # 🆕 评估器测试
│   ├── test_integration.py # 集成测试
│   └── __init__.py
├── __init__.py             # 包初始化和API导出
└── README.md              # 本文档
```

## 核心组件

### 模型工厂
- **get_model()**: 创建不同类型的模型
- **当前支持**: `lstm_predictor` - LSTM预测模型

### LSTMPredictor模型
- **输入**: 历史时间序列数据 (batch_size, seq_len, input_dim)
- **输出**: 预测的未来值 (batch_size, pred_len, input_dim)
- **架构**: LSTM编码器 + 全连接预测头

### BaseTrainer训练器基类
- **通用功能**: 优化器设置、梯度处理、验证、早停、最佳模型保存
- **可扩展**: 子类可以重写特定方法以实现自定义逻辑

### LSTMPredictorTrainer
- **继承**: BaseTrainer
- **特化**: 使用MindSpore `value_and_grad`进行高效梯度计算
- **损失函数**: 均方误差 (MSE) - 适合预测任务

### PredictionDataset
- **功能**: 创建输入序列-目标序列对用于监督学习
- **滑动窗口**: 支持可配置的序列长度和预测长度

## 使用方法

### 🆕 推荐用法：分离训练和评估

#### 阶段1：训练阶段（只需正常数据）
```python
from cloud.src.anomaly_detection.core.dataset import quick_lstm_dataloaders

# 只加载训练数据，测试数据可选
train_loader, val_loader, test_loader = quick_lstm_dataloaders(
    train_data_path="normal_train_data.csv",
    seq_len=100,
    batch_size=32
)

# test_loader 为 None，因为未提供测试数据
print(f"测试加载器: {test_loader}")  # None

# 训练模型
model = LSTMPredictor(input_dim=3, hidden_dim=64)
trainer = LSTMPredictorTrainer(model, train_loader, val_loader)
trainer.train(epochs=100)
trainer.save_model("trained_model.ckpt")
```

#### 阶段2：评估阶段（单独加载测试数据）
```python
from cloud.src.anomaly_detection.core.evaluator import evaluate_model

# 加载训练好的模型
model = LSTMPredictor.load("trained_model.ckpt")

# 一键评估异常检测性能
metrics = evaluate_model(
    model=model,
    test_data_path="test_data_with_anomalies.csv",
    test_labels_path="test_anomaly_labels.csv",
    seq_len=100,
    batch_size=32,
    output_dir="./evaluation_results"  # 保存详细结果
)

print(f"评估指标: {metrics}")
```

#### 高级用法：使用评估器类
```python
from cloud.src.anomaly_detection.core.evaluator import LSTMPredictionEvaluator

# 创建评估器
evaluator = LSTMPredictionEvaluator(model)

# 加载测试数据
evaluator.load_test_data(
    test_data_path="test_data.csv",
    test_labels_path="test_labels.csv",
    seq_len=100,
    batch_size=32
)

# 自定义阈值进行异常检测
predictions = evaluator.predict_anomalies(threshold=0.8)

# 获取详细评估指标
metrics = evaluator.calculate_metrics()

# 生成完整报告
evaluator.print_report()

# 保存结果和可视化
evaluator.save_results("./detailed_evaluation")
```

## 配置参数

### 通用参数
- `model_type`: 模型类型 (默认: 'lstm_predictor')
- `data_path`: 数据文件路径
- `input_dim`: 输入特征维度
- `seq_len`: 输入序列长度 (默认: 100)
- `batch_size`: 批次大小 (默认: 32)
- `test_size`: 测试集比例 (默认: 0.2)

### LSTM预测模型参数
- `hidden_dim`: LSTM隐藏层维度 (默认: 64)
- `num_layers`: LSTM层数 (默认: 2)
- `pred_len`: 预测序列长度 (默认: 1)
- `predictor_depth`: 预测头深度（可选，默认: 2，系统自动计算层大小）
- `predictor_activation`: 预测头激活函数（可选，默认: 'relu'，系统自动生成激活函数列表）
- `reduction_strategy`: 层大小递减策略（可选，默认: 'geometric'，选项: 'geometric'/'linear'）
- `prediction_mode`: 预测模式（可选，默认: 'last_timestep'，选项: 'last_timestep'/'mean_pooling'/'max_pooling'/'concat_layers'）

### 支持的激活函数
- `relu`: ReLU激活函数
- `tanh`: Tanh激活函数
- `sigmoid`: Sigmoid激活函数
- `leakyrelu`: LeakyReLU激活函数
- `none`: 无激活函数

### 层大小递减策略
- `geometric`: 几何递减，每层大小减半，快速压缩适合深层网络
- `linear`: 线性递减，均匀递减，平滑过渡适合浅层网络

### 预测模式
- `last_timestep`: 使用最后时间步的隐藏状态（默认，适合大多数场景）
- `mean_pooling`: 对所有时间步的输出做平均池化（捕捉全局特征）
- `max_pooling`: 对所有时间步的输出做最大池化（突出重要特征）
- `concat_layers`: 拼接所有LSTM层的隐藏状态（利用多层特征）

### 训练参数
- `learning_rate`: 学习率 (默认: 0.001)
- `weight_decay`: 权重衰减 (默认: 1e-4)
- `clip_grad_norm`: 梯度裁剪范数 (默认: 5.0)
- `num_epochs`: 训练轮数 (默认: 50)
- `patience`: 早停耐心值 (默认: 10)

## 异常检测原理

### 训练阶段
1. **正常数据学习**: LSTM学习正常时间序列的模式，能够预测未来值
2. **无监督学习**: 只使用正常数据训练，无需异常标签
3. **模式捕获**: 学习时间序列的正常波动和周期性特征

### 评估阶段
1. **测试数据加载**: 单独加载包含正常+异常数据的测试集
2. **预测误差计算**: 使用训练好的模型预测测试数据
3. **异常识别**: 计算预测误差，超过阈值视为异常
4. **阈值确定**: 默认使用训练数据重构误差的95%分位数

### 评估指标
- **准确率/精确率/召回率/F1分数**: 分类性能指标
- **AUC/PR-AUC**: 排名性能指标
- **混淆矩阵**: 详细分类结果
- **异常分数分布**: 可视化分析

### 数据泄露防护
- 预处理器只使用训练数据拟合
- 测试数据使用训练数据的预处理器进行变换
- 确保评估的公平性和可靠性

## 依赖项

- mindspore >= 2.0
- numpy
- pandas
- scikit-learn

## 注意事项

- 系统采用模块化设计，支持扩展新的模型类型
- 当前支持 `lstm_predictor` 模型，未来可轻松添加其他模型
- 确保输入数据格式正确：CSV或TXT文件，第一行为列名
- 建议对数据进行适当的预处理（如标准化）以获得更好的性能
- **训练器架构**：`BaseTrainer` 只提供最核心的通用功能（如优化器设置、梯度处理），具体训练逻辑由子类实现
- **定制化原则**：损失函数、验证逻辑、早停机制等都由专用训练器（如 `LSTMPredictorTrainer`）负责

## 🆕 迁移指南

### 从旧版本升级
之前的版本要求在初始化时必须提供测试数据和标签。现在支持更灵活的使用方式：

#### 旧代码（强制要求测试数据）
```python
# 旧版本 - 必须提供测试数据
dataset = LSTMPredictorDataset(
    train_data_path="train.csv",
    test_data_path="test.csv",      # 必需
    test_labels_path="labels.csv",  # 必需
)
```

#### 新代码（测试数据可选）
```python
# 新版本 - 训练阶段可选测试数据
dataset = LSTMPredictorDataset(
    train_data_path="train.csv"
    # test_data_path 和 test_labels_path 现在是可选的
)

# 评估阶段单独进行
from cloud.src.anomaly_detection.core.evaluator import evaluate_model
metrics = evaluate_model(model, "test.csv", "labels.csv")
```

### 主要优势
1. **更真实的异常检测流程**：训练只用正常数据，评估用混合数据
2. **避免数据泄露**：预处理器只用训练数据拟合
3. **灵活的开发流程**：可以专注于训练，然后单独评估
4. **专业的评估工具**：详细的指标和可视化分析