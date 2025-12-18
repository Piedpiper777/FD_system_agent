#!/usr/bin/env python3
"""
LSTM异常检测系统 - 命令行训练脚本 (Server Machine Dataset)

专门用于训练Server Machine Dataset的LSTM异常检测模型

数据集格式：
- 训练数据：timestamp,col_0,col_1,...,col_37 (38个特征列)
- 测试数据：timestamp,col_0,col_1,...,col_37,label (38个特征列 + 1个标签列)

使用方法：
python cli_train.py --data_path /path/to/machine-1-1_train.csv --model_path models/machine_model.ckpt
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import mindspore as ms

# 添加项目路径
current_dir = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(current_dir))

from src.anomaly_detection.core.lstm_predicton.data_processor import DataProcessor, TimeSeriesData
from src.anomaly_detection.core.lstm_predicton.model_builder import ModelBuilder
from src.anomaly_detection.core.lstm_predicton.trainer import Trainer
from src.anomaly_detection.core.lstm_predicton.threshold_calculator import ThresholdCalculator


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LSTM异常检测系统 - Server Machine Dataset训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据集信息：
  - 训练数据格式：timestamp,col_0,col_1,...,col_37 (38个特征列)
  - 测试数据格式：timestamp,col_0,col_1,...,col_37,label (38个特征列 + 标签列)
  - 数据已预处理为0-1范围

使用示例：
  python cli_train.py --data_path data/machine-1-1_train.csv --model_path models/machine_model.ckpt

  python cli_train.py --data_path data/machine-1-1_train.csv --model_path models/machine_model.ckpt \\
                      --sequence_length 100 --epochs 50 --batch_size 64
        """
    )

    # 数据相关参数
    parser.add_argument('--data_path', type=str, required=True,
                       help='训练数据文件路径 (CSV格式)')
    parser.add_argument('--timestamp_column', type=str, default='timestamp',
                       help='时间戳列名 (默认: timestamp)')

    # 模型相关参数
    parser.add_argument('--sequence_length', type=int, default=50,
                       help='序列长度 (默认: 50)')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                       help='预测步长 (默认: 1)')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='LSTM隐藏单元数 (默认: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数 (默认: 2)')

    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小 (默认: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数 (默认: 30)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值 (默认: 10)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例 (默认: 0.8)')

    # 输出相关参数
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型保存路径 (默认: models/anomaly_detection/lstm/lstm_anomaly_model_YYYYMMDD_HHMMSS.ckpt)')
    parser.add_argument('--threshold_path', type=str, default=None,
                       help='阈值保存路径 (默认: model_path同目录的threshold.json)')
    parser.add_argument('--scaler_path', type=str, default=None,
                       help='标准化参数保存路径 (默认: model_path同目录的scaler.npz)')

    # 其他参数
    parser.add_argument('--device_target', type=str, default='CPU',
                       choices=['CPU', 'GPU', 'Ascend'],
                       help='运行设备 (默认: CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出模式')

    return parser.parse_args()


def setup_mindspore_context(device_target: str, seed: int):
    """设置MindSpore上下文"""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device(device_target)
    ms.set_seed(seed)
    print(f"🔧 MindSpore上下文设置完成: 设备={device_target}, 种子={seed}")


def load_and_preprocess_data(args) -> Tuple[TimeSeriesData, TimeSeriesData, DataProcessor]:
    """加载和预处理数据"""
    print("📊 开始加载和预处理Server Machine数据...")

    # 初始化数据处理器
    processor = DataProcessor(
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        normalize=False  # 数据已经标准化
    )

    # 加载数据并处理流水线
    try:
        train_data, val_data = processor.process_pipeline(
            args.data_path,
            train_ratio=args.train_ratio
        )

        print("✅ 数据处理完成")
        print(f"  - 训练集: {len(train_data.sequences)} 序列")
        print(f"  - 验证集: {len(val_data.sequences)} 序列")
        print(f"  - 特征数: {train_data.sequences.shape[2]}")
        print(f"  - 序列长度: {args.sequence_length}")
        print(f"  - 数据范围: [{train_data.sequences.min():.3f}, {train_data.sequences.max():.3f}]")

        return train_data, val_data, processor

    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        sys.exit(1)


def create_and_train_model(train_data: TimeSeriesData, val_data: TimeSeriesData,
                          processor: DataProcessor, args) -> Trainer:
    """创建和训练模型"""
    print("\n🤖 开始创建和训练LSTM模型...")

    # 创建模型
    input_shape = (args.sequence_length, train_data.sequences.shape[2])
    model = ModelBuilder.create_model(
        'lstm_predictor',
        input_shape=input_shape,
        hidden_units=args.hidden_size,
        num_layers=args.num_layers
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate
    )

    # 创建数据集
    train_dataset = ms.dataset.NumpySlicesDataset(
        {'sequences': train_data.sequences, 'targets': train_data.targets},
        shuffle=True
    ).batch(args.batch_size)

    val_dataset = ms.dataset.NumpySlicesDataset(
        {'sequences': val_data.sequences, 'targets': val_data.targets},
        shuffle=False
    ).batch(args.batch_size)

    # 训练模型
    print(f"🚀 开始训练: {args.epochs}轮, 批大小={args.batch_size}, 学习率={args.learning_rate}")
    trained_model = trainer.train(
        train_loader=train_dataset,
        num_epochs=args.epochs,
        val_loader=val_dataset,
        patience=args.patience
    )

    # 获取训练指标
    metrics = trainer.get_training_metrics()
    print("✅ 训练完成")
    print(f"  - 实际训练轮数: {metrics['epochs_trained']}")
    print(f"  - 最终训练损失: {metrics.get('final_train_loss', 'N/A')}")
    print(f"  - 最终验证损失: {metrics.get('final_val_loss', 'N/A')}")

    return trainer


def calculate_and_save_threshold(trainer: Trainer, train_data: TimeSeriesData,
                               threshold_path: str):
    """计算并保存异常检测阈值"""
    print("\n🎯 开始计算异常检测阈值...")

    # 从训练数据中采样预测结果来计算阈值
    model = trainer.model

    # 为了计算阈值，我们使用训练数据的前N个序列进行预测
    n_samples = min(2000, len(train_data.sequences))  # 最多使用2000个样本
    sample_sequences = train_data.sequences[:n_samples]
    sample_targets = train_data.targets[:n_samples]

    predictions = []
    for i in range(0, len(sample_sequences), 32):  # 批处理预测
        batch_seq = sample_sequences[i:i+32]
        batch_tensor = ms.Tensor(batch_seq, ms.float32)
        batch_pred = model(batch_tensor)
        predictions.extend(batch_pred.asnumpy())

    predictions = np.array(predictions)
    actuals = sample_targets

    # 创建阈值计算器
    threshold_calculator = ThresholdCalculator(residual_method='l2_norm')

    # 计算阈值
    threshold = threshold_calculator.fit_threshold(
        predictions, actuals,
        method='percentile', percentile=95.0
    )

    # 保存阈值
    threshold_calculator.save_threshold(threshold_path)
    print(f"✅ 阈值计算完成: {threshold:.6f}")
    print(f"💾 阈值已保存: {threshold_path}")

    return threshold_calculator


def save_scaler_params(processor: DataProcessor, scaler_path: str):
    """保存标准化参数"""
    scaler_params = processor.get_scaler_params()
    processor.save_scaler_params(scaler_path)
    print(f"💾 标准化参数已保存: {scaler_path}")


def save_model(trainer: Trainer, model_path: str):
    """保存模型"""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    ms.save_checkpoint(trainer.model, str(model_path))
    print(f"💾 模型已保存: {model_path}")


def save_training_config(args, config_path: str):
    """保存训练配置"""
    config = {
        'data_path': args.data_path,
        'timestamp_column': args.timestamp_column,
        'sequence_length': args.sequence_length,
        'prediction_horizon': args.prediction_horizon,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'train_ratio': args.train_ratio,
        'device_target': args.device_target,
        'seed': args.seed,
        'dataset_info': {
            'name': 'Server Machine Dataset',
            'features': 38,
            'description': '服务器机器异常检测数据集，已预处理'
        }
    }

    config_path = Path(config_path)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"💾 训练配置已保存: {config_path}")


def main():
    """主函数"""
    print("🤖 LSTM异常检测系统 - Server Machine Dataset训练脚本")
    print("=" * 70)

    # 解析命令行参数
    args = parse_arguments()

    # 设置输出路径 - 优化版本
    if args.model_path is None:
        # 默认保存到 anomaly_detection/lstm 目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path("models/anomaly_detection/lstm")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"lstm_anomaly_model_{timestamp}.ckpt"
    else:
        model_path = Path(args.model_path)

    # 如果没有指定完整路径，自动创建带时间戳的目录结构
    if not args.threshold_path and not args.scaler_path:
        # 创建模型保存目录（如果不存在）
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # 生成带时间戳的模型文件名（如果用户没有指定具体文件名）
        if model_path.name == model_path.parent.name or not model_path.suffix:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"lstm_anomaly_model_{timestamp}.ckpt"
            model_path = model_path.parent / model_name

        # 设置相关文件路径
        base_name = model_path.stem  # 不含扩展名的文件名
        if args.threshold_path is None:
            args.threshold_path = str(model_path.parent / f"{base_name}_threshold.json")
        if args.scaler_path is None:
            args.scaler_path = str(model_path.parent / f"{base_name}_scaler.npz")

    config_path = str(model_path.parent / f"{model_path.stem}_config.json")

    print(f"📁 输出目录: {model_path.parent}")
    print(f"📄 模型路径: {model_path}")
    print(f"📄 阈值路径: {args.threshold_path}")
    print(f"📄 标准化参数路径: {args.scaler_path}")
    print()

    try:
        # 设置MindSpore
        setup_mindspore_context(args.device_target, args.seed)

        # 加载和预处理数据
        train_data, val_data, processor = load_and_preprocess_data(args)

        # 创建和训练模型
        trainer = create_and_train_model(train_data, val_data, processor, args)

        # 保存模型
        save_model(trainer, str(model_path))

        # 计算并保存阈值
        threshold_calculator = calculate_and_save_threshold(
            trainer, train_data, args.threshold_path
        )

        # 保存标准化参数
        save_scaler_params(processor, args.scaler_path)

        # 保存训练配置
        save_training_config(args, config_path)

        print("\n🎉 训练流程完成！")
        print("=" * 70)
        print("📊 训练总结:")
        print(f"  - 训练数据: {args.data_path}")
        print(f"  - 数据集: Server Machine Dataset (38个特征)")
        print(f"  - 模型保存: {model_path}")
        print(f"  - 阈值保存: {args.threshold_path}")
        print(f"  - 标准化参数: {args.scaler_path}")
        print(f"  - 训练配置: {config_path}")
        print("\n🚀 现在可以使用训练好的模型进行异常检测！")
        print(f"   python cli_evaluate.py --model_path {model_path} --test_data_path your_test_data.csv")

    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 训练过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()