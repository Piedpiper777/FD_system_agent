#!/usr/bin/env python3
"""
LSTM异常检测系统 - Server Machine Dataset评估脚本
用于评估训练好的LSTM模型在测试数据上的性能
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict

import mindspore as ms
import numpy as np
from sklearn.metrics import classification_report

# 导入项目模块
import sys
from pathlib import Path

# 从cli目录向上找到cloud目录: cli -> lstm_predicton -> core -> anomaly_detection -> src -> cloud
cloud_dir = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(cloud_dir))

from src.anomaly_detection.core.lstm_predicton.data_processor import DataProcessor, TimeSeriesData
from src.anomaly_detection.core.lstm_predicton.model_builder import ModelBuilder
from src.anomaly_detection.core.lstm_predicton.evaluator import Evaluator
from src.anomaly_detection.core.lstm_predicton.threshold_calculator import ThresholdCalculator


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LSTM异常检测系统 - 模型评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python cli_evaluate.py --model_path models/test_machine_model.ckpt --test_data_path data/test.csv
  python cli_evaluate.py --model_path models/model.ckpt --test_data_path data/test.csv --threshold_path models/threshold.json --scaler_path models/scaler.npz --output_dir results/
        """
    )

    # 必需参数
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='训练好的模型文件路径 (.ckpt)，默认在 models/anomaly_detection/lstm/ 目录中查找最新的模型'
    )

    parser.add_argument(
        '--test_data_path',
        type=str,
        required=True,
        help='测试数据文件路径 (.csv)'
    )

    # 可选参数
    parser.add_argument(
        '--threshold_path',
        type=str,
        default=None,
        help='异常检测阈值文件路径 (.json)，如果不提供则自动从模型目录查找'
    )

    parser.add_argument(
        '--scaler_path',
        type=str,
        default=None,
        help='标准化参数文件路径 (.npz)，如果不提供则自动从模型目录查找'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='评估结果输出目录 (默认: model_path/evaluations)，使用"auto"自动创建带时间戳的目录'
    )

    parser.add_argument(
        '--sequence_length',
        type=int,
        default=50,
        help='序列长度 (默认: 50)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批处理大小 (默认: 32)'
    )

    parser.add_argument(
        '--timestamp_column',
        type=str,
        default='timestamp',
        help='时间戳列名 (默认: timestamp)'
    )

    parser.add_argument(
        '--label_column',
        type=str,
        default='label',
        help='标签列名，用于真实异常标签 (默认: label)'
    )

    parser.add_argument(
        '--device_target',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU', 'Ascend'],
        help='运行设备 (默认: CPU)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='启用详细输出'
    )

    return parser.parse_args()


def setup_mindspore_context(device_target: str = 'CPU', seed: int = 42):
    """设置MindSpore上下文"""
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_device(device_target)
    ms.set_seed(seed)
    print(f"🔧 MindSpore上下文设置完成: 设备={device_target}, 种子={seed}")


def load_model_and_components(model_path: str, threshold_path: Optional[str] = None,
                            scaler_path: Optional[str] = None) -> Tuple[ms.nn.Cell, ThresholdCalculator, DataProcessor]:
    """
    加载模型和相关组件

    Args:
        model_path: 模型文件路径
        threshold_path: 阈值文件路径
        scaler_path: 标准化参数文件路径

    Returns:
        模型、阈值计算器、数据处理器
    """
    model_path = Path(model_path)

    # 如果没有指定路径，自动从模型目录查找
    if threshold_path is None:
        threshold_path = model_path.parent / "threshold.json"
    else:
        threshold_path = Path(threshold_path)

    if scaler_path is None:
        scaler_path = model_path.parent / "scaler.npz"
    else:
        scaler_path = Path(scaler_path)

    print(f"📂 加载模型和组件...")
    print(f"  - 模型路径: {model_path}")
    print(f"  - 阈值路径: {threshold_path}")
    print(f"  - 标准化参数路径: {scaler_path}")

    # 加载模型
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 首先创建模型结构（需要从配置中获取参数）
    # 这里简化处理，使用默认参数
    model_builder = ModelBuilder()
    model = model_builder.build_lstm_predictor(
        input_shape=(50, 38),  # 基于训练配置
        hidden_units=128,
        num_layers=2
    )

    # 加载权重
    param_dict = ms.load_checkpoint(str(model_path))
    ms.load_param_into_net(model, param_dict)
    print(f"✅ 模型加载完成: {model_path}")

    # 加载阈值计算器
    if not threshold_path.exists():
        print(f"⚠️ 阈值文件不存在: {threshold_path}，将使用默认阈值")
        threshold_calculator = ThresholdCalculator(residual_method='l2_norm')
        threshold_calculator.threshold = 0.1  # 默认阈值
    else:
        threshold_calculator = ThresholdCalculator(residual_method='l2_norm')
        threshold_calculator.load_threshold(str(threshold_path))
        print(f"✅ 阈值加载完成: {threshold_path} (阈值: {threshold_calculator.threshold:.6f})")

    # 加载数据处理器
    processor = DataProcessor(
        sequence_length=50,
        prediction_horizon=1,
        normalize=False  # 测试时不进行标准化
    )

    if scaler_path.exists():
        processor.load_scaler_params(str(scaler_path))
        print(f"✅ 标准化参数加载完成: {scaler_path}")
    else:
        print(f"⚠️ 标准化参数文件不存在: {scaler_path}，将使用原始数据")

    return model, threshold_calculator, processor


def load_and_preprocess_test_data(data_path: str, processor: DataProcessor,
                                timestamp_column: str = 'timestamp',
                                label_column: str = 'label') -> Tuple[TimeSeriesData, np.ndarray]:
    """
    加载和预处理测试数据

    Args:
        data_path: 测试数据路径
        processor: 数据处理器
        timestamp_column: 时间戳列名
        label_column: 标签列名

    Returns:
        处理后的数据和真实标签
    """
    print(f"📊 开始加载和预处理测试数据...")
    print(f"  - 数据路径: {data_path}")
    print(f"  - 时间戳列: {timestamp_column}")
    print(f"  - 标签列: {label_column}")

    # 加载数据
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"测试数据文件不存在: {data_path}")

    # 加载CSV数据
    df = processor.load_data(str(data_path), timestamp_column=timestamp_column, label_column=label_column)

    # 从processor中获取识别的列信息
    timestamp_col = processor.timestamp_column
    feature_cols = processor.feature_names
    label_col = processor.label_column

    print(f"🔍 列识别完成:")
    print(f"  - 时间戳列: {timestamp_col}")
    print(f"  - 特征列: {feature_cols}")
    print(f"  - 标签列: {label_col}")

    # 提取特征和标签
    features = df[feature_cols].values
    true_labels = df[label_col].values if label_col else np.zeros(len(df))

    print(f"📊 数据加载完成: {data_path}")
    print(f"  - 数据形状: {df.shape}")
    print(f"  - 特征数: {len(feature_cols)}")
    print(f"  - 异常样本数: {np.sum(true_labels)} / {len(true_labels)} ({np.sum(true_labels)/len(true_labels)*100:.2f}%)")

    # 创建序列数据
    time_series_data = processor.create_sequences(features)
    sequences = time_series_data.sequences
    targets = time_series_data.targets
    print(f"🔄 序列创建完成")
    print(f"  - 生成序列数: {len(sequences)}")
    print(f"  - 序列形状: {sequences.shape}")

    # 创建TimeSeriesData对象
    test_data = time_series_data

    return test_data, true_labels[len(true_labels) - len(sequences):]  # 对齐序列长度


def perform_inference(model: ms.nn.Cell, test_data: TimeSeriesData,
                     batch_size: int = 32) -> np.ndarray:
    """
    执行推理

    Args:
        model: 训练好的模型
        test_data: 测试数据
        batch_size: 批处理大小

    Returns:
        预测结果
    """
    print(f"🔮 开始模型推理...")

    model.set_train(False)
    predictions = []

    # 分批处理
    for i in range(0, len(test_data.sequences), batch_size):
        batch_sequences = test_data.sequences[i:i+batch_size]
        batch_tensor = ms.Tensor(batch_sequences, ms.float32)

        batch_pred = model(batch_tensor)
        predictions.extend(batch_pred.asnumpy())

    predictions = np.array(predictions)
    print(f"✅ 推理完成: {len(predictions)} 个预测结果")

    return predictions


def calculate_anomaly_scores(predictions: np.ndarray, actuals: np.ndarray,
                           method: str = 'l2_norm') -> np.ndarray:
    """
    计算异常分数

    Args:
        predictions: 预测值
        actuals: 实际值
        method: 计算方法

    Returns:
        异常分数
    """
    if method == 'l2_norm':
        # L2范数（欧几里得距离）
        scores = np.linalg.norm(predictions - actuals, axis=1)
    elif method == 'l1_norm':
        # L1范数（曼哈顿距离）
        scores = np.sum(np.abs(predictions - actuals), axis=1)
    elif method == 'mse':
        # 均方误差
        scores = np.mean((predictions - actuals) ** 2, axis=1)
    else:
        raise ValueError(f"不支持的异常分数计算方法: {method}")

    return scores


def evaluate_model(predictions: np.ndarray, actuals: np.ndarray, true_labels: np.ndarray,
                  threshold: float, anomaly_scores: np.ndarray) -> Dict:
    """
    评估模型性能

    Args:
        predictions: 预测值
        actuals: 实际值
        true_labels: 真实标签
        threshold: 异常检测阈值
        anomaly_scores: 异常分数

    Returns:
        评估结果字典
    """
    print(f"📊 开始模型评估...")

    # 创建评估器
    evaluator = Evaluator()

    # 评估预测性能
    metrics = evaluator.evaluate(
        predictions=predictions,
        actuals=actuals,
        true_labels=true_labels,
        anomaly_scores=anomaly_scores
    )

    # 基于阈值的异常检测
    pred_anomalies = (anomaly_scores > threshold).astype(int)

    # 计算异常检测指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    detection_accuracy = accuracy_score(true_labels, pred_anomalies)
    detection_precision = precision_score(true_labels, pred_anomalies, zero_division=0)
    detection_recall = recall_score(true_labels, pred_anomalies, zero_division=0)
    detection_f1 = f1_score(true_labels, pred_anomalies, zero_division=0)

    # 扩展评估结果
    evaluation_results = {
        'prediction_metrics': metrics,
        'anomaly_detection': {
            'threshold': float(threshold),
            'accuracy': float(detection_accuracy),
            'precision': float(detection_precision),
            'recall': float(detection_recall),
            'f1_score': float(detection_f1),
            'predicted_anomalies': int(np.sum(pred_anomalies)),
            'true_anomalies': int(np.sum(true_labels))
        },
        'anomaly_scores': {
            'mean': float(np.mean(anomaly_scores)),
            'std': float(np.std(anomaly_scores)),
            'min': float(np.min(anomaly_scores)),
            'max': float(np.max(anomaly_scores)),
            'percentiles': {
                '50': float(np.percentile(anomaly_scores, 50)),
                '75': float(np.percentile(anomaly_scores, 75)),
                '90': float(np.percentile(anomaly_scores, 90)),
                '95': float(np.percentile(anomaly_scores, 95)),
                '99': float(np.percentile(anomaly_scores, 99))
            }
        }
    }

    print(f"✅ 评估完成")
    print(f"  - 预测准确率: {metrics.get('accuracy', 0):.4f}")
    print(f"  - 异常检测准确率: {detection_accuracy:.4f}")
    print(f"  - 异常检测精确率: {detection_precision:.4f}")
    print(f"  - 异常检测召回率: {detection_recall:.4f}")
    print(f"  - 异常检测F1分数: {detection_f1:.4f}")

    return evaluation_results


def save_evaluation_results(results: Dict, output_dir: str, model_name: str):
    """
    保存评估结果

    Args:
        results: 评估结果
        output_dir: 输出目录
        model_name: 模型名称
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细结果
    results_path = output_dir / f"{model_name}_evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 评估结果已保存: {results_path}")

    # 保存性能摘要
    summary_path = output_dir / f"{model_name}_evaluation_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("LSTM异常检测模型评估报告\n")
        f.write("=" * 50 + "\n\n")

        # 预测性能
        pred_metrics = results['prediction_metrics']
        f.write("预测性能指标:\n")
        f.write(f"  样本总数: {pred_metrics['total_samples']}\n")
        f.write(f"  异常样本数: {pred_metrics['n_anomalies']}\n")
        f.write(f"  异常比例: {pred_metrics['anomaly_ratio']:.4f}\n")
        f.write(f"  准确率: {pred_metrics['accuracy']:.4f}\n")
        f.write(f"  精确率: {pred_metrics['precision']:.4f}\n")
        f.write(f"  召回率: {pred_metrics['recall']:.4f}\n")
        f.write(f"  F1分数: {pred_metrics['f1_score']:.4f}\n")
        if pred_metrics['auc']:
            f.write(f"  AUC: {pred_metrics['auc']:.4f}\n")
        f.write("\n")

        # 异常检测性能
        det_metrics = results['anomaly_detection']
        f.write("异常检测性能指标:\n")
        f.write(f"  检测阈值: {det_metrics['threshold']:.6f}\n")
        f.write(f"  准确率: {det_metrics['accuracy']:.4f}\n")
        f.write(f"  精确率: {det_metrics['precision']:.4f}\n")
        f.write(f"  召回率: {det_metrics['recall']:.4f}\n")
        f.write(f"  F1分数: {det_metrics['f1_score']:.4f}\n")
        f.write(f"  预测异常数: {det_metrics['predicted_anomalies']}\n")
        f.write(f"  真实异常数: {det_metrics['true_anomalies']}\n")
        f.write("\n")

        # 异常分数统计
        score_stats = results['anomaly_scores']
        f.write("异常分数统计:\n")
        f.write(f"  均值: {score_stats['mean']:.6f}\n")
        f.write(f"  标准差: {score_stats['std']:.6f}\n")
        f.write(f"  最小值: {score_stats['min']:.6f}\n")
        f.write(f"  最大值: {score_stats['max']:.6f}\n")
        f.write("  分位数:\n")
        for p, v in score_stats['percentiles'].items():
            f.write(f"    {p}%: {v:.6f}\n")

    print(f"💾 评估摘要已保存: {summary_path}")


def print_evaluation_summary(results: Dict):
    """打印评估摘要"""
    print("\n" + "=" * 70)
    print("🎯 LSTM异常检测模型评估结果")
    print("=" * 70)

    # 预测性能
    pred_metrics = results['prediction_metrics']
    print("\n📈 预测性能指标:")
    print(f"  样本总数: {pred_metrics['total_samples']}")
    print(f"  异常样本数: {pred_metrics['n_anomalies']} ({pred_metrics['anomaly_ratio']:.2f}%)")
    print(f"  准确率: {pred_metrics['accuracy']:.4f}")
    print(f"  精确率: {pred_metrics['precision']:.4f}")
    print(f"  召回率: {pred_metrics['recall']:.4f}")
    print(f"  F1分数: {pred_metrics['f1_score']:.4f}")
    if pred_metrics['auc']:
        print(f"  AUC: {pred_metrics['auc']:.4f}")

    # 异常检测性能
    det_metrics = results['anomaly_detection']
    print("\n🚨 异常检测性能指标:")
    print(f"  检测阈值: {det_metrics['threshold']:.6f}")
    print(f"  准确率: {det_metrics['accuracy']:.4f}")
    print(f"  精确率: {det_metrics['precision']:.4f}")
    print(f"  召回率: {det_metrics['recall']:.4f}")
    print(f"  F1分数: {det_metrics['f1_score']:.4f}")
    print(f"  预测异常数: {det_metrics['predicted_anomalies']}")
    print(f"  真实异常数: {det_metrics['true_anomalies']}")

    # 异常分数统计
    score_stats = results['anomaly_scores']
    print("\n📊 异常分数统计:")
    print(f"  均值: {score_stats['mean']:.6f} ± {score_stats['std']:.6f}")
    print(f"  范围: [{score_stats['min']:.6f}, {score_stats['max']:.6f}]")
    print("  分位数:")
    for p, v in score_stats['percentiles'].items():
        print(f"    {p}%: {v:.6f}")

    print("\n" + "=" * 70)


def main():
    """主函数"""
    print("🤖 LSTM异常检测系统 - Server Machine Dataset评估脚本")
    print("=" * 70)

    # 解析命令行参数
    args = parse_arguments()

    # 如果没有指定模型路径，自动查找最新的LSTM模型
    if args.model_path is None:
        lstm_dir = Path("models/anomaly_detection/lstm")
        if lstm_dir.exists():
            # 查找所有.ckpt文件，按修改时间排序
            ckpt_files = list(lstm_dir.glob("*.ckpt"))
            if ckpt_files:
                # 选择最新的模型文件
                latest_model = max(ckpt_files, key=lambda x: x.stat().st_mtime)
                args.model_path = str(latest_model)
                print(f"🔍 自动选择最新的模型: {args.model_path}")
            else:
                print("❌ 在 models/anomaly_detection/lstm/ 目录中未找到模型文件")
                sys.exit(1)
        else:
            print("❌ models/anomaly_detection/lstm/ 目录不存在")
            sys.exit(1)

    # 设置输出目录 - 优化版本
    model_path = Path(args.model_path)
    model_name = model_path.stem

    # 优化输出目录逻辑
    if args.output_dir is None:
        # 默认在模型目录下创建evaluations子目录
        args.output_dir = str(model_path.parent / "evaluations")
    elif args.output_dir == "auto":
        # 自动创建带时间戳的评估目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir_name = f"evaluation_{model_name}_{timestamp}"
        args.output_dir = str(model_path.parent / "evaluations" / eval_dir_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 输出目录: {output_dir}")
    print(f"📄 模型名称: {model_name}")
    print(f"📄 模型路径: {model_path}")
    print()

    try:
        # 设置MindSpore
        setup_mindspore_context(args.device_target, args.seed)

        # 加载模型和组件
        model, threshold_calculator, processor = load_model_and_components(
            args.model_path, args.threshold_path, args.scaler_path
        )

        # 加载和预处理测试数据
        test_data, true_labels = load_and_preprocess_test_data(
            args.test_data_path, processor, args.timestamp_column, args.label_column
        )

        # 执行推理
        predictions = perform_inference(model, test_data, args.batch_size)

        # 计算异常分数
        anomaly_scores = calculate_anomaly_scores(
            predictions, test_data.targets, method='l2_norm'
        )

        # 评估模型性能
        evaluation_results = evaluate_model(
            predictions, test_data.targets, true_labels,
            threshold_calculator.threshold, anomaly_scores
        )

        # 保存评估结果
        save_evaluation_results(evaluation_results, args.output_dir, model_name)

        # 打印评估摘要
        print_evaluation_summary(evaluation_results)

        print("\n🎉 评估流程完成！")
        print("=" * 70)
        print("📊 评估总结:")
        print(f"  - 测试数据: {args.test_data_path}")
        print(f"  - 模型文件: {args.model_path}")
        print(f"  - 评估结果目录: {output_dir}")
        print(f"  - 详细结果: {output_dir}/{model_name}_evaluation_results.json")
        print(f"  - 性能摘要: {output_dir}/{model_name}_evaluation_summary.txt")
        print("\n🚀 评估完成，可以查看详细结果！")

    except KeyboardInterrupt:
        print("\n⏹️ 评估被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 评估过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()