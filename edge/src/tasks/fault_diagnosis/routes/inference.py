"""
故障诊断推理路由
"""

from flask import Blueprint, request, jsonify, render_template
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

fd_inference_bp = Blueprint('fd_inference', __name__, url_prefix='/fault_diagnosis')


@fd_inference_bp.route('/inference')
def inference_page():
    """故障诊断推理页面"""
    return render_template('fault_diagnosis/inference.html')


@fd_inference_bp.route('/api/inference/data_files', methods=['GET'])
def get_data_files():
    """获取可用的数据文件列表（从 processed/FaultDiagnosis 目录）"""
    try:
        edge_dir = Path(__file__).resolve().parents[4]
        data_dir = edge_dir / 'data' / 'processed' / 'FaultDiagnosis'
        
        files = []
        if data_dir.exists():
            for file_path in data_dir.glob('*.csv'):
                stat = file_path.stat()
                files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': stat.st_size,
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # 按修改时间降序排列
        files.sort(key=lambda x: x['modified_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': files,
            'total_count': len(files)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@fd_inference_bp.route('/api/inference/run', methods=['POST'])
def run_inference():
    """执行故障诊断推理"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        data_file = data.get('data_file')
        batch_size = data.get('batch_size', 32)
        condition_values = data.get('condition_values', {})  # 工况值
        
        if not model_id:
            return jsonify({'success': False, 'error': '请选择模型'}), 400
        if not data_file:
            return jsonify({'success': False, 'error': '请选择数据文件'}), 400
        
        # 查找模型目录 (模型按类型保存在子目录: cnn_1d/{task_id} 或 lstm/{task_id})
        edge_dir = Path(__file__).resolve().parents[4]
        models_dir = edge_dir / 'models' / 'fault_diagnosis'
        
        # 在所有模型类型子目录中查找
        model_dir = None
        for model_type_dir in ['cnn_1d', 'lstm', 'resnet_1d']:
            potential_dir = models_dir / model_type_dir / model_id
            if potential_dir.exists():
                model_dir = potential_dir
                break
        
        # 如果在子目录中未找到，尝试直接查找（向后兼容）
        if model_dir is None:
            direct_dir = models_dir / model_id
            if direct_dir.exists():
                model_dir = direct_dir
        
        if model_dir is None or not model_dir.exists():
            return jsonify({'success': False, 'error': f'模型 {model_id} 不存在'}), 404
        
        # 加载模型配置
        config_file = model_dir / 'model_config.json'
        if not config_file.exists():
            return jsonify({'success': False, 'error': '模型配置文件不存在'}), 404
        
        with open(config_file, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        
        # 验证工况值（如果模型配置中有工况）
        model_conditions = model_config.get('conditions', [])
        if model_conditions:
            for cond in model_conditions:
                cond_name = cond.get('name')
                if cond_name not in condition_values:
                    return jsonify({'success': False, 'error': f'缺少工况值: {cond_name}'}), 400
        
        # 加载数据文件
        data_path = Path(data_file)
        if not data_path.exists():
            return jsonify({'success': False, 'error': f'数据文件不存在: {data_file}'}), 404
        
        # 执行推理
        result = _run_inference(
            model_dir=model_dir,
            model_config=model_config,
            data_path=data_path,
            batch_size=batch_size,
            condition_values=condition_values
        )
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def _run_inference(model_dir: Path, model_config: dict, data_path: Path, batch_size: int, condition_values: dict = None) -> dict:
    """执行推理逻辑"""
    start_time = time.time()
    
    # 获取配置参数
    sequence_length = model_config.get('sequence_length', 100)
    n_features = model_config.get('n_features', 5)
    labels = model_config.get('labels', [])
    model_type = model_config.get('model_type', 'cnn_1d_classifier')
    model_conditions = model_config.get('conditions', [])
    
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 检测并排除非特征列（时间戳、标签等）
    exclude_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['time', 'timestamp', 'date', 'label', 'class', 'target']):
            exclude_cols.append(col)
    
    # 获取特征列
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 如果第一列是索引或时间戳类型，也排除
    if len(feature_cols) > 0:
        first_col = feature_cols[0]
        if df[first_col].dtype == 'object' or 'unnamed' in first_col.lower():
            feature_cols = feature_cols[1:]
    
    # 获取原始特征数据
    # 计算原始特征数（总特征数减去工况数）
    num_conditions = len(model_conditions)
    original_n_features = n_features - num_conditions
    
    if len(feature_cols) < original_n_features:
        # 使用所有数值列
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:original_n_features]
    
    features = df[feature_cols[:original_n_features]].values if len(feature_cols) >= original_n_features else df[feature_cols].values
    
    # 添加工况列到特征数据（与训练时保持一致）
    if condition_values and model_conditions:
        # 按模型配置中工况的顺序添加工况值列
        for cond in model_conditions:
            cond_name = cond.get('name')
            cond_value = condition_values.get(cond_name, 0)
            # 创建一列全为该工况值的数组
            cond_col = np.full((len(features), 1), cond_value, dtype=np.float32)
            features = np.hstack([features, cond_col])
        
        print(f"添加工况后特征维度: {features.shape[1]} (原始: {original_n_features}, 工况数: {num_conditions})")
    
    # 创建序列
    sequences = _create_sequences(features, sequence_length)
    
    if len(sequences) == 0:
        raise ValueError(f'数据长度不足以创建序列（需要至少 {sequence_length} 个样本）')
    
    # 加载模型并进行推理
    predictions, confidences = _load_and_predict(
        model_dir=model_dir,
        model_type=model_type,
        sequences=sequences,
        num_classes=len(labels) if labels else model_config.get('num_classes', 3),
        batch_size=batch_size
    )
    
    # 统计结果
    inference_time = time.time() - start_time
    
    # 计算类别分布
    unique, counts = np.unique(predictions, return_counts=True)
    class_distribution = {}
    for idx, count in zip(unique, counts):
        label_name = labels[idx] if idx < len(labels) else f'类别{idx}'
        class_distribution[label_name] = int(count)
    
    # 找出主要预测类别
    main_class_idx = np.argmax(counts)
    main_class_count = counts[main_class_idx]
    predicted_label = labels[unique[main_class_idx]] if unique[main_class_idx] < len(labels) else f'类别{unique[main_class_idx]}'
    
    # 计算平均置信度
    avg_confidence = float(np.mean(confidences))
    
    # 主类别的平均置信度
    main_confidence = float(np.mean([confidences[i] for i in range(len(predictions)) if predictions[i] == unique[main_class_idx]]))
    
    inference_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    result = {
        'inference_id': inference_id,
        'model_id': model_dir.name,
        'data_file': data_path.name,
        'predicted_label': predicted_label,
        'confidence': main_confidence,
        'total_samples': len(sequences),
        'class_distribution': class_distribution,
        'avg_confidence': avg_confidence,
        'confidences': confidences.tolist() if isinstance(confidences, np.ndarray) else confidences,
        'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        'inference_time': inference_time,
        'created_at': datetime.now().isoformat(),
        'model_config': {
            'sequence_length': sequence_length,
            'n_features': n_features,
            'labels': labels,
            'model_type': model_type
        }
    }
    
    # 保存推理结果到 inference_tasks 目录
    _save_inference_result(result)
    
    return result


def _save_inference_result(result: dict):
    """保存推理结果到文件"""
    try:
        edge_dir = Path(__file__).resolve().parents[4]
        inference_dir = edge_dir / 'inference_tasks' / 'fault_diagnosis' / result['inference_id']
        inference_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整结果
        result_file = inference_dir / 'inference_result.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存摘要（不包含大数组，用于快速加载列表）
        summary = {
            'inference_id': result['inference_id'],
            'model_id': result['model_id'],
            'data_file': result['data_file'],
            'predicted_label': result['predicted_label'],
            'confidence': result['confidence'],
            'total_samples': result['total_samples'],
            'class_distribution': result['class_distribution'],
            'avg_confidence': result['avg_confidence'],
            'inference_time': result['inference_time'],
            'created_at': result['created_at'],
            'model_config': result['model_config']
        }
        summary_file = inference_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"推理结果已保存到: {inference_dir}")
        
    except Exception as e:
        print(f"保存推理结果失败: {e}")


def _create_sequences(data: np.ndarray, sequence_length: int, stride: int = 1) -> np.ndarray:
    """创建时间序列窗口"""
    if len(data) < sequence_length:
        return np.array([])
    
    sequences = []
    for i in range(0, len(data) - sequence_length + 1, stride):
        sequences.append(data[i:i + sequence_length])
    
    return np.array(sequences)


def _load_and_predict(model_dir: Path, model_type: str, sequences: np.ndarray, 
                      num_classes: int, batch_size: int) -> tuple:
    """加载模型并进行预测"""
    try:
        import mindspore as ms
        from mindspore import Tensor
        
        # 设置MindSpore为推理模式
        ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU')
        
        # 加载模型
        model_path = model_dir / 'model.ckpt'
        if not model_path.exists():
            raise FileNotFoundError(f'模型文件不存在: {model_path}')
        
        # 从配置中获取模型参数
        config_file = model_dir / 'model_config.json'
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 根据模型类型创建网络（使用 Edge 端本地的模型构建器）
        input_shape = (sequences.shape[1], sequences.shape[2])
        
        if 'resnet' in model_type.lower():
            from ..core.resnet_1d.model_builder import ModelBuilder
            
            model = ModelBuilder.build_resnet_1d_classifier(
                input_shape=input_shape,
                num_classes=num_classes,
                base_channels=config.get('base_channels', 64),
                block_config=config.get('block_config', 'resnet_small'),
                kernel_size=config.get('kernel_size', 3),
                dropout=config.get('dropout', 0.3)
            )
        elif 'cnn' in model_type.lower():
            from ..core.cnn_1d.model_builder import ModelBuilder
            
            model = ModelBuilder.build_cnn_1d_classifier(
                input_shape=input_shape,
                num_classes=num_classes,
                num_filters=config.get('num_filters', 64),
                kernel_size=config.get('kernel_size', 3),
                num_conv_layers=config.get('num_conv_layers', 3),
                dropout=config.get('dropout', 0.3)
            )
        else:
            from ..core.lstm.model_builder import ModelBuilder
            
            model = ModelBuilder.build_lstm_classifier(
                input_shape=input_shape,
                num_classes=num_classes,
                hidden_units=config.get('hidden_units', 128),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.3),
                bidirectional=config.get('bidirectional', False),
                use_attention=config.get('use_attention', False)
            )
        
        # 加载参数
        param_dict = ms.load_checkpoint(str(model_path))
        ms.load_param_into_net(model, param_dict)
        model.set_train(False)
        
        # 批量预测
        all_predictions = []
        all_confidences = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_tensor = Tensor(batch.astype(np.float32))
            
            # 前向传播
            output = model(batch_tensor)
            probs = ms.ops.Softmax(axis=1)(output)
            
            # 获取预测结果
            pred = probs.argmax(axis=1).asnumpy()
            conf = probs.max(axis=1).asnumpy()
            
            all_predictions.extend(pred)
            all_confidences.extend(conf)
        
        return np.array(all_predictions), np.array(all_confidences)
        
    except ImportError as e:
        print(f"MindSpore导入失败: {e}")
        raise
    except Exception as e:
        print(f"模型加载或推理失败: {e}")
        raise


@fd_inference_bp.route('/diagnose', methods=['POST'])
def diagnose_fault():
    """故障诊断推理API（兼容旧接口）"""
    return run_inference()


@fd_inference_bp.route('/api/inference/history', methods=['GET'])
def get_inference_history():
    """获取推理历史列表"""
    try:
        edge_dir = Path(__file__).resolve().parents[4]
        inference_base_dir = edge_dir / 'inference_tasks' / 'fault_diagnosis'
        
        if not inference_base_dir.exists():
            return jsonify({
                'success': True,
                'tasks': [],
                'total': 0
            })
        
        tasks = []
        
        # 遍历所有推理结果目录
        for inference_dir in sorted(inference_base_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not inference_dir.is_dir():
                continue
            
            # 读取摘要文件
            summary_file = inference_dir / 'summary.json'
            if not summary_file.exists():
                # 尝试读取完整结果文件
                result_file = inference_dir / 'inference_result.json'
                if not result_file.exists():
                    continue
                summary_file = result_file
            
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                tasks.append(summary)
            except Exception as e:
                print(f"读取推理结果失败 {inference_dir}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'tasks': tasks,
            'total': len(tasks)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@fd_inference_bp.route('/api/inference/result/<inference_id>', methods=['GET'])
def get_inference_result(inference_id):
    """获取特定推理任务的完整结果"""
    try:
        edge_dir = Path(__file__).resolve().parents[4]
        inference_dir = edge_dir / 'inference_tasks' / 'fault_diagnosis' / inference_id
        
        if not inference_dir.exists():
            return jsonify({
                'success': False,
                'error': f'推理结果不存在: {inference_id}'
            }), 404
        
        # 读取完整结果文件
        result_file = inference_dir / 'inference_result.json'
        if not result_file.exists():
            return jsonify({
                'success': False,
                'error': f'推理结果文件不存在: {inference_id}'
            }), 404
        
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@fd_inference_bp.route('/api/inference/delete/<inference_id>', methods=['DELETE'])
def delete_inference_result(inference_id):
    """删除推理结果"""
    try:
        import shutil
        
        edge_dir = Path(__file__).resolve().parents[4]
        inference_dir = edge_dir / 'inference_tasks' / 'fault_diagnosis' / inference_id
        
        if not inference_dir.exists():
            return jsonify({
                'success': False,
                'error': f'推理结果不存在: {inference_id}'
            }), 404
        
        shutil.rmtree(inference_dir)
        
        return jsonify({
            'success': True,
            'message': f'已删除推理结果: {inference_id}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
