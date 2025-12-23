"""
异常检测训练路由
"""

import json
import os
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template
from ..services.trainer import AnomalyDetectionTrainer
from ....utils.parameter_validator import validate_training_config

ad_training_bp = Blueprint('ad_training', __name__, url_prefix='/anomaly_detection')

# 延迟初始化训练服务
_trainer = None

def get_trainer():
    """获取训练服务实例（延迟初始化）"""
    global _trainer
    if _trainer is None:
        _trainer = AnomalyDetectionTrainer()
    return _trainer


@ad_training_bp.route('/train', methods=['GET'])
def train_page():
    """异常检测模型训练页面"""
    return render_template('anomaly_detection/train.html')


def _normalize_training_payload(payload: dict) -> dict:
    """将前端传入的嵌套配置展开并转换为训练器需要的字段"""
    print("🔥🔥 _normalize_training_payload被调用")
    print(f"🔥 原始payload: {payload}")
    
    if not isinstance(payload, dict):
        return {}

    merged: dict = {}

    section_keys = ('model_config', 'training_config', 'dataset_config')
    for section_key in section_keys:
        section = payload.get(section_key)
        if isinstance(section, dict):
            merged.update(section)

    for key, value in payload.items():
        if key in section_keys:
            continue
        if key not in merged:
            merged[key] = value

    print(f"🔥 合并后的merged (在setdefault之前): {merged}")

    if isinstance(merged.get('bidirectional'), str):
        merged['bidirectional'] = merged['bidirectional'].strip().lower() in {'true', '1', 'yes', 'y'}

    int_fields = ['epochs', 'batch_size', 'sequence_length', 'input_dim', 'hidden_units', 'num_layers', 'prediction_horizon', 'num_filters', 'kernel_size', 'bottleneck_size', 'num_conv_layers', 'stride']
    float_fields = ['learning_rate', 'weight_decay', 'dropout', 'train_ratio', 'val_ratio', 'test_ratio', 'val_ratio_from_train', 'validation_split']

    for field in int_fields:
        if field in merged and merged[field] not in (None, ''):
            try:
                merged[field] = int(merged[field])
            except (TypeError, ValueError):
                pass

    for field in float_fields:
        if field in merged and merged[field] not in (None, ''):
            try:
                merged[field] = float(merged[field])
            except (TypeError, ValueError):
                pass

    merged.setdefault('module', 'anomaly_detection')
    merged.setdefault('model_type', 'lstm_predictor')
    # 注释掉强制设置dataset_mode为'one'，让前端的'processed_file'模式保持
    # merged.setdefault('dataset_mode', 'one')

    # 对于condition_filtered模式，根据validation_split计算train_ratio和val_ratio
    # 这样验证器就能找到这些必填字段
    dataset_mode = merged.get('dataset_mode', 'processed_file')
    if dataset_mode == 'condition_filtered':
        validation_split = merged.get('validation_split', 0.2)
        if validation_split not in (None, ''):
            try:
                validation_split = float(validation_split)
                merged['train_ratio'] = 1.0 - validation_split
                merged['val_ratio'] = validation_split
            except (TypeError, ValueError):
                pass
    elif dataset_mode == 'processed_file':
        # processed_file模式也需要计算train_ratio和val_ratio
        validation_split = merged.get('validation_split', 0.2)
        if validation_split not in (None, ''):
            try:
                validation_split = float(validation_split)
                merged['train_ratio'] = 1.0 - validation_split
                merged['val_ratio'] = validation_split
            except (TypeError, ValueError):
                pass

    print(f"🔥 最终merged (在setdefault之后): {merged}")

    if not merged.get('output_path'):
        merged['output_path'] = f"models/{merged['model_type']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    print(f"🔥 返回的merged: {merged}")
    return merged


@ad_training_bp.route('/api/train', methods=['POST'])
def train_model():
    """异常检测模型训练API"""
    print("🚀 /api/train路由被调用")
    try:
        model_config = request.get_json()
        print(f"🚀 原始request.get_json(): {model_config}")
        if not model_config:
            return jsonify({
                'status': 'error',
                'message': '无效的配置数据',
                'validation': {
                    'is_valid': False,
                    'errors': ['请求中没有配置数据'],
                    'warnings': [],
                    'suggestions': []
                }
            }), 400

        print("🚀 即将调用_normalize_training_payload")
        normalized_payload = _normalize_training_payload(model_config)
        print(f"🚀 _normalize_training_payload返回: {normalized_payload}")

        # 进行参数验证
        validation_result = validate_training_config(normalized_payload)

        if not validation_result['is_valid']:
            print('参数验证失败，validation_result =', validation_result)
            return jsonify({
                'status': 'validation_error',
                'message': '参数验证失败，请检查输入参数',
                'validation': validation_result
            }), 422

        trainer = get_trainer()
        result = trainer.train(normalized_payload)

        if validation_result['warnings'] or validation_result['suggestions']:
            result['validation'] = validation_result

        return jsonify(result)

    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': f'参数错误: {str(ve)}',
            'validation': {
                'is_valid': False,
                'errors': [str(ve)],
                'warnings': [],
                'suggestions': []
            }
        }), 400

    except Exception as e:
        print(f"Anomaly detection training error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'训练失败: {str(e)}',
            'validation': {
                'is_valid': False,
                'errors': [f'服务器内部错误: {str(e)}'],
                'warnings': [],
                'suggestions': ['请检查服务器日志或联系管理员']
            }
        }), 500


@ad_training_bp.route('/training_progress', methods=['GET'])
def training_progress_page():
    """训练进度监控页面"""
    return render_template('anomaly_detection/training_progress.html')


@ad_training_bp.route('/api/training_status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    """获取训练状态API"""
    try:
        trainer = get_trainer()
        status = trainer.get_training_status(task_id)
        return jsonify(status)
    except Exception as e:
        print(f"Get training status error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'获取训练状态失败: {str(e)}'
        }), 500


@ad_training_bp.route('/api/pause_training/<task_id>', methods=['POST'])
def pause_training(task_id):
    """暂停训练API"""
    try:
        trainer = get_trainer()
        result = trainer.pause_training(task_id)
        return jsonify(result)
    except Exception as e:
        print(f"Pause training error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'暂停训练失败: {str(e)}'
        }), 500


@ad_training_bp.route('/api/stop_training/<task_id>', methods=['POST'])
def stop_training(task_id):
    """停止训练API"""
    try:
        trainer = get_trainer()
        result = trainer.stop_training(task_id)
        return jsonify(result)
    except Exception as e:
        print(f"Stop training error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'停止训练失败: {str(e)}'
        }), 500


@ad_training_bp.route('/api/validate', methods=['POST'])
def validate_parameters():
    """参数验证API（独立验证接口）"""
    try:
        # 获取JSON配置
        config = request.get_json()
        if not config:
            return jsonify({
                'is_valid': False,
                'errors': ['请求中没有配置数据'],
                'warnings': [],
                'suggestions': []
            }), 400

        # 进行参数验证
        validation_result = validate_training_config(config)
        return jsonify(validation_result)

    except Exception as e:
        print(f"Parameter validation error: {e}")
        return jsonify({
            'is_valid': False,
            'errors': [f'验证过程出错: {str(e)}'],
            'warnings': [],
            'suggestions': []
        }), 500


@ad_training_bp.route('/api/calculate_threshold/<task_id>', methods=['POST'])
def calculate_threshold_proxy(task_id):
    """阈值计算API代理（转发到云端）"""
    try:
        # 获取前端发送的阈值参数
        threshold_params = request.get_json() or {}
        trainer = get_trainer()
        result = trainer.calculate_threshold(task_id, threshold_params)
        return jsonify(result)
    except Exception as e:
        print(f"Calculate threshold error: {e}")
        return jsonify({
            'success': False,
            'error': f'阈值计算失败: {str(e)}'
        }), 500


@ad_training_bp.route('/api/processed_data', methods=['GET'])
def get_processed_data_files():
    """获取已标注的数据文件列表（从labeled目录）"""
    try:
        # 数据文件目录 - 使用labeled目录
        edge_root = Path(__file__).resolve().parents[4]  # 从 training.py 到 edge 目录
        labeled_dir = edge_root / 'data' / 'labeled' / 'AnomalyDetection'
        
        if not labeled_dir.exists():
            return jsonify({
                'success': True,
                'files': [],
                'message': '标注数据目录不存在 (edge/data/labeled/AnomalyDetection)'
            })
        
        # 获取所有CSV文件
        files = []
        for filename in os.listdir(labeled_dir):
            if filename.endswith('.csv'):
                file_path = labeled_dir / filename
                file_stat = file_path.stat()
                
                # 解析文件名获取信息
                file_info = {
                    'filename': filename,
                    'display_name': filename,
                    'size': file_stat.st_size,
                    'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 尝试从元数据文件获取标签信息
                meta_file_path = edge_root / 'data' / 'meta' / 'AnomalyDetection' / (filename.replace('.csv', '.json'))
                if meta_file_path.exists():
                    try:
                        with open(meta_file_path, 'r', encoding='utf-8') as f:
                            meta_data = json.load(f)
                            file_info['display_name'] = meta_data.get('display_name', filename)
                            tags_label = meta_data.get('tags_label', [])
                            if tags_label and len(tags_label) > 0:
                                # 支持两种格式：dict格式 {'value': '正常'} 或直接是字符串
                                first_label = tags_label[0]
                                if isinstance(first_label, dict):
                                    file_info['label'] = first_label.get('value', '')
                                else:
                                    file_info['label'] = str(first_label)
                            
                            # 读取工况信息
                            tags_condition = meta_data.get('tags_condition', [])
                            if tags_condition:
                                file_info['conditions'] = {}
                                for cond in tags_condition:
                                    if isinstance(cond, dict) and 'key' in cond and 'value' in cond:
                                        file_info['conditions'][cond['key']] = cond['value']
                    except Exception:
                        pass
                
                files.append(file_info)
        
        # 按修改时间排序（最新的在前）
        files.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': files,
            'total': len(files)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取数据文件列表失败: {str(e)}'
        }), 500


@ad_training_bp.route('/api/condition_keys', methods=['GET'])
def get_condition_keys():
    """获取所有工况key列表"""
    try:
        edge_root = Path(__file__).resolve().parents[4]
        meta_dir = edge_root / 'data' / 'meta' / 'AnomalyDetection'
        
        print(f"🔍 查找元数据目录: {meta_dir}")
        print(f"🔍 目录是否存在: {meta_dir.exists()}")
        
        if not meta_dir.exists():
            print(f"⚠️ 元数据目录不存在: {meta_dir}")
            return jsonify({
                'success': True,
                'keys': []
            })
        
        # 收集所有唯一的工况key
        condition_keys = set()
        file_count = 0
        
        for meta_file in meta_dir.glob('*.json'):
            file_count += 1
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    tags_condition = meta_data.get('tags_condition', [])
                    print(f"📄 文件 {meta_file.name}: {len(tags_condition)} 个工况")
                    for cond in tags_condition:
                        if isinstance(cond, dict) and 'key' in cond:
                            condition_keys.add(cond['key'])
                            print(f"  - 找到工况key: {cond['key']}")
            except Exception as e:
                print(f"❌ 读取元文件失败 {meta_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"✅ 共处理 {file_count} 个元文件，找到 {len(condition_keys)} 个唯一的工况key: {sorted(list(condition_keys))}")
        
        return jsonify({
            'success': True,
            'keys': sorted(list(condition_keys))
        })
        
    except Exception as e:
        print(f"❌ 获取工况key列表异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'获取工况key列表失败: {str(e)}'
        }), 500


@ad_training_bp.route('/api/condition_values', methods=['GET'])
def get_condition_values():
    """获取指定工况key的所有value列表"""
    try:
        key = request.args.get('key', '').strip()
        if not key:
            return jsonify({
                'success': False,
                'error': '请指定工况key'
            }), 400
        
        edge_root = Path(__file__).resolve().parents[4]
        meta_dir = edge_root / 'data' / 'meta' / 'AnomalyDetection'
        
        print(f"🔍 查找工况值: key={key}, 目录={meta_dir}")
        
        if not meta_dir.exists():
            print(f"⚠️ 元数据目录不存在: {meta_dir}")
            return jsonify({
                'success': True,
                'values': []
            })
        
        # 收集所有唯一的value（转换为字符串以保持一致性）
        condition_values = set()
        
        for meta_file in meta_dir.glob('*.json'):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    tags_condition = meta_data.get('tags_condition', [])
                    for cond in tags_condition:
                        if isinstance(cond, dict) and cond.get('key') == key and 'value' in cond:
                            # 将value转换为字符串，确保类型一致
                            value = str(cond['value'])
                            condition_values.add(value)
            except Exception as e:
                print(f"❌ 读取元文件失败 {meta_file}: {e}")
                continue
        
        sorted_values = sorted(list(condition_values), key=lambda x: (float(x) if x.replace('.', '').replace('-', '').isdigit() else float('inf'), x))
        print(f"✅ 找到 {len(sorted_values)} 个值: {sorted_values}")
        
        return jsonify({
            'success': True,
            'key': key,
            'values': sorted_values
        })
        
    except Exception as e:
        print(f"❌ 获取工况value列表异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'获取工况value列表失败: {str(e)}'
        }), 500


@ad_training_bp.route('/api/filter_files', methods=['POST'])
def filter_files():
    """根据工况条件筛选文件"""
    try:
        data = request.get_json()
        conditions = data.get('conditions', {})  # {key: [value1, value2, ...]}
        file_type = data.get('file_type', 'train')  # 'train' 或 'test'
        
        edge_root = Path(__file__).resolve().parents[4]
        meta_dir = edge_root / 'data' / 'meta' / 'AnomalyDetection'
        labeled_dir = edge_root / 'data' / 'labeled' / 'AnomalyDetection'
        
        if not meta_dir.exists() or not labeled_dir.exists():
            return jsonify({
                'success': True,
                'files': []
            })
        
        matched_files = []
        
        # 遍历所有元文件
        for meta_file in meta_dir.glob('*.json'):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                # 检查标签
                tags_label = meta_data.get('tags_label', [])
                if not tags_label:
                    continue
                
                # 获取第一个标签的值
                first_label_value = tags_label[0].get('value', '').strip()
                
                # 检查对应的数据文件是否存在（先获取文件名）
                data_filename = meta_file.stem + '.csv'
                
                # 根据file_type筛选标签
                if file_type == 'train':
                    # 训练集：只选择标签为"正常"的文件
                    if first_label_value != '正常':
                        continue
                elif file_type == 'test':
                    # 测试集：选择标签为"正常"或"异常"的文件
                    if first_label_value not in ['正常', '异常']:
                        continue
                
                # 检查工况条件
                tags_condition = meta_data.get('tags_condition', [])
                condition_dict = {}
                for cond in tags_condition:
                    if isinstance(cond, dict) and 'key' in cond and 'value' in cond:
                        condition_dict[cond['key']] = cond['value']
                
                # 验证是否满足所有条件
                satisfies_all = True
                for key, required_values in conditions.items():
                    if not required_values:  # 如果没有选择任何value，跳过这个key
                        continue
                    file_value = condition_dict.get(key)
                    if file_value is None or file_value not in required_values:
                        satisfies_all = False
                        break
                
                if not satisfies_all:
                    continue
                data_file_path = labeled_dir / data_filename
                
                if not data_file_path.exists():
                    continue
                
                # 构建文件信息
                file_stat = data_file_path.stat()
                file_info = {
                    'filename': data_filename,
                    'display_name': meta_data.get('display_name', data_filename),
                    'label': first_label_value,
                    'conditions': condition_dict,
                    'size': file_stat.st_size,
                    'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'meta_file': meta_file.name
                }
                
                matched_files.append(file_info)
                
            except Exception as e:
                print(f"处理元文件失败 {meta_file}: {e}")
                continue
        
        # 按文件名排序
        matched_files.sort(key=lambda x: x['filename'])
        
        return jsonify({
            'success': True,
            'files': matched_files,
            'total': len(matched_files)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'筛选文件失败: {str(e)}'
        }), 500
