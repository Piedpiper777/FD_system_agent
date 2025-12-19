"""
RUL预测训练路由
"""

from flask import Blueprint, request, jsonify, render_template
from pathlib import Path
import json
from ..services.trainer import RULPredictionTrainer

rup_training_bp = Blueprint('rup_training', __name__, url_prefix='/rul_prediction')

# 延迟初始化训练服务
_trainer = None

def get_trainer():
    """获取训练服务实例（延迟初始化）"""
    global _trainer
    if _trainer is None:
        _trainer = RULPredictionTrainer()
    return _trainer


@rup_training_bp.route('/train', methods=['GET'])
def train_page():
    """RUL预测模型训练页面"""
    try:
        return render_template('rul_prediction/train.html')
    except Exception as e:
        print(f"RUL prediction train page error: {e}")
        raise


@rup_training_bp.route('/training_progress', methods=['GET'])
def training_progress_page():
    """RUL预测训练进度页面"""
    try:
        return render_template('rul_prediction/training_progress.html')
    except Exception as e:
        print(f"RUL prediction training progress page error: {e}")
        raise


@rup_training_bp.route('/api/train', methods=['POST'])
def train_model():
    """RUL预测模型训练API"""
    try:
        # 获取JSON配置
        model_config = request.get_json()
        if not model_config:
            return jsonify({'success': False, 'error': '无效的配置数据'})

        # 调用训练服务
        trainer = get_trainer()
        result = trainer.train(model_config)
        
        # 转换返回格式
        if result.get('status') == 'success':
            return jsonify({
                'success': True,
                'task_id': result.get('task_id'),
                'message': result.get('message', '训练任务已提交')
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('message', '训练失败')
            }), 500

    except Exception as e:
        print(f"RUL prediction training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'训练失败: {str(e)}'}), 500


@rup_training_bp.route('/api/processed_data', methods=['GET'])
def get_processed_data_files():
    """获取所有已处理的数据文件列表"""
    try:
        edge_root = Path(__file__).resolve().parents[4]
        meta_dir = edge_root / 'data' / 'meta' / 'RULPrediction'
        labeled_dir = edge_root / 'data' / 'labeled' / 'RULPrediction'
        
        if not meta_dir.exists() or not labeled_dir.exists():
            return jsonify({
                'success': True,
                'files': []
            })
        
        files = []
        
        # 遍历所有元文件
        for meta_file in meta_dir.glob('*.json'):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                # RUL预测需要rul_config（包含rul_unit和max_rul）
                # 注意：failure_row_index不需要检查，因为labeled文件已经截断，最后一个点就是故障点
                rul_config = meta_data.get('rul_config', {})
                if not rul_config:
                    print(f"⚠️ [RUL预测] 文件 {meta_file.name} 缺少rul_config，跳过")
                    continue
                
                # 检查对应的数据文件是否存在
                data_filename = meta_file.stem + '.csv'
                data_file_path = labeled_dir / data_filename
                
                if not data_file_path.exists():
                    print(f"⚠️ [RUL预测] 数据文件不存在: {data_filename}，跳过")
                    continue
                
                # 验证文件不为空
                try:
                    import pandas as pd
                    df = pd.read_csv(data_file_path, nrows=1)
                    if df.empty:
                        print(f"⚠️ [RUL预测] 数据文件为空: {data_filename}，跳过")
                        continue
                except Exception as e:
                    print(f"⚠️ [RUL预测] 无法读取数据文件: {data_filename}，错误: {e}，跳过")
                    continue
                
                # 获取文件信息
                file_size = data_file_path.stat().st_size
                display_name = meta_data.get('display_name', data_filename)
                
                # 获取工况信息
                tags_condition = meta_data.get('tags_condition', [])
                conditions = {}
                for cond in tags_condition:
                    if isinstance(cond, dict) and 'key' in cond and 'value' in cond:
                        conditions[cond['key']] = cond['value']
                
                files.append({
                    'filename': data_filename,
                    'display_name': display_name,
                    'size': file_size,
                    'conditions': conditions
                })
            except Exception as e:
                print(f"读取元文件失败 {meta_file}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'files': files
        })
        
    except Exception as e:
        print(f"获取数据文件列表失败: {e}")
        return jsonify({
            'success': False,
            'error': f'获取数据文件列表失败: {str(e)}'
        }), 500


@rup_training_bp.route('/api/condition_keys', methods=['GET'])
def get_condition_keys():
    """获取所有工况key列表"""
    try:
        edge_root = Path(__file__).resolve().parents[4]
        meta_dir = edge_root / 'data' / 'meta' / 'RULPrediction'
        
        print(f"🔍 [RUL预测] 查找元数据目录: {meta_dir}")
        print(f"🔍 [RUL预测] 目录是否存在: {meta_dir.exists()}")
        
        if not meta_dir.exists():
            print(f"⚠️ [RUL预测] 元数据目录不存在: {meta_dir}")
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
                    print(f"📄 [RUL预测] 文件 {meta_file.name}: {len(tags_condition)} 个工况")
                    for cond in tags_condition:
                        if isinstance(cond, dict) and 'key' in cond:
                            condition_keys.add(cond['key'])
                            print(f"  - 找到工况key: {cond['key']}")
            except Exception as e:
                print(f"❌ [RUL预测] 读取元文件失败 {meta_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"✅ [RUL预测] 共处理 {file_count} 个元文件，找到 {len(condition_keys)} 个唯一的工况key: {sorted(list(condition_keys))}")
        
        return jsonify({
            'success': True,
            'keys': sorted(list(condition_keys))
        })
        
    except Exception as e:
        print(f"❌ [RUL预测] 加载工况key失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'获取工况key列表失败: {str(e)}'
        }), 500


@rup_training_bp.route('/api/condition_values', methods=['GET'])
def get_condition_values():
    """获取指定工况key的所有value列表"""
    try:
        key = request.args.get('key')
        if not key:
            return jsonify({
                'success': False,
                'error': '缺少key参数'
            }), 400
        
        edge_root = Path(__file__).resolve().parents[4]
        meta_dir = edge_root / 'data' / 'meta' / 'RULPrediction'
        
        if not meta_dir.exists():
            return jsonify({
                'success': True,
                'values': []
            })
        
        # 收集所有唯一的value
        condition_values = set()
        
        for meta_file in meta_dir.glob('*.json'):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    tags_condition = meta_data.get('tags_condition', [])
                    for cond in tags_condition:
                        if isinstance(cond, dict) and cond.get('key') == key and 'value' in cond:
                            condition_values.add(cond['value'])
            except Exception as e:
                print(f"读取元文件失败 {meta_file}: {e}")
                continue
        
        # 尝试将value转换为数字并排序，如果无法转换则按字符串排序
        try:
            values_list = sorted(list(condition_values), key=lambda x: (float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '').replace('-', '').isdigit()) else float('inf'), str(x)))
        except:
            values_list = sorted(list(condition_values))
        
        return jsonify({
            'success': True,
            'values': values_list
        })
        
    except Exception as e:
        print(f"获取工况value列表失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'获取工况value列表失败: {str(e)}'
        }), 500


@rup_training_bp.route('/api/filter_files', methods=['POST'])
def filter_files():
    """根据工况条件筛选文件"""
    try:
        data = request.get_json()
        conditions = data.get('conditions', {})  # {key: [value1, value2, ...]}
        file_type = data.get('file_type', 'train')  # 'train' 或 'test'
        
        edge_root = Path(__file__).resolve().parents[4]
        meta_dir = edge_root / 'data' / 'meta' / 'RULPrediction'
        labeled_dir = edge_root / 'data' / 'labeled' / 'RULPrediction'
        
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
                
                # RUL预测需要rul_config（包含rul_unit和max_rul）
                # 注意：failure_row_index不需要检查，因为labeled文件已经截断，最后一个点就是故障点
                rul_config = meta_data.get('rul_config', {})
                if not rul_config:
                    continue
                
                # 检查对应的数据文件是否存在
                data_filename = meta_file.stem + '.csv'
                data_file_path = labeled_dir / data_filename
                
                if not data_file_path.exists():
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
                
                # 获取文件信息
                file_size = data_file_path.stat().st_size
                display_name = meta_data.get('display_name', data_filename)
                
                matched_files.append({
                    'filename': data_filename,
                    'display_name': display_name,
                    'size': file_size,
                    'conditions': condition_dict
                })
            except Exception as e:
                print(f"处理文件失败 {meta_file}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'files': matched_files
        })
        
    except Exception as e:
        print(f"筛选文件失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'筛选文件失败: {str(e)}'
        }), 500


@rup_training_bp.route('/api/task/<task_id>/status', methods=['GET'])
def get_training_status(task_id):
    """获取训练任务状态"""
    try:
        trainer = get_trainer()
        result = trainer.get_training_status(task_id)
        return jsonify(result)
    except Exception as e:
        print(f"RUL prediction get training status error: {e}")
        return jsonify({'success': False, 'error': f'获取训练状态失败: {str(e)}'}), 500


@rup_training_bp.route('/api/download_model/<task_id>', methods=['POST'])
def download_model_to_edge(task_id):
    """下载模型到Edge本地（从云端下载zip并解压）"""
    try:
        import requests
        import zipfile
        import tempfile
        import os
        from flask import current_app
        
        # 从配置中获取云端服务URL
        cloud_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
        
        payload = request.get_json(silent=True) or {}
        model_type = payload.get('model_type')

        if not model_type:
            # 尝试从云端任务状态中获取模型类型
            try:
                status_resp = requests.get(
                    f'{cloud_url}/api/rul_prediction/task/{task_id}/status',
                    timeout=10
                )
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    if status_data.get('success'):
                        task_info = status_data.get('task', {})
                        model_type = task_info.get('model_type') or task_info.get('config', {}).get('model_type')
            except Exception as status_error:
                print(f"无法从云端获取模型类型，使用默认值: {status_error}")

        if not model_type:
            model_type = 'bilstm_gru_regressor'
        
        print(f"正在从云端下载模型: {cloud_url}/api/rul_prediction/download_model/{task_id}")
        print(f"模型类型: {model_type}")
        
        # 从云端下载模型（zip文件）
        response = requests.get(
            f'{cloud_url}/api/rul_prediction/download_model/{task_id}',
            stream=True,
            timeout=300
        )
        
        if response.status_code != 200:
            error_msg = f'从云端下载模型失败: HTTP {response.status_code}'
            try:
                error_data = response.json()
                error_msg = error_data.get('message', error_msg)
            except:
                pass
            return jsonify({
                'success': False,
                'message': error_msg
            }), 500
        
        # 保存到edge/models/rul_prediction/{模型类型}
        edge_root = Path(__file__).resolve().parents[4]
        models_dir = edge_root / 'models' / 'rul_prediction' / model_type
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # 先保存zip文件到临时目录
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip_path = temp_zip.name
        
        for chunk in response.iter_content(chunk_size=8192):
            temp_zip.write(chunk)
        temp_zip.close()
        
        print(f"模型zip文件已下载到: {temp_zip_path}")
        
        # 解压zip文件到models目录
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zipf:
                zipf.extractall(models_dir)
                extracted_files = zipf.namelist()
                print(f"已解压文件: {extracted_files}")
        finally:
            # 删除临时zip文件
            os.unlink(temp_zip_path)
        
        # 查找解压后的模型目录
        model_folder = models_dir / task_id
        if not model_folder.exists():
            # 可能解压到了子目录中
            for item in models_dir.iterdir():
                if item.is_dir() and task_id in item.name:
                    model_folder = item
                    break
        
        if not model_folder.exists():
            return jsonify({
                'success': False,
                'message': f'解压后找不到模型目录: {task_id}'
            }), 500
        
        # 验证关键文件是否存在（权重文件任选其一）
        weight_candidates = ['best_model.pt', 'model.pt', 'model.ckpt']
        has_weight = any((model_folder / f).exists() for f in weight_candidates)
        config_exists = (model_folder / 'model_config.json').exists()

        if not has_weight or not config_exists:
            missing_parts = []
            if not has_weight:
                missing_parts.append('模型权重(best_model.pt/model.pt/model.ckpt)')
            if not config_exists:
                missing_parts.append('model_config.json')
            return jsonify({
                'success': False,
                'message': f'模型文件不完整，缺少: {", ".join(missing_parts)}'
            }), 500
        
        return jsonify({
            'success': True,
            'message': '模型下载成功',
            'path': str(model_folder),
            'model_id': task_id
        })
        
    except Exception as e:
        print(f"下载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'下载模型失败: {str(e)}'
        }), 500