"""
故障诊断训练路由
"""

import json
import os
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template
from pathlib import Path
from ..services.trainer import FaultDiagnosisTrainer

fd_training_bp = Blueprint('fd_training', __name__, url_prefix='/fault_diagnosis')

# 延迟初始化训练服务
_trainer = None

def get_trainer():
    """获取训练服务实例（延迟初始化）"""
    global _trainer
    if _trainer is None:
        _trainer = FaultDiagnosisTrainer()
    return _trainer


@fd_training_bp.route('/train', methods=['GET'])
def train_page():
    """故障诊断模型训练页面"""
    try:
        return render_template('fault_diagnosis/train.html')
    except Exception as e:
        print(f"Fault diagnosis train page error: {e}")
        raise


@fd_training_bp.route('/training_progress', methods=['GET'])
def training_progress_page():
    """故障诊断训练进度页面"""
    try:
        task_id = request.args.get('task_id')
        if not task_id:
            return render_template('error.html', error_message='缺少任务ID参数'), 400

        return render_template('fault_diagnosis/training_progress.html', task_id=task_id)
    except Exception as e:
        print(f"Fault diagnosis training progress page error: {e}")
        raise


@fd_training_bp.route('/api/train', methods=['POST'])
def train_model():
    """故障诊断模型训练API"""
    try:
        # 获取JSON配置
        model_config = request.get_json()
        if not model_config:
            return jsonify({'status': 'error', 'message': '无效的配置数据'})

        # 调用训练服务
        trainer = get_trainer()
        result = trainer.train(model_config)
        return jsonify(result)

    except Exception as e:
        print(f"Fault diagnosis training error: {e}")
        return jsonify({'status': 'error', 'message': f'训练失败: {str(e)}'}), 500

@fd_training_bp.route('/train/processed_files', methods=['GET'])
def get_processed_files():
    """获取标注后的数据文件列表（从labeled/FaultDiagnosis文件夹）"""
    try:
        # 获取edge目录路径
        edge_root = Path(__file__).resolve().parents[4]  # 从 training.py 到 edge 目录
        # 故障诊断数据从标注目录读取：edge/data/labeled/FaultDiagnosis
        processed_dir = edge_root / 'data' / 'labeled' / 'FaultDiagnosis'
        
        files = []
        
        if not processed_dir.exists():
            return jsonify({
                'status': 'success',
                'files': [],
                'message': '预处理数据目录不存在'
            })
        
        # 遍历processed/fd目录中的所有CSV文件
        for file_path in processed_dir.glob('*.csv'):
            filename = file_path.name
            
            files.append({
                'filename': filename,
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'modified': file_path.stat().st_mtime
            })
        
        # 按修改时间排序（最新的在前）
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'files': files,
            'count': len(files)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取文件列表失败: {str(e)}',
            'files': []
        }), 500


@fd_training_bp.route('/api/processed_data', methods=['GET'])
def get_processed_data_files():
    """获取已标注的数据文件列表（从labeled目录，包含标签和工况信息）"""
    try:
        # 数据文件目录 - 使用labeled目录
        edge_root = Path(__file__).resolve().parents[4]  # 从 training.py 到 edge 目录
        labeled_dir = edge_root / 'data' / 'labeled' / 'FaultDiagnosis'
        
        if not labeled_dir.exists():
            return jsonify({
                'success': True,
                'files': [],
                'message': '标注数据目录不存在 (edge/data/labeled/FaultDiagnosis)'
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
                meta_file_path = edge_root / 'data' / 'meta' / 'FaultDiagnosis' / (filename.replace('.csv', '.json'))
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


@fd_training_bp.route('/api/condition_keys', methods=['GET'])
def get_condition_keys():
    """获取所有工况key列表"""
    try:
        edge_root = Path(__file__).resolve().parents[4]
        meta_dir = edge_root / 'data' / 'meta' / 'FaultDiagnosis'
        
        print(f"🔍 [故障诊断] 查找元数据目录: {meta_dir}")
        print(f"🔍 [故障诊断] 目录是否存在: {meta_dir.exists()}")
        
        if not meta_dir.exists():
            print(f"⚠️ [故障诊断] 元数据目录不存在: {meta_dir}")
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
                    print(f"📄 [故障诊断] 文件 {meta_file.name}: {len(tags_condition)} 个工况")
                    for cond in tags_condition:
                        if isinstance(cond, dict) and 'key' in cond:
                            condition_keys.add(cond['key'])
                            print(f"  - 找到工况key: {cond['key']}")
            except Exception as e:
                print(f"❌ [故障诊断] 读取元文件失败 {meta_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"✅ [故障诊断] 共处理 {file_count} 个元文件，找到 {len(condition_keys)} 个唯一的工况key: {sorted(list(condition_keys))}")
        
        return jsonify({
            'success': True,
            'keys': sorted(list(condition_keys))
        })
        
    except Exception as e:
        print(f"❌ [故障诊断] 获取工况key列表异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'获取工况key列表失败: {str(e)}'
        }), 500


@fd_training_bp.route('/api/condition_values', methods=['GET'])
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
        meta_dir = edge_root / 'data' / 'meta' / 'FaultDiagnosis'
        
        print(f"🔍 [故障诊断] 查找工况值: key={key}, 目录={meta_dir}")
        
        if not meta_dir.exists():
            print(f"⚠️ [故障诊断] 元数据目录不存在: {meta_dir}")
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
                print(f"❌ [故障诊断] 读取元文件失败 {meta_file}: {e}")
                continue
        
        sorted_values = sorted(list(condition_values), key=lambda x: (float(x) if x.replace('.', '').replace('-', '').isdigit() else float('inf'), x))
        print(f"✅ [故障诊断] 找到 {len(sorted_values)} 个值: {sorted_values}")
        
        return jsonify({
            'success': True,
            'key': key,
            'values': sorted_values
        })
        
    except Exception as e:
        print(f"❌ [故障诊断] 获取工况value列表异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'获取工况value列表失败: {str(e)}'
        }), 500


@fd_training_bp.route('/api/filter_files', methods=['POST'])
def filter_files():
    """根据工况条件筛选文件（故障诊断：不过滤标签，显示所有文件）"""
    try:
        data = request.get_json()
        conditions = data.get('conditions', {})  # {key: [value1, value2, ...]}
        file_type = data.get('file_type', 'train')  # 'train' 或 'test'
        
        edge_root = Path(__file__).resolve().parents[4]
        meta_dir = edge_root / 'data' / 'meta' / 'FaultDiagnosis'
        labeled_dir = edge_root / 'data' / 'labeled' / 'FaultDiagnosis'
        
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
                
                # 检查标签（故障诊断：不过滤标签，所有标签都显示）
                tags_label = meta_data.get('tags_label', [])
                if not tags_label:
                    continue
                
                # 获取第一个标签的值
                first_label_value = tags_label[0].get('value', '').strip()
                
                # 检查对应的数据文件是否存在（先获取文件名）
                data_filename = meta_file.stem + '.csv'
                
                # 故障诊断：不根据file_type筛选标签，显示所有文件
                # （训练集和测试集都显示所有标签的文件）
                
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
                print(f"[故障诊断] 处理元文件失败 {meta_file}: {e}")
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

@fd_training_bp.route('/train/download_file', methods=['GET'])
def download_training_file():
    """下载训练数据文件或元文件（供云端调用）"""
    try:
        from flask import send_file
        
        task_id = request.args.get('task_id')
        filename = request.args.get('filename')
        file_type = request.args.get('file_type', 'data')  # 'data' 或 'meta'
        
        if not task_id or not filename:
            return jsonify({
                'status': 'error',
                'message': '缺少参数: task_id 或 filename'
            }), 400
        
        edge_root = Path(__file__).resolve().parents[4]
        
        if file_type == 'meta':
            # 下载元文件
            meta_dir = edge_root / 'data' / 'meta' / 'FaultDiagnosis'
            file_path = meta_dir / filename
        else:
            # 下载数据文件
            training_dir = edge_root / 'data' / 'training' / 'FaultDiagnosis' / task_id
            file_path = training_dir / filename
        
        if not file_path.exists():
            return jsonify({
                'status': 'error',
                'message': f'文件不存在: {filename}'
            }), 404
        
        # 返回文件
        return send_file(
            str(file_path),
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'下载文件失败: {str(e)}'
        }), 500


@fd_training_bp.route('/api/training_status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    """获取训练状态"""
    try:
        trainer = get_trainer()
        status = trainer.get_training_status(task_id)

        if status:
            return jsonify({
                'success': True,
                'task': status
            })
        else:
            return jsonify({
                'success': False,
                'message': '任务不存在或已完成'
            }), 404

    except Exception as e:
        print(f"Get training status error: {e}")
        return jsonify({
            'success': False,
            'message': f'获取训练状态失败: {str(e)}'
        }), 500


@fd_training_bp.route('/api/pause_training/<task_id>', methods=['POST'])
def pause_training(task_id):
    """暂停训练"""
    try:
        trainer = get_trainer()
        result = trainer.pause_training(task_id)

        return jsonify({
            'success': result,
            'message': '训练已暂停' if result else '暂停训练失败'
        })

    except Exception as e:
        print(f"Pause training error: {e}")
        return jsonify({
            'success': False,
            'message': f'暂停训练失败: {str(e)}'
        }), 500


@fd_training_bp.route('/api/stop_training/<task_id>', methods=['POST'])
def stop_training(task_id):
    """停止训练"""
    try:
        trainer = get_trainer()
        result = trainer.stop_training(task_id)

        return jsonify({
            'success': result,
            'message': '训练已停止' if result else '停止训练失败'
        })

    except Exception as e:
        print(f"Stop training error: {e}")
        return jsonify({
            'success': False,
            'message': f'停止训练失败: {str(e)}'
        }), 500


@fd_training_bp.route('/api/download_model/<task_id>', methods=['GET'])
def download_model(task_id):
    """下载训练好的模型"""
    try:
        # 重定向到云端下载
        trainer = get_trainer()
        cloud_url = trainer._get_cloud_url()
        
        # 重定向到云端的下载接口
        from flask import redirect
        return redirect(f'{cloud_url}/api/fault_diagnosis/download_model/{task_id}')

    except Exception as e:
        print(f"Download model error: {e}")
        return jsonify({
            'success': False,
            'message': f'下载模型失败: {str(e)}'
        }), 500


@fd_training_bp.route('/api/download_model_to_edge/<task_id>', methods=['POST'])
def download_model_to_edge(task_id):
    """下载模型到Edge本地（从云端下载zip并解压）"""
    try:
        import requests
        import zipfile
        import tempfile
        import os
        import json as json_module
        
        trainer = get_trainer()
        cloud_url = trainer._get_cloud_url()
        
        # 首先获取模型信息，确定模型类型
        model_type = 'cnn_1d'  # 默认
        try:
            info_response = requests.get(
                f'{cloud_url}/api/fault_diagnosis/models/{task_id}/info',
                timeout=10
            )
            if info_response.status_code == 200:
                info_data = info_response.json()
                if info_data.get('success'):
                    model_info = info_data.get('model_info', {})
                    # 优先使用 model_type_dir（已经是目录名格式）
                    model_type_dir = model_info.get('model_type_dir')
                    if model_type_dir:
                        model_type = model_type_dir
                    else:
                        # 回退：从 model_type 推断目录名
                        raw_model_type = model_info.get('model_type', 'cnn_1d_classifier')
                        if 'resnet' in raw_model_type.lower():
                            model_type = 'resnet_1d'
                        elif 'lstm' in raw_model_type.lower():
                            model_type = 'lstm'
                        else:
                            model_type = 'cnn_1d'
            else:
                print(f"获取模型信息失败: HTTP {info_response.status_code}")
        except Exception as e:
            print(f"获取模型信息失败，使用默认模型类型: {e}")
        
        print(f"正在从云端下载模型: {cloud_url}/api/fault_diagnosis/download_model/{task_id}")
        print(f"模型类型: {model_type}")
        
        # 从云端下载模型（zip文件）
        response = requests.get(
            f'{cloud_url}/api/fault_diagnosis/download_model/{task_id}',
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
        
        # 保存到edge/models/fault_diagnosis/{模型类型}
        edge_root = Path(__file__).resolve().parents[4]
        models_dir = edge_root / 'models' / 'fault_diagnosis' / model_type
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
        
        return jsonify({
            'success': True,
            'message': '模型下载成功',
            'path': str(model_folder),
            'model_type': model_type
        })
        
    except Exception as e:
        print(f"Download model to edge error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'下载模型失败: {str(e)}'
        }), 500


@fd_training_bp.route('/api/evaluation_results/<task_id>', methods=['GET'])
def get_evaluation_results(task_id):
    """获取评估结果"""
    try:
        import requests
        
        trainer = get_trainer()
        cloud_url = trainer._get_cloud_url()
        
        # 直接调用云端的 evaluation_results API
        eval_response = requests.get(
            f'{cloud_url}/api/fault_diagnosis/evaluation_results/{task_id}',
            timeout=10
        )
        
        if eval_response.status_code == 200:
            eval_result = eval_response.json()
            if eval_result.get('success') and eval_result.get('evaluation'):
                return jsonify({
                    'success': True,
                    'evaluation': eval_result['evaluation']
                })
        
        # 如果直接获取失败，尝试从任务状态中获取
        status_response = requests.get(
            f'{cloud_url}/api/fault_diagnosis/training_status/{task_id}',
            timeout=10
        )
        
        if status_response.status_code != 200:
            return jsonify({
                'success': False,
                'message': '无法获取任务状态'
            }), 404
        
        status_result = status_response.json()
        if not status_result.get('success'):
            return jsonify({
                'success': False,
                'message': '任务不存在'
            }), 404
        
        task = status_result.get('task', {})
        if task.get('status') not in ['completed', 'finished']:
            return jsonify({
                'success': False,
                'message': '训练尚未完成'
            }), 400
        
        # 尝试从任务状态中获取评估结果
        evaluation = task.get('evaluation_results') or task.get('evaluation')
        
        if evaluation:
            return jsonify({
                'success': True,
                'evaluation': evaluation
            })
        else:
            return jsonify({
                'success': False,
                'message': '评估结果不可用'
            }), 404
        
    except Exception as e:
        print(f"Get evaluation results error: {e}")
        return jsonify({
            'success': False,
            'message': f'获取评估结果失败: {str(e)}'
        }), 500