"""
异常检测模型管理路由
"""

from flask import Blueprint, request, jsonify, send_from_directory, send_file, current_app
import os
import json
from datetime import datetime

ad_models_bp = Blueprint('ad_models', __name__, url_prefix='/anomaly_detection')

@ad_models_bp.route('/api/models', methods=['GET'])
def list_models():
    """获取模型列表API - 结合本地和云端模型"""
    try:
        # 1. 读取本地模型
        local_models = _get_local_models()
        
        # 2. 获取云端可用模型列表（失败不影响本地模型显示）
        cloud_models = []
        try:
            cloud_models = _get_cloud_models()
        except Exception as e:
            # 云端模型获取失败不影响本地模型显示
            print(f"警告: 获取云端模型列表失败，仅显示本地模型: {e}")
        
        # 3. 合并并标记模型状态
        all_models = _merge_model_lists(local_models, cloud_models)
        
        return jsonify({
            'success': True,
            'models': all_models,
            'total_count': len(all_models),
            'local_count': len(local_models),
            'cloud_count': len(cloud_models)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def _get_local_models():
    """获取本地模型列表"""
    models = []
    # 修复路径：确保指向 edge/models/anomaly_detection
    from pathlib import Path
    edge_dir = Path(__file__).resolve().parents[4]  # 回到edge目录
    models_dir = str(edge_dir / 'models' / 'anomaly_detection')
    
    if os.path.exists(models_dir):
        # 遍历所有模型类型目录
        for model_type in os.listdir(models_dir):
            model_type_dir = os.path.join(models_dir, model_type)
            if not os.path.isdir(model_type_dir):
                continue
                
            # 遍历所有任务目录
            for task_id in os.listdir(model_type_dir):
                task_dir = os.path.join(model_type_dir, task_id)
                if not os.path.isdir(task_dir):
                    continue
                    
                # 检查必要文件
                config_path = os.path.join(task_dir, 'config.json')
                model_files = [f for f in os.listdir(task_dir) 
                             if f.endswith(('.ckpt', '.pth', '.h5'))]
                
                if not os.path.exists(config_path) or not model_files:
                    continue
                
                try:
                    # 加载配置
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    model_file = model_files[0]
                    model_path = os.path.join(task_dir, model_file)
                    model_stat = os.stat(model_path)
                    
                    # 检查阈值文件
                    threshold_info = None
                    threshold_path = os.path.join(task_dir, 'threshold.json')
                    if os.path.exists(threshold_path):
                        try:
                            with open(threshold_path, 'r', encoding='utf-8') as f:
                                threshold_info = json.load(f)
                        except:
                            pass
                    
                    model_info = {
                        'task_id': task_id,
                        'model_type': model_type,
                        'filename': model_file,
                        'size': model_stat.st_size,
                        'created_at': datetime.fromtimestamp(model_stat.st_ctime).isoformat(),
                        'modified_at': datetime.fromtimestamp(model_stat.st_mtime).isoformat(),
                        'config': {
                            'sequence_length': config.get('sequence_length', 50),
                            'hidden_units': config.get('hidden_units', 128),
                            'num_layers': config.get('num_layers', 2),
                            'epochs': config.get('epochs', 50),
                            'batch_size': config.get('batch_size', 32),
                            'feature_dim': config.get('feature_dim'),
                        },
                        'files': {
                            'model': {
                                'exists': True,
                                'size': model_stat.st_size,
                                'filename': model_file
                            },
                            'scaler': {
                                'exists': os.path.exists(os.path.join(task_dir, 'scaler.pkl'))
                            },
                            'threshold': {
                                'exists': os.path.exists(threshold_path),
                                'value': threshold_info.get('threshold_value') if threshold_info else None,
                                'method': threshold_info.get('method') if threshold_info else None
                            }
                        },
                        'status': 'local',  # 标记为本地已有
                        'download_url': f"/anomaly_detection/api/models/{task_id}/download",
                        'info_url': f"/anomaly_detection/api/models/{task_id}/info"
                    }
                    
                    models.append(model_info)
                    
                except Exception as e:
                    print(f"读取本地模型配置失败 {task_dir}: {e}")
                    continue
    
    return models

def _get_cloud_models():
    """获取云端模型列表"""
    try:
        import requests
        from requests.exceptions import RequestException, Timeout, ConnectionError
        
        # 从配置中获取云端服务URL
        cloud_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
        try:
            response = requests.get(f"{cloud_url}/api/anomaly_detection/models", timeout=5)
            
            if response.status_code == 200:
                cloud_data = response.json()
                # 检查返回数据格式
                if cloud_data.get('success'):
                    models = cloud_data.get('models', [])
                    # 记录调试信息
                    print(f"从云端获取到 {len(models)} 个模型")
                    return models
                else:
                    # 如果success为False，说明有错误
                    error_msg = cloud_data.get('error', '云端返回失败状态')
                    print(f"云端返回失败: {error_msg}")
                    raise Exception(f"云端返回失败: {error_msg}")
            else:
                error_msg = f"获取云端模型列表失败: HTTP {response.status_code}"
                print(error_msg)
                raise Exception(error_msg)
        except Timeout:
            error_msg = f"连接云端服务器超时（{cloud_url}），请检查云端服务器是否运行"
            print(error_msg)
            raise Exception(error_msg)
        except ConnectionError as e:
            error_msg = f"无法连接到云端服务器（{cloud_url}），请确保云端服务器已启动: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
        except RequestException as e:
            error_msg = f"获取云端模型列表异常: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    except Exception as e:
        # 重新抛出异常，让调用者知道具体错误
        raise

def _merge_model_lists(local_models, cloud_models):
    """合并本地和云端模型列表"""
    # 创建本地模型的task_id集合
    local_task_ids = {model['task_id'] for model in local_models}
    
    # 处理云端模型
    merged_models = local_models.copy()
    
    for cloud_model in cloud_models:
        task_id = cloud_model['task_id']
        
        if task_id in local_task_ids:
            # 本地已有此模型，标记为已同步
            for local_model in merged_models:
                if local_model['task_id'] == task_id:
                    local_model['status'] = 'synced'
                    local_model['cloud_available'] = True
                    break
        else:
            # 本地没有此模型，添加为可下载
            cloud_model['status'] = 'cloud_only'
            cloud_model['cloud_available'] = True
            cloud_model['download_url'] = f"/anomaly_detection/api/models/{task_id}/sync"  # 下载接口
            merged_models.append(cloud_model)
    
    # 按创建时间降序排列
    merged_models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return merged_models

@ad_models_bp.route('/api/models/<task_id>/download', methods=['GET'])
def download_model(task_id):
    """下载模型文件 - 从本地edge模型目录"""
    try:
        import tempfile
        import zipfile
        
        # 查找本地模型目录
        models_dir = os.path.join(current_app.root_path, 'models', 'anomaly_detection')
        task_dir = None
        
        if os.path.exists(models_dir):
            for model_type in os.listdir(models_dir):
                model_type_dir = os.path.join(models_dir, model_type)
                if not os.path.isdir(model_type_dir):
                    continue
                potential_task_dir = os.path.join(model_type_dir, task_id)
                if os.path.exists(potential_task_dir):
                    task_dir = potential_task_dir
                    break
        
        if not task_dir:
            return jsonify({'error': '模型不存在'}), 404
        
        # 创建临时ZIP文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加所有模型文件到ZIP
                for filename in os.listdir(task_dir):
                    file_path = os.path.join(task_dir, filename)
                    if os.path.isfile(file_path):
                        zipf.write(file_path, filename)
            
            from flask import send_file
            return send_file(
                tmp_file.name,
                as_attachment=True,
                download_name=f"{task_id}_model.zip",
                mimetype='application/zip'
            )
    
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@ad_models_bp.route('/api/models/<task_id>', methods=['DELETE'])
def delete_model(task_id):
    """删除本地模型"""
    try:
        import shutil
        from pathlib import Path
        
        # 查找本地模型目录
        edge_dir = Path(__file__).resolve().parents[4]  # 回到edge目录
        models_dir = edge_dir / 'models' / 'anomaly_detection'
        
        task_dir = None
        if models_dir.exists():
            for model_type in os.listdir(models_dir):
                model_type_dir = models_dir / model_type
                if not model_type_dir.is_dir():
                    continue
                potential_task_dir = model_type_dir / task_id
                if potential_task_dir.exists():
                    task_dir = potential_task_dir
                    break
        
        if not task_dir:
            return jsonify({'success': False, 'error': '模型不存在'}), 404
        
        # 删除模型目录
        shutil.rmtree(task_dir)
        
        return jsonify({
            'success': True,
            'message': f'模型 {task_id} 已成功删除'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'删除失败: {str(e)}'}), 500

@ad_models_bp.route('/api/models/<task_id>/info', methods=['GET'])
def get_model_info(task_id):
    """获取模型详细信息 - 从本地edge模型目录"""
    try:
        # 查找本地模型目录
        models_dir = os.path.join(current_app.root_path, 'models', 'anomaly_detection')
        task_dir = None
        
        if os.path.exists(models_dir):
            for model_type in os.listdir(models_dir):
                model_type_dir = os.path.join(models_dir, model_type)
                if not os.path.isdir(model_type_dir):
                    continue
                potential_task_dir = os.path.join(model_type_dir, task_id)
                if os.path.exists(potential_task_dir):
                    task_dir = potential_task_dir
                    break
        
        if not task_dir:
            return jsonify({'error': '模型不存在'}), 404
        
        # 收集所有信息
        model_info = {
            'task_id': task_id,
            'model_type': os.path.basename(os.path.dirname(task_dir)),
            'files': {},
            'config': {},
            'training_logs': []
        }
        
        # 配置文件
        config_path = os.path.join(task_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                model_info['config'] = json.load(f)
        
        # 文件信息
        for filename in os.listdir(task_dir):
            file_path = os.path.join(task_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                model_info['files'][filename] = {
                    'size': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取模型信息失败: {str(e)}'
        }), 500


@ad_models_bp.route('/api/models/<task_id>/sync', methods=['POST'])
def sync_model(task_id):
    """从云端下载并同步模型"""
    try:
        # 1. 检查本地是否已有模型
        models_dir = os.path.join(current_app.root_path, 'models', 'anomaly_detection')
        local_model_info = _find_local_model(task_id)
        
        if local_model_info:
            return jsonify({
                'success': True, 
                'message': '模型已存在于本地',
                'status': 'already_exists',
                'model': local_model_info
            })
        
        # 2. 从云端下载模型
        download_success, download_info = download_model_from_cloud(task_id)
        
        if not download_success:
            return jsonify({
                'success': False, 
                'error': download_info.get('error', '下载失败')
            }), 400
        
        # 3. 验证下载的模型
        downloaded_model = _find_local_model(task_id)
        if not downloaded_model:
            return jsonify({
                'success': False, 
                'error': '下载后无法找到模型文件'
            }), 500
        
        return jsonify({
            'success': True,
            'message': '模型下载同步成功',
            'status': 'downloaded',
            'model': downloaded_model,
            'download_info': download_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ad_models_bp.route('/api/models/sync_all', methods=['POST'])
def sync_all_models():
    """批量同步所有云端模型到本地"""
    try:
        # 1. 获取本地模型列表
        local_models = _get_local_models()
        
        # 2. 尝试获取云端模型列表（失败时给出友好提示）
        cloud_models = []
        cloud_error = None
        try:
            cloud_models = _get_cloud_models()
        except Exception as e:
            cloud_error = str(e)
            print(f"获取云端模型列表失败: {cloud_error}")
        
        # 3. 如果无法获取云端模型列表，返回友好错误信息
        if cloud_error:
            # 从配置中获取云端服务URL用于错误提示
            cloud_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
            # 根据错误类型提供更具体的提示
            error_msg = cloud_error
            if 'timeout' in cloud_error.lower() or 'timed out' in cloud_error.lower():
                error_msg = f'连接云端服务器超时，请检查云端服务器是否运行（{cloud_url}）'
            elif 'connection' in cloud_error.lower() or 'refused' in cloud_error.lower() or '无法连接' in cloud_error:
                error_msg = f'无法连接到云端服务器（{cloud_url}），请确保云端服务器已启动'
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'cloud_available': False
            }), 400
        
        # 如果云端返回空列表，给出提示但不作为错误（可能确实没有模型）
        if not cloud_models:
            return jsonify({
                'success': True,
                'message': '云端暂无模型可同步',
                'synced_count': 0,
                'failed_count': 0,
                'total_count': 0,
                'results': [],
                'cloud_available': True,
                'empty_list': True
            })
        
        # 2. 找出需要下载的模型（云端有但本地没有的）
        local_task_ids = {model['task_id'] for model in local_models}
        models_to_sync = [model for model in cloud_models if model['task_id'] not in local_task_ids]
        
        if not models_to_sync:
            return jsonify({
                'success': True,
                'message': '所有云端模型已同步到本地',
                'synced_count': 0,
                'total_count': len(cloud_models),
                'results': []
            })
        
        # 3. 逐个下载模型
        results = []
        success_count = 0
        fail_count = 0
        
        for model in models_to_sync:
            task_id = model['task_id']
            try:
                download_success, download_info = download_model_from_cloud(task_id)
                
                if download_success:
                    success_count += 1
                    results.append({
                        'task_id': task_id,
                        'status': 'success',
                        'message': download_info.get('message', '下载成功')
                    })
                else:
                    fail_count += 1
                    results.append({
                        'task_id': task_id,
                        'status': 'failed',
                        'error': download_info.get('error', '下载失败')
                    })
            except Exception as e:
                fail_count += 1
                results.append({
                    'task_id': task_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'message': f'同步完成：成功 {success_count} 个，失败 {fail_count} 个',
            'synced_count': success_count,
            'failed_count': fail_count,
            'total_count': len(models_to_sync),
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'批量同步失败: {str(e)}'
        }), 500

def download_model_from_cloud(task_id):
    """从云端下载指定模型到本地的内部函数"""
    try:
        import requests
        import zipfile
        import tempfile
        
        # 从配置中获取云端服务URL
        cloud_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
        
        # 1. 先从云端获取模型信息，确定模型类型
        info_response = requests.get(f"{cloud_url}/api/anomaly_detection/models/{task_id}/info", timeout=10)
        model_type_dir = 'lstm_prediction'  # 默认值
        
        if info_response.status_code == 200:
            info_data = info_response.json()
            if info_data.get('success') and 'model_info' in info_data:
                cloud_model_type = info_data['model_info'].get('model_type', '')
                # 将云端模型类型映射到本地目录名
                if cloud_model_type == 'lstm_predictor':
                    model_type_dir = 'lstm_prediction'
                elif cloud_model_type == 'lstm_autoencoder':
                    model_type_dir = 'lstm_autoencoder'
                elif cloud_model_type == 'cnn_1d_autoencoder':
                    model_type_dir = 'cnn_1d_autoencoder'
                else:
                    # 如果云端返回的是目录名格式，直接使用
                    model_type_dir = cloud_model_type if cloud_model_type else 'lstm_prediction'
        
        # 2. 从云端下载模型包
        response = requests.get(f"{cloud_url}/api/anomaly_detection/models/{task_id}/download_package", 
                               stream=True, timeout=30)
        
        if response.status_code != 200:
            return False, {
                'error': f'从云端下载失败: HTTP {response.status_code}'
            }
        
        # 3. 创建本地模型目录（使用正确的模型类型目录）
        models_dir = os.path.join(current_app.root_path, 'models', 'anomaly_detection')
        task_dir = os.path.join(models_dir, model_type_dir, task_id)
        
        # 清理现有目录
        if os.path.exists(task_dir):
            import shutil
            shutil.rmtree(task_dir, ignore_errors=True)
        
        # 创建新目录
        os.makedirs(task_dir, exist_ok=True)
        
        # 保存下载的ZIP文件到临时位置
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_zip_path = tmp_file.name
        
        # 解压ZIP文件到模型目录
        extract_count = 0
        with zipfile.ZipFile(tmp_zip_path, 'r') as zipf:
            zipf.extractall(task_dir)
            extract_count = len(zipf.namelist())
        
        # 清理临时文件
        os.unlink(tmp_zip_path)
        
        return True, {
            'message': f'模型 {task_id} 已成功下载到本地 ({extract_count} 个文件)',
            'task_dir': task_dir,
            'files_count': extract_count
        }
        
    except requests.exceptions.RequestException as e:
        return False, {
            'error': f'网络请求失败: {str(e)}'
        }
    except Exception as e:
        return False, {
            'error': f'下载模型失败: {str(e)}'
        }

@ad_models_bp.route('/api/evaluations', methods=['GET'])
def list_evaluations():
    """获取评估结果列表API"""
    try:
        evaluations_dir = os.path.join(current_app.root_path, 'models', 'anomaly_detection', 'evaluations')
        if not os.path.exists(evaluations_dir):
            return jsonify({'success': True, 'evaluations': []})

        evaluations = []
        for dirname in os.listdir(evaluations_dir):
            eval_path = os.path.join(evaluations_dir, dirname)
            if os.path.isdir(eval_path):
                # 检查是否有评估结果文件
                metrics_file = os.path.join(eval_path, 'metrics.json')
                summary_file = os.path.join(eval_path, 'evaluation_summary.json')

                eval_data = {
                    'name': dirname,
                    'created_at': datetime.fromtimestamp(os.path.getctime(eval_path)).isoformat(),
                    'has_results': False,
                    'has_summary': False
                }

                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r', encoding='utf-8') as f:
                            eval_data['metrics'] = json.load(f)
                        eval_data['has_results'] = True
                    except:
                        pass

                if os.path.exists(summary_file):
                    eval_data['has_summary'] = True

                evaluations.append(eval_data)

        # 按创建时间排序，最新的在前
        evaluations.sort(key=lambda x: x['created_at'], reverse=True)

        return jsonify({'success': True, 'evaluations': evaluations})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ad_models_bp.route('/api/evaluations/<eval_name>/summary/download', methods=['GET'])
def download_evaluation_summary(eval_name):
    """下载评估摘要"""
    try:
        evaluations_dir = os.path.join(current_app.root_path, 'models', 'anomaly_detection', 'evaluations', eval_name)
        return send_from_directory(evaluations_dir, 'evaluation_summary.json', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@ad_models_bp.route('/api/download_model_from_cloud/<task_id>', methods=['POST'])
def download_model_from_cloud_after_training(task_id):
    """训练完成后从云端下载模型到本地"""
    try:
        import requests
        import zipfile
        import tempfile
        import os
        
        # 从配置中获取云端服务器地址
        cloud_server_base = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
        
        # 1. 先从云端获取模型信息，确定模型类型
        info_response = requests.get(f"{cloud_server_base}/api/anomaly_detection/models/{task_id}/info", timeout=10)
        model_type_dir = 'lstm_prediction'  # 默认值
        
        if info_response.status_code == 200:
            info_data = info_response.json()
            if info_data.get('success') and 'model_info' in info_data:
                cloud_model_type = info_data['model_info'].get('model_type', '')
                # 将云端模型类型映射到本地目录名
                if cloud_model_type == 'lstm_predictor':
                    model_type_dir = 'lstm_prediction'
                elif cloud_model_type == 'lstm_autoencoder':
                    model_type_dir = 'lstm_autoencoder'
                elif cloud_model_type == 'cnn_1d_autoencoder':
                    model_type_dir = 'cnn_1d_autoencoder'
                else:
                    # 如果云端返回的是目录名格式，直接使用
                    model_type_dir = cloud_model_type if cloud_model_type else 'lstm_prediction'
        
        # 2. 从云端下载模型包
        response = requests.get(f"{cloud_server_base}/api/anomaly_detection/models/{task_id}/download_package", 
                               stream=True, timeout=60)
        
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': f'从云端服务器下载失败: HTTP {response.status_code}'
            }), response.status_code
        
        # 3. 本地目标路径（使用正确的模型类型目录）
        local_models_base = os.path.join(current_app.root_path, 'models', 'anomaly_detection', model_type_dir)
        local_task_dir = os.path.join(local_models_base, task_id)
        
        # 确保目录存在
        os.makedirs(local_task_dir, exist_ok=True)
        
        # 清理已存在的文件
        if os.path.exists(local_task_dir):
            import shutil
            for file_name in os.listdir(local_task_dir):
                file_path = os.path.join(local_task_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        
        # 保存ZIP文件到临时位置
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            tmp_zip_path = tmp_file.name
        
        # 解压ZIP文件到本地目标目录
        extracted_files = []
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_file:
            zip_file.extractall(local_task_dir)
            extracted_files = zip_file.namelist()
        
        # 清理临时ZIP文件
        os.unlink(tmp_zip_path)
        
        # 验证关键文件是否存在
        config_path = os.path.join(local_task_dir, 'config.json')
        model_files = [f for f in extracted_files if f.endswith(('.ckpt', '.pth', '.h5'))]
        
        if not os.path.exists(config_path):
            return jsonify({
                'success': False,
                'error': '下载的模型缺少配置文件 config.json'
            }), 500
        
        if not model_files:
            return jsonify({
                'success': False,
                'error': '下载的模型缺少模型权重文件'
            }), 500
        
        return jsonify({
            'success': True,
            'message': f'模型 {task_id} 已成功下载到本地',
            'local_path': local_task_dir,
            'files_count': len(extracted_files),
            'extracted_files': extracted_files,
            'model_files': model_files
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'网络请求失败: {str(e)}'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'下载模型失败: {str(e)}'
        }), 500

def _find_local_model(task_id):
    """查找本地特定task_id的模型"""
    models_dir = os.path.join(current_app.root_path, 'models', 'anomaly_detection')
    
    if not os.path.exists(models_dir):
        return None
        
    for model_type in os.listdir(models_dir):
        model_type_dir = os.path.join(models_dir, model_type)
        if not os.path.isdir(model_type_dir):
            continue
            
        task_dir = os.path.join(model_type_dir, task_id)
        if not os.path.exists(task_dir):
            continue
            
        config_path = os.path.join(task_dir, 'config.json')
        if not os.path.exists(config_path):
            continue
            
        # 找到对应的模型文件
        model_files = [f for f in os.listdir(task_dir) 
                      if f.endswith(('.ckpt', '.pth', '.h5'))]
        
        if not model_files:
            continue
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            model_file = model_files[0]
            model_path = os.path.join(task_dir, model_file)
            model_stat = os.stat(model_path)
            
            return {
                'task_id': task_id,
                'model_type': model_type,
                'filename': model_file,
                'size': model_stat.st_size,
                'created_at': datetime.fromtimestamp(model_stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(model_stat.st_mtime).isoformat(),
                'config': config,
                'path': task_dir
            }
            
        except Exception as e:
            print(f"读取模型配置失败 {task_dir}: {e}")
            continue
    
    return None