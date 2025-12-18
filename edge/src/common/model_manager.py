"""
边侧模型管理器
负责从云端下载训练好的模型，管理本地模型文件
"""

import os
import json
import shutil
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EdgeModelManager:
    """边侧模型管理器"""
    
    def __init__(self, edge_root: str = None):
        self.edge_root = Path(edge_root) if edge_root else Path(__file__).parent.parent.parent
        self.models_dir = self.edge_root / 'models'
        self.data_dir = self.edge_root / 'data'
        self.inference_tasks_dir = self.edge_root / 'inference_tasks'
        
        # 创建基础目录结构
        self._setup_directories()
        
    def _setup_directories(self):
        """创建边侧目录结构"""
        directories = [
            # 数据目录
            self.data_dir / 'collected',
            self.data_dir / 'processed', 
            self.data_dir / 'uploaded',
            
            # 模型目录
            self.models_dir / 'anomaly_detection',
            self.models_dir / 'fault_diagnosis',
            self.models_dir / 'rul_prediction',
            
            # 推理任务目录
            self.inference_tasks_dir / 'anomaly_detection',
            self.inference_tasks_dir / 'fault_diagnosis',
            self.inference_tasks_dir / 'rul_prediction'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def download_trained_model(self, module: str, model_type: str, task_id: str, 
                             cloud_url: str = None) -> bool:
        """
        从云端下载训练好的模型到边侧
        
        Args:
            module: 模块名 (anomaly_detection, fault_diagnosis, rul_prediction)
            model_type: 模型类型 (lstm_predictor, svm_classifier等)
            task_id: 训练任务ID
            cloud_url: 云端服务URL（如果为None，将从Flask配置中获取）
            
        Returns:
            bool: 下载是否成功
        """
        try:
            # 如果未提供 cloud_url，尝试从 Flask 配置中获取
            if cloud_url is None:
                try:
                    from flask import current_app
                    cloud_url = current_app.config.get('CLOUD_BASE_URL', 'http://localhost:5001')
                except RuntimeError:
                    # 不在 Flask 应用上下文中，使用默认值
                    cloud_url = 'http://localhost:5001'
            # 创建本地模型目录
            local_model_dir = self.models_dir / module / model_type / f"task_{task_id}"
            local_model_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取云端模型文件列表
            files_to_download = [
                'config.json',
                'model.ckpt', 
                'scaler.pkl',
                'scaler.pkl.npz',
                'threshold.json',
                'threshold.npz'
            ]
            
            downloaded_files = []
            
            for filename in files_to_download:
                try:
                    # 从云端下载文件
                    download_url = f"{cloud_url}/api/models/download/{module}/{model_type}/task_{task_id}/{filename}"
                    response = requests.get(download_url, timeout=30)
                    
                    if response.status_code == 200:
                        local_file_path = local_model_dir / filename
                        with open(local_file_path, 'wb') as f:
                            f.write(response.content)
                        downloaded_files.append(filename)
                        logger.info(f"Downloaded {filename} to {local_file_path}")
                    elif response.status_code == 404:
                        logger.debug(f"File {filename} not found on cloud (optional)")
                    else:
                        logger.warning(f"Failed to download {filename}: {response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"Error downloading {filename}: {e}")
                    
            # 检查必要文件是否下载成功
            required_files = ['config.json', 'model.ckpt']
            missing_required = [f for f in required_files if f not in downloaded_files]
            
            if missing_required:
                logger.error(f"Missing required files: {missing_required}")
                return False
                
            # 创建下载记录
            download_record = {
                'task_id': task_id,
                'module': module,
                'model_type': model_type,
                'downloaded_at': datetime.now().isoformat(),
                'downloaded_files': downloaded_files,
                'cloud_url': cloud_url
            }
            
            record_path = local_model_dir / 'download_record.json'
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(download_record, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully downloaded model {task_id} to {local_model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model {task_id}: {e}")
            return False
            
    def list_local_models(self, module: str = None) -> List[Dict]:
        """
        列出本地已下载的模型
        
        Args:
            module: 模块名过滤，None表示所有模块
            
        Returns:
            List[Dict]: 模型信息列表
        """
        models = []
        
        modules_to_check = [module] if module else ['anomaly_detection', 'fault_diagnosis', 'rul_prediction']
        
        for mod in modules_to_check:
            module_dir = self.models_dir / mod
            if not module_dir.exists():
                continue
                
            for model_type_dir in module_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                    
                for task_dir in model_type_dir.iterdir():
                    if not task_dir.is_dir() or not task_dir.name.startswith('task_'):
                        continue
                        
                    # 读取配置文件
                    config_path = task_dir / 'config.json'
                    download_record_path = task_dir / 'download_record.json'
                    
                    model_info = {
                        'module': mod,
                        'model_type': model_type_dir.name,
                        'task_id': task_dir.name.replace('task_', ''),
                        'local_path': str(task_dir),
                        'has_config': config_path.exists(),
                        'has_model': (task_dir / 'model.ckpt').exists(),
                        'has_scaler': any((task_dir / f).exists() for f in ['scaler.pkl', 'scaler.pkl.npz']),
                        'has_threshold': any((task_dir / f).exists() for f in ['threshold.json', 'threshold.npz'])
                    }
                    
                    # 读取配置信息
                    if config_path.exists():
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            model_info.update({
                                'created_at': config.get('created_at'),
                                'model_params': {
                                    'sequence_length': config.get('sequence_length'),
                                    'input_dim': config.get('input_dim'),
                                    'hidden_units': config.get('hidden_units'),
                                    'epochs_trained': config.get('epochs_trained')
                                }
                            })
                        except Exception as e:
                            logger.warning(f"Failed to read config for {task_dir}: {e}")
                            
                    # 读取下载记录
                    if download_record_path.exists():
                        try:
                            with open(download_record_path, 'r', encoding='utf-8') as f:
                                download_record = json.load(f)
                            model_info['downloaded_at'] = download_record.get('downloaded_at')
                            model_info['downloaded_files'] = download_record.get('downloaded_files', [])
                        except Exception as e:
                            logger.warning(f"Failed to read download record for {task_dir}: {e}")
                            
                    models.append(model_info)
                    
        return sorted(models, key=lambda x: x.get('downloaded_at', ''), reverse=True)
        
    def create_inference_task(self, module: str, model_type: str, model_task_id: str) -> Path:
        """
        创建推理任务目录
        
        Args:
            module: 模块名
            model_type: 模型类型  
            model_task_id: 源训练任务ID
            
        Returns:
            Path: 推理任务目录路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        counter = 1
        
        base_dir = self.inference_tasks_dir / module / model_type
        base_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            task_id = f"{timestamp}_{counter:03d}"
            inference_dir = base_dir / f'inference_{task_id}'
            if not inference_dir.exists():
                inference_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建推理任务配置
                config = {
                    'inference_task_id': task_id,
                    'module': module,
                    'model_type': model_type,
                    'source_model_task_id': model_task_id,
                    'created_at': datetime.now().isoformat(),
                    'status': 'created'
                }
                
                with open(inference_dir / 'config.json', 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                    
                return inference_dir
            counter += 1
            
    def copy_model_for_inference(self, inference_dir: Path, model_task_id: str, 
                                module: str, model_type: str) -> bool:
        """
        将本地模型文件复制到推理任务目录
        
        Args:
            inference_dir: 推理任务目录
            model_task_id: 模型任务ID
            module: 模块名
            model_type: 模型类型
            
        Returns:
            bool: 复制是否成功
        """
        try:
            source_model_dir = self.models_dir / module / model_type / f'task_{model_task_id}'
            if not source_model_dir.exists():
                logger.error(f"Source model directory not found: {source_model_dir}")
                return False
                
            # 创建模型子目录
            model_dir = inference_dir / 'model'
            model_dir.mkdir(exist_ok=True)
            
            # 复制模型文件
            files_to_copy = ['config.json', 'model.ckpt', 'scaler.pkl', 'scaler.pkl.npz', 
                           'threshold.json', 'threshold.npz']
            
            copied_files = []
            for filename in files_to_copy:
                source_file = source_model_dir / filename
                if source_file.exists():
                    target_file = model_dir / filename
                    shutil.copy2(source_file, target_file)
                    copied_files.append(filename)
                    
            logger.info(f"Copied model files to {model_dir}: {copied_files}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy model files: {e}")
            return False
            
    def cleanup_old_models(self, module: str = None, keep_recent: int = 5):
        """
        清理旧的模型文件，保留最近的几个
        
        Args:
            module: 模块名，None表示所有模块
            keep_recent: 保留最近的模型数量
        """
        modules_to_clean = [module] if module else ['anomaly_detection', 'fault_diagnosis', 'rul_prediction']
        
        for mod in modules_to_clean:
            module_dir = self.models_dir / mod
            if not module_dir.exists():
                continue
                
            for model_type_dir in module_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                    
                # 获取所有任务目录，按时间排序
                task_dirs = []
                for task_dir in model_type_dir.iterdir():
                    if task_dir.is_dir() and task_dir.name.startswith('task_'):
                        download_record_path = task_dir / 'download_record.json'
                        if download_record_path.exists():
                            try:
                                with open(download_record_path, 'r', encoding='utf-8') as f:
                                    record = json.load(f)
                                downloaded_at = record.get('downloaded_at', '')
                                task_dirs.append((downloaded_at, task_dir))
                            except:
                                # 如果无法读取下载记录，使用目录修改时间
                                mtime = datetime.fromtimestamp(task_dir.stat().st_mtime).isoformat()
                                task_dirs.append((mtime, task_dir))
                                
                # 删除旧的模型（保留最近的keep_recent个）
                task_dirs.sort(reverse=True)  # 按时间倒序
                for _, task_dir in task_dirs[keep_recent:]:
                    try:
                        shutil.rmtree(task_dir)
                        logger.info(f"Cleaned up old model: {task_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {task_dir}: {e}")


# 全局实例
edge_model_manager = EdgeModelManager()