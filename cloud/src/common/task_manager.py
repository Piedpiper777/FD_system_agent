"""
云端训练任务管理系统
支持异步训练任务的提交、查询和管理
"""

import uuid
import threading
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# 设置日志
logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """训练任务状态"""
    QUEUED = "queued"  # 已提交
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


@dataclass
class TrainingTask:
    """训练任务数据类"""
    task_id: str
    module: str
    model_type: str
    output_path: str
    input_dim: int
    data_path: str = None  # 可选，某些模式可能不需要
    dataset_mode: str = 'one'  # 新增：数据集模式
    _raw_config: dict = None  # 保存完整的原始配置（包括train_files, test_files等）
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    train_ratio: float = None
    val_ratio: float = None
    test_ratio: float = None
    val_ratio_from_train: float = None
    sequence_length: int = 50
    prediction_horizon: int = 1
    hidden_units: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = 'tanh'
    bidirectional: bool = False
    preprocess_method: str = None
    status: str = 'queued'
    created_at: str = None
    updated_at: str = None
    progress: int = 0
    message: str = ""
    logs: str = ""
    error: str = ""
    model_save_path: str = ""  # 新增：模型保存路径
    edge_host: str = None  # 新增：边缘端主机
    edge_port: int = None  # 新增：边缘端端口
    dataset_file: str = None  # 新增：数据集文件
    train_file: str = None  # 新增：训练文件
    val_file: str = None  # 新增：验证文件
    test_file: str = None  # 新增：测试文件
    scaler_path: str = ""
    threshold_path: str = ""
    threshold_value: float = None
    threshold_metadata: dict = None
    # 训练进度跟踪
    current_epoch: int = 0
    completed_epochs: int = 0
    # 当前训练轮次的损失值（用于实时显示）
    current_train_loss: float = None
    current_val_loss: float = None
    # 最终损失值（用于前端显示）
    final_train_loss: float = None
    final_val_loss: float = None
    # 阈值计算参数
    threshold_method: str = 'percentile'  # percentile, 3sigma, contamination
    percentile: float = 95.0
    residual_metric: str = 'rmse'  # rmse, l1, max

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()

    @property
    def config(self):
        """返回训练配置字典"""
        # 如果有原始配置，优先使用原始配置（包含train_files, test_files等完整信息）
        if self._raw_config is not None:
            return self._raw_config
        
        # 否则返回从字段构建的配置（向后兼容）
        return {
            'module': self.module,
            'model_type': self.model_type,
            'data_path': self.data_path,
            'output_path': self.output_path,
            'input_dim': self.input_dim,
            'dataset_mode': self.dataset_mode,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'val_ratio_from_train': self.val_ratio_from_train,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'hidden_units': self.hidden_units,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'activation': self.activation,
            'bidirectional': self.bidirectional,
            'preprocess_method': self.preprocess_method,
            'edge_host': self.edge_host,
            'edge_port': self.edge_port,
            'dataset_file': self.dataset_file,
            'train_file': self.train_file,
            'val_file': self.val_file,
            'test_file': self.test_file,
            'threshold_method': self.threshold_method,
            'percentile': self.percentile,
            'residual_metric': self.residual_metric,
        }

    def to_dict(self):
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'module': self.module,
            'model_type': self.model_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'model_save_path': self.model_save_path,  # 包含模型保存路径
            'current_epoch': self.current_epoch,
            'completed_epochs': self.completed_epochs,
            'current_train_loss': self.current_train_loss,
            'current_val_loss': self.current_val_loss,
            'final_train_loss': self.final_train_loss,
            'final_val_loss': self.final_val_loss,
            'scaler_path': self.scaler_path,
            'threshold_path': self.threshold_path,
            'threshold_value': self.threshold_value,
            'threshold_metadata': self.threshold_metadata or {},
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'config': {
                'module': self.module,
                'model_type': self.model_type,
                'data_path': self.data_path,
                'output_path': self.output_path,
                'input_dim': self.input_dim,
                'dataset_mode': self.dataset_mode,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'validation_split': self.validation_split,
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
                'val_ratio_from_train': self.val_ratio_from_train,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'hidden_units': self.hidden_units,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'activation': self.activation,
                'bidirectional': self.bidirectional,
                'preprocess_method': self.preprocess_method,
                'edge_host': self.edge_host,  # 新增
                'edge_port': self.edge_port,  # 新增
                'dataset_file': self.dataset_file,  # 新增
                'train_file': self.train_file,  # 新增
                'val_file': self.val_file,  # 新增
                'test_file': self.test_file,  # 新增
                'threshold_method': self.threshold_method,
                'percentile': self.percentile,
                'residual_metric': self.residual_metric,
            }
        }


class TrainingTaskManager:
    """训练任务管理器"""

    def __init__(self):
        """初始化任务管理器"""
        self.tasks: Dict[str, TrainingTask] = {}
        self.task_threads: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        self._task_counter = 0  # 用于避免同一毫秒内的重复
        self._last_ad_task_timestamp: Optional[str] = None

    def _generate_task_id(self, module: Optional[str] = None) -> str:
        """生成基于时间的任务ID
        
        格式: YYYYMMDD_HHMMSS_XXX
        示例: 20251119_143052_001
        
        Returns:
            str: 唯一的任务ID
        """
        if module == 'anomaly_detection':
            while True:
                time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
                with self.lock:
                    if time_part != self._last_ad_task_timestamp and time_part not in self.tasks:
                        self._last_ad_task_timestamp = time_part
                        return time_part
                time.sleep(0.2)

        with self.lock:
            self._task_counter = (self._task_counter + 1) % 1000
            now = datetime.now()
            time_part = now.strftime("%Y%m%d_%H%M%S")
            counter_part = f"{self._task_counter:03d}"
            return f"{time_part}_{counter_part}"

    def create_task(self, config: dict) -> TrainingTask:
        """创建训练任务
        
        Args:
            config (dict): 训练配置
        
        Returns:
            TrainingTask: 新创建的训练任务
        """
        module_name = config.get('module', 'anomaly_detection')
        task_id = self._generate_task_id(module_name)
        
        # 处理数据划分参数 - 兼容不同的参数格式
        validation_split = 0.2  # 默认值
        
        # 对于LSTM预测器，使用val_ratio作为validation_split
        if config.get('model_type') == 'lstm_predictor':
            if 'val_ratio' in config:
                validation_split = float(config['val_ratio'])
            elif 'validation_split' in config:
                validation_split = float(config['validation_split'])
        else:
            # 其他模型使用标准的validation_split参数
            validation_split = float(config.get('validation_split', 0.2))
        
        def _to_int(value, default):
            try:
                if value is None or value == '':
                    return default
                return int(value)
            except (TypeError, ValueError):
                return default

        def _to_float(value, default):
            try:
                if value is None or value == '':
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default

        def _to_bool(value, default=False):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ('true', '1', 'yes', 'y')
            return bool(value)

        sequence_length = _to_int(config.get('sequence_length', config.get('seq_len')), 50)
        prediction_horizon = _to_int(config.get('prediction_horizon', config.get('pred_len')), 1)
        hidden_units = _to_int(config.get('hidden_units', config.get('hidden_dim')), 128)
        num_layers = _to_int(config.get('num_layers'), 2)
        dropout = _to_float(config.get('dropout'), 0.1)
        activation = config.get('activation', 'tanh')
        bidirectional = _to_bool(config.get('bidirectional'), False)

        train_ratio = _to_float(config.get('train_ratio'), None)
        val_ratio = _to_float(config.get('val_ratio'), None)
        test_ratio = _to_float(config.get('test_ratio'), None)
        val_ratio_from_train = _to_float(config.get('val_ratio_from_train'), None)

        preprocess_method = config.get('preprocess_method')

        # 处理阈值计算参数
        threshold_method = config.get('threshold_method', 'percentile')
        percentile = _to_float(config.get('percentile'), 95.0)
        residual_metric = config.get('residual_metric', 'rmse')

        task = TrainingTask(
            task_id=task_id,
            module=module_name,
            model_type=config.get('model_type', 'lstm'),
            data_path=config.get('data_path') or config.get('dataset_file'),  # 尝试多种字段名
            output_path=config.get('output_path', f'output/model_{task_id}.ckpt'),
            input_dim=_to_int(config.get('input_dim'), 10),
            dataset_mode=config.get('dataset_mode', 'one'),
            epochs=_to_int(config.get('epochs'), 100),
            batch_size=_to_int(config.get('batch_size'), 32),
            learning_rate=_to_float(config.get('learning_rate'), 0.001),
            validation_split=validation_split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            val_ratio_from_train=val_ratio_from_train,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            hidden_units=hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            bidirectional=bidirectional,
            preprocess_method=preprocess_method,
            edge_host=config.get('edge_host'),  # 新增
            edge_port=int(config.get('edge_port', 5000)) if config.get('edge_port') else None,  # 新增
            dataset_file=config.get('dataset_file'),  # 新增
            train_file=config.get('train_file'),  # 新增
            val_file=config.get('val_file'),  # 新增
            test_file=config.get('test_file'),  # 新增
            threshold_method=threshold_method,
            percentile=percentile,
            residual_metric=residual_metric,
        )
        
        # 调试：打印接收到的配置和创建的任务参数
        print(f"🔍 任务创建调试信息:")
        print(f"   接收到的config: {config}")
        print(f"   创建的任务epochs: {task.epochs}")
        print(f"   创建的任务batch_size: {task.batch_size}")
        print(f"   创建的任务learning_rate: {task.learning_rate}")

        # 保存完整的原始配置（包括train_files, test_files等）
        task._raw_config = config.copy()

        with self.lock:
            self.tasks[task_id] = task

        logger.info(f"Created training task: {task_id}")
        return task

    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """获取训练任务
        
        Args:
            task_id (str): 任务ID
        
        Returns:
            TrainingTask: 训练任务，如果不存在则返回 None
        """
        with self.lock:
            return self.tasks.get(task_id)

    def list_tasks(self, status: Optional[str] = None) -> List[TrainingTask]:
        """列出所有训练任务
        
        Args:
            status (str, optional): 筛选状态
        
        Returns:
            List[TrainingTask]: 任务列表
        """
        with self.lock:
            if status:
                return [t for t in self.tasks.values() if t.status == status]
            return list(self.tasks.values())

    def update_task_status(self, task_id: str, status: str, message: str = "", progress: int = None, current_epoch: int = None, train_loss: float = None, val_loss: float = None):
        """更新任务状态
        
        Args:
            task_id (str): 任务ID
            status (str): 新状态
            message (str): 状态消息
            progress (int): 进度百分比
            current_epoch (int): 当前训练轮次
            train_loss (float): 当前训练损失值
            val_loss (float): 当前验证损失值
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = status
                task.updated_at = datetime.now().isoformat()
                if message:
                    task.message = message
                if progress is not None:
                    task.progress = progress
                if current_epoch is not None:
                    task.current_epoch = current_epoch
                    # 如果状态为完成，更新completed_epochs
                    if status in ['completed', 'threshold_completed']:
                        task.completed_epochs = current_epoch
                if train_loss is not None:
                    task.current_train_loss = train_loss
                if val_loss is not None:
                    task.current_val_loss = val_loss
                logger.info(f"Updated task {task_id}: status={status}, progress={progress}%, epoch={current_epoch}, train_loss={train_loss}, val_loss={val_loss}")

    def update_final_losses(self, task_id: str, train_loss: float = None, val_loss: float = None):
        """更新任务的最终损失值
        
        Args:
            task_id (str): 任务ID
            train_loss (float): 最终训练损失
            val_loss (float): 最终验证损失
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                if train_loss is not None:
                    task.final_train_loss = train_loss
                if val_loss is not None:
                    task.final_val_loss = val_loss
                task.updated_at = datetime.now().isoformat()
                logger.info(f"Updated final losses for task {task_id}: train_loss={train_loss}, val_loss={val_loss}")

    def update_model_save_path(self, task_id: str, model_path: str):
        """更新模型保存路径
        
        Args:
            task_id (str): 任务ID
            model_path (str): 模型保存路径
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.model_save_path = model_path
                task.updated_at = datetime.now().isoformat()
                logger.info(f"Updated model save path for task {task_id}: {model_path}")

    def update_scaler_path(self, task_id: str, scaler_path: str):
        """记录标准化参数文件路径"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.scaler_path = scaler_path
                task.updated_at = datetime.now().isoformat()
                logger.info(f"Updated scaler path for task {task_id}: {scaler_path}")

    def update_threshold_info(self, task_id: str, threshold_path: str, threshold_value: float, metadata: Optional[dict] = None):
        """记录阈值文件及信息"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.threshold_path = threshold_path
                task.threshold_value = threshold_value
                task.threshold_metadata = metadata or {}
                task.updated_at = datetime.now().isoformat()
                logger.info(
                    f"Updated threshold info for task {task_id}: value={threshold_value}, path={threshold_path}"
                )

    def update_task_logs(self, task_id: str, logs: str):
        """更新任务日志
        
        Args:
            task_id (str): 任务ID
            logs (str): 日志内容
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.logs = logs

    def add_log(self, task_id: str, log_message: str):
        """添加日志消息
        
        Args:
            task_id (str): 任务ID
            log_message (str): 日志消息
        """
        # 直接使用日志系统记录
        logger.info(f"[{task_id}] {log_message}")
        
        # 同时累积到任务对象的logs字段，添加时间戳
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                # 添加时间戳到日志消息（格式：YYYY-MM-DD HH:MM:SS,mmm）
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]  # 保留毫秒
                log_with_timestamp = f"{timestamp} - {log_message}"
                if task.logs:
                    task.logs += f"\n{log_with_timestamp}"
                else:
                    task.logs = log_with_timestamp

    def fail_task(self, task_id: str, error: str):
        """标记任务为失败
        
        Args:
            task_id (str): 任务ID
            error (str): 错误信息
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = 'failed'
                task.error = error
                task.updated_at = datetime.now().isoformat()
                logger.error(f"Task {task_id} failed: {error}")

    def start_training(self, task_id: str, training_function):
        """启动异步训练任务
        
        Args:
            task_id (str): 任务ID
            training_function (callable): 训练函数
        """
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        # 标记为运行中
        self.update_task_status(task_id, 'running', "开始训练...")

        # 创建后台线程执行训练
        thread = threading.Thread(
            target=self._execute_training,
            args=(task_id, training_function),
            daemon=True
        )
        thread.start()
        self.task_threads[task_id] = thread

    def _execute_training(self, task_id: str, training_function):
        """执行训练的内部函数（在后台线程中运行）
        
        Args:
            task_id (str): 任务ID
            training_function (callable): 训练函数
        """
        try:
            task = self.get_task(task_id)
            if not task:
                return

            # 执行训练函数，传递task_id而不是task对象
            result = training_function(task_id)

            # 检查训练函数是否返回了结果
            if result is None:
                # 如果没有返回结果，检查任务当前状态
                current_task = self.get_task(task_id)
                if current_task and current_task.status == 'completed':
                    logger.info(f"Task {task_id} completed successfully (no return result)")
                else:
                    logger.warning(f"Task {task_id} finished without result and status is: {current_task.status if current_task else 'not found'}")
            elif result.get('success'):
                # 只有在训练函数明确返回成功结果时才更新状态（避免重复更新）
                logger.info(f"Task {task_id} completed successfully with result")
            else:
                # 处理明确的失败结果
                error_msg = result.get('error', '训练失败')
                self.fail_task(task_id, error_msg)

        except Exception as e:
            error_msg = f"训练异常: {str(e)}"
            self.fail_task(task_id, error_msg)
            logger.exception(f"Error in training task {task_id}")

    def cancel_task(self, task_id: str) -> bool:
        """取消训练任务
        
        Args:
            task_id (str): 任务ID
        
        Returns:
            bool: 是否成功取消
        """
        task = self.get_task(task_id)
        if not task:
            return False

        if task.status in ['completed', 'failed']:
            return False

        self.update_task_status(task_id, 'cancelled', "任务已取消")
        return True


# 全局任务管理器实例
task_manager = TrainingTaskManager()


def get_task_manager() -> TrainingTaskManager:
    """获取全局任务管理器实例
    
    Returns:
        TrainingTaskManager: 任务管理器
    """
    return task_manager
