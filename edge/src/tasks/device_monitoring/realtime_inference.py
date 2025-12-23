"""
实时推理服务模块
提供基于监控数据的实时异常检测推理功能
"""

import json
import time
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Blueprint, jsonify, request
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from pandas.errors import ParserError

# 实时推理相关变量
realtime_detector = None
realtime_inference_running = False
realtime_inference_thread = None
realtime_model_task_id = None
realtime_inference_results = []  # 存储最近的推理结果
max_results_history = 100  # 最多保存100条历史记录
realtime_inference_waiting = False  # 是否正在等待数据积累
realtime_inference_waiting_message = ""  # 等待状态消息

DEFAULT_MONITOR_COLUMNS = ['timestamp', 'col_0', 'col_1', 'col_2', 'col_3']


def _safe_read_monitor_csv(monitor_file_path: Path) -> Optional[pd.DataFrame]:
    """Robustly load monitor CSV even if headers are missing or rows are malformed."""
    if not monitor_file_path.exists():
        return None

    read_kwargs = {'engine': 'python', 'on_bad_lines': 'skip'}
    primary_error = None

    try:
        data = pd.read_csv(monitor_file_path, **read_kwargs)
        if 'timestamp' in data.columns:
            return data
    except ParserError as exc:
        primary_error = exc
    except Exception as exc:  # pylint: disable=broad-except
        primary_error = exc

    try:
        fallback = pd.read_csv(monitor_file_path, header=None, **read_kwargs)
        if fallback.empty:
            return fallback

        column_count = fallback.shape[1]
        if column_count == 0:
            return fallback

        if column_count <= len(DEFAULT_MONITOR_COLUMNS):
            columns = DEFAULT_MONITOR_COLUMNS[:column_count]
        else:
            columns = ['timestamp'] + [f'col_{idx}' for idx in range(column_count - 1)]

        fallback = fallback.iloc[:, :len(columns)]
        fallback.columns = columns
        return fallback
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[实时推理] 读取监控文件失败: {exc}")
        if primary_error:
            print(f"[实时推理] 原始读取错误: {primary_error}")
        return None


def _get_required_sample_count(detector=None) -> int:
    """Return number of consecutive samples needed for realtime inference window."""
    detector = detector or realtime_detector
    if detector is None:
        return 0
    model_type = getattr(detector, 'model_type', '')
    if model_type in ['lstm_autoencoder', 'cnn_1d_autoencoder']:
        return detector.sequence_length
    # Predictor-style models need one extra point for the target step.
    return detector.sequence_length + 1


def _load_realtime_detector(task_id: str) -> bool:
    """加载实时推理检测器"""
    global realtime_detector, realtime_model_task_id
    
    try:
        # 查找模型目录
        models_dir = Path(__file__).resolve().parents[3] / 'models' / 'anomaly_detection'
        model_dir = None
        
        for model_type_dir in models_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
            task_dir = model_type_dir / task_id
            if task_dir.exists() and task_dir.is_dir():
                model_dir = task_dir
                break
        
        if not model_dir:
            raise FileNotFoundError(f"找不到模型目录: {task_id}")
        
        # 读取模型配置
        config_path = model_dir / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        
        model_type = model_config.get('model_type', 'lstm_predictor')
        sequence_length = model_config.get('sequence_length', 50)
        
        # 检查必要文件
        # 支持 .pth 和 .ckpt 格式
        model_path = model_dir / 'model.pth'
        if not model_path.exists():
            model_path = model_dir / 'model.ckpt'  # 兼容旧格式
        threshold_path = model_dir / 'threshold.json'
        scaler_path = model_dir / 'scaler.pkl'
        
        if not all([model_path.exists(), threshold_path.exists(), scaler_path.exists()]):
            raise FileNotFoundError(f"模型文件不完整: {model_dir}")
        
        # 导入并初始化检测器
        from src.tasks.anomaly_detection.services.inferencer import LocalAnomalyDetector
        
        realtime_detector = LocalAnomalyDetector(
            model_path=model_path,
            threshold_path=threshold_path,
            scaler_path=scaler_path,
            sequence_length=sequence_length,
            model_type=model_type
        )
        
        realtime_model_task_id = task_id
        
        print(f"[实时推理] 模型加载成功: {task_id} (类型: {model_type}, 序列长度: {sequence_length})")
        return True
        
    except Exception as e:
        print(f"[实时推理] 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        realtime_detector = None
        realtime_model_task_id = None
        return False


def _get_continuous_data_count(monitor_file_path: Path, max_gap_seconds: float = 2.0) -> Tuple[int, Optional[datetime]]:
    """
    获取从最新数据开始往前回溯的连续数据条数
    Args:
        monitor_file_path: 监控文件路径
        max_gap_seconds: 允许的最大时间间隔（秒），超过此间隔认为数据不连续
    Returns:
        (连续数据条数, 最新数据的时间戳)
    """
    try:
        if not monitor_file_path.exists():
            return 0, None

        data = _safe_read_monitor_csv(monitor_file_path)

        if data is None or len(data) == 0:
            return 0, None
        
        # 确保timestamp列存在
        if 'timestamp' not in data.columns:
            return 0, None
        
        # 将timestamp转换为datetime
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        except:
            return 0, None
        
        # 从最新数据开始往前检查
        data = data.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        if len(data) == 0:
            return 0, None
        
        latest_timestamp = data.iloc[0]['timestamp']
        continuous_count = 1
        previous_timestamp = latest_timestamp

        # 往前检查连续性：只要相邻数据的时间差不超过阈值，就认为是连续的
        for i in range(1, len(data)):
            current_timestamp = data.iloc[i]['timestamp']
            time_gap = (previous_timestamp - current_timestamp).total_seconds()

            # 如果时间倒流或间隔过大，则认为数据不连续
            if time_gap < 0:
                time_gap = abs(time_gap)
            if time_gap > max_gap_seconds:
                break

            continuous_count += 1
            previous_timestamp = current_timestamp
        
        return continuous_count, latest_timestamp
        
    except Exception as e:
        print(f"[实时推理] 检查连续数据失败: {e}")
        import traceback
        traceback.print_exc()
        return 0, None


def _preprocess_monitor_data(data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
    """
    对监控数据进行预处理（默认使用z-score标准化）
    
    Args:
        data: 原始监控数据 DataFrame（包含 timestamp 和 col_0, col_1, col_2, col_3）
        method: 预处理方法，默认 'zscore'
    
    Returns:
        预处理后的 DataFrame（保留 timestamp 列）
    """
    try:
        # 复制数据，避免修改原始数据
        processed_data = data.copy()
        
        # 提取特征列（排除 timestamp）
        feature_columns = [col for col in data.columns if col != 'timestamp']
        
        if not feature_columns:
            return processed_data
        
        # 提取特征数据
        feature_data = data[feature_columns].values
        
        # 检查是否有NaN值
        if np.isnan(feature_data).any():
            print(f"[实时推理] 警告: 数据中包含NaN值，将使用前向填充处理")
            feature_df = pd.DataFrame(feature_data, columns=feature_columns)
            feature_df = feature_df.ffill().fillna(0)  # 前向填充，如果还有NaN则填充0
            feature_data = feature_df.values
        
        # 应用预处理
        if method == 'zscore':
            # Z-score标准化
            scaler = StandardScaler()
            processed_features = scaler.fit_transform(feature_data)
        else:
            # 默认不处理（如果以后需要支持其他方法）
            processed_features = feature_data
        
        # 更新DataFrame中的特征列
        for i, col in enumerate(feature_columns):
            processed_data[col] = processed_features[:, i]
        
        return processed_data
        
    except Exception as e:
        print(f"[实时推理] 数据预处理失败: {e}")
        import traceback
        traceback.print_exc()
        # 如果预处理失败，返回原始数据
        return data


def _get_current_data_count(monitor_file_path: Path) -> int:
    """获取当前监控文件中的连续数据条数（兼容旧接口）"""
    count, _ = _get_continuous_data_count(monitor_file_path)
    return count


def _read_latest_data_window(monitor_file_path: Path, sequence_length: int, max_gap_seconds: float = 2.0) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    从监控文件读取最新的连续数据窗口
    Args:
        monitor_file_path: 监控文件路径
        sequence_length: 需要的序列长度
        max_gap_seconds: 允许的最大时间间隔（秒），超过此间隔认为数据不连续
    Returns:
        (数据窗口DataFrame, 错误信息)
    """
    try:
        if not monitor_file_path.exists():
            return None, "监控文件不存在"
        
        # 读取CSV文件
        data = _safe_read_monitor_csv(monitor_file_path)
        
        if data is None or len(data) == 0:
            return None, "监控文件为空"
        
        # 确保timestamp列存在
        if 'timestamp' not in data.columns:
            return None, "监控文件缺少timestamp列"
        
        # 将timestamp转换为datetime
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        except Exception as e:
            return None, f"时间戳格式错误: {str(e)}"
        
        # 按时间戳排序（从新到旧）
        data = data.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        # 检查连续数据量
        continuous_count, latest_timestamp = _get_continuous_data_count(monitor_file_path, max_gap_seconds)
        
        if continuous_count < sequence_length:
            return None, f"连续数据不足，需要至少 {sequence_length} 条连续数据，当前只有 {continuous_count} 条连续数据"
        
        # 获取最新的连续 sequence_length 条数据
        window_data = data.head(sequence_length).copy()
        
        # 按时间正序排列（从旧到新），符合模型输入要求
        window_data = window_data.sort_values('timestamp', ascending=True).reset_index(drop=True)
        
        return window_data, None
        
    except Exception as e:
        return None, f"读取数据失败: {str(e)}"


def _wait_for_sufficient_data(monitor_file_path: Path, sequence_length: int, max_wait_time: int = 300) -> bool:
    """
    等待连续数据积累到足够的数量
    Args:
        monitor_file_path: 监控文件路径
        sequence_length: 需要的序列长度
        max_wait_time: 最大等待时间（秒），默认5分钟
    Returns:
        True: 数据已足够，False: 超时
    """
    global realtime_inference_waiting, realtime_inference_waiting_message
    
    realtime_inference_waiting = True
    start_time = time.time()
    check_interval = 1.0  # 每秒检查一次
    
    while time.time() - start_time < max_wait_time:
        if not realtime_inference_running:
            # 如果推理被停止，退出等待
            realtime_inference_waiting = False
            return False
        
        continuous_count, latest_timestamp = _get_continuous_data_count(monitor_file_path)
        
        if latest_timestamp:
            # 计算数据的新鲜度（距离现在的时间）
            now = datetime.now()
            if isinstance(latest_timestamp, pd.Timestamp):
                latest_dt = latest_timestamp.to_pydatetime()
            else:
                latest_dt = latest_timestamp
            
            age_seconds = (now - latest_dt).total_seconds()
            
            if age_seconds > 5:
                # 数据太旧，可能系统刚启动
                realtime_inference_waiting_message = f"等待新数据: {continuous_count}/{sequence_length} (最新数据: {int(age_seconds)}秒前)"
            else:
                realtime_inference_waiting_message = f"等待连续数据积累: {continuous_count}/{sequence_length}"
        else:
            realtime_inference_waiting_message = f"等待连续数据积累: {continuous_count}/{sequence_length}"
        
        if continuous_count >= sequence_length:
            realtime_inference_waiting = False
            realtime_inference_waiting_message = ""
            print(f"[实时推理] 连续数据已足够，开始推理 (当前: {continuous_count} 条连续数据)")
            return True
        
        time.sleep(check_interval)
    
    # 超时
    realtime_inference_waiting = False
    realtime_inference_waiting_message = ""
    print(f"[实时推理] 等待连续数据超时 (最大等待时间: {max_wait_time} 秒)")
    return False


def _run_realtime_inference_loop():
    """实时推理循环"""
    global realtime_inference_running, realtime_detector, realtime_inference_results, realtime_inference_waiting, realtime_inference_waiting_message
    
    monitor_file_path = Path(__file__).resolve().parents[3] / 'data' / 'monitor' / 'monitor_data.csv'
    
    # 首次启动时，等待数据积累
    if realtime_detector is not None:
        required_sample_count = _get_required_sample_count(realtime_detector)
        print(f"[实时推理] 检查初始数据量，需要 {required_sample_count} 条数据")
        if not _wait_for_sufficient_data(monitor_file_path, required_sample_count):
            print("[实时推理] 数据等待失败或超时，推理循环退出")
            realtime_inference_running = False
            return
    
    while realtime_inference_running:
        try:
            if realtime_detector is None:
                time.sleep(1)
                continue
            required_sample_count = _get_required_sample_count(realtime_detector)
            
            # 读取最新数据窗口
            window_data, error = _read_latest_data_window(
                monitor_file_path,
                required_sample_count
            )
            
            if window_data is None:
                # 如果连续数据不足，进入等待状态
                if not realtime_inference_waiting:
                    print(f"[实时推理] {error}，等待连续数据积累...")
                    realtime_inference_waiting = True
                
                continuous_count, latest_timestamp = _get_continuous_data_count(monitor_file_path)
                
                if latest_timestamp:
                    # 计算数据的新鲜度
                    now = datetime.now()
                    if isinstance(latest_timestamp, pd.Timestamp):
                        latest_dt = latest_timestamp.to_pydatetime()
                    else:
                        latest_dt = latest_timestamp
                    
                    age_seconds = (now - latest_dt).total_seconds()
                    
                    if age_seconds > 5:
                        realtime_inference_waiting_message = (
                            f"等待新数据: {continuous_count}/{required_sample_count} (最新数据: {int(age_seconds)}秒前)"
                        )
                    else:
                        realtime_inference_waiting_message = (
                            f"等待连续数据积累: {continuous_count}/{required_sample_count}"
                        )
                else:
                    realtime_inference_waiting_message = (
                        f"等待连续数据积累: {continuous_count}/{required_sample_count}"
                    )
                
                # 检查是否已经有足够连续数据
                if required_sample_count and continuous_count >= required_sample_count:
                    realtime_inference_waiting = False
                    realtime_inference_waiting_message = ""
                    continue  # 重新尝试读取数据
                
                time.sleep(1)
                continue
            
            # 对原始监控数据进行预处理（z-score标准化）
            # 注意：训练时使用的数据是预处理后的，所以监控数据也需要先预处理
            preprocessed_data = _preprocess_monitor_data(window_data)
            
            # 使用检测器的preprocess_data方法（会使用scaler.pkl进行标准化）
            sequences, targets = realtime_detector.preprocess_data(preprocessed_data)
            
            if len(sequences) == 0:
                time.sleep(1)
                continue
            
            # 只使用最后一个序列进行推理（最新的数据）
            last_sequence = sequences[-1:].astype(np.float32)
            last_target = targets[-1:]
            
            # 执行推理
            predictions, anomaly_scores, anomaly_flags = realtime_detector.detect_anomalies(
                last_sequence, last_target
            )
            
            # 获取最新时间戳
            if 'timestamp' in window_data.columns:
                ts = window_data.iloc[-1]['timestamp']
                # 如果是pandas Timestamp，转换为ISO格式字符串
                if isinstance(ts, pd.Timestamp):
                    timestamp = ts.isoformat()
                elif isinstance(ts, datetime):
                    timestamp = ts.isoformat()
                else:
                    timestamp = str(ts)
            else:
                timestamp = datetime.now().isoformat()
            
            # 保存结果
            result = {
                'timestamp': timestamp,
                'anomaly_score': float(anomaly_scores[0]),
                'is_anomaly': bool(anomaly_flags[0]),
                'threshold': realtime_detector.threshold_value
            }
            
            realtime_inference_results.append(result)
            
            # 限制历史记录数量
            if len(realtime_inference_results) > max_results_history:
                realtime_inference_results.pop(0)
            
            # 每秒推理一次
            time.sleep(1.0)
            
        except Exception as e:
            print(f"[实时推理] 推理过程出错: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)


def get_available_models() -> Dict[str, Any]:
    """获取可用于实时推理的模型列表"""
    try:
        models_dir = Path(__file__).resolve().parents[3] / 'models' / 'anomaly_detection'
        models = []
        
        if not models_dir.exists():
            return {
                'success': True,
                'models': [],
                'total': 0
            }
        
        # 遍历所有模型类型目录
        for model_type_dir in models_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
            
            # 遍历所有任务目录
            for task_dir in model_type_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                # 检查必要文件（支持 .pth 和 .ckpt 格式）
                config_path = task_dir / 'config.json'
                model_path = task_dir / 'model.pth'
                if not model_path.exists():
                    model_path = task_dir / 'model.ckpt'  # 兼容旧格式
                threshold_path = task_dir / 'threshold.json'
                scaler_path = task_dir / 'scaler.pkl'
                
                if not all([config_path.exists(), model_path.exists(), threshold_path.exists(), scaler_path.exists()]):
                    continue
                
                try:
                    # 读取配置
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # 使用配置文件中的model_type，而不是目录名
                    # 目录名可能是 lstm_prediction，但配置中是 lstm_predictor
                    model_type = config.get('model_type', model_type_dir.name)
                    
                    models.append({
                        'task_id': task_dir.name,
                        'model_type': model_type,
                        'sequence_length': config.get('sequence_length', 50),
                        'hidden_units': config.get('hidden_units', 128),
                        'epochs': config.get('epochs', 50),
                        'trained_at': config.get('trained_at', ''),
                        'threshold_value': None  # 稍后从threshold.json读取
                    })
                    
                    # 读取阈值
                    try:
                        with open(threshold_path, 'r', encoding='utf-8') as f:
                            threshold_info = json.load(f)
                            models[-1]['threshold_value'] = threshold_info.get(
                                'threshold_value',
                                threshold_info.get('threshold')
                            )
                    except Exception:
                        pass
                        
                except Exception as e:
                    print(f"读取模型配置失败 {task_dir}: {e}")
                    continue
        
        # 按训练时间降序排列
        models.sort(key=lambda x: x.get('trained_at', ''), reverse=True)
        
        return {
            'success': True,
            'models': models,
            'total': len(models)
        }
        
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'获取模型列表失败: {str(e)}'
        }


def start_inference(task_id: str) -> Dict[str, Any]:
    """启动实时推理"""
    global realtime_inference_running, realtime_inference_thread, realtime_inference_waiting, realtime_inference_waiting_message
    
    if realtime_inference_running:
        return {
            'success': False,
            'error': '实时推理已在运行中'
        }
    
    try:
        # 加载模型
        if not _load_realtime_detector(task_id):
            return {
                'success': False,
                'error': '模型加载失败'
            }
        
        # 检查初始连续数据量
        monitor_file_path = Path(__file__).resolve().parents[3] / 'data' / 'monitor' / 'monitor_data.csv'
        required_count = _get_required_sample_count(realtime_detector)
        continuous_count, latest_timestamp = _get_continuous_data_count(monitor_file_path)
        
        if continuous_count < required_count:
            if latest_timestamp:
                now = datetime.now()
                if isinstance(latest_timestamp, pd.Timestamp):
                    latest_dt = latest_timestamp.to_pydatetime()
                else:
                    latest_dt = latest_timestamp
                age_seconds = (now - latest_dt).total_seconds()
                
                if age_seconds > 5:
                    print(
                        f"[实时推理] 连续数据不足，当前 {continuous_count} 条连续数据，需要 {required_count} 条，最新数据 {int(age_seconds)} 秒前，将在后台等待新数据"
                    )
                    realtime_inference_waiting_message = (
                        f"等待新数据: {continuous_count}/{required_count} (最新数据: {int(age_seconds)}秒前)"
                    )
                else:
                    print(
                        f"[实时推理] 连续数据不足，当前 {continuous_count} 条连续数据，需要 {required_count} 条，将在后台等待数据积累"
                    )
                    realtime_inference_waiting_message = f"等待连续数据积累: {continuous_count}/{required_count}"
            else:
                print(
                    f"[实时推理] 连续数据不足，当前 {continuous_count} 条连续数据，需要 {required_count} 条，将在后台等待数据积累"
                )
                realtime_inference_waiting_message = f"等待连续数据积累: {continuous_count}/{required_count}"
            
            realtime_inference_waiting = True
        else:
            realtime_inference_waiting = False
            realtime_inference_waiting_message = ""
        
        # 启动推理线程
        realtime_inference_running = True
        realtime_inference_results.clear()
        realtime_inference_thread = threading.Thread(target=_run_realtime_inference_loop, daemon=True)
        realtime_inference_thread.start()
        
        print(f"[实时推理] 实时推理已启动，使用模型: {task_id}")
        
        return {
            'success': True,
            'message': '实时推理已启动' + (
                f'，等待连续数据积累 ({continuous_count}/{required_count})' if continuous_count < required_count else ''
            ),
            'task_id': task_id,
            'waiting': continuous_count < required_count,
            'current_count': continuous_count,
            'required_count': required_count
        }
        
    except Exception as e:
        print(f"[实时推理] 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'启动实时推理失败: {str(e)}'
        }


def stop_inference() -> Dict[str, Any]:
    """停止实时推理"""
    global realtime_inference_running, realtime_detector, realtime_model_task_id, realtime_inference_waiting, realtime_inference_waiting_message
    
    if not realtime_inference_running:
        return {
            'success': False,
            'error': '实时推理未运行'
        }
    
    try:
        realtime_inference_running = False
        realtime_inference_waiting = False
        realtime_inference_waiting_message = ""
        
        # 等待线程结束
        if realtime_inference_thread and realtime_inference_thread.is_alive():
            realtime_inference_thread.join(timeout=2.0)
        
        # 清理资源
        realtime_detector = None
        realtime_model_task_id = None
        
        print("[实时推理] 实时推理已停止")
        
        return {
            'success': True,
            'message': '实时推理已停止'
        }
        
    except Exception as e:
        print(f"[实时推理] 停止失败: {e}")
        return {
            'success': False,
            'error': f'停止实时推理失败: {str(e)}'
        }


def get_inference_status() -> Dict[str, Any]:
    """获取实时推理状态"""
    global realtime_inference_running, realtime_inference_results, realtime_model_task_id, realtime_detector
    global realtime_inference_waiting, realtime_inference_waiting_message
    
    latest_result = realtime_inference_results[-1] if realtime_inference_results else None
    
    # 构建返回结果（兼容前端期望的格式）
    result = {
        'running': realtime_inference_running,
        'waiting': realtime_inference_waiting,
        'waiting_message': realtime_inference_waiting_message,
        'latest': latest_result,
        'model_info': None
    }
    
    if realtime_detector and realtime_model_task_id:
        result['model_info'] = {
            'task_id': realtime_model_task_id,
            'model_type': realtime_detector.model_type,
            'sequence_length': realtime_detector.sequence_length,
            'threshold_value': realtime_detector.threshold_value
        }
        
        # 如果正在等待，添加当前连续数据量信息
        if realtime_inference_waiting:
            monitor_file_path = Path(__file__).resolve().parents[3] / 'data' / 'monitor' / 'monitor_data.csv'
            continuous_count, latest_timestamp = _get_continuous_data_count(monitor_file_path)
            result['current_data_count'] = continuous_count
            result['required_data_count'] = _get_required_sample_count(realtime_detector)
            if latest_timestamp:
                now = datetime.now()
                if isinstance(latest_timestamp, pd.Timestamp):
                    latest_dt = latest_timestamp.to_pydatetime()
                else:
                    latest_dt = latest_timestamp
                age_seconds = (now - latest_dt).total_seconds()
                result['latest_data_age_seconds'] = int(age_seconds)
    
    # 转换为前端期望的格式
    if latest_result:
        result['results'] = {
            'latest_score': latest_result.get('anomaly_score'),
            'latest_is_anomaly': latest_result.get('is_anomaly'),
            'latest_timestamp': latest_result.get('timestamp'),
            'model_info': result['model_info'],
            'score_history': realtime_inference_results[-100:]  # 最近100条
        }
    else:
        result['results'] = {
            'latest_score': None,
            'latest_is_anomaly': False,
            'latest_timestamp': None,
            'model_info': result['model_info'],
            'score_history': []
        }
    
    return result


def get_status_info() -> Dict[str, Any]:
    """获取状态信息（用于 /api/status）"""
    global realtime_inference_running, realtime_model_task_id
    return {
        'realtime_inference': realtime_inference_running,
        'realtime_model': realtime_model_task_id
    }

