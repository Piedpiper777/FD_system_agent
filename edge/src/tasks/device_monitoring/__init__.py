"""
设备监测模块
提供实时设备数据监测功能
"""

import json
import time
import threading
import csv
import os
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Blueprint, render_template, jsonify, request
from datetime import datetime

# 创建蓝图
device_monitoring_bp = Blueprint('device_monitoring', __name__, url_prefix='/device_monitoring')

# 配置文件路径
EDGE_DIR = Path(__file__).resolve().parents[3]
SENSOR_CONFIG_FILE = EDGE_DIR / 'data' / 'monitor' / 'sensor_config.json'
MONITOR_DATA_FILE = EDGE_DIR / 'data' / 'monitor' / 'monitor_data.csv'

# 默认传感器配置
DEFAULT_SENSOR_CONFIG = [
    {
        'id': 'sensor_0',
        'name': 'X轴加速度',
        'description': '轴承径向（水平）振动',
        'unit': 'm/s²',
        'column': 'col_0',
        'color': '#0d6efd',  # 蓝色
        'yAxisMin': -3,
        'yAxisMax': 3,
        'normalThreshold': 1.0,
        'cautionThreshold': 1.5
    },
    {
        'id': 'sensor_1',
        'name': 'Y轴加速度',
        'description': '轴承轴向（垂直）振动',
        'unit': 'm/s²',
        'column': 'col_1',
        'color': '#198754',  # 绿色
        'yAxisMin': -3,
        'yAxisMax': 3,
        'normalThreshold': 1.0,
        'cautionThreshold': 1.5
    },
    {
        'id': 'sensor_2',
        'name': 'Z轴加速度',
        'description': '轴承轴向位移/垂直振动',
        'unit': 'm/s²',
        'column': 'col_2',
        'color': '#0dcaf0',  # 青色
        'yAxisMin': 8,
        'yAxisMax': 12,
        'normalThreshold': 1.0,
        'cautionThreshold': 1.5,
        'baseValue': 9.8  # Z轴重力基准
    }
]

# 缓存当前传感器配置
_cached_sensor_config = None

def get_sensor_config():
    """获取传感器配置"""
    global _cached_sensor_config
    
    if _cached_sensor_config is not None:
        return _cached_sensor_config
    
    if SENSOR_CONFIG_FILE.exists():
        try:
            with open(SENSOR_CONFIG_FILE, 'r', encoding='utf-8') as f:
                _cached_sensor_config = json.load(f)
                return _cached_sensor_config
        except Exception as e:
            print(f"加载传感器配置失败: {e}")
    
    # 返回默认配置
    _cached_sensor_config = DEFAULT_SENSOR_CONFIG.copy()
    return _cached_sensor_config

def save_sensor_config(config):
    """保存传感器配置"""
    global _cached_sensor_config
    
    try:
        # 确保目录存在
        SENSOR_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(SENSOR_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        _cached_sensor_config = config
        return True
    except Exception as e:
        print(f"保存传感器配置失败: {e}")
        return False

def get_available_columns():
    """获取监控数据文件中可用的列"""
    if MONITOR_DATA_FILE.exists():
        try:
            with open(MONITOR_DATA_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                # 过滤掉timestamp列，返回数据列
                return [col for col in headers if col != 'timestamp']
        except Exception as e:
            print(f"读取数据列失败: {e}")
    return ['col_0', 'col_1', 'col_2', 'col_3']

# 默认数据（仅当监控文件不存在时使用）
default_device_data = {
    'adxl345_x': {'value': 0.0, 'unit': 'm/s²', 'status': 'normal', 'description': '轴承径向（水平）振动'},
    'adxl345_y': {'value': 0.0, 'unit': 'm/s²', 'status': 'normal', 'description': '轴承轴向（垂直）振动'},
    'adxl345_z': {'value': 9.8, 'unit': 'm/s²', 'status': 'normal', 'description': '轴承轴向位移或垂直方向振动'},
    'timestamp': datetime.now().isoformat()
}

# 数据采集相关变量
csv_writer = None
csv_file = None
collection_filepath = None  # 当前采集文件的路径
collection_start_time = None  # 采集开始时间
is_collecting = False
collection_thread = None
last_collected_line = 0  # 记录已采集的最后一行

# 导入实时推理服务
from .realtime_inference import (
    get_available_models,
    start_inference,
    stop_inference,
    get_inference_status,
    get_status_info
)

def collect_from_monitor_file(monitor_file_path, output_filepath, start_line_count):
    """从监控文件读取数据并写入采集文件"""
    global is_collecting, csv_writer, csv_file, last_collected_line
    
    try:
        last_collected_line = start_line_count
        
        while is_collecting:
            try:
                # 读取监控文件的新数据
                with open(monitor_file_path, 'r', encoding='utf-8') as f:
                    # 使用CSV reader正确解析（处理引号等）
                    reader = csv.reader(f)
                    all_rows = list(reader)
                    
                    # 如果有新数据（行数增加）
                    if len(all_rows) > last_collected_line + 1:  # +1 因为第一行是表头
                        # 读取新行
                        new_rows = all_rows[last_collected_line + 1:]  # +1 跳过表头
                        
                        # 写入采集文件
                        for row in new_rows:
                            if len(row) == 5:  # 确保是有效的数据行
                                csv_writer.writerow(row)
                                csv_file.flush()  # 立即刷新到磁盘
                        
                        last_collected_line = len(all_rows) - 1  # 更新已采集的行数
                
                # 每秒检查一次
                time.sleep(1.0)
                
            except Exception as e:
                print(f"采集数据时出错: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)
                
    except Exception as e:
        print(f"采集线程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保文件被关闭
        if csv_file:
            csv_file.flush()

@device_monitoring_bp.route('/')
def index():
    """设备监测主页"""
    return render_template('device_monitoring/index.html')

@device_monitoring_bp.route('/api/data')
def get_data():
    """获取设备数据API - 从监控文件读取最新数据（根据传感器配置）"""
    try:
        if not MONITOR_DATA_FILE.exists():
            print(f"[API] 监控文件不存在: {MONITOR_DATA_FILE}")
            return jsonify({'sensors': [], 'timestamp': datetime.now().isoformat()})
        
        # 获取传感器配置
        sensor_config = get_sensor_config()
        
        try:
            # 读取监控文件
            with open(MONITOR_DATA_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if len(rows) < 2:
                    print(f"[API] 监控文件数据不足，行数: {len(rows)}")
                    return jsonify({'sensors': [], 'timestamp': datetime.now().isoformat()})
                
                # 获取表头和最后一行数据
                headers = rows[0]
                last_row = rows[-1]
                
                # 创建列名到索引的映射
                col_index = {col: idx for idx, col in enumerate(headers)}
                
                # 获取时间戳
                timestamp = last_row[0] if len(last_row) > 0 else datetime.now().isoformat()
                
                # 根据配置读取传感器数据
                sensors = []
                for sensor in sensor_config:
                    column = sensor.get('column', '')
                    if column not in col_index:
                        print(f"[API] 传感器 {sensor.get('name')} 的数据列 {column} 不存在")
                        continue
                    
                    try:
                        value = float(last_row[col_index[column]])
                        
                        # 计算状态
                        base_value = sensor.get('baseValue', 0)
                        normal_threshold = sensor.get('normalThreshold', 1.0)
                        caution_threshold = sensor.get('cautionThreshold', 2.0)
                        
                        diff = abs(value - base_value)
                        if diff <= normal_threshold:
                            status = 'normal'
                        elif diff <= caution_threshold:
                            status = 'caution'
                        else:
                            status = 'warning'
                        
                        sensors.append({
                            'id': sensor.get('id'),
                            'name': sensor.get('name'),
                            'description': sensor.get('description', ''),
                            'value': value,
                            'unit': sensor.get('unit', ''),
                            'status': status,
                            'color': sensor.get('color', '#0d6efd'),
                            'yAxisMin': sensor.get('yAxisMin', -10),
                            'yAxisMax': sensor.get('yAxisMax', 10)
                        })
                    except (ValueError, IndexError) as e:
                        print(f"[API] 解析传感器 {sensor.get('name')} 数据失败: {e}")
                
                return jsonify({
                    'sensors': sensors,
                    'timestamp': timestamp
                })
                
        except Exception as e:
            print(f"[API] 读取监控文件失败: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'sensors': [], 'timestamp': datetime.now().isoformat()})
            
    except Exception as e:
        print(f"[API] 从监控文件读取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'sensors': [], 'timestamp': datetime.now().isoformat()})

@device_monitoring_bp.route('/api/start_monitoring')
def start_monitoring():
    """启动数据监测（已废弃，数据由后台生成器自动生成）"""
    # 数据现在由 monitor_data_generator.py 后台进程自动生成
    # 此接口保留用于兼容性，但不再执行任何操作
    monitor_file_path = Path(__file__).resolve().parents[3] / 'data' / 'monitor' / 'monitor_data.csv'
    if monitor_file_path.exists():
        return jsonify({
            'status': 'already_running',
            'message': '数据监测已自动运行（由后台数据生成器提供）'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': '监控数据文件不存在，请确保数据生成器正在运行'
        })

@device_monitoring_bp.route('/api/stop_monitoring')
def stop_monitoring():
    """停止数据监测（已废弃，数据由后台生成器自动生成）"""
    # 数据现在由 monitor_data_generator.py 后台进程自动生成
    # 此接口保留用于兼容性，但不再执行任何操作
    return jsonify({
        'status': 'info',
        'message': '数据监测由后台生成器自动运行，无法通过此接口停止'
    })

@device_monitoring_bp.route('/api/start_collection')
def start_collection():
    """启动数据采集 - 从监控文件读取数据"""
    global csv_writer, csv_file, collection_filepath, collection_start_time, is_collecting, collection_thread

    if is_collecting:
        return jsonify({'status': 'already_collecting', 'message': '数据采集已在进行中'})

    try:
        # 检查监控文件是否存在
        monitor_file_path = Path(__file__).resolve().parents[3] / 'data' / 'monitor' / 'monitor_data.csv'
        if not monitor_file_path.exists():
            return jsonify({'status': 'error', 'message': '监控数据文件不存在，请确保数据生成器正在运行'})

        # 记录开始时间
        collection_start_time = datetime.now()
        start_timestamp = collection_start_time.strftime('%Y%m%d_%H%M%S')
        
        # 创建collected目录（如果不存在）
        collected_dir = Path(__file__).resolve().parents[3] / 'data' / 'collected'
        collected_dir.mkdir(parents=True, exist_ok=True)

        # 创建临时CSV文件（停止时会重命名为最终文件名）
        # 临时文件名：年月日_开始时间_temp.csv
        temp_filename = f'{start_timestamp}_temp.csv'
        collection_filepath = collected_dir / temp_filename

        csv_file = open(collection_filepath, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)

        # 写入表头
        csv_writer.writerow(['timestamp', 'col_0', 'col_1', 'col_2', 'col_3'])

        # 记录开始采集时的文件行数（用于增量读取）
        with open(monitor_file_path, 'r', encoding='utf-8') as f:
            initial_line_count = sum(1 for _ in f) - 1  # 减去表头

        is_collecting = True
        
        # 启动采集线程，从监控文件读取数据
        collection_thread = threading.Thread(
            target=collect_from_monitor_file,
            args=(monitor_file_path, str(collection_filepath), initial_line_count),
            daemon=True
        )
        collection_thread.start()
        
        print(f"开始数据采集: {collection_filepath} (从监控文件: {monitor_file_path})")
        return jsonify({'status': 'started', 'message': '数据采集已启动', 'temp_file': temp_filename})

    except Exception as e:
        print(f"创建数据采集文件失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'创建数据采集文件失败: {e}'})

@device_monitoring_bp.route('/api/stop_collection')
def stop_collection():
    """停止数据采集"""
    global csv_writer, csv_file, is_collecting, collection_filepath, collection_start_time, collection_thread

    if not is_collecting:
        return jsonify({'status': 'not_collecting', 'message': '数据采集未启动'})

    try:
        # 停止采集标志
        is_collecting = False
        
        # 等待采集线程结束（最多等待2秒）
        if collection_thread and collection_thread.is_alive():
            collection_thread.join(timeout=2.0)
        
        # 关闭文件
        if csv_file:
            csv_file.flush()
            csv_file.close()
            csv_file = None
        csv_writer = None
        
        # 记录结束时间并重命名文件
        end_time = datetime.now()
        start_timestamp = collection_start_time.strftime('%Y%m%d_%H%M%S')
        end_timestamp = end_time.strftime('%Y%m%d_%H%M%S')
        
        # 生成最终文件名：年月日_开始时间-年月日_结束时间.csv
        final_filename = f'{start_timestamp}-{end_timestamp}.csv'
        final_filepath = collection_filepath.parent / final_filename
        
        # 重命名文件
        if collection_filepath.exists():
            collection_filepath.rename(final_filepath)
            print(f"数据采集已停止，文件已保存: {final_filepath}")
            return jsonify({
                'status': 'stopped',
                'message': '数据采集已停止',
                'filename': final_filename,
                'file_path': str(final_filepath)
            })
        else:
            print(f"警告：临时文件不存在: {collection_filepath}")
            return jsonify({
                'status': 'stopped',
                'message': '数据采集已停止，但文件可能未正确保存',
                'filename': None
            })
    except Exception as e:
        print(f"停止数据采集时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'停止数据采集时出错: {e}'})
    finally:
        # 清理全局变量
        collection_filepath = None
        collection_start_time = None

@device_monitoring_bp.route('/api/status')
def get_status():
    """获取监测状态"""
    # 计算监控文件路径：从 edge/src/tasks/device_monitoring/__init__.py 到 edge/data/monitor/monitor_data.csv
    # parents[3] = edge/ 目录
    monitor_file_path = Path(__file__).resolve().parents[3] / 'data' / 'monitor' / 'monitor_data.csv'
    monitoring = monitor_file_path.exists()
    
    # 调试输出
    if not monitoring:
        print(f"[状态检查] 监控文件不存在: {monitor_file_path}")
        print(f"[状态检查] 当前文件: {__file__}")
        print(f"[状态检查] parents[3]: {Path(__file__).resolve().parents[3]}")
    else:
        # 检查文件是否有数据（至少2行：表头+数据）
        try:
            with open(monitor_file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
                if line_count < 2:
                    monitoring = False
                    print(f"[状态检查] 监控文件存在但数据不足（只有 {line_count} 行）")
        except Exception as e:
            print(f"[状态检查] 检查监控文件时出错: {e}")
            monitoring = False
    
    # 获取实时推理状态
    inference_status = get_status_info()
    
    return jsonify({
        'monitoring': monitoring,  # 监控文件存在且有数据即表示监测中
        'collecting': is_collecting,
        'collection_file': str(collection_filepath) if collection_filepath else None,
        'realtime_inference': inference_status['realtime_inference'],
        'realtime_model': inference_status['realtime_model']
    })

# ==================== 实时推理功能 ====================

@device_monitoring_bp.route('/api/realtime_inference/models', methods=['GET'])
def get_realtime_inference_models():
    """获取可用于实时推理的模型列表"""
    result = get_available_models()
    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 500


@device_monitoring_bp.route('/api/realtime_inference/start', methods=['POST'])
def start_realtime_inference():
    """启动实时推理"""
    try:
        data = request.get_json()
        task_id = data.get('task_id', '').strip()
        
        if not task_id:
            return jsonify({
                'success': False,
                'error': '请选择模型'
            }), 400
        
        result = start_inference(task_id)
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        print(f"[实时推理] 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'启动实时推理失败: {str(e)}'
        }), 500


@device_monitoring_bp.route('/api/realtime_inference/stop', methods=['POST'])
def stop_realtime_inference():
    """停止实时推理"""
    result = stop_inference()
    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 400


@device_monitoring_bp.route('/api/realtime_inference/status', methods=['GET'])
def get_realtime_inference_status():
    """获取实时推理状态和最新结果"""
    status = get_inference_status()
    return jsonify({
        'success': True,
        **status
    })

# ==================== 传感器配置功能 ====================

@device_monitoring_bp.route('/api/sensor_config', methods=['GET'])
def api_get_sensor_config():
    """获取传感器配置"""
    try:
        config = get_sensor_config()
        columns = get_available_columns()
        return jsonify({
            'success': True,
            'config': config,
            'available_columns': columns
        })
    except Exception as e:
        print(f"获取传感器配置失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@device_monitoring_bp.route('/api/sensor_config', methods=['POST'])
def api_save_sensor_config():
    """保存传感器配置"""
    try:
        data = request.get_json()
        config = data.get('config', [])
        
        # 验证配置格式
        if not isinstance(config, list):
            return jsonify({
                'success': False,
                'error': '配置格式错误，应为数组'
            }), 400
        
        # 验证每个传感器配置
        for i, sensor in enumerate(config):
            if not sensor.get('name'):
                return jsonify({
                    'success': False,
                    'error': f'第 {i+1} 个传感器缺少名称'
                }), 400
            if not sensor.get('column'):
                return jsonify({
                    'success': False,
                    'error': f'第 {i+1} 个传感器缺少数据列'
                }), 400
            
            # 为每个传感器生成唯一ID
            if not sensor.get('id'):
                sensor['id'] = f'sensor_{i}'
            
            # 设置默认值
            sensor.setdefault('description', '')
            sensor.setdefault('unit', '')
            sensor.setdefault('color', '#0d6efd')
            sensor.setdefault('yAxisMin', -10)
            sensor.setdefault('yAxisMax', 10)
            sensor.setdefault('normalThreshold', 1.0)
            sensor.setdefault('cautionThreshold', 2.0)
        
        # 保存配置
        if save_sensor_config(config):
            return jsonify({
                'success': True,
                'message': '传感器配置已保存',
                'config': config
            })
        else:
            return jsonify({
                'success': False,
                'error': '保存配置失败'
            }), 500
            
    except Exception as e:
        print(f"保存传感器配置失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@device_monitoring_bp.route('/api/sensor_config/reset', methods=['POST'])
def api_reset_sensor_config():
    """重置传感器配置为默认值"""
    global _cached_sensor_config
    try:
        _cached_sensor_config = None
        if SENSOR_CONFIG_FILE.exists():
            SENSOR_CONFIG_FILE.unlink()
        return jsonify({
            'success': True,
            'message': '配置已重置为默认值',
            'config': DEFAULT_SENSOR_CONFIG
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500