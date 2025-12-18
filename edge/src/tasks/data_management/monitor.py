"""
监控数据相关API
"""
import csv
from datetime import datetime
from flask import Blueprint, jsonify, request
from .config import MONITOR_DATA_FILE, COLLECTED_DIR
from .utils import parse_datetime, format_file_size


def register_monitor_routes(bp: Blueprint):
    """注册监控数据相关路由"""
    
    @bp.route('/api/monitor_data_info')
    def get_monitor_data_info():
        """获取监控数据文件的基本信息（开始时间、结束时间、数据行数）"""
        try:
            if not MONITOR_DATA_FILE.exists():
                return jsonify({
                    'success': False,
                    'error': '监控数据文件不存在'
                }), 404
            
            with open(MONITOR_DATA_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if len(rows) < 2:
                return jsonify({
                    'success': False,
                    'error': '监控数据文件为空或只有表头'
                }), 400
            
            # 获取表头
            headers = rows[0]
            
            # 获取第一行和最后一行数据
            first_row = rows[1]
            last_row = rows[-1]
            
            # 时间戳在第一列
            start_time = first_row[0] if len(first_row) > 0 else None
            end_time = last_row[0] if len(last_row) > 0 else None
            
            # 数据行数（不包括表头）
            data_count = len(rows) - 1
            
            # 文件大小
            file_size = MONITOR_DATA_FILE.stat().st_size
            file_size_str = format_file_size(file_size)
            
            return jsonify({
                'success': True,
                'info': {
                    'headers': headers,
                    'start_time': start_time,
                    'end_time': end_time,
                    'data_count': data_count,
                    'file_size': file_size_str,
                    'file_path': str(MONITOR_DATA_FILE)
                }
            })
            
        except Exception as e:
            print(f"获取监控数据信息失败: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @bp.route('/api/extract_data', methods=['POST'])
    def extract_data():
        """截取指定时间范围的数据"""
        try:
            data = request.get_json()
            start_time = data.get('start_time', '').strip()
            end_time = data.get('end_time', '').strip()
            
            if not start_time or not end_time:
                return jsonify({
                    'success': False,
                    'error': '请指定开始时间和结束时间'
                }), 400
            
            if not MONITOR_DATA_FILE.exists():
                return jsonify({
                    'success': False,
                    'error': '监控数据文件不存在'
                }), 404
            
            # 解析时间
            try:
                # 支持多种时间格式
                start_dt = parse_datetime(start_time)
                end_dt = parse_datetime(end_time)
            except ValueError as e:
                return jsonify({
                    'success': False,
                    'error': f'时间格式错误: {e}'
                }), 400
            
            if start_dt >= end_dt:
                return jsonify({
                    'success': False,
                    'error': '开始时间必须早于结束时间'
                }), 400
            
            # 读取数据并筛选
            with open(MONITOR_DATA_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if len(rows) < 2:
                return jsonify({
                    'success': False,
                    'error': '监控数据文件为空'
                }), 400
            
            headers = rows[0]
            extracted_rows = []
            
            for row in rows[1:]:
                if len(row) < 1:
                    continue
                
                try:
                    row_time = parse_datetime(row[0])
                    if start_dt <= row_time <= end_dt:
                        extracted_rows.append(row)
                except ValueError:
                    continue
            
            if len(extracted_rows) == 0:
                return jsonify({
                    'success': False,
                    'error': '指定时间范围内没有数据'
                }), 400
            
            # 创建输出目录
            COLLECTED_DIR.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名（使用简化的时间格式）
            start_str = start_dt.strftime('%Y%m%d_%H%M%S')
            end_str = end_dt.strftime('%Y%m%d_%H%M%S')
            filename = f'{start_str}-{end_str}.csv'
            output_path = COLLECTED_DIR / filename
            
            # 写入文件
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(extracted_rows)
            
            # 计算文件大小
            file_size = output_path.stat().st_size
            file_size_str = format_file_size(file_size)
            
            return jsonify({
                'success': True,
                'message': f'成功截取 {len(extracted_rows)} 条数据',
                'result': {
                    'filename': filename,
                    'data_count': len(extracted_rows),
                    'file_size': file_size_str,
                    'save_path': str(output_path),
                    'start_time': start_time,
                    'end_time': end_time
                }
            })
            
        except Exception as e:
            print(f"截取数据失败: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @bp.route('/api/collected_files')
    def get_collected_files():
        """获取已采集/截取的数据文件列表"""
        try:
            if not COLLECTED_DIR.exists():
                return jsonify({
                    'success': True,
                    'files': []
                })
            
            files = []
            # 只扫描直接放在collected目录下的CSV文件（不再支持子文件夹）
            for item in sorted(COLLECTED_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if item.is_file() and item.suffix == '.csv':
                    stat = item.stat()
                    files.append({
                        'name': item.name,
                        'size': format_file_size(stat.st_size),
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'path': str(item)
                    })
            
            return jsonify({
                'success': True,
                'files': files
            })
            
        except Exception as e:
            print(f"获取文件列表失败: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @bp.route('/api/delete_file', methods=['DELETE'])
    def delete_file():
        """删除指定的数据文件"""
        try:
            data = request.get_json()
            filename = data.get('filename', '').strip()
            
            if not filename:
                return jsonify({
                    'success': False,
                    'error': '请指定要删除的文件名'
                }), 400
            
            # 安全检查：只允许删除 collected 目录下的文件
            file_path = COLLECTED_DIR / filename
            
            # 检查路径是否在 collected 目录内
            try:
                file_path.resolve().relative_to(COLLECTED_DIR.resolve())
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': '非法的文件路径'
                }), 400
            
            if not file_path.exists():
                return jsonify({
                    'success': False,
                    'error': '文件不存在'
                }), 404
            
            file_path.unlink()
            
            return jsonify({
                'success': True,
                'message': f'已删除文件: {filename}'
            })
            
        except Exception as e:
            print(f"删除文件失败: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

