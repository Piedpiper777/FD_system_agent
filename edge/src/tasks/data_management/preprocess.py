"""
数据预处理相关API
"""
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Blueprint, jsonify, request
from .config import TASK_TYPES, COLLECTED_DIR, PROCESSED_DIR
from .utils import format_file_size


def register_preprocess_routes(bp: Blueprint):
    """注册数据预处理相关路由"""
    
    @bp.route('/api/preprocess/task_types')
    def get_task_types():
        """获取支持的任务类型"""
        return jsonify({
            'success': True,
            'task_types': [
                {'key': k, 'name': v['name'], 'has_label': v['has_label']}
                for k, v in TASK_TYPES.items()
            ]
        })

    @bp.route('/api/preprocess/source_files')
    def get_source_files():
        """获取可用于预处理的源文件（从collected目录）"""
        try:
            if not COLLECTED_DIR.exists():
                return jsonify({'success': True, 'files': []})
            
            files = []
            # 遍历 collected 目录下的所有 CSV 文件
            for item in COLLECTED_DIR.rglob('*.csv'):
                if item.is_file():
                    stat = item.stat()
                    # 计算相对路径
                    rel_path = item.relative_to(COLLECTED_DIR)
                    files.append({
                        'name': item.name,
                        'path': str(rel_path),
                        'full_path': str(item),
                        'size': format_file_size(stat.st_size),
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            # 按修改时间降序排序
            files.sort(key=lambda x: x['modified'], reverse=True)
            
            return jsonify({'success': True, 'files': files})
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/api/preprocess/run', methods=['POST'])
    def run_preprocess():
        """执行数据预处理"""
        try:
            data = request.get_json()
            
            task_type = data.get('task_type', '').strip()
            source_file = data.get('source_file', '').strip()
            method = data.get('method', 'minmax').strip()
            
            if task_type not in TASK_TYPES:
                return jsonify({'success': False, 'error': '无效的任务类型'}), 400
            
            if not source_file:
                return jsonify({'success': False, 'error': '请选择源文件'}), 400
            
            # 构建路径
            source_path = COLLECTED_DIR / source_file
            
            # 安全检查
            try:
                source_path.resolve().relative_to(COLLECTED_DIR.resolve())
            except ValueError:
                return jsonify({'success': False, 'error': '非法的文件路径'}), 400
            
            if not source_path.exists():
                return jsonify({'success': False, 'error': '源文件不存在'}), 404
            
            # 创建输出目录
            task_config = TASK_TYPES[task_type]
            output_dir = PROCESSED_DIR / task_config['processed_subdir']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 读取数据
            df = pd.read_csv(source_path)
            
            # 自动处理所有数值列（排除时间戳列）
            columns = [col for col in df.columns 
                      if pd.api.types.is_numeric_dtype(df[col]) 
                      and col.lower() not in ['timestamp', 'time', '时间']]
            
            if not columns:
                return jsonify({'success': False, 'error': '没有可处理的数值列'}), 400
            
            # 执行预处理
            original_stats = {}
            processed_stats = {}
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                col_data = df[col].values.astype(float)
                
                # 处理NaN值：先填充NaN，然后计算统计信息
                col_data_clean = col_data.copy()
                has_nan = np.isnan(col_data_clean).any()
                
                if has_nan:
                    # 使用前向填充和后向填充来处理NaN
                    col_series = pd.Series(col_data_clean)
                    col_data_clean = col_series.ffill().bfill().fillna(0).values
                
                # 计算原始统计信息（处理NaN）
                def safe_float(value):
                    """安全地将值转换为float，NaN转换为None"""
                    if np.isnan(value) or np.isinf(value):
                        return None
                    return float(value)
                
                original_stats[col] = {
                    'mean': safe_float(np.nanmean(col_data)),
                    'std': safe_float(np.nanstd(col_data)),
                    'min': safe_float(np.nanmin(col_data)),
                    'max': safe_float(np.nanmax(col_data))
                }
                
                # 根据方法进行标准化（使用清理后的数据）
                if method == 'zscore':
                    mean = np.nanmean(col_data_clean)
                    std = np.nanstd(col_data_clean)
                    if std > 0 and not np.isnan(std):
                        df[col] = (col_data - mean) / std
                    else:
                        df[col] = col_data - mean
                    # 处理结果中的NaN
                    df[col] = df[col].fillna(0)
                elif method == 'minmax':
                    min_val = np.nanmin(col_data_clean)
                    max_val = np.nanmax(col_data_clean)
                    if max_val > min_val and not (np.isnan(min_val) or np.isnan(max_val)):
                        df[col] = (col_data - min_val) / (max_val - min_val)
                    else:
                        df[col] = 0.0
                    # 处理结果中的NaN
                    df[col] = df[col].fillna(0)
                elif method == 'robust':
                    median = np.nanmedian(col_data_clean)
                    q1 = np.nanpercentile(col_data_clean, 25)
                    q3 = np.nanpercentile(col_data_clean, 75)
                    iqr = q3 - q1
                    if iqr > 0 and not (np.isnan(median) or np.isnan(iqr)):
                        df[col] = (col_data - median) / iqr
                    else:
                        df[col] = col_data - median
                    # 处理结果中的NaN
                    df[col] = df[col].fillna(0)
                
                # 计算处理后统计信息（处理NaN）
                processed_stats[col] = {
                    'mean': safe_float(np.nanmean(df[col].values)),
                    'std': safe_float(np.nanstd(df[col].values)),
                    'min': safe_float(np.nanmin(df[col].values)),
                    'max': safe_float(np.nanmax(df[col].values))
                }
            
            # 生成输出文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{source_path.stem}_preprocessed_{method}_{timestamp}.csv"
            output_path = output_dir / output_filename
            
            # 保存处理后的数据
            df.to_csv(output_path, index=False)
            
            return jsonify({
                'success': True,
                'message': '数据预处理完成',
                'result': {
                    'output_file': output_filename,
                    'output_path': str(output_path),
                    'method': method,
                    'processed_columns': columns,
                    'total_rows': len(df),
                    'original_stats': original_stats,
                    'processed_stats': processed_stats
                }
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/api/preprocess/processed_files')
    def get_processed_files():
        """获取已预处理的文件列表"""
        try:
            print(f"[数据管理] 收到预处理文件列表请求 - 这是 Edge 端的请求")
            task_type = request.args.get('task_type', '').strip()
            print(f"[数据管理] task_type={task_type}, PROCESSED_DIR={PROCESSED_DIR}")
            
            if task_type and task_type in TASK_TYPES:
                search_dir = PROCESSED_DIR / TASK_TYPES[task_type]['processed_subdir']
            else:
                search_dir = PROCESSED_DIR
            
            print(f"[数据管理] 搜索目录: {search_dir}, 存在: {search_dir.exists()}")
            
            if not search_dir.exists():
                print(f"[数据管理] 目录不存在，返回空列表")
                return jsonify({'success': True, 'files': []})
            
            files = []
            # 如果search_dir是子目录（如AnomalyDetection），直接在该目录下查找
            # 如果search_dir是PROCESSED_DIR，需要递归查找所有子目录
            if search_dir == PROCESSED_DIR:
                # 递归查找所有CSV文件
                for item in search_dir.rglob('*.csv'):
                    if item.is_file():
                        stat = item.stat()
                        rel_path = item.relative_to(PROCESSED_DIR)
                        files.append({
                            'name': item.name,
                            'path': str(rel_path),
                            'full_path': str(item),
                            'size': format_file_size(stat.st_size),
                            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        })
            else:
                # 只在指定子目录下查找（不递归）
                for item in search_dir.glob('*.csv'):
                    if item.is_file():
                        stat = item.stat()
                        rel_path = item.relative_to(PROCESSED_DIR)
                        files.append({
                            'name': item.name,
                            'path': str(rel_path),
                            'full_path': str(item),
                            'size': format_file_size(stat.st_size),
                            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        })
            
            files.sort(key=lambda x: x['modified'], reverse=True)
            
            return jsonify({'success': True, 'files': files})
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

