"""
数据管理工具函数
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
import re
from .config import META_DIR, TASK_TYPES
from ...common.utils.path_utils import to_relative_path


def parse_datetime(time_str):
    """解析多种格式的时间字符串"""
    formats = [
        '%Y-%m-%dT%H:%M:%S.%f',    # ISO格式带微秒
        '%Y-%m-%dT%H:%M:%S',       # ISO格式不带微秒
        '%Y-%m-%d %H:%M:%S.%f',    # 标准格式带微秒
        '%Y-%m-%d %H:%M:%S',       # 标准格式不带微秒
        '%Y-%m-%d %H:%M',          # 不带秒
        '%Y/%m/%d %H:%M:%S',       # 斜杠分隔
        '%Y%m%d_%H%M%S',           # 紧凑格式
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f'无法解析时间格式: {time_str}')


def format_file_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return f'{size_bytes} B'
    elif size_bytes < 1024 * 1024:
        return f'{size_bytes / 1024:.2f} KB'
    elif size_bytes < 1024 * 1024 * 1024:
        return f'{size_bytes / (1024 * 1024):.2f} MB'
    else:
        return f'{size_bytes / (1024 * 1024 * 1024):.2f} GB'


def get_dir_display_name(key):
    """获取目录的显示名称"""
    names = {
        'processed': '预处理数据',
        'labeled': '标注数据',
        'training': '训练数据',
    }
    return names.get(key, key)


def get_parent_subdir(subdir):
    """获取父级子目录名称"""
    parts = subdir.split('/')
    if len(parts) > 1:
        return '/'.join(parts[:-1])
    return ''


def get_meta_file_path(task_type: str, filename: str) -> Path:
    """获取元文件路径"""
    if task_type not in TASK_TYPES:
        return None
    
    task_config = TASK_TYPES[task_type]
    # 元文件存放在 edge/data/meta/{task_subdir} 目录
    meta_subdir = task_config.get('labeled_subdir') or task_config.get('processed_subdir')
    if not meta_subdir:
        return None
    
    meta_dir = META_DIR / meta_subdir
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # 元文件名与数据文件名相同，扩展名为.json
    meta_filename = Path(filename).stem + '.json'
    return meta_dir / meta_filename


def extract_time_from_filename(filename: str) -> tuple:
    """从文件名中提取时间信息
    支持格式：
    - 20251128_131550-20251128_131607.csv
    - 20251128_131550_20251128_131607.csv
    - 2025-11-28_13:15:50-2025-11-28_13:16:07.csv
    """
    # 移除扩展名
    name = Path(filename).stem
    
    # 尝试匹配格式：YYYYMMDD_HHMMSS-YYYYMMDD_HHMMSS
    pattern1 = r'(\d{8}_\d{6})-(\d{8}_\d{6})'
    match1 = re.search(pattern1, name)
    if match1:
        start_str = match1.group(1)
        end_str = match1.group(2)
        try:
            start_time = datetime.strptime(start_str, '%Y%m%d_%H%M%S').isoformat()
            end_time = datetime.strptime(end_str, '%Y%m%d_%H%M%S').isoformat()
            return start_time, end_time
        except ValueError:
            pass
    
    # 尝试匹配格式：YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS
    pattern2 = r'(\d{8}_\d{6})_(\d{8}_\d{6})'
    match2 = re.search(pattern2, name)
    if match2:
        start_str = match2.group(1)
        end_str = match2.group(2)
        try:
            start_time = datetime.strptime(start_str, '%Y%m%d_%H%M%S').isoformat()
            end_time = datetime.strptime(end_str, '%Y%m%d_%H%M%S').isoformat()
            return start_time, end_time
        except ValueError:
            pass
    
    return None, None


def create_meta_file(task_type: str, filename: str, file_path: str, meta_data: dict) -> dict:
    """创建或更新元文件"""
    try:
        meta_file_path = get_meta_file_path(task_type, filename)
        if not meta_file_path:
            return {'success': False, 'error': '无法确定元文件路径'}
        
        # 如果元文件已存在，读取现有数据
        existing_meta = {}
        if meta_file_path.exists():
            try:
                with open(meta_file_path, 'r', encoding='utf-8') as f:
                    existing_meta = json.load(f)
            except:
                pass
        
        # 从文件名中提取时间信息
        start_time, end_time = extract_time_from_filename(filename)
        now = datetime.now().isoformat()
        
        # 如果元文件已存在，保留原有的created_at；否则使用文件名中的开始时间或当前时间
        if existing_meta.get('created_at'):
            created_at = existing_meta['created_at']
        elif start_time:
            created_at = start_time
        else:
            created_at = now
        
        # display_name 默认值为文件名（不含扩展名）
        filename_without_ext = Path(filename).stem
        display_name = meta_data.get('display_name', filename_without_ext)
        if not display_name:
            display_name = filename_without_ext
        
        relative_file_path = to_relative_path(file_path)

        meta_content = {
            'file_id': existing_meta.get('file_id', f"auto_{uuid.uuid4().hex[:12]}"),
            'file_path': relative_file_path,
            'display_name': display_name,
            'created_at': created_at,
            'updated_at': now,
            'tags_label': meta_data.get('tags_label', []),
            'tags_condition': meta_data.get('tags_condition', []),
            'comment': meta_data.get('comment', '')
        }
        
        # 如果从文件名中提取到了时间，添加到元数据中（可选）
        if start_time and end_time:
            meta_content['data_start_time'] = start_time
            meta_content['data_end_time'] = end_time
        
        # RUL预测：添加rul_config
        if task_type == 'rul_prediction':
            rul_config = meta_data.get('rul_config', {})
            if rul_config:
                meta_content['rul_config'] = {
                    'failure_row_index': rul_config.get('failure_row_index'),
                    'rul_unit': rul_config.get('rul_unit', 'cycle'),
                    'max_rul': rul_config.get('max_rul', 200)
                }
        else:
            # 非RUL预测：确保标签至少有一行
            if not meta_content['tags_label']:
                meta_content['tags_label'] = [{'key': '健康状态', 'value': '正常'}]
        
        # 保存元文件
        with open(meta_file_path, 'w', encoding='utf-8') as f:
            json.dump(meta_content, f, indent=2, ensure_ascii=False)
        
        return {
            'success': True,
            'meta_file': to_relative_path(meta_file_path),
            'meta_data': meta_content
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

