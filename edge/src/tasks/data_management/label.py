"""
数据标注相关API
"""
import json
import shutil
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from flask import Blueprint, jsonify, request
from .config import TASK_TYPES, PROCESSED_DIR, LABELED_DIR
from .utils import format_file_size, get_meta_file_path, extract_time_from_filename, create_meta_file
from ...common.utils.path_utils import to_relative_path


def register_label_routes(bp: Blueprint):
    """注册数据标注相关路由"""
    
    @bp.route('/api/label/run', methods=['POST'])
    def run_label():
        """执行数据标注（支持异常检测的元文件）"""
        try:
            data = request.get_json()
            
            task_type = data.get('task_type', '').strip()
            source_file = data.get('source_file', '').strip()
            
            if task_type not in TASK_TYPES:
                return jsonify({'success': False, 'error': '无效的任务类型'}), 400
            
            task_config = TASK_TYPES[task_type]
            if not task_config['has_label']:
                return jsonify({'success': False, 'error': '该任务类型不需要标注'}), 400
            
            if not source_file:
                return jsonify({'success': False, 'error': '请选择源文件'}), 400
            
            # 源文件来自 processed 目录
            source_dir = PROCESSED_DIR / task_config['processed_subdir']
            source_path = source_dir / source_file
            
            # 安全检查
            try:
                source_path.resolve().relative_to(source_dir.resolve())
            except ValueError:
                return jsonify({'success': False, 'error': '非法的文件路径'}), 400
            
            if not source_path.exists():
                return jsonify({'success': False, 'error': '源文件不存在'}), 404
            
            # 创建输出目录
            output_dir = LABELED_DIR / task_config['labeled_subdir']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 异常检测、故障诊断和RUL预测：使用元文件，根据display_name或标签工况重命名
            if task_type in ['anomaly_detection', 'fault_diagnosis', 'rul_prediction']:
                # 获取元数据
                meta_data = data.get('meta_data', {})
                display_name = meta_data.get('display_name', '').strip()
                original_filename = source_path.name
                original_filename_without_ext = source_path.stem
                
                # 判断用户是否修改了display_name
                # 如果display_name等于原文件名（不含扩展名），则认为用户没有修改
                is_display_name_modified = display_name and display_name != original_filename_without_ext
                
                if is_display_name_modified:
                    # 用户修改了display_name，使用display_name作为文件名
                    output_filename = f'{display_name}.csv'
                else:
                    # 用户没有修改display_name，根据标签和工况生成文件名
                    tags_condition = meta_data.get('tags_condition', [])
                    name_parts = []
                    
                    # RUL预测：只根据工况重命名（没有标签）
                    if task_type == 'rul_prediction':
                        # 只添加工况值（RUL预测没有标签）
                        for cond in tags_condition:
                            cond_key = cond.get('key', '').strip()
                            cond_value = cond.get('value', '').strip()
                            if cond_key and cond_value:
                                name_parts.append(f"{cond_key}{cond_value}")
                    else:
                        # 异常检测和故障诊断：根据标签和工况生成文件名
                        tags_label = meta_data.get('tags_label', [])
                        
                        # 添加标签值（取第一个标签的值）
                        if tags_label and len(tags_label) > 0:
                            first_label_value = tags_label[0].get('value', '').strip()
                            if first_label_value:
                                name_parts.append(first_label_value)
                        
                        # 添加工况值
                        for cond in tags_condition:
                            cond_key = cond.get('key', '').strip()
                            cond_value = cond.get('value', '').strip()
                            if cond_key and cond_value:
                                name_parts.append(f"{cond_key}{cond_value}")
                    
                    # 添加原文件名
                    name_parts.append(original_filename_without_ext)
                    
                    # 生成新文件名
                    if name_parts:
                        output_filename = '_'.join(name_parts) + '.csv'
                    else:
                        # 如果没有标签和工况，使用原文件名
                        output_filename = original_filename
                
                output_path = output_dir / output_filename
                
                # 检查目标文件是否已存在（避免覆盖）
                if output_path.exists():
                    # 如果文件已存在，添加时间戳后缀
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_filename_stem = Path(output_filename).stem
                    output_filename_ext = Path(output_filename).suffix
                    output_filename = f'{output_filename_stem}_{timestamp}{output_filename_ext}'
                    output_path = output_dir / output_filename
                
                # 如果文件被重命名了，需要删除旧文件名的元文件（如果存在）
                if output_filename != original_filename:
                    old_meta_path = get_meta_file_path(task_type, original_filename)
                    if old_meta_path and old_meta_path.exists():
                        try:
                            old_meta_path.unlink()
                            print(f"已删除旧元文件: {old_meta_path}")
                        except Exception as e:
                            print(f"删除旧元文件失败: {e}")
                
                # 更新meta_data中的display_name，使其等于新文件名（不含扩展名）
                # 这样元文件中的display_name就会与文件名保持一致
                output_filename_without_ext = Path(output_filename).stem
                meta_data['display_name'] = output_filename_without_ext
                
                # RUL预测：数据在生成时已经截断到故障点，这里直接复制即可
                # 不需要二次截断，避免逻辑混乱
                if task_type == 'rul_prediction':
                    # 直接复制文件（数据在采集/生成阶段已经截断到故障点）
                    shutil.copy2(source_path, output_path)
                    
                    # 验证数据长度与元文件一致性
                    rul_config = meta_data.get('rul_config', {})
                    failure_row_index = rul_config.get('failure_row_index')
                    
                    if failure_row_index is not None:
                        df = pd.read_csv(source_path)
                        actual_length = len(df)
                        expected_length = failure_row_index + 1
                        
                        if actual_length == expected_length:
                            print(f"✅ [RUL预测] 数据验证通过: 文件长度({actual_length})与故障点索引({failure_row_index})一致")
                        else:
                            print(f"⚠️ [RUL预测] 警告: 文件长度({actual_length})与预期长度({expected_length})不一致")
                            # 更新元文件中的failure_row_index为实际最后一行的索引
                            meta_data['rul_config']['failure_row_index'] = actual_length - 1
                            print(f"   已更新failure_row_index为: {actual_length - 1}")
                    else:
                        print(f"⚠️ [RUL预测] 警告：元文件中缺少failure_row_index，将使用数据最后一行作为故障点")
                else:
                    # 其他任务类型：直接复制文件
                    shutil.copy2(source_path, output_path)
                
                # 创建元文件（使用新的文件名）
                meta_result = create_meta_file(
                    task_type=task_type,
                    filename=output_filename,
                    file_path=str(output_path),
                    meta_data=meta_data
                )
                
                if not meta_result['success']:
                    # 如果元文件创建失败，删除已复制的文件
                    if output_path.exists():
                        output_path.unlink()
                    return jsonify({
                        'success': False,
                        'error': f'元文件创建失败: {meta_result.get("error", "未知错误")}'
                    }), 500
                
                return jsonify({
                    'success': True,
                    'message': '数据标注完成',
                    'result': {
                        'output_file': output_filename,
                        'output_path': to_relative_path(output_path),
                        'meta_file': meta_result.get('meta_file'),
                        'meta_data': meta_result.get('meta_data')
                    }
                })
            
            else:
                # 其他任务类型：使用原有的标签和工况逻辑
                label = data.get('label', '').strip()
                conditions = data.get('conditions', {})
                
                if not label:
                    return jsonify({'success': False, 'error': '请输入标签'}), 400
                
                # 构建输出文件名
                # 格式：标签_工况1值_工况2值_..._原文件名
                name_parts = [label]
                if conditions:
                    for cond_name, cond_value in conditions.items():
                        name_parts.append(f"{cond_name}{cond_value}")
                name_parts.append(source_path.name)
                
                output_filename = '_'.join(name_parts)
                output_path = output_dir / output_filename
                
                # 复制文件
                shutil.copy2(source_path, output_path)
                
                return jsonify({
                    'success': True,
                    'message': '数据标注完成',
                    'result': {
                        'output_file': output_filename,
                        'output_path': str(output_path),
                        'label': label,
                        'conditions': conditions
                    }
                })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/api/label/labeled_files')
    def get_labeled_files():
        """获取已标注的文件列表"""
        try:
            task_type = request.args.get('task_type', '').strip()
            
            if task_type and task_type in TASK_TYPES:
                task_config = TASK_TYPES[task_type]
                if not task_config['labeled_subdir']:
                    return jsonify({'success': True, 'files': []})
                search_dir = LABELED_DIR / task_config['labeled_subdir']
            else:
                search_dir = LABELED_DIR
            
            if not search_dir.exists():
                return jsonify({'success': True, 'files': []})
            
            files = []
            for item in search_dir.rglob('*.csv'):
                if item.is_file():
                    stat = item.stat()
                    rel_path = item.relative_to(LABELED_DIR)
                    
                    file_info = {
                        'name': item.name,
                        'path': str(rel_path),
                        'full_path': str(item),
                        'size': format_file_size(stat.st_size),
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # 如果是异常检测或故障诊断，尝试加载元文件信息
                    if task_type in ['anomaly_detection', 'fault_diagnosis']:
                        meta_file = get_meta_file_path(task_type, item.name)
                        if meta_file and meta_file.exists():
                            try:
                                with open(meta_file, 'r', encoding='utf-8') as f:
                                    meta_data = json.load(f)
                                    file_info['has_meta'] = True
                                    file_info['display_name'] = meta_data.get('display_name', item.name)
                            except:
                                file_info['has_meta'] = False
                        else:
                            file_info['has_meta'] = False
                    
                    files.append(file_info)
            
            files.sort(key=lambda x: x['modified'], reverse=True)
            
            return jsonify({'success': True, 'files': files})
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/api/label/meta/<task_type>/<path:filename>')
    def get_meta_file(task_type: str, filename: str):
        """获取元文件内容"""
        try:
            if task_type not in TASK_TYPES:
                return jsonify({'success': False, 'error': '无效的任务类型'}), 400
            
            meta_file_path = get_meta_file_path(task_type, filename)
            if not meta_file_path or not meta_file_path.exists():
                # 如果元文件不存在，返回空模板
                # 从文件名中提取时间信息
                start_time, end_time = extract_time_from_filename(filename)
                created_at = start_time if start_time else datetime.now().isoformat()
                
                # display_name 默认值为原文件名（不含扩展名）
                filename_without_ext = Path(filename).stem
                
                # 根据任务类型设置默认标签
                if task_type == 'anomaly_detection':
                    default_tags_label = [{'key': '健康状态', 'value': '正常'}]
                elif task_type == 'fault_diagnosis':
                    default_tags_label = [{'key': '故障类型', 'value': '正常'}]
                elif task_type == 'rul_prediction':
                    default_tags_label = []  # RUL预测不需要标签
                else:
                    default_tags_label = [{'key': '标签', 'value': '正常'}]
                
                default_meta = {
                    'file_id': f"auto_{uuid.uuid4().hex[:12]}",
                    'file_path': '',  # 将在保存时填充
                    'display_name': filename_without_ext,  # 默认等于原文件名（不含扩展名）
                    'created_at': created_at,
                    'updated_at': datetime.now().isoformat(),
                    'tags_label': default_tags_label,
                    'tags_condition': [],
                    'comment': ''
                }
                
                # RUL预测的默认配置
                if task_type == 'rul_prediction':
                    default_meta['rul_config'] = {
                        'failure_row_index': None,
                        'rul_unit': 'cycle',
                        'max_rul': 200
                    }
                
                # 如果从文件名中提取到了时间，添加到元数据中
                if start_time and end_time:
                    default_meta['data_start_time'] = start_time
                    default_meta['data_end_time'] = end_time
                return jsonify({
                    'success': True,
                    'meta_data': default_meta,
                    'is_new': True
                })
            
            with open(meta_file_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            return jsonify({
                'success': True,
                'meta_data': meta_data,
                'is_new': False
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/api/label/meta/<task_type>/<path:filename>', methods=['POST'])
    def save_meta_file(task_type: str, filename: str):
        """保存元文件"""
        try:
            if task_type not in TASK_TYPES:
                return jsonify({'success': False, 'error': '无效的任务类型'}), 400
            
            data = request.get_json()
            meta_data = data.get('meta_data', {})
            
            # 验证标签至少有一行（RUL预测不需要标签）
            if task_type != 'rul_prediction':
                tags_label = meta_data.get('tags_label', [])
                if not tags_label or len(tags_label) == 0:
                    return jsonify({
                        'success': False,
                        'error': '标签至少需要一行'
                    }), 400
            
            # 验证标签格式
            tags_label = meta_data.get('tags_label', [])
            for tag in tags_label:
                if not isinstance(tag, dict) or 'key' not in tag or 'value' not in tag:
                    return jsonify({
                        'success': False,
                        'error': '标签格式错误，必须包含key和value字段'
                    }), 400
            
            # 验证工况格式（如果有）
            tags_condition = meta_data.get('tags_condition', [])
            for cond in tags_condition:
                if not isinstance(cond, dict) or 'key' not in cond or 'value' not in cond:
                    return jsonify({
                        'success': False,
                        'error': '工况格式错误，必须包含key和value字段'
                    }), 400
            
            # 获取文件路径（从processed目录或labeled目录）
            task_config = TASK_TYPES[task_type]
            processed_path = PROCESSED_DIR / task_config['processed_subdir'] / filename
            labeled_path = LABELED_DIR / task_config['labeled_subdir'] / filename
            
            file_path = ''
            if labeled_path.exists():
                file_path = to_relative_path(labeled_path)
            elif processed_path.exists():
                file_path = to_relative_path(processed_path)
            else:
                return jsonify({
                    'success': False,
                    'error': '对应的数据文件不存在'
                }), 404
            
            # 更新文件路径
            meta_data['file_path'] = file_path
            
            # 创建或更新元文件
            result = create_meta_file(
                task_type=task_type,
                filename=filename,
                file_path=file_path,
                meta_data=meta_data
            )
            
            if not result['success']:
                return jsonify({
                    'success': False,
                    'error': result.get('error', '保存失败')
                }), 500
            
            return jsonify({
                'success': True,
                'message': '元文件保存成功',
                'meta_data': result['meta_data']
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/api/label/preview_file/<task_type>/<path:filename>')
    def preview_file_for_label(task_type: str, filename: str):
        """预览文件内容用于标注（返回前N行数据）"""
        try:
            if task_type not in TASK_TYPES:
                return jsonify({'success': False, 'error': '无效的任务类型'}), 400
            
            # 获取文件路径（从processed目录或labeled目录）
            task_config = TASK_TYPES[task_type]
            processed_path = PROCESSED_DIR / task_config['processed_subdir'] / filename
            labeled_path = LABELED_DIR / task_config['labeled_subdir'] / filename
            
            file_path = None
            if labeled_path.exists():
                file_path = labeled_path
            elif processed_path.exists():
                file_path = processed_path
            else:
                return jsonify({
                    'success': False,
                    'error': '文件不存在'
                }), 404
            
            # 读取CSV文件（RUL预测需要读取全部数据，其他任务可以限制行数）
            max_lines = request.args.get('max_lines')
            if max_lines:
                max_lines = int(max_lines)
                max_lines = min(max_lines, 500)  # 限制最多500行
            else:
                # 如果没有指定max_lines，读取全部数据（用于RUL预测）
                max_lines = None
            
            try:
                # 使用pandas读取CSV
                if max_lines:
                    df = pd.read_csv(file_path, nrows=max_lines)
                else:
                    df = pd.read_csv(file_path)
                
                # 转换为字典列表，便于前端显示
                data = []
                for idx, row in df.iterrows():
                    row_dict = {'index': int(idx)}
                    for col in df.columns:
                        # 处理NaN值
                        value = row[col]
                        if pd.isna(value):
                            row_dict[col] = None
                        else:
                            # 如果是数值，保留原始类型；如果是字符串，直接返回
                            row_dict[col] = value
                    data.append(row_dict)
                
                return jsonify({
                    'success': True,
                    'data': data,
                    'columns': list(df.columns),
                    'total_rows': len(df),
                        'file_path': to_relative_path(file_path)
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'读取文件失败: {str(e)}'
                }), 500
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

