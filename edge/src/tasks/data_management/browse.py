"""
文件浏览相关API
"""
from datetime import datetime
from flask import Blueprint, jsonify, request
from .config import ALLOWED_DATA_DIRS
from .utils import format_file_size, get_dir_display_name, get_parent_subdir


def register_browse_routes(bp: Blueprint):
    """注册文件浏览相关路由"""
    
    @bp.route('/api/browse/directories')
    def get_browse_directories():
        """获取可浏览的数据目录列表"""
        try:
            print(f"[数据管理] 收到目录列表请求 - 这是 Edge 端的请求")
            directories = []
            
            for key, dir_path in ALLOWED_DATA_DIRS.items():
                # 创建目录（如果不存在）
                if not dir_path.exists():
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        print(f"创建目录: {dir_path}")
                    except Exception as e:
                        print(f"创建目录失败 {dir_path}: {e}")
                
                if dir_path.exists():
                    # 获取子目录列表
                    subdirs = []
                    try:
                        for item in sorted(dir_path.iterdir()):
                            if item.is_dir():
                                # 统计子目录中的文件数（包括所有子目录中的文件）
                                file_count = sum(1 for f in item.rglob('*') if f.is_file())
                                subdirs.append({
                                    'name': item.name,
                                    'file_count': file_count
                                })
                    except Exception as e:
                        print(f"读取目录 {dir_path} 失败: {e}")
                        import traceback
                        traceback.print_exc()
                        subdirs = []
                    
                    directories.append({
                        'key': key,
                        'name': get_dir_display_name(key),
                        'path': str(dir_path),
                        'exists': True,
                        'subdirs': subdirs
                    })
                else:
                    directories.append({
                        'key': key,
                        'name': get_dir_display_name(key),
                        'path': str(dir_path),
                        'exists': False,
                        'subdirs': []
                    })
            
            return jsonify({
                'success': True,
                'directories': directories
            })
            
        except Exception as e:
            print(f"获取目录列表失败: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @bp.route('/api/browse/files')
    def get_browse_files():
        """获取指定目录下的文件和子目录列表"""
        try:
            dir_key = request.args.get('dir', '').strip()
            subdir = request.args.get('subdir', '').strip()
            
            if not dir_key or dir_key not in ALLOWED_DATA_DIRS:
                return jsonify({
                    'success': False,
                    'error': '无效的目录'
                }), 400
            
            base_dir = ALLOWED_DATA_DIRS[dir_key]
            
            if subdir:
                target_dir = base_dir / subdir
                # 安全检查
                try:
                    target_dir.resolve().relative_to(base_dir.resolve())
                except ValueError:
                    return jsonify({
                        'success': False,
                        'error': '非法的目录路径'
                    }), 400
            else:
                target_dir = base_dir
            
            if not target_dir.exists():
                return jsonify({
                    'success': True,
                    'files': [],
                    'folders': [],
                    'current_path': str(target_dir),
                    'parent_subdir': get_parent_subdir(subdir)
                })
            
            files = []
            folders = []
            
            for item in sorted(target_dir.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                stat = item.stat()
                
                if item.is_dir():
                    # 统计子目录中的文件数
                    file_count = sum(1 for f in item.rglob('*') if f.is_file())
                    folders.append({
                        'name': item.name,
                        'file_count': file_count,
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'subdir_path': f"{subdir}/{item.name}" if subdir else item.name
                    })
                else:
                    # 获取文件行数（仅对CSV文件）
                    line_count = None
                    if item.suffix.lower() == '.csv':
                        try:
                            with open(item, 'r', encoding='utf-8') as f:
                                line_count = sum(1 for _ in f)
                        except:
                            pass
                    
                    # 使用相对路径标识符而不是完整路径
                    relative_path = f"{subdir}/{item.name}" if subdir else item.name
                    
                    files.append({
                        'name': item.name,
                        'size': format_file_size(stat.st_size),
                        'size_bytes': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'extension': item.suffix.lower(),
                        'line_count': line_count,
                        'relative_path': relative_path,  # 相对于 base_dir 的路径
                        'dir_key': dir_key
                    })
            
            return jsonify({
                'success': True,
                'files': files,
                'folders': folders,
                'current_path': str(target_dir),
                'dir_key': dir_key,
                'subdir': subdir,
                'parent_subdir': get_parent_subdir(subdir)
            })
            
        except Exception as e:
            print(f"获取文件列表失败: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @bp.route('/api/browse/preview')
    def preview_file():
        """预览文件内容（前N行）"""
        try:
            # 使用 dir_key 和 relative_path 来定位文件
            dir_key = request.args.get('dir', '').strip()
            relative_path = request.args.get('file', '').strip()
            lines = int(request.args.get('lines', 50))
            
            if not dir_key or not relative_path:
                return jsonify({
                    'success': False,
                    'error': '请指定目录和文件路径'
                }), 400
            
            if dir_key not in ALLOWED_DATA_DIRS:
                return jsonify({
                    'success': False,
                    'error': '无效的目录'
                }), 400
            
            # 限制预览行数
            lines = min(lines, 200)
            
            # 构建完整路径
            base_dir = ALLOWED_DATA_DIRS[dir_key]
            file_path = base_dir / relative_path
            
            # 安全检查
            try:
                file_path.resolve().relative_to(base_dir.resolve())
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
            
            # 读取文件内容
            content_lines = []
            headers = []
            total_lines = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        total_lines += 1
                        if i < lines:
                            content_lines.append(line.rstrip('\n\r'))
                        if i == 0 and file_path.suffix.lower() == '.csv':
                            # CSV文件的表头
                            headers = line.rstrip('\n\r').split(',')
            except UnicodeDecodeError:
                # 尝试其他编码
                with open(file_path, 'r', encoding='gbk') as f:
                    for i, line in enumerate(f):
                        total_lines += 1
                        if i < lines:
                            content_lines.append(line.rstrip('\n\r'))
                        if i == 0 and file_path.suffix.lower() == '.csv':
                            headers = line.rstrip('\n\r').split(',')
            
            return jsonify({
                'success': True,
                'preview': {
                    'filename': file_path.name,
                    'total_lines': total_lines,
                    'preview_lines': len(content_lines),
                    'headers': headers,
                    'content': content_lines,
                    'is_csv': file_path.suffix.lower() == '.csv'
                }
            })
            
        except Exception as e:
            print(f"预览文件失败: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @bp.route('/api/browse/delete', methods=['DELETE'])
    def delete_browse_file():
        """删除指定的数据文件"""
        try:
            data = request.get_json()
            dir_key = data.get('dir', '').strip()
            relative_path = data.get('file', '').strip()
            
            if not dir_key or not relative_path:
                return jsonify({
                    'success': False,
                    'error': '请指定目录和文件路径'
                }), 400
            
            if dir_key not in ALLOWED_DATA_DIRS:
                return jsonify({
                    'success': False,
                    'error': '无效的目录'
                }), 400
            
            # 构建完整路径
            base_dir = ALLOWED_DATA_DIRS[dir_key]
            file_path = base_dir / relative_path
            
            # 安全检查
            try:
                file_path.resolve().relative_to(base_dir.resolve())
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
            
            filename = file_path.name
            file_path.unlink()
            
            return jsonify({
                'success': True,
                'message': f'已删除文件: {filename}'
            })
            
        except Exception as e:
            print(f"删除文件失败: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

