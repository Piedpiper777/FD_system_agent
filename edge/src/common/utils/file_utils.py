"""
文件处理工具
"""

import os
import uuid
from werkzeug.utils import secure_filename as werkzeug_secure_filename
import sys
from pathlib import Path

# 添加项目根目录到path
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import Config

# 使用配置文件中的允许扩展名设置


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def secure_filename(filename):
    """安全处理文件名"""
    return werkzeug_secure_filename(filename)


def save_uploaded_file(file):
    """保存上传的文件"""
    if not allowed_file(file.filename):
        raise ValueError("File type not allowed")

    # 生成唯一文件名
    original_filename = secure_filename(file.filename)
    file_extension = original_filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{file_extension}"

    # 确保上传目录存在
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    # 保存文件
    file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    return file_path
