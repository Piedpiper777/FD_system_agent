"""
文件处理工具
"""

import os
import uuid
from werkzeug.utils import secure_filename
import sys
from pathlib import Path

# 不需要添加路径，子气处理应详形成拂包
# 当从 edge/ 目录执行时，config.py 会自动被找到
from config import Config

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_filename(filename):
    """安全处理文件名"""
    return secure_filename(filename)

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