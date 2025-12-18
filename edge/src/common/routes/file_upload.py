"""
文件上传路由
"""

from flask import Blueprint, request, jsonify, send_file
import os
import sys
from pathlib import Path

# 不需要添加路径，形成拂包结构会自动被找到
from config import Config
from src.common.utils.file_utils import allowed_file, secure_filename

file_upload_bp = Blueprint('file_upload', __name__, url_prefix='/api')


@file_upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """文件上传API"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file part'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error', 
                'message': 'No selected file'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': 'File type not allowed'
            }), 400

        # 安全保存文件
        filename = secure_filename(file.filename)
        upload_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(upload_path)

        return jsonify({
            'status': 'success',
            'message': '文件上传成功',
            'file_info': {
                'filename': filename,
                'path': upload_path,
                'size': os.path.getsize(upload_path) if os.path.exists(upload_path) else 0
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'文件上传失败: {str(e)}'
        }), 500


@file_upload_bp.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """文件下载API"""
    try:
        # 安全检查文件名
        if not filename or '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({
                'status': 'error',
                'message': 'Invalid filename'
            }), 400

        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)

        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404

        # 检查文件是否在上传目录中（防止目录遍历攻击）
        if not os.path.abspath(file_path).startswith(os.path.abspath(Config.UPLOAD_FOLDER)):
            return jsonify({
                'status': 'error',
                'message': 'Access denied'
            }), 403

        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'文件下载失败: {str(e)}'
        }), 500
