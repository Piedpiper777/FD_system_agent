"""
悬浮对话窗占位路由：
- POST /api/chat/input  将前端输入追加写入 input.txt（JSONL）
- GET  /api/chat/output 读取 output.txt 的全文，供前端显示
后续替换为总智能体时，只需替换内部实现，不改前端接口。
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from flask import Blueprint, current_app, jsonify, request

chat_stub_bp = Blueprint('chat_stub', __name__, url_prefix='/api/chat')
_file_lock = threading.Lock()


def _ensure_parent(path: Path) -> None:
    """确保文件父目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, payload: dict) -> None:
    """线程安全地向文件追加一行 JSON。"""
    _ensure_parent(path)
    line = json.dumps(payload, ensure_ascii=False)
    with _file_lock:
        with path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')


@chat_stub_bp.route('/input', methods=['POST'])
def write_input():
    """接收前端输入，写入 input.txt（JSONL）。"""
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'status': 'error', 'message': 'text 不能为空'}), 400

    session_id = (data.get('session_id') or 'default').strip() or 'default'
    trace_id = (data.get('trace_id') or '').strip()
    timestamp = datetime.utcnow().isoformat() + 'Z'

    payload = {
        'timestamp': timestamp,
        'session_id': session_id,
        'trace_id': trace_id,
        'text': text
    }

    input_path = Path(current_app.config['CHAT_INPUT_FILE'])
    try:
        _append_jsonl(input_path, payload)
    except Exception as e:  # noqa: BLE001
        return jsonify({'status': 'error', 'message': f'写入失败: {e}'}), 500

    return jsonify({'status': 'ok'})


@chat_stub_bp.route('/output', methods=['GET'])
def read_output():
    """读取 output.txt 全文，用于在前端输出区域展示。"""
    output_path = Path(current_app.config['CHAT_OUTPUT_FILE'])
    _ensure_parent(output_path)

    try:
        content = output_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        content = ''
    except Exception as e:  # noqa: BLE001
        return jsonify({'status': 'error', 'message': f'读取失败: {e}'}), 500

    return jsonify({'status': 'ok', 'content': content})

