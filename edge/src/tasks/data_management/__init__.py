"""
数据管理模块
提供数据截取、导出、标注、预处理等功能
"""

from flask import Blueprint, render_template

# 创建蓝图
data_management_bp = Blueprint('data_management', __name__, url_prefix='/data_management')

# 导入并注册所有路由
from . import monitor
from . import browse
from . import preprocess
from . import label

# 注册路由
monitor.register_monitor_routes(data_management_bp)
browse.register_browse_routes(data_management_bp)
preprocess.register_preprocess_routes(data_management_bp)
label.register_label_routes(data_management_bp)


@data_management_bp.route('/')
def index():
    """数据管理主页"""
    return render_template('data_management/index.html')
