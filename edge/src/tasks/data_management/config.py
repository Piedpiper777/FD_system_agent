"""
数据管理模块配置
"""
from pathlib import Path

# 路径配置
EDGE_DIR = Path(__file__).resolve().parents[3]
MONITOR_DATA_FILE = EDGE_DIR / 'data' / 'monitor' / 'monitor_data.csv'
COLLECTED_DIR = EDGE_DIR / 'data' / 'collected'
PROCESSED_DIR = EDGE_DIR / 'data' / 'processed'
LABELED_DIR = EDGE_DIR / 'data' / 'labeled'
META_DIR = EDGE_DIR / 'data' / 'meta'  # 元文件目录

# 任务类型配置
TASK_TYPES = {
    'anomaly_detection': {
        'name': '异常检测',
        'has_label': True,  # 异常检测也需要标注（元数据）
        'processed_subdir': 'AnomalyDetection',
        'labeled_subdir': 'AnomalyDetection',  # 异常检测的标注数据目录
    },
    'fault_diagnosis': {
        'name': '故障诊断',
        'has_label': True,
        'processed_subdir': 'FaultDiagnosis',
        'labeled_subdir': 'FaultDiagnosis',
    },
    'rul_prediction': {
        'name': 'RUL预测',
        'has_label': True,
        'processed_subdir': 'RULPrediction',
        'labeled_subdir': 'RULPrediction',
    }
}

# 允许浏览的数据目录
ALLOWED_DATA_DIRS = {
    'processed': EDGE_DIR / 'data' / 'processed',
    'labeled': EDGE_DIR / 'data' / 'labeled',
    'training': EDGE_DIR / 'data' / 'training',
}

