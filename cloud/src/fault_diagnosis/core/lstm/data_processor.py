"""
故障诊断 LSTM - 数据处理器
负责加载、预处理和构建分类任务的时序数据

注意：LSTM 分类任务的数据处理逻辑与 CNN 1D 完全相同，
因此直接复用 CNN 1D 的数据处理器。

标签处理说明：
    标签由前端传递的 label_index 决定，而不是从文件名提取。
"""

# 直接导入并重新导出 CNN 1D 的数据处理器
from ..cnn_1d.data_processor import (
    ClassificationData,
    DataProcessor,
)

__all__ = [
    "ClassificationData",
    "DataProcessor",
]

