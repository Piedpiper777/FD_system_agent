"""
故障诊断 ResNet 1D - 数据处理器
负责加载、预处理和构建分类任务的时序数据

注意：ResNet 1D 分类任务的数据处理逻辑与 CNN 1D 完全相同，
因此直接复用 CNN 1D 的数据处理器。

标签处理说明：
    标签由前端传递的 label_index 决定，而不是从文件名提取。
    用户在训练页面设置标签列表（如"正常,内圈故障,外圈故障"），
    然后为每个标签选择对应的数据文件，标签与文件的绑定关系在前端完成。
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
