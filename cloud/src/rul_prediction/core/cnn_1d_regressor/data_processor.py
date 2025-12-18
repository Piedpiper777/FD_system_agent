"""
CNN 1D 回归器数据处理
当前复用BiLSTM/GRU回归器的数据处理逻辑。
"""

from ..bilstm_gru_regressor.data_processor import DataProcessor, RegressionData

__all__ = ['DataProcessor', 'RegressionData']
