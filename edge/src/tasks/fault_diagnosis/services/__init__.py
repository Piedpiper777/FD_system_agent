"""
故障诊断服务层初始化
"""

from .inferencer import FaultDiagnosisInferencer
from .trainer import FaultDiagnosisTrainer

__all__ = [
    'FaultDiagnosisInferencer',
    'FaultDiagnosisTrainer'
]
