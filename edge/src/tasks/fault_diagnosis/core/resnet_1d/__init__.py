"""
故障诊断 ResNet 1D 模块 (Edge端)
用于模型推理
"""

from .model_builder import (
    ResNet1DClassifier,
    ResidualBlock1D,
    ModelBuilder,
)

__all__ = [
    "ResNet1DClassifier",
    "ResidualBlock1D",
    "ModelBuilder",
]

