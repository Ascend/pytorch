import torch
import torch_npu
from torch_npu.contrib.function._matmul_transpose import MatmulApply

__all__ = ["matmul_transpose"]


matmul_transpose = MatmulApply.apply
