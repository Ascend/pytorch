import torch
from torch.distributed.tensor._ops._view_ops import register_op_strategy_map
npu = torch.ops.npu


register_op_strategy_map(npu.npu_transpose.default, torch.permute)
