import torch_npu.distributed.fsdp._fsdp_collectives
from ._add_fsdp_patch import fully_shard

fully_shard.__module__ = __name__

__all__ = ["fully_shard"]
