import torch

__all__ = ["initialize_afd_bindings"]

_OP_NAMES = [
    "attention_worker_scheduler_",
    "attention_worker_scheduler",
    "ffn_worker_scheduler_",
    "ffn_worker_scheduler"]


def initialize_afd_bindings():
    import torch_npu._afd as npu_afd
    for name in _OP_NAMES:
        setattr(npu_afd, name, getattr(torch.ops.npu, name))
