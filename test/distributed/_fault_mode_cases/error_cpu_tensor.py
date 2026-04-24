import os
import torch
import torch.distributed as dist
import torch_npu


def cpu_tensor():
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    backend = "hccl"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    input_ = torch.rand(2, 3)
    dist.all_reduce(input_)


cpu_tensor()
