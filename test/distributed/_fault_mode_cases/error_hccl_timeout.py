import os
import torch
import torch.distributed as dist
import torch_npu


def hccl_timeout():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    os.environ["HCCL_EXEC_TIMEOUT"] = "180"
    backend = "hccl"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    tensor = torch.randn(10, 10, dtype=torch.float16).npu()
    zeros = torch.zeros(10, 10, dtype=torch.float16).npu()
    out = zeros if rank > 0 else tensor
    dist.all_reduce(out)
    if rank == 1:
        exit()
    dist.all_reduce(out)


hccl_timeout()
