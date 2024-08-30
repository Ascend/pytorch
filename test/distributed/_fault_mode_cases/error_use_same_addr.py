import os
import torch
import torch.distributed as dist
import torch_npu


def same_addr():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    backend = "hccl"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    input_ = torch.randn(100, 100, 20).npu()
    dist.all_reduce(input_)


same_addr()
