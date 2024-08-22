import os
import torch
import torch.distributed as dist
import torch_npu


def discontinuous_tensor():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    backend = "hccl"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    input_ = torch.rand(2, 3).npu()
    input_ = input_.T
    dist.all_reduce(input_)


discontinuous_tensor()
