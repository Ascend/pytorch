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
    input_a = torch.randn(800, 100).npu()
    input_b = torch.randn(100, 600).npu()
    for _ in range(5000):
        result = torch.matmul(input_a, input_b)
        result_mean = result.mean()
    dist.all_reduce(input_a)


same_addr()
