import os
import torch
import torch.distributed as dist
import torch_npu


def error_size():
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    backend = "hccl"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    output = torch.tensor(2).npu()
    input_list = [torch.tensor(var).npu() for var in range(3)]
    dist.reduce_scatter(output, input_list)


error_size()
