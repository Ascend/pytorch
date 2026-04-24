import os
import torch
import torch.distributed as dist
import torch_npu


def diff_type():
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    backend = "hccl"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    output = torch.tensor(2).npu()
    input_list = [torch.tensor(var).npu() for var in range(2)]
    dist.reduce_scatter(input_list, output)


diff_type()
