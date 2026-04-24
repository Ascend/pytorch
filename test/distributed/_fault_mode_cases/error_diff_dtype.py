import os
import torch
import torch.distributed as dist
import torch_npu


def diff_dtype():
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    backend = "hccl"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    output = torch.rand(2, 3).npu()
    input_list = [torch.tensor(var).npu() for var in range(2)]
    dist.reduce_scatter(output, input_list)


diff_dtype()
