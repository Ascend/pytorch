import os
import warnings
import torch
import torch.distributed as dist
import torch_npu

warnings.filterwarnings('always')


def error_param():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    backend = "hccl"
    dist.init_process_group(backend)
    new_pg = dist.new_group([0])
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    input_ = torch.tensor(2).npu()

    dist.broadcast(input_, src=0, group=new_pg)


error_param()
