import os
import time
import datetime
import torch.distributed as dist
import torch
import torch_npu


def main():
    torch.npu.set_compile_mode(jit_compile=True)
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('npu:{}'.format(local_rank))
    torch.npu.set_device(device)
    dist.init_process_group(backend='hccl', rank=rank, world_size=2, timeout=datetime.timedelta(seconds=120))
    tensor = torch.tensor(1).npu()
    dist.all_reduce(tensor)
    if rank == 0:
        x1 = torch.randn(3).float().npu()
        x2 = torch.randn(1).long().npu()
        x3 = torch.randn(1).float().npu()
        y = torch.addcmul(x1, x2, x3)
        dist.all_reduce(tensor)


if __name__ == "__main__":
    main()
