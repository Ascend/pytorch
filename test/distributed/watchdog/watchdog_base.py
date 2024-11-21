import os
import time
import datetime
import torch.distributed as dist
import torch
import torch_npu


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('npu:{}'.format(local_rank))
    torch.npu.set_device(device)

    dist.init_process_group(backend='hccl', rank=rank, world_size=2, timeout=datetime.timedelta(seconds=10))
    tensor = torch.tensor(1).npu()

    dist.all_reduce(tensor)
    if rank == 0:
        dist.all_reduce(tensor)
        time.sleep(10)


if __name__ == "__main__":
    main()
