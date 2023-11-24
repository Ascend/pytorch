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

    dist.init_process_group(backend='hccl', timeout=datetime.timedelta(seconds=1))
    tensor = torch.tensor([1024]).npu(non_blocking=True)

    for i in range(10):
        dist.all_reduce(tensor)
        if rank == 1:
            time.sleep(10)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
