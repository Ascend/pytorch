import os
import time
import datetime
import torch
import torch.distributed as dist
import torch_npu

def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('npu:{}'.format(local_rank))
    torch.npu.set_device(device)

    dist.init_process_group(backend='hccl', rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=30))
    sub = dist.new_group(ranks=[0, 1], backend='hccl',
                            timeout=datetime.timedelta(seconds=10))
    tensor = torch.tensor(1).npu()
    dist.all_reduce(tensor)  # Generate a status record for the global PG.
    dist.all_reduce(tensor, group=sub)  # Align the work in the subgroup.
    if rank == 0:
        # Mismatch: rank 1 has no corresponding work, so the subgroup watchdog
        # on rank 0 times out.
        dist.all_reduce(tensor, group=sub)
    time.sleep(30)  # Allow time for the 10-second timeout and error-state write.
    # Rank 0 is expected to terminate without writing a done sentinel; rank 1
    # finishes normally.


if __name__ == "__main__":
    main()
