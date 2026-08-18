import os
import time
import datetime
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('npu:{}'.format(local_rank))
    torch.npu.set_device(device)

    dist.init_process_group(backend='hccl', rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=30))
    tensor = torch.tensor(1).npu()
    dist.all_reduce(tensor)
    # Cover multiple watchdog polling cycles. Before the fix, an invalid path
    # was fatal during the first cycle.
    time.sleep(4)
    dist.barrier()
    done_dir = os.environ.get("HCCL_UT_DONE_DIR")
    if done_dir:
        open(os.path.join(done_dir, "done_{}".format(rank)), "w").close()

if __name__ == "__main__":
    main()
