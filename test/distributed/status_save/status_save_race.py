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
    dist.init_process_group(backend='hccl', rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30))
    # Create multiple PGs in quick succession so their watchdog threads enter
    # the first runLoop iteration at nearly the same time and concurrently
    # access/create a missing status directory. Before the fix, the EEXIST race
    # could terminate the process.
    sub_groups = [dist.new_group(
        ranks=[0, 1], backend='hccl', timeout=datetime.timedelta(seconds=30)) for _ in range(3)]
    tensor = torch.tensor(1).npu()
    dist.all_reduce(tensor)
    for g in sub_groups:
        dist.all_reduce(tensor, group=g)
    time.sleep(5)  # Cover the first status-save write cycle (2 seconds by default).
    dist.barrier()

    done_dir = os.environ.get("HCCL_UT_DONE_DIR")
    if done_dir:
        open(os.path.join(done_dir, "done_{}".format(rank)), "w").close()

if __name__ == "__main__":
    main()
