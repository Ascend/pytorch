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
    groups = [dist.new_group(ranks=[0, 1], backend='hccl',
                                            timeout=datetime.timedelta(seconds=30)) for _ in range(2)]
    tensor = torch.tensor(1).npu()
    # Continuously generate traffic on the global PG and all subgroups across
    # multiple status-write cycles to stress concurrent status updates.
    for _ in range(5):
        dist.all_reduce(tensor)
        for g in groups:
            dist.all_reduce(tensor, group=g)
        time.sleep(1)
    dist.barrier()

    done_dir = os.environ.get("HCCL_UT_DONE_DIR")
    if done_dir:
        open(os.path.join(done_dir, "done_{}".format(rank)), "w").close()


if __name__ == "__main__":
    main()
