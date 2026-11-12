import datetime
import os
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("npu:{}".format(local_rank))
    torch.npu.set_device(device)

    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    tensor = torch.tensor(1, device=device)

    # Collective and P2P sequence numbers have independent domains, so these
    # two operations both use sequence 1. StatusInfo must still retain the P2P
    # operation instead of treating it as a duplicate of the collective.
    dist.all_reduce(tensor)
    if rank == 0:
        dist.recv(tensor, src=1)
    else:
        dist.send(tensor, dst=0)

    # Let the watchdog observe completion and cover at least one status-save
    # interval without issuing another collective that would replace the P2P
    # status under test.
    time.sleep(4)
    done_dir = os.environ.get("HCCL_UT_DONE_DIR")
    if done_dir:
        open(os.path.join(done_dir, "done_{}".format(rank)), "w").close()


if __name__ == "__main__":
    main()
