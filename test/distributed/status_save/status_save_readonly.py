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
    tensor = torch.tensor(1).npu()
    dist.all_reduce(tensor)

    save_dir = os.environ["TORCH_HCCL_STATUS_SAVE_PATH"]
    # Wait for the watchdog to write the first status file. The default cycle
    # is 2 seconds, and the 10-second deadline provides sufficient margin.
    # This prevents os.chmod(save_dir) from raising FileNotFoundError before the
    # directory has been created.
    deadline = time.time() + 10
    while not os.path.isdir(save_dir):
        if time.time() > deadline:
            raise RuntimeError("watchdog did not create status dir within 10s: " + save_dir)
        time.sleep(0.5)
    # Wait one more cycle to ensure the status file has been persisted.
    time.sleep(2)

    if rank == 0:
        # Make the shared directory read-only. Before the fix, the next
        # checkAndMakePath/write attempt raised from the watchdog and terminated
        # the process.
        os.chmod(save_dir, 0o555)
    dist.barrier()
    # Cover multiple watchdog polling cycles. After the fix, the process should
    # remain alive and only emit a warning.
    time.sleep(6)
    if rank == 0:
        os.chmod(save_dir, 0o755)

    done_dir = os.environ.get("HCCL_UT_DONE_DIR")
    if done_dir:
        open(os.path.join(done_dir, "done_{}".format(rank)), "w").close()


if __name__ == "__main__":
    main()
