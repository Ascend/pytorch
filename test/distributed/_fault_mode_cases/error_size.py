import os
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def error_size():
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    backend = "hccl"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    ndev = torch.npu.device_count()
    # output on this rank's device, input_list tensors on a different device
    # -> reduce_scatter rejects input/output residing on different devices.
    out_dev = rank
    in_dev = (rank + 1) % ndev
    output = torch.zeros(4, dtype=torch.float32, device=f"npu:{out_dev}")
    input_list = [torch.zeros(4, dtype=torch.float32, device=f"npu:{in_dev}") for _ in range(2)]
    dist.reduce_scatter(output, input_list)


error_size()
