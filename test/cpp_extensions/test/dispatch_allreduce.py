import os

os.environ["TASK_QUEUE_ENABLE"] = "1"
os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
os.environ["HCCL_EXEC_TIMEOUT"] = "160"

import torch
import torch.distributed as dist
import torch_npu

import torch_test_cpp_extension.npu as npu_extension

backend = "hccl"
dist.init_process_group(backend)

rank = dist.get_rank()
torch.npu.set_device(rank)
t = torch.rand(2).npu()

dist.all_reduce(t)
t = npu_extension.blocking_ops(t)
dist.all_reduce(t)
