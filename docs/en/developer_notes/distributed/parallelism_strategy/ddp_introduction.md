# DDP Parallelism Strategy

## Introduction

DistributedDataParallel (DDP) is a strategy used for distributed training in PyTorch. It replicates the model across multiple processes, using multi-card parallelism to accelerate training.

The core mechanism of DDP relies on the collective communication capabilities provided by `torch.distributed`, synchronizing gradients across all processes through the `AllReduce` operation. Each process holds a complete replica of the model and processes different input data shards. During backpropagation, gradients are automatically synchronized, ensuring that multi-card training achieves results consistent with single-card training and significantly improving training throughput. At the implementation level, DDP registers an autograd hook for each parameter in the model. During backpropagation, these hooks trigger gradient synchronization, ensuring that gradients are consistent across all processes before parameter updates.

The recommended way to use DDP is to start an independent process for each model replica, with each process bound to one NPU chip. The processes can be deployed on the same machine or distributed across multiple machines. Each process calls DistributedDataParallel once to wrap the model and uses DistributedSampler to ensure that each process loads a different data subset.

In addition, DDP supports saving and restoring training states through distributed checkpoints, and can be flexibly combined with parallelism strategies such as tensor parallelism (TP) and pipeline parallelism (PP).

To systematically understand the design and usage of DDP, it is recommended to start with the [Official PyTorch DDP Example Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) to master the standard practices. Interface details can be found in the [API Documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html). To gain an in-depth understanding of the underlying synchronization mechanism, you can further read the [Developer Notes](https://docs.pytorch.org/docs/2.14/notes/ddp.html).

## Application Scenarios

It applies to various training scenarios from single-machine multi-card to large-scale clusters.

## Usage Guide

When using DDP on TorchNPU, you only need to set the communication backend to `hccl`; the rest of the usage is consistent with native PyTorch DDP. `import torch_npu` automatically adapts through the patching mechanism, but device-related calls still need to be replaced with `.npu()`.

## Usage Examples

The following example uses `torch.nn.Linear` as the local model, wraps it with DDP, and runs one forward pass, one backward pass, and one optimizer step on the NPU device. After the model parameters are updated, the models on all processes are exactly the same. The comments show what needs to be modified when migrating from a GPU device to an NPU device. Assume the code is `train.py` and start it with `python train.py`.

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # Change: the backend is changed from "nccl" to "hccl".
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    torch.npu.set_device(rank)                      # Change: cuda → npu.
    model = nn.Linear(10, 10).npu()                 # Change: cuda() → npu().
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    outputs = ddp_model(torch.randn(20, 10).npu())  # Change: cuda() → npu().
    labels = torch.randn(20, 10).npu()
    loss_fn(outputs, labels).backward()
    optimizer.step()


def main():
    world_size = torch.npu.device_count()            # Change: cuda → npu.
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"            # Can be replaced with another unoccupied port.
    main()
```

## Constraints

None
