# FSDP Sharding Parallelism Strategy

## Introduction

FullyShardedDataParallel (FSDP) is a strategy used for large-scale distributed training in PyTorch. It shards model parameters, gradients, and optimizer states across multiple devices, significantly reducing per-card memory usage. It applies to various training scenarios from single-machine multi-card to large-scale clusters.

The core mechanism of FSDP relies on the collective communication capabilities provided by torch.distributed, synchronizing parameters and gradients across devices through All-Gather and Reduce-Scatter operations. During training, FSDP collects the complete parameters required for the current computation on demand and releases them immediately after computation, thereby greatly saving memory resources while maintaining the efficiency of data parallelism training, and supporting the training of larger-scale models.

FSDP2 (the fully_shard API) is the new implementation recommended by PyTorch. It supports layer-by-layer sharding and more fine-grained control, making it easy to flexibly configure sharding strategies based on the model structure. The recommended way to use FSDP2 is to start an independent process for each model replica, with each process bound to one NPU chip. The processes can be deployed on the same machine or across multiple machines.

In addition, FSDP2 can be flexibly combined with strategies such as tensor parallelism (TP) and pipeline parallelism (PP) to meet the training needs of models of different scales. It also supports distributed checkpoints for saving and restoring training states.

To systematically understand the design and usage of FSDP/FSDP2, it is recommended to start with the [PyTorch FSDP Example Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) to master the standard practices. Interface details can be found in the [API Documentation](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html).

## Application Scenarios

It applies to various training scenarios from single-machine multi-card to large-scale clusters.

## Usage Guide

When using FSDP2 on TorchNPU, you only need to set the communication backend to `hccl`. The usage is consistent with native PyTorch FSDP2. `import torch_npu` automatically adapts through the patching mechanism, and users do not need to modify the original code logic.

## Usage Examples

The following example uses torch.nn.Sequential to build a simple 4-layer Linear model and trains it for 5 epochs on TorchNPU using the fully_shard API of FSDP2. Model parameters, gradients, and optimizer states are sharded and stored on each NPU chip to reduce per-card memory usage. Assume the code is train.py and start it with `torchrun train.py`.

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributed.fsdp import fully_shard


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Change: the backend is changed from "nccl" to "hccl".
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    torch.npu.set_device(rank)

    layers = nn.ModuleList([nn.Linear(512, 512) for _ in range(4)])
    model = nn.Sequential(*layers).npu()

    for layer in model:
        fully_shard(layer)
    fully_shard(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for step in range(5):
        x = torch.randn(32, 512).npu()
        labels = torch.randn(32, 512).npu()
        loss = loss_fn(model(x), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## Constraints

None
