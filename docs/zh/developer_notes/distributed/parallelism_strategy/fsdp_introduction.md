# FSDP分片并行策略

## 简介

FullyShardedDataParallel（FSDP）是PyTorch中一种用于大规模分布式训练的策略，它将模型参数、梯度和优化器状态分片到多个设备上，显著降低单卡显存占用，适用于从单机多卡到大规模集群的各种训练场景。

FSDP的核心机制依靠torch.distributed提供的集合通信能力，通过All-Gather和Reduce-Scatter操作在各设备间同步参数和梯度。训练时，FSDP按需收集当前计算所需的完整参数，计算完成后立即释放，从而在保持数据并行训练效率的同时，大幅节省显存资源，支持更大规模模型的训练。

FSDP2（fully_shard API）是PyTorch推荐的新版实现，支持逐层分片和更细粒度的控制，便于根据模型结构灵活配置分片策略。推荐使用FSDP2的方式是为每个模型副本启动一个独立的进程，每个进程绑定一个NPU芯片。进程可部署于同一机器或多台机器上。

此外，FSDP2可与张量并行（TP）、流水线并行（PP）等策略灵活组合，满足不同规模模型的训练需求，同时支持分布式checkpoint实现训练状态的保存与恢复。

如需系统了解FSDP/FSDP2的设计与用法，建议从[PyTorch FSDP示例教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)入手掌握标准实践，接口细节可通过[API 文档](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)查阅。

## 使用场景

适用于从单机多卡到大规模集群的各种训练场景。

## 使用指导

在TorchNPU上使用FSDP2时，只需将通信后端设为`hccl`，用法与PyTorch原生FSDP2一致，`import torch_npu`时会通过补丁机制进行自动适配，用户无需修改原有代码逻辑。

## 使用样例

以下示例使用torch.nn.Sequential构建一个4层Linear的简单模型，使用FSDP2的fully_shard API在TorchNPU上训练5个epoch。模型参数、梯度和优化器状态被分片存储在各NPU芯片上，以降低单卡显存占用。假设代码为train.py，通过`torchrun train.py`进行启动。

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
    # 改动：后端从 "nccl" 改为 "hccl"
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

## 约束说明

无
