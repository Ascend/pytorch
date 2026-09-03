# TorchNPU FSDP 使用说明

## 特性简介

FullyShardedDataParallel（FSDP） 是 PyTorch 中一种用于大规模分布式训练的策略，它将模型参数、梯度和优化器状态分片到多个设备上，显著降低单卡显存占用，适用于从单机多卡到大规模集群的各种训练场景。

FSDP 的核心机制依靠 torch.distributed 提供的集合通信能力，通过 All-Gather 和 Reduce-Scatter 操作在各设备间同步参数和梯度。训练时，FSDP 按需收集当前计算所需的完整参数，计算完成后立即释放，从而在保持数据并行训练效率的同时，大幅节省显存资源，支持更大规模模型的训练。

推荐使用 FSDP 的方式是为每个模型副本启动一个独立的进程，每个进程绑定一个 NPU 芯片。
进程可部署于同一机器或多机器之上。

FSDP2（fully_shard API）是 PyTorch 推荐的新版实现，支持逐层分片和更细粒度的控制，便于根据模型结构灵活配置分片策略。在 TorchNPU 上使用 FSDP2 时，用法与 PyTorch 原生 FSDP2 基本一致，import torch_npu 时会通过补丁机制进行自动适配，用户无需修改原有代码逻辑。

此外，FSDP2 可与张量并行（TP）、流水线并行（PP）等策略灵活组合，满足不同规模模型的训练需求，同时支持分布式 checkpoint 实现训练状态的保存与恢复。

## 代码示例

以下示例使用 torch.nn.Sequential 构建一个 4 层 Linear 的简单模型，使用 FSDP2 的 fully_shard API 在 TorchNPU 上训练 5 个 epoch。模型参数、梯度和优化器状态被分片存储在各 NPU 芯片上，以降低单卡显存占用。假设代码为train.py,通过`torchrun train.py`进行启动。

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

## 参考文档

如需系统了解 FSDP/FSDP2 的设计与用法，建议从[PyTorch FSDP 示例教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)入手掌握标准实践，接口细节可通过[API 文档](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html)查阅。
