# TorchNPU DDP 使用说明

## 特性简介

DistributedDataParallel（DDP）是 PyTorch 中一种用于分布式训练的策略，它能够将模型复制到多个进程中，利用多卡并行加速训练，适用于从单机多卡到大规模集群的各种训练场景。

DDP 的核心机制依靠torch.distributed提供的集合通信能力，通过AllReduce操作在所有进程间同步梯度。每个进程持有模型的一份完整副本，处理不同的输入数据分片，在反向传播时梯度自动同步，保证多卡训练的效果与单卡训练效果一致，显著提升了训练吞吐量。在实现层面，DDP 为模型中的每个参数注册了 autograd 钩子。反向传播时，这些钩子触发梯度同步，确保各进程梯度一致后再进行参数更新。

推荐使用DDP的方式是为每个模型副本启动一个独立的进程，每个进程绑定一个 NPU 芯片。进程可以部署在同一台机器上，也可以分布在多台机器上。每个进程都会调用一次 DistributedDataParallel 来包装模型，并通过DistributedSampler确保各进程加载不同的数据子集。

在 TorchNPU 上使用 DDP 时，只需将通信后端设为`hccl`，其余用法与 PyTorch 原生 DDP 基本一致。import torch_npu 时会通过补丁机制进行自动适配，用户无需修改原有代码逻辑。

此外，DDP 支持通过分布式 checkpoint 实现训练状态保存与恢复，并且可以灵活与张量并行（TP）、流水线并行（PP）等并行策略进行组合。

## 代码示例

以下示例使用 `torch.nn.Linear` 作为本地模型，用 DDP 包装，在NPU设备上运行一次前向传播、一次反向传播和一次优化器步骤。模型参数更新之后，所有进程上的模型会完全相同。注释中展示了从GPU设备迁移到NPU设备需要修改的地方。假设代码为train.py,通过`python train.py`进行启动。

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
    # 改动：后端从 "nccl" 改为 "hccl"
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    torch.npu.set_device(rank)                      # 改动：cuda → npu
    model = nn.Linear(10, 10).npu()                 # 改动：cuda() → npu()
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    outputs = ddp_model(torch.randn(20, 10).npu())  # 改动：cuda() → npu()
    labels = torch.randn(20, 10).npu()
    loss_fn(outputs, labels).backward()
    optimizer.step()


def main():
    world_size = torch.npu.device_count()            # 改动：cuda → npu
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"             #可替换为未被占用的其余端口
    main()
```

## 参考文档

如需系统了解 DDP 的设计与用法，建议从[PyTorch 官方 DDP 示例教程](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)入手掌握标准实践，接口细节可通过[API 文档](https://docs.pytorch.org/docs/2.13/generated/torch.nn.parallel.DistributedDataParallel.html)查阅，若希望深入底层同步机制，可进一步阅读[开发者笔记](https://docs.pytorch.org/docs/2.13/notes/ddp.html)。
