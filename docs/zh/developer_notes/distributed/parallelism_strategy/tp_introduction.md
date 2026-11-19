# TP并行策略

## 简介

Tensor Parallelism（TP）是PyTorch中一种用于大规模分布式训练的策略，它将带权重的层（如Linear、Attention）沿特定维度切分到多个设备上并行计算，降低单卡显存占用和单卡计算量。通常部署在单机多卡场景，通过节点内的高速互联完成通信；大规模集群训练时，建议与FSDP、PP等策略组合使用。

TP的核心机制依靠torch.distributed.tensor.parallel提供的并行化能力，通过parallelize_module配合ColwiseParallel、RowwiseParallel等切分方式，将权重矩阵按列或按行切开，分配到不同设备上。在保持模型数学等价的前提下支持更大规模模型的训练。

此外，TP常与FSDP2组合使用形成二维并行：将单机内的多卡归入一个TP组，组内通过节点内高速互联通信；跨机方向由FSDP负责参数分片与All-Gather，跨机通信的数据量随TP切分而显著降低，从而减小通信延迟带来的影响，满足不同规模模型的训练需求。

如需系统了解TP的设计与用法，建议从[PyTorch TP示例教程](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)入手掌握标准实践，接口细节可通过[API 文档](https://docs.pytorch.org/docs/2.15/distributed.tensor.parallel.html)查阅。

## 使用场景

以单机多卡（节点内高速互联）为主；跨机大规模训练时建议与FSDP2、PP等策略组合使用。

## 使用指导

在TorchNPU上使用TP时，只需将通信后端设为`hccl`，其余用法与PyTorch原生TP一致，`import torch_npu`时会通过补丁机制进行自动适配，用户无需修改除设备相关调用外的原有代码逻辑。

## 使用样例

以下示例使用torch.nn.Sequential构建一个4层Linear的简单模型，使用张量并行（TP）的parallelize_module API在TorchNPU上训练5个step。各层Linear的权重通过ColwiseParallel与RowwiseParallel逐层切分并存储在各NPU芯片上，以降低单卡显存占用。假设代码文件为train_tp.py，通过执行`torchrun --nproc_per_node=4 train_tp.py`启动。

```python
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch_npu
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

SEED = 42
HIDDEN = 512
NUM_LAYERS = 4
STEPS = 5
BATCH = 32

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 后端从 "nccl" 改为 "hccl"
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    torch.npu.set_device(local_rank)

    torch.manual_seed(SEED)
    torch.npu.manual_seed(SEED)

    assert HIDDEN % world_size == 0, "hidden 维度必须能被卡数整除"

    layers = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN) for _ in range(NUM_LAYERS)])
    model = nn.Sequential(*layers).npu()

    mesh = init_device_mesh("npu", (world_size,))

    for i, layer in enumerate(model):
        if i % 2 == 0:
            parallelize_module(layer, mesh, ColwiseParallel())
        else:
            parallelize_module(layer, mesh, RowwiseParallel())

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    gen = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(BATCH, HIDDEN, generator=gen).npu()
    labels = torch.randn(BATCH, HIDDEN, generator=gen).npu()

    for step in range(STEPS):
        loss = loss_fn(model(x), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_vec = loss.detach().reshape(1)
        gathered = [torch.empty_like(loss_vec) for _ in range(world_size)]
        dist.all_gather(gathered, loss_vec)
        if rank == 0:
            vals = ", ".join(f"{g.item():.6f}" for g in gathered)
            print(f"Step {step}: loss(按 rank 0..{world_size - 1}) = [{vals}]", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## 约束说明

在进行TP时，请确认被切分的特征维度（如隐藏层维度、注意力头数等）能够被TP组内的卡数整除，否则前向计算会因分片形状不匹配而报错。
