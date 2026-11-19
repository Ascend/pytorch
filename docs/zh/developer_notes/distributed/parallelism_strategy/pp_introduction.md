# PP并行策略

## 简介

Pipeline Parallelism（PP）是PyTorch中一种用于大规模分布式训练的流水线并行策略，它将模型按层切分为多个阶段，每个阶段放置在不同设备上，训练时将输入数据切分为多个微批次，使各阶段以流水线方式并行处理不同的微批次，降低单卡显存占用，适用于从单机多卡到大规模集群的各种训练场景。

PP的核心机制依靠torch.distributed.pipelining提供的流水线能力，由拆分前端和分布式运行时两部分组成：拆分前端负责处理模型代码，将其拆分为多个模型分区，并捕获分区间的数据流关系；分布式运行时在不同设备上并行执行各个流水线阶段，处理微批次拆分、调度、通信和梯度传播等任务。

此外，PP可与数据并行（DDP、FSDP）、张量并行（TP）等策略灵活组合，例如TorchTitan在Llama模型上采用FSDP（实现数据并行）、TP与PP组合的“3D并行”。

如需系统了解PP的设计与用法，建议从[PyTorch PP示例教程](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html)入手掌握标准实践，接口细节可通过[API 文档](https://docs.pytorch.org/docs/2.15/distributed.pipelining.html)查阅。

## 使用场景

适用于从单机多卡到大规模集群的各种训练场景。

## 使用指导

在TorchNPU上使用PP时，只需将通信后端设为`hccl`，其余用法与PyTorch原生PP一致，`import torch_npu`时会通过补丁机制对点对点通信等接口进行自动适配，用户无需修改除设备相关调用外的原有代码逻辑。

## 使用样例

以下示例使用torch.nn.Sequential构建一个4层Linear的简单模型，使用流水线并行（PP）的PipelineStage与ScheduleGPipe API在TorchNPU上训练5个step。模型按层切分为多个stage并分配到各NPU芯片上，每张卡在NPU上只保存并计算自己负责的部分层，stage之间通过点对点通信传递激活值，以降低单卡显存占用。假设代码文件为train_pp.py，通过执行`torchrun --nproc_per_node=4 train_pp.py`启动。

```python
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch_npu
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

SEED = 42
HIDDEN = 512
NUM_LAYERS = 4
STEPS = 5
BATCH = 32
NUM_MICROBATCHES = 4


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 后端从 "nccl" 改为 "hccl"
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    torch.npu.set_device(local_rank)

    torch.manual_seed(SEED)
    torch.npu.manual_seed(SEED)

    assert NUM_LAYERS % world_size == 0, "层数必须能被卡数整除"
    layers_per_stage = NUM_LAYERS // world_size

    layers = [nn.Linear(HIDDEN, HIDDEN) for _ in range(NUM_LAYERS)]
    start = rank * layers_per_stage
    stage_layers = layers[start : start + layers_per_stage]
    model = nn.Sequential(*stage_layers).npu()

    device = torch.device(f"npu:{local_rank}")

    stage = PipelineStage(model, rank, world_size, device)
    schedule = ScheduleGPipe(stage, NUM_MICROBATCHES, loss_fn=nn.MSELoss())
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    gen = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(BATCH, HIDDEN, generator=gen).npu()
    labels = torch.randn(BATCH, HIDDEN, generator=gen).npu()

    for step in range(STEPS):
        if stage.is_first:
            schedule.step(x)
        elif stage.is_last:
            losses = []
            schedule.step(target=labels, losses=losses)
        else:
            schedule.step()
        optimizer.step()
        optimizer.zero_grad()

        if stage.is_last:
            step_loss = sum(l.item() for l in losses) / len(losses)
            print(f"Step {step}: loss = {step_loss:.6f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## 约束说明

本示例按层均分的方式切分模型，要求模型层数能被卡数整除，否则代码会在启动时断言失败；如需各stage层数不同，可在手动切分时按需分配。
