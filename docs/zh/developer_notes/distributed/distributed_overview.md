# TorchNPU 分布式训练特性总览

本文档介绍TorchNPU在分布式训练场景下支持的核心特性，涵盖并行策略、分片原语、通信策略、分布式启动与容错等关键能力，并说明各项特性在NPU上的使用方式以及和原生PyTorch的异同。

---

## 并行策略

TorchNPU 完整继承了 PyTorch 的并行策略，这些高层 API 可直接组合到现有模型中。以下并行策略在 TorchNPU 环境下均可使用，后端统一切换为 HCCL。

- **DistributedDataParallel（DDP）** — 数据并行策略，模型在每个进程上保存完整副本，每个进程处理不同数据分片，梯度通过 AllReduce 同步。TorchNPU提供与原生PyTorch一致的DDP能力，使用方式与原生类似。可以先阅读 [PyTorch DDP 教程](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) 了解通用设计思路，再结合[TorchNPU DDP 使用说明](ddp_introduction.md)掌握NPU环境下的具体操作与注意事项。
  
- **FullyShardedDataParallel（FSDP / FSDP2）** — 全分片数据并行策略，将模型参数、梯度、优化器状态分片到多个设备上，降低单卡显存需求。TorchNPU 提供与原生PyTorch一致的FSDP能力，使用方式与原生类似。可以先阅读[PyTorch FSDP 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)了解通用设计思路，再结合[TorchNPU FSDP 使用说明](fsdp_introduction.md)掌握NPU环境下的具体操作与注意事项。

- **Tensor Parallel（TP）** — 张量并行策略，将单个算子（如 Linear、Attention）的权重沿特定维度切分到多个设备上并行计算。TorchNPU 提供与原生PyTorch一致的TP能力，可以阅读[PyTorch TP 教程](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)了解通用设计思路。
  
- **Pipeline Parallel（PP）** — 流水线并行策略，将模型按层切分为多个 stage，每个 stage 分配到不同设备，微批次流水线执行。TorchNPU 提供与原生PyTorch一致的PP能力，可以阅读[PyTorch PP 教程](https://docs.pytorch.org/docs/2.13/distributed.pipelining.html)了解通用设计思路。

---

## 分片原语

`DTensor` 和 `DeviceMesh` 是构建并行策略的基础组件，用于在 N 维进程组上表达张量的分片和复制关系。

- **DTensor** — 表示分片或复制的分布式张量，其布局信息可被分布式运行时用于指导算子执行时的重分片通信。TorchNPU 提供与原生PyTorch一致的DTensor能力，使用方式与原生类似。可以阅读[PyTorch DTensor 教程](https://docs.pytorch.org/docs/2.13/distributed.tensor.html)进行了解。

- **DeviceMesh** — 一种分布式通信域的抽象，将加速器集群拓扑表示为一个多维数组，并管理其背后对应的 ProcessGroup 实例。它支撑多维并行组合（如 TP+PP+DP 的 3D 并行），为上层提供统一的设备视图与通信接口。TorchNPU 只需设置 `device_type="npu"`，其余与原生一致。可以阅读[PyTorch DeviceMesh 教程](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)进行了解。

---

## 通信策略

PyTorch 分布式通信层（C10D）提供了集合通信 API（如 all_reduce、all_gather）和点对点通信 API（如 send、isend），这些API在 TorchNPU 环境下通过 HCCL 后端提供相同的语义。

- **HCCL 通信后端** — TorchNPU 分布式训练的集合通信库，接口语义与原生 NCCL 一致，支持 AllReduce、Broadcast、AllGather、ReduceScatter 等操作。可以先阅读[NCCL 通信说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)来了解原生通信接口，再阅读[HCCL 通信说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/commlib/hcclug/docs/zh/user_guide/hccl_intro.md)来掌握NPU环境下的具体操作与注意事项。

- **集合通信 API** — 通过 `torch.distributed` 接口在通信后端 `backend="hccl"` 下使用 AllReduce、Broadcast、AllGather、ReduceScatter 等，使用方式与原生一致。可以阅读原生[PyTorch 集合通信说明](https://pytorch.org/docs/stable/distributed.html)掌握集合通信。

- **点对点通信 API** — 通过 `torch.distributed` 接口在通信后端 `backend="hccl"` 下使用 send、isend、recv、irecv 等，使用方式与原生一致。可以阅读原生[PyTorch P2P 通信说明](https://pytorch.org/docs/stable/distributed.html#torch.distributed.send)掌握P2P通信。

- **Ranktable 建链** — TorchNPU 提供的一种特殊机制。通过 `RANK_TABLE_FILE` 环境变量指定 JSON 格式的 rank 映射文件，预置各 rank 的 IP 和设备信息，跳过默认协商流程直接建链。可以阅读[ranktable 建链说明](https://www.hiascend.com/document/detail/zh/Pytorch/master/devguide/dist/docs/zh/developer_notes/distributed/ranktable_link_setup.md)来掌握技术细节与注意事项。

---

## 分布式启动与容错

在分布式训练中，启动器负责协调多机多卡任务的初始化与容错。以下介绍了 PyTorch 社区的标准启动器`torchrun`，以及在TorchNPU上针对大规模集群提供的增强工具与稳定性机制。

- **torchrun** — PyTorch 原生启动器，TorchNPU支持与原生相同的使用方法，可以阅读[PyTorch torchrun 说明](https://docs.pytorch.org/docs/2.13/elastic/run.html)了解具体使用方式与参数配置。

- **torch_npu_run** — TorchNPU 推荐启动器，是`torchrun` 在NPU芯片上的大集群改进版，支持分层建链，大幅提升了大规模集群的启动速度。推荐在大规模场景下优先使用。可以阅读[torch_npu_run 说明](https://www.hiascend.com/document/detail/zh/Pytorch/master/devguide/dist/docs/zh/developer_notes/distributed/torch_npu_run.md)了解具体使用方式与参数配置。

- **WatchDog 机制** — TorchNPU 提供的一种特殊机制。在不影响训练性能的前提下快速检测并报告通信错误，显著缩短故障检测时间。

- **分布式 Checkpoint** — 支持保存和恢复分布式训练状态。TorchNPU提供与PyTorch 一致的能力，使用方法与原生类似。可以阅读[PyTorch 分布式 checkpoint 说明](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)了解使用方法。

---

## 基础设施与工具

- **DistributedSampler** — 确保分布式训练过程中每个进程加载不同的数据子集。TorchNPU 提供与 PyTorch 一致的用法。可以阅读[PyTorch DistributedSampler 说明](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)了解基础用法。

- **分布式环境初始化** — TorchNPU 提供与 PyTorch 一致的使用方法，只需要更换后端为`hccl`即可调用 `dist.init_process_group(backend="hccl")`进行初始化。可以阅读
[PyTorch init_process_group 说明](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)了解基础用法。

---

## 如何选择并行策略

在决定使用哪种并行策略时，可以参考以下通用指南：

1. **模型能单卡容纳，希望多卡加速** — 使用 **DDP**。配合 `torchrun` 启动多进程。如果数据加载成为瓶颈，使用 `DistributedSampler` 确保各进程数据不重叠，从而有效避免算力浪费，最大化数据加载与计算的并行效率。

2. **模型无法单卡容纳** — 使用 **FSDP**。将参数、梯度、优化器状态分片到多个设备上。

3. **FSDP 达到扩展瓶颈** — 叠加 **TP** 或 **PP**，组成多维并行（2D/3D 并行）。使用 `DeviceMesh` 组织多维通信域，`DTensor` 表达张量分片关系。

4. **大规模多节点训练** — 使用 **torch_npu_run** 替代 `torchrun`。
