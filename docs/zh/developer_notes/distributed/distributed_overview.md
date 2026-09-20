# 概述

本文档介绍TorchNPU在分布式训练场景下支持的核心特性，涵盖并行策略、分片原语、通信策略、分布式启动与容错等关键能力，并说明各项特性在NPU上的使用方式以及和原生PyTorch的异同。

## 并行策略

TorchNPU完整继承了PyTorch的并行策略，这些高层API可直接组合到现有模型中。以下并行策略在TorchNPU环境下均可使用，后端统一切换为HCCL。

- **DistributedDataParallel（DDP）**：分布式数据并行策略，模型在每个进程上保存完整副本，每个进程处理不同数据分片，梯度通过AllReduce同步。TorchNPU提供与原生PyTorch一致的DDP能力，API使用方式与原生一致。可以先阅读[PyTorch DDP教程](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)了解通用设计思路，再结合[DDP并行策略](parallelism_strategy/ddp_introduction.md)掌握NPU环境下的具体操作与注意事项。

- **FullyShardedDataParallel（FSDP/FSDP2）**：全分片数据并行策略，将模型参数、梯度、优化器状态分片到多个设备上，降低单卡显存需求。TorchNPU提供与原生PyTorch一致的FSDP能力，API使用方式与原生一致。可以先阅读[PyTorch FSDP教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)了解通用设计思路，再结合[FSDP分片并行策略](parallelism_strategy/fsdp_introduction.md)掌握NPU环境下的具体操作与注意事项。

- **Tensor Parallel（TP）**：张量并行策略，将单个算子（如Linear、Attention）的权重沿特定维度切分到多个设备上并行计算。TorchNPU提供与原生PyTorch一致的TP能力，可以阅读[PyTorch TP教程](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)了解通用设计思路。

- **Pipeline Parallel（PP）**：流水线并行策略，将模型按层切分为多个stage，每个stage分配到不同设备，微批次流水线执行。TorchNPU提供与原生PyTorch一致的PP能力，可以阅读[PyTorch PP教程](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html)了解通用设计思路。

## 分片原语

`DTensor`和`DeviceMesh`是构建并行策略的基础组件，用于在N维进程组上表达张量的分片和复制关系。

- **DTensor**：表示分片或复制的分布式张量，记录数据在多卡上的分布方式（整卡复制`Replicate`或按维度切分`Shard`）。开发者可将其视为普通`Tensor`进行操作，跨卡的数据搬运由框架在需要时自动完成。TorchNPU提供与原生PyTorch一致的DTensor能力，API使用方式与原生一致。可以阅读[PyTorch DTensor教程](https://docs.pytorch.org/docs/2.14/distributed.tensor.html)进行了解。

- **DeviceMesh**：一种分布式通信域的抽象，将加速器集群拓扑表示为一个多维数组，并管理其背后对应的ProcessGroup实例。它支撑多维并行组合（如TP+PP+DP的3D并行），为上层提供统一的设备视图与通信接口。TorchNPU只需设置`device_type="npu"`，其余与原生一致。可以阅读[PyTorch DeviceMesh教程](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)进行了解。

## 分布式启动与容错

在分布式训练中，启动器负责协调多机多卡任务的初始化与容错。以下介绍了PyTorch社区的标准启动器`torchrun`，以及在TorchNPU上针对大规模集群提供的增强工具与稳定性机制。

- **torchrun**：PyTorch原生启动器，TorchNPU支持与原生相同的使用方法，可以阅读[PyTorch torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html)了解具体使用方式与参数配置。

- **torch_npu_run**：TorchNPU推荐启动器，是`torchrun`在NPU芯片上的大集群改进版，支持分层建链，大幅提升了大规模集群的启动速度。推荐在大规模场景下优先使用。可以阅读[torch_npu_run](./startup_and_fault_tolerance/torch_npu_run.md)了解具体使用方式与参数配置。

- **WatchDog机制**：TorchNPU提供的一种集合通信监控机制。在不影响训练性能的前提下快速检测并报告通信错误，显著缩短故障检测时间。可以阅读[WatchDog](../fault_diagnosis/watchdog.md)了解具体使用方式与参数配置。

- **分布式Checkpoint**：支持在分布式训练中保存和恢复模型。TorchNPU提供与PyTorch一致的能力，API使用方式与原生一致。可以阅读[PyTorch 分布式checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)了解使用方法。

## 通信策略

PyTorch分布式通信层（C10D）提供了集合通信API（如all_reduce、all_gather）和点对点通信API（如send、isend），这些API在TorchNPU环境下通过HCCL后端提供相同的语义。

- **HCCL通信后端**：TorchNPU分布式训练的集合通信库，接口语义与原生NCCL一致，支持AllReduce、Broadcast、AllGather、ReduceScatter等操作。可以先阅读[NCCL 通信说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)来了解原生通信接口，再阅读[HCCL集合通信库](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/commlib/hcclug/docs/zh/user_guide/hccl_intro.md)来掌握NPU环境下的具体操作与注意事项。

- **集合通信API**：通过`torch.distributed`接口在通信后端`backend="hccl"`下使用AllReduce、Broadcast、AllGather、ReduceScatter等，使用方式与原生一致。可以阅读原生[PyTorch 集合通信](https://pytorch.org/docs/stable/distributed.html)掌握集合通信。

- **点对点通信API**：通过`torch.distributed`接口在通信后端`backend="hccl"`下使用send、isend、recv、irecv等，使用方式与原生一致。可以阅读原生[PyTorch P2P通信](https://pytorch.org/docs/stable/distributed.html#torch.distributed.send)掌握P2P通信。

- **Ranktable建链**：TorchNPU提供的一种集群建链机制。通过`RANK_TABLE_FILE`环境变量指定JSON格式的rank映射文件，预置各rank的IP和设备信息，跳过默认协商流程直接建链。可以阅读[ranktable 建链](./communication_strategy/ranktable_link_setup.md)来掌握技术细节与注意事项。

- **Scalable RootInfo分级建链**：TorchNPU在默认RootInfo协商路径上提供的可选扩展。大规模通信域可以划分为多个连续且均衡的group，由多个root分别承担RootInfo协商工作，降低单root的建链压力。可以阅读[Scalable RootInfo分级建链](./communication_strategy/scalable_rootinfo_link_setup.md)了解配置方法与使用约束。

## 基础设施与工具

- **DistributedSampler**：确保分布式训练过程中每个进程加载不同的数据子集。TorchNPU提供与PyTorch一致的用法。可以阅读[PyTorch DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)了解基础用法。

- **分布式环境初始化**：TorchNPU提供与PyTorch一致的使用方法，只需要更换后端为`hccl`即可调用`dist.init_process_group(backend="hccl")`进行初始化。可以阅读
[PyTorch init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)了解基础用法。

## 参数配置

**通过pg_options配置HCCL通信域参数**：TorchNPU支持通过`pg_options`为不同的通信域配置HCCL参数（如`hccl_buffer_size`、`group_name`等），相比于环境变量的进程级统一设置，该特性可按照通信域进行差异化调优，为通信效率和显存占用的优化提供更细粒度的控制能力。可以阅读[通过pg_options配置HCCL通信域参数](./parameter_configuration/setting_HCCL_communicator_parameter.md)了解支持的参数与使用样例。

## 如何选择并行策略

在决定使用哪种并行策略时，可以参考以下通用指南：

1. **模型能单卡容纳，需要多卡加速**：使用**DDP**。配合`torchrun`启动多进程。如果数据加载成为瓶颈，使用`DistributedSampler`确保各进程数据不重叠，从而有效避免算力浪费，最大化数据加载与计算的并行效率。

2. **模型无法单卡容纳**：使用**FSDP**。将参数、梯度、优化器状态分片到多个设备上。

3. **FSDP达到扩展瓶颈**：叠加**TP**或**PP**，组成多维并行（2D/3D并行）。使用`DeviceMesh`组织多维通信域，`DTensor`表达张量分片关系。

4. **大规模多节点训练**：使用**torch_npu_run**替代`torchrun`。
