# Overview

This document describes the core features supported by TorchNPU in distributed training scenarios, covering key capabilities such as parallelism strategies, sharding primitives, communication strategies, distributed startup and fault tolerance, and explains how each feature is used on the NPU and the differences from native PyTorch.

## Parallelism Strategies

TorchNPU fully inherits PyTorch's parallelism strategies, and these high-level APIs can be directly combined into existing models. The following parallelism strategies are all available in the TorchNPU environment, with the backend uniformly switched to HCCL.

- **DistributedDataParallel (DDP)**: A distributed data parallelism strategy in which each process holds a complete replica of the model, each process processes different data shards, and gradients are synchronized through AllReduce. TorchNPU provides DDP capabilities consistent with native PyTorch, and the APIs are used in the same way as native ones. You can first read the [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) to understand the general design, and then refer to the [DDP Parallelism Strategy](parallelism_strategy/ddp_introduction.md) to learn the specific operations and precautions in the NPU environment.

- **FullyShardedDataParallel (FSDP/FSDP2)**: A fully sharded data parallelism strategy that shards model parameters, gradients, and optimizer states across multiple devices to reduce the memory requirement of a single card. TorchNPU provides FSDP capabilities consistent with native PyTorch, and the APIs are used in the same way as native ones. You can first read the [PyTorch FSDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) to understand the general design, and then refer to the [FSDP Sharded Parallelism Strategy](parallelism_strategy/fsdp_introduction.md) to learn the specific operations and precautions in the NPU environment.

- **Tensor Parallel (TP)**: A tensor parallelism strategy that splits the weights of a single operator (such as Linear and Attention) along a specific dimension across multiple devices for parallel computation. TorchNPU provides TP capabilities consistent with native PyTorch. You can read the [PyTorch TP Tutorial](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) to understand the general design.

- **Pipeline Parallel (PP)**: A pipeline parallelism strategy that splits the model by layers into multiple stages, assigns each stage to a different device, and executes micro-batches in a pipeline. TorchNPU provides PP capabilities consistent with native PyTorch. You can read the [PyTorch PP Tutorial](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html) to understand the general design.

## Sharding Primitives

`DTensor` and `DeviceMesh` are the basic components for building parallelism strategies. They are used to express the sharding and replication relationships of tensors on an N-dimensional process group.

- **DTensor**: Represents a distributed tensor that is sharded or replicated, recording how data is distributed across multiple cards (whole-card replication `Replicate` or dimension-wise splitting `Shard`). Developers can treat it as an ordinary `Tensor`, and cross-card data movement is automatically completed by the framework when needed. TorchNPU provides DTensor capabilities consistent with native PyTorch, and the APIs are used in the same way as native ones. You can read the [PyTorch DTensor Tutorial](https://docs.pytorch.org/docs/2.14/distributed.tensor.html) to learn more.

- **DeviceMesh**: An abstraction of a distributed communicator that represents the accelerator cluster topology as a multidimensional array and manages the corresponding ProcessGroup instances behind it. It supports multidimensional parallelism combinations (such as 3D parallelism of TP+PP+DP) and provides a unified device view and communication interface for upper layers. For TorchNPU, you only need to set `device_type="npu"`; the rest is consistent with native PyTorch. You can read the [PyTorch DeviceMesh Tutorial](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html) to learn more.

## Distributed Startup and Fault Tolerance

In distributed training, the launcher is responsible for coordinating the initialization and fault tolerance of multi-machine, multi-card tasks. The following introduces the standard launcher `torchrun` in the PyTorch community, as well as the enhanced tools and stability mechanisms provided on TorchNPU for large-scale clusters.

- **torchrun**: The native PyTorch launcher. TorchNPU supports the same usage as native PyTorch. You can read [PyTorch torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html) to learn the specific usage and parameter configuration.

- **torch_npu_run**: The recommended launcher of TorchNPU. It is an improved version of `torchrun` for large clusters on NPU chips, supporting hierarchical link setup, which greatly improves the startup speed of large-scale clusters. It is recommended for use in large-scale scenarios. You can read [torch_npu_run](./startup_and_fault_tolerance/torch_npu_run.md) to learn the specific usage and parameter configuration.

- **WatchDog Mechanism**: A collective communication monitoring mechanism provided by TorchNPU. It quickly detects and reports communication errors without affecting training performance, significantly shortening fault detection time. You can read [WatchDog](../fault_diagnosis/watchdog.md) to learn the specific usage and parameter configuration.

- **Distributed Checkpoint**: Supports saving and restoring models during distributed training. TorchNPU provides capabilities consistent with PyTorch, and the APIs are used in the same way as native ones. You can read [PyTorch Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) to learn how to use it.

## Communication Strategies

The PyTorch distributed communication layer (C10D) provides collective communication APIs (such as all_reduce and all_gather) and point-to-point communication APIs (such as send and isend). These APIs provide the same semantics through the HCCL backend in the TorchNPU environment.

- **HCCL Communication Backend**: The collective communication library for TorchNPU distributed training. Its interface semantics are consistent with native NCCL, and it supports operations such as AllReduce, Broadcast, AllGather, and ReduceScatter. You can first read the [NCCL Communication Description](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html) to understand the native communication interfaces, and then read the [HCCL Collective Communication Library](https://gitcode.com/cann/hccl/blob/9.2.0/docs/en/user_guide/hccl_intro.md) to learn the specific operations and precautions in the NPU environment.

- **Collective Communication APIs**: Use AllReduce, Broadcast, AllGather, ReduceScatter, and others through the `torch.distributed` interface with the communication backend set to `backend="hccl"`. The usage is consistent with native PyTorch. You can read the native [PyTorch Collective Communication](https://pytorch.org/docs/stable/distributed.html) to master collective communication.

- **Point-to-Point Communication APIs**: Use send, isend, recv, irecv, and others through the `torch.distributed` interface with the communication backend set to `backend="hccl"`. The usage is consistent with native PyTorch. You can read the native [PyTorch P2P Communication](https://pytorch.org/docs/stable/distributed.html#torch.distributed.send) to master P2P communication.

- **Ranktable Link Setup**: A cluster link setup mechanism provided by TorchNPU. It specifies a JSON-format rank mapping file through the `RANK_TABLE_FILE` environment variable, presetting the IP and device information of each rank and skipping the default negotiation process to establish links directly. You can read [ranktable Link Setup](./communication_strategy/ranktable_link_setup.md) to learn the technical details and precautions.

- **Scalable RootInfo Hierarchical Link Setup**: An optional extension provided by TorchNPU on the default RootInfo negotiation path. A large-scale communicator can be divided into multiple consecutive and balanced groups, with multiple roots sharing the RootInfo negotiation work, reducing the link setup pressure on a single root. You can read [Scalable RootInfo Hierarchical Link Setup](./communication_strategy/scalable_rootinfo_link_setup.md) to learn the configuration method and usage constraints.

## Infrastructure and Tools

- **DistributedSampler**: Ensures that each process loads a different data subset during distributed training. TorchNPU provides usage consistent with PyTorch. You can read [PyTorch DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) to learn the basic usage.

- **Distributed Environment Initialization**: TorchNPU provides usage consistent with PyTorch. You only need to change the backend to `hccl` and call `dist.init_process_group(backend="hccl")` for initialization. You can read
[PyTorch init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) to learn the basic usage.

## Parameter Configuration

**Configuring HCCL Communicator Parameters via pg_options**: TorchNPU supports configuring HCCL parameters for different communicators through `pg_options`, such as `hccl_buffer_size` and `group_name`. Compared with the process-level unified settings of environment variables, this feature enables differentiated tuning by communicator, providing more fine-grained control for optimizing communication efficiency and memory usage. You can read [Configuring HCCL Communicator Parameters via pg_options](./parameter_configuration/setting_HCCL_communicator_parameter.md) to learn the supported parameters and usage examples.

## How to Choose a Parallelism Strategy

When deciding which parallelism strategy to use, refer to the following general guidelines:

1. **The model fits on a single card but requires multi-card acceleration**: Use **DDP**. Start multiple processes with `torchrun`. If data loading becomes a bottleneck, use `DistributedSampler` to ensure that data does not overlap across processes, effectively avoiding compute waste and maximizing the parallelism efficiency of data loading and computation.

2. **The model cannot fit on a single card**: Use **FSDP**. Shard parameters, gradients, and optimizer states across multiple devices.

3. **FSDP reaches its scaling bottleneck**: Add **TP** or **PP** to form multidimensional parallelism (2D/3D parallelism). Use `DeviceMesh` to organize multidimensional communicators and `DTensor` to express tensor sharding relationships.

4. **Large-scale multi-node training**: Use **torch_npu_run** instead of `torchrun`.
