# torch_npu_run

## Introduction

`torch_npu_run` is an improved version of `torchrun` for large-scale cluster scenarios, improving cluster link setup performance.

`torch_npu_run` has the following improvements over `torchrun`:

1. `torch_npu_run` uses epoll to implement a multi-threaded TCP server. It can efficiently handle a large number of concurrent connections and quickly respond to client requests, thereby significantly improving the overall performance and throughput of the system.
2. `torch_npu_run` supports tiered link setup. By setting `enable_tiered_parallel_tcpstore` to true, you can enable tiered link setup.

    In distributed training, each node typically starts one `torchrun` process, also known as an agent process. The agent manages the startup and termination of multiple training processes on the node, and the training processes are also known as workers. Native `torchrun` requires all agents and workers to establish TCP connections with the agent on node 0 (node0), as shown in [Figure 1](#figure1). In the `torchrun` mode, the link setup time increases linearly as the number of training processes grows, leading to a performance bottleneck.

    **Figure 1**  Original torchrun link setup method  <a id="figure1"></a>  
    
    ![figure1](../figures/link_setup_in_torchrun_mode.png)

    `torch_npu_run` introduces the TCPStore tiered architecture scheme based on `torchrun`. On each node, the agent starts a new role, proxy, to manage the communication of workers. Workers on the node establish UnixSocket connections with the proxy, and all proxies establish TCP connections with the proxy on node0. This implements tiered communication, breaks the linear bottleneck of link setup time, and reduces the time complexity of link setup from $O(n)$ to $O(\sqrt{n})$, as shown in [Figure 2](#figure2).

    **Figure 2**  torch_npu_run link setup method   <a id="figure2"></a>  
    ![figure2](../figures/link_setup_in_torch_npu_run_mode.png)

## Use Scenarios

When launching distributed training tasks, you are advised to use this feature.

## Usage Guide

`torch_npu_run` is used in a similar way to `torchrun`. Some optional configuration parameters of `torch_npu_run` are as follows:

- `nnodes`: The number of nodes, or a range of node counts, in the format `<min_nodes>:<max_nodes>`.
- `nproc_per_node`: The number of worker processes per node. Supported values include `auto`, `cpu`, `gpu`, or an integer.
- `node_rank`: The rank of the node in multi-node distributed training.
- `rdzv_backend`: The backend mechanism for establishing collective communication connections.
- `rdzv_endpoint`: The backend service address used for rendezvous, in the format `<hostname>:<port>`.
- `rdzv_id`: A user-defined ID that uniquely identifies the worker group for a job. Each node uses this ID to join a specific worker group.
- `standalone`: Indicates running a distributed training job on a single machine, applicable to single-node multi-process jobs.
- `master_addr`: The network address of the master node (rank 0), used only for static rendezvous.
- `master_port`: The port of the master node (rank 0), used only for static rendezvous.
- `local_addr`: The IP address of the current node.
- `enable_tiered_parallel_tcpstore`: Whether to enable tiered link setup to further improve link setup performance, that is, whether to establish links within nodes and between nodes separately. You are advised to use it in large-scale cluster scenarios. Supported values are `true` and `false`, and the default value is `false` (disabled).

## Usage Examples

Example of launching a single-node 8-card training task:

```shell
export MASTER_IP_ADDR=**  # Replace ** with the IP address of node_rank0 
export MASTER_PORT=**  # Replace ** with an available TCP port number 
torch_npu_run --rdzv_backend=parallel --master_addr=$MASTER_IP_ADDR --master_port=$MASTER_PORT --nnodes=1 --nproc_per_node=8 ddp_test.py
```

Example of launching a two-node 16-card training task:

- Tiered link setup disabled

    ```shell
    # First machine 
    export MASTER_IP_ADDR=**  # Replace ** with the IP address of node_rank0 
    export MASTER_PORT=**  # Replace ** with an available TCP port number 
    torch_npu_run --rdzv_backend=parallel --master_addr=$MASTER_IP_ADDR --master_port=$MASTER_PORT --nnodes=2 --node_rank 0 --nproc_per_node=8 ddp_test.py  
    
    # Second machine 
    export MASTER_IP_ADDR=** # Replace ** with the IP address of node_rank0 
    export MASTER_PORT=** # Replace ** with an available TCP port number 
    torch_npu_run --rdzv_backend=parallel --master_addr=$MASTER_IP_ADDR --master_port=$MASTER_PORT --nnodes=2 --node_rank 1 --nproc_per_node=8 ddp_test.py
    ```

- Tiered link setup enabled

    ```shell
    # First machine 
    export MASTER_IP_ADDR=**  # Replace ** with the IP address of node_rank0 
    export MASTER_PORT=**  # Replace ** with an available TCP port number 
    torch_npu_run --rdzv_backend=parallel --master_addr=$MASTER_IP_ADDR --master_port=$MASTER_PORT --nnodes=2 --node_rank 0 --nproc_per_node=8 --enable_tiered_parallel_tcpstore=true ddp_test.py
      
    # Second machine 
    export MASTER_IP_ADDR=** # Replace ** with the IP address of node_rank0 
    export MASTER_PORT=** # Replace ** with an available TCP port number 
    torch_npu_run --rdzv_backend=parallel --master_addr=$MASTER_IP_ADDR --master_port=$MASTER_PORT --nnodes=2 --node_rank 1 --nproc_per_node=8 --enable_tiered_parallel_tcpstore=true ddp_test.py
    ```

> [!NOTE]  
>
> `ddp_test.py` is the model training script and is only an example. You can modify it according to the actual script name.

## Constraints

None
