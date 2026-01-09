# 通过pg\_options配置HCCL通信域参数

## 简介

本特性可以针对不同的通信域配置不同的HCCL配置。通过pg\_options添加hccl\_config配置，将HCCL配置参数从Python层通过Ascend Extension for PyTorch传递到HCCL供使用。

目前支持的通信域参数配置如下：

-   hccl\_buffer\_size
-   group\_name
-   qos\_service\_level、qos\_traffic\_class
-   hccl\_op\_expansion\_mode

## 使用场景

在模型脚本中按通信域粒度配置HCCL参数。

## 使用指导

>  [!NOTE]  
> 如果同时设置环境变量和pg\_options，参数取值以代码中的pg\_options方式优先。

支持配置HCCL通信域参数：

-   hccl\_buffer\_size：设置通信域的hccl\_buffer\_size大小，默认值为环境变量**HCCL\_BUFFSIZE**的取值，若环境变量HCCL\_BUFFSIZE未设置则该参数默认值为200。环境变量**HCCL\_BUFFSIZE**的详情请参见《CANN 环境变量参考》中的“[HCCL\_BUFFSIZE](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0080.html)”章节。
-   group\_name：设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。
-   qos\_service\_level、qos\_traffic\_class：设置RDMA网卡的service level和traffic class。
    -   qos\_service\_level：该参数取值范围0\~7。默认值为0xffffffff，此时HCCL会读取环境变量**HCCL\_RDMA\_SL**的取值。环境变量**HCCL\_RDMA\_SL**的详情请参见《CANN 环境变量参考》中的“[HCCL\_RDMA\_SL](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0090.html)”章节。
    -   qos\_traffic\_class：该参数取值范围0\~255。默认值为0xffffffff，此时HCCL会读取环境变量**HCCL\_RDMA\_TC**的取值。环境变量**HCCL\_RDMA\_TC**的详情请参见《CANN 环境变量参考》中的“[HCCL\_RDMA\_TC](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0089.html)”章节。

-   hccl\_op\_expansion\_mode：设置通信算法的编排展开位置。

    -   0：默认值，代表通信算法的编排展开位置，和**HCCL\_OP\_EXPANSION\_MODE**环境变量的取值保持一致**。**
    -   1：代表通信算法的编排展开位置为Host侧CPU。
    -   2：代表通信算法的编排展开位置在device侧的AI CPU计算单元。
    -   3：代表通信算法的编排展开位置在device侧的AI Vector Core计算单元。

    环境变量**HCCL\_OP\_EXPANSION\_MODE**的详情请参见《CANN 环境变量参考》中的“[HCCL\_OP\_EXPANSION\_MODE](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0096.html)”章节。

## 使用样例

配置hccl\_buffer\_size示例：

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {"hccl_buffer_size": 200}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

配置group\_name示例：

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {"group_name": "group0"}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

配置qos\_service\_level、qos\_traffic\_class示例：

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config = {"qos_service_level":7, "qos_traffic_class":224}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

配置hccl\_op\_expansion\_mode示例：

```Python
options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
options.hccl_config ={"hccl_op_expansion_mode":3}
torch.distributed.init_process_group(backend="hccl", pg_options=options)
```

## 约束说明

无

