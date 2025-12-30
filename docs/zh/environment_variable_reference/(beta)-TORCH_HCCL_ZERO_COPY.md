# \(beta\) TORCH\_HCCL\_ZERO\_COPY

> [!NOTICE]  
> 此功能尚在实验阶段，请谨慎使用。

## 功能描述

训练或在线推理场景下，可通过此环境变量开启集合通信片内零拷贝功能，减少通信算子在通信过程中片内拷贝次数，提升集合通信效率，降低通信耗时。同时在计算通信并行场景下，降低通信过程中对显存带宽的抢占。

-   0：关闭集合通信零拷贝功能。
-   1：开启集合通信零拷贝功能。

默认值为0。

## 配置示例

```
export TORCH_HCCL_ZERO_COPY=1
```

## 使用约束

-   该环境变量依赖Ascend Extension for PyTorch虚拟内存管理功能，参见[PYTORCH\_NPU\_ALLOC\_CONF](PYTORCH_NPU_ALLOC_CONF.md)，要求配置满足：

    ```
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    ```

-   此环境变量不支持在PyTorch图模式（TorchAir）场景下使用。
-   其他约束请参见《CANN  HCCL API（C）》中“零拷贝功能 \> [使用前必读](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/hcclapiref/hcclcpp_07_0053.html)”章节。

## 支持的型号

<term>Atlas A3 训练系列产品</term>

