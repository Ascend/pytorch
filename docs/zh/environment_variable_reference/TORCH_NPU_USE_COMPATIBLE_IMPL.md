# TORCH_NPU_USE_COMPATIBLE_IMPL

## 功能描述

该环境变量用于控制是否开启一致性配置，开启时，算子API的实现与PyTorch原生社区完全对齐。该环境变量仅用于切换API所调用的底层算子。 

-   未设置或者设置为“0”时，表示关闭一致性配置。此环境变量默认值为`None`，默认状态为关闭状态。
-   设置为“1”时，表示开启一致性配置。

## 配置示例

``` bash
export TORCH_NPU_USE_COMPATIBLE_IMPL=1
```

## 使用约束

-   此环境变量需要在`import torch`之前配置才能生效。

-   目前仅支持`torch.nn.functional.gelu`、`torch.matmul`。

-   配置`TORCH_NPU_USE_COMPATIBLE_IMPL`会影响[torch_npu.npu.use_compatible_impl(is_enable)](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/torch_npu-npu-use_compatible_impl.md)
的状态。配置`TORCH_NPU_USE_COMPATIBLE_IMPL=1`时会自动配置`torch_npu.npu.use_compatible_impl(True)`。

-   此环境变量适用于Ascend Extension for PyTorch 26.0.0及之后版本。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>

