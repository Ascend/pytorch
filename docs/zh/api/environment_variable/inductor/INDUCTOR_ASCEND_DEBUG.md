# INDUCTOR\_ASCEND\_DEBUG

## 功能描述

通过此环境变量可控制Triton编译时是否启用调试模式。启用后，Triton编译器会生成包含调试信息的编译元数据，辅助定位编译阶段的问题。

- 默认值为`0`，不启用调试模式。
- 配置为`1`：启用调试模式。

> [!NOTE]
>
> - 此环境变量仅在Triton模式（`TORCHINDUCTOR_NPU_BACKEND="default"`）下生效。
> - 实际生效还需满足`assert_indirect_indexing`为True且非HIP（Heterogeneous-compute Interface for Portability，AMD异构计算接口）设备。
> - 仅影响Triton编译过程的调试信息输出，不影响算子功能。

PyTorch通过`torch._inductor.config.debug`配置调试模式，TorchNPU通过此环境变量提供相关配置。

## 配置示例

```bash
export INDUCTOR_ASCEND_DEBUG=1
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
