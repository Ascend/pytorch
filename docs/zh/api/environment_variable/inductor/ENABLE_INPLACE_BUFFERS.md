# ENABLE\_INPLACE\_BUFFERS

## 功能描述

通过此环境变量可控制是否启用原地缓冲区（inplace buffers）优化。启用后，Inductor允许内核将输入缓冲区复用为输出缓冲区，减少内存分配。

- 默认值为`1`（`true`、`yes`等效），启用原地缓冲区。
- 配置为`0`：禁用原地缓冲区。

> [!NOTE]
>
> - 禁用此选项后，Inductor的`inplace_buffers`配置将被设为`False`，同时会禁用NPU-IR的多缓冲区优化。

PyTorch通过`torch._inductor.config.inplace_buffers`配置项控制原地缓冲区，TorchNPU通过此环境变量提供相关配置。

## 配置示例

禁用原地缓冲区：

```bash
export ENABLE_INPLACE_BUFFERS=0
```

## 使用约束

需在导入`torch_npu`之前设置。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
