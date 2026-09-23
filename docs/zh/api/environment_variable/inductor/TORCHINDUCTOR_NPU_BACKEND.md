# TORCHINDUCTOR\_NPU\_BACKEND

## 功能描述

通过此环境变量可配置图模式（Inductor）下的编译模式，支持在Triton、DVM、Ascend C等模式之间切换。

- 默认值为`default`：使用默认的Triton模式。
- 配置为`dvm`：使用DVM模式。
- 配置为`ascendc`：使用Ascend C模式。

> [!NOTE]
>
> 此环境变量在首次调用`torch.compile`前设置即可（后端在首次编译时惰性加载，`import torch`/`import torch_npu`之后设置同样生效）。

PyTorch通过[`torch._inductor.config`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)配置后端行为；TorchNPU提供此环境变量选择NPU上的Inductor编译模式。

## 配置示例

使用默认的Triton模式：

```bash
export TORCHINDUCTOR_NPU_BACKEND="default"
```

使用DVM模式：

```bash
export TORCHINDUCTOR_NPU_BACKEND="dvm"
```

使用Ascend C模式：

```bash
export TORCHINDUCTOR_NPU_BACKEND="ascendc"
```

## 使用约束

- Ascend C模式需PyTorch 2.9.0及以上版本支持。
- 仅支持Inductor后端编译器中的Triton模式、DVM模式和Ascend C模式。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id4 -->
