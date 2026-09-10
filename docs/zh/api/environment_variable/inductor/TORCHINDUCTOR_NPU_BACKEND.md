# TORCHINDUCTOR\_NPU\_BACKEND

## 功能描述

通过此环境变量可配置图模式（Inductor）下的编译模式，支持在Triton、DVM、Ascend C等模式之间切换。

- 默认值为“default”，使用默认的Triton模式。
- 配置为“dvm”：使用DVM模式。
- 配置为“ascendc”：使用Ascend C模式。

PyTorch通过`torch._inductor.config`配置后端行为；TorchNPU提供此环境变量选择NPU上的Inductor编译模式。

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

- 此环境变量在首次调用`torch.compile`前设置即可（后端在首次编译时惰性加载，`import torch`/`import torch_npu`之后设置同样生效）。
- Ascend C模式需PyTorch 2.9.0及以上版本支持。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>（仅支持Inductor后端编译器中的Triton模式、DVM模式和Ascend C模式）
