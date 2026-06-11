# CostModel特性介绍

## 概述

面向A5+PyTorch-v2.9.0，Inductor-Ascend在Triton后端precompile阶段提供CostModel预筛选能力。该能力通过Triton-Ascend CostModel后端对候选config进行静态性能预测，并将预测耗时较短的config优先送入后续编译和实测profiling流程，从而降低候选config数量较多时的首次编译开销。

CostModel适用于自动Tiling产生大量候选config的融合算子。它不改变算子计算语义，也不直接决定最终运行时使用的kernel。最终可用config仍会经过precompile校验，后续autotune流程会继续基于实际编译和profiling结果完成选择。

## 关键概念解释

### Config

Config是Triton kernel的候选编译配置，通常包含block大小、stage数量、warp数量等影响代码生成和运行性能的参数。Inductor-Ascend会为同一个融合算子生成多个候选config，并在precompile和autotune阶段选择可用且性能较优的config。

### TTIR

TTIR是Triton kernel在编译过程中的中间表示。CostModel基于每个config生成对应TTIR，并结合运行时标量参数进行静态仿真，预测该config的耗时。

### Arg Bindings

Arg Bindings用于向CostModel传入TTIR中标量参数的实际值，例如`arg3=98432,pid_x=0`。当TTIR中包含`tt.get_num_programs x`时，还会补充`num_programs_x`。这些信息用于提升CostModel对动态shape和运行时参数相关表达式的解析能力。

## 作用

- 减少precompile和autotune需要处理的config数量，降低首次编译耗时。
- 按CostModel预测耗时对候选config排序，使更可能高性能的config优先进入后续流程。
- 在CostModel筛选出的config全部编译失败时，使用被CostModel筛掉的config进行兜底编译，避免因为预筛选过窄导致无可用config。
- 通过Triton-Ascend CostModel缓存减少重复TTIR预测开销。

## 原理

启用CostModel后，Inductor-Ascend的Triton后端会在precompile前执行以下流程：

1. 根据当前kernel的候选config逐个生成TTIR。
2. 从运行时输入和grid信息中解析CostModel需要的Arg Bindings。
3. 调用Triton-Ascend提供的`costmodel_bench`接口，获得`config -> 预测耗时`的结果。
4. 过滤掉预测耗时为无穷大的config，并按预测耗时从小到大排序。
5. 根据`INDUCTOR_ASCEND_COSTMODEL_RATIO`保留排序靠前的config，替换原始config集合进入precompile。
6. 将未被选中的config记录为兜底config。如果CostModel选中的config没有产生有效编译结果，则使用兜底config再次执行precompile。

CostModel预测过程由Triton-Ascend运行时完成。

## 使用方法

通过环境变量开启CostModel：

```shell
export INDUCTOR_ASCEND_ENABLE_COSTMODEL=1
```

可按需调整预筛选比例：

```shell
export INDUCTOR_ASCEND_COSTMODEL_RATIO=0.25
```

配置完成后，正常使用`torch.compile`即可进入CostModel链路：

```python
import torch
import torch_npu


@torch.compile(backend="inductor")
def fn(x, y):
    return x + y


out = fn(torch.randn(1024, device="npu"), torch.randn(1024, device="npu"))
```

## 使用约束

- CostModel仅在`INDUCTOR_ASCEND_ENABLE_COSTMODEL=1`、候选config数量大于1且`INDUCTOR_ASCEND_COSTMODEL_RATIO`位于(0, 1)时执行预筛选。
- CostModel依赖Triton-Ascend包中的后端能力。如果后端不可用或调用异常，会跳过CostModel预筛选，继续使用原始config集合。
- CostModel预测结果是静态估计值，不等价于真实profiling耗时，最终性能选择仍以实际编译和profiling流程为准。
- Arg Bindings依赖运行时输入值和grid信息。无法解析的标量参数可能导致对应config预测失败，该config会被视为无有效预测结果。
- CostModel缓存的命中条件包含TTIR内容和传入CostModel的参数，kernel或config变化后会生成新的缓存项。
