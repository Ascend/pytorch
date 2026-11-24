# NPU Inductor Planned Fast Launch 使用指南

本文档介绍Planned Fast Launch功能，说明其边界、前置条件、启用方式、生命周期与回退机制，适用于使用torch_npu并通过torch.compile(backend="inductor") 生成NPU Triton kernel的用户。

## 1. 功能说明

Planned Fast Launch用于降低 `torch.compile` 生成的Python Wrapper在稳态运行时
下发NPU Triton融合kernel的Host侧固定开销。launcher稳定后，Wrapper会创建
C++ `FastLaunchPlan`；后续调用复用计划，只更新stream、grid和参数。

该功能有以下边界：

- 仅覆盖NPU Inductor Python Wrapper生成的Triton kernel callsite。
- 不覆盖eager/ACLNN算子、C++ Wrapper和图分区子图。
- compile-time autotune保留原codegen；grouped dynamic-shape autotuner不附加
  Fast Launch launcher元数据，运行时直接绑定原 `run` 入口。
- PyTorch profiler、FX graph/launch-params dump和准确性检查等状态不进入
  planned path（计划路径），由原路径保留对应语义。
- Triton launcher scope中空的launch `HookChain` 对象本身不阻止planned path；
  注册真实回调后，当前调用临时使用原launcher，回调移除后自动恢复planned path。
- 默认关闭，需要显式启用。

## 2. 前置条件

- 使用 `torch_npu` 2.10系列中包含本功能的构建产物。
- PyTorch、torch_npu、CANN和NPU驱动版本相互匹配。
- Fast Launch加速特性最低支持 **CANN 8.5.0**。
- Fast Launch加速需要运行时CANN提供 `aclrtLaunchKernelWithHostArgs`。
  该接口按需动态加载；缺少接口时不会因它导致 `torch`/`torch_npu` 导入失败。
  即使启用Fast Launch，计划创建也会在任务提交前回退到原Triton launcher，
  并缓存当前launcher不可使用Fast Launch的结果。旧版CANN不保留该加速。
- 模型通过 `torch.compile(..., backend="inductor")` 执行，并实际生成NPU Triton
  kernel。

环境变量在Python进程导入 `torch_npu` 以及执行 `torch.compile` **之前**设置。
运行中修改环境变量不会改写已经生成的Wrapper；需要重启进程并重新编译。

## 3. 启用方式

Linux Shell：

```bash
export TORCHINDUCTOR_NPU_FAST_LAUNCH=1
python run_model.py
```

Python启动脚本也可以在导入PyTorch/torch_npu前设置：

```python
import os

os.environ["TORCHINDUCTOR_NPU_FAST_LAUNCH"] = "1"

import torch
import torch_npu
```

最小使用示例：

```python
# 需先设置TORCHINDUCTOR_NPU_FAST_LAUNCH=1（见第3节启用方式）
# 适用场景：稳态反复调用torch.compile生成的NPU Triton kernel，降低Host侧下发开销
import torch
import torch_npu


def fn(x, y):
    return torch.sin(x + y) * 2


# 输入参数：x、y均为长度4096的一维NPU float32张量
x = torch.randn(4096, device="npu")
y = torch.randn(4096, device="npu")
# backend="inductor"是本功能生效前提：Planned Fast Launch仅覆盖NPU Inductor Python Wrapper生成的Triton kernel
compiled_fn = torch.compile(fn, backend="inductor")

result = compiled_fn(x, y)
# 返回值：长度4096的一维NPU张量，数值与未编译版本fn(x, y)一致
```

无需修改模型代码或显式调用 `_npu_inductor_fast_launch_with_plan`。计划创建和路由
由生成的Wrapper自动完成。初始调用可能包含编译、autotune和计划创建开销；若启用
coordinate-descent tuning或kernel binary保存，也会先完成相应生命周期工作。

## 4. 生命周期、ABI 和回退

生成的 Wrapper 会在每个 Triton 调用点按需创建绑定对象。
若 launcher 尚未稳定，本次先执行 `NPUCachingAutotuner.run`，调用结束后再尝试创建 `FastLaunchPlan`。
若 launcher 已稳定且满足下列条件，可以在本次调用中创建并使用计划。
遇到 profiler、dump 等需要保留原入口的场景时，本次只执行原入口。

创建计划需要满足以下条件：

- 已选定唯一且稳定的launcher，且不是fallback launcher。
- autotune已完成。
- 启用coordinate-descent tuning时，调优已经完成。
- 启用kernel binary保存时，保存已经完成。
- 最终编译ABI、C++ backend和所需资源均受支持。

codegen schema不完整本身不会拒绝planned path；launcher稳定后会使用最终编译ABI补全
参数类型。schema明确冲突、最终ABI或资源不受支持，kernel、stub、grid resolver、
FFTS状态或backend等必要信息缺失，以及plan创建失败等稳定问题，会为当前
launcher安装负缓存并继续使用原launcher。动态grid等单次调用错误不会污染已有
plan。

当前计划backend支持以下ABI参数：

- `torch.Tensor` 参数，提交时按其 `data_ptr` 打包。
- `i32`、`i64`、`u32`、`u64`。
- `f32`、`f64`。
- 按 `int32` 打包的 `bool`。
- 由最终launcher ABI提供、且类型属于上述范围的runtime block参数。

Plan保存kernel name、kernel stub及其owner、参数布局、SIMT/shared memory配置，
以及目标设备需要的FFTS地址。FFTS地址在plan创建时查询一次。Plan不记录或校验
device id；C++ backend对传入stream只检查非空，不核对其所属设备，因此同一个plan
不应跨设备复用。以下信息在每次调用时重新解析并打包：

- 当前非空NPU stream。
- 当前三维grid。
- Tensor `data_ptr`。
- scalar和runtime block参数值。

Grid的每个维度会先执行 `int(value)` 转换；转换结果必须恰好包含三个正整数，每维
不超过 `INT32_MAX`，三维乘积不超过65535。plan创建时会检查SIMT配置和shared
memory范围；每次调用会检查参数数量、stream和grid。提交继续经过
`OpCommand.Run`。

以下状态会进入原 `NPUCachingAutotuner.run` 完整入口：

- grouped dynamic-shape autotuner。
- launcher生命周期尚未稳定，或当前为fallback launcher。
- benchmark run、runtime kwargs或 `TRITON_INTERPRET`。
- `INDUCTOR_ASCEND_DUMP_FX_GRAPH`、`INDUCTOR_ASCEND_CHECK_ACCURACY` 或
  launch-params dump。
- PyTorch profiler正在运行。

Triton launch `HookChain` 中存在已注册回调（或旧版Triton提供非空的单个hook）时，
当前调用直接使用已选定的原launcher，以保留enter/exit hook及launch metadata
语义。该状态不安装负缓存，已有plan继续保留；hook移除后的下一次调用会自动恢复
planned path。

以下稳定问题会阻止plan创建并为当前launcher安装负缓存：

- 运行时CANN不提供 `aclrtLaunchKernelWithHostArgs`。
- codegen schema与最终ABI冲突，或最终ABI不受支持。
- launcher元数据明确报告需要非零workspace、sync block lock或device print
  缓冲区。
- kernel、stub、grid resolver、FFTS状态或C++ backend等必要信息缺失，或plan
  创建失败。

负缓存首次安装前可能先进入完整入口；没有上述完整入口条件时，后续命中同一个
launcher会直接调用已选定的原launcher，不再重复进入autotuner。launcher变化后
负缓存失效并重新判断。

Python端发现schema、backend或动态grid等可恢复问题时可以使用原launcher。一旦
调用C++ Fast Launch 入口，包括参数打包、Tensor/scalar转换、stream检查和实际提交
阶段发生的异常，均按“可能已经提交”处理，不再重放原launcher，避免同一kernel
重复执行。

## 5. 关闭和故障处理

关闭功能：

```bash
export TORCHINDUCTOR_NPU_FAST_LAUNCH=0
```

修改后重启Python进程并重新执行 `torch.compile`。

常见问题：

| 现象 | 检查项 |
| --- | --- |
| 只有部分callsite使用Fast Launch | 其他callsite可能不是NPU Triton Python Wrapper，或依赖fallback launcher、不支持的ABI或额外资源 |
| PyTorch profiler、dump或准确性检查下未使用planned path | 这是预期行为；这些状态保留原完整入口语义 |
| C++ Fast Launch 调用失败后没有自动重试 | 进入C++ 入口后不会重放原launcher，以避免kernel重复执行 |
