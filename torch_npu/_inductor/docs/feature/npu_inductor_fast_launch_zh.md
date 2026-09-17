# NPU Inductor Planned Fast Launch 使用指南

## 1. 功能说明

Planned Fast Launch 用于降低 `torch.compile` 生成的 Python Wrapper 在稳态运行时
下发 NPU Triton 融合 kernel 的 Host 侧固定开销。launcher 稳定后，Wrapper 会创建
C++ `FastLaunchPlan`；后续调用复用计划，只更新 stream、grid 和参数。

该功能有以下边界：

- 仅覆盖 NPU Inductor Python Wrapper 生成的 Triton kernel callsite。
- 不覆盖 eager/ACLNN 算子、C++ Wrapper 和图分区子图。
- compile-time autotune 保留原 codegen；grouped dynamic-shape autotuner 不附加
  Fast Launch launcher 元数据，运行时直接绑定原 `run` 入口。
- PyTorch profiler、FX graph/launch-params dump 和准确性检查等状态不进入
  planned path，由原路径保留对应语义。
- Triton launcher scope 中空的 launch `HookChain` 对象本身不阻止 planned path；
  注册真实回调后，当前调用临时使用原 launcher，回调移除后自动恢复 planned path。
- 默认关闭，需要显式启用。

## 2. 前置条件

- 使用 `torch_npu` 2.10 系列中包含本功能的构建产物。
- PyTorch、torch_npu、CANN 和 NPU 驱动版本相互匹配。
- 模型通过 `torch.compile(..., backend="inductor")` 执行，并实际生成 NPU Triton
  kernel。

环境变量在 Python 进程导入 `torch_npu` 以及执行 `torch.compile` **之前**设置。
运行中修改环境变量不会改写已经生成的 Wrapper；需要重启进程并重新编译。

## 3. 启用方式

Linux Shell：

```bash
export TORCHINDUCTOR_NPU_FAST_LAUNCH=1
python run_model.py
```

Python 启动脚本也可以在导入 PyTorch/torch_npu 前设置：

```python
import os

os.environ["TORCHINDUCTOR_NPU_FAST_LAUNCH"] = "1"

import torch
import torch_npu
```

最小使用示例：

```python
import torch
import torch_npu


def fn(x, y):
    return torch.sin(x + y) * 2


x = torch.randn(4096, device="npu")
y = torch.randn(4096, device="npu")
compiled_fn = torch.compile(fn, backend="inductor")

result = compiled_fn(x, y)
```

无需修改模型代码或显式调用 `_npu_inductor_fast_launch_with_plan`。计划创建和路由
由生成的 Wrapper 自动完成。初始调用可能包含编译、autotune 和计划创建开销；若启用
coordinate-descent tuning 或 kernel binary 保存，也会先完成相应生命周期工作。

## 4. 生命周期、ABI 和回退

生成的 Wrapper 会为每个 Triton callsite 惰性创建绑定对象。没有完整入口条件时，
如果选定 launcher 尚未稳定，本次调用使用原 `NPUCachingAutotuner.run`，调用结束后
再尝试创建 `FastLaunchPlan`；如果 launcher 已经稳定，则可以直接创建计划并进入
快路径。存在完整入口条件时只执行原入口，不尝试创建 plan。

创建计划需要满足以下条件：

- 已选定唯一且稳定的 launcher，且不是 fallback launcher。
- autotune 已完成。
- 启用 coordinate-descent tuning 时，调优已经完成。
- 启用 kernel binary 保存时，保存已经完成。
- 最终编译 ABI、C++ backend 和所需资源均受支持。

codegen schema 不完整本身不会拒绝快路径；launcher 稳定后会使用最终编译 ABI 补全
参数类型。schema 明确冲突、最终 ABI 或资源不受支持，kernel、stub、grid resolver、
FFTS 状态或 backend 等必要信息缺失，以及 plan 创建失败等稳定问题，会为当前
launcher 安装负缓存并继续使用原 launcher。动态 grid 等单次调用错误不会污染已有
plan。

当前计划 backend 支持以下 ABI 参数：

- `torch.Tensor` 参数，提交时按其 `data_ptr` 打包。
- `i32`、`i64`、`u32`、`u64`。
- `f32`、`f64`。
- 按 `int32` 打包的 `bool`。
- 由最终 launcher ABI 提供、且类型属于上述范围的 runtime block 参数。

Plan 保存 kernel name、kernel stub 及其 owner、参数布局、SIMT/shared memory 配置，
以及目标设备需要的 FFTS 地址。FFTS 地址在 plan 创建时查询一次。Plan 不记录或校验
device id；C++ backend 对传入 stream 只检查非空，不核对其所属设备，因此同一个 plan
不应跨设备复用。以下信息在每次调用时重新解析并打包：

- 当前非空 NPU stream。
- 当前三维 grid。
- Tensor `data_ptr`。
- scalar 和 runtime block 参数值。

Grid 的每个维度会先执行 `int(value)` 转换；转换结果必须恰好包含三个正整数，每维
不超过 `INT32_MAX`，三维乘积不超过 65535。plan 创建时会检查 SIMT 配置和 shared
memory 范围；每次调用会检查参数数量、stream 和 grid。提交继续经过
`OpCommand.Run`。

以下状态会进入原 `NPUCachingAutotuner.run` 完整入口：

- grouped dynamic-shape autotuner。
- launcher 生命周期尚未稳定，或当前为 fallback launcher。
- benchmark run、runtime kwargs 或 `TRITON_INTERPRET`。
- `INDUCTOR_ASCEND_DUMP_FX_GRAPH`、`INDUCTOR_ASCEND_CHECK_ACCURACY` 或
  launch-params dump。
- PyTorch profiler 正在运行。

Triton launch `HookChain` 中存在已注册回调（或旧版 Triton 提供非空单 hook）时，
当前调用直接使用已选定的原 launcher，以保留 enter/exit hook 及 launch metadata
语义。该状态不安装负缓存，已有 plan 继续保留；hook 移除后的下一次调用会自动恢复
planned path。

以下稳定问题会阻止 plan 创建并为当前 launcher 安装负缓存：

- codegen schema 与最终 ABI 冲突，或最终 ABI 不受支持。
- launcher 元数据明确报告需要非零 workspace、sync block lock 或 device print
  缓冲区。
- kernel、stub、grid resolver、FFTS 状态或 C++ backend 等必要信息缺失，或 plan
  创建失败。

负缓存首次安装前可能先进入完整入口；没有上述完整入口条件时，后续命中同一个
launcher 会直接调用已选定的原 launcher，不再重复进入 autotuner。launcher 变化后
负缓存失效并重新判断。

Python 端发现 schema、backend 或动态 grid 等可恢复问题时可以使用原 launcher。一旦
调用 C++ fast-launch 入口，包括参数打包、Tensor/scalar 转换、stream 检查和实际提交
阶段发生的异常，均按“可能已经提交”处理，不再重放原 launcher，避免同一 kernel
重复执行。

## 5. 关闭和故障处理

关闭功能：

```bash
export TORCHINDUCTOR_NPU_FAST_LAUNCH=0
```

修改后重启 Python 进程并重新执行 `torch.compile`。

常见问题：

| 现象 | 检查项 |
| --- | --- |
| 只有部分 callsite 使用 fast launch | 其他 callsite 可能不是 NPU Triton Python Wrapper，或依赖 fallback launcher、不支持的 ABI 或额外资源 |
| PyTorch profiler、dump 或准确性检查下未使用 planned path | 这是预期行为；这些状态保留原完整入口语义 |
| C++ fast-launch 调用失败后没有自动重试 | 进入 C++ 入口后不会重放原 launcher，以避免 kernel 重复执行 |
