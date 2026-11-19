# Dynamic Shapes

Dynamic Shapes 用于处理输入形状会变化的模型。编译器用符号表示可变尺寸，在约束成立时复用已编译的图和 kernel，避免每次尺寸变化都重新编译。

TorchNPU 支持 PyTorch 的动态形状功能。使用时导入 `torch_npu`，将设备和张量替换为 NPU，动态维度标记、动态来源配置和编译排障的 API 与参数与原生保持一致。

## 动态形状基础

动态形状用符号化尺寸表达可变维度，并用 Guard 记录编译时的约束。Guard 不满足时，编译器会重新生成适合的图和代码。

- **Dynamic Shapes 概览**：介绍动态形状的用途、开启方式和整体流程。TorchNPU 提供与原生 PyTorch 一致的使用方式，可以阅读 [PyTorch Dynamic Shapes](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html) 了解具体用法。

- **核心概念**：说明符号化、Guard、编译缓存和重编译的总体机制。可以阅读 [Dynamic Shapes 核心概念](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html) 进行了解。

## 动态行为控制

PyTorch 提供自动动态、逐项标记、按 source 配置和强制静态等控制方式。TorchNPU 上的 API 和参数与原生保持一致。

- **自动动态**：模型连续遇到不同尺寸时，PyTorch 自动把变化的维度标记为动态，不需要修改代码。可以阅读 Dynamic Shapes 概览中的[动态行为启用方式](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#enabling-dynamic-behavior)了解具体行为。

- **torch._dynamo.mark_dynamic**：把指定张量的指定维度显式标记为动态。API 和参数与原生一致，可以阅读[控制 dynamic shape 的进阶选项](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_advanced_control_options.html)了解用法。

- **torch._dynamo.mark_unbacked**：用于数据依赖产生的、没有真实数据支撑的符号尺寸，通常配合 `torch._check` 使用。API 和参数与原生一致，输入 0 维空张量时需要 triton-ascend 大于等于 3.6.0。

- **Dynamic Source List 与 Static Source List**：按 source 名称控制哪些输入或维度保持动态或静态。Static Source List 需要 PyTorch 2.15 及以上版本，配置语法与原生一致。可以阅读 [Dynamic Shapes 概览](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html) 了解优先级和示例。

- **Profile-Guided Optimization（PGO）**：把自动动态的决策保存下来，在多次运行之间复用，减少重复探索。环境变量和使用方式与原生一致，可以阅读[进阶选项中的 PGO](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_advanced_control_options.html#profile-guided-optimization-pgo)了解配置方法。

- **Compiler Collective（编译器集合通信）**：在多个 rank 之间共享观察到的尺寸，让各 rank 对动态性得出一致结论。TorchNPU 使用 HCCL 完成集合通信，需要用 `dist.init_process_group(backend="hccl")` 初始化进程组，并用 `with torch._dynamo.config.patch(enable_compiler_collectives=True):` 启用该功能。所有 rank 必须同时编译、控制流不分叉；容器内主机名需要可解析，否则多卡初始化可能挂起。通用机制见[编译器集合通信](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_advanced_control_options.html#compiler-collective)，HCCL 初始化和分布式环境说明见[分布式概述](https://www.hiascend.com/document/detail/zh/Pytorch/latest/devguide/dist/docs/zh/developer_notes/distributed/distributed_overview.md)。

以下示例使用两个 NPU 演示 Compiler Collective 的实际流程：

```python
# compiler_collective.py
import datetime
import os

import torch
import torch.distributed as dist
import torch_npu  # 注册 NPU 后端

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = 2
torch.npu.set_device(local_rank)

# 1. 使用 HCCL 初始化两个 rank 的进程组。
dist.init_process_group(
    backend="hccl",
    rank=rank,
    world_size=world_size,
    timeout=datetime.timedelta(seconds=90),
)


def fn(x, y):
    # 2. 显式读取形状，让编译器观察各 rank 的输入尺寸。
    _ = x.shape, y.shape
    return x.sum() + y.sum()


# 3. 所有 rank 同时进入编译，并启用 Compiler Collective。
with torch._dynamo.config.patch(enable_compiler_collectives=True):
    compiled = torch.compile(fn, backend="inductor")
    sizes = (4, 4) if rank == 0 else (3, 4)
    for size in sizes:
        x = torch.arange(size, dtype=torch.float32, device=f"npu:{local_rank}")
        y = x.clone()
        result = compiled(x, y)
        expected = torch.tensor(size * (size - 1), dtype=torch.float32, device=result.device)
        torch.testing.assert_close(result, expected)

# 4. 各 rank 必须执行相同数量和顺序的编译/集合通信，再退出。
dist.destroy_process_group()
```

保存后在容器内启动。两个 rank 需同时可见 NPU，且主机名可解析：

```bash
torchrun --standalone --nproc_per_node=2 compiler_collective.py
```

示例中两个 rank 的输入尺寸不同，启用 Compiler Collective 后会在编译阶段交换尺寸信息，使各 rank 对动态性得出一致结论。控制流不能出现只有部分 rank 进入的分支，否则集合通信会等待。

- **空张量与 unbacked 动态维度**：用于处理长度为 0、1 等满足约束的输入。在 TorchNPU 中使用时，API 和参数与原生保持一致；输入 0 维空张量时需要确保 triton-ascend 大于等于 3.6.0，否则可能出现 `coreDim=0` 错误。通用调试方法见[动态形状调试文档](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_debugging_tlparse_torch_logs.html)，正确写法如下：

```python
import torch
import torch_npu  # 注册 NPU 后端


def fn(x, y):
    torch._check(x.shape[0] < 10)
    if x.shape[0] < 10:
        return x * y


device = torch.device("npu:0")
first_x = torch.arange(1, 6, dtype=torch.float32, device=device)
first_y = first_x / 2
torch._dynamo.decorators.mark_unbacked(first_x, 0)
torch._dynamo.decorators.mark_unbacked(first_y, 0)
compiled = torch.compile(fn, backend="inductor")

for n in (5, 0, 1):
    x = first_x if n == 5 else torch.arange(1, n + 1, dtype=torch.float32, device=device)
    y = first_y if n == 5 else x / 2
    actual = compiled(x, y)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), (x * y).cpu())

# 长度 10 会违反 torch._check，按预期抛出运行时断言错误。
bad_x = torch.arange(1, 11, dtype=torch.float32, device=device)
compiled(bad_x, bad_x / 2)
```

- **自定义 Triton 算子**：`tl.constexpr` 参数与原生一样会按取值 specialize，不同取值可能产生重编译。排查思路见[Dynamic Shapes 后续步骤](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#where-do-i-go-from-here)。

## 进阶原理

以下主题用于理解编译器内部行为，属于原生 PyTorch 机制，TorchNPU 上同样适用。

- **进阶原理**：汇总 Guard、动态性和 specialization 的进阶主题。可以阅读 [Beyond the Basics](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_beyond_the_basics.html) 进行了解。

- **0/1 Specialization 问题**：说明长度为 0 和 1 的张量为什么会被单独 specialize，以及如何用 unbacked 符号减少额外 Guard。可以阅读 [The Zero-One Specialization Problem](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_zero_one_specialization.html) 了解处理方式。

- **Backed 与 Unbacked SymInt**：说明两类符号尺寸的区别、来源和适用范围。可以阅读 [Backed vs Unbacked Symints](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_backed_unbacked.html) 进行了解。

## 故障排查

动态形状问题通常表现为重编译次数过多、Guard 失败或数据依赖导致无法编译，可以使用以下入口定位原因。

- **故障排查总览**：汇总常见问题类型和排查入口。可以阅读 [Troubleshooting Dynamic Shapes](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_troubleshooting.html) 进行了解。

- **TORCH_LOGS=dynamic 与 tlparse**：用 `TORCH_LOGS=dynamic` 观察符号创建和 Guard 添加，用 tlparse 分析编译产物和 specialize 原因。NPU 日志与社区日志的内容相同，但 NPU 输出省略了时间戳、进程号和源码路径，只保留 `[编译序号] create_symbol ...`、`[编译序号] eval ... [guard added]` 等事件。完整流程见[动态形状调试文档](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_debugging_tlparse_torch_logs.html)。

同一段 specialization 示例在社区与 NPU 上的日志对比如下：

```text
# PyTorch 社区日志（节选，包含时间、进程号和源码路径）
I0721 11:10:00.950000 845259 symbolic_shapes.py:3776] [0/0] create_symbol s77 = 5 for L['x'].size()[0] [2, int_oo]
I0721 11:10:01.038000 845259 symbolic_shapes.py:7211] [0/0] eval Eq(s77, 5) [guard added]

# Ascend NPU 日志（节选，省略时间、进程号和源码路径）
[0/0] create_symbol s77 = 5 for L['x'].size()[0] [2, int_oo]
[0/0] eval Eq(s77, 5) [guard added]
```

查看 NPU 日志时：

- 方括号中的 `[0/0]` 是本次编译的序号，不是时间或进程号；同一序号下的事件属于同一次编译。
- `create_symbol s77 = 5` 表示为 `L['x'].size()[0]` 创建符号，当前观察到的尺寸为 5，`[2, int_oo]` 是符号的取值范围。
- `[guard added]` 表示新增了 Guard。示例中的 `Eq(s77, 5)` 会把符号限定为 5；后续输入尺寸不是 5 时会重新编译。
- 排查时先用 `grep` 过滤 `create_symbol` 和 `[guard added]`，再根据编译序号和输入名称定位 specialization 的来源。

- **GuardOnDataDependentSymNode 错误**：数据依赖的控制流或取值无法转成 Guard 时会出现该错误。可以阅读 [GuardOnDataDependentSymNode 错误排查](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_troubleshooting_guardon_errors.html) 了解定位和改写方式。

## 如何选择动态化方式

1. **只是输入尺寸会变化**：优先使用自动动态，必要时用 `mark_dynamic` 逐项标记。
2. **尺寸来自数据本身**：使用 `mark_unbacked` 配合 `torch._check`，明确运行时约束。
3. **多卡需要一致的动态性判断**：开启 Compiler Collective，并使用 HCCL 后端。
4. **重编译仍然过多**：用 PGO 复用决策，或用 Static Source List 固定不必要的动态维度。
5. **需要定位具体原因**：用 `TORCH_LOGS=dynamic` 或 tlparse 查看符号创建和 Guard。
