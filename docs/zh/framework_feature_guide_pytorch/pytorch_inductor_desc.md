# Inductor

## 简介

Inductor 是 `torch.compile()` 的默认后端编译器，通过 "Dynamo 前端图捕获 + Inductor 后端优化 + 高性能算子生成" 的协同方式，在不改变模型逻辑的前提下自动进行算子融合和代码生成，显著提升训练或推理的吞吐量。

Inductor 后端包含如下核心组件：

| 组件 | 定位 | 作用 |
| - | - | - |
| Dynamo | 前端编译器（代码转换器） | TorchDynamo 能够 JIT（即时）将用户的 eager（动态图）代码编译为 FX Graph（PyTorch 的中间表示），进而交给 Inductor 进行编译，最终生成优化过后的底层机器代码。 |
| Inductor | 后端编译器（高效代码生成器） | 具备基于多种模式（包括 Triton/MLIR/DVM）的自动生成高性能算子能力，能够显著减少开发者手动设计 Tiling、管理内存等工作量。支持算子融合等图优化策略，通过减少内存访问次数来提升性能。 |

## 算子编译器

Inductor 后端支持三种算子编译器，可根据场景选择：

| 编译器 | 模式 | 说明 |
| - | - | - |
| Triton-Ascend | 默认模式 | 基于 Triton-Ascend 生成融合算子，是 Inductor 后端的默认选择。详细介绍参考 [Triton-Ascend 官方仓库](https://gitcode.com/Ascend/triton-ascend)。 |
| Torch-MLIR | MLIR 模式 | 基于 Torch-MLIR 生成融合算子。详细介绍参考 [Torch-MLIR 官方仓库](https://github.com/llvm/torch-mlir)。 |
| DVM | DVM 模式 | 基于 DVM 生成融合算子。详细介绍参考 [DVM 官方仓库](https://gitcode.com/mindspore/dvm/tree/master)。 |

## 使用场景

Inductor 后端以降低 Python 开销和 kernel 启动开销为核心，适合迭代次数多、单步计算量中等的场景。

- **Triton 模式（默认）**：通过 `torch.compile(backend="inductor")` 使能，基于 Triton-Ascend 生成融合算子，适用于大多数场景。
- **MLIR 模式**：通过 `torch.compile(backend="inductor", options={"npu_backend": "mlir"})` 使能，基于 Torch-MLIR 生成融合算子，需额外安装 Torch-MLIR 依赖包。
- **DVM 模式**：通过 `torch.compile(backend="inductor", options={"npu_backend": "dvm"})` 使能，基于 DVM 生成融合算子。

## 使用指导

> **注意**
>
> - Inductor 后端需安装最新版本的 Triton-Ascend 依赖包，具体可参考 [Triton-Ascend 说明文档](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)。
> - Inductor 后端使用 MLIR 模式时需额外安装 Torch-MLIR 依赖包，可以在 [Torch-MLIR 归档地址](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/) 下载。

### 接口原型

```python
def compile(model, *, fullgraph=False, dynamic=None, backend="inductor",
            mode=None, options=None, disable=False)
```

**参数说明：**

- **model**：必选参数，要编译的模型或者函数。
- **fullgraph**：可选参数，是否强制整图编译，默认值为 False。
- **dynamic**：可选参数，是否需要动态 shape 编译，默认值为 None。
- **backend**：可选参数，编译后端，支持 inductor 和 npugraphs，默认值为 inductor。
- **mode**：可选参数，编译模式，目前支持 None（默认值）和 "reduce-overhead"。
- **options**：可选参数，编译选项，目前支持：
  - `triton.cudagraphs` — Triton 相关配置
  - `trace.enabled` — 跟踪开关
  - `enable_shape_handling` — 形状处理配置
  - `npu_backend` — 指定算子编译器（`"mlir"` 或 `"dvm"`，默认为 Triton）
- **disable**：可选参数，是否关闭 torch.compile 能力，默认值为 False。

该接口详情可参考原生 [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)。

### 使用样例

```python
import torch
import torch_npu
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleMLP().npu()
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead"
)
# 如需指定算子编译器，可添加 options={"npu_backend": "mlir"} 或 options={"npu_backend": "dvm"}

optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 128).npu()
y = torch.randint(0, 10, (32,)).npu()

for _ in range(100):
    output = compiled_model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 约束说明

1. 优化器（optimizer）通常不入图，优化器的 `step()` 包含 Python 侧动态逻辑（如学习率调度、梯度累积、自适应更新规则），难以被静态图捕获。
2. `torch.compile(backend="inductor")` 支持动态形状，但会触发重新编译（增加开销），建议尽量固定形状。

---

## Triton（默认模式）

Triton 是 Inductor 的默认算子编译器，在保留 torch.compile 动态图捕获 + 静态图优化 + 高效代码生成的核心逻辑的基础上，借助 Triton-Ascend 的自动 Tiling 和融合算子生成能力，显著减少开发者手动设计 Tiling、管理内存等工作量，通过减少内存访问次数来提升性能。关于 Triton-Ascend 的详细介绍，可以参考 [Triton-Ascend 官方仓库](https://gitcode.com/Ascend/triton-ascend)。

Inductor-Triton 包含如下核心组件：

| 组件 | 定位 | 作用 |
| - | - | - |
| Dynamo | 前端编译器（代码转换器） | 延续 TorchDynamo 的核心能力，JIT 将用户的 eager（动态图）代码编译为 FX Graph，完成动态图到静态中间表示的捕获，为 Inductor-Triton 后端提供统一的输入，保证前端编译逻辑的一致性。 |
| Inductor-Triton | 核心编译器（融合算子生成器） | 基于 Triton-Ascend 自动生成高性能融合算子，通过算子融合、数据类型优化、计算图化简等策略，兼顾编译灵活性与优化深度。 |
| Triton CodeGen | 代码生成器（硬件感知代码生成器） | 接收优化后的计算图，针对目标硬件（NPU）生成专属的高性能底层代码，自动完成 Tiling 和共享内存管理。 |

### 环境配置

- torch_npu：2.6.0 或之后版本的 torch_npu 支持 torch.compile 接口。
- Triton-Ascend：Inductor 后端的 Triton 模式依赖 Triton-Ascend。用户可以从 [Triton-Ascend 官方仓库](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md) 获取并安装最新版本的 Triton-Ascend 依赖包。

### 使用方式

Triton 使用方式有三种，任选一种即可调用：

1. 不传入任何环境变量，默认为 Triton。

   ```python
   torch.compile(op_calc)(x)
   ```

2. 环境变量导入（npu 初始化之前导入）。

   ```python
   import os
   os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'default'
   torch.compile(op_calc)(x)
   ```

### 示例

```python
import os
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'default'

import torch
import torch_npu

# 定义模型
def op_calc(x, y):
    return x * y

x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
std_out = op_calc(x, y)

compile_func = torch.compile(op_calc)
compile_out, codes = run_and_get_code(compile_func, x, y)
print(codes[0])
```

### 输出 Triton 融合算子

Triton 编译后会在内部生成融合算子代码，通过 `codes[0]` 可以进行查看：

```python
def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        buf0 = empty_strided((3, ), (1, ), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg0_1, arg1_1, buf0, 3, stream=stream0)
        del arg0_1
        del arg1_1
    return (buf0, )
```

其中 `triton_poi_fused_mul_0` 是 Triton 编译生成的核心执行接口，用于在指定 NPU 设备和计算流上执行融合算子计算。

---

## MLIR

Inductor 后端的扩展升级，在保留 torch.compile 动态图捕获 + 静态图优化 + 高效代码生成的核心逻辑的基础上，借助 MLIR 的多层级、跨架构编译能力，适配多类硬件架构的深度优化需求，显著提升不同算力平台下模型训练与推理的编译灵活性和性能上限。关于 Torch-MLIR 的详细介绍，可以参考 [Torch-MLIR 官方仓库](https://github.com/llvm/torch-mlir)。

Inductor-MLIR 包含如下核心组件：

| 组件 | 定位 | 作用 |
| - | - | - |
| Dynamo | 前端编译器（代码转换器） | 延续 TorchDynamo 的核心能力，JIT 将用户的 eager（动态图）代码编译为 FX Graph，完成动态图到静态中间表示的捕获，为 Inductor-MLIR 后端提供统一的输入，保证前端编译逻辑的一致性。 |
| Inductor-MLIR | 核心编译器（多层级 IR 优化器） | 将 FX Graph 转换为 MLIR 多层级中间表示（IR），通过跨层级的优化策略（如算子拆分与融合、数据类型优化、计算图化简），兼顾编译灵活性与优化深度，适配不同硬件的编译特性。 |
| MLIR CodeGen | 代码生成器（硬件感知代码生成器） | 接收优化后的 MLIR IR，针对目标硬件生成专属的高性能底层代码。 |

### 环境配置

- torch_npu：2.6.0 或之后版本的 torch_npu 支持 torch.compile 接口。
- torch-mlir：Inductor 后端的 MLIR 模式依赖 torch-mlir。用户可以从 [Torch-MLIR 归档地址](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/) 获取 torch-mlir 软件包。

### 使用方式

MLIR 使用方式有三种，任选一种即可调用：

1. config 导入（torch.compile 之前导入）。

   ```python
   torch._inductor.config.npu_backend = "mlir"
   torch.compile(op_calc)(x)
   ```

2. 环境变量导入（npu 初始化之前导入）。

   ```python
   import os
   os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'
   torch.compile(op_calc)(x)
   ```

3. options 导入。

   ```python
   torch.compile(op_calc, options={"npu_backend": "mlir"})(x)
   ```

### 示例

```python
import os
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'

import torch
import torch_npu
from torch._inductor.utils import run_and_get_code
import torch_npu._inductor

# 定义模型
def op_calc(x, y):
    return x * y

x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
std_out = op_calc(x, y)

compile_func = torch.compile(op_calc, options={"npu_backend": "mlir"})
compile_out, codes = run_and_get_code(compile_func, x, y)
print(codes[0])
```

### 输出 MLIR 融合算子

`codes` 为捕获到的 MLIR 输出代码片段，该代码中包含 call 函数与 benchmark_compiled_model 函数，其中 call 函数内部封装了 MLIR 编译生成的算子逻辑：

```python
def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        buf0 = empty_strided((3, ), (1, ), device='npu', dtype=torch.float32)
        stream0 = get_raw_stream(0)
        mlir_fused_mul_0.run(arg0_1, arg1_1, buf0, stream=stream0)
        del arg0_1
        del arg1_1
    return (buf0, )
```

其中 `mlir_fused_mul_0.run` 是 MLIR 编译生成的核心执行接口，用于在指定 NPU 设备和计算流上执行融合乘法算子计算，接收输入张量并将计算结果写入输出张量，是 MLIR 对原始乘法逻辑优化后的底层实现。

---

## DVM

DVM 是 Inductor 的可选算子编译器，通过 `torch.compile(backend="inductor", options={"npu_backend": "dvm"})` 使能。基于 DVM 生成融合算子。关于 DVM 的详细介绍，可以参考 [DVM 官方仓库](https://gitcode.com/mindspore/dvm/tree/master)。

DVM 当前支持两种融合模式：mlir_fusion 与 graph_fusion，分别对应复用 Inductor-MLIR 融合结果与自定义图融合路径两种策略。

### mlir_fusion 融合模式

**设计思路**

- 基于 Inductor-MLIR 已生成的融合算子 FX 图。
- 在 Inductor 的 MLIR 编译路径中获取已融合的 FX 子图。
- 使用 DVM codegen 将该融合子图转化为 DVM 融合算子并执行。

**特点**

- 复用 Inductor 既有的 lowering 与 MLIR 融合能力。
- 不引入额外的 pattern 识别逻辑，强调与 Inductor 编译链路协同。
- 适合快速承接 Inductor-MLIR 的融合收益。

### graph_fusion 融合模式

**设计思路**

- 不依赖 Inductor 的 lowering / MLIR 流程。
- 通过 CapabilityBasedPartitioner 和 DVM 自身能力识别可融合 pattern。
- 构建融合子图后，使用 DVM codegen 生成并执行融合算子。

**特点**

- 提供独立的融合路径，不受 Inductor 融合策略限制。
- 支持更灵活、可控的 pattern 设计与扩展。
- 适合复杂融合、定制优化或实验性融合策略。

### 使用方式

DVM 使用方式有三种，默认为 mlir_fusion 融合模式，任选一种即可调用：

1. config 导入（torch.compile 之前导入）。

   ```python
   torch._inductor.config.npu_backend = "dvm"
   torch.compile(op_calc)(x)
   ```

2. 环境变量导入（npu 初始化之前导入）。

   ```python
   import os
   os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'dvm'
   torch.compile(op_calc)(x)
   ```

3. options 导入。

   ```python
   torch.compile(op_calc, options={"npu_backend": "dvm"})(x)
   ```

graph_fusion 使能方式为 torch.compile 之前导入。

```python
from torch_npu._inductor.dvm.graph_fusion import DvmGraphFusionPatch
DvmGraphFusionPatch.enable()
```

或通过 with DvmGraphFusionPatch 导入：

```python
from torch_npu._inductor.dvm.graph_fusion import DvmGraphFusionPatch

with DvmGraphFusionPatch():
    dvm_compiled_model = torch.compile(
        model, backend="inductor", dynamic=is_dynamic
    )
```

### 示例

```python
import os
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'dvm'

import torch
import torch_npu
import torch_npu._inductor

# mlir_fusion 模式
torch._inductor.config.npu_backend = "dvm"

# 定义模型
def op_calc(x, y):
    return x * y

x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")

# 编译执行
compile_func = torch.compile(op_calc, backend="inductor", options={"npu_backend": "dvm"})
out = compile_func(x, y)
```

### 输出 DVM 融合算子

DVM 编译后会生成对应的融合算子代码。两种模式的输出示例分别如下：

**mlir_fusion 模式输出：**

```python
# 声明算子
@dvm.kernel(ktype='vector', dyn_shape=True)
def dvm_fused_add_mul_sum_0_build(k):
    arg0_1 = k.load([-1, 1], torch.bfloat16)
    arg1_1 = k.intref()
    arg2_1 = k.intref()
    arg3_1 = k.intref()
    arg4_1 = k.view_load([-1, -1, -1], [-1, -1, 1], torch.bfloat16)
    arg5_1 = k.load([1, -1], torch.bfloat16)
    expand = k.broadcast(arg0_1, [arg1_1, arg2_1, arg3_1])
    mul = k.mul(expand, arg4_1)
    expand_1 = k.broadcast(arg5_1, [arg1_1, arg2_1, arg3_1])
    add = k.add(mul, expand_1)
    _npu_dtype_cast = k.cast(add, torch.float32)
    sum_1 = k.sum(_npu_dtype_cast, [0, 2], True)
    k.store(sum_1, torch.float32)
    k.set_kernel_info(
        'dvm_fused_add_mul_sum',
        'dvm_fused_add_mul_sum_0',
        [True, True, True, True, True, True, True],
    )

# 使用
buf2 = empty_strided((1, s0, 1), (s0, 1, s0), device='npu', dtype=torch.float32)
stream0 = get_raw_stream(0)
dvm_fused_add_mul_sum_0.run(arg4_1, s1, s0, s2, buf1, arg5_1, buf2, stream=stream0)
```

**graph_fusion 模式输出：**

```python
# 声明算子
@dvm.kernel(ktype='split', dyn_shape=True)
def dvm_graph_fused_0(k):
    arg4_1 = k.load([-1, 1], torch.bfloat16)
    permute = k.view_load([-1, -1, -1], [-1, -1, 1], torch.bfloat16)
    mul_3 = k.mul(arg4_1, permute)
    arg5_1 = k.load([1, -1], torch.bfloat16)
    add_8 = k.add(mul_3, arg5_1)
    convert_element_type_default = k.cast(add_8, torch.float32)
    sum_1 = k.sum(convert_element_type_default, [0, 2], True)
    convert_element_type_default_1 = k.cast(sum_1, torch.bfloat16)
    k.store(convert_element_type_default_1, torch.bfloat16)

# 使用
buf0 = dvm_graph_fused_0(arg4_1, reinterpret_tensor(arg3_1, (s1, s0, s2), (s2, s1*s2, 1), 0), arg5_1)
```

### 简要对比

| 模式 | 依赖 Inductor MLIR | Pattern 控制 | 适用场景 |
| - | - | - | - |
| mlir_fusion | 是 | 低 | 复用 Inductor 融合结果 |
| graph_fusion | 否 | 高 | 定制/复杂/实验性融合 |

---

## 选项配置

Inductor 支持的编译选项（`options` 参数）：

| 选项 | 说明 |
| - | - |
| `triton.cudagraphs` | Triton 相关配置 |
| `trace.enabled` | 跟踪开关 |
| `enable_shape_handling` | 形状处理配置 |
| `npu_backend` | 指定算子编译器（`"mlir"` 或 `"dvm"`，默认 Triton） |

### reduce-overhead（NPUGraph Tree）

`mode="reduce-overhead"` 是降低开销的优化策略，核心对应 **NPUGraph Tree** 的逻辑：

- **动态形状路由**：管理多个有关联的 NPUGraphs，让 NPUGraph 的优化收益能覆盖动态形状场景，而非仅局限于固定形状。
- **子图内存优化**：优化段图场景多个子图的内存使用。
- **工作原理**：在多次不同形状的输入下，Tree 会缓存多个捕获的图版本，根据输入形状自动路由到最匹配的子图进行重放。

```python
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead"  # 启用 NPUGraph Tree 逻辑
)
```

**关键源文件（torch_npu 仓库内）：**

- `torch_npu/npu/_graph_tree.py` — NPUGraphTreeManager、GraphNode、WarmupNode
- `torch_npu/utils/_graph_tree.py` — NpugraphsBackend、后端注册
