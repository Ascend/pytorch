# Inductor

## 简介

Inductor是`torch.compile()`的默认后端编译器，通过 "Dynamo前端图捕获 + Inductor后端优化 + 高性能算子生成" 的协同方式，在不改变模型逻辑的前提下自动进行算子融合和代码生成，显著提升训练或推理的吞吐量。

Inductor后端支持四种算子编译器，可根据场景选择：

| 编译器 | 模式 | 开启方法 | 说明 |
| - | - | - | - |
| Triton | 默认模式 |`torch.compile(backend="inductor")`| 基于Triton生成融合算子，是Inductor后端的默认选择，适用于大多数场景。详细介绍参见[Triton-Ascend 官方仓库](https://gitcode.com/Ascend/triton-ascend)。|
| Torch-MLIR | MLIR模式 |`torch.compile(backend="inductor", options={"npu_backend": "mlir"})` | 基于Torch-MLIR生成融合算子，详细介绍参见[Torch-MLIR 官方仓库](https://github.com/llvm/torch-mlir)。<br><term>Ascend 950DT</term>暂不支持MLIR模式。|
| DVM | DVM模式 |`torch.compile(backend="inductor", options={"npu_backend": "dvm"})` | 基于DVM生成融合算子。详细介绍参见[DVM 官方仓库](https://gitcode.com/mindspore/dvm/tree/master)。|
| Ascend C | Ascend C模式 |`torch.compile(backend="inductor", options={"npu_backend": "ascendc"})` | 基于Ascend C生成融合算子，详细介绍参见[Autofuse官方仓库](https://gitcode.com/cann/graph-autofusion/blob/master/autofuse/README.md)。|

> [!NOTICE]
>
> Inductor后端需安装最新版本的Triton-Ascend依赖包，具体可参见[Triton-Ascend说明文档](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)。<br>
> Inductor后端使用MLIR模式/DVM模式时需额外安装Torch-MLIR依赖包，可以在[Torch-MLIR归档地址](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/)下载。<br>
> 在import torch前，也可通过TORCHINDUCTOR_NPU_BACKEND环境变量选择编译器模式，具体可参见[TORCHINDUCTOR_NPU_BACKEND](../environment_variable_reference/TORCHINDUCTOR_NPU_BACKEND.md)。

## 编译器

### Triton（默认模式）

Triton是Inductor的默认算子编译器，在保留torch.compile动态图捕获 + 静态图优化 + 高效代码生成的核心逻辑的基础上，借助Triton的自动Tiling和融合算子生成能力，显著减少开发者手动设计Tiling、管理内存等工作量，通过减少内存访问次数来提升性能。

Triton编译器包含如下核心组件：

| 组件 | 定位 | 作用 |
| - | - | - |
| Inductor-Triton | 核心编译器（融合算子生成器） | 基于Triton-Ascend自动生成高性能融合算子，通过算子融合、数据类型优化、计算图化简等策略，兼顾编译灵活性与优化深度。 |
| Triton CodeGen | 代码生成器（硬件感知代码生成器） | 接收优化后的计算图，针对目标硬件（NPU）生成专属的高性能底层代码，自动完成Tiling和共享内存管理。 |

#### 调用示例

```python
import os
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'default'

import torch
import torch_npu
from torch._inductor.utils import run_and_get_code

# 定义模型
def op_calc(x, y):
    return x * y

x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
std_out = op_calc(x, y)

compile_func = torch.compile(op_calc, backend="inductor")
compile_out, codes = run_and_get_code(compile_func, x, y)
print(codes[0])
```

#### 输出Triton融合算子

Triton编译后会在内部生成融合算子代码，通过`codes[0]`可以进行查看：

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

其中`triton_poi_fused_mul_0`是Triton编译生成的核心执行接口，用于在指定NPU设备和计算流上执行融合算子计算。

### MLIR

Inductor后端的扩展升级，在保留torch.compile动态图捕获 + 静态图优化 + 高效代码生成的核心逻辑的基础上，借助MLIR的多层级、跨架构编译能力，适配多类硬件架构的深度优化需求，显著提升不同算力平台下模型训练与推理的编译灵活性和性能上限。

Inductor-MLIR编译器包含如下核心组件：

| 组件 | 定位 | 作用 |
| - | - | - |
| Inductor-MLIR | 核心编译器（多层级IR优化器） | 将FX Graph转换为MLIR多层级中间表示（IR），通过跨层级的优化策略（如算子拆分与融合、数据类型优化、计算图化简），兼顾编译灵活性与优化深度，适配不同硬件的编译特性。 |
| MLIR CodeGen | 代码生成器（硬件感知代码生成器） | 接收优化后的MLIR IR，针对目标硬件生成专属的高性能底层代码。 |

#### 调用示例

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

#### 输出MLIR融合算子

`codes`为捕获到的MLIR输出代码片段，该代码中包含call函数与benchmark_compiled_model函数，其中call函数内部封装了MLIR编译生成的算子逻辑：

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

其中`mlir_fused_mul_0.run`是MLIR编译生成的核心执行接口，用于在指定NPU设备和计算流上执行融合乘法算子计算，接收输入张量并将计算结果写入输出张量，是MLIR对原始乘法逻辑优化后的底层实现。

### DVM

DVM是Inductor的可选算子编译器，基于DVM生成融合算子。

DVM当前支持两种融合模式：mlir_fusion与graph_fusion，分别对应复用Inductor-MLIR融合结果与自定义图融合路径两种策略。

| 模式 | 依赖Inductor MLIR | Pattern控制 | 适用场景 |
| - | - | - | - |
| mlir_fusion | 是 | 低 | 复用Inductor融合结果 |
| graph_fusion | 否 | 高 | 定制/复杂/实验性融合 | 

#### 调用示例

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

#### 输出DVM融合算子

DVM编译后会生成对应的融合算子代码。两种模式的输出示例分别如下：

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

### Ascend C

Ascend C是Inductor的可选算子编译器，基于Ascend C生成融合算子。

#### 调用示例

```python
import os
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'ascendc'

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

compile_func = torch.compile(op_calc, options={"npu_backend": "ascendc"})
compile_out, codes = run_and_get_code(compile_func, x, y)
print(codes[0])
```

#### 输出Ascend C融合算子

Ascend C编译后会在内部生成融合算子代码，通过`codes[0]`可以进行查看：

```python
def call(self, args):
    arg0_1, arg1_1 = args
    args.clear()
    # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
    # Source node to ATen node mapping:
    #   mul => mul
    # Graph fragment:
    #   %arg0_1 : Tensor "f32[3][1]npu:0" = PlaceHolder[target=arg0_1]
    #   %arg1_1 : Tensor "f32[3][1]npu:0" = PlaceHolder[target=arg1_1]
    #   %mul : Tensor "f32[3][1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
    #   return %mul
    buf0 = empty_strided((3, ), (1, ), device='npu', dtype=torch.float32)
    from torch_npu._inductor.ascendc.common.fused_layout_check import check_fused_layout
    _enable_layout_check = os.getenv("TORCH_COMPILE_DEBUG", "0") == "1" or os.getenv("TORCHINDUCTOR_NPU_EXT_LAYOUT_CHECK", "0") == "1"
    if _enable_layout_check:
        kernel_name='autofused_mul_da488cd9dfe43526f023de214ab2ff6a'
        check_fused_layout(kernel_name, 'arg0_1', arg0_1, (3,), (1,), torch.float32, 'npu')
        check_fused_layout(kernel_name, 'arg1_1', arg1_1, (3,), (1,), torch.float32, 'npu')
    autofused_mul_da488cd9dfe43526f023de214ab2ff6a(arg0_1, arg1_1, buf0)
    del arg0_1
    del arg1_1
    return (buf0, )
```

其中`autofused_mul_da488cd9dfe43526f023de214ab2ff6a`是Ascend C编译生成的核心执行接口，用于在指定NPU设备和计算流上执行融合算子计算。

## 编译模式

`mode="reduce-overhead"` 是降低开销的优化策略，核心对应**NPUGraph Tree**的逻辑：

- **动态形状路由**：管理多个有关联的NPUGraphs，让NPUGraph的优化收益能覆盖动态形状场景，而非仅局限于固定形状。
- **子图内存优化**：优化段图场景多个子图的内存使用。
- **工作原理**：在多次不同形状的输入下，Tree会缓存多个捕获的图版本，根据输入形状自动路由到最匹配的子图进行重放。

更多详情可参见[CUDAGraph Trees](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)。

NPUGraph Tree逻辑启用示例：

```python
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead"  # 启用NPUGraph Tree逻辑
)
```

## 编译选项

Inductor支持的编译选项（`options`参数）：

| 选项 | 说明 |
| - | - |
| `triton.cudagraphs` | Triton相关配置 |
| `trace.enabled` | 跟踪开关 |
| `enable_shape_handling` | 形状处理配置 |
| `npu_backend` | 指定算子编译器（`"mlir"`、`"dvm"`或`"ascendc"`，默认Triton） |

## 约束说明

1. 优化器（optimizer）通常不入图，优化器的 `step()` 包含 Python 侧动态逻辑（如学习率调度、梯度累积、自适应更新规则），难以被静态图捕获。
2. `torch.compile(backend="inductor")` 支持动态形状，但会触发重新编译（增加开销），建议尽量固定形状。
