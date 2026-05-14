# 简介

MFusion 是 PyTorch Inductor 的图融合优化组件，基于 MLIR（Multi-Level Intermediate Representation）技术实现高性能的算子融合与代码生成。

MFusion 包含如下核心组件：

| 组件              | 定位           | 作用                                                                           |
| ----------------- | -------------- | ------------------------------------------------------------------------------ |
| graph_fusion      | 核心融合引擎   | 负责 FX Graph 的子图提取、MLIR 转换与融合算子生成，支持动态 shape 和静态图场景 |
| fx_mlir_converter | FX-MLIR 转换器 | 将 FX Graph 转换为 MLIR 中间表示，保留算子语义并生成融合算子                   |
| subgraph_registry | 子图注册表     | 管理融合子图的元数据，包括 FX GraphModule、MLIR 代码和动态/静态标记            |
| decomp            | 算子分解       | 提供 Inductor 融合前的算子预处理，确保融合路径的正确性                         |

# 使用指导

## 环境配置

* mfusion：MFusion 组件的 Python wheel 包
* torch_npu：2.7.1 和 2.9.0 支持 MFusion 功能
* torch-mlir：MFusion 依赖 torch-mlir 进行 MLIR 代码生成，用户可从 [LINK](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/) 获取

## 环境变量

| 环境变量                     | 说明                      | 默认值 |
| ---------------------------- | ------------------------- | ------ |
| TORCHINDUCTOR_ENABLE_MFUSION | 是否启用 MFusion 融合优化 | "0"    |

设置 `TORCHINDUCTOR_ENABLE_MFUSION=1` 即可启用 MFusion 融合功能。

## 使用方式

### 方式一：环境变量启用（推荐）

```Python
import os
os.environ['TORCHINDUCTOR_ENABLE_MFUSION'] = '1'

import torch
import torch_npu

# 定义模型
def op_calc(x, y):
    return x * y + x

x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")

# 使用 torch.compile 编译
compile_func = torch.compile(op_calc, backend="inductor")
compile_out = compile_func(x, y)
```

### 方式二：MFusionPatch 上下文管理器

```Python
import torch
from torch_npu._inductor.mfusion import MFusionPatch

def op_calc(x, y):
    return x * y + x

with MFusionPatch():
    compile_func = torch.compile(op_calc, backend="inductor")
    compile_out = compile_func(x, y)
```

## 示例

以下示例展示如何使用 MFusionPatch 对包含多个算子的模型进行融合优化：

```Python
import torch
import torch_npu

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, arg0, arg1, arg2, arg3, arg4):
        reshape_default = torch.ops.aten.reshape.default(arg0, (384, 1, 1))
        sub_tensor = torch.ops.aten.sub.Tensor(arg1, reshape_default)
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, arg2)
        reshape_default_1 = torch.ops.aten.reshape.default(arg3, (384, 1, 1))
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, reshape_default_1)
        reshape_default_2 = torch.ops.aten.reshape.default(arg4, (384, 1, 1))
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, reshape_default_2)
        return add_tensor


arg0 = torch.empty_strided((384,), (1,), dtype=torch.float32, device='npu').uniform_(0, 1)
arg1 = torch.empty_strided((4, 384, 14, 14), (75264, 196, 14, 1), dtype=torch.float32, device='npu').uniform_(0, 1)
arg2 = torch.empty_strided((384, 1, 1), (1, 1, 1), dtype=torch.float32, device='npu').uniform_(0, 1)
arg3 = torch.empty_strided((384,), (1,), dtype=torch.float32, device='npu').uniform_(0, 1)
arg4 = torch.empty_strided((384,), (1,), dtype=torch.float32, device='npu').uniform_(0, 1)

model = TestModel().npu()

from torch_npu._inductor.mfusion import MFusionPatch
with MFusionPatch():
    compiled = torch.compile(model, backend="inductor", dynamic=False)
    out = compiled(arg0, arg1, arg2, arg3, arg4)
```

## 生成融合 kernel

启用 MFusion 后，`torch.compile` 会生成融合后的 kernel 代码。如需查看融合算子的详细信息，可通过设置环境变量 export TORCH_COMPILE_DEBUG=1 开启调试模式，相关信息会输出到torch_compile_debug目录下的output_code.py 文件中。

融合 kernel 通过 `@dvm.kernel(..., mfusion=True)` 装饰器标识：

```Python
@dvm.kernel(ktype='vector', dyn_shape=False, mfusion=True)
def mfusion_dvm_0(k):
    arg0 = k.load([32], dvm.float32)
    arg1 = k.load([32], dvm.float32)
    ...
    add_tensor = k.add(mul_tensor_1, reshape_default_3)
    k.store(add_tensor, dvm.float32)
```

## 使用约束

* 此环境变量仅可在 PyTorch2.7.1 和 PyTorch2.9.0 版本使用。
* 在 torch.compile 图编译后端为"Inductor"时生效
