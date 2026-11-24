# Inductor-MLIR特性介绍

## 简介

PyTorch的Inductor-MLIR编译优化是基于torch.compile接口、融合MLIR（Multi-Level Intermediate Representation）技术的新一代后端优化方案。Inductor-MLIR是PyTorch编译体系对原生Inductor后端的扩展升级，在保留torch.compile动态图捕获 + 静态图优化 + 高效代码生成的核心逻辑的基础上，借助MLIR的多层级、跨架构编译能力，适配多类硬件架构的深度优化需求，显著提升不同算力平台下模型训练与推理的编译灵活性和性能上限。
Inductor-MLIR包含如下核心组件：

|组件| 定位 | 作用 |
|-|-|-|
|Dynamo |前端编译器（代码转换器） |延续TorchDynamo的核心能力，JIT将用户的eager（动态图）代码编译为FX Graph，完成动态图到静态中间表示的捕获，为Inductor-MLIR后端提供统一的输入，保证前端编译逻辑的一致性|
|Inductor-MLIR |核心编译器（多层级IR优化器） |作为Inductor-MLIR的核心层，将FX Graph转换为MLIR多层级中间表示（IR）；通过跨层级的优化策略（如算子拆分与融合、数据类型优化、计算图化简），兼顾编译灵活性与优化深度，适配不同硬件的编译特性|
|MLIR CodeGen |代码生成器（硬件感知代码生成器） |接收优化后的MLIR IR，针对目标硬件生成专属的高性能底层代码|

## 使用指导

### 环境配置

* torch_npu：2.6.0或之后版本的torch_npu支持torch.compile接口；
* torch-mlir：Inductor后端的MLIR模式依赖torch-mlir。用户可以从[torch-mlir软件包仓库](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/)获取torch-mlir软件包。

### 使用方式

MLIR使用方式有三种，任选一种即可调用。

1. config导入（torch.compile之前导入）

    ```python
    torch._inductor.config.npu_backend = "mlir"
    torch.compile(op_calc)(x)
    ```

2. 环境变量导入（NPU初始化之前导入）

    ```python
    import os
    os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'
    torch.compile(op_calc)(x)
    ```
    
3. options导入

    ```python
    torch.compile(op_calc,options={"npu_backend":"mlir"})(x)
    ```

### 示例

```python
import os
# 环境变量设置需要在torch_npu初始化之前
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'

import torch
import torch_npu
from torch._inductor.utils import run_and_get_code
import torch_npu._inductor
import torch
import triton

# config导入在compile执行之前
torch._inductor.config.npu_backend = "mlir"

# 定义模型
def op_calc(x,y):
    return x * y
x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
std_out = op_calc(x, y)

# options调用，修改compile参数
compile_func = torch.compile(op_calc, options={"npu_backend": "mlir"})
compile_out, codes = run_and_get_code(compile_func,x,y)
print(codes[0])
```

### 输出MLIR融合算子

codes为捕获到的MLIR输出代码片段，该代码中包含call函数与benchmark_compiled_model函数，其中call函数内部封装了MLIR编译生成的算子逻辑。

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

其中mlir_fused_mul_0.run是MLIR编译生成的核心执行接口，用于在指定NPU设备和计算流上执行融合乘法算子计算，接收输入张量并将计算结果写入输出张量，是MLIR对原始乘法逻辑优化后的底层实现。
