# 简介
Torch 的 Inductor-MLIR 编译优化是基于 torch.compile 接口、融合 MLIR（Multi-Level Intermediate Representation）技术的新一代后端优化方案。Inductor-MLIR 是 PyTorch 编译体系对原生 Inductor 后端的扩展升级，在保留 torch.compile 动态图捕获 + 静态图优化 + 高效代码生成的核心逻辑的基础上，借助 MLIR 的多层级、跨架构编译能力，适配多类硬件架构的深度优化需求，显著提升不同算力平台下模型训练与推理的编译灵活性和性能上限。
Inductor-MLIR 包含如下核心组件：
|组件|	定位	|作用|
|-|-|-|
|Dynamo	|前端编译器（代码转换器）	|延续 TorchDynamo 的核心能力，JIT 将用户的 eager（动态图）代码编译为 FX Graph，完成动态图到静态中间表示的捕获，为 Inductor-MLIR 后端提供统一的输入，保证前端编译逻辑的一致性|
|Inductor-MLIR |核心编译器（多层级 IR 优化器）	|作为 Inductor-MLIR 的核心层，将 FX Graph 转换为 MLIR 多层级中间表示（IR）；通过跨层级的优化策略（如算子拆分与融合、数据类型优化、计算图化简），兼顾编译灵活性与优化深度，适配不同硬件的编译特性|
|MLIR CodeGen	|代码生成器（硬件感知代码生成器）	|接收优化后的 MLIR IR，针对目标硬件生成专属的高性能底层代码|

# 使用指导
## 环境配置

* torch_npu：2.6.0或之后版本的torch_npu支持torch.compile接口；
* torch-mlir：Inductor后端的MLIR模式依赖torch-mlir。用户可以从[LINK](https://repo.oepkgs.net/ascend/pytorch/vllm/torch/)获取torch-mlir软件包;

## 使用方式
MLIR使用方式有三种, 任选一种即可调用

1. config导入（torch.compile之前导入）
```
torch._inductor.config.npu_backend == "mlir"
torch.compile(op_calc)(x)
```

2. 环境变量导入（npu初始化之前导入）

```
import os
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'
torch.compile(op_calc)(x)
```
3. options导入
```
torch.compile(op_calc,options={"npu_backend":"mlir"})(x)
```
## 示例

```
import os
# 环境变量调用需要在torch_npu初始化之前
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
def op_calc (x , y):
    return x * y
x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
std_out = op_calc(x, y)

# options调用，修改compile参数
compile_func = torch.compile(op_calc, options={"npu_backend": "mlir"})
compile_out, codes = run_and_get_code(compile_func,x,y)
print(codes[0])
```
## 输出mlir融合算子
codes 为捕获到的 MLIR 输出代码片段，该代码中包含 call 函数与 benchmark_compiled_model 函数，其中 call 函数内部封装了 MLIR 编译生成的算子逻辑。
```
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
其中mlir_fused_mul_0.run 是 MLIR 编译生成的核心执行接口，用于在指定 NPU 设备和计算流上执行融合乘法算子计算，接收输入张量并将计算结果写入输出张量，是 MLIR 对原始乘法逻辑优化后的底层实现。