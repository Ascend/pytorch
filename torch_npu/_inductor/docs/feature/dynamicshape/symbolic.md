# Symbolic 符号化特性介绍

当模型输入的shape经常变化时，可以使用Symbolic符号化特性，让多种输入形状复用编译结果。本文介绍该特性的使用方式，以及动态shape分组autotune的配置和回退机制。

## 特性简介

Symbolic（符号化）是PyTorch Ascend扩展中用于处理动态shape输入的核心特性。通过使用符号变量（SymInt/SymFloat）表示编译时未知的维度，编译器可以在运行时根据实际输入形状动态适配kernel实现，实现“一次编译，多种形状复用”的编译优化效果。

主要功能包括：

- 支持输入形状在运行时变化的动态shape场景
- 通过符号变量技术减少因形状变化导致的重复编译
- 与PyTorch Dynamo深度集成，提供透明的编译优化
- 优化动态shape下的kernel选择和资源分配

## 基本概念

### 符号变量

符号变量（Symbolic Variable）是PyTorch编译框架中用于表示未知形状维度的特殊类型。与静态编译中使用具体整数值不同，符号变量允许编译器在不知道具体数值的情况下进行图分析和优化。

```python
import torch
import torch_npu

# 定义模型
def model_fn(x):
    return torch.relu(x)

# 使用 dynamic=True 启用符号化编译
compiled_fn = torch.compile(
    model_fn, 
    backend='inductor', 
    dynamic=True  # 启用符号化处理
)

# 即使输入形状变化，也不需要重新编译
x1 = torch.randn(32, 128, device="npu")
x2 = torch.randn(64, 256, device="npu")  # 不同形状，共享编译结果

out1 = compiled_fn(x1)
out2 = compiled_fn(x2)
```

### 动态编译模式

| 模式 | 参数设置 | 特点 | 适用场景 |
|------|----------|------|----------|
| 静态编译 | `dynamic=False` | 编译时确定所有形状 | 形状固定的场景 |
| 符号化编译 | `dynamic=True` | 使用符号变量处理未知形状 | 形状动态变化的场景 |
| 自动模式 | `dynamic=None` | 由框架自动判断 | 不确定形状变化规律的通用场景 |

## 如何使用

### 基本使用方法

通过 `torch.compile` 的 `dynamic` 参数启用符号化特性：

```python
import torch
import torch_npu

# 定义模型
def model_fn(A, B):
    return torch.matmul(A, B)

# 启用符号化编译
compiled_fn = torch.compile(
    model_fn, 
    backend='inductor', 
    dynamic=True
)

# 使用编译后的模型
A = torch.randn(32, 64, device="npu")
B = torch.randn(64, 128, device="npu")
out = compiled_fn(A, B)
```

### 编译模式详解

#### dynamic=False（静态编译）

静态编译模式在编译时确定所有张量的形状。适用于形状固定的场景，可以获得最佳的编译优化效果。

```python
import torch
import torch_npu

# 定义模型
def model_fn(A, B):
    return torch.matmul(A, B)

# 使用静态编译
compiled_fn = torch.compile(
    model_fn, 
    backend='inductor', 
    dynamic=False  # 静态编译
)

# 输入形状固定
A = torch.randn(32, 64, device="npu")
B = torch.randn(64, 128, device="npu")
out = compiled_fn(A, B)
```

**适用场景**：

- 模型推理时输入形状完全固定
- 对性能有极致要求，形状变化不频繁
- 部署环境资源有限，需要最小化运行时开销
- 生产环境的批量推理任务

#### dynamic=True（符号化编译）

符号化编译模式使用符号变量表示未知维度，允许运行时根据实际输入形状动态适配。

```python
import torch
import torch_npu

# 定义模型
def model_fn(x):
    return torch.nn.functional.relu(x)

# 使用符号化编译
compiled_fn = torch.compile(
    model_fn, 
    backend='inductor', 
    dynamic=True  # 符号化编译
)

# 测试不同形状的输入
shapes = [(16, 128), (32, 128), (64, 256), (128, 512)]
for shape in shapes:
    x = torch.randn(shape, device="npu")
    out = compiled_fn(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
```

**适用场景**：

- 输入形状在运行时动态变化，无法预知所有可能值
- 需要支持真正的运行时形状变化
- 大语言模型推理（BatchSize和SequenceLength经常变化）
- 多模态模型（处理不同分辨率的图像或不同长度的文本）
- API服务或在线推理系统（需要处理用户请求的不同尺寸输入）

#### dynamic=None（自动模式）

自动模式由框架自动判断是否启用动态shape支持。当设置为 `None` 时，PyTorch编译器会根据模型特征和输入情况自动选择最优的编译策略。

```python
import torch
import torch_npu

# 定义模型
def model_fn(x):
    return torch.nn.functional.relu(x)

# 使用自动模式
compiled_fn = torch.compile(
    model_fn, 
    backend='inductor', 
    dynamic=None  # 自动模式
)
```

**适用场景**：

- 不确定形状变化规律，希望框架自动处理
- 混合使用场景，部分模块形状固定，部分模块形状变化
- 快速原型开发和实验阶段，无需手动调优
- 通用推理服务，输入形状变化范围不确定
- 迁移PyTorch模型到Ascend NPU，希望开箱即用

**工作原理**：

- 框架会分析模型的计算图和依赖关系
- 对于形状敏感的算子，自动启用符号化处理
- 对于形状固定的算子，保持静态编译以获得最佳性能
- 平衡编译灵活性和运行时性能

**使用建议**：

```python
import torch
import torch_npu

# 场景1：开发调试阶段，使用自动模式快速验证
def model_fn(x):
    return torch.nn.functional.relu(x)

compiled_model = torch.compile(model_fn, backend='inductor', dynamic=None)

# 场景2：生产环境，根据实际形状特征选择固定模式
if is_production and shapes_are_fixed:
    compiled_model = torch.compile(model_fn, backend='inductor', dynamic=False)
else:
    compiled_model = torch.compile(model_fn, backend='inductor', dynamic=None)
```

## 动态shape分组autotune（Symbolic Grouped Autotune）

### 特性简介

在Symbolic（动态shape）编译场景下，同一个Triton kernel可能以多种不同的运行时shape被调用。默认的autotune流程会为每一种实际shape各自做一次benchmark选优，shape变化越多，重复编译和调优开销越大，动态shape下的内核运行性能下降明显。

动态shape分组autotune（grouped autotune）将运行时shape空间按shape特征（如动态轴的维度长度、归约维度乘积等）划分为多个分组（bucket），每个分组使用一个“代表shape”做一次autotune benchmark并选定最优config；运行时根据实际shape所处的分组直接复用该分组的调优结果，实现“一次调优，多种shape复用”。相比每种shape单独autotune，可显著降低动态shape场景下的重复编译和调优开销。

### 基本原理

- **分组（Grouping）**：Inductor-Ascend从kernel中提取分组特征（动态轴的维度长度、归约乘积等），为每个特征构造分桶边界（bucket），将可能的shape区间划分成若干个分组。
- **代表shape（Representative）**：为每个分组在其边界内选取一个代表shape，用于生成该分组的网格（grid）与平铺（tiling）参数并执行benchmark。
- **按组调优（Grouped Autotune）**：以代表shape为输入，对每个分组执行一次autotune benchmark，为该分组选定最优候选config与launcher。
- **运行时分发（Runtime Dispatch）**：kernel每次执行时，从实际传入的numel参数中提取特征值，通过bucketize映射到对应的分组（group_id），直接复用该分组预调的best launcher，并按策略实例化运行时block（如 `R0BLOCK_SUB`）完成启动。

### 适用场景

- 使用 `torch.compile(..., dynamic=True)` 符号化编译，且图中存在pointwise / reduction / persistent_reduction类Triton kernel的场景。
- 动态shape轴（如BatchSize、SequenceLength）在运行时会变化，希望降低重复编译和autotune开销的场景。
- 支持AOT（cpp wrapper）场景下的分组调度与kernel落盘复用。

### 环境变量控制

- `INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE`：分组autotune的总开关。
- `INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES`：控制哪些模板类型的kernel参与分组autotune。

### 使用约束与自动回退

- 仅对存在动态shape轴的Triton kernel生效，静态shape kernel行为不受影响。
- 仅对 `pointwise`、`reduction`、`persistent_reduction` 三类模板kernel生效，参与调优的模板类型可通过 `INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES` 控制。
- 当前为灰度开关。当分组计划不受支持（kernel访问签名不一致、无法构造分组代表等）或分组benchmark的显存占用超过预算时，会自动回退到普通autotune流程。

各环境变量的详细说明请参见：

- **[INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE](./INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE.md)**
- **[INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES](./INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES.md)**

## 与 ShapeHandling 的对比

### 设计理念

| 特性 | 设计理念 | 核心方法 |
|------|----------|----------|
| **Symbolic** | 符号化抽象 | 使用符号变量表示未知维度，保持编译灵活性 |
| **ShapeHandling** | 分档处理 | 将动态shape映射到固定档位，减少编译开销 |

### 使用场景

- **Symbolic适用场景**：
  - 输入形状完全动态，无法预知所有可能值
  - 需要支持真正的运行时形状变化
  - 追求编译结果的最大复用性

- **ShapeHandling适用场景**：
  - 输入形状在一定范围内变化
  - 可以接受padding/splitting操作
  - 希望减少编译次数，提高推理效率

### 编译结果复用情况对比

```python
import torch
import torch_npu

# 定义模型
def model_fn(A, B):
    return torch.matmul(A, B)

# 测试不同编译模式
test_shapes = [(16, 32), (32, 32), (64, 32), (128, 32), (256, 32)]

# 静态编译（dynamic=False）
static_compiled = torch.compile(model_fn, backend='inductor', dynamic=False)

# 符号化编译（dynamic=True）
symbolic_compiled = torch.compile(model_fn, backend='inductor', dynamic=True)

# 自动模式（dynamic=None）
auto_compiled = torch.compile(model_fn, backend='inductor', dynamic=None)

# shape_handling 编译
shape_options = {
    "enable_shape_handling": True,
    "shape_handling_configs": [{
        "type": "BATCHSIZE",
        "dimensions": 0,
        "min_size": 1,
        "max_size": 256,
        "policy": "TIMES"
    }]
}
shape_compiled = torch.compile(model_fn, backend='inductor', dynamic=False, options=shape_options)

# 运行测试
print("测试不同形状的输入复用情况：")
print(f"Static (dynamic=False): 每个形状都需要独立编译")
print(f"Symbolic (dynamic=True): 使用符号变量，形状变化时可能复用编译结果")
print(f"Auto (dynamic=None): 框架自动选择最优策略")
print(f"ShapeHandling: 将形状映射到档位，相同档位复用编译结果")
```

## 最佳实践

### 1. 选择合适的编译模式

```python
import torch
import torch_npu

def model(A, B):
    return torch.matmul(A, B)

# 场景1：形状完全固定
model_fixed = torch.compile(model, backend='inductor', dynamic=False)

# 场景2：形状动态变化，但变化范围可控
model_dynamic_controlled = torch.compile(
    model, 
    backend='inductor', 
    dynamic=True,
    options = {
        "enable_shape_handling": True,
        "shape_handling_configs": [{
        "type": "BATCHSIZE",
        "dimensions": 0,
        "min_size": 1,
        "max_size": 256,
        "policy": "TIMES"
        }]
    }
)

# 场景3：形状完全动态，无法预知
model_dynamic_full = torch.compile(model, backend='inductor', dynamic=True)

# 场景4：不确定形状变化规律，希望框架自动处理
model_auto = torch.compile(model, backend='inductor', dynamic=None)
```

### 2. 使用动态shape分组autotune

在动态shape（`dynamic=True`）编译场景下，若图中存在大量随BatchSize、SequenceLength等动态轴变化的逐点/归约类Triton Kernel，可通过开启分组autotune，用代表shape完成一次调优、供同组多种shape复用，从而降低重复编译和调优开销。

```bash
# 开启动态shape分组autotune（默认关闭）
export INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1

# 可选：限定参与分组的模板类型（默认 pointwise,reduction,persistent_reduction）
# export INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES=pointwise,reduction
```

```python
import torch
import torch_npu

def model_fn(x):
    return torch.nn.functional.relu(x).sum(dim=-1)

# 动态shape编译，BatchSize 运行时变化
compiled_fn = torch.compile(model_fn, backend='inductor', dynamic=True)

# 不同 BatchSize 下，分组autotune按shape自动分配分组并复用组内已调优config，
# 对同组shape无需重复编译和autotune
for bs in (16, 17, 32, 33, 64):
    x = torch.randn(bs, 128, device="npu")
    out = compiled_fn(x)
```

**建议**：当动态shape变化范围内出现大量相似shape时，开启分组autotune收益最明显；当分组计划不受支持或分组benchmark的显存占用超过预算时，框架会自动回退到普通autotune流程。

### 3. 调试符号化问题

```python
import torch
import torch_npu

# 1. 定义一个简单的模型函数
def model_fn(A, B):
    return torch.matmul(A, B)

# 2. 准备 NPU 上的测试数据
A = torch.randn((16, 128), device="npu")
B = torch.randn((128, 16), device="npu")

# 3. 示例一：使用 torch.compile 编译运行
compiled_fn = torch.compile(model_fn, backend='inductor', dynamic=True)
out = compiled_fn(A, B)
print("1. torch.compile 运行成功，输出形状:", out.shape)

# 4. 示例二：使用 torch._dynamo.export 导出计算图并打印
graph_module, _ = torch._dynamo.export(model_fn)(A, B)
print("\n2. torch._dynamo.export 导出成功，生成的图代码如下:")
print(graph_module.code)
```

## 注意事项

1. **性能权衡**：符号化编译虽然提高了灵活性，但可能带来一定的运行时开销。

2. **兼容性问题**：部分复杂模型可能在符号化编译时遇到问题，建议进行充分测试。

3. **模式选择**：建议在开发阶段使用 `dynamic=None` 自动模式，在生产环境根据实际形状特征选择合适的固定模式。

4. **混合使用**：可以根据模型不同模块的特点，混合使用不同的编译模式。

## 适用场景

- **大语言模型推理**：BatchSize和SequenceLength经常变化的场景
- **实时推理系统**：需要处理不同尺寸输入的在线服务
- **多模态模型**：处理不同分辨率或长度的图像和文本
- **动态批处理**：批次大小根据负载动态调整的系统
- **通用推理服务**：输入形状变化范围不确定，希望框架自动处理

## 总结

Symbolic符号化特性为PyTorch Ascend扩展提供了强大的动态shape处理能力。通过使用符号变量技术，开发者可以在保持代码简洁的同时，实现高效的动态shape处理。建议根据具体应用场景选择合适的编译模式：

- `dynamic=False`：形状完全固定的场景
- `dynamic=True`：形状动态变化且无法预知的场景
- `dynamic=None`：不确定形状变化规律，希望框架自动处理的通用场景
