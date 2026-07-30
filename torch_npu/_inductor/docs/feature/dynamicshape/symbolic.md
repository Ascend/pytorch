# Symbolic 符号化特性介绍

## 特性简介

Symbolic（符号化）是 PyTorch Ascend 扩展中用于处理动态形状输入的核心特性。通过使用符号变量（SymInt/SymFloat）表示编译时未知的维度，编译器可以在运行时根据实际输入形状动态适配 kernel 实现，实现"一次编译，多种形状复用"的编译优化效果。

主要功能包括：

- 支持输入形状在运行时变化的动态 shape 场景
- 通过符号变量技术减少因形状变化导致的重复编译
- 与 PyTorch Dynamo 深度集成，提供透明的编译优化
- 优化动态形状下的 kernel 选择和资源分配

## 基本概念

### 符号变量

符号变量（Symbolic Variable）是 PyTorch 编译框架中用于表示未知形状维度的特殊类型。与静态编译中使用具体整数值不同，符号变量允许编译器在不知道具体数值的情况下进行图分析和优化。

```python
import torch
import torch_npu

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
- 大语言模型推理（BatchSize 和 SequenceLength 经常变化）
- 多模态模型（处理不同分辨率的图像或不同长度的文本）
- API 服务或在线推理系统（需要处理用户请求的不同尺寸输入）

#### dynamic=None（自动模式）

自动模式由框架自动判断是否启用动态 shape 支持。当设置为 `None` 时，PyTorch 编译器会根据模型特征和输入情况自动选择最优的编译策略。

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
- 迁移 PyTorch 模型到 Ascend NPU，希望开箱即用

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

## 与 shapeHandling 的对比

### 设计理念

| 特性 | 设计理念 | 核心方法 |
|------|----------|----------|
| **Symbolic** | 符号化抽象 | 使用符号变量表示未知维度，保持编译灵活性 |
| **ShapeHandling** | 分档处理 | 将动态形状映射到固定档位，减少编译开销 |

### 使用场景

- **Symbolic 适用场景**：
  - 输入形状完全动态，无法预知所有可能值
  - 需要支持真正的运行时形状变化
  - 追求编译结果的最大复用性

- **ShapeHandling 适用场景**：
  - 输入形状在一定范围内变化
  - 可以接受 padding/splitting 操作
  - 希望减少编译次数，提高推理效率

### 性能对比

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

### 2. 调试符号化问题

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

- **大语言模型推理**：BatchSize 和 SequenceLength 经常变化的场景
- **实时推理系统**：需要处理不同尺寸输入的在线服务
- **多模态模型**：处理不同分辨率或长度的图像和文本
- **动态批处理**：批次大小根据负载动态调整的系统
- **通用推理服务**：输入形状变化范围不确定，希望框架自动处理

## 总结

Symbolic 符号化特性为 PyTorch Ascend 扩展提供了强大的动态形状处理能力。通过使用符号变量技术，开发者可以在保持代码简洁的同时，实现高效的动态 shape 处理。建议根据具体应用场景选择合适的编译模式：

- `dynamic=False`：形状完全固定的场景
- `dynamic=True`：形状动态变化且无法预知的场景
- `dynamic=None`：不确定形状变化规律，希望框架自动处理的通用场景
