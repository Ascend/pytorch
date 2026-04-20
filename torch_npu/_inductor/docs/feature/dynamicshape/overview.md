# shapeHandling 特性介绍

## 特性简介

ShapeHandling 是 PyTorch Ascend 扩展(针对华为 Ascend 加速卡的PyTorch 适配层)中的一个特性，用于优化动态形状输入下的模型编译和执行性能。它通过对输入张量的形状进行分档处理，减少编译次数，在结合aclgraph的场景减少图捕获次数，提高模型推理效率。

主要功能包括：
- 对张量的 batchsize 和 sequence length 维度进行分档处理。
- 自动 padding（填充）和 splitting（分割）操作。
- 支持自定义分档策略。
- 与 PyTorch Dynamo 集成，提供编译优化。
- 降低因输入 shape 变化导致的编译次数。
- 降低因输入 shape 变化导致的图捕获次数。

## 如何使用

### 基本使用方法

通过 `torch.compile` 的 `options` 参数启用 shape_handling 功能：

```python
import torch
import torch_npu

# 定义模型
def model_fn(A, B):
    return A + B

# 配置 shape_handling
shape_options = {
    "enable_shape_handling": True,
    "shape_handling_configs": [
        {
            "type": "BATCHSIZE",
            "dimensions": 0,
            "value": 0.0,
            "min_size": 1,
            "max_size": 1024,
            "policy": "TIMES",
        },
        {
            "type": "SEQLEN",
            "dimensions": [1],
            "value": 0.0,
            "min_size": 1,
            "max_size": 1024,
            "policy": "TIMES",
        }
    ]
}

# 编译模型
compiled_fn = torch.compile(
    model_fn, 
    backend='inductor', 
    dynamic=False,
    options=shape_options
)

# 使用编译后的模型
A = torch.randn(32, 128, device="npu")
B = torch.randn(32, 128, device="npu")
out = compiled_fn(A, B)
```


## 配置参数说明


###  `torch.compile(..., options=...)` 新增字段
| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `enable_shape_handling` | `bool` | 开启时必填 | `False` | 是否开启 shape handling。 |
| `shape_handling_configs` | `list[dict]` | 开启时必填 | `None` | shape 规则列表。最多支持两个维度类型（`BATCHSIZE`、`SEQLEN`）。 |
| `shape_handling_dict` | `dict` | 否 | `None` | 自定义预处理/后处理函数集合。 |

#### `shape_handling_configs` 
| 字段 | 类型 | 是否必填 | 默认值 | 说明                                        |
|---|---|---|---|-------------------------------------------|
| `type` | `str` | 是 | 无 | 维度类型：`BATCHSIZE` 或 `SEQLEN`。              |
| `dimensions` | `int` 或 `list[int]` | 否 | `BATCHSIZE->[0]`，`SEQLEN`按规则推导 | 目标维度下标。`SEQLEN` 支持按 Tensor 分别指定。          |
| `indices` | `list[int]` | 否 | `[]` | 应用到哪些输入 Tensor。空列表表示运行时自动推导可处理 Tensor。    |
| `value` | `int`/`float` | 否 | `0.0` | pad 填充值。                                  |
| `gears` | `list[int]` | 否 | `[]` | 明确指定档位。非空时优先于 `min_size/max_size/policy`。 |
| `min_size` | `int` | 否 | `1` | 自动生成 gear 的最小值。                           |
| `max_size` | `int` | 否 | `1024` | 自动生成 gear 的最大值。                           |
| `policy` | `str` | 否 | `TIMES` | gear 生成策略。当前 `min/max` 路径仅支持 `TIMES`。     |

#### `shape_handling_dict` 
| 字段 | 推荐签名 | 说明 |
|---|---|---|
| `trans_pre_fn` | `(*args, **kwargs) -> list[torch.Tensor]` | transform 前，将调用输入转换成 Tensor 列表。 |
| `trans_post_fn` | `(list[list[torch.Tensor]]) -> tuple[list[tuple], list[dict]]` | transform 后，将分组 Tensor 还原为“多组 args/kwargs”。 |
| `re_pre_fn` | `(list[Any]) -> list[list[torch.Tensor]]` | recover 前，将多组结果转回 Tensor 组。 |
| `re_post_fn` | `(list[torch.Tensor]) -> Any` | recover 后，恢复成最终返回结构。 |

### 维度类型说明

- **BATCHSIZE**: 批次维度，所有受影响的张量必须共享相同的批次维度
- **SEQLEN**: 序列长度维度，可以为不同的张量指定不同的序列维度位置

### 档位生成策略

- **TIMES**: 根据 min_size 和 max_size 生成 2 的整数幂档位
- **CUSTOM**: 使用自定义档位（通过 gears 参数指定）

## 特性开启前后对比

### 开启前

当不使用 shape_handling 时，不同形状的输入会触发多次编译：

```python
import torch
import torch_npu

compiled_fn = torch.compile(model_fn, backend='inductor', dynamic=False)

# 运行不同形状，会触发多次编译
test_shapes = [(3, 20), (4, 20), (5, 20), (6, 20)]
for shape in test_shapes:
    A = torch.randn(shape, device="npu")
    B = torch.randn(shape, device="npu")
    out = compiled_fn(A, B)

# 编译次数: 4
```

### 开启后

当使用 shape_handling 时，相同档位的输入共享编译结果：

```python
import torch
import torch_npu

compiled_fn = torch.compile(model_fn, backend='inductor', dynamic=False, options=shape_options)

# 运行不同形状，只触发 2 次编译
test_shapes = [(3, 32), (4, 32), (5, 32), (6, 32)]
for shape in test_shapes:
    A = torch.randn(shape, device="npu")
    B = torch.randn(shape, device="npu")
    out = compiled_fn(A, B)

# 编译次数: 2
```

## 高级功能

### 自定义预处理和后处理函数

可以通过 `transform_pre_fn`、`transform_post_fn`、`recover_pre_fn` 和 `recover_post_fn` 参数来自定义预处理和后处理逻辑：

```python
def custom_transform_pre_fn(*args, **kwargs):
    # 自定义预处理逻辑
    return [args[0], args[1]]

def custom_transform_post_fn(trans_outputs):
    # 自定义后处理逻辑
    return zip(*trans_outputs)

shape_handling = NPUShapeHandling(
    configs=configs,
    transform_pre_fn=custom_transform_pre_fn,
    transform_post_fn=custom_transform_post_fn
)
```

## 注意事项

1. ShapeHandling 目前仅支持最多两个维度的处理（BATCHSIZE 和 SEQLEN）。
2. 对于 BATCHSIZE 类型，dimensions 列表中只有第一个元素会被使用。
3. 对于 SEQLEN 类型，如果提供了单个维度值，会自动应用到所有受影响的张量。
4. 当输入形状超过 max_size 时，会自动进行分割处理。
5. 当输入形状不在任何档位时，会自动 padding 到最近的上一个档位。

## 性能优势

- **减少编译次数**：相同档位的输入共享编译结果，避免重复编译。
- **提高推理效率**：通过分档处理，优化计算资源利用。
- **支持动态形状**：即使输入形状变化，也能保持高效执行。
- **灵活配置**：支持自定义档位和处理策略，适应不同场景需求。

## 适用场景

- 模型推理时输入形状经常变化的场景。
- 批处理大小不固定的场景。
- 序列长度变化较大的场景。
- 对编译时间和推理性能有较高要求的场景。

## 示例：不同编译模式对比

| 编译模式 | 编译次数（4个不同形状输入，共享两个档位） | 特点 |
|---------|---------------------------|------|
| 静态形状 (dynamic=False) | 4 | 每个形状都需要编译 |
| 静态形状 + shape_handling | 2 | 相同档位共享编译结果 |
| aclgraph 单独使用 | 4 | 每个形状都需要重新捕获 aclgraph |
| aclgraph + shape_handling | 2 | 相同档位共享 aclgraph |

## 与 aclgraph（NPUGraph）的结合使用

### aclgraph 简介

aclgraph（NPUGraph）是一种硬件级下沉优化技术，通过捕获一系列 NPU 操作组成静态图缓存在 NPU 设备上，一次捕获、多次复跑，避免重复的 kernel 启动开销。它通过 `torch.compile(backend="npugraphs")` 启用，适合输入形状固定的场景。

### 结合使用的优势

当 shape_handling 与 aclgraph 结合使用时，可以获得以下优势：

1. **扩展 aclgraph 的适用场景**：aclgraph 原本要求输入形状完全固定，而 shape_handling 可以将不同但相近的形状映射到相同的档位，使得 aclgraph 能够处理更多的输入形状。

2. **减少 aclgraph 捕获次数**：通过分档处理，相同档位的输入可以共享同一个 aclgraph，减少重复捕获的开销。

3. **保持 aclgraph 的性能优势**：在每个档位内部，aclgraph 仍然可以发挥其一次捕获、多次复跑的性能优势。

### 使用示例

```python
import torch
import torch_npu

# 定义模型
def model_fn(A, B):
    return A + B

# 配置 shape_handling
shape_options = {
    "enable_shape_handling": True,
    "shape_handling_configs": [
        {
            "type": "BATCHSIZE",
            "dimensions": 0,
            "min_size": 1,
            "max_size": 1024,
            "policy": "TIMES",
        }
    ]
}

# 使用 aclgraph 后端编译模型
compiled_fn = torch.compile(
    model_fn, 
    backend='npugraphs',  # 使用 aclgraph 后端
    options=shape_options
)

# 运行不同形状的输入，相同档位的输入会共享 aclgraph
test_shapes = [(3, 20), (4, 20), (5, 20), (6, 20)]
for shape in test_shapes:
    A = torch.randn(shape, device="npu")
    B = torch.randn(shape, device="npu")
    out = compiled_fn(A, B)
    print(f"Input shape: {shape}, Output shape: {out.shape}")
```

### 对比用例

#### 不使用 shape_handling 的情况

```python
# 不使用 shape_handling
compiled_fn = torch.compile(model_fn, backend='npugraphs')

# 运行不同形状的输入，每个形状都会触发新的 aclgraph 捕获
test_shapes = [(3, 20), (4, 20), (5, 20), (6, 20)]
for shape in test_shapes:
    A = torch.randn(shape, device="npu")
    B = torch.randn(shape, device="npu")
    out = compiled_fn(A, B)  # 每次都会重新捕获 aclgraph

# 捕获次数: 4
```

#### 使用 shape_handling 的情况

```python
# 使用 shape_handling
compiled_fn = torch.compile(model_fn, backend='npugraphs', options=shape_options)

# 运行不同形状的输入，相同档位的输入共享 aclgraph
test_shapes = [(3, 20), (4, 20), (5, 20), (6, 20)]
for shape in test_shapes:
    A = torch.randn(shape, device="npu")
    B = torch.randn(shape, device="npu")
    out = compiled_fn(A, B)  # 相同档位的输入共享 aclgraph

# 捕获次数: 2 (档位配置同前面用例)
```

### 性能对比

| 配置 | 捕获次数（4个不同形状输入） | 首次执行时间 | 后续执行时间 | 适用场景 |
|------|---------------------------|-------------|-------------|----------|
| aclgraph 单独使用 | 4 | 较慢（每次都要捕获） | 快（复跑） | 输入形状固定的场景 |
| aclgraph + shape_handling | 2 | 较慢（仅首次捕获） | 快（复跑） | 输入形状在一定范围内变化的场景 |

### 注意事项

1. 当使用 aclgraph 后端时，shape_handling 会在捕获 aclgraph 之前对输入进行处理，确保相同档位的输入具有相同的形状。

2. 对于 aclgraph 后端，建议将 `dynamic` 参数设置为 `False`，因为 aclgraph 本身不支持动态形状。

3. 结合使用时，需要合理配置 shape_handling 的档位，确保相同档位内的形状变化不会影响模型的正确性。

## 总结

ShapeHandling 是一个强大的特性，可以显著减少动态形状输入下的编译次数，提高模型推理效率。通过合理配置分档策略，可以在保持模型正确性的同时，获得更好的性能表现。

特别地，当与 aclgraph（NPUGraph）结合使用时，shape_handling 可以扩展 aclgraph 的适用场景，使其能够处理更多的输入形状，同时保持 aclgraph 的性能优势。这对于输入形状在一定范围内变化的场景尤为有用。