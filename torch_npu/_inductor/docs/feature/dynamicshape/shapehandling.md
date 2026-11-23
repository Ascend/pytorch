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

### `torch.compile(..., options=...)` 新增字段

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
test_shapes = [(3, 20), (4, 20), (5, 20), (6, 20)]
for shape in test_shapes:
    A = torch.randn(shape, device="npu")
    B = torch.randn(shape, device="npu")
    out = compiled_fn(A, B)

# 编译次数: 2
```

## 高级功能

### 自定义预处理和后处理函数

#### NPUShapeHandling 类

`NPUShapeHandling` 是 shape handling 的核心实现类，Python 侧封装位于 `torch_npu/utils/_dynamo.py`，底层由 `torch_npu._C._NPUShapeHandling` 实现，并通过 `torch_npu._inductor` 对外导出：

```python
from torch_npu._inductor import NPUShapeHandling
# 等价的访问方式：torch_npu._inductor.NPUShapeHandling
```

构造参数如下：

| 参数 | 类型 | 是否必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `configs` | `list[dict]` | 否 | 处理 dim 0 的 `BATCHSIZE` 默认规则 | 分档规则列表，字段与 `shape_handling_configs` 完全一致（见“配置参数说明”）。为空时采用延迟初始化，在首次 `transform` 时根据实际输入推导 `indices`。 |
| `transform_pre_fn` | `Callable[..., list[torch.Tensor]]` | 否 | `None` | transform 前处理。入参为 `(*args, **kwargs)`，需返回参与分档的 Tensor 列表。未设置时自动展平输入结构并取出其中所有 Tensor。设置此项时需同时提供 `transform_post_fn`。 |
| `transform_post_fn` | `Callable[[list[list[torch.Tensor]]], tuple[list[tuple], list[dict]]]` | 否 | `None` | transform 后处理。将每组 Tensor 还原为一组调用参数，返回 `(args_list, kwargs_list)`。未设置时按原始 args/kwargs 结构自动还原，但仅在未设置 `transform_pre_fn` 时可用。 |
| `recover_pre_fn` | `Callable[[list[Any]], list[list[torch.Tensor]]]` | 否 | `None` | recover 前处理。将每组执行结果转换为 `recover` 所需的 Tensor 分组。未设置时自动展平各组输出。设置此项时需同时提供 `recover_post_fn`。 |
| `recover_post_fn` | `Callable[[list[torch.Tensor]], Any]` | 否 | `None` | recover 后处理。将裁剪、拼接后的 Tensor 列表转换为最终返回结构。未设置时按原始输出结构自动还原，但仅在未设置 `recover_pre_fn` 时可用。 |

#### 自定义函数的生效时机

四个函数分别挂接在“分档（transform）”和“还原（recover）”流程上，执行顺序如下：

```text
transform_hook(*args, **kwargs):
  ① transform_pre_fn(*args, **kwargs)      -> list[Tensor]
  ② transform(tensors)                     -> list[list[Tensor]]   # 按档位填充，处理批次维时，超过最大档位则分片
  ③ transform_post_fn(trans_outputs)       -> (args_list, kwargs_list)

recover_hook(groups):
  ① recover_pre_fn(groups)                 -> list[list[Tensor]]
  ② recover(tensor_groups)                 -> list[Tensor]         # 去除 padding、拼接分片
  ③ recover_post_fn(re_outputs)            -> 最终返回结构
```

> 设置自定义前处理时，须同时提供对应的后处理；仅自定义后处理时，前处理使用默认逻辑。
>
> 命名对应关系：通过 `torch.compile` 使用时，四个函数放在 `shape_handling_dict` 中，键名为 `trans_pre_fn`、`trans_post_fn`、`re_pre_fn`、`re_post_fn`；直接构造 `NPUShapeHandling` 时，参数名为 `transform_pre_fn`、`transform_post_fn`、`recover_pre_fn`、`recover_post_fn`。两套名称一一对应。

#### 示例 1：直接使用 NPUShapeHandling

自定义函数会在输入/输出不是标准 Tensor 容器时使用。下面示例不设置自定义函数，直接实例化 `NPUShapeHandling`，用默认钩子手动调用 `transform_hook` / `recover_hook` 完成一次完整的分档执行流程：

```python
import torch
import torch_npu
from torch_npu._inductor import NPUShapeHandling

# 分档规则：按 batch 维（dim 0）分档，档位为 [4, 8]
configs = [
    {
        "type": "BATCHSIZE",
        "dimensions": [0],
        "indices": [],      # 自动推导可以分档的 Tensor，无对应维度的 Tensor 会被跳过
        "value": 0.0,       # padding 填充值
        "gears": [4, 8],    # 显式指定档位
    }
]

handler = NPUShapeHandling(configs=configs)

def model_fn(a, b):
    return a + b

a = torch.randn(3, 20, device="npu")
b = torch.randn(3, 20, device="npu")

# ① 分档：向上取到不小于输入的档位（3 -> 4），并还原成“逐组的 args/kwargs”
new_args, new_kwargs = handler.transform_hook(a, b)

# ② 逐组直接执行模型函数（本例未使用 torch.compile）
outputs = [model_fn(*args, **kwargs) for args, kwargs in zip(new_args, new_kwargs)]

# ③ 还原：去除 padding，恢复为原始 shape
out = handler.recover_hook(outputs)
print(out.shape)  # torch.Size([3, 20])
```

#### 示例 2：通过 torch.compile 注入自定义函数

若模型入参包含非 Tensor 参数或自定义结构，自动展平无法覆盖时，可通过 `shape_handling_dict` 注入自定义函数。下面示例将缩放系数固定为 `SCALE`，由 `trans_post_fn` 为每组传入：

```python
import torch
import torch_npu

configs = [
    {
        "type": "BATCHSIZE",
        "dimensions": [0],
        "indices": [],
        "value": 0.0,
        "gears": [4, 8],
    }
]

SCALE = 2.0

def model_fn(tensors, scale):
    return [t * scale for t in tensors]

# transform 前：只把 Tensor 列表交给分档逻辑
def trans_pre_fn(tensors, scale):
    return list(tensors)

# transform 后：重新组装每组的位置参数与关键字参数
def trans_post_fn(trans_outputs):
    new_args = [([*group],) for group in trans_outputs]
    new_kwargs = [{"scale": SCALE} for _ in trans_outputs]
    return new_args, new_kwargs

# recover 前：将每组的执行结果转回 Tensor 分组
def re_pre_fn(outputs):
    return [list(out) for out in outputs]

# recover 后：整理为最终返回结构
def re_post_fn(re_outputs):
    return list(re_outputs)

shape_options = {
    "enable_shape_handling": True,
    "shape_handling_configs": configs,
    "shape_handling_dict": {
        "trans_pre_fn": trans_pre_fn,
        "trans_post_fn": trans_post_fn,
        "re_pre_fn": re_pre_fn,
        "re_post_fn": re_post_fn,
    },
}

compiled_fn = torch.compile(
    model_fn,
    backend="inductor",
    dynamic=False,
    options=shape_options,
)

for batch in (3, 5):
    tensors = [torch.randn(batch, 16, device="npu") for _ in range(2)]
    out = compiled_fn(tensors, SCALE)
    print(batch, [t.shape for t in out])  # 3 [torch.Size([3, 16]), ...]；5 [torch.Size([5, 16]), ...]
```

## 注意事项

1. ShapeHandling 目前仅支持最多两个维度的处理（BATCHSIZE 和 SEQLEN）。
2. 对于 BATCHSIZE 类型，dimensions 列表中只有第一个元素会被使用。
3. 对于 SEQLEN 类型，如果提供了单个维度值，会自动应用到所有受影响的张量。
4. 批次维（`BATCHSIZE`）超过最大档位时会自动分片，序列维（`SEQLEN`）超过最大档位时会报错。显式设置 `gears` 时取其中最大值，否则取 `max_size`。
5. 待处理的维度未命中档位且存在更大的档位时，会填充到其中最小的档位。

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
