# Guard Filter

## 简介

`guard_filter_fn`是PyTorch Dynamo层的编译选项，作用于图捕获阶段，与下游编译后端无关。guard的生成和校验逻辑均由Dynamo统一管理，因此`guard_filter_fn`对所有后端通用。

PyTorch Dynamo（PyTorch的图捕获前端，负责将Python字节码转换为计算图）在每次进入编译入口时会生成一组guard（守卫条件），用于检测运行时状态是否发生变化。当guard失效时会触发重编译（recompilation），严重影响推理性能。

**表1** 支持的后端

| 编译后端 | 适用场景 | guard_filter_fn支持 |
|---|---|---|
| inductor | NPU/CPU推理与训练 | 支持 |
| npugraph_ex | NPU推理与训练 | 支持 |
| npugraph | NPU推理与训练 | 支持 |
| aot_eager | 调试用途 | 支持 |
| TorchAir-GE | NPU推理与训练 | 支持 |

由于guard机制完全在Dynamo层实现，切换后端不影响`guard_filter_fn`的行为。同一个filter函数可以在不同后端间复用。

## 使用场景

在NPU训练与推理场景中，可能存在以下几类guard频繁触发重编译，但实际运行时对应的状态并不会影响编译产物的正确性：

**表2** 重触发场景

| Guard类型 | 触发场景 |
|---|---|
| `DICT_VERSION` / `DICT_KEYS` / `DICT_KEYS_MATCH` / `DICT_CONTAINS` | HuggingFace `generation_config`、KV cache state dict、attention kwargs，每步version自增 |
| `TYPE_MATCH` / `OPTIONAL_TENSOR` | `past_key_values=None`（prefill）↔ `tuple`（decode）、`attention_mask=None` ↔ Tensor |
| `HASATTR` / `NOT_PRESENT_IN_GENERIC_DICT` | 特性开关、首次forward后才挂上的字段 |
| `GRAD_MODE` / `TORCH_FUNCTION_STATE` / `DEFAULT_DEVICE` / `DETERMINISTIC_ALGORITHMS` / `AUTOCAST_STATE` / `FSDP_TRAINING_STATE` | 进程级一次性配置，但每个编译入口都会生成guard |

原生PyTorch提供了`guard_filter_fn`编译选项，允许用户自定义过滤逻辑，选择性跳过不必要的guard。

> [!NOTE]
>
> `guard_filter_fn`具有unsafe语义，如果在运行时改变了被过滤guard对应的状态，编译产物将静默地产生错误结果。

使用前请确认：

1. 被过滤的状态在整个推理过程中确实不会发生语义上的变化。
2. 通过parity测试验证输出正确性。
3. 通过recompile count测试验证重编译已消除。

## 使用指导

接口原型：

```python
torch.compile(model, options={"guard_filter_fn": filter_fn})
```

参数说明：

`filter_fn`签名：

```python
def filter_fn(entries: list) -> list[bool]:
    """
    参数:
        entries: guard条目列表，每个条目包含以下属性：
            - guard_type (str): guard类型，如"DICT_VERSION"、"GRAD_MODE"等
            - name (str): guard关联的变量名
            - is_global (bool): 是否为全局变量的guard
            - value: guard关联的值（可选）
    返回:
        与entries等长的布尔列表，True表示保留该guard，False表示过滤掉
    """
```

> [!NOTICE]
> 
> 原生PyTorch还提供了内置的helper，例如：
>
> - `torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe`：跳过内置nn.Module（如`Linear`、`Conv2d`等PyTorch自带模块）属性变化的guard。适用于模型权重在推理期间不会修改的场景，避免因module属性version自增导致重编译。
> - `torch.compiler.skip_guard_on_all_nn_modules_unsafe`：跳过所有nn.Module（包括用户自定义模块）属性变化的guard。覆盖范围比上一个更广，适用于整个模型在推理期间完全静态的场景。
> - `torch.compiler.skip_guard_on_globals_unsafe`：跳过全局变量的guard。适用于全局配置（如feature flag、debug开关）在推理期间不会改变的场景，避免因全局变量version变化触发重编译。

## 使用示例

示例1：过滤字典版本guard

对应表2中`DICT_VERSION`/`DICT_KEYS`/`DICT_KEYS_MATCH`/`DICT_CONTAINS`类guard。适用于HuggingFace模型推理中`generation_config`、KV cache state dict等字典每步version自增导致重编译的场景：

```python
import torch
import torch_npu

_DICT_GUARD_TYPES = frozenset({
    "DICT_VERSION", "DICT_KEYS", "DICT_KEYS_MATCH", "DICT_CONTAINS",
})

def filter_dict_guards(entries):
    return [entry.guard_type not in _DICT_GUARD_TYPES for entry in entries]

model = MyModel().npu()  # 注意：此处MyModel仅为示例占位类，实际使用时需要替换为自定义的真实模型类
compiled = torch.compile(model, options={"guard_filter_fn": filter_dict_guards})
```

示例2：过滤运行时状态guard

对应表2中`GRAD_MODE`/`TORCH_FUNCTION_STATE`/`DEFAULT_DEVICE`等进程级配置类guard。适用于推理管线中交替切换`torch.no_grad()` / `torch.enable_grad()`的场景：

```python
import torch
import torch_npu

_RUNTIME_STATE_GUARD_TYPES = frozenset({
    "GRAD_MODE", "TORCH_FUNCTION_STATE", "GLOBAL_STATE",
    "DEFAULT_DEVICE", "DETERMINISTIC_ALGORITHMS", "AUTOCAST_STATE",
    "FSDP_TRAINING_STATE",
})

def filter_runtime_state_guards(entries):
    return [entry.guard_type not in _RUNTIME_STATE_GUARD_TYPES for entry in entries]

model = MyModel().npu()  # 注意：此处MyModel仅为示例占位类，实际使用时需要替换为自定义的真实模型类
compiled = torch.compile(model, options={"guard_filter_fn": filter_runtime_state_guards})

# 切换grad mode不会触发重编译
with torch.no_grad():
    compiled(x)
with torch.enable_grad():
    compiled(x)  # 不会重编译
```

示例3：组合多类guard过滤

一次性过滤表2中多类guard（字典版本、可选类型、hasattr、运行时状态），适用于需要全面消除重编译的场景：

```python
import torch
import torch_npu

_FILTER_GUARD_TYPES = frozenset({
    # 字典版本
    "DICT_VERSION", "DICT_KEYS", "DICT_KEYS_MATCH", "DICT_CONTAINS",
    # 可选类型
    "TYPE_MATCH", "OPTIONAL_TENSOR",
    # hasattr
    "HASATTR", "NOT_PRESENT_IN_GENERIC_DICT",
    # 运行时状态
    "GRAD_MODE", "TORCH_FUNCTION_STATE", "GLOBAL_STATE",
    "DEFAULT_DEVICE", "DETERMINISTIC_ALGORITHMS", "AUTOCAST_STATE",
    "FSDP_TRAINING_STATE",
})

def npu_guard_filter(entries):
    return [entry.guard_type not in _FILTER_GUARD_TYPES for entry in entries]

model = MyModel().npu()  # 注意：此处MyModel仅为示例占位类，实际使用时需要替换为自定义的真实模型类
compiled = torch.compile(model, options={"guard_filter_fn": npu_guard_filter})
```

示例4：按变量名或属性过滤

对应表2中`TYPE_MATCH`/`OPTIONAL_TENSOR`类guard。当特定变量在prefill和decode阶段类型交替变化时，可按变量名精确过滤：

```python
import torch
import torch_npu

def filter_by_name(entries):
    return [
        not (entry.name == "y" and entry.value is None)
        for entry in entries
    ]

@torch.compile(fullgraph=True, options={"guard_filter_fn": filter_by_name})
def fn(x, y):
    if y is not None:
        x += y
    return x
```

示例5：过滤全局变量guard

对应表2中`HASATTR`/`NOT_PRESENT_IN_GENERIC_DICT`等与全局变量相关的guard。适用于全局特性开关、debug配置在推理期间不会改变的场景：

```python
import torch
import torch_npu

def filter_globals(entries):
    return [not entry.is_global for entry in entries]

model = MyModel().npu()  # 注意：此处MyModel仅为示例占位类，实际使用时需要替换为自定义的真实模型类
compiled = torch.compile(model, options={"guard_filter_fn": filter_globals})
```

示例6：结合内置helper使用

```python
import torch
import torch_npu

# 使用PyTorch内置的nn.Module guard过滤
model = MyModel().npu()  # 注意：此处MyModel仅为示例占位类，实际使用时需要替换为自定义的真实模型类
compiled = torch.compile(
    model,
    options={
        "guard_filter_fn": torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe
    },
)
```

## 调试与验证

- 确认重编译是否消除

  使用`torch.compiler.set_stance("fail_on_recompile")`验证：

  ```python
  compiled = torch.compile(model, options={"guard_filter_fn": npu_guard_filter})

  # 首次编译
  compiled(x)

  # 验证不会重编译
  with torch.compiler.set_stance("fail_on_recompile"):
      compiled(x)  # 如果重编译会抛出异常
  ```

- 查看guard日志

  通过环境变量开启guard日志，定位触发重编译的guard类型：

  ```bash
  TORCH_LOGS=guards,recompiles python your_script.py
  ```

- 验证输出正确性

  过滤guard后务必验证编译输出与eager模式一致：

  ```python
  model.eval()
  x = torch.randn(2, 8).npu()

  with torch.no_grad():
      eager_out = model(x)

  compiled = torch.compile(model, options={"guard_filter_fn": npu_guard_filter})
  with torch.no_grad():
      compiled_out = compiled(x)

  assert torch.allclose(eager_out, compiled_out, atol=1e-5)
  ```

## 约束说明

- PyTorch版本：必须为2.9.0及以上版本。
- TorchNPU版本：必须安装与PyTorch版本配套的版本，具体请参考[版本说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E7%9B%B8%E5%85%B3%E4%BA%A7%E5%93%81%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E8%AF%B4%E6%98%8E)。
