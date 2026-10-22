# Guard Filter

## Introduction

`guard_filter_fn` is a compilation option at the PyTorch Dynamo layer. It takes effect in the graph capture phase and is independent of downstream compilation backends. The generation and verification logic of guards is uniformly managed by Dynamo, so `guard_filter_fn` applies to all backends.

PyTorch Dynamo (PyTorch's graph capture frontend, responsible for converting Python bytecode into computation graphs) generates a set of guards (guard conditions) each time a compilation entry point is entered, to detect whether the runtime state has changed. When a guard becomes invalid, recompilation is triggered, which seriously affects inference performance.

**Table 1** Supported backends

| Compilation Backend | Applicable Scenarios | guard_filter_fn Support |
|---|---|---|
| inductor | NPU/CPU inference and training | Supported |
| npugraph_ex | NPU inference and training | Supported |
| npugraph | NPU inference and training | Supported |
| aot_eager | Debugging purposes | Supported |
| TorchAir-GE | NPU inference and training | Supported |

Since the guard mechanism is implemented entirely at the Dynamo layer, switching backends does not affect the behavior of `guard_filter_fn`. The same filter function can be reused across different backends.

## Application Scenarios

In NPU training and inference scenarios, the following types of guards may frequently trigger recompilation, even though the corresponding runtime states do not affect the correctness of the compilation artifacts:

**Table 2** Re-triggering scenarios

| Guard Type | Trigger Scenario |
|---|---|
| `DICT_VERSION` / `DICT_KEYS` / `DICT_KEYS_MATCH` / `DICT_CONTAINS` | HuggingFace `generation_config`, KV cache state dict, and attention kwargs, where version increments at each step |
| `TYPE_MATCH` / `OPTIONAL_TENSOR` | `past_key_values=None` (prefill) ↔ `tuple` (decode), `attention_mask=None` ↔ Tensor |
| `HASATTR` / `NOT_PRESENT_IN_GENERIC_DICT` | Feature switches and fields attached only after the first forward call |
| `GRAD_MODE` / `TORCH_FUNCTION_STATE` / `DEFAULT_DEVICE` / `DETERMINISTIC_ALGORITHMS` / `AUTOCAST_STATE` / `FSDP_TRAINING_STATE` | Process-level one-time configurations, but a guard is generated at every compilation entry |

Native PyTorch provides the `guard_filter_fn` compilation option, allowing users to customize the filtering logic and selectively skip unnecessary guards.

> [!NOTE]
>
> `guard_filter_fn` has unsafe semantics. If the state corresponding to a filtered guard is changed at runtime, the compilation artifacts will silently produce incorrect results.

Before use, confirm the following:

1. The filtered states will indeed not change semantically during the entire inference process.
2. Verify output correctness through parity tests.
3. Verify through recompile count tests that recompilation has been eliminated.

## Usage Guide

Prototype:

```python
torch.compile(model, options={"guard_filter_fn": filter_fn})
```

Parameters

`filter_fn` signature:

```python
def filter_fn(entries: list) -> list[bool]:
    """
    Parameters:
        entries: list of guard entries, each entry containing the following attributes:
            - guard_type (str): guard type, such as "DICT_VERSION" and "GRAD_MODE"
            - name (str): variable name associated with the guard
            - is_global (bool): whether the guard is for a global variable
            - value: value associated with the guard (optional)
    Returns:
        A boolean list of the same length as entries; True means keep the guard, and False means filter it out
    """
```

> [!NOTICE]
> 
> Native PyTorch also provides built-in helpers, for example:
>
> - `torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe`: Skips guards for attribute changes of built-in nn.Modules (such as PyTorch's built-in modules like `Linear` and `Conv2d`). It is suitable for scenarios where model weights are not modified during inference, avoiding recompilation caused by version increments of module attributes.
> - `torch.compiler.skip_guard_on_all_nn_modules_unsafe`: Skips guards for attribute changes of all nn.Modules (including user-defined modules). It covers a wider range than the previous one and is suitable for scenarios where the entire model is completely static during inference.
> - `torch.compiler.skip_guard_on_globals_unsafe`: Skips guards for global variables. It is suitable for scenarios where global configurations (such as feature flags and debug switches) do not change during inference, avoiding recompilation triggered by version changes of global variables.

## Usage Examples

Example 1: Filter dictionary version guards

This corresponds to the `DICT_VERSION`/`DICT_KEYS`/`DICT_KEYS_MATCH`/`DICT_CONTAINS` guards in Table 2. It is suitable for scenarios in HuggingFace model inference where dictionaries such as `generation_config` and the KV cache state dict increment their version at each step, causing recompilation:

```python
import torch
import torch_npu

_DICT_GUARD_TYPES = frozenset({
    "DICT_VERSION", "DICT_KEYS", "DICT_KEYS_MATCH", "DICT_CONTAINS",
})

def filter_dict_guards(entries):
    return [entry.guard_type not in _DICT_GUARD_TYPES for entry in entries]

model = MyModel().npu()  # Note: MyModel here is only a sample placeholder class; replace it with your own real model class in actual use
compiled = torch.compile(model, options={"guard_filter_fn": filter_dict_guards})
```

Example 2: Filter runtime state guards

This corresponds to process-level configuration guards such as `GRAD_MODE`/`TORCH_FUNCTION_STATE`/`DEFAULT_DEVICE` in Table 2. It is suitable for scenarios where `torch.no_grad()` / `torch.enable_grad()` are alternately switched in the inference pipeline:

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

model = MyModel().npu()  # Note: MyModel here is only a sample placeholder class; replace it with your own real model class in actual use
compiled = torch.compile(model, options={"guard_filter_fn": filter_runtime_state_guards})

# Switching grad mode does not trigger recompilation
with torch.no_grad():
    compiled(x)
with torch.enable_grad():
    compiled(x)  # No recompilation occurs
```

Example 3: Combine filtering of multiple guard types

Filters multiple types of guards in Table 2 at once (dictionary version, optional types, hasattr, and runtime states), suitable for scenarios where recompilation needs to be comprehensively eliminated:

```python
import torch
import torch_npu

_FILTER_GUARD_TYPES = frozenset({
    # Dictionary version
    "DICT_VERSION", "DICT_KEYS", "DICT_KEYS_MATCH", "DICT_CONTAINS",
    # Optional types
    "TYPE_MATCH", "OPTIONAL_TENSOR",
    # hasattr
    "HASATTR", "NOT_PRESENT_IN_GENERIC_DICT",
    # Runtime state
    "GRAD_MODE", "TORCH_FUNCTION_STATE", "GLOBAL_STATE",
    "DEFAULT_DEVICE", "DETERMINISTIC_ALGORITHMS", "AUTOCAST_STATE",
    "FSDP_TRAINING_STATE",
})

def npu_guard_filter(entries):
    return [entry.guard_type not in _FILTER_GUARD_TYPES for entry in entries]

model = MyModel().npu()  # Note: MyModel here is only a sample placeholder class; replace it with your own real model class in actual use
compiled = torch.compile(model, options={"guard_filter_fn": npu_guard_filter})
```

Example 4: Filter by variable name or attribute

This corresponds to the `TYPE_MATCH`/`OPTIONAL_TENSOR` guards in Table 2. When the type of a specific variable alternates between the prefill and decode stages, you can filter precisely by variable name:

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

Example 5: Filter global variable guards

This corresponds to guards related to global variables, such as `HASATTR`/`NOT_PRESENT_IN_GENERIC_DICT` in Table 2. It is suitable for scenarios where global feature switches and debug configurations do not change during inference:

```python
import torch
import torch_npu

def filter_globals(entries):
    return [not entry.is_global for entry in entries]

model = MyModel().npu()  # Note: MyModel here is only a sample placeholder class; replace it with your own real model class in actual use
compiled = torch.compile(model, options={"guard_filter_fn": filter_globals})
```

Example 6: Use with built-in helpers

```python
import torch
import torch_npu

# Use PyTorch's built-in nn.Module guard filtering
model = MyModel().npu()  # Note: MyModel here is only a sample placeholder class; replace it with your own real model class in actual use
compiled = torch.compile(
    model,
    options={
        "guard_filter_fn": torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe
    },
)
```

## Debugging and Verification

- Confirm whether recompilation has been eliminated

  Verify using `torch.compiler.set_stance("fail_on_recompile")`:

  ```python
  compiled = torch.compile(model, options={"guard_filter_fn": npu_guard_filter})

  # First compilation
  compiled(x)

  # Verify that no recompilation occurs
  with torch.compiler.set_stance("fail_on_recompile"):
      compiled(x)  # An exception is raised if recompilation occurs
  ```

- View guard logs

  Enable guard logs through the environment variable to identify the guard types that trigger recompilation:

  ```bash
  TORCH_LOGS=guards,recompiles python your_script.py
  ```

- Verify output correctness

  After filtering guards, be sure to verify that the compilation output is consistent with Eager mode:

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

## Constraints

- PyTorch version: Must be 2.9.0 or later.
- TorchNPU version: You must install the version matching the PyTorch version. For details, refer to [Version Compatibility](../../../../../COMPATIBILITY.md).

## Related Documentation

For Guard filtering and verification overhead optimization, see [Reducing Guard Overhead](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.reducing_guard_overhead.html) in the official PyTorch 2.14 documentation. For more general usage and core concepts, see the [Reference Documentation](../overview.md#reference-documentation) section in the overview document.
