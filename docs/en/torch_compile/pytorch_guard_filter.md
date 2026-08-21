# Guard Filter

## Overview

`guard_filter_fn` is a compilation option at the PyTorch Dynamo layer, which takes effect during the graph capture phase and is independent of the downstream compilation backend. Both guard generation and validation logic are managed uniformly by Dynamo, making `guard_filter_fn` applicable to all backends.

PyTorch Dynamo (PyTorch's graph capture frontend, responsible for converting Python bytecode into computation graphs) generates a set of guards at each compilation entry point to detect whether the runtime state has changed. When a guard fails, it triggers recompilation, which severely impacts inference performance.

**Table 1** Supported Backends

| Compilation Backend | Applicable Scenario | guard_filter_fn Support |
| --- | --- | --- |
| inductor | NPU/CPU inference and training | Supported |
| npugraph_ex | NPU inference and training | Supported |
| npugraph | NPU inference and training | Supported |
| aot_eager | Debugging purposes | Supported |
| TorchAir-GE | NPU inference and training | Supported |

Since the guard mechanism is fully implemented at the Dynamo layer, switching the backend does not affect the behavior of `guard_filter_fn`. The same filter function can be reused across different backends.

## Use Cases

In NPU training and inference scenarios, the following types of guards may frequently trigger recompilation, even though the corresponding states do not actually affect the correctness of the compiled output at runtime:

**Table 2** Re-trigger scenarios

| Guard Type | Trigger Scenario |
| --- | --- |
| `DICT_VERSION` / `DICT_KEYS` / `DICT_KEYS_MATCH` / `DICT_CONTAINS` | HuggingFace `generation_config`, KV cache state dict, attention kwargs, where the version increments at each step |
| `TYPE_MATCH` / `OPTIONAL_TENSOR` | `past_key_values=None` (prefill) ↔ `tuple` (decode), `attention_mask=None` ↔ Tensor |
| `HASATTR` / `NOT_PRESENT_IN_GENERIC_DICT` | Feature flags, fields attached only after the first forward pass |
| `GRAD_MODE` / `TORCH_FUNCTION_STATE` / `DEFAULT_DEVICE` / `DETERMINISTIC_ALGORITHMS` / `AUTOCAST_STATE` / `FSDP_TRAINING_STATE` | Process-level one-time configurations, but guards are generated for every compilation entry |

Native PyTorch provides the `guard_filter_fn` compilation option, allowing users to customize filtering logic and selectively skip unnecessary guards.

> [!NOTE]
>
> `guard_filter_fn` has unsafe semantics. If the state corresponding to a filtered guard changes at runtime, the compiled output will silently produce incorrect results.

Before using it, confirm the following:

1. The filtered state does not undergo semantic changes throughout the entire inference process.
2. Verify output correctness through parity tests.
3. Verify that recompilation has been eliminated through recompile count tests.

## Usage Guide

Interface Prototype:

```python
torch.compile(model, options={"guard_filter_fn": filter_fn})
```

Parameter description:

`filter_fn` signature:

```python
def filter_fn(entries: list) -> list[bool]:
    """
    Parameters:
        entries: list of guard entries, each containing the following attributes:
            - guard_type (str): guard type, such as "DICT_VERSION", "GRAD_MODE", and so on
            - name (str): name of the variable associated with the guard
            - is_global (bool): whether it is a guard for a global variable
            - value: value associated with the guard (optional)
    Returns:
        a boolean list of the same length as entries, where True means keep the guard and False means filter it out
    """
```

> [!NOTICE]
>
> Native PyTorch also provides built-in helpers, for example:
>
> - `torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe`: Skips guards on attribute changes of built-in nn.Module (such as `Linear`, `Conv2d`, and other PyTorch built-in modules). Suitable for scenarios where model weights are not modified during inference, avoiding recompilation caused by module attribute version auto-increment.
> - `torch.compiler.skip_guard_on_all_nn_modules_unsafe`: Skips guards on attribute changes of all nn.Module (including user-defined modules). Broader in scope than the previous one, suitable for scenarios where the entire model is completely static during inference.
> - `torch.compiler.skip_guard_on_globals_unsafe`: Skips guards on global variables. Suitable for scenarios where global configurations (such as feature flags and debug switches) do not change during inference, avoiding recompilation triggered by global variable version changes.

## Usage Examples

Example 1: Filtering dictionary version guards

Corresponds to `DICT_VERSION`/`DICT_KEYS`/`DICT_KEYS_MATCH`/`DICT_CONTAINS` type guards in Table 2. Suitable for scenarios in HuggingFace model inference where dictionaries such as `generation_config` and KV cache state dict cause recompilation due to per-step version auto-increment:

```python
import torch
import torch_npu

_DICT_GUARD_TYPES = frozenset({
    "DICT_VERSION", "DICT_KEYS", "DICT_KEYS_MATCH", "DICT_CONTAINS",
})

def filter_dict_guards(entries):
    return [entry.guard_type not in _DICT_GUARD_TYPES for entry in entries]

model = MyModel().npu()  # Note: MyModel here is only a placeholder class for demonstration. Replace it with your real model class when you use it.
compiled = torch.compile(model, options={"guard_filter_fn": filter_dict_guards})
```

Example 2: Filtering runtime state guards

Corresponds to process-level configuration guards such as `GRAD_MODE`/`TORCH_FUNCTION_STATE`/`DEFAULT_DEVICE` in Table 2. Suitable for scenarios where `torch.no_grad()` / `torch.enable_grad()` are alternately switched in the inference pipeline:

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

model = MyModel().npu()  # Note: MyModel here is only a placeholder class for demonstration. Replace it with your real model class when you use it.
compiled = torch.compile(model, options={"guard_filter_fn": filter_runtime_state_guards})

# Switching grad mode does not trigger recompilation
with torch.no_grad():
    compiled(x)
with torch.enable_grad():
    compiled(x)  # No recompilation
```

Example 3: Combining multiple guard filters

Filter multiple guard types from Table 2 at once (dictionary version, optional type, hasattr, runtime state), suitable for scenarios that require comprehensive elimination of recompilation:

```python
import torch
import torch_npu

_FILTER_GUARD_TYPES = frozenset({
    # Dictionary version
    "DICT_VERSION", "DICT_KEYS", "DICT_KEYS_MATCH", "DICT_CONTAINS",
    # Optional type
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

model = MyModel().npu()  # Note: MyModel here is only a placeholder class for demonstration. Replace it with your real model class when you use it.
compiled = torch.compile(model, options={"guard_filter_fn": npu_guard_filter})
```

Example 4: Filtering by variable name or attribute

Corresponds to the `TYPE_MATCH`/`OPTIONAL_TENSOR` type guards in Table 2. When a specific variable alternates between types during the prefill and decode phases, you can filter precisely by variable name:

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

Example 5: Filtering global variable guards

Corresponds to guards related to global variables such as `HASATTR`/`NOT_PRESENT_IN_GENERIC_DICT` in Table 2. Suitable for scenarios where global feature flags and debug configurations do not change during inference:

```python
import torch
import torch_npu

def filter_globals(entries):
    return [not entry.is_global for entry in entries]

model = MyModel().npu()  # Note: MyModel here is only a placeholder class for demonstration. Replace it with your real model class when you use it.
compiled = torch.compile(model, options={"guard_filter_fn": filter_globals})
```

Example 6: Use with Built-in Helpers

```python
import torch
import torch_npu

# Use PyTorch's built-in nn.Module guard filtering
model = MyModel().npu()  # Note: MyModel here is only a placeholder class for demonstration. Replace it with your real model class when you use it.
compiled = torch.compile(
    model,
    options={
        "guard_filter_fn": torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe
    },
)
```

## Debugging and Verification

- Confirm whether recompilation is eliminated

  Use `torch.compiler.set_stance("fail_on_recompile")` to verify:

  ```python
  compiled = torch.compile(model, options={"guard_filter_fn": npu_guard_filter})

  # First compilation
  compiled(x)

  # Verify that no recompilation occurs
  with torch.compiler.set_stance("fail_on_recompile"):
      compiled(x)  # Throws an exception if recompilation occurs
  ```

- View guard logs

  Enable guard logs via environment variables to identify the guard types that trigger recompilation:

  ```bash
  TORCH_LOGS=guards,recompiles python your_script.py
  ```

- Verify output correctness

  After filtering guards, always verify that the compiled output matches the eager mode:

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
- TorchNPU version: Must install the version compatible with the PyTorch version. For details, see [Version Notes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md).
