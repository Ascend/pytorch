# INF\_NAN\_MODE\_ENABLE

## Function Description

This environment variable controls the AI processor's handling of Inf/NaN input data, that is, whether the AI processor uses saturation mode or INF_NAN mode. The default value is `1`.

- 0: Saturation mode. When an overflow occurs during computation (Inf), the computation result saturates to the floating-point extreme value (±MAX). When an uncomputable value occurs (NaN), the computation result becomes `0`.
- 1: INF_NAN mode. The computation result of Inf/NaN is output according to its definition.
<!-- npu="910,310p,310b" id3 -->
For <term>Atlas training products</term>/<term>Atlas inference products</term>/<term>Atlas 200I/500 A2 inference products</term>, only saturation mode is supported, and this environment variable does not take effect.
<!-- end id3 -->

<!-- npu="A3,910b" id4 -->
For <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>, the default value is "1" (INF_NAN mode), and "0" (saturation mode) is supported.
<!-- end id4 -->

<!-- npu="950" id5 -->
For <term>Ascend 950DT products</term>, only INF_NAN mode is supported, and this environment variable does not take effect.
<!-- end id5 -->

Saturation mode: Inf is set to max, NaN is set to 0.

Inf example

```python
torch.exp(torch.tensor([12], dtype=torch.float16).npu()) 
# tensor([65504.], device='npu:0', dtype=torch.float16)
```

NaN example

```python
torch.sqrt(torch.tensor([-1.0], dtype=torch.float16).npu()) 
# tensor([0.], device='npu:0', dtype=torch.float16)
```

INF_NAN mode: IEEE 754 standard mode.

Inf example

```python
torch.exp(torch.tensor([12], dtype=torch.float16).npu()) 
# tensor([inf], device='npu:0', dtype=torch.float16)
```

NaN example

```python
torch.sqrt(torch.tensor([-1.0], dtype=torch.float16).npu())
# tensor([nan], device='npu:0', dtype=torch.float16)
```

<!-- npu="A3,910b,910" id6 -->
> [!NOTE]  
>
> For <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>, if precision alignment with <term>Atlas training products</term> is required, you can configure it to "0" saturation mode. In saturation mode, Inf and NaN are converted to the maximum value and 0 of the corresponding data type during computation, which may cause differences in subsequent calculation results. You are advised not to configure it except in special circumstances. <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> have an interception mechanism for the saturation mode configuration. If you need to forcibly enable saturation mode, configure [INF\_NAN\_MODE\_FORCE\_DISABLE](INF_NAN_MODE_FORCE_DISABLE.md)=1.
<!-- end id6 -->

This environment variable is provided by TorchNPU. PyTorch does not have a directly corresponding environment variable.

## Configuration Example

```bash
export INF_NAN_MODE_ENABLE=1
```

## Usage Constraints

- This environment variable must be configured before the Python process starts. Changes made while the process is running do not take effect.
- Only "0" or "1" is supported.

## Supported Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>
<!-- end id2 -->
