# INF\_NAN\_MODE\_ENABLE

## Feature Description

This environment variable controls the AI processor's handling of Inf/NaN input data, that is, whether the AI processor uses saturation mode or INF_NAN mode. The default value is `1`.

- 0: Saturation mode. When an overflow occurs during computation (Inf), the computation result saturates to the floating-point extreme value (±MAX). When an uncomputable value occurs (NaN), the computation result becomes `0`.
- 1: INF_NAN mode. The computation result of Inf/NaN is output according to its definition.

For the Atlas training products/Atlas inference products/Atlas 200I/500 A2 inference products, only saturation mode is supported, and this environment variable does not take effect.

For the Atlas A2 training products/Atlas A3 training products, the default value is `1` INF_NAN mode, and it can be configured to `0` saturation mode.

For the Ascend 950DT, only INF_NAN mode is supported, and this environment variable does not take effect.

> [!NOTICE]  
>
> For the Atlas A2 training products/Atlas A3 training products, if precision alignment with the Atlas training products is required, you can configure it to `0` saturation mode. In saturation mode, Inf and NaN are converted to the maximum value and `0` of the corresponding data type during computation, which may cause differences in subsequent calculation results. You are advised not to configure it except in special circumstances. The Atlas A2 training products/Atlas A3 training products have an interception mechanism for the saturation mode configuration. If you need to forcibly enable saturation mode, configure [INF\_NAN\_MODE\_FORCE\_DISABLE](INF_NAN_MODE_FORCE_DISABLE.md)=1.

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

## Configuration Example

```bash
export INF_NAN_MODE_ENABLE=1
```

## Usage Constraints

None

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
