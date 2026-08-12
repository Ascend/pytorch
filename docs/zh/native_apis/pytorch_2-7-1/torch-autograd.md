# torch.autograd

> [!NOTE]
>
> - API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 目录

- [base API](#base-api)
- [Context method mixins](#context-method-mixins)
- [Profiler](#profiler)
- [Default gradient layouts](#default-gradient-layouts)
- [Tensor autograd functions](#tensor-autograd-functions)
- [Forward-mode Automatic Differentiation](#forward-mode-automatic-differentiation)
- [Functional higher level API](#functional-higher-level-api)
- [Numerical gradient checking](#numerical-gradient-checking)
- [Debugging and anomaly detection](#debugging-and-anomaly-detection)

## base API

### torch.autograd.detect_anomaly

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.autograd.set_detect_anomaly

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.autograd.graph.saved_tensors_hooks

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.autograd.graph.save_on_cpu

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.autograd.graph.disable_saved_tensors_hooks

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.autograd.graph.register_multi_grad_hook

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.autograd.graph.allow_mutation_on_saved_tensors

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`input`仅支持fp32

</div>

### Function.forward

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### Function.backward

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### Function.jvp

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### Function.vmap

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### FunctionCtx.mark_dirty

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### FunctionCtx.mark_non_differentiable

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### FunctionCtx.save_for_backward

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### FunctionCtx.set_materialize_grads

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### Node.name

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### Node.metadata

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### Node.next_functions

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### Node.register_hook

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### Node.register_prehook

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## Context method mixins

### _`class`_ torch.autograd.Function

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

## Profiler

### torch.autograd.profiler.profile

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：采集NPU上的profiling数据时，“`use_device`”需设置为“npu”，例如：

```python
with torch.autograd.profiler.profile(use_device="npu") as prof:
    ...
```

</div>

### torch.autograd.profiler.emit_nvtx

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.autograd.profiler.emit_itt

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### profile.export_chrome_trace

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### profile.key_averages

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### profile.self_cpu_time_total

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### profile.total_average

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.autograd.profiler.load_nvprof

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## Default gradient layouts

### torch.autograd.backward

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64
- 不支持稀疏张量

</div>

## Tensor autograd functions

### torch.autograd.grad

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

## Forward-mode Automatic Differentiation

### torch.autograd.forward_ad.dual_level

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.autograd.forward_ad.make_dual

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`input`仅支持fp32

</div>

### torch.autograd.forward_ad.unpack_dual

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`input`仅支持fp32

</div>

## Functional higher level API

### torch.autograd.functional.jacobian

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`input`仅支持fp32

</div>

### torch.autograd.functional.hessian

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`input`仅支持fp32

</div>

### torch.autograd.functional.vjp

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`input`仅支持fp32

</div>

### torch.autograd.functional.jvp

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`input`仅支持fp32

</div>

### torch.autograd.functional.vhp

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`input`仅支持fp32

</div>

### torch.autograd.functional.hvp

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`input`仅支持fp32

</div>

## Numerical gradient checking

### torch.autograd.gradcheck.gradcheck

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.autograd.gradcheck.gradgradcheck

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## Debugging and anomaly detection

### torch.autograd.grad_mode.set_multithreading_enabled

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>
