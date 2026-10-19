# torch.nn

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/nn.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Containers](#containers)
- [Convolution Layers](#convolution-layers)
- [Pooling layers](#pooling-layers)
- [Padding Layers](#padding-layers)
- [Non-linear Activations (weighted sum, nonlinearity)](#non-linear-activations-weighted-sum-nonlinearity)
- [Non-linear Activations (other)](#non-linear-activations-other)
- [Normalization Layers](#normalization-layers)
- [Recurrent Layers](#recurrent-layers)
- [Transformer Layers](#transformer-layers)
- [Linear Layers](#linear-layers)
- [Dropout Layers](#dropout-layers)
- [Sparse Layers](#sparse-layers)
- [Loss Functions](#loss-functions)
- [Vision Layers](#vision-layers)
- [Shuffle Layers](#shuffle-layers)
- [DataParallel Layers (multi-GPU, distributed)](#dataparallel-layers-multi-gpu-distributed)
- [Utilities](#utilities)
- [Lazy Modules Initialization](#lazy-modules-initialization)

</div>

<div style="display:none;">

## &#8203;torch.nn

</div>

### <code><i>class</i></code> torch.nn.parameter.Parameter

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.nn.parameter.Buffer

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.nn.parameter.UninitializedParameter

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.parameter.UninitializedBuffer

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

## Containers

### <code><i>class</i></code> torch.nn.Module

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">add_module()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">buffers()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">children()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">compile()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">eval()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32, int64

</div>

> <font size="3">extra_repr()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">get_buffer()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">get_extra_state()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">get_parameter()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">get_submodule()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">ipu()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">modules()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">named_buffers()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">named_children()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">named_modules()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">named_parameters()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">register_backward_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_buffer()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_forward_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_forward_pre_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_full_backward_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_full_backward_pre_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_module()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_parameter()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">requires_grad_()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">set_extra_state()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">set_submodule()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">share_memory()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">to_empty()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">train()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp16, fp32, int64

</div>

> <font size="3">xpu()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: The NPU form is named torch.nn.Module.npu.

</div>

> <font size="3">npu()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.Sequential

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ModuleList

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">extend()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">insert()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ModuleDict

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">clear()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">items()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">pop()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">update()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ParameterList

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">extend()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ParameterDict

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">clear()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">copy()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">fromkeys()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">items()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">pop()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">popitem()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">setdefault()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">update()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### torch.nn.modules.module.register_module_forward_pre_hook

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.modules.module.register_module_forward_hook

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.modules.module.register_module_backward_hook

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.modules.module.register_module_full_backward_pre_hook

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.modules.module.register_module_full_backward_hook

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.nn.modules.module.register_module_buffer_registration_hook

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.modules.module.register_module_module_registration_hook

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.modules.module.register_module_parameter_registration_hook

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

## Convolution Layers

### <code><i>class</i></code> torch.nn.Conv1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Conv2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- For <term>Atlas A2 training products</term>, in the default scenario, if compilation is frequently triggered, it is recommended to manually set torch.npu.config.allow_internal_format to False so that input parameters do not enable the internal format, avoiding online compilation.

</div>

### <code><i>class</i></code> torch.nn.Conv3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.ConvTranspose1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.nn.ConvTranspose2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32
- For <term>Atlas training products</term>/<term>Atlas A2 training products</term>, you need to manually set torch.npu.config.allow_internal_format to False to support 3-dimensional input.

</div>

### <code><i>class</i></code> torch.nn.ConvTranspose3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.LazyConv1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConv2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConv3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConvTranspose1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConvTranspose2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConvTranspose3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.Unfold

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Fold

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16

</div>

## Pooling layers

### <code><i>class</i></code> torch.nn.MaxPool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.MaxPool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- By setting torch_npu.npu.use_compatible_impl(True), the interface is aligned with the community interface of the same name in memory consistency.

</div>

### <code><i>class</i></code> torch.nn.MaxPool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.MaxUnpool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.MaxUnpool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.MaxUnpool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.AvgPool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.AvgPool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.AvgPool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.LPPool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.LPPool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.AdaptiveMaxPool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.AdaptiveMaxPool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.AdaptiveMaxPool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32, fp64

</div>

### <code><i>class</i></code> torch.nn.AdaptiveAvgPool1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.AdaptiveAvgPool2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.AdaptiveAvgPool3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

## Padding Layers

### <code><i>class</i></code> torch.nn.ReflectionPad1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.ReflectionPad2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.ReflectionPad3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.ReplicationPad1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, complex64, complex128

</div>

### <code><i>class</i></code> torch.nn.ReplicationPad2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, complex64, complex128

</div>

### <code><i>class</i></code> torch.nn.ReplicationPad3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.ZeroPad1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, complex64, complex128
- `input` only supports 2-3 dimensions

</div>

### <code><i>class</i></code> torch.nn.ZeroPad2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.ZeroPad3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, complex64, complex128
- `input` only supports 5-6 dimensions

</div>

### <code><i>class</i></code> torch.nn.ConstantPad1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports int8, bool
- Performance may degrade when input `x` has six or more dimensions.

</div>

### <code><i>class</i></code> torch.nn.ConstantPad2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Performance may degrade when input `x` has six or more dimensions.

</div>

### <code><i>class</i></code> torch.nn.ConstantPad3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Performance may degrade when input `x` has six or more dimensions.

</div>

## Non-linear Activations (weighted sum, nonlinearity)

### <code><i>class</i></code> torch.nn.ELU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### <code><i>class</i></code> torch.nn.Hardshrink

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.Hardsigmoid

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32, int32
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.Hardtanh

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.Hardswish

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.LeakyReLU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### <code><i>class</i></code> torch.nn.LogSigmoid

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.MultiheadAttention

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `query`, `key`, `value` only supports bf16, fp16, fp32

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `query`, `key`, `value` only supports bf16, fp16, fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.PReLU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.nn.ReLU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

### <code><i>class</i></code> torch.nn.ReLU6

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.RReLU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.SELU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.CELU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.GELU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- The `approximate` parameter only supports being set to `tanh`.

</div>

### <code><i>class</i></code> torch.nn.Sigmoid

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### <code><i>class</i></code> torch.nn.SiLU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Mish

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Softplus

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Softshrink

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Softsign

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.Tanh

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.Tanhshrink

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.Threshold

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.GLU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

## Non-linear Activations (other)

### <code><i>class</i></code> torch.nn.Softmin

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Softmax

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### <code><i>class</i></code> torch.nn.Softmax2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.LogSoftmax

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.AdaptiveLogSoftmaxWithLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">predict()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

## Normalization Layers

### <code><i>class</i></code> torch.nn.BatchNorm1d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.BatchNorm2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.BatchNorm3d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.LazyBatchNorm1d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyBatchNorm2d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyBatchNorm3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.GroupNorm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp32
- The `eps` parameter must be greater than 0.
- Does not support the scenario where `jit_compile=True`.
- This API only supports input `input` with 2 or more dimensions. During backpropagation of this API, the input must be 4-dimensional, `num_groups` must be divisible by 32, and the C-axis dimension must be divisible by (10 * `num_groups`).

</div>

### <code><i>class</i></code> torch.nn.SyncBatchNorm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

> <font size="3">convert_sync_batchnorm()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyInstanceNorm1d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyInstanceNorm2d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyInstanceNorm3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.LayerNorm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- By setting torch_npu.npu.use_compatible_impl(True), this interface switches from the `aclnnLayerNorm` operator to the `aclnnFastLayerNorm` operator, ensuring alignment with the community interface of the same name in memory consistency.

</div>

## Recurrent Layers

### <code><i>class</i></code> torch.nn.RNNBase

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">flatten_parameters()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.RNN

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.LSTM

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `input` only supports fp32
- Does not support the `proj_size` parameter.
- Does not support the `dropout` parameter.
- The `input` parameter does not support 2 dimensions.

</div>

### <code><i>class</i></code> torch.nn.GRU

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.RNNCell

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.LSTMCell

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: The API does not currently support `jit_compile=False`. To use it in this mode, add "DynamicGRUV2" to the "NPU_FUZZY_COMPILE_BLACKLIST" option. For details, see [Example of Adding a Binary Blacklist](../example_of_adding_a_binary_blocklist.md).

</div>

### <code><i>class</i></code> torch.nn.GRUCell

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

## Transformer Layers

### <code><i>class</i></code> torch.nn.Transformer

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `src`, `tgt` only supports fp16, fp32

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerEncoder

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `src` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerDecoder

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerEncoderLayer

<div style="margin-left: 2em">

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerDecoderLayer

<div style="margin-left: 2em">

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

## Linear Layers

### <code><i>class</i></code> torch.nn.Identity

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.nn.Linear

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Bilinear

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input1`, `input2` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.LazyLinear

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

## Dropout Layers

### <code><i>class</i></code> torch.nn.Dropout

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Dropout2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.AlphaDropout

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.FeatureAlphaDropout

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

## Sparse Layers

### <code><i>class</i></code> torch.nn.Embedding

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `input` only supports int32, int64
- The `max_norm` attribute only supports non-negative values and does not support nan.

> <font size="3">from_pretrained()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `embeddings` only supports fp64

</div>

</div>

### <code><i>class</i></code> torch.nn.EmbeddingBag

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports int32, int64
- Only supports `max_norm` greater than or equal to 0.

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input`, `offsets` only supports int64

</div>

> <font size="3">from_pretrained()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `embeddings` only supports int64

</div>

</div>

## Loss Functions

### <code><i>class</i></code> torch.nn.L1Loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, int64

</div>

### <code><i>class</i></code> torch.nn.MSELoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.CrossEntropyLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.CTCLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `log_probs` only supports fp32, fp64
- Does not support 2D input of `log_probs`.

</div>

### <code><i>class</i></code> torch.nn.NLLLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32
- Each element value in `target` should be greater than or equal to 0 and less than the number of classes of `input`.

</div>

### <code><i>class</i></code> torch.nn.PoissonNLLLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int64

</div>

### <code><i>class</i></code> torch.nn.GaussianNLLLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.KLDivLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- Currently, the `log_target` parameter only supports False.

</div>

### <code><i>class</i></code> torch.nn.BCELoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.BCEWithLogitsLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- The `target` parameter does not support backward computation.

</div>

### <code><i>class</i></code> torch.nn.MarginRankingLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input1`, `input2`, `target` only supports bf16, fp16, fp32, int8, int32, int64

</div>

### <code><i>class</i></code> torch.nn.HingeEmbeddingLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.MultiLabelMarginLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.HuberLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp32, fp64
- `target` supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool.
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.SmoothL1Loss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.MultiLabelSoftMarginLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.CosineEmbeddingLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.MultiMarginLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp32, fp64
- `target` supports int64.
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.TripletMarginLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `anchor`, `positive`, `negative` only supports fp16, fp32, uint8, int8, int16, int32, int64
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.TripletMarginWithDistanceLoss

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `anchor` only supports bf16, fp16, fp32

</div>

## Vision Layers

### <code><i>class</i></code> torch.nn.PixelShuffle

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.PixelUnshuffle

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.Upsample

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### <code><i>class</i></code> torch.nn.UpsamplingNearest2d

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8
- May fall back to CPU execution

</div>

## Shuffle Layers

### <code><i>class</i></code> torch.nn.ChannelShuffle

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

## DataParallel Layers (multi-GPU, distributed)

### <code><i>class</i></code> torch.nn.DataParallel

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.parallel.DistributedDataParallel

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

> <font size="3">join()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">no_sync()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">register_comm_hook()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

## Utilities

### torch.nn.utils.clip_grad_norm_

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.nn.utils.clip_grad_norm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.clip_grad_value_

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.utils.vector_to_parameters

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `vec` only supports bf16, fp16, fp32, fp64, complex64

</div>

### torch.nn.utils.weight_norm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.nn.utils.spectral_norm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.nn.utils.remove_spectral_norm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.nn.utils.skip_init

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### <code><i>class</i></code> torch.nn.utils.prune.BasePruningMethod

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.PruningContainer

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">add_pruning_method()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.Identity

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.RandomUnstructured

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.L1Unstructured

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.RandomStructured

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.LnStructured

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.CustomFromMask

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports int64

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports int64

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### torch.nn.utils.prune.random_unstructured

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.prune.l1_unstructured

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.prune.random_structured

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.prune.ln_structured

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.prune.global_unstructured

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.prune.identity

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.prune.custom_from_mask

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports int64

</div>

### torch.nn.utils.prune.remove

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.prune.is_pruned

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.parametrizations.orthogonal

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.parametrizations.spectral_norm

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.parametrize.register_parametrization

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.nn.utils.parametrize.remove_parametrizations

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.nn.utils.parametrize.cached

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.nn.utils.parametrize.is_parametrized

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.utils.parametrize.ParametrizationList

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

> <font size="3">right_inverse()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### torch.nn.utils.stateless.functional_call

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.utils.rnn.PackedSequence

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32, int64

> <font size="3">batch_sizes()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">count()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">data()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">index()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">sorted_indices()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32, int64

</div>

> <font size="3">unsorted_indices()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### torch.nn.utils.rnn.pack_padded_sequence

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.rnn.pad_packed_sequence

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.rnn.pad_sequence

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `sequences` only supports fp16, fp32

</div>

### torch.nn.utils.rnn.pack_sequence

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.rnn.unpack_sequence

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.nn.utils.rnn.unpad_sequence

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### <code><i>class</i></code> torch.nn.modules.flatten.Flatten

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### <code><i>class</i></code> torch.nn.modules.flatten.Unflatten

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

> <font size="3">NamedShape()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

## Lazy Modules Initialization

### <code><i>class</i></code> torch.nn.modules.lazy.LazyModuleMixin

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports fp32

> <font size="3">has_uninitialized_params()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">initialize_parameters()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: `self` only supports fp32

</div>

</div>
