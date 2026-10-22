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

**Native Documentation**: [torch.nn.parameter.Parameter](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.Parameter.html)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.nn.parameter.Buffer

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.parameter.Buffer](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.Buffer.html)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.nn.parameter.UninitializedParameter

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.parameter.UninitializedParameter](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.UninitializedParameter.html)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.parameter.UninitializedParameter.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.UninitializedParameter.html#torch.nn.parameter.UninitializedParameter.cls_to_become)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.parameter.UninitializedBuffer

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.parameter.UninitializedBuffer](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.UninitializedBuffer.html)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

</div>

## Containers

### <code><i>class</i></code> torch.nn.Module

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">add_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.add_module](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.add_module)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.apply](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.apply)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

</div>

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.bfloat16](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.bfloat16)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

</div>

> <font size="3">buffers()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.buffers](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.buffers)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

</div>

> <font size="3">children()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.children](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.children)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

</div>

> <font size="3">compile()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.compile](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.compile)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id36 -->

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.cpu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.cpu)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.cuda](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.cuda)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.double](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.double)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id45 -->

</div>

> <font size="3">eval()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.eval](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.eval)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

**Constraints & Notes**: `self` only supports fp32, int64

</div>

> <font size="3">extra_repr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.extra_repr](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.extra_repr)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.float](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.float)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.forward](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.forward)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">get_buffer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.get_buffer](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_buffer)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

</div>

> <font size="3">get_extra_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.get_extra_state](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_extra_state)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

> <font size="3">get_parameter()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.get_parameter](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_parameter)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">get_submodule()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.get_submodule](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_submodule)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id69 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.half](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.half)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">ipu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.ipu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.ipu)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id75 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">modules()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.modules](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.modules)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">named_buffers()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.named_buffers](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_buffers)

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id84 -->

</div>

> <font size="3">named_children()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.named_children](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_children)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">named_modules()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.named_modules](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_modules)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id90 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">named_parameters()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.named_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_parameters)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

</div>

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.parameters](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.parameters)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

</div>

> <font size="3">register_backward_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.register_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_backward_hook)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id99 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_buffer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.register_buffer](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id102 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_forward_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.register_forward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id105 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_forward_pre_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.register_forward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id108 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_full_backward_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.register_full_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_full_backward_pre_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.register_full_backward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id114 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_load_state_dict_post_hook)

**Support Status**:

<!-- npu="910b" id115 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id117 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.register_module](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_module)

**Support Status**:

<!-- npu="910b" id118 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id120 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_parameter()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.register_parameter](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_parameter)

**Support Status**:

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id123 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_state_dict_pre_hook)

**Support Status**:

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id126 -->

</div>

> <font size="3">requires_grad_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.requires_grad_](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.requires_grad_)

**Support Status**:

<!-- npu="910b" id127 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id129 -->

</div>

> <font size="3">set_extra_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.set_extra_state](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.set_extra_state)

**Support Status**:

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id132 -->

</div>

> <font size="3">set_submodule()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.set_submodule](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.set_submodule)

**Support Status**:

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id135 -->

</div>

> <font size="3">share_memory()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.share_memory](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.share_memory)

**Support Status**:

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id138 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.state_dict](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.state_dict)

**Version Notes**: Passing `destination`, `prefix`, and `keep_vars` as positional arguments is deprecated; use keyword arguments.

**Support Status**:

<!-- npu="910b" id139 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id141 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.to](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.to)

**Support Status**:

<!-- npu="910b" id142 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id144 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">to_empty()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.to_empty](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.to_empty)

**Support Status**:

<!-- npu="910b" id145 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id147 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">train()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.train](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.train)

**Support Status**:

<!-- npu="910b" id148 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id150 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.type](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.type)

**Support Status**:

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id153 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int64

</div>

> <font size="3">xpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.xpu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.xpu)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id156 -->

**Constraints & Notes**: The NPU form is named torch.nn.Module.npu.

</div>

> <font size="3">npu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.npu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html)

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id159 -->

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Module.zero_grad](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.zero_grad)

**Support Status**:

<!-- npu="910b" id160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id162 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.Sequential

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Sequential](https://pytorch.org/docs/2.9/generated/torch.nn.Sequential.html)

**Support Status**:

<!-- npu="910b" id163 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id165 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Sequential.append](https://pytorch.org/docs/2.9/generated/torch.nn.Sequential.html#torch.nn.Sequential.append)

**Support Status**:

<!-- npu="910b" id166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id168 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ModuleList

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleList](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html)

**Support Status**:

<!-- npu="910b" id169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id171 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleList.append](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.append)

**Support Status**:

<!-- npu="910b" id172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id174 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">extend()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleList.extend](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.extend)

**Support Status**:

<!-- npu="910b" id175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id177 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">insert()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleList.insert](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.insert)

**Support Status**:

<!-- npu="910b" id178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id180 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ModuleDict

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleDict](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html)

**Support Status**:

<!-- npu="910b" id181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id183 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">clear()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleDict.clear](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.clear)

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id186 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">items()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleDict.items](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.items)

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id189 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleDict.keys](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.keys)

**Support Status**:

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id192 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">pop()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleDict.pop](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.pop)

**Support Status**:

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id195 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">update()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleDict.update](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.update)

**Support Status**:

<!-- npu="910b" id196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id198 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ModuleDict.values](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.values)

**Support Status**:

<!-- npu="910b" id199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id201 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ParameterList

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterList](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterList.html)

**Support Status**:

<!-- npu="910b" id202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id204 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterList.append](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterList.html#torch.nn.ParameterList.append)

**Support Status**:

<!-- npu="910b" id205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id207 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">extend()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterList.extend](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterList.html#torch.nn.ParameterList.extend)

**Support Status**:

<!-- npu="910b" id208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id210 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ParameterDict

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html)

**Support Status**:

<!-- npu="910b" id211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id213 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">clear()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.clear](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.clear)

**Support Status**:

<!-- npu="910b" id214 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id216 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">copy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.copy](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.copy)

**Support Status**:

<!-- npu="910b" id217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id219 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">fromkeys()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.fromkeys](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.fromkeys)

**Support Status**:

<!-- npu="910b" id220 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id222 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.get](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.get)

**Support Status**:

<!-- npu="910b" id223 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id225 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">items()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.items](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.items)

**Support Status**:

<!-- npu="910b" id226 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id228 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.keys](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.keys)

**Support Status**:

<!-- npu="910b" id229 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id231 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">pop()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.pop](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.pop)

**Support Status**:

<!-- npu="910b" id232 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id234 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">popitem()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.popitem](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.popitem)

**Support Status**:

<!-- npu="910b" id235 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id237 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">setdefault()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.setdefault](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.setdefault)

**Support Status**:

<!-- npu="910b" id238 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id238 -->
<!-- npu="A3" id239 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id239 -->
<!-- npu="950" id240 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id240 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">update()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.update](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.update)

**Support Status**:

<!-- npu="910b" id241 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id243 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ParameterDict.values](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.values)

**Support Status**:

<!-- npu="910b" id244 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id244 -->
<!-- npu="A3" id245 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id245 -->
<!-- npu="950" id246 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id246 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### torch.nn.modules.module.register_module_forward_pre_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.module.register_module_forward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_forward_pre_hook.html)

**Support Status**:

<!-- npu="910b" id247 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id247 -->
<!-- npu="A3" id248 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id248 -->
<!-- npu="950" id249 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id249 -->

</div>

### torch.nn.modules.module.register_module_forward_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.module.register_module_forward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_forward_hook.html)

**Support Status**:

<!-- npu="910b" id250 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id252 -->

</div>

### torch.nn.modules.module.register_module_backward_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.module.register_module_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_backward_hook.html)

**Support Status**:

<!-- npu="910b" id253 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id255 -->

</div>

### torch.nn.modules.module.register_module_full_backward_pre_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.module.register_module_full_backward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_full_backward_pre_hook.html)

**Support Status**:

<!-- npu="910b" id256 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id256 -->
<!-- npu="A3" id257 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id257 -->
<!-- npu="950" id258 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id258 -->

</div>

### torch.nn.modules.module.register_module_full_backward_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.module.register_module_full_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_full_backward_hook.html)

**Support Status**:

<!-- npu="910b" id259 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id259 -->
<!-- npu="A3" id260 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id260 -->
<!-- npu="950" id261 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id261 -->

</div>

### torch.nn.modules.module.register_module_buffer_registration_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.module.register_module_buffer_registration_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_buffer_registration_hook.html)

**Support Status**:

<!-- npu="910b" id262 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id264 -->

</div>

### torch.nn.modules.module.register_module_module_registration_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.module.register_module_module_registration_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_module_registration_hook.html)

**Support Status**:

<!-- npu="910b" id265 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id265 -->
<!-- npu="A3" id266 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id266 -->
<!-- npu="950" id267 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id267 -->

</div>

### torch.nn.modules.module.register_module_parameter_registration_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.module.register_module_parameter_registration_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_parameter_registration_hook.html)

**Support Status**:

<!-- npu="910b" id268 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id268 -->
<!-- npu="A3" id269 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id269 -->
<!-- npu="950" id270 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id270 -->

</div>

## Convolution Layers

### <code><i>class</i></code> torch.nn.Conv1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Conv1d](https://pytorch.org/docs/2.9/generated/torch.nn.Conv1d.html)

**Support Status**:

<!-- npu="910b" id271 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id271 -->
<!-- npu="A3" id272 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id272 -->
<!-- npu="950" id273 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id273 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Conv2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Conv2d](https://pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)

**Support Status**:

<!-- npu="910b" id274 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id274 -->
<!-- npu="A3" id275 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id275 -->
<!-- npu="950" id276 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id276 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32

<!-- npu="910b" id277 -->
- On <term>Atlas A2 training products</term>, in the default scenario, if compilation is triggered frequently, it is recommended to manually set `torch.npu.config.allow_internal_format` to False so that input parameters do not enable the internal format, avoiding online compilation, for example:

  ```python
  import torch_npu
  torch_npu.npu.config.allow_internal_format = False
  ```
<!-- end id277 -->

</div>

### <code><i>class</i></code> torch.nn.Conv3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Conv3d](https://pytorch.org/docs/2.9/generated/torch.nn.Conv3d.html)

**Support Status**:

<!-- npu="910b" id278 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id278 -->
<!-- npu="A3" id279 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id279 -->
<!-- npu="950" id280 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id280 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.ConvTranspose1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ConvTranspose1d](https://pytorch.org/docs/2.9/generated/torch.nn.ConvTranspose1d.html)

**Support Status**:

<!-- npu="910b" id281 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id281 -->
<!-- npu="A3" id282 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id282 -->
<!-- npu="950" id283 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id283 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.nn.ConvTranspose2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ConvTranspose2d](https://pytorch.org/docs/2.9/generated/torch.nn.ConvTranspose2d.html)

**Support Status**:

<!-- npu="910b" id284 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id284 -->
<!-- npu="A3" id285 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id285 -->
<!-- npu="950" id286 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id286 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32

<!-- npu="910b,910" id287 -->
- On <term>Atlas training products</term>/<term>Atlas A2 training products</term>, 3-dimensional input is supported only after manually setting `torch.npu.config.allow_internal_format` to False
<!-- end id287 -->

</div>

### <code><i>class</i></code> torch.nn.ConvTranspose3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ConvTranspose3d](https://pytorch.org/docs/2.9/generated/torch.nn.ConvTranspose3d.html)

**Support Status**:

<!-- npu="910b" id288 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id288 -->
<!-- npu="A3" id289 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id289 -->
<!-- npu="950" id290 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id290 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.LazyConv1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyConv1d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv1d.html)

**Support Status**:

<!-- npu="910b" id291 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id291 -->
<!-- npu="A3" id292 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id292 -->
<!-- npu="950" id293 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id293 -->

**Constraints & Notes**: `input` only supports fp16, fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyConv1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv1d.html#torch.nn.LazyConv1d.cls_to_become)

**Support Status**:

<!-- npu="910b" id294 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id294 -->
<!-- npu="A3" id295 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id295 -->
<!-- npu="950" id296 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id296 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConv2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyConv2d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv2d.html)

**Support Status**:

<!-- npu="910b" id297 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id297 -->
<!-- npu="A3" id298 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id298 -->
<!-- npu="950" id299 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id299 -->

**Constraints & Notes**: `input` only supports fp16, fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyConv2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d.cls_to_become)

**Support Status**:

<!-- npu="910b" id300 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id300 -->
<!-- npu="A3" id301 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id301 -->
<!-- npu="950" id302 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id302 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConv3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyConv3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv3d.html#torch.nn.LazyConv3d.cls_to_become)

**Support Status**:

<!-- npu="910b" id303 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id303 -->
<!-- npu="A3" id304 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id304 -->
<!-- npu="950" id305 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id305 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConvTranspose1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyConvTranspose1d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose1d.html)

**Support Status**:

<!-- npu="910b" id306 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id306 -->
<!-- npu="A3" id307 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id307 -->
<!-- npu="950" id308 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id308 -->

**Constraints & Notes**: `input` only supports fp16

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyConvTranspose1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose1d.html#torch.nn.LazyConvTranspose1d.cls_to_become)

**Support Status**:

<!-- npu="910b" id309 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id309 -->
<!-- npu="A3" id310 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id310 -->
<!-- npu="950" id311 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id311 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConvTranspose2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyConvTranspose2d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose2d.html)

**Support Status**:

<!-- npu="910b" id312 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id312 -->
<!-- npu="A3" id313 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id313 -->
<!-- npu="950" id314 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id314 -->

**Constraints & Notes**: `input` only supports fp16, fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyConvTranspose2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose2d.html#torch.nn.LazyConvTranspose2d.cls_to_become)

**Support Status**:

<!-- npu="910b" id315 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id315 -->
<!-- npu="A3" id316 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id316 -->
<!-- npu="950" id317 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id317 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConvTranspose3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyConvTranspose3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose3d.html#torch.nn.LazyConvTranspose3d.cls_to_become)

**Support Status**:

<!-- npu="910b" id318 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id318 -->
<!-- npu="A3" id319 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id319 -->
<!-- npu="950" id320 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id320 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.Unfold

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Unfold](https://pytorch.org/docs/2.9/generated/torch.nn.Unfold.html)

**Support Status**:

<!-- npu="910b" id321 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id321 -->
<!-- npu="A3" id322 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id322 -->
<!-- npu="950" id323 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id323 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Fold

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Fold](https://pytorch.org/docs/2.9/generated/torch.nn.Fold.html)

**Support Status**:

<!-- npu="910b" id324 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id324 -->
<!-- npu="A3" id325 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id325 -->
<!-- npu="950" id326 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id326 -->

**Constraints & Notes**: `input` only supports fp16

</div>

## Pooling layers

### <code><i>class</i></code> torch.nn.MaxPool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MaxPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxPool1d.html)

**Support Status**:

<!-- npu="910b" id327 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id327 -->
<!-- npu="A3" id328 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id328 -->
<!-- npu="950" id329 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id329 -->

</div>

### <code><i>class</i></code> torch.nn.MaxPool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MaxPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxPool2d.html)

**Support Status**:

<!-- npu="910b" id330 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id330 -->
<!-- npu="A3" id331 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id331 -->
<!-- npu="950" id332 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id332 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- By setting `torch_npu.npu.use_compatible_impl(True)`, the memory consistency is aligned with the PyTorch API of the same name, for example:

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

### <code><i>class</i></code> torch.nn.MaxPool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MaxPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxPool3d.html)

**Support Status**:

<!-- npu="910b" id333 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id333 -->
<!-- npu="A3" id334 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id334 -->
<!-- npu="950" id335 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id335 -->

</div>

### <code><i>class</i></code> torch.nn.MaxUnpool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MaxUnpool1d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxUnpool1d.html)

**Support Status**:

<!-- npu="910b" id336 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id336 -->
<!-- npu="A3" id337 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id337 -->
<!-- npu="950" id338 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id338 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.MaxUnpool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MaxUnpool2d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxUnpool2d.html)

**Support Status**:

<!-- npu="910b" id339 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id339 -->
<!-- npu="A3" id340 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id340 -->
<!-- npu="950" id341 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id341 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.MaxUnpool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MaxUnpool3d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxUnpool3d.html)

**Support Status**:

<!-- npu="910b" id342 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id342 -->
<!-- npu="A3" id343 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id343 -->
<!-- npu="950" id344 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id344 -->

</div>

### <code><i>class</i></code> torch.nn.AvgPool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AvgPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.AvgPool1d.html)

**Support Status**:

<!-- npu="910b" id345 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id345 -->
<!-- npu="A3" id346 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id346 -->
<!-- npu="950" id347 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id347 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.AvgPool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AvgPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.AvgPool2d.html)

**Support Status**:

<!-- npu="910b" id348 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id348 -->
<!-- npu="A3" id349 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id349 -->
<!-- npu="950" id350 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id350 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.AvgPool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AvgPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.AvgPool3d.html)

**Support Status**:

<!-- npu="910b" id351 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id351 -->
<!-- npu="A3" id352 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id352 -->
<!-- npu="950" id353 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id353 -->

</div>

### <code><i>class</i></code> torch.nn.LPPool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LPPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.LPPool1d.html)

**Support Status**:

<!-- npu="910b" id354 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id354 -->
<!-- npu="A3" id355 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id355 -->
<!-- npu="950" id356 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id356 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.LPPool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LPPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.LPPool2d.html)

**Support Status**:

<!-- npu="910b" id357 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id357 -->
<!-- npu="A3" id358 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id358 -->
<!-- npu="950" id359 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id359 -->

**Constraints & Notes**: `input` only supports fp16, fp32, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.AdaptiveMaxPool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AdaptiveMaxPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveMaxPool1d.html)

**Support Status**:

<!-- npu="910b" id360 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id360 -->
<!-- npu="A3" id361 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id361 -->
<!-- npu="950" id362 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id362 -->

</div>

### <code><i>class</i></code> torch.nn.AdaptiveMaxPool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AdaptiveMaxPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveMaxPool2d.html)

**Support Status**:

<!-- npu="910b" id363 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id363 -->
<!-- npu="A3" id364 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id364 -->
<!-- npu="950" id365 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id365 -->

</div>

### <code><i>class</i></code> torch.nn.AdaptiveMaxPool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AdaptiveMaxPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveMaxPool3d.html)

**Support Status**:

<!-- npu="910b" id366 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id366 -->
<!-- npu="A3" id367 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id367 -->
<!-- npu="950" id368 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id368 -->

**Constraints & Notes**: `input` only supports fp32, fp64

</div>

### <code><i>class</i></code> torch.nn.AdaptiveAvgPool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AdaptiveAvgPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool1d.html)

**Support Status**:

<!-- npu="910b" id369 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id369 -->
<!-- npu="A3" id370 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id370 -->
<!-- npu="950" id371 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id371 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.AdaptiveAvgPool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AdaptiveAvgPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool2d.html)

**Support Status**:

<!-- npu="910b" id372 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id372 -->
<!-- npu="A3" id373 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id373 -->
<!-- npu="950" id374 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id374 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.AdaptiveAvgPool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AdaptiveAvgPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool3d.html)

**Support Status**:

<!-- npu="910b" id375 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id375 -->
<!-- npu="A3" id376 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id376 -->
<!-- npu="950" id377 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id377 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

## Padding Layers

### <code><i>class</i></code> torch.nn.ReflectionPad1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ReflectionPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ReflectionPad1d.html)

**Support Status**:

<!-- npu="910b" id378 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id378 -->
<!-- npu="A3" id379 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id379 -->
<!-- npu="950" id380 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id380 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.ReflectionPad2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ReflectionPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ReflectionPad2d.html)

**Support Status**:

<!-- npu="910b" id381 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id381 -->
<!-- npu="A3" id382 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id382 -->
<!-- npu="950" id383 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id383 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.ReflectionPad3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ReflectionPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ReflectionPad3d.html)

**Support Status**:

<!-- npu="910b" id384 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id384 -->
<!-- npu="A3" id385 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id385 -->
<!-- npu="950" id386 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id386 -->

</div>

### <code><i>class</i></code> torch.nn.ReplicationPad1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ReplicationPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ReplicationPad1d.html)

**Support Status**:

<!-- npu="910b" id387 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id387 -->
<!-- npu="A3" id388 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id388 -->
<!-- npu="950" id389 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id389 -->

**Constraints & Notes**: `input` only supports fp16, fp32, complex64, complex128

</div>

### <code><i>class</i></code> torch.nn.ReplicationPad2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ReplicationPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ReplicationPad2d.html)

**Support Status**:

<!-- npu="910b" id390 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id390 -->
<!-- npu="A3" id391 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id391 -->
<!-- npu="950" id392 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id392 -->

**Constraints & Notes**: `input` only supports fp16, fp32, complex64, complex128

</div>

### <code><i>class</i></code> torch.nn.ReplicationPad3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ReplicationPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ReplicationPad3d.html)

**Support Status**:

<!-- npu="910b" id393 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id393 -->
<!-- npu="A3" id394 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id394 -->
<!-- npu="950" id395 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id395 -->

</div>

### <code><i>class</i></code> torch.nn.ZeroPad1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ZeroPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ZeroPad1d.html)

**Support Status**:

<!-- npu="910b" id396 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id396 -->
<!-- npu="A3" id397 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id397 -->
<!-- npu="950" id398 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id398 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, complex64, complex128
- Supports 2-3 dimensions

</div>

### <code><i>class</i></code> torch.nn.ZeroPad2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ZeroPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ZeroPad2d.html)

**Support Status**:

<!-- npu="910b" id399 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id399 -->
<!-- npu="A3" id400 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id400 -->
<!-- npu="950" id401 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id401 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.ZeroPad3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ZeroPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ZeroPad3d.html)

**Support Status**:

<!-- npu="910b" id402 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id402 -->
<!-- npu="A3" id403 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id403 -->
<!-- npu="950" id404 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id404 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, complex64, complex128
- Supports 5-6 dimensions

</div>

### <code><i>class</i></code> torch.nn.ConstantPad1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ConstantPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ConstantPad1d.html)

**Support Status**:

<!-- npu="910b" id405 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id405 -->
<!-- npu="A3" id406 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id406 -->
<!-- npu="950" id407 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id407 -->

**Constraints & Notes**:

- `input` only supports int8, bool
- Performance degradation may occur when the input `x` has six or more dimensions

</div>

### <code><i>class</i></code> torch.nn.ConstantPad2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ConstantPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ConstantPad2d.html)

**Support Status**:

<!-- npu="910b" id408 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id408 -->
<!-- npu="A3" id409 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id409 -->
<!-- npu="950" id410 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id410 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Performance degradation may occur when the input `x` has six or more dimensions

</div>

### <code><i>class</i></code> torch.nn.ConstantPad3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ConstantPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ConstantPad3d.html)

**Support Status**:

<!-- npu="910b" id411 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id411 -->
<!-- npu="A3" id412 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id412 -->
<!-- npu="950" id413 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id413 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Performance degradation may occur when the input `x` has six or more dimensions

</div>

## Non-linear Activations (weighted sum, nonlinearity)

### <code><i>class</i></code> torch.nn.ELU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ELU](https://pytorch.org/docs/2.9/generated/torch.nn.ELU.html)

**Support Status**:

<!-- npu="910b" id414 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id414 -->
<!-- npu="A3" id415 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id415 -->
<!-- npu="950" id416 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id416 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### <code><i>class</i></code> torch.nn.Hardshrink

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Hardshrink](https://pytorch.org/docs/2.9/generated/torch.nn.Hardshrink.html)

**Support Status**:

<!-- npu="910b" id417 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id417 -->
<!-- npu="A3" id418 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id418 -->
<!-- npu="950" id419 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id419 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.Hardsigmoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Hardsigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.Hardsigmoid.html)

**Support Status**:

<!-- npu="910b" id420 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id420 -->
<!-- npu="A3" id421 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id421 -->
<!-- npu="950" id422 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id422 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, int32
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.Hardtanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Hardtanh](https://pytorch.org/docs/2.9/generated/torch.nn.Hardtanh.html)

**Support Status**:

<!-- npu="910b" id423 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id423 -->
<!-- npu="A3" id424 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id424 -->
<!-- npu="950" id425 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id425 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.Hardswish

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Hardswish](https://pytorch.org/docs/2.9/generated/torch.nn.Hardswish.html)

**Support Status**:

<!-- npu="910b" id426 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id426 -->
<!-- npu="A3" id427 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id427 -->
<!-- npu="950" id428 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id428 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.LeakyReLU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LeakyReLU](https://pytorch.org/docs/2.9/generated/torch.nn.LeakyReLU.html)

**Support Status**:

<!-- npu="910b" id429 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id429 -->
<!-- npu="A3" id430 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id430 -->
<!-- npu="950" id431 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id431 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### <code><i>class</i></code> torch.nn.LogSigmoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LogSigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.LogSigmoid.html)

**Support Status**:

<!-- npu="910b" id432 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id432 -->
<!-- npu="A3" id433 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id433 -->
<!-- npu="950" id434 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id434 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.MultiheadAttention

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MultiheadAttention](https://pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html)

**Support Status**:

<!-- npu="910b" id435 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id435 -->
<!-- npu="A3" id436 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id436 -->
<!-- npu="950" id437 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id437 -->

**Constraints & Notes**: `query`, `key`, and `value` support only bf16, fp16, and fp32

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MultiheadAttention.forward](https://pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward)

**Support Status**:

<!-- npu="910b" id438 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id438 -->
<!-- npu="A3" id439 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id439 -->
<!-- npu="950" id440 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id440 -->

**Constraints & Notes**: `query`, `key`, and `value` support only bf16, fp16, and fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.PReLU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.PReLU](https://pytorch.org/docs/2.9/generated/torch.nn.PReLU.html)

**Support Status**:

<!-- npu="910b" id441 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id441 -->
<!-- npu="A3" id442 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id442 -->
<!-- npu="950" id443 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id443 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.nn.ReLU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ReLU](https://pytorch.org/docs/2.9/generated/torch.nn.ReLU.html)

**Support Status**:

<!-- npu="910b" id444 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id444 -->
<!-- npu="A3" id445 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id445 -->
<!-- npu="950" id446 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id446 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

### <code><i>class</i></code> torch.nn.ReLU6

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ReLU6](https://pytorch.org/docs/2.9/generated/torch.nn.ReLU6.html)

**Support Status**:

<!-- npu="910b" id447 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id447 -->
<!-- npu="A3" id448 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id448 -->
<!-- npu="950" id449 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id449 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.RReLU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.RReLU](https://pytorch.org/docs/2.9/generated/torch.nn.RReLU.html)

**Support Status**:

<!-- npu="910b" id450 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id450 -->
<!-- npu="A3" id451 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id451 -->
<!-- npu="950" id452 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id452 -->

</div>

### <code><i>class</i></code> torch.nn.SELU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.SELU](https://pytorch.org/docs/2.9/generated/torch.nn.SELU.html)

**Support Status**:

<!-- npu="910b" id453 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id453 -->
<!-- npu="A3" id454 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id454 -->
<!-- npu="950" id455 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id455 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.CELU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.CELU](https://pytorch.org/docs/2.9/generated/torch.nn.CELU.html)

**Support Status**:

<!-- npu="910b" id456 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id456 -->
<!-- npu="A3" id457 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id457 -->
<!-- npu="950" id458 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id458 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.GELU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.GELU](https://pytorch.org/docs/2.9/generated/torch.nn.GELU.html)

**Support Status**:

<!-- npu="910b" id459 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id459 -->
<!-- npu="A3" id460 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id460 -->
<!-- npu="950" id461 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id461 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- The `approximate` parameter only supports being set to `tanh`.

</div>

### <code><i>class</i></code> torch.nn.Sigmoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Sigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.Sigmoid.html)

**Support Status**:

<!-- npu="910b" id462 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id462 -->
<!-- npu="A3" id463 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id463 -->
<!-- npu="950" id464 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id464 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### <code><i>class</i></code> torch.nn.SiLU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.SiLU](https://pytorch.org/docs/2.9/generated/torch.nn.SiLU.html)

**Support Status**:

<!-- npu="910b" id465 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id465 -->
<!-- npu="A3" id466 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id466 -->
<!-- npu="950" id467 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id467 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Mish

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Mish](https://pytorch.org/docs/2.9/generated/torch.nn.Mish.html)

**Support Status**:

<!-- npu="910b" id468 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id468 -->
<!-- npu="A3" id469 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id469 -->
<!-- npu="950" id470 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id470 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Softplus

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Softplus](https://pytorch.org/docs/2.9/generated/torch.nn.Softplus.html)

**Support Status**:

<!-- npu="910b" id471 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id471 -->
<!-- npu="A3" id472 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id472 -->
<!-- npu="950" id473 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id473 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Softshrink

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Softshrink](https://pytorch.org/docs/2.9/generated/torch.nn.Softshrink.html)

**Support Status**:

<!-- npu="910b" id474 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id474 -->
<!-- npu="A3" id475 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id475 -->
<!-- npu="950" id476 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id476 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Softsign

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Softsign](https://pytorch.org/docs/2.9/generated/torch.nn.Softsign.html)

**Support Status**:

<!-- npu="910b" id477 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id477 -->
<!-- npu="A3" id478 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id478 -->
<!-- npu="950" id479 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id479 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.Tanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Tanh](https://pytorch.org/docs/2.9/generated/torch.nn.Tanh.html)

**Support Status**:

<!-- npu="910b" id480 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id480 -->
<!-- npu="A3" id481 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id481 -->
<!-- npu="950" id482 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id482 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.Tanhshrink

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Tanhshrink](https://pytorch.org/docs/2.9/generated/torch.nn.Tanhshrink.html)

**Support Status**:

<!-- npu="910b" id483 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id483 -->
<!-- npu="A3" id484 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id484 -->
<!-- npu="950" id485 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id485 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.Threshold

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Threshold](https://pytorch.org/docs/2.9/generated/torch.nn.Threshold.html)

**Support Status**:

<!-- npu="910b" id486 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id486 -->
<!-- npu="A3" id487 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id487 -->
<!-- npu="950" id488 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id488 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.GLU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.GLU](https://pytorch.org/docs/2.9/generated/torch.nn.GLU.html)

**Support Status**:

<!-- npu="910b" id489 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id489 -->
<!-- npu="A3" id490 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id490 -->
<!-- npu="950" id491 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id491 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

## Non-linear Activations (other)

### <code><i>class</i></code> torch.nn.Softmin

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Softmin](https://pytorch.org/docs/2.9/generated/torch.nn.Softmin.html)

**Support Status**:

<!-- npu="910b" id492 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id492 -->
<!-- npu="A3" id493 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id493 -->
<!-- npu="950" id494 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id494 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Softmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Softmax](https://pytorch.org/docs/2.9/generated/torch.nn.Softmax.html)

**Support Status**:

<!-- npu="910b" id495 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id495 -->
<!-- npu="A3" id496 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id496 -->
<!-- npu="950" id497 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id497 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### <code><i>class</i></code> torch.nn.Softmax2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Softmax2d](https://pytorch.org/docs/2.9/generated/torch.nn.Softmax2d.html)

**Support Status**:

<!-- npu="910b" id498 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id498 -->
<!-- npu="A3" id499 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id499 -->
<!-- npu="950" id500 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id500 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.LogSoftmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LogSoftmax](https://pytorch.org/docs/2.9/generated/torch.nn.LogSoftmax.html)

**Support Status**:

<!-- npu="910b" id501 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id501 -->
<!-- npu="A3" id502 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id502 -->
<!-- npu="950" id503 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id503 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.AdaptiveLogSoftmaxWithLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AdaptiveLogSoftmaxWithLoss](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html)

**Support Status**:

<!-- npu="910b" id504 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id504 -->
<!-- npu="A3" id505 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id505 -->
<!-- npu="950" id506 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id506 -->

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob)

**Support Status**:

<!-- npu="910b" id507 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id507 -->
<!-- npu="A3" id508 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id508 -->
<!-- npu="950" id509 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id509 -->

</div>

> <font size="3">predict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AdaptiveLogSoftmaxWithLoss.predict](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss.predict)

**Support Status**:

<!-- npu="910b" id510 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id510 -->
<!-- npu="A3" id511 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id511 -->
<!-- npu="950" id512 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id512 -->

</div>

</div>

## Normalization Layers

### <code><i>class</i></code> torch.nn.BatchNorm1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.BatchNorm1d](https://pytorch.org/docs/2.9/generated/torch.nn.BatchNorm1d.html)

**Support Status**:

<!-- npu="910b" id513 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id513 -->
<!-- npu="A3" id514 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id514 -->
<!-- npu="950" id515 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id515 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.BatchNorm2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.BatchNorm2d](https://pytorch.org/docs/2.9/generated/torch.nn.BatchNorm2d.html)

**Support Status**:

<!-- npu="910b" id516 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id516 -->
<!-- npu="A3" id517 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id517 -->
<!-- npu="950" id518 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id518 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.BatchNorm3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.BatchNorm3d](https://pytorch.org/docs/2.9/generated/torch.nn.BatchNorm3d.html)

**Support Status**:

<!-- npu="910b" id519 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id519 -->
<!-- npu="A3" id520 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id520 -->
<!-- npu="950" id521 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id521 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.LazyBatchNorm1d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyBatchNorm1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyBatchNorm1d.html#torch.nn.LazyBatchNorm1d.cls_to_become)

**Support Status**:

<!-- npu="910b" id522 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id522 -->
<!-- npu="A3" id523 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id523 -->
<!-- npu="950" id524 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id524 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyBatchNorm2d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyBatchNorm2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyBatchNorm2d.html#torch.nn.LazyBatchNorm2d.cls_to_become)

**Support Status**:

<!-- npu="910b" id525 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id525 -->
<!-- npu="A3" id526 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id526 -->
<!-- npu="950" id527 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id527 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyBatchNorm3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyBatchNorm3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyBatchNorm3d.html#torch.nn.LazyBatchNorm3d.cls_to_become)

**Support Status**:

<!-- npu="910b" id528 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id528 -->
<!-- npu="A3" id529 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id529 -->
<!-- npu="950" id530 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id530 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.GroupNorm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.GroupNorm](https://pytorch.org/docs/2.9/generated/torch.nn.GroupNorm.html)

**Support Status**:

<!-- npu="910b" id531 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id531 -->
<!-- npu="A3" id532 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id532 -->
<!-- npu="950" id533 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id533 -->

**Constraints & Notes**:

- `input` only supports fp32
- The `eps` parameter must be greater than 0.
- Does not support the scenario where `jit_compile=True`.
- This API only supports input `input` with 2 or more dimensions. During backpropagation of this API, the input must be 4-dimensional, `num_groups` must be divisible by 32, and the C-axis dimension must be divisible by (10 * `num_groups`).

</div>

### <code><i>class</i></code> torch.nn.SyncBatchNorm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.SyncBatchNorm](https://pytorch.org/docs/2.9/generated/torch.nn.SyncBatchNorm.html)

**Support Status**:

<!-- npu="910b" id534 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id534 -->
<!-- npu="A3" id535 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id535 -->
<!-- npu="950" id536 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id536 -->

**Constraints & Notes**: `input` only supports fp16, fp32

> <font size="3">convert_sync_batchnorm()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.SyncBatchNorm.convert_sync_batchnorm](https://pytorch.org/docs/2.9/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm)

**Support Status**:

<!-- npu="910b" id537 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id537 -->
<!-- npu="A3" id538 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id538 -->
<!-- npu="950" id539 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id539 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyInstanceNorm1d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyInstanceNorm1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyInstanceNorm1d.html#torch.nn.LazyInstanceNorm1d.cls_to_become)

**Support Status**:

<!-- npu="910b" id540 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id540 -->
<!-- npu="A3" id541 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id541 -->
<!-- npu="950" id542 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id542 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyInstanceNorm2d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyInstanceNorm2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyInstanceNorm2d.html#torch.nn.LazyInstanceNorm2d.cls_to_become)

**Support Status**:

<!-- npu="910b" id543 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id543 -->
<!-- npu="A3" id544 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id544 -->
<!-- npu="950" id545 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id545 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyInstanceNorm3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyInstanceNorm3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyInstanceNorm3d.html#torch.nn.LazyInstanceNorm3d.cls_to_become)

**Support Status**:

<!-- npu="910b" id546 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id546 -->
<!-- npu="A3" id547 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id547 -->
<!-- npu="950" id548 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id548 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LayerNorm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LayerNorm](https://pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html)

**Support Status**:

<!-- npu="910b" id549 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id549 -->
<!-- npu="A3" id550 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id550 -->
<!-- npu="950" id551 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id551 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- By setting `torch_npu.npu.use_compatible_impl(True)`, this API switches from the `aclnnLayerNorm` operator to the `aclnnFastLayerNorm` operator, ensuring memory consistency alignment with the PyTorch API of the same name, for example:

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

## Recurrent Layers

### <code><i>class</i></code> torch.nn.RNNBase

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.RNNBase](https://pytorch.org/docs/2.9/generated/torch.nn.RNNBase.html)

**Support Status**:

<!-- npu="910b" id552 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id552 -->
<!-- npu="A3" id553 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id553 -->
<!-- npu="950" id554 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id554 -->

> <font size="3">flatten_parameters()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.RNNBase.flatten_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.RNNBase.html#torch.nn.RNNBase.flatten_parameters)

**Support Status**:

<!-- npu="910b" id555 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id555 -->
<!-- npu="A3" id556 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id556 -->
<!-- npu="950" id557 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id557 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.RNN

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.RNN](https://pytorch.org/docs/2.9/generated/torch.nn.RNN.html)

**Support Status**:

<!-- npu="910b" id558 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id558 -->
<!-- npu="A3" id559 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id559 -->
<!-- npu="950" id560 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id560 -->

</div>

### <code><i>class</i></code> torch.nn.LSTM

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LSTM](https://pytorch.org/docs/2.9/generated/torch.nn.LSTM.html)

**Support Status**:

<!-- npu="910b" id561 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id561 -->
<!-- npu="A3" id562 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id562 -->
<!-- npu="950" id563 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id563 -->

**Constraints & Notes**:

- `input` only supports fp32
- Does not support the `proj_size` parameter.
- Does not support the `dropout` parameter.
- The `input` parameter does not support 2 dimensions.

</div>

### <code><i>class</i></code> torch.nn.GRU

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.GRU](https://pytorch.org/docs/2.9/generated/torch.nn.GRU.html)

**Support Status**:

<!-- npu="910b" id564 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id564 -->
<!-- npu="A3" id565 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id565 -->
<!-- npu="950" id566 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id566 -->

**Constraints & Notes**:

- Does not support the `dropout` parameter.
- Variable-length sequence input is not supported

</div>

### <code><i>class</i></code> torch.nn.RNNCell

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.RNNCell](https://pytorch.org/docs/2.9/generated/torch.nn.RNNCell.html)

**Support Status**:

<!-- npu="910b" id567 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id567 -->
<!-- npu="A3" id568 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id568 -->
<!-- npu="950" id569 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id569 -->

</div>

### <code><i>class</i></code> torch.nn.LSTMCell

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LSTMCell](https://pytorch.org/docs/2.9/generated/torch.nn.LSTMCell.html)

**Support Status**:

<!-- npu="910b" id570 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id570 -->
<!-- npu="A3" id571 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id571 -->
<!-- npu="950" id572 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id572 -->

**Constraints & Notes**: The API does not currently support `jit_compile=False`. To use it in this mode, add "DynamicGRUV2" to the "NPU_FUZZY_COMPILE_BLACKLIST" option. For details, see [Example of Adding a Binary Blocklist](../appendixes/example_of_adding_a_binary_blocklist.md)

</div>

### <code><i>class</i></code> torch.nn.GRUCell

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.GRUCell](https://pytorch.org/docs/2.9/generated/torch.nn.GRUCell.html)

**Support Status**:

<!-- npu="910b" id573 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id573 -->
<!-- npu="A3" id574 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id574 -->
<!-- npu="950" id575 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id575 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

## Transformer Layers

### <code><i>class</i></code> torch.nn.Transformer

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Transformer](https://pytorch.org/docs/2.9/generated/torch.nn.Transformer.html)

**Support Status**:

<!-- npu="910b" id576 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id576 -->
<!-- npu="A3" id577 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id577 -->
<!-- npu="950" id578 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id578 -->

**Constraints & Notes**: `src` and `tgt` support only fp16 and fp32

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Transformer.forward](https://pytorch.org/docs/2.9/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward)

**Support Status**:

<!-- npu="910b" id579 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id579 -->
<!-- npu="A3" id580 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id580 -->
<!-- npu="950" id581 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id581 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerEncoder

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.TransformerEncoder](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoder.html)

**Support Status**:

<!-- npu="910b" id582 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id582 -->
<!-- npu="A3" id583 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id583 -->
<!-- npu="950" id584 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id584 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.TransformerEncoder.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder.forward)

**Support Status**:

<!-- npu="910b" id585 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id585 -->
<!-- npu="A3" id586 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id586 -->
<!-- npu="950" id587 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id587 -->

**Constraints & Notes**: `src` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerDecoder

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.TransformerDecoder](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerDecoder.html)

**Support Status**:

<!-- npu="910b" id588 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id588 -->
<!-- npu="A3" id589 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id589 -->
<!-- npu="950" id590 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id590 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.TransformerDecoder.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder.forward)

**Support Status**:

<!-- npu="910b" id591 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id591 -->
<!-- npu="A3" id592 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id592 -->
<!-- npu="950" id593 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id593 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerEncoderLayer

<div style="margin-left: 2em">

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.TransformerEncoderLayer.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer.forward)

**Support Status**:

<!-- npu="910b" id594 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id594 -->
<!-- npu="A3" id595 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id595 -->
<!-- npu="950" id596 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id596 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerDecoderLayer

<div style="margin-left: 2em">

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.TransformerDecoderLayer.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer.forward)

**Support Status**:

<!-- npu="910b" id597 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id597 -->
<!-- npu="A3" id598 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id598 -->
<!-- npu="950" id599 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id599 -->

</div>

</div>

## Linear Layers

### <code><i>class</i></code> torch.nn.Identity

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Identity](https://pytorch.org/docs/2.9/generated/torch.nn.Identity.html)

**Support Status**:

<!-- npu="910b" id600 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id600 -->
<!-- npu="A3" id601 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id601 -->
<!-- npu="950" id602 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id602 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.nn.Linear

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Linear](https://pytorch.org/docs/2.9/generated/torch.nn.Linear.html)

**Support Status**:

<!-- npu="910b" id603 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id603 -->
<!-- npu="A3" id604 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id604 -->
<!-- npu="950" id605 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id605 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Bilinear

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Bilinear](https://pytorch.org/docs/2.9/generated/torch.nn.Bilinear.html)

**Support Status**:

<!-- npu="910b" id606 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id606 -->
<!-- npu="A3" id607 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id607 -->
<!-- npu="950" id608 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id608 -->

**Constraints & Notes**: `input1` and `input2` support only bf16, fp16, and fp32

</div>

### <code><i>class</i></code> torch.nn.LazyLinear

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyLinear](https://pytorch.org/docs/2.9/generated/torch.nn.LazyLinear.html)

**Support Status**:

<!-- npu="910b" id609 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id609 -->
<!-- npu="A3" id610 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id610 -->
<!-- npu="950" id611 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id611 -->

**Constraints & Notes**: `input` only supports fp16, fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.LazyLinear.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear.cls_to_become)

**Support Status**:

<!-- npu="910b" id612 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id612 -->
<!-- npu="A3" id613 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id613 -->
<!-- npu="950" id614 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id614 -->

</div>

</div>

## Dropout Layers

### <code><i>class</i></code> torch.nn.Dropout

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Dropout](https://pytorch.org/docs/2.9/generated/torch.nn.Dropout.html)

**Support Status**:

<!-- npu="910b" id615 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id615 -->
<!-- npu="A3" id616 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id616 -->
<!-- npu="950" id617 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id617 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.Dropout2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Dropout2d](https://pytorch.org/docs/2.9/generated/torch.nn.Dropout2d.html)

**Support Status**:

<!-- npu="910b" id618 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id618 -->
<!-- npu="A3" id619 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id619 -->
<!-- npu="950" id620 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id620 -->

**Constraints & Notes**: `input` only supports fp16, fp32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.AlphaDropout

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.AlphaDropout](https://pytorch.org/docs/2.9/generated/torch.nn.AlphaDropout.html)

**Support Status**:

<!-- npu="910b" id621 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id621 -->
<!-- npu="A3" id622 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id622 -->
<!-- npu="950" id623 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id623 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.FeatureAlphaDropout

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.FeatureAlphaDropout](https://pytorch.org/docs/2.9/generated/torch.nn.FeatureAlphaDropout.html)

**Support Status**:

<!-- npu="910b" id624 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id624 -->
<!-- npu="A3" id625 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id625 -->
<!-- npu="950" id626 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id626 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

## Sparse Layers

### <code><i>class</i></code> torch.nn.Embedding

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Embedding](https://pytorch.org/docs/2.9/generated/torch.nn.Embedding.html)

**Support Status**:

<!-- npu="910b" id627 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id627 -->
<!-- npu="A3" id628 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id628 -->
<!-- npu="950" id629 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id629 -->

**Constraints & Notes**:

- `input` only supports int32, int64
- The `max_norm` attribute only supports non-negative values and does not support nan.

> <font size="3">from_pretrained()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Embedding.from_pretrained](https://pytorch.org/docs/2.9/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained)

**Support Status**:

<!-- npu="910b" id630 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id630 -->
<!-- npu="A3" id631 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id631 -->
<!-- npu="950" id632 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id632 -->

**Constraints & Notes**: `embeddings` only supports fp64

</div>

</div>

### <code><i>class</i></code> torch.nn.EmbeddingBag

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.EmbeddingBag](https://pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html)

**Support Status**:

<!-- npu="910b" id633 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id633 -->
<!-- npu="A3" id634 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id634 -->
<!-- npu="950" id635 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id635 -->

**Constraints & Notes**:

- `input` only supports int32, int64
- Only supports `max_norm` greater than or equal to 0.

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.EmbeddingBag.forward](https://pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward)

**Support Status**:

<!-- npu="910b" id636 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id636 -->
<!-- npu="A3" id637 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id637 -->
<!-- npu="950" id638 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id638 -->

**Constraints & Notes**: `input` and `offsets` support only int64

</div>

> <font size="3">from_pretrained()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.EmbeddingBag.from_pretrained](https://pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.from_pretrained)

**Support Status**:

<!-- npu="910b" id639 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id639 -->
<!-- npu="A3" id640 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id640 -->
<!-- npu="950" id641 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id641 -->

**Constraints & Notes**: `embeddings` only supports int64

</div>

</div>

## Loss Functions

### <code><i>class</i></code> torch.nn.L1Loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.L1Loss](https://pytorch.org/docs/2.9/generated/torch.nn.L1Loss.html)

**Support Status**:

<!-- npu="910b" id642 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id642 -->
<!-- npu="A3" id643 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id643 -->
<!-- npu="950" id644 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id644 -->

**Constraints & Notes**: `input` only supports fp16, fp32, int64

</div>

### <code><i>class</i></code> torch.nn.MSELoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MSELoss](https://pytorch.org/docs/2.9/generated/torch.nn.MSELoss.html)

**Support Status**:

<!-- npu="910b" id645 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id645 -->
<!-- npu="A3" id646 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id646 -->
<!-- npu="950" id647 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id647 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.CrossEntropyLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)

**Support Status**:

<!-- npu="910b" id648 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id648 -->
<!-- npu="A3" id649 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id649 -->
<!-- npu="950" id650 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id650 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.CTCLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.CTCLoss](https://pytorch.org/docs/2.9/generated/torch.nn.CTCLoss.html)

**Support Status**:

<!-- npu="910b" id651 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id651 -->
<!-- npu="A3" id652 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id652 -->
<!-- npu="950" id653 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id653 -->

**Constraints & Notes**:

- `log_probs` only supports fp32, fp64
- Does not support 2D input of `log_probs`.

</div>

### <code><i>class</i></code> torch.nn.NLLLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.NLLLoss](https://pytorch.org/docs/2.9/generated/torch.nn.NLLLoss.html)

**Support Status**:

<!-- npu="910b" id654 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id654 -->
<!-- npu="A3" id655 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id655 -->
<!-- npu="950" id656 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id656 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32
- Each element value in `target` should be greater than or equal to 0 and less than the number of classes of `input`.

</div>

### <code><i>class</i></code> torch.nn.PoissonNLLLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.PoissonNLLLoss](https://pytorch.org/docs/2.9/generated/torch.nn.PoissonNLLLoss.html)

**Support Status**:

<!-- npu="910b" id657 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id657 -->
<!-- npu="A3" id658 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id658 -->
<!-- npu="950" id659 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id659 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int64

</div>

### <code><i>class</i></code> torch.nn.GaussianNLLLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.GaussianNLLLoss](https://pytorch.org/docs/2.9/generated/torch.nn.GaussianNLLLoss.html)

**Support Status**:

<!-- npu="910b" id660 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id660 -->
<!-- npu="A3" id661 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id661 -->
<!-- npu="950" id662 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id662 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.KLDivLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.KLDivLoss](https://pytorch.org/docs/2.9/generated/torch.nn.KLDivLoss.html)

**Support Status**:

<!-- npu="910b" id663 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id663 -->
<!-- npu="A3" id664 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id664 -->
<!-- npu="950" id665 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id665 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- Currently, the `log_target` parameter only supports False.

</div>

### <code><i>class</i></code> torch.nn.BCELoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.BCELoss](https://pytorch.org/docs/2.9/generated/torch.nn.BCELoss.html)

**Support Status**:

<!-- npu="910b" id666 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id666 -->
<!-- npu="A3" id667 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id667 -->
<!-- npu="950" id668 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id668 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.BCEWithLogitsLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html)

**Support Status**:

<!-- npu="910b" id669 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id669 -->
<!-- npu="A3" id670 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id670 -->
<!-- npu="950" id671 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id671 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- The `target` parameter does not support backward computation.

</div>

### <code><i>class</i></code> torch.nn.MarginRankingLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MarginRankingLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MarginRankingLoss.html)

**Support Status**:

<!-- npu="910b" id672 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id672 -->
<!-- npu="A3" id673 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id673 -->
<!-- npu="950" id674 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id674 -->

**Constraints & Notes**: `input1`, `input2`, and `target` support only bf16, fp16, fp32, int8, int32, and int64

</div>

### <code><i>class</i></code> torch.nn.HingeEmbeddingLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.HingeEmbeddingLoss](https://pytorch.org/docs/2.9/generated/torch.nn.HingeEmbeddingLoss.html)

**Support Status**:

<!-- npu="910b" id675 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id675 -->
<!-- npu="A3" id676 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id676 -->
<!-- npu="950" id677 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id677 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### <code><i>class</i></code> torch.nn.MultiLabelMarginLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MultiLabelMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MultiLabelMarginLoss.html)

**Support Status**:

<!-- npu="910b" id678 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id678 -->
<!-- npu="A3" id679 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id679 -->
<!-- npu="950" id680 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id680 -->

</div>

### <code><i>class</i></code> torch.nn.HuberLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.HuberLoss](https://pytorch.org/docs/2.9/generated/torch.nn.HuberLoss.html)

**Support Status**:

<!-- npu="910b" id681 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id681 -->
<!-- npu="A3" id682 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id682 -->
<!-- npu="950" id683 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id683 -->

**Constraints & Notes**:

- `input` only supports fp32, fp64
- `target` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.SmoothL1Loss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.SmoothL1Loss](https://pytorch.org/docs/2.9/generated/torch.nn.SmoothL1Loss.html)

**Support Status**:

<!-- npu="910b" id684 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id684 -->
<!-- npu="A3" id685 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id685 -->
<!-- npu="950" id686 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id686 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.MultiLabelSoftMarginLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MultiLabelSoftMarginLoss.html)

**Support Status**:

<!-- npu="910b" id687 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id687 -->
<!-- npu="A3" id688 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id688 -->
<!-- npu="950" id689 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id689 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### <code><i>class</i></code> torch.nn.CosineEmbeddingLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/2.9/generated/torch.nn.CosineEmbeddingLoss.html)

**Support Status**:

<!-- npu="910b" id690 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id690 -->
<!-- npu="A3" id691 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id691 -->
<!-- npu="950" id692 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id692 -->

</div>

### <code><i>class</i></code> torch.nn.MultiMarginLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.MultiMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MultiMarginLoss.html)

**Support Status**:

<!-- npu="910b" id693 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id693 -->
<!-- npu="A3" id694 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id694 -->
<!-- npu="950" id695 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id695 -->

**Constraints & Notes**:

- `input` only supports fp32, fp64
- `target` only supports int64
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.TripletMarginLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.TripletMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.TripletMarginLoss.html)

**Support Status**:

<!-- npu="910b" id696 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id696 -->
<!-- npu="A3" id697 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id697 -->
<!-- npu="950" id698 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id698 -->

**Constraints & Notes**:

- `anchor`, `positive`, and `negative` support only fp16, fp32, uint8, int8, int16, int32, and int64
- May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.nn.TripletMarginWithDistanceLoss

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/2.9/generated/torch.nn.TripletMarginWithDistanceLoss.html)

**Support Status**:

<!-- npu="910b" id699 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id699 -->
<!-- npu="A3" id700 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id700 -->
<!-- npu="950" id701 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id701 -->

**Constraints & Notes**: `anchor` only supports bf16, fp16, fp32

</div>

## Vision Layers

### <code><i>class</i></code> torch.nn.PixelShuffle

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.PixelShuffle](https://pytorch.org/docs/2.9/generated/torch.nn.PixelShuffle.html)

**Support Status**:

<!-- npu="910b" id702 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id702 -->
<!-- npu="A3" id703 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id703 -->
<!-- npu="950" id704 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id704 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.PixelUnshuffle

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.PixelUnshuffle](https://pytorch.org/docs/2.9/generated/torch.nn.PixelUnshuffle.html)

**Support Status**:

<!-- npu="910b" id705 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id705 -->
<!-- npu="A3" id706 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id706 -->
<!-- npu="950" id707 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id707 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.nn.Upsample

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Upsample](https://pytorch.org/docs/2.9/generated/torch.nn.Upsample.html)

**Support Status**:

<!-- npu="910b" id708 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id708 -->
<!-- npu="A3" id709 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id709 -->
<!-- npu="950" id710 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id710 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64

</div>

### <code><i>class</i></code> torch.nn.UpsamplingNearest2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.UpsamplingNearest2d](https://pytorch.org/docs/2.9/generated/torch.nn.UpsamplingNearest2d.html)

**Version Notes**: This class has been deprecated; use `torch.nn.functional.interpolate`.

**Support Status**:

<!-- npu="910b" id711 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id711 -->
<!-- npu="A3" id712 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id712 -->
<!-- npu="950" id713 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id713 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8
- May fall back to CPU execution

</div>

## Shuffle Layers

### <code><i>class</i></code> torch.nn.ChannelShuffle

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.ChannelShuffle](https://pytorch.org/docs/2.9/generated/torch.nn.ChannelShuffle.html)

**Support Status**:

<!-- npu="910b" id714 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id714 -->
<!-- npu="A3" id715 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id715 -->
<!-- npu="950" id716 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id716 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

## DataParallel Layers (multi-GPU, distributed)

### <code><i>class</i></code> torch.nn.DataParallel

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.DataParallel](https://pytorch.org/docs/2.9/generated/torch.nn.DataParallel.html)

**Support Status**:

<!-- npu="910b" id717 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id717 -->
<!-- npu="A3" id718 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id718 -->
<!-- npu="950" id719 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id719 -->

</div>

### <code><i>class</i></code> torch.nn.parallel.DistributedDataParallel

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html)

**Support Status**:

<!-- npu="910b" id720 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id720 -->
<!-- npu="A3" id721 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id721 -->
<!-- npu="950" id722 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id722 -->

> <font size="3">join()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.parallel.DistributedDataParallel.join](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join)

**Support Status**:

<!-- npu="910b" id723 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id723 -->
<!-- npu="A3" id724 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id724 -->
<!-- npu="950" id725 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id725 -->

</div>

> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.parallel.DistributedDataParallel.join_hook](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join_hook)

**Support Status**:

<!-- npu="910b" id726 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id726 -->
<!-- npu="A3" id727 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id727 -->
<!-- npu="950" id728 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id728 -->

</div>

> <font size="3">no_sync()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.parallel.DistributedDataParallel.no_sync](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync)

**Support Status**:

<!-- npu="910b" id729 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id729 -->
<!-- npu="A3" id730 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id730 -->
<!-- npu="950" id731 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id731 -->

</div>

> <font size="3">register_comm_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.parallel.DistributedDataParallel.register_comm_hook](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook)

**Support Status**:

<!-- npu="910b" id732 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id732 -->
<!-- npu="A3" id733 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id733 -->
<!-- npu="950" id734 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id734 -->

</div>

</div>

## Utilities

### torch.nn.utils.clip_grad_norm_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm_.html)

**Support Status**:

<!-- npu="910b" id735 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id735 -->
<!-- npu="A3" id736 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id736 -->
<!-- npu="950" id737 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id737 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.nn.utils.clip_grad_norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.clip_grad_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm.html)

**Version Notes**: This API has been deprecated; use `torch.nn.utils.clip_grad_norm_`.

**Support Status**:

<!-- npu="910b" id738 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id738 -->
<!-- npu="A3" id739 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id739 -->
<!-- npu="950" id740 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id740 -->

</div>

### torch.nn.utils.clip_grad_value_

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_value_.html)

**Support Status**:

<!-- npu="910b" id741 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id741 -->
<!-- npu="A3" id742 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id742 -->
<!-- npu="950" id743 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id743 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.nn.utils.vector_to_parameters

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.vector_to_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.utils.vector_to_parameters.html)

**Support Status**:

<!-- npu="910b" id744 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id744 -->
<!-- npu="A3" id745 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id745 -->
<!-- npu="950" id746 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id746 -->

**Constraints & Notes**: `vec` only supports bf16, fp16, fp32, fp64, complex64

</div>

### torch.nn.utils.weight_norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.weight_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.weight_norm.html)

**Version Notes**: This API has been deprecated; use `torch.nn.utils.parametrizations.weight_norm`. For parameter access and removal, see the migration notes in the Native Documentation.

**Support Status**:

<!-- npu="910b" id747 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id747 -->
<!-- npu="A3" id748 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id748 -->
<!-- npu="950" id749 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id749 -->

</div>

### torch.nn.utils.spectral_norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.spectral_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.spectral_norm.html)

**Support Status**:

<!-- npu="910b" id750 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id750 -->
<!-- npu="A3" id751 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id751 -->
<!-- npu="950" id752 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id752 -->

</div>

### torch.nn.utils.remove_spectral_norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.remove_spectral_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.remove_spectral_norm.html)

**Support Status**:

<!-- npu="910b" id753 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id753 -->
<!-- npu="A3" id754 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id754 -->
<!-- npu="950" id755 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id755 -->

</div>

### torch.nn.utils.skip_init

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.skip_init](https://pytorch.org/docs/2.9/generated/torch.nn.utils.skip_init.html)

**Support Status**:

<!-- npu="910b" id756 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id756 -->
<!-- npu="A3" id757 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id757 -->
<!-- npu="950" id758 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id758 -->

</div>

### <code><i>class</i></code> torch.nn.utils.prune.BasePruningMethod

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.BasePruningMethod](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html)

**Support Status**:

<!-- npu="910b" id759 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id759 -->
<!-- npu="A3" id760 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id760 -->
<!-- npu="950" id761 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id761 -->

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.BasePruningMethod.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.apply)

**Support Status**:

<!-- npu="910b" id762 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id762 -->
<!-- npu="A3" id763 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id763 -->
<!-- npu="950" id764 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id764 -->

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.BasePruningMethod.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.apply_mask)

**Support Status**:

<!-- npu="910b" id765 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id765 -->
<!-- npu="A3" id766 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id766 -->
<!-- npu="950" id767 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id767 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.BasePruningMethod.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.compute_mask)

**Support Status**:

<!-- npu="910b" id768 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id768 -->
<!-- npu="A3" id769 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id769 -->
<!-- npu="950" id770 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id770 -->

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.BasePruningMethod.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.prune)

**Support Status**:

<!-- npu="910b" id771 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id771 -->
<!-- npu="A3" id772 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id772 -->
<!-- npu="950" id773 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id773 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.BasePruningMethod.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.remove)

**Support Status**:

<!-- npu="910b" id774 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id774 -->
<!-- npu="A3" id775 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id775 -->
<!-- npu="950" id776 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id776 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.PruningContainer

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.PruningContainer](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html)

**Support Status**:

<!-- npu="910b" id777 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id777 -->
<!-- npu="A3" id778 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id778 -->
<!-- npu="950" id779 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id779 -->

> <font size="3">add_pruning_method()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.PruningContainer.add_pruning_method](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.add_pruning_method)

**Support Status**:

<!-- npu="910b" id780 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id780 -->
<!-- npu="A3" id781 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id781 -->
<!-- npu="950" id782 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id782 -->

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.PruningContainer.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.apply)

**Support Status**:

<!-- npu="910b" id783 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id783 -->
<!-- npu="A3" id784 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id784 -->
<!-- npu="950" id785 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id785 -->

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.PruningContainer.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.apply_mask)

**Support Status**:

<!-- npu="910b" id786 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id786 -->
<!-- npu="A3" id787 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id787 -->
<!-- npu="950" id788 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id788 -->

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.PruningContainer.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.compute_mask)

**Support Status**:

<!-- npu="910b" id789 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id789 -->
<!-- npu="A3" id790 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id790 -->
<!-- npu="950" id791 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id791 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.PruningContainer.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.prune)

**Support Status**:

<!-- npu="910b" id792 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id792 -->
<!-- npu="A3" id793 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id793 -->
<!-- npu="950" id794 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id794 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.PruningContainer.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.remove)

**Support Status**:

<!-- npu="910b" id795 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id795 -->
<!-- npu="A3" id796 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id796 -->
<!-- npu="950" id797 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id797 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.Identity

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.Identity](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity)

**Support Status**:

<!-- npu="910b" id798 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id798 -->
<!-- npu="A3" id799 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id799 -->
<!-- npu="950" id800 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id800 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.Identity.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity.apply)

**Support Status**:

<!-- npu="910b" id801 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id801 -->
<!-- npu="A3" id802 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id802 -->
<!-- npu="950" id803 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id803 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.Identity.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity.apply_mask)

**Support Status**:

<!-- npu="910b" id804 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id804 -->
<!-- npu="A3" id805 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id805 -->
<!-- npu="950" id806 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id806 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.Identity.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity.prune)

**Support Status**:

<!-- npu="910b" id807 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id807 -->
<!-- npu="A3" id808 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id808 -->
<!-- npu="950" id809 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id809 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.Identity.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity.remove)

**Support Status**:

<!-- npu="910b" id810 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id810 -->
<!-- npu="A3" id811 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id811 -->
<!-- npu="950" id812 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id812 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.RandomUnstructured

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomUnstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html)

**Support Status**:

<!-- npu="910b" id813 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id813 -->
<!-- npu="A3" id814 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id814 -->
<!-- npu="950" id815 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id815 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomUnstructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.apply)

**Support Status**:

<!-- npu="910b" id816 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id816 -->
<!-- npu="A3" id817 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id817 -->
<!-- npu="950" id818 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id818 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomUnstructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.apply_mask)

**Support Status**:

<!-- npu="910b" id819 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id819 -->
<!-- npu="A3" id820 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id820 -->
<!-- npu="950" id821 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id821 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomUnstructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.prune)

**Support Status**:

<!-- npu="910b" id822 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id822 -->
<!-- npu="A3" id823 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id823 -->
<!-- npu="950" id824 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id824 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomUnstructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.remove)

**Support Status**:

<!-- npu="910b" id825 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id825 -->
<!-- npu="A3" id826 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id826 -->
<!-- npu="950" id827 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id827 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.L1Unstructured

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.L1Unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html)

**Support Status**:

<!-- npu="910b" id828 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id828 -->
<!-- npu="A3" id829 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id829 -->
<!-- npu="950" id830 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id830 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.L1Unstructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.apply)

**Support Status**:

<!-- npu="910b" id831 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id831 -->
<!-- npu="A3" id832 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id832 -->
<!-- npu="950" id833 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id833 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.L1Unstructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.apply_mask)

**Support Status**:

<!-- npu="910b" id834 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id834 -->
<!-- npu="A3" id835 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id835 -->
<!-- npu="950" id836 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id836 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.L1Unstructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.prune)

**Support Status**:

<!-- npu="910b" id837 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id837 -->
<!-- npu="A3" id838 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id838 -->
<!-- npu="950" id839 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id839 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.L1Unstructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.remove)

**Support Status**:

<!-- npu="910b" id840 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id840 -->
<!-- npu="A3" id841 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id841 -->
<!-- npu="950" id842 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id842 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.RandomStructured

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomStructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html)

**Support Status**:

<!-- npu="910b" id843 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id843 -->
<!-- npu="A3" id844 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id844 -->
<!-- npu="950" id845 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id845 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomStructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.apply)

**Support Status**:

<!-- npu="910b" id846 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id846 -->
<!-- npu="A3" id847 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id847 -->
<!-- npu="950" id848 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id848 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomStructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.apply_mask)

**Support Status**:

<!-- npu="910b" id849 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id849 -->
<!-- npu="A3" id850 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id850 -->
<!-- npu="950" id851 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id851 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomStructured.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.compute_mask)

**Support Status**:

<!-- npu="910b" id852 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id852 -->
<!-- npu="A3" id853 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id853 -->
<!-- npu="950" id854 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id854 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomStructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.prune)

**Support Status**:

<!-- npu="910b" id855 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id855 -->
<!-- npu="A3" id856 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id856 -->
<!-- npu="950" id857 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id857 -->

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.RandomStructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.remove)

**Support Status**:

<!-- npu="910b" id858 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id858 -->
<!-- npu="A3" id859 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id859 -->
<!-- npu="950" id860 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id860 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.LnStructured

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.LnStructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html)

**Support Status**:

<!-- npu="910b" id861 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id861 -->
<!-- npu="A3" id862 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id862 -->
<!-- npu="950" id863 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id863 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.LnStructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.apply)

**Support Status**:

<!-- npu="910b" id864 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id864 -->
<!-- npu="A3" id865 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id865 -->
<!-- npu="950" id866 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id866 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.LnStructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.apply_mask)

**Support Status**:

<!-- npu="910b" id867 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id867 -->
<!-- npu="A3" id868 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id868 -->
<!-- npu="950" id869 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id869 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.LnStructured.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.compute_mask)

**Support Status**:

<!-- npu="910b" id870 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id870 -->
<!-- npu="A3" id871 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id871 -->
<!-- npu="950" id872 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id872 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.LnStructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.prune)

**Support Status**:

<!-- npu="910b" id873 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id873 -->
<!-- npu="A3" id874 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id874 -->
<!-- npu="950" id875 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id875 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.LnStructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.remove)

**Support Status**:

<!-- npu="910b" id876 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id876 -->
<!-- npu="A3" id877 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id877 -->
<!-- npu="950" id878 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id878 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.CustomFromMask

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.CustomFromMask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html)

**Support Status**:

<!-- npu="910b" id879 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id879 -->
<!-- npu="A3" id880 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id880 -->
<!-- npu="950" id881 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id881 -->

**Constraints & Notes**: `input` only supports int64

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.CustomFromMask.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.apply)

**Support Status**:

<!-- npu="910b" id882 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id882 -->
<!-- npu="A3" id883 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id883 -->
<!-- npu="950" id884 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id884 -->

**Constraints & Notes**: `self` only supports int64

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.CustomFromMask.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.apply_mask)

**Support Status**:

<!-- npu="910b" id885 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id885 -->
<!-- npu="A3" id886 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id886 -->
<!-- npu="950" id887 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id887 -->

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.CustomFromMask.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.prune)

**Support Status**:

<!-- npu="910b" id888 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id888 -->
<!-- npu="A3" id889 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id889 -->
<!-- npu="950" id890 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id890 -->

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.CustomFromMask.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.remove)

**Support Status**:

<!-- npu="910b" id891 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id891 -->
<!-- npu="A3" id892 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id892 -->
<!-- npu="950" id893 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id893 -->

</div>

</div>

### torch.nn.utils.prune.random_unstructured

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.random_unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.random_unstructured.html)

**Support Status**:

<!-- npu="910b" id894 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id894 -->
<!-- npu="A3" id895 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id895 -->
<!-- npu="950" id896 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id896 -->

</div>

### torch.nn.utils.prune.l1_unstructured

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.l1_unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.l1_unstructured.html)

**Support Status**:

<!-- npu="910b" id897 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id897 -->
<!-- npu="A3" id898 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id898 -->
<!-- npu="950" id899 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id899 -->

</div>

### torch.nn.utils.prune.random_structured

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.random_structured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.random_structured.html)

**Support Status**:

<!-- npu="910b" id900 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id900 -->
<!-- npu="A3" id901 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id901 -->
<!-- npu="950" id902 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id902 -->

</div>

### torch.nn.utils.prune.ln_structured

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.ln_structured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.ln_structured.html)

**Support Status**:

<!-- npu="910b" id903 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id903 -->
<!-- npu="A3" id904 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id904 -->
<!-- npu="950" id905 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id905 -->

</div>

### torch.nn.utils.prune.global_unstructured

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.global_unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.global_unstructured.html)

**Support Status**:

<!-- npu="910b" id906 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id906 -->
<!-- npu="A3" id907 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id907 -->
<!-- npu="950" id908 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id908 -->

</div>

### torch.nn.utils.prune.identity

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.identity](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.identity.html)

**Support Status**:

<!-- npu="910b" id909 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id909 -->
<!-- npu="A3" id910 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id910 -->
<!-- npu="950" id911 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id911 -->

</div>

### torch.nn.utils.prune.custom_from_mask

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.custom_from_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.custom_from_mask.html)

**Support Status**:

<!-- npu="910b" id912 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id912 -->
<!-- npu="A3" id913 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id913 -->
<!-- npu="950" id914 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id914 -->

**Constraints & Notes**: `input` only supports int64

</div>

### torch.nn.utils.prune.remove

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.remove.html)

**Support Status**:

<!-- npu="910b" id915 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id915 -->
<!-- npu="A3" id916 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id916 -->
<!-- npu="950" id917 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id917 -->

</div>

### torch.nn.utils.prune.is_pruned

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.prune.is_pruned](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.is_pruned.html)

**Support Status**:

<!-- npu="910b" id918 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id918 -->
<!-- npu="A3" id919 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id919 -->
<!-- npu="950" id920 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id920 -->

</div>

### torch.nn.utils.parametrizations.orthogonal

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.parametrizations.orthogonal](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrizations.orthogonal.html)

**Support Status**:

<!-- npu="910b" id921 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id921 -->
<!-- npu="A3" id922 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id922 -->
<!-- npu="950" id923 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id923 -->

</div>

### torch.nn.utils.parametrizations.spectral_norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.parametrizations.spectral_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrizations.spectral_norm.html)

**Support Status**:

<!-- npu="910b" id924 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id924 -->
<!-- npu="A3" id925 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id925 -->
<!-- npu="950" id926 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id926 -->

</div>

### torch.nn.utils.parametrize.register_parametrization

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.parametrize.register_parametrization](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.register_parametrization.html)

**Support Status**:

<!-- npu="910b" id927 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id927 -->
<!-- npu="A3" id928 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id928 -->
<!-- npu="950" id929 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id929 -->

</div>

### torch.nn.utils.parametrize.remove_parametrizations

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.parametrize.remove_parametrizations](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.remove_parametrizations.html)

**Support Status**:

<!-- npu="910b" id930 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id930 -->
<!-- npu="A3" id931 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id931 -->
<!-- npu="950" id932 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id932 -->

</div>

### torch.nn.utils.parametrize.cached

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.parametrize.cached](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.cached.html)

**Support Status**:

<!-- npu="910b" id933 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id933 -->
<!-- npu="A3" id934 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id934 -->
<!-- npu="950" id935 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id935 -->

</div>

### torch.nn.utils.parametrize.is_parametrized

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.parametrize.is_parametrized](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.is_parametrized.html)

**Support Status**:

<!-- npu="910b" id936 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id936 -->
<!-- npu="A3" id937 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id937 -->
<!-- npu="950" id938 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id938 -->

</div>

### <code><i>class</i></code> torch.nn.utils.parametrize.ParametrizationList

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.parametrize.ParametrizationList](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.ParametrizationList.html)

**Support Status**:

<!-- npu="910b" id939 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id939 -->
<!-- npu="A3" id940 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id940 -->
<!-- npu="950" id941 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id941 -->

> <font size="3">right_inverse()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.parametrize.ParametrizationList.right_inverse](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.ParametrizationList.html#torch.nn.utils.parametrize.ParametrizationList.right_inverse)

**Support Status**:

<!-- npu="910b" id942 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id942 -->
<!-- npu="A3" id943 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id943 -->
<!-- npu="950" id944 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id944 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>

### torch.nn.utils.stateless.functional_call

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.stateless.functional_call](https://pytorch.org/docs/2.9/generated/torch.nn.utils.stateless.functional_call.html)

**Version Notes**: This API has been deprecated; use `torch.func.functional_call`.

**Support Status**:

<!-- npu="910b" id945 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id945 -->
<!-- npu="A3" id946 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id946 -->
<!-- npu="950" id947 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id947 -->

</div>

### <code><i>class</i></code> torch.nn.utils.rnn.PackedSequence

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.PackedSequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html)

**Support Status**:

<!-- npu="910b" id948 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id948 -->
<!-- npu="A3" id949 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id949 -->
<!-- npu="950" id950 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id950 -->

**Constraints & Notes**: `input` only supports fp32, int64

> <font size="3">batch_sizes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.PackedSequence.batch_sizes](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.batch_sizes)

**Support Status**:

<!-- npu="910b" id951 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id951 -->
<!-- npu="A3" id952 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id952 -->
<!-- npu="950" id953 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id953 -->

</div>

> <font size="3">count()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.PackedSequence.count](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.count)

**Support Status**:

<!-- npu="910b" id954 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id954 -->
<!-- npu="A3" id955 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id955 -->
<!-- npu="950" id956 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id956 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">data()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.PackedSequence.data](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.data)

**Support Status**:

<!-- npu="910b" id957 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id957 -->
<!-- npu="A3" id958 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id958 -->
<!-- npu="950" id959 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id959 -->

</div>

> <font size="3">index()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.PackedSequence.index](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.index)

**Support Status**:

<!-- npu="910b" id960 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id960 -->
<!-- npu="A3" id961 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id961 -->
<!-- npu="950" id962 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id962 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.PackedSequence.is_cuda](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.is_cuda)

**Support Status**:

<!-- npu="910b" id963 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id963 -->
<!-- npu="A3" id964 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id964 -->
<!-- npu="950" id965 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id965 -->

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.PackedSequence.is_pinned](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.is_pinned)

**Support Status**:

<!-- npu="910b" id966 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id966 -->
<!-- npu="A3" id967 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id967 -->
<!-- npu="950" id968 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id968 -->

</div>

> <font size="3">sorted_indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.PackedSequence.sorted_indices](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.sorted_indices)

**Support Status**:

<!-- npu="910b" id969 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id969 -->
<!-- npu="A3" id970 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id970 -->
<!-- npu="950" id971 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id971 -->

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.PackedSequence.to](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.to)

**Support Status**:

<!-- npu="910b" id972 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id972 -->
<!-- npu="A3" id973 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id973 -->
<!-- npu="950" id974 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id974 -->

**Constraints & Notes**: `self` only supports fp32, int64

</div>

> <font size="3">unsorted_indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.PackedSequence.unsorted_indices](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.unsorted_indices)

**Support Status**:

<!-- npu="910b" id975 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id975 -->
<!-- npu="A3" id976 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id976 -->
<!-- npu="950" id977 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id977 -->

</div>

</div>

### torch.nn.utils.rnn.pack_padded_sequence

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pack_padded_sequence.html)

**Support Status**:

<!-- npu="910b" id978 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id978 -->
<!-- npu="A3" id979 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id979 -->
<!-- npu="950" id980 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id980 -->

</div>

### torch.nn.utils.rnn.pad_packed_sequence

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.pad_packed_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pad_packed_sequence.html)

**Support Status**:

<!-- npu="910b" id981 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id981 -->
<!-- npu="A3" id982 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id982 -->
<!-- npu="950" id983 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id983 -->

</div>

### torch.nn.utils.rnn.pad_sequence

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pad_sequence.html)

**Support Status**:

<!-- npu="910b" id984 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id984 -->
<!-- npu="A3" id985 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id985 -->
<!-- npu="950" id986 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id986 -->

**Constraints & Notes**: `sequences` only supports fp16, fp32

</div>

### torch.nn.utils.rnn.pack_sequence

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.pack_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pack_sequence.html)

**Support Status**:

<!-- npu="910b" id987 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id987 -->
<!-- npu="A3" id988 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id988 -->
<!-- npu="950" id989 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id989 -->

</div>

### torch.nn.utils.rnn.unpack_sequence

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.unpack_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.unpack_sequence.html)

**Support Status**:

<!-- npu="910b" id990 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id990 -->
<!-- npu="A3" id991 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id991 -->
<!-- npu="950" id992 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id992 -->

</div>

### torch.nn.utils.rnn.unpad_sequence

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.utils.rnn.unpad_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.unpad_sequence.html)

**Support Status**:

<!-- npu="910b" id993 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id993 -->
<!-- npu="A3" id994 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id994 -->
<!-- npu="950" id995 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id995 -->

</div>

### <code><i>class</i></code> torch.nn.Flatten

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Flatten](https://docs.pytorch.org/docs/2.9/generated/torch.nn.modules.flatten.Flatten.html#torch.nn.modules.flatten.Flatten)

**Support Status**:

<!-- npu="910b" id996 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id996 -->
<!-- npu="A3" id997 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id997 -->
<!-- npu="950" id998 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id998 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### <code><i>class</i></code> torch.nn.Unflatten

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Unflatten](https://docs.pytorch.org/docs/2.9/generated/torch.nn.modules.flatten.Unflatten.html#torch.nn.modules.flatten.Unflatten)

**Support Status**:

<!-- npu="910b" id999 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id999 -->
<!-- npu="A3" id1000 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1000 -->
<!-- npu="950" id1001 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1001 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

> <font size="3">NamedShape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.Unflatten.NamedShape](https://docs.pytorch.org/docs/2.9/generated/torch.nn.modules.flatten.Unflatten.html#torch.nn.modules.flatten.Unflatten.NamedShape)

**Support Status**:

<!-- npu="910b" id1002 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1002 -->
<!-- npu="A3" id1003 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1003 -->
<!-- npu="950" id1004 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1004 -->

</div>

</div>

## Lazy Modules Initialization

### <code><i>class</i></code> torch.nn.modules.lazy.LazyModuleMixin

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.lazy.LazyModuleMixin](https://pytorch.org/docs/2.9/generated/torch.nn.modules.lazy.LazyModuleMixin.html)

**Support Status**:

<!-- npu="910b" id1005 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1005 -->
<!-- npu="A3" id1006 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1006 -->
<!-- npu="950" id1007 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1007 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">has_uninitialized_params()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params](https://pytorch.org/docs/2.9/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params)

**Support Status**:

<!-- npu="910b" id1008 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1008 -->
<!-- npu="A3" id1009 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1009 -->
<!-- npu="950" id1010 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1010 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">initialize_parameters()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters)

**Support Status**:

<!-- npu="910b" id1011 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1011 -->
<!-- npu="A3" id1012 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1012 -->
<!-- npu="950" id1013 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1013 -->

**Constraints & Notes**: `self` only supports fp32

</div>

</div>
