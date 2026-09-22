# torch.nn

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.14/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.14/nn.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

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

**原生文档**：[torch.nn.parameter.Parameter](https://pytorch.org/docs/2.14/generated/torch.nn.parameter.Parameter.html)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT</term>：支持
<!-- end id3 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.nn.parameter.Buffer

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parameter.Buffer](https://pytorch.org/docs/2.14/generated/torch.nn.parameter.Buffer.html)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT</term>：支持
<!-- end id6 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.nn.parameter.UninitializedParameter

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parameter.UninitializedParameter](https://pytorch.org/docs/2.14/generated/torch.nn.parameter.UninitializedParameter.html)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT</term>：支持
<!-- end id9 -->

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parameter.UninitializedParameter.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.parameter.UninitializedParameter.html#torch.nn.parameter.UninitializedParameter.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id12 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.parameter.UninitializedBuffer

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parameter.UninitializedBuffer](https://pytorch.org/docs/2.14/generated/torch.nn.parameter.UninitializedBuffer.html)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT</term>：支持
<!-- end id15 -->

</div>

## Containers

### <code><i>class</i></code> torch.nn.Module

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT</term>：支持
<!-- end id18 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">add_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.add_module](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.add_module)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT</term>：支持
<!-- end id21 -->

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.apply](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.apply)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT</term>：支持
<!-- end id24 -->

</div>

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.bfloat16](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.bfloat16)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT</term>：支持
<!-- end id27 -->

</div>

> <font size="3">buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.buffers](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.buffers)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT</term>：支持
<!-- end id30 -->

</div>

> <font size="3">children()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.children](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.children)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT</term>：支持
<!-- end id33 -->

</div>

> <font size="3">compile()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.compile](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.compile)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id36 -->

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.cpu](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.cpu)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id39 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.cuda](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.cuda)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id42 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.double](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.double)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id45 -->

</div>

> <font size="3">eval()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.eval](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.eval)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id48 -->

**限制与说明**： `self`仅支持fp32，int64

</div>

> <font size="3">extra_repr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.extra_repr](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.extra_repr)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id51 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.float](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.float)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id54 -->

**限制与说明**： `self`仅支持fp16，fp32

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.forward](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.forward)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id57 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">get_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.get_buffer](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.get_buffer)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id60 -->

</div>

> <font size="3">get_extra_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.get_extra_state](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.get_extra_state)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id63 -->

</div>

> <font size="3">get_parameter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.get_parameter](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.get_parameter)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id66 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">get_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.get_submodule](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.get_submodule)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id69 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.half](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.half)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id72 -->

**限制与说明**： `self`仅支持fp16，fp32

</div>

> <font size="3">ipu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.ipu](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.ipu)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id75 -->

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.load_state_dict](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id78 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.modules](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.modules)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT</term>：支持
<!-- end id81 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">named_buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.named_buffers](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.named_buffers)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id84 -->

</div>

> <font size="3">named_children()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.named_children](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.named_children)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id87 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">named_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.named_modules](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.named_modules)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT</term>：支持
<!-- end id90 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">named_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.named_parameters](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.named_parameters)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT</term>：支持
<!-- end id93 -->

</div>

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.parameters](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.parameters)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id96 -->

</div>

> <font size="3">register_backward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_backward_hook](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_backward_hook)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id99 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">register_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_buffer](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id102 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">register_forward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_forward_hook](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT</term>：支持
<!-- end id105 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">register_forward_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_forward_pre_hook](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT</term>：支持
<!-- end id108 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">register_full_backward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_full_backward_hook](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id111 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">register_full_backward_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_full_backward_pre_hook](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id114 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_load_state_dict_post_hook](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_load_state_dict_post_hook)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT</term>：支持
<!-- end id117 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">register_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_module](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_module)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id120 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">register_parameter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_parameter](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_parameter)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id123 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_state_dict_pre_hook](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_state_dict_pre_hook)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id126 -->

</div>

> <font size="3">requires_grad_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.requires_grad_](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.requires_grad_)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id129 -->

</div>

> <font size="3">set_extra_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.set_extra_state](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.set_extra_state)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id132 -->

</div>

> <font size="3">set_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.set_submodule](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.set_submodule)

**产品支持情况**：

<!-- npu="910b" id133 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id135 -->

</div>

> <font size="3">share_memory()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.share_memory](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.share_memory)

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id138 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.state_dict](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.state_dict)

**版本说明**：`destination`、`prefix`和`keep_vars`的位置参数传法已废弃，请使用关键字参数。

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT</term>：支持
<!-- end id141 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.to](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.to)

**产品支持情况**：

<!-- npu="910b" id142 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id144 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">to_empty()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.to_empty](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.to_empty)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id147 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">train()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.train](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.train)

**产品支持情况**：

<!-- npu="910b" id148 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id150 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.type](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.type)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id153 -->

**限制与说明**： `self`仅支持fp16，fp32，int64

</div>

> <font size="3">xpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.xpu](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.xpu)

**产品支持情况**：

<!-- npu="910b" id154 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id156 -->

**限制与说明**： NPU形式名称为`torch.nn.Module.npu`

</div>

> <font size="3">npu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.npu](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html)

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id159 -->

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.zero_grad](https://pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.zero_grad)

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id162 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.Sequential

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Sequential](https://pytorch.org/docs/2.14/generated/torch.nn.Sequential.html)

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT</term>：支持
<!-- end id165 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Sequential.append](https://pytorch.org/docs/2.14/generated/torch.nn.Sequential.html#torch.nn.Sequential.append)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id168 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ModuleList

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleList](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleList.html)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT</term>：支持
<!-- end id171 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleList.append](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.append)

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id174 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">extend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleList.extend](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.extend)

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id177 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">insert()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleList.insert](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.insert)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id180 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ModuleDict

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleDict.html)

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT</term>：支持
<!-- end id183 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">clear()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.clear](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.clear)

**产品支持情况**：

<!-- npu="910b" id184 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id186 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">items()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.items](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.items)

**产品支持情况**：

<!-- npu="910b" id187 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id189 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.keys](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.keys)

**产品支持情况**：

<!-- npu="910b" id190 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id192 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">pop()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.pop](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.pop)

**产品支持情况**：

<!-- npu="910b" id193 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id195 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">update()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.update](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.update)

**产品支持情况**：

<!-- npu="910b" id196 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id198 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.values](https://pytorch.org/docs/2.14/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.values)

**产品支持情况**：

<!-- npu="910b" id199 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id201 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ParameterList

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterList](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterList.html)

**产品支持情况**：

<!-- npu="910b" id202 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT</term>：支持
<!-- end id204 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterList.append](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterList.html#torch.nn.ParameterList.append)

**产品支持情况**：

<!-- npu="910b" id205 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id207 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">extend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterList.extend](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterList.html#torch.nn.ParameterList.extend)

**产品支持情况**：

<!-- npu="910b" id208 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id210 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.ParameterDict

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html)

**产品支持情况**：

<!-- npu="910b" id211 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT</term>：支持
<!-- end id213 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">clear()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.clear](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.clear)

**产品支持情况**：

<!-- npu="910b" id214 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id216 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">copy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.copy](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.copy)

**产品支持情况**：

<!-- npu="910b" id217 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id219 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">fromkeys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.fromkeys](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.fromkeys)

**产品支持情况**：

<!-- npu="910b" id220 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id222 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.get](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.get)

**产品支持情况**：

<!-- npu="910b" id223 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id225 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">items()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.items](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.items)

**产品支持情况**：

<!-- npu="910b" id226 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id228 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.keys](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.keys)

**产品支持情况**：

<!-- npu="910b" id229 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id231 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">pop()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.pop](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.pop)

**产品支持情况**：

<!-- npu="910b" id232 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id234 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">popitem()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.popitem](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.popitem)

**产品支持情况**：

<!-- npu="910b" id235 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id237 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">setdefault()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.setdefault](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.setdefault)

**产品支持情况**：

<!-- npu="910b" id238 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id238 -->
<!-- npu="A3" id239 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id239 -->
<!-- npu="950" id240 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id240 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">update()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.update](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.update)

**产品支持情况**：

<!-- npu="910b" id241 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id243 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.values](https://pytorch.org/docs/2.14/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.values)

**产品支持情况**：

<!-- npu="910b" id244 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id244 -->
<!-- npu="A3" id245 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id245 -->
<!-- npu="950" id246 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id246 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### torch.nn.modules.module.register_module_forward_pre_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_forward_pre_hook](https://pytorch.org/docs/2.14/generated/torch.nn.modules.module.register_module_forward_pre_hook.html)

**产品支持情况**：

<!-- npu="910b" id247 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id247 -->
<!-- npu="A3" id248 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id248 -->
<!-- npu="950" id249 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id249 -->

</div>

### torch.nn.modules.module.register_module_forward_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_forward_hook](https://pytorch.org/docs/2.14/generated/torch.nn.modules.module.register_module_forward_hook.html)

**产品支持情况**：

<!-- npu="910b" id250 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id252 -->

</div>

### torch.nn.modules.module.register_module_backward_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_backward_hook](https://pytorch.org/docs/2.14/generated/torch.nn.modules.module.register_module_backward_hook.html)

**产品支持情况**：

<!-- npu="910b" id253 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id255 -->

</div>

### torch.nn.modules.module.register_module_full_backward_pre_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_full_backward_pre_hook](https://pytorch.org/docs/2.14/generated/torch.nn.modules.module.register_module_full_backward_pre_hook.html)

**产品支持情况**：

<!-- npu="910b" id256 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id256 -->
<!-- npu="A3" id257 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id257 -->
<!-- npu="950" id258 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id258 -->

</div>

### torch.nn.modules.module.register_module_full_backward_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_full_backward_hook](https://pytorch.org/docs/2.14/generated/torch.nn.modules.module.register_module_full_backward_hook.html)

**产品支持情况**：

<!-- npu="910b" id259 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id259 -->
<!-- npu="A3" id260 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id260 -->
<!-- npu="950" id261 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id261 -->

</div>

### torch.nn.modules.module.register_module_buffer_registration_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_buffer_registration_hook](https://pytorch.org/docs/2.14/generated/torch.nn.modules.module.register_module_buffer_registration_hook.html)

**产品支持情况**：

<!-- npu="910b" id262 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id264 -->

</div>

### torch.nn.modules.module.register_module_module_registration_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_module_registration_hook](https://pytorch.org/docs/2.14/generated/torch.nn.modules.module.register_module_module_registration_hook.html)

**产品支持情况**：

<!-- npu="910b" id265 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id265 -->
<!-- npu="A3" id266 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id266 -->
<!-- npu="950" id267 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id267 -->

</div>

### torch.nn.modules.module.register_module_parameter_registration_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_parameter_registration_hook](https://pytorch.org/docs/2.14/generated/torch.nn.modules.module.register_module_parameter_registration_hook.html)

**产品支持情况**：

<!-- npu="910b" id268 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id268 -->
<!-- npu="A3" id269 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id269 -->
<!-- npu="950" id270 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id270 -->

</div>

## Convolution Layers

### <code><i>class</i></code> torch.nn.Conv1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Conv1d](https://pytorch.org/docs/2.14/generated/torch.nn.Conv1d.html)

**产品支持情况**：

<!-- npu="910b" id271 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id271 -->
<!-- npu="A3" id272 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id272 -->
<!-- npu="950" id273 -->
- <term>Ascend 950DT</term>：支持
<!-- end id273 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.Conv2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Conv2d](https://pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html)

**产品支持情况**：

<!-- npu="910b" id274 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id274 -->
<!-- npu="A3" id275 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id275 -->
<!-- npu="950" id276 -->
- <term>Ascend 950DT</term>：支持
<!-- end id276 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32

<!-- npu="910b" id277 -->
- <term>Atlas A2 训练系列产品</term>，默认场景下，如果频繁触发编译，建议手动设置`torch.npu.config.allow_internal_format`为False，控制入参不开启内部格式，避免在线编译，例如：

  ```python
  import torch_npu
  torch_npu.npu.config.allow_internal_format = False
  ```
<!-- end id277 -->

</div>

### <code><i>class</i></code> torch.nn.Conv3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Conv3d](https://pytorch.org/docs/2.14/generated/torch.nn.Conv3d.html)

**产品支持情况**：

<!-- npu="910b" id278 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id278 -->
<!-- npu="A3" id279 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id279 -->
<!-- npu="950" id280 -->
- <term>Ascend 950DT</term>：支持
<!-- end id280 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.ConvTranspose1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConvTranspose1d](https://pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose1d.html)

**产品支持情况**：

<!-- npu="910b" id281 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id281 -->
<!-- npu="A3" id282 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id282 -->
<!-- npu="950" id283 -->
- <term>Ascend 950DT</term>：支持
<!-- end id283 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.nn.ConvTranspose2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConvTranspose2d](https://pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html)

**产品支持情况**：

<!-- npu="910b" id284 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id284 -->
<!-- npu="A3" id285 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id285 -->
<!-- npu="950" id286 -->
- <term>Ascend 950DT</term>：支持
<!-- end id286 -->

**限制与说明**：

- `input`仅支持fp16，fp32

<!-- npu="910b,910" id287 -->
- <term>Atlas 训练系列产品</term>/<term>Atlas A2 训练系列产品</term>，需手动设置`torch.npu.config.allow_internal_format`为False，才可支持3维输入
<!-- end id287 -->

</div>

### <code><i>class</i></code> torch.nn.ConvTranspose3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConvTranspose3d](https://pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose3d.html)

**产品支持情况**：

<!-- npu="910b" id288 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id288 -->
<!-- npu="A3" id289 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id289 -->
<!-- npu="950" id290 -->
- <term>Ascend 950DT</term>：支持
<!-- end id290 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.LazyConv1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConv1d](https://pytorch.org/docs/2.14/generated/torch.nn.LazyConv1d.html)

**产品支持情况**：

<!-- npu="910b" id291 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id291 -->
<!-- npu="A3" id292 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id292 -->
<!-- npu="950" id293 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id293 -->

**限制与说明**： `input`仅支持fp16，fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConv1d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyConv1d.html#torch.nn.LazyConv1d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id294 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id294 -->
<!-- npu="A3" id295 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id295 -->
<!-- npu="950" id296 -->
- <term>Ascend 950DT</term>：支持
<!-- end id296 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConv2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConv2d](https://pytorch.org/docs/2.14/generated/torch.nn.LazyConv2d.html)

**产品支持情况**：

<!-- npu="910b" id297 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id297 -->
<!-- npu="A3" id298 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id298 -->
<!-- npu="950" id299 -->
- <term>Ascend 950DT</term>：支持
<!-- end id299 -->

**限制与说明**： `input`仅支持fp16，fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConv2d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id300 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id300 -->
<!-- npu="A3" id301 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id301 -->
<!-- npu="950" id302 -->
- <term>Ascend 950DT</term>：支持
<!-- end id302 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConv3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConv3d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyConv3d.html#torch.nn.LazyConv3d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id303 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id303 -->
<!-- npu="A3" id304 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id304 -->
<!-- npu="950" id305 -->
- <term>Ascend 950DT</term>：支持
<!-- end id305 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConvTranspose1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConvTranspose1d](https://pytorch.org/docs/2.14/generated/torch.nn.LazyConvTranspose1d.html)

**产品支持情况**：

<!-- npu="910b" id306 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id306 -->
<!-- npu="A3" id307 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id307 -->
<!-- npu="950" id308 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id308 -->

**限制与说明**： `input`仅支持fp16

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConvTranspose1d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyConvTranspose1d.html#torch.nn.LazyConvTranspose1d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id309 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id309 -->
<!-- npu="A3" id310 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id310 -->
<!-- npu="950" id311 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id311 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConvTranspose2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConvTranspose2d](https://pytorch.org/docs/2.14/generated/torch.nn.LazyConvTranspose2d.html)

**产品支持情况**：

<!-- npu="910b" id312 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id312 -->
<!-- npu="A3" id313 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id313 -->
<!-- npu="950" id314 -->
- <term>Ascend 950DT</term>：支持
<!-- end id314 -->

**限制与说明**： `input`仅支持fp16，fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConvTranspose2d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyConvTranspose2d.html#torch.nn.LazyConvTranspose2d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id315 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id315 -->
<!-- npu="A3" id316 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id316 -->
<!-- npu="950" id317 -->
- <term>Ascend 950DT</term>：支持
<!-- end id317 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyConvTranspose3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConvTranspose3d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyConvTranspose3d.html#torch.nn.LazyConvTranspose3d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id318 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id318 -->
<!-- npu="A3" id319 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id319 -->
<!-- npu="950" id320 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id320 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.Unfold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Unfold](https://pytorch.org/docs/2.14/generated/torch.nn.Unfold.html)

**产品支持情况**：

<!-- npu="910b" id321 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id321 -->
<!-- npu="A3" id322 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id322 -->
<!-- npu="950" id323 -->
- <term>Ascend 950DT</term>：支持
<!-- end id323 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.Fold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Fold](https://pytorch.org/docs/2.14/generated/torch.nn.Fold.html)

**产品支持情况**：

<!-- npu="910b" id324 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id324 -->
<!-- npu="A3" id325 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id325 -->
<!-- npu="950" id326 -->
- <term>Ascend 950DT</term>：支持
<!-- end id326 -->

**限制与说明**： `input`仅支持fp16

</div>

## Pooling layers

### <code><i>class</i></code> torch.nn.MaxPool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxPool1d](https://pytorch.org/docs/2.14/generated/torch.nn.MaxPool1d.html)

**产品支持情况**：

<!-- npu="910b" id327 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id327 -->
<!-- npu="A3" id328 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id328 -->
<!-- npu="950" id329 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id329 -->

</div>

### <code><i>class</i></code> torch.nn.MaxPool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxPool2d](https://pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html)

**产品支持情况**：

<!-- npu="910b" id330 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id330 -->
<!-- npu="A3" id331 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id331 -->
<!-- npu="950" id332 -->
- <term>Ascend 950DT</term>：支持
<!-- end id332 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 通过设置`torch_npu.npu.use_compatible_impl(True)`，保证与PyTorch同名接口在内存一致性上对齐，例如：

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

### <code><i>class</i></code> torch.nn.MaxPool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxPool3d](https://pytorch.org/docs/2.14/generated/torch.nn.MaxPool3d.html)

**产品支持情况**：

<!-- npu="910b" id333 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id333 -->
<!-- npu="A3" id334 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id334 -->
<!-- npu="950" id335 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id335 -->

</div>

### <code><i>class</i></code> torch.nn.MaxUnpool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxUnpool1d](https://pytorch.org/docs/2.14/generated/torch.nn.MaxUnpool1d.html)

**产品支持情况**：

<!-- npu="910b" id336 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id336 -->
<!-- npu="A3" id337 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id337 -->
<!-- npu="950" id338 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id338 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.MaxUnpool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxUnpool2d](https://pytorch.org/docs/2.14/generated/torch.nn.MaxUnpool2d.html)

**产品支持情况**：

<!-- npu="910b" id339 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id339 -->
<!-- npu="A3" id340 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id340 -->
<!-- npu="950" id341 -->
- <term>Ascend 950DT</term>：支持
<!-- end id341 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.MaxUnpool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxUnpool3d](https://pytorch.org/docs/2.14/generated/torch.nn.MaxUnpool3d.html)

**产品支持情况**：

<!-- npu="910b" id342 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id342 -->
<!-- npu="A3" id343 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id343 -->
<!-- npu="950" id344 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id344 -->

</div>

### <code><i>class</i></code> torch.nn.AvgPool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AvgPool1d](https://pytorch.org/docs/2.14/generated/torch.nn.AvgPool1d.html)

**产品支持情况**：

<!-- npu="910b" id345 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id345 -->
<!-- npu="A3" id346 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id346 -->
<!-- npu="950" id347 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id347 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.AvgPool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AvgPool2d](https://pytorch.org/docs/2.14/generated/torch.nn.AvgPool2d.html)

**产品支持情况**：

<!-- npu="910b" id348 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id348 -->
<!-- npu="A3" id349 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id349 -->
<!-- npu="950" id350 -->
- <term>Ascend 950DT</term>：支持
<!-- end id350 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.AvgPool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AvgPool3d](https://pytorch.org/docs/2.14/generated/torch.nn.AvgPool3d.html)

**产品支持情况**：

<!-- npu="910b" id351 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id351 -->
<!-- npu="A3" id352 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id352 -->
<!-- npu="950" id353 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id353 -->

</div>

### <code><i>class</i></code> torch.nn.LPPool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LPPool1d](https://pytorch.org/docs/2.14/generated/torch.nn.LPPool1d.html)

**产品支持情况**：

<!-- npu="910b" id354 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id354 -->
<!-- npu="A3" id355 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id355 -->
<!-- npu="950" id356 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id356 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### <code><i>class</i></code> torch.nn.LPPool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LPPool2d](https://pytorch.org/docs/2.14/generated/torch.nn.LPPool2d.html)

**产品支持情况**：

<!-- npu="910b" id357 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id357 -->
<!-- npu="A3" id358 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id358 -->
<!-- npu="950" id359 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id359 -->

**限制与说明**： `input`仅支持fp16，fp32，int16，int32，int64，bool

</div>

### <code><i>class</i></code> torch.nn.AdaptiveMaxPool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveMaxPool1d](https://pytorch.org/docs/2.14/generated/torch.nn.AdaptiveMaxPool1d.html)

**产品支持情况**：

<!-- npu="910b" id360 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id360 -->
<!-- npu="A3" id361 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id361 -->
<!-- npu="950" id362 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id362 -->

</div>

### <code><i>class</i></code> torch.nn.AdaptiveMaxPool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveMaxPool2d](https://pytorch.org/docs/2.14/generated/torch.nn.AdaptiveMaxPool2d.html)

**产品支持情况**：

<!-- npu="910b" id363 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id363 -->
<!-- npu="A3" id364 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id364 -->
<!-- npu="950" id365 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id365 -->

</div>

### <code><i>class</i></code> torch.nn.AdaptiveMaxPool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveMaxPool3d](https://pytorch.org/docs/2.14/generated/torch.nn.AdaptiveMaxPool3d.html)

**产品支持情况**：

<!-- npu="910b" id366 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id366 -->
<!-- npu="A3" id367 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id367 -->
<!-- npu="950" id368 -->
- <term>Ascend 950DT</term>：支持
<!-- end id368 -->

**限制与说明**： `input`仅支持fp32，fp64

</div>

### <code><i>class</i></code> torch.nn.AdaptiveAvgPool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveAvgPool1d](https://pytorch.org/docs/2.14/generated/torch.nn.AdaptiveAvgPool1d.html)

**产品支持情况**：

<!-- npu="910b" id369 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id369 -->
<!-- npu="A3" id370 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id370 -->
<!-- npu="950" id371 -->
- <term>Ascend 950DT</term>：支持
<!-- end id371 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.AdaptiveAvgPool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveAvgPool2d](https://pytorch.org/docs/2.14/generated/torch.nn.AdaptiveAvgPool2d.html)

**产品支持情况**：

<!-- npu="910b" id372 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id372 -->
<!-- npu="A3" id373 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id373 -->
<!-- npu="950" id374 -->
- <term>Ascend 950DT</term>：支持
<!-- end id374 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.AdaptiveAvgPool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveAvgPool3d](https://pytorch.org/docs/2.14/generated/torch.nn.AdaptiveAvgPool3d.html)

**产品支持情况**：

<!-- npu="910b" id375 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id375 -->
<!-- npu="A3" id376 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id376 -->
<!-- npu="950" id377 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id377 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

## Padding Layers

### <code><i>class</i></code> torch.nn.ReflectionPad1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReflectionPad1d](https://pytorch.org/docs/2.14/generated/torch.nn.ReflectionPad1d.html)

**产品支持情况**：

<!-- npu="910b" id378 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id378 -->
<!-- npu="A3" id379 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id379 -->
<!-- npu="950" id380 -->
- <term>Ascend 950DT</term>：支持
<!-- end id380 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.ReflectionPad2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReflectionPad2d](https://pytorch.org/docs/2.14/generated/torch.nn.ReflectionPad2d.html)

**产品支持情况**：

<!-- npu="910b" id381 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id381 -->
<!-- npu="A3" id382 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id382 -->
<!-- npu="950" id383 -->
- <term>Ascend 950DT</term>：支持
<!-- end id383 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.ReflectionPad3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReflectionPad3d](https://pytorch.org/docs/2.14/generated/torch.nn.ReflectionPad3d.html)

**产品支持情况**：

<!-- npu="910b" id384 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id384 -->
<!-- npu="A3" id385 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id385 -->
<!-- npu="950" id386 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id386 -->

</div>

### <code><i>class</i></code> torch.nn.ReplicationPad1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReplicationPad1d](https://pytorch.org/docs/2.14/generated/torch.nn.ReplicationPad1d.html)

**产品支持情况**：

<!-- npu="910b" id387 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id387 -->
<!-- npu="A3" id388 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id388 -->
<!-- npu="950" id389 -->
- <term>Ascend 950DT</term>：支持
<!-- end id389 -->

**限制与说明**： `input`仅支持fp16，fp32，complex64，complex128

</div>

### <code><i>class</i></code> torch.nn.ReplicationPad2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReplicationPad2d](https://pytorch.org/docs/2.14/generated/torch.nn.ReplicationPad2d.html)

**产品支持情况**：

<!-- npu="910b" id390 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id390 -->
<!-- npu="A3" id391 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id391 -->
<!-- npu="950" id392 -->
- <term>Ascend 950DT</term>：支持
<!-- end id392 -->

**限制与说明**： `input`仅支持fp16，fp32，complex64，complex128

</div>

### <code><i>class</i></code> torch.nn.ReplicationPad3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReplicationPad3d](https://pytorch.org/docs/2.14/generated/torch.nn.ReplicationPad3d.html)

**产品支持情况**：

<!-- npu="910b" id393 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id393 -->
<!-- npu="A3" id394 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id394 -->
<!-- npu="950" id395 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id395 -->

</div>

### <code><i>class</i></code> torch.nn.ZeroPad1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ZeroPad1d](https://pytorch.org/docs/2.14/generated/torch.nn.ZeroPad1d.html)

**产品支持情况**：

<!-- npu="910b" id396 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id396 -->
<!-- npu="A3" id397 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id397 -->
<!-- npu="950" id398 -->
- <term>Ascend 950DT</term>：支持
<!-- end id398 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，complex64，complex128
- 支持2-3维

</div>

### <code><i>class</i></code> torch.nn.ZeroPad2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ZeroPad2d](https://pytorch.org/docs/2.14/generated/torch.nn.ZeroPad2d.html)

**产品支持情况**：

<!-- npu="910b" id399 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id399 -->
<!-- npu="A3" id400 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id400 -->
<!-- npu="950" id401 -->
- <term>Ascend 950DT</term>：支持
<!-- end id401 -->

**限制与说明**： 可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.nn.ZeroPad3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ZeroPad3d](https://pytorch.org/docs/2.14/generated/torch.nn.ZeroPad3d.html)

**产品支持情况**：

<!-- npu="910b" id402 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id402 -->
<!-- npu="A3" id403 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id403 -->
<!-- npu="950" id404 -->
- <term>Ascend 950DT</term>：支持
<!-- end id404 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，complex64，complex128
- 支持5-6维

</div>

### <code><i>class</i></code> torch.nn.ConstantPad1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConstantPad1d](https://pytorch.org/docs/2.14/generated/torch.nn.ConstantPad1d.html)

**产品支持情况**：

<!-- npu="910b" id405 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id405 -->
<!-- npu="A3" id406 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id406 -->
<!-- npu="950" id407 -->
- <term>Ascend 950DT</term>：支持
<!-- end id407 -->

**限制与说明**：

- `input`仅支持int8，bool
- 在输入`x`为六维以上时可能会出现性能下降问题

</div>

### <code><i>class</i></code> torch.nn.ConstantPad2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConstantPad2d](https://pytorch.org/docs/2.14/generated/torch.nn.ConstantPad2d.html)

**产品支持情况**：

<!-- npu="910b" id408 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id408 -->
<!-- npu="A3" id409 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id409 -->
<!-- npu="950" id410 -->
- <term>Ascend 950DT</term>：支持
<!-- end id410 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 在输入`x`为六维以上时可能会出现性能下降问题

</div>

### <code><i>class</i></code> torch.nn.ConstantPad3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConstantPad3d](https://pytorch.org/docs/2.14/generated/torch.nn.ConstantPad3d.html)

**产品支持情况**：

<!-- npu="910b" id411 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id411 -->
<!-- npu="A3" id412 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id412 -->
<!-- npu="950" id413 -->
- <term>Ascend 950DT</term>：支持
<!-- end id413 -->

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 在输入`x`为六维以上时可能会出现性能下降问题

</div>

## Non-linear Activations (weighted sum, nonlinearity)

### <code><i>class</i></code> torch.nn.ELU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ELU](https://pytorch.org/docs/2.14/generated/torch.nn.ELU.html)

**产品支持情况**：

<!-- npu="910b" id414 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id414 -->
<!-- npu="A3" id415 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id415 -->
<!-- npu="950" id416 -->
- <term>Ascend 950DT</term>：支持
<!-- end id416 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64

</div>

### <code><i>class</i></code> torch.nn.Hardshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Hardshrink](https://pytorch.org/docs/2.14/generated/torch.nn.Hardshrink.html)

**产品支持情况**：

<!-- npu="910b" id417 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id417 -->
<!-- npu="A3" id418 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id418 -->
<!-- npu="950" id419 -->
- <term>Ascend 950DT</term>：支持
<!-- end id419 -->

**限制与说明**：

- `input`仅支持fp16，fp32
- 可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.nn.Hardsigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Hardsigmoid](https://pytorch.org/docs/2.14/generated/torch.nn.Hardsigmoid.html)

**产品支持情况**：

<!-- npu="910b" id420 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id420 -->
<!-- npu="A3" id421 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id421 -->
<!-- npu="950" id422 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id422 -->

**限制与说明**：

- `input`仅支持fp16，fp32，int32
- 可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.nn.Hardtanh

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Hardtanh](https://pytorch.org/docs/2.14/generated/torch.nn.Hardtanh.html)

**产品支持情况**：

<!-- npu="910b" id423 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id423 -->
<!-- npu="A3" id424 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id424 -->
<!-- npu="950" id425 -->
- <term>Ascend 950DT</term>：支持
<!-- end id425 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### <code><i>class</i></code> torch.nn.Hardswish

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Hardswish](https://pytorch.org/docs/2.14/generated/torch.nn.Hardswish.html)

**产品支持情况**：

<!-- npu="910b" id426 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id426 -->
<!-- npu="A3" id427 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id427 -->
<!-- npu="950" id428 -->
- <term>Ascend 950DT</term>：支持
<!-- end id428 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.LeakyReLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LeakyReLU](https://pytorch.org/docs/2.14/generated/torch.nn.LeakyReLU.html)

**产品支持情况**：

<!-- npu="910b" id429 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id429 -->
<!-- npu="A3" id430 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id430 -->
<!-- npu="950" id431 -->
- <term>Ascend 950DT</term>：支持
<!-- end id431 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64

</div>

### <code><i>class</i></code> torch.nn.LogSigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LogSigmoid](https://pytorch.org/docs/2.14/generated/torch.nn.LogSigmoid.html)

**产品支持情况**：

<!-- npu="910b" id432 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id432 -->
<!-- npu="A3" id433 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id433 -->
<!-- npu="950" id434 -->
- <term>Ascend 950DT</term>：支持
<!-- end id434 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.MultiheadAttention

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MultiheadAttention](https://pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)

**产品支持情况**：

<!-- npu="910b" id435 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id435 -->
<!-- npu="A3" id436 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id436 -->
<!-- npu="950" id437 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id437 -->

**限制与说明**： `query`、`key`、`value`仅支持bf16，fp16，fp32

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MultiheadAttention.forward](https://pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward)

**产品支持情况**：

<!-- npu="910b" id438 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id438 -->
<!-- npu="A3" id439 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id439 -->
<!-- npu="950" id440 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id440 -->

**限制与说明**： `query`、`key`、`value`仅支持bf16，fp16，fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.PReLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.PReLU](https://pytorch.org/docs/2.14/generated/torch.nn.PReLU.html)

**产品支持情况**：

<!-- npu="910b" id441 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id441 -->
<!-- npu="A3" id442 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id442 -->
<!-- npu="950" id443 -->
- <term>Ascend 950DT</term>：支持
<!-- end id443 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.nn.ReLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReLU](https://pytorch.org/docs/2.14/generated/torch.nn.ReLU.html)

**产品支持情况**：

<!-- npu="910b" id444 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id444 -->
<!-- npu="A3" id445 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id445 -->
<!-- npu="950" id446 -->
- <term>Ascend 950DT</term>：支持
<!-- end id446 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### <code><i>class</i></code> torch.nn.ReLU6

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReLU6](https://pytorch.org/docs/2.14/generated/torch.nn.ReLU6.html)

**产品支持情况**：

<!-- npu="910b" id447 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id447 -->
<!-- npu="A3" id448 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id448 -->
<!-- npu="950" id449 -->
- <term>Ascend 950DT</term>：支持
<!-- end id449 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### <code><i>class</i></code> torch.nn.RReLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.RReLU](https://pytorch.org/docs/2.14/generated/torch.nn.RReLU.html)

**产品支持情况**：

<!-- npu="910b" id450 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id450 -->
<!-- npu="A3" id451 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id451 -->
<!-- npu="950" id452 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id452 -->

</div>

### <code><i>class</i></code> torch.nn.SELU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.SELU](https://pytorch.org/docs/2.14/generated/torch.nn.SELU.html)

**产品支持情况**：

<!-- npu="910b" id453 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id453 -->
<!-- npu="A3" id454 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id454 -->
<!-- npu="950" id455 -->
- <term>Ascend 950DT</term>：支持
<!-- end id455 -->

**限制与说明**： `input`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### <code><i>class</i></code> torch.nn.CELU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.CELU](https://pytorch.org/docs/2.14/generated/torch.nn.CELU.html)

**产品支持情况**：

<!-- npu="910b" id456 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id456 -->
<!-- npu="A3" id457 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id457 -->
<!-- npu="950" id458 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id458 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.GELU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GELU](https://pytorch.org/docs/2.14/generated/torch.nn.GELU.html)

**产品支持情况**：

<!-- npu="910b" id459 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id459 -->
<!-- npu="A3" id460 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id460 -->
<!-- npu="950" id461 -->
- <term>Ascend 950DT</term>：支持
<!-- end id461 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- `approximate`参数仅支持设置为`tanh`

</div>

### <code><i>class</i></code> torch.nn.Sigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Sigmoid](https://pytorch.org/docs/2.14/generated/torch.nn.Sigmoid.html)

**产品支持情况**：

<!-- npu="910b" id462 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id462 -->
<!-- npu="A3" id463 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id463 -->
<!-- npu="950" id464 -->
- <term>Ascend 950DT</term>：支持
<!-- end id464 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### <code><i>class</i></code> torch.nn.SiLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.SiLU](https://pytorch.org/docs/2.14/generated/torch.nn.SiLU.html)

**产品支持情况**：

<!-- npu="910b" id465 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id465 -->
<!-- npu="A3" id466 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id466 -->
<!-- npu="950" id467 -->
- <term>Ascend 950DT</term>：支持
<!-- end id467 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.Mish

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Mish](https://pytorch.org/docs/2.14/generated/torch.nn.Mish.html)

**产品支持情况**：

<!-- npu="910b" id468 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id468 -->
<!-- npu="A3" id469 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id469 -->
<!-- npu="950" id470 -->
- <term>Ascend 950DT</term>：支持
<!-- end id470 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.Softplus

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softplus](https://pytorch.org/docs/2.14/generated/torch.nn.Softplus.html)

**产品支持情况**：

<!-- npu="910b" id471 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id471 -->
<!-- npu="A3" id472 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id472 -->
<!-- npu="950" id473 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id473 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.Softshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softshrink](https://pytorch.org/docs/2.14/generated/torch.nn.Softshrink.html)

**产品支持情况**：

<!-- npu="910b" id474 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id474 -->
<!-- npu="A3" id475 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id475 -->
<!-- npu="950" id476 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id476 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.Softsign

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softsign](https://pytorch.org/docs/2.14/generated/torch.nn.Softsign.html)

**产品支持情况**：

<!-- npu="910b" id477 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id477 -->
<!-- npu="A3" id478 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id478 -->
<!-- npu="950" id479 -->
- <term>Ascend 950DT</term>：支持
<!-- end id479 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### <code><i>class</i></code> torch.nn.Tanh

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Tanh](https://pytorch.org/docs/2.14/generated/torch.nn.Tanh.html)

**产品支持情况**：

<!-- npu="910b" id480 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id480 -->
<!-- npu="A3" id481 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id481 -->
<!-- npu="950" id482 -->
- <term>Ascend 950DT</term>：支持
<!-- end id482 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### <code><i>class</i></code> torch.nn.Tanhshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Tanhshrink](https://pytorch.org/docs/2.14/generated/torch.nn.Tanhshrink.html)

**产品支持情况**：

<!-- npu="910b" id483 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id483 -->
<!-- npu="A3" id484 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id484 -->
<!-- npu="950" id485 -->
- <term>Ascend 950DT</term>：支持
<!-- end id485 -->

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64
- 可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.nn.Threshold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Threshold](https://pytorch.org/docs/2.14/generated/torch.nn.Threshold.html)

**产品支持情况**：

<!-- npu="910b" id486 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id486 -->
<!-- npu="A3" id487 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id487 -->
<!-- npu="950" id488 -->
- <term>Ascend 950DT</term>：支持
<!-- end id488 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

### <code><i>class</i></code> torch.nn.GLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GLU](https://pytorch.org/docs/2.14/generated/torch.nn.GLU.html)

**产品支持情况**：

<!-- npu="910b" id489 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id489 -->
<!-- npu="A3" id490 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id490 -->
<!-- npu="950" id491 -->
- <term>Ascend 950DT</term>：支持
<!-- end id491 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

## Non-linear Activations (other)

### <code><i>class</i></code> torch.nn.Softmin

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softmin](https://pytorch.org/docs/2.14/generated/torch.nn.Softmin.html)

**产品支持情况**：

<!-- npu="910b" id492 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id492 -->
<!-- npu="A3" id493 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id493 -->
<!-- npu="950" id494 -->
- <term>Ascend 950DT</term>：支持
<!-- end id494 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.Softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softmax](https://pytorch.org/docs/2.14/generated/torch.nn.Softmax.html)

**产品支持情况**：

<!-- npu="910b" id495 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id495 -->
<!-- npu="A3" id496 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id496 -->
<!-- npu="950" id497 -->
- <term>Ascend 950DT</term>：支持
<!-- end id497 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64

</div>

### <code><i>class</i></code> torch.nn.Softmax2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softmax2d](https://pytorch.org/docs/2.14/generated/torch.nn.Softmax2d.html)

**产品支持情况**：

<!-- npu="910b" id498 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id498 -->
<!-- npu="A3" id499 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id499 -->
<!-- npu="950" id500 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id500 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.LogSoftmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LogSoftmax](https://pytorch.org/docs/2.14/generated/torch.nn.LogSoftmax.html)

**产品支持情况**：

<!-- npu="910b" id501 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id501 -->
<!-- npu="A3" id502 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id502 -->
<!-- npu="950" id503 -->
- <term>Ascend 950DT</term>：支持
<!-- end id503 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.AdaptiveLogSoftmaxWithLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveLogSoftmaxWithLoss](https://pytorch.org/docs/2.14/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html)

**产品支持情况**：

<!-- npu="910b" id504 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id504 -->
<!-- npu="A3" id505 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id505 -->
<!-- npu="950" id506 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id506 -->

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob](https://pytorch.org/docs/2.14/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob)

**产品支持情况**：

<!-- npu="910b" id507 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id507 -->
<!-- npu="A3" id508 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id508 -->
<!-- npu="950" id509 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id509 -->

</div>

> <font size="3">predict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveLogSoftmaxWithLoss.predict](https://pytorch.org/docs/2.14/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss.predict)

**产品支持情况**：

<!-- npu="910b" id510 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id510 -->
<!-- npu="A3" id511 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id511 -->
<!-- npu="950" id512 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id512 -->

</div>

</div>

## Normalization Layers

### <code><i>class</i></code> torch.nn.BatchNorm1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.BatchNorm1d](https://pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html)

**产品支持情况**：

<!-- npu="910b" id513 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id513 -->
<!-- npu="A3" id514 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id514 -->
<!-- npu="950" id515 -->
- <term>Ascend 950DT</term>：支持
<!-- end id515 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.BatchNorm2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.BatchNorm2d](https://pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)

**产品支持情况**：

<!-- npu="910b" id516 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id516 -->
<!-- npu="A3" id517 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id517 -->
<!-- npu="950" id518 -->
- <term>Ascend 950DT</term>：支持
<!-- end id518 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.BatchNorm3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.BatchNorm3d](https://pytorch.org/docs/2.14/generated/torch.nn.BatchNorm3d.html)

**产品支持情况**：

<!-- npu="910b" id519 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id519 -->
<!-- npu="A3" id520 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id520 -->
<!-- npu="950" id521 -->
- <term>Ascend 950DT</term>：支持
<!-- end id521 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.LazyBatchNorm1d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyBatchNorm1d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyBatchNorm1d.html#torch.nn.LazyBatchNorm1d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id522 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id522 -->
<!-- npu="A3" id523 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id523 -->
<!-- npu="950" id524 -->
- <term>Ascend 950DT</term>：支持
<!-- end id524 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyBatchNorm2d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyBatchNorm2d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyBatchNorm2d.html#torch.nn.LazyBatchNorm2d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id525 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id525 -->
<!-- npu="A3" id526 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id526 -->
<!-- npu="950" id527 -->
- <term>Ascend 950DT</term>：支持
<!-- end id527 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyBatchNorm3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyBatchNorm3d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyBatchNorm3d.html#torch.nn.LazyBatchNorm3d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id528 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id528 -->
<!-- npu="A3" id529 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id529 -->
<!-- npu="950" id530 -->
- <term>Ascend 950DT</term>：支持
<!-- end id530 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.GroupNorm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GroupNorm](https://pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html)

**产品支持情况**：

<!-- npu="910b" id531 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id531 -->
<!-- npu="A3" id532 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id532 -->
<!-- npu="950" id533 -->
- <term>Ascend 950DT</term>：支持
<!-- end id533 -->

**限制与说明**：

- `input`仅支持fp32
- `eps`参数需大于0
- 不支持`jit_compile=True`的场景
- 该API仅支持2维及以上的输入`input`。该API反向传播时，要求输入维度为4维、`num_groups`能被32整除、C轴维度能被(10 * `num_groups`)整除

</div>

### <code><i>class</i></code> torch.nn.SyncBatchNorm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.SyncBatchNorm](https://pytorch.org/docs/2.14/generated/torch.nn.SyncBatchNorm.html)

**产品支持情况**：

<!-- npu="910b" id534 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id534 -->
<!-- npu="A3" id535 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id535 -->
<!-- npu="950" id536 -->
- <term>Ascend 950DT</term>：支持
<!-- end id536 -->

**限制与说明**： `input`仅支持fp16，fp32

> <font size="3">convert_sync_batchnorm()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.SyncBatchNorm.convert_sync_batchnorm](https://pytorch.org/docs/2.14/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm)

**产品支持情况**：

<!-- npu="910b" id537 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id537 -->
<!-- npu="A3" id538 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id538 -->
<!-- npu="950" id539 -->
- <term>Ascend 950DT</term>：支持
<!-- end id539 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyInstanceNorm1d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyInstanceNorm1d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyInstanceNorm1d.html#torch.nn.LazyInstanceNorm1d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id540 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id540 -->
<!-- npu="A3" id541 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id541 -->
<!-- npu="950" id542 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id542 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyInstanceNorm2d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyInstanceNorm2d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyInstanceNorm2d.html#torch.nn.LazyInstanceNorm2d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id543 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id543 -->
<!-- npu="A3" id544 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id544 -->
<!-- npu="950" id545 -->
- <term>Ascend 950DT</term>：支持
<!-- end id545 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LazyInstanceNorm3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyInstanceNorm3d.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyInstanceNorm3d.html#torch.nn.LazyInstanceNorm3d.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id546 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id546 -->
<!-- npu="A3" id547 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id547 -->
<!-- npu="950" id548 -->
- <term>Ascend 950DT</term>：支持
<!-- end id548 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.LayerNorm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LayerNorm](https://pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)

**产品支持情况**：

<!-- npu="910b" id549 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id549 -->
<!-- npu="A3" id550 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id550 -->
<!-- npu="950" id551 -->
- <term>Ascend 950DT</term>：支持
<!-- end id551 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 通过`torch_npu.npu.use_compatible_impl(True)`，设置该接口从`aclnnLayerNorm`算子切换为`aclnnFastLayerNorm`算子，保证与PyTorch同名接口在内存一致性上对齐，例如：

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

## Recurrent Layers

### <code><i>class</i></code> torch.nn.RNNBase

<div style="margin-left: 2em">

**原生文档**：[torch.nn.RNNBase](https://pytorch.org/docs/2.14/generated/torch.nn.RNNBase.html)

**产品支持情况**：

<!-- npu="910b" id552 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id552 -->
<!-- npu="A3" id553 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id553 -->
<!-- npu="950" id554 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id554 -->

> <font size="3">flatten_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.RNNBase.flatten_parameters](https://pytorch.org/docs/2.14/generated/torch.nn.RNNBase.html#torch.nn.RNNBase.flatten_parameters)

**产品支持情况**：

<!-- npu="910b" id555 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id555 -->
<!-- npu="A3" id556 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id556 -->
<!-- npu="950" id557 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id557 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.RNN

<div style="margin-left: 2em">

**原生文档**：[torch.nn.RNN](https://pytorch.org/docs/2.14/generated/torch.nn.RNN.html)

**产品支持情况**：

<!-- npu="910b" id558 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id558 -->
<!-- npu="A3" id559 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id559 -->
<!-- npu="950" id560 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id560 -->

</div>

### <code><i>class</i></code> torch.nn.LSTM

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LSTM](https://pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)

**产品支持情况**：

<!-- npu="910b" id561 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id561 -->
<!-- npu="A3" id562 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id562 -->
<!-- npu="950" id563 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id563 -->

**限制与说明**：

- `input`仅支持fp32
- 不支持`proj_size`参数
- 不支持`dropout`参数
- 入参`input`不支持2维

</div>

### <code><i>class</i></code> torch.nn.GRU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GRU](https://pytorch.org/docs/2.14/generated/torch.nn.GRU.html)

**产品支持情况**：

<!-- npu="910b" id564 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id564 -->
<!-- npu="A3" id565 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id565 -->
<!-- npu="950" id566 -->
- <term>Ascend 950DT</term>：支持
<!-- end id566 -->

**限制与说明**：

- 不支持`dropout`参数
- 不支持变长序列输入

</div>

### <code><i>class</i></code> torch.nn.RNNCell

<div style="margin-left: 2em">

**原生文档**：[torch.nn.RNNCell](https://pytorch.org/docs/2.14/generated/torch.nn.RNNCell.html)

**产品支持情况**：

<!-- npu="910b" id567 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id567 -->
<!-- npu="A3" id568 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id568 -->
<!-- npu="950" id569 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id569 -->

</div>

### <code><i>class</i></code> torch.nn.LSTMCell

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LSTMCell](https://pytorch.org/docs/2.14/generated/torch.nn.LSTMCell.html)

**产品支持情况**：

<!-- npu="910b" id570 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id570 -->
<!-- npu="A3" id571 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id571 -->
<!-- npu="950" id572 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id572 -->

**限制与说明**： 接口暂不支持`jit_compile=False`，如需在该模式下使用，请将"DynamicGRUV2"添加至"NPU_FUZZY_COMPILE_BLACKLIST"选项内，具体操作可参考[添加二进制黑名单示例](../appendixes/example_of_adding_a_binary_blocklist.md)

</div>

### <code><i>class</i></code> torch.nn.GRUCell

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GRUCell](https://pytorch.org/docs/2.14/generated/torch.nn.GRUCell.html)

**产品支持情况**：

<!-- npu="910b" id573 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id573 -->
<!-- npu="A3" id574 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id574 -->
<!-- npu="950" id575 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id575 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

## Transformer Layers

### <code><i>class</i></code> torch.nn.Transformer

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Transformer](https://pytorch.org/docs/2.14/generated/torch.nn.Transformer.html)

**产品支持情况**：

<!-- npu="910b" id576 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id576 -->
<!-- npu="A3" id577 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id577 -->
<!-- npu="950" id578 -->
- <term>Ascend 950DT</term>：支持
<!-- end id578 -->

**限制与说明**： `src`、`tgt`仅支持fp16，fp32

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Transformer.forward](https://pytorch.org/docs/2.14/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward)

**产品支持情况**：

<!-- npu="910b" id579 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id579 -->
<!-- npu="A3" id580 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id580 -->
<!-- npu="950" id581 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id581 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerEncoder

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerEncoder](https://pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html)

**产品支持情况**：

<!-- npu="910b" id582 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id582 -->
<!-- npu="A3" id583 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id583 -->
<!-- npu="950" id584 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id584 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerEncoder.forward](https://pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder.forward)

**产品支持情况**：

<!-- npu="910b" id585 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id585 -->
<!-- npu="A3" id586 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id586 -->
<!-- npu="950" id587 -->
- <term>Ascend 950DT</term>：支持
<!-- end id587 -->

**限制与说明**： `src`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerDecoder

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerDecoder](https://pytorch.org/docs/2.14/generated/torch.nn.TransformerDecoder.html)

**产品支持情况**：

<!-- npu="910b" id588 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id588 -->
<!-- npu="A3" id589 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id589 -->
<!-- npu="950" id590 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id590 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerDecoder.forward](https://pytorch.org/docs/2.14/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder.forward)

**产品支持情况**：

<!-- npu="910b" id591 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id591 -->
<!-- npu="A3" id592 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id592 -->
<!-- npu="950" id593 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id593 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerEncoderLayer

<div style="margin-left: 2em">

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerEncoderLayer.forward](https://pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer.forward)

**产品支持情况**：

<!-- npu="910b" id594 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id594 -->
<!-- npu="A3" id595 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id595 -->
<!-- npu="950" id596 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id596 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.TransformerDecoderLayer

<div style="margin-left: 2em">

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerDecoderLayer.forward](https://pytorch.org/docs/2.14/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer.forward)

**产品支持情况**：

<!-- npu="910b" id597 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id597 -->
<!-- npu="A3" id598 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id598 -->
<!-- npu="950" id599 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id599 -->

</div>

</div>

## Linear Layers

### <code><i>class</i></code> torch.nn.Identity

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Identity](https://pytorch.org/docs/2.14/generated/torch.nn.Identity.html)

**产品支持情况**：

<!-- npu="910b" id600 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id600 -->
<!-- npu="A3" id601 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id601 -->
<!-- npu="950" id602 -->
- <term>Ascend 950DT</term>：支持
<!-- end id602 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.nn.Linear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Linear](https://pytorch.org/docs/2.14/generated/torch.nn.Linear.html)

**产品支持情况**：

<!-- npu="910b" id603 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id603 -->
<!-- npu="A3" id604 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id604 -->
<!-- npu="950" id605 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id605 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.Bilinear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Bilinear](https://pytorch.org/docs/2.14/generated/torch.nn.Bilinear.html)

**产品支持情况**：

<!-- npu="910b" id606 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id606 -->
<!-- npu="A3" id607 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id607 -->
<!-- npu="950" id608 -->
- <term>Ascend 950DT</term>：支持
<!-- end id608 -->

**限制与说明**： `input1`、`input2`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.LazyLinear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyLinear](https://pytorch.org/docs/2.14/generated/torch.nn.LazyLinear.html)

**产品支持情况**：

<!-- npu="910b" id609 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id609 -->
<!-- npu="A3" id610 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id610 -->
<!-- npu="950" id611 -->
- <term>Ascend 950DT</term>：支持
<!-- end id611 -->

**限制与说明**： `input`仅支持fp16，fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyLinear.cls_to_become](https://pytorch.org/docs/2.14/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear.cls_to_become)

**产品支持情况**：

<!-- npu="910b" id612 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id612 -->
<!-- npu="A3" id613 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id613 -->
<!-- npu="950" id614 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id614 -->

</div>

</div>

## Dropout Layers

### <code><i>class</i></code> torch.nn.Dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Dropout](https://pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)

**产品支持情况**：

<!-- npu="910b" id615 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id615 -->
<!-- npu="A3" id616 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id616 -->
<!-- npu="950" id617 -->
- <term>Ascend 950DT</term>：支持
<!-- end id617 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.Dropout2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Dropout2d](https://pytorch.org/docs/2.14/generated/torch.nn.Dropout2d.html)

**产品支持情况**：

<!-- npu="910b" id618 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id618 -->
<!-- npu="A3" id619 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id619 -->
<!-- npu="950" id620 -->
- <term>Ascend 950DT</term>：支持
<!-- end id620 -->

**限制与说明**： `input`仅支持fp16，fp32，int64，bool

</div>

### <code><i>class</i></code> torch.nn.AlphaDropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AlphaDropout](https://pytorch.org/docs/2.14/generated/torch.nn.AlphaDropout.html)

**产品支持情况**：

<!-- npu="910b" id621 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id621 -->
<!-- npu="A3" id622 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id622 -->
<!-- npu="950" id623 -->
- <term>Ascend 950DT</term>：支持
<!-- end id623 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.FeatureAlphaDropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.FeatureAlphaDropout](https://pytorch.org/docs/2.14/generated/torch.nn.FeatureAlphaDropout.html)

**产品支持情况**：

<!-- npu="910b" id624 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id624 -->
<!-- npu="A3" id625 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id625 -->
<!-- npu="950" id626 -->
- <term>Ascend 950DT</term>：支持
<!-- end id626 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

## Sparse Layers

### <code><i>class</i></code> torch.nn.Embedding

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Embedding](https://pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)

**产品支持情况**：

<!-- npu="910b" id627 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id627 -->
<!-- npu="A3" id628 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id628 -->
<!-- npu="950" id629 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id629 -->

**限制与说明**：

- `input`仅支持int32，int64
- 属性`max_norm`仅支持非负值，不支持nan

> <font size="3">from_pretrained()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Embedding.from_pretrained](https://pytorch.org/docs/2.14/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained)

**产品支持情况**：

<!-- npu="910b" id630 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id630 -->
<!-- npu="A3" id631 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id631 -->
<!-- npu="950" id632 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id632 -->

**限制与说明**： `embeddings`仅支持fp64

</div>

</div>

### <code><i>class</i></code> torch.nn.EmbeddingBag

<div style="margin-left: 2em">

**原生文档**：[torch.nn.EmbeddingBag](https://pytorch.org/docs/2.14/generated/torch.nn.EmbeddingBag.html)

**产品支持情况**：

<!-- npu="910b" id633 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id633 -->
<!-- npu="A3" id634 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id634 -->
<!-- npu="950" id635 -->
- <term>Ascend 950DT</term>：支持
<!-- end id635 -->

**限制与说明**：

- `input`仅支持int32，int64
- 仅支持`max_norm`大于等于0

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.EmbeddingBag.forward](https://pytorch.org/docs/2.14/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward)

**产品支持情况**：

<!-- npu="910b" id636 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id636 -->
<!-- npu="A3" id637 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id637 -->
<!-- npu="950" id638 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id638 -->

**限制与说明**： `input`、`offsets`仅支持int64

</div>

> <font size="3">from_pretrained()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.EmbeddingBag.from_pretrained](https://pytorch.org/docs/2.14/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.from_pretrained)

**产品支持情况**：

<!-- npu="910b" id639 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id639 -->
<!-- npu="A3" id640 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id640 -->
<!-- npu="950" id641 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id641 -->

**限制与说明**： `embeddings`仅支持int64

</div>

</div>

## Loss Functions

### <code><i>class</i></code> torch.nn.L1Loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.L1Loss](https://pytorch.org/docs/2.14/generated/torch.nn.L1Loss.html)

**产品支持情况**：

<!-- npu="910b" id642 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id642 -->
<!-- npu="A3" id643 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id643 -->
<!-- npu="950" id644 -->
- <term>Ascend 950DT</term>：支持
<!-- end id644 -->

**限制与说明**： `input`仅支持fp16，fp32，int64

</div>

### <code><i>class</i></code> torch.nn.MSELoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MSELoss](https://pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html)

**产品支持情况**：

<!-- npu="910b" id645 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id645 -->
<!-- npu="A3" id646 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id646 -->
<!-- npu="950" id647 -->
- <term>Ascend 950DT</term>：支持
<!-- end id647 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.CrossEntropyLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.CrossEntropyLoss](https://pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)

**产品支持情况**：

<!-- npu="910b" id648 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id648 -->
<!-- npu="A3" id649 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id649 -->
<!-- npu="950" id650 -->
- <term>Ascend 950DT</term>：支持
<!-- end id650 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.CTCLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.CTCLoss](https://pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html)

**产品支持情况**：

<!-- npu="910b" id651 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id651 -->
<!-- npu="A3" id652 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id652 -->
<!-- npu="950" id653 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id653 -->

**限制与说明**：

- `log_probs`仅支持fp32，fp64
- 不支持`log_probs` 2D输入

</div>

### <code><i>class</i></code> torch.nn.NLLLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.NLLLoss](https://pytorch.org/docs/2.14/generated/torch.nn.NLLLoss.html)

**产品支持情况**：

<!-- npu="910b" id654 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id654 -->
<!-- npu="A3" id655 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id655 -->
<!-- npu="950" id656 -->
- <term>Ascend 950DT</term>：支持
<!-- end id656 -->

**限制与说明**：

- `input`仅支持fp16，fp32
- `target`中的每个元素值应大于等于0且小于`input`的类别数

</div>

### <code><i>class</i></code> torch.nn.PoissonNLLLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.PoissonNLLLoss](https://pytorch.org/docs/2.14/generated/torch.nn.PoissonNLLLoss.html)

**产品支持情况**：

<!-- npu="910b" id657 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id657 -->
<!-- npu="A3" id658 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id658 -->
<!-- npu="950" id659 -->
- <term>Ascend 950DT</term>：支持
<!-- end id659 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，int64

</div>

### <code><i>class</i></code> torch.nn.GaussianNLLLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GaussianNLLLoss](https://pytorch.org/docs/2.14/generated/torch.nn.GaussianNLLLoss.html)

**产品支持情况**：

<!-- npu="910b" id660 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id660 -->
<!-- npu="A3" id661 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id661 -->
<!-- npu="950" id662 -->
- <term>Ascend 950DT</term>：支持
<!-- end id662 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，int16，int32，int64

</div>

### <code><i>class</i></code> torch.nn.KLDivLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.KLDivLoss](https://pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html)

**产品支持情况**：

<!-- npu="910b" id663 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id663 -->
<!-- npu="A3" id664 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id664 -->
<!-- npu="950" id665 -->
- <term>Ascend 950DT</term>：支持
<!-- end id665 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 当前`log_target`参数仅支持False

</div>

### <code><i>class</i></code> torch.nn.BCELoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.BCELoss](https://pytorch.org/docs/2.14/generated/torch.nn.BCELoss.html)

**产品支持情况**：

<!-- npu="910b" id666 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id666 -->
<!-- npu="A3" id667 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id667 -->
<!-- npu="950" id668 -->
- <term>Ascend 950DT</term>：支持
<!-- end id668 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.BCEWithLogitsLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html)

**产品支持情况**：

<!-- npu="910b" id669 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id669 -->
<!-- npu="A3" id670 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id670 -->
<!-- npu="950" id671 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id671 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 入参`target`不支持反向计算

</div>

### <code><i>class</i></code> torch.nn.MarginRankingLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MarginRankingLoss](https://pytorch.org/docs/2.14/generated/torch.nn.MarginRankingLoss.html)

**产品支持情况**：

<!-- npu="910b" id672 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id672 -->
<!-- npu="A3" id673 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id673 -->
<!-- npu="950" id674 -->
- <term>Ascend 950DT</term>：支持
<!-- end id674 -->

**限制与说明**： `input1`、`input2`、`target`仅支持bf16，fp16，fp32，int8，int32，int64

</div>

### <code><i>class</i></code> torch.nn.HingeEmbeddingLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.HingeEmbeddingLoss](https://pytorch.org/docs/2.14/generated/torch.nn.HingeEmbeddingLoss.html)

**产品支持情况**：

<!-- npu="910b" id675 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id675 -->
<!-- npu="A3" id676 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id676 -->
<!-- npu="950" id677 -->
- <term>Ascend 950DT</term>：支持
<!-- end id677 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### <code><i>class</i></code> torch.nn.MultiLabelMarginLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MultiLabelMarginLoss](https://pytorch.org/docs/2.14/generated/torch.nn.MultiLabelMarginLoss.html)

**产品支持情况**：

<!-- npu="910b" id678 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id678 -->
<!-- npu="A3" id679 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id679 -->
<!-- npu="950" id680 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id680 -->

</div>

### <code><i>class</i></code> torch.nn.HuberLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.HuberLoss](https://pytorch.org/docs/2.14/generated/torch.nn.HuberLoss.html)

**产品支持情况**：

<!-- npu="910b" id681 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id681 -->
<!-- npu="A3" id682 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id682 -->
<!-- npu="950" id683 -->
- <term>Ascend 950DT</term>：支持
<!-- end id683 -->

**限制与说明**：

- `input`仅支持fp32，fp64
- `target`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.nn.SmoothL1Loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.SmoothL1Loss](https://pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html)

**产品支持情况**：

<!-- npu="910b" id684 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id684 -->
<!-- npu="A3" id685 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id685 -->
<!-- npu="950" id686 -->
- <term>Ascend 950DT</term>：支持
<!-- end id686 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.MultiLabelSoftMarginLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/2.14/generated/torch.nn.MultiLabelSoftMarginLoss.html)

**产品支持情况**：

<!-- npu="910b" id687 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id687 -->
<!-- npu="A3" id688 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id688 -->
<!-- npu="950" id689 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id689 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### <code><i>class</i></code> torch.nn.CosineEmbeddingLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/2.14/generated/torch.nn.CosineEmbeddingLoss.html)

**产品支持情况**：

<!-- npu="910b" id690 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id690 -->
<!-- npu="A3" id691 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id691 -->
<!-- npu="950" id692 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id692 -->

</div>

### <code><i>class</i></code> torch.nn.MultiMarginLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MultiMarginLoss](https://pytorch.org/docs/2.14/generated/torch.nn.MultiMarginLoss.html)

**产品支持情况**：

<!-- npu="910b" id693 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id693 -->
<!-- npu="A3" id694 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id694 -->
<!-- npu="950" id695 -->
- <term>Ascend 950DT</term>：支持
<!-- end id695 -->

**限制与说明**：

- `input`仅支持fp32，fp64
- `target`仅支持int64
- 可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.nn.TripletMarginLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TripletMarginLoss](https://pytorch.org/docs/2.14/generated/torch.nn.TripletMarginLoss.html)

**产品支持情况**：

<!-- npu="910b" id696 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id696 -->
<!-- npu="A3" id697 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id697 -->
<!-- npu="950" id698 -->
- <term>Ascend 950DT</term>：支持
<!-- end id698 -->

**限制与说明**：

- `anchor`、`positive`、`negative`仅支持fp16，fp32，uint8，int8，int16，int32，int64
- 可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.nn.TripletMarginWithDistanceLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/2.14/generated/torch.nn.TripletMarginWithDistanceLoss.html)

**产品支持情况**：

<!-- npu="910b" id699 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id699 -->
<!-- npu="A3" id700 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id700 -->
<!-- npu="950" id701 -->
- <term>Ascend 950DT</term>：支持
<!-- end id701 -->

**限制与说明**： `anchor`仅支持bf16，fp16，fp32

</div>

## Vision Layers

### <code><i>class</i></code> torch.nn.PixelShuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.PixelShuffle](https://pytorch.org/docs/2.14/generated/torch.nn.PixelShuffle.html)

**产品支持情况**：

<!-- npu="910b" id702 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id702 -->
<!-- npu="A3" id703 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id703 -->
<!-- npu="950" id704 -->
- <term>Ascend 950DT</term>：支持
<!-- end id704 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### <code><i>class</i></code> torch.nn.PixelUnshuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.PixelUnshuffle](https://pytorch.org/docs/2.14/generated/torch.nn.PixelUnshuffle.html)

**产品支持情况**：

<!-- npu="910b" id705 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id705 -->
<!-- npu="A3" id706 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id706 -->
<!-- npu="950" id707 -->
- <term>Ascend 950DT</term>：支持
<!-- end id707 -->

**限制与说明**： `input`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### <code><i>class</i></code> torch.nn.Upsample

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Upsample](https://pytorch.org/docs/2.14/generated/torch.nn.Upsample.html)

**产品支持情况**：

<!-- npu="910b" id708 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id708 -->
<!-- npu="A3" id709 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id709 -->
<!-- npu="950" id710 -->
- <term>Ascend 950DT</term>：支持
<!-- end id710 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64

</div>

### <code><i>class</i></code> torch.nn.UpsamplingNearest2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.UpsamplingNearest2d](https://pytorch.org/docs/2.14/generated/torch.nn.UpsamplingNearest2d.html)

**版本说明**：该类已废弃，请使用`torch.nn.functional.interpolate`。

**产品支持情况**：

<!-- npu="910b" id711 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id711 -->
<!-- npu="A3" id712 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id712 -->
<!-- npu="950" id713 -->
- <term>Ascend 950DT</term>：支持
<!-- end id713 -->

**限制与说明**：

- `input`仅支持fp16，fp32，uint8
- 可能回退至CPU执行

</div>

## Shuffle Layers

### <code><i>class</i></code> torch.nn.ChannelShuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ChannelShuffle](https://pytorch.org/docs/2.14/generated/torch.nn.ChannelShuffle.html)

**产品支持情况**：

<!-- npu="910b" id714 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id714 -->
<!-- npu="A3" id715 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id715 -->
<!-- npu="950" id716 -->
- <term>Ascend 950DT</term>：支持
<!-- end id716 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

## DataParallel Layers (multi-GPU, distributed)

### <code><i>class</i></code> torch.nn.DataParallel

<div style="margin-left: 2em">

**原生文档**：[torch.nn.DataParallel](https://pytorch.org/docs/2.14/generated/torch.nn.DataParallel.html)

**产品支持情况**：

<!-- npu="910b" id717 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id717 -->
<!-- npu="A3" id718 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id718 -->
<!-- npu="950" id719 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id719 -->

</div>

### <code><i>class</i></code> torch.nn.parallel.DistributedDataParallel

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)

**产品支持情况**：

<!-- npu="910b" id720 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id720 -->
<!-- npu="A3" id721 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id721 -->
<!-- npu="950" id722 -->
- <term>Ascend 950DT</term>：支持
<!-- end id722 -->

> <font size="3">join()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.DistributedDataParallel.join](https://pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join)

**产品支持情况**：

<!-- npu="910b" id723 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id723 -->
<!-- npu="A3" id724 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id724 -->
<!-- npu="950" id725 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id725 -->

</div>

> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.DistributedDataParallel.join_hook](https://pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join_hook)

**产品支持情况**：

<!-- npu="910b" id726 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id726 -->
<!-- npu="A3" id727 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id727 -->
<!-- npu="950" id728 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id728 -->

</div>

> <font size="3">no_sync()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.DistributedDataParallel.no_sync](https://pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync)

**产品支持情况**：

<!-- npu="910b" id729 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id729 -->
<!-- npu="A3" id730 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id730 -->
<!-- npu="950" id731 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id731 -->

</div>

> <font size="3">register_comm_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.DistributedDataParallel.register_comm_hook](https://pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook)

**产品支持情况**：

<!-- npu="910b" id732 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id732 -->
<!-- npu="A3" id733 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id733 -->
<!-- npu="950" id734 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id734 -->

</div>

</div>

## Utilities

### torch.nn.utils.clip_grad_norm_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html)

**产品支持情况**：

<!-- npu="910b" id735 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id735 -->
<!-- npu="A3" id736 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id736 -->
<!-- npu="950" id737 -->
- <term>Ascend 950DT</term>：支持
<!-- end id737 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.nn.utils.clip_grad_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.clip_grad_norm](https://pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm.html)

**版本说明**：该接口已废弃，请使用`torch.nn.utils.clip_grad_norm_`。

**产品支持情况**：

<!-- npu="910b" id738 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id738 -->
<!-- npu="A3" id739 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id739 -->
<!-- npu="950" id740 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id740 -->

</div>

### torch.nn.utils.clip_grad_value_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_value_.html)

**产品支持情况**：

<!-- npu="910b" id741 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id741 -->
<!-- npu="A3" id742 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id742 -->
<!-- npu="950" id743 -->
- <term>Ascend 950DT</term>：支持
<!-- end id743 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.utils.vector_to_parameters

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.vector_to_parameters](https://pytorch.org/docs/2.14/generated/torch.nn.utils.vector_to_parameters.html)

**产品支持情况**：

<!-- npu="910b" id744 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id744 -->
<!-- npu="A3" id745 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id745 -->
<!-- npu="950" id746 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id746 -->

**限制与说明**： `vec`仅支持bf16，fp16，fp32，fp64，complex64

</div>

### torch.nn.utils.weight_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.weight_norm](https://pytorch.org/docs/2.14/generated/torch.nn.utils.weight_norm.html)

**版本说明**：该接口已废弃，请使用`torch.nn.utils.parametrizations.weight_norm`；参数访问和移除方式请参考原生文档的迁移说明。

**产品支持情况**：

<!-- npu="910b" id747 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id747 -->
<!-- npu="A3" id748 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id748 -->
<!-- npu="950" id749 -->
- <term>Ascend 950DT</term>：支持
<!-- end id749 -->

</div>

### torch.nn.utils.spectral_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.spectral_norm](https://pytorch.org/docs/2.14/generated/torch.nn.utils.spectral_norm.html)

**产品支持情况**：

<!-- npu="910b" id750 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id750 -->
<!-- npu="A3" id751 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id751 -->
<!-- npu="950" id752 -->
- <term>Ascend 950DT</term>：支持
<!-- end id752 -->

</div>

### torch.nn.utils.remove_spectral_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.remove_spectral_norm](https://pytorch.org/docs/2.14/generated/torch.nn.utils.remove_spectral_norm.html)

**产品支持情况**：

<!-- npu="910b" id753 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id753 -->
<!-- npu="A3" id754 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id754 -->
<!-- npu="950" id755 -->
- <term>Ascend 950DT</term>：支持
<!-- end id755 -->

</div>

### torch.nn.utils.skip_init

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.skip_init](https://pytorch.org/docs/2.14/generated/torch.nn.utils.skip_init.html)

**产品支持情况**：

<!-- npu="910b" id756 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id756 -->
<!-- npu="A3" id757 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id757 -->
<!-- npu="950" id758 -->
- <term>Ascend 950DT</term>：支持
<!-- end id758 -->

</div>

### torch.nn.parameter.is_lazy

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parameter.is_lazy](https://pytorch.org/docs/2.14/generated/torch.nn.parameter.is_lazy.html)

**产品支持情况**：

<!-- npu="910b" id759 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id759 -->
<!-- npu="A3" id760 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id760 -->
<!-- npu="950" id761 -->
- <term>Ascend 950DT</term>：支持
<!-- end id761 -->

</div>

### <code><i>class</i></code> torch.nn.utils.prune.BasePruningMethod

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.BasePruningMethod.html)

**产品支持情况**：

<!-- npu="910b" id762 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id762 -->
<!-- npu="A3" id763 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id763 -->
<!-- npu="950" id764 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id764 -->

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod.apply](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.apply)

**产品支持情况**：

<!-- npu="910b" id765 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id765 -->
<!-- npu="A3" id766 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id766 -->
<!-- npu="950" id767 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id767 -->

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod.apply_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.apply_mask)

**产品支持情况**：

<!-- npu="910b" id768 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id768 -->
<!-- npu="A3" id769 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id769 -->
<!-- npu="950" id770 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id770 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod.compute_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.compute_mask)

**产品支持情况**：

<!-- npu="910b" id771 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id771 -->
<!-- npu="A3" id772 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id772 -->
<!-- npu="950" id773 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id773 -->

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod.prune](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.prune)

**产品支持情况**：

<!-- npu="910b" id774 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id774 -->
<!-- npu="A3" id775 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id775 -->
<!-- npu="950" id776 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id776 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod.remove](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.remove)

**产品支持情况**：

<!-- npu="910b" id777 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id777 -->
<!-- npu="A3" id778 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id778 -->
<!-- npu="950" id779 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id779 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.PruningContainer

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.PruningContainer.html)

**产品支持情况**：

<!-- npu="910b" id780 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id780 -->
<!-- npu="A3" id781 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id781 -->
<!-- npu="950" id782 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id782 -->

> <font size="3">add_pruning_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.add_pruning_method](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.add_pruning_method)

**产品支持情况**：

<!-- npu="910b" id783 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id783 -->
<!-- npu="A3" id784 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id784 -->
<!-- npu="950" id785 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id785 -->

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.apply](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.apply)

**产品支持情况**：

<!-- npu="910b" id786 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id786 -->
<!-- npu="A3" id787 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id787 -->
<!-- npu="950" id788 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id788 -->

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.apply_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.apply_mask)

**产品支持情况**：

<!-- npu="910b" id789 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id789 -->
<!-- npu="A3" id790 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id790 -->
<!-- npu="950" id791 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id791 -->

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.compute_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.compute_mask)

**产品支持情况**：

<!-- npu="910b" id792 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id792 -->
<!-- npu="A3" id793 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id793 -->
<!-- npu="950" id794 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id794 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.prune](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.prune)

**产品支持情况**：

<!-- npu="910b" id795 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id795 -->
<!-- npu="A3" id796 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id796 -->
<!-- npu="950" id797 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id797 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.remove](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.remove)

**产品支持情况**：

<!-- npu="910b" id798 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id798 -->
<!-- npu="A3" id799 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id799 -->
<!-- npu="950" id800 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id800 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.Identity

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.Identity](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.Identity_class.html#torch.nn.utils.prune.Identity)

**产品支持情况**：

<!-- npu="910b" id801 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id801 -->
<!-- npu="A3" id802 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id802 -->
<!-- npu="950" id803 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id803 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.Identity.apply](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.Identity_class.html#torch.nn.utils.prune.Identity.apply)

**产品支持情况**：

<!-- npu="910b" id804 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id804 -->
<!-- npu="A3" id805 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id805 -->
<!-- npu="950" id806 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id806 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.Identity.apply_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.Identity_class.html#torch.nn.utils.prune.Identity.apply_mask)

**产品支持情况**：

<!-- npu="910b" id807 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id807 -->
<!-- npu="A3" id808 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id808 -->
<!-- npu="950" id809 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id809 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.Identity.prune](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.Identity_class.html#torch.nn.utils.prune.Identity.prune)

**产品支持情况**：

<!-- npu="910b" id810 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id810 -->
<!-- npu="A3" id811 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id811 -->
<!-- npu="950" id812 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id812 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.Identity.remove](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.Identity_class.html#torch.nn.utils.prune.Identity.remove)

**产品支持情况**：

<!-- npu="910b" id813 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id813 -->
<!-- npu="A3" id814 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id814 -->
<!-- npu="950" id815 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id815 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.RandomUnstructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomUnstructured](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomUnstructured.html)

**产品支持情况**：

<!-- npu="910b" id816 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id816 -->
<!-- npu="A3" id817 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id817 -->
<!-- npu="950" id818 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id818 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomUnstructured.apply](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.apply)

**产品支持情况**：

<!-- npu="910b" id819 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id819 -->
<!-- npu="A3" id820 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id820 -->
<!-- npu="950" id821 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id821 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomUnstructured.apply_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.apply_mask)

**产品支持情况**：

<!-- npu="910b" id822 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id822 -->
<!-- npu="A3" id823 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id823 -->
<!-- npu="950" id824 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id824 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomUnstructured.prune](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.prune)

**产品支持情况**：

<!-- npu="910b" id825 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id825 -->
<!-- npu="A3" id826 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id826 -->
<!-- npu="950" id827 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id827 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomUnstructured.remove](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.remove)

**产品支持情况**：

<!-- npu="910b" id828 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id828 -->
<!-- npu="A3" id829 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id829 -->
<!-- npu="950" id830 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id830 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.L1Unstructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.L1Unstructured](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.L1Unstructured.html)

**产品支持情况**：

<!-- npu="910b" id831 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id831 -->
<!-- npu="A3" id832 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id832 -->
<!-- npu="950" id833 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id833 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.L1Unstructured.apply](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.apply)

**产品支持情况**：

<!-- npu="910b" id834 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id834 -->
<!-- npu="A3" id835 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id835 -->
<!-- npu="950" id836 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id836 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.L1Unstructured.apply_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.apply_mask)

**产品支持情况**：

<!-- npu="910b" id837 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id837 -->
<!-- npu="A3" id838 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id838 -->
<!-- npu="950" id839 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id839 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.L1Unstructured.prune](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.prune)

**产品支持情况**：

<!-- npu="910b" id840 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id840 -->
<!-- npu="A3" id841 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id841 -->
<!-- npu="950" id842 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id842 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.L1Unstructured.remove](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.remove)

**产品支持情况**：

<!-- npu="910b" id843 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id843 -->
<!-- npu="A3" id844 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id844 -->
<!-- npu="950" id845 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id845 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.RandomStructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomStructured.html)

**产品支持情况**：

<!-- npu="910b" id846 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id846 -->
<!-- npu="A3" id847 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id847 -->
<!-- npu="950" id848 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id848 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured.apply](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.apply)

**产品支持情况**：

<!-- npu="910b" id849 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id849 -->
<!-- npu="A3" id850 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id850 -->
<!-- npu="950" id851 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id851 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured.apply_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.apply_mask)

**产品支持情况**：

<!-- npu="910b" id852 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id852 -->
<!-- npu="A3" id853 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id853 -->
<!-- npu="950" id854 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id854 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured.compute_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.compute_mask)

**产品支持情况**：

<!-- npu="910b" id855 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id855 -->
<!-- npu="A3" id856 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id856 -->
<!-- npu="950" id857 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id857 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured.prune](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.prune)

**产品支持情况**：

<!-- npu="910b" id858 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id858 -->
<!-- npu="A3" id859 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id859 -->
<!-- npu="950" id860 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id860 -->

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured.remove](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.remove)

**产品支持情况**：

<!-- npu="910b" id861 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id861 -->
<!-- npu="A3" id862 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id862 -->
<!-- npu="950" id863 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id863 -->

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.LnStructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.LnStructured.html)

**产品支持情况**：

<!-- npu="910b" id864 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id864 -->
<!-- npu="A3" id865 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id865 -->
<!-- npu="950" id866 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id866 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured.apply](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.apply)

**产品支持情况**：

<!-- npu="910b" id867 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id867 -->
<!-- npu="A3" id868 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id868 -->
<!-- npu="950" id869 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id869 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured.apply_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.apply_mask)

**产品支持情况**：

<!-- npu="910b" id870 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id870 -->
<!-- npu="A3" id871 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id871 -->
<!-- npu="950" id872 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id872 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured.compute_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.compute_mask)

**产品支持情况**：

<!-- npu="910b" id873 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id873 -->
<!-- npu="A3" id874 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id874 -->
<!-- npu="950" id875 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id875 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured.prune](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.prune)

**产品支持情况**：

<!-- npu="910b" id876 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id876 -->
<!-- npu="A3" id877 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id877 -->
<!-- npu="950" id878 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id878 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured.remove](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.remove)

**产品支持情况**：

<!-- npu="910b" id879 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id879 -->
<!-- npu="A3" id880 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id880 -->
<!-- npu="950" id881 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id881 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.nn.utils.prune.CustomFromMask

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.CustomFromMask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.CustomFromMask.html)

**产品支持情况**：

<!-- npu="910b" id882 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id882 -->
<!-- npu="A3" id883 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id883 -->
<!-- npu="950" id884 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id884 -->

**限制与说明**： `input`仅支持int64

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.CustomFromMask.apply](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.apply)

**产品支持情况**：

<!-- npu="910b" id885 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id885 -->
<!-- npu="A3" id886 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id886 -->
<!-- npu="950" id887 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id887 -->

**限制与说明**： `self`仅支持int64

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.CustomFromMask.apply_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.apply_mask)

**产品支持情况**：

<!-- npu="910b" id888 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id888 -->
<!-- npu="A3" id889 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id889 -->
<!-- npu="950" id890 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id890 -->

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.CustomFromMask.prune](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.prune)

**产品支持情况**：

<!-- npu="910b" id891 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id891 -->
<!-- npu="A3" id892 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id892 -->
<!-- npu="950" id893 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id893 -->

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.CustomFromMask.remove](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.remove)

**产品支持情况**：

<!-- npu="910b" id894 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id894 -->
<!-- npu="A3" id895 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id895 -->
<!-- npu="950" id896 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id896 -->

</div>

</div>

### torch.nn.utils.prune.random_unstructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.random_unstructured](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.random_unstructured.html)

**产品支持情况**：

<!-- npu="910b" id897 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id897 -->
<!-- npu="A3" id898 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id898 -->
<!-- npu="950" id899 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id899 -->

</div>

### torch.nn.utils.prune.l1_unstructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.l1_unstructured](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.l1_unstructured.html)

**产品支持情况**：

<!-- npu="910b" id900 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id900 -->
<!-- npu="A3" id901 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id901 -->
<!-- npu="950" id902 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id902 -->

</div>

### torch.nn.utils.prune.random_structured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.random_structured](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.random_structured.html)

**产品支持情况**：

<!-- npu="910b" id903 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id903 -->
<!-- npu="A3" id904 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id904 -->
<!-- npu="950" id905 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id905 -->

</div>

### torch.nn.utils.prune.ln_structured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.ln_structured](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.ln_structured.html)

**产品支持情况**：

<!-- npu="910b" id906 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id906 -->
<!-- npu="A3" id907 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id907 -->
<!-- npu="950" id908 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id908 -->

</div>

### torch.nn.utils.prune.global_unstructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.global_unstructured](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.global_unstructured.html)

**产品支持情况**：

<!-- npu="910b" id909 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id909 -->
<!-- npu="A3" id910 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id910 -->
<!-- npu="950" id911 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id911 -->

</div>

### torch.nn.utils.prune.identity

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.identity](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.identity_function.html)

**产品支持情况**：

<!-- npu="910b" id912 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id912 -->
<!-- npu="A3" id913 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id913 -->
<!-- npu="950" id914 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id914 -->

</div>

### torch.nn.utils.prune.custom_from_mask

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.custom_from_mask](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.custom_from_mask.html)

**产品支持情况**：

<!-- npu="910b" id915 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id915 -->
<!-- npu="A3" id916 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id916 -->
<!-- npu="950" id917 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id917 -->

**限制与说明**： `input`仅支持int64

</div>

### torch.nn.utils.prune.remove

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.remove](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.remove.html)

**产品支持情况**：

<!-- npu="910b" id918 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id918 -->
<!-- npu="A3" id919 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id919 -->
<!-- npu="950" id920 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id920 -->

</div>

### torch.nn.utils.prune.is_pruned

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.is_pruned](https://pytorch.org/docs/2.14/generated/torch.nn.utils.prune.is_pruned.html)

**产品支持情况**：

<!-- npu="910b" id921 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id921 -->
<!-- npu="A3" id922 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id922 -->
<!-- npu="950" id923 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id923 -->

</div>

### torch.nn.utils.parametrizations.orthogonal

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrizations.orthogonal](https://pytorch.org/docs/2.14/generated/torch.nn.utils.parametrizations.orthogonal.html)

**产品支持情况**：

<!-- npu="910b" id924 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id924 -->
<!-- npu="A3" id925 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id925 -->
<!-- npu="950" id926 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id926 -->

</div>

### torch.nn.utils.parametrizations.spectral_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrizations.spectral_norm](https://pytorch.org/docs/2.14/generated/torch.nn.utils.parametrizations.spectral_norm.html)

**产品支持情况**：

<!-- npu="910b" id927 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id927 -->
<!-- npu="A3" id928 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id928 -->
<!-- npu="950" id929 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id929 -->

</div>

### torch.nn.utils.parametrize.register_parametrization

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.register_parametrization](https://pytorch.org/docs/2.14/generated/torch.nn.utils.parametrize.register_parametrization.html)

**产品支持情况**：

<!-- npu="910b" id930 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id930 -->
<!-- npu="A3" id931 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id931 -->
<!-- npu="950" id932 -->
- <term>Ascend 950DT</term>：支持
<!-- end id932 -->

</div>

### torch.nn.utils.parametrize.remove_parametrizations

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.remove_parametrizations](https://pytorch.org/docs/2.14/generated/torch.nn.utils.parametrize.remove_parametrizations.html)

**产品支持情况**：

<!-- npu="910b" id933 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id933 -->
<!-- npu="A3" id934 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id934 -->
<!-- npu="950" id935 -->
- <term>Ascend 950DT</term>：支持
<!-- end id935 -->

</div>

### torch.nn.utils.parametrize.cached

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.cached](https://pytorch.org/docs/2.14/generated/torch.nn.utils.parametrize.cached.html)

**产品支持情况**：

<!-- npu="910b" id936 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id936 -->
<!-- npu="A3" id937 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id937 -->
<!-- npu="950" id938 -->
- <term>Ascend 950DT</term>：支持
<!-- end id938 -->

</div>

### torch.nn.utils.parametrize.is_parametrized

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.is_parametrized](https://pytorch.org/docs/2.14/generated/torch.nn.utils.parametrize.is_parametrized.html)

**产品支持情况**：

<!-- npu="910b" id939 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id939 -->
<!-- npu="A3" id940 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id940 -->
<!-- npu="950" id941 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id941 -->

</div>

### <code><i>class</i></code> torch.nn.utils.parametrize.ParametrizationList

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.ParametrizationList](https://pytorch.org/docs/2.14/generated/torch.nn.utils.parametrize.ParametrizationList.html)

**产品支持情况**：

<!-- npu="910b" id942 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id942 -->
<!-- npu="A3" id943 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id943 -->
<!-- npu="950" id944 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id944 -->

> <font size="3">right_inverse()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.ParametrizationList.right_inverse](https://pytorch.org/docs/2.14/generated/torch.nn.utils.parametrize.ParametrizationList.html#torch.nn.utils.parametrize.ParametrizationList.right_inverse)

**产品支持情况**：

<!-- npu="910b" id945 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id945 -->
<!-- npu="A3" id946 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id946 -->
<!-- npu="950" id947 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id947 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>

### torch.nn.utils.stateless.functional_call

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.stateless.functional_call](https://pytorch.org/docs/2.14/generated/torch.nn.utils.stateless.functional_call.html)

**版本说明**：该接口已废弃，请使用`torch.func.functional_call`。

**产品支持情况**：

<!-- npu="910b" id948 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id948 -->
<!-- npu="A3" id949 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id949 -->
<!-- npu="950" id950 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id950 -->

</div>

### <code><i>class</i></code> torch.nn.utils.rnn.PackedSequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.PackedSequence.html)

**产品支持情况**：

<!-- npu="910b" id951 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id951 -->
<!-- npu="A3" id952 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id952 -->
<!-- npu="950" id953 -->
- <term>Ascend 950DT</term>：支持
<!-- end id953 -->

**限制与说明**： `input`仅支持fp32，int64

> <font size="3">batch_sizes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.batch_sizes](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.batch_sizes)

**产品支持情况**：

<!-- npu="910b" id954 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id954 -->
<!-- npu="A3" id955 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id955 -->
<!-- npu="950" id956 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id956 -->

</div>

> <font size="3">count()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.count](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.count)

**产品支持情况**：

<!-- npu="910b" id957 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id957 -->
<!-- npu="A3" id958 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id958 -->
<!-- npu="950" id959 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id959 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">data()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.data](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.data)

**产品支持情况**：

<!-- npu="910b" id960 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id960 -->
<!-- npu="A3" id961 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id961 -->
<!-- npu="950" id962 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id962 -->

</div>

> <font size="3">index()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.index](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.index)

**产品支持情况**：

<!-- npu="910b" id963 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id963 -->
<!-- npu="A3" id964 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id964 -->
<!-- npu="950" id965 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id965 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.is_cuda](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.is_cuda)

**产品支持情况**：

<!-- npu="910b" id966 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id966 -->
<!-- npu="A3" id967 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id967 -->
<!-- npu="950" id968 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id968 -->

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.is_pinned](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.is_pinned)

**产品支持情况**：

<!-- npu="910b" id969 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id969 -->
<!-- npu="A3" id970 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id970 -->
<!-- npu="950" id971 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id971 -->

</div>

> <font size="3">sorted_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.sorted_indices](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.sorted_indices)

**产品支持情况**：

<!-- npu="910b" id972 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id972 -->
<!-- npu="A3" id973 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id973 -->
<!-- npu="950" id974 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id974 -->

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.to](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.to)

**产品支持情况**：

<!-- npu="910b" id975 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id975 -->
<!-- npu="A3" id976 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id976 -->
<!-- npu="950" id977 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id977 -->

**限制与说明**： `self`仅支持fp32，int64

</div>

> <font size="3">unsorted_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.unsorted_indices](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.unsorted_indices)

**产品支持情况**：

<!-- npu="910b" id978 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id978 -->
<!-- npu="A3" id979 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id979 -->
<!-- npu="950" id980 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id980 -->

</div>

</div>

### torch.nn.utils.rnn.pack_padded_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pack_padded_sequence.html)

**产品支持情况**：

<!-- npu="910b" id981 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id981 -->
<!-- npu="A3" id982 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id982 -->
<!-- npu="950" id983 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id983 -->

</div>

### torch.nn.utils.rnn.pad_packed_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.pad_packed_sequence](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pad_packed_sequence.html)

**产品支持情况**：

<!-- npu="910b" id984 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id984 -->
<!-- npu="A3" id985 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id985 -->
<!-- npu="950" id986 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id986 -->

</div>

### torch.nn.utils.rnn.pad_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pad_sequence.html)

**产品支持情况**：

<!-- npu="910b" id987 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id987 -->
<!-- npu="A3" id988 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id988 -->
<!-- npu="950" id989 -->
- <term>Ascend 950DT</term>：支持
<!-- end id989 -->

**限制与说明**： `sequences`仅支持fp16，fp32

</div>

### torch.nn.utils.rnn.pack_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.pack_sequence](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pack_sequence.html)

**产品支持情况**：

<!-- npu="910b" id990 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id990 -->
<!-- npu="A3" id991 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id991 -->
<!-- npu="950" id992 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id992 -->

</div>

### torch.nn.utils.rnn.unpack_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.unpack_sequence](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.unpack_sequence.html)

**产品支持情况**：

<!-- npu="910b" id993 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id993 -->
<!-- npu="A3" id994 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id994 -->
<!-- npu="950" id995 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id995 -->

</div>

### torch.nn.utils.rnn.unpad_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.unpad_sequence](https://pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.unpad_sequence.html)

**产品支持情况**：

<!-- npu="910b" id996 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id996 -->
<!-- npu="A3" id997 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id997 -->
<!-- npu="950" id998 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id998 -->

</div>

### <code><i>class</i></code> torch.nn.modules.flatten.Flatten

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.flatten.Flatten](https://pytorch.org/docs/2.14/generated/torch.nn.modules.flatten.Flatten.html)

**产品支持情况**：

<!-- npu="910b" id999 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id999 -->
<!-- npu="A3" id1000 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id1000 -->
<!-- npu="950" id1001 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id1001 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### <code><i>class</i></code> torch.nn.modules.flatten.Unflatten

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.flatten.Unflatten](https://pytorch.org/docs/2.14/generated/torch.nn.modules.flatten.Unflatten.html)

**产品支持情况**：

<!-- npu="910b" id1002 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id1002 -->
<!-- npu="A3" id1003 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id1003 -->
<!-- npu="950" id1004 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id1004 -->

**限制与说明**： `input`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

## Lazy Modules Initialization

### <code><i>class</i></code> torch.nn.modules.lazy.LazyModuleMixin

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.lazy.LazyModuleMixin](https://pytorch.org/docs/2.14/generated/torch.nn.modules.lazy.LazyModuleMixin.html)

**产品支持情况**：

<!-- npu="910b" id1005 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id1005 -->
<!-- npu="A3" id1006 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id1006 -->
<!-- npu="950" id1007 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1007 -->

**限制与说明**： `input`仅支持fp32

> <font size="3">has_uninitialized_params()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params](https://pytorch.org/docs/2.14/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params)

**产品支持情况**：

<!-- npu="910b" id1008 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id1008 -->
<!-- npu="A3" id1009 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id1009 -->
<!-- npu="950" id1010 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id1010 -->

**限制与说明**： `self`仅支持fp32

</div>

> <font size="3">initialize_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters](https://pytorch.org/docs/2.14/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters)

**产品支持情况**：

<!-- npu="910b" id1011 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id1011 -->
<!-- npu="A3" id1012 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id1012 -->
<!-- npu="950" id1013 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id1013 -->

**限制与说明**： `self`仅支持fp32

</div>

</div>
