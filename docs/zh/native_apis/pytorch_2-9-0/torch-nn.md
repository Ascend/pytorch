# torch.nn

> [!NOTE]   
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
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

## base API

### _`class`_ torch.nn.parameter.Parameter

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parameter.Parameter](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.Parameter.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.nn.parameter.Buffer

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parameter.Buffer](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.Buffer.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.nn.parameter.UninitializedParameter

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parameter.UninitializedParameter](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.UninitializedParameter.html)

**是否支持**：是

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parameter.UninitializedParameter.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.UninitializedParameter.html#torch.nn.parameter.UninitializedParameter.cls_to_become)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.nn.parameter.UninitializedBuffer

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parameter.UninitializedBuffer](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.UninitializedBuffer.html)

**是否支持**：是

</div>

## Containers

### _`class`_ torch.nn.Module

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html)

**是否支持**：是

**限制与说明**： 支持fp32

> <font size="3">add_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.add_module](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.add_module)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.apply](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.apply)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.bfloat16](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.bfloat16)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.buffers](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.buffers)

**是否支持**：是

</div>

> <font size="3">children()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.children](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.children)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">compile()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.compile](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.compile)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.cpu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.cpu)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.cuda](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.cuda)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.double](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.double)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">eval()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.eval](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.eval)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32，int64

</div>

> <font size="3">extra_repr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.extra_repr](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.extra_repr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.float](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.float)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.forward](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.forward)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">get_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.get_buffer](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_buffer)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">get_extra_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.get_extra_state](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_extra_state)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">get_parameter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.get_parameter](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_parameter)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">get_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.get_submodule](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_submodule)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.half](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.half)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">ipu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.ipu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.ipu)

**是否支持**：否

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.modules](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.modules)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">named_buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.named_buffers](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_buffers)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">named_children()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.named_children](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_children)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">named_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.named_modules](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_modules)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">named_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.named_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_parameters)

**是否支持**：是

</div>

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.parameters](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.parameters)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_backward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_backward_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_buffer](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_forward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_forward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">register_forward_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_forward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">register_full_backward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_full_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_full_backward_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_full_backward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_load_state_dict_post_hook)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">register_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_module](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_module)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_parameter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_parameter](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_parameter)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">requires_grad_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.requires_grad_](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.requires_grad_)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_extra_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.set_extra_state](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.set_extra_state)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.set_submodule](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.set_submodule)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">share_memory()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.share_memory](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.share_memory)

**是否支持**：否

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.state_dict](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.state_dict)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.to](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.to)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">to_empty()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.to_empty](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.to_empty)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">train()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.train](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.train)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.type](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.type)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，int64

</div>

> <font size="3">xpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.xpu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.xpu)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： NPU形式名称为torch.nn.Module.npu

</div>

> <font size="3">npu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.npu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html)

**是否支持**：否

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Module.zero_grad](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.Sequential

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Sequential](https://pytorch.org/docs/2.9/generated/torch.nn.Sequential.html)

**是否支持**：是

**限制与说明**： 支持fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Sequential.append](https://pytorch.org/docs/2.9/generated/torch.nn.Sequential.html#torch.nn.Sequential.append)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.ModuleList

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleList](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html)

**是否支持**：是

**限制与说明**： 支持fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleList.append](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.append)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">extend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleList.extend](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.extend)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">insert()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleList.insert](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.insert)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.ModuleDict

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html)

**是否支持**：是

**限制与说明**： 支持fp32

> <font size="3">clear()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.clear](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.clear)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">items()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.items](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.items)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.keys](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.keys)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">pop()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.pop](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.pop)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">update()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.update](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.update)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ModuleDict.values](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.values)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.ParameterList

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterList](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterList.html)

**是否支持**：是

**限制与说明**： 支持fp32

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterList.append](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterList.html#torch.nn.ParameterList.append)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">extend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterList.extend](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterList.html#torch.nn.ParameterList.extend)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.ParameterDict

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html)

**是否支持**：是

**限制与说明**： 支持fp32

> <font size="3">clear()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.clear](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.clear)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">copy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.copy](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.copy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">fromkeys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.fromkeys](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.fromkeys)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.get](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.get)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">items()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.items](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.items)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.keys](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.keys)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">pop()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.pop](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.pop)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">popitem()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.popitem](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.popitem)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">setdefault()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.setdefault](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.setdefault)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">update()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.update](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.update)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ParameterDict.values](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.values)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### torch.nn.modules.module.register_module_forward_pre_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_forward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_forward_pre_hook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.modules.module.register_module_forward_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_forward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_forward_hook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.modules.module.register_module_backward_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_backward_hook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.modules.module.register_module_full_backward_pre_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_full_backward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_full_backward_pre_hook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.modules.module.register_module_full_backward_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_full_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_full_backward_hook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.modules.module.register_module_buffer_registration_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_buffer_registration_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_buffer_registration_hook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.modules.module.register_module_module_registration_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_module_registration_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_module_registration_hook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.modules.module.register_module_parameter_registration_hook

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.module.register_module_parameter_registration_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_parameter_registration_hook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Convolution Layers

### _`class`_ torch.nn.Conv1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Conv1d](https://pytorch.org/docs/2.9/generated/torch.nn.Conv1d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.Conv2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Conv2d](https://pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- <term>Atlas A2 训练系列产品</term>，默认场景下，如果频繁触发编译，建议手动设置torch.npu.config.allow_internal_format为False，控制入参不开启内部格式，避免在线编译

</div>

### _`class`_ torch.nn.Conv3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Conv3d](https://pytorch.org/docs/2.9/generated/torch.nn.Conv3d.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.ConvTranspose1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConvTranspose1d](https://pytorch.org/docs/2.9/generated/torch.nn.ConvTranspose1d.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.nn.ConvTranspose2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConvTranspose2d](https://pytorch.org/docs/2.9/generated/torch.nn.ConvTranspose2d.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- <term>Atlas 训练系列产品</term>/<term>Atlas A2 训练系列产品</term>，需手动设置torch.npu.config.allow_internal_format为False，才可支持3维输入

</div>

### _`class`_ torch.nn.ConvTranspose3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConvTranspose3d](https://pytorch.org/docs/2.9/generated/torch.nn.ConvTranspose3d.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.LazyConv1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConv1d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv1d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConv1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv1d.html#torch.nn.LazyConv1d.cls_to_become)

**是否支持**：是

</div>

</div>

### _`class`_ torch.nn.LazyConv2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConv2d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv2d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConv2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d.cls_to_become)

**是否支持**：是

</div>

</div>

### _`class`_ torch.nn.LazyConv3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConv3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv3d.html#torch.nn.LazyConv3d.cls_to_become)

**是否支持**：是

</div>

</div>

### _`class`_ torch.nn.LazyConvTranspose1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConvTranspose1d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose1d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConvTranspose1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose1d.html#torch.nn.LazyConvTranspose1d.cls_to_become)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.nn.LazyConvTranspose2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConvTranspose2d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose2d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConvTranspose2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose2d.html#torch.nn.LazyConvTranspose2d.cls_to_become)

**是否支持**：是

</div>

</div>

### _`class`_ torch.nn.LazyConvTranspose3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyConvTranspose3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose3d.html#torch.nn.LazyConvTranspose3d.cls_to_become)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.nn.Unfold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Unfold](https://pytorch.org/docs/2.9/generated/torch.nn.Unfold.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.Fold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Fold](https://pytorch.org/docs/2.9/generated/torch.nn.Fold.html)

**是否支持**：是

**限制与说明**： 支持fp16

</div>

## Pooling layers

### _`class`_ torch.nn.MaxPool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxPool1d.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.MaxPool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxPool2d.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 通过设置torch_npu.npu.use_compatible_impl(True)，保证与社区同名接口在内存一致性上对齐

</div>

### _`class`_ torch.nn.MaxPool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxPool3d.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.MaxUnpool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxUnpool1d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxUnpool1d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.MaxUnpool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxUnpool2d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxUnpool2d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.MaxUnpool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MaxUnpool3d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxUnpool3d.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.AvgPool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AvgPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.AvgPool1d.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.AvgPool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AvgPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.AvgPool2d.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.AvgPool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AvgPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.AvgPool3d.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.LPPool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LPPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.LPPool1d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### _`class`_ torch.nn.LPPool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LPPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.LPPool2d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，int16，int32，int64，bool

</div>

### _`class`_ torch.nn.AdaptiveMaxPool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveMaxPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveMaxPool1d.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.AdaptiveMaxPool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveMaxPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveMaxPool2d.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.AdaptiveMaxPool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveMaxPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveMaxPool3d.html)

**是否支持**：是

**限制与说明**： 支持fp32，fp64

</div>

### _`class`_ torch.nn.AdaptiveAvgPool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveAvgPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool1d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.AdaptiveAvgPool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveAvgPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool2d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.AdaptiveAvgPool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveAvgPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool3d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

## Padding Layers

### _`class`_ torch.nn.ReflectionPad1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReflectionPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ReflectionPad1d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.ReflectionPad2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReflectionPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ReflectionPad2d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.ReflectionPad3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReflectionPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ReflectionPad3d.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.ReplicationPad1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReplicationPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ReplicationPad1d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

### _`class`_ torch.nn.ReplicationPad2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReplicationPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ReplicationPad2d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

### _`class`_ torch.nn.ReplicationPad3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReplicationPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ReplicationPad3d.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.ZeroPad1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ZeroPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ZeroPad1d.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，complex64，complex128
- 支持2-3维

</div>

### _`class`_ torch.nn.ZeroPad2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ZeroPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ZeroPad2d.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

### _`class`_ torch.nn.ZeroPad3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ZeroPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ZeroPad3d.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，complex64，complex128
- 支持5-6维

</div>

### _`class`_ torch.nn.ConstantPad1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConstantPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ConstantPad1d.html)

**是否支持**：是

**限制与说明**：

- 支持int8，bool
- 在输入x为六维以上时可能会出现性能下降问题

</div>

### _`class`_ torch.nn.ConstantPad2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConstantPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ConstantPad2d.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 在输入x为六维以上时可能会出现性能下降问题

</div>

### _`class`_ torch.nn.ConstantPad3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ConstantPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ConstantPad3d.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 在输入x为六维以上时可能会出现性能下降问题

</div>

## Non-linear Activations (weighted sum, nonlinearity)

### _`class`_ torch.nn.ELU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ELU](https://pytorch.org/docs/2.9/generated/torch.nn.ELU.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

### _`class`_ torch.nn.Hardshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Hardshrink](https://pytorch.org/docs/2.9/generated/torch.nn.Hardshrink.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- 可能回退至CPU执行

</div>

### _`class`_ torch.nn.Hardsigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Hardsigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.Hardsigmoid.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp16，fp32，int32
- 可能回退至CPU执行

</div>

### _`class`_ torch.nn.Hardtanh

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Hardtanh](https://pytorch.org/docs/2.9/generated/torch.nn.Hardtanh.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### _`class`_ torch.nn.Hardswish

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Hardswish](https://pytorch.org/docs/2.9/generated/torch.nn.Hardswish.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.LeakyReLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LeakyReLU](https://pytorch.org/docs/2.9/generated/torch.nn.LeakyReLU.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

### _`class`_ torch.nn.LogSigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LogSigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.LogSigmoid.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.MultiheadAttention

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MultiheadAttention](https://pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MultiheadAttention.forward](https://pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

</div>

### _`class`_ torch.nn.PReLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.PReLU](https://pytorch.org/docs/2.9/generated/torch.nn.PReLU.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.nn.ReLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReLU](https://pytorch.org/docs/2.9/generated/torch.nn.ReLU.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### _`class`_ torch.nn.ReLU6

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ReLU6](https://pytorch.org/docs/2.9/generated/torch.nn.ReLU6.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### _`class`_ torch.nn.RReLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.RReLU](https://pytorch.org/docs/2.9/generated/torch.nn.RReLU.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.SELU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.SELU](https://pytorch.org/docs/2.9/generated/torch.nn.SELU.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### _`class`_ torch.nn.CELU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.CELU](https://pytorch.org/docs/2.9/generated/torch.nn.CELU.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.GELU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GELU](https://pytorch.org/docs/2.9/generated/torch.nn.GELU.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- approximate参数仅支持设置为tanh

</div>

### _`class`_ torch.nn.Sigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Sigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.Sigmoid.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### _`class`_ torch.nn.SiLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.SiLU](https://pytorch.org/docs/2.9/generated/torch.nn.SiLU.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.Mish

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Mish](https://pytorch.org/docs/2.9/generated/torch.nn.Mish.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.Softplus

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softplus](https://pytorch.org/docs/2.9/generated/torch.nn.Softplus.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.Softshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softshrink](https://pytorch.org/docs/2.9/generated/torch.nn.Softshrink.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.Softsign

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softsign](https://pytorch.org/docs/2.9/generated/torch.nn.Softsign.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### _`class`_ torch.nn.Tanh

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Tanh](https://pytorch.org/docs/2.9/generated/torch.nn.Tanh.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### _`class`_ torch.nn.Tanhshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Tanhshrink](https://pytorch.org/docs/2.9/generated/torch.nn.Tanhshrink.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64
- 可能回退至CPU执行

</div>

### _`class`_ torch.nn.Threshold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Threshold](https://pytorch.org/docs/2.9/generated/torch.nn.Threshold.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

### _`class`_ torch.nn.GLU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GLU](https://pytorch.org/docs/2.9/generated/torch.nn.GLU.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

## Non-linear Activations (other)

### _`class`_ torch.nn.Softmin

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softmin](https://pytorch.org/docs/2.9/generated/torch.nn.Softmin.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.Softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softmax](https://pytorch.org/docs/2.9/generated/torch.nn.Softmax.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

### _`class`_ torch.nn.Softmax2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Softmax2d](https://pytorch.org/docs/2.9/generated/torch.nn.Softmax2d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.LogSoftmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LogSoftmax](https://pytorch.org/docs/2.9/generated/torch.nn.LogSoftmax.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.AdaptiveLogSoftmaxWithLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveLogSoftmaxWithLoss](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html)

**是否支持**：否

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob)

**是否支持**：否

</div>

> <font size="3">predict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AdaptiveLogSoftmaxWithLoss.predict](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss.predict)

**是否支持**：否

</div>

</div>

## Normalization Layers

### _`class`_ torch.nn.BatchNorm1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.BatchNorm1d](https://pytorch.org/docs/2.9/generated/torch.nn.BatchNorm1d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.BatchNorm2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.BatchNorm2d](https://pytorch.org/docs/2.9/generated/torch.nn.BatchNorm2d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.BatchNorm3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.BatchNorm3d](https://pytorch.org/docs/2.9/generated/torch.nn.BatchNorm3d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.LazyBatchNorm1d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyBatchNorm1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyBatchNorm1d.html#torch.nn.LazyBatchNorm1d.cls_to_become)

**是否支持**：是

</div>

</div>

### _`class`_ torch.nn.LazyBatchNorm2d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyBatchNorm2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyBatchNorm2d.html#torch.nn.LazyBatchNorm2d.cls_to_become)

**是否支持**：是

</div>

</div>

### _`class`_ torch.nn.LazyBatchNorm3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyBatchNorm3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyBatchNorm3d.html#torch.nn.LazyBatchNorm3d.cls_to_become)

**是否支持**：是

</div>

</div>

### _`class`_ torch.nn.GroupNorm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GroupNorm](https://pytorch.org/docs/2.9/generated/torch.nn.GroupNorm.html)

**是否支持**：是

**限制与说明**：

- 支持fp32
- eps参数需大于0
- 不支持jit_compile=True的场景
- 该API仅支持2维及以上的输入input。该API反向传播时，不支持输入维度不为4维，或输入num_groups非32整除，或C轴维度非(10 * num_groups)整除的场景

</div>

### _`class`_ torch.nn.SyncBatchNorm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.SyncBatchNorm](https://pytorch.org/docs/2.9/generated/torch.nn.SyncBatchNorm.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

> <font size="3">convert_sync_batchnorm()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.SyncBatchNorm.convert_sync_batchnorm](https://pytorch.org/docs/2.9/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm)

**是否支持**：是

</div>

</div>

### _`class`_ torch.nn.LazyInstanceNorm1d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyInstanceNorm1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyInstanceNorm1d.html#torch.nn.LazyInstanceNorm1d.cls_to_become)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.nn.LazyInstanceNorm2d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyInstanceNorm2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyInstanceNorm2d.html#torch.nn.LazyInstanceNorm2d.cls_to_become)

**是否支持**：是

</div>

</div>

### _`class`_ torch.nn.LazyInstanceNorm3d

<div style="margin-left: 2em">

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyInstanceNorm3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyInstanceNorm3d.html#torch.nn.LazyInstanceNorm3d.cls_to_become)

**是否支持**：是

</div>

</div>

### _`class`_ torch.nn.LayerNorm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LayerNorm](https://pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 通过torch_npu.npu.use_compatible_impl(True)，设置该接口从aclnnLayerNorm算子切换为aclnnFastLayerNorm算子，保证与社区同名接口在内存一致性上对齐。

</div>

## Recurrent Layers

### _`class`_ torch.nn.RNNBase

<div style="margin-left: 2em">

**原生文档**：[torch.nn.RNNBase](https://pytorch.org/docs/2.9/generated/torch.nn.RNNBase.html)

**是否支持**：否

> <font size="3">flatten_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.RNNBase.flatten_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.RNNBase.html#torch.nn.RNNBase.flatten_parameters)

**是否支持**：否

</div>

</div>

### _`class`_ torch.nn.RNN

<div style="margin-left: 2em">

**原生文档**：[torch.nn.RNN](https://pytorch.org/docs/2.9/generated/torch.nn.RNN.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.LSTM

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LSTM](https://pytorch.org/docs/2.9/generated/torch.nn.LSTM.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp32
- 不支持proj_size参数
- 不支持dropout参数
- 入参input不支持2维

</div>

### _`class`_ torch.nn.GRU

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GRU](https://pytorch.org/docs/2.9/generated/torch.nn.GRU.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.RNNCell

<div style="margin-left: 2em">

**原生文档**：[torch.nn.RNNCell](https://pytorch.org/docs/2.9/generated/torch.nn.RNNCell.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.LSTMCell

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LSTMCell](https://pytorch.org/docs/2.9/generated/torch.nn.LSTMCell.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 接口暂不支持jit_compile=False，需要在该模式下使用时请将"DynamicGRUV2"添加至"NPU_FUZZY_COMPILE_BLACKLIST"选项内，具体操作可参考[添加二进制黑名单示例](../example_of_adding_a_binary_blocklist.md)

</div>

### _`class`_ torch.nn.GRUCell

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GRUCell](https://pytorch.org/docs/2.9/generated/torch.nn.GRUCell.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

## Transformer Layers

### _`class`_ torch.nn.Transformer

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Transformer](https://pytorch.org/docs/2.9/generated/torch.nn.Transformer.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Transformer.forward](https://pytorch.org/docs/2.9/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward)

**是否支持**：否

</div>

</div>

### _`class`_ torch.nn.TransformerEncoder

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerEncoder](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoder.html)

**是否支持**：否

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerEncoder.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder.forward)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.TransformerDecoder

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerDecoder](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerDecoder.html)

**是否支持**：否

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerDecoder.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder.forward)

**是否支持**：否

</div>

</div>

### _`class`_ torch.nn.TransformerEncoderLayer

<div style="margin-left: 2em">

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerEncoderLayer.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer.forward)

**是否支持**：否

</div>

</div>

### _`class`_ torch.nn.TransformerDecoderLayer

<div style="margin-left: 2em">

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TransformerDecoderLayer.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer.forward)

**是否支持**：否

</div>

</div>

## Linear Layers

### _`class`_ torch.nn.Identity

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Identity](https://pytorch.org/docs/2.9/generated/torch.nn.Identity.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.nn.Linear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Linear](https://pytorch.org/docs/2.9/generated/torch.nn.Linear.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.Bilinear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Bilinear](https://pytorch.org/docs/2.9/generated/torch.nn.Bilinear.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.LazyLinear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyLinear](https://pytorch.org/docs/2.9/generated/torch.nn.LazyLinear.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

> <font size="3">cls_to_become()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.LazyLinear.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear.cls_to_become)

**是否支持**：否

</div>

</div>

## Dropout Layers

### _`class`_ torch.nn.Dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Dropout](https://pytorch.org/docs/2.9/generated/torch.nn.Dropout.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.Dropout2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Dropout2d](https://pytorch.org/docs/2.9/generated/torch.nn.Dropout2d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int64，bool

</div>

### _`class`_ torch.nn.AlphaDropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.AlphaDropout](https://pytorch.org/docs/2.9/generated/torch.nn.AlphaDropout.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.FeatureAlphaDropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.FeatureAlphaDropout](https://pytorch.org/docs/2.9/generated/torch.nn.FeatureAlphaDropout.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

## Sparse Layers

### _`class`_ torch.nn.Embedding

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Embedding](https://pytorch.org/docs/2.9/generated/torch.nn.Embedding.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持int32，int64
- 属性max_norm不支持nan，仅支持非负值

> <font size="3">from_pretrained()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Embedding.from_pretrained](https://pytorch.org/docs/2.9/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp64

</div>

</div>

### _`class`_ torch.nn.EmbeddingBag

<div style="margin-left: 2em">

**原生文档**：[torch.nn.EmbeddingBag](https://pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html)

**是否支持**：是

**限制与说明**：

- 支持int32，int64
- 仅支持max_norm大于等于0

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.EmbeddingBag.forward](https://pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持int64

</div>

> <font size="3">from_pretrained()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.EmbeddingBag.from_pretrained](https://pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.from_pretrained)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持int64

</div>

</div>

## Loss Functions

### _`class`_ torch.nn.L1Loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.L1Loss](https://pytorch.org/docs/2.9/generated/torch.nn.L1Loss.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int64

</div>

### _`class`_ torch.nn.MSELoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MSELoss](https://pytorch.org/docs/2.9/generated/torch.nn.MSELoss.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.CrossEntropyLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.CrossEntropyLoss](https://pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.CTCLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.CTCLoss](https://pytorch.org/docs/2.9/generated/torch.nn.CTCLoss.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp32，fp64
- 不支持log_probs 2D输入

</div>

### _`class`_ torch.nn.NLLLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.NLLLoss](https://pytorch.org/docs/2.9/generated/torch.nn.NLLLoss.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- target中的每个元素值应大于等于0且小于input的类别数

</div>

### _`class`_ torch.nn.PoissonNLLLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.PoissonNLLLoss](https://pytorch.org/docs/2.9/generated/torch.nn.PoissonNLLLoss.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int64

</div>

### _`class`_ torch.nn.GaussianNLLLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.GaussianNLLLoss](https://pytorch.org/docs/2.9/generated/torch.nn.GaussianNLLLoss.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int16，int32，int64

</div>

### _`class`_ torch.nn.KLDivLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.KLDivLoss](https://pytorch.org/docs/2.9/generated/torch.nn.KLDivLoss.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 当前log_target参数仅支持False

</div>

### _`class`_ torch.nn.BCELoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.BCELoss](https://pytorch.org/docs/2.9/generated/torch.nn.BCELoss.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.BCEWithLogitsLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持bf16，fp16，fp32
- 入参target不支持反向计算

</div>

### _`class`_ torch.nn.MarginRankingLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MarginRankingLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MarginRankingLoss.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int8，int32，int64

</div>

### _`class`_ torch.nn.HingeEmbeddingLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.HingeEmbeddingLoss](https://pytorch.org/docs/2.9/generated/torch.nn.HingeEmbeddingLoss.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### _`class`_ torch.nn.MultiLabelMarginLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MultiLabelMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MultiLabelMarginLoss.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.HuberLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.HuberLoss](https://pytorch.org/docs/2.9/generated/torch.nn.HuberLoss.html)

**是否支持**：是

**限制与说明**：

- input支持fp32，fp64
- target支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 可能回退至CPU执行

</div>

### _`class`_ torch.nn.SmoothL1Loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.SmoothL1Loss](https://pytorch.org/docs/2.9/generated/torch.nn.SmoothL1Loss.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.nn.MultiLabelSoftMarginLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MultiLabelSoftMarginLoss.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### _`class`_ torch.nn.CosineEmbeddingLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/2.9/generated/torch.nn.CosineEmbeddingLoss.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.MultiMarginLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.MultiMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MultiMarginLoss.html)

**是否支持**：是

**限制与说明**：

- input支持fp32，fp64
- target支持int64
- 可能回退至CPU执行

</div>

### _`class`_ torch.nn.TripletMarginLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TripletMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.TripletMarginLoss.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64
- 可能回退至CPU执行

</div>

### _`class`_ torch.nn.TripletMarginWithDistanceLoss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/2.9/generated/torch.nn.TripletMarginWithDistanceLoss.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

## Vision Layers

### _`class`_ torch.nn.PixelShuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.PixelShuffle](https://pytorch.org/docs/2.9/generated/torch.nn.PixelShuffle.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### _`class`_ torch.nn.PixelUnshuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.PixelUnshuffle](https://pytorch.org/docs/2.9/generated/torch.nn.PixelUnshuffle.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### _`class`_ torch.nn.Upsample

<div style="margin-left: 2em">

**原生文档**：[torch.nn.Upsample](https://pytorch.org/docs/2.9/generated/torch.nn.Upsample.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

### _`class`_ torch.nn.UpsamplingNearest2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.UpsamplingNearest2d](https://pytorch.org/docs/2.9/generated/torch.nn.UpsamplingNearest2d.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8
- 可能回退至CPU执行

</div>

## Shuffle Layers

### _`class`_ torch.nn.ChannelShuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.ChannelShuffle](https://pytorch.org/docs/2.9/generated/torch.nn.ChannelShuffle.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

## DataParallel Layers (multi-GPU, distributed)

### _`class`_ torch.nn.DataParallel

<div style="margin-left: 2em">

**原生文档**：[torch.nn.DataParallel](https://pytorch.org/docs/2.9/generated/torch.nn.DataParallel.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.parallel.DistributedDataParallel

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html)

**是否支持**：是

> <font size="3">join()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.DistributedDataParallel.join](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.DistributedDataParallel.join_hook](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">no_sync()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.DistributedDataParallel.no_sync](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_comm_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.DistributedDataParallel.register_comm_hook](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Utilities

### torch.nn.utils.clip_grad_norm_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.nn.utils.clip_grad_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.clip_grad_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm.html)

**是否支持**：否

</div>

### torch.nn.utils.clip_grad_value_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_value_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.utils.vector_to_parameters

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.vector_to_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.utils.vector_to_parameters.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，complex64

</div>

### torch.nn.utils.weight_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.weight_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.weight_norm.html)

**是否支持**：是

</div>

### torch.nn.utils.spectral_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.spectral_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.spectral_norm.html)

**是否支持**：是

</div>

### torch.nn.utils.remove_spectral_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.remove_spectral_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.remove_spectral_norm.html)

**是否支持**：是

</div>

### torch.nn.utils.skip_init

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.skip_init](https://pytorch.org/docs/2.9/generated/torch.nn.utils.skip_init.html)

**是否支持**：是

</div>

### _`class`_ torch.nn.utils.prune.BasePruningMethod

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.apply)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.apply_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.compute_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.prune)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.BasePruningMethod.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.remove)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.utils.prune.PruningContainer

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">add_pruning_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.add_pruning_method](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.add_pruning_method)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.apply)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.apply_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.compute_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.prune)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.PruningContainer.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.remove)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.utils.prune.Identity

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.Identity](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.Identity.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity.apply)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.Identity.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity.apply_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.Identity.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity.prune)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.Identity.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity.remove)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.utils.prune.RandomUnstructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomUnstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomUnstructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.apply)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomUnstructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.apply_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomUnstructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.prune)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomUnstructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.remove)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.nn.utils.prune.L1Unstructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.L1Unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.L1Unstructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.apply)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.L1Unstructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.apply_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.L1Unstructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.prune)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.L1Unstructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.remove)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.utils.prune.RandomStructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.apply)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.apply_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.compute_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.prune)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.RandomStructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.remove)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.nn.utils.prune.LnStructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.apply)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.apply_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">compute_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.compute_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.prune)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.LnStructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.remove)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.nn.utils.prune.CustomFromMask

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.CustomFromMask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持int64

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.CustomFromMask.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.apply)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持int64

</div>

> <font size="3">apply_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.CustomFromMask.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.apply_mask)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">prune()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.CustomFromMask.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.prune)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">remove()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.CustomFromMask.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.remove)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### torch.nn.utils.prune.random_unstructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.random_unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.random_unstructured.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.utils.prune.l1_unstructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.l1_unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.l1_unstructured.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.utils.prune.random_structured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.random_structured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.random_structured.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.utils.prune.ln_structured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.ln_structured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.ln_structured.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.utils.prune.global_unstructured

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.global_unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.global_unstructured.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.utils.prune.identity

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.identity](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.identity.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.utils.prune.custom_from_mask

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.custom_from_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.custom_from_mask.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持int64

</div>

### torch.nn.utils.prune.remove

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.remove.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.utils.prune.is_pruned

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.prune.is_pruned](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.is_pruned.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.utils.parametrizations.orthogonal

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrizations.orthogonal](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrizations.orthogonal.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.utils.parametrizations.spectral_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrizations.spectral_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrizations.spectral_norm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.nn.utils.parametrize.register_parametrization

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.register_parametrization](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.register_parametrization.html)

**是否支持**：是

</div>

### torch.nn.utils.parametrize.remove_parametrizations

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.remove_parametrizations](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.remove_parametrizations.html)

**是否支持**：是

</div>

### torch.nn.utils.parametrize.cached

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.cached](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.cached.html)

**是否支持**：是

</div>

### torch.nn.utils.parametrize.is_parametrized

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.is_parametrized](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.is_parametrized.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.nn.utils.parametrize.ParametrizationList

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.ParametrizationList](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.ParametrizationList.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">right_inverse()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.parametrize.ParametrizationList.right_inverse](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.ParametrizationList.html#torch.nn.utils.parametrize.ParametrizationList.right_inverse)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### torch.nn.utils.stateless.functional_call

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.stateless.functional_call](https://pytorch.org/docs/2.9/generated/torch.nn.utils.stateless.functional_call.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.nn.utils.rnn.PackedSequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html)

**是否支持**：是

**限制与说明**： 支持fp32，int64

> <font size="3">batch_sizes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.batch_sizes](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.batch_sizes)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">count()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.count](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.count)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">data()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.data](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.data)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">index()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.index](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.index)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.is_cuda](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.is_cuda)

**是否支持**：否

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.is_pinned](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.is_pinned)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sorted_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.sorted_indices](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.sorted_indices)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.to](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.to)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32，int64

</div>

> <font size="3">unsorted_indices()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.PackedSequence.unsorted_indices](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.unsorted_indices)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### torch.nn.utils.rnn.pack_padded_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pack_padded_sequence.html)

**是否支持**：否

</div>

### torch.nn.utils.rnn.pad_packed_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.pad_packed_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pad_packed_sequence.html)

**是否支持**：否

</div>

### torch.nn.utils.rnn.pad_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pad_sequence.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.utils.rnn.pack_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.pack_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pack_sequence.html)

**是否支持**：否

</div>

### torch.nn.utils.rnn.unpack_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.unpack_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.unpack_sequence.html)

**是否支持**：否

</div>

### torch.nn.utils.rnn.unpad_sequence

<div style="margin-left: 2em">

**原生文档**：[torch.nn.utils.rnn.unpad_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.unpad_sequence.html)

**是否支持**：否

</div>

### _`class`_ torch.nn.modules.flatten.Flatten

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.flatten.Flatten](https://pytorch.org/docs/2.9/generated/torch.nn.modules.flatten.Flatten.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### _`class`_ torch.nn.modules.flatten.Unflatten

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.flatten.Unflatten](https://pytorch.org/docs/2.9/generated/torch.nn.modules.flatten.Unflatten.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

> <font size="3">NamedShape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.flatten.Unflatten.NamedShape](https://docs.pytorch.org/docs/2.9/generated/torch.nn.modules.flatten.Unflatten.html#torch.nn.modules.flatten.Unflatten.NamedShape)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Lazy Modules Initialization

### _`class`_ torch.nn.modules.lazy.LazyModuleMixin

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.lazy.LazyModuleMixin](https://pytorch.org/docs/2.9/generated/torch.nn.modules.lazy.LazyModuleMixin.html)

**是否支持**：是

**限制与说明**： 支持fp32

> <font size="3">has_uninitialized_params()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params](https://pytorch.org/docs/2.9/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">initialize_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>
