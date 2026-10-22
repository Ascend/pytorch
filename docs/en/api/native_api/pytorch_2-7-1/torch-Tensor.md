# torch.Tensor

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/tensors.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [torch.Tensor](#torchtensor)
  - [​torch.Tensor](#torchtensor-1)
  - [Tensor class reference](#tensor-class-reference)
    - [class torch.Tensor](#class-torchtensor)

</div>

<div style="display:none;">

## &#8203;torch.Tensor

</div>

## Tensor class reference

### <code><i>class</i></code> torch.Tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tensor](https://pytorch.org/docs/2.7/tensors.html)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id3 -->

> <font size="3">T</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.T](https://pytorch.org/docs/2.7/tensors.html#torch.Tensor.T)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id6 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">H</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.H](https://pytorch.org/docs/2.7/tensors.html#torch.Tensor.H)

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

</div>

> <font size="3">mT</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.mT](https://pytorch.org/docs/2.7/tensors.html#torch.Tensor.mT)

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

> <font size="3">mH</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.mH](https://pytorch.org/docs/2.7/tensors.html#torch.Tensor.mH)

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

> <font size="3">new_tensor()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.new_tensor](https://pytorch.org/docs/2.7/generated/torch.Tensor.new_tensor.html)

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

</div>

> <font size="3">new_full()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.new_full](https://pytorch.org/docs/2.7/generated/torch.Tensor.new_full.html)

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

**Constraints & Notes**: `self` only supports int64

</div>

> <font size="3">new_empty()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.new_empty](https://pytorch.org/docs/2.7/generated/torch.Tensor.new_empty.html)

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

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">new_ones()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.new_ones](https://pytorch.org/docs/2.7/generated/torch.Tensor.new_ones.html)

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

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">new_zeros()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.new_zeros](https://pytorch.org/docs/2.7/generated/torch.Tensor.new_zeros.html)

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

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_cuda](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_cuda.html)

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

> <font size="3">is_quantized()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_quantized](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_quantized.html)

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

> <font size="3">is_meta()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_meta](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_meta.html)

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

</div>

> <font size="3">device()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.device](https://pytorch.org/docs/2.7/generated/torch.Tensor.device.html)

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

</div>

> <font size="3">grad()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.grad](https://pytorch.org/docs/2.7/generated/torch.Tensor.grad.html)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">ndim()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ndim](https://pytorch.org/docs/2.7/generated/torch.Tensor.ndim.html)

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

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">real()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.real](https://pytorch.org/docs/2.7/generated/torch.Tensor.real.html)

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

</div>

> <font size="3">imag()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.imag](https://pytorch.org/docs/2.7/generated/torch.Tensor.imag.html)

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

</div>

> <font size="3">nbytes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.nbytes](https://pytorch.org/docs/2.7/generated/torch.Tensor.nbytes.html)

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

</div>

> <font size="3">itemsize()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.itemsize](https://pytorch.org/docs/2.7/generated/torch.Tensor.itemsize.html)

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

> <font size="3">abs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.abs](https://pytorch.org/docs/2.7/generated/torch.Tensor.abs.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">abs_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.abs_](https://pytorch.org/docs/2.7/generated/torch.Tensor.abs_.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">absolute()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.absolute](https://pytorch.org/docs/2.7/generated/torch.Tensor.absolute.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">absolute_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.absolute_](https://pytorch.org/docs/2.7/generated/torch.Tensor.absolute_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">acos()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.acos](https://pytorch.org/docs/2.7/generated/torch.Tensor.acos.html)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

> <font size="3">acos_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.acos_](https://pytorch.org/docs/2.7/generated/torch.Tensor.acos_.html)

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

**Constraints & Notes**:

- `self` only supports fp16, fp32, fp64

<!-- npu="A3" id79 -->
- <term>Atlas A3 training products</term> additionally support bf16
<!-- end id79 -->

</div>

> <font size="3">arccos()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arccos](https://pytorch.org/docs/2.7/generated/torch.Tensor.arccos.html)

**Support Status**:

<!-- npu="910b" id80 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id80 -->
<!-- npu="A3" id81 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id81 -->
<!-- npu="950" id82 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id82 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">arccos_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arccos_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arccos_.html)

**Support Status**:

<!-- npu="910b" id83 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id83 -->
<!-- npu="A3" id84 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id84 -->
<!-- npu="950" id85 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id85 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32, fp64

<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term> additionally support bf16
<!-- end id86 -->

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.add](https://pytorch.org/docs/2.7/generated/torch.Tensor.add.html)

**Support Status**:

<!-- npu="910b" id87 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id87 -->
<!-- npu="A3" id88 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id88 -->
<!-- npu="950" id89 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id89 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.add_](https://pytorch.org/docs/2.7/generated/torch.Tensor.add_.html)

**Support Status**:

<!-- npu="910b" id90 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id90 -->
<!-- npu="A3" id91 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id91 -->
<!-- npu="950" id92 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id92 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">addbmm()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addbmm](https://pytorch.org/docs/2.7/generated/torch.Tensor.addbmm.html)

**Support Status**:

<!-- npu="910b" id93 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id93 -->
<!-- npu="A3" id94 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id94 -->
<!-- npu="950" id95 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id95 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">addbmm_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addbmm_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addbmm_.html)

**Support Status**:

<!-- npu="910b" id96 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id96 -->
<!-- npu="A3" id97 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id97 -->
<!-- npu="950" id98 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id98 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">addcdiv()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addcdiv](https://pytorch.org/docs/2.7/generated/torch.Tensor.addcdiv.html)

**Support Status**:

<!-- npu="910b" id99 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id99 -->
<!-- npu="A3" id100 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id100 -->
<!-- npu="950" id101 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id101 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, int64
- The int64 type does not support simultaneous broadcasting of three `tensor`s

</div>

> <font size="3">addcdiv_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addcdiv_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addcdiv_.html)

**Support Status**:

<!-- npu="910b" id102 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id102 -->
<!-- npu="A3" id103 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id103 -->
<!-- npu="950" id104 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id104 -->

**Constraints & Notes**:

<!-- npu="A3,910b" id105 -->
- <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>: `self` only supports bf16, fp16, fp32, and fp64
<!-- end id105 -->
<!-- npu="910" id106 -->
- <term>Atlas training products</term>: `self` only supports fp16, fp32, and fp64
<!-- end id106 -->
- The int64 type does not support simultaneous broadcasting of three `tensor`s

</div>

> <font size="3">addcmul()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addcmul](https://pytorch.org/docs/2.7/generated/torch.Tensor.addcmul.html)

**Support Status**:

<!-- npu="910b" id107 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id107 -->
<!-- npu="A3" id108 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id108 -->
<!-- npu="950" id109 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id109 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32, int64
- The int64 type does not support simultaneous broadcasting of three `tensor`s

</div>

> <font size="3">addcmul_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addcmul_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addcmul_.html)

**Support Status**:

<!-- npu="910b" id110 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id110 -->
<!-- npu="A3" id111 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id111 -->
<!-- npu="950" id112 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id112 -->

**Constraints & Notes**:

<!-- npu="A3,910b" id113 -->
- <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, and int64
<!-- end id113 -->
<!-- npu="910" id114 -->
- <term>Atlas training products</term>: `self` only supports fp16, fp32, fp64, uint8, int8, int32, and int64
<!-- end id114 -->
- The int64 type does not support simultaneous broadcasting of three `tensor`s

</div>

> <font size="3">addmm()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addmm](https://pytorch.org/docs/2.7/generated/torch.Tensor.addmm.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32

</div>

> <font size="3">addmm_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addmm_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addmm_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">sspaddmm()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sspaddmm](https://pytorch.org/docs/2.7/generated/torch.Tensor.sspaddmm.html)

**Support Status**:

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id123 -->

</div>

> <font size="3">addmv()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addmv](https://pytorch.org/docs/2.7/generated/torch.Tensor.addmv.html)

**Support Status**:

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id126 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">addmv_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addmv_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addmv_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">addr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addr](https://pytorch.org/docs/2.7/generated/torch.Tensor.addr.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">addr_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.addr_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addr_.html)

**Support Status**:

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id135 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">adjoint()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.adjoint](https://pytorch.org/docs/2.7/generated/torch.Tensor.adjoint.html)

**Support Status**:

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id138 -->

</div>

> <font size="3">allclose()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.allclose](https://pytorch.org/docs/2.7/generated/torch.Tensor.allclose.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">amax()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.amax](https://pytorch.org/docs/2.7/generated/torch.Tensor.amax.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">amin()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.amin](https://pytorch.org/docs/2.7/generated/torch.Tensor.amin.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">aminmax()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.aminmax](https://pytorch.org/docs/2.7/generated/torch.Tensor.aminmax.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">angle()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.angle](https://pytorch.org/docs/2.7/generated/torch.Tensor.angle.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

> <font size="3">apply_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.apply_](https://pytorch.org/docs/2.7/generated/torch.Tensor.apply_.html)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id156 -->

**Constraints & Notes**: CPU-only

</div>

> <font size="3">argmax()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.argmax](https://pytorch.org/docs/2.7/generated/torch.Tensor.argmax.html)

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id159 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">argmin()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.argmin](https://pytorch.org/docs/2.7/generated/torch.Tensor.argmin.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">argsort()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.argsort](https://pytorch.org/docs/2.7/generated/torch.Tensor.argsort.html)

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

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

<!-- npu="950" id166 -->
- For <term>Ascend 950DT products</term>, due to underlying implementation limitations, `"stable"` only supports True. If set to False, it is automatically changed to True during execution
<!-- end id166 -->

</div>

> <font size="3">argwhere()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.argwhere](https://pytorch.org/docs/2.7/generated/torch.Tensor.argwhere.html)

**Support Status**:

<!-- npu="910b" id167 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id167 -->
<!-- npu="A3" id168 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id168 -->
<!-- npu="950" id169 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id169 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">asin()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.asin](https://pytorch.org/docs/2.7/generated/torch.Tensor.asin.html)

**Support Status**:

<!-- npu="910b" id170 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id170 -->
<!-- npu="A3" id171 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id171 -->
<!-- npu="950" id172 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id172 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">asin_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.asin_](https://pytorch.org/docs/2.7/generated/torch.Tensor.asin_.html)

**Support Status**:

<!-- npu="910b" id173 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id173 -->
<!-- npu="A3" id174 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id174 -->
<!-- npu="950" id175 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id175 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">arcsin()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arcsin](https://pytorch.org/docs/2.7/generated/torch.Tensor.arcsin.html)

**Support Status**:

<!-- npu="910b" id176 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id176 -->
<!-- npu="A3" id177 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id177 -->
<!-- npu="950" id178 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id178 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">arcsin_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arcsin_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arcsin_.html)

**Support Status**:

<!-- npu="910b" id179 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id179 -->
<!-- npu="A3" id180 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id180 -->
<!-- npu="950" id181 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id181 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">as_strided()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.as_strided](https://pytorch.org/docs/2.7/generated/torch.Tensor.as_strided.html)

**Support Status**:

<!-- npu="910b" id182 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id182 -->
<!-- npu="A3" id183 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id183 -->
<!-- npu="950" id184 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id184 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">atan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.atan](https://pytorch.org/docs/2.7/generated/torch.Tensor.atan.html)

**Support Status**:

<!-- npu="910b" id185 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id185 -->
<!-- npu="A3" id186 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id186 -->
<!-- npu="950" id187 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id187 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">atan_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.atan_](https://pytorch.org/docs/2.7/generated/torch.Tensor.atan_.html)

**Support Status**:

<!-- npu="910b" id188 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id188 -->
<!-- npu="A3" id189 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id189 -->
<!-- npu="950" id190 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id190 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">arctan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arctan](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctan.html)

**Support Status**:

<!-- npu="910b" id191 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id191 -->
<!-- npu="A3" id192 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id192 -->
<!-- npu="950" id193 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id193 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">arctan_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arctan_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctan_.html)

**Support Status**:

<!-- npu="910b" id194 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id194 -->
<!-- npu="A3" id195 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id195 -->
<!-- npu="950" id196 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id196 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">atan2()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.atan2](https://pytorch.org/docs/2.7/generated/torch.Tensor.atan2.html)

**Support Status**:

<!-- npu="910b" id197 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id197 -->
<!-- npu="A3" id198 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id198 -->
<!-- npu="950" id199 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id199 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">atan2_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.atan2_](https://pytorch.org/docs/2.7/generated/torch.Tensor.atan2_.html)

**Support Status**:

<!-- npu="910b" id200 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id200 -->
<!-- npu="A3" id201 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id201 -->
<!-- npu="950" id202 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id202 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32
- May fall back to CPU execution

</div>

> <font size="3">arctan2()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arctan2](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctan2.html)

**Support Status**:

<!-- npu="910b" id203 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id203 -->
<!-- npu="A3" id204 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id204 -->
<!-- npu="950" id205 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id205 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">arctan2_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arctan2_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctan2_.html)

**Support Status**:

<!-- npu="910b" id206 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id206 -->
<!-- npu="A3" id207 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id207 -->
<!-- npu="950" id208 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id208 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">all()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.all](https://pytorch.org/docs/2.7/generated/torch.Tensor.all.html)

**Support Status**:

<!-- npu="910b" id209 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id209 -->
<!-- npu="A3" id210 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id210 -->
<!-- npu="950" id211 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id211 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">any()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.any](https://pytorch.org/docs/2.7/generated/torch.Tensor.any.html)

**Support Status**:

<!-- npu="910b" id212 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id212 -->
<!-- npu="A3" id213 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id213 -->
<!-- npu="950" id214 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id214 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">backward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.backward](https://pytorch.org/docs/2.7/generated/torch.Tensor.backward.html)

**Support Status**:

<!-- npu="910b" id215 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id215 -->
<!-- npu="A3" id216 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id216 -->
<!-- npu="950" id217 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id217 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">baddbmm()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.baddbmm](https://pytorch.org/docs/2.7/generated/torch.Tensor.baddbmm.html)

**Support Status**:

<!-- npu="910b" id218 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id218 -->
<!-- npu="A3" id219 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id219 -->
<!-- npu="950" id220 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id220 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">baddbmm_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.baddbmm_](https://pytorch.org/docs/2.7/generated/torch.Tensor.baddbmm_.html)

**Support Status**:

<!-- npu="910b" id221 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id221 -->
<!-- npu="A3" id222 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id222 -->
<!-- npu="950" id223 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id223 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">bernoulli()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bernoulli](https://pytorch.org/docs/2.7/generated/torch.Tensor.bernoulli.html)

**Support Status**:

<!-- npu="910b" id224 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id224 -->
<!-- npu="A3" id225 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id225 -->
<!-- npu="950" id226 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id226 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32
- May fall back to CPU execution

</div>

> <font size="3">bernoulli_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bernoulli_](https://pytorch.org/docs/2.7/generated/torch.Tensor.bernoulli_.html)

**Support Status**:

<!-- npu="910b" id227 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id227 -->
<!-- npu="A3" id228 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id228 -->
<!-- npu="950" id229 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id229 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bfloat16](https://pytorch.org/docs/2.7/generated/torch.Tensor.bfloat16.html)

**Support Status**:

<!-- npu="910b" id230 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id230 -->
<!-- npu="A3" id231 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id231 -->
<!-- npu="950" id232 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id232 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">bincount()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bincount](https://pytorch.org/docs/2.7/generated/torch.Tensor.bincount.html)

**Support Status**:

<!-- npu="910b" id233 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id233 -->
<!-- npu="A3" id234 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id234 -->
<!-- npu="950" id235 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id235 -->

**Constraints & Notes**:

- `self` only supports uint8, int8, int16, int32, int64
- The dimension of `weights` must be the same as that of `input`

</div>

> <font size="3">bitwise_not()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bitwise_not](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_not.html)

**Support Status**:

<!-- npu="910b" id236 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id236 -->
<!-- npu="A3" id237 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id237 -->
<!-- npu="950" id238 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id238 -->

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_not_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bitwise_not_](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_not_.html)

**Support Status**:

<!-- npu="910b" id239 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id239 -->
<!-- npu="A3" id240 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id240 -->
<!-- npu="950" id241 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id241 -->

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_and()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bitwise_and](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_and.html)

**Support Status**:

<!-- npu="910b" id242 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id242 -->
<!-- npu="A3" id243 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id243 -->
<!-- npu="950" id244 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id244 -->

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_and_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bitwise_and_](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_and_.html)

**Support Status**:

<!-- npu="910b" id245 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id245 -->
<!-- npu="A3" id246 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id246 -->
<!-- npu="950" id247 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id247 -->

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_or()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bitwise_or](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_or.html)

**Support Status**:

<!-- npu="910b" id248 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id248 -->
<!-- npu="A3" id249 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id249 -->
<!-- npu="950" id250 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id250 -->

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_or_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bitwise_or_](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_or_.html)

**Support Status**:

<!-- npu="910b" id251 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id251 -->
<!-- npu="A3" id252 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id252 -->
<!-- npu="950" id253 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id253 -->

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_xor()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bitwise_xor](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_xor.html)

**Support Status**:

<!-- npu="910b" id254 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id254 -->
<!-- npu="A3" id255 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id255 -->
<!-- npu="950" id256 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id256 -->

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bitwise_xor_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bitwise_xor_](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_xor_.html)

**Support Status**:

<!-- npu="910b" id257 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id257 -->
<!-- npu="A3" id258 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id258 -->
<!-- npu="950" id259 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id259 -->

**Constraints & Notes**: `self` only supports uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">bmm()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bmm](https://pytorch.org/docs/2.7/generated/torch.Tensor.bmm.html)

**Support Status**:

<!-- npu="910b" id260 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id260 -->
<!-- npu="A3" id261 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id261 -->
<!-- npu="950" id262 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id262 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32

</div>

> <font size="3">bool()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.bool](https://pytorch.org/docs/2.7/generated/torch.Tensor.bool.html)

**Support Status**:

<!-- npu="910b" id263 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id263 -->
<!-- npu="A3" id264 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id264 -->
<!-- npu="950" id265 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id265 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">byte()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.byte](https://pytorch.org/docs/2.7/generated/torch.Tensor.byte.html)

**Support Status**:

<!-- npu="910b" id266 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id266 -->
<!-- npu="A3" id267 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id267 -->
<!-- npu="950" id268 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id268 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">broadcast_to()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.broadcast_to](https://pytorch.org/docs/2.7/generated/torch.Tensor.broadcast_to.html)

**Support Status**:

<!-- npu="910b" id269 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id269 -->
<!-- npu="A3" id270 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id270 -->
<!-- npu="950" id271 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id271 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">ceil()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ceil](https://pytorch.org/docs/2.7/generated/torch.Tensor.ceil.html)

**Support Status**:

<!-- npu="910b" id272 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id272 -->
<!-- npu="A3" id273 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id273 -->
<!-- npu="950" id274 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id274 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">ceil_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ceil_](https://pytorch.org/docs/2.7/generated/torch.Tensor.ceil_.html)

**Support Status**:

<!-- npu="910b" id275 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id275 -->
<!-- npu="A3" id276 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id276 -->
<!-- npu="950" id277 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id277 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">char()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.char](https://pytorch.org/docs/2.7/generated/torch.Tensor.char.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">chunk()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.chunk](https://pytorch.org/docs/2.7/generated/torch.Tensor.chunk.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">clamp()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.clamp](https://pytorch.org/docs/2.7/generated/torch.Tensor.clamp.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">clamp_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.clamp_](https://pytorch.org/docs/2.7/generated/torch.Tensor.clamp_.html)

**Support Status**:

<!-- npu="910b" id287 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id287 -->
<!-- npu="A3" id288 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id288 -->
<!-- npu="950" id289 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id289 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">clip()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.clip](https://pytorch.org/docs/2.7/generated/torch.Tensor.clip.html)

**Support Status**:

<!-- npu="910b" id290 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id290 -->
<!-- npu="A3" id291 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id291 -->
<!-- npu="950" id292 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id292 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">clip_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.clip_](https://pytorch.org/docs/2.7/generated/torch.Tensor.clip_.html)

**Support Status**:

<!-- npu="910b" id293 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id293 -->
<!-- npu="A3" id294 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id294 -->
<!-- npu="950" id295 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id295 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.clone](https://pytorch.org/docs/2.7/generated/torch.Tensor.clone.html)

**Support Status**:

<!-- npu="910b" id296 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id296 -->
<!-- npu="A3" id297 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id297 -->
<!-- npu="950" id298 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id298 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">contiguous()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.contiguous](https://pytorch.org/docs/2.7/generated/torch.Tensor.contiguous.html)

**Support Status**:

<!-- npu="910b" id299 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id299 -->
<!-- npu="A3" id300 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id300 -->
<!-- npu="950" id301 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id301 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">copy_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.copy_](https://pytorch.org/docs/2.7/generated/torch.Tensor.copy_.html)

**Support Status**:

<!-- npu="910b" id302 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id302 -->
<!-- npu="A3" id303 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id303 -->
<!-- npu="950" id304 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id304 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- int16 does not support inputs with 5 or more dimensions

</div>

> <font size="3">conj()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.conj](https://pytorch.org/docs/2.7/generated/torch.Tensor.conj.html)

**Support Status**:

<!-- npu="910b" id305 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id305 -->
<!-- npu="A3" id306 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id306 -->
<!-- npu="950" id307 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id307 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">resolve_conj()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.resolve_conj](https://pytorch.org/docs/2.7/generated/torch.Tensor.resolve_conj.html)

**Support Status**:

<!-- npu="910b" id308 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id308 -->
<!-- npu="A3" id309 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id309 -->
<!-- npu="950" id310 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id310 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">resolve_neg()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.resolve_neg](https://pytorch.org/docs/2.7/generated/torch.Tensor.resolve_neg.html)

**Support Status**:

<!-- npu="910b" id311 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id311 -->
<!-- npu="A3" id312 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id312 -->
<!-- npu="950" id313 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id313 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">copysign()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.copysign](https://pytorch.org/docs/2.7/generated/torch.Tensor.copysign.html)

**Support Status**:

<!-- npu="910b" id314 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id314 -->
<!-- npu="A3" id315 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id315 -->
<!-- npu="950" id316 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id316 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

> <font size="3">cos()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cos](https://pytorch.org/docs/2.7/generated/torch.Tensor.cos.html)

**Support Status**:

<!-- npu="910b" id317 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id317 -->
<!-- npu="A3" id318 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id318 -->
<!-- npu="950" id319 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id319 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">cos_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cos_](https://pytorch.org/docs/2.7/generated/torch.Tensor.cos_.html)

**Support Status**:

<!-- npu="910b" id320 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id320 -->
<!-- npu="A3" id321 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id321 -->
<!-- npu="950" id322 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id322 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, complex64, complex128

</div>

> <font size="3">cosh()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cosh](https://pytorch.org/docs/2.7/generated/torch.Tensor.cosh.html)

**Support Status**:

<!-- npu="910b" id323 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id323 -->
<!-- npu="A3" id324 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id324 -->
<!-- npu="950" id325 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id325 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">cosh_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cosh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.cosh_.html)

**Support Status**:

<!-- npu="910b" id326 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id326 -->
<!-- npu="A3" id327 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id327 -->
<!-- npu="950" id328 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id328 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">count_nonzero()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.count_nonzero](https://pytorch.org/docs/2.7/generated/torch.Tensor.count_nonzero.html)

**Support Status**:

<!-- npu="910b" id329 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id329 -->
<!-- npu="A3" id330 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id330 -->
<!-- npu="950" id331 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id331 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">cov()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cov](https://pytorch.org/docs/2.7/generated/torch.Tensor.cov.html)

**Support Status**:

<!-- npu="910b" id332 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id332 -->
<!-- npu="A3" id333 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id333 -->
<!-- npu="950" id334 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id334 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, int64

</div>

> <font size="3">acosh()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.acosh](https://pytorch.org/docs/2.7/generated/torch.Tensor.acosh.html)

**Support Status**:

<!-- npu="910b" id335 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id335 -->
<!-- npu="A3" id336 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id336 -->
<!-- npu="950" id337 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id337 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- May fall back to CPU execution

</div>

> <font size="3">acosh_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.acosh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.acosh_.html)

**Support Status**:

<!-- npu="910b" id338 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id338 -->
<!-- npu="A3" id339 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id339 -->
<!-- npu="950" id340 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id340 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">arccosh()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arccosh](https://pytorch.org/docs/2.7/generated/torch.Tensor.arccosh.html)

**Support Status**:

<!-- npu="910b" id341 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id341 -->
<!-- npu="A3" id342 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id342 -->
<!-- npu="950" id343 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id343 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">arccosh_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arccosh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arccosh_.html)

**Support Status**:

<!-- npu="910b" id344 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id344 -->
<!-- npu="A3" id345 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id345 -->
<!-- npu="950" id346 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id346 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cpu](https://pytorch.org/docs/2.7/generated/torch.Tensor.cpu.html)

**Support Status**:

<!-- npu="910b" id347 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id347 -->
<!-- npu="A3" id348 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id348 -->
<!-- npu="950" id349 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id349 -->

</div>

> <font size="3">cross()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cross](https://pytorch.org/docs/2.7/generated/torch.Tensor.cross.html)

**Support Status**:

<!-- npu="910b" id350 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id350 -->
<!-- npu="A3" id351 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id351 -->
<!-- npu="950" id352 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id352 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128
- The shapes of the two inputs must be consistent

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cuda](https://pytorch.org/docs/2.7/generated/torch.Tensor.cuda.html)

**Support Status**:

<!-- npu="910b" id353 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id353 -->
<!-- npu="A3" id354 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id354 -->
<!-- npu="950" id355 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id355 -->

**Constraints & Notes**: The NPU-Adapted interface is `Tensor.npu`, whose `memory_format` parameter only supports torch.contiguous_format

</div>

> <font size="3">cummax()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cummax](https://pytorch.org/docs/2.7/generated/torch.Tensor.cummax.html)

**Support Status**:

<!-- npu="910b" id356 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id356 -->
<!-- npu="A3" id357 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id357 -->
<!-- npu="950" id358 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id358 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

> <font size="3">cummin()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cummin](https://pytorch.org/docs/2.7/generated/torch.Tensor.cummin.html)

**Support Status**:

<!-- npu="910b" id359 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id359 -->
<!-- npu="A3" id360 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id360 -->
<!-- npu="950" id361 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id361 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">cumsum()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cumsum](https://pytorch.org/docs/2.7/generated/torch.Tensor.cumsum.html)

**Support Status**:

<!-- npu="910b" id362 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id362 -->
<!-- npu="A3" id363 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id363 -->
<!-- npu="950" id364 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id364 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Named Tensor is supported

</div>

> <font size="3">cumsum_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cumsum_](https://pytorch.org/docs/2.7/generated/torch.Tensor.cumsum_.html)

**Support Status**:

<!-- npu="910b" id365 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id365 -->
<!-- npu="A3" id366 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id366 -->
<!-- npu="950" id367 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id367 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">chalf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.chalf](https://pytorch.org/docs/2.7/generated/torch.Tensor.chalf.html)

**Support Status**:

<!-- npu="910b" id368 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id368 -->
<!-- npu="A3" id369 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id369 -->
<!-- npu="950" id370 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id370 -->

</div>

> <font size="3">cfloat()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cfloat](https://pytorch.org/docs/2.7/generated/torch.Tensor.cfloat.html)

**Support Status**:

<!-- npu="910b" id371 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id371 -->
<!-- npu="A3" id372 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id372 -->
<!-- npu="950" id373 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id373 -->

</div>

> <font size="3">cdouble()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.cdouble](https://pytorch.org/docs/2.7/generated/torch.Tensor.cdouble.html)

**Support Status**:

<!-- npu="910b" id374 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id374 -->
<!-- npu="A3" id375 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id375 -->
<!-- npu="950" id376 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id376 -->

</div>

> <font size="3">data_ptr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.data_ptr](https://pytorch.org/docs/2.7/generated/torch.Tensor.data_ptr.html)

**Support Status**:

<!-- npu="910b" id377 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id377 -->
<!-- npu="A3" id378 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id378 -->
<!-- npu="950" id379 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id379 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">deg2rad()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.deg2rad](https://pytorch.org/docs/2.7/generated/torch.Tensor.deg2rad.html)

**Support Status**:

<!-- npu="910b" id380 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id380 -->
<!-- npu="A3" id381 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id381 -->
<!-- npu="950" id382 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id382 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">dequantize()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.dequantize](https://pytorch.org/docs/2.7/generated/torch.Tensor.dequantize.html)

**Support Status**:

<!-- npu="910b" id383 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id383 -->
<!-- npu="A3" id384 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id384 -->
<!-- npu="950" id385 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id385 -->

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">dense_dim()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.dense_dim](https://pytorch.org/docs/2.7/generated/torch.Tensor.dense_dim.html)

**Support Status**:

<!-- npu="910b" id386 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id386 -->
<!-- npu="A3" id387 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id387 -->
<!-- npu="950" id388 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id388 -->

</div>

> <font size="3">detach()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.detach](https://pytorch.org/docs/2.7/generated/torch.Tensor.detach.html)

**Support Status**:

<!-- npu="910b" id389 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id389 -->
<!-- npu="A3" id390 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id390 -->
<!-- npu="950" id391 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id391 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">detach_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.detach_](https://pytorch.org/docs/2.7/generated/torch.Tensor.detach_.html)

**Support Status**:

<!-- npu="910b" id392 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id392 -->
<!-- npu="A3" id393 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id393 -->
<!-- npu="950" id394 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id394 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">diag()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.diag](https://pytorch.org/docs/2.7/generated/torch.Tensor.diag.html)

**Support Status**:

<!-- npu="910b" id395 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id395 -->
<!-- npu="A3" id396 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id396 -->
<!-- npu="950" id397 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id397 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

> <font size="3">diag_embed()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.diag_embed](https://pytorch.org/docs/2.7/generated/torch.Tensor.diag_embed.html)

**Support Status**:

<!-- npu="910b" id398 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id398 -->
<!-- npu="A3" id399 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id399 -->
<!-- npu="950" id400 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id400 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">diagflat()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.diagflat](https://pytorch.org/docs/2.7/generated/torch.Tensor.diagflat.html)

**Support Status**:

<!-- npu="910b" id401 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id401 -->
<!-- npu="A3" id402 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id402 -->
<!-- npu="950" id403 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id403 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

> <font size="3">diagonal()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.diagonal](https://pytorch.org/docs/2.7/generated/torch.Tensor.diagonal.html)

**Support Status**:

<!-- npu="910b" id404 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id404 -->
<!-- npu="A3" id405 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id405 -->
<!-- npu="950" id406 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id406 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">diagonal_scatter()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.diagonal_scatter](https://pytorch.org/docs/2.7/generated/torch.Tensor.diagonal_scatter.html)

**Support Status**:

<!-- npu="910b" id407 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id407 -->
<!-- npu="A3" id408 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id408 -->
<!-- npu="950" id409 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id409 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">fill_diagonal_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.fill_diagonal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.fill_diagonal_.html)

**Support Status**:

<!-- npu="910b" id410 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id410 -->
<!-- npu="A3" id411 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id411 -->
<!-- npu="950" id412 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id412 -->

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">diff()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.diff](https://pytorch.org/docs/2.7/generated/torch.Tensor.diff.html)

**Support Status**:

<!-- npu="910b" id413 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id413 -->
<!-- npu="A3" id414 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id414 -->
<!-- npu="950" id415 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id415 -->

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">dim()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.dim](https://pytorch.org/docs/2.7/generated/torch.Tensor.dim.html)

**Support Status**:

<!-- npu="910b" id416 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id416 -->
<!-- npu="A3" id417 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id417 -->
<!-- npu="950" id418 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id418 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">dim_order()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.dim_order](https://pytorch.org/docs/2.7/generated/torch.Tensor.dim_order.html)

**Support Status**:

<!-- npu="910b" id419 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id419 -->
<!-- npu="A3" id420 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id420 -->
<!-- npu="950" id421 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id421 -->

</div>

> <font size="3">dist()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.dist](https://pytorch.org/docs/2.7/generated/torch.Tensor.dist.html)

**Support Status**:

<!-- npu="910b" id422 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id422 -->
<!-- npu="A3" id423 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id423 -->
<!-- npu="950" id424 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id424 -->

<!-- npu="950" id1387 -->
**Constraints & Notes**: <term>Ascend 950DT products</term>: do not support fp64, complex64, or complex128
<!-- end id1387 -->

</div>

> <font size="3">div()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.div](https://pytorch.org/docs/2.7/generated/torch.Tensor.div.html)

**Support Status**:

<!-- npu="910b" id425 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id425 -->
<!-- npu="A3" id426 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id426 -->
<!-- npu="950" id427 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id427 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">div_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.div_](https://pytorch.org/docs/2.7/generated/torch.Tensor.div_.html)

**Support Status**:

<!-- npu="910b" id428 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id428 -->
<!-- npu="A3" id429 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id429 -->
<!-- npu="950" id430 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id430 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64

</div>

> <font size="3">divide()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.divide](https://pytorch.org/docs/2.7/generated/torch.Tensor.divide.html)

**Support Status**:

<!-- npu="910b" id431 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id431 -->
<!-- npu="A3" id432 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id432 -->
<!-- npu="950" id433 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id433 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">divide_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.divide_](https://pytorch.org/docs/2.7/generated/torch.Tensor.divide_.html)

**Support Status**:

<!-- npu="910b" id434 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id434 -->
<!-- npu="A3" id435 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id435 -->
<!-- npu="950" id436 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id436 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">dot()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.dot](https://pytorch.org/docs/2.7/generated/torch.Tensor.dot.html)

**Support Status**:

<!-- npu="910b" id437 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id437 -->
<!-- npu="A3" id438 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id438 -->
<!-- npu="950" id439 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id439 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.double](https://pytorch.org/docs/2.7/generated/torch.Tensor.double.html)

**Support Status**:

<!-- npu="910b" id440 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id440 -->
<!-- npu="A3" id441 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id441 -->
<!-- npu="950" id442 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id442 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Some interfaces on the NPU currently do not support the double type. For compatibility, fp32 is returned by default. After support is completed, fp64 will be returned properly.
- May fall back to CPU execution

</div>

> <font size="3">dsplit()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.dsplit](https://pytorch.org/docs/2.7/generated/torch.Tensor.dsplit.html)

**Support Status**:

<!-- npu="910b" id443 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id443 -->
<!-- npu="A3" id444 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id444 -->
<!-- npu="950" id445 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id445 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">element_size()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.element_size](https://pytorch.org/docs/2.7/generated/torch.Tensor.element_size.html)

**Support Status**:

<!-- npu="910b" id446 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id446 -->
<!-- npu="A3" id447 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id447 -->
<!-- npu="950" id448 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id448 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">eq()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.eq](https://pytorch.org/docs/2.7/generated/torch.Tensor.eq.html)

**Support Status**:

<!-- npu="910b" id449 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id449 -->
<!-- npu="A3" id450 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id450 -->
<!-- npu="950" id451 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id451 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">eq_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.eq_](https://pytorch.org/docs/2.7/generated/torch.Tensor.eq_.html)

**Support Status**:

<!-- npu="910b" id452 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id452 -->
<!-- npu="A3" id453 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id453 -->
<!-- npu="950" id454 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id454 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">equal()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.equal](https://pytorch.org/docs/2.7/generated/torch.Tensor.equal.html)

**Support Status**:

<!-- npu="910b" id455 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id455 -->
<!-- npu="A3" id456 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id456 -->
<!-- npu="950" id457 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id457 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">erf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.erf](https://pytorch.org/docs/2.7/generated/torch.Tensor.erf.html)

**Support Status**:

<!-- npu="910b" id458 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id458 -->
<!-- npu="A3" id459 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id459 -->
<!-- npu="950" id460 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id460 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">erf_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.erf_](https://pytorch.org/docs/2.7/generated/torch.Tensor.erf_.html)

**Support Status**:

<!-- npu="910b" id461 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id461 -->
<!-- npu="A3" id462 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id462 -->
<!-- npu="950" id463 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id463 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">erfc()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.erfc](https://pytorch.org/docs/2.7/generated/torch.Tensor.erfc.html)

**Support Status**:

<!-- npu="910b" id464 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id464 -->
<!-- npu="A3" id465 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id465 -->
<!-- npu="950" id466 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id466 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">erfc_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.erfc_](https://pytorch.org/docs/2.7/generated/torch.Tensor.erfc_.html)

**Support Status**:

<!-- npu="910b" id467 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id467 -->
<!-- npu="A3" id468 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id468 -->
<!-- npu="950" id469 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id469 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">erfinv()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.erfinv](https://pytorch.org/docs/2.7/generated/torch.Tensor.erfinv.html)

**Support Status**:

<!-- npu="910b" id470 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id470 -->
<!-- npu="A3" id471 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id471 -->
<!-- npu="950" id472 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id472 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">erfinv_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.erfinv_](https://pytorch.org/docs/2.7/generated/torch.Tensor.erfinv_.html)

**Support Status**:

<!-- npu="910b" id473 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id473 -->
<!-- npu="A3" id474 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id474 -->
<!-- npu="950" id475 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id475 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32

</div>

> <font size="3">exp()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.exp](https://pytorch.org/docs/2.7/generated/torch.Tensor.exp.html)

**Support Status**:

<!-- npu="910b" id476 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id476 -->
<!-- npu="A3" id477 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id477 -->
<!-- npu="950" id478 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id478 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, int64, bool

</div>

> <font size="3">exp_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.exp_](https://pytorch.org/docs/2.7/generated/torch.Tensor.exp_.html)

**Support Status**:

<!-- npu="910b" id479 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id479 -->
<!-- npu="A3" id480 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id480 -->
<!-- npu="950" id481 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id481 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, complex64, complex128

</div>

> <font size="3">expm1()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.expm1](https://pytorch.org/docs/2.7/generated/torch.Tensor.expm1.html)

**Support Status**:

<!-- npu="910b" id482 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id482 -->
<!-- npu="A3" id483 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id483 -->
<!-- npu="950" id484 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id484 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">expm1_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.expm1_](https://pytorch.org/docs/2.7/generated/torch.Tensor.expm1_.html)

**Support Status**:

<!-- npu="910b" id485 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id485 -->
<!-- npu="A3" id486 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id486 -->
<!-- npu="950" id487 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id487 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.expand](https://pytorch.org/docs/2.7/generated/torch.Tensor.expand.html)

**Support Status**:

<!-- npu="910b" id488 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id488 -->
<!-- npu="A3" id489 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id489 -->
<!-- npu="950" id490 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id490 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">expand_as()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.expand_as](https://pytorch.org/docs/2.7/generated/torch.Tensor.expand_as.html)

**Support Status**:

<!-- npu="910b" id491 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id491 -->
<!-- npu="A3" id492 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id492 -->
<!-- npu="950" id493 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id493 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">exponential_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.exponential_](https://pytorch.org/docs/2.7/generated/torch.Tensor.exponential_.html)

**Support Status**:

<!-- npu="910b" id494 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id494 -->
<!-- npu="A3" id495 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id495 -->
<!-- npu="950" id496 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id496 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64

</div>

> <font size="3">fix()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.fix](https://pytorch.org/docs/2.7/generated/torch.Tensor.fix.html)

**Support Status**:

<!-- npu="910b" id497 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id497 -->
<!-- npu="A3" id498 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id498 -->
<!-- npu="950" id499 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id499 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">fix_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.fix_](https://pytorch.org/docs/2.7/generated/torch.Tensor.fix_.html)

**Support Status**:

<!-- npu="910b" id500 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id500 -->
<!-- npu="A3" id501 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id501 -->
<!-- npu="950" id502 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id502 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">fill_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.fill_](https://pytorch.org/docs/2.7/generated/torch.Tensor.fill_.html)

**Support Status**:

<!-- npu="910b" id503 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id503 -->
<!-- npu="A3" id504 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id504 -->
<!-- npu="950" id505 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id505 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">flatten()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.flatten](https://pytorch.org/docs/2.7/generated/torch.Tensor.flatten.html)

**Support Status**:

<!-- npu="910b" id506 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id506 -->
<!-- npu="A3" id507 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id507 -->
<!-- npu="950" id508 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id508 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">flip()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.flip](https://pytorch.org/docs/2.7/generated/torch.Tensor.flip.html)

**Support Status**:

<!-- npu="910b" id509 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id509 -->
<!-- npu="A3" id510 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id510 -->
<!-- npu="950" id511 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id511 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">fliplr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.fliplr](https://pytorch.org/docs/2.7/generated/torch.Tensor.fliplr.html)

**Support Status**:

<!-- npu="910b" id512 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id512 -->
<!-- npu="A3" id513 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id513 -->
<!-- npu="950" id514 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id514 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">flipud()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.flipud](https://pytorch.org/docs/2.7/generated/torch.Tensor.flipud.html)

**Support Status**:

<!-- npu="910b" id515 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id515 -->
<!-- npu="A3" id516 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id516 -->
<!-- npu="950" id517 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id517 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.float](https://pytorch.org/docs/2.7/generated/torch.Tensor.float.html)

**Support Status**:

<!-- npu="910b" id518 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id518 -->
<!-- npu="A3" id519 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id519 -->
<!-- npu="950" id520 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id520 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">float_power()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.float_power](https://pytorch.org/docs/2.7/generated/torch.Tensor.float_power.html)

**Support Status**:

<!-- npu="910b" id521 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id521 -->
<!-- npu="A3" id522 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id522 -->
<!-- npu="950" id523 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id523 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex128

</div>

> <font size="3">float_power_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.float_power_](https://pytorch.org/docs/2.7/generated/torch.Tensor.float_power_.html)

**Support Status**:

<!-- npu="910b" id524 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id524 -->
<!-- npu="A3" id525 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id525 -->
<!-- npu="950" id526 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id526 -->

**Constraints & Notes**: `self` only supports double

</div>

> <font size="3">floor()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.floor](https://pytorch.org/docs/2.7/generated/torch.Tensor.floor.html)

**Support Status**:

<!-- npu="910b" id527 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id527 -->
<!-- npu="A3" id528 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id528 -->
<!-- npu="950" id529 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id529 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">floor_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.floor_](https://pytorch.org/docs/2.7/generated/torch.Tensor.floor_.html)

**Support Status**:

<!-- npu="910b" id530 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id530 -->
<!-- npu="A3" id531 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id531 -->
<!-- npu="950" id532 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id532 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">floor_divide()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.floor_divide](https://pytorch.org/docs/2.7/generated/torch.Tensor.floor_divide.html)

**Support Status**:

<!-- npu="910b" id533 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id533 -->
<!-- npu="A3" id534 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id534 -->
<!-- npu="950" id535 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id535 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">floor_divide_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.floor_divide_](https://pytorch.org/docs/2.7/generated/torch.Tensor.floor_divide_.html)

**Support Status**:

<!-- npu="910b" id536 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id536 -->
<!-- npu="A3" id537 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id537 -->
<!-- npu="950" id538 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id538 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">fmod()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.fmod](https://pytorch.org/docs/2.7/generated/torch.Tensor.fmod.html)

**Support Status**:

<!-- npu="910b" id539 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id539 -->
<!-- npu="A3" id540 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id540 -->
<!-- npu="950" id541 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id541 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int32, int64

</div>

> <font size="3">fmod_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.fmod_](https://pytorch.org/docs/2.7/generated/torch.Tensor.fmod_.html)

**Support Status**:

<!-- npu="910b" id542 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id542 -->
<!-- npu="A3" id543 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id543 -->
<!-- npu="950" id544 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id544 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int32, int64

</div>

> <font size="3">frac()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.frac](https://pytorch.org/docs/2.7/generated/torch.Tensor.frac.html)

**Support Status**:

<!-- npu="910b" id545 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id545 -->
<!-- npu="A3" id546 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id546 -->
<!-- npu="950" id547 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id547 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">frac_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.frac_](https://pytorch.org/docs/2.7/generated/torch.Tensor.frac_.html)

**Support Status**:

<!-- npu="910b" id548 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id548 -->
<!-- npu="A3" id549 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id549 -->
<!-- npu="950" id550 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id550 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">gather()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.gather](https://pytorch.org/docs/2.7/generated/torch.Tensor.gather.html)

**Support Status**:

<!-- npu="910b" id551 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id551 -->
<!-- npu="A3" id552 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id552 -->
<!-- npu="950" id553 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id553 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- The dimension of `index` must be the same as that of `input`

<!-- npu="950,A3,910b" id554 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist
<!-- end id554 -->

</div>

> <font size="3">ge()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ge](https://pytorch.org/docs/2.7/generated/torch.Tensor.ge.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">ge_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ge_](https://pytorch.org/docs/2.7/generated/torch.Tensor.ge_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">greater_equal()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.greater_equal](https://pytorch.org/docs/2.7/generated/torch.Tensor.greater_equal.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">greater_equal_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.greater_equal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.greater_equal_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">geometric_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.geometric_](https://pytorch.org/docs/2.7/generated/torch.Tensor.geometric_.html)

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

> <font size="3">ger()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ger](https://pytorch.org/docs/2.7/generated/torch.Tensor.ger.html)

**Support Status**:

<!-- npu="910b" id570 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id570 -->
<!-- npu="A3" id571 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id571 -->
<!-- npu="950" id572 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id572 -->

</div>

> <font size="3">get_device()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.get_device](https://pytorch.org/docs/2.7/generated/torch.Tensor.get_device.html)

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

</div>

> <font size="3">gt()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.gt](https://pytorch.org/docs/2.7/generated/torch.Tensor.gt.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">gt_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.gt_](https://pytorch.org/docs/2.7/generated/torch.Tensor.gt_.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">greater()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.greater](https://pytorch.org/docs/2.7/generated/torch.Tensor.greater.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">greater_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.greater_](https://pytorch.org/docs/2.7/generated/torch.Tensor.greater_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.half](https://pytorch.org/docs/2.7/generated/torch.Tensor.half.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">hardshrink()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.hardshrink](https://pytorch.org/docs/2.7/generated/torch.Tensor.hardshrink.html)

**Support Status**:

<!-- npu="910b" id591 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id591 -->
<!-- npu="A3" id592 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id592 -->
<!-- npu="950" id593 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id593 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">heaviside()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.heaviside](https://pytorch.org/docs/2.7/generated/torch.Tensor.heaviside.html)

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

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">histc()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.histc](https://pytorch.org/docs/2.7/generated/torch.Tensor.histc.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">hsplit()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.hsplit](https://pytorch.org/docs/2.7/generated/torch.Tensor.hsplit.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">index_add_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_add_](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_add_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, int64, bool

</div>

> <font size="3">index_add()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_add](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_add.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">index_copy_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_copy_](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_copy_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, bool

</div>

> <font size="3">index_copy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_copy](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_copy.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">index_fill_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_fill_](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_fill_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, int32, int64, bool

</div>

> <font size="3">index_fill()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_fill](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_fill.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int32, int64, bool

</div>

> <font size="3">index_put_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_put_](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_put_.html)

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

**Constraints & Notes**:

- `self` only supports int64

<!-- npu="950,A3,910b" id624 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist
<!-- end id624 -->

</div>

> <font size="3">index_put()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_put](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_put.html)

**Support Status**:

<!-- npu="910b" id625 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id625 -->
<!-- npu="A3" id626 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id626 -->
<!-- npu="950" id627 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id627 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

<!-- npu="950,A3,910b" id628 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist
<!-- end id628 -->

</div>

> <font size="3">index_reduce_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_reduce_](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_reduce_.html)

**Support Status**:

<!-- npu="910b" id629 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id629 -->
<!-- npu="A3" id630 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id630 -->
<!-- npu="950" id631 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id631 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">index_reduce()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_reduce](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_reduce.html)

**Support Status**:

<!-- npu="910b" id632 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id632 -->
<!-- npu="A3" id633 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id633 -->
<!-- npu="950" id634 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id634 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">index_select()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.index_select](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_select.html)

**Support Status**:

<!-- npu="910b" id635 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id635 -->
<!-- npu="A3" id636 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id636 -->
<!-- npu="950" id637 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id637 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">indices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.indices](https://pytorch.org/docs/2.7/generated/torch.Tensor.indices.html)

**Support Status**:

<!-- npu="910b" id638 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id638 -->
<!-- npu="A3" id639 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id639 -->
<!-- npu="950" id640 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id640 -->

</div>

> <font size="3">inner()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.inner](https://pytorch.org/docs/2.7/generated/torch.Tensor.inner.html)

**Support Status**:

<!-- npu="910b" id641 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id641 -->
<!-- npu="A3" id642 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id642 -->
<!-- npu="950" id643 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id643 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">int()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.int](https://pytorch.org/docs/2.7/generated/torch.Tensor.int.html)

**Support Status**:

<!-- npu="910b" id644 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id644 -->
<!-- npu="A3" id645 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id645 -->
<!-- npu="950" id646 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id646 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">int_repr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.int_repr](https://pytorch.org/docs/2.7/generated/torch.Tensor.int_repr.html)

**Support Status**:

<!-- npu="910b" id647 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id647 -->
<!-- npu="A3" id648 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id648 -->
<!-- npu="950" id649 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id649 -->

</div>

> <font size="3">isclose()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.isclose](https://pytorch.org/docs/2.7/generated/torch.Tensor.isclose.html)

**Support Status**:

<!-- npu="910b" id650 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id650 -->
<!-- npu="A3" id651 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id651 -->
<!-- npu="950" id652 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id652 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int32, int64, bool

</div>

> <font size="3">isfinite()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.isfinite](https://pytorch.org/docs/2.7/generated/torch.Tensor.isfinite.html)

**Support Status**:

<!-- npu="910b" id653 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id653 -->
<!-- npu="A3" id654 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id654 -->
<!-- npu="950" id655 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id655 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">isinf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.isinf](https://pytorch.org/docs/2.7/generated/torch.Tensor.isinf.html)

**Support Status**:

<!-- npu="910b" id656 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id656 -->
<!-- npu="A3" id657 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id657 -->
<!-- npu="950" id658 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id658 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">isposinf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.isposinf](https://pytorch.org/docs/2.7/generated/torch.Tensor.isposinf.html)

**Support Status**:

<!-- npu="910b" id659 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id659 -->
<!-- npu="A3" id660 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id660 -->
<!-- npu="950" id661 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id661 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">isneginf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.isneginf](https://pytorch.org/docs/2.7/generated/torch.Tensor.isneginf.html)

**Support Status**:

<!-- npu="910b" id662 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id662 -->
<!-- npu="A3" id663 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id663 -->
<!-- npu="950" id664 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id664 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">isnan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.isnan](https://pytorch.org/docs/2.7/generated/torch.Tensor.isnan.html)

**Support Status**:

<!-- npu="910b" id665 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id665 -->
<!-- npu="A3" id666 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id666 -->
<!-- npu="950" id667 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id667 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">is_contiguous()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_contiguous](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_contiguous.html)

**Support Status**:

<!-- npu="910b" id668 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id668 -->
<!-- npu="A3" id669 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id669 -->
<!-- npu="950" id670 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id670 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_complex()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_complex](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_complex.html)

**Support Status**:

<!-- npu="910b" id671 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id671 -->
<!-- npu="A3" id672 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id672 -->
<!-- npu="950" id673 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id673 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">is_conj()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_conj](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_conj.html)

**Support Status**:

<!-- npu="910b" id674 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id674 -->
<!-- npu="A3" id675 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id675 -->
<!-- npu="950" id676 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id676 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">is_floating_point()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_floating_point](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_floating_point.html)

**Support Status**:

<!-- npu="910b" id677 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id677 -->
<!-- npu="A3" id678 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id678 -->
<!-- npu="950" id679 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id679 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_inference()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_inference](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_inference.html)

**Support Status**:

<!-- npu="910b" id680 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id680 -->
<!-- npu="A3" id681 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id681 -->
<!-- npu="950" id682 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id682 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_leaf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_leaf](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_leaf.html)

**Support Status**:

<!-- npu="910b" id683 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id683 -->
<!-- npu="A3" id684 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id684 -->
<!-- npu="950" id685 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id685 -->

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_pinned](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_pinned.html)

**Support Status**:

<!-- npu="910b" id686 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id686 -->
<!-- npu="A3" id687 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id687 -->
<!-- npu="950" id688 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id688 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_set_to()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_set_to](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_set_to.html)

**Support Status**:

<!-- npu="910b" id689 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id689 -->
<!-- npu="A3" id690 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id690 -->
<!-- npu="950" id691 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id691 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_shared()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_shared](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_shared.html)

**Support Status**:

<!-- npu="910b" id692 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id692 -->
<!-- npu="A3" id693 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id693 -->
<!-- npu="950" id694 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id694 -->

</div>

> <font size="3">is_signed()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_signed](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_signed.html)

**Support Status**:

<!-- npu="910b" id695 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id695 -->
<!-- npu="A3" id696 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id696 -->
<!-- npu="950" id697 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id697 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">is_sparse()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.is_sparse](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_sparse.html)

**Support Status**:

<!-- npu="910b" id698 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id698 -->
<!-- npu="A3" id699 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id699 -->
<!-- npu="950" id700 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id700 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">isreal()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.isreal](https://pytorch.org/docs/2.7/generated/torch.Tensor.isreal.html)

**Support Status**:

<!-- npu="910b" id701 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id701 -->
<!-- npu="A3" id702 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id702 -->
<!-- npu="950" id703 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id703 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">item()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.item](https://pytorch.org/docs/2.7/generated/torch.Tensor.item.html)

**Support Status**:

<!-- npu="910b" id704 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id704 -->
<!-- npu="A3" id705 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id705 -->
<!-- npu="950" id706 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id706 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">kthvalue()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.kthvalue](https://pytorch.org/docs/2.7/generated/torch.Tensor.kthvalue.html)

**Support Status**:

<!-- npu="910b" id707 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id707 -->
<!-- npu="A3" id708 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id708 -->
<!-- npu="950" id709 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id709 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int32

</div>

> <font size="3">ldexp()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ldexp](https://pytorch.org/docs/2.7/generated/torch.Tensor.ldexp.html)

**Support Status**:

<!-- npu="910b" id710 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id710 -->
<!-- npu="A3" id711 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id711 -->
<!-- npu="950" id712 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id712 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">ldexp_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ldexp_](https://pytorch.org/docs/2.7/generated/torch.Tensor.ldexp_.html)

**Support Status**:

<!-- npu="910b" id713 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id713 -->
<!-- npu="A3" id714 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id714 -->
<!-- npu="950" id715 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id715 -->

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">le()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.le](https://pytorch.org/docs/2.7/generated/torch.Tensor.le.html)

**Support Status**:

<!-- npu="910b" id716 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id716 -->
<!-- npu="A3" id717 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id717 -->
<!-- npu="950" id718 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id718 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">le_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.le_](https://pytorch.org/docs/2.7/generated/torch.Tensor.le_.html)

**Support Status**:

<!-- npu="910b" id719 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id719 -->
<!-- npu="A3" id720 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id720 -->
<!-- npu="950" id721 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id721 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">less_equal()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.less_equal](https://pytorch.org/docs/2.7/generated/torch.Tensor.less_equal.html)

**Support Status**:

<!-- npu="910b" id722 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id722 -->
<!-- npu="A3" id723 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id723 -->
<!-- npu="950" id724 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id724 -->

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">less_equal_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.less_equal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.less_equal_.html)

**Support Status**:

<!-- npu="910b" id725 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id725 -->
<!-- npu="A3" id726 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id726 -->
<!-- npu="950" id727 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id727 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">lerp()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.lerp](https://pytorch.org/docs/2.7/generated/torch.Tensor.lerp.html)

**Support Status**:

<!-- npu="910b" id728 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id728 -->
<!-- npu="A3" id729 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id729 -->
<!-- npu="950" id730 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id730 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">lerp_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.lerp_](https://pytorch.org/docs/2.7/generated/torch.Tensor.lerp_.html)

**Support Status**:

<!-- npu="910b" id731 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id731 -->
<!-- npu="A3" id732 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id732 -->
<!-- npu="950" id733 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id733 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">log()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.log](https://pytorch.org/docs/2.7/generated/torch.Tensor.log.html)

**Support Status**:

<!-- npu="910b" id734 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id734 -->
<!-- npu="A3" id735 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id735 -->
<!-- npu="950" id736 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id736 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">log_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.log_](https://pytorch.org/docs/2.7/generated/torch.Tensor.log_.html)

**Support Status**:

<!-- npu="910b" id737 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id737 -->
<!-- npu="A3" id738 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id738 -->
<!-- npu="950" id739 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id739 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, complex64, complex128

</div>

> <font size="3">log10()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.log10](https://pytorch.org/docs/2.7/generated/torch.Tensor.log10.html)

**Support Status**:

<!-- npu="910b" id740 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id740 -->
<!-- npu="A3" id741 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id741 -->
<!-- npu="950" id742 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id742 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">log10_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.log10_](https://pytorch.org/docs/2.7/generated/torch.Tensor.log10_.html)

**Support Status**:

<!-- npu="910b" id743 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id743 -->
<!-- npu="A3" id744 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id744 -->
<!-- npu="950" id745 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id745 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">log1p()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.log1p](https://pytorch.org/docs/2.7/generated/torch.Tensor.log1p.html)

**Support Status**:

<!-- npu="910b" id746 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id746 -->
<!-- npu="A3" id747 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id747 -->
<!-- npu="950" id748 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id748 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">log1p_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.log1p_](https://pytorch.org/docs/2.7/generated/torch.Tensor.log1p_.html)

**Support Status**:

<!-- npu="910b" id749 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id749 -->
<!-- npu="A3" id750 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id750 -->
<!-- npu="950" id751 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id751 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">log2()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.log2](https://pytorch.org/docs/2.7/generated/torch.Tensor.log2.html)

**Support Status**:

<!-- npu="910b" id752 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id752 -->
<!-- npu="A3" id753 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id753 -->
<!-- npu="950" id754 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id754 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">log2_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.log2_](https://pytorch.org/docs/2.7/generated/torch.Tensor.log2_.html)

**Support Status**:

<!-- npu="910b" id755 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id755 -->
<!-- npu="A3" id756 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id756 -->
<!-- npu="950" id757 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id757 -->

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">logaddexp()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logaddexp](https://pytorch.org/docs/2.7/generated/torch.Tensor.logaddexp.html)

**Support Status**:

<!-- npu="910b" id758 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id758 -->
<!-- npu="A3" id759 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id759 -->
<!-- npu="950" id760 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id760 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, int64, bool

</div>

> <font size="3">logaddexp2()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logaddexp2](https://pytorch.org/docs/2.7/generated/torch.Tensor.logaddexp2.html)

**Support Status**:

<!-- npu="910b" id761 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id761 -->
<!-- npu="A3" id762 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id762 -->
<!-- npu="950" id763 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id763 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">logsumexp()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logsumexp](https://pytorch.org/docs/2.7/generated/torch.Tensor.logsumexp.html)

**Support Status**:

<!-- npu="910b" id764 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id764 -->
<!-- npu="A3" id765 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id765 -->
<!-- npu="950" id766 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id766 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">logical_and()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logical_and](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_and.html)

**Support Status**:

<!-- npu="910b" id767 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id767 -->
<!-- npu="A3" id768 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id768 -->
<!-- npu="950" id769 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id769 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">logical_and_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logical_and_](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_and_.html)

**Support Status**:

<!-- npu="910b" id770 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id770 -->
<!-- npu="A3" id771 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id771 -->
<!-- npu="950" id772 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id772 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">logical_not()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logical_not](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_not.html)

**Support Status**:

<!-- npu="910b" id773 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id773 -->
<!-- npu="A3" id774 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id774 -->
<!-- npu="950" id775 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id775 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">logical_not_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logical_not_](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_not_.html)

**Support Status**:

<!-- npu="910b" id776 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id776 -->
<!-- npu="A3" id777 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id777 -->
<!-- npu="950" id778 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id778 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">logical_or()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logical_or](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_or.html)

**Support Status**:

<!-- npu="910b" id779 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id779 -->
<!-- npu="A3" id780 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id780 -->
<!-- npu="950" id781 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id781 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">logical_or_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logical_or_](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_or_.html)

**Support Status**:

<!-- npu="910b" id782 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id782 -->
<!-- npu="A3" id783 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id783 -->
<!-- npu="950" id784 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id784 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">logical_xor()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logical_xor](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_xor.html)

**Support Status**:

<!-- npu="910b" id785 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id785 -->
<!-- npu="A3" id786 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id786 -->
<!-- npu="950" id787 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id787 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- May fall back to CPU execution

</div>

> <font size="3">logical_xor_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logical_xor_](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_xor_.html)

**Support Status**:

<!-- npu="910b" id788 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id788 -->
<!-- npu="A3" id789 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id789 -->
<!-- npu="950" id790 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id790 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">logit()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logit](https://pytorch.org/docs/2.7/generated/torch.Tensor.logit.html)

**Support Status**:

<!-- npu="910b" id791 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id791 -->
<!-- npu="A3" id792 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id792 -->
<!-- npu="950" id793 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id793 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- When the value of `eps` is greater than 1, the output is nan; when the value of `eps` is 1, the output is inf

</div>

> <font size="3">logit_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.logit_](https://pytorch.org/docs/2.7/generated/torch.Tensor.logit_.html)

**Support Status**:

<!-- npu="910b" id794 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id794 -->
<!-- npu="A3" id795 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id795 -->
<!-- npu="950" id796 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id796 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- When the value of `eps` is greater than 1, the output is nan; when the value of `eps` is 1, the output is inf

</div>

> <font size="3">long()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.long](https://pytorch.org/docs/2.7/generated/torch.Tensor.long.html)

**Support Status**:

<!-- npu="910b" id797 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id797 -->
<!-- npu="A3" id798 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id798 -->
<!-- npu="950" id799 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id799 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">lt()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.lt](https://pytorch.org/docs/2.7/generated/torch.Tensor.lt.html)

**Support Status**:

<!-- npu="910b" id800 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id800 -->
<!-- npu="A3" id801 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id801 -->
<!-- npu="950" id802 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id802 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">lt_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.lt_](https://pytorch.org/docs/2.7/generated/torch.Tensor.lt_.html)

**Support Status**:

<!-- npu="910b" id803 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id803 -->
<!-- npu="A3" id804 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id804 -->
<!-- npu="950" id805 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id805 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">less()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.less](https://pytorch.org/docs/2.7/generated/torch.Tensor.less.html)

**Support Status**:

<!-- npu="910b" id806 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id806 -->
<!-- npu="A3" id807 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id807 -->
<!-- npu="950" id808 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id808 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">less_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.less_](https://pytorch.org/docs/2.7/generated/torch.Tensor.less_.html)

**Support Status**:

<!-- npu="910b" id809 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id809 -->
<!-- npu="A3" id810 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id810 -->
<!-- npu="950" id811 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id811 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">as_subclass()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.as_subclass](https://pytorch.org/docs/2.7/generated/torch.Tensor.as_subclass.html)

**Support Status**:

<!-- npu="910b" id812 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id812 -->
<!-- npu="A3" id813 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id813 -->
<!-- npu="950" id814 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id814 -->

</div>

> <font size="3">map_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.map_](https://pytorch.org/docs/2.7/generated/torch.Tensor.map_.html)

**Support Status**:

<!-- npu="910b" id815 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id815 -->
<!-- npu="A3" id816 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id816 -->
<!-- npu="950" id817 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id817 -->

**Constraints & Notes**: CPU-only

</div>

> <font size="3">masked_scatter_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.masked_scatter_](https://pytorch.org/docs/2.7/generated/torch.Tensor.masked_scatter_.html)

**Support Status**:

<!-- npu="910b" id818 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id818 -->
<!-- npu="A3" id819 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id819 -->
<!-- npu="950" id820 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id820 -->

**Constraints & Notes**: `self` only supports fp32, int64, bool

</div>

> <font size="3">masked_scatter()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.masked_scatter](https://pytorch.org/docs/2.7/generated/torch.Tensor.masked_scatter.html)

**Support Status**:

<!-- npu="910b" id821 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id821 -->
<!-- npu="A3" id822 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id822 -->
<!-- npu="950" id823 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id823 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">masked_fill_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.masked_fill_](https://pytorch.org/docs/2.7/generated/torch.Tensor.masked_fill_.html)

**Support Status**:

<!-- npu="910b" id824 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id824 -->
<!-- npu="A3" id825 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id825 -->
<!-- npu="950" id826 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id826 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int8, int32, int64, bool

</div>

> <font size="3">masked_fill()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.masked_fill](https://pytorch.org/docs/2.7/generated/torch.Tensor.masked_fill.html)

**Support Status**:

<!-- npu="910b" id827 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id827 -->
<!-- npu="A3" id828 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id828 -->
<!-- npu="950" id829 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id829 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int8, int32, int64, bool

</div>

> <font size="3">masked_select()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.masked_select](https://pytorch.org/docs/2.7/generated/torch.Tensor.masked_select.html)

**Support Status**:

<!-- npu="910b" id830 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id830 -->
<!-- npu="A3" id831 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id831 -->
<!-- npu="950" id832 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id832 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">matmul()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.matmul](https://pytorch.org/docs/2.7/generated/torch.Tensor.matmul.html)

**Support Status**:

<!-- npu="910b" id833 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id833 -->
<!-- npu="A3" id834 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id834 -->
<!-- npu="950" id835 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id835 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- Named Tensor is supported

</div>

> <font size="3">matrix_power()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.matrix_power](https://pytorch.org/docs/2.7/generated/torch.Tensor.matrix_power.html)

**Support Status**:

<!-- npu="910b" id836 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id836 -->
<!-- npu="A3" id837 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id837 -->
<!-- npu="950" id838 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id838 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">max()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.max](https://pytorch.org/docs/2.7/generated/torch.Tensor.max.html)

**Support Status**:

<!-- npu="910b" id839 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id839 -->
<!-- npu="A3" id840 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id840 -->
<!-- npu="950" id841 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id841 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int64, bool

</div>

> <font size="3">maximum()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.maximum](https://pytorch.org/docs/2.7/generated/torch.Tensor.maximum.html)

**Support Status**:

<!-- npu="910b" id842 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id842 -->
<!-- npu="A3" id843 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id843 -->
<!-- npu="950" id844 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id844 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.mean](https://pytorch.org/docs/2.7/generated/torch.Tensor.mean.html)

**Support Status**:

<!-- npu="910b" id845 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id845 -->
<!-- npu="A3" id846 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id846 -->
<!-- npu="950" id847 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id847 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">nanmean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.nanmean](https://pytorch.org/docs/2.7/generated/torch.Tensor.nanmean.html)

**Support Status**:

<!-- npu="910b" id848 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id848 -->
<!-- npu="A3" id849 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id849 -->
<!-- npu="950" id850 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id850 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">median()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.median](https://pytorch.org/docs/2.7/generated/torch.Tensor.median.html)

**Support Status**:

<!-- npu="910b" id851 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id851 -->
<!-- npu="A3" id852 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id852 -->
<!-- npu="950" id853 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id853 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64
- When `input` is bf16, `dim` cannot be a dimension whose axis value is 1 in `input`

</div>

> <font size="3">min()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.min](https://pytorch.org/docs/2.7/generated/torch.Tensor.min.html)

**Support Status**:

<!-- npu="910b" id854 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id854 -->
<!-- npu="A3" id855 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id855 -->
<!-- npu="950" id856 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id856 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int64, bool

</div>

> <font size="3">minimum()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.minimum](https://pytorch.org/docs/2.7/generated/torch.Tensor.minimum.html)

**Support Status**:

<!-- npu="910b" id857 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id857 -->
<!-- npu="A3" id858 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id858 -->
<!-- npu="950" id859 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id859 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">mm()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.mm](https://pytorch.org/docs/2.7/generated/torch.Tensor.mm.html)

**Support Status**:

<!-- npu="910b" id860 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id860 -->
<!-- npu="A3" id861 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id861 -->
<!-- npu="950" id862 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id862 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32

</div>

> <font size="3">smm()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.smm](https://pytorch.org/docs/2.7/generated/torch.Tensor.smm.html)

**Support Status**:

<!-- npu="910b" id863 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id863 -->
<!-- npu="A3" id864 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id864 -->
<!-- npu="950" id865 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id865 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.mode](https://pytorch.org/docs/2.7/generated/torch.Tensor.mode.html)

**Support Status**:

<!-- npu="910b" id866 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id866 -->
<!-- npu="A3" id867 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id867 -->
<!-- npu="950" id868 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id868 -->

</div>

> <font size="3">movedim()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.movedim](https://pytorch.org/docs/2.7/generated/torch.Tensor.movedim.html)

**Support Status**:

<!-- npu="910b" id869 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id869 -->
<!-- npu="A3" id870 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id870 -->
<!-- npu="950" id871 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id871 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">moveaxis()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.moveaxis](https://pytorch.org/docs/2.7/generated/torch.Tensor.moveaxis.html)

**Support Status**:

<!-- npu="910b" id872 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id872 -->
<!-- npu="A3" id873 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id873 -->
<!-- npu="950" id874 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id874 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">msort()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.msort](https://pytorch.org/docs/2.7/generated/torch.Tensor.msort.html)

**Support Status**:

<!-- npu="910b" id875 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id875 -->
<!-- npu="A3" id876 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id876 -->
<!-- npu="950" id877 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id877 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">mul()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.mul](https://pytorch.org/docs/2.7/generated/torch.Tensor.mul.html)

**Support Status**:

<!-- npu="910b" id878 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id878 -->
<!-- npu="A3" id879 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id879 -->
<!-- npu="950" id880 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id880 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">mul_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.mul_](https://pytorch.org/docs/2.7/generated/torch.Tensor.mul_.html)

**Support Status**:

<!-- npu="910b" id881 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id881 -->
<!-- npu="A3" id882 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id882 -->
<!-- npu="950" id883 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id883 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">multiply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.multiply](https://pytorch.org/docs/2.7/generated/torch.Tensor.multiply.html)

**Support Status**:

<!-- npu="910b" id884 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id884 -->
<!-- npu="A3" id885 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id885 -->
<!-- npu="950" id886 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id886 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">multiply_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.multiply_](https://pytorch.org/docs/2.7/generated/torch.Tensor.multiply_.html)

**Support Status**:

<!-- npu="910b" id887 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id887 -->
<!-- npu="A3" id888 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id888 -->
<!-- npu="950" id889 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id889 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">multinomial()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.multinomial](https://pytorch.org/docs/2.7/generated/torch.Tensor.multinomial.html)

**Support Status**:

<!-- npu="910b" id890 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id890 -->
<!-- npu="A3" id891 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id891 -->
<!-- npu="950" id892 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id892 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">nansum()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.nansum](https://pytorch.org/docs/2.7/generated/torch.Tensor.nansum.html)

**Support Status**:

<!-- npu="910b" id893 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id893 -->
<!-- npu="A3" id894 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id894 -->
<!-- npu="950" id895 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id895 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

<!-- npu="950" id896 -->
- <term>Ascend 950DT products</term>: Not Supported for uint8, int8, int16, int32, int64, bool
<!-- end id896 -->

</div>

> <font size="3">narrow()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.narrow](https://pytorch.org/docs/2.7/generated/torch.Tensor.narrow.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">narrow_copy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.narrow_copy](https://pytorch.org/docs/2.7/generated/torch.Tensor.narrow_copy.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">ndimension()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ndimension](https://pytorch.org/docs/2.7/generated/torch.Tensor.ndimension.html)

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

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">nan_to_num()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.nan_to_num](https://pytorch.org/docs/2.7/generated/torch.Tensor.nan_to_num.html)

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

<!-- npu="950" id1388 -->
**Constraints & Notes**: <term>Ascend 950DT products</term>: do not support fp64, complex64, or complex128
<!-- end id1388 -->

</div>

> <font size="3">nan_to_num_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.nan_to_num_](https://pytorch.org/docs/2.7/generated/torch.Tensor.nan_to_num_.html)

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

> <font size="3">ne()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ne](https://pytorch.org/docs/2.7/generated/torch.Tensor.ne.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">ne_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ne_](https://pytorch.org/docs/2.7/generated/torch.Tensor.ne_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">nextafter_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.nextafter_](https://pytorch.org/docs/2.7/generated/torch.Tensor.nextafter_.html)

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

**Constraints & Notes**: Falls back to CPU execution

</div>

> <font size="3">not_equal()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.not_equal](https://pytorch.org/docs/2.7/generated/torch.Tensor.not_equal.html)

**Support Status**:

<!-- npu="910b" id921 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id921 -->
<!-- npu="A3" id922 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id922 -->
<!-- npu="950" id923 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id923 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- May fall back to CPU execution

</div>

> <font size="3">not_equal_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.not_equal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.not_equal_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">neg()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.neg](https://pytorch.org/docs/2.7/generated/torch.Tensor.neg.html)

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

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, int8, int32, int64

</div>

> <font size="3">neg_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.neg_](https://pytorch.org/docs/2.7/generated/torch.Tensor.neg_.html)

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

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128
- May fall back to CPU execution

</div>

> <font size="3">negative()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.negative](https://pytorch.org/docs/2.7/generated/torch.Tensor.negative.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, int8, int32, int64, complex64, complex128

</div>

> <font size="3">negative_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.negative_](https://pytorch.org/docs/2.7/generated/torch.Tensor.negative_.html)

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

**Constraints & Notes**: `self` only supports fp16, fp32, int8, int32, int64, complex64, complex128

</div>

> <font size="3">nelement()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.nelement](https://pytorch.org/docs/2.7/generated/torch.Tensor.nelement.html)

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

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">nonzero()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.nonzero](https://pytorch.org/docs/2.7/generated/torch.Tensor.nonzero.html)

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

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- The nan scenario is not supported

</div>

> <font size="3">norm()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.norm](https://pytorch.org/docs/2.7/generated/torch.Tensor.norm.html)

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

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64

<!-- npu="950" id948 -->
- <term>Ascend 950DT products</term>: Not Supported for fp64
<!-- end id948 -->

</div>

> <font size="3">normal_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.normal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.normal_.html)

**Support Status**:

<!-- npu="910b" id949 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id949 -->
<!-- npu="A3" id950 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id950 -->
<!-- npu="950" id951 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id951 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- May fall back to CPU execution

</div>

> <font size="3">numel()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.numel](https://pytorch.org/docs/2.7/generated/torch.Tensor.numel.html)

**Support Status**:

<!-- npu="910b" id952 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id952 -->
<!-- npu="A3" id953 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id953 -->
<!-- npu="950" id954 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id954 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">numpy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.numpy](https://pytorch.org/docs/2.7/generated/torch.Tensor.numpy.html)

**Support Status**:

<!-- npu="910b" id955 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id955 -->
<!-- npu="A3" id956 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id956 -->
<!-- npu="950" id957 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id957 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">outer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.outer](https://pytorch.org/docs/2.7/generated/torch.Tensor.outer.html)

**Support Status**:

<!-- npu="910b" id958 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id958 -->
<!-- npu="A3" id959 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id959 -->
<!-- npu="950" id960 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id960 -->

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">permute()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.permute](https://pytorch.org/docs/2.7/generated/torch.Tensor.permute.html)

**Support Status**:

<!-- npu="910b" id961 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id961 -->
<!-- npu="A3" id962 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id962 -->
<!-- npu="950" id963 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id963 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">positive()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.positive](https://pytorch.org/docs/2.7/generated/torch.Tensor.positive.html)

**Support Status**:

<!-- npu="910b" id964 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id964 -->
<!-- npu="A3" id965 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id965 -->
<!-- npu="950" id966 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id966 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128

</div>

> <font size="3">pow()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.pow](https://pytorch.org/docs/2.7/generated/torch.Tensor.pow.html)

**Support Status**:

<!-- npu="910b" id967 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id967 -->
<!-- npu="A3" id968 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id968 -->
<!-- npu="950" id969 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id969 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">pow_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.pow_](https://pytorch.org/docs/2.7/generated/torch.Tensor.pow_.html)

**Support Status**:

<!-- npu="910b" id970 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id970 -->
<!-- npu="A3" id971 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id971 -->
<!-- npu="950" id972 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id972 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, int64

</div>

> <font size="3">prod()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.prod](https://pytorch.org/docs/2.7/generated/torch.Tensor.prod.html)

**Support Status**:

<!-- npu="910b" id973 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id973 -->
<!-- npu="A3" id974 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id974 -->
<!-- npu="950" id975 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id975 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">put_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.put_](https://pytorch.org/docs/2.7/generated/torch.Tensor.put_.html)

**Support Status**:

<!-- npu="910b" id976 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id976 -->
<!-- npu="A3" id977 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id977 -->
<!-- npu="950" id978 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id978 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128

</div>

> <font size="3">qscheme()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.qscheme](https://pytorch.org/docs/2.7/generated/torch.Tensor.qscheme.html)

**Support Status**:

<!-- npu="910b" id979 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id979 -->
<!-- npu="A3" id980 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id980 -->
<!-- npu="950" id981 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id981 -->

</div>

> <font size="3">quantile()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.quantile](https://pytorch.org/docs/2.7/generated/torch.Tensor.quantile.html)

**Support Status**:

<!-- npu="910b" id982 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id982 -->
<!-- npu="A3" id983 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id983 -->
<!-- npu="950" id984 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id984 -->

</div>

> <font size="3">rad2deg()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.rad2deg](https://pytorch.org/docs/2.7/generated/torch.Tensor.rad2deg.html)

**Support Status**:

<!-- npu="910b" id985 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id985 -->
<!-- npu="A3" id986 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id986 -->
<!-- npu="950" id987 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id987 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">random_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.random_](https://pytorch.org/docs/2.7/generated/torch.Tensor.random_.html)

**Support Status**:

<!-- npu="910b" id988 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id988 -->
<!-- npu="A3" id989 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id989 -->
<!-- npu="950" id990 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id990 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">ravel()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.ravel](https://pytorch.org/docs/2.7/generated/torch.Tensor.ravel.html)

**Support Status**:

<!-- npu="910b" id991 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id991 -->
<!-- npu="A3" id992 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id992 -->
<!-- npu="950" id993 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id993 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">reciprocal()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.reciprocal](https://pytorch.org/docs/2.7/generated/torch.Tensor.reciprocal.html)

**Support Status**:

<!-- npu="910b" id994 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id994 -->
<!-- npu="A3" id995 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id995 -->
<!-- npu="950" id996 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id996 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">reciprocal_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.reciprocal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.reciprocal_.html)

**Support Status**:

<!-- npu="910b" id997 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id997 -->
<!-- npu="A3" id998 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id998 -->
<!-- npu="950" id999 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id999 -->

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">record_stream()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.record_stream](https://pytorch.org/docs/2.7/generated/torch.Tensor.record_stream.html)

**Support Status**:

<!-- npu="910b" id1000 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1000 -->
<!-- npu="A3" id1001 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1001 -->
<!-- npu="950" id1002 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1002 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.register_hook](https://pytorch.org/docs/2.7/generated/torch.Tensor.register_hook.html)

**Support Status**:

<!-- npu="910b" id1003 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1003 -->
<!-- npu="A3" id1004 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1004 -->
<!-- npu="950" id1005 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1005 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">register_post_accumulate_grad_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.register_post_accumulate_grad_hook](https://pytorch.org/docs/2.7/generated/torch.Tensor.register_post_accumulate_grad_hook.html)

**Support Status**:

<!-- npu="910b" id1006 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1006 -->
<!-- npu="A3" id1007 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1007 -->
<!-- npu="950" id1008 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1008 -->

</div>

> <font size="3">remainder()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.remainder](https://pytorch.org/docs/2.7/generated/torch.Tensor.remainder.html)

**Support Status**:

<!-- npu="910b" id1009 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1009 -->
<!-- npu="A3" id1010 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1010 -->
<!-- npu="950" id1011 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1011 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, int32, int64

</div>

> <font size="3">remainder_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.remainder_](https://pytorch.org/docs/2.7/generated/torch.Tensor.remainder_.html)

**Support Status**:

<!-- npu="910b" id1012 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1012 -->
<!-- npu="A3" id1013 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1013 -->
<!-- npu="950" id1014 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1014 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int32, int64

</div>

> <font size="3">repeat()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.repeat](https://pytorch.org/docs/2.7/generated/torch.Tensor.repeat.html)

**Support Status**:

<!-- npu="910b" id1015 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1015 -->
<!-- npu="A3" id1016 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1016 -->
<!-- npu="950" id1017 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1017 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">repeat_interleave()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.repeat_interleave](https://pytorch.org/docs/2.7/generated/torch.Tensor.repeat_interleave.html)

**Support Status**:

<!-- npu="910b" id1018 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1018 -->
<!-- npu="A3" id1019 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1019 -->
<!-- npu="950" id1020 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1020 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool
- After the input tensor is repeated to obtain the output, the number of elements in the output must be less than $2^{22}$

</div>

> <font size="3">requires_grad()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.requires_grad](https://pytorch.org/docs/2.7/generated/torch.Tensor.requires_grad.html)

**Support Status**:

<!-- npu="910b" id1021 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1021 -->
<!-- npu="A3" id1022 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1022 -->
<!-- npu="950" id1023 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1023 -->

</div>

> <font size="3">requires_grad_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.requires_grad_](https://pytorch.org/docs/2.7/generated/torch.Tensor.requires_grad_.html)

**Support Status**:

<!-- npu="910b" id1024 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1024 -->
<!-- npu="A3" id1025 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1025 -->
<!-- npu="950" id1026 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1026 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">reshape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.reshape](https://pytorch.org/docs/2.7/generated/torch.Tensor.reshape.html)

**Support Status**:

<!-- npu="910b" id1027 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1027 -->
<!-- npu="A3" id1028 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1028 -->
<!-- npu="950" id1029 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1029 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">reshape_as()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.reshape_as](https://pytorch.org/docs/2.7/generated/torch.Tensor.reshape_as.html)

**Support Status**:

<!-- npu="910b" id1030 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1030 -->
<!-- npu="A3" id1031 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1031 -->
<!-- npu="950" id1032 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1032 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">resize_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.resize_](https://pytorch.org/docs/2.7/generated/torch.Tensor.resize_.html)

**Support Status**:

<!-- npu="910b" id1033 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1033 -->
<!-- npu="A3" id1034 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1034 -->
<!-- npu="950" id1035 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1035 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- `memory_format` only supports torch.contiguous_format and torch.preserve_format

</div>

> <font size="3">resize_as_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.resize_as_](https://pytorch.org/docs/2.7/generated/torch.Tensor.resize_as_.html)

**Support Status**:

<!-- npu="910b" id1036 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1036 -->
<!-- npu="A3" id1037 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1037 -->
<!-- npu="950" id1038 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1038 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- `memory_format` only supports torch.contiguous_format and torch.preserve_format

</div>

> <font size="3">retain_grad()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.retain_grad](https://pytorch.org/docs/2.7/generated/torch.Tensor.retain_grad.html)

**Support Status**:

<!-- npu="910b" id1039 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1039 -->
<!-- npu="A3" id1040 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1040 -->
<!-- npu="950" id1041 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1041 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">retains_grad()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.retains_grad](https://pytorch.org/docs/2.7/generated/torch.Tensor.retains_grad.html)

**Support Status**:

<!-- npu="910b" id1042 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1042 -->
<!-- npu="A3" id1043 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1043 -->
<!-- npu="950" id1044 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1044 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">roll()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.roll](https://pytorch.org/docs/2.7/generated/torch.Tensor.roll.html)

**Support Status**:

<!-- npu="910b" id1045 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1045 -->
<!-- npu="A3" id1046 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1046 -->
<!-- npu="950" id1047 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1047 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int32, int64, bool

</div>

> <font size="3">rot90()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.rot90](https://pytorch.org/docs/2.7/generated/torch.Tensor.rot90.html)

**Support Status**:

<!-- npu="910b" id1048 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1048 -->
<!-- npu="A3" id1049 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1049 -->
<!-- npu="950" id1050 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1050 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">round()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.round](https://pytorch.org/docs/2.7/generated/torch.Tensor.round.html)

**Support Status**:

<!-- npu="910b" id1051 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1051 -->
<!-- npu="A3" id1052 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1052 -->
<!-- npu="950" id1053 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1053 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">round_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.round_](https://pytorch.org/docs/2.7/generated/torch.Tensor.round_.html)

**Support Status**:

<!-- npu="910b" id1054 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1054 -->
<!-- npu="A3" id1055 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1055 -->
<!-- npu="950" id1056 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1056 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">rsqrt()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.rsqrt](https://pytorch.org/docs/2.7/generated/torch.Tensor.rsqrt.html)

**Support Status**:

<!-- npu="910b" id1057 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1057 -->
<!-- npu="A3" id1058 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1058 -->
<!-- npu="950" id1059 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1059 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">rsqrt_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.rsqrt_](https://pytorch.org/docs/2.7/generated/torch.Tensor.rsqrt_.html)

**Support Status**:

<!-- npu="910b" id1060 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1060 -->
<!-- npu="A3" id1061 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1061 -->
<!-- npu="950" id1062 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1062 -->

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">scatter()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.scatter](https://pytorch.org/docs/2.7/generated/torch.Tensor.scatter.html)

**Support Status**:

<!-- npu="910b" id1063 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1063 -->
<!-- npu="A3" id1064 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1064 -->
<!-- npu="950" id1065 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1065 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32, fp64, int8, int16, int32, int64, bool

<!-- npu="950,A3,910b" id1066 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist
<!-- end id1066 -->

</div>

> <font size="3">scatter_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.scatter_](https://pytorch.org/docs/2.7/generated/torch.Tensor.scatter_.html)

**Support Status**:

<!-- npu="910b" id1067 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1067 -->
<!-- npu="A3" id1068 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1068 -->
<!-- npu="950" id1069 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1069 -->

**Constraints & Notes**:

- The `tensor`, `index`, and `src` parameters cannot be empty or scalars
- May fall back to CPU execution

<!-- npu="950,A3,910b" id1070 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist
<!-- end id1070 -->

</div>

> <font size="3">scatter_add_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.scatter_add_](https://pytorch.org/docs/2.7/generated/torch.Tensor.scatter_add_.html)

**Support Status**:

<!-- npu="910b" id1071 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1071 -->
<!-- npu="A3" id1072 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1072 -->
<!-- npu="950" id1073 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1073 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

<!-- npu="950,A3,910b" id1074 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist
<!-- end id1074 -->

</div>

> <font size="3">scatter_add()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.scatter_add](https://pytorch.org/docs/2.7/generated/torch.Tensor.scatter_add.html)

**Support Status**:

<!-- npu="910b" id1075 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1075 -->
<!-- npu="A3" id1076 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1076 -->
<!-- npu="950" id1077 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1077 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

<!-- npu="950,A3,910b" id1078 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist
<!-- end id1078 -->

</div>

> <font size="3">scatter_reduce()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.scatter_reduce](https://pytorch.org/docs/2.7/generated/torch.Tensor.scatter_reduce.html)

**Support Status**:

<!-- npu="910b" id1079 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1079 -->
<!-- npu="A3" id1080 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1080 -->
<!-- npu="950" id1081 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1081 -->

**Constraints & Notes**:

<!-- npu="A3,910b" id1082 -->
- NPU acceleration is enabled only on <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when all of the following conditions are met:
  - `including_self` is True.
  - `reduce` is "sum" or "add".
  - The input data type is fp32.
<!-- end id1082 -->
<!-- npu="950,A3,910b" id1083 -->
- Configurations of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> other than the scenarios above, as well as all <term>Ascend 950DT products</term> scenarios, fall back to CPU execution (Fallback).
<!-- end id1083 -->

</div>

> <font size="3">select()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.select](https://pytorch.org/docs/2.7/generated/torch.Tensor.select.html)

**Support Status**:

<!-- npu="910b" id1084 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1084 -->
<!-- npu="A3" id1085 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1085 -->
<!-- npu="950" id1086 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1086 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">select_scatter()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.select_scatter](https://pytorch.org/docs/2.7/generated/torch.Tensor.select_scatter.html)

**Support Status**:

<!-- npu="910b" id1087 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1087 -->
<!-- npu="A3" id1088 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1088 -->
<!-- npu="950" id1089 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1089 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

> <font size="3">set_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.set_](https://pytorch.org/docs/2.7/generated/torch.Tensor.set_.html)

**Support Status**:

<!-- npu="910b" id1090 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1090 -->
<!-- npu="A3" id1091 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1091 -->
<!-- npu="950" id1092 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1092 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">share_memory_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.share_memory_](https://pytorch.org/docs/2.7/generated/torch.Tensor.share_memory_.html)

**Support Status**:

<!-- npu="910b" id1093 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1093 -->
<!-- npu="A3" id1094 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1094 -->
<!-- npu="950" id1095 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1095 -->

</div>

> <font size="3">short()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.short](https://pytorch.org/docs/2.7/generated/torch.Tensor.short.html)

**Support Status**:

<!-- npu="910b" id1096 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1096 -->
<!-- npu="A3" id1097 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1097 -->
<!-- npu="950" id1098 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1098 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">sigmoid()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sigmoid](https://pytorch.org/docs/2.7/generated/torch.Tensor.sigmoid.html)

**Support Status**:

<!-- npu="910b" id1099 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1099 -->
<!-- npu="A3" id1100 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1100 -->
<!-- npu="950" id1101 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1101 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">sigmoid_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sigmoid_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sigmoid_.html)

**Support Status**:

<!-- npu="910b" id1102 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1102 -->
<!-- npu="A3" id1103 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1103 -->
<!-- npu="950" id1104 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1104 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">sign()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sign](https://pytorch.org/docs/2.7/generated/torch.Tensor.sign.html)

**Support Status**:

<!-- npu="910b" id1105 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1105 -->
<!-- npu="A3" id1106 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1106 -->
<!-- npu="950" id1107 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1107 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, int32, int64, bool

</div>

> <font size="3">sign_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sign_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sign_.html)

**Support Status**:

<!-- npu="910b" id1108 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1108 -->
<!-- npu="A3" id1109 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1109 -->
<!-- npu="950" id1110 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1110 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int32, int64, bool

</div>

> <font size="3">sgn()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sgn](https://pytorch.org/docs/2.7/generated/torch.Tensor.sgn.html)

**Support Status**:

<!-- npu="910b" id1111 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1111 -->
<!-- npu="A3" id1112 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1112 -->
<!-- npu="950" id1113 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1113 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int32, int64, bool, complex64, complex128

</div>

> <font size="3">sgn_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sgn_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sgn_.html)

**Support Status**:

<!-- npu="910b" id1114 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1114 -->
<!-- npu="A3" id1115 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1115 -->
<!-- npu="950" id1116 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1116 -->

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, int32, int64, bool

</div>

> <font size="3">sin()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sin](https://pytorch.org/docs/2.7/generated/torch.Tensor.sin.html)

**Support Status**:

<!-- npu="910b" id1117 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1117 -->
<!-- npu="A3" id1118 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1118 -->
<!-- npu="950" id1119 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1119 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">sin_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sin_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sin_.html)

**Support Status**:

<!-- npu="910b" id1120 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1120 -->
<!-- npu="A3" id1121 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1121 -->
<!-- npu="950" id1122 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1122 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, complex64, complex128

</div>

> <font size="3">sinh()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sinh](https://pytorch.org/docs/2.7/generated/torch.Tensor.sinh.html)

**Support Status**:

<!-- npu="910b" id1123 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1123 -->
<!-- npu="A3" id1124 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1124 -->
<!-- npu="950" id1125 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1125 -->

**Constraints & Notes**: `self` only supports fp16, fp32, fp64

</div>

> <font size="3">sinh_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sinh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sinh_.html)

**Support Status**:

<!-- npu="910b" id1126 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1126 -->
<!-- npu="A3" id1127 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1127 -->
<!-- npu="950" id1128 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1128 -->

**Constraints & Notes**: `self` only supports fp16, fp32, fp64

</div>

> <font size="3">asinh()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.asinh](https://pytorch.org/docs/2.7/generated/torch.Tensor.asinh.html)

**Support Status**:

<!-- npu="910b" id1129 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1129 -->
<!-- npu="A3" id1130 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1130 -->
<!-- npu="950" id1131 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1131 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">asinh_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.asinh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.asinh_.html)

**Support Status**:

<!-- npu="910b" id1132 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1132 -->
<!-- npu="A3" id1133 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1133 -->
<!-- npu="950" id1134 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1134 -->

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">arcsinh()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arcsinh](https://pytorch.org/docs/2.7/generated/torch.Tensor.arcsinh.html)

**Support Status**:

<!-- npu="910b" id1135 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1135 -->
<!-- npu="A3" id1136 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1136 -->
<!-- npu="950" id1137 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1137 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">arcsinh_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arcsinh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arcsinh_.html)

**Support Status**:

<!-- npu="910b" id1138 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1138 -->
<!-- npu="A3" id1139 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1139 -->
<!-- npu="950" id1140 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1140 -->

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.shape](https://pytorch.org/docs/2.7/generated/torch.Tensor.shape.html)

**Support Status**:

<!-- npu="910b" id1141 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1141 -->
<!-- npu="A3" id1142 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1142 -->
<!-- npu="950" id1143 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1143 -->

</div>

> <font size="3">size()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.size](https://pytorch.org/docs/2.7/generated/torch.Tensor.size.html)

**Support Status**:

<!-- npu="910b" id1144 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1144 -->
<!-- npu="A3" id1145 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1145 -->
<!-- npu="950" id1146 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1146 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">slogdet()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.slogdet](https://pytorch.org/docs/2.7/generated/torch.Tensor.slogdet.html)

**Support Status**:

<!-- npu="910b" id1147 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1147 -->
<!-- npu="A3" id1148 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1148 -->
<!-- npu="950" id1149 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1149 -->

**Constraints & Notes**: `self` only supports fp32, complex64, complex128

</div>

> <font size="3">slice_scatter()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.slice_scatter](https://pytorch.org/docs/2.7/generated/torch.Tensor.slice_scatter.html)

**Support Status**:

<!-- npu="910b" id1150 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1150 -->
<!-- npu="A3" id1151 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1151 -->
<!-- npu="950" id1152 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1152 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">softmax()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.softmax](https://pytorch.org/docs/2.7/generated/torch.Tensor.softmax.html)

**Support Status**:

<!-- npu="910b" id1153 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1153 -->
<!-- npu="A3" id1154 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1154 -->
<!-- npu="950" id1155 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1155 -->

**Constraints & Notes**: `self` only supports fp16, fp32, fp64

</div>

> <font size="3">sort()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sort](https://pytorch.org/docs/2.7/generated/torch.Tensor.sort.html)

**Support Status**:

<!-- npu="910b" id1156 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1156 -->
<!-- npu="A3" id1157 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1157 -->
<!-- npu="950" id1158 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1158 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

<!-- npu="950" id1159 -->
- For <term>Ascend 950DT products</term>, due to underlying implementation limitations, `"stable"` only supports True. If set to False, it is automatically changed to True during execution
<!-- end id1159 -->

</div>

> <font size="3">split()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.split](https://pytorch.org/docs/2.7/generated/torch.Tensor.split.html)

**Support Status**:

<!-- npu="910b" id1160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1160 -->
<!-- npu="A3" id1161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1161 -->
<!-- npu="950" id1162 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1162 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">sparse_mask()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sparse_mask](https://pytorch.org/docs/2.7/generated/torch.Tensor.sparse_mask.html)

**Support Status**:

<!-- npu="910b" id1163 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id1163 -->
<!-- npu="A3" id1164 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id1164 -->
<!-- npu="950" id1165 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1165 -->

</div>

> <font size="3">sparse_dim()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sparse_dim](https://pytorch.org/docs/2.7/generated/torch.Tensor.sparse_dim.html)

**Support Status**:

<!-- npu="910b" id1166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1166 -->
<!-- npu="A3" id1167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1167 -->
<!-- npu="950" id1168 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1168 -->

</div>

> <font size="3">sqrt()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sqrt](https://pytorch.org/docs/2.7/generated/torch.Tensor.sqrt.html)

**Support Status**:

<!-- npu="910b" id1169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1169 -->
<!-- npu="A3" id1170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1170 -->
<!-- npu="950" id1171 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1171 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">sqrt_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sqrt_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sqrt_.html)

**Support Status**:

<!-- npu="910b" id1172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1172 -->
<!-- npu="A3" id1173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1173 -->
<!-- npu="950" id1174 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1174 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

> <font size="3">square()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.square](https://pytorch.org/docs/2.7/generated/torch.Tensor.square.html)

**Support Status**:

<!-- npu="910b" id1175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1175 -->
<!-- npu="A3" id1176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1176 -->
<!-- npu="950" id1177 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1177 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">square_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.square_](https://pytorch.org/docs/2.7/generated/torch.Tensor.square_.html)

**Support Status**:

<!-- npu="910b" id1178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1178 -->
<!-- npu="A3" id1179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1179 -->
<!-- npu="950" id1180 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1180 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128

</div>

> <font size="3">squeeze()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.squeeze](https://pytorch.org/docs/2.7/generated/torch.Tensor.squeeze.html)

**Support Status**:

<!-- npu="910b" id1181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1181 -->
<!-- npu="A3" id1182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1182 -->
<!-- npu="950" id1183 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1183 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">squeeze_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.squeeze_](https://pytorch.org/docs/2.7/generated/torch.Tensor.squeeze_.html)

**Support Status**:

<!-- npu="910b" id1184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1184 -->
<!-- npu="A3" id1185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1185 -->
<!-- npu="950" id1186 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1186 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">std()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.std](https://pytorch.org/docs/2.7/generated/torch.Tensor.std.html)

**Support Status**:

<!-- npu="910b" id1187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1187 -->
<!-- npu="A3" id1188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1188 -->
<!-- npu="950" id1189 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1189 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- `input` does not support scalar`tensor`
- The value of the `correction` parameter cannot exceed the maximum value of int32

</div>

> <font size="3">storage()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.storage](https://pytorch.org/docs/2.7/generated/torch.Tensor.storage.html)

**Support Status**:

<!-- npu="910b" id1190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1190 -->
<!-- npu="A3" id1191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1191 -->
<!-- npu="950" id1192 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1192 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">untyped_storage()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.untyped_storage](https://pytorch.org/docs/2.7/generated/torch.Tensor.untyped_storage.html)

**Support Status**:

<!-- npu="910b" id1193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1193 -->
<!-- npu="A3" id1194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1194 -->
<!-- npu="950" id1195 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1195 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">storage_offset()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.storage_offset](https://pytorch.org/docs/2.7/generated/torch.Tensor.storage_offset.html)

**Support Status**:

<!-- npu="910b" id1196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1196 -->
<!-- npu="A3" id1197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1197 -->
<!-- npu="950" id1198 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1198 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">storage_type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.storage_type](https://pytorch.org/docs/2.7/generated/torch.Tensor.storage_type.html)

**Support Status**:

<!-- npu="910b" id1199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1199 -->
<!-- npu="A3" id1200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1200 -->
<!-- npu="950" id1201 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1201 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">stride()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.stride](https://pytorch.org/docs/2.7/generated/torch.Tensor.stride.html)

**Support Status**:

<!-- npu="910b" id1202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1202 -->
<!-- npu="A3" id1203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1203 -->
<!-- npu="950" id1204 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1204 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">sub()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sub](https://pytorch.org/docs/2.7/generated/torch.Tensor.sub.html)

**Support Status**:

<!-- npu="910b" id1205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1205 -->
<!-- npu="A3" id1206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1206 -->
<!-- npu="950" id1207 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1207 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

> <font size="3">sub_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sub_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sub_.html)

**Support Status**:

<!-- npu="910b" id1208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1208 -->
<!-- npu="A3" id1209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1209 -->
<!-- npu="950" id1210 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1210 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128

</div>

> <font size="3">subtract_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.subtract_](https://pytorch.org/docs/2.7/generated/torch.Tensor.subtract_.html)

**Support Status**:

<!-- npu="910b" id1211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1211 -->
<!-- npu="A3" id1212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1212 -->
<!-- npu="950" id1213 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1213 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

> <font size="3">sum()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sum](https://pytorch.org/docs/2.7/generated/torch.Tensor.sum.html)

**Support Status**:

<!-- npu="910b" id1214 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1214 -->
<!-- npu="A3" id1215 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1215 -->
<!-- npu="950" id1216 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1216 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">sum_to_size()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.sum_to_size](https://pytorch.org/docs/2.7/generated/torch.Tensor.sum_to_size.html)

**Support Status**:

<!-- npu="910b" id1217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1217 -->
<!-- npu="A3" id1218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1218 -->
<!-- npu="950" id1219 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1219 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">swapaxes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.swapaxes](https://pytorch.org/docs/2.7/generated/torch.Tensor.swapaxes.html)

**Support Status**:

<!-- npu="910b" id1220 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1220 -->
<!-- npu="A3" id1221 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1221 -->
<!-- npu="950" id1222 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1222 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">swapdims()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.swapdims](https://pytorch.org/docs/2.7/generated/torch.Tensor.swapdims.html)

**Support Status**:

<!-- npu="910b" id1223 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1223 -->
<!-- npu="A3" id1224 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1224 -->
<!-- npu="950" id1225 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1225 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">t()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.t](https://pytorch.org/docs/2.7/generated/torch.Tensor.t.html)

**Support Status**:

<!-- npu="910b" id1226 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1226 -->
<!-- npu="A3" id1227 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1227 -->
<!-- npu="950" id1228 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1228 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128

</div>

> <font size="3">t_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.t_](https://pytorch.org/docs/2.7/generated/torch.Tensor.t_.html)

**Support Status**:

<!-- npu="910b" id1229 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1229 -->
<!-- npu="A3" id1230 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1230 -->
<!-- npu="950" id1231 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1231 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64

</div>

> <font size="3">tensor_split()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.tensor_split](https://pytorch.org/docs/2.7/generated/torch.Tensor.tensor_split.html)

**Support Status**:

<!-- npu="910b" id1232 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1232 -->
<!-- npu="A3" id1233 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1233 -->
<!-- npu="950" id1234 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1234 -->

**Constraints & Notes**: CPU-only

</div>

> <font size="3">tile()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.tile](https://pytorch.org/docs/2.7/generated/torch.Tensor.tile.html)

**Support Status**:

<!-- npu="910b" id1235 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1235 -->
<!-- npu="A3" id1236 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1236 -->
<!-- npu="950" id1237 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1237 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- If the length of the `dims` parameter is less than the length of `Tensor.shape`, 1s are automatically prepended to `dims` to align its length with `Tensor.shape`. The padded `dims` must meet the following restrictions:
  - When the first axis needs to be repeated, a maximum of 4 dimensions can be repeated at the same time (that is, the number of elements greater than 1 in `dims` ≤ 4). For example, `Tensor.tile([2, 3, 4, 5, 6])` is not supported, while `Tensor.tile([2, 3, 1, 5, 6])` is supported.
  - When the first axis does not need to be repeated, a maximum of 3 dimensions can be repeated at the same time (that is, the number of elements greater than 1 in `dims` ≤ 3). For example, `Tensor.tile([1, 3, 4, 5, 6])` is not supported, while `Tensor.tile([1, 3, 1, 5, 6])` is supported.
  - In backward computation, the number of dimensions of `Tensor` plus the number of elements greater than 1 in `dims` must not exceed 8.

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.to](https://pytorch.org/docs/2.7/generated/torch.Tensor.to.html)

**Support Status**:

<!-- npu="910b" id1238 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1238 -->
<!-- npu="A3" id1239 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1239 -->
<!-- npu="950" id1240 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1240 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Current NPU devices only support setting `memory_format` to torch.contiguous_format or torch.preserve_format

<!-- npu="310p" id1241 -->
- <term>Atlas inference products</term> do not support cross-NPU copy
<!-- end id1241 -->

</div>

> <font size="3">to_mkldnn()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.to_mkldnn](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_mkldnn.html)

**Support Status**:

<!-- npu="910b" id1242 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1242 -->
<!-- npu="A3" id1243 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1243 -->
<!-- npu="950" id1244 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1244 -->

</div>

> <font size="3">take()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.take](https://pytorch.org/docs/2.7/generated/torch.Tensor.take.html)

**Support Status**:

<!-- npu="910b" id1245 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1245 -->
<!-- npu="A3" id1246 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1246 -->
<!-- npu="950" id1247 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1247 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, bool

</div>

> <font size="3">take_along_dim()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.take_along_dim](https://pytorch.org/docs/2.7/generated/torch.Tensor.take_along_dim.html)

**Support Status**:

<!-- npu="910b" id1248 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1248 -->
<!-- npu="A3" id1249 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1249 -->
<!-- npu="950" id1250 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1250 -->

**Constraints & Notes**: `self` only supports fp16, fp32, int16, int32, int64, bool

</div>

> <font size="3">tan()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.tan](https://pytorch.org/docs/2.7/generated/torch.Tensor.tan.html)

**Support Status**:

<!-- npu="910b" id1251 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1251 -->
<!-- npu="A3" id1252 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1252 -->
<!-- npu="950" id1253 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1253 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128 with a value range of [-65504,65504]

</div>

> <font size="3">tan_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.tan_](https://pytorch.org/docs/2.7/generated/torch.Tensor.tan_.html)

**Support Status**:

<!-- npu="910b" id1254 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1254 -->
<!-- npu="A3" id1255 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1255 -->
<!-- npu="950" id1256 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1256 -->

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">tanh()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.tanh](https://pytorch.org/docs/2.7/generated/torch.Tensor.tanh.html)

**Support Status**:

<!-- npu="910b" id1257 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1257 -->
<!-- npu="A3" id1258 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1258 -->
<!-- npu="950" id1259 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1259 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

<!-- npu="950" id1260 -->
- <term>Ascend 950DT products</term>: Not Supported for fp64
<!-- end id1260 -->

</div>

> <font size="3">tanh_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.tanh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.tanh_.html)

**Support Status**:

<!-- npu="910b" id1261 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1261 -->
<!-- npu="A3" id1262 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1262 -->
<!-- npu="950" id1263 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1263 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">atanh()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.atanh](https://pytorch.org/docs/2.7/generated/torch.Tensor.atanh.html)

**Support Status**:

<!-- npu="910b" id1264 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1264 -->
<!-- npu="A3" id1265 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1265 -->
<!-- npu="950" id1266 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1266 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">atanh_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.atanh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.atanh_.html)

**Support Status**:

<!-- npu="910b" id1267 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1267 -->
<!-- npu="A3" id1268 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1268 -->
<!-- npu="950" id1269 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1269 -->

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">arctanh()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arctanh](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctanh.html)

**Support Status**:

<!-- npu="910b" id1270 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1270 -->
<!-- npu="A3" id1271 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1271 -->
<!-- npu="950" id1272 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1272 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">arctanh_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.arctanh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctanh_.html)

**Support Status**:

<!-- npu="910b" id1273 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1273 -->
<!-- npu="A3" id1274 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1274 -->
<!-- npu="950" id1275 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1275 -->

**Constraints & Notes**: `self` only supports fp16, fp32, complex64, complex128

</div>

> <font size="3">tolist()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.tolist](https://pytorch.org/docs/2.7/generated/torch.Tensor.tolist.html)

**Support Status**:

<!-- npu="910b" id1276 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1276 -->
<!-- npu="A3" id1277 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1277 -->
<!-- npu="950" id1278 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1278 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">topk()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.topk](https://pytorch.org/docs/2.7/generated/torch.Tensor.topk.html)

**Support Status**:

<!-- npu="910b" id1279 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1279 -->
<!-- npu="A3" id1280 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1280 -->
<!-- npu="950" id1281 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1281 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64
- Due to hardware differences, the npu `topk` index results are inconsistent with those of GPU/CPU. Current NPUs only support returning computation results with `sorted` set to true.
- Scalar `tensor` is not supported.

</div>

> <font size="3">to_dense()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.to_dense](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_dense.html)

**Support Status**:

<!-- npu="910b" id1282 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1282 -->
<!-- npu="A3" id1283 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1283 -->
<!-- npu="950" id1284 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1284 -->

</div>

> <font size="3">to_sparse()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.to_sparse](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse.html)

**Support Status**:

<!-- npu="910b" id1285 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id1285 -->
<!-- npu="A3" id1286 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id1286 -->
<!-- npu="950" id1287 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1287 -->

</div>

> <font size="3">to_sparse_csr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.to_sparse_csr](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse_csr.html)

**Support Status**:

<!-- npu="910b" id1288 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1288 -->
<!-- npu="A3" id1289 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1289 -->
<!-- npu="950" id1290 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1290 -->

</div>

> <font size="3">to_sparse_csc()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.to_sparse_csc](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse_csc.html)

**Support Status**:

<!-- npu="910b" id1291 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1291 -->
<!-- npu="A3" id1292 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1292 -->
<!-- npu="950" id1293 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1293 -->

</div>

> <font size="3">to_sparse_bsr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.to_sparse_bsr](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse_bsr.html)

**Support Status**:

<!-- npu="910b" id1294 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1294 -->
<!-- npu="A3" id1295 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1295 -->
<!-- npu="950" id1296 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1296 -->

</div>

> <font size="3">to_sparse_bsc()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.to_sparse_bsc](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse_bsc.html)

**Support Status**:

<!-- npu="910b" id1297 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1297 -->
<!-- npu="A3" id1298 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1298 -->
<!-- npu="950" id1299 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1299 -->

</div>

> <font size="3">transpose()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.transpose](https://pytorch.org/docs/2.7/generated/torch.Tensor.transpose.html)

**Support Status**:

<!-- npu="910b" id1300 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1300 -->
<!-- npu="A3" id1301 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1301 -->
<!-- npu="950" id1302 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1302 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">transpose_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.transpose_](https://pytorch.org/docs/2.7/generated/torch.Tensor.transpose_.html)

**Support Status**:

<!-- npu="910b" id1303 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1303 -->
<!-- npu="A3" id1304 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1304 -->
<!-- npu="950" id1305 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1305 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">tril()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.tril](https://pytorch.org/docs/2.7/generated/torch.Tensor.tril.html)

**Support Status**:

<!-- npu="910b" id1306 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1306 -->
<!-- npu="A3" id1307 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1307 -->
<!-- npu="950" id1308 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1308 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">tril_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.tril_](https://pytorch.org/docs/2.7/generated/torch.Tensor.tril_.html)

**Support Status**:

<!-- npu="910b" id1309 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1309 -->
<!-- npu="A3" id1310 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1310 -->
<!-- npu="950" id1311 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1311 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">triu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.triu](https://pytorch.org/docs/2.7/generated/torch.Tensor.triu.html)

**Support Status**:

<!-- npu="910b" id1312 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1312 -->
<!-- npu="A3" id1313 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1313 -->
<!-- npu="950" id1314 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1314 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">triu_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.triu_](https://pytorch.org/docs/2.7/generated/torch.Tensor.triu_.html)

**Support Status**:

<!-- npu="910b" id1315 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1315 -->
<!-- npu="A3" id1316 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1316 -->
<!-- npu="950" id1317 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1317 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">true_divide()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.true_divide](https://pytorch.org/docs/2.7/generated/torch.Tensor.true_divide.html)

**Support Status**:

<!-- npu="910b" id1318 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1318 -->
<!-- npu="A3" id1319 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1319 -->
<!-- npu="950" id1320 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1320 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">true_divide_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.true_divide_](https://pytorch.org/docs/2.7/generated/torch.Tensor.true_divide_.html)

**Support Status**:

<!-- npu="910b" id1321 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1321 -->
<!-- npu="A3" id1322 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1322 -->
<!-- npu="950" id1323 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1323 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">trunc()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.trunc](https://pytorch.org/docs/2.7/generated/torch.Tensor.trunc.html)

**Support Status**:

<!-- npu="910b" id1324 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1324 -->
<!-- npu="A3" id1325 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1325 -->
<!-- npu="950" id1326 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1326 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">trunc_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.trunc_](https://pytorch.org/docs/2.7/generated/torch.Tensor.trunc_.html)

**Support Status**:

<!-- npu="910b" id1327 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1327 -->
<!-- npu="A3" id1328 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1328 -->
<!-- npu="950" id1329 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1329 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.type](https://pytorch.org/docs/2.7/generated/torch.Tensor.type.html)

**Support Status**:

<!-- npu="910b" id1330 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1330 -->
<!-- npu="A3" id1331 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1331 -->
<!-- npu="950" id1332 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1332 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">type_as()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.type_as](https://pytorch.org/docs/2.7/generated/torch.Tensor.type_as.html)

**Support Status**:

<!-- npu="910b" id1333 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1333 -->
<!-- npu="A3" id1334 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1334 -->
<!-- npu="950" id1335 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1335 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">unbind()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.unbind](https://pytorch.org/docs/2.7/generated/torch.Tensor.unbind.html)

**Support Status**:

<!-- npu="910b" id1336 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1336 -->
<!-- npu="A3" id1337 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1337 -->
<!-- npu="950" id1338 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1338 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">unflatten()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.unflatten](https://pytorch.org/docs/2.7/generated/torch.Tensor.unflatten.html)

**Support Status**:

<!-- npu="910b" id1339 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1339 -->
<!-- npu="A3" id1340 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1340 -->
<!-- npu="950" id1341 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1341 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">unfold()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.unfold](https://pytorch.org/docs/2.7/generated/torch.Tensor.unfold.html)

**Support Status**:

<!-- npu="910b" id1342 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1342 -->
<!-- npu="A3" id1343 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1343 -->
<!-- npu="950" id1344 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1344 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">uniform_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.uniform_](https://pytorch.org/docs/2.7/generated/torch.Tensor.uniform_.html)

**Support Status**:

<!-- npu="910b" id1345 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1345 -->
<!-- npu="A3" id1346 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1346 -->
<!-- npu="950" id1347 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1347 -->

**Constraints & Notes**:

- `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64
- Following the PyTorch community convention, processing bool-type data is no longer supported. For existing bool-type data, use the following replacement methods: if all True values are required, use Tensor.bernoulli_(p=1.0). If uniformly distributed bool values are required, use Tensor.bernoulli_(p=0.5).

</div>

> <font size="3">unique()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.unique](https://pytorch.org/docs/2.7/generated/torch.Tensor.unique.html)

**Support Status**:

<!-- npu="910b" id1348 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1348 -->
<!-- npu="A3" id1349 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1349 -->
<!-- npu="950" id1350 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1350 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- When the input contains 0, the output may contain positive 0 and negative 0 instead of only one 0

</div>

> <font size="3">unique_consecutive()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.unique_consecutive](https://pytorch.org/docs/2.7/generated/torch.Tensor.unique_consecutive.html)

**Support Status**:

<!-- npu="910b" id1351 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1351 -->
<!-- npu="A3" id1352 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1352 -->
<!-- npu="950" id1353 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1353 -->

**Constraints & Notes**: `self` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">unsqueeze()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.unsqueeze](https://pytorch.org/docs/2.7/generated/torch.Tensor.unsqueeze.html)

**Support Status**:

<!-- npu="910b" id1354 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1354 -->
<!-- npu="A3" id1355 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1355 -->
<!-- npu="950" id1356 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1356 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">unsqueeze_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.unsqueeze_](https://pytorch.org/docs/2.7/generated/torch.Tensor.unsqueeze_.html)

**Support Status**:

<!-- npu="910b" id1357 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1357 -->
<!-- npu="A3" id1358 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1358 -->
<!-- npu="950" id1359 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1359 -->

**Constraints & Notes**: `self` only supports fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.values](https://pytorch.org/docs/2.7/generated/torch.Tensor.values.html)

**Support Status**:

<!-- npu="910b" id1360 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1360 -->
<!-- npu="A3" id1361 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1361 -->
<!-- npu="950" id1362 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1362 -->

**Constraints & Notes**: Depends on sparse `tensor`

</div>

> <font size="3">var()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.var](https://pytorch.org/docs/2.7/generated/torch.Tensor.var.html)

**Support Status**:

<!-- npu="910b" id1363 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1363 -->
<!-- npu="A3" id1364 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1364 -->
<!-- npu="950" id1365 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1365 -->

**Constraints & Notes**:

- `self` only supports bf16, fp16, fp32
- The value of the `correction` parameter cannot exceed the maximum value of int32

</div>

> <font size="3">view()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.view](https://pytorch.org/docs/2.7/generated/torch.Tensor.view.html)

**Support Status**:

<!-- npu="910b" id1366 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1366 -->
<!-- npu="A3" id1367 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1367 -->
<!-- npu="950" id1368 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1368 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">view_as()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.view_as](https://pytorch.org/docs/2.7/generated/torch.Tensor.view_as.html)

**Support Status**:

<!-- npu="910b" id1369 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1369 -->
<!-- npu="A3" id1370 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1370 -->
<!-- npu="950" id1371 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1371 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">vsplit()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.vsplit](https://pytorch.org/docs/2.7/generated/torch.Tensor.vsplit.html)

**Support Status**:

<!-- npu="910b" id1372 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1372 -->
<!-- npu="A3" id1373 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1373 -->
<!-- npu="950" id1374 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1374 -->

**Constraints & Notes**: `self` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">where()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.where](https://pytorch.org/docs/2.7/generated/torch.Tensor.where.html)

**Support Status**:

<!-- npu="910b" id1375 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1375 -->
<!-- npu="A3" id1376 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1376 -->
<!-- npu="950" id1377 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1377 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">xlogy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.xlogy](https://pytorch.org/docs/2.7/generated/torch.Tensor.xlogy.html)

**Support Status**:

<!-- npu="910b" id1378 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1378 -->
<!-- npu="A3" id1379 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1379 -->
<!-- npu="950" id1380 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1380 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">xlogy_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.xlogy_](https://pytorch.org/docs/2.7/generated/torch.Tensor.xlogy_.html)

**Support Status**:

<!-- npu="910b" id1381 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1381 -->
<!-- npu="A3" id1382 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1382 -->
<!-- npu="950" id1383 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1383 -->

**Constraints & Notes**: `self` only supports fp16, fp32

</div>

> <font size="3">zero_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Tensor.zero_](https://pytorch.org/docs/2.7/generated/torch.Tensor.zero_.html)

**Support Status**:

<!-- npu="910b" id1384 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1384 -->
<!-- npu="A3" id1385 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1385 -->
<!-- npu="950" id1386 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1386 -->

**Constraints & Notes**: `self` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

</div>
