# torch

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/torch.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Tensors](#tensors)
- [Generators](#generators)
- [Random sampling](#random-sampling)
- [Serialization](#serialization)
- [Parallelism](#parallelism)
- [Locally disabling gradient computation](#locally-disabling-gradient-computation)
- [Math operations](#math-operations)
- [Utilities](#utilities)
- [Symbolic Numbers](#symbolic-numbers)
- [Optimizations](#optimizations)
- [Operator Tags](#operator-tags)

</div>

<div style="display:none;">

## &#8203;torch

</div>

## Tensors

### torch.is_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_tensor](https://pytorch.org/docs/2.7/generated/torch.is_tensor.html)

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

</div>

### torch.is_storage

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_storage](https://pytorch.org/docs/2.7/generated/torch.is_storage.html)

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

</div>

### torch.is_complex

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_complex](https://pytorch.org/docs/2.7/generated/torch.is_complex.html)

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

**Constraints & Notes**: `input` only supports complex64, complex128

</div>

### torch.is_conj

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_conj](https://pytorch.org/docs/2.7/generated/torch.is_conj.html)

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

### torch.is_floating_point

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_floating_point](https://pytorch.org/docs/2.7/generated/torch.is_floating_point.html)

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

### torch.is_nonzero

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_nonzero](https://pytorch.org/docs/2.7/generated/torch.is_nonzero.html)

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

### torch.set_default_dtype

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_default_dtype](https://pytorch.org/docs/2.7/generated/torch.set_default_dtype.html)

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

### torch.get_default_dtype

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_default_dtype](https://pytorch.org/docs/2.7/generated/torch.get_default_dtype.html)

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

### torch.set_default_device

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_default_device](https://pytorch.org/docs/2.7/generated/torch.set_default_device.html)

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

### torch.set_default_tensor_type

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_default_tensor_type](https://pytorch.org/docs/2.7/generated/torch.set_default_tensor_type.html)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id30 -->

**Constraints & Notes**: Does not support passing in the torch.npu.DtypeTensor type.

</div>

### torch.numel

<div style="margin-left: 2em">

**Native Documentation**: [torch.numel](https://pytorch.org/docs/2.7/generated/torch.numel.html)

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

### torch.set_printoptions

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_printoptions](https://pytorch.org/docs/2.7/generated/torch.set_printoptions.html)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

</div>

### torch.set_flush_denormal

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_flush_denormal](https://pytorch.org/docs/2.7/generated/torch.set_flush_denormal.html)

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

### torch.tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.tensor](https://pytorch.org/docs/2.7/generated/torch.tensor.html)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id42 -->

</div>

### torch.sparse_coo_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse_coo_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_coo_tensor.html)

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

**Constraints & Notes**:

- `indices` only supports int32, int64
- `values` only supports fp16, fp32, int32
- The `dtype` parameter must be the same as the `dtype` of `values`

</div>

### torch.sparse_csr_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse_csr_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_csr_tensor.html)

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

</div>

### torch.sparse_csc_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse_csc_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_csc_tensor.html)

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

### torch.sparse_bsr_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse_bsr_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_bsr_tensor.html)

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

### torch.sparse_bsc_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.sparse_bsc_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_bsc_tensor.html)

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

### torch.asarray

<div style="margin-left: 2em">

**Native Documentation**: [torch.asarray](https://pytorch.org/docs/2.7/generated/torch.asarray.html)

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

**Constraints & Notes**: `obj` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.as_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.as_tensor](https://pytorch.org/docs/2.7/generated/torch.as_tensor.html)

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

**Constraints & Notes**: `data` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.as_strided

<div style="margin-left: 2em">

**Native Documentation**: [torch.as_strided](https://pytorch.org/docs/2.7/generated/torch.as_strided.html)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.from_numpy

<div style="margin-left: 2em">

**Native Documentation**: [torch.from_numpy](https://pytorch.org/docs/2.7/generated/torch.from_numpy.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.from_dlpack

<div style="margin-left: 2em">

**Native Documentation**: [torch.from_dlpack](https://pytorch.org/docs/2.7/generated/torch.from_dlpack.html)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id72 -->

</div>

### torch.frombuffer

<div style="margin-left: 2em">

**Native Documentation**: [torch.frombuffer](https://pytorch.org/docs/2.7/generated/torch.frombuffer.html)

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

**Constraints & Notes**: `dtype` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.zeros

<div style="margin-left: 2em">

**Native Documentation**: [torch.zeros](https://pytorch.org/docs/2.7/generated/torch.zeros.html)

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

**Constraints & Notes**: `dtype` only supports bf16, fp16, fp32, uint8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.zeros_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.zeros_like](https://pytorch.org/docs/2.7/generated/torch.zeros_like.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.ones

<div style="margin-left: 2em">

**Native Documentation**: [torch.ones](https://pytorch.org/docs/2.7/generated/torch.ones.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.ones_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.ones_like](https://pytorch.org/docs/2.7/generated/torch.ones_like.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.arange

<div style="margin-left: 2em">

**Native Documentation**: [torch.arange](https://pytorch.org/docs/2.7/generated/torch.arange.html)

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

**Constraints & Notes**: `dtype` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.range

<div style="margin-left: 2em">

**Native Documentation**: [torch.range](https://pytorch.org/docs/2.7/generated/torch.range.html)

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

### torch.linspace

<div style="margin-left: 2em">

**Native Documentation**: [torch.linspace](https://pytorch.org/docs/2.7/generated/torch.linspace.html)

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

**Constraints & Notes**:

- `dtype` only supports bf16, fp16, fp32, fp64, int16, int32, int64, bool, complex64, complex128
- Creates a 1-D vector with a sequence size of `steps`

</div>

### torch.eye

<div style="margin-left: 2em">

**Native Documentation**: [torch.eye](https://pytorch.org/docs/2.7/generated/torch.eye.html)

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

**Constraints & Notes**: `dtype` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.empty

<div style="margin-left: 2em">

**Native Documentation**: [torch.empty](https://pytorch.org/docs/2.7/generated/torch.empty.html)

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

**Constraints & Notes**: `dtype` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.empty_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.empty_like](https://pytorch.org/docs/2.7/generated/torch.empty_like.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.empty_strided

<div style="margin-left: 2em">

**Native Documentation**: [torch.empty_strided](https://pytorch.org/docs/2.7/generated/torch.empty_strided.html)

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

</div>

### torch.full

<div style="margin-left: 2em">

**Native Documentation**: [torch.full](https://pytorch.org/docs/2.7/generated/torch.full.html)

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

**Constraints & Notes**: `dtype` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.full_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.full_like](https://pytorch.org/docs/2.7/generated/torch.full_like.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.quantize_per_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.quantize_per_tensor](https://pytorch.org/docs/2.7/generated/torch.quantize_per_tensor.html)

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

</div>

### torch.quantize_per_channel

<div style="margin-left: 2em">

**Native Documentation**: [torch.quantize_per_channel](https://pytorch.org/docs/2.7/generated/torch.quantize_per_channel.html)

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

</div>

### torch.dequantize

<div style="margin-left: 2em">

**Native Documentation**: [torch.dequantize](https://pytorch.org/docs/2.7/generated/torch.dequantize.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.complex

<div style="margin-left: 2em">

**Native Documentation**: [torch.complex](https://pytorch.org/docs/2.7/generated/torch.complex.html)

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

</div>

### torch.polar

<div style="margin-left: 2em">

**Native Documentation**: [torch.polar](https://pytorch.org/docs/2.7/generated/torch.polar.html)

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

**Constraints & Notes**:

- `abs` only supports fp32
- The dimensions of the `abs` and `angle` parameters must be equal

</div>

### torch.heaviside

<div style="margin-left: 2em">

**Native Documentation**: [torch.heaviside](https://pytorch.org/docs/2.7/generated/torch.heaviside.html)

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

### torch.argwhere

<div style="margin-left: 2em">

**Native Documentation**: [torch.argwhere](https://pytorch.org/docs/2.7/generated/torch.argwhere.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.cat

<div style="margin-left: 2em">

**Native Documentation**: [torch.cat](https://pytorch.org/docs/2.7/generated/torch.cat.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.concat

<div style="margin-left: 2em">

**Native Documentation**: [torch.concat](https://pytorch.org/docs/2.7/generated/torch.concat.html)

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

**Constraints & Notes**:

- `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64

<!-- npu="950" id142 -->
- <term>Ascend 950DT products</term>: Not Supported for complex64
<!-- end id142 -->

</div>

### torch.concatenate

<div style="margin-left: 2em">

**Native Documentation**: [torch.concatenate](https://pytorch.org/docs/2.7/generated/torch.concatenate.html)

**Support Status**:

<!-- npu="910b" id143 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id143 -->
<!-- npu="A3" id144 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id144 -->
<!-- npu="950" id145 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id145 -->

**Constraints & Notes**:

- `tensors` only supports bf16, fp16, fp32, int64, bool, complex64

<!-- npu="950" id146 -->
- <term>Ascend 950DT products</term>: Not Supported for complex64
<!-- end id146 -->

</div>

### torch.conj

<div style="margin-left: 2em">

**Native Documentation**: [torch.conj](https://pytorch.org/docs/2.7/generated/torch.conj.html)

**Support Status**:

<!-- npu="910b" id147 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id147 -->
<!-- npu="A3" id148 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id148 -->
<!-- npu="950" id149 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id149 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.chunk

<div style="margin-left: 2em">

**Native Documentation**: [torch.chunk](https://pytorch.org/docs/2.7/generated/torch.chunk.html)

**Support Status**:

<!-- npu="910b" id150 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id150 -->
<!-- npu="A3" id151 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id151 -->
<!-- npu="950" id152 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id152 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.dsplit

<div style="margin-left: 2em">

**Native Documentation**: [torch.dsplit](https://pytorch.org/docs/2.7/generated/torch.dsplit.html)

**Support Status**:

<!-- npu="910b" id153 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id153 -->
<!-- npu="A3" id154 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id154 -->
<!-- npu="950" id155 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id155 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.dstack

<div style="margin-left: 2em">

**Native Documentation**: [torch.dstack](https://pytorch.org/docs/2.7/generated/torch.dstack.html)

**Support Status**:

<!-- npu="910b" id156 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id156 -->
<!-- npu="A3" id157 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id157 -->
<!-- npu="950" id158 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id158 -->

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

### torch.gather

<div style="margin-left: 2em">

**Native Documentation**: [torch.gather](https://pytorch.org/docs/2.7/generated/torch.gather.html)

**Support Status**:

<!-- npu="910b" id159 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id159 -->
<!-- npu="A3" id160 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id160 -->
<!-- npu="950" id161 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id161 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- The number of dimensions of `index` must be the same as that of `input`

<!-- npu="950,A3,910b" id162 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist, for example:

  ```python
  # Example of duplicate indexes in index: index 0 is used multiple times. On Ascend 950DT products, the results may differ in accuracy from those on A2/A3.
  x = torch.tensor([[1, 2], [3, 4]], device='npu')
  index = torch.tensor([[0], [0]], device='npu')  # Index 0 appears repeatedly.
  out = torch.gather(x, 0, index)
  ```
<!-- end id162 -->

</div>

### torch.hsplit

<div style="margin-left: 2em">

**Native Documentation**: [torch.hsplit](https://pytorch.org/docs/2.7/generated/torch.hsplit.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.hstack

<div style="margin-left: 2em">

**Native Documentation**: [torch.hstack](https://pytorch.org/docs/2.7/generated/torch.hstack.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64

</div>

### torch.index_add

<div style="margin-left: 2em">

**Native Documentation**: [torch.index_add](https://pytorch.org/docs/2.7/generated/torch.index_add.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.index_copy

<div style="margin-left: 2em">

**Native Documentation**: [torch.index_copy](https://pytorch.org/docs/2.7/generated/torch.index_copy.html)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.index_reduce

<div style="margin-left: 2em">

**Native Documentation**: [torch.index_reduce](https://pytorch.org/docs/2.7/generated/torch.index_reduce.html)

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

**Constraints & Notes**: May fall back to CPU execution

</div>

### torch.index_select

<div style="margin-left: 2em">

**Native Documentation**: [torch.index_select](https://pytorch.org/docs/2.7/generated/torch.index_select.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, uint8, int16, int32, int64, bool, complex64, complex128

<!-- npu="950,A3,910b" id181 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist, for example:

  ```python
  # Example of duplicate indexes in index: index 0 is used multiple times. On Ascend 950DT products, the results may differ in accuracy from those on A2/A3.
  x = torch.tensor([[1, 2], [3, 4]], device='npu')
  index = torch.tensor([0, 0], device='npu')  # Index 0 appears repeatedly.
  out = torch.index_select(x, 0, index)
  ```
<!-- end id181 -->

</div>

### torch.masked_select

<div style="margin-left: 2em">

**Native Documentation**: [torch.masked_select](https://pytorch.org/docs/2.7/generated/torch.masked_select.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.movedim

<div style="margin-left: 2em">

**Native Documentation**: [torch.movedim](https://pytorch.org/docs/2.7/generated/torch.movedim.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.moveaxis

<div style="margin-left: 2em">

**Native Documentation**: [torch.moveaxis](https://pytorch.org/docs/2.7/generated/torch.moveaxis.html)

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

**Constraints & Notes**: `input` only supports fp32, int64, complex128

</div>

### torch.narrow

<div style="margin-left: 2em">

**Native Documentation**: [torch.narrow](https://pytorch.org/docs/2.7/generated/torch.narrow.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.narrow_copy

<div style="margin-left: 2em">

**Native Documentation**: [torch.narrow_copy](https://pytorch.org/docs/2.7/generated/torch.narrow_copy.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool and may fall back to CPU execution

</div>

### torch.nonzero

<div style="margin-left: 2em">

**Native Documentation**: [torch.nonzero](https://pytorch.org/docs/2.7/generated/torch.nonzero.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.permute

<div style="margin-left: 2em">

**Native Documentation**: [torch.permute](https://pytorch.org/docs/2.7/generated/torch.permute.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.reshape

<div style="margin-left: 2em">

**Native Documentation**: [torch.reshape](https://pytorch.org/docs/2.7/generated/torch.reshape.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.row_stack

<div style="margin-left: 2em">

**Native Documentation**: [torch.row_stack](https://pytorch.org/docs/2.7/generated/torch.row_stack.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

### torch.scatter

<div style="margin-left: 2em">

**Native Documentation**: [torch.scatter](https://pytorch.org/docs/2.7/generated/torch.scatter.html)

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

**Constraints & Notes**:

- `input` only supports fp16, fp32, fp64, int8, int16, int32, int64, bool
- May fall back to CPU execution

<!-- npu="950,A3,910b" id212 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist, for example:

  ```python
  # Example of duplicate indexes in index: index 0 is used multiple times and the same position is written multiple times. On Ascend 950DT products, the results may differ in accuracy from those on A2/A3.
  x = torch.tensor([[1, 2], [3, 4]], device='npu')
  src = torch.tensor([[10, 20], [30, 40]], device='npu')
  index = torch.tensor([[0], [0]], device='npu')  # Index 0 appears repeatedly.
  out = torch.scatter(x, 0, index, src)
  ```
<!-- end id212 -->

</div>

### torch.diagonal_scatter

<div style="margin-left: 2em">

**Native Documentation**: [torch.diagonal_scatter](https://pytorch.org/docs/2.7/generated/torch.diagonal_scatter.html)

**Support Status**:

<!-- npu="910b" id213 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id213 -->
<!-- npu="A3" id214 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id214 -->
<!-- npu="950" id215 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id215 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int16, int32, int64, bool

</div>

### torch.select_scatter

<div style="margin-left: 2em">

**Native Documentation**: [torch.select_scatter](https://pytorch.org/docs/2.7/generated/torch.select_scatter.html)

**Support Status**:

<!-- npu="910b" id216 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id216 -->
<!-- npu="A3" id217 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id217 -->
<!-- npu="950" id218 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id218 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.slice_scatter

<div style="margin-left: 2em">

**Native Documentation**: [torch.slice_scatter](https://pytorch.org/docs/2.7/generated/torch.slice_scatter.html)

**Support Status**:

<!-- npu="910b" id219 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id219 -->
<!-- npu="A3" id220 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id220 -->
<!-- npu="950" id221 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id221 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.scatter_add

<div style="margin-left: 2em">

**Native Documentation**: [torch.scatter_add](https://pytorch.org/docs/2.7/generated/torch.scatter_add.html)

**Support Status**:

<!-- npu="910b" id222 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id222 -->
<!-- npu="A3" id223 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id223 -->
<!-- npu="950" id224 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id224 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

<!-- npu="950,A3,910b" id225 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term> when duplicate indices exist, for example:

  ```python
  # Example of duplicate indexes in index: index 0 is used multiple times and the same position is accumulated multiple times. On Ascend 950DT products, the results may differ in accuracy from those on A2/A3.
  x = torch.tensor([[1, 2], [3, 4]], device='npu')
  src = torch.tensor([[10, 20], [30, 40]], device='npu')
  index = torch.tensor([[0], [0]], device='npu')  # Index 0 appears repeatedly.
  out = torch.scatter_add(x, 0, index, src)
  ```
<!-- end id225 -->

</div>

### torch.scatter_reduce

<div style="margin-left: 2em">

**Native Documentation**: [torch.scatter_reduce](https://pytorch.org/docs/2.7/generated/torch.scatter_reduce.html)

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

</div>

### torch.split

<div style="margin-left: 2em">

**Native Documentation**: [torch.split](https://pytorch.org/docs/2.7/generated/torch.split.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.squeeze

<div style="margin-left: 2em">

**Native Documentation**: [torch.squeeze](https://pytorch.org/docs/2.7/generated/torch.squeeze.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.stack

<div style="margin-left: 2em">

**Native Documentation**: [torch.stack](https://pytorch.org/docs/2.7/generated/torch.stack.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.swapaxes

<div style="margin-left: 2em">

**Native Documentation**: [torch.swapaxes](https://pytorch.org/docs/2.7/generated/torch.swapaxes.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.swapdims

<div style="margin-left: 2em">

**Native Documentation**: [torch.swapdims](https://pytorch.org/docs/2.7/generated/torch.swapdims.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.t

<div style="margin-left: 2em">

**Native Documentation**: [torch.t](https://pytorch.org/docs/2.7/generated/torch.t.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.take

<div style="margin-left: 2em">

**Native Documentation**: [torch.take](https://pytorch.org/docs/2.7/generated/torch.take.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, int16, int32, int64, bool

</div>

### torch.take_along_dim

<div style="margin-left: 2em">

**Native Documentation**: [torch.take_along_dim](https://pytorch.org/docs/2.7/generated/torch.take_along_dim.html)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.tensor_split

<div style="margin-left: 2em">

**Native Documentation**: [torch.tensor_split](https://pytorch.org/docs/2.7/generated/torch.tensor_split.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.tile

<div style="margin-left: 2em">

**Native Documentation**: [torch.tile](https://pytorch.org/docs/2.7/generated/torch.tile.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- If the length of the `dims` parameter is less than the length of `input`.shape, 1s are automatically prepended to `dims` so that its length is aligned with `input`.shape. The padded `dims` must meet the following constraints:
  - When the first axis needs to be repeated, repetition can be performed on at most 4 dimensions at the same time (that is, the number of elements greater than 1 in `dims` is ≤ 4). For example, `torch.tile(input, [2, 3, 4, 5, 6])` is not supported, while `torch.tile(input, [2, 3, 1, 5, 6])` is supported
  - When the first axis does not need to be repeated, repetition can be performed on at most 3 dimensions at the same time (that is, the number of elements greater than 1 in `dims` is ≤ 3). For example, `torch.tile(input, [1, 3, 4, 5, 6])` is not supported, while `torch.tile(input, [1, 3, 1, 5, 6])` is supported
  - In backward computation, the sum of the number of dimensions of the input `Tensor` and the number of elements greater than 1 in the `dims` parameter must not exceed 8.

</div>

### torch.transpose

<div style="margin-left: 2em">

**Native Documentation**: [torch.transpose](https://pytorch.org/docs/2.7/generated/torch.transpose.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.unsqueeze

<div style="margin-left: 2em">

**Native Documentation**: [torch.unsqueeze](https://pytorch.org/docs/2.7/generated/torch.unsqueeze.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.vsplit

<div style="margin-left: 2em">

**Native Documentation**: [torch.vsplit](https://pytorch.org/docs/2.7/generated/torch.vsplit.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.vstack

<div style="margin-left: 2em">

**Native Documentation**: [torch.vstack](https://pytorch.org/docs/2.7/generated/torch.vstack.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64

</div>

### torch.where

<div style="margin-left: 2em">

**Native Documentation**: [torch.where](https://pytorch.org/docs/2.7/generated/torch.where.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- An 8-dimensional shape is not supported

</div>

## Generators

### <code><i>class</i></code> torch.Generator

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator](https://pytorch.org/docs/2.7/generated/torch.Generator.html)

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

> <font size="3">device()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.device](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.device)

**Support Status**:

<!-- npu="910b" id277 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id277 -->
<!-- npu="A3" id278 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id278 -->
<!-- npu="950" id279 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id279 -->

</div>

> <font size="3">get_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.get_state](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.get_state)

**Support Status**:

<!-- npu="910b" id280 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id280 -->
<!-- npu="A3" id281 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id281 -->
<!-- npu="950" id282 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id282 -->

</div>

> <font size="3">initial_seed()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.initial_seed](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.initial_seed)

**Support Status**:

<!-- npu="910b" id283 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id283 -->
<!-- npu="A3" id284 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id284 -->
<!-- npu="950" id285 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id285 -->

</div>

> <font size="3">manual_seed()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.manual_seed](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.manual_seed)

**Support Status**:

<!-- npu="910b" id286 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id286 -->
<!-- npu="A3" id287 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id287 -->
<!-- npu="950" id288 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id288 -->

</div>

> <font size="3">seed()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.seed](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.seed)

**Support Status**:

<!-- npu="910b" id289 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id289 -->
<!-- npu="A3" id290 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id290 -->
<!-- npu="950" id291 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id291 -->

</div>

> <font size="3">set_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.set_state](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.set_state)

**Support Status**:

<!-- npu="910b" id292 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id292 -->
<!-- npu="A3" id293 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id293 -->
<!-- npu="950" id294 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id294 -->

</div>

> <font size="3">clone_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.clone_state](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.clone_state)

**Support Status**:

<!-- npu="910b" id295 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id295 -->
<!-- npu="A3" id296 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id296 -->
<!-- npu="950" id297 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id297 -->

</div>

> <font size="3">graphsafe_set_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.graphsafe_set_state](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.graphsafe_set_state)

**Support Status**:

<!-- npu="910b" id298 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id298 -->
<!-- npu="A3" id299 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id299 -->
<!-- npu="950" id300 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id300 -->

</div>

</div>

## Random sampling

### torch.default_generator

<div style="margin-left: 2em">

**Native Documentation**: [torch.torch.default_generator](https://pytorch.org/docs/2.7/torch.html#torch.torch.default_generator)

**Support Status**:

<!-- npu="910b" id301 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id301 -->
<!-- npu="A3" id302 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id302 -->
<!-- npu="950" id303 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id303 -->

</div>

### torch.rand

<div style="margin-left: 2em">

**Native Documentation**: [torch.rand](https://pytorch.org/docs/2.7/generated/torch.rand.html)

**Support Status**:

<!-- npu="910b" id304 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id304 -->
<!-- npu="A3" id305 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id305 -->
<!-- npu="950" id306 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id306 -->

</div>

### torch.rand_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.rand_like](https://pytorch.org/docs/2.7/generated/torch.rand_like.html)

**Support Status**:

<!-- npu="910b" id307 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id307 -->
<!-- npu="A3" id308 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id308 -->
<!-- npu="950" id309 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id309 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64
- Following the PyTorch community convention, processing bool-type data is no longer supported. For existing bool-type data, use the following replacement methods: if all True values are required, use `torch.bernoulli(input, 1)`. If uniformly distributed bool values are required, use `torch.bernoulli(input, 0.5)`.

</div>

### torch.randint

<div style="margin-left: 2em">

**Native Documentation**: [torch.randint](https://pytorch.org/docs/2.7/generated/torch.randint.html)

**Support Status**:

<!-- npu="910b" id310 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id310 -->
<!-- npu="A3" id311 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id311 -->
<!-- npu="950" id312 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id312 -->

</div>

### torch.randint_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.randint_like](https://pytorch.org/docs/2.7/generated/torch.randint_like.html)

**Support Status**:

<!-- npu="910b" id313 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id313 -->
<!-- npu="A3" id314 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id314 -->
<!-- npu="950" id315 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id315 -->

**Constraints & Notes**: `input` only supports fp16, fp32, int64

</div>

### torch.randn

<div style="margin-left: 2em">

**Native Documentation**: [torch.randn](https://pytorch.org/docs/2.7/generated/torch.randn.html)

**Support Status**:

<!-- npu="910b" id316 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id316 -->
<!-- npu="A3" id317 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id317 -->
<!-- npu="950" id318 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id318 -->

</div>

### torch.randn_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.randn_like](https://pytorch.org/docs/2.7/generated/torch.randn_like.html)

**Support Status**:

<!-- npu="910b" id319 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id319 -->
<!-- npu="A3" id320 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id320 -->
<!-- npu="950" id321 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id321 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.randperm

<div style="margin-left: 2em">

**Native Documentation**: [torch.randperm](https://pytorch.org/docs/2.7/generated/torch.randperm.html)

**Support Status**:

<!-- npu="910b" id322 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id322 -->
<!-- npu="A3" id323 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id323 -->
<!-- npu="950" id324 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id324 -->

</div>

### torch.seed

<div style="margin-left: 2em">

**Native Documentation**: [torch.seed](https://pytorch.org/docs/2.7/generated/torch.seed.html)

**Support Status**:

<!-- npu="910b" id325 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id325 -->
<!-- npu="A3" id326 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id326 -->
<!-- npu="950" id327 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id327 -->

</div>

### torch.manual_seed

<div style="margin-left: 2em">

**Native Documentation**: [torch.manual_seed](https://pytorch.org/docs/2.7/generated/torch.manual_seed.html)

**Support Status**:

<!-- npu="910b" id328 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id328 -->
<!-- npu="A3" id329 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id329 -->
<!-- npu="950" id330 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id330 -->

</div>

### torch.initial_seed

<div style="margin-left: 2em">

**Native Documentation**: [torch.initial_seed](https://pytorch.org/docs/2.7/generated/torch.initial_seed.html)

**Support Status**:

<!-- npu="910b" id331 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id331 -->
<!-- npu="A3" id332 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id332 -->
<!-- npu="950" id333 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id333 -->

</div>

### torch.get_rng_state

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_rng_state](https://pytorch.org/docs/2.7/generated/torch.get_rng_state.html)

**Support Status**:

<!-- npu="910b" id334 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id334 -->
<!-- npu="A3" id335 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id335 -->
<!-- npu="950" id336 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id336 -->

</div>

### torch.set_rng_state

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_rng_state](https://pytorch.org/docs/2.7/generated/torch.set_rng_state.html)

**Support Status**:

<!-- npu="910b" id337 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id337 -->
<!-- npu="A3" id338 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id338 -->
<!-- npu="950" id339 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id339 -->

</div>

### torch.bernoulli

<div style="margin-left: 2em">

**Native Documentation**: [torch.bernoulli](https://pytorch.org/docs/2.7/generated/torch.bernoulli.html)

**Support Status**:

<!-- npu="910b" id340 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id340 -->
<!-- npu="A3" id341 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id341 -->
<!-- npu="950" id342 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id342 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64

</div>

### torch.multinomial

<div style="margin-left: 2em">

**Native Documentation**: [torch.multinomial](https://pytorch.org/docs/2.7/generated/torch.multinomial.html)

**Support Status**:

<!-- npu="910b" id343 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id343 -->
<!-- npu="A3" id344 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id344 -->
<!-- npu="950" id345 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id345 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.normal

<div style="margin-left: 2em">

**Native Documentation**: [torch.normal](https://pytorch.org/docs/2.7/generated/torch.normal.html)

**Support Status**:

<!-- npu="910b" id346 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id346 -->
<!-- npu="A3" id347 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id347 -->
<!-- npu="950" id348 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id348 -->

**Constraints & Notes**: `mean` and `std` only support fp16, fp32, and fp64

</div>

### torch.poisson

<div style="margin-left: 2em">

**Native Documentation**: [torch.poisson](https://pytorch.org/docs/2.7/generated/torch.poisson.html)

**Support Status**:

<!-- npu="910b" id349 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id349 -->
<!-- npu="A3" id350 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id350 -->
<!-- npu="950" id351 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id351 -->

</div>

### <code><i>class</i></code> torch.quasirandom.SobolEngine

<div style="margin-left: 2em">

> <font size="3">draw()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.quasirandom.SobolEngine.draw](https://pytorch.org/docs/2.7/generated/torch.quasirandom.SobolEngine.html#torch.quasirandom.SobolEngine.draw)

**Support Status**:

<!-- npu="910b" id352 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id352 -->
<!-- npu="A3" id353 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id353 -->
<!-- npu="950" id354 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id354 -->

**Constraints & Notes**: `self` only supports fp32, fp64

</div>

</div>

## Serialization

### torch.save

<div style="margin-left: 2em">

**Native Documentation**: [torch.save](https://pytorch.org/docs/2.7/generated/torch.save.html)

**Support Status**:

<!-- npu="910b" id355 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id355 -->
<!-- npu="A3" id356 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id356 -->
<!-- npu="950" id357 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id357 -->

</div>

### torch.load

<div style="margin-left: 2em">

**Native Documentation**: [torch.load](https://pytorch.org/docs/2.7/generated/torch.load.html)

**Support Status**:

<!-- npu="910b" id358 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id358 -->
<!-- npu="A3" id359 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id359 -->
<!-- npu="950" id360 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id360 -->

</div>

### torch.serialization.get_safe_globals

<div style="margin-left: 2em">

**Native Documentation**: [torch.serialization.get_safe_globals](https://pytorch.org/docs/2.7/notes/serialization.html#torch.serialization.get_safe_globals)

**Support Status**:

<!-- npu="910b" id361 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id361 -->
<!-- npu="A3" id362 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id362 -->
<!-- npu="950" id363 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id363 -->

</div>

### <code><i>class</i></code> torch.serialization.safe_globals

<div style="margin-left: 2em">

**Native Documentation**: [torch.serialization.safe_globals](https://pytorch.org/docs/2.7/notes/serialization.html#torch.serialization.safe_globals)

**Support Status**:

<!-- npu="910b" id364 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id364 -->
<!-- npu="A3" id365 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id365 -->
<!-- npu="950" id366 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id366 -->

</div>

## Parallelism

### torch.get_num_threads

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_num_threads](https://pytorch.org/docs/2.7/generated/torch.get_num_threads.html)

**Support Status**:

<!-- npu="910b" id367 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id367 -->
<!-- npu="A3" id368 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id368 -->
<!-- npu="950" id369 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id369 -->

</div>

### torch.set_num_threads

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_num_threads](https://pytorch.org/docs/2.7/generated/torch.set_num_threads.html)

**Support Status**:

<!-- npu="910b" id370 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id370 -->
<!-- npu="A3" id371 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id371 -->
<!-- npu="950" id372 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id372 -->

</div>

### torch.get_num_interop_threads

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_num_interop_threads](https://pytorch.org/docs/2.7/generated/torch.get_num_interop_threads.html)

**Support Status**:

<!-- npu="910b" id373 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id373 -->
<!-- npu="A3" id374 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id374 -->
<!-- npu="950" id375 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id375 -->

</div>

### torch.set_num_interop_threads

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_num_interop_threads](https://pytorch.org/docs/2.7/generated/torch.set_num_interop_threads.html)

**Support Status**:

<!-- npu="910b" id376 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id376 -->
<!-- npu="A3" id377 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id377 -->
<!-- npu="950" id378 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id378 -->

</div>

## Locally disabling gradient computation

### torch.no_grad

<div style="margin-left: 2em">

**Native Documentation**: [torch.no_grad](https://pytorch.org/docs/2.7/generated/torch.no_grad.html)

**Support Status**:

<!-- npu="910b" id379 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id379 -->
<!-- npu="A3" id380 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id380 -->
<!-- npu="950" id381 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id381 -->

</div>

### torch.enable_grad

<div style="margin-left: 2em">

**Native Documentation**: [torch.enable_grad](https://pytorch.org/docs/2.7/generated/torch.enable_grad.html)

**Support Status**:

<!-- npu="910b" id382 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id382 -->
<!-- npu="A3" id383 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id383 -->
<!-- npu="950" id384 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id384 -->

</div>

### torch.autograd.grad_mode.set_grad_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.grad_mode.set_grad_enabled](https://pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.set_grad_enabled.html)

**Support Status**:

<!-- npu="910b" id385 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id385 -->
<!-- npu="A3" id386 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id386 -->
<!-- npu="950" id387 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id387 -->

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.grad_mode.set_grad_enabled.clone](https://pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.set_grad_enabled.html#torch.autograd.grad_mode.set_grad_enabled.clone)

**Support Status**:

<!-- npu="910b" id388 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id388 -->
<!-- npu="A3" id389 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id389 -->
<!-- npu="950" id390 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id390 -->

</div>

</div>

### torch.is_grad_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_grad_enabled](https://pytorch.org/docs/2.7/generated/torch.is_grad_enabled.html)

**Support Status**:

<!-- npu="910b" id391 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id391 -->
<!-- npu="A3" id392 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id392 -->
<!-- npu="950" id393 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id393 -->

</div>

### torch.autograd.grad_mode.inference_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.grad_mode.inference_mode](https://pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.inference_mode.html)

**Support Status**:

<!-- npu="910b" id394 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id394 -->
<!-- npu="A3" id395 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id395 -->
<!-- npu="950" id396 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id396 -->

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.grad_mode.inference_mode.clone](https://pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.inference_mode.html#torch.autograd.grad_mode.inference_mode.clone)

**Support Status**:

<!-- npu="910b" id397 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id397 -->
<!-- npu="A3" id398 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id398 -->
<!-- npu="950" id399 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id399 -->

</div>

</div>

### torch.is_inference_mode_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_inference_mode_enabled](https://pytorch.org/docs/2.7/generated/torch.is_inference_mode_enabled.html)

**Support Status**:

<!-- npu="910b" id400 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id400 -->
<!-- npu="A3" id401 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id401 -->
<!-- npu="950" id402 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id402 -->

</div>

## Math operations

### torch.abs

<div style="margin-left: 2em">

**Native Documentation**: [torch.abs](https://pytorch.org/docs/2.7/generated/torch.abs.html)

**Support Status**:

<!-- npu="910b" id403 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id403 -->
<!-- npu="A3" id404 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id404 -->
<!-- npu="950" id405 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id405 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.angle

<div style="margin-left: 2em">

**Native Documentation**: [torch.angle](https://pytorch.org/docs/2.7/generated/torch.angle.html)

**Support Status**:

<!-- npu="910b" id406 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id406 -->
<!-- npu="A3" id407 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id407 -->
<!-- npu="950" id408 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id408 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

### torch.real

<div style="margin-left: 2em">

**Native Documentation**: [torch.real](https://pytorch.org/docs/2.7/generated/torch.real.html)

**Support Status**:

<!-- npu="910b" id409 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id409 -->
<!-- npu="A3" id410 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id410 -->
<!-- npu="950" id411 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id411 -->

**Constraints & Notes**: `input` only supports fp16, fp32, complex64, complex128

</div>

### torch.absolute

<div style="margin-left: 2em">

**Native Documentation**: [torch.absolute](https://pytorch.org/docs/2.7/generated/torch.absolute.html)

**Support Status**:

<!-- npu="910b" id412 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id412 -->
<!-- npu="A3" id413 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id413 -->
<!-- npu="950" id414 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id414 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.acos

<div style="margin-left: 2em">

**Native Documentation**: [torch.acos](https://pytorch.org/docs/2.7/generated/torch.acos.html)

**Support Status**:

<!-- npu="910b" id415 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id415 -->
<!-- npu="A3" id416 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id416 -->
<!-- npu="950" id417 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id417 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.arccos

<div style="margin-left: 2em">

**Native Documentation**: [torch.arccos](https://pytorch.org/docs/2.7/generated/torch.arccos.html)

**Support Status**:

<!-- npu="910b" id418 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id418 -->
<!-- npu="A3" id419 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id419 -->
<!-- npu="950" id420 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id420 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.acosh

<div style="margin-left: 2em">

**Native Documentation**: [torch.acosh](https://pytorch.org/docs/2.7/generated/torch.acosh.html)

**Support Status**:

<!-- npu="910b" id421 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id421 -->
<!-- npu="A3" id422 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id422 -->
<!-- npu="950" id423 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id423 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- May fall back to CPU execution

</div>

### torch.arccosh

<div style="margin-left: 2em">

**Native Documentation**: [torch.arccosh](https://pytorch.org/docs/2.7/generated/torch.arccosh.html)

**Support Status**:

<!-- npu="910b" id424 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id424 -->
<!-- npu="A3" id425 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id425 -->
<!-- npu="950" id426 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id426 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.add

<div style="margin-left: 2em">

**Native Documentation**: [torch.add](https://pytorch.org/docs/2.7/generated/torch.add.html)

**Support Status**:

<!-- npu="910b" id427 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id427 -->
<!-- npu="A3" id428 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id428 -->
<!-- npu="950" id429 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id429 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.addcdiv

<div style="margin-left: 2em">

**Native Documentation**: [torch.addcdiv](https://pytorch.org/docs/2.7/generated/torch.addcdiv.html)

**Support Status**:

<!-- npu="910b" id430 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id430 -->
<!-- npu="A3" id431 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id431 -->
<!-- npu="950" id432 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id432 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32

</div>

### torch.addcmul

<div style="margin-left: 2em">

**Native Documentation**: [torch.addcmul](https://pytorch.org/docs/2.7/generated/torch.addcmul.html)

**Support Status**:

<!-- npu="910b" id433 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id433 -->
<!-- npu="A3" id434 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id434 -->
<!-- npu="950" id435 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id435 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64
- fp64, uint8, int8, and int64 types do not support simultaneous broadcasting of three `tensor`s

</div>

### torch.asin

<div style="margin-left: 2em">

**Native Documentation**: [torch.asin](https://pytorch.org/docs/2.7/generated/torch.asin.html)

**Support Status**:

<!-- npu="910b" id436 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id436 -->
<!-- npu="A3" id437 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id437 -->
<!-- npu="950" id438 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id438 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.arcsin

<div style="margin-left: 2em">

**Native Documentation**: [torch.arcsin](https://pytorch.org/docs/2.7/generated/torch.arcsin.html)

**Support Status**:

<!-- npu="910b" id439 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id439 -->
<!-- npu="A3" id440 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id440 -->
<!-- npu="950" id441 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id441 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.asinh

<div style="margin-left: 2em">

**Native Documentation**: [torch.asinh](https://pytorch.org/docs/2.7/generated/torch.asinh.html)

**Support Status**:

<!-- npu="910b" id442 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id442 -->
<!-- npu="A3" id443 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id443 -->
<!-- npu="950" id444 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id444 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.arcsinh

<div style="margin-left: 2em">

**Native Documentation**: [torch.arcsinh](https://pytorch.org/docs/2.7/generated/torch.arcsinh.html)

**Support Status**:

<!-- npu="910b" id445 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id445 -->
<!-- npu="A3" id446 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id446 -->
<!-- npu="950" id447 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id447 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.atan

<div style="margin-left: 2em">

**Native Documentation**: [torch.atan](https://pytorch.org/docs/2.7/generated/torch.atan.html)

**Support Status**:

<!-- npu="910b" id448 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id448 -->
<!-- npu="A3" id449 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id449 -->
<!-- npu="950" id450 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id450 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.arctan

<div style="margin-left: 2em">

**Native Documentation**: [torch.arctan](https://pytorch.org/docs/2.7/generated/torch.arctan.html)

**Support Status**:

<!-- npu="910b" id451 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id451 -->
<!-- npu="A3" id452 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id452 -->
<!-- npu="950" id453 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id453 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.atanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.atanh](https://pytorch.org/docs/2.7/generated/torch.atanh.html)

**Support Status**:

<!-- npu="910b" id454 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id454 -->
<!-- npu="A3" id455 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id455 -->
<!-- npu="950" id456 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id456 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.arctanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.arctanh](https://pytorch.org/docs/2.7/generated/torch.arctanh.html)

**Support Status**:

<!-- npu="910b" id457 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id457 -->
<!-- npu="A3" id458 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id458 -->
<!-- npu="950" id459 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id459 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.atan2

<div style="margin-left: 2em">

**Native Documentation**: [torch.atan2](https://pytorch.org/docs/2.7/generated/torch.atan2.html)

**Support Status**:

<!-- npu="910b" id460 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id460 -->
<!-- npu="A3" id461 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id461 -->
<!-- npu="950" id462 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id462 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.arctan2

<div style="margin-left: 2em">

**Native Documentation**: [torch.arctan2](https://pytorch.org/docs/2.7/generated/torch.arctan2.html)

**Support Status**:

<!-- npu="910b" id463 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id463 -->
<!-- npu="A3" id464 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id464 -->
<!-- npu="950" id465 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id465 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.bitwise_not

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_not](https://pytorch.org/docs/2.7/generated/torch.bitwise_not.html)

**Support Status**:

<!-- npu="910b" id466 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id466 -->
<!-- npu="A3" id467 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id467 -->
<!-- npu="950" id468 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id468 -->

**Constraints & Notes**: `input` only supports uint8, int8, int16, int32, int64, bool

</div>

### torch.bitwise_and

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_and](https://pytorch.org/docs/2.7/generated/torch.bitwise_and.html)

**Support Status**:

<!-- npu="910b" id469 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id469 -->
<!-- npu="A3" id470 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id470 -->
<!-- npu="950" id471 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id471 -->

**Constraints & Notes**: `input` only supports uint8, int8, int16, int32, int64, bool

</div>

### torch.bitwise_or

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_or](https://pytorch.org/docs/2.7/generated/torch.bitwise_or.html)

**Support Status**:

<!-- npu="910b" id472 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id472 -->
<!-- npu="A3" id473 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id473 -->
<!-- npu="950" id474 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id474 -->

**Constraints & Notes**: `input` only supports uint8, int8, int16, int32, int64, bool

</div>

### torch.bitwise_xor

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_xor](https://pytorch.org/docs/2.7/generated/torch.bitwise_xor.html)

**Support Status**:

<!-- npu="910b" id475 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id475 -->
<!-- npu="A3" id476 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id476 -->
<!-- npu="950" id477 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id477 -->

**Constraints & Notes**: `input` only supports uint8, int8, int16, int32, int64, bool

</div>

### torch.bitwise_left_shift

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_left_shift](https://pytorch.org/docs/2.7/generated/torch.bitwise_left_shift.html)

**Support Status**:

<!-- npu="910b" id478 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id478 -->
<!-- npu="A3" id479 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id479 -->
<!-- npu="950" id480 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id480 -->

**Constraints & Notes**:

- `input` only supports uint8, int8, int16, int32, int64

- Precision is guaranteed to be error-free only when the value of shiftBits is less than the bit width of the self data type

</div>

### torch.bitwise_right_shift

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_right_shift](https://pytorch.org/docs/2.7/generated/torch.bitwise_right_shift.html)

**Support Status**:

<!-- npu="910b" id481 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id481 -->
<!-- npu="A3" id482 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id482 -->
<!-- npu="950" id483 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id483 -->

**Constraints & Notes**:

- `input` only supports uint8, int8, int16, int32, int64

- Precision is guaranteed to be error-free only when the value of shiftBits is less than the bit width of the self data type

</div>

### torch.ceil

<div style="margin-left: 2em">

**Native Documentation**: [torch.ceil](https://pytorch.org/docs/2.7/generated/torch.ceil.html)

**Support Status**:

<!-- npu="910b" id484 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id484 -->
<!-- npu="A3" id485 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id485 -->
<!-- npu="950" id486 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id486 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.clamp

<div style="margin-left: 2em">

**Native Documentation**: [torch.clamp](https://pytorch.org/docs/2.7/generated/torch.clamp.html)

**Support Status**:

<!-- npu="910b" id487 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id487 -->
<!-- npu="A3" id488 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id488 -->
<!-- npu="950" id489 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id489 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.clip

<div style="margin-left: 2em">

**Native Documentation**: [torch.clip](https://pytorch.org/docs/2.7/generated/torch.clip.html)

**Support Status**:

<!-- npu="910b" id490 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id490 -->
<!-- npu="A3" id491 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id491 -->
<!-- npu="950" id492 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id492 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.copysign

<div style="margin-left: 2em">

**Native Documentation**: [torch.copysign](https://pytorch.org/docs/2.7/generated/torch.copysign.html)

**Support Status**:

<!-- npu="910b" id493 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id493 -->
<!-- npu="A3" id494 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id494 -->
<!-- npu="950" id495 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id495 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

### torch.cos

<div style="margin-left: 2em">

**Native Documentation**: [torch.cos](https://pytorch.org/docs/2.7/generated/torch.cos.html)

**Support Status**:

<!-- npu="910b" id496 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id496 -->
<!-- npu="A3" id497 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id497 -->
<!-- npu="950" id498 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id498 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.cosh

<div style="margin-left: 2em">

**Native Documentation**: [torch.cosh](https://pytorch.org/docs/2.7/generated/torch.cosh.html)

**Support Status**:

<!-- npu="910b" id499 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id499 -->
<!-- npu="A3" id500 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id500 -->
<!-- npu="950" id501 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id501 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.deg2rad

<div style="margin-left: 2em">

**Native Documentation**: [torch.deg2rad](https://pytorch.org/docs/2.7/generated/torch.deg2rad.html)

**Support Status**:

<!-- npu="910b" id502 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id502 -->
<!-- npu="A3" id503 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id503 -->
<!-- npu="950" id504 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id504 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.div

<div style="margin-left: 2em">

**Native Documentation**: [torch.div](https://pytorch.org/docs/2.7/generated/torch.div.html)

**Support Status**:

<!-- npu="910b" id505 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id505 -->
<!-- npu="A3" id506 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id506 -->
<!-- npu="950" id507 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id507 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.divide

<div style="margin-left: 2em">

**Native Documentation**: [torch.divide](https://pytorch.org/docs/2.7/generated/torch.divide.html)

**Support Status**:

<!-- npu="910b" id508 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id508 -->
<!-- npu="A3" id509 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id509 -->
<!-- npu="950" id510 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id510 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.erf

<div style="margin-left: 2em">

**Native Documentation**: [torch.erf](https://pytorch.org/docs/2.7/generated/torch.erf.html)

**Support Status**:

<!-- npu="910b" id511 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id511 -->
<!-- npu="A3" id512 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id512 -->
<!-- npu="950" id513 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id513 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int64, bool

</div>

### torch.erfc

<div style="margin-left: 2em">

**Native Documentation**: [torch.erfc](https://pytorch.org/docs/2.7/generated/torch.erfc.html)

**Support Status**:

<!-- npu="910b" id514 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id514 -->
<!-- npu="A3" id515 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id515 -->
<!-- npu="950" id516 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id516 -->

**Constraints & Notes**: `input` only supports fp16, fp32, int64, bool

</div>

### torch.erfinv

<div style="margin-left: 2em">

**Native Documentation**: [torch.erfinv](https://pytorch.org/docs/2.7/generated/torch.erfinv.html)

**Support Status**:

<!-- npu="910b" id517 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id517 -->
<!-- npu="A3" id518 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id518 -->
<!-- npu="950" id519 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id519 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.exp

<div style="margin-left: 2em">

**Native Documentation**: [torch.exp](https://pytorch.org/docs/2.7/generated/torch.exp.html)

**Support Status**:

<!-- npu="910b" id520 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id520 -->
<!-- npu="A3" id521 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id521 -->
<!-- npu="950" id522 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id522 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int64, bool

</div>

### torch.exp2

<div style="margin-left: 2em">

**Native Documentation**: [torch.exp2](https://pytorch.org/docs/2.7/generated/torch.exp2.html)

**Support Status**:

<!-- npu="910b" id523 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id523 -->
<!-- npu="A3" id524 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id524 -->
<!-- npu="950" id525 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id525 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.expm1

<div style="margin-left: 2em">

**Native Documentation**: [torch.expm1](https://pytorch.org/docs/2.7/generated/torch.expm1.html)

**Support Status**:

<!-- npu="910b" id526 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id526 -->
<!-- npu="A3" id527 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id527 -->
<!-- npu="950" id528 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id528 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int64, bool

</div>

### torch.fix

<div style="margin-left: 2em">

**Native Documentation**: [torch.fix](https://pytorch.org/docs/2.7/generated/torch.fix.html)

**Support Status**:

<!-- npu="910b" id529 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id529 -->
<!-- npu="A3" id530 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id530 -->
<!-- npu="950" id531 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id531 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.float_power

<div style="margin-left: 2em">

**Native Documentation**: [torch.float_power](https://pytorch.org/docs/2.7/generated/torch.float_power.html)

**Support Status**:

<!-- npu="910b" id532 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id532 -->
<!-- npu="A3" id533 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id533 -->
<!-- npu="950" id534 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id534 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex128

</div>

### torch.floor

<div style="margin-left: 2em">

**Native Documentation**: [torch.floor](https://pytorch.org/docs/2.7/generated/torch.floor.html)

**Support Status**:

<!-- npu="910b" id535 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id535 -->
<!-- npu="A3" id536 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id536 -->
<!-- npu="950" id537 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id537 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.floor_divide

<div style="margin-left: 2em">

**Native Documentation**: [torch.floor_divide](https://pytorch.org/docs/2.7/generated/torch.floor_divide.html)

**Support Status**:

<!-- npu="910b" id538 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id538 -->
<!-- npu="A3" id539 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id539 -->
<!-- npu="950" id540 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id540 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.fmod

<div style="margin-left: 2em">

**Native Documentation**: [torch.fmod](https://pytorch.org/docs/2.7/generated/torch.fmod.html)

**Support Status**:

<!-- npu="910b" id541 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id541 -->
<!-- npu="A3" id542 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id542 -->
<!-- npu="950" id543 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id543 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

### torch.gradient

<div style="margin-left: 2em">

**Native Documentation**: [torch.gradient](https://pytorch.org/docs/2.7/generated/torch.gradient.html)

**Support Status**:

<!-- npu="910b" id544 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id544 -->
<!-- npu="A3" id545 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id545 -->
<!-- npu="950" id546 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id546 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int8, int16, int32, int64

</div>

### torch.ldexp

<div style="margin-left: 2em">

**Native Documentation**: [torch.ldexp](https://pytorch.org/docs/2.7/generated/torch.ldexp.html)

**Support Status**:

<!-- npu="910b" id547 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id547 -->
<!-- npu="A3" id548 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id548 -->
<!-- npu="950" id549 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id549 -->

**Constraints & Notes**: `input` only supports fp16, fp64, complex64

</div>

### torch.lerp

<div style="margin-left: 2em">

**Native Documentation**: [torch.lerp](https://pytorch.org/docs/2.7/generated/torch.lerp.html)

**Support Status**:

<!-- npu="910b" id550 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id550 -->
<!-- npu="A3" id551 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id551 -->
<!-- npu="950" id552 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id552 -->

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.log

<div style="margin-left: 2em">

**Native Documentation**: [torch.log](https://pytorch.org/docs/2.7/generated/torch.log.html)

**Support Status**:

<!-- npu="910b" id553 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id553 -->
<!-- npu="A3" id554 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id554 -->
<!-- npu="950" id555 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id555 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.log10

<div style="margin-left: 2em">

**Native Documentation**: [torch.log10](https://pytorch.org/docs/2.7/generated/torch.log10.html)

**Support Status**:

<!-- npu="910b" id556 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id556 -->
<!-- npu="A3" id557 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id557 -->
<!-- npu="950" id558 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id558 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- When the input `input` is uint8, int8, int16, int32, int64, or bool, the output `out` must be fp32
- For other supported data types, the output `out` must be consistent with the input `input`

</div>

### torch.log1p

<div style="margin-left: 2em">

**Native Documentation**: [torch.log1p](https://pytorch.org/docs/2.7/generated/torch.log1p.html)

**Support Status**:

<!-- npu="910b" id559 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id559 -->
<!-- npu="A3" id560 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id560 -->
<!-- npu="950" id561 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id561 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.log2

<div style="margin-left: 2em">

**Native Documentation**: [torch.log2](https://pytorch.org/docs/2.7/generated/torch.log2.html)

**Support Status**:

<!-- npu="910b" id562 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id562 -->
<!-- npu="A3" id563 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id563 -->
<!-- npu="950" id564 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id564 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.logaddexp

<div style="margin-left: 2em">

**Native Documentation**: [torch.logaddexp](https://pytorch.org/docs/2.7/generated/torch.logaddexp.html)

**Support Status**:

<!-- npu="910b" id565 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id565 -->
<!-- npu="A3" id566 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id566 -->
<!-- npu="950" id567 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id567 -->

**Constraints & Notes**: Does not support the double data type.

</div>

### torch.logaddexp2

<div style="margin-left: 2em">

**Native Documentation**: [torch.logaddexp2](https://pytorch.org/docs/2.7/generated/torch.logaddexp2.html)

**Support Status**:

<!-- npu="910b" id568 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id568 -->
<!-- npu="A3" id569 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id569 -->
<!-- npu="950" id570 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id570 -->

**Constraints & Notes**: Does not support the double data type.

</div>

### torch.logical_and

<div style="margin-left: 2em">

**Native Documentation**: [torch.logical_and](https://pytorch.org/docs/2.7/generated/torch.logical_and.html)

**Support Status**:

<!-- npu="910b" id571 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id571 -->
<!-- npu="A3" id572 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id572 -->
<!-- npu="950" id573 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id573 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.logical_not

<div style="margin-left: 2em">

**Native Documentation**: [torch.logical_not](https://pytorch.org/docs/2.7/generated/torch.logical_not.html)

**Support Status**:

<!-- npu="910b" id574 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id574 -->
<!-- npu="A3" id575 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id575 -->
<!-- npu="950" id576 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id576 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.logical_or

<div style="margin-left: 2em">

**Native Documentation**: [torch.logical_or](https://pytorch.org/docs/2.7/generated/torch.logical_or.html)

**Support Status**:

<!-- npu="910b" id577 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id577 -->
<!-- npu="A3" id578 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id578 -->
<!-- npu="950" id579 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id579 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.logical_xor

<div style="margin-left: 2em">

**Native Documentation**: [torch.logical_xor](https://pytorch.org/docs/2.7/generated/torch.logical_xor.html)

**Support Status**:

<!-- npu="910b" id580 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id580 -->
<!-- npu="A3" id581 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id581 -->
<!-- npu="950" id582 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id582 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.logit

<div style="margin-left: 2em">

**Native Documentation**: [torch.logit](https://pytorch.org/docs/2.7/generated/torch.logit.html)

**Support Status**:

<!-- npu="910b" id583 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id583 -->
<!-- npu="A3" id584 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id584 -->
<!-- npu="950" id585 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id585 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- When the value of `eps` is greater than 1, the output is nan; when the value of `eps` is 1, the output is inf

</div>

### torch.mul

<div style="margin-left: 2em">

**Native Documentation**: [torch.mul](https://pytorch.org/docs/2.7/generated/torch.mul.html)

**Support Status**:

<!-- npu="910b" id586 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id586 -->
<!-- npu="A3" id587 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id587 -->
<!-- npu="950" id588 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id588 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.multiply

<div style="margin-left: 2em">

**Native Documentation**: [torch.multiply](https://pytorch.org/docs/2.7/generated/torch.multiply.html)

**Support Status**:

<!-- npu="910b" id589 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id589 -->
<!-- npu="A3" id590 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id590 -->
<!-- npu="950" id591 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id591 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nan_to_num

<div style="margin-left: 2em">

**Native Documentation**: [torch.nan_to_num](https://pytorch.org/docs/2.7/generated/torch.nan_to_num.html)

**Support Status**:

<!-- npu="910b" id592 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id592 -->
<!-- npu="A3" id593 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id593 -->
<!-- npu="950" id594 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id594 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.neg

<div style="margin-left: 2em">

**Native Documentation**: [torch.neg](https://pytorch.org/docs/2.7/generated/torch.neg.html)

**Support Status**:

<!-- npu="910b" id595 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id595 -->
<!-- npu="A3" id596 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id596 -->
<!-- npu="950" id597 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id597 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int8, int32, int64

</div>

### torch.negative

<div style="margin-left: 2em">

**Native Documentation**: [torch.negative](https://pytorch.org/docs/2.7/generated/torch.negative.html)

**Support Status**:

<!-- npu="910b" id598 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id598 -->
<!-- npu="A3" id599 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id599 -->
<!-- npu="950" id600 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id600 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128

</div>

### torch.positive

<div style="margin-left: 2em">

**Native Documentation**: [torch.positive](https://pytorch.org/docs/2.7/generated/torch.positive.html)

**Support Status**:

<!-- npu="910b" id601 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id601 -->
<!-- npu="A3" id602 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id602 -->
<!-- npu="950" id603 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id603 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128

</div>

### torch.pow

<div style="margin-left: 2em">

**Native Documentation**: [torch.pow](https://pytorch.org/docs/2.7/generated/torch.pow.html)

**Support Status**:

<!-- npu="910b" id604 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id604 -->
<!-- npu="A3" id605 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id605 -->
<!-- npu="950" id606 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id606 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.rad2deg

<div style="margin-left: 2em">

**Native Documentation**: [torch.rad2deg](https://pytorch.org/docs/2.7/generated/torch.rad2deg.html)

**Support Status**:

<!-- npu="910b" id607 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id607 -->
<!-- npu="A3" id608 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id608 -->
<!-- npu="950" id609 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id609 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.reciprocal

<div style="margin-left: 2em">

**Native Documentation**: [torch.reciprocal](https://pytorch.org/docs/2.7/generated/torch.reciprocal.html)

**Support Status**:

<!-- npu="910b" id610 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id610 -->
<!-- npu="A3" id611 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id611 -->
<!-- npu="950" id612 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id612 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.remainder

<div style="margin-left: 2em">

**Native Documentation**: [torch.remainder](https://pytorch.org/docs/2.7/generated/torch.remainder.html)

**Support Status**:

<!-- npu="910b" id613 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id613 -->
<!-- npu="A3" id614 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id614 -->
<!-- npu="950" id615 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id615 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int32, int64

</div>

### torch.round

<div style="margin-left: 2em">

**Native Documentation**: [torch.round](https://pytorch.org/docs/2.7/generated/torch.round.html)

**Support Status**:

<!-- npu="910b" id616 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id616 -->
<!-- npu="A3" id617 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id617 -->
<!-- npu="950" id618 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id618 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int32, int64

</div>

### torch.rsqrt

<div style="margin-left: 2em">

**Native Documentation**: [torch.rsqrt](https://pytorch.org/docs/2.7/generated/torch.rsqrt.html)

**Support Status**:

<!-- npu="910b" id619 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id619 -->
<!-- npu="A3" id620 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id620 -->
<!-- npu="950" id621 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id621 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.sigmoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.sigmoid](https://pytorch.org/docs/2.7/generated/torch.sigmoid.html)

**Support Status**:

<!-- npu="910b" id622 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id622 -->
<!-- npu="A3" id623 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id623 -->
<!-- npu="950" id624 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id624 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.sign

<div style="margin-left: 2em">

**Native Documentation**: [torch.sign](https://pytorch.org/docs/2.7/generated/torch.sign.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int32, int64, bool

</div>

### torch.sgn

<div style="margin-left: 2em">

**Native Documentation**: [torch.sgn](https://pytorch.org/docs/2.7/generated/torch.sgn.html)

**Support Status**:

<!-- npu="910b" id628 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id628 -->
<!-- npu="A3" id629 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id629 -->
<!-- npu="950" id630 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id630 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int32, int64, bool, complex64, complex128

</div>

### torch.sin

<div style="margin-left: 2em">

**Native Documentation**: [torch.sin](https://pytorch.org/docs/2.7/generated/torch.sin.html)

**Support Status**:

<!-- npu="910b" id631 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id631 -->
<!-- npu="A3" id632 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id632 -->
<!-- npu="950" id633 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id633 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.sinh

<div style="margin-left: 2em">

**Native Documentation**: [torch.sinh](https://pytorch.org/docs/2.7/generated/torch.sinh.html)

**Support Status**:

<!-- npu="910b" id634 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id634 -->
<!-- npu="A3" id635 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id635 -->
<!-- npu="950" id636 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id636 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64

</div>

### torch.softmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.softmax](https://pytorch.org/docs/2.7/generated/torch.softmax.html)

**Support Status**:

<!-- npu="910b" id637 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id637 -->
<!-- npu="A3" id638 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id638 -->
<!-- npu="950" id639 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id639 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64
- Named Tensor is supported

</div>

### torch.sqrt

<div style="margin-left: 2em">

**Native Documentation**: [torch.sqrt](https://pytorch.org/docs/2.7/generated/torch.sqrt.html)

**Support Status**:

<!-- npu="910b" id640 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id640 -->
<!-- npu="A3" id641 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id641 -->
<!-- npu="950" id642 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id642 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.square

<div style="margin-left: 2em">

**Native Documentation**: [torch.square](https://pytorch.org/docs/2.7/generated/torch.square.html)

**Support Status**:

<!-- npu="910b" id643 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id643 -->
<!-- npu="A3" id644 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id644 -->
<!-- npu="950" id645 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id645 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.sub

<div style="margin-left: 2em">

**Native Documentation**: [torch.sub](https://pytorch.org/docs/2.7/generated/torch.sub.html)

**Support Status**:

<!-- npu="910b" id646 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id646 -->
<!-- npu="A3" id647 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id647 -->
<!-- npu="950" id648 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id648 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.tan

<div style="margin-left: 2em">

**Native Documentation**: [torch.tan](https://pytorch.org/docs/2.7/generated/torch.tan.html)

**Support Status**:

<!-- npu="910b" id649 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id649 -->
<!-- npu="A3" id650 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id650 -->
<!-- npu="950" id651 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id651 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- The value range is [-65504, 65504]

</div>

### torch.tanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.tanh](https://pytorch.org/docs/2.7/generated/torch.tanh.html)

**Support Status**:

<!-- npu="910b" id652 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id652 -->
<!-- npu="A3" id653 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id653 -->
<!-- npu="950" id654 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id654 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.true_divide

<div style="margin-left: 2em">

**Native Documentation**: [torch.true_divide](https://pytorch.org/docs/2.7/generated/torch.true_divide.html)

**Support Status**:

<!-- npu="910b" id655 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id655 -->
<!-- npu="A3" id656 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id656 -->
<!-- npu="950" id657 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id657 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.trunc

<div style="margin-left: 2em">

**Native Documentation**: [torch.trunc](https://pytorch.org/docs/2.7/generated/torch.trunc.html)

**Support Status**:

<!-- npu="910b" id658 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id658 -->
<!-- npu="A3" id659 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id659 -->
<!-- npu="950" id660 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id660 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32
- May fall back to CPU execution

</div>

### torch.xlogy

<div style="margin-left: 2em">

**Native Documentation**: [torch.xlogy](https://pytorch.org/docs/2.7/generated/torch.xlogy.html)

**Support Status**:

<!-- npu="910b" id661 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id661 -->
<!-- npu="A3" id662 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id662 -->
<!-- npu="950" id663 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id663 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.argmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.argmax](https://pytorch.org/docs/2.7/generated/torch.argmax.html)

**Support Status**:

<!-- npu="910b" id664 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id664 -->
<!-- npu="A3" id665 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id665 -->
<!-- npu="950" id666 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id666 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.argmin

<div style="margin-left: 2em">

**Native Documentation**: [torch.argmin](https://pytorch.org/docs/2.7/generated/torch.argmin.html)

**Support Status**:

<!-- npu="910b" id667 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id667 -->
<!-- npu="A3" id668 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id668 -->
<!-- npu="950" id669 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id669 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.amax

<div style="margin-left: 2em">

**Native Documentation**: [torch.amax](https://pytorch.org/docs/2.7/generated/torch.amax.html)

**Support Status**:

<!-- npu="910b" id670 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id670 -->
<!-- npu="A3" id671 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id671 -->
<!-- npu="950" id672 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id672 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.amin

<div style="margin-left: 2em">

**Native Documentation**: [torch.amin](https://pytorch.org/docs/2.7/generated/torch.amin.html)

**Support Status**:

<!-- npu="910b" id673 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id673 -->
<!-- npu="A3" id674 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id674 -->
<!-- npu="950" id675 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id675 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.aminmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.aminmax](https://pytorch.org/docs/2.7/generated/torch.aminmax.html)

**Support Status**:

<!-- npu="910b" id676 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id676 -->
<!-- npu="A3" id677 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id677 -->
<!-- npu="950" id678 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id678 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.all

<div style="margin-left: 2em">

**Native Documentation**: [torch.all](https://pytorch.org/docs/2.7/generated/torch.all.html)

**Support Status**:

<!-- npu="910b" id679 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id679 -->
<!-- npu="A3" id680 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id680 -->
<!-- npu="950" id681 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id681 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.any

<div style="margin-left: 2em">

**Native Documentation**: [torch.any](https://pytorch.org/docs/2.7/generated/torch.any.html)

**Support Status**:

<!-- npu="910b" id682 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id682 -->
<!-- npu="A3" id683 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id683 -->
<!-- npu="950" id684 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id684 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.max

<div style="margin-left: 2em">

**Native Documentation**: [torch.max](https://pytorch.org/docs/2.7/generated/torch.max.html)

**Support Status**:

<!-- npu="910b" id685 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id685 -->
<!-- npu="A3" id686 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id686 -->
<!-- npu="950" id687 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id687 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int64, bool

</div>

### torch.min

<div style="margin-left: 2em">

**Native Documentation**: [torch.min](https://pytorch.org/docs/2.7/generated/torch.min.html)

**Support Status**:

<!-- npu="910b" id688 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id688 -->
<!-- npu="A3" id689 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id689 -->
<!-- npu="950" id690 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id690 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int64, bool

</div>

### torch.dist

<div style="margin-left: 2em">

**Native Documentation**: [torch.dist](https://pytorch.org/docs/2.7/generated/torch.dist.html)

**Support Status**:

<!-- npu="910b" id691 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id691 -->
<!-- npu="A3" id692 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id692 -->
<!-- npu="950" id693 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id693 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.logsumexp

<div style="margin-left: 2em">

**Native Documentation**: [torch.logsumexp](https://pytorch.org/docs/2.7/generated/torch.logsumexp.html)

**Support Status**:

<!-- npu="910b" id694 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id694 -->
<!-- npu="A3" id695 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id695 -->
<!-- npu="950" id696 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id696 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.mean

<div style="margin-left: 2em">

**Native Documentation**: [torch.mean](https://pytorch.org/docs/2.7/generated/torch.mean.html)

**Support Status**:

<!-- npu="910b" id697 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id697 -->
<!-- npu="A3" id698 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id698 -->
<!-- npu="950" id699 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id699 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, complex64, complex128

</div>

### torch.nanmean

<div style="margin-left: 2em">

**Native Documentation**: [torch.nanmean](https://pytorch.org/docs/2.7/generated/torch.nanmean.html)

**Support Status**:

<!-- npu="910b" id700 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id700 -->
<!-- npu="A3" id701 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id701 -->
<!-- npu="950" id702 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id702 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.median

<div style="margin-left: 2em">

**Native Documentation**: [torch.median](https://pytorch.org/docs/2.7/generated/torch.median.html)

**Support Status**:

<!-- npu="910b" id703 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id703 -->
<!-- npu="A3" id704 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id704 -->
<!-- npu="950" id705 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id705 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.norm](https://pytorch.org/docs/2.7/generated/torch.norm.html)

**Support Status**:

<!-- npu="910b" id706 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id706 -->
<!-- npu="A3" id707 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id707 -->
<!-- npu="950" id708 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id708 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- When the `dim` parameter specifies an axis whose shape dimension value in the input `tensor` is 1, the computation result may have precision errors

</div>

### torch.nansum

<div style="margin-left: 2em">

**Native Documentation**: [torch.nansum](https://pytorch.org/docs/2.7/generated/torch.nansum.html)

**Support Status**:

<!-- npu="910b" id709 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id709 -->
<!-- npu="A3" id710 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id710 -->
<!-- npu="950" id711 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id711 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.prod

<div style="margin-left: 2em">

**Native Documentation**: [torch.prod](https://pytorch.org/docs/2.7/generated/torch.prod.html)

**Support Status**:

<!-- npu="910b" id712 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id712 -->
<!-- npu="A3" id713 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id713 -->
<!-- npu="950" id714 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id714 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nanquantile

<div style="margin-left: 2em">

**Native Documentation**: [torch.nanquantile](https://pytorch.org/docs/2.7/generated/torch.nanquantile.html)

**Support Status**:

<!-- npu="910b" id715 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id715 -->
<!-- npu="A3" id716 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id716 -->
<!-- npu="950" id717 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id717 -->

</div>

### torch.std

<div style="margin-left: 2em">

**Native Documentation**: [torch.std](https://pytorch.org/docs/2.7/generated/torch.std.html)

**Support Status**:

<!-- npu="910b" id718 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id718 -->
<!-- npu="A3" id719 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id719 -->
<!-- npu="950" id720 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id720 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

### torch.std_mean

<div style="margin-left: 2em">

**Native Documentation**: [torch.std_mean](https://pytorch.org/docs/2.7/generated/torch.std_mean.html)

**Support Status**:

<!-- npu="910b" id721 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id721 -->
<!-- npu="A3" id722 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id722 -->
<!-- npu="950" id723 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id723 -->

</div>

### torch.sum

<div style="margin-left: 2em">

**Native Documentation**: [torch.sum](https://pytorch.org/docs/2.7/generated/torch.sum.html)

**Support Status**:

<!-- npu="910b" id724 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id724 -->
<!-- npu="A3" id725 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id725 -->
<!-- npu="950" id726 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id726 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- The `dtype` parameter is not supported

</div>

### torch.unique

<div style="margin-left: 2em">

**Native Documentation**: [torch.unique](https://pytorch.org/docs/2.7/generated/torch.unique.html)

**Support Status**:

<!-- npu="910b" id727 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id727 -->
<!-- npu="A3" id728 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id728 -->
<!-- npu="950" id729 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id729 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- fp16 is not supported in the scenario with `dim`
- When the input contains 0, the output may contain positive 0 and negative 0 instead of only one 0

</div>

### torch.unique_consecutive

<div style="margin-left: 2em">

**Native Documentation**: [torch.unique_consecutive](https://pytorch.org/docs/2.7/generated/torch.unique_consecutive.html)

**Support Status**:

<!-- npu="910b" id730 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id730 -->
<!-- npu="A3" id731 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id731 -->
<!-- npu="950" id732 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id732 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.var

<div style="margin-left: 2em">

**Native Documentation**: [torch.var](https://pytorch.org/docs/2.7/generated/torch.var.html)

**Support Status**:

<!-- npu="910b" id733 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id733 -->
<!-- npu="A3" id734 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id734 -->
<!-- npu="950" id735 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id735 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.var_mean

<div style="margin-left: 2em">

**Native Documentation**: [torch.var_mean](https://pytorch.org/docs/2.7/generated/torch.var_mean.html)

**Support Status**:

<!-- npu="910b" id736 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id736 -->
<!-- npu="A3" id737 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id737 -->
<!-- npu="950" id738 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id738 -->

</div>

### torch.count_nonzero

<div style="margin-left: 2em">

**Native Documentation**: [torch.count_nonzero](https://pytorch.org/docs/2.7/generated/torch.count_nonzero.html)

**Support Status**:

<!-- npu="910b" id739 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id739 -->
<!-- npu="A3" id740 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id740 -->
<!-- npu="950" id741 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id741 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.allclose

<div style="margin-left: 2em">

**Native Documentation**: [torch.allclose](https://pytorch.org/docs/2.7/generated/torch.allclose.html)

**Support Status**:

<!-- npu="910b" id742 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id742 -->
<!-- npu="A3" id743 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id743 -->
<!-- npu="950" id744 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id744 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.argsort

<div style="margin-left: 2em">

**Native Documentation**: [torch.argsort](https://pytorch.org/docs/2.7/generated/torch.argsort.html)

**Support Status**:

<!-- npu="910b" id745 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id745 -->
<!-- npu="A3" id746 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id746 -->
<!-- npu="950" id747 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id747 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

<!-- npu="950" id748 -->
- For <term>Ascend 950DT products</term>, due to underlying implementation limitations, `stable` only supports True. If set to False, it is automatically changed to True during execution
<!-- end id748 -->

</div>

### torch.eq

<div style="margin-left: 2em">

**Native Documentation**: [torch.eq](https://pytorch.org/docs/2.7/generated/torch.eq.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.equal

<div style="margin-left: 2em">

**Native Documentation**: [torch.equal](https://pytorch.org/docs/2.7/generated/torch.equal.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.ge

<div style="margin-left: 2em">

**Native Documentation**: [torch.ge](https://pytorch.org/docs/2.7/generated/torch.ge.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.greater_equal

<div style="margin-left: 2em">

**Native Documentation**: [torch.greater_equal](https://pytorch.org/docs/2.7/generated/torch.greater_equal.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.gt

<div style="margin-left: 2em">

**Native Documentation**: [torch.gt](https://pytorch.org/docs/2.7/generated/torch.gt.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.greater

<div style="margin-left: 2em">

**Native Documentation**: [torch.greater](https://pytorch.org/docs/2.7/generated/torch.greater.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.isclose

<div style="margin-left: 2em">

**Native Documentation**: [torch.isclose](https://pytorch.org/docs/2.7/generated/torch.isclose.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.isfinite

<div style="margin-left: 2em">

**Native Documentation**: [torch.isfinite](https://pytorch.org/docs/2.7/generated/torch.isfinite.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.isin

<div style="margin-left: 2em">

**Native Documentation**: [torch.isin](https://pytorch.org/docs/2.7/generated/torch.isin.html)

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

**Constraints & Notes**:

- The constraints for the scenario with two `tensor` inputs are as follows:
  - `elements` and `test_elements` only support fp16, fp32, uint8, int8, int16, int32, and int64
  - The first input `tensor` must have no more than 7 dimensions, and the second input `tensor` must have no more than 8 dimensions
- The constraints for the scenario with a single `tensor` input are as follows:
  - `elements` and `test_elements` only support bf16, fp16, fp32, fp64, uint8, int8, int16, int32, and int64
  - The input `tensor` must have no more than 8 dimensions

</div>

### torch.isinf

<div style="margin-left: 2em">

**Native Documentation**: [torch.isinf](https://pytorch.org/docs/2.7/generated/torch.isinf.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.isposinf

<div style="margin-left: 2em">

**Native Documentation**: [torch.isposinf](https://pytorch.org/docs/2.7/generated/torch.isposinf.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.isneginf

<div style="margin-left: 2em">

**Native Documentation**: [torch.isneginf](https://pytorch.org/docs/2.7/generated/torch.isneginf.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.isnan

<div style="margin-left: 2em">

**Native Documentation**: [torch.isnan](https://pytorch.org/docs/2.7/generated/torch.isnan.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.kthvalue

<div style="margin-left: 2em">

**Native Documentation**: [torch.kthvalue](https://pytorch.org/docs/2.7/generated/torch.kthvalue.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, int32

</div>

### torch.le

<div style="margin-left: 2em">

**Native Documentation**: [torch.le](https://pytorch.org/docs/2.7/generated/torch.le.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.less_equal

<div style="margin-left: 2em">

**Native Documentation**: [torch.less_equal](https://pytorch.org/docs/2.7/generated/torch.less_equal.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, uint16, int8, int16, int32, int64, bool

</div>

### torch.lt

<div style="margin-left: 2em">

**Native Documentation**: [torch.lt](https://pytorch.org/docs/2.7/generated/torch.lt.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.less

<div style="margin-left: 2em">

**Native Documentation**: [torch.less](https://pytorch.org/docs/2.7/generated/torch.less.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.maximum

<div style="margin-left: 2em">

**Native Documentation**: [torch.maximum](https://pytorch.org/docs/2.7/generated/torch.maximum.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.minimum

<div style="margin-left: 2em">

**Native Documentation**: [torch.minimum](https://pytorch.org/docs/2.7/generated/torch.minimum.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.ne

<div style="margin-left: 2em">

**Native Documentation**: [torch.ne](https://pytorch.org/docs/2.7/generated/torch.ne.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.not_equal

<div style="margin-left: 2em">

**Native Documentation**: [torch.not_equal](https://pytorch.org/docs/2.7/generated/torch.not_equal.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.sort

<div style="margin-left: 2em">

**Native Documentation**: [torch.sort](https://pytorch.org/docs/2.7/generated/torch.sort.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

<!-- npu="950" id818 -->
- For <term>Ascend 950DT products</term>, due to underlying implementation limitations, `stable` only supports True. If set to False, it is automatically changed to True during execution
<!-- end id818 -->

</div>

### torch.topk

<div style="margin-left: 2em">

**Native Documentation**: [torch.topk](https://pytorch.org/docs/2.7/generated/torch.topk.html)

**Support Status**:

<!-- npu="910b" id819 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id819 -->
<!-- npu="A3" id820 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id820 -->
<!-- npu="950" id821 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id821 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64
- The scenario with `sorted=False` is not supported

</div>

### torch.msort

<div style="margin-left: 2em">

**Native Documentation**: [torch.msort](https://pytorch.org/docs/2.7/generated/torch.msort.html)

**Support Status**:

<!-- npu="910b" id822 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id822 -->
<!-- npu="A3" id823 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id823 -->
<!-- npu="950" id824 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id824 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.stft

<div style="margin-left: 2em">

**Native Documentation**: [torch.stft](https://pytorch.org/docs/2.7/generated/torch.stft.html)

**Support Status**:

<!-- npu="910b" id825 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id825 -->
<!-- npu="A3" id826 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id826 -->
<!-- npu="950" id827 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id827 -->

**Constraints & Notes**:

- `input` only supports fp32, fp64, complex64, complex128
- If an operator times out, use the official interface `set_op_execute_time_out` to increase the timeout threshold to extend the determination time

</div>

### torch.hann_window

<div style="margin-left: 2em">

**Native Documentation**: [torch.hann_window](https://pytorch.org/docs/2.7/generated/torch.hann_window.html)

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

**Constraints & Notes**:

- `dtype` only supports bf16, fp16, fp32
- When the data type is fp32 and the `window_length` parameter is greater than 10000, the computation result may have errors

</div>

### torch.atleast_1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.atleast_1d](https://pytorch.org/docs/2.7/generated/torch.atleast_1d.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.atleast_2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.atleast_2d](https://pytorch.org/docs/2.7/generated/torch.atleast_2d.html)

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

**Constraints & Notes**: `tensors` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.atleast_3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.atleast_3d](https://pytorch.org/docs/2.7/generated/torch.atleast_3d.html)

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

**Constraints & Notes**: `tensors` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.bincount

<div style="margin-left: 2em">

**Native Documentation**: [torch.bincount](https://pytorch.org/docs/2.7/generated/torch.bincount.html)

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

**Constraints & Notes**:

- `input` only supports uint8, int8, int16, int32, int64
- The dimension of `weights` must be the same as that of `input`

</div>

### torch.block_diag

<div style="margin-left: 2em">

**Native Documentation**: [torch.block_diag](https://pytorch.org/docs/2.7/generated/torch.block_diag.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.broadcast_tensors

<div style="margin-left: 2em">

**Native Documentation**: [torch.broadcast_tensors](https://pytorch.org/docs/2.7/generated/torch.broadcast_tensors.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.broadcast_to

<div style="margin-left: 2em">

**Native Documentation**: [torch.broadcast_to](https://pytorch.org/docs/2.7/generated/torch.broadcast_to.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.broadcast_shapes

<div style="margin-left: 2em">

**Native Documentation**: [torch.broadcast_shapes](https://pytorch.org/docs/2.7/generated/torch.broadcast_shapes.html)

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

</div>

### torch.cdist

<div style="margin-left: 2em">

**Native Documentation**: [torch.cdist](https://pytorch.org/docs/2.7/generated/torch.cdist.html)

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

**Constraints & Notes**:

- `x1` and `x2` only support bf16, fp16, and fp32
- When `p=2.0`, `compute_mode` only supports the "donot_use_mm_for_euclid_dist" mode. If any other value is passed, it is automatically changed to this mode

<!-- npu="950,A3,910b" id858 -->
- For <term>Ascend 950DT products</term>, when the input is fp16, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>
<!-- end id858 -->

</div>

### torch.clone

<div style="margin-left: 2em">

**Native Documentation**: [torch.clone](https://pytorch.org/docs/2.7/generated/torch.clone.html)

**Support Status**:

<!-- npu="910b" id859 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id859 -->
<!-- npu="A3" id860 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id860 -->
<!-- npu="950" id861 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id861 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.combinations

<div style="margin-left: 2em">

**Native Documentation**: [torch.combinations](https://pytorch.org/docs/2.7/generated/torch.combinations.html)

**Support Status**:

<!-- npu="910b" id862 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id862 -->
<!-- npu="A3" id863 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id863 -->
<!-- npu="950" id864 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id864 -->

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.cov

<div style="margin-left: 2em">

**Native Documentation**: [torch.cov](https://pytorch.org/docs/2.7/generated/torch.cov.html)

**Support Status**:

<!-- npu="910b" id865 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id865 -->
<!-- npu="A3" id866 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id866 -->
<!-- npu="950" id867 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id867 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.cross

<div style="margin-left: 2em">

**Native Documentation**: [torch.cross](https://pytorch.org/docs/2.7/generated/torch.cross.html)

**Support Status**:

<!-- npu="910b" id868 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id868 -->
<!-- npu="A3" id869 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id869 -->
<!-- npu="950" id870 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id870 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128
- The shapes of the two inputs must be consistent

</div>

### torch.cummax

<div style="margin-left: 2em">

**Native Documentation**: [torch.cummax](https://pytorch.org/docs/2.7/generated/torch.cummax.html)

**Support Status**:

<!-- npu="910b" id871 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id871 -->
<!-- npu="A3" id872 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id872 -->
<!-- npu="950" id873 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id873 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.cummin

<div style="margin-left: 2em">

**Native Documentation**: [torch.cummin](https://pytorch.org/docs/2.7/generated/torch.cummin.html)

**Support Status**:

<!-- npu="910b" id874 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id874 -->
<!-- npu="A3" id875 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id875 -->
<!-- npu="950" id876 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id876 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- When the input is int32, the value range is within [-16777216, 16777216]

</div>

### torch.cumprod

<div style="margin-left: 2em">

**Native Documentation**: [torch.cumprod](https://pytorch.org/docs/2.7/generated/torch.cumprod.html)

**Support Status**:

<!-- npu="910b" id877 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id877 -->
<!-- npu="A3" id878 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id878 -->
<!-- npu="950" id879 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id879 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.cumsum

<div style="margin-left: 2em">

**Native Documentation**: [torch.cumsum](https://pytorch.org/docs/2.7/generated/torch.cumsum.html)

**Support Status**:

<!-- npu="910b" id880 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id880 -->
<!-- npu="A3" id881 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id881 -->
<!-- npu="950" id882 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id882 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Named Tensor is supported

</div>

### torch.diag

<div style="margin-left: 2em">

**Native Documentation**: [torch.diag](https://pytorch.org/docs/2.7/generated/torch.diag.html)

**Support Status**:

<!-- npu="910b" id883 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id883 -->
<!-- npu="A3" id884 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id884 -->
<!-- npu="950" id885 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id885 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

### torch.diag_embed

<div style="margin-left: 2em">

**Native Documentation**: [torch.diag_embed](https://pytorch.org/docs/2.7/generated/torch.diag_embed.html)

**Support Status**:

<!-- npu="910b" id886 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id886 -->
<!-- npu="A3" id887 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id887 -->
<!-- npu="950" id888 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id888 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.diagonal

<div style="margin-left: 2em">

**Native Documentation**: [torch.diagonal](https://pytorch.org/docs/2.7/generated/torch.diagonal.html)

**Support Status**:

<!-- npu="910b" id889 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id889 -->
<!-- npu="A3" id890 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id890 -->
<!-- npu="950" id891 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id891 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.diff

<div style="margin-left: 2em">

**Native Documentation**: [torch.diff](https://pytorch.org/docs/2.7/generated/torch.diff.html)

**Support Status**:

<!-- npu="910b" id892 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id892 -->
<!-- npu="A3" id893 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id893 -->
<!-- npu="950" id894 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id894 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.einsum

<div style="margin-left: 2em">

**Native Documentation**: [torch.einsum](https://pytorch.org/docs/2.7/generated/torch.einsum.html)

**Support Status**:

<!-- npu="910b" id895 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id895 -->
<!-- npu="A3" id896 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id896 -->
<!-- npu="950" id897 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id897 -->

**Constraints & Notes**: `operands` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.flatten

<div style="margin-left: 2em">

**Native Documentation**: [torch.flatten](https://pytorch.org/docs/2.7/generated/torch.flatten.html)

**Support Status**:

<!-- npu="910b" id898 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id898 -->
<!-- npu="A3" id899 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id899 -->
<!-- npu="950" id900 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id900 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.flip

<div style="margin-left: 2em">

**Native Documentation**: [torch.flip](https://pytorch.org/docs/2.7/generated/torch.flip.html)

**Support Status**:

<!-- npu="910b" id901 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id901 -->
<!-- npu="A3" id902 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id902 -->
<!-- npu="950" id903 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id903 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.fliplr

<div style="margin-left: 2em">

**Native Documentation**: [torch.fliplr](https://pytorch.org/docs/2.7/generated/torch.fliplr.html)

**Support Status**:

<!-- npu="910b" id904 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id904 -->
<!-- npu="A3" id905 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id905 -->
<!-- npu="950" id906 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id906 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.flipud

<div style="margin-left: 2em">

**Native Documentation**: [torch.flipud](https://pytorch.org/docs/2.7/generated/torch.flipud.html)

**Support Status**:

<!-- npu="910b" id907 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id907 -->
<!-- npu="A3" id908 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id908 -->
<!-- npu="950" id909 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id909 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.kron

<div style="margin-left: 2em">

**Native Documentation**: [torch.kron](https://pytorch.org/docs/2.7/generated/torch.kron.html)

**Support Status**:

<!-- npu="910b" id910 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id910 -->
<!-- npu="A3" id911 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id911 -->
<!-- npu="950" id912 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id912 -->

**Constraints & Notes**: Does not support inputs with 5 or more dimensions.

</div>

### torch.histc

<div style="margin-left: 2em">

**Native Documentation**: [torch.histc](https://pytorch.org/docs/2.7/generated/torch.histc.html)

**Support Status**:

<!-- npu="910b" id913 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id913 -->
<!-- npu="A3" id914 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id914 -->
<!-- npu="950" id915 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id915 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64
- When the value of the input `tensor` is at the boundary between counting intervals, there may be an error in whether the value is counted in the left interval or the right interval

</div>

### torch.meshgrid

<div style="margin-left: 2em">

**Native Documentation**: [torch.meshgrid](https://pytorch.org/docs/2.7/generated/torch.meshgrid.html)

**Support Status**:

<!-- npu="910b" id916 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id916 -->
<!-- npu="A3" id917 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id917 -->
<!-- npu="950" id918 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id918 -->

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.ravel

<div style="margin-left: 2em">

**Native Documentation**: [torch.ravel](https://pytorch.org/docs/2.7/generated/torch.ravel.html)

**Support Status**:

<!-- npu="910b" id919 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id919 -->
<!-- npu="A3" id920 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id920 -->
<!-- npu="950" id921 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id921 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.repeat_interleave

<div style="margin-left: 2em">

**Native Documentation**: [torch.repeat_interleave](https://pytorch.org/docs/2.7/generated/torch.repeat_interleave.html)

**Support Status**:

<!-- npu="910b" id922 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id922 -->
<!-- npu="A3" id923 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id923 -->
<!-- npu="950" id924 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id924 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool
- After the input tensor is repeated to obtain the output, the number of elements in the output must be less than $2^{22}$

</div>

### torch.roll

<div style="margin-left: 2em">

**Native Documentation**: [torch.roll](https://pytorch.org/docs/2.7/generated/torch.roll.html)

**Support Status**:

<!-- npu="910b" id925 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id925 -->
<!-- npu="A3" id926 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id926 -->
<!-- npu="950" id927 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id927 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64, bool

</div>

### torch.searchsorted

<div style="margin-left: 2em">

**Native Documentation**: [torch.searchsorted](https://pytorch.org/docs/2.7/generated/torch.searchsorted.html)

**Support Status**:

<!-- npu="910b" id928 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id928 -->
<!-- npu="A3" id929 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id929 -->
<!-- npu="950" id930 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id930 -->

**Constraints & Notes**: `sorted_sequence` and `values` only support fp16, fp32, fp64, uint8, int8, int16, int32, and int64

</div>

### torch.tensordot

<div style="margin-left: 2em">

**Native Documentation**: [torch.tensordot](https://pytorch.org/docs/2.7/generated/torch.tensordot.html)

**Support Status**:

<!-- npu="910b" id931 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id931 -->
<!-- npu="A3" id932 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id932 -->
<!-- npu="950" id933 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id933 -->

**Constraints & Notes**: `a` and `b` only support fp16 and fp32

</div>

### torch.tril

<div style="margin-left: 2em">

**Native Documentation**: [torch.tril](https://pytorch.org/docs/2.7/generated/torch.tril.html)

**Support Status**:

<!-- npu="910b" id934 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id934 -->
<!-- npu="A3" id935 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id935 -->
<!-- npu="950" id936 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id936 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.tril_indices

<div style="margin-left: 2em">

**Native Documentation**: [torch.tril_indices](https://pytorch.org/docs/2.7/generated/torch.tril_indices.html)

**Support Status**:

<!-- npu="910b" id937 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id937 -->
<!-- npu="A3" id938 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id938 -->
<!-- npu="950" id939 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id939 -->

</div>

### torch.triu

<div style="margin-left: 2em">

**Native Documentation**: [torch.triu](https://pytorch.org/docs/2.7/generated/torch.triu.html)

**Support Status**:

<!-- npu="910b" id940 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id940 -->
<!-- npu="A3" id941 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id941 -->
<!-- npu="950" id942 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id942 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.triu_indices

<div style="margin-left: 2em">

**Native Documentation**: [torch.triu_indices](https://pytorch.org/docs/2.7/generated/torch.triu_indices.html)

**Support Status**:

<!-- npu="910b" id943 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id943 -->
<!-- npu="A3" id944 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id944 -->
<!-- npu="950" id945 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id945 -->

</div>

### torch.unflatten

<div style="margin-left: 2em">

**Native Documentation**: [torch.unflatten](https://pytorch.org/docs/2.7/generated/torch.unflatten.html)

**Support Status**:

<!-- npu="910b" id946 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id946 -->
<!-- npu="A3" id947 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id947 -->
<!-- npu="950" id948 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id948 -->

</div>

### torch.view_as_real

<div style="margin-left: 2em">

**Native Documentation**: [torch.view_as_real](https://pytorch.org/docs/2.7/generated/torch.view_as_real.html)

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

**Constraints & Notes**: `input` only supports complex64, complex128

</div>

### torch.view_as_complex

<div style="margin-left: 2em">

**Native Documentation**: [torch.view_as_complex](https://pytorch.org/docs/2.7/generated/torch.view_as_complex.html)

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

**Constraints & Notes**: `input` only supports fp32, fp64

</div>

### torch.resolve_conj

<div style="margin-left: 2em">

**Native Documentation**: [torch.resolve_conj](https://pytorch.org/docs/2.7/generated/torch.resolve_conj.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.resolve_neg

<div style="margin-left: 2em">

**Native Documentation**: [torch.resolve_neg](https://pytorch.org/docs/2.7/generated/torch.resolve_neg.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.addbmm

<div style="margin-left: 2em">

**Native Documentation**: [torch.addbmm](https://pytorch.org/docs/2.7/generated/torch.addbmm.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.addmm

<div style="margin-left: 2em">

**Native Documentation**: [torch.addmm](https://pytorch.org/docs/2.7/generated/torch.addmm.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.addmv

<div style="margin-left: 2em">

**Native Documentation**: [torch.addmv](https://pytorch.org/docs/2.7/generated/torch.addmv.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.addr

<div style="margin-left: 2em">

**Native Documentation**: [torch.addr](https://pytorch.org/docs/2.7/generated/torch.addr.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.baddbmm

<div style="margin-left: 2em">

**Native Documentation**: [torch.baddbmm](https://pytorch.org/docs/2.7/generated/torch.baddbmm.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.bmm

<div style="margin-left: 2em">

**Native Documentation**: [torch.bmm](https://pytorch.org/docs/2.7/generated/torch.bmm.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.dot

<div style="margin-left: 2em">

**Native Documentation**: [torch.dot](https://pytorch.org/docs/2.7/generated/torch.dot.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32

</div>

### torch.slogdet

<div style="margin-left: 2em">

**Native Documentation**: [torch.slogdet](https://pytorch.org/docs/2.7/generated/torch.slogdet.html)

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

**Constraints & Notes**:

- `input` only supports fp32, complex64, complex128
- May fall back to CPU execution

</div>

### torch.matmul

<div style="margin-left: 2em">

**Native Documentation**: [torch.matmul](https://pytorch.org/docs/2.7/generated/torch.matmul.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- Named Tensor is supported
- The input supports a maximum of 6 dimensions.

</div>

### torch.mm

<div style="margin-left: 2em">

**Native Documentation**: [torch.mm](https://pytorch.org/docs/2.7/generated/torch.mm.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.outer

<div style="margin-left: 2em">

**Native Documentation**: [torch.outer](https://pytorch.org/docs/2.7/generated/torch.outer.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.qr

<div style="margin-left: 2em">

**Native Documentation**: [torch.qr](https://pytorch.org/docs/2.7/generated/torch.qr.html)

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

</div>

### torch.trapezoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.trapezoid](https://pytorch.org/docs/2.7/generated/torch.trapezoid.html)

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

**Constraints & Notes**: `y` and `x` only support bf16, fp16, fp32, uint8, int8, int16, int32, and int64

</div>

### torch.cumulative_trapezoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.cumulative_trapezoid](https://pytorch.org/docs/2.7/generated/torch.cumulative_trapezoid.html)

**Support Status**:

<!-- npu="910b" id1000 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1000 -->
<!-- npu="A3" id1001 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1001 -->
<!-- npu="950" id1002 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1002 -->

**Constraints & Notes**: `y` and `x` only support fp16, fp32, fp64, uint8, int8, int16, int32, and int64

</div>

### torch.vdot

<div style="margin-left: 2em">

**Native Documentation**: [torch.vdot](https://pytorch.org/docs/2.7/generated/torch.vdot.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.bucketize

<div style="margin-left: 2em">

**Native Documentation**: [torch.bucketize](https://pytorch.org/docs/2.7/generated/torch.bucketize.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.cartesian_prod

<div style="margin-left: 2em">

**Native Documentation**: [torch.cartesian_prod](https://pytorch.org/docs/2.7/generated/torch.cartesian_prod.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.mv

<div style="margin-left: 2em">

**Native Documentation**: [torch.mv](https://pytorch.org/docs/2.7/generated/torch.mv.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch._foreach_sqrt

<div style="margin-left: 2em">

**Native Documentation**: [torch._foreach_sqrt](https://pytorch.org/docs/2.7/generated/torch._foreach_sqrt.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch._foreach_asin

<div style="margin-left: 2em">

**Native Documentation**: [torch._foreach_asin](https://pytorch.org/docs/2.7/generated/torch._foreach_asin.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.\_foreach_neg\_

<div style="margin-left: 2em">

**Native Documentation**: [torch.\\_foreach_neg_](https://pytorch.org/docs/2.7/generated/torch._foreach_neg_.html#torch._foreach_neg_)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int8, int32, int64

</div>

### torch.corrcoef

<div style="margin-left: 2em">

**Native Documentation**: [torch.corrcoef](https://pytorch.org/docs/2.7/generated/torch.corrcoef.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

## Utilities

### torch.compiled_with_cxx11_abi

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiled_with_cxx11_abi](https://pytorch.org/docs/2.7/generated/torch.compiled_with_cxx11_abi.html)

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

</div>

### torch.result_type

<div style="margin-left: 2em">

**Native Documentation**: [torch.result_type](https://pytorch.org/docs/2.7/generated/torch.result_type.html)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.can_cast

<div style="margin-left: 2em">

**Native Documentation**: [torch.can_cast](https://pytorch.org/docs/2.7/generated/torch.can_cast.html)

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

</div>

### torch.promote_types

<div style="margin-left: 2em">

**Native Documentation**: [torch.promote_types](https://pytorch.org/docs/2.7/generated/torch.promote_types.html)

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

</div>

### torch.use_deterministic_algorithms

<div style="margin-left: 2em">

**Native Documentation**: [torch.use_deterministic_algorithms](https://pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html)

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

**Constraints & Notes**:

- When HCCL_DETERMINISTIC and `torch.use_deterministic_algorithms` are set at the same time, if HCCL_DETERMINISTIC enables determinism, the HCCL interfaces enable determinism; otherwise, HCCL determinism is controlled by the `torch.use_deterministic_algorithms` interface
- When `torch.use_deterministic_algorithms` is set, PyTorch fills uninitialized memory by default (`torch.utils.deterministic.fill_uninitialized_memory` defaults to True), while TorchNPU does not fill it by default. To make the filling take effect, manually set `torch.utils.deterministic.fill_uninitialized_memory` to True

</div>

### torch.are_deterministic_algorithms_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.are_deterministic_algorithms_enabled](https://pytorch.org/docs/2.7/generated/torch.are_deterministic_algorithms_enabled.html)

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

</div>

### torch.is_deterministic_algorithms_warn_only_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_deterministic_algorithms_warn_only_enabled](https://pytorch.org/docs/2.7/generated/torch.is_deterministic_algorithms_warn_only_enabled.html)

**Support Status**:

<!-- npu="910b" id1045 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id1045 -->
<!-- npu="A3" id1046 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id1046 -->
<!-- npu="950" id1047 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1047 -->

</div>

### torch.set_deterministic_debug_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_deterministic_debug_mode](https://pytorch.org/docs/2.7/generated/torch.set_deterministic_debug_mode.html)

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

</div>

### torch.get_deterministic_debug_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_deterministic_debug_mode](https://pytorch.org/docs/2.7/generated/torch.get_deterministic_debug_mode.html)

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

</div>

### torch.set_float32_matmul_precision

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_float32_matmul_precision](https://pytorch.org/docs/2.7/generated/torch.set_float32_matmul_precision.html)

**Support Status**:

<!-- npu="910b" id1054 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1054 -->
<!-- npu="A3" id1055 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1055 -->
<!-- npu="950" id1056 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1056 -->

**Constraints & Notes**: This API can be called on the NPU, but it has no actual effect.

</div>

### torch.get_float32_matmul_precision

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_float32_matmul_precision](https://pytorch.org/docs/2.7/generated/torch.get_float32_matmul_precision.html)

**Support Status**:

<!-- npu="910b" id1057 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1057 -->
<!-- npu="A3" id1058 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1058 -->
<!-- npu="950" id1059 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1059 -->

</div>

### torch.set_warn_always

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_warn_always](https://pytorch.org/docs/2.7/generated/torch.set_warn_always.html)

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

</div>

### torch.is_warn_always_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_warn_always_enabled](https://pytorch.org/docs/2.7/generated/torch.is_warn_always_enabled.html)

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

</div>

### torch.vmap

<div style="margin-left: 2em">

**Native Documentation**: [torch.vmap](https://pytorch.org/docs/2.7/generated/torch.vmap.html)

**Support Status**:

<!-- npu="910b" id1066 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1066 -->
<!-- npu="A3" id1067 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1067 -->
<!-- npu="950" id1068 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1068 -->

</div>

### torch._assert

<div style="margin-left: 2em">

**Native Documentation**: [torch._assert](https://pytorch.org/docs/2.7/generated/torch._assert.html)

**Support Status**:

<!-- npu="910b" id1069 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1069 -->
<!-- npu="A3" id1070 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1070 -->
<!-- npu="950" id1071 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1071 -->

</div>

## Symbolic Numbers

### <code><i>class</i></code> torch.SymInt

<div style="margin-left: 2em">

**Native Documentation**: [torch.SymInt](https://pytorch.org/docs/2.7/torch.html#torch.SymInt)

**Support Status**:

<!-- npu="910b" id1072 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1072 -->
<!-- npu="A3" id1073 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1073 -->
<!-- npu="950" id1074 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1074 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.SymBool

<div style="margin-left: 2em">

**Native Documentation**: [torch.SymBool](https://pytorch.org/docs/2.7/torch.html#torch.SymBool)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.SymFloat

<div style="margin-left: 2em">

**Native Documentation**: [torch.SymFloat](https://pytorch.org/docs/2.7/torch.html#torch.SymFloat)

**Support Status**:

<!-- npu="910b" id1078 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1078 -->
<!-- npu="A3" id1079 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1079 -->
<!-- npu="950" id1080 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1080 -->

**Constraints & Notes**: -

> <font size="3">is_integer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.SymFloat.is_integer](https://pytorch.org/docs/2.7/torch.html#torch.SymFloat.is_integer)

**Support Status**:

<!-- npu="910b" id1081 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1081 -->
<!-- npu="A3" id1082 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1082 -->
<!-- npu="950" id1083 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1083 -->

</div>

</div>

### torch.sym_float

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_float](https://pytorch.org/docs/2.7/generated/torch.sym_float.html)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.sym_int

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_int](https://pytorch.org/docs/2.7/generated/torch.sym_int.html)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.sym_ite

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_ite](https://pytorch.org/docs/2.7/generated/torch.sym_ite.html)

**Support Status**:

<!-- npu="910b" id1090 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1090 -->
<!-- npu="A3" id1091 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1091 -->
<!-- npu="950" id1092 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1092 -->

</div>

### torch.sym_max

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_max](https://pytorch.org/docs/2.7/generated/torch.sym_max.html)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.sym_min

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_min](https://pytorch.org/docs/2.7/generated/torch.sym_min.html)

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

</div>

### torch.sym_not

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_not](https://pytorch.org/docs/2.7/generated/torch.sym_not.html)

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

</div>

## Optimizations

### torch.compile

<div style="margin-left: 2em">

**Native Documentation**: [torch.compile](https://pytorch.org/docs/2.7/generated/torch.compile.html)

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

**Constraints & Notes**: `backend` supports npugraphs, and the overall functionality is the same as `backend`="cudagraphs".

</div>

## Operator Tags

### <code><i>class</i></code> torch.Tag

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tag](https://pytorch.org/docs/2.7/torch.html#torch.Tag)

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

> <font size="3">name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tag.name](https://pytorch.org/docs/2.7/torch.html#torch.Tag.name)

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

</div>

</div>
