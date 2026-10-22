# torch

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.12/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.12/torch.html).

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

**Native Documentation**: [torch.is_tensor](https://pytorch.org/docs/2.12/generated/torch.is_tensor.html)

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

**Native Documentation**: [torch.is_storage](https://pytorch.org/docs/2.12/generated/torch.is_storage.html)

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

**Native Documentation**: [torch.is_complex](https://pytorch.org/docs/2.12/generated/torch.is_complex.html)

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

**Native Documentation**: [torch.is_conj](https://pytorch.org/docs/2.12/generated/torch.is_conj.html)

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

**Native Documentation**: [torch.is_floating_point](https://pytorch.org/docs/2.12/generated/torch.is_floating_point.html)

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

**Native Documentation**: [torch.is_nonzero](https://pytorch.org/docs/2.12/generated/torch.is_nonzero.html)

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

**Native Documentation**: [torch.set_default_dtype](https://pytorch.org/docs/2.12/generated/torch.set_default_dtype.html)

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

**Native Documentation**: [torch.get_default_dtype](https://pytorch.org/docs/2.12/generated/torch.get_default_dtype.html)

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

**Native Documentation**: [torch.set_default_device](https://pytorch.org/docs/2.12/generated/torch.set_default_device.html)

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

**Native Documentation**: [torch.set_default_tensor_type](https://pytorch.org/docs/2.12/generated/torch.set_default_tensor_type.html)

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

**Native Documentation**: [torch.numel](https://pytorch.org/docs/2.12/generated/torch.numel.html)

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

**Native Documentation**: [torch.set_printoptions](https://pytorch.org/docs/2.12/generated/torch.set_printoptions.html)

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

**Native Documentation**: [torch.set_flush_denormal](https://pytorch.org/docs/2.12/generated/torch.set_flush_denormal.html)

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

**Native Documentation**: [torch.tensor](https://pytorch.org/docs/2.12/generated/torch.tensor.html)

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

**Native Documentation**: [torch.sparse_coo_tensor](https://pytorch.org/docs/2.12/generated/torch.sparse_coo_tensor.html)

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

**Native Documentation**: [torch.sparse_csr_tensor](https://pytorch.org/docs/2.12/generated/torch.sparse_csr_tensor.html)

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

**Native Documentation**: [torch.sparse_csc_tensor](https://pytorch.org/docs/2.12/generated/torch.sparse_csc_tensor.html)

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

**Native Documentation**: [torch.sparse_bsr_tensor](https://pytorch.org/docs/2.12/generated/torch.sparse_bsr_tensor.html)

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

**Native Documentation**: [torch.sparse_bsc_tensor](https://pytorch.org/docs/2.12/generated/torch.sparse_bsc_tensor.html)

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

**Native Documentation**: [torch.asarray](https://pytorch.org/docs/2.12/generated/torch.asarray.html)

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

**Native Documentation**: [torch.as_tensor](https://pytorch.org/docs/2.12/generated/torch.as_tensor.html)

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

**Native Documentation**: [torch.as_strided](https://pytorch.org/docs/2.12/generated/torch.as_strided.html)

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

**Native Documentation**: [torch.from_numpy](https://pytorch.org/docs/2.12/generated/torch.from_numpy.html)

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

**Native Documentation**: [torch.from_dlpack](https://pytorch.org/docs/2.12/generated/torch.from_dlpack.html)

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

**Native Documentation**: [torch.frombuffer](https://pytorch.org/docs/2.12/generated/torch.frombuffer.html)

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

**Native Documentation**: [torch.zeros](https://pytorch.org/docs/2.12/generated/torch.zeros.html)

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

</div>

### torch.zeros_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.zeros_like](https://pytorch.org/docs/2.12/generated/torch.zeros_like.html)

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

**Native Documentation**: [torch.ones](https://pytorch.org/docs/2.12/generated/torch.ones.html)

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

**Native Documentation**: [torch.ones_like](https://pytorch.org/docs/2.12/generated/torch.ones_like.html)

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

**Native Documentation**: [torch.arange](https://pytorch.org/docs/2.12/generated/torch.arange.html)

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

**Constraints & Notes**: `dtype` only supports bf16, fp16, fp32, fp64, int32, int64

</div>

### torch.range

<div style="margin-left: 2em">

**Native Documentation**: [torch.range](https://pytorch.org/docs/2.12/generated/torch.range.html)

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

**Native Documentation**: [torch.linspace](https://pytorch.org/docs/2.12/generated/torch.linspace.html)

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

**Native Documentation**: [torch.eye](https://pytorch.org/docs/2.12/generated/torch.eye.html)

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

**Native Documentation**: [torch.empty](https://pytorch.org/docs/2.12/generated/torch.empty.html)

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

**Native Documentation**: [torch.empty_like](https://pytorch.org/docs/2.12/generated/torch.empty_like.html)

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

**Native Documentation**: [torch.empty_strided](https://pytorch.org/docs/2.12/generated/torch.empty_strided.html)

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

**Native Documentation**: [torch.full](https://pytorch.org/docs/2.12/generated/torch.full.html)

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

**Native Documentation**: [torch.full_like](https://pytorch.org/docs/2.12/generated/torch.full_like.html)

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

**Native Documentation**: [torch.quantize_per_tensor](https://pytorch.org/docs/2.12/generated/torch.quantize_per_tensor.html)

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

**Native Documentation**: [torch.quantize_per_channel](https://pytorch.org/docs/2.12/generated/torch.quantize_per_channel.html)

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

**Native Documentation**: [torch.dequantize](https://pytorch.org/docs/2.12/generated/torch.dequantize.html)

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

**Native Documentation**: [torch.complex](https://pytorch.org/docs/2.12/generated/torch.complex.html)

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

**Native Documentation**: [torch.polar](https://pytorch.org/docs/2.12/generated/torch.polar.html)

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

**Native Documentation**: [torch.heaviside](https://pytorch.org/docs/2.12/generated/torch.heaviside.html)

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

**Native Documentation**: [torch.argwhere](https://pytorch.org/docs/2.12/generated/torch.argwhere.html)

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

**Native Documentation**: [torch.cat](https://pytorch.org/docs/2.12/generated/torch.cat.html)

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

**Native Documentation**: [torch.concat](https://pytorch.org/docs/2.12/generated/torch.concat.html)

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

**Native Documentation**: [torch.concatenate](https://pytorch.org/docs/2.12/generated/torch.concatenate.html)

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

**Native Documentation**: [torch.conj](https://pytorch.org/docs/2.12/generated/torch.conj.html)

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

**Native Documentation**: [torch.chunk](https://pytorch.org/docs/2.12/generated/torch.chunk.html)

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

**Native Documentation**: [torch.dsplit](https://pytorch.org/docs/2.12/generated/torch.dsplit.html)

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

**Native Documentation**: [torch.dstack](https://pytorch.org/docs/2.12/generated/torch.dstack.html)

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

**Native Documentation**: [torch.gather](https://pytorch.org/docs/2.12/generated/torch.gather.html)

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

- `input` only supports fp16, fp32, int16, int32, int64, bool
- The number of dimensions of `index` must be the same as that of `input`

<!-- npu="950,A3,910b" id162 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, when there are duplicate indices, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>. For example:

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

**Native Documentation**: [torch.hsplit](https://pytorch.org/docs/2.12/generated/torch.hsplit.html)

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

**Native Documentation**: [torch.hstack](https://pytorch.org/docs/2.12/generated/torch.hstack.html)

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

**Native Documentation**: [torch.index_add](https://pytorch.org/docs/2.12/generated/torch.index_add.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, int64, bool

</div>

### torch.index_copy

<div style="margin-left: 2em">

**Native Documentation**: [torch.index_copy](https://pytorch.org/docs/2.12/generated/torch.index_copy.html)

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

**Native Documentation**: [torch.index_reduce](https://pytorch.org/docs/2.12/generated/torch.index_reduce.html)

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

**Native Documentation**: [torch.index_select](https://pytorch.org/docs/2.12/generated/torch.index_select.html)

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

- `input` only supports bf16, fp16, fp32, int16, int32, int64, bool

<!-- npu="950,A3,910b" id181 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, when there are duplicate indices, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>. For example:

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

**Native Documentation**: [torch.masked_select](https://pytorch.org/docs/2.12/generated/torch.masked_select.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, int16, int32, int64, bool

</div>

### torch.movedim

<div style="margin-left: 2em">

**Native Documentation**: [torch.movedim](https://pytorch.org/docs/2.12/generated/torch.movedim.html)

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

**Native Documentation**: [torch.moveaxis](https://pytorch.org/docs/2.12/generated/torch.moveaxis.html)

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

**Native Documentation**: [torch.narrow](https://pytorch.org/docs/2.12/generated/torch.narrow.html)

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

**Native Documentation**: [torch.narrow_copy](https://pytorch.org/docs/2.12/generated/torch.narrow_copy.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

### torch.nonzero

<div style="margin-left: 2em">

**Native Documentation**: [torch.nonzero](https://pytorch.org/docs/2.12/generated/torch.nonzero.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.permute

<div style="margin-left: 2em">

**Native Documentation**: [torch.permute](https://pytorch.org/docs/2.12/generated/torch.permute.html)

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

**Native Documentation**: [torch.reshape](https://pytorch.org/docs/2.12/generated/torch.reshape.html)

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

**Native Documentation**: [torch.row_stack](https://pytorch.org/docs/2.12/generated/torch.row_stack.html)

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

**Native Documentation**: [torch.scatter](https://pytorch.org/docs/2.12/generated/torch.scatter.html)

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

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- May fall back to CPU execution

<!-- npu="950,A3,910b" id212 -->
- For <term>Ascend 950DT products</term>, due to hardware differences, when there are duplicate indices, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>. For example:

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

**Native Documentation**: [torch.diagonal_scatter](https://pytorch.org/docs/2.12/generated/torch.diagonal_scatter.html)

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

**Native Documentation**: [torch.select_scatter](https://pytorch.org/docs/2.12/generated/torch.select_scatter.html)

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

**Native Documentation**: [torch.slice_scatter](https://pytorch.org/docs/2.12/generated/torch.slice_scatter.html)

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

### torch.scatter_reduce

<div style="margin-left: 2em">

**Native Documentation**: [torch.scatter_reduce](https://pytorch.org/docs/2.12/generated/torch.scatter_reduce.html)

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

</div>

### torch.split

<div style="margin-left: 2em">

**Native Documentation**: [torch.split](https://pytorch.org/docs/2.12/generated/torch.split.html)

**Support Status**:

<!-- npu="910b" id225 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id225 -->
<!-- npu="A3" id226 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id226 -->
<!-- npu="950" id227 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id227 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.squeeze

<div style="margin-left: 2em">

**Native Documentation**: [torch.squeeze](https://pytorch.org/docs/2.12/generated/torch.squeeze.html)

**Support Status**:

<!-- npu="910b" id228 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id228 -->
<!-- npu="A3" id229 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id229 -->
<!-- npu="950" id230 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id230 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.stack

<div style="margin-left: 2em">

**Native Documentation**: [torch.stack](https://pytorch.org/docs/2.12/generated/torch.stack.html)

**Support Status**:

<!-- npu="910b" id231 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id231 -->
<!-- npu="A3" id232 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id232 -->
<!-- npu="950" id233 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id233 -->

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.swapaxes

<div style="margin-left: 2em">

**Native Documentation**: [torch.swapaxes](https://pytorch.org/docs/2.12/generated/torch.swapaxes.html)

**Support Status**:

<!-- npu="910b" id234 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id234 -->
<!-- npu="A3" id235 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id235 -->
<!-- npu="950" id236 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id236 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.swapdims

<div style="margin-left: 2em">

**Native Documentation**: [torch.swapdims](https://pytorch.org/docs/2.12/generated/torch.swapdims.html)

**Support Status**:

<!-- npu="910b" id237 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id237 -->
<!-- npu="A3" id238 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id238 -->
<!-- npu="950" id239 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id239 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.t

<div style="margin-left: 2em">

**Native Documentation**: [torch.t](https://pytorch.org/docs/2.12/generated/torch.t.html)

**Support Status**:

<!-- npu="910b" id240 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id240 -->
<!-- npu="A3" id241 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id241 -->
<!-- npu="950" id242 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id242 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.take

<div style="margin-left: 2em">

**Native Documentation**: [torch.take](https://pytorch.org/docs/2.12/generated/torch.take.html)

**Support Status**:

<!-- npu="910b" id243 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id243 -->
<!-- npu="A3" id244 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id244 -->
<!-- npu="950" id245 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id245 -->

**Constraints & Notes**: `input` only supports fp16, fp32, int16, int32, int64, bool

</div>

### torch.take_along_dim

<div style="margin-left: 2em">

**Native Documentation**: [torch.take_along_dim](https://pytorch.org/docs/2.12/generated/torch.take_along_dim.html)

**Support Status**:

<!-- npu="910b" id246 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id246 -->
<!-- npu="A3" id247 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id247 -->
<!-- npu="950" id248 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id248 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.tensor_split

<div style="margin-left: 2em">

**Native Documentation**: [torch.tensor_split](https://pytorch.org/docs/2.12/generated/torch.tensor_split.html)

**Support Status**:

<!-- npu="910b" id249 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id249 -->
<!-- npu="A3" id250 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id250 -->
<!-- npu="950" id251 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id251 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.tile

<div style="margin-left: 2em">

**Native Documentation**: [torch.tile](https://pytorch.org/docs/2.12/generated/torch.tile.html)

**Support Status**:

<!-- npu="910b" id252 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id252 -->
<!-- npu="A3" id253 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id253 -->
<!-- npu="950" id254 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id254 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- If the length of the `dims` parameter is smaller than the length of `input`.shape, 1s are automatically prepended to `dims` to align its length with `input`.shape. The padded `dims` must meet the following restrictions:
  - When the first axis needs to be repeated, a maximum of 4 dimensions can be repeated at the same time (that is, the number of elements greater than 1 in `dims` ≤ 4). For example, `torch.tile(input, [2, 3, 4, 5, 6])` is not supported, while `torch.tile(input, [2, 3, 1, 5, 6])` is supported.
  - When the first axis does not need to be repeated, a maximum of 3 dimensions can be repeated at the same time (that is, the number of elements greater than 1 in `dims` ≤ 3). For example, `torch.tile(input, [1, 3, 4, 5, 6])` is not supported, while `torch.tile(input, [1, 3, 1, 5, 6])` is supported.
  - In backward computation, the sum of the number of dimensions of the input `Tensor` and the number of elements greater than 1 in the `dims` parameter must not exceed 8.

</div>

### torch.transpose

<div style="margin-left: 2em">

**Native Documentation**: [torch.transpose](https://pytorch.org/docs/2.12/generated/torch.transpose.html)

**Support Status**:

<!-- npu="910b" id255 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id255 -->
<!-- npu="A3" id256 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id256 -->
<!-- npu="950" id257 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id257 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.unsqueeze

<div style="margin-left: 2em">

**Native Documentation**: [torch.unsqueeze](https://pytorch.org/docs/2.12/generated/torch.unsqueeze.html)

**Support Status**:

<!-- npu="910b" id258 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id258 -->
<!-- npu="A3" id259 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id259 -->
<!-- npu="950" id260 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id260 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.vsplit

<div style="margin-left: 2em">

**Native Documentation**: [torch.vsplit](https://pytorch.org/docs/2.12/generated/torch.vsplit.html)

**Support Status**:

<!-- npu="910b" id261 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id261 -->
<!-- npu="A3" id262 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id262 -->
<!-- npu="950" id263 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id263 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.vstack

<div style="margin-left: 2em">

**Native Documentation**: [torch.vstack](https://pytorch.org/docs/2.12/generated/torch.vstack.html)

**Support Status**:

<!-- npu="910b" id264 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id264 -->
<!-- npu="A3" id265 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id265 -->
<!-- npu="950" id266 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id266 -->

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64

</div>

### torch.where

<div style="margin-left: 2em">

**Native Documentation**: [torch.where](https://pytorch.org/docs/2.12/generated/torch.where.html)

**Support Status**:

<!-- npu="910b" id267 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id267 -->
<!-- npu="A3" id268 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id268 -->
<!-- npu="950" id269 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id269 -->

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- An 8-dimensional shape is not supported

</div>

## Generators

### <code><i>class</i></code> torch.Generator

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator](https://pytorch.org/docs/2.12/generated/torch.Generator.html)

**Support Status**:

<!-- npu="910b" id270 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id270 -->
<!-- npu="A3" id271 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id271 -->
<!-- npu="950" id272 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id272 -->

> <font size="3">device()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.device](https://pytorch.org/docs/2.12/generated/torch.Generator.html#torch.Generator.device)

**Support Status**:

<!-- npu="910b" id273 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id273 -->
<!-- npu="A3" id274 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id274 -->
<!-- npu="950" id275 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id275 -->

</div>

> <font size="3">get_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.get_state](https://pytorch.org/docs/2.12/generated/torch.Generator.html#torch.Generator.get_state)

**Support Status**:

<!-- npu="910b" id276 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id276 -->
<!-- npu="A3" id277 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id277 -->
<!-- npu="950" id278 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id278 -->

</div>

> <font size="3">initial_seed()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.initial_seed](https://pytorch.org/docs/2.12/generated/torch.Generator.html#torch.Generator.initial_seed)

**Support Status**:

<!-- npu="910b" id279 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id279 -->
<!-- npu="A3" id280 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id280 -->
<!-- npu="950" id281 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id281 -->

</div>

> <font size="3">manual_seed()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.manual_seed](https://pytorch.org/docs/2.12/generated/torch.Generator.html#torch.Generator.manual_seed)

**Support Status**:

<!-- npu="910b" id282 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id282 -->
<!-- npu="A3" id283 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id283 -->
<!-- npu="950" id284 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id284 -->

</div>

> <font size="3">seed()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.seed](https://pytorch.org/docs/2.12/generated/torch.Generator.html#torch.Generator.seed)

**Support Status**:

<!-- npu="910b" id285 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id285 -->
<!-- npu="A3" id286 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id286 -->
<!-- npu="950" id287 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id287 -->

</div>

> <font size="3">set_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.set_state](https://pytorch.org/docs/2.12/generated/torch.Generator.html#torch.Generator.set_state)

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

</div>

> <font size="3">clone_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.clone_state](https://pytorch.org/docs/2.12/generated/torch.Generator.html#torch.Generator.clone_state)

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

</div>

> <font size="3">graphsafe_set_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Generator.graphsafe_set_state](https://pytorch.org/docs/2.12/generated/torch.Generator.html#torch.Generator.graphsafe_set_state)

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

## Random sampling

### torch.default_generator

<div style="margin-left: 2em">

**Native Documentation**: [torch.torch.default_generator](https://pytorch.org/docs/2.12/torch.html#torch.torch.default_generator)

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

</div>

### torch.rand

<div style="margin-left: 2em">

**Native Documentation**: [torch.rand](https://pytorch.org/docs/2.12/generated/torch.rand.html)

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

### torch.rand_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.rand_like](https://pytorch.org/docs/2.12/generated/torch.rand_like.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64
- Following the PyTorch community convention, processing bool-type data is no longer supported. For existing bool-type data, use the following replacement methods: if all True values are required, use `torch.bernoulli(input, 1)`. If uniformly distributed bool values are required, use `torch.bernoulli(input, 0.5)`.

</div>

### torch.randint

<div style="margin-left: 2em">

**Native Documentation**: [torch.randint](https://pytorch.org/docs/2.12/generated/torch.randint.html)

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

</div>

### torch.randint_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.randint_like](https://pytorch.org/docs/2.12/generated/torch.randint_like.html)

**Support Status**:

<!-- npu="910b" id309 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id309 -->
<!-- npu="A3" id310 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id310 -->
<!-- npu="950" id311 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id311 -->

**Constraints & Notes**: `input` only supports fp16, fp32, int64

</div>

### torch.randn

<div style="margin-left: 2em">

**Native Documentation**: [torch.randn](https://pytorch.org/docs/2.12/generated/torch.randn.html)

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

</div>

### torch.randn_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.randn_like](https://pytorch.org/docs/2.12/generated/torch.randn_like.html)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.randperm

<div style="margin-left: 2em">

**Native Documentation**: [torch.randperm](https://pytorch.org/docs/2.12/generated/torch.randperm.html)

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

### torch.seed

<div style="margin-left: 2em">

**Native Documentation**: [torch.seed](https://pytorch.org/docs/2.12/generated/torch.seed.html)

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

</div>

### torch.manual_seed

<div style="margin-left: 2em">

**Native Documentation**: [torch.manual_seed](https://pytorch.org/docs/2.12/generated/torch.manual_seed.html)

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

</div>

### torch.initial_seed

<div style="margin-left: 2em">

**Native Documentation**: [torch.initial_seed](https://pytorch.org/docs/2.12/generated/torch.initial_seed.html)

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

### torch.get_rng_state

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_rng_state](https://pytorch.org/docs/2.12/generated/torch.get_rng_state.html)

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

</div>

### torch.set_rng_state

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_rng_state](https://pytorch.org/docs/2.12/generated/torch.set_rng_state.html)

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

### torch.bernoulli

<div style="margin-left: 2em">

**Native Documentation**: [torch.bernoulli](https://pytorch.org/docs/2.12/generated/torch.bernoulli.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, fp64

</div>

### torch.multinomial

<div style="margin-left: 2em">

**Native Documentation**: [torch.multinomial](https://pytorch.org/docs/2.12/generated/torch.multinomial.html)

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

### torch.normal

<div style="margin-left: 2em">

**Native Documentation**: [torch.normal](https://pytorch.org/docs/2.12/generated/torch.normal.html)

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

**Constraints & Notes**: `mean` and `std` only support fp16 and fp32.

</div>

### torch.poisson

<div style="margin-left: 2em">

**Native Documentation**: [torch.poisson](https://pytorch.org/docs/2.12/generated/torch.poisson.html)

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

</div>

### <code><i>class</i></code> torch.quasirandom.SobolEngine

<div style="margin-left: 2em">

> <font size="3">draw()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.quasirandom.SobolEngine.draw](https://pytorch.org/docs/2.12/generated/torch.quasirandom.SobolEngine.html#torch.quasirandom.SobolEngine.draw)

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

**Constraints & Notes**: `self` only supports fp32, fp64

</div>

</div>

## Serialization

### torch.save

<div style="margin-left: 2em">

**Native Documentation**: [torch.save](https://pytorch.org/docs/2.12/generated/torch.save.html)

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

### torch.load

<div style="margin-left: 2em">

**Native Documentation**: [torch.load](https://pytorch.org/docs/2.12/generated/torch.load.html)

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

</div>

### torch.serialization.get_safe_globals

<div style="margin-left: 2em">

**Native Documentation**: [torch.serialization.get_safe_globals](https://pytorch.org/docs/2.12/notes/serialization.html#torch.serialization.get_safe_globals)

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

</div>

### torch.serialization.default_restore_location

<div style="margin-left: 2em">

**Native Documentation**: [torch.serialization.default_restore_location](https://pytorch.org/docs/2.12/torch.html#torch.serialization.default_restore_location)

**Support Status**:

<!-- npu="910b" id1104 -->
<term>Atlas A2 training products</term>: Supported
<!-- end id1104 -->
<!-- npu="A3" id1105 -->
<term>Atlas A3 training products</term>: Supported
<!-- end id1105 -->
<!-- npu="950" id1106 -->
<term>Ascend 950DT products</term>: Supported
<!-- end id1106 -->

</div>

### <code><i>class</i></code> torch.serialization.safe_globals

<div style="margin-left: 2em">

**Native Documentation**: [torch.serialization.safe_globals](https://pytorch.org/docs/2.12/notes/serialization.html#torch.serialization.safe_globals)

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

## Parallelism

### torch.get_num_threads

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_num_threads](https://pytorch.org/docs/2.12/generated/torch.get_num_threads.html)

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

### torch.set_num_threads

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_num_threads](https://pytorch.org/docs/2.12/generated/torch.set_num_threads.html)

**Support Status**:

<!-- npu="910b" id366 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id366 -->
<!-- npu="A3" id367 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id367 -->
<!-- npu="950" id368 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id368 -->

</div>

### torch.get_num_interop_threads

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_num_interop_threads](https://pytorch.org/docs/2.12/generated/torch.get_num_interop_threads.html)

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

</div>

### torch.set_num_interop_threads

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_num_interop_threads](https://pytorch.org/docs/2.12/generated/torch.set_num_interop_threads.html)

**Support Status**:

<!-- npu="910b" id372 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id372 -->
<!-- npu="A3" id373 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id373 -->
<!-- npu="950" id374 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id374 -->

</div>

## Locally disabling gradient computation

### torch.no_grad

<div style="margin-left: 2em">

**Native Documentation**: [torch.no_grad](https://pytorch.org/docs/2.12/generated/torch.no_grad.html)

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

</div>

### torch.enable_grad

<div style="margin-left: 2em">

**Native Documentation**: [torch.enable_grad](https://pytorch.org/docs/2.12/generated/torch.enable_grad.html)

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

</div>

### <code><i>class</i></code> torch.autograd.grad_mode.set_grad_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.grad_mode.set_grad_enabled](https://pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.set_grad_enabled.html)

**Support Status**:

<!-- npu="910b" id381 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id381 -->
<!-- npu="A3" id382 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id382 -->
<!-- npu="950" id383 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id383 -->

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.grad_mode.set_grad_enabled.clone](https://pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.set_grad_enabled.html#torch.autograd.grad_mode.set_grad_enabled.clone)

**Support Status**:

<!-- npu="910b" id384 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id384 -->
<!-- npu="A3" id385 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id385 -->
<!-- npu="950" id386 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id386 -->

</div>

</div>

### torch.is_grad_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_grad_enabled](https://pytorch.org/docs/2.12/generated/torch.is_grad_enabled.html)

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

</div>

### <code><i>class</i></code> torch.autograd.grad_mode.inference_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.grad_mode.inference_mode](https://pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.inference_mode.html)

**Support Status**:

<!-- npu="910b" id390 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id390 -->
<!-- npu="A3" id391 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id391 -->
<!-- npu="950" id392 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id392 -->

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.grad_mode.inference_mode.clone](https://pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.inference_mode.html#torch.autograd.grad_mode.inference_mode.clone)

**Support Status**:

<!-- npu="910b" id393 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id393 -->
<!-- npu="A3" id394 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id394 -->
<!-- npu="950" id395 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id395 -->

</div>

</div>

### torch.is_inference_mode_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_inference_mode_enabled](https://pytorch.org/docs/2.12/generated/torch.is_inference_mode_enabled.html)

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

</div>

## Math operations

### torch.abs

<div style="margin-left: 2em">

**Native Documentation**: [torch.abs](https://pytorch.org/docs/2.12/generated/torch.abs.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.angle

<div style="margin-left: 2em">

**Native Documentation**: [torch.angle](https://pytorch.org/docs/2.12/generated/torch.angle.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

### torch.real

<div style="margin-left: 2em">

**Native Documentation**: [torch.real](https://pytorch.org/docs/2.12/generated/torch.real.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, complex64, complex128

</div>

### torch.absolute

<div style="margin-left: 2em">

**Native Documentation**: [torch.absolute](https://pytorch.org/docs/2.12/generated/torch.absolute.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.acos

<div style="margin-left: 2em">

**Native Documentation**: [torch.acos](https://pytorch.org/docs/2.12/generated/torch.acos.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.arccos

<div style="margin-left: 2em">

**Native Documentation**: [torch.arccos](https://pytorch.org/docs/2.12/generated/torch.arccos.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.acosh

<div style="margin-left: 2em">

**Native Documentation**: [torch.acosh](https://pytorch.org/docs/2.12/generated/torch.acosh.html)

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

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- May fall back to CPU execution

</div>

### torch.arccosh

<div style="margin-left: 2em">

**Native Documentation**: [torch.arccosh](https://pytorch.org/docs/2.12/generated/torch.arccosh.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.add

<div style="margin-left: 2em">

**Native Documentation**: [torch.add](https://pytorch.org/docs/2.12/generated/torch.add.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.addcdiv

<div style="margin-left: 2em">

**Native Documentation**: [torch.addcdiv](https://pytorch.org/docs/2.12/generated/torch.addcdiv.html)

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

**Constraints & Notes**:

- `input` only supports fp16, fp32

</div>

### torch.addcmul

<div style="margin-left: 2em">

**Native Documentation**: [torch.addcmul](https://pytorch.org/docs/2.12/generated/torch.addcmul.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64
- fp64, uint8, int8, and int64 types do not support simultaneous broadcasting of three `tensor`s

</div>

### torch.asin

<div style="margin-left: 2em">

**Native Documentation**: [torch.asin](https://pytorch.org/docs/2.12/generated/torch.asin.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.arcsin

<div style="margin-left: 2em">

**Native Documentation**: [torch.arcsin](https://pytorch.org/docs/2.12/generated/torch.arcsin.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.asinh

<div style="margin-left: 2em">

**Native Documentation**: [torch.asinh](https://pytorch.org/docs/2.12/generated/torch.asinh.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.arcsinh

<div style="margin-left: 2em">

**Native Documentation**: [torch.arcsinh](https://pytorch.org/docs/2.12/generated/torch.arcsinh.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.atan

<div style="margin-left: 2em">

**Native Documentation**: [torch.atan](https://pytorch.org/docs/2.12/generated/torch.atan.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.arctan

<div style="margin-left: 2em">

**Native Documentation**: [torch.arctan](https://pytorch.org/docs/2.12/generated/torch.arctan.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.atanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.atanh](https://pytorch.org/docs/2.12/generated/torch.atanh.html)

**Support Status**:

<!-- npu="910b" id450 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id450 -->
<!-- npu="A3" id451 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id451 -->
<!-- npu="950" id452 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id452 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.arctanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.arctanh](https://pytorch.org/docs/2.12/generated/torch.arctanh.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.atan2

<div style="margin-left: 2em">

**Native Documentation**: [torch.atan2](https://pytorch.org/docs/2.12/generated/torch.atan2.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.arctan2

<div style="margin-left: 2em">

**Native Documentation**: [torch.arctan2](https://pytorch.org/docs/2.12/generated/torch.arctan2.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.bitwise_not

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_not](https://pytorch.org/docs/2.12/generated/torch.bitwise_not.html)

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

**Constraints & Notes**: `input` only supports uint8, int8, int16, int32, int64, bool

</div>

### torch.bitwise_and

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_and](https://pytorch.org/docs/2.12/generated/torch.bitwise_and.html)

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

**Constraints & Notes**: `input` only supports uint8, int8, int16, int32, int64, bool

</div>

### torch.bitwise_or

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_or](https://pytorch.org/docs/2.12/generated/torch.bitwise_or.html)

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

**Constraints & Notes**: `input` only supports uint8, int8, int16, int32, int64, bool

</div>

### torch.bitwise_xor

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_xor](https://pytorch.org/docs/2.12/generated/torch.bitwise_xor.html)

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

**Constraints & Notes**: `input` only supports uint8, int8, int16, int32, int64, bool

</div>

### torch.bitwise_left_shift

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_left_shift](https://pytorch.org/docs/2.12/generated/torch.bitwise_left_shift.html)

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

**Constraints & Notes**:

- `input` only supports uint8, int8, int16, int32, int64

- Precision can be guaranteed to be error-free only when the value of shiftBits is smaller than the bit width of the self data type.

</div>

### torch.bitwise_right_shift

<div style="margin-left: 2em">

**Native Documentation**: [torch.bitwise_right_shift](https://pytorch.org/docs/2.12/generated/torch.bitwise_right_shift.html)

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

**Constraints & Notes**:

- `input` only supports uint8, int8, int16, int32, int64

- Precision can be guaranteed to be error-free only when the value of shiftBits is smaller than the bit width of the self data type.

</div>

### torch.ceil

<div style="margin-left: 2em">

**Native Documentation**: [torch.ceil](https://pytorch.org/docs/2.12/generated/torch.ceil.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.clamp

<div style="margin-left: 2em">

**Native Documentation**: [torch.clamp](https://pytorch.org/docs/2.12/generated/torch.clamp.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.clip

<div style="margin-left: 2em">

**Native Documentation**: [torch.clip](https://pytorch.org/docs/2.12/generated/torch.clip.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.copysign

<div style="margin-left: 2em">

**Native Documentation**: [torch.copysign](https://pytorch.org/docs/2.12/generated/torch.copysign.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool
- May fall back to CPU execution

</div>

### torch.cos

<div style="margin-left: 2em">

**Native Documentation**: [torch.cos](https://pytorch.org/docs/2.12/generated/torch.cos.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.cosh

<div style="margin-left: 2em">

**Native Documentation**: [torch.cosh](https://pytorch.org/docs/2.12/generated/torch.cosh.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.deg2rad

<div style="margin-left: 2em">

**Native Documentation**: [torch.deg2rad](https://pytorch.org/docs/2.12/generated/torch.deg2rad.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.div

<div style="margin-left: 2em">

**Native Documentation**: [torch.div](https://pytorch.org/docs/2.12/generated/torch.div.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.divide

<div style="margin-left: 2em">

**Native Documentation**: [torch.divide](https://pytorch.org/docs/2.12/generated/torch.divide.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.erf

<div style="margin-left: 2em">

**Native Documentation**: [torch.erf](https://pytorch.org/docs/2.12/generated/torch.erf.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int64, bool

</div>

### torch.erfc

<div style="margin-left: 2em">

**Native Documentation**: [torch.erfc](https://pytorch.org/docs/2.12/generated/torch.erfc.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, int64, bool

</div>

### torch.erfinv

<div style="margin-left: 2em">

**Native Documentation**: [torch.erfinv](https://pytorch.org/docs/2.12/generated/torch.erfinv.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.exp

<div style="margin-left: 2em">

**Native Documentation**: [torch.exp](https://pytorch.org/docs/2.12/generated/torch.exp.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int64, bool, complex64, complex128

</div>

### torch.exp2

<div style="margin-left: 2em">

**Native Documentation**: [torch.exp2](https://pytorch.org/docs/2.12/generated/torch.exp2.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.expm1

<div style="margin-left: 2em">

**Native Documentation**: [torch.expm1](https://pytorch.org/docs/2.12/generated/torch.expm1.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int64, bool

</div>

### torch.fix

<div style="margin-left: 2em">

**Native Documentation**: [torch.fix](https://pytorch.org/docs/2.12/generated/torch.fix.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.float_power

<div style="margin-left: 2em">

**Native Documentation**: [torch.float_power](https://pytorch.org/docs/2.12/generated/torch.float_power.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex128

</div>

### torch.floor

<div style="margin-left: 2em">

**Native Documentation**: [torch.floor](https://pytorch.org/docs/2.12/generated/torch.floor.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.floor_divide

<div style="margin-left: 2em">

**Native Documentation**: [torch.floor_divide](https://pytorch.org/docs/2.12/generated/torch.floor_divide.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.fmod

<div style="margin-left: 2em">

**Native Documentation**: [torch.fmod](https://pytorch.org/docs/2.12/generated/torch.fmod.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

### torch.gradient

<div style="margin-left: 2em">

**Native Documentation**: [torch.gradient](https://pytorch.org/docs/2.12/generated/torch.gradient.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int8, int16, int32, int64

</div>

### torch.ldexp

<div style="margin-left: 2em">

**Native Documentation**: [torch.ldexp](https://pytorch.org/docs/2.12/generated/torch.ldexp.html)

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

**Constraints & Notes**: `input` only supports fp16, fp64, complex64

</div>

### torch.lerp

<div style="margin-left: 2em">

**Native Documentation**: [torch.lerp](https://pytorch.org/docs/2.12/generated/torch.lerp.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.log

<div style="margin-left: 2em">

**Native Documentation**: [torch.log](https://pytorch.org/docs/2.12/generated/torch.log.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.log10

<div style="margin-left: 2em">

**Native Documentation**: [torch.log10](https://pytorch.org/docs/2.12/generated/torch.log10.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- When the input `input` is uint8, int8, int16, int32, int64, or bool, the output `out` must be fp32
- For other supported data types, the output `out` must be consistent with the input `input`

</div>

### torch.log1p

<div style="margin-left: 2em">

**Native Documentation**: [torch.log1p](https://pytorch.org/docs/2.12/generated/torch.log1p.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.log2

<div style="margin-left: 2em">

**Native Documentation**: [torch.log2](https://pytorch.org/docs/2.12/generated/torch.log2.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.logaddexp

<div style="margin-left: 2em">

**Native Documentation**: [torch.logaddexp](https://pytorch.org/docs/2.12/generated/torch.logaddexp.html)

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

**Constraints & Notes**: Does not support the double data type.

</div>

### torch.logaddexp2

<div style="margin-left: 2em">

**Native Documentation**: [torch.logaddexp2](https://pytorch.org/docs/2.12/generated/torch.logaddexp2.html)

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

**Constraints & Notes**: Does not support the double data type.

</div>

### torch.logical_and

<div style="margin-left: 2em">

**Native Documentation**: [torch.logical_and](https://pytorch.org/docs/2.12/generated/torch.logical_and.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.logical_not

<div style="margin-left: 2em">

**Native Documentation**: [torch.logical_not](https://pytorch.org/docs/2.12/generated/torch.logical_not.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.logical_or

<div style="margin-left: 2em">

**Native Documentation**: [torch.logical_or](https://pytorch.org/docs/2.12/generated/torch.logical_or.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.logical_xor

<div style="margin-left: 2em">

**Native Documentation**: [torch.logical_xor](https://pytorch.org/docs/2.12/generated/torch.logical_xor.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.logit

<div style="margin-left: 2em">

**Native Documentation**: [torch.logit](https://pytorch.org/docs/2.12/generated/torch.logit.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- When the value of `eps` is greater than 1, the output is nan; when the value of `eps` is 1, the output is inf

</div>

### torch.mul

<div style="margin-left: 2em">

**Native Documentation**: [torch.mul](https://pytorch.org/docs/2.12/generated/torch.mul.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.multiply

<div style="margin-left: 2em">

**Native Documentation**: [torch.multiply](https://pytorch.org/docs/2.12/generated/torch.multiply.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nan_to_num

<div style="margin-left: 2em">

**Native Documentation**: [torch.nan_to_num](https://pytorch.org/docs/2.12/generated/torch.nan_to_num.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.neg

<div style="margin-left: 2em">

**Native Documentation**: [torch.neg](https://pytorch.org/docs/2.12/generated/torch.neg.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128

</div>

### torch.negative

<div style="margin-left: 2em">

**Native Documentation**: [torch.negative](https://pytorch.org/docs/2.12/generated/torch.negative.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128

</div>

### torch.positive

<div style="margin-left: 2em">

**Native Documentation**: [torch.positive](https://pytorch.org/docs/2.12/generated/torch.positive.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128

</div>

### torch.pow

<div style="margin-left: 2em">

**Native Documentation**: [torch.pow](https://pytorch.org/docs/2.12/generated/torch.pow.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int16, int64

</div>

### torch.rad2deg

<div style="margin-left: 2em">

**Native Documentation**: [torch.rad2deg](https://pytorch.org/docs/2.12/generated/torch.rad2deg.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.reciprocal

<div style="margin-left: 2em">

**Native Documentation**: [torch.reciprocal](https://pytorch.org/docs/2.12/generated/torch.reciprocal.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.remainder

<div style="margin-left: 2em">

**Native Documentation**: [torch.remainder](https://pytorch.org/docs/2.12/generated/torch.remainder.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, int16, int32, int64

</div>

### torch.round

<div style="margin-left: 2em">

**Native Documentation**: [torch.round](https://pytorch.org/docs/2.12/generated/torch.round.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, int32, int64

</div>

### torch.rsqrt

<div style="margin-left: 2em">

**Native Documentation**: [torch.rsqrt](https://pytorch.org/docs/2.12/generated/torch.rsqrt.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.sigmoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.sigmoid](https://pytorch.org/docs/2.12/generated/torch.sigmoid.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.sign

<div style="margin-left: 2em">

**Native Documentation**: [torch.sign](https://pytorch.org/docs/2.12/generated/torch.sign.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int32, int64, bool

</div>

### torch.sgn

<div style="margin-left: 2em">

**Native Documentation**: [torch.sgn](https://pytorch.org/docs/2.12/generated/torch.sgn.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int32, int64, bool, complex64, complex128

</div>

### torch.sin

<div style="margin-left: 2em">

**Native Documentation**: [torch.sin](https://pytorch.org/docs/2.12/generated/torch.sin.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.sinh

<div style="margin-left: 2em">

**Native Documentation**: [torch.sinh](https://pytorch.org/docs/2.12/generated/torch.sinh.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, fp64

</div>

### torch.softmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.softmax](https://pytorch.org/docs/2.12/generated/torch.softmax.html)

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

- `input` only supports fp32
- Named Tensor is supported

</div>

### torch.sqrt

<div style="margin-left: 2em">

**Native Documentation**: [torch.sqrt](https://pytorch.org/docs/2.12/generated/torch.sqrt.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.square

<div style="margin-left: 2em">

**Native Documentation**: [torch.square](https://pytorch.org/docs/2.12/generated/torch.square.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.sub

<div style="margin-left: 2em">

**Native Documentation**: [torch.sub](https://pytorch.org/docs/2.12/generated/torch.sub.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.tan

<div style="margin-left: 2em">

**Native Documentation**: [torch.tan](https://pytorch.org/docs/2.12/generated/torch.tan.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128
- The value range is [-65504, 65504]

</div>

### torch.tanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.tanh](https://pytorch.org/docs/2.12/generated/torch.tanh.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.true_divide

<div style="margin-left: 2em">

**Native Documentation**: [torch.true_divide](https://pytorch.org/docs/2.12/generated/torch.true_divide.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.trunc

<div style="margin-left: 2em">

**Native Documentation**: [torch.trunc](https://pytorch.org/docs/2.12/generated/torch.trunc.html)

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
- May fall back to CPU execution

</div>

### torch.xlogy

<div style="margin-left: 2em">

**Native Documentation**: [torch.xlogy](https://pytorch.org/docs/2.12/generated/torch.xlogy.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.argmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.argmax](https://pytorch.org/docs/2.12/generated/torch.argmax.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.argmin

<div style="margin-left: 2em">

**Native Documentation**: [torch.argmin](https://pytorch.org/docs/2.12/generated/torch.argmin.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.amax

<div style="margin-left: 2em">

**Native Documentation**: [torch.amax](https://pytorch.org/docs/2.12/generated/torch.amax.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.amin

<div style="margin-left: 2em">

**Native Documentation**: [torch.amin](https://pytorch.org/docs/2.12/generated/torch.amin.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.aminmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.aminmax](https://pytorch.org/docs/2.12/generated/torch.aminmax.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.all

<div style="margin-left: 2em">

**Native Documentation**: [torch.all](https://pytorch.org/docs/2.12/generated/torch.all.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.any

<div style="margin-left: 2em">

**Native Documentation**: [torch.any](https://pytorch.org/docs/2.12/generated/torch.any.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.max

<div style="margin-left: 2em">

**Native Documentation**: [torch.max](https://pytorch.org/docs/2.12/generated/torch.max.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int64, bool

</div>

### torch.min

<div style="margin-left: 2em">

**Native Documentation**: [torch.min](https://pytorch.org/docs/2.12/generated/torch.min.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int64, bool

</div>

### torch.dist

<div style="margin-left: 2em">

**Native Documentation**: [torch.dist](https://pytorch.org/docs/2.12/generated/torch.dist.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.logsumexp

<div style="margin-left: 2em">

**Native Documentation**: [torch.logsumexp](https://pytorch.org/docs/2.12/generated/torch.logsumexp.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.mean

<div style="margin-left: 2em">

**Native Documentation**: [torch.mean](https://pytorch.org/docs/2.12/generated/torch.mean.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, complex64, complex128

</div>

### torch.nanmean

<div style="margin-left: 2em">

**Native Documentation**: [torch.nanmean](https://pytorch.org/docs/2.12/generated/torch.nanmean.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.median

<div style="margin-left: 2em">

**Native Documentation**: [torch.median](https://pytorch.org/docs/2.12/generated/torch.median.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.norm](https://pytorch.org/docs/2.12/generated/torch.norm.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32
- When the `dim` parameter specifies an axis whose shape dimension value in the input `tensor` is 1, the computation result may have precision errors

</div>

### torch.nansum

<div style="margin-left: 2em">

**Native Documentation**: [torch.nansum](https://pytorch.org/docs/2.12/generated/torch.nansum.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.prod

<div style="margin-left: 2em">

**Native Documentation**: [torch.prod](https://pytorch.org/docs/2.12/generated/torch.prod.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.nanquantile

<div style="margin-left: 2em">

**Native Documentation**: [torch.nanquantile](https://pytorch.org/docs/2.12/generated/torch.nanquantile.html)

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

</div>

### torch.std

<div style="margin-left: 2em">

**Native Documentation**: [torch.std](https://pytorch.org/docs/2.12/generated/torch.std.html)

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

**Constraints & Notes**: May fall back to CPU execution

</div>

### torch.std_mean

<div style="margin-left: 2em">

**Native Documentation**: [torch.std_mean](https://pytorch.org/docs/2.12/generated/torch.std_mean.html)

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

### torch.sum

<div style="margin-left: 2em">

**Native Documentation**: [torch.sum](https://pytorch.org/docs/2.12/generated/torch.sum.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- The `dtype` parameter is not supported

</div>

### torch.unique

<div style="margin-left: 2em">

**Native Documentation**: [torch.unique](https://pytorch.org/docs/2.12/generated/torch.unique.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- fp16 is not supported in the scenario with `dim`
- When the input contains 0, the output may contain positive 0 and negative 0 instead of only one 0

</div>

### torch.unique_consecutive

<div style="margin-left: 2em">

**Native Documentation**: [torch.unique_consecutive](https://pytorch.org/docs/2.12/generated/torch.unique_consecutive.html)

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

### torch.var

<div style="margin-left: 2em">

**Native Documentation**: [torch.var](https://pytorch.org/docs/2.12/generated/torch.var.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.var_mean

<div style="margin-left: 2em">

**Native Documentation**: [torch.var_mean](https://pytorch.org/docs/2.12/generated/torch.var_mean.html)

**Support Status**:

<!-- npu="910b" id732 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id732 -->
<!-- npu="A3" id733 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id733 -->
<!-- npu="950" id734 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id734 -->

</div>

### torch.count_nonzero

<div style="margin-left: 2em">

**Native Documentation**: [torch.count_nonzero](https://pytorch.org/docs/2.12/generated/torch.count_nonzero.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.allclose

<div style="margin-left: 2em">

**Native Documentation**: [torch.allclose](https://pytorch.org/docs/2.12/generated/torch.allclose.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.argsort

<div style="margin-left: 2em">

**Native Documentation**: [torch.argsort](https://pytorch.org/docs/2.12/generated/torch.argsort.html)

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

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

<!-- npu="950" id744 -->
- For <term>Ascend 950DT products</term>, due to underlying implementation limitations, `stable` only supports True. If set to False, it is automatically changed to True at execution time.
<!-- end id744 -->

</div>

### torch.eq

<div style="margin-left: 2em">

**Native Documentation**: [torch.eq](https://pytorch.org/docs/2.12/generated/torch.eq.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.equal

<div style="margin-left: 2em">

**Native Documentation**: [torch.equal](https://pytorch.org/docs/2.12/generated/torch.equal.html)

**Support Status**:

<!-- npu="910b" id748 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id748 -->
<!-- npu="A3" id749 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id749 -->
<!-- npu="950" id750 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id750 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.ge

<div style="margin-left: 2em">

**Native Documentation**: [torch.ge](https://pytorch.org/docs/2.12/generated/torch.ge.html)

**Support Status**:

<!-- npu="910b" id751 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id751 -->
<!-- npu="A3" id752 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id752 -->
<!-- npu="950" id753 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id753 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.greater_equal

<div style="margin-left: 2em">

**Native Documentation**: [torch.greater_equal](https://pytorch.org/docs/2.12/generated/torch.greater_equal.html)

**Support Status**:

<!-- npu="910b" id754 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id754 -->
<!-- npu="A3" id755 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id755 -->
<!-- npu="950" id756 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id756 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.gt

<div style="margin-left: 2em">

**Native Documentation**: [torch.gt](https://pytorch.org/docs/2.12/generated/torch.gt.html)

**Support Status**:

<!-- npu="910b" id757 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id757 -->
<!-- npu="A3" id758 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id758 -->
<!-- npu="950" id759 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id759 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.greater

<div style="margin-left: 2em">

**Native Documentation**: [torch.greater](https://pytorch.org/docs/2.12/generated/torch.greater.html)

**Support Status**:

<!-- npu="910b" id760 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id760 -->
<!-- npu="A3" id761 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id761 -->
<!-- npu="950" id762 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id762 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.isclose

<div style="margin-left: 2em">

**Native Documentation**: [torch.isclose](https://pytorch.org/docs/2.12/generated/torch.isclose.html)

**Support Status**:

<!-- npu="910b" id763 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id763 -->
<!-- npu="A3" id764 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id764 -->
<!-- npu="950" id765 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id765 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.isfinite

<div style="margin-left: 2em">

**Native Documentation**: [torch.isfinite](https://pytorch.org/docs/2.12/generated/torch.isfinite.html)

**Support Status**:

<!-- npu="910b" id766 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id766 -->
<!-- npu="A3" id767 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id767 -->
<!-- npu="950" id768 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id768 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.isin

<div style="margin-left: 2em">

**Native Documentation**: [torch.isin](https://pytorch.org/docs/2.12/generated/torch.isin.html)

**Support Status**:

<!-- npu="910b" id769 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id769 -->
<!-- npu="A3" id770 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id770 -->
<!-- npu="950" id771 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id771 -->

**Constraints & Notes**:

- The constraints for the scenario with two `tensor` inputs are as follows:
  - `elements` and `test_elements` only support fp16, fp32, uint8, int8, int16, int32, and int64.
  - The first input `tensor` must have no more than 7 dimensions, and the second input `tensor` must have no more than 8 dimensions
- The constraints for the scenario with a single `tensor` input are as follows:
  - `elements` and `test_elements` only support bf16, fp16, fp32, fp64, uint8, int8, int16, int32, and int64.
  - The input `tensor` must have no more than 8 dimensions

</div>

### torch.isinf

<div style="margin-left: 2em">

**Native Documentation**: [torch.isinf](https://pytorch.org/docs/2.12/generated/torch.isinf.html)

**Support Status**:

<!-- npu="910b" id772 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id772 -->
<!-- npu="A3" id773 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id773 -->
<!-- npu="950" id774 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id774 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.isposinf

<div style="margin-left: 2em">

**Native Documentation**: [torch.isposinf](https://pytorch.org/docs/2.12/generated/torch.isposinf.html)

**Support Status**:

<!-- npu="910b" id775 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id775 -->
<!-- npu="A3" id776 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id776 -->
<!-- npu="950" id777 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id777 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.isneginf

<div style="margin-left: 2em">

**Native Documentation**: [torch.isneginf](https://pytorch.org/docs/2.12/generated/torch.isneginf.html)

**Support Status**:

<!-- npu="910b" id778 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id778 -->
<!-- npu="A3" id779 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id779 -->
<!-- npu="950" id780 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id780 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.isnan

<div style="margin-left: 2em">

**Native Documentation**: [torch.isnan](https://pytorch.org/docs/2.12/generated/torch.isnan.html)

**Support Status**:

<!-- npu="910b" id781 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id781 -->
<!-- npu="A3" id782 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id782 -->
<!-- npu="950" id783 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id783 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.kthvalue

<div style="margin-left: 2em">

**Native Documentation**: [torch.kthvalue](https://pytorch.org/docs/2.12/generated/torch.kthvalue.html)

**Support Status**:

<!-- npu="910b" id784 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id784 -->
<!-- npu="A3" id785 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id785 -->
<!-- npu="950" id786 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id786 -->

**Constraints & Notes**: `input` only supports fp16, fp32, int32

</div>

### torch.le

<div style="margin-left: 2em">

**Native Documentation**: [torch.le](https://pytorch.org/docs/2.12/generated/torch.le.html)

**Support Status**:

<!-- npu="910b" id787 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id787 -->
<!-- npu="A3" id788 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id788 -->
<!-- npu="950" id789 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id789 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.less_equal

<div style="margin-left: 2em">

**Native Documentation**: [torch.less_equal](https://pytorch.org/docs/2.12/generated/torch.less_equal.html)

**Support Status**:

<!-- npu="910b" id790 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id790 -->
<!-- npu="A3" id791 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id791 -->
<!-- npu="950" id792 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id792 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, uint16, int8, int16, int32, int64, bool

</div>

### torch.lt

<div style="margin-left: 2em">

**Native Documentation**: [torch.lt](https://pytorch.org/docs/2.12/generated/torch.lt.html)

**Support Status**:

<!-- npu="910b" id793 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id793 -->
<!-- npu="A3" id794 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id794 -->
<!-- npu="950" id795 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id795 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.less

<div style="margin-left: 2em">

**Native Documentation**: [torch.less](https://pytorch.org/docs/2.12/generated/torch.less.html)

**Support Status**:

<!-- npu="910b" id796 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id796 -->
<!-- npu="A3" id797 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id797 -->
<!-- npu="950" id798 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id798 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.maximum

<div style="margin-left: 2em">

**Native Documentation**: [torch.maximum](https://pytorch.org/docs/2.12/generated/torch.maximum.html)

**Support Status**:

<!-- npu="910b" id799 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id799 -->
<!-- npu="A3" id800 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id800 -->
<!-- npu="950" id801 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id801 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.minimum

<div style="margin-left: 2em">

**Native Documentation**: [torch.minimum](https://pytorch.org/docs/2.12/generated/torch.minimum.html)

**Support Status**:

<!-- npu="910b" id802 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id802 -->
<!-- npu="A3" id803 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id803 -->
<!-- npu="950" id804 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id804 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.ne

<div style="margin-left: 2em">

**Native Documentation**: [torch.ne](https://pytorch.org/docs/2.12/generated/torch.ne.html)

**Support Status**:

<!-- npu="910b" id805 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id805 -->
<!-- npu="A3" id806 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id806 -->
<!-- npu="950" id807 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id807 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.not_equal

<div style="margin-left: 2em">

**Native Documentation**: [torch.not_equal](https://pytorch.org/docs/2.12/generated/torch.not_equal.html)

**Support Status**:

<!-- npu="910b" id808 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id808 -->
<!-- npu="A3" id809 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id809 -->
<!-- npu="950" id810 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id810 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.sort

<div style="margin-left: 2em">

**Native Documentation**: [torch.sort](https://pytorch.org/docs/2.12/generated/torch.sort.html)

**Support Status**:

<!-- npu="910b" id811 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id811 -->
<!-- npu="A3" id812 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id812 -->
<!-- npu="950" id813 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id813 -->

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

<!-- npu="950" id814 -->
- For <term>Ascend 950DT products</term>, due to underlying implementation limitations, `stable` only supports True. If set to False, it is automatically changed to True at execution time.
<!-- end id814 -->

</div>

### torch.topk

<div style="margin-left: 2em">

**Native Documentation**: [torch.topk](https://pytorch.org/docs/2.12/generated/torch.topk.html)

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

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64
- The scenario with `sorted=False` is not supported

</div>

### torch.msort

<div style="margin-left: 2em">

**Native Documentation**: [torch.msort](https://pytorch.org/docs/2.12/generated/torch.msort.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.stft

<div style="margin-left: 2em">

**Native Documentation**: [torch.stft](https://pytorch.org/docs/2.12/generated/torch.stft.html)

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

**Constraints & Notes**:

- `input` only supports fp32, fp64, complex64, complex128
- If an operator times out, use the official interface `set_op_execute_time_out` to increase the timeout threshold to extend the determination time

</div>

### torch.hann_window

<div style="margin-left: 2em">

**Native Documentation**: [torch.hann_window](https://pytorch.org/docs/2.12/generated/torch.hann_window.html)

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

**Constraints & Notes**:

- `dtype` only supports bf16, fp16, fp32
- When the data type is fp32 and the `window_length` parameter is greater than 10000, the computation result may have errors

</div>

### torch.atleast_1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.atleast_1d](https://pytorch.org/docs/2.12/generated/torch.atleast_1d.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.atleast_2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.atleast_2d](https://pytorch.org/docs/2.12/generated/torch.atleast_2d.html)

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

**Constraints & Notes**: `tensors` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.atleast_3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.atleast_3d](https://pytorch.org/docs/2.12/generated/torch.atleast_3d.html)

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

**Constraints & Notes**: `tensors` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.bincount

<div style="margin-left: 2em">

**Native Documentation**: [torch.bincount](https://pytorch.org/docs/2.12/generated/torch.bincount.html)

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

**Constraints & Notes**:

- `input` only supports uint8, int8, int16, int32, int64
- The dimension of `weights` must be the same as that of `input`

</div>

### torch.block_diag

<div style="margin-left: 2em">

**Native Documentation**: [torch.block_diag](https://pytorch.org/docs/2.12/generated/torch.block_diag.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.broadcast_tensors

<div style="margin-left: 2em">

**Native Documentation**: [torch.broadcast_tensors](https://pytorch.org/docs/2.12/generated/torch.broadcast_tensors.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.broadcast_to

<div style="margin-left: 2em">

**Native Documentation**: [torch.broadcast_to](https://pytorch.org/docs/2.12/generated/torch.broadcast_to.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.broadcast_shapes

<div style="margin-left: 2em">

**Native Documentation**: [torch.broadcast_shapes](https://pytorch.org/docs/2.12/generated/torch.broadcast_shapes.html)

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

</div>

### torch.cdist

<div style="margin-left: 2em">

**Native Documentation**: [torch.cdist](https://pytorch.org/docs/2.12/generated/torch.cdist.html)

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

- `x1` and `x2` only support bf16, fp16, and fp32.
- When `p=2.0`, `compute_mode` only supports the "donot_use_mm_for_euclid_dist" mode. If other values are passed, they are automatically changed to this mode.

<!-- npu="950,A3,910b" id854 -->
- For <term>Ascend 950DT products</term>, when the input is fp16, the precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>.
<!-- end id854 -->

</div>

### torch.clone

<div style="margin-left: 2em">

**Native Documentation**: [torch.clone](https://pytorch.org/docs/2.12/generated/torch.clone.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.combinations

<div style="margin-left: 2em">

**Native Documentation**: [torch.combinations](https://pytorch.org/docs/2.12/generated/torch.combinations.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.cov

<div style="margin-left: 2em">

**Native Documentation**: [torch.cov](https://pytorch.org/docs/2.12/generated/torch.cov.html)

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

</div>

### torch.cross

<div style="margin-left: 2em">

**Native Documentation**: [torch.cross](https://pytorch.org/docs/2.12/generated/torch.cross.html)

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

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128
- The shapes of the two inputs must be consistent

</div>

### torch.cummax

<div style="margin-left: 2em">

**Native Documentation**: [torch.cummax](https://pytorch.org/docs/2.12/generated/torch.cummax.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.cummin

<div style="margin-left: 2em">

**Native Documentation**: [torch.cummin](https://pytorch.org/docs/2.12/generated/torch.cummin.html)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- When the input is int32, the value range is within [-16777216, 16777216]

</div>

### torch.cumprod

<div style="margin-left: 2em">

**Native Documentation**: [torch.cumprod](https://pytorch.org/docs/2.12/generated/torch.cumprod.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch.cumsum

<div style="margin-left: 2em">

**Native Documentation**: [torch.cumsum](https://pytorch.org/docs/2.12/generated/torch.cumsum.html)

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

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128
- Named Tensor is supported

</div>

### torch.diag

<div style="margin-left: 2em">

**Native Documentation**: [torch.diag](https://pytorch.org/docs/2.12/generated/torch.diag.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64

</div>

### torch.diag_embed

<div style="margin-left: 2em">

**Native Documentation**: [torch.diag_embed](https://pytorch.org/docs/2.12/generated/torch.diag_embed.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.diagonal

<div style="margin-left: 2em">

**Native Documentation**: [torch.diagonal](https://pytorch.org/docs/2.12/generated/torch.diagonal.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.diff

<div style="margin-left: 2em">

**Native Documentation**: [torch.diff](https://pytorch.org/docs/2.12/generated/torch.diff.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.einsum

<div style="margin-left: 2em">

**Native Documentation**: [torch.einsum](https://pytorch.org/docs/2.12/generated/torch.einsum.html)

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

**Constraints & Notes**: `operands` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.flatten

<div style="margin-left: 2em">

**Native Documentation**: [torch.flatten](https://pytorch.org/docs/2.12/generated/torch.flatten.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.flip

<div style="margin-left: 2em">

**Native Documentation**: [torch.flip](https://pytorch.org/docs/2.12/generated/torch.flip.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.fliplr

<div style="margin-left: 2em">

**Native Documentation**: [torch.fliplr](https://pytorch.org/docs/2.12/generated/torch.fliplr.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.flipud

<div style="margin-left: 2em">

**Native Documentation**: [torch.flipud](https://pytorch.org/docs/2.12/generated/torch.flipud.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.kron

<div style="margin-left: 2em">

**Native Documentation**: [torch.kron](https://pytorch.org/docs/2.12/generated/torch.kron.html)

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

**Constraints & Notes**: Does not support inputs with 5 or more dimensions.

</div>

### torch.histc

<div style="margin-left: 2em">

**Native Documentation**: [torch.histc](https://pytorch.org/docs/2.12/generated/torch.histc.html)

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

**Constraints & Notes**:

- `input` only supports fp16, fp32
- When the value of the input `tensor` is at the boundary between counting intervals, there may be an error in whether the value is counted in the left interval or the right interval

</div>

### torch.meshgrid

<div style="margin-left: 2em">

**Native Documentation**: [torch.meshgrid](https://pytorch.org/docs/2.12/generated/torch.meshgrid.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.ravel

<div style="margin-left: 2em">

**Native Documentation**: [torch.ravel](https://pytorch.org/docs/2.12/generated/torch.ravel.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.repeat_interleave

<div style="margin-left: 2em">

**Native Documentation**: [torch.repeat_interleave](https://pytorch.org/docs/2.12/generated/torch.repeat_interleave.html)

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

**Constraints & Notes**:

- `input` only supports fp16, fp32, int16, int32, int64, bool
- After the input tensor is repeated to obtain the output, the number of elements in the output must be less than $2^{22}$

</div>

### torch.roll

<div style="margin-left: 2em">

**Native Documentation**: [torch.roll](https://pytorch.org/docs/2.12/generated/torch.roll.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, int32, int64, bool

</div>

### torch.searchsorted

<div style="margin-left: 2em">

**Native Documentation**: [torch.searchsorted](https://pytorch.org/docs/2.12/generated/torch.searchsorted.html)

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

**Constraints & Notes**: `sorted_sequence` and `values` only support fp16, fp32, fp64, uint8, int8, int16, int32, and int64.

</div>

### torch.tensordot

<div style="margin-left: 2em">

**Native Documentation**: [torch.tensordot](https://pytorch.org/docs/2.12/generated/torch.tensordot.html)

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

**Constraints & Notes**: `a` and `b` only support fp16 and fp32.

</div>

### torch.tril

<div style="margin-left: 2em">

**Native Documentation**: [torch.tril](https://pytorch.org/docs/2.12/generated/torch.tril.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.tril_indices

<div style="margin-left: 2em">

**Native Documentation**: [torch.tril_indices](https://pytorch.org/docs/2.12/generated/torch.tril_indices.html)

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

### torch.triu

<div style="margin-left: 2em">

**Native Documentation**: [torch.triu](https://pytorch.org/docs/2.12/generated/torch.triu.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.triu_indices

<div style="margin-left: 2em">

**Native Documentation**: [torch.triu_indices](https://pytorch.org/docs/2.12/generated/torch.triu_indices.html)

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

</div>

### torch.unflatten

<div style="margin-left: 2em">

**Native Documentation**: [torch.unflatten](https://pytorch.org/docs/2.12/generated/torch.unflatten.html)

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

</div>

### torch.view_as_real

<div style="margin-left: 2em">

**Native Documentation**: [torch.view_as_real](https://pytorch.org/docs/2.12/generated/torch.view_as_real.html)

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

**Constraints & Notes**: `input` only supports complex64, complex128

</div>

### torch.view_as_complex

<div style="margin-left: 2em">

**Native Documentation**: [torch.view_as_complex](https://pytorch.org/docs/2.12/generated/torch.view_as_complex.html)

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

**Constraints & Notes**: `input` only supports fp32, fp64

</div>

### torch.resolve_conj

<div style="margin-left: 2em">

**Native Documentation**: [torch.resolve_conj](https://pytorch.org/docs/2.12/generated/torch.resolve_conj.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.resolve_neg

<div style="margin-left: 2em">

**Native Documentation**: [torch.resolve_neg](https://pytorch.org/docs/2.12/generated/torch.resolve_neg.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.addbmm

<div style="margin-left: 2em">

**Native Documentation**: [torch.addbmm](https://pytorch.org/docs/2.12/generated/torch.addbmm.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.addmm

<div style="margin-left: 2em">

**Native Documentation**: [torch.addmm](https://pytorch.org/docs/2.12/generated/torch.addmm.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.addmv

<div style="margin-left: 2em">

**Native Documentation**: [torch.addmv](https://pytorch.org/docs/2.12/generated/torch.addmv.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.addr

<div style="margin-left: 2em">

**Native Documentation**: [torch.addr](https://pytorch.org/docs/2.12/generated/torch.addr.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.baddbmm

<div style="margin-left: 2em">

**Native Documentation**: [torch.baddbmm](https://pytorch.org/docs/2.12/generated/torch.baddbmm.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.bmm

<div style="margin-left: 2em">

**Native Documentation**: [torch.bmm](https://pytorch.org/docs/2.12/generated/torch.bmm.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.dot

<div style="margin-left: 2em">

**Native Documentation**: [torch.dot](https://pytorch.org/docs/2.12/generated/torch.dot.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32

</div>

### torch.slogdet

<div style="margin-left: 2em">

**Native Documentation**: [torch.slogdet](https://pytorch.org/docs/2.12/generated/torch.slogdet.html)

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

**Constraints & Notes**:

- `input` only supports fp32, complex64, complex128
- May fall back to CPU execution

</div>

### torch.matmul

<div style="margin-left: 2em">

**Native Documentation**: [torch.matmul](https://pytorch.org/docs/2.12/generated/torch.matmul.html)

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

**Constraints & Notes**:

- `input` only supports fp16, fp32
- Named Tensor is supported
- The input supports a maximum of 6 dimensions.

</div>

### torch.mm

<div style="margin-left: 2em">

**Native Documentation**: [torch.mm](https://pytorch.org/docs/2.12/generated/torch.mm.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.outer

<div style="margin-left: 2em">

**Native Documentation**: [torch.outer](https://pytorch.org/docs/2.12/generated/torch.outer.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.qr

<div style="margin-left: 2em">

**Native Documentation**: [torch.qr](https://pytorch.org/docs/2.12/generated/torch.qr.html)

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

### torch.trapezoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.trapezoid](https://pytorch.org/docs/2.12/generated/torch.trapezoid.html)

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

**Constraints & Notes**: `y` and `x` only support bf16, fp16, fp32, uint8, int8, int16, int32, and int64.

</div>

### torch.cumulative_trapezoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.cumulative_trapezoid](https://pytorch.org/docs/2.12/generated/torch.cumulative_trapezoid.html)

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

**Constraints & Notes**: `y` and `x` only support fp16, fp32, fp64, uint8, int8, int16, int32, and int64.

</div>

### torch.vdot

<div style="margin-left: 2em">

**Native Documentation**: [torch.vdot](https://pytorch.org/docs/2.12/generated/torch.vdot.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch.bucketize

<div style="margin-left: 2em">

**Native Documentation**: [torch.bucketize](https://pytorch.org/docs/2.12/generated/torch.bucketize.html)

**Support Status**:

<!-- npu="910b" id1002 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1002 -->
<!-- npu="A3" id1003 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1003 -->
<!-- npu="950" id1004 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1004 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64

</div>

### torch.cartesian_prod

<div style="margin-left: 2em">

**Native Documentation**: [torch.cartesian_prod](https://pytorch.org/docs/2.12/generated/torch.cartesian_prod.html)

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

**Constraints & Notes**: `tensors` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.mv

<div style="margin-left: 2em">

**Native Documentation**: [torch.mv](https://pytorch.org/docs/2.12/generated/torch.mv.html)

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

**Constraints & Notes**: `input` only supports fp16, fp32

</div>

### torch._foreach_sqrt

<div style="margin-left: 2em">

**Native Documentation**: [torch._foreach_sqrt](https://pytorch.org/docs/2.12/generated/torch._foreach_sqrt.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch._foreach_asin

<div style="margin-left: 2em">

**Native Documentation**: [torch._foreach_asin](https://pytorch.org/docs/2.12/generated/torch._foreach_asin.html)

**Support Status**:

<!-- npu="910b" id1014 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1014 -->
<!-- npu="A3" id1015 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1015 -->
<!-- npu="950" id1016 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1016 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.\_foreach_neg\_

<div style="margin-left: 2em">

**Native Documentation**: [torch.\\_foreach_neg_](https://pytorch.org/docs/2.12/generated/torch._foreach_neg_.html#torch._foreach_neg_)

**Support Status**:

<!-- npu="910b" id1017 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1017 -->
<!-- npu="A3" id1018 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1018 -->
<!-- npu="950" id1019 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1019 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, int8, int32, int64

</div>

### torch.corrcoef

<div style="margin-left: 2em">

**Native Documentation**: [torch.corrcoef](https://pytorch.org/docs/2.12/generated/torch.corrcoef.html)

**Support Status**:

<!-- npu="910b" id1020 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1020 -->
<!-- npu="A3" id1021 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1021 -->
<!-- npu="950" id1022 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1022 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

## Utilities

### torch.compiled_with_cxx11_abi

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiled_with_cxx11_abi](https://pytorch.org/docs/2.12/generated/torch.compiled_with_cxx11_abi.html)

**Support Status**:

<!-- npu="910b" id1023 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1023 -->
<!-- npu="A3" id1024 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1024 -->
<!-- npu="950" id1025 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1025 -->

</div>

### torch.result_type

<div style="margin-left: 2em">

**Native Documentation**: [torch.result_type](https://pytorch.org/docs/2.12/generated/torch.result_type.html)

**Support Status**:

<!-- npu="910b" id1026 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1026 -->
<!-- npu="A3" id1027 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1027 -->
<!-- npu="950" id1028 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1028 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.can_cast

<div style="margin-left: 2em">

**Native Documentation**: [torch.can_cast](https://pytorch.org/docs/2.12/generated/torch.can_cast.html)

**Support Status**:

<!-- npu="910b" id1029 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1029 -->
<!-- npu="A3" id1030 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1030 -->
<!-- npu="950" id1031 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1031 -->

</div>

### torch.promote_types

<div style="margin-left: 2em">

**Native Documentation**: [torch.promote_types](https://pytorch.org/docs/2.12/generated/torch.promote_types.html)

**Support Status**:

<!-- npu="910b" id1032 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1032 -->
<!-- npu="A3" id1033 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1033 -->
<!-- npu="950" id1034 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1034 -->

</div>

### torch.use_deterministic_algorithms

<div style="margin-left: 2em">

**Native Documentation**: [torch.use_deterministic_algorithms](https://pytorch.org/docs/2.12/generated/torch.use_deterministic_algorithms.html)

**Support Status**:

<!-- npu="910b" id1035 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1035 -->
<!-- npu="A3" id1036 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1036 -->
<!-- npu="950" id1037 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1037 -->

**Constraints & Notes**:

- When HCCL_DETERMINISTIC and `torch.use_deterministic_algorithms` are set at the same time, if HCCL_DETERMINISTIC enables determinism, the HCCL APIs use deterministic mode; otherwise, HCCL determinism is controlled by the `torch.use_deterministic_algorithms` API.
- When `torch.use_deterministic_algorithms` is set, PyTorch fills uninitialized memory by default (the default value of `torch.utils.deterministic.fill_uninitialized_memory` is True), while TorchNPU does not fill it by default. To make the filling take effect, manually set `torch.utils.deterministic.fill_uninitialized_memory` to True.

</div>

### torch.are_deterministic_algorithms_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.are_deterministic_algorithms_enabled](https://pytorch.org/docs/2.12/generated/torch.are_deterministic_algorithms_enabled.html)

**Support Status**:

<!-- npu="910b" id1038 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1038 -->
<!-- npu="A3" id1039 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1039 -->
<!-- npu="950" id1040 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1040 -->

</div>

### torch.is_deterministic_algorithms_warn_only_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_deterministic_algorithms_warn_only_enabled](https://pytorch.org/docs/2.12/generated/torch.is_deterministic_algorithms_warn_only_enabled.html)

**Support Status**:

<!-- npu="910b" id1041 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id1041 -->
<!-- npu="A3" id1042 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id1042 -->
<!-- npu="950" id1043 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1043 -->

</div>

### torch.set_deterministic_debug_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_deterministic_debug_mode](https://pytorch.org/docs/2.12/generated/torch.set_deterministic_debug_mode.html)

**Support Status**:

<!-- npu="910b" id1044 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1044 -->
<!-- npu="A3" id1045 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1045 -->
<!-- npu="950" id1046 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1046 -->

</div>

### torch.get_deterministic_debug_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_deterministic_debug_mode](https://pytorch.org/docs/2.12/generated/torch.get_deterministic_debug_mode.html)

**Support Status**:

<!-- npu="910b" id1047 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1047 -->
<!-- npu="A3" id1048 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1048 -->
<!-- npu="950" id1049 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1049 -->

</div>

### torch.set_float32_matmul_precision

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_float32_matmul_precision](https://pytorch.org/docs/2.12/generated/torch.set_float32_matmul_precision.html)

**Support Status**:

<!-- npu="910b" id1050 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1050 -->
<!-- npu="A3" id1051 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1051 -->
<!-- npu="950" id1052 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1052 -->

**Constraints & Notes**: This API can be called on the NPU, but it has no actual effect.

</div>

### torch.get_float32_matmul_precision

<div style="margin-left: 2em">

**Native Documentation**: [torch.get_float32_matmul_precision](https://pytorch.org/docs/2.12/generated/torch.get_float32_matmul_precision.html)

**Support Status**:

<!-- npu="910b" id1053 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1053 -->
<!-- npu="A3" id1054 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1054 -->
<!-- npu="950" id1055 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1055 -->

</div>

### torch.set_warn_always

<div style="margin-left: 2em">

**Native Documentation**: [torch.set_warn_always](https://pytorch.org/docs/2.12/generated/torch.set_warn_always.html)

**Support Status**:

<!-- npu="910b" id1056 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1056 -->
<!-- npu="A3" id1057 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1057 -->
<!-- npu="950" id1058 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1058 -->

</div>

### torch.is_warn_always_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.is_warn_always_enabled](https://pytorch.org/docs/2.12/generated/torch.is_warn_always_enabled.html)

**Support Status**:

<!-- npu="910b" id1059 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1059 -->
<!-- npu="A3" id1060 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1060 -->
<!-- npu="950" id1061 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1061 -->

</div>

### torch.vmap

<div style="margin-left: 2em">

**Native Documentation**: [torch.vmap](https://pytorch.org/docs/2.12/generated/torch.vmap.html)

**Support Status**:

<!-- npu="910b" id1062 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1062 -->
<!-- npu="A3" id1063 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1063 -->
<!-- npu="950" id1064 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1064 -->

</div>

### torch._assert

<div style="margin-left: 2em">

**Native Documentation**: [torch._assert](https://pytorch.org/docs/2.12/generated/torch._assert.html)

**Support Status**:

<!-- npu="910b" id1065 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1065 -->
<!-- npu="A3" id1066 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1066 -->
<!-- npu="950" id1067 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1067 -->

</div>

## Symbolic Numbers

### <code><i>class</i></code> torch.SymBool

<div style="margin-left: 2em">

**Native Documentation**: [torch.SymBool](https://pytorch.org/docs/2.12/torch.html#torch.SymBool)

**Support Status**:

<!-- npu="910b" id1068 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1068 -->
<!-- npu="A3" id1069 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1069 -->
<!-- npu="950" id1070 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1070 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.SymInt

<div style="margin-left: 2em">

**Native Documentation**: [torch.SymInt](https://pytorch.org/docs/2.12/torch.html#torch.SymInt)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.SymFloat

<div style="margin-left: 2em">

**Native Documentation**: [torch.SymFloat](https://pytorch.org/docs/2.12/torch.html#torch.SymFloat)

**Support Status**:

<!-- npu="910b" id1074 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1074 -->
<!-- npu="A3" id1075 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1075 -->
<!-- npu="950" id1076 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1076 -->

**Constraints & Notes**: -

</div>

### torch.sym_float

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_float](https://pytorch.org/docs/2.12/generated/torch.sym_float.html)

**Support Status**:

<!-- npu="910b" id1077 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1077 -->
<!-- npu="A3" id1078 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1078 -->
<!-- npu="950" id1079 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1079 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.sym_int

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_int](https://pytorch.org/docs/2.12/generated/torch.sym_int.html)

**Support Status**:

<!-- npu="910b" id1080 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1080 -->
<!-- npu="A3" id1081 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1081 -->
<!-- npu="950" id1082 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1082 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.sym_ite

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_ite](https://pytorch.org/docs/2.12/generated/torch.sym_ite.html)

**Support Status**:

<!-- npu="910b" id1083 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1083 -->
<!-- npu="A3" id1084 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1084 -->
<!-- npu="950" id1085 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1085 -->

</div>

### torch.sym_max

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_max](https://pytorch.org/docs/2.12/generated/torch.sym_max.html)

**Support Status**:

<!-- npu="910b" id1086 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1086 -->
<!-- npu="A3" id1087 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1087 -->
<!-- npu="950" id1088 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1088 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.sym_min

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_min](https://pytorch.org/docs/2.12/generated/torch.sym_min.html)

**Support Status**:

<!-- npu="910b" id1089 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1089 -->
<!-- npu="A3" id1090 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1090 -->
<!-- npu="950" id1091 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1091 -->

</div>

### torch.sym_not

<div style="margin-left: 2em">

**Native Documentation**: [torch.sym_not](https://pytorch.org/docs/2.12/generated/torch.sym_not.html)

**Support Status**:

<!-- npu="910b" id1092 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1092 -->
<!-- npu="A3" id1093 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1093 -->
<!-- npu="950" id1094 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1094 -->

</div>

## Optimizations

### torch.compile

<div style="margin-left: 2em">

**Native Documentation**: [torch.compile](https://pytorch.org/docs/2.12/generated/torch.compile.html)

**Support Status**:

<!-- npu="910b" id1095 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1095 -->
<!-- npu="A3" id1096 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1096 -->
<!-- npu="950" id1097 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1097 -->

**Constraints & Notes**: `backend` supports npugraphs, and the overall functionality is the same as `backend`="cudagraphs".

</div>

## Operator Tags

### <code><i>class</i></code> torch.Tag

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tag](https://pytorch.org/docs/2.12/torch.html#torch.Tag)

**Support Status**:

<!-- npu="910b" id1098 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1098 -->
<!-- npu="A3" id1099 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1099 -->
<!-- npu="950" id1100 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1100 -->

> <font size="3">name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tag.name](https://pytorch.org/docs/2.12/torch.html#torch.Tag.name)

**Support Status**:

<!-- npu="910b" id1101 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1101 -->
<!-- npu="A3" id1102 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1102 -->
<!-- npu="950" id1103 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1103 -->

</div>

</div>
