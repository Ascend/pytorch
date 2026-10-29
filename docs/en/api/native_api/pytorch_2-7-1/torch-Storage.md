# torch.Storage

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/storage.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Special cases](#special-cases)
- [Legacy Typed Storage](#legacy-typed-storage)

</div>

<div style="display:none;">

## &#8203;torch.Storage

</div>

## Special cases

### <code><i>class</i></code> torch.UntypedStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage)

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

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.bfloat16](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.bfloat16)

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

</div>

> <font size="3">bool()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.bool](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.bool)

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

> <font size="3">byte()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.byte](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.byte)

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

> <font size="3">byteswap()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.byteswap](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.byteswap)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id15 -->

</div>

> <font size="3">char()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.char](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.char)

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

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.clone](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.clone)

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

> <font size="3">complex_double()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.complex_double](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.complex_double)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id24 -->

</div>

> <font size="3">complex_float()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.complex_float](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.complex_float)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id27 -->

</div>

> <font size="3">copy_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.copy_](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.copy_)

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

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.cpu](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.cpu)

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

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.cuda](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.cuda)

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

> <font size="3">data_ptr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.data_ptr](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.data_ptr)

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

**Native Documentation**: [torch.UntypedStorage.device](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.device)

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

> <font size="3">double()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.double](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.double)

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

</div>

> <font size="3">element_size()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.element_size](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.element_size)

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

> <font size="3">filename()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.filename](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.filename)

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

> <font size="3">fill_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.fill_](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.fill_)

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

> <font size="3">float()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.float](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.float)

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

> <font size="3">float8_e4m3fn()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.float8_e4m3fn](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.float8_e4m3fn)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id60 -->

</div>

> <font size="3">float8_e5m2()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.float8_e5m2](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.float8_e5m2)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id63 -->

</div>

> <font size="3">float8_e4m3fnuz()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.float8_e4m3fnuz](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.float8_e4m3fnuz)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id66 -->

</div>

> <font size="3">float8_e5m2fnuz()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.float8_e5m2fnuz](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.float8_e5m2fnuz)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id69 -->

</div>

> <font size="3">from_buffer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.from_buffer](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.from_buffer)

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

</div>

> <font size="3">from_file()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.from_file](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.from_file)

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

</div>

> <font size="3">get_device()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.get_device](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.get_device)

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

> <font size="3">half()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.half](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.half)

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

</div>

> <font size="3">hpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.hpu](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.hpu)

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id84 -->

</div>

> <font size="3">int()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.int](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.int)

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

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.is_cuda](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_cuda)

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

</div>

> <font size="3">is_hpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.is_hpu](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_hpu)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id93 -->

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.is_pinned](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_pinned)

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

> <font size="3">is_shared()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.is_shared](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_shared)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id99 -->

</div>

> <font size="3">is_sparse()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.is_sparse](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_sparse)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id102 -->

</div>

> <font size="3">is_sparse_csr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.is_sparse_csr](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_sparse_csr)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id105 -->

</div>

> <font size="3">long()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.long](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.long)

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

> <font size="3">mps()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.mps](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.mps)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id111 -->

</div>

> <font size="3">nbytes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.nbytes](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.nbytes)

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

</div>

> <font size="3">new()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.new](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.new)

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

> <font size="3">pin_memory()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.pin_memory](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.pin_memory)

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

> <font size="3">resizable()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.resizable](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.resizable)

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

> <font size="3">resize_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.resize_](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.resize_)

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

> <font size="3">share_memory_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.share_memory_](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.share_memory_)

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

> <font size="3">short()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.short](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.short)

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

> <font size="3">size()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.size](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.size)

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

> <font size="3">tolist()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.tolist](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.tolist)

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

> <font size="3">type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.type](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.type)

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

> <font size="3">untyped()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.UntypedStorage.untyped](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.untyped)

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

</div>

</div>

## Legacy Typed Storage

### <code><i>class</i></code> torch.DoubleStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.DoubleStorage](https://pytorch.org/docs/2.7/storage.html#torch.DoubleStorage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.DoubleStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.DoubleStorage.dtype)

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

</div>

</div>

### <code><i>class</i></code> torch.HalfStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.HalfStorage](https://pytorch.org/docs/2.7/storage.html#torch.HalfStorage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.HalfStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.HalfStorage.dtype)

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

</div>

</div>

### <code><i>class</i></code> torch.LongStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.LongStorage](https://pytorch.org/docs/2.7/storage.html#torch.LongStorage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.LongStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.LongStorage.dtype)

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

</div>

</div>

### <code><i>class</i></code> torch.ShortStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.ShortStorage](https://pytorch.org/docs/2.7/storage.html#torch.ShortStorage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ShortStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.ShortStorage.dtype)

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

</div>

</div>

### <code><i>class</i></code> torch.CharStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.CharStorage](https://pytorch.org/docs/2.7/storage.html#torch.CharStorage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.CharStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.CharStorage.dtype)

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

</div>

</div>

### <code><i>class</i></code> torch.ByteStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.ByteStorage](https://pytorch.org/docs/2.7/storage.html#torch.ByteStorage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ByteStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.ByteStorage.dtype)

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

</div>

</div>

### <code><i>class</i></code> torch.BoolStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.BoolStorage](https://pytorch.org/docs/2.7/storage.html#torch.BoolStorage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.BoolStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.BoolStorage.dtype)

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

</div>

</div>

### <code><i>class</i></code> torch.BFloat16Storage

<div style="margin-left: 2em">

**Native Documentation**: [torch.BFloat16Storage](https://pytorch.org/docs/2.7/storage.html#torch.BFloat16Storage)

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id186 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.BFloat16Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.BFloat16Storage.dtype)

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id189 -->

</div>

</div>

### <code><i>class</i></code> torch.ComplexDoubleStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.ComplexDoubleStorage](https://pytorch.org/docs/2.7/storage.html#torch.ComplexDoubleStorage)

**Support Status**:

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id192 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ComplexDoubleStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.ComplexDoubleStorage.dtype)

**Support Status**:

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id195 -->

</div>

</div>

### <code><i>class</i></code> torch.ComplexFloatStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.ComplexFloatStorage](https://pytorch.org/docs/2.7/storage.html#torch.ComplexFloatStorage)

**Support Status**:

<!-- npu="910b" id196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id198 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ComplexFloatStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.ComplexFloatStorage.dtype)

**Support Status**:

<!-- npu="910b" id199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id201 -->

</div>

</div>

### <code><i>class</i></code> torch.QUInt8Storage

<div style="margin-left: 2em">

**Native Documentation**: [torch.QUInt8Storage](https://pytorch.org/docs/2.7/storage.html#torch.QUInt8Storage)

**Support Status**:

<!-- npu="910b" id202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id204 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.QUInt8Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.QUInt8Storage.dtype)

**Support Status**:

<!-- npu="910b" id205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id207 -->

</div>

**Constraints & Notes**: Supports uint8

</div>

### <code><i>class</i></code> torch.QInt8Storage

<div style="margin-left: 2em">

**Native Documentation**: [torch.QInt8Storage](https://pytorch.org/docs/2.7/storage.html#torch.QInt8Storage)

**Support Status**:

<!-- npu="910b" id208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id210 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.QInt8Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.QInt8Storage.dtype)

**Support Status**:

<!-- npu="910b" id211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id213 -->

**Constraints & Notes**: `self` only supports int8

</div>

</div>

### <code><i>class</i></code> torch.QInt32Storage

<div style="margin-left: 2em">

**Native Documentation**: [torch.QInt32Storage](https://pytorch.org/docs/2.7/storage.html#torch.QInt32Storage)

**Support Status**:

<!-- npu="910b" id214 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id216 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.QInt32Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.QInt32Storage.dtype)

**Support Status**:

<!-- npu="910b" id217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id219 -->

**Constraints & Notes**: `self` only supports int32

</div>

</div>

### <code><i>class</i></code> torch.QUInt4x2Storage

<div style="margin-left: 2em">

**Native Documentation**: [torch.QUInt4x2Storage](https://pytorch.org/docs/2.7/storage.html#torch.QUInt4x2Storage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.QUInt4x2Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.QUInt4x2Storage.dtype)

**Support Status**:

<!-- npu="910b" id223 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id225 -->

**Constraints & Notes**: `self` only supports uint8

</div>

</div>

### <code><i>class</i></code> torch.QUInt2x4Storage

<div style="margin-left: 2em">

**Native Documentation**: [torch.QUInt2x4Storage](https://pytorch.org/docs/2.7/storage.html#torch.QUInt2x4Storage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.QUInt2x4Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.QUInt2x4Storage.dtype)

**Support Status**:

<!-- npu="910b" id229 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id231 -->

**Constraints & Notes**: `self` only supports uint8

</div>

</div>

### <code><i>class</i></code> torch.TypedStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage)

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

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.bfloat16](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.bfloat16)

**Support Status**:

<!-- npu="910b" id235 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id237 -->

</div>

> <font size="3">bool()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.bool](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.bool)

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

</div>

> <font size="3">byte()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.byte](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.byte)

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

</div>

> <font size="3">char()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.char](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.char)

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

</div>

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.clone](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.clone)

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

> <font size="3">complex_double()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.complex_double](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.complex_double)

**Support Status**:

<!-- npu="910b" id250 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id252 -->

</div>

> <font size="3">complex_float()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.complex_float](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.complex_float)

**Support Status**:

<!-- npu="910b" id253 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id255 -->

</div>

> <font size="3">copy_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.copy_](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.copy_)

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

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.cpu](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.cpu)

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

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.cuda](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.cuda)

**Support Status**:

<!-- npu="910b" id262 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id264 -->

</div>

> <font size="3">data_ptr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.data_ptr](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.data_ptr)

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

> <font size="3">device()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.device](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.device)

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

> <font size="3">double()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.double](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.double)

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

</div>

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.dtype)

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

</div>

> <font size="3">element_size()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.element_size](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.element_size)

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

> <font size="3">filename()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.filename](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.filename)

**Support Status**:

<!-- npu="910b" id280 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id280 -->
<!-- npu="A3" id281 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id281 -->
<!-- npu="950" id282 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id282 -->

</div>

> <font size="3">fill_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.fill_](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.fill_)

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

> <font size="3">float()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.float](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.float)

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

> <font size="3">float8_e4m3fn()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.float8_e4m3fn](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.float8_e4m3fn)

**Support Status**:

<!-- npu="910b" id289 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id289 -->
<!-- npu="A3" id290 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id290 -->
<!-- npu="950" id291 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id291 -->

</div>

> <font size="3">float8_e4m3fnuz()</font>
 
<div style="margin-left: 2em">
 
**Native Documentation**: [torch.TypedStorage.float8_e4m3fnuz](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.float8_e4m3fnuz)
 
**Support Status**:

<!-- npu="910b" id373 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id373 -->
<!-- npu="A3" id374 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id374 -->
<!-- npu="950" id375 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id375 -->
 
</div>
 
> <font size="3">float8_e5m2()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.float8_e5m2](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.float8_e5m2)

**Support Status**:

<!-- npu="910b" id292 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id292 -->
<!-- npu="A3" id293 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id293 -->
<!-- npu="950" id294 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id294 -->

</div>

> <font size="3">from_buffer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.from_buffer](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.from_buffer)

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

> <font size="3">from_file()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.from_file](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.from_file)

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

> <font size="3">get_device()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.get_device](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.get_device)

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

> <font size="3">half()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.half](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.half)

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

> <font size="3">hpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.hpu](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.hpu)

**Support Status**:

<!-- npu="910b" id307 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id307 -->
<!-- npu="A3" id308 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id308 -->
<!-- npu="950" id309 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id309 -->

</div>

> <font size="3">int()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.int](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.int)

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

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.is_cuda](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.is_cuda)

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

</div>

> <font size="3">is_hpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.is_hpu](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.is_hpu)

**Support Status**:

<!-- npu="910b" id316 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id316 -->
<!-- npu="A3" id317 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id317 -->
<!-- npu="950" id318 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id318 -->

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.is_pinned](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.is_pinned)

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

</div>

> <font size="3">is_shared()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.is_shared](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.is_shared)

**Support Status**:

<!-- npu="910b" id322 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id322 -->
<!-- npu="A3" id323 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id323 -->
<!-- npu="950" id324 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id324 -->

</div>

> <font size="3">is_sparse()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.is_sparse](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.is_sparse)

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

> <font size="3">long()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.long](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.long)

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

> <font size="3">nbytes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.nbytes](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.nbytes)

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

> <font size="3">pickle_storage_type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.pickle_storage_type](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.pickle_storage_type)

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

> <font size="3">pin_memory()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.pin_memory](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.pin_memory)

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

> <font size="3">resize_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.resize_](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.resize_)

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

</div>

> <font size="3">share_memory_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.share_memory_](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.share_memory_)

**Support Status**:

<!-- npu="910b" id343 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id343 -->
<!-- npu="A3" id344 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id344 -->
<!-- npu="950" id345 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id345 -->

</div>

> <font size="3">short()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.short](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.short)

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

</div>

> <font size="3">size()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.size](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.size)

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

> <font size="3">tolist()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.tolist](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.tolist)

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

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.type](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.type)

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

> <font size="3">untyped()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.TypedStorage.untyped](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.untyped)

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

</div>

### <code><i>class</i></code> torch.FloatStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.FloatStorage](https://pytorch.org/docs/2.7/storage.html#torch.FloatStorage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.FloatStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.FloatStorage.dtype)

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

</div>

### <code><i>class</i></code> torch.IntStorage

<div style="margin-left: 2em">

**Native Documentation**: [torch.IntStorage](https://pytorch.org/docs/2.7/storage.html#torch.IntStorage)

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

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.IntStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.IntStorage.dtype)

**Support Status**:

<!-- npu="910b" id370 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id370 -->
<!-- npu="A3" id371 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id371 -->
<!-- npu="950" id372 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id372 -->

</div>

</div>
