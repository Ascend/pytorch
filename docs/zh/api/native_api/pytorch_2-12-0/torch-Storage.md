# torch.Storage

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/storage.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Special cases](#special-cases)
- [Legacy Typed Storage](#legacy-typed-storage)

</div>

<div style="display:none;">

## &#8203;torch.Storage

</div>

## Special cases

### <code><i>class</i></code> torch.UntypedStorage

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.bfloat16](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.bfloat16)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

</div>

> <font size="3">bool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.bool](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.bool)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id9 -->

</div>

> <font size="3">byte()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.byte](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.byte)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

</div>

> <font size="3">byteswap()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.byteswap](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.byteswap)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id15 -->

</div>

> <font size="3">char()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.char](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.char)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id18 -->

</div>

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.clone](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.clone)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id21 -->

</div>

> <font size="3">complex_double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.complex_double](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.complex_double)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id24 -->

</div>

> <font size="3">complex_float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.complex_float](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.complex_float)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id27 -->

</div>

> <font size="3">copy_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.copy_](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.copy_)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id30 -->

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.cpu](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.cpu)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id33 -->

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.cuda](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.cuda)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id36 -->

</div>

> <font size="3">data_ptr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.data_ptr](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.data_ptr)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id39 -->

</div>

> <font size="3">device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.device](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.device)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id42 -->

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.double](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.double)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id45 -->

</div>

> <font size="3">element_size()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.element_size](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.element_size)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id48 -->

</div>

> <font size="3">filename()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.filename](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.filename)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id51 -->

</div>

> <font size="3">fill_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.fill_](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.fill_)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id54 -->

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.float](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.float)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id57 -->

</div>

> <font size="3">float8_e4m3fn()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.float8_e4m3fn](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.float8_e4m3fn)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id60 -->

</div>

> <font size="3">float8_e5m2()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.float8_e5m2](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.float8_e5m2)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id63 -->

</div>

> <font size="3">float8_e4m3fnuz()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.float8_e4m3fnuz](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.float8_e4m3fnuz)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id66 -->

</div>

> <font size="3">float8_e5m2fnuz()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.float8_e5m2fnuz](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.float8_e5m2fnuz)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id69 -->

</div>

> <font size="3">from_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.from_buffer](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.from_buffer)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id72 -->

</div>

> <font size="3">from_file()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.from_file](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.from_file)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id75 -->

</div>

> <font size="3">get_device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.get_device](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.get_device)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id78 -->

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.half](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.half)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id81 -->

</div>

> <font size="3">hpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.hpu](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.hpu)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id84 -->

</div>

> <font size="3">int()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.int](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.int)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id87 -->

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_cuda](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.is_cuda)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id90 -->

</div>

> <font size="3">is_hpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_hpu](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.is_hpu)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id93 -->

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_pinned](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.is_pinned)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id96 -->

</div>

> <font size="3">is_shared()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_shared](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.is_shared)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id99 -->

</div>

> <font size="3">is_sparse()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_sparse](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.is_sparse)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id102 -->

</div>

> <font size="3">is_sparse_csr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_sparse_csr](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.is_sparse_csr)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id105 -->

</div>

> <font size="3">long()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.long](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.long)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id108 -->

</div>

> <font size="3">mps()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.mps](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.mps)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id111 -->

</div>

> <font size="3">nbytes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.nbytes](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.nbytes)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id114 -->

</div>

> <font size="3">new()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.new](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.new)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id117 -->

</div>

> <font size="3">pin_memory()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.pin_memory](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.pin_memory)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id120 -->

</div>

> <font size="3">resize_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.resize_](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.resize_)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id123 -->

</div>

> <font size="3">share_memory_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.share_memory_](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.share_memory_)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id126 -->

</div>

> <font size="3">short()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.short](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.short)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id129 -->

</div>

> <font size="3">size()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.size](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.size)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id132 -->

</div>

> <font size="3">tolist()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.tolist](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.tolist)

**产品支持情况**：

<!-- npu="910b" id133 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id135 -->

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.type](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.type)

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id138 -->

</div>

> <font size="3">untyped()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.untyped](https://pytorch.org/docs/2.12/storage.html#torch.UntypedStorage.untyped)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id141 -->

</div>

</div>

## Legacy Typed Storage

### <code><i>class</i></code> torch.DoubleStorage

<div style="margin-left: 2em">

**原生文档**：[torch.DoubleStorage](https://pytorch.org/docs/2.12/storage.html#torch.DoubleStorage)

**产品支持情况**：

<!-- npu="910b" id142 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id144 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.DoubleStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.DoubleStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id147 -->

</div>

</div>

### <code><i>class</i></code> torch.HalfStorage

<div style="margin-left: 2em">

**原生文档**：[torch.HalfStorage](https://pytorch.org/docs/2.12/storage.html#torch.HalfStorage)

**产品支持情况**：

<!-- npu="910b" id148 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id150 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.HalfStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.HalfStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id153 -->

</div>

</div>

### <code><i>class</i></code> torch.LongStorage

<div style="margin-left: 2em">

**原生文档**：[torch.LongStorage](https://pytorch.org/docs/2.12/storage.html#torch.LongStorage)

**产品支持情况**：

<!-- npu="910b" id154 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id156 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.LongStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.LongStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id159 -->

</div>

</div>

### <code><i>class</i></code> torch.ShortStorage

<div style="margin-left: 2em">

**原生文档**：[torch.ShortStorage](https://pytorch.org/docs/2.12/storage.html#torch.ShortStorage)

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id162 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ShortStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.ShortStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id165 -->

</div>

</div>

### <code><i>class</i></code> torch.CharStorage

<div style="margin-left: 2em">

**原生文档**：[torch.CharStorage](https://pytorch.org/docs/2.12/storage.html#torch.CharStorage)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id168 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.CharStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.CharStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id171 -->

</div>

</div>

### <code><i>class</i></code> torch.ByteStorage

<div style="margin-left: 2em">

**原生文档**：[torch.ByteStorage](https://pytorch.org/docs/2.12/storage.html#torch.ByteStorage)

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id174 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ByteStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.ByteStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id177 -->

</div>

</div>

### <code><i>class</i></code> torch.BoolStorage

<div style="margin-left: 2em">

**原生文档**：[torch.BoolStorage](https://pytorch.org/docs/2.12/storage.html#torch.BoolStorage)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id180 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.BoolStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.BoolStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id183 -->

</div>

</div>

### <code><i>class</i></code> torch.BFloat16Storage

<div style="margin-left: 2em">

**原生文档**：[torch.BFloat16Storage](https://pytorch.org/docs/2.12/storage.html#torch.BFloat16Storage)

**产品支持情况**：

<!-- npu="910b" id184 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id186 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.BFloat16Storage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.BFloat16Storage.dtype)

**产品支持情况**：

<!-- npu="910b" id187 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id189 -->

</div>

</div>

### <code><i>class</i></code> torch.ComplexDoubleStorage

<div style="margin-left: 2em">

**原生文档**：[torch.ComplexDoubleStorage](https://pytorch.org/docs/2.12/storage.html#torch.ComplexDoubleStorage)

**产品支持情况**：

<!-- npu="910b" id190 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id192 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ComplexDoubleStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.ComplexDoubleStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id193 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id195 -->

</div>

</div>

### <code><i>class</i></code> torch.ComplexFloatStorage

<div style="margin-left: 2em">

**原生文档**：[torch.ComplexFloatStorage](https://pytorch.org/docs/2.12/storage.html#torch.ComplexFloatStorage)

**产品支持情况**：

<!-- npu="910b" id196 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id198 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ComplexFloatStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.ComplexFloatStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id199 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id201 -->

</div>

</div>

### <code><i>class</i></code> torch.QUInt8Storage

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt8Storage](https://pytorch.org/docs/2.12/storage.html#torch.QUInt8Storage)

**产品支持情况**：

<!-- npu="910b" id202 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id204 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt8Storage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.QUInt8Storage.dtype)

**产品支持情况**：

<!-- npu="910b" id205 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id207 -->

</div>

**限制与说明**： 支持uint8

</div>

### <code><i>class</i></code> torch.QInt8Storage

<div style="margin-left: 2em">

**原生文档**：[torch.QInt8Storage](https://pytorch.org/docs/2.12/storage.html#torch.QInt8Storage)

**产品支持情况**：

<!-- npu="910b" id208 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id210 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.QInt8Storage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.QInt8Storage.dtype)

**产品支持情况**：

<!-- npu="910b" id211 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id213 -->

**限制与说明**： `self`仅支持int8

</div>

</div>

### <code><i>class</i></code> torch.QInt32Storage

<div style="margin-left: 2em">

**原生文档**：[torch.QInt32Storage](https://pytorch.org/docs/2.12/storage.html#torch.QInt32Storage)

**产品支持情况**：

<!-- npu="910b" id214 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id216 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.QInt32Storage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.QInt32Storage.dtype)

**产品支持情况**：

<!-- npu="910b" id217 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id219 -->

**限制与说明**： `self`仅支持int32

</div>

</div>

### <code><i>class</i></code> torch.QUInt4x2Storage

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt4x2Storage](https://pytorch.org/docs/2.12/storage.html#torch.QUInt4x2Storage)

**产品支持情况**：

<!-- npu="910b" id220 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id222 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt4x2Storage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.QUInt4x2Storage.dtype)

**产品支持情况**：

<!-- npu="910b" id223 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id225 -->

**限制与说明**： `self`仅支持uint8

</div>

</div>

### <code><i>class</i></code> torch.QUInt2x4Storage

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt2x4Storage](https://pytorch.org/docs/2.12/storage.html#torch.QUInt2x4Storage)

**产品支持情况**：

<!-- npu="910b" id226 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id228 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt2x4Storage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.QUInt2x4Storage.dtype)

**产品支持情况**：

<!-- npu="910b" id229 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id231 -->

**限制与说明**： `self`仅支持uint8

</div>

</div>

### <code><i>class</i></code> torch.TypedStorage

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage)

**产品支持情况**：

<!-- npu="910b" id232 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id234 -->

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.bfloat16](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.bfloat16)

**产品支持情况**：

<!-- npu="910b" id235 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id237 -->

</div>

> <font size="3">bool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.bool](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.bool)

**产品支持情况**：

<!-- npu="910b" id238 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id238 -->
<!-- npu="A3" id239 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id239 -->
<!-- npu="950" id240 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id240 -->

</div>

> <font size="3">byte()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.byte](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.byte)

**产品支持情况**：

<!-- npu="910b" id241 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id243 -->

</div>

> <font size="3">char()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.char](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.char)

**产品支持情况**：

<!-- npu="910b" id244 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id244 -->
<!-- npu="A3" id245 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id245 -->
<!-- npu="950" id246 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id246 -->

</div>

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.clone](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.clone)

**产品支持情况**：

<!-- npu="910b" id247 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id247 -->
<!-- npu="A3" id248 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id248 -->
<!-- npu="950" id249 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id249 -->

</div>

> <font size="3">complex_double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.complex_double](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.complex_double)

**产品支持情况**：

<!-- npu="910b" id250 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id252 -->

</div>

> <font size="3">complex_float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.complex_float](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.complex_float)

**产品支持情况**：

<!-- npu="910b" id253 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id255 -->

</div>

> <font size="3">copy_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.copy_](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.copy_)

**产品支持情况**：

<!-- npu="910b" id256 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id256 -->
<!-- npu="A3" id257 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id257 -->
<!-- npu="950" id258 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id258 -->

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.cpu](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.cpu)

**产品支持情况**：

<!-- npu="910b" id259 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id259 -->
<!-- npu="A3" id260 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id260 -->
<!-- npu="950" id261 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id261 -->

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.cuda](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.cuda)

**产品支持情况**：

<!-- npu="910b" id262 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id264 -->

</div>

> <font size="3">data_ptr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.data_ptr](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.data_ptr)

**产品支持情况**：

<!-- npu="910b" id265 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id265 -->
<!-- npu="A3" id266 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id266 -->
<!-- npu="950" id267 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id267 -->

</div>

> <font size="3">device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.device](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.device)

**产品支持情况**：

<!-- npu="910b" id268 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id268 -->
<!-- npu="A3" id269 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id269 -->
<!-- npu="950" id270 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id270 -->

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.double](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.double)

**产品支持情况**：

<!-- npu="910b" id271 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id271 -->
<!-- npu="A3" id272 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id272 -->
<!-- npu="950" id273 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id273 -->

</div>

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id274 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id274 -->
<!-- npu="A3" id275 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id275 -->
<!-- npu="950" id276 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id276 -->

</div>

> <font size="3">element_size()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.element_size](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.element_size)

**产品支持情况**：

<!-- npu="910b" id277 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id277 -->
<!-- npu="A3" id278 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id278 -->
<!-- npu="950" id279 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id279 -->

</div>

> <font size="3">filename()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.filename](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.filename)

**产品支持情况**：

<!-- npu="910b" id280 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id280 -->
<!-- npu="A3" id281 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id281 -->
<!-- npu="950" id282 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id282 -->

</div>

> <font size="3">fill_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.fill_](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.fill_)

**产品支持情况**：

<!-- npu="910b" id283 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id283 -->
<!-- npu="A3" id284 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id284 -->
<!-- npu="950" id285 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id285 -->

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.float](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.float)

**产品支持情况**：

<!-- npu="910b" id286 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id286 -->
<!-- npu="A3" id287 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id287 -->
<!-- npu="950" id288 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id288 -->

</div>

> <font size="3">float8_e4m3fn()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.float8_e4m3fn](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.float8_e4m3fn)

**产品支持情况**：

<!-- npu="910b" id289 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id289 -->
<!-- npu="A3" id290 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id290 -->
<!-- npu="950" id291 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id291 -->

</div>

> <font size="3">float8_e5m2()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.float8_e5m2](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.float8_e5m2)

**产品支持情况**：

<!-- npu="910b" id292 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id292 -->
<!-- npu="A3" id293 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id293 -->
<!-- npu="950" id294 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id294 -->

</div>

> <font size="3">from_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.from_buffer](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.from_buffer)

**产品支持情况**：

<!-- npu="910b" id295 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id295 -->
<!-- npu="A3" id296 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id296 -->
<!-- npu="950" id297 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id297 -->

</div>

> <font size="3">from_file()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.from_file](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.from_file)

**产品支持情况**：

<!-- npu="910b" id298 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id298 -->
<!-- npu="A3" id299 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id299 -->
<!-- npu="950" id300 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id300 -->

</div>

> <font size="3">get_device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.get_device](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.get_device)

**产品支持情况**：

<!-- npu="910b" id301 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id301 -->
<!-- npu="A3" id302 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id302 -->
<!-- npu="950" id303 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id303 -->

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.half](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.half)

**产品支持情况**：

<!-- npu="910b" id304 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id304 -->
<!-- npu="A3" id305 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id305 -->
<!-- npu="950" id306 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id306 -->

</div>

> <font size="3">hpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.hpu](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.hpu)

**产品支持情况**：

<!-- npu="910b" id307 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id307 -->
<!-- npu="A3" id308 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id308 -->
<!-- npu="950" id309 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id309 -->

</div>

> <font size="3">int()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.int](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.int)

**产品支持情况**：

<!-- npu="910b" id310 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id310 -->
<!-- npu="A3" id311 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id311 -->
<!-- npu="950" id312 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id312 -->

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.is_cuda](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.is_cuda)

**产品支持情况**：

<!-- npu="910b" id313 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id313 -->
<!-- npu="A3" id314 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id314 -->
<!-- npu="950" id315 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id315 -->

</div>

> <font size="3">is_hpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.is_hpu](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.is_hpu)

**产品支持情况**：

<!-- npu="910b" id316 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id316 -->
<!-- npu="A3" id317 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id317 -->
<!-- npu="950" id318 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id318 -->

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.is_pinned](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.is_pinned)

**产品支持情况**：

<!-- npu="910b" id319 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id319 -->
<!-- npu="A3" id320 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id320 -->
<!-- npu="950" id321 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id321 -->

</div>

> <font size="3">is_shared()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.is_shared](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.is_shared)

**产品支持情况**：

<!-- npu="910b" id322 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id322 -->
<!-- npu="A3" id323 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id323 -->
<!-- npu="950" id324 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id324 -->

</div>

> <font size="3">is_sparse()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.is_sparse](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.is_sparse)

**产品支持情况**：

<!-- npu="910b" id325 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id325 -->
<!-- npu="A3" id326 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id326 -->
<!-- npu="950" id327 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id327 -->

</div>

> <font size="3">long()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.long](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.long)

**产品支持情况**：

<!-- npu="910b" id328 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id328 -->
<!-- npu="A3" id329 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id329 -->
<!-- npu="950" id330 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id330 -->

</div>

> <font size="3">nbytes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.nbytes](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.nbytes)

**产品支持情况**：

<!-- npu="910b" id331 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id331 -->
<!-- npu="A3" id332 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id332 -->
<!-- npu="950" id333 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id333 -->

</div>

> <font size="3">pickle_storage_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.pickle_storage_type](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.pickle_storage_type)

**产品支持情况**：

<!-- npu="910b" id334 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id334 -->
<!-- npu="A3" id335 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id335 -->
<!-- npu="950" id336 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id336 -->

</div>

> <font size="3">pin_memory()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.pin_memory](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.pin_memory)

**产品支持情况**：

<!-- npu="910b" id337 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id337 -->
<!-- npu="A3" id338 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id338 -->
<!-- npu="950" id339 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id339 -->

</div>

> <font size="3">resize_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.resize_](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.resize_)

**产品支持情况**：

<!-- npu="910b" id340 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id340 -->
<!-- npu="A3" id341 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id341 -->
<!-- npu="950" id342 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id342 -->

</div>

> <font size="3">share_memory_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.share_memory_](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.share_memory_)

**产品支持情况**：

<!-- npu="910b" id343 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id343 -->
<!-- npu="A3" id344 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id344 -->
<!-- npu="950" id345 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id345 -->

</div>

> <font size="3">short()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.short](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.short)

**产品支持情况**：

<!-- npu="910b" id346 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id346 -->
<!-- npu="A3" id347 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id347 -->
<!-- npu="950" id348 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id348 -->

</div>

> <font size="3">size()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.size](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.size)

**产品支持情况**：

<!-- npu="910b" id349 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id349 -->
<!-- npu="A3" id350 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id350 -->
<!-- npu="950" id351 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id351 -->

</div>

> <font size="3">tolist()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.tolist](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.tolist)

**产品支持情况**：

<!-- npu="910b" id352 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id352 -->
<!-- npu="A3" id353 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id353 -->
<!-- npu="950" id354 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id354 -->

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.type](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.type)

**产品支持情况**：

<!-- npu="910b" id355 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id355 -->
<!-- npu="A3" id356 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id356 -->
<!-- npu="950" id357 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id357 -->

</div>

> <font size="3">untyped()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.untyped](https://pytorch.org/docs/2.12/storage.html#torch.TypedStorage.untyped)

**产品支持情况**：

<!-- npu="910b" id358 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id358 -->
<!-- npu="A3" id359 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id359 -->
<!-- npu="950" id360 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id360 -->

</div>

</div>

### <code><i>class</i></code> torch.FloatStorage

<div style="margin-left: 2em">

**原生文档**：[torch.FloatStorage](https://pytorch.org/docs/2.12/storage.html#torch.FloatStorage)

**产品支持情况**：

<!-- npu="910b" id361 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id361 -->
<!-- npu="A3" id362 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id362 -->
<!-- npu="950" id363 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id363 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.FloatStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.FloatStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id364 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id364 -->
<!-- npu="A3" id365 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id365 -->
<!-- npu="950" id366 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id366 -->

</div>

</div>

### <code><i>class</i></code> torch.IntStorage

<div style="margin-left: 2em">

**原生文档**：[torch.IntStorage](https://pytorch.org/docs/2.12/storage.html#torch.IntStorage)

**产品支持情况**：

<!-- npu="910b" id367 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id367 -->
<!-- npu="A3" id368 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id368 -->
<!-- npu="950" id369 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id369 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.IntStorage.dtype](https://pytorch.org/docs/2.12/storage.html#torch.IntStorage.dtype)

**产品支持情况**：

<!-- npu="910b" id370 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id370 -->
<!-- npu="A3" id371 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id371 -->
<!-- npu="950" id372 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id372 -->

</div>

</div>
