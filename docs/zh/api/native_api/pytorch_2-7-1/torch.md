# torch

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/torch.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

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

### _`class`_ torch.classes.torchvision.GPUDecoder

<div style="margin-left: 2em">

**原生文档**：[torch.classes.torchvision.GPUDecoder](https://pytorch.org/vision/stable/io.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：该接口依赖torchvision GPU decoder扩展，当前Ascend/aarch64环境未提供gpu_decoder.so，无法注册到torch.classes.torchvision

</div>

## Tensors

### torch.is_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.is_tensor](https://pytorch.org/docs/2.7/generated/torch.is_tensor.html)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id3 -->

</div>

### torch.is_storage

<div style="margin-left: 2em">

**原生文档**：[torch.is_storage](https://pytorch.org/docs/2.7/generated/torch.is_storage.html)

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

### torch.is_complex

<div style="margin-left: 2em">

**原生文档**：[torch.is_complex](https://pytorch.org/docs/2.7/generated/torch.is_complex.html)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

**限制与说明**： `input`仅支持complex64，complex128

</div>

### torch.is_conj

<div style="margin-left: 2em">

**原生文档**：[torch.is_conj](https://pytorch.org/docs/2.7/generated/torch.is_conj.html)

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

### torch.is_floating_point

<div style="margin-left: 2em">

**原生文档**：[torch.is_floating_point](https://pytorch.org/docs/2.7/generated/torch.is_floating_point.html)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id15 -->

</div>

### torch.is_nonzero

<div style="margin-left: 2em">

**原生文档**：[torch.is_nonzero](https://pytorch.org/docs/2.7/generated/torch.is_nonzero.html)

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

### torch.set_default_dtype

<div style="margin-left: 2em">

**原生文档**：[torch.set_default_dtype](https://pytorch.org/docs/2.7/generated/torch.set_default_dtype.html)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id21 -->

</div>

### torch.get_default_dtype

<div style="margin-left: 2em">

**原生文档**：[torch.get_default_dtype](https://pytorch.org/docs/2.7/generated/torch.get_default_dtype.html)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id24 -->

</div>

### torch.set_default_device

<div style="margin-left: 2em">

**原生文档**：[torch.set_default_device](https://pytorch.org/docs/2.7/generated/torch.set_default_device.html)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id27 -->

</div>

### torch.set_default_tensor_type

<div style="margin-left: 2em">

**原生文档**：[torch.set_default_tensor_type](https://pytorch.org/docs/2.7/generated/torch.set_default_tensor_type.html)

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

**限制与说明**： 不支持传入`torch.npu.DtypeTensor`类型

</div>

### torch.numel

<div style="margin-left: 2em">

**原生文档**：[torch.numel](https://pytorch.org/docs/2.7/generated/torch.numel.html)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id33 -->

</div>

### torch.set_printoptions

<div style="margin-left: 2em">

**原生文档**：[torch.set_printoptions](https://pytorch.org/docs/2.7/generated/torch.set_printoptions.html)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id36 -->

</div>

### torch.set_flush_denormal

<div style="margin-left: 2em">

**原生文档**：[torch.set_flush_denormal](https://pytorch.org/docs/2.7/generated/torch.set_flush_denormal.html)

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

### torch.tensor

<div style="margin-left: 2em">

**原生文档**：[torch.tensor](https://pytorch.org/docs/2.7/generated/torch.tensor.html)

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

### torch.sparse_coo_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_coo_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_coo_tensor.html)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id45 -->

**限制与说明**：

- `indices`仅支持int32，int64
- `values`仅支持fp16，fp32，int32
- `dtype`参数与`values`的dtype保持一致

</div>

### torch.sparse_csr_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_csr_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_csr_tensor.html)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id48 -->

</div>

### torch.sparse_csc_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_csc_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_csc_tensor.html)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id51 -->

</div>

### torch.sparse_bsr_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_bsr_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_bsr_tensor.html)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id54 -->

</div>

### torch.sparse_bsc_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_bsc_tensor](https://pytorch.org/docs/2.7/generated/torch.sparse_bsc_tensor.html)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id57 -->

</div>

### torch.asarray

<div style="margin-left: 2em">

**原生文档**：[torch.asarray](https://pytorch.org/docs/2.7/generated/torch.asarray.html)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id60 -->

**限制与说明**： `obj`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.as_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.as_tensor](https://pytorch.org/docs/2.7/generated/torch.as_tensor.html)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id63 -->

**限制与说明**： `data`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.as_strided

<div style="margin-left: 2em">

**原生文档**：[torch.as_strided](https://pytorch.org/docs/2.7/generated/torch.as_strided.html)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id66 -->

**限制与说明**：`input`仅支持fp32

</div>

### torch.from_numpy

<div style="margin-left: 2em">

**原生文档**：[torch.from_numpy](https://pytorch.org/docs/2.7/generated/torch.from_numpy.html)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id69 -->

**限制与说明**： `input`仅支持输出fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.from_dlpack

<div style="margin-left: 2em">

**原生文档**：[torch.from_dlpack](https://pytorch.org/docs/2.7/generated/torch.from_dlpack.html)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id72 -->

</div>

### torch.frombuffer

<div style="margin-left: 2em">

**原生文档**：[torch.frombuffer](https://pytorch.org/docs/2.7/generated/torch.frombuffer.html)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id75 -->

**限制与说明**： `dtype`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.zeros

<div style="margin-left: 2em">

**原生文档**：[torch.zeros](https://pytorch.org/docs/2.7/generated/torch.zeros.html)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id78 -->

**限制与说明**： `dtype`仅支持bf16，fp16，fp32，uint8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.zeros_like

<div style="margin-left: 2em">

**原生文档**：[torch.zeros_like](https://pytorch.org/docs/2.7/generated/torch.zeros_like.html)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id81 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.ones

<div style="margin-left: 2em">

**原生文档**：[torch.ones](https://pytorch.org/docs/2.7/generated/torch.ones.html)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id84 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.ones_like

<div style="margin-left: 2em">

**原生文档**：[torch.ones_like](https://pytorch.org/docs/2.7/generated/torch.ones_like.html)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id87 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.arange

<div style="margin-left: 2em">

**原生文档**：[torch.arange](https://pytorch.org/docs/2.7/generated/torch.arange.html)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id90 -->

**限制与说明**： `dtype`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.range

<div style="margin-left: 2em">

**原生文档**：[torch.range](https://pytorch.org/docs/2.7/generated/torch.range.html)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id93 -->

</div>

### torch.linspace

<div style="margin-left: 2em">

**原生文档**：[torch.linspace](https://pytorch.org/docs/2.7/generated/torch.linspace.html)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id96 -->

**限制与说明**：

- `dtype`仅支持bf16，fp16，fp32，fp64，int16，int32，int64，bool，complex64，complex128
- 创建序列大小为`steps`的1维向量

</div>

### torch.eye

<div style="margin-left: 2em">

**原生文档**：[torch.eye](https://pytorch.org/docs/2.7/generated/torch.eye.html)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id99 -->

**限制与说明**： `dtype`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.empty

<div style="margin-left: 2em">

**原生文档**：[torch.empty](https://pytorch.org/docs/2.7/generated/torch.empty.html)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id102 -->

**限制与说明**： `dtype`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.empty_like

<div style="margin-left: 2em">

**原生文档**：[torch.empty_like](https://pytorch.org/docs/2.7/generated/torch.empty_like.html)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id105 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.empty_strided

<div style="margin-left: 2em">

**原生文档**：[torch.empty_strided](https://pytorch.org/docs/2.7/generated/torch.empty_strided.html)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id108 -->

</div>

### torch.full

<div style="margin-left: 2em">

**原生文档**：[torch.full](https://pytorch.org/docs/2.7/generated/torch.full.html)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id111 -->

**限制与说明**： `dtype`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.full_like

<div style="margin-left: 2em">

**原生文档**：[torch.full_like](https://pytorch.org/docs/2.7/generated/torch.full_like.html)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id114 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.quantize_per_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.quantize_per_tensor](https://pytorch.org/docs/2.7/generated/torch.quantize_per_tensor.html)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id117 -->

</div>

### torch.quantize_per_channel

<div style="margin-left: 2em">

**原生文档**：[torch.quantize_per_channel](https://pytorch.org/docs/2.7/generated/torch.quantize_per_channel.html)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id120 -->

</div>

### torch.dequantize

<div style="margin-left: 2em">

**原生文档**：[torch.dequantize](https://pytorch.org/docs/2.7/generated/torch.dequantize.html)

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

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.complex

<div style="margin-left: 2em">

**原生文档**：[torch.complex](https://pytorch.org/docs/2.7/generated/torch.complex.html)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id126 -->

</div>

### torch.polar

<div style="margin-left: 2em">

**原生文档**：[torch.polar](https://pytorch.org/docs/2.7/generated/torch.polar.html)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id129 -->

**限制与说明**：

- `abs`仅支持fp32
- 入参`abs`和`angle`的维度需相等

</div>

### torch.heaviside

<div style="margin-left: 2em">

**原生文档**：[torch.heaviside](https://pytorch.org/docs/2.7/generated/torch.heaviside.html)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id132 -->

</div>

### torch.argwhere

<div style="margin-left: 2em">

**原生文档**：[torch.argwhere](https://pytorch.org/docs/2.7/generated/torch.argwhere.html)

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

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.cat

<div style="margin-left: 2em">

**原生文档**：[torch.cat](https://pytorch.org/docs/2.7/generated/torch.cat.html)

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id138 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.concat

<div style="margin-left: 2em">

**原生文档**：[torch.concat](https://pytorch.org/docs/2.7/generated/torch.concat.html)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id141 -->

**限制与说明**：

- `tensors`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64

<!-- npu="950" id142 -->
- <term>Ascend 950DT系列产品</term>：不支持complex64
<!-- end id142 -->

</div>

### torch.concatenate

<div style="margin-left: 2em">

**原生文档**：[torch.concatenate](https://pytorch.org/docs/2.7/generated/torch.concatenate.html)

**产品支持情况**：

<!-- npu="910b" id143 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="A3" id144 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id144 -->
<!-- npu="950" id145 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id145 -->

**限制与说明**：

- `tensors`仅支持bf16，fp16，fp32，int64，bool，complex64

<!-- npu="950" id146 -->
- <term>Ascend 950DT系列产品</term>：不支持complex64
<!-- end id146 -->

</div>

### torch.conj

<div style="margin-left: 2em">

**原生文档**：[torch.conj](https://pytorch.org/docs/2.7/generated/torch.conj.html)

**产品支持情况**：

<!-- npu="910b" id147 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id147 -->
<!-- npu="A3" id148 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="950" id149 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id149 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.chunk

<div style="margin-left: 2em">

**原生文档**：[torch.chunk](https://pytorch.org/docs/2.7/generated/torch.chunk.html)

**产品支持情况**：

<!-- npu="910b" id150 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id150 -->
<!-- npu="A3" id151 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="950" id152 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id152 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.dsplit

<div style="margin-left: 2em">

**原生文档**：[torch.dsplit](https://pytorch.org/docs/2.7/generated/torch.dsplit.html)

**产品支持情况**：

<!-- npu="910b" id153 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id153 -->
<!-- npu="A3" id154 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="950" id155 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id155 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.dstack

<div style="margin-left: 2em">

**原生文档**：[torch.dstack](https://pytorch.org/docs/2.7/generated/torch.dstack.html)

**产品支持情况**：

<!-- npu="910b" id156 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id156 -->
<!-- npu="A3" id157 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="950" id158 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id158 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.gather

<div style="margin-left: 2em">

**原生文档**：[torch.gather](https://pytorch.org/docs/2.7/generated/torch.gather.html)

**产品支持情况**：

<!-- npu="910b" id159 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id159 -->
<!-- npu="A3" id160 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="950" id161 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id161 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- `index`的维度数需与`input`的维度数一致

<!-- npu="950,A3,910b" id162 -->
- 针对<term>Ascend 950DT系列产品</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2训练系列产品</term>/<term>Atlas A3训练系列产品</term>存在差异，例如：

  ```python
  # index存在重复索引的示例：索引0被多次使用，Ascend 950DT系列产品上结果可能与A2/A3存在精度差异
  x = torch.tensor([[1, 2], [3, 4]], device='npu')
  index = torch.tensor([[0], [0]], device='npu')  # 索引0重复出现
  out = torch.gather(x, 0, index)
  ```
<!-- end id162 -->

</div>

### torch.hsplit

<div style="margin-left: 2em">

**原生文档**：[torch.hsplit](https://pytorch.org/docs/2.7/generated/torch.hsplit.html)

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id165 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.hstack

<div style="margin-left: 2em">

**原生文档**：[torch.hstack](https://pytorch.org/docs/2.7/generated/torch.hstack.html)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id168 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.index_add

<div style="margin-left: 2em">

**原生文档**：[torch.index_add](https://pytorch.org/docs/2.7/generated/torch.index_add.html)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id171 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.index_copy

<div style="margin-left: 2em">

**原生文档**：[torch.index_copy](https://pytorch.org/docs/2.7/generated/torch.index_copy.html)

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

**限制与说明**：`input`仅支持fp32

</div>

### torch.index_reduce

<div style="margin-left: 2em">

**原生文档**：[torch.index_reduce](https://pytorch.org/docs/2.7/generated/torch.index_reduce.html)

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id177 -->

**限制与说明**： 可能回退至CPU执行

</div>

### torch.index_select

<div style="margin-left: 2em">

**原生文档**：[torch.index_select](https://pytorch.org/docs/2.7/generated/torch.index_select.html)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id180 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，uint8，int16，int32，int64，bool，complex64，complex128

<!-- npu="950,A3,910b" id181 -->
- 针对<term>Ascend 950DT系列产品</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2训练系列产品</term>/<term>Atlas A3训练系列产品</term>存在差异，例如：

  ```python
  # index存在重复索引的示例：索引0被多次使用，Ascend 950DT系列产品上结果可能与A2/A3存在精度差异
  x = torch.tensor([[1, 2], [3, 4]], device='npu')
  index = torch.tensor([0, 0], device='npu')  # 索引0重复出现
  out = torch.index_select(x, 0, index)
  ```
<!-- end id181 -->

</div>

### torch.masked_select

<div style="margin-left: 2em">

**原生文档**：[torch.masked_select](https://pytorch.org/docs/2.7/generated/torch.masked_select.html)

**产品支持情况**：

<!-- npu="910b" id182 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="A3" id183 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id183 -->
<!-- npu="950" id184 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id184 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.movedim

<div style="margin-left: 2em">

**原生文档**：[torch.movedim](https://pytorch.org/docs/2.7/generated/torch.movedim.html)

**产品支持情况**：

<!-- npu="910b" id185 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="A3" id186 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id186 -->
<!-- npu="950" id187 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id187 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.moveaxis

<div style="margin-left: 2em">

**原生文档**：[torch.moveaxis](https://pytorch.org/docs/2.7/generated/torch.moveaxis.html)

**产品支持情况**：

<!-- npu="910b" id188 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="A3" id189 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id189 -->
<!-- npu="950" id190 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id190 -->

**限制与说明**：`input`仅支持fp32，int64，complex128

</div>

### torch.narrow

<div style="margin-left: 2em">

**原生文档**：[torch.narrow](https://pytorch.org/docs/2.7/generated/torch.narrow.html)

**产品支持情况**：

<!-- npu="910b" id191 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="A3" id192 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id192 -->
<!-- npu="950" id193 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id193 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.narrow_copy

<div style="margin-left: 2em">

**原生文档**：[torch.narrow_copy](https://pytorch.org/docs/2.7/generated/torch.narrow_copy.html)

**产品支持情况**：

<!-- npu="910b" id194 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="A3" id195 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id195 -->
<!-- npu="950" id196 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id196 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool可能回退至CPU执行

</div>

### torch.nonzero

<div style="margin-left: 2em">

**原生文档**：[torch.nonzero](https://pytorch.org/docs/2.7/generated/torch.nonzero.html)

**产品支持情况**：

<!-- npu="910b" id197 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="A3" id198 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id198 -->
<!-- npu="950" id199 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id199 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.permute

<div style="margin-left: 2em">

**原生文档**：[torch.permute](https://pytorch.org/docs/2.7/generated/torch.permute.html)

**产品支持情况**：

<!-- npu="910b" id200 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="A3" id201 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id201 -->
<!-- npu="950" id202 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id202 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.reshape

<div style="margin-left: 2em">

**原生文档**：[torch.reshape](https://pytorch.org/docs/2.7/generated/torch.reshape.html)

**产品支持情况**：

<!-- npu="910b" id203 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="A3" id204 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id204 -->
<!-- npu="950" id205 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id205 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.row_stack

<div style="margin-left: 2em">

**原生文档**：[torch.row_stack](https://pytorch.org/docs/2.7/generated/torch.row_stack.html)

**产品支持情况**：

<!-- npu="910b" id206 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="A3" id207 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id207 -->
<!-- npu="950" id208 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id208 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.scatter

<div style="margin-left: 2em">

**原生文档**：[torch.scatter](https://pytorch.org/docs/2.7/generated/torch.scatter.html)

**产品支持情况**：

<!-- npu="910b" id209 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="A3" id210 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id210 -->
<!-- npu="950" id211 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id211 -->

**限制与说明**：

- `input`仅支持fp16，fp32，fp64，int8，int16，int32，int64，bool
- 可能回退至CPU执行

<!-- npu="950,A3,910b" id212 -->
- 针对<term>Ascend 950DT系列产品</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2训练系列产品</term>/<term>Atlas A3训练系列产品</term>存在差异，例如：

  ```python
  # index存在重复索引的示例：索引0被多次使用，同一位置被多次写入，Ascend 950DT系列产品上结果可能与A2/A3存在精度差异
  x = torch.tensor([[1, 2], [3, 4]], device='npu')
  src = torch.tensor([[10, 20], [30, 40]], device='npu')
  index = torch.tensor([[0], [0]], device='npu')  # 索引0重复出现
  out = torch.scatter(x, 0, index, src)
  ```
<!-- end id212 -->

</div>

### torch.diagonal_scatter

<div style="margin-left: 2em">

**原生文档**：[torch.diagonal_scatter](https://pytorch.org/docs/2.7/generated/torch.diagonal_scatter.html)

**产品支持情况**：

<!-- npu="910b" id213 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id213 -->
<!-- npu="A3" id214 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id214 -->
<!-- npu="950" id215 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id215 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，int16，int32，int64，bool

</div>

### torch.select_scatter

<div style="margin-left: 2em">

**原生文档**：[torch.select_scatter](https://pytorch.org/docs/2.7/generated/torch.select_scatter.html)

**产品支持情况**：

<!-- npu="910b" id216 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id216 -->
<!-- npu="A3" id217 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="950" id218 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id218 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.slice_scatter

<div style="margin-left: 2em">

**原生文档**：[torch.slice_scatter](https://pytorch.org/docs/2.7/generated/torch.slice_scatter.html)

**产品支持情况**：

<!-- npu="910b" id219 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id219 -->
<!-- npu="A3" id220 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="950" id221 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id221 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.scatter_add

<div style="margin-left: 2em">

**原生文档**：[torch.scatter_add](https://pytorch.org/docs/2.7/generated/torch.scatter_add.html)

**产品支持情况**：

<!-- npu="910b" id222 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id222 -->
<!-- npu="A3" id223 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id223 -->
<!-- npu="950" id224 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id224 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

<!-- npu="950,A3,910b" id225 -->
- 针对<term>Ascend 950DT系列产品</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2训练系列产品</term>/<term>Atlas A3训练系列产品</term>存在差异，例如：

  ```python
  # index存在重复索引的示例：索引0被多次使用，同一位置累加多次，Ascend 950DT系列产品上结果可能与A2/A3存在精度差异
  x = torch.tensor([[1, 2], [3, 4]], device='npu')
  src = torch.tensor([[10, 20], [30, 40]], device='npu')
  index = torch.tensor([[0], [0]], device='npu')  # 索引0重复出现
  out = torch.scatter_add(x, 0, index, src)
  ```
<!-- end id225 -->

</div>

### torch.scatter_reduce

<div style="margin-left: 2em">

**原生文档**：[torch.scatter_reduce](https://pytorch.org/docs/2.7/generated/torch.scatter_reduce.html)

**产品支持情况**：

<!-- npu="910b" id226 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id228 -->

</div>

### torch.split

<div style="margin-left: 2em">

**原生文档**：[torch.split](https://pytorch.org/docs/2.7/generated/torch.split.html)

**产品支持情况**：

<!-- npu="910b" id229 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id231 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.squeeze

<div style="margin-left: 2em">

**原生文档**：[torch.squeeze](https://pytorch.org/docs/2.7/generated/torch.squeeze.html)

**产品支持情况**：

<!-- npu="910b" id232 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id234 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.stack

<div style="margin-left: 2em">

**原生文档**：[torch.stack](https://pytorch.org/docs/2.7/generated/torch.stack.html)

**产品支持情况**：

<!-- npu="910b" id235 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id237 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.swapaxes

<div style="margin-left: 2em">

**原生文档**：[torch.swapaxes](https://pytorch.org/docs/2.7/generated/torch.swapaxes.html)

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

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.swapdims

<div style="margin-left: 2em">

**原生文档**：[torch.swapdims](https://pytorch.org/docs/2.7/generated/torch.swapdims.html)

**产品支持情况**：

<!-- npu="910b" id241 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id243 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.t

<div style="margin-left: 2em">

**原生文档**：[torch.t](https://pytorch.org/docs/2.7/generated/torch.t.html)

**产品支持情况**：

<!-- npu="910b" id244 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id244 -->
<!-- npu="A3" id245 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id245 -->
<!-- npu="950" id246 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id246 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.take

<div style="margin-left: 2em">

**原生文档**：[torch.take](https://pytorch.org/docs/2.7/generated/torch.take.html)

**产品支持情况**：

<!-- npu="910b" id247 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id247 -->
<!-- npu="A3" id248 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id248 -->
<!-- npu="950" id249 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id249 -->

**限制与说明**： `input`仅支持fp16，fp32，int16，int32，int64，bool

</div>

### torch.take_along_dim

<div style="margin-left: 2em">

**原生文档**：[torch.take_along_dim](https://pytorch.org/docs/2.7/generated/torch.take_along_dim.html)

**产品支持情况**：

<!-- npu="910b" id250 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id252 -->

**限制与说明**：`input`仅支持fp32

</div>

### torch.tensor_split

<div style="margin-left: 2em">

**原生文档**：[torch.tensor_split](https://pytorch.org/docs/2.7/generated/torch.tensor_split.html)

**产品支持情况**：

<!-- npu="910b" id253 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id255 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.tile

<div style="margin-left: 2em">

**原生文档**：[torch.tile](https://pytorch.org/docs/2.7/generated/torch.tile.html)

**产品支持情况**：

<!-- npu="910b" id256 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id256 -->
<!-- npu="A3" id257 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id257 -->
<!-- npu="950" id258 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id258 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 若入参`dims`的长度小于`input`.shape的长度，则会在`dims`前自动补全1，使其长度与`input`.shape对齐。补全后的`dims`，需要满足如下限制：
  - 当需要对第一根轴进行重复时，最多允许同时对4个维度进行重复操作（即`dims`中大于1的元素个数 ≤ 4），例如：不支持`torch.tile(input, [2, 3, 4, 5, 6])` ，支持`torch.tile(input, [2, 3, 1, 5, 6])`
  - 当不需要对第一根轴进行重复时，最多允许同时对3个维度进行重复操作（即`dims`中大于1的元素个数 ≤ 3），例如：不支持`torch.tile(input, [1, 3, 4, 5, 6])` ，支持`torch.tile(input, [1, 3, 1, 5, 6])`
  - 若执行反向计算，输入`Tensor`的维度数与入参`dims`中大于1的元素个数之和不得超过8

</div>

### torch.transpose

<div style="margin-left: 2em">

**原生文档**：[torch.transpose](https://pytorch.org/docs/2.7/generated/torch.transpose.html)

**产品支持情况**：

<!-- npu="910b" id259 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id259 -->
<!-- npu="A3" id260 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id260 -->
<!-- npu="950" id261 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id261 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.unsqueeze

<div style="margin-left: 2em">

**原生文档**：[torch.unsqueeze](https://pytorch.org/docs/2.7/generated/torch.unsqueeze.html)

**产品支持情况**：

<!-- npu="910b" id262 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id264 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.vsplit

<div style="margin-left: 2em">

**原生文档**：[torch.vsplit](https://pytorch.org/docs/2.7/generated/torch.vsplit.html)

**产品支持情况**：

<!-- npu="910b" id265 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id265 -->
<!-- npu="A3" id266 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id266 -->
<!-- npu="950" id267 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id267 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.vstack

<div style="margin-left: 2em">

**原生文档**：[torch.vstack](https://pytorch.org/docs/2.7/generated/torch.vstack.html)

**产品支持情况**：

<!-- npu="910b" id268 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id268 -->
<!-- npu="A3" id269 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id269 -->
<!-- npu="950" id270 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id270 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.where

<div style="margin-left: 2em">

**原生文档**：[torch.where](https://pytorch.org/docs/2.7/generated/torch.where.html)

**产品支持情况**：

<!-- npu="910b" id271 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id271 -->
<!-- npu="A3" id272 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id272 -->
<!-- npu="950" id273 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id273 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 不支持8维度的shape

</div>

## Generators

### <code><i>class</i></code> torch.Generator

<div style="margin-left: 2em">

**原生文档**：[torch.Generator](https://pytorch.org/docs/2.7/generated/torch.Generator.html)

**产品支持情况**：

<!-- npu="910b" id274 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id274 -->
<!-- npu="A3" id275 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id275 -->
<!-- npu="950" id276 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id276 -->

> <font size="3">device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.device](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.device)

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

> <font size="3">get_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.get_state](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.get_state)

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

> <font size="3">initial_seed()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.initial_seed](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.initial_seed)

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

> <font size="3">manual_seed()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.manual_seed](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.manual_seed)

**产品支持情况**：

<!-- npu="910b" id286 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id286 -->
<!-- npu="A3" id287 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id287 -->
<!-- npu="950" id288 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id288 -->

</div>

> <font size="3">seed()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.seed](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.seed)

**产品支持情况**：

<!-- npu="910b" id289 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id289 -->
<!-- npu="A3" id290 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id290 -->
<!-- npu="950" id291 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id291 -->

</div>

> <font size="3">set_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.set_state](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.set_state)

**产品支持情况**：

<!-- npu="910b" id292 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id292 -->
<!-- npu="A3" id293 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id293 -->
<!-- npu="950" id294 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id294 -->

</div>

> <font size="3">clone_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.clone_state](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.clone_state)

**产品支持情况**：

<!-- npu="910b" id295 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id295 -->
<!-- npu="A3" id296 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id296 -->
<!-- npu="950" id297 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id297 -->

</div>

> <font size="3">graphsafe_set_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.graphsafe_set_state](https://pytorch.org/docs/2.7/generated/torch.Generator.html#torch.Generator.graphsafe_set_state)

**产品支持情况**：

<!-- npu="910b" id298 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id298 -->
<!-- npu="A3" id299 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id299 -->
<!-- npu="950" id300 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id300 -->

</div>

</div>

## Random sampling

### torch.default_generator

<div style="margin-left: 2em">

**原生文档**：[torch.torch.default_generator](https://pytorch.org/docs/2.7/torch.html#torch.torch.default_generator)

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

### torch.rand

<div style="margin-left: 2em">

**原生文档**：[torch.rand](https://pytorch.org/docs/2.7/generated/torch.rand.html)

**产品支持情况**：

<!-- npu="910b" id304 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id304 -->
<!-- npu="A3" id305 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id305 -->
<!-- npu="950" id306 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id306 -->

</div>

### torch.rand_like

<div style="margin-left: 2em">

**原生文档**：[torch.rand_like](https://pytorch.org/docs/2.7/generated/torch.rand_like.html)

**产品支持情况**：

<!-- npu="910b" id307 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id307 -->
<!-- npu="A3" id308 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id308 -->
<!-- npu="950" id309 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id309 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64
- 遵循PyTorch社区规范，不再支持对bool类型数据进行处理。针对存量bool类型数据可以通过如下方案进行替换：如果需要输出全True，可以采用`torch.bernoulli(input, 1)`。如果需要输出均匀分布的bool类型，则采用`torch.bernoulli(input, 0.5)`

</div>

### torch.randint

<div style="margin-left: 2em">

**原生文档**：[torch.randint](https://pytorch.org/docs/2.7/generated/torch.randint.html)

**产品支持情况**：

<!-- npu="910b" id310 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id310 -->
<!-- npu="A3" id311 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id311 -->
<!-- npu="950" id312 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id312 -->

</div>

### torch.randint_like

<div style="margin-left: 2em">

**原生文档**：[torch.randint_like](https://pytorch.org/docs/2.7/generated/torch.randint_like.html)

**产品支持情况**：

<!-- npu="910b" id313 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id313 -->
<!-- npu="A3" id314 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id314 -->
<!-- npu="950" id315 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id315 -->

**限制与说明**： `input`仅支持fp16，fp32，int64

</div>

### torch.randn

<div style="margin-left: 2em">

**原生文档**：[torch.randn](https://pytorch.org/docs/2.7/generated/torch.randn.html)

**产品支持情况**：

<!-- npu="910b" id316 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id316 -->
<!-- npu="A3" id317 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id317 -->
<!-- npu="950" id318 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id318 -->

</div>

### torch.randn_like

<div style="margin-left: 2em">

**原生文档**：[torch.randn_like](https://pytorch.org/docs/2.7/generated/torch.randn_like.html)

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

**限制与说明**： `input`仅支持fp32

</div>

### torch.randperm

<div style="margin-left: 2em">

**原生文档**：[torch.randperm](https://pytorch.org/docs/2.7/generated/torch.randperm.html)

**产品支持情况**：

<!-- npu="910b" id322 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id322 -->
<!-- npu="A3" id323 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id323 -->
<!-- npu="950" id324 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id324 -->

</div>

### torch.seed

<div style="margin-left: 2em">

**原生文档**：[torch.seed](https://pytorch.org/docs/2.7/generated/torch.seed.html)

**产品支持情况**：

<!-- npu="910b" id325 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id325 -->
<!-- npu="A3" id326 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id326 -->
<!-- npu="950" id327 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id327 -->

</div>

### torch.manual_seed

<div style="margin-left: 2em">

**原生文档**：[torch.manual_seed](https://pytorch.org/docs/2.7/generated/torch.manual_seed.html)

**产品支持情况**：

<!-- npu="910b" id328 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id328 -->
<!-- npu="A3" id329 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id329 -->
<!-- npu="950" id330 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id330 -->

</div>

### torch.initial_seed

<div style="margin-left: 2em">

**原生文档**：[torch.initial_seed](https://pytorch.org/docs/2.7/generated/torch.initial_seed.html)

**产品支持情况**：

<!-- npu="910b" id331 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id331 -->
<!-- npu="A3" id332 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id332 -->
<!-- npu="950" id333 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id333 -->

</div>

### torch.get_rng_state

<div style="margin-left: 2em">

**原生文档**：[torch.get_rng_state](https://pytorch.org/docs/2.7/generated/torch.get_rng_state.html)

**产品支持情况**：

<!-- npu="910b" id334 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id334 -->
<!-- npu="A3" id335 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id335 -->
<!-- npu="950" id336 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id336 -->

</div>

### torch.set_rng_state

<div style="margin-left: 2em">

**原生文档**：[torch.set_rng_state](https://pytorch.org/docs/2.7/generated/torch.set_rng_state.html)

**产品支持情况**：

<!-- npu="910b" id337 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id337 -->
<!-- npu="A3" id338 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id338 -->
<!-- npu="950" id339 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id339 -->

</div>

### torch.bernoulli

<div style="margin-left: 2em">

**原生文档**：[torch.bernoulli](https://pytorch.org/docs/2.7/generated/torch.bernoulli.html)

**产品支持情况**：

<!-- npu="910b" id340 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id340 -->
<!-- npu="A3" id341 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id341 -->
<!-- npu="950" id342 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id342 -->

**限制与说明**： `input`仅支持fp16，fp32，fp64

</div>

### torch.multinomial

<div style="margin-left: 2em">

**原生文档**：[torch.multinomial](https://pytorch.org/docs/2.7/generated/torch.multinomial.html)

**产品支持情况**：

<!-- npu="910b" id343 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id343 -->
<!-- npu="A3" id344 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id344 -->
<!-- npu="950" id345 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id345 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### torch.normal

<div style="margin-left: 2em">

**原生文档**：[torch.normal](https://pytorch.org/docs/2.7/generated/torch.normal.html)

**产品支持情况**：

<!-- npu="910b" id346 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id346 -->
<!-- npu="A3" id347 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id347 -->
<!-- npu="950" id348 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id348 -->

**限制与说明**： `mean`、`std`仅支持fp16，fp32，fp64

</div>

### torch.poisson

<div style="margin-left: 2em">

**原生文档**：[torch.poisson](https://pytorch.org/docs/2.7/generated/torch.poisson.html)

**产品支持情况**：

<!-- npu="910b" id349 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id349 -->
<!-- npu="A3" id350 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id350 -->
<!-- npu="950" id351 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id351 -->

</div>

### <code><i>class</i></code> torch.quasirandom.SobolEngine

<div style="margin-left: 2em">

> <font size="3">draw()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.quasirandom.SobolEngine.draw](https://pytorch.org/docs/2.7/generated/torch.quasirandom.SobolEngine.html#torch.quasirandom.SobolEngine.draw)

**产品支持情况**：

<!-- npu="910b" id352 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id352 -->
<!-- npu="A3" id353 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id353 -->
<!-- npu="950" id354 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id354 -->

**限制与说明**：`self`仅支持fp32，fp64

</div>

</div>

## Serialization

### torch.save

<div style="margin-left: 2em">

**原生文档**：[torch.save](https://pytorch.org/docs/2.7/generated/torch.save.html)

**产品支持情况**：

<!-- npu="910b" id355 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id355 -->
<!-- npu="A3" id356 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id356 -->
<!-- npu="950" id357 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id357 -->

</div>

### torch.load

<div style="margin-left: 2em">

**原生文档**：[torch.load](https://pytorch.org/docs/2.7/generated/torch.load.html)

**产品支持情况**：

<!-- npu="910b" id358 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id358 -->
<!-- npu="A3" id359 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id359 -->
<!-- npu="950" id360 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id360 -->

</div>

### torch.serialization.get_safe_globals

<div style="margin-left: 2em">

**原生文档**：[torch.serialization.get_safe_globals](https://pytorch.org/docs/2.7/notes/serialization.html#torch.serialization.get_safe_globals)

**产品支持情况**：

<!-- npu="910b" id361 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id361 -->
<!-- npu="A3" id362 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id362 -->
<!-- npu="950" id363 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id363 -->

</div>

### <code><i>class</i></code> torch.serialization.safe_globals

<div style="margin-left: 2em">

**原生文档**：[torch.serialization.safe_globals](https://pytorch.org/docs/2.7/notes/serialization.html#torch.serialization.safe_globals)

**支持情况**：

<!-- npu="910b" id364 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id364 -->
<!-- npu="A3" id365 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id365 -->
<!-- npu="950" id366 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id366 -->

</div>

### torch.serialization.add_safe_globals

<div style="margin-left: 2em">

**原生文档**：[torch.serialization.add_safe_globals](https://pytorch.org/docs/2.7/notes/serialization.html#torch.serialization.add_safe_globals)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

## Parallelism

### torch.get_num_threads

<div style="margin-left: 2em">

**原生文档**：[torch.get_num_threads](https://pytorch.org/docs/2.7/generated/torch.get_num_threads.html)

**产品支持情况**：

<!-- npu="910b" id367 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id367 -->
<!-- npu="A3" id368 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id368 -->
<!-- npu="950" id369 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id369 -->

</div>

### torch.set_num_threads

<div style="margin-left: 2em">

**原生文档**：[torch.set_num_threads](https://pytorch.org/docs/2.7/generated/torch.set_num_threads.html)

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

### torch.get_num_interop_threads

<div style="margin-left: 2em">

**原生文档**：[torch.get_num_interop_threads](https://pytorch.org/docs/2.7/generated/torch.get_num_interop_threads.html)

**产品支持情况**：

<!-- npu="910b" id373 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id373 -->
<!-- npu="A3" id374 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id374 -->
<!-- npu="950" id375 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id375 -->

</div>

### torch.set_num_interop_threads

<div style="margin-left: 2em">

**原生文档**：[torch.set_num_interop_threads](https://pytorch.org/docs/2.7/generated/torch.set_num_interop_threads.html)

**产品支持情况**：

<!-- npu="910b" id376 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id376 -->
<!-- npu="A3" id377 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id377 -->
<!-- npu="950" id378 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id378 -->

</div>

## Locally disabling gradient computation

### torch.no_grad

<div style="margin-left: 2em">

**原生文档**：[torch.no_grad](https://pytorch.org/docs/2.7/generated/torch.no_grad.html)

**产品支持情况**：

<!-- npu="910b" id379 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id379 -->
<!-- npu="A3" id380 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id380 -->
<!-- npu="950" id381 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id381 -->

</div>

### torch.enable_grad

<div style="margin-left: 2em">

**原生文档**：[torch.enable_grad](https://pytorch.org/docs/2.7/generated/torch.enable_grad.html)

**产品支持情况**：

<!-- npu="910b" id382 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id382 -->
<!-- npu="A3" id383 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id383 -->
<!-- npu="950" id384 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id384 -->

</div>

### torch.autograd.grad_mode.set_grad_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.grad_mode.set_grad_enabled](https://pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.set_grad_enabled.html)

**产品支持情况**：

<!-- npu="910b" id385 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id385 -->
<!-- npu="A3" id386 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id386 -->
<!-- npu="950" id387 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id387 -->

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.grad_mode.set_grad_enabled.clone](https://pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.set_grad_enabled.html#torch.autograd.grad_mode.set_grad_enabled.clone)

**产品支持情况**：

<!-- npu="910b" id388 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id388 -->
<!-- npu="A3" id389 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id389 -->
<!-- npu="950" id390 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id390 -->

</div>

</div>

### torch.is_grad_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.is_grad_enabled](https://pytorch.org/docs/2.7/generated/torch.is_grad_enabled.html)

**产品支持情况**：

<!-- npu="910b" id391 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id391 -->
<!-- npu="A3" id392 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id392 -->
<!-- npu="950" id393 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id393 -->

</div>

### torch.autograd.grad_mode.inference_mode

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.grad_mode.inference_mode](https://pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.inference_mode.html)

**产品支持情况**：

<!-- npu="910b" id394 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id394 -->
<!-- npu="A3" id395 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id395 -->
<!-- npu="950" id396 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id396 -->

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.grad_mode.inference_mode.clone](https://pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.inference_mode.html#torch.autograd.grad_mode.inference_mode.clone)

**产品支持情况**：

<!-- npu="910b" id397 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id397 -->
<!-- npu="A3" id398 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id398 -->
<!-- npu="950" id399 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id399 -->

</div>

</div>

### torch.is_inference_mode_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.is_inference_mode_enabled](https://pytorch.org/docs/2.7/generated/torch.is_inference_mode_enabled.html)

**产品支持情况**：

<!-- npu="910b" id400 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id400 -->
<!-- npu="A3" id401 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id401 -->
<!-- npu="950" id402 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id402 -->

</div>

## Math operations

### torch.abs

<div style="margin-left: 2em">

**原生文档**：[torch.abs](https://pytorch.org/docs/2.7/generated/torch.abs.html)

**产品支持情况**：

<!-- npu="910b" id403 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id403 -->
<!-- npu="A3" id404 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id404 -->
<!-- npu="950" id405 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id405 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.angle

<div style="margin-left: 2em">

**原生文档**：[torch.angle](https://pytorch.org/docs/2.7/generated/torch.angle.html)

**产品支持情况**：

<!-- npu="910b" id406 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id406 -->
<!-- npu="A3" id407 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id407 -->
<!-- npu="950" id408 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id408 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.real

<div style="margin-left: 2em">

**原生文档**：[torch.real](https://pytorch.org/docs/2.7/generated/torch.real.html)

**产品支持情况**：

<!-- npu="910b" id409 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id409 -->
<!-- npu="A3" id410 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id410 -->
<!-- npu="950" id411 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id411 -->

**限制与说明**： `input`仅支持fp16，fp32，complex64，complex128

</div>

### torch.absolute

<div style="margin-left: 2em">

**原生文档**：[torch.absolute](https://pytorch.org/docs/2.7/generated/torch.absolute.html)

**产品支持情况**：

<!-- npu="910b" id412 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id412 -->
<!-- npu="A3" id413 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id413 -->
<!-- npu="950" id414 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id414 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.acos

<div style="margin-left: 2em">

**原生文档**：[torch.acos](https://pytorch.org/docs/2.7/generated/torch.acos.html)

**产品支持情况**：

<!-- npu="910b" id415 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id415 -->
<!-- npu="A3" id416 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id416 -->
<!-- npu="950" id417 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id417 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.arccos

<div style="margin-left: 2em">

**原生文档**：[torch.arccos](https://pytorch.org/docs/2.7/generated/torch.arccos.html)

**产品支持情况**：

<!-- npu="910b" id418 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id418 -->
<!-- npu="A3" id419 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id419 -->
<!-- npu="950" id420 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id420 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.acosh

<div style="margin-left: 2em">

**原生文档**：[torch.acosh](https://pytorch.org/docs/2.7/generated/torch.acosh.html)

**产品支持情况**：

<!-- npu="910b" id421 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id421 -->
<!-- npu="A3" id422 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id422 -->
<!-- npu="950" id423 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id423 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 可能回退至CPU执行

</div>

### torch.arccosh

<div style="margin-left: 2em">

**原生文档**：[torch.arccosh](https://pytorch.org/docs/2.7/generated/torch.arccosh.html)

**产品支持情况**：

<!-- npu="910b" id424 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id424 -->
<!-- npu="A3" id425 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id425 -->
<!-- npu="950" id426 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id426 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.add

<div style="margin-left: 2em">

**原生文档**：[torch.add](https://pytorch.org/docs/2.7/generated/torch.add.html)

**产品支持情况**：

<!-- npu="910b" id427 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id427 -->
<!-- npu="A3" id428 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id428 -->
<!-- npu="950" id429 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id429 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.addcdiv

<div style="margin-left: 2em">

**原生文档**：[torch.addcdiv](https://pytorch.org/docs/2.7/generated/torch.addcdiv.html)

**产品支持情况**：

<!-- npu="910b" id430 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id430 -->
<!-- npu="A3" id431 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id431 -->
<!-- npu="950" id432 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id432 -->

**限制与说明**：

- `input`仅支持fp16，fp32

</div>

### torch.addcmul

<div style="margin-left: 2em">

**原生文档**：[torch.addcmul](https://pytorch.org/docs/2.7/generated/torch.addcmul.html)

**产品支持情况**：

<!-- npu="910b" id433 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id433 -->
<!-- npu="A3" id434 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id434 -->
<!-- npu="950" id435 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id435 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64
- 在fp64，uint8，int8，int64类型下不支持三个`tensor`同时广播

</div>

### torch.asin

<div style="margin-left: 2em">

**原生文档**：[torch.asin](https://pytorch.org/docs/2.7/generated/torch.asin.html)

**产品支持情况**：

<!-- npu="910b" id436 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id436 -->
<!-- npu="A3" id437 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id437 -->
<!-- npu="950" id438 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id438 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.arcsin

<div style="margin-left: 2em">

**原生文档**：[torch.arcsin](https://pytorch.org/docs/2.7/generated/torch.arcsin.html)

**产品支持情况**：

<!-- npu="910b" id439 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id439 -->
<!-- npu="A3" id440 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id440 -->
<!-- npu="950" id441 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id441 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.asinh

<div style="margin-left: 2em">

**原生文档**：[torch.asinh](https://pytorch.org/docs/2.7/generated/torch.asinh.html)

**产品支持情况**：

<!-- npu="910b" id442 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id442 -->
<!-- npu="A3" id443 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id443 -->
<!-- npu="950" id444 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id444 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.arcsinh

<div style="margin-left: 2em">

**原生文档**：[torch.arcsinh](https://pytorch.org/docs/2.7/generated/torch.arcsinh.html)

**产品支持情况**：

<!-- npu="910b" id445 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id445 -->
<!-- npu="A3" id446 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id446 -->
<!-- npu="950" id447 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id447 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.atan

<div style="margin-left: 2em">

**原生文档**：[torch.atan](https://pytorch.org/docs/2.7/generated/torch.atan.html)

**产品支持情况**：

<!-- npu="910b" id448 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id448 -->
<!-- npu="A3" id449 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id449 -->
<!-- npu="950" id450 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id450 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.arctan

<div style="margin-left: 2em">

**原生文档**：[torch.arctan](https://pytorch.org/docs/2.7/generated/torch.arctan.html)

**产品支持情况**：

<!-- npu="910b" id451 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id451 -->
<!-- npu="A3" id452 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id452 -->
<!-- npu="950" id453 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id453 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.atanh

<div style="margin-left: 2em">

**原生文档**：[torch.atanh](https://pytorch.org/docs/2.7/generated/torch.atanh.html)

**产品支持情况**：

<!-- npu="910b" id454 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id454 -->
<!-- npu="A3" id455 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id455 -->
<!-- npu="950" id456 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id456 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.arctanh

<div style="margin-left: 2em">

**原生文档**：[torch.arctanh](https://pytorch.org/docs/2.7/generated/torch.arctanh.html)

**产品支持情况**：

<!-- npu="910b" id457 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id457 -->
<!-- npu="A3" id458 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id458 -->
<!-- npu="950" id459 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id459 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.atan2

<div style="margin-left: 2em">

**原生文档**：[torch.atan2](https://pytorch.org/docs/2.7/generated/torch.atan2.html)

**产品支持情况**：

<!-- npu="910b" id460 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id460 -->
<!-- npu="A3" id461 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id461 -->
<!-- npu="950" id462 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id462 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.arctan2

<div style="margin-left: 2em">

**原生文档**：[torch.arctan2](https://pytorch.org/docs/2.7/generated/torch.arctan2.html)

**产品支持情况**：

<!-- npu="910b" id463 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id463 -->
<!-- npu="A3" id464 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id464 -->
<!-- npu="950" id465 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id465 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.bitwise_not

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_not](https://pytorch.org/docs/2.7/generated/torch.bitwise_not.html)

**产品支持情况**：

<!-- npu="910b" id466 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id466 -->
<!-- npu="A3" id467 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id467 -->
<!-- npu="950" id468 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id468 -->

**限制与说明**： `input`仅支持uint8，int8，int16，int32，int64，bool

</div>

### torch.bitwise_and

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_and](https://pytorch.org/docs/2.7/generated/torch.bitwise_and.html)

**产品支持情况**：

<!-- npu="910b" id469 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id469 -->
<!-- npu="A3" id470 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id470 -->
<!-- npu="950" id471 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id471 -->

**限制与说明**： `input`仅支持uint8，int8，int16，int32，int64，bool

</div>

### torch.bitwise_or

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_or](https://pytorch.org/docs/2.7/generated/torch.bitwise_or.html)

**产品支持情况**：

<!-- npu="910b" id472 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id472 -->
<!-- npu="A3" id473 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id473 -->
<!-- npu="950" id474 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id474 -->

**限制与说明**： `input`仅支持uint8，int8，int16，int32，int64，bool

</div>

### torch.bitwise_xor

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_xor](https://pytorch.org/docs/2.7/generated/torch.bitwise_xor.html)

**产品支持情况**：

<!-- npu="910b" id475 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id475 -->
<!-- npu="A3" id476 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id476 -->
<!-- npu="950" id477 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id477 -->

**限制与说明**： `input`仅支持uint8，int8，int16，int32，int64，bool

</div>

### torch.bitwise_left_shift

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_left_shift](https://pytorch.org/docs/2.7/generated/torch.bitwise_left_shift.html)

**产品支持情况**：

<!-- npu="910b" id478 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id478 -->
<!-- npu="A3" id479 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id479 -->
<!-- npu="950" id480 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id480 -->

**限制与说明**：

- `input`仅支持uint8，int8，int16，int32，int64

- 只能保证shiftBits的数值小于self数据类型位宽时，精度无误差

</div>

### torch.bitwise_right_shift

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_right_shift](https://pytorch.org/docs/2.7/generated/torch.bitwise_right_shift.html)

**产品支持情况**：

<!-- npu="910b" id481 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id481 -->
<!-- npu="A3" id482 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id482 -->
<!-- npu="950" id483 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id483 -->

**限制与说明**：

- `input`仅支持uint8，int8，int16，int32，int64

- 只能保证shiftBits的数值小于self数据类型位宽时，精度无误差

</div>

### torch.ceil

<div style="margin-left: 2em">

**原生文档**：[torch.ceil](https://pytorch.org/docs/2.7/generated/torch.ceil.html)

**产品支持情况**：

<!-- npu="910b" id484 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id484 -->
<!-- npu="A3" id485 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id485 -->
<!-- npu="950" id486 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id486 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### torch.clamp

<div style="margin-left: 2em">

**原生文档**：[torch.clamp](https://pytorch.org/docs/2.7/generated/torch.clamp.html)

**产品支持情况**：

<!-- npu="910b" id487 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id487 -->
<!-- npu="A3" id488 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id488 -->
<!-- npu="950" id489 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id489 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.clip

<div style="margin-left: 2em">

**原生文档**：[torch.clip](https://pytorch.org/docs/2.7/generated/torch.clip.html)

**产品支持情况**：

<!-- npu="910b" id490 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id490 -->
<!-- npu="A3" id491 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id491 -->
<!-- npu="950" id492 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id492 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.copysign

<div style="margin-left: 2em">

**原生文档**：[torch.copysign](https://pytorch.org/docs/2.7/generated/torch.copysign.html)

**产品支持情况**：

<!-- npu="910b" id493 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id493 -->
<!-- npu="A3" id494 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id494 -->
<!-- npu="950" id495 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id495 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool
- 可能回退至CPU执行

</div>

### torch.cos

<div style="margin-left: 2em">

**原生文档**：[torch.cos](https://pytorch.org/docs/2.7/generated/torch.cos.html)

**产品支持情况**：

<!-- npu="910b" id496 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id496 -->
<!-- npu="A3" id497 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id497 -->
<!-- npu="950" id498 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id498 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.cosh

<div style="margin-left: 2em">

**原生文档**：[torch.cosh](https://pytorch.org/docs/2.7/generated/torch.cosh.html)

**产品支持情况**：

<!-- npu="910b" id499 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id499 -->
<!-- npu="A3" id500 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id500 -->
<!-- npu="950" id501 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id501 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.deg2rad

<div style="margin-left: 2em">

**原生文档**：[torch.deg2rad](https://pytorch.org/docs/2.7/generated/torch.deg2rad.html)

**产品支持情况**：

<!-- npu="910b" id502 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id502 -->
<!-- npu="A3" id503 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id503 -->
<!-- npu="950" id504 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id504 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.div

<div style="margin-left: 2em">

**原生文档**：[torch.div](https://pytorch.org/docs/2.7/generated/torch.div.html)

**产品支持情况**：

<!-- npu="910b" id505 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id505 -->
<!-- npu="A3" id506 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id506 -->
<!-- npu="950" id507 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id507 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.divide

<div style="margin-left: 2em">

**原生文档**：[torch.divide](https://pytorch.org/docs/2.7/generated/torch.divide.html)

**产品支持情况**：

<!-- npu="910b" id508 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id508 -->
<!-- npu="A3" id509 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id509 -->
<!-- npu="950" id510 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id510 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.erf

<div style="margin-left: 2em">

**原生文档**：[torch.erf](https://pytorch.org/docs/2.7/generated/torch.erf.html)

**产品支持情况**：

<!-- npu="910b" id511 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id511 -->
<!-- npu="A3" id512 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id512 -->
<!-- npu="950" id513 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id513 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，int64，bool

</div>

### torch.erfc

<div style="margin-left: 2em">

**原生文档**：[torch.erfc](https://pytorch.org/docs/2.7/generated/torch.erfc.html)

**产品支持情况**：

<!-- npu="910b" id514 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id514 -->
<!-- npu="A3" id515 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id515 -->
<!-- npu="950" id516 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id516 -->

**限制与说明**： `input`仅支持fp16，fp32，int64，bool

</div>

### torch.erfinv

<div style="margin-left: 2em">

**原生文档**：[torch.erfinv](https://pytorch.org/docs/2.7/generated/torch.erfinv.html)

**产品支持情况**：

<!-- npu="910b" id517 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id517 -->
<!-- npu="A3" id518 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id518 -->
<!-- npu="950" id519 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id519 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.exp

<div style="margin-left: 2em">

**原生文档**：[torch.exp](https://pytorch.org/docs/2.7/generated/torch.exp.html)

**产品支持情况**：

<!-- npu="910b" id520 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id520 -->
<!-- npu="A3" id521 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id521 -->
<!-- npu="950" id522 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id522 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，int64，bool

</div>

### torch.exp2

<div style="margin-left: 2em">

**原生文档**：[torch.exp2](https://pytorch.org/docs/2.7/generated/torch.exp2.html)

**产品支持情况**：

<!-- npu="910b" id523 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id523 -->
<!-- npu="A3" id524 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id524 -->
<!-- npu="950" id525 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id525 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.expm1

<div style="margin-left: 2em">

**原生文档**：[torch.expm1](https://pytorch.org/docs/2.7/generated/torch.expm1.html)

**产品支持情况**：

<!-- npu="910b" id526 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id526 -->
<!-- npu="A3" id527 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id527 -->
<!-- npu="950" id528 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id528 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，int64，bool

</div>

### torch.fix

<div style="margin-left: 2em">

**原生文档**：[torch.fix](https://pytorch.org/docs/2.7/generated/torch.fix.html)

**产品支持情况**：

<!-- npu="910b" id529 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id529 -->
<!-- npu="A3" id530 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id530 -->
<!-- npu="950" id531 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id531 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.float_power

<div style="margin-left: 2em">

**原生文档**：[torch.float_power](https://pytorch.org/docs/2.7/generated/torch.float_power.html)

**产品支持情况**：

<!-- npu="910b" id532 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id532 -->
<!-- npu="A3" id533 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id533 -->
<!-- npu="950" id534 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id534 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex128

</div>

### torch.floor

<div style="margin-left: 2em">

**原生文档**：[torch.floor](https://pytorch.org/docs/2.7/generated/torch.floor.html)

**产品支持情况**：

<!-- npu="910b" id535 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id535 -->
<!-- npu="A3" id536 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id536 -->
<!-- npu="950" id537 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id537 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.floor_divide

<div style="margin-left: 2em">

**原生文档**：[torch.floor_divide](https://pytorch.org/docs/2.7/generated/torch.floor_divide.html)

**产品支持情况**：

<!-- npu="910b" id538 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id538 -->
<!-- npu="A3" id539 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id539 -->
<!-- npu="950" id540 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id540 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.fmod

<div style="margin-left: 2em">

**原生文档**：[torch.fmod](https://pytorch.org/docs/2.7/generated/torch.fmod.html)

**产品支持情况**：

<!-- npu="910b" id541 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id541 -->
<!-- npu="A3" id542 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id542 -->
<!-- npu="950" id543 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id543 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.gradient

<div style="margin-left: 2em">

**原生文档**：[torch.gradient](https://pytorch.org/docs/2.7/generated/torch.gradient.html)

**产品支持情况**：

<!-- npu="910b" id544 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id544 -->
<!-- npu="A3" id545 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id545 -->
<!-- npu="950" id546 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id546 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，int8，int16，int32，int64

</div>

### torch.ldexp

<div style="margin-left: 2em">

**原生文档**：[torch.ldexp](https://pytorch.org/docs/2.7/generated/torch.ldexp.html)

**产品支持情况**：

<!-- npu="910b" id547 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id547 -->
<!-- npu="A3" id548 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id548 -->
<!-- npu="950" id549 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id549 -->

**限制与说明**： `input`仅支持fp16，fp64，complex64

</div>

### torch.lerp

<div style="margin-left: 2em">

**原生文档**：[torch.lerp](https://pytorch.org/docs/2.7/generated/torch.lerp.html)

**产品支持情况**：

<!-- npu="910b" id550 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id550 -->
<!-- npu="A3" id551 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id551 -->
<!-- npu="950" id552 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id552 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### torch.log

<div style="margin-left: 2em">

**原生文档**：[torch.log](https://pytorch.org/docs/2.7/generated/torch.log.html)

**产品支持情况**：

<!-- npu="910b" id553 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id553 -->
<!-- npu="A3" id554 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id554 -->
<!-- npu="950" id555 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id555 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.log10

<div style="margin-left: 2em">

**原生文档**：[torch.log10](https://pytorch.org/docs/2.7/generated/torch.log10.html)

**产品支持情况**：

<!-- npu="910b" id556 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id556 -->
<!-- npu="A3" id557 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id557 -->
<!-- npu="950" id558 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id558 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 当输入`input`为uint8，int8，int16，int32，int64，bool时，输出`out`必须为fp32
- 其余支持数据类型输出`out`和输入`input`保持一致

</div>

### torch.log1p

<div style="margin-left: 2em">

**原生文档**：[torch.log1p](https://pytorch.org/docs/2.7/generated/torch.log1p.html)

**产品支持情况**：

<!-- npu="910b" id559 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id559 -->
<!-- npu="A3" id560 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id560 -->
<!-- npu="950" id561 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id561 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.log2

<div style="margin-left: 2em">

**原生文档**：[torch.log2](https://pytorch.org/docs/2.7/generated/torch.log2.html)

**产品支持情况**：

<!-- npu="910b" id562 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id562 -->
<!-- npu="A3" id563 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id563 -->
<!-- npu="950" id564 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id564 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.logaddexp

<div style="margin-left: 2em">

**原生文档**：[torch.logaddexp](https://pytorch.org/docs/2.7/generated/torch.logaddexp.html)

**产品支持情况**：

<!-- npu="910b" id565 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id565 -->
<!-- npu="A3" id566 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id566 -->
<!-- npu="950" id567 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id567 -->

**限制与说明**： 不支持double数据类型

</div>

### torch.logaddexp2

<div style="margin-left: 2em">

**原生文档**：[torch.logaddexp2](https://pytorch.org/docs/2.7/generated/torch.logaddexp2.html)

**产品支持情况**：

<!-- npu="910b" id568 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id568 -->
<!-- npu="A3" id569 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id569 -->
<!-- npu="950" id570 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id570 -->

**限制与说明**： 不支持double数据类型

</div>

### torch.logical_and

<div style="margin-left: 2em">

**原生文档**：[torch.logical_and](https://pytorch.org/docs/2.7/generated/torch.logical_and.html)

**产品支持情况**：

<!-- npu="910b" id571 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id571 -->
<!-- npu="A3" id572 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id572 -->
<!-- npu="950" id573 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id573 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.logical_not

<div style="margin-left: 2em">

**原生文档**：[torch.logical_not](https://pytorch.org/docs/2.7/generated/torch.logical_not.html)

**产品支持情况**：

<!-- npu="910b" id574 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id574 -->
<!-- npu="A3" id575 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id575 -->
<!-- npu="950" id576 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id576 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.logical_or

<div style="margin-left: 2em">

**原生文档**：[torch.logical_or](https://pytorch.org/docs/2.7/generated/torch.logical_or.html)

**产品支持情况**：

<!-- npu="910b" id577 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id577 -->
<!-- npu="A3" id578 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id578 -->
<!-- npu="950" id579 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id579 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.logical_xor

<div style="margin-left: 2em">

**原生文档**：[torch.logical_xor](https://pytorch.org/docs/2.7/generated/torch.logical_xor.html)

**产品支持情况**：

<!-- npu="910b" id580 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id580 -->
<!-- npu="A3" id581 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id581 -->
<!-- npu="950" id582 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id582 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.logit

<div style="margin-left: 2em">

**原生文档**：[torch.logit](https://pytorch.org/docs/2.7/generated/torch.logit.html)

**产品支持情况**：

<!-- npu="910b" id583 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id583 -->
<!-- npu="A3" id584 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id584 -->
<!-- npu="950" id585 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id585 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- `eps`取值大于1时输出为nan，`eps`取值为1时输出为inf

</div>

### torch.mul

<div style="margin-left: 2em">

**原生文档**：[torch.mul](https://pytorch.org/docs/2.7/generated/torch.mul.html)

**产品支持情况**：

<!-- npu="910b" id586 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id586 -->
<!-- npu="A3" id587 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id587 -->
<!-- npu="950" id588 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id588 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.multiply

<div style="margin-left: 2em">

**原生文档**：[torch.multiply](https://pytorch.org/docs/2.7/generated/torch.multiply.html)

**产品支持情况**：

<!-- npu="910b" id589 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id589 -->
<!-- npu="A3" id590 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id590 -->
<!-- npu="950" id591 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id591 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nan_to_num

<div style="margin-left: 2em">

**原生文档**：[torch.nan_to_num](https://pytorch.org/docs/2.7/generated/torch.nan_to_num.html)

**产品支持情况**：

<!-- npu="910b" id592 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id592 -->
<!-- npu="A3" id593 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id593 -->
<!-- npu="950" id594 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id594 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.neg

<div style="margin-left: 2em">

**原生文档**：[torch.neg](https://pytorch.org/docs/2.7/generated/torch.neg.html)

**产品支持情况**：

<!-- npu="910b" id595 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id595 -->
<!-- npu="A3" id596 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id596 -->
<!-- npu="950" id597 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id597 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，int8，int32，int64

</div>

### torch.negative

<div style="margin-left: 2em">

**原生文档**：[torch.negative](https://pytorch.org/docs/2.7/generated/torch.negative.html)

**产品支持情况**：

<!-- npu="910b" id598 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id598 -->
<!-- npu="A3" id599 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id599 -->
<!-- npu="950" id600 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id600 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，int8，int32，int64，complex64，complex128

</div>

### torch.positive

<div style="margin-left: 2em">

**原生文档**：[torch.positive](https://pytorch.org/docs/2.7/generated/torch.positive.html)

**产品支持情况**：

<!-- npu="910b" id601 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id601 -->
<!-- npu="A3" id602 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id602 -->
<!-- npu="950" id603 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id603 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，complex64，complex128

</div>

### torch.pow

<div style="margin-left: 2em">

**原生文档**：[torch.pow](https://pytorch.org/docs/2.7/generated/torch.pow.html)

**产品支持情况**：

<!-- npu="910b" id604 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id604 -->
<!-- npu="A3" id605 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id605 -->
<!-- npu="950" id606 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id606 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.rad2deg

<div style="margin-left: 2em">

**原生文档**：[torch.rad2deg](https://pytorch.org/docs/2.7/generated/torch.rad2deg.html)

**产品支持情况**：

<!-- npu="910b" id607 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id607 -->
<!-- npu="A3" id608 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id608 -->
<!-- npu="950" id609 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id609 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.reciprocal

<div style="margin-left: 2em">

**原生文档**：[torch.reciprocal](https://pytorch.org/docs/2.7/generated/torch.reciprocal.html)

**产品支持情况**：

<!-- npu="910b" id610 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id610 -->
<!-- npu="A3" id611 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id611 -->
<!-- npu="950" id612 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id612 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.remainder

<div style="margin-left: 2em">

**原生文档**：[torch.remainder](https://pytorch.org/docs/2.7/generated/torch.remainder.html)

**产品支持情况**：

<!-- npu="910b" id613 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id613 -->
<!-- npu="A3" id614 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id614 -->
<!-- npu="950" id615 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id615 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，int32，int64

</div>

### torch.round

<div style="margin-left: 2em">

**原生文档**：[torch.round](https://pytorch.org/docs/2.7/generated/torch.round.html)

**产品支持情况**：

<!-- npu="910b" id616 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id616 -->
<!-- npu="A3" id617 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id617 -->
<!-- npu="950" id618 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id618 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，int32，int64

</div>

### torch.rsqrt

<div style="margin-left: 2em">

**原生文档**：[torch.rsqrt](https://pytorch.org/docs/2.7/generated/torch.rsqrt.html)

**产品支持情况**：

<!-- npu="910b" id619 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id619 -->
<!-- npu="A3" id620 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id620 -->
<!-- npu="950" id621 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id621 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.sigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.sigmoid](https://pytorch.org/docs/2.7/generated/torch.sigmoid.html)

**产品支持情况**：

<!-- npu="910b" id622 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id622 -->
<!-- npu="A3" id623 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id623 -->
<!-- npu="950" id624 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id624 -->

**限制与说明**： `input`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.sign

<div style="margin-left: 2em">

**原生文档**：[torch.sign](https://pytorch.org/docs/2.7/generated/torch.sign.html)

**产品支持情况**：

<!-- npu="910b" id625 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id625 -->
<!-- npu="A3" id626 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id626 -->
<!-- npu="950" id627 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id627 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，int32，int64，bool

</div>

### torch.sgn

<div style="margin-left: 2em">

**原生文档**：[torch.sgn](https://pytorch.org/docs/2.7/generated/torch.sgn.html)

**产品支持情况**：

<!-- npu="910b" id628 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id628 -->
<!-- npu="A3" id629 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id629 -->
<!-- npu="950" id630 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id630 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，int32，int64，bool，complex64，complex128

</div>

### torch.sin

<div style="margin-left: 2em">

**原生文档**：[torch.sin](https://pytorch.org/docs/2.7/generated/torch.sin.html)

**产品支持情况**：

<!-- npu="910b" id631 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id631 -->
<!-- npu="A3" id632 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id632 -->
<!-- npu="950" id633 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id633 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.sinh

<div style="margin-left: 2em">

**原生文档**：[torch.sinh](https://pytorch.org/docs/2.7/generated/torch.sinh.html)

**产品支持情况**：

<!-- npu="910b" id634 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id634 -->
<!-- npu="A3" id635 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id635 -->
<!-- npu="950" id636 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id636 -->

**限制与说明**： `input`仅支持fp16，fp32，fp64

</div>

### torch.softmax

<div style="margin-left: 2em">

**原生文档**：[torch.softmax](https://pytorch.org/docs/2.7/generated/torch.softmax.html)

**产品支持情况**：

<!-- npu="910b" id637 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id637 -->
<!-- npu="A3" id638 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id638 -->
<!-- npu="950" id639 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id639 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64
- 支持Named Tensor

</div>

### torch.sqrt

<div style="margin-left: 2em">

**原生文档**：[torch.sqrt](https://pytorch.org/docs/2.7/generated/torch.sqrt.html)

**产品支持情况**：

<!-- npu="910b" id640 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id640 -->
<!-- npu="A3" id641 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id641 -->
<!-- npu="950" id642 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id642 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.square

<div style="margin-left: 2em">

**原生文档**：[torch.square](https://pytorch.org/docs/2.7/generated/torch.square.html)

**产品支持情况**：

<!-- npu="910b" id643 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id643 -->
<!-- npu="A3" id644 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id644 -->
<!-- npu="950" id645 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id645 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.sub

<div style="margin-left: 2em">

**原生文档**：[torch.sub](https://pytorch.org/docs/2.7/generated/torch.sub.html)

**产品支持情况**：

<!-- npu="910b" id646 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id646 -->
<!-- npu="A3" id647 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id647 -->
<!-- npu="950" id648 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id648 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.tan

<div style="margin-left: 2em">

**原生文档**：[torch.tan](https://pytorch.org/docs/2.7/generated/torch.tan.html)

**产品支持情况**：

<!-- npu="910b" id649 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id649 -->
<!-- npu="A3" id650 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id650 -->
<!-- npu="950" id651 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id651 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 取值范围[-65504,65504]

</div>

### torch.tanh

<div style="margin-left: 2em">

**原生文档**：[torch.tanh](https://pytorch.org/docs/2.7/generated/torch.tanh.html)

**产品支持情况**：

<!-- npu="910b" id652 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id652 -->
<!-- npu="A3" id653 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id653 -->
<!-- npu="950" id654 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id654 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.true_divide

<div style="margin-left: 2em">

**原生文档**：[torch.true_divide](https://pytorch.org/docs/2.7/generated/torch.true_divide.html)

**产品支持情况**：

<!-- npu="910b" id655 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id655 -->
<!-- npu="A3" id656 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id656 -->
<!-- npu="950" id657 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id657 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.trunc

<div style="margin-left: 2em">

**原生文档**：[torch.trunc](https://pytorch.org/docs/2.7/generated/torch.trunc.html)

**产品支持情况**：

<!-- npu="910b" id658 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id658 -->
<!-- npu="A3" id659 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id659 -->
<!-- npu="950" id660 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id660 -->

**限制与说明**：

- `input`仅支持fp16，fp32
- 可能回退至CPU执行

</div>

### torch.xlogy

<div style="margin-left: 2em">

**原生文档**：[torch.xlogy](https://pytorch.org/docs/2.7/generated/torch.xlogy.html)

**产品支持情况**：

<!-- npu="910b" id661 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id661 -->
<!-- npu="A3" id662 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id662 -->
<!-- npu="950" id663 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id663 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.argmax

<div style="margin-left: 2em">

**原生文档**：[torch.argmax](https://pytorch.org/docs/2.7/generated/torch.argmax.html)

**产品支持情况**：

<!-- npu="910b" id664 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id664 -->
<!-- npu="A3" id665 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id665 -->
<!-- npu="950" id666 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id666 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.argmin

<div style="margin-left: 2em">

**原生文档**：[torch.argmin](https://pytorch.org/docs/2.7/generated/torch.argmin.html)

**产品支持情况**：

<!-- npu="910b" id667 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id667 -->
<!-- npu="A3" id668 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id668 -->
<!-- npu="950" id669 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id669 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.amax

<div style="margin-left: 2em">

**原生文档**：[torch.amax](https://pytorch.org/docs/2.7/generated/torch.amax.html)

**产品支持情况**：

<!-- npu="910b" id670 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id670 -->
<!-- npu="A3" id671 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id671 -->
<!-- npu="950" id672 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id672 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.amin

<div style="margin-left: 2em">

**原生文档**：[torch.amin](https://pytorch.org/docs/2.7/generated/torch.amin.html)

**产品支持情况**：

<!-- npu="910b" id673 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id673 -->
<!-- npu="A3" id674 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id674 -->
<!-- npu="950" id675 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id675 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.aminmax

<div style="margin-left: 2em">

**原生文档**：[torch.aminmax](https://pytorch.org/docs/2.7/generated/torch.aminmax.html)

**产品支持情况**：

<!-- npu="910b" id676 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id676 -->
<!-- npu="A3" id677 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id677 -->
<!-- npu="950" id678 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id678 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.all

<div style="margin-left: 2em">

**原生文档**：[torch.all](https://pytorch.org/docs/2.7/generated/torch.all.html)

**产品支持情况**：

<!-- npu="910b" id679 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id679 -->
<!-- npu="A3" id680 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id680 -->
<!-- npu="950" id681 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id681 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.any

<div style="margin-left: 2em">

**原生文档**：[torch.any](https://pytorch.org/docs/2.7/generated/torch.any.html)

**产品支持情况**：

<!-- npu="910b" id682 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id682 -->
<!-- npu="A3" id683 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id683 -->
<!-- npu="950" id684 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id684 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.max

<div style="margin-left: 2em">

**原生文档**：[torch.max](https://pytorch.org/docs/2.7/generated/torch.max.html)

**产品支持情况**：

<!-- npu="910b" id685 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id685 -->
<!-- npu="A3" id686 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id686 -->
<!-- npu="950" id687 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id687 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，int64，bool

</div>

### torch.min

<div style="margin-left: 2em">

**原生文档**：[torch.min](https://pytorch.org/docs/2.7/generated/torch.min.html)

**产品支持情况**：

<!-- npu="910b" id688 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id688 -->
<!-- npu="A3" id689 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id689 -->
<!-- npu="950" id690 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id690 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，int64，bool

</div>

### torch.dist

<div style="margin-left: 2em">

**原生文档**：[torch.dist](https://pytorch.org/docs/2.7/generated/torch.dist.html)

**产品支持情况**：

<!-- npu="910b" id691 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id691 -->
<!-- npu="A3" id692 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id692 -->
<!-- npu="950" id693 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id693 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.logsumexp

<div style="margin-left: 2em">

**原生文档**：[torch.logsumexp](https://pytorch.org/docs/2.7/generated/torch.logsumexp.html)

**产品支持情况**：

<!-- npu="910b" id694 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id694 -->
<!-- npu="A3" id695 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id695 -->
<!-- npu="950" id696 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id696 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.mean

<div style="margin-left: 2em">

**原生文档**：[torch.mean](https://pytorch.org/docs/2.7/generated/torch.mean.html)

**产品支持情况**：

<!-- npu="910b" id697 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id697 -->
<!-- npu="A3" id698 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id698 -->
<!-- npu="950" id699 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id699 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，complex64，complex128

</div>

### torch.nanmean

<div style="margin-left: 2em">

**原生文档**：[torch.nanmean](https://pytorch.org/docs/2.7/generated/torch.nanmean.html)

**产品支持情况**：

<!-- npu="910b" id700 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id700 -->
<!-- npu="A3" id701 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id701 -->
<!-- npu="950" id702 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id702 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.median

<div style="margin-left: 2em">

**原生文档**：[torch.median](https://pytorch.org/docs/2.7/generated/torch.median.html)

**产品支持情况**：

<!-- npu="910b" id703 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id703 -->
<!-- npu="A3" id704 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id704 -->
<!-- npu="950" id705 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id705 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.norm

<div style="margin-left: 2em">

**原生文档**：[torch.norm](https://pytorch.org/docs/2.7/generated/torch.norm.html)

**产品支持情况**：

<!-- npu="910b" id706 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id706 -->
<!-- npu="A3" id707 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id707 -->
<!-- npu="950" id708 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id708 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 参数`dim`指定为输入`tensor`中shape维度值为1的轴时，计算结果可能存在精度误差

</div>

### torch.nansum

<div style="margin-left: 2em">

**原生文档**：[torch.nansum](https://pytorch.org/docs/2.7/generated/torch.nansum.html)

**产品支持情况**：

<!-- npu="910b" id709 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id709 -->
<!-- npu="A3" id710 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id710 -->
<!-- npu="950" id711 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id711 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.prod

<div style="margin-left: 2em">

**原生文档**：[torch.prod](https://pytorch.org/docs/2.7/generated/torch.prod.html)

**产品支持情况**：

<!-- npu="910b" id712 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id712 -->
<!-- npu="A3" id713 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id713 -->
<!-- npu="950" id714 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id714 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nanquantile

<div style="margin-left: 2em">

**原生文档**：[torch.nanquantile](https://pytorch.org/docs/2.7/generated/torch.nanquantile.html)

**产品支持情况**：

<!-- npu="910b" id715 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id715 -->
<!-- npu="A3" id716 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id716 -->
<!-- npu="950" id717 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id717 -->

</div>

### torch.std

<div style="margin-left: 2em">

**原生文档**：[torch.std](https://pytorch.org/docs/2.7/generated/torch.std.html)

**产品支持情况**：

<!-- npu="910b" id718 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id718 -->
<!-- npu="A3" id719 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id719 -->
<!-- npu="950" id720 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id720 -->

**限制与说明**： 可能回退至CPU执行

</div>

### torch.std_mean

<div style="margin-left: 2em">

**原生文档**：[torch.std_mean](https://pytorch.org/docs/2.7/generated/torch.std_mean.html)

**产品支持情况**：

<!-- npu="910b" id721 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id721 -->
<!-- npu="A3" id722 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id722 -->
<!-- npu="950" id723 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id723 -->

</div>

### torch.sum

<div style="margin-left: 2em">

**原生文档**：[torch.sum](https://pytorch.org/docs/2.7/generated/torch.sum.html)

**产品支持情况**：

<!-- npu="910b" id724 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id724 -->
<!-- npu="A3" id725 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id725 -->
<!-- npu="950" id726 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id726 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 不支持`dtype`参数

</div>

### torch.unique

<div style="margin-left: 2em">

**原生文档**：[torch.unique](https://pytorch.org/docs/2.7/generated/torch.unique.html)

**产品支持情况**：

<!-- npu="910b" id727 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id727 -->
<!-- npu="A3" id728 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id728 -->
<!-- npu="950" id729 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id729 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 带`dim`场景不支持fp16
- 在输入包含0的情况下，输出中可能会包含正0和负0，而非只输出一个0

</div>

### torch.unique_consecutive

<div style="margin-left: 2em">

**原生文档**：[torch.unique_consecutive](https://pytorch.org/docs/2.7/generated/torch.unique_consecutive.html)

**产品支持情况**：

<!-- npu="910b" id730 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id730 -->
<!-- npu="A3" id731 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id731 -->
<!-- npu="950" id732 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id732 -->

**限制与说明**： `input`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.var

<div style="margin-left: 2em">

**原生文档**：[torch.var](https://pytorch.org/docs/2.7/generated/torch.var.html)

**产品支持情况**：

<!-- npu="910b" id733 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id733 -->
<!-- npu="A3" id734 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id734 -->
<!-- npu="950" id735 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id735 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.var_mean

<div style="margin-left: 2em">

**原生文档**：[torch.var_mean](https://pytorch.org/docs/2.7/generated/torch.var_mean.html)

**产品支持情况**：

<!-- npu="910b" id736 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id736 -->
<!-- npu="A3" id737 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id737 -->
<!-- npu="950" id738 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id738 -->

</div>

### torch.count_nonzero

<div style="margin-left: 2em">

**原生文档**：[torch.count_nonzero](https://pytorch.org/docs/2.7/generated/torch.count_nonzero.html)

**产品支持情况**：

<!-- npu="910b" id739 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id739 -->
<!-- npu="A3" id740 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id740 -->
<!-- npu="950" id741 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id741 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.allclose

<div style="margin-left: 2em">

**原生文档**：[torch.allclose](https://pytorch.org/docs/2.7/generated/torch.allclose.html)

**产品支持情况**：

<!-- npu="910b" id742 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id742 -->
<!-- npu="A3" id743 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id743 -->
<!-- npu="950" id744 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id744 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.argsort

<div style="margin-left: 2em">

**原生文档**：[torch.argsort](https://pytorch.org/docs/2.7/generated/torch.argsort.html)

**产品支持情况**：

<!-- npu="910b" id745 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id745 -->
<!-- npu="A3" id746 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id746 -->
<!-- npu="950" id747 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id747 -->

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64

<!-- npu="950" id748 -->
- 针对<term>Ascend 950DT系列产品</term>，由于底层实现限制，`stable`仅支持True，若设置为False，执行时会被自动修改为True
<!-- end id748 -->

</div>

### torch.eq

<div style="margin-left: 2em">

**原生文档**：[torch.eq](https://pytorch.org/docs/2.7/generated/torch.eq.html)

**产品支持情况**：

<!-- npu="910b" id749 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id749 -->
<!-- npu="A3" id750 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id750 -->
<!-- npu="950" id751 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id751 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.equal

<div style="margin-left: 2em">

**原生文档**：[torch.equal](https://pytorch.org/docs/2.7/generated/torch.equal.html)

**产品支持情况**：

<!-- npu="910b" id752 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id752 -->
<!-- npu="A3" id753 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id753 -->
<!-- npu="950" id754 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id754 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.ge

<div style="margin-left: 2em">

**原生文档**：[torch.ge](https://pytorch.org/docs/2.7/generated/torch.ge.html)

**产品支持情况**：

<!-- npu="910b" id755 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id755 -->
<!-- npu="A3" id756 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id756 -->
<!-- npu="950" id757 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id757 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.greater_equal

<div style="margin-left: 2em">

**原生文档**：[torch.greater_equal](https://pytorch.org/docs/2.7/generated/torch.greater_equal.html)

**产品支持情况**：

<!-- npu="910b" id758 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id758 -->
<!-- npu="A3" id759 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id759 -->
<!-- npu="950" id760 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id760 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.gt

<div style="margin-left: 2em">

**原生文档**：[torch.gt](https://pytorch.org/docs/2.7/generated/torch.gt.html)

**产品支持情况**：

<!-- npu="910b" id761 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id761 -->
<!-- npu="A3" id762 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id762 -->
<!-- npu="950" id763 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id763 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.greater

<div style="margin-left: 2em">

**原生文档**：[torch.greater](https://pytorch.org/docs/2.7/generated/torch.greater.html)

**产品支持情况**：

<!-- npu="910b" id764 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id764 -->
<!-- npu="A3" id765 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id765 -->
<!-- npu="950" id766 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id766 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.isclose

<div style="margin-left: 2em">

**原生文档**：[torch.isclose](https://pytorch.org/docs/2.7/generated/torch.isclose.html)

**产品支持情况**：

<!-- npu="910b" id767 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id767 -->
<!-- npu="A3" id768 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id768 -->
<!-- npu="950" id769 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id769 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.isfinite

<div style="margin-left: 2em">

**原生文档**：[torch.isfinite](https://pytorch.org/docs/2.7/generated/torch.isfinite.html)

**产品支持情况**：

<!-- npu="910b" id770 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id770 -->
<!-- npu="A3" id771 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id771 -->
<!-- npu="950" id772 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id772 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.isin

<div style="margin-left: 2em">

**原生文档**：[torch.isin](https://pytorch.org/docs/2.7/generated/torch.isin.html)

**产品支持情况**：

<!-- npu="910b" id773 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id773 -->
<!-- npu="A3" id774 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id774 -->
<!-- npu="950" id775 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id775 -->

**限制与说明**：

- 双tensor输入的场景约束如下：
  - `elements`、`test_elements`仅支持fp16，fp32，uint8，int8，int16，int32，int64
  - 第一个输入`tensor`维度不能大于7维，第二个输入`tensor`维度不能大于8维
- 单tensor输入的场景约束如下：
  - `elements`、`test_elements`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64
  - 输入`tensor`的维度不大于8维

</div>

### torch.isinf

<div style="margin-left: 2em">

**原生文档**：[torch.isinf](https://pytorch.org/docs/2.7/generated/torch.isinf.html)

**产品支持情况**：

<!-- npu="910b" id776 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id776 -->
<!-- npu="A3" id777 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id777 -->
<!-- npu="950" id778 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id778 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.isposinf

<div style="margin-left: 2em">

**原生文档**：[torch.isposinf](https://pytorch.org/docs/2.7/generated/torch.isposinf.html)

**产品支持情况**：

<!-- npu="910b" id779 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id779 -->
<!-- npu="A3" id780 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id780 -->
<!-- npu="950" id781 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id781 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.isneginf

<div style="margin-left: 2em">

**原生文档**：[torch.isneginf](https://pytorch.org/docs/2.7/generated/torch.isneginf.html)

**产品支持情况**：

<!-- npu="910b" id782 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id782 -->
<!-- npu="A3" id783 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id783 -->
<!-- npu="950" id784 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id784 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.isnan

<div style="margin-left: 2em">

**原生文档**：[torch.isnan](https://pytorch.org/docs/2.7/generated/torch.isnan.html)

**产品支持情况**：

<!-- npu="910b" id785 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id785 -->
<!-- npu="A3" id786 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id786 -->
<!-- npu="950" id787 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id787 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.kthvalue

<div style="margin-left: 2em">

**原生文档**：[torch.kthvalue](https://pytorch.org/docs/2.7/generated/torch.kthvalue.html)

**产品支持情况**：

<!-- npu="910b" id788 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id788 -->
<!-- npu="A3" id789 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id789 -->
<!-- npu="950" id790 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id790 -->

**限制与说明**： `input`仅支持fp16，fp32，int32

</div>

### torch.le

<div style="margin-left: 2em">

**原生文档**：[torch.le](https://pytorch.org/docs/2.7/generated/torch.le.html)

**产品支持情况**：

<!-- npu="910b" id791 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id791 -->
<!-- npu="A3" id792 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id792 -->
<!-- npu="950" id793 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id793 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.less_equal

<div style="margin-left: 2em">

**原生文档**：[torch.less_equal](https://pytorch.org/docs/2.7/generated/torch.less_equal.html)

**产品支持情况**：

<!-- npu="910b" id794 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id794 -->
<!-- npu="A3" id795 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id795 -->
<!-- npu="950" id796 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id796 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，uint16，int8，int16，int32，int64，bool

</div>

### torch.lt

<div style="margin-left: 2em">

**原生文档**：[torch.lt](https://pytorch.org/docs/2.7/generated/torch.lt.html)

**产品支持情况**：

<!-- npu="910b" id797 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id797 -->
<!-- npu="A3" id798 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id798 -->
<!-- npu="950" id799 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id799 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.less

<div style="margin-left: 2em">

**原生文档**：[torch.less](https://pytorch.org/docs/2.7/generated/torch.less.html)

**产品支持情况**：

<!-- npu="910b" id800 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id800 -->
<!-- npu="A3" id801 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id801 -->
<!-- npu="950" id802 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id802 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.maximum

<div style="margin-left: 2em">

**原生文档**：[torch.maximum](https://pytorch.org/docs/2.7/generated/torch.maximum.html)

**产品支持情况**：

<!-- npu="910b" id803 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id803 -->
<!-- npu="A3" id804 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id804 -->
<!-- npu="950" id805 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id805 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.minimum

<div style="margin-left: 2em">

**原生文档**：[torch.minimum](https://pytorch.org/docs/2.7/generated/torch.minimum.html)

**产品支持情况**：

<!-- npu="910b" id806 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id806 -->
<!-- npu="A3" id807 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id807 -->
<!-- npu="950" id808 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id808 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.ne

<div style="margin-left: 2em">

**原生文档**：[torch.ne](https://pytorch.org/docs/2.7/generated/torch.ne.html)

**产品支持情况**：

<!-- npu="910b" id809 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id809 -->
<!-- npu="A3" id810 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id810 -->
<!-- npu="950" id811 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id811 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.not_equal

<div style="margin-left: 2em">

**原生文档**：[torch.not_equal](https://pytorch.org/docs/2.7/generated/torch.not_equal.html)

**产品支持情况**：

<!-- npu="910b" id812 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id812 -->
<!-- npu="A3" id813 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id813 -->
<!-- npu="950" id814 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id814 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.sort

<div style="margin-left: 2em">

**原生文档**：[torch.sort](https://pytorch.org/docs/2.7/generated/torch.sort.html)

**产品支持情况**：

<!-- npu="910b" id815 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id815 -->
<!-- npu="A3" id816 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id816 -->
<!-- npu="950" id817 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id817 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

<!-- npu="950" id818 -->
- 针对<term>Ascend 950DT系列产品</term>，由于底层实现限制，`stable`仅支持True，若设置为False，执行时会被自动修改为True
<!-- end id818 -->

</div>

### torch.topk

<div style="margin-left: 2em">

**原生文档**：[torch.topk](https://pytorch.org/docs/2.7/generated/torch.topk.html)

**产品支持情况**：

<!-- npu="910b" id819 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id819 -->
<!-- npu="A3" id820 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id820 -->
<!-- npu="950" id821 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id821 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64
- 不支持`sorted=False`场景

</div>

### torch.msort

<div style="margin-left: 2em">

**原生文档**：[torch.msort](https://pytorch.org/docs/2.7/generated/torch.msort.html)

**产品支持情况**：

<!-- npu="910b" id822 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id822 -->
<!-- npu="A3" id823 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id823 -->
<!-- npu="950" id824 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id824 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.stft

<div style="margin-left: 2em">

**原生文档**：[torch.stft](https://pytorch.org/docs/2.7/generated/torch.stft.html)

**产品支持情况**：

<!-- npu="910b" id825 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id825 -->
<!-- npu="A3" id826 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id826 -->
<!-- npu="950" id827 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id827 -->

**限制与说明**：

- `input`仅支持fp32，fp64，complex64，complex128
- 若算子超时，需要用官方接口`set_op_execute_time_out`进行设置，调高超时阈值以延长判断时间

</div>

### torch.hann_window

<div style="margin-left: 2em">

**原生文档**：[torch.hann_window](https://pytorch.org/docs/2.7/generated/torch.hann_window.html)

**产品支持情况**：

<!-- npu="910b" id828 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id828 -->
<!-- npu="A3" id829 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id829 -->
<!-- npu="950" id830 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id830 -->

**限制与说明**：

- `dtype`仅支持bf16，fp16，fp32
- 数据类型为fp32时，参数`window_length`在大于10000的情况下，计算结果可能存在误差

</div>

### torch.atleast_1d

<div style="margin-left: 2em">

**原生文档**：[torch.atleast_1d](https://pytorch.org/docs/2.7/generated/torch.atleast_1d.html)

**产品支持情况**：

<!-- npu="910b" id831 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id831 -->
<!-- npu="A3" id832 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id832 -->
<!-- npu="950" id833 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id833 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.atleast_2d

<div style="margin-left: 2em">

**原生文档**：[torch.atleast_2d](https://pytorch.org/docs/2.7/generated/torch.atleast_2d.html)

**产品支持情况**：

<!-- npu="910b" id834 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id834 -->
<!-- npu="A3" id835 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id835 -->
<!-- npu="950" id836 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id836 -->

**限制与说明**： `tensors`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.atleast_3d

<div style="margin-left: 2em">

**原生文档**：[torch.atleast_3d](https://pytorch.org/docs/2.7/generated/torch.atleast_3d.html)

**产品支持情况**：

<!-- npu="910b" id837 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id837 -->
<!-- npu="A3" id838 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id838 -->
<!-- npu="950" id839 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id839 -->

**限制与说明**： `tensors`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.bincount

<div style="margin-left: 2em">

**原生文档**：[torch.bincount](https://pytorch.org/docs/2.7/generated/torch.bincount.html)

**产品支持情况**：

<!-- npu="910b" id840 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id840 -->
<!-- npu="A3" id841 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id841 -->
<!-- npu="950" id842 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id842 -->

**限制与说明**：

- `input`仅支持uint8，int8，int16，int32，int64
- `weights`维度需与`input`维度一致

</div>

### torch.block_diag

<div style="margin-left: 2em">

**原生文档**：[torch.block_diag](https://pytorch.org/docs/2.7/generated/torch.block_diag.html)

**产品支持情况**：

<!-- npu="910b" id843 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id843 -->
<!-- npu="A3" id844 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id844 -->
<!-- npu="950" id845 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id845 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.broadcast_tensors

<div style="margin-left: 2em">

**原生文档**：[torch.broadcast_tensors](https://pytorch.org/docs/2.7/generated/torch.broadcast_tensors.html)

**产品支持情况**：

<!-- npu="910b" id846 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id846 -->
<!-- npu="A3" id847 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id847 -->
<!-- npu="950" id848 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id848 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.broadcast_to

<div style="margin-left: 2em">

**原生文档**：[torch.broadcast_to](https://pytorch.org/docs/2.7/generated/torch.broadcast_to.html)

**产品支持情况**：

<!-- npu="910b" id849 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id849 -->
<!-- npu="A3" id850 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id850 -->
<!-- npu="950" id851 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id851 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.broadcast_shapes

<div style="margin-left: 2em">

**原生文档**：[torch.broadcast_shapes](https://pytorch.org/docs/2.7/generated/torch.broadcast_shapes.html)

**产品支持情况**：

<!-- npu="910b" id852 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id852 -->
<!-- npu="A3" id853 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id853 -->
<!-- npu="950" id854 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id854 -->

</div>

### torch.cdist

<div style="margin-left: 2em">

**原生文档**：[torch.cdist](https://pytorch.org/docs/2.7/generated/torch.cdist.html)

**产品支持情况**：

<!-- npu="910b" id855 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id855 -->
<!-- npu="A3" id856 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id856 -->
<!-- npu="950" id857 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id857 -->

**限制与说明**：

- `x1`、`x2`仅支持bf16，fp16，fp32
- 当`p=2.0`时，`compute_mode`仅支持"donot_use_mm_for_euclid_dist"模式，传入其他值时，会自动修改为此模式

<!-- npu="950,A3,910b" id858 -->
- 针对<term>Ascend 950DT系列产品</term>，输入为fp16时，精度可能和<term>Atlas A2训练系列产品</term>/<term>Atlas A3训练系列产品</term>存在差异
<!-- end id858 -->

</div>

### torch.clone

<div style="margin-left: 2em">

**原生文档**：[torch.clone](https://pytorch.org/docs/2.7/generated/torch.clone.html)

**产品支持情况**：

<!-- npu="910b" id859 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id859 -->
<!-- npu="A3" id860 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id860 -->
<!-- npu="950" id861 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id861 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.combinations

<div style="margin-left: 2em">

**原生文档**：[torch.combinations](https://pytorch.org/docs/2.7/generated/torch.combinations.html)

**产品支持情况**：

<!-- npu="910b" id862 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id862 -->
<!-- npu="A3" id863 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id863 -->
<!-- npu="950" id864 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id864 -->

**限制与说明**： `input`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.cov

<div style="margin-left: 2em">

**原生文档**：[torch.cov](https://pytorch.org/docs/2.7/generated/torch.cov.html)

**产品支持情况**：

<!-- npu="910b" id865 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id865 -->
<!-- npu="A3" id866 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id866 -->
<!-- npu="950" id867 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id867 -->

**限制与说明**：`input`仅支持fp32

</div>

### torch.cross

<div style="margin-left: 2em">

**原生文档**：[torch.cross](https://pytorch.org/docs/2.7/generated/torch.cross.html)

**产品支持情况**：

<!-- npu="910b" id868 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id868 -->
<!-- npu="A3" id869 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id869 -->
<!-- npu="950" id870 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id870 -->

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，complex64，complex128
- 两个输入的shape要保持一致

</div>

### torch.cummax

<div style="margin-left: 2em">

**原生文档**：[torch.cummax](https://pytorch.org/docs/2.7/generated/torch.cummax.html)

**产品支持情况**：

<!-- npu="910b" id871 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id871 -->
<!-- npu="A3" id872 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id872 -->
<!-- npu="950" id873 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id873 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.cummin

<div style="margin-left: 2em">

**原生文档**：[torch.cummin](https://pytorch.org/docs/2.7/generated/torch.cummin.html)

**产品支持情况**：

<!-- npu="910b" id874 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id874 -->
<!-- npu="A3" id875 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id875 -->
<!-- npu="950" id876 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id876 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 输入为int32时，数值范围在[-16777216, 16777216]内

</div>

### torch.cumprod

<div style="margin-left: 2em">

**原生文档**：[torch.cumprod](https://pytorch.org/docs/2.7/generated/torch.cumprod.html)

**产品支持情况**：

<!-- npu="910b" id877 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id877 -->
<!-- npu="A3" id878 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id878 -->
<!-- npu="950" id879 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id879 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.cumsum

<div style="margin-left: 2em">

**原生文档**：[torch.cumsum](https://pytorch.org/docs/2.7/generated/torch.cumsum.html)

**产品支持情况**：

<!-- npu="910b" id880 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id880 -->
<!-- npu="A3" id881 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id881 -->
<!-- npu="950" id882 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id882 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 支持Named Tensor

</div>

### torch.diag

<div style="margin-left: 2em">

**原生文档**：[torch.diag](https://pytorch.org/docs/2.7/generated/torch.diag.html)

**产品支持情况**：

<!-- npu="910b" id883 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id883 -->
<!-- npu="A3" id884 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id884 -->
<!-- npu="950" id885 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id885 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.diag_embed

<div style="margin-left: 2em">

**原生文档**：[torch.diag_embed](https://pytorch.org/docs/2.7/generated/torch.diag_embed.html)

**产品支持情况**：

<!-- npu="910b" id886 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id886 -->
<!-- npu="A3" id887 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id887 -->
<!-- npu="950" id888 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id888 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.diagonal

<div style="margin-left: 2em">

**原生文档**：[torch.diagonal](https://pytorch.org/docs/2.7/generated/torch.diagonal.html)

**产品支持情况**：

<!-- npu="910b" id889 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id889 -->
<!-- npu="A3" id890 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id890 -->
<!-- npu="950" id891 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id891 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.diff

<div style="margin-left: 2em">

**原生文档**：[torch.diff](https://pytorch.org/docs/2.7/generated/torch.diff.html)

**产品支持情况**：

<!-- npu="910b" id892 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id892 -->
<!-- npu="A3" id893 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id893 -->
<!-- npu="950" id894 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id894 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.einsum

<div style="margin-left: 2em">

**原生文档**：[torch.einsum](https://pytorch.org/docs/2.7/generated/torch.einsum.html)

**产品支持情况**：

<!-- npu="910b" id895 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id895 -->
<!-- npu="A3" id896 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id896 -->
<!-- npu="950" id897 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id897 -->

**限制与说明**： `operands`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.flatten

<div style="margin-left: 2em">

**原生文档**：[torch.flatten](https://pytorch.org/docs/2.7/generated/torch.flatten.html)

**产品支持情况**：

<!-- npu="910b" id898 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id898 -->
<!-- npu="A3" id899 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id899 -->
<!-- npu="950" id900 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id900 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.flip

<div style="margin-left: 2em">

**原生文档**：[torch.flip](https://pytorch.org/docs/2.7/generated/torch.flip.html)

**产品支持情况**：

<!-- npu="910b" id901 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id901 -->
<!-- npu="A3" id902 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id902 -->
<!-- npu="950" id903 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id903 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.fliplr

<div style="margin-left: 2em">

**原生文档**：[torch.fliplr](https://pytorch.org/docs/2.7/generated/torch.fliplr.html)

**产品支持情况**：

<!-- npu="910b" id904 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id904 -->
<!-- npu="A3" id905 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id905 -->
<!-- npu="950" id906 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id906 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.flipud

<div style="margin-left: 2em">

**原生文档**：[torch.flipud](https://pytorch.org/docs/2.7/generated/torch.flipud.html)

**产品支持情况**：

<!-- npu="910b" id907 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id907 -->
<!-- npu="A3" id908 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id908 -->
<!-- npu="950" id909 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id909 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.kron

<div style="margin-left: 2em">

**原生文档**：[torch.kron](https://pytorch.org/docs/2.7/generated/torch.kron.html)

**产品支持情况**：

<!-- npu="910b" id910 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id910 -->
<!-- npu="A3" id911 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id911 -->
<!-- npu="950" id912 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id912 -->

**限制与说明**： 不支持5维度及以上输入

</div>

### torch.histc

<div style="margin-left: 2em">

**原生文档**：[torch.histc](https://pytorch.org/docs/2.7/generated/torch.histc.html)

**产品支持情况**：

<!-- npu="910b" id913 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id913 -->
<!-- npu="A3" id914 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id914 -->
<!-- npu="950" id915 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id915 -->

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64
- 当输入`tensor`值处于计数区间交界时，归于左区间计数还是右区间计数可能存在误差

</div>

### torch.meshgrid

<div style="margin-left: 2em">

**原生文档**：[torch.meshgrid](https://pytorch.org/docs/2.7/generated/torch.meshgrid.html)

**产品支持情况**：

<!-- npu="910b" id916 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id916 -->
<!-- npu="A3" id917 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id917 -->
<!-- npu="950" id918 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id918 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.ravel

<div style="margin-left: 2em">

**原生文档**：[torch.ravel](https://pytorch.org/docs/2.7/generated/torch.ravel.html)

**产品支持情况**：

<!-- npu="910b" id919 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id919 -->
<!-- npu="A3" id920 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id920 -->
<!-- npu="950" id921 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id921 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.repeat_interleave

<div style="margin-left: 2em">

**原生文档**：[torch.repeat_interleave](https://pytorch.org/docs/2.7/generated/torch.repeat_interleave.html)

**产品支持情况**：

<!-- npu="910b" id922 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id922 -->
<!-- npu="A3" id923 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id923 -->
<!-- npu="950" id924 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id924 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool
- 输入张量在重复后得到输出，输出中元素个数需小于$2^{22}$

</div>

### torch.roll

<div style="margin-left: 2em">

**原生文档**：[torch.roll](https://pytorch.org/docs/2.7/generated/torch.roll.html)

**产品支持情况**：

<!-- npu="910b" id925 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id925 -->
<!-- npu="A3" id926 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id926 -->
<!-- npu="950" id927 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id927 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64，bool

</div>

### torch.searchsorted

<div style="margin-left: 2em">

**原生文档**：[torch.searchsorted](https://pytorch.org/docs/2.7/generated/torch.searchsorted.html)

**产品支持情况**：

<!-- npu="910b" id928 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id928 -->
<!-- npu="A3" id929 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id929 -->
<!-- npu="950" id930 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id930 -->

**限制与说明**： `sorted_sequence`、`values`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.tensordot

<div style="margin-left: 2em">

**原生文档**：[torch.tensordot](https://pytorch.org/docs/2.7/generated/torch.tensordot.html)

**产品支持情况**：

<!-- npu="910b" id931 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id931 -->
<!-- npu="A3" id932 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id932 -->
<!-- npu="950" id933 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id933 -->

**限制与说明**： `a`、`b`仅支持fp16，fp32

</div>

### torch.tril

<div style="margin-left: 2em">

**原生文档**：[torch.tril](https://pytorch.org/docs/2.7/generated/torch.tril.html)

**产品支持情况**：

<!-- npu="910b" id934 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id934 -->
<!-- npu="A3" id935 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id935 -->
<!-- npu="950" id936 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id936 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.tril_indices

<div style="margin-left: 2em">

**原生文档**：[torch.tril_indices](https://pytorch.org/docs/2.7/generated/torch.tril_indices.html)

**产品支持情况**：

<!-- npu="910b" id937 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id937 -->
<!-- npu="A3" id938 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id938 -->
<!-- npu="950" id939 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id939 -->

</div>

### torch.triu

<div style="margin-left: 2em">

**原生文档**：[torch.triu](https://pytorch.org/docs/2.7/generated/torch.triu.html)

**产品支持情况**：

<!-- npu="910b" id940 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id940 -->
<!-- npu="A3" id941 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id941 -->
<!-- npu="950" id942 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id942 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.triu_indices

<div style="margin-left: 2em">

**原生文档**：[torch.triu_indices](https://pytorch.org/docs/2.7/generated/torch.triu_indices.html)

**产品支持情况**：

<!-- npu="910b" id943 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id943 -->
<!-- npu="A3" id944 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id944 -->
<!-- npu="950" id945 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id945 -->

</div>

### torch.unflatten

<div style="margin-left: 2em">

**原生文档**：[torch.unflatten](https://pytorch.org/docs/2.7/generated/torch.unflatten.html)

**产品支持情况**：

<!-- npu="910b" id946 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id946 -->
<!-- npu="A3" id947 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id947 -->
<!-- npu="950" id948 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id948 -->

</div>

### torch.view_as_real

<div style="margin-left: 2em">

**原生文档**：[torch.view_as_real](https://pytorch.org/docs/2.7/generated/torch.view_as_real.html)

**产品支持情况**：

<!-- npu="910b" id949 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id949 -->
<!-- npu="A3" id950 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id950 -->
<!-- npu="950" id951 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id951 -->

**限制与说明**： `input`仅支持complex64，complex128

</div>

### torch.view_as_complex

<div style="margin-left: 2em">

**原生文档**：[torch.view_as_complex](https://pytorch.org/docs/2.7/generated/torch.view_as_complex.html)

**产品支持情况**：

<!-- npu="910b" id952 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id952 -->
<!-- npu="A3" id953 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id953 -->
<!-- npu="950" id954 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id954 -->

**限制与说明**：`input`仅支持fp32，fp64

</div>

### torch.resolve_conj

<div style="margin-left: 2em">

**原生文档**：[torch.resolve_conj](https://pytorch.org/docs/2.7/generated/torch.resolve_conj.html)

**产品支持情况**：

<!-- npu="910b" id955 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id955 -->
<!-- npu="A3" id956 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id956 -->
<!-- npu="950" id957 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id957 -->

**限制与说明**： `input`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.resolve_neg

<div style="margin-left: 2em">

**原生文档**：[torch.resolve_neg](https://pytorch.org/docs/2.7/generated/torch.resolve_neg.html)

**产品支持情况**：

<!-- npu="910b" id958 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id958 -->
<!-- npu="A3" id959 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id959 -->
<!-- npu="950" id960 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id960 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.addbmm

<div style="margin-left: 2em">

**原生文档**：[torch.addbmm](https://pytorch.org/docs/2.7/generated/torch.addbmm.html)

**产品支持情况**：

<!-- npu="910b" id961 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id961 -->
<!-- npu="A3" id962 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id962 -->
<!-- npu="950" id963 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id963 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### torch.addmm

<div style="margin-left: 2em">

**原生文档**：[torch.addmm](https://pytorch.org/docs/2.7/generated/torch.addmm.html)

**产品支持情况**：

<!-- npu="910b" id964 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id964 -->
<!-- npu="A3" id965 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id965 -->
<!-- npu="950" id966 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id966 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.addmv

<div style="margin-left: 2em">

**原生文档**：[torch.addmv](https://pytorch.org/docs/2.7/generated/torch.addmv.html)

**产品支持情况**：

<!-- npu="910b" id967 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id967 -->
<!-- npu="A3" id968 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id968 -->
<!-- npu="950" id969 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id969 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.addr

<div style="margin-left: 2em">

**原生文档**：[torch.addr](https://pytorch.org/docs/2.7/generated/torch.addr.html)

**产品支持情况**：

<!-- npu="910b" id970 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id970 -->
<!-- npu="A3" id971 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id971 -->
<!-- npu="950" id972 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id972 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.baddbmm

<div style="margin-left: 2em">

**原生文档**：[torch.baddbmm](https://pytorch.org/docs/2.7/generated/torch.baddbmm.html)

**产品支持情况**：

<!-- npu="910b" id973 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id973 -->
<!-- npu="A3" id974 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id974 -->
<!-- npu="950" id975 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id975 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.bmm

<div style="margin-left: 2em">

**原生文档**：[torch.bmm](https://pytorch.org/docs/2.7/generated/torch.bmm.html)

**产品支持情况**：

<!-- npu="910b" id976 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id976 -->
<!-- npu="A3" id977 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id977 -->
<!-- npu="950" id978 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id978 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.dot

<div style="margin-left: 2em">

**原生文档**：[torch.dot](https://pytorch.org/docs/2.7/generated/torch.dot.html)

**产品支持情况**：

<!-- npu="910b" id979 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id979 -->
<!-- npu="A3" id980 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id980 -->
<!-- npu="950" id981 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id981 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32

</div>

### torch.slogdet

<div style="margin-left: 2em">

**原生文档**：[torch.slogdet](https://pytorch.org/docs/2.7/generated/torch.slogdet.html)

**产品支持情况**：

<!-- npu="910b" id982 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id982 -->
<!-- npu="A3" id983 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id983 -->
<!-- npu="950" id984 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id984 -->

**限制与说明**：

- `input`仅支持fp32，complex64，complex128
- 可能回退至CPU执行

</div>

### torch.matmul

<div style="margin-left: 2em">

**原生文档**：[torch.matmul](https://pytorch.org/docs/2.7/generated/torch.matmul.html)

**产品支持情况**：

<!-- npu="910b" id985 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id985 -->
<!-- npu="A3" id986 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id986 -->
<!-- npu="950" id987 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id987 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 支持Named Tensor
- 输入最大支持6维

</div>

### torch.mm

<div style="margin-left: 2em">

**原生文档**：[torch.mm](https://pytorch.org/docs/2.7/generated/torch.mm.html)

**产品支持情况**：

<!-- npu="910b" id988 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id988 -->
<!-- npu="A3" id989 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id989 -->
<!-- npu="950" id990 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id990 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.outer

<div style="margin-left: 2em">

**原生文档**：[torch.outer](https://pytorch.org/docs/2.7/generated/torch.outer.html)

**产品支持情况**：

<!-- npu="910b" id991 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id991 -->
<!-- npu="A3" id992 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id992 -->
<!-- npu="950" id993 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id993 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.qr

<div style="margin-left: 2em">

**原生文档**：[torch.qr](https://pytorch.org/docs/2.7/generated/torch.qr.html)

**产品支持情况**：

<!-- npu="910b" id994 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id994 -->
<!-- npu="A3" id995 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id995 -->
<!-- npu="950" id996 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id996 -->

</div>

### torch.trapezoid

<div style="margin-left: 2em">

**原生文档**：[torch.trapezoid](https://pytorch.org/docs/2.7/generated/torch.trapezoid.html)

**产品支持情况**：

<!-- npu="910b" id997 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id997 -->
<!-- npu="A3" id998 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id998 -->
<!-- npu="950" id999 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id999 -->

**限制与说明**： `y`、`x`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.cumulative_trapezoid

<div style="margin-left: 2em">

**原生文档**：[torch.cumulative_trapezoid](https://pytorch.org/docs/2.7/generated/torch.cumulative_trapezoid.html)

**产品支持情况**：

<!-- npu="910b" id1000 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1000 -->
<!-- npu="A3" id1001 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1001 -->
<!-- npu="950" id1002 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1002 -->

**限制与说明**： `y`、`x`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.vdot

<div style="margin-left: 2em">

**原生文档**：[torch.vdot](https://pytorch.org/docs/2.7/generated/torch.vdot.html)

**产品支持情况**：

<!-- npu="910b" id1003 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1003 -->
<!-- npu="A3" id1004 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1004 -->
<!-- npu="950" id1005 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1005 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### torch.bucketize

<div style="margin-left: 2em">

**原生文档**：[torch.bucketize](https://pytorch.org/docs/2.7/generated/torch.bucketize.html)

**产品支持情况**：

<!-- npu="910b" id1006 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1006 -->
<!-- npu="A3" id1007 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1007 -->
<!-- npu="950" id1008 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1008 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.cartesian_prod

<div style="margin-left: 2em">

**原生文档**：[torch.cartesian_prod](https://pytorch.org/docs/2.7/generated/torch.cartesian_prod.html)

**产品支持情况**：

<!-- npu="910b" id1009 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1009 -->
<!-- npu="A3" id1010 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1010 -->
<!-- npu="950" id1011 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1011 -->

**限制与说明**： `tensors`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.mv

<div style="margin-left: 2em">

**原生文档**：[torch.mv](https://pytorch.org/docs/2.7/generated/torch.mv.html)

**产品支持情况**：

<!-- npu="910b" id1012 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1012 -->
<!-- npu="A3" id1013 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1013 -->
<!-- npu="950" id1014 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1014 -->

**限制与说明**： `input`仅支持fp16，fp32

</div>

### torch._foreach_sqrt

<div style="margin-left: 2em">

**原生文档**：[torch._foreach_sqrt](https://pytorch.org/docs/2.7/generated/torch._foreach_sqrt.html)

**产品支持情况**：

<!-- npu="910b" id1015 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1015 -->
<!-- npu="A3" id1016 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1016 -->
<!-- npu="950" id1017 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1017 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch._foreach_asin

<div style="margin-left: 2em">

**原生文档**：[torch._foreach_asin](https://pytorch.org/docs/2.7/generated/torch._foreach_asin.html)

**产品支持情况**：

<!-- npu="910b" id1018 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1018 -->
<!-- npu="A3" id1019 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1019 -->
<!-- npu="950" id1020 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1020 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch._foreach_neg_

<div style="margin-left: 2em">

**原生文档**：[torch._foreach_neg_](https://pytorch.org/docs/2.7/generated/torch._foreach_neg_.html#torch._foreach_neg_)

**产品支持情况**：

<!-- npu="910b" id1021 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1021 -->
<!-- npu="A3" id1022 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1022 -->
<!-- npu="950" id1023 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1023 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，int8，int32，int64

</div>

### torch.corrcoef

<div style="margin-left: 2em">

**原生文档**：[torch.corrcoef](https://pytorch.org/docs/2.7/generated/torch.corrcoef.html)

**产品支持情况**：

<!-- npu="910b" id1024 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1024 -->
<!-- npu="A3" id1025 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1025 -->
<!-- npu="950" id1026 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1026 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

## Utilities

### torch.compiled_with_cxx11_abi

<div style="margin-left: 2em">

**原生文档**：[torch.compiled_with_cxx11_abi](https://pytorch.org/docs/2.7/generated/torch.compiled_with_cxx11_abi.html)

**产品支持情况**：

<!-- npu="910b" id1027 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1027 -->
<!-- npu="A3" id1028 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1028 -->
<!-- npu="950" id1029 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1029 -->

</div>

### torch.result_type

<div style="margin-left: 2em">

**原生文档**：[torch.result_type](https://pytorch.org/docs/2.7/generated/torch.result_type.html)

**产品支持情况**：

<!-- npu="910b" id1030 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1030 -->
<!-- npu="A3" id1031 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1031 -->
<!-- npu="950" id1032 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1032 -->

**限制与说明**：`input`仅支持fp32

</div>

### torch.can_cast

<div style="margin-left: 2em">

**原生文档**：[torch.can_cast](https://pytorch.org/docs/2.7/generated/torch.can_cast.html)

**产品支持情况**：

<!-- npu="910b" id1033 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1033 -->
<!-- npu="A3" id1034 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1034 -->
<!-- npu="950" id1035 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1035 -->

</div>

### torch.promote_types

<div style="margin-left: 2em">

**原生文档**：[torch.promote_types](https://pytorch.org/docs/2.7/generated/torch.promote_types.html)

**产品支持情况**：

<!-- npu="910b" id1036 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1036 -->
<!-- npu="A3" id1037 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1037 -->
<!-- npu="950" id1038 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1038 -->

</div>

### torch.use_deterministic_algorithms

<div style="margin-left: 2em">

**原生文档**：[torch.use_deterministic_algorithms](https://pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html)

**产品支持情况**：

<!-- npu="910b" id1039 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1039 -->
<!-- npu="A3" id1040 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1040 -->
<!-- npu="950" id1041 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1041 -->

**限制与说明**：

- 同时设置HCCL_DETERMINISTIC和`torch.use_deterministic_algorithms`时，若HCCL_DETERMINISTIC开启确定性则HCCL接口启用确定性，否则HCCL确定性由`torch.use_deterministic_algorithms`接口控制
- 设置`torch.use_deterministic_algorithms`时，PyTorch默认会填充未初始化内存（`torch.utils.deterministic.fill_uninitialized_memory`默认值为True），而TorchNPU默认不填充。如需填充生效，需手动将`torch.utils.deterministic.fill_uninitialized_memory`设置为True

</div>

### torch.are_deterministic_algorithms_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.are_deterministic_algorithms_enabled](https://pytorch.org/docs/2.7/generated/torch.are_deterministic_algorithms_enabled.html)

**产品支持情况**：

<!-- npu="910b" id1042 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1042 -->
<!-- npu="A3" id1043 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1043 -->
<!-- npu="950" id1044 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1044 -->

</div>

### torch.is_deterministic_algorithms_warn_only_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.is_deterministic_algorithms_warn_only_enabled](https://pytorch.org/docs/2.7/generated/torch.is_deterministic_algorithms_warn_only_enabled.html)

**产品支持情况**：

<!-- npu="910b" id1045 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id1045 -->
<!-- npu="A3" id1046 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id1046 -->
<!-- npu="950" id1047 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1047 -->

</div>

### torch.set_deterministic_debug_mode

<div style="margin-left: 2em">

**原生文档**：[torch.set_deterministic_debug_mode](https://pytorch.org/docs/2.7/generated/torch.set_deterministic_debug_mode.html)

**产品支持情况**：

<!-- npu="910b" id1048 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1048 -->
<!-- npu="A3" id1049 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1049 -->
<!-- npu="950" id1050 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1050 -->

</div>

### torch.get_deterministic_debug_mode

<div style="margin-left: 2em">

**原生文档**：[torch.get_deterministic_debug_mode](https://pytorch.org/docs/2.7/generated/torch.get_deterministic_debug_mode.html)

**产品支持情况**：

<!-- npu="910b" id1051 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1051 -->
<!-- npu="A3" id1052 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1052 -->
<!-- npu="950" id1053 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1053 -->

</div>

### torch.set_float32_matmul_precision

<div style="margin-left: 2em">

**原生文档**：[torch.set_float32_matmul_precision](https://pytorch.org/docs/2.7/generated/torch.set_float32_matmul_precision.html)

**产品支持情况**：

<!-- npu="910b" id1054 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1054 -->
<!-- npu="A3" id1055 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1055 -->
<!-- npu="950" id1056 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1056 -->

</div>

### torch.get_float32_matmul_precision

<div style="margin-left: 2em">

**原生文档**：[torch.get_float32_matmul_precision](https://pytorch.org/docs/2.7/generated/torch.get_float32_matmul_precision.html)

**产品支持情况**：

<!-- npu="910b" id1057 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1057 -->
<!-- npu="A3" id1058 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1058 -->
<!-- npu="950" id1059 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1059 -->

</div>

### torch.set_warn_always

<div style="margin-left: 2em">

**原生文档**：[torch.set_warn_always](https://pytorch.org/docs/2.7/generated/torch.set_warn_always.html)

**产品支持情况**：

<!-- npu="910b" id1060 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1060 -->
<!-- npu="A3" id1061 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1061 -->
<!-- npu="950" id1062 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1062 -->

</div>

### torch.is_warn_always_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.is_warn_always_enabled](https://pytorch.org/docs/2.7/generated/torch.is_warn_always_enabled.html)

**产品支持情况**：

<!-- npu="910b" id1063 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1063 -->
<!-- npu="A3" id1064 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1064 -->
<!-- npu="950" id1065 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1065 -->

</div>

### torch.vmap

<div style="margin-left: 2em">

**原生文档**：[torch.vmap](https://pytorch.org/docs/2.7/generated/torch.vmap.html)

**产品支持情况**：

<!-- npu="910b" id1066 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1066 -->
<!-- npu="A3" id1067 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1067 -->
<!-- npu="950" id1068 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1068 -->

</div>

### torch._assert

<div style="margin-left: 2em">

**原生文档**：[torch._assert](https://pytorch.org/docs/2.7/generated/torch._assert.html)

**产品支持情况**：

<!-- npu="910b" id1069 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1069 -->
<!-- npu="A3" id1070 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1070 -->
<!-- npu="950" id1071 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1071 -->

</div>

## Symbolic Numbers

### <code><i>class</i></code> torch.SymInt

<div style="margin-left: 2em">

**原生文档**：[torch.SymInt](https://pytorch.org/docs/2.7/torch.html#torch.SymInt)

**产品支持情况**：

<!-- npu="910b" id1072 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1072 -->
<!-- npu="A3" id1073 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1073 -->
<!-- npu="950" id1074 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1074 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.SymBool

<div style="margin-left: 2em">

**原生文档**：[torch.SymBool](https://pytorch.org/docs/2.7/torch.html#torch.SymBool)

**产品支持情况**：

<!-- npu="910b" id1075 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1075 -->
<!-- npu="A3" id1076 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1076 -->
<!-- npu="950" id1077 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1077 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.SymFloat

<div style="margin-left: 2em">

**原生文档**：[torch.SymFloat](https://pytorch.org/docs/2.7/torch.html#torch.SymFloat)

**产品支持情况**：

<!-- npu="910b" id1078 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1078 -->
<!-- npu="A3" id1079 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1079 -->
<!-- npu="950" id1080 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1080 -->

**限制与说明**： -

> <font size="3">is_integer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.SymFloat.is_integer](https://pytorch.org/docs/2.7/torch.html#torch.SymFloat.is_integer)

**产品支持情况**：

<!-- npu="910b" id1081 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1081 -->
<!-- npu="A3" id1082 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1082 -->
<!-- npu="950" id1083 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1083 -->

</div>

</div>

### torch.sym_float

<div style="margin-left: 2em">

**原生文档**：[torch.sym_float](https://pytorch.org/docs/2.7/generated/torch.sym_float.html)

**产品支持情况**：

<!-- npu="910b" id1084 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1084 -->
<!-- npu="A3" id1085 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1085 -->
<!-- npu="950" id1086 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1086 -->

**限制与说明**： `input`仅支持fp32

</div>

### torch.sym_int

<div style="margin-left: 2em">

**原生文档**：[torch.sym_int](https://pytorch.org/docs/2.7/generated/torch.sym_int.html)

**产品支持情况**：

<!-- npu="910b" id1087 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1087 -->
<!-- npu="A3" id1088 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1088 -->
<!-- npu="950" id1089 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1089 -->

**限制与说明**： `input`仅支持fp32

</div>

### torch.sym_ite

<div style="margin-left: 2em">

**原生文档**：[torch.sym_ite](https://pytorch.org/docs/2.7/generated/torch.sym_ite.html)

**产品支持情况**：

<!-- npu="910b" id1090 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1090 -->
<!-- npu="A3" id1091 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1091 -->
<!-- npu="950" id1092 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1092 -->

</div>

### torch.sym_max

<div style="margin-left: 2em">

**原生文档**：[torch.sym_max](https://pytorch.org/docs/2.7/generated/torch.sym_max.html)

**产品支持情况**：

<!-- npu="910b" id1093 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1093 -->
<!-- npu="A3" id1094 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1094 -->
<!-- npu="950" id1095 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1095 -->

**限制与说明**： `input`仅支持fp32

</div>

### torch.sym_min

<div style="margin-left: 2em">

**原生文档**：[torch.sym_min](https://pytorch.org/docs/2.7/generated/torch.sym_min.html)

**产品支持情况**：

<!-- npu="910b" id1096 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1096 -->
<!-- npu="A3" id1097 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1097 -->
<!-- npu="950" id1098 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1098 -->

</div>

### torch.sym_not

<div style="margin-left: 2em">

**原生文档**：[torch.sym_not](https://pytorch.org/docs/2.7/generated/torch.sym_not.html)

**产品支持情况**：

<!-- npu="910b" id1099 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1099 -->
<!-- npu="A3" id1100 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1100 -->
<!-- npu="950" id1101 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1101 -->

</div>

## Optimizations

### torch.compile

<div style="margin-left: 2em">

**原生文档**：[torch.compile](https://pytorch.org/docs/2.7/generated/torch.compile.html)

**产品支持情况**：

<!-- npu="910b" id1102 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1102 -->
<!-- npu="A3" id1103 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1103 -->
<!-- npu="950" id1104 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1104 -->

**限制与说明**： `backend`可支持npugraphs，整体功能与`backend`="cudagraphs"一致

</div>

## Operator Tags

### <code><i>class</i></code> torch.Tag

<div style="margin-left: 2em">

**原生文档**：[torch.Tag](https://pytorch.org/docs/2.7/torch.html#torch.Tag)

**产品支持情况**：

<!-- npu="910b" id1105 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1105 -->
<!-- npu="A3" id1106 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1106 -->
<!-- npu="950" id1107 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1107 -->

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tag.name](https://pytorch.org/docs/2.7/torch.html#torch.Tag.name)

**产品支持情况**：

<!-- npu="910b" id1108 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1108 -->
<!-- npu="A3" id1109 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1109 -->
<!-- npu="950" id1110 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id1110 -->

</div>

</div>
