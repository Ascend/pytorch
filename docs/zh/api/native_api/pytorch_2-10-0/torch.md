# torch

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [Symbolic Numbers](#symbolic-numbers)
- [Tensors](#tensors)
- [Generators](#generators)
- [Random sampling](#random-sampling)
- [Serialization](#serialization)
- [Parallelism](#parallelism)
- [Locally disabling gradient computation](#locally-disabling-gradient-computation)
- [Math operations](#math-operations)
- [Utilities](#utilities)
- [Optimizations](#optimizations)

## base API

### torch.default_generator

<div style="margin-left: 2em">

**原生文档**：[torch.default_generator](https://pytorch.org/docs/2.10/torch.html#torch.torch.default_generator)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.SymBool

<div style="margin-left: 2em">

**原生文档**：[torch.SymBool](https://pytorch.org/docs/2.10/torch.html#torch.SymBool)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.Tag

<div style="margin-left: 2em">

**原生文档**：[torch.Tag](https://pytorch.org/docs/2.10/torch.html#torch.Tag)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tag.name](https://pytorch.org/docs/2.10/torch.html#torch.Tag.name)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Symbolic Numbers

### _`class`_ torch.SymInt

<div style="margin-left: 2em">

**原生文档**：[torch.SymInt](https://pytorch.org/docs/2.10/torch.html#torch.SymInt)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.SymFloat

<div style="margin-left: 2em">

**原生文档**：[torch.SymFloat](https://pytorch.org/docs/2.10/torch.html#torch.SymFloat)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.sym_float

<div style="margin-left: 2em">

**原生文档**：[torch.sym_float](https://pytorch.org/docs/2.10/generated/torch.sym_float.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.sym_int

<div style="margin-left: 2em">

**原生文档**：[torch.sym_int](https://pytorch.org/docs/2.10/generated/torch.sym_int.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### torch.sym_max

<div style="margin-left: 2em">

**原生文档**：[torch.sym_max](https://pytorch.org/docs/2.10/generated/torch.sym_max.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.sym_min

<div style="margin-left: 2em">

**原生文档**：[torch.sym_min](https://pytorch.org/docs/2.10/generated/torch.sym_min.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.sym_not

<div style="margin-left: 2em">

**原生文档**：[torch.sym_not](https://pytorch.org/docs/2.10/generated/torch.sym_not.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

## Tensors

### torch.is_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.is_tensor](https://pytorch.org/docs/2.10/generated/torch.is_tensor.html)

**是否支持**：是

</div>

### torch.is_storage

<div style="margin-left: 2em">

**原生文档**：[torch.is_storage](https://pytorch.org/docs/2.10/generated/torch.is_storage.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.is_complex

<div style="margin-left: 2em">

**原生文档**：[torch.is_complex](https://pytorch.org/docs/2.10/generated/torch.is_complex.html)

**是否支持**：是

**限制与说明**： 支持complex64，complex128

</div>

### torch.is_conj

<div style="margin-left: 2em">

**原生文档**：[torch.is_conj](https://pytorch.org/docs/2.10/generated/torch.is_conj.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.is_floating_point

<div style="margin-left: 2em">

**原生文档**：[torch.is_floating_point](https://pytorch.org/docs/2.10/generated/torch.is_floating_point.html)

**是否支持**：是

</div>

### torch.is_nonzero

<div style="margin-left: 2em">

**原生文档**：[torch.is_nonzero](https://pytorch.org/docs/2.10/generated/torch.is_nonzero.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.set_default_dtype

<div style="margin-left: 2em">

**原生文档**：[torch.set_default_dtype](https://pytorch.org/docs/2.10/generated/torch.set_default_dtype.html)

**是否支持**：是

</div>

### torch.get_default_dtype

<div style="margin-left: 2em">

**原生文档**：[torch.get_default_dtype](https://pytorch.org/docs/2.10/generated/torch.get_default_dtype.html)

**是否支持**：是

</div>

### torch.set_default_device

<div style="margin-left: 2em">

**原生文档**：[torch.set_default_device](https://pytorch.org/docs/2.10/generated/torch.set_default_device.html)

**是否支持**：是

</div>

### torch.set_default_tensor_type

<div style="margin-left: 2em">

**原生文档**：[torch.set_default_tensor_type](https://pytorch.org/docs/2.10/generated/torch.set_default_tensor_type.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 不支持传入torch.npu.DtypeTensor类型

</div>

### torch.numel

<div style="margin-left: 2em">

**原生文档**：[torch.numel](https://pytorch.org/docs/2.10/generated/torch.numel.html)

**是否支持**：是

</div>

### torch.set_printoptions

<div style="margin-left: 2em">

**原生文档**：[torch.set_printoptions](https://pytorch.org/docs/2.10/generated/torch.set_printoptions.html)

**是否支持**：是

</div>

### torch.set_flush_denormal

<div style="margin-left: 2em">

**原生文档**：[torch.set_flush_denormal](https://pytorch.org/docs/2.10/generated/torch.set_flush_denormal.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.tensor

<div style="margin-left: 2em">

**原生文档**：[torch.tensor](https://pytorch.org/docs/2.10/generated/torch.tensor.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.sparse_coo_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_coo_tensor](https://pytorch.org/docs/2.10/generated/torch.sparse_coo_tensor.html)

**是否支持**：是

**限制与说明**：

- indices支持int32，int64
- values支持fp16，fp32，int32
- dtype参数与values的dtype保持一致

</div>

### torch.sparse_csr_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_csr_tensor](https://pytorch.org/docs/2.10/generated/torch.sparse_csr_tensor.html)

**是否支持**：否

</div>

### torch.sparse_csc_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_csc_tensor](https://pytorch.org/docs/2.10/generated/torch.sparse_csc_tensor.html)

**是否支持**：否

</div>

### torch.sparse_bsr_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_bsr_tensor](https://pytorch.org/docs/2.10/generated/torch.sparse_bsr_tensor.html)

**是否支持**：否

</div>

### torch.sparse_bsc_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.sparse_bsc_tensor](https://pytorch.org/docs/2.10/generated/torch.sparse_bsc_tensor.html)

**是否支持**：否

</div>

### torch.asarray

<div style="margin-left: 2em">

**原生文档**：[torch.asarray](https://pytorch.org/docs/2.10/generated/torch.asarray.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.as_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.as_tensor](https://pytorch.org/docs/2.10/generated/torch.as_tensor.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.as_strided

<div style="margin-left: 2em">

**原生文档**：[torch.as_strided](https://pytorch.org/docs/2.10/generated/torch.as_strided.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### torch.from_numpy

<div style="margin-left: 2em">

**原生文档**：[torch.from_numpy](https://pytorch.org/docs/2.10/generated/torch.from_numpy.html)

**是否支持**：是

**限制与说明**： 支持输出fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.from_dlpack

<div style="margin-left: 2em">

**原生文档**：[torch.from_dlpack](https://pytorch.org/docs/2.10/generated/torch.from_dlpack.html)

**是否支持**：否

</div>

### torch.frombuffer

<div style="margin-left: 2em">

**原生文档**：[torch.frombuffer](https://pytorch.org/docs/2.10/generated/torch.frombuffer.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.zeros

<div style="margin-left: 2em">

**原生文档**：[torch.zeros](https://pytorch.org/docs/2.10/generated/torch.zeros.html)

**是否支持**：是

</div>

### torch.zeros_like

<div style="margin-left: 2em">

**原生文档**：[torch.zeros_like](https://pytorch.org/docs/2.10/generated/torch.zeros_like.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.ones

<div style="margin-left: 2em">

**原生文档**：[torch.ones](https://pytorch.org/docs/2.10/generated/torch.ones.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.ones_like

<div style="margin-left: 2em">

**原生文档**：[torch.ones_like](https://pytorch.org/docs/2.10/generated/torch.ones_like.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.arange

<div style="margin-left: 2em">

**原生文档**：[torch.arange](https://pytorch.org/docs/2.10/generated/torch.arange.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，int32，int64

</div>

### torch.range

<div style="margin-left: 2em">

**原生文档**：[torch.range](https://pytorch.org/docs/2.10/generated/torch.range.html)

**是否支持**：是

</div>

### torch.linspace

<div style="margin-left: 2em">

**原生文档**：[torch.linspace](https://pytorch.org/docs/2.10/generated/torch.linspace.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，int16，int32，int64，bool，complex64，complex128
- 创建序列大小为steps的1维向量

</div>

### torch.eye

<div style="margin-left: 2em">

**原生文档**：[torch.eye](https://pytorch.org/docs/2.10/generated/torch.eye.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.empty

<div style="margin-left: 2em">

**原生文档**：[torch.empty](https://pytorch.org/docs/2.10/generated/torch.empty.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.empty_like

<div style="margin-left: 2em">

**原生文档**：[torch.empty_like](https://pytorch.org/docs/2.10/generated/torch.empty_like.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.empty_strided

<div style="margin-left: 2em">

**原生文档**：[torch.empty_strided](https://pytorch.org/docs/2.10/generated/torch.empty_strided.html)

**是否支持**：是

</div>

### torch.full

<div style="margin-left: 2em">

**原生文档**：[torch.full](https://pytorch.org/docs/2.10/generated/torch.full.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.full_like

<div style="margin-left: 2em">

**原生文档**：[torch.full_like](https://pytorch.org/docs/2.10/generated/torch.full_like.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.quantize_per_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.quantize_per_tensor](https://pytorch.org/docs/2.10/generated/torch.quantize_per_tensor.html)

**是否支持**：否

</div>

### torch.quantize_per_channel

<div style="margin-left: 2em">

**原生文档**：[torch.quantize_per_channel](https://pytorch.org/docs/2.10/generated/torch.quantize_per_channel.html)

**是否支持**：否

</div>

### torch.dequantize

<div style="margin-left: 2em">

**原生文档**：[torch.dequantize](https://pytorch.org/docs/2.10/generated/torch.dequantize.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.complex

<div style="margin-left: 2em">

**原生文档**：[torch.complex](https://pytorch.org/docs/2.10/generated/torch.complex.html)

**是否支持**：是

</div>

### torch.polar

<div style="margin-left: 2em">

**原生文档**：[torch.polar](https://pytorch.org/docs/2.10/generated/torch.polar.html)

**是否支持**：是

**限制与说明**：

- 支持fp32
- 入参abs和angle的维度需相等

</div>

### torch.heaviside

<div style="margin-left: 2em">

**原生文档**：[torch.heaviside](https://pytorch.org/docs/2.10/generated/torch.heaviside.html)

**是否支持**：否

</div>

### torch.argwhere

<div style="margin-left: 2em">

**原生文档**：[torch.argwhere](https://pytorch.org/docs/2.10/generated/torch.argwhere.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.cat

<div style="margin-left: 2em">

**原生文档**：[torch.cat](https://pytorch.org/docs/2.10/generated/torch.cat.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.concat

<div style="margin-left: 2em">

**原生文档**：[torch.concat](https://pytorch.org/docs/2.10/generated/torch.concat.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64
- <term>Ascend 950DT</term>：不支持complex64

</div>

### torch.concatenate

<div style="margin-left: 2em">

**原生文档**：[torch.concatenate](https://pytorch.org/docs/2.10/generated/torch.concatenate.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，int64，bool，complex64
- <term>Ascend 950DT</term>：不支持complex64

</div>

### torch.conj

<div style="margin-left: 2em">

**原生文档**：[torch.conj](https://pytorch.org/docs/2.10/generated/torch.conj.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.chunk

<div style="margin-left: 2em">

**原生文档**：[torch.chunk](https://pytorch.org/docs/2.10/generated/torch.chunk.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.dsplit

<div style="margin-left: 2em">

**原生文档**：[torch.dsplit](https://pytorch.org/docs/2.10/generated/torch.dsplit.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.dstack

<div style="margin-left: 2em">

**原生文档**：[torch.dstack](https://pytorch.org/docs/2.10/generated/torch.dstack.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.gather

<div style="margin-left: 2em">

**原生文档**：[torch.gather](https://pytorch.org/docs/2.10/generated/torch.gather.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，int16，int32，int64，bool
- index的维度数需与input的维度数一致
- 针对<term>Ascend 950DT</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

### torch.hsplit

<div style="margin-left: 2em">

**原生文档**：[torch.hsplit](https://pytorch.org/docs/2.10/generated/torch.hsplit.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.hstack

<div style="margin-left: 2em">

**原生文档**：[torch.hstack](https://pytorch.org/docs/2.10/generated/torch.hstack.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.index_add

<div style="margin-left: 2em">

**原生文档**：[torch.index_add](https://pytorch.org/docs/2.10/generated/torch.index_add.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int64，bool

</div>

### torch.index_copy

<div style="margin-left: 2em">

**原生文档**：[torch.index_copy](https://pytorch.org/docs/2.10/generated/torch.index_copy.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.index_reduce

<div style="margin-left: 2em">

**原生文档**：[torch.index_reduce](https://pytorch.org/docs/2.10/generated/torch.index_reduce.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

### torch.index_select

<div style="margin-left: 2em">

**原生文档**：[torch.index_select](https://pytorch.org/docs/2.10/generated/torch.index_select.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，int16，int32，int64，bool
- 针对<term>Ascend 950DT</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

### torch.masked_select

<div style="margin-left: 2em">

**原生文档**：[torch.masked_select](https://pytorch.org/docs/2.10/generated/torch.masked_select.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int16，int32，int64，bool

</div>

### torch.movedim

<div style="margin-left: 2em">

**原生文档**：[torch.movedim](https://pytorch.org/docs/2.10/generated/torch.movedim.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.moveaxis

<div style="margin-left: 2em">

**原生文档**：[torch.moveaxis](https://pytorch.org/docs/2.10/generated/torch.moveaxis.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32，int64，complex128

</div>

### torch.narrow

<div style="margin-left: 2em">

**原生文档**：[torch.narrow](https://pytorch.org/docs/2.10/generated/torch.narrow.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.narrow_copy

<div style="margin-left: 2em">

**原生文档**：[torch.narrow_copy](https://pytorch.org/docs/2.10/generated/torch.narrow_copy.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 可能回退至CPU执行

</div>

### torch.nonzero

<div style="margin-left: 2em">

**原生文档**：[torch.nonzero](https://pytorch.org/docs/2.10/generated/torch.nonzero.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.permute

<div style="margin-left: 2em">

**原生文档**：[torch.permute](https://pytorch.org/docs/2.10/generated/torch.permute.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.reshape

<div style="margin-left: 2em">

**原生文档**：[torch.reshape](https://pytorch.org/docs/2.10/generated/torch.reshape.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.row_stack

<div style="margin-left: 2em">

**原生文档**：[torch.row_stack](https://pytorch.org/docs/2.10/generated/torch.row_stack.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.scatter

<div style="margin-left: 2em">

**原生文档**：[torch.scatter](https://pytorch.org/docs/2.10/generated/torch.scatter.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 可能回退至CPU执行
- 针对<term>Ascend 950DT</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

### torch.diagonal_scatter

<div style="margin-left: 2em">

**原生文档**：[torch.diagonal_scatter](https://pytorch.org/docs/2.10/generated/torch.diagonal_scatter.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int16，int32，int64，bool

</div>

### torch.select_scatter

<div style="margin-left: 2em">

**原生文档**：[torch.select_scatter](https://pytorch.org/docs/2.10/generated/torch.select_scatter.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.slice_scatter

<div style="margin-left: 2em">

**原生文档**：[torch.slice_scatter](https://pytorch.org/docs/2.10/generated/torch.slice_scatter.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.scatter_reduce

<div style="margin-left: 2em">

**原生文档**：[torch.scatter_reduce](https://pytorch.org/docs/2.10/generated/torch.scatter_reduce.html)

**是否支持**：否

</div>

### torch.split

<div style="margin-left: 2em">

**原生文档**：[torch.split](https://pytorch.org/docs/2.10/generated/torch.split.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.squeeze

<div style="margin-left: 2em">

**原生文档**：[torch.squeeze](https://pytorch.org/docs/2.10/generated/torch.squeeze.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.stack

<div style="margin-left: 2em">

**原生文档**：[torch.stack](https://pytorch.org/docs/2.10/generated/torch.stack.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.swapaxes

<div style="margin-left: 2em">

**原生文档**：[torch.swapaxes](https://pytorch.org/docs/2.10/generated/torch.swapaxes.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.swapdims

<div style="margin-left: 2em">

**原生文档**：[torch.swapdims](https://pytorch.org/docs/2.10/generated/torch.swapdims.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.t

<div style="margin-left: 2em">

**原生文档**：[torch.t](https://pytorch.org/docs/2.10/generated/torch.t.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.take

<div style="margin-left: 2em">

**原生文档**：[torch.take](https://pytorch.org/docs/2.10/generated/torch.take.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int16，int32，int64，bool

</div>

### torch.take_along_dim

<div style="margin-left: 2em">

**原生文档**：[torch.take_along_dim](https://pytorch.org/docs/2.10/generated/torch.take_along_dim.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### torch.tensor_split

<div style="margin-left: 2em">

**原生文档**：[torch.tensor_split](https://pytorch.org/docs/2.10/generated/torch.tensor_split.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.tile

<div style="margin-left: 2em">

**原生文档**：[torch.tile](https://pytorch.org/docs/2.10/generated/torch.tile.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 若入参dims的长度小于input.shape的长度，则会在dims前自动补全1，使其长度与 input.shape对齐。补全后的dims，需要满足如下限制：
- - 当需要对第一根轴进行重复时，最多允许同时对4个维度进行重复操作（即dims中大于1的元素个数 ≤ 4），例如：不支持torch.tile(input, [2, 3, 4, 5, 6]) ，支持torch.tile(input, [2, 3, 1, 5, 6])
- - 当不需要对第一根轴进行重复时，最多允许同时对3个维度进行重复操作（即dims中大于1的元素个数 ≤ 3），例如：不支持torch.tile(input, [1, 3, 4, 5, 6]) ，支持torch.tile(input, [1, 3, 1, 5, 6])
- - 若执行反向计算，输入Tensor的维度数与入参dims中大于1的元素个数之和不得超过8

</div>

### torch.transpose

<div style="margin-left: 2em">

**原生文档**：[torch.transpose](https://pytorch.org/docs/2.10/generated/torch.transpose.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.unsqueeze

<div style="margin-left: 2em">

**原生文档**：[torch.unsqueeze](https://pytorch.org/docs/2.10/generated/torch.unsqueeze.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.vsplit

<div style="margin-left: 2em">

**原生文档**：[torch.vsplit](https://pytorch.org/docs/2.10/generated/torch.vsplit.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.vstack

<div style="margin-left: 2em">

**原生文档**：[torch.vstack](https://pytorch.org/docs/2.10/generated/torch.vstack.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.where

<div style="margin-left: 2em">

**原生文档**：[torch.where](https://pytorch.org/docs/2.10/generated/torch.where.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 不支持8维度的shape

</div>

### torch.rand

<div style="margin-left: 2em">

**原生文档**：[torch.rand](https://pytorch.org/docs/2.10/generated/torch.rand.html)

**是否支持**：是

</div>

### torch.rand_like

<div style="margin-left: 2em">

**原生文档**：[torch.rand_like](https://pytorch.org/docs/2.10/generated/torch.rand_like.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64
- 遵循PyTorch社区规范，不再支持对bool类型数据进行处理。针对存量bool类型数据可以通过如下方案进行替换：如果需要输出全True，可以采用torch.bernoulli(input, 1)。如果需要输出均匀分布的bool类型，则采用torch.bernoulli(input, 0.5)

</div>

### torch.randint

<div style="margin-left: 2em">

**原生文档**：[torch.randint](https://pytorch.org/docs/2.10/generated/torch.randint.html)

**是否支持**：是

</div>

### torch.randint_like

<div style="margin-left: 2em">

**原生文档**：[torch.randint_like](https://pytorch.org/docs/2.10/generated/torch.randint_like.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int64

</div>

### torch.randn

<div style="margin-left: 2em">

**原生文档**：[torch.randn](https://pytorch.org/docs/2.10/generated/torch.randn.html)

**是否支持**：是

</div>

### torch.randn_like

<div style="margin-left: 2em">

**原生文档**：[torch.randn_like](https://pytorch.org/docs/2.10/generated/torch.randn_like.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.randperm

<div style="margin-left: 2em">

**原生文档**：[torch.randperm](https://pytorch.org/docs/2.10/generated/torch.randperm.html)

**是否支持**：是

</div>

### torch.abs

<div style="margin-left: 2em">

**原生文档**：[torch.abs](https://pytorch.org/docs/2.10/generated/torch.abs.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.angle

<div style="margin-left: 2em">

**原生文档**：[torch.angle](https://pytorch.org/docs/2.10/generated/torch.angle.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.real

<div style="margin-left: 2em">

**原生文档**：[torch.real](https://pytorch.org/docs/2.10/generated/torch.real.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

## Generators

### _`class`_ torch.Generator

<div style="margin-left: 2em">

**原生文档**：[torch.Generator](https://pytorch.org/docs/2.10/generated/torch.Generator.html)

**是否支持**：是

> <font size="3">device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.device](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.device)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">get_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.get_state](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.get_state)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">initial_seed()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.initial_seed](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.initial_seed)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">manual_seed()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.manual_seed](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.manual_seed)

**是否支持**：是

</div>

> <font size="3">seed()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.seed](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.seed)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Generator.set_state](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.set_state)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Random sampling

### torch.seed

<div style="margin-left: 2em">

**原生文档**：[torch.seed](https://pytorch.org/docs/2.10/generated/torch.seed.html)

**是否支持**：是

</div>

### torch.manual_seed

<div style="margin-left: 2em">

**原生文档**：[torch.manual_seed](https://pytorch.org/docs/2.10/generated/torch.manual_seed.html)

**是否支持**：是

</div>

### torch.initial_seed

<div style="margin-left: 2em">

**原生文档**：[torch.initial_seed](https://pytorch.org/docs/2.10/generated/torch.initial_seed.html)

**是否支持**：是

</div>

### torch.get_rng_state

<div style="margin-left: 2em">

**原生文档**：[torch.get_rng_state](https://pytorch.org/docs/2.10/generated/torch.get_rng_state.html)

**是否支持**：是

</div>

### torch.set_rng_state

<div style="margin-left: 2em">

**原生文档**：[torch.set_rng_state](https://pytorch.org/docs/2.10/generated/torch.set_rng_state.html)

**是否支持**：是

</div>

### torch.bernoulli

<div style="margin-left: 2em">

**原生文档**：[torch.bernoulli](https://pytorch.org/docs/2.10/generated/torch.bernoulli.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64

</div>

### torch.multinomial

<div style="margin-left: 2em">

**原生文档**：[torch.multinomial](https://pytorch.org/docs/2.10/generated/torch.multinomial.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.normal

<div style="margin-left: 2em">

**原生文档**：[torch.normal](https://pytorch.org/docs/2.10/generated/torch.normal.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.poisson

<div style="margin-left: 2em">

**原生文档**：[torch.poisson](https://pytorch.org/docs/2.10/generated/torch.poisson.html)

**是否支持**：否

</div>

### _`class`_ torch.quasirandom.SobolEngine

<div style="margin-left: 2em">

> <font size="3">draw()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.quasirandom.SobolEngine.draw](https://pytorch.org/docs/2.10/generated/torch.quasirandom.SobolEngine.html#torch.quasirandom.SobolEngine.draw)

**是否支持**：是

**限制与说明**： 支持fp32，fp64

</div>

</div>

## Serialization

### torch.save

<div style="margin-left: 2em">

**原生文档**：[torch.save](https://pytorch.org/docs/2.10/generated/torch.save.html)

**是否支持**：是

</div>

### torch.load

<div style="margin-left: 2em">

**原生文档**：[torch.load](https://pytorch.org/docs/2.10/generated/torch.load.html)

**是否支持**：是

</div>

## Parallelism

### torch.get_num_threads

<div style="margin-left: 2em">

**原生文档**：[torch.get_num_threads](https://pytorch.org/docs/2.10/generated/torch.get_num_threads.html)

**是否支持**：是

</div>

### torch.set_num_threads

<div style="margin-left: 2em">

**原生文档**：[torch.set_num_threads](https://pytorch.org/docs/2.10/generated/torch.set_num_threads.html)

**是否支持**：是

</div>

### torch.get_num_interop_threads

<div style="margin-left: 2em">

**原生文档**：[torch.get_num_interop_threads](https://pytorch.org/docs/2.10/generated/torch.get_num_interop_threads.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.set_num_interop_threads

<div style="margin-left: 2em">

**原生文档**：[torch.set_num_interop_threads](https://pytorch.org/docs/2.10/generated/torch.set_num_interop_threads.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Locally disabling gradient computation

### torch.no_grad

<div style="margin-left: 2em">

**原生文档**：[torch.no_grad](https://pytorch.org/docs/2.10/generated/torch.no_grad.html)

**是否支持**：是

</div>

### torch.enable_grad

<div style="margin-left: 2em">

**原生文档**：[torch.enable_grad](https://pytorch.org/docs/2.10/generated/torch.enable_grad.html)

**是否支持**：是

</div>

### torch.autograd.grad_mode.set_grad_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.grad_mode.set_grad_enabled](https://pytorch.org/docs/2.10/generated/torch.autograd.grad_mode.set_grad_enabled.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.is_grad_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.is_grad_enabled](https://pytorch.org/docs/2.10/generated/torch.is_grad_enabled.html)

**是否支持**：是

</div>

### torch.autograd.grad_mode.inference_mode

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.grad_mode.inference_mode](https://pytorch.org/docs/2.10/generated/torch.autograd.grad_mode.inference_mode.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.is_inference_mode_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.is_inference_mode_enabled](https://pytorch.org/docs/2.10/generated/torch.is_inference_mode_enabled.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Math operations

### torch.absolute

<div style="margin-left: 2em">

**原生文档**：[torch.absolute](https://pytorch.org/docs/2.10/generated/torch.absolute.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.acos

<div style="margin-left: 2em">

**原生文档**：[torch.acos](https://pytorch.org/docs/2.10/generated/torch.acos.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.arccos

<div style="margin-left: 2em">

**原生文档**：[torch.arccos](https://pytorch.org/docs/2.10/generated/torch.arccos.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.acosh

<div style="margin-left: 2em">

**原生文档**：[torch.acosh](https://pytorch.org/docs/2.10/generated/torch.acosh.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 可能回退至CPU执行

</div>

### torch.arccosh

<div style="margin-left: 2em">

**原生文档**：[torch.arccosh](https://pytorch.org/docs/2.10/generated/torch.arccosh.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.add

<div style="margin-left: 2em">

**原生文档**：[torch.add](https://pytorch.org/docs/2.10/generated/torch.add.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.addcdiv

<div style="margin-left: 2em">

**原生文档**：[torch.addcdiv](https://pytorch.org/docs/2.10/generated/torch.addcdiv.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，int64
- 在int64类型下不支持三个tensor同时广播

</div>

### torch.addcmul

<div style="margin-left: 2em">

**原生文档**：[torch.addcmul](https://pytorch.org/docs/2.10/generated/torch.addcmul.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64
- 在fp64，uint8，int8，int64类型下不支持三个tensor同时广播

</div>

### torch.asin

<div style="margin-left: 2em">

**原生文档**：[torch.asin](https://pytorch.org/docs/2.10/generated/torch.asin.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.arcsin

<div style="margin-left: 2em">

**原生文档**：[torch.arcsin](https://pytorch.org/docs/2.10/generated/torch.arcsin.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.asinh

<div style="margin-left: 2em">

**原生文档**：[torch.asinh](https://pytorch.org/docs/2.10/generated/torch.asinh.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.arcsinh

<div style="margin-left: 2em">

**原生文档**：[torch.arcsinh](https://pytorch.org/docs/2.10/generated/torch.arcsinh.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.atan

<div style="margin-left: 2em">

**原生文档**：[torch.atan](https://pytorch.org/docs/2.10/generated/torch.atan.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.arctan

<div style="margin-left: 2em">

**原生文档**：[torch.arctan](https://pytorch.org/docs/2.10/generated/torch.arctan.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.atanh

<div style="margin-left: 2em">

**原生文档**：[torch.atanh](https://pytorch.org/docs/2.10/generated/torch.atanh.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.arctanh

<div style="margin-left: 2em">

**原生文档**：[torch.arctanh](https://pytorch.org/docs/2.10/generated/torch.arctanh.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.atan2

<div style="margin-left: 2em">

**原生文档**：[torch.atan2](https://pytorch.org/docs/2.10/generated/torch.atan2.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.arctan2

<div style="margin-left: 2em">

**原生文档**：[torch.arctan2](https://pytorch.org/docs/2.10/generated/torch.arctan2.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.bitwise_not

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_not](https://pytorch.org/docs/2.10/generated/torch.bitwise_not.html)

**是否支持**：是

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

### torch.bitwise_and

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_and](https://pytorch.org/docs/2.10/generated/torch.bitwise_and.html)

**是否支持**：是

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

### torch.bitwise_or

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_or](https://pytorch.org/docs/2.10/generated/torch.bitwise_or.html)

**是否支持**：是

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

### torch.bitwise_xor

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_xor](https://pytorch.org/docs/2.10/generated/torch.bitwise_xor.html)

**是否支持**：是

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

### torch.bitwise_left_shift

<div style="margin-left: 2em">

**原生文档**：[torch.bitwise_left_shift](https://pytorch.org/docs/2.10/generated/torch.bitwise_left_shift.html)

**是否支持**：是

**限制与说明**： 支持uint8，int8，int16，int32，int64

</div>

### torch.ceil

<div style="margin-left: 2em">

**原生文档**：[torch.ceil](https://pytorch.org/docs/2.10/generated/torch.ceil.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.clamp

<div style="margin-left: 2em">

**原生文档**：[torch.clamp](https://pytorch.org/docs/2.10/generated/torch.clamp.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.clip

<div style="margin-left: 2em">

**原生文档**：[torch.clip](https://pytorch.org/docs/2.10/generated/torch.clip.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.copysign

<div style="margin-left: 2em">

**原生文档**：[torch.copysign](https://pytorch.org/docs/2.10/generated/torch.copysign.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool
- 可能回退至CPU执行

</div>

### torch.cos

<div style="margin-left: 2em">

**原生文档**：[torch.cos](https://pytorch.org/docs/2.10/generated/torch.cos.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.cosh

<div style="margin-left: 2em">

**原生文档**：[torch.cosh](https://pytorch.org/docs/2.10/generated/torch.cosh.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.deg2rad

<div style="margin-left: 2em">

**原生文档**：[torch.deg2rad](https://pytorch.org/docs/2.10/generated/torch.deg2rad.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.div

<div style="margin-left: 2em">

**原生文档**：[torch.div](https://pytorch.org/docs/2.10/generated/torch.div.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.divide

<div style="margin-left: 2em">

**原生文档**：[torch.divide](https://pytorch.org/docs/2.10/generated/torch.divide.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.erf

<div style="margin-left: 2em">

**原生文档**：[torch.erf](https://pytorch.org/docs/2.10/generated/torch.erf.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，int64，bool

</div>

### torch.erfc

<div style="margin-left: 2em">

**原生文档**：[torch.erfc](https://pytorch.org/docs/2.10/generated/torch.erfc.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int64，bool

</div>

### torch.erfinv

<div style="margin-left: 2em">

**原生文档**：[torch.erfinv](https://pytorch.org/docs/2.10/generated/torch.erfinv.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.exp

<div style="margin-left: 2em">

**原生文档**：[torch.exp](https://pytorch.org/docs/2.10/generated/torch.exp.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，int64，bool，complex64，complex128

</div>

### torch.exp2

<div style="margin-left: 2em">

**原生文档**：[torch.exp2](https://pytorch.org/docs/2.10/generated/torch.exp2.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.expm1

<div style="margin-left: 2em">

**原生文档**：[torch.expm1](https://pytorch.org/docs/2.10/generated/torch.expm1.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，int64，bool

</div>

### torch.fix

<div style="margin-left: 2em">

**原生文档**：[torch.fix](https://pytorch.org/docs/2.10/generated/torch.fix.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.float_power

<div style="margin-left: 2em">

**原生文档**：[torch.float_power](https://pytorch.org/docs/2.10/generated/torch.float_power.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex128

</div>

### torch.floor

<div style="margin-left: 2em">

**原生文档**：[torch.floor](https://pytorch.org/docs/2.10/generated/torch.floor.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.floor_divide

<div style="margin-left: 2em">

**原生文档**：[torch.floor_divide](https://pytorch.org/docs/2.10/generated/torch.floor_divide.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.fmod

<div style="margin-left: 2em">

**原生文档**：[torch.fmod](https://pytorch.org/docs/2.10/generated/torch.fmod.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.gradient

<div style="margin-left: 2em">

**原生文档**：[torch.gradient](https://pytorch.org/docs/2.10/generated/torch.gradient.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，int8，int16，int32，int64

</div>

### torch.ldexp

<div style="margin-left: 2em">

**原生文档**：[torch.ldexp](https://pytorch.org/docs/2.10/generated/torch.ldexp.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp64，complex64

</div>

### torch.lerp

<div style="margin-left: 2em">

**原生文档**：[torch.lerp](https://pytorch.org/docs/2.10/generated/torch.lerp.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.log

<div style="margin-left: 2em">

**原生文档**：[torch.log](https://pytorch.org/docs/2.10/generated/torch.log.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.log10

<div style="margin-left: 2em">

**原生文档**：[torch.log10](https://pytorch.org/docs/2.10/generated/torch.log10.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 当输入input为uint8，int8，int16，int32，int64，bool时，输出out必须为fp32
- 其余支持数据类型输出out和输入input保持一致

</div>

### torch.log1p

<div style="margin-left: 2em">

**原生文档**：[torch.log1p](https://pytorch.org/docs/2.10/generated/torch.log1p.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.log2

<div style="margin-left: 2em">

**原生文档**：[torch.log2](https://pytorch.org/docs/2.10/generated/torch.log2.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.logaddexp

<div style="margin-left: 2em">

**原生文档**：[torch.logaddexp](https://pytorch.org/docs/2.10/generated/torch.logaddexp.html)

**是否支持**：是

**限制与说明**： 不支持double数据类型

</div>

### torch.logaddexp2

<div style="margin-left: 2em">

**原生文档**：[torch.logaddexp2](https://pytorch.org/docs/2.10/generated/torch.logaddexp2.html)

**是否支持**：是

**限制与说明**： 不支持double数据类型

</div>

### torch.logical_and

<div style="margin-left: 2em">

**原生文档**：[torch.logical_and](https://pytorch.org/docs/2.10/generated/torch.logical_and.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.logical_not

<div style="margin-left: 2em">

**原生文档**：[torch.logical_not](https://pytorch.org/docs/2.10/generated/torch.logical_not.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.logical_or

<div style="margin-left: 2em">

**原生文档**：[torch.logical_or](https://pytorch.org/docs/2.10/generated/torch.logical_or.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.logical_xor

<div style="margin-left: 2em">

**原生文档**：[torch.logical_xor](https://pytorch.org/docs/2.10/generated/torch.logical_xor.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.logit

<div style="margin-left: 2em">

**原生文档**：[torch.logit](https://pytorch.org/docs/2.10/generated/torch.logit.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- eps取值大于1时输出为nan，eps取值为1时输出为inf

</div>

### torch.mul

<div style="margin-left: 2em">

**原生文档**：[torch.mul](https://pytorch.org/docs/2.10/generated/torch.mul.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.multiply

<div style="margin-left: 2em">

**原生文档**：[torch.multiply](https://pytorch.org/docs/2.10/generated/torch.multiply.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nan_to_num

<div style="margin-left: 2em">

**原生文档**：[torch.nan_to_num](https://pytorch.org/docs/2.10/generated/torch.nan_to_num.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.neg

<div style="margin-left: 2em">

**原生文档**：[torch.neg](https://pytorch.org/docs/2.10/generated/torch.neg.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int8，int32，int64，complex64，complex128

</div>

### torch.negative

<div style="margin-left: 2em">

**原生文档**：[torch.negative](https://pytorch.org/docs/2.10/generated/torch.negative.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int8，int32，int64，complex64，complex128

</div>

### torch.positive

<div style="margin-left: 2em">

**原生文档**：[torch.positive](https://pytorch.org/docs/2.10/generated/torch.positive.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，complex64，complex128

</div>

### torch.pow

<div style="margin-left: 2em">

**原生文档**：[torch.pow](https://pytorch.org/docs/2.10/generated/torch.pow.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，int16，int64

</div>

### torch.rad2deg

<div style="margin-left: 2em">

**原生文档**：[torch.rad2deg](https://pytorch.org/docs/2.10/generated/torch.rad2deg.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.reciprocal

<div style="margin-left: 2em">

**原生文档**：[torch.reciprocal](https://pytorch.org/docs/2.10/generated/torch.reciprocal.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.remainder

<div style="margin-left: 2em">

**原生文档**：[torch.remainder](https://pytorch.org/docs/2.10/generated/torch.remainder.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int16，int32，int64

</div>

### torch.round

<div style="margin-left: 2em">

**原生文档**：[torch.round](https://pytorch.org/docs/2.10/generated/torch.round.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，int32，int64

</div>

### torch.rsqrt

<div style="margin-left: 2em">

**原生文档**：[torch.rsqrt](https://pytorch.org/docs/2.10/generated/torch.rsqrt.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.sigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.sigmoid](https://pytorch.org/docs/2.10/generated/torch.sigmoid.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.sign

<div style="margin-left: 2em">

**原生文档**：[torch.sign](https://pytorch.org/docs/2.10/generated/torch.sign.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int32，int64，bool

</div>

### torch.sgn

<div style="margin-left: 2em">

**原生文档**：[torch.sgn](https://pytorch.org/docs/2.10/generated/torch.sgn.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int32，int64，bool，complex64，complex128

</div>

### torch.sin

<div style="margin-left: 2em">

**原生文档**：[torch.sin](https://pytorch.org/docs/2.10/generated/torch.sin.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.sinh

<div style="margin-left: 2em">

**原生文档**：[torch.sinh](https://pytorch.org/docs/2.10/generated/torch.sinh.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64

</div>

### torch.softmax

<div style="margin-left: 2em">

**原生文档**：[torch.softmax](https://pytorch.org/docs/2.10/generated/torch.softmax.html)

**是否支持**：是

**限制与说明**：

- 支持fp32
- 支持Named Tensor

</div>

### torch.sqrt

<div style="margin-left: 2em">

**原生文档**：[torch.sqrt](https://pytorch.org/docs/2.10/generated/torch.sqrt.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.square

<div style="margin-left: 2em">

**原生文档**：[torch.square](https://pytorch.org/docs/2.10/generated/torch.square.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.sub

<div style="margin-left: 2em">

**原生文档**：[torch.sub](https://pytorch.org/docs/2.10/generated/torch.sub.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.tan

<div style="margin-left: 2em">

**原生文档**：[torch.tan](https://pytorch.org/docs/2.10/generated/torch.tan.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 取值范围[-65504,65504]

</div>

### torch.tanh

<div style="margin-left: 2em">

**原生文档**：[torch.tanh](https://pytorch.org/docs/2.10/generated/torch.tanh.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.true_divide

<div style="margin-left: 2em">

**原生文档**：[torch.true_divide](https://pytorch.org/docs/2.10/generated/torch.true_divide.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.trunc

<div style="margin-left: 2em">

**原生文档**：[torch.trunc](https://pytorch.org/docs/2.10/generated/torch.trunc.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- 可能回退至CPU执行

</div>

### torch.xlogy

<div style="margin-left: 2em">

**原生文档**：[torch.xlogy](https://pytorch.org/docs/2.10/generated/torch.xlogy.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.argmax

<div style="margin-left: 2em">

**原生文档**：[torch.argmax](https://pytorch.org/docs/2.10/generated/torch.argmax.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.argmin

<div style="margin-left: 2em">

**原生文档**：[torch.argmin](https://pytorch.org/docs/2.10/generated/torch.argmin.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.amax

<div style="margin-left: 2em">

**原生文档**：[torch.amax](https://pytorch.org/docs/2.10/generated/torch.amax.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.amin

<div style="margin-left: 2em">

**原生文档**：[torch.amin](https://pytorch.org/docs/2.10/generated/torch.amin.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.aminmax

<div style="margin-left: 2em">

**原生文档**：[torch.aminmax](https://pytorch.org/docs/2.10/generated/torch.aminmax.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.all

<div style="margin-left: 2em">

**原生文档**：[torch.all](https://pytorch.org/docs/2.10/generated/torch.all.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.any

<div style="margin-left: 2em">

**原生文档**：[torch.any](https://pytorch.org/docs/2.10/generated/torch.any.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.max

<div style="margin-left: 2em">

**原生文档**：[torch.max](https://pytorch.org/docs/2.10/generated/torch.max.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int64，bool

</div>

### torch.min

<div style="margin-left: 2em">

**原生文档**：[torch.min](https://pytorch.org/docs/2.10/generated/torch.min.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int64，bool

</div>

### torch.dist

<div style="margin-left: 2em">

**原生文档**：[torch.dist](https://pytorch.org/docs/2.10/generated/torch.dist.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.logsumexp

<div style="margin-left: 2em">

**原生文档**：[torch.logsumexp](https://pytorch.org/docs/2.10/generated/torch.logsumexp.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.mean

<div style="margin-left: 2em">

**原生文档**：[torch.mean](https://pytorch.org/docs/2.10/generated/torch.mean.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，complex64，complex128

</div>

### torch.nanmean

<div style="margin-left: 2em">

**原生文档**：[torch.nanmean](https://pytorch.org/docs/2.10/generated/torch.nanmean.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.median

<div style="margin-left: 2em">

**原生文档**：[torch.median](https://pytorch.org/docs/2.10/generated/torch.median.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.norm

<div style="margin-left: 2em">

**原生文档**：[torch.norm](https://pytorch.org/docs/2.10/generated/torch.norm.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 参数dim指定为输入tensor中shape维度值为1的轴时，计算结果可能存在精度误差

</div>

### torch.nansum

<div style="margin-left: 2em">

**原生文档**：[torch.nansum](https://pytorch.org/docs/2.10/generated/torch.nansum.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.prod

<div style="margin-left: 2em">

**原生文档**：[torch.prod](https://pytorch.org/docs/2.10/generated/torch.prod.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nanquantile

<div style="margin-left: 2em">

**原生文档**：[torch.nanquantile](https://pytorch.org/docs/2.10/generated/torch.nanquantile.html)

**是否支持**：否

</div>

### torch.std

<div style="margin-left: 2em">

**原生文档**：[torch.std](https://pytorch.org/docs/2.10/generated/torch.std.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

### torch.std_mean

<div style="margin-left: 2em">

**原生文档**：[torch.std_mean](https://pytorch.org/docs/2.10/generated/torch.std_mean.html)

**是否支持**：否

</div>

### torch.sum

<div style="margin-left: 2em">

**原生文档**：[torch.sum](https://pytorch.org/docs/2.10/generated/torch.sum.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 不支持dtype参数

</div>

### torch.unique

<div style="margin-left: 2em">

**原生文档**：[torch.unique](https://pytorch.org/docs/2.10/generated/torch.unique.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 带dim场景不支持fp16
- 在输入包含0的情况下，输出中可能会包含正0和负0，而非只输出一个0

</div>

### torch.unique_consecutive

<div style="margin-left: 2em">

**原生文档**：[torch.unique_consecutive](https://pytorch.org/docs/2.10/generated/torch.unique_consecutive.html)

**是否支持**：否

</div>

### torch.var

<div style="margin-left: 2em">

**原生文档**：[torch.var](https://pytorch.org/docs/2.10/generated/torch.var.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.var_mean

<div style="margin-left: 2em">

**原生文档**：[torch.var_mean](https://pytorch.org/docs/2.10/generated/torch.var_mean.html)

**是否支持**：否

</div>

### torch.count_nonzero

<div style="margin-left: 2em">

**原生文档**：[torch.count_nonzero](https://pytorch.org/docs/2.10/generated/torch.count_nonzero.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.allclose

<div style="margin-left: 2em">

**原生文档**：[torch.allclose](https://pytorch.org/docs/2.10/generated/torch.allclose.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.argsort

<div style="margin-left: 2em">

**原生文档**：[torch.argsort](https://pytorch.org/docs/2.10/generated/torch.argsort.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64
- 针对<term>Ascend 950DT</term>，由于底层实现限制，"stable"仅支持True，设置为False在执行时会自动修改为True

</div>

### torch.eq

<div style="margin-left: 2em">

**原生文档**：[torch.eq](https://pytorch.org/docs/2.10/generated/torch.eq.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.equal

<div style="margin-left: 2em">

**原生文档**：[torch.equal](https://pytorch.org/docs/2.10/generated/torch.equal.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.ge

<div style="margin-left: 2em">

**原生文档**：[torch.ge](https://pytorch.org/docs/2.10/generated/torch.ge.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.greater_equal

<div style="margin-left: 2em">

**原生文档**：[torch.greater_equal](https://pytorch.org/docs/2.10/generated/torch.greater_equal.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.gt

<div style="margin-left: 2em">

**原生文档**：[torch.gt](https://pytorch.org/docs/2.10/generated/torch.gt.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.greater

<div style="margin-left: 2em">

**原生文档**：[torch.greater](https://pytorch.org/docs/2.10/generated/torch.greater.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.isclose

<div style="margin-left: 2em">

**原生文档**：[torch.isclose](https://pytorch.org/docs/2.10/generated/torch.isclose.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.isfinite

<div style="margin-left: 2em">

**原生文档**：[torch.isfinite](https://pytorch.org/docs/2.10/generated/torch.isfinite.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.isin

<div style="margin-left: 2em">

**原生文档**：[torch.isin](https://pytorch.org/docs/2.10/generated/torch.isin.html)

**是否支持**：是

**限制与说明**：

- 双tensor输入的场景约束如下：
- - 支持fp16，fp32，uint8，int8，int16，int32，int64
- - 第一个输入tensor维度不能大于7维，第二个输入tensor维度不能大于8维
- 单tensor输入的场景约束如下：
- - 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64
- - 输入tensor的维度不大于8维

</div>

### torch.isinf

<div style="margin-left: 2em">

**原生文档**：[torch.isinf](https://pytorch.org/docs/2.10/generated/torch.isinf.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.isposinf

<div style="margin-left: 2em">

**原生文档**：[torch.isposinf](https://pytorch.org/docs/2.10/generated/torch.isposinf.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.isneginf

<div style="margin-left: 2em">

**原生文档**：[torch.isneginf](https://pytorch.org/docs/2.10/generated/torch.isneginf.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.isnan

<div style="margin-left: 2em">

**原生文档**：[torch.isnan](https://pytorch.org/docs/2.10/generated/torch.isnan.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.kthvalue

<div style="margin-left: 2em">

**原生文档**：[torch.kthvalue](https://pytorch.org/docs/2.10/generated/torch.kthvalue.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int32

</div>

### torch.le

<div style="margin-left: 2em">

**原生文档**：[torch.le](https://pytorch.org/docs/2.10/generated/torch.le.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.less_equal

<div style="margin-left: 2em">

**原生文档**：[torch.less_equal](https://pytorch.org/docs/2.10/generated/torch.less_equal.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.lt

<div style="margin-left: 2em">

**原生文档**：[torch.lt](https://pytorch.org/docs/2.10/generated/torch.lt.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.less

<div style="margin-left: 2em">

**原生文档**：[torch.less](https://pytorch.org/docs/2.10/generated/torch.less.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.maximum

<div style="margin-left: 2em">

**原生文档**：[torch.maximum](https://pytorch.org/docs/2.10/generated/torch.maximum.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.minimum

<div style="margin-left: 2em">

**原生文档**：[torch.minimum](https://pytorch.org/docs/2.10/generated/torch.minimum.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.ne

<div style="margin-left: 2em">

**原生文档**：[torch.ne](https://pytorch.org/docs/2.10/generated/torch.ne.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.not_equal

<div style="margin-left: 2em">

**原生文档**：[torch.not_equal](https://pytorch.org/docs/2.10/generated/torch.not_equal.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.sort

<div style="margin-left: 2em">

**原生文档**：[torch.sort](https://pytorch.org/docs/2.10/generated/torch.sort.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64
- 针对<term>Ascend 950DT</term>，由于底层实现限制，"stable"仅支持True，设置为False在执行时会自动修改为True

</div>

### torch.topk

<div style="margin-left: 2em">

**原生文档**：[torch.topk](https://pytorch.org/docs/2.10/generated/torch.topk.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64
- 不支持sorted=False场景

</div>

### torch.msort

<div style="margin-left: 2em">

**原生文档**：[torch.msort](https://pytorch.org/docs/2.10/generated/torch.msort.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.stft

<div style="margin-left: 2em">

**原生文档**：[torch.stft](https://pytorch.org/docs/2.10/generated/torch.stft.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp32，fp64，complex64，complex128
- 若算子超时，需要用官方接口set_op_execute_time_out进行设置，调高超时阈值以延长判断时间

</div>

### torch.hann_window

<div style="margin-left: 2em">

**原生文档**：[torch.hann_window](https://pytorch.org/docs/2.10/generated/torch.hann_window.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 数据类型为fp32时，参数window_length在大于10000的情况下，计算结果可能存在误差

</div>

### torch.atleast_1d

<div style="margin-left: 2em">

**原生文档**：[torch.atleast_1d](https://pytorch.org/docs/2.10/generated/torch.atleast_1d.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.atleast_2d

<div style="margin-left: 2em">

**原生文档**：[torch.atleast_2d](https://pytorch.org/docs/2.10/generated/torch.atleast_2d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.atleast_3d

<div style="margin-left: 2em">

**原生文档**：[torch.atleast_3d](https://pytorch.org/docs/2.10/generated/torch.atleast_3d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.bincount

<div style="margin-left: 2em">

**原生文档**：[torch.bincount](https://pytorch.org/docs/2.10/generated/torch.bincount.html)

**是否支持**：是

**限制与说明**：

- 支持uint8，int8，int16，int32，int64
- weights维度需与input维度一致

</div>

### torch.block_diag

<div style="margin-left: 2em">

**原生文档**：[torch.block_diag](https://pytorch.org/docs/2.10/generated/torch.block_diag.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.broadcast_tensors

<div style="margin-left: 2em">

**原生文档**：[torch.broadcast_tensors](https://pytorch.org/docs/2.10/generated/torch.broadcast_tensors.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.broadcast_to

<div style="margin-left: 2em">

**原生文档**：[torch.broadcast_to](https://pytorch.org/docs/2.10/generated/torch.broadcast_to.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.broadcast_shapes

<div style="margin-left: 2em">

**原生文档**：[torch.broadcast_shapes](https://pytorch.org/docs/2.10/generated/torch.broadcast_shapes.html)

**是否支持**：是

</div>

### torch.cdist

<div style="margin-left: 2em">

**原生文档**：[torch.cdist](https://pytorch.org/docs/2.10/generated/torch.cdist.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 当 p=2.0 时，"compute_mode"仅支持"donot_use_mm_for_euclid_dist"模式，传入其他值会自动修改为此模式
- 针对<term>Ascend 950DT</term>，输入为fp16时，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

### torch.clone

<div style="margin-left: 2em">

**原生文档**：[torch.clone](https://pytorch.org/docs/2.10/generated/torch.clone.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.combinations

<div style="margin-left: 2em">

**原生文档**：[torch.combinations](https://pytorch.org/docs/2.10/generated/torch.combinations.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.cov

<div style="margin-left: 2em">

**原生文档**：[torch.cov](https://pytorch.org/docs/2.10/generated/torch.cov.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### torch.cross

<div style="margin-left: 2em">

**原生文档**：[torch.cross](https://pytorch.org/docs/2.10/generated/torch.cross.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，complex64，complex128
- 两个输入的shape要保持一致

</div>

### torch.cummax

<div style="margin-left: 2em">

**原生文档**：[torch.cummax](https://pytorch.org/docs/2.10/generated/torch.cummax.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.cummin

<div style="margin-left: 2em">

**原生文档**：[torch.cummin](https://pytorch.org/docs/2.10/generated/torch.cummin.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 输入为int32时，数值范围在[-16777216, 16777216]内

</div>

### torch.cumprod

<div style="margin-left: 2em">

**原生文档**：[torch.cumprod](https://pytorch.org/docs/2.10/generated/torch.cumprod.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.cumsum

<div style="margin-left: 2em">

**原生文档**：[torch.cumsum](https://pytorch.org/docs/2.10/generated/torch.cumsum.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 支持Named Tensor

</div>

### torch.diag

<div style="margin-left: 2em">

**原生文档**：[torch.diag](https://pytorch.org/docs/2.10/generated/torch.diag.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

### torch.diag_embed

<div style="margin-left: 2em">

**原生文档**：[torch.diag_embed](https://pytorch.org/docs/2.10/generated/torch.diag_embed.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.diagonal

<div style="margin-left: 2em">

**原生文档**：[torch.diagonal](https://pytorch.org/docs/2.10/generated/torch.diagonal.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.diff

<div style="margin-left: 2em">

**原生文档**：[torch.diff](https://pytorch.org/docs/2.10/generated/torch.diff.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.einsum

<div style="margin-left: 2em">

**原生文档**：[torch.einsum](https://pytorch.org/docs/2.10/generated/torch.einsum.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.flatten

<div style="margin-left: 2em">

**原生文档**：[torch.flatten](https://pytorch.org/docs/2.10/generated/torch.flatten.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.flip

<div style="margin-left: 2em">

**原生文档**：[torch.flip](https://pytorch.org/docs/2.10/generated/torch.flip.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.fliplr

<div style="margin-left: 2em">

**原生文档**：[torch.fliplr](https://pytorch.org/docs/2.10/generated/torch.fliplr.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.flipud

<div style="margin-left: 2em">

**原生文档**：[torch.flipud](https://pytorch.org/docs/2.10/generated/torch.flipud.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.kron

<div style="margin-left: 2em">

**原生文档**：[torch.kron](https://pytorch.org/docs/2.10/generated/torch.kron.html)

**是否支持**：是

**限制与说明**： 不支持5维度及以上输入

</div>

### torch.histc

<div style="margin-left: 2em">

**原生文档**：[torch.histc](https://pytorch.org/docs/2.10/generated/torch.histc.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- 当输入tensor值处于计数区间交界时，归于左区间计数还是右区间计数可能存在误差

</div>

### torch.meshgrid

<div style="margin-left: 2em">

**原生文档**：[torch.meshgrid](https://pytorch.org/docs/2.10/generated/torch.meshgrid.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.ravel

<div style="margin-left: 2em">

**原生文档**：[torch.ravel](https://pytorch.org/docs/2.10/generated/torch.ravel.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.repeat_interleave

<div style="margin-left: 2em">

**原生文档**：[torch.repeat_interleave](https://pytorch.org/docs/2.10/generated/torch.repeat_interleave.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，int16，int32，int64，bool
- 输入张量在重复后得到输出，输出中元素个数需小于$2^{22}$

</div>

### torch.roll

<div style="margin-left: 2em">

**原生文档**：[torch.roll](https://pytorch.org/docs/2.10/generated/torch.roll.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int32，int64，bool

</div>

### torch.searchsorted

<div style="margin-left: 2em">

**原生文档**：[torch.searchsorted](https://pytorch.org/docs/2.10/generated/torch.searchsorted.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.tensordot

<div style="margin-left: 2em">

**原生文档**：[torch.tensordot](https://pytorch.org/docs/2.10/generated/torch.tensordot.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.tril

<div style="margin-left: 2em">

**原生文档**：[torch.tril](https://pytorch.org/docs/2.10/generated/torch.tril.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.tril_indices

<div style="margin-left: 2em">

**原生文档**：[torch.tril_indices](https://pytorch.org/docs/2.10/generated/torch.tril_indices.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.triu

<div style="margin-left: 2em">

**原生文档**：[torch.triu](https://pytorch.org/docs/2.10/generated/torch.triu.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.triu_indices

<div style="margin-left: 2em">

**原生文档**：[torch.triu_indices](https://pytorch.org/docs/2.10/generated/torch.triu_indices.html)

**是否支持**：是

</div>

### torch.unflatten

<div style="margin-left: 2em">

**原生文档**：[torch.unflatten](https://pytorch.org/docs/2.10/generated/torch.unflatten.html)

**是否支持**：是

</div>

### torch.view_as_real

<div style="margin-left: 2em">

**原生文档**：[torch.view_as_real](https://pytorch.org/docs/2.10/generated/torch.view_as_real.html)

**是否支持**：是

**限制与说明**： 支持complex64，complex128

</div>

### torch.view_as_complex

<div style="margin-left: 2em">

**原生文档**：[torch.view_as_complex](https://pytorch.org/docs/2.10/generated/torch.view_as_complex.html)

**是否支持**：是

**限制与说明**： 支持fp32，fp64

</div>

### torch.resolve_conj

<div style="margin-left: 2em">

**原生文档**：[torch.resolve_conj](https://pytorch.org/docs/2.10/generated/torch.resolve_conj.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.resolve_neg

<div style="margin-left: 2em">

**原生文档**：[torch.resolve_neg](https://pytorch.org/docs/2.10/generated/torch.resolve_neg.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.addbmm

<div style="margin-left: 2em">

**原生文档**：[torch.addbmm](https://pytorch.org/docs/2.10/generated/torch.addbmm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### torch.addmm

<div style="margin-left: 2em">

**原生文档**：[torch.addmm](https://pytorch.org/docs/2.10/generated/torch.addmm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### torch.addmv

<div style="margin-left: 2em">

**原生文档**：[torch.addmv](https://pytorch.org/docs/2.10/generated/torch.addmv.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.addr

<div style="margin-left: 2em">

**原生文档**：[torch.addr](https://pytorch.org/docs/2.10/generated/torch.addr.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.baddbmm

<div style="margin-left: 2em">

**原生文档**：[torch.baddbmm](https://pytorch.org/docs/2.10/generated/torch.baddbmm.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.bmm

<div style="margin-left: 2em">

**原生文档**：[torch.bmm](https://pytorch.org/docs/2.10/generated/torch.bmm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### torch.dot

<div style="margin-left: 2em">

**原生文档**：[torch.dot](https://pytorch.org/docs/2.10/generated/torch.dot.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int32

</div>

### torch.slogdet

<div style="margin-left: 2em">

**原生文档**：[torch.slogdet](https://pytorch.org/docs/2.10/generated/torch.slogdet.html)

**是否支持**：是

**限制与说明**：

- 支持fp32，complex64，complex128
- 可能回退至CPU执行

</div>

### torch.matmul

<div style="margin-left: 2em">

**原生文档**：[torch.matmul](https://pytorch.org/docs/2.10/generated/torch.matmul.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp16，fp32
- 支持Named Tensor
- 输入最大支持6维

</div>

### torch.mm

<div style="margin-left: 2em">

**原生文档**：[torch.mm](https://pytorch.org/docs/2.10/generated/torch.mm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### torch.outer

<div style="margin-left: 2em">

**原生文档**：[torch.outer](https://pytorch.org/docs/2.10/generated/torch.outer.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.qr

<div style="margin-left: 2em">

**原生文档**：[torch.qr](https://pytorch.org/docs/2.10/generated/torch.qr.html)

**是否支持**：是

</div>

### torch.trapezoid

<div style="margin-left: 2em">

**原生文档**：[torch.trapezoid](https://pytorch.org/docs/2.10/generated/torch.trapezoid.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.cumulative_trapezoid

<div style="margin-left: 2em">

**原生文档**：[torch.cumulative_trapezoid](https://pytorch.org/docs/2.10/generated/torch.cumulative_trapezoid.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.vdot

<div style="margin-left: 2em">

**原生文档**：[torch.vdot](https://pytorch.org/docs/2.10/generated/torch.vdot.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.bucketize

<div style="margin-left: 2em">

**原生文档**：[torch.bucketize](https://pytorch.org/docs/2.10/generated/torch.bucketize.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.cartesian_prod

<div style="margin-left: 2em">

**原生文档**：[torch.cartesian_prod](https://pytorch.org/docs/2.10/generated/torch.cartesian_prod.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.mv

<div style="margin-left: 2em">

**原生文档**：[torch.mv](https://pytorch.org/docs/2.10/generated/torch.mv.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch._foreach_sqrt

<div style="margin-left: 2em">

**原生文档**：[torch._foreach_sqrt](https://pytorch.org/docs/2.10/generated/torch._foreach_sqrt.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch._foreach_asin

<div style="margin-left: 2em">

**原生文档**：[torch._foreach_asin](https://pytorch.org/docs/2.10/generated/torch._foreach_asin.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch._foreach_neg_

<div style="margin-left: 2em">

**原生文档**：[torch._foreach_neg_](https://docs.pytorch.org/docs/2.10/generated/torch._foreach_neg_.html#torch._foreach_neg_)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，int8，int32，int64

</div>

### torch.corrcoef

<div style="margin-left: 2em">

**原生文档**：[torch.corrcoef](https://pytorch.org/docs/2.10/generated/torch.corrcoef.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

## Utilities

### torch.compiled_with_cxx11_abi

<div style="margin-left: 2em">

**原生文档**：[torch.compiled_with_cxx11_abi](https://pytorch.org/docs/2.10/generated/torch.compiled_with_cxx11_abi.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.result_type

<div style="margin-left: 2em">

**原生文档**：[torch.result_type](https://pytorch.org/docs/2.10/generated/torch.result_type.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.can_cast

<div style="margin-left: 2em">

**原生文档**：[torch.can_cast](https://pytorch.org/docs/2.10/generated/torch.can_cast.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.promote_types

<div style="margin-left: 2em">

**原生文档**：[torch.promote_types](https://pytorch.org/docs/2.10/generated/torch.promote_types.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.use_deterministic_algorithms

<div style="margin-left: 2em">

**原生文档**：[torch.use_deterministic_algorithms](https://pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html)

**是否支持**：是

**限制与说明**： 同时设置HCCL_DETERMINISTIC和torch.use_deterministic_algorithms时，若HCCL_DETERMINISTIC开启确定性则HCCL接口启用确定性，否则HCCL确定性由torch.use_deterministic_algorithms接口控制

</div>

### torch.are_deterministic_algorithms_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.are_deterministic_algorithms_enabled](https://pytorch.org/docs/2.10/generated/torch.are_deterministic_algorithms_enabled.html)

**是否支持**：是

</div>

### torch.is_deterministic_algorithms_warn_only_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.is_deterministic_algorithms_warn_only_enabled](https://pytorch.org/docs/2.10/generated/torch.is_deterministic_algorithms_warn_only_enabled.html)

**是否支持**：否

</div>

### torch.set_deterministic_debug_mode

<div style="margin-left: 2em">

**原生文档**：[torch.set_deterministic_debug_mode](https://pytorch.org/docs/2.10/generated/torch.set_deterministic_debug_mode.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.get_deterministic_debug_mode

<div style="margin-left: 2em">

**原生文档**：[torch.get_deterministic_debug_mode](https://pytorch.org/docs/2.10/generated/torch.get_deterministic_debug_mode.html)

**是否支持**：是

</div>

### torch.set_float32_matmul_precision

<div style="margin-left: 2em">

**原生文档**：[torch.set_float32_matmul_precision](https://pytorch.org/docs/2.10/generated/torch.set_float32_matmul_precision.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.get_float32_matmul_precision

<div style="margin-left: 2em">

**原生文档**：[torch.get_float32_matmul_precision](https://pytorch.org/docs/2.10/generated/torch.get_float32_matmul_precision.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.set_warn_always

<div style="margin-left: 2em">

**原生文档**：[torch.set_warn_always](https://pytorch.org/docs/2.10/generated/torch.set_warn_always.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.is_warn_always_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.is_warn_always_enabled](https://pytorch.org/docs/2.10/generated/torch.is_warn_always_enabled.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.vmap

<div style="margin-left: 2em">

**原生文档**：[torch.vmap](https://pytorch.org/docs/2.10/generated/torch.vmap.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch._assert

<div style="margin-left: 2em">

**原生文档**：[torch._assert](https://pytorch.org/docs/2.10/generated/torch._assert.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Optimizations

### torch.compile

<div style="margin-left: 2em">

**原生文档**：[torch.compile](https://pytorch.org/docs/2.10/generated/torch.compile.html)

**是否支持**：是

**限制与说明**： backend可支持npugraphs，整体功能与backend="cudagraphs"一致

</div>
