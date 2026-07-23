# torch.Tensor

> [!NOTE]
>
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

## Tensor class reference

### _`class`_ torch.Tensor

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor](https://pytorch.org/docs/2.11/tensors.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">T</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.T](https://pytorch.org/docs/2.11/tensors.html#torch.Tensor.T)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">H</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.H](https://pytorch.org/docs/2.11/tensors.html#torch.Tensor.H)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mT</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.mT](https://pytorch.org/docs/2.11/tensors.html#torch.Tensor.mT)

**是否支持**：是

</div>

> <font size="3">mH</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.mH](https://pytorch.org/docs/2.11/tensors.html#torch.Tensor.mH)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">new_tensor()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.new_tensor](https://pytorch.org/docs/2.11/generated/torch.Tensor.new_tensor.html)

**是否支持**：是

</div>

> <font size="3">new_full()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.new_full](https://pytorch.org/docs/2.11/generated/torch.Tensor.new_full.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持int64

</div>

> <font size="3">new_empty()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.new_empty](https://pytorch.org/docs/2.11/generated/torch.Tensor.new_empty.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">new_ones()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.new_ones](https://pytorch.org/docs/2.11/generated/torch.Tensor.new_ones.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">new_zeros()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.new_zeros](https://pytorch.org/docs/2.11/generated/torch.Tensor.new_zeros.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_cuda](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_cuda.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">is_quantized()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_quantized](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_quantized.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">is_meta()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_meta](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_meta.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">device()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.device](https://pytorch.org/docs/2.11/generated/torch.Tensor.device.html)

**是否支持**：是

</div>

> <font size="3">grad()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.grad](https://pytorch.org/docs/2.11/generated/torch.Tensor.grad.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">ndim()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ndim](https://pytorch.org/docs/2.11/generated/torch.Tensor.ndim.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">real()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.real](https://pytorch.org/docs/2.11/generated/torch.Tensor.real.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">imag()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.imag](https://pytorch.org/docs/2.11/generated/torch.Tensor.imag.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">nbytes()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.nbytes](https://pytorch.org/docs/2.11/generated/torch.Tensor.nbytes.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">itemsize()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.itemsize](https://pytorch.org/docs/2.11/generated/torch.Tensor.itemsize.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">abs()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.abs](https://pytorch.org/docs/2.11/generated/torch.Tensor.abs.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">abs_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.abs_](https://pytorch.org/docs/2.11/generated/torch.Tensor.abs_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">absolute()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.absolute](https://pytorch.org/docs/2.11/generated/torch.Tensor.absolute.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">absolute_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.absolute_](https://pytorch.org/docs/2.11/generated/torch.Tensor.absolute_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">acos()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.acos](https://pytorch.org/docs/2.11/generated/torch.Tensor.acos.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，bool
- 可能回退至CPU执行

</div>

> <font size="3">acos_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.acos_](https://pytorch.org/docs/2.11/generated/torch.Tensor.acos_.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32, fp64
- <term>Atlas A3 训练系列产品</term>额外支持bf16

</div>

> <font size="3">arccos()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arccos](https://pytorch.org/docs/2.11/generated/torch.Tensor.arccos.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">arccos_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arccos_](https://pytorch.org/docs/2.11/generated/torch.Tensor.arccos_.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32, fp64
- <term>Atlas A3 训练系列产品</term>额外支持bf16

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.add](https://pytorch.org/docs/2.11/generated/torch.Tensor.add.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.add_](https://pytorch.org/docs/2.11/generated/torch.Tensor.add_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">addbmm()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addbmm](https://pytorch.org/docs/2.11/generated/torch.Tensor.addbmm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">addbmm_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addbmm_](https://pytorch.org/docs/2.11/generated/torch.Tensor.addbmm_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">addcdiv()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addcdiv](https://pytorch.org/docs/2.11/generated/torch.Tensor.addcdiv.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，int64
- int64类型不支持三个tensor同时广播

</div>

> <font size="3">addcdiv_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addcdiv_](https://pytorch.org/docs/2.11/generated/torch.Tensor.addcdiv_.html)

**是否支持**：是

**限制与说明**：

- <term>Atlas A2 训练系列产品</term>、<term>Atlas A3 训练系列产品</term>：支持bf16，fp16，fp32，fp64
- <term>Atlas 训练系列产品</term>：支持fp16，fp32，fp64
- int64类型不支持三个tensor同时广播

</div>

> <font size="3">addcmul()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addcmul](https://pytorch.org/docs/2.11/generated/torch.Tensor.addcmul.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，int64
- int64类型不支持三个tensor同时广播

</div>

> <font size="3">addcmul_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addcmul_](https://pytorch.org/docs/2.11/generated/torch.Tensor.addcmul_.html)

**是否支持**：是

**限制与说明**：

- <term>Atlas A2 训练系列产品</term>、<term>Atlas A3 训练系列产品</term>：支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64
- <term>Atlas 训练系列产品</term>：支持fp16，fp32，fp64，uint8，int8，int32，int64
- int64类型不支持三个tensor同时广播

</div>

> <font size="3">addmm()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addmm](https://pytorch.org/docs/2.11/generated/torch.Tensor.addmm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">addmm_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addmm_](https://pytorch.org/docs/2.11/generated/torch.Tensor.addmm_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">sspaddmm()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sspaddmm](https://pytorch.org/docs/2.11/generated/torch.Tensor.sspaddmm.html)

**是否支持**：否

</div>

> <font size="3">addmv()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addmv](https://pytorch.org/docs/2.11/generated/torch.Tensor.addmv.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">addmv_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addmv_](https://pytorch.org/docs/2.11/generated/torch.Tensor.addmv_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">addr()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addr](https://pytorch.org/docs/2.11/generated/torch.Tensor.addr.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">addr_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.addr_](https://pytorch.org/docs/2.11/generated/torch.Tensor.addr_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">adjoint()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.adjoint](https://pytorch.org/docs/2.11/generated/torch.Tensor.adjoint.html)

**是否支持**：是

</div>

> <font size="3">allclose()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.allclose](https://pytorch.org/docs/2.11/generated/torch.Tensor.allclose.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">amax()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.amax](https://pytorch.org/docs/2.11/generated/torch.Tensor.amax.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">amin()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.amin](https://pytorch.org/docs/2.11/generated/torch.Tensor.amin.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">aminmax()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.aminmax](https://pytorch.org/docs/2.11/generated/torch.Tensor.aminmax.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">angle()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.angle](https://pytorch.org/docs/2.11/generated/torch.Tensor.angle.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

> <font size="3">apply_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.apply_](https://pytorch.org/docs/2.11/generated/torch.Tensor.apply_.html)

**是否支持**：是

**限制与说明**： 仅CPU支持

</div>

> <font size="3">argmax()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.argmax](https://pytorch.org/docs/2.11/generated/torch.Tensor.argmax.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

> <font size="3">argmin()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.argmin](https://pytorch.org/docs/2.11/generated/torch.Tensor.argmin.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">argsort()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.argsort](https://pytorch.org/docs/2.11/generated/torch.Tensor.argsort.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64
- 针对<term>Ascend 950DT</term>，由于底层实现限制，"stable"仅支持True，设置为False在执行时会自动修改为True

</div>

> <font size="3">argwhere()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.argwhere](https://pytorch.org/docs/2.11/generated/torch.Tensor.argwhere.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">asin()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.asin](https://pytorch.org/docs/2.11/generated/torch.Tensor.asin.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">asin_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.asin_](https://pytorch.org/docs/2.11/generated/torch.Tensor.asin_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">arcsin()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arcsin](https://pytorch.org/docs/2.11/generated/torch.Tensor.arcsin.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">arcsin_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arcsin_](https://pytorch.org/docs/2.11/generated/torch.Tensor.arcsin_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">as_strided()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.as_strided](https://pytorch.org/docs/2.11/generated/torch.Tensor.as_strided.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">atan()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.atan](https://pytorch.org/docs/2.11/generated/torch.Tensor.atan.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">atan_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.atan_](https://pytorch.org/docs/2.11/generated/torch.Tensor.atan_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">arctan()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arctan](https://pytorch.org/docs/2.11/generated/torch.Tensor.arctan.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">arctan_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arctan_](https://pytorch.org/docs/2.11/generated/torch.Tensor.arctan_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">atan2()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.atan2](https://pytorch.org/docs/2.11/generated/torch.Tensor.atan2.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">atan2_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.atan2_](https://pytorch.org/docs/2.11/generated/torch.Tensor.atan2_.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- 可能回退至CPU执行

</div>

> <font size="3">arctan2()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arctan2](https://pytorch.org/docs/2.11/generated/torch.Tensor.arctan2.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">arctan2_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arctan2_](https://pytorch.org/docs/2.11/generated/torch.Tensor.arctan2_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">all()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.all](https://pytorch.org/docs/2.11/generated/torch.Tensor.all.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">any()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.any](https://pytorch.org/docs/2.11/generated/torch.Tensor.any.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">backward()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.backward](https://pytorch.org/docs/2.11/generated/torch.Tensor.backward.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">baddbmm()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.baddbmm](https://pytorch.org/docs/2.11/generated/torch.Tensor.baddbmm.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">baddbmm_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.baddbmm_](https://pytorch.org/docs/2.11/generated/torch.Tensor.baddbmm_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">bernoulli()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bernoulli](https://pytorch.org/docs/2.11/generated/torch.Tensor.bernoulli.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- 可能回退至CPU执行

</div>

> <font size="3">bernoulli_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bernoulli_](https://pytorch.org/docs/2.11/generated/torch.Tensor.bernoulli_.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bfloat16](https://pytorch.org/docs/2.11/generated/torch.Tensor.bfloat16.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">bincount()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bincount](https://pytorch.org/docs/2.11/generated/torch.Tensor.bincount.html)

**是否支持**：是

**限制与说明**：

- 支持uint8，int8，int16，int32，int64
- weights维度需与input维度一致

</div>

> <font size="3">bitwise_not()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bitwise_not](https://pytorch.org/docs/2.11/generated/torch.Tensor.bitwise_not.html)

**是否支持**：是

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">bitwise_not_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bitwise_not_](https://pytorch.org/docs/2.11/generated/torch.Tensor.bitwise_not_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">bitwise_and()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bitwise_and](https://pytorch.org/docs/2.11/generated/torch.Tensor.bitwise_and.html)

**是否支持**：是

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">bitwise_and_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bitwise_and_](https://pytorch.org/docs/2.11/generated/torch.Tensor.bitwise_and_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">bitwise_or()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bitwise_or](https://pytorch.org/docs/2.11/generated/torch.Tensor.bitwise_or.html)

**是否支持**：是

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">bitwise_or_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bitwise_or_](https://pytorch.org/docs/2.11/generated/torch.Tensor.bitwise_or_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">bitwise_xor()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bitwise_xor](https://pytorch.org/docs/2.11/generated/torch.Tensor.bitwise_xor.html)

**是否支持**：是

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">bitwise_xor_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bitwise_xor_](https://pytorch.org/docs/2.11/generated/torch.Tensor.bitwise_xor_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">bmm()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bmm](https://pytorch.org/docs/2.11/generated/torch.Tensor.bmm.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">bool()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.bool](https://pytorch.org/docs/2.11/generated/torch.Tensor.bool.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">byte()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.byte](https://pytorch.org/docs/2.11/generated/torch.Tensor.byte.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">broadcast_to()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.broadcast_to](https://pytorch.org/docs/2.11/generated/torch.Tensor.broadcast_to.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">ceil()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ceil](https://pytorch.org/docs/2.11/generated/torch.Tensor.ceil.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">ceil_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ceil_](https://pytorch.org/docs/2.11/generated/torch.Tensor.ceil_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">char()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.char](https://pytorch.org/docs/2.11/generated/torch.Tensor.char.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">chunk()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.chunk](https://pytorch.org/docs/2.11/generated/torch.Tensor.chunk.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">clamp()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.clamp](https://pytorch.org/docs/2.11/generated/torch.Tensor.clamp.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">clamp_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.clamp_](https://pytorch.org/docs/2.11/generated/torch.Tensor.clamp_.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">clip()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.clip](https://pytorch.org/docs/2.11/generated/torch.Tensor.clip.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

> <font size="3">clip_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.clip_](https://pytorch.org/docs/2.11/generated/torch.Tensor.clip_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.clone](https://pytorch.org/docs/2.11/generated/torch.Tensor.clone.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">contiguous()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.contiguous](https://pytorch.org/docs/2.11/generated/torch.Tensor.contiguous.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">copy_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.copy_](https://pytorch.org/docs/2.11/generated/torch.Tensor.copy_.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- int16不支持5维以上

</div>

> <font size="3">conj()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.conj](https://pytorch.org/docs/2.11/generated/torch.Tensor.conj.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">resolve_conj()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.resolve_conj](https://pytorch.org/docs/2.11/generated/torch.Tensor.resolve_conj.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">resolve_neg()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.resolve_neg](https://pytorch.org/docs/2.11/generated/torch.Tensor.resolve_neg.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">copysign()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.copysign](https://pytorch.org/docs/2.11/generated/torch.Tensor.copysign.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool
- 可能回退至CPU执行

</div>

> <font size="3">cos()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cos](https://pytorch.org/docs/2.11/generated/torch.Tensor.cos.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">cos_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cos_](https://pytorch.org/docs/2.11/generated/torch.Tensor.cos_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，complex64，complex128

</div>

> <font size="3">cosh()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cosh](https://pytorch.org/docs/2.11/generated/torch.Tensor.cosh.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">cosh_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cosh_](https://pytorch.org/docs/2.11/generated/torch.Tensor.cosh_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，complex64，complex128

</div>

> <font size="3">count_nonzero()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.count_nonzero](https://pytorch.org/docs/2.11/generated/torch.Tensor.count_nonzero.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">cov()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cov](https://pytorch.org/docs/2.11/generated/torch.Tensor.cov.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int16，int32，int64

</div>

> <font size="3">acosh()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.acosh](https://pytorch.org/docs/2.11/generated/torch.Tensor.acosh.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 可能回退至CPU执行

</div>

> <font size="3">acosh_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.acosh_](https://pytorch.org/docs/2.11/generated/torch.Tensor.acosh_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，complex64，complex128

</div>

> <font size="3">arccosh()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arccosh](https://pytorch.org/docs/2.11/generated/torch.Tensor.arccosh.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">arccosh_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arccosh_](https://pytorch.org/docs/2.11/generated/torch.Tensor.arccosh_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，complex64，complex128

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cpu](https://pytorch.org/docs/2.11/generated/torch.Tensor.cpu.html)

**是否支持**：是

</div>

> <font size="3">cross()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cross](https://pytorch.org/docs/2.11/generated/torch.Tensor.cross.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，complex64，complex128
- 两个输入的shape要保持一致

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cuda](https://pytorch.org/docs/2.11/generated/torch.Tensor.cuda.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： NPU对应接口为Tensor.npu，其memory_format参数仅支持传入torch.contiguous_format

</div>

> <font size="3">cummax()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cummax](https://pytorch.org/docs/2.11/generated/torch.Tensor.cummax.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，bool
- 可能回退至CPU执行

</div>

> <font size="3">cummin()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cummin](https://pytorch.org/docs/2.11/generated/torch.Tensor.cummin.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">cumsum()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cumsum](https://pytorch.org/docs/2.11/generated/torch.Tensor.cumsum.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 支持Named Tensor

</div>

> <font size="3">cumsum_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cumsum_](https://pytorch.org/docs/2.11/generated/torch.Tensor.cumsum_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，int64，bool

</div>

> <font size="3">chalf()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.chalf](https://pytorch.org/docs/2.11/generated/torch.Tensor.chalf.html)

**是否支持**：否

</div>

> <font size="3">cfloat()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cfloat](https://pytorch.org/docs/2.11/generated/torch.Tensor.cfloat.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">cdouble()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.cdouble](https://pytorch.org/docs/2.11/generated/torch.Tensor.cdouble.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">data_ptr()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.data_ptr](https://pytorch.org/docs/2.11/generated/torch.Tensor.data_ptr.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">deg2rad()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.deg2rad](https://pytorch.org/docs/2.11/generated/torch.Tensor.deg2rad.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">dequantize()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.dequantize](https://pytorch.org/docs/2.11/generated/torch.Tensor.dequantize.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">dense_dim()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.dense_dim](https://pytorch.org/docs/2.11/generated/torch.Tensor.dense_dim.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">detach()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.detach](https://pytorch.org/docs/2.11/generated/torch.Tensor.detach.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">detach_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.detach_](https://pytorch.org/docs/2.11/generated/torch.Tensor.detach_.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">diag()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.diag](https://pytorch.org/docs/2.11/generated/torch.Tensor.diag.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

> <font size="3">diag_embed()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.diag_embed](https://pytorch.org/docs/2.11/generated/torch.Tensor.diag_embed.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">diagflat()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.diagflat](https://pytorch.org/docs/2.11/generated/torch.Tensor.diagflat.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64

</div>

> <font size="3">diagonal()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.diagonal](https://pytorch.org/docs/2.11/generated/torch.Tensor.diagonal.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">diagonal_scatter()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.diagonal_scatter](https://pytorch.org/docs/2.11/generated/torch.Tensor.diagonal_scatter.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">fill_diagonal_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.fill_diagonal_](https://pytorch.org/docs/2.11/generated/torch.Tensor.fill_diagonal_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">diff()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.diff](https://pytorch.org/docs/2.11/generated/torch.Tensor.diff.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">dim()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.dim](https://pytorch.org/docs/2.11/generated/torch.Tensor.dim.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">dim_order()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.dim_order](https://pytorch.org/docs/2.11/generated/torch.Tensor.dim_order.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">dist()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.dist](https://pytorch.org/docs/2.11/generated/torch.Tensor.dist.html)

**是否支持**：是

**限制与说明**： <term>Ascend 950DT</term>：不支持fp64，complex64，complex128

</div>

> <font size="3">div()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.div](https://pytorch.org/docs/2.11/generated/torch.Tensor.div.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">div_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.div_](https://pytorch.org/docs/2.11/generated/torch.Tensor.div_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

> <font size="3">divide()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.divide](https://pytorch.org/docs/2.11/generated/torch.Tensor.divide.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">divide_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.divide_](https://pytorch.org/docs/2.11/generated/torch.Tensor.divide_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">dot()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.dot](https://pytorch.org/docs/2.11/generated/torch.Tensor.dot.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.double](https://pytorch.org/docs/2.11/generated/torch.Tensor.double.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 当前NPU上部分接口暂不支持double类型，出于兼容性考虑默认返回fp32，后续完成支持后将正常返回fp64
- 可能回退至CPU执行

</div>

> <font size="3">dsplit()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.dsplit](https://pytorch.org/docs/2.11/generated/torch.Tensor.dsplit.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">element_size()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.element_size](https://pytorch.org/docs/2.11/generated/torch.Tensor.element_size.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">eq()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.eq](https://pytorch.org/docs/2.11/generated/torch.Tensor.eq.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">eq_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.eq_](https://pytorch.org/docs/2.11/generated/torch.Tensor.eq_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">equal()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.equal](https://pytorch.org/docs/2.11/generated/torch.Tensor.equal.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">erf()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.erf](https://pytorch.org/docs/2.11/generated/torch.Tensor.erf.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int64，bool

</div>

> <font size="3">erf_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.erf_](https://pytorch.org/docs/2.11/generated/torch.Tensor.erf_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">erfc()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.erfc](https://pytorch.org/docs/2.11/generated/torch.Tensor.erfc.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int64，bool

</div>

> <font size="3">erfc_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.erfc_](https://pytorch.org/docs/2.11/generated/torch.Tensor.erfc_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">erfinv()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.erfinv](https://pytorch.org/docs/2.11/generated/torch.Tensor.erfinv.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">erfinv_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.erfinv_](https://pytorch.org/docs/2.11/generated/torch.Tensor.erfinv_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

> <font size="3">exp()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.exp](https://pytorch.org/docs/2.11/generated/torch.Tensor.exp.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int64，bool，complex64，complex128

</div>

> <font size="3">exp_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.exp_](https://pytorch.org/docs/2.11/generated/torch.Tensor.exp_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，complex64，complex128

</div>

> <font size="3">expm1()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.expm1](https://pytorch.org/docs/2.11/generated/torch.Tensor.expm1.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int64，bool

</div>

> <font size="3">expm1_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.expm1_](https://pytorch.org/docs/2.11/generated/torch.Tensor.expm1_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.expand](https://pytorch.org/docs/2.11/generated/torch.Tensor.expand.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">expand_as()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.expand_as](https://pytorch.org/docs/2.11/generated/torch.Tensor.expand_as.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">exponential_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.exponential_](https://pytorch.org/docs/2.11/generated/torch.Tensor.exponential_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

> <font size="3">fix()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.fix](https://pytorch.org/docs/2.11/generated/torch.Tensor.fix.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">fix_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.fix_](https://pytorch.org/docs/2.11/generated/torch.Tensor.fix_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">fill_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.fill_](https://pytorch.org/docs/2.11/generated/torch.Tensor.fill_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">flatten()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.flatten](https://pytorch.org/docs/2.11/generated/torch.Tensor.flatten.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">flip()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.flip](https://pytorch.org/docs/2.11/generated/torch.Tensor.flip.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">fliplr()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.fliplr](https://pytorch.org/docs/2.11/generated/torch.Tensor.fliplr.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">flipud()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.flipud](https://pytorch.org/docs/2.11/generated/torch.Tensor.flipud.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.float](https://pytorch.org/docs/2.11/generated/torch.Tensor.float.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">float_power()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.float_power](https://pytorch.org/docs/2.11/generated/torch.Tensor.float_power.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex128

</div>

> <font size="3">float_power_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.float_power_](https://pytorch.org/docs/2.11/generated/torch.Tensor.float_power_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持double

</div>

> <font size="3">floor()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.floor](https://pytorch.org/docs/2.11/generated/torch.Tensor.floor.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

> <font size="3">floor_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.floor_](https://pytorch.org/docs/2.11/generated/torch.Tensor.floor_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

> <font size="3">floor_divide()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.floor_divide](https://pytorch.org/docs/2.11/generated/torch.Tensor.floor_divide.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">floor_divide_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.floor_divide_](https://pytorch.org/docs/2.11/generated/torch.Tensor.floor_divide_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">fmod()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.fmod](https://pytorch.org/docs/2.11/generated/torch.Tensor.fmod.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int32，int64

</div>

> <font size="3">fmod_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.fmod_](https://pytorch.org/docs/2.11/generated/torch.Tensor.fmod_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int32，int64

</div>

> <font size="3">frac()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.frac](https://pytorch.org/docs/2.11/generated/torch.Tensor.frac.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">frac_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.frac_](https://pytorch.org/docs/2.11/generated/torch.Tensor.frac_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">gather()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.gather](https://pytorch.org/docs/2.11/generated/torch.Tensor.gather.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，int64
- index维度需与input维度一致
- 针对<term>Ascend 950DT</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

> <font size="3">ge()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ge](https://pytorch.org/docs/2.11/generated/torch.Tensor.ge.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">ge_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ge_](https://pytorch.org/docs/2.11/generated/torch.Tensor.ge_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">greater_equal()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.greater_equal](https://pytorch.org/docs/2.11/generated/torch.Tensor.greater_equal.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">greater_equal_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.greater_equal_](https://pytorch.org/docs/2.11/generated/torch.Tensor.greater_equal_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">geometric_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.geometric_](https://pytorch.org/docs/2.11/generated/torch.Tensor.geometric_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">ger()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ger](https://pytorch.org/docs/2.11/generated/torch.Tensor.ger.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">get_device()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.get_device](https://pytorch.org/docs/2.11/generated/torch.Tensor.get_device.html)

**是否支持**：是

</div>

> <font size="3">gt()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.gt](https://pytorch.org/docs/2.11/generated/torch.Tensor.gt.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">gt_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.gt_](https://pytorch.org/docs/2.11/generated/torch.Tensor.gt_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">greater()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.greater](https://pytorch.org/docs/2.11/generated/torch.Tensor.greater.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">greater_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.greater_](https://pytorch.org/docs/2.11/generated/torch.Tensor.greater_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.half](https://pytorch.org/docs/2.11/generated/torch.Tensor.half.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">hardshrink()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.hardshrink](https://pytorch.org/docs/2.11/generated/torch.Tensor.hardshrink.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">heaviside()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.heaviside](https://pytorch.org/docs/2.11/generated/torch.Tensor.heaviside.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">histc()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.histc](https://pytorch.org/docs/2.11/generated/torch.Tensor.histc.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">hsplit()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.hsplit](https://pytorch.org/docs/2.11/generated/torch.Tensor.hsplit.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">index_add_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_add_](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_add_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int64，bool

</div>

> <font size="3">index_add()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_add](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_add.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">index_copy_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_copy_](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_copy_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int16，int32，bool

</div>

> <font size="3">index_copy()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_copy](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_copy.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">index_fill_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_fill_](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_fill_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int32，int64，bool

</div>

> <font size="3">index_fill()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_fill](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_fill.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int32，bool

</div>

> <font size="3">index_put_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_put_](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_put_.html)

**是否支持**：是

**限制与说明**：

- 支持int64
- 针对<term>Ascend 950DT</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

> <font size="3">index_put()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_put](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_put.html)

**是否支持**：是

**限制与说明**：

- 支持int64
- 针对<term>Ascend 950DT</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

> <font size="3">index_reduce_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_reduce_](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_reduce_.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">index_reduce()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_reduce](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_reduce.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">index_select()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.index_select](https://pytorch.org/docs/2.11/generated/torch.Tensor.index_select.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int16，int32，int64，bool

</div>

> <font size="3">indices()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.indices](https://pytorch.org/docs/2.11/generated/torch.Tensor.indices.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">inner()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.inner](https://pytorch.org/docs/2.11/generated/torch.Tensor.inner.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">int()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.int](https://pytorch.org/docs/2.11/generated/torch.Tensor.int.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">int_repr()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.int_repr](https://pytorch.org/docs/2.11/generated/torch.Tensor.int_repr.html)

**是否支持**：否

</div>

> <font size="3">isclose()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.isclose](https://pytorch.org/docs/2.11/generated/torch.Tensor.isclose.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int32，int64，bool

</div>

> <font size="3">isfinite()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.isfinite](https://pytorch.org/docs/2.11/generated/torch.Tensor.isfinite.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">isinf()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.isinf](https://pytorch.org/docs/2.11/generated/torch.Tensor.isinf.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">isposinf()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.isposinf](https://pytorch.org/docs/2.11/generated/torch.Tensor.isposinf.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">isneginf()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.isneginf](https://pytorch.org/docs/2.11/generated/torch.Tensor.isneginf.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">isnan()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.isnan](https://pytorch.org/docs/2.11/generated/torch.Tensor.isnan.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">is_contiguous()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_contiguous](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_contiguous.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">is_complex()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_complex](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_complex.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">is_conj()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_conj](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_conj.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">is_floating_point()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_floating_point](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_floating_point.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">is_inference()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_inference](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_inference.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">is_leaf()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_leaf](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_leaf.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_pinned](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_pinned.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">is_set_to()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_set_to](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_set_to.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">is_shared()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_shared](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_shared.html)

**是否支持**：否

</div>

> <font size="3">is_signed()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_signed](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_signed.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">is_sparse()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.is_sparse](https://pytorch.org/docs/2.11/generated/torch.Tensor.is_sparse.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">isreal()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.isreal](https://pytorch.org/docs/2.11/generated/torch.Tensor.isreal.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">item()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.item](https://pytorch.org/docs/2.11/generated/torch.Tensor.item.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">kthvalue()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.kthvalue](https://pytorch.org/docs/2.11/generated/torch.Tensor.kthvalue.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int32

</div>

> <font size="3">ldexp()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ldexp](https://pytorch.org/docs/2.11/generated/torch.Tensor.ldexp.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">ldexp_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ldexp_](https://pytorch.org/docs/2.11/generated/torch.Tensor.ldexp_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

> <font size="3">le()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.le](https://pytorch.org/docs/2.11/generated/torch.Tensor.le.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">le_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.le_](https://pytorch.org/docs/2.11/generated/torch.Tensor.le_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">less_equal()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.less_equal](https://pytorch.org/docs/2.11/generated/torch.Tensor.less_equal.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">less_equal_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.less_equal_](https://pytorch.org/docs/2.11/generated/torch.Tensor.less_equal_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">lerp()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.lerp](https://pytorch.org/docs/2.11/generated/torch.Tensor.lerp.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">lerp_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.lerp_](https://pytorch.org/docs/2.11/generated/torch.Tensor.lerp_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">log()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.log](https://pytorch.org/docs/2.11/generated/torch.Tensor.log.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int64，bool，complex64，complex128

</div>

> <font size="3">log_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.log_](https://pytorch.org/docs/2.11/generated/torch.Tensor.log_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，complex64，complex128

</div>

> <font size="3">log10()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.log10](https://pytorch.org/docs/2.11/generated/torch.Tensor.log10.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">log10_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.log10_](https://pytorch.org/docs/2.11/generated/torch.Tensor.log10_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，complex64，complex128

</div>

> <font size="3">log1p()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.log1p](https://pytorch.org/docs/2.11/generated/torch.Tensor.log1p.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">log1p_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.log1p_](https://pytorch.org/docs/2.11/generated/torch.Tensor.log1p_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">log2()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.log2](https://pytorch.org/docs/2.11/generated/torch.Tensor.log2.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">log2_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.log2_](https://pytorch.org/docs/2.11/generated/torch.Tensor.log2_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

> <font size="3">logaddexp()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logaddexp](https://pytorch.org/docs/2.11/generated/torch.Tensor.logaddexp.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int16，int32，int64，bool

</div>

> <font size="3">logaddexp2()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logaddexp2](https://pytorch.org/docs/2.11/generated/torch.Tensor.logaddexp2.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">logsumexp()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logsumexp](https://pytorch.org/docs/2.11/generated/torch.Tensor.logsumexp.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">logical_and()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logical_and](https://pytorch.org/docs/2.11/generated/torch.Tensor.logical_and.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">logical_and_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logical_and_](https://pytorch.org/docs/2.11/generated/torch.Tensor.logical_and_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">logical_not()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logical_not](https://pytorch.org/docs/2.11/generated/torch.Tensor.logical_not.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">logical_not_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logical_not_](https://pytorch.org/docs/2.11/generated/torch.Tensor.logical_not_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">logical_or()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logical_or](https://pytorch.org/docs/2.11/generated/torch.Tensor.logical_or.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">logical_or_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logical_or_](https://pytorch.org/docs/2.11/generated/torch.Tensor.logical_or_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">logical_xor()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logical_xor](https://pytorch.org/docs/2.11/generated/torch.Tensor.logical_xor.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 可能回退至CPU执行

</div>

> <font size="3">logical_xor_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logical_xor_](https://pytorch.org/docs/2.11/generated/torch.Tensor.logical_xor_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">logit()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logit](https://pytorch.org/docs/2.11/generated/torch.Tensor.logit.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- eps取值大于1时输出为nan，eps取值为1时输出为inf

</div>

> <font size="3">logit_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.logit_](https://pytorch.org/docs/2.11/generated/torch.Tensor.logit_.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- eps取值大于1时输出为nan，eps取值为1时输出为inf

</div>

> <font size="3">long()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.long](https://pytorch.org/docs/2.11/generated/torch.Tensor.long.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">lt()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.lt](https://pytorch.org/docs/2.11/generated/torch.Tensor.lt.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">lt_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.lt_](https://pytorch.org/docs/2.11/generated/torch.Tensor.lt_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">less()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.less](https://pytorch.org/docs/2.11/generated/torch.Tensor.less.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">less_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.less_](https://pytorch.org/docs/2.11/generated/torch.Tensor.less_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">as_subclass()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.as_subclass](https://pytorch.org/docs/2.11/generated/torch.Tensor.as_subclass.html)

**是否支持**：是

</div>

> <font size="3">map_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.map_](https://pytorch.org/docs/2.11/generated/torch.Tensor.map_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 仅CPU支持

</div>

> <font size="3">masked_scatter_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.masked_scatter_](https://pytorch.org/docs/2.11/generated/torch.Tensor.masked_scatter_.html)

**是否支持**：是

**限制与说明**： 支持fp32，int64，bool

</div>

> <font size="3">masked_scatter()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.masked_scatter](https://pytorch.org/docs/2.11/generated/torch.Tensor.masked_scatter.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">masked_fill_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.masked_fill_](https://pytorch.org/docs/2.11/generated/torch.Tensor.masked_fill_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int8，int32，int64，bool

</div>

> <font size="3">masked_fill()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.masked_fill](https://pytorch.org/docs/2.11/generated/torch.Tensor.masked_fill.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int64，bool

</div>

> <font size="3">masked_select()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.masked_select](https://pytorch.org/docs/2.11/generated/torch.Tensor.masked_select.html)

**是否支持**：是

**限制与说明**： 支持fp32，bool

</div>

> <font size="3">matmul()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.matmul](https://pytorch.org/docs/2.11/generated/torch.Tensor.matmul.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 支持Named Tensor

</div>

> <font size="3">matrix_power()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.matrix_power](https://pytorch.org/docs/2.11/generated/torch.Tensor.matrix_power.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">max()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.max](https://pytorch.org/docs/2.11/generated/torch.Tensor.max.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int64，bool

</div>

> <font size="3">maximum()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.maximum](https://pytorch.org/docs/2.11/generated/torch.Tensor.maximum.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.mean](https://pytorch.org/docs/2.11/generated/torch.Tensor.mean.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，complex64，complex128

</div>

> <font size="3">nanmean()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.nanmean](https://pytorch.org/docs/2.11/generated/torch.Tensor.nanmean.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">median()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.median](https://pytorch.org/docs/2.11/generated/torch.Tensor.median.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64
- input为bf16时，dim不取input轴值为1的维度

</div>

> <font size="3">min()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.min](https://pytorch.org/docs/2.11/generated/torch.Tensor.min.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int64，bool

</div>

> <font size="3">minimum()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.minimum](https://pytorch.org/docs/2.11/generated/torch.Tensor.minimum.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">mm()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.mm](https://pytorch.org/docs/2.11/generated/torch.Tensor.mm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">smm()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.smm](https://pytorch.org/docs/2.11/generated/torch.Tensor.smm.html)

**是否支持**：否

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.mode](https://pytorch.org/docs/2.11/generated/torch.Tensor.mode.html)

**是否支持**：否

</div>

> <font size="3">movedim()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.movedim](https://pytorch.org/docs/2.11/generated/torch.Tensor.movedim.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">moveaxis()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.moveaxis](https://pytorch.org/docs/2.11/generated/torch.Tensor.moveaxis.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">msort()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.msort](https://pytorch.org/docs/2.11/generated/torch.Tensor.msort.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">mul()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.mul](https://pytorch.org/docs/2.11/generated/torch.Tensor.mul.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">mul_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.mul_](https://pytorch.org/docs/2.11/generated/torch.Tensor.mul_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">multiply()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.multiply](https://pytorch.org/docs/2.11/generated/torch.Tensor.multiply.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">multiply_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.multiply_](https://pytorch.org/docs/2.11/generated/torch.Tensor.multiply_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">multinomial()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.multinomial](https://pytorch.org/docs/2.11/generated/torch.Tensor.multinomial.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">nansum()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.nansum](https://pytorch.org/docs/2.11/generated/torch.Tensor.nansum.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool
- <term>Ascend 950DT</term>：不支持uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">narrow()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.narrow](https://pytorch.org/docs/2.11/generated/torch.Tensor.narrow.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">narrow_copy()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.narrow_copy](https://pytorch.org/docs/2.11/generated/torch.Tensor.narrow_copy.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">ndimension()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ndimension](https://pytorch.org/docs/2.11/generated/torch.Tensor.ndimension.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">nan_to_num()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.nan_to_num](https://pytorch.org/docs/2.11/generated/torch.Tensor.nan_to_num.html)

**是否支持**：是

**限制与说明**： <term>Ascend 950DT</term>：不支持fp64，complex64，complex128

</div>

> <font size="3">nan_to_num_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.nan_to_num_](https://pytorch.org/docs/2.11/generated/torch.Tensor.nan_to_num_.html)

**是否支持**：是

</div>

> <font size="3">ne()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ne](https://pytorch.org/docs/2.11/generated/torch.Tensor.ne.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">ne_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ne_](https://pytorch.org/docs/2.11/generated/torch.Tensor.ne_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">nextafter_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.nextafter_](https://pytorch.org/docs/2.11/generated/torch.Tensor.nextafter_.html)

**是否支持**：是

**限制与说明**： 回退至CPU执行

</div>

> <font size="3">not_equal()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.not_equal](https://pytorch.org/docs/2.11/generated/torch.Tensor.not_equal.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 可能回退至CPU执行

</div>

> <font size="3">not_equal_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.not_equal_](https://pytorch.org/docs/2.11/generated/torch.Tensor.not_equal_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">neg()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.neg](https://pytorch.org/docs/2.11/generated/torch.Tensor.neg.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int8，int32，int64，complex64，complex128

</div>

> <font size="3">neg_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.neg_](https://pytorch.org/docs/2.11/generated/torch.Tensor.neg_.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，int8，int32，int64，complex64，complex128
- 可能回退至CPU执行

</div>

> <font size="3">negative()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.negative](https://pytorch.org/docs/2.11/generated/torch.Tensor.negative.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，int8，int32，int64，complex64，complex128

</div>

> <font size="3">negative_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.negative_](https://pytorch.org/docs/2.11/generated/torch.Tensor.negative_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int8，int32，int64，complex64，complex128

</div>

> <font size="3">nelement()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.nelement](https://pytorch.org/docs/2.11/generated/torch.Tensor.nelement.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">nonzero()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.nonzero](https://pytorch.org/docs/2.11/generated/torch.Tensor.nonzero.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，bool
- 不支持nan场景

</div>

> <font size="3">norm()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.norm](https://pytorch.org/docs/2.11/generated/torch.Tensor.norm.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64
- <term>Ascend 950DT</term>：不支持fp64

</div>

> <font size="3">normal_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.normal_](https://pytorch.org/docs/2.11/generated/torch.Tensor.normal_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持bf16，fp16，fp32
- 可能回退至CPU执行

</div>

> <font size="3">numel()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.numel](https://pytorch.org/docs/2.11/generated/torch.Tensor.numel.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">numpy()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.numpy](https://pytorch.org/docs/2.11/generated/torch.Tensor.numpy.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">outer()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.outer](https://pytorch.org/docs/2.11/generated/torch.Tensor.outer.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">permute()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.permute](https://pytorch.org/docs/2.11/generated/torch.Tensor.permute.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">positive()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.positive](https://pytorch.org/docs/2.11/generated/torch.Tensor.positive.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，complex64，complex128

</div>

> <font size="3">pow()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.pow](https://pytorch.org/docs/2.11/generated/torch.Tensor.pow.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，int16，int64

</div>

> <font size="3">pow_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.pow_](https://pytorch.org/docs/2.11/generated/torch.Tensor.pow_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，int64

</div>

> <font size="3">prod()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.prod](https://pytorch.org/docs/2.11/generated/torch.Tensor.prod.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">put_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.put_](https://pytorch.org/docs/2.11/generated/torch.Tensor.put_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，complex64，complex128

</div>

> <font size="3">qscheme()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.qscheme](https://pytorch.org/docs/2.11/generated/torch.Tensor.qscheme.html)

**是否支持**：否

</div>

> <font size="3">quantile()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.quantile](https://pytorch.org/docs/2.11/generated/torch.Tensor.quantile.html)

**是否支持**：是

</div>

> <font size="3">rad2deg()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.rad2deg](https://pytorch.org/docs/2.11/generated/torch.Tensor.rad2deg.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">random_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.random_](https://pytorch.org/docs/2.11/generated/torch.Tensor.random_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">ravel()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.ravel](https://pytorch.org/docs/2.11/generated/torch.Tensor.ravel.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">reciprocal()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.reciprocal](https://pytorch.org/docs/2.11/generated/torch.Tensor.reciprocal.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">reciprocal_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.reciprocal_](https://pytorch.org/docs/2.11/generated/torch.Tensor.reciprocal_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

> <font size="3">record_stream()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.record_stream](https://pytorch.org/docs/2.11/generated/torch.Tensor.record_stream.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.register_hook](https://pytorch.org/docs/2.11/generated/torch.Tensor.register_hook.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">register_post_accumulate_grad_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.register_post_accumulate_grad_hook](https://pytorch.org/docs/2.11/generated/torch.Tensor.register_post_accumulate_grad_hook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">remainder()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.remainder](https://pytorch.org/docs/2.11/generated/torch.Tensor.remainder.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int32，int64

</div>

> <font size="3">remainder_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.remainder_](https://pytorch.org/docs/2.11/generated/torch.Tensor.remainder_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，int32，int64

</div>

> <font size="3">repeat()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.repeat](https://pytorch.org/docs/2.11/generated/torch.Tensor.repeat.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">repeat_interleave()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.repeat_interleave](https://pytorch.org/docs/2.11/generated/torch.Tensor.repeat_interleave.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，int16，int32，bool
- 输入张量在重复后得到输出，输出中元素个数需小于$2^{22}$

</div>

> <font size="3">requires_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.requires_grad](https://pytorch.org/docs/2.11/generated/torch.Tensor.requires_grad.html)

**是否支持**：是

</div>

> <font size="3">requires_grad_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.requires_grad_](https://pytorch.org/docs/2.11/generated/torch.Tensor.requires_grad_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">reshape()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.reshape](https://pytorch.org/docs/2.11/generated/torch.Tensor.reshape.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">reshape_as()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.reshape_as](https://pytorch.org/docs/2.11/generated/torch.Tensor.reshape_as.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">resize_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.resize_](https://pytorch.org/docs/2.11/generated/torch.Tensor.resize_.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128
- memory_format仅支持torch.contiguous_format和torch.preserve_format

</div>

> <font size="3">resize_as_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.resize_as_](https://pytorch.org/docs/2.11/generated/torch.Tensor.resize_as_.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128
- memory_format仅支持torch.contiguous_format和torch.preserve_format

</div>

> <font size="3">retain_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.retain_grad](https://pytorch.org/docs/2.11/generated/torch.Tensor.retain_grad.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">retains_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.retains_grad](https://pytorch.org/docs/2.11/generated/torch.Tensor.retains_grad.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">roll()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.roll](https://pytorch.org/docs/2.11/generated/torch.Tensor.roll.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int32，int64，bool

</div>

> <font size="3">rot90()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.rot90](https://pytorch.org/docs/2.11/generated/torch.Tensor.rot90.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">round()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.round](https://pytorch.org/docs/2.11/generated/torch.Tensor.round.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">round_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.round_](https://pytorch.org/docs/2.11/generated/torch.Tensor.round_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">rsqrt()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.rsqrt](https://pytorch.org/docs/2.11/generated/torch.Tensor.rsqrt.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">rsqrt_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.rsqrt_](https://pytorch.org/docs/2.11/generated/torch.Tensor.rsqrt_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

> <font size="3">scatter()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.scatter](https://pytorch.org/docs/2.11/generated/torch.Tensor.scatter.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，int16，int32，bool
- 针对<term>Ascend 950DT</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

> <font size="3">scatter_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.scatter_](https://pytorch.org/docs/2.11/generated/torch.Tensor.scatter_.html)

**是否支持**：是

**限制与说明**：

- tensor、index、src参数不能为空且不能为scalar
- 可能回退至CPU执行
- 针对<term>Ascend 950DT</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

> <font size="3">scatter_add_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.scatter_add_](https://pytorch.org/docs/2.11/generated/torch.Tensor.scatter_add_.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 针对<term>Ascend 950DT</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

> <font size="3">scatter_add()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.scatter_add](https://pytorch.org/docs/2.11/generated/torch.Tensor.scatter_add.html)

**是否支持**：是

**限制与说明**：

- 支持fp32
- 针对<term>Ascend 950DT</term>，由于硬件差异，在索引存在重复的情况下，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>

> <font size="3">scatter_reduce()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.scatter_reduce](https://pytorch.org/docs/2.11/generated/torch.Tensor.scatter_reduce.html)

**是否支持**：是

**限制与说明**：

- 支持fp32，int64
- 可能回退至CPU执行

</div>

> <font size="3">select()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.select](https://pytorch.org/docs/2.11/generated/torch.Tensor.select.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">select_scatter()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.select_scatter](https://pytorch.org/docs/2.11/generated/torch.Tensor.select_scatter.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，bool
- 可能回退至CPU执行

</div>

> <font size="3">set_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.set_](https://pytorch.org/docs/2.11/generated/torch.Tensor.set_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">share_memory_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.share_memory_](https://pytorch.org/docs/2.11/generated/torch.Tensor.share_memory_.html)

**是否支持**：否

</div>

> <font size="3">short()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.short](https://pytorch.org/docs/2.11/generated/torch.Tensor.short.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">sigmoid_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sigmoid_](https://pytorch.org/docs/2.11/generated/torch.Tensor.sigmoid_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，complex64，complex128

</div>

> <font size="3">sign()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sign](https://pytorch.org/docs/2.11/generated/torch.Tensor.sign.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，int32，int64，bool

</div>

> <font size="3">sign_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sign_](https://pytorch.org/docs/2.11/generated/torch.Tensor.sign_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int32，int64，bool

</div>

> <font size="3">sgn()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sgn](https://pytorch.org/docs/2.11/generated/torch.Tensor.sgn.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int32，int64，bool，complex64，complex128

</div>

> <font size="3">sgn_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sgn_](https://pytorch.org/docs/2.11/generated/torch.Tensor.sgn_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，int32，int64，bool

</div>

> <font size="3">sin()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sin](https://pytorch.org/docs/2.11/generated/torch.Tensor.sin.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">sin_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sin_](https://pytorch.org/docs/2.11/generated/torch.Tensor.sin_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，complex64，complex128

</div>

> <font size="3">sinh()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sinh](https://pytorch.org/docs/2.11/generated/torch.Tensor.sinh.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64

</div>

> <font size="3">sinh_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sinh_](https://pytorch.org/docs/2.11/generated/torch.Tensor.sinh_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64

</div>

> <font size="3">asinh()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.asinh](https://pytorch.org/docs/2.11/generated/torch.Tensor.asinh.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">asinh_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.asinh_](https://pytorch.org/docs/2.11/generated/torch.Tensor.asinh_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

> <font size="3">arcsinh()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arcsinh](https://pytorch.org/docs/2.11/generated/torch.Tensor.arcsinh.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">arcsinh_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arcsinh_](https://pytorch.org/docs/2.11/generated/torch.Tensor.arcsinh_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

> <font size="3">shape()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.shape](https://pytorch.org/docs/2.11/generated/torch.Tensor.shape.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">size()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.size](https://pytorch.org/docs/2.11/generated/torch.Tensor.size.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">slogdet()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.slogdet](https://pytorch.org/docs/2.11/generated/torch.Tensor.slogdet.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32，complex64，complex128

</div>

> <font size="3">slice_scatter()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.slice_scatter](https://pytorch.org/docs/2.11/generated/torch.Tensor.slice_scatter.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">softmax()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.softmax](https://pytorch.org/docs/2.11/generated/torch.Tensor.softmax.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64

</div>

> <font size="3">sort()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sort](https://pytorch.org/docs/2.11/generated/torch.Tensor.sort.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64
- 针对<term>Ascend 950DT</term>，由于底层实现限制，"stable"仅支持True，设置为False在执行时会自动修改为True

</div>

> <font size="3">split()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.split](https://pytorch.org/docs/2.11/generated/torch.Tensor.split.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">sparse_mask()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sparse_mask](https://pytorch.org/docs/2.11/generated/torch.Tensor.sparse_mask.html)

**是否支持**：否

</div>

> <font size="3">sparse_dim()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sparse_dim](https://pytorch.org/docs/2.11/generated/torch.Tensor.sparse_dim.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sqrt()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sqrt](https://pytorch.org/docs/2.11/generated/torch.Tensor.sqrt.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">sqrt_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sqrt_](https://pytorch.org/docs/2.11/generated/torch.Tensor.sqrt_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，complex64，complex128

</div>

> <font size="3">square()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.square](https://pytorch.org/docs/2.11/generated/torch.Tensor.square.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">square_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.square_](https://pytorch.org/docs/2.11/generated/torch.Tensor.square_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，complex64，complex128

</div>

> <font size="3">squeeze()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.squeeze](https://pytorch.org/docs/2.11/generated/torch.Tensor.squeeze.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">squeeze_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.squeeze_](https://pytorch.org/docs/2.11/generated/torch.Tensor.squeeze_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">std()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.std](https://pytorch.org/docs/2.11/generated/torch.Tensor.std.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- input不支持标量tensor
- correction参数值不能超过int32的最大值

</div>

> <font size="3">storage()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.storage](https://pytorch.org/docs/2.11/generated/torch.Tensor.storage.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">untyped_storage()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.untyped_storage](https://pytorch.org/docs/2.11/generated/torch.Tensor.untyped_storage.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">storage_offset()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.storage_offset](https://pytorch.org/docs/2.11/generated/torch.Tensor.storage_offset.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">storage_type()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.storage_type](https://pytorch.org/docs/2.11/generated/torch.Tensor.storage_type.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">stride()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.stride](https://pytorch.org/docs/2.11/generated/torch.Tensor.stride.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">sub()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sub](https://pytorch.org/docs/2.11/generated/torch.Tensor.sub.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

> <font size="3">sub_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sub_](https://pytorch.org/docs/2.11/generated/torch.Tensor.sub_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，complex64，complex128

</div>

> <font size="3">subtract_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.subtract_](https://pytorch.org/docs/2.11/generated/torch.Tensor.subtract_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64

</div>

> <font size="3">sum()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sum](https://pytorch.org/docs/2.11/generated/torch.Tensor.sum.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int32

</div>

> <font size="3">sum_to_size()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.sum_to_size](https://pytorch.org/docs/2.11/generated/torch.Tensor.sum_to_size.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">swapaxes()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.swapaxes](https://pytorch.org/docs/2.11/generated/torch.Tensor.swapaxes.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">swapdims()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.swapdims](https://pytorch.org/docs/2.11/generated/torch.Tensor.swapdims.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">t()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.t](https://pytorch.org/docs/2.11/generated/torch.Tensor.t.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，complex64，complex128

</div>

> <font size="3">t_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.t_](https://pytorch.org/docs/2.11/generated/torch.Tensor.t_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64

</div>

> <font size="3">tensor_split()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.tensor_split](https://pytorch.org/docs/2.11/generated/torch.Tensor.tensor_split.html)

**是否支持**：是

**限制与说明**： 仅CPU支持

</div>

> <font size="3">tile()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.tile](https://pytorch.org/docs/2.11/generated/torch.Tensor.tile.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 若入参dims的长度小于Tensor.shape的长度，则会在dims前自动补全1，使其长度与 Tensor.shape对齐。补全后的dims，需要满足如下限制：
- - 当需要对第一根轴进行重复时，最多允许同时对4个维度进行重复操作（即dims中大于1的元素个数 ≤ 4），例如：不支持Tensor.tile([2, 3, 4, 5, 6]) ，支持Tensor.tile([2, 3, 1, 5, 6])
- - 当不需要对第一根轴进行重复时，最多允许同时对3个维度进行重复操作（即dims中大于1的元素个数 ≤ 3），例如：不支持Tensor.tile([1, 3, 4, 5, 6]) ，支持Tensor.tile([1, 3, 1, 5, 6])
- - 若执行反向计算，Tensor的维度数加上dims中大于1的元素个数之和不得超过8

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to](https://pytorch.org/docs/2.11/generated/torch.Tensor.to.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128
- 当前NPU设备仅支持设置memory_format为torch.contiguous_format或torch.preserve_format
- <term>Atlas 推理系列产品</term>不支持跨NPU拷贝

</div>

> <font size="3">to_mkldnn()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_mkldnn](https://pytorch.org/docs/2.11/generated/torch.Tensor.to_mkldnn.html)

**是否支持**：否

</div>

> <font size="3">take()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.take](https://pytorch.org/docs/2.11/generated/torch.Tensor.take.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，int16，int32，bool

</div>

> <font size="3">take_along_dim()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.take_along_dim](https://pytorch.org/docs/2.11/generated/torch.Tensor.take_along_dim.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int16，int32，int64，bool

</div>

> <font size="3">tan()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.tan](https://pytorch.org/docs/2.11/generated/torch.Tensor.tan.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128取值范围[-65504,65504]

</div>

> <font size="3">tan_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.tan_](https://pytorch.org/docs/2.11/generated/torch.Tensor.tan_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

> <font size="3">tanh()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.tanh](https://pytorch.org/docs/2.11/generated/torch.Tensor.tanh.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64
- <term>Ascend 950DT</term>：不支持fp64

</div>

> <font size="3">tanh_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.tanh_](https://pytorch.org/docs/2.11/generated/torch.Tensor.tanh_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">atanh()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.atanh](https://pytorch.org/docs/2.11/generated/torch.Tensor.atanh.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">atanh_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.atanh_](https://pytorch.org/docs/2.11/generated/torch.Tensor.atanh_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

> <font size="3">arctanh()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arctanh](https://pytorch.org/docs/2.11/generated/torch.Tensor.arctanh.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">arctanh_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.arctanh_](https://pytorch.org/docs/2.11/generated/torch.Tensor.arctanh_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，complex64，complex128

</div>

> <font size="3">tolist()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.tolist](https://pytorch.org/docs/2.11/generated/torch.Tensor.tolist.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

> <font size="3">topk()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.topk](https://pytorch.org/docs/2.11/generated/torch.Tensor.topk.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64
- 由于硬件差异，npu topk索引结果与GPU/CPU不一致。当前NPU仅支持返回sorted为true的计算结果
- 不支持标量tensor

</div>

> <font size="3">to_dense()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_dense](https://pytorch.org/docs/2.11/generated/torch.Tensor.to_dense.html)

**是否支持**：否

</div>

> <font size="3">to_sparse()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_sparse](https://pytorch.org/docs/2.11/generated/torch.Tensor.to_sparse.html)

**是否支持**：否

</div>

> <font size="3">to_sparse_csr()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_sparse_csr](https://pytorch.org/docs/2.11/generated/torch.Tensor.to_sparse_csr.html)

**是否支持**：否

</div>

> <font size="3">to_sparse_csc()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_sparse_csc](https://pytorch.org/docs/2.11/generated/torch.Tensor.to_sparse_csc.html)

**是否支持**：否

</div>

> <font size="3">to_sparse_bsr()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_sparse_bsr](https://pytorch.org/docs/2.11/generated/torch.Tensor.to_sparse_bsr.html)

**是否支持**：否

</div>

> <font size="3">to_sparse_bsc()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.to_sparse_bsc](https://pytorch.org/docs/2.11/generated/torch.Tensor.to_sparse_bsc.html)

**是否支持**：否

</div>

> <font size="3">transpose()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.transpose](https://pytorch.org/docs/2.11/generated/torch.Tensor.transpose.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">transpose_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.transpose_](https://pytorch.org/docs/2.11/generated/torch.Tensor.transpose_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">tril()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.tril](https://pytorch.org/docs/2.11/generated/torch.Tensor.tril.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">tril_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.tril_](https://pytorch.org/docs/2.11/generated/torch.Tensor.tril_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">triu()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.triu](https://pytorch.org/docs/2.11/generated/torch.Tensor.triu.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">triu_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.triu_](https://pytorch.org/docs/2.11/generated/torch.Tensor.triu_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">true_divide()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.true_divide](https://pytorch.org/docs/2.11/generated/torch.Tensor.true_divide.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">true_divide_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.true_divide_](https://pytorch.org/docs/2.11/generated/torch.Tensor.true_divide_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">trunc()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.trunc](https://pytorch.org/docs/2.11/generated/torch.Tensor.trunc.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">trunc_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.trunc_](https://pytorch.org/docs/2.11/generated/torch.Tensor.trunc_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.type](https://pytorch.org/docs/2.11/generated/torch.Tensor.type.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">type_as()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.type_as](https://pytorch.org/docs/2.11/generated/torch.Tensor.type_as.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">unbind()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.unbind](https://pytorch.org/docs/2.11/generated/torch.Tensor.unbind.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">unflatten()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.unflatten](https://pytorch.org/docs/2.11/generated/torch.Tensor.unflatten.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">unfold()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.unfold](https://pytorch.org/docs/2.11/generated/torch.Tensor.unfold.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">uniform_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.uniform_](https://pytorch.org/docs/2.11/generated/torch.Tensor.uniform_.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64
- 遵循PyTorch社区规范，不再支持对bool类型数据进行处理。针对存量bool类型数据可以通过如下方案进行替换：如果需要输出全True，可以采用Tensor.bernoulli_(p=1.0)。如果需要输出均匀分布的bool类型，则采用Tensor.bernoulli_(p=0.5)

</div>

> <font size="3">unique()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.unique](https://pytorch.org/docs/2.11/generated/torch.Tensor.unique.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 在输入包含0的情况下，输出中可能会包含正0和负0，而非只输出一个0

</div>

> <font size="3">unique_consecutive()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.unique_consecutive](https://pytorch.org/docs/2.11/generated/torch.Tensor.unique_consecutive.html)

**是否支持**：否

</div>

> <font size="3">unsqueeze()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.unsqueeze](https://pytorch.org/docs/2.11/generated/torch.Tensor.unsqueeze.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">unsqueeze_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.unsqueeze_](https://pytorch.org/docs/2.11/generated/torch.Tensor.unsqueeze_.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">values()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.values](https://pytorch.org/docs/2.11/generated/torch.Tensor.values.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 依赖稀疏tensor

</div>

> <font size="3">var()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.var](https://pytorch.org/docs/2.11/generated/torch.Tensor.var.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- correction参数值不能超过int32的最大值

</div>

> <font size="3">view()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.view](https://pytorch.org/docs/2.11/generated/torch.Tensor.view.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">view_as()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.view_as](https://pytorch.org/docs/2.11/generated/torch.Tensor.view_as.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">vsplit()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.vsplit](https://pytorch.org/docs/2.11/generated/torch.Tensor.vsplit.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">where()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.where](https://pytorch.org/docs/2.11/generated/torch.Tensor.where.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

> <font size="3">xlogy()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.xlogy](https://pytorch.org/docs/2.11/generated/torch.Tensor.xlogy.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">xlogy_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.xlogy_](https://pytorch.org/docs/2.11/generated/torch.Tensor.xlogy_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

> <font size="3">zero_()</font>

<div style="margin-left: 2em">

**原生文档**：[Tensor.zero_](https://pytorch.org/docs/2.11/generated/torch.Tensor.zero_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

</div>
