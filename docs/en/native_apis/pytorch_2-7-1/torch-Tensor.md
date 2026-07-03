# torch.Tensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:11:02.553Z pushedAt=2026-06-15T02:04:36.618Z -->

> [!NOTE]
> If the "Support" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Support|Restrictions and Notes|
|--|--|--|
|[torch.Tensor](https://pytorch.org/docs/2.7/tensors.html)|Yes|-|
|[Tensor.T](https://pytorch.org/docs/2.7/tensors.html#torch.Tensor.T)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.H](https://pytorch.org/docs/2.7/tensors.html#torch.Tensor.H)|Yes|-|
|[Tensor.mT](https://pytorch.org/docs/2.7/tensors.html#torch.Tensor.mT)|Yes|-|
|[Tensor.mH](https://pytorch.org/docs/2.7/tensors.html#torch.Tensor.mH)|Yes|-|
|[Tensor.new_tensor](https://pytorch.org/docs/2.7/generated/torch.Tensor.new_tensor.html)|Yes|-|
|[Tensor.new_full](https://pytorch.org/docs/2.7/generated/torch.Tensor.new_full.html)|Yes|supports int64|
|[Tensor.new_empty](https://pytorch.org/docs/2.7/generated/torch.Tensor.new_empty.html)|Yes|supports fp32|
|[Tensor.new_ones](https://pytorch.org/docs/2.7/generated/torch.Tensor.new_ones.html)|Yes|supports fp32|
|[Tensor.new_zeros](https://pytorch.org/docs/2.7/generated/torch.Tensor.new_zeros.html)|Yes|supports fp32|
|[Tensor.is_cuda](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_cuda.html)|Yes|-|
|[Tensor.is_quantized](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_quantized.html)|Yes|-|
|[Tensor.is_meta](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_meta.html)|Yes|-|
|[Tensor.device](https://pytorch.org/docs/2.7/generated/torch.Tensor.device.html)|Yes|-|
|[Tensor.grad](https://pytorch.org/docs/2.7/generated/torch.Tensor.grad.html)|Yes|supports fp32|
|[Tensor.ndim](https://pytorch.org/docs/2.7/generated/torch.Tensor.ndim.html)|Yes|supports fp32|
|[Tensor.real](https://pytorch.org/docs/2.7/generated/torch.Tensor.real.html)|Yes|-|
|[Tensor.imag](https://pytorch.org/docs/2.7/generated/torch.Tensor.imag.html)|Yes|-|
|[Tensor.nbytes](https://pytorch.org/docs/2.7/generated/torch.Tensor.nbytes.html)|Yes|-|
|[Tensor.itemsize](https://pytorch.org/docs/2.7/generated/torch.Tensor.itemsize.html)|Yes|-|
|[Tensor.abs](https://pytorch.org/docs/2.7/generated/torch.Tensor.abs.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.abs_](https://pytorch.org/docs/2.7/generated/torch.Tensor.abs_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.absolute](https://pytorch.org/docs/2.7/generated/torch.Tensor.absolute.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.absolute_](https://pytorch.org/docs/2.7/generated/torch.Tensor.absolute_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.acos](https://pytorch.org/docs/2.7/generated/torch.Tensor.acos.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>may fall back to CPU|
|[Tensor.acos_](https://pytorch.org/docs/2.7/generated/torch.Tensor.acos_.html)|Yes|Supports fp16, fp32|
|[Tensor.arccos](https://pytorch.org/docs/2.7/generated/torch.Tensor.arccos.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.arccos_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arccos_.html)|Yes|Supports fp16, fp32|
|[Tensor.add](https://pytorch.org/docs/2.7/generated/torch.Tensor.add.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.add_](https://pytorch.org/docs/2.7/generated/torch.Tensor.add_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.addbmm](https://pytorch.org/docs/2.7/generated/torch.Tensor.addbmm.html)|Yes|Supports fp16, fp32|
|[Tensor.addbmm_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addbmm_.html)|Yes|Supports fp16, fp32|
|[Tensor.addcdiv](https://pytorch.org/docs/2.7/generated/torch.Tensor.addcdiv.html)|Yes|Supports bf16, fp16, fp32, int64<br>int64 type does not support broadcasting three tensors simultaneously|
|[Tensor.addcdiv_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addcdiv_.html)|Yes|<term>Atlas A2 training series</term>/<term>Atlas A3 training series</term>: Supports bf16, fp16, fp32, fp64<br><term>Atlas training series</term>: Supports fp16, fp32, fp64<br>int64 type does not support broadcasting three tensors simultaneously|
|[Tensor.addcmul](https://pytorch.org/docs/2.7/generated/torch.Tensor.addcmul.html)|Yes|Supports fp16, fp32, int64<br>int64 type does not support broadcasting three tensors simultaneously|
|[Tensor.addcmul_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addcmul_.html)|Yes|<term>Atlas A2 training series</term>/<term>Atlas A3 training series</term>: Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64<br><term>Atlas training series</term>: Supports fp16, fp32, fp64, uint8, int8, int32, int64<br>int64 type does not support broadcasting three tensors simultaneously|
|[Tensor.addmm](https://pytorch.org/docs/2.7/generated/torch.Tensor.addmm.html)|Yes|Supports bf16, fp16, fp32|
|[Tensor.addmm_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addmm_.html)|Yes|Supports fp16, fp32|
|[Tensor.sspaddmm](https://pytorch.org/docs/2.7/generated/torch.Tensor.sspaddmm.html)|No|-|
|[Tensor.addmv](https://pytorch.org/docs/2.7/generated/torch.Tensor.addmv.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.addmv_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addmv_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.addr](https://pytorch.org/docs/2.7/generated/torch.Tensor.addr.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.addr_](https://pytorch.org/docs/2.7/generated/torch.Tensor.addr_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.adjoint](https://pytorch.org/docs/2.7/generated/torch.Tensor.adjoint.html)|Yes|-|
|[Tensor.allclose](https://pytorch.org/docs/2.7/generated/torch.Tensor.allclose.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.amax](https://pytorch.org/docs/2.7/generated/torch.Tensor.amax.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.amin](https://pytorch.org/docs/2.7/generated/torch.Tensor.amin.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.aminmax](https://pytorch.org/docs/2.7/generated/torch.Tensor.aminmax.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.angle](https://pytorch.org/docs/2.7/generated/torch.Tensor.angle.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|[Tensor.apply_](https://pytorch.org/docs/2.7/generated/torch.Tensor.apply_.html)|Yes|CPU only|
|[Tensor.argmax](https://pytorch.org/docs/2.7/generated/torch.Tensor.argmax.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.argmin](https://pytorch.org/docs/2.7/generated/torch.Tensor.argmin.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.argsort](https://pytorch.org/docs/2.7/generated/torch.Tensor.argsort.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.argwhere](https://pytorch.org/docs/2.7/generated/torch.Tensor.argwhere.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.asin](https://pytorch.org/docs/2.7/generated/torch.Tensor.asin.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.asin_](https://pytorch.org/docs/2.7/generated/torch.Tensor.asin_.html)|Yes|Supports fp16, fp32|
|[Tensor.arcsin](https://pytorch.org/docs/2.7/generated/torch.Tensor.arcsin.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.arcsin_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arcsin_.html)|Yes|Supports fp16, fp32|
|[Tensor.as_strided](https://pytorch.org/docs/2.7/generated/torch.Tensor.as_strided.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.atan](https://pytorch.org/docs/2.7/generated/torch.Tensor.atan.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.atan_](https://pytorch.org/docs/2.7/generated/torch.Tensor.atan_.html)|Yes|Supports fp16, fp32|
|[Tensor.arctan](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctan.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.arctan_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctan_.html)|Yes|Supports fp16, fp32|
|[Tensor.atan2](https://pytorch.org/docs/2.7/generated/torch.Tensor.atan2.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.atan2_](https://pytorch.org/docs/2.7/generated/torch.Tensor.atan2_.html)|Yes|Supports fp16, fp32<br>may fall back to CPU|
|[Tensor.arctan2](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctan2.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.arctan2_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctan2_.html)|Yes|Supports fp16, fp32|
|[Tensor.all](https://pytorch.org/docs/2.7/generated/torch.Tensor.all.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.any](https://pytorch.org/docs/2.7/generated/torch.Tensor.any.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.backward](https://pytorch.org/docs/2.7/generated/torch.Tensor.backward.html)|Yes|supports fp32|
|[Tensor.baddbmm](https://pytorch.org/docs/2.7/generated/torch.Tensor.baddbmm.html)|Yes|Supports fp16, fp32|
|[Tensor.baddbmm_](https://pytorch.org/docs/2.7/generated/torch.Tensor.baddbmm_.html)|Yes|Supports fp16, fp32|
|[Tensor.bernoulli](https://pytorch.org/docs/2.7/generated/torch.Tensor.bernoulli.html)|Yes|Supports fp16, fp32<br>may fall back to CPU|
|[Tensor.bernoulli_](https://pytorch.org/docs/2.7/generated/torch.Tensor.bernoulli_.html)|Yes|may fall back to CPU|
|[Tensor.bfloat16](https://pytorch.org/docs/2.7/generated/torch.Tensor.bfloat16.html)|Yes|Supports fp16, fp32|
|[Tensor.bincount](https://pytorch.org/docs/2.7/generated/torch.Tensor.bincount.html)|Yes|Supports uint8, int8, int16, int32, int64<br>The weights dimension must be consistent with the input dimension|
|[Tensor.bitwise_not](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_not.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_not_](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_not_.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_and](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_and.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_and_](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_and_.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_or](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_or.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_or_](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_or_.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_xor](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_xor.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_xor_](https://pytorch.org/docs/2.7/generated/torch.Tensor.bitwise_xor_.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bmm](https://pytorch.org/docs/2.7/generated/torch.Tensor.bmm.html)|Yes|Supports bf16, fp16, fp32|
|[Tensor.bool](https://pytorch.org/docs/2.7/generated/torch.Tensor.bool.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.byte](https://pytorch.org/docs/2.7/generated/torch.Tensor.byte.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.broadcast_to](https://pytorch.org/docs/2.7/generated/torch.Tensor.broadcast_to.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.ceil](https://pytorch.org/docs/2.7/generated/torch.Tensor.ceil.html)|Yes|Supports fp16, fp32|
|[Tensor.ceil_](https://pytorch.org/docs/2.7/generated/torch.Tensor.ceil_.html)|Yes|Supports fp16, fp32|
|[Tensor.char](https://pytorch.org/docs/2.7/generated/torch.Tensor.char.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.chunk](https://pytorch.org/docs/2.7/generated/torch.Tensor.chunk.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.clamp](https://pytorch.org/docs/2.7/generated/torch.Tensor.clamp.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.clamp_](https://pytorch.org/docs/2.7/generated/torch.Tensor.clamp_.html)|Yes|may fall back to CPU|
|[Tensor.clip](https://pytorch.org/docs/2.7/generated/torch.Tensor.clip.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.clip_](https://pytorch.org/docs/2.7/generated/torch.Tensor.clip_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.clone](https://pytorch.org/docs/2.7/generated/torch.Tensor.clone.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.contiguous](https://pytorch.org/docs/2.7/generated/torch.Tensor.contiguous.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.copy_](https://pytorch.org/docs/2.7/generated/torch.Tensor.copy_.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>int16 does not support dimensions greater than 5|
|[Tensor.conj](https://pytorch.org/docs/2.7/generated/torch.Tensor.conj.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.resolve_conj](https://pytorch.org/docs/2.7/generated/torch.Tensor.resolve_conj.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.resolve_neg](https://pytorch.org/docs/2.7/generated/torch.Tensor.resolve_neg.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.copysign](https://pytorch.org/docs/2.7/generated/torch.Tensor.copysign.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool<br>may fall back to CPU|
|[Tensor.cos](https://pytorch.org/docs/2.7/generated/torch.Tensor.cos.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.cos_](https://pytorch.org/docs/2.7/generated/torch.Tensor.cos_.html)|Yes|supports bf16, fp16, fp32, complex64, complex128|
|[Tensor.cosh](https://pytorch.org/docs/2.7/generated/torch.Tensor.cosh.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.cosh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.cosh_.html)|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.count_nonzero](https://pytorch.org/docs/2.7/generated/torch.Tensor.count_nonzero.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.cov](https://pytorch.org/docs/2.7/generated/torch.Tensor.cov.html)|Yes|supports fp16, fp32, int16, int32, int64|
|[Tensor.acosh](https://pytorch.org/docs/2.7/generated/torch.Tensor.acosh.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>may fall back to CPU|
|[Tensor.acosh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.acosh_.html)|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.arccosh](https://pytorch.org/docs/2.7/generated/torch.Tensor.arccosh.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.arccosh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arccosh_.html)|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.cpu](https://pytorch.org/docs/2.7/generated/torch.Tensor.cpu.html)|Yes|-|
|[Tensor.cross](https://pytorch.org/docs/2.7/generated/torch.Tensor.cross.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128<br>The shapes of the two inputs must be consistent|
|[Tensor.cuda](https://pytorch.org/docs/2.7/generated/torch.Tensor.cuda.html)|Yes|The corresponding NPU interface is Tensor.npu, and memory_format only supports passing torch.contiguous_format|
|[Tensor.cummax](https://pytorch.org/docs/2.7/generated/torch.Tensor.cummax.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>may fall back to CPU|
|[Tensor.cummin](https://pytorch.org/docs/2.7/generated/torch.Tensor.cummin.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.cumsum](https://pytorch.org/docs/2.7/generated/torch.Tensor.cumsum.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>supports Named Tensor|
|[Tensor.cumsum_](https://pytorch.org/docs/2.7/generated/torch.Tensor.cumsum_.html)|Yes|supports fp16, fp32, int64, bool|
|[Tensor.chalf](https://pytorch.org/docs/2.7/generated/torch.Tensor.chalf.html)|No|-|
|[Tensor.cfloat](https://pytorch.org/docs/2.7/generated/torch.Tensor.cfloat.html)|Yes|-|
|[Tensor.cdouble](https://pytorch.org/docs/2.7/generated/torch.Tensor.cdouble.html)|Yes|-|
|[Tensor.data_ptr](https://pytorch.org/docs/2.7/generated/torch.Tensor.data_ptr.html)|Yes|supports fp32|
|[Tensor.deg2rad](https://pytorch.org/docs/2.7/generated/torch.Tensor.deg2rad.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.dequantize](https://pytorch.org/docs/2.7/generated/torch.Tensor.dequantize.html)|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.dense_dim](https://pytorch.org/docs/2.7/generated/torch.Tensor.dense_dim.html)|Yes|-|
|[Tensor.detach](https://pytorch.org/docs/2.7/generated/torch.Tensor.detach.html)|Yes|supports fp32|
|[Tensor.detach_](https://pytorch.org/docs/2.7/generated/torch.Tensor.detach_.html)|Yes|supports fp32|
|[Tensor.diag](https://pytorch.org/docs/2.7/generated/torch.Tensor.diag.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|[Tensor.diag_embed](https://pytorch.org/docs/2.7/generated/torch.Tensor.diag_embed.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.diagflat](https://pytorch.org/docs/2.7/generated/torch.Tensor.diagflat.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|[Tensor.diagonal](https://pytorch.org/docs/2.7/generated/torch.Tensor.diagonal.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.diagonal_scatter](https://pytorch.org/docs/2.7/generated/torch.Tensor.diagonal_scatter.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.fill_diagonal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.fill_diagonal_.html)|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.diff](https://pytorch.org/docs/2.7/generated/torch.Tensor.diff.html)|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.dim](https://pytorch.org/docs/2.7/generated/torch.Tensor.dim.html)|Yes|supports fp32|
|[Tensor.dim_order](https://pytorch.org/docs/2.7/generated/torch.Tensor.dim_order.html)|Yes|-|
|[Tensor.dist](https://pytorch.org/docs/2.7/generated/torch.Tensor.dist.html)|Yes|-|
|[Tensor.div](https://pytorch.org/docs/2.7/generated/torch.Tensor.div.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.div_](https://pytorch.org/docs/2.7/generated/torch.Tensor.div_.html)|Yes|supports bf16, fp16, fp32, fp64|
|[Tensor.divide](https://pytorch.org/docs/2.7/generated/torch.Tensor.divide.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.divide_](https://pytorch.org/docs/2.7/generated/torch.Tensor.divide_.html)|Yes|supports fp16, fp32|
|[Tensor.dot](https://pytorch.org/docs/2.7/generated/torch.Tensor.dot.html)|Yes|supports fp16, fp32|
|[Tensor.double](https://pytorch.org/docs/2.7/generated/torch.Tensor.double.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Currently, some NPU interfaces do not support the double type. For compatibility, fp32 is returned by default. fp64 will be returned normally after support is completed.<br>may fall back to CPU|
|[Tensor.dsplit](https://pytorch.org/docs/2.7/generated/torch.Tensor.dsplit.html)|Yes|supports fp32|
|[Tensor.element_size](https://pytorch.org/docs/2.7/generated/torch.Tensor.element_size.html)|Yes|supports fp32|
|[Tensor.eq](https://pytorch.org/docs/2.7/generated/torch.Tensor.eq.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.eq_](https://pytorch.org/docs/2.7/generated/torch.Tensor.eq_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.equal](https://pytorch.org/docs/2.7/generated/torch.Tensor.equal.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.erf](https://pytorch.org/docs/2.7/generated/torch.Tensor.erf.html)|Yes|supports fp16, fp32, int64, bool|
|[Tensor.erf_](https://pytorch.org/docs/2.7/generated/torch.Tensor.erf_.html)|Yes|supports fp16, fp32|
|[Tensor.erfc](https://pytorch.org/docs/2.7/generated/torch.Tensor.erfc.html)|Yes|supports fp16, fp32, int64, bool|
|[Tensor.erfc_](https://pytorch.org/docs/2.7/generated/torch.Tensor.erfc_.html)|Yes|supports fp16, fp32|
|[Tensor.erfinv](https://pytorch.org/docs/2.7/generated/torch.Tensor.erfinv.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.erfinv_](https://pytorch.org/docs/2.7/generated/torch.Tensor.erfinv_.html)|Yes|supports bf16, fp16, fp32|
|[Tensor.exp](https://pytorch.org/docs/2.7/generated/torch.Tensor.exp.html)|Yes|supports bf16, fp16, fp32, fp64, int64, bool|
|[Tensor.exp_](https://pytorch.org/docs/2.7/generated/torch.Tensor.exp_.html)|Yes|supports bf16, fp16, fp32, complex64, complex128|
|[Tensor.expm1](https://pytorch.org/docs/2.7/generated/torch.Tensor.expm1.html)|Yes|supports fp16, fp32, int64, bool|
|[Tensor.expm1_](https://pytorch.org/docs/2.7/generated/torch.Tensor.expm1_.html)|Yes|supports fp16, fp32|
|[Tensor.expand](https://pytorch.org/docs/2.7/generated/torch.Tensor.expand.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.expand_as](https://pytorch.org/docs/2.7/generated/torch.Tensor.expand_as.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.exponential_](https://pytorch.org/docs/2.7/generated/torch.Tensor.exponential_.html)|Yes|supports bf16, fp16, fp32, fp64|
|[Tensor.fix](https://pytorch.org/docs/2.7/generated/torch.Tensor.fix.html)|Yes|supports fp16, fp32|
|[Tensor.fix_](https://pytorch.org/docs/2.7/generated/torch.Tensor.fix_.html)|Yes|supports fp16, fp32|
|[Tensor.fill_](https://pytorch.org/docs/2.7/generated/torch.Tensor.fill_.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.flatten](https://pytorch.org/docs/2.7/generated/torch.Tensor.flatten.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.flip](https://pytorch.org/docs/2.7/generated/torch.Tensor.flip.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.fliplr](https://pytorch.org/docs/2.7/generated/torch.Tensor.fliplr.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.flipud](https://pytorch.org/docs/2.7/generated/torch.Tensor.flipud.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.float](https://pytorch.org/docs/2.7/generated/torch.Tensor.float.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.float_power](https://pytorch.org/docs/2.7/generated/torch.Tensor.float_power.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex128|
|[Tensor.float_power_](https://pytorch.org/docs/2.7/generated/torch.Tensor.float_power_.html)|Yes|supports double|
|[Tensor.floor](https://pytorch.org/docs/2.7/generated/torch.Tensor.floor.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.floor_](https://pytorch.org/docs/2.7/generated/torch.Tensor.floor_.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.floor_divide](https://pytorch.org/docs/2.7/generated/torch.Tensor.floor_divide.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.floor_divide_](https://pytorch.org/docs/2.7/generated/torch.Tensor.floor_divide_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.fmod](https://pytorch.org/docs/2.7/generated/torch.Tensor.fmod.html)|Yes|supports fp16, fp32, uint8, int8, int32, int64|
|[Tensor.fmod_](https://pytorch.org/docs/2.7/generated/torch.Tensor.fmod_.html)|Yes|supports fp16, fp32, uint8, int8, int32, int64|
|[Tensor.frac](https://pytorch.org/docs/2.7/generated/torch.Tensor.frac.html)|Yes|supports fp16, fp32|
|[Tensor.frac_](https://pytorch.org/docs/2.7/generated/torch.Tensor.frac_.html)|Yes|supports fp16, fp32|
|[Tensor.gather](https://pytorch.org/docs/2.7/generated/torch.Tensor.gather.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>The index dimension must be consistent with the input dimension|
|[Tensor.ge](https://pytorch.org/docs/2.7/generated/torch.Tensor.ge.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.ge_](https://pytorch.org/docs/2.7/generated/torch.Tensor.ge_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.greater_equal](https://pytorch.org/docs/2.7/generated/torch.Tensor.greater_equal.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.greater_equal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.greater_equal_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.geometric_](https://pytorch.org/docs/2.7/generated/torch.Tensor.geometric_.html)|Yes|-|
|[Tensor.ger](https://pytorch.org/docs/2.7/generated/torch.Tensor.ger.html)|Yes|-|
|[Tensor.get_device](https://pytorch.org/docs/2.7/generated/torch.Tensor.get_device.html)|Yes|-|
|[Tensor.gt](https://pytorch.org/docs/2.7/generated/torch.Tensor.gt.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.gt_](https://pytorch.org/docs/2.7/generated/torch.Tensor.gt_.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.greater](https://pytorch.org/docs/2.7/generated/torch.Tensor.greater.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.greater_](https://pytorch.org/docs/2.7/generated/torch.Tensor.greater_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.half](https://pytorch.org/docs/2.7/generated/torch.Tensor.half.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.hardshrink](https://pytorch.org/docs/2.7/generated/torch.Tensor.hardshrink.html)|Yes|supports fp16, fp32|
|[Tensor.heaviside](https://pytorch.org/docs/2.7/generated/torch.Tensor.heaviside.html)|Yes|may fall back to CPU|
|[Tensor.histc](https://pytorch.org/docs/2.7/generated/torch.Tensor.histc.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.hsplit](https://pytorch.org/docs/2.7/generated/torch.Tensor.hsplit.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.index_add_](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_add_.html)|Yes|supports fp16, fp32, int64, bool|
|[Tensor.index_add](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_add.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.index_copy_](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_copy_.html)|Yes|supports fp16, fp32, int16, int32, bool|
|[Tensor.index_copy](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_copy.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.index_fill_](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_fill_.html)|Yes|supports fp16, fp32, int32, int64, bool|
|[Tensor.index_fill](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_fill.html)|Yes|supports bf16, fp16, fp32, int32, int64, bool|
|[Tensor.index_put_](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_put_.html)|Yes|supports int64|
|[Tensor.index_put](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_put.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.index_reduce_](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_reduce_.html)|Yes|may fall back to CPU|
|[Tensor.index_reduce](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_reduce.html)|Yes|may fall back to CPU|
|[Tensor.index_select](https://pytorch.org/docs/2.7/generated/torch.Tensor.index_select.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.indices](https://pytorch.org/docs/2.7/generated/torch.Tensor.indices.html)|Yes|-|
|[Tensor.inner](https://pytorch.org/docs/2.7/generated/torch.Tensor.inner.html)|Yes|supports fp16, fp32|
|[Tensor.int](https://pytorch.org/docs/2.7/generated/torch.Tensor.int.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.int_repr](https://pytorch.org/docs/2.7/generated/torch.Tensor.int_repr.html)|No|-|
|[Tensor.isclose](https://pytorch.org/docs/2.7/generated/torch.Tensor.isclose.html)|Yes|supports fp16, fp32, uint8, int32, int64, bool|
|[Tensor.isfinite](https://pytorch.org/docs/2.7/generated/torch.Tensor.isfinite.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.isinf](https://pytorch.org/docs/2.7/generated/torch.Tensor.isinf.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.isposinf](https://pytorch.org/docs/2.7/generated/torch.Tensor.isposinf.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.isneginf](https://pytorch.org/docs/2.7/generated/torch.Tensor.isneginf.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.isnan](https://pytorch.org/docs/2.7/generated/torch.Tensor.isnan.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.is_contiguous](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_contiguous.html)|Yes|supports fp32|
|[Tensor.is_complex](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_complex.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.is_conj](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_conj.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.is_floating_point](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_floating_point.html)|Yes|supports fp32|
|[Tensor.is_inference](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_inference.html)|Yes|supports fp32|
|[Tensor.is_leaf](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_leaf.html)|Yes|-|
|[Tensor.is_pinned](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_pinned.html)|Yes|supports fp32|
|[Tensor.is_set_to](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_set_to.html)|Yes|supports fp32|
|[Tensor.is_shared](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_shared.html)|No|-|
|[Tensor.is_signed](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_signed.html)|Yes|supports fp32|
|[Tensor.is_sparse](https://pytorch.org/docs/2.7/generated/torch.Tensor.is_sparse.html)|Yes|supports fp32|
|[Tensor.isreal](https://pytorch.org/docs/2.7/generated/torch.Tensor.isreal.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.item](https://pytorch.org/docs/2.7/generated/torch.Tensor.item.html)|Yes|supports fp32|
|[Tensor.kthvalue](https://pytorch.org/docs/2.7/generated/torch.Tensor.kthvalue.html)|Yes|supports fp16, fp32, int32|
|[Tensor.ldexp](https://pytorch.org/docs/2.7/generated/torch.Tensor.ldexp.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.ldexp_](https://pytorch.org/docs/2.7/generated/torch.Tensor.ldexp_.html)|Yes|supports fp16, fp32, complex64, complex128|
|[Tensor.le](https://pytorch.org/docs/2.7/generated/torch.Tensor.le.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.le_](https://pytorch.org/docs/2.7/generated/torch.Tensor.le_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.less_equal](https://pytorch.org/docs/2.7/generated/torch.Tensor.less_equal.html)|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.less_equal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.less_equal_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.lerp](https://pytorch.org/docs/2.7/generated/torch.Tensor.lerp.html)|Yes|supports fp16, fp32|
|[Tensor.lerp_](https://pytorch.org/docs/2.7/generated/torch.Tensor.lerp_.html)|Yes|supports fp16, fp32|
|[Tensor.log](https://pytorch.org/docs/2.7/generated/torch.Tensor.log.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.log_](https://pytorch.org/docs/2.7/generated/torch.Tensor.log_.html)|Yes|supports bf16, fp16, fp32, complex64, complex128|
|[Tensor.log10](https://pytorch.org/docs/2.7/generated/torch.Tensor.log10.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.log10_](https://pytorch.org/docs/2.7/generated/torch.Tensor.log10_.html)|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.log1p](https://pytorch.org/docs/2.7/generated/torch.Tensor.log1p.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.log1p_](https://pytorch.org/docs/2.7/generated/torch.Tensor.log1p_.html)|Yes|supports fp16, fp32|
|[Tensor.log2](https://pytorch.org/docs/2.7/generated/torch.Tensor.log2.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.log2_](https://pytorch.org/docs/2.7/generated/torch.Tensor.log2_.html)|Yes|supports fp16, fp32, complex64, complex128|
|[Tensor.logaddexp](https://pytorch.org/docs/2.7/generated/torch.Tensor.logaddexp.html)|Yes|supports fp16, fp32, int16, int32, int64, bool|
|[Tensor.logaddexp2](https://pytorch.org/docs/2.7/generated/torch.Tensor.logaddexp2.html)|Yes|supports fp16, fp32|
|[Tensor.logsumexp](https://pytorch.org/docs/2.7/generated/torch.Tensor.logsumexp.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.logical_and](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_and.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.logical_and_](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_and_.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.logical_not](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_not.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.logical_not_](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_not_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.logical_or](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_or.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.logical_or_](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_or_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.logical_xor](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_xor.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>may fall back to CPU|
|[Tensor.logical_xor_](https://pytorch.org/docs/2.7/generated/torch.Tensor.logical_xor_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.logit](https://pytorch.org/docs/2.7/generated/torch.Tensor.logit.html)|Yes|supports bf16, fp16, fp32<br>output is nan when eps is greater than 1, output is inf when eps is 1|
|[Tensor.logit_](https://pytorch.org/docs/2.7/generated/torch.Tensor.logit_.html)|Yes|supports bf16, fp16, fp32<br>output is nan when eps is greater than 1, output is inf when eps is 1|
|[Tensor.long](https://pytorch.org/docs/2.7/generated/torch.Tensor.long.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.lt](https://pytorch.org/docs/2.7/generated/torch.Tensor.lt.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.lt_](https://pytorch.org/docs/2.7/generated/torch.Tensor.lt_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.less](https://pytorch.org/docs/2.7/generated/torch.Tensor.less.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.less_](https://pytorch.org/docs/2.7/generated/torch.Tensor.less_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.as_subclass](https://pytorch.org/docs/2.7/generated/torch.Tensor.as_subclass.html)|Yes|-|
|[Tensor.map_](https://pytorch.org/docs/2.7/generated/torch.Tensor.map_.html)|Yes|CPU only|
|[Tensor.masked_scatter_](https://pytorch.org/docs/2.7/generated/torch.Tensor.masked_scatter_.html)|Yes|supports fp32, int64, bool|
|[Tensor.masked_scatter](https://pytorch.org/docs/2.7/generated/torch.Tensor.masked_scatter.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.masked_fill_](https://pytorch.org/docs/2.7/generated/torch.Tensor.masked_fill_.html)|Yes|supports bf16, fp16, fp32, int8, int32, int64, bool|
|[Tensor.masked_fill](https://pytorch.org/docs/2.7/generated/torch.Tensor.masked_fill.html)|Yes|supports bf16, fp16, fp32, int8, int32, int64, bool|
|[Tensor.masked_select](https://pytorch.org/docs/2.7/generated/torch.Tensor.masked_select.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.matmul](https://pytorch.org/docs/2.7/generated/torch.Tensor.matmul.html)|Yes|supports bf16, fp16, fp32<br>supports Named Tensor|
|[Tensor.matrix_power](https://pytorch.org/docs/2.7/generated/torch.Tensor.matrix_power.html)|Yes|supports fp16, fp32|
|[Tensor.max](https://pytorch.org/docs/2.7/generated/torch.Tensor.max.html)|Yes|supports bf16, fp16, fp32, int64, bool|
|[Tensor.maximum](https://pytorch.org/docs/2.7/generated/torch.Tensor.maximum.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.mean](https://pytorch.org/docs/2.7/generated/torch.Tensor.mean.html)|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.nanmean](https://pytorch.org/docs/2.7/generated/torch.Tensor.nanmean.html)|Yes|supports fp16, fp32|
|[Tensor.median](https://pytorch.org/docs/2.7/generated/torch.Tensor.median.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64<br>when input is bf16, dim does not take the dimension where the input axis value is 1|
|[Tensor.min](https://pytorch.org/docs/2.7/generated/torch.Tensor.min.html)|Yes|supports bf16, fp16, fp32, int64, bool|
|[Tensor.minimum](https://pytorch.org/docs/2.7/generated/torch.Tensor.minimum.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.mm](https://pytorch.org/docs/2.7/generated/torch.Tensor.mm.html)|Yes|supports bf16, fp16, fp32|
|[Tensor.smm](https://pytorch.org/docs/2.7/generated/torch.Tensor.smm.html)|No|-|
|[Tensor.mode](https://pytorch.org/docs/2.7/generated/torch.Tensor.mode.html)|No|-|
|[Tensor.movedim](https://pytorch.org/docs/2.7/generated/torch.Tensor.movedim.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.moveaxis](https://pytorch.org/docs/2.7/generated/torch.Tensor.moveaxis.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.msort](https://pytorch.org/docs/2.7/generated/torch.Tensor.msort.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.mul](https://pytorch.org/docs/2.7/generated/torch.Tensor.mul.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.mul_](https://pytorch.org/docs/2.7/generated/torch.Tensor.mul_.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.multiply](https://pytorch.org/docs/2.7/generated/torch.Tensor.multiply.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.multiply_](https://pytorch.org/docs/2.7/generated/torch.Tensor.multiply_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.multinomial](https://pytorch.org/docs/2.7/generated/torch.Tensor.multinomial.html)|Yes|supports fp16, fp32|
|[Tensor.nansum](https://pytorch.org/docs/2.7/generated/torch.Tensor.nansum.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.narrow](https://pytorch.org/docs/2.7/generated/torch.Tensor.narrow.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.narrow_copy](https://pytorch.org/docs/2.7/generated/torch.Tensor.narrow_copy.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.ndimension](https://pytorch.org/docs/2.7/generated/torch.Tensor.ndimension.html)|Yes|supports fp32|
|[Tensor.nan_to_num](https://pytorch.org/docs/2.7/generated/torch.Tensor.nan_to_num.html)|Yes|-|
|[Tensor.nan_to_num_](https://pytorch.org/docs/2.7/generated/torch.Tensor.nan_to_num_.html)|Yes|-|
|[Tensor.ne](https://pytorch.org/docs/2.7/generated/torch.Tensor.ne.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.ne_](https://pytorch.org/docs/2.7/generated/torch.Tensor.ne_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.nextafter_](https://pytorch.org/docs/2.7/generated/torch.Tensor.nextafter_.html)|Yes|falls back to CPU|
|[Tensor.not_equal](https://pytorch.org/docs/2.7/generated/torch.Tensor.not_equal.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>may fall back to CPU|
|[Tensor.not_equal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.not_equal_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.neg](https://pytorch.org/docs/2.7/generated/torch.Tensor.neg.html)|Yes|supports bf16, fp16, fp32, int8, int32, int64|
|[Tensor.neg_](https://pytorch.org/docs/2.7/generated/torch.Tensor.neg_.html)|Yes|supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128<br>may fall back to CPU|
|[Tensor.negative](https://pytorch.org/docs/2.7/generated/torch.Tensor.negative.html)|Yes|supports fp16, fp32, int8, int32, int64, complex64, complex128|
|[Tensor.negative_](https://pytorch.org/docs/2.7/generated/torch.Tensor.negative_.html)|Yes|supports fp16, fp32, int8, int32, int64, complex64, complex128|
|[Tensor.nelement](https://pytorch.org/docs/2.7/generated/torch.Tensor.nelement.html)|Yes|supports fp32|
|[Tensor.nonzero](https://pytorch.org/docs/2.7/generated/torch.Tensor.nonzero.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>does not support nan scenarios|
|[Tensor.norm](https://pytorch.org/docs/2.7/generated/torch.Tensor.norm.html)|Yes|supports bf16, fp16, fp32, fp64|
|[Tensor.normal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.normal_.html)|Yes|supports bf16, fp16, fp32<br>may fall back to CPU|
|[Tensor.numel](https://pytorch.org/docs/2.7/generated/torch.Tensor.numel.html)|Yes|supports fp32|
|[Tensor.numpy](https://pytorch.org/docs/2.7/generated/torch.Tensor.numpy.html)|Yes|supports fp32|
|[Tensor.outer](https://pytorch.org/docs/2.7/generated/torch.Tensor.outer.html)|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.permute](https://pytorch.org/docs/2.7/generated/torch.Tensor.permute.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.positive](https://pytorch.org/docs/2.7/generated/torch.Tensor.positive.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|[Tensor.pow](https://pytorch.org/docs/2.7/generated/torch.Tensor.pow.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.pow_](https://pytorch.org/docs/2.7/generated/torch.Tensor.pow_.html)|Yes|supports bf16, fp16, fp32, fp64, int64|
|[Tensor.prod](https://pytorch.org/docs/2.7/generated/torch.Tensor.prod.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.put_](https://pytorch.org/docs/2.7/generated/torch.Tensor.put_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|[Tensor.qscheme](https://pytorch.org/docs/2.7/generated/torch.Tensor.qscheme.html)|No|-|
|[Tensor.quantile](https://pytorch.org/docs/2.7/generated/torch.Tensor.quantile.html)|Yes|-|
|[Tensor.rad2deg](https://pytorch.org/docs/2.7/generated/torch.Tensor.rad2deg.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.random_](https://pytorch.org/docs/2.7/generated/torch.Tensor.random_.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.ravel](https://pytorch.org/docs/2.7/generated/torch.Tensor.ravel.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.reciprocal](https://pytorch.org/docs/2.7/generated/torch.Tensor.reciprocal.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.reciprocal_](https://pytorch.org/docs/2.7/generated/torch.Tensor.reciprocal_.html)|Yes|supports fp16, fp32, complex64, complex128|
|[Tensor.record_stream](https://pytorch.org/docs/2.7/generated/torch.Tensor.record_stream.html)|Yes|supports fp32|
|[Tensor.register_hook](https://pytorch.org/docs/2.7/generated/torch.Tensor.register_hook.html)|Yes|supports fp32|
|[Tensor.register_post_accumulate_grad_hook](https://pytorch.org/docs/2.7/generated/torch.Tensor.register_post_accumulate_grad_hook.html)|Yes|-|
|[Tensor.remainder](https://pytorch.org/docs/2.7/generated/torch.Tensor.remainder.html)|Yes|supports bf16, fp16, fp32, fp64, int32, int64|
|[Tensor.remainder_](https://pytorch.org/docs/2.7/generated/torch.Tensor.remainder_.html)|Yes|supports fp16, fp32, int32, int64|
|[Tensor.repeat](https://pytorch.org/docs/2.7/generated/torch.Tensor.repeat.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.repeat_interleave](https://pytorch.org/docs/2.7/generated/torch.Tensor.repeat_interleave.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool<br>the input tensor is repeated to produce the output, and the number of elements in the output must be less than $2^{22}$|
|[Tensor.requires_grad](https://pytorch.org/docs/2.7/generated/torch.Tensor.requires_grad.html)|Yes|-|
|[Tensor.requires_grad_](https://pytorch.org/docs/2.7/generated/torch.Tensor.requires_grad_.html)|Yes|supports fp32|
|[Tensor.reshape](https://pytorch.org/docs/2.7/generated/torch.Tensor.reshape.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.reshape_as](https://pytorch.org/docs/2.7/generated/torch.Tensor.reshape_as.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.resize_](https://pytorch.org/docs/2.7/generated/torch.Tensor.resize_.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>memory_format only supports torch.contiguous_format and torch.preserve_format|
|[Tensor.resize_as_](https://pytorch.org/docs/2.7/generated/torch.Tensor.resize_as_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>memory_format only supports torch.contiguous_format and torch.preserve_format|
|[Tensor.retain_grad](https://pytorch.org/docs/2.7/generated/torch.Tensor.retain_grad.html)|Yes|supports fp32|
|[Tensor.retains_grad](https://pytorch.org/docs/2.7/generated/torch.Tensor.retains_grad.html)|Yes|supports fp32|
|[Tensor.roll](https://pytorch.org/docs/2.7/generated/torch.Tensor.roll.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int32, int64, bool|
|[Tensor.rot90](https://pytorch.org/docs/2.7/generated/torch.Tensor.rot90.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.round](https://pytorch.org/docs/2.7/generated/torch.Tensor.round.html)|Yes|supports fp16, fp32|
|[Tensor.round_](https://pytorch.org/docs/2.7/generated/torch.Tensor.round_.html)|Yes|supports fp16, fp32|
|[Tensor.rsqrt](https://pytorch.org/docs/2.7/generated/torch.Tensor.rsqrt.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.rsqrt_](https://pytorch.org/docs/2.7/generated/torch.Tensor.rsqrt_.html)|Yes|supports fp16, fp32, complex64, complex128|
|[Tensor.scatter](https://pytorch.org/docs/2.7/generated/torch.Tensor.scatter.html)|Yes|supports fp16, fp32, fp64, int8, int16, int32, int64, bool|
|[Tensor.scatter_](https://pytorch.org/docs/2.7/generated/torch.Tensor.scatter_.html)|Yes|tensor, index, src parameters cannot be empty and cannot be scalar<br>may fall back to CPU|
|[Tensor.scatter_add_](https://pytorch.org/docs/2.7/generated/torch.Tensor.scatter_add_.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.scatter_add](https://pytorch.org/docs/2.7/generated/torch.Tensor.scatter_add.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.scatter_reduce](https://pytorch.org/docs/2.7/generated/torch.Tensor.scatter_reduce.html)|Yes|supports fp32, int64<br>may fall back to CPU|
|[Tensor.select](https://pytorch.org/docs/2.7/generated/torch.Tensor.select.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.select_scatter](https://pytorch.org/docs/2.7/generated/torch.Tensor.select_scatter.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>may fall back to CPU|
|[Tensor.set_](https://pytorch.org/docs/2.7/generated/torch.Tensor.set_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.share_memory_](https://pytorch.org/docs/2.7/generated/torch.Tensor.share_memory_.html)|No|-|
|[Tensor.short](https://pytorch.org/docs/2.7/generated/torch.Tensor.short.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.sigmoid](https://pytorch.org/docs/2.7/generated/torch.Tensor.sigmoid.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.sigmoid_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sigmoid_.html)|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.sign](https://pytorch.org/docs/2.7/generated/torch.Tensor.sign.html)|Yes|supports bf16, fp16, fp32, fp64, int32, int64, bool|
|[Tensor.sign_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sign_.html)|Yes|supports fp16, fp32, int32, int64, bool|
|[Tensor.sgn](https://pytorch.org/docs/2.7/generated/torch.Tensor.sgn.html)|Yes|supports fp16, fp32, int32, int64, bool, complex64, complex128|
|[Tensor.sgn_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sgn_.html)|Yes|supports fp16, fp32, fp64, int32, int64, bool|
|[Tensor.sin](https://pytorch.org/docs/2.7/generated/torch.Tensor.sin.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.sin_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sin_.html)|Yes|supports bf16, fp16, fp32, complex64, complex128|
|[Tensor.sinh](https://pytorch.org/docs/2.7/generated/torch.Tensor.sinh.html)|Yes|supports fp16, fp32, fp64|
|[Tensor.sinh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sinh_.html)|Yes|supports fp16, fp32, fp64|
|[Tensor.asinh](https://pytorch.org/docs/2.7/generated/torch.Tensor.asinh.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.asinh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.asinh_.html)|Yes|supports fp16, fp32, complex64, complex128|
|[Tensor.arcsinh](https://pytorch.org/docs/2.7/generated/torch.Tensor.arcsinh.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.arcsinh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arcsinh_.html)|Yes|supports fp16, fp32, complex64, complex128|
|[Tensor.shape](https://pytorch.org/docs/2.7/generated/torch.Tensor.shape.html)|Yes|-|
|[Tensor.size](https://pytorch.org/docs/2.7/generated/torch.Tensor.size.html)|Yes|supports fp32|
|[Tensor.slogdet](https://pytorch.org/docs/2.7/generated/torch.Tensor.slogdet.html)|Yes|supports fp32, complex64, complex128|
|[Tensor.slice_scatter](https://pytorch.org/docs/2.7/generated/torch.Tensor.slice_scatter.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.softmax](https://pytorch.org/docs/2.7/generated/torch.Tensor.softmax.html)||supports fp16, fp16, fp32, fp64|
|[Tensor.sort](https://pytorch.org/docs/2.7/generated/torch.Tensor.sort.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.split](https://pytorch.org/docs/2.7/generated/torch.Tensor.split.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.sparse_mask](https://pytorch.org/docs/2.7/generated/torch.Tensor.sparse_mask.html)|No|-|
|[Tensor.sparse_dim](https://pytorch.org/docs/2.7/generated/torch.Tensor.sparse_dim.html)|Yes|-|
|[Tensor.sqrt](https://pytorch.org/docs/2.7/generated/torch.Tensor.sqrt.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.sqrt_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sqrt_.html)|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.square](https://pytorch.org/docs/2.7/generated/torch.Tensor.square.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.square_](https://pytorch.org/docs/2.7/generated/torch.Tensor.square_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|[Tensor.squeeze](https://pytorch.org/docs/2.7/generated/torch.Tensor.squeeze.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.squeeze_](https://pytorch.org/docs/2.7/generated/torch.Tensor.squeeze_.html)|Yes|supports fp32|
|[Tensor.std](https://pytorch.org/docs/2.7/generated/torch.Tensor.std.html)|Yes|supports bf16, fp16, fp32<br>input does not support scalar tensor<br>correction is not greater than the range of int32|
|[Tensor.storage](https://pytorch.org/docs/2.7/generated/torch.Tensor.storage.html)|Yes|supports fp32|
|[Tensor.untyped_storage](https://pytorch.org/docs/2.7/generated/torch.Tensor.untyped_storage.html)|Yes|supports fp32|
|[Tensor.storage_offset](https://pytorch.org/docs/2.7/generated/torch.Tensor.storage_offset.html)|Yes|supports fp32|
|[Tensor.storage_type](https://pytorch.org/docs/2.7/generated/torch.Tensor.storage_type.html)|Yes|supports fp32|
|[Tensor.stride](https://pytorch.org/docs/2.7/generated/torch.Tensor.stride.html)|Yes|supports fp32|
|[Tensor.sub](https://pytorch.org/docs/2.7/generated/torch.Tensor.sub.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.sub_](https://pytorch.org/docs/2.7/generated/torch.Tensor.sub_.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128|
|[Tensor.subtract_](https://pytorch.org/docs/2.7/generated/torch.Tensor.subtract_.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.sum](https://pytorch.org/docs/2.7/generated/torch.Tensor.sum.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.sum_to_size](https://pytorch.org/docs/2.7/generated/torch.Tensor.sum_to_size.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.swapaxes](https://pytorch.org/docs/2.7/generated/torch.Tensor.swapaxes.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.swapdims](https://pytorch.org/docs/2.7/generated/torch.Tensor.swapdims.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.t](https://pytorch.org/docs/2.7/generated/torch.Tensor.t.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128|
|[Tensor.t_](https://pytorch.org/docs/2.7/generated/torch.Tensor.t_.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64|
|[Tensor.tensor_split](https://pytorch.org/docs/2.7/generated/torch.Tensor.tensor_split.html)|Yes|CPU only|
|[Tensor.tile](https://pytorch.org/docs/2.7/generated/torch.Tensor.tile.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>If the length of the input parameter dims is less than the length of Tensor.shape, 1s are automatically prepended to dims to align its length with Tensor.shape. The completed dims must satisfy the following restrictions:<br>- When repeating the first axis, a maximum of 4 dimensions can be repeated simultaneously (i.e., the number of elements greater than 1 in dims ≤ 4). For example: Tensor.tile([2, 3, 4, 5, 6]) is not supported, Tensor.tile([2, 3, 1, 5, 6]) is supported<br>- When the first axis does not need to be repeated, a maximum of 3 dimensions can be repeated simultaneously (i.e., the number of elements greater than 1 in dims ≤ 3). For example: Tensor.tile([1, 3, 4, 5, 6]) is not supported, Tensor.tile([1, 3, 1, 5, 6]) is supported<br>- If backward computation is performed, the sum of the number of Tensor dimensions and the number of elements greater than 1 in the input parameter dims must not exceed 8|
|[Tensor.to](https://pytorch.org/docs/2.7/generated/torch.Tensor.to.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>The current NPU device only supports setting memory_format to torch.contiguous_format or torch.preserve_format<br><term>Atlas Inference Series Product</term> does not support cross-NPU copy|
|[to](https://pytorch.org/docs/2.7/generated/torch.Tensor.to.html)|Yes|-|
|[Tensor.to_mkldnn](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_mkldnn.html)|No|-|
|[Tensor.take](https://pytorch.org/docs/2.7/generated/torch.Tensor.take.html)|Yes|supports fp16, fp32, int16, int32, bool|
|[Tensor.take_along_dim](https://pytorch.org/docs/2.7/generated/torch.Tensor.take_along_dim.html)|Yes|supports fp16, fp32, int16, int32, int64, bool|
|[Tensor.tan](https://pytorch.org/docs/2.7/generated/torch.Tensor.tan.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128, value range [-65504, 65504]|
|[Tensor.tan_](https://pytorch.org/docs/2.7/generated/torch.Tensor.tan_.html)|Yes|supports fp16, fp32, complex64, complex128|
|[Tensor.tanh](https://pytorch.org/docs/2.7/generated/torch.Tensor.tanh.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.tanh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.tanh_.html)|Yes|supports fp16, fp32|
|[Tensor.atanh](https://pytorch.org/docs/2.7/generated/torch.Tensor.atanh.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.atanh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.atanh_.html)|Yes|supports fp16, fp32, complex64, complex128|
|[Tensor.arctanh](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctanh.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.arctanh_](https://pytorch.org/docs/2.7/generated/torch.Tensor.arctanh_.html)|Yes|supports fp16, fp32, complex64, complex128|
|[Tensor.tolist](https://pytorch.org/docs/2.7/generated/torch.Tensor.tolist.html)|Yes|supports fp32|
|[Tensor.topk](https://pytorch.org/docs/2.7/generated/torch.Tensor.topk.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Due to hardware differences, the npu topk index results are inconsistent with gpu/cpu. Currently, npu only supports returning computation results with sorted set to true<br>Scalar tensor is not supported|
|[Tensor.to_dense](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_dense.html)|No|-|
|[Tensor.to_sparse](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse.html)|No|-|
|[to_sparse](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse.html)|No|-|
|[Tensor.to_sparse_csr](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse_csr.html)|No|-|
|[Tensor.to_sparse_csc](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse_csc.html)|No|-|
|[Tensor.to_sparse_bsr](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse_bsr.html)|No|-|
|[Tensor.to_sparse_bsc](https://pytorch.org/docs/2.7/generated/torch.Tensor.to_sparse_bsc.html)|No|-|
|[Tensor.transpose](https://pytorch.org/docs/2.7/generated/torch.Tensor.transpose.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.transpose_](https://pytorch.org/docs/2.7/generated/torch.Tensor.transpose_.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.tril](https://pytorch.org/docs/2.7/generated/torch.Tensor.tril.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.tril_](https://pytorch.org/docs/2.7/generated/torch.Tensor.tril_.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.triu](https://pytorch.org/docs/2.7/generated/torch.Tensor.triu.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.triu_](https://pytorch.org/docs/2.7/generated/torch.Tensor.triu_.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.true_divide](https://pytorch.org/docs/2.7/generated/torch.Tensor.true_divide.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.true_divide_](https://pytorch.org/docs/2.7/generated/torch.Tensor.true_divide_.html)|Yes|supports fp16, fp32|
|[Tensor.trunc](https://pytorch.org/docs/2.7/generated/torch.Tensor.trunc.html)|Yes|supports fp16, fp32|
|[Tensor.trunc_](https://pytorch.org/docs/2.7/generated/torch.Tensor.trunc_.html)|Yes|supports fp16, fp32|
|[Tensor.type](https://pytorch.org/docs/2.7/generated/torch.Tensor.type.html)|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.type_as](https://pytorch.org/docs/2.7/generated/torch.Tensor.type_as.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.unbind](https://pytorch.org/docs/2.7/generated/torch.Tensor.unbind.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.unflatten](https://pytorch.org/docs/2.7/generated/torch.Tensor.unflatten.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.unfold](https://pytorch.org/docs/2.7/generated/torch.Tensor.unfold.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.uniform_](https://pytorch.org/docs/2.7/generated/torch.Tensor.uniform_.html)|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Following the PyTorch community specification, processing of bool type data is no longer supported. For existing bool type data, the following alternatives can be used: If the output needs to be all True, Tensor.bernoulli_(p=1.0) can be used. If uniformly distributed bool type output is needed, Tensor.bernoulli_(p=0.5) can be used|
|[Tensor.unique](https://pytorch.org/docs/2.7/generated/torch.Tensor.unique.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>When the input contains 0, the output may include both positive 0 and negative 0, rather than outputting only one 0|
|[Tensor.unique_consecutive](https://pytorch.org/docs/2.7/generated/torch.Tensor.unique_consecutive.html)|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.unsqueeze](https://pytorch.org/docs/2.7/generated/torch.Tensor.unsqueeze.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.unsqueeze_](https://pytorch.org/docs/2.7/generated/torch.Tensor.unsqueeze_.html)|Yes|supports fp32|
|[Tensor.values](https://pytorch.org/docs/2.7/generated/torch.Tensor.values.html)|Yes|Depends on sparse tensor|
|[Tensor.var](https://pytorch.org/docs/2.7/generated/torch.Tensor.var.html)|Yes|supports bf16, fp16, fp32<br>correction does not exceed the range of int32|
|[Tensor.view](https://pytorch.org/docs/2.7/generated/torch.Tensor.view.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[view](https://pytorch.org/docs/2.7/generated/torch.Tensor.view.html)|Yes|-|
|[Tensor.view_as](https://pytorch.org/docs/2.7/generated/torch.Tensor.view_as.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.vsplit](https://pytorch.org/docs/2.7/generated/torch.Tensor.vsplit.html)|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.where](https://pytorch.org/docs/2.7/generated/torch.Tensor.where.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.xlogy](https://pytorch.org/docs/2.7/generated/torch.Tensor.xlogy.html)|Yes|supports fp16, fp32|
|[Tensor.xlogy_](https://pytorch.org/docs/2.7/generated/torch.Tensor.xlogy_.html)|Yes|supports fp16, fp32|
|[Tensor.zero_](https://pytorch.org/docs/2.7/generated/torch.Tensor.zero_.html)|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
