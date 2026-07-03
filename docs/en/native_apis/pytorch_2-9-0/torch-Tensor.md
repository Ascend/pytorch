# torch.Tensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:22:16.928Z pushedAt=2026-06-15T03:25:49.217Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.Tensor](https://pytorch.org/docs/2.9/tensors.html)|Yes|-|
|[Tensor.T](https://pytorch.org/docs/2.9/tensors.html#torch.Tensor.T)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.H](https://pytorch.org/docs/2.9/tensors.html#torch.Tensor.H)|Yes|-|
|[Tensor.mT](https://pytorch.org/docs/2.9/tensors.html#torch.Tensor.mT)|Yes|-|
|[Tensor.mH](https://pytorch.org/docs/2.9/tensors.html#torch.Tensor.mH)|Yes|-|
|[Tensor.new_tensor](https://pytorch.org/docs/2.9/generated/torch.Tensor.new_tensor.html)|Yes|-|
|[Tensor.new_full](https://pytorch.org/docs/2.9/generated/torch.Tensor.new_full.html)|Yes|Supports int64|
|[Tensor.new_empty](https://pytorch.org/docs/2.9/generated/torch.Tensor.new_empty.html)|Yes|Supports fp32|
|[Tensor.new_ones](https://pytorch.org/docs/2.9/generated/torch.Tensor.new_ones.html)|Yes|Supports fp32|
|[Tensor.new_zeros](https://pytorch.org/docs/2.9/generated/torch.Tensor.new_zeros.html)|Yes|Supports fp32|
|[Tensor.is_cuda](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_cuda.html)|Yes|-|
|[Tensor.is_quantized](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_quantized.html)|Yes|-|
|[Tensor.is_meta](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_meta.html)|Yes|-|
|[Tensor.device](https://pytorch.org/docs/2.9/generated/torch.Tensor.device.html)|Yes|-|
|[Tensor.grad](https://pytorch.org/docs/2.9/generated/torch.Tensor.grad.html)|Yes|Supports fp32|
|[Tensor.ndim](https://pytorch.org/docs/2.9/generated/torch.Tensor.ndim.html)|Yes|Supports fp32|
|[Tensor.real](https://pytorch.org/docs/2.9/generated/torch.Tensor.real.html)|Yes|-|
|[Tensor.imag](https://pytorch.org/docs/2.9/generated/torch.Tensor.imag.html)|Yes|-|
|[Tensor.nbytes](https://pytorch.org/docs/2.9/generated/torch.Tensor.nbytes.html)|Yes|-|
|[Tensor.itemsize](https://pytorch.org/docs/2.9/generated/torch.Tensor.itemsize.html)|Yes|-|
|[Tensor.abs](https://pytorch.org/docs/2.9/generated/torch.Tensor.abs.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.abs_](https://pytorch.org/docs/2.9/generated/torch.Tensor.abs_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.absolute](https://pytorch.org/docs/2.9/generated/torch.Tensor.absolute.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.absolute_](https://pytorch.org/docs/2.9/generated/torch.Tensor.absolute_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.acos](https://pytorch.org/docs/2.9/generated/torch.Tensor.acos.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU execution|
|[Tensor.acos_](https://pytorch.org/docs/2.9/generated/torch.Tensor.acos_.html)|Yes|Supports fp16, fp32|
|[Tensor.arccos](https://pytorch.org/docs/2.9/generated/torch.Tensor.arccos.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.arccos_](https://pytorch.org/docs/2.9/generated/torch.Tensor.arccos_.html)|Yes|Supports fp16, fp32|
|[Tensor.add](https://pytorch.org/docs/2.9/generated/torch.Tensor.add.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.add_](https://pytorch.org/docs/2.9/generated/torch.Tensor.add_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.addbmm](https://pytorch.org/docs/2.9/generated/torch.Tensor.addbmm.html)|Yes|Supports fp16, fp32|
|[Tensor.addbmm_](https://pytorch.org/docs/2.9/generated/torch.Tensor.addbmm_.html)|Yes|Supports fp16, fp32|
|[Tensor.addcdiv](https://pytorch.org/docs/2.9/generated/torch.Tensor.addcdiv.html)|Yes|Supports bf16, fp16, fp32, int64<br>int64 does not support broadcasting three tensors simultaneously|
|[Tensor.addcdiv_](https://pytorch.org/docs/2.9/generated/torch.Tensor.addcdiv_.html)|Yes|<term>Atlas A2 Training Series</term>/<term>Atlas A3 Training Series</term>: Supports bf16, fp16, fp32, fp64<br><term>Atlas Training Series</term>: Supports fp16, fp32, fp64<br>int64 does not support broadcasting three tensors simultaneously|
|[Tensor.addcmul](https://pytorch.org/docs/2.9/generated/torch.Tensor.addcmul.html)|Yes|Supports fp16, fp32, int64<br>int64 does not support broadcasting three tensors simultaneously|
|[Tensor.addcmul_](https://pytorch.org/docs/2.9/generated/torch.Tensor.addcmul_.html)|Yes|<term>Atlas A2 Training Series</term><term>Atlas A3 Training Series</term>: Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64<br><term>Atlas Training Series</term>: Supports fp16, fp32, fp64, uint8, int8, int32, int64<br>int64 does not support broadcasting three tensors simultaneously|
|[Tensor.addmm](https://pytorch.org/docs/2.9/generated/torch.Tensor.addmm.html)|Yes|Supports fp16, fp32|
|[Tensor.addmm_](https://pytorch.org/docs/2.9/generated/torch.Tensor.addmm_.html)|Yes|Supports fp16, fp32|
|[Tensor.sspaddmm](https://pytorch.org/docs/2.9/generated/torch.Tensor.sspaddmm.html)|No|-|
|[Tensor.addmv](https://pytorch.org/docs/2.9/generated/torch.Tensor.addmv.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.addmv_](https://pytorch.org/docs/2.9/generated/torch.Tensor.addmv_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.addr](https://pytorch.org/docs/2.9/generated/torch.Tensor.addr.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.addr_](https://pytorch.org/docs/2.9/generated/torch.Tensor.addr_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.adjoint](https://pytorch.org/docs/2.9/generated/torch.Tensor.adjoint.html)|Yes|-|
|[Tensor.allclose](https://pytorch.org/docs/2.9/generated/torch.Tensor.allclose.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.amax](https://pytorch.org/docs/2.9/generated/torch.Tensor.amax.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.amin](https://pytorch.org/docs/2.9/generated/torch.Tensor.amin.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.aminmax](https://pytorch.org/docs/2.9/generated/torch.Tensor.aminmax.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.angle](https://pytorch.org/docs/2.9/generated/torch.Tensor.angle.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|[Tensor.apply_](https://pytorch.org/docs/2.9/generated/torch.Tensor.apply_.html)|Yes|CPU Only|
|[Tensor.argmax](https://pytorch.org/docs/2.9/generated/torch.Tensor.argmax.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.argmin](https://pytorch.org/docs/2.9/generated/torch.Tensor.argmin.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.argsort](https://pytorch.org/docs/2.9/generated/torch.Tensor.argsort.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.argwhere](https://pytorch.org/docs/2.9/generated/torch.Tensor.argwhere.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.asin](https://pytorch.org/docs/2.9/generated/torch.Tensor.asin.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.asin_](https://pytorch.org/docs/2.9/generated/torch.Tensor.asin_.html)|Yes|Supports fp16, fp32|
|[Tensor.arcsin](https://pytorch.org/docs/2.9/generated/torch.Tensor.arcsin.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.arcsin_](https://pytorch.org/docs/2.9/generated/torch.Tensor.arcsin_.html)|Yes|Supports fp16, fp32|
|[Tensor.as_strided](https://pytorch.org/docs/2.9/generated/torch.Tensor.as_strided.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.atan](https://pytorch.org/docs/2.9/generated/torch.Tensor.atan.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.atan_](https://pytorch.org/docs/2.9/generated/torch.Tensor.atan_.html)|Yes|Supports fp16, fp32|
|[Tensor.arctan](https://pytorch.org/docs/2.9/generated/torch.Tensor.arctan.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.arctan_](https://pytorch.org/docs/2.9/generated/torch.Tensor.arctan_.html)|Yes|Supports fp16, fp32|
|[Tensor.atan2](https://pytorch.org/docs/2.9/generated/torch.Tensor.atan2.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.atan2_](https://pytorch.org/docs/2.9/generated/torch.Tensor.atan2_.html)|Yes|Supports fp16, fp32<br>May fall back to CPU execution|
|[Tensor.arctan2](https://pytorch.org/docs/2.9/generated/torch.Tensor.arctan2.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.arctan2_](https://pytorch.org/docs/2.9/generated/torch.Tensor.arctan2_.html)|Yes|Supports fp16, fp32|
|[Tensor.all](https://pytorch.org/docs/2.9/generated/torch.Tensor.all.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.any](https://pytorch.org/docs/2.9/generated/torch.Tensor.any.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.backward](https://pytorch.org/docs/2.9/generated/torch.Tensor.backward.html)|Yes|Supports fp32|
|[Tensor.baddbmm](https://pytorch.org/docs/2.9/generated/torch.Tensor.baddbmm.html)|Yes|Supports fp16, fp32|
|[Tensor.baddbmm_](https://pytorch.org/docs/2.9/generated/torch.Tensor.baddbmm_.html)|Yes|Supports fp16, fp32|
|[Tensor.bernoulli](https://pytorch.org/docs/2.9/generated/torch.Tensor.bernoulli.html)|Yes|Supports fp16, fp32<br>May fall back to CPU execution|
|[Tensor.bernoulli_](https://pytorch.org/docs/2.9/generated/torch.Tensor.bernoulli_.html)|Yes|May fall back to CPU execution|
|[Tensor.bfloat16](https://pytorch.org/docs/2.9/generated/torch.Tensor.bfloat16.html)|Yes|Supports fp16, fp32|
|[Tensor.bincount](https://pytorch.org/docs/2.9/generated/torch.Tensor.bincount.html)|Yes|Supports uint8, int8, int16, int32, int64<br>The weights dimension must be consistent with the input dimension|
|[Tensor.bitwise_not](https://pytorch.org/docs/2.9/generated/torch.Tensor.bitwise_not.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_not_](https://pytorch.org/docs/2.9/generated/torch.Tensor.bitwise_not_.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_and](https://pytorch.org/docs/2.9/generated/torch.Tensor.bitwise_and.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_and_](https://pytorch.org/docs/2.9/generated/torch.Tensor.bitwise_and_.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_or](https://pytorch.org/docs/2.9/generated/torch.Tensor.bitwise_or.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_or_](https://pytorch.org/docs/2.9/generated/torch.Tensor.bitwise_or_.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_xor](https://pytorch.org/docs/2.9/generated/torch.Tensor.bitwise_xor.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bitwise_xor_](https://pytorch.org/docs/2.9/generated/torch.Tensor.bitwise_xor_.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[Tensor.bmm](https://pytorch.org/docs/2.9/generated/torch.Tensor.bmm.html)|Yes|Supports fp16, fp32|
|[Tensor.bool](https://pytorch.org/docs/2.9/generated/torch.Tensor.bool.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.byte](https://pytorch.org/docs/2.9/generated/torch.Tensor.byte.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.broadcast_to](https://pytorch.org/docs/2.9/generated/torch.Tensor.broadcast_to.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.ceil](https://pytorch.org/docs/2.9/generated/torch.Tensor.ceil.html)|Yes|Supports fp16, fp32|
|[Tensor.ceil_](https://pytorch.org/docs/2.9/generated/torch.Tensor.ceil_.html)|Yes|Supports fp16, fp32|
|[Tensor.char](https://pytorch.org/docs/2.9/generated/torch.Tensor.char.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.chunk](https://pytorch.org/docs/2.9/generated/torch.Tensor.chunk.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.clamp](https://pytorch.org/docs/2.9/generated/torch.Tensor.clamp.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.clamp_](https://pytorch.org/docs/2.9/generated/torch.Tensor.clamp_.html)|Yes|May fall back to CPU execution|
|[Tensor.clip](https://pytorch.org/docs/2.9/generated/torch.Tensor.clip.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.clip_](https://pytorch.org/docs/2.9/generated/torch.Tensor.clip_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.clone](https://pytorch.org/docs/2.9/generated/torch.Tensor.clone.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.contiguous](https://pytorch.org/docs/2.9/generated/torch.Tensor.contiguous.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.copy_](https://pytorch.org/docs/2.9/generated/torch.Tensor.copy_.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>int16 does not support 5 or more dimensions|
|[Tensor.conj](https://pytorch.org/docs/2.9/generated/torch.Tensor.conj.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.resolve_conj](https://pytorch.org/docs/2.9/generated/torch.Tensor.resolve_conj.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.resolve_neg](https://pytorch.org/docs/2.9/generated/torch.Tensor.resolve_neg.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.copysign](https://pytorch.org/docs/2.9/generated/torch.Tensor.copysign.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU execution|
|[Tensor.cos](https://pytorch.org/docs/2.9/generated/torch.Tensor.cos.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.cos_](https://pytorch.org/docs/2.9/generated/torch.Tensor.cos_.html)|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|[Tensor.cosh](https://pytorch.org/docs/2.9/generated/torch.Tensor.cosh.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.cosh_](https://pytorch.org/docs/2.9/generated/torch.Tensor.cosh_.html)|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.count_nonzero](https://pytorch.org/docs/2.9/generated/torch.Tensor.count_nonzero.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.cov](https://pytorch.org/docs/2.9/generated/torch.Tensor.cov.html)|Yes|Supports fp16, fp32, int16, int32, int64|
|[Tensor.acosh](https://pytorch.org/docs/2.9/generated/torch.Tensor.acosh.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May fall back to CPU execution|
|[Tensor.acosh_](https://pytorch.org/docs/2.9/generated/torch.Tensor.acosh_.html)|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.arccosh](https://pytorch.org/docs/2.9/generated/torch.Tensor.arccosh.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.arccosh_](https://pytorch.org/docs/2.9/generated/torch.Tensor.arccosh_.html)|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.cpu](https://pytorch.org/docs/2.9/generated/torch.Tensor.cpu.html)|Yes|-|
|[Tensor.cross](https://pytorch.org/docs/2.9/generated/torch.Tensor.cross.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128<br>The shapes of the two inputs must be consistent|
|[Tensor.cuda](https://pytorch.org/docs/2.9/generated/torch.Tensor.cuda.html)|Yes|The NPU corresponding interface is Tensor.npu, and memory_format only supports passing torch.contiguous_format|
|[Tensor.cummax](https://pytorch.org/docs/2.9/generated/torch.Tensor.cummax.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU execution|
|[Tensor.cummin](https://pytorch.org/docs/2.9/generated/torch.Tensor.cummin.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.cumsum](https://pytorch.org/docs/2.9/generated/torch.Tensor.cumsum.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Supports Named Tensor|
|[Tensor.cumsum_](https://pytorch.org/docs/2.9/generated/torch.Tensor.cumsum_.html)|Yes|Supports fp16, fp32, int64, bool|
|[Tensor.chalf](https://pytorch.org/docs/2.9/generated/torch.Tensor.chalf.html)|No|-|
|[Tensor.cfloat](https://pytorch.org/docs/2.9/generated/torch.Tensor.cfloat.html)|Yes|-|
|[Tensor.cdouble](https://pytorch.org/docs/2.9/generated/torch.Tensor.cdouble.html)|Yes|-|
|[Tensor.data_ptr](https://pytorch.org/docs/2.9/generated/torch.Tensor.data_ptr.html)|Yes|Supports fp32|
|[Tensor.deg2rad](https://pytorch.org/docs/2.9/generated/torch.Tensor.deg2rad.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.dequantize](https://pytorch.org/docs/2.9/generated/torch.Tensor.dequantize.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.dense_dim](https://pytorch.org/docs/2.9/generated/torch.Tensor.dense_dim.html)|Yes|-|
|[Tensor.detach](https://pytorch.org/docs/2.9/generated/torch.Tensor.detach.html)|Yes|Supports fp32|
|[Tensor.detach_](https://pytorch.org/docs/2.9/generated/torch.Tensor.detach_.html)|Yes|Supports fp32|
|[Tensor.diag](https://pytorch.org/docs/2.9/generated/torch.Tensor.diag.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|[Tensor.diag_embed](https://pytorch.org/docs/2.9/generated/torch.Tensor.diag_embed.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.diagflat](https://pytorch.org/docs/2.9/generated/torch.Tensor.diagflat.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|[Tensor.diagonal](https://pytorch.org/docs/2.9/generated/torch.Tensor.diagonal.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.diagonal_scatter](https://pytorch.org/docs/2.9/generated/torch.Tensor.diagonal_scatter.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.fill_diagonal_](https://pytorch.org/docs/2.9/generated/torch.Tensor.fill_diagonal_.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.diff](https://pytorch.org/docs/2.9/generated/torch.Tensor.diff.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.dim](https://pytorch.org/docs/2.9/generated/torch.Tensor.dim.html)|Yes|Supports fp32|
|[Tensor.dim_order](https://pytorch.org/docs/2.9/generated/torch.Tensor.dim_order.html)|Yes|-|
|[Tensor.dist](https://pytorch.org/docs/2.9/generated/torch.Tensor.dist.html)|Yes|-|
|[Tensor.div](https://pytorch.org/docs/2.9/generated/torch.Tensor.div.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.div_](https://pytorch.org/docs/2.9/generated/torch.Tensor.div_.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[Tensor.divide](https://pytorch.org/docs/2.9/generated/torch.Tensor.divide.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.divide_](https://pytorch.org/docs/2.9/generated/torch.Tensor.divide_.html)|Yes|Supports fp16, fp32|
|[Tensor.dot](https://pytorch.org/docs/2.9/generated/torch.Tensor.dot.html)|Yes|Supports fp16, fp32|
|[Tensor.double](https://pytorch.org/docs/2.9/generated/torch.Tensor.double.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Currently some NPU interfaces do not support the double type. For compatibility, fp32 is returned by default. fp64 will be returned normally after support is completed in the future.<br>May fall back to CPU execution|
|[Tensor.dsplit](https://pytorch.org/docs/2.9/generated/torch.Tensor.dsplit.html)|Yes|Supports fp32|
|[Tensor.element_size](https://pytorch.org/docs/2.9/generated/torch.Tensor.element_size.html)|Yes|Supports fp32|
|[Tensor.eq](https://pytorch.org/docs/2.9/generated/torch.Tensor.eq.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.eq_](https://pytorch.org/docs/2.9/generated/torch.Tensor.eq_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.equal](https://pytorch.org/docs/2.9/generated/torch.Tensor.equal.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.erf](https://pytorch.org/docs/2.9/generated/torch.Tensor.erf.html)|Yes|Supports fp16, fp32, int64, bool|
|[Tensor.erf_](https://pytorch.org/docs/2.9/generated/torch.Tensor.erf_.html)|Yes|Supports fp16, fp32|
|[Tensor.erfc](https://pytorch.org/docs/2.9/generated/torch.Tensor.erfc.html)|Yes|Supports fp16, fp32, int64, bool|
|[Tensor.erfc_](https://pytorch.org/docs/2.9/generated/torch.Tensor.erfc_.html)|Yes|Supports fp16, fp32|
|[Tensor.erfinv](https://pytorch.org/docs/2.9/generated/torch.Tensor.erfinv.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.erfinv_](https://pytorch.org/docs/2.9/generated/torch.Tensor.erfinv_.html)|Yes|Supports bf16, fp16, fp32|
|[Tensor.exp](https://pytorch.org/docs/2.9/generated/torch.Tensor.exp.html)|Yes|Supports bf16, fp16, fp32, int64, bool, complex64, complex128|
|[Tensor.exp_](https://pytorch.org/docs/2.9/generated/torch.Tensor.exp_.html)|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|[Tensor.expm1](https://pytorch.org/docs/2.9/generated/torch.Tensor.expm1.html)|Yes|Supports fp16, fp32, int64, bool|
|[Tensor.expm1_](https://pytorch.org/docs/2.9/generated/torch.Tensor.expm1_.html)|Yes|Supports fp16, fp32|
|[Tensor.expand](https://pytorch.org/docs/2.9/generated/torch.Tensor.expand.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.expand_as](https://pytorch.org/docs/2.9/generated/torch.Tensor.expand_as.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.exponential_](https://pytorch.org/docs/2.9/generated/torch.Tensor.exponential_.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[Tensor.fix](https://pytorch.org/docs/2.9/generated/torch.Tensor.fix.html)|Yes|Supports fp16, fp32|
|[Tensor.fix_](https://pytorch.org/docs/2.9/generated/torch.Tensor.fix_.html)|Yes|Supports fp16, fp32|
|[Tensor.fill_](https://pytorch.org/docs/2.9/generated/torch.Tensor.fill_.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.flatten](https://pytorch.org/docs/2.9/generated/torch.Tensor.flatten.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.flip](https://pytorch.org/docs/2.9/generated/torch.Tensor.flip.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.fliplr](https://pytorch.org/docs/2.9/generated/torch.Tensor.fliplr.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.flipud](https://pytorch.org/docs/2.9/generated/torch.Tensor.flipud.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.float](https://pytorch.org/docs/2.9/generated/torch.Tensor.float.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.float_power](https://pytorch.org/docs/2.9/generated/torch.Tensor.float_power.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex128|
|[Tensor.float_power_](https://pytorch.org/docs/2.9/generated/torch.Tensor.float_power_.html)|Yes|Supports double|
|[Tensor.floor](https://pytorch.org/docs/2.9/generated/torch.Tensor.floor.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.floor_](https://pytorch.org/docs/2.9/generated/torch.Tensor.floor_.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.floor_divide](https://pytorch.org/docs/2.9/generated/torch.Tensor.floor_divide.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.floor_divide_](https://pytorch.org/docs/2.9/generated/torch.Tensor.floor_divide_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.fmod](https://pytorch.org/docs/2.9/generated/torch.Tensor.fmod.html)|Yes|Supports fp16, fp32, uint8, int8, int32, int64|
|[Tensor.fmod_](https://pytorch.org/docs/2.9/generated/torch.Tensor.fmod_.html)|Yes|Supports fp16, fp32, uint8, int8, int32, int64|
|[Tensor.frac](https://pytorch.org/docs/2.9/generated/torch.Tensor.frac.html)|Yes|Supports fp16, fp32|
|[Tensor.frac_](https://pytorch.org/docs/2.9/generated/torch.Tensor.frac_.html)|Yes|Supports fp16, fp32|
|[Tensor.gather](https://pytorch.org/docs/2.9/generated/torch.Tensor.gather.html)|Yes|Supports fp16, fp32, int64<br>The index dimension must be consistent with the input dimension|
|[Tensor.ge](https://pytorch.org/docs/2.9/generated/torch.Tensor.ge.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.ge_](https://pytorch.org/docs/2.9/generated/torch.Tensor.ge_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.greater_equal](https://pytorch.org/docs/2.9/generated/torch.Tensor.greater_equal.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.greater_equal_](https://pytorch.org/docs/2.9/generated/torch.Tensor.greater_equal_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.geometric_](https://pytorch.org/docs/2.9/generated/torch.Tensor.geometric_.html)|Yes|-|
|[Tensor.ger](https://pytorch.org/docs/2.9/generated/torch.Tensor.ger.html)|Yes|-|
|[Tensor.get_device](https://pytorch.org/docs/2.9/generated/torch.Tensor.get_device.html)|Yes|-|
|[Tensor.gt](https://pytorch.org/docs/2.9/generated/torch.Tensor.gt.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.gt_](https://pytorch.org/docs/2.9/generated/torch.Tensor.gt_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.greater](https://pytorch.org/docs/2.9/generated/torch.Tensor.greater.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.greater_](https://pytorch.org/docs/2.9/generated/torch.Tensor.greater_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.half](https://pytorch.org/docs/2.9/generated/torch.Tensor.half.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.hardshrink](https://pytorch.org/docs/2.9/generated/torch.Tensor.hardshrink.html)|Yes|Supports fp16, fp32|
|[Tensor.heaviside](https://pytorch.org/docs/2.9/generated/torch.Tensor.heaviside.html)|Yes|May fall back to CPU execution|
|[Tensor.histc](https://pytorch.org/docs/2.9/generated/torch.Tensor.histc.html)|Yes|Supports fp16, fp32|
|[Tensor.hsplit](https://pytorch.org/docs/2.9/generated/torch.Tensor.hsplit.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.index_add_](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_add_.html)|Yes|Supports fp16, fp32, int64, bool|
|[Tensor.index_add](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_add.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.index_copy_](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_copy_.html)|Yes|Supports fp16, fp32, int16, int32, bool|
|[Tensor.index_copy](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_copy.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.index_fill_](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_fill_.html)|Yes|Supports fp16, fp32, int32, int64, bool|
|[Tensor.index_fill](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_fill.html)|Yes|Supports fp16, fp32, int32, bool|
|[Tensor.index_put_](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_put_.html)|Yes|Supports int64|
|[Tensor.index_put](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_put.html)|Yes|Supports int64|
|[Tensor.index_reduce_](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_reduce_.html)|Yes|May fall back to CPU execution|
|[Tensor.index_reduce](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_reduce.html)|Yes|May fall back to CPU execution|
|[Tensor.index_select](https://pytorch.org/docs/2.9/generated/torch.Tensor.index_select.html)|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|[Tensor.indices](https://pytorch.org/docs/2.9/generated/torch.Tensor.indices.html)|Yes|-|
|[Tensor.inner](https://pytorch.org/docs/2.9/generated/torch.Tensor.inner.html)|Yes|Supports fp16, fp32|
|[Tensor.int](https://pytorch.org/docs/2.9/generated/torch.Tensor.int.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.int_repr](https://pytorch.org/docs/2.9/generated/torch.Tensor.int_repr.html)|No|-|
|[Tensor.isclose](https://pytorch.org/docs/2.9/generated/torch.Tensor.isclose.html)|Yes|Supports fp16, fp32, uint8, int32, int64, bool|
|[Tensor.isfinite](https://pytorch.org/docs/2.9/generated/torch.Tensor.isfinite.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.isinf](https://pytorch.org/docs/2.9/generated/torch.Tensor.isinf.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.isposinf](https://pytorch.org/docs/2.9/generated/torch.Tensor.isposinf.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.isneginf](https://pytorch.org/docs/2.9/generated/torch.Tensor.isneginf.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.isnan](https://pytorch.org/docs/2.9/generated/torch.Tensor.isnan.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.is_contiguous](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_contiguous.html)|Yes|Supports fp32|
|[Tensor.is_complex](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_complex.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.is_conj](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_conj.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.is_floating_point](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_floating_point.html)|Yes|Supports fp32|
|[Tensor.is_inference](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_inference.html)|Yes|Supports fp32|
|[Tensor.is_leaf](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_leaf.html)|Yes|-|
|[Tensor.is_pinned](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_pinned.html)|Yes|Supports fp32|
|[Tensor.is_set_to](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_set_to.html)|Yes|Supports fp32|
|[Tensor.is_shared](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_shared.html)|No|-|
|[Tensor.is_signed](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_signed.html)|Yes|Supports fp32|
|[Tensor.is_sparse](https://pytorch.org/docs/2.9/generated/torch.Tensor.is_sparse.html)|Yes|Supports fp32|
|[Tensor.isreal](https://pytorch.org/docs/2.9/generated/torch.Tensor.isreal.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.item](https://pytorch.org/docs/2.9/generated/torch.Tensor.item.html)|Yes|Supports fp32|
|[Tensor.kthvalue](https://pytorch.org/docs/2.9/generated/torch.Tensor.kthvalue.html)|Yes|Supports fp16, fp32, int32|
|[Tensor.ldexp](https://pytorch.org/docs/2.9/generated/torch.Tensor.ldexp.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.ldexp_](https://pytorch.org/docs/2.9/generated/torch.Tensor.ldexp_.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[Tensor.le](https://pytorch.org/docs/2.9/generated/torch.Tensor.le.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.le_](https://pytorch.org/docs/2.9/generated/torch.Tensor.le_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.less_equal](https://pytorch.org/docs/2.9/generated/torch.Tensor.less_equal.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.less_equal_](https://pytorch.org/docs/2.9/generated/torch.Tensor.less_equal_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.lerp](https://pytorch.org/docs/2.9/generated/torch.Tensor.lerp.html)|Yes|Supports fp16, fp32|
|[Tensor.lerp_](https://pytorch.org/docs/2.9/generated/torch.Tensor.lerp_.html)|Yes|Supports fp16, fp32|
|[Tensor.log](https://pytorch.org/docs/2.9/generated/torch.Tensor.log.html)|Yes|Supports bf16, fp16, fp32, int64, bool, complex64, complex128|
|[Tensor.log_](https://pytorch.org/docs/2.9/generated/torch.Tensor.log_.html)|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|[Tensor.log10](https://pytorch.org/docs/2.9/generated/torch.Tensor.log10.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.log10_](https://pytorch.org/docs/2.9/generated/torch.Tensor.log10_.html)|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.log1p](https://pytorch.org/docs/2.9/generated/torch.Tensor.log1p.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.log1p_](https://pytorch.org/docs/2.9/generated/torch.Tensor.log1p_.html)|Yes|Supports fp16, fp32|
|[Tensor.log2](https://pytorch.org/docs/2.9/generated/torch.Tensor.log2.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.log2_](https://pytorch.org/docs/2.9/generated/torch.Tensor.log2_.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[Tensor.logaddexp](https://pytorch.org/docs/2.9/generated/torch.Tensor.logaddexp.html)|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|[Tensor.logaddexp2](https://pytorch.org/docs/2.9/generated/torch.Tensor.logaddexp2.html)|Yes|Supports fp16, fp32|
|[Tensor.logsumexp](https://pytorch.org/docs/2.9/generated/torch.Tensor.logsumexp.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.logical_and](https://pytorch.org/docs/2.9/generated/torch.Tensor.logical_and.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.logical_and_](https://pytorch.org/docs/2.9/generated/torch.Tensor.logical_and_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.logical_not](https://pytorch.org/docs/2.9/generated/torch.Tensor.logical_not.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.logical_not_](https://pytorch.org/docs/2.9/generated/torch.Tensor.logical_not_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.logical_or](https://pytorch.org/docs/2.9/generated/torch.Tensor.logical_or.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.logical_or_](https://pytorch.org/docs/2.9/generated/torch.Tensor.logical_or_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.logical_xor](https://pytorch.org/docs/2.9/generated/torch.Tensor.logical_xor.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May fall back to CPU execution|
|[Tensor.logical_xor_](https://pytorch.org/docs/2.9/generated/torch.Tensor.logical_xor_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.logit](https://pytorch.org/docs/2.9/generated/torch.Tensor.logit.html)|Yes|Supports bf16, fp16, fp32<br>Output is nan when eps is greater than 1, output is inf when eps is 1|
|[Tensor.logit_](https://pytorch.org/docs/2.9/generated/torch.Tensor.logit_.html)|Yes|Supports bf16, fp16, fp32<br>Output is nan when eps is greater than 1, output is inf when eps is 1|
|[Tensor.long](https://pytorch.org/docs/2.9/generated/torch.Tensor.long.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.lt](https://pytorch.org/docs/2.9/generated/torch.Tensor.lt.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.lt_](https://pytorch.org/docs/2.9/generated/torch.Tensor.lt_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.less](https://pytorch.org/docs/2.9/generated/torch.Tensor.less.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.less_](https://pytorch.org/docs/2.9/generated/torch.Tensor.less_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.as_subclass](https://pytorch.org/docs/2.9/generated/torch.Tensor.as_subclass.html)|Yes|-|
|[Tensor.map_](https://pytorch.org/docs/2.9/generated/torch.Tensor.map_.html)|Yes|CPU Only|
|[Tensor.masked_scatter_](https://pytorch.org/docs/2.9/generated/torch.Tensor.masked_scatter_.html)|Yes|Supports fp32, int64, bool|
|[Tensor.masked_scatter](https://pytorch.org/docs/2.9/generated/torch.Tensor.masked_scatter.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.masked_fill_](https://pytorch.org/docs/2.9/generated/torch.Tensor.masked_fill_.html)|Yes|Supports bf16, fp16, fp32, int8, int32, int64, bool|
|[Tensor.masked_fill](https://pytorch.org/docs/2.9/generated/torch.Tensor.masked_fill.html)|Yes|Supports fp16, fp32, int64, bool|
|[Tensor.masked_select](https://pytorch.org/docs/2.9/generated/torch.Tensor.masked_select.html)|Yes|Supports fp32, bool|
|[Tensor.matmul](https://pytorch.org/docs/2.9/generated/torch.Tensor.matmul.html)|Yes|Supports bf16, fp16, fp32<br>Supports Named Tensor|
|[Tensor.matrix_power](https://pytorch.org/docs/2.9/generated/torch.Tensor.matrix_power.html)|Yes|Supports fp16, fp32|
|[Tensor.max](https://pytorch.org/docs/2.9/generated/torch.Tensor.max.html)|Yes|Supports bf16, fp16, fp32, int64, bool|
|[Tensor.maximum](https://pytorch.org/docs/2.9/generated/torch.Tensor.maximum.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.mean](https://pytorch.org/docs/2.9/generated/torch.Tensor.mean.html)|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|[Tensor.nanmean](https://pytorch.org/docs/2.9/generated/torch.Tensor.nanmean.html)|Yes|Supports fp16, fp32|
|[Tensor.median](https://pytorch.org/docs/2.9/generated/torch.Tensor.median.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64<br>When input is bf16, dim does not take the dimension where the input axis value is 1|
|[Tensor.min](https://pytorch.org/docs/2.9/generated/torch.Tensor.min.html)|Yes|Supports bf16, fp16, fp32, int64, bool|
|[Tensor.minimum](https://pytorch.org/docs/2.9/generated/torch.Tensor.minimum.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.mm](https://pytorch.org/docs/2.9/generated/torch.Tensor.mm.html)|Yes|Supports fp16, fp32|
|[Tensor.smm](https://pytorch.org/docs/2.9/generated/torch.Tensor.smm.html)|No|-|
|[Tensor.mode](https://pytorch.org/docs/2.9/generated/torch.Tensor.mode.html)|No|-|
|[Tensor.movedim](https://pytorch.org/docs/2.9/generated/torch.Tensor.movedim.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.moveaxis](https://pytorch.org/docs/2.9/generated/torch.Tensor.moveaxis.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.msort](https://pytorch.org/docs/2.9/generated/torch.Tensor.msort.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.mul](https://pytorch.org/docs/2.9/generated/torch.Tensor.mul.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.mul_](https://pytorch.org/docs/2.9/generated/torch.Tensor.mul_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.multiply](https://pytorch.org/docs/2.9/generated/torch.Tensor.multiply.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.multiply_](https://pytorch.org/docs/2.9/generated/torch.Tensor.multiply_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.multinomial](https://pytorch.org/docs/2.9/generated/torch.Tensor.multinomial.html)|Yes|Supports fp16, fp32|
|[Tensor.nansum](https://pytorch.org/docs/2.9/generated/torch.Tensor.nansum.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.narrow](https://pytorch.org/docs/2.9/generated/torch.Tensor.narrow.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.narrow_copy](https://pytorch.org/docs/2.9/generated/torch.Tensor.narrow_copy.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.ndimension](https://pytorch.org/docs/2.9/generated/torch.Tensor.ndimension.html)|Yes|Supports fp32|
|[Tensor.nan_to_num](https://pytorch.org/docs/2.9/generated/torch.Tensor.nan_to_num.html)|Yes|-|
|[Tensor.nan_to_num_](https://pytorch.org/docs/2.9/generated/torch.Tensor.nan_to_num_.html)|Yes|-|
|[Tensor.ne](https://pytorch.org/docs/2.9/generated/torch.Tensor.ne.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.ne_](https://pytorch.org/docs/2.9/generated/torch.Tensor.ne_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.nextafter_](https://pytorch.org/docs/2.9/generated/torch.Tensor.nextafter_.html)|Yes|Falls back to CPU execution|
|[Tensor.not_equal](https://pytorch.org/docs/2.9/generated/torch.Tensor.not_equal.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May fall back to CPU execution|
|[Tensor.not_equal_](https://pytorch.org/docs/2.9/generated/torch.Tensor.not_equal_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.neg](https://pytorch.org/docs/2.9/generated/torch.Tensor.neg.html)|Yes|Supports fp16, fp32, int8, int32, int64, complex64, complex128|
|[Tensor.neg_](https://pytorch.org/docs/2.9/generated/torch.Tensor.neg_.html)|Yes|Supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128<br>May fall back to CPU execution|
|[Tensor.negative](https://pytorch.org/docs/2.9/generated/torch.Tensor.negative.html)|Yes|Supports fp16, fp32, int8, int32, int64, complex64, complex128|
|[Tensor.negative_](https://pytorch.org/docs/2.9/generated/torch.Tensor.negative_.html)|Yes|Supports fp16, fp32, int8, int32, int64, complex64, complex128|
|[Tensor.nelement](https://pytorch.org/docs/2.9/generated/torch.Tensor.nelement.html)|Yes|Supports fp32|
|[Tensor.nonzero](https://pytorch.org/docs/2.9/generated/torch.Tensor.nonzero.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>NaN scenarios are unsupported|
|[Tensor.norm](https://pytorch.org/docs/2.9/generated/torch.Tensor.norm.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[Tensor.normal_](https://pytorch.org/docs/2.9/generated/torch.Tensor.normal_.html)|Yes|Supports bf16, fp16, fp32<br>May fall back to CPU execution|
|[Tensor.numel](https://pytorch.org/docs/2.9/generated/torch.Tensor.numel.html)|Yes|Supports fp32|
|[Tensor.numpy](https://pytorch.org/docs/2.9/generated/torch.Tensor.numpy.html)|Yes|Supports fp32|
|[Tensor.outer](https://pytorch.org/docs/2.9/generated/torch.Tensor.outer.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.permute](https://pytorch.org/docs/2.9/generated/torch.Tensor.permute.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.positive](https://pytorch.org/docs/2.9/generated/torch.Tensor.positive.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|[Tensor.pow](https://pytorch.org/docs/2.9/generated/torch.Tensor.pow.html)|Yes|Supports bf16, fp16, fp32, fp64, int16, int64|
|[Tensor.pow_](https://pytorch.org/docs/2.9/generated/torch.Tensor.pow_.html)|Yes|Supports bf16, fp16, fp32, fp64, int64|
|[Tensor.prod](https://pytorch.org/docs/2.9/generated/torch.Tensor.prod.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.put_](https://pytorch.org/docs/2.9/generated/torch.Tensor.put_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|[Tensor.qscheme](https://pytorch.org/docs/2.9/generated/torch.Tensor.qscheme.html)|No|-|
|[Tensor.quantile](https://pytorch.org/docs/2.9/generated/torch.Tensor.quantile.html)|Yes|-|
|[Tensor.rad2deg](https://pytorch.org/docs/2.9/generated/torch.Tensor.rad2deg.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.random_](https://pytorch.org/docs/2.9/generated/torch.Tensor.random_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.ravel](https://pytorch.org/docs/2.9/generated/torch.Tensor.ravel.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.reciprocal](https://pytorch.org/docs/2.9/generated/torch.Tensor.reciprocal.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.reciprocal_](https://pytorch.org/docs/2.9/generated/torch.Tensor.reciprocal_.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[Tensor.record_stream](https://pytorch.org/docs/2.9/generated/torch.Tensor.record_stream.html)|Yes|Supports fp32|
|[Tensor.register_hook](https://pytorch.org/docs/2.9/generated/torch.Tensor.register_hook.html)|Yes|Supports fp32|
|[Tensor.register_post_accumulate_grad_hook](https://pytorch.org/docs/2.9/generated/torch.Tensor.register_post_accumulate_grad_hook.html)|Yes|-|
|[Tensor.remainder](https://pytorch.org/docs/2.9/generated/torch.Tensor.remainder.html)|Yes|Supports fp16, fp32, int32, int64|
|[Tensor.remainder_](https://pytorch.org/docs/2.9/generated/torch.Tensor.remainder_.html)|Yes|Supports fp16, fp32, int32, int64|
|[Tensor.repeat](https://pytorch.org/docs/2.9/generated/torch.Tensor.repeat.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.repeat_interleave](https://pytorch.org/docs/2.9/generated/torch.Tensor.repeat_interleave.html)|Yes|Supports fp16, fp32, int16, int32, bool<br>The input tensor is repeated to produce the output, and the number of elements in the output must be less than $2^{22}$|
|[Tensor.requires_grad](https://pytorch.org/docs/2.9/generated/torch.Tensor.requires_grad.html)|Yes|-|
|[Tensor.requires_grad_](https://pytorch.org/docs/2.9/generated/torch.Tensor.requires_grad_.html)|Yes|Supports fp32|
|[Tensor.reshape](https://pytorch.org/docs/2.9/generated/torch.Tensor.reshape.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.reshape_as](https://pytorch.org/docs/2.9/generated/torch.Tensor.reshape_as.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.resize_](https://pytorch.org/docs/2.9/generated/torch.Tensor.resize_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>memory_format only supports torch.contiguous_format and torch.preserve_format|
|[Tensor.resize_as_](https://pytorch.org/docs/2.9/generated/torch.Tensor.resize_as_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>memory_format only supports torch.contiguous_format and torch.preserve_format|
|[Tensor.retain_grad](https://pytorch.org/docs/2.9/generated/torch.Tensor.retain_grad.html)|Yes|Supports fp32|
|[Tensor.retains_grad](https://pytorch.org/docs/2.9/generated/torch.Tensor.retains_grad.html)|Yes|Supports fp32|
|[Tensor.roll](https://pytorch.org/docs/2.9/generated/torch.Tensor.roll.html)|Yes|Supports fp16, fp32, int32, int64, bool|
|[Tensor.rot90](https://pytorch.org/docs/2.9/generated/torch.Tensor.rot90.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.round](https://pytorch.org/docs/2.9/generated/torch.Tensor.round.html)|Yes|Supports fp16, fp32|
|[Tensor.round_](https://pytorch.org/docs/2.9/generated/torch.Tensor.round_.html)|Yes|Supports fp16, fp32|
|[Tensor.rsqrt](https://pytorch.org/docs/2.9/generated/torch.Tensor.rsqrt.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.rsqrt_](https://pytorch.org/docs/2.9/generated/torch.Tensor.rsqrt_.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[Tensor.scatter](https://pytorch.org/docs/2.9/generated/torch.Tensor.scatter.html)|Yes|Supports fp16, fp32, int16, int32, bool|
|[Tensor.scatter_](https://pytorch.org/docs/2.9/generated/torch.Tensor.scatter_.html)|Yes|The tensor, index, and src parameters cannot be empty and cannot be scalars<br>May fall back to CPU execution|
|[Tensor.scatter_add_](https://pytorch.org/docs/2.9/generated/torch.Tensor.scatter_add_.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.scatter_add](https://pytorch.org/docs/2.9/generated/torch.Tensor.scatter_add.html)|Yes|Supports fp32|
|[Tensor.scatter_reduce](https://pytorch.org/docs/2.9/generated/torch.Tensor.scatter_reduce.html)|Yes|Supports fp32, int64<br>May fall back to CPU execution|
|[Tensor.select](https://pytorch.org/docs/2.9/generated/torch.Tensor.select.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.select_scatter](https://pytorch.org/docs/2.9/generated/torch.Tensor.select_scatter.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU execution|
|[Tensor.set_](https://pytorch.org/docs/2.9/generated/torch.Tensor.set_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.share_memory_](https://pytorch.org/docs/2.9/generated/torch.Tensor.share_memory_.html)|No|-|
|[Tensor.short](https://pytorch.org/docs/2.9/generated/torch.Tensor.short.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.sigmoid_](https://pytorch.org/docs/2.9/generated/torch.Tensor.sigmoid_.html)|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.sign](https://pytorch.org/docs/2.9/generated/torch.Tensor.sign.html)|Yes|Supports bf16, fp16, fp32, int32, int64, bool|
|[Tensor.sign_](https://pytorch.org/docs/2.9/generated/torch.Tensor.sign_.html)|Yes|Supports fp16, fp32, int32, int64, bool|
|[Tensor.sgn](https://pytorch.org/docs/2.9/generated/torch.Tensor.sgn.html)|Yes|Supports fp16, fp32, int32, int64, bool, complex64, complex128|
|[Tensor.sgn_](https://pytorch.org/docs/2.9/generated/torch.Tensor.sgn_.html)|Yes|Supports fp16, fp32, fp64, int32, int64, bool|
|[Tensor.sin](https://pytorch.org/docs/2.9/generated/torch.Tensor.sin.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.sin_](https://pytorch.org/docs/2.9/generated/torch.Tensor.sin_.html)|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|[Tensor.sinh](https://pytorch.org/docs/2.9/generated/torch.Tensor.sinh.html)|Yes|Supports fp16, fp32, fp64|
|[Tensor.sinh_](https://pytorch.org/docs/2.9/generated/torch.Tensor.sinh_.html)|Yes|Supports fp16, fp32, fp64|
|[Tensor.asinh](https://pytorch.org/docs/2.9/generated/torch.Tensor.asinh.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.asinh_](https://pytorch.org/docs/2.9/generated/torch.Tensor.asinh_.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[Tensor.arcsinh](https://pytorch.org/docs/2.9/generated/torch.Tensor.arcsinh.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.arcsinh_](https://pytorch.org/docs/2.9/generated/torch.Tensor.arcsinh_.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[Tensor.shape](https://pytorch.org/docs/2.9/generated/torch.Tensor.shape.html)|Yes|-|
|[Tensor.size](https://pytorch.org/docs/2.9/generated/torch.Tensor.size.html)|Yes|Supports fp32|
|[Tensor.slogdet](https://pytorch.org/docs/2.9/generated/torch.Tensor.slogdet.html)|Yes|Supports fp32, complex64, complex128|
|[Tensor.slice_scatter](https://pytorch.org/docs/2.9/generated/torch.Tensor.slice_scatter.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.sort](https://pytorch.org/docs/2.9/generated/torch.Tensor.sort.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.split](https://pytorch.org/docs/2.9/generated/torch.Tensor.split.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.sparse_mask](https://pytorch.org/docs/2.9/generated/torch.Tensor.sparse_mask.html)|No|-|
|[Tensor.sparse_dim](https://pytorch.org/docs/2.9/generated/torch.Tensor.sparse_dim.html)|Yes|-|
|[Tensor.sqrt](https://pytorch.org/docs/2.9/generated/torch.Tensor.sqrt.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[Tensor.sqrt_](https://pytorch.org/docs/2.9/generated/torch.Tensor.sqrt_.html)|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|[Tensor.square](https://pytorch.org/docs/2.9/generated/torch.Tensor.square.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.square_](https://pytorch.org/docs/2.9/generated/torch.Tensor.square_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|[Tensor.squeeze](https://pytorch.org/docs/2.9/generated/torch.Tensor.squeeze.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.squeeze_](https://pytorch.org/docs/2.9/generated/torch.Tensor.squeeze_.html)|Yes|Supports fp32|
|[Tensor.std](https://pytorch.org/docs/2.9/generated/torch.Tensor.std.html)|Yes|Supports bf16, fp16, fp32<br>input does not support scalar tensors<br>correction must not exceed the range of int32|
|[Tensor.storage](https://pytorch.org/docs/2.9/generated/torch.Tensor.storage.html)|Yes|Supports fp32|
|[Tensor.untyped_storage](https://pytorch.org/docs/2.9/generated/torch.Tensor.untyped_storage.html)|Yes|Supports fp32|
|[Tensor.storage_offset](https://pytorch.org/docs/2.9/generated/torch.Tensor.storage_offset.html)|Yes|Supports fp32|
|[Tensor.storage_type](https://pytorch.org/docs/2.9/generated/torch.Tensor.storage_type.html)|Yes|Supports fp32|
|[Tensor.stride](https://pytorch.org/docs/2.9/generated/torch.Tensor.stride.html)|Yes|Supports fp32|
|[Tensor.sub](https://pytorch.org/docs/2.9/generated/torch.Tensor.sub.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.sub_](https://pytorch.org/docs/2.9/generated/torch.Tensor.sub_.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128|
|[Tensor.subtract_](https://pytorch.org/docs/2.9/generated/torch.Tensor.subtract_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[Tensor.sum](https://pytorch.org/docs/2.9/generated/torch.Tensor.sum.html)|Yes|Supports bf16, fp16, fp32, int32|
|[Tensor.sum_to_size](https://pytorch.org/docs/2.9/generated/torch.Tensor.sum_to_size.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.swapaxes](https://pytorch.org/docs/2.9/generated/torch.Tensor.swapaxes.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.swapdims](https://pytorch.org/docs/2.9/generated/torch.Tensor.swapdims.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.t](https://pytorch.org/docs/2.9/generated/torch.Tensor.t.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128|
|[Tensor.t_](https://pytorch.org/docs/2.9/generated/torch.Tensor.t_.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64|
|[Tensor.tensor_split](https://pytorch.org/docs/2.9/generated/torch.Tensor.tensor_split.html)|Yes|CPU Only|
|[Tensor.tile](https://pytorch.org/docs/2.9/generated/torch.Tensor.tile.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>If the length of the input parameter dims is less than the length of Tensor.shape, 1s are automatically prepended to dims to align its length with Tensor.shape. The completed dims must satisfy the following restrictions:<br>- When repeating along the first axis, a maximum of 4 dimensions can be repeated simultaneously (i.e., the number of elements greater than 1 in dims ≤ 4). For example: Tensor.tile([2, 3, 4, 5, 6]) is unsupported, Tensor.tile([2, 3, 1, 5, 6]) is supported<br>- When not repeating along the first axis, a maximum of 3 dimensions can be repeated simultaneously (i.e., the number of elements greater than 1 in dims ≤ 3). For example: Tensor.tile([1, 3, 4, 5, 6]) is unsupported, Tensor.tile([1, 3, 1, 5, 6]) is supported<br>- If backward computation is performed, the sum of the number of Tensor dimensions and the number of elements greater than 1 in the input parameter dims must not exceed 8|
|[Tensor.to](https://pytorch.org/docs/2.9/generated/torch.Tensor.to.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Currently, the NPU device only supports setting memory_format to torch.contiguous_format or torch.preserve_format<br><term>Atlas Inference Series Product</term> does not support cross-NPU copy|
|[to](https://pytorch.org/docs/2.9/generated/torch.Tensor.to.html)|Yes|-|
|[Tensor.to_mkldnn](https://pytorch.org/docs/2.9/generated/torch.Tensor.to_mkldnn.html)|No|-|
|[Tensor.take](https://pytorch.org/docs/2.9/generated/torch.Tensor.take.html)|Yes|Supports fp16, fp32, int16, int32, bool|
|[Tensor.take_along_dim](https://pytorch.org/docs/2.9/generated/torch.Tensor.take_along_dim.html)|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|[Tensor.tan](https://pytorch.org/docs/2.9/generated/torch.Tensor.tan.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128. Value range [-65504, 65504]|
|[Tensor.tan_](https://pytorch.org/docs/2.9/generated/torch.Tensor.tan_.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[Tensor.tanh](https://pytorch.org/docs/2.9/generated/torch.Tensor.tanh.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[Tensor.tanh_](https://pytorch.org/docs/2.9/generated/torch.Tensor.tanh_.html)|Yes|Supports fp16, fp32|
|[Tensor.atanh](https://pytorch.org/docs/2.9/generated/torch.Tensor.atanh.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.atanh_](https://pytorch.org/docs/2.9/generated/torch.Tensor.atanh_.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[Tensor.arctanh](https://pytorch.org/docs/2.9/generated/torch.Tensor.arctanh.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.arctanh_](https://pytorch.org/docs/2.9/generated/torch.Tensor.arctanh_.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[Tensor.tolist](https://pytorch.org/docs/2.9/generated/torch.Tensor.tolist.html)|Yes|Supports fp32|
|[Tensor.topk](https://pytorch.org/docs/2.9/generated/torch.Tensor.topk.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Due to hardware differences, the NPU topk index results differ from GPU/CPU. Currently, the NPU only supports returning computation results with sorted=true<br>Scalar tensors are unsupported|
|[Tensor.to_dense](https://pytorch.org/docs/2.9/generated/torch.Tensor.to_dense.html)|No|-|
|[Tensor.to_sparse](https://pytorch.org/docs/2.9/generated/torch.Tensor.to_sparse.html)|No|-|
|[to_sparse](https://pytorch.org/docs/2.9/generated/torch.Tensor.to_sparse.html)|No|-|
|[Tensor.to_sparse_csr](https://pytorch.org/docs/2.9/generated/torch.Tensor.to_sparse_csr.html)|No|-|
|[Tensor.to_sparse_csc](https://pytorch.org/docs/2.9/generated/torch.Tensor.to_sparse_csc.html)|No|-|
|[Tensor.to_sparse_bsr](https://pytorch.org/docs/2.9/generated/torch.Tensor.to_sparse_bsr.html)|No|-|
|[Tensor.to_sparse_bsc](https://pytorch.org/docs/2.9/generated/torch.Tensor.to_sparse_bsc.html)|No|-|
|[Tensor.transpose](https://pytorch.org/docs/2.9/generated/torch.Tensor.transpose.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.transpose_](https://pytorch.org/docs/2.9/generated/torch.Tensor.transpose_.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.tril](https://pytorch.org/docs/2.9/generated/torch.Tensor.tril.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.tril_](https://pytorch.org/docs/2.9/generated/torch.Tensor.tril_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.triu](https://pytorch.org/docs/2.9/generated/torch.Tensor.triu.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.triu_](https://pytorch.org/docs/2.9/generated/torch.Tensor.triu_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.true_divide](https://pytorch.org/docs/2.9/generated/torch.Tensor.true_divide.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.true_divide_](https://pytorch.org/docs/2.9/generated/torch.Tensor.true_divide_.html)|Yes|Supports fp16, fp32|
|[Tensor.trunc](https://pytorch.org/docs/2.9/generated/torch.Tensor.trunc.html)|Yes|Supports fp16, fp32|
|[Tensor.trunc_](https://pytorch.org/docs/2.9/generated/torch.Tensor.trunc_.html)|Yes|Supports fp16, fp32|
|[Tensor.type](https://pytorch.org/docs/2.9/generated/torch.Tensor.type.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.type_as](https://pytorch.org/docs/2.9/generated/torch.Tensor.type_as.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.unbind](https://pytorch.org/docs/2.9/generated/torch.Tensor.unbind.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.unflatten](https://pytorch.org/docs/2.9/generated/torch.Tensor.unflatten.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.unfold](https://pytorch.org/docs/2.9/generated/torch.Tensor.unfold.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[Tensor.uniform_](https://pytorch.org/docs/2.9/generated/torch.Tensor.uniform_.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Following PyTorch community specifications, processing of bool type data is no longer supported. For existing bool type data, the following alternatives can be used: If output of all True is required, use Tensor.bernoulli_(p=1.0). If uniformly distributed bool type output is required, use Tensor.bernoulli_(p=0.5)|
|[Tensor.unique](https://pytorch.org/docs/2.9/generated/torch.Tensor.unique.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>When the input contains 0, the output may include both positive 0 and negative 0, rather than outputting only one 0|
|[Tensor.unique_consecutive](https://pytorch.org/docs/2.9/generated/torch.Tensor.unique_consecutive.html)|No|-|
|[Tensor.unsqueeze](https://pytorch.org/docs/2.9/generated/torch.Tensor.unsqueeze.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.unsqueeze_](https://pytorch.org/docs/2.9/generated/torch.Tensor.unsqueeze_.html)|Yes|Supports fp32|
|[Tensor.values](https://pytorch.org/docs/2.9/generated/torch.Tensor.values.html)|Yes|Depends on sparse tensor|
|[Tensor.var](https://pytorch.org/docs/2.9/generated/torch.Tensor.var.html)|Yes|Supports bf16, fp16, fp32<br>correction must not exceed the int32 range|
|[Tensor.view](https://pytorch.org/docs/2.9/generated/torch.Tensor.view.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[view](https://pytorch.org/docs/2.9/generated/torch.Tensor.view.html)|Yes|-|
|[Tensor.view_as](https://pytorch.org/docs/2.9/generated/torch.Tensor.view_as.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.vsplit](https://pytorch.org/docs/2.9/generated/torch.Tensor.vsplit.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.where](https://pytorch.org/docs/2.9/generated/torch.Tensor.where.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[Tensor.xlogy](https://pytorch.org/docs/2.9/generated/torch.Tensor.xlogy.html)|Yes|Supports fp16, fp32|
|[Tensor.xlogy_](https://pytorch.org/docs/2.9/generated/torch.Tensor.xlogy_.html)|Yes|Supports fp16, fp32|
|[Tensor.zero_](https://pytorch.org/docs/2.9/generated/torch.Tensor.zero_.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
