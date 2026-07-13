# torch.Tensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T08:00:19.994Z pushedAt=2026-06-14T09:16:34.786Z -->

> **NOTE**
>
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.Tensor|Yes|-|
|Tensor.T|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.H|Yes|-|
|Tensor.mT|Yes|-|
|Tensor.mH|Yes|-|
|Tensor.new_tensor|Yes|-|
|Tensor.new_full|Yes|supports int64|
|Tensor.new_empty|Yes|supports fp32|
|Tensor.new_ones|Yes|supports fp32|
|Tensor.new_zeros|Yes|supports fp32|
|Tensor.is_cuda|Yes|-|
|Tensor.is_quantized|Yes|-|
|Tensor.is_meta|Yes|-|
|Tensor.device|Yes|-|
|Tensor.grad|Yes|supports fp32|
|Tensor.ndim|Yes|supports fp32|
|Tensor.real|Yes|-|
|Tensor.imag|Yes|-|
|Tensor.nbytes|Yes|-|
|Tensor.itemsize|Yes|-|
|Tensor.abs|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.abs_|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64|
|Tensor.absolute|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.absolute_|Yes|supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.acos|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU for execution|
|Tensor.acos_|Yes|supports float16, float32|
|Tensor.arccos|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.arccos_|Yes|supports float16, float32|
|Tensor.add|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.add_|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.addbmm|Yes|supports float16, float32|
|Tensor.addbmm_|Yes|supports float16, float32|
|Tensor.addcdiv|Yes|supports bfloat16, float16, float32, int64<br>Broadcasting of three tensors is not supported for int64|
|Tensor.addcdiv_|Yes|Atlas A2 Training Series Products/Atlas A3 Training Series Products: supports bfloat16, float16, float32, float64<br>Atlas Training Series Products: supports float16, float32, float64<br>Broadcasting of three tensors is not supported for int64|
|Tensor.addcmul|Yes|supports float16, float32, int64<br>Broadcasting of three tensors is not supported for int64|
|Tensor.addcmul_|Yes|Atlas A2 Training Series Products/Atlas A3 Training Series Products: supports bfloat16, float16, float32, float64, uint8, int8, int32, int64<br>Atlas Training Series Products: supports float16, float32, float64, uint8, int8, int32, int64<br>Broadcasting of three tensors is not supported for int64|
|Tensor.addmm|Yes|supports float16, float32|
|Tensor.addmm_|Yes|supports float16, float32|
|Tensor.sspaddmm|No|-|
|Tensor.addmv|Yes|supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.addmv_|Yes|supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.addr|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.addr_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.adjoint|Yes|-|
|Tensor.allclose|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.amax|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.amin|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.aminmax|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.angle|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.apply_|Yes|CPU-only|
|Tensor.argmax|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64|
|Tensor.argmin|Yes|supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.argsort|Yes|supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.argwhere|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.asin|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.asin_|Yes|supports float16, float32|
|Tensor.arcsin|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.arcsin_|Yes|supports float16, float32|
|Tensor.as_strided|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.atan|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.atan_|Yes|supports float16, float32|
|Tensor.arctan|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.arctan_|Yes|supports float16, float32|
|Tensor.atan2|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.atan2_|Yes|supports float16, float32<br>May fall back to CPU for execution|
|Tensor.arctan2|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.arctan2_|Yes|supports float16, float32|
|Tensor.all|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.any|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.backward|Yes|supports fp32|
|Tensor.baddbmm|Yes|supports float16, float32|
|Tensor.baddbmm_|Yes|supports float16, float32|
|Tensor.bernoulli|Yes|supports float16, float32<br>May fall back to CPU for execution|
|Tensor.bernoulli_|Yes|May fall back to CPU for execution|
|Tensor.bfloat16|Yes|supports float16, float32|
|Tensor.bincount|Yes|supports uint8, int8, int16, int32, int64<br>The weights dimension must be consistent with the input dimension|
|Tensor.bitwise_not|Yes|supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_not_|Yes|supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_and|Yes|supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_and_|Yes|supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_or|Yes|supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_or_|Yes|supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_xor|Yes|supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_xor_|Yes|supports uint8, int8, int16, int32, int64, bool|
|Tensor.bmm|Yes|supports float16, float32|
|Tensor.bool|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.byte|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.broadcast_to|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.ceil|Yes|supports float16, float32|
|Tensor.ceil_|Yes|supports float16, float32|
|Tensor.char|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.chunk|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.clamp|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64|
|Tensor.clamp_|Yes|May fall back to CPU for execution|
|Tensor.clip|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64|
|Tensor.clip_|Yes|supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.clone|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.contiguous|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.copy_|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>int16 does not support dimensions above 5|
|Tensor.conj|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.resolve_conj|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.resolve_neg|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.copysign|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU for execution|
|Tensor.cos|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.cos_|Yes|supports bfloat16, float16, float32, complex64, complex128|
|Tensor.cosh|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.cosh_|Yes|supports bfloat16, float16, float32, float64, complex64, complex128|
|Tensor.count_nonzero|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.cov|Yes|supports float16, float32, int16, int32, int64|
|Tensor.acosh|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May fall back to CPU for execution|
|Tensor.acosh_|Yes|supports bfloat16, float16, float32, float64, complex64, complex128|
|Tensor.arccosh|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.arccosh_|Yes|supports bfloat16, float16, float32, float64, complex64, complex128|
|Tensor.cpu|Yes|-|
|Tensor.cross|Yes|supports float16, float32, uint8, int8, int16, int32, int64, complex64, complex128<br>The shapes of the two inputs must be consistent|
|Tensor.cuda|Yes|The corresponding NPU interface is Tensor.npu, and memory_format only supports passing torch.contiguous_format|
|Tensor.cummax|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU for execution|
|Tensor.cummin|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.cumsum|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>supports Named Tensor|
|Tensor.cumsum_|Yes|supports float16, float32, int64, bool|
|Tensor.chalf|No|-|
|Tensor.cfloat|Yes|-|
|Tensor.cdouble|Yes|-|
|Tensor.data_ptr|Yes|supports float32|
|Tensor.deg2rad|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.dequantize|Yes|supports float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.dense_dim|Yes|-|
|Tensor.detach|Yes|supports float32|
|Tensor.detach_|Yes|supports float32|
|Tensor.diag|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.diag_embed|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.diagflat|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.diagonal|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.diagonal_scatter|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.fill_diagonal_|Yes|supports float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.diff|Yes|supports float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.dim|Yes|supports float32|
|Tensor.dim_order|Yes|-|
|Tensor.dist|Yes|-|
|Tensor.div|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.div_|Yes|supports bfloat16, float16, float32, float64|
|Tensor.divide|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.divide_|Yes|supports float16, float32|
|Tensor.dot|Yes|supports float16, float32|
|Tensor.double|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Some NPU interfaces currently do not support the double type. For compatibility, float32 is returned by default. float64 will be returned normally once support is completed.<br>May fall back to CPU for execution|
|Tensor.dsplit|Yes|supports float32|
|Tensor.element_size|Yes|supports float32|
|Tensor.eq|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.eq_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.equal|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.erf|Yes|supports float16, float32, int64, bool|
|Tensor.erf_|Yes|supports float16, float32|
|Tensor.erfc|Yes|supports float16, float32, int64, bool|
|Tensor.erfc_|Yes|supports float16, float32|
|Tensor.erfinv|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.erfinv_|Yes|supports bfloat16, float16, float32|
|Tensor.exp|Yes|supports bfloat16, float16, float32, int64, bool, complex64, complex128|
|Tensor.exp_|Yes|supports bfloat16, float16, float32, complex64, complex128|
|Tensor.expm1|Yes|supports float16, float32, int64, bool|
|Tensor.expm1_|Yes|supports float16, float32|
|Tensor.expand|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.expand_as|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.exponential_|Yes|supports bfloat16, float16, float32, float64|
|Tensor.fix|Yes|supports float16, float32|
|Tensor.fix_|Yes|supports float16, float32|
|Tensor.fill_|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.flatten|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.flip|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.fliplr|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.flipud|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.float|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.float_power|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex128|
|Tensor.float_power_|Yes|supports double|
|Tensor.floor|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64|
|Tensor.floor_|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64|
|Tensor.floor_divide|Yes|supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.floor_divide_|Yes|supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.fmod|Yes|supports float16, float32, uint8, int8, int32, int64|
|Tensor.fmod_|Yes|supports float16, float32, uint8, int8, int32, int64|
|Tensor.frac|Yes|supports float16, float32|
|Tensor.frac_|Yes|supports float16, float32|
|Tensor.gather|Yes|supports float16, float32, int64<br>The index dimension must be consistent with the input dimension|
|Tensor.ge|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.ge_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater_equal|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater_equal_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.geometric_|Yes|-|
|Tensor.ger|Yes|-|
|Tensor.get_device|Yes|-|
|Tensor.gt|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.gt_|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.half|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.hardshrink|Yes|supports float16, float32|
|Tensor.heaviside|Yes|May fall back to CPU for execution|
|Tensor.histc|Yes|supports float16, float32|
|Tensor.hsplit|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.index_add_|Yes|supports float16, float32, int64, bool|
|Tensor.index_add|Yes|supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.index_copy_|Yes|supports float16, float32, int16, int32, bool|
|Tensor.index_copy|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.index_fill_|Yes|supports float16, float32, int32, int64, bool|
|Tensor.index_fill|Yes|supports float16, float32, int32, bool|
|Tensor.index_put_|Yes|supports int64|
|Tensor.index_put|Yes|supports int64|
|Tensor.index_reduce_|Yes|May fall back to CPU for execution|
|Tensor.index_reduce|Yes|May fall back to CPU for execution|
|Tensor.index_select|Yes|supports float16, float32, int16, int32, int64, bool|
|Tensor.indices|Yes|-|
|Tensor.inner|Yes|supports float16, float32|
|Tensor.int|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.int_repr|No|-|
|Tensor.isclose|Yes|supports float16, float32, uint8, int32, int64, bool|
|Tensor.isfinite|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.isinf|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.isposinf|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.isneginf|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.isnan|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.is_contiguous|Yes|supports float32|
|Tensor.is_complex|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.is_conj|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.is_floating_point|Yes|supports float32|
|Tensor.is_inference|Yes|supports float32|
|Tensor.is_leaf|Yes|-|
|Tensor.is_pinned|Yes|supports float32|
|Tensor.is_set_to|Yes|supports float32|
|Tensor.is_shared|No|-|
|Tensor.is_signed|Yes|supports float32|
|Tensor.is_sparse|Yes|supports float32|
|Tensor.isreal|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.item|Yes|supports float32|
|Tensor.kthvalue|Yes|supports float16, float32, int32|
|Tensor.ldexp|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.ldexp_|Yes|supports float16, float32, complex64, complex128|
|Tensor.le|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.le_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.less_equal|Yes|supports float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.less_equal_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.lerp|Yes|supports float16, float32|
|Tensor.lerp_|Yes|supports float16, float32|
|Tensor.log|Yes|supports bfloat16, float16, float32, int64, bool, complex64, complex128|
|Tensor.log_|Yes|supports bfloat16, float16, float32, complex64, complex128|
|Tensor.log10|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.log10_|Yes|supports bfloat16, float16, float32, float64, complex64, complex128|
|Tensor.log1p|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.log1p_|Yes|supports float16, float32|
|Tensor.log2|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.log2_|Yes|supports float16, float32, complex64, complex128|
|Tensor.logaddexp|Yes|supports float16, float32, int16, int32, int64, bool|
|Tensor.logaddexp2|Yes|supports float16, float32|
|Tensor.logsumexp|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.logical_and|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_and_|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_not|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.logical_not_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.logical_or|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_or_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_xor|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May fall back to CPU for execution|
|Tensor.logical_xor_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logit|Yes|supports bfloat16, float16, float32<br>Output is nan when eps > 1, and inf when eps = 1|
|Tensor.logit_|Yes|supports bfloat16, float16, float32<br>Output is nan when eps > 1, and inf when eps = 1|
|Tensor.long|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.lt|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.lt_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.less|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.less_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.as_subclass|Yes|-|
|Tensor.map_|Yes|CPU-only|
|Tensor.masked_scatter_|Yes|supports float32, int64, bool|
|Tensor.masked_scatter|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.masked_fill_|Yes|supports bfloat16, float16, float32, int8, int32, int64, bool|
|Tensor.masked_fill|Yes|supports float16, float32, int64, bool|
|Tensor.masked_select|Yes|supports float32, bool|
|Tensor.matmul|Yes|supports bfloat16, float16, float32<br>Supports Named Tensor|
|Tensor.matrix_power|Yes|supports float16, float32|
|Tensor.max|Yes|supports bfloat16, float16, float32, int64, bool|
|Tensor.maximum|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.mean|Yes|supports bfloat16, float16, float32, complex64, complex128|
|Tensor.nanmean|Yes|supports float16, float32|
|Tensor.median|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64<br>When input is bfloat16, dim cannot be a dimension where the input axis value is 1|
|Tensor.min|Yes|supports bfloat16, float16, float32, int64, bool|
|Tensor.minimum|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.mm|Yes|supports float16, float32|
|Tensor.smm|No|-|
|Tensor.mode|No|-|
|Tensor.movedim|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.moveaxis|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.msort|Yes|supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.mul|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.mul_|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.multiply|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.multiply_|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.multinomial|Yes|supports float16, float32|
|Tensor.nansum|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.narrow|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.narrow_copy|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.ndimension|Yes|supports float32|
|Tensor.nan_to_num|Yes|-|
|Tensor.nan_to_num_|Yes|-|
|Tensor.ne|Yes|Supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.ne_|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.nextafter_|Yes|Falls back to CPU for execution|
|Tensor.not_equal|Yes|Supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May fall back to CPU for execution|
|Tensor.not_equal_|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.neg|Yes|Supports float16, float32, int8, int32, int64, complex64, complex128|
|Tensor.neg_|Yes|Supports bfloat16, float16, float32, int8, int32, int64, complex64, complex128<br>May fall back to CPU for execution|
|Tensor.negative|Yes|Supports float16, float32, int8, int32, int64, complex64, complex128|
|Tensor.negative_|Yes|Supports float16, float32, int8, int32, int64, complex64, complex128|
|Tensor.nelement|Yes|Supports float32|
|Tensor.nonzero|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool<br>NaN scenarios are not supported|
|Tensor.norm|Yes|Supports bfloat16, float16, float32, float64|
|Tensor.normal_|Yes|Supports bfloat16, float16, float32<br>May fall back to CPU for execution|
|Tensor.numel|Yes|Supports float32|
|Tensor.numpy|Yes|Supports float32|
|Tensor.outer|Yes|Supports float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.permute|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.positive|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.pow|Yes|Supports bfloat16, float16, float32, float64, int16, int64|
|Tensor.pow_|Yes|Supports bfloat16, float16, float32, float64, int64|
|Tensor.prod|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.put_|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.qscheme|No|-|
|Tensor.quantile|Yes|-|
|Tensor.rad2deg|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.random_|Yes|Supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.ravel|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.reciprocal|Yes|Supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.reciprocal_|Yes|Supports float16, float32, complex64, complex128|
|Tensor.record_stream|Yes|Supports float32|
|Tensor.register_hook|Yes|Supports float32|
|Tensor.register_post_accumulate_grad_hook|Yes|-|
|Tensor.remainder|Yes|Supports float16, float32, int32, int64|
|Tensor.remainder_|Yes|Supports float16, float32, int32, int64|
|Tensor.repeat|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.repeat_interleave|Yes|Supports float16, float32, int16, int32, bool<br>The input tensor is repeated to produce the output, and the number of elements in the output must be less than $2^{22}$|
|Tensor.requires_grad|Yes|-|
|Tensor.requires_grad_|Yes|Supports float32|
|Tensor.reshape|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.reshape_as|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.resize_|Yes|Supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>memory_format only supports torch.contiguous_format and torch.preserve_format|
|Tensor.resize_as_|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>memory_format only supports torch.contiguous_format and torch.preserve_format|
|Tensor.retain_grad|Yes|Supports float32|
|Tensor.retains_grad|Yes|Supports float32|
|Tensor.roll|Yes|Supports float16, float32, int32, int64, bool|
|Tensor.rot90|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.round|Yes|Supports float16, float32|
|Tensor.round_|Yes|Supports float16, float32|
|Tensor.rsqrt|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.rsqrt_|Yes|Supports float16, float32, complex64, complex128|
|Tensor.scatter|Yes|Supports float16, float32, int16, int32, bool|
|Tensor.scatter_|Yes|The tensor, index, and src parameters cannot be empty and cannot be scalars<br>May fall back to CPU for execution|
|Tensor.scatter_add_|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.scatter_add|Yes|Supports float32|
|Tensor.scatter_reduce|Yes|Supports float32, int64<br>May fall back to CPU for execution|
|Tensor.select|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.select_scatter|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU for execution|
|Tensor.set_|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.share_memory_|No|-|
|Tensor.short|Yes|Supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.sigmoid_|Yes|Supports bfloat16, float16, float32, float64, complex64, complex128|
|Tensor.sign|Yes|Supports bfloat16, float16, float32, int32, int64, bool|
|Tensor.sign_|Yes|Supports float16, float32, int32, int64, bool|
|Tensor.sgn|Yes|Supports float16, float32, int32, int64, bool, complex64, complex128|
|Tensor.sgn_|Yes|Supports float16, float32, float64, int32, int64, bool|
|Tensor.sin|Yes|Supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.sin_|Yes|Supports bfloat16, float16, float32, complex64, complex128|
|Tensor.sinh|Yes|Supports float16, float32, float64|
|Tensor.sinh_|Yes|Supports float16, float32, float64|
|Tensor.asinh|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.asinh_|Yes|Supports float16, float32, complex64, complex128|
|Tensor.arcsinh|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.arcsinh_|Yes|Supports float16, float32, complex64, complex128|
|Tensor.shape|Yes|-|
|Tensor.size|Yes|Supports float32|
|Tensor.slogdet|Yes|Supports float32, complex64, complex128|
|Tensor.slice_scatter|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.sort|Yes|Supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.split|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.sparse_mask|No|-|
|Tensor.sparse_dim|Yes|-|
|Tensor.sqrt|Yes|Supports float16, float32, float64, uint8, int8, int16, int32, int64, bool|
|Tensor.sqrt_|Yes|Supports bfloat16, float16, float32, float64, complex64, complex128|
|Tensor.square|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.square_|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.squeeze|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.squeeze_|Yes|Supports float32|
|Tensor.std|Yes|Supports bfloat16, float16, float32<br>Input does not support scalar tensors<br>correction must not exceed the range of int32|
|Tensor.storage|Yes|Supports float32|
|Tensor.untyped_storage|Yes|Supports float32|
|Tensor.storage_offset|Yes|Supports float32|
|Tensor.storage_type|Yes|Supports float32|
|Tensor.stride|Yes|Supports float32|
|Tensor.sub|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64|
|Tensor.sub_|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.subtract_|Yes|Supports float16, float32, uint8, int8, int16, int32, int64|
|Tensor.sum|Yes|Supports bfloat16, float16, float32, int32|
|Tensor.sum_to_size|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.swapaxes|Yes|Supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.swapdims|Yes|Supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.t|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.t_|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.tensor_split|Yes|CPU-only|
|Tensor.tile|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>If the length of the input parameter dims is less than the length of Tensor.shape, 1s are automatically prepended to dims to align its length with Tensor.shape. The completed dims must satisfy the following restrictions:<br>- When repeating along the first axis, repeat operations are allowed on at most 4 dimensions simultaneously (i.e., the number of elements greater than 1 in dims ≤ 4). For example: Tensor.tile([2, 3, 4, 5, 6]) is Unsupported, while Tensor.tile([2, 3, 1, 5, 6]) is supported.<br>- When not repeating along the first axis, repeat operations are allowed on at most 3 dimensions simultaneously (i.e., the number of elements greater than 1 in dims ≤ 3). For example: Tensor.tile([1, 3, 4, 5, 6]) is Unsupported, while Tensor.tile([1, 3, 1, 5, 6]) is supported.<br>- If backward computation is performed, the sum of the number of Tensor dimensions and the number of elements greater than 1 in the input parameter dims must not exceed 8|
|Tensor.to|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Currently, NPU devices only support setting memory_format to torch.contiguous_format or torch.preserve_format<br>Atlas Inference Series Products do not support cross-NPU copying|
|to|Yes|-|
|Tensor.to_mkldnn|No|-|
|Tensor.take|Yes|supports float16, float32, int16, int32, bool|
|Tensor.take_along_dim|Yes|supports float16, float32, int16, int32, int64, bool|
|Tensor.tan|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128 with value range [-65504, 65504]|
|Tensor.tan_|Yes|supports float16, float32, complex64, complex128|
|Tensor.tanh|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64|
|Tensor.tanh_|Yes|supports float16, float32|
|Tensor.atanh|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.atanh_|Yes|supports float16, float32, complex64, complex128|
|Tensor.arctanh|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.arctanh_|Yes|supports float16, float32, complex64, complex128|
|Tensor.tolist|Yes|supports float32|
|Tensor.topk|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64<br>Due to hardware differences, NPU topk index results may differ from GPU/CPU. Currently, NPU only supports returning computation results with sorted=true<br>Scalar tensors are Unsupported|
|Tensor.to_dense|No|-|
|Tensor.to_sparse|No|-|
|to_sparse|No|-|
|Tensor.to_sparse_csr|No|-|
|Tensor.to_sparse_csc|No|-|
|Tensor.to_sparse_bsr|No|-|
|Tensor.to_sparse_bsc|No|-|
|Tensor.transpose|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.transpose_|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.tril|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.tril_|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.triu|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.triu_|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.true_divide|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.true_divide_|Yes|supports float16, float32|
|Tensor.trunc|Yes|supports float16, float32|
|Tensor.trunc_|Yes|supports float16, float32|
|Tensor.type|Yes|supports bfloat16, float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.type_as|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unbind|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unflatten|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unfold|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool|
|Tensor.uniform_|Yes|supports float16, float32, float64, uint8, int8, int16, int32, int64<br>Following PyTorch community conventions, processing of bool type data is no longer supported. For existing bool type data, the following alternatives can be used: If you need to output all True, use Tensor.bernoulli_(p=1.0). If you need uniformly distributed bool type output, use Tensor.bernoulli_(p=0.5)|
|Tensor.unique|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool<br>When the input contains 0, the output may contain both positive 0 and negative 0, rather than only one 0|
|Tensor.unique_consecutive|No|-|
|Tensor.unsqueeze|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unsqueeze_|Yes|supports float32|
|Tensor.values|Yes|Depends on sparse tensor|
|Tensor.var|Yes|supports bfloat16, float16, float32<br>correction does not exceed the range of int32|
|Tensor.view|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|view|Yes|-|
|Tensor.view_as|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.vsplit|Yes|supports float16, float32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.where|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.xlogy|Yes|supports float16, float32|
|Tensor.xlogy_|Yes|supports float16, float32|
|Tensor.zero_|Yes|supports bfloat16, float16, float32, float64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
