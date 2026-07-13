# torch.Tensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:11:02.553Z pushedAt=2026-06-15T02:04:36.618Z -->

> [!NOTE]
> If the "Support" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Support|Restrictions and Notes|
|--|--|--|
|torch.Tensor|Yes|-|
|Tensor.T|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
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
|Tensor.abs|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.abs_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.absolute|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.absolute_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.acos|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>may fall back to CPU|
|Tensor.acos_|Yes|Supports fp16, fp32|
|Tensor.arccos|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.arccos_|Yes|Supports fp16, fp32|
|Tensor.add|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.add_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.addbmm|Yes|Supports fp16, fp32|
|Tensor.addbmm_|Yes|Supports fp16, fp32|
|Tensor.addcdiv|Yes|Supports bf16, fp16, fp32, int64<br>int64 type does not support broadcasting three tensors simultaneously|
|Tensor.addcdiv_|Yes|Atlas A2 training series/Atlas A3 training series: Supports bf16, fp16, fp32, fp64<br>Atlas training series: Supports fp16, fp32, fp64<br>int64 type does not support broadcasting three tensors simultaneously|
|Tensor.addcmul|Yes|Supports fp16, fp32, int64<br>int64 type does not support broadcasting three tensors simultaneously|
|Tensor.addcmul_|Yes|Atlas A2 training series/Atlas A3 training series: Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64<br>Atlas training series: Supports fp16, fp32, fp64, uint8, int8, int32, int64<br>int64 type does not support broadcasting three tensors simultaneously|
|Tensor.addmm|Yes|Supports bf16, fp16, fp32|
|Tensor.addmm_|Yes|Supports fp16, fp32|
|Tensor.sspaddmm|No|-|
|Tensor.addmv|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.addmv_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.addr|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.addr_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.adjoint|Yes|-|
|Tensor.allclose|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.amax|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.amin|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.aminmax|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.angle|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.apply_|Yes|CPU only|
|Tensor.argmax|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.argmin|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.argsort|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.argwhere|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.asin|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.asin_|Yes|Supports fp16, fp32|
|Tensor.arcsin|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.arcsin_|Yes|Supports fp16, fp32|
|Tensor.as_strided|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.atan|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.atan_|Yes|Supports fp16, fp32|
|Tensor.arctan|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.arctan_|Yes|Supports fp16, fp32|
|Tensor.atan2|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.atan2_|Yes|Supports fp16, fp32<br>may fall back to CPU|
|Tensor.arctan2|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.arctan2_|Yes|Supports fp16, fp32|
|Tensor.all|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.any|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.backward|Yes|supports fp32|
|Tensor.baddbmm|Yes|Supports fp16, fp32|
|Tensor.baddbmm_|Yes|Supports fp16, fp32|
|Tensor.bernoulli|Yes|Supports fp16, fp32<br>may fall back to CPU|
|Tensor.bernoulli_|Yes|may fall back to CPU|
|Tensor.bfloat16|Yes|Supports fp16, fp32|
|Tensor.bincount|Yes|Supports uint8, int8, int16, int32, int64<br>The weights dimension must be consistent with the input dimension|
|Tensor.bitwise_not|Yes|Supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_not_|Yes|Supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_and|Yes|Supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_and_|Yes|Supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_or|Yes|Supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_or_|Yes|Supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_xor|Yes|Supports uint8, int8, int16, int32, int64, bool|
|Tensor.bitwise_xor_|Yes|Supports uint8, int8, int16, int32, int64, bool|
|Tensor.bmm|Yes|Supports bf16, fp16, fp32|
|Tensor.bool|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.byte|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.broadcast_to|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.ceil|Yes|Supports fp16, fp32|
|Tensor.ceil_|Yes|Supports fp16, fp32|
|Tensor.char|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.chunk|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.clamp|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.clamp_|Yes|may fall back to CPU|
|Tensor.clip|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.clip_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.clone|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.contiguous|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.copy_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>int16 does not support dimensions greater than 5|
|Tensor.conj|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.resolve_conj|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.resolve_neg|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.copysign|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool<br>may fall back to CPU|
|Tensor.cos|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.cos_|Yes|supports bf16, fp16, fp32, complex64, complex128|
|Tensor.cosh|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.cosh_|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.count_nonzero|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.cov|Yes|supports fp16, fp32, int16, int32, int64|
|Tensor.acosh|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>may fall back to CPU|
|Tensor.acosh_|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.arccosh|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.arccosh_|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.cpu|Yes|-|
|Tensor.cross|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128<br>The shapes of the two inputs must be consistent|
|Tensor.cuda|Yes|The corresponding NPU interface is Tensor.npu, and memory_format only supports passing torch.contiguous_format|
|Tensor.cummax|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>may fall back to CPU|
|Tensor.cummin|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.cumsum|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>supports Named Tensor|
|Tensor.cumsum_|Yes|supports fp16, fp32, int64, bool|
|Tensor.chalf|No|-|
|Tensor.cfloat|Yes|-|
|Tensor.cdouble|Yes|-|
|Tensor.data_ptr|Yes|supports fp32|
|Tensor.deg2rad|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.dequantize|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.dense_dim|Yes|-|
|Tensor.detach|Yes|supports fp32|
|Tensor.detach_|Yes|supports fp32|
|Tensor.diag|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.diag_embed|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.diagflat|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.diagonal|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.diagonal_scatter|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.fill_diagonal_|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.diff|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.dim|Yes|supports fp32|
|Tensor.dim_order|Yes|-|
|Tensor.dist|Yes|-|
|Tensor.div|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.div_|Yes|supports bf16, fp16, fp32, fp64|
|Tensor.divide|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.divide_|Yes|supports fp16, fp32|
|Tensor.dot|Yes|supports fp16, fp32|
|Tensor.double|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Currently, some NPU interfaces do not support the double type. For compatibility, fp32 is returned by default. fp64 will be returned normally after support is completed.<br>may fall back to CPU|
|Tensor.dsplit|Yes|supports fp32|
|Tensor.element_size|Yes|supports fp32|
|Tensor.eq|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.eq_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.equal|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.erf|Yes|supports fp16, fp32, int64, bool|
|Tensor.erf_|Yes|supports fp16, fp32|
|Tensor.erfc|Yes|supports fp16, fp32, int64, bool|
|Tensor.erfc_|Yes|supports fp16, fp32|
|Tensor.erfinv|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.erfinv_|Yes|supports bf16, fp16, fp32|
|Tensor.exp|Yes|supports bf16, fp16, fp32, fp64, int64, bool|
|Tensor.exp_|Yes|supports bf16, fp16, fp32, complex64, complex128|
|Tensor.expm1|Yes|supports fp16, fp32, int64, bool|
|Tensor.expm1_|Yes|supports fp16, fp32|
|Tensor.expand|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.expand_as|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.exponential_|Yes|supports bf16, fp16, fp32, fp64|
|Tensor.fix|Yes|supports fp16, fp32|
|Tensor.fix_|Yes|supports fp16, fp32|
|Tensor.fill_|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.flatten|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.flip|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.fliplr|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.flipud|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.float|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.float_power|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex128|
|Tensor.float_power_|Yes|supports double|
|Tensor.floor|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.floor_|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.floor_divide|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.floor_divide_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.fmod|Yes|supports fp16, fp32, uint8, int8, int32, int64|
|Tensor.fmod_|Yes|supports fp16, fp32, uint8, int8, int32, int64|
|Tensor.frac|Yes|supports fp16, fp32|
|Tensor.frac_|Yes|supports fp16, fp32|
|Tensor.gather|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>The index dimension must be consistent with the input dimension|
|Tensor.ge|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.ge_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater_equal|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater_equal_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.geometric_|Yes|-|
|Tensor.ger|Yes|-|
|Tensor.get_device|Yes|-|
|Tensor.gt|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.gt_|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.half|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.hardshrink|Yes|supports fp16, fp32|
|Tensor.heaviside|Yes|may fall back to CPU|
|Tensor.histc|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.hsplit|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.index_add_|Yes|supports fp16, fp32, int64, bool|
|Tensor.index_add|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.index_copy_|Yes|supports fp16, fp32, int16, int32, bool|
|Tensor.index_copy|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.index_fill_|Yes|supports fp16, fp32, int32, int64, bool|
|Tensor.index_fill|Yes|supports bf16, fp16, fp32, int32, int64, bool|
|Tensor.index_put_|Yes|supports int64|
|Tensor.index_put|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.index_reduce_|Yes|may fall back to CPU|
|Tensor.index_reduce|Yes|may fall back to CPU|
|Tensor.index_select|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.indices|Yes|-|
|Tensor.inner|Yes|supports fp16, fp32|
|Tensor.int|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.int_repr|No|-|
|Tensor.isclose|Yes|supports fp16, fp32, uint8, int32, int64, bool|
|Tensor.isfinite|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.isinf|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.isposinf|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.isneginf|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.isnan|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.is_contiguous|Yes|supports fp32|
|Tensor.is_complex|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.is_conj|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.is_floating_point|Yes|supports fp32|
|Tensor.is_inference|Yes|supports fp32|
|Tensor.is_leaf|Yes|-|
|Tensor.is_pinned|Yes|supports fp32|
|Tensor.is_set_to|Yes|supports fp32|
|Tensor.is_shared|No|-|
|Tensor.is_signed|Yes|supports fp32|
|Tensor.is_sparse|Yes|supports fp32|
|Tensor.isreal|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.item|Yes|supports fp32|
|Tensor.kthvalue|Yes|supports fp16, fp32, int32|
|Tensor.ldexp|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.ldexp_|Yes|supports fp16, fp32, complex64, complex128|
|Tensor.le|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.le_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.less_equal|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.less_equal_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.lerp|Yes|supports fp16, fp32|
|Tensor.lerp_|Yes|supports fp16, fp32|
|Tensor.log|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.log_|Yes|supports bf16, fp16, fp32, complex64, complex128|
|Tensor.log10|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.log10_|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.log1p|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.log1p_|Yes|supports fp16, fp32|
|Tensor.log2|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.log2_|Yes|supports fp16, fp32, complex64, complex128|
|Tensor.logaddexp|Yes|supports fp16, fp32, int16, int32, int64, bool|
|Tensor.logaddexp2|Yes|supports fp16, fp32|
|Tensor.logsumexp|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.logical_and|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_and_|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_not|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.logical_not_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.logical_or|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_or_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_xor|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>may fall back to CPU|
|Tensor.logical_xor_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logit|Yes|supports bf16, fp16, fp32<br>output is nan when eps is greater than 1, output is inf when eps is 1|
|Tensor.logit_|Yes|supports bf16, fp16, fp32<br>output is nan when eps is greater than 1, output is inf when eps is 1|
|Tensor.long|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.lt|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.lt_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.less|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.less_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.as_subclass|Yes|-|
|Tensor.map_|Yes|CPU only|
|Tensor.masked_scatter_|Yes|supports fp32, int64, bool|
|Tensor.masked_scatter|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.masked_fill_|Yes|supports bf16, fp16, fp32, int8, int32, int64, bool|
|Tensor.masked_fill|Yes|supports bf16, fp16, fp32, int8, int32, int64, bool|
|Tensor.masked_select|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.matmul|Yes|supports bf16, fp16, fp32<br>supports Named Tensor|
|Tensor.matrix_power|Yes|supports fp16, fp32|
|Tensor.max|Yes|supports bf16, fp16, fp32, int64, bool|
|Tensor.maximum|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.mean|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.nanmean|Yes|supports fp16, fp32|
|Tensor.median|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64<br>when input is bf16, dim does not take the dimension where the input axis value is 1|
|Tensor.min|Yes|supports bf16, fp16, fp32, int64, bool|
|Tensor.minimum|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.mm|Yes|supports bf16, fp16, fp32|
|Tensor.smm|No|-|
|Tensor.mode|No|-|
|Tensor.movedim|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.moveaxis|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.msort|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.mul|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.mul_|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.multiply|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.multiply_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.multinomial|Yes|supports fp16, fp32|
|Tensor.nansum|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.narrow|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.narrow_copy|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.ndimension|Yes|supports fp32|
|Tensor.nan_to_num|Yes|-|
|Tensor.nan_to_num_|Yes|-|
|Tensor.ne|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.ne_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.nextafter_|Yes|falls back to CPU|
|Tensor.not_equal|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>may fall back to CPU|
|Tensor.not_equal_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.neg|Yes|supports bf16, fp16, fp32, int8, int32, int64|
|Tensor.neg_|Yes|supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128<br>may fall back to CPU|
|Tensor.negative|Yes|supports fp16, fp32, int8, int32, int64, complex64, complex128|
|Tensor.negative_|Yes|supports fp16, fp32, int8, int32, int64, complex64, complex128|
|Tensor.nelement|Yes|supports fp32|
|Tensor.nonzero|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>does not support nan scenarios|
|Tensor.norm|Yes|supports bf16, fp16, fp32, fp64|
|Tensor.normal_|Yes|supports bf16, fp16, fp32<br>may fall back to CPU|
|Tensor.numel|Yes|supports fp32|
|Tensor.numpy|Yes|supports fp32|
|Tensor.outer|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.permute|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.positive|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.pow|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.pow_|Yes|supports bf16, fp16, fp32, fp64, int64|
|Tensor.prod|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.put_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.qscheme|No|-|
|Tensor.quantile|Yes|-|
|Tensor.rad2deg|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.random_|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.ravel|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.reciprocal|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.reciprocal_|Yes|supports fp16, fp32, complex64, complex128|
|Tensor.record_stream|Yes|supports fp32|
|Tensor.register_hook|Yes|supports fp32|
|Tensor.register_post_accumulate_grad_hook|Yes|-|
|Tensor.remainder|Yes|supports bf16, fp16, fp32, fp64, int32, int64|
|Tensor.remainder_|Yes|supports fp16, fp32, int32, int64|
|Tensor.repeat|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.repeat_interleave|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool<br>the input tensor is repeated to produce the output, and the number of elements in the output must be less than $2^{22}$|
|Tensor.requires_grad|Yes|-|
|Tensor.requires_grad_|Yes|supports fp32|
|Tensor.reshape|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.reshape_as|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.resize_|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>memory_format only supports torch.contiguous_format and torch.preserve_format|
|Tensor.resize_as_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>memory_format only supports torch.contiguous_format and torch.preserve_format|
|Tensor.retain_grad|Yes|supports fp32|
|Tensor.retains_grad|Yes|supports fp32|
|Tensor.roll|Yes|supports bf16, fp16, fp32, uint8, int8, int32, int64, bool|
|Tensor.rot90|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.round|Yes|supports fp16, fp32|
|Tensor.round_|Yes|supports fp16, fp32|
|Tensor.rsqrt|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.rsqrt_|Yes|supports fp16, fp32, complex64, complex128|
|Tensor.scatter|Yes|supports fp16, fp32, fp64, int8, int16, int32, int64, bool|
|Tensor.scatter_|Yes|tensor, index, src parameters cannot be empty and cannot be scalar<br>may fall back to CPU|
|Tensor.scatter_add_|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.scatter_add|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.scatter_reduce|Yes|supports fp32, int64<br>may fall back to CPU|
|Tensor.select|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.select_scatter|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>may fall back to CPU|
|Tensor.set_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.share_memory_|No|-|
|Tensor.short|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.sigmoid|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.sigmoid_|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.sign|Yes|supports bf16, fp16, fp32, fp64, int32, int64, bool|
|Tensor.sign_|Yes|supports fp16, fp32, int32, int64, bool|
|Tensor.sgn|Yes|supports fp16, fp32, int32, int64, bool, complex64, complex128|
|Tensor.sgn_|Yes|supports fp16, fp32, fp64, int32, int64, bool|
|Tensor.sin|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.sin_|Yes|supports bf16, fp16, fp32, complex64, complex128|
|Tensor.sinh|Yes|supports fp16, fp32, fp64|
|Tensor.sinh_|Yes|supports fp16, fp32, fp64|
|Tensor.asinh|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.asinh_|Yes|supports fp16, fp32, complex64, complex128|
|Tensor.arcsinh|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.arcsinh_|Yes|supports fp16, fp32, complex64, complex128|
|Tensor.shape|Yes|-|
|Tensor.size|Yes|supports fp32|
|Tensor.slogdet|Yes|supports fp32, complex64, complex128|
|Tensor.slice_scatter|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.softmax||supports fp16, fp16, fp32, fp64|
|Tensor.sort|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.split|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.sparse_mask|No|-|
|Tensor.sparse_dim|Yes|-|
|Tensor.sqrt|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.sqrt_|Yes|supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.square|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.square_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.squeeze|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.squeeze_|Yes|supports fp32|
|Tensor.std|Yes|supports bf16, fp16, fp32<br>input does not support scalar tensor<br>correction is not greater than the range of int32|
|Tensor.storage|Yes|supports fp32|
|Tensor.untyped_storage|Yes|supports fp32|
|Tensor.storage_offset|Yes|supports fp32|
|Tensor.storage_type|Yes|supports fp32|
|Tensor.stride|Yes|supports fp32|
|Tensor.sub|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.sub_|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.subtract_|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.sum|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.sum_to_size|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.swapaxes|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.swapdims|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.t|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.t_|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.tensor_split|Yes|CPU only|
|Tensor.tile|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>If the length of the input parameter dims is less than the length of Tensor.shape, 1s are automatically prepended to dims to align its length with Tensor.shape. The completed dims must satisfy the following restrictions:<br>- When repeating the first axis, a maximum of 4 dimensions can be repeated simultaneously (i.e., the number of elements greater than 1 in dims ≤ 4). For example: Tensor.tile([2, 3, 4, 5, 6]) is not supported, Tensor.tile([2, 3, 1, 5, 6]) is supported<br>- When the first axis does not need to be repeated, a maximum of 3 dimensions can be repeated simultaneously (i.e., the number of elements greater than 1 in dims ≤ 3). For example: Tensor.tile([1, 3, 4, 5, 6]) is not supported, Tensor.tile([1, 3, 1, 5, 6]) is supported<br>- If backward computation is performed, the sum of the number of Tensor dimensions and the number of elements greater than 1 in the input parameter dims must not exceed 8|
|Tensor.to|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>The current NPU device only supports setting memory_format to torch.contiguous_format or torch.preserve_format<br>Atlas Inference Series Product does not support cross-NPU copy|
|to|Yes|-|
|Tensor.to_mkldnn|No|-|
|Tensor.take|Yes|supports fp16, fp32, int16, int32, bool|
|Tensor.take_along_dim|Yes|supports fp16, fp32, int16, int32, int64, bool|
|Tensor.tan|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128, value range [-65504, 65504]|
|Tensor.tan_|Yes|supports fp16, fp32, complex64, complex128|
|Tensor.tanh|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.tanh_|Yes|supports fp16, fp32|
|Tensor.atanh|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.atanh_|Yes|supports fp16, fp32, complex64, complex128|
|Tensor.arctanh|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.arctanh_|Yes|supports fp16, fp32, complex64, complex128|
|Tensor.tolist|Yes|supports fp32|
|Tensor.topk|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Due to hardware differences, the npu topk index results are inconsistent with gpu/cpu. Currently, npu only supports returning computation results with sorted set to true<br>Scalar tensor is not supported|
|Tensor.to_dense|No|-|
|Tensor.to_sparse|No|-|
|to_sparse|No|-|
|Tensor.to_sparse_csr|No|-|
|Tensor.to_sparse_csc|No|-|
|Tensor.to_sparse_bsr|No|-|
|Tensor.to_sparse_bsc|No|-|
|Tensor.transpose|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.transpose_|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.tril|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.tril_|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.triu|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.triu_|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.true_divide|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.true_divide_|Yes|supports fp16, fp32|
|Tensor.trunc|Yes|supports fp16, fp32|
|Tensor.trunc_|Yes|supports fp16, fp32|
|Tensor.type|Yes|supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.type_as|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unbind|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unflatten|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unfold|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.uniform_|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Following the PyTorch community specification, processing of bool type data is no longer supported. For existing bool type data, the following alternatives can be used: If the output needs to be all True, Tensor.bernoulli_(p=1.0) can be used. If uniformly distributed bool type output is needed, Tensor.bernoulli_(p=0.5) can be used|
|Tensor.unique|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>When the input contains 0, the output may include both positive 0 and negative 0, rather than outputting only one 0|
|Tensor.unique_consecutive|Yes|supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.unsqueeze|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unsqueeze_|Yes|supports fp32|
|Tensor.values|Yes|Depends on sparse tensor|
|Tensor.var|Yes|supports bf16, fp16, fp32<br>correction does not exceed the range of int32|
|Tensor.view|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|view|Yes|-|
|Tensor.view_as|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.vsplit|Yes|supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.where|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.xlogy|Yes|supports fp16, fp32|
|Tensor.xlogy_|Yes|supports fp16, fp32|
|Tensor.zero_|Yes|supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
