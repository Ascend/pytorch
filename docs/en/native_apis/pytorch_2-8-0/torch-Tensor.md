# torch.Tensor

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:34:00.040Z pushedAt=2026-07-09T08:44:08.316Z -->

> [!NOTE]   
> If an API's "Supported" column shows "Yes" and "Restrictions and Notes" shows "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.Tensor|Yes|-|
|Tensor.T|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.H|Yes|-|
|Tensor.mT|Yes|-|
|Tensor.mH|Yes|-|
|Tensor.new_tensor|Yes|-|
|Tensor.new_full|Yes|Supports int64|
|Tensor.new_empty|Yes|Supports fp32|
|Tensor.new_ones|Yes|Supports fp32|
|Tensor.new_zeros|Yes|Supports fp32|
|Tensor.is_cuda|Yes|-|
|Tensor.is_quantized|Yes|-|
|Tensor.is_meta|Yes|-|
|Tensor.device|Yes|-|
|Tensor.grad|Yes|Supports fp32|
|Tensor.ndim|Yes|Supports fp32|
|Tensor.real|Yes|-|
|Tensor.imag|Yes|-|
|Tensor.nbytes|Yes|-|
|Tensor.itemsize|Yes|-|
|Tensor.abs|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.abs_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.absolute|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.absolute_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.acos|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool. May Fall Back to CPU Execution|
|Tensor.acos_|Yes|Supports fp16, fp32|
|Tensor.arccos|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.arccos_|Yes|Supports fp16, fp32|
|Tensor.add|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.add_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.addbmm|Yes|Supports fp16, fp32|
|Tensor.addbmm_|Yes|Supports fp16, fp32|
|Tensor.addcdiv|Yes|Supports bf16, fp16, fp32, int64<br>int64 type does not support simultaneous broadcasting of three tensors|
|Tensor.addcdiv_|Yes|<term>Atlas A2 Training Series</term>/<term>Atlas A3 Training Series</term>: Supports bf16, fp16, fp32, fp64<br><term>Atlas Training Series</term>: Supports fp16, fp32, fp64<br>int64 type does not support simultaneous broadcasting of three tensors|
|Tensor.addcmul|Yes|Supports fp16, fp32, int64<br>int64 type does not support simultaneous broadcasting of three tensors|
|Tensor.addcmul_|Yes|<term>Atlas A2 Training Series</term>/<term>Atlas A3 Training Series</term>: Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64<br><term>Atlas Training Series</term>: Supports fp16, fp32, fp64, uint8, int8, int32, int64<br>int64 type does not support simultaneous broadcasting of three tensors|
|Tensor.addmm|Yes|Supports fp16, fp32|
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
|Tensor.apply_|Yes|CPU Only|
|Tensor.argmax|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.argmin|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
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
|Tensor.atan2_|Yes|Supports fp16, fp32<br>May Fall Back to CPU Execution|
|Tensor.arctan2|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.arctan2_|Yes|Supports fp16, fp32|
|Tensor.all|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.any|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.backward|Yes|Supports fp32|
|Tensor.baddbmm|Yes|Supports fp16, fp32|
|Tensor.baddbmm_|Yes|Supports fp16, fp32|
|Tensor.bernoulli|Yes|Supports fp16, fp32<br>May Fall Back to CPU Execution|
|Tensor.bernoulli_|Yes|May Fall Back to CPU Execution|
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
|Tensor.bmm|Yes|Supports fp16, fp32|
|Tensor.bool|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.byte|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.broadcast_to|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.ceil|Yes|Supports fp16, fp32|
|Tensor.ceil_|Yes|Supports fp16, fp32|
|Tensor.char|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.chunk|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.clamp|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.clamp_|Yes|May Fall Back to CPU Execution|
|Tensor.clip|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.clip_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.clone|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.contiguous|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.copy_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>int16 does not support 5 or more dimensions|
|Tensor.conj|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.resolve_conj|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.resolve_neg|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.copysign|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool<br>May Fall Back to CPU Execution|
|Tensor.cos|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.cos_|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|Tensor.cosh|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.cosh_|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.count_nonzero|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.cov|Yes|Supports fp16, fp32, int16, int32, int64|
|Tensor.acosh|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May Fall Back to CPU Execution|
|Tensor.acosh_|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.arccosh|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.arccosh_|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.cpu|Yes|-|
|Tensor.cross|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128<br>The shapes of the two inputs must be consistent|
|Tensor.cuda|Yes|The corresponding NPU interface is Tensor.npu. memory_format only supports passing torch.contiguous_format|
|Tensor.cummax|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>May Fall Back to CPU Execution|
|Tensor.cummin|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.cumsum|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Supports Named Tensor|
|Tensor.cumsum_|Yes|Supports fp16, fp32, int64, bool|
|Tensor.chalf|No|-|
|Tensor.cfloat|Yes|-|
|Tensor.cdouble|Yes|-|
|Tensor.data_ptr|Yes|Supports fp32|
|Tensor.deg2rad|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.dequantize|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.dense_dim|Yes|-|
|Tensor.detach|Yes|Supports fp32|
|Tensor.detach_|Yes|Supports fp32|
|Tensor.diag|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.diag_embed|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.diagflat|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.diagonal|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.diagonal_scatter|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.fill_diagonal_|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.diff|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.dim|Yes|Supports fp32|
|Tensor.dim_order|Yes|-|
|Tensor.dist|Yes|-|
|Tensor.div|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.div_|Yes|Supports bf16, fp16, fp32, fp64|
|Tensor.divide|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.divide_|Yes|Supports fp16, fp32|
|Tensor.dot|Yes|Supports fp16, fp32|
|Tensor.double|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Some NPU interfaces currently do not support the double type. For compatibility, fp32 is returned by default. fp64 will be returned normally once support is completed.<br>May Fall Back to CPU Execution|
|Tensor.dsplit|Yes|Supports fp32|
|Tensor.element_size|Yes|Supports fp32|
|Tensor.eq|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.eq_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.equal|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.erf|Yes|Supports fp16, fp32, int64, bool|
|Tensor.erf_|Yes|Supports fp16, fp32|
|Tensor.erfc|Yes|Supports fp16, fp32, int64, bool|
|Tensor.erfc_|Yes|Supports fp16, fp32|
|Tensor.erfinv|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.erfinv_|Yes|Supports bf16, fp16, fp32|
|Tensor.exp|Yes|Supports bf16, fp16, fp32, int64, bool, complex64, complex128|
|Tensor.exp_|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|Tensor.expm1|Yes|Supports fp16, fp32, int64, bool|
|Tensor.expm1_|Yes|Supports fp16, fp32|
|Tensor.expand|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.expand_as|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.exponential_|Yes|Supports bf16, fp16, fp32, fp64|
|Tensor.fix|Yes|Supports fp16, fp32|
|Tensor.fix_|Yes|Supports fp16, fp32|
|Tensor.fill_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.flatten|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.flip|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.fliplr|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.flipud|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.float|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.float_power|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex128|
|Tensor.float_power_|Yes|Supports double|
|Tensor.floor|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.floor_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.floor_divide|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.floor_divide_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.fmod|Yes|Supports fp16, fp32, uint8, int8, int32, int64|
|Tensor.fmod_|Yes|Supports fp16, fp32, uint8, int8, int32, int64|
|Tensor.frac|Yes|Supports fp16, fp32|
|Tensor.frac_|Yes|Supports fp16, fp32|
|Tensor.gather|Yes|Supports fp16, fp32, int64<br>The index dimension must be consistent with the input dimension|
|Tensor.ge|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.ge_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater_equal|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater_equal_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.geometric_|Yes|-|
|Tensor.ger|Yes|-|
|Tensor.get_device|Yes|-|
|Tensor.gt|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.gt_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.greater_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.half|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.hardshrink|Yes|Supports fp16, fp32|
|Tensor.heaviside|Yes|May Fall Back to CPU Execution|
|Tensor.histc|Yes|Supports fp16, fp32|
|Tensor.hsplit|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.index_add_|Yes|Supports fp16, fp32, int64, bool|
|Tensor.index_add|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.index_copy_|Yes|Supports fp16, fp32, int16, int32, bool|
|Tensor.index_copy|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.index_fill_|Yes|Supports fp16, fp32, int32, int64, bool|
|Tensor.index_fill|Yes|Supports fp16, fp32, int32, bool|
|Tensor.index_put_|Yes|Supports int64|
|Tensor.index_put|Yes|Supports int64|
|Tensor.index_reduce_|Yes|May Fall Back to CPU Execution|
|Tensor.index_reduce|Yes|May Fall Back to CPU Execution|
|Tensor.index_select|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|Tensor.indices|Yes|-|
|Tensor.inner|Yes|Supports fp16, fp32|
|Tensor.int|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.int_repr|No|-|
|Tensor.isclose|Yes|Supports fp16, fp32, uint8, int32, int64, bool|
|Tensor.isfinite|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.isinf|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.isposinf|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.isneginf|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.isnan|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.is_contiguous|Yes|Supports fp32|
|Tensor.is_complex|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.is_conj|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.is_floating_point|Yes|Supports fp32|
|Tensor.is_inference|Yes|Supports fp32|
|Tensor.is_leaf|Yes|-|
|Tensor.is_pinned|Yes|Supports fp32|
|Tensor.is_set_to|Yes|Supports fp32|
|Tensor.is_shared|No|-|
|Tensor.is_signed|Yes|Supports fp32|
|Tensor.is_sparse|Yes|Supports fp32|
|Tensor.isreal|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.item|Yes|Supports fp32|
|Tensor.kthvalue|Yes|Supports fp16, fp32, int32|
|Tensor.ldexp|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.ldexp_|Yes|Supports fp16, fp32, complex64, complex128|
|Tensor.le|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.le_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.less_equal|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.less_equal_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.lerp|Yes|Supports fp16, fp32|
|Tensor.lerp_|Yes|Supports fp16, fp32|
|Tensor.log|Yes|Supports bf16, fp16, fp32, int64, bool, complex64, complex128|
|Tensor.log_|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|Tensor.log10|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.log10_|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.log1p|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.log1p_|Yes|Supports fp16, fp32|
|Tensor.log2|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.log2_|Yes|Supports fp16, fp32, complex64, complex128|
|Tensor.logaddexp|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|Tensor.logaddexp2|Yes|Supports fp16, fp32|
|Tensor.logsumexp|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.logical_and|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_and_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_not|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.logical_not_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.logical_or|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_or_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logical_xor|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May Fall Back to CPU Execution|
|Tensor.logical_xor_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.logit|Yes|Supports bf16, fp16, fp32<br>Output is nan when eps is greater than 1, and inf when eps equals 1|
|Tensor.logit_|Yes|Supports bf16, fp16, fp32<br>Output is nan when eps is greater than 1, and inf when eps equals 1|
|Tensor.long|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.lt|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.lt_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.less|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.less_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.as_subclass|Yes|-|
|Tensor.map_|Yes|CPU Only|
|Tensor.masked_scatter_|Yes|Supports fp32, int64, bool|
|Tensor.masked_scatter|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.masked_fill_|Yes|Supports bf16, fp16, fp32, int8, int32, int64, bool|
|Tensor.masked_fill|Yes|Supports fp16, fp32, int64, bool|
|Tensor.masked_select|Yes|Supports fp32, bool|
|Tensor.matmul|Yes|Supports bf16, fp16, fp32<br>Supports Named Tensor|
|Tensor.matrix_power|Yes|Supports fp16, fp32|
|Tensor.max|Yes|Supports bf16, fp16, fp32, int64, bool|
|Tensor.maximum|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.mean|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|Tensor.nanmean|Yes|Supports fp16, fp32|
|Tensor.median|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64<br>When input is bf16, dim must not be a dimension where the input axis value is 1|
|Tensor.min|Yes|Supports bf16, fp16, fp32, int64, bool|
|Tensor.minimum|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.mm|Yes|Supports fp16, fp32|
|Tensor.smm|No|-|
|Tensor.mode|No|-|
|Tensor.movedim|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.moveaxis|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.msort|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.mul|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.mul_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.multiply|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.multiply_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.multinomial|Yes|Supports fp16, fp32|
|Tensor.nansum|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.narrow|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.narrow_copy|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.ndimension|Yes|Supports fp32|
|Tensor.nan_to_num|Yes|-|
|Tensor.nan_to_num_|Yes|-|
|Tensor.ne|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.ne_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.not_equal|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May Fall Back to CPU Execution|
|Tensor.not_equal_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.neg|Yes|Supports fp16, fp32, int8, int32, int64, complex64, complex128|
|Tensor.neg_|Yes|Supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128<br>May Fall Back to CPU Execution|
|Tensor.negative|Yes|Supports fp16, fp32, int8, int32, int64, complex64, complex128|
|Tensor.negative_|Yes|Supports fp16, fp32, int8, int32, int64, complex64, complex128|
|Tensor.nelement|Yes|Supports fp32|
|Tensor.nonzero|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>NaN scenarios not supported|
|Tensor.norm|Yes|Supports bf16, fp16, fp32, fp64|
|Tensor.normal_|Yes|Supports bf16, fp16, fp32<br>May Fall Back to CPU Execution|
|Tensor.numel|Yes|Supports fp32|
|Tensor.numpy|Yes|Supports fp32|
|Tensor.outer|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.permute|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.positive|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.pow|Yes|Supports bf16, fp16, fp32, fp64, int16, int64|
|Tensor.pow_|Yes|Supports bf16, fp16, fp32, fp64, int64|
|Tensor.prod|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.put_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.qscheme|No|-|
|Tensor.quantile|Yes|-|
|Tensor.rad2deg|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.random_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.ravel|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.reciprocal|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.reciprocal_|Yes|Supports fp16, fp32, complex64, complex128|
|Tensor.record_stream|Yes|Supports fp32|
|Tensor.register_hook|Yes|Supports fp32|
|Tensor.register_post_accumulate_grad_hook|Yes|-|
|Tensor.remainder|Yes|Supports fp16, fp32, int32, int64|
|Tensor.remainder_|Yes|Supports fp16, fp32, int32, int64|
|Tensor.repeat|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.repeat_interleave|Yes|Supports fp16, fp32, int16, int32, bool<br>The input tensor is repeated to produce the output, and the number of elements in the output must be less than $2^{22}$|
|Tensor.requires_grad|Yes|-|
|Tensor.requires_grad_|Yes|Supports fp32|
|Tensor.reshape|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.reshape_as|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.resize_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128. memory_format only supports torch.contiguous_format and torch.preserve_format|
|Tensor.resize_as_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128. memory_format only supports torch.contiguous_format and torch.preserve_format|
|Tensor.retain_grad|Yes|Supports fp32|
|Tensor.retains_grad|Yes|Supports fp32|
|Tensor.roll|Yes|Supports fp16, fp32, int32, int64, bool|
|Tensor.rot90|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.round|Yes|Supports fp16, fp32|
|Tensor.round_|Yes|Supports fp16, fp32|
|Tensor.rsqrt|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.rsqrt_|Yes|Supports fp16, fp32, complex64, complex128|
|Tensor.scatter|Yes|Supports fp16, fp32, int16, int32, bool|
|Tensor.scatter_|Yes|The tensor, index, and src parameters cannot be empty and cannot be scalars<br>May Fall Back to CPU Execution|
|Tensor.scatter_add_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.scatter_add|Yes|Supports fp32|
|Tensor.scatter_reduce|Yes|Supports fp32, int64<br>May Fall Back to CPU Execution|
|Tensor.select|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.select_scatter|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool<br>May Fall Back to CPU Execution|
|Tensor.set_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.share_memory_|No|-|
|Tensor.short|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.sigmoid_|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.sign|Yes|Supports bf16, fp16, fp32, int32, int64, bool|
|Tensor.sign_|Yes|Supports fp16, fp32, int32, int64, bool|
|Tensor.sgn|Yes|Supports fp16, fp32, int32, int64, bool, complex64, complex128|
|Tensor.sgn_|Yes|Supports fp16, fp32, fp64, int32, int64, bool|
|Tensor.sin|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.sin_|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|Tensor.sinh|Yes|Supports fp16, fp32, fp64|
|Tensor.sinh_|Yes|Supports fp16, fp32, fp64|
|Tensor.asinh|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.asinh_|Yes|Supports fp16, fp32, complex64, complex128|
|Tensor.arcsinh|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.arcsinh_|Yes|Supports fp16, fp32, complex64, complex128|
|Tensor.shape|Yes|-|
|Tensor.size|Yes|Supports fp32|
|Tensor.slogdet|Yes|Supports fp32, complex64, complex128|
|Tensor.slice_scatter|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.softmax|Yes|Supports fp16, fp32, fp64|
|Tensor.sort|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.split|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.sparse_mask|No|-|
|Tensor.sparse_dim|Yes|-|
|Tensor.sqrt|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|Tensor.sqrt_|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|Tensor.square|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.square_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.squeeze|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.squeeze_|Yes|Supports fp32|
|Tensor.std|Yes|Supports bf16, fp16, fp32<br>input does not support scalar tensors<br>correction must not exceed the range of int32|
|Tensor.storage|Yes|Supports fp32|
|Tensor.untyped_storage|Yes|Supports fp32|
|Tensor.storage_offset|Yes|Supports fp32|
|Tensor.storage_type|Yes|Supports fp32|
|Tensor.stride|Yes|Supports fp32|
|Tensor.sub|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.sub_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.subtract_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|Tensor.sum|Yes|Supports bf16, fp16, fp32, int32|
|Tensor.sum_to_size|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.swapaxes|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.swapdims|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.t|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, complex64, complex128|
|Tensor.t_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64|
|Tensor.tensor_split|Yes|CPU Only|
|Tensor.tile|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>If the length of the dims argument is less than the length of Tensor.shape, 1s are automatically prepended to dims to align its length with Tensor.shape. The padded dims must satisfy the following constraints:<br>- When repeating along the first axis, a maximum of 4 dimensions can be repeated simultaneously (i.e., the number of elements greater than 1 in dims ≤ 4). For example: Tensor.tile([2, 3, 4, 5, 6]) is not supported, while Tensor.tile([2, 3, 1, 5, 6]) is supported.<br>- When not repeating along the first axis, a maximum of 3 dimensions can be repeated simultaneously (i.e., the number of elements greater than 1 in dims ≤ 3). For example: Tensor.tile([1, 3, 4, 5, 6]) is not supported, while Tensor.tile([1, 3, 1, 5, 6]) is supported.<br>- If backpropagation is performed, the sum of the number of Tensor dimensions and the number of elements greater than 1 in dims must not exceed 8|
|Tensor.to|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Currently, the NPU device only supports setting memory_format to torch.contiguous_format or torch.preserve_format<br><term>Atlas Inference Series</term> does not support cross-NPU copying|
|to|Yes|-|
|Tensor.to_mkldnn|No|-|
|Tensor.take|Yes|Supports fp16, fp32, int16, int32, bool|
|Tensor.take_along_dim|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|Tensor.tan|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Value range [-65504, 65504]|
|Tensor.tan_|Yes|Supports fp16, fp32, complex64, complex128|
|Tensor.tanh|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|Tensor.tanh_|Yes|Supports fp16, fp32|
|Tensor.atanh|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.atanh_|Yes|Supports fp16, fp32, complex64, complex128|
|Tensor.arctanh|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.arctanh_|Yes|Supports fp16, fp32, complex64, complex128|
|Tensor.tolist|Yes|Supports fp32|
|Tensor.topk|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Due to hardware differences, NPU topk index results may differ from GPU/CPU. Currently, NPU only supports returning results with sorted=true<br>Scalar tensors are not supported|
|Tensor.to_dense|No|-|
|Tensor.to_sparse|No|-|
|to_sparse|No|-|
|Tensor.to_sparse_csr|No|-|
|Tensor.to_sparse_csc|No|-|
|Tensor.to_sparse_bsr|No|-|
|Tensor.to_sparse_bsc|No|-|
|Tensor.transpose|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.transpose_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.tril|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.tril_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.triu|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.triu_|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.true_divide|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.true_divide_|Yes|Supports fp16, fp32|
|Tensor.trunc|Yes|Supports fp16, fp32|
|Tensor.trunc_|Yes|Supports fp16, fp32|
|Tensor.type|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.type_as|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unbind|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unflatten|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unfold|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|Tensor.uniform_|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Following PyTorch community specifications, processing of bool type data is no longer supported. For existing bool type data, the following alternatives can be used: if all-True output is needed, use Tensor.bernoulli_(p=1.0). If uniformly distributed bool type output is needed, use Tensor.bernoulli_(p=0.5)|
|Tensor.unique|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>When the input contains 0, the output may include both positive zero and negative zero, rather than only a single zero|
|Tensor.unique_consecutive|No|-|
|Tensor.unsqueeze|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.unsqueeze_|Yes|Supports fp32|
|Tensor.values|Yes|Depends on sparse tensor|
|Tensor.var|Yes|Supports bf16, fp16, fp32<br>correction does not exceed the int32 range|
|Tensor.view|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|view|Yes|-|
|Tensor.view_as|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.vsplit|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.where|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|Tensor.xlogy|Yes|Supports fp16, fp32|
|Tensor.xlogy_|Yes|Supports fp16, fp32|
|Tensor.zero_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
