# torch

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:37:36.490Z pushedAt=2026-07-09T08:44:08.363Z -->

> [!NOTE]  
> If the "Supported" column of an API is "Yes" and "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.default_generator|Yes|-|
|torch.SymInt|Yes|Supports fp32|
|torch.SymFloat|Yes|Supports fp32|
|torch.SymBool|Yes|Supports fp32|
|torch.Tag|Yes|-|
|torch.Tag.name|Yes|-|
|torch.is_tensor|Yes|-|
|torch.is_storage|Yes|-|
|torch.is_complex|Yes|Supports complex64, complex128|
|torch.is_conj|Yes|-|
|torch.is_floating_point|Yes|-|
|torch.is_nonzero|Yes|-|
|torch.set_default_dtype|Yes|-|
|torch.get_default_dtype|Yes|-|
|torch.set_default_device|Yes|-|
|torch.set_default_tensor_type|Yes|Noted: passing torch.npu.DtypeTensor type|
|torch.numel|Yes|-|
|torch.set_printoptions|Yes|-|
|torch.set_flush_denormal|Yes|-|
|torch.tensor|Yes|-|
|torch.sparse_coo_tensor|Yes|indices supports int32, int64<br>values supports fp16, fp32, int32<br>The dtype parameter must be consistent with the dtype of values|
|torch.sparse_csr_tensor|No|-|
|torch.sparse_csc_tensor|No|-|
|torch.sparse_bsr_tensor|No|-|
|torch.sparse_bsc_tensor|No|-|
|torch.asarray|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.as_tensor|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.as_strided|Yes|Supports fp32|
|torch.from_numpy|Yes|Supports output of fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.from_dlpack|No|-|
|torch.frombuffer|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.zeros|Yes|Supports bf16, fp16, fp32, uint8, int16, int32, int64, bool, complex64, complex128|
|torch.zeros_like|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.ones|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.ones_like|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.arange|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.range|Yes|-|
|torch.linspace|Yes|Supports bf16, fp16, fp32, fp64, int16, int32, int64, bool, complex64, complex128<br>Creates a 1-dimensional vector with a sequence size of steps|
|torch.eye|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.empty|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.empty_like|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.empty_strided|Yes|-|
|torch.full|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.full_like|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.quantize_per_tensor|No|-|
|torch.quantize_per_channel|No|-|
|torch.dequantize|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.complex|Yes|-|
|torch.polar|Yes|Supports fp32<br>The dimensions of input parameters abs and angle must be equal|
|torch.heaviside|No|-|
|torch.argwhere|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.cat|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.concat|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64|
|torch.concatenate|Yes|Supports bf16, fp16, fp32, int64, bool, complex64|
|torch.conj|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.chunk|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.dsplit|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.dstack|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|torch.gather|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>The Number of Dimensions of index must be consistent with the Number of Dimensions of input|
|torch.hsplit|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.hstack|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64|
|torch.index_add|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.index_copy|Yes|Supports fp32|
|torch.index_reduce|Yes|Possible Fallback to CPU Execution|
|torch.index_select|Yes|Supports bf16, fp16, fp32, uint8, int16, int32, int64, bool, complex64, complex128|
|torch.masked_select|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.movedim|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.moveaxis|Yes|Supports fp32, int64, complex128|
|torch.narrow|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.narrow_copy|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool. Possible Fallback to CPU Execution|
|torch.nonzero|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.permute|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.reshape|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.row_stack|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|torch.scatter|Yes|Supports fp16, fp32, fp64, int8, int16, int32, int64, bool<br>Possible Fallback to CPU Execution|
|torch.diagonal_scatter|Yes|Supports bf16, fp16, fp32, int16, int32, int64, bool|
|torch.select_scatter|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.slice_scatter|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.scatter_add|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.scatter_reduce|No|-|
|torch.split|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.squeeze|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.stack|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.swapaxes|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.swapdims|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.t|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.take|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|torch.take_along_dim|Yes|Supports fp32|
|torch.tensor_split|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.tile|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>If the length of the input parameter dims is less than the length of input.shape, ones will be automatically prepended to dims to align its length with input.shape. The padded dims must satisfy the following restrictions:<br>- When repeating along the first axis, a maximum of 4 dimensions may be repeated simultaneously (i.e., the Number of Elements Greater Than 1 in dims ≤ 4). For example: torch.tile(input, [2, 3, 4, 5, 6]) is Not Supported, while torch.tile(input, [2, 3, 1, 5, 6]) is supported.<br>- When not repeating along the first axis, a maximum of 3 dimensions may be repeated simultaneously (i.e., the Number of Elements Greater Than 1 in dims ≤ 3). For example: torch.tile(input, [1, 3, 4, 5, 6]) is Not Supported, while torch.tile(input, [1, 3, 1, 5, 6]) is supported.<br>- If Backpropagation is performed, the sum of the Number of Dimensions of the input Tensor and the Number of Elements Greater Than 1 in the input parameter dims must not exceed 8|
|torch.transpose|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.unsqueeze|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.vsplit|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.vstack|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64|
|torch.where|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>Not Supported: 8-dimensional shapes|
|torch.Generator|Yes|-|
|torch.Generator.device|Yes|-|
|torch.Generator.get_state|Yes|-|
|torch.Generator.initial_seed|Yes|-|
|torch.Generator.manual_seed|Yes|-|
|torch.Generator.seed|Yes|-|
|torch.Generator.set_state|Yes|-|
|torch.seed|Yes|-|
|torch.manual_seed|Yes|-|
|torch.initial_seed|Yes|-|
|torch.get_rng_state|Yes|-|
|torch.set_rng_state|Yes|-|
|torch.bernoulli|Yes|Supports fp16, fp32, fp64|
|torch.multinomial|Yes|Supports fp16, fp32|
|torch.normal|Yes|Supports fp16, fp32, fp64|
|torch.poisson|No|-|
|torch.rand|Yes|-|
|torch.rand_like|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Following PyTorch community specifications, processing of bool type data is no longer supported. For existing bool type data, the following workaround can be used: if you need to output all True, use torch.bernoulli(input, 1). If you need uniformly distributed bool type output, use torch.bernoulli(input, 0.5)|
|torch.randint|Yes|-|
|torch.randint_like|Yes|Supports fp16, fp32, int64|
|torch.randn|Yes|-|
|torch.randn_like|Yes|Supports fp32|
|torch.randperm|Yes|-|
|torch.save|Yes|-|
|torch.load|Yes|-|
|torch.get_num_threads|Yes|-|
|torch.set_num_threads|Yes|-|
|torch.get_num_interop_threads|Yes|-|
|torch.set_num_interop_threads|Yes|-|
|torch.no_grad|Yes|-|
|torch.enable_grad|Yes|-|
|torch.autograd.grad_mode.set_grad_enabled|Yes|-|
|torch.is_grad_enabled|Yes|-|
|torch.autograd.grad_mode.inference_mode|Yes|-|
|torch.is_inference_mode_enabled|Yes|-|
|torch.abs|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.absolute|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|torch.acos|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.arccos|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.acosh|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Possible Fallback to CPU Execution|
|torch.arccosh|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.add|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.addcdiv|Yes|Supports fp16, fp32, int64<br>Simultaneous broadcasting of three tensors is not supported for int64 type|
|torch.addcmul|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64<br>Simultaneous broadcasting of three tensors is not supported for fp64, uint8, int8, int64 types|
|torch.angle|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|torch.asin|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.arcsin|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.asinh|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.arcsinh|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.atan|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.arctan|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.atanh|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.arctanh|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.atan2|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.arctan2|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.bitwise_not|Yes|uint8, int8, int16, int32, int64, bool|
|torch.bitwise_and|Yes|uint8, int8, int16, int32, int64, bool|
|torch.bitwise_or|Yes|uint8, int8, int16, int32, int64, bool|
|torch.bitwise_xor|Yes|uint8, int8, int16, int32, int64, bool|
|torch.bitwise_left_shift|Yes|uint8, int8, int16, int32, int64|
|torch.ceil|Yes|Supports fp16, fp32|
|torch.clamp|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.clip|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.copysign|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool<br>Possible Fallback to CPU Execution|
|torch.cos|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.cosh|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.deg2rad|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.div|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.divide|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.erf|Yes|Supports bf16, fp16, fp32, fp64, int64, bool|
|torch.erfc|Yes|Supports fp16, fp32, int64, bool|
|torch.erfinv|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.exp|Yes|Supports bf16, fp16, fp32, fp64, int64, bool|
|torch.exp2|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.expm1|Yes|Supports bf16, fp16, fp32, fp64, int64, bool|
|torch.fix|Yes|Supports bf16, fp16, fp32|
|torch.float_power|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex128|
|torch.floor|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.floor_divide|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.fmod|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64|
|torch.gradient|Yes|Supports bf16, fp16, fp32, int8, int16, int32, int64|
|torch.ldexp|Yes|Supports fp16, fp64, complex64|
|torch.lerp|Yes|Supports fp16, fp32|
|torch.log|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.log10|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>When input is uint8, int8, int16, int32, int64, bool, output out must be fp32<br>For other supported data types, output out remains consistent with input|
|torch.log1p|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.log2|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.logaddexp|Yes|double data type is not supported|
|torch.logaddexp2|Yes|double data type is not supported|
|torch.logical_and|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.logical_not|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.logical_or|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.logical_xor|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.logit|Yes|Supports bf16, fp16, fp32<br>When eps is greater than 1, output is nan; when eps equals 1, output is inf|
|torch.mul|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.multiply|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nan_to_num|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.neg|Yes|Supports bf16, fp16, fp32, int8, int32, int64|
|torch.negative|Yes|Supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128|
|torch.positive|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|torch.pow|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.rad2deg|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.real|Yes|Supports fp16, fp32, complex64, complex128|
|torch.reciprocal|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.remainder|Yes|Supports bf16, fp16, fp32, fp64, int32, int64|
|torch.round|Yes|Supports bf16, fp16, fp32, fp64, int32, int64|
|torch.rsqrt|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.sigmoid|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.sign|Yes|Supports bf16, fp16, fp32, fp64, int32, int64, bool|
|torch.sgn|Yes|Supports bf16, fp16, fp32, int32, int64, bool, complex64, complex128|
|torch.sin|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.sinh|Yes|Supports fp16, fp32, fp64|
|torch.softmax|Yes|Supports bf16, fp16, fp32, fp64<br>Supports Named Tensor|
|torch.sqrt|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.square|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.sub|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.tan|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Value range [-65504, 65504]|
|torch.tanh|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.true_divide|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.trunc|Yes|Supports fp16, fp32<br>Possible Fallback to CPU Execution|
|torch.xlogy|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.argmax|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.argmin|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.amax|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.amin|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.aminmax|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.all|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.any|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.max|Yes|Supports bf16, fp16, fp32, int64, bool|
|torch.min|Yes|Supports bf16, fp16, fp32, int64, bool|
|torch.dist|Yes|Supports bf16, fp16, fp32|
|torch.logsumexp|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.mean|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128|
|torch.median|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|torch.norm|Yes|Supports bf16, fp16, fp32, fp64<br>When parameter p is negative, the computation result may have precision errors<br>When parameter dim specifies an axis with a shape dimension value of 1 in the input tensor, the computation result may have precision errors|
|torch.nansum|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.prod|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nanquantile|No|-|
|torch.std|Yes|Possible Fallback to CPU Execution|
|torch.std_mean|No|-|
|torch.sum|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>Does not support the dtype parameter|
|torch.unique|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>fp16 is not supported in scenarios with dim<br>When the input contains 0, the output may include both positive 0 and negative 0, rather than only one 0|
|torch.unique_consecutive|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.var|Yes|Supports fp16, fp32|
|torch.var_mean|No|-|
|torch.count_nonzero|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.allclose|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.argsort|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|torch.eq|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.equal|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.ge|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.greater_equal|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.gt|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.greater|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.isclose|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.isfinite|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.isin|Yes|Constraints for the dual-tensor input scenario are as follows:<br>- Supports fp16, fp32, uint8, int8, int16, int32, int64<br>- The first input tensor cannot have more than 7 dimensions, and the second input tensor cannot have more than 8 dimensions<br>Constraints for the single-tensor input scenario are as follows:<br>- Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>- The input tensor cannot have more than 8 dimensions|
|torch.isinf|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.isposinf|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.isneginf|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.isnan|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.kthvalue|Yes|Supports fp16, fp32, int32|
|torch.le|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.less_equal|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.lt|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.less|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.maximum|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.minimum|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.ne|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.not_equal|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.sort|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|torch.topk|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Does not support the sorted=False scenario|
|torch.msort|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|torch.stft|Yes|Supports fp32, fp64, complex64, complex128<br>If the operator times out, use the official interface set_op_execute_time_out to increase the timeout threshold to extend the judgment time|
|torch.hann_window|Yes|Supports bf16, fp16, fp32<br>When the data type is fp32 and the parameter window_length is greater than 10000, the computation result may have errors|
|torch.atleast_1d|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.atleast_2d|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.atleast_3d|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.bincount|Yes|Supports uint8, int8, int16, int32, int64<br>The weights dimension must be consistent with the input dimension|
|torch.block_diag|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.broadcast_tensors|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.broadcast_to|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.broadcast_shapes|Yes|-|
|torch.cdist|Yes|Supports bf16, fp16, fp32<br>When p=2.0, "compute_mode" only supports the "donot_use_mm_for_euclid_dist" mode; passing other values will be automatically changed to this mode|
|torch.clone|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.combinations|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.cov|Yes|Supports fp32|
|torch.cross|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128<br>The shapes of the two inputs must be consistent|
|torch.cummax|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.cummin|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>When the input is int32, the value range is within [-16777216, 16777216]|
|torch.cumprod|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.cumsum|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Supports Named Tensor|
|torch.diag|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|torch.diag_embed|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.diagonal|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.diff|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.einsum|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.flatten|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.flip|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.fliplr|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.flipud|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.kron|Yes|Does not support inputs with 5 or more dimensions|
|torch.histc|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64<br>When the input tensor value falls on the boundary between counting intervals, whether it is counted in the left interval or the right interval may have errors|
|torch.meshgrid|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.ravel|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.repeat_interleave|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool<br>The input tensor is repeated to produce the output, and the number of elements in the output must be less than $2^{22}$|
|torch.roll|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64, bool|
|torch.searchsorted|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.tensordot|Yes|Supports fp16, fp32|
|torch.tril|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.tril_indices|Yes|-|
|torch.triu|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.triu_indices|Yes|-|
|torch.unflatten|Yes|-|
|torch.view_as_real|Yes|Supports complex64, complex128|
|torch.view_as_complex|Yes|Supports fp32, fp64|
|torch.resolve_conj|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.resolve_neg|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.addbmm|Yes|Supports fp16, fp32|
|torch.addmm|Yes|Supports bf16, fp16, fp32|
|torch.addmv|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|torch.addr|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.baddbmm|Yes|Supports bf16, fp16, fp32|
|torch.bmm|Yes|Supports bf16, fp16, fp32|
|torch.dot|Yes|Supports bf16, fp16, fp32, uint8, int8, int32|
|torch.slogdet|Yes|Supports fp32, complex64, complex128<br>Possible Fallback to CPU Execution|
|torch.matmul|Yes|Supports bf16, fp16, fp32<br>Supports Named Tensor<br>Input supports up to 6 dimensions|
|torch.mm|Yes|Supports bf16, fp16, fp32|
|torch.outer|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.qr|Yes|-|
|torch.trapezoid|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|torch.cumulative_trapezoid|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.vdot|Yes|Supports fp16, fp32|
|torch.compiled_with_cxx11_abi|Yes|-|
|torch.result_type|Yes|Supports fp32|
|torch.can_cast|Yes|-|
|torch.promote_types|Yes|-|
|torch.use_deterministic_algorithms|Yes|When both HCCL_DETERMINISTIC and torch.use_deterministic_algorithms are set, if HCCL_DETERMINISTIC enables determinism, the HCCL interface uses determinism; otherwise, HCCL determinism is controlled by the torch.use_deterministic_algorithms interface|
|torch.are_deterministic_algorithms_enabled|Yes|-|
|torch.is_deterministic_algorithms_warn_only_enabled|No|-|
|torch.set_deterministic_debug_mode|Yes|-|
|torch.get_deterministic_debug_mode|Yes|-|
|torch.set_float32_matmul_precision|Yes|-|
|torch.get_float32_matmul_precision|Yes|-|
|torch.set_warn_always|Yes|-|
|torch.is_warn_always_enabled|Yes|-|
|torch.vmap|Yes|-|
|torch._assert|Yes|-|
|torch.sym_float|Yes|Supports fp32|
|torch.sym_int|Yes|Supports fp32|
|torch.sym_max|Yes|Supports fp32|
|torch.sym_min|Yes|Supports fp32|
|torch.sym_not|Yes|Supports fp32|
|torch.compile|Yes|The backend supports npugraphs, with overall functionality consistent with backend="cudagraphs"|
|torch.bucketize|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|torch.cartesian_prod|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.mv|Yes|Supports fp16, fp32|
|torch.quasirandom.SobolEngine.draw|Yes|Supports fp32, fp64|
|torch._foreach_sqrt|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch._foreach_asin|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.\_foreach\_neg\_|Yes|Supports bf16, fp16, fp32, int8, int32, int64|
