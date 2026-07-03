# torch

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T08:01:51.320Z pushedAt=2026-06-14T09:16:34.804Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.default_generator](https://pytorch.org/docs/2.10/torch.html#torch.torch.default_generator)|Yes|-|
|[torch.SymInt](https://pytorch.org/docs/2.10/torch.html#torch.SymInt)|Yes|Supports fp32|
|[torch.SymFloat](https://pytorch.org/docs/2.10/torch.html#torch.SymFloat)|Yes|Supports fp32|
|[torch.SymBool](https://pytorch.org/docs/2.10/torch.html#torch.SymBool)|Yes|Supports fp32|
|[torch.Tag](https://pytorch.org/docs/2.10/torch.html#torch.Tag)|Yes|-|
|[torch.Tag.name](https://pytorch.org/docs/2.10/torch.html#torch.Tag.name)|Yes|-|
|[torch.is_tensor](https://pytorch.org/docs/2.10/generated/torch.is_tensor.html)|Yes|-|
|[torch.is_storage](https://pytorch.org/docs/2.10/generated/torch.is_storage.html)|Yes|-|
|[torch.is_complex](https://pytorch.org/docs/2.10/generated/torch.is_complex.html)|Yes|Supports complex64, complex128|
|[torch.is_conj](https://pytorch.org/docs/2.10/generated/torch.is_conj.html)|Yes|-|
|[torch.is_floating_point](https://pytorch.org/docs/2.10/generated/torch.is_floating_point.html)|Yes|-|
|[torch.is_nonzero](https://pytorch.org/docs/2.10/generated/torch.is_nonzero.html)|Yes|-|
|[torch.set_default_dtype](https://pytorch.org/docs/2.10/generated/torch.set_default_dtype.html)|Yes|-|
|[torch.get_default_dtype](https://pytorch.org/docs/2.10/generated/torch.get_default_dtype.html)|Yes|-|
|[torch.set_default_device](https://pytorch.org/docs/2.10/generated/torch.set_default_device.html)|Yes|-|
|[torch.set_default_tensor_type](https://pytorch.org/docs/2.10/generated/torch.set_default_tensor_type.html)|Yes|Does not support passing torch.npu.DtypeTensor types|
|[torch.numel](https://pytorch.org/docs/2.10/generated/torch.numel.html)|Yes|-|
|[torch.set_printoptions](https://pytorch.org/docs/2.10/generated/torch.set_printoptions.html)|Yes|-|
|[torch.set_flush_denormal](https://pytorch.org/docs/2.10/generated/torch.set_flush_denormal.html)|Yes|-|
|[torch.tensor](https://pytorch.org/docs/2.10/generated/torch.tensor.html)|Yes|-|
|[torch.sparse_coo_tensor](https://pytorch.org/docs/2.10/generated/torch.sparse_coo_tensor.html)|Yes|Indices support int32, int64<br>Values support fp16, fp32, int32<br>The dtype parameter must be consistent with the dtype of values|
|[torch.sparse_csr_tensor](https://pytorch.org/docs/2.10/generated/torch.sparse_csr_tensor.html)|No|-|
|[torch.sparse_csc_tensor](https://pytorch.org/docs/2.10/generated/torch.sparse_csc_tensor.html)|No|-|
|[torch.sparse_bsr_tensor](https://pytorch.org/docs/2.10/generated/torch.sparse_bsr_tensor.html)|No|-|
|[torch.sparse_bsc_tensor](https://pytorch.org/docs/2.10/generated/torch.sparse_bsc_tensor.html)|No|-|
|[torch.asarray](https://pytorch.org/docs/2.10/generated/torch.asarray.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.as_tensor](https://pytorch.org/docs/2.10/generated/torch.as_tensor.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.as_strided](https://pytorch.org/docs/2.10/generated/torch.as_strided.html)|Yes|Supports fp32|
|[torch.from_numpy](https://pytorch.org/docs/2.10/generated/torch.from_numpy.html)|Yes|Supports output fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.from_dlpack](https://pytorch.org/docs/2.10/generated/torch.from_dlpack.html)|No|-|
|[torch.frombuffer](https://pytorch.org/docs/2.10/generated/torch.frombuffer.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.zeros](https://pytorch.org/docs/2.10/generated/torch.zeros.html)|Yes|-|
|[torch.zeros_like](https://pytorch.org/docs/2.10/generated/torch.zeros_like.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.ones](https://pytorch.org/docs/2.10/generated/torch.ones.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.ones_like](https://pytorch.org/docs/2.10/generated/torch.ones_like.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.arange](https://pytorch.org/docs/2.10/generated/torch.arange.html)|Yes|Supports bf16, fp16, fp32, fp64, int32, int64|
|[torch.range](https://pytorch.org/docs/2.10/generated/torch.range.html)|Yes|-|
|[torch.linspace](https://pytorch.org/docs/2.10/generated/torch.linspace.html)|Yes|Supports bf16, fp16, fp32, fp64, int16, int32, int64, bool, complex64, complex128<br>Creates a 1-dimensional vector with a sequence size of steps|
|[torch.eye](https://pytorch.org/docs/2.10/generated/torch.eye.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.empty](https://pytorch.org/docs/2.10/generated/torch.empty.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.empty_like](https://pytorch.org/docs/2.10/generated/torch.empty_like.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.empty_strided](https://pytorch.org/docs/2.10/generated/torch.empty_strided.html)|Yes|-|
|[torch.full](https://pytorch.org/docs/2.10/generated/torch.full.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.full_like](https://pytorch.org/docs/2.10/generated/torch.full_like.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.quantize_per_tensor](https://pytorch.org/docs/2.10/generated/torch.quantize_per_tensor.html)|No|-|
|[torch.quantize_per_channel](https://pytorch.org/docs/2.10/generated/torch.quantize_per_channel.html)|No|-|
|[torch.dequantize](https://pytorch.org/docs/2.10/generated/torch.dequantize.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.complex](https://pytorch.org/docs/2.10/generated/torch.complex.html)|Yes|-|
|[torch.polar](https://pytorch.org/docs/2.10/generated/torch.polar.html)|Yes|Supports fp32<br>The dimensions of the input parameters abs and angle must be equal|
|[torch.heaviside](https://pytorch.org/docs/2.10/generated/torch.heaviside.html)|No|-|
|[torch.argwhere](https://pytorch.org/docs/2.10/generated/torch.argwhere.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.cat](https://pytorch.org/docs/2.10/generated/torch.cat.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.concat](https://pytorch.org/docs/2.10/generated/torch.concat.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64|
|[torch.concatenate](https://pytorch.org/docs/2.10/generated/torch.concatenate.html)|Yes|Supports bf16, fp16, fp32, int64, bool, complex64|
|[torch.conj](https://pytorch.org/docs/2.10/generated/torch.conj.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.chunk](https://pytorch.org/docs/2.10/generated/torch.chunk.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.dsplit](https://pytorch.org/docs/2.10/generated/torch.dsplit.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.dstack](https://pytorch.org/docs/2.10/generated/torch.dstack.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|[torch.gather](https://pytorch.org/docs/2.10/generated/torch.gather.html)|Yes|Supports fp16, fp32, int16, int32, int64, bool<br>The number of dimensions of index must be consistent with the number of dimensions of input|
|[torch.hsplit](https://pytorch.org/docs/2.10/generated/torch.hsplit.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.hstack](https://pytorch.org/docs/2.10/generated/torch.hstack.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64|
|[torch.index_add](https://pytorch.org/docs/2.10/generated/torch.index_add.html)|Yes|Supports fp16, fp32, int64, bool|
|[torch.index_copy](https://pytorch.org/docs/2.10/generated/torch.index_copy.html)|Yes|Supports fp32|
|[torch.index_reduce](https://pytorch.org/docs/2.10/generated/torch.index_reduce.html)|Yes|May fall back to CPU execution|
|[torch.index_select](https://pytorch.org/docs/2.10/generated/torch.index_select.html)|Yes|Supports bf16, fp16, fp32, int16, int32, int64, bool|
|[torch.masked_select](https://pytorch.org/docs/2.10/generated/torch.masked_select.html)|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|[torch.movedim](https://pytorch.org/docs/2.10/generated/torch.movedim.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.moveaxis](https://pytorch.org/docs/2.10/generated/torch.moveaxis.html)|Yes|Supports fp32, int64, complex128|
|[torch.narrow](https://pytorch.org/docs/2.10/generated/torch.narrow.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.narrow_copy](https://pytorch.org/docs/2.10/generated/torch.narrow_copy.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU execution|
|[torch.nonzero](https://pytorch.org/docs/2.10/generated/torch.nonzero.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.permute](https://pytorch.org/docs/2.10/generated/torch.permute.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.reshape](https://pytorch.org/docs/2.10/generated/torch.reshape.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.row_stack](https://pytorch.org/docs/2.10/generated/torch.row_stack.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|[torch.scatter](https://pytorch.org/docs/2.10/generated/torch.scatter.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May fall back to CPU execution|
|[torch.diagonal_scatter](https://pytorch.org/docs/2.10/generated/torch.diagonal_scatter.html)|Yes|Supports bf16, fp16, fp32, int16, int32, int64, bool|
|[torch.select_scatter](https://pytorch.org/docs/2.10/generated/torch.select_scatter.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.slice_scatter](https://pytorch.org/docs/2.10/generated/torch.slice_scatter.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.scatter_reduce](https://pytorch.org/docs/2.10/generated/torch.scatter_reduce.html)|No|-|
|[torch.split](https://pytorch.org/docs/2.10/generated/torch.split.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.squeeze](https://pytorch.org/docs/2.10/generated/torch.squeeze.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.stack](https://pytorch.org/docs/2.10/generated/torch.stack.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.swapaxes](https://pytorch.org/docs/2.10/generated/torch.swapaxes.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.swapdims](https://pytorch.org/docs/2.10/generated/torch.swapdims.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.t](https://pytorch.org/docs/2.10/generated/torch.t.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.take](https://pytorch.org/docs/2.10/generated/torch.take.html)|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|[torch.take_along_dim](https://pytorch.org/docs/2.10/generated/torch.take_along_dim.html)|Yes|Supports fp32|
|[torch.tensor_split](https://pytorch.org/docs/2.10/generated/torch.tensor_split.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.tile](https://pytorch.org/docs/2.10/generated/torch.tile.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>If the length of the input parameter dims is less than the length of input.shape, 1s are automatically prepended to dims to align its length with input.shape. The adjusted dims must satisfy the following restrictions:<br>- When repeating is required on the first axis, repeat operations are allowed on at most 4 dimensions simultaneously (i.e., the number of elements greater than 1 in dims ≤ 4). For example: torch.tile(input, [2, 3, 4, 5, 6]) is not supported, while torch.tile(input, [2, 3, 1, 5, 6]) is supported.<br>- When repeating is not required on the first axis, repeat operations are allowed on at most 3 dimensions simultaneously (i.e., the number of elements greater than 1 in dims ≤ 3). For example: torch.tile(input, [1, 3, 4, 5, 6]) is not supported, while torch.tile(input, [1, 3, 1, 5, 6]) is supported.<br>- If backward computation is performed, the sum of the number of dimensions of the input Tensor and the number of elements greater than 1 in the input parameter dims must not exceed 8|
|[torch.transpose](https://pytorch.org/docs/2.10/generated/torch.transpose.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.unsqueeze](https://pytorch.org/docs/2.10/generated/torch.unsqueeze.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.vsplit](https://pytorch.org/docs/2.10/generated/torch.vsplit.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.vstack](https://pytorch.org/docs/2.10/generated/torch.vstack.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64|
|[torch.where](https://pytorch.org/docs/2.10/generated/torch.where.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>Does not support 8-dimensional shapes|
|[torch.Generator](https://pytorch.org/docs/2.10/generated/torch.Generator.html)|Yes|-|
|[torch.Generator.device](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.device)|Yes|-|
|[torch.Generator.get_state](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.get_state)|Yes|-|
|[torch.Generator.initial_seed](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.initial_seed)|Yes|-|
|[torch.Generator.manual_seed](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.manual_seed)|Yes|-|
|[torch.Generator.seed](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.seed)|Yes|-|
|[torch.Generator.set_state](https://pytorch.org/docs/2.10/generated/torch.Generator.html#torch.Generator.set_state)|Yes|-|
|[torch.seed](https://pytorch.org/docs/2.10/generated/torch.seed.html)|Yes|-|
|[torch.manual_seed](https://pytorch.org/docs/2.10/generated/torch.manual_seed.html)|Yes|-|
|[torch.initial_seed](https://pytorch.org/docs/2.10/generated/torch.initial_seed.html)|Yes|-|
|[torch.get_rng_state](https://pytorch.org/docs/2.10/generated/torch.get_rng_state.html)|Yes|-|
|[torch.set_rng_state](https://pytorch.org/docs/2.10/generated/torch.set_rng_state.html)|Yes|-|
|[torch.bernoulli](https://pytorch.org/docs/2.10/generated/torch.bernoulli.html)|Yes|Supports fp16, fp32, fp64|
|[torch.multinomial](https://pytorch.org/docs/2.10/generated/torch.multinomial.html)|Yes|Supports fp16, fp32|
|[torch.normal](https://pytorch.org/docs/2.10/generated/torch.normal.html)|Yes|Supports fp16, fp32|
|[torch.poisson](https://pytorch.org/docs/2.10/generated/torch.poisson.html)|No|-|
|[torch.rand](https://pytorch.org/docs/2.10/generated/torch.rand.html)|Yes|-|
|[torch.rand_like](https://pytorch.org/docs/2.10/generated/torch.rand_like.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Following PyTorch community standards, processing of bool type data is no longer supported. For existing bool type data, the following alternatives can be used: if full True output is needed, use torch.bernoulli(input, 1). If uniformly distributed bool type output is needed, use torch.bernoulli(input, 0.5)|
|[torch.randint](https://pytorch.org/docs/2.10/generated/torch.randint.html)|Yes|-|
|[torch.randint_like](https://pytorch.org/docs/2.10/generated/torch.randint_like.html)|Yes|Supports fp16, fp32, int64|
|[torch.randn](https://pytorch.org/docs/2.10/generated/torch.randn.html)|Yes|-|
|[torch.randn_like](https://pytorch.org/docs/2.10/generated/torch.randn_like.html)|Yes|Supports fp32|
|[torch.randperm](https://pytorch.org/docs/2.10/generated/torch.randperm.html)|Yes|-|
|[torch.save](https://pytorch.org/docs/2.10/generated/torch.save.html)|Yes|-|
|[torch.load](https://pytorch.org/docs/2.10/generated/torch.load.html)|Yes|-|
|[torch.get_num_threads](https://pytorch.org/docs/2.10/generated/torch.get_num_threads.html)|Yes|-|
|[torch.set_num_threads](https://pytorch.org/docs/2.10/generated/torch.set_num_threads.html)|Yes|-|
|[torch.get_num_interop_threads](https://pytorch.org/docs/2.10/generated/torch.get_num_interop_threads.html)|Yes|-|
|[torch.set_num_interop_threads](https://pytorch.org/docs/2.10/generated/torch.set_num_interop_threads.html)|Yes|-|
|[torch.no_grad](https://pytorch.org/docs/2.10/generated/torch.no_grad.html)|Yes|-|
|[torch.enable_grad](https://pytorch.org/docs/2.10/generated/torch.enable_grad.html)|Yes|-|
|[torch.autograd.grad_mode.set_grad_enabled](https://pytorch.org/docs/2.10/generated/torch.autograd.grad_mode.set_grad_enabled.html)|Yes|-|
|[torch.is_grad_enabled](https://pytorch.org/docs/2.10/generated/torch.is_grad_enabled.html)|Yes|-|
|[torch.autograd.grad_mode.inference_mode](https://pytorch.org/docs/2.10/generated/torch.autograd.grad_mode.inference_mode.html)|Yes|-|
|[torch.is_inference_mode_enabled](https://pytorch.org/docs/2.10/generated/torch.is_inference_mode_enabled.html)|Yes|-|
|[torch.abs](https://pytorch.org/docs/2.10/generated/torch.abs.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.absolute](https://pytorch.org/docs/2.10/generated/torch.absolute.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.acos](https://pytorch.org/docs/2.10/generated/torch.acos.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.arccos](https://pytorch.org/docs/2.10/generated/torch.arccos.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.acosh](https://pytorch.org/docs/2.10/generated/torch.acosh.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>May fall back to CPU execution|
|[torch.arccosh](https://pytorch.org/docs/2.10/generated/torch.arccosh.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.add](https://pytorch.org/docs/2.10/generated/torch.add.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.addcdiv](https://pytorch.org/docs/2.10/generated/torch.addcdiv.html)|Yes|Supports fp16, fp32, int64<br>Simultaneous broadcasting of three tensors is not supported for int64 type|
|[torch.addcmul](https://pytorch.org/docs/2.10/generated/torch.addcmul.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64<br>Simultaneous broadcasting of three tensors is not supported for fp64, uint8, int8, int64 types|
|[torch.angle](https://pytorch.org/docs/2.10/generated/torch.angle.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|[torch.asin](https://pytorch.org/docs/2.10/generated/torch.asin.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.arcsin](https://pytorch.org/docs/2.10/generated/torch.arcsin.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.asinh](https://pytorch.org/docs/2.10/generated/torch.asinh.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.arcsinh](https://pytorch.org/docs/2.10/generated/torch.arcsinh.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.atan](https://pytorch.org/docs/2.10/generated/torch.atan.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.arctan](https://pytorch.org/docs/2.10/generated/torch.arctan.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.atanh](https://pytorch.org/docs/2.10/generated/torch.atanh.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.arctanh](https://pytorch.org/docs/2.10/generated/torch.arctanh.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.atan2](https://pytorch.org/docs/2.10/generated/torch.atan2.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.arctan2](https://pytorch.org/docs/2.10/generated/torch.arctan2.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.bitwise_not](https://pytorch.org/docs/2.10/generated/torch.bitwise_not.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[torch.bitwise_and](https://pytorch.org/docs/2.10/generated/torch.bitwise_and.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[torch.bitwise_or](https://pytorch.org/docs/2.10/generated/torch.bitwise_or.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[torch.bitwise_xor](https://pytorch.org/docs/2.10/generated/torch.bitwise_xor.html)|Yes|Supports uint8, int8, int16, int32, int64, bool|
|[torch.bitwise_left_shift](https://pytorch.org/docs/2.10/generated/torch.bitwise_left_shift.html)|Yes|Supports uint8, int8, int16, int32, int64|
|[torch.ceil](https://pytorch.org/docs/2.10/generated/torch.ceil.html)|Yes|Supports fp16, fp32|
|[torch.clamp](https://pytorch.org/docs/2.10/generated/torch.clamp.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.clip](https://pytorch.org/docs/2.10/generated/torch.clip.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.copysign](https://pytorch.org/docs/2.10/generated/torch.copysign.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU execution|
|[torch.cos](https://pytorch.org/docs/2.10/generated/torch.cos.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.cosh](https://pytorch.org/docs/2.10/generated/torch.cosh.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.deg2rad](https://pytorch.org/docs/2.10/generated/torch.deg2rad.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.div](https://pytorch.org/docs/2.10/generated/torch.div.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.divide](https://pytorch.org/docs/2.10/generated/torch.divide.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.erf](https://pytorch.org/docs/2.10/generated/torch.erf.html)|Yes|Supports bf16, fp16, fp32, fp64, int64, bool|
|[torch.erfc](https://pytorch.org/docs/2.10/generated/torch.erfc.html)|Yes|Supports fp16, fp32, int64, bool|
|[torch.erfinv](https://pytorch.org/docs/2.10/generated/torch.erfinv.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.exp](https://pytorch.org/docs/2.10/generated/torch.exp.html)|Yes|Supports bf16, fp16, fp32, fp64, int64, bool, complex64, complex128|
|[torch.exp2](https://pytorch.org/docs/2.10/generated/torch.exp2.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.expm1](https://pytorch.org/docs/2.10/generated/torch.expm1.html)|Yes|Supports bf16, fp16, fp32, fp64, int64, bool|
|[torch.fix](https://pytorch.org/docs/2.10/generated/torch.fix.html)|Yes|Supports bf16, fp16, fp32|
|[torch.float_power](https://pytorch.org/docs/2.10/generated/torch.float_power.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex128|
|[torch.floor](https://pytorch.org/docs/2.10/generated/torch.floor.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.floor_divide](https://pytorch.org/docs/2.10/generated/torch.floor_divide.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.fmod](https://pytorch.org/docs/2.10/generated/torch.fmod.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64|
|[torch.gradient](https://pytorch.org/docs/2.10/generated/torch.gradient.html)|Yes|Supports bf16, fp16, fp32, int8, int16, int32, int64|
|[torch.ldexp](https://pytorch.org/docs/2.10/generated/torch.ldexp.html)|Yes|Supports fp16, fp64, complex64|
|[torch.lerp](https://pytorch.org/docs/2.10/generated/torch.lerp.html)|Yes|Supports fp16, fp32|
|[torch.log](https://pytorch.org/docs/2.10/generated/torch.log.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.log10](https://pytorch.org/docs/2.10/generated/torch.log10.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>When input is uint8, int8, int16, int32, int64, bool, output out must be fp32<br>For other supported data types, output out remains consistent with input|
|[torch.log1p](https://pytorch.org/docs/2.10/generated/torch.log1p.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.log2](https://pytorch.org/docs/2.10/generated/torch.log2.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.logaddexp](https://pytorch.org/docs/2.10/generated/torch.logaddexp.html)|Yes|Does not support double data type|
|[torch.logaddexp2](https://pytorch.org/docs/2.10/generated/torch.logaddexp2.html)|Yes|Does not support double data type|
|[torch.logical_and](https://pytorch.org/docs/2.10/generated/torch.logical_and.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.logical_not](https://pytorch.org/docs/2.10/generated/torch.logical_not.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.logical_or](https://pytorch.org/docs/2.10/generated/torch.logical_or.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.logical_xor](https://pytorch.org/docs/2.10/generated/torch.logical_xor.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.logit](https://pytorch.org/docs/2.10/generated/torch.logit.html)|Yes|Supports bf16, fp16, fp32<br>When eps is greater than 1, output is nan; when eps is 1, output is inf|
|[torch.mul](https://pytorch.org/docs/2.10/generated/torch.mul.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.multiply](https://pytorch.org/docs/2.10/generated/torch.multiply.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nan_to_num](https://pytorch.org/docs/2.10/generated/torch.nan_to_num.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.neg](https://pytorch.org/docs/2.10/generated/torch.neg.html)|Yes|Supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128|
|[torch.negative](https://pytorch.org/docs/2.10/generated/torch.negative.html)|Yes|Supports bf16, fp16, fp32, int8, int32, int64, complex64, complex128|
|[torch.positive](https://pytorch.org/docs/2.10/generated/torch.positive.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128|
|[torch.pow](https://pytorch.org/docs/2.10/generated/torch.pow.html)|Yes|Supports bf16, fp16, fp32, fp64, int16, int64|
|[torch.rad2deg](https://pytorch.org/docs/2.10/generated/torch.rad2deg.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.real](https://pytorch.org/docs/2.10/generated/torch.real.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[torch.reciprocal](https://pytorch.org/docs/2.10/generated/torch.reciprocal.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.remainder](https://pytorch.org/docs/2.10/generated/torch.remainder.html)|Yes|Supports fp16, fp32, int16, int32, int64|
|[torch.round](https://pytorch.org/docs/2.10/generated/torch.round.html)|Yes|Supports bf16, fp16, fp32, fp64, int32, int64|
|[torch.rsqrt](https://pytorch.org/docs/2.10/generated/torch.rsqrt.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.sigmoid](https://pytorch.org/docs/2.10/generated/torch.sigmoid.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.sign](https://pytorch.org/docs/2.10/generated/torch.sign.html)|Yes|Supports bf16, fp16, fp32, int32, int64, bool|
|[torch.sgn](https://pytorch.org/docs/2.10/generated/torch.sgn.html)|Yes|Supports bf16, fp16, fp32, int32, int64, bool, complex64, complex128|
|[torch.sin](https://pytorch.org/docs/2.10/generated/torch.sin.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.sinh](https://pytorch.org/docs/2.10/generated/torch.sinh.html)|Yes|Supports fp16, fp32, fp64|
|[torch.softmax](https://pytorch.org/docs/2.10/generated/torch.softmax.html)|Yes|Supports fp32<br>Supports Named Tensor|
|[torch.sqrt](https://pytorch.org/docs/2.10/generated/torch.sqrt.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.square](https://pytorch.org/docs/2.10/generated/torch.square.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.sub](https://pytorch.org/docs/2.10/generated/torch.sub.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.tan](https://pytorch.org/docs/2.10/generated/torch.tan.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Value range [-65504, 65504]|
|[torch.tanh](https://pytorch.org/docs/2.10/generated/torch.tanh.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.true_divide](https://pytorch.org/docs/2.10/generated/torch.true_divide.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.trunc](https://pytorch.org/docs/2.10/generated/torch.trunc.html)|Yes|Supports fp16, fp32<br>May fall back to CPU execution|
|[torch.xlogy](https://pytorch.org/docs/2.10/generated/torch.xlogy.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.argmax](https://pytorch.org/docs/2.10/generated/torch.argmax.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.argmin](https://pytorch.org/docs/2.10/generated/torch.argmin.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.amax](https://pytorch.org/docs/2.10/generated/torch.amax.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.amin](https://pytorch.org/docs/2.10/generated/torch.amin.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.aminmax](https://pytorch.org/docs/2.10/generated/torch.aminmax.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.all](https://pytorch.org/docs/2.10/generated/torch.all.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.any](https://pytorch.org/docs/2.10/generated/torch.any.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.max](https://pytorch.org/docs/2.10/generated/torch.max.html)|Yes|Supports bf16, fp16, fp32, int64, bool|
|[torch.min](https://pytorch.org/docs/2.10/generated/torch.min.html)|Yes|Supports bf16, fp16, fp32, int64, bool|
|[torch.dist](https://pytorch.org/docs/2.10/generated/torch.dist.html)|Yes|Supports bf16, fp16, fp32|
|[torch.logsumexp](https://pytorch.org/docs/2.10/generated/torch.logsumexp.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.mean](https://pytorch.org/docs/2.10/generated/torch.mean.html)|Yes|Supports bf16, fp16, fp32, complex64, complex128|
|[torch.nanmean](https://pytorch.org/docs/2.10/generated/torch.nanmean.html)|Yes|Supports bf16, fp16, fp32|
|[torch.median](https://pytorch.org/docs/2.10/generated/torch.median.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.norm](https://pytorch.org/docs/2.10/generated/torch.norm.html)|Yes|Supports bf16, fp16, fp32, fp64<br>When parameter p is negative, the computation result may have precision errors<br>When parameter dim specifies an axis with a shape dimension value of 1 in the input tensor, the computation result may have precision errors|
|[torch.nansum](https://pytorch.org/docs/2.10/generated/torch.nansum.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.prod](https://pytorch.org/docs/2.10/generated/torch.prod.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nanquantile](https://pytorch.org/docs/2.10/generated/torch.nanquantile.html)|No|-|
|[torch.std](https://pytorch.org/docs/2.10/generated/torch.std.html)|Yes|May fall back to CPU execution|
|[torch.std_mean](https://pytorch.org/docs/2.10/generated/torch.std_mean.html)|No|-|
|[torch.sum](https://pytorch.org/docs/2.10/generated/torch.sum.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>Does not support the dtype parameter|
|[torch.unique](https://pytorch.org/docs/2.10/generated/torch.unique.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>fp16 is not supported in scenarios with dim<br>When the input contains 0, the output may include both positive 0 and negative 0, rather than outputting only one 0|
|[torch.unique_consecutive](https://pytorch.org/docs/2.10/generated/torch.unique_consecutive.html)|No|-|
|[torch.var](https://pytorch.org/docs/2.10/generated/torch.var.html)|Yes|Supports fp16, fp32|
|[torch.var_mean](https://pytorch.org/docs/2.10/generated/torch.var_mean.html)|No|-|
|[torch.count_nonzero](https://pytorch.org/docs/2.10/generated/torch.count_nonzero.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.allclose](https://pytorch.org/docs/2.10/generated/torch.allclose.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.argsort](https://pytorch.org/docs/2.10/generated/torch.argsort.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.eq](https://pytorch.org/docs/2.10/generated/torch.eq.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.equal](https://pytorch.org/docs/2.10/generated/torch.equal.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.ge](https://pytorch.org/docs/2.10/generated/torch.ge.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.greater_equal](https://pytorch.org/docs/2.10/generated/torch.greater_equal.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.gt](https://pytorch.org/docs/2.10/generated/torch.gt.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.greater](https://pytorch.org/docs/2.10/generated/torch.greater.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.isclose](https://pytorch.org/docs/2.10/generated/torch.isclose.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.isfinite](https://pytorch.org/docs/2.10/generated/torch.isfinite.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.isin](https://pytorch.org/docs/2.10/generated/torch.isin.html)|Yes|Constraints for the two-tensor input scenario are as follows:<br>- Supports fp16, fp32, uint8, int8, int16, int32, int64<br>- The first input tensor cannot have more than 7 dimensions, and the second input tensor cannot have more than 8 dimensions<br>Constraints for the single-tensor input scenario are as follows:<br>- Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>- The input tensor cannot have more than 8 dimensions|
|[torch.isinf](https://pytorch.org/docs/2.10/generated/torch.isinf.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.isposinf](https://pytorch.org/docs/2.10/generated/torch.isposinf.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.isneginf](https://pytorch.org/docs/2.10/generated/torch.isneginf.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.isnan](https://pytorch.org/docs/2.10/generated/torch.isnan.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.kthvalue](https://pytorch.org/docs/2.10/generated/torch.kthvalue.html)|Yes|Supports fp16, fp32, int32|
|[torch.le](https://pytorch.org/docs/2.10/generated/torch.le.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.less_equal](https://pytorch.org/docs/2.10/generated/torch.less_equal.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.lt](https://pytorch.org/docs/2.10/generated/torch.lt.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.less](https://pytorch.org/docs/2.10/generated/torch.less.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.maximum](https://pytorch.org/docs/2.10/generated/torch.maximum.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.minimum](https://pytorch.org/docs/2.10/generated/torch.minimum.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.ne](https://pytorch.org/docs/2.10/generated/torch.ne.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.not_equal](https://pytorch.org/docs/2.10/generated/torch.not_equal.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.sort](https://pytorch.org/docs/2.10/generated/torch.sort.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.topk](https://pytorch.org/docs/2.10/generated/torch.topk.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64<br>Does not support the sorted=False scenario|
|[torch.msort](https://pytorch.org/docs/2.10/generated/torch.msort.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.stft](https://pytorch.org/docs/2.10/generated/torch.stft.html)|Yes|Supports fp32, fp64, complex64, complex128<br>If the operator times out, use the official interface set_op_execute_time_out to set a higher timeout threshold to extend the judgment time|
|[torch.hann_window](https://pytorch.org/docs/2.10/generated/torch.hann_window.html)|Yes|Supports bf16, fp16, fp32<br>When the data type is fp32 and the parameter window_length is greater than 10000, the computation result may have errors|
|[torch.atleast_1d](https://pytorch.org/docs/2.10/generated/torch.atleast_1d.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.atleast_2d](https://pytorch.org/docs/2.10/generated/torch.atleast_2d.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.atleast_3d](https://pytorch.org/docs/2.10/generated/torch.atleast_3d.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.bincount](https://pytorch.org/docs/2.10/generated/torch.bincount.html)|Yes|Supports uint8, int8, int16, int32, int64<br>The weights dimension must be consistent with the input dimension|
|[torch.block_diag](https://pytorch.org/docs/2.10/generated/torch.block_diag.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.broadcast_tensors](https://pytorch.org/docs/2.10/generated/torch.broadcast_tensors.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.broadcast_to](https://pytorch.org/docs/2.10/generated/torch.broadcast_to.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.broadcast_shapes](https://pytorch.org/docs/2.10/generated/torch.broadcast_shapes.html)|Yes|-|
|[torch.cdist](https://pytorch.org/docs/2.10/generated/torch.cdist.html)|Yes|Supports bf16, fp16, fp32<br>When p=2.0, "compute_mode" only supports the "donot_use_mm_for_euclid_dist" mode; passing other values will automatically be changed to this mode|
|[torch.clone](https://pytorch.org/docs/2.10/generated/torch.clone.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.combinations](https://pytorch.org/docs/2.10/generated/torch.combinations.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.cov](https://pytorch.org/docs/2.10/generated/torch.cov.html)|Yes|Supports fp32|
|[torch.cross](https://pytorch.org/docs/2.10/generated/torch.cross.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, complex64, complex128<br>The shapes of the two inputs must be consistent|
|[torch.cummax](https://pytorch.org/docs/2.10/generated/torch.cummax.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.cummin](https://pytorch.org/docs/2.10/generated/torch.cummin.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>When the input is int32, the value range is within [-16777216, 16777216]|
|[torch.cumprod](https://pytorch.org/docs/2.10/generated/torch.cumprod.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.cumsum](https://pytorch.org/docs/2.10/generated/torch.cumsum.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Supports Named Tensor|
|[torch.diag](https://pytorch.org/docs/2.10/generated/torch.diag.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64|
|[torch.diag_embed](https://pytorch.org/docs/2.10/generated/torch.diag_embed.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.diagonal](https://pytorch.org/docs/2.10/generated/torch.diagonal.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.diff](https://pytorch.org/docs/2.10/generated/torch.diff.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.einsum](https://pytorch.org/docs/2.10/generated/torch.einsum.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.flatten](https://pytorch.org/docs/2.10/generated/torch.flatten.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.flip](https://pytorch.org/docs/2.10/generated/torch.flip.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.fliplr](https://pytorch.org/docs/2.10/generated/torch.fliplr.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.flipud](https://pytorch.org/docs/2.10/generated/torch.flipud.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.kron](https://pytorch.org/docs/2.10/generated/torch.kron.html)|Yes|Does not support inputs with 5 or more dimensions|
|[torch.histc](https://pytorch.org/docs/2.10/generated/torch.histc.html)|Yes|Supports fp16, fp32<br>When the input tensor value falls on the boundary of counting intervals, there may be errors in whether it is counted in the left interval or the right interval|
|[torch.meshgrid](https://pytorch.org/docs/2.10/generated/torch.meshgrid.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.ravel](https://pytorch.org/docs/2.10/generated/torch.ravel.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.repeat_interleave](https://pytorch.org/docs/2.10/generated/torch.repeat_interleave.html)|Yes|Supports fp16, fp32, int16, int32, int64, bool<br>The input tensor is repeated to produce the output, and the number of elements in the output must be less than $2^{22}$|
|[torch.roll](https://pytorch.org/docs/2.10/generated/torch.roll.html)|Yes|Supports fp16, fp32, int32, int64, bool|
|[torch.searchsorted](https://pytorch.org/docs/2.10/generated/torch.searchsorted.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.tensordot](https://pytorch.org/docs/2.10/generated/torch.tensordot.html)|Yes|Supports fp16, fp32|
|[torch.tril](https://pytorch.org/docs/2.10/generated/torch.tril.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.tril_indices](https://pytorch.org/docs/2.10/generated/torch.tril_indices.html)|Yes|-|
|[torch.triu](https://pytorch.org/docs/2.10/generated/torch.triu.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.triu_indices](https://pytorch.org/docs/2.10/generated/torch.triu_indices.html)|Yes|-|
|[torch.unflatten](https://pytorch.org/docs/2.10/generated/torch.unflatten.html)|Yes|-|
|[torch.view_as_real](https://pytorch.org/docs/2.10/generated/torch.view_as_real.html)|Yes|Supports complex64, complex128|
|[torch.view_as_complex](https://pytorch.org/docs/2.10/generated/torch.view_as_complex.html)|Yes|Supports fp32, fp64|
|[torch.resolve_conj](https://pytorch.org/docs/2.10/generated/torch.resolve_conj.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.resolve_neg](https://pytorch.org/docs/2.10/generated/torch.resolve_neg.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.addbmm](https://pytorch.org/docs/2.10/generated/torch.addbmm.html)|Yes|Supports fp16, fp32|
|[torch.addmm](https://pytorch.org/docs/2.10/generated/torch.addmm.html)|Yes|Supports fp16, fp32|
|[torch.addmv](https://pytorch.org/docs/2.10/generated/torch.addmv.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.addr](https://pytorch.org/docs/2.10/generated/torch.addr.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.baddbmm](https://pytorch.org/docs/2.10/generated/torch.baddbmm.html)|Yes|Supports bf16, fp16, fp32|
|[torch.bmm](https://pytorch.org/docs/2.10/generated/torch.bmm.html)|Yes|Supports fp16, fp32|
|[torch.dot](https://pytorch.org/docs/2.10/generated/torch.dot.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int32|
|[torch.slogdet](https://pytorch.org/docs/2.10/generated/torch.slogdet.html)|Yes|Supports fp32, complex64, complex128<br>May fall back to CPU execution|
|[torch.matmul](https://pytorch.org/docs/2.10/generated/torch.matmul.html)|Yes|Supports fp16, fp32<br>Supports Named Tensor<br>Input supports up to 6 dimensions|
|[torch.mm](https://pytorch.org/docs/2.10/generated/torch.mm.html)|Yes|Supports fp16, fp32|
|[torch.outer](https://pytorch.org/docs/2.10/generated/torch.outer.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.qr](https://pytorch.org/docs/2.10/generated/torch.qr.html)|Yes|-|
|[torch.trapezoid](https://pytorch.org/docs/2.10/generated/torch.trapezoid.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.cumulative_trapezoid](https://pytorch.org/docs/2.10/generated/torch.cumulative_trapezoid.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.vdot](https://pytorch.org/docs/2.10/generated/torch.vdot.html)|Yes|Supports fp16, fp32|
|[torch.compiled_with_cxx11_abi](https://pytorch.org/docs/2.10/generated/torch.compiled_with_cxx11_abi.html)|Yes|-|
|[torch.result_type](https://pytorch.org/docs/2.10/generated/torch.result_type.html)|Yes|Supports fp32|
|[torch.can_cast](https://pytorch.org/docs/2.10/generated/torch.can_cast.html)|Yes|-|
|[torch.promote_types](https://pytorch.org/docs/2.10/generated/torch.promote_types.html)|Yes|-|
|[torch.use_deterministic_algorithms](https://pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html)|Yes|When both HCCL_DETERMINISTIC and torch.use_deterministic_algorithms are set, if HCCL_DETERMINISTIC enables determinism, the HCCL interface uses determinism; otherwise, HCCL determinism is controlled by the torch.use_deterministic_algorithms interface|
|[torch.are_deterministic_algorithms_enabled](https://pytorch.org/docs/2.10/generated/torch.are_deterministic_algorithms_enabled.html)|Yes|-|
|[torch.is_deterministic_algorithms_warn_only_enabled](https://pytorch.org/docs/2.10/generated/torch.is_deterministic_algorithms_warn_only_enabled.html)|No|-|
|[torch.set_deterministic_debug_mode](https://pytorch.org/docs/2.10/generated/torch.set_deterministic_debug_mode.html)|Yes|-|
|[torch.get_deterministic_debug_mode](https://pytorch.org/docs/2.10/generated/torch.get_deterministic_debug_mode.html)|Yes|-|
|[torch.set_float32_matmul_precision](https://pytorch.org/docs/2.10/generated/torch.set_float32_matmul_precision.html)|Yes|-|
|[torch.get_float32_matmul_precision](https://pytorch.org/docs/2.10/generated/torch.get_float32_matmul_precision.html)|Yes|-|
|[torch.set_warn_always](https://pytorch.org/docs/2.10/generated/torch.set_warn_always.html)|Yes|-|
|[torch.is_warn_always_enabled](https://pytorch.org/docs/2.10/generated/torch.is_warn_always_enabled.html)|Yes|-|
|[torch.vmap](https://pytorch.org/docs/2.10/generated/torch.vmap.html)|Yes|-|
|[torch._assert](https://pytorch.org/docs/2.10/generated/torch._assert.html)|Yes|-|
|[torch.sym_float](https://pytorch.org/docs/2.10/generated/torch.sym_float.html)|Yes|Supports fp32|
|[torch.sym_int](https://pytorch.org/docs/2.10/generated/torch.sym_int.html)|Yes|Supports fp32|
|[torch.sym_max](https://pytorch.org/docs/2.10/generated/torch.sym_max.html)|Yes|Supports fp32|
|[torch.sym_min](https://pytorch.org/docs/2.10/generated/torch.sym_min.html)|Yes|Supports fp32|
|[torch.sym_not](https://pytorch.org/docs/2.10/generated/torch.sym_not.html)|Yes|Supports fp32|
|[torch.compile](https://pytorch.org/docs/2.10/generated/torch.compile.html)|Yes|The backend can support npugraphs, and the overall functionality is consistent with backend="cudagraphs"|
|[torch.bucketize](https://pytorch.org/docs/2.10/generated/torch.bucketize.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.cartesian_prod](https://pytorch.org/docs/2.10/generated/torch.cartesian_prod.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.mv](https://pytorch.org/docs/2.10/generated/torch.mv.html)|Yes|Supports fp16, fp32|
|[torch.quasirandom.SobolEngine.draw](https://pytorch.org/docs/2.10/generated/torch.quasirandom.SobolEngine.html#torch.quasirandom.SobolEngine.draw)|Yes|Supports fp32, fp64|
|[torch._foreach_sqrt](https://pytorch.org/docs/2.10/generated/torch._foreach_sqrt.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch._foreach_asin](https://pytorch.org/docs/2.10/generated/torch._foreach_asin.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.\_foreach\_neg\_](https://pytorch.org/docs/2.10/torch.html#torch.\_foreach\_neg\_)|Yes|Supports bf16, fp16, fp32, int8, int32, int64|
