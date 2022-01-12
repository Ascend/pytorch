#  PyTorch 1.8.1 API Support

## Tensors

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [is_tensor](https://pytorch.org/docs/1.8.1/generated/torch.is_tensor.html) | Unsupported           |
| 2    | [is_storage](https://pytorch.org/docs/1.8.1/generated/torch.is_storage.html) | Unsupported           |
| 3    | [is_complex](https://pytorch.org/docs/1.8.1/generated/torch.is_complex.html) | Unsupported           |
| 4    | [is_floating_point](https://pytorch.org/docs/1.8.1/generated/torch.is_floating_point.html) | Unsupported           |
| 5    | [is_nonzero](https://pytorch.org/docs/1.8.1/generated/torch.is_nonzero.html) | Unsupported           |
| 6    | [set_default_dtype](https://pytorch.org/docs/1.8.1/generated/torch.set_default_dtype.html) | Unsupported           |
| 7    | [get_default_dtype](https://pytorch.org/docs/1.8.1/generated/torch.get_default_dtype.html) | Unsupported           |
| 8    | [set_default_tensor_type](https://pytorch.org/docs/1.8.1/generated/torch.set_default_tensor_type.html) | Unsupported           |
| 9    | [numel](https://pytorch.org/docs/1.8.1/generated/torch.numel.html) | Unsupported           |
| 10   | [set_printoptions](https://pytorch.org/docs/1.8.1/generated/torch.set_printoptions.html) | Unsupported           |
| 11   | [set_flush_denormal](https://pytorch.org/docs/1.8.1/generated/torch.set_flush_denormal.html) | Unsupported           |

### Creation Ops

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [tensor](https://pytorch.org/docs/1.8.1/generated/torch.tensor.html) | Unsupported           |
| 2    | [sparse_coo_tensor](https://pytorch.org/docs/1.8.1/generated/torch.sparse_coo_tensor.html) | Unsupported           |
| 3    | [as_tensor](https://pytorch.org/docs/1.8.1/generated/torch.as_tensor.html) | Unsupported           |
| 4    | [as_strided](https://pytorch.org/docs/1.8.1/generated/torch.as_strided.html) | Unsupported           |
| 5    | [from_numpy](https://pytorch.org/docs/1.8.1/generated/torch.from_numpy.html) | Unsupported           |
| 6    | [zeros](https://pytorch.org/docs/1.8.1/generated/torch.zeros.html) | Unsupported           |
| 7    | [zeros_like](https://pytorch.org/docs/1.8.1/generated/torch.zeros_like.html) | Unsupported           |
| 8    | [ones](https://pytorch.org/docs/1.8.1/generated/torch.ones.html) | Unsupported           |
| 9    | [ones_like](https://pytorch.org/docs/1.8.1/generated/torch.ones_like.html) | Unsupported           |
| 10   | [arange](https://pytorch.org/docs/1.8.1/generated/torch.arange.html) | Unsupported           |
| 11   | [range](https://pytorch.org/docs/1.8.1/generated/torch.range.html) | Unsupported           |
| 12   | [linspace](https://pytorch.org/docs/1.8.1/generated/torch.linspace.html) | Unsupported           |
| 13   | [logspace](https://pytorch.org/docs/1.8.1/generated/torch.logspace.html) | Unsupported           |
| 14   | [eye](https://pytorch.org/docs/1.8.1/generated/torch.eye.html) | Unsupported           |
| 15   | [empty](https://pytorch.org/docs/1.8.1/generated/torch.empty.html) | Unsupported           |
| 16   | [empty_like](https://pytorch.org/docs/1.8.1/generated/torch.empty_like.html) | Unsupported           |
| 17   | [empty_strided](https://pytorch.org/docs/1.8.1/generated/torch.empty_strided.html) | Unsupported           |
| 18   | [full](https://pytorch.org/docs/1.8.1/generated/torch.full.html) | Unsupported           |
| 19   | [full_like](https://pytorch.org/docs/1.8.1/generated/torch.full_like.html) | Unsupported           |
| 20   | [quantize_per_tensor](https://pytorch.org/docs/1.8.1/generated/torch.quantize_per_tensor.html) | Unsupported           |
| 21   | [quantize_per_channel](https://pytorch.org/docs/1.8.1/generated/torch.quantize_per_channel.html) | Unsupported           |
| 22   | [dequantize](https://pytorch.org/docs/1.8.1/generated/torch.dequantize.html) | Unsupported           |
| 23   | [complex](https://pytorch.org/docs/1.8.1/generated/torch.complex.html) | Unsupported           |
| 24   | [polar](https://pytorch.org/docs/1.8.1/generated/torch.polar.html) | Unsupported           |
| 25   | [heaviside](https://pytorch.org/docs/1.8.1/generated/torch.heaviside.html) | Unsupported           |

### Indexing, Slicing, Joining, Mutating Ops

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [cat](https://pytorch.org/docs/1.8.1/generated/torch.cat.html) | Unsupported           |
| 2    | [chunk](https://pytorch.org/docs/1.8.1/generated/torch.chunk.html) | Unsupported           |
| 3    | [column_stack](https://pytorch.org/docs/1.8.1/generated/torch.column_stack.html) | Unsupported           |
| 4    | [dstack](https://pytorch.org/docs/1.8.1/generated/torch.dstack.html) | Unsupported           |
| 5    | [gather](https://pytorch.org/docs/1.8.1/generated/torch.gather.html) | Unsupported           |
| 6    | [hstack](https://pytorch.org/docs/1.8.1/generated/torch.hstack.html) | Unsupported           |
| 7    | [index_select](https://pytorch.org/docs/1.8.1/generated/torch.index_select.html) | Unsupported           |
| 8    | [masked_select](https://pytorch.org/docs/1.8.1/generated/torch.masked_select.html) | Unsupported           |
| 9    | [movedim](https://pytorch.org/docs/1.8.1/generated/torch.movedim.html) | Unsupported           |
| 10   | [moveaxis](https://pytorch.org/docs/1.8.1/generated/torch.moveaxis.html) | Unsupported           |
| 11   | [narrow](https://pytorch.org/docs/1.8.1/generated/torch.narrow.html) | Unsupported           |
| 12   | [nonzero](https://pytorch.org/docs/1.8.1/generated/torch.nonzero.html) | Unsupported           |
| 13   | [reshape](https://pytorch.org/docs/1.8.1/generated/torch.reshape.html) | Unsupported           |
| 14   | [row_stack](https://pytorch.org/docs/1.8.1/generated/torch.row_stack.html) | Unsupported           |
| 15   | [scatter](https://pytorch.org/docs/1.8.1/generated/torch.scatter.html) | Unsupported           |
| 16   | [scatter_add](https://pytorch.org/docs/1.8.1/generated/torch.scatter_add.html) | Unsupported           |
| 17   | [split](https://pytorch.org/docs/1.8.1/generated/torch.split.html) | Unsupported           |
| 18   | [squeeze](https://pytorch.org/docs/1.8.1/generated/torch.squeeze.html) | Unsupported           |
| 19   | [stack](https://pytorch.org/docs/1.8.1/generated/torch.stack.html) | Unsupported           |
| 20   | [swapaxes](https://pytorch.org/docs/1.8.1/generated/torch.swapaxes.html) | Unsupported           |
| 21   | [swapdims](https://pytorch.org/docs/1.8.1/generated/torch.swapdims.html) | Unsupported           |
| 22   | [t](https://pytorch.org/docs/1.8.1/generated/torch.t.html)   | Unsupported           |
| 23   | [take](https://pytorch.org/docs/1.8.1/generated/torch.take.html) | Unsupported           |
| 24   | [tensor_split](https://pytorch.org/docs/1.8.1/generated/torch.tensor_split.html) | Unsupported           |
| 25   | [tile](https://pytorch.org/docs/1.8.1/generated/torch.tile.html) | Unsupported           |
| 26   | [transpose](https://pytorch.org/docs/1.8.1/generated/torch.transpose.html) | Unsupported           |
| 27   | [unbind](https://pytorch.org/docs/1.8.1/generated/torch.unbind.html) | Unsupported           |
| 28   | [unsqueeze](https://pytorch.org/docs/1.8.1/generated/torch.unsqueeze.html) | Unsupported           |
| 29   | [vstack](https://pytorch.org/docs/1.8.1/generated/torch.vstack.html) | Unsupported           |
| 30   | [where](https://pytorch.org/docs/1.8.1/generated/torch.where.html) | Unsupported           |

## Generators

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [Generator](https://pytorch.org/docs/1.8.1/generated/torch.Generator.html) | Unsupported           |

## Random Sampling

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [seed](https://pytorch.org/docs/1.8.1/generated/torch.seed.html) | Unsupported           |
| 2    | [manual_seed](https://pytorch.org/docs/1.8.1/generated/torch.manual_seed.html) | Unsupported           |
| 3    | [initial_seed](https://pytorch.org/docs/1.8.1/generated/torch.initial_seed.html) | Unsupported           |
| 4    | [get_rng_state](https://pytorch.org/docs/1.8.1/generated/torch.get_rng_state.html) | Unsupported           |
| 5    | [set_rng_state](https://pytorch.org/docs/1.8.1/generated/torch.set_rng_state.html) | Unsupported           |
| 6    | [bernoulli](https://pytorch.org/docs/1.8.1/generated/torch.bernoulli.html) | Unsupported           |
| 7    | [multinomial](https://pytorch.org/docs/1.8.1/generated/torch.multinomial.html) | Unsupported           |
| 8    | [normal](https://pytorch.org/docs/1.8.1/generated/torch.normal.html) | Unsupported           |
| 9    | [poisson](https://pytorch.org/docs/1.8.1/generated/torch.poisson.html) | Unsupported           |
| 10   | [rand](https://pytorch.org/docs/1.8.1/generated/torch.rand.html) | Unsupported           |
| 11   | [rand_like](https://pytorch.org/docs/1.8.1/generated/torch.rand_like.html) | Unsupported           |
| 12   | [randint](https://pytorch.org/docs/1.8.1/generated/torch.randint.html) | Unsupported           |
| 13   | [randint_like](https://pytorch.org/docs/1.8.1/generated/torch.randint_like.html) | Unsupported           |
| 14   | [randn](https://pytorch.org/docs/1.8.1/generated/torch.randn.html) | Unsupported           |
| 15   | [randn_like](https://pytorch.org/docs/1.8.1/generated/torch.randn_like.html) | Unsupported           |
| 16   | [randperm](https://pytorch.org/docs/1.8.1/generated/torch.randperm.html) | Unsupported           |

### In-place Random Sampling

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [torch.Tensor.bernoulli_()](https://pytorch.org/docs/1.8.1/tensors.html) | Unsupported           |
| 2    | [torch.Tensor.cauchy_()](https://pytorch.org/docs/1.8.1/tensors.html) | Unsupported           |
| 3    | [torch.Tensor.exponential_()](https://pytorch.org/docs/1.8.1/tensors.html) | Unsupported           |
| 4    | [torch.Tensor.geometric_()](https://pytorch.org/docs/1.8.1/tensors.html) | Unsupported           |
| 5    | [torch.Tensor.log_normal_()](https://pytorch.org/docs/1.8.1/tensors.html) | Unsupported           |
| 6    | [torch.Tensor.normal_()](https://pytorch.org/docs/1.8.1/tensors.html) | Unsupported           |
| 7    | [torch.Tensor.random_()](https://pytorch.org/docs/1.8.1/tensors.html) | Unsupported           |
| 8    | [torch.Tensor.uniform_()](https://pytorch.org/docs/1.8.1/tensors.html) | Unsupported           |

### Quasi-random Sampling

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [quasirandom.SobolEngine](https://pytorch.org/docs/1.8.1/generated/torch.quasirandom.SobolEngine.html) | Unsupported           |

## Serialization

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [save](https://pytorch.org/docs/1.8.1/generated/torch.save.html) | Unsupported           |
| 2    | [load](https://pytorch.org/docs/1.8.1/generated/torch.load.html) | Unsupported           |

## Parallelism

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [get_num_threads](https://pytorch.org/docs/1.8.1/generated/torch.get_num_threads.html) | Unsupported           |
| 2    | [set_num_threads](https://pytorch.org/docs/1.8.1/generated/torch.set_num_threads.html) | Unsupported           |
| 3    | [get_num_interop_threads](https://pytorch.org/docs/1.8.1/generated/torch.get_num_interop_threads.html) | Unsupported           |
| 4    | [set_num_interop_threads](https://pytorch.org/docs/1.8.1/generated/torch.set_num_interop_threads.html) | Unsupported           |

## Locally Disabling Gradient Computation

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [no_grad](https://pytorch.org/docs/1.8.1/generated/torch.no_grad.html#torch.no_grad) | Unsupported           |
| 2    | [enable_grad](https://pytorch.org/docs/1.8.1/generated/torch.enable_grad.html#torch.enable_grad) | Unsupported           |
| 3    | set_grad_enabled                                             | Unsupported           |

## Math Operations

### Pointwise Ops

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [abs](https://pytorch.org/docs/1.8.1/generated/torch.abs.html#torch.abs) | Unsupported           |
| 2    | [absolute](https://pytorch.org/docs/1.8.1/generated/torch.absolute.html#torch.absolute) | Unsupported           |
| 3    | [acos](https://pytorch.org/docs/1.8.1/generated/torch.acos.html#torch.acos) | Unsupported           |
| 4    | [arccos](https://pytorch.org/docs/1.8.1/generated/torch.arccos.html#torch.arccos) | Unsupported           |
| 5    | [acosh](https://pytorch.org/docs/1.8.1/generated/torch.acosh.html#torch.acosh) | Unsupported           |
| 6    | [arccosh](https://pytorch.org/docs/1.8.1/generated/torch.arccosh.html#torch.arccosh) | Unsupported           |
| 7    | [add](https://pytorch.org/docs/1.8.1/generated/torch.add.html#torch.add) | Unsupported           |
| 8    | [addcdiv](https://pytorch.org/docs/1.8.1/generated/torch.addcdiv.html#torch.addcdiv) | Unsupported           |
| 9    | [addcmul](https://pytorch.org/docs/1.8.1/generated/torch.addcmul.html#torch.addcmul) | Unsupported           |
| 10   | [angle](https://pytorch.org/docs/1.8.1/generated/torch.angle.html#torch.angle) | Unsupported           |
| 11   | [asin](https://pytorch.org/docs/1.8.1/generated/torch.asin.html#torch.asin) | Unsupported           |
| 12   | [arcsin](https://pytorch.org/docs/1.8.1/generated/torch.arcsin.html#torch.arcsin) | Unsupported           |
| 13   | [asinh](https://pytorch.org/docs/1.8.1/generated/torch.asinh.html#torch.asinh) | Unsupported           |
| 14   | [arcsinh](https://pytorch.org/docs/1.8.1/generated/torch.arcsinh.html#torch.arcsinh) | Unsupported           |
| 15   | [atan](https://pytorch.org/docs/1.8.1/generated/torch.atan.html#torch.atan) | Unsupported           |
| 16   | [arctan](https://pytorch.org/docs/1.8.1/generated/torch.arctan.html#torch.arctan) | Unsupported           |
| 17   | [atanh](https://pytorch.org/docs/1.8.1/generated/torch.atanh.html#torch.atanh) | Unsupported           |
| 18   | [arctanh](https://pytorch.org/docs/1.8.1/generated/torch.arctanh.html#torch.arctanh) | Unsupported           |
| 19   | [atan2](https://pytorch.org/docs/1.8.1/generated/torch.atan2.html#torch.atan2) | Unsupported           |
| 20   | [bitwise_not](https://pytorch.org/docs/1.8.1/generated/torch.bitwise_not.html#torch.bitwise_not) | Unsupported           |
| 21   | [bitwise_and](https://pytorch.org/docs/1.8.1/generated/torch.bitwise_and.html#torch.bitwise_and) | Unsupported           |
| 22   | [bitwise_or](https://pytorch.org/docs/1.8.1/generated/torch.bitwise_or.html#torch.bitwise_or) | Unsupported           |
| 23   | [bitwise_xor](https://pytorch.org/docs/1.8.1/generated/torch.bitwise_xor.html#torch.bitwise_xor) | Unsupported           |
| 24   | [ceil](https://pytorch.org/docs/1.8.1/generated/torch.ceil.html#torch.ceil) | Unsupported           |
| 25   | [clamp](https://pytorch.org/docs/1.8.1/generated/torch.clamp.html#torch.clamp) | Unsupported           |
| 26   | [clip](https://pytorch.org/docs/1.8.1/generated/torch.clip.html#torch.clip) | Unsupported           |
| 27   | [conj](https://pytorch.org/docs/1.8.1/generated/torch.conj.html#torch.conj) | Unsupported           |
| 28   | [copysign](https://pytorch.org/docs/1.8.1/generated/torch.copysign.html#torch.copysign) | Unsupported           |
| 29   | [cos](https://pytorch.org/docs/1.8.1/generated/torch.cos.html#torch.cos) | Unsupported           |
| 30   | [cosh](https://pytorch.org/docs/1.8.1/generated/torch.cosh.html#torch.cosh) | Unsupported           |
| 31   | [deg2rad](https://pytorch.org/docs/1.8.1/generated/torch.deg2rad.html#torch.deg2rad) | Unsupported           |
| 32   | [div](https://pytorch.org/docs/1.8.1/generated/torch.div.html#torch.div) | Unsupported           |
| 33   | [divide](https://pytorch.org/docs/1.8.1/generated/torch.divide.html#torch.divide) | Unsupported           |
| 34   | [digamma](https://pytorch.org/docs/1.8.1/generated/torch.digamma.html#torch.digamma) | Unsupported           |
| 35   | [erf](https://pytorch.org/docs/1.8.1/generated/torch.erf.html#torch.erf) | Unsupported           |
| 36   | [erfc](https://pytorch.org/docs/1.8.1/generated/torch.erfc.html#torch.erfc) | Unsupported           |
| 37   | [erfinv](https://pytorch.org/docs/1.8.1/generated/torch.erfinv.html#torch.erfinv) | Unsupported           |
| 38   | [exp](https://pytorch.org/docs/1.8.1/generated/torch.exp.html#torch.exp) | Unsupported           |
| 39   | [exp2](https://pytorch.org/docs/1.8.1/generated/torch.exp2.html#torch.exp2) | Unsupported           |
| 40   | [expm1](https://pytorch.org/docs/1.8.1/generated/torch.expm1.html#torch.expm1) | Unsupported           |
| 41   | [fake_quantize_per_channel_affine](https://pytorch.org/docs/1.8.1/generated/torch.fake_quantize_per_channel_affine.html#torch.fake_quantize_per_channel_affine) | Unsupported           |
| 42   | [fake_quantize_per_tensor_affine](https://pytorch.org/docs/1.8.1/generated/torch.fake_quantize_per_tensor_affine.html#torch.fake_quantize_per_tensor_affine) | Unsupported           |
| 43   | [fix](https://pytorch.org/docs/1.8.1/generated/torch.fix.html#torch.fix) | Unsupported           |
| 44   | [float_power](https://pytorch.org/docs/1.8.1/generated/torch.float_power.html#torch.float_power) | Unsupported           |
| 45   | [floor](https://pytorch.org/docs/1.8.1/generated/torch.floor.html#torch.floor) | Unsupported           |
| 46   | [floor_divide](https://pytorch.org/docs/1.8.1/generated/torch.floor_divide.html#torch.floor_divide) | Unsupported           |
| 47   | [fmod](https://pytorch.org/docs/1.8.1/generated/torch.fmod.html#torch.fmod) | Unsupported           |
| 48   | [frac](https://pytorch.org/docs/1.8.1/generated/torch.frac.html#torch.frac) | Unsupported           |
| 49   | [imag](https://pytorch.org/docs/1.8.1/generated/torch.imag.html#torch.imag) | Unsupported           |
| 50   | [ldexp](https://pytorch.org/docs/1.8.1/generated/torch.ldexp.html#torch.ldexp) | Unsupported           |
| 51   | [lerp](https://pytorch.org/docs/1.8.1/generated/torch.lerp.html#torch.lerp) | Unsupported           |
| 52   | [lgamma](https://pytorch.org/docs/1.8.1/generated/torch.lgamma.html#torch.lgamma) | Unsupported           |
| 53   | [log](https://pytorch.org/docs/1.8.1/generated/torch.log.html#torch.log) | Unsupported           |
| 54   | [log10](https://pytorch.org/docs/1.8.1/generated/torch.log10.html#torch.log10) | Unsupported           |
| 55   | [log1p](https://pytorch.org/docs/1.8.1/generated/torch.log1p.html#torch.log1p) | Unsupported           |
| 56   | [log2](https://pytorch.org/docs/1.8.1/generated/torch.log2.html#torch.log2) | Unsupported           |
| 57   | [logaddexp](https://pytorch.org/docs/1.8.1/generated/torch.logaddexp.html#torch.logaddexp) | Unsupported           |
| 58   | [logaddexp2](https://pytorch.org/docs/1.8.1/generated/torch.logaddexp2.html#torch.logaddexp2) | Unsupported           |
| 59   | [logical_and](https://pytorch.org/docs/1.8.1/generated/torch.logical_and.html#torch.logical_and) | Unsupported           |
| 60   | [logical_not](https://pytorch.org/docs/1.8.1/generated/torch.logical_not.html#torch.logical_not) | Unsupported           |
| 61   | [logical_or](https://pytorch.org/docs/1.8.1/generated/torch.logical_or.html#torch.logical_or) | Unsupported           |
| 62   | [logical_xor](https://pytorch.org/docs/1.8.1/generated/torch.logical_xor.html#torch.logical_xor) | Unsupported           |
| 63   | [logit](https://pytorch.org/docs/1.8.1/generated/torch.logit.html#torch.logit) | Unsupported           |
| 64   | [hypot](https://pytorch.org/docs/1.8.1/generated/torch.hypot.html#torch.hypot) | Unsupported           |
| 65   | [i0](https://pytorch.org/docs/1.8.1/generated/torch.i0.html#torch.i0) | Unsupported           |
| 66   | [igamma](https://pytorch.org/docs/1.8.1/generated/torch.igamma.html#torch.igamma) | Unsupported           |
| 67   | [igammac](https://pytorch.org/docs/1.8.1/generated/torch.igammac.html#torch.igammac) | Unsupported           |
| 68   | [mul](https://pytorch.org/docs/1.8.1/generated/torch.mul.html#torch.mul) | Unsupported           |
| 69   | [multiply](https://pytorch.org/docs/1.8.1/generated/torch.multiply.html#torch.multiply) | Unsupported           |
| 70   | [mvlgamma](https://pytorch.org/docs/1.8.1/generated/torch.mvlgamma.html#torch.mvlgamma) | Unsupported           |
| 71   | [nan_to_num](https://pytorch.org/docs/1.8.1/generated/torch.nan_to_num.html#torch.nan_to_num) | Unsupported           |
| 72   | [neg](https://pytorch.org/docs/1.8.1/generated/torch.neg.html#torch.neg) | Unsupported           |
| 73   | [negative](https://pytorch.org/docs/1.8.1/generated/torch.negative.html#torch.negative) | Unsupported           |
| 74   | [nextafter](https://pytorch.org/docs/1.8.1/generated/torch.nextafter.html#torch.nextafter) | Unsupported           |
| 75   | [polygamma](https://pytorch.org/docs/1.8.1/generated/torch.polygamma.html#torch.polygamma) | Unsupported           |
| 76   | [pow](https://pytorch.org/docs/1.8.1/generated/torch.pow.html#torch.pow) | Unsupported           |
| 77   | [rad2deg](https://pytorch.org/docs/1.8.1/generated/torch.rad2deg.html#torch.rad2deg) | Unsupported           |
| 78   | [real](https://pytorch.org/docs/1.8.1/generated/torch.real.html#torch.real) | Unsupported           |
| 79   | [reciprocal](https://pytorch.org/docs/1.8.1/generated/torch.reciprocal.html#torch.reciprocal) | Unsupported           |
| 80   | [remainder](https://pytorch.org/docs/1.8.1/generated/torch.remainder.html#torch.remainder) | Unsupported           |
| 81   | [round](https://pytorch.org/docs/1.8.1/generated/torch.round.html#torch.round) | Unsupported           |
| 82   | [rsqrt](https://pytorch.org/docs/1.8.1/generated/torch.rsqrt.html#torch.rsqrt) | Unsupported           |
| 83   | [sigmoid](https://pytorch.org/docs/1.8.1/generated/torch.sigmoid.html#torch.sigmoid) | Unsupported           |
| 84   | [sign](https://pytorch.org/docs/1.8.1/generated/torch.sign.html#torch.sign) | Unsupported           |
| 85   | [sgn](https://pytorch.org/docs/1.8.1/generated/torch.sgn.html#torch.sgn) | Unsupported           |
| 86   | [signbit](https://pytorch.org/docs/1.8.1/generated/torch.signbit.html#torch.signbit) | Unsupported           |
| 87   | [sin](https://pytorch.org/docs/1.8.1/generated/torch.sin.html#torch.sin) | Unsupported           |
| 88   | [sinc](https://pytorch.org/docs/1.8.1/generated/torch.sinc.html#torch.sinc) | Unsupported           |
| 89   | [sinh](https://pytorch.org/docs/1.8.1/generated/torch.sinh.html#torch.sinh) | Unsupported           |
| 90   | [sqrt](https://pytorch.org/docs/1.8.1/generated/torch.sqrt.html#torch.sqrt) | Unsupported           |
| 91   | [square](https://pytorch.org/docs/1.8.1/generated/torch.square.html#torch.square) | Unsupported           |
| 92   | [sub](https://pytorch.org/docs/1.8.1/generated/torch.sub.html#torch.sub) | Unsupported           |
| 93   | [subtract](https://pytorch.org/docs/1.8.1/generated/torch.subtract.html#torch.subtract) | Unsupported           |
| 94   | [tan](https://pytorch.org/docs/1.8.1/generated/torch.tan.html#torch.tan) | Unsupported           |
| 95   | [tanh](https://pytorch.org/docs/1.8.1/generated/torch.tanh.html#torch.tanh) | Unsupported           |
| 96   | [true_divide](https://pytorch.org/docs/1.8.1/generated/torch.true_divide.html#torch.true_divide) | Unsupported           |
| 97   | [trunc](https://pytorch.org/docs/1.8.1/generated/torch.trunc.html#torch.trunc) | Unsupported           |
| 98   | [xlogy](https://pytorch.org/docs/1.8.1/generated/torch.xlogy.html#torch.xlogy) | Unsupported           |

### Reduction Ops

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [argmax](https://pytorch.org/docs/1.8.1/generated/torch.argmax.html#torch.argmax) | Unsupported           |
| 2    | [argmin](https://pytorch.org/docs/1.8.1/generated/torch.argmin.html#torch.argmin) | Unsupported           |
| 3    | [amax](https://pytorch.org/docs/1.8.1/generated/torch.amax.html#torch.amax) | Unsupported           |
| 4    | [amin](https://pytorch.org/docs/1.8.1/generated/torch.amin.html#torch.amin) | Unsupported           |
| 5    | [all](https://pytorch.org/docs/1.8.1/generated/torch.all.html#torch.all) | Unsupported           |
| 6    | [any](https://pytorch.org/docs/1.8.1/generated/torch.any.html#torch.any) | Unsupported           |
| 7    | [max](https://pytorch.org/docs/1.8.1/generated/torch.max.html#torch.max) | Unsupported           |
| 8    | [min](https://pytorch.org/docs/1.8.1/generated/torch.min.html#torch.min) | Unsupported           |
| 9    | [dist](https://pytorch.org/docs/1.8.1/generated/torch.dist.html#torch.dist) | Unsupported           |
| 10   | [logsumexp](https://pytorch.org/docs/1.8.1/generated/torch.logsumexp.html#torch.logsumexp) | Unsupported           |
| 11   | [mean](https://pytorch.org/docs/1.8.1/generated/torch.mean.html#torch.mean) | Unsupported           |
| 12   | [median](https://pytorch.org/docs/1.8.1/generated/torch.median.html#torch.median) | Unsupported           |
| 13   | [nanmedian](https://pytorch.org/docs/1.8.1/generated/torch.nanmedian.html#torch.nanmedian) | Unsupported           |
| 14   | [mode](https://pytorch.org/docs/1.8.1/generated/torch.mode.html#torch.mode) | Unsupported           |
| 15   | [norm](https://pytorch.org/docs/1.8.1/generated/torch.norm.html#torch.norm) | Unsupported           |
| 16   | [nansum](https://pytorch.org/docs/1.8.1/generated/torch.nansum.html#torch.nansum) | Unsupported           |
| 17   | [prod](https://pytorch.org/docs/1.8.1/generated/torch.prod.html#torch.prod) | Unsupported           |
| 18   | [quantile](https://pytorch.org/docs/1.8.1/generated/torch.quantile.html#torch.quantile) | Unsupported           |
| 19   | [nanquantile](https://pytorch.org/docs/1.8.1/generated/torch.nanquantile.html#torch.nanquantile) | Unsupported           |
| 20   | [std](https://pytorch.org/docs/1.8.1/generated/torch.std.html#torch.std) | Unsupported           |
| 21   | [std_mean](https://pytorch.org/docs/1.8.1/generated/torch.std_mean.html#torch.std_mean) | Unsupported           |
| 22   | [sum](https://pytorch.org/docs/1.8.1/generated/torch.sum.html#torch.sum) | Unsupported           |
| 23   | [unique](https://pytorch.org/docs/1.8.1/generated/torch.unique.html#torch.unique) | Unsupported           |
| 24   | [unique_consecutive](https://pytorch.org/docs/1.8.1/generated/torch.unique_consecutive.html#torch.unique_consecutive) | Unsupported           |
| 25   | [var](https://pytorch.org/docs/1.8.1/generated/torch.var.html#torch.var) | Unsupported           |
| 26   | [var_mean](https://pytorch.org/docs/1.8.1/generated/torch.var_mean.html#torch.var_mean) | Unsupported           |
| 27   | [count_nonzero](https://pytorch.org/docs/1.8.1/generated/torch.count_nonzero.html#torch.count_nonzero) | Unsupported           |

### Comparison Ops

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [allclose](https://pytorch.org/docs/1.8.1/generated/torch.allclose.html#torch.allclose) | Unsupported           |
| 2    | [argsort](https://pytorch.org/docs/1.8.1/generated/torch.argsort.html#torch.argsort) | Unsupported           |
| 3    | [eq](https://pytorch.org/docs/1.8.1/generated/torch.eq.html#torch.eq) | Unsupported           |
| 4    | [equal](https://pytorch.org/docs/1.8.1/generated/torch.equal.html#torch.equal) | Unsupported           |
| 5    | [ge](https://pytorch.org/docs/1.8.1/generated/torch.ge.html#torch.ge) | Unsupported           |
| 6    | [greater_equal](https://pytorch.org/docs/1.8.1/generated/torch.greater_equal.html#torch.greater_equal) | Unsupported           |
| 7    | [gt](https://pytorch.org/docs/1.8.1/generated/torch.gt.html#torch.gt) | Unsupported           |
| 8    | [greater](https://pytorch.org/docs/1.8.1/generated/torch.greater.html#torch.greater) | Unsupported           |
| 9    | [isclose](https://pytorch.org/docs/1.8.1/generated/torch.isclose.html#torch.isclose) | Unsupported           |
| 10   | [isfinite](https://pytorch.org/docs/1.8.1/generated/torch.isfinite.html#torch.isfinite) | Unsupported           |
| 11   | [isinf](https://pytorch.org/docs/1.8.1/generated/torch.isinf.html#torch.isinf) | Unsupported           |
| 12   | [isposinf](https://pytorch.org/docs/1.8.1/generated/torch.isposinf.html#torch.isposinf) | Unsupported           |
| 13   | [isneginf](https://pytorch.org/docs/1.8.1/generated/torch.isneginf.html#torch.isneginf) | Unsupported           |
| 14   | [isnan](https://pytorch.org/docs/1.8.1/generated/torch.isnan.html#torch.isnan) | Unsupported           |
| 15   | [isreal](https://pytorch.org/docs/1.8.1/generated/torch.isreal.html#torch.isreal) | Unsupported           |
| 16   | [kthvalue](https://pytorch.org/docs/1.8.1/generated/torch.kthvalue.html#torch.kthvalue) | Unsupported           |
| 17   | [le](https://pytorch.org/docs/1.8.1/generated/torch.le.html#torch.le) | Unsupported           |
| 18   | [less_equal](https://pytorch.org/docs/1.8.1/generated/torch.less_equal.html#torch.less_equal) | Unsupported           |
| 19   | [lt](https://pytorch.org/docs/1.8.1/generated/torch.lt.html#torch.lt) | Unsupported           |
| 20   | [less](https://pytorch.org/docs/1.8.1/generated/torch.less.html#torch.less) | Unsupported           |
| 21   | [maximum](https://pytorch.org/docs/1.8.1/generated/torch.maximum.html#torch.maximum) | Unsupported           |
| 22   | [minimum](https://pytorch.org/docs/1.8.1/generated/torch.minimum.html#torch.minimum) | Unsupported           |
| 23   | [fmax](https://pytorch.org/docs/1.8.1/generated/torch.fmax.html#torch.fmax) | Unsupported           |
| 24   | [fmin](https://pytorch.org/docs/1.8.1/generated/torch.fmin.html#torch.fmin) | Unsupported           |
| 25   | [ne](https://pytorch.org/docs/1.8.1/generated/torch.ne.html#torch.ne) | Unsupported           |
| 26   | [not_equal](https://pytorch.org/docs/1.8.1/generated/torch.not_equal.html#torch.not_equal) | Unsupported           |
| 27   | [sort](https://pytorch.org/docs/1.8.1/generated/torch.sort.html#torch.sort) | Unsupported           |
| 28   | [topk](https://pytorch.org/docs/1.8.1/generated/torch.topk.html#torch.topk) | Unsupported           |
| 29   | [msort](https://pytorch.org/docs/1.8.1/generated/torch.msort.html#torch.msort) | Unsupported           |

### Spectral Ops

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [stft](https://pytorch.org/docs/1.8.1/generated/torch.stft.html#torch.stft) | Unsupported           |
| 2    | [istft](https://pytorch.org/docs/1.8.1/generated/torch.istft.html#torch.istft) | Unsupported           |
| 3    | [bartlett_window](https://pytorch.org/docs/1.8.1/generated/torch.bartlett_window.html#torch.bartlett_window) | Unsupported           |
| 4    | [blackman_window](https://pytorch.org/docs/1.8.1/generated/torch.blackman_window.html#torch.blackman_window) | Unsupported           |
| 5    | [hamming_window](https://pytorch.org/docs/1.8.1/generated/torch.hamming_window.html#torch.hamming_window) | Unsupported           |
| 6    | [hann_window](https://pytorch.org/docs/1.8.1/generated/torch.hann_window.html#torch.hann_window) | Unsupported           |
| 7    | [kaiser_window](https://pytorch.org/docs/1.8.1/generated/torch.kaiser_window.html#torch.kaiser_window) | Unsupported           |

### Other Operations

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [atleast_1d](https://pytorch.org/docs/1.8.1/generated/torch.atleast_1d.html#torch.atleast_1d) | Unsupported           |
| 2    | [atleast_2d](https://pytorch.org/docs/1.8.1/generated/torch.atleast_2d.html#torch.atleast_2d) | Unsupported           |
| 3    | [atleast_3d](https://pytorch.org/docs/1.8.1/generated/torch.atleast_3d.html#torch.atleast_3d) | Unsupported           |
| 4    | [bincount](https://pytorch.org/docs/1.8.1/generated/torch.bincount.html#torch.bincount) | Unsupported           |
| 5    | [block_diag](https://pytorch.org/docs/1.8.1/generated/torch.block_diag.html#torch.block_diag) | Unsupported           |
| 6    | [broadcast_tensors](https://pytorch.org/docs/1.8.1/generated/torch.broadcast_tensors.html#torch.broadcast_tensors) | Unsupported           |
| 7    | [broadcast_to](https://pytorch.org/docs/1.8.1/generated/torch.broadcast_to.html#torch.broadcast_to) | Unsupported           |
| 8    | [broadcast_shapes](https://pytorch.org/docs/1.8.1/generated/torch.broadcast_shapes.html#torch.broadcast_shapes) | Unsupported           |
| 9    | [bucketize](https://pytorch.org/docs/1.8.1/generated/torch.bucketize.html#torch.bucketize) | Unsupported           |
| 10   | [cartesian_prod](https://pytorch.org/docs/1.8.1/generated/torch.cartesian_prod.html#torch.cartesian_prod) | Unsupported           |
| 11   | [cdist](https://pytorch.org/docs/1.8.1/generated/torch.cdist.html#torch.cdist) | Unsupported           |
| 12   | [clone](https://pytorch.org/docs/1.8.1/generated/torch.clone.html#torch.clone) | Unsupported           |
| 13   | [combinations](https://pytorch.org/docs/1.8.1/generated/torch.combinations.html#torch.combinations) | Unsupported           |
| 14   | [cross](https://pytorch.org/docs/1.8.1/generated/torch.cross.html#torch.cross) | Unsupported           |
| 15   | [cummax](https://pytorch.org/docs/1.8.1/generated/torch.cummax.html#torch.cummax) | Unsupported           |
| 16   | [cummin](https://pytorch.org/docs/1.8.1/generated/torch.cummin.html#torch.cummin) | Unsupported           |
| 17   | [cumprod](https://pytorch.org/docs/1.8.1/generated/torch.cumprod.html#torch.cumprod) | Unsupported           |
| 18   | [cumsum](https://pytorch.org/docs/1.8.1/generated/torch.cumsum.html#torch.cumsum) | Unsupported           |
| 19   | [diag](https://pytorch.org/docs/1.8.1/generated/torch.diag.html#torch.diag) | Unsupported           |
| 20   | [diag_embed](https://pytorch.org/docs/1.8.1/generated/torch.diag_embed.html#torch.diag_embed) | Unsupported           |
| 21   | [diagflat](https://pytorch.org/docs/1.8.1/generated/torch.diagflat.html#torch.diagflat) | Unsupported           |
| 22   | [diagonal](https://pytorch.org/docs/1.8.1/generated/torch.diagonal.html#torch.diagonal) | Unsupported           |
| 23   | [diff](https://pytorch.org/docs/1.8.1/generated/torch.diff.html#torch.diff) | Unsupported           |
| 24   | [einsum](https://pytorch.org/docs/1.8.1/generated/torch.einsum.html#torch.einsum) | Unsupported           |
| 25   | [flatten](https://pytorch.org/docs/1.8.1/generated/torch.flatten.html#torch.flatten) | Unsupported           |
| 26   | [flip](https://pytorch.org/docs/1.8.1/generated/torch.flip.html#torch.flip) | Unsupported           |
| 27   | [fliplr](https://pytorch.org/docs/1.8.1/generated/torch.fliplr.html#torch.fliplr) | Unsupported           |
| 28   | [flipud](https://pytorch.org/docs/1.8.1/generated/torch.flipud.html#torch.flipud) | Unsupported           |
| 29   | [kron](https://pytorch.org/docs/1.8.1/generated/torch.kron.html#torch.kron) | Unsupported           |
| 30   | [rot90](https://pytorch.org/docs/1.8.1/generated/torch.rot90.html#torch.rot90) | Unsupported           |
| 31   | [gcd](https://pytorch.org/docs/1.8.1/generated/torch.gcd.html#torch.gcd) | Unsupported           |
| 32   | [histc](https://pytorch.org/docs/1.8.1/generated/torch.histc.html#torch.histc) | Unsupported           |
| 33   | [meshgrid](https://pytorch.org/docs/1.8.1/generated/torch.meshgrid.html#torch.meshgrid) | Unsupported           |
| 34   | [lcm](https://pytorch.org/docs/1.8.1/generated/torch.lcm.html#torch.lcm) | Unsupported           |
| 35   | [logcumsumexp](https://pytorch.org/docs/1.8.1/generated/torch.logcumsumexp.html#torch.logcumsumexp) | Unsupported           |
| 36   | [ravel](https://pytorch.org/docs/1.8.1/generated/torch.ravel.html#torch.ravel) | Unsupported           |
| 37   | [renorm](https://pytorch.org/docs/1.8.1/generated/torch.renorm.html#torch.renorm) | Unsupported           |
| 38   | [repeat_interleave](https://pytorch.org/docs/1.8.1/generated/torch.repeat_interleave.html#torch.repeat_interleave) | Unsupported           |
| 39   | [roll](https://pytorch.org/docs/1.8.1/generated/torch.roll.html#torch.roll) | Unsupported           |
| 40   | [searchsorted](https://pytorch.org/docs/1.8.1/generated/torch.searchsorted.html#torch.searchsorted) | Unsupported           |
| 41   | [tensordot](https://pytorch.org/docs/1.8.1/generated/torch.tensordot.html#torch.tensordot) | Unsupported           |
| 42   | [trace](https://pytorch.org/docs/1.8.1/generated/torch.trace.html#torch.trace) | Unsupported           |
| 43   | [tril](https://pytorch.org/docs/1.8.1/generated/torch.tril.html#torch.tril) | Unsupported           |
| 44   | [tril_indices](https://pytorch.org/docs/1.8.1/generated/torch.tril_indices.html#torch.tril_indices) | Unsupported           |
| 45   | [triu](https://pytorch.org/docs/1.8.1/generated/torch.triu.html#torch.triu) | Unsupported           |
| 46   | [triu_indices](https://pytorch.org/docs/1.8.1/generated/torch.triu_indices.html#torch.triu_indices) | Unsupported           |
| 47   | [vander](https://pytorch.org/docs/1.8.1/generated/torch.vander.html#torch.vander) | Unsupported           |
| 48   | [view_as_real](https://pytorch.org/docs/1.8.1/generated/torch.view_as_real.html#torch.view_as_real) | Unsupported           |
| 49   | [view_as_complex](https://pytorch.org/docs/1.8.1/generated/torch.view_as_complex.html#torch.view_as_complex) | Unsupported           |

### BLAS and LAPACK Operations

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [addbmm](https://pytorch.org/docs/1.8.1/generated/torch.addbmm.html#torch.addbmm) | Unsupported           |
| 2    | [addmm](https://pytorch.org/docs/1.8.1/generated/torch.addmm.html#torch.addmm) | Unsupported           |
| 3    | [addmv](https://pytorch.org/docs/1.8.1/generated/torch.addmv.html#torch.addmv) | Unsupported           |
| 4    | [addr](https://pytorch.org/docs/1.8.1/generated/torch.addr.html#torch.addr) | Unsupported           |
| 5    | [baddbmm](https://pytorch.org/docs/1.8.1/generated/torch.baddbmm.html#torch.baddbmm) | Unsupported           |
| 6    | [bmm](https://pytorch.org/docs/1.8.1/generated/torch.bmm.html#torch.bmm) | Unsupported           |
| 7    | [chain_matmul](https://pytorch.org/docs/1.8.1/generated/torch.chain_matmul.html#torch.chain_matmul) | Unsupported           |
| 8    | [cholesky](https://pytorch.org/docs/1.8.1/generated/torch.cholesky.html#torch.cholesky) | Unsupported           |
| 9    | [cholesky_inverse](https://pytorch.org/docs/1.8.1/generated/torch.cholesky_inverse.html#torch.cholesky_inverse) | Unsupported           |
| 10   | [cholesky_solve](https://pytorch.org/docs/1.8.1/generated/torch.cholesky_solve.html#torch.cholesky_solve) | Unsupported           |
| 11   | [dot](https://pytorch.org/docs/1.8.1/generated/torch.dot.html#torch.dot) | Unsupported           |
| 12   | [eig](https://pytorch.org/docs/1.8.1/generated/torch.eig.html#torch.eig) | Unsupported           |
| 13   | [geqrf](https://pytorch.org/docs/1.8.1/generated/torch.geqrf.html#torch.geqrf) | Unsupported           |
| 14   | [ger](https://pytorch.org/docs/1.8.1/generated/torch.ger.html#torch.ger) | Unsupported           |
| 15   | [inner](https://pytorch.org/docs/1.8.1/generated/torch.inner.html#torch.inner) | Unsupported           |
| 16   | [inverse](https://pytorch.org/docs/1.8.1/generated/torch.inverse.html#torch.inverse) | Unsupported           |
| 17   | [det](https://pytorch.org/docs/1.8.1/generated/torch.det.html#torch.det) | Unsupported           |
| 18   | [logdet](https://pytorch.org/docs/1.8.1/generated/torch.logdet.html#torch.logdet) | Unsupported           |
| 19   | [slogdet](https://pytorch.org/docs/1.8.1/generated/torch.slogdet.html#torch.slogdet) | Unsupported           |
| 20   | [lstsq](https://pytorch.org/docs/1.8.1/generated/torch.lstsq.html#torch.lstsq) | Unsupported           |
| 21   | [lu](https://pytorch.org/docs/1.8.1/generated/torch.lu.html#torch.lu) | Unsupported           |
| 22   | [lu_solve](https://pytorch.org/docs/1.8.1/generated/torch.lu_solve.html#torch.lu_solve) | Unsupported           |
| 23   | [lu_unpack](https://pytorch.org/docs/1.8.1/generated/torch.lu_unpack.html#torch.lu_unpack) | Unsupported           |
| 24   | [matmul](https://pytorch.org/docs/1.8.1/generated/torch.matmul.html#torch.matmul) | Unsupported           |
| 25   | [matrix_power](https://pytorch.org/docs/1.8.1/generated/torch.matrix_power.html#torch.matrix_power) | Unsupported           |
| 26   | [matrix_rank](https://pytorch.org/docs/1.8.1/generated/torch.matrix_rank.html#torch.matrix_rank) | Unsupported           |
| 27   | [matrix_exp](https://pytorch.org/docs/1.8.1/generated/torch.matrix_exp.html#torch.matrix_exp) | Unsupported           |
| 28   | [mm](https://pytorch.org/docs/1.8.1/generated/torch.mm.html#torch.mm) | Unsupported           |
| 29   | [mv](https://pytorch.org/docs/1.8.1/generated/torch.mv.html#torch.mv) | Unsupported           |
| 30   | [orgqr](https://pytorch.org/docs/1.8.1/generated/torch.orgqr.html#torch.orgqr) | Unsupported           |
| 31   | [ormqr](https://pytorch.org/docs/1.8.1/generated/torch.ormqr.html#torch.ormqr) | Unsupported           |
| 32   | [outer](https://pytorch.org/docs/1.8.1/generated/torch.outer.html#torch.outer) | Unsupported           |
| 33   | [pinverse](https://pytorch.org/docs/1.8.1/generated/torch.pinverse.html#torch.pinverse) | Unsupported           |
| 34   | [qr](https://pytorch.org/docs/1.8.1/generated/torch.qr.html#torch.qr) | Unsupported           |
| 35   | [solve](https://pytorch.org/docs/1.8.1/generated/torch.solve.html#torch.solve) | Unsupported           |
| 36   | [svd](https://pytorch.org/docs/1.8.1/generated/torch.svd.html#torch.svd) | Unsupported           |
| 37   | [svd_lowrank](https://pytorch.org/docs/1.8.1/generated/torch.svd_lowrank.html#torch.svd_lowrank) | Unsupported           |
| 38   | [pca_lowrank](https://pytorch.org/docs/1.8.1/generated/torch.pca_lowrank.html#torch.pca_lowrank) | Unsupported           |
| 39   | [symeig](https://pytorch.org/docs/1.8.1/generated/torch.symeig.html#torch.symeig) | Unsupported           |
| 40   | [lobpcg](https://pytorch.org/docs/1.8.1/generated/torch.lobpcg.html#torch.lobpcg) | Unsupported           |
| 41   | [trapz](https://pytorch.org/docs/1.8.1/generated/torch.trapz.html#torch.trapz) | Unsupported           |
| 42   | [triangular_solve](https://pytorch.org/docs/1.8.1/generated/torch.triangular_solve.html#torch.triangular_solve) | Unsupported           |
| 43   | [vdot](https://pytorch.org/docs/1.8.1/generated/torch.vdot.html#torch.vdot) | Unsupported           |

## Utilities

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [compiled_with_cxx11_abi](https://pytorch.org/docs/1.8.1/generated/torch.compiled_with_cxx11_abi.html#torch.compiled_with_cxx11_abi) | Unsupported           |
| 2    | [result_type](https://pytorch.org/docs/1.8.1/generated/torch.result_type.html#torch.result_type) | Unsupported           |
| 3    | [can_cast](https://pytorch.org/docs/1.8.1/generated/torch.can_cast.html#torch.can_cast) | Unsupported           |
| 4    | [promote_types](https://pytorch.org/docs/1.8.1/generated/torch.promote_types.html#torch.promote_types) | Unsupported           |
| 5    | [use_deterministic_algorithms](https://pytorch.org/docs/1.8.1/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms) | Unsupported           |
| 6    | [are_deterministic_algorithms_enabled](https://pytorch.org/docs/1.8.1/generated/torch.are_deterministic_algorithms_enabled.html#torch.are_deterministic_algorithms_enabled) | Unsupported           |
| 7    | [_assert](https://pytorch.org/docs/1.8.1/generated/torch._assert.html#torch._assert) | Unsupported           |

# Layers (torch.nn)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [Parameter](https://pytorch.org/docs/1.8.1/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) | Unsupported           |
| 2    | [UninitializedParameter](https://pytorch.org/docs/1.8.1/generated/torch.nn.parameter.UninitializedParameter.html#torch.nn.parameter.UninitializedParameter) | Unsupported           |

## [Containers](https://pytorch.org/docs/1.8.1/nn.html#id1)


| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [Module](https://pytorch.org/docs/1.8.1/generated/torch.nn.Module.html#torch.nn.Module) | Unsupported           |
| 2    | [Sequential](https://pytorch.org/docs/1.8.1/generated/torch.nn.Sequential.html#torch.nn.Sequential) | Unsupported           |
| 3    | [ModuleList](https://pytorch.org/docs/1.8.1/generated/torch.nn.ModuleList.html#torch.nn.ModuleList) | Unsupported           |
| 4    | [ModuleDict](https://pytorch.org/docs/1.8.1/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict) | Unsupported           |
| 5    | [ParameterList](https://pytorch.org/docs/1.8.1/generated/torch.nn.ParameterList.html#torch.nn.ParameterList) | Unsupported           |
| 6    | [ParameterDict](https://pytorch.org/docs/1.8.1/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict) | Unsupported           |

### Global Hooks For Module

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [register_module_forward_pre_hook](https://pytorch.org/docs/1.8.1/generated/torch.nn.modules.module.register_module_forward_pre_hook.html#torch.nn.modules.module.register_module_forward_pre_hook) | Unsupported           |
| 2    | [register_module_forward_hook](https://pytorch.org/docs/1.8.1/generated/torch.nn.modules.module.register_module_forward_hook.html#torch.nn.modules.module.register_module_forward_hook) | Unsupported           |
| 3    | [register_module_backward_hook](https://pytorch.org/docs/1.8.1/generated/torch.nn.modules.module.register_module_backward_hook.html#torch.nn.modules.module.register_module_backward_hook) | Unsupported           |

## [Convolution Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.Conv1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv1d.html#torch.nn.Conv1d) | Unsupported           |
| 2    | [nn.Conv2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv2d.html#torch.nn.Conv2d) | Unsupported           |
| 3    | [nn.Conv3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv3d.html#torch.nn.Conv3d) | Unsupported           |
| 4    | [nn.ConvTranspose1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose1d.html#torch.nn.ConvTranspose1d) | Unsupported           |
| 5    | [nn.ConvTranspose2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d) | Unsupported           |
| 6    | [nn.ConvTranspose3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose3d.html#torch.nn.ConvTranspose3d) | Unsupported           |
| 7    | [nn.LazyConv1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConv1d.html#torch.nn.LazyConv1d) | Unsupported           |
| 8    | [nn.LazyConv2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d) | Unsupported           |
| 9    | [nn.LazyConv3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConv3d.html#torch.nn.LazyConv3d) | Unsupported           |
| 10   | [nn.LazyConvTranspose1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConvTranspose1d.html#torch.nn.LazyConvTranspose1d) | Unsupported           |
| 11   | [nn.LazyConvTranspose2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConvTranspose2d.html#torch.nn.LazyConvTranspose2d) | Unsupported           |
| 12   | [nn.LazyConvTranspose3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConvTranspose3d.html#torch.nn.LazyConvTranspose3d) | Unsupported           |
| 13   | [nn.Unfold](https://pytorch.org/docs/1.8.1/generated/torch.nn.Unfold.html#torch.nn.Unfold) | Unsupported           |
| 14   | [nn.Fold](https://pytorch.org/docs/1.8.1/generated/torch.nn.Fold.html#torch.nn.Fold) | Unsupported           |

## [Pooling Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.MaxPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d) | Unsupported           |
| 2    | [nn.MaxPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d) | Unsupported           |
| 3    | [nn.MaxPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool3d.html#torch.nn.MaxPool3d) | Unsupported           |
| 4    | [nn.MaxUnpool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxUnpool1d.html#torch.nn.MaxUnpool1d) | Unsupported           |
| 5    | [nn.MaxUnpool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxUnpool2d.html#torch.nn.MaxUnpool2d) | Unsupported           |
| 6    | [nn.MaxUnpool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxUnpool3d.html#torch.nn.MaxUnpool3d) | Unsupported           |
| 7    | [nn.AvgPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool1d.html#torch.nn.AvgPool1d) | Unsupported           |
| 8    | [nn.AvgPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d) | Unsupported           |
| 9    | [nn.AvgPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool3d.html#torch.nn.AvgPool3d) | Unsupported           |
| 10   | [nn.FractionalMaxPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.FractionalMaxPool2d.html#torch.nn.FractionalMaxPool2d) | Unsupported           |
| 11   | [nn.LPPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LPPool1d.html#torch.nn.LPPool1d) | Unsupported           |
| 12   | [nn.LPPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LPPool2d.html#torch.nn.LPPool2d) | Unsupported           |
| 13   | [nn.AdaptiveMaxPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveMaxPool1d.html#torch.nn.AdaptiveMaxPool1d) | Unsupported           |
| 14   | [nn.AdaptiveMaxPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveMaxPool2d.html#torch.nn.AdaptiveMaxPool2d) | Unsupported           |
| 15   | [nn.AdaptiveMaxPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveMaxPool3d.html#torch.nn.AdaptiveMaxPool3d) | Unsupported           |
| 16   | [nn.AdaptiveAvgPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveAvgPool1d.html#torch.nn.AdaptiveAvgPool1d) | Unsupported           |
| 17   | [nn.AdaptiveAvgPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d) | Unsupported           |
| 18   | [nn.AdaptiveAvgPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveAvgPool3d.html#torch.nn.AdaptiveAvgPool3d) | Unsupported           |

## [Padding Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.ReflectionPad1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReflectionPad1d.html#torch.nn.ReflectionPad1d) | Unsupported           |
| 2    | [nn.ReflectionPad2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReflectionPad2d.html#torch.nn.ReflectionPad2d) | Unsupported           |
| 3    | [nn.ReplicationPad1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReplicationPad1d.html#torch.nn.ReplicationPad1d) | Unsupported           |
| 4    | [nn.ReplicationPad2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReplicationPad2d.html#torch.nn.ReplicationPad2d) | Unsupported           |
| 5    | [nn.ReplicationPad3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReplicationPad3d.html#torch.nn.ReplicationPad3d) | Unsupported           |
| 6    | [nn.ZeroPad2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ZeroPad2d.html#torch.nn.ZeroPad2d) | Unsupported           |
| 7    | [nn.ConstantPad1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConstantPad1d.html#torch.nn.ConstantPad1d) | Unsupported           |
| 8    | [nn.ConstantPad2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConstantPad2d.html#torch.nn.ConstantPad2d) | Unsupported           |
| 9    | [nn.ConstantPad3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConstantPad3d.html#torch.nn.ConstantPad3d) | Unsupported           |



## [Non-Linear Activations (Weighted sum, Nonlinearity)](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.ELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.ELU.html#torch.nn.ELU) | Unsupported           |
| 2    | [nn.Hardshrink](https://pytorch.org/docs/1.8.1/generated/torch.nn.Hardshrink.html#torch.nn.Hardshrink) | Unsupported           |
| 3    | [nn.Hardsigmoid](https://pytorch.org/docs/1.8.1/generated/torch.nn.Hardsigmoid.html#torch.nn.Hardsigmoid) | Unsupported           |
| 4    | [nn.Hardtanh](https://pytorch.org/docs/1.8.1/generated/torch.nn.Hardtanh.html#torch.nn.Hardtanh) | Unsupported           |
| 5    | [nn.Hardswish](https://pytorch.org/docs/1.8.1/generated/torch.nn.Hardswish.html#torch.nn.Hardswish) | Unsupported           |
| 6    | [nn.LeakyReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU) | Unsupported           |
| 7    | [nn.LogSigmoid](https://pytorch.org/docs/1.8.1/generated/torch.nn.LogSigmoid.html#torch.nn.LogSigmoid) | Unsupported           |
| 8    | [nn.MultiheadAttention](https://pytorch.org/docs/1.8.1/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention) | Unsupported           |
| 9    | [nn.PReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.PReLU.html#torch.nn.PReLU) | Unsupported           |
| 10   | [nn.ReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReLU.html#torch.nn.ReLU) | Unsupported           |
| 11   | [nn.ReLU6](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReLU6.html#torch.nn.ReLU6) | Unsupported           |
| 12   | [nn.RReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.RReLU.html#torch.nn.RReLU) | Unsupported           |
| 13   | [nn.SELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.SELU.html#torch.nn.SELU) | Unsupported           |
| 14   | [nn.CELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.CELU.html#torch.nn.CELU) | Unsupported           |
| 15   | [nn.GELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.GELU.html#torch.nn.GELU) | Unsupported           |
| 16   | [nn.Sigmoid](https://pytorch.org/docs/1.8.1/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid) | Unsupported           |
| 17   | [nn.SiLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.SiLU.html#torch.nn.SiLU) | Unsupported           |
| 18   | [nn.Softplus](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softplus.html#torch.nn.Softplus) | Unsupported           |
| 19   | [nn.Softshrink](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softshrink.html#torch.nn.Softshrink) | Unsupported           |
| 20   | [nn.Softsign](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softsign.html#torch.nn.Softsign) | Unsupported           |
| 21   | [nn.Tanh](https://pytorch.org/docs/1.8.1/generated/torch.nn.Tanh.html#torch.nn.Tanh) | Unsupported           |
| 22   | [nn.Tanhshrink](https://pytorch.org/docs/1.8.1/generated/torch.nn.Tanhshrink.html#torch.nn.Tanhshrink) | Unsupported           |
| 23   | [nn.Threshold](https://pytorch.org/docs/1.8.1/generated/torch.nn.Threshold.html#torch.nn.Threshold) | Unsupported           |

## [Non-Linear Activations (Other)](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.Softmin](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmin.html#torch.nn.Softmin) | Unsupported           |
| 2    | [nn.Softmax](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmax.html#torch.nn.Softmax) | Unsupported           |
| 3    | [nn.Softmax2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmax2d.html#torch.nn.Softmax2d) | Unsupported           |
| 4    | [nn.LogSoftmax](https://pytorch.org/docs/1.8.1/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) | Unsupported           |
| 5    | [nn.AdaptiveLogSoftmaxWithLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss) | Unsupported           |

## [Normalization Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.BatchNorm1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d) | Unsupported           |
| 2    | [nn.BatchNorm2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d) | Unsupported           |
| 3    | [nn.BatchNorm3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm3d.html#torch.nn.BatchNorm3d) | Unsupported           |
| 4    | [nn.GroupNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.GroupNorm.html#torch.nn.GroupNorm) | Unsupported           |
| 5    | [nn.SyncBatchNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm) | Unsupported           |
| 6    | [nn.InstanceNorm1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.InstanceNorm1d.html#torch.nn.InstanceNorm1d) | Unsupported           |
| 7    | [nn.InstanceNorm2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d) | Unsupported           |
| 8    | [nn.InstanceNorm3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.InstanceNorm3d.html#torch.nn.InstanceNorm3d) | Unsupported           |
| 9    | [nn.LayerNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm) | Unsupported           |
| 10   | [nn.LocalResponseNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.LocalResponseNorm.html#torch.nn.LocalResponseNorm) | Unsupported           |



## [Recurrent Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.RNNBase](https://pytorch.org/docs/1.8.1/generated/torch.nn.RNNBase.html#torch.nn.RNNBase) | Unsupported           |
| 2    | [nn.RNN](https://pytorch.org/docs/1.8.1/generated/torch.nn.RNN.html#torch.nn.RNN) | Unsupported           |
| 3    | [nn.LSTM](https://pytorch.org/docs/1.8.1/generated/torch.nn.LSTM.html#torch.nn.LSTM) | Unsupported           |
| 4    | [nn.GRU](https://pytorch.org/docs/1.8.1/generated/torch.nn.GRU.html#torch.nn.GRU) | Unsupported           |
| 5    | [nn.RNNCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.RNNCell.html#torch.nn.RNNCell) | Unsupported           |
| 6    | [nn.LSTMCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.LSTMCell.html#torch.nn.LSTMCell) | Unsupported           |
| 7    | [nn.GRUCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.GRUCell.html#torch.nn.GRUCell) | Unsupported           |



## [Transformer Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.Transformer](https://pytorch.org/docs/1.8.1/generated/torch.nn.Transformer.html#torch.nn.Transformer) | Unsupported           |
| 2    | [nn.TransformerEncoder](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder) | Unsupported           |
| 3    | [nn.TransformerDecoder](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder) | Unsupported           |
| 4    | [nn.TransformerEncoderLayer](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer) | Unsupported           |
| 5    | [nn.TransformerDecoderLayer](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer) | Unsupported           |



## [Linear Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.Identity](https://pytorch.org/docs/1.8.1/generated/torch.nn.Identity.html#torch.nn.Identity) | Unsupported           |
| 2    | [nn.Linear](https://pytorch.org/docs/1.8.1/generated/torch.nn.Linear.html#torch.nn.Linear) | Unsupported           |
| 3    | [nn.Bilinear](https://pytorch.org/docs/1.8.1/generated/torch.nn.Bilinear.html#torch.nn.Bilinear) | Unsupported           |
| 4    | [nn.LazyLinear](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear) | Unsupported           |



## [Dropout Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)



| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.Dropout](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout.html#torch.nn.Dropout) | Unsupported           |
| 2    | [nn.Dropout2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout2d.html#torch.nn.Dropout2d) | Unsupported           |
| 3    | [nn.Dropout3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout3d.html#torch.nn.Dropout3d) | Unsupported           |
| 4    | [nn.AlphaDropout](https://pytorch.org/docs/1.8.1/generated/torch.nn.AlphaDropout.html#torch.nn.AlphaDropout) | Unsupported           |

## [Sparse Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.Embedding](https://pytorch.org/docs/1.8.1/generated/torch.nn.Embedding.html#torch.nn.Embedding) | Unsupported           |
| 2    | [nn.EmbeddingBag](https://pytorch.org/docs/1.8.1/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag) | Unsupported           |



## [Distance Functions](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.CosineSimilarity](https://pytorch.org/docs/1.8.1/generated/torch.nn.CosineSimilarity.html#torch.nn.CosineSimilarity) | Unsupported           |
| 2    | [nn.PairwiseDistance](https://pytorch.org/docs/1.8.1/generated/torch.nn.PairwiseDistance.html#torch.nn.PairwiseDistance) | Unsupported           |



## [Loss Functions](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.L1Loss](https://pytorch.org/docs/1.8.1/generated/torch.nn.L1Loss.html#torch.nn.L1Loss) | Unsupported           |
| 2    | [nn.MSELoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) | Unsupported           |
| 3    | [nn.CrossEntropyLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) | Unsupported           |
| 4    | [nn.CTCLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss) | Unsupported           |
| 5    | [nn.NLLLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) | Unsupported           |
| 6    | [nn.PoissonNLLLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.PoissonNLLLoss.html#torch.nn.PoissonNLLLoss) | Unsupported           |
| 7    | [nn.GaussianNLLLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.GaussianNLLLoss.html#torch.nn.GaussianNLLLoss) | Unsupported           |
| 8    | [nn.KLDivLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss) | Unsupported           |
| 9    | [nn.BCELoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.BCELoss.html#torch.nn.BCELoss) | Unsupported           |
| 10   | [nn.BCEWithLogitsLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss) | Unsupported           |
| 11   | [nn.MarginRankingLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss) | Unsupported           |
| 12   | [nn.HingeEmbeddingLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss) | Unsupported           |
| 13   | [nn.MultiLabelMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MultiLabelMarginLoss.html#torch.nn.MultiLabelMarginLoss) | Unsupported           |
| 14   | [nn.SmoothL1Loss](https://pytorch.org/docs/1.8.1/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss) | Unsupported           |
| 15   | [nn.SoftMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.SoftMarginLoss.html#torch.nn.SoftMarginLoss) | Unsupported           |
| 16   | [nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MultiLabelSoftMarginLoss.html#torch.nn.MultiLabelSoftMarginLoss) | Unsupported           |
| 17   | [nn.CosineEmbeddingLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss) | Unsupported           |
| 18   | [nn.MultiMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MultiMarginLoss.html#torch.nn.MultiMarginLoss) | Unsupported           |
| 19   | [nn.TripletMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss) | Unsupported           |
| 20   | [nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss) | Unsupported           |

## [Vision Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.PixelShuffle](https://pytorch.org/docs/1.8.1/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle) | Unsupported           |
| 2    | [nn.PixelUnshuffle](https://pytorch.org/docs/1.8.1/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle) | Unsupported           |
| 3    | [nn.Upsample](https://pytorch.org/docs/1.8.1/generated/torch.nn.Upsample.html#torch.nn.Upsample) | Unsupported           |
| 4    | [nn.UpsamplingNearest2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.UpsamplingNearest2d.html#torch.nn.UpsamplingNearest2d) | Unsupported           |
| 5    | [nn.UpsamplingBilinear2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.UpsamplingBilinear2d.html#torch.nn.UpsamplingBilinear2d) | Unsupported           |



## [Shuffle Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.ChannelShuffle](https://pytorch.org/docs/1.8.1/generated/torch.nn.ChannelShuffle.html#torch.nn.ChannelShuffle) | Unsupported           |



## [DataParallel Layers (Multi-GPU, Distributed)](https://pytorch.org/docs/1.8.1/nn.html#id1)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.DataParallel](https://pytorch.org/docs/1.8.1/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) | Unsupported           |
| 2    | [nn.parallel.DistributedDataParallel](https://pytorch.org/docs/1.8.1/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) | Unsupported           |

## [Utilities](https://pytorch.org/docs/1.8.1/nn.html#id1)



From the `torch.nn.utils` module

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [clip_grad_norm_](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_) | Unsupported           |
| 2    | [clip_grad_value_](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.clip_grad_value_.html#torch.nn.utils.clip_grad_value_) | Unsupported           |
| 3    | [parameters_to_vector](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.parameters_to_vector.html#torch.nn.utils.parameters_to_vector) | Unsupported           |
| 4    | [vector_to_parameters](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.vector_to_parameters.html#torch.nn.utils.vector_to_parameters) | Unsupported           |
| 5    | [prune.BasePruningMethod](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod) | Unsupported           |
| 6    | [prune.PruningContainer](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer) | Unsupported           |
| 7    | [prune.Identity](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity) | Unsupported           |
| 8    | [prune.RandomUnstructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured) | Unsupported           |
| 9    | [prune.L1Unstructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured) | Unsupported           |
| 10   | [prune.RandomStructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured) | Unsupported           |
| 11   | [prune.LnStructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured) | Unsupported           |
| 12   | [prune.CustomFromMask](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask) | Unsupported           |
| 13   | [prune.identity](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.identity.html#torch.nn.utils.prune.identity) | Unsupported           |
| 14   | [prune.random_unstructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.random_unstructured.html#torch.nn.utils.prune.random_unstructured) | Unsupported           |
| 15   | [prune.l1_unstructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.l1_unstructured.html#torch.nn.utils.prune.l1_unstructured) | Unsupported           |
| 16   | [prune.random_structured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.random_structured.html#torch.nn.utils.prune.random_structured) | Unsupported           |
| 17   | [prune.ln_structured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.ln_structured.html#torch.nn.utils.prune.ln_structured) | Unsupported           |
| 18   | [prune.global_unstructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.global_unstructured.html#torch.nn.utils.prune.global_unstructured) | Unsupported           |
| 19   | [prune.custom_from_mask](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.custom_from_mask.html#torch.nn.utils.prune.custom_from_mask) | Unsupported           |
| 20   | [prune.remove](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.remove.html#torch.nn.utils.prune.remove) | Unsupported           |
| 21   | [prune.is_pruned](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.is_pruned.html#torch.nn.utils.prune.is_pruned) | Unsupported           |
| 22   | [weight_norm](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.weight_norm.html#torch.nn.utils.weight_norm) | Unsupported           |
| 23   | [remove_weight_norm](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.remove_weight_norm.html#torch.nn.utils.remove_weight_norm) | Unsupported           |
| 24   | [spectral_norm](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.spectral_norm.html#torch.nn.utils.spectral_norm) | Unsupported           |
| 25   | [remove_spectral_norm](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.remove_spectral_norm.html#torch.nn.utils.remove_spectral_norm) | Unsupported           |



### Utility Functions in Other Modules

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.utils.rnn.PackedSequence](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence) | Unsupported           |
| 2    | [nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence) | Unsupported           |
| 3    | [nn.utils.rnn.pad_packed_sequence](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch.nn.utils.rnn.pad_packed_sequence) | Unsupported           |
| 4    | [nn.utils.rnn.pad_sequence](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.rnn.pad_sequence.html#torch.nn.utils.rnn.pad_sequence) | Unsupported           |
| 5    | [nn.utils.rnn.pack_sequence](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence) | Unsupported           |
| 6    | [nn.Flatten](https://pytorch.org/docs/1.8.1/generated/torch.nn.Flatten.html#torch.nn.Flatten) | Unsupported           |
| 7    | [nn.Unflatten](https://pytorch.org/docs/1.8.1/generated/torch.nn.Unflatten.html#torch.nn.Unflatten) | Unsupported           |

### Lazy Modules Initialization

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [nn.modules.lazy.LazyModuleMixin](https://pytorch.org/docs/1.8.1/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin) | Unsupported           |











# Functions(torch.nn.functional)

## [Convolution Functions](https://pytorch.org/docs/1.8.1/nn.functional.html#convolution-functions)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [conv1d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv1d) | Unsupported           |
| 2    | [conv2d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv2d) | Unsupported           |
| 3    | [conv3d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv3d) | Unsupported           |
| 4    | [conv_transpose1d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv-transpose1d) | Unsupported           |
| 5    | [conv_transpose2d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv-transpose2d) | Unsupported           |
| 6    | [conv_transpose3d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv-transpose3d) | Unsupported           |
| 7    | [unfold](https://pytorch.org/docs/1.8.1/nn.functional.html#unfold) | Unsupported           |
| 8    | [fold](https://pytorch.org/docs/1.8.1/nn.functional.html#fold) | Unsupported           |

## [Pooling Functions](https://pytorch.org/docs/1.8.1/nn.functional.html#pooling-functions)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [avg_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#avg-pool1d) | Unsupported           |
| 2    | [avg_pool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#avg-pool2d) | Unsupported           |
| 3    | [avg_pool3d](https://pytorch.org/docs/1.8.1/nn.functional.html#avg-pool3d) | Unsupported           |
| 4    | [max_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-pool1d) | Unsupported           |
| 5    | [max_pool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-pool2d) | Unsupported           |
| 6    | [max_pool3d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-pool3d) | Unsupported           |
| 7    | [max_unpool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-unpool1d) | Unsupported           |
| 8    | [max_unpool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-unpool2d) | Unsupported           |
| 9    | [max_unpool3d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-unpool3d) | Unsupported           |
| 10   | [lp_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#lp-pool1d) | Unsupported           |
| 11   | [lp_pool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#lp-pool2d) | Unsupported           |
| 12   | [adaptive_max_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-max-pool1d) | Unsupported           |
| 13   | [adaptive_max_pool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-max-pool2d) | Unsupported           |
| 14   | [adaptive_max_pool3d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-max-pool3d) | Unsupported           |
| 15   | [adaptive_avg_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-avg-pool1d) | Unsupported           |
| 16   | [adaptive_avg_pool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-avg-pool2d) | Unsupported           |
| 17   | [adaptive_avg_pool3d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-avg-pool3d) | Unsupported           |

## [Non-Linear Activation Functions](https://pytorch.org/docs/1.8.1/nn.functional.html#non-linear-activation-functions)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [threshold](https://pytorch.org/docs/1.8.1/nn.functional.html#threshold) | Unsupported           |
| 2    | [relu](https://pytorch.org/docs/1.8.1/nn.functional.html#relu) | Unsupported           |
| 3    | [hardtanh](https://pytorch.org/docs/1.8.1/nn.functional.html#hardtanh) | Unsupported           |
| 4    | [hardswish](https://pytorch.org/docs/1.8.1/nn.functional.html#hardswish) | Unsupported           |
| 5    | [relu6](https://pytorch.org/docs/1.8.1/nn.functional.html#relu6) | Unsupported           |
| 6    | [elu](https://pytorch.org/docs/1.8.1/nn.functional.html#elu) | Unsupported           |
| 7    | [selu](https://pytorch.org/docs/1.8.1/nn.functional.html#selu) | Unsupported           |
| 8    | [celu](https://pytorch.org/docs/1.8.1/nn.functional.html#celu) | Unsupported           |
| 9    | [leaky_relu](https://pytorch.org/docs/1.8.1/nn.functional.html#leaky-relu) | Unsupported           |
| 10   | [prelu](https://pytorch.org/docs/1.8.1/nn.functional.html#prelu) | Unsupported           |
| 11   | [rrelu](https://pytorch.org/docs/1.8.1/nn.functional.html#rrelu) | Unsupported           |
| 12   | [glu](https://pytorch.org/docs/1.8.1/nn.functional.html#glu) | Unsupported           |
| 13   | [gelu](https://pytorch.org/docs/1.8.1/nn.functional.html#gelu) | Unsupported           |
| 14   | [logsigmoid](https://pytorch.org/docs/1.8.1/nn.functional.html#logsigmoid) | Unsupported           |
| 15   | [hardshrink](https://pytorch.org/docs/1.8.1/nn.functional.html#hardshrink) | Unsupported           |
| 16   | [tanhshrink](https://pytorch.org/docs/1.8.1/nn.functional.html#tanhshrink) | Unsupported           |
| 17   | [softsign](https://pytorch.org/docs/1.8.1/nn.functional.html#softsign) | Unsupported           |
| 18   | [softplus](https://pytorch.org/docs/1.8.1/nn.functional.html#softplus) | Unsupported           |
| 19   | [softmin](https://pytorch.org/docs/1.8.1/nn.functional.html#softmin) | Unsupported           |
| 20   | [softmax](https://pytorch.org/docs/1.8.1/nn.functional.html#softmax) | Unsupported           |
| 21   | [softshrink](https://pytorch.org/docs/1.8.1/nn.functional.html#softshrink) | Unsupported           |
| 22   | [gumbel_softmax](https://pytorch.org/docs/1.8.1/nn.functional.html#gumbel-softmax) | Unsupported           |
| 23   | [log_softmax](https://pytorch.org/docs/1.8.1/nn.functional.html#log-softmax) | Unsupported           |
| 24   | [tanh](https://pytorch.org/docs/1.8.1/nn.functional.html#tanh) | Unsupported           |
| 25   | [sigmoid](https://pytorch.org/docs/1.8.1/nn.functional.html#sigmoid) | Unsupported           |
| 26   | [hardsigmoid](https://pytorch.org/docs/1.8.1/nn.functional.html#hardsigmoid) | Unsupported           |
| 27   | [silu](https://pytorch.org/docs/1.8.1/nn.functional.html#silu) | Unsupported           |

## [Normalization Functions](https://pytorch.org/docs/1.8.1/nn.functional.html#normalization-functions)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [batch_norm](https://pytorch.org/docs/1.8.1/nn.functional.html#batch-norm) | Unsupported           |
| 2    | [instance_norm](https://pytorch.org/docs/1.8.1/nn.functional.html#instance-norm) | Unsupported           |
| 3    | [layer_norm](https://pytorch.org/docs/1.8.1/nn.functional.html#layer-norm) | Unsupported           |
| 4    | [local_response_norm](https://pytorch.org/docs/1.8.1/nn.functional.html#local-response-norm) | Unsupported           |
| 5    | [normalize](https://pytorch.org/docs/1.8.1/nn.functional.html#normalize) | Unsupported           |

## [Linear Functions](https://pytorch.org/docs/1.8.1/nn.functional.html#linear-functions)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [linear](https://pytorch.org/docs/1.8.1/nn.functional.html#linear) | Unsupported           |
| 2    | [bilinear](https://pytorch.org/docs/1.8.1/nn.functional.html#bilinear) | Unsupported           |

## [Dropout Functions](https://pytorch.org/docs/1.8.1/nn.functional.html#dropout-functions)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [dropout](https://pytorch.org/docs/1.8.1/nn.functional.html#dropout) | Unsupported           |
| 2    | [alpha_dropout](https://pytorch.org/docs/1.8.1/nn.functional.html#alpha-dropout) | Unsupported           |
| 3    | [feature_alpha_dropout](https://pytorch.org/docs/1.8.1/nn.functional.html#feature-alpha-dropout) | Unsupported           |
| 4    | [dropout2d](https://pytorch.org/docs/1.8.1/nn.functional.html#dropout2d) | Unsupported           |
| 5    | [dropout3d](https://pytorch.org/docs/1.8.1/nn.functional.html#dropout3d) | Unsupported           |

## [Sparse Functions](https://pytorch.org/docs/1.8.1/nn.functional.html#sparse-functions)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [embedding](https://pytorch.org/docs/1.8.1/nn.functional.html#embedding) | Unsupported           |
| 2    | [embedding_bag](https://pytorch.org/docs/1.8.1/nn.functional.html#embedding-bag) | Unsupported           |
| 3    | [one_hot](https://pytorch.org/docs/1.8.1/nn.functional.html#one-hot) | Unsupported           |

## [Distance Functions](https://pytorch.org/docs/1.8.1/nn.functional.html#distance-functions)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [pairwise_distance](https://pytorch.org/docs/1.8.1/nn.functional.html#pairwise-distance) | Unsupported           |
| 2    | [cosine_similarity](https://pytorch.org/docs/1.8.1/nn.functional.html#cosine-similarity) | Unsupported           |
| 3    | [pdist](https://pytorch.org/docs/1.8.1/nn.functional.html#pdist) | Unsupported           |

## [Loss Functions](https://pytorch.org/docs/1.8.1/nn.functional.html#loss-functions)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [binary_cross_entropy](https://pytorch.org/docs/1.8.1/nn.functional.html#binary-cross-entropy) | Unsupported           |
| 2    | [binary_cross_entropy_with_logits](https://pytorch.org/docs/1.8.1/nn.functional.html#binary-cross-entropy-with-logits) | Unsupported           |
| 3    | [poisson_nll_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#poisson-nll-loss) | Unsupported           |
| 4    | [cosine_embedding_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#cosine-embedding-loss) | Unsupported           |
| 5    | [cross_entropy](https://pytorch.org/docs/1.8.1/nn.functional.html#cross-entropy) | Unsupported           |
| 6    | [ctc_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#ctc-loss) | Unsupported           |
| 7    | [hinge_embedding_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#hinge-embedding-loss) | Unsupported           |
| 8    | [kl_div](https://pytorch.org/docs/1.8.1/nn.functional.html#kl-div) | Unsupported           |
| 9    | [l1_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#l1-loss) | Unsupported           |
| 10   | [mse_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#mse-loss) | Unsupported           |
| 11   | [margin_ranking_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#margin-ranking-loss) | Unsupported           |
| 12   | [multilabel_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#multilabel-margin-loss) | Unsupported           |
| 13   | [multilabel_soft_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#multilabel-soft-margin-loss) | Unsupported           |
| 14   | [multi_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#multi-margin-loss) | Unsupported           |
| 15   | [nll_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#nll-loss) | Unsupported           |
| 16   | [smooth_l1_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#smooth-l1-loss) | Unsupported           |
| 17   | [soft_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#soft-margin-loss) | Unsupported           |
| 18   | [triplet_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#triplet-margin-loss) | Unsupported           |
| 19   | [triplet_margin_with_distance_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#triplet-margin-with-distance-loss) | Unsupported           |

## [Vision Functions](https://pytorch.org/docs/1.8.1/nn.functional.html#vision-functions)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [pixel_shuffle](https://pytorch.org/docs/1.8.1/nn.functional.html#pixel-shuffle) | Unsupported           |
| 2    | [pixel_unshuffle](https://pytorch.org/docs/1.8.1/nn.functional.html#pixel-unshuffle) | Unsupported           |
| 3    | [pad](https://pytorch.org/docs/1.8.1/nn.functional.html#pad) | Unsupported           |
| 4    | [interpolate](https://pytorch.org/docs/1.8.1/nn.functional.html#interpolate) | Unsupported           |
| 5    | [upsample](https://pytorch.org/docs/1.8.1/nn.functional.html#upsample) | Unsupported           |
| 6    | [upsample_nearest](https://pytorch.org/docs/1.8.1/nn.functional.html#upsample-nearest) | Unsupported           |
| 7    | [upsample_bilinear](https://pytorch.org/docs/1.8.1/nn.functional.html#upsample-bilinear) | Unsupported           |
| 8    | [grid_sample](https://pytorch.org/docs/1.8.1/nn.functional.html#grid-sample) | Unsupported           |
| 9    | [affine_grid](https://pytorch.org/docs/1.8.1/nn.functional.html#affine-grid) | Unsupported           |

## [Data Parallel Functions (Multi-GPU, Distributed)](https://pytorch.org/docs/1.8.1/nn.functional.html#dataparallel-functions-multi-gpu-distributed)

| No.  | API                                                          | Supported/Unsupported |
| ---- | ------------------------------------------------------------ | --------------------- |
| 1    | [data_parallel](https://pytorch.org/docs/1.8.1/nn.functional.html#data-parallel) | Unsupported           |

# [torch.distributed](https://pytorch.org/docs/1.8.1/distributed.html)

| No.  | API                                       | Supported/Unsupported |
| ---- | ----------------------------------------- | --------------------- |
| 1    | torch.distributed.is_available            | Unsupported           |
| 2    | torch.distributed.init_process_group      | Unsupported           |
| 3    | torch.distributed.Backend                 | Unsupported           |
| 4    | torch.distributed.get_backend             | Unsupported           |
| 5    | torch.distributed.get_rank                | Unsupported           |
| 6    | torch.distributed.get_world_size          | Unsupported           |
| 7    | torch.distributed.is_initialized          | Unsupported           |
| 8    | torch.distributed.is_mpi_available        | Unsupported           |
| 9    | torch.distributed.is_nccl_available       | Unsupported           |
| 10   | torch.distributed.Store                   | Unsupported           |
| 11   | torch.distributed.TCPStore                | Unsupported           |
| 12   | torch.distributed.HashStore               | Unsupported           |
| 13   | torch.distributed.FileStore               | Unsupported           |
| 14   | torch.distributed.PrefixStore             | Unsupported           |
| 15   | torch.distributed.Store.set               | Unsupported           |
| 16   | torch.distributed.Store.get               | Unsupported           |
| 17   | torch.distributed.Store.add               | Unsupported           |
| 18   | torch.distributed.Store.wait              | Unsupported           |
| 19   | torch.distributed.Store.num_keys          | Unsupported           |
| 20   | torch.distributed.Store.delete_key        | Unsupported           |
| 21   | torch.distributed.Store.set_timeout       | Unsupported           |
| 22   | torch.distributed.new_group               | Unsupported           |
| 23   | torch.distributed.send                    | Unsupported           |
| 24   | torch.distributed.recv                    | Unsupported           |
| 25   | torch.distributed.isend                   | Unsupported           |
| 26   | torch.distributed.irecv                   | Unsupported           |
| 27   | is_completed                              | Unsupported           |
| 28   | wait                                      | Unsupported           |
| 29   | torch.distributed.broadcast               | Unsupported           |
| 30   | torch.distributed.broadcast_object_list   | Unsupported           |
| 31   | torch.distributed.all_reduce              | Unsupported           |
| 32   | torch.distributed.reduce                  | Unsupported           |
| 33   | torch.distributed.all_gather              | Unsupported           |
| 34   | torch.distributed.all_gather_object       | Unsupported           |
| 35   | torch.distributed.gather                  | Unsupported           |
| 36   | torch.distributed.gather_object           | Unsupported           |
| 37   | torch.distributed.scatter                 | Unsupported           |
| 38   | torch.distributed.scatter_object_list     | Unsupported           |
| 39   | torch.distributed.reduce_scatter          | Unsupported           |
| 40   | torch.distributed.all_to_all              | Unsupported           |
| 41   | torch.distributed.barrier                 | Unsupported           |
| 42   | torch.distributed.ReduceOp                | Unsupported           |
| 43   | torch.distributed.reduce_op               | Unsupported           |
| 44   | torch.distributed.broadcast_multigpu      | Unsupported           |
| 45   | torch.distributed.all_reduce_multigpu     | Unsupported           |
| 46   | torch.distributed.reduce_multigpu         | Unsupported           |
| 47   | torch.distributed.all_gather_multigpu     | Unsupported           |
| 48   | torch.distributed.reduce_scatter_multigpu | Unsupported           |
| 49   | torch.distributed.launch                  | Unsupported           |
| 50   | torch.multiprocessing.spawn               | Unsupported           |

# torch.npu

| No.  | API                                   | NPU API                              | Supported/Unsupported |
| ---- | ------------------------------------- | :----------------------------------- | --------------------- |
| 1    | torch.cuda.current_blas_handle        | torch.npu.current_blas_handle        | Unsupported           |
| 2    | torch.cuda.current_device             | torch.npu.current_device             | Supported             |
| 3    | torch.cuda.current_stream             | torch.npu.current_stream             | Supported             |
| 4    | torch.cuda.default_stream             | torch.npu.default_stream             | Supported             |
| 5    | torch.cuda.device                     | torch.npu.device                     | Unsupported           |
| 6    | torch.cuda.device_count               | torch.npu.device_count               | Supported             |
| 7    | torch.cuda.device_of                  | torch.npu.device_of                  | Unsupported           |
| 8    | torch.cuda.get_device_capability      | torch.npu.get_device_capability      | Unsupported           |
| 9    | torch.cuda.get_device_name            | torch.npu.get_device_name            | Unsupported           |
| 10   | torch.cuda.init                       | torch.npu.init                       | Supported             |
| 11   | torch.cuda.ipc_collect                | torch.npu.ipc_collect                | Unsupported           |
| 12   | torch.cuda.is_available               | torch.npu.is_available               | Supported             |
| 13   | torch.cuda.is_initialized             | torch.npu.is_initialized             | Supported             |
| 14   | torch.cuda.set_device                 | torch.npu.set_device                 | Partially supported   |
| 15   | torch.cuda.stream                     | torch.npu.stream                     | Supported             |
| 16   | torch.cuda.synchronize                | torch.npu.synchronize                | Supported             |
| 17   | torch.cuda.get_rng_state              | torch.npu.get_rng_state              | Unsupported           |
| 18   | torch.cuda.get_rng_state_all          | torch.npu.get_rng_state_all          | Unsupported           |
| 19   | torch.cuda.set_rng_state              | torch.npu.set_rng_state              | Unsupported           |
| 20   | torch.cuda.set_rng_state_all          | torch.npu.set_rng_state_all          | Unsupported           |
| 21   | torch.cuda.manual_seed                | torch.npu.manual_seed                | Unsupported           |
| 22   | torch.cuda.manual_seed_all            | torch.npu.manual_seed_all            | Unsupported           |
| 23   | torch.cuda.seed                       | torch.npu.seed                       | Unsupported           |
| 24   | torch.cuda.seed_all                   | torch.npu.seed_all                   | Unsupported           |
| 25   | torch.cuda.initial_seed               | torch.npu.initial_seed               | Unsupported           |
| 26   | torch.cuda.comm.broadcast             | torch.npu.comm.broadcast             | Unsupported           |
| 27   | torch.cuda.comm.broadcast_coalesced   | torch.npu.comm.broadcast_coalesced   | Unsupported           |
| 28   | torch.cuda.comm.reduce_add            | torch.npu.comm.reduce_add            | Unsupported           |
| 29   | torch.cuda.comm.scatter               | torch.npu.comm.scatter               | Unsupported           |
| 30   | torch.cuda.comm.gather                | torch.npu.comm.gather                | Unsupported           |
| 31   | torch.cuda.Stream                     | torch.npu.Stream                     | Supported             |
| 32   | torch.cuda.Stream.query               | torch.npu.Stream.query               | Unsupported           |
| 33   | torch.cuda.Stream.record_event        | torch.npu.Stream.record_event        | Supported             |
| 34   | torch.cuda.Stream.synchronize         | torch.npu.Stream.synchronize         | Supported             |
| 35   | torch.cuda.Stream.wait_event          | torch.npu.Stream.wait_event          | Supported             |
| 36   | torch.cuda.Stream.wait_stream         | torch.npu.Stream.wait_stream         | Supported             |
| 37   | torch.cuda.Event                      | torch.npu.Event                      | Supported             |
| 38   | torch.cuda.Event.elapsed_time         | torch.npu.Event.elapsed_time         | Supported             |
| 39   | torch.cuda.Event.from_ipc_handle      | torch.npu.Event.from_ipc_handle      | Unsupported           |
| 40   | torch.cuda.Event.ipc_handle           | torch.npu.Event.ipc_handle           | Unsupported           |
| 41   | torch.cuda.Event.query                | torch.npu.Event.query                | Supported             |
| 42   | torch.cuda.Event.record               | torch.npu.Event.record               | Supported             |
| 43   | torch.cuda.Event.synchronize          | torch.npu.Event.synchronize          | Supported             |
| 44   | torch.cuda.Event.wait                 | torch.npu.Event.wait                 | Supported             |
| 45   | torch.cuda.empty_cache                | torch.npu.empty_cache                | Supported             |
| 46   | torch.cuda.memory_stats               | torch.npu.memory_stats               | Supported             |
| 47   | torch.cuda.memory_summary             | torch.npu.memory_summary             | Supported             |
| 48   | torch.cuda.memory_snapshot            | torch.npu.memory_snapshot            | Supported             |
| 49   | torch.cuda.memory_allocated           | torch.npu.memory_allocated           | Supported             |
| 50   | torch.cuda.max_memory_allocated       | torch.npu.max_memory_allocated       | Supported             |
| 51   | torch.cuda.reset_max_memory_allocated | torch.npu.reset_max_memory_allocated | Supported             |
| 52   | torch.cuda.memory_reserved            | torch.npu.memory_reserved            | Supported             |
| 53   | torch.cuda.max_memory_reserved        | torch.npu.max_memory_reserved        | Supported             |
| 54   | torch.cuda.memory_cached              | torch.npu.memory_cached              | Supported             |
| 55   | torch.cuda.max_memory_cached          | torch.npu.max_memory_cached          | Supported             |
| 56   | torch.cuda.reset_max_memory_cached    | torch.npu.reset_max_memory_cached    | Supported             |
| 57   | torch.cuda.nvtx.mark                  | torch.npu.nvtx.mark                  | Unsupported           |
| 58   | torch.cuda.nvtx.range_push            | torch.npu.nvtx.range_push            | Unsupported           |
| 59   | torch.cuda.nvtx.range_pop             | torch.npu.nvtx.range_pop             | Unsupported           |
| 60   | torch.cuda._sleep                     | torch.npu._sleep                     | Unsupported           |
| 61   | torch.cuda.Stream.priority_range      | torch.npu.Stream.priority_range      | Unsupported           |
| 62   | torch.cuda.get_device_properties      | torch.npu.get_device_properties      | Unsupported           |
| 63   | torch.cuda.amp.GradScaler             | torch.npu.amp.GradScaler             | Unsupported           |

# NPU Custom Operators

| No.  | Operator                                       |
| ---- | ---------------------------------------------- |
| 1    | npu_convolution_transpose                      |
| 2    | npu_conv_transpose2d                           |
| 3    | npu_convolution_transpose_backward             |
| 4    | npu_conv_transpose2d_backward                  |
| 5    | npu_conv_transpose3d_backward                  |
| 6    | npu_convolution                                |
| 7    | npu_convolution_backward                       |
| 8    | npu_convolution_double_backward                |
| 9    | npu_conv2d                                     |
| 10   | npu_conv2d.out                                 |
| 11   | npu_conv2d_backward                            |
| 12   | npu_conv3d                                     |
| 13   | npu_conv3d.out                                 |
| 14   | npu_conv3d_backward                            |
| 15   | one_                                           |
| 16   | npu_sort_v2.out                                |
| 17   | npu_sort_v2                                    |
| 18   | npu_format_cast                                |
| 19   | npu_format_cast_.acl_format                    |
| 20   | npu_format_cast_.src                           |
| 21   | npu_transpose_to_contiguous                    |
| 22   | npu_transpose                                  |
| 23   | npu_transpose.out                              |
| 24   | npu_broadcast                                  |
| 25   | npu_broadcast.out                              |
| 26   | npu_dtype_cast                                 |
| 27   | npu_dtype_cast_.Tensor                         |
| 28   | npu_roi_alignbk                                |
| 29   | empty_with_format                              |
| 30   | empty_with_format.names                        |
| 31   | copy_memory_                                   |
| 32   | npu_one_hot                                    |
| 33   | npu_stride_add                                 |
| 34   | npu_softmax_cross_entropy_with_logits          |
| 35   | npu_softmax_cross_entropy_with_logits_backward |
| 36   | npu_ps_roi_pooling                             |
| 37   | npu_ps_roi_pooling_backward                    |
| 38   | npu_roi_align                                  |
| 39   | npu_nms_v4                                     |
| 40   | npu_lstm                                       |
| 41   | npu_lstm_backward                              |
| 42   | npu_iou                                        |
| 43   | npu_ptiou                                      |
| 44   | npu_nms_with_mask                              |
| 45   | npu_pad                                        |
| 46   | npu_bounding_box_encode                        |
| 47   | npu_bounding_box_decode                        |
| 48   | npu_gru                                        |
| 49   | npu_gru_backward                               |
| 50   | npu_set_.source_Storage_storage_offset_format  |
| 51   | npu_random_choice_with_mask                    |
| 52   | npu_batch_nms                                  |
| 53   | npu_slice                                      |
| 54   | npu_slice.out                                  |
| 55   | npu_dropoutV2                                  |
| 56   | npu_dropoutV2_backward                         |
| 57   | _npu_dropout                                   |
| 58   | _npu_dropout_inplace                           |
| 59   | npu_dropout_backward                           |
| 60   | npu_indexing                                   |
| 61   | npu_indexing.out                               |
| 62   | npu_ifmr                                       |
| 63   | npu_max.dim                                    |
| 64   | npu_max.names_dim                              |
| 65   | npu_scatter                                    |
| 66   | npu_max_backward                               |
| 67   | npu_apply_adam                                 |
| 68   | npu_layer_norm_eval                            |
| 69   | npu_alloc_float_status                         |
| 70   | npu_get_float_status                           |
| 71   | npu_clear_float_status                         |
| 72   | npu_confusion_transpose                        |
| 73   | npu_confusion_transpose_backward               |
| 74   | npu_bmmV2                                      |
| 75   | fast_gelu                                      |
| 76   | fast_gelu_backward                             |
| 77   | npu_sub_sample                                 |
| 78   | npu_deformable_conv2d                          |
| 79   | npu_deformable_conv2dbk                        |
| 80   | npu_mish                                       |
| 81   | npu_anchor_response_flags                      |
| 82   | npu_yolo_boxes_encode                          |
| 83   | npu_grid_assign_positive                       |
| 84   | npu_mish_backward                              |
| 85   | npu_normalize_batch                            |
| 86   | npu_masked_fill_range                          |
| 87   | npu_linear                                     |
| 88   | npu_linear_backward                            |
| 89   | npu_bert_apply_adam                            |
| 90   | npu_giou                                       |
| 91   | npu_giou_backward                              |

Operator descriptions:

> ```
> npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out = (var, m, v))
> ```

count adam result.

- Parameters：
  - **beta1_power** (Number) - power of beta1.
  - **beta2_power** (Number) - power of beta2.
  - **lr** (Number) -  learning rate.
  - **beta1** (Number) - exponential decay rate for the 1st moment estimates.
  - **beta2** (Number) - exponential decay rate for the 2nd moment estimates.
  - **epsilon** (Number) - term added to the denominator to improve numerical stability.
  - **grad** (Tensor) - the gradient.
  - **use_locking** (bool) - If `True` use locks for update operations.
  - **use_nesterov** (bool) -If `True`, uses the nesterov update.
  - **var** (Tensor) - variables to be optimized.
  - **m** (Tensor) - mean value of variables.
  - **v** (Tensor) - variance of variables.

- constraints：

  None

- Examples：

  None 

> npu_bert_apply_adam(var, m, v, lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, )

count adam result in bert.

- Parameters：
  - **lr** (Number) - learning rate.
  - **beta1** (Number) - exponential decay rate for the 1st moment estimates.
  - **beta2** (Number) - exponential decay rate for the 2nd moment estimates.
  - **epsilon** (Number) - term added to the denominator to improve numerical stability.
  - **grad** (Tensor) - the gradient.
  - **max_grad_norm** (Number) - maximum norm for the gradients.
  - **global_grad_norm** (Number) - L2_norm for the gradients.
  - **weight_decay** (Number) - weight decay
  - **var** (Tensor) - variables to be optimized.
  - **m** (Tensor) -mean value of variables.
  - **v** (Tensor) - variance of variables.

- constraints：

  None

- Examples：

  ```python
  >>> var_in = torch.rand(321538).uniform_(-32.,21.).npu()
  >>> var_in
  tensor([  0.6119,   5.8193,   3.0683,  ..., -28.5832,  12.9402, -24.0488],
         device='npu:0')
  >>> m_in = torch.zeros(321538).npu()
  >>> v_in = torchzeros(321538).npu()
  >>> grad = torch.rand(321538).uniform_(-0.05,0.03).npu()
  >>> grad
  tensor([-0.0315, -0.0113, -0.0132,  ...,  0.0106, -0.0226, -0.0252],
         device='npu:0')
  >>> max_grad_norm = -1.
  >>> beta1 = 0.9
  >>> beta2 = 0.99
  >>> weight_decay = 0.
  >>> lr = 0.1
  >>> epsilon = 1e-06
  >>> global_grad_norm = 0.
  >>> var_out, m_out, v_out = torch.npu_bert_apply_adam(var_in, m_in, v_in, lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay)
  >>> var_out
  tensor([  0.7118,   5.9192,   3.1682,  ..., -28.6831,  13.0402, -23.9489],
         device='npu:0')
  >>> m_out
  tensor([-0.0032, -0.0011, -0.0013,  ...,  0.0011, -0.0023, -0.0025],
         device='npu:0')
  >>> v_out
  tensor([9.9431e-06, 1.2659e-06, 1.7328e-06,  ..., 1.1206e-06, 5.0933e-06,
          6.3495e-06], device='npu:0')
  ```
  
  