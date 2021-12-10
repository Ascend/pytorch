# Torch

## Tensors

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [is_tensor](https://pytorch.org/docs/1.8.1/generated/torch.is_tensor.html) | 否       |
| 2    | [is_storage](https://pytorch.org/docs/1.8.1/generated/torch.is_storage.html) | 否       |
| 3    | [is_complex](https://pytorch.org/docs/1.8.1/generated/torch.is_complex.html) | 否       |
| 4    | [is_floating_point](https://pytorch.org/docs/1.8.1/generated/torch.is_floating_point.html) | 否       |
| 5    | [is_nonzero](https://pytorch.org/docs/1.8.1/generated/torch.is_nonzero.html) | 否       |
| 6    | [set_default_dtype](https://pytorch.org/docs/1.8.1/generated/torch.set_default_dtype.html) | 否       |
| 7    | [get_default_dtype](https://pytorch.org/docs/1.8.1/generated/torch.get_default_dtype.html) | 否       |
| 8    | [set_default_tensor_type](https://pytorch.org/docs/1.8.1/generated/torch.set_default_tensor_type.html) | 否       |
| 9    | [numel](https://pytorch.org/docs/1.8.1/generated/torch.numel.html) | 否       |
| 10   | [set_printoptions](https://pytorch.org/docs/1.8.1/generated/torch.set_printoptions.html) | 否       |
| 11   | [set_flush_denormal](https://pytorch.org/docs/1.8.1/generated/torch.set_flush_denormal.html) | 否       |

### Creation Ops

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [tensor](https://pytorch.org/docs/1.8.1/generated/torch.tensor.html) | 否       |
| 2    | [sparse_coo_tensor](https://pytorch.org/docs/1.8.1/generated/torch.sparse_coo_tensor.html) | 否       |
| 3    | [as_tensor](https://pytorch.org/docs/1.8.1/generated/torch.as_tensor.html) | 否       |
| 4    | [as_strided](https://pytorch.org/docs/1.8.1/generated/torch.as_strided.html) | 否       |
| 5    | [from_numpy](https://pytorch.org/docs/1.8.1/generated/torch.from_numpy.html) | 否       |
| 6    | [zeros](https://pytorch.org/docs/1.8.1/generated/torch.zeros.html) | 否       |
| 7    | [zeros_like](https://pytorch.org/docs/1.8.1/generated/torch.zeros_like.html) | 否       |
| 8    | [ones](https://pytorch.org/docs/1.8.1/generated/torch.ones.html) | 否       |
| 9    | [ones_like](https://pytorch.org/docs/1.8.1/generated/torch.ones_like.html) | 否       |
| 10   | [arange](https://pytorch.org/docs/1.8.1/generated/torch.arange.html) | 否       |
| 11   | [range](https://pytorch.org/docs/1.8.1/generated/torch.range.html) | 否       |
| 12   | [linspace](https://pytorch.org/docs/1.8.1/generated/torch.linspace.html) | 否       |
| 13   | [logspace](https://pytorch.org/docs/1.8.1/generated/torch.logspace.html) | 否       |
| 14   | [eye](https://pytorch.org/docs/1.8.1/generated/torch.eye.html) | 否       |
| 15   | [empty](https://pytorch.org/docs/1.8.1/generated/torch.empty.html) | 否       |
| 16   | [empty_like](https://pytorch.org/docs/1.8.1/generated/torch.empty_like.html) | 否       |
| 17   | [empty_strided](https://pytorch.org/docs/1.8.1/generated/torch.empty_strided.html) | 否       |
| 18   | [full](https://pytorch.org/docs/1.8.1/generated/torch.full.html) | 否       |
| 19   | [full_like](https://pytorch.org/docs/1.8.1/generated/torch.full_like.html) | 否       |
| 20   | [quantize_per_tensor](https://pytorch.org/docs/1.8.1/generated/torch.quantize_per_tensor.html) | 否       |
| 21   | [quantize_per_channel](https://pytorch.org/docs/1.8.1/generated/torch.quantize_per_channel.html) | 否       |
| 22   | [dequantize](https://pytorch.org/docs/1.8.1/generated/torch.dequantize.html) | 否       |
| 23   | [complex](https://pytorch.org/docs/1.8.1/generated/torch.complex.html) | 否       |
| 24   | [polar](https://pytorch.org/docs/1.8.1/generated/torch.polar.html) | 否       |
| 25   | [heaviside](https://pytorch.org/docs/1.8.1/generated/torch.heaviside.html) | 否       |

### Indexing, Slicing, Joining, Mutating Ops

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [cat](https://pytorch.org/docs/1.8.1/generated/torch.cat.html) | 否       |
| 2    | [chunk](https://pytorch.org/docs/1.8.1/generated/torch.chunk.html) | 否       |
| 3    | [column_stack](https://pytorch.org/docs/1.8.1/generated/torch.column_stack.html) | 否       |
| 4    | [dstack](https://pytorch.org/docs/1.8.1/generated/torch.dstack.html) | 否       |
| 5    | [gather](https://pytorch.org/docs/1.8.1/generated/torch.gather.html) | 否       |
| 6    | [hstack](https://pytorch.org/docs/1.8.1/generated/torch.hstack.html) | 否       |
| 7    | [index_select](https://pytorch.org/docs/1.8.1/generated/torch.index_select.html) | 否       |
| 8    | [masked_select](https://pytorch.org/docs/1.8.1/generated/torch.masked_select.html) | 否       |
| 9    | [movedim](https://pytorch.org/docs/1.8.1/generated/torch.movedim.html) | 否       |
| 10   | [moveaxis](https://pytorch.org/docs/1.8.1/generated/torch.moveaxis.html) | 否       |
| 11   | [narrow](https://pytorch.org/docs/1.8.1/generated/torch.narrow.html) | 否       |
| 12   | [nonzero](https://pytorch.org/docs/1.8.1/generated/torch.nonzero.html) | 否       |
| 13   | [reshape](https://pytorch.org/docs/1.8.1/generated/torch.reshape.html) | 否       |
| 14   | [row_stack](https://pytorch.org/docs/1.8.1/generated/torch.row_stack.html) | 否       |
| 15   | [scatter](https://pytorch.org/docs/1.8.1/generated/torch.scatter.html) | 否       |
| 16   | [scatter_add](https://pytorch.org/docs/1.8.1/generated/torch.scatter_add.html) | 否       |
| 17   | [split](https://pytorch.org/docs/1.8.1/generated/torch.split.html) | 否       |
| 18   | [squeeze](https://pytorch.org/docs/1.8.1/generated/torch.squeeze.html) | 否       |
| 19   | [stack](https://pytorch.org/docs/1.8.1/generated/torch.stack.html) | 否       |
| 20   | [swapaxes](https://pytorch.org/docs/1.8.1/generated/torch.swapaxes.html) | 否       |
| 21   | [swapdims](https://pytorch.org/docs/1.8.1/generated/torch.swapdims.html) | 否       |
| 22   | [t](https://pytorch.org/docs/1.8.1/generated/torch.t.html)   | 否       |
| 23   | [take](https://pytorch.org/docs/1.8.1/generated/torch.take.html) | 否       |
| 24   | [tensor_split](https://pytorch.org/docs/1.8.1/generated/torch.tensor_split.html) | 否       |
| 25   | [tile](https://pytorch.org/docs/1.8.1/generated/torch.tile.html) | 否       |
| 26   | [transpose](https://pytorch.org/docs/1.8.1/generated/torch.transpose.html) | 否       |
| 27   | [unbind](https://pytorch.org/docs/1.8.1/generated/torch.unbind.html) | 否       |
| 28   | [unsqueeze](https://pytorch.org/docs/1.8.1/generated/torch.unsqueeze.html) | 否       |
| 29   | [vstack](https://pytorch.org/docs/1.8.1/generated/torch.vstack.html) | 否       |
| 30   | [where](https://pytorch.org/docs/1.8.1/generated/torch.where.html) | 否       |

## Generators

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [Generator](https://pytorch.org/docs/1.8.1/generated/torch.Generator.html) | 否       |

## Random sampling

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [seed](https://pytorch.org/docs/1.8.1/generated/torch.seed.html) | 否       |
| 2    | [manual_seed](https://pytorch.org/docs/1.8.1/generated/torch.manual_seed.html) | 否       |
| 3    | [initial_seed](https://pytorch.org/docs/1.8.1/generated/torch.initial_seed.html) | 否       |
| 4    | [get_rng_state](https://pytorch.org/docs/1.8.1/generated/torch.get_rng_state.html) | 否       |
| 5    | [set_rng_state](https://pytorch.org/docs/1.8.1/generated/torch.set_rng_state.html) | 否       |
| 6    | [bernoulli](https://pytorch.org/docs/1.8.1/generated/torch.bernoulli.html) | 否       |
| 7    | [multinomial](https://pytorch.org/docs/1.8.1/generated/torch.multinomial.html) | 否       |
| 8    | [normal](https://pytorch.org/docs/1.8.1/generated/torch.normal.html) | 否       |
| 9    | [poisson](https://pytorch.org/docs/1.8.1/generated/torch.poisson.html) | 否       |
| 10   | [rand](https://pytorch.org/docs/1.8.1/generated/torch.rand.html) | 否       |
| 11   | [rand_like](https://pytorch.org/docs/1.8.1/generated/torch.rand_like.html) | 否       |
| 12   | [randint](https://pytorch.org/docs/1.8.1/generated/torch.randint.html) | 否       |
| 13   | [randint_like](https://pytorch.org/docs/1.8.1/generated/torch.randint_like.html) | 否       |
| 14   | [randn](https://pytorch.org/docs/1.8.1/generated/torch.randn.html) | 否       |
| 15   | [randn_like](https://pytorch.org/docs/1.8.1/generated/torch.randn_like.html) | 否       |
| 16   | [randperm](https://pytorch.org/docs/1.8.1/generated/torch.randperm.html) | 否       |

### In-place random sampling

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [torch.Tensor.bernoulli_()](https://pytorch.org/docs/1.8.1/tensors.html) | 否       |
| 2    | [torch.Tensor.cauchy_()](https://pytorch.org/docs/1.8.1/tensors.html) | 否       |
| 3    | [torch.Tensor.exponential_()](https://pytorch.org/docs/1.8.1/tensors.html) | 否       |
| 4    | [torch.Tensor.geometric_()](https://pytorch.org/docs/1.8.1/tensors.html) | 否       |
| 5    | [torch.Tensor.log_normal_()](https://pytorch.org/docs/1.8.1/tensors.html) | 否       |
| 6    | [torch.Tensor.normal_()](https://pytorch.org/docs/1.8.1/tensors.html) | 否       |
| 7    | [torch.Tensor.random_()](https://pytorch.org/docs/1.8.1/tensors.html) | 否       |
| 8    | [torch.Tensor.uniform_()](https://pytorch.org/docs/1.8.1/tensors.html) | 否       |

### Quasi-random sampling

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [quasirandom.SobolEngine](https://pytorch.org/docs/1.8.1/generated/torch.quasirandom.SobolEngine.html) | 否       |

## Serialization

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [save](https://pytorch.org/docs/1.8.1/generated/torch.save.html) | 否       |
| 2    | [load](https://pytorch.org/docs/1.8.1/generated/torch.load.html) | 否       |

## Parallelism

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [get_num_threads](https://pytorch.org/docs/1.8.1/generated/torch.get_num_threads.html) | 否       |
| 2    | [set_num_threads](https://pytorch.org/docs/1.8.1/generated/torch.set_num_threads.html) | 否       |
| 3    | [get_num_interop_threads](https://pytorch.org/docs/1.8.1/generated/torch.get_num_interop_threads.html) | 否       |
| 4    | [set_num_interop_threads](https://pytorch.org/docs/1.8.1/generated/torch.set_num_interop_threads.html) | 否       |

## Locally disabling gradient computation

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [no_grad](https://pytorch.org/docs/1.8.1/generated/torch.no_grad.html#torch.no_grad) | 否       |
| 2    | [enable_grad](https://pytorch.org/docs/1.8.1/generated/torch.enable_grad.html#torch.enable_grad) | 否       |
| 3    | set_grad_enabled                                             | 否       |

## Math operations

### Pointwise Ops

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [abs](https://pytorch.org/docs/1.8.1/generated/torch.abs.html#torch.abs) | 否       |
| 2    | [absolute](https://pytorch.org/docs/1.8.1/generated/torch.absolute.html#torch.absolute) | 否       |
| 3    | [acos](https://pytorch.org/docs/1.8.1/generated/torch.acos.html#torch.acos) | 否       |
| 4    | [arccos](https://pytorch.org/docs/1.8.1/generated/torch.arccos.html#torch.arccos) | 否       |
| 5    | [acosh](https://pytorch.org/docs/1.8.1/generated/torch.acosh.html#torch.acosh) | 否       |
| 6    | [arccosh](https://pytorch.org/docs/1.8.1/generated/torch.arccosh.html#torch.arccosh) | 否       |
| 7    | [add](https://pytorch.org/docs/1.8.1/generated/torch.add.html#torch.add) | 否       |
| 8    | [addcdiv](https://pytorch.org/docs/1.8.1/generated/torch.addcdiv.html#torch.addcdiv) | 否       |
| 9    | [addcmul](https://pytorch.org/docs/1.8.1/generated/torch.addcmul.html#torch.addcmul) | 否       |
| 10   | [angle](https://pytorch.org/docs/1.8.1/generated/torch.angle.html#torch.angle) | 否       |
| 11   | [asin](https://pytorch.org/docs/1.8.1/generated/torch.asin.html#torch.asin) | 否       |
| 12   | [arcsin](https://pytorch.org/docs/1.8.1/generated/torch.arcsin.html#torch.arcsin) | 否       |
| 13   | [asinh](https://pytorch.org/docs/1.8.1/generated/torch.asinh.html#torch.asinh) | 否       |
| 14   | [arcsinh](https://pytorch.org/docs/1.8.1/generated/torch.arcsinh.html#torch.arcsinh) | 否       |
| 15   | [atan](https://pytorch.org/docs/1.8.1/generated/torch.atan.html#torch.atan) | 否       |
| 16   | [arctan](https://pytorch.org/docs/1.8.1/generated/torch.arctan.html#torch.arctan) | 否       |
| 17   | [atanh](https://pytorch.org/docs/1.8.1/generated/torch.atanh.html#torch.atanh) | 否       |
| 18   | [arctanh](https://pytorch.org/docs/1.8.1/generated/torch.arctanh.html#torch.arctanh) | 否       |
| 19   | [atan2](https://pytorch.org/docs/1.8.1/generated/torch.atan2.html#torch.atan2) | 否       |
| 20   | [bitwise_not](https://pytorch.org/docs/1.8.1/generated/torch.bitwise_not.html#torch.bitwise_not) | 否       |
| 21   | [bitwise_and](https://pytorch.org/docs/1.8.1/generated/torch.bitwise_and.html#torch.bitwise_and) | 否       |
| 22   | [bitwise_or](https://pytorch.org/docs/1.8.1/generated/torch.bitwise_or.html#torch.bitwise_or) | 否       |
| 23   | [bitwise_xor](https://pytorch.org/docs/1.8.1/generated/torch.bitwise_xor.html#torch.bitwise_xor) | 否       |
| 24   | [ceil](https://pytorch.org/docs/1.8.1/generated/torch.ceil.html#torch.ceil) | 否       |
| 25   | [clamp](https://pytorch.org/docs/1.8.1/generated/torch.clamp.html#torch.clamp) | 否       |
| 26   | [clip](https://pytorch.org/docs/1.8.1/generated/torch.clip.html#torch.clip) | 否       |
| 27   | [conj](https://pytorch.org/docs/1.8.1/generated/torch.conj.html#torch.conj) | 否       |
| 28   | [copysign](https://pytorch.org/docs/1.8.1/generated/torch.copysign.html#torch.copysign) | 否       |
| 29   | [cos](https://pytorch.org/docs/1.8.1/generated/torch.cos.html#torch.cos) | 否       |
| 30   | [cosh](https://pytorch.org/docs/1.8.1/generated/torch.cosh.html#torch.cosh) | 否       |
| 31   | [deg2rad](https://pytorch.org/docs/1.8.1/generated/torch.deg2rad.html#torch.deg2rad) | 否       |
| 32   | [div](https://pytorch.org/docs/1.8.1/generated/torch.div.html#torch.div) | 否       |
| 33   | [divide](https://pytorch.org/docs/1.8.1/generated/torch.divide.html#torch.divide) | 否       |
| 34   | [digamma](https://pytorch.org/docs/1.8.1/generated/torch.digamma.html#torch.digamma) | 否       |
| 35   | [erf](https://pytorch.org/docs/1.8.1/generated/torch.erf.html#torch.erf) | 否       |
| 36   | [erfc](https://pytorch.org/docs/1.8.1/generated/torch.erfc.html#torch.erfc) | 否       |
| 37   | [erfinv](https://pytorch.org/docs/1.8.1/generated/torch.erfinv.html#torch.erfinv) | 否       |
| 38   | [exp](https://pytorch.org/docs/1.8.1/generated/torch.exp.html#torch.exp) | 否       |
| 39   | [exp2](https://pytorch.org/docs/1.8.1/generated/torch.exp2.html#torch.exp2) | 否       |
| 40   | [expm1](https://pytorch.org/docs/1.8.1/generated/torch.expm1.html#torch.expm1) | 否       |
| 41   | [fake_quantize_per_channel_affine](https://pytorch.org/docs/1.8.1/generated/torch.fake_quantize_per_channel_affine.html#torch.fake_quantize_per_channel_affine) | 否       |
| 42   | [fake_quantize_per_tensor_affine](https://pytorch.org/docs/1.8.1/generated/torch.fake_quantize_per_tensor_affine.html#torch.fake_quantize_per_tensor_affine) | 否       |
| 43   | [fix](https://pytorch.org/docs/1.8.1/generated/torch.fix.html#torch.fix) | 否       |
| 44   | [float_power](https://pytorch.org/docs/1.8.1/generated/torch.float_power.html#torch.float_power) | 否       |
| 45   | [floor](https://pytorch.org/docs/1.8.1/generated/torch.floor.html#torch.floor) | 否       |
| 46   | [floor_divide](https://pytorch.org/docs/1.8.1/generated/torch.floor_divide.html#torch.floor_divide) | 否       |
| 47   | [fmod](https://pytorch.org/docs/1.8.1/generated/torch.fmod.html#torch.fmod) | 否       |
| 48   | [frac](https://pytorch.org/docs/1.8.1/generated/torch.frac.html#torch.frac) | 否       |
| 49   | [imag](https://pytorch.org/docs/1.8.1/generated/torch.imag.html#torch.imag) | 否       |
| 50   | [ldexp](https://pytorch.org/docs/1.8.1/generated/torch.ldexp.html#torch.ldexp) | 否       |
| 51   | [lerp](https://pytorch.org/docs/1.8.1/generated/torch.lerp.html#torch.lerp) | 否       |
| 52   | [lgamma](https://pytorch.org/docs/1.8.1/generated/torch.lgamma.html#torch.lgamma) | 否       |
| 53   | [log](https://pytorch.org/docs/1.8.1/generated/torch.log.html#torch.log) | 否       |
| 54   | [log10](https://pytorch.org/docs/1.8.1/generated/torch.log10.html#torch.log10) | 否       |
| 55   | [log1p](https://pytorch.org/docs/1.8.1/generated/torch.log1p.html#torch.log1p) | 否       |
| 56   | [log2](https://pytorch.org/docs/1.8.1/generated/torch.log2.html#torch.log2) | 否       |
| 57   | [logaddexp](https://pytorch.org/docs/1.8.1/generated/torch.logaddexp.html#torch.logaddexp) | 否       |
| 58   | [logaddexp2](https://pytorch.org/docs/1.8.1/generated/torch.logaddexp2.html#torch.logaddexp2) | 否       |
| 59   | [logical_and](https://pytorch.org/docs/1.8.1/generated/torch.logical_and.html#torch.logical_and) | 否       |
| 60   | [logical_not](https://pytorch.org/docs/1.8.1/generated/torch.logical_not.html#torch.logical_not) | 否       |
| 61   | [logical_or](https://pytorch.org/docs/1.8.1/generated/torch.logical_or.html#torch.logical_or) | 否       |
| 62   | [logical_xor](https://pytorch.org/docs/1.8.1/generated/torch.logical_xor.html#torch.logical_xor) | 否       |
| 63   | [logit](https://pytorch.org/docs/1.8.1/generated/torch.logit.html#torch.logit) | 否       |
| 64   | [hypot](https://pytorch.org/docs/1.8.1/generated/torch.hypot.html#torch.hypot) | 否       |
| 65   | [i0](https://pytorch.org/docs/1.8.1/generated/torch.i0.html#torch.i0) | 否       |
| 66   | [igamma](https://pytorch.org/docs/1.8.1/generated/torch.igamma.html#torch.igamma) | 否       |
| 67   | [igammac](https://pytorch.org/docs/1.8.1/generated/torch.igammac.html#torch.igammac) | 否       |
| 68   | [mul](https://pytorch.org/docs/1.8.1/generated/torch.mul.html#torch.mul) | 否       |
| 69   | [multiply](https://pytorch.org/docs/1.8.1/generated/torch.multiply.html#torch.multiply) | 否       |
| 70   | [mvlgamma](https://pytorch.org/docs/1.8.1/generated/torch.mvlgamma.html#torch.mvlgamma) | 否       |
| 71   | [nan_to_num](https://pytorch.org/docs/1.8.1/generated/torch.nan_to_num.html#torch.nan_to_num) | 否       |
| 72   | [neg](https://pytorch.org/docs/1.8.1/generated/torch.neg.html#torch.neg) | 否       |
| 73   | [negative](https://pytorch.org/docs/1.8.1/generated/torch.negative.html#torch.negative) | 否       |
| 74   | [nextafter](https://pytorch.org/docs/1.8.1/generated/torch.nextafter.html#torch.nextafter) | 否       |
| 75   | [polygamma](https://pytorch.org/docs/1.8.1/generated/torch.polygamma.html#torch.polygamma) | 否       |
| 76   | [pow](https://pytorch.org/docs/1.8.1/generated/torch.pow.html#torch.pow) | 否       |
| 77   | [rad2deg](https://pytorch.org/docs/1.8.1/generated/torch.rad2deg.html#torch.rad2deg) | 否       |
| 78   | [real](https://pytorch.org/docs/1.8.1/generated/torch.real.html#torch.real) | 否       |
| 79   | [reciprocal](https://pytorch.org/docs/1.8.1/generated/torch.reciprocal.html#torch.reciprocal) | 否       |
| 80   | [remainder](https://pytorch.org/docs/1.8.1/generated/torch.remainder.html#torch.remainder) | 否       |
| 81   | [round](https://pytorch.org/docs/1.8.1/generated/torch.round.html#torch.round) | 否       |
| 82   | [rsqrt](https://pytorch.org/docs/1.8.1/generated/torch.rsqrt.html#torch.rsqrt) | 否       |
| 83   | [sigmoid](https://pytorch.org/docs/1.8.1/generated/torch.sigmoid.html#torch.sigmoid) | 否       |
| 84   | [sign](https://pytorch.org/docs/1.8.1/generated/torch.sign.html#torch.sign) | 否       |
| 85   | [sgn](https://pytorch.org/docs/1.8.1/generated/torch.sgn.html#torch.sgn) | 否       |
| 86   | [signbit](https://pytorch.org/docs/1.8.1/generated/torch.signbit.html#torch.signbit) | 否       |
| 87   | [sin](https://pytorch.org/docs/1.8.1/generated/torch.sin.html#torch.sin) | 否       |
| 88   | [sinc](https://pytorch.org/docs/1.8.1/generated/torch.sinc.html#torch.sinc) | 否       |
| 89   | [sinh](https://pytorch.org/docs/1.8.1/generated/torch.sinh.html#torch.sinh) | 否       |
| 90   | [sqrt](https://pytorch.org/docs/1.8.1/generated/torch.sqrt.html#torch.sqrt) | 否       |
| 91   | [square](https://pytorch.org/docs/1.8.1/generated/torch.square.html#torch.square) | 否       |
| 92   | [sub](https://pytorch.org/docs/1.8.1/generated/torch.sub.html#torch.sub) | 否       |
| 93   | [subtract](https://pytorch.org/docs/1.8.1/generated/torch.subtract.html#torch.subtract) | 否       |
| 94   | [tan](https://pytorch.org/docs/1.8.1/generated/torch.tan.html#torch.tan) | 否       |
| 95   | [tanh](https://pytorch.org/docs/1.8.1/generated/torch.tanh.html#torch.tanh) | 否       |
| 96   | [true_divide](https://pytorch.org/docs/1.8.1/generated/torch.true_divide.html#torch.true_divide) | 否       |
| 97   | [trunc](https://pytorch.org/docs/1.8.1/generated/torch.trunc.html#torch.trunc) | 否       |
| 98   | [xlogy](https://pytorch.org/docs/1.8.1/generated/torch.xlogy.html#torch.xlogy) | 否       |

### Reduction Ops

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [argmax](https://pytorch.org/docs/1.8.1/generated/torch.argmax.html#torch.argmax) | 否       |
| 2    | [argmin](https://pytorch.org/docs/1.8.1/generated/torch.argmin.html#torch.argmin) | 否       |
| 3    | [amax](https://pytorch.org/docs/1.8.1/generated/torch.amax.html#torch.amax) | 否       |
| 4    | [amin](https://pytorch.org/docs/1.8.1/generated/torch.amin.html#torch.amin) | 否       |
| 5    | [all](https://pytorch.org/docs/1.8.1/generated/torch.all.html#torch.all) | 否       |
| 6    | [any](https://pytorch.org/docs/1.8.1/generated/torch.any.html#torch.any) | 否       |
| 7    | [max](https://pytorch.org/docs/1.8.1/generated/torch.max.html#torch.max) | 否       |
| 8    | [min](https://pytorch.org/docs/1.8.1/generated/torch.min.html#torch.min) | 否       |
| 9    | [dist](https://pytorch.org/docs/1.8.1/generated/torch.dist.html#torch.dist) | 否       |
| 10   | [logsumexp](https://pytorch.org/docs/1.8.1/generated/torch.logsumexp.html#torch.logsumexp) | 否       |
| 11   | [mean](https://pytorch.org/docs/1.8.1/generated/torch.mean.html#torch.mean) | 否       |
| 12   | [median](https://pytorch.org/docs/1.8.1/generated/torch.median.html#torch.median) | 否       |
| 13   | [nanmedian](https://pytorch.org/docs/1.8.1/generated/torch.nanmedian.html#torch.nanmedian) | 否       |
| 14   | [mode](https://pytorch.org/docs/1.8.1/generated/torch.mode.html#torch.mode) | 否       |
| 15   | [norm](https://pytorch.org/docs/1.8.1/generated/torch.norm.html#torch.norm) | 否       |
| 16   | [nansum](https://pytorch.org/docs/1.8.1/generated/torch.nansum.html#torch.nansum) | 否       |
| 17   | [prod](https://pytorch.org/docs/1.8.1/generated/torch.prod.html#torch.prod) | 否       |
| 18   | [quantile](https://pytorch.org/docs/1.8.1/generated/torch.quantile.html#torch.quantile) | 否       |
| 19   | [nanquantile](https://pytorch.org/docs/1.8.1/generated/torch.nanquantile.html#torch.nanquantile) | 否       |
| 20   | [std](https://pytorch.org/docs/1.8.1/generated/torch.std.html#torch.std) | 否       |
| 21   | [std_mean](https://pytorch.org/docs/1.8.1/generated/torch.std_mean.html#torch.std_mean) | 否       |
| 22   | [sum](https://pytorch.org/docs/1.8.1/generated/torch.sum.html#torch.sum) | 否       |
| 23   | [unique](https://pytorch.org/docs/1.8.1/generated/torch.unique.html#torch.unique) | 否       |
| 24   | [unique_consecutive](https://pytorch.org/docs/1.8.1/generated/torch.unique_consecutive.html#torch.unique_consecutive) | 否       |
| 25   | [var](https://pytorch.org/docs/1.8.1/generated/torch.var.html#torch.var) | 否       |
| 26   | [var_mean](https://pytorch.org/docs/1.8.1/generated/torch.var_mean.html#torch.var_mean) | 否       |
| 27   | [count_nonzero](https://pytorch.org/docs/1.8.1/generated/torch.count_nonzero.html#torch.count_nonzero) | 否       |

### Comparison Ops

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [allclose](https://pytorch.org/docs/1.8.1/generated/torch.allclose.html#torch.allclose) | 否       |
| 2    | [argsort](https://pytorch.org/docs/1.8.1/generated/torch.argsort.html#torch.argsort) | 否       |
| 3    | [eq](https://pytorch.org/docs/1.8.1/generated/torch.eq.html#torch.eq) | 否       |
| 4    | [equal](https://pytorch.org/docs/1.8.1/generated/torch.equal.html#torch.equal) | 否       |
| 5    | [ge](https://pytorch.org/docs/1.8.1/generated/torch.ge.html#torch.ge) | 否       |
| 6    | [greater_equal](https://pytorch.org/docs/1.8.1/generated/torch.greater_equal.html#torch.greater_equal) | 否       |
| 7    | [gt](https://pytorch.org/docs/1.8.1/generated/torch.gt.html#torch.gt) | 否       |
| 8    | [greater](https://pytorch.org/docs/1.8.1/generated/torch.greater.html#torch.greater) | 否       |
| 9    | [isclose](https://pytorch.org/docs/1.8.1/generated/torch.isclose.html#torch.isclose) | 否       |
| 10   | [isfinite](https://pytorch.org/docs/1.8.1/generated/torch.isfinite.html#torch.isfinite) | 否       |
| 11   | [isinf](https://pytorch.org/docs/1.8.1/generated/torch.isinf.html#torch.isinf) | 否       |
| 12   | [isposinf](https://pytorch.org/docs/1.8.1/generated/torch.isposinf.html#torch.isposinf) | 否       |
| 13   | [isneginf](https://pytorch.org/docs/1.8.1/generated/torch.isneginf.html#torch.isneginf) | 否       |
| 14   | [isnan](https://pytorch.org/docs/1.8.1/generated/torch.isnan.html#torch.isnan) | 否       |
| 15   | [isreal](https://pytorch.org/docs/1.8.1/generated/torch.isreal.html#torch.isreal) | 否       |
| 16   | [kthvalue](https://pytorch.org/docs/1.8.1/generated/torch.kthvalue.html#torch.kthvalue) | 否       |
| 17   | [le](https://pytorch.org/docs/1.8.1/generated/torch.le.html#torch.le) | 否       |
| 18   | [less_equal](https://pytorch.org/docs/1.8.1/generated/torch.less_equal.html#torch.less_equal) | 否       |
| 19   | [lt](https://pytorch.org/docs/1.8.1/generated/torch.lt.html#torch.lt) | 否       |
| 20   | [less](https://pytorch.org/docs/1.8.1/generated/torch.less.html#torch.less) | 否       |
| 21   | [maximum](https://pytorch.org/docs/1.8.1/generated/torch.maximum.html#torch.maximum) | 否       |
| 22   | [minimum](https://pytorch.org/docs/1.8.1/generated/torch.minimum.html#torch.minimum) | 否       |
| 23   | [fmax](https://pytorch.org/docs/1.8.1/generated/torch.fmax.html#torch.fmax) | 否       |
| 24   | [fmin](https://pytorch.org/docs/1.8.1/generated/torch.fmin.html#torch.fmin) | 否       |
| 25   | [ne](https://pytorch.org/docs/1.8.1/generated/torch.ne.html#torch.ne) | 否       |
| 26   | [not_equal](https://pytorch.org/docs/1.8.1/generated/torch.not_equal.html#torch.not_equal) | 否       |
| 27   | [sort](https://pytorch.org/docs/1.8.1/generated/torch.sort.html#torch.sort) | 否       |
| 28   | [topk](https://pytorch.org/docs/1.8.1/generated/torch.topk.html#torch.topk) | 否       |
| 29   | [msort](https://pytorch.org/docs/1.8.1/generated/torch.msort.html#torch.msort) | 否       |

### Spectral Ops

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [stft](https://pytorch.org/docs/1.8.1/generated/torch.stft.html#torch.stft) | 否       |
| 2    | [istft](https://pytorch.org/docs/1.8.1/generated/torch.istft.html#torch.istft) | 否       |
| 3    | [bartlett_window](https://pytorch.org/docs/1.8.1/generated/torch.bartlett_window.html#torch.bartlett_window) | 否       |
| 4    | [blackman_window](https://pytorch.org/docs/1.8.1/generated/torch.blackman_window.html#torch.blackman_window) | 否       |
| 5    | [hamming_window](https://pytorch.org/docs/1.8.1/generated/torch.hamming_window.html#torch.hamming_window) | 否       |
| 6    | [hann_window](https://pytorch.org/docs/1.8.1/generated/torch.hann_window.html#torch.hann_window) | 否       |
| 7    | [kaiser_window](https://pytorch.org/docs/1.8.1/generated/torch.kaiser_window.html#torch.kaiser_window) | 否       |

### Other Operations

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [atleast_1d](https://pytorch.org/docs/1.8.1/generated/torch.atleast_1d.html#torch.atleast_1d) | 否       |
| 2    | [atleast_2d](https://pytorch.org/docs/1.8.1/generated/torch.atleast_2d.html#torch.atleast_2d) | 否       |
| 3    | [atleast_3d](https://pytorch.org/docs/1.8.1/generated/torch.atleast_3d.html#torch.atleast_3d) | 否       |
| 4    | [bincount](https://pytorch.org/docs/1.8.1/generated/torch.bincount.html#torch.bincount) | 否       |
| 5    | [block_diag](https://pytorch.org/docs/1.8.1/generated/torch.block_diag.html#torch.block_diag) | 否       |
| 6    | [broadcast_tensors](https://pytorch.org/docs/1.8.1/generated/torch.broadcast_tensors.html#torch.broadcast_tensors) | 否       |
| 7    | [broadcast_to](https://pytorch.org/docs/1.8.1/generated/torch.broadcast_to.html#torch.broadcast_to) | 否       |
| 8    | [broadcast_shapes](https://pytorch.org/docs/1.8.1/generated/torch.broadcast_shapes.html#torch.broadcast_shapes) | 否       |
| 9    | [bucketize](https://pytorch.org/docs/1.8.1/generated/torch.bucketize.html#torch.bucketize) | 否       |
| 10   | [cartesian_prod](https://pytorch.org/docs/1.8.1/generated/torch.cartesian_prod.html#torch.cartesian_prod) | 否       |
| 11   | [cdist](https://pytorch.org/docs/1.8.1/generated/torch.cdist.html#torch.cdist) | 否       |
| 12   | [clone](https://pytorch.org/docs/1.8.1/generated/torch.clone.html#torch.clone) | 否       |
| 13   | [combinations](https://pytorch.org/docs/1.8.1/generated/torch.combinations.html#torch.combinations) | 否       |
| 14   | [cross](https://pytorch.org/docs/1.8.1/generated/torch.cross.html#torch.cross) | 否       |
| 15   | [cummax](https://pytorch.org/docs/1.8.1/generated/torch.cummax.html#torch.cummax) | 否       |
| 16   | [cummin](https://pytorch.org/docs/1.8.1/generated/torch.cummin.html#torch.cummin) | 否       |
| 17   | [cumprod](https://pytorch.org/docs/1.8.1/generated/torch.cumprod.html#torch.cumprod) | 否       |
| 18   | [cumsum](https://pytorch.org/docs/1.8.1/generated/torch.cumsum.html#torch.cumsum) | 否       |
| 19   | [diag](https://pytorch.org/docs/1.8.1/generated/torch.diag.html#torch.diag) | 否       |
| 20   | [diag_embed](https://pytorch.org/docs/1.8.1/generated/torch.diag_embed.html#torch.diag_embed) | 否       |
| 21   | [diagflat](https://pytorch.org/docs/1.8.1/generated/torch.diagflat.html#torch.diagflat) | 否       |
| 22   | [diagonal](https://pytorch.org/docs/1.8.1/generated/torch.diagonal.html#torch.diagonal) | 否       |
| 23   | [diff](https://pytorch.org/docs/1.8.1/generated/torch.diff.html#torch.diff) | 否       |
| 24   | [einsum](https://pytorch.org/docs/1.8.1/generated/torch.einsum.html#torch.einsum) | 否       |
| 25   | [flatten](https://pytorch.org/docs/1.8.1/generated/torch.flatten.html#torch.flatten) | 否       |
| 26   | [flip](https://pytorch.org/docs/1.8.1/generated/torch.flip.html#torch.flip) | 否       |
| 27   | [fliplr](https://pytorch.org/docs/1.8.1/generated/torch.fliplr.html#torch.fliplr) | 否       |
| 28   | [flipud](https://pytorch.org/docs/1.8.1/generated/torch.flipud.html#torch.flipud) | 否       |
| 29   | [kron](https://pytorch.org/docs/1.8.1/generated/torch.kron.html#torch.kron) | 否       |
| 30   | [rot90](https://pytorch.org/docs/1.8.1/generated/torch.rot90.html#torch.rot90) | 否       |
| 31   | [gcd](https://pytorch.org/docs/1.8.1/generated/torch.gcd.html#torch.gcd) | 否       |
| 32   | [histc](https://pytorch.org/docs/1.8.1/generated/torch.histc.html#torch.histc) | 否       |
| 33   | [meshgrid](https://pytorch.org/docs/1.8.1/generated/torch.meshgrid.html#torch.meshgrid) | 否       |
| 34   | [lcm](https://pytorch.org/docs/1.8.1/generated/torch.lcm.html#torch.lcm) | 否       |
| 35   | [logcumsumexp](https://pytorch.org/docs/1.8.1/generated/torch.logcumsumexp.html#torch.logcumsumexp) | 否       |
| 36   | [ravel](https://pytorch.org/docs/1.8.1/generated/torch.ravel.html#torch.ravel) | 否       |
| 37   | [renorm](https://pytorch.org/docs/1.8.1/generated/torch.renorm.html#torch.renorm) | 否       |
| 38   | [repeat_interleave](https://pytorch.org/docs/1.8.1/generated/torch.repeat_interleave.html#torch.repeat_interleave) | 否       |
| 39   | [roll](https://pytorch.org/docs/1.8.1/generated/torch.roll.html#torch.roll) | 否       |
| 40   | [searchsorted](https://pytorch.org/docs/1.8.1/generated/torch.searchsorted.html#torch.searchsorted) | 否       |
| 41   | [tensordot](https://pytorch.org/docs/1.8.1/generated/torch.tensordot.html#torch.tensordot) | 否       |
| 42   | [trace](https://pytorch.org/docs/1.8.1/generated/torch.trace.html#torch.trace) | 否       |
| 43   | [tril](https://pytorch.org/docs/1.8.1/generated/torch.tril.html#torch.tril) | 否       |
| 44   | [tril_indices](https://pytorch.org/docs/1.8.1/generated/torch.tril_indices.html#torch.tril_indices) | 否       |
| 45   | [triu](https://pytorch.org/docs/1.8.1/generated/torch.triu.html#torch.triu) | 否       |
| 46   | [triu_indices](https://pytorch.org/docs/1.8.1/generated/torch.triu_indices.html#torch.triu_indices) | 否       |
| 47   | [vander](https://pytorch.org/docs/1.8.1/generated/torch.vander.html#torch.vander) | 否       |
| 48   | [view_as_real](https://pytorch.org/docs/1.8.1/generated/torch.view_as_real.html#torch.view_as_real) | 否       |
| 49   | [view_as_complex](https://pytorch.org/docs/1.8.1/generated/torch.view_as_complex.html#torch.view_as_complex) | 否       |

### BLAS and LAPACK Operations

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [addbmm](https://pytorch.org/docs/1.8.1/generated/torch.addbmm.html#torch.addbmm) | 否       |
| 2    | [addmm](https://pytorch.org/docs/1.8.1/generated/torch.addmm.html#torch.addmm) | 否       |
| 3    | [addmv](https://pytorch.org/docs/1.8.1/generated/torch.addmv.html#torch.addmv) | 否       |
| 4    | [addr](https://pytorch.org/docs/1.8.1/generated/torch.addr.html#torch.addr) | 否       |
| 5    | [baddbmm](https://pytorch.org/docs/1.8.1/generated/torch.baddbmm.html#torch.baddbmm) | 否       |
| 6    | [bmm](https://pytorch.org/docs/1.8.1/generated/torch.bmm.html#torch.bmm) | 否       |
| 7    | [chain_matmul](https://pytorch.org/docs/1.8.1/generated/torch.chain_matmul.html#torch.chain_matmul) | 否       |
| 8    | [cholesky](https://pytorch.org/docs/1.8.1/generated/torch.cholesky.html#torch.cholesky) | 否       |
| 9    | [cholesky_inverse](https://pytorch.org/docs/1.8.1/generated/torch.cholesky_inverse.html#torch.cholesky_inverse) | 否       |
| 10   | [cholesky_solve](https://pytorch.org/docs/1.8.1/generated/torch.cholesky_solve.html#torch.cholesky_solve) | 否       |
| 11   | [dot](https://pytorch.org/docs/1.8.1/generated/torch.dot.html#torch.dot) | 否       |
| 12   | [eig](https://pytorch.org/docs/1.8.1/generated/torch.eig.html#torch.eig) | 否       |
| 13   | [geqrf](https://pytorch.org/docs/1.8.1/generated/torch.geqrf.html#torch.geqrf) | 否       |
| 14   | [ger](https://pytorch.org/docs/1.8.1/generated/torch.ger.html#torch.ger) | 否       |
| 15   | [inner](https://pytorch.org/docs/1.8.1/generated/torch.inner.html#torch.inner) | 否       |
| 16   | [inverse](https://pytorch.org/docs/1.8.1/generated/torch.inverse.html#torch.inverse) | 否       |
| 17   | [det](https://pytorch.org/docs/1.8.1/generated/torch.det.html#torch.det) | 否       |
| 18   | [logdet](https://pytorch.org/docs/1.8.1/generated/torch.logdet.html#torch.logdet) | 否       |
| 19   | [slogdet](https://pytorch.org/docs/1.8.1/generated/torch.slogdet.html#torch.slogdet) | 否       |
| 20   | [lstsq](https://pytorch.org/docs/1.8.1/generated/torch.lstsq.html#torch.lstsq) | 否       |
| 21   | [lu](https://pytorch.org/docs/1.8.1/generated/torch.lu.html#torch.lu) | 否       |
| 22   | [lu_solve](https://pytorch.org/docs/1.8.1/generated/torch.lu_solve.html#torch.lu_solve) | 否       |
| 23   | [lu_unpack](https://pytorch.org/docs/1.8.1/generated/torch.lu_unpack.html#torch.lu_unpack) | 否       |
| 24   | [matmul](https://pytorch.org/docs/1.8.1/generated/torch.matmul.html#torch.matmul) | 否       |
| 25   | [matrix_power](https://pytorch.org/docs/1.8.1/generated/torch.matrix_power.html#torch.matrix_power) | 否       |
| 26   | [matrix_rank](https://pytorch.org/docs/1.8.1/generated/torch.matrix_rank.html#torch.matrix_rank) | 否       |
| 27   | [matrix_exp](https://pytorch.org/docs/1.8.1/generated/torch.matrix_exp.html#torch.matrix_exp) | 否       |
| 28   | [mm](https://pytorch.org/docs/1.8.1/generated/torch.mm.html#torch.mm) | 否       |
| 29   | [mv](https://pytorch.org/docs/1.8.1/generated/torch.mv.html#torch.mv) | 否       |
| 30   | [orgqr](https://pytorch.org/docs/1.8.1/generated/torch.orgqr.html#torch.orgqr) | 否       |
| 31   | [ormqr](https://pytorch.org/docs/1.8.1/generated/torch.ormqr.html#torch.ormqr) | 否       |
| 32   | [outer](https://pytorch.org/docs/1.8.1/generated/torch.outer.html#torch.outer) | 否       |
| 33   | [pinverse](https://pytorch.org/docs/1.8.1/generated/torch.pinverse.html#torch.pinverse) | 否       |
| 34   | [qr](https://pytorch.org/docs/1.8.1/generated/torch.qr.html#torch.qr) | 否       |
| 35   | [solve](https://pytorch.org/docs/1.8.1/generated/torch.solve.html#torch.solve) | 否       |
| 36   | [svd](https://pytorch.org/docs/1.8.1/generated/torch.svd.html#torch.svd) | 否       |
| 37   | [svd_lowrank](https://pytorch.org/docs/1.8.1/generated/torch.svd_lowrank.html#torch.svd_lowrank) | 否       |
| 38   | [pca_lowrank](https://pytorch.org/docs/1.8.1/generated/torch.pca_lowrank.html#torch.pca_lowrank) | 否       |
| 39   | [symeig](https://pytorch.org/docs/1.8.1/generated/torch.symeig.html#torch.symeig) | 否       |
| 40   | [lobpcg](https://pytorch.org/docs/1.8.1/generated/torch.lobpcg.html#torch.lobpcg) | 否       |
| 41   | [trapz](https://pytorch.org/docs/1.8.1/generated/torch.trapz.html#torch.trapz) | 否       |
| 42   | [triangular_solve](https://pytorch.org/docs/1.8.1/generated/torch.triangular_solve.html#torch.triangular_solve) | 否       |
| 43   | [vdot](https://pytorch.org/docs/1.8.1/generated/torch.vdot.html#torch.vdot) | 否       |

## Utilities

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [compiled_with_cxx11_abi](https://pytorch.org/docs/1.8.1/generated/torch.compiled_with_cxx11_abi.html#torch.compiled_with_cxx11_abi) | 否       |
| 2    | [result_type](https://pytorch.org/docs/1.8.1/generated/torch.result_type.html#torch.result_type) | 否       |
| 3    | [can_cast](https://pytorch.org/docs/1.8.1/generated/torch.can_cast.html#torch.can_cast) | 否       |
| 4    | [promote_types](https://pytorch.org/docs/1.8.1/generated/torch.promote_types.html#torch.promote_types) | 否       |
| 5    | [use_deterministic_algorithms](https://pytorch.org/docs/1.8.1/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms) | 否       |
| 6    | [are_deterministic_algorithms_enabled](https://pytorch.org/docs/1.8.1/generated/torch.are_deterministic_algorithms_enabled.html#torch.are_deterministic_algorithms_enabled) | 否       |
| 7    | [_assert](https://pytorch.org/docs/1.8.1/generated/torch._assert.html#torch._assert) | 否       |

## Other

## torch.Tensor

# Layers (torch.nn)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [Parameter](https://pytorch.org/docs/1.8.1/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) | 否       |
| 2    | [UninitializedParameter](https://pytorch.org/docs/1.8.1/generated/torch.nn.parameter.UninitializedParameter.html#torch.nn.parameter.UninitializedParameter) | 否       |

## [Containers](https://pytorch.org/docs/1.8.1/nn.html#id1)


| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [Module](https://pytorch.org/docs/1.8.1/generated/torch.nn.Module.html#torch.nn.Module) | 否       |
| 2    | [Sequential](https://pytorch.org/docs/1.8.1/generated/torch.nn.Sequential.html#torch.nn.Sequential) | 否       |
| 3    | [ModuleList](https://pytorch.org/docs/1.8.1/generated/torch.nn.ModuleList.html#torch.nn.ModuleList) | 否       |
| 4    | [ModuleDict](https://pytorch.org/docs/1.8.1/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict) | 否       |
| 5    | [ParameterList](https://pytorch.org/docs/1.8.1/generated/torch.nn.ParameterList.html#torch.nn.ParameterList) | 否       |
| 6    | [ParameterDict](https://pytorch.org/docs/1.8.1/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict) | 否       |

### Global Hooks For Module

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [register_module_forward_pre_hook](https://pytorch.org/docs/1.8.1/generated/torch.nn.modules.module.register_module_forward_pre_hook.html#torch.nn.modules.module.register_module_forward_pre_hook) | 否       |
| 2    | [register_module_forward_hook](https://pytorch.org/docs/1.8.1/generated/torch.nn.modules.module.register_module_forward_hook.html#torch.nn.modules.module.register_module_forward_hook) | 否       |
| 3    | [register_module_backward_hook](https://pytorch.org/docs/1.8.1/generated/torch.nn.modules.module.register_module_backward_hook.html#torch.nn.modules.module.register_module_backward_hook) | 否       |

## [Convolution Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.Conv1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv1d.html#torch.nn.Conv1d) | 否       |
| 2    | [nn.Conv2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv2d.html#torch.nn.Conv2d) | 否       |
| 3    | [nn.Conv3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv3d.html#torch.nn.Conv3d) | 否       |
| 4    | [nn.ConvTranspose1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose1d.html#torch.nn.ConvTranspose1d) | 否       |
| 5    | [nn.ConvTranspose2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d) | 否       |
| 6    | [nn.ConvTranspose3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose3d.html#torch.nn.ConvTranspose3d) | 否       |
| 7    | [nn.LazyConv1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConv1d.html#torch.nn.LazyConv1d) | 否       |
| 8    | [nn.LazyConv2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d) | 否       |
| 9    | [nn.LazyConv3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConv3d.html#torch.nn.LazyConv3d) | 否       |
| 10   | [nn.LazyConvTranspose1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConvTranspose1d.html#torch.nn.LazyConvTranspose1d) | 否       |
| 11   | [nn.LazyConvTranspose2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConvTranspose2d.html#torch.nn.LazyConvTranspose2d) | 否       |
| 12   | [nn.LazyConvTranspose3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyConvTranspose3d.html#torch.nn.LazyConvTranspose3d) | 否       |
| 13   | [nn.Unfold](https://pytorch.org/docs/1.8.1/generated/torch.nn.Unfold.html#torch.nn.Unfold) | 否       |
| 14   | [nn.Fold](https://pytorch.org/docs/1.8.1/generated/torch.nn.Fold.html#torch.nn.Fold) | 否       |

## [Pooling layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.MaxPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d) | 否       |
| 2    | [nn.MaxPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d) | 否       |
| 3    | [nn.MaxPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxPool3d.html#torch.nn.MaxPool3d) | 否       |
| 4    | [nn.MaxUnpool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxUnpool1d.html#torch.nn.MaxUnpool1d) | 否       |
| 5    | [nn.MaxUnpool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxUnpool2d.html#torch.nn.MaxUnpool2d) | 否       |
| 6    | [nn.MaxUnpool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.MaxUnpool3d.html#torch.nn.MaxUnpool3d) | 否       |
| 7    | [nn.AvgPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool1d.html#torch.nn.AvgPool1d) | 否       |
| 8    | [nn.AvgPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d) | 否       |
| 9    | [nn.AvgPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AvgPool3d.html#torch.nn.AvgPool3d) | 否       |
| 10   | [nn.FractionalMaxPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.FractionalMaxPool2d.html#torch.nn.FractionalMaxPool2d) | 否       |
| 11   | [nn.LPPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LPPool1d.html#torch.nn.LPPool1d) | 否       |
| 12   | [nn.LPPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.LPPool2d.html#torch.nn.LPPool2d) | 否       |
| 13   | [nn.AdaptiveMaxPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveMaxPool1d.html#torch.nn.AdaptiveMaxPool1d) | 否       |
| 14   | [nn.AdaptiveMaxPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveMaxPool2d.html#torch.nn.AdaptiveMaxPool2d) | 否       |
| 15   | [nn.AdaptiveMaxPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveMaxPool3d.html#torch.nn.AdaptiveMaxPool3d) | 否       |
| 16   | [nn.AdaptiveAvgPool1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveAvgPool1d.html#torch.nn.AdaptiveAvgPool1d) | 否       |
| 17   | [nn.AdaptiveAvgPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d) | 否       |
| 18   | [nn.AdaptiveAvgPool3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveAvgPool3d.html#torch.nn.AdaptiveAvgPool3d) | 否       |

## [Padding Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.ReflectionPad1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReflectionPad1d.html#torch.nn.ReflectionPad1d) | 否       |
| 2    | [nn.ReflectionPad2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReflectionPad2d.html#torch.nn.ReflectionPad2d) | 否       |
| 3    | [nn.ReplicationPad1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReplicationPad1d.html#torch.nn.ReplicationPad1d) | 否       |
| 4    | [nn.ReplicationPad2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReplicationPad2d.html#torch.nn.ReplicationPad2d) | 否       |
| 5    | [nn.ReplicationPad3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReplicationPad3d.html#torch.nn.ReplicationPad3d) | 否       |
| 6    | [nn.ZeroPad2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ZeroPad2d.html#torch.nn.ZeroPad2d) | 否       |
| 7    | [nn.ConstantPad1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConstantPad1d.html#torch.nn.ConstantPad1d) | 否       |
| 8    | [nn.ConstantPad2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConstantPad2d.html#torch.nn.ConstantPad2d) | 否       |
| 9    | [nn.ConstantPad3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConstantPad3d.html#torch.nn.ConstantPad3d) | 否       |



## [Non-linear Activations (weighted sum, nonlinearity)](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.ELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.ELU.html#torch.nn.ELU) | 否       |
| 2    | [nn.Hardshrink](https://pytorch.org/docs/1.8.1/generated/torch.nn.Hardshrink.html#torch.nn.Hardshrink) | 否       |
| 3    | [nn.Hardsigmoid](https://pytorch.org/docs/1.8.1/generated/torch.nn.Hardsigmoid.html#torch.nn.Hardsigmoid) | 否       |
| 4    | [nn.Hardtanh](https://pytorch.org/docs/1.8.1/generated/torch.nn.Hardtanh.html#torch.nn.Hardtanh) | 否       |
| 5    | [nn.Hardswish](https://pytorch.org/docs/1.8.1/generated/torch.nn.Hardswish.html#torch.nn.Hardswish) | 否       |
| 6    | [nn.LeakyReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU) | 否       |
| 7    | [nn.LogSigmoid](https://pytorch.org/docs/1.8.1/generated/torch.nn.LogSigmoid.html#torch.nn.LogSigmoid) | 否       |
| 8    | [nn.MultiheadAttention](https://pytorch.org/docs/1.8.1/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention) | 否       |
| 9    | [nn.PReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.PReLU.html#torch.nn.PReLU) | 否       |
| 10   | [nn.ReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReLU.html#torch.nn.ReLU) | 否       |
| 11   | [nn.ReLU6](https://pytorch.org/docs/1.8.1/generated/torch.nn.ReLU6.html#torch.nn.ReLU6) | 否       |
| 12   | [nn.RReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.RReLU.html#torch.nn.RReLU) | 否       |
| 13   | [nn.SELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.SELU.html#torch.nn.SELU) | 否       |
| 14   | [nn.CELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.CELU.html#torch.nn.CELU) | 否       |
| 15   | [nn.GELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.GELU.html#torch.nn.GELU) | 否       |
| 16   | [nn.Sigmoid](https://pytorch.org/docs/1.8.1/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid) | 否       |
| 17   | [nn.SiLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.SiLU.html#torch.nn.SiLU) | 否       |
| 18   | [nn.Softplus](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softplus.html#torch.nn.Softplus) | 否       |
| 19   | [nn.Softshrink](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softshrink.html#torch.nn.Softshrink) | 否       |
| 20   | [nn.Softsign](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softsign.html#torch.nn.Softsign) | 否       |
| 21   | [nn.Tanh](https://pytorch.org/docs/1.8.1/generated/torch.nn.Tanh.html#torch.nn.Tanh) | 否       |
| 22   | [nn.Tanhshrink](https://pytorch.org/docs/1.8.1/generated/torch.nn.Tanhshrink.html#torch.nn.Tanhshrink) | 否       |
| 23   | [nn.Threshold](https://pytorch.org/docs/1.8.1/generated/torch.nn.Threshold.html#torch.nn.Threshold) | 否       |

## [Non-linear Activations (other)](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.Softmin](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmin.html#torch.nn.Softmin) | 否       |
| 2    | [nn.Softmax](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmax.html#torch.nn.Softmax) | 否       |
| 3    | [nn.Softmax2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmax2d.html#torch.nn.Softmax2d) | 否       |
| 4    | [nn.LogSoftmax](https://pytorch.org/docs/1.8.1/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) | 否       |
| 5    | [nn.AdaptiveLogSoftmaxWithLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss) | 否       |

## [Normalization Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.BatchNorm1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d) | 否       |
| 2    | [nn.BatchNorm2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d) | 否       |
| 3    | [nn.BatchNorm3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm3d.html#torch.nn.BatchNorm3d) | 否       |
| 4    | [nn.GroupNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.GroupNorm.html#torch.nn.GroupNorm) | 否       |
| 5    | [nn.SyncBatchNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm) | 否       |
| 6    | [nn.InstanceNorm1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.InstanceNorm1d.html#torch.nn.InstanceNorm1d) | 否       |
| 7    | [nn.InstanceNorm2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.InstanceNorm2d.html#torch.nn.InstanceNorm2d) | 否       |
| 8    | [nn.InstanceNorm3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.InstanceNorm3d.html#torch.nn.InstanceNorm3d) | 否       |
| 9    | [nn.LayerNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm) | 否       |
| 10   | [nn.LocalResponseNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.LocalResponseNorm.html#torch.nn.LocalResponseNorm) | 否       |



## [Recurrent Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.RNNBase](https://pytorch.org/docs/1.8.1/generated/torch.nn.RNNBase.html#torch.nn.RNNBase) | 否       |
| 2    | [nn.RNN](https://pytorch.org/docs/1.8.1/generated/torch.nn.RNN.html#torch.nn.RNN) | 否       |
| 3    | [nn.LSTM](https://pytorch.org/docs/1.8.1/generated/torch.nn.LSTM.html#torch.nn.LSTM) | 否       |
| 4    | [nn.GRU](https://pytorch.org/docs/1.8.1/generated/torch.nn.GRU.html#torch.nn.GRU) | 否       |
| 5    | [nn.RNNCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.RNNCell.html#torch.nn.RNNCell) | 否       |
| 6    | [nn.LSTMCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.LSTMCell.html#torch.nn.LSTMCell) | 否       |
| 7    | [nn.GRUCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.GRUCell.html#torch.nn.GRUCell) | 否       |



## [Transformer Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.Transformer](https://pytorch.org/docs/1.8.1/generated/torch.nn.Transformer.html#torch.nn.Transformer) | 否       |
| 2    | [nn.TransformerEncoder](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder) | 否       |
| 3    | [nn.TransformerDecoder](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder) | 否       |
| 4    | [nn.TransformerEncoderLayer](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer) | 否       |
| 5    | [nn.TransformerDecoderLayer](https://pytorch.org/docs/1.8.1/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer) | 否       |



## [Linear Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.Identity](https://pytorch.org/docs/1.8.1/generated/torch.nn.Identity.html#torch.nn.Identity) | 否       |
| 2    | [nn.Linear](https://pytorch.org/docs/1.8.1/generated/torch.nn.Linear.html#torch.nn.Linear) | 否       |
| 3    | [nn.Bilinear](https://pytorch.org/docs/1.8.1/generated/torch.nn.Bilinear.html#torch.nn.Bilinear) | 否       |
| 4    | [nn.LazyLinear](https://pytorch.org/docs/1.8.1/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear) | 否       |



## [Dropout Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)



| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.Dropout](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout.html#torch.nn.Dropout) | 否       |
| 2    | [nn.Dropout2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout2d.html#torch.nn.Dropout2d) | 否       |
| 3    | [nn.Dropout3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout3d.html#torch.nn.Dropout3d) | 否       |
| 4    | [nn.AlphaDropout](https://pytorch.org/docs/1.8.1/generated/torch.nn.AlphaDropout.html#torch.nn.AlphaDropout) | 否       |

## [Sparse Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.Embedding](https://pytorch.org/docs/1.8.1/generated/torch.nn.Embedding.html#torch.nn.Embedding) | 否       |
| 2    | [nn.EmbeddingBag](https://pytorch.org/docs/1.8.1/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag) | 否       |



## [Distance Functions](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.CosineSimilarity](https://pytorch.org/docs/1.8.1/generated/torch.nn.CosineSimilarity.html#torch.nn.CosineSimilarity) | 否       |
| 2    | [nn.PairwiseDistance](https://pytorch.org/docs/1.8.1/generated/torch.nn.PairwiseDistance.html#torch.nn.PairwiseDistance) | 否       |



## [Loss Functions](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.L1Loss](https://pytorch.org/docs/1.8.1/generated/torch.nn.L1Loss.html#torch.nn.L1Loss) | 否       |
| 2    | [nn.MSELoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) | 否       |
| 3    | [nn.CrossEntropyLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) | 否       |
| 4    | [nn.CTCLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss) | 否       |
| 5    | [nn.NLLLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) | 否       |
| 6    | [nn.PoissonNLLLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.PoissonNLLLoss.html#torch.nn.PoissonNLLLoss) | 否       |
| 7    | [nn.GaussianNLLLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.GaussianNLLLoss.html#torch.nn.GaussianNLLLoss) | 否       |
| 8    | [nn.KLDivLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss) | 否       |
| 9    | [nn.BCELoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.BCELoss.html#torch.nn.BCELoss) | 否       |
| 10   | [nn.BCEWithLogitsLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss) | 否       |
| 11   | [nn.MarginRankingLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss) | 否       |
| 12   | [nn.HingeEmbeddingLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss) | 否       |
| 13   | [nn.MultiLabelMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MultiLabelMarginLoss.html#torch.nn.MultiLabelMarginLoss) | 否       |
| 14   | [nn.SmoothL1Loss](https://pytorch.org/docs/1.8.1/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss) | 否       |
| 15   | [nn.SoftMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.SoftMarginLoss.html#torch.nn.SoftMarginLoss) | 否       |
| 16   | [nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MultiLabelSoftMarginLoss.html#torch.nn.MultiLabelSoftMarginLoss) | 否       |
| 17   | [nn.CosineEmbeddingLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss) | 否       |
| 18   | [nn.MultiMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.MultiMarginLoss.html#torch.nn.MultiMarginLoss) | 否       |
| 19   | [nn.TripletMarginLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss) | 否       |
| 20   | [nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss) | 否       |

## [Vision Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.PixelShuffle](https://pytorch.org/docs/1.8.1/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle) | 否       |
| 2    | [nn.PixelUnshuffle](https://pytorch.org/docs/1.8.1/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle) | 否       |
| 3    | [nn.Upsample](https://pytorch.org/docs/1.8.1/generated/torch.nn.Upsample.html#torch.nn.Upsample) | 否       |
| 4    | [nn.UpsamplingNearest2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.UpsamplingNearest2d.html#torch.nn.UpsamplingNearest2d) | 否       |
| 5    | [nn.UpsamplingBilinear2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.UpsamplingBilinear2d.html#torch.nn.UpsamplingBilinear2d) | 否       |



## [Shuffle Layers](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.ChannelShuffle](https://pytorch.org/docs/1.8.1/generated/torch.nn.ChannelShuffle.html#torch.nn.ChannelShuffle) | 否       |



## [DataParallel Layers (multi-GPU, distributed)](https://pytorch.org/docs/1.8.1/nn.html#id1)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.DataParallel](https://pytorch.org/docs/1.8.1/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) | 否       |
| 2    | [nn.parallel.DistributedDataParallel](https://pytorch.org/docs/1.8.1/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) | 否       |

## [Utilities](https://pytorch.org/docs/1.8.1/nn.html#id1)



From the `torch.nn.utils` module

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [clip_grad_norm_](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_) | 否       |
| 2    | [clip_grad_value_](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.clip_grad_value_.html#torch.nn.utils.clip_grad_value_) | 否       |
| 3    | [parameters_to_vector](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.parameters_to_vector.html#torch.nn.utils.parameters_to_vector) | 否       |
| 4    | [vector_to_parameters](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.vector_to_parameters.html#torch.nn.utils.vector_to_parameters) | 否       |
| 5    | [prune.BasePruningMethod](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod) | 否       |
| 6    | [prune.PruningContainer](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer) | 否       |
| 7    | [prune.Identity](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.Identity.html#torch.nn.utils.prune.Identity) | 否       |
| 8    | [prune.RandomUnstructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured) | 否       |
| 9    | [prune.L1Unstructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured) | 否       |
| 10   | [prune.RandomStructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured) | 否       |
| 11   | [prune.LnStructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured) | 否       |
| 12   | [prune.CustomFromMask](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask) | 否       |
| 13   | [prune.identity](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.identity.html#torch.nn.utils.prune.identity) | 否       |
| 14   | [prune.random_unstructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.random_unstructured.html#torch.nn.utils.prune.random_unstructured) | 否       |
| 15   | [prune.l1_unstructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.l1_unstructured.html#torch.nn.utils.prune.l1_unstructured) | 否       |
| 16   | [prune.random_structured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.random_structured.html#torch.nn.utils.prune.random_structured) | 否       |
| 17   | [prune.ln_structured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.ln_structured.html#torch.nn.utils.prune.ln_structured) | 否       |
| 18   | [prune.global_unstructured](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.global_unstructured.html#torch.nn.utils.prune.global_unstructured) | 否       |
| 19   | [prune.custom_from_mask](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.custom_from_mask.html#torch.nn.utils.prune.custom_from_mask) | 否       |
| 20   | [prune.remove](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.remove.html#torch.nn.utils.prune.remove) | 否       |
| 21   | [prune.is_pruned](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.prune.is_pruned.html#torch.nn.utils.prune.is_pruned) | 否       |
| 22   | [weight_norm](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.weight_norm.html#torch.nn.utils.weight_norm) | 否       |
| 23   | [remove_weight_norm](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.remove_weight_norm.html#torch.nn.utils.remove_weight_norm) | 否       |
| 24   | [spectral_norm](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.spectral_norm.html#torch.nn.utils.spectral_norm) | 否       |
| 25   | [remove_spectral_norm](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.remove_spectral_norm.html#torch.nn.utils.remove_spectral_norm) | 否       |



### Utility functions in other modules

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.utils.rnn.PackedSequence](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence) | 否       |
| 2    | [nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence) | 否       |
| 3    | [nn.utils.rnn.pad_packed_sequence](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch.nn.utils.rnn.pad_packed_sequence) | 否       |
| 4    | [nn.utils.rnn.pad_sequence](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.rnn.pad_sequence.html#torch.nn.utils.rnn.pad_sequence) | 否       |
| 5    | [nn.utils.rnn.pack_sequence](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence) | 否       |
| 6    | [nn.Flatten](https://pytorch.org/docs/1.8.1/generated/torch.nn.Flatten.html#torch.nn.Flatten) | 否       |
| 7    | [nn.Unflatten](https://pytorch.org/docs/1.8.1/generated/torch.nn.Unflatten.html#torch.nn.Unflatten) | 否       |

### Lazy Modules Initialization

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [nn.modules.lazy.LazyModuleMixin](https://pytorch.org/docs/1.8.1/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin) | 否       |











## Functions(torch.nn.functional)

### [Convolution functions](https://pytorch.org/docs/1.8.1/nn.functional.html#convolution-functions)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [conv1d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv1d) | 否       |
| 2    | [conv2d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv2d) | 否       |
| 3    | [conv3d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv3d) | 否       |
| 4    | [conv_transpose1d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv-transpose1d) | 否       |
| 5    | [conv_transpose2d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv-transpose2d) | 否       |
| 6    | [conv_transpose3d](https://pytorch.org/docs/1.8.1/nn.functional.html#conv-transpose3d) | 否       |
| 7    | [unfold](https://pytorch.org/docs/1.8.1/nn.functional.html#unfold) | 否       |
| 8    | [fold](https://pytorch.org/docs/1.8.1/nn.functional.html#fold) | 否       |

### [Pooling functions](https://pytorch.org/docs/1.8.1/nn.functional.html#pooling-functions)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [avg_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#avg-pool1d) | 否       |
| 2    | [avg_pool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#avg-pool2d) | 否       |
| 3    | [avg_pool3d](https://pytorch.org/docs/1.8.1/nn.functional.html#avg-pool3d) | 否       |
| 4    | [max_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-pool1d) | 否       |
| 5    | [max_pool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-pool2d) | 否       |
| 6    | [max_pool3d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-pool3d) | 否       |
| 7    | [max_unpool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-unpool1d) | 否       |
| 8    | [max_unpool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-unpool2d) | 否       |
| 9    | [max_unpool3d](https://pytorch.org/docs/1.8.1/nn.functional.html#max-unpool3d) | 否       |
| 10   | [lp_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#lp-pool1d) | 否       |
| 11   | [lp_pool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#lp-pool2d) | 否       |
| 12   | [adaptive_max_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-max-pool1d) | 否       |
| 13   | [adaptive_max_pool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-max-pool2d) | 否       |
| 14   | [adaptive_max_pool3d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-max-pool3d) | 否       |
| 15   | [adaptive_avg_pool1d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-avg-pool1d) | 否       |
| 16   | [adaptive_avg_pool2d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-avg-pool2d) | 否       |
| 17   | [adaptive_avg_pool3d](https://pytorch.org/docs/1.8.1/nn.functional.html#adaptive-avg-pool3d) | 否       |

### [Non-linear activation functions](https://pytorch.org/docs/1.8.1/nn.functional.html#non-linear-activation-functions)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [threshold](https://pytorch.org/docs/1.8.1/nn.functional.html#threshold) | 否       |
| 2    | [relu](https://pytorch.org/docs/1.8.1/nn.functional.html#relu) | 否       |
| 3    | [hardtanh](https://pytorch.org/docs/1.8.1/nn.functional.html#hardtanh) | 否       |
| 4    | [hardswish](https://pytorch.org/docs/1.8.1/nn.functional.html#hardswish) | 否       |
| 5    | [relu6](https://pytorch.org/docs/1.8.1/nn.functional.html#relu6) | 否       |
| 6    | [elu](https://pytorch.org/docs/1.8.1/nn.functional.html#elu) | 否       |
| 7    | [selu](https://pytorch.org/docs/1.8.1/nn.functional.html#selu) | 否       |
| 8    | [celu](https://pytorch.org/docs/1.8.1/nn.functional.html#celu) | 否       |
| 9    | [leaky_relu](https://pytorch.org/docs/1.8.1/nn.functional.html#leaky-relu) | 否       |
| 10   | [prelu](https://pytorch.org/docs/1.8.1/nn.functional.html#prelu) | 否       |
| 11   | [rrelu](https://pytorch.org/docs/1.8.1/nn.functional.html#rrelu) | 否       |
| 12   | [glu](https://pytorch.org/docs/1.8.1/nn.functional.html#glu) | 否       |
| 13   | [gelu](https://pytorch.org/docs/1.8.1/nn.functional.html#gelu) | 否       |
| 14   | [logsigmoid](https://pytorch.org/docs/1.8.1/nn.functional.html#logsigmoid) | 否       |
| 15   | [hardshrink](https://pytorch.org/docs/1.8.1/nn.functional.html#hardshrink) | 否       |
| 16   | [tanhshrink](https://pytorch.org/docs/1.8.1/nn.functional.html#tanhshrink) | 否       |
| 17   | [softsign](https://pytorch.org/docs/1.8.1/nn.functional.html#softsign) | 否       |
| 18   | [softplus](https://pytorch.org/docs/1.8.1/nn.functional.html#softplus) | 否       |
| 19   | [softmin](https://pytorch.org/docs/1.8.1/nn.functional.html#softmin) | 否       |
| 20   | [softmax](https://pytorch.org/docs/1.8.1/nn.functional.html#softmax) | 否       |
| 21   | [softshrink](https://pytorch.org/docs/1.8.1/nn.functional.html#softshrink) | 否       |
| 22   | [gumbel_softmax](https://pytorch.org/docs/1.8.1/nn.functional.html#gumbel-softmax) | 否       |
| 23   | [log_softmax](https://pytorch.org/docs/1.8.1/nn.functional.html#log-softmax) | 否       |
| 24   | [tanh](https://pytorch.org/docs/1.8.1/nn.functional.html#tanh) | 否       |
| 25   | [sigmoid](https://pytorch.org/docs/1.8.1/nn.functional.html#sigmoid) | 否       |
| 26   | [hardsigmoid](https://pytorch.org/docs/1.8.1/nn.functional.html#hardsigmoid) | 否       |
| 27   | [silu](https://pytorch.org/docs/1.8.1/nn.functional.html#silu) | 否       |

### [Normalization functions](https://pytorch.org/docs/1.8.1/nn.functional.html#normalization-functions)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [batch_norm](https://pytorch.org/docs/1.8.1/nn.functional.html#batch-norm) | 否       |
| 2    | [instance_norm](https://pytorch.org/docs/1.8.1/nn.functional.html#instance-norm) | 否       |
| 3    | [layer_norm](https://pytorch.org/docs/1.8.1/nn.functional.html#layer-norm) | 否       |
| 4    | [local_response_norm](https://pytorch.org/docs/1.8.1/nn.functional.html#local-response-norm) | 否       |
| 5    | [normalize](https://pytorch.org/docs/1.8.1/nn.functional.html#normalize) | 否       |

### [Linear functions](https://pytorch.org/docs/1.8.1/nn.functional.html#linear-functions)[Linear functions](https://pytorch.org/docs/1.8.1/nn.functional.html#linear-functions)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [linear](https://pytorch.org/docs/1.8.1/nn.functional.html#linear) | 否       |
| 2    | [bilinear](https://pytorch.org/docs/1.8.1/nn.functional.html#bilinear) | 否       |

### [Dropout functions](https://pytorch.org/docs/1.8.1/nn.functional.html#dropout-functions)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [dropout](https://pytorch.org/docs/1.8.1/nn.functional.html#dropout) | 否       |
| 2    | [alpha_dropout](https://pytorch.org/docs/1.8.1/nn.functional.html#alpha-dropout) | 否       |
| 3    | [feature_alpha_dropout](https://pytorch.org/docs/1.8.1/nn.functional.html#feature-alpha-dropout) | 否       |
| 4    | [dropout2d](https://pytorch.org/docs/1.8.1/nn.functional.html#dropout2d) | 否       |
| 5    | [dropout3d](https://pytorch.org/docs/1.8.1/nn.functional.html#dropout3d) | 否       |

### [Sparse functions](https://pytorch.org/docs/1.8.1/nn.functional.html#sparse-functions)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [embedding](https://pytorch.org/docs/1.8.1/nn.functional.html#embedding) | 否       |
| 2    | [embedding_bag](https://pytorch.org/docs/1.8.1/nn.functional.html#embedding-bag) | 否       |
| 3    | [one_hot](https://pytorch.org/docs/1.8.1/nn.functional.html#one-hot) | 否       |

### [Distance functions](https://pytorch.org/docs/1.8.1/nn.functional.html#distance-functions)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [pairwise_distance](https://pytorch.org/docs/1.8.1/nn.functional.html#pairwise-distance) | 否       |
| 2    | [cosine_similarity](https://pytorch.org/docs/1.8.1/nn.functional.html#cosine-similarity) | 否       |
| 3    | [pdist](https://pytorch.org/docs/1.8.1/nn.functional.html#pdist) | 否       |

### [Loss functions](https://pytorch.org/docs/1.8.1/nn.functional.html#loss-functions)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [binary_cross_entropy](https://pytorch.org/docs/1.8.1/nn.functional.html#binary-cross-entropy) | 否       |
| 2    | [binary_cross_entropy_with_logits](https://pytorch.org/docs/1.8.1/nn.functional.html#binary-cross-entropy-with-logits) | 否       |
| 3    | [poisson_nll_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#poisson-nll-loss) | 否       |
| 4    | [cosine_embedding_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#cosine-embedding-loss) | 否       |
| 5    | [cross_entropy](https://pytorch.org/docs/1.8.1/nn.functional.html#cross-entropy) | 否       |
| 6    | [ctc_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#ctc-loss) | 否       |
| 7    | [hinge_embedding_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#hinge-embedding-loss) | 否       |
| 8    | [kl_div](https://pytorch.org/docs/1.8.1/nn.functional.html#kl-div) | 否       |
| 9    | [l1_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#l1-loss) | 否       |
| 10   | [mse_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#mse-loss) | 否       |
| 11   | [margin_ranking_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#margin-ranking-loss) | 否       |
| 12   | [multilabel_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#multilabel-margin-loss) | 否       |
| 13   | [multilabel_soft_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#multilabel-soft-margin-loss) | 否       |
| 14   | [multi_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#multi-margin-loss) | 否       |
| 15   | [nll_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#nll-loss) | 否       |
| 16   | [smooth_l1_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#smooth-l1-loss) | 否       |
| 17   | [soft_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#soft-margin-loss) | 否       |
| 18   | [triplet_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#triplet-margin-loss) | 否       |
| 19   | [triplet_margin_with_distance_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#triplet-margin-with-distance-loss) | 否       |

### [Vision functions](https://pytorch.org/docs/1.8.1/nn.functional.html#vision-functions)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [pixel_shuffle](https://pytorch.org/docs/1.8.1/nn.functional.html#pixel-shuffle) | 否       |
| 2    | [pixel_unshuffle](https://pytorch.org/docs/1.8.1/nn.functional.html#pixel-unshuffle) | 否       |
| 3    | [pad](https://pytorch.org/docs/1.8.1/nn.functional.html#pad) | 否       |
| 4    | [interpolate](https://pytorch.org/docs/1.8.1/nn.functional.html#interpolate) | 否       |
| 5    | [upsample](https://pytorch.org/docs/1.8.1/nn.functional.html#upsample) | 否       |
| 6    | [upsample_nearest](https://pytorch.org/docs/1.8.1/nn.functional.html#upsample-nearest) | 否       |
| 7    | [upsample_bilinear](https://pytorch.org/docs/1.8.1/nn.functional.html#upsample-bilinear) | 否       |
| 8    | [grid_sample](https://pytorch.org/docs/1.8.1/nn.functional.html#grid-sample) | 否       |
| 9    | [affine_grid](https://pytorch.org/docs/1.8.1/nn.functional.html#affine-grid) | 否       |

### [DataParallel functions (multi-GPU, distributed)](https://pytorch.org/docs/1.8.1/nn.functional.html#dataparallel-functions-multi-gpu-distributed)

| 序号 | API名称                                                      | 支持情况 |
| ---- | ------------------------------------------------------------ | -------- |
| 1    | [data_parallel](https://pytorch.org/docs/1.8.1/nn.functional.html#data-parallel) | 否       |





## [torch.distributed](https://pytorch.org/docs/1.8.1/distributed.html)

| 序号 | API名称                                   | 支持情况 |
| ---- | ----------------------------------------- | -------- |
| 1    | torch.distributed.is_available            | 否       |
| 2    | torch.distributed.init_process_group      | 否       |
| 3    | torch.distributed.Backend                 | 否       |
| 4    | torch.distributed.get_backend             | 否       |
| 5    | torch.distributed.get_rank                | 否       |
| 6    | torch.distributed.get_world_size          | 否       |
| 7    | torch.distributed.is_initialized          | 否       |
| 8    | torch.distributed.is_mpi_available        | 否       |
| 9    | torch.distributed.is_nccl_available       | 否       |
| 10   | torch.distributed.Store                   | 否       |
| 11   | torch.distributed.TCPStore                | 否       |
| 12   | torch.distributed.HashStore               | 否       |
| 13   | torch.distributed.FileStore               | 否       |
| 14   | torch.distributed.PrefixStore             | 否       |
| 15   | torch.distributed.Store.set               | 否       |
| 16   | torch.distributed.Store.get               | 否       |
| 17   | torch.distributed.Store.add               | 否       |
| 18   | torch.distributed.Store.wait              | 否       |
| 19   | torch.distributed.Store.num_keys          | 否       |
| 20   | torch.distributed.Store.delete_key        | 否       |
| 21   | torch.distributed.Store.set_timeout       | 否       |
| 22   | torch.distributed.new_group               | 否       |
| 23   | torch.distributed.send                    | 否       |
| 24   | torch.distributed.recv                    | 否       |
| 25   | torch.distributed.isend                   | 否       |
| 26   | torch.distributed.irecv                   | 否       |
| 27   | is_completed                              | 否       |
| 28   | wait                                      | 否       |
| 29   | torch.distributed.broadcast               | 否       |
| 30   | torch.distributed.broadcast_object_list   | 否       |
| 31   | torch.distributed.all_reduce              | 否       |
| 32   | torch.distributed.reduce                  | 否       |
| 33   | torch.distributed.all_gather              | 否       |
| 34   | torch.distributed.all_gather_object       | 否       |
| 35   | torch.distributed.gather                  | 否       |
| 36   | torch.distributed.gather_object           | 否       |
| 37   | torch.distributed.scatter                 | 否       |
| 38   | torch.distributed.scatter_object_list     | 否       |
| 39   | torch.distributed.reduce_scatter          | 否       |
| 40   | torch.distributed.all_to_all              | 否       |
| 41   | torch.distributed.barrier                 | 否       |
| 42   | torch.distributed.ReduceOp                | 否       |
| 43   | torch.distributed.reduce_op               | 否       |
| 44   | torch.distributed.broadcast_multigpu      | 否       |
| 45   | torch.distributed.all_reduce_multigpu     | 否       |
| 46   | torch.distributed.reduce_multigpu         | 否       |
| 47   | torch.distributed.all_gather_multigpu     | 否       |
| 48   | torch.distributed.reduce_scatter_multigpu | 否       |
| 49   | torch.distributed.launch                  | 否       |
| 50   | torch.multiprocessing.spawn               | 否       |

## torch.npu

| 序号 | API名称                               | npu对应API名称                       | 是否支持 |
| ---- | ------------------------------------- | ------------------------------------ | -------- |
| 1    | torch.cuda.current_blas_handle        | torch.npu.current_blas_handle        | 否       |
| 2    | torch.cuda.current_device             | torch.npu.current_device             | 是       |
| 3    | torch.cuda.current_stream             | torch.npu.current_stream             | 是       |
| 4    | torch.cuda.default_stream             | torch.npu.default_stream             | 是       |
| 5    | torch.cuda.device                     | torch.npu.device                     | 否       |
| 6    | torch.cuda.device_count               | torch.npu.device_count               | 是       |
| 7    | torch.cuda.device_of                  | torch.npu.device_of                  | 否       |
| 8    | torch.cuda.get_device_capability      | torch.npu.get_device_capability      | 否       |
| 9    | torch.cuda.get_device_name            | torch.npu.get_device_name            | 否       |
| 10   | torch.cuda.init                       | torch.npu.init                       | 是       |
| 11   | torch.cuda.ipc_collect                | torch.npu.ipc_collect                | 否       |
| 12   | torch.cuda.is_available               | torch.npu.is_available               | 是       |
| 13   | torch.cuda.is_initialized             | torch.npu.is_initialized             | 是       |
| 14   | torch.cuda.set_device                 | torch.npu.set_device                 | 部分支持 |
| 15   | torch.cuda.stream                     | torch.npu.stream                     | 是       |
| 16   | torch.cuda.synchronize                | torch.npu.synchronize                | 是       |
| 17   | torch.cuda.get_rng_state              | torch.npu.get_rng_state              | 否       |
| 18   | torch.cuda.get_rng_state_all          | torch.npu.get_rng_state_all          | 否       |
| 19   | torch.cuda.set_rng_state              | torch.npu.set_rng_state              | 否       |
| 20   | torch.cuda.set_rng_state_all          | torch.npu.set_rng_state_all          | 否       |
| 21   | torch.cuda.manual_seed                | torch.npu.manual_seed                | 否       |
| 22   | torch.cuda.manual_seed_all            | torch.npu.manual_seed_all            | 否       |
| 23   | torch.cuda.seed                       | torch.npu.seed                       | 否       |
| 24   | torch.cuda.seed_all                   | torch.npu.seed_all                   | 否       |
| 25   | torch.cuda.initial_seed               | torch.npu.initial_seed               | 否       |
| 26   | torch.cuda.comm.broadcast             | torch.npu.comm.broadcast             | 否       |
| 27   | torch.cuda.comm.broadcast_coalesced   | torch.npu.comm.broadcast_coalesced   | 否       |
| 28   | torch.cuda.comm.reduce_add            | torch.npu.comm.reduce_add            | 否       |
| 29   | torch.cuda.comm.scatter               | torch.npu.comm.scatter               | 否       |
| 30   | torch.cuda.comm.gather                | torch.npu.comm.gather                | 否       |
| 31   | torch.cuda.Stream                     | torch.npu.Stream                     | 是       |
| 32   | torch.cuda.Stream.query               | torch.npu.Stream.query               | 否       |
| 33   | torch.cuda.Stream.record_event        | torch.npu.Stream.record_event        | 是       |
| 34   | torch.cuda.Stream.synchronize         | torch.npu.Stream.synchronize         | 是       |
| 35   | torch.cuda.Stream.wait_event          | torch.npu.Stream.wait_event          | 是       |
| 36   | torch.cuda.Stream.wait_stream         | torch.npu.Stream.wait_stream         | 是       |
| 37   | torch.cuda.Event                      | torch.npu.Event                      | 是       |
| 38   | torch.cuda.Event.elapsed_time         | torch.npu.Event.elapsed_time         | 是       |
| 39   | torch.cuda.Event.from_ipc_handle      | torch.npu.Event.from_ipc_handle      | 否       |
| 40   | torch.cuda.Event.ipc_handle           | torch.npu.Event.ipc_handle           | 否       |
| 41   | torch.cuda.Event.query                | torch.npu.Event.query                | 是       |
| 42   | torch.cuda.Event.record               | torch.npu.Event.record               | 是       |
| 43   | torch.cuda.Event.synchronize          | torch.npu.Event.synchronize          | 是       |
| 44   | torch.cuda.Event.wait                 | torch.npu.Event.wait                 | 是       |
| 45   | torch.cuda.empty_cache                | torch.npu.empty_cache                | 是       |
| 46   | torch.cuda.memory_stats               | torch.npu.memory_stats               | 是       |
| 47   | torch.cuda.memory_summary             | torch.npu.memory_summary             | 是       |
| 48   | torch.cuda.memory_snapshot            | torch.npu.memory_snapshot            | 是       |
| 49   | torch.cuda.memory_allocated           | torch.npu.memory_allocated           | 是       |
| 50   | torch.cuda.max_memory_allocated       | torch.npu.max_memory_allocated       | 是       |
| 51   | torch.cuda.reset_max_memory_allocated | torch.npu.reset_max_memory_allocated | 是       |
| 52   | torch.cuda.memory_reserved            | torch.npu.memory_reserved            | 是       |
| 53   | torch.cuda.max_memory_reserved        | torch.npu.max_memory_reserved        | 是       |
| 54   | torch.cuda.memory_cached              | torch.npu.memory_cached              | 是       |
| 55   | torch.cuda.max_memory_cached          | torch.npu.max_memory_cached          | 是       |
| 56   | torch.cuda.reset_max_memory_cached    | torch.npu.reset_max_memory_cached    | 是       |
| 57   | torch.cuda.nvtx.mark                  | torch.npu.nvtx.mark                  | 否       |
| 58   | torch.cuda.nvtx.range_push            | torch.npu.nvtx.range_push            | 否       |
| 59   | torch.cuda.nvtx.range_pop             | torch.npu.nvtx.range_pop             | 否       |
| 60   | torch.cuda._sleep                     | torch.npu._sleep                     | 否       |
| 61   | torch.cuda.Stream.priority_range      | torch.npu.Stream.priority_range      | 否       |
| 62   | torch.cuda.get_device_properties      | torch.npu.get_device_properties      | 否       |
| 63   | torch.cuda.amp.GradScaler             | torch.npu.amp.GradScaler             | 否       |

## NPU自定义算子

| 序号 | PyTorch 算子（由昇腾开发）                     | 昇腾适配算子                                   |
| ---- | ---------------------------------------------- | ---------------------------------------------- |
| 1    | npu_convolution_transpose                      | npu_convolution_transpose                      |
| 2    | npu_conv_transpose2d                           | conv_transpose2d_npu                           |
| 3    | npu_convolution_transpose_backward             | npu_convolution_transpose_backward             |
| 4    | npu_conv_transpose2d_backward                  | conv_transpose2d_backward_npu                  |
| 5    | npu_conv_transpose3d_backward                  | conv_transpose3d_backward_npu                  |
| 6    | npu_convolution                                | npu_convolution                                |
| 7    | npu_convolution_backward                       | npu_convolution_backward                       |
| 8    | npu_convolution_double_backward                | npu_convolution_double_backward                |
| 9    | npu_conv2d                                     | conv2d_npu                                     |
| 10   | npu_conv2d.out                                 | conv2d_out_npu                                 |
| 11   | npu_conv2d_backward                            | conv2d_backward_npu                            |
| 12   | npu_conv3d                                     | conv3d_npu                                     |
| 13   | npu_conv3d.out                                 | conv3d_out_npu                                 |
| 14   | npu_conv3d_backward                            | conv3d_backward_npu                            |
| 15   | one_                                           | one_npu_                                       |
| 16   | npu_sort_v2.out                                | sort_without_indices_out_npu                   |
| 17   | npu_sort_v2                                    | sort_without_indices_npu                       |
| 18   | npu_format_cast                                | format_cast_npu                                |
| 19   | npu_format_cast_.acl_format                    | format_cast_npu_                               |
| 20   | npu_format_cast_.src                           | format_cast_npu_                               |
| 21   | npu_transpose_to_contiguous                    | transpose_to_contiguous_npu                    |
| 22   | npu_transpose                                  | transpose_npu                                  |
| 23   | npu_transpose.out                              | transpose_out_npu                              |
| 24   | npu_broadcast                                  | broadcast_npu                                  |
| 25   | npu_broadcast.out                              | broadcast_out_npu                              |
| 26   | npu_dtype_cast                                 | dtype_cast_npu                                 |
| 27   | npu_dtype_cast_.Tensor                         | dtype_cast_npu_                                |
| 28   | npu_roi_alignbk                                | roi_align_backward_npu                         |
| 29   | empty_with_format                              | empty_with_format_npu                          |
| 30   | empty_with_format.names                        | empty_with_format_npu                          |
| 31   | copy_memory_                                   | copy_memory_npu_                               |
| 32   | npu_one_hot                                    | one_hot_npu                                    |
| 33   | npu_stride_add                                 | stride_add_npu                                 |
| 34   | npu_softmax_cross_entropy_with_logits          | softmax_cross_entropy_with_logits_npu          |
| 35   | npu_softmax_cross_entropy_with_logits_backward | softmax_cross_entropy_with_logits_backward_npu |
| 36   | npu_ps_roi_pooling                             | ps_roi_pooling_npu                             |
| 37   | npu_ps_roi_pooling_backward                    | ps_roi_pooling_backward_npu                    |
| 38   | npu_roi_align                                  | roi_align_npu                                  |
| 39   | npu_nms_v4                                     | nms_v4_npu                                     |
| 40   | npu_lstm                                       | lstm_npu                                       |
| 41   | npu_lstm_backward                              | lstm_backward_npu                              |
| 42   | npu_iou                                        | iou_npu                                        |
| 43   | npu_ptiou                                      | ptiou_npu                                      |
| 44   | npu_nms_with_mask                              | nms_with_mask_npu                              |
| 45   | npu_pad                                        | pad_npu                                        |
| 46   | npu_bounding_box_encode                        | bounding_box_encode_npu                        |
| 47   | npu_bounding_box_decode                        | bounding_box_decode_npu                        |
| 48   | npu_gru                                        | gru_npu                                        |
| 49   | npu_gru_backward                               | gru_backward_npu                               |
| 50   | npu_set_.source_Storage_storage_offset_format  | set_npu_                                       |
| 51   | npu_random_choice_with_mask                    | random_choice_with_mask_npu                    |
| 52   | npu_batch_nms                                  | batch_nms_npu                                  |
| 53   | npu_slice                                      | slice_npu                                      |
| 54   | npu_slice.out                                  | slice_out_npu                                  |
| 55   | npu_dropoutV2                                  | dropout_v2_npu                                 |
| 56   | npu_dropoutV2_backward                         | dropout_v2_backward_npu                        |
| 57   | _npu_dropout                                   | _dropout_npu                                   |
| 58   | _npu_dropout_inplace                           | _dropout_npu_inplace                           |
| 59   | npu_dropout_backward                           | dropout_backward_npu                           |
| 60   | npu_indexing                                   | indexing_npu                                   |
| 61   | npu_indexing.out                               | indexing_out_npu                               |
| 62   | npu_ifmr                                       | ifmr_npu                                       |
| 63   | npu_max.dim                                    | max_v1_npu                                     |
| 64   | npu_max.names_dim                              | max_v1_npu                                     |
| 65   | npu_scatter                                    | scatter_npu                                    |
| 66   | npu_max_backward                               | max_backward_npu                               |
| 67   | npu_apply_adam                                 | apply_adam_npu                                 |
| 68   | npu_layer_norm_eval                            | layer_norm_eval_npu                            |
| 69   | npu_alloc_float_status                         | alloc_float_status_npu                         |
| 70   | npu_get_float_status                           | get_float_status_npu                           |
| 71   | npu_clear_float_status                         | clear_float_status_npu                         |
| 72   | npu_confusion_transpose                        | confusion_transpose_npu                        |
| 73   | npu_confusion_transpose_backward               | confusion_transpose_backward_npu               |
| 74   | npu_bmmV2                                      | bmm_v2_npu                                     |
| 75   | fast_gelu                                      | fast_gelu_npu                                  |
| 76   | fast_gelu_backward                             | fast_gelu_backward_npu                         |
| 77   | npu_sub_sample                                 | sub_sample_npu                                 |
| 78   | npu_deformable_conv2d                          | deformable_conv2d_npu                          |
| 79   | npu_deformable_conv2dbk                        | deformable_conv2d_backward_npu                 |
| 80   | npu_mish                                       | mish_npu                                       |
| 81   | npu_anchor_response_flags                      | anchor_response_flags_npu                      |
| 82   | npu_yolo_boxes_encode                          | yolo_boxes_encode_npu                          |
| 83   | npu_grid_assign_positive                       | grid_assign_positive_npu                       |
| 84   | npu_mish_backward                              | mish_backward_npu                              |
| 85   | npu_normalize_batch                            | normalize_batch_npu                            |
| 86   | npu_masked_fill_range                          | masked_fill_range_npu                          |
| 87   | npu_linear                                     | linear_npu                                     |
| 88   | npu_linear_backward                            | linear_backward_npu                            |
| 89   | npu_bert_apply_adam                            | bert_apply_adam_npu                            |
| 90   | npu_giou                                       | giou_npu                                       |
| 91   | npu_giou_backward                              | giou_backward_npu                              |

算子接口算子接口说明：

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
  
  

