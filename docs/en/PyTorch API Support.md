## [Tensors](https://pytorch.org/docs/1.8.1/torch.html)

| No.| API                      |                Supported/Unsupported               |
| ---- | ----------------------------- | :------------------------------------: |
| 1    | torch.is_tensor               |                   Supported                  |
| 2    | torch.is_storage              |                   Supported                  |
| 3    | torch.is_complex              | Supported. The judgment is supported, but the complex number is not supported by the current hardware.|
| 4    | torch.is_floating_point       |                   Supported                  |
| 5    | torch.is_nonzero              |                   Supported                  |
| 6    | torch.set_default_dtype       | Supported (NPU data type are not supported)                  |
| 7    | torch.get_default_dtype       | Supported (NPU data type are not supported)                  |
| 8    | torch.set_default_tensor_type | Supported (NPU data type are not supported)                  |
| 9    | torch.numel                   |                   Supported                  |
| 10   | torch.set_printoptions        |                   Supported                  |
| 11   | torch.set_flush_denormal      |                   Supported                  |
| 12   | torch.tensor                  |                   Supported                  |
| 13   | torch.sparse_coo_tensor       |                   Unsupported                  |
| 14   | torch.as_tensor               |                   Supported                  |
| 15   | torch.as_strided              |                   Supported                  |
| 16   | torch.from_numpy              |                   Supported                  |
| 17   | torch.zeros                   |                   Supported                  |
| 18   | torch.zeros_like              |                   Supported                  |
| 19   | torch.ones                    |                   Supported                  |
| 20   | torch.ones_like               |                   Supported                  |
| 21   | torch.arange                  |                   Supported                  |
| 22   | torch.range                   |                   Supported                  |
| 23   | torch.linspace                |                   Supported                  |
| 24   | torch.logspace                |                   Supported                  |
| 25   | torch.eye                     |                   Supported                  |
| 26   | torch.empty                   |                   Supported                  |
| 27   | torch.empty_like              |                   Supported                  |
| 28   | torch.empty_strided           |                   Supported                  |
| 29   | torch.full                    |                   Supported                  |
| 30   | torch.full_like               |                   Supported                  |
| 31   | torch.quantize_per_tensor     |                   Supported                  |
| 32   | torch.quantize_per_channel    |                   Supported                  |
| 33   | torch.dequantize              |                   Unsupported                  |
| 34   | torch.complex                 |                   Supported                  |
| 35   | torch.polar                   |                   Supported                  |
| 36   | torch.heaviside               |                   Unsupported                  |
| 37   | torch.cat                     |                   Supported                  |
| 38   | torch.chunk                   |                   Supported                  |
| 39   | torch.column_stack            |                   Supported                  |
| 40   | torch.dstack                  |                   Supported                  |
| 41   | torch.hstack                  |                   Supported                  |
| 42   | torch.gather                  |                   Supported                  |
| 43   | torch.index_select            |                   Supported                  |
| 44   | torch.masked_select           |                   Supported                  |
| 45   | torch.movedim                 |                   Supported                  |
| 46   | torch.moveaxis                |                   Supported                  |
| 47   | torch.narrow                  |                   Supported                  |
| 48   | torch.nonzero                 |                   Supported                  |
| 49   | torch.reshape                 |                   Supported                  |
| 50   | torch.row_stack               |                   Supported                  |
| 51   | torch.scatter                 |                   Supported                  |
| 52   | torch.scatter_add             |                   Supported                  |
| 53   | torch.split                   |                   Supported                  |
| 54   | torch.squeeze                 |                   Supported                  |
| 55   | torch.stack                   |                   Supported                  |
| 56   | torch.swapaxes                |                   Supported                  |
| 57   | torch.swapdims                |                   Supported                  |
| 58   | torch.t                       |                   Supported                  |
| 59   | torch.take                    |                   Supported                  |
| 60   | torch.tensor_split            |                   Supported                  |
| 61   | torch.tile                    |                   Supported                  |
| 62   | torch.transpose               |                   Supported                  |
| 63   | torch.unbind                  |                   Supported                  |
| 64   | torch.unsqueeze               |                   Supported                  |
| 65   | torch.vstack                  |                   Supported                  |
| 66   | torch.where                   |                   Supported                  |

## Generators

| No.| API                         | Supported/Unsupported|
| ---- | -------------------------------- | -------- |
| 1    | torch._C.Generator               | Supported      |
| 2    | torch.*C.torch.default*generator | Supported      |

## Random sampling

| No.| API                      | Supported/Unsupported|
| ---- | ----------------------------- | -------- |
| 1    | torch.seed                    | Supported      |
| 2    | torch.manual_seed             | Supported      |
| 3    | torch.initial_seed            | Supported      |
| 4    | torch.get_rng_state           | Supported      |
| 5    | torch.set_rng_state           | Supported      |
| 7    | torch.bernoulli               | Supported      |
| 8    | torch.multinomial             | Supported      |
| 9    | torch.normal                  | Supported      |
| 10   | torch.poisson                 | Unsupported      |
| 11   | torch.rand                    | Supported      |
| 12   | torch.rand_like               | Supported      |
| 13   | torch.randint                 | Supported      |
| 14   | torch.randint_like            | Supported      |
| 15   | torch.randn                   | Supported      |
| 16   | torch.randn_like              | Supported      |
| 17   | torch.randperm                | Supported      |
| 18   | torch.Tensor.bernoulli        | Supported      |
| 19   | torch.Tensor.bernoulli_       | Supported      |
| 20   | torch.Tensor.cauchy_          | Supported      |
| 21   | torch.Tensor.exponential_     | Unsupported      |
| 22   | torch.Tensor.geometric_       | Unsupported      |
| 23   | torch.Tensor.log_normal_      | Unsupported      |
| 24   | torch.Tensor.normal_          | Supported      |
| 25   | torch.Tensor.random_          | Supported      |
| 26   | torch.Tensor.uniform_         | Supported      |
| 27   | torch.quasirandom.SobolEngine | Supported      |

## Serialization

| No.| API   | Supported/Unsupported|
| ---- | ---------- | -------- |
| 1    | torch.save | Supported      |
| 2    | torch.load | Supported      |

## Math operations

| No.| API                               | Supported/Unsupported                                                    |
| ---- | -------------------------------------- | ------------------------------------------------------------ |
| 1    | torch.abs                              | Supported                                                          |
| 2    | torch.absolute                         | Supported                                                          |
| 3    | torch.acos                             | Supported                                                          |
| 4    | torch.arccos                           | Unsupported                                                          |
| 5    | torch.acosh                            | Supported                                                          |
| 6    | torch.arccosh                          | Supported                                                          |
| 7    | torch.add                              | Supported                                                          |
| 8    | torch.addcdiv                          | Supported                                                          |
| 9    | torch.addcmul                          | Supported                                                          |
| 10   | torch.angle                            | Unsupported                                                          |
| 11   | torch.asin                             | Supported                                                          |
| 12   | torch.arcsin                           | Supported                                                          |
| 13   | torch.asinh                            | Supported                                                          |
| 14   | torch.arcsinh                          | Supported                                                          |
| 15   | torch.atan                             | Supported                                                          |
| 16   | torch.arctan                           | Supported                                                          |
| 17   | torch.atanh                            | Supported                                                          |
| 18   | torch.arctanh                          | Supported                                                          |
| 19   | torch.atan2                            | Supported                                                          |
| 20   | torch.bitwise_not                      | Supported                                                          |
| 21   | torch.bitwise_and                      | Supported                                                          |
| 22   | torch.bitwise_or                       | Supported                                                          |
| 23   | torch.bitwise_xor                      | Supported                                                          |
| 24   | torch.ceil                             | Supported                                                          |
| 25   | torch.clamp                            | Supported                                                          |
| 26   | torch.clip                             | Supported                                                          |
| 27   | torch.conj                             | Unsupported                                                          |
| 28   | torch.copysign                         | Supported                                                          |
| 29   | torch.cos                              | Supported                                                          |
| 30   | torch.cosh                             | Supported                                                          |
| 31   | torch.deg2rad                          | Supported                                                          |
| 32   | torch.div                              | Supported                                                          |
| 33   | torch.divide                           | Supported                                                          |
| 34   | torch.digamma                          | Supported                                                          |
| 35   | torch.erf                              | Supported                                                          |
| 36   | torch.erfc                             | Supported                                                          |
| 37   | torch.erfinv                           | Supported                                                          |
| 38   | torch.exp                              | Supported                                                          |
| 39   | torch.exp2                             | Unsupported                                                          |
| 40   | torch.expm1                            | Supported                                                          |
| 41   | torch.fake_quantize_per_channel_affine | Unsupported                                                          |
| 42   | torch.fake_quantize_per_tensor_affine  | Unsupported                                                          |
| 43   | torch.fix                              | Supported                                                          |
| 44   | torch.float_power                      | Supported                                                          |
| 45   | torch.floor                            | Supported                                                          |
| 46   | torch.floor_divide                     | Supported                                                          |
| 47   | torch.fmod                             | Supported                                                          |
| 48   | torch.frac                             | Supported                                                          |
| 49   | torch.imag                             | Unsupported                                                          |
| 50   | torch.ldexp                            | Supported                                                          |
| 51   | torch.lerp                             | Supported                                                          |
| 52   | torch.lgamma                           | Supported                                                          |
| 53   | torch.log                              | Supported                                                          |
| 54   | torch.log10                            | Supported                                                          |
| 55   | torch.log1p                            | Supported                                                          |
| 56   | torch.log2                             | Supported                                                          |
| 57   | torch.logaddexp                        | Unsupported                                                          |
| 58   | torch.logaddexp2                       | Unsupported                                                          |
| 59   | torch.logical_and                      | Supported                                                          |
| 60   | torch.logical_not                      | Supported                                                          |
| 61   | torch.logical_or                       | Supported                                                          |
| 62   | torch.logical_xor                      | Unsupported                                                          |
| 63   | torch.logit                            | Supported                                                          |
| 64   | torch.hypot                            | Supported                                                          |
| 65   | torch.i0                               | Unsupported                                                          |
| 66   | torch.igamma                           | Supported                                                          |
| 67   | torch.igammac                          | Supported                                                          |
| 68   | torch.mul                              | Supported                                                          |
| 69   | torch.multiply                         | Supported                                                          |
| 70   | torch.mvlgamma                         | Supported                                                          |
| 71   | torch.nan_to_num                       | Unsupported                                                          |
| 72   | torch.neg                              | Supported                                                          |
| 73   | torch.negative                         | Supported                                                          |
| 74   | torch.nextafter                        | Unsupported                                                          |
| 75   | torch.polygamma                        | Unsupported                                                          |
| 76   | torch.pow                              | Supported                                                          |
| 77   | torch.rad2deg                          | Supported                                                          |
| 78   | torch.real                             | Supported                                                          |
| 79   | torch.reciprocal                       | Supported                                                          |
| 80   | torch.remainder                        | Supported                                                          |
| 81   | torch.round                            | Supported                                                          |
| 82   | torch.rsqrt                            | Supported                                                          |
| 83   | torch.sigmoid                          | Supported                                                          |
| 84   | torch.sign                             | Supported                                                          |
| 85   | torch.sgn                              | Unsupported                                                          |
| 86   | torch.signbit                          | Unsupported                                                          |
| 87   | torch.sin                              | Supported                                                          |
| 88   | torch.sinc                             | Unsupported                                                          |
| 89   | torch.sinh                             | Supported                                                          |
| 90   | torch.sqrt                             | Supported                                                          |
| 91   | torch.square                           | Supported                                                          |
| 92   | torch.sub                              | Supported                                                          |
| 93   | torch.subtract                         | Supported                                                          |
| 94   | torch.tan                              | Supported                                                          |
| 95   | torch.tanh                             | Supported                                                          |
| 96   | torch.true_divide                      | Supported                                                          |
| 97   | torch.trunc                            | Supported                                                          |
| 98   | torch.xlogy                            | Supported                                                          |
| 99   | torch.argmax                           | Supported                                                          |
| 100  | torch.argmin                           | Supported                                                          |
| 101  | torch.amax                             | Unsupported                                                          |
| 102  | torch.amin                             | Unsupported                                                          |
| 103  | torch.all                              | Supported                                                          |
| 104  | torch.any                              | Supported                                                          |
| 105  | torch.max                              | Supported                                                          |
| 106  | torch.min                              | Supported                                                          |
| 107  | torch.dist                             | Supported                                                          |
| 108  | torch.logsumexp                        | Supported                                                          |
| 109  | torch.mean                             | Supported                                                          |
| 110  | torch.median                           | Supported                                                          |
| 111  | torch.nanmedian                        | Supported                                                          |
| 112  | torch.mode                             | Supported                                                          |
| 113  | torch.norm                             | Supported                                                          |
| 114  | torch.nansum                           | Supported                                                          |
| 115  | torch.prod                             | Supported                                                          |
| 116  | torch.quantile                         | Unsupported                                                          |
| 117  | torch.nanquantile                      | Unsupported                                                          |
| 118  | torch.std                              | Supported                                                          |
| 119  | torch.std_mean                         | Supported                                                          |
| 120  | torch.sum                              | Supported                                                          |
| 121  | torch.unique                           | Supported                                                          |
| 122  | torch.unique_consecutive               | Supported. Use keywords during parameter transfer. Otherwise, the accuracy does not meet the requirements.return_inverse=return_inverse,return_counts=return_counts,dim=dim |
| 123  | torch.var                              | Unsupported                                                          |
| 124  | torch.var_mean                         | Unsupported                                                          |
| 125  | torch.count_nonzero                    | Unsupported                                                          |
| 126  | torch.allclose                         | Supported                                                          |
| 127  | torch.argsort                          | Supported                                                          |
| 128  | torch.eq                               | Supported                                                          |
| 129  | torch.equal                            | Supported                                                          |
| 130  | torch.ge                               | Supported                                                          |
| 131  | torch.greater_equal                    | Supported                                                          |
| 132  | torch.gt                               | Supported                                                          |
| 133  | torch.greater                          | Supported                                                          |
| 134  | torch.isclose                          | Unsupported                                                          |
| 135  | torch.isfinite                         | Supported                                                          |
| 136  | torch.isinf                            | Supported                                                          |
| 137  | torch.isposinf                         | Supported                                                          |
| 138  | torch.isneginf                         | Supported                                                          |
| 139  | torch.isnan                            | Supported                                                          |
| 140  | torch.isreal                           | Supported                                                          |
| 141  | torch.kthvalue                         | Supported                                                          |
| 142  | torch.le                               | Supported                                                          |
| 143  | torch.less_equal                       | Supported                                                          |
| 144  | torch.lt                               | Supported                                                          |
| 145  | torch.less                             | Supported                                                          |
| 146  | torch.maximum                          | Supported                                                          |
| 147  | torch.minimum                          | Supported                                                          |
| 148  | torch.fmax                             | Unsupported                                                          |
| 149  | torch.fmin                             | Unsupported                                                          |
| 150  | torch.ne                               | Supported                                                          |
| 151  | torch.not_equal                        | Supported                                                          |
| 152  | torch.sort                             | Supported                                                          |
| 153  | torch.topk                             | Supported                                                          |
| 154  | torch.msort                            | Unsupported                                                          |
| 155  | torch.fft                              | Unsupported                                                          |
| 156  | torch.ifft                             | Unsupported                                                          |
| 157  | torch.rfft                             | Unsupported                                                          |
| 158  | torch.irfft                            | Unsupported                                                          |
| 159  | torch.stft                             | Unsupported                                                          |
| 160  | torch.istft                            | Unsupported                                                          |
| 161  | torch.bartlett_window                  | Supported                                                          |
| 162  | torch.blackman_window                  | Supported                                                          |
| 163  | torch.hamming_window                   | Supported                                                          |
| 164  | torch.hann_window                      | Supported                                                          |
| 165  | torch.kasier_window                    | Unsupported                                                          |
| 166  | torch.atleast_1d                       | Supported                                                          |
| 167  | torch.atleast_2d                       | Supported                                                          |
| 168  | torch.atleast_3d                       | Supported                                                          |
| 169  | torch.bincount                         | Supported                                                          |
| 170  | torch.block_diag                       | Supported                                                          |
| 171  | torch.broadcast_tensors                | Supported                                                          |
| 172  | torch.broadcast_to                     | Supported                                                          |
| 173  | torch.broadcast_shapes                 | Supported                                                          |
| 174  | torch.bucketize                        | Supported                                                          |
| 175  | torch.cartesian_prod                   | Supported                                                          |
| 176  | torch.cdist                            | Supported. Only **mode=donot_use_mm_for_euclid_dist** is supported.                 |
| 177  | torch.clone                            | Supported                                                          |
| 178  | torch.combinations                     | Supported                                                          |
| 179  | torch.cross                            | Supported                                                          |
| 180  | torch.cummax                           | Supported                                                          |
| 181  | torch.cummin                           | Supported                                                          |
| 182  | torch.cumprod                          | Supported                                                          |
| 183  | torch.cumsum                           | Supported                                                          |
| 184  | torch.diag                             | Supported only in the diagonal=0 scenario.                                    |
| 185  | torch.diag_embed                       | Supported                                                          |
| 186  | torch.diagflat                         | Supported                                                          |
| 187  | torch.diagonal                         | Supported                                                          |
| 188  | torch.diff                             | Supported                                                          |
| 189  | torch.einsum                           | Supported                                                          |
| 190  | torch.flatten                          | Supported                                                          |
| 191  | torch.flip                             | Supported                                                          |
| 192  | torch.fliplr                           | Supported                                                          |
| 193  | torch.flipud                           | Supported                                                          |
| 194  | torch.kron                             | Supported                                                          |
| 195  | torch.rot90                            | Supported                                                          |
| 196  | torch.gcd                              | Unsupported                                                          |
| 197  | torch.histc                            | Unsupported                                                          |
| 198  | torch.meshgrid                         | Supported                                                          |
| 199  | torch.lcm                              | Unsupported                                                          |
| 200  | torhc.logcumsumexp                     | Unsupported                                                          |
| 201  | torch.ravel                            | Supported                                                          |
| 202  | torch.renorm                           | Supported                                                          |
| 203  | torch.repeat_interleave                | Supported                                                          |
| 204  | torch.roll                             | Supported                                                          |
| 205  | torch.searchsorted                     | Supported                                                          |
| 206  | torch.tensordot                        | Supported                                                          |
| 207  | torch.trace                            | Unsupported                                                          |
| 208  | torch.tril                             | Supported                                                          |
| 209  | torch.tril_indices                     | Supported                                                          |
| 210  | torch.triu                             | Supported                                                          |
| 211  | torch.triu_indices                     | Supported                                                          |
| 212  | torch.vander                           | Supported                                                          |
| 213  | torch.view_as_real                     | Supported                                                          |
| 214  | torch.view_as_complex                  | Supported                                                          |
| 215  | torch.addbmm                           | Supported                                                          |
| 216  | torch.addmm                            | Supported                                                          |
| 217  | torch.addmv                            | Supported                                                          |
| 218  | torch.addr                             | Supported                                                          |
| 219  | torch.baddbmm                          | Supported                                                          |
| 220  | torch.bmm                              | Supported                                                          |
| 221  | torch.chain_matmul                     | Supported                                                          |
| 222  | torch.cholesky                         | Unsupported                                                          |
| 223  | torch.cholesky_inverse                 | Unsupported                                                          |
| 224  | torch.cholesky_solve                   | Unsupported                                                          |
| 225  | torch.dot                              | Supported                                                          |
| 226  | torch.eig                              | Unsupported                                                          |
| 227  | torch.geqrf                            | Unsupported                                                          |
| 228  | torch.ger                              | Supported                                                          |
| 229  | torch.inner                            | Supported                                                          |
| 230  | torch.inverse                          | Supported                                                          |
| 231  | torch.det                              | Unsupported                                                          |
| 232  | torch.logdet                           | Unsupported                                                          |
| 233  | torch.slogdet                          | Supported                                                          |
| 234  | torch.lstsq                            | Unsupported                                                          |
| 235  | torch.lu                               | Unsupported                                                          |
| 236  | torch.lu_solve                         | Unsupported                                                          |
| 237  | torch.lu_unpack                        | Unsupported                                                          |
| 238  | torch.matmul                           | Supported                                                          |
| 239  | torch.matrix_power                     | Supported                                                          |
| 240  | torch.matrix_rank                      | Supported                                                          |
| 241  | torch.matrix_exp                       | Supported                                                          |
| 242  | torch.mm                               | Supported                                                          |
| 243  | torch.mv                               | Supported                                                          |
| 244  | torch.orgqr                            | Unsupported                                                          |
| 245  | torch.ormqr                            | Unsupported                                                          |
| 246  | torch.outer                            | Supported                                                          |
| 247  | torch.pinverse                         | Supported                                                          |
| 248  | torch.qr                               | Supported                                                          |
| 249  | torch.solve                            | Unsupported                                                          |
| 250  | torch.svd                              | Supported                                                          |
| 251  | torch.svd_lowrank                      | Supported                                                          |
| 252  | torch.pca_lowrank                      | Supported                                                          |
| 253  | torch.symeig                           | Supported                                                          |
| 254  | torch.lobpcg                           | Unsupported                                                          |
| 255  | torch.trapz                            | Supported                                                          |
| 256  | torch.triangular_solve                 | Supported                                                          |
| 257  | torch.vdot                             | Supported                                                          |

## Utilities

| No.| API                                   | Supported/Unsupported|
| ---- | ------------------------------------------ | -------- |
| 1    | torch.compiled_with_cxx11_abi              | Supported      |
| 2    | torch.result_type                          | Supported      |
| 3    | torch.can_cast                             | Supported      |
| 4    | torch.promote_types                        | Supported      |
| 6    | torch.use_deterministic_algorithms         | Supported      |
| 7    | torch.are_deterministic_algorithms_enabled | Supported      |
| 8    | torch._assert                              | Supported      |

## Other

| No.| API                      | Supported/Unsupported|
| ---- | ----------------------------- | -------- |
| 1    | torch.no_grad                 | Supported      |
| 2    | torch.enable_grad             | Supported      |
| 3    | torch.set_grad_enabled        | Supported      |
| 4    | torch.get_num_threads         | Supported      |
| 5    | torch.set_num_threads         | Supported      |
| 6    | torch.get_num_interop_threads | Supported      |
| 7    | torch.set_num_interop_threads | Supported      |

## torch.Tensor

| No.| API                               | Supported/Unsupported|
| :--- | -------------------------------------- | -------- |
| 1    | torch.Tensor                           | Supported      |
| 2    | torch.Tensor.new_tensor                | Supported      |
| 3    | torch.Tensor.new_full                  | Supported      |
| 4    | torch.Tensor.new_empty                 | Supported      |
| 5    | torch.Tensor.new_ones                  | Supported      |
| 6    | torch.Tensor.new_zeros                 | Supported      |
| 7    | torch.Tensor.is_cuda                   | Supported      |
| 8    | torch.Tensor.is_quantized              | Supported      |
| 9    | torch.Tensor.device                    | Supported      |
| 10   | torch.Tensor.ndim                      | Supported      |
| 11   | torch.Tensor.T                         | Supported      |
| 12   | torch.Tensor.abs                       | Supported      |
| 13   | torch.Tensor.abs_                      | Supported      |
| 14   | torch.Tensor.acos                      | Supported      |
| 15   | torch.Tensor.acos_                     | Supported      |
| 16   | torch.Tensor.add                       | Supported      |
| 17   | torch.Tensor.add_                      | Supported      |
| 18   | torch.Tensor.addbmm                    | Supported      |
| 19   | torch.Tensor.addbmm_                   | Supported      |
| 20   | torch.Tensor.addcdiv                   | Supported      |
| 21   | torch.Tensor.addcdiv_                  | Supported      |
| 22   | torch.Tensor.addcmul                   | Supported      |
| 23   | torch.Tensor.addcmul_                  | Supported      |
| 24   | torch.Tensor.addmm                     | Supported      |
| 25   | torch.Tensor.addmm_                    | Supported      |
| 26   | torch.Tensor.addmv                     | Supported      |
| 27   | torch.Tensor.addmv_                    | Supported      |
| 28   | torch.Tensor.addr                      | Supported      |
| 29   | torch.Tensor.addr_                     | Supported      |
| 30   | torch.Tensor.allclose                  | Supported      |
| 31   | torch.Tensor.angle                     | Unsupported      |
| 32   | torch.Tensor.apply_                    | Unsupported      |
| 33   | torch.Tensor.argmax                    | Supported      |
| 34   | torch.Tensor.argmin                    | Supported      |
| 35   | torch.Tensor.argsort                   | Supported      |
| 36   | torch.Tensor.asin                      | Supported      |
| 37   | torch.Tensor.asin_                     | Supported      |
| 38   | torch.Tensor.as_strided                | Supported      |
| 39   | torch.Tensor.atan                      | Supported      |
| 40   | torch.Tensor.atan2                     | Supported      |
| 41   | torch.Tensor.atan2_                    | Supported      |
| 42   | torch.Tensor.atan_                     | Supported      |
| 43   | torch.Tensor.baddbmm                   | Supported      |
| 44   | torch.Tensor.baddbmm_                  | Supported      |
| 45   | torch.Tensor.bernoulli                 | Supported      |
| 46   | torch.Tensor.bernoulli_                | Supported      |
| 47   | torch.Tensor.bfloat16                  | Unsupported      |
| 48   | torch.Tensor.bincount                  | Supported      |
| 49   | torch.Tensor.bitwise_not               | Supported      |
| 50   | torch.Tensor.bitwise_not_              | Supported      |
| 51   | torch.Tensor.bitwise_and               | Supported      |
| 52   | torch.Tensor.bitwise_and_              | Supported      |
| 53   | torch.Tensor.bitwise_or                | Supported      |
| 54   | torch.Tensor.bitwise_or_               | Supported      |
| 55   | torch.Tensor.bitwise_xor               | Supported      |
| 56   | torch.Tensor.bitwise_xor_              | Supported      |
| 57   | torch.Tensor.bmm                       | Supported      |
| 58   | torch.Tensor.bool                      | Supported      |
| 59   | torch.Tensor.byte                      | Supported      |
| 60   | torch.Tensor.cauchy_                   | Unsupported      |
| 61   | torch.Tensor.ceil                      | Supported      |
| 62   | torch.Tensor.ceil_                     | Supported      |
| 63   | torch.Tensor.char                      | Supported      |
| 64   | torch.Tensor.chain_matul               | Supported      |
| 65   | torch.Tensor.cholesky                  | Unsupported      |
| 66   | torch.Tensor.cholesky_inverse          | Unsupported      |
| 67   | torch.Tensor.cholesky_solve            | Unsupported      |
| 68   | torch.Tensor.chunk                     | Supported      |
| 69   | torch.Tensor.clamp                     | Supported      |
| 70   | torch.Tensor.clamp_                    | Supported      |
| 71   | torch.Tensor.clone                     | Supported      |
| 72   | torch.Tensor.contiguous                | Supported      |
| 73   | torch.Tensor.copy_                     | Supported      |
| 74   | torch.Tensor.conj                      | Unsupported      |
| 75   | torch.Tensor.cos                       | Supported      |
| 76   | torch.Tensor.cos_                      | Supported      |
| 77   | torch.Tensor.cosh                      | Supported      |
| 78   | torch.Tensor.cosh_                     | Supported      |
| 79   | torch.Tensor.cpu                       | Supported      |
| 80   | torch.Tensor.cross                     | Supported      |
| 81   | torch.Tensor.cuda                      | Unsupported      |
| 82   | torch.Tensor.cummax                    | Supported      |
| 83   | torch.Tensor.cummin                    | Supported      |
| 84   | torch.Tensor.cumprod                   | Supported      |
| 85   | torch.Tensor.cumsum                    | Supported      |
| 86   | torch.Tensor.data_ptr                  | Supported      |
| 87   | torch.Tensor.dequantize                | Unsupported      |
| 88   | torch.Tensor.det                       | Unsupported      |
| 89   | torch.Tensor.dense_dim                 | Unsupported      |
| 90   | torch.Tensor.diag                      | Supported      |
| 91   | torch.Tensor.diag_embed                | Supported      |
| 92   | torch.Tensor.diagflat                  | Supported      |
| 93   | torch.Tensor.diagonal                  | Supported      |
| 94   | torch.Tensor.fill_diagonal_            | Supported      |
| 95   | torch.Tensor.digamma                   | Unsupported      |
| 96   | torch.Tensor.digamma_                  | Unsupported      |
| 97   | torch.Tensor.dim                       | Supported      |
| 98   | torch.Tensor.dist                      | Supported      |
| 99   | torch.Tensor.div                       | Supported      |
| 100  | torch.Tensor.div_                      | Supported      |
| 101  | torch.Tensor.dot                       | Supported      |
| 102  | torch.Tensor.double                    | Unsupported      |
| 103  | torch.Tensor.eig                       | Unsupported      |
| 104  | torch.Tensor.element_size              | Supported      |
| 105  | torch.Tensor.eq                        | Supported      |
| 106  | torch.Tensor.eq_                       | Supported      |
| 107  | torch.Tensor.equal                     | Supported      |
| 108  | torch.Tensor.erf                       | Supported      |
| 109  | torch.Tensor.erf_                      | Supported      |
| 110  | torch.Tensor.erfc                      | Supported      |
| 111  | torch.Tensor.erfc_                     | Supported      |
| 112  | torch.Tensor.erfinv                    | Supported      |
| 113  | torch.Tensor.erfinv_                   | Supported      |
| 114  | torch.Tensor.exp                       | Supported      |
| 115  | torch.Tensor.exp_                      | Supported      |
| 116  | torch.Tensor.expm1                     | Supported      |
| 117  | torch.Tensor.expm1_                    | Supported      |
| 118  | torch.Tensor.expand                    | Supported      |
| 119  | torch.Tensor.expand_as                 | Supported      |
| 120  | torch.Tensor.exponential_              | Unsupported      |
| 121  | torch.Tensor.fft                       | Unsupported      |
| 122  | torch.Tensor.fill_                     | Supported      |
| 123  | torch.Tensor.flatten                   | Supported      |
| 124  | torch.Tensor.flip                      | Supported      |
| 125  | torch.Tensor.float                     | Supported      |
| 126  | torch.Tensor.floor                     | Supported      |
| 127  | torch.Tensor.floor_                    | Supported      |
| 128  | torch.Tensor.floor_divide              | Supported      |
| 129  | torch.Tensor.floor_divide_             | Supported      |
| 130  | torch.Tensor.fmod                      | Supported      |
| 131  | torch.Tensor.fmod_                     | Supported      |
| 132  | torch.Tensor.frac                      | Supported      |
| 133  | torch.Tensor.frac_                     | Supported      |
| 134  | torch.Tensor.gather                    | Supported      |
| 135  | torch.Tensor.ge                        | Supported      |
| 136  | torch.Tensor.ge_                       | Supported      |
| 137  | torch.Tensor.geometric_                | Unsupported      |
| 138  | torch.Tensor.geqrf                     | Unsupported      |
| 139  | torch.Tensor.ger                       | Supported      |
| 140  | torch.Tensor.get_device                | Supported      |
| 141  | torch.Tensor.gt                        | Supported      |
| 142  | torch.Tensor.gt_                       | Supported      |
| 143  | torch.Tensor.half                      | Supported      |
| 144  | torch.Tensor.hardshrink                | Supported      |
| 145  | torch.Tensor.histc                     | Unsupported      |
| 146  | torch.Tensor.ifft                      | Unsupported      |
| 147  | torch.Tensor.index_add_                | Supported      |
| 148  | torch.Tensor.index_add                 | Supported      |
| 149  | torch.Tensor.index_copy_               | Supported      |
| 150  | torch.Tensor.index_copy                | Supported      |
| 151  | torch.Tensor.index_fill_               | Supported      |
| 152  | torch.Tensor.index_fill                | Supported      |
| 153  | torch.Tensor.index_put_                | Supported      |
| 154  | torch.Tensor.index_put                 | Supported      |
| 155  | torch.Tensor.index_select              | Supported      |
| 156  | torch.Tensor.indices                   | Unsupported      |
| 157  | torch.Tensor.int                       | Supported      |
| 158  | torch.Tensor.int_repr                  | Unsupported      |
| 159  | torch.Tensor.inner                     | Supported      |
| 160  | torch.Tensor.inverse                   | Supported      |
| 161  | torch.Tensor.irfft                     | Unsupported      |
| 162  | torch.Tensor.is_contiguous             | Supported      |
| 163  | torch.Tensor.is_complex                | Supported      |
| 164  | torch.Tensor.is_floating_point         | Supported      |
| 165  | torch.Tensor.is_pinned                 | Supported      |
| 166  | torch.Tensor.is_set_to                 | Unsupported      |
| 167  | torch.Tensor.is_shared                 | Supported      |
| 168  | torch.Tensor.is_signed                 | Supported      |
| 169  | torch.Tensor.is_sparse                 | Supported      |
| 170  | torch.Tensor.item                      | Supported      |
| 171  | torch.Tensor.kthvalue                  | Supported      |
| 172  | torch.Tensor.le                        | Supported      |
| 173  | torch.Tensor.le_                       | Supported      |
| 174  | torch.Tensor.lerp                      | Supported      |
| 175  | torch.Tensor.lerp_                     | Supported      |
| 176  | torch.Tensor.lgamma                    | Unsupported      |
| 177  | torch.Tensor.lgamma_                   | Unsupported      |
| 178  | torch.Tensor.lobpcg                    | Supported      |
| 179  | torch.Tensor.log                       | Supported      |
| 180  | torch.Tensor.log_                      | Supported      |
| 181  | torch.Tensor.logdet                    | Unsupported      |
| 182  | torch.Tensor.log10                     | Supported      |
| 183  | torch.Tensor.log10_                    | Supported      |
| 184  | torch.Tensor.log1p                     | Supported      |
| 185  | torch.Tensor.log1p_                    | Supported      |
| 186  | torch.Tensor.log2                      | Supported      |
| 187  | torch.Tensor.log2_                     | Supported      |
| 188  | torch.Tensor.log_normal_               | Unsupported      |
| 189  | torch.Tensor.logsumexp                 | Supported      |
| 190  | torch.Tensor.logical_and               | Supported      |
| 191  | torch.Tensor.logical_and_              | Supported      |
| 192  | torch.Tensor.logical_not               | Supported      |
| 193  | torch.Tensor.logical_not_              | Supported      |
| 194  | torch.Tensor.logical_or                | Supported      |
| 195  | torch.Tensor.logical_or_               | Supported      |
| 196  | torch.Tensor.logical_xor               | Unsupported      |
| 197  | torch.Tensor.logical_xor_              | Unsupported      |
| 198  | torch.Tensor.long                      | Supported      |
| 199  | torch.Tensor.lstsq                     | Unsupported      |
| 200  | torch.Tensor.lt                        | Supported      |
| 201  | torch.Tensor.lt_                       | Supported      |
| 202  | torch.Tensor.lu                        | Supported      |
| 203  | torch.Tensor.lu_solve                  | Supported      |
| 204  | torch.Tensor.lu_unpack                 | Supported      |
| 205  | torch.Tensor.map_                      | Unsupported      |
| 206  | torch.Tensor.masked_scatter_           | Supported      |
| 207  | torch.Tensor.masked_scatter            | Supported      |
| 208  | torch.Tensor.masked_fill_              | Supported      |
| 209  | torch.Tensor.masked_fill               | Supported      |
| 210  | torch.Tensor.masked_select             | Supported      |
| 211  | torch.Tensor.matmul                    | Supported      |
| 212  | torch.Tensor.matrix_power              | Supported      |
| 213  | torch.Tensor.matrix_rank               | Supported      |
| 214  | torch.Tensor.matrix_exp                | Supported      |
| 215  | torch.Tensor.max                       | Supported      |
| 216  | torch.Tensor.mean                      | Supported      |
| 217  | torch.Tensor.median                    | Supported      |
| 218  | torch.Tensor.min                       | Supported      |
| 219  | torch.Tensor.mm                        | Supported      |
| 220  | torch.Tensor.mode                      | Unsupported      |
| 221  | torch.Tensor.mul                       | Supported      |
| 222  | torch.Tensor.mul_                      | Supported      |
| 223  | torch.Tensor.multinomial               | Supported      |
| 224  | torch.Tensor.mv                        | Supported      |
| 225  | torch.Tensor.mvlgamma                  | Unsupported      |
| 226  | torch.Tensor.mvlgamma_                 | Unsupported      |
| 227  | torch.Tensor.narrow                    | Supported      |
| 228  | torch.Tensor.narrow_copy               | Supported      |
| 229  | torch.Tensor.ndimension                | Supported      |
| 230  | torch.Tensor.ne                        | Supported      |
| 231  | torch.Tensor.ne_                       | Supported      |
| 232  | torch.Tensor.neg                       | Supported      |
| 233  | torch.Tensor.neg_                      | Supported      |
| 234  | torch.Tensor.nelement                  | Supported      |
| 235  | torch.Tensor.nonzero                   | Supported      |
| 236  | torch.Tensor.norm                      | Supported      |
| 237  | torch.Tensor.normal_                   | Supported      |
| 238  | torch.Tensor.numel                     | Supported      |
| 239  | torch.Tensor.numpy                     | Unsupported      |
| 240  | torch.Tensor.orgqr                     | Unsupported      |
| 241  | torch.Tensor.ormqr                     | Unsupported      |
| 242  | torch.Tensor.outer                     | Supported      |
| 243  | torch.Tensor.permute                   | Supported      |
| 244  | torch.Tensor.pca_lowrank               | Supported      |
| 245  | torch.Tensor.pin_memory                | Unsupported      |
| 246  | torch.Tensor.pinverse                  | Supported      |
| 247  | torch.Tensor.polygamma                 | Unsupported      |
| 248  | torch.Tensor.polygamma_                | Unsupported      |
| 249  | torch.Tensor.pow                       | Supported      |
| 250  | torch.Tensor.pow_                      | Supported      |
| 251  | torch.Tensor.prod                      | Supported      |
| 252  | torch.Tensor.put_                      | Supported      |
| 253  | torch.Tensor.qr                        | Supported      |
| 254  | torch.Tensor.qscheme                   | Unsupported      |
| 255  | torch.Tensor.q_scale                   | Unsupported      |
| 256  | torch.Tensor.q_zero_point              | Unsupported      |
| 257  | torch.Tensor.q_per_channel_scales      | Unsupported      |
| 258  | torch.Tensor.q_per_channel_zero_points | Unsupported      |
| 259  | torch.Tensor.q_per_channel_axis        | Unsupported      |
| 260  | torch.Tensor.random_                   | Supported      |
| 261  | torch.Tensor.reciprocal                | Supported      |
| 262  | torch.Tensor.reciprocal_               | Supported      |
| 263  | torch.Tensor.record_stream             | Supported      |
| 264  | torch.Tensor.remainder                 | Supported      |
| 265  | torch.Tensor.remainder_                | Supported      |
| 266  | torch.Tensor.renorm                    | Supported      |
| 267  | torch.Tensor.renorm_                   | Supported      |
| 268  | torch.Tensor.repeat                    | Supported      |
| 269  | torch.Tensor.repeat_interleave         | Supported      |
| 270  | torch.Tensor.requires_grad_            | Supported      |
| 271  | torch.Tensor.reshape                   | Supported      |
| 272  | torch.Tensor.reshape_as                | Supported      |
| 273  | torch.Tensor.resize_                   | Supported      |
| 274  | torch.Tensor.resize_as_                | Supported      |
| 275  | torch.Tensor.rfft                      | Unsupported      |
| 276  | torch.Tensor.roll                      | Supported      |
| 277  | torch.Tensor.rot90                     | Supported      |
| 278  | torch.Tensor.round                     | Supported      |
| 279  | torch.Tensor.round_                    | Supported      |
| 280  | torch.Tensor.rsqrt                     | Supported      |
| 281  | torch.Tensor.rsqrt_                    | Supported      |
| 282  | torch.Tensor.scatter                   | Supported      |
| 283  | torch.Tensor.scatter_                  | Supported      |
| 284  | torch.Tensor.scatter_add_              | Supported      |
| 285  | torch.Tensor.scatter_add               | Supported      |
| 286  | torch.Tensor.select                    | Supported      |
| 287  | torch.Tensor.set_                      | Supported      |
| 288  | torch.Tensor.share_memory_             | Unsupported      |
| 289  | torch.Tensor.short                     | Supported      |
| 290  | torch.Tensor.sigmoid                   | Supported      |
| 291  | torch.Tensor.sigmoid_                  | Supported      |
| 292  | torch.Tensor.sign                      | Supported      |
| 293  | torch.Tensor.sign_                     | Supported      |
| 294  | torch.Tensor.sin                       | Supported      |
| 295  | torch.Tensor.sin_                      | Supported      |
| 296  | torch.Tensor.sinh                      | Supported      |
| 297  | torch.Tensor.sinh_                     | Supported      |
| 298  | torch.Tensor.size                      | Supported      |
| 299  | torch.Tensor.slogdet                   | Supported      |
| 300  | torch.Tensor.solve                     | Unsupported      |
| 301  | torch.Tensor.sort                      | Supported      |
| 302  | torch.Tensor.split                     | Supported      |
| 303  | torch.Tensor.sparse_mask               | Unsupported      |
| 304  | torch.Tensor.sparse_dim                | Unsupported      |
| 305  | torch.Tensor.sqrt                      | Supported      |
| 306  | torch.Tensor.sqrt_                     | Supported      |
| 307  | torch.Tensor.square                    | Supported      |
| 308  | torch.Tensor.square_                   | Supported      |
| 309  | torch.Tensor.squeeze                   | Supported      |
| 310  | torch.Tensor.squeeze_                  | Supported      |
| 311  | torch.Tensor.std                       | Supported      |
| 312  | torch.Tensor.stft                      | Unsupported      |
| 313  | torch.Tensor.storage                   | Supported      |
| 314  | torch.Tensor.storage_offset            | Supported      |
| 315  | torch.Tensor.storage_type              | Supported      |
| 316  | torch.Tensor.stride                    | Supported      |
| 317  | torch.Tensor.sub                       | Supported      |
| 318  | torch.Tensor.sub_                      | Supported      |
| 319  | torch.Tensor.sum                       | Supported      |
| 320  | torch.Tensor.sum_to_size               | Supported      |
| 321  | torch.Tensor.svd                       | Supported      |
| 322  | torch.Tensor.svd_lowrank               | Supported      |
| 323  | torch.Tensor.symeig                    | Supported      |
| 324  | torch.Tensor.t                         | Supported      |
| 325  | torch.Tensor.t_                        | Supported      |
| 326  | torch.Tensor.to                        | Supported      |
| 327  | torch.Tensor.to_mkldnn                 | Unsupported      |
| 328  | torch.Tensor.take                      | Supported      |
| 329  | torch.Tensor.tan                       | Supported      |
| 330  | torch.Tensor.tan_                      | Supported      |
| 331  | torch.Tensor.tanh                      | Supported      |
| 332  | torch.Tensor.tanh_                     | Supported      |
| 333  | torch.Tensor.tolist                    | Supported      |
| 334  | torch.Tensor.topk                      | Supported      |
| 335  | torch.Tensor.to_sparse                 | Unsupported      |
| 336  | torch.Tensor.trace                     | Unsupported      |
| 337  | torch.Tensor.trapz                     | Supported      |
| 338  | torch.Tensor.transpose                 | Supported      |
| 339  | torch.Tensor.transpose_                | Supported      |
| 340  | torch.Tensor.triangular_solve          | Supported      |
| 341  | torch.Tensor.tril                      | Supported      |
| 342  | torch.Tensor.tril_                     | Supported      |
| 343  | torch.Tensor.triu                      | Supported      |
| 344  | torch.Tensor.triu_                     | Supported      |
| 345  | torch.Tensor.true_divide               | Supported      |
| 346  | torch.Tensor.true_divide_              | Supported      |
| 347  | torch.Tensor.trunc                     | Supported      |
| 348  | torch.Tensor.trunc_                    | Supported      |
| 349  | torch.Tensor.type                      | Supported      |
| 350  | torch.Tensor.type_as                   | Supported      |
| 351  | torch.Tensor.unbind                    | Supported      |
| 352  | torch.Tensor.unfold                    | Supported      |
| 353  | torch.Tensor.uniform_                  | Supported      |
| 354  | torch.Tensor.unique                    | Supported      |
| 355  | torch.Tensor.unique_consecutive        | Unsupported      |
| 356  | torch.Tensor.unsqueeze                 | Supported      |
| 357  | torch.Tensor.unsqueeze_                | Supported      |
| 358  | torch.Tensor.values                    | Unsupported      |
| 359  | torch.Tensor.var                       | Unsupported      |
| 360  | torch.Tensor.vdot                      | Supported      |
| 361  | torch.Tensor.view                      | Supported      |
| 362  | torch.Tensor.view_as                   | Supported      |
| 363  | torch.Tensor.where                     | Supported      |
| 364  | torch.Tensor.zero_                     | Supported      |
| 365  | torch.BoolTensor                       | Supported      |
| 366  | torch.BoolTensor.all                   | Supported      |
| 367  | torch.BoolTensor.any                   | Supported      |

## Layers (torch.nn)

| No.| API                                                 | Supported/Unsupported                    |
| ---- | -------------------------------------------------------- | ---------------------------- |
| 1    | torch.nn.Parameter                                       | Supported                          |
| 2    | torch.nn.UninitializedParameter                          | Supported                          |
| 3    | torch.nn.Module                                          | Supported                          |
| 4    | torch.nn.Module.add_module                               | Supported                          |
| 5    | torch.nn.Module.apply                                    | Supported                          |
| 6    | torch.nn.Module.bfloat16                                 | Unsupported                          |
| 7    | torch.nn.Module.buffers                                  | Supported                          |
| 8    | torch.nn.Module.children                                 | Supported                          |
| 9    | torch.nn.Module.cpu                                      | Supported                          |
| 10   | torch.nn.Module.cuda                                     | Unsupported                          |
| 11   | torch.nn.Module.double                                   | Unsupported                          |
| 12   | torch.nn.Module.dump_patches                             | Supported                          |
| 13   | torch.nn.Module.eval                                     | Supported                          |
| 14   | torch.nn.Module.extra_repr                               | Supported                          |
| 15   | torch.nn.Module.float                                    | Supported                          |
| 16   | torch.nn.Module.forward                                  | Supported                          |
| 17   | torch.nn.Module.half                                     | Supported                          |
| 18   | torch.nn.Module.load_state_dict                          | Supported                          |
| 19   | torch.nn.Module.modules                                  | Supported                          |
| 20   | torch.nn.Module.named_buffers                            | Supported                          |
| 21   | torch.nn.Module.named_children                           | Supported                          |
| 22   | torch.nn.Module.named_modules                            | Supported                          |
| 23   | torch.nn.Module.named_parameters                         | Supported                          |
| 24   | torch.nn.Module.parameters                               | Supported                          |
| 25   | torch.nn.Module.register_backward_hook                   | Supported                          |
| 26   | torch.nn.Module.register_buffer                          | Supported                          |
| 27   | torch.nn.Module.register_forward_hook                    | Supported                          |
| 28   | torch.nn.Module.register_forward_pre_hook                | Supported                          |
| 29   | torch.nn.Module.register_parameter                       | Supported                          |
| 30   | torch.nn.register_module_forward_pre_hook                | Unsupported                          |
| 31   | torch.nn.register_module_forward_hook                    | Unsupported                          |
| 32   | torch.nn.register_module_backward_hook                   | Unsupported                          |
| 33   | torch.nn.Module.requires_grad_                           | Supported                          |
| 34   | torch.nn.Module.state_dict                               | Supported                          |
| 35   | torch.nn.Module.to                                       | Supported                          |
| 36   | torch.nn.Module.train                                    | Supported                          |
| 37   | torch.nn.Module.type                                     | Supported                          |
| 38   | torch.nn.Module.zero_grad                                | Supported                          |
| 39   | torch.nn.Sequential                                      | Supported                          |
| 40   | torch.nn.ModuleList                                      | Supported                          |
| 41   | torch.nn.ModuleList.append                               | Supported                          |
| 42   | torch.nn.ModuleList.extend                               | Supported                          |
| 43   | torch.nn.ModuleList.insert                               | Supported                          |
| 44   | torch.nn.ModuleDict                                      | Supported                          |
| 45   | torch.nn.ModuleDict.clear                                | Supported                          |
| 46   | torch.nn.ModuleDict.items                                | Supported                          |
| 47   | torch.nn.ModuleDict.keys                                 | Supported                          |
| 48   | torch.nn.ModuleDict.pop                                  | Supported                          |
| 49   | torch.nn.ModuleDict.update                               | Supported                          |
| 50   | torch.nn.ModuleDict.values                               | Supported                          |
| 51   | torch.nn.ParameterList                                   | Supported                          |
| 52   | torch.nn.ParameterList.append                            | Supported                          |
| 53   | torch.nn.ParameterList.extend                            | Supported                          |
| 54   | torch.nn.ParameterDict                                   | Supported                          |
| 55   | torch.nn.ParameterDict.clear                             | Supported                          |
| 56   | torch.nn.ParameterDict.items                             | Supported                          |
| 57   | torch.nn.ParameterDict.keys                              | Supported                          |
| 58   | torch.nn.ParameterDict.pop                               | Supported                          |
| 59   | torch.nn.ParameterDict.update                            | Supported                          |
| 60   | torch.nn.ParameterDict.values                            | Supported                          |
| 61   | torch.nn.Conv1d                                          | Supported                          |
| 62   | torch.nn.Conv2d                                          | Supported                          |
| 63   | torch.nn.Conv3d                                          | Supported                          |
| 64   | torch.nn.ConvTranspose1d                                 | Supported                          |
| 65   | torch.nn.ConvTranspose2d                                 | Supported                          |
| 66   | torch.nn.ConvTranspose3d                                 | Supported                          |
| 67   | torch.nn.LazyConv1d                                      | Supported                          |
| 68   | torch.nn.LazyConv2d                                      | Supported                          |
| 69   | torch.nn.LazyConv3d                                      | Supported                          |
| 70   | torch.nn.LazyConvTranspose1d                             | Supported                          |
| 71   | torch.nn.LazyConvTranspose2d                             | Supported                          |
| 72   | torch.nn.LazyConvTranspose3d                             | Supported                          |
| 73   | torch.nn.Unfold                                          | Supported                          |
| 74   | torch.nn.Fold                                            | Supported                          |
| 75   | torch.nn.MaxPool1d                                       | Supported                          |
| 76   | torch.nn.MaxPool2d                                       | Supported                          |
| 77   | torch.nn.MaxPool3d                                       | Supported                          |
| 78   | torch.nn.MaxUnpool1d                                     | Supported                          |
| 79   | torch.nn.MaxUnpool2d                                     | Supported                          |
| 80   | torch.nn.MaxUnpool3d                                     | Supported                          |
| 81   | torch.nn.AvgPool1d                                       | Supported                          |
| 82   | torch.nn.AvgPool2d                                       | Supported                          |
| 83   | torch.nn.AvgPool3d                                       | Supported                          |
| 84   | torch.nn.FractionalMaxPool2d                             | Unsupported                          |
| 85   | torch.nn.LPPool1d                                        | Supported                          |
| 86   | torch.nn.LPPool2d                                        | Supported                          |
| 87   | torch.nn.AdaptiveMaxPool1d                               | Supported                          |
| 88   | torch.nn.AdaptiveMaxPool2d                               | Supported                          |
| 89   | torch.nn.AdaptiveMaxPool3d                               | Unsupported                          |
| 90   | torch.nn.AdaptiveAvgPool1d                               | Supported                          |
| 91   | torch.nn.AdaptiveAvgPool2d                               | Supported                          |
| 92   | torch.nn.AdaptiveAvgPool3d                               | Supported only in the D=1, H=1, and W=1 scenario. |
| 93   | torch.nn.ReflectionPad1d                                 | Supported                          |
| 94   | torch.nn.ReflectionPad2d                                 | Supported                          |
| 95   | torch.nn.ReplicationPad1d                                | Supported                          |
| 96   | torch.nn.ReplicationPad2d                                | Supported                          |
| 97   | torch.nn.ReplicationPad3d                                | Unsupported                          |
| 98   | torch.nn.ZeroPad2d                                       | Supported                          |
| 99   | torch.nn.ConstantPad1d                                   | Supported                          |
| 100  | torch.nn.ConstantPad2d                                   | Supported                          |
| 101  | torch.nn.ConstantPad3d                                   | Supported                          |
| 102  | torch.nn.ELU                                             | Supported                          |
| 103  | torch.nn.Hardshrink                                      | Supported                          |
| 104  | torch.nn.Hardsigmoid                                     | Supported                          |
| 105  | torch.nn.Hardtanh                                        | Supported                          |
| 106  | torch.nn.Hardswish                                       | Supported                          |
| 107  | torch.nn.LeakyReLU                                       | Supported                          |
| 108  | torch.nn.LogSigmoid                                      | Supported                          |
| 109  | torch.nn.MultiheadAttention                              | Supported                          |
| 110  | torch.nn.PReLU                                           | Supported                          |
| 111  | torch.nn.ReLU                                            | Supported                          |
| 112  | torch.nn.ReLU6                                           | Supported                          |
| 113  | torch.nn.RReLU                                           | Supported                          |
| 114  | torch.nn.SELU                                            | Supported                          |
| 115  | torch.nn.CELU                                            | Supported                          |
| 116  | torch.nn.GELU                                            | Supported                          |
| 117  | torch.nn.Sigmoid                                         | Supported                          |
| 118  | torch.nn.SiLU                                            | Supported                          |
| 119  | torch.nn.Softplus                                        | Supported                          |
| 120  | torch.nn.Softshrink                                      | Supported. However, it is not supported in SoftShrink scenarios currently.  |
| 121  | torch.nn.Softsign                                        | Supported                          |
| 122  | torch.nn.Tanh                                            | Supported                          |
| 123  | torch.nn.Tanhshrink                                      | Supported                          |
| 124  | torch.nn.Threshold                                       | Supported                          |
| 125  | torch.nn.Softmin                                         | Supported                          |
| 126  | torch.nn.Softmax                                         | Supported                          |
| 127  | torch.nn.Softmax2d                                       | Supported                          |
| 128  | torch.nn.LogSoftmax                                      | Supported                          |
| 129  | torch.nn.AdaptiveLogSoftmaxWithLoss                      | Unsupported                          |
| 130  | torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob             | Unsupported                          |
| 131  | torch.nn.AdaptiveLogSoftmaxWithLoss.predict              | Unsupported                          |
| 132  | torch.nn.BatchNorm1d                                     | Supported                          |
| 133  | torch.nn.BatchNorm2d                                     | Supported                          |
| 134  | torch.nn.BatchNorm3d                                     | Supported                          |
| 135  | torch.nn.GroupNorm                                       | Supported                          |
| 136  | torch.nn.SyncBatchNorm                                   | Supported                          |
| 137  | torch.nn.SyncBatchNorm.convert_sync_batchnorm            | Supported                          |
| 138  | torch.nn.InstanceNorm1d                                  | Supported                          |
| 139  | torch.nn.InstanceNorm2d                                  | Supported                          |
| 140  | torch.nn.InstanceNorm3d                                  | Supported                          |
| 141  | torch.nn.LayerNorm                                       | Supported                          |
| 142  | torch.nn.LocalResponseNorm                               | Supported                          |
| 143  | torch.nn.RNNBase                                         | Supported                          |
| 144  | torch.nn.RNNBase.flatten_parameters                      | Supported                          |
| 145  | torch.nn.RNN                                             | Supported                          |
| 146  | torch.nn.LSTM                                            | Supported                          |
| 147  | torch.nn.GRU                                             | Supported. However, it is not supported in the DynamicGRUV2 scenario currently.|
| 148  | torch.nn.RNNCell                                         | Supported                          |
| 149  | torch.nn.LSTMCell                                        | Supported. However, it is not supported in the non-16-pixel alignment scenario currently.    |
| 150  | torch.nn.GRUCell                                         | Supported                          |
| 151  | torch.nn.Transformer                                     | Supported                          |
| 152  | torch.nn.Transformer.forward                             | Supported                          |
| 153  | torch.nn.Transformer.generate_square_subsequent_mask     | Supported                          |
| 154  | torch.nn.TransformerEncoder                              | Supported                          |
| 155  | torch.nn.TransformerEncoder.forward                      | Supported                          |
| 156  | torch.nn.TransformerDecoder                              | Supported                          |
| 157  | torch.nn.TransformerDecoder.forward                      | Supported                          |
| 158  | torch.nn.TransformerEncoderLayer                         | Supported                          |
| 159  | torch.nn.TransformerEncoderLayer.forward                 | Supported                          |
| 160  | torch.nn.TransformerDecoderLayer                         | Supported                          |
| 161  | torch.nn.TransformerDecoderLayer.forward                 | Supported                          |
| 162  | torch.nn.Identity                                        | Supported                          |
| 163  | torch.nn.Linear                                          | Supported                          |
| 164  | torch.nn.Bilinear                                        | Supported                          |
| 165  | torch.nn.LazyLinear                                      | Supported                          |
| 166  | torch.nn.Dropout                                         | Supported                          |
| 167  | torch.nn.Dropout2d                                       | Supported                          |
| 168  | torch.nn.Dropout3d                                       | Supported                          |
| 169  | torch.nn.AlphaDropout                                    | Supported                          |
| 170  | torch.nn.Embedding                                       | Supported                          |
| 171  | torch.nn.Embedding.from_pretrained                       | Supported                          |
| 172  | torch.nn.EmbeddingBag                                    | Supported                          |
| 173  | torch.nn.EmbeddingBag.from_pretrained                    | Supported                          |
| 174  | torch.nn.CosineSimilarity                                | Supported                          |
| 175  | torch.nn.PairwiseDistance                                | Supported                          |
| 176  | torch.nn.L1Loss                                          | Supported                          |
| 177  | torch.nn.MSELoss                                         | Supported                          |
| 178  | torch.nn.CrossEntropyLoss                                | Supported                          |
| 179  | torch.nn.CTCLoss                                         | Supported                          |
| 180  | torch.nn.NLLLoss                                         | Supported                          |
| 181  | torch.nn.PoissonNLLLoss                                  | Supported                          |
| 182  | torch.nn.GaussianNLLLoss                                 | Supported                          |
| 183  | torch.nn.KLDivLoss                                       | Supported                          |
| 184  | torch.nn.BCELoss                                         | Supported                          |
| 185  | torch.nn.BCEWithLogitsLoss                               | Supported                          |
| 186  | torch.nn.MarginRankingLoss                               | Supported                          |
| 187  | torch.nn.HingeEmbeddingLoss                              | Supported                          |
| 188  | torch.nn.MultiLabelMarginLoss                            | Supported                          |
| 189  | torch.nn.SmoothL1Loss                                    | Supported                          |
| 190  | torch.nn.SoftMarginLoss                                  | Supported                          |
| 191  | torch.nn.MultiLabelSoftMarginLoss                        | Supported                          |
| 192  | torch.nn.CosineEmbeddingLoss                             | Supported                          |
| 193  | torch.nn.MultiMarginLoss                                 | Unsupported                          |
| 194  | torch.nn.TripletMarginLoss                               | Supported                          |
| 195  | torch.nn.TripletMarginLossWithDistanceLoss               | Supported                          |
| 196  | torch.nn.PixelShuffle                                    | Supported                          |
| 197  | torch.nn.PixelUnshuffle                                  | Supported                          |
| 198  | torch.nn.Upsample                                        | Supported                          |
| 199  | torch.nn.UpsamplingNearest2d                             | Supported                          |
| 200  | torch.nn.UpsamplingBilinear2d                            | Supported                          |
| 201  | torch.nn.ChannelShuffle                                  | Supported                          |
| 202  | torch.nn.DataParallel                                    | Supported                          |
| 203  | torch.nn.parallel.DistributedDataParallel                | Supported                          |
| 204  | torch.nn.parallel.DistributedDataParallel.no_sync        | Supported                          |
| 205  | torch.nn.utils.clip_grad_norm_                           | Supported                          |
| 206  | torch.nn.utils.clip_grad_value_                          | Supported                          |
| 207  | torch.nn.utils.parameters_to_vector                      | Supported                          |
| 208  | torch.nn.utils.vector_to_parameters                      | Supported                          |
| 209  | torch.nn.utils.Prune.BasePruningMethod                   | Supported                          |
| 210  | torch.nn.utils.prune.PruningContainer                    | Supported                          |
| 211  | torch.nn.utils.prune.PruningContainer.add_pruning_method | Supported                          |
| 212  | torch.nn.utils.prune.PruningContainer.apply              | Supported                          |
| 213  | torch.nn.utils.prune.PruningContainer.apply_mask         | Supported                          |
| 214  | torch.nn.utils.prune.PruningContainer.compute_mask       | Supported                          |
| 215  | torch.nn.utils.prune.PruningContainer.prune              | Supported                          |
| 216  | torch.nn.utils.prune.PruningContainer.remove             | Supported                          |
| 217  | torch.nn.utils.prune.Identity                            | Supported                          |
| 218  | torch.nn.utils.prune.Identity.apply                      | Supported                          |
| 219  | torch.nn.utils.prune.Identity.apply_mask                 | Supported                          |
| 220  | torch.nn.utils.prune.Identity.prune                      | Supported                          |
| 221  | torch.nn.utils.prune.Identity.remove                     | Supported                          |
| 222  | torch.nn.utils.prune.RandomUnstructured                  | Supported                          |
| 223  | torch.nn.utils.prune.RandomUnstructured.apply            | Supported                          |
| 224  | torch.nn.utils.prune.RandomUnstructured.apply_mask       | Supported                          |
| 225  | torch.nn.utils.prune.RandomUnstructured.prune            | Supported                          |
| 226  | torch.nn.utils.prune.RandomUnstructured.remove           | Supported                          |
| 227  | torch.nn.utils.prune.L1Unstructured                      | Supported                          |
| 228  | torch.nn.utils.prune.L1Unstructured.apply                | Supported                          |
| 229  | torch.nn.utils.prune.L1Unstructured.apply_mask           | Supported                          |
| 230  | torch.nn.utils.prune.L1Unstructured.prune                | Supported                          |
| 231  | torch.nn.utils.prune.L1Unstructured.remove               | Supported                          |
| 232  | torch.nn.utils.prune.RandomStructured                    | Supported                          |
| 233  | torch.nn.utils.prune.RandomStructured.apply              | Supported                          |
| 234  | torch.nn.utils.prune.RandomStructured.apply_mask         | Supported                          |
| 235  | torch.nn.utils.prune.RandomStructured.compute_mask       | Supported                          |
| 236  | torch.nn.utils.prune.RandomStructured.prune              | Supported                          |
| 237  | torch.nn.utils.prune.RandomStructured.remove             | Supported                          |
| 238  | torch.nn.utils.prune.LnStructured                        | Supported                          |
| 239  | torch.nn.utils.prune.LnStructured.apply                  | Supported                          |
| 240  | torch.nn.utils.prune.LnStructured.apply_mask             | Supported                          |
| 241  | torch.nn.utils.prune.LnStructured.compute_mask           | Supported                          |
| 242  | torch.nn.utils.prune.LnStructured.prune                  | Supported                          |
| 243  | torch.nn.utils.prune.LnStructured.remove                 | Supported                          |
| 244  | torch.nn.utils.prune.CustomFromMask                      | Supported                          |
| 245  | torch.nn.utils.prune.CustomFromMask.apply                | Supported                          |
| 246  | torch.nn.utils.prune.CustomFromMask.apply_mask           | Supported                          |
| 247  | torch.nn.utils.prune.CustomFromMask.prune                | Supported                          |
| 248  | torch.nn.utils.prune.CustomFromMask.remove               | Supported                          |
| 249  | torch.nn.utils.prune.identity                            | Supported                          |
| 250  | torch.nn.utils.prune.random_unstructured                 | Supported                          |
| 251  | torch.nn.utils.prune.l1_unstructured                     | Supported                          |
| 252  | torch.nn.utils.prune.random_structured                   | Supported                          |
| 253  | torch.nn.utils.prune.ln_structured                       | Supported                          |
| 254  | torch.nn.utils.prune.global_unstructured                 | Supported                          |
| 255  | torch.nn.utils.prune.custom_from_mask                    | Supported                          |
| 256  | torch.nn.utils.prune.remove                              | Supported                          |
| 257  | torch.nn.utils.prune.is_pruned                           | Supported                          |
| 258  | torch.nn.utils.weight_norm                               | Supported                          |
| 259  | torch.nn.utils.remove_weight_norm                        | Supported                          |
| 260  | torch.nn.utils.spectral_norm                             | Supported                          |
| 261  | torch.nn.utils.remove_spectral_norm                      | Supported                          |
| 262  | torch.nn.utils.rnn.PackedSequence                        | Supported                          |
| 263  | torch.nn.utils.rnn.pack_padded_sequence                  | Supported                          |
| 264  | torch.nn.utils.rnn.pad_packed_sequence                   | Supported                          |
| 265  | torch.nn.utils.rnn.pad_sequence                          | Supported                          |
| 266  | torch.nn.utils.rnn.pack_sequence                         | Supported                          |
| 267  | torch.nn.Flatten                                         | Supported                          |
| 268  | torch.nn.Unflatten                                       | Supported                          |
| 269  | torch.nn.modules.lazy.LazyModuleMixin                    | Supported                          |
| 270  | torch.quantization.quantize                              | Unsupported                          |
| 271  | torch.quantization.quantize_dynamic                      | Unsupported                          |
| 272  | torch.quantization.quantize_qat                          | Unsupported                          |
| 273  | torch.quantization.prepare                               | Supported                          |
| 274  | torch.quantization.prepare_qat                           | Unsupported                          |
| 275  | torch.quantization.convert                               | Unsupported                          |
| 276  | torch.quantization.QConfig                               | Supported                          |
| 277  | torch.quantization.QConfigDynamic                        | Supported                          |
| 278  | torch.quantization.fuse_modules                          | Supported                          |
| 279  | torch.quantization.QuantStub                             | Supported                          |
| 280  | torch.quantization.DeQuantStub                           | Supported                          |
| 281  | torch.quantization.QuantWrapper                          | Supported                          |
| 282  | torch.quantization.add_quant_dequant                     | Supported                          |
| 283  | torch.quantization.add_observer_                         | Supported                          |
| 284  | torch.quantization.swap_module                           | Supported                          |
| 285  | torch.quantization.propagate_qconfig_                    | Supported                          |
| 286  | torch.quantization.default_eval_fn                       | Supported                          |
| 287  | torch.quantization.MinMaxObserver                        | Supported                          |
| 288  | torch.quantization.MovingAverageMinMaxObserver           | Supported                          |
| 289  | torch.quantization.PerChannelMinMaxObserver              | Supported                          |
| 290  | torch.quantization.MovingAveragePerChannelMinMaxObserver | Supported                          |
| 291  | torch.quantization.HistogramObserver                     | Unsupported                          |
| 292  | torch.quantization.FakeQuantize                          | Unsupported                          |
| 293  | torch.quantization.NoopObserver                          | Supported                          |
| 294  | torch.quantization.get_observer_dict                     | Supported                          |
| 295  | torch.quantization.RecordingObserver                     | Supported                          |
| 296  | torch.nn.intrinsic.ConvBn2d                              | Supported                          |
| 297  | torch.nn.intrinsic.ConvBnReLU2d                          | Supported                          |
| 298  | torch.nn.intrinsic.ConvReLU2d                            | Supported                          |
| 299  | torch.nn.intrinsic.ConvReLU3d                            | Supported                          |
| 300  | torch.nn.intrinsic.LinearReLU                            | Supported                          |
| 301  | torch.nn.intrinsic.qat.ConvBn2d                          | Unsupported                          |
| 302  | torch.nn.intrinsic.qat.ConvBnReLU2d                      | Unsupported                          |
| 303  | torch.nn.intrinsic.qat.ConvReLU2d                        | Unsupported                          |
| 304  | torch.nn.intrinsic.qat.LinearReLU                        | Unsupported                          |
| 305  | torch.nn.intrinsic.quantized.ConvReLU2d                  | Unsupported                          |
| 306  | torch.nn.intrinsic.quantized.ConvReLU3d                  | Unsupported                          |
| 307  | torch.nn.intrinsic.quantized.LinearReLU                  | Unsupported                          |
| 308  | torch.nn.qat.Conv2d                                      | Unsupported                          |
| 309  | torch.nn.qat.Conv2d.from_float                           | Unsupported                          |
| 310  | torch.nn.qat.Linear                                      | Unsupported                          |
| 311  | torch.nn.qat.Linear.from_float                           | Unsupported                          |
| 312  | torch.nn.quantized.functional.relu                       | Unsupported                          |
| 313  | torch.nn.quantized.functional.linear                     | Unsupported                          |
| 314  | torch.nn.quantized.functional.conv2d                     | Unsupported                          |
| 315  | torch.nn.quantized.functional.conv3d                     | Unsupported                          |
| 316  | torch.nn.quantized.functional.max_pool2d                 | Unsupported                          |
| 317  | torch.nn.quantized.functional.adaptive_avg_pool2d        | Unsupported                          |
| 318  | torch.nn.quantized.functional.avg_pool2d                 | Unsupported                          |
| 319  | torch.nn.quantized.functional.interpolate                | Unsupported                          |
| 320  | torch.nn.quantized.functional.upsample                   | Unsupported                          |
| 321  | torch.nn.quantized.functional.upsample_bilinear          | Unsupported                          |
| 322  | torch.nn.quantized.functional.upsample_nearest           | Unsupported                          |
| 323  | torch.nn.quantized.ReLU                                  | Unsupported                          |
| 324  | torch.nn.quantized.ReLU6                                 | Unsupported                          |
| 325  | torch.nn.quantized.Conv2d                                | Unsupported                          |
| 326  | torch.nn.quantized.Conv2d.from_float                     | Unsupported                          |
| 327  | torch.nn.quantized.Conv3d                                | Unsupported                          |
| 328  | torch.nn.quantized.Conv3d.from_float                     | Unsupported                          |
| 329  | torch.nn.quantized.FloatFunctional                       | Supported                          |
| 330  | torch.nn.quantized.QFunctional                           | Unsupported                          |
| 331  | torch.nn.quantized.Quantize                              | Supported                          |
| 332  | torch.nn.quantized.DeQuantize                            | Unsupported                          |
| 333  | torch.nn.quantized.Linear                                | Unsupported                          |
| 334  | torch.nn.quantized.Linear.from_float                     | Unsupported                          |
| 335  | torch.nn.quantized.dynamic.Linear                        | Unsupported                          |
| 336  | torch.nn.quantized.dynamic.Linear.from_float             | Unsupported                          |
| 337  | torch.nn.quantized.dynamic.LSTM                          | Unsupported                          |

## Functions (torch.nn.functional)

| No.| API                                              | Supported/Unsupported                   |
| ---- | ----------------------------------------------------- | --------------------------- |
| 1    | torch.nn.functional.conv1d                            | Supported                         |
| 2    | torch.nn.functional.conv2d                            | Supported                         |
| 3    | torch.nn.functional.conv3d                            | Supported                         |
| 4    | torch.nn.functional.conv_transpose1d                  | Supported                         |
| 5    | torch.nn.functional.conv_transpose2d                  | Supported                         |
| 6    | torch.nn.functional.conv_transpose3d                  | Supported                         |
| 7    | torch.nn.functional.unfold                            | Supported                         |
| 8    | torch.nn.functional.fold                              | Supported                         |
| 9    | torch.nn.functional.avg_pool1d                        | Supported                         |
| 10   | torch.nn.functional.avg_pool2d                        | Supported                         |
| 11   | torch.nn.functional.avg_pool3d                        | Supported                         |
| 12   | torch.nn.functional.max_pool1d                        | Supported                         |
| 13   | torch.nn.functional.max_pool2d                        | Supported                         |
| 14   | torch.nn.functional.max_pool3d                        | Supported                         |
| 15   | torch.nn.functional.max_unpool1d                      | Supported                         |
| 16   | torch.nn.functional.max_unpool2d                      | Supported                         |
| 17   | torch.nn.functional.max_unpool3d                      | Supported                         |
| 18   | torch.nn.functional.lp_pool1d                         | Supported                         |
| 19   | torch.nn.functional.lp_pool2d                         | Supported                         |
| 20   | torch.nn.functional.adaptive_max_pool1d               | Supported                         |
| 21   | torch.nn.functional.adaptive_max_pool2d               | Supported                         |
| 22   | torch.nn.functional.adaptive_max_pool3d               | Unsupported                         |
| 23   | torch.nn.functional.adaptive_avg_pool1d               | Supported                         |
| 24   | torch.nn.functional.adaptive_avg_pool2d               | Supported                         |
| 25   | torch.nn.functional.adaptive_avg_pool3d               | Supported only in the D=1, H=1, and W=1 scenario.|
| 26   | torch.nn.functional.threshold                         | Supported                         |
| 27   | torch.nn.functional.threshold_                        | Supported                         |
| 28   | torch.nn.functional.relu                              | Supported                         |
| 29   | torch.nn.functional.relu_                             | Supported                         |
| 30   | torch.nn.functional.hardtanh                          | Supported                         |
| 31   | torch.nn.functional.hardtanh_                         | Supported                         |
| 32   | torch.nn.functional.swish                             | Supported                         |
| 33   | torch.nn.functional.relu6                             | Supported                         |
| 34   | torch.nn.functional.elu                               | Supported                         |
| 35   | torch.nn.functional.elu_                              | Supported                         |
| 36   | torch.nn.functional.selu                              | Supported                         |
| 37   | torch.nn.functional.celu                              | Supported                         |
| 38   | torch.nn.functional.leaky_relu                        | Supported                         |
| 39   | torch.nn.functional.leaky_relu_                       | Supported                         |
| 40   | torch.nn.functional.prelu                             | Supported                         |
| 41   | torch.nn.functional.rrelu                             | Supported                         |
| 42   | torch.nn.functional.rrelu_                            | Supported                         |
| 43   | torch.nn.functional.glu                               | Supported                         |
| 44   | torch.nn.functional.gelu                              | Supported                         |
| 45   | torch.nn.functional.logsigmoid                        | Supported                         |
| 46   | torch.nn.functional.hardshrink                        | Supported                         |
| 47   | torch.nn.functional.tanhshrink                        | Supported                         |
| 48   | torch.nn.functional.softsign                          | Supported                         |
| 49   | torch.nn.functional.softplus                          | Supported                         |
| 50   | torch.nn.functional.softmin                           | Supported                         |
| 51   | torch.nn.functional.softmax                           | Supported                         |
| 52   | torch.nn.functional.softshrink                        | Supported                         |
| 53   | torch.nn.functional.gumbel_softmax                    | Unsupported                         |
| 54   | torch.nn.functional.log_softmax                       | Supported                         |
| 55   | torch.nn.functional.tanh                              | Supported                         |
| 56   | torch.nn.functional.sigmoid                           | Supported                         |
| 57   | torch.nn.functional.hardsigmoid                       | Supported                         |
| 58   | torch.nn.functional.hardswish                         | Supported                         |
| 59   | torch.nn.functional.feature_alpha_dropout             | Supported                         |
| 60   | torch.nn.functional.silu                              | Supported                         |
| 61   | torch.nn.functional.batch_norm                        | Supported                         |
| 62   | torch.nn.functional.instance_norm                     | Supported                         |
| 63   | torch.nn.functional.layer_norm                        | Supported                         |
| 64   | torch.nn.functional.local_response_norm               | Supported                         |
| 65   | torch.nn.functional.normalize                         | Supported                         |
| 66   | torch.nn.functional.linear                            | Supported                         |
| 67   | torch.nn.functional.bilinear                          | Supported                         |
| 68   | torch.nn.functional.dropout                           | Supported                         |
| 69   | torch.nn.functional.alpha_dropout                     | Supported                         |
| 70   | torch.nn.functional.dropout2d                         | Supported                         |
| 71   | torch.nn.functional.dropout3d                         | Supported                         |
| 72   | torch.nn.functional.embedding                         | Supported                         |
| 73   | torch.nn.functional.embedding_bag                     | Supported                         |
| 74   | torch.nn.functional.one_hot                           | Supported                         |
| 75   | torch.nn.functional.pairwise_distance                 | Supported                         |
| 76   | torch.nn.functional.cosine_similarity                 | Supported                         |
| 77   | torch.nn.functional.pdist                             | Supported                         |
| 78   | torch.nn.functional.binary_cross_entropy              | Supported. The value of **y** can only be **1** or **0**.   |
| 79   | torch.nn.functional.binary_cross_entropy_with_logits  | Supported                         |
| 80   | torch.nn.functional.poisson_nll_loss                  | Supported                         |
| 81   | torch.nn.functional.cosine_embedding_loss             | Supported                         |
| 82   | torch.nn.functional.cross_entropy                     | Supported                         |
| 83   | torch.nn.functional.ctc_loss                          | Supported. Only 2-dimensional inputs are supported.        |
| 84   | torch.nn.functional.hinge_embedding_loss              | Supported                         |
| 85   | torch.nn.functional.kl_div                            | Supported                         |
| 86   | torch.nn.functional.l1_loss                           | Supported                         |
| 87   | torch.nn.functional.mse_loss                          | Supported                         |
| 88   | torch.nn.functional.margin_ranking_loss               | Supported                         |
| 89   | torch.nn.functional.multilabel_margin_loss            | Supported                         |
| 90   | torch.nn.functional.multilabel_soft_margin_loss       | Supported                         |
| 91   | torch.nn.functional.multi_margin_loss                 | Supported                         |
| 92   | torch.nn.functional.nll_loss                          | Supported                         |
| 93   | torch.nn.functional.smooth_l1_loss                    | Supported                         |
| 94   | torch.nn.functional.soft_margin_loss                  | Supported                         |
| 95   | torch.nn.functional.triplet_margin_loss               | Supported                         |
| 96   | torch.nn.functional.triplet_margin_with_distance_loss | Supported                         |
| 97   | torch.nn.functional.pixel_shuffle                     | Unsupported                         |
| 98   | torch.nn.functional.pixel_unshuffle                   | Unsupported                         |
| 99   | torch.nn.functional.pad                               | Supported                         |
| 100  | torch.nn.functional.interpolate                       | Supported                         |
| 101  | torch.nn.functional.upsample                          | Supported                         |
| 102  | torch.nn.functional.upsample_nearest                  | Supported                         |
| 103  | torch.nn.functional.upsample_bilinear                 | Supported                         |
| 104  | torch.nn.functional.grid_sample                       | Supported                         |
| 105  | torch.nn.functional.affine_grid                       | Supported                         |
| 106  | torch.nn.parallel.data_parallel                       | Unsupported                         |

## torch.distributed

| No.| API                                  | Supported/Unsupported|
| ---- | ----------------------------------------- | -------- |
| 1    | torch.distributed.is_available            | Supported      |
| 2    | torch.distributed.init_process_group      | Supported      |
| 3    | torch.distributed.Backend                 | Supported      |
| 4    | torch.distributed.get_backend             | Supported      |
| 5    | torch.distributed.get_rank                | Supported      |
| 6    | torch.distributed.get_world_size          | Supported      |
| 7    | torch.distributed.is_initialized          | Supported      |
| 8    | torch.distributed.is_mpi_available        | Supported      |
| 9    | torch.distributed.is_nccl_available       | Supported      |
| 10   | torch.distributed.new_group               | Supported      |
| 11   | torch.distributed.Store                   | Supported      |
| 12   | torch.distributed.TCPStore                | Supported      |
| 13   | torch.distributed.HashStore               | Supported      |
| 14   | torch.distributed.FileStore               | Supported      |
| 15   | torch.distributed.PrefixStore             | Supported      |
| 16   | torch.distributed.Store.set               | Supported      |
| 17   | torch.distributed.Store.get               | Supported      |
| 18   | torch.distributed.Store.add               | Supported      |
| 19   | torch.distributed.Store.wait              | Supported      |
| 20   | torch.distributed.Store.num_keys          | Supported      |
| 21   | torch.distributed.Store.delete_keys       | Supported      |
| 22   | torch.distributed.Store.set_timeout       | Supported      |
| 23   | torch.distributed.send                    | Unsupported      |
| 24   | torch.distributed.recv                    | Unsupported      |
| 25   | torch.distributed.isend                   | Unsupported      |
| 26   | torch.distributed.irecv                   | Unsupported      |
| 27   | is_completed                              | Supported      |
| 28   | wait                                      | Supported      |
| 29   | torch.distributed.broadcast               | Supported      |
| 30   | torch.distributed.broadcast_object_list   | Supported      |
| 31   | torch.distributed.all_reduce              | Supported      |
| 32   | torch.distributed.reduce                  | Unsupported      |
| 33   | torch.distributed.all_gather              | Supported      |
| 34   | torch.distributed.all_gather_object       | Supported      |
| 35   | torch.distributed.gather_object           | Supported      |
| 36   | torch.distributed.gather                  | Unsupported      |
| 37   | torch.distributed.scatter                 | Unsupported      |
| 38   | torch.distributed.scatter_object_list     | Supported      |
| 39   | torch.distributed.reduce_scatter          | Supported      |
| 40   | torch.distributed.reduce_scatter_multigpu | Supported      |
| 41   | torch.distributed.all_to_all              | Supported      |
| 42   | torch.distributed.barrier                 | Supported      |
| 43   | torch.distributed.ReduceOp                | Supported      |
| 44   | torch.distributed.reduce_op               | Supported      |
| 45   | torch.distributed.broadcast_multigpu      | Unsupported      |
| 46   | torch.distributed.all_reduce_multigpu     | Unsupported      |
| 47   | torch.distributed.reduce_multigpu         | Unsupported      |
| 48   | torch.distributed.all_gather_multigpu     | Unsupported      |
| 49   | torch.distributed.launch                  | Supported      |
| 50   | torch.multiprocessing.spawn               | Supported      |

## torch_npu.npu

| No.| API                                      | NPU API Name                                  | Supported/Unsupported      |
| ---- | --------------------------------------------- | ------------------------------------------------ | -------------- |
| 1    | torch.cuda.can_device_access_peer             | torch_npu.npu.can_device_access_peer             | Supported            |
| 2    | torch.cuda.current_blas_handle                | torch_npu.npu.current_blas_handle                | Unsupported            |
| 3    | torch.cuda.current_device                     | torch_npu.npu.current_device                     | Supported            |
| 4    | torch.cuda.current_stream                     | torch_npu.npu.current_stream                     | Supported            |
| 5    | torch.cuda.default_stream                     | torch_npu.npu.default_stream                     | Supported            |
| 6    | torch.cuda.device                             | torch_npu.npu.device                             | Supported            |
| 7    | torch.cuda.device_count                       | torch_npu.npu.device_count                       | Supported            |
| 8    | torch.cuda.device_of                          | torch_npu.npu.device_of                          | Supported            |
| 9    | torch.cuda.get_device_capability              | torch_npu.npu.get_device_capability              | Unsupported            |
| 10   | torch.cuda.get_device_name                    | torch_npu.npu.get_device_name                    | Unsupported            |
| 11   | torch.cuda.get_device_properties              | torch_npu.npu.get_device_properties              | Unsupported            |
| 12   | torch.cuda.get_gencode_flags                  | torch_npu.npu.get_gencode_flags                  | Supported            |
| 13   | torch.cuda.init                               | torch_npu.npu.init                               | Supported            |
| 14   | torch.cuda.ipc_collect                        | torch_npu.npu.ipc_collect                        | Unsupported            |
| 15   | torch.cuda.is_available                       | torch_npu.npu.is_available                       | Supported            |
| 16   | torch.cuda.is_initialized                     | torch_npu.npu.is_initialized                     | Supported            |
| 17   | torch.cuda.set_device                         | torch_npu.npu.set_device                         | Partially supported|
| 18   | torch.cuda.stream                             | torch_npu.npu.stream                             | Supported            |
| 19   | torch.cuda.synchronize                        | torch_npu.npu.synchronize                        | Supported            |
| 20   | torch.cuda.get_arch_list                      | torch_npu.npu.get_arch_list                      | Supported            |
| 21   | torch.cuda.get_rng_state                      | torch_npu.npu.get_rng_state                      | Supported            |
| 22   | torch.cuda.get_rng_state_all                  | torch_npu.npu.get_rng_state_all                  | Supported            |
| 23   | torch.cuda.set_rng_state                      | torch_npu.npu.set_rng_state                      | Supported            |
| 24   | torch.cuda.set_rng_state_all                  | torch_npu.npu.set_rng_state_all                  | Supported            |
| 25   | torch.cuda.manual_seed                        | torch_npu.npu.manual_seed                        | Supported            |
| 26   | torch.cuda.manual_seed_all                    | torch_npu.npu.manual_seed_all                    | Supported            |
| 27   | torch.cuda.seed                               | torch_npu.npu.seed                               | Supported            |
| 28   | torch.cuda.seed_all                           | torch_npu.npu.seed_all                           | Supported            |
| 29   | torch.cuda.initial_seed                       | torch_npu.npu.initial_seed                       | Supported            |
| 30   | torch.cuda.comm.broadcast                     | torch_npu.npu.comm.broadcast                     | Unsupported            |
| 31   | torch.cuda.comm.broadcast_coalesced           | torch_npu.npu.comm.broadcast_coalesced           | Unsupported            |
| 32   | torch.cuda.comm.reduce_add                    | torch_npu.npu.comm.reduce_add                    | Unsupported            |
| 33   | torch.cuda.comm.scatter                       | torch_npu.npu.comm.scatter                       | Unsupported            |
| 34   | torch.cuda.comm.gather                        | torch_npu.npu.comm.gather                        | Unsupported            |
| 35   | torch.cuda.Stream                             | torch_npu.npu.Stream                             | Supported            |
| 36   | torch.cuda.Stream.query                       | torch_npu.npu.Stream.query                       | Unsupported            |
| 37   | torch.cuda.Stream.record_event                | torch_npu.npu.Stream.record_event                | Supported            |
| 38   | torch.cuda.Stream.synchronize                 | torch_npu.npu.Stream.synchronize                 | Supported            |
| 39   | torch.cuda.Stream.wait_event                  | torch_npu.npu.Stream.wait_event                  | Supported            |
| 40   | torch.cuda.Stream.wait_stream                 | torch_npu.npu.Stream.wait_stream                 | Supported            |
| 41   | torch.cuda.Event                              | torch_npu.npu.Event                              | Supported            |
| 42   | torch.cuda.Event.elapsed_time                 | torch_npu.npu.Event.elapsed_time                 | Supported            |
| 43   | torch.cuda.Event.from_ipc_handle              | torch_npu.npu.Event.from_ipc_handle              | Unsupported            |
| 44   | torch.cuda.Event.ipc_handle                   | torch_npu.npu.Event.ipc_handle                   | Unsupported            |
| 45   | torch.cuda.Event.query                        | torch_npu.npu.Event.query                        | Supported            |
| 46   | torch.cuda.Event.record                       | torch_npu.npu.Event.record                       | Supported            |
| 47   | torch.cuda.Event.synchronize                  | torch_npu.npu.Event.synchronize                  | Supported            |
| 48   | torch.cuda.Event.wait                         | torch_npu.npu.Event.wait                         | Supported            |
| 49   | torch.cuda.empty_cache                        | torch_npu.npu.empty_cache                        | Supported            |
| 50   | torch.cuda.list_gpu_processes                 | torch_npu.npu.list_gpu_processes                 | Supported            |
| 51   | torch.cuda.memory_stats                       | torch_npu.npu.memory_stats                       | Supported            |
| 52   | torch.cuda.memory_summary                     | torch_npu.npu.memory_summary                     | Supported            |
| 53   | torch.cuda.memory_snapshot                    | torch_npu.npu.memory_snapshot                    | Supported            |
| 54   | torch.cuda.memory_allocated                   | torch_npu.npu.memory_allocated                   | Supported            |
| 55   | torch.cuda.max_memory_allocated               | torch_npu.npu.max_memory_allocated               | Supported            |
| 56   | torch.cuda.reset_max_memory_allocated         | torch_npu.npu.reset_max_memory_allocated         | Supported            |
| 57   | torch.cuda.memory_reserved                    | torch_npu.npu.memory_reserved                    | Supported            |
| 58   | torch.cuda.max_memory_reserved                | torch_npu.npu.max_memory_reserved                | Supported            |
| 59   | torch.cuda.set_per_process_memory_fraction    | torch_npu.npu.set_per_process_memory_fraction    | Supported            |
| 60   | torch.cuda.memory_cached                      | torch_npu.npu.memory_cached                      | Supported            |
| 61   | torch.cuda.max_memory_cached                  | torch_npu.npu.max_memory_cached                  | Supported            |
| 62   | torch.cuda.reset_max_memory_cached            | torch_npu.npu.reset_max_memory_cached            | Supported            |
| 63   | torch.cuda.nvtx.mark                          | torch_npu.npu.nvtx.mark                          | Unsupported            |
| 64   | torch.cuda.nvtx.range_push                    | torch_npu.npu.nvtx.range_push                    | Unsupported            |
| 65   | torch.cuda.nvtx.range_pop                     | torch_npu.npu.nvtx.range_pop                     | Unsupported            |
| 66   | torch.cuda.amp.autocast                       | torch_npu.npu.amp.autocast                       | Supported            |
| 67   | torch.cuda.amp.custom_fwd                     | torch_npu.npu.amp.custom_fwd                     | Supported            |
| 68   | torch.cuda.amp.custom_bwd                     | torch_npu.npu.amp.custom_bwd                     | Supported            |
| 69   | torch.cuda._sleep                             | torch_npu.npu._sleep                             | Unsupported            |
| 70   | torch.cuda.Stream.priority_range              | torch_npu.npu.Stream.priority_range              | Unsupported            |
| 71   | torch.cuda.amp.GradScaler                     | torch_npu.npu.amp.GradScaler                     | Supported            |
| 72   | torch.cuda.amp.GradScaler.get_backoff_factor  | torch_npu.npu.amp.GradScaler.get_backoff_factor  | Supported            |
| 73   | torch.cuda.amp.GradScaler.get_growth_factor   | torch_npu.npu.amp.GradScaler.get_growth_factor   | Supported            |
| 74   | torch.cuda.amp.GradScaler.get_growth_interval | torch_npu.npu.amp.GradScaler.get_growth_interval | Supported            |
| 75   | torch.cuda.amp.GradScaler.get_scale           | torch_npu.npu.amp.GradScaler.get_scale           | Supported            |
| 76   | torch.cuda.amp.GradScaler.is_enabled          | torch_npu.npu.amp.GradScaler.is_enabled          | Supported            |
| 77   | torch.cuda.amp.GradScaler.load_state_dict     | torch_npu.npu.amp.GradScaler.load_state_dict     | Supported            |
| 78   | torch.cuda.amp.GradScaler.scale               | torch_npu.npu.amp.GradScaler.scale               | Supported            |
| 79   | torch.cuda.amp.GradScaler.set_backoff_factor  | torch_npu.npu.amp.GradScaler.set_backoff_factor  | Supported            |
| 80   | torch.cuda.amp.GradScaler.set_growth_factor   | torch_npu.npu.amp.GradScaler.set_growth_factor   | Supported            |
| 81   | torch.cuda.amp.GradScaler.set_growth_interval | torch_npu.npu.amp.GradScaler.set_growth_interval | Supported            |
| 82   | torch.cuda.amp.GradScaler.state_dict          | torch_npu.npu.amp.GradScaler.state_dict          | Supported            |
| 83   | torch.cuda.amp.GradScaler.step                | torch_npu.npu.amp.GradScaler.step                | Supported            |
| 84   | torch.cuda.amp.GradScaler.unscale_            | torch_npu.npu.amp.GradScaler.unscale_            | Supported            |
| 85   | torch.cuda.amp.GradScaler.update              | torch_npu.npu.amp.GradScaler.update              | Supported            |

The **torch_npu.npu.set_device()** API can be used to specify the device only at the starting position of the program by using **set_device**. The device cannot be specified for multiple times or switched by using **with torch_npu.npu.device(id)**.  

## NPU Custom Operators

| No.| Operator                                       |
| ---- | ----------------------------------------------- |
| 1    | torch_npu.npu_convolution_transpose             |
| 2    | torch_npu.npu_conv_transpose2d                  |
| 3    | torch_npu.npu_convolution                       |
| 4    | torch_npu.npu_conv2d                            |
| 5    | torch_npu.npu_conv3d                            |
| 6    | torch_npu.one_                                  |
| 7    | torch_npu.npu_sort_v2                           |
| 8    | torch_npu.npu_format_cast                       |
| 9    | torch_npu.npu_format_cast_.src                  |
| 10   | torch_npu.npu_transpose                         |
| 11   | torch_npu.npu_broadcast                         |
| 12   | torch_npu.npu_dtype_cast                        |
| 13   | torch_npu.empty_with_format                     |
| 14   | torch_npu.copy_memory_                          |
| 15   | torch_npu.npu_one_hot                           |
| 16   | torch_npu.npu_stride_add                        |
| 17   | torch_npu.npu_softmax_cross_entropy_with_logits |
| 18   | torch_npu.npu_ps_roi_pooling                    |
| 19   | torch_npu.npu_roi_align                         |
| 20   | torch_npu.npu_nms_v4                            |
| 21   | torch_npu.npu_lstm                              |
| 22   | torch_npu.npu_iou                               |
| 23   | torch_npu.npu_ptiou                             |
| 24   | torch_npu.npu_nms_with_mask                     |
| 25   | torch_npu.npu_pad                               |
| 26   | torch_npu.npu_bounding_box_encode               |
| 27   | torch_npu.npu_bounding_box_decode               |
| 28   | torch_npu.npu_gru                               |
| 29   | torch_npu.npu_random_choice_with_mask           |
| 30   | torch_npu.npu_batch_nms                         |
| 31   | torch_npu.npu_slice                             |
| 32   | torch_npu.npu_dropoutV2                         |
| 33   | torch_npu.*npu*dropout                          |
| 34   | torch_npu.*npu*dropout_inplace                  |
| 35   | torch_npu.npu_indexing                          |
| 36   | torch_npu.npu_ifmr                              |
| 37   | torch_npu.npu_max.dim                           |
| 38   | torch_npu.npu_scatter                           |
| 39   | torch_npu.npu_apply_adam                        |
| 40   | torch_npu.npu_layer_norm_eval                   |
| 41   | torch_npu.npu_alloc_float_status                |
| 42   | torch_npu.npu_get_float_status                  |
| 43   | torch_npu.npu_clear_float_status                |
| 44   | torch_npu.npu_confusion_transpose               |
| 45   | torch_npu.npu_bmmV2                             |
| 46   | torch_npu.fast_gelu                             |
| 47   | torch_npu.npu_sub_sample                        |
| 48   | torch_npu.npu_deformable_conv2d                 |
| 49   | torch_npu.npu_mish                              |
| 50   | torch_npu.npu_anchor_response_flags             |
| 51   | torch_npu.npu_yolo_boxes_encode                 |
| 52   | torch_npu.npu_grid_assign_positive              |
| 53   | torch_npu.npu_normalize_batch                   |
| 54   | torch_npu.npu_masked_fill_range                 |
| 55   | torch_npu.npu_linear                            |
| 56   | torch_npu.npu_bert_apply_adam                   |
| 57   | torch_npu.npu_giou                              |
| 58   | torch_npu.npu_ciou                              |
| 59   | torch_npu.npu_ciou_backward                     |
| 60   | torch_npu.npu_diou                              |
| 61   | torch_npu.npu_diou_backward                     |
| 62   | torch_npu.npu_sign_bits_pack                    |
| 63   | torch_npu.npu_sign_bits_unpack                  |

Operator descriptions:

> torch_npu.npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out = (var, m, v))

count adam result.

- Parameters:
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

- Constraints:

  None

- Examples:

  None

> torch_npu.npu_convolution_transpose(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor

Applies a 2D or 3D transposed convolution operator over an input image composed of several input planes, sometimes also called "deconvolution".

- Parameters:
  - **input** (Tensor) - input tensor of shape(minibatch, in_channels, iH, iW) or (minibatch, in_channels, iT, iH, iW)
  - **weight** (Tensor) - filters of shape(in_channels, out_channels/groups, kH, kW) or (in_channels, out_channels/groups, kT, kH, kW)
  - **bias** (Tensor, optional) - optional bias of shape(out_channels)
  - **padding** (ListInt) - (dilation * (kernel_size - 1) - padding) zero-padding will be added to both sides of each dimension in the input
  - **output_padding** (ListInt) - additional size added to one side of each dimension in the output shape.
  - **stride** (ListInt) - the stride of the convolving kernel
  - **dilation** (ListInt) - the spacing between kernel elements
  - **groups** (Number) - split input into groups, in_channels should be divisible by the number of groups

- Constraints:

  None

- Examples:

  None

> torch_npu.npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor

Applies a 2D transposed convolution operator over an input image composed of several input planes, sometimes also called "deconvolution".

- Parameters:
  - **input** (Tensor) - input tensor of shape(minibatch, in_channels, iH, iW)
  - **weight** (Tensor) - filters of shape(in_channels, out_channels/groups, kH, kW)
  - **bias** (Tensor, optional) - optional bias of shape(out_channels)
  - **padding** (ListInt) - (dilation * (kernel_size - 1) - padding) zero-padding will be added to both sides of each dimension in the input
  - **output_padding** (ListInt) - additional size added to one side of each dimension in the output shape.
  - **stride** (ListInt) - the stride of the convolving kernel
  - **dilation** (ListInt) - the spacing between kernel elements
  - **groups** (Number) - split input into groups, in_channels should be divisible by the number of groups

- Constraints:

  None

- Examples:

  None

> torch_npu.npu_convolution(input, weight, bias, stride, padding, dilation, groups) -> Tensor

Applies a 2D or 3D convolution over an input image composed of several input planes.

- Parameters:
  - **input** (Tensor) - input tensor of shape(minibatch, in_channels, iH, iW) or (minibatch, in_channels, iT, iH, iW)
  - **weight** (Tensor) - filters of shape(out_channels, in_channels/groups, kH, kW) or (out_channels, in_channels/groups, kT, kH, kW)
  - **bias** (Tensor, optional) - optional bias of shape(out_channels)
  - **stride** (ListInt) - the stride of the convolving kernel
  - **padding** (ListInt) - implicit paddings on both sides of the input
  - **dilation** (ListInt) - the spacing between kernel elements
  - **groups** (ListInt) - split input into groups, in_channels should be divisible by the number of groups

- Constraints:

  None

- Examples:

  None

> torch_npu.npu_conv2d(input, weight, bias, stride, padding, dilation, groups) -> Tensor

Applies a 2D convolution over an input image composed of several input planes.

- Parameters:
  - **input** (Tensor) - input tensor of shape(minibatch, in_channels, iH, iW)
  - **weight** (Tensor) - filters of shape(out_channels, in_channels/groups, kH, kW)
  - **bias** (Tensor, optional) - optional bias of shape(out_channels)
  - **stride** (ListInt) - the stride of the convolving kernel
  - **padding** (ListInt) - implicit paddings on both sides of the input
  - **dilation** (ListInt) - the spacing between kernel elements
  - **groups** (ListInt) - split input into groups, in_channels should be divisible by the number of groups

- Constraints:

  None

- Examples:

  None

> torch_npu.npu_conv3d(input, weight, bias, stride, padding, dilation, groups) -> Tensor

Applies a 3D convolution over an input image composed of several input planes.

- Parameters:
  - **input** (Tensor) - input tensor of shape(minibatch, in_channels, iT, iH, iW)
  - **weight** (Tensor) - filters of shape(out_channels, in_channels/groups, kT, kH, kW)
  - **bias** (Tensor, optional) - optional bias of shape(out_channels)
  - **stride** (ListInt) - the stride of the convolving kernel
  - **padding** (ListInt) - implicit paddings on both sides of the input
  - **dilation** (ListInt) - the spacing between kernel elements
  - **groups** (ListInt) - split input into groups, in_channels should be divisible by the number of groups

- Constraints:

  None

- Examples:

  None

> torch_npu.one_(self) -> Tensor

Fills self tensor with ones.

- Parameters:
  
- **self** (Tensor) - input tensor
  
- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.rand(2, 3).npu()
  >>> x
  tensor([[0.6072, 0.9726, 0.3475],
          [0.3717, 0.6135, 0.6788]], device='npu:0')
  >>> x.one_()
  tensor([[1., 1., 1.],
          [1., 1., 1.]], device='npu:0')
  ```

> torch_npu.npu_sort_v2(self, dim=-1, descending=False, out=None) -> Tensor

Sorts the elements of the input tensor along a given dimension in ascending order by value without indices.
If dim is not given, the last dimension of the input is chosen.
If descending is True then the elements are sorted in descending order by value.

- Parameters:
  - **self** (Tensor) - the input tensor
  - **dim** (int, optional) - the dimension to sort along
  - **descending** (bool, optional) - controls the sorting order (ascending or descending)
  
- Constraints:

  At present only support the last dim(-1).

- Examples:

  ```python
  >>> x = torch.randn(3, 4).npu()
  >>> x
  tensor([[-0.0067,  1.7790,  0.5031, -1.7217],
          [ 1.1685, -1.0486, -0.2938,  1.3241],
          [ 0.1880, -2.7447,  1.3976,  0.7380]], device='npu:0')
  >>> sorted_x = torch_npu.npu_sort_v2(x)
  >>> sorted_x
  tensor([[-1.7217, -0.0067,  0.5029,  1.7793],
          [-1.0488, -0.2937,  1.1689,  1.3242],
          [-2.7441,  0.1880,  0.7378,  1.3975]], device='npu:0')
  ```

> torch_npu.npu_format_cast(self, acl_format) -> Tensor

Change the format of a npu tensor.

- Parameters:
  - **self** (Tensor) - the input tensor
  - **acl_format** (int) - the target format to transform

- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.rand(2, 3, 4, 5).npu()
  >>> torch_npu.get_npu_format(x)
  0
  >>> x1 = x.npu_format_cast(29)
  >>> torch_npu.get_npu_format(x1)
  29
  ```

> torch_npu.npu_format_cast_

>   torch_npu.npu_format_cast_.src(self, src) -> Tensor

  In-place Change the format of self, with the same format as src.

  - Parameters:
    - **self** (Tensor) - the input tensor
    - **src** (Tensor) - the target format to transform

  - Constraints:

    None

  - Examples:

    ```python
    >>> x = torch.rand(2, 3, 4, 5).npu()
    >>> torch_npu.get_npu_format(x)
    0
    >>> torch_npu.get_npu_format(x.npu_format_cast_(29))
    29
    ```

> torch_npu.npu_transpose(self, perm, bool require_contiguous=True) -> Tensor

Returns a view of the original tensor with its dimensions permuted, and make the result contiguous.

- Parameters:
  - **self** (Tensor) - the input tensor
  - **perm** (ListInt) - The desired ordering of dimensions
  - **require_contiguous** (bool) - Used to specify whether trans-contiguous of self is needed.

- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.randn(2, 3, 5).npu()
  >>> x.shape
  torch.Size([2, 3, 5])
  >>> x1 = torch_npu.npu_transpose(x, (2, 0, 1))
  >>> x1.shape
  torch.Size([5, 2, 3])
  >>> x2 = x.npu_transpose(2, 0, 1)
  >>> x2.shape
  torch.Size([5, 2, 3])
  ```

> torch_npu.npu_broadcast(self, perm) -> Tensor

Returns a new view of the self tensor with singleton dimensions expanded to a larger size, and make the result contiguous.

Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front.

- Parameters:
  - **self** (Tensor) - the input tensor
  - **perm** (ListInt) - the desired expanded size

- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.tensor([[1], [2], [3]]).npu()
  >>> x.shape
  torch.Size([3, 1])
  >>> x.npu_broadcast(3, 4)
  tensor([[1, 1, 1, 1],
          [2, 2, 2, 2],
          [3, 3, 3, 3]], device='npu:0')
  ```
  
> torch_npu.npu_dtype_cast(input, dtype) -> Tensor

Performs Tensor dtype conversion.

- Parameters:
  - **input** (Tensor) - the input tensor.
  - **dtype** (torch.dtype) - the desired data type of returned Tensor.

- Constraints:

  None

- Examples:

  ```python
  >>> torch_npu.npu_dtype_cast(torch.tensor([0, 0.5, -1.]).npu(), dtype=torch.int)
  tensor([ 0,  0, -1], device='npu:0', dtype=torch.int32)
  ```

> torch_npu.empty_with_format(size, dtype, layout, device, pin_memory, acl_format) -> Tensor

Returns a tensor filled with uninitialized data. The shape of the tensor is defined by the variable argument size. The format of the tensor is defined by the variable argument acl_format.

- Parameters:

  - **size** (int...) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.

  - **dtype** (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_tensor_type()).

  - **layout** (torch.layout, optional) – the desired layout of returned Tensor. Default: None.

  - **device** (torch.device, optional) – the desired device of returned tensor. Default: None

  - **pin_memory** (bool, optional) – If set, returned tensor would be allocated in the pinned memory. Default: None.

  - **acl_format** (Number) – the desired memory format of returned Tensor. Default: 2.

- Constraints:

  None

- Examples:
  ```python
  >>> torch_npu.empty_with_format((2, 3), dtype=torch.float32, device="npu")
  tensor([[1., 1., 1.],
          [1., 1., 1.]], device='npu:0')
  ```

> torch_npu.copy_memory_(dst, src, non_blocking=False) -> Tensor

Copies the elements from src into self tensor and returns self.

- Parameters:
  - **dst** (Tensor) - the source tensor to copy from.
  - **src** (Tensor) - the desired data type of returned Tensor.
  - **non_blocking** (bool) - if True and this copy is between CPU and NPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect.

- Constraints:

  copy_memory_ only support npu tensor.
  input tensors of copy_memory_ should have same dtype.
  input tensors of copy_memory_ should have same device index.

- Examples:

  ```python
  >>> a=torch.IntTensor([0,  0, -1]).npu()
  >>> b=torch.IntTensor([1, 1, 1]).npu()
  >>> a.copy_memory_(b)
  tensor([1, 1, 1], device='npu:0', dtype=torch.int32)
  ```

> torch_npu.npu_one_hot(input, num_classes=-1, depth=1, on_value=1, off_value=0) -> Tensor

Returns a one-hot tensor. The locations represented by index in "x" take value "on_value", while all other locations take value "off_value". 

- Parameters:
  - **input** (Tensor) - class values of any shape.
  - **num_classes** (int) - The axis to fill. Defaults to "-1". 
  - **depth** (Number) - The depth of the one hot dimension. 
  - **on_value** (Number) - The value to fill in output when indices[j] = i.
  - **off_value** (Number) - The value to fill in output when indices[j] != i.

- Constraints:

  None

- Examples:
  ```python
  >>> a=torch.IntTensor([5, 3, 2, 1]).npu()
  >>> b=torch_npu.npu_one_hot(a, depth=5)
  >>> b
  tensor([[0., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0.],
          [0., 0., 1., 0., 0.],
          [0., 1., 0., 0., 0.]], device='npu:0')
  ```

> torch_npu.npu_stride_add(x1, x2, offset1, offset2, c1_len) -> Tensor

Add the partial values of two tensors in format NC1HWC0. 

- Parameters:
  - **x1** (Tensor) -  A Tensor in 5HD.
  - **x2** (Tensor) - A Tensor of the same type as "x1", and the same shape as "x1", except for the C1 value.  
  - **offset1** (Number) - A required int. Offset value of C1 in "x1". 
  - **offset2** (Number) - A required int. Offset value of C1 in "x2". 
  - **c1_len** (Number) - A required int. C1 len of "y". The value must be less than the difference between C1 and offset in "x1" and "x2". 

- Constraints:

  None

- Examples:
  ```python
  >>> a=torch.tensor([[[[[1.]]]]]).npu()
  >>> b=torch_npu.npu_stride_add(a, a, 0, 0, 1)
  >>> b
  tensor([[[[[2.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]],
          [[[0.]]]]], device='npu:0')
  ```

> torch_npu.npu_softmax_cross_entropy_with_logits(features, labels) -> Tensor

Computes softmax cross entropy cost.

- Parameters:
  - **features** (Tensor) - A Tensor.  A "batch_size * num_classes" matrix. 
  - **labels** (Tensor) - A Tensor of the same type as "features". A "batch_size * num_classes" matrix. 

- Constraints:

  None

- Examples:

  None

> torch_npu.npu_ps_roi_pooling(x, rois, spatial_scale, group_size, output_dim) -> Tensor

Performs Position Sensitive PS ROI Pooling. 

- Parameters:
  - **x** (Tensor) - An NC1HWC0 tensor, describing the feature map, dimension C1 must be equal to (int(output_dim+15)/C0))*group_size*group_size. 
  - **rois** (Tensor) - A tensor with shape [batch, 5, rois_num], describing the ROIs, each ROI consists of five elements: "batch_id", "x1", "y1", "x2", and "y2", which "batch_id" indicates the index of the input feature map, "x1", "y1", "x2", or "y2" must be greater than or equal to "0.0".  
  - **spatial_scale** (Number) - A required float32, scaling factor for mapping the input coordinates to the ROI coordinates . 
  - **group_size** (Number) - A required int32, specifying the number of groups to encode position-sensitive score maps, must be within the range (0, 128). 
  - **output_dim** (Number) - A required int32, specifying the number of output channels, must be greater than 0. 

- Constraints:

  None

- Examples:
  ```python
  >>> roi = torch.tensor([[[1], [2], [3], [4], [5]],
                          [[6], [7], [8], [9], [10]]], dtype = torch.float16).npu()
  >>> x = torch.tensor([[[[ 1]], [[ 2]], [[ 3]], [[ 4]],
                        [[ 5]], [[ 6]], [[ 7]], [[ 8]]],
                        [[[ 9]], [[10]], [[11]], [[12]],
                        [[13]], [[14]], [[15]], [[16]]]], dtype = torch.float16).npu()
  >>> out = torch_npu.npu_ps_roi_pooling(x, roi, 0.5, 2, 2)
  >>> out
  tensor([[[[0., 0.],
            [0., 0.]],
          [[0., 0.],
            [0., 0.]]],
          [[[0., 0.],
            [0., 0.]],
          [[0., 0.],
            [0., 0.]]]], device='npu:0', dtype=torch.float16)
  ```

> torch_npu.npu_roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode) -> Tensor

Obtains the ROI feature matrix from the feature map. It is a customized FasterRcnn operator. 

- Parameters:
  - **features** (Tensor) -  A Tensor in 5HD.
  - **rois** (Tensor) - ROI position. A 2D Tensor with shape (N, 5). "N" indicates the number of ROIs, the value "5" indicates the indexes of images where the ROIs are located, "x0", "y0", "x1", and "y1". 
  - **spatial_scale** (Number) - A required attribute of type float32, specifying the scaling ratio of "features" to the original image. 
  - **pooled_height** (Number) - A required attribute of type int32, specifying the H dimension. 
  - **pooled_width** (Number) - A required attribute of type int32, specifying the W dimension. 
  - **sample_num** (Number) - An optional attribute of type int32, specifying the horizontal and vertical sampling frequency of each output. If this attribute is set to "0", the sampling frequency is equal to the rounded up value of "rois", which is a floating point number. Defaults to "2".
  - **roi_end_mode** (Number) - An optional attribute of type int32. Defaults to "1".

- Constraints:

  None

- Examples:
  ```python
  >>> x = torch.FloatTensor([[[[1, 2, 3 , 4, 5, 6],
                              [7, 8, 9, 10, 11, 12],
                              [13, 14, 15, 16, 17, 18],
                              [19, 20, 21, 22, 23, 24],
                              [25, 26, 27, 28, 29, 30],
                              [31, 32, 33, 34, 35, 36]]]]).npu()
  >>> rois = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
  >>> out = torch_npu.npu_roi_align(x, rois, 0.25, 3, 3, 2, 0)
  >>> out
  tensor([[[[ 4.5000,  6.5000,  8.5000],
            [16.5000, 18.5000, 20.5000],
            [28.5000, 30.5000, 32.5000]]]], device='npu:0')
  ```

> torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size=False) -> (Tensor, Tensor)

Greedily selects a subset of bounding boxes in descending order of score. 

- Parameters:
  - **boxes** (Tensor) -  A 2-D float tensor of shape [num_boxes, 4]. 
  - **scores** (Tensor) - A 1-D float tensor of shape [num_boxes] representing a single score corresponding to each box (each row of boxes). 
  - **max_output_size** (Number) - A scalar representing the maximum number of boxes to be selected by non max suppression.
  - **iou_threshold** (Tensor) - A 0-D float tensor representing the threshold for deciding whether boxes overlap too much with respect to IOU. 
  - **scores_threshold** (Tensor) -  A 0-D float tensor representing the threshold for deciding when to remove boxes based on score. 
  - **pad_to_max_output_size** (bool) - If true, the output selected_indices is padded to be of length max_output_size. Defaults to false. 

- Returns:
  - **selected_indices** - A 1-D integer tensor of shape [M] representing the selected indices from the boxes tensor, where M <= max_output_size. 
  - **valid_outputs** - A 0-D integer tensor representing the number of valid elements in selected_indices, with the valid elements appearing first. 

- Constraints:

  None

- Examples:
  ```python
  >>> boxes=torch.randn(100,4).npu()
  >>> scores=torch.randn(100).npu()
  >>> boxes.uniform_(0,100)
  >>> scores.uniform_(0,1)
  >>> max_output_size = 20
  >>> iou_threshold = torch.tensor(0.5).npu()
  >>> scores_threshold = torch.tensor(0.3).npu()
  >>> npu_output = torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold)
  >>> npu_output
  (tensor([57, 65, 25, 45, 43, 12, 52, 91, 23, 78, 53, 11, 24, 62, 22, 67,  9, 94,
          54, 92], device='npu:0', dtype=torch.int32), tensor(20, device='npu:0', dtype=torch.int32))
  ```

> torch_npu.npu_nms_rotated(dets, scores, iou_threshold, scores_threshold=0, max_output_size=-1, mode=0) -> (Tensor, Tensor)

Greedy selects a subset of the rotated bounding boxes in descending fractional order.

- Parameters:
  - **dets** (Tensor) -  A 2-D float tensor of shape [num_boxes, 5]. 
  - **scores** (Tensor) - A 1-D float tensor of shape [num_boxes] representing a single score corresponding to each box (each row of boxes). 
  - **iou_threshold** (Number) - A scalar representing the threshold for deciding whether boxes overlap too much with respect to IOU.   
  - **scores_threshold** (Number) -  A scalar representing the threshold for deciding when to remove boxes based on score. Defaults to "0". 
  - **max_output_size** (Number) - A scalar integer tensor representing the maximum number of boxes to be selected by non max suppression. Defaults to "-1", that is, no constraint is imposed. 
  - **mode** (Number) - This parameter specifies the layout type of the dets. The default value is 0. If mode is set to 0, the input values of dets are x, y, w, h, and angle. If mode is set to 1, the input values of dets are x1, y1, x2, y2, and angle. Defaults to "0".

- Returns:
  - **selected_index** - A 1-D integer tensor of shape [M] representing the selected indices from the dets tensor, where M <= max_output_size. 
  - **selected_num** - A 0-D integer tensor representing the number of valid elements in selected_indices. 

- Constraints:

  None

- Examples:
  ```python
  >>> dets=torch.randn(100,5).npu()
  >>> scores=torch.randn(100).npu()
  >>> dets.uniform_(0,100)
  >>> scores.uniform_(0,1)
  >>> output1, output2 = torch_npu.npu_nms_rotated(dets, scores, 0.2, 0, -1, 1)
  >>> output1
  tensor([76, 48, 15, 65, 91, 82, 21, 96, 62, 90, 13, 59,  0, 18, 47, 23,  8, 56,
          55, 63, 72, 39, 97, 81, 16, 38, 17, 25, 74, 33, 79, 44, 36, 88, 83, 37,
          64, 45, 54, 41, 22, 28, 98, 40, 30, 20,  1, 86, 69, 57, 43,  9, 42, 27,
          71, 46, 19, 26, 78, 66,  3, 52], device='npu:0', dtype=torch.int32)
  >>> output2
  tensor([62], device='npu:0', dtype=torch.int32)
  ```

> torch_npu.npu_lstm(x, weight, bias, seq_len, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction)

DynamicRNN calculation. 

- Parameters:
  - **x** (Tensor) -  A required 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ.  
  - **weight** (Tensor) - A required 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_ZN_LSTM.  
  - **bias** (Tensor) -  A required 1D Tensor. Must be one of the following types: float16, float32. The format must be ND.
  - **seq_len** (Tensor) - A optional Tensor. Only Support float16 in FRACTAL_NZ and int32 in ND.
  - **h** (Tensor) -  A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **c** (Tensor) - A optional 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **has_biases** (bool) -  If the value is true, bias exists. 
  - **num_layers** (Number) - Number of recurrent layers. Only Support single layer currently.
  - **dropout** (Number) - If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. unsupport currently.  
  - **train** (bool) -  An bool identifying is training in the op. Default to true . 
  - **bidirectional** (bool) - If True, becomes a bidirectional LSTM. unsupport currently.
  - **batch_first** (bool) - If True, then the input and output tensors are provided as (batch, seq, feature). unsupport currently.
  - **flag_seq** (bool) - If True, then the input is PackSequnce. unsupport currently.
  - **direction** (bool) - If True, then the direction is "REDIRECTIONAL", otherwise is "UNIDIRECTIONAL".

- Returns:
  - **y** - A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **output_h** - A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **output_c** - A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **i** - A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **j** - A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **f** - A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **o** - A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **tanhct** - A 4D Tensor. Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 

- Constraints:

  None

- Examples:
  
  None

>torch_npu.npu_iou(bboxes, gtboxes, mode=0) -> Tensor
>torch_npu.npu_ptiou(bboxes, gtboxes, mode=0) -> Tensor

Computes the intersection over union (iou) or the intersection over. foreground (iof) based on the ground-truth and predicted regions.

- Parameters:
  - **bboxes** (Tensor) - the input tensor.
  - **gtboxes** (Tensor) - the input tensor.
  - **mode** (Number) - 0 1 corresponds to two modes iou iof.

- Constraints:

  None

- Examples:

  ```python
  >>> bboxes = torch.tensor([[0, 0, 10, 10],
                             [10, 10, 20, 20],
                             [32, 32, 38, 42]], dtype=torch.float16).to("npu")
  >>> gtboxes = torch.tensor([[0, 0, 10, 20],
                              [0, 10, 10, 10],
                              [10, 10, 20, 20]], dtype=torch.float16).to("npu")
  >>> output_iou = torch_npu.npu_iou(bboxes, gtboxes, 0)
  >>> output_iou
  tensor([[0.4985, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.9961, 0.0000]], device='npu:0', dtype=torch.float16)
  ```

>torch_npu.npu_pad(input, paddings) -> Tensor

Pads a tensor

- Parameters:
  - **input** (Tensor) - the input tensor.
  - **paddings** (ListInt) -  type int32 or int64.

- Constraints:

  None

- Examples:

  ```python
  >>> input = torch.tensor([[20, 20, 10, 10]], dtype=torch.float16).to("npu")
  >>> paddings = [1, 1, 1, 1]
  >>> output = torch_npu.npu_pad(input, paddings)
  >>> output
  tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
          [ 0., 20., 20., 10., 10.,  0.],
          [ 0.,  0.,  0.,  0.,  0.,  0.]], device='npu:0', dtype=torch.float16)
  ```

>torch_npu.npu_nms_with_mask(input, iou_threshold) -> (Tensor, Tensor, Tensor)

The value 01 is generated for the nms operator to determine the valid bit

- Parameters:
  - **input** (Tensor) - the input tensor.
  - **iou_threshold** (Number) -  Threshold. If the value exceeds this threshold, the value is 1. Otherwise, the value is 0.

- Returns:

  - **selected_boxes** - 2-D tensor with shape of [N,5], representing filtered boxes including proposal boxes and corresponding confidence scores. 
  - **selected_idx** - 1-D tensor with shape of [N], representing the index of input proposal boxes. 
  - **selected_mask** - 1-D tensor with shape of [N], the symbol judging whether the output proposal boxes is valid . 

- Constraints:

  The 2nd-dim of input box_scores must be equal to 8.

- Examples:

  ```python
  >>> input = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.6], [6.0, 7.0, 8.0, 9.0, 0.4]], dtype=torch.float16).to("npu")
  >>> iou_threshold = 0.5
  >>> output1, output2, output3, = torch_npu.npu_nms_with_mask(input, iou_threshold)
  >>> output1
  tensor([[0.0000, 1.0000, 2.0000, 3.0000, 0.6001],
          [6.0000, 7.0000, 8.0000, 9.0000, 0.3999]], device='npu:0',
        dtype=torch.float16)
  >>> output2
  tensor([0, 1], device='npu:0', dtype=torch.int32)
  >>> output3
  tensor([1, 1], device='npu:0', dtype=torch.uint8)
  ```

>torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3) -> Tensor

Computes the coordinate variations between bboxes and ground truth boxes. It is a customized FasterRcnn operator

- Parameters:
  - **anchor_box** (Tensor) - the input tensor.Anchor boxes. A 2D Tensor of float32 with shape (N, 4). "N" indicates the number of bounding boxes, and the value "4" refers to "x0", "x1", "y0", and "y1". 
  - **ground_truth_box** (Tensor) -  the input tensor.Ground truth boxes. A 2D Tensor of float32 with shape (N, 4). "N" indicates the number of bounding boxes, and the value "4" refers to "x0", "x1", "y0", and "y1" 
  - **means0** (Number) -  An index of type int
  - **means1** (Number) -  An index of type int
  - **means2** (Number) - An index of type int
  - **means3** (Number) -  An index of type int. Defaults to [0,0,0,0]. "deltas" = "deltas" x "stds" + "means". 
  - **stds0** (Number) - An index of type int
  - **stds1** (Number) - An index of type int
  - **stds2** (Number) - An index of type int
  - **stds3** (Number) - An index of type int  Defaults to [1.0,1.0,1.0,1.0]. "deltas" = "deltas" x "stds" + "means" . 

- Constraints:

  None

- Examples:

  ```python
  >>> anchor_box = torch.tensor([[1., 2., 3., 4.], [3.,4., 5., 6.]], dtype = torch.float32).to("npu")
  >>> ground_truth_box = torch.tensor([[5., 6., 7., 8.], [7.,8., 9., 6.]], dtype = torch.float32).to("npu")
  >>> output = torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)
  >>> output
  tensor([[13.3281, 13.3281,  0.0000,  0.0000],
          [13.3281,  6.6641,  0.0000, -5.4922]], device='npu:0')
  >>>
  ```

>torch_npu.npu_bounding_box_decode(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip) -> Tensor

Generates bounding boxes based on "rois" and "deltas". It is a customized FasterRcnn operator .

- Parameters:
  - **rois** (Tensor) - Region of interests (ROIs) generated by the region proposal network (RPN). A 2D Tensor of type float32 or float16 with shape (N, 4). "N" indicates the number of ROIs, and the value "4" refers to "x0", "x1", "y0", and "y1". 
  - **deltas** (Tensor) -  Absolute variation between the ROIs generated by the RPN and ground truth boxes. A 2D Tensor of type float32 or float16 with shape (N, 4). "N" indicates the number of errors, and 4 indicates "dx", "dy", "dw", and "dh" . 
  - **means0** (Number) -  An index of type int
  - **means1** (Number) -  An index of type int
  - **means2** (Number) - An index of type int
  - **means3** (Number) -  An index of type int. Defaults to [0,0,0,0]. "deltas" = "deltas" x "stds" + "means". 
  - **stds0** (Number) - An index of type int
  - **stds1** (Number) - An index of type int
  - **stds2** (Number) - An index of type int 
  - **stds3** (Number) - An index of type int  Defaults to [1.0,1.0,1.0,1.0]. "deltas" = "deltas" x "stds" + "means" .
  - **max_shape** (ListInt) - Shape [h, w], specifying the size of the image transferred to the network. Used to ensure that the bbox shape after conversion does not exceed "max_shape
  - **wh_ratio_clip** (Number) - Defaults to "16/1000". The values of "dw" and "dh" fall within (-wh_ratio_clip, wh_ratio_clip) . 

- Constraints:

  None

- Examples:

  ```python
  >>> rois = torch.tensor([[1., 2., 3., 4.], [3.,4., 5., 6.]], dtype = torch.float32).to("npu")
  >>> deltas = torch.tensor([[5., 6., 7., 8.], [7.,8., 9., 6.]], dtype = torch.float32).to("npu")
  >>> output = torch_npu.npu_bounding_box_decode(rois, deltas, 0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)
  >>> output
  tensor([[2.5000, 6.5000, 9.0000, 9.0000],
          [9.0000, 9.0000, 9.0000, 9.0000]], device='npu:0')
  ```

>torch_npu.npu_gru(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)

DynamicGRUV2 calculation.

- Parameters:
  - **input** (Tensor) - Must be one of the following types: float16. The format must be FRACTAL_NZ. 
  - **hx** (Tensor) -  Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **weight_input** (Tensor) -  Must be one of the following types: float16. The format must be FRACTAL_Z. 
  - **weight_hidden** (Tensor) -  Must be one of the following types: float16. The format must be FRACTAL_Z. 
  - **bias_input** (Tensor) -  Must be one of the following types: float16, float32. The format must be ND. 
  - **bias_hidden** (Tensor) -  Must be one of the following types: float16, float32. The format must be ND. 
  - **seq_length** (Tensor) - Must be one of the following types: int32. The format must be ND. 
  - **has_biases** (bool) - Default to true.
  - **num_layers** (Number)
  - **dropout** (Number)
  - **train** (bool) - An bool identifying is training in the op. Default to true.
  - **bidirectional** (bool) - Default to true.
  - **batch_first** (bool) - Default to true.

- Returns:

  - **y** (Tensor) - Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **output_h** (Tensor) - output_h:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **update** (Tensor) - update:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **reset** (Tensor) - reset:Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **new** (Tensor) - Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 
  - **hidden_new** (Tensor) - Must be one of the following types: float16, float32. The format must be FRACTAL_NZ. 

- Constraints:

  None

- Examples:
  
  None

>torch_npu.npu_random_choice_with_mask(x, count=256, seed=0, seed2=0) -> (Tensor, Tensor)

Shuffle index of no-zero element

- Parameters:
  - **x** (Tensor) - the input tensor.
  - **count** (Number) -  the count of output, if 0, out all no-zero elements.
  - **seed** (Number) -  type int32 or int64.
  - **seed2** (Number) -  type int32 or int64.

- Returns:

  - **y** - 2-D tensor, no-zero element index. 
  - **mask** - 1-D, whether the corresponding index is valid. 

- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.tensor([1, 0, 1, 0], dtype=torch.bool).to("npu")
  >>> result, mask = torch_npu.npu_random_choice_with_mask(x, 2, 1, 0)
  >>> result
  tensor([[0],
          [2]], device='npu:0', dtype=torch.int32)
  >>> mask
  tensor([True, True], device='npu:0')
  ```

>torch_npu.npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame=False, transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)

Computes nms for input boxes and score, support multiple batch and classes. will do clip to window, score filter, top_k, and nms

- Parameters:
  - **self** (Tensor) - the input tensor.
  - **scores** (Tensor) -  the input tensor.
  - **score_threshold** (Number) -  A required attribute of type float32, specifying the score filter iou iou_threshold.
  - **iou_threshold** (Number) -  A required attribute of type float32, specifying the nms iou iou_threshold.
  - **max_size_per_class** (Number) - A required attribute of type int, specifying the nms output num per class.
  - **max_total_size** (Number) -  A required attribute of type int, specifying the the nms output num per batch.
  - **change_coordinate_frame** (bool) - A optional attribute of type bool, whether to normalize coordinates after clipping.
  - **transpose_box** (bool) - A optional attribute of type bool, whether inserted transpose before this op. must be "false"

- Returns:

  - **nmsed_boxes** (Tensor) - A 3D Tensor of type float16 with shape (batch, max_total_size, 4),specifying the output nms boxes per batch.
  - **nmsed_scores** (Tensor) - A 2D Tensor of type float16 with shape (batch, max_total_size),specifying the output nms score per batch.
  - **nmsed_classes** (Tensor) - A 2D Tensor of type float16 with shape (batch, max_total_size),specifying the output nms class per batch.
  - **nmsed_num** (Tensor) - A 1D Tensor of type int32 with shape (batch), specifying the valid num of nmsed_boxes.

- Constraints:

  None

- Examples:

  ```python
  >>> boxes = torch.randn(8, 2, 4, 4, dtype = torch.float32).to("npu")
  >>> scores = torch.randn(3, 2, 4, dtype = torch.float32).to("npu")
  >>> nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 3, 4)
  >>> nmsed_boxes
  >>> nmsed_scores
  >>> nmsed_classes
  >>> nmsed_num
  ```

>torch_npu.npu_slice(self, offsets, size) -> Tensor

Extracts a slice from a tensor

- Parameters:
  - **self** (Tensor) - the input tensor.
  - **offsets** (ListInt) -  type int32 or int64.
  - **size** (ListInt) -  type int32 or int64.

- Constraints:

  None

- Examples:

  ```python
  >>> input = torch.tensor([[1,2,3,4,5], [6,7,8,9,10]], dtype=torch.float16).to("npu")
  >>> offsets = [0, 0]
  >>> size = [2, 2]
  >>> output = torch_npu.npu_slice(input, offsets, size)
  >>> output
  tensor([[1., 2.],
          [6., 7.]], device='npu:0', dtype=torch.float16)
  ```
  
>torch_npu.npu_dropoutV2(self, seed, p) -> (Tensor, Tensor, Tensor(a!))

count dropout result with seed

- Parameters:
  - **self** (Tensor) - The input Tensor.
  - **seed** (Tensor) - The input Tensor.
  - **p** (Float) - Dropout probability.

- Returns:

  - **y**  - A tensor with the same shape and type as "x". 
  - **mask**  - A tensor with the same shape and type as "x". 
  - **new_seed**  - A tensor with the same shape and type as "seed". 

- Constraints:

  None

- Examples:

  ```python
  >>> input = torch.tensor([1.,2.,3.,4.]).npu()
  >>> input
  tensor([1., 2., 3., 4.], device='npu:0')
  >>> seed = torch.rand((32,),dtype=torch.float32).npu()
  >>> seed
  tensor([0.4368, 0.7351, 0.8459, 0.4657, 0.6783, 0.8914, 0.8995, 0.4401, 0.4408,
        0.4453, 0.2404, 0.9680, 0.0999, 0.8665, 0.2993, 0.5787, 0.0251, 0.6783,
        0.7411, 0.0670, 0.9430, 0.9165, 0.3983, 0.5849, 0.7722, 0.4659, 0.0486,
        0.2693, 0.6451, 0.2734, 0.3176, 0.0176], device='npu:0')
  >>> prob = 0.3
  >>> output, mask, out_seed = torch_npu.npu_dropoutV2(input, seed, prob)
  >>> output
  tensor([0.4408, 0.4453, 0.2404, 0.9680], device='npu:0')
  >>> mask
  tensor([0., 0., 0., 0.], device='npu:0')
  >>> out_seed
  tensor([0.4408, 0.4453, 0.2404, 0.9680, 0.0999, 0.8665, 0.2993, 0.5787, 0.0251,
          0.6783, 0.7411, 0.0670, 0.9430, 0.9165, 0.3983, 0.5849, 0.7722, 0.4659,
          0.0486, 0.2693, 0.6451, 0.2734, 0.3176, 0.0176, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000], device='npu:0')
  ```

>torch_npu.*npu*dropout(self, p) -> (Tensor, Tensor)

count dropout result without seed

- Parameters:
  Similar to `torch.dropout`, optimize implemention to npu device.
  - **self** (Tensor) - The input Tensor.
  - **p** (Float) - Dropout probability.

- Constraints:

  None

- Examples:

  ```python
  >>> input = torch.tensor([1.,2.,3.,4.]).npu()
  >>> input
  tensor([1., 2., 3., 4.], device='npu:0')
  >>> prob = 0.3
  >>> output, mask = torch_npu._npu_dropout(input, prob)
  >>> output
  tensor([0.0000, 2.8571, 0.0000, 0.0000], device='npu:0')
  >>> mask
  tensor([ 98, 255, 188, 186, 120, 157, 175, 159,  77, 223, 127,  79, 247, 151,
        253, 255], device='npu:0', dtype=torch.uint8)
  ```

>torch_npu.*npu*dropout_inplace(result, p) -> (Tensor(a!), Tensor)

count dropout result inplace.

- Parameters:
  Similar to `torch.dropout_`, optimize implemention to npu device.
  - **result** (Tensor) - The Tensor dropout inplace.
  - **p** (Float) - Dropout probability.

- Constraints:

  None

- Examples:

  ```python
  >>> input = torch.tensor([1.,2.,3.,4.]).npu()
  >>> input
  tensor([1., 2., 3., 4.], device='npu:0')
  >>> prob = 0.3
  >>> output, mask = torch_npu._npu_dropout_inplace(input, prob)
  >>> output
  tensor([0.0000, 2.8571, 0.0000, 0.0000], device='npu:0')
  >>> input
  tensor([0.0000, 2.8571, 4.2857, 5.7143], device='npu:0')
  >>> mask
  tensor([ 98, 255, 188, 186, 120, 157, 175, 159,  77, 223, 127,  79, 247, 151,
        253, 255], device='npu:0', dtype=torch.uint8)
  ```

>torch_npu.npu_indexing(self, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0) -> Tensor

count indexing result by begin,end,strides array.

- Parameters:
  - **self** (Tensor) - A Input Tensor.
  - **begin** (ListInt) - The index of the first value to select.
  - **end** (ListInt) - The index of the last value to select.
  - **strides** (ListInt) - The index increment.
  - **begin_mask** (Number) - A bitmask where a bit "i" being "1" means to ignore the begin
    value and instead use the largest interval possible.
  - **end_mask** (Number) - Analogous to "begin_mask".
  - **ellipsis_mask** (Number) - A bitmask where bit "i" being "1" means the "i"th position
    is actually an ellipsis.
  - **new_axis_mask** (Number) - A bitmask where bit "i" being "1" means the "i"th
    specification creates a new shape 1 dimension.
  - **shrink_axis_mask** (Number) - A bitmask where bit "i" implies that the "i"th
    specification should shrink the dimensionality.

- Constraints:

  None

- Examples:

  ```python
  >>> input = torch.tensor([[1, 2, 3, 4],[5, 6, 7, 8]], dtype=torch.int32).to("npu")
  >>> input
  tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]], device='npu:0', dtype=torch.int32)
  >>> output = torch_npu.npu_indexing(input1, [0, 0], [2, 2], [1, 1])
  >>> output
  tensor([[1, 2],
        [5, 6]], device='npu:0', dtype=torch.int32)
  ```

>torch_npu.npu_ifmr(Tensor data, Tensor data_min, Tensor data_max, Tensor cumsum, float min_percentile, float max_percentile, float search_start, float search_end, float search_step, bool with_offset) -> (Tensor, Tensor)

count ifmr result by begin,end,strides array, Input Feature Map Reconstruction

- Parameters:
  - **data** (Tensor) - A Tensor of feature map.
  - **data_min** (Tensor) - A Tensor of min value of feature map.
  - **data_max** (Tensor) - A Tensor of max value of feature map.
  - **cumsum** (Tensor) - A Tensor of cumsum bin of data.
  - **min_percentile** (Float) - min init percentile.
  - **max_percentile** (Float) - max init percentile.
  - **search_start** (Float) - search start.
  - **search_end** (Float) -  search end.
  - **search_step** (Float) -  step size of searching.
  - **with_offset** (bool) -  whether using offset.
  
- Returns:

  - **scale** - optimal scale. 
  - **offset** - optimal offset . 

- Constraints:

  None

- Examples:

  ```python
  >>> input = torch.rand((2,2,3,4),dtype=torch.float32).npu()
  >>> input
  tensor([[[[0.4508, 0.6513, 0.4734, 0.1924],
            [0.0402, 0.5502, 0.0694, 0.9032],
            [0.4844, 0.5361, 0.9369, 0.7874]],
  
          [[0.5157, 0.1863, 0.4574, 0.8033],
            [0.5986, 0.8090, 0.7605, 0.8252],
            [0.4264, 0.8952, 0.2279, 0.9746]]],
  
          [[[0.0803, 0.7114, 0.8773, 0.2341],
            [0.6497, 0.0423, 0.8407, 0.9515],
            [0.1821, 0.5931, 0.7160, 0.4968]],
    
          [[0.7977, 0.0899, 0.9572, 0.0146],
            [0.2804, 0.8569, 0.2292, 0.1118],
            [0.5747, 0.4064, 0.8370, 0.1611]]]], device='npu:0')
    >>> min_value = torch.min(input)
    >>> min_value
    tensor(0.0146, device='npu:0')
    >>> max_value = torch.max(input)
    >>> max_value
    tensor(0.9746, device='npu:0')
    >>> hist = torch.histc(input.to('cpu'),
                           bins=128,
                           min=min_value.to('cpu'),
                           max=max_value.to('cpu'))
    >>> hist
    tensor([1., 0., 0., 2., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 2., 1., 0., 0., 0., 0., 2., 1., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            1., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1.,
            0., 0., 1., 0., 0., 2., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.,
            0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 2., 0., 0.,
            1., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 1.,
            0., 1.])
    >>> cdf = torch.cumsum(hist,dim=0).int().npu()
    >>> cdf
    tensor([ 1,  1,  1,  3,  3,  3,  3,  4,  5,  5,  6,  6,  7,  7,  7,  7,  7,  7,
            7,  8,  8,  8, 10, 11, 11, 11, 11, 11, 13, 14, 14, 14, 14, 14, 14, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16,
            17, 17, 17, 17, 18, 19, 19, 20, 21, 21, 22, 22, 23, 23, 23, 24, 24, 25,
            25, 25, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 30, 30, 30, 30, 30, 30,
            30, 30, 31, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 34, 35, 37, 37, 37,
            38, 39, 40, 40, 41, 41, 41, 42, 42, 43, 44, 44, 44, 44, 45, 45, 46, 47,
            47, 48], device='npu:0', dtype=torch.int32)
    >>> scale, offset = torch_npu.npu_ifmr(input,
                                       min_value,
                                       max_value,
                                       cdf,
                                       min_percentile=0.999999,
                                       max_percentile=0.999999,
                                       search_start=0.7,
                                       search_end=1.3,
                                       search_step=0.01,
                                       with_offset=False)
    >>> scale
    tensor(0.0080, device='npu:0')
    >>> offset
    tensor(0., device='npu:0')
  ```

>torch_npu.npu_max.dim(self, dim, keepdim=False) -> (Tensor, Tensor)

count max result with dim.

- Parameters:
  Similar to `torch.max`, optimize implemention to npu device.
  
  - **self** (Tensor) – the input tensor.
  - **dim** (Number) – the dimension to reduce.
  - **keepdim** (bool) – whether the output tensor has dim retained or not.
  
- Returns:

  - **values** - max values in the input tensor.
  - **indices** - index of max values in the input tensor.

- Constraints:

  None

- Examples:

  ```python
  >>> input = torch.randn(2, 2, 2, 2, dtype = torch.float32).npu()
  >>> input
  tensor([[[[-1.8135,  0.2078],
            [-0.6678,  0.7846]],
  
          [[ 0.6458, -0.0923],
            [-0.2124, -1.9112]]],
  
          [[[-0.5800, -0.4979],
            [ 0.2580,  1.1335]],
    
          [[ 0.6669,  0.1876],
            [ 0.1160, -0.1061]]]], device='npu:0')
  >>> outputs, indices = torch_npu.npu_max(input, 2)
  >>> outputs
  tensor([[[-0.6678,  0.7846],
          [ 0.6458, -0.0923]],
  
          [[ 0.2580,  1.1335],
          [ 0.6669,  0.1876]]], device='npu:0')
  >>> indices
  tensor([[[1, 1],
          [0, 0]],
  
          [[1, 1],
          [0, 0]]], device='npu:0', dtype=torch.int32)
  ```

>torch_npu.npu_min.dim(self, dim, keepdim=False) -> (Tensor, Tensor)

count min result with dim.

- Parameters:
  Similar to `torch.min`, optimize implemention to npu device.
  - **self** (Tensor) – the input tensor.
  - **dim** (Number) – the dimension to reduce.
  - **keepdim** (bool) – whether the output tensor has dim retained or not.

- Returns:

  - **values** - min values in the input tensor.
  - **indices** - index of min values in the input tensor.

- Constraints:

  None

- Examples:

  ```python
  >>> input = torch.randn(2, 2, 2, 2, dtype = torch.float32).npu()
  >>> input
  tensor([[[[-0.9909, -0.2369],
            [-0.9569, -0.6223]],
  
          [[ 0.1157, -0.3147],
            [-0.7761,  0.1344]]],
  
          [[[ 1.6292,  0.5953],
            [ 0.6940, -0.6367]],
    
          [[-1.2335,  0.2131],
            [ 1.0748, -0.7046]]]], device='npu:0')
  >>> outputs, indices = torch_npu.npu_min(input, 2)
  >>> outputs
  tensor([[[-0.9909, -0.6223],
          [-0.7761, -0.3147]],
  
          [[ 0.6940, -0.6367],
          [-1.2335, -0.7046]]], device='npu:0')
  >>> indices
  tensor([[[0, 1],
          [1, 0]],
  
          [[1, 1],
          [0, 1]]], device='npu:0', dtype=torch.int32)
  ```

>torch_npu.npu_scatter(self, indices, updates, dim) -> Tensor

count scatter result with dim.

- Parameters:
  Similar to `torch.scatter`, optimize implemention to npu device.
  
  - **self** (Tensor) - the input tensor.
  - **indices** (Tensor) – the indices of elements to scatter, can be either empty or of the same dimensionality as src. When empty, the operation returns self unchanged.
  - **updates** (Tensor) – the source element(s) to scatter.
- **dim** (Number) – the axis along which to index
  
- Constraints:

  None

- Examples:

  ```python
  >>> input    = torch.tensor([[1.6279, 0.1226], [0.9041, 1.0980]]).npu()
  >>> input
  tensor([[1.6279, 0.1226],
          [0.9041, 1.0980]], device='npu:0')
  >>> indices  = torch.tensor([0, 1],dtype=torch.int32).npu()
  >>> indices
  tensor([0, 1], device='npu:0', dtype=torch.int32)
  >>> updates  = torch.tensor([-1.1993, -1.5247]).npu()
  >>> updates
  tensor([-1.1993, -1.5247], device='npu:0')
  >>> dim = 0
  >>> output = torch_npu.npu_scatter(input, indices, updates, dim)
  >>> output
  tensor([[-1.1993,  0.1226],
          [ 0.9041, -1.5247]], device='npu:0')
  ```

>torch_npu.npu_layer_norm_eval(input, normalized_shape, weight=None, bias=None, eps=1e-05) -> Tensor

count layer norm result.

- Parameters:
  The same as `torch.nn.functional.layer_norm`, optimize implemention to npu device.
  - **input** (Tensor) - The input Tensor.
  - **normalized_shape** (ListInt) – input shape from an expected input of size.
  - **weight** (Tensor) - The gamma Tensor.
  - **bias** (Tensor) - The beta Tensor.
  - **eps** (Float) – The epsilon value added to the denominator for numerical stability. Default: 1e-5.

- Constraints:

  None

- Examples:

  ```python
  >>> input = torch.rand((6, 4), dtype=torch.float32).npu()
  >>> input
  tensor([[0.1863, 0.3755, 0.1115, 0.7308],
          [0.6004, 0.6832, 0.8951, 0.2087],
          [0.8548, 0.0176, 0.8498, 0.3703],
          [0.5609, 0.0114, 0.5021, 0.1242],
          [0.3966, 0.3022, 0.2323, 0.3914],
          [0.1554, 0.0149, 0.1718, 0.4972]], device='npu:0')
  >>> normalized_shape = input.size()[1:]
  >>> normalized_shape
  torch.Size([4])
  >>> weight = torch.Tensor(*normalized_shape).npu()
  >>> weight
  tensor([        nan,  6.1223e-41, -8.3159e-20,  9.1834e-41], device='npu:0')
  >>> bias = torch.Tensor(*normalized_shape).npu()
  >>> bias
  tensor([5.6033e-39, 6.1224e-41, 6.1757e-39, 6.1224e-41], device='npu:0')
  >>> output = torch_npu.npu_layer_norm_eval(input, normalized_shape, weight, bias, 1e-5)
  >>> output
  tensor([[        nan,  6.7474e-41,  8.3182e-20,  2.0687e-40],
          [        nan,  8.2494e-41, -9.9784e-20, -8.2186e-41],
          [        nan, -2.6695e-41, -7.7173e-20,  2.1353e-41],
          [        nan, -1.3497e-41, -7.1281e-20, -6.9827e-42],
          [        nan,  3.5663e-41,  1.2002e-19,  1.4314e-40],
          [        nan, -6.2792e-42,  1.7902e-20,  2.1050e-40]], device='npu:0')
  ```

>torch_npu.npu_alloc_float_status(self) -> Tensor

Produces eight numbers with a value of zero

- Parameters:
  
  - **self** (Tensor) - Any Tensor
  
- Constraints:

  None

- Examples:

  ```python
  >>> input    = torch.randn([1,2,3]).npu()
  >>> output = torch_npu.npu_alloc_float_status(input)
  >>> input
  tensor([[[ 2.2324,  0.2478, -0.1056],
          [ 1.1273, -0.2573,  1.0558]]], device='npu:0')
  >>> output
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
  ```

> torch_npu.npu_get_float_status(self) -> Tensor

Computes NPU get float status operator function.

- Parameters:
  
  - **self** (Tensor) -  A Tensor of data memory address. Must be float32 .
  
- Constraints:

  None

- Examples:
  
  ```python
  >>> x = torch.rand(2).npu()
  >>> torch_npu.npu_get_float_status(x)
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
  ```

> torch_npu.npu_clear_float_status(self) -> Tensor

Set the value of address 0x40000 to 0 in each core.

- Parameters:
  
  - **self** (Tensor) -  A tensor of type float32.
  
- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.rand(2).npu()
  >>> torch_npu.npu_clear_float_status(x)
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
  ```

> torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first) -> Tensor

Confuse reshape and transpose.

- Parameters:
  
  - **self** (Tensor) -  A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
  - **perm** (ListInt) -  A permutation of the dimensions of "x".
  - **shape** (ListInt) -  The shape of the input.
  - **transpose_first** (bool) -  If True, the transpose is first, otherwise the reshape is first.
  
- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.rand(2, 3, 4, 6).npu()
  >>> x.shape
  torch.Size([2, 3, 4, 6])
  >>> y = torch_npu.npu_confusion_transpose(x, (0, 2, 1, 3), (2, 4, 18), True)
  >>> y.shape
  torch.Size([2, 4, 18])
  >>> y2 = torch_npu.npu_confusion_transpose(x, (0, 2, 1), (2, 12, 6), False)
  >>> y2.shape
  torch.Size([2, 6, 12])
  ```

> torch_npu.npu_bmmV2(self, mat2, output_sizes) -> Tensor

Multiplies matrix "a" by matrix "b", producing "a * b" . 

- Parameters:
  - **self** (Tensor) -  A matrix Tensor. Must be one of the following types: float16, float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ]. 
  - **mat2** (Tensor) -  A matrix Tensor. Must be one of the following types: float16, float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ]. 
  - **output_sizes** (ListInt) - Output's shape, used in matmul's backpropagation, default [].
  
- Constraints:

  None

- Examples:

  ```python
  >>> mat1 = torch.randn(10, 3, 4).npu()
  >>> mat2 = torch.randn(10, 4, 5).npu()
  >>> res = torch_npu.npu_bmmV2(mat1, mat2, [])
  >>> res.shape
  torch.Size([10, 3, 5])
  ```

> torch_npu.fast_gelu(self) -> Tensor

Computes the gradient for the fast_gelu of "x" . 

- Parameters:
  
  - **self** (Tensor) -  A Tensor. Must be one of the following types: float16, float32
  
- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.rand(2).npu()
  >>> x
  tensor([0.5991, 0.4094], device='npu:0')
  >>> torch_npu.fast_gelu(x)
  tensor([0.4403, 0.2733], device='npu:0')
  ```

> torch_npu.npu_sub_sample(self, per_images, positive_fraction) -> Tensor

Randomly sample a subset of positive and negative examples,and overwrite the label vector to the ignore value (-1) for all elements that are not included in the sample.

- Parameters:

  - **self** (Tensor) -  shape of labels,(N, ) label vector with values.
  - **per_images** (Number) -  A require attribute of type int.
  - **positive_fraction** (Float) -  A require attribute of type float.

- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.tensor([-2, 3, 6, -7, -2, 8, 1, -5, 7, 4]).int().npu()
  >>> x
  tensor([-2,  3,  6, -7, -2,  8,  1, -5,  7,  4], device='npu:0',
        dtype=torch.int32)
  >>> torch_npu.npu_sub_sample(x, 5, 0.6)
  tensor([-1, -1, -1, -1, -1, -1,  1, -1, -1, -1], device='npu:0',
        dtype=torch.int32)
  ```

> torch_npu.npu_deformable_conv2d(self, weight, offset, bias, kernel_size, stride, padding, dilation=[1,1,1,1], groups=1, deformable_groups=1, modulated=True) -> (Tensor, Tensor)

Computes the deformed convolution output with the expected input. 

- Parameters:

  - **self** (Tensor) -  A 4D tensor of input image. With the format "NHWC", the data is stored in the order of: [batch, in_height, in_width, in_channels]. 
  - **weight** (Tensor) -  A 4D tensor of learnable filters. Must have the same type as "x". With the format "HWCN" , the data is stored in the order of: [filter_height, filter_width, in_channels / groups, out_channels]. 
  - **offset** (Tensor) -  A 4D tensor of x-y coordinates offset and mask. With the format "NHWC", the data is stored in the order of: [batch, out_height, out_width, deformable_groups * filter_height * filter_width * 3]. 
  - **bias** (Tensor) -  An optional 1D tensor of additive biases to the filter outputs. The data is stored in the order of: [out_channels]. 
  - **kernel_size** (ListInt) -  A tuple/list of 2 integers.kernel size. 
  - **stride** (ListInt) -  Required. A list of 4 integers. The stride of the sliding window for each dimension of input. The dimension order is interpreted according to the data format of "x". The N and C dimensions must be set to 1. 
  - **padding** (ListInt) -  Required. A list of 4 integers. The number of pixels to add to each (top, bottom, left, right) side of the input. 
  - **dilations** (ListInt) -  Optional. A list of 4 integers. The dilation factor for each dimension of input. The dimension order is interpreted according to the data format of "x". The N and C dimensions must be set to 1. Defaults to [1, 1, 1, 1]. 
  - **groups** (Number) -  Optional. An integer of type int32. The number of blocked connections from input channels to output channels. In_channels and out_channels must both be divisible by "groups". Defaults to 1. 
  - **deformable_groups** (Number) -  Optional. An integer of type int32. The number of deformable group partitions. In_channels must be divisible by "deformable_groups". Defaults to 1. 
  - **modulated** (bool) -  Optional. Specify version of DeformableConv2D, true means v2, false means v1, currently only support v2. 

- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.rand(16, 32, 32, 32).npu()
  >>> weight = torch.rand(32, 32, 5, 5).npu()
  >>> offset = torch.rand(16, 75, 32, 32).npu()
  >>> output, _ = torch_npu.npu_deformable_conv2d(x, weight, offset, None, kernel_size=[5, 5], stride = [1, 1, 1, 1], padding = [2, 2, 2, 2])
  >>> output.shape
  torch.Size([16, 32, 32, 32])
  ```
  
> torch_npu.npu_mish(self) -> Tensor

Computes hyperbolic tangent of "x" element-wise.

- Parameters:

  - **self** (Tensor) -  A Tensor. Must be one of the following types: float16, float32.
  
- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.rand(10, 30, 10).npu()
  >>> y = torch_npu.npu_mish(x)
  >>> y.shape
  torch.Size([10, 30, 10])
  ```
  
> torch_npu.npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors) -> Tensor

Generate the responsible flags of anchor in a single feature map. 

- Parameters:
  - **self** (Tensor) -  Ground truth box, 2-D Tensor with shape [batch, 4].
  - **featmap_size** (ListInt) -  The size of feature maps, listint. 
  - **strides** (ListInt) -  Stride of current level, listint. 
  - **num_base_anchors** (Number) -  The number of base anchors. 

- Constraints:

  None

- Examples:

  ```python
  >>> x = torch.rand(100, 4).npu()
  >>> y = torch_npu.npu_anchor_response_flags(x, [60, 60], [2, 2], 9)
  >>> y.shape
  torch.Size([32400])
  ```
  
> torch_npu.npu_yolo_boxes_encode(self, gt_bboxes, stride, performance_mode=False) -> Tensor

Generates bounding boxes based on yolo's "anchor" and "ground-truth" boxes. It is a customized mmdetection operator. 

- Parameters:
  - **self** (Tensor) -  anchor boxes generated by the yolo training set. A 2D Tensor of type float32 or float16 with shape (N, 4). "N" indicates the number of ROIs, "N" indicates the number of ROIs, and the value "4" refers to (tx, ty, tw, th). 
  - **gt_bboxes** (Tensor) -  target of the transformation, e.g, ground-truth boxes. A 2D Tensor of type float32 or float16 with shape (N, 4). "N" indicates the number of ROIs, and 4 indicates "dx", "dy", "dw", and "dh". 
  - **strides** (Tensor) -  Scale for each box. A 1D Tensor of type int32 shape (N,). "N" indicates the number of ROIs. 
- **performance_mode** (bool) - Select performance mode, "high_precision" or "high_performance". select "high_precision" when input type is float32, the output tensor precision will be smaller than 0.0001, select "high_performance" when input type is float32, the ops will be best performance, but precision will be only smaller than 0.005. 
  
- Constraints:

  input anchor boxes only support maximum N=20480. 

- Examples:

  ```python
  >>> anchor_boxes = torch.rand(2, 4).npu()
  >>> gt_bboxes = torch.rand(2, 4).npu()
  >>> stride = torch.tensor([2, 2], dtype=torch.int32).npu()
  >>> output = torch_npu.npu_yolo_boxes_encode(anchor_boxes, gt_bboxes, stride, False)
  >>> output.shape
  torch.Size([2, 4])
  ```
  
> torch_npu.npu_grid_assign_positive(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all) -> Tensor

Performs Position Sensitive PS ROI Pooling Grad. 

- Parameters:
  - **self** (Tensor) -  Tensor of type float16 or float32, shape (n, ) 
  - **overlaps** (Tensor) -   A Tensor. Datatype is same as assigned_gt_inds. IOU between gt_bboxes and bboxes. shape(k, n) 
  - **box_responsible_flags** (Tensor) -  A Tensor. Support uint8. Flag to indicate whether box is responsible. 
  - **max_overlaps** (Tensor) -  A Tensor. Datatype is same as assigned_gt_inds. overlaps.max(axis=0). 
  - **argmax_overlaps** (Tensor) -  A Tensor. Support int32. overlaps.argmax(axis=0). 
  - **gt_max_overlaps** (Tensor) -  A Tensor. Datatype is same as assigned_gt_inds. overlaps.max(axis=1). 
  - **gt_argmax_overlaps** (Tensor) -  A Tensor. Support int32. overlaps.argmax(axis=1). 
  - **num_gts** (Tensor) -  A Tensor. Support int32. real k. shape (1, ) 
  - **pos_iou_thr** (Float) -  loat. IOU threshold for positive bboxes. 
  - **min_pos_iou** (Float) -  float. minimum iou for a bbox to be considered as a positive bbox 
  - **gt_max_assign_all** (bool) -  bool. whether to assign all bboxes with the same highest overlap with some gt to that gt. 

- Constraints:

  None

- Examples:

  ```python
  >>> assigned_gt_inds = torch.rand(4).npu()
  >>> overlaps = torch.rand(2,4).npu()
  >>> box_responsible_flags = torch.tensor([1, 1, 1, 0], dtype=torch.uint8).npu()
  >>> max_overlap = torch.rand(4).npu()
  >>> argmax_overlap = torch.tensor([1, 0, 1, 0], dtype=torch.int32).npu()
  >>> gt_max_overlaps = torch.rand(2).npu()
  >>> gt_argmax_overlaps = torch.tensor([1, 0],dtype=torch.int32).npu()
  >>> output = torch_npu.npu_grid_assign_positive(assigned_gt_inds, overlaps, box_responsible_flags, max_overlap, argmax_overlap, gt_max_overlaps, gt_argmax_overlaps, 128, 0.5, 0., True)
  >>> output.shape
  torch.Size([4])
  ```

> torch_npu.npu_normalize_batch(self, seq_len, normalize_type=0) -> Tensor

 Performs batch normalization . 

- Parameters:

  - **self** (Tensor) - A Tensor. Support float32. shape (n, c, d). 
  - **seq_len** (Tensor) - A Tensor. Each batch normalize data num. Support Int32. Shape (n, ). 
  - **normalize_type** (Number) - Str. Support "per_feature" or "all_features". 

- Constraints:

  None

- Examples:
  ```python
  >>> a=np.random.uniform(1,10,(2,3,6)).astype(np.float32)
  >>> b=np.random.uniform(3,6,(2)).astype(np.int32)
  >>> x=torch.from_numpy(a).to("npu")
  >>> seqlen=torch.from_numpy(b).to("npu")
  >>> out = torch_npu.npu_normalize_batch(x, seqlen, 0)
  >>> out
  tensor([[[ 1.1496, -0.6685, -0.4812,  1.7611, -0.5187,  0.7571],
          [ 1.1445, -0.4393, -0.7051,  1.0474, -0.2646, -0.1582],
          [ 0.1477,  0.9179, -1.0656, -6.8692, -6.7437,  2.8621]],
  
          [[-0.6880,  0.1337,  1.3623, -0.8081, -1.2291, -0.9410],
          [ 0.3070,  0.5489, -1.4858,  0.6300,  0.6428,  0.0433],
          [-0.5387,  0.8204, -1.1401,  0.8584, -0.3686,  0.8444]]],
        device='npu:0')
  ```

> torch_npu.npu_masked_fill_range(self, start, end, value, axis=-1) -> Tensor

masked fill tensor along with one axis by range.boxes. It is a customized masked fill range operator .

- Parameters:

  - **self** (Tensor) - input tensor. A ND Tensor of float32/float16/int32/int8 with shapes  1-D (D,), 2-D(N, D), 3-D(N, C, D).
  - **start** (Tensor) - masked fill start pos. A 3D Tensor of int32 with shape (num, N).
  - **end** (Tensor) - masked fill end pos. A 3D Tensor of int32 with shape (num, N).
  - **value** (Tensor) - masked fill value. A 2D Tensor of float32/float16/int32/int8 with shape (num,).
  - **axis** (Number) - axis with masked fill of int32. Defaults to -1. 

- Constraints:

  None

- Examples:
  ```python
  >>> a=torch.rand(4,4).npu()
  >>> a
  tensor([[0.9419, 0.4919, 0.2874, 0.6560],
          [0.6691, 0.6668, 0.0330, 0.1006],
          [0.3888, 0.7011, 0.7141, 0.7878],
          [0.0366, 0.9738, 0.4689, 0.0979]], device='npu:0')
  >>> start = torch.tensor([[0,1,2]], dtype=torch.int32).npu()
  >>> end = torch.tensor([[1,2,3]], dtype=torch.int32).npu()
  >>> value = torch.tensor([1], dtype=torch.float).npu()
  >>> out = torch_npu.npu_masked_fill_range(a, start, end, value, 1)
  >>> out
  tensor([[1.0000, 0.4919, 0.2874, 0.6560],
          [0.6691, 1.0000, 0.0330, 0.1006],
          [0.3888, 0.7011, 1.0000, 0.7878],
          [0.0366, 0.9738, 0.4689, 0.0979]], device='npu:0')
  ```

> torch_npu.npu_linear(input, weight, bias=None) -> Tensor

  Multiplies matrix "a" by matrix "b", producing "a * b" . 

- Parameters:

  -  **input** (Tensor) - A matrix Tensor. 2D. Must be one of the following types: float32, float16, int32, int8. Has format [ND, NHWC, FRACTAL_NZ]. 
  -  **weight** (Tensor) - A matrix Tensor. 2D. Must be one of the following types: float32, float16, int32, int8. Has format [ND, NHWC, FRACTAL_NZ]. 
  -  **bias** (Tensor) - A 1D Tensor. Must be one of the following types: float32, float16, int32. Has format [ND, NHWC]. 

- Constraints:

  None

- Examples:
  ```python
  >>> x=torch.rand(2,16).npu()
  >>> w=torch.rand(4,16).npu()
  >>> b=torch.rand(4).npu()
  >>> output = torch_npu.npu_linear(x, w, b)
  >>> output
  tensor([[3.6335, 4.3713, 2.4440, 2.0081],
          [5.3273, 6.3089, 3.9601, 3.2410]], device='npu:0')
  ```

> torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size=None, adam_mode=0, *, out= (var,m,v) )

   count adam result. 

- Parameters:

  - **var** (Tensor) - A Tensor. Support float16/float32.
  - **m** (Tensor) - A Tensor. Datatype and shape are same as exp_avg.
  - **v** (Tensor) - A Tensor. Datatype and shape are same as exp_avg.
  - **lr** (Number) - Datatype is same as exp_avg.
  - **beta1** (Number) - Datatype is same as exp_avg. 
  - **beta2** (Number) - Datatype is same as exp_avg. 
  - **epsilon** (Number) - Datatype is same as exp_avg. 
  - **grad** (Tensor) - A Tensor. Datatype and shape are same as exp_avg.
  - **max_grad_norm** (Number) - Datatype is same as exp_avg. 
  - **global_grad_norm** (Number) - Datatype is same as exp_avg. 
  - **weight_decay** (Number) - Datatype is same as exp_avg.

- Keyword Arguments :

  - **out** :A Tensor, optional. The output tensor. 

- constraints:

  None

- Examples:
  ```python
  >>> var_in = torch.rand(321538).uniform_(-32., 21.).npu()
  >>> m_in = torch.zeros(321538).npu()
  >>> v_in = torch.zeros(321538).npu()
  >>> grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
  >>> max_grad_norm = -1.
  >>> beta1 = 0.9
  >>> beta2 = 0.99
  >>> weight_decay = 0.
  >>> lr = 0.
  >>> epsilon = 1e-06
  >>> global_grad_norm = 0.
  >>> var_out, m_out, v_out = torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, out=(var_in, m_in, v_in))
  >>> var_out
  tensor([ 14.7733, -30.1218,  -1.3647,  ..., -16.6840,   7.1518,   8.4872],
        device='npu:0')
  ```

> torch_npu.npu_giou(self, gtboxes, trans=False, is_cross=False, mode=0) -> Tensor

First calculate the minimum closure area of the two boxes, IoU, the proportion of the closed area that does not belong to the two boxes in the closure area, and finally subtract this proportion from IoU to get GIoU .

- Parameters:

  - **self** (Tensor) - Bounding boxes, a 2D Tensor of type float16 or float32 with shape (N, 4). "N" indicates the number of bounding boxes, and the value "4" refers to [x1, y1, x2, y2] or [x, y, w, h].
  - **gtboxes** (Tensor) - Ground-truth boxes, a 2D Tensor of type float16 or float32 with shape (M, 4). "M" indicates the number of ground truth boxes, and the value "4" refers to [x1, y1, x2, y2] or [x, y, w, h].
  - **trans** (bool) - An optional bool, true for 'xywh', false for 'xyxy'. 
  - **is_cross** (bool) - An optional bool, control whether the output shape is [M, N] or [1, N]. 
  - **mode:** (Number) - Computation mode, a character string with the value range of [iou, iof] . 

- Constraints:

  None

- Examples:
  ```python
  >>> a=np.random.uniform(0,1,(4,10)).astype(np.float16)
  >>> b=np.random.uniform(0,1,(4,10)).astype(np.float16)
  >>> box1=torch.from_numpy(a).to("npu")
  >>> box2=torch.from_numpy(a).to("npu")
  >>> output = torch_npu.npu_giou(box1, box2, trans=True, is_cross=False, mode=0)
  >>> output
  tensor([[1.],
          [1.],
          [1.],
          [1.],
          [1.],
          [1.],
          [1.],
          [1.],
          [1.],
          [1.]], device='npu:0', dtype=torch.float16)
  ```

> torch_npu.npu_silu(self) -> Tensor

Computes the for the Swish of "x" .

- Parameters:

  - **self** (Tensor) - A Tensor. Must be one of the following types: float16, float32 

- Constraints:

  None

- Examples:
```python
>>> a=torch.rand(2,8).npu()
>>> output = torch_npu.npu_silu(a)
>>> output
tensor([[0.4397, 0.7178, 0.5190, 0.2654, 0.2230, 0.2674, 0.6051, 0.3522],
        [0.4679, 0.1764, 0.6650, 0.3175, 0.0530, 0.4787, 0.5621, 0.4026]],
       device='npu:0')
```

> torch_npu.npu_reshape(self, shape, bool can_refresh=False) -> Tensor

Reshapes a tensor. Only the tensor shape is changed, without changing the data. 

- Parameters:

  - **self** (Tensor) - A Tensor.
  - **shape** (ListInt) - Defines the shape of the output tensor. 
  - **can_refresh** (bool) - Used to specify whether reshape can be refreshed in place.

- Constraints:

   This operator cannot be directly called by the acllopExecute API. 

- Examples:
  ```python
  >>> a=torch.rand(2,8).npu()
  >>> out=torch_npu.npu_reshape(a,(4,4))
  >>> out
  tensor([[0.6657, 0.9857, 0.7614, 0.4368],
          [0.3761, 0.4397, 0.8609, 0.5544],
          [0.7002, 0.3063, 0.9279, 0.5085],
          [0.1009, 0.7133, 0.8118, 0.6193]], device='npu:0')
  ```

> torch_npu.npu_rotated_overlaps(self, query_boxes, trans=False) -> Tensor

Calculate the overlapping area of the rotated box.

- Parameters:

  - **self** (Tensor) - data of grad increment, a 3D Tensor of type float32 with shape (B, 5, N). 
  - **query_boxes** (Tensor) - Bounding boxes, a 3D Tensor of type float32 with shape (B, 5, K).
  - **trans** (bool) - An optional attr, true for 'xyxyt', false for 'xywht'. 

- Constraints:

  None

- Examples:
  ```python
  >>> a=np.random.uniform(0,1,(1,3,5)).astype(np.float16)
  >>> b=np.random.uniform(0,1,(1,2,5)).astype(np.float16)
  >>> box1=torch.from_numpy(a).to("npu")
  >>> box2=torch.from_numpy(a).to("npu")
  >>> output = torch_npu.npu_rotated_overlaps(box1, box2, trans=False)
  >>> output
  tensor([[[0.0000, 0.1562, 0.0000],
          [0.1562, 0.3713, 0.0611],
          [0.0000, 0.0611, 0.0000]]], device='npu:0', dtype=torch.float16)
  ```

> torch_npu.npu_rotated_iou(self, query_boxes, trans=False, mode=0, is_cross=True) -> Tensor

Calculate the IOU of the rotated box.

- Parameters:

  - **self** (Tensor) - data of grad increment, a 3D Tensor of type float32 with shape (B, 5, N). 
  - **query_boxes** (Tensor) - Bounding boxes, a 3D Tensor of type float32 with shape (B, 5, K).
  - **trans** (bool) - An optional attr, true for 'xyxyt', false for 'xywht'. 
  - **is_cross** (bool) -Cross calculation when it is True, and one-to-one calculation when it is False.
  - **mode** (Number) - Computation mode, a character string with the value range of [iou, iof, giou] . 

- Constraints:

  None

- Examples:
  ```python
  >>> a=np.random.uniform(0,1,(2,2,5)).astype(np.float16)
  >>> b=np.random.uniform(0,1,(2,3,5)).astype(np.float16)
  >>> box1=torch.from_numpy(a).to("npu")
  >>> box2=torch.from_numpy(a).to("npu")
  >>> output = torch_npu.npu_rotated_iou(box1, box2, trans=False, mode=0, is_cross=True)
  >>> output
  tensor([[[3.3325e-01, 1.0162e-01],
          [1.0162e-01, 1.0000e+00]],
  
          [[0.0000e+00, 0.0000e+00],
          [0.0000e+00, 5.9605e-08]]], device='npu:0', dtype=torch.float16)
  ```

> torch_npu.npu_rotated_box_encode(anchor_box, gt_bboxes, weight) -> Tensor

Rotate Bounding Box Encoding.

- Parameters:

  - anchor_box (Tensor) -  A 3D Tensor with shape (B, 5, N). the input tensor.Anchor boxes. "B" indicates the number of batch size, "N" indicates the number of bounding boxes, and the value "5" refers to "x0", "x1", "y0", "y1" and "angle" .
  - gt_bboxes (Tensor) - A 3D Tensor of float32 (float16) with shape (B, 5, N).   
  - weight (Tensor) - A float list for "x0", "x1", "y0", "y1" and "angle", defaults to [1.0, 1.0, 1.0, 1.0, 1.0].

- Constraints:

  None

- Examples:

  ```
  >>> anchor_boxes = torch.tensor([[[30.69], [32.6], [45.94], [59.88], [-44.53]]], dtype=torch.float16).to("npu")
      >>> gt_bboxes = torch.tensor([[[30.44], [18.72], [33.22], [45.56], [8.5]]], dtype=torch.float16).to("npu")
      >>> weight = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float16).npu()
      >>> out = torch_npu.npu_rotated_box_encode(anchor_boxes, gt_bboxes, weight)
      >>> out
      tensor([[[-0.4253],
              [-0.5166],
              [-1.7021],
              [-0.0162],
              [ 1.1328]]], device='npu:0', dtype=torch.float16)
  ```

  >   torch_npu.npu_rotated_box_decode(anchor_boxes, deltas, weight) -> Tensor

  Rotate Bounding Box Encoding

  - Parameters:

    - anchor_box (Tensor) -  A 3D Tensor with shape (B, 5, N). the input tensor.Anchor boxes. "B" indicates the number of batch size, "N" indicates the number of bounding boxes, and the value "5" refers to "x0", "x1", "y0", "y1" and "angle" .
    - deltas (Tensor) - A 3D Tensor of float32 (float16) with shape (B, 5, N).   
    - weight (Tensor) - A float list for "x0", "x1", "y0", "y1" and "angle", defaults to [1.0, 1.0, 1.0, 1.0, 1.0].

  - Constraints:

    None

  - Examples:

    ```
     >>> anchor_boxes = torch.tensor([[[4.137],[33.72],[29.4], [54.06], [41.28]]], dtype=torch.float16).to("npu")
        >>> deltas = torch.tensor([[[0.0244], [-1.992], [0.2109], [0.315], [-37.25]]], dtype=torch.float16).to("npu")
        >>> weight = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float16).npu()
        >>> out = torch_npu.npu_rotated_box_decode(anchor_boxes, deltas, weight)
        >>> out
        tensor([[[  1.7861],
                [-10.5781],
                [ 33.0000],
                [ 17.2969],
                [-88.4375]]], device='npu:0', dtype=torch.float16)
    ```

>   torch_npu.npu_ciou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> Tensor

Applies an NPU based CIOU operation.

 A penalty item is added on the basis of DIoU, and CIoU is proposed.

- Notes:

  Util now, ciou backward only support trans==True, is_cross==False, mode==0('iou') current version if you need to back propagation, please ensure your parameter is correct!

- Args:

  - boxes1 (Tensor): Predicted bboxes of format xywh, shape (4, n).
  - boxes2 (Tensor): Corresponding gt bboxes, shape (4, n).
  - trans (Bool): Whether there is an offset
  - is_cross (Bool): Whether there is a cross operation between box1 and box2.
  - mode (int):  Select the calculation mode of diou.
  - atan_sub_flag (Bool): whether to pass the second value of the forward to the reverse.

- Returns:

  torch.Tensor: The result of the mask operation

- Examples:

  ```
      >>> box1 = torch.randn(4, 32).npu()
      >>> box1.requires_grad = True
      >>> box2 = torch.randn(4, 32).npu()
      >>> box2.requires_grad = True
      >>> ciou = torch_npu.contrib.function.npu_ciou(box1, box2) 
      >>> l = ciou.sum()
      >>> l.backward()
  ```

>   torch_npu.npu_diou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=False, int mode=0) -> Tensor

Applies an NPU based DIOU operation.

Taking into account the distance between the targets,the overlap rate of the distance and the range, different targets or boundaries will tend to be stable.

- Notes:

  Util now, diou backward only support trans==True, is_cross==False, mode==0('iou') current version if you need to back propagation, please ensure your parameter is correct!

- Args:

  - boxes1 (Tensor): Predicted bboxes of format xywh, shape (4, n).
  - boxes2 (Tensor): Corresponding gt bboxes, shape (4, n).
  - trans (Bool): Whether there is an offset
  - is_cross (Bool): Whether there is a cross operation between box1 and box2.
  - mode (int):  Select the calculation mode of diou.

- Returns:

  torch.Tensor: The result of the mask operation

- Examples:

  ```
      >>> box1 = torch.randn(4, 32).npu()
      >>> box1.requires_grad = True
      >>> box2 = torch.randn(4, 32).npu()
      >>> box2.requires_grad = True
      >>> ciou = torch_npu.contrib.function.npu_diou(box1, box2) 
      >>> l = diou.sum()
      >>> l.backward()
  ```

  >    torch_npu.npu_sign_bits_pack(Tensor self, int size) -> Tensor

  one-bit Adam pack of float into uint8.

  - Args:

    - x(Tensor) - A floats Tensor in 1D.
    - size(Number) - A required int. First dimension of output tensor when reshaping.

    - Constraints:

      Size needs to be divisible by output of packing floats. If size of x is divisible by 8, size of output is (size of x) / 8;
      otherwise, size of output is (size of x // 8) + 1, -1 float values will be added to fill divisibility, at little endian positions.
      910 and 710 chips support input type float32 and float16, 310 chips only supports input type float16.

  - Examples:

    ```
        >>>a = torch.tensor([5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2],dtype=torch.float32).npu()
        >>>b = torch_npu.sign_bits_pack(a, 2)
        >>>b
        >>>tensor([[159],[15]], device='npu:0')
        >>>(binary form of 159 is ob10011111, corresponds to 4, -2, -1, 0, 2, 3, 4, 5 respectively)
    ```


  >    torch_npu.sign_bits_unpack(x, dtype, size) -> Tensor

  one-bit Adam unpack of uint8 into float.

  - Args:

    - x(Tensor) - A uint8 Tensor in 1D.
    - dtype(Number) - A required int. 1 sets float16 as output, 0 sets float32 as output.
    - size(Number) - A required int. First dimension of output tensor when reshaping.

  - Constraints:

    Size needs to be divisible by output of unpacking uint8s. Size of output is (size of x) * 8;

  - Examples:

    ```
        >>>a = torch.tensor([159, 15], dtype=torch.uint8).npu()
        >>>b = torch_npu.sign_bits_unpack(a, 0, 2)
        >>>b
        >>>tensor([[1., 1., 1., 1., 1., -1., -1., 1.],
        >>>[1., 1., 1., 1., -1., -1., -1., -1.]], device='npu:0')
    (binary form of 159 is ob00001111)
    ```

    

## Affinity Library

The following affinity library applies to PyTorch 1.8.1.

>   **def fuse_add_softmax_dropout**(training, dropout, attn_mask, attn_scores, attn_head_size, p=0.5, dim=-1):

Using NPU custom operator to replace the native writing method to improve performance

- Args:

  - training (bool): Whether it is training mode.
  - dropout (nn.Module): the dropout layer
  - attn_mask (Tensor): the attention mask.
  - attn_scores (Tensor): the raw attention scores
  - attn_head_size (float): the head size
  - p (float): probability of an element to be zeroed
  - dim (int): A dimension along which softmax will be computed.

- Returns:

  torch.Tensor: The result of the mask operation

- Examples:

  ```
      >>> training = True
      >>> dropout = nn.DropoutWithByteMask(0.1)
      >>> npu_input1 = torch.rand(96, 12, 384, 384).half().npu()
      >>> npu_input2 = torch.rand(96, 12, 384, 384).half().npu()
      >>> alpha = 0.125
      >>> axis = -1
      >>> output = torch_npu.contrib.function.fuse_add_softmax_dropout(training, dropout, npu_input1, npu_input2, alpha, p=axis)
  ```

>   **def** **npu_diou**(boxes1,boxes2,trans=True, is_cross=False, mode=0):

Applies an NPU based DIOU operation.

Taking into account the distance between the targets,the overlap rate of the distance and the range, different targets or boundaries will tend to be stable.

- Notes:

  Util now, diou backward only support trans==True, is_cross==False, mode==0('iou') current version if you need to back propagation, please ensure your parameter is correct!

- Args:

  - boxes1 (Tensor): Predicted bboxes of format xywh, shape (4, n).
  - boxes2 (Tensor): Corresponding gt bboxes, shape (4, n).
  - trans (Bool): Whether there is an offset
  - is_cross (Bool): Whether there is a cross operation between box1 and box2.
  -  mode (int):  Select the calculation mode of diou.

- Returns:

  torch.Tensor: The result of the mask operation

- Examples:

  ```
      >>> box1 = torch.randn(4, 32).npu()
      >>> box1.requires_grad = True
      >>> box2 = torch.randn(4, 32).npu()
      >>> box2.requires_grad = True
      >>> ciou = torch_npu.contrib.function.npu_diou(box1, box2) 
      >>> l = diou.sum()
      >>> l.backward()
  ```
  

>   **def** **npu_ciou**(boxes1,boxes2,trans=True, is_cross=False, mode=0):

Applies an NPU based CIOU operation.

 A penalty item is added on the basis of DIoU, and CIoU is proposed.

- Notes:

   Util now, ciou backward only support trans==True, is_cross==False, mode==0('iou') current version if you need to back propagation, please ensure your parameter is correct!

- Args:

  - boxes1 (Tensor): Predicted bboxes of format xywh, shape (4, n).
  - boxes2 (Tensor): Corresponding gt bboxes, shape (4, n).
  - trans (Bool): Whether there is an offset
  - is_cross (Bool): Whether there is a cross operation between box1 and box2.
  -  mode (int):  Select the calculation mode of diou.
  - atan_sub_flag (Bool): whether to pass the second value of the forward to the reverse.

- Returns:

  torch.Tensor: The result of the mask operation

- Examples:

  ```
      >>> box1 = torch.randn(4, 32).npu()
      >>> box1.requires_grad = True
      >>> box2 = torch.randn(4, 32).npu()
      >>> box2.requires_grad = True
      >>> ciou = torch_npu.contrib.function.npu_ciou(box1, box2) 
      >>> l = ciou.sum()
      >>> l.backward()
  ```

>   **class** **NpuFairseqDropout**(torch.nn.Dropout):

FairseqDropout using on npu device

- Args:
  -  p (float): probability of an element to be zeroed.
  - module_name (string): the name of the model

>   **class** **MultiheadAttention**(nn.Module):

Multi-headed attention.

- Args:

  - embed_dim (int): Total dimension of the model.

  - num_heads (int): Number of parallel attention heads. 

  - kdim(int): Total number of features for keys. Default: None

  - vdim(int): Total number of features for values. Default: None

  - dropout (float): Dropout probability 

  - bias (bool):  If specified, adds bias to input / output projection layers. Default: True.

  - add_bias_kv (bool): If specified, adds bias to the key and value sequences at dim=0. Default: False.

  - add_zero_attn (bool): If specified, adds a new batch of zeros to the key and value sequences at dim=1. 
                                  Default: False.

  - self_attention (bool): Calculate your own attention score. Default: False.

  - encoder_decoder_attention (bool): The input is the output of the encoder and the self-attention output of the decoder, where the self-attention of the encoder is used as the key and value, and the self-attention of the decoder is used as the query. Default: False.

  - q_noise(float): amount of Quantization Noise.
  
  - qn_block_size(int): size of the blocks for subsequent quantization with iPQ.
  
    
