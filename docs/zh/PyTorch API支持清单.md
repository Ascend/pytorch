## [Tensors](https://pytorch.org/docs/1.8.1/torch.html)

| 序号 | API名称                       |                支持情况                |
| ---- | ----------------------------- | :------------------------------------: |
| 1    | torch.is_tensor               |                   是                   |
| 2    | torch.is_storage              |                   是                   |
| 3    | torch.is_complex              | 是，支持判断，但当前硬件限制不支持复数 |
| 4    | torch.is_floating_point       |                   是                   |
| 5    | torch.is_nonzero              |                   是                   |
| 6    | torch.set_default_dtype       |                   是                   |
| 7    | torch.get_default_dtype       |                   是                   |
| 8    | torch.set_default_tensor_type |                   是                   |
| 9    | torch.numel                   |                   是                   |
| 10   | torch.set_printoptions        |                   是                   |
| 11   | torch.set_flush_denormal      |                   是                   |
| 12   | torch.tensor                  |                   是                   |
| 13   | torch.sparse_coo_tensor       |                   否                   |
| 14   | torch.as_tensor               |                   是                   |
| 15   | torch.as_strided              |                   是                   |
| 16   | torch.from_numpy              |                   是                   |
| 17   | torch.zeros                   |                   是                   |
| 18   | torch.zeros_like              |                   是                   |
| 19   | torch.ones                    |                   是                   |
| 20   | torch.ones_like               |                   是                   |
| 21   | torch.arange                  |                   是                   |
| 22   | torch.range                   |                   是                   |
| 23   | torch.linspace                |                   是                   |
| 24   | torch.logspace                |                   是                   |
| 25   | torch.eye                     |                   是                   |
| 26   | torch.empty                   |                   是                   |
| 27   | torch.empty_like              |                   是                   |
| 28   | torch.empty_strided           |                   是                   |
| 29   | torch.full                    |                   是                   |
| 30   | torch.full_like               |                   是                   |
| 31   | torch.quantize_per_tensor     |                   是                   |
| 32   | torch.quantize_per_channel    |                   是                   |
| 33   | torch.dequantize              |                   否                   |
| 34   | torch.complex                 |                   是                   |
| 35   | torch.polar                   |                   是                   |
| 36   | torch.heaviside               |                   否                   |
| 37   | torch.cat                     |                   是                   |
| 38   | torch.chunk                   |                   是                   |
| 39   | torch.column_stack            |                   是                   |
| 40   | torch.dstack                  |                   是                   |
| 41   | torch.hstack                  |                   是                   |
| 42   | torch.gather                  |                   是                   |
| 43   | torch.index_select            |                   是                   |
| 44   | torch.masked_select           |                   是                   |
| 45   | torch.movedim                 |                   是                   |
| 46   | torch.moveaxis                |                   是                   |
| 47   | torch.narrow                  |                   是                   |
| 48   | torch.nonzero                 |                   是                   |
| 49   | torch.reshape                 |                   是                   |
| 50   | torch.row_stack               |                   是                   |
| 51   | torch.scatter                 |                   是                   |
| 52   | torch.scatter_add             |                   是                   |
| 53   | torch.split                   |                   是                   |
| 54   | torch.squeeze                 |                   是                   |
| 55   | torch.stack                   |                   是                   |
| 56   | torch.swapaxes                |                   是                   |
| 57   | torch.swapdims                |                   是                   |
| 58   | torch.t                       |                   是                   |
| 59   | torch.take                    |                   是                   |
| 60   | torch.tensor_split            |                   是                   |
| 61   | torch.tile                    |                   是                   |
| 62   | torch.transpose               |                   是                   |
| 63   | torch.unbind                  |                   是                   |
| 64   | torch.unsqueeze               |                   是                   |
| 65   | torch.vstack                  |                   是                   |
| 66   | torch.where                   |                   是                   |

## Generators

| 序号 | API名称                          | 是否支持 |
| ---- | -------------------------------- | -------- |
| 1    | torch._C.Generator               | 是       |
| 2    | torch._C.torch.default_generator | 是       |

## Random sampling

| 序号 | API名称                       | 是否支持 |
| ---- | ----------------------------- | -------- |
| 1    | torch.seed                    | 是       |
| 2    | torch.manual_seed             | 是       |
| 3    | torch.initial_seed            | 是       |
| 4    | torch.get_rng_state           | 是       |
| 5    | torch.set_rng_state           | 是       |
| 7    | torch.bernoulli               | 是       |
| 8    | torch.multinomial             | 是       |
| 9    | torch.normal                  | 是       |
| 10   | torch.poisson                 | 否       |
| 11   | torch.rand                    | 是       |
| 12   | torch.rand_like               | 是       |
| 13   | torch.randint                 | 是       |
| 14   | torch.randint_like            | 是       |
| 15   | torch.randn                   | 是       |
| 16   | torch.randn_like              | 是       |
| 17   | torch.randperm                | 是       |
| 18   | torch.Tensor.bernoulli        | 是       |
| 19   | torch.Tensor.bernoulli_       | 是       |
| 20   | torch.Tensor.cauchy_          | 是       |
| 21   | torch.Tensor.exponential_     | 否       |
| 22   | torch.Tensor.geometric_       | 否       |
| 23   | torch.Tensor.log_normal_      | 否       |
| 24   | torch.Tensor.normal_          | 是       |
| 25   | torch.Tensor.random_          | 是       |
| 26   | torch.Tensor.uniform_         | 是       |
| 27   | torch.quasirandom.SobolEngine | 是       |

## Serialization

| 序号 | API名称    | 是否支持 |
| ---- | ---------- | -------- |
| 1    | torch.save | 是       |
| 2    | torch.load | 是       |

## Math operations

| 序号 | API名称                                | 是否支持                                                     |
| ---- | -------------------------------------- | ------------------------------------------------------------ |
| 1    | torch.abs                              | 是                                                           |
| 2    | torch.absolute                         | 是                                                           |
| 3    | torch.acos                             | 是                                                           |
| 4    | torch.arccos                           | 否                                                           |
| 5    | torch.acosh                            | 是                                                           |
| 6    | torch.arccosh                          | 是                                                           |
| 7    | torch.add                              | 是                                                           |
| 8    | torch.addcdiv                          | 是                                                           |
| 9    | torch.addcmul                          | 是                                                           |
| 10   | torch.angle                            | 否                                                           |
| 11   | torch.asin                             | 是                                                           |
| 12   | torch.arcsin                           | 是                                                           |
| 13   | torch.asinh                            | 是                                                           |
| 14   | torch.arcsinh                          | 是                                                           |
| 15   | torch.atan                             | 是                                                           |
| 16   | torch.arctan                           | 是                                                           |
| 17   | torch.atanh                            | 是                                                           |
| 18   | torch.arctanh                          | 是                                                           |
| 19   | torch.atan2                            | 是                                                           |
| 20   | torch.bitwise_not                      | 是                                                           |
| 21   | torch.bitwise_and                      | 是                                                           |
| 22   | torch.bitwise_or                       | 是                                                           |
| 23   | torch.bitwise_xor                      | 是                                                           |
| 24   | torch.ceil                             | 是                                                           |
| 25   | torch.clamp                            | 是                                                           |
| 26   | torch.clip                             | 是                                                           |
| 27   | torch.conj                             | 否                                                           |
| 28   | torch.copysign                         | 是                                                           |
| 29   | torch.cos                              | 是                                                           |
| 30   | torch.cosh                             | 是                                                           |
| 31   | torch.deg2rad                          | 是                                                           |
| 32   | torch.div                              | 是                                                           |
| 33   | torch.divide                           | 是                                                           |
| 34   | torch.digamma                          | 是                                                           |
| 35   | torch.erf                              | 是                                                           |
| 36   | torch.erfc                             | 是                                                           |
| 37   | torch.erfinv                           | 是                                                           |
| 38   | torch.exp                              | 是                                                           |
| 39   | torch.exp2                             | 否                                                           |
| 40   | torch.expm1                            | 是                                                           |
| 41   | torch.fake_quantize_per_channel_affine | 否                                                           |
| 42   | torch.fake_quantize_per_tensor_affine  | 否                                                           |
| 43   | torch.fix                              | 是                                                           |
| 44   | torch.float_power                      | 是                                                           |
| 45   | torch.floor                            | 是                                                           |
| 46   | torch.floor_divide                     | 是                                                           |
| 47   | torch.fmod                             | 是                                                           |
| 48   | torch.frac                             | 是                                                           |
| 49   | torch.imag                             | 否                                                           |
| 50   | torch.ldexp                            | 是                                                           |
| 51   | torch.lerp                             | 是                                                           |
| 52   | torch.lgamma                           | 是                                                           |
| 53   | torch.log                              | 是                                                           |
| 54   | torch.log10                            | 是                                                           |
| 55   | torch.log1p                            | 是                                                           |
| 56   | torch.log2                             | 是                                                           |
| 57   | torch.logaddexp                        | 否                                                           |
| 58   | torch.logaddexp2                       | 否                                                           |
| 59   | torch.logical_and                      | 是                                                           |
| 60   | torch.logical_not                      | 是                                                           |
| 61   | torch.logical_or                       | 是                                                           |
| 62   | torch.logical_xor                      | 否                                                           |
| 63   | torch.logit                            | 是                                                           |
| 64   | torch.hypot                            | 是                                                           |
| 65   | torch.i0                               | 否                                                           |
| 66   | torch.igamma                           | 是                                                           |
| 67   | torch.igammac                          | 是                                                           |
| 68   | torch.mul                              | 是                                                           |
| 69   | torch.multiply                         | 是                                                           |
| 70   | torch.mvlgamma                         | 是                                                           |
| 71   | torch.nan_to_num                       | 否                                                           |
| 72   | torch.neg                              | 是                                                           |
| 73   | torch.negative                         | 是                                                           |
| 74   | torch.nextafter                        | 否                                                           |
| 75   | torch.polygamma                        | 否                                                           |
| 76   | torch.pow                              | 是                                                           |
| 77   | torch.rad2deg                          | 是                                                           |
| 78   | torch.real                             | 是                                                           |
| 79   | torch.reciprocal                       | 是                                                           |
| 80   | torch.remainder                        | 是                                                           |
| 81   | torch.round                            | 是                                                           |
| 82   | torch.rsqrt                            | 是                                                           |
| 83   | torch.sigmoid                          | 是                                                           |
| 84   | torch.sign                             | 是                                                           |
| 85   | torch.sgn                              | 否                                                           |
| 86   | torch.signbit                          | 否                                                           |
| 87   | torch.sin                              | 是                                                           |
| 88   | torch.sinc                             | 否                                                           |
| 89   | torch.sinh                             | 是                                                           |
| 90   | torch.sqrt                             | 是                                                           |
| 91   | torch.square                           | 是                                                           |
| 92   | torch.sub                              | 是                                                           |
| 93   | torch.subtract                         | 是                                                           |
| 94   | torch.tan                              | 是                                                           |
| 95   | torch.tanh                             | 是                                                           |
| 96   | torch.true_divide                      | 是                                                           |
| 97   | torch.trunc                            | 是                                                           |
| 98   | torch.xlogy                            | 是                                                           |
| 99   | torch.argmax                           | 是                                                           |
| 100  | torch.argmin                           | 是                                                           |
| 101  | torch.amax                             | 否                                                           |
| 102  | torch.amin                             | 否                                                           |
| 103  | torch.all                              | 是                                                           |
| 104  | torch.any                              | 是                                                           |
| 105  | torch.max                              | 是                                                           |
| 106  | torch.min                              | 是                                                           |
| 107  | torch.dist                             | 是                                                           |
| 108  | torch.logsumexp                        | 是                                                           |
| 109  | torch.mean                             | 是                                                           |
| 110  | torch.median                           | 是                                                           |
| 111  | torch.nanmedian                        | 是                                                           |
| 112  | torch.mode                             | 是                                                           |
| 113  | torch.norm                             | 是                                                           |
| 114  | torch.nansum                           | 是                                                           |
| 115  | torch.prod                             | 是                                                           |
| 116  | torch.quantile                         | 否                                                           |
| 117  | torch.nanquantile                      | 否                                                           |
| 118  | torch.std                              | 是                                                           |
| 119  | torch.std_mean                         | 是                                                           |
| 120  | torch.sum                              | 是                                                           |
| 121  | torch.unique                           | 是                                                           |
| 122  | torch.unique_consecutive               | 是，传参时必须使用关键字，否则精度不达标。return_inverse=return_inverse,return_counts=return_counts,dim=dim |
| 123  | torch.var                              | 否                                                           |
| 124  | torch.var_mean                         | 否                                                           |
| 125  | torch.count_nonzero                    | 否                                                           |
| 126  | torch.allclose                         | 是                                                           |
| 127  | torch.argsort                          | 是                                                           |
| 128  | torch.eq                               | 是                                                           |
| 129  | torch.equal                            | 是                                                           |
| 130  | torch.ge                               | 是                                                           |
| 131  | torch.greater_equal                    | 是                                                           |
| 132  | torch.gt                               | 是                                                           |
| 133  | torch.greater                          | 是                                                           |
| 134  | torch.isclose                          | 否                                                           |
| 135  | torch.isfinite                         | 是                                                           |
| 136  | torch.isinf                            | 是                                                           |
| 137  | torch.isposinf                         | 是                                                           |
| 138  | torch.isneginf                         | 是                                                           |
| 139  | torch.isnan                            | 是                                                           |
| 140  | torch.isreal                           | 是                                                           |
| 141  | torch.kthvalue                         | 是                                                           |
| 142  | torch.le                               | 是                                                           |
| 143  | torch.less_equal                       | 是                                                           |
| 144  | torch.lt                               | 是                                                           |
| 145  | torch.less                             | 是                                                           |
| 146  | torch.maximum                          | 是                                                           |
| 147  | torch.minimum                          | 是                                                           |
| 148  | torch.fmax                             | 否                                                           |
| 149  | torch.fmin                             | 否                                                           |
| 150  | torch.ne                               | 是                                                           |
| 151  | torch.not_equal                        | 是                                                           |
| 152  | torch.sort                             | 是                                                           |
| 153  | torch.topk                             | 是                                                           |
| 154  | torch.msort                            | 否                                                           |
| 155  | torch.fft                              | 否                                                           |
| 156  | torch.ifft                             | 否                                                           |
| 157  | torch.rfft                             | 否                                                           |
| 158  | torch.irfft                            | 否                                                           |
| 159  | torch.stft                             | 否                                                           |
| 160  | torch.istft                            | 否                                                           |
| 161  | torch.bartlett_window                  | 是                                                           |
| 162  | torch.blackman_window                  | 是                                                           |
| 163  | torch.hamming_window                   | 是                                                           |
| 164  | torch.hann_window                      | 是                                                           |
| 165  | torch.kasier_window                    | 否                                                           |
| 166  | torch.atleast_1d                       | 是                                                           |
| 167  | torch.atleast_2d                       | 是                                                           |
| 168  | torch.atleast_3d                       | 是                                                           |
| 169  | torch.bincount                         | 是                                                           |
| 170  | torch.block_diag                       | 是                                                           |
| 171  | torch.broadcast_tensors                | 是                                                           |
| 172  | torch.broadcast_to                     | 是                                                           |
| 173  | torch.broadcast_shapes                 | 是                                                           |
| 174  | torch.bucketize                        | 是                                                           |
| 175  | torch.cartesian_prod                   | 是                                                           |
| 176  | torch.cdist                            | 是，仅支持mode=donot_use_mm_for_euclid_dist                  |
| 177  | torch.clone                            | 是                                                           |
| 178  | torch.combinations                     | 是                                                           |
| 179  | torch.cross                            | 是                                                           |
| 180  | torch.cummax                           | 是                                                           |
| 181  | torch.cummin                           | 是                                                           |
| 182  | torch.cumprod                          | 是                                                           |
| 183  | torch.cumsum                           | 是                                                           |
| 184  | torch.diag                             | 是，仅支持diagonal=0场景                                     |
| 185  | torch.diag_embed                       | 是                                                           |
| 186  | torch.diagflat                         | 是                                                           |
| 187  | torch.diagonal                         | 是                                                           |
| 188  | torch.diff                             | 是                                                           |
| 189  | torch.einsum                           | 是                                                           |
| 190  | torch.flatten                          | 是                                                           |
| 191  | torch.flip                             | 是                                                           |
| 192  | torch.fliplr                           | 是                                                           |
| 193  | torch.flipud                           | 是                                                           |
| 194  | torch.kron                             | 是                                                           |
| 195  | torch.rot90                            | 是                                                           |
| 196  | torch.gcd                              | 否                                                           |
| 197  | torch.histc                            | 否                                                           |
| 198  | torch.meshgrid                         | 是                                                           |
| 199  | torch.lcm                              | 否                                                           |
| 200  | torhc.logcumsumexp                     | 否                                                           |
| 201  | torch.ravel                            | 是                                                           |
| 202  | torch.renorm                           | 是                                                           |
| 203  | torch.repeat_interleave                | 是                                                           |
| 204  | torch.roll                             | 是                                                           |
| 205  | torch.searchsorted                     | 是                                                           |
| 206  | torch.tensordot                        | 是                                                           |
| 207  | torch.trace                            | 否                                                           |
| 208  | torch.tril                             | 是                                                           |
| 209  | torch.tril_indices                     | 是                                                           |
| 210  | torch.triu                             | 是                                                           |
| 211  | torch.triu_indices                     | 是                                                           |
| 212  | torch.vander                           | 是                                                           |
| 213  | torch.view_as_real                     | 是                                                           |
| 214  | torch.view_as_complex                  | 是                                                           |
| 215  | torch.addbmm                           | 是                                                           |
| 216  | torch.addmm                            | 是                                                           |
| 217  | torch.addmv                            | 是                                                           |
| 218  | torch.addr                             | 是                                                           |
| 219  | torch.baddbmm                          | 是                                                           |
| 220  | torch.bmm                              | 是                                                           |
| 221  | torch.chain_matmul                     | 是                                                           |
| 222  | torch.cholesky                         | 否                                                           |
| 223  | torch.cholesky_inverse                 | 否                                                           |
| 224  | torch.cholesky_solve                   | 否                                                           |
| 225  | torch.dot                              | 是                                                           |
| 226  | torch.eig                              | 否                                                           |
| 227  | torch.geqrf                            | 否                                                           |
| 228  | torch.ger                              | 是                                                           |
| 229  | torch.inner                            | 是                                                           |
| 230  | torch.inverse                          | 是                                                           |
| 231  | torch.det                              | 否                                                           |
| 232  | torch.logdet                           | 否                                                           |
| 233  | torch.slogdet                          | 是                                                           |
| 234  | torch.lstsq                            | 否                                                           |
| 235  | torch.lu                               | 否                                                           |
| 236  | torch.lu_solve                         | 否                                                           |
| 237  | torch.lu_unpack                        | 否                                                           |
| 238  | torch.matmul                           | 是                                                           |
| 239  | torch.matrix_power                     | 是                                                           |
| 240  | torch.matrix_rank                      | 是                                                           |
| 241  | torch.matrix_exp                       | 是                                                           |
| 242  | torch.mm                               | 是                                                           |
| 243  | torch.mv                               | 是                                                           |
| 244  | torch.orgqr                            | 否                                                           |
| 245  | torch.ormqr                            | 否                                                           |
| 246  | torch.outer                            | 是                                                           |
| 247  | torch.pinverse                         | 是                                                           |
| 248  | torch.qr                               | 是                                                           |
| 249  | torch.solve                            | 否                                                           |
| 250  | torch.svd                              | 是                                                           |
| 251  | torch.svd_lowrank                      | 是                                                           |
| 252  | torch.pca_lowrank                      | 是                                                           |
| 253  | torch.symeig                           | 是                                                           |
| 254  | torch.lobpcg                           | 否                                                           |
| 255  | torch.trapz                            | 是                                                           |
| 256  | torch.triangular_solve                 | 是                                                           |
| 257  | torch.vdot                             | 是                                                           |

## Utilities

| 序号 | API名称                                    | 是否支持 |
| ---- | ------------------------------------------ | -------- |
| 1    | torch.compiled_with_cxx11_abi              | 是       |
| 2    | torch.result_type                          | 是       |
| 3    | torch.can_cast                             | 是       |
| 4    | torch.promote_types                        | 是       |
| 6    | torch.use_deterministic_algorithms         | 是       |
| 7    | torch.are_deterministic_algorithms_enabled | 是       |
| 8    | torch._assert                              | 是       |

## Other

| 序号 | API名称                       | 是否支持 |
| ---- | ----------------------------- | -------- |
| 1    | torch.no_grad                 | 是       |
| 2    | torch.enable_grad             | 是       |
| 3    | torch.set_grad_enabled        | 是       |
| 4    | torch.get_num_threads         | 是       |
| 5    | torch.set_num_threads         | 是       |
| 6    | torch.get_num_interop_threads | 是       |
| 7    | torch.set_num_interop_threads | 是       |

## torch.Tensor

| 序号 | API名称                                | 是否支持 |
| :--- | -------------------------------------- | -------- |
| 1    | torch.Tensor                           | 是       |
| 2    | torch.Tensor.new_tensor                | 是       |
| 3    | torch.Tensor.new_full                  | 是       |
| 4    | torch.Tensor.new_empty                 | 是       |
| 5    | torch.Tensor.new_ones                  | 是       |
| 6    | torch.Tensor.new_zeros                 | 是       |
| 7    | torch.Tensor.is_cuda                   | 是       |
| 8    | torch.Tensor.is_quantized              | 是       |
| 9    | torch.Tensor.device                    | 是       |
| 10   | torch.Tensor.ndim                      | 是       |
| 11   | torch.Tensor.T                         | 是       |
| 12   | torch.Tensor.abs                       | 是       |
| 13   | torch.Tensor.abs_                      | 是       |
| 14   | torch.Tensor.acos                      | 是       |
| 15   | torch.Tensor.acos_                     | 是       |
| 16   | torch.Tensor.add                       | 是       |
| 17   | torch.Tensor.add_                      | 是       |
| 18   | torch.Tensor.addbmm                    | 是       |
| 19   | torch.Tensor.addbmm_                   | 是       |
| 20   | torch.Tensor.addcdiv                   | 是       |
| 21   | torch.Tensor.addcdiv_                  | 是       |
| 22   | torch.Tensor.addcmul                   | 是       |
| 23   | torch.Tensor.addcmul_                  | 是       |
| 24   | torch.Tensor.addmm                     | 是       |
| 25   | torch.Tensor.addmm_                    | 是       |
| 26   | torch.Tensor.addmv                     | 是       |
| 27   | torch.Tensor.addmv_                    | 是       |
| 28   | torch.Tensor.addr                      | 是       |
| 29   | torch.Tensor.addr_                     | 是       |
| 30   | torch.Tensor.allclose                  | 是       |
| 31   | torch.Tensor.angle                     | 否       |
| 32   | torch.Tensor.apply_                    | 否       |
| 33   | torch.Tensor.argmax                    | 是       |
| 34   | torch.Tensor.argmin                    | 是       |
| 35   | torch.Tensor.argsort                   | 是       |
| 36   | torch.Tensor.asin                      | 是       |
| 37   | torch.Tensor.asin_                     | 是       |
| 38   | torch.Tensor.as_strided                | 是       |
| 39   | torch.Tensor.atan                      | 是       |
| 40   | torch.Tensor.atan2                     | 是       |
| 41   | torch.Tensor.atan2_                    | 是       |
| 42   | torch.Tensor.atan_                     | 是       |
| 43   | torch.Tensor.baddbmm                   | 是       |
| 44   | torch.Tensor.baddbmm_                  | 是       |
| 45   | torch.Tensor.bernoulli                 | 是       |
| 46   | torch.Tensor.bernoulli_                | 是       |
| 47   | torch.Tensor.bfloat16                  | 否       |
| 48   | torch.Tensor.bincount                  | 是       |
| 49   | torch.Tensor.bitwise_not               | 是       |
| 50   | torch.Tensor.bitwise_not_              | 是       |
| 51   | torch.Tensor.bitwise_and               | 是       |
| 52   | torch.Tensor.bitwise_and_              | 是       |
| 53   | torch.Tensor.bitwise_or                | 是       |
| 54   | torch.Tensor.bitwise_or_               | 是       |
| 55   | torch.Tensor.bitwise_xor               | 是       |
| 56   | torch.Tensor.bitwise_xor_              | 是       |
| 57   | torch.Tensor.bmm                       | 是       |
| 58   | torch.Tensor.bool                      | 是       |
| 59   | torch.Tensor.byte                      | 是       |
| 60   | torch.Tensor.cauchy_                   | 否       |
| 61   | torch.Tensor.ceil                      | 是       |
| 62   | torch.Tensor.ceil_                     | 是       |
| 63   | torch.Tensor.char                      | 是       |
| 64   | torch.Tensor.chain_matul               | 是       |
| 65   | torch.Tensor.cholesky                  | 否       |
| 66   | torch.Tensor.cholesky_inverse          | 否       |
| 67   | torch.Tensor.cholesky_solve            | 否       |
| 68   | torch.Tensor.chunk                     | 是       |
| 69   | torch.Tensor.clamp                     | 是       |
| 70   | torch.Tensor.clamp_                    | 是       |
| 71   | torch.Tensor.clone                     | 是       |
| 72   | torch.Tensor.contiguous                | 是       |
| 73   | torch.Tensor.copy_                     | 是       |
| 74   | torch.Tensor.conj                      | 否       |
| 75   | torch.Tensor.cos                       | 是       |
| 76   | torch.Tensor.cos_                      | 是       |
| 77   | torch.Tensor.cosh                      | 是       |
| 78   | torch.Tensor.cosh_                     | 是       |
| 79   | torch.Tensor.cpu                       | 是       |
| 80   | torch.Tensor.cross                     | 是       |
| 81   | torch.Tensor.cuda                      | 否       |
| 82   | torch.Tensor.cummax                    | 是       |
| 83   | torch.Tensor.cummin                    | 是       |
| 84   | torch.Tensor.cumprod                   | 是       |
| 85   | torch.Tensor.cumsum                    | 是       |
| 86   | torch.Tensor.data_ptr                  | 是       |
| 87   | torch.Tensor.dequantize                | 否       |
| 88   | torch.Tensor.det                       | 否       |
| 89   | torch.Tensor.dense_dim                 | 否       |
| 90   | torch.Tensor.diag                      | 是       |
| 91   | torch.Tensor.diag_embed                | 是       |
| 92   | torch.Tensor.diagflat                  | 是       |
| 93   | torch.Tensor.diagonal                  | 是       |
| 94   | torch.Tensor.fill_diagonal_            | 是       |
| 95   | torch.Tensor.digamma                   | 否       |
| 96   | torch.Tensor.digamma_                  | 否       |
| 97   | torch.Tensor.dim                       | 是       |
| 98   | torch.Tensor.dist                      | 是       |
| 99   | torch.Tensor.div                       | 是       |
| 100  | torch.Tensor.div_                      | 是       |
| 101  | torch.Tensor.dot                       | 是       |
| 102  | torch.Tensor.double                    | 否       |
| 103  | torch.Tensor.eig                       | 否       |
| 104  | torch.Tensor.element_size              | 是       |
| 105  | torch.Tensor.eq                        | 是       |
| 106  | torch.Tensor.eq_                       | 是       |
| 107  | torch.Tensor.equal                     | 是       |
| 108  | torch.Tensor.erf                       | 是       |
| 109  | torch.Tensor.erf_                      | 是       |
| 110  | torch.Tensor.erfc                      | 是       |
| 111  | torch.Tensor.erfc_                     | 是       |
| 112  | torch.Tensor.erfinv                    | 是       |
| 113  | torch.Tensor.erfinv_                   | 是       |
| 114  | torch.Tensor.exp                       | 是       |
| 115  | torch.Tensor.exp_                      | 是       |
| 116  | torch.Tensor.expm1                     | 是       |
| 117  | torch.Tensor.expm1_                    | 是       |
| 118  | torch.Tensor.expand                    | 是       |
| 119  | torch.Tensor.expand_as                 | 是       |
| 120  | torch.Tensor.exponential_              | 否       |
| 121  | torch.Tensor.fft                       | 否       |
| 122  | torch.Tensor.fill_                     | 是       |
| 123  | torch.Tensor.flatten                   | 是       |
| 124  | torch.Tensor.flip                      | 是       |
| 125  | torch.Tensor.float                     | 是       |
| 126  | torch.Tensor.floor                     | 是       |
| 127  | torch.Tensor.floor_                    | 是       |
| 128  | torch.Tensor.floor_divide              | 是       |
| 129  | torch.Tensor.floor_divide_             | 是       |
| 130  | torch.Tensor.fmod                      | 是       |
| 131  | torch.Tensor.fmod_                     | 是       |
| 132  | torch.Tensor.frac                      | 是       |
| 133  | torch.Tensor.frac_                     | 是       |
| 134  | torch.Tensor.gather                    | 是       |
| 135  | torch.Tensor.ge                        | 是       |
| 136  | torch.Tensor.ge_                       | 是       |
| 137  | torch.Tensor.geometric_                | 否       |
| 138  | torch.Tensor.geqrf                     | 否       |
| 139  | torch.Tensor.ger                       | 是       |
| 140  | torch.Tensor.get_device                | 是       |
| 141  | torch.Tensor.gt                        | 是       |
| 142  | torch.Tensor.gt_                       | 是       |
| 143  | torch.Tensor.half                      | 是       |
| 144  | torch.Tensor.hardshrink                | 是       |
| 145  | torch.Tensor.histc                     | 否       |
| 146  | torch.Tensor.ifft                      | 否       |
| 147  | torch.Tensor.index_add_                | 是       |
| 148  | torch.Tensor.index_add                 | 是       |
| 149  | torch.Tensor.index_copy_               | 是       |
| 150  | torch.Tensor.index_copy                | 是       |
| 151  | torch.Tensor.index_fill_               | 是       |
| 152  | torch.Tensor.index_fill                | 是       |
| 153  | torch.Tensor.index_put_                | 是       |
| 154  | torch.Tensor.index_put                 | 是       |
| 155  | torch.Tensor.index_select              | 是       |
| 156  | torch.Tensor.indices                   | 否       |
| 157  | torch.Tensor.int                       | 是       |
| 158  | torch.Tensor.int_repr                  | 否       |
| 159  | torch.Tensor.inner                     | 是       |
| 160  | torch.Tensor.inverse                   | 是       |
| 161  | torch.Tensor.irfft                     | 否       |
| 162  | torch.Tensor.is_contiguous             | 是       |
| 163  | torch.Tensor.is_complex                | 是       |
| 164  | torch.Tensor.is_floating_point         | 是       |
| 165  | torch.Tensor.is_pinned                 | 是       |
| 166  | torch.Tensor.is_set_to                 | 否       |
| 167  | torch.Tensor.is_shared                 | 是       |
| 168  | torch.Tensor.is_signed                 | 是       |
| 169  | torch.Tensor.is_sparse                 | 是       |
| 170  | torch.Tensor.item                      | 是       |
| 171  | torch.Tensor.kthvalue                  | 是       |
| 172  | torch.Tensor.le                        | 是       |
| 173  | torch.Tensor.le_                       | 是       |
| 174  | torch.Tensor.lerp                      | 是       |
| 175  | torch.Tensor.lerp_                     | 是       |
| 176  | torch.Tensor.lgamma                    | 否       |
| 177  | torch.Tensor.lgamma_                   | 否       |
| 178  | torch.Tensor.lobpcg                    | 是       |
| 179  | torch.Tensor.log                       | 是       |
| 180  | torch.Tensor.log_                      | 是       |
| 181  | torch.Tensor.logdet                    | 否       |
| 182  | torch.Tensor.log10                     | 是       |
| 183  | torch.Tensor.log10_                    | 是       |
| 184  | torch.Tensor.log1p                     | 是       |
| 185  | torch.Tensor.log1p_                    | 是       |
| 186  | torch.Tensor.log2                      | 是       |
| 187  | torch.Tensor.log2_                     | 是       |
| 188  | torch.Tensor.log_normal_               | 否       |
| 189  | torch.Tensor.logsumexp                 | 是       |
| 190  | torch.Tensor.logical_and               | 是       |
| 191  | torch.Tensor.logical_and_              | 是       |
| 192  | torch.Tensor.logical_not               | 是       |
| 193  | torch.Tensor.logical_not_              | 是       |
| 194  | torch.Tensor.logical_or                | 是       |
| 195  | torch.Tensor.logical_or_               | 是       |
| 196  | torch.Tensor.logical_xor               | 否       |
| 197  | torch.Tensor.logical_xor_              | 否       |
| 198  | torch.Tensor.long                      | 是       |
| 199  | torch.Tensor.lstsq                     | 否       |
| 200  | torch.Tensor.lt                        | 是       |
| 201  | torch.Tensor.lt_                       | 是       |
| 202  | torch.Tensor.lu                        | 是       |
| 203  | torch.Tensor.lu_solve                  | 是       |
| 204  | torch.Tensor.lu_unpack                 | 是       |
| 205  | torch.Tensor.map_                      | 否       |
| 206  | torch.Tensor.masked_scatter_           | 是       |
| 207  | torch.Tensor.masked_scatter            | 是       |
| 208  | torch.Tensor.masked_fill_              | 是       |
| 209  | torch.Tensor.masked_fill               | 是       |
| 210  | torch.Tensor.masked_select             | 是       |
| 211  | torch.Tensor.matmul                    | 是       |
| 212  | torch.Tensor.matrix_power              | 是       |
| 213  | torch.Tensor.matrix_rank               | 是       |
| 214  | torch.Tensor.matrix_exp                | 是       |
| 215  | torch.Tensor.max                       | 是       |
| 216  | torch.Tensor.mean                      | 是       |
| 217  | torch.Tensor.median                    | 是       |
| 218  | torch.Tensor.min                       | 是       |
| 219  | torch.Tensor.mm                        | 是       |
| 220  | torch.Tensor.mode                      | 否       |
| 221  | torch.Tensor.mul                       | 是       |
| 222  | torch.Tensor.mul_                      | 是       |
| 223  | torch.Tensor.multinomial               | 是       |
| 224  | torch.Tensor.mv                        | 是       |
| 225  | torch.Tensor.mvlgamma                  | 否       |
| 226  | torch.Tensor.mvlgamma_                 | 否       |
| 227  | torch.Tensor.narrow                    | 是       |
| 228  | torch.Tensor.narrow_copy               | 是       |
| 229  | torch.Tensor.ndimension                | 是       |
| 230  | torch.Tensor.ne                        | 是       |
| 231  | torch.Tensor.ne_                       | 是       |
| 232  | torch.Tensor.neg                       | 是       |
| 233  | torch.Tensor.neg_                      | 是       |
| 234  | torch.Tensor.nelement                  | 是       |
| 235  | torch.Tensor.nonzero                   | 是       |
| 236  | torch.Tensor.norm                      | 是       |
| 237  | torch.Tensor.normal_                   | 是       |
| 238  | torch.Tensor.numel                     | 是       |
| 239  | torch.Tensor.numpy                     | 否       |
| 240  | torch.Tensor.orgqr                     | 否       |
| 241  | torch.Tensor.ormqr                     | 否       |
| 242  | torch.Tensor.outer                     | 是       |
| 243  | torch.Tensor.permute                   | 是       |
| 244  | torch.Tensor.pca_lowrank               | 是       |
| 245  | torch.Tensor.pin_memory                | 否       |
| 246  | torch.Tensor.pinverse                  | 是       |
| 247  | torch.Tensor.polygamma                 | 否       |
| 248  | torch.Tensor.polygamma_                | 否       |
| 249  | torch.Tensor.pow                       | 是       |
| 250  | torch.Tensor.pow_                      | 是       |
| 251  | torch.Tensor.prod                      | 是       |
| 252  | torch.Tensor.put_                      | 是       |
| 253  | torch.Tensor.qr                        | 是       |
| 254  | torch.Tensor.qscheme                   | 否       |
| 255  | torch.Tensor.q_scale                   | 否       |
| 256  | torch.Tensor.q_zero_point              | 否       |
| 257  | torch.Tensor.q_per_channel_scales      | 否       |
| 258  | torch.Tensor.q_per_channel_zero_points | 否       |
| 259  | torch.Tensor.q_per_channel_axis        | 否       |
| 260  | torch.Tensor.random_                   | 是       |
| 261  | torch.Tensor.reciprocal                | 是       |
| 262  | torch.Tensor.reciprocal_               | 是       |
| 263  | torch.Tensor.record_stream             | 是       |
| 264  | torch.Tensor.remainder                 | 是       |
| 265  | torch.Tensor.remainder_                | 是       |
| 266  | torch.Tensor.renorm                    | 是       |
| 267  | torch.Tensor.renorm_                   | 是       |
| 268  | torch.Tensor.repeat                    | 是       |
| 269  | torch.Tensor.repeat_interleave         | 是       |
| 270  | torch.Tensor.requires_grad_            | 是       |
| 271  | torch.Tensor.reshape                   | 是       |
| 272  | torch.Tensor.reshape_as                | 是       |
| 273  | torch.Tensor.resize_                   | 是       |
| 274  | torch.Tensor.resize_as_                | 是       |
| 275  | torch.Tensor.rfft                      | 否       |
| 276  | torch.Tensor.roll                      | 是       |
| 277  | torch.Tensor.rot90                     | 是       |
| 278  | torch.Tensor.round                     | 是       |
| 279  | torch.Tensor.round_                    | 是       |
| 280  | torch.Tensor.rsqrt                     | 是       |
| 281  | torch.Tensor.rsqrt_                    | 是       |
| 282  | torch.Tensor.scatter                   | 是       |
| 283  | torch.Tensor.scatter_                  | 是       |
| 284  | torch.Tensor.scatter_add_              | 是       |
| 285  | torch.Tensor.scatter_add               | 是       |
| 286  | torch.Tensor.select                    | 是       |
| 287  | torch.Tensor.set_                      | 是       |
| 288  | torch.Tensor.share_memory_             | 否       |
| 289  | torch.Tensor.short                     | 是       |
| 290  | torch.Tensor.sigmoid                   | 是       |
| 291  | torch.Tensor.sigmoid_                  | 是       |
| 292  | torch.Tensor.sign                      | 是       |
| 293  | torch.Tensor.sign_                     | 是       |
| 294  | torch.Tensor.sin                       | 是       |
| 295  | torch.Tensor.sin_                      | 是       |
| 296  | torch.Tensor.sinh                      | 是       |
| 297  | torch.Tensor.sinh_                     | 是       |
| 298  | torch.Tensor.size                      | 是       |
| 299  | torch.Tensor.slogdet                   | 是       |
| 300  | torch.Tensor.solve                     | 否       |
| 301  | torch.Tensor.sort                      | 是       |
| 302  | torch.Tensor.split                     | 是       |
| 303  | torch.Tensor.sparse_mask               | 否       |
| 304  | torch.Tensor.sparse_dim                | 否       |
| 305  | torch.Tensor.sqrt                      | 是       |
| 306  | torch.Tensor.sqrt_                     | 是       |
| 307  | torch.Tensor.square                    | 是       |
| 308  | torch.Tensor.square_                   | 是       |
| 309  | torch.Tensor.squeeze                   | 是       |
| 310  | torch.Tensor.squeeze_                  | 是       |
| 311  | torch.Tensor.std                       | 是       |
| 312  | torch.Tensor.stft                      | 否       |
| 313  | torch.Tensor.storage                   | 是       |
| 314  | torch.Tensor.storage_offset            | 是       |
| 315  | torch.Tensor.storage_type              | 是       |
| 316  | torch.Tensor.stride                    | 是       |
| 317  | torch.Tensor.sub                       | 是       |
| 318  | torch.Tensor.sub_                      | 是       |
| 319  | torch.Tensor.sum                       | 是       |
| 320  | torch.Tensor.sum_to_size               | 是       |
| 321  | torch.Tensor.svd                       | 是       |
| 322  | torch.Tensor.svd_lowrank               | 是       |
| 323  | torch.Tensor.symeig                    | 是       |
| 324  | torch.Tensor.t                         | 是       |
| 325  | torch.Tensor.t_                        | 是       |
| 326  | torch.Tensor.to                        | 是       |
| 327  | torch.Tensor.to_mkldnn                 | 否       |
| 328  | torch.Tensor.take                      | 是       |
| 329  | torch.Tensor.tan                       | 是       |
| 330  | torch.Tensor.tan_                      | 是       |
| 331  | torch.Tensor.tanh                      | 是       |
| 332  | torch.Tensor.tanh_                     | 是       |
| 333  | torch.Tensor.tolist                    | 是       |
| 334  | torch.Tensor.topk                      | 是       |
| 335  | torch.Tensor.to_sparse                 | 否       |
| 336  | torch.Tensor.trace                     | 否       |
| 337  | torch.Tensor.trapz                     | 是       |
| 338  | torch.Tensor.transpose                 | 是       |
| 339  | torch.Tensor.transpose_                | 是       |
| 340  | torch.Tensor.triangular_solve          | 是       |
| 341  | torch.Tensor.tril                      | 是       |
| 342  | torch.Tensor.tril_                     | 是       |
| 343  | torch.Tensor.triu                      | 是       |
| 344  | torch.Tensor.triu_                     | 是       |
| 345  | torch.Tensor.true_divide               | 是       |
| 346  | torch.Tensor.true_divide_              | 是       |
| 347  | torch.Tensor.trunc                     | 是       |
| 348  | torch.Tensor.trunc_                    | 是       |
| 349  | torch.Tensor.type                      | 是       |
| 350  | torch.Tensor.type_as                   | 是       |
| 351  | torch.Tensor.unbind                    | 是       |
| 352  | torch.Tensor.unfold                    | 是       |
| 353  | torch.Tensor.uniform_                  | 是       |
| 354  | torch.Tensor.unique                    | 是       |
| 355  | torch.Tensor.unique_consecutive        | 否       |
| 356  | torch.Tensor.unsqueeze                 | 是       |
| 357  | torch.Tensor.unsqueeze_                | 是       |
| 358  | torch.Tensor.values                    | 否       |
| 359  | torch.Tensor.var                       | 否       |
| 360  | torch.Tensor.vdot                      | 是       |
| 361  | torch.Tensor.view                      | 是       |
| 362  | torch.Tensor.view_as                   | 是       |
| 363  | torch.Tensor.where                     | 是       |
| 364  | torch.Tensor.zero_                     | 是       |
| 365  | torch.BoolTensor                       | 是       |
| 366  | torch.BoolTensor.all                   | 是       |
| 367  | torch.BoolTensor.any                   | 是       |

## Layers (torch.nn)

| 序号 | API名称                                                  | 是否支持                     |
| ---- | -------------------------------------------------------- | ---------------------------- |
| 1    | torch.nn.Parameter                                       | 是                           |
| 2    | torch.nn.UninitializedParameter                          | 是                           |
| 3    | torch.nn.Module                                          | 是                           |
| 4    | torch.nn.Module.add_module                               | 是                           |
| 5    | torch.nn.Module.apply                                    | 是                           |
| 6    | torch.nn.Module.bfloat16                                 | 否                           |
| 7    | torch.nn.Module.buffers                                  | 是                           |
| 8    | torch.nn.Module.children                                 | 是                           |
| 9    | torch.nn.Module.cpu                                      | 是                           |
| 10   | torch.nn.Module.cuda                                     | 否                           |
| 11   | torch.nn.Module.double                                   | 否                           |
| 12   | torch.nn.Module.dump_patches                             | 是                           |
| 13   | torch.nn.Module.eval                                     | 是                           |
| 14   | torch.nn.Module.extra_repr                               | 是                           |
| 15   | torch.nn.Module.float                                    | 是                           |
| 16   | torch.nn.Module.forward                                  | 是                           |
| 17   | torch.nn.Module.half                                     | 是                           |
| 18   | torch.nn.Module.load_state_dict                          | 是                           |
| 19   | torch.nn.Module.modules                                  | 是                           |
| 20   | torch.nn.Module.named_buffers                            | 是                           |
| 21   | torch.nn.Module.named_children                           | 是                           |
| 22   | torch.nn.Module.named_modules                            | 是                           |
| 23   | torch.nn.Module.named_parameters                         | 是                           |
| 24   | torch.nn.Module.parameters                               | 是                           |
| 25   | torch.nn.Module.register_backward_hook                   | 是                           |
| 26   | torch.nn.Module.register_buffer                          | 是                           |
| 27   | torch.nn.Module.register_forward_hook                    | 是                           |
| 28   | torch.nn.Module.register_forward_pre_hook                | 是                           |
| 29   | torch.nn.Module.register_parameter                       | 是                           |
| 30   | torch.nn.register_module_forward_pre_hook                | 否                           |
| 31   | torch.nn.register_module_forward_hook                    | 否                           |
| 32   | torch.nn.register_module_backward_hook                   | 否                           |
| 33   | torch.nn.Module.requires_grad_                           | 是                           |
| 34   | torch.nn.Module.state_dict                               | 是                           |
| 35   | torch.nn.Module.to                                       | 是                           |
| 36   | torch.nn.Module.train                                    | 是                           |
| 37   | torch.nn.Module.type                                     | 是                           |
| 38   | torch.nn.Module.zero_grad                                | 是                           |
| 39   | torch.nn.Sequential                                      | 是                           |
| 40   | torch.nn.ModuleList                                      | 是                           |
| 41   | torch.nn.ModuleList.append                               | 是                           |
| 42   | torch.nn.ModuleList.extend                               | 是                           |
| 43   | torch.nn.ModuleList.insert                               | 是                           |
| 44   | torch.nn.ModuleDict                                      | 是                           |
| 45   | torch.nn.ModuleDict.clear                                | 是                           |
| 46   | torch.nn.ModuleDict.items                                | 是                           |
| 47   | torch.nn.ModuleDict.keys                                 | 是                           |
| 48   | torch.nn.ModuleDict.pop                                  | 是                           |
| 49   | torch.nn.ModuleDict.update                               | 是                           |
| 50   | torch.nn.ModuleDict.values                               | 是                           |
| 51   | torch.nn.ParameterList                                   | 是                           |
| 52   | torch.nn.ParameterList.append                            | 是                           |
| 53   | torch.nn.ParameterList.extend                            | 是                           |
| 54   | torch.nn.ParameterDict                                   | 是                           |
| 55   | torch.nn.ParameterDict.clear                             | 是                           |
| 56   | torch.nn.ParameterDict.items                             | 是                           |
| 57   | torch.nn.ParameterDict.keys                              | 是                           |
| 58   | torch.nn.ParameterDict.pop                               | 是                           |
| 59   | torch.nn.ParameterDict.update                            | 是                           |
| 60   | torch.nn.ParameterDict.values                            | 是                           |
| 61   | torch.nn.Conv1d                                          | 是                           |
| 62   | torch.nn.Conv2d                                          | 是                           |
| 63   | torch.nn.Conv3d                                          | 是                           |
| 64   | torch.nn.ConvTranspose1d                                 | 是                           |
| 65   | torch.nn.ConvTranspose2d                                 | 是                           |
| 66   | torch.nn.ConvTranspose3d                                 | 是                           |
| 67   | torch.nn.LazyConv1d                                      | 是                           |
| 68   | torch.nn.LazyConv2d                                      | 是                           |
| 69   | torch.nn.LazyConv3d                                      | 是                           |
| 70   | torch.nn.LazyConvTranspose1d                             | 是                           |
| 71   | torch.nn.LazyConvTranspose2d                             | 是                           |
| 72   | torch.nn.LazyConvTranspose3d                             | 是                           |
| 73   | torch.nn.Unfold                                          | 是                           |
| 74   | torch.nn.Fold                                            | 是                           |
| 75   | torch.nn.MaxPool1d                                       | 是                           |
| 76   | torch.nn.MaxPool2d                                       | 是                           |
| 77   | torch.nn.MaxPool3d                                       | 是                           |
| 78   | torch.nn.MaxUnpool1d                                     | 是                           |
| 79   | torch.nn.MaxUnpool2d                                     | 是                           |
| 80   | torch.nn.MaxUnpool3d                                     | 是                           |
| 81   | torch.nn.AvgPool1d                                       | 是                           |
| 82   | torch.nn.AvgPool2d                                       | 是                           |
| 83   | torch.nn.AvgPool3d                                       | 是                           |
| 84   | torch.nn.FractionalMaxPool2d                             | 否                           |
| 85   | torch.nn.LPPool1d                                        | 是                           |
| 86   | torch.nn.LPPool2d                                        | 是                           |
| 87   | torch.nn.AdaptiveMaxPool1d                               | 是                           |
| 88   | torch.nn.AdaptiveMaxPool2d                               | 是                           |
| 89   | torch.nn.AdaptiveMaxPool3d                               | 否                           |
| 90   | torch.nn.AdaptiveAvgPool1d                               | 是                           |
| 91   | torch.nn.AdaptiveAvgPool2d                               | 是                           |
| 92   | torch.nn.AdaptiveAvgPool3d                               | 是，仅支持D=1，H=1，W=1场景  |
| 93   | torch.nn.ReflectionPad1d                                 | 是                           |
| 94   | torch.nn.ReflectionPad2d                                 | 是                           |
| 95   | torch.nn.ReplicationPad1d                                | 是                           |
| 96   | torch.nn.ReplicationPad2d                                | 是                           |
| 97   | torch.nn.ReplicationPad3d                                | 否                           |
| 98   | torch.nn.ZeroPad2d                                       | 是                           |
| 99   | torch.nn.ConstantPad1d                                   | 是                           |
| 100  | torch.nn.ConstantPad2d                                   | 是                           |
| 101  | torch.nn.ConstantPad3d                                   | 是                           |
| 102  | torch.nn.ELU                                             | 是                           |
| 103  | torch.nn.Hardshrink                                      | 是                           |
| 104  | torch.nn.Hardsigmoid                                     | 是                           |
| 105  | torch.nn.Hardtanh                                        | 是                           |
| 106  | torch.nn.Hardswish                                       | 是                           |
| 107  | torch.nn.LeakyReLU                                       | 是                           |
| 108  | torch.nn.LogSigmoid                                      | 是                           |
| 109  | torch.nn.MultiheadAttention                              | 是                           |
| 110  | torch.nn.PReLU                                           | 是                           |
| 111  | torch.nn.ReLU                                            | 是                           |
| 112  | torch.nn.ReLU6                                           | 是                           |
| 113  | torch.nn.RReLU                                           | 是                           |
| 114  | torch.nn.SELU                                            | 是                           |
| 115  | torch.nn.CELU                                            | 是                           |
| 116  | torch.nn.GELU                                            | 是                           |
| 117  | torch.nn.Sigmoid                                         | 是                           |
| 118  | torch.nn.SiLU                                            | 是                           |
| 119  | torch.nn.Softplus                                        | 是                           |
| 120  | torch.nn.Softshrink                                      | 是，SoftShrink场景暂不支持   |
| 121  | torch.nn.Softsign                                        | 是                           |
| 122  | torch.nn.Tanh                                            | 是                           |
| 123  | torch.nn.Tanhshrink                                      | 是                           |
| 124  | torch.nn.Threshold                                       | 是                           |
| 125  | torch.nn.Softmin                                         | 是                           |
| 126  | torch.nn.Softmax                                         | 是                           |
| 127  | torch.nn.Softmax2d                                       | 是                           |
| 128  | torch.nn.LogSoftmax                                      | 是                           |
| 129  | torch.nn.AdaptiveLogSoftmaxWithLoss                      | 否                           |
| 130  | torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob             | 否                           |
| 131  | torch.nn.AdaptiveLogSoftmaxWithLoss.predict              | 否                           |
| 132  | torch.nn.BatchNorm1d                                     | 是                           |
| 133  | torch.nn.BatchNorm2d                                     | 是                           |
| 134  | torch.nn.BatchNorm3d                                     | 是                           |
| 135  | torch.nn.GroupNorm                                       | 是                           |
| 136  | torch.nn.SyncBatchNorm                                   | 是                           |
| 137  | torch.nn.SyncBatchNorm.convert_sync_batchnorm            | 是                           |
| 138  | torch.nn.InstanceNorm1d                                  | 是                           |
| 139  | torch.nn.InstanceNorm2d                                  | 是                           |
| 140  | torch.nn.InstanceNorm3d                                  | 是                           |
| 141  | torch.nn.LayerNorm                                       | 是                           |
| 142  | torch.nn.LocalResponseNorm                               | 是                           |
| 143  | torch.nn.RNNBase                                         | 是                           |
| 144  | torch.nn.RNNBase.flatten_parameters                      | 是                           |
| 145  | torch.nn.RNN                                             | 是                           |
| 146  | torch.nn.LSTM                                            | 是                           |
| 147  | torch.nn.GRU                                             | 是，DynamicGRUV2场景暂不支持 |
| 148  | torch.nn.RNNCell                                         | 是                           |
| 149  | torch.nn.LSTMCell                                        | 是，非16对齐场景暂不支持     |
| 150  | torch.nn.GRUCell                                         | 是                           |
| 151  | torch.nn.Transformer                                     | 是                           |
| 152  | torch.nn.Transformer.forward                             | 是                           |
| 153  | torch.nn.Transformer.generate_square_subsequent_mask     | 是                           |
| 154  | torch.nn.TransformerEncoder                              | 是                           |
| 155  | torch.nn.TransformerEncoder.forward                      | 是                           |
| 156  | torch.nn.TransformerDecoder                              | 是                           |
| 157  | torch.nn.TransformerDecoder.forward                      | 是                           |
| 158  | torch.nn.TransformerEncoderLayer                         | 是                           |
| 159  | torch.nn.TransformerEncoderLayer.forward                 | 是                           |
| 160  | torch.nn.TransformerDecoderLayer                         | 是                           |
| 161  | torch.nn.TransformerDecoderLayer.forward                 | 是                           |
| 162  | torch.nn.Identity                                        | 是                           |
| 163  | torch.nn.Linear                                          | 是                           |
| 164  | torch.nn.Bilinear                                        | 是                           |
| 165  | torch.nn.LazyLinear                                      | 是                           |
| 166  | torch.nn.Dropout                                         | 是                           |
| 167  | torch.nn.Dropout2d                                       | 是                           |
| 168  | torch.nn.Dropout3d                                       | 是                           |
| 169  | torch.nn.AlphaDropout                                    | 是                           |
| 170  | torch.nn.Embedding                                       | 是                           |
| 171  | torch.nn.Embedding.from_pretrained                       | 是                           |
| 172  | torch.nn.EmbeddingBag                                    | 是                           |
| 173  | torch.nn.EmbeddingBag.from_pretrained                    | 是                           |
| 174  | torch.nn.CosineSimilarity                                | 是                           |
| 175  | torch.nn.PairwiseDistance                                | 是                           |
| 176  | torch.nn.L1Loss                                          | 是                           |
| 177  | torch.nn.MSELoss                                         | 是                           |
| 178  | torch.nn.CrossEntropyLoss                                | 是                           |
| 179  | torch.nn.CTCLoss                                         | 是                           |
| 180  | torch.nn.NLLLoss                                         | 是                           |
| 181  | torch.nn.PoissonNLLLoss                                  | 是                           |
| 182  | torch.nn.GaussianNLLLoss                                 | 是                           |
| 183  | torch.nn.KLDivLoss                                       | 是                           |
| 184  | torch.nn.BCELoss                                         | 是                           |
| 185  | torch.nn.BCEWithLogitsLoss                               | 是                           |
| 186  | torch.nn.MarginRankingLoss                               | 是                           |
| 187  | torch.nn.HingeEmbeddingLoss                              | 是                           |
| 188  | torch.nn.MultiLabelMarginLoss                            | 是                           |
| 189  | torch.nn.SmoothL1Loss                                    | 是                           |
| 190  | torch.nn.SoftMarginLoss                                  | 是                           |
| 191  | torch.nn.MultiLabelSoftMarginLoss                        | 是                           |
| 192  | torch.nn.CosineEmbeddingLoss                             | 是                           |
| 193  | torch.nn.MultiMarginLoss                                 | 否                           |
| 194  | torch.nn.TripletMarginLoss                               | 是                           |
| 195  | torch.nn.TripletMarginLossWithDistanceLoss               | 是                           |
| 196  | torch.nn.PixelShuffle                                    | 是                           |
| 197  | torch.nn.PixelUnshuffle                                  | 是                           |
| 198  | torch.nn.Upsample                                        | 是                           |
| 199  | torch.nn.UpsamplingNearest2d                             | 是                           |
| 200  | torch.nn.UpsamplingBilinear2d                            | 是                           |
| 201  | torch.nn.ChannelShuffle                                  | 是                           |
| 202  | torch.nn.DataParallel                                    | 是                           |
| 203  | torch.nn.parallel.DistributedDataParallel                | 是                           |
| 204  | torch.nn.parallel.DistributedDataParallel.no_sync        | 是                           |
| 205  | torch.nn.utils.clip_grad_norm_                           | 是                           |
| 206  | torch.nn.utils.clip_grad_value_                          | 是                           |
| 207  | torch.nn.utils.parameters_to_vector                      | 是                           |
| 208  | torch.nn.utils.vector_to_parameters                      | 是                           |
| 209  | torch.nn.utils.Prune.BasePruningMethod                   | 是                           |
| 210  | torch.nn.utils.prune.PruningContainer                    | 是                           |
| 211  | torch.nn.utils.prune.PruningContainer.add_pruning_method | 是                           |
| 212  | torch.nn.utils.prune.PruningContainer.apply              | 是                           |
| 213  | torch.nn.utils.prune.PruningContainer.apply_mask         | 是                           |
| 214  | torch.nn.utils.prune.PruningContainer.compute_mask       | 是                           |
| 215  | torch.nn.utils.prune.PruningContainer.prune              | 是                           |
| 216  | torch.nn.utils.prune.PruningContainer.remove             | 是                           |
| 217  | torch.nn.utils.prune.Identity                            | 是                           |
| 218  | torch.nn.utils.prune.Identity.apply                      | 是                           |
| 219  | torch.nn.utils.prune.Identity.apply_mask                 | 是                           |
| 220  | torch.nn.utils.prune.Identity.prune                      | 是                           |
| 221  | torch.nn.utils.prune.Identity.remove                     | 是                           |
| 222  | torch.nn.utils.prune.RandomUnstructured                  | 是                           |
| 223  | torch.nn.utils.prune.RandomUnstructured.apply            | 是                           |
| 224  | torch.nn.utils.prune.RandomUnstructured.apply_mask       | 是                           |
| 225  | torch.nn.utils.prune.RandomUnstructured.prune            | 是                           |
| 226  | torch.nn.utils.prune.RandomUnstructured.remove           | 是                           |
| 227  | torch.nn.utils.prune.L1Unstructured                      | 是                           |
| 228  | torch.nn.utils.prune.L1Unstructured.apply                | 是                           |
| 229  | torch.nn.utils.prune.L1Unstructured.apply_mask           | 是                           |
| 230  | torch.nn.utils.prune.L1Unstructured.prune                | 是                           |
| 231  | torch.nn.utils.prune.L1Unstructured.remove               | 是                           |
| 232  | torch.nn.utils.prune.RandomStructured                    | 是                           |
| 233  | torch.nn.utils.prune.RandomStructured.apply              | 是                           |
| 234  | torch.nn.utils.prune.RandomStructured.apply_mask         | 是                           |
| 235  | torch.nn.utils.prune.RandomStructured.compute_mask       | 是                           |
| 236  | torch.nn.utils.prune.RandomStructured.prune              | 是                           |
| 237  | torch.nn.utils.prune.RandomStructured.remove             | 是                           |
| 238  | torch.nn.utils.prune.LnStructured                        | 是                           |
| 239  | torch.nn.utils.prune.LnStructured.apply                  | 是                           |
| 240  | torch.nn.utils.prune.LnStructured.apply_mask             | 是                           |
| 241  | torch.nn.utils.prune.LnStructured.compute_mask           | 是                           |
| 242  | torch.nn.utils.prune.LnStructured.prune                  | 是                           |
| 243  | torch.nn.utils.prune.LnStructured.remove                 | 是                           |
| 244  | torch.nn.utils.prune.CustomFromMask                      | 是                           |
| 245  | torch.nn.utils.prune.CustomFromMask.apply                | 是                           |
| 246  | torch.nn.utils.prune.CustomFromMask.apply_mask           | 是                           |
| 247  | torch.nn.utils.prune.CustomFromMask.prune                | 是                           |
| 248  | torch.nn.utils.prune.CustomFromMask.remove               | 是                           |
| 249  | torch.nn.utils.prune.identity                            | 是                           |
| 250  | torch.nn.utils.prune.random_unstructured                 | 是                           |
| 251  | torch.nn.utils.prune.l1_unstructured                     | 是                           |
| 252  | torch.nn.utils.prune.random_structured                   | 是                           |
| 253  | torch.nn.utils.prune.ln_structured                       | 是                           |
| 254  | torch.nn.utils.prune.global_unstructured                 | 是                           |
| 255  | torch.nn.utils.prune.custom_from_mask                    | 是                           |
| 256  | torch.nn.utils.prune.remove                              | 是                           |
| 257  | torch.nn.utils.prune.is_pruned                           | 是                           |
| 258  | torch.nn.utils.weight_norm                               | 是                           |
| 259  | torch.nn.utils.remove_weight_norm                        | 是                           |
| 260  | torch.nn.utils.spectral_norm                             | 是                           |
| 261  | torch.nn.utils.remove_spectral_norm                      | 是                           |
| 262  | torch.nn.utils.rnn.PackedSequence                        | 是                           |
| 263  | torch.nn.utils.rnn.pack_padded_sequence                  | 是                           |
| 264  | torch.nn.utils.rnn.pad_packed_sequence                   | 是                           |
| 265  | torch.nn.utils.rnn.pad_sequence                          | 是                           |
| 266  | torch.nn.utils.rnn.pack_sequence                         | 是                           |
| 267  | torch.nn.Flatten                                         | 是                           |
| 268  | torch.nn.Unflatten                                       | 是                           |
| 269  | torch.nn.modules.lazy.LazyModuleMixin                    | 是                           |
| 270  | torch.quantization.quantize                              | 否                           |
| 271  | torch.quantization.quantize_dynamic                      | 否                           |
| 272  | torch.quantization.quantize_qat                          | 否                           |
| 273  | torch.quantization.prepare                               | 是                           |
| 274  | torch.quantization.prepare_qat                           | 否                           |
| 275  | torch.quantization.convert                               | 否                           |
| 276  | torch.quantization.QConfig                               | 是                           |
| 277  | torch.quantization.QConfigDynamic                        | 是                           |
| 278  | torch.quantization.fuse_modules                          | 是                           |
| 279  | torch.quantization.QuantStub                             | 是                           |
| 280  | torch.quantization.DeQuantStub                           | 是                           |
| 281  | torch.quantization.QuantWrapper                          | 是                           |
| 282  | torch.quantization.add_quant_dequant                     | 是                           |
| 283  | torch.quantization.add_observer_                         | 是                           |
| 284  | torch.quantization.swap_module                           | 是                           |
| 285  | torch.quantization.propagate_qconfig_                    | 是                           |
| 286  | torch.quantization.default_eval_fn                       | 是                           |
| 287  | torch.quantization.MinMaxObserver                        | 是                           |
| 288  | torch.quantization.MovingAverageMinMaxObserver           | 是                           |
| 289  | torch.quantization.PerChannelMinMaxObserver              | 是                           |
| 290  | torch.quantization.MovingAveragePerChannelMinMaxObserver | 是                           |
| 291  | torch.quantization.HistogramObserver                     | 否                           |
| 292  | torch.quantization.FakeQuantize                          | 否                           |
| 293  | torch.quantization.NoopObserver                          | 是                           |
| 294  | torch.quantization.get_observer_dict                     | 是                           |
| 295  | torch.quantization.RecordingObserver                     | 是                           |
| 296  | torch.nn.intrinsic.ConvBn2d                              | 是                           |
| 297  | torch.nn.intrinsic.ConvBnReLU2d                          | 是                           |
| 298  | torch.nn.intrinsic.ConvReLU2d                            | 是                           |
| 299  | torch.nn.intrinsic.ConvReLU3d                            | 是                           |
| 300  | torch.nn.intrinsic.LinearReLU                            | 是                           |
| 301  | torch.nn.intrinsic.qat.ConvBn2d                          | 否                           |
| 302  | torch.nn.intrinsic.qat.ConvBnReLU2d                      | 否                           |
| 303  | torch.nn.intrinsic.qat.ConvReLU2d                        | 否                           |
| 304  | torch.nn.intrinsic.qat.LinearReLU                        | 否                           |
| 305  | torch.nn.intrinsic.quantized.ConvReLU2d                  | 否                           |
| 306  | torch.nn.intrinsic.quantized.ConvReLU3d                  | 否                           |
| 307  | torch.nn.intrinsic.quantized.LinearReLU                  | 否                           |
| 308  | torch.nn.qat.Conv2d                                      | 否                           |
| 309  | torch.nn.qat.Conv2d.from_float                           | 否                           |
| 310  | torch.nn.qat.Linear                                      | 否                           |
| 311  | torch.nn.qat.Linear.from_float                           | 否                           |
| 312  | torch.nn.quantized.functional.relu                       | 否                           |
| 313  | torch.nn.quantized.functional.linear                     | 否                           |
| 314  | torch.nn.quantized.functional.conv2d                     | 否                           |
| 315  | torch.nn.quantized.functional.conv3d                     | 否                           |
| 316  | torch.nn.quantized.functional.max_pool2d                 | 否                           |
| 317  | torch.nn.quantized.functional.adaptive_avg_pool2d        | 否                           |
| 318  | torch.nn.quantized.functional.avg_pool2d                 | 否                           |
| 319  | torch.nn.quantized.functional.interpolate                | 否                           |
| 320  | torch.nn.quantized.functional.upsample                   | 否                           |
| 321  | torch.nn.quantized.functional.upsample_bilinear          | 否                           |
| 322  | torch.nn.quantized.functional.upsample_nearest           | 否                           |
| 323  | torch.nn.quantized.ReLU                                  | 否                           |
| 324  | torch.nn.quantized.ReLU6                                 | 否                           |
| 325  | torch.nn.quantized.Conv2d                                | 否                           |
| 326  | torch.nn.quantized.Conv2d.from_float                     | 否                           |
| 327  | torch.nn.quantized.Conv3d                                | 否                           |
| 328  | torch.nn.quantized.Conv3d.from_float                     | 否                           |
| 329  | torch.nn.quantized.FloatFunctional                       | 是                           |
| 330  | torch.nn.quantized.QFunctional                           | 否                           |
| 331  | torch.nn.quantized.Quantize                              | 是                           |
| 332  | torch.nn.quantized.DeQuantize                            | 否                           |
| 333  | torch.nn.quantized.Linear                                | 否                           |
| 334  | torch.nn.quantized.Linear.from_float                     | 否                           |
| 335  | torch.nn.quantized.dynamic.Linear                        | 否                           |
| 336  | torch.nn.quantized.dynamic.Linear.from_float             | 否                           |
| 337  | torch.nn.quantized.dynamic.LSTM                          | 否                           |

## Functions(torch.nn.functional)

| 序号 | API名称                                               | 是否支持                    |
| ---- | ----------------------------------------------------- | --------------------------- |
| 1    | torch.nn.functional.conv1d                            | 是                          |
| 2    | torch.nn.functional.conv2d                            | 是                          |
| 3    | torch.nn.functional.conv3d                            | 是                          |
| 4    | torch.nn.functional.conv_transpose1d                  | 是                          |
| 5    | torch.nn.functional.conv_transpose2d                  | 是                          |
| 6    | torch.nn.functional.conv_transpose3d                  | 是                          |
| 7    | torch.nn.functional.unfold                            | 是                          |
| 8    | torch.nn.functional.fold                              | 是                          |
| 9    | torch.nn.functional.avg_pool1d                        | 是                          |
| 10   | torch.nn.functional.avg_pool2d                        | 是                          |
| 11   | torch.nn.functional.avg_pool3d                        | 是                          |
| 12   | torch.nn.functional.max_pool1d                        | 是                          |
| 13   | torch.nn.functional.max_pool2d                        | 是                          |
| 14   | torch.nn.functional.max_pool3d                        | 是                          |
| 15   | torch.nn.functional.max_unpool1d                      | 是                          |
| 16   | torch.nn.functional.max_unpool2d                      | 是                          |
| 17   | torch.nn.functional.max_unpool3d                      | 是                          |
| 18   | torch.nn.functional.lp_pool1d                         | 是                          |
| 19   | torch.nn.functional.lp_pool2d                         | 是                          |
| 20   | torch.nn.functional.adaptive_max_pool1d               | 是                          |
| 21   | torch.nn.functional.adaptive_max_pool2d               | 是                          |
| 22   | torch.nn.functional.adaptive_max_pool3d               | 否                          |
| 23   | torch.nn.functional.adaptive_avg_pool1d               | 是                          |
| 24   | torch.nn.functional.adaptive_avg_pool2d               | 是                          |
| 25   | torch.nn.functional.adaptive_avg_pool3d               | 是，仅支持D=1，H=1，W=1场景 |
| 26   | torch.nn.functional.threshold                         | 是                          |
| 27   | torch.nn.functional.threshold_                        | 是                          |
| 28   | torch.nn.functional.relu                              | 是                          |
| 29   | torch.nn.functional.relu_                             | 是                          |
| 30   | torch.nn.functional.hardtanh                          | 是                          |
| 31   | torch.nn.functional.hardtanh_                         | 是                          |
| 32   | torch.nn.functional.swish                             | 是                          |
| 33   | torch.nn.functional.relu6                             | 是                          |
| 34   | torch.nn.functional.elu                               | 是                          |
| 35   | torch.nn.functional.elu_                              | 是                          |
| 36   | torch.nn.functional.selu                              | 是                          |
| 37   | torch.nn.functional.celu                              | 是                          |
| 38   | torch.nn.functional.leaky_relu                        | 是                          |
| 39   | torch.nn.functional.leaky_relu_                       | 是                          |
| 40   | torch.nn.functional.prelu                             | 是                          |
| 41   | torch.nn.functional.rrelu                             | 是                          |
| 42   | torch.nn.functional.rrelu_                            | 是                          |
| 43   | torch.nn.functional.glu                               | 是                          |
| 44   | torch.nn.functional.gelu                              | 是                          |
| 45   | torch.nn.functional.logsigmoid                        | 是                          |
| 46   | torch.nn.functional.hardshrink                        | 是                          |
| 47   | torch.nn.functional.tanhshrink                        | 是                          |
| 48   | torch.nn.functional.softsign                          | 是                          |
| 49   | torch.nn.functional.softplus                          | 是                          |
| 50   | torch.nn.functional.softmin                           | 是                          |
| 51   | torch.nn.functional.softmax                           | 是                          |
| 52   | torch.nn.functional.softshrink                        | 是                          |
| 53   | torch.nn.functional.gumbel_softmax                    | 否                          |
| 54   | torch.nn.functional.log_softmax                       | 是                          |
| 55   | torch.nn.functional.tanh                              | 是                          |
| 56   | torch.nn.functional.sigmoid                           | 是                          |
| 57   | torch.nn.functional.hardsigmoid                       | 是                          |
| 58   | torch.nn.functional.hardswish                         | 是                          |
| 59   | torch.nn.functional.feature_alpha_dropout             | 是                          |
| 60   | torch.nn.functional.silu                              | 是                          |
| 61   | torch.nn.functional.batch_norm                        | 是                          |
| 62   | torch.nn.functional.instance_norm                     | 是                          |
| 63   | torch.nn.functional.layer_norm                        | 是                          |
| 64   | torch.nn.functional.local_response_norm               | 是                          |
| 65   | torch.nn.functional.normalize                         | 是                          |
| 66   | torch.nn.functional.linear                            | 是                          |
| 67   | torch.nn.functional.bilinear                          | 是                          |
| 68   | torch.nn.functional.dropout                           | 是                          |
| 69   | torch.nn.functional.alpha_dropout                     | 是                          |
| 70   | torch.nn.functional.dropout2d                         | 是                          |
| 71   | torch.nn.functional.dropout3d                         | 是                          |
| 72   | torch.nn.functional.embedding                         | 是                          |
| 73   | torch.nn.functional.embedding_bag                     | 是                          |
| 74   | torch.nn.functional.one_hot                           | 是                          |
| 75   | torch.nn.functional.pairwise_distance                 | 是                          |
| 76   | torch.nn.functional.cosine_similarity                 | 是                          |
| 77   | torch.nn.functional.pdist                             | 是                          |
| 78   | torch.nn.functional.binary_cross_entropy              | 是（参数y仅支持y=1,y=0）    |
| 79   | torch.nn.functional.binary_cross_entropy_with_logits  | 是                          |
| 80   | torch.nn.functional.poisson_nll_loss                  | 是                          |
| 81   | torch.nn.functional.cosine_embedding_loss             | 是                          |
| 82   | torch.nn.functional.cross_entropy                     | 是                          |
| 83   | torch.nn.functional.ctc_loss                          | 是（仅支持2维输入）         |
| 84   | torch.nn.functional.hinge_embedding_loss              | 是                          |
| 85   | torch.nn.functional.kl_div                            | 是                          |
| 86   | torch.nn.functional.l1_loss                           | 是                          |
| 87   | torch.nn.functional.mse_loss                          | 是                          |
| 88   | torch.nn.functional.margin_ranking_loss               | 是                          |
| 89   | torch.nn.functional.multilabel_margin_loss            | 是                          |
| 90   | torch.nn.functional.multilabel_soft_margin_loss       | 是                          |
| 91   | torch.nn.functional.multi_margin_loss                 | 是                          |
| 92   | torch.nn.functional.nll_loss                          | 是                          |
| 93   | torch.nn.functional.smooth_l1_loss                    | 是                          |
| 94   | torch.nn.functional.soft_margin_loss                  | 是                          |
| 95   | torch.nn.functional.triplet_margin_loss               | 是                          |
| 96   | torch.nn.functional.triplet_margin_with_distance_loss | 是                          |
| 97   | torch.nn.functional.pixel_shuffle                     | 否                          |
| 98   | torch.nn.functional.pixel_unshuffle                   | 否                          |
| 99   | torch.nn.functional.pad                               | 是                          |
| 100  | torch.nn.functional.interpolate                       | 是                          |
| 101  | torch.nn.functional.upsample                          | 是                          |
| 102  | torch.nn.functional.upsample_nearest                  | 是                          |
| 103  | torch.nn.functional.upsample_bilinear                 | 是                          |
| 104  | torch.nn.functional.grid_sample                       | 是                          |
| 105  | torch.nn.functional.affine_grid                       | 是                          |
| 106  | torch.nn.parallel.data_parallel                       | 否                          |

## torch.distributed

| 序号 | API名称                                   | 是否支持 |
| ---- | ----------------------------------------- | -------- |
| 1    | torch.distributed.is_available            | 是       |
| 2    | torch.distributed.init_process_group      | 是       |
| 3    | torch.distributed.Backend                 | 是       |
| 4    | torch.distributed.get_backend             | 是       |
| 5    | torch.distributed.get_rank                | 是       |
| 6    | torch.distributed.get_world_size          | 是       |
| 7    | torch.distributed.is_initialized          | 是       |
| 8    | torch.distributed.is_mpi_available        | 是       |
| 9    | torch.distributed.is_nccl_available       | 是       |
| 10   | torch.distributed.new_group               | 是       |
| 11   | torch.distributed.Store                   | 是       |
| 12   | torch.distributed.TCPStore                | 是       |
| 13   | torch.distributed.HashStore               | 是       |
| 14   | torch.distributed.FileStore               | 是       |
| 15   | torch.distributed.PrefixStore             | 是       |
| 16   | torch.distributed.Store.set               | 是       |
| 17   | torch.distributed.Store.get               | 是       |
| 18   | torch.distributed.Store.add               | 是       |
| 19   | torch.distributed.Store.wait              | 是       |
| 20   | torch.distributed.Store.num_keys          | 是       |
| 21   | torch.distributed.Store.delete_keys       | 是       |
| 22   | torch.distributed.Store.set_timeout       | 是       |
| 23   | torch.distributed.send                    | 否       |
| 24   | torch.distributed.recv                    | 否       |
| 25   | torch.distributed.isend                   | 否       |
| 26   | torch.distributed.irecv                   | 否       |
| 27   | is_completed                              | 是       |
| 28   | wait                                      | 是       |
| 29   | torch.distributed.broadcast               | 是       |
| 30   | torch.distributed.broadcast_object_list   | 是       |
| 31   | torch.distributed.all_reduce              | 是       |
| 32   | torch.distributed.reduce                  | 否       |
| 33   | torch.distributed.all_gather              | 是       |
| 34   | torch.distributed.all_gather_object       | 是       |
| 35   | torch.distributed.gather_object           | 是       |
| 36   | torch.distributed.gather                  | 否       |
| 37   | torch.distributed.scatter                 | 否       |
| 38   | torch.distributed.scatter_object_list     | 是       |
| 39   | torch.distributed.reduce_scatter          | 是       |
| 40   | torch.distributed.reduce_scatter_multigpu | 是       |
| 41   | torch.distributed.all_to_all              | 是       |
| 42   | torch.distributed.barrier                 | 是       |
| 43   | torch.distributed.ReduceOp                | 是       |
| 44   | torch.distributed.reduce_op               | 是       |
| 45   | torch.distributed.broadcast_multigpu      | 否       |
| 46   | torch.distributed.all_reduce_multigpu     | 否       |
| 47   | torch.distributed.reduce_multigpu         | 否       |
| 48   | torch.distributed.all_gather_multigpu     | 否       |
| 49   | torch.distributed.launch                  | 是       |
| 50   | torch.multiprocessing.spawn               | 是       |

## torch_npu.npu

| 序号 | API名称                                       | npu对应API名称                                   | 是否支持       |
| ---- | --------------------------------------------- | ------------------------------------------------ | -------------- |
| 1    | torch.cuda.can_device_access_peer             | torch_npu.npu.can_device_access_peer             | 是             |
| 2    | torch.cuda.current_blas_handle                | torch_npu.npu.current_blas_handle                | 否             |
| 3    | torch.cuda.current_device                     | torch_npu.npu.current_device                     | 是             |
| 4    | torch.cuda.current_stream                     | torch_npu.npu.current_stream                     | 是             |
| 5    | torch.cuda.default_stream                     | torch_npu.npu.default_stream                     | 是             |
| 6    | torch.cuda.device                             | torch_npu.npu.device                             | 是             |
| 7    | torch.cuda.device_count                       | torch_npu.npu.device_count                       | 是             |
| 8    | torch.cuda.device_of                          | torch_npu.npu.device_of                          | 是             |
| 9    | torch.cuda.get_device_capability              | torch_npu.npu.get_device_capability              | 否             |
| 10   | torch.cuda.get_device_name                    | torch_npu.npu.get_device_name                    | 否             |
| 11   | torch.cuda.get_device_properties              | torch_npu.npu.get_device_properties              | 否             |
| 12   | torch.cuda.get_gencode_flags                  | torch_npu.npu.get_gencode_flags                  | 是             |
| 13   | torch.cuda.init                               | torch_npu.npu.init                               | 是             |
| 14   | torch.cuda.ipc_collect                        | torch_npu.npu.ipc_collect                        | 否             |
| 15   | torch.cuda.is_available                       | torch_npu.npu.is_available                       | 是             |
| 16   | torch.cuda.is_initialized                     | torch_npu.npu.is_initialized                     | 是             |
| 17   | torch.cuda.set_device                         | torch_npu.npu.set_device                         | 是（部分支持） |
| 18   | torch.cuda.stream                             | torch_npu.npu.stream                             | 是             |
| 19   | torch.cuda.synchronize                        | torch_npu.npu.synchronize                        | 是             |
| 20   | torch.cuda.get_arch_list                      | torch_npu.npu.get_arch_list                      | 是             |
| 21   | torch.cuda.get_rng_state                      | torch_npu.npu.get_rng_state                      | 是             |
| 22   | torch.cuda.get_rng_state_all                  | torch_npu.npu.get_rng_state_all                  | 是             |
| 23   | torch.cuda.set_rng_state                      | torch_npu.npu.set_rng_state                      | 是             |
| 24   | torch.cuda.set_rng_state_all                  | torch_npu.npu.set_rng_state_all                  | 是             |
| 25   | torch.cuda.manual_seed                        | torch_npu.npu.manual_seed                        | 是             |
| 26   | torch.cuda.manual_seed_all                    | torch_npu.npu.manual_seed_all                    | 是             |
| 27   | torch.cuda.seed                               | torch_npu.npu.seed                               | 是             |
| 28   | torch.cuda.seed_all                           | torch_npu.npu.seed_all                           | 是             |
| 29   | torch.cuda.initial_seed                       | torch_npu.npu.initial_seed                       | 是             |
| 30   | torch.cuda.comm.broadcast                     | torch_npu.npu.comm.broadcast                     | 否             |
| 31   | torch.cuda.comm.broadcast_coalesced           | torch_npu.npu.comm.broadcast_coalesced           | 否             |
| 32   | torch.cuda.comm.reduce_add                    | torch_npu.npu.comm.reduce_add                    | 否             |
| 33   | torch.cuda.comm.scatter                       | torch_npu.npu.comm.scatter                       | 否             |
| 34   | torch.cuda.comm.gather                        | torch_npu.npu.comm.gather                        | 否             |
| 35   | torch.cuda.Stream                             | torch_npu.npu.Stream                             | 是             |
| 36   | torch.cuda.Stream.query                       | torch_npu.npu.Stream.query                       | 否             |
| 37   | torch.cuda.Stream.record_event                | torch_npu.npu.Stream.record_event                | 是             |
| 38   | torch.cuda.Stream.synchronize                 | torch_npu.npu.Stream.synchronize                 | 是             |
| 39   | torch.cuda.Stream.wait_event                  | torch_npu.npu.Stream.wait_event                  | 是             |
| 40   | torch.cuda.Stream.wait_stream                 | torch_npu.npu.Stream.wait_stream                 | 是             |
| 41   | torch.cuda.Event                              | torch_npu.npu.Event                              | 是             |
| 42   | torch.cuda.Event.elapsed_time                 | torch_npu.npu.Event.elapsed_time                 | 是             |
| 43   | torch.cuda.Event.from_ipc_handle              | torch_npu.npu.Event.from_ipc_handle              | 否             |
| 44   | torch.cuda.Event.ipc_handle                   | torch_npu.npu.Event.ipc_handle                   | 否             |
| 45   | torch.cuda.Event.query                        | torch_npu.npu.Event.query                        | 是             |
| 46   | torch.cuda.Event.record                       | torch_npu.npu.Event.record                       | 是             |
| 47   | torch.cuda.Event.synchronize                  | torch_npu.npu.Event.synchronize                  | 是             |
| 48   | torch.cuda.Event.wait                         | torch_npu.npu.Event.wait                         | 是             |
| 49   | torch.cuda.empty_cache                        | torch_npu.npu.empty_cache                        | 是             |
| 50   | torch.cuda.list_gpu_processes                 | torch_npu.npu.list_gpu_processes                 | 是             |
| 51   | torch.cuda.memory_stats                       | torch_npu.npu.memory_stats                       | 是             |
| 52   | torch.cuda.memory_summary                     | torch_npu.npu.memory_summary                     | 是             |
| 53   | torch.cuda.memory_snapshot                    | torch_npu.npu.memory_snapshot                    | 是             |
| 54   | torch.cuda.memory_allocated                   | torch_npu.npu.memory_allocated                   | 是             |
| 55   | torch.cuda.max_memory_allocated               | torch_npu.npu.max_memory_allocated               | 是             |
| 56   | torch.cuda.reset_max_memory_allocated         | torch_npu.npu.reset_max_memory_allocated         | 是             |
| 57   | torch.cuda.memory_reserved                    | torch_npu.npu.memory_reserved                    | 是             |
| 58   | torch.cuda.max_memory_reserved                | torch_npu.npu.max_memory_reserved                | 是             |
| 59   | torch.cuda.set_per_process_memory_fraction    | torch_npu.npu.set_per_process_memory_fraction    | 是             |
| 60   | torch.cuda.memory_cached                      | torch_npu.npu.memory_cached                      | 是             |
| 61   | torch.cuda.max_memory_cached                  | torch_npu.npu.max_memory_cached                  | 是             |
| 62   | torch.cuda.reset_max_memory_cached            | torch_npu.npu.reset_max_memory_cached            | 是             |
| 63   | torch.cuda.nvtx.mark                          | torch_npu.npu.nvtx.mark                          | 否             |
| 64   | torch.cuda.nvtx.range_push                    | torch_npu.npu.nvtx.range_push                    | 否             |
| 65   | torch.cuda.nvtx.range_pop                     | torch_npu.npu.nvtx.range_pop                     | 否             |
| 66   | torch.cuda.amp.autocast                       | torch_npu.npu.amp.autocast                       | 是             |
| 67   | torch.cuda.amp.custom_fwd                     | torch_npu.npu.amp.custom_fwd                     | 是             |
| 68   | torch.cuda.amp.custom_bwd                     | torch_npu.npu.amp.custom_bwd                     | 是             |
| 69   | torch.cuda._sleep                             | torch_npu.npu._sleep                             | 否             |
| 70   | torch.cuda.Stream.priority_range              | torch_npu.npu.Stream.priority_range              | 否             |
| 71   | torch.cuda.amp.GradScaler                     | torch_npu.npu.amp.GradScaler                     | 是             |
| 72   | torch.cuda.amp.GradScaler.get_backoff_factor  | torch_npu.npu.amp.GradScaler.get_backoff_factor  | 是             |
| 73   | torch.cuda.amp.GradScaler.get_growth_factor   | torch_npu.npu.amp.GradScaler.get_growth_factor   | 是             |
| 74   | torch.cuda.amp.GradScaler.get_growth_interval | torch_npu.npu.amp.GradScaler.get_growth_interval | 是             |
| 75   | torch.cuda.amp.GradScaler.get_scale           | torch_npu.npu.amp.GradScaler.get_scale           | 是             |
| 76   | torch.cuda.amp.GradScaler.is_enabled          | torch_npu.npu.amp.GradScaler.is_enabled          | 是             |
| 77   | torch.cuda.amp.GradScaler.load_state_dict     | torch_npu.npu.amp.GradScaler.load_state_dict     | 是             |
| 78   | torch.cuda.amp.GradScaler.scale               | torch_npu.npu.amp.GradScaler.scale               | 是             |
| 79   | torch.cuda.amp.GradScaler.set_backoff_factor  | torch_npu.npu.amp.GradScaler.set_backoff_factor  | 是             |
| 80   | torch.cuda.amp.GradScaler.set_growth_factor   | torch_npu.npu.amp.GradScaler.set_growth_factor   | 是             |
| 81   | torch.cuda.amp.GradScaler.set_growth_interval | torch_npu.npu.amp.GradScaler.set_growth_interval | 是             |
| 82   | torch.cuda.amp.GradScaler.state_dict          | torch_npu.npu.amp.GradScaler.state_dict          | 是             |
| 83   | torch.cuda.amp.GradScaler.step                | torch_npu.npu.amp.GradScaler.step                | 是             |
| 84   | torch.cuda.amp.GradScaler.unscale_            | torch_npu.npu.amp.GradScaler.unscale_            | 是             |
| 85   | torch.cuda.amp.GradScaler.update              | torch_npu.npu.amp.GradScaler.update              | 是             |

torch_npu.npu.set_device()接口只支持在程序开始的位置通过set_device进行指定，不支持多次指定和with torch_npu.npu.device(id)方式的device切换

## NPU自定义算子

| 序号 | 算子名称                                        |
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
| 33   | torch_npu._npu_dropout                          |
| 34   | torch_npu._npu_dropout_inplace                  |
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

详细算子接口说明：

> torch_npu.npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out = (var, m, v))

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

> torch_npu.npu_convolution_transpose(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor

Applies a 2D or 3D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.

- Parameters：
  - **input** (Tensor) - input tensor of shape(minibatch, in_channels, iH, iW) or (minibatch, in_channels, iT, iH, iW)
  - **weight** (Tensor) - filters of shape(in_channels, out_channels/groups, kH, kW) or (in_channels, out_channels/groups, kT, kH, kW)
  - **bias** (Tensor, optional) - optional bias of shape(out_channels)
  - **padding** (ListInt) - (dilation * (kernel_size - 1) - padding) zero-padding will be added to both sides of each dimension in the input
  - **output_padding** (ListInt) - additional size added to one side of each dimension in the output shape.
  - **stride** (ListInt) - the stride of the convolving kernel
  - **dilation** (ListInt) - the spacing between kernel elements
  - **groups** (Number) - split input into groups, in_channels should be divisible by the number of groups

- constraints：

  None

- Examples：

  None

> torch_npu.npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor

Applies a 2D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.

- Parameters：
  - **input** (Tensor) - input tensor of shape(minibatch, in_channels, iH, iW)
  - **weight** (Tensor) - filters of shape(in_channels, out_channels/groups, kH, kW)
  - **bias** (Tensor, optional) - optional bias of shape(out_channels)
  - **padding** (ListInt) - (dilation * (kernel_size - 1) - padding) zero-padding will be added to both sides of each dimension in the input
  - **output_padding** (ListInt) - additional size added to one side of each dimension in the output shape.
  - **stride** (ListInt) - the stride of the convolving kernel
  - **dilation** (ListInt) - the spacing between kernel elements
  - **groups** (Number) - split input into groups, in_channels should be divisible by the number of groups

- constraints：

  None

- Examples：

  None

> torch_npu.npu_convolution(input, weight, bias, stride, padding, dilation, groups) -> Tensor

Applies a 2D or 3D convolution over an input image composed of several input planes.

- Parameters：
  - **input** (Tensor) - input tensor of shape(minibatch, in_channels, iH, iW) or (minibatch, in_channels, iT, iH, iW)
  - **weight** (Tensor) - filters of shape(out_channels, in_channels/groups, kH, kW) or (out_channels, in_channels/groups, kT, kH, kW)
  - **bias** (Tensor, optional) - optional bias of shape(out_channels)
  - **stride** (ListInt) - the stride of the convolving kernel
  - **padding** (ListInt) - implicit paddings on both sides of the input
  - **dilation** (ListInt) - the spacing between kernel elements
  - **groups** (ListInt) - split input into groups, in_channels should be divisible by the number of groups

- constraints：

  None

- Examples：

  None

> torch_npu.npu_conv2d(input, weight, bias, stride, padding, dilation, groups) -> Tensor

Applies a 2D convolution over an input image composed of several input planes.

- Parameters：
  - **input** (Tensor) - input tensor of shape(minibatch, in_channels, iH, iW)
  - **weight** (Tensor) - filters of shape(out_channels, in_channels/groups, kH, kW)
  - **bias** (Tensor, optional) - optional bias of shape(out_channels)
  - **stride** (ListInt) - the stride of the convolving kernel
  - **padding** (ListInt) - implicit paddings on both sides of the input
  - **dilation** (ListInt) - the spacing between kernel elements
  - **groups** (ListInt) - split input into groups, in_channels should be divisible by the number of groups

- constraints：

  None

- Examples：

  None

> torch_npu.npu_conv3d(input, weight, bias, stride, padding, dilation, groups) -> Tensor

Applies a 3D convolution over an input image composed of several input planes.

- Parameters：
  - **input** (Tensor) - input tensor of shape(minibatch, in_channels, iT, iH, iW)
  - **weight** (Tensor) - filters of shape(out_channels, in_channels/groups, kT, kH, kW)
  - **bias** (Tensor, optional) - optional bias of shape(out_channels)
  - **stride** (ListInt) - the stride of the convolving kernel
  - **padding** (ListInt) - implicit paddings on both sides of the input
  - **dilation** (ListInt) - the spacing between kernel elements
  - **groups** (ListInt) - split input into groups, in_channels should be divisible by the number of groups

- constraints：

  None

- Examples：

  None

> torch_npu.one_(self) -> Tensor

Fills self tensor with ones.

- Parameters：
  
- **self** (Tensor) - input tensor
  
- constraints：

  None

- Examples：

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

- Parameters：
  - **self** (Tensor) - the input tensor
  - **dim** (int, optional) - the dimension to sort along
  - **descending** (bool, optional) - controls the sorting order (ascending or descending)
  
- constraints：

  At present only support the last dim(-1).

- Examples：

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

- Parameters：
  - **self** (Tensor) - the input tensor
  - **acl_format** (int) - the target format to transform

- constraints：

  None

- Examples：

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

  - Parameters：
    - **self** (Tensor) - the input tensor
    - **src** (Tensor) - the target format to transform

  - constraints：

    None

  - Examples：

    ```python
    >>> x = torch.rand(2, 3, 4, 5).npu()
    >>> torch_npu.get_npu_format(x)
    0
    >>> torch_npu.get_npu_format(x.npu_format_cast_(29))
    29
    ```

> torch_npu.npu_transpose(self, perm) -> Tensor

Returns a view of the original tensor with its dimensions permuted, and make the result contiguous.

- Parameters：
  - **self** (Tensor) - the input tensor
  - **perm** (ListInt) - The desired ordering of dimensions

- constraints：

  None

- Examples：

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

- Parameters：
  - **self** (Tensor) - the input tensor
  - **perm** (ListInt) - the desired expanded size

- constraints：

  None

- Examples：

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

- Parameters：
  - **input** (Tensor) - the input tensor.
  - **dtype** (torch.dtype) - the desired data type of returned Tensor.

- constraints：

  None

- Examples：

  ```python
  >>> torch_npu.npu_dtype_cast(torch.tensor([0, 0.5, -1.]).npu(), dtype=torch.int)
  tensor([ 0,  0, -1], device='npu:0', dtype=torch.int32)
  ```

> torch_npu.empty_with_format(size, dtype, layout, device, pin_memory, acl_format) -> Tensor

Returns a tensor filled with uninitialized data. The shape of the tensor is defined by the variable argument size. The format of the tensor is defined by the variable argument acl_format.

- Parameters：

  - **size** (int...) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.

  - **dtype** (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_tensor_type()).

  - **layout** (torch.layout, optional) – the desired layout of returned Tensor. Default: None.

  - **device** (torch.device, optional) – the desired device of returned tensor. Default: None

  - **pin_memory** (bool, optional) – If set, returned tensor would be allocated in the pinned memory. Default: None.

  - **acl_format** (Number) – the desired memory format of returned Tensor. Default: 2.

- constraints：

  None

- Examples：
  ```python
  >>> torch_npu.empty_with_format((2, 3), dtype=torch.float32, device="npu")
  tensor([[1., 1., 1.],
          [1., 1., 1.]], device='npu:0')
  ```

> torch_npu.copy_memory_(dst, src, non_blocking=False) -> Tensor

Copies the elements from src into self tensor and returns self.

- Parameters：
  - **dst** (Tensor) - the source tensor to copy from.
  - **src** (Tensor) - the desired data type of returned Tensor.
  - **non_blocking** (bool) - if True and this copy is between CPU and NPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect.

- constraints：

  copy_memory_ only support npu tensor.
  input tensors of copy_memory_ should have same dtype.
  input tensors of copy_memory_ should have same device index.

- Examples：

  ```python
  >>> a=torch.IntTensor([0,  0, -1]).npu()
  >>> b=torch.IntTensor([1, 1, 1]).npu()
  >>> a.copy_memory_(b)
  tensor([1, 1, 1], device='npu:0', dtype=torch.int32)
  ```

> torch_npu.npu_one_hot(input, num_classes=-1, depth=1, on_value=1, off_value=0) -> Tensor

Returns a one-hot tensor. The locations represented by index in "x" take value "on_value", while all other locations take value "off_value". 

- Parameters：
  - **input** (Tensor) - class values of any shape.
  - **num_classes** (int) - The axis to fill. Defaults to "-1". 
  - **depth** (Number) - The depth of the one hot dimension. 
  - **on_value** (Number) - The value to fill in output when indices[j] = i.
  - **off_value** (Number) - The value to fill in output when indices[j] != i.

- constraints：

  None

- Examples：
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

- Parameters：
  - **x1** (Tensor) -  A Tensor in 5HD.
  - **x2** (Tensor) - A Tensor of the same type as "x1", and the same shape as "x1", except for the C1 value.  
  - **offset1** (Number) - A required int. Offset value of C1 in "x1". 
  - **offset2** (Number) - A required int. Offset value of C1 in "x2". 
  - **c1_len** (Number) - A required int. C1 len of "y". The value must be less than the difference between C1 and offset in "x1" and "x2". 

- constraints：

  None

- Examples：
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

- Parameters：
  - **features** (Tensor) - A Tensor.  A "batch_size * num_classes" matrix. 
  - **labels** (Tensor) - A Tensor of the same type as "features". A "batch_size * num_classes" matrix. 

- constraints：

  None

- Examples：

  None

> torch_npu.npu_ps_roi_pooling(x, rois, spatial_scale, group_size, output_dim) -> Tensor

Performs Position Sensitive PS ROI Pooling. 

- Parameters：
  - **x** (Tensor) - An NC1HWC0 tensor, describing the feature map, dimension C1 must be equal to (int(output_dim+15)/C0))*group_size*group_size. 
  - **rois** (Tensor) - A tensor with shape [batch, 5, rois_num], describing the ROIs, each ROI consists of five elements: "batch_id", "x1", "y1", "x2", and "y2", which "batch_id" indicates the index of the input feature map, "x1", "y1", "x2", or "y2" must be greater than or equal to "0.0".  
  - **spatial_scale** (Number) - A required float32, scaling factor for mapping the input coordinates to the ROI coordinates . 
  - **group_size** (Number) - A required int32, specifying the number of groups to encode position-sensitive score maps, must be within the range (0, 128). 
  - **output_dim** (Number) - A required int32, specifying the number of output channels, must be greater than 0. 

- constraints：

  None

- Examples：
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

- Parameters：
  - **features** (Tensor) -  A Tensor in 5HD.
  - **rois** (Tensor) - ROI position. A 2D Tensor with shape (N, 5). "N" indicates the number of ROIs, the value "5" indicates the indexes of images where the ROIs are located, "x0", "y0", "x1", and "y1". 
  - **spatial_scale** (Number) - A required attribute of type float32, specifying the scaling ratio of "features" to the original image. 
  - **pooled_height** (Number) - A required attribute of type int32, specifying the H dimension. 
  - **pooled_width** (Number) - A required attribute of type int32, specifying the W dimension. 
  - **sample_num** (Number) - An optional attribute of type int32, specifying the horizontal and vertical sampling frequency of each output. If this attribute is set to "0", the sampling frequency is equal to the rounded up value of "rois", which is a floating point number. Defaults to "2".
  - **roi_end_mode** (Number) - An optional attribute of type int32. Defaults to "1".

- constraints：

  None

- Examples：
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

- Parameters：
  - **boxes** (Tensor) -  A 2-D float tensor of shape [num_boxes, 4]. 
  - **scores** (Tensor) - A 1-D float tensor of shape [num_boxes] representing a single score corresponding to each box (each row of boxes). 
  - **max_output_size** (Number) - A scalar representing the maximum number of boxes to be selected by non max suppression.
  - **iou_threshold** (Tensor) - A 0-D float tensor representing the threshold for deciding whether boxes overlap too much with respect to IOU. 
  - **scores_threshold** (Tensor) -  A 0-D float tensor representing the threshold for deciding when to remove boxes based on score. 
  - **pad_to_max_output_size** (bool) - If true, the output selected_indices is padded to be of length max_output_size. Defaults to false. 

- Returns:
  - **selected_indices** - A 1-D integer tensor of shape [M] representing the selected indices from the boxes tensor, where M <= max_output_size. 
  - **valid_outputs** - A 0-D integer tensor representing the number of valid elements in selected_indices, with the valid elements appearing first. 

- constraints：

  None

- Examples：
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

- Parameters：
  - **dets** (Tensor) -  A 2-D float tensor of shape [num_boxes, 5]. 
  - **scores** (Tensor) - A 1-D float tensor of shape [num_boxes] representing a single score corresponding to each box (each row of boxes). 
  - **iou_threshold** (Number) - A scalar representing the threshold for deciding whether boxes overlap too much with respect to IOU.   
  - **scores_threshold** (Number) -  A scalar representing the threshold for deciding when to remove boxes based on score. Defaults to "0". 
  - **max_output_size** (Number) - A scalar integer tensor representing the maximum number of boxes to be selected by non max suppression. Defaults to "-1", that is, no constraint is imposed. 
  - **mode** (Number) - This parameter specifies the layout type of the dets. The default value is 0. If mode is set to 0, the input values of dets are x, y, w, h, and angle. If mode is set to 1, the input values of dets are x1, y1, x2, y2, and angle. Defaults to "0".

- Returns:
  - **selected_index** - A 1-D integer tensor of shape [M] representing the selected indices from the dets tensor, where M <= max_output_size. 
  - **selected_num** - A 0-D integer tensor representing the number of valid elements in selected_indices. 

- constraints：

  None

- Examples：
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

- Parameters：
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

- constraints：

  None

- Examples：
  
  None

>torch_npu.npu_iou(bboxes, gtboxes, mode=0) -> Tensor
>torch_npu.npu_ptiou(bboxes, gtboxes, mode=0) -> Tensor

Computes the intersection over union (iou) or the intersection over. foreground (iof) based on the ground-truth and predicted regions.

- Parameters：
  - **bboxes** (Tensor) - the input tensor.
  - **gtboxes** (Tensor) - the input tensor.
  - **mode** (Number) - 0 1 corresponds to two modes iou iof.

- constraints：

  None

- Examples：

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

- Parameters：
  - **input** (Tensor) - the input tensor.
  - **paddings** (ListInt) -  type int32 or int64.

- constraints：

  None

- Examples：

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

- Parameters：
  - **input** (Tensor) - the input tensor.
  - **iou_threshold** (Number) -  Threshold. If the value exceeds this threshold, the value is 1. Otherwise, the value is 0.

- Returns:

  - **selected_boxes** - 2-D tensor with shape of [N,5], representing filtered boxes including proposal boxes and corresponding confidence scores. 
  - **selected_idx** - 1-D tensor with shape of [N], representing the index of input proposal boxes. 
  - **selected_mask** - 1-D tensor with shape of [N], the symbol judging whether the output proposal boxes is valid . 

- constraints：

  The 2nd-dim of input box_scores must be equal to 8.

- Examples：

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

- Parameters：
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

- constraints：

  None

- Examples：

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

- Parameters：
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

- constraints：

  None

- Examples：

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

- Parameters：
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

- constraints：

  None

- Examples：
  
  None

>torch_npu.npu_random_choice_with_mask(x, count=256, seed=0, seed2=0) -> (Tensor, Tensor)

Shuffle index of no-zero element

- Parameters：
  - **x** (Tensor) - the input tensor.
  - **count** (Number) -  the count of output, if 0, out all no-zero elements.
  - **seed** (Number) -  type int32 or int64.
  - **seed2** (Number) -  type int32 or int64.

- Returns:

  - **y** - 2-D tensor, no-zero element index. 
  - **mask** - 1-D, whether the corresponding index is valid. 

- constraints：

  None

- Examples：

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

- Parameters：
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

- constraints：

  None

- Examples：

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

- Parameters：
  - **self** (Tensor) - the input tensor.
  - **offsets** (ListInt) -  type int32 or int64.
  - **size** (ListInt) -  type int32 or int64.

- constraints：

  None

- Examples：

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

- Parameters：
  - **self** (Tensor) - The input Tensor.
  - **seed** (Tensor) - The input Tensor.
  - **p** (Float) - Dropout probability.

- Returns：

  - **y**  - A tensor with the same shape and type as "x". 
  - **mask**  - A tensor with the same shape and type as "x". 
  - **new_seed**  - A tensor with the same shape and type as "seed". 

- constraints：

  None

- Examples：

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

>torch_npu._npu_dropout(self, p) -> (Tensor, Tensor)

count dropout result without seed

- Parameters：
  Similar to `torch.dropout`, optimize implemention to npu device.
  - **self** (Tensor) - The input Tensor.
  - **p** (Float) - Dropout probability.

- constraints：

  None

- Examples：

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

>torch_npu._npu_dropout_inplace(result, p) -> (Tensor(a!), Tensor)

count dropout result inplace.

- Parameters：
  Similar to `torch.dropout_`, optimize implemention to npu device.
  - **result** (Tensor) - The Tensor dropout inplace.
  - **p** (Float) - Dropout probability.

- constraints：

  None

- Examples：

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

- Parameters：
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

- constraints：

  None

- Examples：

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

- Parameters：
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

- constraints：

  None

- Examples：

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

- Parameters：
  Similar to `torch.max`, optimize implemention to npu device.
  
  - **self** (Tensor) – the input tensor.
  - **dim** (Number) – the dimension to reduce.
  - **keepdim** (bool) – whether the output tensor has dim retained or not.
  
- Returns:

  - **values** - max values in the input tensor.
  - **indices** - index of max values in the input tensor.

- constraints：

  None

- Examples：

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

- Parameters：
  Similar to `torch.min`, optimize implemention to npu device.
  - **self** (Tensor) – the input tensor.
  - **dim** (Number) – the dimension to reduce.
  - **keepdim** (bool) – whether the output tensor has dim retained or not.

- Returns:

  - **values** - min values in the input tensor.
  - **indices** - index of min values in the input tensor.

- constraints：

  None

- Examples：

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

- Parameters：
  Similar to `torch.scatter`, optimize implemention to npu device.
  
  - **self** (Tensor) - the input tensor.
  - **indices** (Tensor) – the indices of elements to scatter, can be either empty or of the same dimensionality as src. When empty, the operation returns self unchanged.
  - **updates** (Tensor) – the source element(s) to scatter.
- **dim** (Number) – the axis along which to index
  
- constraints：

  None

- Examples：

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

- Parameters：
  The same as `torch.nn.functional.layer_norm`, optimize implemention to npu device.
  - **input** (Tensor) - The input Tensor.
  - **normalized_shape** (ListInt) – input shape from an expected input of size.
  - **weight** (Tensor) - The gamma Tensor.
  - **bias** (Tensor) - The beta Tensor.
  - **eps** (Float) – The epsilon value added to the denominator for numerical stability. Default: 1e-5.

- constraints：

  None

- Examples：

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

- Parameters：
  
  - **self** (Tensor) - Any Tensor
  
- constraints：

  None

- Examples：

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

- Parameters：
  
  - **self** (Tensor) -  A Tensor of data memory address. Must be float32 .
  
- Constraints：

  None

- Examples：
  
  ```python
  >>> x = torch.rand(2).npu()
  >>> torch_npu.npu_get_float_status(x)
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
  ```

> torch_npu.npu_clear_float_status(self) -> Tensor

Set the value of address 0x40000 to 0 in each core.

- Parameters：
  
  - **self** (Tensor) -  A tensor of type float32.
  
- Constraints：

  None

- Examples：

  ```python
  >>> x = torch.rand(2).npu()
  >>> torch_npu.npu_clear_float_status(x)
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
  ```

> torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first) -> Tensor

Confuse reshape and transpose.

- Parameters：
  
  - **self** (Tensor) -  A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
  - **perm** (ListInt) -  A permutation of the dimensions of "x".
  - **shape** (ListInt) -  The shape of the input.
  - **transpose_first** (bool) -  If True, the transpose is first, otherwise the reshape is first.
  
- Constraints：

  None

- Examples：

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

- Parameters：
  - **self** (Tensor) -  A matrix Tensor. Must be one of the following types: float16, float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ]. 
  - **mat2** (Tensor) -  A matrix Tensor. Must be one of the following types: float16, float32, int32. 2D or higher. Has format [ND, NHWC, FRACTAL_NZ]. 
  - **output_sizes** (ListInt) - Output's shape, used in matmul's backpropagation, default [].
  
- Constraints：

  None

- Examples：

  ```python
  >>> mat1 = torch.randn(10, 3, 4).npu()
  >>> mat2 = torch.randn(10, 4, 5).npu()
  >>> res = torch_npu.npu_bmmV2(mat1, mat2, [])
  >>> res.shape
  torch.Size([10, 3, 5])
  ```

> torch_npu.fast_gelu(self) -> Tensor

Computes the gradient for the fast_gelu of "x" . 

- Parameters：
  
  - **self** (Tensor) -  A Tensor. Must be one of the following types: float16, float32
  
- Constraints：

  None

- Examples：

  ```python
  >>> x = torch.rand(2).npu()
  >>> x
  tensor([0.5991, 0.4094], device='npu:0')
  >>> torch_npu.fast_gelu(x)
  tensor([0.4403, 0.2733], device='npu:0')
  ```

> torch_npu.npu_sub_sample(self, per_images, positive_fraction) -> Tensor

Randomly sample a subset of positive and negative examples,and overwrite the label vector to the ignore value (-1) for all elements that are not included in the sample.

- Parameters：

  - **self** (Tensor) -  shape of labels,(N, ) label vector with values.
  - **per_images** (Number) -  A require attribute of type int.
  - **positive_fraction** (Float) -  A require attribute of type float.

- Constraints：

  None

- Examples：

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

- Parameters：

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

- Constraints：

  None

- Examples：

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

- Parameters：

  - **self** (Tensor) -  A Tensor. Must be one of the following types: float16, float32.
  
- Constraints：

  None

- Examples：

  ```python
  >>> x = torch.rand(10, 30, 10).npu()
  >>> y = torch_npu.npu_mish(x)
  >>> y.shape
  torch.Size([10, 30, 10])
  ```
  
> torch_npu.npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors) -> Tensor

Generate the responsible flags of anchor in a single feature map. 

- Parameters：
  - **self** (Tensor) -  Ground truth box, 2-D Tensor with shape [batch, 4].
  - **featmap_size** (ListInt) -  The size of feature maps, listint. 
  - **strides** (ListInt) -  Stride of current level, listint. 
  - **num_base_anchors** (Number) -  The number of base anchors. 

- Constraints：

  None

- Examples：

  ```python
  >>> x = torch.rand(100, 4).npu()
  >>> y = torch_npu.npu_anchor_response_flags(x, [60, 60], [2, 2], 9)
  >>> y.shape
  torch.Size([32400])
  ```
  
> torch_npu.npu_yolo_boxes_encode(self, gt_bboxes, stride, performance_mode=False) -> Tensor

Generates bounding boxes based on yolo's "anchor" and "ground-truth" boxes. It is a customized mmdetection operator. 

- Parameters：
  - **self** (Tensor) -  anchor boxes generated by the yolo training set. A 2D Tensor of type float32 or float16 with shape (N, 4). "N" indicates the number of ROIs, "N" indicates the number of ROIs, and the value "4" refers to (tx, ty, tw, th). 
  - **gt_bboxes** (Tensor) -  target of the transformation, e.g, ground-truth boxes. A 2D Tensor of type float32 or float16 with shape (N, 4). "N" indicates the number of ROIs, and 4 indicates "dx", "dy", "dw", and "dh". 
  - **strides** (Tensor) -  Scale for each box. A 1D Tensor of type int32 shape (N,). "N" indicates the number of ROIs. 
- **performance_mode** (bool) - Select performance mode, "high_precision" or "high_performance". select "high_precision" when input type is float32, the output tensor precision will be smaller than 0.0001, select "high_performance" when input type is float32, the ops will be best performance, but precision will be only smaller than 0.005. 
  
- Constraints：

  input anchor boxes only support maximum N=20480. 

- Examples：

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

- Parameters：
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

- Constraints：

  None

- Examples：

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

- Parameters：

  - **self** (Tensor) - A Tensor. Support float32. shape (n, c, d). 
  - **seq_len** (Tensor) - A Tensor. Each batch normalize data num. Support Int32. Shape (n, ). 
  - **normalize_type** (Number) - Str. Support "per_feature" or "all_features". 

- constraints：

  None

- Examples：
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

- Parameters：

  - **self** (Tensor) - input tensor. A ND Tensor of float32/float16/int32/int8 with shapes  1-D (D,), 2-D(N, D), 3-D(N, C, D).
  - **start** (Tensor) - masked fill start pos. A 3D Tensor of int32 with shape (num, N).
  - **end** (Tensor) - masked fill end pos. A 3D Tensor of int32 with shape (num, N).
  - **value** (Tensor) - masked fill value. A 2D Tensor of float32/float16/int32/int8 with shape (num,).
  - **axis** (Number) - axis with masked fill of int32. Defaults to -1. 

- constraints：

  None

- Examples：
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

- Parameters：

  -  **input** (Tensor) - A matrix Tensor. 2D. Must be one of the following types: float32, float16, int32, int8. Has format [ND, NHWC, FRACTAL_NZ]. 
  -  **weight** (Tensor) - A matrix Tensor. 2D. Must be one of the following types: float32, float16, int32, int8. Has format [ND, NHWC, FRACTAL_NZ]. 
  -  **bias** (Tensor) - A 1D Tensor. Must be one of the following types: float32, float16, int32. Has format [ND, NHWC]. 

- constraints：

  None

- Examples：
  ```python
  >>> x=torch.rand(2,16).npu()
  >>> w=torch.rand(4,16).npu()
  >>> b=torch.rand(4).npu()
  >>> output = torch_npu.npu_linear(x, w, b)
  >>> output
  tensor([[3.6335, 4.3713, 2.4440, 2.0081],
          [5.3273, 6.3089, 3.9601, 3.2410]], device='npu:0')
  ```

> torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size=None, adam_mode=0, *, out=（var,m,v）)

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

  - **out** :A Tensor, optional. The output tensor. 

- constraints:

  None

- Examples：
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

- Parameters：

  - **self** (Tensor) - Bounding boxes, a 2D Tensor of type float16 or float32 with shape (N, 4). "N" indicates the number of bounding boxes, and the value "4" refers to [x1, y1, x2, y2] or [x, y, w, h].
  - **gtboxes** (Tensor) - Ground-truth boxes, a 2D Tensor of type float16 or float32 with shape (M, 4). "M" indicates the number of ground truth boxes, and the value "4" refers to [x1, y1, x2, y2] or [x, y, w, h].
  - **trans** (bool) - An optional bool, true for 'xywh', false for 'xyxy'. 
  - **is_cross** (bool) - An optional bool, control whether the output shape is [M, N] or [1, N]. 
  - **mode:** (Number) - Computation mode, a character string with the value range of [iou, iof] . 

- constraints：

  None

- Examples：
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

Computes the for the Swish of "x" .

- Parameters：

  - **self** (Tensor) - A Tensor. Must be one of the following types: float16, float32 

- constraints：

  None

- Examples：
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

- Parameters：

  - **self** (Tensor) - A Tensor.
  - **shape** (ListInt) - Defines the shape of the output tensor. 
  - **can_refresh** (bool) - Used to specify whether reshape can be refreshed in place.

- constraints：

   This operator cannot be directly called by the acllopExecute API. 

- Examples：
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

- Parameters：

  - **self** (Tensor) - data of grad increment, a 3D Tensor of type float32 with shape (B, 5, N). 
  - **query_boxes** (Tensor) - Bounding boxes, a 3D Tensor of type float32 with shape (B, 5, K).
  - **trans** (bool) - An optional attr, true for 'xyxyt', false for 'xywht'. 

- constraints：

  None

- Examples：
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

- Parameters：

  - **self** (Tensor) - data of grad increment, a 3D Tensor of type float32 with shape (B, 5, N). 
  - **query_boxes** (Tensor) - Bounding boxes, a 3D Tensor of type float32 with shape (B, 5, K).
  - **trans** (bool) - An optional attr, true for 'xyxyt', false for 'xywht'. 
  - **is_cross** (bool) -Cross calculation when it is True, and one-to-one calculation when it is False.
  - **mode** (Number) - Computation mode, a character string with the value range of [iou, iof, giou] . 

- constraints：

  None

- Examples：
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

- Parameters：

  - anchor_box (Tensor) -  A 3D Tensor with shape (B, 5, N). the input tensor.Anchor boxes. "B" indicates the number of batch size, "N" indicates the number of bounding boxes, and the value "5" refers to "x0", "x1", "y0", "y1" and "angle" .
  - gt_bboxes (Tensor) - A 3D Tensor of float32 (float16) with shape (B, 5, N).   
  - weight (Tensor) - A float list for "x0", "x1", "y0", "y1" and "angle", defaults to [1.0, 1.0, 1.0, 1.0, 1.0].

- constraints：

  None

- Examples：

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

  - Parameters：

    - anchor_box (Tensor) -  A 3D Tensor with shape (B, 5, N). the input tensor.Anchor boxes. "B" indicates the number of batch size, "N" indicates the number of bounding boxes, and the value "5" refers to "x0", "x1", "y0", "y1" and "angle" .
    - deltas (Tensor) - A 3D Tensor of float32 (float16) with shape (B, 5, N).   
    - weight (Tensor) - A float list for "x0", "x1", "y0", "y1" and "angle", defaults to [1.0, 1.0, 1.0, 1.0, 1.0].

  - constraints：

    None

  - Examples：

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

- Args：

  - boxes1 (Tensor): Predicted bboxes of format xywh, shape (4, n).
  - boxes2 (Tensor): Corresponding gt bboxes, shape (4, n).
  - trans (Bool): Whether there is an offset
  - is_cross (Bool): Whether there is a cross operation between box1 and box2.
  - mode (int):  Select the calculation mode of diou.
  - atan_sub_flag (Bool): whether to pass the second value of the forward to the reverse.

- Returns：

  torch.Tensor: The result of the mask operation

- Examples：

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

- Args：

  - boxes1 (Tensor): Predicted bboxes of format xywh, shape (4, n).
  - boxes2 (Tensor): Corresponding gt bboxes, shape (4, n).
  - trans (Bool): Whether there is an offset
  - is_cross (Bool): Whether there is a cross operation between box1 and box2.
  - mode (int):  Select the calculation mode of diou.

- Returns：

  torch.Tensor: The result of the mask operation

- Examples：

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

  - Args：

    - x(Tensor) - A floats Tensor in 1D.
    - size(Number) - A required int. First dimension of output tensor when reshaping.

    - constraints：

      Size needs to be divisible by output of packing floats. If size of x is divisible by 8, size of output is (size of x) / 8;
      otherwise, size of output is (size of x // 8) + 1, -1 float values will be added to fill divisibility, at little endian positions.
      910 and 710 chips support input type float32 and float16, 310 chips only supports input type float16.

  - Examples：

    ```
        >>>a = torch.tensor([5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2],dtype=torch.float32).npu()
        >>>b = torch_npu.sign_bits_pack(a, 2)
        >>>b
        >>>tensor([[159],[15]], device='npu:0')
        >>>(binary form of 159 is ob10011111, corresponds to 4, -2, -1, 0, 2, 3, 4, 5 respectively)
    ```


  >    torch_npu.sign_bits_unpack(x, dtype, size) -> Tensor

  one-bit Adam unpack of uint8 into float.

  - Args：

    - x(Tensor) - A uint8 Tensor in 1D.
    - dtype(Number) - A required int. 1 sets float16 as output, 0 sets float32 as output.
    - size(Number) - A required int. First dimension of output tensor when reshaping.

  - constraints：

    Size needs to be divisible by output of unpacking uint8s. Size of output is (size of x) * 8;

  - Examples：

    ```
        >>>a = torch.tensor([159, 15], dtype=torch.uint8).npu()
        >>>b = torch_npu.sign_bits_unpack(a, 0, 2)
        >>>b
        >>>tensor([[1., 1., 1., 1., 1., -1., -1., 1.],
        >>>[1., 1., 1., 1., -1., -1., -1., -1.]], device='npu:0')
    (binary form of 159 is ob00001111)
    ```

    

## 亲和库

以下亲和库适用于PyTorch 1.8.1版本。

>   **def fuse_add_softmax_dropout**(training, dropout, attn_mask, attn_scores, attn_head_size, p=0.5, dim=-1):

Using NPU custom operator to replace the native writing method to improve performance

- Args：

  - training (bool): Whether it is training mode.
  - dropout (nn.Module): the dropout layer
  - attn_mask (Tensor): the attention mask.
  - attn_scores (Tensor): the raw attention scores
  - attn_head_size (float): the head size
  - p (float): probability of an element to be zeroed
  - dim (int): A dimension along which softmax will be computed.

- Returns：

  torch.Tensor: The result of the mask operation

- Examples：

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

- Args：

  - boxes1 (Tensor): Predicted bboxes of format xywh, shape (4, n).
  - boxes2 (Tensor): Corresponding gt bboxes, shape (4, n).
  - trans (Bool): Whether there is an offset
  - is_cross (Bool): Whether there is a cross operation between box1 and box2.
  -  mode (int):  Select the calculation mode of diou.

- Returns：

  torch.Tensor: The result of the mask operation

- Examples：

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

- Args：

  - boxes1 (Tensor): Predicted bboxes of format xywh, shape (4, n).
  - boxes2 (Tensor): Corresponding gt bboxes, shape (4, n).
  - trans (Bool): Whether there is an offset
  - is_cross (Bool): Whether there is a cross operation between box1 and box2.
  -  mode (int):  Select the calculation mode of diou.
  - atan_sub_flag (Bool): whether to pass the second value of the forward to the reverse.

- Returns：

  torch.Tensor: The result of the mask operation

- Examples：

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

- Args：
  -  p (float): probability of an element to be zeroed.
  - module_name (string): the name of the model

>   **class** **MultiheadAttention**(nn.Module):

Multi-headed attention.

- Args：

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
  
    
