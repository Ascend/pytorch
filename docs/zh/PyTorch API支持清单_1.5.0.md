## [Tensors](https://pytorch.org/docs/1.5.0/torch.html)

| 序号 | API名称                       | PyTorch1.5.0支持情况                   |
| ---- | ----------------------------- | -------------------------------------- |
| 1    | torch.is_tensor               | 是                                     |
| 2    | torch.is_storage              | 是                                     |
| 3    | torch.is_complex              | 是，支持判断，但当前硬件限制不支持复数 |
| 4    | torch.is_floating_point       | 是                                     |
| 5    | torch.set_default_dtype       | 是                                     |
| 6    | torch.get_default_dtype       | 是                                     |
| 7    | torch.set_default_tensor_type | 是                                     |
| 8    | torch.numel                   | 是                                     |
| 9    | torch.set_printoptions        | 是                                     |
| 10   | torch.set_flush_denormal      | 是                                     |
| 11   | torch.tensor                  | 是                                     |
| 12   | torch.sparse_coo_tensor       | 否                                     |
| 13   | torch.as_tensor               | 是                                     |
| 14   | torch.as_strided              | 是                                     |
| 15   | torch.from_numpy              | 是                                     |
| 16   | torch.zeros                   | 是                                     |
| 17   | torch.zeros_like              | 是                                     |
| 18   | torch.ones                    | 是                                     |
| 19   | torch.ones_like               | 是                                     |
| 20   | torch.arange                  | 是                                     |
| 21   | torch.range                   | 是                                     |
| 22   | torch.linspace                | 是                                     |
| 23   | torch.logspace                | 是                                     |
| 24   | torch.eye                     | 是                                     |
| 25   | torch.empty                   | 是                                     |
| 26   | torch.empty_like              | 是                                     |
| 27   | torch.empty_strided           | 是                                     |
| 28   | torch.full                    | 是                                     |
| 29   | torch.full_like               | 是                                     |
| 30   | torch.quantize_per_tensor     | 是                                     |
| 31   | torch.quantize_per_channel    | 是                                     |
| 32   | torch.cat                     | 是                                     |
| 33   | torch.chunk                   | 是                                     |
| 34   | torch.gather                  | 是                                     |
| 35   | torch.index_select            | 是                                     |
| 36   | torch.masked_select           | 是                                     |
| 37   | torch.narrow                  | 是                                     |
| 38   | torch.nonzero                 | 是                                     |
| 39   | torch.reshape                 | 是                                     |
| 40   | torch.split                   | 是                                     |
| 41   | torch.squeeze                 | 是                                     |
| 42   | torch.stack                   | 是                                     |
| 43   | torch.t                       | 是                                     |
| 44   | torch.take                    | 是                                     |
| 45   | torch.transpose               | 是                                     |
| 46   | torch.unbind                  | 是                                     |
| 47   | torch.unsqueeze               | 是                                     |
| 48   | torch.where                   | 是                                     |

## Generators

| 序号 | API名称                         | 是否支持（PyTorch1.5.0） |
| ---- | ------------------------------- | ------------------------ |
| 1    | torch._C.Generator              | 是                       |
| 2    | torch._C.Generator.device       | 是                       |
| 3    | torch._C.Generator.get_state    | 否                       |
| 4    | torch._C.Generator.initial_seed | 是                       |
| 5    | torch._C.Generator.manual_seed  | 是                       |
| 6    | torch._C.Generator.seed         | 是                       |
| 7    | torch._C.Generator.set_state    | 否                       |

## Random sampling

| 序号 | API名称                                    | 是否支持（PyTorch1.5.0） |
| ---- | ------------------------------------------ | ------------------------ |
| 1    | torch.seed                                 | 是                       |
| 2    | torch.manual_seed                          | 是                       |
| 3    | torch.initial_seed                         | 是                       |
| 4    | torch.get_rng_state                        | 是                       |
| 5    | torch.set_rng_state                        | 是                       |
| 6    | torch.torch.default_generator              | 是                       |
| 7    | torch.bernoulli                            | 是                       |
| 8    | torch.multinomial                          | 是                       |
| 9    | torch.normal                               | 是                       |
| 10   | torch.poisson                              | 否                       |
| 11   | torch.rand                                 | 是                       |
| 12   | torch.rand_like                            | 是                       |
| 13   | torch.randint                              | 是                       |
| 14   | torch.randint_like                         | 是                       |
| 15   | torch.randn                                | 是                       |
| 16   | torch.randn_like                           | 是                       |
| 17   | torch.randperm                             | 是                       |
| 18   | torch.Tensor.bernoulli_()                  | 是                       |
| 19   | torch.Tensor.bernoulli_()                  | 是                       |
| 20   | torch.Tensor.exponential_()                | 否                       |
| 21   | torch.Tensor.geometric_()                  | 否                       |
| 22   | torch.Tensor.log_normal_()                 | 否                       |
| 23   | torch.Tensor.normal_()                     | 是                       |
| 24   | torch.Tensor.random_()                     | 是                       |
| 25   | torch.Tensor.uniform_()                    | 是                       |
| 26   | torch.quasirandom.SobolEngine              | 是                       |
| 27   | torch.quasirandom.SobolEngine.draw         | 是                       |
| 28   | torch.quasirandom.SobolEngine.fast_forward | 是                       |
| 29   | torch.quasirandom.SobolEngine.reset        | 是                       |

## Serialization

| 序号 | API名称    | 是否支持（PyTorch1.5.0） |
| ---- | ---------- | ------------------------ |
| 1    | torch.save | 是                       |
| 2    | torch.load | 是                       |

## Math operations

| 序号 | API名称                  | 是否支持（PyTorch1.5.0） |
| ---- | ------------------------ | ------------------------ |
| 1    | torch.abs                | 是                       |
| 2    | torch.acos               | 是                       |
| 3    | torch.add                | 是                       |
| 4    | torch.addcdiv            | 是                       |
| 5    | torch.addcmul            | 是                       |
| 6    | torch.angle              | 否                       |
| 7    | torch.asin               | 是                       |
| 8    | torch.atan               | 是                       |
| 9    | torch.atan2              | 是                       |
| 10   | torch.bitwise_not        | 是                       |
| 11   | torch.bitwise_and        | 是                       |
| 12   | torch.bitwise_or         | 是                       |
| 13   | torch.bitwise_xor        | 是                       |
| 14   | torch.ceil               | 是                       |
| 15   | torch.clamp              | 是                       |
| 16   | torch.conj               | 否                       |
| 17   | torch.cos                | 是                       |
| 18   | torch.cosh               | 是                       |
| 19   | torch.div                | 是                       |
| 20   | torch.digamma            | 否                       |
| 21   | torch.erf                | 是                       |
| 22   | torch.erfc               | 是                       |
| 23   | torch.erfinv             | 是                       |
| 24   | torch.exp                | 是                       |
| 25   | torch.expm1              | 是                       |
| 26   | torch.floor              | 是                       |
| 27   | torch.floor_divide       | 是                       |
| 28   | torch.fmod               | 是                       |
| 29   | torch.frac               | 是                       |
| 30   | torch.imag               | 否                       |
| 31   | torch.lerp               | 是                       |
| 32   | torch.lgamma             | 否                       |
| 33   | torch.log                | 是                       |
| 34   | torch.log10              | 是                       |
| 35   | torch.log1p              | 是                       |
| 36   | torch.log2               | 是                       |
| 37   | torch.logical_and        | 是                       |
| 38   | torch.logical_not        | 是                       |
| 39   | torch.logical_or         | 是                       |
| 40   | torch.logical_xor        | 是                       |
| 41   | torch.mul                | 是                       |
| 42   | torch.mvlgamma           | 否                       |
| 43   | torch.neg                | 是                       |
| 44   | torch.polygamma          | 否                       |
| 45   | torch.pow                | 是                       |
| 46   | torch.real               | 是                       |
| 47   | torch.reciprocal         | 是                       |
| 48   | torch.remainder          | 是                       |
| 49   | torch.round              | 是                       |
| 50   | torch.rsqrt              | 是                       |
| 51   | torch.sigmoid            | 是                       |
| 52   | torch.sign               | 是                       |
| 53   | torch.sin                | 是                       |
| 54   | torch.sinh               | 是                       |
| 55   | torch.sqrt               | 是                       |
| 56   | torch.square             | 是                       |
| 57   | torch.tan                | 是                       |
| 58   | torch.tanh               | 是                       |
| 59   | torch.true_divide        | 是                       |
| 60   | torch.trunc              | 是                       |
| 61   | torch.argmax             | 是                       |
| 62   | torch.argmin             | 是                       |
| 63   | torch.dist               | 是                       |
| 64   | torch.logsumexp          | 是                       |
| 65   | torch.mean               | 是                       |
| 66   | torch.median             | 是                       |
| 67   | torch.mode               | 否                       |
| 68   | torch.norm               | 是                       |
| 69   | torch.prod               | 是                       |
| 70   | torch.std                | 是                       |
| 71   | torch.std_mean           | 是                       |
| 72   | torch.sum                | 是                       |
| 73   | torch.unique             | 是                       |
| 74   | torch.unique_consecutive | 否                       |
| 75   | torch.var                | 否                       |
| 76   | torch.var_mean           | 否                       |
| 77   | torch.allclose           | 是                       |
| 78   | torch.argsort            | 是                       |
| 79   | torch.eq                 | 是                       |
| 80   | torch.equal              | 是                       |
| 81   | torch.ge                 | 是                       |
| 82   | torch.gt                 | 是                       |
| 83   | torch.isfinite           | 是                       |
| 84   | torch.isinf              | 是                       |
| 85   | torch.isnan              | 是                       |
| 86   | torch.kthvalue           | 是                       |
| 87   | torch.le                 | 是                       |
| 88   | torch.lt                 | 是                       |
| 89   | torch.max                | 是                       |
| 90   | torch.min                | 是                       |
| 91   | torch.ne                 | 是                       |
| 92   | torch.sort               | 是                       |
| 93   | torch.topk               | 是                       |
| 94   | torch.fft                | 否                       |
| 95   | torch.ifft               | 否                       |
| 96   | torch.rfft               | 否                       |
| 97   | torch.irfft              | 否                       |
| 98   | torch.stft               | 否                       |
| 99   | torch.bartlett_window    | 是                       |
| 100  | torch.blackman_window    | 是                       |
| 101  | torch.hamming_window     | 是                       |
| 102  | torch.hann_window        | 是                       |
| 103  | torch.bincount           | 是                       |
| 104  | torch.broadcast_tensors  | 是                       |
| 105  | torch.cartesian_prod     | 是                       |
| 106  | torch.cdist              | 是                       |
| 107  | torch.combinations       | 否                       |
| 108  | torch.cross              | 是                       |
| 109  | torch.cummax             | 是                       |
| 110  | torch.cummin             | 是                       |
| 111  | torch.cumprod            | 是                       |
| 112  | torch.cumsum             | 是                       |
| 113  | torch.diag               | 是                       |
| 114  | torch.diag_embed         | 是                       |
| 115  | torch.diagflat           | 是                       |
| 116  | torch.diagonal           | 是                       |
| 117  | torch.einsum             | 是                       |
| 118  | torch.flatten            | 是                       |
| 119  | torch.flip               | 是                       |
| 120  | torch.rot90              | 是                       |
| 121  | torch.histc              | 否                       |
| 122  | torch.meshgrid           | 是                       |
| 123  | torch.renorm             | 是                       |
| 124  | torch.repeat_interleave  | 是                       |
| 125  | torch.roll               | 是                       |
| 126  | torch.tensordot          | 是                       |
| 127  | torch.trace              | 否                       |
| 128  | torch.tril               | 是                       |
| 129  | torch.tril_indices       | 是                       |
| 130  | torch.triu               | 是                       |
| 131  | torch.triu_indices       | 是                       |
| 132  | torch.addbmm             | 是                       |
| 133  | torch.addmm              | 是                       |
| 134  | torch.addmv              | 是                       |
| 135  | torch.addr               | 是                       |
| 136  | torch.baddbmm            | 是                       |
| 137  | torch.bmm                | 是                       |
| 138  | torch.chain_matmul       | 是                       |
| 139  | torch.cholesky           | 否                       |
| 140  | torch.cholesky_inverse   | 否                       |
| 141  | torch.cholesky_solve     | 否                       |
| 142  | torch.dot                | 是                       |
| 143  | torch.eig                | 否                       |
| 144  | torch.geqrf              | 否                       |
| 145  | torch.ger                | 是                       |
| 146  | torch.inverse            | 是                       |
| 147  | torch.det                | 否                       |
| 148  | torch.logdet             | 否                       |
| 149  | torch.slogdet            | 是                       |
| 150  | torch.lstsq              | 否                       |
| 151  | torch.lu                 | 否                       |
| 152  | torch.lu_solve           | 否                       |
| 153  | torch.lu_unpack          | 否                       |
| 154  | torch.matmul             | 是                       |
| 155  | torch.matrix_power       | 是                       |
| 156  | torch.matrix_rank        | 是                       |
| 157  | torch.mm                 | 是                       |
| 158  | torch.mv                 | 是                       |
| 159  | torch.orgqr              | 否                       |
| 160  | torch.ormqr              | 否                       |
| 161  | torch.pinverse           | 是                       |
| 162  | torch.qr                 | 是                       |
| 163  | torch.solve              | 否                       |
| 164  | torch.svd                | 是                       |
| 165  | torch.svd_lowrank        | 是                       |
| 166  | torch.pca_lowrank        | 是                       |
| 167  | torch.symeig             | 是                       |
| 168  | torch.lobpcg             | 否                       |
| 169  | torch.trapz              | 是                       |
| 170  | torch.triangular_solve   | 是                       |

## Utilities

| 序号 | API名称                       | 是否支持（PyTorch1.5.0） |
| ---- | ----------------------------- | ------------------------ |
| 1    | torch.compiled_with_cxx11_abi | 是                       |
| 2    | torch.result_type             | 是                       |
| 3    | torch.can_cast                | 是                       |
| 4    | torch.promote_types           | 是                       |

## Other

| 序号 | API名称                       | 是否支持（PyTorch1.5.0） |
| ---- | ----------------------------- | ------------------------ |
| 1    | torch.no_grad                 | 是                       |
| 2    | torch.enable_grad             | 是                       |
| 3    | torch.set_grad_enabled        | 是                       |
| 4    | torch.get_num_threads         | 是                       |
| 5    | torch.set_num_threads         | 是                       |
| 6    | torch.get_num_interop_threads | 是                       |
| 7    | torch.set_num_interop_threads | 是                       |

## torch.Tensor

| 序号 | API名称                                | 是否支持（PyTorch1.5.0） |
| ---- | -------------------------------------- | ------------------------ |
| 1    | torch.Tensor                           | 是                       |
| 2    | torch.Tensor.new_tensor                | 是                       |
| 3    | torch.Tensor.new_full                  | 是                       |
| 4    | torch.Tensor.new_empty                 | 是                       |
| 5    | torch.Tensor.new_ones                  | 是                       |
| 6    | torch.Tensor.new_zeros                 | 是                       |
| 7    | torch.Tensor.is_cuda                   | 是                       |
| 8    | torch.Tensor.is_quantized              | 是                       |
| 9    | torch.Tensor.device                    | 是                       |
| 10   | torch.Tensor.ndim                      | 是                       |
| 11   | torch.Tensor.T                         | 是                       |
| 12   | torch.Tensor.abs                       | 是                       |
| 13   | torch.Tensor.abs_                      | 是                       |
| 14   | torch.Tensor.acos                      | 是                       |
| 15   | torch.Tensor.acos_                     | 是                       |
| 16   | torch.Tensor.add                       | 是                       |
| 17   | torch.Tensor.add_                      | 是                       |
| 18   | torch.Tensor.addbmm                    | 是                       |
| 19   | torch.Tensor.addbmm_                   | 是                       |
| 20   | torch.Tensor.addcdiv                   | 是                       |
| 21   | torch.Tensor.addcdiv_                  | 是                       |
| 22   | torch.Tensor.addcmul                   | 是                       |
| 23   | torch.Tensor.addcmul_                  | 是                       |
| 24   | torch.Tensor.addmm                     | 是                       |
| 25   | torch.Tensor.addmm_                    | 是                       |
| 26   | torch.Tensor.addmv                     | 是                       |
| 27   | torch.Tensor.addmv_                    | 是                       |
| 28   | torch.Tensor.addr                      | 是                       |
| 29   | torch.Tensor.addr_                     | 是                       |
| 30   | torch.Tensor.allclose                  | 是                       |
| 31   | torch.Tensor.angle                     | 否                       |
| 32   | torch.Tensor.apply_                    | 否                       |
| 33   | torch.Tensor.argmax                    | 是                       |
| 34   | torch.Tensor.argmin                    | 是                       |
| 35   | torch.Tensor.argsort                   | 是                       |
| 36   | torch.Tensor.asin                      | 是                       |
| 37   | torch.Tensor.asin_                     | 是                       |
| 38   | torch.Tensor.as_strided                | 是                       |
| 39   | torch.Tensor.atan                      | 是                       |
| 40   | torch.Tensor.atan2                     | 是                       |
| 41   | torch.Tensor.atan2_                    | 是                       |
| 42   | torch.Tensor.atan_                     | 是                       |
| 43   | torch.Tensor.baddbmm                   | 是                       |
| 44   | torch.Tensor.baddbmm_                  | 是                       |
| 45   | torch.Tensor.bernoulli                 | 是                       |
| 46   | torch.Tensor.bernoulli_                | 是                       |
| 47   | torch.Tensor.bfloat16                  | 否                       |
| 48   | torch.Tensor.bincount                  | 是                       |
| 49   | torch.Tensor.bitwise_not               | 是                       |
| 50   | torch.Tensor.bitwise_not_              | 是                       |
| 51   | torch.Tensor.bitwise_and               | 是                       |
| 52   | torch.Tensor.bitwise_and_              | 是                       |
| 53   | torch.Tensor.bitwise_or                | 是                       |
| 54   | torch.Tensor.bitwise_or_               | 是                       |
| 55   | torch.Tensor.bitwise_xor               | 是                       |
| 56   | torch.Tensor.bitwise_xor_              | 是                       |
| 57   | torch.Tensor.bmm                       | 是                       |
| 58   | torch.Tensor.bool                      | 是                       |
| 59   | torch.Tensor.byte                      | 是                       |
| 60   | torch.Tensor.cauchy_                   | 否                       |
| 61   | torch.Tensor.ceil                      | 是                       |
| 62   | torch.Tensor.ceil_                     | 是                       |
| 63   | torch.Tensor.char                      | 是                       |
| 64   | torch.Tensor.cholesky                  | 否                       |
| 65   | torch.Tensor.cholesky_inverse          | 否                       |
| 66   | torch.Tensor.cholesky_solve            | 否                       |
| 67   | torch.Tensor.chunk                     | 是                       |
| 68   | torch.Tensor.clamp                     | 是                       |
| 69   | torch.Tensor.clamp_                    | 是                       |
| 70   | torch.Tensor.clone                     | 是                       |
| 71   | torch.Tensor.contiguous                | 是                       |
| 72   | torch.Tensor.copy_                     | 是                       |
| 73   | torch.Tensor.conj                      | 否                       |
| 74   | torch.Tensor.cos                       | 是                       |
| 75   | torch.Tensor.cos_                      | 是                       |
| 76   | torch.Tensor.cosh                      | 是                       |
| 77   | torch.Tensor.cosh_                     | 是                       |
| 78   | torch.Tensor.cpu                       | 是                       |
| 79   | torch.Tensor.cross                     | 是                       |
| 80   | torch.Tensor.cuda                      | 否                       |
| 81   | torch.Tensor.cummax                    | 是                       |
| 82   | torch.Tensor.cummin                    | 是                       |
| 83   | torch.Tensor.cumprod                   | 是                       |
| 84   | torch.Tensor.cumsum                    | 是                       |
| 85   | torch.Tensor.data_ptr                  | 是                       |
| 86   | torch.Tensor.dequantize                | 否                       |
| 87   | torch.Tensor.det                       | 否                       |
| 88   | torch.Tensor.dense_dim                 | 否                       |
| 89   | torch.Tensor.diag                      | 是                       |
| 90   | torch.Tensor.diag_embed                | 是                       |
| 91   | torch.Tensor.diagflat                  | 是                       |
| 92   | torch.Tensor.diagonal                  | 是                       |
| 93   | torch.Tensor.fill_diagonal_            | 是                       |
| 94   | torch.Tensor.digamma                   | 否                       |
| 95   | torch.Tensor.digamma_                  | 否                       |
| 96   | torch.Tensor.dim                       | 是                       |
| 97   | torch.Tensor.dist                      | 是                       |
| 98   | torch.Tensor.div                       | 是                       |
| 99   | torch.Tensor.div_                      | 是                       |
| 100  | torch.Tensor.dot                       | 是                       |
| 101  | torch.Tensor.double                    | 否                       |
| 102  | torch.Tensor.eig                       | 否                       |
| 103  | torch.Tensor.element_size              | 是                       |
| 104  | torch.Tensor.eq                        | 是                       |
| 105  | torch.Tensor.eq_                       | 是                       |
| 106  | torch.Tensor.equal                     | 是                       |
| 107  | torch.Tensor.erf                       | 是                       |
| 108  | torch.Tensor.erf_                      | 是                       |
| 109  | torch.Tensor.erfc                      | 是                       |
| 110  | torch.Tensor.erfc_                     | 是                       |
| 111  | torch.Tensor.erfinv                    | 是                       |
| 112  | torch.Tensor.erfinv_                   | 是                       |
| 113  | torch.Tensor.exp                       | 是                       |
| 114  | torch.Tensor.exp_                      | 是                       |
| 115  | torch.Tensor.expm1                     | 是                       |
| 116  | torch.Tensor.expm1_                    | 是                       |
| 117  | torch.Tensor.expand                    | 是                       |
| 118  | torch.Tensor.expand_as                 | 是                       |
| 119  | torch.Tensor.exponential_              | 否                       |
| 120  | torch.Tensor.fft                       | 否                       |
| 121  | torch.Tensor.fill_                     | 是                       |
| 122  | torch.Tensor.flatten                   | 是                       |
| 123  | torch.Tensor.flip                      | 是                       |
| 124  | torch.Tensor.float                     | 是                       |
| 125  | torch.Tensor.floor                     | 是                       |
| 126  | torch.Tensor.floor_                    | 是                       |
| 127  | torch.Tensor.floor_divide              | 是                       |
| 128  | torch.Tensor.floor_divide_             | 是                       |
| 129  | torch.Tensor.fmod                      | 是                       |
| 130  | torch.Tensor.fmod_                     | 是                       |
| 131  | torch.Tensor.frac                      | 是                       |
| 132  | torch.Tensor.frac_                     | 是                       |
| 133  | torch.Tensor.gather                    | 是                       |
| 134  | torch.Tensor.ge                        | 是                       |
| 135  | torch.Tensor.ge_                       | 是                       |
| 136  | torch.Tensor.geometric_                | 否                       |
| 137  | torch.Tensor.geqrf                     | 否                       |
| 138  | torch.Tensor.ger                       | 是                       |
| 139  | torch.Tensor.get_device                | 是                       |
| 140  | torch.Tensor.gt                        | 是                       |
| 141  | torch.Tensor.gt_                       | 是                       |
| 142  | torch.Tensor.half                      | 是                       |
| 143  | torch.Tensor.hardshrink                | 是                       |
| 144  | torch.Tensor.histc                     | 否                       |
| 145  | torch.Tensor.ifft                      | 否                       |
| 146  | torch.Tensor.index_add_                | 是                       |
| 147  | torch.Tensor.index_add                 | 是                       |
| 148  | torch.Tensor.index_copy_               | 是                       |
| 149  | torch.Tensor.index_copy                | 是                       |
| 150  | torch.Tensor.index_fill_               | 是                       |
| 151  | torch.Tensor.index_fill                | 是                       |
| 152  | torch.Tensor.index_put_                | 是                       |
| 153  | torch.Tensor.index_put                 | 是                       |
| 154  | torch.Tensor.index_select              | 是                       |
| 155  | torch.Tensor.indices                   | 否                       |
| 156  | torch.Tensor.int                       | 是                       |
| 157  | torch.Tensor.int_repr                  | 否                       |
| 158  | torch.Tensor.inverse                   | 是                       |
| 159  | torch.Tensor.irfft                     | 否                       |
| 160  | torch.Tensor.is_contiguous             | 是                       |
| 161  | torch.Tensor.is_complex                | 是                       |
| 162  | torch.Tensor.is_floating_point         | 是                       |
| 163  | torch.Tensor.is_pinned                 | 是                       |
| 164  | torch.Tensor.is_set_to                 | 否                       |
| 165  | torch.Tensor.is_shared                 | 是                       |
| 166  | torch.Tensor.is_signed                 | 是                       |
| 167  | torch.Tensor.is_sparse                 | 是                       |
| 168  | torch.Tensor.item                      | 是                       |
| 169  | torch.Tensor.kthvalue                  | 是                       |
| 170  | torch.Tensor.le                        | 是                       |
| 171  | torch.Tensor.le_                       | 是                       |
| 172  | torch.Tensor.lerp                      | 是                       |
| 173  | torch.Tensor.lerp_                     | 是                       |
| 174  | torch.Tensor.lgamma                    | 否                       |
| 175  | torch.Tensor.lgamma_                   | 否                       |
| 176  | torch.Tensor.log                       | 是                       |
| 177  | torch.Tensor.log_                      | 是                       |
| 178  | torch.Tensor.logdet                    | 否                       |
| 179  | torch.Tensor.log10                     | 是                       |
| 180  | torch.Tensor.log10_                    | 是                       |
| 181  | torch.Tensor.log1p                     | 是                       |
| 182  | torch.Tensor.log1p_                    | 是                       |
| 183  | torch.Tensor.log2                      | 是                       |
| 184  | torch.Tensor.log2_                     | 是                       |
| 185  | torch.Tensor.log_normal_               | 是                       |
| 186  | torch.Tensor.logsumexp                 | 是                       |
| 187  | torch.Tensor.logical_and               | 是                       |
| 188  | torch.Tensor.logical_and_              | 是                       |
| 189  | torch.Tensor.logical_not               | 是                       |
| 190  | torch.Tensor.logical_not_              | 是                       |
| 191  | torch.Tensor.logical_or                | 是                       |
| 192  | torch.Tensor.logical_or_               | 是                       |
| 193  | torch.Tensor.logical_xor               | 否                       |
| 194  | torch.Tensor.logical_xor_              | 否                       |
| 195  | torch.Tensor.long                      | 是                       |
| 196  | torch.Tensor.lstsq                     | 否                       |
| 197  | torch.Tensor.lt                        | 是                       |
| 198  | torch.Tensor.lt_                       | 是                       |
| 199  | torch.Tensor.lu                        | 是                       |
| 200  | torch.Tensor.lu_solve                  | 是                       |
| 201  | torch.Tensor.map_                      | 否                       |
| 202  | torch.Tensor.masked_scatter_           | 是                       |
| 203  | torch.Tensor.masked_scatter            | 是                       |
| 204  | torch.Tensor.masked_fill_              | 是                       |
| 205  | torch.Tensor.masked_fill               | 是                       |
| 206  | torch.Tensor.masked_select             | 是                       |
| 207  | torch.Tensor.matmul                    | 是                       |
| 208  | torch.Tensor.matrix_power              | 是                       |
| 209  | torch.Tensor.max                       | 是                       |
| 210  | torch.Tensor.mean                      | 是                       |
| 211  | torch.Tensor.median                    | 是                       |
| 212  | torch.Tensor.min                       | 是                       |
| 213  | torch.Tensor.mm                        | 是                       |
| 214  | torch.Tensor.mode                      | 否                       |
| 215  | torch.Tensor.mul                       | 是                       |
| 216  | torch.Tensor.mul_                      | 是                       |
| 217  | torch.Tensor.multinomial               | 是                       |
| 218  | torch.Tensor.mv                        | 是                       |
| 219  | torch.Tensor.mvlgamma                  | 否                       |
| 220  | torch.Tensor.mvlgamma_                 | 否                       |
| 221  | torch.Tensor.narrow                    | 是                       |
| 222  | torch.Tensor.narrow_copy               | 是                       |
| 223  | torch.Tensor.ndimension                | 是                       |
| 224  | torch.Tensor.ne                        | 是                       |
| 225  | torch.Tensor.ne_                       | 是                       |
| 226  | torch.Tensor.neg                       | 是                       |
| 227  | torch.Tensor.neg_                      | 是                       |
| 228  | torch.Tensor.nelement                  | 是                       |
| 229  | torch.Tensor.nonzero                   | 是                       |
| 230  | torch.Tensor.norm                      | 是                       |
| 231  | torch.Tensor.normal_                   | 是                       |
| 232  | torch.Tensor.numel                     | 是                       |
| 233  | torch.Tensor.numpy                     | 否                       |
| 234  | torch.Tensor.orgqr                     | 否                       |
| 235  | torch.Tensor.ormqr                     | 否                       |
| 236  | torch.Tensor.permute                   | 是                       |
| 237  | torch.Tensor.pin_memory                | 否                       |
| 238  | torch.Tensor.pinverse                  | 是                       |
| 239  | torch.Tensor.polygamma                 | 否                       |
| 240  | torch.Tensor.polygamma_                | 否                       |
| 241  | torch.Tensor.pow                       | 是                       |
| 242  | torch.Tensor.pow_                      | 是                       |
| 243  | torch.Tensor.prod                      | 是                       |
| 244  | torch.Tensor.put_                      | 是                       |
| 245  | torch.Tensor.qr                        | 是                       |
| 246  | torch.Tensor.qscheme                   | 否                       |
| 247  | torch.Tensor.q_scale                   | 否                       |
| 248  | torch.Tensor.q_zero_point              | 否                       |
| 249  | torch.Tensor.q_per_channel_scales      | 否                       |
| 250  | torch.Tensor.q_per_channel_zero_points | 否                       |
| 251  | torch.Tensor.q_per_channel_axis        | 否                       |
| 252  | torch.Tensor.random_                   | 是                       |
| 253  | torch.Tensor.reciprocal                | 是                       |
| 254  | torch.Tensor.reciprocal_               | 是                       |
| 255  | torch.Tensor.record_stream             | 是                       |
| 256  | torch.Tensor.remainder                 | 是                       |
| 257  | torch.Tensor.remainder_                | 是                       |
| 258  | torch.Tensor.renorm                    | 是                       |
| 259  | torch.Tensor.renorm_                   | 是                       |
| 260  | torch.Tensor.repeat                    | 是                       |
| 261  | torch.Tensor.repeat_interleave         | 是                       |
| 262  | torch.Tensor.requires_grad_            | 是                       |
| 263  | torch.Tensor.reshape                   | 是                       |
| 264  | torch.Tensor.reshape_as                | 是                       |
| 265  | torch.Tensor.resize_                   | 是                       |
| 266  | torch.Tensor.resize_as_                | 是                       |
| 267  | torch.Tensor.rfft                      | 否                       |
| 268  | torch.Tensor.roll                      | 是                       |
| 269  | torch.Tensor.rot90                     | 是                       |
| 270  | torch.Tensor.round                     | 是                       |
| 271  | torch.Tensor.round_                    | 是                       |
| 272  | torch.Tensor.rsqrt                     | 是                       |
| 273  | torch.Tensor.rsqrt_                    | 是                       |
| 274  | torch.Tensor.scatter                   | 是                       |
| 275  | torch.Tensor.scatter_                  | 是                       |
| 276  | torch.Tensor.scatter_add_              | 是                       |
| 277  | torch.Tensor.scatter_add               | 是                       |
| 278  | torch.Tensor.select                    | 是                       |
| 279  | torch.Tensor.set_                      | 是                       |
| 280  | torch.Tensor.share_memory_             | 否                       |
| 281  | torch.Tensor.short                     | 是                       |
| 282  | torch.Tensor.sigmoid                   | 是                       |
| 283  | torch.Tensor.sigmoid_                  | 是                       |
| 284  | torch.Tensor.sign                      | 是                       |
| 285  | torch.Tensor.sign_                     | 是                       |
| 286  | torch.Tensor.sin                       | 是                       |
| 287  | torch.Tensor.sin_                      | 是                       |
| 288  | torch.Tensor.sinh                      | 是                       |
| 289  | torch.Tensor.sinh_                     | 是                       |
| 290  | torch.Tensor.size                      | 是                       |
| 291  | torch.Tensor.slogdet                   | 是                       |
| 292  | torch.Tensor.solve                     | 否                       |
| 293  | torch.Tensor.sort                      | 是                       |
| 294  | torch.Tensor.split                     | 是                       |
| 295  | torch.Tensor.sparse_mask               | 否                       |
| 296  | torch.Tensor.sparse_dim                | 否                       |
| 297  | torch.Tensor.sqrt                      | 是                       |
| 298  | torch.Tensor.sqrt_                     | 是                       |
| 299  | torch.Tensor.square                    | 是                       |
| 300  | torch.Tensor.square_                   | 是                       |
| 301  | torch.Tensor.squeeze                   | 是                       |
| 302  | torch.Tensor.squeeze_                  | 是                       |
| 303  | torch.Tensor.std                       | 是                       |
| 304  | torch.Tensor.stft                      | 否                       |
| 305  | torch.Tensor.storage                   | 是                       |
| 306  | torch.Tensor.storage_offset            | 是                       |
| 307  | torch.Tensor.storage_type              | 是                       |
| 308  | torch.Tensor.stride                    | 是                       |
| 309  | torch.Tensor.sub                       | 是                       |
| 310  | torch.Tensor.sub_                      | 是                       |
| 311  | torch.Tensor.sum                       | 是                       |
| 312  | torch.Tensor.sum_to_size               | 是                       |
| 313  | torch.Tensor.svd                       | 是                       |
| 314  | torch.Tensor.symeig                    | 是                       |
| 315  | torch.Tensor.t                         | 是                       |
| 316  | torch.Tensor.t_                        | 是                       |
| 317  | torch.Tensor.to                        | 是                       |
| 318  | torch.Tensor.to_mkldnn                 | 否                       |
| 319  | torch.Tensor.take                      | 是                       |
| 320  | torch.Tensor.tan                       | 是                       |
| 321  | torch.Tensor.tan_                      | 是                       |
| 322  | torch.Tensor.tanh                      | 是                       |
| 323  | torch.Tensor.tanh_                     | 是                       |
| 324  | torch.Tensor.tolist                    | 是                       |
| 325  | torch.Tensor.topk                      | 是                       |
| 326  | torch.Tensor.to_sparse                 | 否                       |
| 327  | torch.Tensor.trace                     | 否                       |
| 328  | torch.Tensor.transpose                 | 是                       |
| 329  | torch.Tensor.transpose_                | 是                       |
| 330  | torch.Tensor.triangular_solve          | 是                       |
| 331  | torch.Tensor.tril                      | 是                       |
| 332  | torch.Tensor.tril_                     | 是                       |
| 333  | torch.Tensor.triu                      | 是                       |
| 334  | torch.Tensor.triu_                     | 是                       |
| 335  | torch.Tensor.true_divide               | 是                       |
| 336  | torch.Tensor.true_divide_              | 是                       |
| 337  | torch.Tensor.trunc                     | 是                       |
| 338  | torch.Tensor.trunc_                    | 是                       |
| 339  | torch.Tensor.type                      | 是                       |
| 340  | torch.Tensor.type_as                   | 是                       |
| 341  | torch.Tensor.unbind                    | 是                       |
| 342  | torch.Tensor.unfold                    | 是                       |
| 343  | torch.Tensor.uniform_                  | 是                       |
| 344  | torch.Tensor.unique                    | 是                       |
| 345  | torch.Tensor.unique_consecutive        | 否                       |
| 346  | torch.Tensor.unsqueeze                 | 是                       |
| 347  | torch.Tensor.unsqueeze_                | 是                       |
| 348  | torch.Tensor.values                    | 否                       |
| 349  | torch.Tensor.var                       | 否                       |
| 350  | torch.Tensor.view                      | 是                       |
| 351  | torch.Tensor.view_as                   | 是                       |
| 352  | torch.Tensor.where                     | 是                       |
| 353  | torch.Tensor.zero_                     | 是                       |
| 354  | torch.BoolTensor                       | 是                       |
| 355  | torch.BoolTensor.all                   | 是                       |
| 356  | torch.BoolTensor.any                   | 是                       |

## Layers (torch.nn)

| 序号 | API名称                                                  | 是否支持（PyTorch1.5.0）     |
| ---- | -------------------------------------------------------- | ---------------------------- |
| 1    | torch.nn.Parameter                                       | 是                           |
| 2    | torch.nn.Module                                          | 是                           |
| 3    | torch.nn.Module.add_module                               | 是                           |
| 4    | torch.nn.Module.apply                                    | 是                           |
| 5    | torch.nn.Module.bfloat16                                 | 否                           |
| 6    | torch.nn.Module.buffers                                  | 是                           |
| 7    | torch.nn.Module.children                                 | 是                           |
| 8    | torch.nn.Module.cpu                                      | 是                           |
| 9    | torch.nn.Module.cuda                                     | 否                           |
| 10   | torch.nn.Module.double                                   | 否                           |
| 11   | torch.nn.Module.dump_patches                             | 是                           |
| 12   | torch.nn.Module.eval                                     | 是                           |
| 13   | torch.nn.Module.extra_repr                               | 是                           |
| 14   | torch.nn.Module.float                                    | 是                           |
| 15   | torch.nn.Module.forward                                  | 是                           |
| 16   | torch.nn.Module.half                                     | 是                           |
| 17   | torch.nn.Module.load_state_dict                          | 是                           |
| 18   | torch.nn.Module.modules                                  | 是                           |
| 19   | torch.nn.Module.named_buffers                            | 是                           |
| 20   | torch.nn.Module.named_children                           | 是                           |
| 21   | torch.nn.Module.named_modules                            | 是                           |
| 22   | torch.nn.Module.named_parameters                         | 是                           |
| 23   | torch.nn.Module.parameters                               | 是                           |
| 24   | torch.nn.Module.register_backward_hook                   | 是                           |
| 25   | torch.nn.Module.register_buffer                          | 是                           |
| 26   | torch.nn.Module.register_forward_hook                    | 是                           |
| 27   | torch.nn.Module.register_forward_pre_hook                | 是                           |
| 28   | torch.nn.Module.register_parameter                       | 是                           |
| 29   | torch.nn.Module.requires_grad_                           | 是                           |
| 30   | torch.nn.Module.state_dict                               | 是                           |
| 31   | torch.nn.Module.to                                       | 是                           |
| 32   | torch.nn.Module.train                                    | 是                           |
| 33   | torch.nn.Module.type                                     | 是                           |
| 34   | torch.nn.Module.zero_grad                                | 是                           |
| 35   | torch.nn.Sequential                                      | 是                           |
| 36   | torch.nn.ModuleList                                      | 是                           |
| 37   | torch.nn.ModuleList.append                               | 是                           |
| 38   | torch.nn.ModuleList.extend                               | 是                           |
| 39   | torch.nn.ModuleList.insert                               | 是                           |
| 40   | torch.nn.ModuleDict                                      | 是                           |
| 41   | torch.nn.ModuleDict.clear                                | 是                           |
| 42   | torch.nn.ModuleDict.items                                | 是                           |
| 43   | torch.nn.ModuleDict.keys                                 | 是                           |
| 44   | torch.nn.ModuleDict.pop                                  | 是                           |
| 45   | torch.nn.ModuleDict.update                               | 是                           |
| 46   | torch.nn.ModuleDict.values                               | 是                           |
| 47   | torch.nn.ParameterList                                   | 是                           |
| 48   | torch.nn.ParameterList.append                            | 是                           |
| 49   | torch.nn.ParameterList.extend                            | 是                           |
| 50   | torch.nn.ParameterDict                                   | 是                           |
| 51   | torch.nn.ParameterDict.clear                             | 是                           |
| 52   | torch.nn.ParameterDict.items                             | 是                           |
| 53   | torch.nn.ParameterDict.keys                              | 是                           |
| 54   | torch.nn.ParameterDict.pop                               | 是                           |
| 55   | torch.nn.ParameterDict.update                            | 是                           |
| 56   | torch.nn.ParameterDict.values                            | 是                           |
| 57   | torch.nn.Conv1d                                          | 是                           |
| 58   | torch.nn.Conv2d                                          | 是                           |
| 59   | torch.nn.Conv3d                                          | 是                           |
| 60   | torch.nn.ConvTranspose1d                                 | 是                           |
| 61   | torch.nn.ConvTranspose2d                                 | 是                           |
| 62   | torch.nn.ConvTranspose3d                                 | 是                           |
| 63   | torch.nn.Unfold                                          | 是                           |
| 64   | torch.nn.Fold                                            | 是                           |
| 65   | torch.nn.MaxPool1d                                       | 是                           |
| 66   | torch.nn.MaxPool2d                                       | 是                           |
| 67   | torch.nn.MaxPool3d                                       | 是                           |
| 68   | torch.nn.MaxUnpool1d                                     | 是                           |
| 69   | torch.nn.MaxUnpool2d                                     | 是                           |
| 70   | torch.nn.MaxUnpool3d                                     | 是                           |
| 71   | torch.nn.AvgPool1d                                       | 是                           |
| 72   | torch.nn.AvgPool2d                                       | 是                           |
| 73   | torch.nn.AvgPool3d                                       | 是                           |
| 74   | torch.nn.FractionalMaxPool2d                             | 否                           |
| 75   | torch.nn.LPPool1d                                        | 是                           |
| 76   | torch.nn.LPPool2d                                        | 是                           |
| 77   | torch.nn.AdaptiveMaxPool1d                               | 是                           |
| 78   | torch.nn.AdaptiveMaxPool2d                               | 是                           |
| 79   | torch.nn.AdaptiveMaxPool3d                               | 否                           |
| 80   | torch.nn.AdaptiveAvgPool1d                               | 是                           |
| 81   | torch.nn.AdaptiveAvgPool2d                               | 是                           |
| 82   | torch.nn.AdaptiveAvgPool3d                               | 是，仅支持D=1，H=1，W=1场景  |
| 83   | torch.nn.ReflectionPad1d                                 | 否                           |
| 84   | torch.nn.ReflectionPad2d                                 | 是                           |
| 85   | torch.nn.ReplicationPad1d                                | 否                           |
| 86   | torch.nn.ReplicationPad2d                                | 是                           |
| 87   | torch.nn.ReplicationPad3d                                | 否                           |
| 88   | torch.nn.ZeroPad2d                                       | 是                           |
| 89   | torch.nn.ConstantPad1d                                   | 是                           |
| 90   | torch.nn.ConstantPad2d                                   | 是                           |
| 91   | torch.nn.ConstantPad3d                                   | 是                           |
| 92   | torch.nn.ELU                                             | 是                           |
| 93   | torch.nn.Hardshrink                                      | 是                           |
| 94   | torch.nn.Hardtanh                                        | 是                           |
| 95   | torch.nn.LeakyReLU                                       | 是                           |
| 96   | torch.nn.LogSigmoid                                      | 是                           |
| 97   | torch.nn.MultiheadAttention                              | 是                           |
| 98   | torch.nn.PReLU                                           | 是                           |
| 99   | torch.nn.ReLU                                            | 是                           |
| 100  | torch.nn.ReLU6                                           | 是                           |
| 101  | torch.nn.RReLU                                           | 是                           |
| 102  | torch.nn.SELU                                            | 是                           |
| 103  | torch.nn.CELU                                            | 是                           |
| 104  | torch.nn.GELU                                            | 是                           |
| 105  | torch.nn.Sigmoid                                         | 是                           |
| 106  | torch.nn.Softplus                                        | 是                           |
| 107  | torch.nn.Softshrink                                      | 是，SoftShrink场景暂不支持   |
| 108  | torch.nn.Softsign                                        | 是                           |
| 109  | torch.nn.Tanh                                            | 是                           |
| 110  | torch.nn.Tanhshrink                                      | 是                           |
| 111  | torch.nn.Threshold                                       | 是                           |
| 112  | torch.nn.Softmin                                         | 是                           |
| 113  | torch.nn.Softmax                                         | 是                           |
| 114  | torch.nn.Softmax2d                                       | 是                           |
| 115  | torch.nn.LogSoftmax                                      | 是                           |
| 116  | torch.nn.AdaptiveLogSoftmaxWithLoss                      | 否                           |
| 117  | torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob             | 否                           |
| 118  | torch.nn.AdaptiveLogSoftmaxWithLoss.predict              | 否                           |
| 119  | torch.nn.BatchNorm1d                                     | 是                           |
| 120  | torch.nn.BatchNorm2d                                     | 是                           |
| 121  | torch.nn.BatchNorm3d                                     | 是                           |
| 122  | torch.nn.GroupNorm                                       | 是                           |
| 123  | torch.nn.SyncBatchNorm                                   | 否                           |
| 124  | torch.nn.SyncBatchNorm.convert_sync_batchnorm            | 否                           |
| 125  | torch.nn.InstanceNorm1d                                  | 是                           |
| 126  | torch.nn.InstanceNorm2d                                  | 是                           |
| 127  | torch.nn.InstanceNorm3d                                  | 是                           |
| 128  | torch.nn.LayerNorm                                       | 是                           |
| 129  | torch.nn.LocalResponseNorm                               | 是                           |
| 130  | torch.nn.RNNBase                                         | 是                           |
| 131  | torch.nn.RNNBase.flatten_parameters                      | 是                           |
| 132  | torch.nn.RNN                                             | 是                           |
| 133  | torch.nn.LSTM                                            | 是                           |
| 134  | torch.nn.GRU                                             | 是，DynamicGRUV2场景暂不支持 |
| 135  | torch.nn.RNNCell                                         | 是                           |
| 136  | torch.nn.LSTMCell                                        | 是                           |
| 137  | torch.nn.GRUCell                                         | 是                           |
| 138  | torch.nn.Transformer                                     | 是                           |
| 139  | torch.nn.Transformer.forward                             | 是                           |
| 140  | torch.nn.Transformer.generate_square_subsequent_mask     | 是                           |
| 141  | torch.nn.TransformerEncoder                              | 是                           |
| 142  | torch.nn.TransformerEncoder.forward                      | 是                           |
| 143  | torch.nn.TransformerDecoder                              | 是                           |
| 144  | torch.nn.TransformerDecoder.forward                      | 是                           |
| 145  | torch.nn.TransformerEncoderLayer                         | 是                           |
| 146  | torch.nn.TransformerEncoderLayer.forward                 | 是                           |
| 147  | torch.nn.TransformerDecoderLayer                         | 是                           |
| 148  | torch.nn.TransformerDecoderLayer.forward                 | 是                           |
| 149  | torch.nn.Identity                                        | 是                           |
| 150  | torch.nn.Linear                                          | 是                           |
| 151  | torch.nn.Bilinear                                        | 是                           |
| 152  | torch.nn.Dropout                                         | 是                           |
| 153  | torch.nn.Dropout2d                                       | 是                           |
| 154  | torch.nn.Dropout3d                                       | 是                           |
| 155  | torch.nn.AlphaDropout                                    | 是                           |
| 156  | torch.nn.Embedding                                       | 是                           |
| 157  | torch.nn.Embedding.from_pretrained                       | 是                           |
| 158  | torch.nn.EmbeddingBag                                    | 是                           |
| 159  | torch.nn.EmbeddingBag.from_pretrained                    | 是                           |
| 160  | torch.nn.CosineSimilarity                                | 是                           |
| 161  | torch.nn.PairwiseDistance                                | 是                           |
| 162  | torch.nn.L1Loss                                          | 是                           |
| 163  | torch.nn.MSELoss                                         | 是                           |
| 164  | torch.nn.CrossEntropyLoss                                | 是                           |
| 165  | torch.nn.CTCLoss                                         | 是                           |
| 166  | torch.nn.NLLLoss                                         | 是                           |
| 167  | torch.nn.PoissonNLLLoss                                  | 是                           |
| 168  | torch.nn.KLDivLoss                                       | 是                           |
| 169  | torch.nn.BCELoss                                         | 是                           |
| 170  | torch.nn.BCEWithLogitsLoss                               | 是                           |
| 171  | torch.nn.MarginRankingLoss                               | 是                           |
| 172  | torch.nn.HingeEmbeddingLoss                              | 是                           |
| 173  | torch.nn.MultiLabelMarginLoss                            | 是                           |
| 174  | torch.nn.SmoothL1Loss                                    | 是                           |
| 175  | torch.nn.SoftMarginLoss                                  | 是                           |
| 176  | torch.nn.MultiLabelSoftMarginLoss                        | 是                           |
| 177  | torch.nn.CosineEmbeddingLoss                             | 是                           |
| 178  | torch.nn.MultiMarginLoss                                 | 否                           |
| 179  | torch.nn.TripletMarginLoss                               | 是                           |
| 180  | torch.nn.PixelShuffle                                    | 是                           |
| 181  | torch.nn.Upsample                                        | 是                           |
| 182  | torch.nn.UpsamplingNearest2d                             | 是                           |
| 183  | torch.nn.UpsamplingBilinear2d                            | 是                           |
| 184  | torch.nn.DataParallel                                    | 否                           |
| 185  | torch.nn.parallel.DistributedDataParallel                | 是                           |
| 186  | torch.nn.parallel.DistributedDataParallel.no_sync        | 是                           |
| 187  | torch.nn.utils.clip_grad_norm_                           | 是                           |
| 188  | torch.nn.utils.clip_grad_value_                          | 是                           |
| 189  | torch.nn.utils.parameters_to_vector                      | 是                           |
| 190  | torch.nn.utils.vector_to_parameters                      | 是                           |
| 197  | torch.nn.utils.prune.PruningContainer                    | 是                           |
| 198  | torch.nn.utils.prune.PruningContainer.add_pruning_method | 是                           |
| 199  | torch.nn.utils.prune.PruningContainer.apply              | 是                           |
| 200  | torch.nn.utils.prune.PruningContainer.apply_mask         | 是                           |
| 201  | torch.nn.utils.prune.PruningContainer.compute_mask       | 是                           |
| 202  | torch.nn.utils.prune.PruningContainer.prune              | 是                           |
| 203  | torch.nn.utils.prune.PruningContainer.remove             | 是                           |
| 204  | torch.nn.utils.prune.Identity                            | 是                           |
| 205  | torch.nn.utils.prune.Identity.apply                      | 是                           |
| 206  | torch.nn.utils.prune.Identity.apply_mask                 | 是                           |
| 207  | torch.nn.utils.prune.Identity.prune                      | 是                           |
| 208  | torch.nn.utils.prune.Identity.remove                     | 是                           |
| 209  | torch.nn.utils.prune.RandomUnstructured                  | 是                           |
| 210  | torch.nn.utils.prune.RandomUnstructured.apply            | 是                           |
| 211  | torch.nn.utils.prune.RandomUnstructured.apply_mask       | 是                           |
| 212  | torch.nn.utils.prune.RandomUnstructured.prune            | 是                           |
| 213  | torch.nn.utils.prune.RandomUnstructured.remove           | 是                           |
| 214  | torch.nn.utils.prune.L1Unstructured                      | 是                           |
| 215  | torch.nn.utils.prune.L1Unstructured.apply                | 是                           |
| 216  | torch.nn.utils.prune.L1Unstructured.apply_mask           | 是                           |
| 217  | torch.nn.utils.prune.L1Unstructured.prune                | 是                           |
| 218  | torch.nn.utils.prune.L1Unstructured.remove               | 是                           |
| 219  | torch.nn.utils.prune.RandomStructured                    | 是                           |
| 220  | torch.nn.utils.prune.RandomStructured.apply              | 是                           |
| 221  | torch.nn.utils.prune.RandomStructured.apply_mask         | 是                           |
| 222  | torch.nn.utils.prune.RandomStructured.compute_mask       | 是                           |
| 223  | torch.nn.utils.prune.RandomStructured.prune              | 是                           |
| 224  | torch.nn.utils.prune.RandomStructured.remove             | 是                           |
| 225  | torch.nn.utils.prune.LnStructured                        | 是                           |
| 226  | torch.nn.utils.prune.LnStructured.apply                  | 是                           |
| 227  | torch.nn.utils.prune.LnStructured.apply_mask             | 是                           |
| 228  | torch.nn.utils.prune.LnStructured.compute_mask           | 是                           |
| 229  | torch.nn.utils.prune.LnStructured.prune                  | 是                           |
| 230  | torch.nn.utils.prune.LnStructured.remove                 | 是                           |
| 231  | torch.nn.utils.prune.CustomFromMask                      | 是                           |
| 232  | torch.nn.utils.prune.CustomFromMask.apply                | 是                           |
| 233  | torch.nn.utils.prune.CustomFromMask.apply_mask           | 是                           |
| 234  | torch.nn.utils.prune.CustomFromMask.prune                | 是                           |
| 235  | torch.nn.utils.prune.CustomFromMask.remove               | 是                           |
| 236  | torch.nn.utils.prune.identity                            | 是                           |
| 237  | torch.nn.utils.prune.random_unstructured                 | 是                           |
| 238  | torch.nn.utils.prune.l1_unstructured                     | 是                           |
| 239  | torch.nn.utils.prune.random_structured                   | 是                           |
| 240  | torch.nn.utils.prune.ln_structured                       | 是                           |
| 241  | torch.nn.utils.prune.global_unstructured                 | 是                           |
| 242  | torch.nn.utils.prune.custom_from_mask                    | 是                           |
| 243  | torch.nn.utils.prune.remove                              | 是                           |
| 244  | torch.nn.utils.prune.is_pruned                           | 是                           |
| 245  | torch.nn.utils.weight_norm                               | 是                           |
| 246  | torch.nn.utils.remove_weight_norm                        | 是                           |
| 247  | torch.nn.utils.spectral_norm                             | 是                           |
| 248  | torch.nn.utils.remove_spectral_norm                      | 是                           |
| 249  | torch.nn.utils.rnn.PackedSequence                        | 是                           |
| 250  | torch.nn.utils.rnn.pack_padded_sequence                  | 是                           |
| 251  | torch.nn.utils.rnn.pad_packed_sequence                   | 否                           |
| 252  | torch.nn.utils.rnn.pad_sequence                          | 是                           |
| 253  | torch.nn.utils.rnn.pack_sequence                         | 否                           |
| 254  | torch.nn.Flatten                                         | 是                           |
| 255  | torch.quantization.quantize                              | 否                           |
| 256  | torch.quantization.quantize_dynamic                      | 否                           |
| 257  | torch.quantization.quantize_qat                          | 否                           |
| 258  | torch.quantization.prepare                               | 是                           |
| 259  | torch.quantization.prepare_qat                           | 否                           |
| 260  | torch.quantization.convert                               | 否                           |
| 261  | torch.quantization.QConfig                               | 是                           |
| 262  | torch.quantization.QConfigDynamic                        | 是                           |
| 263  | torch.quantization.fuse_modules                          | 是                           |
| 264  | torch.quantization.QuantStub                             | 是                           |
| 265  | torch.quantization.DeQuantStub                           | 是                           |
| 266  | torch.quantization.QuantWrapper                          | 是                           |
| 267  | torch.quantization.add_quant_dequant                     | 是                           |
| 268  | torch.quantization.add_observer_                         | 是                           |
| 269  | torch.quantization.swap_module                           | 是                           |
| 270  | torch.quantization.propagate_qconfig_                    | 是                           |
| 271  | torch.quantization.default_eval_fn                       | 是                           |
| 272  | torch.quantization.MinMaxObserver                        | 是                           |
| 273  | torch.quantization.MovingAverageMinMaxObserver           | 是                           |
| 274  | torch.quantization.PerChannelMinMaxObserver              | 是                           |
| 275  | torch.quantization.MovingAveragePerChannelMinMaxObserver | 是                           |
| 276  | torch.quantization.HistogramObserver                     | 否                           |
| 277  | torch.quantization.FakeQuantize                          | 否                           |
| 278  | torch.quantization.NoopObserver                          | 是                           |
| 279  | torch.quantization.get_observer_dict                     | 是                           |
| 280  | torch.quantization.RecordingObserver                     | 是                           |
| 281  | torch.nn.intrinsic.ConvBn2d                              | 是                           |
| 282  | torch.nn.intrinsic.ConvBnReLU2d                          | 是                           |
| 283  | torch.nn.intrinsic.ConvReLU2d                            | 是                           |
| 284  | torch.nn.intrinsic.ConvReLU3d                            | 是                           |
| 285  | torch.nn.intrinsic.LinearReLU                            | 是                           |
| 286  | torch.nn.intrinsic.qat.ConvBn2d                          | 否                           |
| 287  | torch.nn.intrinsic.qat.ConvBnReLU2d                      | 否                           |
| 288  | torch.nn.intrinsic.qat.ConvReLU2d                        | 否                           |
| 289  | torch.nn.intrinsic.qat.LinearReLU                        | 否                           |
| 290  | torch.nn.intrinsic.quantized.ConvReLU2d                  | 否                           |
| 291  | torch.nn.intrinsic.quantized.ConvReLU3d                  | 否                           |
| 292  | torch.nn.intrinsic.quantized.LinearReLU                  | 否                           |
| 293  | torch.nn.qat.Conv2d                                      | 否                           |
| 294  | torch.nn.qat.Conv2d.from_float                           | 否                           |
| 295  | torch.nn.qat.Linear                                      | 否                           |
| 296  | torch.nn.qat.Linear.from_float                           | 否                           |
| 297  | torch.nn.quantized.functional.relu                       | 否                           |
| 298  | torch.nn.quantized.functional.linear                     | 否                           |
| 299  | torch.nn.quantized.functional.conv2d                     | 否                           |
| 300  | torch.nn.quantized.functional.conv3d                     | 否                           |
| 301  | torch.nn.quantized.functional.max_pool2d                 | 否                           |
| 302  | torch.nn.quantized.functional.adaptive_avg_pool2d        | 否                           |
| 303  | torch.nn.quantized.functional.avg_pool2d                 | 否                           |
| 304  | torch.nn.quantized.functional.interpolate                | 否                           |
| 305  | torch.nn.quantized.functional.upsample                   | 否                           |
| 306  | torch.nn.quantized.functional.upsample_bilinear          | 否                           |
| 307  | torch.nn.quantized.functional.upsample_nearest           | 否                           |
| 308  | torch.nn.quantized.ReLU                                  | 否                           |
| 309  | torch.nn.quantized.ReLU6                                 | 否                           |
| 310  | torch.nn.quantized.Conv2d                                | 否                           |
| 311  | torch.nn.quantized.Conv2d.from_float                     | 否                           |
| 312  | torch.nn.quantized.Conv3d                                | 否                           |
| 313  | torch.nn.quantized.Conv3d.from_float                     | 否                           |
| 314  | torch.nn.quantized.FloatFunctional                       | 是                           |
| 315  | torch.nn.quantized.QFunctional                           | 否                           |
| 316  | torch.nn.quantized.Quantize                              | 是                           |
| 317  | torch.nn.quantized.DeQuantize                            | 否                           |
| 318  | torch.nn.quantized.Linear                                | 否                           |
| 319  | torch.nn.quantized.Linear.from_float                     | 否                           |
| 320  | torch.nn.quantized.dynamic.Linear                        | 否                           |
| 321  | torch.nn.quantized.dynamic.Linear.from_float             | 否                           |
| 322  | torch.nn.quantized.dynamic.LSTM                          | 否                           |

## Functions(torch.nn.functional)

| 序号 | API名称                                              | 是否支持（PyTorch1.5.0）    |
| ---- | ---------------------------------------------------- | --------------------------- |
| 1    | torch.nn.functional.conv1d                           | 是                          |
| 2    | torch.nn.functional.conv2d                           | 是                          |
| 3    | torch.nn.functional.conv3d                           | 是                          |
| 4    | torch.nn.functional.conv_transpose1d                 | 是                          |
| 5    | torch.nn.functional.conv_transpose2d                 | 是                          |
| 6    | torch.nn.functional.conv_transpose3d                 | 是                          |
| 7    | torch.nn.functional.unfold                           | 是                          |
| 8    | torch.nn.functional.fold                             | 是                          |
| 9    | torch.nn.functional.avg_pool1d                       | 是                          |
| 10   | torch.nn.functional.avg_pool2d                       | 是                          |
| 11   | torch.nn.functional.avg_pool3d                       | 是                          |
| 12   | torch.nn.functional.max_pool1d                       | 是                          |
| 13   | torch.nn.functional.max_pool2d                       | 是                          |
| 14   | torch.nn.functional.max_pool3d                       | 是                          |
| 15   | torch.nn.functional.max_unpool1d                     | 是                          |
| 16   | torch.nn.functional.max_unpool2d                     | 是                          |
| 17   | torch.nn.functional.max_unpool3d                     | 是                          |
| 18   | torch.nn.functional.lp_pool1d                        | 是                          |
| 19   | torch.nn.functional.lp_pool2d                        | 是                          |
| 20   | torch.nn.functional.adaptive_max_pool1d              | 是                          |
| 21   | torch.nn.functional.adaptive_max_pool2d              | 是                          |
| 22   | torch.nn.functional.adaptive_max_pool3d              | 否                          |
| 23   | torch.nn.functional.adaptive_avg_pool1d              | 是                          |
| 24   | torch.nn.functional.adaptive_avg_pool2d              | 是                          |
| 25   | torch.nn.functional.adaptive_avg_pool3d              | 是，仅支持D=1，H=1，W=1场景 |
| 26   | torch.nn.functional.threshold                        | 是                          |
| 27   | torch.nn.functional.threshold_                       | 是                          |
| 28   | torch.nn.functional.relu                             | 是                          |
| 29   | torch.nn.functional.relu_                            | 是                          |
| 30   | torch.nn.functional.hardtanh                         | 是                          |
| 31   | torch.nn.functional.hardtanh_                        | 是                          |
| 32   | torch.nn.functional.relu6                            | 是                          |
| 33   | torch.nn.functional.elu                              | 是                          |
| 34   | torch.nn.functional.elu_                             | 是                          |
| 35   | torch.nn.functional.selu                             | 是                          |
| 36   | torch.nn.functional.celu                             | 是                          |
| 37   | torch.nn.functional.leaky_relu                       | 是                          |
| 38   | torch.nn.functional.leaky_relu_                      | 是                          |
| 39   | torch.nn.functional.prelu                            | 是                          |
| 40   | torch.nn.functional.rrelu                            | 是                          |
| 41   | torch.nn.functional.rrelu_                           | 是                          |
| 42   | torch.nn.functional.glu                              | 是                          |
| 43   | torch.nn.functional.gelu                             | 是                          |
| 44   | torch.nn.functional.logsigmoid                       | 是                          |
| 45   | torch.nn.functional.hardshrink                       | 是                          |
| 46   | torch.nn.functional.tanhshrink                       | 是                          |
| 47   | torch.nn.functional.softsign                         | 是                          |
| 48   | torch.nn.functional.softplus                         | 是                          |
| 49   | torch.nn.functional.softmin                          | 是                          |
| 50   | torch.nn.functional.softmax                          | 是                          |
| 51   | torch.nn.functional.softshrink                       | 是                          |
| 52   | torch.nn.functional.gumbel_softmax                   | 否                          |
| 53   | torch.nn.functional.log_softmax                      | 是                          |
| 54   | torch.nn.functional.tanh                             | 是                          |
| 55   | torch.nn.functional.sigmoid                          | 是                          |
| 56   | torch.nn.functional.batch_norm                       | 是                          |
| 57   | torch.nn.functional.instance_norm                    | 是                          |
| 58   | torch.nn.functional.layer_norm                       | 是                          |
| 59   | torch.nn.functional.local_response_norm              | 是                          |
| 60   | torch.nn.functional.normalize                        | 是                          |
| 61   | torch.nn.functional.linear                           | 是                          |
| 62   | torch.nn.functional.bilinear                         | 是                          |
| 63   | torch.nn.functional.dropout                          | 是                          |
| 64   | torch.nn.functional.alpha_dropout                    | 是                          |
| 65   | torch.nn.functional.dropout2d                        | 是                          |
| 66   | torch.nn.functional.dropout3d                        | 是                          |
| 67   | torch.nn.functional.embedding                        | 是                          |
| 68   | torch.nn.functional.embedding_bag                    | 是                          |
| 69   | torch.nn.functional.one_hot                          | 是                          |
| 70   | torch.nn.functional.pairwise_distance                | 是                          |
| 71   | torch.nn.functional.cosine_similarity                | 是                          |
| 72   | torch.nn.functional.pdist                            | 是                          |
| 73   | torch.nn.functional.binary_cross_entropy             | 是                          |
| 74   | torch.nn.functional.binary_cross_entropy_with_logits | 是                          |
| 75   | torch.nn.functional.poisson_nll_loss                 | 是                          |
| 76   | torch.nn.functional.cosine_embedding_loss            | 是                          |
| 77   | torch.nn.functional.cross_entropy                    | 是                          |
| 78   | torch.nn.functional.ctc_loss                         | 是（仅支持2维输入）         |
| 79   | torch.nn.functional.hinge_embedding_loss             | 是                          |
| 80   | torch.nn.functional.kl_div                           | 是                          |
| 81   | torch.nn.functional.l1_loss                          | 是                          |
| 82   | torch.nn.functional.mse_loss                         | 是                          |
| 83   | torch.nn.functional.margin_ranking_loss              | 是                          |
| 84   | torch.nn.functional.multilabel_margin_loss           | 是                          |
| 85   | torch.nn.functional.multilabel_soft_margin_loss      | 是                          |
| 86   | torch.nn.functional.multi_margin_loss                | 否                          |
| 87   | torch.nn.functional.nll_loss                         | 是                          |
| 88   | torch.nn.functional.smooth_l1_loss                   | 是                          |
| 89   | torch.nn.functional.soft_margin_loss                 | 是                          |
| 90   | torch.nn.functional.triplet_margin_loss              | 是                          |
| 91   | torch.nn.functional.pixel_shuffle                    | 是                          |
| 92   | torch.nn.functional.pad                              | 是                          |
| 93   | torch.nn.functional.interpolate                      | 是                          |
| 94   | torch.nn.functional.upsample                         | 是                          |
| 95   | torch.nn.functional.upsample_nearest                 | 是                          |
| 96   | torch.nn.functional.upsample_bilinear                | 是                          |
| 97   | torch.nn.functional.grid_sample                      | 是                          |
| 98   | torch.nn.functional.affine_grid                      | 是                          |
| 99   | torch.nn.parallel.data_parallel                      | 否                          |

## torch.distributed

| 序号 | API名称                               | 是否支持（PyTorch1.5.0） |
| ---- | ------------------------------------- | ------------------------ |
| 1    | torch.distributed.init_process_group  | 是                       |
| 2    | torch.distributed.Backend             | 是                       |
| 3    | torch.distributed.get_backend         | 是                       |
| 4    | torch.distributed.get_rank            | 是                       |
| 5    | torch.distributed.get_world_size      | 是                       |
| 6    | torch.distributed.is_initialized      | 是                       |
| 7    | torch.distributed.is_mpi_available    | 是                       |
| 8    | torch.distributed.is_nccl_available   | 是                       |
| 9    | torch.distributed.new_group           | 是                       |
| 10   | torch.distributed.send                | 否                       |
| 11   | torch.distributed.recv                | 否                       |
| 12   | torch.distributed.isend               | 否                       |
| 13   | torch.distributed.irecv               | 否                       |
| 14   | is_completed                          | 是                       |
| 15   | wait                                  | 是                       |
| 16   | torch.distributed.broadcast           | 是                       |
| 17   | torch.distributed.all_reduce          | 是                       |
| 18   | torch.distributed.reduce              | 否                       |
| 19   | torch.distributed.all_gather          | 是                       |
| 20   | torch.distributed.gather              | 否                       |
| 21   | torch.distributed.scatter             | 否                       |
| 22   | torch.distributed.barrier             | 是                       |
| 23   | torch.distributed.ReduceOp            | 是                       |
| 24   | torch.distributed.reduce_op           | 是                       |
| 25   | torch.distributed.broadcast_multigpu  | 否                       |
| 26   | torch.distributed.all_reduce_multigpu | 否                       |
| 27   | torch.distributed.reduce_multigpu     | 否                       |
| 28   | torch.distributed.all_gather_multigpu | 否                       |
| 29   | torch.distributed.launch              | 是                       |
| 30   | torch.multiprocessing.spawn           | 是                       |

## torch.npu

| 序号 | API名称                               | npu对应API名称                       | 是否支持（PyTorch1.5.0） |
| ---- | ------------------------------------- | ------------------------------------ | ------------------------ |
| 1    | torch.cuda.current_blas_handle        | torch.npu.current_blas_handle        | 否                       |
| 2    | torch.cuda.current_device             | torch.npu.current_device             | 是                       |
| 3    | torch.cuda.current_stream             | torch.npu.current_stream             | 是                       |
| 4    | torch.cuda.default_stream             | torch.npu.default_stream             | 是                       |
| 5    | torch.cuda.device                     | torch.npu.device                     | 是                       |
| 6    | torch.cuda.device_count               | torch.npu.device_count               | 是                       |
| 7    | torch.cuda.device_of                  | torch.npu.device_of                  | 是                       |
| 8    | torch.cuda.get_device_capability      | torch.npu.get_device_capability      | 否                       |
| 9    | torch.cuda.get_device_name            | torch.npu.get_device_name            | 否                       |
| 10   | torch.cuda.init                       | torch.npu.init                       | 是                       |
| 11   | torch.cuda.ipc_collect                | torch.npu.ipc_collect                | 否                       |
| 12   | torch.cuda.is_available               | torch.npu.is_available               | 是                       |
| 13   | torch.cuda.is_initialized             | torch.npu.is_initialized             | 是                       |
| 14   | torch.cuda.set_device                 | torch.npu.set_device                 | 部分支持                 |
| 15   | torch.cuda.stream                     | torch.npu.stream                     | 是                       |
| 16   | torch.cuda.synchronize                | torch.npu.synchronize                | 是                       |
| 17   | torch.cuda.get_rng_state              | torch.npu.get_rng_state              | 否                       |
| 18   | torch.cuda.get_rng_state_all          | torch.npu.get_rng_state_all          | 否                       |
| 19   | torch.cuda.set_rng_state              | torch.npu.set_rng_state              | 否                       |
| 20   | torch.cuda.set_rng_state_all          | torch.npu.set_rng_state_all          | 否                       |
| 21   | torch.cuda.manual_seed                | torch.npu.manual_seed                | 否                       |
| 22   | torch.cuda.manual_seed_all            | torch.npu.manual_seed_all            | 否                       |
| 23   | torch.cuda.seed                       | torch.npu.seed                       | 否                       |
| 24   | torch.cuda.seed_all                   | torch.npu.seed_all                   | 否                       |
| 25   | torch.cuda.initial_seed               | torch.npu.initial_seed               | 否                       |
| 26   | torch.cuda.comm.broadcast             | torch.npu.comm.broadcast             | 否                       |
| 27   | torch.cuda.comm.broadcast_coalesced   | torch.npu.comm.broadcast_coalesced   | 否                       |
| 28   | torch.cuda.comm.reduce_add            | torch.npu.comm.reduce_add            | 否                       |
| 29   | torch.cuda.comm.scatter               | torch.npu.comm.scatter               | 否                       |
| 30   | torch.cuda.comm.gather                | torch.npu.comm.gather                | 否                       |
| 31   | torch.cuda.Stream                     | torch.npu.Stream                     | 是                       |
| 32   | torch.cuda.Stream.query               | torch.npu.Stream.query               | 是                       |
| 33   | torch.cuda.Stream.record_event        | torch.npu.Stream.record_event        | 是                       |
| 34   | torch.cuda.Stream.synchronize         | torch.npu.Stream.synchronize         | 是                       |
| 35   | torch.cuda.Stream.wait_event          | torch.npu.Stream.wait_event          | 是                       |
| 36   | torch.cuda.Stream.wait_stream         | torch.npu.Stream.wait_stream         | 是                       |
| 37   | torch.cuda.Event                      | torch.npu.Event                      | 是                       |
| 38   | torch.cuda.Event.elapsed_time         | torch.npu.Event.elapsed_time         | 是                       |
| 39   | torch.cuda.Event.from_ipc_handle      | torch.npu.Event.from_ipc_handle      | 否                       |
| 40   | torch.cuda.Event.ipc_handle           | torch.npu.Event.ipc_handle           | 否                       |
| 41   | torch.cuda.Event.query                | torch.npu.Event.query                | 是                       |
| 42   | torch.cuda.Event.record               | torch.npu.Event.record               | 是                       |
| 43   | torch.cuda.Event.synchronize          | torch.npu.Event.synchronize          | 是                       |
| 44   | torch.cuda.Event.wait                 | torch.npu.Event.wait                 | 是                       |
| 45   | torch.cuda.empty_cache                | torch.npu.empty_cache                | 是                       |
| 46   | torch.cuda.memory_stats               | torch.npu.memory_stats               | 是                       |
| 47   | torch.cuda.memory_summary             | torch.npu.memory_summary             | 是                       |
| 48   | torch.cuda.memory_snapshot            | torch.npu.memory_snapshot            | 是                       |
| 49   | torch.cuda.memory_allocated           | torch.npu.memory_allocated           | 是                       |
| 50   | torch.cuda.max_memory_allocated       | torch.npu.max_memory_allocated       | 是                       |
| 51   | torch.cuda.reset_max_memory_allocated | torch.npu.reset_max_memory_allocated | 是                       |
| 52   | torch.cuda.memory_reserved            | torch.npu.memory_reserved            | 是                       |
| 53   | torch.cuda.max_memory_reserved        | torch.npu.max_memory_reserved        | 是                       |
| 54   | torch.cuda.memory_cached              | torch.npu.memory_cached              | 是                       |
| 55   | torch.cuda.max_memory_cached          | torch.npu.max_memory_cached          | 是                       |
| 56   | torch.cuda.reset_max_memory_cached    | torch.npu.reset_max_memory_cached    | 是                       |
| 57   | torch.cuda.nvtx.mark                  | torch.npu.nvtx.mark                  | 否                       |
| 58   | torch.cuda.nvtx.range_push            | torch.npu.nvtx.range_push            | 否                       |
| 59   | torch.cuda.nvtx.range_pop             | torch.npu.nvtx.range_pop             | 否                       |
| 60   | torch.cuda._sleep                     | torch.npu._sleep                     | 否                       |
| 61   | torch.cuda.Stream.priority_range      | torch.npu.Stream.priority_range      | 否                       |
| 62   | torch.cuda.get_device_properties      | torch.npu.get_device_properties      | 否                       |
| 63   | torch.cuda.amp.GradScaler             | torch.npu.amp.GradScaler             | 否                       |

torch.npu.set_device()接口只支持在程序开始的位置通过set_device进行指定，不支持多次指定和with torch.npu.device(id)方式的device切换

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

算子接口说明：

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

> npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, out = (var, m, v))

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



