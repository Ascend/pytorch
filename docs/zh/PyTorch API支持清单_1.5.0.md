## torch<a name="ZH-CN_TOPIC_0000001398211686"></a>

| 分类                                       | API名称                   | 是否支持                                    |
| ------------------------------------------ | ------------------------- | ------------------------------------------- |
| **Tensors**                                | is_tensor                 | 是                                          |
|                                            | is_storage                | 是                                          |
|                                            | is_complex                | 是，支持判断，但当前硬件限制不支持复数      |
|                                            | is_floating_point         | 是                                          |
|                                            | set_default_dtype         | 是                                          |
|                                            | get_default_dtype         | 是                                          |
|                                            | set_default_tensor_type   | 是                                          |
|                                            | numel                     | 是                                          |
|                                            | set_printoptions          | 是                                          |
|                                            | set_flush_denormal        | 是                                          |
|                                            | tensor                    | 是                                          |
|                                            | sparse_coo_tensor         | 否                                          |
|                                            | as_tensor                 | 是                                          |
|                                            | as_strided                | 是                                          |
|                                            | from_numpy                | 是                                          |
|                                            | zeros                     | 是                                          |
|                                            | zeros_like                | 是                                          |
|                                            | ones                      | 是                                          |
|                                            | ones_like                 | 是                                          |
|                                            | arange                    | 是                                          |
|                                            | range                     | 是                                          |
|                                            | linspace                  | 是                                          |
|                                            | logspace                  | 是                                          |
|                                            | eye                       | 是                                          |
|                                            | empty                     | 是                                          |
|                                            | empty_like                | 是                                          |
|                                            | empty_strided             | 是                                          |
|                                            | full                      | 是                                          |
|                                            | full_like                 | 是                                          |
|                                            | quantize_per_tensor       | 是                                          |
|                                            | quantize_per_channel      | 是                                          |
|                                            | cat                       | 是                                          |
|                                            | chunk                     | 是                                          |
|                                            | gather                    | 是                                          |
|                                            | index_select              | 是                                          |
|                                            | masked_select             | 是                                          |
|                                            | narrow                    | 是                                          |
|                                            | nonzero                   | 是                                          |
|                                            | reshape                   | 是                                          |
|                                            | split                     | 是                                          |
|                                            | squeeze                   | 是                                          |
|                                            | stack                     | 是                                          |
|                                            | t                         | 是                                          |
|                                            | take                      | 是                                          |
|                                            | transpose                 | 是                                          |
|                                            | unbind                    | 是                                          |
|                                            | unsqueeze                 | 是                                          |
|                                            | where                     | 是                                          |
| **Generators**                             | Generator                 | 是，但不支持get_state和set_state接口        |
| **Random sampling**                        | seed                      | 是                                          |
|                                            | manual_seed               | 是                                          |
|                                            | initial_seed              | 是                                          |
|                                            | get_rng_state             | 是                                          |
|                                            | set_rng_state             | 是                                          |
|                                            | torch.default_generator   | 是                                          |
|                                            | bernoulli                 | 是                                          |
|                                            | multinomial               | 是                                          |
|                                            | normal                    | 是                                          |
|                                            | poisson                   | 否                                          |
|                                            | rand                      | 是                                          |
|                                            | rand_like                 | 是                                          |
|                                            | randint                   | 是                                          |
|                                            | randint_like              | 是                                          |
|                                            | randn                     | 是                                          |
|                                            | randn_like                | 是                                          |
|                                            | randperm                  | 是                                          |
|                                            | torch.Tensor.bernoulli_   | 是                                          |
|                                            | torch.Tensor.cauchy_      | 是                                          |
|                                            | torch.Tensor.exponential_ | 否                                          |
|                                            | torch.Tensor.geometric_   | 否                                          |
|                                            | torch.Tensor.log_normal_  | 否                                          |
|                                            | torch.Tensor.normal_      | 是                                          |
|                                            | torch.Tensor.random_      | 是                                          |
|                                            | torch.Tensor.uniform_     | 是                                          |
|                                            | quasirandom.SobolEngine   | 是                                          |
| **Serialization**                          | save                      | 是                                          |
|                                            | load                      | 是                                          |
| **Parallelism**                            | get_num_threads           | 是                                          |
|                                            | set_num_threads           | 是                                          |
|                                            | get_num_interop_threads   | 是                                          |
|                                            | set_num_interop_threads   | 是                                          |
| **Locally disabling gradient computation** | no_grad                   | 是                                          |
|                                            | enable_grad               | 是                                          |
|                                            | set_grad_enabled          | 是                                          |
| **Math operations**                        | abs                       | 是                                          |
|                                            | acos                      | 是                                          |
|                                            | add                       | 是                                          |
|                                            | addcdiv                   | 是                                          |
|                                            | addcmul                   | 是                                          |
|                                            | angle                     | 否                                          |
|                                            | asin                      | 是                                          |
|                                            | atan                      | 是                                          |
|                                            | atan2                     | 是                                          |
|                                            | bitwise_not               | 是                                          |
|                                            | bitwise_and               | 是                                          |
|                                            | bitwise_or                | 是                                          |
|                                            | bitwise_xor               | 是                                          |
|                                            | ceil                      | 是                                          |
|                                            | clamp                     | 是                                          |
|                                            | conj                      | 否                                          |
|                                            | cos                       | 是                                          |
|                                            | cosh                      | 是                                          |
|                                            | div                       | 是                                          |
|                                            | digamma                   | 否                                          |
|                                            | erf                       | 是                                          |
|                                            | erfc                      | 是                                          |
|                                            | erfinv                    | 是                                          |
|                                            | exp                       | 是                                          |
|                                            | expm1                     | 是                                          |
|                                            | floor                     | 是                                          |
|                                            | floor_divide              | 是                                          |
|                                            | fmod                      | 是                                          |
|                                            | frac                      | 是                                          |
|                                            | imag                      | 否                                          |
|                                            | lerp                      | 是                                          |
|                                            | lgamma                    | 否                                          |
|                                            | log                       | 是                                          |
|                                            | log10                     | 是                                          |
|                                            | log1p                     | 是                                          |
|                                            | log2                      | 是                                          |
|                                            | logical_and               | 是                                          |
|                                            | logical_not               | 是                                          |
|                                            | logical_or                | 是                                          |
|                                            | logical_xor               | 是                                          |
|                                            | mul                       | 是                                          |
|                                            | mvlgamma                  | 否                                          |
|                                            | neg                       | 是                                          |
|                                            | polygamma                 | 否                                          |
|                                            | pow                       | 是                                          |
|                                            | real                      | 是                                          |
|                                            | reciprocal                | 是                                          |
|                                            | remainder                 | 是                                          |
|                                            | round                     | 是                                          |
|                                            | rsqrt                     | 是                                          |
|                                            | sigmoid                   | 是                                          |
|                                            | sign                      | 是                                          |
|                                            | sin                       | 是                                          |
|                                            | sinh                      | 是                                          |
|                                            | sqrt                      | 是                                          |
|                                            | square                    | 是                                          |
|                                            | tan                       | 是                                          |
|                                            | tanh                      | 是                                          |
|                                            | true_divide               | 是                                          |
|                                            | trunc                     | 是                                          |
|                                            | argmax                    | 是                                          |
|                                            | argmin                    | 是                                          |
|                                            | max                       | 是                                          |
|                                            | min                       | 是                                          |
|                                            | dist                      | 是                                          |
|                                            | logsumexp                 | 是                                          |
|                                            | mean                      | 否                                          |
|                                            | median                    | 是                                          |
|                                            | mode                      | 是                                          |
|                                            | norm                      | 是                                          |
|                                            | prod                      | 是                                          |
|                                            | std                       | 是                                          |
|                                            | std_mean                  | 是                                          |
|                                            | sum                       | 是                                          |
|                                            | unique                    | 否                                          |
|                                            | unique_consecutive        | 否                                          |
|                                            | var                       | 是                                          |
|                                            | var_mean                  | 是                                          |
|                                            | allclose                  | 是                                          |
|                                            | argsort                   | 是                                          |
|                                            | eq                        | 是                                          |
|                                            | equal                     | 是                                          |
|                                            | ge                        | 是                                          |
|                                            | gt                        | 是                                          |
|                                            | isfinite                  | 是                                          |
|                                            | isinf                     | 是                                          |
|                                            | isnan                     | 是                                          |
|                                            | kthvalue                  | 是                                          |
|                                            | le                        | 是                                          |
|                                            | lt                        | 是                                          |
|                                            | ne                        | 是                                          |
|                                            | sort                      | 是                                          |
|                                            | topk                      | 是                                          |
|                                            | fft                       | 否                                          |
|                                            | ifft                      | 否                                          |
|                                            | rfft                      | 否                                          |
|                                            | rfft                      | 否                                          |
|                                            | stft                      | 否                                          |
|                                            | bartlett_window           | 是                                          |
|                                            | blackman_window           | 是                                          |
|                                            | hamming_window            | 是                                          |
|                                            | hann_window               | 是                                          |
|                                            | bincount                  | 是                                          |
|                                            | broadcast_tensors         | 是                                          |
|                                            | cartesian_prod            | 是                                          |
|                                            | cdist                     | 是，仅支持mode=donot_use_mm_for_euclid_dist |
|                                            | combinations              | 否                                          |
|                                            | cross                     | 是                                          |
|                                            | cummax                    | 是                                          |
|                                            | cummin                    | 是                                          |
|                                            | cumprod                   | 是                                          |
|                                            | cumsum                    | 是                                          |
|                                            | diag                      | 是，仅支持diagonal=0场景                    |
|                                            | diag_embed                | 是                                          |
|                                            | diagflat                  | 是                                          |
|                                            | diagonal                  | 是                                          |
|                                            | einsum                    | 是                                          |
|                                            | flatten                   | 是                                          |
|                                            | flip                      | 是                                          |
|                                            | rot90                     | 是                                          |
|                                            | histc                     | 否                                          |
|                                            | meshgrid                  | 是                                          |
|                                            | renorm                    | 是                                          |
|                                            | repeat_interleave         | 是                                          |
|                                            | roll                      | 是                                          |
|                                            | tensordot                 | 是                                          |
|                                            | trace                     | 否                                          |
|                                            | tril                      | 是                                          |
|                                            | tril_indices              | 是                                          |
|                                            | triu                      | 是                                          |
|                                            | triu_indices              | 是                                          |
|                                            | addbmm                    | 是                                          |
|                                            | addmm                     | 是                                          |
|                                            | addmv                     | 是                                          |
|                                            | addr                      | 是                                          |
|                                            | baddbmm                   | 是                                          |
|                                            | bmm                       | 是                                          |
|                                            | chain_matmul              | 是                                          |
|                                            | cholesky                  | 否                                          |
|                                            | cholesky_inverse          | 否                                          |
|                                            | cholesky_solve            | 否                                          |
|                                            | dot                       | 是                                          |
|                                            | eig                       | 否                                          |
|                                            | geqrf                     | 否                                          |
|                                            | ger                       | 是                                          |
|                                            | inverse                   | 是                                          |
|                                            | det                       | 否                                          |
|                                            | logdet                    | 否                                          |
|                                            | slogdet                   | 是                                          |
|                                            | lstsq                     | 否                                          |
|                                            | lu                        | 否                                          |
|                                            | lu_solve                  | 否                                          |
|                                            | lu_unpack                 | 否                                          |
|                                            | matmul                    | 是                                          |
|                                            | matrix_power              | 是                                          |
|                                            | matrix_rank               | 是                                          |
|                                            | mm                        | 是                                          |
|                                            | mv                        | 是                                          |
|                                            | orgqr                     | 否                                          |
|                                            | ormqr                     | 否                                          |
|                                            | pinverse                  | 是                                          |
|                                            | qr                        | 是                                          |
|                                            | solve                     | 否                                          |
|                                            | svd                       | 是                                          |
|                                            | svd_lowrank               | 是                                          |
|                                            | pca_lowrank               | 是                                          |
|                                            | symeig                    | 是                                          |
|                                            | lobpcg                    | 否                                          |
|                                            | trapz                     | 是                                          |
|                                            | triangular_solve          | 是                                          |
| **Utilities**                              | compiled_with_cxx11_abi   | 是                                          |
|                                            | result_type               | 是                                          |
|                                            | can_cast                  | 是                                          |
|                                            | promote_types             | 是                                          |


## torch.nn<a name="ZH-CN_TOPIC_0000001448371437"></a>

| 分类                                                    | API名称                             | 是否支持                    |
| ------------------------------------------------------- | ----------------------------------- | --------------------------- |
| **torch.nn**                                            | Parameter                           | 是                          |
| **Containers**                                          | Module                              | 是                          |
|                                                         | Sequential                          | 是                          |
|                                                         | ModuleList                          | 是                          |
|                                                         | ModuleDict                          | 是                          |
|                                                         | ParameterList                       | 是                          |
|                                                         | ParameterDict                       | 是                          |
| **Convolution Layers**                                  | nn.Conv1d                           | 是                          |
|                                                         | nn.Conv2d                           | 是                          |
|                                                         | nn.Conv3d                           | 是                          |
|                                                         | nn.ConvTranspose1d                  | 是                          |
|                                                         | nn.ConvTranspose2d                  | 是                          |
|                                                         | nn.ConvTranspose3d                  | 是                          |
|                                                         | nn.Unfold                           | 是                          |
|                                                         | nn.Fold                             | 是                          |
| **Pooling layers**                                      | nn.MaxPool1d                        | 是                          |
|                                                         | nn.MaxPool2d                        | 是                          |
|                                                         | nn.MaxPool3d                        | 是                          |
|                                                         | nn.MaxUnpool1d                      | 是                          |
|                                                         | nn.MaxUnpool2d                      | 是                          |
|                                                         | nn.MaxUnpool3d                      | 是                          |
|                                                         | nn.AvgPool1d                        | 是                          |
|                                                         | nn.AvgPool2d                        | 是                          |
|                                                         | nn.AvgPool3d                        | 是                          |
|                                                         | nn.FractionalMaxPool2d              | 否                          |
|                                                         | nn.LPPool1d                         | 是                          |
|                                                         | nn.LPPool2d                         | 是                          |
|                                                         | nn.AdaptiveMaxPool1d                | 是                          |
|                                                         | nn.AdaptiveMaxPool2d                | 是                          |
|                                                         | nn.AdaptiveMaxPool3d                | 否                          |
|                                                         | nn.AdaptiveAvgPool1d                | 是                          |
|                                                         | nn.AdaptiveAvgPool2d                | 是                          |
|                                                         | nn.AdaptiveAvgPool3d                | 是，仅支持D=1，H=1，W=1场景 |
| **Padding Layers**                                      | nn.ReflectionPad1d                  | 否                          |
|                                                         | nn.ReflectionPad2d                  | 是                          |
|                                                         | nn.ReplicationPad1d                 | 否                          |
|                                                         | nn.ReplicationPad2d                 | 是                          |
|                                                         | nn.ReplicationPad3d                 | 否                          |
|                                                         | nn.ZeroPad2d                        | 是                          |
|                                                         | nn.ConstantPad1d                    | 是                          |
|                                                         | nn.ConstantPad2d                    | 是                          |
|                                                         | nn.ConstantPad3d                    | 是                          |
| **Non-linear Activations (weighted sum, nonlinearity)** | nn.ELU                              | 是                          |
|                                                         | nn.Hardshrink                       | 是                          |
|                                                         | nn.Hardtanh                         | 是                          |
|                                                         | nn.LeakyReLU                        | 是                          |
|                                                         | nn.LogSigmoid                       | 是                          |
|                                                         | nn.MultiheadAttention               | 是                          |
|                                                         | nn.PReLU                            | 是                          |
|                                                         | nn.ReLU                             | 是                          |
|                                                         | nn.ReLU6                            | 是                          |
|                                                         | nn.RReLU                            | 是                          |
|                                                         | nn.SELU                             | 是                          |
|                                                         | nn.CELU                             | 是                          |
|                                                         | nn.GELU                             | 是                          |
|                                                         | nn.Sigmoid                          | 是                          |
|                                                         | nn.Softplus                         | 是                          |
|                                                         | nn.Softshrink                       | 是                          |
|                                                         | nn.Softsign                         | 是                          |
|                                                         | nn.Tanh                             | 是                          |
|                                                         | nn.Tanhshrink                       | 是                          |
|                                                         | nn.Threshold                        | 是                          |
| **Non-linear Activations (other)**                      | nn.Softmin                          | 是                          |
|                                                         | nn.Softmax                          | 是                          |
|                                                         | nn.Softmax2d                        | 是                          |
|                                                         | nn.LogSoftmax                       | 是                          |
|                                                         | nn.AdaptiveLogSoftmaxWithLoss       | 否                          |
| **Non-linear Activations**                              | nn.BatchNorm1d                      | 是                          |
|                                                         | nn.BatchNorm2d                      | 是                          |
|                                                         | nn.BatchNorm3d                      | 是                          |
|                                                         | nn.GroupNorm                        | 是                          |
|                                                         | nn.SyncBatchNorm                    | 是                          |
|                                                         | nn.InstanceNorm1d                   | 是                          |
|                                                         | nn.InstanceNorm2d                   | 是                          |
|                                                         | nn.InstanceNorm3d                   | 是                          |
|                                                         | nn.LayerNorm                        | 是                          |
|                                                         | nn.LocalResponseNorm                | 是                          |
| **Recurrent Layers**                                    | nn.RNNBase                          | 是                          |
|                                                         | nn.RNN                              | 是                          |
|                                                         | nn.LSTM                             | 是                          |
|                                                         | nn.GRU                              | 是                          |
|                                                         | nn.RNNCell                          | 是，非16对齐场景暂不支持    |
|                                                         | nn.LSTMCell                         | 是                          |
|                                                         | nn.GRUCell                          | 是                          |
| **Transformer Layers**                                  | nn.Transformer                      | 是                          |
|                                                         | nn.TransformerEncoder               | 是                          |
|                                                         | nn.TransformerDecoder               | 是                          |
|                                                         | nn.TransformerEncoderLayer          | 是                          |
|                                                         | nn.TransformerDecoderLayer          | 是                          |
| **Linear Layers**                                       | nn.Identity                         | 是                          |
|                                                         | nn.Linear                           | 是                          |
|                                                         | nn.Bilinear                         | 是                          |
|                                                         | nn.LazyLinear                       | 是                          |
| **Dropout Layers**                                      | nn.Dropout                          | 是                          |
|                                                         | nn.Dropout2d                        | 是                          |
|                                                         | nn.Dropout3d                        | 是                          |
|                                                         | nn.AlphaDropout                     | 是                          |
| **Sparse Layers**                                       | nn.Embedding                        | 是                          |
|                                                         | nn.EmbeddingBag                     | 是                          |
| **Distance Functions**                                  | nn.CosineSimilarity                 | 是                          |
|                                                         | nn.PairwiseDistance                 | 是                          |
| **Loss Functions**                                      | nn.L1Loss                           | 是                          |
|                                                         | nn.MSELoss                          | 是                          |
|                                                         | nn.CrossEntropyLoss                 | 是                          |
|                                                         | nn.CTCLoss                          | 是                          |
|                                                         | nn.NLLLoss                          | 是                          |
|                                                         | nn.PoissonNLLLoss                   | 是                          |
|                                                         | nn.KLDivLoss                        | 是                          |
|                                                         | nn.BCELoss                          | 是                          |
|                                                         | nn.BCEWithLogitsLoss                | 是                          |
|                                                         | nn.MarginRankingLoss                | 是                          |
|                                                         | nn.HingeEmbeddingLoss               | 是                          |
|                                                         | nn.MultiLabelMarginLoss             | 是                          |
|                                                         | nn.SmoothL1Loss                     | 是                          |
|                                                         | nn.SoftMarginLoss                   | 是                          |
|                                                         | nn.MultiLabelSoftMarginLoss         | 是                          |
|                                                         | nn.CosineEmbeddingLoss              | 是                          |
|                                                         | nn.MultiMarginLoss                  | 否                          |
|                                                         | nn.TripletMarginLoss                | 是                          |
| **Vision Layers**                                       | nn.PixelShuffle                     | 是                          |
|                                                         | nn.Upsample                         | 是                          |
|                                                         | nn.UpsamplingNearest2d              | 是                          |
|                                                         | nn.UpsamplingBilinear2d             | 是                          |
| **DataParallel Layers (multi-GPU, distributed)**        | nn.DataParallel                     | 否                          |
|                                                         | nn.parallel.DistributedDataParallel | 是                          |
| **Utilities**                                           | clip_grad_norm_                     | 是                          |
|                                                         | clip_grad_value_                    | 是                          |
|                                                         | parameters_to_vector                | 是                          |
|                                                         | vector_to_parameters                | 是                          |
|                                                         | prune.BasePruningMethod             | 是                          |
|                                                         | prune.PruningContainer              | 是                          |
|                                                         | prune.Identity                      | 是                          |
|                                                         | prune.RandomUnstructured            | 是                          |
|                                                         | prune.L1Unstructured                | 是                          |
|                                                         | prune.RandomStructured              | 是                          |
|                                                         | prune.LnStructured                  | 是                          |
|                                                         | prune.CustomFromMask                | 是                          |
|                                                         | prune.identity                      | 是                          |
|                                                         | prune.random_unstructured           | 是                          |
|                                                         | prune.l1_unstructured               | 是                          |
|                                                         | prune.random_structured             | 是                          |
|                                                         | prune.ln_structured                 | 是                          |
|                                                         | prune.global_unstructured           | 是                          |
|                                                         | prune.custom_from_mask              | 是                          |
|                                                         | prune.remove                        | 是                          |
|                                                         | prune.is_pruned                     | 是                          |
|                                                         | weight_norm                         | 是                          |
|                                                         | remove_weight_norm                  | 是                          |
|                                                         | spectral_norm                       | 是                          |
|                                                         | remove_spectral_norm                | 是                          |
|                                                         | nn.utils.rnn.PackedSequence         | 是                          |
|                                                         | nn.utils.rnn.pack_padded_sequence   | 是                          |
|                                                         | nn.utils.rnn.pad_packed_sequence    | 否                          |
|                                                         | nn.utils.rnn.pad_sequence           | 是                          |
|                                                         | nn.utils.rnn.pack_sequence          | 否                          |
|                                                         | nn.Flatten                          | 是                          |


## torch.nn.functional<a name="ZH-CN_TOPIC_0000001398051798"></a>

| 分类                                | API名称                          | 是否支持                    |
| ----------------------------------- | -------------------------------- | --------------------------- |
| **Convolution functions**           | conv1d                           | 是                          |
|                                     | conv2d                           | 是                          |
|                                     | conv3d                           | 是                          |
|                                     | conv_transpose1d                 | 是                          |
|                                     | conv_transpose2d                 | 是                          |
|                                     | conv_transpose3d                 | 是                          |
|                                     | unfold                           | 是                          |
|                                     | fold                             | 是                          |
| **Pooling functions**               | avg_pool1d                       | 是                          |
|                                     | avg_pool2d                       | 是                          |
|                                     | avg_pool3d                       | 是                          |
|                                     | max_pool1d                       | 是                          |
|                                     | max_pool2d                       | 是                          |
|                                     | max_pool3d                       | 是                          |
|                                     | max_unpool1d                     | 是                          |
|                                     | max_unpool2d                     | 是                          |
|                                     | max_unpool3d                     | 是                          |
|                                     | lp_pool1d                        | 是                          |
|                                     | lp_pool2d                        | 是                          |
|                                     | adaptive_max_pool1d              | 是                          |
|                                     | adaptive_max_pool2d              | 是                          |
|                                     | adaptive_max_pool3d              | 否                          |
|                                     | adaptive_avg_pool1d              | 是                          |
|                                     | adaptive_avg_pool2d              | 是                          |
|                                     | adaptive_avg_pool3d              | 是，仅支持D=1，H=1，W=1场景 |
| **Non-linear activation functions** | threshold                        | 是                          |
|                                     | threshold_                       | 是                          |
|                                     | relu                             | 是                          |
|                                     | relu_                            | 是                          |
|                                     | hardtanh                         | 是                          |
|                                     | hardtanh_                        | 是                          |
|                                     | relu6                            | 是                          |
|                                     | elu                              | 是                          |
|                                     | elu_                             | 是                          |
|                                     | selu                             | 是                          |
|                                     | celu                             | 是                          |
|                                     | leaky_relu                       | 是                          |
|                                     | leaky_relu_                      | 是                          |
|                                     | prelu                            | 是                          |
|                                     | rrelu                            | 是                          |
|                                     | rrelu_                           | 是                          |
|                                     | glu                              | 是                          |
|                                     | gelu                             | 是                          |
|                                     | logsigmoid                       | 是                          |
|                                     | hardshrink                       | 是                          |
|                                     | tanhshrink                       | 是                          |
|                                     | softsign                         | 是                          |
|                                     | softplus                         | 是                          |
|                                     | softmin                          | 是                          |
|                                     | softmax                          | 是                          |
|                                     | softshrink                       | 是                          |
|                                     | gumbel_softmax                   | 否                          |
|                                     | log_softmax                      | 是                          |
|                                     | tanh                             | 是                          |
|                                     | sigmoid                          | 是                          |
| **Normalization functions**         | batch_norm                       | 是                          |
|                                     | instance_norm                    | 是                          |
|                                     | layer_norm                       | 是                          |
|                                     | local_response_norm              | 是                          |
|                                     | normalize                        | 是                          |
| **Linear functions**                | linear                           | 是                          |
|                                     | bilinear                         | 是                          |
| **Dropout functions**               | dropout                          | 是                          |
|                                     | alpha_dropout                    | 是                          |
|                                     | dropout2d                        | 是                          |
|                                     | dropout3d                        | 是                          |
| **Sparse functions**                | embedding                        | 是                          |
|                                     | embedding_bag                    | 是                          |
|                                     | one_hot                          | 是                          |
| **Distance functions**              | pairwise_distance                | 是                          |
|                                     | cosine_similarity                | 是                          |
|                                     | pdist                            | 是                          |
| **Loss functions**                  | binary_cross_entropy             | 是                          |
|                                     | binary_cross_entropy_with_logits | 是                          |
|                                     | poisson_nll_loss                 | 是                          |
|                                     | cosine_embedding_loss            | 是                          |
|                                     | cross_entropy                    | 是                          |
|                                     | ctc_loss                         | 是，仅支持2维输入           |
|                                     | hinge_embedding_loss             | 是                          |
|                                     | kl_div                           | 是                          |
|                                     | l1_loss                          | 是                          |
|                                     | mse_loss                         | 是                          |
|                                     | margin_ranking_loss              | 是                          |
|                                     | multilabel_margin_loss           | 是                          |
|                                     | multilabel_soft_margin_loss      | 是                          |
|                                     | multi_margin_loss                | 否                          |
|                                     | nll_loss                         | 是                          |
|                                     | smooth_l1_loss                   | 是                          |
|                                     | soft_margin_loss                 | 是                          |
|                                     | triplet_margin_loss              | 是                          |
| **Vision functions**                | pixel_shuffle                    | 是                          |
|                                     | pad                              | 是                          |
|                                     | interpolate                      | 是                          |
|                                     | upsample                         | 是                          |
|                                     | upsample_nearest                 | 是                          |
|                                     | upsample_bilinear                | 是                          |
|                                     | grid_sample                      | 是                          |
|                                     | affine_grid                      | 是                          |
| **DataParallel functions**          | torch.nn.parallel.data_parallel  | 否                          |


## torch.distributed<a name="ZH-CN_TOPIC_0000001398531486"></a>

| 分类                                                         | API名称                               | 是否支持 |
| ------------------------------------------------------------ | ------------------------------------- | -------- |
| **Initialization**                                           | torch.distributed.init_process_group  | 是       |
|                                                              | torch.distributed.Backend             | 是       |
|                                                              | torch.distributed.get_backend         | 是       |
|                                                              | torch.distributed.get_rank            | 是       |
|                                                              | torch.distributed.get_world_size      | 是       |
|                                                              | torch.distributed.is_initialized      | 是       |
|                                                              | torch.distributed.is_mpi_available    | 是       |
|                                                              | torch.distributed.is_nccl_available   | 是       |
| **Groups**                                                   | torch.distributed.new_group           | 是       |
| **Point-to-point communication**                             | torch.distributed.send                | 否       |
|                                                              | torch.distributed.recv                | 否       |
|                                                              | torch.distributed.isend               | 否       |
|                                                              | torch.distributed.irecv               | 否       |
| **Synchronous <br/>and <br/>asynchronous collective<br/>operations** | is_completed                          | 是       |
|                                                              | wait                                  | 是       |
| **Collective functions**                                     | torch.distributed.broadcast           | 是       |
|                                                              | torch.distributed.all_reduce          | 是       |
|                                                              | torch.distributed.reduce              | 否       |
|                                                              | torch.distributed.all_gather          | 是       |
|                                                              | torch.distributed.gather              | 否       |
|                                                              | torch.distributed.scatter             | 否       |
|                                                              | torch.distributed.barrier             | 是       |
|                                                              | torch.distributed.ReduceOp            | 是       |
|                                                              | torch.distributed.reduce_op           | 是       |
| **Multi-GPU collective functions**                           | torch.distributed.broadcast_multigpu  | 否       |
|                                                              | torch.distributed.all_reduce_multigpu | 否       |
|                                                              | torch.distributed.reduce_multigpu     | 否       |
|                                                              | torch.distributed.all_gather_multigpu | 否       |
| **Launch utility**                                           | torch.distributed.launch              | 是       |
| **Spawn utility**                                            | torch.multiprocessing.spawn           | 是       |


## torch.cuda<a name="ZH-CN_TOPIC_0000001398371514"></a>

| 分类                              | API名称                               | npu对应API名称                       | 是否支持                                           |
| --------------------------------- | ------------------------------------- | ------------------------------------ | -------------------------------------------------- |
| **torch.cuda**                    | torch.cuda.current_blas_handle        | torch.npu.current_blas_handle        | 否                                                 |
|                                   | torch.cuda.current_device             | torch.npu.current_device             | 是                                                 |
|                                   | torch.cuda.current_stream             | torch.npu.current_stream             | 是                                                 |
|                                   | torch.cuda.default_stream             | torch.npu.default_stream             | 是                                                 |
|                                   | torch.cuda.device                     | torch.npu.device                     | 是                                                 |
|                                   | torch.cuda.device_count               | torch.npu.device_count               | 是                                                 |
|                                   | torch.cuda.device_of                  | torch.npu.device_of                  | 是                                                 |
|                                   | torch.cuda.get_device_capability      | torch.npu.get_device_capability      | 否                                                 |
|                                   | torch.cuda.get_device_name            | torch.npu.get_device_name            | 否                                                 |
|                                   | torch.cuda.init                       | torch.npu.init                       | 是                                                 |
|                                   | torch.cuda.ipc_collect                | torch.npu.ipc_collect                | 否                                                 |
|                                   | torch.cuda.is_available               | torch.npu.is_available               | 是                                                 |
|                                   | torch.cuda.is_initialized             | torch.npu.is_initialized             | 是                                                 |
|                                   | torch.cuda.set_device                 | torch.npu.set_device                 | 是，只支持程序开始时指定device，不支持切换device。 |
|                                   | torch.cuda.stream                     | torch.npu.stream                     | 是                                                 |
|                                   | torch.cuda.synchronize                | torch.npu.synchronize                | 是                                                 |
| **Communication collectives**     | torch.cuda.comm.broadcast             | torch.npu.comm.broadcast             | 否                                                 |
|                                   | torch.cuda.comm.broadcast_coalesced   | torch.npu.comm.broadcast_coalesced   | 否                                                 |
|                                   | torch.cuda.comm.reduce_add            | torch.npu.comm.reduce_add            | 否                                                 |
|                                   | torch.cuda.comm.scatter               | torch.npu.comm.scatter               | 否                                                 |
|                                   | torch.cuda.comm.gather                | torch.npu.comm.gather                | 否                                                 |
| **Streams and events**            | torch.cuda.Stream                     | torch.npu.Stream                     | 是                                                 |
|                                   | torch.cuda.Event                      | torch.npu.Event                      | 是，不支持from_ipc_handle和ipc_handle              |
| **Memory management**             | torch.cuda.empty_cache                | torch.npu.empty_cache                | 是                                                 |
|                                   | torch.cuda.memory_stats               | torch.npu.memory_stats               | 是                                                 |
|                                   | torch.cuda.memory_summary             | torch.npu.memory_summary             | 是                                                 |
|                                   | torch.cuda.memory_snapshot            | torch.npu.memory_snapshot            | 是                                                 |
|                                   | torch.cuda.memory_allocated           | torch.npu.memory_allocated           | 是                                                 |
|                                   | torch.cuda.max_memory_allocated       | torch.npu.max_memory_allocated       | 是                                                 |
|                                   | torch.cuda.reset_max_memory_allocated | torch.npu.reset_max_memory_allocated | 是                                                 |
|                                   | torch.cuda.memory_reserved            | torch.npu.memory_reserved            | 是                                                 |
|                                   | torch.cuda.max_memory_reserved        | torch.npu.max_memory_reserved        | 是                                                 |
|                                   | torch.cuda.memory_cached              | torch.npu.memory_cached              | 是                                                 |
|                                   | torch.cuda.max_memory_cached          | torch.npu.max_memory_cached          | 是                                                 |
|                                   | torch.cuda.reset_max_memory_cached    | torch.npu.reset_max_memory_cached    | 是                                                 |
| **NVIDIA Tools Extension (NVTX)** | torch.cuda.nvtx.mark                  | torch.npu.nvtx.mark                  | 否                                                 |
|                                   | torch.cuda.nvtx.range_push            | torch.npu.nvtx.range_push            | 否                                                 |
|                                   | torch.cuda.nvtx.range_pop             | torch.npu.nvtx.range_pop             | 否                                                 |

## torch.cuda.amp<a name="ZH-CN_TOPIC_0000001448251789"></a>

| API名称                   | npu对应API名称           | 是否支持 |
| ------------------------- | ------------------------ | -------- |
| torch.cuda.amp.GradScaler | torch.npu.amp.GradScaler | 否       |

## NPU自定义算子

| 序号 | 算子名称                              |
| ---- | ------------------------------------- |
| 1    | npu_convolution_transpose             |
| 2    | npu_conv_transpose2d                  |
| 3    | npu_convolution                       |
| 4    | npu_conv2d                            |
| 5    | npu_conv3d                            |
| 6    | one_                                  |
| 7    | npu_sort_v2                           |
| 8    | npu_format_cast                       |
| 9    | npu_format_cast_.src                  |
| 10   | npu_transpose                         |
| 11   | npu_broadcast                         |
| 12   | npu_dtype_cast                        |
| 13   | empty_with_format                     |
| 14   | copy_memory_                          |
| 15   | npu_one_hot                           |
| 16   | npu_stride_add                        |
| 17   | npu_softmax_cross_entropy_with_logits |
| 18   | npu_ps_roi_pooling                    |
| 19   | npu_roi_align                         |
| 20   | npu_nms_v4                            |
| 21   | npu_lstm                              |
| 22   | npu_iou                               |
| 23   | npu_ptiou                             |
| 24   | npu_nms_with_mask                     |
| 25   | npu_pad                               |
| 26   | npu_bounding_box_encode               |
| 27   | npu_bounding_box_decode               |
| 28   | npu_gru                               |
| 29   | npu_random_choice_with_mask           |
| 30   | npu_batch_nms                         |
| 31   | npu_slice                             |
| 32   | npu_dropoutV2                         |
| 33   | _npu_dropout                          |
| 34   | _npu_dropout_inplace                  |
| 35   | npu_indexing                          |
| 36   | npu_ifmr                              |
| 37   | npu_max.dim                           |
| 38   | npu_scatter                           |
| 39   | npu_apply_adam                        |
| 40   | npu_layer_norm_eval                   |
| 41   | npu_alloc_float_status                |
| 42   | npu_get_float_status                  |
| 43   | npu_clear_float_status                |
| 44   | npu_confusion_transpose               |
| 45   | npu_bmmV2                             |
| 46   | fast_gelu                             |
| 47   | npu_sub_sample                        |
| 48   | npu_deformable_conv2d                 |
| 49   | npu_mish                              |
| 50   | npu_anchor_response_flags             |
| 51   | npu_yolo_boxes_encode                 |
| 52   | npu_grid_assign_positive              |
| 53   | npu_normalize_batch                   |
| 54   | npu_masked_fill_range                 |
| 55   | npu_linear                            |
| 56   | npu_bert_apply_adam                   |
| 57   | npu_giou                              |
| 58   | npu_min.dim                           |
| 59   | npu_nms_rotated                       |
| 60   | npu_silu                              |
| 61   | npu_reshape                           |
| 62   | npu_rotated_iou                       |
| 63   | npu_rotated_box_encode                |
| 64   | npu_rotated_box_decode                |

详细算子接口说明：

> npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out = (var, m, v))

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

> npu_convolution_transpose(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor

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

> npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor

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

> npu_convolution(input, weight, bias, stride, padding, dilation, groups) -> Tensor

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

> npu_conv2d(input, weight, bias, stride, padding, dilation, groups) -> Tensor

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

> npu_conv3d(input, weight, bias, stride, padding, dilation, groups) -> Tensor

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

> one_(self) -> Tensor

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

> npu_sort_v2(self, dim=-1, descending=False, out=None) -> Tensor

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
  >>> sorted_x = torch.npu_sort_v2(x)
  >>> sorted_x
  tensor([[-1.7217, -0.0067,  0.5029,  1.7793],
          [-1.0488, -0.2937,  1.1689,  1.3242],
          [-2.7441,  0.1880,  0.7378,  1.3975]], device='npu:0')
  ```

> npu_format_cast(self, acl_format) -> Tensor

Change the format of a npu tensor.

- Parameters：
  - **self** (Tensor) - the input tensor
  - **acl_format** (int) - the target format to transform

- constraints：

  None

- Examples：

  ```python
  >>> x = torch.rand(2, 3, 4, 5).npu()
  >>> x.storage().npu_format()
  0
  >>> x1 = x.npu_format_cast(29)
  >>> x1.storage().npu_format()
  29
  ```

>   npu_format_cast_.src(self, src) -> Tensor

  In-place Change the format of self, with the same format as src.

  - Parameters：
    - **self** (Tensor) - the input tensor
    - **src** (Tensor) - the target format to transform

  - constraints：

    None

  - Examples：

    ```python
    >>> x = torch.rand(2, 3, 4, 5).npu()
    >>> x.storage().npu_format()
    0
    >>> x.npu_format_cast_(29).storage().npu_format()
    29
    ```

> npu_transpose(self, perm) -> Tensor

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
  >>> x1 = torch.npu_transpose(x, (2, 0, 1))
  >>> x1.shape
  torch.Size([5, 2, 3])
  >>> x2 = x.npu_transpose(2, 0, 1)
  >>> x2.shape
  torch.Size([5, 2, 3])
  ```

> npu_broadcast(self, perm) -> Tensor

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
  
> npu_dtype_cast(input, dtype) -> Tensor

Performs Tensor dtype conversion.

- Parameters：
  - **input** (Tensor) - the input tensor.
  - **dtype** (torch.dtype) - the desired data type of returned Tensor.

- constraints：

  None

- Examples：

  ```python
  >>> torch. npu_dtype_cast (torch.tensor([0, 0.5, -1.]).npu(), dtype=torch.int)
  tensor([ 0,  0, -1], device='npu:0', dtype=torch.int32)
  ```

> empty_with_format(size, dtype, layout, device, pin_memory, acl_format) -> Tensor

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
  >>> torch.empty_with_format((2, 3), dtype=torch.float32, device="npu")
  tensor([[1., 1., 1.],
          [1., 1., 1.]], device='npu:0')
  ```

> copy_memory_(dst, src, non_blocking=False) -> Tensor

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

> npu_one_hot(input, num_classes=-1, depth=1, on_value=1, off_value=0) -> Tensor

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
  >>> b=torch.npu_one_hot(a, depth=5)
  >>> b
  tensor([[0., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0.],
          [0., 0., 1., 0., 0.],
          [0., 1., 0., 0., 0.]], device='npu:0')
  ```

> npu_stride_add(x1, x2, offset1, offset2, c1_len) -> Tensor

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
  >>> b=torch.npu_stride_add(a, a, 0, 0, 1)
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

> npu_softmax_cross_entropy_with_logits(features, labels) -> Tensor

Computes softmax cross entropy cost.

- Parameters：
  - **features** (Tensor) - A Tensor.  A "batch_size * num_classes" matrix. 
  - **labels** (Tensor) - A Tensor of the same type as "features". A "batch_size * num_classes" matrix. 

- constraints：

  None

- Examples：

  None

> npu_ps_roi_pooling(x, rois, spatial_scale, group_size, output_dim) -> Tensor

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
  >>> out = torch.npu_ps_roi_pooling(x, roi, 0.5, 2, 2)
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

> npu_roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode) -> Tensor

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
  >>> out = torch.npu_roi_align(x, rois, 0.25, 3, 3, 2, 0)
  >>> out
  tensor([[[[ 4.5000,  6.5000,  8.5000],
            [16.5000, 18.5000, 20.5000],
            [28.5000, 30.5000, 32.5000]]]], device='npu:0')
  ```

> npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size=False) -> (Tensor, Tensor)

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
  >>> npu_output = torch.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold)
  >>> npu_output
  (tensor([57, 65, 25, 45, 43, 12, 52, 91, 23, 78, 53, 11, 24, 62, 22, 67,  9, 94,
          54, 92], device='npu:0', dtype=torch.int32), tensor(20, device='npu:0', dtype=torch.int32))
  ```

> npu_nms_rotated(dets, scores, iou_threshold, scores_threshold=0, max_output_size=-1, mode=0) -> (Tensor, Tensor)

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
  >>> output1, output2 = torch.npu_nms_rotated(dets, scores, 0.2, 0, -1, 1)
  >>> output1
  tensor([76, 48, 15, 65, 91, 82, 21, 96, 62, 90, 13, 59,  0, 18, 47, 23,  8, 56,
          55, 63, 72, 39, 97, 81, 16, 38, 17, 25, 74, 33, 79, 44, 36, 88, 83, 37,
          64, 45, 54, 41, 22, 28, 98, 40, 30, 20,  1, 86, 69, 57, 43,  9, 42, 27,
          71, 46, 19, 26, 78, 66,  3, 52], device='npu:0', dtype=torch.int32)
  >>> output2
  tensor([62], device='npu:0', dtype=torch.int32)
  ```

> npu_lstm(x, weight, bias, seq_len, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction)

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

>npu_iou(bboxes, gtboxes, mode=0) -> Tensor
>npu_ptiou(bboxes, gtboxes, mode=0) -> Tensor

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
  >>> output_iou = torch.npu_iou(bboxes, gtboxes, 0)
  >>> output_iou
  tensor([[0.4985, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.9961, 0.0000]], device='npu:0', dtype=torch.float16)
  ```

>npu_pad(input, paddings) -> Tensor

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
  >>> output = torch.npu_pad(input, paddings)
  >>> output
  tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
          [ 0., 20., 20., 10., 10.,  0.],
          [ 0.,  0.,  0.,  0.,  0.,  0.]], device='npu:0', dtype=torch.float16)
  ```

>npu_nms_with_mask(input, iou_threshold) -> (Tensor, Tensor, Tensor)

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
  >>> output1, output2, output3, = torch.npu_nms_with_mask(input, iou_threshold)
  >>> output1
  tensor([[0.0000, 1.0000, 2.0000, 3.0000, 0.6001],
          [6.0000, 7.0000, 8.0000, 9.0000, 0.3999]], device='npu:0',
        dtype=torch.float16)
  >>> output2
  tensor([0, 1], device='npu:0', dtype=torch.int32)
  >>> output3
  tensor([1, 1], device='npu:0', dtype=torch.uint8)
  ```

>npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3) -> Tensor

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
  >>> output = torch.npu_bounding_box_encode(anchor_box, ground_truth_box, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)
  >>> output
  tensor([[13.3281, 13.3281,  0.0000,  0.0000],
          [13.3281,  6.6641,  0.0000, -5.4922]], device='npu:0')
  >>>
  ```

>npu_bounding_box_decode(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip) -> Tensor

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
  >>> output = torch.npu_bounding_box_decode(rois, deltas, 0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)
  >>> output
  tensor([[2.5000, 6.5000, 9.0000, 9.0000],
          [9.0000, 9.0000, 9.0000, 9.0000]], device='npu:0')
  ```

>npu_gru(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)

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

>npu_random_choice_with_mask(x, count=256, seed=0, seed2=0) -> (Tensor, Tensor)

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
  >>> result, mask = torch.npu_random_choice_with_mask(x, 2, 1, 0)
  >>> result
  tensor([[0],
          [2]], device='npu:0', dtype=torch.int32)
  >>> mask
  tensor([True, True], device='npu:0')
  ```

>npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame=False, transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)

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
  >>> nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch.npu_batch_nms(boxes, scores, 0.3, 0.5, 3, 4)
  >>> nmsed_boxes
  >>> nmsed_scores
  >>> nmsed_classes
  >>> nmsed_num
  ```

>npu_slice(self, offsets, size) -> Tensor

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
  >>> output = torch.npu_slice(input, offsets, size)
  >>> output
  tensor([[1., 2.],
          [6., 7.]], device='npu:0', dtype=torch.float16)
  ```
  
>npu_dropoutV2(self, seed, p) -> (Tensor, Tensor, Tensor(a!))

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
  >>> output, mask, out_seed = torch.npu_dropoutV2(input, seed, prob)
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

>_npu_dropout(self, p) -> (Tensor, Tensor)

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
  >>> output, mask = torch._npu_dropout(input, prob)
  >>> output
  tensor([0.0000, 2.8571, 0.0000, 0.0000], device='npu:0')
  >>> mask
  tensor([ 98, 255, 188, 186, 120, 157, 175, 159,  77, 223, 127,  79, 247, 151,
        253, 255], device='npu:0', dtype=torch.uint8)
  ```

>_npu_dropout_inplace(result, p) -> (Tensor(a!), Tensor)

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
  >>> output, mask = torch._npu_dropout_inplace(input, prob)
  >>> output
  tensor([0.0000, 2.8571, 0.0000, 0.0000], device='npu:0')
  >>> input
  tensor([0.0000, 2.8571, 4.2857, 5.7143], device='npu:0')
  >>> mask
  tensor([ 98, 255, 188, 186, 120, 157, 175, 159,  77, 223, 127,  79, 247, 151,
        253, 255], device='npu:0', dtype=torch.uint8)
  ```

>npu_indexing(self, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0) -> Tensor

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
  >>> output = torch.npu_indexing(input1, [0, 0], [2, 2], [1, 1])
  >>> output
  tensor([[1, 2],
        [5, 6]], device='npu:0', dtype=torch.int32)
  ```

>npu_ifmr(Tensor data, Tensor data_min, Tensor data_max, Tensor cumsum, float min_percentile, float max_percentile, float search_start, float search_end, float search_step, bool with_offset) -> (Tensor, Tensor)

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
    >>> scale, offset = torch.npu_ifmr(input,
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

>npu_max.dim(self, dim, keepdim=False) -> (Tensor, Tensor)

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
  >>> outputs, indices = torch.npu_max(input, 2)
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

>npu_min.dim(self, dim, keepdim=False) -> (Tensor, Tensor)

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
  >>> outputs, indices = torch.npu_min(input, 2)
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

>npu_scatter(self, indices, updates, dim) -> Tensor

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
  >>> output = torch.npu_scatter(input, indices, updates, dim)
  >>> output
  tensor([[-1.1993,  0.1226],
          [ 0.9041, -1.5247]], device='npu:0')
  ```

>npu_layer_norm_eval(input, normalized_shape, weight=None, bias=None, eps=1e-05) -> Tensor

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
  >>> output = torch.npu_layer_norm_eval(input, normalized_shape, weight, bias, 1e-5)
  >>> output
  tensor([[        nan,  6.7474e-41,  8.3182e-20,  2.0687e-40],
          [        nan,  8.2494e-41, -9.9784e-20, -8.2186e-41],
          [        nan, -2.6695e-41, -7.7173e-20,  2.1353e-41],
          [        nan, -1.3497e-41, -7.1281e-20, -6.9827e-42],
          [        nan,  3.5663e-41,  1.2002e-19,  1.4314e-40],
          [        nan, -6.2792e-42,  1.7902e-20,  2.1050e-40]], device='npu:0')
  ```

>npu_alloc_float_status(self) -> Tensor

Produces eight numbers with a value of zero

- Parameters：
  
  - **self** (Tensor) - Any Tensor
  
- constraints：

  None

- Examples：

  ```python
  >>> input    = torch.randn([1,2,3]).npu()
  >>> output = torch.npu_alloc_float_status(input)
  >>> input
  tensor([[[ 2.2324,  0.2478, -0.1056],
          [ 1.1273, -0.2573,  1.0558]]], device='npu:0')
  >>> output
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
  ```

> npu_get_float_status(self) -> Tensor

Computes NPU get float status operator function.

- Parameters：
  
  - **self** (Tensor) -  A Tensor of data memory address. Must be float32 .
  
- Constraints：

  None

- Examples：
  
  ```python
  >>> x = torch.rand(2).npu()
  >>> torch.npu_get_float_status(x)
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
  ```

> npu_clear_float_status(self) -> Tensor

Set the value of address 0x40000 to 0 in each core.

- Parameters：
  
  - **self** (Tensor) -  A tensor of type float32.
  
- Constraints：

  None

- Examples：

  ```python
  >>> x = torch.rand(2).npu()
  >>> torch.npu_clear_float_status(x)
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
  ```

> npu_confusion_transpose(self, perm, shape, transpose_first) -> Tensor

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
  >>> y = torch.npu_confusion_transpose(x, (0, 2, 1, 3), (2, 4, 18), True)
  >>> y.shape
  torch.Size([2, 4, 18])
  >>> y2 = torch.npu_confusion_transpose(x, (0, 2, 1), (2, 12, 6), False)
  >>> y2.shape
  torch.Size([2, 6, 12])
  ```

> npu_bmmV2(self, mat2, output_sizes) -> Tensor

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
  >>> res = torch.npu_bmmV2(mat1, mat2, [])
  >>> res.shape
  torch.Size([10, 3, 5])
  ```

> fast_gelu(self) -> Tensor

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
  >>> torch.fast_gelu(x)
  tensor([0.4403, 0.2733], device='npu:0')
  ```

> npu_sub_sample(self, per_images, positive_fraction) -> Tensor

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
  >>> torch.npu_sub_sample(x, 5, 0.6)
  tensor([-1, -1, -1, -1, -1, -1,  1, -1, -1, -1], device='npu:0',
        dtype=torch.int32)
  ```

> npu_deformable_conv2d(self, weight, offset, bias, kernel_size, stride, padding, dilation=[1,1,1,1], groups=1, deformable_groups=1, modulated=True) -> (Tensor, Tensor)

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
  >>> output, _ = torch.npu_deformable_conv2d(x, weight, offset, None, kernel_size=[5, 5], stride = [1, 1, 1, 1], padding = [2, 2, 2, 2])
  >>> output.shape
  torch.Size([16, 32, 32, 32])
  ```
  
> npu_mish(self) -> Tensor

Computes hyperbolic tangent of "x" element-wise.

- Parameters：

  - **self** (Tensor) -  A Tensor. Must be one of the following types: float16, float32.
  
- Constraints：

  None

- Examples：

  ```python
  >>> x = torch.rand(10, 30, 10).npu()
  >>> y = torch.npu_mish(x)
  >>> y.shape
  torch.Size([10, 30, 10])
  ```
  
> npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors) -> Tensor

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
  >>> y = torch.npu_anchor_response_flags(x, [60, 60], [2, 2], 9)
  >>> y.shape
  torch.Size([32400])
  ```
  
> npu_yolo_boxes_encode(self, gt_bboxes, stride, performance_mode=False) -> Tensor

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
  >>> output = torch.npu_yolo_boxes_encode(anchor_boxes, gt_bboxes, stride, False)
  >>> output.shape
  torch.Size([2, 4])
  ```
  
> npu_grid_assign_positive(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all) -> Tensor

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
  >>> output = torch.npu_grid_assign_positive(assigned_gt_inds, overlaps, box_responsible_flags, max_overlap, argmax_overlap, gt_max_overlaps, gt_argmax_overlaps, 128, 0.5, 0., True)
  >>> output.shape
  torch.Size([4])
  ```

> npu_normalize_batch(self, seq_len, normalize_type=0) -> Tensor

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
  >>> out = torch.npu_normalize_batch(x, seqlen, 0)
  >>> out
  tensor([[[ 1.1496, -0.6685, -0.4812,  1.7611, -0.5187,  0.7571],
          [ 1.1445, -0.4393, -0.7051,  1.0474, -0.2646, -0.1582],
          [ 0.1477,  0.9179, -1.0656, -6.8692, -6.7437,  2.8621]],
  
          [[-0.6880,  0.1337,  1.3623, -0.8081, -1.2291, -0.9410],
          [ 0.3070,  0.5489, -1.4858,  0.6300,  0.6428,  0.0433],
          [-0.5387,  0.8204, -1.1401,  0.8584, -0.3686,  0.8444]]],
        device='npu:0')
  ```

> npu_masked_fill_range(self, start, end, value, axis=-1) -> Tensor

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
  >>> out = torch.npu_masked_fill_range(a, start, end, value, 1)
  >>> out
  tensor([[1.0000, 0.4919, 0.2874, 0.6560],
          [0.6691, 1.0000, 0.0330, 0.1006],
          [0.3888, 0.7011, 1.0000, 0.7878],
          [0.0366, 0.9738, 0.4689, 0.0979]], device='npu:0')
  ```

> npu_linear(input, weight, bias=None) -> Tensor

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
  >>> output = torch.npu_linear(x, w, b)
  >>> output
  tensor([[3.6335, 4.3713, 2.4440, 2.0081],
          [5.3273, 6.3089, 3.9601, 3.2410]], device='npu:0')
  ```

> npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size=None, adam_mode=0, *, out=（var,m,v）)

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
  >>> var_out, m_out, v_out = torch.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, out=(var_in, m_in, v_in))
  >>> var_out
  tensor([ 14.7733, -30.1218,  -1.3647,  ..., -16.6840,   7.1518,   8.4872],
        device='npu:0')
  ```

> npu_giou(self, gtboxes, trans=False, is_cross=False, mode=0) -> Tensor

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
  >>> output = torch.npu_giou(box1, box2, trans=True, is_cross=False, mode=0)
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

> npu_silu(self) -> Tensor

Computes the for the Swish of "x" .

- Parameters：

  - **self** (Tensor) - A Tensor. Must be one of the following types: float16, float32 

- constraints：

  None

- Examples：
```python
>>> a=torch.rand(2,8).npu()
>>> output = torch.npu_silu(a)
>>> output
tensor([[0.4397, 0.7178, 0.5190, 0.2654, 0.2230, 0.2674, 0.6051, 0.3522],
        [0.4679, 0.1764, 0.6650, 0.3175, 0.0530, 0.4787, 0.5621, 0.4026]],
       device='npu:0')
```

> npu_reshape(self, shape, bool can_refresh=False) -> Tensor

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
  >>> out=torch.npu_reshape(a,(4,4))
  >>> out
  tensor([[0.6657, 0.9857, 0.7614, 0.4368],
          [0.3761, 0.4397, 0.8609, 0.5544],
          [0.7002, 0.3063, 0.9279, 0.5085],
          [0.1009, 0.7133, 0.8118, 0.6193]], device='npu:0')
  ```

> npu_rotated_overlaps(self, query_boxes, trans=False) -> Tensor

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
  >>> output = torch.npu_rotated_overlaps(box1, box2, trans=False)
  >>> output
  tensor([[[0.0000, 0.1562, 0.0000],
          [0.1562, 0.3713, 0.0611],
          [0.0000, 0.0611, 0.0000]]], device='npu:0', dtype=torch.float16)
  ```

> npu_rotated_iou(self, query_boxes, trans=False, mode=0, is_cross=True) -> Tensor

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
  >>> output = torch.npu_rotated_iou(box1, box2, trans=False, mode=0, is_cross=True)
  >>> output
  tensor([[[3.3325e-01, 1.0162e-01],
          [1.0162e-01, 1.0000e+00]],
  
          [[0.0000e+00, 0.0000e+00],
          [0.0000e+00, 5.9605e-08]]], device='npu:0', dtype=torch.float16)
  ```

> npu_rotated_box_encode(anchor_box, gt_bboxes, weight) -> Tensor

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
      >>> out = torch.npu_rotated_box_encode(anchor_boxes, gt_bboxes, weight)
      >>> out
      tensor([[[-0.4253],
              [-0.5166],
              [-1.7021],
              [-0.0162],
              [ 1.1328]]], device='npu:0', dtype=torch.float16)
  ```

  >   npu_rotated_box_decode(anchor_boxes, deltas, weight) -> Tensor

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
        >>> out = torch.npu_rotated_box_decode(anchor_boxes, deltas, weight)
        >>> out
        tensor([[[  1.7861],
                [-10.5781],
                [ 33.0000],
                [ 17.2969],
                [-88.4375]]], device='npu:0', dtype=torch.float16)
    ```

    
