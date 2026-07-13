# torch.nn.functional

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:18:57.736Z pushedAt=2026-06-15T03:25:49.187Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.nn.functional.conv1d|Yes|Supports fp16, fp32|
|torch.nn.functional.conv2d|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.conv3d|Yes|Supports bf16, fp16, fp32, complex64|
|torch.nn.functional.conv_transpose1d|Yes|Supports fp32|
|torch.nn.functional.conv_transpose2d|Yes|Supports fp16, fp32|
|torch.nn.functional.conv_transpose3d|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.unfold|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.fold|Yes|Supports fp16, fp32|
|torch.nn.functional.avg_pool1d|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.avg_pool2d|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.avg_pool3d|No|-|
|torch.nn.functional.max_pool1d|No|-|
|torch.nn.functional.max_pool2d|Yes|Supports bf16, fp16, fp32<br>By setting torch_npu.npu.use_compatible_impl(True), ensures memory consistency alignment with the community interface of the same name|
|torch.nn.functional.max_pool3d|Yes|Supports bf16, fp16, fp32<br>The dilation value currently only supports being set to 1 or (1,1,1) on NPU<br>When return_indices is True, the data type of the returned argmax is int32|
|torch.nn.functional.max_unpool1d|Yes|Supports fp16, fp32, fp64, uint8, int8, int32, int64|
|torch.nn.functional.max_unpool2d|Yes|Supports fp16, fp32, fp64, uint8, int8, int32, int64<br>When jit_compile=False (binary mode), the product of output_size must be greater than or equal to the product of H and W of the input|
|torch.nn.functional.max_unpool3d|No|-|
|torch.nn.functional.lp_pool1d|Yes|Supports fp16, fp32|
|torch.nn.functional.lp_pool2d|Yes|Supports fp16, fp32|
|torch.nn.functional.adaptive_max_pool1d|No|-|
|torch.nn.functional.adaptive_max_pool2d|No|-|
|torch.nn.functional.adaptive_max_pool3d|Yes|Supports fp32, fp64|
|torch.nn.functional.adaptive_avg_pool1d|Yes|Supports fp16, fp32|
|torch.nn.functional.adaptive_avg_pool2d|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.adaptive_avg_pool3d|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.fractional_max_pool2d|Yes|Potential Fallback to CPU execution|
|torch.nn.functional.fractional_max_pool3d|Yes|-|
|torch.nn.functional.scaled_dot_product_attention<br>In the current version, the sdpa (scaled_dot_product_attention) interface is only available as a trial feature. This functionality may be adjusted or improved in subsequent versions. Users should pay attention to subsequent version iterations during use.|Atlas A2 Training Series/Atlas A3 Training Series: Yes|Supports bf16, fp16, and fp32. All parameter inputs comply with the following constraints<br>**Constraints**:<br>All parameter inputs comply with specifications:<br>&#8226; N of input query, key, value: batch size, currently only supports the layout of [N, head_num, S(L), E(Ev)], value range 1-2K<br>&#8226; The head num of input query and the head num of key/value must be in a proportional relationship, i.e., Nq/Nkv must be a non-zero integer, value range 1-256<br>&#8226; L of input query: Target sequence length, value range 1-512K<br>&#8226; S of input key, value: Source sequence length, value range 1-512K<br>E of input query, key, value: Embedding dimension of the query and key, value range 1-512<br>&#8226; Ev of input value: Embedding dimension of the value, must be equal to E<br>&#8226; Input attn_mask: currently supports [N, 1, L, S], [N, head_num, L, S], [1, 1, L, S], [L, S], and bool type masks that can be broadcast to [N, head_num, L, S], such as [L, 1], [1, S], [1, 1] layouts<br>&#8226; When is_causal computation is enabled, attn_mask must be None; when is_causal is not enabled, if attn_mask inputs valid data, the input data type must be bool<br>&#8226; The data types of input query, key, value are bf16, fp16, fp32<br>&#8226; By setting torch_npu.npu.use_compatible_impl(True), supports specifying the MATH backend according to the SDPA backend selection context<br>Differences from the original interface besides specification restrictions:<br>&#8226; The random algorithm part of NPU is implemented using DSA hardware. The algorithm is fixed in the DSA engine and differs from the GPU algorithm implementation, resulting in inconsistency between the dropout functionality and GPU results<br>&#8226; The current interface supports unequal lengths of head num for input query and key/value, while the native PyTorch interface does not support this|
|torch.nn.functional.threshold|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64<br>When the input is of int32 type exceeding 16,777,216 (i.e., 2<sup>24</sup>), precision loss may occur|
|torch.nn.functional.threshold_|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64<br>When the input is of int32 type exceeding 16,777,216 (i.e., 2<sup>24</sup>), precision loss may occur|
|torch.nn.functional.relu|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64|
|torch.nn.functional.relu_|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64|
|torch.nn.functional.hardtanh|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.nn.functional.hardtanh_|Yes|Supports fp16, fp32, int8, int16, int32, int64|
|torch.nn.functional.hardswish|Yes|Supports fp16, fp32<br>Potential Fallback to CPU execution|
|torch.nn.functional.relu6|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.nn.functional.elu|Yes|Supports bf16, fp16, fp32, fp64|
|torch.nn.functional.elu_|Yes|Supports fp16, fp32|
|torch.nn.functional.selu|Yes|Supports fp16, fp32<br>In the backward scenario of fp16, there is cumulative precision error compared to GPU, which can be avoided by the following method:<br>Replace the forward call of torch.nn.functional.selu with torch.ops.aten.elu, for example: replace torch.nn.functional.selu(input_x) with torch.ops.aten.elu(input_x, 1.6732632423543772848170429916717, 1.0507009873554804934193349852946)|
|torch.nn.functional.celu|Yes|Supports fp16, fp32|
|torch.nn.functional.leaky_relu|Yes|Supports bf16, fp16, fp32, fp64|
|torch.nn.functional.leaky_relu_|Yes|Supports fp16, fp32, fp64|
|torch.nn.functional.prelu|Yes|Supports fp16, fp32<br>Input only supports 1-8 dimensions|
|torch.nn.functional.rrelu|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.glu|Yes|Supports bf16, fp16, fp32, fp64|
|torch.nn.functional.gelu|Yes|Supports bf16, fp16, fp32<br>The approximate parameter only supports being set to tanh|
|torch.nn.functional.logsigmoid|Yes|Supports fp16, fp32|
|torch.nn.functional.hardshrink|Yes|Supports fp16, fp32|
|torch.nn.functional.softsign|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|torch.nn.functional.softplus|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.softmax|Yes|Supports bf16, fp16, fp32, fp64|
|torch.nn.functional.softshrink|Yes|Supports fp16, fp32|
|torch.nn.functional.gumbel_softmax|No|-|
|torch.nn.functional.log_softmax|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.tanh|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.nn.functional.sigmoid|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nn.functional.hardsigmoid|Yes|Supports fp16, fp32|
|torch.nn.functional.silu|Yes|Supports fp16, fp32|
|torch.nn.functional.mish|Yes|Supports fp16, fp32|
|torch.nn.functional.batch_norm|Yes|Supports fp16, fp32<br>weight and bias only support 1-dimensional scenarios<br>The shape of bias is 1-dimensional, with length equal to the length of the channel axis in the input parameter|
|torch.nn.functional.group_norm|Yes|Supports bf16, fp16, fp32<br>This API only supports input with 2 or more dimensions<br>The eps parameter must be greater than 0|
|torch.nn.functional.layer_norm|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.normalize|Yes|Supports bf16, fp16, fp32, fp64|
|torch.nn.functional.linear|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.bilinear|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.dropout|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nn.functional.alpha_dropout|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nn.functional.feature_alpha_dropout|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nn.functional.dropout2d|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nn.functional.embedding|Yes|Supports bf16, fp16, fp32, int32, int64<br>The attribute max_norm only supports non-negative values|
|torch.nn.functional.embedding_bag|No|-|
|torch.nn.functional.one_hot|Yes|Supports int32, int64|
|torch.nn.functional.cosine_similarity|Yes|Supports fp16, fp32|
|torch.nn.functional.pdist|No|-|
|torch.nn.functional.binary_cross_entropy|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.binary_cross_entropy_with_logits|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.poisson_nll_loss|Yes|Supports bf16, fp16, fp32, int64<br>Potential Fallback to CPU execution|
|torch.nn.functional.cross_entropy|Yes|Supports fp16, fp32|
|torch.nn.functional.ctc_loss|Yes|Supports fp32, fp64<br>The length of the target sequence does not support 0, i.e., the value of the attribute target_lengths cannot contain 0|
|torch.nn.functional.gaussian_nll_loss|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|torch.nn.functional.kl_div|Yes|Supports bf16, fp16, fp32<br>Currently the log_target parameter only supports False<br>Currently target does not support differentiation|
|torch.nn.functional.l1_loss|Yes|Supports bf16, fp16, fp32, int64|
|torch.nn.functional.mse_loss|Yes|Supports fp16, fp32|
|torch.nn.functional.margin_ranking_loss|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.multilabel_margin_loss|Yes|Supports fp16, fp32<br>The number of elements in the input tensor cannot exceed 100,000|
|torch.nn.functional.multilabel_soft_margin_loss|No|-|
|torch.nn.functional.nll_loss|Yes|Supports fp32<br>Each element value in target should be greater than or equal to 0 and less than the number of classes of input|
|torch.nn.functional.smooth_l1_loss|Yes|Supports bf16, fp16, fp32|
|torch.nn.functional.soft_margin_loss|Yes|Supports bf16, fp16, fp32, does not support double, complex64, complex128 data types|
|torch.nn.functional.triplet_margin_loss|No|-|
|torch.nn.functional.triplet_margin_with_distance_loss|No|-|
|torch.nn.functional.pixel_shuffle|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nn.functional.pixel_unshuffle|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.nn.functional.pad|Yes|When the attribute mode is constant, supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>When the attribute mode is not constant, supports fp16, fp32, fp64<br>Performance degradation may occur when the input x has six or more dimensions|
|torch.nn.functional.interpolate|Yes|Supports bf16, fp16, fp32, fp64<br>Supports nearest, linear, bilinear, bicubic, trilinear, area<br>Does not support scale_factors|
|torch.nn.functional.upsample|Yes|Supports fp16, fp32, fp64<br>Only supports mode = nearest|
|torch.nn.functional.upsample_nearest|Yes|Supports fp16, fp32, fp64<br>Only supports 3-5 dimensions|
|torch.nn.functional.upsample_bilinear|Yes|Supports fp16, fp32|
|torch.nn.functional.grid_sample|Yes|Supports fp16, fp32, fp64|
|torch.nn.functional.affine_grid|Yes|Supports fp16, fp32|
|torch.nn.parallel.data_parallel|No|-|
