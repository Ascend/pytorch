# torch.nn.functional

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:18:57.736Z pushedAt=2026-06-15T03:25:49.187Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.nn.functional.conv1d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.conv1d.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.conv2d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.conv2d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.conv3d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.conv3d.html)|Yes|Supports bf16, fp16, fp32, complex64|
|[torch.nn.functional.conv_transpose1d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.conv_transpose1d.html)|Yes|Supports fp32|
|[torch.nn.functional.conv_transpose2d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.conv_transpose2d.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.conv_transpose3d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.conv_transpose3d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.unfold](https://pytorch.org/docs/2.9/generated/torch.nn.functional.unfold.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.fold](https://pytorch.org/docs/2.9/generated/torch.nn.functional.fold.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.avg_pool1d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.avg_pool1d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.avg_pool2d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.avg_pool2d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.avg_pool3d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.avg_pool3d.html)|No|-|
|[torch.nn.functional.max_pool1d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.max_pool1d.html)|No|-|
|[torch.nn.functional.max_pool2d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.max_pool2d.html)|Yes|Supports bf16, fp16, fp32<br>By setting torch_npu.npu.use_compatible_impl(True), ensures memory consistency alignment with the community interface of the same name|
|[torch.nn.functional.max_pool3d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.max_pool3d.html)|Yes|Supports bf16, fp16, fp32<br>The dilation value currently only supports being set to 1 or (1,1,1) on NPU<br>When return_indices is True, the data type of the returned argmax is int32|
|[torch.nn.functional.max_unpool1d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.max_unpool1d.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int32, int64|
|[torch.nn.functional.max_unpool2d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.max_unpool2d.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int32, int64<br>When jit_compile=False (binary mode), the product of output_size must be greater than or equal to the product of H and W of the input|
|[torch.nn.functional.max_unpool3d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.max_unpool3d.html)|No|-|
|[torch.nn.functional.lp_pool1d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.lp_pool1d.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.lp_pool2d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.lp_pool2d.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.adaptive_max_pool1d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.adaptive_max_pool1d.html)|No|-|
|[torch.nn.functional.adaptive_max_pool2d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.adaptive_max_pool2d.html)|No|-|
|[torch.nn.functional.adaptive_max_pool3d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.adaptive_max_pool3d.html)|Yes|Supports fp32, fp64|
|[torch.nn.functional.adaptive_avg_pool1d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.adaptive_avg_pool1d.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.adaptive_avg_pool2d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.adaptive_avg_pool2d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.adaptive_avg_pool3d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.adaptive_avg_pool3d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.fractional_max_pool2d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.fractional_max_pool2d.html)|Yes|Potential Fallback to CPU execution|
|[torch.nn.functional.fractional_max_pool3d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.fractional_max_pool3d.html)|Yes|-|
|[torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention)<br>In the current version, the sdpa (scaled_dot_product_attention) interface is only available as a trial feature. This functionality may be adjusted or improved in subsequent versions. Users should pay attention to subsequent version iterations during use.|Atlas A2 Training Series/Atlas A3 Training Series: Yes|Supports bf16, fp16, and fp32. All parameter inputs comply with the following constraints<br>**Constraints**:<br>All parameter inputs comply with specifications:<br>&#8226; N of input query, key, value: batch size, currently only supports the layout of [N, head_num, S(L), E(Ev)], value range 1-2K<br>&#8226; The head num of input query and the head num of key/value must be in a proportional relationship, i.e., Nq/Nkv must be a non-zero integer, value range 1-256<br>&#8226; L of input query: Target sequence length, value range 1-512K<br>&#8226; S of input key, value: Source sequence length, value range 1-512K<br>E of input query, key, value: Embedding dimension of the query and key, value range 1-512<br>&#8226; Ev of input value: Embedding dimension of the value, must be equal to E<br>&#8226; Input attn_mask: currently supports [N, 1, L, S], [N, head_num, L, S], [1, 1, L, S], [L, S], and bool type masks that can be broadcast to [N, head_num, L, S], such as [L, 1], [1, S], [1, 1] layouts<br>&#8226; When is_causal computation is enabled, attn_mask must be None; when is_causal is not enabled, if attn_mask inputs valid data, the input data type must be bool<br>&#8226; The data types of input query, key, value are bf16, fp16, fp32<br>&#8226; By setting torch_npu.npu.use_compatible_impl(True), supports specifying the MATH backend according to the SDPA backend selection context<br>Differences from the original interface besides specification restrictions:<br>&#8226; The random algorithm part of NPU is implemented using DSA hardware. The algorithm is fixed in the DSA engine and differs from the GPU algorithm implementation, resulting in inconsistency between the dropout functionality and GPU results<br>&#8226; The current interface supports unequal lengths of head num for input query and key/value, while the native PyTorch interface does not support this|
|[torch.nn.functional.threshold](https://pytorch.org/docs/2.9/generated/torch.nn.functional.threshold.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64<br>When the input is of int32 type exceeding 16,777,216 (i.e., 2<sup>24</sup>), precision loss may occur|
|[torch.nn.functional.threshold_](https://pytorch.org/docs/2.9/generated/torch.nn.functional.threshold_.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64<br>When the input is of int32 type exceeding 16,777,216 (i.e., 2<sup>24</sup>), precision loss may occur|
|[torch.nn.functional.relu](https://pytorch.org/docs/2.9/generated/torch.nn.functional.relu.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64|
|[torch.nn.functional.relu_](https://pytorch.org/docs/2.9/generated/torch.nn.functional.relu_.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64|
|[torch.nn.functional.hardtanh](https://pytorch.org/docs/2.9/generated/torch.nn.functional.hardtanh.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.nn.functional.hardtanh_](https://pytorch.org/docs/2.9/generated/torch.nn.functional.hardtanh_.html)|Yes|Supports fp16, fp32, int8, int16, int32, int64|
|[torch.nn.functional.hardswish](https://pytorch.org/docs/2.9/generated/torch.nn.functional.hardswish.html)|Yes|Supports fp16, fp32<br>Potential Fallback to CPU execution|
|[torch.nn.functional.relu6](https://pytorch.org/docs/2.9/generated/torch.nn.functional.relu6.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.nn.functional.elu](https://pytorch.org/docs/2.9/generated/torch.nn.functional.elu.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[torch.nn.functional.elu_](https://pytorch.org/docs/2.9/generated/torch.nn.functional.elu_.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.selu](https://pytorch.org/docs/2.9/generated/torch.nn.functional.selu.html)|Yes|Supports fp16, fp32<br>In the backward scenario of fp16, there is cumulative precision error compared to GPU, which can be avoided by the following method:<br>Replace the forward call of torch.nn.functional.selu with torch.ops.aten.elu, for example: replace torch.nn.functional.selu(input_x) with torch.ops.aten.elu(input_x, 1.6732632423543772848170429916717, 1.0507009873554804934193349852946)|
|[torch.nn.functional.celu](https://pytorch.org/docs/2.9/generated/torch.nn.functional.celu.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.leaky_relu](https://pytorch.org/docs/2.9/generated/torch.nn.functional.leaky_relu.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[torch.nn.functional.leaky_relu_](https://pytorch.org/docs/2.9/generated/torch.nn.functional.leaky_relu_.html)|Yes|Supports fp16, fp32, fp64|
|[torch.nn.functional.prelu](https://pytorch.org/docs/2.9/generated/torch.nn.functional.prelu.html)|Yes|Supports fp16, fp32<br>Input only supports 1-8 dimensions|
|[torch.nn.functional.rrelu](https://pytorch.org/docs/2.9/generated/torch.nn.functional.rrelu.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.glu](https://pytorch.org/docs/2.9/generated/torch.nn.functional.glu.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[torch.nn.functional.gelu](https://pytorch.org/docs/2.9/generated/torch.nn.functional.gelu.html)|Yes|Supports bf16, fp16, fp32<br>The approximate parameter only supports being set to tanh|
|[torch.nn.functional.logsigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.functional.logsigmoid.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.hardshrink](https://pytorch.org/docs/2.9/generated/torch.nn.functional.hardshrink.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.softsign](https://pytorch.org/docs/2.9/generated/torch.nn.functional.softsign.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.nn.functional.softplus](https://pytorch.org/docs/2.9/generated/torch.nn.functional.softplus.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.softmax](https://pytorch.org/docs/2.9/generated/torch.nn.functional.softmax.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[torch.nn.functional.softshrink](https://pytorch.org/docs/2.9/generated/torch.nn.functional.softshrink.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.gumbel_softmax](https://pytorch.org/docs/2.9/generated/torch.nn.functional.gumbel_softmax.html)|No|-|
|[torch.nn.functional.log_softmax](https://pytorch.org/docs/2.9/generated/torch.nn.functional.log_softmax.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.tanh](https://pytorch.org/docs/2.9/generated/torch.nn.functional.tanh.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.nn.functional.sigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.functional.sigmoid.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nn.functional.hardsigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.functional.hardsigmoid.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.silu](https://pytorch.org/docs/2.9/generated/torch.nn.functional.silu.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.mish](https://pytorch.org/docs/2.9/generated/torch.nn.functional.mish.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.batch_norm](https://pytorch.org/docs/2.9/generated/torch.nn.functional.batch_norm.html)|Yes|Supports fp16, fp32<br>weight and bias only support 1-dimensional scenarios<br>The shape of bias is 1-dimensional, with length equal to the length of the channel axis in the input parameter|
|[torch.nn.functional.group_norm](https://pytorch.org/docs/2.9/generated/torch.nn.functional.group_norm.html)|Yes|Supports bf16, fp16, fp32<br>This API only supports input with 2 or more dimensions<br>The eps parameter must be greater than 0|
|[torch.nn.functional.layer_norm](https://pytorch.org/docs/2.9/generated/torch.nn.functional.layer_norm.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.normalize](https://pytorch.org/docs/2.9/generated/torch.nn.functional.normalize.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[torch.nn.functional.linear](https://pytorch.org/docs/2.9/generated/torch.nn.functional.linear.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.bilinear](https://pytorch.org/docs/2.9/generated/torch.nn.functional.bilinear.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.dropout](https://pytorch.org/docs/2.9/generated/torch.nn.functional.dropout.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nn.functional.alpha_dropout](https://pytorch.org/docs/2.9/generated/torch.nn.functional.alpha_dropout.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nn.functional.feature_alpha_dropout](https://pytorch.org/docs/2.9/generated/torch.nn.functional.feature_alpha_dropout.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nn.functional.dropout2d](https://pytorch.org/docs/2.9/generated/torch.nn.functional.dropout2d.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nn.functional.embedding](https://pytorch.org/docs/2.9/generated/torch.nn.functional.embedding.html)|Yes|Supports bf16, fp16, fp32, int32, int64<br>The attribute max_norm only supports non-negative values|
|[torch.nn.functional.embedding_bag](https://pytorch.org/docs/2.9/generated/torch.nn.functional.embedding_bag.html)|No|-|
|[torch.nn.functional.one_hot](https://pytorch.org/docs/2.9/generated/torch.nn.functional.one_hot.html)|Yes|Supports int32, int64|
|[torch.nn.functional.cosine_similarity](https://pytorch.org/docs/2.9/generated/torch.nn.functional.cosine_similarity.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.pdist](https://pytorch.org/docs/2.9/generated/torch.nn.functional.pdist.html)|No|-|
|[torch.nn.functional.binary_cross_entropy](https://pytorch.org/docs/2.9/generated/torch.nn.functional.binary_cross_entropy.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.binary_cross_entropy_with_logits](https://pytorch.org/docs/2.9/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.poisson_nll_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.poisson_nll_loss.html)|Yes|Supports bf16, fp16, fp32, int64<br>Potential Fallback to CPU execution|
|[torch.nn.functional.cross_entropy](https://pytorch.org/docs/2.9/generated/torch.nn.functional.cross_entropy.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.ctc_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.ctc_loss.html)|Yes|Supports fp32, fp64<br>The length of the target sequence does not support 0, i.e., the value of the attribute target_lengths cannot contain 0|
|[torch.nn.functional.gaussian_nll_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.gaussian_nll_loss.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.nn.functional.kl_div](https://pytorch.org/docs/2.9/generated/torch.nn.functional.kl_div.html)|Yes|Supports bf16, fp16, fp32<br>Currently the log_target parameter only supports False<br>Currently target does not support differentiation|
|[torch.nn.functional.l1_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.l1_loss.html)|Yes|Supports bf16, fp16, fp32, int64|
|[torch.nn.functional.mse_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.mse_loss.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.margin_ranking_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.margin_ranking_loss.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.multilabel_margin_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.multilabel_margin_loss.html)|Yes|Supports fp16, fp32<br>The number of elements in the input tensor cannot exceed 100,000|
|[torch.nn.functional.multilabel_soft_margin_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.multilabel_soft_margin_loss.html)|No|-|
|[torch.nn.functional.nll_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.nll_loss.html)|Yes|Supports fp32<br>Each element value in target should be greater than or equal to 0 and less than the number of classes of input|
|[torch.nn.functional.smooth_l1_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.smooth_l1_loss.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.functional.soft_margin_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.soft_margin_loss.html)|Yes|Supports bf16, fp16, fp32, does not support double, complex64, complex128 data types|
|[torch.nn.functional.triplet_margin_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.triplet_margin_loss.html)|No|-|
|[torch.nn.functional.triplet_margin_with_distance_loss](https://pytorch.org/docs/2.9/generated/torch.nn.functional.triplet_margin_with_distance_loss.html)|No|-|
|[torch.nn.functional.pixel_shuffle](https://pytorch.org/docs/2.9/generated/torch.nn.functional.pixel_shuffle.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nn.functional.pixel_unshuffle](https://pytorch.org/docs/2.9/generated/torch.nn.functional.pixel_unshuffle.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.nn.functional.pad](https://pytorch.org/docs/2.9/generated/torch.nn.functional.pad.html)|Yes|When the attribute mode is constant, supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>When the attribute mode is not constant, supports fp16, fp32, fp64<br>Performance degradation may occur when the input x has six or more dimensions|
|[torch.nn.functional.interpolate](https://pytorch.org/docs/2.9/generated/torch.nn.functional.interpolate.html)|Yes|Supports bf16, fp16, fp32, fp64<br>Supports nearest, linear, bilinear, bicubic, trilinear, area<br>Does not support scale_factors|
|[torch.nn.functional.upsample](https://pytorch.org/docs/2.9/generated/torch.nn.functional.upsample.html)|Yes|Supports fp16, fp32, fp64<br>Only supports mode = nearest|
|[torch.nn.functional.upsample_nearest](https://pytorch.org/docs/2.9/generated/torch.nn.functional.upsample_nearest.html)|Yes|Supports fp16, fp32, fp64<br>Only supports 3-5 dimensions|
|[torch.nn.functional.upsample_bilinear](https://pytorch.org/docs/2.9/generated/torch.nn.functional.upsample_bilinear.html)|Yes|Supports fp16, fp32|
|[torch.nn.functional.grid_sample](https://pytorch.org/docs/2.9/generated/torch.nn.functional.grid_sample.html)|Yes|Supports fp16, fp32, fp64|
|[torch.nn.functional.affine_grid](https://pytorch.org/docs/2.9/generated/torch.nn.functional.affine_grid.html)|Yes|Supports fp16, fp32|
|[torch.nn.parallel.data_parallel](https://pytorch.org/docs/2.9/nn.html#torch.nn.parallel.data_parallel)|No|-|
