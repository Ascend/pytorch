# torch.nn

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:34:03.234Z pushedAt=2026-07-09T08:44:08.318Z -->

> [!NOTE]
> If an API's "Supported" column shows "Yes" and the "Restrictions and Notes" column shows "-", it means the API support is consistent with the native API.

| API Name | Supported | Restrictions and Notes |
|--|--|--|
| torch.nn.parameter.Parameter | Yes | Supports fp32 |
| torch.nn.parameter.UninitializedParameter | Yes | - |
| torch.nn.parameter.UninitializedParameter.cls_to_become | Yes | - |
| torch.nn.parameter.UninitializedBuffer | Yes | - |
| torch.nn.Module | Yes | Supports fp32 |
| torch.nn.Module.add_module | Yes | Supports fp32 |
| torch.nn.Module.apply | Yes | Supports fp32 |
| torch.nn.Module.bfloat16 | Yes | - |
| torch.nn.Module.buffers | Yes | - |
| torch.nn.Module.children | Yes | Supports fp32 |
| torch.nn.Module.compile | Yes | - |
| torch.nn.Module.cpu | Yes | Supports fp32 |
| torch.nn.Module.cuda | Yes | Supports fp32 |
| torch.nn.Module.double | Yes | - |
| torch.nn.Module.eval | Yes | Supports fp32, int64 |
| torch.nn.Module.extra_repr | Yes | Supports fp32 |
| torch.nn.Module.float | Yes | Supports fp16, fp32 |
| torch.nn.Module.forward | Yes | Supports fp32 |
| torch.nn.Module.get_buffer | Yes | - |
| torch.nn.Module.get_extra_state | Yes | - |
| torch.nn.Module.get_parameter | Yes | Supports fp32 |
| torch.nn.Module.get_submodule | Yes | Supports fp32 |
| torch.nn.Module.half | Yes | Supports fp16, fp32 |
| torch.nn.Module.ipu | No | - |
| torch.nn.Module.load_state_dict | Yes | Supports fp32 |
| torch.nn.Module.modules | Yes | Supports fp32 |
| torch.nn.Module.named_buffers | Yes | - |
| torch.nn.Module.named_children | Yes | Supports fp32 |
| torch.nn.Module.named_modules | Yes | Supports fp32 |
| torch.nn.Module.named_parameters | Yes | - |
| torch.nn.Module.parameters | Yes | - |
| torch.nn.Module.register_backward_hook | Yes | Supports fp32 |
| torch.nn.Module.register_buffer | Yes | Supports fp32 |
| torch.nn.Module.register_forward_hook | Yes | Supports fp32 |
| torch.nn.Module.register_forward_pre_hook | Yes | Supports fp32 |
| torch.nn.Module.register_full_backward_hook | Yes | Supports fp32 |
| torch.nn.Module.register_full_backward_pre_hook | Yes | Supports fp32 |
| torch.nn.Module.register_load_state_dict_post_hook | Yes | Supports fp32 |
| torch.nn.Module.register_module | Yes | Supports fp32 |
| torch.nn.Module.register_parameter | Yes | - |
| torch.nn.Module.register_state_dict_pre_hook | Yes | - |
| torch.nn.Module.requires_grad_ | Yes | - |
| torch.nn.Module.set_extra_state | Yes | - |
| torch.nn.Module.share_memory | No | - |
| torch.nn.Module.state_dict | Yes | Supports fp32 |
| torch.nn.Module.to | Yes | Supports fp32 |
| torch.nn.Module.to_empty | Yes | Supports fp32 |
| torch.nn.Module.train | Yes | Supports fp32 |
| torch.nn.Module.type | Yes | Supports fp16, fp32, int64 |
| torch.nn.Module.xpu | No | - |
| torch.nn.Module.zero_grad | Yes | Supports fp32 |
| torch.nn.Sequential | Yes | Supports fp32 |
| torch.nn.Sequential.append | Yes | Supports fp32 |
| torch.nn.ModuleList | Yes | Supports fp32 |
| torch.nn.ModuleList.append | Yes | Supports fp32 |
| torch.nn.ModuleList.extend | Yes | Supports fp32 |
| torch.nn.ModuleList.insert | Yes | Supports fp32 |
| torch.nn.ModuleDict | Yes | Supports fp32 |
| torch.nn.ModuleDict.clear | Yes | Supports fp32 |
| torch.nn.ModuleDict.items | Yes | Supports fp32 |
| torch.nn.ModuleDict.keys | Yes | Supports fp32 |
| torch.nn.ModuleDict.pop | Yes | Supports fp32 |
| torch.nn.ModuleDict.update | Yes | Supports fp32 |
| torch.nn.ModuleDict.values | Yes | Supports fp32 |
| torch.nn.ParameterList | Yes | Supports fp32 |
| torch.nn.ParameterList.append | Yes | Supports fp32 |
| torch.nn.ParameterList.extend | Yes | Supports fp32 |
| torch.nn.ParameterDict | Yes | Supports fp32 |
| torch.nn.ParameterDict.clear | Yes | Supports fp32 |
| torch.nn.ParameterDict.copy | Yes | Supports fp32 |
| torch.nn.ParameterDict.fromkeys | Yes | Supports fp32 |
| torch.nn.ParameterDict.get | Yes | Supports fp32 |
| torch.nn.ParameterDict.items | Yes | Supports fp32 |
| torch.nn.ParameterDict.keys | Yes | Supports fp32 |
| torch.nn.ParameterDict.pop | Yes | Supports fp32 |
| torch.nn.ParameterDict.popitem | Yes | Supports fp32 |
| torch.nn.ParameterDict.setdefault | Yes | Supports fp32 |
| torch.nn.ParameterDict.update | Yes | Supports fp32 |
| torch.nn.ParameterDict.values | Yes | Supports fp32 |
| torch.nn.modules.module.register_module_forward_pre_hook | Yes | Supports fp32 |
| torch.nn.modules.module.register_module_forward_hook | Yes | Supports fp32 |
| torch.nn.modules.module.register_module_backward_hook | Yes | Supports fp32 |
| torch.nn.modules.module.register_module_full_backward_pre_hook | No | - |
| torch.nn.modules.module.register_module_full_backward_hook | Yes | Supports fp32 |
| torch.nn.modules.module.register_module_buffer_registration_hook | No | - |
| torch.nn.modules.module.register_module_module_registration_hook | No | - |
| torch.nn.modules.module.register_module_parameter_registration_hook | No | - |
| torch.nn.Conv1d | Yes | Supports fp16, fp32 |
| torch.nn.Conv2d | Yes | Supports bf16, fp16, fp32<br>Atlas A2 Training Series, in default scenarios, if compilation is triggered frequently, it is recommended to manually set torch.npu.config.allow_internal_format to False to disable internal format for input parameters and avoid online compilation |
| torch.nn.Conv3d | Yes | Supports bf16, fp16, fp32 |
| torch.nn.ConvTranspose1d | Yes | Supports fp32 |
| torch.nn.ConvTranspose2d | Yes | Supports fp16, fp32<br>Atlas Training Series/Atlas A2 Training Series, torch.npu.config.allow_internal_format must be manually set to False to support 3-dimensional input |
| torch.nn.ConvTranspose3d | Yes | Supports bf16, fp16, fp32 |
| torch.nn.LazyConv1d | Yes | Supports fp16, fp32 |
| torch.nn.LazyConv1d.cls_to_become | Yes | - |
| torch.nn.LazyConv2d | Yes | Supports fp16, fp32 |
| torch.nn.LazyConv2d.cls_to_become | Yes | - |
| torch.nn.LazyConv3d.cls_to_become | Yes | - |
| torch.nn.LazyConvTranspose1d | Yes | Supports fp16 |
| torch.nn.LazyConvTranspose1d.cls_to_become | Yes | - |
|torch.nn.LazyConvTranspose2d|Yes|Supports fp16, fp32|
|torch.nn.LazyConvTranspose2d.cls_to_become|Yes|-|
|torch.nn.LazyConvTranspose3d.cls_to_become|Yes|-|
|torch.nn.Unfold|Yes|Supports bf16, fp16, fp32|
|torch.nn.Fold|Yes|Supports fp16|
|torch.nn.MaxPool1d|No|-|
|torch.nn.MaxPool2d|Yes|Supports bf16, fp16, fp32<br>By setting torch_npu.npu.use_compatible_impl(True), ensures memory consistency alignment with the community counterpart interface|
|torch.nn.MaxPool3d|No|-|
|torch.nn.MaxUnpool1d|Yes|Supports fp16, fp32|
|torch.nn.MaxUnpool2d|Yes|Supports fp16, fp32|
|torch.nn.MaxUnpool3d|No|-|
|torch.nn.AvgPool1d|Yes|Supports bf16, fp16, fp32|
|torch.nn.AvgPool2d|Yes|Supports bf16, fp16, fp32|
|torch.nn.AvgPool3d|No|-|
|torch.nn.LPPool1d|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.nn.LPPool2d|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|torch.nn.AdaptiveMaxPool1d|No|-|
|torch.nn.AdaptiveMaxPool2d|No|-|
|torch.nn.AdaptiveMaxPool3d|Yes|Supports fp32, fp64|
|torch.nn.AdaptiveAvgPool1d|Yes|Supports fp16, fp32|
|torch.nn.AdaptiveAvgPool2d|Yes|Supports fp16, fp32|
|torch.nn.AdaptiveAvgPool3d|Yes|Supports bf16, fp16, fp32|
|torch.nn.ReflectionPad1d|Yes|Supports fp16, fp32|
|torch.nn.ReflectionPad2d|Yes|Supports fp16, fp32|
|torch.nn.ReflectionPad3d|No|-|
|torch.nn.ReplicationPad1d|Yes|Supports fp16, fp32, complex64, complex128|
|torch.nn.ReplicationPad2d|Yes|Supports fp16, fp32, complex64, complex128|
|torch.nn.ReplicationPad3d|No|-|
|torch.nn.ZeroPad1d|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128<br>Supports 2-3 dimensions|
|torch.nn.ZeroPad2d|Yes|May fall back to CPU execution|
|torch.nn.ZeroPad3d|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128<br>Supports 5-6 dimensions|
|torch.nn.ConstantPad1d|Yes|Supports int8, bool<br>Performance degradation may occur when input x has six or more dimensions|
|torch.nn.ConstantPad2d|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Performance degradation may occur when input x has six or more dimensions|
|torch.nn.ConstantPad3d|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Performance degradation may occur when input x has six or more dimensions|
|torch.nn.ELU|Yes|Supports bf16, fp16, fp32, fp64|
|torch.nn.Hardshrink|Yes|Supports fp16, fp32<br>May fall back to CPU execution|
|torch.nn.Hardsigmoid|Yes|Supports fp16, fp32, int32<br>May fall back to CPU execution|
|torch.nn.Hardtanh|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.nn.Hardswish|Yes|Supports fp16, fp32|
|torch.nn.LeakyReLU|Yes|Supports bf16, fp16, fp32, fp64|
|torch.nn.LogSigmoid|Yes|Supports fp16, fp32|
|torch.nn.MultiheadAttention|Yes|Supports bf16, fp16, fp32|
|torch.nn.MultiheadAttention.forward|Yes|Supports bf16, fp16, fp32|
|torch.nn.PReLU|Yes|Supports fp32|
|torch.nn.ReLU|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64|
|torch.nn.ReLU6|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.nn.RReLU|No|-|
|torch.nn.SELU|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.nn.CELU|Yes|Supports fp16, fp32|
|torch.nn.GELU|Yes|Supports bf16, fp16, fp32<br>The approximate parameter only supports being set to tanh|
|torch.nn.Sigmoid|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nn.SiLU|Yes|Supports bf16, fp16, fp32|
|torch.nn.Mish|Yes|Supports fp16, fp32|
|torch.nn.Softplus|Yes|Supports bf16, fp16, fp32|
|torch.nn.Softshrink|Yes|Supports bf16, fp16, fp32|
|torch.nn.Softsign|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|torch.nn.Tanh|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.nn.Tanhshrink|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64<br>May fall back to CPU execution|
|torch.nn.Threshold|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|torch.nn.GLU|Yes|Supports fp16, fp32|
|torch.nn.Softmin|Yes|Supports bf16, fp16, fp32|
|torch.nn.Softmax|Yes|Supports bf16, fp16, fp32, fp64|
|torch.nn.Softmax2d|Yes|Supports bf16, fp16, fp32|
|torch.nn.LogSoftmax|Yes|Supports bf16, fp16, fp32|
|torch.nn.AdaptiveLogSoftmaxWithLoss|No|-|
|torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob|No|-|
|torch.nn.AdaptiveLogSoftmaxWithLoss.predict|No|-|
|torch.nn.BatchNorm1d|Yes|Supports fp16, fp32|
|torch.nn.BatchNorm2d|Yes|Supports fp16, fp32|
|torch.nn.BatchNorm3d|Yes|Supports fp16, fp32|
|torch.nn.LazyBatchNorm1d.cls_to_become|Yes|-|
|torch.nn.LazyBatchNorm2d.cls_to_become|Yes|-|
|torch.nn.LazyBatchNorm3d.cls_to_become|Yes|-|
|torch.nn.GroupNorm|Yes|Supports fp32<br>The eps parameter must be greater than 0<br>Does not support the scenario with jit_compile=True<br>This API only supports input with 2 or more dimensions. During backpropagation, it does not support scenarios where the input dimension is not 4, or where num_groups is not divisible by 32, or where the C-axis dimension is not divisible by (10 * num_groups)|
|torch.nn.SyncBatchNorm|Yes|Supports fp16, fp32|
|torch.nn.SyncBatchNorm.convert_sync_batchnorm|Yes|-|
|torch.nn.LazyInstanceNorm1d.cls_to_become|Yes|-|
|torch.nn.LazyInstanceNorm2d.cls_to_become|Yes|-|
|torch.nn.LazyInstanceNorm3d.cls_to_become|Yes|-|
|torch.nn.LayerNorm|Yes|Supports bf16, fp16, fp32<br>By setting torch_npu.npu.use_compatible_impl(True), switches this interface from the aclnnLayerNorm operator to the aclnnFastLayerNorm operator, ensuring memory consistency alignment with the community counterpart interface.|
|torch.nn.RNNBase|No|-|
|torch.nn.RNNBase.flatten_parameters|No|-|
|torch.nn.RNN|No|-|
|torch.nn.LSTM|Yes|Supports fp32<br>Does not support the proj_size parameter<br>Does not support the dropout parameter<br>The input parameter does not support 2 dimensions|
|torch.nn.GRU|No|-|
|torch.nn.RNNCell|No|-|
|torch.nn.LSTMCell|Yes|The interface does not currently support jit_compile=False. If you need to use it in this mode, please add "DynamicGRUV2" to the "NPU_FUZZY_COMPILE_BLACKLIST" option. For details, see [Example of Adding a Binary Blocklist](../example_of_adding_a_binary_blocklist.md)|
|torch.nn.GRUCell|Yes|Supports fp16, fp32|
|torch.nn.Transformer|No|-|
|torch.nn.Transformer.forward|Yes|Supports fp32|
|torch.nn.TransformerEncoder|No|-|
|torch.nn.TransformerEncoder.forward|Yes|Supports fp32|
|torch.nn.TransformerDecoder|No|-|
|torch.nn.TransformerDecoder.forward|No|-|
|torch.nn.TransformerEncoderLayer.forward|No|-|
|torch.nn.TransformerDecoderLayer.forward|No|-|
|torch.nn.Identity|Yes|Supports fp32|
|torch.nn.Linear|Yes|Supports fp16, fp32|
|torch.nn.Bilinear|Yes|Supports bf16, fp16, fp32|
|torch.nn.LazyLinear|Yes|Supports fp16, fp32|
|torch.nn.LazyLinear.cls_to_become|No|-|
|torch.nn.Dropout|Yes|Supports bf16, fp16, fp32|
|torch.nn.Dropout2d|Yes|Supports fp16, fp32, int64, bool|
|torch.nn.AlphaDropout|Yes|Supports fp16, fp32|
|torch.nn.FeatureAlphaDropout|Yes|Supports fp16, fp32|
|torch.nn.Embedding|Yes|Supports int32, int64<br>The max_norm attribute does not support NaN, only non-negative values|
|torch.nn.Embedding.from_pretrained|Yes|Supports fp64|
|torch.nn.EmbeddingBag|Yes|Supports int32, int64<br>Only supports max_norm greater than or equal to 0|
|torch.nn.EmbeddingBag.forward|Yes|Supports int64|
|torch.nn.EmbeddingBag.from_pretrained|Yes|Supports int64|
|torch.nn.L1Loss|Yes|Supports fp16, fp32, int64|
|torch.nn.MSELoss|Yes|Supports fp16, fp32|
|torch.nn.CrossEntropyLoss|Yes|Supports fp16, fp32|
|torch.nn.CTCLoss|Yes|Supports fp32, fp64<br>2D input for log_probs is not supported|
|torch.nn.NLLLoss|Yes|Supports fp16, fp32<br>Each element in target must be greater than or equal to 0 and less than the number of classes in input|
|torch.nn.PoissonNLLLoss|Yes|Supports bf16, fp16, fp32, int64|
|torch.nn.GaussianNLLLoss|Yes|Supports bf16, fp16, fp32, int16, int32, int64|
|torch.nn.KLDivLoss|Yes|Supports bf16, fp16, fp32<br>The log_target parameter currently only supports False|
|torch.nn.BCELoss|Yes|Supports fp16, fp32|
|torch.nn.BCEWithLogitsLoss|Yes|Supports bf16, fp16, fp32<br>The target input does not support backward computation|
|torch.nn.MarginRankingLoss|Yes|Supports bf16, fp16, fp32, int8, int32, int64|
|torch.nn.HingeEmbeddingLoss|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|torch.nn.MultiLabelMarginLoss|No|-|
|torch.nn.HuberLoss|Yes|input supports fp32, fp64<br>target supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>May fall back to CPU execution|
|torch.nn.SmoothL1Loss|Yes|Supports bf16, fp16, fp32|
|torch.nn.MultiLabelSoftMarginLoss|Yes|Supports fp16, fp32|
|torch.nn.CosineEmbeddingLoss|No|-|
|torch.nn.MultiMarginLoss|Yes|input supports fp32, fp64<br>target supports int64<br>May fall back to CPU execution|
|torch.nn.TripletMarginLoss|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64<br>May fall back to CPU execution|
|torch.nn.TripletMarginWithDistanceLoss|Yes|Supports bf16, fp16, fp32|
|torch.nn.PixelShuffle|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.nn.PixelUnshuffle|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.nn.Upsample|Yes|Supports bf16, fp16, fp32, fp64|
|torch.nn.UpsamplingNearest2d|Yes|Supports fp16, fp32, uint8<br>May fall back to CPU execution|
|torch.nn.ChannelShuffle|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nn.DataParallel|No|-|
|torch.nn.parallel.DistributedDataParallel|Yes|-|
|torch.nn.parallel.DistributedDataParallel.join|Yes|-|
|torch.nn.parallel.DistributedDataParallel.join_hook|Yes|-|
|torch.nn.parallel.DistributedDataParallel.no_sync|Yes|-|
|torch.nn.parallel.DistributedDataParallel.register_comm_hook|Yes|-|
|torch.nn.utils.clip_grad_norm_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.nn.utils.clip_grad_norm|No|-|
|torch.nn.utils.clip_grad_value_|Yes|Supports bf16, fp16, fp32|
|torch.nn.utils.vector_to_parameters|Yes|Supports bf16, fp16, fp32, fp64, complex64|
|torch.nn.utils.weight_norm|Yes|-|
|torch.nn.utils.spectral_norm|Yes|-|
|torch.nn.utils.remove_spectral_norm|Yes|-|
|torch.nn.utils.skip_init|Yes|-|
|torch.nn.utils.prune.BasePruningMethod|Yes|-|
|torch.nn.utils.prune.BasePruningMethod.apply|Yes|-|
|torch.nn.utils.prune.BasePruningMethod.apply_mask|Yes|Supports fp32|
|torch.nn.utils.prune.BasePruningMethod.compute_mask|Yes|-|
|torch.nn.utils.prune.BasePruningMethod.prune|Yes|Supports fp32|
|torch.nn.utils.prune.BasePruningMethod.remove|Yes|Supports fp32|
|torch.nn.utils.prune.PruningContainer|Yes|-|
|torch.nn.utils.prune.PruningContainer.add_pruning_method|Yes|-|
|torch.nn.utils.prune.PruningContainer.apply|Yes|-|
|torch.nn.utils.prune.PruningContainer.apply_mask|Yes|-|
|torch.nn.utils.prune.PruningContainer.compute_mask|Yes|Supports fp32|
|torch.nn.utils.prune.PruningContainer.prune|Yes|Supports fp32|
|torch.nn.utils.prune.PruningContainer.remove|Yes|Supports fp32|
|torch.nn.utils.prune.Identity|Yes|Supports fp32|
|torch.nn.utils.prune.Identity.apply|Yes|Supports fp32|
|torch.nn.utils.prune.Identity.apply_mask|Yes|Supports fp32|
|torch.nn.utils.prune.Identity.prune|Yes|Supports fp32|
|torch.nn.utils.prune.Identity.remove|Yes|Supports fp32|
|torch.nn.utils.prune.RandomUnstructured|Yes|Supports fp32|
|torch.nn.utils.prune.RandomUnstructured.apply|Yes|Supports fp32|
|torch.nn.utils.prune.RandomUnstructured.apply_mask|Yes|Supports fp32|
|torch.nn.utils.prune.RandomUnstructured.prune|Yes|Supports fp32|
|torch.nn.utils.prune.RandomUnstructured.remove|Yes|-|
|torch.nn.utils.prune.L1Unstructured|Yes|Supports fp32|
|torch.nn.utils.prune.L1Unstructured.apply|Yes|Supports fp32|
|torch.nn.utils.prune.L1Unstructured.apply_mask|Yes|Supports fp32|
|torch.nn.utils.prune.L1Unstructured.prune|Yes|Supports fp32|
|torch.nn.utils.prune.L1Unstructured.remove|Yes|Supports fp32|
|torch.nn.utils.prune.RandomStructured|Yes|Supports fp32|
|torch.nn.utils.prune.RandomStructured.apply|Yes|Supports fp32|
|torch.nn.utils.prune.RandomStructured.apply_mask|Yes|Supports fp32|
|torch.nn.utils.prune.RandomStructured.compute_mask|Yes|Supports fp32|
|torch.nn.utils.prune.RandomStructured.prune|Yes|-|
|torch.nn.utils.prune.RandomStructured.remove|Yes|-|
|torch.nn.utils.prune.LnStructured|Yes|Supports fp32|
|torch.nn.utils.prune.LnStructured.apply|Yes|Supports fp32|
|torch.nn.utils.prune.LnStructured.apply_mask|Yes|Supports fp32|
|torch.nn.utils.prune.LnStructured.compute_mask|Yes|Supports fp32|
|torch.nn.utils.prune.LnStructured.prune|Yes|Supports fp32|
|torch.nn.utils.prune.LnStructured.remove|Yes|Supports fp32|
|torch.nn.utils.prune.CustomFromMask|Yes|Supports int64|
|torch.nn.utils.prune.CustomFromMask.apply|Yes|Supports int64|
|torch.nn.utils.prune.CustomFromMask.apply_mask|Yes|-|
|torch.nn.utils.prune.CustomFromMask.prune|Yes|-|
|torch.nn.utils.prune.CustomFromMask.remove|Yes|-|
|torch.nn.utils.prune.random_unstructured|Yes|-|
|torch.nn.utils.prune.l1_unstructured|Yes|-|
|torch.nn.utils.prune.random_structured|Yes|-|
|torch.nn.utils.prune.ln_structured|Yes|-|
|torch.nn.utils.prune.global_unstructured|Yes|-|
|torch.nn.utils.prune.custom_from_mask|Yes|Supports int64|
|torch.nn.utils.prune.remove|Yes|-|
|torch.nn.utils.prune.is_pruned|Yes|-|
|torch.nn.utils.parametrizations.orthogonal|Yes|-|
|torch.nn.utils.parametrizations.spectral_norm|Yes|-|
|torch.nn.utils.parametrize.register_parametrization|Yes|-|
|torch.nn.utils.parametrize.remove_parametrizations|Yes|-|
|torch.nn.utils.parametrize.cached|Yes|-|
|torch.nn.utils.parametrize.is_parametrized|Yes|-|
|torch.nn.utils.parametrize.ParametrizationList|Yes|-|
|torch.nn.utils.parametrize.ParametrizationList.right_inverse|Yes|Supports fp32|
|torch.nn.utils.stateless.functional_call|Yes|-|
|torch.nn.utils.rnn.PackedSequence|Yes|Supports fp32, int64|
|torch.nn.utils.rnn.PackedSequence.batch_sizes|Yes|-|
|torch.nn.utils.rnn.PackedSequence.count|Yes|Supports fp32|
|torch.nn.utils.rnn.PackedSequence.data|Yes|-|
|torch.nn.utils.rnn.PackedSequence.index|Yes|Supports fp32|
|torch.nn.utils.rnn.PackedSequence.is_cuda|No|-|
|torch.nn.utils.rnn.PackedSequence.is_pinned|Yes|-|
|torch.nn.utils.rnn.PackedSequence.sorted_indices|Yes|-|
|torch.nn.utils.rnn.PackedSequence.to|Yes|Supports fp32, int64|
|torch.nn.utils.rnn.PackedSequence.unsorted_indices|Yes|-|
|torch.nn.utils.rnn.pack_padded_sequence|No|-|
|torch.nn.utils.rnn.pad_packed_sequence|No|-|
|torch.nn.utils.rnn.pad_sequence|Yes|Supports fp16, fp32|
|torch.nn.utils.rnn.pack_sequence|No|-|
|torch.nn.utils.rnn.unpack_sequence|No|-|
|torch.nn.utils.rnn.unpad_sequence|No|-|
|torch.nn.Flatten|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.nn.Unflatten|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.nn.modules.lazy.LazyModuleMixin|Yes|Supports fp32|
|torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params|Yes|Supports fp32|
|torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters|Yes|Supports fp32|
