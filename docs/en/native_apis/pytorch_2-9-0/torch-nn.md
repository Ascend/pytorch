# torch.nn

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:20:47.489Z pushedAt=2026-06-15T03:25:49.204Z -->

> [!NOTE]   
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.nn.parameter.Parameter](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.Parameter.html)|Yes|supports FP32|
|[torch.nn.parameter.Buffer](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.Buffer.html)|Yes|supports FP32|
|[torch.nn.parameter.UninitializedParameter](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.UninitializedParameter.html)|Yes|-|
|[torch.nn.parameter.UninitializedParameter.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.UninitializedParameter.html#torch.nn.parameter.UninitializedParameter.cls_to_become)|Yes|-|
|[torch.nn.parameter.UninitializedBuffer](https://pytorch.org/docs/2.9/generated/torch.nn.parameter.UninitializedBuffer.html)|Yes|-|
|[torch.nn.Module](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html)|Yes|supports FP32|
|[torch.nn.Module.add_module](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.add_module)|Yes|supports FP32|
|[torch.nn.Module.apply](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.apply)|Yes|supports FP32|
|[torch.nn.Module.bfloat16](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.bfloat16)|Yes|-|
|[torch.nn.Module.buffers](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.buffers)|Yes|-|
|[torch.nn.Module.children](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.children)|Yes|supports FP32|
|[torch.nn.Module.compile](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.compile)|Yes|-|
|[torch.nn.Module.cpu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.cpu)|Yes|supports FP32|
|[torch.nn.Module.cuda](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.cuda)|Yes|supports FP32|
|[torch.nn.Module.double](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.double)|Yes|-|
|[torch.nn.Module.eval](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.eval)|Yes|supports FP32, INT64|
|[torch.nn.Module.extra_repr](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.extra_repr)|Yes|supports FP32|
|[torch.nn.Module.float](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.float)|Yes|supports FP16, FP32|
|[torch.nn.Module.forward](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.forward)|Yes|supports FP32|
|[torch.nn.Module.get_buffer](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_buffer)|Yes|-|
|[torch.nn.Module.get_extra_state](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_extra_state)|Yes|-|
|[torch.nn.Module.get_parameter](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_parameter)|Yes|supports FP32|
|[torch.nn.Module.get_submodule](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.get_submodule)|Yes|supports FP32|
|[torch.nn.Module.half](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.half)|Yes|supports FP16, FP32|
|[torch.nn.Module.ipu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.ipu)|No|-|
|[torch.nn.Module.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)|Yes|supports FP32|
|[torch.nn.Module.modules](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.modules)|Yes|supports FP32|
|[torch.nn.Module.named_buffers](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_buffers)|Yes|-|
|[torch.nn.Module.named_children](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_children)|Yes|supports FP32|
|[torch.nn.Module.named_modules](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_modules)|Yes|supports FP32|
|[torch.nn.Module.named_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.named_parameters)|Yes|-|
|[torch.nn.Module.parameters](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.parameters)|Yes|-|
|[torch.nn.Module.register_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_backward_hook)|Yes|supports FP32|
|[torch.nn.Module.register_buffer](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)|Yes|supports FP32|
|[torch.nn.Module.register_forward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)|Yes|supports FP32|
|[torch.nn.Module.register_forward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)|Yes|supports FP32|
|[torch.nn.Module.register_full_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook)|Yes|supports FP32|
|[torch.nn.Module.register_full_backward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook)|Yes|supports FP32|
|[torch.nn.Module.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_load_state_dict_post_hook)|Yes|supports FP32|
|[torch.nn.Module.register_module](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_module)|Yes|supports FP32|
|[torch.nn.Module.register_parameter](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_parameter)|Yes|-|
|[torch.nn.Module.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_state_dict_pre_hook)|Yes|-|
|[torch.nn.Module.requires_grad_](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.requires_grad_)|Yes|-|
|[torch.nn.Module.set_extra_state](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.set_extra_state)|Yes|-|
|[torch.nn.Module.set_submodule](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.set_submodule)|Yes|-|
|[torch.nn.Module.share_memory](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.share_memory)|No|-|
|[torch.nn.Module.state_dict](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.state_dict)|Yes|supports FP32|
|[torch.nn.Module.to](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.to)|Yes|supports FP32|
|[torch.nn.Module.to_empty](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.to_empty)|Yes|supports FP32|
|[torch.nn.Module.train](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.train)|Yes|supports FP32|
|[torch.nn.Module.type](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.type)|Yes|supports FP16, FP32, INT64|
|[torch.nn.Module.xpu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.xpu)|Yes|NPU-adapted name is torch.nn.Module.npu|
|[torch.nn.Module.npu](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.npu)|No|-|
|[torch.nn.Module.zero_grad](https://pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.zero_grad)|Yes|supports FP32|
|[torch.nn.Sequential](https://pytorch.org/docs/2.9/generated/torch.nn.Sequential.html)|Yes|supports FP32|
|[torch.nn.Sequential.append](https://pytorch.org/docs/2.9/generated/torch.nn.Sequential.html#torch.nn.Sequential.append)|Yes|supports FP32|
|[torch.nn.ModuleList](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html)|Yes|supports FP32|
|[torch.nn.ModuleList.append](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.append)|Yes|supports FP32|
|[torch.nn.ModuleList.extend](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.extend)|Yes|supports FP32|
|[torch.nn.ModuleList.insert](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.insert)|Yes|supports FP32|
|[torch.nn.ModuleDict](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html)|Yes|supports FP32|
|[torch.nn.ModuleDict.clear](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.clear)|Yes|supports FP32|
|[torch.nn.ModuleDict.items](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.items)|Yes|supports FP32|
|[torch.nn.ModuleDict.keys](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.keys)|Yes|supports FP32|
|[torch.nn.ModuleDict.pop](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.pop)|Yes|supports FP32|
|[torch.nn.ModuleDict.update](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.update)|Yes|supports FP32|
|[torch.nn.ModuleDict.values](https://pytorch.org/docs/2.9/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.values)|Yes|supports FP32|
|[torch.nn.ParameterList](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterList.html)|Yes|supports FP32|
|[torch.nn.ParameterList.append](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterList.html#torch.nn.ParameterList.append)|Yes|supports FP32|
|[torch.nn.ParameterList.extend](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterList.html#torch.nn.ParameterList.extend)|Yes|supports FP32|
|[torch.nn.ParameterDict](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html)|Yes|supports FP32|
|[torch.nn.ParameterDict.clear](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.clear)|Yes|supports FP32|
|[torch.nn.ParameterDict.copy](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.copy)|Yes|supports FP32|
|[torch.nn.ParameterDict.fromkeys](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.fromkeys)|Yes|supports FP32|
|[torch.nn.ParameterDict.get](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.get)|Yes|supports FP32|
|[torch.nn.ParameterDict.items](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.items)|Yes|supports FP32|
|[torch.nn.ParameterDict.keys](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.keys)|Yes|supports FP32|
|[torch.nn.ParameterDict.pop](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.pop)|Yes|supports FP32|
|[torch.nn.ParameterDict.popitem](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.popitem)|Yes|supports FP32|
|[torch.nn.ParameterDict.setdefault](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.setdefault)|Yes|supports FP32|
|[torch.nn.ParameterDict.update](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.update)|Yes|supports FP32|
|[torch.nn.ParameterDict.values](https://pytorch.org/docs/2.9/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.values)|Yes|supports FP32|
|[torch.nn.modules.module.register_module_forward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_forward_pre_hook.html)|Yes|supports FP32|
|[torch.nn.modules.module.register_module_forward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_forward_hook.html)|Yes|supports FP32|
|[torch.nn.modules.module.register_module_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_backward_hook.html)|Yes|supports FP32|
|[torch.nn.modules.module.register_module_full_backward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_full_backward_pre_hook.html)|No|-|
|[torch.nn.modules.module.register_module_full_backward_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_full_backward_hook.html)|Yes|supports FP32|
|[torch.nn.modules.module.register_module_buffer_registration_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_buffer_registration_hook.html)|No|-|
|[torch.nn.modules.module.register_module_module_registration_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_module_registration_hook.html)|No|-|
|[torch.nn.modules.module.register_module_parameter_registration_hook](https://pytorch.org/docs/2.9/generated/torch.nn.modules.module.register_module_parameter_registration_hook.html)|No|-|
|[torch.nn.Conv1d](https://pytorch.org/docs/2.9/generated/torch.nn.Conv1d.html)|Yes|supports FP16, FP32|
|[torch.nn.Conv2d](https://pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)|Yes|supports BF16, FP16, FP32<br><term>Atlas A2 Training Series</term>, in default scenarios, if compilation is triggered frequently, it is recommended to manually set torch.npu.config.allow_internal_format to False to control the input parameters to disable internal format and avoid online compilation|
|[torch.nn.Conv3d](https://pytorch.org/docs/2.9/generated/torch.nn.Conv3d.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.ConvTranspose1d](https://pytorch.org/docs/2.9/generated/torch.nn.ConvTranspose1d.html)|Yes|supports FP32|
|[torch.nn.ConvTranspose2d](https://pytorch.org/docs/2.9/generated/torch.nn.ConvTranspose2d.html)|Yes|supports FP16, FP32<br><term>Atlas Training Series</term>/<term>Atlas A2 Training Series</term>, torch.npu.config.allow_internal_format must be manually set to False to support 3-dimensional input|
|[torch.nn.ConvTranspose3d](https://pytorch.org/docs/2.9/generated/torch.nn.ConvTranspose3d.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.LazyConv1d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv1d.html)|Yes|supports FP16, FP32|
|[torch.nn.LazyConv1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv1d.html#torch.nn.LazyConv1d.cls_to_become)|Yes|-|
|[torch.nn.LazyConv2d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv2d.html)|Yes|supports FP16, FP32|
|[torch.nn.LazyConv2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d.cls_to_become)|Yes|-|
|[torch.nn.LazyConv3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConv3d.html#torch.nn.LazyConv3d.cls_to_become)|Yes|-|
|[torch.nn.LazyConvTranspose1d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose1d.html)|Yes|supports FP16|
|[torch.nn.LazyConvTranspose1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose1d.html#torch.nn.LazyConvTranspose1d.cls_to_become)|Yes|-|
|[torch.nn.LazyConvTranspose2d](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose2d.html)|Yes|supports FP16, FP32|
|[torch.nn.LazyConvTranspose2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose2d.html#torch.nn.LazyConvTranspose2d.cls_to_become)|Yes|-|
|[torch.nn.LazyConvTranspose3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyConvTranspose3d.html#torch.nn.LazyConvTranspose3d.cls_to_become)|Yes|-|
|[torch.nn.Unfold](https://pytorch.org/docs/2.9/generated/torch.nn.Unfold.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.Fold](https://pytorch.org/docs/2.9/generated/torch.nn.Fold.html)|Yes|supports FP16|
|[torch.nn.MaxPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxPool1d.html)|No|-|
|[torch.nn.MaxPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxPool2d.html)|Yes|supports BF16, FP16, FP32<br>By setting torch_npu.npu.use_compatible_impl(True), memory consistency with the community counterpart is ensured|
|[torch.nn.MaxPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxPool3d.html)|No|-|
|[torch.nn.MaxUnpool1d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxUnpool1d.html)|Yes|supports FP16, FP32|
|[torch.nn.MaxUnpool2d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxUnpool2d.html)|Yes|supports FP16, FP32|
|[torch.nn.MaxUnpool3d](https://pytorch.org/docs/2.9/generated/torch.nn.MaxUnpool3d.html)|No|-|
|[torch.nn.AvgPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.AvgPool1d.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.AvgPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.AvgPool2d.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.AvgPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.AvgPool3d.html)|No|-|
|[torch.nn.LPPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.LPPool1d.html)|Yes|supports FP16, FP32, UINT8, INT8, INT16, INT32, INT64, BOOL|
|[torch.nn.LPPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.LPPool2d.html)|Yes|supports FP16, FP32, INT16, INT32, INT64, BOOL|
|[torch.nn.AdaptiveMaxPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveMaxPool1d.html)|No|-|
|[torch.nn.AdaptiveMaxPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveMaxPool2d.html)|No|-|
|[torch.nn.AdaptiveMaxPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveMaxPool3d.html)|Yes|supports FP32, FP64|
|[torch.nn.AdaptiveAvgPool1d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool1d.html)|Yes|supports FP16, FP32|
|[torch.nn.AdaptiveAvgPool2d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool2d.html)|Yes|supports FP16, FP32|
|[torch.nn.AdaptiveAvgPool3d](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool3d.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.ReflectionPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ReflectionPad1d.html)|Yes|supports FP16, FP32|
|[torch.nn.ReflectionPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ReflectionPad2d.html)|Yes|supports FP16, FP32|
|[torch.nn.ReflectionPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ReflectionPad3d.html)|No|-|
|[torch.nn.ReplicationPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ReplicationPad1d.html)|Yes|supports FP16, FP32, COMPLEX64, COMPLEX128|
|[torch.nn.ReplicationPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ReplicationPad2d.html)|Yes|supports FP16, FP32, COMPLEX64, COMPLEX128|
|[torch.nn.ReplicationPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ReplicationPad3d.html)|No|-|
|[torch.nn.ZeroPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ZeroPad1d.html)|Yes|supports BF16, FP16, FP32, FP64, COMPLEX64, COMPLEX128<br>Supports 2-3 dimensions|
|[torch.nn.ZeroPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ZeroPad2d.html)|Yes|May Fallback to CPU execution|
|[torch.nn.ZeroPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ZeroPad3d.html)|Yes|supports BF16, FP16, FP32, FP64, COMPLEX64, COMPLEX128<br>Supports 5-6 dimensions|
|[torch.nn.ConstantPad1d](https://pytorch.org/docs/2.9/generated/torch.nn.ConstantPad1d.html)|Yes|supports INT8, BOOL<br>Performance degradation may occur when input x has six or more dimensions|
|[torch.nn.ConstantPad2d](https://pytorch.org/docs/2.9/generated/torch.nn.ConstantPad2d.html)|Yes|supports BF16, FP16, FP32, FP64, UINT8, INT8, INT16, INT32, INT64, BOOL, COMPLEX64, COMPLEX128<br>Performance degradation may occur when input x has six or more dimensions|
|[torch.nn.ConstantPad3d](https://pytorch.org/docs/2.9/generated/torch.nn.ConstantPad3d.html)|Yes|supports FP16, FP32, UINT8, INT8, INT16, INT32, INT64, BOOL, COMPLEX64, COMPLEX128<br>Performance degradation may occur when input x has six or more dimensions|
|[torch.nn.ELU](https://pytorch.org/docs/2.9/generated/torch.nn.ELU.html)|Yes|supports BF16, FP16, FP32, FP64|
|[torch.nn.Hardshrink](https://pytorch.org/docs/2.9/generated/torch.nn.Hardshrink.html)|Yes|supports FP16, FP32<br>May Fallback to CPU execution|
|[torch.nn.Hardsigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.Hardsigmoid.html)|Yes|supports FP16, FP32, INT32<br>May Fallback to CPU execution|
|[torch.nn.Hardtanh](https://pytorch.org/docs/2.9/generated/torch.nn.Hardtanh.html)|Yes|supports BF16, FP16, FP32, FP64, UINT8, INT8, INT16, INT32, INT64|
|[torch.nn.Hardswish](https://pytorch.org/docs/2.9/generated/torch.nn.Hardswish.html)|Yes|supports FP16, FP32|
|[torch.nn.LeakyReLU](https://pytorch.org/docs/2.9/generated/torch.nn.LeakyReLU.html)|Yes|supports BF16, FP16, FP32, FP64|
|[torch.nn.LogSigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.LogSigmoid.html)|Yes|supports FP16, FP32|
|[torch.nn.MultiheadAttention](https://pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.MultiheadAttention.forward](https://pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward)|Yes|supports BF16, FP16, FP32|
|[torch.nn.PReLU](https://pytorch.org/docs/2.9/generated/torch.nn.PReLU.html)|Yes|supports FP32|
|[torch.nn.ReLU](https://pytorch.org/docs/2.9/generated/torch.nn.ReLU.html)|Yes|supports BF16, FP16, FP32, UINT8, INT8, INT32, INT64|
|[torch.nn.ReLU6](https://pytorch.org/docs/2.9/generated/torch.nn.ReLU6.html)|Yes|supports BF16, FP16, FP32, FP64, UINT8, INT8, INT16, INT32, INT64|
|[torch.nn.RReLU](https://pytorch.org/docs/2.9/generated/torch.nn.RReLU.html)|No|-|
|[torch.nn.SELU](https://pytorch.org/docs/2.9/generated/torch.nn.SELU.html)|Yes|supports FP16, FP32, FP64, UINT8, INT8, INT16, INT32, INT64, BOOL|
|[torch.nn.CELU](https://pytorch.org/docs/2.9/generated/torch.nn.CELU.html)|Yes|supports FP16, FP32|
|[torch.nn.GELU](https://pytorch.org/docs/2.9/generated/torch.nn.GELU.html)|Yes|supports BF16, FP16, FP32<br>The approximate parameter only supports being set to tanh|
|[torch.nn.Sigmoid](https://pytorch.org/docs/2.9/generated/torch.nn.Sigmoid.html)|Yes|supports BF16, FP16, FP32, UINT8, INT8, INT16, INT32, INT64, BOOL, COMPLEX64, COMPLEX128|
|[torch.nn.SiLU](https://pytorch.org/docs/2.9/generated/torch.nn.SiLU.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.Mish](https://pytorch.org/docs/2.9/generated/torch.nn.Mish.html)|Yes|supports FP16, FP32|
|[torch.nn.Softplus](https://pytorch.org/docs/2.9/generated/torch.nn.Softplus.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.Softshrink](https://pytorch.org/docs/2.9/generated/torch.nn.Softshrink.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.Softsign](https://pytorch.org/docs/2.9/generated/torch.nn.Softsign.html)|Yes|supports BF16, FP16, FP32, UINT8, INT8, INT16, INT32, INT64|
|[torch.nn.Tanh](https://pytorch.org/docs/2.9/generated/torch.nn.Tanh.html)|Yes|supports BF16, FP16, FP32, UINT8, INT8, INT16, INT32, INT64, BOOL|
|[torch.nn.Tanhshrink](https://pytorch.org/docs/2.9/generated/torch.nn.Tanhshrink.html)|Yes|supports FP16, FP32, UINT8, INT8, INT16, INT32, INT64<br>May Fallback to CPU execution|
|[torch.nn.Threshold](https://pytorch.org/docs/2.9/generated/torch.nn.Threshold.html)|Yes|supports FP16, FP32, UINT8, INT8, INT16, INT32, INT64|
|[torch.nn.GLU](https://pytorch.org/docs/2.9/generated/torch.nn.GLU.html)|Yes|supports FP16, FP32|
|[torch.nn.Softmin](https://pytorch.org/docs/2.9/generated/torch.nn.Softmin.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.Softmax](https://pytorch.org/docs/2.9/generated/torch.nn.Softmax.html)|Yes|supports BF16, FP16, FP32, FP64|
|[torch.nn.Softmax2d](https://pytorch.org/docs/2.9/generated/torch.nn.Softmax2d.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.LogSoftmax](https://pytorch.org/docs/2.9/generated/torch.nn.LogSoftmax.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.AdaptiveLogSoftmaxWithLoss](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html)|No|-|
|[torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob)|No|-|
|[torch.nn.AdaptiveLogSoftmaxWithLoss.predict](https://pytorch.org/docs/2.9/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss.predict)|No|-|
|[torch.nn.BatchNorm1d](https://pytorch.org/docs/2.9/generated/torch.nn.BatchNorm1d.html)|Yes|supports FP16, FP32|
|[torch.nn.BatchNorm2d](https://pytorch.org/docs/2.9/generated/torch.nn.BatchNorm2d.html)|Yes|supports FP16, FP32|
|[torch.nn.BatchNorm3d](https://pytorch.org/docs/2.9/generated/torch.nn.BatchNorm3d.html)|Yes|supports FP16, FP32|
|[torch.nn.LazyBatchNorm1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyBatchNorm1d.html#torch.nn.LazyBatchNorm1d.cls_to_become)|Yes|-|
|[torch.nn.LazyBatchNorm2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyBatchNorm2d.html#torch.nn.LazyBatchNorm2d.cls_to_become)|Yes|-|
|[torch.nn.LazyBatchNorm3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyBatchNorm3d.html#torch.nn.LazyBatchNorm3d.cls_to_become)|Yes|-|
|[torch.nn.GroupNorm](https://pytorch.org/docs/2.9/generated/torch.nn.GroupNorm.html)|Yes|supports FP32<br>The eps parameter must be greater than 0<br>The scenario with jit_compile=True is not supported<br>This API only supports input with 2 or more dimensions. During backward propagation, this API does not support scenarios where the input dimension is not 4, or the input num_groups is not divisible by 32, or the C-axis dimension is not divisible by (10 * num_groups)|
|[torch.nn.SyncBatchNorm](https://pytorch.org/docs/2.9/generated/torch.nn.SyncBatchNorm.html)|Yes|supports FP16, FP32|
|[torch.nn.SyncBatchNorm.convert_sync_batchnorm](https://pytorch.org/docs/2.9/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm)|Yes|-|
|[torch.nn.LazyInstanceNorm1d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyInstanceNorm1d.html#torch.nn.LazyInstanceNorm1d.cls_to_become)|Yes|-|
|[torch.nn.LazyInstanceNorm2d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyInstanceNorm2d.html#torch.nn.LazyInstanceNorm2d.cls_to_become)|Yes|-|
|[torch.nn.LazyInstanceNorm3d.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyInstanceNorm3d.html#torch.nn.LazyInstanceNorm3d.cls_to_become)|Yes|-|
|[torch.nn.LayerNorm](https://pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html)|Yes|supports BF16, FP16, FP32<br>By setting torch_npu.npu.use_compatible_impl(True), this interface switches from the aclnnLayerNorm operator to the aclnnFastLayerNorm operator, ensuring memory consistency with the community counterpart.|
|[torch.nn.RNNBase](https://pytorch.org/docs/2.9/generated/torch.nn.RNNBase.html)|No|-|
|[torch.nn.RNNBase.flatten_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.RNNBase.html#torch.nn.RNNBase.flatten_parameters)|No|-|
|[torch.nn.RNN](https://pytorch.org/docs/2.9/generated/torch.nn.RNN.html)|No|-|
|[torch.nn.LSTM](https://pytorch.org/docs/2.9/generated/torch.nn.LSTM.html)|Yes|supports FP32<br>The proj_size parameter is not supported<br>The dropout parameter is not supported<br>The input parameter does not support 2 dimensions|
|[torch.nn.GRU](https://pytorch.org/docs/2.9/generated/torch.nn.GRU.html)|No|-|
|[torch.nn.RNNCell](https://pytorch.org/docs/2.9/generated/torch.nn.RNNCell.html)|No|-|
|[torch.nn.LSTMCell](https://pytorch.org/docs/2.9/generated/torch.nn.LSTMCell.html)|Yes|The interface currently does not support jit_compile=False. When using in this mode, add "DynamicGRUV2" to the "NPU_FUZZY_COMPILE_BLACKLIST" option. For details, refer to [Example of Adding a Binary Blocklist](../example_of_adding_a_binary_blocklist.md)|
|[torch.nn.GRUCell](https://pytorch.org/docs/2.9/generated/torch.nn.GRUCell.html)|Yes|supports FP16, FP32|
|[torch.nn.Transformer](https://pytorch.org/docs/2.9/generated/torch.nn.Transformer.html)|Yes|supports FP16, FP32|
|[torch.nn.Transformer.forward](https://pytorch.org/docs/2.9/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward)|No|-|
|[torch.nn.TransformerEncoder](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoder.html)|No|-|
|[torch.nn.TransformerEncoder.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder.forward)|Yes|supports FP32|
|[torch.nn.TransformerDecoder](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerDecoder.html)|No|-|
|[torch.nn.TransformerDecoder.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder.forward)|No|-|
|[torch.nn.TransformerEncoderLayer.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer.forward)|No|-|
|[torch.nn.TransformerDecoderLayer.forward](https://pytorch.org/docs/2.9/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer.forward)|No|-|
|[torch.nn.Identity](https://pytorch.org/docs/2.9/generated/torch.nn.Identity.html)|Yes|supports FP32|
|[torch.nn.Linear](https://pytorch.org/docs/2.9/generated/torch.nn.Linear.html)|Yes|supports FP16, FP32|
|[torch.nn.Bilinear](https://pytorch.org/docs/2.9/generated/torch.nn.Bilinear.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.LazyLinear](https://pytorch.org/docs/2.9/generated/torch.nn.LazyLinear.html)|Yes|supports FP16, FP32|
|[torch.nn.LazyLinear.cls_to_become](https://pytorch.org/docs/2.9/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear.cls_to_become)|No|-|
|[torch.nn.Dropout](https://pytorch.org/docs/2.9/generated/torch.nn.Dropout.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.Dropout2d](https://pytorch.org/docs/2.9/generated/torch.nn.Dropout2d.html)|Yes|supports FP16, FP32, INT64, bool|
|[torch.nn.AlphaDropout](https://pytorch.org/docs/2.9/generated/torch.nn.AlphaDropout.html)|Yes|supports FP16, FP32|
|[torch.nn.FeatureAlphaDropout](https://pytorch.org/docs/2.9/generated/torch.nn.FeatureAlphaDropout.html)|Yes|supports FP16, FP32|
|[torch.nn.Embedding](https://pytorch.org/docs/2.9/generated/torch.nn.Embedding.html)|Yes|supports INT32, INT64<br>The attribute max_norm does not support NaN, only non-negative values are supported|
|[torch.nn.Embedding.from_pretrained](https://pytorch.org/docs/2.9/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained)|Yes|supports FP64|
|[torch.nn.EmbeddingBag](https://pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html)|Yes|supports INT32, INT64<br>Only max_norm greater than or equal to 0 is supported|
|[torch.nn.EmbeddingBag.forward](https://pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward)|Yes|supports INT64|
|[torch.nn.EmbeddingBag.from_pretrained](https://pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.from_pretrained)|Yes|supports INT64|
|[torch.nn.L1Loss](https://pytorch.org/docs/2.9/generated/torch.nn.L1Loss.html)|Yes|supports FP16, FP32, INT64|
|[torch.nn.MSELoss](https://pytorch.org/docs/2.9/generated/torch.nn.MSELoss.html)|Yes|supports FP16, FP32|
|[torch.nn.CrossEntropyLoss](https://pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)|Yes|supports FP16, FP32|
|[torch.nn.CTCLoss](https://pytorch.org/docs/2.9/generated/torch.nn.CTCLoss.html)|Yes|supports FP32, FP64<br>2D input for log_probs is not supported|
|[torch.nn.NLLLoss](https://pytorch.org/docs/2.9/generated/torch.nn.NLLLoss.html)|Yes|supports FP16, FP32<br>Each element value in target must be greater than or equal to 0 and less than the number of classes in input|
|[torch.nn.PoissonNLLLoss](https://pytorch.org/docs/2.9/generated/torch.nn.PoissonNLLLoss.html)|Yes|supports BF16, FP16, FP32, INT64|
|[torch.nn.GaussianNLLLoss](https://pytorch.org/docs/2.9/generated/torch.nn.GaussianNLLLoss.html)|Yes|supports BF16, FP16, FP32, INT16, INT32, INT64|
|[torch.nn.KLDivLoss](https://pytorch.org/docs/2.9/generated/torch.nn.KLDivLoss.html)|Yes|supports BF16, FP16, FP32<br>The log_target parameter currently only supports False|
|[torch.nn.BCELoss](https://pytorch.org/docs/2.9/generated/torch.nn.BCELoss.html)|Yes|supports FP16, FP32|
|[torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html)|Yes|supports BF16, FP16, FP32<br>The input argument target does not support backward computation|
|[torch.nn.MarginRankingLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MarginRankingLoss.html)|Yes|supports BF16, FP16, FP32, INT8, INT32, INT64|
|[torch.nn.HingeEmbeddingLoss](https://pytorch.org/docs/2.9/generated/torch.nn.HingeEmbeddingLoss.html)|Yes|supports BF16, FP16, FP32, UINT8, INT8, INT16, INT32, INT64|
|[torch.nn.MultiLabelMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MultiLabelMarginLoss.html)|No|-|
|[torch.nn.HuberLoss](https://pytorch.org/docs/2.9/generated/torch.nn.HuberLoss.html)|Yes|input supports FP32, FP64<br>target supports BF16, FP16, FP32, FP64, UINT8, INT8, INT16, INT32, INT64, bool<br>May Fallback to CPU execution|
|[torch.nn.SmoothL1Loss](https://pytorch.org/docs/2.9/generated/torch.nn.SmoothL1Loss.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MultiLabelSoftMarginLoss.html)|Yes|supports FP16, FP32|
|[torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/2.9/generated/torch.nn.CosineEmbeddingLoss.html)|No|-|
|[torch.nn.MultiMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.MultiMarginLoss.html)|Yes|input supports FP32, FP64<br>target supports INT64<br>May Fallback to CPU execution|
|[torch.nn.TripletMarginLoss](https://pytorch.org/docs/2.9/generated/torch.nn.TripletMarginLoss.html)|Yes|supports FP16, FP32, UINT8, INT8, INT16, INT32, INT64<br>May Fallback to CPU execution|
|[torch.nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/2.9/generated/torch.nn.TripletMarginWithDistanceLoss.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.PixelShuffle](https://pytorch.org/docs/2.9/generated/torch.nn.PixelShuffle.html)|Yes|supports BF16, FP16, FP32, FP64, UINT8, INT8, INT16, INT32, INT64, bool|
|[torch.nn.PixelUnshuffle](https://pytorch.org/docs/2.9/generated/torch.nn.PixelUnshuffle.html)|Yes|supports FP16, FP32, FP64, UINT8, INT8, INT16, INT32, INT64, bool|
|[torch.nn.Upsample](https://pytorch.org/docs/2.9/generated/torch.nn.Upsample.html)|Yes|supports BF16, FP16, FP32, FP64|
|[torch.nn.UpsamplingNearest2d](https://pytorch.org/docs/2.9/generated/torch.nn.UpsamplingNearest2d.html)|Yes|supports FP16, FP32, UINT8<br>May Fallback to CPU execution|
|[torch.nn.ChannelShuffle](https://pytorch.org/docs/2.9/generated/torch.nn.ChannelShuffle.html)|Yes|supports BF16, FP16, FP32, UINT8, INT8, INT16, INT32, INT64, bool, complex64, complex128|
|[torch.nn.DataParallel](https://pytorch.org/docs/2.9/generated/torch.nn.DataParallel.html)|No|-|
|[torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html)|Yes|-|
|[torch.nn.parallel.DistributedDataParallel.join](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join)|Yes|-|
|[torch.nn.parallel.DistributedDataParallel.join_hook](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join_hook)|Yes|-|
|[torch.nn.parallel.DistributedDataParallel.no_sync](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync)|Yes|-|
|[torch.nn.parallel.DistributedDataParallel.register_comm_hook](https://pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook)|Yes|-|
|[torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm_.html)|Yes|supports BF16, FP16, FP32, FP64, UINT8, INT8, INT16, INT32, INT64, bool|
|[torch.nn.utils.clip_grad_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm.html)|No|-|
|[torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_value_.html)|Yes|supports BF16, FP16, FP32|
|[torch.nn.utils.vector_to_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.utils.vector_to_parameters.html)|Yes|supports BF16, FP16, FP32, FP64, complex64|
|[torch.nn.utils.weight_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.weight_norm.html)|Yes|-|
|[torch.nn.utils.spectral_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.spectral_norm.html)|Yes|-|
|[torch.nn.utils.remove_spectral_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.remove_spectral_norm.html)|Yes|-|
|[torch.nn.utils.skip_init](https://pytorch.org/docs/2.9/generated/torch.nn.utils.skip_init.html)|Yes|-|
|[torch.nn.utils.prune.BasePruningMethod](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html)|Yes|-|
|[torch.nn.utils.prune.BasePruningMethod.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.apply)|Yes|-|
|[torch.nn.utils.prune.BasePruningMethod.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.apply_mask)|Yes|supports FP32|
|[torch.nn.utils.prune.BasePruningMethod.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.compute_mask)|Yes|-|
|[torch.nn.utils.prune.BasePruningMethod.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.prune)|Yes|supports FP32|
|[torch.nn.utils.prune.BasePruningMethod.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.remove)|Yes|supports FP32|
|[torch.nn.utils.prune.PruningContainer](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html)|Yes|-|
|[torch.nn.utils.prune.PruningContainer.add_pruning_method](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.add_pruning_method)|Yes|-|
|[torch.nn.utils.prune.PruningContainer.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.apply)|Yes|-|
|[torch.nn.utils.prune.PruningContainer.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.apply_mask)|Yes|-|
|[torch.nn.utils.prune.PruningContainer.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.compute_mask)|Yes|supports FP32|
|[torch.nn.utils.prune.PruningContainer.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.prune)|Yes|supports FP32|
|[torch.nn.utils.prune.PruningContainer.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.remove)|Yes|supports FP32|
|[torch.nn.utils.prune.Identity](https://pytorch.org/docs/2.9/utils.html#torch.nn.utils.prune.Identity)|Yes|supports FP32|
|[torch.nn.utils.prune.Identity.apply](https://pytorch.org/docs/2.9/nn.html#torch.nn.utils.prune.Identity.apply)|Yes|supports FP32|
|[torch.nn.utils.prune.Identity.apply_mask](https://pytorch.org/docs/2.9/nn.html#torch.nn.utils.prune.Identity.apply_mask)|Yes|supports FP32|
|[torch.nn.utils.prune.Identity.prune](https://pytorch.org/docs/2.9/nn.html#torch.nn.utils.prune.Identity.prune)|Yes|supports FP32|
|[torch.nn.utils.prune.Identity.remove](https://pytorch.org/docs/2.9/nn.html#torch.nn.utils.prune.Identity.remove)|Yes|supports FP32|
|[torch.nn.utils.prune.RandomUnstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html)|Yes|supports FP32|
|[torch.nn.utils.prune.RandomUnstructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.apply)|Yes|supports FP32|
|[torch.nn.utils.prune.RandomUnstructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.apply_mask)|Yes|supports FP32|
|[torch.nn.utils.prune.RandomUnstructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.prune)|Yes|supports FP32|
|[torch.nn.utils.prune.RandomUnstructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.remove)|Yes|-|
|[torch.nn.utils.prune.L1Unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html)|Yes|supports FP32|
|[torch.nn.utils.prune.L1Unstructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.apply)|Yes|supports FP32|
|[torch.nn.utils.prune.L1Unstructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.apply_mask)|Yes|supports FP32|
|[torch.nn.utils.prune.L1Unstructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.prune)|Yes|supports FP32|
|[torch.nn.utils.prune.L1Unstructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.remove)|Yes|supports FP32|
|[torch.nn.utils.prune.RandomStructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html)|Yes|supports FP32|
|[torch.nn.utils.prune.RandomStructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.apply)|Yes|supports FP32|
|[torch.nn.utils.prune.RandomStructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.apply_mask)|Yes|supports FP32|
|[torch.nn.utils.prune.RandomStructured.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.compute_mask)|Yes|supports FP32|
|[torch.nn.utils.prune.RandomStructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.prune)|Yes|-|
|[torch.nn.utils.prune.RandomStructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.remove)|Yes|-|
|[torch.nn.utils.prune.LnStructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html)|Yes|supports FP32|
|[torch.nn.utils.prune.LnStructured.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.apply)|Yes|supports FP32|
|[torch.nn.utils.prune.LnStructured.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.apply_mask)|Yes|supports FP32|
|[torch.nn.utils.prune.LnStructured.compute_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.compute_mask)|Yes|supports FP32|
|[torch.nn.utils.prune.LnStructured.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.prune)|Yes|supports FP32|
|[torch.nn.utils.prune.LnStructured.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.remove)|Yes|supports FP32|
|[torch.nn.utils.prune.CustomFromMask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html)|Yes|supports INT64|
|[torch.nn.utils.prune.CustomFromMask.apply](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.apply)|Yes|supports INT64|
|[torch.nn.utils.prune.CustomFromMask.apply_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.apply_mask)|Yes|-|
|[torch.nn.utils.prune.CustomFromMask.prune](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.prune)|Yes|-|
|[torch.nn.utils.prune.CustomFromMask.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.remove)|Yes|-|
|[torch.nn.utils.prune.random_unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.random_unstructured.html)|Yes|-|
|[torch.nn.utils.prune.l1_unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.l1_unstructured.html)|Yes|-|
|[torch.nn.utils.prune.random_structured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.random_structured.html)|Yes|-|
|[torch.nn.utils.prune.ln_structured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.ln_structured.html)|Yes|-|
|[torch.nn.utils.prune.global_unstructured](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.global_unstructured.html)|Yes|-|
|[torch.nn.utils.prune.identity](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.identity.html)|Yes|-|
|[torch.nn.utils.prune.custom_from_mask](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.custom_from_mask.html)|Yes|supports INT64|
|[torch.nn.utils.prune.remove](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.remove.html)|Yes|-|
|[torch.nn.utils.prune.is_pruned](https://pytorch.org/docs/2.9/generated/torch.nn.utils.prune.is_pruned.html)|Yes|-|
|[torch.nn.utils.parametrizations.orthogonal](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrizations.orthogonal.html)|Yes|-|
|[torch.nn.utils.parametrizations.spectral_norm](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrizations.spectral_norm.html)|Yes|-|
|[torch.nn.utils.parametrize.register_parametrization](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.register_parametrization.html)|Yes|-|
|[torch.nn.utils.parametrize.remove_parametrizations](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.remove_parametrizations.html)|Yes|-|
|[torch.nn.utils.parametrize.cached](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.cached.html)|Yes|-|
|[torch.nn.utils.parametrize.is_parametrized](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.is_parametrized.html)|Yes|-|
|[torch.nn.utils.parametrize.ParametrizationList](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.ParametrizationList.html)|Yes|-|
|[torch.nn.utils.parametrize.ParametrizationList.right_inverse](https://pytorch.org/docs/2.9/generated/torch.nn.utils.parametrize.ParametrizationList.html#torch.nn.utils.parametrize.ParametrizationList.right_inverse)|Yes|supports FP32|
|[torch.nn.utils.stateless.functional_call](https://pytorch.org/docs/2.9/generated/torch.nn.utils.stateless.functional_call.html)|Yes|-|
|[torch.nn.utils.rnn.PackedSequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html)|Yes|supports FP32, INT64|
|[torch.nn.utils.rnn.PackedSequence.batch_sizes](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.batch_sizes)|Yes|-|
|[torch.nn.utils.rnn.PackedSequence.count](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.count)|Yes|supports FP32|
|[torch.nn.utils.rnn.PackedSequence.data](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.data)|Yes|-|
|[torch.nn.utils.rnn.PackedSequence.index](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.index)|Yes|supports FP32|
|[torch.nn.utils.rnn.PackedSequence.is_cuda](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.is_cuda)|No|-|
|[torch.nn.utils.rnn.PackedSequence.is_pinned](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.is_pinned)|Yes|-|
|[torch.nn.utils.rnn.PackedSequence.sorted_indices](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.sorted_indices)|Yes|-|
|[torch.nn.utils.rnn.PackedSequence.to](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.to)|Yes|supports FP32, INT64|
|[torch.nn.utils.rnn.PackedSequence.unsorted_indices](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.unsorted_indices)|Yes|-|
|[torch.nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pack_padded_sequence.html)|No|-|
|[torch.nn.utils.rnn.pad_packed_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pad_packed_sequence.html)|No|-|
|[torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pad_sequence.html)|Yes|supports FP16, FP32|
|[torch.nn.utils.rnn.pack_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pack_sequence.html)|No|-|
|[torch.nn.utils.rnn.unpack_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.unpack_sequence.html)|No|-|
|[torch.nn.utils.rnn.unpad_sequence](https://pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.unpad_sequence.html)|No|-|
|[torch.nn.modules.flatten.Flatten](https://pytorch.org/docs/2.9/generated/torch.nn.modules.flatten.Flatten.html)|Yes|supports BF16, FP16, FP32, UINT8, INT8, INT16, INT32, INT64, bool, complex64, complex128|
|[torch.nn.modules.flatten.Unflatten](https://pytorch.org/docs/2.9/generated/torch.nn.modules.flatten.Unflatten.html)|Yes|supports FP16, FP32, FP64, UINT8, INT8, INT16, INT32, INT64, bool|
|[torch.nn.modules.lazy.LazyModuleMixin](https://pytorch.org/docs/2.9/generated/torch.nn.modules.lazy.LazyModuleMixin.html)|Yes|supports FP32|
|[torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params](https://pytorch.org/docs/2.9/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params)|Yes|supports FP32|
|[torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters](https://pytorch.org/docs/2.9/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters)|Yes|supports FP32|
|[torch.nn.Unflatten.NamedShape](https://docs.pytorch.org/docs/2.7/generated/torch.nn.Unflatten.html#torch.nn.Unflatten.NamedShape)|Yes|-|
