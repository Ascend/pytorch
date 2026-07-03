# torch.nn

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T07:58:37.228Z pushedAt=2026-06-14T09:16:34.763Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.nn.parameter.Parameter](https://pytorch.org/docs/2.10/generated/torch.nn.parameter.Parameter.html)|Yes|Supports fp32|
|[torch.nn.parameter.Buffer](https://pytorch.org/docs/2.10/generated/torch.nn.parameter.Buffer.html)|Yes|Supports fp32|
|[torch.nn.parameter.UninitializedParameter](https://pytorch.org/docs/2.10/generated/torch.nn.parameter.UninitializedParameter.html)|Yes|-|
|[torch.nn.parameter.UninitializedParameter.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.parameter.UninitializedParameter.html#torch.nn.parameter.UninitializedParameter.cls_to_become)|Yes|-|
|[torch.nn.parameter.UninitializedBuffer](https://pytorch.org/docs/2.10/generated/torch.nn.parameter.UninitializedBuffer.html)|Yes|-|
|[torch.nn.Module](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html)|Yes|Supports fp32|
|[torch.nn.Module.add_module](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.add_module)|Yes|Supports fp32|
|[torch.nn.Module.apply](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.apply)|Yes|Supports fp32|
|[torch.nn.Module.bfloat16](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.bfloat16)|Yes|-|
|[torch.nn.Module.buffers](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.buffers)|Yes|-|
|[torch.nn.Module.children](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.children)|Yes|Supports fp32|
|[torch.nn.Module.compile](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.compile)|Yes|-|
|[torch.nn.Module.cpu](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.cpu)|Yes|Supports fp32|
|[torch.nn.Module.cuda](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.cuda)|Yes|Supports fp32|
|[torch.nn.Module.double](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.double)|Yes|-|
|[torch.nn.Module.eval](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.eval)|Yes|Supports fp32, int64|
|[torch.nn.Module.extra_repr](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.extra_repr)|Yes|Supports fp32|
|[torch.nn.Module.float](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.float)|Yes|Supports fp16, fp32|
|[torch.nn.Module.forward](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.forward)|Yes|Supports fp32|
|[torch.nn.Module.get_buffer](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.get_buffer)|Yes|-|
|[torch.nn.Module.get_extra_state](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.get_extra_state)|Yes|-|
|[torch.nn.Module.get_parameter](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.get_parameter)|Yes|Supports fp32|
|[torch.nn.Module.get_submodule](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.get_submodule)|Yes|Supports fp32|
|[torch.nn.Module.half](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.half)|Yes|Supports fp16, fp32|
|[torch.nn.Module.ipu](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.ipu)|No|-|
|[torch.nn.Module.load_state_dict](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)|Yes|Supports fp32|
|[torch.nn.Module.modules](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.modules)|Yes|Supports fp32|
|[torch.nn.Module.named_buffers](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.named_buffers)|Yes|-|
|[torch.nn.Module.named_children](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.named_children)|Yes|Supports fp32|
|[torch.nn.Module.named_modules](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.named_modules)|Yes|Supports fp32|
|[torch.nn.Module.named_parameters](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.named_parameters)|Yes|-|
|[torch.nn.Module.parameters](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.parameters)|Yes|-|
|[torch.nn.Module.register_backward_hook](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_backward_hook)|Yes|Supports fp32|
|[torch.nn.Module.register_buffer](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)|Yes|Supports fp32|
|[torch.nn.Module.register_forward_hook](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)|Yes|Supports fp32|
|[torch.nn.Module.register_forward_pre_hook](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)|Yes|Supports fp32|
|[torch.nn.Module.register_full_backward_hook](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook)|Yes|Supports fp32|
|[torch.nn.Module.register_full_backward_pre_hook](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook)|Yes|Supports fp32|
|[torch.nn.Module.register_load_state_dict_post_hook](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_load_state_dict_post_hook)|Yes|Supports fp32|
|[torch.nn.Module.register_module](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_module)|Yes|Supports fp32|
|[torch.nn.Module.register_parameter](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_parameter)|Yes|-|
|[torch.nn.Module.register_state_dict_pre_hook](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_state_dict_pre_hook)|Yes|-|
|[torch.nn.Module.requires_grad_](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.requires_grad_)|Yes|-|
|[torch.nn.Module.set_extra_state](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.set_extra_state)|Yes|-|
|[torch.nn.Module.set_submodule](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.set_submodule)|Yes|-|
|[torch.nn.Module.share_memory](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.share_memory)|No|-|
|[torch.nn.Module.state_dict](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.state_dict)|Yes|Supports fp32|
|[torch.nn.Module.to](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.to)|Yes|Supports fp32|
|[torch.nn.Module.to_empty](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.to_empty)|Yes|Supports fp32|
|[torch.nn.Module.train](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.train)|Yes|Supports fp32|
|[torch.nn.Module.type](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.type)|Yes|Supports fp16, fp32, int64|
|[torch.nn.Module.xpu](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.xpu)|Yes|The NPU form name is torch.nn.Module.npu|
|[torch.nn.Module.npu](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.npu)|No|-|
|[torch.nn.Module.zero_grad](https://pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.zero_grad)|Yes|Supports fp32|
|[torch.nn.Sequential](https://pytorch.org/docs/2.10/generated/torch.nn.Sequential.html)|Yes|Supports fp32|
|[torch.nn.Sequential.append](https://pytorch.org/docs/2.10/generated/torch.nn.Sequential.html#torch.nn.Sequential.append)|Yes|Supports fp32|
|[torch.nn.ModuleList](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleList.html)|Yes|Supports fp32|
|[torch.nn.ModuleList.append](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.append)|Yes|Supports fp32|
|[torch.nn.ModuleList.extend](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.extend)|Yes|Supports fp32|
|[torch.nn.ModuleList.insert](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleList.html#torch.nn.ModuleList.insert)|Yes|Supports fp32|
|[torch.nn.ModuleDict](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleDict.html)|Yes|Supports fp32|
|[torch.nn.ModuleDict.clear](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.clear)|Yes|Supports fp32|
|[torch.nn.ModuleDict.items](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.items)|Yes|Supports fp32|
|[torch.nn.ModuleDict.keys](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.keys)|Yes|Supports fp32|
|[torch.nn.ModuleDict.pop](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.pop)|Yes|Supports fp32|
|[torch.nn.ModuleDict.update](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.update)|Yes|Supports fp32|
|[torch.nn.ModuleDict.values](https://pytorch.org/docs/2.10/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict.values)|Yes|Supports fp32|
|[torch.nn.ParameterList](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterList.html)|Yes|Supports fp32|
|[torch.nn.ParameterList.append](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterList.html#torch.nn.ParameterList.append)|Yes|Supports fp32|
|[torch.nn.ParameterList.extend](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterList.html#torch.nn.ParameterList.extend)|Yes|Supports fp32|
|[torch.nn.ParameterDict](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html)|Yes|Supports fp32|
|[torch.nn.ParameterDict.clear](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.clear)|Yes|Supports fp32|
|[torch.nn.ParameterDict.copy](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.copy)|Yes|Supports fp32|
|[torch.nn.ParameterDict.fromkeys](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.fromkeys)|Yes|Supports fp32|
|[torch.nn.ParameterDict.get](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.get)|Yes|Supports fp32|
|[torch.nn.ParameterDict.items](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.items)|Yes|Supports fp32|
|[torch.nn.ParameterDict.keys](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.keys)|Yes|Supports fp32|
|[torch.nn.ParameterDict.pop](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.pop)|Yes|Supports fp32|
|[torch.nn.ParameterDict.popitem](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.popitem)|Yes|Supports fp32|
|[torch.nn.ParameterDict.setdefault](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.setdefault)|Yes|Supports fp32|
|[torch.nn.ParameterDict.update](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.update)|Yes|Supports fp32|
|[torch.nn.ParameterDict.values](https://pytorch.org/docs/2.10/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict.values)|Yes|Supports fp32|
|[torch.nn.modules.module.register_module_forward_pre_hook](https://pytorch.org/docs/2.10/generated/torch.nn.modules.module.register_module_forward_pre_hook.html)|Yes|Supports fp32|
|[torch.nn.modules.module.register_module_forward_hook](https://pytorch.org/docs/2.10/generated/torch.nn.modules.module.register_module_forward_hook.html)|Yes|Supports fp32|
|[torch.nn.modules.module.register_module_backward_hook](https://pytorch.org/docs/2.10/generated/torch.nn.modules.module.register_module_backward_hook.html)|Yes|Supports fp32|
|[torch.nn.modules.module.register_module_full_backward_pre_hook](https://pytorch.org/docs/2.10/generated/torch.nn.modules.module.register_module_full_backward_pre_hook.html)|No|-|
|[torch.nn.modules.module.register_module_full_backward_hook](https://pytorch.org/docs/2.10/generated/torch.nn.modules.module.register_module_full_backward_hook.html)|Yes|Supports fp32|
|[torch.nn.modules.module.register_module_buffer_registration_hook](https://pytorch.org/docs/2.10/generated/torch.nn.modules.module.register_module_buffer_registration_hook.html)|No|-|
|[torch.nn.modules.module.register_module_module_registration_hook](https://pytorch.org/docs/2.10/generated/torch.nn.modules.module.register_module_module_registration_hook.html)|No|-|
|[torch.nn.modules.module.register_module_parameter_registration_hook](https://pytorch.org/docs/2.10/generated/torch.nn.modules.module.register_module_parameter_registration_hook.html)|No|-|
|[torch.nn.Conv1d](https://pytorch.org/docs/2.10/generated/torch.nn.Conv1d.html)|Yes|Supports fp16, fp32|
|[torch.nn.Conv2d](https://pytorch.org/docs/2.10/generated/torch.nn.Conv2d.html)|Yes|Supports bf16, fp16, fp32<br><term>Atlas A2 Training Series</term>, in default scenarios, if compilation is triggered frequently, it is recommended to manually set torch.npu.config.allow_internal_format to False to control input parameters to not enable internal format, avoiding online compilation|
|[torch.nn.Conv3d](https://pytorch.org/docs/2.10/generated/torch.nn.Conv3d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.ConvTranspose1d](https://pytorch.org/docs/2.10/generated/torch.nn.ConvTranspose1d.html)|Yes|Supports fp32|
|[torch.nn.ConvTranspose2d](https://pytorch.org/docs/2.10/generated/torch.nn.ConvTranspose2d.html)|Yes|Supports fp16, fp32<br><term>Atlas Training Series</term>/<term>Atlas A2 Training Series</term>, requires manually setting torch.npu.config.allow_internal_format to False to support 3-dimensional input|
|[torch.nn.ConvTranspose3d](https://pytorch.org/docs/2.10/generated/torch.nn.ConvTranspose3d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.LazyConv1d](https://pytorch.org/docs/2.10/generated/torch.nn.LazyConv1d.html)|Yes|Supports fp16, fp32|
|[torch.nn.LazyConv1d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyConv1d.html#torch.nn.LazyConv1d.cls_to_become)|Yes|-|
|[torch.nn.LazyConv2d](https://pytorch.org/docs/2.10/generated/torch.nn.LazyConv2d.html)|Yes|Supports fp16, fp32|
|[torch.nn.LazyConv2d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d.cls_to_become)|Yes|-|
|[torch.nn.LazyConv3d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyConv3d.html#torch.nn.LazyConv3d.cls_to_become)|Yes|-|
|[torch.nn.LazyConvTranspose1d](https://pytorch.org/docs/2.10/generated/torch.nn.LazyConvTranspose1d.html)|Yes|Supports fp16|
|[torch.nn.LazyConvTranspose1d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyConvTranspose1d.html#torch.nn.LazyConvTranspose1d.cls_to_become)|Yes|-|
|[torch.nn.LazyConvTranspose2d](https://pytorch.org/docs/2.10/generated/torch.nn.LazyConvTranspose2d.html)|Yes|Supports fp16, fp32|
|[torch.nn.LazyConvTranspose2d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyConvTranspose2d.html#torch.nn.LazyConvTranspose2d.cls_to_become)|Yes|-|
|[torch.nn.LazyConvTranspose3d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyConvTranspose3d.html#torch.nn.LazyConvTranspose3d.cls_to_become)|Yes|-|
|[torch.nn.Unfold](https://pytorch.org/docs/2.10/generated/torch.nn.Unfold.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.Fold](https://pytorch.org/docs/2.10/generated/torch.nn.Fold.html)|Yes|Supports fp16|
|[torch.nn.MaxPool1d](https://pytorch.org/docs/2.10/generated/torch.nn.MaxPool1d.html)|No|-|
|[torch.nn.MaxPool2d](https://pytorch.org/docs/2.10/generated/torch.nn.MaxPool2d.html)|Yes|Supports bf16, fp16, fp32<br>By setting torch_npu.npu.use_compatible_impl(True), ensures memory consistency alignment with the community counterpart interface|
|[torch.nn.MaxPool3d](https://pytorch.org/docs/2.10/generated/torch.nn.MaxPool3d.html)|No|-|
|[torch.nn.MaxUnpool1d](https://pytorch.org/docs/2.10/generated/torch.nn.MaxUnpool1d.html)|Yes|Supports fp16, fp32|
|[torch.nn.MaxUnpool2d](https://pytorch.org/docs/2.10/generated/torch.nn.MaxUnpool2d.html)|Yes|Supports fp16, fp32|
|[torch.nn.MaxUnpool3d](https://pytorch.org/docs/2.10/generated/torch.nn.MaxUnpool3d.html)|No|-|
|[torch.nn.AvgPool1d](https://pytorch.org/docs/2.10/generated/torch.nn.AvgPool1d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.AvgPool2d](https://pytorch.org/docs/2.10/generated/torch.nn.AvgPool2d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.AvgPool3d](https://pytorch.org/docs/2.10/generated/torch.nn.AvgPool3d.html)|No|-|
|[torch.nn.LPPool1d](https://pytorch.org/docs/2.10/generated/torch.nn.LPPool1d.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.nn.LPPool2d](https://pytorch.org/docs/2.10/generated/torch.nn.LPPool2d.html)|Yes|Supports fp16, fp32, int16, int32, int64, bool|
|[torch.nn.AdaptiveMaxPool1d](https://pytorch.org/docs/2.10/generated/torch.nn.AdaptiveMaxPool1d.html)|No|-|
|[torch.nn.AdaptiveMaxPool2d](https://pytorch.org/docs/2.10/generated/torch.nn.AdaptiveMaxPool2d.html)|No|-|
|[torch.nn.AdaptiveMaxPool3d](https://pytorch.org/docs/2.10/generated/torch.nn.AdaptiveMaxPool3d.html)|Yes|Supports fp32, fp64|
|[torch.nn.AdaptiveAvgPool1d](https://pytorch.org/docs/2.10/generated/torch.nn.AdaptiveAvgPool1d.html)|Yes|Supports fp16, fp32|
|[torch.nn.AdaptiveAvgPool2d](https://pytorch.org/docs/2.10/generated/torch.nn.AdaptiveAvgPool2d.html)|Yes|Supports fp16, fp32|
|[torch.nn.AdaptiveAvgPool3d](https://pytorch.org/docs/2.10/generated/torch.nn.AdaptiveAvgPool3d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.ReflectionPad1d](https://pytorch.org/docs/2.10/generated/torch.nn.ReflectionPad1d.html)|Yes|Supports fp16, fp32|
|[torch.nn.ReflectionPad2d](https://pytorch.org/docs/2.10/generated/torch.nn.ReflectionPad2d.html)|Yes|Supports fp16, fp32|
|[torch.nn.ReflectionPad3d](https://pytorch.org/docs/2.10/generated/torch.nn.ReflectionPad3d.html)|No|-|
|[torch.nn.ReplicationPad1d](https://pytorch.org/docs/2.10/generated/torch.nn.ReplicationPad1d.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[torch.nn.ReplicationPad2d](https://pytorch.org/docs/2.10/generated/torch.nn.ReplicationPad2d.html)|Yes|Supports fp16, fp32, complex64, complex128|
|[torch.nn.ReplicationPad3d](https://pytorch.org/docs/2.10/generated/torch.nn.ReplicationPad3d.html)|No|-|
|[torch.nn.ZeroPad1d](https://pytorch.org/docs/2.10/generated/torch.nn.ZeroPad1d.html)|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128<br>Supports 2-3 dimensions|
|[torch.nn.ZeroPad2d](https://pytorch.org/docs/2.10/generated/torch.nn.ZeroPad2d.html)|Yes|May Fallback to CPU execution|
|[torch.nn.ZeroPad3d](https://pytorch.org/docs/2.10/generated/torch.nn.ZeroPad3d.html)|Yes|Supports bf16, fp16, fp32, fp64, complex64, complex128<br>Supports 5-6 dimensions|
|[torch.nn.ConstantPad1d](https://pytorch.org/docs/2.10/generated/torch.nn.ConstantPad1d.html)|Yes|Supports int8, bool<br>Performance degradation may occur when input x has six or more dimensions|
|[torch.nn.ConstantPad2d](https://pytorch.org/docs/2.10/generated/torch.nn.ConstantPad2d.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Performance degradation may occur when input x has six or more dimensions|
|[torch.nn.ConstantPad3d](https://pytorch.org/docs/2.10/generated/torch.nn.ConstantPad3d.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128<br>Performance degradation may occur when input x has six or more dimensions|
|[torch.nn.ELU](https://pytorch.org/docs/2.10/generated/torch.nn.ELU.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[torch.nn.Hardshrink](https://pytorch.org/docs/2.10/generated/torch.nn.Hardshrink.html)|Yes|Supports fp16, fp32<br>May Fallback to CPU execution|
|[torch.nn.Hardsigmoid](https://pytorch.org/docs/2.10/generated/torch.nn.Hardsigmoid.html)|Yes|Supports fp16, fp32, int32<br>May Fallback to CPU execution|
|[torch.nn.Hardtanh](https://pytorch.org/docs/2.10/generated/torch.nn.Hardtanh.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.nn.Hardswish](https://pytorch.org/docs/2.10/generated/torch.nn.Hardswish.html)|Yes|Supports fp16, fp32|
|[torch.nn.LeakyReLU](https://pytorch.org/docs/2.10/generated/torch.nn.LeakyReLU.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[torch.nn.LogSigmoid](https://pytorch.org/docs/2.10/generated/torch.nn.LogSigmoid.html)|Yes|Supports fp16, fp32|
|[torch.nn.MultiheadAttention](https://pytorch.org/docs/2.10/generated/torch.nn.MultiheadAttention.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.MultiheadAttention.forward](https://pytorch.org/docs/2.10/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.PReLU](https://pytorch.org/docs/2.10/generated/torch.nn.PReLU.html)|Yes|Supports fp32|
|[torch.nn.ReLU](https://pytorch.org/docs/2.10/generated/torch.nn.ReLU.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64|
|[torch.nn.ReLU6](https://pytorch.org/docs/2.10/generated/torch.nn.ReLU6.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|[torch.nn.RReLU](https://pytorch.org/docs/2.10/generated/torch.nn.RReLU.html)|No|-|
|[torch.nn.SELU](https://pytorch.org/docs/2.10/generated/torch.nn.SELU.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.nn.CELU](https://pytorch.org/docs/2.10/generated/torch.nn.CELU.html)|Yes|Supports fp16, fp32|
|[torch.nn.GELU](https://pytorch.org/docs/2.10/generated/torch.nn.GELU.html)|Yes|Supports bf16, fp16, fp32<br>The approximate parameter only supports being set to tanh|
|[torch.nn.Sigmoid](https://pytorch.org/docs/2.10/generated/torch.nn.Sigmoid.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nn.SiLU](https://pytorch.org/docs/2.10/generated/torch.nn.SiLU.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.Mish](https://pytorch.org/docs/2.10/generated/torch.nn.Mish.html)|Yes|Supports fp16, fp32|
|[torch.nn.Softplus](https://pytorch.org/docs/2.10/generated/torch.nn.Softplus.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.Softshrink](https://pytorch.org/docs/2.10/generated/torch.nn.Softshrink.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.Softsign](https://pytorch.org/docs/2.10/generated/torch.nn.Softsign.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.nn.Tanh](https://pytorch.org/docs/2.10/generated/torch.nn.Tanh.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.nn.Tanhshrink](https://pytorch.org/docs/2.10/generated/torch.nn.Tanhshrink.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64<br>May Fallback to CPU execution|
|[torch.nn.Threshold](https://pytorch.org/docs/2.10/generated/torch.nn.Threshold.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.nn.GLU](https://pytorch.org/docs/2.10/generated/torch.nn.GLU.html)|Yes|Supports fp16, fp32|
|[torch.nn.Softmin](https://pytorch.org/docs/2.10/generated/torch.nn.Softmin.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.Softmax](https://pytorch.org/docs/2.10/generated/torch.nn.Softmax.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[torch.nn.Softmax2d](https://pytorch.org/docs/2.10/generated/torch.nn.Softmax2d.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.LogSoftmax](https://pytorch.org/docs/2.10/generated/torch.nn.LogSoftmax.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.AdaptiveLogSoftmaxWithLoss](https://pytorch.org/docs/2.10/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html)|No|-|
|[torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob](https://pytorch.org/docs/2.10/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob)|No|-|
|[torch.nn.AdaptiveLogSoftmaxWithLoss.predict](https://pytorch.org/docs/2.10/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html#torch.nn.AdaptiveLogSoftmaxWithLoss.predict)|No|-|
|[torch.nn.BatchNorm1d](https://pytorch.org/docs/2.10/generated/torch.nn.BatchNorm1d.html)|Yes|Supports fp16, fp32|
|[torch.nn.BatchNorm2d](https://pytorch.org/docs/2.10/generated/torch.nn.BatchNorm2d.html)|Yes|Supports fp16, fp32|
|[torch.nn.BatchNorm3d](https://pytorch.org/docs/2.10/generated/torch.nn.BatchNorm3d.html)|Yes|Supports fp16, fp32|
|[torch.nn.LazyBatchNorm1d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyBatchNorm1d.html#torch.nn.LazyBatchNorm1d.cls_to_become)|Yes|-|
|[torch.nn.LazyBatchNorm2d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyBatchNorm2d.html#torch.nn.LazyBatchNorm2d.cls_to_become)|Yes|-|
|[torch.nn.LazyBatchNorm3d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyBatchNorm3d.html#torch.nn.LazyBatchNorm3d.cls_to_become)|Yes|-|
|[torch.nn.GroupNorm](https://pytorch.org/docs/2.10/generated/torch.nn.GroupNorm.html)|Yes|Supports fp32<br>The eps parameter must be greater than 0<br>Does not support scenarios with jit_compile=True<br>This API only supports input with 2 or more dimensions. During backward propagation of this API, scenarios where the input dimension is not 4, or the input num_groups is not divisible by 32, or the C-axis dimension is not divisible by (10 * num_groups) are not supported|
|[torch.nn.SyncBatchNorm](https://pytorch.org/docs/2.10/generated/torch.nn.SyncBatchNorm.html)|Yes|Supports fp16, fp32|
|[torch.nn.SyncBatchNorm.convert_sync_batchnorm](https://pytorch.org/docs/2.10/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm)|Yes|-|
|[torch.nn.LazyInstanceNorm1d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyInstanceNorm1d.html#torch.nn.LazyInstanceNorm1d.cls_to_become)|Yes|-|
|[torch.nn.LazyInstanceNorm2d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyInstanceNorm2d.html#torch.nn.LazyInstanceNorm2d.cls_to_become)|Yes|-|
|[torch.nn.LazyInstanceNorm3d.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyInstanceNorm3d.html#torch.nn.LazyInstanceNorm3d.cls_to_become)|Yes|-|
|[torch.nn.LayerNorm](https://pytorch.org/docs/2.10/generated/torch.nn.LayerNorm.html)|Yes|Supports bf16, fp16, fp32<br>By setting torch_npu.npu.use_compatible_impl(True), this interface switches from the aclnnLayerNorm operator to the aclnnFastLayerNorm operator, ensuring memory consistency alignment with the community counterpart interface.|
|[torch.nn.RNNBase](https://pytorch.org/docs/2.10/generated/torch.nn.RNNBase.html)|No|-|
|[torch.nn.RNNBase.flatten_parameters](https://pytorch.org/docs/2.10/generated/torch.nn.RNNBase.html#torch.nn.RNNBase.flatten_parameters)|No|-|
|[torch.nn.RNN](https://pytorch.org/docs/2.10/generated/torch.nn.RNN.html)|No|-|
|[torch.nn.LSTM](https://pytorch.org/docs/2.10/generated/torch.nn.LSTM.html)|Yes|Supports fp32<br>Does not support the proj_size parameter<br>Does not support the dropout parameter<br>The input parameter does not support 2 dimensions|
|[torch.nn.GRU](https://pytorch.org/docs/2.10/generated/torch.nn.GRU.html)|No|-|
|[torch.nn.RNNCell](https://pytorch.org/docs/2.10/generated/torch.nn.RNNCell.html)|No|-|
|[torch.nn.LSTMCell](https://pytorch.org/docs/2.10/generated/torch.nn.LSTMCell.html)|Yes|The interface currently does not support jit_compile=False. If you need to use it in this mode, please add "DynamicGRUV2" to the "NPU_FUZZY_COMPILE_BLACKLIST" option. For specific operations, refer to [Example of Adding a Binary Blocklist](../example_of_adding_a_binary_blocklist.md)|
|[torch.nn.GRUCell](https://pytorch.org/docs/2.10/generated/torch.nn.GRUCell.html)|Yes|Supports fp16, fp32|
|[torch.nn.Transformer](https://pytorch.org/docs/2.10/generated/torch.nn.Transformer.html)|Yes|Supports fp16, fp32|
|[torch.nn.Transformer.forward](https://pytorch.org/docs/2.10/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward)|No|-|
|[torch.nn.TransformerEncoder](https://pytorch.org/docs/2.10/generated/torch.nn.TransformerEncoder.html)|No|-|
|[torch.nn.TransformerEncoder.forward](https://pytorch.org/docs/2.10/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder.forward)|Yes|Supports fp32|
|[torch.nn.TransformerDecoder](https://pytorch.org/docs/2.10/generated/torch.nn.TransformerDecoder.html)|No|-|
|[torch.nn.TransformerDecoder.forward](https://pytorch.org/docs/2.10/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder.forward)|No|-|
|[torch.nn.TransformerEncoderLayer.forward](https://pytorch.org/docs/2.10/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer.forward)|No|-|
|[torch.nn.TransformerDecoderLayer.forward](https://pytorch.org/docs/2.10/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer.forward)|No|-|
|[torch.nn.Identity](https://pytorch.org/docs/2.10/generated/torch.nn.Identity.html)|Yes|Supports fp32|
|[torch.nn.Linear](https://pytorch.org/docs/2.10/generated/torch.nn.Linear.html)|Yes|Supports fp16, fp32|
|[torch.nn.Bilinear](https://pytorch.org/docs/2.10/generated/torch.nn.Bilinear.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.LazyLinear](https://pytorch.org/docs/2.10/generated/torch.nn.LazyLinear.html)|Yes|Supports fp16, fp32|
|[torch.nn.LazyLinear.cls_to_become](https://pytorch.org/docs/2.10/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear.cls_to_become)|No|-|
|[torch.nn.Dropout](https://pytorch.org/docs/2.10/generated/torch.nn.Dropout.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.Dropout2d](https://pytorch.org/docs/2.10/generated/torch.nn.Dropout2d.html)|Yes|Supports fp16, fp32, int64, bool|
|[torch.nn.AlphaDropout](https://pytorch.org/docs/2.10/generated/torch.nn.AlphaDropout.html)|Yes|Supports fp16, fp32|
|[torch.nn.FeatureAlphaDropout](https://pytorch.org/docs/2.10/generated/torch.nn.FeatureAlphaDropout.html)|Yes|Supports fp16, fp32|
|[torch.nn.Embedding](https://pytorch.org/docs/2.10/generated/torch.nn.Embedding.html)|Yes|Supports int32, int64<br>The attribute max_norm does not support nan, only supports non-negative values|
|[torch.nn.Embedding.from_pretrained](https://pytorch.org/docs/2.10/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained)|Yes|Supports fp64|
|[torch.nn.EmbeddingBag](https://pytorch.org/docs/2.10/generated/torch.nn.EmbeddingBag.html)|Yes|Supports int32, int64<br>Only supports max_norm greater than or equal to 0|
|[torch.nn.EmbeddingBag.forward](https://pytorch.org/docs/2.10/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward)|Yes|Supports int64|
|[torch.nn.EmbeddingBag.from_pretrained](https://pytorch.org/docs/2.10/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.from_pretrained)|Yes|Supports int64|
|[torch.nn.L1Loss](https://pytorch.org/docs/2.10/generated/torch.nn.L1Loss.html)|Yes|Supports fp16, fp32, int64|
|[torch.nn.MSELoss](https://pytorch.org/docs/2.10/generated/torch.nn.MSELoss.html)|Yes|Supports fp16, fp32|
|[torch.nn.CrossEntropyLoss](https://pytorch.org/docs/2.10/generated/torch.nn.CrossEntropyLoss.html)|Yes|Supports fp16, fp32|
|[torch.nn.CTCLoss](https://pytorch.org/docs/2.10/generated/torch.nn.CTCLoss.html)|Yes|Supports fp32, fp64<br>Does not support log_probs 2D input|
|[torch.nn.NLLLoss](https://pytorch.org/docs/2.10/generated/torch.nn.NLLLoss.html)|Yes|Supports fp16, fp32<br>Each element value in target should be greater than or equal to 0 and less than the number of classes of input|
|[torch.nn.PoissonNLLLoss](https://pytorch.org/docs/2.10/generated/torch.nn.PoissonNLLLoss.html)|Yes|Supports bf16, fp16, fp32, int64|
|[torch.nn.GaussianNLLLoss](https://pytorch.org/docs/2.10/generated/torch.nn.GaussianNLLLoss.html)|Yes|Supports bf16, fp16, fp32, int16, int32, int64|
|[torch.nn.KLDivLoss](https://pytorch.org/docs/2.10/generated/torch.nn.KLDivLoss.html)|Yes|Supports bf16, fp16, fp32<br>Currently the log_target parameter only supports False|
|[torch.nn.BCELoss](https://pytorch.org/docs/2.10/generated/torch.nn.BCELoss.html)|Yes|Supports fp16, fp32|
|[torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/2.10/generated/torch.nn.BCEWithLogitsLoss.html)|Yes|Supports bf16, fp16, fp32<br>The input parameter target does not support backward computation|
|[torch.nn.MarginRankingLoss](https://pytorch.org/docs/2.10/generated/torch.nn.MarginRankingLoss.html)|Yes|Supports bf16, fp16, fp32, int8, int32, int64|
|[torch.nn.HingeEmbeddingLoss](https://pytorch.org/docs/2.10/generated/torch.nn.HingeEmbeddingLoss.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64|
|[torch.nn.MultiLabelMarginLoss](https://pytorch.org/docs/2.10/generated/torch.nn.MultiLabelMarginLoss.html)|No|-|
|[torch.nn.HuberLoss](https://pytorch.org/docs/2.10/generated/torch.nn.HuberLoss.html)|Yes|input supports fp32, fp64<br>target supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>May fallback to CPU execution|
|[torch.nn.SmoothL1Loss](https://pytorch.org/docs/2.10/generated/torch.nn.SmoothL1Loss.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/2.10/generated/torch.nn.MultiLabelSoftMarginLoss.html)|Yes|Supports fp16, fp32|
|[torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/2.10/generated/torch.nn.CosineEmbeddingLoss.html)|No|-|
|[torch.nn.MultiMarginLoss](https://pytorch.org/docs/2.10/generated/torch.nn.MultiMarginLoss.html)|Yes|input supports fp32, fp64<br>target supports int64<br>May fallback to CPU execution|
|[torch.nn.TripletMarginLoss](https://pytorch.org/docs/2.10/generated/torch.nn.TripletMarginLoss.html)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64<br>May fallback to CPU execution|
|[torch.nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/2.10/generated/torch.nn.TripletMarginWithDistanceLoss.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.PixelShuffle](https://pytorch.org/docs/2.10/generated/torch.nn.PixelShuffle.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.nn.PixelUnshuffle](https://pytorch.org/docs/2.10/generated/torch.nn.PixelUnshuffle.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.nn.Upsample](https://pytorch.org/docs/2.10/generated/torch.nn.Upsample.html)|Yes|Supports bf16, fp16, fp32, fp64|
|[torch.nn.UpsamplingNearest2d](https://pytorch.org/docs/2.10/generated/torch.nn.UpsamplingNearest2d.html)|Yes|Supports fp16, fp32, uint8<br>May fallback to CPU execution|
|[torch.nn.ChannelShuffle](https://pytorch.org/docs/2.10/generated/torch.nn.ChannelShuffle.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nn.DataParallel](https://pytorch.org/docs/2.10/generated/torch.nn.DataParallel.html)|No|-|
|[torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html)|Yes|-|
|[torch.nn.parallel.DistributedDataParallel.join](https://pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join)|Yes|-|
|[torch.nn.parallel.DistributedDataParallel.join_hook](https://pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join_hook)|Yes|-|
|[torch.nn.parallel.DistributedDataParallel.no_sync](https://pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync)|Yes|-|
|[torch.nn.parallel.DistributedDataParallel.register_comm_hook](https://pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook)|Yes|-|
|[torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/2.10/generated/torch.nn.utils.clip_grad_norm_.html)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.nn.utils.clip_grad_norm](https://pytorch.org/docs/2.10/generated/torch.nn.utils.clip_grad_norm.html)|No|-|
|[torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/2.10/generated/torch.nn.utils.clip_grad_value_.html)|Yes|Supports bf16, fp16, fp32|
|[torch.nn.utils.vector_to_parameters](https://pytorch.org/docs/2.10/generated/torch.nn.utils.vector_to_parameters.html)|Yes|Supports bf16, fp16, fp32, fp64, complex64|
|[torch.nn.utils.weight_norm](https://pytorch.org/docs/2.10/generated/torch.nn.utils.weight_norm.html)|Yes|-|
|[torch.nn.utils.spectral_norm](https://pytorch.org/docs/2.10/generated/torch.nn.utils.spectral_norm.html)|Yes|-|
|[torch.nn.utils.remove_spectral_norm](https://pytorch.org/docs/2.10/generated/torch.nn.utils.remove_spectral_norm.html)|Yes|-|
|[torch.nn.utils.skip_init](https://pytorch.org/docs/2.10/generated/torch.nn.utils.skip_init.html)|Yes|-|
|[torch.nn.utils.prune.BasePruningMethod](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.BasePruningMethod.html)|Yes|-|
|[torch.nn.utils.prune.BasePruningMethod.apply](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.apply)|Yes|-|
|[torch.nn.utils.prune.BasePruningMethod.apply_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.apply_mask)|Yes|Supports fp32|
|[torch.nn.utils.prune.BasePruningMethod.compute_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.compute_mask)|Yes|-|
|[torch.nn.utils.prune.BasePruningMethod.prune](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.prune)|Yes|Supports fp32|
|[torch.nn.utils.prune.BasePruningMethod.remove](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod.remove)|Yes|Supports fp32|
|[torch.nn.utils.prune.PruningContainer](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.PruningContainer.html)|Yes|-|
|[torch.nn.utils.prune.PruningContainer.add_pruning_method](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.add_pruning_method)|Yes|-|
|[torch.nn.utils.prune.PruningContainer.apply](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.apply)|Yes|-|
|[torch.nn.utils.prune.PruningContainer.apply_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.apply_mask)|Yes|-|
|[torch.nn.utils.prune.PruningContainer.compute_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.compute_mask)|Yes|Supports fp32|
|[torch.nn.utils.prune.PruningContainer.prune](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.prune)|Yes|Supports fp32|
|[torch.nn.utils.prune.PruningContainer.remove](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer.remove)|Yes|Supports fp32|
|[torch.nn.utils.prune.Identity](https://pytorch.org/docs/2.10/utils.html#torch.nn.utils.prune.Identity)|Yes|Supports fp32|
|[torch.nn.utils.prune.Identity.apply](https://pytorch.org/docs/2.10/nn.html#torch.nn.utils.prune.Identity.apply)|Yes|Supports fp32|
|[torch.nn.utils.prune.Identity.apply_mask](https://pytorch.org/docs/2.10/nn.html#torch.nn.utils.prune.Identity.apply_mask)|Yes|Supports fp32|
|[torch.nn.utils.prune.Identity.prune](https://pytorch.org/docs/2.10/nn.html#torch.nn.utils.prune.Identity.prune)|Yes|Supports fp32|
|[torch.nn.utils.prune.Identity.remove](https://pytorch.org/docs/2.10/nn.html#torch.nn.utils.prune.Identity.remove)|Yes|Supports fp32|
|[torch.nn.utils.prune.RandomUnstructured](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomUnstructured.html)|Yes|Supports fp32|
|[torch.nn.utils.prune.RandomUnstructured.apply](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.apply)|Yes|Supports fp32|
|[torch.nn.utils.prune.RandomUnstructured.apply_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.apply_mask)|Yes|Supports fp32|
|[torch.nn.utils.prune.RandomUnstructured.prune](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.prune)|Yes|Supports fp32|
|[torch.nn.utils.prune.RandomUnstructured.remove](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured.remove)|Yes|-|
|[torch.nn.utils.prune.L1Unstructured](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.L1Unstructured.html)|Yes|Supports fp32|
|[torch.nn.utils.prune.L1Unstructured.apply](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.apply)|Yes|Supports fp32|
|[torch.nn.utils.prune.L1Unstructured.apply_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.apply_mask)|Yes|Supports fp32|
|[torch.nn.utils.prune.L1Unstructured.prune](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.prune)|Yes|Supports fp32|
|[torch.nn.utils.prune.L1Unstructured.remove](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured.remove)|Yes|Supports fp32|
|[torch.nn.utils.prune.RandomStructured](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomStructured.html)|Yes|Supports fp32|
|[torch.nn.utils.prune.RandomStructured.apply](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.apply)|Yes|Supports fp32|
|[torch.nn.utils.prune.RandomStructured.apply_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.apply_mask)|Yes|Supports fp32|
|[torch.nn.utils.prune.RandomStructured.compute_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.compute_mask)|Yes|Supports fp32|
|[torch.nn.utils.prune.RandomStructured.prune](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.prune)|Yes|-|
|[torch.nn.utils.prune.RandomStructured.remove](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured.remove)|Yes|-|
|[torch.nn.utils.prune.LnStructured](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.LnStructured.html)|Yes|Supports fp32|
|[torch.nn.utils.prune.LnStructured.apply](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.apply)|Yes|Supports fp32|
|[torch.nn.utils.prune.LnStructured.apply_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.apply_mask)|Yes|Supports fp32|
|[torch.nn.utils.prune.LnStructured.compute_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.compute_mask)|Yes|Supports fp32|
|[torch.nn.utils.prune.LnStructured.prune](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.prune)|Yes|Supports fp32|
|[torch.nn.utils.prune.LnStructured.remove](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured.remove)|Yes|Supports fp32|
|[torch.nn.utils.prune.CustomFromMask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.CustomFromMask.html)|Yes|Supports int64|
|[torch.nn.utils.prune.CustomFromMask.apply](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.apply)|Yes|Supports int64|
|[torch.nn.utils.prune.CustomFromMask.apply_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.apply_mask)|Yes|-|
|[torch.nn.utils.prune.CustomFromMask.prune](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.prune)|Yes|-|
|[torch.nn.utils.prune.CustomFromMask.remove](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask.remove)|Yes|-|
|[torch.nn.utils.prune.random_unstructured](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.random_unstructured.html)|Yes|-|
|[torch.nn.utils.prune.l1_unstructured](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.l1_unstructured.html)|Yes|-|
|[torch.nn.utils.prune.random_structured](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.random_structured.html)|Yes|-|
|[torch.nn.utils.prune.ln_structured](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.ln_structured.html)|Yes|-|
|[torch.nn.utils.prune.global_unstructured](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.global_unstructured.html)|Yes|-|
|[torch.nn.utils.prune.identity](https://pytorch.org/docs/2.10/nn.html#torch.nn.utils.prune.identity)|Yes|-|
|[torch.nn.utils.prune.custom_from_mask](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.custom_from_mask.html)|Yes|Supports int64|
|[torch.nn.utils.prune.remove](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.remove.html)|Yes|-|
|[torch.nn.utils.prune.is_pruned](https://pytorch.org/docs/2.10/generated/torch.nn.utils.prune.is_pruned.html)|Yes|-|
|[torch.nn.utils.parametrizations.orthogonal](https://pytorch.org/docs/2.10/generated/torch.nn.utils.parametrizations.orthogonal.html)|Yes|-|
|[torch.nn.utils.parametrizations.spectral_norm](https://pytorch.org/docs/2.10/generated/torch.nn.utils.parametrizations.spectral_norm.html)|Yes|-|
|[torch.nn.utils.parametrize.register_parametrization](https://pytorch.org/docs/2.10/generated/torch.nn.utils.parametrize.register_parametrization.html)|Yes|-|
|[torch.nn.utils.parametrize.remove_parametrizations](https://pytorch.org/docs/2.10/generated/torch.nn.utils.parametrize.remove_parametrizations.html)|Yes|-|
|[torch.nn.utils.parametrize.cached](https://pytorch.org/docs/2.10/generated/torch.nn.utils.parametrize.cached.html)|Yes|-|
|[torch.nn.utils.parametrize.is_parametrized](https://pytorch.org/docs/2.10/generated/torch.nn.utils.parametrize.is_parametrized.html)|Yes|-|
|[torch.nn.utils.parametrize.ParametrizationList](https://pytorch.org/docs/2.10/generated/torch.nn.utils.parametrize.ParametrizationList.html)|Yes|-|
|[torch.nn.utils.parametrize.ParametrizationList.right_inverse](https://pytorch.org/docs/2.10/generated/torch.nn.utils.parametrize.ParametrizationList.html#torch.nn.utils.parametrize.ParametrizationList.right_inverse)|Yes|Supports fp32|
|[torch.nn.utils.stateless.functional_call](https://pytorch.org/docs/2.10/generated/torch.nn.utils.stateless.functional_call.html)|Yes|-|
|[torch.nn.utils.rnn.PackedSequence](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.PackedSequence.html)|Yes|Supports fp32, int64|
|[torch.nn.utils.rnn.PackedSequence.batch_sizes](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.batch_sizes)|Yes|-|
|[torch.nn.utils.rnn.PackedSequence.count](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.count)|Yes|Supports fp32|
|[torch.nn.utils.rnn.PackedSequence.data](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.data)|Yes|-|
|[torch.nn.utils.rnn.PackedSequence.index](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.index)|Yes|Supports fp32|
|[torch.nn.utils.rnn.PackedSequence.is_cuda](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.is_cuda)|No|-|
|[torch.nn.utils.rnn.PackedSequence.is_pinned](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.is_pinned)|Yes|-|
|[torch.nn.utils.rnn.PackedSequence.sorted_indices](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.sorted_indices)|Yes|-|
|[torch.nn.utils.rnn.PackedSequence.to](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.to)|Yes|Supports fp32, int64|
|[torch.nn.utils.rnn.PackedSequence.unsorted_indices](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence.unsorted_indices)|Yes|-|
|[torch.nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.pack_padded_sequence.html)|No|-|
|[torch.nn.utils.rnn.pad_packed_sequence](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.pad_packed_sequence.html)|No|-|
|[torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.pad_sequence.html)|Yes|Supports fp16, fp32|
|[torch.nn.utils.rnn.pack_sequence](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.pack_sequence.html)|No|-|
|[torch.nn.utils.rnn.unpack_sequence](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.unpack_sequence.html)|No|-|
|[torch.nn.utils.rnn.unpad_sequence](https://pytorch.org/docs/2.10/generated/torch.nn.utils.rnn.unpad_sequence.html)|No|-|
|[torch.nn.modules.flatten.Flatten](https://pytorch.org/docs/2.10/generated/torch.nn.modules.flatten.Flatten.html)|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.nn.modules.flatten.Unflatten](https://pytorch.org/docs/2.10/generated/torch.nn.modules.flatten.Unflatten.html)|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.nn.modules.lazy.LazyModuleMixin](https://pytorch.org/docs/2.10/generated/torch.nn.modules.lazy.LazyModuleMixin.html)|Yes|Supports fp32|
|[torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params](https://pytorch.org/docs/2.10/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params)|Yes|Supports fp32|
|[torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters](https://pytorch.org/docs/2.10/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters)|Yes|Supports fp32|
|[torch.nn.Unflatten.NamedShape](https://docs.pytorch.org/docs/2.7/generated/torch.nn.Unflatten.html#torch.nn.Unflatten.NamedShape)|Yes|-|
