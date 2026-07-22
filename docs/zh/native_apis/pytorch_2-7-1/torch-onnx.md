# torch.onnx

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|[torch.onnx.dynamo_export](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.dynamo_export)|是|支持fp32|
|[torch.onnx.ExportOptions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ExportOptions)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.enable_fake_mode](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.enable_fake_mode)|否|-|
|[torch.onnx.ONNXProgram](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.ONNXProgram.model_proto](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.model_proto)|否|-|
|[torch.onnx.ONNXProgram.save](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.save)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.OnnxExporterError](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxExporterError)|否|-|
|[torch.onnx.OnnxRegistry](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry)|否|-|
|[torch.onnx.OnnxRegistry.get_op_functions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.get_op_functions)|否|-|
|[torch.onnx.OnnxRegistry.is_registered_op](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.is_registered_op)|否|-|
|[torch.onnx.OnnxRegistry.opset_version](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.opset_version)|否|-|
|[torch.onnx.OnnxRegistry.register_op](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.register_op)|否|-|
|[torch.onnx.DiagnosticOptions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.DiagnosticOptions)|否|-|
|[torch.onnx.is_onnxrt_backend_supported](https://pytorch.org/docs/2.7/onnx_dynamo_onnxruntime_backend.html#torch.onnx.is_onnxrt_backend_supported)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.export)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.register_custom_op_symbolic](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.register_custom_op_symbolic)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.unregister_custom_op_symbolic](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.unregister_custom_op_symbolic)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.select_model_mode_for_export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.select_model_mode_for_export)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.is_in_onnx_export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.is_in_onnx_export)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.verification.find_mismatch](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.find_mismatch)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.JitScalarType](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.JitScalarType.dtype](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.dtype)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.JitScalarType.from_dtype](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.from_dtype)|是<br>暂不支持<term>Ascend 950DT</term>|支持bool|
|[torch.onnx.JitScalarType.from_value](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.from_value)|是<br>暂不支持<term>Ascend 950DT</term>|支持bool|
|[torch.onnx.JitScalarType.onnx_compatible](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.onnx_compatible)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.JitScalarType.onnx_type](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.onnx_type)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.JitScalarType.scalar_name](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.scalar_name)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.JitScalarType.torch_name](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.torch_name)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.verification.GraphInfo](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.GraphInfo)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.onnx.verification.VerificationOptions](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.VerificationOptions)|是<br>暂不支持<term>Ascend 950DT</term>|-|
