# torch.onnx

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:08:44.289Z pushedAt=2026-06-15T02:04:36.543Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.onnx.dynamo_export](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.dynamo_export)|Yes|Supports fp32|
|[torch.onnx.ExportOptions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ExportOptions)|Yes|-|
|[torch.onnx.enable_fake_mode](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.enable_fake_mode)|No|-|
|[torch.onnx.ONNXProgram](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram)|Yes|-|
|[torch.onnx.ONNXProgram.model_proto](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.model_proto)|No|-|
|[torch.onnx.ONNXProgram.save](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.save)|Yes|-|
|[torch.onnx.OnnxExporterError](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxExporterError)|No|-|
|[torch.onnx.OnnxRegistry](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry)|No|-|
|[torch.onnx.OnnxRegistry.get_op_functions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.get_op_functions)|No|-|
|[torch.onnx.OnnxRegistry.is_registered_op](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.is_registered_op)|No|-|
|[torch.onnx.OnnxRegistry.opset_version](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.opset_version)|No|-|
|[torch.onnx.OnnxRegistry.register_op](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.register_op)|No|-|
|[torch.onnx.DiagnosticOptions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.DiagnosticOptions)|No|-|
|[torch.onnx.is_onnxrt_backend_supported](https://pytorch.org/docs/2.7/onnx_dynamo_onnxruntime_backend.html#torch.onnx.is_onnxrt_backend_supported)|Yes|-|
|[torch.onnx.export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.export)|Yes|-|
|[torch.onnx.register_custom_op_symbolic](https://pytorch.org/docs/2.7/onnx.html#torch.onnx.register_custom_op_symbolic)|Yes|-|
|[torch.onnx.unregister_custom_op_symbolic](https://pytorch.org/docs/2.7/onnx.html#torch.onnx.unregister_custom_op_symbolic)|Yes|-|
|[torch.onnx.select_model_mode_for_export](https://pytorch.org/docs/2.7/onnx.html#torch.onnx.select_model_mode_for_export)|Yes|-|
|[torch.onnx.is_in_onnx_export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.is_in_onnx_export)|Yes|-|
|[torch.onnx.verification.find_mismatch](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.find_mismatch)|Yes|-|
|[torch.onnx.JitScalarType](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html)|Yes|-|
|[torch.onnx.JitScalarType.dtype](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.dtype)|Yes|-|
|[torch.onnx.JitScalarType.from_dtype](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.from_dtype)|Yes|Supports bool|
|[torch.onnx.JitScalarType.from_value](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.from_value)|Yes|Supports bool|
|[torch.onnx.JitScalarType.onnx_compatible](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.onnx_compatible)|Yes|-|
|[torch.onnx.JitScalarType.onnx_type](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.onnx_type)|Yes|-|
|[torch.onnx.JitScalarType.scalar_name](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.scalar_name)|Yes|-|
|[torch.onnx.JitScalarType.torch_name](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.torch_name)|Yes|-|
|[torch.onnx.verification.GraphInfo](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.GraphInfo)|Yes|-|
|[torch.onnx.verification.VerificationOptions](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.VerificationOptions)|Yes|-|
