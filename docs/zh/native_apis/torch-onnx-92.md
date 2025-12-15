# torch.onnx

> [!NOTE]  
> 若API“是否支持“为“是“，“限制与说明“为“-“，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.onnx.dynamo_export|是|支持fp32|
|torch.onnx.ExportOptions|是|-|
|torch.onnx.enable_fake_mode|否|-|
|torch.onnx.ONNXProgram|是|-|
|torch.onnx.ONNXProgram.model_proto|否|-|
|torch.onnx.ONNXProgram.save|是|-|
|torch.onnx.OnnxExporterError|否|-|
|torch.onnx.OnnxRegistry|否|-|
|torch.onnx.OnnxRegistry.get_op_functions|否|-|
|torch.onnx.OnnxRegistry.is_registered_op|否|-|
|torch.onnx.OnnxRegistry.opset_version|否|-|
|torch.onnx.OnnxRegistry.register_op|否|-|
|torch.onnx.DiagnosticOptions|否|-|
|torch.onnx.is_onnxrt_backend_supported|是|-|
|torch.onnx.export|是|-|
|torch.onnx.register_custom_op_symbolic|是|-|
|torch.onnx.unregister_custom_op_symbolic|是|-|
|torch.onnx.select_model_mode_for_export|是|-|
|torch.onnx.is_in_onnx_export|是|-|
|torch.onnx.verification.find_mismatch|是|-|
|torch.onnx.JitScalarType|是|-|
|torch.onnx.JitScalarType.dtype|是|-|
|torch.onnx.JitScalarType.from_dtype|是|支持bool|
|torch.onnx.JitScalarType.from_value|是|支持bool|
|torch.onnx.JitScalarType.onnx_compatible|是|-|
|torch.onnx.JitScalarType.onnx_type|是|-|
|torch.onnx.JitScalarType.scalar_name|是|-|
|torch.onnx.JitScalarType.torch_name|是|-|
|torch.onnx.verification.GraphInfo|是|-|
|torch.onnx.verification.VerificationOptions|是|-|


