# torch.onnx

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [A simple example](#a-simple-example)
- [API Reference](#api-reference)

## base API

### torch.onnx.enable_fake_mode

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.enable_fake_mode](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.enable_fake_mode)

**是否支持**：否

</div>

### _`class`_ torch.onnx.OnnxExporterError

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxExporterError](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxExporterError)

**是否支持**：否

</div>

### torch.onnx.is_onnxrt_backend_supported

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.is_onnxrt_backend_supported](https://pytorch.org/docs/2.7/onnx_dynamo_onnxruntime_backend.html#torch.onnx.is_onnxrt_backend_supported)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.onnx.register_custom_op_symbolic

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.register_custom_op_symbolic](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.register_custom_op_symbolic)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.onnx.unregister_custom_op_symbolic

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.unregister_custom_op_symbolic](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.unregister_custom_op_symbolic)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.onnx.select_model_mode_for_export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.select_model_mode_for_export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.select_model_mode_for_export)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.onnx.is_in_onnx_export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.is_in_onnx_export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.is_in_onnx_export)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.onnx.verification.find_mismatch

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.verification.find_mismatch](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.find_mismatch)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.onnx.JitScalarType

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.dtype](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.dtype)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">from_dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.from_dtype](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.from_dtype)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bool

</div>

> <font size="3">from_value()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.from_value](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.from_value)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bool

</div>

> <font size="3">onnx_compatible()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.onnx_compatible](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.onnx_compatible)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">onnx_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.onnx_type](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.onnx_type)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">scalar_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.scalar_name](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.scalar_name)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">torch_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.torch_name](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.torch_name)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.onnx.verification.GraphInfo

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.verification.GraphInfo](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.GraphInfo)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.onnx.verification.VerificationOptions

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.verification.VerificationOptions](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.VerificationOptions)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## A simple example

### torch.onnx.dynamo_export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.dynamo_export](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.dynamo_export)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.onnx.ONNXProgram

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">model_proto()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram.model_proto](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.model_proto)

**是否支持**：否

</div>

> <font size="3">save()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram.save](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.save)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### torch.onnx.export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.export)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## API Reference

### _`class`_ torch.onnx.ExportOptions

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ExportOptions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ExportOptions)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.onnx.OnnxRegistry

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxRegistry](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry)

**是否支持**：否

> <font size="3">get_op_functions()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxRegistry.get_op_functions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.get_op_functions)

**是否支持**：否

</div>

> <font size="3">is_registered_op()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxRegistry.is_registered_op](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.is_registered_op)

**是否支持**：否

</div>

> <font size="3">opset_version()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxRegistry.opset_version](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.opset_version)

**是否支持**：否

</div>

> <font size="3">register_op()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxRegistry.register_op](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.register_op)

**是否支持**：否

</div>

</div>

### _`class`_ torch.onnx.DiagnosticOptions

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.DiagnosticOptions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.DiagnosticOptions)

**是否支持**：否

</div>
