# torch.onnx

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://docs.pytorch.org/docs/2.7/onnx.html).
> - Note: The modules and some APIs covered in this document are located on independent pages. For related descriptions, see the documentation:
>    - [ONNX Backend for TorchDynamo](https://docs.pytorch.org/docs/2.7/onnx_dynamo_onnxruntime_backend.html)
>    - [torch.onnx.verification](https://docs.pytorch.org/docs/2.7/onnx_verification.html)
>    - [TorchDynamo-based ONNX Exporter](https://docs.pytorch.org/docs/2.7/onnx_dynamo.html)
>    - [TorchScript-based ONNX Exporter](https://docs.pytorch.org/docs/2.7/onnx_torchscript.html)

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [TorchDynamo-based ONNX Exporter](#torchdynamo-based-onnx-exporter)
- [TorchScript-based ONNX Exporter](#torchscript-based-onnx-exporter)

</div>

<div style="display:none;">

## &#8203;torch.onnx

</div>

### torch.onnx.is_onnxrt_backend_supported

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.is_onnxrt_backend_supported](https://pytorch.org/docs/2.7/onnx_dynamo_onnxruntime_backend.html#torch.onnx.is_onnxrt_backend_supported)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id3 -->

</div>

### torch.onnx.verification.check_export_model_diff

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.check_export_model_diff](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.check_export_model_diff)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id108 -->

</div>

### torch.onnx.verification.find_mismatch

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.find_mismatch](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.find_mismatch)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

</div>

### torch.onnx.verification.verify

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.verify](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.verify)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

</div>

### torch.onnx.verification.verify_aten_graph

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.verify_aten_graph](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.verify_aten_graph)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

</div>

### <code><i>class</i></code> torch.onnx.verification.GraphInfo

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.GraphInfo](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.GraphInfo)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

</div>

### <code><i>class</i></code> torch.onnx.verification.GraphInfoPrettyPrinter

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.GraphInfoPrettyPrinter](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.GraphInfoPrettyPrinter)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

</div>

### <code><i>class</i></code> torch.onnx.verification.VerificationOptions

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.VerificationOptions](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.VerificationOptions)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

</div>

## TorchDynamo-based ONNX Exporter

### torch.onnx.dynamo_export

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.dynamo_export](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.dynamo_export)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.onnx.ONNXProgram

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id24 -->

> <font size="3">compute_values()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.compute_values](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.compute_values)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT</term>: Supported
<!-- end id108 -->

</div>

> <font size="3">model_proto()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.model_proto](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.model_proto)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

</div>

> <font size="3">save()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.save](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.save)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

</div>

> <font size="3">initialize_inference_session()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.initialize_inference_session](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.initialize_inference_session)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

</div>

> <font size="3">release()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.release](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.release)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id108 -->

</div>

> <font size="3">optimize()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.optimize](https://docs.pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.optimize)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

</div>

</div>

### <code><i>class</i></code> torch.onnx.ExportOptions

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ExportOptions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ExportOptions)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

</div>

### torch.onnx.enable_fake_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.enable_fake_mode](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.enable_fake_mode)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id39 -->

</div>

### <code><i>class</i></code> torch.onnx.OnnxExporterError

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.OnnxExporterError](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxExporterError)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

</div>

### <code><i>class</i></code> torch.onnx.OnnxRegistry

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.OnnxRegistry](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

> <font size="3">get_op_functions()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.OnnxRegistry.get_op_functions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.get_op_functions)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

</div>

> <font size="3">is_registered_op()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.OnnxRegistry.is_registered_op](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.is_registered_op)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

</div>

> <font size="3"><code><i>property</i></code> opset_version</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.OnnxRegistry.opset_version](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.opset_version)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

</div>

> <font size="3">register_op()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.OnnxRegistry.register_op](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.register_op)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

</div>

</div>

### <code><i>class</i></code> torch.onnx.DiagnosticOptions

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.DiagnosticOptions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.DiagnosticOptions)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

</div>

## TorchScript-based ONNX Exporter

### torch.onnx.export

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.export)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

### torch.onnx.register_custom_op_symbolic

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.register_custom_op_symbolic](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.register_custom_op_symbolic)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

</div>

### torch.onnx.unregister_custom_op_symbolic

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.unregister_custom_op_symbolic](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.unregister_custom_op_symbolic)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id69 -->

</div>

### torch.onnx.select_model_mode_for_export

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.select_model_mode_for_export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.select_model_mode_for_export)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

</div>

### torch.onnx.is_in_onnx_export

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.is_in_onnx_export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.is_in_onnx_export)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

</div>

### <code><i>class</i></code> torch.onnx.JitScalarType

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.JitScalarType](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.JitScalarType.dtype](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.dtype)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

</div>

> <font size="3">from_dtype()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.JitScalarType.from_dtype](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.from_dtype)

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id84 -->

**Constraints & Notes**: `input` only supports bool

</div>

> <font size="3">from_value()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.JitScalarType.from_value](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.from_value)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

**Constraints & Notes**: `input` only supports bool

</div>

> <font size="3">onnx_compatible()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.JitScalarType.onnx_compatible](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.onnx_compatible)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id90 -->

</div>

> <font size="3">onnx_type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.JitScalarType.onnx_type](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.onnx_type)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

</div>

> <font size="3">scalar_name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.JitScalarType.scalar_name](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.scalar_name)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

</div>

> <font size="3">torch_name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.JitScalarType.torch_name](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.torch_name)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id99 -->

</div>

</div>

### <code><i>class</i></code> torch.onnx.ONNXProgram

<div style="margin-left: 2em">

> <font size="3">apply_weights()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.apply_weights](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.apply_weights)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id102 -->

</div>

</div>

### torch.onnx.verification.verify_onnx_program

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.verify_onnx_program](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.verify_onnx_program)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id105 -->

</div>
