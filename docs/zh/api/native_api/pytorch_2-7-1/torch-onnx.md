# torch.onnx

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.7/onnx.html)。
> - 注：当前文档包含的模块及部分API位于独立页面内，相关说明可查看文档：
>    - [ONNX Backend for TorchDynamo](https://docs.pytorch.org/docs/2.7/onnx_dynamo_onnxruntime_backend.html)
>    - [torch.onnx.verification](https://docs.pytorch.org/docs/2.7/onnx_verification.html)
>    - [TorchDynamo-based ONNX Exporter](https://docs.pytorch.org/docs/2.7/onnx_dynamo.html)
>    - [TorchScript-based ONNX Exporter](https://docs.pytorch.org/docs/2.7/onnx_torchscript.html)

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [TorchDynamo-based ONNX Exporter](#torchdynamo-based-onnx-exporter)
- [TorchScript-based ONNX Exporter](#torchscript-based-onnx-exporter)

</div>

<div style="display:none;">

## &#8203;torch.onnx

</div>

### torch.onnx.is_onnxrt_backend_supported

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.is_onnxrt_backend_supported](https://pytorch.org/docs/2.7/onnx_dynamo_onnxruntime_backend.html#torch.onnx.is_onnxrt_backend_supported)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

</div>

### torch.onnx.verification.find_mismatch

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.verification.find_mismatch](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.find_mismatch)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

</div>

### torch.onnx.verification.verify

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.verification.verify](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.verify)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

</div>

### torch.onnx.verification.verify_aten_graph

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.verification.verify_aten_graph](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.verify_aten_graph)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id12 -->

</div>

### <code><i>class</i></code> torch.onnx.verification.GraphInfo

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.verification.GraphInfo](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.GraphInfo)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id15 -->

</div>

### <code><i>class</i></code> torch.onnx.verification.VerificationOptions

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.verification.VerificationOptions](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.VerificationOptions)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id18 -->

</div>

## TorchDynamo-based ONNX Exporter

### torch.onnx.dynamo_export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.dynamo_export](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.dynamo_export)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id21 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.onnx.ONNXProgram

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id24 -->

> <font size="3">model_proto()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram.model_proto](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.model_proto)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id27 -->

</div>

> <font size="3">save()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram.save](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.save)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id30 -->

</div>

> <font size="3">initialize_inference_session()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram.initialize_inference_session](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.initialize_inference_session)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id33 -->

</div>

</div>

### <code><i>class</i></code> torch.onnx.ExportOptions

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ExportOptions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ExportOptions)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id36 -->

</div>

### torch.onnx.enable_fake_mode

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.enable_fake_mode](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.enable_fake_mode)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id39 -->

</div>

### <code><i>class</i></code> torch.onnx.OnnxExporterError

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxExporterError](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxExporterError)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id42 -->

</div>

### <code><i>class</i></code> torch.onnx.OnnxRegistry

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxRegistry](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id45 -->

> <font size="3">get_op_functions()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxRegistry.get_op_functions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.get_op_functions)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id48 -->

</div>

> <font size="3">is_registered_op()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxRegistry.is_registered_op](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.is_registered_op)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id51 -->

</div>

> <font size="3"><code><i>property</i></code> opset_version</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxRegistry.opset_version](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.opset_version)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id54 -->

</div>

> <font size="3">register_op()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxRegistry.register_op](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.OnnxRegistry.register_op)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id57 -->

</div>

</div>

### <code><i>class</i></code> torch.onnx.DiagnosticOptions

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.DiagnosticOptions](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.DiagnosticOptions)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id60 -->

</div>

## TorchScript-based ONNX Exporter

### torch.onnx.export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.export)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id63 -->

</div>

### torch.onnx.register_custom_op_symbolic

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.register_custom_op_symbolic](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.register_custom_op_symbolic)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id66 -->

</div>

### torch.onnx.unregister_custom_op_symbolic

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.unregister_custom_op_symbolic](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.unregister_custom_op_symbolic)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id69 -->

</div>

### torch.onnx.select_model_mode_for_export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.select_model_mode_for_export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.select_model_mode_for_export)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id72 -->

</div>

### torch.onnx.is_in_onnx_export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.is_in_onnx_export](https://pytorch.org/docs/2.7/onnx_torchscript.html#torch.onnx.is_in_onnx_export)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id75 -->

</div>

### <code><i>class</i></code> torch.onnx.JitScalarType

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id78 -->

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.dtype](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.dtype)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id81 -->

</div>

> <font size="3">from_dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.from_dtype](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.from_dtype)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id84 -->

**限制与说明**：`input`仅支持bool

</div>

> <font size="3">from_value()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.from_value](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.from_value)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id87 -->

**限制与说明**：`input`仅支持bool

</div>

> <font size="3">onnx_compatible()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.onnx_compatible](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.onnx_compatible)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id90 -->

</div>

> <font size="3">onnx_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.onnx_type](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.onnx_type)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id93 -->

</div>

> <font size="3">scalar_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.scalar_name](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.scalar_name)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id96 -->

</div>

> <font size="3">torch_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.JitScalarType.torch_name](https://pytorch.org/docs/2.7/generated/torch.onnx.JitScalarType.html#torch.onnx.JitScalarType.torch_name)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id99 -->

</div>

</div>

### <code><i>class</i></code> torch.onnx.ONNXProgram

<div style="margin-left: 2em">

> <font size="3">apply_weights()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram.apply_weights](https://pytorch.org/docs/2.7/onnx_dynamo.html#torch.onnx.ONNXProgram.apply_weights)

**支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id102 -->

</div>

</div>

### torch.onnx.verification.verify_onnx_program

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.verification.verify_onnx_program](https://pytorch.org/docs/2.7/onnx_verification.html#torch.onnx.verification.verify_onnx_program)

**支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id105 -->

</div>
