# torch.onnx

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://docs.pytorch.org/docs/2.9/onnx.html).
> - Note: The modules and some APIs included in this document are located on separate pages; for related descriptions, see the documentation:
>    - [torch.onnx.verification](https://docs.pytorch.org/docs/2.9/onnx_verification.html)
>
> - Since 2.9, the legacy Dynamo export and ONNX Runtime backend interfaces have been removed; `torch.onnx.export` uses `dynamo=True` by default. The legacy TorchScript auxiliary interfaces are deprecated; prefer the new export workflow.

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [torch.onnx APIs](#torchonnx-apis)

</div>

<div style="display:none;">

## &#8203;torch.onnx

</div>

## torch.onnx APIs

### <code><i>class</i></code> torch.onnx.ONNXProgram

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram)

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

> <font size="3">model_proto()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.model_proto](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram.model_proto)

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

> <font size="3">save()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.save](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram.save)

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

> <font size="3">initialize_inference_session()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.initialize_inference_session](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram.initialize_inference_session)

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

> <font size="3">release()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.release](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram.release)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

</div>

> <font size="3">optimize()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.optimize](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram.optimize)

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

</div>

### <code><i>class</i></code> torch.onnx.OnnxExporterError

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.OnnxExporterError](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.OnnxExporterError)

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

### torch.onnx.export

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.export](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.export)

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

### torch.onnx.register_custom_op_symbolic

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.register_custom_op_symbolic](https://docs.pytorch.org/docs/2.9/onnx.html#torch.onnx.register_custom_op_symbolic)

**Version Notes**: This legacy TorchScript export auxiliary interface has been deprecated; use `torch.onnx.export` in the new export workflow.

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

</div>

### torch.onnx.unregister_custom_op_symbolic

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.unregister_custom_op_symbolic](https://docs.pytorch.org/docs/2.9/onnx.html#torch.onnx.unregister_custom_op_symbolic)

**Version Notes**: This legacy TorchScript export auxiliary interface has been deprecated; use `torch.onnx.export` in the new export workflow.

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

</div>

### torch.onnx.select_model_mode_for_export

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.select_model_mode_for_export](https://docs.pytorch.org/docs/2.9/onnx.html#torch.onnx.select_model_mode_for_export)

**Version Notes**: This API has been deprecated; set the model to training or inference mode before export.

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

### torch.onnx.is_in_onnx_export

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.is_in_onnx_export](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.is_in_onnx_export)

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

### <code><i>class</i></code> torch.onnx.ONNXProgram

<div style="margin-left: 2em">

> <font size="3">apply_weights()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.apply_weights](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram.apply_weights)

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

</div>

### torch.onnx.verification.verify_onnx_program

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.verify_onnx_program](https://pytorch.org/docs/2.9/onnx_verification.html#torch.onnx.verification.verify_onnx_program)

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
