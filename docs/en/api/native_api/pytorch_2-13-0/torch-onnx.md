# torch.onnx

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.13/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.13/onnx.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [torch.onnx APIs](#torchonnx-apis)

</div>

<div style="display:none;">

## &#8203;torch.onnx

</div>

## torch.onnx APIs

### <code><i>class</i></code> torch.onnx.OnnxExporterError

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.OnnxExporterError](https://pytorch.org/docs/2.13/onnx_export.html#torch.onnx.OnnxExporterError)

**Support Status**:
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

</div>

### torch.onnx.register_custom_op_symbolic

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.register_custom_op_symbolic](https://pytorch.org/docs/2.13/onnx.html#torch.onnx.register_custom_op_symbolic)

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

### torch.onnx.unregister_custom_op_symbolic

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.unregister_custom_op_symbolic](https://pytorch.org/docs/2.13/onnx.html#torch.onnx.unregister_custom_op_symbolic)

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

### torch.onnx.select_model_mode_for_export

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.select_model_mode_for_export](https://pytorch.org/docs/2.13/onnx.html#torch.onnx.select_model_mode_for_export)

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

### torch.onnx.is_in_onnx_export

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.is_in_onnx_export](https://pytorch.org/docs/2.13/onnx_export.html#torch.onnx.is_in_onnx_export)

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

<!-- npu="910b,A3,950" id37 -->
### <code><i>class</i></code> torch.onnx.ONNXProgram
<!-- end id37 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id38 -->
**Native Documentation**: [torch.onnx.ONNXProgram](https://pytorch.org/docs/2.13/onnx_export.html#torch.onnx.ONNXProgram)

**Support Status**:
<!-- end id38 -->

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->

> <font size="3">apply_weights()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.apply_weights](https://pytorch.org/docs/2.13/onnx_export.html#torch.onnx.ONNXProgram.apply_weights)

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

<!-- npu="910b,A3" id39 -->
> <font size="3">model_proto()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.model_proto](https://pytorch.org/docs/2.13/onnx.html#torch-onnx-apis)

**Support Status**:
<!-- end id39 -->

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->

<!-- npu="910b,A3" id40 -->
</div>
<!-- end id40 -->

<!-- npu="910b,A3" id41 -->
> <font size="3">save()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.save](https://pytorch.org/docs/2.13/onnx.html#torch-onnx-apis)

**Support Status**:
<!-- end id41 -->

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->

<!-- npu="910b,A3" id42 -->
</div>
<!-- end id42 -->

> <font size="3">initialize_inference_session()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.ONNXProgram.initialize_inference_session](https://pytorch.org/docs/2.13/onnx_export.html#torch.onnx.ONNXProgram.initialize_inference_session)

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

</div>

### torch.onnx.export

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.export](https://pytorch.org/docs/2.13/onnx_export.html#torch.onnx.export)

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

### torch.onnx.verification.verify_onnx_program

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.verify_onnx_program](https://pytorch.org/docs/2.13/onnx_verification.html#torch.onnx.verification.verify_onnx_program)

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

### <code><i>class</i></code> torch.onnx.verification.VerificationInfo

<div style="margin-left: 2em">

> <font size="3">from_tensors()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.onnx.verification.VerificationInfo.from\_tensors](https://docs.pytorch.org/docs/2.13/onnx_verification.html#torch.onnx.verification.VerificationInfo.from_tensors)

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

</div>

</div>
