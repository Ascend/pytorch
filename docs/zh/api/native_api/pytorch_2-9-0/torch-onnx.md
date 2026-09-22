# torch.onnx

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.9/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.9/onnx.html)。
> - 注：当前文档包含的模块及部分API位于独立页面内，相关说明可查看文档：
>    - [torch.onnx.verification](https://docs.pytorch.org/docs/2.9/onnx_verification.html)
>
> - 2.9起已移除旧版Dynamo导出和ONNX Runtime后端接口；`torch.onnx.export`默认使用`dynamo=True`。旧版TorchScript辅助接口已废弃，请优先使用新导出流程。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [torch.onnx APIs](#torchonnx-apis)

</div>

<div style="display:none;">

## &#8203;torch.onnx

</div>

## torch.onnx APIs

### <code><i>class</i></code> torch.onnx.ONNXProgram

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">model_proto()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram.model_proto](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram.model_proto)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">save()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram.save](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram.save)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">initialize_inference_session()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram.initialize_inference_session](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram.initialize_inference_session)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

</div>

### <code><i>class</i></code> torch.onnx.OnnxExporterError

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.OnnxExporterError](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.OnnxExporterError)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.onnx.export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.export](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.export)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.onnx.register_custom_op_symbolic

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.register_custom_op_symbolic](https://docs.pytorch.org/docs/2.9/onnx.html#torch.onnx.register_custom_op_symbolic)

**版本说明**：该旧版TorchScript导出辅助接口已废弃；新导出流程请使用`torch.onnx.export`。

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.onnx.unregister_custom_op_symbolic

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.unregister_custom_op_symbolic](https://docs.pytorch.org/docs/2.9/onnx.html#torch.onnx.unregister_custom_op_symbolic)

**版本说明**：该旧版TorchScript导出辅助接口已废弃；新导出流程请使用`torch.onnx.export`。

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.onnx.select_model_mode_for_export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.select_model_mode_for_export](https://docs.pytorch.org/docs/2.9/onnx.html#torch.onnx.select_model_mode_for_export)

**版本说明**：该接口已废弃，请在导出前设置模型训练或推理模式。

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.onnx.is_in_onnx_export

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.is_in_onnx_export](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.is_in_onnx_export)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.onnx.ONNXProgram

<div style="margin-left: 2em">

> <font size="3">apply_weights()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.ONNXProgram.apply_weights](https://docs.pytorch.org/docs/2.9/onnx_export.html#torch.onnx.ONNXProgram.apply_weights)

**支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

</div>

### torch.onnx.verification.verify_onnx_program

<div style="margin-left: 2em">

**原生文档**：[torch.onnx.verification.verify_onnx_program](https://pytorch.org/docs/2.9/onnx_verification.html#torch.onnx.verification.verify_onnx_program)

**支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>
