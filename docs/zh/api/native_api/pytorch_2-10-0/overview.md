# 概述

介绍PyTorch2.10.0版本原生API在昇腾NPU上的支持情况与限制说明，PyTorch2.10.0版本原生API具体使用方法请参考[PyTorch社区文档](https://pytorch.org/docs/2.10/)。

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.10/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 版本继承与兼容性说明

本清单以PyTorch 2.7.1版本的产品支持情况和限制说明为基础，按目标版本修正文档链接，并移除已由上游删除的接口。未收录的新接口不表示已完成昇腾NPU验证。

- PyTorch 2.9已移除旧版ONNX导出接口`torch.onnx.dynamo_export`、`ExportOptions`、`enable_fake_mode`、`OnnxRegistry`、`DiagnosticOptions`和`is_onnxrt_backend_supported`，以及`torch.onnx.verification`中的旧验证接口。本清单不再列出这些接口；导出请使用`torch.onnx.export`，验证请使用`torch.onnx.verification.verify_onnx_program`。
- `torch.onnx.export`从2.9起默认使用`dynamo=True`；需要沿用旧TorchScript导出行为时应显式设置`dynamo=False`。
- 2.10中`torch.onnx.export`的`dynamic_axes`参数已废弃；使用新导出流程时请改用`dynamic_shapes`。`torch.profiler._KinetoProfile.export_memory_timeline`也已废弃。

版本变更依据：[PyTorch 2.9发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.9.0)、[PyTorch 2.10发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.10.0)。
