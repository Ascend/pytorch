# Overview

This section describes the support status and constraints of native PyTorch 2.9.0 APIs on the Ascend NPU. For the specific usage of native PyTorch 2.9.0 APIs, refer to the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/).

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.

## Version Inheritance and Compatibility Notes

This list is based on the product support status and constraints of PyTorch 2.7.1, with document links updated for the target version and interfaces removed by upstream deleted. New interfaces not included in this list do not mean that they have been verified on the Ascend NPU.

- PyTorch 2.9 has removed the legacy ONNX export interfaces `torch.onnx.dynamo_export`, `ExportOptions`, `enable_fake_mode`, `OnnxRegistry`, `DiagnosticOptions`, and `is_onnxrt_backend_supported`, as well as the legacy verification interfaces in `torch.onnx.verification`. This list no longer includes these interfaces; use `torch.onnx.export` for export and `torch.onnx.verification.verify_onnx_program` for verification.
- `torch.onnx.export` uses `dynamo=True` by default starting from 2.9; to retain the legacy TorchScript export behavior, explicitly set `dynamo=False`.

Basis of version changes: [PyTorch 2.9 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.9.0).
