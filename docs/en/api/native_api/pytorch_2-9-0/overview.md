# Overview

This document describes the Support Status and Constraints & Notes of PyTorch 2.9.0 native APIs on Ascend NPU. For the specific usage of PyTorch 2.9.0 native APIs, see the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/).

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.

## Version Inheritance and Compatibility Notes

This list is based on the product Support Status and Constraints & Notes of the PyTorch 2.7.1 version, with document links corrected to the target version and interfaces removed by upstream deleted. New interfaces not included in this list do not indicate that they have been validated on Ascend NPU.

- PyTorch 2.9 has removed the legacy ONNX export interfaces `torch.onnx.dynamo_export`, `ExportOptions`, `enable_fake_mode`, `OnnxRegistry`, `DiagnosticOptions`, and `is_onnxrt_backend_supported`, as well as the legacy verification interfaces in `torch.onnx.verification`. These interfaces are no longer listed in this document; use `torch.onnx.export` for export and `torch.onnx.verification.verify_onnx_program` for verification.
- Since 2.9, `torch.onnx.export` uses `dynamo=True` by default; to retain the legacy TorchScript export behavior, explicitly set `dynamo=False`.

Version changes are based on: [PyTorch 2.9 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.9.0).
