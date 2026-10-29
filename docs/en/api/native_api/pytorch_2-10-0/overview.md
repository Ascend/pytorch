# Overview

This document describes the support status and restrictions of native APIs of PyTorch 2.10.0 on the Ascend NPU. For details about how to use native APIs of PyTorch 2.10.0, see the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/).

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.

## Version Inheritance and Compatibility Notes

This list is based on the product support status and restrictions of PyTorch 2.7.1, with document links corrected for the target version and interfaces deleted upstream removed. New interfaces not included in this list do not mean that Ascend NPU verification has been completed.

- PyTorch 2.9 has removed the legacy ONNX export interfaces `torch.onnx.dynamo_export`, `ExportOptions`, `enable_fake_mode`, `OnnxRegistry`, `DiagnosticOptions`, and `is_onnxrt_backend_supported`, as well as the legacy verification interfaces in `torch.onnx.verification`. These interfaces are no longer listed in this document. For export, use `torch.onnx.export`; for verification, use `torch.onnx.verification.verify_onnx_program`.
- Since 2.9, `torch.onnx.export` uses `dynamo=True` by default. If the legacy TorchScript export behavior is required, set `dynamo=False` explicitly.
- In 2.10, the `dynamic_axes` parameter of `torch.onnx.export` has been deprecated. When using the new export workflow, use `dynamic_shapes` instead. `torch.profiler._KinetoProfile.export_memory_timeline` has also been deprecated.

Basis of version changes: [PyTorch 2.9 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.9.0) and [PyTorch 2.10 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.10.0).
