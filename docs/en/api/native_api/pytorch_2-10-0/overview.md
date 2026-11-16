# Overview

This document describes the support status and constraints of native PyTorch 2.10.0 APIs on the Ascend NPU. For specific usage of native PyTorch 2.10.0 APIs, see the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/).

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.

## Version Inheritance and Compatibility Notes

This list is based on the product support status and constraints of PyTorch 2.7.1, corrects documentation links for the target version, and removes interfaces already deleted upstream. New interfaces not included do not indicate that they have been verified on the Ascend NPU.

- PyTorch 2.9 has removed the legacy ONNX export interfaces `torch.onnx.dynamo_export`, `ExportOptions`, `enable_fake_mode`, `OnnxRegistry`, `DiagnosticOptions`, and `is_onnxrt_backend_supported`, as well as the legacy verification interfaces in `torch.onnx.verification`. This list no longer includes these interfaces; for export, use `torch.onnx.export`, and for verification, use `torch.onnx.verification.verify_onnx_program`.
- Since version 2.9, `torch.onnx.export` uses `dynamo=True` by default; to retain the legacy TorchScript export behavior, explicitly set `dynamo=False`.
- In version 2.10, the `dynamic_axes` parameter of `torch.onnx.export` is deprecated; when using the new export flow, use `dynamic_shapes` instead. `torch.profiler._KinetoProfile.export_memory_timeline` is also deprecated.

Basis for version changes: [PyTorch 2.9 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.9.0), [PyTorch 2.10 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.10.0).
