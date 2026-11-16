# Overview

This document describes the support status and constraints of native PyTorch 2.14.0 APIs on the Ascend NPU. For specific usage of native PyTorch 2.14.0 APIs, see the [PyTorch Community Documentation](https://pytorch.org/docs/2.14/).

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.14/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.

## Version Inheritance and Compatibility Notes

This list inherits the product support status and constraints of PyTorch 2.13.0 and corrects interfaces and links according to version 2.14 changes. New interfaces not included do not indicate that they have been verified on the Ascend NPU.

- Version 2.14 has removed the implementations of `torch.qr`, `torch.Tensor.qr`, `torch.cholesky`, and `torch.Tensor.cholesky`; calling them reports an error. This list removes the original `torch.qr` entry; use `torch.linalg.qr` or `torch.linalg.cholesky` instead. The `some` parameter of `qr` must be changed to `mode`; the Cholesky upper triangular result can be obtained through `torch.linalg.cholesky(A).mH`.
- The `use_cuda` parameter has been removed from `torch.profiler.profile` and `torch.autograd.profiler.profile`; use `activities` and `use_device` to select devices instead. The `with_modules` parameter is deprecated; for dynamic graph module tracking, use `with_stack=True`.

Basis for version changes: [PyTorch 2.14 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.14.0).
