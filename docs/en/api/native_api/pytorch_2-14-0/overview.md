# Overview

This document describes the support status and constraints of the PyTorch 2.14.0 native APIs on Ascend NPU. For details on how to use the PyTorch 2.14.0 native APIs, see [PyTorch Community Documentation](https://pytorch.org/docs/2.14/).

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.14/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.

## Version Inheritance and Compatibility Notes

This list inherits the product support status and constraints of PyTorch 2.13.0 and updates the APIs and links based on the 2.14 changes. New APIs not included in this list do not indicate that Ascend NPU validation has been completed.

- PyTorch 2.14 has removed the implementations of `torch.qr`, `torch.Tensor.qr`, `torch.cholesky`, and `torch.Tensor.cholesky`; calling them will report an error. This list removes the original `torch.qr` entry; use `torch.linalg.qr` or `torch.linalg.cholesky` instead. The `some` parameter of `qr` must be changed to `mode`; the upper triangular result of Cholesky can be obtained through `torch.linalg.cholesky(A).mH`.
- `torch.profiler.profile` and `torch.autograd.profiler.profile` have removed the `use_cuda` parameter; use `activities` and `use_device` respectively to select the device. The `with_modules` parameter is deprecated; for dynamic graph module tracking, use `with_stack=True`.

Version change basis: [PyTorch 2.14 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.14.0).
