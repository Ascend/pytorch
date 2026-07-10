# Overview

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:28:50.282Z pushedAt=2026-07-09T08:44:08.246Z -->

This document introduces the support status and limitations of PyTorch 2.8.0 native APIs on Ascend NPU. For detailed usage of PyTorch 2.8.0 native APIs, refer to the [PyTorch Community Documentation](https://pytorch.org/docs/2.8/). The support status and limitations of native APIs on Ascend NPU fall into the following four categories:

- When "Supported" is "Yes" and "Restrictions and Notes" is "-", the API support is fully consistent with the native API.
- When "Supported" is "Yes" and "Restrictions and Notes" is not "-", the API support is not fully consistent with the native API. Pay attention to the support level on Ascend NPU.
- When "Supported" is "No" and "Restrictions and Notes" is "-", this API is not yet supported on Ascend NPU.
- Some APIs exist in the [PyTorch community documentation](https://pytorch.org/docs/2.8/) but are not included in this document. These are unverified APIs for Ascend NPU. Use them with caution on Ascend NPU. The documentation will be updated after subsequent verification.
