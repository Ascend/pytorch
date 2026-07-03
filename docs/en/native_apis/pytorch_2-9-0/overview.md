# Overview

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:09:57.752Z pushedAt=2026-06-15T03:25:49.132Z -->

This document introduces the support and restrictions of PyTorch 2.9.0 native APIs on the Ascend NPU. For specific usage of PyTorch 2.7.1 native APIs, refer to the [PyTorch community documentation](https://pytorch.org/docs/2.9/). The support and restrictions of native APIs on the Ascend NPU can be classified into the following four categories:

- If "Supported" is "Yes" and "Restrictions and Notes" is "-", it means that the API support is consistent with the native API.

- If "Supported" is "Yes" and "Restrictions and Notes" is not "-", it means that the API support is inconsistent with the native API. Pay attention to the support level on the Ascend NPU.

- If "Supported" is "No" and "Restrictions and Notes" is "-", it means that this API is not yet supported on the Ascend NPU.

- Some APIs are listed in the [PyTorch community documentation](https://pytorch.org/docs/2.9/) but are not included in this document. These are unverified APIs for the Ascend NPU. Use them with caution on the Ascend NPU, and the documentation will be continuously updated after subsequent verification.
