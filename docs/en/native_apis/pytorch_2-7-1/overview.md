# Overview

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:02:56.768Z pushedAt=2026-06-15T02:04:36.425Z -->

This document introduces the support and restrictions of PyTorch 2.7.1 native APIs on the Ascend NPU. For specific usage of PyTorch 2.7.1 native APIs, refer to the [PyTorch community documentation](https://pytorch.org/docs/2.7/). The support and restrictions of native APIs on the Ascend NPU can be classified into the following four categories:

- If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

- If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is not "-", it means the API support is inconsistent with the native API. Note the support status on the Ascend NPU.

- If the "Supported" column for an API is "No" and the "Restrictions and Notes" column is "-", it means this API is not yet supported on the Ascend NPU.

- Some APIs are listed in the [PyTorch community documentation](https://pytorch.org/docs/2.7/) but are not included in this document. These are unverified APIs for the Ascend NPU. Use them with caution on the Ascend NPU, and the documentation will be continuously updated after subsequent verification.
