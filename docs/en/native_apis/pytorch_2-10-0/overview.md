# Overview

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T07:53:07.718Z pushedAt=2026-06-14T09:16:34.704Z -->

This document describes the support status and Restrictions of native APIs in PyTorch 2.10.0 on Ascend NPU. For specific usage of native APIs in PyTorch 2.10.0, refer to the [PyTorch documentation](https://pytorch.org/docs/2.10/). The support status and Restrictions of native APIs on Ascend NPU can be categorized into the following types:

- If "Supported" is "Yes" and "Restrictions and Notes" is "-", it means that the API support is consistent with the native API.

- If "Supported" is "Yes" and "Restrictions and Notes" is not "-", it means that the API support is inconsistent with the native API. Pay attention to the support level on the Ascend NPU.

- If "Supported" is "No" and "Restrictions and Notes" is "-", it means that this API is not currently supported on the Ascend NPU.

- Some APIs are listed in the [PyTorch documentation](https://pytorch.org/docs/2.10/) but are not included in this document. These are unverified APIs on the Ascend NPU. Use them with caution on the Ascend NPU. The document will be continuously updated after subsequent verification.
