# torch.linalg

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:32:06.800Z pushedAt=2026-07-09T08:44:08.296Z -->

> [!NOTE]
> If the API's "Supported" column is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.linalg.cholesky|Yes|Supports fp32|
|torch.linalg.norm|Yes|Supports bf16, fp16, fp32|
|torch.linalg.lstsq|Yes|May fall back to CPU execution|
|torch.linalg.matmul|Yes|Supports fp16, fp32<br>Input supports up to 6 dimensions|
|torch.linalg.ldl_factor|No|-|
|torch.linalg.qr|Yes|Supports fp32, fp64, complex64, complex128|
|torch.linalg.solve_triangular|Yes|Supports fp32, fp64, complex64, complex128|
