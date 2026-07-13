# torch.linalg

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:07:49.544Z pushedAt=2026-06-15T02:04:36.540Z -->

> [!NOTE]   
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.linalg.cholesky|Yes|Supports fp32|
|torch.linalg.norm|Yes|Supports bf16, fp16, fp32|
|torch.linalg.vector_norm|Yes|Supports bf16, fp16, fp32|
|torch.linalg.lstsq|Yes|May fall back to CPU execution|
|torch.linalg.matmul|Yes|Supports fp16, fp32<br>Input supports up to 6 dimensions|
|torch.linalg.ldl_factor|No|-|
|torch.linalg.qr|Yes|Supports fp32, fp64, complex64, complex128|
|torch.linalg.solve_triangular|Yes|Supports fp32, fp64, complex64, complex128|
