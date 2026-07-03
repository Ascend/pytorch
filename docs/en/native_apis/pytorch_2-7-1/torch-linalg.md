# torch.linalg

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:07:49.544Z pushedAt=2026-06-15T02:04:36.540Z -->

> [!NOTE]   
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.linalg.cholesky](https://pytorch.org/docs/2.7/generated/torch.linalg.cholesky.html)|Yes|Supports fp32|
|[torch.linalg.norm](https://pytorch.org/docs/2.7/generated/torch.linalg.norm.html)|Yes|Supports bf16, fp16, fp32|
|[torch.linalg.vector_norm](https://pytorch.org/docs/2.7/generated/torch.linalg.vector_norm.html)|Yes|Supports bf16, fp16, fp32|
|[torch.linalg.lstsq](https://pytorch.org/docs/2.7/generated/torch.linalg.lstsq.html)|Yes|May fall back to CPU execution|
|[torch.linalg.matmul](https://pytorch.org/docs/2.7/generated/torch.linalg.matmul.html)|Yes|Supports fp16, fp32<br>Input supports up to 6 dimensions|
|[torch.linalg.ldl_factor](https://pytorch.org/docs/2.7/generated/torch.linalg.ldl_factor.html)|No|-|
|[torch.linalg.qr](https://pytorch.org/docs/2.7/generated/torch.linalg.qr.html)|Yes|Supports fp32, fp64, complex64, complex128|
|[torch.linalg.solve_triangular](https://pytorch.org/docs/2.7/generated/torch.linalg.solve_triangular.html)|Yes|Supports fp32, fp64, complex64, complex128|
