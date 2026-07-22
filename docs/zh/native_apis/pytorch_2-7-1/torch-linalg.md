# torch.linalg

> [!NOTE]   
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|[torch.linalg.cholesky](https://pytorch.org/docs/2.7/generated/torch.linalg.cholesky.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.linalg.norm](https://pytorch.org/docs/2.7/generated/torch.linalg.norm.html)|是|支持bf16，fp16，fp32|
|[torch.linalg.vector_norm](https://pytorch.org/docs/2.7/generated/torch.linalg.vector_norm.html)|是|支持bf16，fp16，fp32|
|[torch.linalg.lstsq](https://pytorch.org/docs/2.7/generated/torch.linalg.lstsq.html)|是<br>暂不支持<term>Ascend 950DT</term>|可能回退至CPU执行|
|[torch.linalg.matmul](https://pytorch.org/docs/2.7/generated/torch.linalg.matmul.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp16，fp32<br>输入最大支持6维|
|[torch.linalg.ldl_factor](https://pytorch.org/docs/2.7/generated/torch.linalg.ldl_factor.html)|否|-|
|[torch.linalg.qr](https://pytorch.org/docs/2.7/generated/torch.linalg.qr.html)|是|支持fp32，fp64，complex64，complex128|
|[torch.linalg.solve_triangular](https://pytorch.org/docs/2.7/generated/torch.linalg.solve_triangular.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32，fp64，complex64，complex128|
