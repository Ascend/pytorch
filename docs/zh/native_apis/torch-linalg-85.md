# torch.linalg

> [!NOTE]   
> 若API“是否支持“为“是“，“限制与说明“为“-“，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.linalg.cholesky|是|支持fp32|
|torch.linalg.norm|是|支持bf16，fp16，fp32|
|torch.linalg.vector_norm|是|支持bf16，fp16，fp32|
|torch.linalg.lstsq|是|可能回退至CPU执行|
|torch.linalg.matmul|是|支持fp16，fp32<br>输入最大支持6维|
|torch.linalg.ldl_factor|否|-|
|torch.linalg.qr|是|支持fp32，fp64，complex64，complex128|
|torch.linalg.solve_triangular|是|支持fp32，fp64，complex64，complex128|


