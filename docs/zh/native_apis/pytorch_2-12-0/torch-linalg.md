# torch.linalg

> [!NOTE]   
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [Decompositions](#decompositions)
- [Matrix Properties](#matrix-properties)
- [Solvers](#solvers)
- [Matrix Products](#matrix-products)
- [Experimental Functions](#experimental-functions)

## Decompositions

### torch.linalg.cholesky

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.cholesky](https://pytorch.org/docs/2.12/generated/torch.linalg.cholesky.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.linalg.qr

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.qr](https://pytorch.org/docs/2.12/generated/torch.linalg.qr.html)

**是否支持**：是

**限制与说明**： 支持fp32，fp64，complex64，complex128

</div>

## Matrix Properties

### torch.linalg.norm

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.norm](https://pytorch.org/docs/2.12/generated/torch.linalg.norm.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

## Solvers

### torch.linalg.lstsq

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.lstsq](https://pytorch.org/docs/2.12/generated/torch.linalg.lstsq.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 可能回退至CPU执行

</div>

### torch.linalg.solve_triangular

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.solve_triangular](https://pytorch.org/docs/2.12/generated/torch.linalg.solve_triangular.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32，fp64，complex64，complex128

</div>

## Matrix Products

### torch.linalg.matmul

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.matmul](https://pytorch.org/docs/2.12/generated/torch.linalg.matmul.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp16，fp32
- 输入最大支持6维

</div>

## Experimental Functions

### torch.linalg.ldl_factor

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.ldl_factor](https://pytorch.org/docs/2.12/generated/torch.linalg.ldl_factor.html)

**是否支持**：否

</div>
