# torch.linalg

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 本支持清单的范围以[PyTorch社区文档](https://pytorch.org/docs/2.13/)为准，社区文档中未列出的接口不在本清单范围内；已在社区文档中存在但未收录于本清单的API尚未验证，请谨慎使用，我们将在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/linalg.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Matrix Properties](#matrix-properties)
- [Decompositions](#decompositions)
- [Solvers](#solvers)
- [Inverses](#inverses)
- [Matrix Functions](#matrix-functions)
- [Matrix Products](#matrix-products)
- [Tensor Operations](#tensor-operations)
- [Misc](#misc)
- [Experimental Functions](#experimental-functions)

</div>

<div style="display:none;">

## &#8203;torch.linalg

</div>

## Matrix Properties

### torch.linalg.norm

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.norm](https://pytorch.org/docs/2.13/generated/torch.linalg.norm.html)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id3 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.linalg.vector\_norm

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.vector\_norm](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.vector_norm.html#torch.linalg.vector_norm)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id66 -->

</div>

### torch.linalg.matrix_norm

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.matrix_norm](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.matrix_norm.html#torch.linalg.matrix_norm)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id45 -->

</div>

### torch.linalg.det

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.det](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.det.html#torch.linalg.det)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id72 -->

</div>

### torch.linalg.cond

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.cond](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.cond.html#torch.linalg.cond)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id78 -->

</div>

## Decompositions

### torch.linalg.cholesky

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.cholesky](https://pytorch.org/docs/2.13/generated/torch.linalg.cholesky.html)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id6 -->

**限制与说明**：`A`仅支持fp32

</div>

### torch.linalg.qr

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.qr](https://pytorch.org/docs/2.13/generated/torch.linalg.qr.html)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

**限制与说明**：`A`仅支持fp32，fp64，complex64，complex128

</div>

### torch.linalg.lu

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.lu](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.lu.html#torch.linalg.lu)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id39 -->

</div>

### torch.linalg.lu\_factor

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.lu\_factor](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.lu_factor.html#torch.linalg.lu_factor)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id81 -->

</div>

### torch.linalg.eig

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.eig](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.eig.html#torch.linalg.eig)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id84 -->

</div>

### torch.linalg.eigvals

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.eigvals](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.eigvals.html#torch.linalg.eigvals)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id87 -->

</div>

### torch.linalg.eigh

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.eigh](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.eigh.html#torch.linalg.eigh)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id27 -->

</div>

### torch.linalg.eigvalsh

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.eigvalsh](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.eigvalsh.html#torch.linalg.eigvalsh)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id90 -->

</div>

### torch.linalg.svd

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.svd](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.svd.html#torch.linalg.svd)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id69 -->

</div>

### torch.linalg.svdvals

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.svdvals](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.svdvals.html#torch.linalg.svdvals)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id51 -->

</div>

## Solvers

### torch.linalg.solve

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.solve](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.solve.html#torch.linalg.solve)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id93 -->

</div>

### torch.linalg.solve_triangular

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.solve_triangular](https://pytorch.org/docs/2.13/generated/torch.linalg.solve_triangular.html)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id15 -->

**限制与说明**：`A`、`B`仅支持fp32，fp64，complex64，complex128

</div>

### torch.linalg.lu\_solve

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.lu\_solve](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.lu_solve.html#torch.linalg.lu_solve)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id96 -->

</div>

### torch.linalg.lstsq

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.lstsq](https://pytorch.org/docs/2.13/generated/torch.linalg.lstsq.html)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id12 -->

**限制与说明**： 可能回退至CPU执行

</div>

## Inverses

### torch.linalg.inv

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.inv](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.inv.html#torch.linalg.inv)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id30 -->

</div>

### torch.linalg.pinv

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.pinv](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.pinv.html#torch.linalg.pinv)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id48 -->

</div>

## Matrix Functions

### torch.linalg.matrix_exp

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.matrix_exp](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.matrix_exp.html#torch.linalg.matrix_exp)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id42 -->

</div>

### torch.linalg.matrix\_power

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.matrix\_power](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.matrix_power.html#torch.linalg.matrix_power)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id99 -->

</div>

## Matrix Products

### torch.linalg.cross

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.cross](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.cross.html#torch.linalg.cross)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id24 -->

</div>

### torch.linalg.matmul

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.matmul](https://pytorch.org/docs/2.13/generated/torch.linalg.matmul.html)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id18 -->

**限制与说明**：

- `input`仅支持fp16，fp32
- 输入最大支持6维

</div>

### torch.linalg.vecdot

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.vecdot](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.vecdot.html#torch.linalg.vecdot)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id63 -->

</div>

### torch.linalg.multi\_dot

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.multi\_dot](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.multi_dot.html#torch.linalg.multi_dot)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id102 -->

</div>

### torch.linalg.householder\_product

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.householder\_product](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.householder_product.html#torch.linalg.householder_product)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id105 -->

</div>

## Tensor Operations

### torch.linalg.tensorinv

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.tensorinv](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.tensorinv.html#torch.linalg.tensorinv)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id54 -->

</div>

### torch.linalg.tensorsolve

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.tensorsolve](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.tensorsolve.html#torch.linalg.tensorsolve)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id57 -->

</div>

## Misc

### torch.linalg.vander

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.vander](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.vander.html#torch.linalg.vander)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id60 -->

</div>

## Experimental Functions

### torch.linalg.cholesky\_ex

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.cholesky\_ex](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.cholesky_ex.html#torch.linalg.cholesky_ex)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id108 -->

</div>

### torch.linalg.inv_ex

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.inv_ex](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.inv_ex.html#torch.linalg.inv_ex)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id33 -->

</div>

### torch.linalg.solve\_ex

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.solve\_ex](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.solve_ex.html#torch.linalg.solve_ex)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id111 -->

</div>

### torch.linalg.lu\_factor\_ex

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.lu\_factor\_ex](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.lu_factor_ex.html#torch.linalg.lu_factor_ex)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id114 -->

</div>

### torch.linalg.ldl_factor

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.ldl_factor](https://pytorch.org/docs/2.13/generated/torch.linalg.ldl_factor.html)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id21 -->

</div>

### torch.linalg.ldl\_factor\_ex

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.ldl\_factor\_ex](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.ldl_factor_ex.html#torch.linalg.ldl_factor_ex)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id117 -->

</div>

### torch.linalg.ldl_solve

<div style="margin-left: 2em">

**原生文档**：[torch.linalg.ldl_solve](https://docs.pytorch.org/docs/2.13/generated/torch.linalg.ldl_solve.html#torch.linalg.ldl_solve)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id36 -->

</div>
