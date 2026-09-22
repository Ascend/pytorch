# torch.nn.functional

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.11/nn.functional.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Convolution functions](#convolution-functions)
- [Pooling functions](#pooling-functions)
- [Attention Mechanisms](#attention-mechanisms)
- [Non-linear activation functions](#non-linear-activation-functions)
- [Linear functions](#linear-functions)
- [Dropout functions](#dropout-functions)
- [Sparse functions](#sparse-functions)
- [Distance functions](#distance-functions)
- [Loss functions](#loss-functions)
- [Vision functions](#vision-functions)
- [Low-Precision functions](#low-precision-functions)

</div>

<div style="display:none;">

## &#8203;torch.nn.functional

</div>

### torch.nn.parallel.data_parallel

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.data_parallel](https://pytorch.org/docs/2.11/generated/torch.nn.functional.torch.nn.parallel.data_parallel.html#torch.nn.parallel.data_parallel)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id3 -->

</div>

## Convolution functions

### torch.nn.functional.conv1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv1d.html)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT</term>：支持
<!-- end id6 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.conv2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv2d.html)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT</term>：支持
<!-- end id9 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.conv3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv3d.html)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id12 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，complex64

</div>

### torch.nn.functional.conv_transpose1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv_transpose1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv_transpose1d.html)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT</term>：支持
<!-- end id15 -->

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.functional.conv_transpose2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv_transpose2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv_transpose2d.html)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT</term>：支持
<!-- end id18 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.conv_transpose3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv_transpose3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv_transpose3d.html)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id21 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.unfold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.unfold](https://pytorch.org/docs/2.11/generated/torch.nn.functional.unfold.html)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT</term>：支持
<!-- end id24 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.fold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.fold](https://pytorch.org/docs/2.11/generated/torch.nn.functional.fold.html)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT</term>：支持
<!-- end id27 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

## Pooling functions

### torch.nn.functional.avg_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.avg_pool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.avg_pool1d.html)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT</term>：支持
<!-- end id30 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.avg_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.avg_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.avg_pool2d.html)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT</term>：支持
<!-- end id33 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.avg_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.avg_pool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.avg_pool3d.html)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id36 -->

</div>

### torch.nn.functional.max_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_pool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_pool1d.html)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id39 -->

**限制与说明**：

- `input`仅支持fp16，fp32
- `dilation`仅支持1
- 通过设置`torch_npu.npu.use_compatible_impl(True)`，保证与PyTorch同名接口在内存一致性上对齐，例如：

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

- `return_indices`为True时，返回的`argmax`的数据类型为int32

</div>

### torch.nn.functional.max_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_pool2d.html)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT</term>：支持
<!-- end id42 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- `dilation`的值目前在NPU上仅支持设置为1或(1,1)
- 通过设置`torch_npu.npu.use_compatible_impl(True)`，保证与PyTorch同名接口在内存一致性上对齐，例如：

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

- `return_indices`为True时，返回的`argmax`的数据类型为int32

</div>

### torch.nn.functional.max_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_pool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_pool3d.html)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT</term>：支持
<!-- end id45 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- `dilation`的值目前在NPU上仅支持设置为1或(1,1,1)
- `return_indices`为True时，返回的`argmax`的数据类型为int32

</div>

### torch.nn.functional.max_unpool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_unpool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_unpool1d.html)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id48 -->

**限制与说明**：`input`仅支持fp16，fp32，fp64，uint8，int8，int32，int64

</div>

### torch.nn.functional.max_unpool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_unpool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_unpool2d.html)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id51 -->

**限制与说明**：

- `input`仅支持fp16，fp32，fp64，uint8，int8，int32，int64
- `jit_compile=False`即二进制模式时，`output_size`的乘积需要大于等于`input`的H、W的乘积

</div>

### torch.nn.functional.max_unpool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_unpool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_unpool3d.html)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id54 -->

</div>

### torch.nn.functional.lp_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.lp_pool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.lp_pool1d.html)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id57 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.lp_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.lp_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.lp_pool2d.html)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id60 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.adaptive_max_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_max_pool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_max_pool1d.html)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id63 -->

</div>

### torch.nn.functional.adaptive_max_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_max_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_max_pool2d.html)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id66 -->

</div>

### torch.nn.functional.adaptive_max_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_max_pool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_max_pool3d.html)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id69 -->

**限制与说明**： `input`仅支持fp32，fp64

</div>

### torch.nn.functional.adaptive_avg_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_avg_pool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_avg_pool1d.html)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id72 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.adaptive_avg_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_avg_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_avg_pool2d.html)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id75 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.adaptive_avg_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_avg_pool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_avg_pool3d.html)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id78 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.fractional_max_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.fractional_max_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.fractional_max_pool2d.html)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT</term>：支持
<!-- end id81 -->

**限制与说明**： 可能回退至CPU执行

</div>

### torch.nn.functional.fractional_max_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.fractional_max_pool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.fractional_max_pool3d.html)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id84 -->

</div>

## Attention Mechanisms

### torch.nn.functional.scaled_dot_product_attention

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id87 -->

**限制与说明**：

- `query`、`key`、`value`仅支持bf16、fp16和fp32。所有参数输入均符合以下约束
- **约束**：
- 所有参数输入符合规格：
  - 输入`query`、`key`、`value`的N：batch size，当前只支持[N，head_num, S(L), E(Ev)]的排布方式，取值范围1~2K
  - 输入`query`的head num和`key`/`value`的head num必须成比例关系，即Nq/Nkv必须是非0整数，取值范围1~256
  - 输入`query`的L：Target sequence length，取值范围1~512K
  - 输入`key`、`value`的S：Source sequence length，取值范围1~512K
- 输入`query`、`key`、`value`的E：Embedding dimension of the query and key，取值范围1~512
  - 输入`value`的Ev：Embedding dimension of the value，必须与E相等
  - 输入`attn_mask`：当前支持[N, 1, L, S]、[N, head_num, L, S]、[1, 1, L, S]、[L, S]，以及可广播到[N, head_num, L, S]的bool类型mask，例如[L, 1]、[1, S]、[1, 1]等排布方式
  - 在开启`is_causal`计算时，`attn_mask`必须为None；不开启`is_causal`时，若`attn_mask`输入有效数据，输入数据类型必须是bool类型
  - 输入`query`、`key`、`value`的数据类型为bf16、fp16、fp32
  - 通过设置`torch_npu.npu.use_compatible_impl(True)`，支持按SDPA后端选择上下文指定MATH后端，例如：

    ```python
    import torch_npu
    torch_npu.npu.use_compatible_impl(True)
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    ```

- 与原接口除了规格限制之外的差异点：
  - NPU的随机算法部分用DSA硬件实现，算法在DSA引擎固化与GPU算法实现存在差异，导致dropout功能和GPU结果不一致
  - 当前接口支持输入`query`的head num和`key`/`value`的head num不等长，而原生PyTorch接口不支持

</div>

## Non-linear activation functions

### torch.nn.functional.threshold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.threshold](https://pytorch.org/docs/2.11/generated/torch.nn.functional.threshold.html)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT</term>：支持
<!-- end id90 -->

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64
- 当`input`为超过16,777,216（即2<sup>24</sup>）的int32类型时，精度会有损失

</div>

### torch.nn.functional.threshold_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.threshold_](https://pytorch.org/docs/2.11/generated/torch.nn.functional.threshold_.html)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT</term>：支持
<!-- end id93 -->

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64
- 当`input`为超过16,777,216（即2<sup>24</sup>）的int32类型时，精度会有损失

</div>

### torch.nn.functional.relu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.relu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.relu.html)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT</term>：支持
<!-- end id96 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.nn.functional.relu_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.relu_](https://pytorch.org/docs/2.11/generated/torch.nn.functional.relu_.html)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT</term>：支持
<!-- end id99 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.nn.functional.hardtanh

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardtanh](https://pytorch.org/docs/2.11/generated/torch.nn.functional.hardtanh.html)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT</term>：支持
<!-- end id102 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.hardtanh_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardtanh_](https://pytorch.org/docs/2.11/generated/torch.nn.functional.hardtanh_.html)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT</term>：支持
<!-- end id105 -->

**限制与说明**：`input`仅支持fp16，fp32，int8，int16，int32，int64

</div>

### torch.nn.functional.hardswish

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardswish](https://pytorch.org/docs/2.11/generated/torch.nn.functional.hardswish.html)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT</term>：支持
<!-- end id108 -->

**限制与说明**：

- `input`仅支持fp16，fp32
- 可能回退至CPU执行

</div>

### torch.nn.functional.relu6

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.relu6](https://pytorch.org/docs/2.11/generated/torch.nn.functional.relu6.html)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT</term>：支持
<!-- end id111 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.elu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.elu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.elu.html)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT</term>：支持
<!-- end id114 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.elu_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.elu_](https://pytorch.org/docs/2.11/generated/torch.nn.functional.elu_.html)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT</term>：支持
<!-- end id117 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.selu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.selu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.selu.html)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT</term>：支持
<!-- end id120 -->

**限制与说明**：

- `input`仅支持fp16，fp32
- fp16反向场景下，与GPU相比存在精度累加误差，可以通过如下方式进行规避：
- 将正向调用的`torch.nn.functional.selu`替换成`torch.ops.aten.elu`，例如：将`torch.nn.functional.selu(input_x)`替换为`torch.ops.aten.elu(input_x, 1.6732632423543772848170429916717, 1.0507009873554804934193349852946)`

</div>

### torch.nn.functional.celu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.celu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.celu.html)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id123 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.leaky_relu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.leaky_relu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.leaky_relu.html)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id126 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.leaky_relu_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.leaky_relu_](https://pytorch.org/docs/2.11/generated/torch.nn.functional.leaky_relu_.html)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT</term>：支持
<!-- end id129 -->

**限制与说明**：`input`仅支持fp16，fp32，fp64

</div>

### torch.nn.functional.prelu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.prelu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.prelu.html)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT</term>：支持
<!-- end id132 -->

**限制与说明**：

- `input`仅支持fp16，fp32
- `input`仅支持1-8维

</div>

### torch.nn.functional.rrelu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.rrelu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.rrelu.html)

**产品支持情况**：

<!-- npu="910b" id133 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id135 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.glu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.glu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.glu.html)

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT</term>：支持
<!-- end id138 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.gelu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.gelu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.gelu.html)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT</term>：支持
<!-- end id141 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- `approximate`参数仅支持设置为`tanh`

</div>

### torch.nn.functional.logsigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.logsigmoid](https://pytorch.org/docs/2.11/generated/torch.nn.functional.logsigmoid.html)

**产品支持情况**：

<!-- npu="910b" id142 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT</term>：支持
<!-- end id144 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.hardshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardshrink](https://pytorch.org/docs/2.11/generated/torch.nn.functional.hardshrink.html)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT</term>：支持
<!-- end id147 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.softsign

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softsign](https://pytorch.org/docs/2.11/generated/torch.nn.functional.softsign.html)

**产品支持情况**：

<!-- npu="910b" id148 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT</term>：支持
<!-- end id150 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.softplus

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softplus](https://pytorch.org/docs/2.11/generated/torch.nn.functional.softplus.html)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id153 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softmax](https://pytorch.org/docs/2.11/generated/torch.nn.functional.softmax.html)

**产品支持情况**：

<!-- npu="910b" id154 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT</term>：支持
<!-- end id156 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.softshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softshrink](https://pytorch.org/docs/2.11/generated/torch.nn.functional.softshrink.html)

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id159 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.gumbel_softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.gumbel_softmax](https://pytorch.org/docs/2.11/generated/torch.nn.functional.gumbel_softmax.html)

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id162 -->

</div>

### torch.nn.functional.log_softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.log_softmax](https://pytorch.org/docs/2.11/generated/torch.nn.functional.log_softmax.html)

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT</term>：支持
<!-- end id165 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.tanh

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.tanh](https://pytorch.org/docs/2.11/generated/torch.nn.functional.tanh.html)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT</term>：支持
<!-- end id168 -->

**限制与说明**：`input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.nn.functional.sigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.sigmoid](https://pytorch.org/docs/2.11/generated/torch.nn.functional.sigmoid.html)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT</term>：支持
<!-- end id171 -->

**限制与说明**：`input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.hardsigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardsigmoid](https://pytorch.org/docs/2.11/generated/torch.nn.functional.hardsigmoid.html)

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id174 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.silu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.silu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.silu.html)

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT</term>：支持
<!-- end id177 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.mish

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.mish](https://pytorch.org/docs/2.11/generated/torch.nn.functional.mish.html)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT</term>：支持
<!-- end id180 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.batch_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.batch_norm](https://pytorch.org/docs/2.11/generated/torch.nn.functional.batch_norm.html)

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT</term>：支持
<!-- end id183 -->

**限制与说明**：

- `input`仅支持fp16，fp32
- `weight`和`bias`仅支持一维场景
- `bias`的shape为1维，长度与`input`入参中channel轴的长度相等

</div>

### torch.nn.functional.group_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.group_norm](https://pytorch.org/docs/2.11/generated/torch.nn.functional.group_norm.html)

**产品支持情况**：

<!-- npu="910b" id184 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id186 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 该API仅支持2维及以上的输入`input`
- `eps`参数需大于0

</div>

### torch.nn.functional.layer_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.layer_norm](https://pytorch.org/docs/2.11/generated/torch.nn.functional.layer_norm.html)

**产品支持情况**：

<!-- npu="910b" id187 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT</term>：支持
<!-- end id189 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.normalize

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.normalize](https://pytorch.org/docs/2.11/generated/torch.nn.functional.normalize.html)

**产品支持情况**：

<!-- npu="910b" id190 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT</term>：支持
<!-- end id192 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64

</div>

## Linear functions

### torch.nn.functional.linear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.linear](https://pytorch.org/docs/2.11/generated/torch.nn.functional.linear.html)

**产品支持情况**：

<!-- npu="910b" id193 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id195 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.bilinear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.bilinear](https://pytorch.org/docs/2.11/generated/torch.nn.functional.bilinear.html)

**产品支持情况**：

<!-- npu="910b" id196 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id198 -->

**限制与说明**：`input1`、`input2`、`weight`、`bias`仅支持bf16，fp16，fp32

</div>

## Dropout functions

### torch.nn.functional.dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.dropout](https://pytorch.org/docs/2.11/generated/torch.nn.functional.dropout.html)

**产品支持情况**：

<!-- npu="910b" id199 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT</term>：支持
<!-- end id201 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.alpha_dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.alpha_dropout](https://pytorch.org/docs/2.11/generated/torch.nn.functional.alpha_dropout.html)

**产品支持情况**：

<!-- npu="910b" id202 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT</term>：支持
<!-- end id204 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.feature_alpha_dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.feature_alpha_dropout](https://pytorch.org/docs/2.11/generated/torch.nn.functional.feature_alpha_dropout.html)

**产品支持情况**：

<!-- npu="910b" id205 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT</term>：支持
<!-- end id207 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.dropout2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.dropout2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.dropout2d.html)

**产品支持情况**：

<!-- npu="910b" id208 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT</term>：支持
<!-- end id210 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

<!-- npu="950" id211 -->
- <term>Ascend 950DT</term>：不支持complex64，complex128
<!-- end id211 -->

</div>

## Sparse functions

### torch.nn.functional.embedding

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.embedding](https://pytorch.org/docs/2.11/generated/torch.nn.functional.embedding.html)

**产品支持情况**：

<!-- npu="910b" id212 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id212 -->
<!-- npu="A3" id213 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id213 -->
<!-- npu="950" id214 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id214 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，int32，int64
- 属性`max_norm`仅支持非负值

</div>

### torch.nn.functional.embedding_bag

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.embedding_bag](https://pytorch.org/docs/2.11/generated/torch.nn.functional.embedding_bag.html)

**产品支持情况**：

<!-- npu="910b" id215 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id215 -->
<!-- npu="A3" id216 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id216 -->
<!-- npu="950" id217 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id217 -->

</div>

### torch.nn.functional.one_hot

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.one_hot](https://pytorch.org/docs/2.11/generated/torch.nn.functional.one_hot.html)

**产品支持情况**：

<!-- npu="910b" id218 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="A3" id219 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id219 -->
<!-- npu="950" id220 -->
- <term>Ascend 950DT</term>：支持
<!-- end id220 -->

**限制与说明**： `input`仅支持int32，int64

</div>

## Distance functions

### torch.nn.functional.cosine_similarity

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.cosine_similarity](https://pytorch.org/docs/2.11/generated/torch.nn.functional.cosine_similarity.html)

**产品支持情况**：

<!-- npu="910b" id221 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="A3" id222 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id222 -->
<!-- npu="950" id223 -->
- <term>Ascend 950DT</term>：支持
<!-- end id223 -->

**限制与说明**：`x1`、`x2`仅支持fp16，fp32

</div>

### torch.nn.functional.pdist

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pdist](https://pytorch.org/docs/2.11/generated/torch.nn.functional.pdist.html)

**产品支持情况**：

<!-- npu="910b" id224 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id224 -->
<!-- npu="A3" id225 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id225 -->
<!-- npu="950" id226 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id226 -->

</div>

## Loss functions

### torch.nn.functional.binary_cross_entropy

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.binary_cross_entropy](https://pytorch.org/docs/2.11/generated/torch.nn.functional.binary_cross_entropy.html)

**产品支持情况**：

<!-- npu="910b" id227 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id227 -->
<!-- npu="A3" id228 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id228 -->
<!-- npu="950" id229 -->
- <term>Ascend 950DT</term>：支持
<!-- end id229 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.binary_cross_entropy_with_logits

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.binary_cross_entropy_with_logits](https://pytorch.org/docs/2.11/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)

**产品支持情况**：

<!-- npu="910b" id230 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id230 -->
<!-- npu="A3" id231 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id231 -->
<!-- npu="950" id232 -->
- <term>Ascend 950DT</term>：支持
<!-- end id232 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.poisson_nll_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.poisson_nll_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.poisson_nll_loss.html)

**产品支持情况**：

<!-- npu="910b" id233 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id233 -->
<!-- npu="A3" id234 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id234 -->
<!-- npu="950" id235 -->
- <term>Ascend 950DT</term>：支持
<!-- end id235 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，int64
- 可能回退至CPU执行

</div>

### torch.nn.functional.cross_entropy

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.cross_entropy](https://pytorch.org/docs/2.11/generated/torch.nn.functional.cross_entropy.html)

**产品支持情况**：

<!-- npu="910b" id236 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id236 -->
<!-- npu="A3" id237 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id237 -->
<!-- npu="950" id238 -->
- <term>Ascend 950DT</term>：支持
<!-- end id238 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.ctc_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.ctc_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.ctc_loss.html)

**产品支持情况**：

<!-- npu="910b" id239 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id239 -->
<!-- npu="A3" id240 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id240 -->
<!-- npu="950" id241 -->
- <term>Ascend 950DT</term>：支持
<!-- end id241 -->

**限制与说明**：

- `log_probs`仅支持fp32，fp64
- 目标序列的长度不支持0，即属性`target_lengths`的取值不能包含0

</div>

### torch.nn.functional.gaussian_nll_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.gaussian_nll_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.gaussian_nll_loss.html)

**产品支持情况**：

<!-- npu="910b" id242 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id242 -->
<!-- npu="A3" id243 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id243 -->
<!-- npu="950" id244 -->
- <term>Ascend 950DT</term>：支持
<!-- end id244 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.kl_div

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.kl_div](https://pytorch.org/docs/2.11/generated/torch.nn.functional.kl_div.html)

**产品支持情况**：

<!-- npu="910b" id245 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id245 -->
<!-- npu="A3" id246 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id246 -->
<!-- npu="950" id247 -->
- <term>Ascend 950DT</term>：支持
<!-- end id247 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 当前`log_target`参数仅支持False
- 当前`target`不支持求导

</div>

### torch.nn.functional.l1_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.l1_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.l1_loss.html)

**产品支持情况**：

<!-- npu="910b" id248 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id248 -->
<!-- npu="A3" id249 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id249 -->
<!-- npu="950" id250 -->
- <term>Ascend 950DT</term>：支持
<!-- end id250 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，int64

</div>

### torch.nn.functional.mse_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.mse_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.mse_loss.html)

**产品支持情况**：

<!-- npu="910b" id251 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id251 -->
<!-- npu="A3" id252 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id252 -->
<!-- npu="950" id253 -->
- <term>Ascend 950DT</term>：支持
<!-- end id253 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.margin_ranking_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.margin_ranking_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.margin_ranking_loss.html)

**产品支持情况**：

<!-- npu="910b" id254 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id254 -->
<!-- npu="A3" id255 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id255 -->
<!-- npu="950" id256 -->
- <term>Ascend 950DT</term>：支持
<!-- end id256 -->

**限制与说明**：`input1`、`input2`、`target`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.multilabel_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.multilabel_margin_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.multilabel_margin_loss.html)

**产品支持情况**：

<!-- npu="910b" id257 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id257 -->
<!-- npu="A3" id258 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id258 -->
<!-- npu="950" id259 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id259 -->

**限制与说明**：

- `input`仅支持fp16，fp32
- 输入`tensor`的元素个数不能超过10万

</div>

### torch.nn.functional.multilabel_soft_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.multilabel_soft_margin_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.multilabel_soft_margin_loss.html)

**产品支持情况**：

<!-- npu="910b" id260 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id260 -->
<!-- npu="A3" id261 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id261 -->
<!-- npu="950" id262 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id262 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.nll_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.nll_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.nll_loss.html)

**产品支持情况**：

<!-- npu="910b" id263 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id263 -->
<!-- npu="A3" id264 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id264 -->
<!-- npu="950" id265 -->
- <term>Ascend 950DT</term>：支持
<!-- end id265 -->

**限制与说明**：

- `input`仅支持fp32
- `target`中的每个元素值应大于等于0且小于`input`的类别数

</div>

### torch.nn.functional.smooth_l1_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.smooth_l1_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.smooth_l1_loss.html)

**产品支持情况**：

<!-- npu="910b" id266 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id266 -->
<!-- npu="A3" id267 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id267 -->
<!-- npu="950" id268 -->
- <term>Ascend 950DT</term>：支持
<!-- end id268 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.soft_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.soft_margin_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.soft_margin_loss.html)

**产品支持情况**：

<!-- npu="910b" id269 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id269 -->
<!-- npu="A3" id270 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id270 -->
<!-- npu="950" id271 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id271 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，不支持double，complex64，complex128数据类型

</div>

### torch.nn.functional.triplet_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.triplet_margin_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.triplet_margin_loss.html)

**产品支持情况**：

<!-- npu="910b" id272 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id272 -->
<!-- npu="A3" id273 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id273 -->
<!-- npu="950" id274 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id274 -->

</div>

### torch.nn.functional.triplet_margin_with_distance_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.triplet_margin_with_distance_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.triplet_margin_with_distance_loss.html)

**产品支持情况**：

<!-- npu="910b" id275 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id275 -->
<!-- npu="A3" id276 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id276 -->
<!-- npu="950" id277 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id277 -->

</div>

## Vision functions

### torch.nn.functional.pixel_shuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pixel_shuffle](https://pytorch.org/docs/2.11/generated/torch.nn.functional.pixel_shuffle.html)

**产品支持情况**：

<!-- npu="910b" id278 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id278 -->
<!-- npu="A3" id279 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id279 -->
<!-- npu="950" id280 -->
- <term>Ascend 950DT</term>：支持
<!-- end id280 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.pixel_unshuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pixel_unshuffle](https://pytorch.org/docs/2.11/generated/torch.nn.functional.pixel_unshuffle.html)

**产品支持情况**：

<!-- npu="910b" id281 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id281 -->
<!-- npu="A3" id282 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id282 -->
<!-- npu="950" id283 -->
- <term>Ascend 950DT</term>：支持
<!-- end id283 -->

**限制与说明**：`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.nn.functional.pad

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pad](https://pytorch.org/docs/2.11/generated/torch.nn.functional.pad.html)

**产品支持情况**：

<!-- npu="910b" id284 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id284 -->
<!-- npu="A3" id285 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id285 -->
<!-- npu="950" id286 -->
- <term>Ascend 950DT</term>：支持
<!-- end id286 -->

**限制与说明**：

- 属性`mode`为constant时，`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 属性`mode`非constant时，`input`仅支持fp16，fp32，fp64
- 在输入`x`为六维以上时可能会出现性能下降

</div>

### torch.nn.functional.interpolate

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.interpolate](https://pytorch.org/docs/2.11/generated/torch.nn.functional.interpolate.html)

**产品支持情况**：

<!-- npu="910b" id287 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id287 -->
<!-- npu="A3" id288 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id288 -->
<!-- npu="950" id289 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id289 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64
- 支持nearest、linear、bilinear、bicubic、trilinear、area
- 不支持`scale_factor`

</div>

### torch.nn.functional.upsample

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.upsample](https://pytorch.org/docs/2.11/generated/torch.nn.functional.upsample.html)

**产品支持情况**：

<!-- npu="910b" id290 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id290 -->
<!-- npu="A3" id291 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id291 -->
<!-- npu="950" id292 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id292 -->

**限制与说明**：

- `input`仅支持fp16，fp32，fp64
- 只支持`mode` = nearest，例如：

  ```python
  out = torch.nn.functional.upsample(x, size=(256, 256), mode='nearest')
  ```

</div>

### torch.nn.functional.upsample_nearest

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.upsample_nearest](https://pytorch.org/docs/2.11/generated/torch.nn.functional.upsample_nearest.html)

**产品支持情况**：

<!-- npu="910b" id293 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id293 -->
<!-- npu="A3" id294 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id294 -->
<!-- npu="950" id295 -->
- <term>Ascend 950DT</term>：支持
<!-- end id295 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- `input`只支持3-5维

</div>

### torch.nn.functional.upsample_bilinear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.upsample_bilinear](https://pytorch.org/docs/2.11/generated/torch.nn.functional.upsample_bilinear.html)

**产品支持情况**：

<!-- npu="910b" id296 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id296 -->
<!-- npu="A3" id297 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id297 -->
<!-- npu="950" id298 -->
- <term>Ascend 950DT</term>：支持
<!-- end id298 -->

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.grid_sample

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.grid_sample](https://pytorch.org/docs/2.11/generated/torch.nn.functional.grid_sample.html)

**产品支持情况**：

<!-- npu="910b" id299 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id299 -->
<!-- npu="A3" id300 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id300 -->
<!-- npu="950" id301 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id301 -->

**限制与说明**：`input`仅支持fp16，fp32，fp64

</div>

### torch.nn.functional.affine_grid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.affine_grid](https://pytorch.org/docs/2.11/generated/torch.nn.functional.affine_grid.html)

**产品支持情况**：

<!-- npu="910b" id302 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id302 -->
<!-- npu="A3" id303 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id303 -->
<!-- npu="950" id304 -->
- <term>Ascend 950DT</term>：支持
<!-- end id304 -->

**限制与说明**：`theta`仅支持fp16，fp32

</div>

## Low-Precision functions

### torch.nn.functional.scaled_mm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.scaled_mm](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_mm.html)

**产品支持情况**：

<!-- npu="910b" id305 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id305 -->
<!-- npu="A3" id306 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id306 -->
<!-- npu="950" id307 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id307 -->

**限制与说明**： `input`仅支持fp8模式下ScalingType为tensorwise，rowwise和BlockWise1x128，mxfp8模式下ScalingType为BlockWise1x32的排布，mxfp8遵循[aclnnQuantMatmulV5](https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md)要求（`scale_a`和`scale_b`详见约束说明）

</div>

### torch.nn.functional.scaled_grouped_mm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.scaled_grouped_mm](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_grouped_mm.html)

**产品支持情况**：

<!-- npu="910b" id308 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id308 -->
<!-- npu="A3" id309 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id309 -->
<!-- npu="950" id310 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id310 -->

**限制与说明**： `input`仅支持fp8模式下ScalingType为rowwise，mxfp8模式的排布，mxfp8遵循[aclnnGroupedMatmulV5](https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)要求（`scale_a`和`scale_b`详见约束说明）

</div>
