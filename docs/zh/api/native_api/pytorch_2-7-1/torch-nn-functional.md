# torch.nn.functional

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/nn.functional.html)。

## 目录

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

### torch.nn.parallel.data_parallel

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.data_parallel](https://pytorch.org/docs/2.7/generated/torch.nn.functional.torch.nn.parallel.data_parallel.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## Convolution functions

### torch.nn.functional.conv1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv1d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.conv1d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.conv2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv2d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.conv2d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.conv3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv3d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.conv3d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，complex64

</div>

### torch.nn.functional.conv_transpose1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv_transpose1d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.conv_transpose1d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持fp32

</div>

### torch.nn.functional.conv_transpose2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv_transpose2d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.conv_transpose2d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.conv_transpose3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv_transpose3d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.conv_transpose3d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.unfold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.unfold](https://pytorch.org/docs/2.7/generated/torch.nn.functional.unfold.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.fold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.fold](https://pytorch.org/docs/2.7/generated/torch.nn.functional.fold.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

## Pooling functions

### torch.nn.functional.avg_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.avg_pool1d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.avg_pool1d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.avg_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.avg_pool2d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.avg_pool2d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.avg_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.avg_pool3d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.avg_pool3d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.nn.functional.max_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_pool1d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.max_pool1d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

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

**原生文档**：[torch.nn.functional.max_pool2d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.max_pool2d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

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

**原生文档**：[torch.nn.functional.max_pool3d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.max_pool3d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- `dilation`的值目前在NPU上仅支持设置为1或(1,1,1)
- `return_indices`为True时，返回的`argmax`的数据类型为int32

</div>

### torch.nn.functional.max_unpool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_unpool1d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.max_unpool1d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp16，fp32，fp64，uint8，int8，int32，int64

</div>

### torch.nn.functional.max_unpool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_unpool2d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.max_unpool2d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：

- `input`仅支持fp16，fp32，fp64，uint8，int8，int32，int64
- `jit_compile=False`即二进制模式时，`output_size`的乘积需要大于等于`input`的H、W的乘积

</div>

### torch.nn.functional.max_unpool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_unpool3d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.max_unpool3d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.nn.functional.lp_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.lp_pool1d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.lp_pool1d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.lp_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.lp_pool2d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.lp_pool2d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.adaptive_max_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_max_pool1d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.adaptive_max_pool1d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.nn.functional.adaptive_max_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_max_pool2d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.adaptive_max_pool2d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.nn.functional.adaptive_max_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_max_pool3d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.adaptive_max_pool3d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32，fp64

</div>

### torch.nn.functional.adaptive_avg_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_avg_pool1d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.adaptive_avg_pool1d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.adaptive_avg_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_avg_pool2d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.adaptive_avg_pool2d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.adaptive_avg_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_avg_pool3d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.adaptive_avg_pool3d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.fractional_max_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.fractional_max_pool2d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.fractional_max_pool2d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

### torch.nn.functional.fractional_max_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.fractional_max_pool3d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.fractional_max_pool3d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## Attention Mechanisms

### torch.nn.functional.scaled_dot_product_attention

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

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

- 与原接口除了规格限制之外差异点：
  - NPU的随机算法部分用DSA硬件实现，算法在DSA引擎固化与GPU算法实现存在差异，导致dropout功能和GPU结果不一致
  - 当前接口支持输入`query`的head num和`key`/`value`的head num不等长，而原生PyTorch接口不支持

</div>

## Non-linear activation functions

### torch.nn.functional.threshold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.threshold](https://pytorch.org/docs/2.7/generated/torch.nn.functional.threshold.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64
- 当`input`为超过16,777,216（即2<sup>24</sup>）的int32类型时，精度会有损失

</div>

### torch.nn.functional.threshold_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.threshold_](https://pytorch.org/docs/2.7/generated/torch.nn.functional.threshold_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64
- 当`input`为超过16,777,216（即2<sup>24</sup>）的int32类型时，精度会有损失

</div>

### torch.nn.functional.relu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.relu](https://pytorch.org/docs/2.7/generated/torch.nn.functional.relu.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.nn.functional.relu_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.relu_](https://pytorch.org/docs/2.7/generated/torch.nn.functional.relu_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.nn.functional.hardtanh

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardtanh](https://pytorch.org/docs/2.7/generated/torch.nn.functional.hardtanh.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.hardtanh_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardtanh_](https://pytorch.org/docs/2.7/generated/torch.nn.functional.hardtanh_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32，int8，int16，int32，int64

</div>

### torch.nn.functional.hardswish

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardswish](https://pytorch.org/docs/2.7/generated/torch.nn.functional.hardswish.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持fp16，fp32
- 可能回退至CPU执行

</div>

### torch.nn.functional.relu6

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.relu6](https://pytorch.org/docs/2.7/generated/torch.nn.functional.relu6.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.elu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.elu](https://pytorch.org/docs/2.7/generated/torch.nn.functional.elu.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.elu_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.elu_](https://pytorch.org/docs/2.7/generated/torch.nn.functional.elu_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.selu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.selu](https://pytorch.org/docs/2.7/generated/torch.nn.functional.selu.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持fp16，fp32
- fp16反向场景下，与GPU相比存在精度累加误差，可以通过如下方式进行规避：
- 将正向调用的`torch.nn.functional.selu`替换成`torch.ops.aten.elu`，例如：将`torch.nn.functional.selu(input_x)`替换为`torch.ops.aten.elu(input_x, 1.6732632423543772848170429916717, 1.0507009873554804934193349852946)`

</div>

### torch.nn.functional.celu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.celu](https://pytorch.org/docs/2.7/generated/torch.nn.functional.celu.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.leaky_relu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.leaky_relu](https://pytorch.org/docs/2.7/generated/torch.nn.functional.leaky_relu.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.leaky_relu_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.leaky_relu_](https://pytorch.org/docs/2.7/generated/torch.nn.functional.leaky_relu_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32，fp64

</div>

### torch.nn.functional.prelu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.prelu](https://pytorch.org/docs/2.7/generated/torch.nn.functional.prelu.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持fp16，fp32
- `input`仅支持1-8维

</div>

### torch.nn.functional.rrelu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.rrelu](https://pytorch.org/docs/2.7/generated/torch.nn.functional.rrelu.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.glu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.glu](https://pytorch.org/docs/2.7/generated/torch.nn.functional.glu.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.gelu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.gelu](https://pytorch.org/docs/2.7/generated/torch.nn.functional.gelu.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- `approximate`参数仅支持设置为`tanh`

</div>

### torch.nn.functional.logsigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.logsigmoid](https://pytorch.org/docs/2.7/generated/torch.nn.functional.logsigmoid.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.hardshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardshrink](https://pytorch.org/docs/2.7/generated/torch.nn.functional.hardshrink.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.softsign

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softsign](https://pytorch.org/docs/2.7/generated/torch.nn.functional.softsign.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.softplus

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softplus](https://pytorch.org/docs/2.7/generated/torch.nn.functional.softplus.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softmax](https://pytorch.org/docs/2.7/generated/torch.nn.functional.softmax.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.softshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softshrink](https://pytorch.org/docs/2.7/generated/torch.nn.functional.softshrink.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.gumbel_softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.gumbel_softmax](https://pytorch.org/docs/2.7/generated/torch.nn.functional.gumbel_softmax.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.nn.functional.log_softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.log_softmax](https://pytorch.org/docs/2.7/generated/torch.nn.functional.log_softmax.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.tanh

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.tanh](https://pytorch.org/docs/2.7/generated/torch.nn.functional.tanh.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.nn.functional.sigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.sigmoid](https://pytorch.org/docs/2.7/generated/torch.nn.functional.sigmoid.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.hardsigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardsigmoid](https://pytorch.org/docs/2.7/generated/torch.nn.functional.hardsigmoid.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.silu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.silu](https://pytorch.org/docs/2.7/generated/torch.nn.functional.silu.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.mish

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.mish](https://pytorch.org/docs/2.7/generated/torch.nn.functional.mish.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.batch_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.batch_norm](https://pytorch.org/docs/2.7/generated/torch.nn.functional.batch_norm.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持fp16，fp32
- `weight`和`bias`仅支持一维场景
- `bias`的shape为1维，长度与`input`入参中channel轴的长度相等

</div>

### torch.nn.functional.group_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.group_norm](https://pytorch.org/docs/2.7/generated/torch.nn.functional.group_norm.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 该API仅支持2维及以上的输入`input`
- `eps`参数需大于0

</div>

### torch.nn.functional.layer_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.layer_norm](https://pytorch.org/docs/2.7/generated/torch.nn.functional.layer_norm.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.normalize

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.normalize](https://pytorch.org/docs/2.7/generated/torch.nn.functional.normalize.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64

</div>

## Linear functions

### torch.nn.functional.linear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.linear](https://pytorch.org/docs/2.7/generated/torch.nn.functional.linear.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.bilinear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.bilinear](https://pytorch.org/docs/2.7/generated/torch.nn.functional.bilinear.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

## Dropout functions

### torch.nn.functional.dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.dropout](https://pytorch.org/docs/2.7/generated/torch.nn.functional.dropout.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.alpha_dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.alpha_dropout](https://pytorch.org/docs/2.7/generated/torch.nn.functional.alpha_dropout.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.feature_alpha_dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.feature_alpha_dropout](https://pytorch.org/docs/2.7/generated/torch.nn.functional.feature_alpha_dropout.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.dropout2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.dropout2d](https://pytorch.org/docs/2.7/generated/torch.nn.functional.dropout2d.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- <term>Ascend 950DT</term>：不支持complex64，complex128

</div>

## Sparse functions

### torch.nn.functional.embedding

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.embedding](https://pytorch.org/docs/2.7/generated/torch.nn.functional.embedding.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，int32，int64
- 属性`max_norm`仅支持非负值

</div>

### torch.nn.functional.embedding_bag

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.embedding_bag](https://pytorch.org/docs/2.7/generated/torch.nn.functional.embedding_bag.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.nn.functional.one_hot

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.one_hot](https://pytorch.org/docs/2.7/generated/torch.nn.functional.one_hot.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持int64

</div>

## Distance functions

### torch.nn.functional.cosine_similarity

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.cosine_similarity](https://pytorch.org/docs/2.7/generated/torch.nn.functional.cosine_similarity.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.pdist

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pdist](https://pytorch.org/docs/2.7/generated/torch.nn.functional.pdist.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## Loss functions

### torch.nn.functional.binary_cross_entropy

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.binary_cross_entropy](https://pytorch.org/docs/2.7/generated/torch.nn.functional.binary_cross_entropy.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.binary_cross_entropy_with_logits

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.binary_cross_entropy_with_logits](https://pytorch.org/docs/2.7/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.poisson_nll_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.poisson_nll_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.poisson_nll_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，int64
- 可能回退至CPU执行

</div>

### torch.nn.functional.cross_entropy

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.cross_entropy](https://pytorch.org/docs/2.7/generated/torch.nn.functional.cross_entropy.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.ctc_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.ctc_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.ctc_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持fp32，fp64
- 目标序列的长度不支持0，即属性`target_lengths`的取值不能包含0

</div>

### torch.nn.functional.gaussian_nll_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.gaussian_nll_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.gaussian_nll_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.kl_div

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.kl_div](https://pytorch.org/docs/2.7/generated/torch.nn.functional.kl_div.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 当前`log_target`参数仅支持False
- 当前`target`不支持求导

</div>

### torch.nn.functional.l1_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.l1_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.l1_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，int64

</div>

### torch.nn.functional.mse_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.mse_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.mse_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，complex64，complex128

</div>

### torch.nn.functional.margin_ranking_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.margin_ranking_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.margin_ranking_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.multilabel_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.multilabel_margin_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.multilabel_margin_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：

- `input`仅支持fp16，fp32
- 输入`tensor`的元素个数不能超过10万

</div>

### torch.nn.functional.multilabel_soft_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.multilabel_soft_margin_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.multilabel_soft_margin_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.nn.functional.nll_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.nll_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.nll_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持fp32
- `target`中的每个元素值应大于等于0且小于`input`的类别数

</div>

### torch.nn.functional.smooth_l1_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.smooth_l1_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.smooth_l1_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.nn.functional.soft_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.soft_margin_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.soft_margin_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，不支持double，complex64，complex128数据类型

</div>

### torch.nn.functional.triplet_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.triplet_margin_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.triplet_margin_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.nn.functional.triplet_margin_with_distance_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.triplet_margin_with_distance_loss](https://pytorch.org/docs/2.7/generated/torch.nn.functional.triplet_margin_with_distance_loss.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## Vision functions

### torch.nn.functional.pixel_shuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pixel_shuffle](https://pytorch.org/docs/2.7/generated/torch.nn.functional.pixel_shuffle.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.pixel_unshuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pixel_unshuffle](https://pytorch.org/docs/2.7/generated/torch.nn.functional.pixel_unshuffle.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.nn.functional.pad

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pad](https://pytorch.org/docs/2.7/generated/torch.nn.functional.pad.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- 属性`mode`为constant时，`input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 属性`mode`非constant时，`input`仅支持fp16，fp32，fp64
- 在输入`x`为六维以上时可能会出现性能下降

</div>

### torch.nn.functional.interpolate

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.interpolate](https://pytorch.org/docs/2.7/generated/torch.nn.functional.interpolate.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，fp64
- 支持nearest、linear、bilinear、bicubic、trilinear、area
- 不支持`scale_factor`

</div>

### torch.nn.functional.upsample

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.upsample](https://pytorch.org/docs/2.7/generated/torch.nn.functional.upsample.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：

- `input`仅支持fp16，fp32，fp64
- 只支持`mode` = nearest，例如：

  ```python
  out = torch.nn.functional.upsample(x, size=(256, 256), mode='nearest')
  ```

</div>

### torch.nn.functional.upsample_nearest

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.upsample_nearest](https://pytorch.org/docs/2.7/generated/torch.nn.functional.upsample_nearest.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- `input`只支持3-5维

</div>

### torch.nn.functional.upsample_bilinear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.upsample_bilinear](https://pytorch.org/docs/2.7/generated/torch.nn.functional.upsample_bilinear.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32

</div>

### torch.nn.functional.grid_sample

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.grid_sample](https://pytorch.org/docs/2.7/generated/torch.nn.functional.grid_sample.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp16，fp32，fp64

</div>

### torch.nn.functional.affine_grid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.affine_grid](https://pytorch.org/docs/2.7/generated/torch.nn.functional.affine_grid.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp16，fp32

</div>
