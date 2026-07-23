# torch.nn.functional

> [!NOTE]   
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
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

## base API

### torch.nn.parallel.data_parallel

<div style="margin-left: 2em">

**原生文档**：[torch.nn.parallel.data_parallel](https://pytorch.org/docs/2.11/generated/torch.nn.functional.torch.nn.parallel.data_parallel.html#torch.nn.parallel.data_parallel)

**是否支持**：否

</div>

## Convolution functions

### torch.nn.functional.conv1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv1d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.conv2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv2d.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.conv3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv3d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，complex64

</div>

### torch.nn.functional.conv_transpose1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv_transpose1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv_transpose1d.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### torch.nn.functional.conv_transpose2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv_transpose2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv_transpose2d.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.conv_transpose3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.conv_transpose3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.conv_transpose3d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.unfold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.unfold](https://pytorch.org/docs/2.11/generated/torch.nn.functional.unfold.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.fold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.fold](https://pytorch.org/docs/2.11/generated/torch.nn.functional.fold.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

## Pooling functions

### torch.nn.functional.avg_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.avg_pool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.avg_pool1d.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.avg_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.avg_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.avg_pool2d.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.avg_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.avg_pool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.avg_pool3d.html)

**是否支持**：否

</div>

### torch.nn.functional.max_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_pool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_pool1d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp16，fp32
- dilation仅支持1
- 通过设置torch_npu.npu.use_compatible_impl(True)，保证与社区同名接口在内存一致性上对齐
- return_indices为True时，返回的argmax的数据类型为int32

</div>

### torch.nn.functional.max_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_pool2d.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- dilation的值目前在NPU上仅支持设置为1或(1,1)
- 通过设置torch_npu.npu.use_compatible_impl(True)，保证与社区同名接口在内存一致性上对齐
- return_indices为True时，返回的argmax的数据类型为int32

</div>

### torch.nn.functional.max_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_pool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_pool3d.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- dilation的值目前在NPU上仅支持设置为1或(1,1,1)
- return_indices为True时，返回的argmax的数据类型为int32

</div>

### torch.nn.functional.max_unpool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_unpool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_unpool1d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int32，int64

</div>

### torch.nn.functional.max_unpool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_unpool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_unpool2d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp16，fp32，fp64，uint8，int8，int32，int64
- jit_compile=False即二进制模式时，output_size的乘积需要大于等于input的H，W的乘积

</div>

### torch.nn.functional.max_unpool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.max_unpool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.max_unpool3d.html)

**是否支持**：否

</div>

### torch.nn.functional.lp_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.lp_pool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.lp_pool1d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.lp_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.lp_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.lp_pool2d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.adaptive_max_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_max_pool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_max_pool1d.html)

**是否支持**：否

</div>

### torch.nn.functional.adaptive_max_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_max_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_max_pool2d.html)

**是否支持**：否

</div>

### torch.nn.functional.adaptive_max_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_max_pool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_max_pool3d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32，fp64

</div>

### torch.nn.functional.adaptive_avg_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_avg_pool1d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_avg_pool1d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.adaptive_avg_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_avg_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_avg_pool2d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.adaptive_avg_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.adaptive_avg_pool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.adaptive_avg_pool3d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.fractional_max_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.fractional_max_pool2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.fractional_max_pool2d.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

### torch.nn.functional.fractional_max_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.fractional_max_pool3d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.fractional_max_pool3d.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Attention Mechanisms

### torch.nn.functional.scaled_dot_product_attention

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention)

**是否支持**：<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>：是

**限制与说明**：

- 支持bf16、fp16和fp32。所有参数输入均符合以下约束
- **约束**：
- 所有参数输入符合规格：
- &#8226; 输入query、key、value的N：batch size，当前只支持[N，head_num, S(L), E(Ev)]的排布方式，取值范围1~2K
- &#8226; 输入query的head num和key/value的head num必须成比例关系，即Nq/Nkv必须是非0整数，取值范围1~256
- &#8226; 输入query的L：Target sequence length，取值范围1~512K
- &#8226; 输入key、value的S：Source sequence length，取值范围1~512K
- 输入query、key、value的E：Embedding dimension of the query and key，取值范围1~512
- &#8226; 输入value的Ev：Embedding dimension of the value，必须与E相等
- &#8226; 输入attn_mask：当前支持[N, 1, L, S]、[N, head_num, L, S]、[1, 1, L, S]、[L, S]，以及可广播到[N, head_num, L, S]的bool类型mask，例如[L, 1]、[1, S]、[1, 1]等排布方式
- &#8226; 在开启is_causal计算时，attn_mask必须为None；不开启is_causal时，若attn_mask输入有效数据，输入数据类型必须是bool类型
- &#8226; 输入query、key、value的数据类型为bf16、fp16、fp32
- &#8226; 通过设置torch_npu.npu.use_compatible_impl(True)，支持按SDPA后端选择上下文指定MATH后端
- 与原接口除了规格限制之外的差异点：
- &#8226; NPU的随机算法部分用DSA硬件实现，算法在DSA引擎固化与GPU算法实现存在差异，导致dropout功能和GPU结果不一致
- &#8226; 当前接口支持输入query的head num和key/value的head num不等长，而原生PyTorch接口不支持

</div>

## Non-linear activation functions

### torch.nn.functional.threshold

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.threshold](https://pytorch.org/docs/2.11/generated/torch.nn.functional.threshold.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64
- 当输入为超过16,777,216（即2<sup>24</sup>）的int32类型时，精度会有损失

</div>

### torch.nn.functional.threshold_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.threshold_](https://pytorch.org/docs/2.11/generated/torch.nn.functional.threshold_.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64
- 当输入为超过16,777,216（即2<sup>24</sup>）的int32类型时，精度会有损失

</div>

### torch.nn.functional.relu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.relu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.relu.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.nn.functional.relu_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.relu_](https://pytorch.org/docs/2.11/generated/torch.nn.functional.relu_.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.nn.functional.hardtanh

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardtanh](https://pytorch.org/docs/2.11/generated/torch.nn.functional.hardtanh.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.hardtanh_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardtanh_](https://pytorch.org/docs/2.11/generated/torch.nn.functional.hardtanh_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，int8，int16，int32，int64

</div>

### torch.nn.functional.hardswish

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardswish](https://pytorch.org/docs/2.11/generated/torch.nn.functional.hardswish.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- 可能回退至CPU执行

</div>

### torch.nn.functional.relu6

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.relu6](https://pytorch.org/docs/2.11/generated/torch.nn.functional.relu6.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.elu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.elu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.elu.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.elu_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.elu_](https://pytorch.org/docs/2.11/generated/torch.nn.functional.elu_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.selu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.selu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.selu.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- fp16的反向场景下对比GPU存在精度累加误差，可以通过如下方式进行规避：
- 将正向调用的torch.nn.functional.selu替换成torch.ops.aten.elu，例如：将torch.nn.functional.selu(input_x)替换为torch.ops.aten.elu(input_x, 1.6732632423543772848170429916717, 1.0507009873554804934193349852946)

</div>

### torch.nn.functional.celu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.celu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.celu.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.leaky_relu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.leaky_relu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.leaky_relu.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.leaky_relu_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.leaky_relu_](https://pytorch.org/docs/2.11/generated/torch.nn.functional.leaky_relu_.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64

</div>

### torch.nn.functional.prelu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.prelu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.prelu.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- input仅支持1-8维

</div>

### torch.nn.functional.rrelu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.rrelu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.rrelu.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.glu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.glu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.glu.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.gelu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.gelu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.gelu.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- approximate参数仅支持设置为tanh

</div>

### torch.nn.functional.logsigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.logsigmoid](https://pytorch.org/docs/2.11/generated/torch.nn.functional.logsigmoid.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.hardshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardshrink](https://pytorch.org/docs/2.11/generated/torch.nn.functional.hardshrink.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.softsign

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softsign](https://pytorch.org/docs/2.11/generated/torch.nn.functional.softsign.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.softplus

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softplus](https://pytorch.org/docs/2.11/generated/torch.nn.functional.softplus.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softmax](https://pytorch.org/docs/2.11/generated/torch.nn.functional.softmax.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

### torch.nn.functional.softshrink

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.softshrink](https://pytorch.org/docs/2.11/generated/torch.nn.functional.softshrink.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.gumbel_softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.gumbel_softmax](https://pytorch.org/docs/2.11/generated/torch.nn.functional.gumbel_softmax.html)

**是否支持**：否

</div>

### torch.nn.functional.log_softmax

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.log_softmax](https://pytorch.org/docs/2.11/generated/torch.nn.functional.log_softmax.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.tanh

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.tanh](https://pytorch.org/docs/2.11/generated/torch.nn.functional.tanh.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.nn.functional.sigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.sigmoid](https://pytorch.org/docs/2.11/generated/torch.nn.functional.sigmoid.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.hardsigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.hardsigmoid](https://pytorch.org/docs/2.11/generated/torch.nn.functional.hardsigmoid.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.silu

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.silu](https://pytorch.org/docs/2.11/generated/torch.nn.functional.silu.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.mish

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.mish](https://pytorch.org/docs/2.11/generated/torch.nn.functional.mish.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.batch_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.batch_norm](https://pytorch.org/docs/2.11/generated/torch.nn.functional.batch_norm.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32
- weight和bias仅支持一维场景
- bias的shape为1维，长度与input入参中channel轴的长度相等

</div>

### torch.nn.functional.group_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.group_norm](https://pytorch.org/docs/2.11/generated/torch.nn.functional.group_norm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持bf16，fp16，fp32
- 该API仅支持2维及以上的输入input
- eps参数需大于0

</div>

### torch.nn.functional.layer_norm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.layer_norm](https://pytorch.org/docs/2.11/generated/torch.nn.functional.layer_norm.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.normalize

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.normalize](https://pytorch.org/docs/2.11/generated/torch.nn.functional.normalize.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64

</div>

## Linear functions

### torch.nn.functional.linear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.linear](https://pytorch.org/docs/2.11/generated/torch.nn.functional.linear.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.bilinear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.bilinear](https://pytorch.org/docs/2.11/generated/torch.nn.functional.bilinear.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

## Dropout functions

### torch.nn.functional.dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.dropout](https://pytorch.org/docs/2.11/generated/torch.nn.functional.dropout.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.alpha_dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.alpha_dropout](https://pytorch.org/docs/2.11/generated/torch.nn.functional.alpha_dropout.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.feature_alpha_dropout

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.feature_alpha_dropout](https://pytorch.org/docs/2.11/generated/torch.nn.functional.feature_alpha_dropout.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.dropout2d

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.dropout2d](https://pytorch.org/docs/2.11/generated/torch.nn.functional.dropout2d.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128
- <term>Ascend 950DT</term>：不支持complex64，complex128

</div>

## Sparse functions

### torch.nn.functional.embedding

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.embedding](https://pytorch.org/docs/2.11/generated/torch.nn.functional.embedding.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持bf16，fp16，fp32，int32，int64
- 属性max_norm仅支持非负值

</div>

### torch.nn.functional.embedding_bag

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.embedding_bag](https://pytorch.org/docs/2.11/generated/torch.nn.functional.embedding_bag.html)

**是否支持**：否

</div>

### torch.nn.functional.one_hot

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.one_hot](https://pytorch.org/docs/2.11/generated/torch.nn.functional.one_hot.html)

**是否支持**：是

**限制与说明**： 支持int32，int64

</div>

## Distance functions

### torch.nn.functional.cosine_similarity

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.cosine_similarity](https://pytorch.org/docs/2.11/generated/torch.nn.functional.cosine_similarity.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.pdist

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pdist](https://pytorch.org/docs/2.11/generated/torch.nn.functional.pdist.html)

**是否支持**：否

</div>

## Loss functions

### torch.nn.functional.binary_cross_entropy

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.binary_cross_entropy](https://pytorch.org/docs/2.11/generated/torch.nn.functional.binary_cross_entropy.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.binary_cross_entropy_with_logits

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.binary_cross_entropy_with_logits](https://pytorch.org/docs/2.11/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.poisson_nll_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.poisson_nll_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.poisson_nll_loss.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，int64
- 可能回退至CPU执行

</div>

### torch.nn.functional.cross_entropy

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.cross_entropy](https://pytorch.org/docs/2.11/generated/torch.nn.functional.cross_entropy.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.ctc_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.ctc_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.ctc_loss.html)

**是否支持**：是

**限制与说明**：

- 支持fp32，fp64
- 目标序列的长度不支持0，即属性target_lengths的取值不能包含0

</div>

### torch.nn.functional.gaussian_nll_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.gaussian_nll_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.gaussian_nll_loss.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64

</div>

### torch.nn.functional.kl_div

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.kl_div](https://pytorch.org/docs/2.11/generated/torch.nn.functional.kl_div.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 当前log_target参数仅支持False
- 当前target不支持求导

</div>

### torch.nn.functional.l1_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.l1_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.l1_loss.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int64

</div>

### torch.nn.functional.mse_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.mse_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.mse_loss.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.margin_ranking_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.margin_ranking_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.margin_ranking_loss.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.multilabel_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.multilabel_margin_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.multilabel_margin_loss.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp16，fp32
- 输入tensor的元素个数不能超过10万

</div>

### torch.nn.functional.multilabel_soft_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.multilabel_soft_margin_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.multilabel_soft_margin_loss.html)

**是否支持**：否

</div>

### torch.nn.functional.nll_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.nll_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.nll_loss.html)

**是否支持**：是

**限制与说明**：

- 支持fp32
- target中的每个元素值应大于等于0且小于input的类别数

</div>

### torch.nn.functional.smooth_l1_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.smooth_l1_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.smooth_l1_loss.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.nn.functional.soft_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.soft_margin_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.soft_margin_loss.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，不支持double，complex64，complex128数据类型

</div>

### torch.nn.functional.triplet_margin_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.triplet_margin_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.triplet_margin_loss.html)

**是否支持**：否

</div>

### torch.nn.functional.triplet_margin_with_distance_loss

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.triplet_margin_with_distance_loss](https://pytorch.org/docs/2.11/generated/torch.nn.functional.triplet_margin_with_distance_loss.html)

**是否支持**：否

</div>

## Vision functions

### torch.nn.functional.pixel_shuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pixel_shuffle](https://pytorch.org/docs/2.11/generated/torch.nn.functional.pixel_shuffle.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.nn.functional.pixel_unshuffle

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pixel_unshuffle](https://pytorch.org/docs/2.11/generated/torch.nn.functional.pixel_unshuffle.html)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.nn.functional.pad

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.pad](https://pytorch.org/docs/2.11/generated/torch.nn.functional.pad.html)

**是否支持**：是

**限制与说明**：

- 属性mode为constant时，支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 属性mode非constant时，支持fp16，fp32，fp64
- 在输入x为六维以上时可能会出现性能下降问题

</div>

### torch.nn.functional.interpolate

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.interpolate](https://pytorch.org/docs/2.11/generated/torch.nn.functional.interpolate.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持bf16，fp16，fp32，fp64
- 支持nearest，linear，bilinear，bicubic，trilinear，area
- 不支持scale_factor

</div>

### torch.nn.functional.upsample

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.upsample](https://pytorch.org/docs/2.11/generated/torch.nn.functional.upsample.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp16，fp32，fp64
- 只支持mode = nearest

</div>

### torch.nn.functional.upsample_nearest

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.upsample_nearest](https://pytorch.org/docs/2.11/generated/torch.nn.functional.upsample_nearest.html)

**是否支持**：是

**限制与说明**：

- 支持fp16，fp32，fp64
- 只支持3-5维

</div>

### torch.nn.functional.upsample_bilinear

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.upsample_bilinear](https://pytorch.org/docs/2.11/generated/torch.nn.functional.upsample_bilinear.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

### torch.nn.functional.grid_sample

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.grid_sample](https://pytorch.org/docs/2.11/generated/torch.nn.functional.grid_sample.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64

</div>

### torch.nn.functional.affine_grid

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.affine_grid](https://pytorch.org/docs/2.11/generated/torch.nn.functional.affine_grid.html)

**是否支持**：是

**限制与说明**： 支持fp16，fp32

</div>

## Low-Precision functions

### torch.nn.functional.scaled_mm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.scaled_mm](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_mm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp8模式下ScalingType为tensorwise，rowwise和BlockWise1x128，mxfp8模式下ScalingType为BlockWise1x32的排布，mxfp8遵循[aclnnQuantMatmulV5](https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md)要求（scale_a和scale_b详见约束说明）

</div>

### torch.nn.functional.scaled_grouped_mm

<div style="margin-left: 2em">

**原生文档**：[torch.nn.functional.scaled_grouped_mm](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_grouped_mm.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp8模式下ScalingType为rowwise，mxfp8模式的排布，mxfp8遵循[aclnnGroupedMatmulV5](https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)要求（scale_a和scale_b详见约束说明）

</div>
