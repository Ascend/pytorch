# 确定性计算API支持清单

## 简介

在使用PyTorch框架进行训练时，若需要输出结果排除随机性，则需要设置确定性计算开关。在开启确定性计算时，当使用相同的输入在相同的硬件和软件上执行相同的操作，输出的结果每次都是相同的。

> [!NOTE]  
>-   确定性计算固定方法都必须与待固定的网络、算子等在同一个主进程，部分模型脚本中main\(\)与训练网络并不在一个进程中。
>-   当前同一线程中只能设置一次确定性状态，多次设置以第一次有效设置为准，后续设置不会生效。<br>
>    有效设置：在设置确定性状态后，必须实际执行至少一次算子，使其任务下发。仅仅设置而未执行任何算子，只会开启确定性标志，但不会真正影响算子行为。<br>
>    解决方案：
>     1.  暂不推荐在同一线程中多次设置确定性。
>     2.  该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。

## 使用方法

确定性计算的用法和效果具体可参考相应官方文档[torch.use\_deterministic\_algorithms](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms)，本小节仅介绍开启确定性计算的方法。

> [!CAUTION]  
> 开启确定性开关可能会导致性能下降。

1.  开启确定性计算开关：

    ```
    torch.use_deterministic_algorithms(True)
    ```

2.  验证设置是否成功。

    1.  执行如下命令查询接口设置：

        ```
        torch.are_deterministic_algorithms_enabled()
        ```

    2.  回显示例如下：

        ```
        print(torch.are_deterministic_algorithms_enabled())
        ```

    执行训练时，打印此接口的返回值为True时，表示当前已开启确定性计算开关，返回False则表示未开启。

## API支持清单

昇腾支持的确定性计算API列表请参见[表1](#昇腾支持的确定性计算API列表)。

**表 1**  昇腾支持的确定性计算API列表<a id="昇腾支持的确定性计算API列表"></a>

|API名称|
|--|
|torch.nn.functional.batch_norm|
|torch.nn.functional.binary_cross_entropy|
|torch.nn.functional.ctc_loss|
|torch.cumsum|
|torch.dot|
|torch.matmul|
|torch.nn.functional.embedding|
|torch.nn.functional.nll_loss|
|torch.nn.functional.prelu|
|torch.mean|
|torch.nn.functional.adaptive_avg_pool2d|
|torch.nn.functional.avg_pool2d|
|torch.nn.functional.binary_cross_entropy_with_logits|
|torch.nn.functional.mse_loss|
|torch.addbmm|
|torch.Tensor.addbmm_|
|torch.addmv|
|torch.Tensor.addmv_|
|torch.nn.functional.l1_loss|
|torch.nn.functional.smooth_l1_loss|
|torch.addmm|
|torch.Tensor.addmm|
|torch.mm|
|torch.bmm|
|torch.nn.functional.layer_norm|
|Tensor.put_|
|torch.Tensor.index_put|
|torch.Tensor.index_put_|
|torch_npu.npu_convolution_transpose|
|torch_npu.npu_convolution|
|torch_npu.npu_linear|
|torch_npu.npu_deformable_conv2d|
|torch.ops.aten.convolution_backward|
|torch.nn.NLLLoss2d|
|torch_npu.npu_ps_roi_pooling|
|torch.nn.functional.fold|
|torch.nn.functional.unfold|
|torch.nn.functional.kl_div|
|torch.nn.functional.multilabel_margin_loss|
|torch.std_mean|
|torch.std|
|torch.var_mean|
|torch.var|
|torch.sum|
|torch.nn.functional.interpolate|
|torch.nn.functional.soft_margin_loss|
|torch.trace|
|torch.mv|


