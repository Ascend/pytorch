# Supported Deterministic Computation APIs

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:24:09.272Z pushedAt=2026-06-15T03:25:49.236Z -->

## Introduction

When training with the PyTorch framework, if you need deterministic output results (that is, excluding randomness), you must enable the deterministic computation switch. When deterministic computation is enabled, performing the same operation with the same input on the same hardware and software will always produce the same output result.

> [!NOTE]  
>
>- All deterministic computation settings must be in the same process as the network and operators to take effect. In some model scripts, `main()` and the training network do not run in the same process.
>- Currently, the deterministic state can be set only once per thread. If set multiple times, the first valid setting takes effect, and subsequent settings are ignored.
> A "valid setting" requires that at least one operator be actually executed after setting the deterministic state to dispatch the task. Setting the deterministic flag without executing any operator will not affect operator behavior.
> Solutions:
>     1. It is currently not recommended to set determinism multiple times in the same thread.
>     2. This issue exists regardless of whether the binary switch is enabled or disabled, and will be resolved in a future version.

## Usage

For details on the usage and effects of deterministic computation, see the corresponding official documentation [torch.use_deterministic_algorithms](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms). This section only introduces the method for enabling deterministic computation.

> [!CAUTION]
> Enabling the deterministic switch may cause performance degradation.

1. Enable the deterministic computation switch:

    ```python
    torch.use_deterministic_algorithms(True)
    ```

2. Verify that the setting is successful.

    1. Run the following command to query the interface setting:

        ```python
        torch.are_deterministic_algorithms_enabled()
        ```

    2. The output is as follows:

        ```python
        print(torch.are_deterministic_algorithms_enabled())
        ```

During training, when the API returns `True`, it indicates that the deterministic computing switch is currently enabled; when it returns `False`, it indicates that it is not enabled.

## Supported APIs

For the list of supported deterministic computation APIs, see [Table 1](#ascend-supported-deterministic-computation-api-list).

**Table 1** Ascend-supported deterministic computation API list<a id="ascend-supported-deterministic-computation-api-list"></a>

|API Name|
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
