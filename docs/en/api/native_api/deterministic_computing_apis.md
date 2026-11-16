# Deterministic Computation API List

## Introduction

When training with the PyTorch framework, some operators may involve randomness during computation. To exclude randomness from the output results, you need to enable the deterministic computation switch. When deterministic computation is enabled, executing the same operation with the same inputs on the same hardware and software always produces the same output results.

> [!NOTE]
>
>- Deterministic computation must be set in the same main process as the network and operators to be configured. In some model scripts, main\(\) and the training network are not in the same process.
>- Currently, the deterministic status can be set only once in the same thread. If it is set multiple times, the first valid setting prevails, and subsequent settings do not take effect.<br>
> A valid setting: after setting the deterministic status, you must actually execute at least one operator so that its task is delivered. Merely setting the status without executing any operator only enables the deterministic flag but does not actually affect operator behavior.<br>
> Solution:
>     1. Setting determinism multiple times in the same thread is not recommended for now.
>     2. This issue exists regardless of whether the deterministic switch is enabled or disabled, and will be resolved in a later version.

## Usage

For the usage and effects of deterministic computation, see the official documentation [torch.use\_deterministic\_algorithms](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms). This section only describes how to enable deterministic computation.

> [!CAUTION]  
> Enabling the deterministic switch may degrade performance.

1. Enable the deterministic computation switch:

    ```python
    torch.use_deterministic_algorithms(True)
    ```

2. Verify whether the setting is successful.

    1. Run the following command to query the interface setting:

        ```python
        torch.are_deterministic_algorithms_enabled()
        ```

    2. The returned output is as follows:

        ```python
        print(torch.are_deterministic_algorithms_enabled())
        ```

    During training, if the interface returns True, the deterministic computation switch is enabled; if it returns False, the switch is not enabled.

## API List

<!-- npu="A3,910b" id1 -->
When using <term>Atlas A2 training products</term> or <term>Atlas A3 training products</term>, the APIs listed in [Table 1](#deterministic-api-list-1) involve randomness during computation. Enabling the deterministic computation switch keeps the computation results deterministic.

**Table 1** API list<a id="deterministic-api-list-1"></a>

| API |
|-----|
| `torch.nn.functional.batch_norm` |
| `torch.nn.functional.binary_cross_entropy` |
| `torch.nn.functional.ctc_loss` |
| `torch.cumsum` |
| `torch.dot` |
| `torch.matmul` |
| `torch.nn.functional.embedding` |
| `torch.nn.functional.nll_loss` |
| `torch.nn.functional.prelu` |
| `torch.mean` |
| `torch.nn.functional.adaptive_avg_pool2d` |
| `torch.nn.functional.avg_pool2d` |
| `torch.nn.functional.binary_cross_entropy_with_logits` |
| `torch.nn.functional.mse_loss` |
| `torch.addbmm` |
| `torch.Tensor.addbmm_` |
| `torch.addmv` |
| `torch.Tensor.addmv_` |
| `torch.nn.functional.l1_loss` |
| `torch.nn.functional.smooth_l1_loss` |
| `torch.addmm` |
| `torch.Tensor.addmm` |
| `torch.mm` |
| `torch.bmm` |
| `torch.nn.functional.layer_norm` |
| `Tensor.put_` |
| `torch.Tensor.index_put` |
| `torch.Tensor.index_put_` |
| `torch.ops.aten.convolution_backward` |
| `torch.nn.NLLLoss2d` |
| `torch.nn.functional.fold` |
| `torch.nn.functional.unfold` |
| `torch.nn.functional.kl_div` |
| `torch.nn.functional.multilabel_margin_loss` |
| `torch.std_mean` |
| `torch.std` |
| `torch.var_mean` |
| `torch.var` |
| `torch.sum` |
| `torch.nn.functional.interpolate` |
| `torch.nn.functional.soft_margin_loss` |
| `torch.trace` |
| `torch.mv` |

<!-- end id1 -->

<!-- npu="950" id2 -->
When using <term>Ascend 950DT products</term>, the APIs listed in [Table 2](#deterministic-api-list-2) involve randomness during computation. Enabling the deterministic computation switch keeps the computation results deterministic.

**Table 2** API list<a id="deterministic-api-list-2"></a>

| API |
|-----|
| `torch.Tensor.put_` |
| `torch.max_unpool2d` |
| `torch.max_unpool3d` |
| `torch._unique` |
| `torch._unique2` |
| `torch.scatter_add` |
| `torch.Tensor.scatter_add_` |
| `torch.scatter` |
| `torch.Tensor.scatter_` |
| `torch.index_add` |
| `torch.bincount` |
| `torch.index_copy` |
| `torch.index_copy_` |
| `torch.Tensor.index_put_` |
| `torch.linalg.vector_norm` |
| `torch.ops.aten.upsample_bicubic2d` backward |
| `torch.ops.aten.upsample_bilinear2d` backward |
| `torch.ops.aten.upsample_linear1d` backward |
| `torch.ops.aten._upsample_bilinear2d_aa` backward |
| `torch.ops.aten._upsample_bicubic2d_aa` backward |
| `torch.nn.functional.adaptive_max_pool3d` backward |
| `torch.nn.functional.batch_norm` |
| `torch.Tensor.coalesce` |
| `torch.grid_sampler_2d` backward |
| `torch.grid_sampler_3d` backward |
<!-- end id2 -->
