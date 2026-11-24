# SHUT_DOWN_FX_PASS_LIST

## 功能描述

SHUT_DOWN_FX_PASS_LIST=xxx,yyy

1. 环境变量默认为""，此时仅有非默认关闭的pass生效；部分pass默认关闭，需通过各自的环境变量另行使能，清单见 [快速入门](./overview.md)。实际跳过的pass是默认关闭清单与本环境变量的并集。
2. 如果设定了xxx,yyy，则表示关闭xxx和yyy pass (只要某个pass生效，日志会打印"[inductor_fx] xxx works"；通过关闭某几个pass，可以用来精确控制，排查问题)。实际验证方式为观察DEBUG级别下是否出现"Registering function `<pass_name>` from module ..."打印，被关闭的pass不会打印该注册日志，与下方配置示例保持一致。
3. 如果设定为all，则表示关闭所有pass。

## 配置示例

### 开启所有pass

```bash
export SHUT_DOWN_FX_PASS_LIST=""
```

```output
# 验证：观察日志非默认关闭的pass均注册（默认关闭的pass需通过各自环境变量另行使能）
DEBUG - Registering function cat_to_view_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function repeat_to_expand_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_iota_arithmetic_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function broadcast_const_mask_compress from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function masked_add_compose_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function bool_cast_mul_to_where_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function sign_diff_hamming_fuse_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function batch_embedding_fusion_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function cat_slice_cat_fold_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.PRE, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function pad_slice_fold from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.PRE, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_four_op_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_cast from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_cat from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_clone from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_detach from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_expand from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_reduce from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_sink_view from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_slice from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_squeeze from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_to_copy from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function view_fold_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_where from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_redundant_ops from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function dtype_optimal_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.PRE, fx_pass_level=FxPassLevel.LEVEL1
```

### 关闭所有pass

```bash
export SHUT_DOWN_FX_PASS_LIST=all
```

```text
# 验证：观察日志所有pass均未注册
```

### 关闭指定pass（dtype_optimal_pass）

```bash
export SHUT_DOWN_FX_PASS_LIST=dtype_optimal_pass
```

```output
# 验证：观察日志dtype_optimal_pass未注册
DEBUG - Registering function cat_to_view_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function repeat_to_expand_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_iota_arithmetic_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function broadcast_const_mask_compress from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function masked_add_compose_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function bool_cast_mul_to_where_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function sign_diff_hamming_fuse_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function batch_embedding_fusion_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function cat_slice_cat_fold_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.PRE, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function pad_slice_fold from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.PRE, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_four_op_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_cast from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_cat from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_clone from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_detach from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_expand from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_reduce from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_sink_view from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_slice from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_squeeze from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_to_copy from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function view_fold_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_where from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_redundant_ops from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
```

### 关闭多个pass

```bash
export SHUT_DOWN_FX_PASS_LIST=dtype_optimal_pass,fold_redundant_ops
```

```output
# 验证：观察日志dtype_optimal_pass、fold_redundant_ops未注册
DEBUG - Registering function cat_to_view_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function repeat_to_expand_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_iota_arithmetic_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function broadcast_const_mask_compress from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function masked_add_compose_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function bool_cast_mul_to_where_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function sign_diff_hamming_fuse_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function batch_embedding_fusion_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function cat_slice_cat_fold_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.PRE, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function pad_slice_fold from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.PRE, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_four_op_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_cast from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_cat from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_clone from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_detach from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_expand from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_reduce from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_sink_view from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_slice from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_squeeze from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_to_copy from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function view_fold_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
DEBUG - Registering function fold_where from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
```

## 使用约束

环境变量默认为空(SHUT_DOWN_FX_PASS_LIST="")，图优化特性默认开启，当前主要适用于模型推理过程；其中标记了 ignore_inference_check 的pass在训练图上同样生效。

## 支持的型号

<term>Ascend 950DT系列产品</term>
