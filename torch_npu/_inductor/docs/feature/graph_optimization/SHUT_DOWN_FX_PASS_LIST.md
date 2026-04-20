# SHUT_DOWN_FX_PASS_LIST

## 功能描述
SHUT_DOWN_FX_PASS_LIST=xxx,yyy
1. 环境变量默认为""， 即所有pass都生效。
2. 如果设定了xxx,yyy，则表示关闭xxx和yyy pass (只要某个pass生效，日志会打印"[inductor_fx_pass] xxx works"； 通过关闭某几个pass，可以用来精确控制，排查问题)。
3. 如果设定为all，则表示关闭所有pass。

## 配置示例
### 开启所有pass
```
export SHUT_DOWN_FX_PASS_LIST=""
```
```
# 验证：观察日志所有pass均注册
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
```
export SHUT_DOWN_FX_PASS_LIST=all
```
```
# 验证：观察日志所有pass均未注册
```
### 关闭指定pass: dtype_optimal_pass
```
export SHUT_DOWN_FX_PASS_LIST=dtype_optimal_pass
```
```
# 验证：观察日志dtype_optimal_pass未注册
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
```
export SHUT_DOWN_FX_PASS_LIST=dtype_optimal_pass,fold_redundant_ops
```
```
# 验证：观察日志dtype_optimal_pass、fold_redundant_ops未注册
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
环境变量默认为空(SHUT_DOWN_FX_PASS_LIST="")，图优化特性默认开启，当前仅适用于模型推理过程。

## 支持的型号
-   <term>Atlas A5 系列产品</term>
