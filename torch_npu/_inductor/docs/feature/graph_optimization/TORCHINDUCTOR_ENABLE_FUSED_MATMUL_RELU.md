# TORCHINDUCTOR_ENABLE_FUSED_MATMUL_RELU

## 功能描述

TORCHINDUCTOR_ENABLE_FUSED_MATMUL_RELU=1

控制 fused_matmul_relu_pass 的开关。该pass把 relu(addmm) 融合为 npu_fused_matmul（对应 aclnnFusedMatmul，fusedOpType="relu"，即 y = relu(x1@x2 + bias)），relu不再作为独立的pointwise kernel下发，中间结果也不用再写回搬出。

覆盖的图形态如下：

```text
Pattern1: relu(addmm(bias, x, w))       -> npu_fused_matmul(x, w, bias, "relu")
Pattern2: view(relu(view(addmm(...))))  -> view(npu_fused_matmul(...))
          AOT会把3维linear拆成这种形态、relu作用在view上，relu是pointwise的，
          因此吸收进算子，输出仍由原来的view承载
Pattern3: relu(mm(x, w) + bias)         -> npu_fused_matmul(x, w, bias, "relu")
          inductor未把mm+add折叠成addmm时出现的形态
Pattern4: relu(mm(x, w))                -> npu_fused_matmul(x, w, "relu")
          无bias的linear/matmul，算子的bias是可选的，保持None
```

1. 环境变量默认不设置，等价于关闭，此时pass不会被注册，推理和训练两条路径都不会执行它。
2. 取值为 1、true、yes、on、y（不区分大小写）时使能，其余取值均按关闭处理。
3. 该pass注册在POST阶段（pass_type=PassType.POST，fx_pass_level=FxPassLevel.LEVEL1）：post_grad图中linear已被分解，addmm的权重已经是 (K, N)，与算子的x2直接对应，无需额外插入transpose或reshape。
4. 使能后该pass仍可通过 SHUT_DOWN_FX_PASS_LIST 关闭，两个开关取并集。

## 配置示例

观察下列日志需要同时打开inductor的DEBUG日志：

```bash
export INDUCTOR_ASCEND_LOG_LEVEL=DEBUG
```

### 使能fused_matmul_relu_pass

```bash
export TORCHINDUCTOR_ENABLE_FUSED_MATMUL_RELU=1
```

```text
# 验证：观察日志fused_matmul_relu_pass已注册
DEBUG - Registering function fused_matmul_relu_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
```

### 保持关闭（默认）

```bash
unset TORCHINDUCTOR_ENABLE_FUSED_MATMUL_RELU
```

```text
# 验证：观察日志fused_matmul_relu_pass未注册
DEBUG - Ignoring registration of fused_matmul_relu_pass
```

## 使用约束

该pass的匹配条件是严格的，下列任一条不满足时保持原图不变：

- 仅<term>Ascend 950DT 系列产品</term>（Ascend 950）提供 npu_fused_matmul 算子。
- x1与权重必须均为2维，且x2的rank与x1相同，不支持broadcast。
- 所有输入的dtype必须一致，且为float16或bfloat16。
- 带bias时，bias必须是1维且连续。
- 被替换的节点必须是单用户（single-user）。

## 支持的型号

- <term>Ascend 950DT 系列产品</term>
