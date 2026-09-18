# TORCHINDUCTOR_ENABLE_GRAD_MATMUL_TRANSPOSE_OPT

## 功能描述

TORCHINDUCTOR_ENABLE_GRAD_MATMUL_TRANSPOSE_OPT=1

控制 grad_matmul_transpose_opt_pass 的开关。该pass让权重梯度直接以连续layout产出，把反向输出上的 permute(mm(lhs, rhs), [1, 0]) 改写为 mm(rhs.T, lhs.T)，省去对mm结果再做一次转置搬移。

```text
# 改写前
permute(mm(lhs, rhs), [1, 0])

# 改写后
mm(rhs.T, lhs.T)
```

1. 环境变量默认不设置，等价于关闭，此时pass不会被注册，推理和训练两条路径都不会执行它。
2. 取值为 1、true、yes、on、y（不区分大小写）时使能，其余取值均按关闭处理。
3. 该pass注册在POST阶段（pass_type=PassType.POST，fx_pass_level=FxPassLevel.LEVEL1）。
4. 使能后该pass仍可通过 SHUT_DOWN_FX_PASS_LIST 关闭，两个开关取并集。

## 配置示例

观察下列日志需要同时打开inductor的DEBUG日志：

```bash
export INDUCTOR_ASCEND_LOG_LEVEL=DEBUG
```

### 使能grad_matmul_transpose_opt_pass

```bash
export TORCHINDUCTOR_ENABLE_GRAD_MATMUL_TRANSPOSE_OPT=1
```

```text
# 验证：观察日志grad_matmul_transpose_opt_pass已注册
DEBUG - Registering function grad_matmul_transpose_opt_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
```

### 保持关闭（默认）

```bash
unset TORCHINDUCTOR_ENABLE_GRAD_MATMUL_TRANSPOSE_OPT
```

```text
# 验证：观察日志grad_matmul_transpose_opt_pass未注册
DEBUG - Ignoring registration of grad_matmul_transpose_opt_pass
```

## 使用约束

- 只改写反向图中直接作为输出的 permute(mm(...), [1, 0])，该pass面向训练场景。
- 当前版本只对方阵输出生效：这样M/N/K保持不变，把改变GEMM操作数带来的性能风险限制在可控范围内。

## 支持的型号

- <term>Ascend 950DT 系列产品</term>
