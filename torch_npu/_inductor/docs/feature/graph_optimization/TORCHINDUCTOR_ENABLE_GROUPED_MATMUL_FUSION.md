# TORCHINDUCTOR_ENABLE_GROUPED_MATMUL_FUSION

## 功能描述

TORCHINDUCTOR_ENABLE_GROUPED_MATMUL_FUSION=1

控制 grouped_matmul_fusion_pass 的开关。该pass把汇聚到同一个cat的多个互相独立的小GEMM，合并成一次 npu_grouped_matmul 调用，用一个kernel替代原来N次matmul下发。

1. 环境变量默认不设置，等价于关闭，此时pass不会被注册，推理和训练两条路径都不会执行它。
2. 取值为 1、true、yes、on、y（不区分大小写）时使能，其余取值均按关闭处理。
3. 该pass注册在POST阶段（pass_type=PassType.POST，fx_pass_level=FxPassLevel.LEVEL1）。
4. 使能后该pass仍可通过 SHUT_DOWN_FX_PASS_LIST 关闭，两个开关取并集。

## 配置示例

观察下列日志需要同时打开inductor的DEBUG日志：

```bash
export INDUCTOR_ASCEND_LOG_LEVEL=DEBUG
```

### 使能grouped_matmul_fusion_pass

```bash
export TORCHINDUCTOR_ENABLE_GROUPED_MATMUL_FUSION=1
```

```text
# 验证：观察日志grouped_matmul_fusion_pass已注册
DEBUG - Registering function grouped_matmul_fusion_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
```

### 保持关闭（默认）

```bash
unset TORCHINDUCTOR_ENABLE_GROUPED_MATMUL_FUSION
```

```text
# 验证：观察日志grouped_matmul_fusion_pass未注册
DEBUG - Ignoring registration of grouped_matmul_fusion_pass
```

## 使用约束

- 改写保持每个GEMM的操作数不变，但grouped算子的累加方式与逐个matmul不同，结果接近但不是逐比特一致，对精度敏感的场景需评估该差异的影响。
- 分支数少于8个的cat不做合并：这类cat往往是宽GEMM（QKV投影之类），实测grouped kernel比拆开的matmul慢约12%。
- 单次调用的组数上限为32，超出会拆成多次调用：超过32组后算子会走代价高得多的host路径，实测每组耗时从2.18us跳到6.90us。
- 另有行数上限4096，用于兜住已经能填满cube的GEMM；该约束只在行数是静态值时生效，而batch维通常不是静态的，因此实际起决定作用的是上面的分支数下限。

## 支持的型号

- <term>Ascend 950DT 系列产品</term>
