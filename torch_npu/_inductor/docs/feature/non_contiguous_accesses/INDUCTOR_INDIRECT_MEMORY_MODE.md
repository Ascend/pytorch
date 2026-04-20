# INDUCTOR_INDIRECT_MEMORY_MODE

## 功能描述
是否开启离散访存的融合以及配置融合方式，默认值为"simd_simt_mix"。

## 配置示例
若模型中存在离散访存相关的算子，通过`export INDUCTOR_INDIRECT_MEMORY_MODE={mode}`使用相关特性，其中mode可选值为`fallback/simd_simt_mix`，推荐使用`export INDUCTOR_INDIRECT_MEMORY_MODE=simd_simt_mix`。

+ fallback：对于离散访存类的算子不进行Inductor融合。
+ simd_simt_mix：对于离散访存类的算子使用load、store的DSL(与社区一致)，该kernel支持simd模板方案、纯simt方案、simd+simt模板三种不同类型算子，通过Autotune选择调优最优tiling。生成的triton算子在inductor_meta中通过字段标识该算子为模板方案simt算子{'npu_kernel_type': 'simd_simt_mix'}。

## 使用约束
A2、A3不支持离散访存类算子的inductor融合，仅在A5上支持离散访存特性。

## 支持型号
-   <term>Atlas A5 训练系列产品</term>