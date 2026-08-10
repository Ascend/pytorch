# 离散访存特性介绍

## 特性简介

推荐模型的Embedding参数规模极其庞大，在推理及训练过程中需要频繁地对Embedding进行离散访问（embedding、gather等），模型中访存类算子对于inductor中自动融合性能有巨大影响。

| 算子类别               | pytorch算子                                               |
| ------------------ | ------------------------------------------------------- |
| load类(GM到UB为离散访存)  | aten.embedding、aten.index、aten.gather、aten.index_select |
| store类(UB到GM为离散访存) | aten.index_put、aten.scatter                             |

对于这些算子，在inductor中生成的算子将会存在离散访存类型的索引表达式；具体的以`aten.embedding`为例，先通过一次load将索引从gm(in_ptr0)搬运到ub中(tmp0)，再使用索引(tmp0)从gm(in_ptr1)搬运长度为128的向量组成新的ub上的tensor(tmp1)。

``` python
y0=tl.arange(0, Y0BLOCK_SUB) # y0为连续访存，假设Y0BLOCK_SUB为4
# in_ptr0 = [1, 0, 4, 5]
tmp0 = tl.load(in_ptr0 + (y0), y0_mask, other=0.0)
# tmp0 = [1, 0, 8, 10]
# in_ptr1 = [[row0], [row1], ..., [row10]]
tmp1 = tl.load(in_ptr1 + (x1 + 128*tmp0), y0_mask & x1_mask)
# tmp1 = [[row1], [row0], [row8], [row10]]
# tmp1 即为间接访存
```

## 原理

由于A2/A3上仅支持simd访存，对于离散访存场景仅能通过标量搬运，因此上述的离散访存算子将会fallback到eager模式运行。而在A5硬件中加入了simt访存能力，对于间接访存的场景使用simt能够加速离散数据的搬运，本特性将会支持上述的离散访存算子在A5硬件上的inductor融合。

如果在Inductor中生成了间接访存的IR，inductor则需要将该Kernel标记为需要使用间接访存相关算子。对于间接访存相关的算子，由于存在多种不同的Codegen以及Autotune逻辑，使用环境变量进行控制。

1. simd_simt_mix模式：
    该情况算子在底层编译器存在3种处理方案：

    - SIMT方案：整体kernel为SIMT硬件实现。
    - SIMT模板方案：整体kernel为SIMD与SIMT混合的实现方案，对于算子中存在间接访存的load、store使用CCE模板算子通过SIMT实现，其余代码部分使用SIMD进行编译实现，可以同时利用SIMT离散访存和SIMD的计算能力。
    - SIMD甜点方案：整体kernel为SIMD的实现方案，对于存在间接访存的算子在ta层进行SIMD优化。Inductor Autotune通过编译选项在生成不同的算子进行自动选择，将单测性能最优的算子作为整网中运行的算子。
  
2. fallback模式：
    关闭离散访存，离散访存类的算子进行fallback处理。

    ![test](./arch.drawio.svg)
