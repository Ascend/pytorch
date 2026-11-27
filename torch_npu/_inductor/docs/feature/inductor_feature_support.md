# Inductor 特性支持情况

## 概述

本文汇总 Inductor-Ascend 当前支持的特性，便于在模型迁移和调优前确认能力边界。VV 融合指多个 Vector 操作之间的融合，CV 融合指 Cube 操作（如矩阵乘）与前后 Vector 操作的融合。

表中仅列出 torch_npu 当前已支持的特性。已支持不代表任意模型、形状、数据类型及后端组合均无需验证，设备型号、数据类型、Wrapper 形态及其他限制以对应特性文档为准。

## 已支持的特性

<table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse;">
  <tr>
    <th colspan="2" style="text-align:center; vertical-align:middle;">架构元素</th>
    <th style="text-align:center; vertical-align:middle;">特性</th>
    <th style="text-align:center; vertical-align:middle;">功能目标</th>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center; vertical-align:middle;">图优化</td>
    <td style="text-align:center; vertical-align:middle;">对原图做优化（等价替换/冗余消除，pattern match）</td>
    <td style="text-align:center; vertical-align:middle;">命中已启用的 FX 图优化规则时，在原图基础上做相应规则操作</td>
  </tr>
  <tr>
    <td rowspan="7" style="text-align:center; vertical-align:middle;">VV融合</td>
    <td style="text-align:center; vertical-align:middle;">离散访存</td>
    <td style="text-align:center; vertical-align:middle;">gather/scatter类</td>
    <td style="text-align:center; vertical-align:middle;">借助simt_template(simd)和simt_only，使能gather/scatter/index/index_select/index_put等</td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align:center; vertical-align:middle;">连续访存</td>
    <td style="text-align:center; vertical-align:middle;">访存线性化</td>
    <td style="text-align:center; vertical-align:middle;">非连续转连续，使能simd DMA整块搬运</td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle;">cat融合</td>
    <td style="text-align:center; vertical-align:middle;">支持pointwise_cat和concat kernel</td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align:center; vertical-align:middle;">融合规则</td>
    <td style="text-align:center; vertical-align:middle;">combo kernel（水平融合）</td>
    <td style="text-align:center; vertical-align:middle;">将满足融合条件、互不依赖的逐元素或归约算子合并到同一个 Triton Kernel，减少算子下发次数</td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle;">welford规约</td>
    <td style="text-align:center; vertical-align:middle;">在一次归约中累计计算均值和方差所需的统计量，用于优化部分归一化算子</td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align:center; vertical-align:middle;">基础能力</td>
    <td style="text-align:center; vertical-align:middle;">动态shape符号化</td>
    <td style="text-align:center; vertical-align:middle;">支持一维/多维动态shape</td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle;">AutoTiling &amp; Autotune</td>
    <td style="text-align:center; vertical-align:middle;">支持VV融合算子tiling自动生成，并能选出可执行的tiling</td>
  </tr>
  <tr>
    <td rowspan="6" style="text-align:center; vertical-align:middle;">CV融合</td>
    <td rowspan="2" style="text-align:center; vertical-align:middle;">Matmul类</td>
    <td style="text-align:center; vertical-align:middle;">基于Triton</td>
    <td style="text-align:center; vertical-align:middle;">支持mm/addmm/bmm的模板生成以及后融合（epilogue）</td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle;">基于Triton</td>
    <td style="text-align:center; vertical-align:middle;">支持mm/addmm/bmm的模板生成以及前融合（prologue）</td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align:center; vertical-align:middle;">FlexAttention</td>
    <td style="text-align:center; vertical-align:middle;">MaskMod (基于Triton)</td>
    <td style="text-align:center; vertical-align:middle;">支持用户自定义mask_mod函数，生成基于Triton的模板DSL</td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle;">ScoreMod (基于Triton)</td>
    <td style="text-align:center; vertical-align:middle;">支持用户自定义score_mod函数，生成基于Triton的模板DSL</td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align:center; vertical-align:middle;">基础能力</td>
    <td style="text-align:center; vertical-align:middle;">动态shape符号化</td>
    <td style="text-align:center; vertical-align:middle;">支持一维/多维动态shape</td>
  </tr>
  <tr>
    <td style="text-align:center; vertical-align:middle;">AutoTiling &amp; Autotune</td>
    <td style="text-align:center; vertical-align:middle;">支持CV融合算子tiling自动生成，并能选出可执行的tiling</td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align:center; vertical-align:middle;">算子调用</td>
    <td colspan="2" style="text-align:center; vertical-align:middle;">PythonWrapper</td>
    <td style="text-align:center; vertical-align:middle;">生成Triton算子launch的Python Wrapper</td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center; vertical-align:middle;">CppWrapper</td>
    <td style="text-align:center; vertical-align:middle;">生成Triton算子launch的CppWrapper代码</td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align:center; vertical-align:middle;">部署</td>
    <td colspan="2" style="text-align:center; vertical-align:middle;">MegaCache</td>
    <td style="text-align:center; vertical-align:middle;">支持保存和加载编译缓存。在硬件型号和软件编译环境一致时，可在进程或机器之间复用缓存，减少重复编译</td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center; vertical-align:middle;">AOTI</td>
    <td style="text-align:center; vertical-align:middle;">将模型和编译产物导出到pt2文件；在推理服务器上通过C++加载和推理</td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center; vertical-align:middle;">分核/限核</td>
    <td style="text-align:center; vertical-align:middle;">支持按照cube_core_num和vector_core_num进行分核/限核；支撑多实例推理，达成高吞吐</td>
  </tr>
</table>

## 使用建议

1. 表中列出的特性仍需结合目标模型确认覆盖范围，并准备相应的 fallback 路径。
2. 各特性的安装依赖、开关及使用约束请查阅对应文档，不要将“运行时可用”理解为“所有优化路径均已支持”。
