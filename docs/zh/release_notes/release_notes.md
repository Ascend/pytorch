# 版本说明

## 版本配套说明

### 产品版本信息

<table><tbody><tr id="row135479428341"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.1.1"><p id="p125478428345">产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.1.1 "><p id="p3547142103415"><span id="ph4778145519911">TorchNPU</span></p>
</td>
</tr>
<tr id="row11547114203412"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.2.1"><p id="p17547142103418">产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.2.1 "><p id="p2547184216342"><span id="ph1414342615376">26.1.0</span></p>
</td>
</tr>
<tr id="row854711422349"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.3.1"><p id="p354754216341">版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">正式版本</p>
</td>
</tr>
<tr id="row754461214611"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p id="p155445122062">发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p id="p135443128613">2026年7月</p>
</td>
</tr>
<tr id="row954744243418"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.5.1"><p id="p15471742193419">维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.5.1 "><p id="p1154734212344">参考<a href="https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/README.zh.md#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5">分支维护策略</a></p>
</td>
</tr>
</tbody>
</table>

### 相关产品版本配套说明

TorchNPU代码分支名称采用 **\{PyTorch版本\}-\{昇腾版本\}** 的命名规则，前者为TorchNPU匹配的PyTorch版本，后者为TorchNPU版本，详细匹配如下表：

|TorchNPU代码分支名称|PyTorch版本|TorchNPU版本|TorchNPU安装包版本|CANN版本|Python版本|
|--|--|--|--|--|--|
|v2.7.1-26.1.0|2.7.1|26.1.0|2.7.1.post8|9.1.0|Python3.9.*x*、Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.9.0-26.1.0|2.9.0|26.1.0|2.9.0.post6|9.1.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.10.0-26.1.0|2.10.0|26.1.0|2.10.0.post4|9.1.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.11.0-26.1.0|2.11.0|26.1.0|2.11.0|9.1.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.12.0-26.1.0|2.12.0|26.1.0|2.12.0|9.1.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|

## 版本兼容性说明

> [!NOTE]
>
> 表格中“Y”代表兼容，“/”代表不兼容。

<p style="display:none">
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg .tg-rhr9{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>
</p>
<table class="tg"><thead>
  <tr>
    <th class="tg-rhr9" rowspan="2">TorchNPU</th>
    <th class="tg-amwm" colspan="4">CANN版本</th>
  </tr>
  <tr>
    <th class="tg-c3ow">8.3.RC1</th>
    <th class="tg-c3ow">8.5.0</th>
    <th class="tg-c3ow">9.0.0</th>
    <th class="tg-c3ow">9.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">7.3.0</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
  </tr>
  <tr>
    <td class="tg-baqh">26.0.0</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
    <tr>
    <td class="tg-baqh">26.1.0</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
</tbody>
</table>

## 更新说明

### 新增特性

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="18.801880188018803%" id="mcps1.1.4.1.1">组件</th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2">特性</th>
      <th class="cellrowborder" valign="top" width="48.5948594859486%" id="mcps1.1.4.1.3">说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="7" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1">TorchNPU</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Eager模式使能DVM无图融合算子。</td>
      <td class="cellrowborder" valign="top" width="48.5948594859486%" headers="mcps1.1.4.1.3">Eager模式支持DVM（Device Virtual Machine）算子融合，减小调度执行开销，加速网络执行。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">NPU上支持LibTorch Stable ABI。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">NPU已实现与CUDA对齐的LibTorch Stable ABI能力。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">关键计算类API支持异构输入。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">关键计算类API支持输入为CPU上标量Tensor和NPU上非标量Tensor进行计算，提升API一致性。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">fsdp2动态图性能优化。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">提升fsdp2特性的整体性能。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">支持输入张量全轴动态场景下的算子编译。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">在图模式支持输入张量全轴动态（如 [-1,-1,-1,-1]）场景下的算子编译，增加图模式在动态shape场景的泛化性和性能。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">支持<term>Ascend 950DT</term>款型。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">将已有的TorchNPU能力在<term>Ascend 950DT</term>中进行适配。</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">将原torch_npu和Ascend Extension for PyTorch统一更名为TorchNPU。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">除导入名和Whl包名前缀保持不变外，其余部分均修改为TorchNPU。</td>
    </tr>
  </tbody>
</table>

### 删除特性

无

### 接口变更说明

本章节的接口变更说明包括新增、修改、废弃和删除。接口变更只体现代码层面的修改，不包含文档本身在语言、格式、链接等方面的优化改进。

- 新增：表示此次版本新增的接口。
- 修改：表示本接口相比于上个版本有修改。
- 废弃：表示该接口自作出废弃声明的版本起停止演进，且在声明一年后可能被移除。
- 删除：表示该接口在此次版本被移除。

**表 1** Ascend Extension for PyTorch接口变更汇总

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="11.53%" id="mcps1.2.6.1.1">变更版本</th>
      <th class="cellrowborder" valign="top" width="37.68%" id="mcps1.2.6.1.2">类名/API原型</th>
      <th class="cellrowborder" valign="top" width="15.22%" id="mcps1.2.6.1.3">类/API类别</th>
      <th class="cellrowborder" valign="top" width="15.32%" id="mcps1.2.6.1.4">变更类别</th>
      <th class="cellrowborder" valign="top" width="20.25%" id="mcps1.2.6.1.5">变更说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="36" valign="top" width="11.53%" headers="mcps1.2.6.1.1">v2.7.1</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.print_npugraph_tensor</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 8.5.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_chunk_gated_delta_rule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 8.5.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rotate_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_apply_rotary_pos_emb</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">不依赖特定的CANN版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_quant_matmul_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_alltoallv_quant_gmm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_dual_level_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_causal_conv1d</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_dynamic_block_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_masked_causal_conv1d</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mhc_post</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mhc_pre</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mhc_sinkhorn</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_transpose_quant_batchmatmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.save_npugraph_tensor</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 8.5.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scaled_masked_softmax</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_cross_entropy_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_attention_update</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gelu_mul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_top_k_top_p</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gather_sparse_index</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_swiglu_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">（beta）torch_npu.npu.finalize_dump</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">（beta）torch_npu.npu.init_dump</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">（beta）torch_npu.npu.set_dump</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.apply_rotary_pos_emb</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">（beta）torch_npu.npu_ciou</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">（beta）torch_npu.npu_iou</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_swiglu_quant_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mla_prolog_v3</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.0.0及以上版本</td>
    </tr>   
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_lightning_indexer</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_lightning_indexer</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sparse_flash_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">修改</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">依赖CANN 9.1.0版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.9.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.10.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.11.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.12.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
  </tbody>
</table>

> [!NOTE]  
> TorchNPU新增部分API支持及特性支持，具体可参考《[自定义API](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/zh/custom_APIs/overview.md)》或《[原生API](../native_apis/pytorch_2-12-0/overview.md)》。

### 已解决问题

<table><tbody><tr id="row098217197105"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.1.1"><p id="p109824198109">问题描述</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.1.1 "><p id="p9982131912103"><strong id="b59839199105">现象</strong>：GDN算子自定义适配过程中，aclnn_extension模块未传递stream参数。</p>
<p id="p15983141916104"><strong id="b1598312196108">影响</strong>：导致算子间执行顺序异常，数据同步机制未能正确生效。</p>
</td>
</tr>
<tr id="row1298311191102"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.2.1"><p id="p109831119201013">严重级别</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.2.1 "><p id="p18983019161017">一般</p>
</td>
</tr>
<tr id="row598371901017"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.3.1"><p id="p19833192101">根因分析</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.3.1 "><p id="p1798319199103">EXEC_NPU_CMD_V1_EXT与EXEC_NPU_CMD_V2_EXT中内存申请释放未配套，未与对应的EXEC_NPU_CMD_V1及EXEC_NPU_CMD_V2保持一致，导致内存申请或释放有问题，从而导致精度问题。</p>
</td>
</tr>
<tr id="row1298318191109"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.4.1"><p id="p1798321961013">解决方案</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.4.1 "><p id="p119831219181019">EXEC_NPU_CMD_V1_EXT及EXEC_NPU_CMD_V2_EXT中内存（包含workspace）申请与释放修正，与EXEC_NPU_CMD_V1及EXEC_NPU_CMD_V2保持一致。</p>
</td>
</tr>
<tr id="row1198341919103"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.5.1"><p id="p9983219181017">修改影响</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.5.1 "><p id="p15983119101017">修复后，使用GDN算子的模型整网精度正常。</p>
</td>
</tr>
</tbody>
</table>

### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

- 对业务的影响

    软件版本升级过程中会导致业务中断。

- 对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 版本配套文档

|文档名称|内容简介|更新说明|
|---|---|---|
|《[软件安装](../installation_guide/installation_description.md)》|提供在昇腾设备安装PyTorch框架训练环境，以及升级、卸载等操作。|&#8226; 新增适配PyTorch 2.11.0和PyTorch 2.12.0。<br>&#8226; 新增软件安装FAQ。<br>&#8226; 新增支持<term>Ascend 950DT</term>相关内容。 |
|《[Ascend Extension for PyTorch概述](../overview/product_overview.md)》|Ascend Extension for PyTorch（即torch_npu插件）是基于昇腾的深度学习适配框架，使昇腾NPU可以支持PyTorch框架，为PyTorch框架的使用者提供昇腾AI处理器的超强算力。|&#8226; 更新软件架构相关内容。<br>&#8226; 新增torch_npu插件启动阶段的初始化流程相关内容。<br>&#8226; 新增支持<term>Ascend 950DT</term>相关内容。 |
|《[快速入门](../quick_start/quick_start.md)》|提供了一个简单的模型迁移样例，采用了最简单的自动迁移方法，帮助用户快速体验GPU模型脚本迁移到昇腾NPU上的流程。|新增支持<term>Ascend 950DT</term>相关内容。 |
|《[torch.compile](../torch_compile/pytorch_compilation_mode.md)》|通过“动态图捕获+静态图优化+高效代码生成”的方式显著加速模型训练和推理任务。| 内容独立且优化。|
|《[配套软件库](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.1.0/zh/supported_suites_and_third_party_libraries/supported_suites_and_third_party_libraries.md)》|为Ascend Extension for PyTorch提供扩展能力的配套软件库。|&#8226; 仅保留原《套件与三方库支持清单》中的“昇腾自研插件”部分。<br>&#8226; 新增HyperParallel和AKG组件。|
|《[故障处理](../troubleshooting/troubleshooting_process.md)》|以开发者在执行推理、训练过程中可能遇到的各类异常故障现象为入口，提供自助式问题定位、问题处理方法，方便开发者快速定位并解决故障。|新增“使用NZ格式后精度异常”相关内容。|
|《[原生API](../native_apis/pytorch_2-12-0/overview.md)》|提供PyTorch 2.12.0/2.11.0/2.10.0/2.9.0/2.7.1版本原生API在昇腾设备上的支持情况。|&#8226; 新增PyTorch 2.11.0和PyTorch 2.12.0原生API支持清单。<br>&#8226; 新增支持<term>Ascend 950DT</term>相关内容。 |
|《[自定义API](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/zh/custom_APIs/overview.md)》|提供Ascend Extension for PyTorch自定义API的函数原型、功能说明、参数说明与调用示例等。|&#8226; 新增适配PyTorch 2.11.0和PyTorch 2.12.0。<br>&#8226; 新增支持<term>Ascend 950DT</term>相关内容。<br>&#8226; 具体接口变更请参考[接口变更说明](#接口变更说明)。|
|《[环境变量](../environment_variable_reference/env_variable_list.md)》|在Ascend Extension for PyTorch训练和在线推理过程中可使用的环境变量。|&#8226; 新增“TORCHINDUCTOR_USE_AKG”。<br>&#8226; 新增“（beta）TORCHINDUCTOR_ENABLE_MFUSION”。<br>&#8226; 新增“TORCH_NPU_LAZY_FUSION”。<br>&#8226; 新增“TORCH_HCCL_BLOCKING_WAIT”。<br>&#8226; 新增支持<term>Ascend 950DT</term>相关内容。|
|《[框架特性](../framework_feature_guide_pytorch/overview.md)》|基于Ascend Extension for PyTorch提供昇腾AI处理器的超强算力，从内存优化、报错定位、高性能计算等方面打造一系列独有特性。|更新“torch_npu.npu.NPUGraph”相关内容。|
|《[TorchAir](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/zh/overview.md)》|作为昇腾Ascend Extension for PyTorch的图模式能力扩展库，提供昇腾设备亲和的torch.compile图模式后端，实现PyTorch网络在昇腾NPU上的图模式推理加速和优化。|&#8226; npugraph_ex功能增强：新增支持SuperKernel融合优化功能、force_recapture功能、图捕获安全策略配置等。<br>&#8226; GE图模式功能增强：扩展npu_stream_switch接口，支持指定并发策略等。<br>&#8226; 新增支持<term>Ascend 950DT</term>相关内容。|
|《[安全声明](../security_statement/security_statement.md)》|提供了Ascend Extension for PyTorch、OpPlugin、TorchAir和Ascend Extension for TensorPipe组件的软件版本、系统加固要求、安全配置（数据存储、调试接口、运行环境等）、权限配置、防火墙等设置。|例行更新。|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-07-05 08:00:00.0|2026-07-06|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-07-06 10:03:00.0|2026-07-06|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.101156|2026-07-06|无病毒，无恶意|

### 漏洞修补列表

本版本无漏洞修复。
