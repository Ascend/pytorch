# 接口变更说明

本章节的接口变更说明包括新增、修改、废弃和删除。接口变更只体现代码层面的修改，不包含文档本身在语言、格式、链接等方面的优化改进。

-   新增：表示此次版本新增的接口。
-   修改：表示本接口相比于上个版本有修改。
-   废弃：表示该接口自作出废弃声明的版本起停止演进，且在声明一年后可能被移除。
-   删除：表示该接口在此次版本被移除。

**表 1** Ascend Extension for PyTorch接口变更汇总

<table>
  <tr>
    <th>变更版本</th>
    <th>类名/API原型</th>
    <th>类/API类别</th>
    <th>变更类别</th>
    <th>变更说明</th>
  </tr>
  <tr>
    <th rowspan="8">v2.7.1</th>
    <th>torch_npu.npu_grouped_matmul_swiglu_quant_v2</th>
    <th>自定义接口</th>
    <th>新增</th>
    <th>新增接口</th>
  </tr>
  <tr>
    <th>torch_npu.npu_gelu_mul</th>
    <th>自定义接口</th>
    <th>新增</th>
    <th>新增接口</th>
  </tr>
  <tr>
    <th>torch_npu.npu_sim_exponential_</th>
    <th>自定义接口</th>
    <th>新增</th>
    <th>新增接口</th>
  </tr>
  <tr>
    <th>torch_npu.npu_lightning_indexer</th>
    <th>自定义接口</th>
    <th>新增</th>
    <th>新增接口</th>
  </tr>
  <tr>
    <th>torch_npu.npu_lightning_indexer_grad</th>
    <th>自定义接口</th>
    <th>新增</th>
    <th>新增接口</th>
  </tr>
  <tr>
    <th>torch_npu.npu_sparse_flash_attention</th>
    <th>自定义接口</th>
    <th>新增</th>
    <th>新增接口</th>
  </tr>
  <tr>
    <th>torch_npu.npu_sparse_flash_attention_grad</th>
    <th>自定义接口</th>
    <th>新增</th>
    <th>新增接口</th>
  </tr>
  <tr>
    <th>torch_npu.npu_sparse_lightning_indexer_grad_kl_loss</th>
    <th>自定义接口</th>
    <th>新增</th>
    <th>新增接口</th>
  </tr>
  <tr>
    <th>v2.6.0</th>
    <th colspan="4">变更同v2.7.1版本</th>
  </tr>
  <tr>
    <th>v2.8.0</th>
    <th colspan="4">变更同v2.7.1版本</th>
  </tr>
  <tr>
    <th>v2.9.0</th>
    <th colspan="4">变更同v2.7.1版本</th>
  </tr>
</table>

> [!NOTE]  
> Ascend Extension for PyTorch新增部分API支持及特性支持，具体可参考《Ascend Extension for PyTorch 自定义API参考》或《PyTorch 原生API支持度》。

