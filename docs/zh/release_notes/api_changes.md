# 接口变更说明<a name="ZH-CN_TOPIC_0000002503566467"></a>

本章节的接口变更说明包括新增、修改、废弃和删除。接口变更只体现代码层面的修改，不包含文档本身在语言、格式、链接等方面的优化改进。

-   新增：表示此次版本新增的接口。
-   修改：表示本接口相比于上个版本有修改。
-   废弃：表示该接口自作出废弃声明的版本起停止演进，且在声明一年后可能被移除。
-   删除：表示该接口在此次版本被移除。

**表 1** Ascend Extension for PyTorch接口变更汇总

<a name="table14945121216428"></a>
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
      <td class="cellrowborder" rowspan="11" valign="top" width="11.53%" headers="mcps1.2.6.1.1">v2.7.1</td>
      <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.2">torch_npu.npu_grouped_matmul_swiglu_quant_v2</td>
      <td class="cellrowborder" valign="top" width="15.22%" headers="mcps1.2.6.1.3">自定义接口</td>
      <td class="cellrowborder" valign="top" width="15.32%" headers="mcps1.2.6.1.4">新增</td>
      <td class="cellrowborder" valign="top" width="20.25%" headers="mcps1.2.6.1.5">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gelu_mul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sim_exponential_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_lightning_indexer</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sparse_flash_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sparse_lightning_indexer_grad_kl_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">参数变更</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增可选入参sink</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mla_prolog_v3</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_lightning_indexer</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_kv_quant_sparse_flash_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_recurrent_gated_delta_rule_functional</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">自定义接口</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">新增</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">新增接口</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.6.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.8.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5"></a>变更同v2.7.1版本</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.9.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">变更同v2.7.1版本</td>
    </tr>
  </tbody>
</table>

> [!NOTE]  
> Ascend Extension for PyTorch新增部分API支持及特性支持，具体可参考《[Ascend Extension for PyTorch 自定义API参考](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/overview.md)》或《[PyTorch 原生API支持度](https://gitcode.com/Ascend/pytorch/blob/v2.9.0-7.3.0/docs/zh/native_apis/overview.md)》。

