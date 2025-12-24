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
    <tr id="row74795310168">
      <th class="cellrowborder" valign="top" width="11.53%" id="mcps1.2.6.1.1"><p id="p6837169903"><a name="p6837169903"></a><a name="p6837169903"></a>变更版本</p></th>
      <th class="cellrowborder" valign="top" width="37.68%" id="mcps1.2.6.1.2"><p id="p194795341619"><a name="p194795341619"></a><a name="p194795341619"></a>类名/API原型</p></th>
      <th class="cellrowborder" valign="top" width="15.22%" id="mcps1.2.6.1.3"><p id="p2479737165"><a name="p2479737165"></a><a name="p2479737165"></a>类/API类别</p></th>
      <th class="cellrowborder" valign="top" width="15.32%" id="mcps1.2.6.1.4"><p id="p3480332163"><a name="p3480332163"></a><a name="p3480332163"></a>变更类别</p></th>
      <th class="cellrowborder" valign="top" width="20.25%" id="mcps1.2.6.1.5"><p id="p448014391616"><a name="p448014391616"></a><a name="p448014391616"></a>变更说明</p></th>
    </tr>
  </thead>
  <tbody>
    <tr id="row10980145013011">
      <td class="cellrowborder" rowspan="8" valign="top" width="11.53%" headers="mcps1.2.6.1.1"><p id="p66504216149"><a name="p66504216149"></a><a name="p66504216149"></a>v2.7.1</p></td>
      <td class="cellrowborder" valign="top" width="37.68%" headers="mcps1.2.6.1.2"><p id="p14484455112513"><a name="p14484455112513"></a><a name="p14484455112513"></a>torch_npu.npu_grouped_matmul_swiglu_quant_v2</p></td>
      <td class="cellrowborder" valign="top" width="15.22%" headers="mcps1.2.6.1.3"><p id="p16483165562511"><a name="p16483165562511"></a><a name="p16483165562511"></a>自定义接口</p></td>
      <td class="cellrowborder" valign="top" width="15.32%" headers="mcps1.2.6.1.4"><p id="p4483105592518"><a name="p4483105592518"></a><a name="p4483105592518"></a>新增</p></td>
      <td class="cellrowborder" valign="top" width="20.25%" headers="mcps1.2.6.1.5"><p id="p144830558256"><a name="p144830558256"></a><a name="p144830558256"></a>新增接口</p></td>
    </tr>
    <tr id="row33324519015">
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1"><p id="p12483455122518"><a name="p12483455122518"></a><a name="p12483455122518"></a>torch_npu.npu_gelu_mul</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2"><p id="p11483555142513"><a name="p11483555142513"></a><a name="p11483555142513"></a>自定义接口</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3"><p id="p2483145517257"><a name="p2483145517257"></a><a name="p2483145517257"></a>新增</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4"><p id="p8482555132518"><a name="p8482555132518"></a><a name="p8482555132518"></a>新增接口</p></td>
    </tr>
    <tr id="row12584922559">
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1"><p id="p12151535112018"><a name="p12151535112018"></a><a name="p12151535112018"></a>torch_npu.npu_sim_exponential_</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2"><p id="p54826558250"><a name="p54826558250"></a><a name="p54826558250"></a>自定义接口</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3"><p id="p1648205512253"><a name="p1648205512253"></a><a name="p1648205512253"></a>新增</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4"><p id="p114826552254"><a name="p114826552254"></a><a name="p114826552254"></a>新增接口</p></td>
    </tr>
    <tr id="row114683432318">
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1"><p id="p036591082113"><a name="p036591082113"></a><a name="p036591082113"></a>torch_npu.npu_lightning_indexer</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2"><p id="p11481195532514"><a name="p11481195532514"></a><a name="p11481195532514"></a>自定义接口</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3"><p id="p124811355122518"><a name="p124811355122518"></a><a name="p124811355122518"></a>新增</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4"><p id="p10481175512254"><a name="p10481175512254"></a><a name="p10481175512254"></a>新增接口</p></td>
    </tr>
    <tr id="row7509143892314">
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1"><p id="p1684513210216"><a name="p1684513210216"></a><a name="p1684513210216"></a>torch_npu.npu_lightning_indexer_grad</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2"><p id="p8481855182514"><a name="p8481855182514"></a><a name="p8481855182514"></a>自定义接口</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3"><p id="p5480125512513"><a name="p5480125512513"></a><a name="p5480125512513"></a>新增</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4"><p id="p948018556251"><a name="p948018556251"></a><a name="p948018556251"></a>新增接口</p></td>
    </tr>
    <tr id="row14349614112420">
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1"><p id="p748085522516"><a name="p748085522516"></a><a name="p748085522516"></a>torch_npu.npu_sparse_flash_attention</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2"><p id="p648095562519"><a name="p648095562519"></a><a name="p648095562519"></a>自定义接口</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3"><p id="p194802558252"><a name="p194802558252"></a><a name="p194802558252"></a>新增</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4"><p id="p348035592514"><a name="p348035592514"></a><a name="p348035592514"></a>新增接口</p></td>
    </tr>
    <tr id="row119368120287">
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1"><p id="p20479175510254"><a name="p20479175510254"></a><a name="p20479175510254"></a>torch_npu.npu_sparse_flash_attention_grad</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2"><p id="p347914557256"><a name="p347914557256"></a><a name="p347914557256"></a>自定义接口</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3"><p id="p16479165519256"><a name="p16479165519256"></a><a name="p16479165519256"></a>新增</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4"><p id="p347914558259"><a name="p347914558259"></a><a name="p347914558259"></a>新增接口</p></td>
    </tr>
    <tr id="row914472315280">
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1"><p id="p11476195542514"><a name="p11476195542514"></a><a name="p11476195542514"></a>torch_npu.npu_sparse_lightning_indexer_grad_kl_loss</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2"><p id="p8476145592520"><a name="p8476145592520"></a><a name="p8476145592520"></a>自定义接口</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3"><p id="p247695512253"><a name="p247695512253"></a><a name="p247695512253"></a>新增</p></td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4"><p id="p124761255102515"><a name="p124761255102515"></a><a name="p124761255102515"></a>新增接口</p></td>
    </tr>
    <tr id="row15272836104810">
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1"><p id="p1272183616482"><a name="p1272183616482"></a><a name="p1272183616482"></a>v2.6.0</p></td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5"><p id="p1960117424480"><a name="p1960117424480"></a><a name="p1960117424480"></a>变更同v2.7.1版本</p></td>
    </tr>
    <tr id="row17772954701">
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1"><p id="p164221949101414"><a name="p164221949101414"></a><a name="p164221949101414"></a>v2.8.0</p></td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5"><p id="p10691103811319"><a name="p10691103811319"></a><a name="p10691103811319"></a>变更同v2.7.1版本</p></td>
    </tr>
    <tr id="row039910162265">
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1"><p id="p8837182917265"><a name="p8837182917265"></a><a name="p8837182917265"></a>v2.9.0</p></td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5"><p id="p1871122312615"><a name="p1871122312615"></a><a name="p1871122312615"></a>变更同v2.7.1版本</p></td>
    </tr>
  </tbody>
</table>

> [!NOTE]  
> Ascend Extension for PyTorch新增部分API支持及特性支持，具体可参考《[Ascend Extension for PyTorch 自定义API参考](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/overview.md)》或《[PyTorch 原生API支持度](https://gitcode.com/Ascend/pytorch/blob/v2.9.0-7.3.0/docs/zh/native_apis/overview.md)》。

