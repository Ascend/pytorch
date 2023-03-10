# AscendPyTorch

<h2 id="简介md">简介</h2>

本项目开发了PyTorch Adapter插件，用于昇腾适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。用户在准备相关环境进行基于PyTorch框架模型的开发、运行时，可以选择在服务器中手动编译相关模块。


# 文档

有关安装指南、模型迁移和训练/推理教程和API列表等更多详细信息，请参考[昇腾社区PyTorch Adapter](https://www.hiascend.com/software/ai-frameworks/commercial)。

| 文档名称                   | 文档链接                                                     |
| -------------------------- | ------------------------------------------------------------ |
| PyTorch 安装指南           | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/601/envdeployment/instg/instg_000035.html) |
| PyTorch 网络模型迁移和训练 | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/601/modeldevpt/ptmigr/ptmigr_0001.html) |
| PyTorch 在线推理           | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/601/modeldevpt/ptonlineinfer/ptonlineinfer_000001.html) |
| PyTorch 算子适配           | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/601/operatordev/operatordevg/atlasopdev_10_0081.html) |
| PyTorch API清单            | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/601/oplist/fwoperator/fwoperatorlist_0301.html) |

# 快速安装PyTorch

昇腾开发PyTorch Adapter插件用于适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力，本章节指导用户安装PyTorch框架和PyTorch Adapter插件。

**对应分支代码包下载<a name="zh-cn_topic_0000001435374593_section5248152713711"></a>**

>![](D:\project\pzr_pytorch\pytorch\figures\icon-note.gif) **说明：** 
>PyTorch配套的Python版本是：Python3.7.x（3.7.5\~3.7.11）、Python3.8.x（3.8.0\~3.8.11）、Python3.9.x（3.9.0\~3.9.2）。

安装PyTorch时，请参见[表1](#zh-cn_topic_0000001435374593_table723553621419)下载对应分支代码包。

**表 1**  Ascend配套软件

<a name="zh-cn_topic_0000001435374593_table723553621419"></a>

<table><thead align="left"><tr id="zh-cn_topic_0000001435374593_row723593618147"><th class="cellrowborder" valign="top" width="18.16%" id="mcps1.2.6.1.1"><p id="zh-cn_topic_0000001435374593_p12634164910142"><a name="zh-cn_topic_0000001435374593_p12634164910142"></a><a name="zh-cn_topic_0000001435374593_p12634164910142"></a>AscendPyTorch版本</p>
</th>
<th class="cellrowborder" valign="top" width="15.78%" id="mcps1.2.6.1.2"><p id="zh-cn_topic_0000001435374593_p7634174921415"><a name="zh-cn_topic_0000001435374593_p7634174921415"></a><a name="zh-cn_topic_0000001435374593_p7634174921415"></a>CANN版本</p>
</th>
<th class="cellrowborder" valign="top" width="17.080000000000002%" id="mcps1.2.6.1.3"><p id="zh-cn_topic_0000001435374593_p36341149111412"><a name="zh-cn_topic_0000001435374593_p36341149111412"></a><a name="zh-cn_topic_0000001435374593_p36341149111412"></a>支持PyTorch版本</p>
</th>
<th class="cellrowborder" valign="top" width="17.05%" id="mcps1.2.6.1.4"><p id="zh-cn_topic_0000001435374593_p163414971418"><a name="zh-cn_topic_0000001435374593_p163414971418"></a><a name="zh-cn_topic_0000001435374593_p163414971418"></a>代码分支名称</p>
</th>
<th class="cellrowborder" valign="top" width="31.929999999999996%" id="mcps1.2.6.1.5"><p id="p99351717112212"><a name="p99351717112212"></a><a name="p99351717112212"></a>AscendHub镜像版本/名称（<a href="https://ascendhub.huawei.com/#/index" target="_blank" rel="noopener noreferrer">获取链接</a>）</p>
</th>
</tr>
</thead>
<tbody><tr id="row1452975217222"><td class="cellrowborder" rowspan="3" valign="top" width="18.16%" headers="mcps1.2.6.1.1 "><p id="p1849173117229"><a name="p1849173117229"></a><a name="p1849173117229"></a>3.0.0</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" width="15.78%" headers="mcps1.2.6.1.2 "><p id="p13849183142218"><a name="p13849183142218"></a><a name="p13849183142218"></a>CANN 6.0.1</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.6.1.3 "><p id="p197571831182710"><a name="p197571831182710"></a><a name="p197571831182710"></a>1.5.0.post8</p>
</td>
<td class="cellrowborder" valign="top" width="17.05%" headers="mcps1.2.6.1.4 "><p id="p18655452182620"><a name="p18655452182620"></a><a name="p18655452182620"></a>v1.5.0-3.0.0</p>
</td>
<td class="cellrowborder" valign="top" width="31.929999999999996%" headers="mcps1.2.6.1.5 "><p id="p2087128165411"><a name="p2087128165411"></a><a name="p2087128165411"></a>22.0.0/pytorch-modelzoo</p>
</td>
</tr>
<tr id="row16995654102215"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p48506347263"><a name="p48506347263"></a><a name="p48506347263"></a>1.8.1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p10655452142612"><a name="p10655452142612"></a><a name="p10655452142612"></a>v1.8.1-3.0.0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p2087118825410"><a name="p2087118825410"></a><a name="p2087118825410"></a>22.0.0-1.8.1/pytorch-modelzoo</p>
</td>
</tr>
<tr id="row6631331153118"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p20696104313275"><a name="p20696104313275"></a><a name="p20696104313275"></a>1.11.0.rc2（beta)</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p156551352122620"><a name="p156551352122620"></a><a name="p156551352122620"></a>v1.11.0-3.0.0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p587113885411"><a name="p587113885411"></a><a name="p587113885411"></a>-</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001435374593_row423610361140"><td class="cellrowborder" rowspan="3" valign="top" width="18.16%" headers="mcps1.2.6.1.1 "><p id="p597121514227"><a name="p597121514227"></a><a name="p597121514227"></a>3.0.rc3</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" width="15.78%" headers="mcps1.2.6.1.2 "><p id="p69712015122215"><a name="p69712015122215"></a><a name="p69712015122215"></a>CANN 6.0.RC1</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.6.1.3 "><p id="p89718154228"><a name="p89718154228"></a><a name="p89718154228"></a>1.5.0.post7</p>
</td>
<td class="cellrowborder" valign="top" width="17.05%" headers="mcps1.2.6.1.4 "><p id="p139712015142217"><a name="p139712015142217"></a><a name="p139712015142217"></a>v1.5.0-3.0.rc3</p>
</td>
<td class="cellrowborder" valign="top" width="31.929999999999996%" headers="mcps1.2.6.1.5 "><p id="p587138125420"><a name="p587138125420"></a><a name="p587138125420"></a>22.0.RC3/pytorch-modelzoo</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001435374593_row72366365141"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p39711715202219"><a name="p39711715202219"></a><a name="p39711715202219"></a>1.8.1.rc3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p159721515162214"><a name="p159721515162214"></a><a name="p159721515162214"></a>v1.8.1-3.0.rc3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p17935111716221"><a name="p17935111716221"></a><a name="p17935111716221"></a>22.0.RC3-1.8.1/pytorch-modelzoo</p>
</td>
</tr>
<tr id="row14248651114319"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p117920554298"><a name="p117920554298"></a><a name="p117920554298"></a>1.11.0.rc1（beta)</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p10179125515298"><a name="p10179125515298"></a><a name="p10179125515298"></a>v1.11.0-3.0.rc3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p7179115512292"><a name="p7179115512292"></a><a name="p7179115512292"></a>-</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001435374593_row823643641415"><td class="cellrowborder" rowspan="2" valign="top" width="18.16%" headers="mcps1.2.6.1.1 "><p id="p8972131519223"><a name="p8972131519223"></a><a name="p8972131519223"></a>3.0.rc2</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="15.78%" headers="mcps1.2.6.1.2 "><p id="p9972915202212"><a name="p9972915202212"></a><a name="p9972915202212"></a>CANN 5.1.RC2</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.6.1.3 "><p id="p18972215192215"><a name="p18972215192215"></a><a name="p18972215192215"></a>1.5.0.post6</p>
</td>
<td class="cellrowborder" valign="top" width="17.05%" headers="mcps1.2.6.1.4 "><p id="p17972111510220"><a name="p17972111510220"></a><a name="p17972111510220"></a>v1.5.0-3.0.rc2</p>
</td>
<td class="cellrowborder" valign="top" width="31.929999999999996%" headers="mcps1.2.6.1.5 "><p id="p79355171226"><a name="p79355171226"></a><a name="p79355171226"></a>22.0.RC2/pytorch-modelzoo</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001435374593_row17236133611148"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p7972101514227"><a name="p7972101514227"></a><a name="p7972101514227"></a>1.8.1.rc2</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1897217155228"><a name="p1897217155228"></a><a name="p1897217155228"></a>v1.8.1-3.0.rc2</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p109351117122215"><a name="p109351117122215"></a><a name="p109351117122215"></a>22.0.RC2-1.8.1/pytorch-modelzoo</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001435374593_row17237123610141"><td class="cellrowborder" rowspan="2" valign="top" width="18.16%" headers="mcps1.2.6.1.1 "><p id="p169723153221"><a name="p169723153221"></a><a name="p169723153221"></a>3.0.rc1</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="15.78%" headers="mcps1.2.6.1.2 "><p id="p159721915182216"><a name="p159721915182216"></a><a name="p159721915182216"></a>CANN 5.1.RC1</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.6.1.3 "><p id="p18972815162217"><a name="p18972815162217"></a><a name="p18972815162217"></a>1.5.0.post5</p>
</td>
<td class="cellrowborder" valign="top" width="17.05%" headers="mcps1.2.6.1.4 "><p id="p79721155222"><a name="p79721155222"></a><a name="p79721155222"></a>v1.5.0-3.0.rc1</p>
</td>
<td class="cellrowborder" valign="top" width="31.929999999999996%" headers="mcps1.2.6.1.5 "><p id="p20935131722217"><a name="p20935131722217"></a><a name="p20935131722217"></a>22.0.RC1/pytorch-modelzoo</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001435374593_row0237103681420"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p897216157226"><a name="p897216157226"></a><a name="p897216157226"></a>1.8.1.rc1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1972101515223"><a name="p1972101515223"></a><a name="p1972101515223"></a>v1.8.1-3.0.rc1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p16935151722212"><a name="p16935151722212"></a><a name="p16935151722212"></a>-</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001435374593_row15574104531416"><td class="cellrowborder" valign="top" width="18.16%" headers="mcps1.2.6.1.1 "><p id="p99721715182217"><a name="p99721715182217"></a><a name="p99721715182217"></a>2.0.4</p>
</td>
<td class="cellrowborder" valign="top" width="15.78%" headers="mcps1.2.6.1.2 "><p id="p2097281518228"><a name="p2097281518228"></a><a name="p2097281518228"></a>CANN 5.0.4</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.6.1.3 "><p id="p1297218156226"><a name="p1297218156226"></a><a name="p1297218156226"></a>1.5.0.post4</p>
</td>
<td class="cellrowborder" valign="top" width="17.05%" headers="mcps1.2.6.1.4 "><p id="p15973915192218"><a name="p15973915192218"></a><a name="p15973915192218"></a>2.0.4.tr5</p>
</td>
<td class="cellrowborder" valign="top" width="31.929999999999996%" headers="mcps1.2.6.1.5 "><p id="p193515171225"><a name="p193515171225"></a><a name="p193515171225"></a>21.0.4/pytorch-modelzoo</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001435374593_row14742104715146"><td class="cellrowborder" valign="top" width="18.16%" headers="mcps1.2.6.1.1 "><p id="p149734151227"><a name="p149734151227"></a><a name="p149734151227"></a>2.0.3</p>
</td>
<td class="cellrowborder" valign="top" width="15.78%" headers="mcps1.2.6.1.2 "><p id="p1397317155220"><a name="p1397317155220"></a><a name="p1397317155220"></a>CANN 5.0.3</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.6.1.3 "><p id="p79734156229"><a name="p79734156229"></a><a name="p79734156229"></a>1.5.0.post3</p>
</td>
<td class="cellrowborder" valign="top" width="17.05%" headers="mcps1.2.6.1.4 "><p id="p997341592219"><a name="p997341592219"></a><a name="p997341592219"></a>2.0.3.tr5</p>
</td>
<td class="cellrowborder" valign="top" width="31.929999999999996%" headers="mcps1.2.6.1.5 "><p id="p193551715225"><a name="p193551715225"></a><a name="p193551715225"></a>21.0.3/pytorch-modelzoo</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001435374593_row17748104281412"><td class="cellrowborder" valign="top" width="18.16%" headers="mcps1.2.6.1.1 "><p id="p797371515225"><a name="p797371515225"></a><a name="p797371515225"></a>2.0.2</p>
</td>
<td class="cellrowborder" valign="top" width="15.78%" headers="mcps1.2.6.1.2 "><p id="p20973615192217"><a name="p20973615192217"></a><a name="p20973615192217"></a>CANN 5.0.2</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.6.1.3 "><p id="p1997314151227"><a name="p1997314151227"></a><a name="p1997314151227"></a>1.5.0.post2</p>
</td>
<td class="cellrowborder" valign="top" width="17.05%" headers="mcps1.2.6.1.4 "><p id="p697311512219"><a name="p697311512219"></a><a name="p697311512219"></a>2.0.2.tr5</p>
</td>
<td class="cellrowborder" valign="top" width="31.929999999999996%" headers="mcps1.2.6.1.5 "><p id="p693510174221"><a name="p693510174221"></a><a name="p693510174221"></a>21.0.2/pytorch-modelzoo</p>
</td>
</tr>
</tbody>
</table>

**安装PyTorch环境依赖<a name="section311512324315"></a>**

执行如下命令安装。如果使用非root用户安装，需要在命令后加**--user**，例如：**pip3 install pyyaml --user，pip3 install wheel --user**。

```
pip3 install pyyaml
pip3 install wheel
```

**安装PyTorch<a name="section1762728142316"></a>**

推荐用户使用编好的二进制whl包安装PyTorch 1.11.0。用户也可选择编译安装方式安装PyTorch 1.11.0。请参考[编译安装PyTorch](#使用源码编译安装PyTorch框架)。

1. 安装官方torch包。

   - x86\_64

     ```
     pip3 install torch==1.11.0+cpu  
     ```
     
     若执行以上命令安装cpu版本PyTorch报错，请点击下方PyTorch官方链接下载whl包安装。
     
     PyTorch 1.11.0版本：[下载链接](https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp37-cp37m-linux_x86_64.whl)。
     
   - aarch64
   
     1. 进入安装目录，执行如下命令获取鲲鹏文件共享中心上对应版本的whl包。
   
        ```
        wget https://repo.huaweicloud.com/kunpeng/archive/Ascend/PyTorch/torch-1.11.0-cp37-cp37m-linux_aarch64.whl
        ```
   
     2. 执行如下命令安装，如果使用非root用户安装，需要在命令后加**--user。**
   
         ```
         pip3 install torch-1.11.0-cp37-cp37m-linux_aarch64.whl
         ```


2. 安装PyTorch插件torch\_npu。以下命令以在aarch64架构下安装为例。

   1. 进入安装目录，执行如下命令获取PyTorch插件的whl包。

      ```
      # 若用户在x86架构下安装插件，请将命令中文件包名中的“aarch64”改为“x86_64”。
      wget https://gitee.com/ascend/pytorch/releases/download/v3.0.0-pytorch1.11.0/torch_npu-1.11.0rc2-cp37-cp37m-linux_aarch64.whl
      ```
      
   2. 执行如下命令安装。如果使用非root用户安装，需要在命令后加**--user。**
   
      ```
      # 若用户在x86架构下安装插件，请将命令中文件包名中的“aarch64”改为“x86_64”。
      pip3 install torch_npu-1.11.0rc2-cp37-cp37m-linux_aarch64.whl
      ```
   
3. 安装对应框架版本的torchvision。

   ```
   #PyTorch 1.11.0需安装0.12.0版本
   pip3 install torchvision==0.9.1   
   ```

**安装APEX混合精度模块<a name="section154215015416"></a>**

混合精度训练是在训练时混合使用单精度（float32）与半精度\(float16\)数据类型，将两者结合在一起，并使用相同的超参数实现了与float32几乎相同的精度。在迁移完成、训练开始之前，基于NPU芯片的架构特性，用户需要开启混合精度，可以提升模型的性能。APEX混合精度模块是一个集优化性能、精度收敛于一身的综合优化库，可以提供不同场景下的混合精度训练支持。APEX模块的使用介绍可参考《[PyTorch 网络模型迁移和训练指南](https://www.hiascend.com/document/detail/zh/canncommercial/601/modeldevpt/ptmigr/ptmigr_0001.html)》中的“混合精度说明“章节。

请参见[apex: Ascend apex adapter - Gitee.com](https://gitee.com/ascend/apex/tree/v1.11.0/)安装混合精度模块。

# 建议与交流

热忱希望各位在用户社区加入讨论，并贡献您的建议，我们会尽快给您回复。

# 分支维护策略

Ascend PyTorch的版本分支有以下几种维护阶段：

| **状态**          | **持续时间**  | **说明**                                           |
| ----------------- | ------------- | -------------------------------------------------- |
| Planning          | 1 - 3 months  | 特性规划。                                         |
| Development       | 3 months      | 特性开发。                                         |
| Maintained        | 6 - 12 months | 允许所有问题修复的合入，并发布版本。               |
| Unmaintained      | 0 - 3 months  | 允许所有问题修复的合入，无专人维护，不再发布版本。 |
| End Of Life (EOL) | N/A           | 不再接受修改合入该分支。                           |

# 现有分支维护状态

| **分支名**   | **当前状态** | **上线时间** | **后续状态**                           | **EOL 日期** |
| ------------ | ------------ | ------------ | -------------------------------------- | ------------ |
| **v2.0.2**   | EOL          | 2021-07-29   | N/A                                    |              |
| **v2.0.3**   | EOL          | 2021-10-15   | N/A                                    |              |
| **v2.0.4**   | Unmaintained | 2022-01-15   | EOL <br> 2023-04-15 estimated          |              |
| **v3.0.rc1** | Maintained   | 2022-04-10   | Unmaintained <br> 2023-04-10 estimated |              |
| **v3.0.rc2** | Maintained   | 2022-07-15   | Unmaintained <br> 2023-07-15 estimated |              |
| **v3.0.rc3** | Maintained   | 2022-10-20   | Unmaintained <br> 2023-10-20 estimated |              |
| **v3.0.0**   | Maintained   | 2023-1-18    | Unmaintained <br> 2024-1-18 estimated  |              |

# FAQ

## 使用PyTorch原生框架在ARM CPU上算子计算结果异常

当前使用的PyTorch官方原生框架在ARM CPU上运行时，算子计算结果会出现异常，此问题为原生框架社区的已知问题，详细内容可参考PyTorch官方社区[ISSUE](https://github.com/pytorch/pytorch/issues/75411)。

可通过以下方式解决：

- 修改算子输入数据类型，使用float64数据类型进行运算。
- 升级编译arm版本PyTorch使用的gcc编译器至9.4版本及以上，并使用相同编译器重新编译torch_npu、apex、mmcv等其他配套软件（避免因编译器版本不匹配导致兼容性问题）。

## 使用源码编译安装PyTorch框架。

**安装依赖<a name="section1832663918540"></a>**

选择编译安装方式安装时需要安装系统依赖。此处以CentOS与Ubuntu操作系统为例目前支持CentOS与Ubuntu操作系统。

EulerOS、OpenEuler、BCLinux、Kylin、UOS20 1020e系统可参考CentOS进行安装。

Debian、UOS20、UOS20 SP1、Linx系统可参考Ubuntu进行安装。

- CentOS

  ```
  yum install -y patch zlib-devel libffi-devel openssl-devel libjpeg-turbo-devel gcc-c++ sqlite-devel dos2unix openblas git 
  yum install -y gcc==7.5.0 cmake==3.12.0 #gcc7.5.0版本及以上，cmake3.12.0版本及以上。
  ```

- Ubuntu

  ```
  apt-get install -y patch g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev m4 dos2unix libopenblas-dev git 
  apt-get install -y gcc==7.5.0 cmake==3.12.0 #gcc7.5.0版本及以上，cmake3.12.0版本及以上。
  ```

**安装1.11.0<a name="section462918122313"></a>**

以下操作步骤以安装PyTorch 1.11.0版本为例。

1. 安装官方torch包。

   - x86\_64

     ```
     pip3 install torch==1.11.0+cpu  
     ```

     若执行以上命令安装cpu版本PyTorch报错，请点击下方PyTorch官方链接下载whl包安装。

     PyTorch 1.11.0版本：[下载链接](https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp37-cp37m-linux_x86_64.whl)。

   - 在aarch64架构下，用户可以选择编译安装官方torch包。

     1. 下载PyTorch v1.11.0源码包。

        ```
     	git clone -b v1.11.0 https://github.com/pytorch/pytorch.git --depth=1 pytorch_v1.11.0
        ```
     
     2. 进入源码包获取被动依赖代码。

        ```
     	cd pytorch_v1.11.0
        git submodule sync
        git submodule update --init --recursive
        ```
     
     3. 配置环境变量。

        ```
     	export USE_XNNPACK=0
        ```
     
     4. 执行编译安装。

        ```
     	python3 setup.py install
        ```

**安装torch_npu**


1. 编译生成PyTorch插件的二进制安装包。

   ```
   # 下载对应PyTorch版本分支代码，进入插件根目录，以v1.11.0-3.0.0为例
   git clone -b  v1.11.0-3.0.0 https://gitee.com/ascend/pytorch.git 
   cd pytorch    
   # 指定Python版本编包方式，以Python3.7为例，其他Python版本请使用 --python=3.8或--python3.9
   bash ci/build.sh --python=3.7
   ```

2. 安装pytorch/dist目录下生成的插件torch\_npu包，如果使用非root用户安装，需要在命令后加**--user**。

   ```
   pip3 install --upgrade dist/torch_npu-1.11.0-cp37-cp37m-linux_aarch64.whl
   # 若用户在x86架构下安装插件，请替换为对应的whl包。
   ```

3. 安装对应框架版本的torchvision。

   ```
   #PyTorch 1.11.0需安装0.12.0版本
   pip3 install torchvision==0.9.1   
   ```

4. 配置环境变量，验证是否安装成功。

   1. 配置CANN环境变量脚本。

      ```
      source <CANN软件安装目录>/<CANN软件路径>/set_env.sh
      ```

      环境变量脚本的默认路径一般为：/usr/local/Ascend/ascend-toolkit/set_env.sh，其中ascend-toolkit路径取决于安装的CANN软件名称。

   2. 执行单元测试脚本，验证PyTorch是否安装成功。

      ```
      cd test/test_network_ops/
      python3 test_div.py
      ```

      结果显示OK证明PyTorch框架与插件安装成功。

## 在PIP设置为华为源时，安装requirements.txt中的typing依赖后，会导致python环境错误。

在PIP设置为华为源时，需打开requirements.txt文件，删除typing依赖，再执行命令。

```
pip3 install -r requirements.txt
```

## 编译过程执行bash build.sh报错no module named yaml/typing_extensions.

pytorch编译依赖 yaml库和typing_extensions库，需要手动安装。

```
pip3 install pyyaml

pip3 install typing_extensions
```

安装成功后，注意需要执行make clean在执行bash build.sh进行编译，否则可能因缓存出现未知编译错误。

## 运行遇到找不到te问题

开发态:

```
cd /urs/local/Ascend/ascend-toolkit/latest/{arch}-linux/lib64  #{arch}为架构名称

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl
```

用户态:

```
cd /urs/local/Ascend/nnae/latest/{arch}-linux/lib64  #{arch}为架构名称

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl
```

## 命令行安装cmake依赖时提示找不到包、编译cmake报错版本过低，可使用安装脚本或源码编译安装。

方法一：下载安装脚本安装cmake。（参考cmake官网）

​		X86_64环境脚本安装：cmake-3.12.0-Linux-x86_64.sh

​		aarch64环境脚本安装：cmake-3.12.0-Linux-aarch64.sh

1. 执行命令。

   ```
   ./cmake-3.12.0-Linux-{arch}.sh #{arch}为架构名称
   ```

2. 设置软连接。

   ```
   ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake
   ```

3. 执行如下命令验证是否安装成功。

   ```
   cmake --version
   ```

   如显示“cmake version 3.12.0”则表示安装成功。


方法二：使用源码编译安装。

1. 获取cmake软件包。

   ```
   wget https://cmake.org/files/v3.12/cmake-3.12.0.tar.gz --no-check-certificate
   ```

2. 解压并进入软件包目录。

   ```
   tar -xf cmake-3.12.0.tar.gz
   cd cmake-3.12.0/
   ```

3. 执行配置、编译和安装命令。

   ```
   ./configure --prefix=/usr/local/cmake
   make && make install
   ```

4. 设置软连接。

   ```
   ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake
   ```

5. 执行如下命令验证是否安装成功。

   ```
   cmake --version
   ```

   如显示“cmake version 3.12.0”则表示安装成功。

## 命令行安装gcc依赖时提示找不到包、编译时gcc报错问题

部分源下载gcc时会提示无法找到包，需要使用源码编译安装。

以下步骤请在root用户下执行。

1. 下载gcc-7.5.0.tar.gz，下载地址为[https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz](https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz)。

2. 安装gcc时候会占用大量临时空间，所以先执行下面的命令清空/tmp目录：

   ```
   sudo rm -rf /tmp/*
   ```

3. 安装依赖（以CentOS和Ubuntu系统为例）。

   - CentOS执行如下命令安装。

     ```
     yum install bzip2    
     ```

   - Ubuntu执行如下命令安装。

     ```
     apt-get install bzip2    
     ```

4. 编译安装gcc。

   1. 进入gcc-7.5.0.tar.gz源码包所在目录，解压源码包，命令为：

      ```
      tar -zxvf gcc-7.5.0.tar.gz
      ```

   2. 进入解压后的文件夹，执行如下命令下载gcc依赖包：

      ```
      cd gcc-7.5.0
      ./contrib/download_prerequisites
      ```

      如果执行上述命令报错，需要执行如下命令在“gcc-7.5.0/“文件夹下下载依赖包：

      ```
      wget http://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2
      wget http://gcc.gnu.org/pub/gcc/infrastructure/mpfr-3.1.4.tar.bz2
      wget http://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.0.3.tar.gz
      wget http://gcc.gnu.org/pub/gcc/infrastructure/isl-0.16.1.tar.bz2
      ```

      下载好上述依赖包后，重新执行以下命令：

      ```
      ./contrib/download_prerequisites
      ```

      如果命令校验失败，需要确认上述依赖包在文件夹中的唯一性，无重复下载，若存在重复的依赖包，需删除。

   3. <a name="zh-cn_topic_0000001135347812_zh-cn_topic_0000001173199577_zh-cn_topic_0000001172534867_zh-cn_topic_0276688294_li1649343041310"></a>执行配置、编译和安装命令：

      ```
      ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/linux_gcc7.5.0
      make -j15    # 通过grep -w processor /proc/cpuinfo|wc -l查看cpu数，示例为15，用户可自行设置相应参数。
      make install    
      ```

      >![](figures/icon-notice.gif) **须知：** 
      >其中“--prefix“参数用于指定linux\_gcc7.5.0安装路径，用户可自行配置，但注意不要配置为“/usr/local“及“/usr“，因为会与系统使用软件源默认安装的gcc相冲突，导致系统原始gcc编译环境被破坏。示例指定为“/usr/local/linux\_gcc7.5.0“。

   4. 修改软连接。

      ```
      ln -s ${install_path}/gcc-7.5.0/bin/gcc /usr/bin/gcc
      ln -s ${install_path}/gcc-7.5.0/bin/g++ /usr/bin/g++
      ln -s ${install_path}/gcc-7.5.0/bin/c++ /usr/bin/c++
      ```

   5.配置环境变量。

   当用户执行训练时，需要用到gcc升级后的编译环境，因此要在训练脚本中配置环境变量，通过如下命令配置。

   ```
   export LD_LIBRARY_PATH=${install_path}/lib64:${LD_LIBRARY_PATH}
   ```

   其中$\{install\_path\}为[3.](#zh-cn_topic_0000001135347812_zh-cn_topic_0000001173199577_zh-cn_topic_0000001172534867_zh-cn_topic_0276688294_li1649343041310)中配置的gcc7.5.0安装路径，本示例为“/usr/local/gcc7.5.0/“。

   >![](figures/icon-note.gif) **说明：** 
   >本步骤为用户在需要用到gcc升级后的编译环境时才配置环境变量。

若存在pytorch编译不过，请检查软连接的库是否正确。


libstdc++->libstdc++.so.6.0.24(7.5.0)

## 找不到libblas.so问题

环境缺少openblas库，需要安装openblas库

Centos，EulerOS环境

```sh
yum -y install openblas
```

Ubuntu环境

```sh
apt install libopenblas-dev
```

## 容器中未挂载device问题

在容器中运行脚本出现NPU相关ERROR。由于启动容器实例时，未挂载device参数，导致无法正常启动实例。

![](figures/FAQ.png)

请用户参考以下命令，重启容器。

```sh
docker run -it --ipc=host \
--device=/dev/davinciX \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver \
-v /usr/local/dcmi \
-v /usr/local/bin/npu-smi \
${镜像名称}:{tag} \
/bin/bash
```

参数说明：

/dev/davinciX：NPU设配，X是芯片物理ID号例如davinci0。

/dev/davinci_manager：管理设备。

/dev/devmm_svm：管理设备。

/dev/hisi_hdc：管理设备。

/usr/local/Ascend/driver：驱动目录。

/usr/local/dcmi：DCMI目录。

/usr/local/bin/npu-smi：npu-smi工具。

${镜像名称}:{tag}：镜像名称与版本号。

## 安装-torch--whl-提示-torch-1-5-0xxxx-与-torchvision-所依赖的版本不匹配

安装“torch-\*.whl”时，提示"ERROR：torchvision 0.6.0 has requirement torch==1.5.0, but you'll have torch 1.5.0a0+1977093 which is incompatible"。
![](figures/zh-cn_image_0000001190081735.png)

安装torch时，会自动触发torchvision进行依赖版本检查，环境中安装的torchvision版本为0.6.0，检查时发现我们安装的torch-\*.whl的版本号与要求的1.5.0不一致，所以提示报错，但实际安装成功 。

对实际结果无影响，无需处理。

## import torch_npu 显示_has_compatible_shallow_copy_type重复注册warning问题

warning如下图所示，由Tensor.set_data浅拷贝操作触发。主要原因是PyTorch插件化解耦后，`_has_compatible_shallow_copy_type`缺乏对NPU Tensor的浅拷贝判断支持，因此需要重新注册`_has_compatible_shallow_copy_type`。

该warning不影响模型的精度和性能，可以忽略。

待NPU 设备号合入社区或者后续PyTorch版本`_has_compatible_shallow_copy_type`注册方式发生变动，该warning会被解决。

![输入图片说明](https://images.gitee.com/uploads/images/2022/0701/153621_2b5080c4_7902902.png)

## 在编译torch_npu的目录进入python引用torch_npu报错问题

验证torch_npu的引入，请切换至其他目录进行，在编译目录执行会提示如下错误。

<img src="figures/FAQ torch_npu.png" style="zoom:150%;" />

## 在执行import torch_npu时出现ModuleNotFooundError: NO module named '_lzma'报错问题

在python命令行下，执行import torch_npu测试时，出现ModuleNotFooundError: NO module named '_lzma'问题，可能由于Python环境失效，重装Python即可。<img src="figures/QA.png"  />

## 编译过程中出现XNNPACK相关的Make Error报错

编译原生pytorch时，未配置相关环境变量，导致编译不成功。

![](figures/QA1.png)

1. 执行命令设置环境变量

   ```
   export USE_XNNPACK=0
   ```

2. 执行命令清除当前编译内容

   ```
   make clean
   ```

3. 重新编译

## 编译时出现Breakpad error: field 'regs' has incomplete type 'google_breakpad::user_regs_struct'报错

编译原生pytorch时，未配置相关环境变量，导致编译不成功。

1. 执行命令配置环境变量

   ```
   export BUILD_BREAKPAD=0
   ```

2. 执行命令清除当前编译内容

   ```
   make clean
   ```

3. 重新编译

## 多卡训练初始化阶段卡顿至超时

init_process_group 函数中使用了IPV6地址，例如::1(注意localhost 可能指向IPv6的地址)，使用IPv4可以避免这个问题

# 版本说明

版本说明请参阅[ReleseNote](docs/zh/RELEASENOTE)