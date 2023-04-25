# FrameworkPTAdapter 5.0.RC1 版本说明书
-   [FrameworkPTAdapter 5.0.RC1](#FrameworkPTAdapter-5-0-RC1md)
    -   [用户须知](#用户须知md)
    -   [新增特性](#新增特性md)
    -   [特性修改](#特性修改md)
    -   [已修复问题](#已修复问题md)
    -   [已知问题](#已知问题md)
    -   [兼容性](#兼容性md)
-   [FrameworkPTAdapter 3.0.0](#FrameworkPTAdapter-3-0-0md)
    -   [用户须知](#用户须知md)
    -   [新增特性](#新增特性md)
    -   [特性修改](#特性修改md)
    -   [已修复问题](#已修复问题md)
    -   [已知问题](#已知问题md)
    -   [兼容性](#兼容性md)
-   [FrameworkPTAdapter 3.0.RC3](#FrameworkPTAdapter-3-0-RC3md)
    -   [用户须知](#用户须知md)
    -   [新增特性](#新增特性md)
    -   [特性修改](#特性修改md)
    -   [已修复问题](#已修复问题md)
    -   [已知问题](#已知问题md)
    -   [兼容性](#兼容性md)
-   [FrameworkPTAdapter 3.0.RC2](#FrameworkPTAdapter-3-0-RC2md)
    -   [用户须知](#用户须知md)
    -   [新增特性](#新增特性md)
    -   [特性修改](#特性修改md)
    -   [已修复问题](#已修复问题md)
    -   [已知问题](#已知问题md)
    -   [兼容性](#兼容性md)
-   [FrameworkPTAdapter 3.0.RC1](#FrameworkPTAdapter-3-0-RC1md)
    -   [用户须知](#用户须知md)
    -   [新增特性](#新增特性md)
    -   [特性修改](#特性修改md)
    -   [已修复问题](#已修复问题md)
    -   [已知问题](#已知问题md)
    -   [兼容性](#兼容性md)
-   [FrameworkPTAdapter 2.0.4](#FrameworkPTAdapter-2-0-4md)
    -   [用户须知](#用户须知md)
    -   [新增特性](#新增特性md)
    -   [特性修改](#特性修改md)
    -   [已修复问题](#已修复问题md)
    -   [已知问题](#已知问题md)
    -   [兼容性](#兼容性md)
-   [FrameworkPTAdapter 2.0.3](#FrameworkPTAdapter-2-0-3md)
    -   [用户须知](#用户须知md)
    -   [新增特性](#新增特性md)
    -   [特性修改](#特性修改md)
    -   [已修复问题](#已修复问题md)
    -   [已知问题](#已知问题md)
    -   [兼容性](#兼容性md)
-   [FrameworkPTAdapter 2.0.2](#FrameworkPTAdapter-2-0-2md)
    -   [用户须知](#用户须知-0md)
    -   [新增特性](#新增特性-1md)
    -   [特性修改](#特性修改-2md)
    -   [已修复问题](#已修复问题-3md)
    -   [已知问题](#已知问题-4md)
    -   [兼容性](#兼容性-5md)

<h2 id="FrameworkPTAdapter-5-0-RC1md">FrameworkPTAdapter 5.0.RC1</h2>

<h3 id="用户须知md">用户须知</h3>

本框架基于Facebook主导的开源PyTorch进行修改，延续原生的PyTorch特性，使用NPU进行动态图训练；以算子粒度进行模型适配，代码重用性好，支持现有的网络只修改设备类型或数据类型，即可迁移到NPU上使用。

PyTorch1.8.1版本开始采用插件化适配方式，与原生PyTorch框架实现彻底解耦，对用户安装使用及后端算子适配开发提供较好体验。

<h3 id="新增特性md">新增特性</h3>

**表 1** PyTorch支持的版本特性列表

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>
<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="25.590000000000003%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>一级特性</p>
</th>
<th class="cellrowborder" valign="top" width="15.52%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>二级特性</p>
</th>
<th class="cellrowborder" valign="top" width="58.89%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row2620183971614"><td class="cellrowborder" rowspan="3" valign="top" width="25.590000000000003%" headers="mcps1.2.4.1.1 "><p id="p0819102247"><a name="p0819102247"></a><a name="p0819102247"></a>适配NPU的PyTorch1.8.1特性</p>
<p id="p15488161812213"><a name="p15488161812213"></a><a name="p15488161812213"></a></p>
<p id="p17381229135615"><a name="p17381229135615"></a><a name="p17381229135615"></a></p>
</td>
<td class="cellrowborder" valign="top" width="15.52%" headers="mcps1.2.4.1.2 "><p id="p76365489137"><a name="p76365489137"></a><a name="p76365489137"></a>框架基础功能</p>
</td>
<td class="cellrowborder" valign="top" width="58.89%" headers="mcps1.2.4.1.3 "><p id="p363616485131"><a name="p363616485131"></a><a name="p363616485131"></a>优化设备内存分配策略，修复少量BUG</p>
</td>
</tr>
<tr id="row945906124515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1077934311314"><a name="p1077934311314"></a><a name="p1077934311314"></a>FX</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p3634127577"><a name="p3634127577"></a><a name="p3634127577"></a>支持模型IR图构建以及基于IR图的代码生成</p>
</td>
</tr>
<tr id="row938172915567"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p6466153955816"><a name="p6466153955816"></a><a name="p6466153955816"></a>优化器</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p20382294568"><a name="p20382294568"></a><a name="p20382294568"></a>支持npu_fused_adam等融合优化器</p>
</td>
</tr>
<tr id="row3722227133312"><td class="cellrowborder" rowspan="3" valign="top" width="25.590000000000003%" headers="mcps1.2.4.1.1 "><p id="p107221327153315"><a name="p107221327153315"></a><a name="p107221327153315"></a>适配NPU的PyTorch1.11.0特性</p>
<p id="p7778931115613"><a name="p7778931115613"></a><a name="p7778931115613"></a></p>
</td>
<td class="cellrowborder" valign="top" width="15.52%" headers="mcps1.2.4.1.2 "><p id="p153563917719"><a name="p153563917719"></a><a name="p153563917719"></a>框架基础功能</p>
</td>
<td class="cellrowborder" valign="top" width="58.89%" headers="mcps1.2.4.1.3 "><p id="p193246215619"><a name="p193246215619"></a><a name="p193246215619"></a>优化设备内存分配策略，修复少量BUG</p>
</td>
</tr>
<tr id="row6631141014305"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1416445591310"><a name="p1416445591310"></a><a name="p1416445591310"></a>FX</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1281850105718"><a name="p1281850105718"></a><a name="p1281850105718"></a>支持模型IR图构建以及基于IR图的代码生成</p>
</td>
</tr>
<tr id="row577853110564"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p546144505813"><a name="p546144505813"></a><a name="p546144505813"></a>优化器</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p47781431165614"><a name="p47781431165614"></a><a name="p47781431165614"></a>支持npu_fused_adam等融合优化器</p>
</td>
</tr>
</tbody>
</table>
<h3 id="特性修改md">特性修改</h3>

不涉及

<h3 id="已修复问题md">已修复问题</h3>

不涉及

<h3 id="已知问题md">已知问题</h3>

<a name="table1969972073016"></a>
<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>已知问题</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>问题描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>数据类型支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPU不支持float16类型的inf/nan数据输入输出。</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>数据Format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>出现4D以上的format时不能降维。</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>集合通信约束</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>要求一次训练任务中不同device上执行的图相同。</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>当前只支持1/2/4/8P粒度的分配。</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>只支持int8，int32，float16和float32数据类型。</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex功能支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>Apex当前版本的实现方式主要为python实现，不支持APEX中的自定义优化CUDA Kernel。</p>
</td>
</tr>
</tbody>
</table>
<h3 id="兼容性md">兼容性</h3>

A800-9010：CentOS 7.6/Ubuntu 18.04, 2.04/BC-Linux 7.6/Debian 9.9/Debian 10/OpenEuler 20.03 LTS

A800-9000：CentOS 7.6/Ubuntu 18.04, 2.04/Euler 2.8, 2.10/Kylin v10/BC-Linux 7.6/OpenEuler 20.03 LTS/UOS 20 1020e

<h2 id="FrameworkPTAdapter-3-0-0md">FrameworkPTAdapter 3.0.0</h2>

<h3 id="用户须知md">用户须知</h3>

本框架基于Facebook主导的开源PyTorch进行修改，延续原生的PyTorch特性，使用NPU进行动态图训练；以算子粒度进行模型适配，代码重用性好，支持现有的网络只修改设备类型或数据类型，即可迁移到NPU上使用。

PyTorch1.8.1版本开始采用插件化适配方式，与原生PyTorch框架实现彻底解耦，功能、性能与PyTorch1.5.0基本保持一致，对后端算子适配提供较好开发体验。PyTorch 1.11.0当前为beta版本，建议优先使用PyTorch 1.8.1版本。

<h3 id="新增特性md">新增特性</h3>

**表 1** PyTorch支持的版本特性列表

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>

<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="25.590000000000003%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>一级特性</p>
</th>
<th class="cellrowborder" valign="top" width="15.52%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>二级特性</p>
</th>
<th class="cellrowborder" valign="top" width="58.89%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row91681125173610"><td class="cellrowborder" valign="top" width="25.590000000000003%" headers="mcps1.2.4.1.1 "><p id="p1712174492611"><a name="p1712174492611"></a><a name="p1712174492611"></a>适配NPU的PyTorch1.5.0特性</p>
</td>
<td class="cellrowborder" valign="top" width="15.52%" headers="mcps1.2.4.1.2 "><p id="p2168172519364"><a name="p2168172519364"></a><a name="p2168172519364"></a>框架基础功能</p>
</td>
<td class="cellrowborder" valign="top" width="58.89%" headers="mcps1.2.4.1.3 "><p id="p1416815259367"><a name="p1416815259367"></a><a name="p1416815259367"></a>适配CANN算子IR变更。</p>
</td>
</tr>
<tr id="row2620183971614"><td class="cellrowborder" rowspan="3" valign="top" width="25.590000000000003%" headers="mcps1.2.4.1.1 "><p id="p0819102247"><a name="p0819102247"></a><a name="p0819102247"></a>适配NPU的PyTorch1.8.1特性</p>
<p id="p15488161812213"><a name="p15488161812213"></a><a name="p15488161812213"></a></p>
</td>
<td class="cellrowborder" valign="top" width="15.52%" headers="mcps1.2.4.1.2 "><p id="p76365489137"><a name="p76365489137"></a><a name="p76365489137"></a>框架基础功能</p>
</td>
<td class="cellrowborder" valign="top" width="58.89%" headers="mcps1.2.4.1.3 "><p id="p363616485131"><a name="p363616485131"></a><a name="p363616485131"></a>支持NPU Tensor类型，修复少量BUG</p>
</td>
</tr>
<tr id="row945906124515"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1077934311314"><a name="p1077934311314"></a><a name="p1077934311314"></a>序列化存取</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p977954315133"><a name="p977954315133"></a><a name="p977954315133"></a>支持NPU Tensor直接序列化存取，完善torch.save/load接口能力</p>
</td>
</tr>
<tr id="row13545374516"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p44561110214"><a name="p44561110214"></a><a name="p44561110214"></a>精度对比工具</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p412459182110"><a name="p412459182110"></a><a name="p412459182110"></a>支持溢出检测，优化数据dump性能</p>
</td>
</tr>
<tr id="row3722227133312"><td class="cellrowborder" rowspan="3" valign="top" width="25.590000000000003%" headers="mcps1.2.4.1.1 "><p id="p107221327153315"><a name="p107221327153315"></a><a name="p107221327153315"></a>适配NPU的PyTorch1.11.0特性</p>
</td>
<td class="cellrowborder" valign="top" width="15.52%" headers="mcps1.2.4.1.2 "><p id="p153563917719"><a name="p153563917719"></a><a name="p153563917719"></a>框架基础功能</p>
</td>
<td class="cellrowborder" valign="top" width="58.89%" headers="mcps1.2.4.1.3 "><p id="p272242719336"><a name="p272242719336"></a><a name="p272242719336"></a>支持NPU Tensor类型，修复少量BUG</p>
</td>
</tr>
<tr id="row6631141014305"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1416445591310"><a name="p1416445591310"></a><a name="p1416445591310"></a>序列化存取</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p11164185561312"><a name="p11164185561312"></a><a name="p11164185561312"></a>支持NPU Tensor直接序列化存取，完善torch.save/load接口能力</p>
</td>
</tr>
<tr id="row270811563411"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p121645556134"><a name="p121645556134"></a><a name="p121645556134"></a>精度对比工具</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1716425591313"><a name="p1716425591313"></a><a name="p1716425591313"></a>支持溢出检测，优化数据dump性能</p>
</td>
</tr>
</tbody>
</table>

<h3 id="特性修改md">特性修改</h3>

不涉及

<h3 id="已修复问题md">已修复问题</h3>

不涉及

<h3 id="已知问题md">已知问题</h3>

<a name="table1969972073016"></a>

<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>已知问题</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>问题描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>数据类型支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPU不支持float16类型的inf/nan数据输入输出。</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>数据Format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>出现4D以上的format时不能降维。</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>集合通信约束</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>要求一次训练任务中不同device上执行的图相同。</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>当前只支持1/2/4/8P粒度的分配。</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>只支持int8，int32，float16和float32数据类型。</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex功能支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>Apex当前版本的实现方式主要为python实现，不支持APEX中的自定义优化CUDA Kernel。</p>
</td>
</tr>
</tbody>
</table>

<h3 id="兼容性md">兼容性</h3>

A800-9010：CentOS 7.6/Ubuntu 18.04, 2.04/BC-Linux 7.6/Debian 9.9/Debian 10/OpenEuler 20.03 LTS

A800-9000：CentOS 7.6/Ubuntu 18.04, 2.04/Euler 2.8, 2.10/Kylin v10/BC-Linux 7.6/OpenEuler 20.03 LTS/UOS 20 1020e

<h2 id="FrameworkPTAdapter-3-0-RC3md">FrameworkPTAdapter 3.0.RC3</h2>

<h3 id="用户须知md">用户须知</h3>

本框架基于Facebook主导的开源PyTorch进行修改，延续原生的PyTorch特性，使用NPU进行动态图训练；以算子粒度进行模型适配，代码重用性好，支持现有的网络只修改设备类型或数据类型，即可迁移到NPU上使用。

PyTorch1.8.1版本开始采用插件化适配方式，与原生PyTorch框架实现彻底解耦，功能、性能与PyTorch1.5.0基本保持一致，对后端算子适配提供较好开发体验。

<h3 id="新增特性md">新增特性</h3>

**表 1** PyTorch支持的版本特性列表

| 一级特性                   | 二级特性                         | 说明                                           |
| -------------------------- | -------------------------------- | ---------------------------------------------- |
| 适配NPU的PyTorch1.5.0特性  | 框架基础功能                     | 适配CANN算子IR变更。                           |
| 适配NPU的PyTorch1.8.1特性  | 精度对比工具                     | 支持NPU与CPU精度对比工具。                     |
| profiling                  | 支持自定义算子profiling。        |                                                |
| API满足度提升              | 新增部分API适配（详见API清单）。 |                                                |
| 适配NPU的PyTorch1.11.0特性 | 框架基础功能                     | 支持模型训练功能，适配算子API（详见API清单）。 |
| 混合精度                   | 支持apex混合精度训练。           |                                                |
| 分布式                     | 支持DDP分布式训练功能。          |                                                |
| profiling                  | 支持E2E profiling功能。          |                                                |

<h3 id="特性修改md">特性修改</h3>

不涉及

<h3 id="已修复问题md">已修复问题</h3>

不涉及

<h3 id="已知问题md">已知问题</h3>

| 已知问题                                      | 问题描述                                                     |
| --------------------------------------------- | ------------------------------------------------------------ |
| 数据类型支持                                  | NPU不支持float16类型的inf/nan数据输入输出。                  |
| 数据Format                                    | 出现4D以上的format时不能降维。                               |
| 集合通信约束                                  | 要求一次训练任务中不同device上执行的图相同。                 |
| 当前只支持1/2/4/8P粒度的分配。                |                                                              |
| 只支持int8，int32，float16和float32数据类型。 |                                                              |
| Apex功能支持                                  | Apex当前版本的实现方式主要为python实现，不支持APEX中的自定义优化CUDA Kernel。 |

<h3 id="兼容性md">兼容性</h3>

A800-9010：CentOS 7.6/Ubuntu 18.04, 2.04/BC-Linux 7.6/Debian 9.9/Debian 10/OpenEuler 20.03 LTS

A800-9000：CentOS 7.6/Ubuntu 18.04, 2.04/Euler 2.8, 2.10/Kylin v10/BC-Linux 7.6/OpenEuler 20.03 LTS/UOS 20 1020e

<h2 id="FrameworkPTAdapter-3-0-RC2md">FrameworkPTAdapter 3.0.RC2</h2>


<h3 id="用户须知md">用户须知</h3>

本框架基于Facebook主导的开源PyTorch1.5.0和1.8.1版本进行修改，延续原生的PyTorch特性，使用NPU进行动态图训练；以算子粒度进行模型适配，代码重用性好，支持现有的网络只修改设备类型或数据类型，即可迁移到NPU上使用。

PyTorch1.8.1版本采用插件化适配方式，与原生PyTorch框架实现彻底解耦，当前功能、性能与PyTorch1.5.0基本保持一致，对后端算子适配提供较好开发体验。
支持AOE调优工具。

<h3 id="新增特性md">新增特性</h3>

**表 1** PyTorch支持的版本特性列表

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>

<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.459999999999999%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>一级特性</p>
</th>
<th class="cellrowborder" valign="top" width="26.27%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>二级特性</p>
</th>
<th class="cellrowborder" valign="top" width="63.27%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>说明</p>
</th>
</tr>
</thead>
<tbody>
<tr id="row7979351559"><td class="cellrowborder" rowspan="2" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>适配NPU的PyTorch1.5.0特性</p>
</td>
</tr>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a> 框架基础功能。</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a> 少量BUG修复。</p>
</td>
</tr>
<tr id="row7979351559"><td class="cellrowborder" rowspan="3" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>适配NPU的PyTorch1.8.1特性</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>插件化解耦</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>torch_npu插件与原生PyTorch框架实现彻底解耦。</p>
</td>
<tr>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>性能优化</p>
</td> 
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>优化单算子下发以及集合通讯功能</p>
</td>
</tr>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>API满足度提升。</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>新增部分API适配（详情见API清单）。</p>
</td>
</tr>
</tbody>
</table>






<h3 id="特性修改md">特性修改</h3>

不涉及

<h3 id="已修复问题md">已修复问题</h3>

不涉及

<h3 id="已知问题md">已知问题</h3>

<a name="table1969972073016"></a>

<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>已知问题</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>问题描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>数据类型支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPU不支持float16类型的inf/nan数据输入输出。</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>数据Format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>出现4D以上的format时不能降维。</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>集合通信约束</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>要求一次训练任务中不同device上执行的图相同。</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>当前只支持1/2/4/8P粒度的分配。</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>只支持int8，int32，float16和float32数据类型。</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex功能支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>Apex当前版本的实现方式主要为python实现，不支持APEX中的自定义优化CUDA Kernel。</p>
</td>
</tr>
</tbody>
</table>




<h3 id="兼容性md">兼容性</h3>

A800-9010：CentOS 7.6/Ubuntu 18.04, 2.04/BC-Linux 7.6/Debian 9.9/Debian 10/OpenEuler 20.03 LTS

A800-9000：CentOS 7.6/Ubuntu 18.04, 2.04/Euler 2.8, 2.10/Kylin v10/BC-Linux 7.6/OpenEuler 20.03 LTS/UOS 20 1020e

<h2 id="FrameworkPTAdapter-3-0-RC1md">FrameworkPTAdapter 3.0.RC1</h2>


<h3 id="用户须知md">用户须知</h3>

本框架基于Facebook主导的开源PyTorch1.5.0和1.8.1版本进行修改，延续原生的PyTorch特性，使用NPU进行动态图训练；以算子粒度进行模型适配，代码重用性好，支持现有的网络只修改设备类型或数据类型，即可迁移到NPU上使用。

PyTorch1.8.1版本采用插件化适配方式，延续PyTorch1.5.0特性，功能基本保持一致，对后端算子适配提供较好开发体验。
支持AOE调优工具。

<h3 id="新增特性md">新增特性</h3>

**表 1** PyTorch支持的版本特性列表

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>

<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.459999999999999%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>一级特性</p>
</th>
<th class="cellrowborder" valign="top" width="26.27%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>二级特性</p>
</th>
<th class="cellrowborder" valign="top" width="63.27%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>说明</p>
</th>
</tr>
</thead>
<tbody>
<tr id="row7979351559"><td class="cellrowborder" rowspan="2" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>适配NPU的PyTorch1.5.0特性</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>支持AOE调优工具</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>支持AOE自动调优工具，提升模型性能。</p>
</td>
</tr>
<tr id="row13971435754"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p79831331353"><a name="p79831331353"></a><a name="p79831331353"></a>框架基础功能</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a> 新增适配算子开发（详见API清单）。</p>
</td>
</tr>
<tr id="row7979351559"><td class="cellrowborder" rowspan="3" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>适配NPU的PyTorch1.8.1特性</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>插件化解耦</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>将NPU适配代码解耦迁移至torch_npu插件，实现昇腾适配代码与PyTorch原生代码解耦。</p>
</td>
<tr>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>支持AOE调优工具</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>支持AOE自动调优工具，提升模型性能。</p>
</td>
</tr>
<tr id="row109719353511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18983183315515"><a name="p18983183315515"></a><a name="p18983183315515"></a>框架API满足度提升</p>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p836221112812"><a name="p836221112812"></a><a name="p836221112812"></a>支持昇腾PyTorch1.5.0版本已适配API（详情见API清单）。</p>
</td>
</tr>
</tbody>
</table>





<h3 id="特性修改md">特性修改</h3>

不涉及

<h3 id="已修复问题md">已修复问题</h3>

不涉及

<h3 id="已知问题md">已知问题</h3>

<a name="table1969972073016"></a>

<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>已知问题</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>问题描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>数据类型支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPU不支持float16类型的inf/nan数据输入输出。</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>数据Format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>出现4D以上的format时不能降维。</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>集合通信约束</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>要求一次训练任务中不同device上执行的图相同。</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>当前只支持1/2/4/8P粒度的分配。</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>只支持int8，int32，float16和float32数据类型。</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex功能支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>Apex当前版本的实现方式主要为python实现，不支持APEX中的自定义优化CUDA Kernel。</p>
</td>
</tr>
</tbody>
</table>



<h3 id="兼容性md">兼容性</h3>

A800-9010：CentOS 7.6/Ubuntu 18.04, 2.04/BC-Linux 7.6/Debian 9.9/Debian 10/OpenEuler 20.03 LTS

A800-9000：CentOS 7.6/Ubuntu 18.04, 2.04/Euler 2.8, 2.10/Kylin v10/BC-Linux 7.6/OpenEuler 20.03 LTS/UOS 20 1020e

<h2 id="FrameworkPTAdapter-2-0-4md">FrameworkPTAdapter 2.0.4</h2>


<h3 id="用户须知md">用户须知</h3>

本框架基于Facebook主导的开源PyTorch1.5.0和1.8.1版本进行修改，延续原生的PyTorch特性，使用NPU进行动态图训练；以算子粒度进行模型适配，代码重用性好，支持现有的网络只修改设备类型或数据类型，即可迁移到NPU上使用。

PyTorch1.8.1版本延续PyTorch1.5.0特性，功能基本保持一致，对后端算子适配提供较好开发体验。当期1.8.1版本仅支持Resent50网络模型。

<h3 id="新增特性md">新增特性</h3>

**表 1** PyTorch支持的版本特性列表

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>

<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.459999999999999%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>一级特性</p>
</th>
<th class="cellrowborder" valign="top" width="26.27%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>二级特性</p>
</th>
<th class="cellrowborder" valign="top" width="63.27%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row7979351559"><td class="cellrowborder" rowspan="3" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>适配NPU的PyTorch1.5.0特性</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>精度对比工具完善</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>支持IR与对应TBE算子映射，支持在GPU侧加载NPU侧dump数据对比。</p>
</td>
</tr>
<tr id="row109719353511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18983183315515"><a name="p18983183315515"></a><a name="p18983183315515"></a>支持E2E prof工具</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p836221112812"><a name="p836221112812"></a><a name="p836221112812"></a>将PyTorch原生profiling工具和cann prof工具获取到的性能数据统一集成，实现端到端的模型和算子性能分析。</p>
</td>
</tr>
<tr id="row13971435754"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p79831331353"><a name="p79831331353"></a><a name="p79831331353"></a>框架基础功能</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a> 新增适配算子开发（详见算子清单）。</p>
</td>
</tr>
    <tr id="row7979351559"><td class="cellrowborder" rowspan="2" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>适配NPU的PyTorch1.8.1特性</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>支持AMP</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>支持PyTorch原生自动混合精度。</p>
</td>
</tr>
<tr id="row109719353511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18983183315515"><a name="p18983183315515"></a><a name="p18983183315515"></a>支持Profiling</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p836221112812"><a name="p836221112812"></a><a name="p836221112812"></a>支持PyTorch原生Profiling功能。</p>
</td>
</tr>
    <tr id="row7979351559"><td class="cellrowborder" rowspan="2" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>OS兼容性</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>OS兼容性</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>新增支持Ubuntu20.04 x86+arm，Euler 2.10 arm系统。</p>
</td>
<tr id="row13971435754"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p79831331353"><a name="p79831331353"></a><a name="p79831331353"></a>Python版本兼容性</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a>支持python3.9环境编译及使用（仅PyTorch1.8.1）。</p>
</td>
</tr>
</tbody>
</table>



<h3 id="特性修改md">特性修改</h3>

不涉及

<h3 id="已修复问题md">已修复问题</h3>

不涉及

<h3 id="已知问题md">已知问题</h3>

<a name="table1969972073016"></a>

<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>已知问题</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>问题描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>数据类型支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPU不支持float16类型的inf/nan数据输入输出。</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>数据Format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>出现4D以上的format时不能降维。</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>集合通信约束</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>要求一次训练任务中不同device上执行的图相同。</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>当前只支持1/2/4/8P粒度的分配。</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>只支持int8，int32，float16和float32数据类型。</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex功能支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>Apex当前版本的实现方式主要为python实现，不支持APEX中的自定义优化CUDA Kernel。</p>
</td>
</tr>
</tbody>
</table>


<h3 id="兼容性md">兼容性</h3>

A800-9010：CentOS 7.6/Ubuntu 18.04, 2.04/BC-Linux 7.6/Debian 9.9/Debian 10/OpenEuler 20.03 LTS

A800-9000：CentOS 7.6/Ubuntu 18.04, 2.04/Euler 2.8, 2.10/Kylin v10/BC-Linux 7.6/OpenEuler 20.03 LTS/UOS 20 1020e

<h2 id="FrameworkPTAdapter-2-0-3md">FrameworkPTAdapter 2.0.3</h2>


<h3 id="用户须知md">用户须知</h3>

本框架基于Facebook主导的开源PyTorch1.5.0进行修改，延续原生的PyTorch特性，使用NPU进行动态图训练；以算子粒度进行模型适配，代码重用性好，支持现有的网络只修改设备类型或数据类型，即可迁移到NPU上使用。

从此版本开始，PyTorch1.8.1版本提供支持，此版本延续PyTorch1.5.0特性，功能保持一致（profiling工具除外）。除此之外，对后端算子适配提供较好开发体验。当期1.8.1版本仅支持Resent50网络模型。

<h3 id="新增特性md">新增特性</h3>

**表 1** PyTorch支持的版本特性列表

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>
<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.459999999999999%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>一级特性</p>
</th>
<th class="cellrowborder" valign="top" width="26.27%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>二级特性</p>
</th>
<th class="cellrowborder" valign="top" width="63.27%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row7979351559"><td class="cellrowborder" rowspan="5" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>适配NPU的PyTorch特性</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>支持PyTorch1.8.1版本</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>增加PyTorch1.8.1版本, 当前仅支持resnet50网络，包括其DDP分布式训练场景。</p>
</td>
</tr>
<tr id="row109719353511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18983183315515"><a name="p18983183315515"></a><a name="p18983183315515"></a>支持python3.8</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p836221112812"><a name="p836221112812"></a><a name="p836221112812"></a>支持python3.8环境编译及使用。</p>
</td>
</tr>
<tr id="row13971435754"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p79831331353"><a name="p79831331353"></a><a name="p79831331353"></a>提供算子溢出检测工具</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a>支持框架IR级别算子溢出检测，在发生AICORE算子溢出时，提示溢出的IR信息。</p>
</td>
</tr>
<tr id="row185381431133610"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p953883153611"><a name="p953883153611"></a><a name="p953883153611"></a>OS兼容性</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p13538203153618"><a name="p13538203153618"></a><a name="p13538203153618"></a>新增支持UOS 20 1020e arm系统</p>
</td>
</tr>
<tr id="row91681125173610"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p2168172519364"><a name="p2168172519364"></a><a name="p2168172519364"></a>框架基础功能</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1416815259367"><a name="p1416815259367"></a><a name="p1416815259367"></a>新增适配算子开发（详见算子清单）</p>
</td>
</tr>
<tr id="row11970351050"><td class="cellrowborder" rowspan="26" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p298314333514"><a name="p298314333514"></a><a name="p298314333514"></a>训练模型</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p19984193312511"><a name="p19984193312511"></a><a name="p19984193312511"></a>CenterFace</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p7517351258"><a name="p7517351258"></a><a name="p7517351258"></a>-</p>
</td>
</tr>
<tr id="row9965351254"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1984033458"><a name="p1984033458"></a><a name="p1984033458"></a>PCBU</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p298411338513"><a name="p298411338513"></a><a name="p298411338513"></a>-</p>
</td>
</tr>
<tr id="row14966351959"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p198414331957"><a name="p198414331957"></a><a name="p198414331957"></a>Net++</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1798413331756"><a name="p1798413331756"></a><a name="p1798413331756"></a>-</p>
</td>
</tr>
<tr id="row20966351454"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p4984183317518"><a name="p4984183317518"></a><a name="p4984183317518"></a>FCN8S</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1698453311519"><a name="p1698453311519"></a><a name="p1698453311519"></a>-</p>
</td>
</tr>
<tr id="row19653517518"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1298483319512"><a name="p1298483319512"></a><a name="p1298483319512"></a>OSNetRetinaFace</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1798423312519"><a name="p1798423312519"></a><a name="p1798423312519"></a>-</p>
</td>
</tr>
<tr id="row39619351751"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1298414331513"><a name="p1298414331513"></a><a name="p1298414331513"></a>PSPnet</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10984633254"><a name="p10984633254"></a><a name="p10984633254"></a>-</p>
</td>
</tr>
<tr id="row496335654"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19984433351"><a name="p19984433351"></a><a name="p19984433351"></a>EDSR</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p898473314517"><a name="p898473314517"></a><a name="p898473314517"></a>-</p>
</td>
</tr>
<tr id="row17953357517"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p14984123312515"><a name="p14984123312515"></a><a name="p14984123312515"></a>Tsm</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p159841033855"><a name="p159841033855"></a><a name="p159841033855"></a>-</p>
</td>
</tr>
<tr id="row16951435551"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19984933457"><a name="p19984933457"></a><a name="p19984933457"></a>pnasnet5large</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p89846335512"><a name="p89846335512"></a><a name="p89846335512"></a>-</p>
</td>
</tr>
<tr id="row2095113520514"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1798416331952"><a name="p1798416331952"></a><a name="p1798416331952"></a>Gaitset</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p098417336516"><a name="p098417336516"></a><a name="p098417336516"></a>-</p>
</td>
</tr>
<tr id="row89516351511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p11985153312516"><a name="p11985153312516"></a><a name="p11985153312516"></a>fcn</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p169858331155"><a name="p169858331155"></a><a name="p169858331155"></a>-</p>
</td>
</tr>
<tr id="row5953353513"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18985103311514"><a name="p18985103311514"></a><a name="p18985103311514"></a>Albert</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2098513334519"><a name="p2098513334519"></a><a name="p2098513334519"></a>-</p>
</td>
</tr>
<tr id="row119533516513"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p598511336511"><a name="p598511336511"></a><a name="p598511336511"></a>AdvancedEast</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p119850331513"><a name="p119850331513"></a><a name="p119850331513"></a>-</p>
</td>
</tr>
<tr id="row39511356512"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p698553311512"><a name="p698553311512"></a><a name="p698553311512"></a>ReidStrongBaseline</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1198514331659"><a name="p1198514331659"></a><a name="p1198514331659"></a>-</p>
</td>
</tr>
<tr id="row129463518517"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p99851633158"><a name="p99851633158"></a><a name="p99851633158"></a>Fast-scnn</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1298514331359"><a name="p1298514331359"></a><a name="p1298514331359"></a>-</p>
</td>
</tr>
<tr id="row9942035757"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p109854331151"><a name="p109854331151"></a><a name="p109854331151"></a>RDN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1198583317511"><a name="p1198583317511"></a><a name="p1198583317511"></a>-</p>
</td>
</tr>
<tr id="row79415357517"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p6985833351"><a name="p6985833351"></a><a name="p6985833351"></a>SRFlow</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p29851033350"><a name="p29851033350"></a><a name="p29851033350"></a>-</p>
</td>
</tr>
<tr id="row12947357514"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p149859334518"><a name="p149859334518"></a><a name="p149859334518"></a>MGN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19851833850"><a name="p19851833850"></a><a name="p19851833850"></a>-</p>
</td>
</tr>
<tr id="row3941735855"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p898533316518"><a name="p898533316518"></a><a name="p898533316518"></a>Roberta</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p998553318518"><a name="p998553318518"></a><a name="p998553318518"></a>-</p>
</td>
</tr>
<tr id="row1093335856"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1985533451"><a name="p1985533451"></a><a name="p1985533451"></a>RegNetY</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p59851033355"><a name="p59851033355"></a><a name="p59851033355"></a>-</p>
</td>
</tr>
<tr id="row49373518516"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19986173311512"><a name="p19986173311512"></a><a name="p19986173311512"></a>VoVNet-39</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p798683310515"><a name="p798683310515"></a><a name="p798683310515"></a>-</p>
</td>
</tr>
<tr id="row5930351357"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p189862336515"><a name="p189862336515"></a><a name="p189862336515"></a>RegNetX</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1398618331651"><a name="p1398618331651"></a><a name="p1398618331651"></a>-</p>
</td>
</tr>
<tr id="row169312351655"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p149863337513"><a name="p149863337513"></a><a name="p149863337513"></a>RefineNet</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2098615331959"><a name="p2098615331959"></a><a name="p2098615331959"></a>-</p>
</td>
</tr>
<tr id="row11931235957"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1198616331657"><a name="p1198616331657"></a><a name="p1198616331657"></a>RefineDet</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1698611332518"><a name="p1698611332518"></a><a name="p1698611332518"></a>-</p>
</td>
</tr>
<tr id="row189215359511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p16986183319512"><a name="p16986183319512"></a><a name="p16986183319512"></a>AlignedReID</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1898715331055"><a name="p1898715331055"></a><a name="p1898715331055"></a>-</p>
</td>
</tr>
<tr id="row89213351858"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p119878330511"><a name="p119878330511"></a><a name="p119878330511"></a>FaceBoxes</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p129878331158"><a name="p129878331158"></a><a name="p129878331158"></a>-</p>
</td>
</tr>
</tbody>
</table>

<h3 id="特性修改md">特性修改</h3>

不涉及

<h3 id="已修复问题md">已修复问题</h3>

不涉及

<h3 id="已知问题md">已知问题</h3>

<a name="table1969972073016"></a>
<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>已知问题</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>问题描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>数据类型支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPU不支持float16类型的inf/nan数据输入输出。</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>数据Format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>出现4D以上的format时不能降维。</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>集合通信约束</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>要求一次训练任务中不同device上执行的图相同。</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>当前只支持1/2/4/8P粒度的分配。</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>只支持int8，int32，float16和float32数据类型。</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex功能支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>Apex当前版本的实现方式主要为python实现，不支持APEX中的自定义优化CUDA Kernel。</p>
</td>
</tr>
</tbody>
</table>

<h3 id="兼容性md">兼容性</h3>

A800-9010：CentOS 7.6/Ubuntu 18.04/BC-Linux 7.6/Debian 9.9/Debian 10/OpenEuler 20.03 LTS

A800-9000：CentOS 7.6/Euler 2.8/Kylin v10/BC-Linux 7.6/OpenEuler 20.03 LTS/UOS 20 1020e

<h2 id="FrameworkPTAdapter-2-0-2md">FrameworkPTAdapter 2.0.2</h2>


<h3 id="用户须知-0md">用户须知</h3>

本框架基于Facebook主导的开源PyTorch1.5.0进行修改，延续原生的PyTorch特性，使用NPU进行动态图训练；以算子粒度进行模型适配，代码重用性好，支持现有的网络只修改设备类型或数据类型，即可迁移到NPU上使用。

<h3 id="新增特性-1md">新增特性</h3>

**表 1** PyTorch支持的版本特性列表

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>
<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.489999999999998%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>一级特性</p>
</th>
<th class="cellrowborder" valign="top" width="26.3%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>二级特性</p>
</th>
<th class="cellrowborder" valign="top" width="63.21%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row171322953713"><td class="cellrowborder" rowspan="7" valign="top" width="10.489999999999998%" headers="mcps1.2.4.1.1 "><p id="p10237517181"><a name="p10237517181"></a><a name="p10237517181"></a>适配训练模型</p>
</td>
<td class="cellrowborder" valign="top" width="26.3%" headers="mcps1.2.4.1.2 "><p id="p6134294377"><a name="p6134294377"></a><a name="p6134294377"></a>YOLOV4</p>
</td>
<td class="cellrowborder" valign="top" width="63.21%" headers="mcps1.2.4.1.3 "><p id="p314533811397"><a name="p314533811397"></a><a name="p314533811397"></a>-</p>
</td>
</tr>
<tr id="row15990182233714"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p399019224374"><a name="p399019224374"></a><a name="p399019224374"></a>YOLOV3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1714553816396"><a name="p1714553816396"></a><a name="p1714553816396"></a>-</p>
</td>
</tr>
<tr id="row5301255373"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1530325123719"><a name="p1530325123719"></a><a name="p1530325123719"></a>DB</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p5146193819390"><a name="p5146193819390"></a><a name="p5146193819390"></a>-</p>
</td>
</tr>
<tr id="row274203413712"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p197423343379"><a name="p197423343379"></a><a name="p197423343379"></a>RFCN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p15828941153919"><a name="p15828941153919"></a><a name="p15828941153919"></a>-</p>
</td>
</tr>
<tr id="row78671918163714"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p188671318183713"><a name="p188671318183713"></a><a name="p188671318183713"></a>CRNN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10828241123917"><a name="p10828241123917"></a><a name="p10828241123917"></a>-</p>
</td>
</tr>
<tr id="row16912221171"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p6913132676"><a name="p6913132676"></a><a name="p6913132676"></a>Densenset161</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p209131221474"><a name="p209131221474"></a><a name="p209131221474"></a>-</p>
</td>
</tr>
<tr id="row1016073314719"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p3160153320717"><a name="p3160153320717"></a><a name="p3160153320717"></a>Densenset191</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1516012331173"><a name="p1516012331173"></a><a name="p1516012331173"></a>-</p>
</td>
</tr>
<tr id="row9627165910386"><td class="cellrowborder" rowspan="4" valign="top" width="10.489999999999998%" headers="mcps1.2.4.1.1 "><p id="p1561535993811"><a name="p1561535993811"></a><a name="p1561535993811"></a>适配NPU的PyTorch特性</p>
</td>
<td class="cellrowborder" valign="top" width="26.3%" headers="mcps1.2.4.1.2 "><p id="p136151659163819"><a name="p136151659163819"></a><a name="p136151659163819"></a>框架基础功能</p>
</td>
<td class="cellrowborder" valign="top" width="63.21%" headers="mcps1.2.4.1.3 "><p id="p1661535983813"><a name="p1661535983813"></a><a name="p1661535983813"></a>新增适配算子开发（详见算子清单）。</p>
</td>
</tr>
<tr id="row7627155917380"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1561525916383"><a name="p1561525916383"></a><a name="p1561525916383"></a>精度对比工具</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19616115953816"><a name="p19616115953816"></a><a name="p19616115953816"></a>新增精度对比工具，支持训练精度定界。</p>
</td>
</tr>
<tr id="row46269593383"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p0616559163819"><a name="p0616559163819"></a><a name="p0616559163819"></a>昇腾710芯片</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p186161459193813"><a name="p186161459193813"></a><a name="p186161459193813"></a>新增支持昇腾710芯片在线推理。</p>
</td>
</tr>
<tr id="row76261059153817"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p11616175953812"><a name="p11616175953812"></a><a name="p11616175953812"></a>OS兼容性</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p8935115971714"><a name="p8935115971714"></a><a name="p8935115971714"></a>新增支持ubuntu 18.04.5、OpenEuler 20.03 LTS系统</p>
</td>
</tr>
</tbody>
</table>

<h3 id="特性修改-2md">特性修改</h3>

不涉及

<h3 id="已修复问题-3md">已修复问题</h3>

不涉及

<h3 id="已知问题-4md">已知问题</h3>

<a name="table1969972073016"></a>
<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>已知问题</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>问题描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>数据类型支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPU不支持float16类型的inf/nan数据输入输出。</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>数据Format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>出现4D以上的format时不能降维。</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>集合通信约束</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>要求一次训练任务中不同device上执行的图相同。</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>当前只支持1/2/4/8P粒度的分配。</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>只支持int8，int32，float16和float32数据类型。</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex功能支持</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>Apex当前版本的实现方式主要为python实现，不支持APEX中的自定义优化CUDA Kernel。</p>
</td>
</tr>
</tbody>
</table>

<h3 id="兼容性-5md">兼容性</h3>

A800-9010：CentOS 7.6/Ubuntu 18.04/BC-Linux 7.6/Debian 9.9/Debian 10/OpenEuler 20.03 LTS

A800-9000：CentOS 7.6/Euler 2.8/Kylin v10/BC-Linux 7.6/OpenEuler 20.03 LTS

