# PyTorch版本说明书
-   [用户须知](#用户须知.md)
-   [新增特性](#新增特性.md)
-   [特性修改](#特性修改.md)
-   [已修复问题](#已修复问题.md)
-   [已知问题](#已知问题.md)
-   [兼容性](#兼容性.md)
<h2 id="用户须知.md">用户须知</h2>

本框架基于Facebook主导的开源PyTorch1.5.0进行修改，延续原生的PyTorch特性，使用NPU进行动态图训练；以算子粒度进行模型适配，代码重用性好，支持现有的网络只修改设备类型或数据类型，即可迁移到NPU上使用。

<h2 id="新增特性.md">新增特性</h2>

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

<h2 id="特性修改.md">特性修改</h2>

不涉及

<h2 id="已修复问题.md">已修复问题</h2>

不涉及

<h2 id="已知问题.md">已知问题</h2>

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

<h2 id="兼容性.md">兼容性</h2>

A800-9010：CentOS 7.6/Ubuntu 18.04/BC-Linux 7.6/Debian 9.9/Debian 10/OpenEuler 20.03 LTS

A800-9000：CentOS 7.6/Euler 2.8/Kylin v10/BC-Linux 7.6/OpenEuler 20.03 LTS

